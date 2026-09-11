#!/usr/bin/env python
"""
Split a dynamic-feature run into host time and device time.

Exercises only DensityFlow and MarchingObservers with the ``mass`` weight, so the
numbers reflect Nearl's own kernels rather than RDKit/OpenBabel/ChargeFW2.

Reports host-side phase timings only (cache/query/run/dump/device-call). For
device-side (nsys) profiling, run this script under ``nsys profile`` yourself.

Usage:
    python profiling_host_device.py [--dims 32] [--window 10]
"""

import argparse
import os
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

# Deliberately no sys.path surgery here: which `nearl` wins - this checkout,
# an editable install of it, or an unrelated site-packages copy - depends on
# how it was installed, and guessing wrong silently shadows a perfectly good
# install with a stale or incomplete one. main() prints nearl.__file__ up
# front instead, so a mismatch is visible rather than silently "fixed" wrong.
import nearl
import nearl.commands as commands
import nearl.features
import nearl.featurizer
import nearl.io


class PhaseTimer:
    """Accumulate wall time per labelled phase by wrapping bound methods.

    Featurizer.run on this version is a single sequential loop (cache, query,
    run, dump, one trajectory/frame-slice at a time on one thread) - no
    producer/consumer threads, so phases never overlap and a plain label ->
    seconds/calls table, summed against the wall clock, is exactly right.
    """

    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    def wrap(self, obj, name, label):
        original = getattr(obj, name)

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.seconds[label] += time.perf_counter() - t0
                self.calls[label] += 1

        setattr(obj, name, timed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", type=int, default=32, help="grid dimension per axis")
    p.add_argument("--window", type=int, default=10, help="frames per slice")
    p.add_argument("--datadir", default="/tmp/nearl_test", help="example-data folder")
    p.add_argument("--outfile", default="/tmp/prof_dynamic.h5", help="HDF5 output")
    p.add_argument(
        "--cold-start",
        action="store_true",
        help="skip the warm-up, so CUDA context creation is timed too",
    )
    p.add_argument(
        "--trajlist",
        default=None,
        help="file listing one trajectory per line as '<trajectory> <topology>', "
        "used instead of the bundled example data",
    )
    p.add_argument(
        "--multi",
        type=int,
        default=1,
        help="number of distinct trajectories to process (default 1). "
        "Extra trajectories are rotated copies of the example trajectory, "
        "so the CPU/GPU overlap benefit can be measured.",
    )
    return p.parse_args()


def build_multi_trajset(datadir, n):
    """Return a list of ``(nc, pdb)`` tuples with ``n`` distinct trajectories.

    The first entry is the stock example trajectory; each additional entry is a
    copy rotated by a different angle around the z axis. Rotating keeps the
    topology identical (so the per-topology cache still hits) while making the
    coordinates distinct, which is exactly the multi-trajectory workload the
    CPU/GPU pipeline is meant to overlap.
    """
    import numpy as np
    import pytraj as pt

    example = nearl.get_example_data(datadir)
    nc, pdb = example["MINI_TRAJSET"][0]
    base = pt.load(nc, pdb)
    trajs = [(nc, pdb)]
    if n <= 1:
        return trajs

    outdir = os.path.join(datadir, "example_data", "example_traj")
    os.makedirs(outdir, exist_ok=True)
    for i in range(1, n):
        ang = np.deg2rad(360.0 * i / n)
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        out = os.path.join(outdir, f"example_rot_{i}.nc")
        if not os.path.exists(out):
            newtraj = pt.Trajectory()
            newtraj.top = base.top
            for f in range(base.n_frames):
                fr = base[f].copy()
                fr.xyz = fr.xyz @ rot.T
                newtraj.append(fr)
            pt.write_traj(out, newtraj, overwrite=True)
        trajs.append((out, pdb))
    return trajs


def read_trajlist(path):
    """One trajectory per line, whitespace-separated: ``<trajectory> <topology>``."""
    with open(path) as handle:
        return [line.split() for line in handle if line.strip()]


def build_featurizer(args, timer):
    if args.trajlist:
        trajs = read_trajlist(args.trajlist)
    else:
        trajs = build_multi_trajset(args.datadir, args.multi)
    args.n_trajectories = len(trajs)
    loader = nearl.io.TrajectoryLoader(trajs)
    # run() loads each trajectory via __getitem__ at the top of its loop
    # iteration, so time that call to expose the trajectory-load cost.
    timer.wrap(nearl.io.TrajectoryLoader, "__getitem__", "trajectory load")

    featurizer = nearl.featurizer.Featurizer(
        {
            "dimensions": args.dims,
            "lengths": 16,
            "time_window": args.window,
            "sigma": 1.5,
            "cutoff": 3.5,
            "outfile": args.outfile,
        }
    )
    # Only the two dynamic features, weighted by mass: no external toolkit involved.
    featurizer.register_features(
        [
            nearl.features.DensityFlow(
                selection="!:T3P",
                agg="standard_deviation",
                weight_type="mass",
                outkey="df",
            ),
            nearl.features.MarchingObservers(
                selection="!:T3P",
                obs="density",
                agg="mean",
                weight_type="mass",
                outkey="obs",
            ),
        ]
    )
    # Featurizer.run() drives every feature's cache/query/run/dump itself, on
    # one thread, so wrapping each instance directly is enough - no cloning to
    # worry about.
    for feat in featurizer.FEATURESPACE:
        timer.wrap(feat, "run", "feature.run")
        timer.wrap(feat, "query", "query + crop")
        timer.wrap(feat, "dump", "HDF5 dump")
        timer.wrap(feat, "cache", "cache")
    featurizer.register_trajloader(loader)
    featurizer.register_focus([":LIG"], "mask")
    return featurizer


HOST_ROWS = ("cache", "trajectory load", "query + crop", "feature.run", "HDF5 dump")
DEVICE_ROW = "device call"
DEVICE_LABEL = "(device call inside)"
WALL_LABEL = "TOTAL run()"


def warm_up():
    """Create the CUDA context before timing; the first call costs ~0.15 s."""
    import numpy as np

    commands.frame_voxelize(
        np.zeros((8, 3), dtype=np.float32),
        np.ones(8, dtype=np.float32),
        np.array([8, 8, 8], dtype=np.int32),
        0.5,
        5,
        2,
    )


def run_workload(args):
    timer = PhaseTimer()
    for fn in ("density_flow", "marching_observer"):
        timer.wrap(commands, fn, DEVICE_ROW)
    featurizer = build_featurizer(args, timer)

    if not args.cold_start:
        warm_up()
        timer.seconds.clear()
        timer.calls.clear()

    if os.path.exists(args.outfile):
        os.remove(args.outfile)
    t0 = time.perf_counter()
    featurizer.run()
    total = time.perf_counter() - t0

    device_call = timer.seconds[DEVICE_ROW]
    print_report(args, timer, total, device_call)
    return total, device_call


def print_report(args, timer, total, device_call):
    width = 62
    print("\n" + "=" * width)
    print(
        f"dims={args.dims}  time_window={args.window}  "
        f"trajectories={getattr(args, 'n_trajectories', args.multi)}  "
        f"features=DensityFlow,MarchingObservers  weight=mass"
    )
    print("=" * width)
    print(f"{'HOST PHASE':<26}{'seconds':>10}{'calls':>8}{'% wall':>10}")
    print("-" * width)
    accounted = 0.0
    for row in HOST_ROWS:
        print(
            f"{row:<26}{timer.seconds[row]:>10.3f}{timer.calls[row]:>8}"
            f"{100 * timer.seconds[row] / total:>9.1f}%"
        )
        accounted += timer.seconds[row]
    print(
        f"{'  ' + DEVICE_LABEL:<26}{device_call:>10.3f}"
        f"{timer.calls[DEVICE_ROW]:>8}{100 * device_call / total:>9.1f}%"
    )
    print(
        f"{'unattributed':<26}{total - accounted:>10.3f}{'':>8}"
        f"{100 * (total - accounted) / total:>9.1f}%"
    )
    print("-" * width)
    print(f"{WALL_LABEL:<26}{total:>10.3f}")
    print("=" * width)


def main():
    args = parse_args()
    print(f"nearl from : {nearl.__file__}")
    run_workload(args)


if __name__ == "__main__":
    main()
