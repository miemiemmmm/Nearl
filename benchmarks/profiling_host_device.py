#!/usr/bin/env python
"""
Split a dynamic-feature run into host time and device time.

Exercises only DensityFlow and MarchingObservers with the ``mass`` weight, so the
numbers reflect Nearl's own kernels rather than RDKit/OpenBabel/ChargeFW2.

With Nsight Systems present the script re-executes itself under ``nsys`` and adds
the kernel/memory/API tables plus a derived host-vs-device summary. nvprof is not
an option: it is unsupported on compute capability 8.0+, which covers both sm_86
and the GH200's sm_90.

Usage:
    python profiling_host_device.py [--dims 32] [--window 10] [--no-nsys]
"""

import argparse
import csv
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import nearl
import nearl.commands as commands
import nearl.features
import nearl.featurizer
import nearl.io

CHILD_ENV = "NEARL_PROFILING_CHILD"


class PhaseTimer:
    """Accumulate wall time per (phase, thread) by wrapping bound methods.

    Featurizer.run drives three threads at once -- the CPU producer, the GPU
    consumer and the HDF5 writer -- so the phases overlap in wall-clock time and
    do not partition the run. Summing them against the wall clock is what made
    the old report print a large negative "unattributed". Keyed by thread, the
    phases within each thread *are* sequential, which is what makes a residual
    meaningful.

    The lock is load-bearing now that several threads report: ``d[k] += v`` is
    three bytecodes, so concurrent updates would silently drop samples.
    """

    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self._lock = threading.Lock()

    def wrap(self, obj, name, label):
        original = getattr(obj, name)

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                key = (label, threading.current_thread().name)
                with self._lock:
                    self.seconds[key] += dt
                    self.calls[key] += 1

        setattr(obj, name, timed)

    def total(self, label):
        return sum(v for (lbl, _), v in self.seconds.items() if lbl == label)

    def count(self, label):
        return sum(v for (lbl, _), v in self.calls.items() if lbl == label)

    def by_thread(self):
        """{thread: {label: (seconds, calls)}}"""
        out = defaultdict(dict)
        for (label, thread), sec in self.seconds.items():
            out[thread][label] = (sec, self.calls[(label, thread)])
        return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", type=int, default=32, help="grid dimension per axis")
    p.add_argument("--window", type=int, default=10, help="frames per slice")
    p.add_argument("--datadir", default="/tmp/nearl_test", help="example-data folder")
    p.add_argument("--outfile", default="/tmp/prof_dynamic.h5", help="HDF5 output")
    p.add_argument(
        "--no-nsys", action="store_true", help="skip the Nsight Systems pass"
    )
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
    # run() loads trajectories on the CPU producer thread via __getitem__,
    # so time that call to expose the trajectory-load cost.
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
    for feat in featurizer.FEATURESPACE:
        timer.wrap(feat, "run", "feature.run")
        timer.wrap(feat, "query", "query + crop")
        timer.wrap(feat, "dump", "HDF5 dump")
        timer.wrap(feat, "cache", "cache")
    featurizer.register_trajloader(loader)
    featurizer.register_focus([":LIG"], "mask")
    return featurizer


# Real work. feature.run is the only one that contains the device call, so the
# device row is reported nested under it and never added alongside it.
WORK_ROWS = (
    "trajectory load",
    "cache",
    "query + crop",
    "feature.run",
    "HDF5 dump",
)
DEVICE_ROW = "device call"
# Waiting, not work: these name the idle time on each thread.
BLOCKED_ROWS = (
    "blocked: buffer.get",
    "blocked: buffer.put",
    "blocked: writer.submit",
)
WALL_LABEL = "wall clock of run()"
DEVICE_LABEL = "inside CUDA calls (device)"


def instrument_queues(timer):
    """Time the queue waits, so each thread's idle time has a name.

    Only the pipelined featurizer has these. Against the serial one the import
    fails and the report simply carries no blocked rows.
    """
    try:
        from nearl import pipeline
    except ImportError:
        return
    timer.wrap(pipeline.PrefetchBuffer, "get", "blocked: buffer.get")
    timer.wrap(pipeline.PrefetchBuffer, "put", "blocked: buffer.put")
    timer.wrap(pipeline.AsyncWriter, "submit", "blocked: writer.submit")


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
    instrument_queues(timer)

    if not args.cold_start:
        warm_up()
        timer.seconds.clear()
        timer.calls.clear()

    if os.path.exists(args.outfile):
        os.remove(args.outfile)
    t0 = time.perf_counter()
    featurizer.run()
    total = time.perf_counter() - t0

    device_call = timer.total(DEVICE_ROW)
    print_report(args, timer, total, device_call)
    return total, device_call


def print_report(args, timer, total, device_call):
    """Report each phase against the thread it ran on.

    Phases on different threads overlap, so they cannot be laid end to end
    against the wall clock. Within one thread they are sequential, so there the
    leftover is real idle time and is reported as such.
    """
    width = 66
    per_thread = timer.by_thread()
    print("\n" + "=" * width)
    print(
        f"dims={args.dims}  time_window={args.window}  "
        f"trajectories={getattr(args, 'n_trajectories', args.multi)}  "
        f"features=DensityFlow,MarchingObservers  weight=mass"
    )
    print("=" * width)
    print(f"{'THREAD / PHASE':<30}{'seconds':>10}{'calls':>8}{'% wall':>10}")
    print("-" * width)

    order = list(WORK_ROWS) + list(BLOCKED_ROWS)

    # The thread that launched the kernels first, then the rest by busy time.
    def rank(item):
        _thread, rows = item
        return (DEVICE_ROW not in rows, -sum(sec for sec, _ in rows.values()))

    for thread, rows in sorted(per_thread.items(), key=rank):
        print(f"{thread}")
        busy = 0.0
        for label in order + [k for k in rows if k not in order and k != DEVICE_ROW]:
            if label not in rows:
                continue
            sec, calls = rows[label]
            busy += sec
            print(f"{'  ' + label:<30}{sec:>10.3f}{calls:>8}{100 * sec / total:>9.1f}%")
            if label == "feature.run" and DEVICE_ROW in rows:
                sec_d, calls_d = rows[DEVICE_ROW]
                print(
                    f"{'    of which ' + DEVICE_ROW:<30}{sec_d:>10.3f}{calls_d:>8}"
                    f"{100 * sec_d / total:>9.1f}%"
                )
        idle = total - busy
        print(
            f"{'  idle / unaccounted':<30}{idle:>10.3f}{'':>8}"
            f"{100 * idle / total:>9.1f}%"
        )

    work = sum(timer.total(row) for row in WORK_ROWS)
    host = work - device_call
    print("-" * width)
    print(f"{WALL_LABEL:<30}{total:>10.3f}{'':>8}{100.0:>9.1f}%")
    print(
        f"{DEVICE_LABEL:<30}{device_call:>10.3f}{'':>8}"
        f"{100 * device_call / total:>9.1f}%"
    )
    print(
        f"{'host work, summed over threads':<30}{host:>10.3f}{'':>8}"
        f"{100 * host / total:>9.1f}%"
    )
    print(
        f"{'overlap achieved':<30}{work - total:>10.3f}{'':>8}"
        f"{100 * (work - total) / total:>9.1f}%"
    )
    print("=" * width)
    print("Host work is summed across threads, so it can exceed the wall clock;")
    print("the excess is the overlap the pipeline won. 'of which device call' is")
    print("the wall time inside commands.density_flow / marching_observer -- the")
    print("synchronous transfers, kernels and CUDA API overhead -- and is a part")
    print("of feature.run, not a phase beside it.")


def csv_total_ns(path):
    if not pathlib.Path(path).is_file():
        return None
    with open(path) as fh:
        return sum(float(row["Total Time (ns)"]) for row in csv.DictReader(fh))


def nsys_pass(args):
    """Re-run this script under nsys and report the device side."""
    with tempfile.TemporaryDirectory() as workdir:
        report = os.path.join(workdir, "nearl_hostdev")
        env = dict(os.environ, **{CHILD_ENV: "1"})
        child = subprocess.run(
            [
                "nsys",
                "profile",
                "-t",
                "cuda",
                "-o",
                report,
                "--force-overwrite",
                "true",
                sys.executable,
                os.path.abspath(__file__),
                *sys.argv[1:],
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        host_out = child.stdout
        # nsys writes a carriage-return progress bar onto the child's stdout.
        for line in host_out.replace("\r", "\n").splitlines():
            if line.startswith(("[1/1]", "Collecting data", "Generating", "Generated")):
                continue
            if line.startswith("\t") or not line.strip():
                continue
            print(line)
        if child.returncode != 0:
            print(child.stderr[-2000:], file=sys.stderr)
            return

        rep = f"{report}.nsys-rep"
        tables = subprocess.run(
            [
                "nsys",
                "stats",
                "--report",
                "cuda_gpu_kern_sum",
                "--report",
                "cuda_gpu_mem_time_sum",
                "--report",
                "cuda_api_sum",
                "--format",
                "table",
                rep,
            ],
            capture_output=True,
            text=True,
        ).stdout
        print("\n================= DEVICE (Nsight Systems) =================")
        for line in tables.splitlines():
            if (
                line.startswith(("Processing", "NOTICE", "Generating SQLite"))
                or not line.strip()
            ):
                continue
            if line.lstrip().startswith(("It is assumed", "Consider using")):
                continue
            print(line)

        # --force-export: the table pass above already wrote a .sqlite, and nsys
        # refuses to reuse one that is older than the .nsys-rep.
        exported = subprocess.run(
            [
                "nsys",
                "stats",
                "--report",
                "cuda_gpu_kern_sum",
                "--report",
                "cuda_gpu_mem_time_sum",
                "--format",
                "csv",
                "--force-export=true",
                "--output",
                report,
                rep,
            ],
            capture_output=True,
            text=True,
        )
        kernel_ns = csv_total_ns(f"{report}_cuda_gpu_kern_sum.csv")
        memory_ns = csv_total_ns(f"{report}_cuda_gpu_mem_time_sum.csv")
        if kernel_ns is None or memory_ns is None:
            print(f"\nnsys csv export rc={exported.returncode}", file=sys.stderr)
            print(exported.stdout[-600:], file=sys.stderr)
            print(exported.stderr[-600:], file=sys.stderr)
            print(f"files: {sorted(os.listdir(workdir))}", file=sys.stderr)
        summarize(host_out, kernel_ns, memory_ns)


def _labelled_value(text, label):
    """First number on the line starting with `label`, or None."""
    for line in text.splitlines():
        if not line.strip().startswith(label):
            continue
        for token in line.replace("%", " ").split():
            try:
                return float(token)
            except ValueError:
                continue
    return None


def summarize(host_out, kernel_ns, memory_ns):
    if kernel_ns is None or memory_ns is None:
        print("\n(could not read nsys CSV totals; skipping the derived summary)")
        return
    total = _labelled_value(host_out, WALL_LABEL)
    device_call = _labelled_value(host_out, DEVICE_LABEL)
    if total is None or device_call is None:
        return

    kernel, memory = kernel_ns / 1e9, memory_ns / 1e9
    overhead = device_call - kernel - memory
    print("\n" + "=" * 62)
    print("HOST vs DEVICE")
    print("-" * 62)
    for label, value in (
        ("wall clock", total),
        ("time inside device calls", device_call),
        ("  GPU kernel execution", kernel),
        ("  GPU memory operations", memory),
        ("  host-side CUDA overhead", overhead),
    ):
        print(f"{label:<32}{value:>9.3f} s{100 * value / total:>8.1f}% of wall")
    print("=" * 62)
    print("Host-side CUDA overhead is cudaMalloc/cudaFree, the pybind11 and numpy")
    print("marshalling, and blocking in cudaDeviceSynchronize. A warm-up call")
    print("keeps one-off context creation out of it; --cold-start includes it.")


def main():
    args = parse_args()
    if os.environ.get(CHILD_ENV):
        run_workload(args)
        return
    print(f"nearl from : {nearl.__file__}")
    use_nsys = (
        not args.no_nsys
        and subprocess.run(["which", "nsys"], capture_output=True).returncode == 0
    )
    if use_nsys:
        nsys_pass(args)
    else:
        if not args.no_nsys:
            print("nsys not found; reporting host timings only.", file=sys.stderr)
        run_workload(args)


if __name__ == "__main__":
    main()
