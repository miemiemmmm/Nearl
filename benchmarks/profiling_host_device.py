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
    """Accumulate wall time per labelled phase by wrapping bound methods."""

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
        "--no-nsys", action="store_true", help="skip the Nsight Systems pass"
    )
    p.add_argument(
        "--cold-start",
        action="store_true",
        help="skip the warm-up, so CUDA context creation is timed too",
    )
    return p.parse_args()


def build_featurizer(args, timer):
    loader = nearl.io.TrajectoryLoader(
        nearl.get_example_data(args.datadir)["MINI_TRAJSET"]
    )
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


HOST_ROWS = ("cache", "trajectory load", "query + crop", "feature.run", "HDF5 dump")


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
        timer.wrap(commands, fn, "device call")
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

    print("\n" + "=" * 62)
    print(
        f"dims={args.dims}  time_window={args.window}  "
        f"features=DensityFlow,MarchingObservers  weight=mass"
    )
    print("=" * 62)
    print(f"{'HOST PHASE':<26}{'seconds':>10}{'calls':>8}{'% wall':>10}")
    print("-" * 62)
    accounted = 0.0
    for row in HOST_ROWS:
        print(
            f"{row:<26}{timer.seconds[row]:>10.3f}{timer.calls[row]:>8}"
            f"{100 * timer.seconds[row] / total:>9.1f}%"
        )
        accounted += timer.seconds[row]
    device_call = timer.seconds["device call"]
    print(
        f"{'  (device call inside)':<26}{device_call:>10.3f}"
        f"{timer.calls['device call']:>8}{100 * device_call / total:>9.1f}%"
    )
    print(
        f"{'unattributed':<26}{total - accounted:>10.3f}{'':>8}"
        f"{100 * (total - accounted) / total:>9.1f}%"
    )
    print("-" * 62)
    print(f"{'TOTAL run()':<26}{total:>10.3f}")
    print("=" * 62)
    return total, device_call


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


def summarize(host_out, kernel_ns, memory_ns):
    if kernel_ns is None or memory_ns is None:
        print("\n(could not read nsys CSV totals; skipping the derived summary)")
        return
    total = device_call = None
    for line in host_out.splitlines():
        if line.startswith("TOTAL run()"):
            total = float(line.split()[-1])
        elif "(device call inside)" in line:
            device_call = float(line.split()[-3])
    if total is None or device_call is None:
        return

    kernel, memory = kernel_ns / 1e9, memory_ns / 1e9
    overhead = device_call - kernel - memory
    print("\n" + "=" * 62)
    print("HOST vs DEVICE")
    print("-" * 62)
    for label, value in (
        ("wall clock of run()", total),
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
