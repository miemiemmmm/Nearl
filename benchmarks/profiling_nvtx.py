#!/usr/bin/env python
"""
Profile the full featurization pipeline with the NVTX ranges from
:mod:`nearl.profiling`.

Unlike ``profiling_host_device.py``, which times a handful of hand-wrapped
methods, this runs the whole ``Featurizer.run()`` under Nsight Systems and lets
the NVTX ranges attribute the time. It answers "where does the wall clock go",
including the stages that never touch the GPU: trajectory loading, ``cache``
(RDKit/OpenBabel/ChargeFW2 when the weight type needs them), the padding and
cropping inside ``query``, and the gzip'ed HDF5 ``dump``.

The script re-executes itself under ``nsys`` and then prints:

- ``nvtx_pushpop_sum``  -- total/average time per NVTX range, i.e. the host-side
  breakdown. Note that these ranges nest, so ``Featurizer.run`` contains all the
  others; read the leaves (``query.*``, ``dispatch.*``, ``collect.*``,
  ``append_hdf_data``) rather than summing the column.
- ``nvtx_gpu_proj_sum`` -- for each range, how much GPU work it projects onto.
  A range with a large CPU time and near-zero projected GPU time is a host-side
  bottleneck.
- ``cuda_gpu_kern_sum`` -- kernel time, for the device side of the comparison.

nsys CPU sampling is disabled by default (``--sample=none``): it needs
``kernel.perf_event_paranoid <= 2`` and Ubuntu ships 4, so on such a host nsys
warns and silently collects nothing. None of the three reports use it. Pass
``--cpu-sampling`` after lowering the sysctl if you want native backtraces on top.

Usage:
    python benchmarks/profiling_nvtx.py [--dims 32] [--window 10]
    python benchmarks/profiling_nvtx.py --weight partial_charge   # heavy cache()
    python benchmarks/profiling_nvtx.py --no-nsys                 # ranges only, no report
"""

import argparse
import os
import pathlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import nearl
import nearl.features
import nearl.featurizer
import nearl.io

CHILD_ENV = "NEARL_NVTX_PROFILING_CHILD"

REPORTS = ("nvtx_pushpop_sum", "nvtx_gpu_proj_sum", "cuda_gpu_kern_sum")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", type=int, default=32, help="grid dimension per axis")
    p.add_argument("--window", type=int, default=10, help="frames per slice")
    p.add_argument("--datadir", default="/tmp/nearl_test", help="example-data folder")
    p.add_argument("--outfile", default="/tmp/prof_nvtx.h5", help="HDF5 output")
    p.add_argument(
        "--weight",
        default="mass",
        help="weight_type for the dynamic features; 'mass' is cheap to cache, "
        "'partial_charge' pulls in the external toolkits",
    )
    p.add_argument(
        "--compress",
        type=int,
        default=0,
        help="gzip level for the HDF5 dump (0 disables compression)",
    )
    p.add_argument(
        "--no-nsys", action="store_true", help="skip the Nsight Systems pass"
    )
    p.add_argument(
        "--cpu-sampling",
        action="store_true",
        help="also collect nsys CPU sampling. Off by default because it needs "
        "kernel.perf_event_paranoid <= 2 (Ubuntu ships 4), and without it nsys "
        "only prints a warning and drops the samples. NVTX and CUDA tracing do "
        "not depend on it, so the reports below are unaffected either way.",
    )
    p.add_argument("--report", default="/tmp/nearl_nvtx", help="nsys report basename")
    return p.parse_args()


def build_featurizer(args):
    loader = nearl.io.TrajectoryLoader(
        nearl.get_example_data(args.datadir)["MINI_TRAJSET"]
    )
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
    featurizer.register_features(
        [
            nearl.features.DensityFlow(
                selection="!:T3P",
                agg="standard_deviation",
                weight_type=args.weight,
                outkey="df",
                hdf_compress_level=args.compress,
            ),
            nearl.features.MarchingObservers(
                selection="!:T3P",
                obs="density",
                agg="mean",
                weight_type=args.weight,
                outkey="obs",
                hdf_compress_level=args.compress,
            ),
        ]
    )
    featurizer.register_trajloader(loader)
    featurizer.register_focus([":LIG"], "mask")
    return featurizer


def run_workload(args):
    if os.path.exists(args.outfile):
        os.remove(args.outfile)
    build_featurizer(args).run()


def profile_under_nsys(args):
    """Re-run this script under nsys, then print the NVTX and kernel reports."""
    report = pathlib.Path(args.report)
    for suffix in (".nsys-rep", ".sqlite"):
        report.with_suffix(suffix).unlink(missing_ok=True)

    env = dict(os.environ, **{CHILD_ENV: "1"})
    cmd = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--force-overwrite=true",
        "--output",
        str(report),
    ]
    if not args.cpu_sampling:
        # Suppresses the "does not allow enabling CPU profiling" warning on hosts
        # with a restrictive perf_event_paranoid. Nothing below reads the samples.
        cmd += ["--sample=none", "--cpuctxsw=none"]
    cmd += [
        sys.executable,
        os.path.abspath(__file__),
        *sys.argv[1:],
    ]
    print(f"[nsys] {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd, env=env).returncode != 0:
        print("[nsys] profiling run failed", file=sys.stderr)
        return 1

    rep = str(report.with_suffix(".nsys-rep"))
    for name in REPORTS:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}", flush=True)
        subprocess.run(["nsys", "stats", "--report", name, "--force-export=true", rep])
    return 0


def main():
    args = parse_args()
    if os.environ.get(CHILD_ENV) or args.no_nsys:
        run_workload(args)
        return 0
    return profile_under_nsys(args)


if __name__ == "__main__":
    sys.exit(main())
