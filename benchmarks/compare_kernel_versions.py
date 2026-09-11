#!/usr/bin/env python
"""
Time the three GPU kernels on one checkout, so two checkouts can be compared.

``test_benchmark.py`` times the same entry points, but only in-process and only
for one marching-observer observable. This script is version-agnostic on purpose
-- it touches nothing but ``frame_voxelize`` / ``density_flow`` /
``marching_observer``, whose signatures are unchanged since v0.1.0 -- so the same
file can be dropped into an old worktree and the two JSON dumps diffed.

Marching observers is timed four times, because a single observable hides how
differently they load the kernel: ``existence`` (cheap -- an early-out distance
test), ``distinct_count`` (expensive -- a 4 KB per-thread scratch array and an
O(atoms x distinct) search), ``radius_of_gyration`` (expensive for a different
reason -- it walks the neighbourhood twice, once for the centre of mass and once
for the spread about it) and ``dispersion`` (the worst case -- O(atoms^2) per
grid point, and the one observable no optimization has targeted yet).

The default shape is the reference workload the optimization work was tuned on:
dims 32^3, spacing 0.5, cutoff 3.5, sigma 1.5, 10 frames, 1000 atoms.

Usage:
    python compare_kernel_versions.py --label v0.1.0 --out v010.json
    python compare_kernel_versions.py --label main   --out main.json
    python compare_kernel_versions.py --compare v010.json main.json

Reproducing a two-version comparison on a fresh machine
-------------------------------------------------------
Needs nvcc, an NVIDIA driver, and an env with numpy + pybind11. The Makefile
reads the target arch from ``nvidia-smi``, so nothing is pinned to one GPU.

    # 1. build the current checkout and install it into the env
    cd /path/to/nearl
    pip install -e . --no-deps

    # 2. build the old tag in its own worktree
    git worktree add /tmp/nearl-v010 v0.1.0
    cd /tmp/nearl-v010/src && make all_actions
    cp all_actions.so ../nearl/all_actions$(python3-config --extension-suffix)

    # 3. this script is not in the old tag -- copy it across
    cp /path/to/nearl/benchmarks/compare_kernel_versions.py /tmp/nearl-v010/benchmarks/

    # 4. measure each, then diff
    cd /tmp/nearl-v010
    PYTHONPATH=/tmp/nearl-v010 python benchmarks/compare_kernel_versions.py \
        --label v0.1.0 --out /tmp/v010.json
    cd /path/to/nearl
    python benchmarks/compare_kernel_versions.py --label main --out /tmp/main.json
    python benchmarks/compare_kernel_versions.py --compare /tmp/v010.json /tmp/main.json

Three things that will silently give you wrong numbers:

* ``make all_actions`` builds ``src/all_actions.so`` but does **not** install it
  next to ``nearl/__init__.py`` -- ``setup.py`` is what normally copies it. Step 2
  does that by hand; skip it and you time whatever ``.so`` was already there.
* ``PYTHONPATH`` has to point at the worktree, or ``import nearl`` resolves to the
  editable install of the new checkout and both runs measure the same code. The
  ``label`` and ``context`` lines in the output are the check: v0.1.0 has no
  device context and must print ``context : off``.
* Run both halves on an idle GPU in one sitting. Absolute times are specific to
  the card; the ratio is the portable number.

The ``checksum`` column in ``--compare`` is what makes the timings meaningful --
it is the grid sum from each version, so a kernel that got faster by computing
something else shows up as a large relative difference rather than a speedup.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time

import numpy as np

# Observable codes, mirrored from features.SUPPORTED_OBSERVATION (identical on
# both versions). Named here so the script does not import features.py, which
# has drifted between the two checkouts.
OBS_EXISTENCE = 1
OBS_DISTINCT_COUNT = 3
OBS_DISPERSION = 14
OBS_RADIUS_OF_GYRATION = 16
AGG_MEAN = 1

CASES = (
    "frame_voxelize",
    "density_flow",
    "mobs_cheap",
    "mobs_expensive",
    "mobs_gyration",
    "mobs_worstcase",
)

LABELS = {
    "frame_voxelize": "frame_voxelize",
    "density_flow": "density_flow",
    "mobs_cheap": "marching obs. (existence)",
    "mobs_expensive": "marching obs. (distinct_count)",
    "mobs_gyration": "marching obs. (radius_of_gyration)",
    "mobs_worstcase": "marching obs. (dispersion)",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="HEAD", help="name for this checkout")
    p.add_argument("--out", help="write the timings to this JSON file")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    p.add_argument("--dims", type=int, default=32, help="grid dimension per axis")
    p.add_argument("--frames", type=int, default=10, help="frames per slice")
    p.add_argument("--atoms", type=int, default=1000, help="atoms per frame")
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--cutoff", type=float, default=3.5)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--repeats", type=int, default=50, help="timed calls per case")
    p.add_argument("--warmup", type=int, default=10, help="untimed calls per case")
    p.add_argument(
        "--no-context",
        action="store_true",
        help="skip init_context(), i.e. the per-call cudaMalloc path (v0.1.0 has "
        "no context and always runs this way)",
    )
    return p.parse_args()


# Atomic masses of the elements a protein box is actually made of. A constant
# weight would leave ``distinct_count`` with one distinct value to find, which
# is exactly the case its search loop never has to work for.
ELEMENT_MASSES = (1.008, 12.011, 14.007, 15.999, 30.974, 32.06)


def make_inputs(args):
    """One frame slice of atoms, centred on the grid so the box is actually busy."""
    rng = np.random.default_rng(0)
    extent = args.dims * args.spacing
    traj = rng.normal(
        loc=extent / 2, scale=extent / 6, size=(args.frames, args.atoms, 3)
    ).astype(np.float32)
    weights = rng.choice(
        ELEMENT_MASSES,
        size=args.frames * args.atoms,
        p=[0.5, 0.3, 0.08, 0.1, 0.01, 0.01],
    ).astype(np.float32)
    dims = np.array([args.dims] * 3, dtype=np.int32)
    return traj, weights, dims


def build_calls(commands, args):
    traj, weights, dims = make_inputs(args)
    frame0, weights0 = traj[0], weights[: args.atoms]

    return {
        "frame_voxelize": lambda: commands.frame_voxelize(
            frame0, weights0, dims, args.spacing, args.cutoff, args.sigma
        ),
        "density_flow": lambda: commands.density_flow(
            traj, weights, dims, args.spacing, args.cutoff, args.sigma, AGG_MEAN
        ),
        "mobs_cheap": lambda: commands.marching_observer(
            traj, weights, dims, args.spacing, args.cutoff, OBS_EXISTENCE, AGG_MEAN
        ),
        "mobs_expensive": lambda: commands.marching_observer(
            traj, weights, dims, args.spacing, args.cutoff, OBS_DISTINCT_COUNT, AGG_MEAN
        ),
        "mobs_gyration": lambda: commands.marching_observer(
            traj,
            weights,
            dims,
            args.spacing,
            args.cutoff,
            OBS_RADIUS_OF_GYRATION,
            AGG_MEAN,
        ),
        "mobs_worstcase": lambda: commands.marching_observer(
            traj, weights, dims, args.spacing, args.cutoff, OBS_DISPERSION, AGG_MEAN
        ),
    }


def time_case(fn, repeats, warmup):
    """Median/min/max wall time per call, in milliseconds.

    Every one of these host functions ends in a device synchronize, so the wall
    clock around the call is the kernel plus its transfers -- no extra sync
    needed, and none wanted, since the per-call host overhead is part of what
    changed between the two versions.
    """
    for _ in range(warmup):
        fn()

    # A case costing ~150 ms/call does not need 50 samples to be stable, and
    # paying for them makes the script unpleasant to rerun.
    if repeats > 5:
        probe = time.perf_counter()
        fn()
        if (time.perf_counter() - probe) > 0.05:
            repeats = 5

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ret = fn()
        samples.append((time.perf_counter() - t0) * 1e3)

    grid = np.asarray(ret)
    if not np.isfinite(grid).all():
        raise SystemExit("kernel returned non-finite values; timing is meaningless")

    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "checksum": float(grid.sum()),
    }


def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def measure(args):
    from nearl import commands

    context = False
    if not args.no_context and hasattr(commands, "init_context"):
        commands.init_context()
        context = bool(commands.context_valid())

    calls = build_calls(commands, args)
    print(f"label      : {args.label}")
    print(f"gpu        : {gpu_name()}")
    print(
        f"shape      : dims {args.dims}^3, {args.frames} frames, {args.atoms} atoms, "
        f"spacing {args.spacing}, cutoff {args.cutoff}, sigma {args.sigma}"
    )
    print(f"context    : {'on' if context else 'off (per-call cudaMalloc)'}")
    print(f"\n{'kernel':<38}{'median ms':>12}{'min ms':>10}")
    print("-" * 60)

    results = {}
    for case in CASES:
        results[case] = time_case(calls[case], args.repeats, args.warmup)
        print(
            f"{LABELS[case]:<38}{results[case]['median_ms']:>12.3f}"
            f"{results[case]['min_ms']:>10.3f}"
        )

    payload = {
        "label": args.label,
        "gpu": gpu_name(),
        "python": platform.python_version(),
        "context": context,
        "shape": {
            "dims": args.dims,
            "frames": args.frames,
            "atoms": args.atoms,
            "spacing": args.spacing,
            "cutoff": args.cutoff,
            "sigma": args.sigma,
            "repeats": args.repeats,
        },
        "results": results,
    }

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwritten to {args.out}")
    return payload


def compare(before_path, after_path):
    with open(before_path) as fh:
        before = json.load(fh)
    with open(after_path) as fh:
        after = json.load(fh)

    if before["shape"] != after["shape"]:
        print(
            "WARNING: the two runs used different shapes; speedups are not comparable"
        )

    s = before["shape"]
    print(f"gpu   : {after['gpu']}")
    print(
        f"shape : dims {s['dims']}^3, {s['frames']} frames, {s['atoms']} atoms, "
        f"spacing {s['spacing']}, cutoff {s['cutoff']}, sigma {s['sigma']}"
    )
    print(f"\n{'kernel':<38}{before['label']:>12}{after['label']:>12}{'speedup':>10}")
    print("-" * 72)
    for case in CASES:
        b = before["results"][case]["median_ms"]
        a = after["results"][case]["median_ms"]
        print(f"{LABELS[case]:<38}{b:>12.3f}{a:>12.3f}{b / a:>9.1f}x")

    print("\n(median ms per call, including host-device transfers and the sync)")

    # A kernel that got faster by returning something else is not a speedup.
    print(f"\n{'kernel':<38}{'checksum ' + before['label']:>22}{'rel. diff':>14}")
    print("-" * 74)
    for case in CASES:
        cb = before["results"][case]["checksum"]
        ca = after["results"][case]["checksum"]
        rel = abs(ca - cb) / max(abs(cb), 1e-9)
        print(f"{LABELS[case]:<38}{cb:>22.4f}{rel:>14.2e}")


def main():
    args = parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        measure(args)


if __name__ == "__main__":
    main()
