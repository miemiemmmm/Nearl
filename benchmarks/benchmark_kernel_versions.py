"""Compare Nearl's CUDA kernels between two builds (e.g. v0.1.0 and now).

Self-contained and portable: synthetic inputs only, no trajectory files, no
plotting dependency for the measuring step. Run it once per build, then compare
the two JSON files.

    # in a checkout of the old tag, built
    python benchmarks/benchmark_kernel_versions.py run --label v0.1.0 --out old.json
    # in a checkout of the new branch, built
    python benchmarks/benchmark_kernel_versions.py run --label optimized --out new.json
    # anywhere with matplotlib
    python benchmarks/benchmark_kernel_versions.py compare old.json new.json --plot speedup

Notes for a module-based cluster (Alps):
  * Nearl is usually installed non-editable there, so run this from OUTSIDE the
    source checkout or the source directory shadows the installed package. The
    script prints which extension it actually loaded -- check it before
    trusting any number.
  * Only `run` needs a GPU; `compare` needs matplotlib and nothing else.
  * v0.1.0 predates init_context(); the script detects that and reports which
    mode it used, so the two JSONs stay comparable.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

DIMS = [16, 24, 32, 48, 64, 96, 128]
ATOM_NR = 300
FRAME_NR = 50
SPACING = 1.0
CUTOFF = 2.5
SIGMA = 1.0
REPEATS = 20
WARMUP = 5
PASSES = 3


def describe_build():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        from nearl import all_actions

        info["extension"] = all_actions.__file__
    except Exception as exc:  # pragma: no cover - reported, not raised
        info["extension"] = f"unavailable: {exc}"
    for key, cmd in (
        ("commit", ["git", "rev-parse", "--short", "HEAD"]),
        ("describe", ["git", "describe", "--tags", "--always", "--dirty"]),
    ):
        try:
            info[key] = (
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    # resolve against the script's own checkout: on a cluster
                    # this is meant to run from outside the source tree
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                ).stdout.strip()
                or "unknown"
            )
        except Exception:
            info[key] = "unknown"
    try:
        info["gpu"] = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except Exception:
        info["gpu"] = "unknown"
    return info


def timed(fn, sync):
    for _ in range(WARMUP):
        fn()
    sync()
    passes = []
    for _ in range(PASSES):
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            fn()
        sync()
        passes.append((time.perf_counter() - t0) / REPEATS * 1e3)
    return statistics.median(passes)


def measure(label, out_path):
    from nearl import commands

    info = describe_build()
    print(f"extension : {info['extension']}")
    print(f"build     : {info['describe']}  ({info['commit']})")
    print(f"gpu       : {info['gpu']}")

    # v0.1.0 has no persistent context; both builds are still measured the same
    # way, the newer one simply gets the context it is designed around.
    has_context = hasattr(commands, "init_context")
    if has_context:
        commands.init_context()
    print(
        f"context   : {'active' if has_context else 'not available (pre-0.2 build)'}\n"
    )

    def sync():
        # No public device sync in v0.1.0; every command already blocks before
        # returning, so the loop above is already serialized against the device.
        return None

    rng = np.random.default_rng(0)
    coords = rng.normal(size=(ATOM_NR, 3), loc=5, scale=1).astype(np.float32)
    w_frame = np.full((ATOM_NR,), 1.5, dtype=np.float32)
    traj = rng.normal(size=(FRAME_NR, ATOM_NR, 3), loc=5, scale=2).astype(np.float32)
    w_traj = np.full((FRAME_NR * ATOM_NR,), 16.0, dtype=np.float32)

    rows = []
    print(
        f"{'dim':>5} {'frame_voxelize':>16} {'density_flow':>14} {'marching_observer':>19}"
    )
    for dim in DIMS:
        grid = np.array([dim] * 3, dtype=np.int32)
        entry = {"dim": dim}
        entry["voxel"] = timed(
            lambda g=grid: commands.frame_voxelize(
                coords, w_frame, g, SPACING, CUTOFF, SIGMA
            ),
            sync,
        )
        entry["flow"] = timed(
            lambda g=grid: commands.density_flow(
                traj, w_traj, g, SPACING, CUTOFF, SIGMA, 1
            ),
            sync,
        )
        entry["observer"] = timed(
            lambda g=grid: commands.marching_observer(
                traj, w_traj, g, SPACING, CUTOFF, 1, 1
            ),
            sync,
        )
        rows.append(entry)
        print(
            f"{dim:5d} {entry['voxel']:13.3f} ms {entry['flow']:11.3f} ms "
            f"{entry['observer']:16.3f} ms"
        )

    if has_context:
        commands.finalize_context()
    payload = {
        "label": label,
        "build": info,
        "has_context": has_context,
        "params": {
            "atoms": ATOM_NR,
            "frames": FRAME_NR,
            "spacing": SPACING,
            "cutoff": CUTOFF,
            "sigma": SIGMA,
            "repeats": REPEATS,
            "passes": PASSES,
        },
        "rows": rows,
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {out_path}")


TIMES = "\u00d7"  # keep the glyph out of the source

KERNELS = [
    ("voxel", "frame_voxelize", "#2a78d6"),
    ("flow", "density_flow", "#1baf7a"),
    ("observer", "marching_observer", "#eb6834"),
]


def compare(old_path, new_path, stem):
    with open(old_path) as handle:
        old = json.load(handle)
    with open(new_path) as handle:
        new = json.load(handle)
    dims = [
        r["dim"] for r in old["rows"] if r["dim"] in {n["dim"] for n in new["rows"]}
    ]

    def col(payload, key):
        by_dim = {r["dim"]: r[key] for r in payload["rows"]}
        return np.array([by_dim[d] for d in dims], dtype=float)

    print(
        f"{old['label']} ({old['build']['describe']})  ->  "
        f"{new['label']} ({new['build']['describe']})"
    )
    print(f"gpu: {new['build']['gpu']}\n")
    print(
        f"{'kernel':20s} {'dim':>5} {'before ms':>11} {'after ms':>10} {'speedup':>9}"
    )
    for key, name, _ in KERNELS:
        o, n = col(old, key), col(new, key)
        for i in range(len(dims)):
            print(
                f"{name:20s} {dims[i]:5d} {o[i]:11.3f} {n[i]:10.3f} {o[i] / n[i]:8.1f}x"
            )

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib unavailable; skipped the figure)")
        return

    INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
    SURFACE, GRID = "#fcfcfb", "#e7e6e2"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK3)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_xscale("log")
        ax.set_xticks(dims)
        ax.set_xticklabels(dims)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlabel("grid dimension  (voxels per side)", fontsize=9.5, color=INK2)

    for key, name, color in KERNELS:
        o, n = col(old, key), col(new, key)
        ax1.plot(
            dims,
            o,
            color=color,
            lw=2.0,
            ls=(0, (4, 2)),
            marker="o",
            ms=4,
            mfc=SURFACE,
            mew=1.4,
            zorder=3,
        )
        ax1.plot(dims, n, color=color, lw=2.0, marker="o", ms=5, zorder=4)
        ax2.plot(dims, o / n, color=color, lw=2.2, marker="o", ms=5, zorder=4)
        ax2.annotate(
            f"{name}  {o[-1] / n[-1]:.0f}{TIMES}",
            (dims[-1], (o / n)[-1]),
            textcoords="offset points",
            xytext=(7, 0),
            fontsize=8.5,
            color=color,
            va="center",
            fontweight="bold",
        )
    ax1.set_yscale("log")
    ax1.set_ylabel("mean call time  (ms, log)", fontsize=9.5, color=INK2)
    ax1.set_title("Kernel cost", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax1.text(
        0,
        1.015,
        f"dashed {old['label']}   solid {new['label']}",
        transform=ax1.transAxes,
        fontsize=9,
        color=INK2,
        va="bottom",
    )
    ax2.set_yscale("log")
    ax2.axhline(1.0, color=INK3, lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax2.set_ylabel("speedup (log)", fontsize=9.5, color=INK2)
    ax2.set_title("Speedup", fontsize=11, color=INK, loc="left", fontweight="bold")
    ax2.text(
        0,
        1.015,
        f"{new['build']['gpu']}",
        transform=ax2.transAxes,
        fontsize=9,
        color=INK2,
        va="bottom",
    )
    ax2.set_xlim(dims[0] * 0.9, dims[-1] * 2.2)
    fig.tight_layout()
    for ext in ("jpg", "png"):
        fig.savefig(f"{stem}.{ext}", facecolor=SURFACE)
    print(f"\nwrote {stem}.jpg / .png")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run", help="measure the current build")
    run.add_argument("--label", default="current")
    run.add_argument("--out", default="kernel_versions.json")
    cmp_ = sub.add_parser("compare", help="compare two JSON files and plot")
    cmp_.add_argument("old")
    cmp_.add_argument("new")
    cmp_.add_argument("--plot", default="kernel_versions")
    args = parser.parse_args()
    if args.mode == "run":
        measure(args.label, args.out)
    else:
        compare(args.old, args.new, args.plot)


if __name__ == "__main__":
    main()
