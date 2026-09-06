#!/usr/bin/env python
"""
Compare how the three feature families reach the GPU.

Runs the same featurization three times -- once with a static feature, once with
MarchingObservers, once with DensityFlow -- each under Nsight Systems, and
reports the launch pattern rather than only the total time: how many kernels are
launched, how long each one runs, how much is copied, and how much of the wall
clock the device is actually busy.

The workload runs twice and only the second pass is timed. nsys profiles the
whole process, so a synthetic warm-up would add launches to the counts while
sitting outside the timed region; running the real pipeline twice keeps counts
and wall clock consistent, and CUDA context creation lands in the first pass.
Per-task figures therefore divide the totals by PASSES.

The closing section turns those numbers into optimisation leads.

Usage:
    python profiling_kernel_launches.py [--dims 32] [--window 10]
    python profiling_kernel_launches.py --feature density
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")

CHILD_ENV = "NEARL_LAUNCH_PROFILE_CHILD"

FEATURES = ("static", "observers", "density")

# The first pass pays for the CUDA context; the second is what gets timed.
PASSES = 2

# A kernel launch is "small" below this; the launch itself costs a few us.
SMALL_KERNEL_US = 50
# Below this the device is idle most of the wall clock.
LOW_OCCUPANCY = 0.25

# One hue per family, in fixed order. Checked for colour-vision separation
# against a light surface: worst pair dE 12.4 simulated, 17.5 unsimulated.
HUES = {"static": "#5B3A8E", "observers": "#C2477F", "density": "#D9772F"}
SURFACE, INK, MUTED = "#fcfcfb", "#22202b", "#6f6a7d"

# Alongside the pytest-benchmark saves, so one directory holds the evidence.
COMPUTE, MEMORY, IDLE = "#5B3A8E", "#D9772F", "#e6e3ec"

DEFAULT_PLOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".benchmarks",
    "kernel_launches.png",
)
DEFAULT_TIMELINE = DEFAULT_PLOT.replace("kernel_launches", "kernel_timeline")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", type=int, default=32, help="grid dimension per axis")
    p.add_argument("--window", type=int, default=10, help="frames per slice")
    p.add_argument("--datadir", default="/tmp/nearl_test", help="example-data folder")
    p.add_argument(
        "--feature",
        default="all",
        choices=(*FEATURES, "all"),
        help="which feature family to profile",
    )
    # Element-based, so it catches heavy atoms whatever they are named. The
    # ligand alone is 40 atoms, too small to load the kernels realistically.
    p.add_argument(
        "--selection", default="!@/H", help="atoms to voxelize (default: heavy atoms)"
    )
    p.add_argument(
        "--focus", default=":LIG", help="what the box centres on (default: the ligand)"
    )
    p.add_argument("--plot", metavar="PATH", default=DEFAULT_PLOT, help="PNG output")
    p.add_argument(
        "--timeline",
        metavar="PATH",
        default=DEFAULT_TIMELINE,
        help="PNG of the compute/memory split over time",
    )
    p.add_argument(
        "--no-plot", action="store_const", const=None, dest="plot", help="skip the PNG"
    )
    p.add_argument(
        "--no-timeline",
        action="store_const",
        const=None,
        dest="timeline",
        help="skip the timeline PNG",
    )
    return p.parse_args()


###############################################################################
# The workload
###############################################################################


def build_featurizer(args, kind):
    import nearl
    import nearl.features
    import nearl.featurizer
    import nearl.io

    loader = nearl.io.TrajectoryLoader(
        nearl.get_example_data(args.datadir)["MINI_TRAJSET"], mask="!:T3P"
    )
    featurizer = nearl.featurizer.Featurizer(
        {
            "dimensions": args.dims,
            "lengths": 16,
            "time_window": args.window,
            "sigma": 1.5,
            "cutoff": 3.5,
            "outfile": f"/tmp/nearl_launch_{kind}.h5",
        }
    )
    if kind == "static":
        feature = nearl.features.Mass(selection=args.selection, outkey="probe")
    elif kind == "observers":
        feature = nearl.features.MarchingObservers(
            selection=args.selection,
            obs="density",
            weight_type="mass",
            agg="mean",
            outkey="probe",
        )
    else:
        feature = nearl.features.DensityFlow(
            selection=args.selection,
            weight_type="mass",
            agg="mean",
            outkey="probe",
        )
    featurizer.register_features([feature])
    featurizer.register_trajloader(loader)
    featurizer.register_focus([args.focus], "mask")
    return featurizer


def run_child(args):
    """Profiled process: one feature family, one trajectory. Reports as JSON."""
    kind = args.feature
    featurizer = build_featurizer(args, kind)

    outfile = (
        featurizer.PARAMSPACE.get("outfile")
        if hasattr(featurizer, "PARAMSPACE")
        else None
    )
    paths = list(filter(None, [outfile, f"/tmp/nearl_launch_{kind}.h5"]))
    wall = 0.0
    for _ in range(PASSES):
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
        start = time.perf_counter()
        featurizer.run()
        wall = time.perf_counter() - start

    print(
        "NEARL_RESULT "
        + json.dumps(
            {
                "kind": kind,
                "wall_s": wall,
                "slices": getattr(featurizer, "SLICENUMBER", 0),
                "focal": getattr(featurizer, "FOCALNUMBER", 0),
                "passes": PASSES,
            }
        )
    )


###############################################################################
# Nsight Systems
###############################################################################


def _rows(text):
    """Pick the CSV table out of an nsys stats report, skipping its preamble."""
    lines = text.replace("\r", "\n").splitlines()
    for i, line in enumerate(lines):
        if any(k in line for k in ("Total Time (ns)", "Total (MB)", "Start (ns)")):
            return list(csv.DictReader(io.StringIO("\n".join(lines[i:]))))
    return []


def _first(row, *names, default="0"):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _num(value):
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return 0.0


def nsys_report(args, kind, workdir):
    """Profile one feature family; return the child's JSON plus the nsys tables."""
    report = os.path.join(workdir, f"nearl_{kind}")
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
            "--feature",
            kind,
            "--dims",
            str(args.dims),
            "--window",
            str(args.window),
            "--datadir",
            args.datadir,
            "--selection",
            args.selection,
            "--focus",
            args.focus,
        ],
        env=dict(os.environ, **{CHILD_ENV: "1"}),
        capture_output=True,
        text=True,
    )
    if child.returncode != 0:
        print(child.stdout[-1500:], file=sys.stderr)
        print(child.stderr[-1500:], file=sys.stderr)
        raise SystemExit(f"profiling the {kind} feature failed")

    summary = {}
    for line in child.stdout.replace("\r", "\n").splitlines():
        if line.startswith("NEARL_RESULT "):
            summary = json.loads(line[len("NEARL_RESULT ") :])

    tables = {}
    for name in (
        "cuda_gpu_kern_sum",
        "cuda_gpu_mem_time_sum",
        "cuda_gpu_mem_size_sum",
        "cuda_api_sum",
        "cuda_gpu_trace",
    ):
        out = subprocess.run(
            [
                "nsys",
                "stats",
                "--report",
                name,
                "--format",
                "csv",
                "--force-export",
                "true",
                f"{report}.nsys-rep",
            ],
            capture_output=True,
            text=True,
        ).stdout
        tables[name] = _rows(out)
    return summary, tables


###############################################################################
# Reporting
###############################################################################


def digest(summary, tables):
    kernels = [
        {
            "name": _first(row, "Name").split("(")[0][:38],
            "count": int(_num(_first(row, "Instances", "Count", "Num Calls"))),
            "total_ms": _num(_first(row, "Total Time (ns)")) / 1e6,
            "avg_us": _num(_first(row, "Avg (ns)")) / 1e3,
        }
        for row in tables["cuda_gpu_kern_sum"]
    ]
    transfers = [
        {
            "op": _first(row, "Operation"),
            "count": int(_num(_first(row, "Count", "Instances"))),
            "total_ms": _num(_first(row, "Total Time (ns)")) / 1e6,
        }
        for row in tables["cuda_gpu_mem_time_sum"]
    ]
    volumes = {
        _first(row, "Operation"): _num(_first(row, "Total (MB)"))
        for row in tables["cuda_gpu_mem_size_sum"]
    }
    api = {
        _first(row, "Name"): {
            "count": int(_num(_first(row, "Num Calls", "Count", "Instances"))),
            "total_ms": _num(_first(row, "Total Time (ns)")) / 1e6,
        }
        for row in tables["cuda_api_sum"]
    }
    events = []
    for row in tables.get("cuda_gpu_trace", []):
        name = _first(row, "Name")
        start, dur = _num(_first(row, "Start (ns)")), _num(_first(row, "Duration (ns)"))
        if dur > 0:
            events.append((start, dur, name.startswith("[CUDA mem")))
    events.sort()

    tasks = max(1, summary.get("slices", 1) * max(1, summary.get("focal", 1)))
    passes = max(1, summary.get("passes", 1))
    return {
        **summary,
        "tasks": tasks,
        "passes": passes,
        "events": events,
        "kernels": kernels,
        "transfers": transfers,
        "volumes": volumes,
        "api": api,
        "launches": sum(k["count"] for k in kernels),
        "kernel_ms": sum(k["total_ms"] for k in kernels),
        "transfer_ms": sum(t["total_ms"] for t in transfers),
        "transfer_mb": sum(volumes.values()),
    }


def show(d):
    wall_ms = d["wall_s"] * 1000
    print("\n" + "=" * 74)
    print(
        f"{d['kind'].upper():<12} {d['tasks']} tasks x {d['passes']} passes, "
        f"timed pass {d['wall_s']:.3f} s"
    )
    print("=" * 74)

    print(f"{'KERNEL':<40}{'launches':>10}{'total ms':>11}{'avg us':>10}")
    print("-" * 74)
    for k in sorted(d["kernels"], key=lambda k: -k["total_ms"]):
        print(
            f"{k['name']:<40}{k['count']:>10}{k['total_ms']:>11.3f}{k['avg_us']:>10.1f}"
        )
    if not d["kernels"]:
        print("(no kernels recorded)")

    print(f"\n{'TRANSFER':<40}{'count':>10}{'total ms':>11}{'MB':>10}")
    print("-" * 74)
    for t in sorted(d["transfers"], key=lambda t: -t["total_ms"]):
        print(
            f"{t['op'][:38]:<40}{t['count']:>10}{t['total_ms']:>11.3f}"
            f"{d['volumes'].get(t['op'], 0):>10.2f}"
        )

    print(f"\n{'CUDA API':<40}{'calls':>10}{'total ms':>11}{'per task':>10}")
    print("-" * 74)
    for name in (
        "cudaMalloc",
        "cudaFree",
        "cudaMemcpy",
        "cudaLaunchKernel",
        "cudaDeviceSynchronize",
    ):
        row = d["api"].get(name)
        if row:
            print(
                f"{name:<40}{row['count']:>10}{row['total_ms']:>11.3f}"
                f"{row['count'] / d['tasks'] / d['passes']:>10.1f}"
            )

    per_pass_ms = (d["kernel_ms"] + d["transfer_ms"]) / d["passes"]
    busy = per_pass_ms / wall_ms if wall_ms else 0
    print("-" * 74)
    print(
        f"{'device busy':<40}{busy * 100:>9.1f}%"
        f"   per pass: kernels {d['kernel_ms'] / d['passes']:.1f} ms"
        f" + transfers {d['transfer_ms'] / d['passes']:.1f} ms of {wall_ms:.0f} ms"
    )
    print(
        f"{'per task':<40}{d['launches'] / d['tasks'] / d['passes']:>9.1f} launches"
        f"   {d['transfer_mb'] / d['tasks'] / d['passes']:.3f} MB moved"
    )


def insights(digests):
    print("\n" + "=" * 74)
    print("WHERE THE TIME GOES")
    print("=" * 74)

    for d in digests:
        wall_ms = d["wall_s"] * 1000
        work = d["tasks"] * d["passes"]
        busy = (
            (d["kernel_ms"] + d["transfer_ms"]) / d["passes"] / wall_ms
            if wall_ms
            else 0
        )
        avg_us = (d["kernel_ms"] * 1000 / d["launches"]) if d["launches"] else 0
        notes = []

        if busy < LOW_OCCUPANCY:
            notes.append(
                f"device idle {100 * (1 - busy):.0f}% of the run -- the host loop, not "
                "the kernels, sets the pace"
            )
        if avg_us and avg_us < SMALL_KERNEL_US:
            notes.append(
                f"kernels average {avg_us:.0f} us, close to launch overhead; batching "
                f"the {d['tasks']} tasks into one launch would amortise it"
            )
        if d["transfer_ms"] > d["kernel_ms"]:
            notes.append(
                f"transfers ({d['transfer_ms'] / d['passes']:.1f} ms) outweigh "
                f"compute ({d['kernel_ms'] / d['passes']:.1f} ms); pinned buffers "
                "and a copy/compute overlap "
                "would hide them"
            )
        allocs = d["api"].get("cudaMalloc", {}).get("count", 0)
        if allocs > work:
            notes.append(
                f"{allocs / work:.0f} cudaMalloc per task -- allocate once and "
                "reuse the buffers"
            )
        syncs = d["api"].get("cudaDeviceSynchronize", {}).get("count", 0)
        if syncs >= work:
            notes.append(
                f"{syncs / work:.0f} device syncs per task: each one drains the "
                "pipeline before the next task starts"
            )

        print(f"\n{d['kind']}:")
        for note in notes or ["nothing stands out at this problem size"]:
            print(f"  - {note}")

    ranked = sorted(digests, key=lambda d: -d["kernel_ms"])
    print(
        f"\nMost device time: {ranked[0]['kind']} "
        f"({ranked[0]['kernel_ms'] / ranked[0]['passes']:.1f} ms of kernels per "
        "pass). Optimise that one first."
    )


def plot(digests, path, args):
    """Four panels, one per measure: the units differ, so they never share an axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = [d["kind"] for d in digests]
    colors = [HUES.get(k, MUTED) for k in kinds]
    panels = [
        (
            "Kernel launches per task",
            [d["launches"] / d["tasks"] / d["passes"] for d in digests],
            "{:.0f}",
        ),
        (
            "Average kernel duration (us)",
            [
                (d["kernel_ms"] * 1000 / d["launches"]) if d["launches"] else 0
                for d in digests
            ],
            "{:.1f}",
        ),
        (
            "Device busy (% of wall)",
            [
                100
                * (d["kernel_ms"] + d["transfer_ms"])
                / d["passes"]
                / (d["wall_s"] * 1000)
                for d in digests
            ],
            "{:.1f}",
        ),
        (
            "Data moved per task (MB)",
            [d["transfer_mb"] / d["tasks"] / d["passes"] for d in digests],
            "{:.2f}",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.4), facecolor=SURFACE)
    y = range(len(kinds))
    for ax, (title, values, fmt) in zip(axes.ravel(), panels):
        ax.set_facecolor(SURFACE)
        ax.barh(y, values, height=0.55, color=colors, linewidth=0)
        headroom = max(values) * 1.18 if max(values) else 1
        for i, v in zip(y, values):
            ax.text(
                v + headroom * 0.02,
                i,
                fmt.format(v),
                va="center",
                fontsize=9,
                color=INK,
            )
        ax.set_xlim(0, headroom)
        ax.set_yticks(list(y), kinds, fontsize=9.5, color=INK)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
        ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=0)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color="#e6e3ec", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    tasks = digests[0]["tasks"]
    fig.suptitle(
        f"Nearl feature families on the GPU  -  {args.dims}^3 grid, "
        f"{args.window}-frame window, {tasks} tasks, "
        f"selection {args.selection} focused on {args.focus}",
        fontsize=11,
        color=INK,
        x=0.012,
        ha="left",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nwrote {path}")


def plot_timeline(digests, path, args):
    """
    Left: how the GPU-active window splits between compute, memory and idle.
    Right: the same run zoomed to a couple of tasks, so the rhythm is visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    usable = [d for d in digests if d["events"]]
    if not usable:
        print("no GPU trace recorded; skipping the timeline")
        return

    fig = plt.figure(figsize=(11, 1.55 * len(usable) + 1.5), facecolor=SURFACE)
    grid = GridSpec(
        len(usable), 2, width_ratios=[1, 1.9], figure=fig, hspace=0.55, wspace=0.16
    )

    for row, d in enumerate(usable):
        events = d["events"]
        span_start, span_end = events[0][0], max(e[0] + e[1] for e in events)
        span = span_end - span_start
        compute = sum(e[1] for e in events if not e[2])
        memory = sum(e[1] for e in events if e[2])
        idle = max(0.0, span - compute - memory)

        # -- the split ---------------------------------------------------------
        ax = fig.add_subplot(grid[row, 0])
        ax.set_facecolor(SURFACE)
        left = 0.0
        for value, color in ((compute, COMPUTE), (memory, MEMORY), (idle, IDLE)):
            ax.barh(
                0, 100 * value / span, left=left, height=0.42, color=color, linewidth=0
            )
            left += 100 * value / span
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0], [d["kind"]], fontsize=10, color=INK)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=8, colors=MUTED, length=0)
        ax.text(
            0,
            0.34,
            f"compute {100 * compute / span:.1f}%   "
            f"memory {100 * memory / span:.1f}%   "
            f"idle {100 * idle / span:.1f}%",
            fontsize=8.5,
            color=MUTED,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
        )
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d8d3e2")
        if row == len(usable) - 1:
            ax.set_xlabel(
                "share of the GPU-active window (%)", fontsize=8.5, color=MUTED
            )

        # -- the zoom ----------------------------------------------------------
        # Start in the second pass, and show a couple of tasks' worth.
        width = span / d["passes"] / d["tasks"] * 2.5
        origin = next(
            (e[0] for e in events if e[0] >= span_start + span / 2 and not e[2]),
            span_start,
        )
        ax = fig.add_subplot(grid[row, 1])
        ax.set_facecolor(SURFACE)
        for start, dur, is_mem in events:
            if start > origin + width or start + dur < origin:
                continue
            ax.barh(
                0 if is_mem else 1,
                dur / 1e6,
                left=(start - origin) / 1e6,
                height=0.5,
                color=MEMORY if is_mem else COMPUTE,
                linewidth=0,
            )
        ax.set_xlim(0, width / 1e6)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([1, 0], ["compute", "memory"], fontsize=8.5, color=MUTED)
        ax.tick_params(labelsize=8, colors=MUTED, length=0)
        ax.set_title(
            f"{d['kind']} - {width / 1e6:.1f} ms window, mid-run",
            fontsize=9,
            color=INK,
            loc="left",
            pad=4,
        )
        ax.xaxis.grid(True, color="#eeebf3", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d8d3e2")
        if row == len(usable) - 1:
            ax.set_xlabel("milliseconds", fontsize=8.5, color=MUTED)

    fig.suptitle(
        f"Where the GPU time goes  -  {args.dims}^3 grid, {args.window}-frame window, "
        f"selection {args.selection}",
        fontsize=11,
        color=INK,
        x=0.012,
        ha="left",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    args = parse_args()
    if os.environ.get(CHILD_ENV):
        run_child(args)
        return

    import shutil

    if shutil.which("nsys") is None:
        raise SystemExit("nsys not on PATH (it ships with the CUDA toolkit)")

    kinds = FEATURES if args.feature == "all" else (args.feature,)
    digests = []
    with tempfile.TemporaryDirectory() as workdir:
        for kind in kinds:
            print(f"\nprofiling {kind} ...", flush=True)
            summary, tables = nsys_report(args, kind, workdir)
            d = digest(summary, tables)
            show(d)
            digests.append(d)
    if digests:
        insights(digests)
        if args.plot:
            plot(digests, args.plot, args)
        if args.timeline:
            plot_timeline(digests, args.timeline, args)


if __name__ == "__main__":
    main()
