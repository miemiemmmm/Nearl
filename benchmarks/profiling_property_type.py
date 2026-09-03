#!/usr/bin/env python
"""
Time ``cache()`` for every weight/property type.

Separates the properties derived from the topology alone from the ones that call
out to RDKit/OpenBabel/ChargeFW2. Only the cheap ones belong in a GPU benchmark;
the rest measure third-party libraries rather than Nearl.

Usage:
    python profiling_property_type.py [--selection '!:T3P'] [--datadir /tmp/nearl_test]
"""

import argparse
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

import nearl
import nearl.features as F
import nearl.io

# A property an order of magnitude slower than the cheapest is calling a toolkit.
COSTLY_FACTOR = 10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", default="!:T3P", help="atom selection mask")
    p.add_argument("--datadir", default="/tmp/nearl_test", help="example-data folder")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"nearl from : {nearl.__file__}")

    traj = nearl.io.TrajectoryLoader(
        nearl.get_example_data(args.datadir)["MINI_TRAJSET"]
    )[0]
    print(f"trajectory : {traj.top.n_atoms} atoms, {traj.n_frames} frames")
    print(f"selection  : {args.selection}\n")

    print(f"{'property_type':<20}{'code':>6}{'cache() s':>12}  status")
    print("-" * 70)

    timed, broken = [], []
    for name, code in sorted(F.SUPPORTED_FEATURES.items(), key=lambda kv: kv[1]):
        try:
            feat = F.DensityFlow(
                selection=args.selection, agg="mean", weight_type=name, outkey="probe"
            )
            t0 = time.perf_counter()
            feat.cache(traj)
            dt = time.perf_counter() - t0
            timed.append((dt, name))
            print(f"{name:<20}{code:>6}{dt:>12.4f}  ok")
        except Exception as e:
            frame = traceback.extract_tb(e.__traceback__)[-1]
            where = f"{frame.filename.split('/')[-1]}:{frame.lineno}"
            broken.append((name, code, type(e).__name__, str(e), where))
            print(f"{name:<20}{code:>6}{'-':>12}  {type(e).__name__} at {where}")

    if not timed:
        raise SystemExit("no property type could be cached")

    timed.sort()
    fastest = timed[0][0]
    print("\n" + "=" * 70)
    print(f"{'RANKED BY COST':<20}{'cache() s':>12}{'x fastest':>12}")
    print("-" * 70)
    for dt, name in timed:
        print(f"{name:<20}{dt:>12.4f}{dt / fastest:>11.1f}x")

    cheap = [n for dt, n in timed if dt < fastest * COSTLY_FACTOR]
    costly = [n for dt, n in timed if dt >= fastest * COSTLY_FACTOR]
    print("\n" + "=" * 70)
    print(f"Usable in a GPU benchmark ({len(cheap)}):\n  {', '.join(cheap)}")
    if costly:
        print(
            f"\nDominated by external libraries ({len(costly)}):\n  {', '.join(costly)}"
        )
    if broken:
        print(f"\nUnusable as weight_type ({len(broken)}):")
        for name, code, exc, msg, where in broken:
            print(f"  {name:<18} (code {code:>2})  {exc} at {where}")
            print(f"  {'':<18} {msg[:76]}")


if __name__ == "__main__":
    main()
