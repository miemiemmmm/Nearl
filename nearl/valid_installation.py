"""
Validate a Nearl installation.

Run with ``python -m nearl.valid_installation``. Exits non-zero if a required
component is missing or the GPU voxelizer does not return a sane result.
"""

import sys


def _check_core():
    import nearl.commands
    import nearl.features
    import nearl.featurizer
    import nearl.io
    import nearl.utils  # noqa: F401


def _check_pytraj():
    import pytraj

    return pytraj.__version__


def _check_extension():
    from nearl import all_actions  # noqa: F401


def _check_voxelize():
    import numpy as np

    from nearl import commands

    coords = np.random.normal(size=(100, 3), loc=5, scale=2)
    weights = np.full(100, 1.0)
    grid = commands.frame_voxelize(coords, weights, np.array([32, 32, 32]), 0.5, 5, 2)
    if grid.shape != (32, 32, 32):
        raise ValueError(f"unexpected voxel shape {grid.shape}")
    if not np.isfinite(grid).all():
        raise ValueError("voxel grid contains non-finite values")


CHECKS = [
    ("core modules", _check_core),
    ("pytraj backend", _check_pytraj),
    ("CUDA extension (nearl.all_actions)", _check_extension),
    ("GPU voxelization", _check_voxelize),
]


def main():
    try:
        import nearl
    except ImportError as e:
        print(f"ImportError: {e}", file=sys.stderr)
        print("Please check if the package is installed correctly.", file=sys.stderr)
        return 1

    print(f"Nearl version {nearl.__version__}")
    print("Testing the installation...")

    failed = 0
    for i, (name, check) in enumerate(CHECKS, start=1):
        print(f"Performing test {i} - {name:36s}: ", end="", flush=True)
        try:
            detail = check()
            print("OK" if detail is None else f"OK ({detail})", flush=True)
        except Exception as e:
            failed += 1
            print(f"FAILED ({type(e).__name__}: {e})", flush=True)

    if failed:
        print(
            f"Installation validation failed: {failed} of {len(CHECKS)} checks failed.",
            file=sys.stderr,
        )
        return 1
    print(f"Installation validation successful: {len(CHECKS)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
