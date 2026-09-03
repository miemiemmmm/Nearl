"""
Validate a Nearl installation.

Run with ``python -m nearl.valid_installation``. The static checks need no GPU,
so a build can be verified on a GPU-less login node before spending an
allocation. Checks that have to launch a kernel are reported as skipped rather
than passed when no CUDA device is visible; ``--require-gpu`` turns those skips
into failures.
"""

import argparse
import ctypes
import subprocess
import sys

EXPECTED_SYMBOLS = (
    "aggregate",
    "density_flow",
    "frame_observation",
    "frame_voxelize",
    "marching_observer",
    "summation",
)

_CUDART_CANDIDATES = (
    "libcudart.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so.11.0",
)

# cudaDevAttrComputeCapabilityMajor / Minor
_ATTR_CC_MAJOR = 75
_ATTR_CC_MINOR = 76


class Skipped(Exception):
    """Raised by a check that cannot run in this environment."""


class NoDevice(Skipped):
    """Raised when a check needs a CUDA device and there is none reachable."""


def _parse_arch(arch):
    """``sm_86`` -> ``(8, 6)``; trailing feature suffixes (``sm_90a``) are ignored."""
    digits = arch[3:].rstrip("abcdef")
    return int(digits[:-1]), int(digits[-1])


def _cudart():
    for lib in _CUDART_CANDIDATES:
        try:
            return ctypes.CDLL(lib)
        except OSError:
            continue
    return None


def _device_count():
    rt = _cudart()
    if rt is None:
        raise NoDevice("CUDA runtime (libcudart) not found")
    count = ctypes.c_int(0)
    if rt.cudaGetDeviceCount(ctypes.byref(count)) != 0:
        return 0
    return count.value


def _device_arch():
    rt = _cudart()
    caps = []
    for attr in (_ATTR_CC_MAJOR, _ATTR_CC_MINOR):
        value = ctypes.c_int(-1)
        if rt.cudaDeviceGetAttribute(ctypes.byref(value), attr, 0) != 0:
            raise ValueError("could not query the device compute capability")
        caps.append(value.value)
    return f"sm_{caps[0]}{caps[1]}"


def _embedded_archs():
    """GPU architectures baked into the extension, read straight out of the binary."""
    from nearl import all_actions

    sofile = all_actions.__file__
    found = {}
    for flag, key in (("--list-elf", "sass"), ("--list-ptx", "ptx")):
        try:
            proc = subprocess.run(
                ["cuobjdump", flag, sofile],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            raise Skipped("cuobjdump not on PATH (needs the CUDA toolkit)") from e
        archs = set()
        for line in proc.stdout.splitlines():
            for token in line.replace(".", " ").split():
                if token.startswith("sm_") and token[3:4].isdigit():
                    archs.add(token)
        found[key] = archs
    return found


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
    from nearl import all_actions

    missing = [s for s in EXPECTED_SYMBOLS if not hasattr(all_actions, s)]
    if missing:
        raise ValueError(f"missing exported symbols: {', '.join(missing)}")
    return f"{len(EXPECTED_SYMBOLS)} symbols"


def _check_embedded_archs():
    found = _embedded_archs()
    if not found["sass"] and not found["ptx"]:
        raise ValueError("no GPU code found in the extension")
    sass = ", ".join(sorted(found["sass"])) or "none"
    ptx = ", ".join(sorted(found["ptx"])) or "none"
    return f"SASS {sass}; PTX {ptx}"


def _check_device():
    count = _device_count()
    if count == 0:
        raise NoDevice("no CUDA device visible")
    return f"{count} device(s), {_device_arch()}"


def _check_arch_compatibility():
    if _device_count() == 0:
        raise NoDevice("no CUDA device visible")
    device = _device_arch()
    dev_major, dev_minor = _parse_arch(device)
    found = _embedded_archs()

    if device in found["sass"]:
        return f"device {device} has native SASS"
    # A cubin runs on the same major version with an equal or higher minor.
    compatible = [
        a
        for a in found["sass"]
        if _parse_arch(a)[0] == dev_major and _parse_arch(a)[1] <= dev_minor
    ]
    if compatible:
        return f"device {device} runs the {', '.join(sorted(compatible))} SASS"
    # No usable cubin: the driver can still JIT from PTX of an equal or lower arch.
    jittable = [a for a in found["ptx"] if _parse_arch(a) <= (dev_major, dev_minor)]
    if jittable:
        return (
            f"device {device} has no native SASS; will JIT from PTX "
            f"({', '.join(sorted(jittable))}) - rebuild with "
            f"CUDA_COMPUTE_CAPABILITY={device} to tune for this GPU"
        )
    raise ValueError(
        f"the extension is built for {', '.join(sorted(found['sass'])) or 'nothing'} "
        f"and cannot run on this {device} device; rebuild with "
        f"CUDA_COMPUTE_CAPABILITY={device}"
    )


def _check_voxelization():
    if _device_count() == 0:
        raise NoDevice("no CUDA device visible")

    import numpy as np

    from nearl import commands

    rng = np.random.default_rng(0)
    coords = rng.normal(size=(100, 3), loc=5, scale=2)
    weights = np.full(100, 1.0)
    grid = commands.frame_voxelize(coords, weights, np.array([32, 32, 32]), 0.5, 5, 2)
    if grid.shape != (32, 32, 32):
        raise ValueError(f"unexpected voxel shape {grid.shape}")
    if not np.isfinite(grid).all():
        raise ValueError("voxel grid contains non-finite values")
    total = float(grid.sum())
    # CUDA_CHECK in src/ turns a failed call into a RuntimeError, but an empty
    # grid still means nothing was computed, so keep asserting on the value.
    if total <= 0.0:
        raise ValueError("voxel grid is empty; the kernels did not run")
    return f"grid sum {total:.2f}"


STATIC_CHECKS = (
    ("core modules", _check_core),
    ("pytraj backend", _check_pytraj),
    ("CUDA extension symbols", _check_extension),
    ("embedded GPU architectures", _check_embedded_archs),
)

RUNTIME_CHECKS = (
    ("CUDA device", _check_device),
    ("architecture compatibility", _check_arch_compatibility),
    ("GPU voxelization", _check_voxelization),
)


def _run(checks, counter, require_gpu):
    failed = skipped = passed = 0
    for name, check in checks:
        counter += 1
        print(f"  {counter} {name:.<36} ", end="", flush=True)
        try:
            detail = check()
            passed += 1
            print("OK" if detail is None else f"OK ({detail})", flush=True)
        except NoDevice as e:
            if require_gpu:
                failed += 1
                print(f"FAILED ({e}; --require-gpu)", flush=True)
            else:
                skipped += 1
                print(f"SKIPPED ({e})", flush=True)
        except Skipped as e:
            skipped += 1
            print(f"SKIPPED ({e})", flush=True)
        except Exception as e:
            failed += 1
            print(f"FAILED ({type(e).__name__}: {e})", flush=True)
    return counter, passed, failed, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m nearl.valid_installation",
        description="Check a Nearl installation; the static checks need no GPU.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="treat skipped GPU checks as failures (for CI on a GPU node)",
    )
    args = parser.parse_args(argv)

    try:
        import nearl
    except ImportError as e:
        print(f"ImportError: {e}", file=sys.stderr)
        print("Please check if the package is installed correctly.", file=sys.stderr)
        return 1

    print(f"Nearl version {nearl.__version__}")
    counter = passed = failed = skipped = 0

    # --require-gpu escalates only NoDevice; a skip for want of the CUDA toolkit
    # is unrelated to having a GPU and stays a skip in both groups.
    print("\nStatic checks (no GPU required)")
    counter, p, f, s = _run(STATIC_CHECKS, counter, args.require_gpu)
    passed, failed, skipped = passed + p, failed + f, skipped + s

    print("\nRuntime checks (CUDA device required)")
    counter, p, f, s = _run(RUNTIME_CHECKS, counter, args.require_gpu)
    passed, failed, skipped = passed + p, failed + f, skipped + s

    print()
    if failed:
        print(
            f"Installation validation failed: {failed} of {counter} checks failed.",
            file=sys.stderr,
        )
        return 1
    if skipped:
        print(
            f"Installation validation successful: {passed} checks passed, "
            f"{skipped} skipped (see the SKIPPED lines above)."
        )
    else:
        print(f"Installation validation successful: {passed} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
