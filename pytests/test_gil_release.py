"""
Checks that the CUDA entry points hand the GIL back while a kernel runs.

:class:`nearl.featurizer.Featurizer` overlaps CPU preprocessing with GPU compute
by running the producer on a background thread. That only works if the extension
releases the GIL for the duration of the kernel call: a C extension that holds it
cannot be preempted, because the interpreter can only switch threads between
bytecodes. Without the release the producer thread is frozen for every kernel
launch and the pipeline degenerates to the serial schedule.

Nothing else in the suite notices the difference -- the results are identical
either way, only the wall clock changes -- so it is checked here directly.

The measurement compares a spinner thread's progress against two controls, one
that provably never releases the GIL and one that provably does. Absolute rates
vary with the machine, so only the ordering against those controls is asserted.
"""

import sys
import threading
import time

import numpy as np
import pytest


def _extension_built():
    try:
        from nearl import all_actions  # noqa: F401
    except ImportError:
        return False
    return True


requires_extension = pytest.mark.skipif(
    not _extension_built(),
    reason="nearl.all_actions is not built (no nvcc in this environment)",
)


class Spinner:
    """A background thread running pure Python, so it needs the GIL to progress."""

    def __init__(self):
        self._stop = threading.Event()
        self.count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.count += 1

    def __enter__(self):
        self._thread.start()
        time.sleep(0.05)  # let it reach a steady rate
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def rate(self, call, seconds):
        """Iterations per second achieved while `call` is looping on this thread."""
        start, t0 = self.count, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            call()
        elapsed = time.perf_counter() - t0
        return (self.count - start) / elapsed


def _sized_to(build, target, lo, hi):
    """Binary-search a workload size that takes about `target` seconds."""
    call = build(hi)
    for _ in range(24):
        mid = (lo + hi) // 2
        call = build(mid)
        t0 = time.perf_counter()
        call()
        dt = time.perf_counter() - t0
        if dt < target:
            lo = mid
        else:
            hi = mid
        if abs(dt - target) / target < 0.08:
            break
    return call


@requires_extension
def test_a_kernel_launch_releases_the_gil():
    from nearl import all_actions

    coords = np.random.rand(10, 1200, 3).astype(np.float32) * 16.0
    weights = np.random.rand(10 * 1200).astype(np.float32)
    grid = np.array([32] * 3, dtype=int)

    def kernel():
        return all_actions.density_flow(coords, weights, grid, 0.5, 3.5, 1.5, 1)

    try:
        kernel()  # first call builds the CUDA context
    except Exception as exc:  # no usable device on this runner
        pytest.skip(f"no CUDA device available: {exc}")

    t0 = time.perf_counter()
    kernel()
    target = time.perf_counter() - t0

    # Controls sized to the same duration as the kernel call. str.count is a
    # single pure-C call that never releases the GIL; a numpy BLAS matmul does.
    holds = _sized_to(
        lambda n: lambda s="a" * n: s.count("b"), target, 1_000_000, 900_000_000
    )
    releases = _sized_to(
        lambda n: lambda a=np.random.rand(n, n): a @ a, target, 200, 3000
    )

    # A waiting thread keeps the GIL for one switch interval each time it wakes,
    # so at the default 5 ms a call that never releases still looks ~15% free
    # over a 30 ms call. Shrink the interval so the controls actually separate.
    original = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with Spinner() as spinner:
            free = spinner.rate(lambda: time.sleep(target), 0.4)
            held_rate = spinner.rate(holds, 0.4)
            released_rate = spinner.rate(releases, 0.4)
            kernel_rate = spinner.rate(kernel, 0.4)
    finally:
        sys.setswitchinterval(original)

    held = 100 * held_rate / free
    released = 100 * released_rate / free
    measured = 100 * kernel_rate / free
    report = (
        f"GIL left to a background thread: kernel {measured:.1f}%, "
        f"never-releases control {held:.1f}%, releases control {released:.1f}%"
    )

    # Guard the controls first: if they do not separate, the machine is too
    # loaded for the measurement to mean anything.
    assert released - held > 20, f"controls failed to separate -- {report}"
    assert measured > held + 0.5 * (released - held), (
        f"This build of the extension holds the GIL for the whole kernel call, "
        f"so the featurizer's producer thread cannot run while a kernel is in "
        f"flight.\n{report}\nExtension under test: {all_actions.__file__}\n"
        f"If that path is not the one you just built, the test is measuring a "
        f"stale copy: nearl is importable both from the source tree and from "
        f"site-packages, and `pytest` (unlike `python -m pytest`) does not put "
        f"the working directory first. Otherwise, wrap the *_host call in "
        f"py::gil_scoped_release (src/actions_py.cpp)."
    )
