"""Checks for the installation validator, in particular the GPU-less paths."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from nearl import valid_installation as V


def test_a_cuda_failure_raises_instead_of_returning_zeros():
    """
    Regression test for CUDA_CHECK in src/. Without it, a CUDA call that fails
    is silent: the kernels never run and the caller gets its zero-initialised
    buffer back as though it were a result.

    The device has to be hidden before the process starts, so this runs in a
    subprocess with CUDA_VISIBLE_DEVICES emptied.
    """
    script = textwrap.dedent("""
        import numpy as np
        from nearl import commands
        try:
            commands.frame_voxelize(
                np.random.normal(size=(10, 3)), np.full(10, 1.0),
                np.array([32, 32, 32]), 0.5, 5, 2,
            )
        except RuntimeError as e:
            print("RAISED:", e)
        else:
            print("SILENT")
    """)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert "SILENT" not in result.stdout, (
        "a failed CUDA call returned quietly instead of raising:\n" + result.stdout
    )
    assert "RAISED:" in result.stdout, result.stdout + result.stderr
    assert "cudaErrorNoDevice" in result.stdout
    # The message has to say where it happened.
    assert ".cu:" in result.stdout and "cudaMalloc" in result.stdout


def _arch_source(sass, ptx):
    return lambda: {"sass": set(sass), "ptx": set(ptx)}


@pytest.mark.parametrize(
    "arch,expected",
    [("sm_86", (8, 6)), ("sm_80", (8, 0)), ("sm_90", (9, 0)), ("sm_90a", (9, 0))],
)
def test_parse_arch(arch, expected):
    assert V._parse_arch(arch) == expected


def test_extension_symbols_are_all_exported():
    assert V._check_extension() == f"{len(V.EXPECTED_SYMBOLS)} symbols"


def test_extension_check_reports_a_missing_symbol(monkeypatch):
    from nearl import all_actions

    monkeypatch.delattr(all_actions, "frame_voxelize", raising=True)
    with pytest.raises(ValueError, match="missing exported symbols: frame_voxelize"):
        V._check_extension()


def test_native_sass_is_accepted(monkeypatch):
    monkeypatch.setattr(V, "_device_count", lambda: 1)
    monkeypatch.setattr(V, "_device_arch", lambda: "sm_86")
    monkeypatch.setattr(V, "_embedded_archs", _arch_source(["sm_86"], ["sm_86"]))
    assert "native SASS" in V._check_arch_compatibility()


def test_lower_minor_of_the_same_major_is_binary_compatible(monkeypatch):
    """A cubin runs on the same major version with an equal or higher minor."""
    monkeypatch.setattr(V, "_device_count", lambda: 1)
    monkeypatch.setattr(V, "_device_arch", lambda: "sm_86")
    monkeypatch.setattr(V, "_embedded_archs", _arch_source(["sm_80"], ["sm_80"]))
    assert "runs the sm_80 SASS" in V._check_arch_compatibility()


def test_no_usable_sass_falls_back_to_ptx_with_a_rebuild_hint(monkeypatch):
    monkeypatch.setattr(V, "_device_count", lambda: 1)
    monkeypatch.setattr(V, "_device_arch", lambda: "sm_90")
    monkeypatch.setattr(V, "_embedded_archs", _arch_source(["sm_86"], ["sm_86"]))
    message = V._check_arch_compatibility()
    assert "JIT from PTX" in message
    assert "CUDA_COMPUTE_CAPABILITY=sm_90" in message


def test_newer_arch_than_the_device_is_rejected(monkeypatch):
    """The sm_90-build-on-an-sm_86-box mistake has to be caught."""
    monkeypatch.setattr(V, "_device_count", lambda: 1)
    monkeypatch.setattr(V, "_device_arch", lambda: "sm_86")
    monkeypatch.setattr(V, "_embedded_archs", _arch_source(["sm_90"], ["sm_90"]))
    with pytest.raises(ValueError, match="cannot run on this sm_86 device"):
        V._check_arch_compatibility()


@pytest.mark.parametrize(
    "check", [V._check_device, V._check_arch_compatibility, V._check_voxelization]
)
def test_runtime_checks_skip_rather_than_pass_without_a_device(monkeypatch, check):
    monkeypatch.setattr(V, "_device_count", lambda: 0)
    with pytest.raises(V.Skipped):
        check()


def test_empty_voxel_grid_is_a_failure_not_a_pass(monkeypatch):
    """
    The CUDA sources check no error codes, so a launch that never ran returns a
    zero-filled buffer. Asserting only shape/finiteness would pass on a machine
    with no GPU.
    """
    from nearl import commands

    monkeypatch.setattr(V, "_device_count", lambda: 1)
    monkeypatch.setattr(
        commands, "frame_voxelize", lambda *a, **k: np.zeros((32, 32, 32), np.float32)
    )
    with pytest.raises(ValueError, match="kernels did not run"):
        V._check_voxelization()


def test_require_gpu_escalates_a_missing_device_but_not_a_missing_toolkit():
    """
    --require-gpu means "a GPU must be usable here", so it must not fail a check
    that skipped only because cuobjdump (the CUDA toolkit) is absent.
    """

    def no_device():
        raise V.NoDevice("no CUDA device visible")

    def no_toolkit():
        raise V.Skipped("cuobjdump not on PATH")

    _, passed, failed, skipped = V._run(
        [("device", no_device), ("toolkit", no_toolkit)], 0, require_gpu=True
    )
    assert (passed, failed, skipped) == (0, 1, 1)


def test_static_checks_do_not_touch_the_device(monkeypatch):
    """Everything under Static checks must work on a GPU-less machine."""

    def explode():
        raise AssertionError("a static check queried the CUDA device")

    monkeypatch.setattr(V, "_device_count", explode)
    monkeypatch.setattr(V, "_device_arch", explode)
    for _name, check in V.STATIC_CHECKS:
        check()
