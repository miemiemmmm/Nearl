"""Unit tests for the ``*_dlpack`` GPU-output commands.

The three feature families are separate parametrized cases selected by an
index, so each function can be tested on its own:

    python -m pytest pytests/test_dlpack_paths.py -v                # all three
    python -m pytest "pytests/test_dlpack_paths.py::test_dlpack_path[flow]"
    python pytests/test_dlpack_paths.py 2                            # flow only

Each case checks that the dlpack path returns a CUDA tensor whose values
match the numpy-return path and that the tensor is consumable by a CUDA
model (forward and backward).
"""

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nearl import commands

if not torch.cuda.is_available():
    pytest.skip("no CUDA device visible", allow_module_level=True)

atom_nr = 100
frame_nr = 20
dims = np.array([16, 16, 16], dtype=np.int32)
spacing = 0.5
cutoff = 2.5
sigma = 1.0


def _make_cases():
    np.random.seed(0)
    coords = np.random.normal(size=(atom_nr, 3), loc=5, scale=2).astype(np.float32)
    weights_frame = np.full((atom_nr,), 16.0, dtype=np.float32)
    traj = np.random.normal(size=(frame_nr, atom_nr, 3), loc=5, scale=2).astype(
        np.float32
    )
    weights_traj = np.full((frame_nr * atom_nr,), 16.0, dtype=np.float32)
    return [
        (
            "voxelize",
            lambda: commands.frame_voxelize(
                coords, weights_frame, dims, spacing, cutoff, sigma
            ),
            lambda: commands.frame_voxelize_dlpack(
                coords, weights_frame, dims, spacing, cutoff, sigma
            ),
        ),
        (
            "observer",
            lambda: commands.marching_observer(
                traj, weights_traj, dims, spacing, cutoff, 1, 1
            ),
            lambda: commands.marching_observer_dlpack(
                traj, weights_traj, dims, spacing, cutoff, 1, 1
            ),
        ),
        (
            "flow",
            lambda: commands.density_flow(
                traj, weights_traj, dims, spacing, cutoff, sigma, 1
            ),
            lambda: commands.density_flow_dlpack(
                traj, weights_traj, dims, spacing, cutoff, sigma, 1
            ),
        ),
    ]


CASES = _make_cases()


def _check_case(idx):
    name, np_fn, dl_fn = CASES[idx]
    reference = np_fn()
    output = dl_fn()

    assert output.device.type == "cuda", "dlpack path must produce a CUDA tensor"
    assert output.dtype == torch.float32
    assert tuple(output.shape) == tuple(dims)
    assert np.allclose(reference, output.cpu().numpy(), rtol=1e-4, atol=1e-4), (
        f"{name}: dlpack output differs from the numpy-return path"
    )
    assert np.isfinite(output.cpu().numpy()).all()
    output.clone().cpu()

    net = torch.nn.Conv3d(1, 2, 3, padding=1).cuda()
    loss = net(output.unsqueeze(0).unsqueeze(0)).sum()
    loss.backward()
    assert torch.isfinite(loss).item()
    return name


@pytest.fixture()
def device_context():
    commands.init_context()
    assert commands.context_valid()
    yield
    commands.finalize_context()


@pytest.mark.parametrize(
    "idx",
    [
        pytest.param(0, id="voxelize"),
        pytest.param(1, id="observer"),
        pytest.param(2, id="flow"),
    ],
)
def test_dlpack_path(idx, device_context):
    _check_case(idx)


if __name__ == "__main__":
    index = int(sys.argv[1]) if len(sys.argv) > 1 else None
    indices = [index] if index is not None else range(len(CASES))
    commands.init_context()
    try:
        for i in indices:
            print(f"case {i} ({_check_case(i)}): OK")
    finally:
        commands.finalize_context()
