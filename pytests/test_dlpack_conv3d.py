"""Can a DLPack-adopted grid be fed straight into a 3D CNN?

test_dlpack_paths.py checks the protocol -- capsules, ownership, rejection.
This file checks the thing the protocol exists for: that the tensor coming out
the other side is a first-class input to torch's Conv3d and to the models in
nearl.models, with no copy, no reshape and no detour through the host.

The whole module skips when torch or a CUDA device is missing.

    python -m pytest pytests/test_dlpack_conv3d.py -v
"""

import gc

import numpy as np
import pytest

try:
    import torch
except ImportError as exc:  # absent, or installed but unusable
    pytest.skip(f"torch is unavailable: {exc}", allow_module_level=True)

from nearl import commands

if not torch.cuda.is_available():
    pytest.skip("no CUDA device visible", allow_module_level=True)

nn = torch.nn

ATOM_NR = 120
FRAME_NR = 16
DIM = 16
DIMS = (DIM, DIM, DIM)
SPACING, CUTOFF, SIGMA = 0.5, 2.5, 1.0

_rng = np.random.default_rng(7)
TRAJ = _rng.normal(size=(FRAME_NR, ATOM_NR, 3), loc=5, scale=2).astype(np.float32)
WEIGHTS = np.full((FRAME_NR * ATOM_NR,), 12.0, dtype=np.float32)


@pytest.fixture()
def device_context():
    commands.init_context()
    yield
    if commands.context_valid():
        commands.finalize_context()


def flow(**kw):
    return commands.density_flow_dlpack(
        TRAJ, WEIGHTS, DIMS, SPACING, CUTOFF, SIGMA, 1, **kw
    )


def observers(**kw):
    return commands.marching_observer_dlpack(
        TRAJ, WEIGHTS, DIMS, SPACING, CUTOFF, 1, 1, **kw
    )


def flow_reference():
    return commands.density_flow(TRAJ, WEIGHTS, DIMS, SPACING, CUTOFF, SIGMA, 1)


def observers_reference():
    return commands.marching_observer(TRAJ, WEIGHTS, DIMS, SPACING, CUTOFF, 1, 1)


# ------------------------------------------------------------ preconditions --


def test_an_adopted_grid_satisfies_what_conv3d_needs(device_context):
    grid = torch.from_dlpack(flow())
    assert grid.dtype == torch.float32, "Conv3d weights default to float32"
    assert grid.device.type == "cuda"
    assert grid.device.index == torch.cuda.current_device()
    assert grid.is_contiguous(), "cudnn wants a contiguous input"
    assert tuple(grid.shape) == DIMS
    assert torch.isfinite(grid).all()


def test_adding_the_batch_and_channel_axes_does_not_copy(device_context):
    """(D,D,D) -> (1,1,D,D,D) has to stay a view, or 'zero-copy' is a lie."""
    grid = torch.from_dlpack(flow())
    batched = grid.unsqueeze(0).unsqueeze(0)
    assert batched.data_ptr() == grid.data_ptr()
    assert batched.is_contiguous()

    conv = nn.Conv3d(1, 4, 3, padding=1).cuda()
    out = conv(batched)
    assert tuple(out.shape) == (1, 4, DIM, DIM, DIM)
    assert torch.isfinite(out).all()


def test_conv3d_sees_the_same_values_as_a_host_copy(device_context):
    """Guards against reading the buffer before the kernel has finished."""
    grid = torch.from_dlpack(flow())
    conv = nn.Conv3d(1, 2, 3, padding=1).cuda()  # one instance, so weights match

    from_dlpack_input = conv(grid.unsqueeze(0).unsqueeze(0))
    via_host = conv(torch.from_numpy(flow_reference()).cuda().unsqueeze(0).unsqueeze(0))
    assert torch.allclose(from_dlpack_input, via_host, rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------------ autograd --


def test_gradients_flow_back_into_an_adopted_grid(device_context):
    grid = torch.from_dlpack(flow()).unsqueeze(0).unsqueeze(0)
    grid.requires_grad_(True)
    conv = nn.Conv3d(1, 3, 3, padding=1).cuda()
    conv(grid).pow(2).sum().backward()

    assert grid.grad is not None
    assert grid.grad.shape == grid.shape
    assert torch.isfinite(grid.grad).all()
    assert grid.grad.abs().sum().item() > 0, "a zero gradient would hide a dead input"


def test_the_grid_survives_a_backward_pass_after_its_producer_is_dropped(
    device_context,
):
    """The autograd graph holds the tensor long after the DeviceArray is gone."""
    handle = flow()
    grid = torch.from_dlpack(handle).unsqueeze(0).unsqueeze(0)
    grid.requires_grad_(True)
    conv = nn.Conv3d(1, 2, 3, padding=1).cuda()
    loss = conv(grid).sum()

    del handle
    gc.collect()
    flow()  # further device work must not disturb the exported buffer

    loss.backward()
    assert torch.isfinite(loss).item()
    assert torch.isfinite(grid.grad).all()


# ------------------------------------------------------------------ batching --


def test_stacked_grids_form_a_batch(device_context):
    grids = [torch.from_dlpack(flow()) for _ in range(3)]
    batch = torch.stack(grids).unsqueeze(1)
    assert tuple(batch.shape) == (3, 1, *DIMS)

    conv = nn.Conv3d(1, 2, 3, padding=1).cuda()
    assert torch.isfinite(conv(batch)).all()


def test_out_fills_a_preallocated_batch_that_conv3d_then_consumes(device_context):
    """The production shape: (B, C, D, D, D) filled in place, one slice per call.

    Nothing is allocated per sample and nothing is copied -- the conv reads the
    same memory the CUDA kernels wrote.
    """
    batch_size, channels = 3, 2
    batch = torch.zeros(
        (batch_size, channels, *DIMS), dtype=torch.float32, device="cuda"
    )
    for b in range(batch_size):
        flow(out=batch[b, 0])
        observers(out=batch[b, 1])

    expected = [flow_reference(), observers_reference()]
    for b in range(batch_size):
        for c, reference in enumerate(expected):
            assert np.allclose(
                reference, batch[b, c].cpu().numpy(), rtol=1e-4, atol=1e-4
            ), f"sample {b} channel {c} does not hold the feature it was given"

    conv = nn.Conv3d(channels, 4, 3, padding=1).cuda()
    out = conv(batch)
    assert tuple(out.shape) == (batch_size, 4, *DIMS)
    assert torch.isfinite(out).all()


def test_writing_one_slice_leaves_the_other_samples_untouched(device_context):
    batch = torch.zeros((4, 1, *DIMS), dtype=torch.float32, device="cuda")
    flow(out=batch[2, 0])
    assert batch[2, 0].abs().sum().item() > 0
    for b in (0, 1, 3):
        assert batch[b].abs().sum().item() == 0, f"sample {b} was written too"


# -------------------------------------------------------------- real models --


def test_a_nearl_model_trains_on_grids_it_never_copied(device_context):
    """End to end on a model from nearl.models, two features as two channels."""
    voxnet = pytest.importorskip("nearl.models.model_voxnet")

    batch = torch.zeros((2, 2, *DIMS), dtype=torch.float32, device="cuda")
    for b in range(2):
        flow(out=batch[b, 0])
        observers(out=batch[b, 1])

    model = voxnet.VoxNet(input_channels=2, output_dimension=1, input_shape=DIM).cuda()
    prediction = model(batch)
    assert prediction.shape[0] == 2
    assert torch.isfinite(prediction).all()

    loss = prediction.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
