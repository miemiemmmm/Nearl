"""Unit tests for the DLPack interop of the ``*_dlpack`` commands.

Each command has two destinations and both are covered here:

  out=None   allocate a ``DeviceArray`` and hand it back; the caller adopts it
             with ``torch.from_dlpack`` (or cupy, or jax -- nothing here is
             torch-specific, which ``test_importing_commands_does_not_pull_in_torch``
             pins down).
  out=<obj>  write into a caller-owned CUDA buffer, reached through that
             object's ``__dlpack__``.

The rejection tests matter as much as the equality ones. This path used to take
a bare integer pointer, so a CPU tensor, a float64 tensor, a strided view or a
grid shape that disagreed with ``grid_dims`` all produced an out-of-bounds
device write instead of an exception.

    python -m pytest pytests/test_dlpack_paths.py -v
"""

import gc
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nearl import commands

if not torch.cuda.is_available():
    pytest.skip("no CUDA device visible", allow_module_level=True)

ATOM_NR = 100
FRAME_NR = 20
DIMS = (16, 16, 16)
SPACING, CUTOFF, SIGMA = 0.5, 2.5, 1.0
NUMEL = DIMS[0] * DIMS[1] * DIMS[2]

_rng = np.random.default_rng(0)
COORDS = _rng.normal(size=(ATOM_NR, 3), loc=5, scale=2).astype(np.float32)
WEIGHTS_FRAME = np.full((ATOM_NR,), 16.0, dtype=np.float32)
TRAJ = _rng.normal(size=(FRAME_NR, ATOM_NR, 3), loc=5, scale=2).astype(np.float32)
WEIGHTS_TRAJ = np.full((FRAME_NR * ATOM_NR,), 16.0, dtype=np.float32)


# (id, numpy-returning reference, dlpack command taking **kwargs)
CASES = [
    (
        "voxelize",
        lambda: commands.frame_voxelize(
            COORDS, WEIGHTS_FRAME, DIMS, SPACING, CUTOFF, SIGMA
        ),
        lambda **kw: commands.frame_voxelize_dlpack(
            COORDS, WEIGHTS_FRAME, DIMS, SPACING, CUTOFF, SIGMA, **kw
        ),
    ),
    (
        "observation",
        lambda: commands.frame_observation(
            COORDS, WEIGHTS_FRAME, DIMS, SPACING, CUTOFF, 1
        ),
        lambda **kw: commands.frame_observation_dlpack(
            COORDS, WEIGHTS_FRAME, DIMS, SPACING, CUTOFF, 1, **kw
        ),
    ),
    (
        "observer",
        lambda: commands.marching_observer(
            TRAJ, WEIGHTS_TRAJ, DIMS, SPACING, CUTOFF, 1, 1
        ),
        lambda **kw: commands.marching_observer_dlpack(
            TRAJ, WEIGHTS_TRAJ, DIMS, SPACING, CUTOFF, 1, 1, **kw
        ),
    ),
    (
        "flow",
        lambda: commands.density_flow(
            TRAJ, WEIGHTS_TRAJ, DIMS, SPACING, CUTOFF, SIGMA, 1
        ),
        lambda **kw: commands.density_flow_dlpack(
            TRAJ, WEIGHTS_TRAJ, DIMS, SPACING, CUTOFF, SIGMA, 1, **kw
        ),
    ),
]
IDS = [case[0] for case in CASES]


@pytest.fixture()
def device_context():
    commands.init_context()
    assert commands.context_valid()
    yield
    if commands.context_valid():
        commands.finalize_context()


@pytest.fixture(params=CASES, ids=IDS)
def case(request):
    """(reference, dlpack) callables for one command."""
    _, reference, dlpack = request.param
    return reference, dlpack


def assert_matches(reference, actual):
    values = actual.cpu().numpy() if hasattr(actual, "cpu") else np.asarray(actual)
    assert np.isfinite(values).all()
    assert np.allclose(reference, values, rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------------ values --


def test_the_exported_grid_matches_the_numpy_path(case, device_context):
    reference, dlpack = case
    grid = dlpack()
    assert isinstance(grid, commands.DeviceArray)
    tensor = torch.from_dlpack(grid)
    assert tensor.device.type == "cuda"
    assert tensor.dtype == torch.float32
    assert tuple(tensor.shape) == DIMS
    assert_matches(reference(), tensor)


def test_writing_into_a_caller_buffer_matches_the_numpy_path(case, device_context):
    reference, dlpack = case
    buffer = torch.empty(DIMS, dtype=torch.float32, device="cuda")
    returned = dlpack(out=buffer)
    assert returned is buffer, "out= should hand back the caller's own object"
    assert_matches(reference(), buffer)


def test_a_flat_buffer_of_the_right_size_is_accepted(case, device_context):
    """The grid is contiguous, so a 1-D destination is the same memory."""
    reference, dlpack = case
    buffer = torch.empty(NUMEL, dtype=torch.float32, device="cuda")
    dlpack(out=buffer)
    assert_matches(reference().reshape(-1), buffer)


def test_writing_a_batch_slice_leaves_its_neighbours_alone(device_context):
    _, reference, dlpack = CASES[3]
    batch = torch.zeros((3, *DIMS), dtype=torch.float32, device="cuda")
    dlpack(out=batch[1])
    assert_matches(reference(), batch[1])
    assert batch[0].abs().sum().item() == 0.0
    assert batch[2].abs().sum().item() == 0.0


def test_the_grid_trains_a_cuda_model(device_context):
    _, _, dlpack = CASES[3]
    tensor = torch.from_dlpack(dlpack())
    net = torch.nn.Conv3d(1, 2, 3, padding=1).cuda()
    loss = net(tensor.unsqueeze(0).unsqueeze(0)).sum()
    loss.backward()
    assert torch.isfinite(loss).item()


# --------------------------------------------------------------- ownership --


def test_the_grid_outlives_its_producer_and_the_context(device_context):
    """The exported buffer is its own allocation, not a context slot.

    If it came from the reusable device buffers, the two calls below would
    overwrite it and finalize_context() would free it underneath the tensor.
    """
    _, reference, dlpack = CASES[3]
    expected = reference()

    grid = dlpack()
    tensor = torch.from_dlpack(grid)
    del grid
    dlpack()
    dlpack()
    commands.finalize_context()
    assert not commands.context_valid()

    assert_matches(expected, tensor)


def test_two_exports_share_one_allocation(device_context):
    _, _reference, dlpack = CASES[3]
    grid = dlpack()
    first = torch.from_dlpack(grid)
    second = torch.from_dlpack(grid)
    assert first.data_ptr() == second.data_ptr()
    first.fill_(3.0)
    assert second.eq(3.0).all().item()


# ----------------------------------------------------------------- capsule --


@pytest.mark.parametrize(
    ("kwargs", "expected_name"),
    [
        ({"max_version": (1, 0)}, "dltensor_versioned"),
        ({}, "dltensor"),
        ({"max_version": (0, 8)}, "dltensor"),
    ],
)
def test_the_requested_capsule_version_is_honoured(
    kwargs, expected_name, device_context
):
    _, _, dlpack = CASES[3]
    capsule = dlpack().__dlpack__(**kwargs)
    assert expected_name in repr(capsule)
    assert tuple(torch.from_dlpack(capsule).shape) == DIMS


def test_a_capsule_cannot_be_consumed_twice(device_context):
    _, _, dlpack = CASES[3]
    capsule = dlpack().__dlpack__(max_version=(1, 0))
    torch.from_dlpack(capsule)
    with pytest.raises(RuntimeError):
        torch.from_dlpack(capsule)


def test_dlpack_device_reports_this_cuda_device(device_context):
    _, _, dlpack = CASES[3]
    kDLCUDA = 2
    assert dlpack().__dlpack_device__() == (kDLCUDA, torch.cuda.current_device())


@pytest.mark.parametrize(
    "kwargs",
    [{"copy": True}, {"dl_device": (1, 0)}],
    ids=["copy", "other-device"],
)
def test_unsupported_export_requests_are_refused(kwargs, device_context):
    _, _, dlpack = CASES[3]
    with pytest.raises(BufferError):
        dlpack().__dlpack__(**kwargs)


# -------------------------------------------------------------- rejections --


def bad_buffers():
    return {
        "cpu-tensor": (torch.empty(DIMS, dtype=torch.float32), ValueError),
        "float64": (torch.empty(DIMS, dtype=torch.float64, device="cuda"), ValueError),
        "float16": (torch.empty(DIMS, dtype=torch.float16, device="cuda"), ValueError),
        "numpy-array": (np.empty(DIMS, dtype=np.float32), ValueError),
        "too-small": (
            torch.empty((8, 16, 16), dtype=torch.float32, device="cuda"),
            ValueError,
        ),
        "too-large": (
            torch.empty((32, 16, 16), dtype=torch.float32, device="cuda"),
            ValueError,
        ),
        "reshaped": (
            torch.empty((8, 32, 16), dtype=torch.float32, device="cuda"),
            ValueError,
        ),
        "strided-view": (
            torch.empty((16, 16, 32), dtype=torch.float32, device="cuda")[:, :, ::2],
            ValueError,
        ),
        "transposed": (
            torch.empty((16, 16, 32), dtype=torch.float32, device="cuda")
            .narrow(2, 0, 16)
            .transpose(0, 2),
            ValueError,
        ),
        "raw-pointer": (12345, TypeError),
        "none-like-object": (object(), TypeError),
    }


@pytest.mark.parametrize("name", list(bad_buffers()))
def test_an_unusable_out_buffer_raises_instead_of_corrupting_memory(
    name, device_context
):
    buffer, expected = bad_buffers()[name]
    _, _, dlpack = CASES[3]
    with pytest.raises(expected):
        dlpack(out=buffer)


def test_grid_dims_must_hold_three_entries(device_context):
    with pytest.raises(ValueError):
        commands.density_flow_dlpack(
            TRAJ, WEIGHTS_TRAJ, (16, 16), SPACING, CUTOFF, SIGMA, 1
        )


def test_the_context_still_works_after_a_rejected_buffer(device_context):
    """A rejection must not leave the CUDA context or the DLPack import wedged."""
    _, reference, dlpack = CASES[3]
    with pytest.raises(ValueError):
        dlpack(out=torch.empty(DIMS, dtype=torch.float64, device="cuda"))
    assert_matches(reference(), torch.from_dlpack(dlpack()))


# ------------------------------------------------------- framework-neutral --


def test_importing_commands_does_not_pull_in_torch():
    """The whole point of DLPack here: the consumer picks the framework."""
    code = "import sys; import nearl.commands; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cupy_can_adopt_the_grid(device_context):
    cupy = pytest.importorskip("cupy")
    _, reference, dlpack = CASES[3]
    array = cupy.from_dlpack(dlpack())
    assert array.dtype == cupy.float32
    assert array.shape == DIMS
    assert np.allclose(reference(), cupy.asnumpy(array), rtol=1e-4, atol=1e-4)


def test_a_cupy_buffer_can_be_written_into(device_context):
    cupy = pytest.importorskip("cupy")
    _, reference, dlpack = CASES[3]
    buffer = cupy.empty(DIMS, dtype=cupy.float32)
    dlpack(out=buffer)
    assert np.allclose(reference(), cupy.asnumpy(buffer), rtol=1e-4, atol=1e-4)


# ------------------------------------------------------- protocol fallback --


class LegacyProducer:
    """A pre-1.0 DLPack producer: no ``max_version``, legacy capsule only.

    torch always takes the versioned path, so nothing else here reaches the
    fallback in ``call_dlpack`` or the ``DLManagedTensor`` branch of the import.
    """

    def __init__(self, tensor):
        self.tensor = tensor
        self.streams = []

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, stream=None):
        self.streams.append(stream)
        return self.tensor.__dlpack__()


class StreamlessProducer(LegacyProducer):
    """Older still: rejects ``stream`` as well."""

    def __dlpack__(self):
        return self.tensor.__dlpack__()


@pytest.mark.parametrize("producer_type", [LegacyProducer, StreamlessProducer])
def test_a_pre_v1_producer_is_still_accepted(producer_type, device_context):
    _, reference, dlpack = CASES[3]
    buffer = torch.empty(DIMS, dtype=torch.float32, device="cuda")
    producer = producer_type(buffer)
    dlpack(out=producer)
    assert_matches(reference(), buffer)


def test_the_consumer_stream_is_offered_to_the_producer(device_context):
    """DLPack's ordering guarantee: the producer is told which stream we use."""
    _, _, dlpack = CASES[3]
    producer = LegacyProducer(torch.empty(DIMS, dtype=torch.float32, device="cuda"))
    dlpack(out=producer)
    assert len(producer.streams) == 1
    stream = producer.streams[0]
    assert isinstance(stream, int) and stream > 0, (
        "a CUDA consumer must pass a stream token, not None"
    )


@pytest.mark.parametrize("wrap", [lambda t: t, LegacyProducer], ids=["v1", "legacy"])
def test_importing_a_buffer_does_not_leak_its_storage(wrap, device_context):
    """Consuming a capsule means owning it: the deleter has to run.

    torch's deleter holds an ATen tensor handle rather than a Python reference,
    so a missing deleter call is invisible to refcounts -- it shows up as
    storage the caching allocator can never reclaim.
    """
    _, _, dlpack = CASES[3]

    def one_round():
        buffer = torch.empty(DIMS, dtype=torch.float32, device="cuda")
        dlpack(out=wrap(buffer))
        del buffer

    for _ in range(5):
        one_round()  # settle the allocator before taking a baseline
    gc.collect()
    before = torch.cuda.memory_allocated()
    for _ in range(200):
        one_round()
    gc.collect()
    assert torch.cuda.memory_allocated() == before
