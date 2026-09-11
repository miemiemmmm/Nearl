"""Frame-dimension aggregation: correct, and no longer capped at 512 frames.

The aggregation kernel used to stage each grid point's time series into a
``float tmp_array[MAX_FRAME_NUMBER]`` per thread. That array was the frame
limit: past it, ``marching_observer`` raised and ``density_flow`` silently
aggregated only the first 512 frames. The kernel now reads the series straight
out of the trajectory buffer, so the only remaining bound is the CUDA grid
(frames map to ``blockIdx.y``).

    python -m pytest pytests/test_frame_aggregation.py -v
"""

import numpy as np
import pytest

from nearl import constants

all_actions = pytest.importorskip("nearl.all_actions")
from nearl import commands

BINS = 16
GRID = np.array([8, 8, 8], dtype=np.int32)
NPOINTS = 256


def entropy_reference(col):
    lo, hi = col.min(), col.max()
    if hi == lo:
        return 0.0
    binned = np.clip(((col - lo) / (hi - lo) * BINS).astype(int), 0, BINS - 1)
    p = np.bincount(binned, minlength=BINS) / len(col)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def slope_reference(col):
    n = len(col)
    x = np.arange(n, dtype=np.float64)
    denom = n * (x * x).sum() - x.sum() ** 2
    if n <= 1 or denom == 0:
        return 0.0
    return float((n * (x * col).sum() - x.sum() * col.sum()) / denom)


REFERENCE = {
    1: ("mean", lambda c: c.mean()),
    2: ("stdev", lambda c: c.std()),
    3: ("median", lambda c: np.median(c)),
    4: ("variance", lambda c: c.var()),
    5: ("max", lambda c: c.max()),
    6: ("min", lambda c: c.min()),
    7: ("entropy", entropy_reference),
    8: ("slope", slope_reference),
}


@pytest.fixture(scope="module")
def device_context():
    try:
        commands.init_context()
    except Exception as exc:  # no usable device on this runner
        pytest.skip(f"no CUDA device available: {exc}")
    yield
    commands.finalize_context()


@pytest.mark.parametrize("agg", sorted(REFERENCE), ids=lambda a: REFERENCE[a][0])
@pytest.mark.parametrize("frames", [1, 2, 7, 64, 512, 513, 1000])
def test_aggregation_matches_numpy(agg, frames, device_context):
    """513 and 1000 are the cases the old per-thread array could not hold."""
    rng = np.random.default_rng(frames)
    series = rng.normal(3.0, 1.5, size=(frames, NPOINTS)).astype(np.float32)
    _, reference = REFERENCE[agg]

    # the kernel may sort its input in place, so hand it a copy
    got = np.asarray(all_actions.aggregate(series.copy(), agg), dtype=np.float64)
    expected = np.array(
        [reference(series[:, j].astype(np.float64)) for j in range(NPOINTS)]
    )
    scale = max(np.abs(expected).max(), 1e-6)
    assert np.abs(got - expected).max() / scale < 1e-4


def slice_inputs(frames, atoms=80):
    """A cloud that drifts with time, so late frames differ from early ones."""
    rng = np.random.default_rng(11)
    base = rng.uniform(0, 3, size=(1, atoms, 3)).astype(np.float32)
    drift = np.linspace(0, 1.5, frames, dtype=np.float32)[:, None, None]
    traj = (base + drift + rng.normal(0, 0.05, (frames, atoms, 3))).astype(np.float32)
    weights = np.abs(rng.normal(2.0, 0.3, frames * atoms)).astype(np.float32)
    return traj, weights


@pytest.mark.parametrize("frames", [600, 1024])
def test_density_flow_uses_every_frame(frames, device_context):
    """Aggregating on the host over per-frame grids is the independent check.

    Below, this returned the mean of only the first 512 frames -- silently.
    """
    traj, weights = slice_inputs(frames)
    atoms = traj.shape[1]
    fused = np.asarray(commands.density_flow(traj, weights, GRID, 0.5, 2.5, 1.0, 1))
    per_frame = np.stack(
        [
            np.asarray(
                commands.frame_voxelize(
                    traj[f], weights[f * atoms : (f + 1) * atoms], GRID, 0.5, 2.5, 1.0
                )
            )
            for f in range(frames)
        ]
    )
    assert np.allclose(fused.ravel(), per_frame.mean(0).ravel(), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("frames", [600, 1024])
def test_marching_observer_uses_every_frame(frames, device_context):
    traj, weights = slice_inputs(frames)
    atoms = traj.shape[1]
    fused = np.asarray(commands.marching_observer(traj, weights, GRID, 0.5, 2.5, 2, 1))
    per_frame = np.stack(
        [
            np.asarray(
                commands.frame_observation(
                    traj[f], weights[f * atoms : (f + 1) * atoms], GRID, 0.5, 2.5, 2
                )
            )
            for f in range(frames)
        ]
    )
    assert np.allclose(fused.ravel(), per_frame.mean(0).ravel(), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize(
    "call",
    ["density_flow", "marching_observer"],
)
def test_beyond_the_grid_limit_raises_rather_than_truncating(call, device_context):
    frames = constants.MAX_FRAME_NUMBER + 1
    traj = np.zeros((frames, 2, 3), dtype=np.float32)
    weights = np.ones(frames * 2, dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds the maximum"):
        if call == "density_flow":
            commands.density_flow(traj, weights, GRID, 1.0, 2.0, 1.0, 1)
        else:
            commands.marching_observer(traj, weights, GRID, 1.0, 2.0, 1, 1)


def test_the_python_and_cuda_limits_agree(device_context):
    frames = constants.MAX_FRAME_NUMBER + 1
    traj = np.zeros((frames, 2, 3), dtype=np.float32)
    weights = np.ones(frames * 2, dtype=np.float32)
    with pytest.raises(ValueError, match=str(constants.MAX_FRAME_NUMBER)):
        commands.density_flow(traj, weights, GRID, 1.0, 2.0, 1.0, 1)
