"""
Numerical checks for the CPU voxelization reference in :mod:`nearl._voxelize_cpu`.

The CPU path is both the fallback used where no CUDA extension exists and the
oracle the GPU kernels are checked against. Its arithmetic comes from
``src/voxelize_math.h``, which ``frame_interp_global`` also compiles, so the two
backends share one definition rather than two that must be kept in step.

They still cannot agree bitwise. The GPU sums the per-atom normalizer in a
shared-memory tree and scatters with ``atomicAdd`` in nondeterministic order,
while the CPU sums in a fixed order, so float rounding differs. The tolerances
below are measured, not guessed, and the two awkward aggregations are handled on
their own terms:

* **drift** is a small difference of larger numbers -- on the fixture the largest
  drift is ~70x smaller than the grid itself -- so a 5e-7 error on the grid shows
  up as ~6e-3 relative on the slope. It is checked against the grid's scale.
  Measured against a float64 NumPy oracle the CPU slope is accurate to 7e-10 and
  the GPU to 9e-7, so the CPU is the better of the two here.
* **information entropy** bins its input, and a bin edge is a step. A value a
  float-epsilon from an edge can fall either side, so a handful of voxels differ
  outright. What is checked is how few.
"""

import numpy as np
import pytest


def _cpu_available():
    try:
        from nearl import _voxelize_cpu  # noqa: F401
    except ImportError:
        return False
    return True


def _gpu_available():
    try:
        from nearl import all_actions  # noqa: F401
    except ImportError:
        return False
    return True


requires_cpu = pytest.mark.skipif(
    not _cpu_available(), reason="nearl._voxelize_cpu is not built"
)
requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="nearl.all_actions is not built (no nvcc in this environment)",
)

DIMS = np.array([32, 32, 32], dtype=int)
SPACING, CUTOFF, SIGMA = 0.5, 3.5, 1.5
PLACEHOLDER = 99999.0

# Headroom over the measured worst case, which is ~7e-7 for the grid itself.
RTOL = 1e-5
# Drift is ill-conditioned; bound it by the grid's own scale. Measured 8.3e-5.
DRIFT_SCALE_TOL = 1e-3
# Entropy bin edges are steps. Measured worst mismatch 0.0153% of voxels.
ENTROPY_MISMATCH_TOL = 1e-3

AGG_NAMES = {
    1: "mean",
    2: "stddev",
    3: "median",
    4: "variance",
    5: "max",
    6: "min",
    7: "entropy",
    8: "drift",
}
CONTINUOUS_AGGS = [1, 2, 3, 4, 5, 6]


def cpu():
    from nearl import _voxelize_cpu

    return _voxelize_cpu


def gpu():
    from nearl import all_actions

    return all_actions


def rel_diff(a, b):
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30))


def random_frame(n_atoms, seed, span=16.0):
    rng = np.random.default_rng(seed)
    coords = (rng.random((n_atoms, 3)) * span).astype(np.float32)
    weights = (rng.random(n_atoms) + 0.5).astype(np.float32)
    return coords, weights


def random_traj(n_frames, n_atoms, seed, span=16.0):
    rng = np.random.default_rng(seed)
    traj = (rng.random((n_frames, n_atoms, 3)) * span).astype(np.float32)
    weights = (rng.random(n_frames * n_atoms) + 0.5).astype(np.float32)
    return traj, weights


###############################################################################
# CPU alone: invariants that hold with or without a GPU in the machine
###############################################################################


@requires_cpu
def test_an_interior_atom_deposits_exactly_its_weight():
    """Each atom is normalized to its own weight, so the grid integrates to it."""
    coords = np.array([[8.0, 8.0, 8.0], [7.3, 9.1, 8.6]], dtype=np.float32)
    weights = np.array([2.0, 3.0], dtype=np.float32)
    grid = cpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)
    assert np.sum(grid) == pytest.approx(np.sum(weights), rel=1e-5)


@requires_cpu
def test_an_edge_atom_deposits_only_the_part_inside_the_grid():
    """
    The normalizer covers the atom's whole ball but the scatter stops at the
    grid, so a face atom keeps less than its weight. Deliberate, and the reason
    the two passes clip to different extents.
    """
    at_face = np.array([[0.0, 8.0, 8.0]], dtype=np.float32)
    inside = np.array([[8.0, 8.0, 8.0]], dtype=np.float32)
    weight = np.ones(1, dtype=np.float32)
    face_sum = np.sum(
        cpu().frame_voxelize(at_face, weight, DIMS, SPACING, CUTOFF, SIGMA, 0)
    )
    inside_sum = np.sum(
        cpu().frame_voxelize(inside, weight, DIMS, SPACING, CUTOFF, SIGMA, 0)
    )
    assert inside_sum == pytest.approx(1.0, rel=1e-5)
    assert 0.3 < face_sum < 0.7  # roughly the half-ball that lands on the grid


@requires_cpu
@pytest.mark.parametrize(
    "label,coords,weights",
    [
        (
            "placeholder padding",
            np.full((4, 3), PLACEHOLDER, np.float32),
            np.ones(4, np.float32),
        ),
        ("zero weights", np.full((4, 3), 8.0, np.float32), np.zeros(4, np.float32)),
        (
            "far outside the grid",
            np.full((4, 3), 500.0, np.float32),
            np.ones(4, np.float32),
        ),
    ],
)
def test_atoms_that_should_contribute_nothing_contribute_nothing(
    label, coords, weights
):
    grid = cpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)
    assert np.count_nonzero(grid) == 0, label


@requires_cpu
def test_padding_does_not_disturb_the_real_atoms():
    real = np.array([[6.0, 6.0, 6.0]], dtype=np.float32)
    padded = np.array([[6.0, 6.0, 6.0], [PLACEHOLDER] * 3], dtype=np.float32)
    alone = cpu().frame_voxelize(
        real, np.ones(1, np.float32), DIMS, SPACING, CUTOFF, SIGMA, 0
    )
    with_pad = cpu().frame_voxelize(
        padded, np.ones(2, np.float32), DIMS, SPACING, CUTOFF, SIGMA, 0
    )
    assert np.array_equal(alone, with_pad)


@requires_cpu
def test_the_density_peaks_at_the_grid_point_nearest_the_atom():
    coords = np.array([[8.0, 8.0, 8.0]], dtype=np.float32)
    grid = (
        cpu()
        .frame_voxelize(coords, np.ones(1, np.float32), DIMS, SPACING, CUTOFF, SIGMA, 0)
        .reshape(DIMS)
    )
    # index = coordinate / spacing, and the flat layout makes z the fastest axis
    assert np.unravel_index(np.argmax(grid), grid.shape) == (16, 16, 16)


@requires_cpu
def test_the_cpu_path_is_deterministic():
    coords, weights = random_frame(500, seed=3)
    first = cpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)
    second = cpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)
    assert np.array_equal(first, second)


@requires_cpu
def test_a_still_trajectory_aggregates_the_way_the_definitions_say():
    """Every frame identical, so the aggregations have known closed forms."""
    coords, weights = random_frame(200, seed=5)
    n_frames = 6
    traj = np.repeat(coords[None, :, :], n_frames, axis=0)
    w = np.tile(weights, n_frames)
    single = cpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)

    for agg in (1, 3, 5, 6):  # mean, median, max, min of a constant series
        got = cpu().density_flow(traj, w, DIMS, SPACING, CUTOFF, SIGMA, agg)
        assert np.allclose(got, single, rtol=1e-6, atol=1e-8), AGG_NAMES[agg]
    for agg in (2, 4, 7, 8):  # spread, entropy and drift of a constant series
        got = cpu().density_flow(traj, w, DIMS, SPACING, CUTOFF, SIGMA, agg)
        assert np.allclose(got, 0.0, atol=1e-7), AGG_NAMES[agg]


@requires_cpu
def test_drift_recovers_a_known_linear_ramp():
    """Scaling frame f by (1 + f) makes each voxel linear in f with slope = value."""
    coords, weights = random_frame(150, seed=8)
    n_frames = 8
    traj = np.repeat(coords[None, :, :], n_frames, axis=0)
    w = np.concatenate([weights * (1.0 + f) for f in range(n_frames)]).astype(
        np.float32
    )
    unit = cpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)
    drift = cpu().density_flow(traj, w, DIMS, SPACING, CUTOFF, SIGMA, 8)
    assert np.allclose(drift, unit, rtol=1e-4, atol=1e-7)


@requires_cpu
def test_an_unsupported_aggregation_is_rejected():
    traj, w = random_traj(2, 8, seed=1)
    with pytest.raises(ValueError, match="aggregation type"):
        cpu().density_flow(traj, w, DIMS, SPACING, CUTOFF, SIGMA, 999)


@requires_cpu
def test_empty_input_yields_an_empty_grid_rather_than_failing():
    grid = cpu().frame_voxelize(
        np.zeros((0, 3), np.float32),
        np.zeros(0, np.float32),
        DIMS,
        SPACING,
        CUTOFF,
        SIGMA,
        0,
    )
    assert grid.shape == (DIMS.prod(),)
    assert np.count_nonzero(grid) == 0


###############################################################################
# The two backends agree
###############################################################################


@requires_cpu
@requires_gpu
@pytest.mark.parametrize("n_atoms", [1, 7, 64, 1000, 4096])
def test_frame_voxelize_matches_the_gpu(n_atoms):
    coords, weights = random_frame(n_atoms, seed=100 + n_atoms)
    g = gpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)
    c = cpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)
    assert rel_diff(g, c) < RTOL


@requires_cpu
@requires_gpu
def test_atoms_on_and_beyond_the_faces_match_the_gpu():
    """Where the normalizer and scatter extents differ, they must differ alike."""
    offsets = (-4.0, -2.0, -0.25, 0.0, 0.25, 7.75, 8.0, 15.75, 16.0, 18.0, 20.0)
    coords = np.array(
        [[v, 8.0, 8.0] for v in offsets]
        + [[8.0, v, 8.0] for v in offsets]
        + [[8.0, 8.0, v] for v in offsets]
        + [[v, v, v] for v in offsets],
        dtype=np.float32,
    )
    weights = np.ones(len(coords), dtype=np.float32)
    g = gpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)
    c = cpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)
    assert rel_diff(g, c) < RTOL

    for i, point in enumerate(coords):
        single = point.reshape(1, 3)
        one = np.ones(1, dtype=np.float32)
        gi = gpu().frame_voxelize(single, one, DIMS, SPACING, CUTOFF, SIGMA, 0)
        ci = cpu().frame_voxelize(single, one, DIMS, SPACING, CUTOFF, SIGMA, 0)
        assert rel_diff(gi, ci) < RTOL, f"atom {i} at {point}"


@requires_cpu
@requires_gpu
def test_exactly_at_cutoff_geometry_matches_the_gpu():
    """Both sides derive the ball bounds with ceil/floor; the edge must agree."""
    coords = np.array(
        [
            [CUTOFF, 4.0, 4.0],
            [4.0 + CUTOFF, 4.0, 4.0],
            [0.0, CUTOFF, 7.0],
            [CUTOFF + 1e-6, 4.0, 4.0],
            [CUTOFF - 1e-6, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    weights = np.ones(len(coords), dtype=np.float32)
    g = gpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)
    c = cpu().frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)
    assert rel_diff(g, c) < RTOL


@requires_cpu
@requires_gpu
@pytest.mark.parametrize("dims", [[16, 24, 32], [32, 16, 24], [8, 32, 32]])
def test_non_cubic_grids_match_the_gpu(dims):
    """The layout pairs axis x with dims[2]; both backends must do it the same."""
    coords, weights = random_frame(256, seed=42, span=8.0)
    d = np.array(dims, dtype=int)
    g = gpu().frame_voxelize(coords, weights, d, SPACING, CUTOFF, SIGMA, 0)
    c = cpu().frame_voxelize(coords, weights, d, SPACING, CUTOFF, SIGMA, 0)
    assert rel_diff(g, c) < RTOL


@requires_cpu
@requires_gpu
@pytest.mark.parametrize(
    "spacing,cutoff,sigma",
    [(0.5, 3.5, 1.5), (1.0, 4.0, 1.0), (0.25, 2.0, 0.8), (0.5, 8.0, 3.0)],
)
def test_parameter_sweep_matches_the_gpu(spacing, cutoff, sigma):
    coords, weights = random_frame(512, seed=77)
    g = gpu().frame_voxelize(coords, weights, DIMS, spacing, cutoff, sigma, 0)
    c = cpu().frame_voxelize(coords, weights, DIMS, spacing, cutoff, sigma, 0)
    assert rel_diff(g, c) < RTOL


@requires_cpu
@requires_gpu
@pytest.mark.parametrize("n_frames", [1, 2, 10, 37])
@pytest.mark.parametrize("agg", CONTINUOUS_AGGS)
def test_density_flow_matches_the_gpu(n_frames, agg):
    traj, weights = random_traj(n_frames, 300, seed=200 + n_frames)
    g = gpu().density_flow(traj, weights, DIMS, SPACING, CUTOFF, SIGMA, agg)
    c = cpu().density_flow(traj, weights, DIMS, SPACING, CUTOFF, SIGMA, agg)
    assert rel_diff(g, c) < RTOL, AGG_NAMES[agg]


@requires_cpu
@requires_gpu
@pytest.mark.parametrize("n_frames", [10, 37])
def test_drift_matches_the_gpu_to_the_scale_of_the_grid(n_frames):
    """
    Drift is a small difference of larger numbers, so relative error against the
    drift itself is meaningless. Bound it by the grid the slope was taken from.
    """
    traj, weights = random_traj(n_frames, 300, seed=300 + n_frames)
    g = gpu().density_flow(traj, weights, DIMS, SPACING, CUTOFF, SIGMA, 8)
    c = cpu().density_flow(traj, weights, DIMS, SPACING, CUTOFF, SIGMA, 8)
    scale = np.abs(
        cpu().density_flow(traj, weights, DIMS, SPACING, CUTOFF, SIGMA, 1)
    ).max()
    assert np.abs(g - c).max() <= DRIFT_SCALE_TOL * scale


@requires_cpu
@requires_gpu
@pytest.mark.parametrize("n_frames", [10, 37])
def test_entropy_matches_the_gpu_on_all_but_a_few_voxels(n_frames):
    """
    Histogram binning is a step function, so a voxel whose value sits a
    float-epsilon from a bin edge can land in either bin. Only a handful may.
    """
    traj, weights = random_traj(n_frames, 300, seed=400 + n_frames)
    g = gpu().density_flow(traj, weights, DIMS, SPACING, CUTOFF, SIGMA, 7)
    c = cpu().density_flow(traj, weights, DIMS, SPACING, CUTOFF, SIGMA, 7)
    differing = np.count_nonzero(np.abs(g - c) > RTOL * max(np.abs(g).max(), 1e-30))
    assert differing / g.size <= ENTROPY_MISMATCH_TOL


###############################################################################
# The facade picks a backend; nothing else should have to
###############################################################################


@requires_cpu
def test_commands_falls_back_to_the_cpu_without_the_cuda_extension(monkeypatch):
    from nearl import commands

    monkeypatch.setattr(commands, "HAS_CUDA_EXTENSION", False)
    monkeypatch.setattr(commands, "_cpu_fallback_announced", False)
    assert (
        commands._voxelize_backend("frame_voxelize")
        is commands._voxelize_cpu.frame_voxelize
    )

    coords, weights = random_frame(64, seed=9)
    grid = commands.frame_voxelize(coords, weights, DIMS, SPACING, CUTOFF, SIGMA)
    direct = commands._voxelize_cpu.frame_voxelize(
        coords, weights, DIMS, SPACING, CUTOFF, SIGMA, 0
    )
    assert grid.shape == tuple(DIMS)
    assert np.array_equal(grid.ravel(), direct)


@requires_cpu
@requires_gpu
def test_commands_prefers_the_gpu_when_it_is_available():
    from nearl import commands

    assert commands.HAS_CUDA_EXTENSION
    assert (
        commands._voxelize_backend("density_flow") is commands.all_actions.density_flow
    )
