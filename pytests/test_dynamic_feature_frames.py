"""
Frame handling of the two dynamic features, DensityFlow and MarchingObservers.

Both kernels used to be launched once per frame in a host loop. They are now
launched once for the whole slice with ``blockIdx.y`` selecting the frame, which
moves three things into the kernel that the host used to get right by
construction: the offset into the coordinates, the offset into the weights, and
the offset into the per-frame output. A mistake in any of them still produces a
plausible grid, so "not NaN" would not notice.

What pins it down is that the same numbers are reachable a second way. Each
feature has a single-frame entry point -- ``frame_voxelize`` and
``frame_observation`` -- that the fused launch does not touch, so aggregating
those on the host gives an independent answer to compare against.

Aggregations that ignore order (mean, max, ...) would still pass if the frames
were permuted, so ordering is checked separately with drift, and the per-frame
output offset with a slice where only one frame carries atoms.

Tolerances are measured, not guessed: the worst observed disagreement across
these cases is 7.9e-07, from ``atomicAdd`` ordering inside frame_voxelize.
frame_observation happens to be deterministic, but nothing here relies on that.
"""

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

DIMS = np.array([24, 24, 24], dtype=int)
SPACING, CUTOFF, SIGMA = 0.5, 3.5, 1.5
RTOL = 1e-5  # ~13x headroom over the measured 7.9e-07

# Order-insensitive aggregations with a direct NumPy equivalent. Entropy (7) is
# excluded on purpose: it bins its input, so a value a float-epsilon from a bin
# edge lands either side and a voxel can differ outright. Drift (8) is covered
# by the ordering tests below, where the construction makes it well conditioned.
HOST_AGG = {
    1: lambda x: x.mean(0),
    2: lambda x: x.std(0),
    3: lambda x: np.median(x, 0),
    4: lambda x: x.var(0),
    5: lambda x: x.max(0),
    6: lambda x: x.min(0),
}
AGG_NAMES = {1: "mean", 2: "stddev", 3: "median", 4: "variance", 5: "max", 6: "min"}

# The observer kernel branches on the observable: 1 and 2 take the unweighted
# overload, everything else the weighted one. The fusion changed both, so both
# are exercised. 1=existence, 2=direct_count, 12=cumulative_weight, 13=density.
OBSERVABLES = [1, 2, 12, 13]


def actions():
    from nearl import all_actions

    return all_actions


def rel_diff(a, b):
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30))


def random_slice(n_frames, n_atoms, seed, span=12.0):
    rng = np.random.default_rng(seed)
    traj = (rng.random((n_frames, n_atoms, 3)) * span).astype(np.float32)
    weights = (rng.random(n_frames * n_atoms) + 0.5).astype(np.float32)
    return traj, weights


def per_frame_voxelize(traj, weights, dims=DIMS):
    """The single-frame path, stacked. Independent of the fused launch."""
    n_atoms = traj.shape[1]
    return np.stack(
        [
            actions().frame_voxelize(
                traj[f],
                weights[f * n_atoms : (f + 1) * n_atoms],
                dims,
                SPACING,
                CUTOFF,
                SIGMA,
                0,
            )
            for f in range(len(traj))
        ]
    ).astype(np.float64)


def per_frame_observation(traj, weights, observable, dims=DIMS):
    n_atoms = traj.shape[1]
    return np.stack(
        [
            actions().frame_observation(
                traj[f],
                weights[f * n_atoms : (f + 1) * n_atoms],
                dims,
                SPACING,
                CUTOFF,
                observable,
            )
            for f in range(len(traj))
        ]
    ).astype(np.float64)


###############################################################################
# The fused launch agrees with the single-frame path it replaced
###############################################################################


@requires_extension
@pytest.mark.parametrize("agg", list(HOST_AGG), ids=AGG_NAMES.get)
@pytest.mark.parametrize("n_frames", [1, 2, 7, 33])
def test_density_flow_matches_per_frame_voxelize(agg, n_frames):
    traj, weights = random_slice(n_frames, 200, seed=100 + n_frames)
    fused = actions().density_flow(traj, weights, DIMS, SPACING, CUTOFF, SIGMA, agg)
    expected = HOST_AGG[agg](per_frame_voxelize(traj, weights))
    assert rel_diff(expected, fused) < RTOL


@requires_extension
@pytest.mark.parametrize("agg", list(HOST_AGG), ids=AGG_NAMES.get)
@pytest.mark.parametrize("observable", OBSERVABLES)
def test_marching_observer_matches_per_frame_observation(agg, observable):
    traj, weights = random_slice(7, 200, seed=200 + observable)
    fused = actions().marching_observer(
        traj, weights, DIMS, SPACING, CUTOFF, observable, agg
    )
    expected = HOST_AGG[agg](per_frame_observation(traj, weights, observable))
    assert rel_diff(expected, fused) < RTOL


@requires_extension
def test_a_single_frame_slice_equals_the_single_frame_entry_point():
    """The degenerate slice: gridDim.y of 1 must behave like the old 1-D launch."""
    traj, weights = random_slice(1, 300, seed=7)
    flow = actions().density_flow(traj, weights, DIMS, SPACING, CUTOFF, SIGMA, 1)
    assert rel_diff(per_frame_voxelize(traj, weights)[0], flow) < RTOL

    obs = actions().marching_observer(traj, weights, DIMS, SPACING, CUTOFF, 13, 1)
    assert rel_diff(per_frame_observation(traj, weights, 13)[0], obs) < RTOL


@requires_extension
@pytest.mark.parametrize("dims", [[16, 24, 32], [32, 16, 24], [8, 32, 32]])
def test_non_cubic_grids_keep_the_frames_aligned(dims):
    """The stride between per-frame outputs is dims[0]*dims[1]*dims[2]."""
    d = np.array(dims, dtype=int)
    traj, weights = random_slice(5, 200, seed=42, span=8.0)
    flow = actions().density_flow(traj, weights, d, SPACING, CUTOFF, SIGMA, 1)
    assert rel_diff(per_frame_voxelize(traj, weights, d).mean(0), flow) < RTOL

    obs = actions().marching_observer(traj, weights, d, SPACING, CUTOFF, 13, 1)
    assert rel_diff(per_frame_observation(traj, weights, 13, d).mean(0), obs) < RTOL


###############################################################################
# Ordering: mean and max would survive a permuted frame axis, drift would not
###############################################################################


def _linear_ramp(n_frames, n_atoms, seed):
    """Same atoms every frame, weights scaled by (1 + f).

    Every voxel is then linear in the frame index with slope equal to the
    single-frame grid, so the drift has a known closed form and is well
    conditioned -- unlike drift on arbitrary data, which is a small difference
    of larger numbers.
    """
    rng = np.random.default_rng(seed)
    atoms = (rng.random((n_atoms, 3)) * 12.0).astype(np.float32)
    unit_w = (rng.random(n_atoms) + 0.5).astype(np.float32)
    traj = np.repeat(atoms[None], n_frames, axis=0)
    weights = np.concatenate([unit_w * (1.0 + f) for f in range(n_frames)]).astype(
        np.float32
    )
    return atoms, unit_w, traj, weights


@requires_extension
def test_drift_recovers_the_ramp_for_both_features():
    atoms, unit_w, traj, weights = _linear_ramp(8, 150, seed=11)
    unit_flow = actions().frame_voxelize(atoms, unit_w, DIMS, SPACING, CUTOFF, SIGMA, 0)
    drift = actions().density_flow(traj, weights, DIMS, SPACING, CUTOFF, SIGMA, 8)
    assert rel_diff(unit_flow, drift) < RTOL

    unit_obs = actions().frame_observation(atoms, unit_w, DIMS, SPACING, CUTOFF, 13)
    drift_obs = actions().marching_observer(traj, weights, DIMS, SPACING, CUTOFF, 13, 8)
    assert rel_diff(unit_obs, drift_obs) < RTOL


@requires_extension
def test_reversing_the_frames_negates_the_drift():
    """A permuted or mis-strided frame axis would not flip the sign cleanly."""
    _, unit_w, traj, weights = _linear_ramp(8, 150, seed=11)
    n_frames, n_atoms = traj.shape[0], traj.shape[1]
    reversed_w = np.concatenate(
        [unit_w * (1.0 + f) for f in reversed(range(n_frames))]
    ).astype(np.float32)
    assert len(reversed_w) == n_frames * n_atoms

    forward = actions().density_flow(traj, weights, DIMS, SPACING, CUTOFF, SIGMA, 8)
    backward = actions().density_flow(traj, reversed_w, DIMS, SPACING, CUTOFF, SIGMA, 8)
    assert rel_diff(-forward, backward) < RTOL


###############################################################################
# Output offset: only one frame carries atoms, so its grid is identifiable
###############################################################################


@requires_extension
@pytest.mark.parametrize("populated", [0, 3, 7])
def test_only_the_populated_frame_contributes(populated):
    """
    Writing a frame to the wrong output offset, or reading the wrong input
    offset, moves the signal to another frame. Here every other frame sits far
    outside the box and contributes nothing, so max over frames must reproduce
    that one grid and mean must be it divided by the frame count.
    """
    n_frames, n_atoms = 8, 150
    rng = np.random.default_rng(5)
    atoms = (rng.random((n_atoms, 3)) * 12.0).astype(np.float32)
    weights = (rng.random(n_atoms) + 0.5).astype(np.float32)

    traj = np.full((n_frames, n_atoms, 3), 500.0, dtype=np.float32)
    traj[populated] = atoms
    flat_w = np.tile(weights, n_frames).astype(np.float32)
    alone = actions().frame_voxelize(atoms, weights, DIMS, SPACING, CUTOFF, SIGMA, 0)

    largest = actions().density_flow(traj, flat_w, DIMS, SPACING, CUTOFF, SIGMA, 5)
    average = actions().density_flow(traj, flat_w, DIMS, SPACING, CUTOFF, SIGMA, 1)
    assert rel_diff(alone, largest) < RTOL
    assert rel_diff(alone / n_frames, average) < RTOL
