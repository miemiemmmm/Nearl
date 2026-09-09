"""
Absolute checks on the geometry of the voxelization kernel.

``frame_interp_global`` no longer strides the whole grid per atom; it visits the
integer bounding box of the atom's cutoff ball. Two details make that delicate:

* the two passes cover **different** extents. The normalizer runs over the
  buffered grid, which overhangs the output by ``buff_dim`` on every face, while
  the scatter runs only over ``[0, dims)``. An interior atom therefore deposits
  its whole weight and an atom near a face deposits only the part that lands on
  the grid. Collapsing the two boxes into one is the natural refactoring slip and
  changes every edge atom.
* the bounds come from ``ceilf``/``floorf`` on the ball, so an off-by-one drops
  the outermost shell of contributing points.

Neither shows up in a test that compares one code path against another, because
both paths move together. ``test_dynamic_feature_frames`` passes with the ball
shrunk by a voxel per side, and the whole suite passes with the two boxes
collapsed. So everything here is *absolute*: a NumPy oracle, conservation of
weight, and the support of the kernel.

The oracle is deliberately a naive float64 transcription of what the kernel is
specified to do -- not of how it does it -- so it stays a real second opinion.
Agreement is ~2e-07, limited by the kernel's float math and atomicAdd ordering.
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

PLACEHOLDER = 99999.0
RTOL = 5e-6  # ~25x headroom over the measured 2e-07


def voxelize(coords, weights, dims, spacing, cutoff, sigma):
    from nearl import all_actions

    return all_actions.frame_voxelize(
        np.asarray(coords, np.float32),
        np.asarray(weights, np.float32),
        np.asarray(dims, dtype=int),
        spacing,
        cutoff,
        sigma,
        0,
    ).astype(np.float64)


def oracle(coords, weights, dims, spacing, cutoff, sigma):
    """What the kernel is specified to compute, in float64.

    Grid point i sits at ``i * spacing``. Each atom is normalized over the
    buffered grid and deposits only into ``[0, dims)``. The flat index is
    ``x*dims[0]*dims[1] + y*dims[0] + z``, so axis x pairs with dims[2] and z
    with dims[0] -- a transposition the kernel has always had.
    """
    dims = np.asarray(dims, dtype=int)
    buff = int((cutoff + spacing) / spacing)
    extent = (dims[2], dims[1], dims[0])
    out = np.zeros(int(dims.prod()), dtype=np.float64)

    def gauss(d2):
        return np.exp(-0.5 * (np.sqrt(d2) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    def box(lo_hi):
        axes = [np.arange(lo, hi) for lo, hi in lo_hi]
        return np.meshgrid(*axes, indexing="ij")

    for c, w in zip(np.asarray(coords, float), np.asarray(weights, float)):
        if np.all(c == PLACEHOLDER) or w == 0.0:
            continue

        gx, gy, gz = box([(-buff, extent[a] + buff) for a in range(3)])
        d2 = (
            (c[0] - gx * spacing) ** 2
            + (c[1] - gy * spacing) ** 2
            + (c[2] - gz * spacing) ** 2
        )
        total = gauss(d2[d2 < cutoff * cutoff]).sum()
        if total == 0:
            continue

        gx, gy, gz = box([(0, extent[a]) for a in range(3)])
        d2 = (
            (c[0] - gx * spacing) ** 2
            + (c[1] - gy * spacing) ** 2
            + (c[2] - gz * spacing) ** 2
        )
        inside = d2 < cutoff * cutoff
        flat = (gx * dims[0] * dims[1] + gy * dims[0] + gz)[inside]
        np.add.at(out, flat, gauss(d2[inside]) * (w / total))
    return out


def rel_diff(a, b):
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30))


###############################################################################
# Against an independent reference
###############################################################################


@requires_extension
@pytest.mark.parametrize(
    "label,coords",
    [
        ("interior", [[4.0, 4.0, 4.0]]),
        ("on a face", [[0.0, 4.0, 4.0]]),
        ("just inside a face", [[0.2, 4.0, 4.0]]),
        ("outside, ball overlaps", [[-2.0, 4.0, 4.0]]),
        ("outside, beyond cutoff", [[-6.0, 4.0, 4.0]]),
        ("far corner", [[7.9, 7.9, 7.9]]),
        ("corner overhang", [[-1.0, -1.0, -1.0]]),
    ],
)
def test_a_single_atom_matches_the_oracle(label, coords):
    dims, spacing, cutoff, sigma = [16, 16, 16], 0.5, 3.5, 1.5
    w = np.ones(len(coords))
    assert (
        rel_diff(
            oracle(coords, w, dims, spacing, cutoff, sigma),
            voxelize(coords, w, dims, spacing, cutoff, sigma),
        )
        < RTOL
    ), label


@requires_extension
@pytest.mark.parametrize(
    "spacing,cutoff,sigma",
    [(0.5, 3.5, 1.5), (1.0, 4.0, 1.0), (0.25, 2.0, 0.8), (0.5, 1.0, 0.4)],
)
def test_the_parameter_space_matches_the_oracle(spacing, cutoff, sigma):
    """buff_dim and the ball radius both move with spacing and cutoff."""
    rng = np.random.default_rng(4)
    dims = [16, 16, 16]
    coords = rng.random((6, 3)) * 8.0
    weights = rng.random(6) + 0.5
    assert (
        rel_diff(
            oracle(coords, weights, dims, spacing, cutoff, sigma),
            voxelize(coords, weights, dims, spacing, cutoff, sigma),
        )
        < RTOL
    )


@requires_extension
@pytest.mark.parametrize("dims", [[8, 12, 16], [16, 8, 12], [12, 16, 8]])
def test_non_cubic_grids_match_the_oracle(dims):
    """The flat index pairs x with dims[2]; a symmetric grid would hide a swap."""
    rng = np.random.default_rng(9)
    coords = rng.random((6, 3)) * 4.0
    weights = rng.random(6) + 0.5
    assert (
        rel_diff(
            oracle(coords, weights, dims, 0.5, 3.5, 1.5),
            voxelize(coords, weights, dims, 0.5, 3.5, 1.5),
        )
        < RTOL
    )


###############################################################################
# Conservation: the normalizer is what makes an atom carry its own weight
###############################################################################


@requires_extension
@pytest.mark.parametrize(
    "spacing,cutoff,sigma",
    [(0.5, 3.5, 1.5), (1.0, 4.0, 1.0), (0.25, 2.0, 0.8), (0.5, 6.0, 2.5)],
)
def test_interior_atoms_deposit_exactly_their_weight(spacing, cutoff, sigma):
    """
    A ball that is a shell too small, or a normalizer over the wrong extent,
    both break this. Atoms are kept at least `cutoff` from every face so the
    whole ball lands on the grid.
    """
    dims = np.array([32, 32, 32])
    span = dims[0] * spacing
    rng = np.random.default_rng(12)
    coords = cutoff + rng.random((8, 3)) * (span - 2 * cutoff)
    weights = rng.random(8) + 0.5
    grid = voxelize(coords, weights, dims, spacing, cutoff, sigma)
    assert np.sum(grid) == pytest.approx(np.sum(weights), rel=1e-5)


@requires_extension
def test_an_atom_on_a_face_keeps_only_the_part_inside_the_grid():
    """
    The normalizer covers the whole ball, the scatter stops at the grid, so a
    face atom keeps roughly half its weight. Collapsing the two extents into one
    would renormalize over the truncated ball and hand back the full weight.
    """
    dims = [32, 32, 32]
    one = np.ones(1)
    inside = np.sum(voxelize([[8.0, 8.0, 8.0]], one, dims, 0.5, 3.5, 1.5))
    face = np.sum(voxelize([[0.0, 8.0, 8.0]], one, dims, 0.5, 3.5, 1.5))
    corner = np.sum(voxelize([[0.0, 0.0, 0.0]], one, dims, 0.5, 3.5, 1.5))

    assert inside == pytest.approx(1.0, rel=1e-5)
    assert 0.4 < face < 0.6, face  # about half the ball
    assert 0.05 < corner < 0.2, corner  # about an eighth
    assert corner < face < inside


@requires_extension
def test_padding_and_zero_weights_contribute_nothing():
    dims = [16, 16, 16]
    real = [[4.0, 4.0, 4.0]]
    alone = voxelize(real, [1.0], dims, 0.5, 3.5, 1.5)

    padded = voxelize(
        [[4.0, 4.0, 4.0], [PLACEHOLDER] * 3], [1.0, 1.0], dims, 0.5, 3.5, 1.5
    )
    zeroed = voxelize(
        [[4.0, 4.0, 4.0], [6.0, 6.0, 6.0]], [1.0, 0.0], dims, 0.5, 3.5, 1.5
    )
    assert rel_diff(alone, padded) < RTOL
    assert rel_diff(alone, zeroed) < RTOL


###############################################################################
# Support: where the kernel is allowed to write at all
###############################################################################


@requires_extension
@pytest.mark.parametrize("spacing,cutoff", [(0.5, 3.5), (1.0, 4.0), (0.25, 2.0)])
def test_nothing_is_written_beyond_the_cutoff(spacing, cutoff):
    """
    Every non-zero voxel must lie within `cutoff` of the atom. A ball box that
    is too wide would still pass the oracle comparison, since the inner distance
    test filters it -- but it would show here if the test were ever dropped.
    """
    dims = np.array([32, 32, 32])
    centre = np.array([dims[0] * spacing / 2] * 3)
    grid = voxelize([centre], [1.0], dims, spacing, cutoff, 1.5).reshape(dims)

    idx = np.array(np.nonzero(grid))
    assert idx.size, "the atom deposited nothing"
    # index (i, j, k) of the flat layout corresponds to (x, y, z)
    pos = idx.T * spacing
    dist = np.linalg.norm(pos - centre, axis=1)
    assert dist.max() < cutoff, f"wrote {dist.max():.3f} A away, cutoff {cutoff}"


@requires_extension
def test_a_grid_point_exactly_at_the_cutoff_is_excluded():
    """The kernel tests dist_sq < cutoff_sq, so equality contributes nothing."""
    spacing, cutoff, sigma = 0.5, 3.5, 1.5
    dims = np.array([32, 32, 32])
    # Place the atom so the grid point at index 8 on x sits exactly cutoff away.
    atom = np.array([8 * spacing + cutoff, 8 * spacing, 8 * spacing])
    grid = voxelize([atom], [1.0], dims, spacing, cutoff, sigma).reshape(dims)
    assert grid[8, 8, 8] == 0.0

    nudged = atom.copy()
    nudged[0] -= 0.01  # now just inside
    grid = voxelize([nudged], [1.0], dims, spacing, cutoff, sigma).reshape(dims)
    assert grid[8, 8, 8] > 0.0


@requires_extension
def test_an_atom_far_outside_writes_nothing():
    dims = [16, 16, 16]
    grid = voxelize([[500.0, 500.0, 500.0]], [1.0], dims, 0.5, 3.5, 1.5)
    assert np.count_nonzero(grid) == 0
