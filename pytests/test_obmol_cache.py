"""
Checks the OBMol and per-property label caches in :mod:`nearl.chemtools`.

A cache is only correct if it is invisible: whatever the caller asks for, it must
get what an uncached build would have returned. So the tests come in two halves -
the cache hits (that the work really is skipped) and the cache misses (that
changed inputs are never answered from a stale slot).

The fixture is a benzene plus one ion, small enough to assert values outright.
No network and no GPU.
"""

import itertools
import math

import numpy as np
import pytest

import nearl.chemtools as chemtools
import nearl.io
from nearl.chemtools import (
    label_aromaticity,
    label_hbond_acceptor,
    label_hbond_donor,
    label_hybridization,
    label_ring_status,
    traj_to_obmol,
)

LABELLERS = [
    label_aromaticity,
    label_ring_status,
    label_hybridization,
    label_hbond_donor,
    label_hbond_acceptor,
]

N_ATOMS = 13  # 6 C + 6 H + one ion
TAIL = 12  # index of the ion


def _pdb(scale=1.0, tail=("K", "K", "K")):
    """Benzene scaled about the origin, plus one well-separated single atom."""

    def record(serial, name, res, resid, xyz, element):
        x, y, z = xyz
        return (
            f"{'HETATM':<6}{serial:>5} {name:^4} {res:>3}  {resid:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00{'':10}{element:>2}"
        )

    lines, serial = [], 0
    for radius, prefix, element in ((1.39, "C", "C"), (2.48, "H", "H")):
        for i in range(6):
            angle = math.radians(60 * i)
            serial += 1
            xyz = (
                radius * scale * math.cos(angle),
                radius * scale * math.sin(angle),
                0.0,
            )
            lines.append(record(serial, f"{prefix}{i + 1}", "BNZ", 1, xyz, element))
    name, res, element = tail
    lines.append(record(serial + 1, name, res, 2, (40.0, 0.0, 0.0), element))
    return "\n".join(lines) + "\nTER\nEND\n"


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return nearl.io.Trajectory(str(path))


@pytest.fixture
def traj(tmp_path):
    # Function-scoped: several tests mutate the trajectory they are given.
    return _write(tmp_path, "benzene.pdb", _pdb())


@pytest.fixture
def two_frames(tmp_path):
    models = [
        "MODEL        1",
        _pdb().rstrip(),
        "ENDMDL",
        "MODEL        2",
        _pdb(scale=3.0).rstrip(),
        "ENDMDL",
    ]
    return _write(tmp_path, "two_frames.pdb", "\n".join(models) + "\n")


@pytest.fixture
def counters(monkeypatch):
    """Count molecule builds and OpenBabel atom traversals."""
    counts = {"builds": 0, "traversals": 0}
    build, iterate = chemtools._build_obmol, chemtools.ob.OBMolAtomIter

    def counted_build(trajectory, frame_index):
        counts["builds"] += 1
        return build(trajectory, frame_index)

    def counted_iterate(mol):
        counts["traversals"] += 1
        return iterate(mol)

    monkeypatch.setattr(chemtools, "_build_obmol", counted_build)
    monkeypatch.setattr(chemtools.ob, "OBMolAtomIter", counted_iterate)
    return counts


def _ob_coords(mol):
    return np.array(
        [[a.GetX(), a.GetY(), a.GetZ()] for a in chemtools.ob.OBMolAtomIter(mol)]
    )


def test_the_fixture_has_the_chemistry_the_tests_assume(traj):
    assert traj.top.n_atoms == N_ATOMS
    assert [a.atomic_number for a in traj.top.atoms] == [6] * 6 + [1] * 6 + [19]
    aromatic = label_aromaticity(traj)
    assert aromatic[:6].tolist() == [1] * 6  # the ring
    assert aromatic[6:].tolist() == [0] * 7  # hydrogens and the ion


###############################################################################
# Hits: the work is actually skipped
###############################################################################


def test_the_molecule_is_built_once_for_the_whole_feature_set(traj, counters):
    for labeller in LABELLERS:
        labeller(traj)
    traj_to_obmol(traj)
    assert counters["builds"] == 1


def test_a_hit_returns_the_same_molecule(traj):
    assert traj_to_obmol(traj) is traj_to_obmol(traj)


def test_each_property_is_traversed_once_however_often_it_is_asked_for(traj, counters):
    for _ in range(3):
        label_aromaticity(traj)
        label_ring_status(traj)
    assert counters["traversals"] == 2


###############################################################################
# Correctness: a hit equals what an uncached build would have returned
###############################################################################


@pytest.mark.parametrize("labeller", LABELLERS, ids=lambda f: f.__name__)
def test_a_cached_label_matches_an_uncached_one(tmp_path, monkeypatch, labeller):
    reference = _write(tmp_path, "reference.pdb", _pdb())
    keys = itertools.count()
    monkeypatch.setattr(chemtools, "_obmol_key", lambda trajectory, frame: next(keys))
    fresh = labeller(reference)
    monkeypatch.undo()

    cached = _write(tmp_path, "cached.pdb", _pdb())
    labeller(cached)  # populate
    assert np.array_equal(labeller(cached), fresh)


###############################################################################
# Misses: changed inputs are never answered from a stale slot
###############################################################################


def test_moving_the_atoms_rebuilds_the_molecule(traj, counters):
    stale = traj_to_obmol(traj)
    traj.xyz[0] *= 3.0
    rebuilt = traj_to_obmol(traj)

    assert rebuilt is not stale
    assert counters["builds"] == 2
    # The molecule handed out has to carry the coordinates the caller now holds.
    assert _ob_coords(rebuilt) == pytest.approx(traj.xyz[0], rel=1e-4)


def test_moving_the_atoms_recomputes_the_labels(traj, counters, monkeypatch):
    label_hybridization(traj)
    traj.xyz[0] *= 3.0
    moved = label_hybridization(traj)
    assert counters["traversals"] == 2

    # Compare against the same coordinates labelled with the cache defeated: a
    # trajectory rebuilt from a 3x PDB is not the same system, because pytraj
    # would infer no bonds at that separation.
    keys = itertools.count()
    monkeypatch.setattr(chemtools, "_obmol_key", lambda trajectory, frame: next(keys))
    assert np.array_equal(moved, label_hybridization(traj))


def test_replacing_the_topology_invalidates_the_cache(tmp_path, traj):
    """
    The coordinates do not move, so only the topology identity separates the two:
    swapping K+ for an oxygen turns the last atom into an H-bond acceptor.
    """
    assert label_hbond_acceptor(traj)[TAIL] == 0

    variant = _write(tmp_path, "variant.pdb", _pdb(tail=("O", "HOH", "O")))
    assert label_hbond_acceptor(variant)[TAIL] == 1  # guards the test itself

    traj.top = variant.top
    assert label_hbond_acceptor(traj)[TAIL] == 1


def test_a_different_frame_is_not_served_from_the_cache(two_frames):
    assert two_frames.n_frames == 2
    first = traj_to_obmol(two_frames, 0)
    second = traj_to_obmol(two_frames, 1)

    assert second is not first
    assert _ob_coords(second) == pytest.approx(two_frames.xyz[1], rel=1e-4)
    # Going back must not hand out the frame-1 molecule.
    assert _ob_coords(traj_to_obmol(two_frames, 0)) == pytest.approx(
        two_frames.xyz[0], rel=1e-4
    )


###############################################################################
# The label cache holds several properties under one key
###############################################################################


def test_a_second_property_does_not_disturb_the_first(traj):
    aromatic = label_aromaticity(traj)
    label_ring_status(traj)
    assert np.array_equal(label_aromaticity(traj), aromatic)
    assert sorted(getattr(traj, chemtools._LABEL_CACHE_ATTR)[1]) == [
        "IsAromatic",
        "IsInRing",
    ]


def test_a_key_change_drops_every_stale_property(traj):
    label_aromaticity(traj)
    label_ring_status(traj)
    traj.xyz[0] *= 3.0
    label_ring_status(traj)
    assert list(getattr(traj, chemtools._LABEL_CACHE_ATTR)[1]) == ["IsInRing"]


def test_the_caller_gets_a_copy_it_can_safely_modify(traj):
    returned = label_aromaticity(traj)
    returned[:] = 42.0
    assert label_aromaticity(traj)[:6].tolist() == [1] * 6


def test_two_trajectories_keep_separate_caches(tmp_path, traj):
    assert label_hbond_acceptor(traj)[TAIL] == 0
    other = _write(tmp_path, "other.pdb", _pdb(tail=("O", "HOH", "O")))
    assert label_hbond_acceptor(other)[TAIL] == 1
    assert label_hbond_acceptor(traj)[TAIL] == 0


def test_a_trajectory_that_rejects_attributes_still_labels_correctly(traj):
    """Objects that cannot take the cache attribute must still get an answer."""

    class Fixed:
        __slots__ = ("top", "xyz")

        def __init__(self, source):
            self.top = source.top
            self.xyz = source.xyz

    fixed = Fixed(traj)
    assert np.array_equal(label_aromaticity(fixed), label_aromaticity(traj))
    assert not hasattr(fixed, chemtools._LABEL_CACHE_ATTR)
