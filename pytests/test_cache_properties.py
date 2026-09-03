"""
Per-property checks for :func:`nearl.features.cache_properties`.

The fixture builds a small system with known chemistry - benzene, an alanine
residue, a water and a potassium ion - so the expected values can be asserted
outright instead of merely checking array shapes. No network and no GPU.
"""

import math

import numpy as np
import pytest

import nearl.io
from nearl.features import SUPPORTED_FEATURES, cache_properties

# Index layout of the fixture system, in file order.
BENZENE_C = list(range(0, 6))
BENZENE_H = list(range(6, 12))
ALA_N, ALA_CA, ALA_C, ALA_O, ALA_CB, ALA_HA = 12, 13, 14, 15, 16, 17
WATER_O, WATER_H1, WATER_H2 = 18, 19, 20
POTASSIUM = 21
N_ATOMS = 22

ALA_BACKBONE = [ALA_N, ALA_CA, ALA_C, ALA_O, ALA_HA]


def _pdb_record(serial, name, res, resid, xyz, element, record="ATOM"):
    x, y, z = xyz
    return (
        f"{record:<6}{serial:>5} {name:^4} {res:>3}  {resid:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00{'':10}{element:>2}"
    )


def _synthetic_pdb():
    lines, serial = [], 0

    def add(name, res, resid, xyz, element, record="ATOM"):
        nonlocal serial
        serial += 1
        lines.append(_pdb_record(serial, name, res, resid, xyz, element, record))

    # Benzene: planar hexagon, aromatic and fully in a ring.
    for i in range(6):
        angle = math.radians(60 * i)
        add(
            f"C{i + 1}",
            "BNZ",
            1,
            (1.39 * math.cos(angle), 1.39 * math.sin(angle), 0.0),
            "C",
            "HETATM",
        )
    for i in range(6):
        angle = math.radians(60 * i)
        add(
            f"H{i + 1}",
            "BNZ",
            1,
            (2.48 * math.cos(angle), 2.48 * math.sin(angle), 0.0),
            "H",
            "HETATM",
        )

    # Alanine, offset well away so no bonds are inferred across fragments.
    for name, xyz, element in (
        ("N", (30.000, 0.000, 0.000), "N"),
        ("CA", (31.458, 0.000, 0.000), "C"),
        ("C", (32.009, 1.420, 0.000), "C"),
        ("O", (31.251, 2.390, 0.000), "O"),
        ("CB", (31.988, -0.773, -1.199), "C"),
        ("HA", (31.800, -0.540, 0.890), "H"),
    ):
        add(name, "ALA", 2, xyz, element)

    add("O", "WAT", 3, (50.000, 0.000, 0.000), "O")
    add("H1", "WAT", 3, (50.757, 0.586, 0.000), "H")
    add("H2", "WAT", 3, (49.243, 0.586, 0.000), "H")
    add("K", "K", 4, (70.000, 0.000, 0.000), "K", "HETATM")
    return "\n".join(lines) + "\nTER\nEND\n"


@pytest.fixture(scope="module")
def traj(tmp_path_factory):
    path = tmp_path_factory.mktemp("cacheprops") / "synthetic.pdb"
    path.write_text(_synthetic_pdb())
    return nearl.io.Trajectory(str(path))


def cached(traj, name, **kwargs):
    return cache_properties(traj, SUPPORTED_FEATURES[name], **kwargs)


def test_fixture_layout_is_what_the_tests_assume(traj):
    assert traj.top.n_atoms == N_ATOMS
    numbers = [a.atomic_number for a in traj.top.atoms]
    assert numbers[: len(BENZENE_C)] == [6] * 6
    assert numbers[WATER_O] == 8
    assert numbers[POTASSIUM] == 19
    # Benzene must come out bonded, or the OpenBabel-backed properties are meaningless.
    ring_bonds = [
        b for b in traj.top.bonds if set(b.indices) <= set(BENZENE_C + BENZENE_H)
    ]
    assert len(ring_bonds) == 12  # 6 C-C + 6 C-H


@pytest.mark.parametrize("name", sorted(SUPPORTED_FEATURES))
def test_every_property_returns_one_float32_value_per_atom(traj, name):
    """Whatever a property means, its shape and dtype contract is the same."""
    kwargs = {"element_type": 6} if name == "atom_type" else {}
    if name == "partial_charge":
        pytest.skip("needs charges in the topology, which a PDB does not carry")
    arr = cached(traj, name, **kwargs)
    assert arr.shape == (N_ATOMS,)
    assert arr.dtype == np.float32
    assert np.isfinite(arr).all()


def test_atomic_id_is_the_atom_index(traj):
    assert np.array_equal(cached(traj, "atomic_id"), np.arange(N_ATOMS))


def test_residue_id_groups_the_four_residues(traj):
    arr = cached(traj, "residue_id")
    assert set(np.unique(arr)) == {0, 1, 2, 3}
    assert np.array_equal(arr[BENZENE_C], np.zeros(6))
    assert arr[WATER_O] == 2
    assert arr[POTASSIUM] == 3


def test_atomic_number_matches_the_topology(traj):
    expected = np.array([a.atomic_number for a in traj.top.atoms], dtype=np.float32)
    assert np.array_equal(cached(traj, "atomic_number"), expected)


def test_mass_matches_the_topology(traj):
    arr = cached(traj, "mass")
    assert arr[BENZENE_C[0]] == pytest.approx(12.01, abs=0.01)
    assert arr[BENZENE_H[0]] == pytest.approx(1.01, abs=0.01)
    assert arr[WATER_O] == pytest.approx(16.00, abs=0.01)
    assert arr[POTASSIUM] == pytest.approx(39.10, abs=0.01)


def test_radius_is_the_van_der_waals_radius(traj):
    from nearl import constants

    arr = cached(traj, "radius")
    for idx, atomic_number in (
        (BENZENE_C[0], 6),
        (BENZENE_H[0], 1),
        (WATER_O, 8),
        (POTASSIUM, 19),
    ):
        assert arr[idx] == pytest.approx(constants.VDWRADII[atomic_number], abs=1e-4)


def test_electronegativity_is_the_pauling_scale(traj):
    arr = cached(traj, "electronegativity")
    assert arr[BENZENE_C[0]] == pytest.approx(2.55, abs=0.01)
    assert arr[WATER_O] == pytest.approx(3.44, abs=0.01)
    assert arr[POTASSIUM] == pytest.approx(0.82, abs=0.01)


def test_hydrophobicity_is_zero_for_carbon_and_grows_with_polarity(traj):
    arr = cached(traj, "hydrophobicity")
    assert arr[BENZENE_C[0]] == pytest.approx(0.0, abs=1e-6)
    assert arr[WATER_O] > arr[BENZENE_H[0]] > arr[BENZENE_C[0]]


def test_uniformed_is_all_ones_by_default(traj):
    assert np.array_equal(cached(traj, "uniformed"), np.ones(N_ATOMS))


def test_uniformed_honours_a_manual_weight(traj):
    arr = cached(traj, "uniformed", manual_weight=2.5)
    assert np.array_equal(arr, np.full(N_ATOMS, 2.5, dtype=np.float32))


def test_heavy_atom_excludes_hydrogen(traj):
    arr = cached(traj, "heavy_atom")
    numbers = np.array([a.atomic_number for a in traj.top.atoms])
    assert np.array_equal(arr, (numbers != 1).astype(np.float32))


def test_aromaticity_marks_the_benzene_carbons_only(traj):
    arr = cached(traj, "aromaticity")
    assert np.array_equal(np.flatnonzero(arr), BENZENE_C)


def test_ring_marks_the_benzene_carbons_only(traj):
    arr = cached(traj, "ring")
    assert np.array_equal(np.flatnonzero(arr), BENZENE_C)


def test_hybridization_is_sp2_for_aromatic_carbon_and_sp3_for_water(traj):
    arr = cached(traj, "hybridization")
    assert np.all(arr[BENZENE_C] == 2)
    assert arr[WATER_O] == 3


def test_hbond_donor_and_acceptor_pick_out_the_water_oxygen(traj):
    donor = cached(traj, "hbond_donor")
    acceptor = cached(traj, "hbond_acceptor")
    assert donor[WATER_O] == 1
    assert acceptor[WATER_O] == 1
    # An aromatic carbon is neither, and the bare ion cannot donate.
    assert donor[BENZENE_C[0]] == 0
    assert acceptor[BENZENE_C[0]] == 0
    assert donor[POTASSIUM] == 0


def test_backboneness_marks_the_alanine_backbone(traj):
    """
    Regression test: backboneness used to invert its own result, which made it
    identical to sidechainness and marked ligand atoms as backbone.
    """
    arr = cached(traj, "backboneness")
    for idx in ALA_BACKBONE:
        assert arr[idx] == 1, f"atom {idx} should be backbone"
    assert arr[ALA_CB] == 0
    assert np.all(arr[BENZENE_C] == 0)
    assert arr[POTASSIUM] == 0


def test_backboneness_and_sidechainness_are_complementary(traj):
    back = cached(traj, "backboneness")
    side = cached(traj, "sidechainness")
    assert np.array_equal(back, 1.0 - side)


def test_sidechainness_marks_the_alanine_side_chain(traj):
    arr = cached(traj, "sidechainness")
    assert arr[ALA_CB] == 1
    for idx in ALA_BACKBONE:
        assert arr[idx] == 0


def test_atom_type_selects_the_requested_element(traj):
    arr = cached(traj, "atom_type", element_type=6)
    numbers = np.array([a.atomic_number for a in traj.top.atoms])
    assert np.array_equal(arr, (numbers == 6).astype(np.float32))


def test_atom_type_requires_the_element(traj):
    with pytest.raises(ValueError, match="focus element"):
        cached(traj, "atom_type")


def test_partial_charge_reports_an_all_zero_topology(traj):
    """A PDB carries no charges; the failure has to be explicit, not silent zeros."""
    with pytest.raises(ValueError, match="charge"):
        cached(traj, "partial_charge")
