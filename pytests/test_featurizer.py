import pytest

from nearl.features import Mass
from nearl.featurizer import Featurizer


def test_scalar_dimensions_normalize_to_3_vector():
    f = Featurizer({"dimensions": 32, "lengths": 16})
    assert f.dims.ndim == 1
    assert list(f.dims) == [32, 32, 32]


def test_list_dimensions_preserved():
    f = Featurizer({"dimensions": [16, 24, 32], "lengths": 16})
    assert list(f.dims) == [16, 24, 32]


def test_missing_dimensions_stays_none():
    f = Featurizer({"time_window": 1})
    assert f.dims is None


def test_scalar_lengths_normalizes_to_3_vector():
    f = Featurizer({"dimensions": 32, "lengths": 16})
    assert f.lengths.ndim == 1
    assert list(f.lengths) == [16.0, 16.0, 16.0]


def test_list_lengths_preserved():
    f = Featurizer({"dimensions": [16, 24, 32], "lengths": [8, 12, 16]})
    assert list(f.lengths) == [8.0, 12.0, 16.0]


def test_spacing_derived_from_lengths_path():
    f = Featurizer({"dimensions": 32, "lengths": 16})
    assert f.spacing == pytest.approx(0.5)


def test_spacing_derived_from_spacing_path():
    f = Featurizer({"dimensions": 32, "spacing": 0.5})
    assert list(f.lengths) == [16.0, 16.0, 16.0]


def test_malformed_dimensions_raises_assertion_error():
    with pytest.raises(AssertionError):
        Featurizer({"dimensions": [16, 24], "lengths": 16})


def test_register_features_rejects_duplicate_outkeys():
    f = Featurizer({"dimensions": 32, "lengths": 16, "cutoff": 3.5, "sigma": 1.5})
    f.register_feature(Mass(outkey="dup"))
    with pytest.raises(ValueError):
        f.register_feature(Mass(outkey="dup"))


def test_lengths_without_dimensions_currently_raises_type_error():
    # Known pre-existing limitation, pinned rather than fixed here: __init__
    # divides by self.dims before checking it was ever set, when only
    # "lengths" (no "dimensions") is given.
    with pytest.raises(TypeError):
        Featurizer({"lengths": 16})
