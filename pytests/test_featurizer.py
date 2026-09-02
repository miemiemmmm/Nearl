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
