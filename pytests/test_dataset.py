import h5py
import numpy as np
import pytest

from nearl.io.dataset import Dataset, readdata, readlabel, split_array


def test_readlabel_roundtrip(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("label", data=np.array([1.0, 2.0, 3.0]))
    assert readlabel(str(path), 1) == 2.0


def test_readdata_minmax_normalize(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("feat", data=np.array([0.0, 5.0, 10.0]))
    result = readdata(str(path), "feat", slice(None), normalize="minmax")
    assert np.allclose(result, [0.0, 0.5, 1.0])


def test_readdata_zscore_normalize(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("feat", data=np.array([1.0, 2.0, 3.0]))
    result = readdata(str(path), "feat", slice(None), normalize="zscore")
    assert np.isclose(np.mean(result), 0.0)


def test_readdata_normalize_constant_array_returns_zeros(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("feat", data=np.array([3.0, 3.0, 3.0]))
    minmax_result = readdata(str(path), "feat", slice(None), normalize="minmax")
    zscore_result = readdata(str(path), "feat", slice(None), normalize="zscore")
    assert np.allclose(minmax_result, 0.0)
    assert np.allclose(zscore_result, 0.0)


def test_split_array_exact_division():
    batches = split_array(np.arange(10), 5)
    assert [len(b) for b in batches] == [5, 5]


def test_split_array_remainder_batch():
    batches = split_array(np.arange(7), 3)
    assert [len(b) for b in batches] == [3, 3, 1]


def test_split_array_batch_size_larger_than_array():
    batches = split_array(np.arange(3), 10)
    assert [len(b) for b in batches] == [3]


def test_dataset_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Dataset(
            [str(tmp_path / "does_not_exist.h5")], grid_dim=8, feature_keys=["feat"]
        )


def test_dataset_missing_feature_key_raises_key_error(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("label", data=np.array([1.0, 2.0]))
    with pytest.raises(KeyError):
        Dataset([str(path)], grid_dim=8, feature_keys=["missing_feature"])
