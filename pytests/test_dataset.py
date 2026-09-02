import h5py
import numpy as np

from nearl.io.dataset import readlabel


def test_readlabel_roundtrip(tmp_path):
  path = tmp_path / "data.h5"
  with h5py.File(path, "w") as f:
    f.create_dataset("label", data=np.array([1.0, 2.0, 3.0]))
  assert readlabel(str(path), 1) == 2.0
