import pytest
import torch

from nearl.models.model_rfscore import RFScore
from nearl.models.model_resnet3d import ResNet


def test_rfscore_save_load_roundtrip(tmp_path):
  model = RFScore()
  path = tmp_path / "model.joblib"
  model.save(str(path))
  loaded = RFScore.load(str(path))
  assert type(loaded).__name__ == "RFScore"
  assert loaded.n_estimators == model.n_estimators


def test_downsample_basic_block_cpu():
  x = torch.randn(2, 4, 8, 8, 8)
  out = ResNet._downsample_basic_block(None, x, planes=8, stride=1)
  assert not out.is_cuda
  assert out.shape == (2, 8, 8, 8, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
def test_downsample_basic_block_cuda_float16():
  # Regression: the old isinstance(out.data, torch.cuda.FloatTensor) check
  # only matched float32 CUDA tensors, so a float16 CUDA tensor left
  # zero_pads on CPU and crashed torch.cat with a device mismatch.
  x = torch.randn(2, 4, 8, 8, 8, dtype=torch.float16, device="cuda")
  out = ResNet._downsample_basic_block(None, x, planes=8, stride=1)
  assert out.is_cuda
  assert out.shape == (2, 8, 8, 8, 8)
