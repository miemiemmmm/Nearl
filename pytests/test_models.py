import pytest
import torch

from nearl.models.model_rfscore import RFScore
from nearl.models.model_resnet3d import ResNet, generate_model
from nearl.models.model_atom3d import Atom3DNetwork
from nearl.models.model_deeprank import DeepRankNetwork
from nearl.models.model_gnina import GninaNetwork2017, GninaNetwork2018, GninaNetworkDense
from nearl.models.model_kdeep import KDeepNetwork
from nearl.models.model_pafnucy import PafnucyNetwork
from nearl.models.model_voxnet import VoxNet

INPUT_CHANNELS = 4
GRID_SIZE = 32
BATCH_SIZE = 2
OUTPUT_DIM = 1


@pytest.fixture
def input_tensor():
  return torch.randn(BATCH_SIZE, INPUT_CHANNELS, GRID_SIZE, GRID_SIZE, GRID_SIZE)


@pytest.mark.parametrize("model_cls", [
  Atom3DNetwork,
  DeepRankNetwork,
  GninaNetwork2017,
  GninaNetwork2018,
  KDeepNetwork,
  PafnucyNetwork,
  VoxNet,
])
def test_model_forward_pass_shape(model_cls, input_tensor):
  model = model_cls(INPUT_CHANNELS, OUTPUT_DIM, GRID_SIZE)
  out = model(input_tensor)
  assert out.shape == (BATCH_SIZE, OUTPUT_DIM)


def test_gnina_network_dense_forward_pass_shape(input_tensor):
  model = GninaNetworkDense([INPUT_CHANNELS])
  out = model(input_tensor)
  assert out.shape == (BATCH_SIZE, OUTPUT_DIM)


def test_resnet3d_forward_pass_shape(input_tensor):
  model = generate_model(10, n_input_channels=INPUT_CHANNELS, n_classes=OUTPUT_DIM)
  out = model(input_tensor)
  assert out.shape == (BATCH_SIZE, OUTPUT_DIM)


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
