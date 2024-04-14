# This code block is from DeepRank2, an open-source deep learning (DL) framework for data 
# mining of protein-protein interfaces (PPIs) or single-residue variants (SRVs).
# https://github.com/DeepRank/deeprank2/blob/main/deeprank2/neuralnets/cnn/model3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

# ruff: noqa: ANN001, ANN201, ANN202
######################################################################
#
# Model automatically generated by modelGenerator
#
######################################################################

# ----------------------------------------------------------------------
# Network Structure
# ----------------------------------------------------------------------
# conv layer   0: conv | input -1  output  4  kernel  2  post relu
# conv layer   1: pool | kernel  2  post None
# conv layer   2: conv | input  4  output  5  kernel  2  post relu
# conv layer   3: pool | kernel  2  post None
# fc   layer   0: fc   | input -1  output  84  post relu
# fc   layer   1: fc   | input  84  output  1  post None
# ----------------------------------------------------------------------


class DeepRankNetwork(torch.nn.Module):  # noqa: D101
  def __init__(self, 
               input_channel_number: int, 
               output_dimension: int,
               box_shape:int, 
               ):
    super(DeepRankNetwork, self).__init__()
    if isinstance(box_shape, int):
      box_shape = (box_shape, box_shape, box_shape)
    elif isinstance(box_shape, (tuple, list, np.ndarray)):
      box_shape = tuple(box_shape)[:3]

    self.convlayer_000 = torch.nn.Conv3d(input_channel_number, 4, kernel_size=2)
    self.convlayer_001 = torch.nn.MaxPool3d((2, 2, 2))
    self.convlayer_002 = torch.nn.Conv3d(4, 5, kernel_size=2)
    self.convlayer_003 = torch.nn.MaxPool3d((2, 2, 2))

    size = self._get_conv_output(input_channel_number, box_shape)

    self.fclayer_000 = torch.nn.Linear(size, 84)
    self.fclayer_001 = torch.nn.Linear(84, output_dimension)

  def _get_conv_output(self, num_features: int, shape: tuple[int]):
    num_data_points = 1
    input_ = Variable(torch.rand(num_data_points, num_features, *shape))
    output = self._forward_features(input_)
    return output.data.view(num_data_points, -1).size(1)

  def _forward_features(self, x):
    x = F.relu(self.convlayer_000(x))
    x = self.convlayer_001(x)
    x = F.relu(self.convlayer_002(x))
    x = self.convlayer_003(x)
    return x  # noqa:RET504 (unnecessary-assign)

  def forward(self, data):
    x = self._forward_features(data)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fclayer_000(x))
    x = self.fclayer_001(x)
    return x  # noqa:RET504 (unnecessary-assign)


######################################################################
#
# Model automatically generated by modelGenerator
#
######################################################################

# ----------------------------------------------------------------------
# Network Structure
# ----------------------------------------------------------------------
# conv layer   0: conv | input -1  output  4  kernel  2  post relu
# conv layer   1: pool | kernel  2  post None
# conv layer   2: conv | input  4  output  5  kernel  2  post relu
# conv layer   3: pool | kernel  2  post None
# fc   layer   0: fc   | input -1  output  84  post relu
# fc   layer   1: fc   | input  84  output  1  post None
# ----------------------------------------------------------------------


class CnnClassification(torch.nn.Module):  # noqa: D101
  def __init__(self, num_features, box_shape):
    super().__init__()

    self.convlayer_000 = torch.nn.Conv3d(num_features, 4, kernel_size=2)
    self.convlayer_001 = torch.nn.MaxPool3d((2, 2, 2))
    self.convlayer_002 = torch.nn.Conv3d(4, 5, kernel_size=2)
    self.convlayer_003 = torch.nn.MaxPool3d((2, 2, 2))

    size = self._get_conv_output(num_features, box_shape)

    self.fclayer_000 = torch.nn.Linear(size, 84)
    self.fclayer_001 = torch.nn.Linear(84, 2)

  def _get_conv_output(self, num_features, shape):
    inp = Variable(torch.rand(1, num_features, *shape))
    out = self._forward_features(inp)
    return out.data.view(1, -1).size(1)

  def _forward_features(self, x):
    x = F.relu(self.convlayer_000(x))
    x = self.convlayer_001(x)
    x = F.relu(self.convlayer_002(x))
    x = self.convlayer_003(x)
    return x  # noqa:RET504 (unnecessary-assign)

  def forward(self, data):
    x = self._forward_features(data.x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fclayer_000(x))
    x = self.fclayer_001(x)
    return x  # noqa:RET504 (unnecessary-assign)
    