# Original code is from https://github.com/MonteYang/VoxNet.pytorch/blob/master/voxnet.py

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class VoxNet(nn.Module):
  def __init__(self, 
               input_channel_number: int,
               output_dimension: int, 
               input_shape:int, 
               ):
    super(VoxNet, self).__init__()
    if isinstance(input_shape, int):
      self.input_shape = (input_shape, input_shape, input_shape)
    elif isinstance(input_shape, (tuple, list, np.ndarray)):
      self.input_shape = tuple(input_shape)[:3]

    self.feat = torch.nn.Sequential(OrderedDict([
      ('conv3d_1', torch.nn.Conv3d(input_channel_number, 32, kernel_size=5, stride=2)),
      ('relu1', torch.nn.ReLU()),
      ('drop1', torch.nn.Dropout(p=0.2)),
      ('conv3d_2', torch.nn.Conv3d(32, 32, kernel_size=3)),
      ('relu2', torch.nn.ReLU()),
      ('pool2', torch.nn.MaxPool3d(2)),
      ('drop2', torch.nn.Dropout(p=0.3))
    ]))
    x = self.feat(torch.rand((1, input_channel_number) + self.input_shape))
    dim_feat = 1
    for n in x.size()[1:]:
      dim_feat *= n

    self.mlp = torch.nn.Sequential(OrderedDict([
      ('fc1', torch.nn.Linear(dim_feat, 128)),
      ('relu1', torch.nn.ReLU()),
      ('drop3', torch.nn.Dropout(p=0.4)),
      ('fc2', torch.nn.Linear(128, output_dimension))
    ]))

  def forward(self, x):
    x = self.feat(x)
    x = x.view(x.size(0), -1)
    x = self.mlp(x)
    return x

  def save_model(self, path):
    torch.save(self.state_dict(), path)

  def load_model(self, path):
    self.load_state_dict(torch.load(path))
    self.eval()

