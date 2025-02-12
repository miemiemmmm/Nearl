import torch
import torch.nn as nn
from collections import OrderedDict


class VoxNet(nn.Module):
  """
  Original code is from:

  https://github.com/MonteYang/VoxNet.pytorch/blob/master/voxnet.py
  
  Notes
  -----
  Compared with the original model, ReLU is chnaged to PReLU(1, 0.25)
  The higgen layer in the fully connected layer is changed from 128 to 1280
  Default dropout rate changed from [0.2, 0.3, 0.4] to [0.1, 0.1, 0.1]
  """
  def __init__(self, input_channels: int, output_dimension, input_shape, dropout_rates = [0.1, 0.1, 0.1]):
    super(VoxNet, self).__init__()
    self.n_classes = output_dimension
    if isinstance(input_shape, int):
      self.input_shape = (input_shape, input_shape, input_shape)
    elif isinstance(input_shape, (tuple, list)):
      self.input_shape = tuple([int(i) for i in input_shape][:3])
    elif "__iter__" in dir(input_shape):
      self.input_shape = tuple([int(i) for i in input_shape][:3])
    else:
      raise ValueError("input_shape should be a tuple or list of 3 integers")
      
    self.feat = torch.nn.Sequential(OrderedDict([
      ('conv1', torch.nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=2)), 
      ('relu1',    nn.PReLU()),
      # ('relu1',    nn.ReLU()),
      ('drop1',    torch.nn.Dropout(p=dropout_rates[0])),
      ('conv2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
      ('relu2',    nn.PReLU()),
      # ('relu2',    nn.ReLU()),
      ('pool2',    torch.nn.MaxPool3d(2)),
      ('drop2',    torch.nn.Dropout(p=dropout_rates[1])),
    ]))

    dummpy_out = self.feat(torch.rand(1, input_channels, *self.input_shape))
    size = dummpy_out.flatten().size()[0]

    self.mlp = torch.nn.Sequential(OrderedDict([
      ('fc1',   torch.nn.Linear(size, 1280)),
      ('relu1', nn.PReLU()),
      # ('relu1', nn.ReLU()),
      ('drop3', torch.nn.Dropout(p=dropout_rates[2])),
      ('fc2',   torch.nn.Linear(1280, self.n_classes))
    ]))

  def forward(self, x):
    x = self.feat(x)
    x = x.view(x.size(0), -1)
    x = self.mlp(x)
    return x

