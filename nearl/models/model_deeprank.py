import torch
import torch.nn as nn
from collections import OrderedDict


class DeepRankNetwork(nn.Module): 
  """
  DeepRank network model. 

  Notes
  -----
  This code block is adopted from DeepRank2: 

  https://github.com/DeepRank/deeprank2/blob/main/deeprank2/neuralnets/cnn/model3d.py
  """
  def __init__(self, input_channels: int, output_dimension: int, input_shape:int, 
               conv=[4, 5], fc = [84]):
    super(DeepRankNetwork, self).__init__()
    if isinstance(input_shape, int):
      self.input_shape = (input_shape, input_shape, input_shape)
    elif isinstance(input_shape, (tuple, list)):
      self.input_shape = tuple([int(i) for i in input_shape][:3])
    elif "__iter__" in dir(input_shape):
      self.input_shape = tuple([int(i) for i in input_shape][:3])
    else:
      raise ValueError("input_shape should be a tuple or list of 3 integers")

    conv_layers = OrderedDict()
    for i in range(len(conv)):
      if i == 0:
        conv_layers[f'conv{i}'] = nn.Conv3d(input_channels, conv[i], kernel_size=2)
      else: 
        conv_layers[f'conv{i}'] = nn.Conv3d(conv[i-1], conv[i], kernel_size=2)
      conv_layers[f'relu{i}'] = nn.ReLU()  
      conv_layers[f'pool{i}'] = nn.MaxPool3d((2, 2, 2))
    self.conv_blocks = torch.nn.Sequential(conv_layers)

    dummpy_out = self.conv_blocks(torch.rand(1, input_channels, *self.input_shape))
    size = dummpy_out.flatten().size()[0]

    fc_layers = OrderedDict()
    for i in range(len(fc)):
      if i == 0:
        fc_layers[f'fc_{i:d}'] = nn.Linear(size, fc[i])
      else:
        fc_layers[f'fc_{i:d}'] = nn.Linear(fc[i-1], fc[i])
      fc_layers[f'relu_{i:d}'] = nn.ReLU()
    self.fc_layers = torch.nn.Sequential(fc_layers)
    self.output_layer = torch.nn.Linear(fc[-1], output_dimension)

  def forward(self, data):
    data = self.conv_blocks(data)
    data = data.view(data.size(0), -1)
    data = self.fc_layers(data)
    data = self.output_layer(data)
    return data 
