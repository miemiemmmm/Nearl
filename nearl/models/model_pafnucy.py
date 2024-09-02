# An Pytorch implementation of the pafnucy model
# The original model was in outdated TensorFlow 1.x.

import torch
import torch.nn as nn
from collections import OrderedDict

class PafnucyNetwork(nn.Module):
  """
  Pytorch implementation of the Pafnucy model.

  Original paper:
  Stepniewska-Dziubinska, M.M., Zielenkiewicz, P. and Siedlecki, P., 2018. Development and evaluation of a deep learning model for protein–ligand binding affinity prediction. Bioinformatics, 34(21), pp.3666-3674.

  Code reference:
  https://gitlab.com/cheminfIBB/tfbio/-/blob/master/tfbio/net.py

  This model uses dropout regularization for both convolutional and dense layers.


  """
  def __init__(self, input_channels, output_dimension, input_shape, 
               conv_patch = 5, pool_patch = 2, 
               conv_channels = [32, 64, 128],
               dense_sizes = [1024, 512, 256],
               drop_prob = 0.1):
    super(PafnucyNetwork, self).__init__()
    self.n_classes = output_dimension
    if isinstance(input_shape, int):
      self.input_shape = (input_shape, input_shape, input_shape)
    elif isinstance(input_shape, (tuple, list)):
      self.input_shape = tuple([int(i) for i in input_shape][:3])
    elif "__iter__" in dir(input_shape):
      self.input_shape = tuple([int(i) for i in input_shape][:3])
    else:
      raise ValueError("input_shape should be a tuple or list of 3 integers")
    
    conv_layers = OrderedDict()
    for i in range(len(conv_channels)):
      if i == 0:
        conv_layers[f"conv{i}"] = nn.Conv3d(input_channels, conv_channels[i], kernel_size=conv_patch, padding=(conv_patch//2))
      else:
        conv_layers[f"conv{i}"] = nn.Conv3d(conv_channels[i-1], conv_channels[i], kernel_size=conv_patch, padding=(conv_patch//2))
      conv_layers[f"relu{i}"] = nn.ReLU(inplace=True)
      conv_layers[f"pool{i}"] = nn.MaxPool3d(kernel_size=pool_patch, stride=pool_patch)
    self.conv_blocks = nn.Sequential(conv_layers)
    
    dummpy_out = self.conv_blocks(torch.rand(1, input_channels, *self.input_shape))
    size = dummpy_out.flatten().size()[0]

    fc_layers = OrderedDict()
    for i in range(len(dense_sizes)):
      if i == 0:
        fc_layers[f"fc{i}"] = nn.Linear(size, dense_sizes[i])
      else:
        fc_layers[f"fc{i}"] = nn.Linear(dense_sizes[i-1], dense_sizes[i])
      fc_layers[f"relu{i}"] = nn.ReLU(inplace=True)
      fc_layers[f"dropout{i}"] = nn.Dropout(p=drop_prob)
    self.fc_layers = nn.Sequential(fc_layers)
    
    self.output_layer = nn.Linear(dense_sizes[-1], output_dimension)

  def forward(self, x):
    x = self.conv_blocks(x)
    x = torch.flatten(x, 1)
    x = self.fc_layers(x)
    x = self.output_layer(x)
    return x
