# 3D Convolutional Neural Network is adoped from atom3d package
# https://github.com/drorlab/atom3d/blob/master/atom3d/models/cnn.py

import torch
import torch.nn as nn
from collections import OrderedDict

class Atom3DNetwork(nn.Module):
  """Base 3D convolutional neural network architecture for molecular data, consisting of six layers of 3D convolutions, each followed by batch normalization, ReLU activation, and optionally dropout.
  This network uses strided convolutions to downsample the input twice by half, so the original box size must be divisible by 4 (e.g. an atomic environment of side length 20 Å with 1 Å voxels). The final convolution reduces the 3D box to a single 1D vector of length ``out_dim``.
  The desired input and output dimensionality must be specified when instantiating the model.

  :param in_dim: Input dimension.
  :type in_dim: int
  :param out_dim: Output dimension.
  :type out_dim: int
  :param box_size: Size (edge length) of voxelized 3D cube.
  :type box_size: int
  :param hidden_dim: Base number of hidden units, defaults to 64
  :type hidden_dim: int, optional
  :param dropout: Dropout probability, defaults to 0.1
  :type dropout: float, optional
  """
  def __init__(self, input_channel_number: int, output_dimension: int, input_shape:int, 
               hidden_dim=64, dropout=0.1): 
    super(Atom3DNetwork, self).__init__()
    self.n_classes = output_dimension
    if isinstance(input_shape, int):
      self.input_shape = (input_shape, input_shape, input_shape)
    elif isinstance(input_shape, (tuple, list)):
      self.input_shape = tuple([int(i) for i in input_shape][:3])
    elif "__iter__" in dir(input_shape):
      self.input_shape = tuple([int(i) for i in input_shape][:3])
    else:
      raise ValueError("input_shape should be a tuple or list of 3 integers")

    self.out_dim = output_dimension
    conv = [hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8, hidden_dim * 16]
    kernel_sizes = [4, 3, 4, 3, 3]
    strides = [2, 1, 2, 1, 1]
    paddings = [1, 1, 1, 1, 1]
    conv_layers = OrderedDict()
    for i in range(len(conv)): 
      if i == 0:
        conv_layers[f'conv{i:d}'] = nn.Conv3d(input_channel_number, conv[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], bias=False)
      else:
        conv_layers[f'conv{i:d}'] = nn.Conv3d(conv[i-1], conv[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], bias=False)
      
      conv_layers[f'bn{i:d}'] = nn.BatchNorm3d(conv[i])
      conv_layers[f'relu{i:d}'] = nn.ReLU(inplace=True)
      if dropout > 0:
        conv_layers[f'dropout{i:d}'] = nn.Dropout(dropout)
    self.conv_blocks = nn.Sequential(conv_layers)

    final_kenel_size = self.input_shape[0] // 4
    self.output_layer = nn.Conv3d(conv[-1], self.n_classes, kernel_size=final_kenel_size, stride=1, padding=0, bias=False)

  def forward(self, data):
    """
    Forward method.
    :param input: Input data, as voxelized 3D cube of shape (batch_size, in_dim, box_size, box_size, box_size).
    :type input: torch.FloatTensor
    :return: Output of network, of shape (batch_size, out_dim)
    :rtype: torch.FloatTensor
    """
    bs = data.size()[0]
    data = self.conv_blocks(data)
    data = self.output_layer(data)
    # Flatten the output to a 2D tensor
    return data.view(bs, self.out_dim) 


