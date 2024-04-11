# An Pytorch implementation of the pafnucy model
# The original model was in outdated TensorFlow 1.x.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math



class Conv3DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, conv_patch=5, pool_patch=2):
    super(Conv3DBlock, self).__init__()
    self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=conv_patch, padding='same')
    self.pool = nn.MaxPool3d(kernel_size=pool_patch, stride=pool_patch, padding=0)

  def forward(self, x):
    x = F.relu(self.conv(x))
    x = self.pool(x)
    return x


class FullyConnectedLayer(nn.Module):
  def __init__(self, in_features, out_features, keep_prob=1.0):
    super(FullyConnectedLayer, self).__init__()
    self.fc = nn.Linear(in_features, out_features)
    self.keep_prob = keep_prob

  def forward(self, x):
    x = F.relu(self.fc(x))
    x = F.dropout(x, p=1 - self.keep_prob)  # Note: p is the dropout probability
    return x


class PafnucyNetwork(nn.Module):
  def __init__(self, 
               input_channel_number, 
               output_dimension, 
               box_dim, 
               conv_patch = 5, 
               pool_patch = 2, 
               conv_channels = [32, 64, 128],
               dense_sizes = [1024, 512, 256],
               keep_prob = 0.9 ):
    super(PafnucyNetwork, self).__init__()

    conv_layers = OrderedDict()
    for i, num_channels in enumerate(conv_channels):
      if i == 0:
        conv_layers[f"conv_{i}"] = Conv3DBlock(input_channel_number, num_channels, conv_patch, pool_patch)
      else:
        conv_layers[f"conv_{i}"] = Conv3DBlock(conv_channels[i-1], num_channels, conv_patch, pool_patch)
    self.conv_blocks = nn.Sequential(conv_layers)

    
    dummpy_out = self.conv_blocks(torch.rand(1, input_channel_number, box_dim, box_dim, box_dim))
    conv_outsize = 1
    for n in dummpy_out.size()[1:]:
      conv_outsize *= n

    fc_layers = OrderedDict()
    for i, size in enumerate(dense_sizes):
      if i == 0:
        fc_layers[f"fc_{i}"] = FullyConnectedLayer(conv_outsize, size, keep_prob)
      else:
        fc_layers[f"fc_{i}"] = FullyConnectedLayer(dense_sizes[i-1], size, keep_prob)
    self.fc_layers = nn.Sequential(fc_layers)
    
    # self.fc_layers = nn.ModuleList()
    # total_conv_output_size = self.get_conv_output_size(input_channel_number, box_dim)
    # print(total_conv_output_size)
    # print(conv_channels[-1] * math.floor(box_dim / (pool_patch ** len(conv_channels))) ** 3)
    # total_conv_output_size = conv_channels[-1] * math.floor(isize / (pool_patch ** len(conv_channels))) ** 3
    # all_sizes = [total_conv_output_size] + dense_sizes
    # for i in range(len(all_sizes) - 1):
    #   self.fc_layers.append(FullyConnectedLayer(all_sizes[i], all_sizes[i+1], keep_prob))
    self.keep_prob = keep_prob
    self.output_layer = nn.Linear(dense_sizes[-1], output_dimension)

  # def get_conv_output_size(self, input_channel_number, isize):
  #   dummy_tensor = torch.rand(1, input_channel_number, isize, isize, isize)
  #   dummy_tensor = self.conv_blocks(dummy_tensor)
  #   return dummy_tensor.data.view(1, -1).size(1)

  def forward(self, x):
    x = self.conv_blocks(x)
    x = x.view(x.size(0), -1)
    x = self.fc_layers(x)
    x = self.output_layer(x)
    return x
