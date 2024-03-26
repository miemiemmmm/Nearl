# An Pytorch implementation of the pafnucy model
# The original model was in outdated TensorFlow 1.x.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class SBNetwork(nn.Module):
    def __init__(self, isize, in_chnls, osize, conv_patch, pool_patch, conv_channels, dense_sizes, keep_prob=1.0):
        super(SBNetwork, self).__init__()
        self.conv_blocks = nn.ModuleList([Conv3DBlock(in_chnls if i == 0 else conv_channels[i-1], 
                                                      num_channels, 
                                                      conv_patch, 
                                                      pool_patch) for i, num_channels in enumerate(conv_channels)])
        
        self.fc_layers = nn.ModuleList()
        total_conv_output_size = conv_channels[-1] * math.floor(isize / (pool_patch ** len(conv_channels))) ** 3
        all_sizes = [total_conv_output_size] + dense_sizes
        for i in range(len(all_sizes) - 1):
            self.fc_layers.append(FullyConnectedLayer(all_sizes[i], all_sizes[i+1], keep_prob))
        self.output_layer = nn.Linear(dense_sizes[-1], osize)

    def forward(self, x, keep_prob=1.0):
        for conv in self.conv_blocks:
            x = conv(x)
        
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor for the fully connected layers
        
        for fc in self.fc_layers:
            x = fc(x)
        
        x = self.output_layer(x)
        return x
