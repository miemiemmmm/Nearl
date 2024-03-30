# The following network models are implemented based on the 
# https://github.com/gnina/models/tree/master/acs2018

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class GninaNetwork2017(nn.Module):
  def __init__(self, 
    input_channel_number: int, 
    output_dimension: int, 
    box_shape: int
  ):
    super(GninaNetwork2017, self).__init__()
    
    conv_layers = OrderedDict()
    conv_layers["conv1"] = nn.Conv3d(input_channel_number, 32, kernel_size=3, padding=1)
    conv_layers["relu1"] = nn.ReLU(inplace=True)
    conv_layers["pool1"] = nn.MaxPool3d(2, stride=2)

    conv_layers["conv2"] = nn.Conv3d(32, 64, kernel_size=3, padding=1)
    conv_layers["relu2"] = nn.ReLU(inplace=True)
    conv_layers["pool2"] = nn.MaxPool3d(2, stride=2)

    conv_layers["conv3"] = nn.Conv3d(64, 128, kernel_size=3, padding=1)
    conv_layers["relu3"] = nn.ReLU(inplace=True)
    conv_layers["pool3"] = nn.MaxPool3d(2, stride=2)


    # Define the convolutional blocks
    self.conv_blocks = nn.Sequential(conv_layers)
    
    # Assuming a dummy input to compute the size of the flattened features
    with torch.no_grad():
      dummy_input = torch.zeros(1, input_channel_number, box_shape, box_shape, box_shape)
      dummy_output = self.conv_blocks(dummy_input)
      flattened_size = int(torch.prod(torch.tensor(dummy_output.shape[1:])))
    
    # Define the fully connected layers for pose and affinity predictions
    self.pose_output = nn.Linear(flattened_size, output_dimension)
    self.affinity_output = nn.Linear(flattened_size, 1)
  
  def forward(self, x):
    x = self.conv_blocks(x)
    
    # Flatten the output for the fully connected layer
    x = x.view(x.size(0), -1)
    
    # Pose and affinity predictions
    pose_pred = F.softmax(self.pose_output(x), dim=1)
    affinity_pred = self.affinity_output(x)
    
    return pose_pred, affinity_pred


class GninaNetwork2017_FiveLayers(nn.Module):
  def __init__(self, 
    input_channel_number: int, 
    output_dimension: int,
    box_shape: int
  ):
    super(GninaNetwork2017_FiveLayers, self).__init__()

    conv_layers = OrderedDict()
    conv_layers["conv1"] = nn.Conv3d(input_channel_number, 32, kernel_size=3, padding=1)
    conv_layers["relu1"] = nn.ReLU(inplace=True)
    conv_layers["pool1"] = nn.MaxPool3d(2, stride=2)

    conv_layers["conv2"] = nn.Conv3d(32, 64, kernel_size=3, padding=1)
    conv_layers["relu2"] = nn.ReLU(inplace=True)
    conv_layers["pool2"] = nn.MaxPool3d(2, stride=2)

    conv_layers["conv3"] = nn.Conv3d(64, 128, kernel_size=3, padding=1)
    conv_layers["relu3"] = nn.ReLU(inplace=True)
    conv_layers["pool3"] = nn.MaxPool3d(2, stride=2)

    conv_layers["conv4"] = nn.Conv3d(128, 256, kernel_size=3, padding=1)
    conv_layers["relu4"] = nn.ReLU(inplace=True)
    conv_layers["pool4"] = nn.MaxPool3d(2, stride=2)

    conv_layers["conv5"] = nn.Conv3d(256, 512, kernel_size=3, padding=1)
    conv_layers["relu5"] = nn.ReLU(inplace=True)
    conv_layers["pool5"] = nn.MaxPool3d(2, stride=2)
    self.conv_blocks = nn.Sequential(conv_layers)

    # Compute the flattened size after the convolutional and pooling layers
    with torch.no_grad():
      dummpy_out = self.conv_blocks(torch.rand(1, input_channel_number, box_shape, box_shape, box_shape))
      flattened_size = 1 
      for n in dummpy_out.size()[1:]:
        flattened_size *= n

    self.fc = nn.Linear(flattened_size, output_dimension) 

  def forward(self, x):
    x = self.conv_blocks(x)
    # Flatten the output for the fully connected layer
    x = x.view(x.size(0), -1)
    # Fully connected layer for binary classification
    x = self.fc(x)
    return F.softmax(x, dim=1)


class GninaNetwork2018(nn.Module):
  def __init__(self, input_channels, output_dimension, box_shape):
    super(GninaNetwork2018, self).__init__()
    conv_layers = OrderedDict()
    conv_layers["poolavg1"] = nn.AvgPool3d(2, stride=2)
    conv_layers["conv1"] = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1, stride=1)
    conv_layers["relu1"] = nn.ReLU()
    conv_layers["conv_deepen1"] = nn.Conv3d(32, 32, kernel_size=1, stride=1)
    conv_layers["relu_deepen1"] = nn.ReLU()
    conv_layers["poolavg2"] = nn.AvgPool3d(2, stride=2)
    conv_layers["conv2"] = nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=1)
    conv_layers["relu2"] = nn.ReLU()
    conv_layers["conv_deepen2"] = nn.Conv3d(64, 64, kernel_size=1, stride=1)
    conv_layers["relu_deepen2"] = nn.ReLU()
    conv_layers["poolavg3"] = nn.AvgPool3d(2, stride=2)
    conv_layers["conv3"] = nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=1)
    conv_layers["relu3"] = nn.ReLU()
    self.conv_blocks = nn.Sequential(conv_layers)
    
    dummy_input = torch.zeros(1, input_channels, box_shape, box_shape, box_shape)
    dummy_output = self.conv_blocks(dummy_input)
    flattened_feature_size = int(torch.prod(torch.tensor(dummy_output.shape[1:])))
    
    self.pose_output = nn.Linear(flattened_feature_size, 2)
    self.affinity_output = nn.Linear(flattened_feature_size, 1)
    
  def forward(self, x):
    x = self.conv_blocks(x)
    x = torch.flatten(x, 1)
    pose = self.pose_output(x)
    affinity = self.affinity_output(x)
    # Apply softmax to pose for classification
    pose = F.softmax(pose, dim=1)
    
    return pose, affinity


class GninaNetworkDense(nn.Module):
  def __init__(self, dims, num_dense_blocks=3, num_dense_filters=16, num_dense_convs=4):
    super(GninaNetworkDense, self).__init__()

    self.modules = []
    in_feats = dims[0]
    out_feats = 32

    self.data_enc_init_pool = nn.MaxPool3d(2, stride=2)
    self.data_enc_init_conv = nn.Conv3d(in_feats, out_feats, 3, padding=1)

    for idx in range(num_dense_blocks - 1):
      in_feats = out_feats
      dense_block = DenseBlock(in_feats, idx, num_dense_convs, num_dense_filters)
      self.add_module(f"dense_block_{idx}", dense_block)
      self.modules.append(dense_block)
      out_feats = num_dense_convs * num_dense_filters + in_feats
      
      bottleneck = nn.Conv3d(out_feats, out_feats, 1, padding=0)
      self.add_module(f"data_enc_level{idx}_bottleneck", bottleneck)
      self.modules.append(bottleneck)

      max_pool = nn.MaxPool3d(2, stride=2)
      self.add_module(f"data_enc_level{idx+1}_pool", max_pool)
      self.modules.append(max_pool)

    in_feats = out_feats
    dense_block = DenseBlock(in_feats, num_dense_blocks-1, num_dense_convs, num_dense_filters)
    out_feats = num_dense_convs * num_dense_filters + in_feats
    self.add_module(f"dense_block_{num_dense_blocks-1}", dense_block)
    self.modules.append(dense_block)

    global_pool = GlobalMaxPool()
    self.add_module(f"data_enc_level2_global_pool", global_pool)
    self.modules.append(global_pool)
    self.affinity_output = nn.Linear(out_feats,1)
    self.pose_output = nn.Linear(out_feats,2)

  def forward(self,x):
    x = self.data_enc_init_pool(x) 
    x = F.relu(self.data_enc_init_conv(x))
    for module in self.modules:
      x = module(x)
      if isinstance(module, nn.Conv3d):
        x = F.relu(x)
    
    affinity = self.affinity_output(x)
    pose = F.softmax(self.pose_output(x),dim=1)[:,1]
    return pose, affinity

class DenseBlock(nn.Module):
  def __init__(self, input_feats, level, convolutions=4, num_dense_filters=16):
    super().__init__()
    self.modules = []
    current_feats = input_feats
    for idx in range(convolutions):
      bn = nn.BatchNorm3d(current_feats)
      self.add_module(f"data_enc_level{level}_batchnorm_conv{idx}", bn)
      self.modules.append(bn)

      conv = nn.Conv3d(current_feats, num_dense_filters, kernel_size=3, padding=1)
      self.add_module(f"data_enc_level{level}_conv{idx}", conv)
      self.modules.append(conv)
      current_feats += num_dense_filters

  def forward(self, x):
    previous = []
    previous.append(x)
    for module in self.modules:
      x = module(x)
      if isinstance(module, nn.Conv3d):
        x = F.relu(x)
        previous.append(x)
        x = torch.cat(previous, 1)
    return x

class GlobalMaxPool(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return F.max_pool3d(x, kernel_size=x.size()[2:]).view(*x.size()[:2])
  








