import torch
import torch.nn as nn

class Fire_Block(nn.Module):
  def __init__(self, in_channels, sequeeze_channels):
    super(Fire_Block, self).__init__()
    self.squeeze = nn.Sequential(
      nn.Conv3d(in_channels, sequeeze_channels, kernel_size=1),
      nn.ReLU(inplace=True),
    )
    self.expand1 = nn.Sequential(
      nn.Conv3d(sequeeze_channels, sequeeze_channels*4, kernel_size=1),
      nn.ReLU(inplace=True),
    )
    self.expand2 = nn.Sequential(
      nn.Conv3d(sequeeze_channels, sequeeze_channels*4, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
    )
  def forward(self, x): 
    x = self.squeeze(x)
    expand1 = self.expand1(x)
    expand2 = self.expand2(x)
    merge = torch.cat((expand1, expand2), 1)
    return merge


class KDeepNetwork(nn.Module):
  """
  KDeep network model

  Notes
  -----
  Both MaxPool3d and AvgPool3d are used in kernel sizes of 3 and strides of 2.

  This code block is from an pytorch implementation of KDeep: 

  https://github.com/abdulsalam-bande/KDeep/blob/main/pytorch/model.py
  """
  def __init__(self, input_channels: int, output_dimension: int, input_shape):
    super(KDeepNetwork, self).__init__()
    self.n_classes = output_dimension
    if isinstance(input_shape, int):
      self.input_shape = (input_shape, input_shape, input_shape)
    elif isinstance(input_shape, (tuple, list)):
      self.input_shape = tuple([int(i) for i in input_shape][:3])
    elif "__iter__" in dir(input_shape):
      self.input_shape = tuple([int(i) for i in input_shape][:3])
    else:
      raise ValueError("input_shape should be a tuple or list of 3 integers")
    
    self.conv1 = nn.Sequential(
      nn.Conv3d(input_channels, 96, kernel_size=1, stride=2),
      nn.ReLU(inplace=True),
    )

    self.squeeze1 = nn.Sequential(
      Fire_Block(96, 16),
      Fire_Block(128, 16),
      Fire_Block(128, 32),
    )
    self.pool = nn.MaxPool3d(kernel_size=3, stride=2)

    self.squeeze2 = nn.Sequential(
      Fire_Block(256, 32), 
      Fire_Block(256, 48),
      Fire_Block(384, 48),
      Fire_Block(384, 64),
    )
    self.avg_pool = nn.AvgPool3d(kernel_size=3, stride=2)

    dummy_out = self.conv1(torch.rand(1, input_channels, *self.input_shape))
    dummy_out = self.squeeze1(dummy_out)
    dummy_out = self.pool(dummy_out)
    dummy_out = self.squeeze2(dummy_out)
    dummy_out = self.avg_pool(dummy_out)
    size = dummy_out.flatten().size()[0]

    self.dense1 = nn.Linear(size, output_dimension)

  def forward(self, x):
    # Run conv blocks
    x = self.conv1(x)
    x = self.squeeze1(x)
    x = self.pool(x)
    x = self.squeeze2(x)
    x = self.avg_pool(x)
    
    # Flatten the output and make predictions
    x = torch.flatten(x, 1)
    x = self.dense1(x)
    return x
