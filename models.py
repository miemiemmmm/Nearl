import numpy as np
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Create a random forest regressor.
class rfscore(RandomForestRegressor):
  def __init__(self, n_jobs=-1):
    mtry = 21
    super().__init__(
      n_estimators = 500,
      oob_score=True,
      n_jobs=n_jobs,
      max_features=mtry,
      bootstrap=True,
      min_samples_split=2,
      # min_samples_leaf=1,
      max_depth=20,
      # *args,
      # **kwargs
    )
  def save(self, filename):
    joblib.dump(self, filename)
  def load(self, filename):
    self = joblib.load(filename)




class SimpleNetwork(nn.Module):
  def __init__(self, input_dim=36, output_dim=1):
    super(SimpleNetwork, self).__init__()
    self.layer1 = nn.Linear(input_dim, 128)  # First hidden layer with 128 neurons
    self.layer2 = nn.Linear(128, 256)        # Second hidden layer with 64 neurons
    self.layer3 = nn.Linear(256, 64)         # Second hidden layer with 64 neurons
    self.layer4 = nn.Linear(64, 32)          # Third hidden layer with 32 neurons
    self.layer5 = nn.Linear(32, 16)          # Fourth hidden layer with 16 neurons
    self.output_layer = nn.Linear(16, output_dim)  # Output layer

  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = F.relu(self.layer3(x))
    x = F.relu(self.layer4(x))
    x = F.relu(self.layer5(x))
    x = self.output_layer(x)
    return x


class CNN2D(nn.Module):
  def __init__(self):
    super(CNN2D, self).__init__()
    self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1 = nn.Linear(64 * 8 * 8, 128)
    self.fc2 = nn.Linear(128, 1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)

    x = x.view(-1, 64 * 8 * 8)

    x = F.relu(self.fc1(x))
    output = self.fc2(x)

    return output

class CNN3D(nn.Module):
  def __init__(self):
    super(CNN3D, self).__init__()
    # First 3D convolutional layer, takes in a 32x32x32 image
    self.conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm3d(32)
    # Second 3D convolutional layer
    self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm3d(64)
    # 3D max pooling layer
    self.pool = nn.MaxPool3d(2, 2)
    # Fully connected layers
    self.fc1 = nn.Linear(64 * 8 * 8 * 8, 256)
    self.fc2 = nn.Linear(256, 1)

  def forward(self, x):
    # Apply first convolution, followed by ReLU, then max pooling
    x = self.pool(F.relu(self.bn1(self.conv1(x))))
    # Second convolution, followed by ReLU, then max pooling
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    # Flatten the tensor
    x = x.view(x.size(0), -1)
    # Fully connected layers
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x







