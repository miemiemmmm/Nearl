import time

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

from BetaPose import models, printit, data_io, utils


printit("Loading data...")

input_files = [
  "/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_trainset_randomforest.h5",
  "/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_randomforest.h5",
  "/media/yzhang/MieT5/BetaPose/data/trainingdata/pdbbindrefined_v2016_randomforest.h5",
  "/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_randomforest_step10.h5",
  "/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_testset_randomforest.h5",
]

rf_data = [];
label_data = [];
for input_hdfile in input_files:
  with data_io.hdf_operator(input_hdfile, "r") as h5file:
    rf_data.append(h5file.data("rf"))
    label_data.append(h5file.data("label").ravel())
rf_training_data = np.concatenate(rf_data, axis=0)
label_training_data = np.concatenate(label_data, axis=0)
print(f"Training dataset: {rf_training_data.shape} ; Label number: {len(label_training_data)}");

# Load test dataset;
testset_file = "/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_testset_randomforest.h5"
with data_io.hdf_operator(testset_file, "r") as h5file:
  h5file.draw_structure()
  rf_testset = h5file.data("rf")
  label_testset = h5file.data("label").ravel()


printit("Data loaded!!! Good luck!!!");


rf_training_data = np.random.normal(size=(50000, 3, 32, 32));
label_training_data = np.random.normal(size=(50000, 1));

rf_testset = np.random.normal(size=(100, 3, 32,32));
label_testset = np.random.normal(size=(100,1));

print("Training data shapes: ", rf_training_data.shape, label_training_data.shape)
print("Test data shapes: ", rf_testset.shape, label_testset.shape)

##########################################################################################
##########################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

model = models.CNN2D()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 15
batch_size = 256

training_data = torch.from_numpy(rf_training_data).to(dtype=torch.float32)
label_training_data = torch.from_numpy(label_training_data).to(dtype=torch.float32)
print("Input data shape: ", training_data.size(), "Label shape: ", label_training_data.size())
tensor_dataset = TensorDataset(training_data, label_training_data)

# Create a DataLoader. You can tweak the batch_size as needed
data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
st = time.perf_counter();
for epoch in range(epochs):
  print(f"Epoch {epoch+1}\n------------------------------------------------")
  running_loss = 0.0
  for i, data in enumerate(data_loader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)

    loss = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if ((i+1) % 100) == 0:
      print(f"Epoch{epoch+1}, batch{i+1:5d}: running loss: {running_loss:.3f}");
      running_loss = 0.0

print(f'Finished Training, Total time elapsed: {time.perf_counter()-st:.3f} seconds');





