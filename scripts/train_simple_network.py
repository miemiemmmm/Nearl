import time

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats



from BetaPose import utils, data_io, models, printit

st = time.perf_counter();
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
  with data_io.hdf_operator(input_hdfile, read_only=True) as h5file:
    rf_data.append(h5file.data("rf"))
    label_data.append(h5file.data("label").ravel())
rf_training_data = np.concatenate(rf_data, axis=0)
label_training_data = np.concatenate(label_data, axis=0)
print(f"Training dataset: {rf_training_data.shape} ; Label number: {len(label_training_data)}");

# Load test dataset;
testset_file = "/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_testset_randomforest.h5"
with data_io.hdf_operator(testset_file, read_only=True) as h5file:
  h5file.draw_structure()
  rf_testset = h5file.data("rf")
  label_testset = h5file.data("label").ravel()


printit("Data loaded!!! Good luck!!!");

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ratio_test = 0.5
# X_train, X_test, y_train, y_test = train_test_split(rf_training_data, label_training_data, test_size=ratio_test, random_state=42)

# Create an instance of the network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
net = models.SimpleNetwork()
net.to(device)

criterion = nn.MSELoss()                                  # Mean Squared Error Loss for regression.
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Define the optimizer


# Define the number of epochs
epochs = 40

# Check if there is nan or inf in the training data
print("Has nan: ", np.any(np.isnan(rf_training_data)))
print("Has inf: ", np.any(np.isinf(rf_training_data)))


rf_training_data = torch.from_numpy(rf_training_data).to(dtype=torch.float32)
label_training_data = torch.from_numpy(label_training_data).to(dtype=torch.float32)
tensor_dataset = TensorDataset(rf_training_data, label_training_data)

# Create a DataLoader. You can tweak the batch_size as needed
data_loader = DataLoader(tensor_dataset, batch_size=256, shuffle=True)

# Training loop
for epoch in range(epochs):
  print(f"Epoch {epoch+1}\n------------------------------------------------")
  running_loss = 0.0
  for i, data in enumerate(data_loader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    labels = labels.unsqueeze(-1)
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if ((i+1) % 100) == 0:
      print(f"Epoch{epoch+1}, batch{i+1:5d}: running loss: {running_loss:.3f}");
      running_loss = 0.0

print(f'Finished Training, Total time elapsed: {time.perf_counter()-st:.3f} seconds');


def evaluate_model(model, input_data, labels):
  model.eval()  # switch the model to evaluation mode
  if isinstance(input_data, np.ndarray):
    input_data = torch.from_numpy(input_data).to(dtype=torch.float32).to(device)
  elif isinstance(input_data, torch.Tensor):
    input_data = input_data.to(dtype=torch.float32).to(device)

  with torch.no_grad():  # we don't need gradients for evaluation
    y_pred = model(input_data).cpu().numpy().flatten()

  # convert lists to numpy arrays for further processing
  y_true = np.array(labels)
  y_pred = np.array(y_pred)

  # compute metrics
  mse = mean_squared_error(y_true, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_true, y_pred)
  median_residual = np.median(y_true - y_pred)
  median_abs_residual = np.median(np.abs(y_true - y_pred))

  pearson_corr, _ = stats.pearsonr(y_true, y_pred)
  spearman_corr, _ = stats.spearmanr(y_true, y_pred)
  kendall_tau, _ = stats.kendalltau(y_true, y_pred)

  print(f"{'MSE':10s}: {mse:<8.3f} | {'RMSE':10s}: {rmse:<8.3f} | {'R^2':10s}: {r2:<8.3f}");
  print(f"{'MedRes':10s}: {median_residual:<8.3f} | {'MedAbsRes':10s}: {median_abs_residual:<8.3f} | ");
  print(f"{'Pearson':10s}: {pearson_corr:<8.3f} | {'Spearman':10s}: {spearman_corr:<8.3f} | {'Kendall':10s}: {kendall_tau:<8.3f}");

  return [mse, rmse, r2, median_residual, median_abs_residual, pearson_corr, spearman_corr, kendall_tau]

# calculate metrics for the training set
print("On Training set: ")
result_metrics = evaluate_model(net, rf_training_data, label_training_data)

# calculate metrics for the test set
print("\nOn Test set: ")
result_metrics = evaluate_model( net, rf_testset, label_testset)


# print(f"Mean squared error: {mse_test:.3f}, RMSE: {rmse_test:.3f}, R^2: {r2_test:.3f}")
# print(f"Median of residuals: {median_residual_test:.3f}, Median of absolute residuals: {median_abs_residual_test:.3f}")
# print(f"Pearson {pearson_corr_test:.3f}, Spearman {spearman_corr_test:.3f}, Kendall {kendall_tau_test:.3f}")











