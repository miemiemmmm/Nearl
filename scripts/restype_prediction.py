import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from nearl.static import cos_sim
from nearl.static import cos_sim_g as cos_sim
import time, os
import numpy as np
import matplotlib.pyplot as plt

class PointNet(nn.Module):
  def __init__(self, num_classes=21):  # 21 classes, 0-20
    super(PointNet, self).__init__()

    # Shared MLP layers
    self.conv1 = nn.Conv1d(36, 64, 1)
    self.conv2 = nn.Conv1d(64, 128, 1)
    self.conv3 = nn.Conv1d(128, 256, 1)

    # Fully Connected layers
    self.fc1 = nn.Linear(256, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, num_classes)  # Output classes 0-20

  def forward(self, x):
    # Input shape (batch_size, num_points, 3)
    x = x.permute(0, 2, 1)  # Change to (batch_size, 3, num_points)

    # Shared MLP layers
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))

    # Global Max Pooling
    x = torch.max(x, 2, keepdim=True)[0]
    x = x.view(-1, 256)

    # Fully Connected layers
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return F.log_softmax(x, dim=1)

class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN, self).__init__()

    # Input layer: 36 features
    # Hidden layers: 128, 64 units
    # Output layer: 21 units (for labels 0-20)

    self.fc1 = nn.Linear(36, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x

# Instantiate and test the model
if __name__ == "__main__":
  def standardize(data, means=[], stds=[]):
    return (data - means) / (stds + 1e-9)

  def standardize(arr):
    # Avoid division by zero
    min_val = np.min(arr, axis=0, keepdims=True)
    max_val = np.max(arr, axis=0, keepdims=True)
    epsilon = 1e-7
    return (arr - min_val) / (max_val - min_val + epsilon)


  def load_data(dirname, residue_order):
    total_entry_number = 0
    for res in residue_order:
      datai = np.load(f"{dirname}/{res}_all.npy", allow_pickle=True).astype(float)
      total_entry_number += datai.shape[0]
      if np.isnan(datai).any() or np.isinf(datai).any():
        mask1 = np.isnan(datai).any(axis=1)
        mask2 = np.isinf(datai).any(axis=1)
        mask = mask1 | mask2
        correction = np.count_nonzero(mask)
        print("Nan detected in the data file for residue: ", res, " ; Number of nan: ", correction)
        total_entry_number -= correction

    ret_data = np.zeros((total_entry_number, 216), dtype=np.float64)
    start_row = 0
    results = []
    for res in residue_order:
      datai = np.load(f"{dirname}/{res}_all.npy", allow_pickle=True).astype(float)
      # Remove the nan and inf entries
      if np.isnan(datai).any():
        mask1 = np.isnan(datai).any(axis=1)
        mask2 = np.isinf(datai).any(axis=1)
        mask = mask1 | mask2
        datai = datai[~mask]
      entry_number = datai.shape[0]
      ret_data[start_row:start_row + entry_number, :] = np.asarray(datai, dtype=np.float64)
      results += [res] * entry_number
      start_row += entry_number
    return ret_data, results

  FOCUSED_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                      'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                      'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                      'SER', 'THR', 'TRP', 'TYR', 'VAL']
  thedict = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
             'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18,
             'VAL': 19}
  # FOCUSED_RESIDUES = FOCUSED_RESIDUES[:2]
  outfolder = "/home/yzhang/Documents/tests/tempfolder_mlproj/first5000"
  outfolder = "/home/yzhang/Documents/tests/tempfolder_mlproj/second5000_complex"

  st = time.perf_counter()

  # ref_arr, ref_result = load_data(outfolder, FOCUSED_RESIDUES)
  # result_label = np.array([thedict[res] for res in ref_result])[:, np.newaxis].astype(np.float32)
  # # There is very high lenard jones energy in the data, so we need to remove the outliers
  # mask1 = ref_arr[:, 6] < 500
  # print("The number of outliers: ", np.count_nonzero(~mask1))
  # ref_arr = ref_arr[mask1]
  # result_label = result_label[mask1]
  # print(ref_arr.shape, result_label.shape)
  # np.save(os.path.join(outfolder, "ref_arr.npy"), ref_arr)
  # np.save(os.path.join(outfolder, "result_label.npy"), result_label)

  ref_arr = np.load(os.path.join(outfolder, "ref_arr.npy"), allow_pickle=True)
  result_label = np.load(os.path.join(outfolder, "result_label.npy"), allow_pickle=True)
  print(f"Concatenated the dateset: {time.perf_counter()-st:.2f} Seconds")
  
  MEANS = ref_arr.astype(np.float32).mean(axis=0)
  STDS = ref_arr.astype(np.float32).std(axis=0)
  print("Means: ", MEANS, "Stds: ", STDS.tolist())

  st = time.perf_counter()
  # standardized_dataset = standardize(ref_arr, means=MEANS, stds=STDS)
  standardized_dataset = standardize(ref_arr)
  print(f"Finished the standardization, time: {time.perf_counter() - st:.2f} seconds")

  thewhere = np.where(np.isnan(standardized_dataset))
  # print(thewhere[0])
  # print(thewhere[1].tolist())
  print(result_label.shape, len(thewhere[1])/2)
  # exit(0)


  # Instantiate the model and start training
  BATCH_SIZE = 256
  EPOCHS = 1
  DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("The device is: ", DEVICE)

  model = SimpleNN().to(DEVICE)

  print(standardized_dataset[:, :36].shape, result_label.shape)

  dataset = TensorDataset(torch.from_numpy(standardized_dataset[:, :36]).to(dtype=torch.float32),
                          torch.from_numpy(result_label).to(dtype=torch.float32))
  dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

  loss_fn = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)


  st = time.perf_counter()
  for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}\n------------------------------------------------")
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
      inputs, labels = data
      inputs = inputs.to(DEVICE)
      labels = labels.to(DEVICE)
      optimizer.zero_grad()
      outputs = model(inputs)

      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if ((i + 1) % 100) == 0:
        print(f"Epoch{epoch + 1}, batch{i + 1:5d}: running loss: {running_loss:.3f}")
        running_loss = 0.0
  print("Finished Training, Total time elapsed: ", time.perf_counter() - st, " seconds")

  # Evaluate the model
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for data in dataloader:
      inputs, labels = data
      inputs = inputs.to(DEVICE)
      labels = labels.to(DEVICE)
      outputs = model(inputs)
      outputs = torch.round(outputs)

      # print(outputs.ravel(), outputs.shape)
      total += labels.size(0)
      correct += (outputs == labels).sum().item()
    print("total:", total, "correct: ", correct)
    print(f"Accuracy of the network on the {total} test images: {100 * (correct / total)}%")


  # Create dummy data with shape (batch_size, num_points, 3)
  # dummy_data = torch.rand(10, 1024, 3)
  #
  # # Forward pass
  # output = pointnet(dummy_data)
  # print("Output shape:", output.shape)  # Should be [10, 21]
