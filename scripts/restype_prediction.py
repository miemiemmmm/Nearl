import tempfile, subprocess

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
    self.fc1 = nn.Linear(36, 128)
    self.fc2 = nn.Linear(128, 256)
    self.fc3 = nn.Linear(256, 20)
    self.fc4 = nn.Linear(20, 20)
    # count the number of parameters in this model
    self.num_parameters()

  def num_parameters(self):
    num_params = 0
    for param in self.parameters():
      num_params += param.numel()
    print(f"Number of parameters in the network: {num_params}")


  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
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
  outfolder = "/home/yzhang/Documents/tests/tempfolder_mlproj/second5000_complex"
  reffolder = "/home/yzhang/Documents/tests/tempfolder_mlproj/first5000"

  st = time.perf_counter()

  if not os.path.isfile(os.path.join(outfolder, "ref_arr.npy")):
    ref_arr, ref_result = load_data(outfolder, FOCUSED_RESIDUES)
    result_label = np.array([thedict[res] for res in ref_result])[:, np.newaxis].astype(np.float32)
    # There is very high lenard jones energy in the data, so we need to remove the outliers
    mask1 = ref_arr[:, 6] < 500
    ref_arr = ref_arr[mask1]
    result_label = result_label[mask1]
    np.save(os.path.join(outfolder, "ref_arr.npy"), ref_arr)
    np.save(os.path.join(outfolder, "result_label.npy"), result_label)
    print(f"The number of outliers: {np.count_nonzero(~mask1)}; Final result shape: {ref_arr.shape}, {result_label.shape}")


  ref_arr = np.load(os.path.join(outfolder, "ref_arr.npy"), allow_pickle=True)
  result_label = np.load(os.path.join(outfolder, "result_label.npy"), allow_pickle=True)

  testset_arr = np.load(os.path.join(reffolder, "ref_arr.npy"), allow_pickle=True)
  testset_label = np.load(os.path.join(reffolder, "result_label.npy"), allow_pickle=True)

  ref_arr = np.concatenate((ref_arr, testset_arr), axis=0)
  result_label = np.concatenate((result_label, testset_label), axis=0)

  
  # MEANS = ref_arr.astype(np.float32).mean(axis=0)
  # STDS = ref_arr.astype(np.float32).std(axis=0)
  # print("Means: ", MEANS, "Stds: ", STDS.tolist())

  st = time.perf_counter()
  # standardized_dataset = standardize(ref_arr, means=MEANS, stds=STDS)    # Z-score standardization
  standardized_dataset = standardize(ref_arr)   # Min-Max standardization
  print(f"Finished the standardization, time: {time.perf_counter() - st:.2f} seconds")
  print("Standardized data", standardized_dataset[:, :36].shape, result_label.shape)

  # Instantiate the model and start training
  BATCH_SIZE = 500
  EPOCHS = 5
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("The device is: ", DEVICE)

  model = SimpleNN().to(DEVICE)

  mask_train = np.zeros(len(standardized_dataset), dtype=bool)
  train_indices = np.random.choice(len(standardized_dataset), 1500000, replace=False)
  mask_train[train_indices] = True
  training_data = standardized_dataset[train_indices, :36]
  training_label = result_label[train_indices]
  dataset = TensorDataset(torch.from_numpy(training_data).to(DEVICE, dtype=torch.float32),
                          torch.from_numpy(training_label).to(DEVICE, dtype=torch.float32))
  dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

  # loss_fn = nn.MSELoss()
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  st = time.perf_counter()
  for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}\n------------------------------------------------")
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
      inputs, labels = data
      inputs.to(DEVICE)

      labels = labels.squeeze(1)
      labels.to(DEVICE)
      optimizer.zero_grad()
      outputs = model(inputs)
      outputs.to(DEVICE)


      # outputs = torch.max(outputs, dim=1).unsqueeze(-1)
      print(labels, labels.shape)
      print(outputs, outputs.shape)

      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if ((i + 1) % 100) == 0:
        print(f"Epoch{epoch + 1}, batch{i + 1:5d}: running loss: {running_loss:.3f}")
        running_loss = 0.0
  print("Finished Training, Total time elapsed: ", time.perf_counter() - st, " seconds")



  # Evaluate the model
  print("===>>> Evaluating the model on the training dataset")
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
    print(f"Accuracy of the network on dataset ({correct}/{total}); Percentage rate: {100 * (correct / total)}%")

  print("===>>> Evaluating the model on the test dataset")
  st = time.perf_counter()
  test_data = standardized_dataset[~mask_train, :36]
  test_label = result_label[~mask_train]

  _dataset = TensorDataset(
    torch.from_numpy(test_data).to(dtype=torch.float32),
    torch.from_numpy(test_label).to(dtype=torch.float32)
  )
  _dataloader = DataLoader(_dataset, batch_size=BATCH_SIZE, shuffle=True)

  model.eval()
  confusion_matrix = np.zeros((20, 20), dtype=np.int32)
  with torch.no_grad():
    correct = 0
    total = 0
    for data in _dataloader:
      inputs, labels = data
      inputs = inputs.to(DEVICE)
      labels = labels.to(DEVICE)
      outputs = model(inputs)
      outputs = torch.round(outputs)

      total += labels.size(0)
      correct += (outputs == labels).sum().item()
      coords = torch.stack((labels, outputs), dim=1).cpu().numpy().reshape((-1, 2)).astype(np.int32)
      for coord in coords:
        x = max(coord[0], 0)
        x = min(x, 19)
        y = max(coord[1], 0)
        y = min(y, 19)
        confusion_matrix[x, y] += 1

    print(confusion_matrix)
    print(f"Accuracy of the network on the ({correct}/{total}); Percentage rate: {100 * (correct / total)}%")


  def add_text(matrix, threshold=0.01):
    for i in range(matrix.shape[0]):
      for j in range(matrix.shape[1]):
        if matrix[i, j] > threshold:
          plt.text(j, i, f"{matrix[i, j] * 100:.0f}",
                   ha="center",
                   va="center",
                   fontdict={'family': 'serif', 'color': '#A8E4B1', 'weight': 'bold', 'style': 'italic', 'size': 10, }
                   )

  print(np.sum(confusion_matrix, axis=1, keepdims=True))

  print("Drawing the confusion matrix")
  confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True)
  confusion_matrix = np.array(confusion_matrix, dtype=float)

  import seaborn as sns
  # import matplotlib.pyplot as plt

  with tempfile.NamedTemporaryFile(suffix=".npy") as file1:
    np.save(file1.name, confusion_matrix)
    subprocess.run(["python", "testplt.py", file1.name])
    print("Finished drawing the confusion matrix")



  # _matrix = np.zeros((20, 20))
  # sns.heatmap(_matrix)
  # plt.show()
  # sns.savefig("confusion_matrix.png", dpi=300)

  # print(confusion_matrix)
  # plt.figure(figsize=(8, 8))
  # plt.imshow(confusion_matrix, vmax=0.8, vmin=0, cmap="inferno")
  # plt.xticks(np.arange(len(FOCUSED_RESIDUES)), FOCUSED_RESIDUES, rotation=-45)
  # plt.yticks(np.arange(len(FOCUSED_RESIDUES)), FOCUSED_RESIDUES)
  # print("checkpoint 2")
  # plt.xlabel("Overall fingerprint")
  #
  # # plt.show()
  # print("checkpoint 2.5")
  # plt.savefig("confusion_matrix.png", dpi=300)
  # print("checkpoint 3")
