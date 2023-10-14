import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from BetaPose import models, printit, data_io, utils

import dask.array as da


# Define the chunks size according to your memory limit


BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The device is: ", DEVICE)

# rf_training_data = np.random.normal(size=(20000, 3, 32, 32, 32));
st = time.perf_counter()
chunks = (1000, 3, 32, 32, 32)  # Adjust this to a suitable size according to your memory capacity
rf_training_data = da.random.normal(size=(12000, 3, 32, 32, 32), chunks=chunks)
rf_training_data.compute()
rf_training_data = np.asarray(rf_training_data, dtype=np.float32);
print("Input data preared: ", rf_training_data.shape, f"Time elapsed: {time.perf_counter()-st:.3f} s")
# rng = np.random.default_rng()
# rf_training_data = rng.normal(size=(20000, 3, 32, 32, 32), dtype='float32')
label_training_data = np.random.normal(size=(12000, 1));

dataset = TensorDataset(torch.from_numpy(rf_training_data).to(dtype=torch.float32), torch.from_numpy(label_training_data).to(dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = models.CNN3D().to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

st = time.perf_counter();
for epoch in range(EPOCHS):
  print(f"Epoch {epoch+1}\n------------------------------------------------")
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
    if ((i+1) % 100) == 0:
      print(f"Epoch{epoch+1}, batch{i+1:5d}: running loss: {running_loss:.3f}");
      running_loss = 0.0

print("Finished Training, Total time elapsed: ", time.perf_counter()-st, " seconds");










