import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from nearl.models.model_pafnucy import SBNetwork

from torch.utils.tensorboard import SummaryWriter

# Generate some dummy data

dataset = TensorDataset(torch.randn(1000, 19, 40, 40, 40), torch.randn(1000, 1))
train_loader = DataLoader(dataset, batch_size=96, shuffle=True)
val_loader = DataLoader(dataset, batch_size=96, shuffle=False)
num_epochs = 10

tensorboard = SummaryWriter("/MieT5/BetaPose/logs/tensorboard")


# Assuming SBNetwork is your model class


def obtain_network(model, isize, in_chnls, osize, conv_patch, pool_patch, conv_channels, dense_sizes, keep_prob):
  if model == "pafnucy":
    model = SBNetwork(isize, in_chnls, osize, conv_patch, pool_patch, conv_channels, dense_sizes, keep_prob)
    # Adam optimizer with L2 regularization (lambda = 0.001)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.001)
    loss_function = nn.MSELoss()

    return model, loss_function, optimizer
  elif model == "kdeep":
    pass
  elif model == "deeprank":
    pass
  elif model == "voxnet": 
    pass
  else:
    raise ValueError("Model not found")


model, criterion, optimizer = obtain_network("pafnucy",  
  isize=40, 
  in_chnls=19, 
  osize=1, 
  conv_patch=5, 
  pool_patch=2,
  conv_channels=[64, 128, 256, 512], 
  dense_sizes=[1000, 500, 200], 
  keep_prob=0.5
)



# to cuda
if torch.cuda.is_available():
  model = model.cuda()

# Loss function (MSE)
# criterion = nn.MSELoss()

# lmbda = 0.001  # Coefficient for L2 penalty
# optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=lmbda)

# Example training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:  # Assuming train_loader is your DataLoader instance
        # to cuda
        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()  # Clear gradients for this training step
        
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute the loss
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Apply gradients
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader)

        
    # Validation step (optional)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        validation_loss = 0
        for inputs, targets in val_loader:  # Assuming val_loader is your DataLoader instance for validation
            # to cuda
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_loss += loss.item()
            val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}, Validation Loss: {validation_loss / len(val_loader)}")
    tensorboard.add_scalar("Loss/train", train_loss, epoch)
    tensorboard.add_scalar("Loss/val", val_loss, epoch)

tensorboard.close()