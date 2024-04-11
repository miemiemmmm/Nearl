import sys, os
import h5py as h5py
import numpy as np 
import matplotlib.pyplot as plt

# Read data from an HDF5 file
if len(sys.argv) < 2:
  print("Usage: python plot_performance.py performance.h5")
  sys.exit(1)
elif not os.path.exists(sys.argv[1]):
  print(f"File {sys.argv[1]} does not exist")
  sys.exit(1)

perf_file = sys.argv[1]
outputimage = "/tmp/performance.png"


with h5py.File(perf_file, "r") as f:
  print(f.keys())
  teloss = f["test_loss"][:]
  trloss = f["train_loss"][:]

  termse = f["test_rmse"][:]
  trrmse = f["train_rmse"][:]

  tepear = f["test_pearson"][:]
  trpear = f["train_pearson"][:]
  
  tespea = f["test_spearman"][:]
  trspea = f["train_spearman"][:]

ax, fig = plt.subplots(2, 2, figsize=(10, 8))  
fig[0, 0].plot(teloss, label="Test Loss")
fig[0, 0].plot(trloss, label="Train Loss")
fig[0, 0].set_title("Loss")

fig[0, 0].set_ylim(0, 5)
fig[0, 0].legend()

fig[0, 1].plot(termse, label="Test RMSE")
fig[0, 1].plot(trrmse, label="Train RMSE")
fig[0, 1].set_title("RMSE")
fig[0, 1].set_ylim(0.75, 2.4)
fig[0, 1].legend()


fig[1, 0].plot(tepear, label="Test Pearson")
fig[1, 0].plot(trpear, label="Train Pearson")
fig[1, 0].set_title("Pearson")
fig[1, 0].set_ylim(0, 1)
fig[1, 0].legend()

fig[1, 1].plot(tespea, label="Test Spearman")
fig[1, 1].plot(trspea, label="Train Spearman")
fig[1, 1].set_title("Spearman")
fig[1, 1].set_ylim(0, 1)
fig[1, 1].legend()

#
# plt.tight_layout()
print(f"Saving the image to {outputimage}")
plt.savefig(outputimage)





