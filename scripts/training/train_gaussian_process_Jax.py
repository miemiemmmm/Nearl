import time

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

from BetaPose import models, data_io, utils, printit

st = time.perf_counter();
printit("Loading data...")

input_files = [
  # "/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_trainset_randomforest.h5",
  # "/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_randomforest.h5",
  "/media/yzhang/MieT5/BetaPose/data/trainingdata/pdbbindrefined_v2016_randomforest.h5",
  # "/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_randomforest_step10.h5",
  # "/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_testset_randomforest.h5",
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

##########################################################################################
##########################################################################################
from jax import random


# Assume we have some data
rng = random.PRNGKey(3)       # Replace this with your random seed


# Instantiate the model
length_scale = 10.0
variance = 1.0
noise = 0.15
model = models.GaussianProcessRegressor(length_scale=length_scale, variance=variance, noise=noise)

# Train the model and get predictions for the test set
posterior_mean, posterior_cov, posterior_sample = model(rng, rf_training_data, label_training_data, rf_testset)

# The predictions are in the posterior mean
predictions = posterior_mean

# You might want to calculate some error metric here, for example mean squared error
mse = np.mean((predictions - label_testset)**2)  # you need to replace test_targets with your actual test targets

print(f"Mean squared error on test set: {mse}")

# You can also sample from the posterior for Bayesian predictions
print(f"A sample from the posterior: {posterior_sample}")


