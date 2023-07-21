import time

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

from BetaPose import models, data_io, utils, printit

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

##########################################################################################
##########################################################################################


import jax
import optax
import jax.numpy as jnp

from jax import jit, value_and_grad

devices = jax.devices()

print("All devices:", jax.devices())
print("Default device:", jax.default_device())



# Initialize model
model = models.SimpleNetwork_jax(input_dim=36, output_dim=1)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 36)))  # assuming input_dim is 36

# Define loss
def loss_fn(params, inputs, targets):
    preds = model.apply(params, inputs)
    return jnp.mean((preds - targets) ** 2)  # mean squared error

# Create optimizer
optimizer = optax.adam(0.001)
opt_state = optimizer.init(params)

# Data handling
batch_size = 256
rf_training_data = np.asarray(rf_training_data, dtype=np.float32)  # assuming rf_training_data was a numpy array
label_training_data = np.asarray(label_training_data, dtype=np.float32).reshape(-1, 1)  # assuming label_training_data was a numpy array
num_batches = rf_training_data.shape[0] // batch_size

# Training step
@jit
def train_step(params, opt_state, inputs, targets):
    loss, grads = value_and_grad(loss_fn)(params, inputs, targets)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
epochs = 30
st = time.perf_counter()
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n------------------------------------------------")
    running_loss = 0.0
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        inputs = rf_training_data[start:end]
        targets = label_training_data[start:end]
        params, opt_state, batch_loss = train_step(params, opt_state, inputs, targets)
        running_loss += batch_loss
        if ((i+1) % 100) == 0:
            print(f"Epoch{epoch+1}, batch{i+1:5d}: running loss: {running_loss:.3f}")
            running_loss = 0.0
print(f'Finished Training, Total time elapsed: {time.perf_counter()-st:.3f} seconds')


##########################################################################################
##########################################################################################

def evaluate_model(model, params, input_data, labels):
  # Note: no need to switch model to evaluation mode in JAX
  # Convert to JAX array if needed
  if isinstance(input_data, np.ndarray):
    input_data = jnp.array(input_data)

  y_pred = model.apply(params, input_data)
  y_pred = y_pred.flatten()  # flatten the output

  # convert lists to numpy arrays for further processing
  y_true = np.array(labels).ravel()
  y_pred = np.array(y_pred).ravel()

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


print("On Training Set: ")
evaluate_model(model, params, rf_training_data, label_training_data)
print("On Test Set: ")
evaluate_model(model, params, rf_testset, label_testset)




