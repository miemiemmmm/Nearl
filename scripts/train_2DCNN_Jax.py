import time

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

from BetaPose import models, printit, data_io, utils

import jax
import optax
import jax.numpy as jnp
from jax import jit, value_and_grad

def pytorch_to_jax(data):
  # The original dimensions are (batch_size, channels, height, width)
  # The target dimensions are (batch_size, height, width, channels)
  data = jnp.asarray(data)
  return jnp.transpose(data, (0, 2, 3, 1))




training_data = np.random.normal(size=(50000, 3, 32, 32));
label_training_data = np.random.normal(size=(50000, 1));

training_data = pytorch_to_jax(training_data)
print(training_data.shape)
# exit(0)

model = models.CNN2D_JAX()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))

def loss_fn(params, inputs, targets):
  preds = model.apply(params, inputs)
  return jnp.mean((preds - targets) ** 2)  # mean squared error
# Create optimizer
optimizer = optax.adam(0.001)
opt_state = optimizer.init(params)

# Data handling
epochs = 15
batch_size = 256
training_data = jnp.asarray(training_data, dtype=np.float32)  # assuming rf_training_data was a numpy array
label_training_data = jnp.asarray(label_training_data, dtype=np.float32).reshape(-1, 1)  # assuming label_training_data was a numpy array
num_batches = training_data.shape[0] // batch_size

@jit
def train_step(params, opt_state, inputs, targets):
  loss, grads = value_and_grad(loss_fn)(params, inputs, targets)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss

# Training loop
st = time.perf_counter()
for epoch in range(epochs):
  print(f"Epoch {epoch+1}\n------------------------------------------------")
  running_loss = 0.0
  for i in range(num_batches):
    start = i * batch_size
    end = start + batch_size
    inputs = training_data[start:end]
    targets = label_training_data[start:end]
    params, opt_state, batch_loss = train_step(params, opt_state, inputs, targets)
    running_loss += batch_loss
    if ((i+1) % 100) == 0:
      print(f"Epoch{epoch+1}, batch{i+1:5d}: running loss: {running_loss:.3f}")
      running_loss = 0.0
print(f'Finished Training, Total time elapsed: {time.perf_counter()-st:.3f} seconds')


##########################################################################################
##########################################################################################


def apply_model_in_batches(model, params, data, batch_size):
  # This will hold all the model outputs
  outputs = []
  # Start indices for our batches
  start_indices = range(0, data.shape[0], batch_size)
  for start_index in start_indices:
    # Slice a batch from the data
    batch = data[start_index: start_index + batch_size]
    # Apply the model to the batch
    output = model.apply(params, batch)
    # Store the output
    outputs.append(output)
  # Concatenate all outputs along the batch dimension
  outputs = jnp.concatenate(outputs, axis=0)
  return outputs

y_pred = apply_model_in_batches(model, params, training_data, 1000)
print(y_pred.shape, y_pred)

