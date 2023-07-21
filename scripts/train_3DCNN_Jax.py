import time

import numpy as np
import jax
import optax
import jax.numpy as jnp
from jax import jit, value_and_grad
import dask.array as da

from BetaPose import models, printit, data_io, utils


st = time.perf_counter();
chunks = (1000, 3, 32, 32, 32)  # Adjust this to a suitable size according to your memory capacity
rf_training_data = da.random.normal(size=(1000, 3, 32, 32, 32), chunks=chunks)
rf_training_data.compute()
rf_training_data = np.asarray(rf_training_data, dtype=np.float32);
label_training_data = np.random.normal(size=(1000, 1));

print("Input data prepared: ", rf_training_data.shape, f"Time elapsed: {time.perf_counter()-st:.3f} s")

# from jax.config import config
# config.update("jax_platform_name", "cpu")

def move_channel_dim(input_tensor, from_dim=1, to_dim=-1):
  # Generate the permutation order for the axes.
  order = list(range(len(input_tensor.shape)));
  pos = order.index(order[to_dim]);
  order.remove(from_dim);
  order.insert(pos, from_dim);
  output_tensor = input_tensor.transpose(order)
  return output_tensor

training_data = move_channel_dim(rf_training_data);
print(f"Training input data shape: {training_data.shape}")

model = models.CNN3D_JAX()
params = model.init(jax.random.PRNGKey(100), jnp.ones((1, 32, 32, 32, 3)))

def loss_fn(params, inputs, targets):
  preds = model.apply(params, inputs)
  return jnp.mean((preds - targets) ** 2)  # mean squared error
# Create optimizer
optimizer = optax.adam(0.001)
opt_state = optimizer.init(params)

# Data handling
epochs = 15
batch_size = 32
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




