import time

import jax, optax
import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad

from nearl.models import models_jax


# Detect hardware configuration
num_gpus = jax.device_count()
devices = jax.devices()
print(f"Number of GPUs available: {num_gpus}")
print("Devices:", devices)


SAMPLE_NUM = 10000
EPOCHS = 5
BATCH_SIZE = 100

def simple_network():
  # Define loss
  def loss_fn(params, inputs, targets):
    preds = model.apply(params, inputs)
    return jnp.mean((preds - targets) ** 2)  # mean squared error

  @jit
  def train_step(params, opt_state, inputs, targets):
    loss, grads = value_and_grad(loss_fn)(params, inputs, targets)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  # Create the test data
  training_data = np.random.normal(size=(SAMPLE_NUM, 36))
  label_training_data = np.random.normal(size=(SAMPLE_NUM, 1))
  # training_data = models_jax.bchw_to_bhwc(training_data)
  num_batches = SAMPLE_NUM // BATCH_SIZE

  training_data = jnp.asarray(training_data, dtype=np.float32)
  label_training_data = jnp.asarray(label_training_data, dtype=np.float32).reshape(-1, 1)

  # Initialize model
  model = models_jax.SimpleNetwork_JAX(input_dim=36, output_dim=1)
  params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 36)))
  optimizer = optax.adam(0.001)
  opt_state = optimizer.init(params)

  st = time.perf_counter()
  for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}\n------------------------------------------------")
    running_loss = 0.0
    for i in range(num_batches):
      start = i * BATCH_SIZE
      end = start + BATCH_SIZE
      inputs = training_data[start:end]
      targets = label_training_data[start:end]
      params, opt_state, batch_loss = train_step(params, opt_state, inputs, targets)
      running_loss += batch_loss
      if ((i + 1) % 100) == 0:
        print(f"Epoch{epoch + 1}, batch{i + 1:5d}: running loss: {running_loss:.3f}")
        running_loss = 0.0
  print(f'Finished JAX training test')
  print(f'Total time elapsed: {time.perf_counter() - st:.3f} seconds')
  print(f"There are {sum(p.size for p in jax.tree_flatten(params)[0])} parameters in the model.")


def cnn_2d():
  # mean squared error loss
  def loss_fn(params, inputs, targets):
    preds = model.apply(params, inputs)
    return jnp.mean((preds - targets) ** 2)

  @jit
  def train_step(params, opt_state, inputs, targets):
    loss, grads = value_and_grad(loss_fn)(params, inputs, targets)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  # Create the test data
  training_data = np.random.normal(size=(SAMPLE_NUM, 3, 32, 32))
  label_training_data = np.random.normal(size=(SAMPLE_NUM, 1))
  training_data = models_jax.bchw_to_bhwc(training_data)
  num_batches = SAMPLE_NUM // BATCH_SIZE

  training_data = jnp.asarray(training_data, dtype=np.float32)
  label_training_data = jnp.asarray(label_training_data, dtype=np.float32).reshape(-1, 1)

  # Create the model, parameter set and optimizer
  model = models_jax.CNN2D_JAX()
  params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))
  optimizer = optax.adam(0.001)
  opt_state = optimizer.init(params)

  # Training loop
  st = time.perf_counter()
  for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}\n------------------------------------------------")
    running_loss = 0.0
    for i in range(num_batches):
      start = i * BATCH_SIZE
      end = start + BATCH_SIZE
      inputs = training_data[start:end]
      targets = label_training_data[start:end]
      params, opt_state, batch_loss = train_step(params, opt_state, inputs, targets)
      running_loss += batch_loss
      if ((i + 1) % 100) == 0:
        print(f"Epoch{epoch + 1}, batch{i + 1:5d}: running loss: {running_loss:.3f}")
        running_loss = 0.0
  print(f'Finished JAX training test')
  print(f'Total time elapsed: {time.perf_counter() - st:.3f} seconds')
  print(f"There are {sum(p.size for p in jax.tree_flatten(params)[0])} parameters in the model.")

def cnn_3d():
  # mean squared error loss
  def loss_fn(params, inputs, targets):
    preds = model.apply(params, inputs)
    return jnp.mean((preds - targets) ** 2)

  @jit
  def train_step(params, opt_state, inputs, targets):
    loss, grads = value_and_grad(loss_fn)(params, inputs, targets)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  # Create the test data
  training_data = np.random.normal(size=(SAMPLE_NUM, 3, 32, 32, 32))
  label_training_data = np.random.normal(size=(SAMPLE_NUM, 1))
  training_data = models_jax.bchw_to_bhwc(training_data)
  num_batches = SAMPLE_NUM // BATCH_SIZE

  training_data = jnp.asarray(training_data, dtype=np.float32)
  label_training_data = jnp.asarray(label_training_data, dtype=np.float32).reshape(-1, 1)

  # Create the model, parameter set and optimizer
  model = models_jax.CNN3D_JAX()
  params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 32, 3)))
  optimizer = optax.adam(0.001)
  opt_state = optimizer.init(params)

  # Training loop
  st = time.perf_counter()
  for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}\n------------------------------------------------")
    running_loss = 0.0
    for i in range(num_batches):
      start = i * BATCH_SIZE
      end = start + BATCH_SIZE
      inputs = training_data[start:end]
      targets = label_training_data[start:end]
      params, opt_state, batch_loss = train_step(params, opt_state, inputs, targets)
      running_loss += batch_loss
      if ((i + 1) % 100) == 0:
        print(f"Epoch{epoch + 1}, batch{i + 1:5d}: running loss: {running_loss:.3f}")
        running_loss = 0.0
  print(f'Finished JAX training test')
  print(f'Total time elapsed: {time.perf_counter() - st:.3f} seconds')
  print(f"There are {sum(p.size for p in jax.tree_flatten(params)[0])} parameters in the model.")

