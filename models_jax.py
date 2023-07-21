import jax
import jax.numpy as jnp
from flax import linen
import optax

from jax import random
from jax.scipy.linalg import cho_solve, cho_factor

class SimpleNetwork_JAX(linen.Module):
  input_dim: int = 36
  output_dim: int = 1

  def setup(self):
    self.layer1 = linen.Dense(128)
    self.layer2 = linen.Dense(256)
    self.layer3 = linen.Dense(64)
    self.layer4 = linen.Dense(32)
    self.layer5 = linen.Dense(16)
    self.output_layer = linen.Dense(self.output_dim)

  def __call__(self, x):
    x = linen.relu(self.layer1(x))
    x = linen.relu(self.layer2(x))
    x = linen.relu(self.layer3(x))
    x = linen.relu(self.layer4(x))
    x = linen.relu(self.layer5(x))
    x = self.output_layer(x)
    return x


class CNN2D_JAX(linen.Module):
  def setup(self):
    self.conv1 = linen.Conv(features=32, kernel_size=(5, 5))
    self.conv2 = linen.Conv(features=64, kernel_size=(5, 5))
    self.fc1 = linen.Dense(features=1024)
    self.fc2 = linen.Dense(features=1)  # for regression

  def __call__(self, x):
    x = linen.relu(self.conv1(x))
    x = linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = linen.relu(self.conv2(x))
    x = linen.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = linen.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class CNN3D_JAX(linen.Module):
  def setup(self):
    self.conv1 = linen.Conv(features=32, kernel_size=(3, 3, 3))
    self.conv2 = linen.Conv(features=64, kernel_size=(3, 3, 3))
    self.dense = linen.Dense(features=1)

  def __call__(self, x):
    x = self.conv1(x)
    x = linen.relu(x)
    x = linen.max_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
    x = self.conv2(x)
    x = linen.relu(x)
    x = linen.max_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten the tensor
    x = self.dense(x)
    return x


def rbf_kernel(x1, x2, length_scale, variance):
  return variance * jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2) / length_scale ** 2)

# Vectorize the RBF kernel
vmap_rbf = jax.vmap(jax.vmap(rbf_kernel, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))


class GaussianProcessRegressor(linen.Module):
  length_scale: float
  variance: float
  noise: float

  def setup(self):
    pass

  def __call__(self, rng, train_inputs, train_targets, test_inputs):
    # Calculate covariance matrices
    K = vmap_rbf(train_inputs, train_inputs, self.length_scale, self.variance)
    K_s = vmap_rbf(train_inputs, test_inputs, self.length_scale, self.variance)
    K_ss = vmap_rbf(test_inputs, test_inputs, self.length_scale, self.variance)

    print(f"shape of the train_inputs: {train_inputs.shape}")
    print(f"shape of the train_targets: {train_targets.shape}")
    print(f"shape of the test_inputs: {test_inputs.shape}")
    print("Shape of K, K_s, K_ss:", K.shape, K_s.shape, K_ss.shape)

    print(train_inputs.shape)
    # Add noise to the training covariance matrix
    K += self.noise * jnp.eye(K.shape[0])

    # Cholesky decomposition
    L = cho_factor(K)

    # Calculate the posterior mean and covariance
    alpha = cho_solve(L, train_targets)
    v = cho_solve(L, K_s)
    posterior_mean = jnp.dot(K_s.T, alpha)
    posterior_cov = K_ss - jnp.dot(K_s.T, v)

    # Sample from the posterior
    posterior_sample = random.multivariate_normal(rng, mean=posterior_mean, cov=posterior_cov)

    # TODO: Very high variance for the posterior sample, Shall I normalize the data?
    return posterior_mean, posterior_cov, posterior_sample

