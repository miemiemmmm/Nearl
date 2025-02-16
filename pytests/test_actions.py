import time
import numpy as np 
from nearl import commands


np.random.seed(0)

# For dummp data generation
atom_nr = 100
frame_nr = 30

# For the settings of the grid
dims = np.array([32, 32, 32], dtype=np.int32)
spacing = 0.5
cutoff = 0.5
sigma = 1.0

def test_frame_voxelize():
  """
  Voxelize a set of coordinates and weights
  """
  tmp_coords = np.random.normal(size=(atom_nr, 3), loc=5, scale=1).astype(np.float32)
  tmp_weights = np.full((atom_nr,), 16.0, dtype=np.float32)

  st = time.perf_counter()
  ret = commands.frame_voxelize(tmp_coords, tmp_weights, dims, spacing, cutoff, sigma)
  ret = ret.reshape(dims)
  print(f"Time elapsed: {time.perf_counter() - st:10.6f}")
  assert np.isclose(np.sum(ret), np.sum(tmp_weights))
  # No nan values in the output
  assert not np.isnan(ret).any()


def test_marching_observer():
  """
  Perform marching observers on a trajectory of coordinates and weights
  """
  tmp_traj = np.random.normal(size=(frame_nr, atom_nr, 3), loc=5, scale=2).astype(np.float32)
  tmp_weights = np.full((100*10,), 16.0, dtype=np.float32)

  st = time.perf_counter()
  ret = commands.marching_observer(tmp_traj, tmp_weights, dims, spacing, cutoff, 1, 1)
  print(f"Time elapsed: {time.perf_counter() - st:10.6f}")
  ret = ret.reshape(dims)
  assert not np.isnan(ret).any()


def test_density_flow(): 
  tmp_coords = np.random.normal(size=(frame_nr, atom_nr, 3), loc=5, scale=1).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 16.0, dtype=np.float32)

  st = time.perf_counter()
  ret = commands.density_flow(tmp_coords, tmp_weights, dims, spacing, cutoff, sigma, 1)
  ret = ret.reshape(dims)
  print(f"Time elapsed: {time.perf_counter() - st:10.6f}")
  assert not np.isnan(ret).any()

