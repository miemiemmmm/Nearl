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


def test_benchmark_frame_voxelize(benchmark):
  tmp_coords = np.random.normal(size=(atom_nr, 3), loc=5, scale=1).astype(np.float32)
  tmp_weights = np.full((atom_nr,), 16.0, dtype=np.float32)

  def run_benchmark():
    ret = commands.frame_voxelize(tmp_coords, tmp_weights, dims, spacing, cutoff, sigma)
    return ret.reshape(dims)
  ret = benchmark(run_benchmark)

  assert np.isclose(np.sum(ret), np.sum(tmp_weights))
  assert not np.isnan(ret).any()


def test_benchmark_marching_observer(benchmark):
  tmp_traj = np.random.normal(size=(frame_nr, atom_nr, 3), loc=5, scale=2).astype(np.float32)
  tmp_weights = np.full((100*10,), 16.0, dtype=np.float32)

  def run_benchmark():
    ret = commands.marching_observer(tmp_traj, tmp_weights, dims, spacing, cutoff, 1, 1)
    return ret.reshape(dims)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()


def test_benchmark_density_flow(benchmark):
  tmp_coords = np.random.normal(size=(frame_nr, atom_nr, 3), loc=5, scale=1).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 16.0, dtype=np.float32)

  def run_benchmark():
    ret = commands.density_flow(tmp_coords, tmp_weights, dims, spacing, cutoff, sigma, 1)
    return ret.reshape(dims)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()