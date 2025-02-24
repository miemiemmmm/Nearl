import time
import numpy as np 
from nearl import commands


np.random.seed(0)

# For dummp data generation
atom_nr = 300
frame_nr = 50

# For the settings of the grid
dims = np.array([32, 32, 32], dtype=np.int32)
spacing = 1.0
cutoff = 2.5
sigma = 1.0


def test_benchmark_frame_voxelize(benchmark):
  tmp_coords = np.random.normal(size=(atom_nr, 3), loc=5, scale=1).astype(np.float32)
  tmp_weights = np.full((atom_nr,), 1.5, dtype=np.float32)

  def run_benchmark():
    ret = commands.frame_voxelize(tmp_coords, tmp_weights, dims, spacing, cutoff, sigma)
    return ret.reshape(dims)
  ret = benchmark(run_benchmark)

  assert np.isclose(np.sum(ret), np.sum(tmp_weights), rtol=1e-3)
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


def test_benchmark_voxel_dim16(benchmark):
  dim0 = 16
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((atom_nr,), 1.5, dtype=np.float32)

  def run_benchmark():
    ret = commands.frame_voxelize(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)

  assert np.isclose(np.sum(ret), np.sum(tmp_weights), rtol=1e-3)
  assert not np.isnan(ret).any()


def test_benchmark_voxel_dim24(benchmark):
  dim0 = 24
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((atom_nr,), 1.5, dtype=np.float32)

  def run_benchmark():
    ret = commands.frame_voxelize(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)

  assert np.isclose(np.sum(ret), np.sum(tmp_weights), rtol=1e-3)
  assert not np.isnan(ret).any()

def test_benchmark_voxel_dim32(benchmark):
  dim0 = 32
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((atom_nr,), 1.5, dtype=np.float32)

  def run_benchmark():
    ret = commands.frame_voxelize(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)

  assert np.isclose(np.sum(ret), np.sum(tmp_weights), rtol=1e-3)
  assert not np.isnan(ret).any()

def test_benchmark_voxel_dim48(benchmark):
  dim0 = 48
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((atom_nr,), 1.5, dtype=np.float32)

  def run_benchmark():
    ret = commands.frame_voxelize(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)

  assert np.isclose(np.sum(ret), np.sum(tmp_weights), rtol=1e-3)
  assert not np.isnan(ret).any()

def test_benchmark_voxel_dim64(benchmark):
  dim0 = 64
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((atom_nr,), 1.5, dtype=np.float32)

  def run_benchmark():
    ret = commands.frame_voxelize(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)

  assert np.isclose(np.sum(ret), np.sum(tmp_weights), rtol=1e-3)
  assert not np.isnan(ret).any()

def test_benchmark_voxel_dim96(benchmark):
  dim0 = 96
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((atom_nr,), 1.5, dtype=np.float32)

  def run_benchmark():
    ret = commands.frame_voxelize(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)

  assert np.isclose(np.sum(ret), np.sum(tmp_weights), rtol=1e-3)
  assert not np.isnan(ret).any()

def test_benchmark_voxel_dim128(benchmark):
  dim0 = 128
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((atom_nr,), 1.5, dtype=np.float32)

  def run_benchmark():
    ret = commands.frame_voxelize(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)

  assert np.isclose(np.sum(ret), np.sum(tmp_weights), rtol=1e-3)
  assert not np.isnan(ret).any()


#####################################
################ MO #################
#####################################
def test_benchmark_mo_dim16(benchmark):
  dim0 = 16
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_traj = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.marching_observer(tmp_traj, tmp_weights, dimension, spacing, cutoff, 1, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()

def test_benchmark_mo_dim24(benchmark):
  dim0 = 24
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_traj = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.marching_observer(tmp_traj, tmp_weights, dimension, spacing, cutoff, 1, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()

def test_benchmark_mo_dim32(benchmark):
  dim0 = 32
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_traj = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.marching_observer(tmp_traj, tmp_weights, dimension, spacing, cutoff, 1, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()

def test_benchmark_mo_dim48(benchmark):
  dim0 = 48
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_traj = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.marching_observer(tmp_traj, tmp_weights, dimension, spacing, cutoff, 1, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()

def test_benchmark_mo_dim64(benchmark):
  dim0 = 64
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_traj = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.marching_observer(tmp_traj, tmp_weights, dimension, spacing, cutoff, 1, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()

def test_benchmark_mo_dim96(benchmark):
  dim0 = 96
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_traj = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.marching_observer(tmp_traj, tmp_weights, dimension, spacing, cutoff, 1, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()

def test_benchmark_mo_dim128(benchmark):
  dim0 = 128
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_traj = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.marching_observer(tmp_traj, tmp_weights, dimension, spacing, cutoff, 1, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()


#####################################
################ PDF ################
#####################################
def test_benchmark_pdf_dim16(benchmark): 
  dim0 = 16
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.density_flow(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()


def test_benchmark_pdf_dim24(benchmark):
  dim0 = 24
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.density_flow(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()

def test_benchmark_pdf_dim32(benchmark):
  dim0 = 32
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.density_flow(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()

def test_benchmark_pdf_dim48(benchmark):
  dim0 = 48 
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.density_flow(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()

def test_benchmark_pdf_dim64(benchmark):
  dim0 = 64
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.density_flow(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()


def test_benchmark_pdf_dim96(benchmark):
  dim0 = 96
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.density_flow(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()

def test_benchmark_pdf_dim128(benchmark):
  dim0 = 128
  dimension = np.array([dim0, dim0, dim0], dtype=np.int32)
  tmp_coords = np.random.randint(cutoff, dim0-cutoff, size=(frame_nr, atom_nr, 3)).astype(np.float32)
  tmp_weights = np.full((frame_nr*atom_nr,), 1.25, dtype=np.float32)

  def run_benchmark():
    ret = commands.density_flow(tmp_coords, tmp_weights, dimension, spacing, cutoff, sigma, 1)
    return ret.reshape(dimension)
  ret = benchmark(run_benchmark)
  assert not np.isnan(ret).any()