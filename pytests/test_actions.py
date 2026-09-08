import time

import numpy as np

from nearl import all_actions, commands

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


def test_device_buffer_reuse_and_growth():
    """Repeated sizes reuse device memory; larger inputs grow it."""
    commands.finalize_context()
    commands.init_context()
    dims = np.array([16, 16, 16], dtype=np.int32)
    small_coords = np.zeros((100, 3), dtype=np.float32)
    small_weights = np.ones(100, dtype=np.float32)
    large_coords = np.zeros((500, 3), dtype=np.float32)
    large_weights = np.ones(500, dtype=np.float32)

    commands.frame_voxelize(small_coords, small_weights, dims, spacing, cutoff, sigma)
    small_capacity = all_actions._buffer_capacity("coords")
    commands.frame_voxelize(small_coords, small_weights, dims, spacing, cutoff, sigma)
    repeated_capacity = all_actions._buffer_capacity("coords")
    commands.frame_voxelize(large_coords, large_weights, dims, spacing, cutoff, sigma)
    large_capacity = all_actions._buffer_capacity("coords")

    assert repeated_capacity == small_capacity
    assert large_capacity > small_capacity


def test_pinned_buffer_reuse_and_growth():
    """Repeated sizes reuse pinned staging memory; larger inputs grow it."""
    commands.finalize_context()
    commands.init_context()
    dims = np.array([16, 16, 16], dtype=np.int32)
    small_coords = np.zeros((100, 3), dtype=np.float32)
    small_weights = np.ones(100, dtype=np.float32)
    large_coords = np.zeros((500, 3), dtype=np.float32)
    large_weights = np.ones(500, dtype=np.float32)

    commands.frame_voxelize(small_coords, small_weights, dims, spacing, cutoff, sigma)
    small_capacity = all_actions._host_buffer_capacity("coords")
    commands.frame_voxelize(small_coords, small_weights, dims, spacing, cutoff, sigma)
    repeated_capacity = all_actions._host_buffer_capacity("coords")
    commands.frame_voxelize(large_coords, large_weights, dims, spacing, cutoff, sigma)
    large_capacity = all_actions._host_buffer_capacity("coords")

    assert repeated_capacity == small_capacity
    assert large_capacity > small_capacity


def test_marching_observer():
    """
    Perform marching observers on a trajectory of coordinates and weights
    """
    tmp_traj = np.random.normal(size=(frame_nr, atom_nr, 3), loc=5, scale=2).astype(
        np.float32
    )
    tmp_weights = np.full((frame_nr * atom_nr,), 1.0, dtype=np.float32)

    st = time.perf_counter()
    ret = commands.marching_observer(tmp_traj, tmp_weights, dims, spacing, cutoff, 1, 1)
    print(f"Time elapsed: {time.perf_counter() - st:10.6f}")
    ret = ret.reshape(dims)
    assert not np.isnan(ret).any()


def test_density_flow():
    tmp_coords = np.random.normal(size=(frame_nr, atom_nr, 3), loc=5, scale=1).astype(
        np.float32
    )
    tmp_weights = np.full((frame_nr * atom_nr,), 16.0, dtype=np.float32)

    st = time.perf_counter()
    ret = commands.density_flow(
        tmp_coords, tmp_weights, dims, spacing, cutoff, sigma, 1
    )
    ret = ret.reshape(dims)
    print(f"Time elapsed: {time.perf_counter() - st:10.6f}")
    assert not np.isnan(ret).any()
