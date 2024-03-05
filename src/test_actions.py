import time

import numpy as np 
import open3d as o3d

import all_actions

def view_ret(ret):
  points = []
  colors = []
  max_val = np.max(ret)
  for i in range(dims[0]):
    for j in range(dims[1]):
      for k in range(dims[2]):
        if ret[i, j, k] > 0.1:
          points.append([i, j, k])
          colors.append([ret[i, j, k] / max_val, 0, 0])
          
  p3d = o3d.geometry.PointCloud()
  p3d.points = o3d.utility.Vector3dVector(np.array(points))
  p3d.colors = o3d.utility.Vector3dVector(np.array(colors))
  o3d.visualization.draw_geometries([p3d])

np.random.seed(0)

coords = np.random.normal(size=(100, 3), loc=5, scale=2).astype(np.float32)
weights = np.full((100,), 16.0, dtype=np.float32)


coords_f = np.array(coords, dtype=np.float32)
weights_f = np.array(weights, dtype=np.float32)

dims = np.array([32, 32, 32], dtype=np.int32)
spacing_list = [0.5, 0.5, 0.5]
spacing = 0.5

# Using double as input
repeat_nr = 50
st = time.perf_counter()
for i in range(repeat_nr): 
  ret = all_actions.do_voxelize(coords, weights, dims, spacing, 1.0, 1.0)
print(f"Took {time.perf_counter() - st:.2f} seconds; Each run takes: {1000 * (time.perf_counter() - st) / repeat_nr:.2f} ms")

ret = ret.reshape(dims)
print(f"Return shape: {ret.shape}; Check sum {np.sum(ret)} -> {np.sum(weights)}")

# view_ret(ret)

############################################################################### 
print("Testing marching observers algorithm")
tmp_traj = np.random.normal(size=(10, 100, 3), loc=5, scale=2).astype(np.float32)
st = time.perf_counter()
for i in range(repeat_nr): 
  ret = all_actions.do_marching(tmp_traj, dims, spacing, 0.5)
print(f"Took {time.perf_counter() - st:.2f} seconds; Each run takes: {1000 * (time.perf_counter() - st) / repeat_nr:.2f} ms")

ret = ret.reshape(dims)
print(f"Return shape: {ret.shape}; Check sum {np.sum(ret)}")
# view_ret(ret)

print("Done!")


st = time.perf_counter()
ret = all_actions.voxelize_traj(coords, weights, dims, spacing, 10, 1.0, 1.0)
print(f"Took {time.perf_counter() - st:.2f} seconds; Each run takes: {1000 * (time.perf_counter() - st) / repeat_nr:.2f} ms")

print(f"Return shape: {ret.shape}; ")




# Normalization of dynamic features
# A problem of overlapping and non-overlapping problems 

# How to setup the cutoff (viewer range)
# Smaller might be too noisy because of the discretization
# Larger might also be noisy because of the overlapping


# Using discrete atom might cause severe problem with very noisy data
# Could use a cached atomic density to obtain the dynamic features instead


