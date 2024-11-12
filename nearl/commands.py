import os

import numpy as np 

from . import all_actions, printit, utils

__all__ = [
  "voxelize_trajectory",
  "voxelize_coords",
  "marching_observers"
]


def voxelize_coords(coords, weights, grid_dims, spacing, cutoff, sigma):
  """
  Voxelize a set of coordinates and weights

  Parameters
  ----------
  coords : np.ndarray
    The coordinates of the points to be voxellized
  weights : np.ndarray
    The weights of the points to be voxellized
  grid_dims : tuple
    The dimensions of the grid
  spacing : float
    The spacing of the grid
  cutoff : float
    The cutoff distance
  sigma : float
    The sigma value for the Gaussian kernel

  Returns
  -------
  np.ndarray
    The voxellized grid sized grid_dims

  Examples
  --------
  >>> import numpy as np
  >>> from nearl import commands
  >>> coords = np.random.normal(size=(100, 3), loc=5, scale=2)
  >>> weights = np.full(100, 1)
  >>> grid_dims = np.array([32, 32, 32])
  >>> commands.voxelize_coords(coords, weights, grid_dims, 0.5, 5, 2)
  """
  if coords.dtype != np.float32:
    coords = coords.astype(np.float32)
  if weights.dtype != np.float32:
    weights = weights.astype(np.float32)
  grid_dims = np.array(grid_dims, dtype=int)
  spacing = float(spacing)
  cutoff = float(cutoff)
  sigma = float(sigma)
  # NOTE: no auto translation in the C++ part 
  ret_arr = all_actions.do_voxelize(coords, weights, grid_dims, spacing, cutoff, sigma, 0)
  return ret_arr.reshape(grid_dims)


def marching_observers(coords, weights, grid_dims, spacing, cutoff, type_obs, type_agg):
  """
  Marching observers algorithm to create a mesh from a slice of frames. The number of atoms in each frame should be the same. 

  Parameters
  ----------
  coords : np.ndarray
    The coordinates of the points to be voxellized
  grid_dims : tuple
    The dimensions of the grid
  spacing : float
    The spacing of the grid
  cutoff : float
    The cutoff distance
  type_obs : int
    The type of observer
  type_agg : int
    The type of aggregation function

  Returns
  -------
  retgrid : np.ndarray
    The voxellized grid sized grid_dims
  
  """
  if coords.dtype != np.float32:
    coords = coords.astype(np.float32)
  if weights.dtype != np.float32:
    weights = weights.astype(np.float32)
  grid_dims = np.asarray(grid_dims, dtype=int)
  ret_arr = all_actions.do_marching(coords, weights, grid_dims, spacing, cutoff, type_obs, type_agg)
  return ret_arr.reshape(grid_dims)


def voxelize_trajectory(traj, weights, grid_dims, spacing, cutoff, sigma, type_agg):
  """
  Voxelize a trajectory (multiple frames version of commands.voxelize_coords). 

  Parameters
  ----------
  traj : np.ndarray
    The trajectory to be voxellized
  weights : np.ndarray
    The weights of the trajectory
  grid_dims : tuple
    The dimensions of the grid
  spacing : float
    The spacing of the grid
  cutoff : float
    The cutoff distance
  sigma : float
    The sigma value for the Gaussian kernel
  type_agg : int
    The type of aggregation function

  Returns
  -------
  retgrid : np.ndarray
    The voxellized grid sized grid_dims

  Examples
  --------
  >>> import numpy as np
  >>> from nearl import commands
  """
  # Check the data type of the inputs; All arrays should be of type np.float32
  if traj.dtype != np.float32:
    traj = traj.astype(np.float32)
  if weights.dtype != np.float32:
    weights = weights.astype(np.float32)
  grid_dims = np.array(grid_dims, dtype=int)
  spacing = float(spacing)
  cutoff = float(cutoff)
  sigma = float(sigma)
  type_agg = int(type_agg)

  ret_arr = all_actions.voxelize_traj(traj, weights, grid_dims, spacing, cutoff, sigma, type_agg)
  if np.isnan(ret_arr).any():
    printit(f"Warning: Found nan in the return: {np.count_nonzero(np.isnan(ret_arr))}")
  return ret_arr.reshape(grid_dims)


def viewpoint_histogram_xyzr(xyzr_arr, viewpoint, bin_nr, write_ply=False, return_mesh=False): 
  """
  Generate the viewpoint histogram from a set of coordinates and radii (XYZR). 
  
  Wiewpoint is the position of the observer.
  """
  import siesta 
  import open3d as o3d 
  thearray = np.asarray(xyzr_arr, dtype=np.float32)
  vertices, faces = siesta.xyzr_to_surf(thearray, grid_size=0.2) 
  c_vertices = np.mean(vertices, axis=0)
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(vertices)
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.compute_vertex_normals()
  mesh.compute_triangle_normals()

  if write_ply: 
    filename = os.path.join("/tmp/", f"segment_{utils.get_timestamp()}.ply")
    printit(f"Writing the surface to {filename}")
    o3d.io.write_triangle_mesh(filename, mesh, write_ascii=True)

  v_view = viewpoint - c_vertices
  v_view = v_view / np.linalg.norm(v_view)
  # Get the normal of each vertex
  normals = np.array(mesh.vertex_normals)
  # Get the cosine angle and split to bins 
  cos_angle = np.dot(normals, v_view)
  bins = np.linspace(-1, 1, bin_nr+1)
  hist, _ = np.histogram(cos_angle, bins, density=True)
  if return_mesh:
    return hist / np.sum(hist), mesh
  else:
    return hist / np.sum(hist)

def discretize_coord(coords, weights, grid_dims, spacing): 
  if coords.dtype != np.float32:
    coords = coords.astype(np.float32)
  if weights.dtype != np.float32:
    weights = weights.astype(np.float32)
  grid_orig = np.zeros(grid_dims, dtype=np.float32)
  # mid = np.array(grid_dims) / 2
  # coords -= mid   # align the center of coord to the center of the grid 
  coord_transform = np.floor(coords / spacing).astype(np.int32)
  for i in range(len(coords)): 
    if np.any(coord_transform[i] < 0) or np.any(coord_transform[i] >= grid_dims):
      continue
    else: 
      grid_orig[coord_transform[i][0], coord_transform[i][1], coord_transform[i][2]] += weights[i]
  return grid_orig
