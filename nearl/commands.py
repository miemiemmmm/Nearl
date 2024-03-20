from . import all_actions, printit
import numpy as np 

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


def marching_observers(coords, grid_dims, spacing, cutoff, type_obs, type_agg):
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
  grid_dims = np.array(grid_dims, dtype=int)
  ret_arr = all_actions.do_marching(coords, grid_dims, spacing, cutoff, type_obs, type_agg)
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



#   m.def("do_voxelize", &do_voxelize, 
#     py::arg("coords"),
#     py::arg("weights"),
#     py::arg("grid_dims"),
#     py::arg("spacing"),
#     py::arg("cutoff"),
#     py::arg("sigma"),
#     "Voxelize a set of coordinates and weights"
#   );

#   m.def("do_marching", &do_marching_observers, 
#     py::arg("coords"),
#     py::arg("dims"),
#     py::arg("spacing"),
#     py::arg("cutoff"),
#     "Marching cubes algorithm to create a mesh from a 3D grid"
#   );


  # m.def("voxelize_traj", &do_traj_voxelize, 
  #   py::arg("traj"),
  #   py::arg("weights"),
  #   py::arg("grid_dims"),
  #   py::arg("spacing"),
  #   py::arg("cutoff"),
  #   py::arg("sigma"),
  #   py::arg("type_agg"),
  #   "Voxelize a trajectory"
  # );