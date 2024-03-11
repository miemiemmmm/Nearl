from . import all_actions
import numpy as np 

__all__ = [
  "voxelize_trajectory",
  "voxelize_coords",
  "marching_observers"
]


def voxelize_coords(coords, weights, grid_dims, spacing, cutoff, sigma):
  # if coords.dtype != 'float32':
  #   coords = coords.astype('float32')
  # if weights.dtype != 'float32':
  #   weights = weights.astype('float32')
  coords = np.array(coords, dtype=np.float32)
  weights = np.array(weights, dtype=np.float32)
  grid_dims = np.array(grid_dims, dtype=np.int32)
  spacing = float(spacing)
  cutoff = float(cutoff)
  sigma = float(sigma)
  ret_arr = all_actions.do_voxelize(coords, weights, grid_dims, spacing, cutoff, sigma)
  return ret_arr.reshape(grid_dims)


def marching_observers(coords, grid_dims, spacing, cutoff):
  # if coords.dtype != 'float32':
  #   coords = coords.astype('float32')
  ret_arr = all_actions.do_marching(coords, grid_dims, spacing, cutoff)
  return ret_arr.reshape(grid_dims)

def voxelize_trajectory(traj, weights, grid_dims, spacing, interval, cutoff, sigma):
  # Check the data type of the inputs; All arrays should be of type np.float32
  # if traj.dtype != 'float32':
  #   traj = traj.astype('float32')
  # if weights.dtype != 'float32':
  #   weights = weights.astype('float32')
  ret_arr = all_actions.voxelize_traj(traj, weights, grid_dims, spacing, interval, cutoff, sigma)
  return ret_arr.reshape(grid_dims)


# PYBIND11_MODULE(all_actions, m) {
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

#   m.def("voxelize_traj", &voxelize_traj_host, 
#     py::arg("traj"),
#     py::arg("weights"),
#     py::arg("grid_dims"),
#     py::arg("spacing"),
#     py::arg("interval"),
#     py::arg("cutoff"),
#     py::arg("sigma"),
#     "Voxelize a trajectory"
#   );

# }