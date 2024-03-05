#include <iostream>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"


#include "voxelize.cuh"
#include "marching_observers.cuh"
#include "cpuutils.h"


namespace py = pybind11;


py::array_t<float> do_voxelize(
  py::array_t<float> arr_coords, 
  py::array_t<float> arr_weights, 
  py::array_t<int> grid_dims, 
  const float spacing, 
  const float cutoff, 
  const float sigma
){
  py::buffer_info buf_coords = arr_coords.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = grid_dims.request();

  if (buf_coords.shape[0] != buf_weights.shape[0]){
    std::cerr << "Input arrays must have the same length" << std::endl;
    return py::array_t<float>({0});
  }

  // Convert the input arrays to float
  int *dims = static_cast<int*>(buf_dims.ptr);
  unsigned int grid_point_nr = dims[0] * dims[1] * dims[2];

  int atom_nr = buf_coords.shape[0];
  float *coords = new float[atom_nr * 3];
  float *weights = new float[atom_nr];

  for (int i = 0; i < atom_nr; i++){
    coords[i * 3] = *(static_cast<float*>(buf_coords.ptr) + i * 3);
    coords[i * 3 + 1] = *(static_cast<float*>(buf_coords.ptr) + i * 3 + 1);
    coords[i * 3 + 2] = *(static_cast<float*>(buf_coords.ptr) + i * 3 + 2);
    weights[i] = *(static_cast<float*>(buf_weights.ptr) + i);
  }
  
  // No scaling of the coordinates and only translation is applied
  translate_coord(coords, atom_nr, dims, spacing);
  // float new_cutoff = cutoff / spacing;
  // float new_sigma = sigma / spacing;
  

  // Initialize the return array and launch the computation kernel
  py::array_t<float> result({grid_point_nr});
  for (int i = 0; i < grid_point_nr; i++) result.mutable_at(i) = 0; 
  voxelize_host(result.mutable_data(), coords, weights, static_cast<int*>(buf_dims.ptr), atom_nr, spacing, cutoff, sigma);
  delete[] coords;
  delete[] weights;
  return result;
}

py::array_t<float> do_marching_observers(
  py::array_t<float> arr_coord, 
  py::array_t<int> arr_dims, 
  float spacing,
  const float cutoff
){
  py::buffer_info buf_coord = arr_coord.request();
  py::buffer_info buf_dims = arr_dims.request();

  // Coordinate has to be three dimensions shaped by (frame_nr, atom_nr, 3)
  if (buf_coord.ndim != 3){
    std::cerr << "Warning: The input array must have 3 dimensions: (frame_nr, atom_nr, 3)" << std::endl;
    return py::array_t<float>({0});
  }
  
  // Get the shape of the input data and dimensions of the grid
  int *dims = static_cast<int*>(buf_dims.ptr);
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  int frame_nr = buf_coord.shape[0];
  int atom_nr = buf_coord.shape[1];

  // Current hard coded to 0, 0 for type_obs and type_agg
  py::array_t<float> result({gridpoint_nr});
  for (int i = 0; i < gridpoint_nr; i++) result.mutable_at(i) = 0;
  marching_observer_host(
    result.mutable_data(), static_cast<float*>(buf_coord.ptr),
    dims, spacing,
    frame_nr, atom_nr,
    cutoff, 0, 0
  ); 

  return result;
}


py::array_t<float> voxelize_traj_host(
  py::array_t<float> arr_traj, 
  py::array_t<float> arr_weights,
  py::array_t<int> grid_dims, 
  const float spacing, 
  const int interval,
  const float cutoff, 
  const float sigma
){
  py::buffer_info buf_traj = arr_traj.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = grid_dims.request();

  if (buf_traj.shape[0] != buf_weights.shape[0]){
    std::cerr << "Input arrays must have the same length" << std::endl;
    return py::array_t<float>({0});
  }

  int frame_nr = buf_traj.shape[0];
  int atom_nr = buf_traj.shape[1];
  int *dims = static_cast<int*>(buf_dims.ptr);
  unsigned int gridpoint_nr = dims[0] * dims[1] * dims[2];
  int dyn_pool_nr = frame_nr / interval;
  if (dyn_pool_nr * interval != frame_nr){
    std::cerr << "Warning: The interval must be a divisor of the frame number, the last frames will be ignored" << std::endl;
    std::cerr << "The output dimension will be " << dyn_pool_nr << " * " << gridpoint_nr << std::endl;
  }
  py::array_t<float> result({gridpoint_nr * dyn_pool_nr});
  for (int i = 0; i < gridpoint_nr * dyn_pool_nr; i++) result.mutable_at(i) = 0;
  trajectory_voxelization_host(
    result.mutable_data(), 
    static_cast<float*>(buf_traj.ptr), 
    static_cast<float*>(buf_weights.ptr), 
    dims,
    frame_nr, 
    atom_nr, 
    interval, 
    spacing,
    cutoff, 
    sigma
  ); 
  return result;
}









PYBIND11_MODULE(all_actions, m) {
  m.def("do_voxelize", &do_voxelize, 
    py::arg("coords"),
    py::arg("weights"),
    py::arg("grid_dims"),
    py::arg("spacing"),
    py::arg("cutoff"),
    py::arg("sigma"),
    "Voxelize a set of coordinates and weights"
  );

  m.def("do_marching", &do_marching_observers, 
    py::arg("coords"),
    py::arg("dims"),
    py::arg("spacing"),
    py::arg("cutoff"),
    "Marching cubes algorithm to create a mesh from a 3D grid"
  );


  m.def("voxelize_traj", &voxelize_traj_host, 
    py::arg("traj"),
    py::arg("weights"),
    py::arg("grid_dims"),
    py::arg("spacing"),
    py::arg("interval"),
    py::arg("cutoff"),
    py::arg("sigma"),
    "Voxelize a trajectory"
  );

}

