#include <iostream>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "cpuutils.h"
#include "voxelize.cuh"
#include "marching_observers.cuh"

namespace py = pybind11;


py::array_t<float> do_voxelize(
  py::array_t<float> arr_coords, 
  py::array_t<float> arr_weights, 
  py::array_t<int> grid_dims, 
  const float spacing, 
  const float cutoff, 
  const float sigma,
  const int auto_translate
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

  // NOTE: Directly pass from python to cuda host function
  float *coords = static_cast<float*>(buf_coords.ptr);   
  float *weights = static_cast<float*>(buf_weights.ptr);
  
  // No scaling of the coordinates and only translation is applied
  if (auto_translate){
    translate_coord(coords, atom_nr, dims, spacing);
  }

  // Initialize the return array and launch the computation kernel
  py::array_t<float> result({grid_point_nr});
  for (int i = 0; i < grid_point_nr; i++) result.mutable_at(i) = 0; 
  voxelize_host(
    result.mutable_data(), 
    coords, 
    static_cast<float*>(buf_weights.ptr),
    dims, spacing, 
    atom_nr, 
    cutoff, sigma
  );
  return result;
}

py::array_t<float> do_marching_observers(
  py::array_t<float> arr_coord, 
  py::array_t<float> arr_weights,
  py::array_t<int> arr_dims, 
  const float spacing,
  const float cutoff,
  const int type_obs,
  const int type_agg
){
  py::buffer_info buf_coord = arr_coord.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = arr_dims.request();

  // Get the shape of the input data and dimensions of the grid
  const int *dims = static_cast<int*>(buf_dims.ptr);
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const int frame_nr = buf_coord.shape[0];
  const int atom_nr = buf_coord.shape[1];

  // Check the validity of the input data before launching the kernel
  if (buf_coord.ndim != 3){ throw py::value_error("Error: The input array must have 3 dimensions: (frame_nr, atom_nr, 3)"); }
  int supported_mode[OBSERVABLE_COUNT] = SUPPORTED_OBSERVABLES;
  for (int i= 0; i < OBSERVABLE_COUNT; i++){
    if (type_obs == supported_mode[i]){ break; } 
    else if (i == OBSERVABLE_COUNT - 1){ throw py::value_error("The observable type is not supported"); }
  }
  int supported_agg[AGGREGATION_COUNT] = SUPPORTED_AGGREGATIONS;
  for (int i = 0; i < AGGREGATION_COUNT; i++){
    if (type_agg == supported_agg[i]){ break; }
    else if (i == AGGREGATION_COUNT - 1){ throw py::value_error("The aggregation type is not supported"); }
  }
  if (frame_nr > MAX_FRAME_NUMBER){ throw py::value_error("The number of frames " + std::to_string(frame_nr) + " exceeds the maximum number of frames allowed " + std::to_string(MAX_FRAME_NUMBER) + " frames."); }

  // Current hard coded to 0, 0 for type_obs and type_agg
  py::array_t<float> result({gridpoint_nr});
  for (int i = 0; i < gridpoint_nr; i++) result.mutable_at(i) = 0;
  marching_observer_host(
    result.mutable_data(), 
    static_cast<float*>(buf_coord.ptr), 
    static_cast<float*>(buf_weights.ptr), 
    dims, spacing,
    frame_nr, atom_nr,
    cutoff, type_obs, type_agg
  ); 

  return result;
}


py::array_t<float> do_traj_voxelize(
  py::array_t<float> arr_traj, 
  py::array_t<float> arr_weights,
  py::array_t<int> grid_dims, 
  const float spacing, 
  const float cutoff, 
  const float sigma,
  const int type_agg
){
  py::buffer_info buf_traj = arr_traj.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = grid_dims.request();

  const int *dims = static_cast<int*>(buf_dims.ptr);
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const int frame_nr = buf_traj.shape[0];
  const int atom_nr = buf_traj.shape[1];

  
  // Check the validity of the input data before launching the kernel
  int supported_agg[AGGREGATION_COUNT] = SUPPORTED_AGGREGATIONS;
  for (int i = 0; i < AGGREGATION_COUNT; i++){
    if (type_agg == supported_agg[i]){ break; }
    else if (i == AGGREGATION_COUNT - 1){ throw py::value_error("The aggregation type is not supported"); }
  }

  // Initialize the return array, and launch the computation kernel
  py::array_t<float> result({gridpoint_nr});
  for (int i = 0; i < gridpoint_nr; i++) result.mutable_at(i) = 0;
  trajectory_voxelization_host(
    result.mutable_data(), 
    static_cast<float*>(buf_traj.ptr), 
    static_cast<float*>(buf_weights.ptr), 
    dims, spacing, 
    frame_nr, atom_nr, 
    cutoff, sigma,
    type_agg
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
    py::arg("auto_translate"),
    "Voxelize a set of coordinates and weights"
  );

  m.def("do_marching", &do_marching_observers, 
    py::arg("coords"),
    py::arg("weights"),
    py::arg("dims"),
    py::arg("spacing"),
    py::arg("cutoff"),
    py::arg("type_obs"),
    py::arg("type_agg"),
    "Marching cubes algorithm to create a mesh from a 3D grid"
  );

  m.def("voxelize_traj", &do_traj_voxelize, 
    py::arg("traj"),
    py::arg("weights"),
    py::arg("grid_dims"),
    py::arg("spacing"),
    py::arg("cutoff"),
    py::arg("sigma"),
    py::arg("type_agg"),
    "Voxelize a trajectory"
  );

}

