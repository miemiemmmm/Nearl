#include <iostream>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "marching_observers.cuh"

namespace py = pybind11;



py::array_t<float> do_marching(py::array_t<float> arr_coord, 
py::array_t<int> arr_dims, py::array_t<float> arr_spacing,
const float cutoff){
  py::buffer_info buf_coord = arr_coord.request();
  py::buffer_info buf_dims = arr_dims.request();
  py::buffer_info buf_spacing = arr_spacing.request();
  // Get the shape of the input data
  int frame_nr = buf_coord.shape[0];
  int atom_nr = buf_coord.shape[1];

  // Get the required dimensions from arr_dims
  int dims[3] = {
    *static_cast<int*>(buf_dims.ptr),
    *static_cast<int*>(buf_dims.ptr + sizeof(int)),
    *static_cast<int*>(buf_dims.ptr + 2*sizeof(int))
  };
  
  const int grid_nr = dims[0] * dims[1] * dims[2];

  // ##############################################################
  // # Debugging Check if values are correct passed 
  std::cout << "grid_nr: " << grid_nr << std::endl;
  std::cout << "frame_nr: " << frame_nr << std::endl;
  std::cout << "atom_nr: " << atom_nr << std::endl;
  std::cout << "cutoff: " << cutoff << std::endl;
  std::cout << "dimensions: " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
  float mean_x = 0;
  for (int i = 0; i < frame_nr; i++){
    for (int j = 0; j < atom_nr; j++){
      mean_x += *static_cast<float*>(buf_coord.ptr + i * atom_nr * 3 * sizeof(float) + j * 3 * sizeof(float));
    }
  }
  mean_x /= frame_nr * atom_nr;
  std::cout << "mean_x: " << mean_x << std::endl;
  float mean_y = 0;
  for (int i = 0; i < frame_nr; i++){
    for (int j = 0; j < atom_nr; j++){
      mean_y += *static_cast<float*>(buf_coord.ptr + i * atom_nr * 3 * sizeof(float) + j * 3 * sizeof(float) + sizeof(float));
    }
  }
  mean_y /= frame_nr * atom_nr;
  std::cout << "mean_y: " << mean_y << std::endl;
  float mean_z = 0;
  for (int i = 0; i < frame_nr; i++){
    for (int j = 0; j < atom_nr; j++){
      mean_z += *static_cast<float*>(buf_coord.ptr + i * atom_nr * 3 * sizeof(float) + j * 3 * sizeof(float) + 2 * sizeof(float));
    }
  }
  mean_z /= frame_nr * atom_nr;
  std::cout << "mean_z: " << mean_z << std::endl;
  // ##############################################################

  // Initialize the grid with zeros
  float *marched_grid = new float[grid_nr];
  for (int i = 0; i < grid_nr; i++){ marched_grid[i] = 0; }

  // Current hard coded to 0, 0 for type_obs and type_agg
  marching_observer_host(
    marched_grid, static_cast<float*>(buf_coord.ptr),
    static_cast<int*>(buf_dims.ptr), static_cast<float*>(buf_spacing.ptr),
    frame_nr, atom_nr,
    cutoff, 0, 0
  ); 

  py::array_t<float> result({grid_nr});
  py::buffer_info buf_result = result.request();
  float *ptr_result = static_cast<float*>(buf_result.ptr);
  for (int i = 0; i < grid_nr; i++){
    ptr_result[i] = marched_grid[i];
  }
  delete[] marched_grid;
  return result;
}
 




PYBIND11_MODULE(marching_observers, m) {
  m.def("do_marching", &do_marching, 
    py::arg("coords"),
    py::arg("dims"),
    py::arg("spacing"),
    py::arg("cutoff"),
    "Perform the marching observers on the given data."
  );

}
