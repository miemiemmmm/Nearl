// Private bindings for the CPU voxelization fallback.
//
// Deliberately not a peer of nearl.all_actions: this module covers voxelization
// only, and the name says so. Everything user-facing goes through nearl.commands,
// which picks a backend. Built by g++ alone, so it exists on machines with no
// CUDA toolkit, where all_actions cannot be built at all.

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include "constants.h"
#include "cpuutils.h"
#include "voxelize_cpu.h"

namespace py = pybind11;

namespace {
using FloatInput = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntInput = py::array_t<int, py::array::c_style | py::array::forcecast>;

void check_aggregation(const int type_agg) {
  const int supported[AGGREGATION_COUNT] = SUPPORTED_AGGREGATIONS;
  for (int i = 0; i < AGGREGATION_COUNT; i++) {
    if (type_agg == supported[i])
      return;
  }
  throw py::value_error("The aggregation type is not supported");
}

py::array_t<float> cpu_voxelize(FloatInput arr_coords, FloatInput arr_weights, IntInput grid_dims,
                                const float spacing, const float cutoff, const float sigma,
                                const int auto_translate) {
  py::buffer_info buf_coords = arr_coords.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = grid_dims.request();

  if (buf_coords.ndim != 2 || buf_coords.shape[1] != 3)
    throw py::value_error("The coordinates must be shaped (atom_nr, 3)");
  if (buf_coords.shape[0] != buf_weights.shape[0])
    throw py::value_error("Input arrays must have the same length");

  const int *dims = static_cast<int *>(buf_dims.ptr);
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const int atom_nr = static_cast<int>(buf_coords.shape[0]);
  float *coords = static_cast<float *>(buf_coords.ptr);

  if (auto_translate && atom_nr > 0)
    translate_coord(coords, atom_nr, dims, spacing);

  py::array_t<float> result({gridpoint_nr});
  voxelize_host_cpu(result.mutable_data(), coords, static_cast<float *>(buf_weights.ptr), dims,
                    spacing, atom_nr, cutoff, sigma);
  return result;
}

py::array_t<float> cpu_density_flow(FloatInput arr_traj, FloatInput arr_weights, IntInput grid_dims,
                                    const float spacing, const float cutoff, const float sigma,
                                    const int type_agg) {
  py::buffer_info buf_traj = arr_traj.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = grid_dims.request();

  if (buf_traj.ndim != 3 || buf_traj.shape[2] != 3)
    throw py::value_error("The trajectory must be shaped (frame_nr, atom_nr, 3)");
  check_aggregation(type_agg);

  const int *dims = static_cast<int *>(buf_dims.ptr);
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const int frame_nr = static_cast<int>(buf_traj.shape[0]);
  const int atom_nr = static_cast<int>(buf_traj.shape[1]);

  if (frame_nr > MAX_FRAME_NUMBER)
    throw py::value_error("The number of frames " + std::to_string(frame_nr) +
                          " exceeds the maximum number of frames allowed " +
                          std::to_string(MAX_FRAME_NUMBER) + " frames.");

  py::array_t<float> result({gridpoint_nr});
  trajectory_voxelization_host_cpu(result.mutable_data(), static_cast<float *>(buf_traj.ptr),
                                   static_cast<float *>(buf_weights.ptr), dims, spacing, frame_nr,
                                   atom_nr, cutoff, sigma, type_agg);
  return result;
}
} // namespace


PYBIND11_MODULE(_voxelize_cpu, m) {
  m.doc() = "CPU voxelization fallback. Private: use nearl.commands instead.";
  m.def("frame_voxelize", &cpu_voxelize, py::arg("coords"), py::arg("weights"),
        py::arg("grid_dims"), py::arg("spacing"), py::arg("cutoff"), py::arg("sigma"),
        py::arg("auto_translate"), "Voxelize a set of coordinates and weights on the CPU");
  m.def("density_flow", &cpu_density_flow, py::arg("traj"), py::arg("weights"),
        py::arg("grid_dims"), py::arg("spacing"), py::arg("cutoff"), py::arg("sigma"),
        py::arg("type_agg"), "Voxelize a trajectory on the CPU");
}
