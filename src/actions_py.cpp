// Created by: Yang Zhang
// Description: Python bindings for all actions in the library
// Functionality:
// 1. frame_voxelize: Voxelization of a single frame
// 2. frame_observation: Compute the observable for a single frame
// 3. marching_observer: Marching observer algorithm for a slice of frames/trajectory
// 4. density_flow: Property density flow for a slice of frames/trajectory
// 5. aggregate: Aggregate the observable from a slice of frames to a single frame
// 6. summation: Summation of the array on GPU
//
// GIL: every kernel launch below releases the GIL for the duration of the
// *_host call, so the CPU producer thread in nearl.featurizer can prepare the
// next task while a kernel is in flight. Each released block touches raw
// pointers only: the result array is allocated and zeroed before the release,
// and no Python object is read or written until after it.
//
// That makes this module re-entrant from Python, which the global DeviceContext
// is not -- it is one set of device buffers shared by every call. Exactly one
// thread may be inside these functions at a time. Featurizer.run honours that:
// a single consumer thread launches every kernel. A second consumer would need
// its own context.

#include <iostream>
#include <memory>
#include <algorithm>
#include <utility>
#include <cstring>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "constants.h"
#include "cpuutils.h" // For translate_coord
#include "gpuutils.cuh"
#include "voxelize.cuh"
#include "marching_observers.cuh"
#include "dlpack_interop.h"


namespace py = pybind11;


namespace {
using FloatInput = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntInput = py::array_t<int, py::array::c_style | py::array::forcecast>;

// The context owns one reusable pinned output buffer; collection copies it into NumPy memory.
class CommandExecution {
public:
  CommandExecution(size_t count, bool reduce = false) : reduce_(reduce) {
    init_global_device_context();
    ctx_ = get_global_device_context();
    if (ctx_->pending())
      throw std::runtime_error("Collect the previous CUDA result before dispatch");
    output_data_ = static_cast<float *>(ctx_->get_host_buffer(
        std::max(count, size_t(1)) * sizeof(float), static_cast<size_t>(BufferSlot::SCRATCH)));
    output_count_ = count;
    ctx_->begin_call();
    active_ = true;
  }

  CommandExecution(const CommandExecution &) = delete;
  CommandExecution &operator=(const CommandExecution &) = delete;
  CommandExecution &operator=(CommandExecution &&) = delete;

  // Transfer the pending call so the moved-from destructor cannot drain it.
  CommandExecution(CommandExecution &&other) noexcept
      : ctx_(std::exchange(other.ctx_, nullptr)), active_(std::exchange(other.active_, false)),
        reduce_(other.reduce_), output_data_(std::exchange(other.output_data_, nullptr)),
        output_count_(other.output_count_), value_(std::move(other.value_)) {}

  ~CommandExecution() {
    if (active_)
      ctx_->cancel_call();
  }
  float *data() { return output_data_; }
  py::object result() {
    if (active_) {
      {
        // Release the GIL across the wait, not just across the launch. Since the
        // dispatch became asynchronous the *_host calls return immediately, so
        // the release inside them spans almost nothing while the real kernel
        // time is spent here in end_call() -> synchronize(). Holding the GIL
        // through it freezes the featurizer's CPU producer for every kernel and
        // costs exactly the overlap the pipeline exists to win.
        py::gil_scoped_release release;
        ctx_->end_call();
      }
      active_ = false;
    }
    if (!value_) {
      if (reduce_) {
        float sum = 0;
        for (size_t i = 0; i < output_count_; ++i)
          sum += output_data_[i];
        value_ = py::float_(sum);
      } else {
        py::array_t<float> output(static_cast<py::ssize_t>(output_count_));
        // Copy out before reusing the context's pinned buffer; NumPy must own
        // independent storage because the next command overwrites that buffer.
        if (output_count_)
          std::memcpy(output.mutable_data(), output_data_, output_count_ * sizeof(float));
        value_ = std::move(output);
      }
    }
    return value_;
  }

private:
  DeviceContext *ctx_ = nullptr;
  bool active_ = false;
  bool reduce_;
  float *output_data_ = nullptr;
  size_t output_count_ = 0;
  py::object value_;
};

// Keep the public synchronous API and expose internal deferred variants for the featurizer.
template <class... Args, class... Extra>
void bind_action(py::module_ &m, const char *name, CommandExecution (*dispatch)(Args...),
                 const Extra &...extra) {
  m.def(name, [dispatch](Args... args) { return dispatch(args...).result(); }, extra...);
  m.def((std::string("_dispatch_") + name).c_str(), dispatch, extra...);
}


/**
 * @brief
 * Perform the voxelization of a trajectory
 *
 * @param arr_traj: The input array of coordinates
 * @param arr_weights: The input array of weights
 * @param grid_dims: The dimensions of the grid
 * @param spacing: The spacing between the grid points
 * @param cutoff: The cutoff distance for the voxelization
 * @param sigma: The sigma value for the Gaussian kernel
 * @param type_agg: The type of aggregation to use
 * @return py::array_t<float>: The output array of the observable
 *
 * @note
 * The input array of coordinates should be one single frame shaped (atom_nr, 3)
 * The input array of weights should be one single frame shaped (atom_nr)
 * The input array of dimensions should be shaped (3)
 * The output array is shaped (grid_dims[0] * grid_dims[1] * grid_dims[2]). Need to reshape it in
 * the python code.
 */
CommandExecution do_voxelize(FloatInput arr_coords, FloatInput arr_weights, IntInput grid_dims,
                             const float spacing, const float cutoff, const float sigma,
                             const int auto_translate) {
  py::buffer_info buf_coords = arr_coords.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = grid_dims.request();

  if (buf_coords.shape[0] != buf_weights.shape[0]) {
    std::cerr << "Input arrays must have the same length" << std::endl;
    return CommandExecution(0);
  }

  // Convert the input arrays to float
  int *dims = static_cast<int *>(buf_dims.ptr);
  unsigned int grid_point_nr = dims[0] * dims[1] * dims[2];
  int atom_nr = buf_coords.shape[0];

  // NOTE: Directly pass from python to cuda host function
  float *coords = static_cast<float *>(buf_coords.ptr);
  float *weights = static_cast<float *>(buf_weights.ptr);

  // No scaling of the coordinates and only translation is applied
  if (auto_translate) {
    translate_coord(coords, atom_nr, dims, spacing);
  }

  // Initialize the return array and launch the computation kernel
  CommandExecution result(grid_point_nr);
  {
    py::gil_scoped_release release;
    voxelize_host(result.data(), coords, static_cast<float *>(buf_weights.ptr), dims, spacing,
                  atom_nr, cutoff, sigma);
  }
  return result;
}


/**
 * @brief
 * Perform the marching observers algorithm to convert a slice of coordinate sets to a 3D grid
 *
 * @param arr_coord: The input array of coordinates, shaped (frame_nr, atom_nr, 3)
 * @param arr_weights: The input array of weights
 * @param arr_dims: The dimensions of the grid
 * @param spacing: The spacing between the grid points
 * @param cutoff: The cutoff distance for the marching observers algorithm
 * @param type_obs: The type of observable to compute
 * @param type_agg: The type of aggregation to use
 * @return py::array_t<float>: The output array of the observable
 *
 * @note
 * The input array of coordinates must have the shape (frame_nr, atom_nr, 3)
 * The input array of weights must have the shape (frame_nr, atom_nr)
 * The input array of dimensions must have the shape (3)
 * The output array is shaped (arr_dims[0] * arr_dims[1] * arr_dims[2]). Need to reshape it to in
 * the python code.
 *
 */
CommandExecution do_marching_observers(FloatInput arr_coord, FloatInput arr_weights,
                                       IntInput arr_dims, const float spacing, const float cutoff,
                                       const int type_obs, const int type_agg) {
  py::buffer_info buf_coord = arr_coord.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = arr_dims.request();

  // Get the shape of the input data and dimensions of the grid
  const int *dims = static_cast<int *>(buf_dims.ptr);
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const int frame_nr = buf_coord.shape[0];
  const int atom_nr = buf_coord.shape[1];

  // Check the validity of the input data before launching the kernel
  if (buf_coord.ndim != 3) {
    throw py::value_error("Error: The input array must have 3 dimensions: (frame_nr, atom_nr, 3)");
  }
  int supported_mode[OBSERVABLE_COUNT] = SUPPORTED_OBSERVABLES;
  for (int i = 0; i < OBSERVABLE_COUNT; i++) {
    if (type_obs == supported_mode[i]) {
      break;
    } else if (i == OBSERVABLE_COUNT - 1) {
      throw py::value_error("The observable type is not supported");
    }
  }
  int supported_agg[AGGREGATION_COUNT] = SUPPORTED_AGGREGATIONS;
  for (int i = 0; i < AGGREGATION_COUNT; i++) {
    if (type_agg == supported_agg[i]) {
      break;
    } else if (i == AGGREGATION_COUNT - 1) {
      throw py::value_error("The aggregation type is not supported");
    }
  }

  // TODO: Eliminate this constraint in the future
  if (frame_nr > MAX_FRAME_NUMBER) {
    throw py::value_error("The number of frames " + std::to_string(frame_nr) +
                          " exceeds the maximum number of frames allowed " +
                          std::to_string(MAX_FRAME_NUMBER) + " frames.");
  }

  // Current hard coded to 0, 0 for type_obs and type_agg
  CommandExecution result(gridpoint_nr);
  {
    py::gil_scoped_release release;
    marching_observer_host(result.data(), static_cast<float *>(buf_coord.ptr),
                           static_cast<float *>(buf_weights.ptr), dims, spacing, frame_nr, atom_nr,
                           cutoff, type_obs, type_agg);
  }

  return result;
}


/**
 * @brief
 * Perform property density flow for a slice of frames in a trajectory
 *
 * @param arr_traj: The input array of coordinates
 * @param arr_weights: The input array of weights
 * @param grid_dims: The dimensions of the grid
 * @param spacing: The spacing between the grid points
 * @param cutoff: The cutoff distance for the voxelization
 * @param sigma: The sigma value for the Gaussian kernel
 * @param type_agg: The type of aggregation to use
 * @return py::array_t<float>: The output array of the observable
 *
 * @note
 * The input array of coordinates must have the shape (frame_nr, atom_nr, 3)
 * The input array of weights must have the shape (frame_nr, atom_nr)
 * The input array of dimensions must have the shape (3)
 * The output array is shaped (arr_dims[0] * arr_dims[1] * arr_dims[2]). Need to reshape it to in
 * the python code.
 */
CommandExecution do_traj_voxelize(FloatInput arr_traj, FloatInput arr_weights, IntInput grid_dims,
                                  const float spacing, const float cutoff, const float sigma,
                                  const int type_agg) {
  py::buffer_info buf_traj = arr_traj.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = grid_dims.request();

  const int *dims = static_cast<int *>(buf_dims.ptr);
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const int frame_nr = buf_traj.shape[0];
  const int atom_nr = buf_traj.shape[1];

  // Check the validity of the input data before launching the kernel
  int supported_agg[AGGREGATION_COUNT] = SUPPORTED_AGGREGATIONS;
  for (int i = 0; i < AGGREGATION_COUNT; i++) {
    if (type_agg == supported_agg[i]) {
      break;
    } else if (i == AGGREGATION_COUNT - 1) {
      throw py::value_error("The aggregation type is not supported");
    }
  }

  // Initialize the return array, and launch the computation kernel
  CommandExecution result(gridpoint_nr);
  {
    py::gil_scoped_release release;
    trajectory_voxelization_host(result.data(), static_cast<float *>(buf_traj.ptr),
                                 static_cast<float *>(buf_weights.ptr), dims, spacing, frame_nr,
                                 atom_nr, cutoff, sigma, type_agg);
  }
  return result;
}


CommandExecution do_aggregation(FloatInput arr, const int type_agg) {
  py::buffer_info buf_arr = arr.request();

  const int frame_nr = buf_arr.shape[0];
  const int gridpoint_nr = buf_arr.shape[1];

  CommandExecution result(gridpoint_nr);

  {
    py::gil_scoped_release release;
    aggregate_host(static_cast<float *>(buf_arr.ptr), result.data(), frame_nr, gridpoint_nr,
                   type_agg);
  }

  return result;
}

CommandExecution do_summation(FloatInput arr) {
  py::buffer_info buf_arr = arr.request();
  const int arr_length = buf_arr.shape[0];
  CommandExecution result((arr_length + BLOCK_SIZE - 1) / BLOCK_SIZE, true);
  {
    py::gil_scoped_release release;
    if (arr_length)
      sum_reduction_dispatch(static_cast<float *>(buf_arr.ptr), arr_length, result.data());
  }
  return result;
}

CommandExecution do_frame_observation(FloatInput coord_arr, FloatInput weight_arr,
                                      IntInput dims_arr, const float spacing, const float cutoff,
                                      const int type_obs) {
  py::buffer_info buf_coords = coord_arr.request();
  py::buffer_info buf_weights = weight_arr.request();
  py::buffer_info buf_dims = dims_arr.request();

  const int *dims = static_cast<int *>(buf_dims.ptr);
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const int atom_nr = buf_coords.shape[0];
  CommandExecution result(gridpoint_nr);

  {
    py::gil_scoped_release release;
    observe_frame_host(result.data(), static_cast<float *>(buf_coords.ptr),
                       static_cast<float *>(buf_weights.ptr), static_cast<int *>(buf_dims.ptr),
                       spacing, atom_nr, cutoff, type_obs);
  }

  return result;
}

} // namespace


void do_init_context() { init_global_device_context(); }

void do_finalize_context() { finalize_global_device_context(); }

bool do_context_valid() { return global_device_context_valid(); }

size_t do_buffer_capacity(const std::string &name) {
  auto *ctx = get_global_device_context();
  if (!ctx || !ctx->valid())
    return 0;
  if (name == "coords")
    return ctx->buffer_capacity(static_cast<size_t>(BufferSlot::COORDS));
  if (name == "weights")
    return ctx->buffer_capacity(static_cast<size_t>(BufferSlot::WEIGHTS));
  if (name == "output")
    return ctx->buffer_capacity(static_cast<size_t>(BufferSlot::OUTPUT_GRID));
  if (name == "trajectory")
    return ctx->buffer_capacity(static_cast<size_t>(BufferSlot::TRAJ_DYNAMICS));
  throw py::value_error("Unknown device buffer: " + name);
}

size_t do_host_buffer_capacity(const std::string &name) {
  auto *ctx = get_global_device_context();
  if (!ctx || !ctx->valid())
    return 0;
  if (name == "coords")
    return ctx->host_buffer_capacity(static_cast<size_t>(BufferSlot::COORDS));
  if (name == "weights")
    return ctx->host_buffer_capacity(static_cast<size_t>(BufferSlot::WEIGHTS));
  if (name == "dims")
    return ctx->host_buffer_capacity(static_cast<size_t>(BufferSlot::DIMS));
  if (name == "output")
    return ctx->host_buffer_capacity(static_cast<size_t>(BufferSlot::SCRATCH));
  throw py::value_error("Unknown pinned host buffer: " + name);
}


// Where a grid is written: either the caller's own DLPack buffer, or a fresh
// DeviceArray handed back through DLPack. One class so each command has a
// single body instead of one per destination.
class GridDestination {
public:
  GridDestination(const py::object &out, const int *dims) {
    const int device = nearl_dlpack::current_device();
    if (out.is_none()) {
      allocated_ = std::unique_ptr<nearl_dlpack::DeviceArray>(
          new nearl_dlpack::DeviceArray(std::vector<int64_t>{dims[0], dims[1], dims[2]}, device));
      data_ = allocated_->data();
      return;
    }
    DeviceContext *ctx = get_global_device_context();
    cudaStream_t stream = (ctx && ctx->valid()) ? ctx->stream() : nullptr;
    imported_ = nearl_dlpack::import_output(out, stream, dims, device);
    data_ = imported_.data();
  }

  float *data() const { return data_; }

  // None when the caller supplied the buffer: nearl.commands returns the
  // caller's own object in that case, as numpy's out= does.
  py::object result() {
    if (allocated_)
      return py::cast(allocated_.release(), py::return_value_policy::take_ownership);
    return py::none();
  }

private:
  std::unique_ptr<nearl_dlpack::DeviceArray> allocated_;
  nearl_dlpack::ImportedTensor imported_;
  float *data_ = nullptr;
};


// A wrong length here would size the grid from whatever follows in memory and
// the kernel would write past the destination.
const int *checked_dims(const py::buffer_info &buf) {
  if (buf.size != 3)
    throw py::value_error("grid_dims must have exactly 3 entries, got " + std::to_string(buf.size));
  return static_cast<const int *>(buf.ptr);
}


py::object do_voxelize_dlpack(FloatInput arr_coords, FloatInput arr_weights, IntInput grid_dims,
                              const float spacing, const float cutoff, const float sigma,
                              py::object out) {
  py::buffer_info buf_coords = arr_coords.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = grid_dims.request();

  if (buf_coords.shape[0] != buf_weights.shape[0]) {
    throw py::value_error("Input arrays must have the same length");
  }

  const int *dims = checked_dims(buf_dims);
  int atom_nr = buf_coords.shape[0];

  // Destroyed after the GIL is reacquired below: releasing an imported tensor
  // decrefs the producer's Python object.
  GridDestination dest(out, dims);
  {
    py::gil_scoped_release release;
    voxelize_host_into(dest.data(), static_cast<float *>(buf_coords.ptr),
                       static_cast<float *>(buf_weights.ptr), dims, spacing, atom_nr, cutoff,
                       sigma);
  }
  return dest.result();
}


py::object do_traj_voxelize_dlpack(FloatInput arr_traj, FloatInput arr_weights, IntInput grid_dims,
                                   const float spacing, const float cutoff, const float sigma,
                                   const int type_agg, py::object out) {
  py::buffer_info buf_traj = arr_traj.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = grid_dims.request();

  if (buf_traj.ndim != 3) {
    throw py::value_error("The trajectory must have 3 dimensions: (frame_nr, atom_nr, 3)");
  }

  const int *dims = checked_dims(buf_dims);
  int frame_nr = buf_traj.shape[0];
  int atom_nr = buf_traj.shape[1];

  int supported_agg[AGGREGATION_COUNT] = SUPPORTED_AGGREGATIONS;
  for (int i = 0; i < AGGREGATION_COUNT; i++) {
    if (type_agg == supported_agg[i]) {
      break;
    } else if (i == AGGREGATION_COUNT - 1) {
      throw py::value_error("The aggregation type is not supported");
    }
  }

  GridDestination dest(out, dims);
  {
    py::gil_scoped_release release;
    trajectory_voxelization_host_into(dest.data(), static_cast<float *>(buf_traj.ptr),
                                      static_cast<float *>(buf_weights.ptr), dims, spacing,
                                      frame_nr, atom_nr, cutoff, sigma, type_agg);
  }
  return dest.result();
}


py::object do_marching_observers_dlpack(FloatInput arr_coord, FloatInput arr_weights,
                                        IntInput arr_dims, const float spacing, const float cutoff,
                                        const int type_obs, const int type_agg, py::object out) {
  py::buffer_info buf_coord = arr_coord.request();
  py::buffer_info buf_weights = arr_weights.request();
  py::buffer_info buf_dims = arr_dims.request();

  if (buf_coord.ndim != 3) {
    throw py::value_error("Error: The input array must have 3 dimensions: (frame_nr, atom_nr, 3)");
  }

  const int *dims = checked_dims(buf_dims);
  const int frame_nr = buf_coord.shape[0];
  const int atom_nr = buf_coord.shape[1];

  int supported_mode[OBSERVABLE_COUNT] = SUPPORTED_OBSERVABLES;
  for (int i = 0; i < OBSERVABLE_COUNT; i++) {
    if (type_obs == supported_mode[i]) {
      break;
    } else if (i == OBSERVABLE_COUNT - 1) {
      throw py::value_error("The observable type is not supported");
    }
  }
  int supported_agg[AGGREGATION_COUNT] = SUPPORTED_AGGREGATIONS;
  for (int i = 0; i < AGGREGATION_COUNT; i++) {
    if (type_agg == supported_agg[i]) {
      break;
    } else if (i == AGGREGATION_COUNT - 1) {
      throw py::value_error("The aggregation type is not supported");
    }
  }

  if (frame_nr > MAX_FRAME_NUMBER) {
    throw py::value_error("The number of frames " + std::to_string(frame_nr) +
                          " exceeds the maximum number of frames allowed " +
                          std::to_string(MAX_FRAME_NUMBER) + " frames.");
  }

  GridDestination dest(out, dims);
  {
    py::gil_scoped_release release;
    marching_observer_host_into(dest.data(), static_cast<float *>(buf_coord.ptr),
                                static_cast<float *>(buf_weights.ptr), dims, spacing, frame_nr,
                                atom_nr, cutoff, type_obs, type_agg);
  }
  return dest.result();
}


py::object do_frame_observation_dlpack(FloatInput coord_arr, FloatInput weight_arr,
                                       IntInput dims_arr, const float spacing, const float cutoff,
                                       const int type_obs, py::object out) {
  py::buffer_info buf_coords = coord_arr.request();
  py::buffer_info buf_weights = weight_arr.request();
  py::buffer_info buf_dims = dims_arr.request();

  const int *dims = checked_dims(buf_dims);
  const int atom_nr = buf_coords.shape[0];

  if (buf_coords.shape[0] != buf_weights.shape[0]) {
    throw py::value_error("Input arrays must have the same length");
  }

  GridDestination dest(out, dims);
  {
    py::gil_scoped_release release;
    observe_frame_host_into(dest.data(), static_cast<float *>(buf_coords.ptr),
                            static_cast<float *>(buf_weights.ptr), dims, spacing, atom_nr, cutoff,
                            type_obs);
  }
  return dest.result();
}


PYBIND11_MODULE(all_actions, m) {
  py::class_<CommandExecution>(m, "_CommandExecution").def("result", &CommandExecution::result);
  bind_action(m, "frame_voxelize", &do_voxelize, py::arg("coords"), py::arg("weights"),
              py::arg("grid_dims"), py::arg("spacing"), py::arg("cutoff"), py::arg("sigma"),
              py::arg("auto_translate"), "Voxelize a set of coordinates and weights");

  bind_action(m, "frame_observation", &do_frame_observation, py::arg("coords"), py::arg("weights"),
              py::arg("dims"), py::arg("spacing"), py::arg("cutoff"), py::arg("type_obs"),
              "Compute the observable for a single frame");

  bind_action(m, "marching_observer", &do_marching_observers, py::arg("coords"), py::arg("weights"),
              py::arg("dims"), py::arg("spacing"), py::arg("cutoff"), py::arg("type_obs"),
              py::arg("type_agg"), "Marching cubes algorithm to create a mesh from a 3D grid");

  bind_action(m, "density_flow", &do_traj_voxelize, py::arg("traj"), py::arg("weights"),
              py::arg("grid_dims"), py::arg("spacing"), py::arg("cutoff"), py::arg("sigma"),
              py::arg("type_agg"), "Voxelize a trajectory");

  bind_action(m, "aggregate", &do_aggregation, py::arg("arr"), py::arg("type_agg"),
              "Aggregate the observable (nframes, ngridpoints) to a single frame (ngridpoints)");

  bind_action(m, "summation", &do_summation, py::arg("arr"), "Summation of the array on GPU");

  m.def("init_context", &do_init_context,
        "Create the persistent DeviceContext (CUDA stream + cached buffers).");
  m.def("finalize_context", &do_finalize_context,
        "Destroy the persistent DeviceContext and release cached GPU memory.");
  m.def("context_valid", &do_context_valid,
        "Return True if the persistent DeviceContext is active.");
  m.def("_buffer_capacity", &do_buffer_capacity, py::arg("name"),
        "Return the current capacity of a named reusable device buffer.");
  m.def("_host_buffer_capacity", &do_host_buffer_capacity, py::arg("name"),
        "Return the current capacity of a named reusable pinned host buffer.");

  nearl_dlpack::register_device_array(m);

  // out=None allocates a DeviceArray and returns it; passing a DLPack-capable
  // object writes into that object's memory and returns None.
  m.def("frame_voxelize_dlpack", &do_voxelize_dlpack, py::arg("coords"), py::arg("weights"),
        py::arg("grid_dims"), py::arg("spacing"), py::arg("cutoff"), py::arg("sigma"),
        py::kw_only(), py::arg("out") = py::none(),
        "Voxelize a set of coordinates and weights into CUDA memory.");

  m.def("frame_observation_dlpack", &do_frame_observation_dlpack, py::arg("coords"),
        py::arg("weights"), py::arg("dims"), py::arg("spacing"), py::arg("cutoff"),
        py::arg("type_obs"), py::kw_only(), py::arg("out") = py::none(),
        "Compute the observable for a single frame into CUDA memory.");

  m.def("marching_observer_dlpack", &do_marching_observers_dlpack, py::arg("coords"),
        py::arg("weights"), py::arg("dims"), py::arg("spacing"), py::arg("cutoff"),
        py::arg("type_obs"), py::arg("type_agg"), py::kw_only(), py::arg("out") = py::none(),
        "Marching observers on a frame slice into CUDA memory.");

  m.def("density_flow_dlpack", &do_traj_voxelize_dlpack, py::arg("traj"), py::arg("weights"),
        py::arg("grid_dims"), py::arg("spacing"), py::arg("cutoff"), py::arg("sigma"),
        py::arg("type_agg"), py::kw_only(), py::arg("out") = py::none(),
        "Voxelize a trajectory into CUDA memory.");
}
