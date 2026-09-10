// Created by: Yang Zhang
// Description: DLPack import/export for the CUDA grids produced by this module.
//
// Two directions, both needed:
//
//   export -- DeviceArray owns a cudaMalloc'd grid and implements __dlpack__ /
//             __dlpack_device__, so torch.from_dlpack / cupy.from_dlpack / jax
//             can adopt it zero-copy and nothing here has to import torch.
//   import -- import_output() borrows a caller-provided CUDA buffer through the
//             same protocol, replacing the raw integer pointer this path used
//             to take. The integer was unverifiable: a CPU tensor, a float64
//             tensor, a strided view or a mismatched grid shape all produced an
//             out-of-bounds device write instead of an exception.
//
// Both the DLPack v1 (DLManagedTensorVersioned, "dltensor_versioned") and the
// legacy v0 (DLManagedTensor, "dltensor") capsules are handled; which one a
// producer hands back depends on whether it understands max_version.

#ifndef NEARL_DLPACK_INTEROP_INCLUDED
#define NEARL_DLPACK_INTEROP_INCLUDED

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cuda_runtime.h"
#include "pybind11/pybind11.h"

#include "dlpack.h"
#include "gpuutils.cuh" // CUDA_CHECK

namespace nearl_dlpack {

namespace py = pybind11;

inline constexpr const char *CAPSULE = "dltensor";
inline constexpr const char *CAPSULE_USED = "used_dltensor";
inline constexpr const char *CAPSULE_V = "dltensor_versioned";
inline constexpr const char *CAPSULE_USED_V = "used_dltensor_versioned";

inline int current_device() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

// ---------------------------------------------------------------- export ----

// Device memory whose lifetime is independent of the DeviceContext. An exported
// grid must survive the next command and finalize_context(), so it cannot come
// from the context's reusable slots.
class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(&data_, std::max<size_t>(count, 1) * sizeof(float)));
  }
  ~DeviceBuffer() {
    if (data_)
      cudaFree(data_); // a destructor may not throw; nothing to recover anyway
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  float *data() const { return data_; }
  size_t count() const { return count_; }

private:
  float *data_ = nullptr;
  size_t count_ = 0;
};

// Kept alive by the exported capsule: the DLTensor points into shape_/strides_,
// and owner_ keeps the memory alive after the Python DeviceArray is collected.
struct ExportContext {
  std::shared_ptr<DeviceBuffer> owner;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

template <typename Managed> void export_deleter(Managed *self) {
  delete static_cast<ExportContext *>(self->manager_ctx);
  delete self;
}

// Runs when a capsule is collected without ever being consumed. A consumer that
// adopts the tensor renames the capsule to "used_dltensor*" and takes over the
// deleter, which is exactly what these name checks test for.
inline void capsule_destructor(PyObject *capsule) {
  if (PyCapsule_IsValid(capsule, CAPSULE)) {
    auto *managed = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, CAPSULE));
    if (managed && managed->deleter)
      managed->deleter(managed);
  } else if (PyCapsule_IsValid(capsule, CAPSULE_V)) {
    auto *managed =
        static_cast<DLManagedTensorVersioned *>(PyCapsule_GetPointer(capsule, CAPSULE_V));
    if (managed && managed->deleter)
      managed->deleter(managed);
  }
  PyErr_Clear();
}

inline void fill_tensor(DLTensor &tensor, ExportContext *ctx, int device) {
  tensor.data = ctx->owner->data();
  tensor.device = DLDevice{kDLCUDA, device};
  tensor.ndim = static_cast<int32_t>(ctx->shape.size());
  tensor.dtype = DLDataType{kDLFloat, 32, 1};
  tensor.shape = ctx->shape.data();
  tensor.strides = ctx->strides.data();
  tensor.byte_offset = 0;
}

/**
 * @brief A float32 CUDA grid exported through the DLPack protocol.
 *
 * Consume it with torch.from_dlpack(), cupy.from_dlpack() or any other
 * DLPack-aware framework; the memory is not copied.
 */
class DeviceArray {
public:
  DeviceArray(std::vector<int64_t> shape, int device) : shape_(std::move(shape)), device_(device) {
    size_t count = 1;
    for (int64_t extent : shape_) {
      if (extent < 0)
        throw py::value_error("Grid dimensions must be non-negative");
      count *= static_cast<size_t>(extent);
    }
    count_ = count;
    buffer_ = std::make_shared<DeviceBuffer>(count);
  }

  float *data() const { return buffer_->data(); }
  size_t count() const { return count_; }
  int device() const { return device_; }
  const std::vector<int64_t> &shape() const { return shape_; }

  py::tuple shape_tuple() const {
    py::tuple out(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i)
      out[i] = shape_[i];
    return out;
  }

  py::tuple dlpack_device() const { return py::make_tuple(static_cast<int>(kDLCUDA), device_); }

  /**
   * @brief Export as a DLPack capsule. Callable repeatedly; each capsule shares
   *        the same device memory and keeps it alive independently.
   *
   * @p stream is accepted and needs no action: every command that produces a
   * DeviceArray blocks on its own stream before returning, so the contents are
   * already complete on any stream the consumer might use.
   */
  py::capsule dlpack(const py::object &stream, const py::object &max_version,
                     const py::object &dl_device, const py::object &copy) const {
    if (!stream.is_none() && !py::isinstance<py::int_>(stream))
      throw py::type_error("__dlpack__: stream must be None or an integer");
    if (!copy.is_none() && copy.cast<bool>())
      throw py::buffer_error("__dlpack__: copy=True is not supported");
    if (!dl_device.is_none()) {
      auto requested = dl_device.cast<std::pair<int, int>>();
      if (requested.first != static_cast<int>(kDLCUDA) || requested.second != device_)
        throw py::buffer_error("__dlpack__: cannot move the grid to the requested device; "
                               "it lives on CUDA device " +
                               std::to_string(device_));
    }

    bool versioned = false;
    if (!max_version.is_none()) {
      auto requested = max_version.cast<std::pair<int, int>>();
      versioned = requested.first >= 1;
    }

    auto ctx = std::make_unique<ExportContext>();
    ctx->owner = buffer_;
    ctx->shape = shape_;
    ctx->strides = contiguous_strides(shape_);

    if (versioned) {
      auto *managed = new DLManagedTensorVersioned();
      managed->version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
      managed->manager_ctx = ctx.get();
      managed->deleter = &export_deleter<DLManagedTensorVersioned>;
      managed->flags = 0;
      fill_tensor(managed->dl_tensor, ctx.get(), device_);
      ctx.release();
      return py::capsule(managed, CAPSULE_V, &capsule_destructor);
    }
    auto *managed = new DLManagedTensor();
    managed->manager_ctx = ctx.get();
    managed->deleter = &export_deleter<DLManagedTensor>;
    fill_tensor(managed->dl_tensor, ctx.get(), device_);
    ctx.release();
    return py::capsule(managed, CAPSULE, &capsule_destructor);
  }

  static std::vector<int64_t> contiguous_strides(const std::vector<int64_t> &shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i-- > 1;)
      strides[i - 1] = strides[i] * shape[i];
    return strides;
  }

private:
  std::shared_ptr<DeviceBuffer> buffer_;
  std::vector<int64_t> shape_;
  size_t count_ = 0;
  int device_ = 0;
};

// ---------------------------------------------------------------- import ----

/**
 * @brief A borrowed CUDA buffer obtained from a consumer's __dlpack__.
 *
 * Destruction runs the producer's deleter, which for a torch tensor decrefs a
 * Python object -- so an ImportedTensor must be destroyed with the GIL held.
 * Declare it before any py::gil_scoped_release in the same scope: locals are
 * destroyed in reverse order, so the GIL is reacquired first.
 */
class ImportedTensor {
public:
  ImportedTensor() = default;
  ImportedTensor(const ImportedTensor &) = delete;
  ImportedTensor &operator=(const ImportedTensor &) = delete;

  ImportedTensor(ImportedTensor &&other) noexcept { steal(other); }
  ImportedTensor &operator=(ImportedTensor &&other) noexcept {
    if (this != &other) {
      release();
      steal(other);
    }
    return *this;
  }
  ~ImportedTensor() { release(); }

  float *data() const { return data_; }

  void adopt(DLManagedTensor *managed) {
    release();
    legacy_ = managed;
  }
  void adopt(DLManagedTensorVersioned *managed) {
    release();
    versioned_ = managed;
  }
  void set_data(float *data) { data_ = data; }

private:
  void steal(ImportedTensor &other) {
    legacy_ = other.legacy_;
    versioned_ = other.versioned_;
    data_ = other.data_;
    other.legacy_ = nullptr;
    other.versioned_ = nullptr;
    other.data_ = nullptr;
  }
  void release() {
    if (legacy_ && legacy_->deleter)
      legacy_->deleter(legacy_);
    if (versioned_ && versioned_->deleter)
      versioned_->deleter(versioned_);
    legacy_ = nullptr;
    versioned_ = nullptr;
    data_ = nullptr;
  }

  DLManagedTensor *legacy_ = nullptr;
  DLManagedTensorVersioned *versioned_ = nullptr;
  float *data_ = nullptr;
};

inline std::string describe_dtype(const DLDataType &dtype) {
  std::ostringstream text;
  text << "code=" << static_cast<int>(dtype.code) << " bits=" << static_cast<int>(dtype.bits)
       << " lanes=" << dtype.lanes;
  return text.str();
}

// A size-1 axis can carry any stride, so it never constrains contiguity.
inline bool is_c_contiguous(const DLTensor &tensor) {
  if (tensor.strides == nullptr)
    return true; // NULL strides mean contiguous in DLPack < 1.2
  int64_t expected = 1;
  for (int32_t axis = tensor.ndim; axis-- > 0;) {
    if (tensor.shape[axis] != 1 && tensor.strides[axis] != expected)
      return false;
    expected *= tensor.shape[axis];
  }
  return true;
}

inline void validate(const DLTensor &tensor, const int *dims, int device) {
  if (tensor.device.device_type != kDLCUDA)
    throw py::value_error("out= must be CUDA device memory (DLPack device_type " +
                          std::to_string(static_cast<int>(kDLCUDA)) + "); got device_type " +
                          std::to_string(static_cast<int>(tensor.device.device_type)));
  if (tensor.device.device_id != device)
    throw py::value_error("out= is on CUDA device " + std::to_string(tensor.device.device_id) +
                          " but this process computes on device " + std::to_string(device));
  if (tensor.dtype.code != kDLFloat || tensor.dtype.bits != 32 || tensor.dtype.lanes != 1)
    throw py::value_error("out= must be float32; got " + describe_dtype(tensor.dtype));
  if (tensor.data == nullptr)
    throw py::value_error("out= has a null data pointer");
  if (!is_c_contiguous(tensor))
    throw py::value_error("out= must be C-contiguous; pass a contiguous tensor "
                          "or .contiguous() a strided view");

  const int64_t expected =
      static_cast<int64_t>(dims[0]) * static_cast<int64_t>(dims[1]) * static_cast<int64_t>(dims[2]);
  int64_t count = 1;
  for (int32_t axis = 0; axis < tensor.ndim; ++axis)
    count *= tensor.shape[axis];
  if (count != expected) {
    std::ostringstream text;
    text << "out= holds " << count << " elements but the grid needs " << expected << " (" << dims[0]
         << ", " << dims[1] << ", " << dims[2] << ")";
    throw py::value_error(text.str());
  }
  if (tensor.ndim == 3) {
    for (int32_t axis = 0; axis < 3; ++axis) {
      if (tensor.shape[axis] != dims[axis]) {
        std::ostringstream text;
        text << "out= has shape (" << tensor.shape[0] << ", " << tensor.shape[1] << ", "
             << tensor.shape[2] << ") but the grid is (" << dims[0] << ", " << dims[1] << ", "
             << dims[2] << ")";
        throw py::value_error(text.str());
      }
    }
  }
}

// The array API spells a CUDA stream as an integer: 1 is the legacy default
// stream, 2 the per-thread default, anything else a cudaStream_t value.
inline py::int_ stream_token(cudaStream_t stream) {
  if (stream == nullptr)
    return py::int_(1);
  return py::int_(reinterpret_cast<std::uintptr_t>(stream));
}

inline py::object call_dlpack(const py::object &exporter, cudaStream_t stream) {
  py::object token = stream_token(stream);
  try {
    return exporter(py::arg("stream") = token, py::arg("max_version") = py::make_tuple(1, 0));
  } catch (const py::error_already_set &err) {
    if (!err.matches(PyExc_TypeError))
      throw;
  }
  // Producers predating DLPack 1.0 reject max_version, older ones reject stream.
  try {
    return exporter(py::arg("stream") = token);
  } catch (const py::error_already_set &err) {
    if (!err.matches(PyExc_TypeError))
      throw;
  }
  return exporter();
}

/**
 * @brief Borrow @p obj 's CUDA buffer as the destination for a grid of @p dims.
 *
 * @p stream is handed to the producer so it can order its own pending work on
 * that buffer ahead of ours.
 */
inline ImportedTensor import_output(const py::object &obj, cudaStream_t stream, const int *dims,
                                    int device) {
  if (!py::hasattr(obj, "__dlpack__"))
    throw py::type_error("out= must support the DLPack protocol (a torch.Tensor, cupy.ndarray "
                         "or similar); got " +
                         std::string(py::str(py::type::handle_of(obj).attr("__name__"))));

  // Ask where it lives before exporting it. Handing a CUDA stream to a CPU
  // producer makes it raise its own confusing error, and the device is the
  // first thing that has to be right anyway.
  if (py::hasattr(obj, "__dlpack_device__")) {
    auto where = obj.attr("__dlpack_device__")().cast<std::pair<int, int>>();
    if (where.first != static_cast<int>(kDLCUDA))
      throw py::value_error("out= must be CUDA device memory (DLPack device_type " +
                            std::to_string(static_cast<int>(kDLCUDA)) + "); got device_type " +
                            std::to_string(where.first));
    if (where.second != device)
      throw py::value_error("out= is on CUDA device " + std::to_string(where.second) +
                            " but this process computes on device " + std::to_string(device));
  }

  py::object capsule = call_dlpack(obj.attr("__dlpack__"), stream);
  PyObject *raw = capsule.ptr();

  ImportedTensor imported;
  const DLTensor *tensor = nullptr;
  if (PyCapsule_IsValid(raw, CAPSULE_V)) {
    auto *managed = static_cast<DLManagedTensorVersioned *>(PyCapsule_GetPointer(raw, CAPSULE_V));
    if (managed->version.major > DLPACK_MAJOR_VERSION)
      throw py::buffer_error("out= uses DLPack " + std::to_string(managed->version.major) +
                             ".x, which this build does not understand (built against " +
                             std::to_string(DLPACK_MAJOR_VERSION) + "." +
                             std::to_string(DLPACK_MINOR_VERSION) + ")");
    if (managed->flags & DLPACK_FLAG_BITMASK_READ_ONLY)
      throw py::buffer_error("out= is read-only");
    PyCapsule_SetName(raw, CAPSULE_USED_V);
    imported.adopt(managed);
    tensor = &managed->dl_tensor;
  } else if (PyCapsule_IsValid(raw, CAPSULE)) {
    auto *managed = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(raw, CAPSULE));
    PyCapsule_SetName(raw, CAPSULE_USED);
    imported.adopt(managed);
    tensor = &managed->dl_tensor;
  } else {
    throw py::value_error("out=.__dlpack__() did not return a fresh DLPack capsule "
                          "(it may already have been consumed)");
  }

  validate(*tensor, dims, device);
  imported.set_data(
      reinterpret_cast<float *>(static_cast<char *>(tensor->data) + tensor->byte_offset));
  return imported;
}

inline void register_device_array(py::module_ &m) {
  py::class_<DeviceArray>(m, "DeviceArray", R"doc(
A float32 CUDA grid exported through the DLPack protocol.

Adopt it zero-copy with ``torch.from_dlpack(grid)``, ``cupy.from_dlpack(grid)``
or any other DLPack-aware framework. The memory is owned here and released when
both this object and every tensor adopted from it are gone.
)doc")
      .def("__dlpack__", &DeviceArray::dlpack, py::kw_only(), py::arg("stream") = py::none(),
           py::arg("max_version") = py::none(), py::arg("dl_device") = py::none(),
           py::arg("copy") = py::none(), "Export the grid as a DLPack capsule.")
      .def("__dlpack_device__", &DeviceArray::dlpack_device,
           "Return the (device_type, device_id) pair describing where the grid lives.")
      .def_property_readonly("shape", &DeviceArray::shape_tuple)
      .def_property_readonly("device_id", &DeviceArray::device)
      .def_property_readonly("size", &DeviceArray::count)
      .def("__repr__", [](const DeviceArray &self) {
        std::ostringstream text;
        text << "<nearl DeviceArray float32(";
        for (size_t i = 0; i < self.shape().size(); ++i)
          text << (i ? ", " : "") << self.shape()[i];
        text << ") on cuda:" << self.device() << ">";
        return text.str();
      });
}

} // namespace nearl_dlpack

#endif // NEARL_DLPACK_INTEROP_INCLUDED
