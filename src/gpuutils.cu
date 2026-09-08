// Created by Yang Zhang
// Description: Utility functions for GPU computing

#include "cuda_runtime.h"
#include "constants.h"
#include "gpuutils.cuh"

#include <algorithm>
#include <mutex>
#include <cstring>
#include <limits>


DeviceContext::DeviceContext() = default;

DeviceContext::~DeviceContext() noexcept {
  cancel_call();
  try {
    finalize();
  } catch (...) {
  } // Destruction must not throw during error cleanup.
}

void DeviceContext::init() {
  if (initialized_) {
    return;
  }
  CUDA_CHECK(cudaGetDevice(&device_));
  CUDA_CHECK(cudaStreamCreate(&stream_));
  initialized_ = true;
}

void DeviceContext::finalize() {
  if (pending_)
    throw std::runtime_error("Collect the pending CUDA result before finalizing the context");
  if (!initialized_) {
    return;
  }
  CUDA_CHECK(cudaSetDevice(device_));
  if (stream_) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaStreamDestroy(stream_));
    stream_ = 0;
  }
  for (size_t i = 0; i < NUM_SLOTS; ++i) {
    if (buffers_[i].ptr) {
      CUDA_CHECK(cudaFree(buffers_[i].ptr));
      buffers_[i].ptr = nullptr;
      buffers_[i].capacity = 0;
    }
  }
  for (auto &buffer : host_buffers_) {
    if (buffer.ptr)
      CUDA_CHECK(cudaFreeHost(buffer.ptr));
    buffer = Buffer{};
  }
  initialized_ = false;
}

void DeviceContext::synchronize() {
  if (initialized_ && stream_) {
    CUDA_CHECK(cudaSetDevice(device_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void *DeviceContext::get_buffer(size_t min_bytes, size_t slot) {
  if (!initialized_) {
    throw std::runtime_error("DeviceContext::get_buffer called before init()");
  }
  if (slot >= NUM_SLOTS) {
    throw std::runtime_error("DeviceContext: buffer slot out of range");
  }
  return resize_buffer(buffers_[slot], min_bytes, false);
}

size_t DeviceContext::buffer_capacity(size_t slot) const {
  if (slot >= NUM_SLOTS)
    throw std::runtime_error("DeviceContext: buffer slot out of range");
  return buffers_[slot].capacity;
}

void *DeviceContext::resize_buffer(Buffer &buffer, size_t bytes, bool pinned) {
  bytes = std::max(bytes, size_t(1));
  if (buffer.capacity < bytes) {
    if (buffer.ptr) {
      if (pinned)
        CUDA_CHECK(cudaFreeHost(buffer.ptr));
      else
        CUDA_CHECK(cudaFree(buffer.ptr));
      buffer = Buffer{};
    }
    size_t capacity = 1;
    while (capacity < bytes && capacity <= std::numeric_limits<size_t>::max() / 2)
      capacity *= 2;
    capacity = std::max(capacity, bytes);
    if (pinned)
      CUDA_CHECK(cudaMallocHost(&buffer.ptr, capacity));
    else
      CUDA_CHECK(cudaMalloc(&buffer.ptr, capacity));
    buffer.capacity = capacity;
  }
  return buffer.ptr;
}

void *DeviceContext::stage_input(const void *source, size_t bytes, size_t slot) {
  if (!initialized_ || slot >= NUM_SLOTS)
    throw std::runtime_error("Invalid pinned input buffer request");
  void *destination = resize_buffer(host_buffers_[slot], bytes, true);
  if (bytes)
    std::memcpy(destination, source, bytes);
  return destination;
}

void *DeviceContext::get_host_buffer(size_t min_bytes, size_t slot) {
  if (!initialized_ || slot >= NUM_SLOTS)
    throw std::runtime_error("Invalid pinned output buffer request");
  return resize_buffer(host_buffers_[slot], min_bytes, true);
}

size_t DeviceContext::host_buffer_capacity(size_t slot) const {
  if (slot >= NUM_SLOTS)
    throw std::runtime_error("DeviceContext: buffer slot out of range");
  return host_buffers_[slot].capacity;
}

void DeviceContext::begin_call() {
  if (!initialized_)
    throw std::runtime_error("Initialize the CUDA context before dispatch");
  if (pending_)
    throw std::runtime_error("Collect the previous CUDA result before dispatch");
  CUDA_CHECK(cudaSetDevice(device_));
  pending_ = true;
}

void DeviceContext::end_call() {
  synchronize();
  pending_ = false;
}

void DeviceContext::cancel_call() noexcept {
  if (!pending_)
    return;
  if (cudaSetDevice(device_) == cudaSuccess)
    cudaStreamSynchronize(stream_);
  pending_ = false;
}

void copy_h2d_async(DeviceContext *ctx, void *destination, const void *source, size_t bytes,
                    BufferSlot slot, cudaStream_t stream) {
  if (ctx && ctx->valid())
    source = ctx->stage_input(source, bytes, static_cast<size_t>(slot));
  CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
}

namespace {
std::mutex g_context_mutex;
DeviceContext *g_context = nullptr;
} // namespace

DeviceContext *get_global_device_context() { return g_context; }

void init_global_device_context() {
  std::lock_guard<std::mutex> lock(g_context_mutex);
  if (!g_context) {
    g_context = new DeviceContext();
  }
  g_context->init();
}

void finalize_global_device_context() {
  std::lock_guard<std::mutex> lock(g_context_mutex);
  if (g_context) {
    g_context->finalize();
  }
}

bool global_device_context_valid() { return g_context && g_context->valid(); }


__global__ void sum_reduction_global(const float *d_in, float *d_out, const int N) {
  __shared__ float smem[BLOCK_SIZE];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  // Load the input data into shared memory
  smem[tid] = (idx < N) ? d_in[idx] : 0;
  __syncthreads();

  // Perform reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  // Write the result to global memory
  if (tid == 0) {
    d_out[blockIdx.x] = smem[0];
  }
}


__global__ void normalize_array_global(float *d_in, const float sum, const float weight,
                                       const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;
  d_in[idx] = d_in[idx] * weight / sum;
}


__global__ void voxel_addition_global(float *d_parent, float *d_add, const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;
  d_parent[idx] += d_add[idx];
}


__global__ void gridwise_aggregation_global(float *d_in, float *d_out, const int frame_nr,
                                            const int gridpoint_nr, const int type_agg) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= gridpoint_nr)
    return;

  float tmp_array[MAX_FRAME_NUMBER];
  for (int i = 0; i < frame_nr; i++) {
    tmp_array[i] = d_in[i * gridpoint_nr + idx];
  }

  if (type_agg == 1) {
    d_out[idx] = mean_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 2) {
    d_out[idx] = standard_deviation_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 3) {
    d_out[idx] = median_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 4) {
    d_out[idx] = variance_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 5) {
    d_out[idx] = max_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 6) {
    d_out[idx] = min_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 7) {
    d_out[idx] = information_entropy_histogram_device(tmp_array, frame_nr);
  } else if (type_agg == 8) {
    d_out[idx] = slope_device<float>(tmp_array, frame_nr);
  } else {
    // Should throw exception in the python-end
    d_out[idx] = 0;
  }
}


/**
 * @brief Aggregate a per-frame grid trajectory into a single grid.
 *
 * Takes a host array of shape (frame_number, grid_number), uploads it to the
 * GPU, and runs gridwise_aggregation_global to reduce the frame dimension
 * using the requested aggregation (mean, std-dev, median, ...). The result is
 * copied back to result_grid.
 */
void aggregate_host(float *voxel_traj, float *result_grid, const int frame_number,
                    const int grid_number, const int type_agg) {
  unsigned int grid_size = (grid_number + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int _frame_number = frame_number > MAX_FRAME_NUMBER ? MAX_FRAME_NUMBER : frame_number;

  DeviceContext *ctx = get_global_device_context();
  const bool use_ctx = ctx && ctx->valid();
  cudaStream_t stream = use_ctx ? ctx->stream() : 0;

  // Move the voxelized stuff
  float *voxel_traj_gpu;
  float *tmp_grid_gpu;
  if (use_ctx) {
    voxel_traj_gpu = ctx->get_buffer_f(static_cast<size_t>(frame_number) * grid_number,
                                       static_cast<size_t>(BufferSlot::TRAJ_DYNAMICS));
    tmp_grid_gpu = ctx->get_buffer_f(grid_number, static_cast<size_t>(BufferSlot::OUTPUT_GRID));
  } else {
    CUDA_CHECK(cudaMalloc(&voxel_traj_gpu, frame_number * grid_number * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp_grid_gpu, grid_number * sizeof(float)));
  }

  copy_h2d_async(ctx, voxel_traj_gpu, voxel_traj, frame_number * grid_number * sizeof(float),
                 BufferSlot::TRAJ_DYNAMICS, stream);
  CUDA_CHECK(cudaMemsetAsync(tmp_grid_gpu, 0, grid_number * sizeof(float), stream));

  gridwise_aggregation_global<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      voxel_traj_gpu, tmp_grid_gpu, _frame_number, grid_number, type_agg);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpyAsync(result_grid, tmp_grid_gpu, grid_number * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  if (use_ctx) {
    if (!ctx->pending())
      ctx->synchronize();
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(voxel_traj_gpu));
    CUDA_CHECK(cudaFree(tmp_grid_gpu));
  }
}

/**
 * @brief Queue a GPU reduction and copy its block sums to host storage.
 *
 * With a pending context result, partial_host must stay alive until collection.
 * The binding combines these partial sums after synchronization. Raw callers
 * without a pending result retain synchronous behavior.
 */
void sum_reduction_dispatch(float *array, const int arr_length, float *partial_host) {
  unsigned int grid_size = (arr_length + BLOCK_SIZE - 1) / BLOCK_SIZE;

  DeviceContext *ctx = get_global_device_context();
  const bool use_ctx = ctx && ctx->valid();
  cudaStream_t stream = use_ctx ? ctx->stream() : 0;

  float *partial_sums;
  float *array_gpu;
  if (use_ctx) {
    partial_sums = ctx->get_buffer_f(grid_size, static_cast<size_t>(BufferSlot::PARTIAL_SUMS));
    array_gpu = ctx->get_buffer_f(arr_length, static_cast<size_t>(BufferSlot::COORDS));
  } else {
    CUDA_CHECK(cudaMalloc(&partial_sums, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&array_gpu, arr_length * sizeof(float)));
  }

  copy_h2d_async(ctx, array_gpu, array, arr_length * sizeof(float), BufferSlot::COORDS, stream);

  // Perform the sum reduction on the array
  sum_reduction_global<<<grid_size, BLOCK_SIZE, 0, stream>>>(array_gpu, partial_sums, arr_length);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpyAsync(partial_host, partial_sums, grid_size * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  if (use_ctx) {
    if (!ctx->pending())
      ctx->synchronize();
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  if (!use_ctx) {
    CUDA_CHECK(cudaFree(partial_sums));
    CUDA_CHECK(cudaFree(array_gpu));
  }
}

float sum_reduction_host(float *array, const int arr_length) {
  std::vector<float> partials((arr_length + BLOCK_SIZE - 1) / BLOCK_SIZE);
  sum_reduction_dispatch(array, arr_length, partials.data());
  auto *ctx = get_global_device_context();
  if (ctx && ctx->valid())
    ctx->synchronize();
  float sum = 0;
  for (float value : partials)
    sum += value;
  return sum;
}
