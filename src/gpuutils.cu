// Created by Yang Zhang
// Description: Utility functions for GPU computing

#include "cuda_runtime.h"
#include "constants.h"
#include "gpuutils.cuh"

#include <algorithm>
#include <mutex>


DeviceContext::DeviceContext() = default;

DeviceContext::~DeviceContext() {
  if (initialized_) {
    finalize();
  }
}

void DeviceContext::init() {
  if (initialized_) {
    return;
  }
  CUDA_CHECK(cudaStreamCreate(&stream_));
  initialized_ = true;
}

void DeviceContext::finalize() {
  if (!initialized_) {
    return;
  }
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
  initialized_ = false;
}

void DeviceContext::synchronize() {
  if (initialized_ && stream_) {
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
  Buffer &buf = buffers_[slot];
  if (buf.capacity < min_bytes) {
    if (buf.ptr) {
      CUDA_CHECK(cudaFree(buf.ptr));
    }
    // Round up to the next power of two to avoid many small reallocations.
    size_t new_capacity = min_bytes ? 1 : 0;
    while (new_capacity < min_bytes) {
      new_capacity <<= 1;
    }
    CUDA_CHECK(cudaMalloc(&buf.ptr, new_capacity));
    buf.capacity = new_capacity;
  }
  return buf.ptr;
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


////////////////////////////////////////////////////////////////////////////////
// Compile-time dispatch of the aggregations
////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Maps an AggregationType to the __device__ function implementing it.
 *
 * The specializations are generated from AGGREGATION_TYPE_LIST, so the kernel below stays free of
 * any per-aggregation branching.
 */
template <AggregationType Aggregation> struct aggregation_kernel;

#define AGGREGATION_KERNEL_SPECIALIZATION(NAME, VALUE, FN)                                         \
  template <> struct aggregation_kernel<AggregationType::NAME> {                                   \
    __device__ static float apply(float *arr, const int N) { return FN(arr, N); }                  \
  };
AGGREGATION_TYPE_LIST(AGGREGATION_KERNEL_SPECIALIZATION)
#undef AGGREGATION_KERNEL_SPECIALIZATION


template <AggregationType Aggregation>
__global__ void gridwise_aggregation_global(float *d_in, float *d_out, const int frame_nr,
                                            const int gridpoint_nr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= gridpoint_nr)
    return;

  float tmp_array[MAX_FRAME_NUMBER];
  for (int i = 0; i < frame_nr; i++) {
    tmp_array[i] = d_in[i * gridpoint_nr + idx];
  }

  d_out[idx] = aggregation_kernel<Aggregation>::apply(tmp_array, frame_nr);
}


/**
 * @brief Launch the aggregation kernel instantiated for the requested aggregation.
 *
 * This is the only place where the runtime aggregation type is turned into a template argument,
 * which keeps the kernel itself branch-free.
 */
void launch_gridwise_aggregation(const AggregationType type_agg, const unsigned int grid_size,
                                 float *d_in, float *d_out, const int frame_nr,
                                 const int gridpoint_nr, cudaStream_t stream) {
  switch (type_agg) {
#define AGGREGATION_LAUNCH_CASE(NAME, VALUE, FN)                                                   \
  case AggregationType::NAME:                                                                      \
    gridwise_aggregation_global<AggregationType::NAME>                                             \
        <<<grid_size, BLOCK_SIZE, 0, stream>>>(d_in, d_out, frame_nr, gridpoint_nr);               \
    break;
    AGGREGATION_TYPE_LIST(AGGREGATION_LAUNCH_CASE)
#undef AGGREGATION_LAUNCH_CASE
  default:
    throw std::invalid_argument("The aggregation type " +
                                std::to_string(static_cast<int>(type_agg)) + " is not supported");
  }
  CUDA_CHECK_KERNEL();
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
                    const int grid_number, const AggregationType type_agg) {
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

  CUDA_CHECK(cudaMemcpyAsync(voxel_traj_gpu, voxel_traj, frame_number * grid_number * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(tmp_grid_gpu, 0, grid_number * sizeof(float), stream));

  launch_gridwise_aggregation(type_agg, grid_size, voxel_traj_gpu, tmp_grid_gpu, _frame_number,
                              grid_number, stream);
  CUDA_CHECK(cudaMemcpyAsync(result_grid, tmp_grid_gpu, grid_number * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  if (use_ctx) {
    ctx->synchronize();
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(voxel_traj_gpu));
    CUDA_CHECK(cudaFree(tmp_grid_gpu));
  }
}

/**
 * @brief Sum the elements of a host float array on the GPU.
 *
 * Performs a parallel reduction on the GPU and returns the scalar sum to the
 * host. Uses the global DeviceContext for allocations/streaming when active,
 * otherwise falls back to per-call cudaMalloc/cudaFree.
 */
float sum_reduction_host(float *array, const int arr_length) {
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

  CUDA_CHECK(cudaMemcpyAsync(array_gpu, array, arr_length * sizeof(float), cudaMemcpyHostToDevice,
                             stream));

  // Perform the sum reduction on the array
  sum_reduction_global<<<grid_size, BLOCK_SIZE, 0, stream>>>(array_gpu, partial_sums, arr_length);
  CUDA_CHECK_KERNEL();
  if (use_ctx) {
    ctx->synchronize();
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Compute the final sum
  float _partial_sums[grid_size];
  float tmp_sum = 0.0f;
  CUDA_CHECK(cudaMemcpyAsync(_partial_sums, partial_sums, grid_size * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  if (use_ctx) {
    ctx->synchronize();
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  for (int i = 0; i < grid_size; ++i)
    tmp_sum += _partial_sums[i];

  if (!use_ctx) {
    CUDA_CHECK(cudaFree(partial_sums));
    CUDA_CHECK(cudaFree(array_gpu));
  }

  return tmp_sum;
}
