// Created by: Yang Zhang
// Description: The CUDA implementation of the property density flow algorithm

#include <iostream>

#include "constants.h"
#include "cpuutils.h"   // For gaussian_map
#include "gpuutils.cuh" // For CUDA kernels
#include "voxelize.cuh"
#include "voxelize_math.h"

/**
 * @brief Per-atom Gaussian density on the full grid (legacy one-atom kernel).
 *
 * Each thread evaluates the Gaussian contribution of a single atom (coord) to
 * one grid point. The result is written to interpolated[task_index]. This kernel
 * is currently only used by the old _voxelize_host_old path.
 */
__global__ void coordi_interp_global(const float *coord, float *interpolated, const int *dims,
                                     const float spacing, const float cutoff, const float sigma) {
  unsigned int task_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int grid_size = dims[0] * dims[1] * dims[2];
  if (task_index >= grid_size)
    return;

  // Compute the grid coordinate from the grid index
  float grid_coord[3] = {static_cast<float>(task_index / (dims[0] * dims[1])) * spacing,
                         static_cast<float>((task_index / dims[0]) % dims[1]) * spacing,
                         static_cast<float>(task_index % dims[0]) * spacing};
  float dist_square = 0.0f;
  for (int i = 0; i < 3; ++i) {
    dist_square += (coord[i] - grid_coord[i]) * (coord[i] - grid_coord[i]);
  }
  // Process the interpolated array with the task_index (Should not be race conditions)
  if (dist_square < cutoff * cutoff) {
    interpolated[task_index] = gaussian_map_device(sqrt(dist_square), 0.0f, sigma);
  } else {
    // Set to 0 to avoid reuse of the previous value
    interpolated[task_index] = 0.0f;
  }
}


/**
 * @brief Per-frame Gaussian density voxelization using one CUDA block per atom.
 *
 * Each block owns one atom of one frame (blockIdx.x selects the atom,
 * blockIdx.y the frame). Threads first sum the unweighted Gaussian density
 * over a cutoff-bounded sub-grid (shared-memory reduction), then scatter the
 * atom's weighted, normalized contribution into the full output grid via
 * atomicAdd. The normalization guarantees that the integral over the grid for
 * each atom equals the atom's weight.
 *
 * Both passes visit only the axis-aligned integer bounding box of the atom's
 * cutoff ball rather than striding the whole grid, and keep the `dist_sq <
 * cutoff_sq` test inside, so the set of contributing points is unchanged.
 *
 * The two passes deliberately run over *different* extents. The normalizer
 * covers the buffered grid, which reaches buff_dim points outside the output
 * grid on every face; the scatter covers only [0, dims). An atom near a face is
 * therefore normalized over its whole ball but deposits only the in-grid part of
 * its weight, which is why sum(grid) == sum(weights) holds for interior atoms
 * and not for edge ones. Clip the two boxes separately to preserve that.
 */
__global__ void frame_interp_global(const float *coords_frame, const float *weights_frame,
                                    float *interpolated_frame, const int *dims, const float spacing,
                                    const float cutoff, const float sigma, const int atom_nr) {
  // Each block is responsible for one atom
  const int atom_idx = blockIdx.x;
  const int frame_idx = blockIdx.y;
  const int buff_dim = (cutoff + spacing) / spacing;
  const int buff_dims[3] = {dims[0] + buff_dim + buff_dim, dims[1] + buff_dim + buff_dim,
                            dims[2] + buff_dim + buff_dim};
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const float *coord = coords_frame + (frame_idx * atom_nr + atom_idx) * 3;
  const float cutoff_sq = cutoff * cutoff;
  const float weight = weights_frame[frame_idx * atom_nr + atom_idx];
  float *frame_output = interpolated_frame + frame_idx * gridpoint_nr;

  if (nearl_is_placeholder(coord))
    return;
  if (weight == 0.0f)
    return;

  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  // Index range of the atom's cutoff ball along each axis, before clipping.
  // A grid point at integer index i sits at i * spacing, so the ball spans
  // [(c - cutoff) / spacing, (c + cutoff) / spacing].
  int ball_lo[3], ball_hi[3];
  for (int d = 0; d < 3; ++d)
    nearl_ball_bounds(coord[d], cutoff, spacing, &ball_lo[d], &ball_hi[d]);

  float local_sum = 0.0f;
  int x, y, z;
  float dist_sq;

  // Normalizer pass, over the buffered grid: axis i runs over
  // [-buff_dim, buff_dims[i] - buff_dim). NOTE the axis-to-extent pairing below
  // (coord[0] with buff_dims[2], coord[2] with buff_dims[0]) mirrors the index
  // decoding this kernel has always used; it only matters for non-cubic grids,
  // which Nearl does not produce today.
  const int nlo_x = max(ball_lo[0], -buff_dim);
  const int nhi_x = min(ball_hi[0], buff_dims[2] - buff_dim - 1);
  const int nlo_y = max(ball_lo[1], -buff_dim);
  const int nhi_y = min(ball_hi[1], buff_dims[1] - buff_dim - 1);
  const int nlo_z = max(ball_lo[2], -buff_dim);
  const int nhi_z = min(ball_hi[2], buff_dims[0] - buff_dim - 1);

  const int n_nx = nhi_x - nlo_x + 1;
  const int n_ny = nhi_y - nlo_y + 1;
  const int n_nz = nhi_z - nlo_z + 1;
  const int norm_box_nr = (n_nx > 0 && n_ny > 0 && n_nz > 0) ? n_nx * n_ny * n_nz : 0;

  for (int bid = tid; bid < norm_box_nr; bid += num_threads) {
    x = nlo_x + bid / (n_ny * n_nz);
    y = nlo_y + (bid / n_nz) % n_ny;
    z = nlo_z + bid % n_nz;

    dist_sq = nearl_dist_sq(coord, x, y, z, spacing);
    if (dist_sq < cutoff_sq)
      local_sum += nearl_gaussian(sqrtf(dist_sq), sigma);
  }

  // Store partial sum to shared memory
  smem[tid] = local_sum;
  __syncthreads();

  for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  const float total_sum = smem[0];
  if (total_sum == 0)
    return;

  const float inv_sum = weight / total_sum;

  // Scatter pass, over the output grid only: the ball clipped to [0, dims).
  // gid = x * dims[0] * dims[1] + y * dims[0] + z inverts the decoding the
  // kernel used before, so a point keeps the array slot it always had.
  const int slo_x = max(ball_lo[0], 0);
  const int shi_x = min(ball_hi[0], dims[2] - 1);
  const int slo_y = max(ball_lo[1], 0);
  const int shi_y = min(ball_hi[1], dims[1] - 1);
  const int slo_z = max(ball_lo[2], 0);
  const int shi_z = min(ball_hi[2], dims[0] - 1);

  const int s_nx = shi_x - slo_x + 1;
  const int s_ny = shi_y - slo_y + 1;
  const int s_nz = shi_z - slo_z + 1;
  const int scatter_box_nr = (s_nx > 0 && s_ny > 0 && s_nz > 0) ? s_nx * s_ny * s_nz : 0;

  for (int bid = tid; bid < scatter_box_nr; bid += num_threads) {
    x = slo_x + bid / (s_ny * s_nz);
    y = slo_y + (bid / s_nz) % s_ny;
    z = slo_z + bid % s_nz;

    dist_sq = nearl_dist_sq(coord, x, y, z, spacing);
    if (dist_sq < cutoff_sq)
      atomicAdd(frame_output + nearl_grid_index(x, y, z, dims),
                nearl_gaussian(sqrtf(dist_sq), sigma) * inv_sum);
  }
}


/**
 * @brief Older GPU voxelization path kept for comparison (not used by commands).
 *
 * For each atom, runs a full-grid per-atom kernel followed by a separate
 * reduction, normalisation, and accumulation step. Slower than frame_interp_global
 * but useful as a correctness/performance baseline.
 */
void _voxelize_host_old(float *interpolated, const float *coord, const float *weight,
                        const int *dims, const float spacing, const int atom_nr, const float cutoff,
                        const float sigma) {
  unsigned int gridpoint_nr = dims[0] * dims[1] * dims[2];
  unsigned int grid_size = (gridpoint_nr + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float _partial_sums[grid_size];

  float *coord_gpu;
  CUDA_CHECK(cudaMalloc(&coord_gpu, atom_nr * 3 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(coord_gpu, coord, atom_nr * 3 * sizeof(float), cudaMemcpyHostToDevice));
  float *tmp_interp_gpu;
  CUDA_CHECK(cudaMalloc(&tmp_interp_gpu, gridpoint_nr * sizeof(float)));
  CUDA_CHECK(cudaMemset(tmp_interp_gpu, 0, gridpoint_nr * sizeof(float)));
  float *interp_gpu;
  CUDA_CHECK(cudaMalloc(&interp_gpu, gridpoint_nr * sizeof(float)));
  CUDA_CHECK(cudaMemset(interp_gpu, 0, gridpoint_nr * sizeof(float)));
  float *partial_sums;
  CUDA_CHECK(cudaMalloc(&partial_sums, grid_size * sizeof(float)));
  int *dims_gpu;
  CUDA_CHECK(cudaMalloc(&dims_gpu, 3 * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice));

  for (int atm_idx = 0; atm_idx < atom_nr; ++atm_idx) {
    // Copy the coordinates of the atom to the GPU and do interpolation on this atom
    int offset = atm_idx * 3;

    // Skip the padded coordinates (all coordinates are DEFAULT_COORD_PLACEHOLDER) and zero weights
    if (coord[offset] == DEFAULT_COORD_PLACEHOLDER &&
        coord[offset + 1] == DEFAULT_COORD_PLACEHOLDER &&
        coord[offset + 2] == DEFAULT_COORD_PLACEHOLDER) {
      continue;
    }

    if (weight[atm_idx] == 0.0f) {
      // Skip the voxelization if the weight is 0
      continue;
    }

    coordi_interp_global<<<grid_size, BLOCK_SIZE>>>(coord_gpu + offset, tmp_interp_gpu, dims_gpu,
                                                    spacing, cutoff, sigma);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform the sum reduction on the GPU and sum them up
    sum_reduction_global<<<grid_size, BLOCK_SIZE>>>(tmp_interp_gpu, partial_sums, gridpoint_nr);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    float tmp_sum = 0.0f;
    CUDA_CHECK(
        cudaMemcpy(_partial_sums, partial_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < grid_size; ++i)
      tmp_sum += _partial_sums[i];

    // Normalize the temporary array
    if (tmp_sum != 0) {
      normalize_array_global<<<grid_size, BLOCK_SIZE>>>(tmp_interp_gpu, tmp_sum, weight[atm_idx],
                                                        gridpoint_nr);
      CUDA_CHECK_KERNEL();
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Add the interpolated GPU array to the output array
    voxel_addition_global<<<grid_size, BLOCK_SIZE>>>(interp_gpu, tmp_interp_gpu, gridpoint_nr);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Copy the interpolated array to the host
  CUDA_CHECK(
      cudaMemcpy(interpolated, interp_gpu, gridpoint_nr * sizeof(float), cudaMemcpyDeviceToHost));

  // Free the GPU memory
  CUDA_CHECK(cudaFree(coord_gpu));
  CUDA_CHECK(cudaFree(tmp_interp_gpu));
  CUDA_CHECK(cudaFree(interp_gpu));
  CUDA_CHECK(cudaFree(dims_gpu));
  CUDA_CHECK(cudaFree(partial_sums));
}


/**
 * @brief GPU entry point for single-frame Gaussian density voxelization.
 *
 * Uploads coordinates, weights, and grid dimensions to the GPU, launches
 * frame_interp_global with one block per atom, and copies the resulting grid
 * back to host memory. Uses the global DeviceContext when active.
 */
void voxelize_host(float *interpolated, const float *coord, const float *weight, const int *dims,
                   const float spacing, const int atom_nr, const float cutoff, const float sigma) {
  unsigned int gridpoint_nr = dims[0] * dims[1] * dims[2];

  DeviceContext *ctx = get_global_device_context();
  const bool use_ctx = ctx && ctx->valid();
  cudaStream_t stream = use_ctx ? ctx->stream() : 0;

  float *coord_gpu;
  float *weight_gpu;
  int *dims_gpu;
  float *tmp_voxel_gpu;
  if (use_ctx) {
    coord_gpu = ctx->get_buffer_f(atom_nr * 3, static_cast<size_t>(BufferSlot::COORDS));
    weight_gpu = ctx->get_buffer_f(atom_nr, static_cast<size_t>(BufferSlot::WEIGHTS));
    dims_gpu = ctx->get_buffer_i(3, static_cast<size_t>(BufferSlot::DIMS));
    tmp_voxel_gpu = ctx->get_buffer_f(gridpoint_nr, static_cast<size_t>(BufferSlot::OUTPUT_GRID));
  } else {
    CUDA_CHECK(cudaMalloc(&coord_gpu, atom_nr * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight_gpu, atom_nr * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dims_gpu, 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tmp_voxel_gpu, gridpoint_nr * sizeof(float)));
  }

  copy_h2d_async(ctx, coord_gpu, coord, atom_nr * 3 * sizeof(float), BufferSlot::COORDS, stream);
  copy_h2d_async(ctx, weight_gpu, weight, atom_nr * sizeof(float), BufferSlot::WEIGHTS, stream);
  copy_h2d_async(ctx, dims_gpu, dims, 3 * sizeof(int), BufferSlot::DIMS, stream);
  CUDA_CHECK(cudaMemsetAsync(tmp_voxel_gpu, 0.0f, gridpoint_nr * sizeof(float), stream));

  if (atom_nr > 0) {
    frame_interp_global<<<atom_nr, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(
        coord_gpu, weight_gpu, tmp_voxel_gpu, dims_gpu, spacing, cutoff, sigma, atom_nr);
    CUDA_CHECK_KERNEL();
  }
  CUDA_CHECK(cudaMemcpyAsync(interpolated, tmp_voxel_gpu, gridpoint_nr * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));

  if (use_ctx) {
    if (!ctx->pending())
      ctx->synchronize();
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(coord_gpu));
    CUDA_CHECK(cudaFree(weight_gpu));
    CUDA_CHECK(cudaFree(dims_gpu));
    CUDA_CHECK(cudaFree(tmp_voxel_gpu));
  }
}


/**
 * @brief GPU entry point for trajectory density flow with frame aggregation.
 *
 * Voxelizes every frame of a trajectory into a (frame_nr, grid_points) buffer,
 * then reduces the frame dimension with gridwise_aggregation_global according
 * to type_agg (mean, std-dev, ...). The final aggregated grid is copied back to
 * the host. Uses the global DeviceContext when active.
 *
 * @param voxelize_dynamics Host output buffer of size grid_points.
 * @param coord Host coordinates with shape (frame_nr, atom_nr, 3).
 * @param weight Host weights with shape (frame_nr, atom_nr).
 * @param dims Grid dimensions [x, y, z].
 * @param spacing Grid spacing.
 * @param frame_nr Number of frames.
 * @param atom_nr Number of atoms per frame.
 * @param cutoff Cutoff distance for the Gaussian kernel.
 * @param sigma Width of the Gaussian kernel.
 * @param type_agg Aggregation type (see constants.h).
 */
void trajectory_voxelization_host(float *voxelize_dynamics, const float *coord, const float *weight,
                                  const int *dims, const float spacing, const int frame_nr,
                                  const int atom_nr, const float cutoff, const float sigma,
                                  const int type_agg) {
  const unsigned int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const unsigned int grid_size = (gridpoint_nr + BLOCK_SIZE - 1) / BLOCK_SIZE;

  DeviceContext *ctx = get_global_device_context();
  const bool use_ctx = ctx && ctx->valid();
  cudaStream_t stream = use_ctx ? ctx->stream() : 0;

  float *coord_gpu;
  float *weight_gpu;
  float *tmp_voxel_gpu;
  float *voxelize_dynamics_gpu;
  int *dims_gpu;

  if (use_ctx) {
    coord_gpu = ctx->get_buffer_f(static_cast<size_t>(frame_nr) * atom_nr * 3,
                                  static_cast<size_t>(BufferSlot::COORDS));
    weight_gpu = ctx->get_buffer_f(static_cast<size_t>(frame_nr) * atom_nr,
                                   static_cast<size_t>(BufferSlot::WEIGHTS));
    tmp_voxel_gpu = ctx->get_buffer_f(gridpoint_nr, static_cast<size_t>(BufferSlot::TMP_GRID));
    voxelize_dynamics_gpu = ctx->get_buffer_f(static_cast<size_t>(frame_nr) * gridpoint_nr,
                                              static_cast<size_t>(BufferSlot::TRAJ_DYNAMICS));
    dims_gpu = ctx->get_buffer_i(3, static_cast<size_t>(BufferSlot::DIMS));
  } else {
    CUDA_CHECK(cudaMalloc(&coord_gpu, frame_nr * atom_nr * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight_gpu, frame_nr * atom_nr * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp_voxel_gpu, gridpoint_nr * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&voxelize_dynamics_gpu, frame_nr * gridpoint_nr * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dims_gpu, 3 * sizeof(int)));
  }

  copy_h2d_async(ctx, coord_gpu, coord, frame_nr * atom_nr * 3 * sizeof(float), BufferSlot::COORDS,
                 stream);
  copy_h2d_async(ctx, weight_gpu, weight, frame_nr * atom_nr * sizeof(float), BufferSlot::WEIGHTS,
                 stream);
  copy_h2d_async(ctx, dims_gpu, dims, 3 * sizeof(int), BufferSlot::DIMS, stream);
  CUDA_CHECK(cudaMemsetAsync(tmp_voxel_gpu, 0.0f, gridpoint_nr * sizeof(float), stream));
  CUDA_CHECK(
      cudaMemsetAsync(voxelize_dynamics_gpu, 0, frame_nr * gridpoint_nr * sizeof(float), stream));

  // Process every frame in one launch; blockIdx.y selects the frame.
  if (atom_nr > 0)
    frame_interp_global<<<dim3(atom_nr, frame_nr, 1), BLOCK_SIZE, BLOCK_SIZE * sizeof(float),
                          stream>>>(coord_gpu, weight_gpu, voxelize_dynamics_gpu, dims_gpu, spacing,
                                    cutoff, sigma, atom_nr);
  CUDA_CHECK_KERNEL();

  // Aggregate the frames and copy the result to the host
  const int _frame_nr = frame_nr > MAX_FRAME_NUMBER ? MAX_FRAME_NUMBER : frame_nr;
  CUDA_CHECK(cudaMemsetAsync(tmp_voxel_gpu, 0, gridpoint_nr * sizeof(float), stream));
  gridwise_aggregation_global<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      voxelize_dynamics_gpu, tmp_voxel_gpu, _frame_nr, gridpoint_nr, type_agg);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpyAsync(voxelize_dynamics, tmp_voxel_gpu, gridpoint_nr * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));

  if (use_ctx) {
    if (!ctx->pending())
      ctx->synchronize();
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(coord_gpu));
    CUDA_CHECK(cudaFree(weight_gpu));
    CUDA_CHECK(cudaFree(tmp_voxel_gpu));
    CUDA_CHECK(cudaFree(voxelize_dynamics_gpu));
    CUDA_CHECK(cudaFree(dims_gpu));
  }
}
