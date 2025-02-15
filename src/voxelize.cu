// Created by: Yang Zhang 
// Description: The CUDA implementation of the property density flow algorithm

#include <iostream>

#include "constants.h"
#include "gpuutils.cuh"   // For CUDA kernels
// #include "cpuutils.h" 
#include "voxelize.cuh"

/**
 *Interpolate the atomic density to the grid
 */
__global__ void coordi_interp_global(const float *coord, float *interpolated, const int *dims, const float spacing, const float cutoff, const float sigma){
  unsigned int task_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int grid_size = dims[0] * dims[1] * dims[2];
  if (task_index >= grid_size) return;

  // Compute the grid coordinate from the grid index
  float grid_coord[3] = {
    static_cast<float>(task_index / dims[0] / dims[1]) * spacing,
    static_cast<float>(task_index / dims[0] % dims[1]) * spacing,
    static_cast<float>(task_index % dims[0]) * spacing
  };
  float dist_square = 0.0f;
  for (int i = 0; i < 3; ++i) {
    dist_square += (coord[i] - grid_coord[i]) * (coord[i] - grid_coord[i]);
  }
  // Process the interpolated array with the task_index (Should not be race conditions)
  if (dist_square < cutoff * cutoff){
    interpolated[task_index] = gaussian_map_device(sqrt(dist_square), 0.0f, sigma);
  } else {
    // Set to 0 to avoid reuse of the previous value
    interpolated[task_index] = 0.0f;
  }
}


/**
 * @brief Interpolate the atomic density to a grid using the Gaussian function
 */
void voxelize_host(
  float *interpolated, 
  const float *coord, 
  const float *weight, 
  const int *dims, 
  const float spacing, 
  const int atom_nr, 
  const float cutoff, 
  const float sigma
){
  unsigned int gridpoint_nr = dims[0] * dims[1] * dims[2];
  unsigned int grid_size = (gridpoint_nr + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float _partial_sums[grid_size];

  float *coord_gpu; 
  cudaMalloc(&coord_gpu, atom_nr * 3 * sizeof(float));
  cudaMemcpy(coord_gpu, coord, atom_nr * 3 * sizeof(float), cudaMemcpyHostToDevice);
  float *tmp_interp_gpu;
  cudaMalloc(&tmp_interp_gpu, gridpoint_nr * sizeof(float));
  cudaMemset(tmp_interp_gpu, 0, gridpoint_nr * sizeof(float)); 
  float *interp_gpu;
  cudaMalloc(&interp_gpu, gridpoint_nr * sizeof(float)); 
  cudaMemset(interp_gpu, 0, gridpoint_nr * sizeof(float)); 
  float *partial_sums; 
  cudaMalloc(&partial_sums, grid_size * sizeof(float)); 
  int *dims_gpu;
  cudaMalloc(&dims_gpu, 3 * sizeof(int)); 
  cudaMemcpy(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice); 

  for (int atm_idx = 0; atm_idx < atom_nr; ++atm_idx) {
    // Copy the coordinates of the atom to the GPU and do interpolation on this atom
    int offset = atm_idx * 3; 

    // Skip the padded coordinates (all coordinates are DEFAULT_COORD_PLACEHOLDER) and zero weights
    if (coord[offset] == DEFAULT_COORD_PLACEHOLDER && 
        coord[offset+1] == DEFAULT_COORD_PLACEHOLDER && 
        coord[offset+2] == DEFAULT_COORD_PLACEHOLDER){
      continue;
    } 

    if (weight[atm_idx] == 0.0f){
      // Skip the voxelization if the weight is 0
      continue;
    }

    coordi_interp_global<<<grid_size, BLOCK_SIZE>>>(coord_gpu + offset, tmp_interp_gpu, dims_gpu, spacing, cutoff, sigma);
    cudaDeviceSynchronize();

    // Perform the sum reduction on the GPU and sum them up 
    sum_reduction_global<<<grid_size, BLOCK_SIZE>>>(tmp_interp_gpu, partial_sums, gridpoint_nr); 
    cudaDeviceSynchronize(); 
    float tmp_sum = 0.0f; 
    cudaMemcpy(_partial_sums, partial_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost); 
    for (int i = 0; i < grid_size; ++i) tmp_sum += _partial_sums[i]; 

    // Normalize the temporary array
    if (tmp_sum != 0){ 
      normalize_array_global<<<grid_size, BLOCK_SIZE>>>(tmp_interp_gpu, tmp_sum, weight[atm_idx], gridpoint_nr);
      cudaDeviceSynchronize();
    } 

    // Add the interpolated GPU array to the output array
    voxel_addition_global<<<grid_size, BLOCK_SIZE>>>(interp_gpu, tmp_interp_gpu, gridpoint_nr);
    cudaDeviceSynchronize();
  }

  // Copy the interpolated array to the host
  cudaMemcpy(interpolated, interp_gpu, gridpoint_nr * sizeof(float), cudaMemcpyDeviceToHost);

  // Free the GPU memory
  cudaFree(coord_gpu); 
  cudaFree(tmp_interp_gpu);
  cudaFree(interp_gpu);
  cudaFree(dims_gpu);
  cudaFree(partial_sums);
}


/**
 * @brief Voxelization of the trajectory and aggregation of the frames 
 * 
 * @param voxelize_dynamics The output array for the voxelized trajectory
 * @param coord The atomic coordinates with shape (frame_nr, atom_nr, 3) 
 * @param weight The atomic weights
 * @param dims The dimensions of the grid
 * 
 */
void trajectory_voxelization_host(
  float *voxelize_dynamics, 
  const float *coord, 
  const float *weight, 
  const int *dims, 
  const float spacing, 
  const int frame_nr, 
  const int atom_nr, 
  const float cutoff, 
  const float sigma,
  const int type_agg
){
  unsigned int gridpoint_nr = dims[0] * dims[1] * dims[2];
  unsigned int grid_size = (gridpoint_nr + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float _partial_sums[grid_size];

  float *coord_gpu; 
  cudaMalloc(&coord_gpu, frame_nr * atom_nr * 3 * sizeof(float));
  cudaMemcpy(coord_gpu, coord, frame_nr * atom_nr * 3 * sizeof(float), cudaMemcpyHostToDevice);
  float *tmp_interp_gpu;
  cudaMalloc(&tmp_interp_gpu, gridpoint_nr * sizeof(float));
  float *interp_gpu;
  cudaMalloc(&interp_gpu, gridpoint_nr * sizeof(float));
  float *voxelized_traj; 
  cudaMalloc(&voxelized_traj, frame_nr * gridpoint_nr * sizeof(float));
  cudaMemset(voxelized_traj, 0, frame_nr * gridpoint_nr * sizeof(float));
  float *partial_sums; 
  cudaMalloc(&partial_sums, grid_size * sizeof(float)); 
  int *dims_gpu;
  cudaMalloc(&dims_gpu, 3 * sizeof(int));
  cudaMemcpy(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

  for (int frame_idx = 0; frame_idx < frame_nr; ++frame_idx) {
    cudaMemset(interp_gpu, 0, gridpoint_nr * sizeof(float));
    for (int ai = 0; ai < atom_nr; ++ai) {
      int offset = frame_idx * atom_nr * 3 + ai * 3;

      // Skip the padded coordinates (all coordinates are DEFAULT_COORD_PLACEHOLDER) and zero weights
      if (coord[offset] == DEFAULT_COORD_PLACEHOLDER && 
          coord[offset+1] == DEFAULT_COORD_PLACEHOLDER && 
          coord[offset+2] == DEFAULT_COORD_PLACEHOLDER){
        continue;
      }
      if (weight[frame_idx * atom_nr + ai] == 0.0f){
        continue;
      }

      coordi_interp_global<<<grid_size, BLOCK_SIZE>>>(coord_gpu + offset, tmp_interp_gpu, dims_gpu, spacing, cutoff, sigma);
      cudaDeviceSynchronize();

      // Perform the sum reduction on the GPU
      sum_reduction_global<<<grid_size, BLOCK_SIZE>>>(tmp_interp_gpu, partial_sums, gridpoint_nr);
      cudaDeviceSynchronize();
      cudaMemcpy(_partial_sums, partial_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
      float tmp_sum = 0.0f;
      for (int i = 0; i < grid_size; ++i) tmp_sum += _partial_sums[i];

      if (tmp_sum != 0) {
        normalize_array_global<<<grid_size, BLOCK_SIZE>>>(tmp_interp_gpu, tmp_sum, weight[frame_idx * atom_nr + ai], gridpoint_nr);
        cudaDeviceSynchronize();
      }

      // Add the interpolated GPU array to the interp_gpu
      voxel_addition_global<<<grid_size, BLOCK_SIZE>>>(interp_gpu, tmp_interp_gpu, gridpoint_nr);
      cudaDeviceSynchronize();
    }
    // Assign the interpolated array to the voxelized trajectory in GPU
    cudaMemcpy(voxelized_traj + frame_idx * gridpoint_nr, interp_gpu, gridpoint_nr * sizeof(float), cudaMemcpyDeviceToDevice);

    // Skip the frames if their index exceeds the maximum number of frames allowed due to the GPU-based aggregation
    if (frame_idx+1 >= MAX_FRAME_NUMBER){ 
      continue; 
    }
  }

  // Perform frame-wise aggregation on the voxelized trajectory 
  unsigned int _frame_nr = frame_nr > MAX_FRAME_NUMBER ? MAX_FRAME_NUMBER : frame_nr; 
  cudaMemset(tmp_interp_gpu, 0, gridpoint_nr * sizeof(float)); 
  gridwise_aggregation_global<<<grid_size, BLOCK_SIZE>>>(voxelized_traj, tmp_interp_gpu, _frame_nr, gridpoint_nr, type_agg);
  cudaDeviceSynchronize();

  // Copy the aggregated voxelized trajectory to the host
  cudaMemcpy(voxelize_dynamics, tmp_interp_gpu, gridpoint_nr * sizeof(float), cudaMemcpyDeviceToHost);

  // Free the GPU memory
  cudaFree(coord_gpu);
  cudaFree(tmp_interp_gpu);
  cudaFree(interp_gpu);
  cudaFree(voxelized_traj);
  cudaFree(partial_sums);
  cudaFree(dims_gpu);
}


