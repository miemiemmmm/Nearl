//
// Created by yzhang on 06.10.23.
// Description: This file contains the utility functions for GPU computing. 
//

#include "cuda_runtime.h"
#include "constants.h"
#include "gpuutils.cuh"


__global__ void sum_reduction_global(const float *d_in, float *d_out, const int N){
  __shared__ float smem[BLOCK_SIZE];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  // Load the input data into shared memory 
  smem[tid] = (idx < N) ? d_in[idx] : 0; 
  __syncthreads();

  // Perform reduction in shared memory 
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
    if (tid < stride){
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  // Write the result to global memory 
  if (tid == 0){
    d_out[blockIdx.x] = smem[0];
  }
}


__global__ void normalize_array_global(float *d_in, const float sum, const float weight, const int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  d_in[idx] = d_in[idx] * weight / sum ;
}


__global__ void voxel_addition_global(float *d_parent, float *d_add, const int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  d_parent[idx] += d_add[idx];
}


__global__ void gridwise_aggregation_global(float *d_in, float *d_out, const int frame_nr, const int gridpoint_nr, const int type_agg){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= gridpoint_nr) return;
  
  float tmp_array[MAX_FRAME_NUMBER]; 
  for (int i = 0; i < frame_nr; i++){
    tmp_array[i] = d_in[i * gridpoint_nr + idx];
  }

  if (type_agg == 1){
    d_out[idx] = mean_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 2){
    d_out[idx] = standard_deviation_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 3){
    d_out[idx] = median_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 4){
    d_out[idx] = variance_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 5){
    d_out[idx] = max_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 6){
    d_out[idx] = min_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 7){
    d_out[idx] = information_entropy_device(tmp_array, frame_nr);
  } else if (type_agg == 8){
    d_out[idx] = slope_device<float>(tmp_array, frame_nr);
  } else {
    // Should throw exception in the python-end 
    d_out[idx] = 0; 
  } 
}


void aggregate_host(
  float *voxel_traj, 
  float *result_grid, 
  const int frame_number,
  const int grid_number, 
  const int type_agg 
){
  unsigned int grid_size = (grid_number + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int _frame_number = frame_number > MAX_FRAME_NUMBER ? MAX_FRAME_NUMBER : frame_number; 

  // Move the voxelized stuff 
  float *voxel_traj_gpu; 
  cudaMalloc(&voxel_traj_gpu, frame_number * grid_number * sizeof(float));
  cudaMemcpy(voxel_traj_gpu, voxel_traj, frame_number * grid_number * sizeof(float), cudaMemcpyHostToDevice); 

  float *tmp_grid_gpu; 
  cudaMalloc(&tmp_grid_gpu, grid_number * sizeof(float)); 
  cudaMemset(tmp_grid_gpu, 0, grid_number * sizeof(float)); 

  gridwise_aggregation_global<<<grid_size, BLOCK_SIZE>>>(voxel_traj_gpu, tmp_grid_gpu, _frame_number, grid_number, type_agg);
  cudaMemcpy(result_grid, tmp_grid_gpu, grid_number * sizeof(float), cudaMemcpyDeviceToHost); 

  // Free the memory 
  cudaFree(voxel_traj_gpu);
  cudaFree(tmp_grid_gpu);
}

float sum_reduction_host(
  float *array, 
  const int arr_length
){
  unsigned int grid_size = (arr_length + BLOCK_SIZE - 1) / BLOCK_SIZE; 

  float *partial_sums; 
  cudaMalloc(&partial_sums, grid_size * sizeof(float)); 
  float *array_gpu; 
  cudaMalloc(&array_gpu, arr_length * sizeof(float)); 
  cudaMemcpy(array_gpu, array, arr_length * sizeof(float), cudaMemcpyHostToDevice); 

  // Perform the sum reduction on the array 
  sum_reduction_global<<<grid_size, BLOCK_SIZE>>>(array_gpu, partial_sums, arr_length); 
  cudaDeviceSynchronize();

  // Compute the final sum 
  float _partial_sums[grid_size]; 
  float tmp_sum = 0.0f;
  cudaMemcpy(_partial_sums, partial_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < grid_size; ++i) tmp_sum += _partial_sums[i]; 

  // Free the memory
  cudaFree(partial_sums);
  cudaFree(array_gpu);

  return tmp_sum;
}
