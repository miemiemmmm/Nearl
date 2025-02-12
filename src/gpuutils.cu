//
// Created by yzhang on 06.10.23.
// Description: This file contains the utility functions for GPU computing. 
//

#include "cuda_runtime.h"
#include "constants.h"
#include "gpuutils.cuh"


__global__ void sum_reduction_global(float *d_in, float *d_out, const int N){
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
  d_in[idx] = d_in[idx] / sum * weight;
}


__global__ void voxel_addition_global(float *d_in, float *d_out, const int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  d_out[idx] += d_in[idx];
}


__global__ void gridwise_aggregation_global(float *d_in, float *d_out, const int frame_nr, const int gridpoint_nr, const int type_agg){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= gridpoint_nr) return;
  
  // TODO: Check if this works correctly 
  float tmp_array[MAX_FRAME_NUMBER]; 
  
  if (type_agg == 0){
    d_out[idx] = mean_device(tmp_array, frame_nr);
  } else if (type_agg == 1){
    d_out[idx] = median_device(tmp_array, frame_nr);
  } else if (type_agg == 2){
    d_out[idx] = standard_deviation_device(tmp_array, frame_nr);
  } else if (type_agg == 3){
    d_out[idx] = variance_device(tmp_array, frame_nr);
  } else if (type_agg == 4){
    d_out[idx] = max_device<float>(tmp_array, frame_nr);
  } else if (type_agg == 5){
    d_out[idx] = min_device(tmp_array, frame_nr);
  } else if (type_agg == 6){
    d_out[idx] = information_entropy_device(tmp_array, frame_nr);
  } else if (type_agg == 7){
    d_out[idx] = slope_device(tmp_array, frame_nr);
  } else {
    // Should throw exception in the python-end 
    d_out[idx] = 0; 
  } 
}

