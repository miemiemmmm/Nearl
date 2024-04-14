#include <iostream>
#include <vector>

#include "gpuutils.cuh"
#include "cpuutils.h"

#define DEFAULT_COORD_PLACEHOLDER 99999.0f

// TODO : Simplify this module. 

__global__ void sum_kernel(const float *array, float *result, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N){
    atomicAdd(result, array[index]);
  }
}

__global__ void add_temparray_kernel(const float *temp_array, float *interpolated, const int grid_size){
  unsigned int task_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (task_index < grid_size){
    interpolated[task_index] += temp_array[task_index];
  }
}


/**
 * 
 * C++ wrapper of the CUDA kernel function sum_host
 * Host functions ???
 * TODO: Check the necessity of the following host functions
 */
float sum_host(std::vector<float> input_array) {
  
  int N = input_array.size();
  float *arr, *result;
  cudaMallocManaged(&arr, N * sizeof(float));
  cudaMallocManaged(&result, sizeof(float));
  cudaMemcpy(arr, input_array.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(result, 0, sizeof(float));

  // int threads_per_block = 256;
  int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sum_kernel<<<grid_size, BLOCK_SIZE>>>(arr, result, N);
  cudaDeviceSynchronize();

  float result_host;
  cudaMemcpy(&result_host, result, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(arr);
  cudaFree(result);

  return result_host;
}


/**
 * C++ wrapper of the CUDA kernel function sum_host
 */
float sum_host(const float *input_array, int N) {
  
  float *arr, *result;
  cudaMallocManaged(&arr, N * sizeof(float));
  cudaMallocManaged(&result, sizeof(float));
  cudaMemcpy(arr, input_array, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(result, 0, sizeof(float));

  int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sum_kernel<<<grid_size, BLOCK_SIZE>>>(arr, result, N);
  cudaDeviceSynchronize();

  float result_host;
  cudaMemcpy(&result_host, result, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(arr);
  cudaFree(result);

  return result_host;
}


__global__ void normalize_kernel(float *array, const float *sum, const float weight, const int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N)
    array[index] = (array[index] * weight) / *sum;
}


__global__ void pool_per_frame_global(float *array, float *result, const int pool_size, const int stride){
  /*
    Pool the array per frame
  */
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < stride){
    // strides equals to frame_nr * gridpoint_nr
    // Mapping from (atom_nr * frame_nr * gridpoint_nr) to (frame_nr, gridpoint_nr)
    for (int i = 0; i < pool_size; ++i){
      result[index] += array[index + i * stride]; 
    }
  }
}

__global__ void aggregate_per_frame_global(float *array, float *result, const int pool_size, const int stride, const int func_type){
  /*
    Aggregate the array per frame 
    TODO 
  */
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < stride){
    // strides equals to frame_nr * gridpoint_nr
    // Mapping from (atom_nr * frame_nr * gridpoint_nr) to (frame_nr, gridpoint_nr)
    if (func_type == 0){
      // Use device kernel 

    }
    for (int i = 0; i < pool_size; ++i){
      result[index] += array[index + i * stride]; 
    }
  }
}


__global__ void coordi_interp_kernel(const float *coord, float *interpolated, const int *dims, const float spacing, const float cutoff, const float sigma){
  /*
    Interpolate the atomic density to the grid
   */
  unsigned int task_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int grid_size = dims[0] * dims[1] * dims[2];
  if (task_index < grid_size){
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
  int grid_size = (gridpoint_nr + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float *tmp_interp_cpu = new float[gridpoint_nr];
  float coordi[3] = {0.0f, 0.0f, 0.0f};
  float tmp_sum = 0.0f;

  float *coordi_gpu;
  cudaMallocManaged(&coordi_gpu, 3 * sizeof(float));
  float *tmp_interp_gpu;
  cudaMallocManaged(&tmp_interp_gpu, gridpoint_nr * sizeof(float));
  float *interp_gpu;
  cudaMallocManaged(&interp_gpu, gridpoint_nr * sizeof(float));
  int *dims_gpu;
  cudaMallocManaged(&dims_gpu, 3 * sizeof(int));
  cudaMemcpy(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

  for (int i = 0; i < gridpoint_nr; ++i) { 
    interpolated[i] = 0.0f; 
  }

  for (int atm_idx = 0; atm_idx < atom_nr; ++atm_idx) {
    // Copy the coordinates of the atom to the GPU and do interpolation on this atom
    coordi[0] = coord[atm_idx*3];
    coordi[1] = coord[atm_idx*3+1];
    coordi[2] = coord[atm_idx*3+2];

    if (coordi[0] == DEFAULT_COORD_PLACEHOLDER || coordi[1] == DEFAULT_COORD_PLACEHOLDER || coordi[2] == DEFAULT_COORD_PLACEHOLDER){
      // Skip the voxelization if the coordinate is the default value
      std::cout << "Skipping coordinate: " << coordi[0] << " " << coordi[1] << " " << coordi[2] << "; " << std::endl;
      continue;
    } else if (weight[atm_idx] == 0.0f){
      // Skip the voxelization if the weight is 0
      continue;
    }

    cudaMemcpy(coordi_gpu, coordi, 3 * sizeof(float), cudaMemcpyHostToDevice);
    coordi_interp_kernel<<<grid_size, BLOCK_SIZE>>>(coordi_gpu, tmp_interp_gpu, dims_gpu, spacing, cutoff, sigma);
    cudaDeviceSynchronize();

    // Perform the normalization with CPU
    cudaMemcpy(tmp_interp_cpu, tmp_interp_gpu, gridpoint_nr * sizeof(float), cudaMemcpyDeviceToHost);
    tmp_sum = sum(tmp_interp_cpu, gridpoint_nr);
    if (tmp_sum - 0 < 0.001) {
      /* The sum of the temporary array is used for normalization, skip if the sum is 0 */
      continue;
    } else {
      // Normalize the temporary array
      for (int i = 0; i < gridpoint_nr; ++i) {
        interpolated[i] += tmp_interp_cpu[i] * weight[atm_idx] / tmp_sum;
      }
    }
  }

  // Final check of the interpolation (Check sum)
  float final_sum_check = 0.0f;
  float weight_sum = 0.0f;
  for (int i = 0; i < gridpoint_nr; ++i) {
    final_sum_check += interpolated[i];
  }
  for (int i = 0; i < atom_nr; ++i) {
    weight_sum += weight[i];
  }

  // Copy the interpolated array to the host and free the GPU memory
  delete[] tmp_interp_cpu;
  cudaFree(tmp_interp_gpu);
  cudaFree(interp_gpu);
  cudaFree(coordi_gpu);
  cudaFree(dims_gpu);
}


/**
 * @brief 
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
  int grid_size = (gridpoint_nr + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float *tmp_interp_cpu = new float[gridpoint_nr];
  float *voxelized_traj = new float[frame_nr * gridpoint_nr];
  float coordi[3] = {0.0f, 0.0f, 0.0f};
  float tmp_sum = 0.0f;

  float *coordi_gpu;
  cudaMallocManaged(&coordi_gpu, 3 * sizeof(float));
  float *tmp_interp_gpu;
  cudaMallocManaged(&tmp_interp_gpu, gridpoint_nr * sizeof(float));
  float *interp_gpu;
  cudaMallocManaged(&interp_gpu, gridpoint_nr * sizeof(float));
  int *dims_gpu;
  cudaMallocManaged(&dims_gpu, 3 * sizeof(int));
  cudaMemcpy(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

  for (int i = 0; i < frame_nr * gridpoint_nr; ++i) { voxelized_traj[i] = 0.0f; }

  for (int frame_idx = 0; frame_idx < frame_nr; ++frame_idx) {
    int stride_per_frame = frame_idx * atom_nr * 3; 
    for (int ai = 0; ai < atom_nr; ++ai) {
      coordi[0] = coord[stride_per_frame + ai*3];
      coordi[1] = coord[stride_per_frame + ai*3 + 1];
      coordi[2] = coord[stride_per_frame + ai*3 + 2];
      if (coordi[0] == DEFAULT_COORD_PLACEHOLDER && coordi[1] == DEFAULT_COORD_PLACEHOLDER && coordi[2] == DEFAULT_COORD_PLACEHOLDER){
        // Skip the voxelization if the coordinate is the default value
        // std::cout << "Skipping the default coordinate: " << coordi[0] << " " << coordi[1] << " " << coordi[2] << "; " << std::endl;
        continue;
      }

      cudaMemcpy(coordi_gpu, coordi, 3 * sizeof(float), cudaMemcpyHostToDevice);
      coordi_interp_kernel<<<grid_size, BLOCK_SIZE>>>(coordi_gpu, tmp_interp_gpu, dims_gpu, spacing, cutoff, sigma);
      cudaDeviceSynchronize();

      // Perform the normalization with CPU
      cudaMemcpy(tmp_interp_cpu, tmp_interp_gpu, gridpoint_nr * sizeof(float), cudaMemcpyDeviceToHost);
      tmp_sum = sum(tmp_interp_cpu, gridpoint_nr);
      
      if (tmp_sum - 0 < 0.001) {
        /* The sum of the temporary array is used for normalization, skip if the sum is 0 */
        // std::cerr << "Warning: The sum of the temporary array is 0; It might be due to CUDA error or the coordinate is out of the box; " << std::endl;
        // std::cerr << "Coordinate: " << coordi[0] << " " << coordi[1] << " " << coordi[2] << "; ";
        // std::cerr << "Boundary: " << dims[0]*spacing << " " << dims[1]*spacing << " " << dims[2]*spacing << std::endl;
        continue;
      } else {
        // Normalize the temporary array
        for (int i = 0; i < gridpoint_nr; ++i) {
          voxelized_traj[frame_idx*gridpoint_nr + i] += tmp_interp_cpu[i] * weight[frame_idx * atom_nr + ai] / tmp_sum;
        }
      }
    }
  }

  // Aggregate the atomic density to the voxelized trajectory every N frames
  float tmp_array[frame_nr]; 
  for (int i = 0; i < gridpoint_nr; ++i){
    for (int j = 0; j < frame_nr; ++j){
      tmp_array[j] = voxelized_traj[j * gridpoint_nr + i];
    }
    if (type_agg == 0){
      voxelize_dynamics[i] = mean(tmp_array, frame_nr);
    } else if (type_agg == 1){
      voxelize_dynamics[i] = median(tmp_array, frame_nr);
    } else if (type_agg == 2){
      voxelize_dynamics[i] = standard_deviation(tmp_array, frame_nr);
    } else if (type_agg == 3){
      voxelize_dynamics[i] = variance(tmp_array, frame_nr);
    } else if (type_agg == 4){
      voxelize_dynamics[i] = max(tmp_array, frame_nr);
    } else if (type_agg == 5){
      voxelize_dynamics[i] = min(tmp_array, frame_nr);
    } else {
      // Throw Exception
      throw std::invalid_argument("The aggregation type " + std::to_string(type_agg) + " is not supported");
    }
  }
}


