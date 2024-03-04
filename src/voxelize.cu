#include <iostream>
#include <vector>

#include "gpuutils.cuh"
#include "voxelize.cuh" // TODO: Check the necessity of this file

// TODO : and simplify this module. 


__global__ void sum_kernel(const float *array, float *result, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N)
    atomicAdd(result, array[index]);
}


__global__ void add_temparray_kernel(const float *temp_array, float *interpolated, const int grid_size){
  /*
  */
  unsigned int task_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (task_index < grid_size){
    interpolated[task_index] += temp_array[task_index];
  }
}

// Host functions ???
// TODO: Check the necessity of the following host functions
float sum_host(std::vector<float> input_array) {
  /*
  C++ wrapper of the CUDA kernel function sum_host
  */
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

float sum_host(const float *input_array, int N) {
  /*
  C++ wrapper of the CUDA kernel function sum_host
  */
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


__global__ void coordi_interp_kernel(const float *coord, float *interpolated, const int *dims, const float cutoff, const float sigma){
  /*
  Interpolate the atomic density to the grid
  */
  unsigned int task_index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int grid_size = dims[0] * dims[1] * dims[2];

  if (task_index < grid_size){
    // Compute the grid coordinate from the grid index
    // TODO: Check why the grid coordinate is like this???
    float grid_coord[3] = {
      static_cast<float>(task_index / dims[0] / dims[1]),
      static_cast<float>(task_index / dims[0] % dims[1]),
      static_cast<float>(task_index % dims[0]),
    };
    float dist_square = 0.0f;
    for (int i = 0; i < 3; ++i) {
      float diff = coord[i] - grid_coord[i];
      dist_square += diff * diff;
    }
    // Process the interpolated array with the task_index (Should not be race condition)
    if (dist_square < cutoff * cutoff)
      interpolated[task_index] = gaussian_map_device(sqrt(dist_square), 0.0f, static_cast<float>(sigma));
  }
}


__global__ void g_grid_entropy(double* data1, double* data2, int* atominfo, double* gridpoints,
  int atom_nr, int gridpoint_nr, int d, double cutoff_sq) {
  /*
  Compute the entropy of each grid by modifying the gridpoints[i];
  */
  /*
  data1: coordinates of the grid points
  data2: coordinates of the atoms
  atominfo: atom information
  gridpoints: grid points to be calculated
  */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Note there is limitation in the Max number of shared memory in CUDA; ~48KB, ~12K int, ~6K double
  const int MAX_INFO_VALUE = 10000;
  __shared__ int temp_atominfo[MAX_INFO_VALUE];

  if (i < gridpoint_nr) {
    // Compute the entropy of each grid by modifying the gridpoints[i];
    double dist_sq;
    bool skip;
    for (int j = 0; j < atom_nr; j++){
      dist_sq = 0.0;
      skip = false;
      for (int k = 0; k < d; k++){
        double diff = data1[j*d + k] - data2[i*d + k];
        if (abs(diff) > cutoff_sq) { skip = true; break; }
        dist_sq += diff * diff;
        if (dist_sq > cutoff_sq) { skip = true; break; }
      }
      if (!skip){
        int idx = atominfo[j] % MAX_INFO_VALUE;
        atomicAdd(&temp_atominfo[idx], 1);
        printf("tid: %d - bid %d; atomic information: %d ; occurrences: %d; \n", threadIdx.x, blockIdx.x, idx, temp_atominfo[idx]);
      }
    }
    double entropy_val = 0.0;
    for (int tmp = 0; tmp < MAX_INFO_VALUE; ++tmp) {
      if (temp_atominfo[tmp] != 0) {
        double prob = temp_atominfo[tmp] / atom_nr;
        entropy_val += prob * log(prob);
      }
    }
    atomicAdd(&gridpoints[i], -entropy_val);
  }
}


void voxelize_host(float *interpolated, const float *coord, const float *weight, const int *dims, 
  const int atom_nr, const float cutoff, const float sigma){
  /*
    Interpolate the atomic density to a grid using the Gaussian function
  */
  unsigned int gridpoint_nr = dims[0] * dims[1] * dims[2];

  float *tmp_interp_cpu = new float[gridpoint_nr];
  float *tmp_interp_gpu;
  cudaMallocManaged(&tmp_interp_gpu, gridpoint_nr * sizeof(float));

  float *interp_gpu; 
  cudaMallocManaged(&interp_gpu, gridpoint_nr * sizeof(float));

  float *coordi_gpu;
  cudaMallocManaged(&coordi_gpu, 3 * sizeof(float));

  int *dims_gpu;
  cudaMallocManaged(&dims_gpu, 3 * sizeof(int));
  cudaMemcpy(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

  int grid_size = (gridpoint_nr + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int atm_idx = 0; atm_idx < atom_nr; ++atm_idx) {
    // Copy the coordinates of the atom to the GPU and do interpolation on this atom
    float coordi[3] = {coord[atm_idx*3], coord[atm_idx*3+1], coord[atm_idx*3+2]};
    cudaMemcpy(coordi_gpu, coordi, 3 * sizeof(float), cudaMemcpyHostToDevice);
    coordi_interp_kernel<<<grid_size, BLOCK_SIZE>>>(coordi_gpu, tmp_interp_gpu, dims_gpu, cutoff, sigma);
    cudaDeviceSynchronize();

    // Perform the normalization with CPU
    cudaMemcpy(tmp_interp_cpu, tmp_interp_gpu, gridpoint_nr * sizeof(float), cudaMemcpyDeviceToHost);
    float tmp_sum = 0; 
    // Compute the sum of the temporary array
    for (int i = 0; i < gridpoint_nr; ++i) {
      tmp_sum += tmp_interp_cpu[i];
    }
    if (tmp_sum - 0 < 0.001) {
      /* The sum of the temporary array is used for normalization, skip if the sum is 0 */
      std::cerr << "Warning: The sum of the temporary array is 0; It might be due to CUDA error or the coordinate is out of the box; " << std::endl;
      std::cerr << "Coordinate: " << coordi[0] << " " << coordi[1] << " " << coordi[2] << std::endl;
      std::cerr << "Boundary: " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
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
  if (std::abs(final_sum_check - weight_sum) > 0.0001*weight_sum){
    std::cerr << "Warning: The sum of the interpolated array is not equal to the sum of the weights: " << final_sum_check << "/" << weight_sum << std::endl;
  } else if (final_sum_check - 0 < 0.0001*weight_sum) {
    std::cerr << "Warning: The sum of the interpolated array is 0" << std::endl;
  }

  // Copy the interpolated array to the host and free the GPU memory
  delete[] tmp_interp_cpu;
  cudaFree(tmp_interp_gpu);
  cudaFree(interp_gpu);
  cudaFree(coordi_gpu);
  cudaFree(dims_gpu);
}


