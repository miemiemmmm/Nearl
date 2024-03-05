#include <iostream>
#include <vector>


#include "gpuutils.cuh"
// #include "voxelize.cuh" // TODO: Check the necessity of this file

// TODO : and simplify this module. 


__global__ void sum_kernel(const float *array, float *result, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N){
    atomicAdd(result, array[index]);
  }
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


void voxelize_host(float *interpolated, 
  const float *coord, 
  const float *weight, 
  const int *dims, 
  const int atom_nr, 
  const float spacing, 
  const float cutoff, 
  const float sigma
){
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
    coordi_interp_kernel<<<grid_size, BLOCK_SIZE>>>(coordi_gpu, tmp_interp_gpu, dims_gpu, spacing, cutoff, sigma);
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


void trajectory_voxelization_host(
  float *voxelize_dynamics, 
  const float *coord, 
  const float *weight, 
  const int *dims, 
  const int frame_nr, 
  const int atom_nr, 
  const int interval, 
  const float spacing,
  const float cutoff, 
  const float sigma
){
  /*
    Interpolate the trajectory to a grid using the Gaussian distance mapping
    interpolated should be a 1D array of 4D things sized as (frame_nr, dims[0], dims[1], dims[2])
   */
  unsigned int gridpoint_nr = dims[0] * dims[1] * dims[2];
  float *tmp_framei_cpu = new float[gridpoint_nr];
  float *voxelized_trajectory = new float[frame_nr * gridpoint_nr];

  float *coordi_gpu;
  cudaMallocManaged(&coordi_gpu, 3 * sizeof(float));

  int *dims_gpu;
  cudaMallocManaged(&dims_gpu, 3 * sizeof(int));
  cudaMemcpy(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

  float *tmp_interp_gpu;
  cudaMallocManaged(&tmp_interp_gpu, gridpoint_nr * sizeof(float));

  float *voxelized_atoms_gpu = new float(frame_nr * atom_nr * gridpoint_nr * sizeof(float));
  cudaMallocManaged(&voxelized_atoms_gpu, frame_nr * atom_nr * gridpoint_nr * sizeof(float));

  float *tmp_sum;
  cudaMallocManaged(&tmp_sum, sizeof(float));

  float coordi[3] = {0.0f, 0.0f, 0.0f};
  int grid_size = (gridpoint_nr + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int frame_idx = 0; frame_idx < frame_nr; ++frame_idx) {
    // Copy the coordinates of the frame to the GPU and do interpolation by frames and atoms
    for (int ai = 0; ai < atom_nr; ++ai) {
      int stride = frame_idx * atom_nr * 3 + ai * 3;
      coordi[0] = coord[stride];
      coordi[1] = coord[stride + 1];
      coordi[2] = coord[stride + 2];
      cudaMemcpy(coordi_gpu, coordi, 3 * sizeof(float), cudaMemcpyHostToDevice);

      int start_idx = frame_idx * atom_nr * gridpoint_nr + ai * gridpoint_nr;
      coordi_interp_kernel<<<grid_size, BLOCK_SIZE>>>(coordi_gpu, voxelized_atoms_gpu + start_idx, dims_gpu, spacing, cutoff, sigma);
      sum_kernel<<<grid_size, BLOCK_SIZE>>>(voxelized_atoms_gpu + start_idx, tmp_sum, gridpoint_nr);
      normalize_kernel<<<grid_size, BLOCK_SIZE>>>(voxelized_atoms_gpu + start_idx, tmp_sum, weight[ai], gridpoint_nr);
    }
  }

  // Pool the atomic density for each frame
  float *voxel_cache;
  cudaMallocManaged(&voxel_cache, frame_nr * gridpoint_nr * sizeof(float));
  cudaMemset(voxel_cache, 0, frame_nr * gridpoint_nr * sizeof(float));

  int grid_size3 = (frame_nr * gridpoint_nr + BLOCK_SIZE - 1) / BLOCK_SIZE;
  pool_per_frame_global<<<grid_size3, BLOCK_SIZE>>>(voxelized_atoms_gpu, voxel_cache, atom_nr, gridpoint_nr*frame_nr);

  // Copy the pooled atomic density to the host
  cudaMemcpy(voxelized_trajectory, voxel_cache, frame_nr * gridpoint_nr * sizeof(float), cudaMemcpyDeviceToHost);

  // Aggregate the atomic density to the voxelized trajectory every N steps
  int ret_size = frame_nr / interval;
  if (ret_size != frame_nr / interval){
    std::cerr << "Warning: The frame number is not divisible by the step size; " << std::endl; 
  }

  // float tmp_array[interval]; 
  float stat; 
  for (int i = 0; i < ret_size; ++i){
    for (int j = 0; j < gridpoint_nr; ++j){
      // Obtain the temporary array
      // for (int k = 0; k < interval; ++k){
      //   tmp_array[k] = voxelized_trajectory[i * interval * gridpoint_nr  + k * gridpoint_nr + j];
      // }
      // Hard-coded statistical function: mean
      stat = 0.0f;
      for (int k = 0; k < interval; ++k){
        // stat += tmp_array[k];
        stat += voxelized_trajectory[i * interval * gridpoint_nr  + k * gridpoint_nr + j];
      }
      stat /= interval;
      voxelize_dynamics[i*gridpoint_nr + j] = stat;
    }
  }

  // Delete the temporary CPU memory
  delete[] voxelized_trajectory;    // (frame_nr, dims[0], dims[1], dims[2])
  delete[] tmp_framei_cpu;          // (dims[0], dims[1], dims[2])
  // Free the GPU memory
  cudaFree(coordi_gpu);
  cudaFree(dims_gpu);
  cudaFree(tmp_interp_gpu);
  cudaFree(voxelized_atoms_gpu);
  cudaFree(voxel_cache);
  cudaFree(tmp_sum);
}


