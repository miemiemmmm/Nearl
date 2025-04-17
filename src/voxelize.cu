// Created by: Yang Zhang 
// Description: The CUDA implementation of the property density flow algorithm

#include <iostream>

#include "constants.h"
#include "gpuutils.cuh"   // For CUDA kernels
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
    static_cast<float>(task_index / (dims[0] * dims[1])) * spacing,
    static_cast<float>((task_index / dims[0]) % dims[1]) * spacing,
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


__global__ void frame_interp_global(
  const float *coords_frame, 
  const float *weights_frame,
  float *interpolated_frame,
  const int *dims,
  const float spacing,
  const float cutoff,
  const float sigma, 
  const int atom_nr
){
  // Each block is responsible for one atom 
  const int atom_idx = blockIdx.x; 
  const int buff_dim = (cutoff + spacing) / spacing; 
  const int buff_dims[3] = {dims[0] + buff_dim + buff_dim, dims[1] + buff_dim + buff_dim, dims[2] + buff_dim + buff_dim};
  const int gridpoint_buff_nr = buff_dims[0] * buff_dims[1] * buff_dims[2];
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const float *coord = coords_frame + atom_idx * 3;
  const float cutoff_sq = cutoff * cutoff;
  const float weight = weights_frame[atom_idx];

  if (coord[0] == DEFAULT_COORD_PLACEHOLDER && 
      coord[1] == DEFAULT_COORD_PLACEHOLDER && 
      coord[2] == DEFAULT_COORD_PLACEHOLDER){
    return;
  } 
  if (weight == 0.0f) return; 
  
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x; 

  float local_sum = 0.0f; 
  int x, y, z, rem; 
  float grid_x, grid_y, grid_z, dist_sq; 

  // For each block, compute the partial sum 
  // for (int gid = tid; gid < gridpoint_nr; gid += num_threads){
  //   x = gid / dims[0] / dims[1];
  //   y = gid / dims[0] % dims[1];
  //   z = gid % dims[0];
  
  for (int gid = tid; gid < gridpoint_buff_nr; gid += num_threads){
    x = gid / buff_dims[0] / buff_dims[1];
    y = gid / buff_dims[0] % buff_dims[1];
    z = gid % buff_dims[0];

    x -= buff_dim; 
    y -= buff_dim; 
    z -= buff_dim; 

    grid_x = x * spacing;
    grid_y = y * spacing;
    grid_z = z * spacing;

    dist_sq = (coord[0] - grid_x) * (coord[0] - grid_x) + 
              (coord[1] - grid_y) * (coord[1] - grid_y) + 
              (coord[2] - grid_z) * (coord[2] - grid_z);
    if (dist_sq < cutoff_sq) local_sum += gaussian_map_device(sqrt(dist_sq), 0.0f, sigma);
  }

  // Store partial sum to shared memory 
  smem[tid] = local_sum;
  __syncthreads();

  for (int stride = num_threads / 2; stride > 0; stride >>= 1){
    if (tid < stride){
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  const float total_sum = smem[0];
  if (total_sum == 0) return; 

  const float inv_sum = weight / total_sum;
  for (int gid = tid; gid < gridpoint_nr; gid += num_threads){
    x = gid / dims[0] / dims[1];
    y = gid / dims[0] % dims[1];
    z = gid % dims[0];

    grid_x = x * spacing; 
    grid_y = y * spacing; 
    grid_z = z * spacing; 

    dist_sq = (coord[0] - grid_x) * (coord[0] - grid_x) + 
              (coord[1] - grid_y) * (coord[1] - grid_y) + 
              (coord[2] - grid_z) * (coord[2] - grid_z);

    if (dist_sq < cutoff_sq){
      // interpolated_frame[gid] += gaussian_map_device(sqrt(dist_sq), 0.0f, sigma) * inv_sum; 
      atomicAdd(interpolated_frame + gid, gaussian_map_device(sqrt(dist_sq), 0.0f, sigma) * inv_sum);
    }
  }
}


/**
 * @brief Interpolate the atomic density to a grid using the Gaussian function
 */
void voxelize_host_cpu(
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

  int ai=0,aj=0,ak=0,gix=-1;
  int damax = ceil(cutoff/spacing);

  // std::cout << "damax: " << damax << std::endl;

  float dvec[3],pvec[3],grid_spac[3],grid_llim[3];
  float c2 = cutoff*cutoff, d2=0.0, netw=0.0;

  for (int dd=0;dd<3;dd++) {grid_spac[dd] = spacing; } // ??????????????????

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

    ai = floor((coord[offset])/grid_spac[0]);
    aj = floor((coord[offset+1])/grid_spac[1]);
    ak = floor((coord[offset+2])/grid_spac[2]);
    netw = 0.0;
    for (int ii=ai-damax;ii<=ai+damax;ii++) {
      dvec[0] = coord[offset]-(grid_llim[0]+(ii+0.5)*grid_spac[0]);
      for (int jj=aj-damax;jj<=aj+damax;jj++) {
        dvec[1] = coord[offset+1]-(grid_llim[1]+(jj+0.5)*grid_spac[1]);
        for (int kk=ak-damax;kk<=ak+damax;kk++) {
          dvec[2] = coord[offset+2]-(grid_llim[2]+(kk+0.5)*grid_spac[2]);
          d2 = dvec[0]*dvec[0]+dvec[1]*dvec[1]+dvec[2]*dvec[2];
          if (d2 > c2) {continue;}
          netw += exp(-0.5/sigma);
        }
      }
    }
    netw = 1.0/netw;
    for (int ii=ai-damax;ii<=ai+damax;ii++) {
      dvec[0] = coord[offset]-(grid_llim[0]+(ii+0.5)*grid_spac[0]);
      for (int jj=aj-damax;jj<=aj+damax;jj++) {
        dvec[1] = coord[offset+1]-(grid_llim[1]+(jj+0.5)*grid_spac[1]);
        for (int kk=ak-damax;kk<=ak+damax;kk++) {
          gix = ii*dims[0]*dims[0]+jj*dims[1]+kk;
          if ((gix < 0) || (gix >= gridpoint_nr)) {continue;} 
          dvec[2] = coord[offset+2]-(grid_llim[2]+(kk+0.5)*grid_spac[2]);
          d2 = dvec[0]*dvec[0]+dvec[1]*dvec[1]+dvec[2]*dvec[2];
          if (d2 > c2) {continue;}
          interpolated[gix] += netw*weight[atm_idx]*exp(-0.5/sigma);
        }
      } 
    }
  }
}



/**
 * @brief Interpolate the atomic density to a grid using the Gaussian function
 */
void _voxelize_host_old(
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

  float *coord_gpu;
  cudaMalloc(&coord_gpu, atom_nr * 3 * sizeof(float));
  cudaMemcpy(coord_gpu, coord, atom_nr * 3 * sizeof(float), cudaMemcpyHostToDevice);
  float *weight_gpu;
  cudaMalloc(&weight_gpu, atom_nr * sizeof(float));
  cudaMemcpy(weight_gpu, weight, atom_nr * sizeof(float), cudaMemcpyHostToDevice);
  int *dims_gpu;
  cudaMalloc(&dims_gpu, 3 * sizeof(int));
  cudaMemcpy(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);
  float *tmp_voxel_gpu;
  cudaMalloc(&tmp_voxel_gpu, gridpoint_nr * sizeof(float));
  cudaMemset(tmp_voxel_gpu, 0.0f, gridpoint_nr * sizeof(float));

  frame_interp_global<<<atom_nr, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
    coord_gpu, 
    weight_gpu, 
    tmp_voxel_gpu, 
    dims_gpu, 
    spacing, 
    cutoff, 
    sigma, 
    atom_nr
  ); 
  cudaDeviceSynchronize(); 

  // Copy the interpolated array to the host 
  cudaMemcpy(interpolated, tmp_voxel_gpu, gridpoint_nr * sizeof(float), cudaMemcpyDeviceToHost); 

  cudaFree(coord_gpu);
  cudaFree(weight_gpu);
  cudaFree(dims_gpu);
  cudaFree(tmp_voxel_gpu);
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
  const unsigned int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const unsigned int grid_size = (gridpoint_nr + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float *coord_gpu; 
  cudaMalloc(&coord_gpu, frame_nr * atom_nr * 3 * sizeof(float));
  cudaMemcpy(coord_gpu, coord, frame_nr * atom_nr * 3 * sizeof(float), cudaMemcpyHostToDevice);
  float *weight_gpu;
  cudaMalloc(&weight_gpu, frame_nr * atom_nr * sizeof(float));
  cudaMemcpy(weight_gpu, weight, frame_nr * atom_nr * sizeof(float), cudaMemcpyHostToDevice);

  float *tmp_voxel_gpu;
  cudaMalloc(&tmp_voxel_gpu, gridpoint_nr * sizeof(float));
  cudaMemset(tmp_voxel_gpu, 0.0f, gridpoint_nr * sizeof(float));

  float *voxelize_dynamics_gpu;
  cudaMalloc(&voxelize_dynamics_gpu, frame_nr * gridpoint_nr * sizeof(float));
  cudaMemset(voxelize_dynamics_gpu, 0, frame_nr * gridpoint_nr * sizeof(float));

  int *dims_gpu;
  cudaMalloc(&dims_gpu, 3 * sizeof(int));
  cudaMemcpy(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

  for (int frame_idx = 0; frame_idx < frame_nr; ++frame_idx) {
    // Perform the observation of all the grid points (observers) in the frame i 
    frame_interp_global<<<atom_nr, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
      coord_gpu + frame_idx * atom_nr * 3, 
      weight_gpu + frame_idx * atom_nr, 
      voxelize_dynamics_gpu + frame_idx * gridpoint_nr,
      dims_gpu, 
      spacing, 
      cutoff, 
      sigma, 
      atom_nr
    );
    cudaDeviceSynchronize();
    if (frame_idx+1 >= MAX_FRAME_NUMBER){ 
      continue; 
    }
  }

  // Aggregate the frames and copy the result to the host 
  const int _frame_nr = frame_nr > MAX_FRAME_NUMBER ? MAX_FRAME_NUMBER : frame_nr; 
  cudaMemset(tmp_voxel_gpu, 0, gridpoint_nr * sizeof(float));
  gridwise_aggregation_global<<<grid_size, BLOCK_SIZE>>>(voxelize_dynamics_gpu, tmp_voxel_gpu, _frame_nr, gridpoint_nr, type_agg);
  cudaDeviceSynchronize(); 
  cudaMemcpy(voxelize_dynamics, tmp_voxel_gpu, gridpoint_nr * sizeof(float), cudaMemcpyDeviceToHost); 

  // Free the GPU memory 
  cudaFree(coord_gpu);
  cudaFree(weight_gpu);
  cudaFree(tmp_voxel_gpu);
  cudaFree(voxelize_dynamics_gpu);
  cudaFree(dims_gpu);
}

/* 
Only a test function for basic performance comparison 
*/
void trajectory_voxelization_host_cpu(
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
  const unsigned int gridpoint_nr = dims[0] * dims[1] * dims[2];
  const unsigned int grid_size = (gridpoint_nr + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int frame_idx = 0; frame_idx < frame_nr; ++frame_idx) { 
    voxelize_host_cpu(
      voxelize_dynamics, 
      coord + frame_idx * atom_nr * 3, 
      weight + frame_idx * atom_nr, 
      dims, 
      spacing, 
      atom_nr, 
      cutoff, 
      sigma
    ); 
  }
}

