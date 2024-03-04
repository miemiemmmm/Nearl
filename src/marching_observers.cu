#include <iostream>

#include "gpuutils.cuh"  // For hard-coded BLOCK_SIZE and device functions: mean_device, mean_device, standard_deviation_device


// Device kernel (observation): particle count in frame i
__device__ float particle_count_device(const float *coord, float *coord_framei, 
  int atomnr, float cutoff){
  // Work on each atoms in a frame to calculate the observable in that frame
  float dist, dist1, dist2, dist3;
  float retval = 0.0;
  float cutoff_sq = cutoff * cutoff;
  for (int j = 0; j < atomnr; j++){
    // Calculate the distance
    dist1 = coord[0] - coord_framei[j*3];
    dist2 = coord[1] - coord_framei[j*3+1];
    dist3 = coord[2] - coord_framei[j*3+2];
    
    if (dist1 > cutoff || dist2 > cutoff || dist3 > cutoff){
      continue;
    } else {
      dist = dist1*dist1 + dist2*dist2 + dist3*dist3;
      if (dist > cutoff_sq){
        continue;
      } else {
        // Within the cutoff count the particle
        // TODO: might in some stage pass the weight of the particle HERE
        retval += 1.0;
      }
    }
  }
  return retval;
}

// Device kernel (observation): particle exist in frame i
__device__ float particle_exist_device(const float *coord, float *coord_framei, 
  int atomnr, float cutoff){
  // Work on each atoms in a frame to calculate the observable in that frame
  float dist, dist1, dist2, dist3;
  float cutoff_sq = cutoff * cutoff;
  for (int j = 0; j < atomnr; j++){
    // Calculate the distance
    dist1 = coord[0] - coord_framei[j*3];
    dist2 = coord[1] - coord_framei[j*3+1];
    dist3 = coord[2] - coord_framei[j*3+2];
    
    if (dist1 > cutoff || dist2 > cutoff || dist3 > cutoff){
      continue;
    } else {
      dist = dist1*dist1 + dist2*dist2 + dist3*dist3;
      if (dist > cutoff_sq){
        continue;
      } else {
        // Within the cutoff count the particle
        return 1.0f;
      }
    }
  }
  return 0.0f;
}

// Device kernel (observation): mean distance to observer in frame i
__device__ float mean_distance_device(const float *coord, float *coord_framei,
  int atomnr, float cutoff){
  // Work on each atoms in a frame to calculate the observable in that frame
  float dist, dist1, dist2, dist3;
  float retval = 0.0;
  float cutoff_sq = cutoff * cutoff;
  int count = 0;
  for (int j = 0; j < atomnr; j++){
    // Calculate the distance
    dist1 = coord[0] - coord_framei[j*3];
    dist2 = coord[1] - coord_framei[j*3+1];
    dist3 = coord[2] - coord_framei[j*3+2];
    
    if (dist1 > cutoff || dist2 > cutoff || dist3 > cutoff){
      continue;
    } else {
      dist = dist1*dist1 + dist2*dist2 + dist3*dist3;
      if (dist > cutoff_sq){
        continue;
      } else {
        // Within the cutoff count the particle
        retval += sqrt(dist);
        count += 1;
      }
    }
  }
  if (count > 0){
    return retval / count;
  } else {
    return 0.0;
  }
}


// Device kernel (observation): Radius of gyration in frame i
__device__ float radius_of_gyration_device(const float *coord, const float *coord_framei,
  const int atomnr, const float cutoff){
  // Work on each atoms in a frame to calculate the observable in that frame
  float dist, dist1, dist2, dist3;
  float retval = 0.0;
  float cutoff_sq = cutoff * cutoff;
  int count = 0;

  float *centroid = new float[3];
  
  // __device__ void centroid_device(const float *coord, float *centroid, const int point_nr, const int dim);
  centroid_device(coord_framei, centroid, atomnr, 3);

  for (int j = 0; j < atomnr; j++){
    // Calculate the distance
    dist1 = coord[0] - coord_framei[j*3];
    dist2 = coord[1] - coord_framei[j*3+1];
    dist3 = coord[2] - coord_framei[j*3+2];
    
    if (dist1 > cutoff || dist2 > cutoff || dist3 > cutoff){
      continue;
    } else {
      dist = dist1*dist1 + dist2*dist2 + dist3*dist3;
      if (dist > cutoff_sq){
        continue;
      } else {
        // Within the cutoff count the particle
        retval += (dist1 - centroid[0]) * (dist1 - centroid[0]) + 
                  (dist2 - centroid[1]) * (dist2 - centroid[1]) + 
                  (dist3 - centroid[2]) * (dist3 - centroid[2]);
        count += 1;
      }
    }
  }
  delete[] centroid;
  if (count > 0){
    return sqrt(retval / count);
  } else {
    return 0.0;
  }
}


// Global kernel: particle count
__global__ void grid_dynamic_global(float *result, const float *coord_frames, 
  const int *dims, const float spacing, 
  const int frame_number, const int atomnr, 
  const float cutoff, const int type_observable, const int type_aggregation){ 
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dims[0]*dims[1]*dims[2]){
    float series[1000]; 
    float coord_framei[3000];
    if (frame_number > 1000 || atomnr > 1000){
      printf("The frame number or atom number is too large, please reduce the number of frames or atoms");
      return;
    }

    // Get the coordinate of the grid point
    float coord[3] = {
      static_cast<float>(index / (dims[0] * dims[1])) * spacing,
      static_cast<float>((index / dims[0]) % dims[1]) * spacing,
      static_cast<float>(index % dims[0]) * spacing
    }; 
    
    // Iterate over frames
    int coord_start_index = 0; 
    for (int i = 0; i < frame_number; i++){
      // Extract the coordinates of the frame
      for (int j = 0; j < 3*atomnr; j++){
        coord_framei[j] = coord_frames[coord_start_index + j];
      }
      // Calculate the observable in the frame
      float ret_framei;
      if (type_observable == 0){
        ret_framei = particle_count_device(coord, coord_framei, atomnr, cutoff);
      } else if (type_observable == 1){
        ret_framei = particle_exist_device(coord, coord_framei, atomnr, cutoff);
      } else if (type_observable == 2){
        ret_framei = mean_distance_device(coord, coord_framei , atomnr, cutoff);
      } else if (type_observable == 3){
        ret_framei = radius_of_gyration_device(coord, coord_framei, atomnr, cutoff);
      }
      coord_start_index += atomnr*3; 
      series[i] = ret_framei; 
    }

    // Formulate the return value from the time series of the observer
    float ret_series; 
    if (type_aggregation == 0){
      ret_series = mean_device(series, frame_number);
    } else if (type_aggregation == 1){
      ret_series = standard_deviation_device(series, frame_number);
    } else if (type_aggregation == 2){
      // ret_series = median_device(series, frame_number);
    } else if (type_aggregation == 3){
      ret_series = variance_device(series, frame_number);
    }

    // printf("Result of observer %d is: %f\n", index, ret_series);
    result[index] = static_cast<float>(ret_series);
  }
}


// Normal host function available to C++ part 
void marching_observer_host(float *grid_return, const float *coord, 
  const int *dims, const float spacing, 
  const int frame_number, const int atom_per_frame,
  const float cutoff, const int type_obs, const int type_agg){
  // Determine the number of observers
  const int observer_number = dims[0] * dims[1] * dims[2];

  // Set all numbers in the return array to 0
  float *ret_arr; 
  cudaMalloc(&ret_arr, observer_number * sizeof(float));
  cudaMemset(ret_arr, 0, observer_number * sizeof(float));

  float *coord_device;
  cudaMalloc(&coord_device, atom_per_frame * frame_number * 3 * sizeof(float));
  cudaMemcpy(coord_device, coord, atom_per_frame * frame_number * 3 * sizeof(float), cudaMemcpyHostToDevice);

  int *dims_device;
  cudaMalloc(&dims_device, 3 * sizeof(int));
  cudaMemcpy(dims_device, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

  // NOTE: The coordinate should be uniformed meaning each frame have the same number of atoms
  int grid_size = (observer_number + BLOCK_SIZE - 1) / BLOCK_SIZE;
  grid_dynamic_global<<<grid_size, BLOCK_SIZE>>>(
    ret_arr, coord_device, 
    dims_device, spacing, 
    frame_number, atom_per_frame, 
    cutoff, type_obs, type_agg
  );
  cudaDeviceSynchronize();
  cudaMemcpy(grid_return, ret_arr, observer_number * sizeof(float), cudaMemcpyDeviceToHost);
  
  // TODO: Is there necessity to normalize the return array?
  // NOTE: Currently, normalize the return by the number of atoms in the frame
  float sum_return = 0.0;
  for (int i = 0; i < observer_number; i++) sum_return += grid_return[i]; 
  for (int i = 0; i < observer_number; i++) grid_return[i] = grid_return[i] * atom_per_frame / sum_return;

  cudaFree(ret_arr);
  cudaFree(coord_device);
  cudaFree(dims_device);
}


