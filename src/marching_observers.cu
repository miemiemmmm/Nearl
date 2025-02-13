#include <iostream>

#include "constants.h"              // For hard-coded variables: BLOCK_SIZE, MAX_FRAME_NUMBER      
#include "gpuutils.cuh"             // For hard-coded BLOCK_SIZE and device functions: mean_device, mean_device, standard_deviation_device
#include "marching_observers.cuh"   // For the hard coded MAX_FRAME_NUMBER

// TODO: need a throughout check on the observation functions. 

////////////////////////////////////////////////////////////////////////////////
// Direct count-based observables
////////////////////////////////////////////////////////////////////////////////


/**
 * @brief Check the existence of particles in a frame
 *
 * Determines if any atom within a specified cutoff distance exists relative to the observer's 
 * position. This function iterates over all atoms in a given frame and checks if at least one
 * atom exists within the cutoff distance from the reference coordinate. 
 *
 * @param coord The reference coordinate of the observer
 * @param coord_framei The coordinates of all atoms in the frame
 * @param atomnr The number of atoms in the frame
 * @param cutoff The cutoff distance
 * 
 * @return 1.0 if at least one atom exists within the cutoff distance, 0.0 otherwise
 * 
 * @note This direct count-based observation does not consider atom weights.
 */
__device__ float existence_device(const float *coord, const float *coord_framei, 
  const int atomnr, const float cutoff){
  // Work on each atoms in a frame to calculate the observable in that frame
  float dist_sq; 
  float cutoff_sq = cutoff * cutoff;
  for (int j = 0; j < atomnr; ++j){
    // Filter atoms outside the cutoff distance
    dist_sq = square_distance_device(coord, coord_framei + j*3);

    // Return 1 if any atom shows up within the cutoff distance
    if (dist_sq > cutoff_sq){
      continue;
    } else {
      return 1.0f;
    } 
  }
  return 0.0f;
}


/**
 * @brief Count atoms within a specified cutoff from an observer. 
 *
 * This CUDA device function iterates over all atoms in a specified frame and accumulates
 * a count of those within a given cutoff distance from a reference coordinate. 
 *
 * @param coord Pointer to the reference coordinate's float array (x, y, z).
 * @param coord_framei Pointer to the frame's atom coordinates float array, with each 
 *        atom's coordinates stored consecutively as (x, y, z).
 * @param atomnr The total number of atoms in the frame.
 * @param cutoff The distance threshold for counting an atom. Only atoms within this
 *        distance from the reference coordinate are counted.
 *
 * @return The count of atoms within the cutoff distance from the reference coordinate.
 * 
 * @note This direct count-based observation does not consider atom weights.
 */
__device__ float direct_count_device(const float *coord, const float *coord_framei, 
  int atomnr, float cutoff){
  // Work on each atoms in a frame to calculate the observable in that frame
  float dist_sq, retval = 0.0;
  float cutoff_sq = cutoff * cutoff;
  for (int j = 0; j < atomnr; ++j){
    // Filter atoms outside the cutoff distance
    dist_sq = square_distance_device(coord, coord_framei + j*3);

    // Count 1 if the atom is within the cutoff distance
    if (dist_sq > cutoff_sq){
      continue; 
    } else {
      retval += 1.0;
    }
  }
  return retval;
}


/**
 * @brief Count distinct weighted atoms within a cutoff distance from an observer.
 *
 * This CUDA device function iterates over atoms in a frame, checking each atom's
 * distance to a reference coordinate. It counts atoms with unique weights within
 * the specified cutoff distance. Assumes a fixed-size array for tracking encountered
 * values, with a limit of 1000 (hard coded) distinct weights.
 *
 * @param coord Pointer to the reference coordinate's float array (x, y, z).
 * @param coord_framei Pointer to the frame's atom coordinates float array (x, y, z for each atom).
 * @param weight_framei Pointer to the frame's atom weights float array, one per atom.
 * @param atomnr Total number of atoms in the frame.
 * @param cutoff Distance threshold for including an atom in the count.
 * 
 * @return The count of unique-weight atoms within the cutoff distance.
 * 
 * @note This direct count-based observation considers atom weights as discrete identification of 
 * the particles. It is suitable for discrete atomic identities such as atom indices, residue indices, 
 * or other discrete identifiers.
 * The continuous weights (float numbers) are acceptable but will be rounded to the nearest integer 
 * after multiplying by 10 (To avoid floating-point comparison issues).
 * 
 */
__device__ float distinct_count_device(const float *coord, const float *coord_framei, const float *weight_framei,
  int atomnr, float cutoff){
  // Work on each atoms in a frame to calculate the observable in that frame
  float dist_sq; 
  float cutoff_sq = cutoff * cutoff; 
  float distinct_count = 0.0f; 

  // Initialize the encountered values with a default placeholder 
  float encountered_values[DISTINCT_LIMIT];
  for (int i = 0; i < DISTINCT_LIMIT; ++i){
    encountered_values[i] = DEFAULT_PLACEHOLDER;
  }

  float val_check;
  for (int j = 0; j < atomnr; ++j){
    // Filter atoms outside the cutoff distance 
    dist_sq = square_distance_device(coord, coord_framei + j*3); 
    if (dist_sq > cutoff_sq) continue; 
    
    val_check = weight_framei[j];
    // Linear search unique values with early termination and insertion
    for (int k = 0; k < DISTINCT_LIMIT; ++k){
      if (val_check == encountered_values[k]){
        break; 
      } else if (encountered_values[k] == DEFAULT_PLACEHOLDER){
        encountered_values[k] = val_check; 
        distinct_count += 1; 
        break;
      } 
    }
  }
  return distinct_count;
}


////////////////////////////////////////////////////////////////////////////////
// Weight-based observables
////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Weighted mean distance of particles in frame i
 */
__device__ float mean_distance_device(const float *coord, const float *coord_framei, const float *weight_framei,
  const int atomnr, const float cutoff){
  // Work on each atoms in a frame to calculate the observable in that frame
  float dist_sq;
  float cutoff_sq = cutoff * cutoff;

  float retval = 0.0;
  float weight_sum = 0;
  for (int j = 0; j < atomnr; ++j){
    dist_sq = square_distance_device<float>(coord, coord_framei + j*3); 
    if (dist_sq > cutoff_sq) continue; 

    retval += sqrt(dist_sq) * weight_framei[j];
    weight_sum += weight_framei[j];
  }

  return weight_sum;
}


/**
 * @brief Calculate the cumulative weight of particles within a specified cutoff distance.
 */
__device__ float cumulative_weight_device(const float *coord, const float *coord_framei, const float *weight_framei,
  int atomnr, float cutoff){
  float dist_sq, retval = 0.0, cutoff_sq = cutoff * cutoff;
  for (int j = 0; j < atomnr; j++){
    dist_sq = square_distance_device<float>(coord, coord_framei + j*3);
    if (dist_sq > cutoff_sq) continue;  

    // Count the weight if the atom is within the cutoff distance 
    retval += weight_framei[j];
  }
  return retval;
}


/**
 * @brief Calculates the density of particles within a specified cutoff radius from a given point. 
 */
__device__ float density_device(const float *coord, const float *coord_framei, const float *weight_framei,
  const int atomnr, const float cutoff){
  float weight_sum = cumulative_weight_device(coord, coord_framei, weight_framei, atomnr, cutoff);
  float volume = (4.0 / 3.0) * M_PI * cutoff * cutoff * cutoff;
  return weight_sum / volume; 
}


/**
 * @brief Computes the weighted dispersion of particles within a specified cutoff distance.
 *
 * This device function calculates the dispersion of particles based on their pairwise distances
 * and weights. It considers only those pairs of particles that are within a given cutoff distance
 * from each other. The dispersion is computed as a weighted sum of the the pairwise distance, 
 * normalized by the total weight of all considered particle pairs.
 *
 * Parameters are the same as for the other observables. 
 * 
 * @return The calculated weighted dispersion value. If the weight sum of considered particle pairs is zero,
 *         the function returns 0.0, indicating no dispersion or an invalid state.
 *
 */
__device__ float dispersion_device(const float *coord, const float *coord_framei, const float *weight_framei,
  const int atomnr, const float cutoff){
  // Work on each atoms in a frame to calculate the observable in that frame
  float dist_sq, dist_sq_j, dist_sq_k;
  float cutoff_sq = cutoff * cutoff;

  float weight_sum = 0.0f;
  float retval = 0.0f;
  // Pairwise distance summation
  for (int j = 0; j < atomnr; ++j){
    // Filter atom 1 outside the cutoff distance 
    dist_sq_j = square_distance_device(coord, coord_framei + j*3);
    if (dist_sq_j > cutoff_sq) continue; 

    for (int k = j + 1; k < atomnr; ++k){
      // Filter atom 2 outside the cutoff distance
      dist_sq_k = square_distance_device(coord, coord_framei + k*3);
      if (dist_sq_k > cutoff_sq) continue;  

      dist_sq = square_distance_device(coord_framei + j*3, coord_framei + k*3);
      // Need to clarify what is the formula it follows to characterize the dispersion
      retval += weight_framei[j] * weight_framei[k] * sqrt(dist_sq);  
      weight_sum += weight_framei[j] * weight_framei[k];
    }
  }
  if (weight_sum == 0.0){
    return 0.0;
  } else {
    return retval / weight_sum;
  }
}


/**
 * @brief Compute the distance between center of mass to the observer 
 * 
 * This observable is inspired by the eccentricity of cable (distance between 
 * the center of the conductor and the center of the insulation). It computes
 * the weighted center of mass (COM) for a collection of particles within a 
 * specified cutoff distance from a reference point (observer) and then calculates
 * the distance from this COM to the reference point. This function can be used
 * to assess the dispersion or distribution of particles around the observer in 
 * a three-dimensional space.
 * 
 * Parameters are the same as for the other observables. 
 * 
 * @return The distance between the weighted center of mass of the particles (within
 * cutoff) and the observer. If no particles are within the cutoff distance, returns 
 * the cutoff distance as an indication of high dispersion.
 * 
 */
__device__ float eccentricity_device(const float *coord, const float *coord_framei, const float *weight_framei,
  const int atomnr, const float cutoff){
  float dist_sq;
  float cutoff_sq = cutoff * cutoff;
  
  float com[3] = {0.0, 0.0, 0.0};
  float weight_sum = 0.0f;
  // Get the center of mass
  for (int j = 0; j < atomnr; ++j){
    dist_sq = square_distance_device(coord, coord_framei + j*3);
    if (dist_sq > cutoff_sq) continue; 
    com[0] += weight_framei[j] * coord_framei[j*3];
    com[1] += weight_framei[j] * coord_framei[j*3+1];
    com[2] += weight_framei[j] * coord_framei[j*3+2];
    weight_sum += weight_framei[j];
  }

  if (weight_sum > 0){
    com[0] = com[0] / weight_sum;
    com[1] = com[1] / weight_sum;
    com[2] = com[2] / weight_sum;
  } else {
    return cutoff; // Return the cutoff distance if no particles within the cutoff
  }

  float retval = sqrt(square_distance_device(coord, com));
  return retval;
}


/**
 * @brief Radius of gyration of particles within a specified cutoff distance.
 * 
 * This function calculates the radius of gyration for a collection of particles, considering only those
 * within a specified cutoff distance from a given reference point (`coord`).
 * 
 * @result The radius of gyration of the particles within the cutoff distance from the reference point, calculated
 * relative to their center of mass. If no particles are within the cutoff distance, the function returns 0.0.
 * 
 * @note The signs of weights should be geater than 0 (otherwise Center of Mass will be wrong)
 */
__device__ float radius_of_gyration_device(const float *coord, const float *coord_framei, const float *weight_framei,
  const int atomnr, const float cutoff){
  // Work on each atoms in a frame to calculate the observable in that frame
  float dist_sq, cutoff_sq = cutoff * cutoff; 

  // Calculate the center of mass
  float com[3] = {0.0, 0.0, 0.0}; 
  float weight_sum = 0.0f;
  int count = 0; 
  for (int j = 0; j < atomnr; ++j){
    // Filter atoms outside of the cutoff distance 
    dist_sq = square_distance_device(coord, coord_framei+j*3);
    if (dist_sq > cutoff_sq) continue; 

    com[0] += weight_framei[j] * coord_framei[j*3];
    com[1] += weight_framei[j] * coord_framei[j*3+1];
    com[2] += weight_framei[j] * coord_framei[j*3+2];
    weight_sum += weight_framei[j];
    count += 1; 
  }

  if (count <= 1){
    // If no particle or only 1 particle in the cutoff distance, return 0.0
    return 0.0f; 
  } else if (weight_sum != 0){
    com[0] /= weight_sum;
    com[1] /= weight_sum;
    com[2] /= weight_sum;
  } else {
    return 0.0f; 
  }

  // Calculate the radius of gyration to the center of mass
  float retval = 0.0f; 
  for (int j = 0; j < atomnr; ++j){
    dist_sq = square_distance_device(coord, coord_framei+j*3);
    if (dist_sq > cutoff_sq) continue; 

    dist_sq = square_distance_device(com, coord_framei+j*3);
    retval += weight_framei[j] * dist_sq; 
  }
  return sqrt(retval / weight_sum); 
}


/**
 * @brief The device function to calculate the observable in frame i
 */
__device__ float make_observation_device(const float *coord, const float *coord_framei, 
  const int atomnr, const float cutoff, const int type_obs){
  // Does not consider the weight of the particles
  float ret_framei = 0.0f;
  if (type_obs == 1){
    ret_framei = existence_device(coord, coord_framei, atomnr, cutoff);
  } else if (type_obs == 2){
    ret_framei = direct_count_device(coord, coord_framei, atomnr, cutoff);
  }
  return ret_framei; 
}


/**
 * @brief The device function to calculate the observable in frame i
 */
__device__ float make_observation_device(const float *coord, const float *coord_framei, const float *weight_framei,
  const int atomnr, const float cutoff, const int type_obs){
  float ret_framei = 0.0f;
  if (type_obs == 3){
    ret_framei = distinct_count_device(coord, coord_framei, weight_framei, atomnr, cutoff);
  } else if (type_obs == 11){
    ret_framei = mean_distance_device(coord, coord_framei, weight_framei, atomnr, cutoff);
  } else if (type_obs == 12){
    ret_framei = cumulative_weight_device(coord, coord_framei, weight_framei, atomnr, cutoff);
  } else if (type_obs == 13){
    ret_framei = density_device(coord, coord_framei, weight_framei, atomnr, cutoff);
  } else if (type_obs == 14){
    ret_framei = dispersion_device(coord, coord_framei, weight_framei, atomnr, cutoff);
  } else if (type_obs == 15){
    ret_framei = eccentricity_device(coord, coord_framei, weight_framei, atomnr, cutoff);
  } else if (type_obs == 16){
    ret_framei = radius_of_gyration_device(coord, coord_framei, weight_framei, atomnr, cutoff);
  }
  return ret_framei;
}


////////////////////////////////////////////////////////////////////////////////
// Direct particle count-based observables
////////////////////////////////////////////////////////////////////////////////
/**
 * @brief The global kernel function to calculate the observable in a grid point
 */
__global__ void marching_observer_global(
  float *mobs_ret, const float *coord_frame, const float *weight_frame,
  const int *dims, const float spacing, 
  const int frame_number, const int atomnr, 
  const float cutoff, const int type_observable
){ 
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int grid_size = dims[0] * dims[1] * dims[2];
  if (index >= grid_size) return; 

  // Get the coordinate of the grid point (Observer) in real space 
  float coord[3] = {
    static_cast<float>(index / dims[0] / dims[1]) * spacing,
    static_cast<float>(index / dims[0] % dims[1]) * spacing,
    static_cast<float>(index % dims[0]) * spacing
  }; 

  // Calculate the observable of grid point at index in the given frame
  if ((type_observable == 1) || (type_observable == 2)){
    // Hard-coded for the direct count-based observables 
    mobs_ret[index] = make_observation_device(coord, coord_frame, atomnr, cutoff, type_observable); 
  } else {
    mobs_ret[index] = make_observation_device(coord, coord_frame, weight_frame, atomnr, cutoff, type_observable); 
  }
}


/**
 * @brief The host function to perform the marching observer algorithm
 */
void marching_observer_host(
  float *mobs_dynamics, 
  const float *coord, 
  const float *weights,
  const int *dims, 
  const float spacing, 
  const int frame_number, 
  const int atom_per_frame,
  const float cutoff, 
  const int type_obs, 
  const int type_agg
){
  unsigned int observer_number = dims[0] * dims[1] * dims[2];
  unsigned int grid_size = (observer_number + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float _partial_sums[grid_size]; 

  // The (frame_number, observer_number) observed trajectory 
  float *mobs_traj; 
  cudaMalloc(&mobs_traj, frame_number * observer_number * sizeof(float));
  cudaMemset(mobs_traj, 0, frame_number * observer_number * sizeof(float));

  // The resultant aggregated mobs feature (Initialize all digits in the return array to 0)
  float *tmp_mobs_gpu; 
  cudaMalloc(&tmp_mobs_gpu, observer_number * sizeof(float));
  cudaMemset(tmp_mobs_gpu, 0, observer_number * sizeof(float));

  // The atomic coordinates and weights of the frame i in the device memory
  float *coords_device;  
  cudaMalloc(&coords_device, frame_number * atom_per_frame * 3 * sizeof(float));
  float *weights_device;
  cudaMalloc(&weights_device, frame_number * atom_per_frame * sizeof(float));

  int *dims_device;
  cudaMalloc(&dims_device, 3 * sizeof(int));
  cudaMemcpy(dims_device, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

  float *partial_sums; 
  cudaMalloc(&partial_sums, grid_size * sizeof(float)); 

  // NOTE: The coordinate should be uniformed meaning each frame have the same number of atoms
  for (int frame_idx = 0; frame_idx < frame_number; ++frame_idx) {
    // Perform the observation of all the grid points (observers) in the frame i 
    marching_observer_global<<<grid_size, BLOCK_SIZE>>>(
      tmp_mobs_gpu, 
      coords_device + frame_idx * atom_per_frame * 3, 
      weights_device + frame_idx * atom_per_frame, 
      dims_device, spacing, 
      frame_number, atom_per_frame, 
      cutoff, type_obs
    ); 
    cudaDeviceSynchronize();

    // After calculating the frame i, copy the result to the frame-wise array 
    cudaMemcpy(mobs_traj + frame_idx * observer_number, tmp_mobs_gpu, observer_number * sizeof(float), cudaMemcpyDeviceToDevice);

    // Skip the frames if their index exceeds the maximum number of frames allowed due to the GPU-based aggregation
    if (frame_idx+1 >= MAX_FRAME_NUMBER) continue; 
  }

  // Perform frame-wise aggregation on the voxelized trajectory 
  unsigned int _frame_number = frame_number > MAX_FRAME_NUMBER ? MAX_FRAME_NUMBER : frame_number; 
  cudaMemset(tmp_mobs_gpu, 0, observer_number * sizeof(float)); 
  gridwise_aggregation_global<<<grid_size, BLOCK_SIZE>>>(mobs_traj, tmp_mobs_gpu, _frame_number, observer_number, type_agg);

  // IMPORTANT: NEED normalization since the high-diversity of observables and aggregation methods
  // TODO: Check if it is needed or not to normalize the return
  // Perform the normalization if needed 
  float tmp_sum = 0.0f;
  sum_reduction_global<<<grid_size, BLOCK_SIZE>>>(tmp_mobs_gpu, partial_sums, observer_number);
  cudaDeviceSynchronize();
  cudaMemcpy(_partial_sums, partial_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < grid_size; ++i) tmp_sum += partial_sums[i];

  if (tmp_sum != 0.0){
    normalize_array_global<<<grid_size, BLOCK_SIZE>>>(tmp_mobs_gpu, tmp_sum, 1.0f, observer_number); 
  } 

  // Copy the final result to the host memory
  cudaMemcpy(mobs_dynamics, tmp_mobs_gpu, observer_number * sizeof(float), cudaMemcpyDeviceToHost); 

  // Free the memory
  cudaFree(mobs_traj);
  cudaFree(tmp_mobs_gpu);
  cudaFree(coords_device);
  cudaFree(weights_device);
  cudaFree(dims_device);
  cudaFree(partial_sums);
}



void observe_frame_host(float *results, const float *coord_frame, const float *weight_frame, const int *dims, 
  const float spacing, const int atomnr, const float cutoff, const int type_obs
){
  unsigned int observer_number = dims[0] * dims[1] * dims[2];
  unsigned int grid_size = (observer_number + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  int frame_nr = 1; 

  float *results_gpu; 
  cudaMalloc(&results_gpu, frame_nr * observer_number * sizeof(float));
  cudaMemset(results_gpu, 0.0f, frame_nr * observer_number * sizeof(float));
  int *dims_gpu;
  cudaMalloc(&dims_gpu, 3 * sizeof(int));
  cudaMemcpy(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice); 
  float *coord_frame_gpu;
  cudaMalloc(&coord_frame_gpu, frame_nr * atomnr * 3 * sizeof(float));
  cudaMemcpy(coord_frame_gpu, coord_frame, frame_nr * atomnr * 3 * sizeof(float), cudaMemcpyHostToDevice);
  float *weight_frame_gpu;
  cudaMalloc(&weight_frame_gpu, frame_nr * atomnr * sizeof(float));
  cudaMemcpy(weight_frame_gpu, weight_frame, frame_nr * atomnr * sizeof(float), cudaMemcpyHostToDevice);

  marching_observer_global<<<grid_size, BLOCK_SIZE>>>(
    results_gpu, coord_frame_gpu, weight_frame_gpu, 
    dims_gpu, spacing, frame_nr, atomnr, cutoff, type_obs
  );
  cudaDeviceSynchronize();

  cudaMemcpy(results, results_gpu, observer_number * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(results_gpu);
  cudaFree(dims_gpu);

}
