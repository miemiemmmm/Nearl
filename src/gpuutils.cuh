//
// Created by yzhang on 06.10.23.
// Description: This file contains the utility functions for GPU computing. 
//

#ifndef GPU_UTILS_INCLUDED
#define GPU_UTILS_INCLUDED

#include "cuda_runtime.h"
#include "constants.h"


template <typename T>
__device__ T max_device(const T *Arr, const int N){
  T max = Arr[0];
  for (int i = 1; i < N; i++){
    if (Arr[i] > max){
      max = Arr[i];
    }
  }
  return max;
}


template <typename T>
__device__ T min_device(const T *Arr, const int N){
  T min = Arr[0];
  for (int i = 1; i < N; i++){
    if (Arr[i] < min){
      min = Arr[i];
    }
  }
  return min;
}


template <typename T>
__device__ T sum_device(const T *Arr, const int N){
  T sum = 0;
  for (int i = 0; i < N; i++){
    sum += Arr[i];
  }
  return sum;
}


template <typename T>
__device__ float mean_device(const T *Arr, const int N){
  T thesum = sum_device(Arr, N);
  float ret = static_cast<float>(sum_device(Arr, N)) / N;
  return ret;
}


template <typename T>
__device__ float standard_deviation_device(const T *Arr, const int N){
  float mean = mean_device(Arr, N);
  T sum = 0;
  for (int i = 0; i < N; i++){
    sum += (Arr[i] - mean) * (Arr[i] - mean);
  }
  float ret = sqrtf(sum / N); 
  return ret;
}



template <typename T>
__device__ float variance_device(const T *Arr, const int N){
  float mean = mean_device(Arr, N);
  T sum = 0;
  for (int i = 0; i < N; i++){
    sum += (Arr[i] - mean) * (Arr[i] - mean);
  }
  return sum / N;
}


/**
 * @brief Calculates the information entropy of an array on the CUDA device.
 * 
 * This function estimates the information entropy by first creating a probability
 * distribution of the elements in the input array. Each element is scaled and
 * rounded to produce a discrete set of values. The implementation assumes a finite 
 * set of possible values (up to 256 unique values) after the scaling and rounding process.
 * 
 * @tparam T Numerical data type of the array elements
 * @param Arr Pointer to the input array of type T located in device memory
 * @param N The number of elements in the input array
 * @return The computed information entropy as a float
 * 
 * @note The function multiplies each array element by 10 and rounds it to
 *       the nearest integer to simplify the value range. This is a heuristic
 *       approach and might not be suitable for all types of data or applications.
 * 
 */
template <typename T>
__device__ float information_entropy_device(const T *Arr, const int N){
  if (N <= 1){return 0.0f;}
  int met_values[256] = {0};
  int met_counts[256] = {0};
  float entropy_val = 0.0f;

  for (int i = 0; i < N; i++){
    // Trick to rounded any number to a int (1 digit)
    int val = Arr[i] * 10; 
    bool met = false;
    for (int j = 0; j < 256; j++){
      if (met_values[j] == val){
        met = true;
        met_counts[j] += 1;
        break;
      }
    }
    if (!met){
      for (int j = 0; j < 256; j++){
        if (met_values[j] == 0){
          met_values[j] = val;
          met_counts[j] = 1;
          break;
        }
      }
    }
  }
  // Calculate the result based on the probability distribution
  for (int i = 0; i < 256; i++){
    if (met_values[i] == 0){
      break;
    }
    float prob = static_cast<float>(met_counts[i]) / N;
    entropy_val -= prob * log2f(prob);
  }
  return entropy_val;
}


/**
 * @brief Calculates the information entropy of an array based on the histogram on the CUDA device.
 * In this function, it calculates the histogram with 16 bins based on the input array.
 * For each bin, calculate the probability and then the entropy.
 * 
 * TODO: test this function. 
 */
template <typename T>
__device__ float information_entropy_histogram_device(const T *Arr, const int N){
  T min = min_device(Arr, N);
  T max = max_device(Arr, N);
  if (min == max){
    return 0;
  } else {
    int bins = 16;
    T range = max - min;
    T bin_width = range / bins;
    if (bin_width == 0) bin_width = 1;
    
    int hist[bins] = {0};
    int bin = 0;
    for (int i = 0; i < N; ++i){
      bin = (Arr[i] - min) / bin_width;
      bin = bin >= bins ? bins - 1 : bin;
      hist[bin] += 1;
    }

    float entropy_val = 0.0f;
    float prob = 0.0f;
    for (int i = 0; i < bins; ++i){
      if (hist[i] > 0){
        prob = static_cast<float>(hist[i]) / N;
        entropy_val -= prob * log2f(prob);
      }
    }
    return entropy_val;
  }
}


/**
 * @brief Calculate the drift of a time-series data
 *
 * This function calculates the drift of a time-series data. The drift is defined as 
 * the slope of the linear regression line fitted to the data points. 
 * 
 * @tparam T Numerical data type of the time-series data
 * @param arr Pointer to the input array of type T located in device memory
 * @param N The number of elements in the input array
 * 
 * @return The computed drift value as a float
 * 
 * @note The function assumes that the input array contains at least two elements.
 */
template <typename T>
__device__ float slope_device(const T *arr, const int N){
  if (N <= 1) return 0; 

  float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
  for (int i = 0; i < N; i++){
    sum_x += i;
    sum_y += arr[i];
    sum_xy += i * arr[i];
    sum_x2 += i * i;
  }

  const float denom = N * sum_x2 - sum_x * sum_x;
  if (denom == 0) return 0;
  return (N * sum_xy - sum_x * sum_y) / denom; 
}


/**
 * @brief Calculates the median value of an array on the CUDA device
 * 
 * This function firstly sorts the input array in ascending order via the bubble 
 * sort algorithm. Then, it calculates the median value based on the number of
 * elements in the array
 * 
 * @tparam T Numerical data type of the array elements
 * @param Arr Pointer to the input array of type T located in device memory
 * @param N The number of elements in the input array
 * @return The computed median value as a float
 * 
 * @note Since this function modifies the input array in-place, make a copy of
 * the array before calling this function if the original order of elements
 * needs to be preserved.
 * @note This function uses bubble sort for sorting, which has O(N^2) complexity
 * and is not efficient for large arrays.
 * 
 */
template <typename T>
__device__ float median_device(T *Arr, const int N){
  // Sort the array
  for (int i = 0; i < N; i++){
    for (int j = i + 1; j < N; j++){
      if (Arr[i] > Arr[j]){
        T temp = Arr[i];
        Arr[i] = Arr[j];
        Arr[j] = temp;
      }
    }
  }
  // Calculate the median
  if (N % 2 == 0){
    return (Arr[N / 2 - 1] + Arr[N / 2]) / 2;
  } else {
    return Arr[N / 2];
  }
}


/**
 * @brief Calculates the cosine similarity between two vectors on the CUDA device.
 * 
 * This function calculates the cosine similarity between two vectors of the same
 * dimension. The cosine similarity is a measure of similarity between two non-zero
 * vectors of an inner product space that measures the cosine of the angle between
 * them. The cosine of 0 degrees is 1, and it is less than 1 for any other angle.
 * 
 * @tparam T Numerical data type of the vector elements
 * @param vec1 Pointer to the first vector of type T located in device memory
 * @param vec2 Pointer to the second vector of type T located in device memory
 * @param N The number of elements in the input vectors
 */
template <typename T>
__device__ float cosine_similarity(const T *vec1, const T *vec2, int N){
  float dot = 0;
  float norm1 = 0;
  float norm2 = 0;
  for (int i = 0; i < N; i++){
    dot += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }
  return dot / (sqrtf(norm1) * sqrtf(norm2));
}


template <typename T>
__device__ void centroid_device(const T *coord, T *centroid, const int point_nr, const int dim){
  for (int i = 0; i < dim; i++){
    centroid[i] = 0;
  }
  for (int i = 0; i < point_nr; i++){
    for (int j = 0; j < dim; j++){
      centroid[j] += coord[i * dim + j];
    }
  }
  for (int i = 0; i < dim; i++){
    centroid[i] /= point_nr;
  }
}


template <typename T>
__device__ void com_device(const T *coord, const T *mass, T *com, const int point_nr, const int dim){
  for (int i = 0; i < dim; i++){
    com[i] = 0;
  }
  T total_mass = 0;
  for (int i = 0; i < point_nr; i++){
    for (int j = 0; j < dim; j++){
      com[j] += coord[i * dim + j] * mass[i];
    }
    total_mass += mass[i];
  }
  for (int i = 0; i < dim; i++){
    com[i] /= total_mass;
  }
}


// To calculate the distance based gaussian map.
template <typename T>
__device__ float gaussian_map_device(const T distance, const T mu, const T sigma){
  if (sigma == 0){
    return 0;
  } else {
    return exp(-0.5 * ((distance - mu) / sigma) * ((distance - mu) / sigma)) / (sigma * sqrtf(2 * M_PI));
  }
}

template <typename T>
__device__ double gaussian_map_nd_device(const T *x, const T *mu, const T *sigma, const int dim){
  // this function aims to calculate the 
  double ret = 1;
  for (int i = 0; i < dim; i++){
    ret *= gaussian_map_device(x[i], mu[i], sigma[i]); 
  }
  return ret;
}

template <typename T>
__device__ float distance_device(const T *coord1, const T *coord2, const int dim) {
  float sum = 0;
  for (int i = 0; i < dim; i++){
    sum += (coord1[i] - coord2[i]) * (coord1[i] - coord2[i]);
  }
  return sqrtf(sum);
}

template <typename T>
__device__ float square_distance_device(const T *coord1, const T *coord2) {
  float sum = 0;
  sum += (coord1[0] - coord2[0]) * (coord1[0] - coord2[0]);
  sum += (coord1[1] - coord2[1]) * (coord1[1] - coord2[1]);
  sum += (coord1[2] - coord2[2]) * (coord1[2] - coord2[2]);
  return sum;
}


extern __global__ void sum_reduction_global(const float *d_in, float *d_out, const int N); 
extern __global__ void normalize_array_global(float *d_in, const float sum, const float weight, const int N); 
extern __global__ void voxel_addition_global(float *d_in, float *d_out, const int N); 
extern __global__ void gridwise_aggregation_global(float *d_in, float *d_out, const int frame_nr, const int gridpoint_nr, const int type_agg); 

extern void aggregate_host(float *voxel_traj, float *tmp_grid, const int frame_number, const int grid_number, const int type_agg); 
extern float sum_reduction_host(float *array, const int arr_length); 

#endif
