//
// Created by yzhang on 06.10.23.
//
#include "cuda_runtime.h"

#ifndef GPU_UTILS_INCLUDED
#define GPU_UTILS_INCLUDED

#define BLOCK_SIZE 256


template <typename T>
__device__ T max_device(const T *x, const int n){
  T max = x[0];
  for (int i = 1; i < n; i++){
    if (x[i] > max){
      max = x[i];
    }
  }
  return max;
}

template <typename T>
__device__ T min_device(const T *x, const int n){
  T min = x[0];
  for (int i = 1; i < n; i++){
    if (x[i] < min){
      min = x[i];
    }
  }
  return min;
}

template <typename T>
__device__ T sum_device(const T *x, const int n){
  T sum = 0;
  for (int i = 0; i < n; i++){
    sum += x[i];
  }
  return sum;
}


// TODO test the template. 
template <typename T>
__device__ float mean_device(const T *x, const int n){
  T thesum = sum_device(x, n);
  float ret = static_cast<float>(sum_device(x, n)) / n;
  return ret;
}


template <typename T>
__device__ float standard_deviation_device(const T *x, const int n){
  float mean = mean_device(x, n);
  T sum = 0;
  for (int i = 0; i < n; i++){
    sum += (x[i] - mean) * (x[i] - mean);
  }
  float ret = sqrtf(sum / n); 
  return ret;
}

template <typename T>
__device__ float variance_device(const T *x, const int n){
  float mean = mean_device(x, n);
  T sum = 0;
  for (int i = 0; i < n; i++){
    sum += (x[i] - mean) * (x[i] - mean);
  }
  return sum / n;
}


template <typename T>
__device__ float cosine_similarity(const T *vec1, const T *vec2, int dim){
  float dot = 0;
  float norm1 = 0;
  float norm2 = 0;
  for (int i = 0; i < dim; i++){
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
__device__ float distance(const T *coord1, const T *coord2, const int dim) {
  float sum = 0;
  for (int i = 0; i < dim; i++){
    sum += (coord1[i] - coord2[i]) * (coord1[i] - coord2[i]);
  }
  return sqrtf(sum);
}



// template <typename T>
// __device__ float median_device(float *x, const int n);
// TODO: Media is not very easy to parallelize.
// __device__ float median_device(float *x, const int n) {
//   thrust::sort(x, x + n);
//   if (n % 2 == 0) {
//     return (x[n / 2 - 1] + x[n / 2]) / 2;
//   } else {
//     return x[n / 2];
//   }
// }


#endif
