//
// Created by yzhang on 06.10.23.
//

// #ifndef NEARL_CPP_BASUTILS_CUH
// #define NEARL_CPP_BASUTILS_CUH
// #include <iostream>
// #include <cmath>

#define BLOCK_SIZE 256

__device__ double sum_device(const double *x, const int n);
__device__ float sum_device(const float *x, const int n);
__device__ float mean_device(const float *x, const int n);
__device__ float standard_deviation_device(const float *x, const int n);
__device__ float variance_device(const float *x, const int n);
__device__ float median_device(float *x, const int n);
__device__ void centroid_device(const float *coord, float *centroid, const int point_nr, const int dim);

void CosineSimilarityBatch(const float *vecs1, const float *vecs2, float *result, int rows1, int rows2, int cols);
void CosineSimilarityQueryMax(const float *vecs1, const float *vecs2, unsigned int *ret_indexes, unsigned int rows1, unsigned int rows2, unsigned int cols);
void CosineSimilarityQueryMin(const float *vecs1, const float *vecs2, unsigned int *ret_indexes, unsigned int rows1, unsigned int rows2, unsigned int cols);

// #endif
