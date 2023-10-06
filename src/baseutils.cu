#include "baseutils.cuh"

using namespace std;

// Use float to obtain the highest speed
__device__ float CosineSimilarity(const float *vec1, const float *vec2, int dim) {
  double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
  for (int idx = 0; idx < dim; ++idx) {
    dot += vec1[idx] * vec2[idx];
    denom_a += vec1[idx] * vec1[idx];
    denom_b += vec2[idx] * vec2[idx];
  }
  return dot / (sqrt(denom_a) * sqrt(denom_b));
}

__global__ void CosineSimilarityKernel(const float *vecs1, const float *vecs2, float *result, int rows1, int rows2, int cols) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows1 * rows2) {
    int idx = index / rows2;
    int idy = index % rows2;
    result[index] = CosineSimilarity(vecs1 + idx * cols, vecs2 + idy * cols, cols);
  }
}


// Compute the cosine similarity between two batches of vectors
// Pure C++ function to do batch similarity computation
void CosineSimilarityBatch(const float *vecs1, const float *vecs2, float *result, int rows1, int rows2, int cols) {
  int total = rows1 * rows2;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  float *vecs1_device, *vecs2_device, *result_device;
  cudaMalloc(&vecs1_device, rows1 * cols * sizeof(float));
  cudaMalloc(&vecs2_device, rows2 * cols * sizeof(float));
  cudaMalloc(&result_device, rows1 * rows2 * sizeof(float));

  cudaMemcpy(vecs1_device, vecs1, rows1 * cols * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vecs2_device, vecs2, rows2 * cols * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(result_device, result, rows1 * rows2 * sizeof(float), cudaMemcpyHostToDevice);

  CosineSimilarityKernel<<<blocks, threads>>>(vecs1_device, vecs2_device, result_device, rows1, rows2, cols);

  cudaMemcpy(result, result_device, rows1 * rows2 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(vecs1_device);
  cudaFree(vecs2_device);
  cudaFree(result_device);
}

//void CosineSimilarityQueryMax(const float *vecs1, const float *vecs2, int rows1, int rows2, int cols) {
//  int total = rows1 * rows2;
//  int threads = 256;
//  int blocks = (total + threads - 1) / threads;
//  float *temp_result = new float[rows1 * rows2];
//  float *vecs1_device, *vecs2_device, *result_device;
//
//  cudaMalloc(&vecs1_device, rows1 * cols * sizeof(float));
//  cudaMalloc(&vecs2_device, rows2 * cols * sizeof(float));
//  cudaMalloc(&result_device, rows1 * cols * sizeof(float));
//
//  cudaMemcpy(vecs1_device, vecs1, rows1 * cols * sizeof(float), cudaMemcpyHostToDevice);
//  cudaMemcpy(vecs2_device, vecs2, rows2 * cols * sizeof(float), cudaMemcpyHostToDevice);
//  cudaMemcpy(result_device, temp_result, rows1 * rows2 * sizeof(float), cudaMemcpyHostToDevice);
//
//  CosineSimilarityKernel<<<blocks, threads>>>(vecs1_device, vecs2_device, result_device, rows1, rows2, cols);
//  cudaMemcpy(temp_result, result_device, rows1 * rows2 * sizeof(float), cudaMemcpyDeviceToHost);
//  cudaFree(vecs1_device);
//  cudaFree(vecs2_device);
//  cudaFree(result_device);
//  for (int idx = 0; idx < rows1; ++idx) {
//    float max = 0.0;
//    for (int idy = 0; idy < rows2; ++idy) {
//      if (temp_result[idx * rows2 + idy] > max) {
//        max = temp_result[idx * rows2 + idy];
//      }
//    }
//    cout << max << endl;
//  }
//  return
//
//
//
//}

