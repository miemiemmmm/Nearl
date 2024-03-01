// Kernel functions
__device__ double gaussian_device(const double x, const double mu, const double sigma) {
	return exp(-0.5 * ((x - mu) / sigma) * ((x - mu) / sigma)) / (sigma * sqrt(2 * M_PI));
}

__device__ double distance(const double *coord1, const double *coord2, const int dim) {
  double sum = 0.0;
  for (int i = 0; i < dim; ++i) {
    sum += (coord1[i] - coord2[i]) * (coord1[i] - coord2[i]);
  }
  return sqrt(sum);
}

__device__ float mean_device(const float *x, const int n) {
  float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += x[i];
  }
  return sum / n;
}

__device__ double sum_device(const double *x, const int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += x[i];
  }
  return sum;
}

__device__ float sum_device(const float *x, const int n) {
  float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += x[i];
  }
  return sum;
}

__device__ float standard_deviation_device(const float *x, const int n) {
  float mean = mean_device(x, n);
  float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += (x[i] - mean) * (x[i] - mean);
  }
  return sqrt(sum / n);
}

__device__ float variance_device(const float *x, const int n) {
  float mean = mean_device(x, n);
  float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += (x[i] - mean) * (x[i] - mean);
  }
  return sum / n;
}

// TODO: determine later 
// __device__ float median_device(float *x, const int n) {
//   thrust::sort(x, x + n);
//   if (n % 2 == 0) {
//     return (x[n / 2 - 1] + x[n / 2]) / 2;
//   } else {
//     return x[n / 2];
//   }
// }


__device__ void centroid_device(const float *coord, float *centroid, const int point_nr, const int dim) {
  for (int i = 0; i < point_nr; ++i) {
    for (int j = 0; j < dim; ++j) {
      centroid[j] += coord[i * dim + j];
    }
  }
  for (int i = 0; i < dim; ++i) {
    centroid[i] /= point_nr;
  }
}









///////////////////////////////////////////////////////////////////////////////
// TODO: Not sure if it is necessary to use the following functions


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

__global__ void CosineSimilarityKernel(const float *vecs1, const float *vecs2, float *result, unsigned int rows1, unsigned int rows2, unsigned int cols) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows1 * rows2) {
    int idx = index / rows2;
    int idy = index % rows2;
    result[index] = CosineSimilarity(vecs1 + idx * cols, vecs2 + idy * cols, cols);
  }
}

__global__ void QueryMax(const float *sim_matrix, unsigned int *result_array, unsigned int rows1, unsigned int rows2) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < rows1) {
		float max = 0.0;
		int max_index = 0;
		for (int idy = 0; idy < rows2; ++idy) {
			if (sim_matrix[index * rows2 + idy] > max) {
				max = sim_matrix[index * rows2 + idy];
				max_index = idy;
			}
		}
		result_array[index] = max_index;
	}
}

__global__ void QueryMin(const float *sim_matrix, unsigned int *result_array, unsigned int rows1, unsigned int rows2) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < rows1) {
		float min = 1.0;
		int min_index = 0;
		for (int idy = 0; idy < rows2; ++idy) {
			if (sim_matrix[index * rows2 + idy] < min) {
				min = sim_matrix[index * rows2 + idy];
				min_index = idy;
			}
		}
		result_array[index] = min_index;
	}
}



// Compute the cosine similarity between two batches of vectors
// Pure C++ function to do batch similarity computation
void CosineSimilarityBatch(const float *vecs1, const float *vecs2, float *result, int rows1, int rows2, int cols) {
  unsigned long int total = rows1 * rows2;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  float *vecs1_device, *vecs2_device, *result_device;
  cudaMalloc(&vecs1_device, rows1 * cols * sizeof(float));
  cudaMalloc(&vecs2_device, rows2 * cols * sizeof(float));
  cudaMalloc(&result_device, total * sizeof(float));

  cudaMemcpy(vecs1_device, vecs1, rows1 * cols * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vecs2_device, vecs2, rows2 * cols * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(result_device, result, total * sizeof(float), cudaMemcpyHostToDevice);

  CosineSimilarityKernel<<<blocks, threads>>>(vecs1_device, vecs2_device, result_device, rows1, rows2, cols);

  cudaMemcpy(result, result_device, total * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(vecs1_device);
  cudaFree(vecs2_device);
  cudaFree(result_device);
}

void CosineSimilarityQueryMax(const float *vecs1, const float *vecs2, unsigned int *ret_indexes, unsigned int rows1, unsigned int rows2, unsigned int cols) {
 unsigned long int total = rows1 * rows2;
 int threads = 256;
 int blocks = (total + threads - 1) / threads;

 unsigned int *result_device;  // cuda variable to host the M*N matrix results
 cudaMalloc(&result_device, rows1 * sizeof(unsigned int));

 float *vecs1_device, *vecs2_device, *sim_matrix_device;  // cuda variable to host the M*X or N*X cooridnate results
 cudaMalloc(&vecs1_device, rows1 * cols * sizeof(float));
 cudaMalloc(&vecs2_device, rows2 * cols * sizeof(float));
 cudaMalloc(&sim_matrix_device, total * sizeof(float));   // No need for output the result

 cudaMemcpy(vecs1_device, vecs1, rows1 * cols * sizeof(float), cudaMemcpyHostToDevice);
 cudaMemcpy(vecs2_device, vecs2, rows2 * cols * sizeof(float), cudaMemcpyHostToDevice);
 CosineSimilarityKernel<<<blocks, threads>>>(vecs1_device, vecs2_device, sim_matrix_device, rows1, rows2, cols);  // Compute the cosine similarity
 QueryMax<<<blocks, threads>>>(sim_matrix_device, result_device, rows1, rows2);

 cudaMemcpy(ret_indexes, result_device, rows1 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
 cudaFree(vecs1_device);
 cudaFree(vecs2_device);
 cudaFree(result_device);
 cudaFree(sim_matrix_device);
}


void CosineSimilarityQueryMin(const float *vecs1, const float *vecs2, unsigned int *ret_indexes, unsigned int rows1, unsigned int rows2, unsigned int cols) {
 unsigned long int total = rows1 * rows2;
 int threads = 256;
 int blocks = (total + threads - 1) / threads;

 unsigned int *result_device;  // cuda variable to host the M*N matrix results
 cudaMalloc(&result_device, rows1 * sizeof(unsigned int));

 float *vecs1_device, *vecs2_device, *sim_matrix_device;  // cuda variable to host the M*X or N*X cooridnate results
 cudaMalloc(&sim_matrix_device, total * sizeof(float));   // No need for output the result
 cudaMalloc(&vecs1_device, rows1 * cols * sizeof(float));
 cudaMalloc(&vecs2_device, rows2 * cols * sizeof(float));

 cudaMemcpy(vecs1_device, vecs1, rows1 * cols * sizeof(float), cudaMemcpyHostToDevice);
 cudaMemcpy(vecs2_device, vecs2, rows2 * cols * sizeof(float), cudaMemcpyHostToDevice);
 CosineSimilarityKernel<<<blocks, threads>>>(vecs1_device, vecs2_device, sim_matrix_device, rows1, rows2, cols);  // Compute the cosine similarity
 QueryMin<<<blocks, threads>>>(sim_matrix_device, result_device, rows1, rows2);

 cudaMemcpy(ret_indexes, result_device, rows1 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
 cudaFree(vecs1_device);
 cudaFree(vecs2_device);
 cudaFree(result_device);
 cudaFree(sim_matrix_device);
}


