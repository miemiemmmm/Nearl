#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <unordered_map>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

using namespace std;
namespace py = pybind11;

__device__ double d_gaussian(double x, double mu, double sigma) {
  return exp(-0.5 * ((x - mu) / sigma) * ((x - mu) / sigma)) / (sigma * sqrt(2 * M_PI));
}

#pragma acc routine seq
double a_gaussian(double x, double mu, double sigma) {
  const double sqrt_2_pi = sqrt(2.0 * M_PI);
  double a = (x - mu) / sigma;
  return exp(-0.5 * a * a) / (sigma * sqrt_2_pi);
}

template <typename T>
double _entropy(T *arr, int size) {
  if (size <= 1) return 0.0;
  // Use a hashmap to count occurrences of each unique element
  unordered_map<T, int> counts;
  int unique_nr = 0;
  for (int i = 0; i < size; ++i) {
    T xi = arr[i];
    if (counts.count(xi) == 0) {
      counts[xi] = 1;
      unique_nr++;
    } else {
      counts[xi]++;
    }
  }
  // Calculate the probability of each unique element
  T includes[unique_nr];
  int _counts[unique_nr];
  int c = 0;
  for (auto &[key, value] : counts){
    includes[c] = key;
    _counts[c] = value;
    c++;
  }
  // Calculate entropy using the formula: -sum(p * log2(p))
  double result_entropy = 0.0;
  for (int i = 0; i < unique_nr; ++i) {
    double prob = static_cast<double>(_counts[i]) / size;
    result_entropy -= prob * log2(prob);
  }
  return result_entropy;
}

__global__ void g_sum(const float *a, float *result, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
        atomicAdd(result, a[i]);
}

__global__ void g_interpolate(const double *data1, const double *data2, const double *weights,
double *interpolated, int n1, int n2, int d, double cutoff_sq, double sigma){
	/*
	n1 = number of target points (atoms)
	n2 = number of grid points (voxels)
	weights = weights of the target points (atoms)
	interpolate = interpolated values of the grid points (voxels)
	d = number of dimensions (3)
	cutoff_sq = cutoff squared
	sigma = sigma of the Gaussian function
	*/
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n1) return;
	for (int j = 0; j < n2; j++) {
		double dist_sq = 0.0;
		bool skip = false;

		for (int k = 0; k < d; k++) {
			double diff = data1[i * d + k] - data2[j * d + k];
			if (abs(diff) > cutoff_sq) { skip = true; break; }
			dist_sq += diff * diff;
			if (dist_sq > cutoff_sq) { skip = true; break; }
		}
		if (!skip) {
			double dist = sqrt(dist_sq);
			double increment = d_gaussian(dist, 0.0, sigma) * weights[i];
			atomicAdd(&interpolated[j], increment);
		}
	}
}

float sum_array(pybind11::array_t<float> input_array) {
  pybind11::buffer_info buf_info = input_array.request();
  int N = buf_info.size;

  float *a, *result;
  cudaMallocManaged(&a, N * sizeof(float));
  cudaMallocManaged(&result, sizeof(float));
  *result = 0.0f;

  cudaMemcpy(a, buf_info.ptr, N * sizeof(float), cudaMemcpyHostToDevice);

  int threads_per_block = 128;
  int number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  g_sum<<<number_of_blocks, threads_per_block>>>(a, result, N);
  cudaDeviceSynchronize();

  float result_host = *result;
  cudaFree(a);
  cudaFree(result);

  return result_host;
}


py::array_t<double> interpolate(py::array_t<double> arr_grid_coords, py::array_t<double> arr_target,
                                py::array_t<double> arr_weights, const double cutoff = 4, const double sigma = 1.5) {
	py::buffer_info grid = arr_grid_coords.request();
	py::buffer_info targets = arr_target.request();
	py::buffer_info weights = arr_weights.request();

	double *target_data = static_cast<double *>(targets.ptr);
  double *grid_data = static_cast<double *>(grid.ptr);
  double *weights_data = static_cast<double *>(weights.ptr);

	if (grid.ndim != 2 || targets.ndim != 2) {
		throw runtime_error("Number of dimensions must be two");
	}
	int dim2 = 3;  // 3 columns for x, y, z
	if (grid.shape[1] != 3 || targets.shape[1] != 3) {
		throw runtime_error("Number of columns must be three");
	} else {
		dim2 = grid.shape[1];
	}
	if (weights.shape[0] != targets.shape[0]) {
		throw runtime_error("Number of weights must be equal to number of target points");
	}

	const int grid_coord_nr = grid.shape[0];
	const int target_coord_nr = targets.shape[0];
	double *interpolated = new double[grid_coord_nr];
	const double cutoff_sq = cutoff * cutoff;

	cudaError_t err0, err1, err2, err3, err4;
	double *d_coord, *d_target, *d_weights, *d_interpolated;
	err1 = cudaMalloc(&d_coord, grid_coord_nr * dim2 * sizeof(double));
	err2 = cudaMalloc(&d_target, target_coord_nr * dim2 * sizeof(double));
	err3 = cudaMalloc(&d_weights, target_coord_nr * sizeof(double));
	err4 = cudaMalloc(&d_interpolated, grid_coord_nr * sizeof(double));
// 	cout << "Malloc in: " << cudaGetErrorString(err1) << " | " <<  cudaGetErrorString(err2) << " | " << cudaGetErrorString(err3) << " | " << cudaGetErrorString(err4) << endl;
	err1 = cudaMemcpy(d_coord,  grid_data, grid_coord_nr * dim2 * sizeof(double), cudaMemcpyHostToDevice);
	err2 = cudaMemcpy(d_target, target_data, target_coord_nr * dim2 * sizeof(double), cudaMemcpyHostToDevice);
	err3 = cudaMemcpy(d_weights, weights_data, target_coord_nr * sizeof(double), cudaMemcpyHostToDevice);
	err4 = cudaMemcpy(d_interpolated, interpolated, grid_coord_nr * sizeof(double), cudaMemcpyHostToDevice);
// 	cout << "Copy in: " << cudaGetErrorString(err1) << " | " <<  cudaGetErrorString(err2) << " | " << cudaGetErrorString(err3) << " | " << cudaGetErrorString(err4) << endl;

 	err0 = cudaPeekAtLastError();
	if (err0 != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err0));
	}

	int threads_per_block = 64;
	int number_of_blocks = (target_coord_nr + threads_per_block - 1) / threads_per_block;
  // threads_per_block should normally be a multiple of 32 for optimal performance (at least 128 or 256)
  // number_of_blocks should be much greater than the number of SMs (Streaming Multiprocessors)
	  cout << "number_of_blocks: " << number_of_blocks << " threads_per_block: " << threads_per_block << endl;
	g_interpolate<<<number_of_blocks, threads_per_block>>>(d_target, d_coord, d_weights, d_interpolated,
		target_coord_nr, grid_coord_nr, dim2, cutoff_sq, sigma);

	cudaDeviceSynchronize();
	err0 = cudaGetLastError();
	if (err0 != cudaSuccess) {
		fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err0));
		return py::array_t<double>();
	}

  // Copy out the result and free the memory
  err0 = cudaMemcpy(interpolated, d_interpolated, grid_coord_nr * sizeof(double), cudaMemcpyDeviceToHost);
	if (err0 != cudaSuccess) {
		fprintf(stderr, "Failed to copy data from device to host (error code %s)!\n", cudaGetErrorString(err0));
    interpolated[grid_coord_nr] = {0};
	}

// 	double sum = 0.0;
//   for (int i = 0; i < grid_coord_nr; i++) {sum += interpolated[i]; }
// 	cout << endl;
// 	cout << "Sum: " << sum << endl;
	err1 = cudaFree(d_coord);
	err2 = cudaFree(d_target);
	err3 = cudaFree(d_weights);
	err4 = cudaFree(d_interpolated);
// 	cout << "Memory free: " << cudaGetErrorString(err1) << " | " <<  cudaGetErrorString(err2) << " | " << cudaGetErrorString(err3) << " | " << cudaGetErrorString(err4) << endl;
// 	cout << "###############################################" << endl;
	py::array_t<double> result({grid_coord_nr}, interpolated);
	delete[] interpolated;
	return result;
}

py::array_t<double> _interpolate(py::array_t<double> arr_grid, py::array_t<double> arr_target,
                                 py::array_t<double> arr_weights, const double cutoff = 4, const double sigma = 1.5) {
	py::buffer_info grid = arr_grid.request();
	py::buffer_info targets = arr_target.request();
	py::buffer_info weights = arr_weights.request();

	if (grid.ndim != 2 || targets.ndim != 2) {
			throw runtime_error("Number of dimensions must be two");
	}
	if (grid.shape[1] != 3 || targets.shape[1] != 3) {
			throw runtime_error("Number of columns must be three");
	}
	if (weights.shape[0] != targets.shape[0]) {
			throw runtime_error("Number of weights must be equal to number of target points");
	}

	const int n1 = targets.shape[0];
	const int n2 = grid.shape[0];
	const int d = targets.shape[1];
	const double cutoff_sq = cutoff * cutoff;

	auto data1 = static_cast<double*>(targets.ptr);
	auto data2 = static_cast<double*>(grid.ptr);
	auto _weights = static_cast<double*>(weights.ptr);

	double* interpolated = new double[n2];
	for (int idx = 0; idx < n2; idx++) { interpolated[idx] = 0.0; }

	// Data transfer to the device
	#pragma acc enter data copyin(data1[0:n1*d], data2[0:n2*d], _weights[0:n1], interpolated[0:n2])
	#pragma acc parallel loop
	for (int i = 0; i < n1; i++) {
		// Iterate over all target points
		#pragma acc loop
		for (int j = 0; j < n2; j++) {
			double dist_sq = 0.0;
			bool skip = false;
			for (int k = 0; k < d; k++) {
				double diff = data1[i * d + k] - data2[j * d + k];
				if (abs(diff) > cutoff_sq) { skip = true; break; }
				dist_sq += diff * diff;
				if (dist_sq > cutoff_sq) { skip = true; break; }
			}
			if (!skip) {
				double dist = sqrt(dist_sq);
        double increment = a_gaussian(dist, 0.0, sigma) * _weights[i];
				#pragma acc atomic update
				interpolated[j] += increment;
			}
		}
	}
	#pragma acc exit data copyout(interpolated[0:n2]) delete(data1[0:n1*d], data2[0:n2*d], _weights[0:n1])
	py::array_t<double> result({n2}, &interpolated[0]);
	delete[] interpolated;
	return result;
}

double entropy(py::array_t<double> arr_grid){
	py::buffer_info data = arr_grid.request();
  double *vec_data = static_cast<double*>(data.ptr);
	double result = _entropy<double>(vec_data, data.shape[0]);
	return result;
}

double gaussian(double distance, double mu, double sigma){
  double result = a_gaussian(distance, mu, sigma);
  return result;
}

__global__ void g_grid_entropy(double* data1, double* data2, int* atominfo, double* gridpoints,
                               int atom_nr, int gridpoint_nr, int d, double cutoff_sq) {
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

py::array_t<double> query_grid_entropy(py::array_t<double> arr_grid_coord, py::array_t<double> arr_target,
                                       py::array_t<int> arr_atominfo, const double cutoff = 8){
  py::buffer_info buffer_grid_coord = arr_grid_coord.request();
  py::buffer_info buffer_targets = arr_target.request();
  py::buffer_info buffer_atom_info = arr_atominfo.request();

  // Coordinate has 2 dimensions (M*N, 3) where M is the number of frames and N is the number of atoms
  // Grid has 2 dimensions (D^3, 3) where D is the number of grid points in each dimension
  if (buffer_grid_coord.ndim != 2 || buffer_targets.ndim != 2) {
    throw runtime_error("Number of dimensions must be three");
  }
  if (buffer_grid_coord.shape[1] != buffer_targets.shape[1]) {
		throw runtime_error("Second dimension must be three");
	}
	if (buffer_targets.shape[0] != buffer_atom_info.shape[0]) {
		throw runtime_error("Number of atoms must be the same");
	}

  auto data1 = static_cast<double *>(buffer_targets.ptr);
  auto data2 = static_cast<double *>(buffer_grid_coord.ptr);
  auto atominfo = static_cast<int *>(buffer_atom_info.ptr);

  const int atom_nr = buffer_targets.shape[0];
  const int gridpoint_nr = buffer_grid_coord.shape[0];
	const int dim2 = buffer_targets.shape[1];
	const double cutoff_sq = cutoff * cutoff;

	cout << "Atom info: ";
  for (int tmp = 0; tmp<atom_nr; ++tmp) {
		cout<< atominfo[tmp] << " ";
	}
	cout << endl;

	cudaError_t err0, err1, err2, err3, err4;

	double *grid_entropy = new double[gridpoint_nr];

  // Allocate memory on the GPU
  double *d_data1, *d_data2, *d_gridpoints;
  int *d_atominfo;
  err1 = cudaMalloc(&d_data1, atom_nr * dim2 * sizeof(double));
  err2 = cudaMalloc(&d_data2, gridpoint_nr * dim2 * sizeof(double));
  err3 = cudaMalloc(&d_atominfo, atom_nr * sizeof(int));
  err4 = cudaMalloc(&d_gridpoints, gridpoint_nr * sizeof(double));
  cout << "Allocation: " << cudaGetErrorString(err1) << " " << cudaGetErrorString(err2) << " " << cudaGetErrorString(err3) << " " << cudaGetErrorString(err4) << endl;

  // Copy data from host to device
  err1 = cudaMemcpy(d_data1, data1, atom_nr * dim2 * sizeof(double), cudaMemcpyHostToDevice);
  err2 = cudaMemcpy(d_data2, data2, gridpoint_nr * dim2 * sizeof(double), cudaMemcpyHostToDevice);
  err3 = cudaMemcpy(d_atominfo, atominfo, atom_nr * sizeof(int), cudaMemcpyHostToDevice);
  err4 = cudaMemcpy(d_gridpoints, grid_entropy, gridpoint_nr * sizeof(double), cudaMemcpyHostToDevice);
  cout << "Copy: " << cudaGetErrorString(err1) << " " << cudaGetErrorString(err2) << " " << cudaGetErrorString(err3) << " " << cudaGetErrorString(err4) << endl;

  // Launch the kernel
  int threads_per_block = 128;
  int number_of_blocks = (gridpoint_nr + threads_per_block - 1) / threads_per_block;
  g_grid_entropy<<<number_of_blocks, threads_per_block>>>(d_data1, d_data2, d_atominfo, d_gridpoints,
  																												atom_nr, gridpoint_nr, dim2, cutoff_sq);
	cudaDeviceSynchronize();
  // Copy the result back to host
//   cudaMemcpy(grid_entropy, d_gridpoints, gridpoint_nr * sizeof(double), cudaMemcpyDeviceToHost);

  // Free GPU memory

  err1 = cudaFree(d_data1);
  err2 = cudaFree(d_data2);
  err3 = cudaFree(d_atominfo);
  err4 = cudaFree(d_gridpoints);
  cout << "Memory free: " << cudaGetErrorString(err1) << ", " << cudaGetErrorString(err2) << ", "
		<< cudaGetErrorString(err3) << ", " << cudaGetErrorString(err4) << endl;

  py::array_t<double> result({gridpoint_nr}, grid_entropy);
  delete[] grid_entropy;
  return result;
}




PYBIND11_MODULE(interpolate_g, m) {
  m.def("sum_array", &sum_array, "A test function to calculate the sum of array");

  m.def("interpolate", &interpolate,
    py::arg("arr_grid"),
    py::arg("arr_target"),
    py::arg("arr_weights"),
    py::arg("cutoff") = 12, py::arg("sigma") = 1.5,
    "Compute the interpolated values of the grid points (CUDA)"
  );
  m.def("interpolate_acc", &_interpolate,
    py::arg("arr_grid"),
    py::arg("arr_target"),
    py::arg("arr_weights"),
    py::arg("cutoff") = 12, py::arg("sigma") = 1.5,
    "Compute the interpolated values of the grid points (OpenACC)"
  );
  m.def("entropy", &entropy,
    py::arg("arr_grid"),
    "Compute the entropy of the array/list"
  );
  m.def("gaussian", &gaussian,
    py::arg("distance"),
    py::arg("mu") = 0.0,
    py::arg("sigma") = 1.0,
    "Compute the value of the Gaussian function from a distance"
  );

  m.def("query_grid_entropy", &query_grid_entropy,
    py::arg("arr_grid"),
    py::arg("arr_target"),
    py::arg("arr_atominfo"),
    py::arg("cutoff") = 8,
    "Compute the entropy of the grid points"
  );

}

