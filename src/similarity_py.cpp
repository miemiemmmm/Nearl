#include <iostream>
#include <omp.h>
#include <stdexcept>


#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "cosine_similarity.cuh"

namespace py = pybind11;


// Binding the pure C++ function to python
//std::vector<std::vector<float>>
py::array_t<float> cosine_similarity(py::array_t<float> pyarr1, py::array_t<float> pyarr2){
  // Pybind11 related functions only runs on the CPU
  // Use native C++ function CosineSimilarityBatch to launch the cuda kernel
  py::buffer_info buf1 = pyarr1.request();
  py::buffer_info buf2 = pyarr2.request();
  if (buf1.ndim != buf2.ndim){
		throw std::runtime_error("Number of dimensions are not equal!");
	}
  unsigned int rows1 = buf1.shape[0];
  unsigned int rows2 = buf2.shape[0];
  unsigned int cols = buf1.shape[1];

  // Prepare the data for the kernel
  float *ptr1 = static_cast<float *>(buf1.ptr);
  float *ptr2 = static_cast<float *>(buf2.ptr);
  float *result = new float[rows1 * rows2];
  CosineSimilarityBatch(ptr1, ptr2, result, rows1, rows2, cols);
  //   py::array_t<float> ret_vector({rows1 * rows2}, &result[0]);
	// reshape the result to a matrix
  py::array_t<float> ret_vector({rows1, rows2}, &result[0]);
  delete[] result;
  return ret_vector;
}


py::array_t<unsigned int>  cosine_similarity_query_max_index(py::array_t<float> pyarr1, py::array_t<float> pyarr2){
  py::buffer_info buf1 = pyarr1.request();
  py::buffer_info buf2 = pyarr2.request();
  // Prepare the data for the kernel
  unsigned int rows1 = buf1.shape[0];
  unsigned int rows2 = buf2.shape[0];
  unsigned int cols = buf1.shape[1];
	float *ptr1 = static_cast<float *>(buf1.ptr);
  float *ptr2 = static_cast<float *>(buf2.ptr);
	unsigned int *ret_index = new unsigned int[rows1];
	// Launch the kernel function
  CosineSimilarityQueryMax(ptr1, ptr2, ret_index, rows1, rows2, cols);
	// Collect results
  py::array_t<unsigned int> ret_vector({rows1}, &ret_index[0]);
  delete[] ret_index;
  return ret_vector;
}

py::array_t<unsigned int>  cosine_similarity_query_min_index(py::array_t<float> pyarr1, py::array_t<float> pyarr2){
	py::buffer_info buf1 = pyarr1.request();
	py::buffer_info buf2 = pyarr2.request();
	// Prepare the data for the kernel
	unsigned int rows1 = buf1.shape[0];
	unsigned int rows2 = buf2.shape[0];
	unsigned int cols = buf1.shape[1];
	float *ptr1 = static_cast<float *>(buf1.ptr);
	float *ptr2 = static_cast<float *>(buf2.ptr);
	unsigned int *ret_index = new unsigned int[rows1];
	// Launch the kernel function
	CosineSimilarityQueryMin(ptr1, ptr2, ret_index, rows1, rows2, cols);
	// Collect results
	py::array_t<unsigned int> ret_vector({rows1}, &ret_index[0]);
	delete[] ret_index;
	return ret_vector;
}



// CPU-based cosine similarity computation. 
// std::vector<std::vector<double>> cosine_similarity(py::array_t<double> pyarr1, py::array_t<double> pyarr2){
// 	py::buffer_info buf1 = pyarr1.request();
// 	py::buffer_info buf2 = pyarr2.request();

// 	if (buf1.ndim != buf2.ndim)
// 			throw std::runtime_error("Number of dimensions are not equal!");

// 	unsigned int rows1 = buf1.shape[0];
// 	unsigned int rows2 = buf2.shape[0];
// 	unsigned int cols = buf1.shape[1];

// 	std::vector<std::vector<double>> vec1(rows1, std::vector<double>(cols));
// 	std::vector<std::vector<double>> vec2(rows2, std::vector<double>(cols));
// 	auto ptr1 = static_cast<double *>(buf1.ptr);
// 	auto ptr2 = static_cast<double *>(buf2.ptr);
// 	#pragma omp parallel for
// 	for (size_t idx = 0; idx < rows1; ++idx) {
// 		for (size_t idy = 0; idy < cols; ++idy) {
// 			vec1[idx][idy] = ptr1[idx * cols + idy];
// 		}
// 	}
// 	#pragma omp parallel for
// 	for (size_t idx = 0; idx < rows2; ++idx) {
// 		for (size_t idy = 0; idy < cols; ++idy) {
// 			vec2[idx][idy] = ptr2[idx * cols + idy];
// 		}
// 	}

// 	// Compute the cosine similarity
// 	std::vector<std::vector<double>> result = CosineSimilarityBatch<double>(vec1, vec2);
// 	return result;
// }

double cosine_similarity_sum(py::array_t<double> pyarr1, py::array_t<double> pyarr2){
  py::buffer_info buf1 = pyarr1.request();
	py::buffer_info buf2 = pyarr2.request();

	if (buf1.ndim != buf2.ndim)
			throw std::runtime_error("Number of dimensions are not equal!");

	unsigned int rows1 = buf1.shape[0];
	unsigned int rows2 = buf2.shape[0];
	unsigned int cols = buf1.shape[1];

	auto ptr1 = static_cast<double *>(buf1.ptr);
	auto ptr2 = static_cast<double *>(buf2.ptr);

  double thesum = 0.0;
  #pragma omp parallel for reduction(+:thesum)
	for (size_t idx1 = 0; idx1 < rows1; ++idx1) {
    std::vector<double> local_vec1(cols), local_vec2(cols);
    for (size_t idx2 = 0; idx2 < rows2; ++idx2) {
      for (size_t idy = 0; idy < cols; ++idy) {
        local_vec1[idy] = ptr1[idx1 * cols + idy];
        local_vec2[idy] = ptr2[idx2 * cols + idy];
      }
      double similarity = CosineSimilarity<double>(local_vec1, local_vec2);
      thesum += similarity;
    }
	}
  return thesum;
}


std::vector<std::vector<double>> euclidean_similarity(py::array_t<double> pyarr1, py::array_t<double> pyarr2){
	py::buffer_info buf1 = pyarr1.request();
	py::buffer_info buf2 = pyarr2.request();
	if (buf1.ndim != buf2.ndim)
		throw std::runtime_error("Number of dimensions are not equal!");
	unsigned int rows1 = buf1.shape[0];
	unsigned int rows2 = buf2.shape[0];
	unsigned int cols = buf1.shape[1];
	std::vector<std::vector<double>> vec1(rows1, std::vector<double>(cols));
	std::vector<std::vector<double>> vec2(rows2, std::vector<double>(cols));
	auto ptr1 = static_cast<double *>(buf1.ptr);
	auto ptr2 = static_cast<double *>(buf2.ptr);
	#pragma omp parallel for
	for (size_t idx = 0; idx < rows1; ++idx) {
		for (size_t idy = 0; idy < cols; ++idy) {
			vec1[idx][idy] = ptr1[idx * cols + idy];
		}
	}
	#pragma omp parallel for
	for (size_t idx = 0; idx < rows2; ++idx) {
		for (size_t idy = 0; idy < cols; ++idy) {
			vec2[idx][idy] = ptr2[idx * cols + idy];
		}
	}
	std::vector<std::vector<double>> result = EuclideanSimilarityBatch<double>(vec1, vec2);
	return result;
}


std::vector<std::vector<double>> manhattan_similarity(py::array_t<double> pyarr1, py::array_t<double> pyarr2){
	py::buffer_info buf1 = pyarr1.request();
	py::buffer_info buf2 = pyarr2.request();
	if (buf1.ndim != buf2.ndim)
		throw std::runtime_error("Number of dimensions are not equal!");
	unsigned int rows1 = buf1.shape[0];
	unsigned int rows2 = buf2.shape[0];
	unsigned int cols = buf1.shape[1];
	std::vector<std::vector<double>> vec1(rows1, std::vector<double>(cols));
	std::vector<std::vector<double>> vec2(rows2, std::vector<double>(cols));
	auto ptr1 = static_cast<double *>(buf1.ptr);
	auto ptr2 = static_cast<double *>(buf2.ptr);
	#pragma omp parallel for
	for (size_t idx = 0; idx < rows1; ++idx) {
		for (size_t idy = 0; idy < cols; ++idy) {
			vec1[idx][idy] = ptr1[idx * cols + idy];
		}
	}
	#pragma omp parallel for
	for (size_t idx = 0; idx < rows2; ++idx) {
		for (size_t idy = 0; idy < cols; ++idy) {
			vec2[idx][idy] = ptr2[idx * cols + idy];
		}
	}
	std::vector<std::vector<double>> result = ManhattanSimilarityBatch<double>(vec1, vec2);
	return result;
}


PYBIND11_MODULE(cos_sim, m) {  // <- replace module name
  m.def("cosine_similarity", &cosine_similarity,
  py::arg("pyarr1"),
  py::arg("pyarr2"),
  "Compute the cosine similarity between two M*N vectors");
}




PYBIND11_MODULE(cos_sim, m) {
  m.def("cosine_similarity", &cosine_similarity,
    py::arg("pyarr1"),
    py::arg("pyarr2"),
    "Compute the cosine similarity between two M*N vectors"
  );
  m.def("cosine_similarity_query_max_index", &cosine_similarity_query_max_index,
    py::arg("pyarr1"),
    py::arg("pyarr2"),
    "Compute the cosine similarity between two M*N vectors and return the max value of each row (without copy back the similarity matrix)"
  );
  m.def("cosine_similarity_query_min_index", &cosine_similarity_query_min_index,
    py::arg("pyarr1"), 
    py::arg("pyarr2"),
    "Compute the cosine similarity between two M*N vectors and return the min value of each row (without copy back the similarity matrix)"
  );
  m.def("cosine_similarity_sum", &cosine_similarity_sum,
    py::arg("pyarr1"),
    py::arg("pyarr2"),
    "Compute the sum of cosine similarity between two M*N vectors"
  );

  m.def("euclidean_similarity", &euclidean_similarity,
    py::arg("pyarr1"),
    py::arg("pyarr2"),
    "Compute the euclidean similarity between two M*N vectors"
  );

  m.def("manhattan_similarity", &manhattan_similarity,
    py::arg("pyarr1"),
    py::arg("pyarr2"),
    "Compute the manhattan similarity between two M*N vectors"
  );

}