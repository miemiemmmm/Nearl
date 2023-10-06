#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>
#include <vector>
#include "baseutils.h"

using namespace std;
namespace py = pybind11;

std::vector<std::vector<double>> cosine_similarity(py::array_t<double> pyarr1, py::array_t<double> pyarr2){
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

	// Compute the cosine similarity
	std::vector<std::vector<double>> result = CosineSimilarityBatch<double>(vec1, vec2);
	return result;
}

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
  m.def("cosine_similarity_sum", &cosine_similarity_sum,
  py::arg("pyarr1"),
  py::arg("pyarr2"),
  "Compute the sum of cosine similarity between two M*N vectors");

	m.def("euclidean_similarity", &euclidean_similarity,
	py::arg("pyarr1"),
	py::arg("pyarr2"),
	"Compute the euclidean similarity between two M*N vectors");

	m.def("manhattan_similarity", &manhattan_similarity,
	py::arg("pyarr1"),
	py::arg("pyarr2"),
	"Compute the manhattan similarity between two M*N vectors");

}


/*
Template command to compile the code into a shared library
g++ -std=c++17 -O3 -shared -fPIC -fopenmp -I$(echo ${CONDA_PREFIX}/include/python3.9) -I/MieT5/BetaPose/external/pybind11/include cos_sim.cpp baseutils.cpp -o cos_sim$(python3-config --extension-suffix)
*/