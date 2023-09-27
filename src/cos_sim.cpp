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

PYBIND11_MODULE(cos_sim, m) {  // <- replace module name
  m.def("cosine_similarity", &cosine_similarity,
  py::arg("pyarr1"),
  py::arg("pyarr2"),
  "template function");   // <- replace function information
}


/*
Template command to compile the code into a shared library
g++ -std=c++17 -O3 -shared -fPIC -fopenmp -I$(echo ${CONDA_PREFIX}/include/python3.9) -I/MieT5/BetaPose/external/pybind11/include cos_sim.cpp baseutils.cpp -o cos_sim$(python3-config --extension-suffix)
*/