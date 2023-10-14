#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>
#include "baseutils.cuh"

using namespace std;
namespace py = pybind11;


// Binding the pure C++ function to python
//std::vector<std::vector<float>>
py::array_t<float> cosine_similarity(py::array_t<float> pyarr1, py::array_t<float> pyarr2){
  // Pybind11 related functions only runs on the CPU
  // Use native C++ function CosineSimilarityBatch to launch the cuda kernel
  py::buffer_info buf1 = pyarr1.request();
  py::buffer_info buf2 = pyarr2.request();
  if (buf1.ndim != buf2.ndim)
    throw std::runtime_error("Number of dimensions are not equal!");
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

PYBIND11_MODULE(cos_sim_g, m) {
  m.def("cosine_similarity", &cosine_similarity,
    py::arg("pyarr1"),
    py::arg("pyarr2"),
    "Compute the cosine similarity between two M*N vectors");
  m.def("cosine_similarity_query_max_index", &cosine_similarity_query_max_index,
    py::arg("pyarr1"),
    py::arg("pyarr2"),
    "Compute the cosine similarity between two M*N vectors and return the max value of each row (without copy back the similarity matrix)");
	m.def("cosine_similarity_query_min_index", &cosine_similarity_query_min_index,
		py::arg("pyarr1"),
		py::arg("pyarr2"),
		"Compute the cosine similarity between two M*N vectors and return the min value of each row (without copy back the similarity matrix)");

}

// Compile the code with:
// pgc++ -std=c++11 -shared -fPIC -I$(echo ${CONDA_PREFIX}/include/python3.9) -I/MieT5/BetaPose/external/pybind11/include -o cos_sim_g.so cos_sim.cu baseutils.cu && cp cos_sim_g.so ../nearl/static/ && echo "Successfully compiled"


// Test the code with:
// python -c "import time; import cos_sim_g, cos_sim; import numpy as np; np.random.seed(0); a = np.random.rand(5000, 512).astype(np.float32); b = -1*a; b=b.astype(np.float32); st = time.perf_counter(); result = np.array(cos_sim_g.cosine_similarity(a, b)); print(result.round(1), f'{time.perf_counter() - st:.3f} seconds'); st = time.perf_counter(); _result = np.array(cos_sim.cosine_similarity(a, b)); print(_result.round(1), f'{time.perf_counter() - st:.3f} seconds'); print('Have difference: ', False in np.isclose(result, _result));"
// python -c "import time; import cos_sim_g; import numpy as np; np.random.seed(1); a = np.random.rand(100000, 128).astype(np.float32); b = np.random.rand(1000, 128).astype(np.float32); st=time.perf_counter(); result = cos_sim_g.cosine_similarity_query_min_index(a, b); print(f'{result}\n{time.perf_counter()-st:.2f}, {np.sum(result)}, {np.mean(result):.3f}'); st=time.perf_counter(); result = cos_sim_g.cosine_similarity_query_max_index(a, b); print(f'{result}\n{time.perf_counter()-st:.2f}, {np.sum(result)}, {np.mean(result):.3f}');"