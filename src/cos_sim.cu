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
  cout << "started cuda kernel" << " rows: " << rows1 << " " << rows2<< endl;
  CosineSimilarityBatch(ptr1, ptr2, result, rows1, rows2, cols);
  cout << "ended cuda kernel" << endl;

  py::array_t<float> ret_vector({rows1 * rows2}, &result[0]);

  delete[] result;
  cout << "exitted the c++ function" << endl;
  return ret_vector;
}

py::array_t<unsigned int>  cosine_similarity_query_max_index(py::array_t<float> pyarr1, py::array_t<float> pyarr2){
  py::buffer_info buf1 = pyarr1.request();
  py::buffer_info buf2 = pyarr2.request();
  unsigned int rows1 = buf1.shape[0];
  unsigned int rows2 = buf2.shape[0];
  unsigned int cols = buf1.shape[1];
  py::array_t<float> result = cosine_similarity(pyarr1, pyarr2);
  float *ptr = static_cast<float *>(result.request().ptr);

//  std::vector<float> ret_vector(result.size());
  unsigned int *ret_index = new unsigned int[rows1];
  for (size_t i = 0; i < rows1; ++i) {
    float max = -999;
    int max_index = 0;
    for (size_t j = 0; j < rows2; ++j) {
      if (ptr[i * rows2 + j] > max) {
        max_index = j;
        max = ptr[i * rows2 + j];
      }
    }
    ret_index[i] = max_index;
  }
  py::array_t<unsigned int> ret_vector({rows1}, &ret_index[0]);
  delete[] ret_index;
  return ret_vector;
}
//
//std::vector<float> cosine_similarity_query_min(py::array_t<float> pyarr1, py::array_t<float> pyarr2){
//  std::vector<std::vector<float>> result = cosine_similarity(pyarr1, pyarr2);
//  std::vector<float> ret_vector(result.size());
//  for (size_t i = 0; i < result.size(); ++i) {
//    float min = 1.0;
//    for (size_t j = 0; j < result[i].size(); ++j) {
//      if (result[i][j] < min) { min = result[i][j]; }
//    }
//    ret_vector[i] = min;
//  }
//  return ret_vector;
//}


PYBIND11_MODULE(cos_sim_g, m) {
  m.def("cosine_similarity", &cosine_similarity,
    py::arg("pyarr1"),
    py::arg("pyarr2"),
    "Compute the cosine similarity between two M*N vectors");
  m.def("cosine_similarity_query_max_index", &cosine_similarity_query_max_index,
    py::arg("pyarr1"),
    py::arg("pyarr2"),
    "Compute the cosine similarity between two M*N vectors and return the max value of each row");
//  m.def("cosine_similarity_query_min", &cosine_similarity_query_min,
//    py::arg("pyarr1"),
//    py::arg("pyarr2"),
//    "Compute the cosine similarity between two M*N vectors and return the min value of each row");

}

// Compile the code with:
// pgc++ -std=c++11 -shared -fPIC -I$(echo ${CONDA_PREFIX}/include/python3.9) -I/MieT5/BetaPose/external/pybind11/include -o cos_sim_g.so cos_sim.cu baseutils.cu && cp cos_sim_g.so ../nearl/static/ && echo "Successfully compiled"
// pgc++ -std=c++11 -shared -fPIC -I$(echo ${CONDA_PREFIX}/include/python3.9) -o cos_sim_g.so cos_sim.cu -lcudart -lcublas -lcusolver -lcusparse -lcurand -lcufft -lnvrtc -lnvrtc-bui
