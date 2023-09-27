#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


#include "icp.h"

namespace py = pybind11;
using namespace std;


Eigen::Matrix4d run_icp(py::array_t<double> pcd1, py::array_t<double> pcd2, const int maxiter = 20, const double tolerance = 0.00001){
  py::buffer_info pcd1_data = pcd1.request();
  py::buffer_info pcd2_data = pcd2.request();
  const int pcd1_number = pcd1_data.shape[0];
  const int pcd2_number = pcd2_data.shape[0];

  auto pcd1_arr = static_cast<double *>(pcd1_data.ptr);
  auto pcd2_arr = static_cast<double *>(pcd2_data.ptr);

  // Initialize the point cloud matrix sized by <N,3>
  Eigen::MatrixXd pcd1_mat(pcd1_number, 3);
  Eigen::MatrixXd pcd2_mat(pcd2_number, 3);
  for (int i; i < pcd1_number; ++i){
    pcd1_mat(i, 0) = pcd1_arr[i*3 + 0];
    pcd1_mat(i, 1) = pcd1_arr[i*3 + 1]; 
    pcd1_mat(i, 2) = pcd1_arr[i*3 + 2]; 
  }
  for (int i; i < pcd2_number; ++i){
    pcd2_mat(i, 0) = pcd2_arr[i*3 + 0]; 
    pcd2_mat(i, 1) = pcd2_arr[i*3 + 1];
    pcd2_mat(i, 2) = pcd2_arr[i*3 + 2];
  }

  // Run the icp computation
  ICP_OUT icp_result;
  icp_result = _icp(pcd1_mat, pcd2_mat, maxiter,  tolerance);

  // Report the transformation matrix and the results. 
  Eigen::Matrix4d T;
  std::vector<float> dist;
  int iter;
  float mean;

  T = icp_result.trans;
  dist = icp_result.distances;
  iter = icp_result.iter;
  mean = std::accumulate(dist.begin(),dist.end(),0.0)/dist.size();

  return T;

}

Eigen::Matrix4d guess_matrix(py::array_t<double> pcd1, py::array_t<double> pcd2, const int round_number = 4){
	// coarse_to_fine
	Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
	py::buffer_info pcd1_data = pcd1.request();
  py::buffer_info pcd2_data = pcd2.request();
  const int pcd1_number = pcd1_data.shape[0];
  const int pcd2_number = pcd2_data.shape[0];

  auto pcd1_arr = static_cast<double *>(pcd1_data.ptr);
  auto pcd2_arr = static_cast<double *>(pcd2_data.ptr);

  // Initialize the point cloud matrix sized by <N,3>
  Eigen::MatrixXd pcd1_mat(pcd1_number, 3);
  Eigen::MatrixXd pcd2_mat(pcd2_number, 3);
  for (int i; i < pcd1_number; ++i){
    pcd1_mat(i, 0) = pcd1_arr[i*3 + 0];
    pcd1_mat(i, 1) = pcd1_arr[i*3 + 1];
    pcd1_mat(i, 2) = pcd1_arr[i*3 + 2];
  }
  for (int i; i < pcd2_number; ++i){
    pcd2_mat(i, 0) = pcd2_arr[i*3 + 0];
    pcd2_mat(i, 1) = pcd2_arr[i*3 + 1];
    pcd2_mat(i, 2) = pcd2_arr[i*3 + 2];
  }
  // Run the icp computation
  ICP_OUT icp_result;
  icp_result = coarse_to_fine(pcd1_mat, pcd2_mat, round_number);
 	T = icp_result.trans;
	return T;
}



PYBIND11_MODULE(testicp, m) {
  m.def("run_icp", &run_icp,
    py::arg("pcd1"),
    py::arg("pcd2"),
    py::arg("maxiter") = 20, py::arg("tolerance") = 0.00001
  );
  m.def("guess_matrix", &guess_matrix,
    py::arg("pcd1"),
    py::arg("pcd2"),
		py::arg("round_number") = 4
  );
}

/*
Compile command:
g++ -std=c++17 -O3 -shared -fPIC -fopenmp -I$(echo ${CONDA_PREFIX}/include/python3.9) -I/MieT5/BetaPose/external/pybind11/include -I/MieT5/BetaPose/external testicp.cpp icp.cpp baseutils.cpp -o testicp$(python3-config --extension-suffix)
*/

