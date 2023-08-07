#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Basic utilities such as iostream, vector are included.
#include "baseutils.h"

using namespace std;
namespace py = pybind11;

// Interpolate between two arrays of points
// arr1: (n1, 3) array of grid points
// arr2: (n2, 3) array of target points
py::array_t<double> interpolate(py::array_t<double> arr_grid, py::array_t<double> arr_target,
py::array_t<double> arr_weights, const double cutoff = 4, const double sigma = 1.5){
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

  // Get dimensions of two arrays: firstly iterate over target, then grid
  int n1 = targets.shape[0];
  int n2 = grid.shape[0];
  int d = targets.shape[1];
  double cutoff_sq = cutoff * cutoff, dist_sq, dist;
  bool skip;

  auto data1 = static_cast<double *>(targets.ptr);
  auto data2 = static_cast<double *>(grid.ptr);
  auto _weights = static_cast<double *>(weights.ptr);
  vector<double> interpolated(n2, 0.0);
  int c =0 ;
  for (int i = 0; i < n1; i++) {
    // Iterate over all target points
//    cout << "Target point" << i << ":" << data1[i*d] << " " << data1[i*d + 1] << " " << data1[i*d + 2] << endl;
    for (int j = 0; j < n2; j++) {
      dist_sq = 0.0;
      skip = false;
      // print the coordinates of the two points
//      cout << "Grid point" << j << ":" << data2[j*d] << " " << data2[j*d + 1] << " " << data2[j*d + 2] << endl;
      for (int k = 0; k < d; k++) {
        double diff = data1[i*d + k] - data2[j*d + k];
        // Check if the distance is larger than cutoff
        if (abs(diff) > cutoff_sq){ c += 1; skip = true; break; }
        dist_sq += diff * diff;
        if (dist_sq > cutoff_sq){ c += 1; skip = true; break; }
      }
      if (! skip){
        dist = sqrt(dist_sq);
//        cout << "Dist: " << dist << endl;
        double increment = gaussian(dist, 0.0, sigma)*_weights[i];
        interpolated[j] += increment;
      }
    }
//    break;  // TODO: Only for testing
  }
//  cout << "Skipped values: " << c << "/" << n1*n2 << endl;
  py::array_t<double> result({n2}, interpolated.data());
  return result;
}
py::array_t<double> grid4entropy(py::array_t<double> arr_grid, py::array_t<double> arr_target,
py::array_t<double> arr_atominfo, const double cutoff = 8){
  py::buffer_info grid = arr_grid.request();
  py::buffer_info targets = arr_target.request();
  py::buffer_info atominfo = arr_atominfo.request();

  // Coordinate has 2 dimensions (M*N, 3) where M is the number of frames and N is the number of atoms
  // Grid has 2 dimensions (D^3, 3) where D is the number of grid points in each dimension
  if (grid.ndim != 2 || targets.ndim != 2) {
    throw runtime_error("Number of dimensions must be three");
  }

  auto data1 = static_cast<double *>(targets.ptr);
  auto data2 = static_cast<double *>(grid.ptr);
  auto _atominfo = static_cast<double *>(atominfo.ptr);

  int atom_nr = targets.shape[0];
  int gridpoint_nr = grid.shape[0];
	int atominfo_nr = targets.shape[0];
	int d = targets.shape[1];
	double cutoff_sq = cutoff * cutoff, dist_sq, dist;
	bool skip;

	vector<double> _gridpoints(gridpoint_nr, 0.0);
	int c = 0;
	for (int i = 0; i < gridpoint_nr; i++) {
//		vector<int> temp_atominfo(atom_nr, 0);
		vector<int> temp_atominfo = {};
		for (int j = 0; j < atom_nr; j++){
			dist_sq = 0.0;
      skip = false;

      for (int k = 0; k < d; k++){
      	double diff = data1[j*d + k] - data2[i*d + k];
      	if (abs(diff) > cutoff_sq){ skip = true; break; }
      	dist_sq += diff * diff;
				if (dist_sq > cutoff_sq){ skip = true; break; }
      }

      if (! skip){
				temp_atominfo.push_back(static_cast<int>(_atominfo[j]));
			}
		}
		if (temp_atominfo.size() == 0){
			_gridpoints[i] = 0;
		} else {
			_gridpoints[i] = entropy(temp_atominfo);
		}
	}
	py::array_t<double> result({gridpoint_nr}, _gridpoints.data());
	return result;
}

double calc_entropy(py::array_t<double> arr_grid){
	py::buffer_info data = arr_grid.request();
	auto data1 = static_cast<double *>(data.ptr);

//	vector<int> vec_data;
	vector<int> vec_data(data.shape[0], 0.0);
	for (int i = 0; i < data.shape[0]; i++){
		vec_data[i] = data1[i];
	}
	double result = entropy(vec_data);
//	cout << result << endl;
	return result;
}


PYBIND11_MODULE(interpolate, m) {
  m.def("interpolate", &interpolate,
  	py::arg("arr_grid"),
  	py::arg("arr_target"),
  	py::arg("arr_weights"),
  	py::arg("cutoff") = 12, py::arg("sigma") = 1.5
  );
  m.def("grid4entropy", &grid4entropy,
  	py::arg("arr_grid"),
  	py::arg("arr_target"),
  	py::arg("arr_atominfo"),
  	py::arg("cutoff") = 8
	);
	m.def("calc_entropy", &calc_entropy,
		py::arg("arr_grid")
	);
}