#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>
#include <stack>

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
  const int n1 = targets.shape[0];
  const int n2 = grid.shape[0];
  const int d = targets.shape[1];
  const double cutoff_sq = cutoff * cutoff;

  auto data1 = static_cast<double *>(targets.ptr);
  auto data2 = static_cast<double *>(grid.ptr);
  auto _weights = static_cast<double *>(weights.ptr);

	double increment, dist_sq, dist;
  bool skip;
  vector<double> interpolated(n2, 0.0);
  #pragma omp parallel for private(skip, dist_sq, dist, increment)
  for (int i = 0; i < n1; i++) {
    // Iterate over all target points
    for (int j = 0; j < n2; j++) {
      dist_sq = 0.0;
      skip = false;
      // Check if the distance is larger than cutoff
      for (int k = 0; k < d; k++) {
        double diff = data1[i*d + k] - data2[j*d + k];
        if (abs(diff) > cutoff_sq){ skip = true; break; }
        dist_sq += diff * diff;
        if (dist_sq > cutoff_sq){ skip = true; break; }
      }
      if (! skip){
        dist = sqrt(dist_sq);
        increment = Gaussian(dist, 0.0, sigma)*_weights[i];
        #pragma omp atomic
				interpolated[j] += increment;
      }
    }
  }
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

  const int atom_nr = targets.shape[0];
  const int gridpoint_nr = grid.shape[0];
	// const int atominfo_nr = targets.shape[0];
	const int d = targets.shape[1];
	const double cutoff_sq = cutoff * cutoff;

	double dist_sq;
	bool skip;
	vector<double> _gridpoints(gridpoint_nr, 0.0);
	#pragma omp parallel for private(dist_sq, skip)
	for (int i = 0; i < gridpoint_nr; i++) {
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
		if (temp_atominfo.size() == 0){ _gridpoints[i] = 0;
		} else { _gridpoints[i] = entropy(temp_atominfo);
		}
	}
	py::array_t<double> result({gridpoint_nr}, _gridpoints.data());
	return result;
}

double calc_entropy(py::array_t<double> arr_grid){
	py::buffer_info data = arr_grid.request();
	auto data1 = static_cast<double *>(data.ptr);
	vector<int> vec_data(data.shape[0], 0.0);
	for (int i = 0; i < data.shape[0]; i++){
		vec_data[i] = data1[i];
	}
	double result = entropy(vec_data);
	return result;
}
int get_index(int coord[3], int dims[3]) {
	for (int i = 0; i < 3; ++i) {
		if (coord[i] < 0 || coord[i] >= dims[i]) {
			std::cerr << "Coordinates out of bounds!" << std::endl;
			return -1;
		}
	}
	return coord[0] * dims[1] * dims[2] + coord[1] * dims[2] + coord[2];
}
void get_coord(int index, int dims[3], int coord[3]) {
	if (index < 0 || index >= dims[0] * dims[1] * dims[2]) {
		std::cerr << "Index out of bounds!" << std::endl;
		return;
	}
	coord[0] = index / (dims[1] * dims[2]); // Compute the x-coordinate
	index %= (dims[1] * dims[2]);
	coord[1] = index / dims[2]; // Compute the y-coordinate
	index %= dims[2];
	coord[2] = index; // Remaining value is the z-coordinate
}

void flood_fill(vector<int>& grid, int shape[3], int start[3]){
	std::stack<vector<int>> stack;
	vector<int> start_vec = {start[0], start[1], start[2]};
  stack.push(start_vec);
	int directions[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
	while (!stack.empty()) {
		vector<int> coord = stack.top();
		stack.pop();
		int _coord[3] = {coord[0], coord[1], coord[2]}; // Convert vector to native int array
		int index = get_index(_coord, shape);
		if (grid[index] == 0) {
			grid[index] = 2;
			for (int i = 0; i < 6; ++i) {
				int nx = coord[0] + directions[i][0];
				int ny = coord[1] + directions[i][1];
				int nz = coord[2] + directions[i][2];
				if (nx >= 0 && nx < shape[0] && ny >= 0 && ny < shape[1] && nz >= 0 && nz < shape[2]) {
					vector<int> nextCoord = {nx,ny,nz};
					stack.push(nextCoord);
				}
			}
		}
	}
}


py::array_t<double> compute_volume(py::array_t<double> vertices, double voxel_size=0.5){
	py::buffer_info vertices_data = vertices.request();
	auto v_data = static_cast<double *>(vertices_data.ptr);
	int vertice_number = vertices_data.shape[0];

	//	Get the Max and Min bounding box
	double max_bound[3] = {-9999.0, -9999.0, -9999.0};
	double min_bound[3] = {9999.0, 9999.0, 9999.0};
	for (int i = 0; i < vertice_number; i++){
		for (int j = 0; j < 3; j++){
			if (v_data[i*3 + j] > max_bound[j]){ max_bound[j] = v_data[i*3 + j]; }
			if (v_data[i*3 + j] < min_bound[j]){ min_bound[j] = v_data[i*3 + j]; }
		}
	}

	// Run flood fill algorithm to get the volume of the point cloud
	int dims[3] = {static_cast<int>(ceil((max_bound[0] - min_bound[0] + voxel_size) / voxel_size)),
		static_cast<int>(ceil((max_bound[1] - min_bound[1] + voxel_size) / voxel_size)),
		static_cast<int>(ceil((max_bound[2] - min_bound[2] + voxel_size) / voxel_size))};
	int point_nr = dims[0] * dims[1] * dims[2];
	vector<int> grid(point_nr, 0);
	for (int i = 0; i < vertice_number; i++){
		int newcoord[3] = {0,0,0};
		for (int j = 0; j < 3; j++){
			newcoord[j] = floor((v_data[i*3 + j] - min_bound[j]) / voxel_size);
		}
		int index = get_index(newcoord, dims);
    grid[index] = 1;
	}
	int start[3] = {0,0,0};
	flood_fill(grid, dims, start);

	// Obtain the results in a python array
	int c_ext = 0, c_int = 0, c_surf = 0;
	for (auto& x: grid){
		if (x == 0){ c_int += 1; }
		if (x == 1){ c_surf += 1; }
		if (x == 2){ c_ext += 1; }
	}
	double sizes[9] = {0};
	if (c_int+c_surf+c_ext != point_nr){
		cout << "Error: Sum of internal/surface/external points does not equal to total point number" << endl;
		double sizes[9] = {0};
	} else {
		sizes[0] = c_int * voxel_size * voxel_size * voxel_size;
		sizes[1] = c_surf * voxel_size * voxel_size * voxel_size;
		sizes[2] = c_ext * voxel_size * voxel_size * voxel_size;
		sizes[3] = c_int;
		sizes[4] = c_surf;
		sizes[5] = c_ext;
		sizes[6] = dims[0];
		sizes[7] = dims[1];
		sizes[8] = dims[2];
	}
	py::array_t<double> result(9, sizes);
	return result;
}

int test(){
	cout << "test" << endl;
	return 1;
}



PYBIND11_MODULE(interpolate_c, m) {
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
	m.def("compute_volume", &compute_volume,
		py::arg("grid"),
		py::arg("voxel_size") = 1.0
	);
	m.def("test", &test);
//	m.def("calc_entropy", &calc_entropy,
//		py::arg("arr_grid")
//	);
}
