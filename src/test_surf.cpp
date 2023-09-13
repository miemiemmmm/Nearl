#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

// Basic utilities such as iostream, vector are included.
#include "baseutils.h"

using namespace std;
namespace py = pybind11;


int get_index(int coord[3], int dims[3]) {
	for (int i = 0; i < 3; ++i) {
		if (coord[i] < 0 || coord[i] >= dims[i]) {
			std::cerr << "Coordinates out of bounds!" << std::endl;
			return -1;
		}
	}
	return coord[0] * dims[1] * dims[2] + coord[1] * dims[2] + coord[2];
}

template <typename T>
vector<vector <T>> cast_vector_2d(T *v, int rows, int cols){
	std::vector<std::vector<T>> v_casted(rows, std::vector<T>(cols));
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			v_casted[i][j] = v[i * cols + j];
		}
	}
	return v_casted;
}

template <typename T>
vector<T> cast_vector_1d(T *v, int size){
	vector<T> v_casted(size);
	for (int i = 0; i < size; ++i) {
		v_casted[i] = v[i];
	}
	return v_casted;
}

template <typename T>
vector<vector <vector <T>>> cast_vector_3d(T *v, int rows, int cols, int depth){
	std::vector<std::vector<std::vector<T>>> v_casted(rows, std::vector<std::vector<T>>(cols, std::vector<T>(depth)));
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j){
			for (int k = 0; k < depth; ++k){
				v_casted[i][j][k] = v[i * cols * depth + j * depth + k];
			}
		}
	}
	return v_casted;
}

template <typename T>
vector<T> flatten_3d_vector(const std::vector<std::vector<std::vector<T>>>& input_vector) {
	int A = input_vector.size();
	int B = (A > 0) ? input_vector[0].size() : 0;
	int C = (B > 0) ? input_vector[0][0].size() : 0;
	std::vector<T> flat_data;
	flat_data.reserve(A * B * C);
	for (const auto& sub_vector1 : input_vector)
		for (const auto& sub_vector2 : sub_vector1)
			for (const auto& elem : sub_vector2)
				flat_data.push_back(elem);
	return flat_data;
}

template <typename T>
vector<T> flatten_2d_vector(const std::vector<std::vector<T>>& input_vector) {
	int A = input_vector.size();
	int B = (A > 0) ? input_vector[0].size() : 0;
	std::vector<T> flat_data;
	flat_data.reserve(A * B);
	for (const auto& sub_vector : input_vector)
		for (const auto& elem : sub_vector)
			flat_data.push_back(elem);
	return flat_data;
}

static void _sphere_surface_distance(const std::vector<std::vector<float>>& centers, const std::vector<float>& radii,
                             const float maxrange, vector<vector<vector<float>>>& matrix){
	const int64_t center_nr = centers.size();
	const int64_t msize[3] = {static_cast<int64_t>(matrix.size()), static_cast<int64_t>(matrix[0].size()), static_cast<int64_t>(matrix[0][0].size())};
	// Iterate all of centers
	int64_t counter = 0;
	for (int64_t c = 0; c < center_nr; ++c) {
		float r = radii[c];
		if (r == 0) continue;
		// For each point cijk, and the min/max index of the grid coordinate
		std::vector<float> cijk(3);
		std::vector<int> ijk_min(3), ijk_max(3);
		for (int p = 0; p < 3; ++p) {
			float x = centers[c][p];
			cijk[p] = x;
			//			ijk_min[p] = std::clamp(static_cast<int>(std::ceil(x - (r + maxrange))), 0, static_cast<int>(msize[2 - p] - 1));
			//			ijk_max[p] = std::clamp(static_cast<int>(std::floor(x + (r + maxrange))), 0, static_cast<int>(msize[2 - p] - 1));
			ijk_min[p] = std::clamp(static_cast<int>(std::ceil(x - (r + maxrange))), 0, static_cast<int>(msize[p] - 1));
			ijk_max[p] = std::clamp(static_cast<int>(std::floor(x + (r + maxrange))), 0, static_cast<int>(msize[p] - 1));
			//			cout << "center: " << c << ", p: " << p << ", x: " << x << ", ijk_min: " << ijk_min[p] << ", ijk_max: " << ijk_max[p] << endl;
		}
		// For each dimension of the grid K, J and I, iterate all the points and get the minimum distance to sphere surface
		for (int k = ijk_min[2]; k <= ijk_max[2]; ++k) {
			float dk = (k - cijk[2]);
			// if (dk>maxrange) continue;
			for (int j = ijk_min[1]; j <= ijk_max[1]; ++j) {
				float dj = (j - cijk[1]);
				// if (dj>maxrange) continue;
				for (int i = ijk_min[0]; i <= ijk_max[0]; ++i) {
					float di = (i - cijk[0]);
					// if (di>maxrange) continue;
					// For each point, calculate the distance of each grid to the sphere surface
					float rd = std::sqrt(di*di + dj*dj + dk*dk) - r;
					if (rd < matrix[i][j][k])matrix[i][j][k] = rd;
					counter++;
				}
			}
		}
	}
	cout << "Surface total computation number: " << counter << endl;
}



py::array_t<float> sphere_surface_distance(py::array_t<float> centers, py::array_t<float> radii, float maxrange, py::array_t<float> grid_shape){
	py::buffer_info centers_info = centers.request();
	py::buffer_info radii_info = radii.request();
	py::buffer_info grid_shape_info = grid_shape.request();

	auto _centers = static_cast<float *>(centers_info.ptr);
	auto v_centers = cast_vector_2d(_centers, centers_info.shape[0], centers_info.shape[1]);
	auto _radii = static_cast<float *>(radii_info.ptr);
	auto v_radii = cast_vector_1d(_radii, radii_info.shape[0]);
	auto _grid_shape = static_cast<float *>(grid_shape_info.ptr);
	auto v_grid_shape = cast_vector_1d(_grid_shape, grid_shape_info.shape[0]);

	int point_nr = v_grid_shape[0]*v_grid_shape[1]*v_grid_shape[2];
	float _grid[point_nr] = {0};
	std::fill(_grid, _grid + point_nr, 2);
	vector<vector<vector <float>>> v_grid = cast_vector_3d(_grid, v_grid_shape[0], v_grid_shape[1], v_grid_shape[2]);

	_sphere_surface_distance(v_centers, v_radii, maxrange, v_grid);

	vector<float> flat_data(point_nr);
	flat_data = flatten_3d_vector(v_grid);

	cout << "Sum of the vector: " << std::accumulate(flat_data.begin(), flat_data.end(), 0.0) << endl;

	py::array_t<float> ret_array({v_grid_shape[0], v_grid_shape[1], v_grid_shape[2]}, flat_data.data());
	return ret_array;
}
int triangle_table[256][16] = {
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 0, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 3, 8, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, 2, 1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 9, 10, 2, 0, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 2, 3, 10, 2, 8, 9, 10, 8, -1, -1, -1, -1, -1, -1, -1},
{11, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 0, 2, 11, 8, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 1, 0, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 1, 2, 9, 1, 11, 8, 9, 11, -1, -1, -1, -1, -1, -1, -1},
{10, 3, 1, 10, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 0, 1, 8, 0, 10, 11, 8, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 3, 0, 11, 3, 9, 10, 11, 9, -1, -1, -1, -1, -1, -1, -1},
{8, 9, 10, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 4, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 4, 0, 3, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 0, 9, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 4, 9, 7, 4, 1, 3, 7, 1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 7, 0, 3, 4, 2, 1, 10, -1, -1, -1, -1, -1, -1, -1},
{2, 9, 10, 0, 9, 2, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 2, 9, 9, 2, 7, 7, 2, 3, 9, 7, 4, -1, -1, -1, -1},
{4, 8, 7, 11, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 11, 7, 2, 11, 4, 0, 2, 4, -1, -1, -1, -1, -1, -1, -1},
{0, 9, 1, 4, 8, 7, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1},
{7, 4, 11, 4, 9, 11, 11, 9, 2, 2, 9, 1, -1, -1, -1, -1},
{10, 3, 1, 11, 3, 10, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1},
{11, 1, 10, 4, 1, 11, 0, 1, 4, 11, 7, 4, -1, -1, -1, -1},
{7, 4, 8, 0, 9, 11, 11, 9, 10, 0, 11, 3, -1, -1, -1, -1},
{7, 4, 11, 11, 4, 9, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 4, 8, 0, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 4, 5, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 8, 4, 3, 8, 5, 1, 3, 5, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, 5, 9, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 8, 2, 1, 10, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1},
{2, 5, 10, 4, 5, 2, 0, 4, 2, -1, -1, -1, -1, -1, -1, -1},
{10, 2, 5, 2, 3, 5, 5, 3, 4, 4, 3, 8, -1, -1, -1, -1},
{5, 9, 4, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 0, 2, 8, 0, 11, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 4, 1, 0, 5, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 5, 5, 2, 8, 8, 2, 11, 8, 4, 5, -1, -1, -1, -1},
{3, 10, 11, 1, 10, 3, 5, 9, 4, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 5, 8, 0, 1, 10, 8, 1, 11, 8, 10, -1, -1, -1, -1},
{4, 5, 0, 0, 5, 11, 11, 5, 10, 0, 11, 3, -1, -1, -1, -1},
{4, 5, 8, 8, 5, 10, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{7, 9, 8, 7, 5, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 9, 0, 5, 9, 3, 7, 5, 3, -1, -1, -1, -1, -1, -1, -1},
{7, 0, 8, 1, 0, 7, 5, 1, 7, -1, -1, -1, -1, -1, -1, -1},
{5, 1, 3, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 9, 8, 5, 9, 7, 1, 10, 2, -1, -1, -1, -1, -1, -1, -1},
{1, 10, 2, 5, 9, 0, 3, 5, 0, 7, 5, 3, -1, -1, -1, -1},
{0, 8, 2, 2, 8, 5, 5, 8, 7, 5, 10, 2, -1, -1, -1, -1},
{10, 2, 5, 5, 2, 3, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 5, 8, 7, 9, 11, 3, 2, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 7, 7, 9, 2, 2, 9, 0, 7, 2, 11, -1, -1, -1, -1},
{3, 2, 11, 1, 0, 8, 7, 1, 8, 5, 1, 7, -1, -1, -1, -1},
{2, 11, 1, 1, 11, 7, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 8, 5, 8, 7, 1, 10, 3, 3, 10, 11, -1, -1, -1, -1},
{7, 5, 0, 0, 5, 9, 11, 7, 0, 0, 1, 10, 10, 11, 0, -1},
{10, 11, 0, 0, 11, 3, 5, 10, 0, 0, 8, 7, 7, 5, 0, -1},
{10, 11, 5, 11, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{6, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 9, 1, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 3, 9, 1, 8, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1},
{6, 1, 5, 6, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{6, 1, 5, 2, 1, 6, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1},
{6, 9, 5, 0, 9, 6, 2, 0, 6, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 8, 8, 5, 2, 2, 5, 6, 2, 3, 8, -1, -1, -1, -1},
{3, 2, 11, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 8, 2, 11, 0, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1},
{1, 0, 9, 3, 2, 11, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 5, 6, 9, 1, 2, 11, 9, 2, 8, 9, 11, -1, -1, -1, -1},
{3, 6, 11, 5, 6, 3, 1, 5, 3, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 11, 11, 0, 5, 5, 0, 1, 11, 5, 6, -1, -1, -1, -1},
{11, 3, 6, 3, 0, 6, 6, 0, 5, 5, 0, 9, -1, -1, -1, -1},
{5, 6, 9, 9, 6, 11, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{10, 5, 6, 7, 4, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 4, 0, 7, 4, 3, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 1, 0, 10, 5, 6, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1},
{6, 10, 5, 9, 1, 7, 7, 1, 3, 9, 7, 4, -1, -1, -1, -1},
{1, 6, 2, 5, 6, 1, 7, 4, 8, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 5, 2, 5, 6, 0, 3, 4, 4, 3, 7, -1, -1, -1, -1},
{4, 8, 7, 0, 9, 5, 6, 0, 5, 2, 0, 6, -1, -1, -1, -1},
{3, 7, 9, 9, 7, 4, 2, 3, 9, 9, 5, 6, 6, 2, 9, -1},
{11, 3, 2, 8, 7, 4, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1},
{10, 5, 6, 7, 4, 2, 2, 4, 0, 7, 2, 11, -1, -1, -1, -1},
{1, 0, 9, 7, 4, 8, 3, 2, 11, 10, 5, 6, -1, -1, -1, -1},
{2, 9, 1, 11, 9, 2, 4, 9, 11, 11, 7, 4, 10, 5, 6, -1},
{4, 8, 7, 11, 3, 5, 5, 3, 1, 11, 5, 6, -1, -1, -1, -1},
{1, 5, 11, 11, 5, 6, 0, 1, 11, 11, 7, 4, 4, 0, 11, -1},
{5, 0, 9, 6, 0, 5, 3, 0, 6, 6, 11, 3, 4, 8, 7, -1},
{5, 6, 9, 9, 6, 11, 7, 4, 9, 11, 7, 9, -1, -1, -1, -1},
{4, 10, 9, 4, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 4, 6, 9, 4, 10, 8, 0, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 10, 1, 6, 10, 0, 4, 6, 0, -1, -1, -1, -1, -1, -1, -1},
{3, 8, 1, 1, 8, 6, 6, 8, 4, 1, 6, 10, -1, -1, -1, -1},
{4, 1, 9, 2, 1, 4, 6, 2, 4, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 8, 2, 1, 9, 4, 2, 9, 6, 2, 4, -1, -1, -1, -1},
{2, 0, 4, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 8, 2, 2, 8, 4, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{4, 10, 9, 6, 10, 4, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 2, 8, 2, 11, 9, 4, 10, 10, 4, 6, -1, -1, -1, -1},
{11, 3, 2, 1, 0, 6, 6, 0, 4, 1, 6, 10, -1, -1, -1, -1},
{4, 6, 1, 1, 6, 10, 8, 4, 1, 1, 2, 11, 11, 8, 1, -1},
{6, 9, 4, 3, 9, 6, 1, 9, 3, 6, 11, 3, -1, -1, -1, -1},
{11, 8, 1, 1, 8, 0, 6, 11, 1, 1, 9, 4, 4, 6, 1, -1},
{11, 3, 6, 6, 3, 0, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 6, 8, 6, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 6, 8, 7, 10, 9, 8, 10, -1, -1, -1, -1, -1, -1, -1},
{7, 0, 3, 10, 0, 7, 9, 0, 10, 7, 6, 10, -1, -1, -1, -1},
{6, 10, 7, 10, 1, 7, 7, 1, 8, 8, 1, 0, -1, -1, -1, -1},
{6, 10, 7, 7, 10, 1, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 6, 6, 1, 8, 8, 1, 9, 6, 8, 7, -1, -1, -1, -1},
{6, 2, 9, 9, 2, 1, 7, 6, 9, 9, 0, 3, 3, 7, 9, -1},
{8, 7, 0, 0, 7, 6, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{3, 7, 2, 7, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 2, 11, 6, 10, 8, 8, 10, 9, 6, 8, 7, -1, -1, -1, -1},
{0, 2, 7, 7, 2, 11, 9, 0, 7, 7, 6, 10, 10, 9, 7, -1},
{8, 1, 0, 7, 1, 8, 10, 1, 7, 7, 6, 10, 3, 2, 11, -1},
{2, 11, 1, 1, 11, 7, 6, 10, 1, 7, 6, 1, -1, -1, -1, -1},
{9, 8, 6, 6, 8, 7, 1, 9, 6, 6, 11, 3, 3, 1, 6, -1},
{9, 0, 1, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 7, 0, 0, 7, 6, 11, 3, 0, 6, 11, 0, -1, -1, -1, -1},
{11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{6, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 8, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 0, 9, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 9, 3, 8, 1, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1},
{1, 10, 2, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, 0, 3, 8, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 0, 10, 2, 9, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1},
{11, 6, 7, 10, 2, 3, 8, 10, 3, 9, 10, 8, -1, -1, -1, -1},
{2, 7, 3, 2, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 8, 6, 7, 0, 2, 6, 0, -1, -1, -1, -1, -1, -1, -1},
{7, 2, 6, 3, 2, 7, 1, 0, 9, -1, -1, -1, -1, -1, -1, -1},
{6, 1, 2, 8, 1, 6, 9, 1, 8, 7, 8, 6, -1, -1, -1, -1},
{7, 10, 6, 1, 10, 7, 3, 1, 7, -1, -1, -1, -1, -1, -1, -1},
{7, 10, 6, 7, 1, 10, 8, 1, 7, 0, 1, 8, -1, -1, -1, -1},
{3, 0, 7, 7, 0, 10, 10, 0, 9, 10, 6, 7, -1, -1, -1, -1},
{6, 7, 10, 10, 7, 8, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{8, 6, 4, 8, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{6, 3, 11, 0, 3, 6, 4, 0, 6, -1, -1, -1, -1, -1, -1, -1},
{6, 8, 11, 4, 8, 6, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 6, 6, 9, 3, 3, 9, 1, 3, 11, 6, -1, -1, -1, -1},
{8, 6, 4, 11, 6, 8, 10, 2, 1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, 0, 3, 11, 6, 0, 11, 4, 0, 6, -1, -1, -1, -1},
{11, 4, 8, 6, 4, 11, 2, 0, 9, 10, 2, 9, -1, -1, -1, -1},
{9, 10, 3, 3, 10, 2, 4, 9, 3, 3, 11, 6, 6, 4, 3, -1},
{2, 8, 3, 4, 8, 2, 6, 4, 2, -1, -1, -1, -1, -1, -1, -1},
{4, 0, 2, 6, 4, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 1, 0, 3, 2, 4, 4, 2, 6, 3, 4, 8, -1, -1, -1, -1},
{9, 1, 4, 4, 1, 2, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 6, 8, 1, 4, 8, 6, 10, 6, 1, -1, -1, -1, -1},
{1, 10, 0, 0, 10, 6, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{6, 4, 3, 3, 4, 8, 10, 6, 3, 3, 0, 9, 9, 10, 3, -1},
{9, 10, 4, 10, 6, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 5, 6, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, 9, 4, 5, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 1, 4, 5, 0, 6, 7, 11, -1, -1, -1, -1, -1, -1, -1},
{7, 11, 6, 3, 8, 4, 5, 3, 4, 1, 3, 5, -1, -1, -1, -1},
{5, 9, 4, 1, 10, 2, 6, 7, 11, -1, -1, -1, -1, -1, -1, -1},
{11, 6, 7, 2, 1, 10, 8, 0, 3, 9, 4, 5, -1, -1, -1, -1},
{6, 7, 11, 4, 5, 10, 2, 4, 10, 0, 4, 2, -1, -1, -1, -1},
{4, 3, 8, 5, 3, 4, 2, 3, 5, 5, 10, 2, 7, 11, 6, -1},
{2, 7, 3, 6, 7, 2, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 4, 8, 0, 6, 6, 0, 2, 8, 6, 7, -1, -1, -1, -1},
{6, 3, 2, 7, 3, 6, 5, 1, 0, 4, 5, 0, -1, -1, -1, -1},
{2, 6, 8, 8, 6, 7, 1, 2, 8, 8, 4, 5, 5, 1, 8, -1},
{5, 9, 4, 1, 10, 6, 7, 1, 6, 3, 1, 7, -1, -1, -1, -1},
{6, 1, 10, 7, 1, 6, 0, 1, 7, 7, 8, 0, 5, 9, 4, -1},
{0, 4, 10, 10, 4, 5, 3, 0, 10, 10, 6, 7, 7, 3, 10, -1},
{6, 7, 10, 10, 7, 8, 4, 5, 10, 8, 4, 10, -1, -1, -1, -1},
{9, 6, 5, 11, 6, 9, 8, 11, 9, -1, -1, -1, -1, -1, -1, -1},
{6, 3, 11, 6, 0, 3, 5, 0, 6, 9, 0, 5, -1, -1, -1, -1},
{11, 0, 8, 5, 0, 11, 1, 0, 5, 6, 5, 11, -1, -1, -1, -1},
{11, 6, 3, 3, 6, 5, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, 5, 9, 11, 11, 9, 8, 5, 11, 6, -1, -1, -1, -1},
{11, 0, 3, 6, 0, 11, 9, 0, 6, 6, 5, 9, 2, 1, 10, -1},
{8, 11, 5, 5, 11, 6, 0, 8, 5, 5, 10, 2, 2, 0, 5, -1},
{11, 6, 3, 3, 6, 5, 10, 2, 3, 5, 10, 3, -1, -1, -1, -1},
{8, 5, 9, 2, 5, 8, 6, 5, 2, 8, 3, 2, -1, -1, -1, -1},
{5, 9, 6, 6, 9, 0, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{5, 1, 8, 8, 1, 0, 6, 5, 8, 8, 3, 2, 2, 6, 8, -1},
{5, 1, 6, 1, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 1, 6, 6, 1, 10, 8, 3, 6, 6, 5, 9, 9, 8, 6, -1},
{1, 10, 0, 0, 10, 6, 5, 9, 0, 6, 5, 0, -1, -1, -1, -1},
{3, 0, 8, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 11, 10, 5, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 11, 10, 7, 11, 5, 3, 8, 0, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 7, 10, 5, 11, 9, 1, 0, -1, -1, -1, -1, -1, -1, -1},
{7, 10, 5, 11, 10, 7, 8, 9, 1, 3, 8, 1, -1, -1, -1, -1},
{1, 11, 2, 7, 11, 1, 5, 7, 1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, 2, 1, 7, 7, 1, 5, 2, 7, 11, -1, -1, -1, -1},
{7, 9, 5, 2, 9, 7, 0, 9, 2, 11, 2, 7, -1, -1, -1, -1},
{5, 7, 2, 2, 7, 11, 9, 5, 2, 2, 3, 8, 8, 9, 2, -1},
{5, 2, 10, 3, 2, 5, 7, 3, 5, -1, -1, -1, -1, -1, -1, -1},
{2, 8, 0, 5, 8, 2, 7, 8, 5, 2, 10, 5, -1, -1, -1, -1},
{0, 9, 1, 10, 5, 3, 3, 5, 7, 10, 3, 2, -1, -1, -1, -1},
{8, 9, 2, 2, 9, 1, 7, 8, 2, 2, 10, 5, 5, 7, 2, -1},
{3, 1, 5, 7, 3, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 7, 7, 0, 1, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 9, 3, 3, 9, 5, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{8, 9, 7, 9, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 5, 4, 10, 5, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 11, 5, 0, 10, 5, 11, 3, 11, 0, -1, -1, -1, -1},
{1, 0, 9, 4, 8, 10, 10, 8, 11, 4, 10, 5, -1, -1, -1, -1},
{11, 10, 4, 4, 10, 5, 3, 11, 4, 4, 9, 1, 1, 3, 4, -1},
{5, 2, 1, 8, 2, 5, 11, 2, 8, 5, 4, 8, -1, -1, -1, -1},
{4, 0, 11, 11, 0, 3, 5, 4, 11, 11, 2, 1, 1, 5, 11, -1},
{2, 0, 5, 5, 0, 9, 11, 2, 5, 5, 4, 8, 8, 11, 5, -1},
{4, 9, 5, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 2, 10, 5, 3, 2, 4, 3, 5, 8, 3, 4, -1, -1, -1, -1},
{10, 5, 2, 2, 5, 4, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{10, 3, 2, 5, 3, 10, 8, 3, 5, 5, 4, 8, 1, 0, 9, -1},
{10, 5, 2, 2, 5, 4, 9, 1, 2, 4, 9, 2, -1, -1, -1, -1},
{4, 8, 5, 5, 8, 3, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{4, 0, 5, 0, 1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 8, 5, 5, 8, 3, 0, 9, 5, 3, 0, 5, -1, -1, -1, -1},
{4, 9, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 4, 7, 9, 4, 11, 10, 9, 11, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, 9, 4, 7, 11, 9, 7, 10, 9, 11, -1, -1, -1, -1},
{10, 1, 11, 11, 1, 4, 4, 1, 0, 4, 7, 11, -1, -1, -1, -1},
{1, 3, 4, 4, 3, 8, 10, 1, 4, 4, 7, 11, 11, 10, 4, -1},
{11, 4, 7, 11, 9, 4, 2, 9, 11, 1, 9, 2, -1, -1, -1, -1},
{7, 9, 4, 11, 9, 7, 1, 9, 11, 11, 2, 1, 8, 0, 3, -1},
{7, 11, 4, 4, 11, 2, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{7, 11, 4, 4, 11, 2, 3, 8, 4, 2, 3, 4, -1, -1, -1, -1},
{9, 2, 10, 7, 2, 9, 3, 2, 7, 4, 7, 9, -1, -1, -1, -1},
{10, 9, 7, 7, 9, 4, 2, 10, 7, 7, 8, 0, 0, 2, 7, -1},
{7, 3, 10, 10, 3, 2, 4, 7, 10, 10, 1, 0, 0, 4, 10, -1},
{10, 1, 2, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 1, 1, 4, 7, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 1, 1, 4, 7, 8, 0, 1, 7, 8, 1, -1, -1, -1, -1},
{0, 4, 3, 4, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 9, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 9, 9, 3, 11, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{1, 0, 10, 10, 0, 8, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 10, 3, 11, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 11, 11, 1, 9, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 9, 9, 3, 11, 2, 1, 9, 11, 2, 9, -1, -1, -1, -1},
{2, 0, 11, 0, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 2, 8, 8, 2, 10, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{10, 9, 2, 9, 0, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 2, 8, 8, 2, 10, 1, 0, 8, 10, 1, 8, -1, -1, -1, -1},
{10, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 1, 8, 1, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
};

int cap_triangle_table[6][16][10] = {
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 12, 3, -1, -1, -1, -1, -1, -1, -1},
{1, 13, 0, -1, -1, -1, -1, -1, -1, -1},
{13, 12, 1, 1, 12, 3, -1, -1, -1, -1},
{2, 14, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 12, 3, 2, 14, 1, -1, -1, -1, -1},
{14, 13, 2, 2, 13, 0, -1, -1, -1, -1},
{3, 13, 12, 2, 13, 3, 14, 13, 2, -1},
{3, 15, 2, -1, -1, -1, -1, -1, -1, -1},
{12, 15, 0, 0, 15, 2, -1, -1, -1, -1},
{1, 13, 0, 3, 15, 2, -1, -1, -1, -1},
{13, 12, 1, 1, 12, 2, 2, 12, 15, -1},
{15, 14, 3, 3, 14, 1, -1, -1, -1, -1},
{12, 15, 0, 0, 15, 1, 1, 15, 14, -1},
{0, 14, 13, 3, 14, 0, 15, 14, 3, -1},
{13, 12, 14, 14, 12, 15, -1, -1, -1, -1},
},
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 12, 0, -1, -1, -1, -1, -1, -1, -1},
{4, 16, 8, -1, -1, -1, -1, -1, -1, -1},
{16, 12, 4, 4, 12, 0, -1, -1, -1, -1},
{9, 17, 4, -1, -1, -1, -1, -1, -1, -1},
{8, 12, 0, 9, 17, 4, -1, -1, -1, -1},
{17, 16, 9, 9, 16, 8, -1, -1, -1, -1},
{0, 16, 12, 9, 16, 0, 17, 16, 9, -1},
{0, 13, 9, -1, -1, -1, -1, -1, -1, -1},
{12, 13, 8, 8, 13, 9, -1, -1, -1, -1},
{4, 16, 8, 0, 13, 9, -1, -1, -1, -1},
{16, 12, 4, 4, 12, 9, 9, 12, 13, -1},
{13, 17, 0, 0, 17, 4, -1, -1, -1, -1},
{12, 13, 8, 8, 13, 4, 4, 13, 17, -1},
{8, 17, 16, 0, 17, 8, 13, 17, 0, -1},
{16, 12, 17, 17, 12, 13, -1, -1, -1, -1},
},
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 13, 1, -1, -1, -1, -1, -1, -1, -1},
{5, 17, 9, -1, -1, -1, -1, -1, -1, -1},
{17, 13, 5, 5, 13, 1, -1, -1, -1, -1},
{10, 18, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 13, 1, 10, 18, 5, -1, -1, -1, -1},
{18, 17, 10, 10, 17, 9, -1, -1, -1, -1},
{1, 17, 13, 10, 17, 1, 18, 17, 10, -1},
{1, 14, 10, -1, -1, -1, -1, -1, -1, -1},
{13, 14, 9, 9, 14, 10, -1, -1, -1, -1},
{5, 17, 9, 1, 14, 10, -1, -1, -1, -1},
{17, 13, 5, 5, 13, 10, 10, 13, 14, -1},
{14, 18, 1, 1, 18, 5, -1, -1, -1, -1},
{13, 14, 9, 9, 14, 5, 5, 14, 18, -1},
{9, 18, 17, 1, 18, 9, 14, 18, 1, -1},
{17, 13, 18, 18, 13, 14, -1, -1, -1, -1},
},
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 14, 2, -1, -1, -1, -1, -1, -1, -1},
{6, 18, 10, -1, -1, -1, -1, -1, -1, -1},
{18, 14, 6, 6, 14, 2, -1, -1, -1, -1},
{11, 19, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 14, 2, 11, 19, 6, -1, -1, -1, -1},
{19, 18, 11, 11, 18, 10, -1, -1, -1, -1},
{2, 18, 14, 11, 18, 2, 19, 18, 11, -1},
{2, 15, 11, -1, -1, -1, -1, -1, -1, -1},
{14, 15, 10, 10, 15, 11, -1, -1, -1, -1},
{6, 18, 10, 2, 15, 11, -1, -1, -1, -1},
{18, 14, 6, 6, 14, 11, 11, 14, 15, -1},
{15, 19, 2, 2, 19, 6, -1, -1, -1, -1},
{14, 15, 10, 10, 15, 6, 6, 15, 19, -1},
{10, 19, 18, 2, 19, 10, 15, 19, 2, -1},
{18, 14, 19, 19, 14, 15, -1, -1, -1, -1},
},
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 15, 3, -1, -1, -1, -1, -1, -1, -1},
{7, 19, 11, -1, -1, -1, -1, -1, -1, -1},
{19, 15, 7, 7, 15, 3, -1, -1, -1, -1},
{8, 16, 7, -1, -1, -1, -1, -1, -1, -1},
{11, 15, 3, 8, 16, 7, -1, -1, -1, -1},
{16, 19, 8, 8, 19, 11, -1, -1, -1, -1},
{3, 19, 15, 8, 19, 3, 16, 19, 8, -1},
{3, 12, 8, -1, -1, -1, -1, -1, -1, -1},
{15, 12, 11, 11, 12, 8, -1, -1, -1, -1},
{7, 19, 11, 3, 12, 8, -1, -1, -1, -1},
{19, 15, 7, 7, 15, 8, 8, 15, 12, -1},
{12, 16, 3, 3, 16, 7, -1, -1, -1, -1},
{15, 12, 11, 11, 12, 7, 7, 12, 16, -1},
{11, 16, 19, 3, 16, 11, 12, 16, 3, -1},
{19, 15, 16, 16, 15, 12, -1, -1, -1, -1},
},
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 16, 4, -1, -1, -1, -1, -1, -1, -1},
{6, 19, 7, -1, -1, -1, -1, -1, -1, -1},
{19, 16, 6, 6, 16, 4, -1, -1, -1, -1},
{5, 18, 6, -1, -1, -1, -1, -1, -1, -1},
{7, 16, 4, 5, 18, 6, -1, -1, -1, -1},
{18, 19, 5, 5, 19, 7, -1, -1, -1, -1},
{4, 19, 16, 5, 19, 4, 18, 19, 5, -1},
{4, 17, 5, -1, -1, -1, -1, -1, -1, -1},
{16, 17, 7, 7, 17, 5, -1, -1, -1, -1},
{6, 19, 7, 4, 17, 5, -1, -1, -1, -1},
{19, 16, 6, 6, 16, 5, 5, 16, 17, -1},
{17, 18, 4, 4, 18, 6, -1, -1, -1, -1},
{16, 17, 7, 7, 17, 6, 6, 17, 18, -1},
{7, 18, 19, 4, 18, 7, 17, 18, 4, -1},
{19, 16, 18, 18, 16, 17, -1, -1, -1, -1},
},
};

int face_corner_bits[6][256] = {
{
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
},
{
0,1,8,9,0,1,8,9,0,1,8,9,0,1,8,9,
2,3,10,11,2,3,10,11,2,3,10,11,2,3,10,11,
4,5,12,13,4,5,12,13,4,5,12,13,4,5,12,13,
6,7,14,15,6,7,14,15,6,7,14,15,6,7,14,15,
0,1,8,9,0,1,8,9,0,1,8,9,0,1,8,9,
2,3,10,11,2,3,10,11,2,3,10,11,2,3,10,11,
4,5,12,13,4,5,12,13,4,5,12,13,4,5,12,13,
6,7,14,15,6,7,14,15,6,7,14,15,6,7,14,15,
0,1,8,9,0,1,8,9,0,1,8,9,0,1,8,9,
2,3,10,11,2,3,10,11,2,3,10,11,2,3,10,11,
4,5,12,13,4,5,12,13,4,5,12,13,4,5,12,13,
6,7,14,15,6,7,14,15,6,7,14,15,6,7,14,15,
0,1,8,9,0,1,8,9,0,1,8,9,0,1,8,9,
2,3,10,11,2,3,10,11,2,3,10,11,2,3,10,11,
4,5,12,13,4,5,12,13,4,5,12,13,4,5,12,13,
6,7,14,15,6,7,14,15,6,7,14,15,6,7,14,15
},
{
0,0,1,1,8,8,9,9,0,0,1,1,8,8,9,9,
0,0,1,1,8,8,9,9,0,0,1,1,8,8,9,9,
2,2,3,3,10,10,11,11,2,2,3,3,10,10,11,11,
2,2,3,3,10,10,11,11,2,2,3,3,10,10,11,11,
4,4,5,5,12,12,13,13,4,4,5,5,12,12,13,13,
4,4,5,5,12,12,13,13,4,4,5,5,12,12,13,13,
6,6,7,7,14,14,15,15,6,6,7,7,14,14,15,15,
6,6,7,7,14,14,15,15,6,6,7,7,14,14,15,15,
0,0,1,1,8,8,9,9,0,0,1,1,8,8,9,9,
0,0,1,1,8,8,9,9,0,0,1,1,8,8,9,9,
2,2,3,3,10,10,11,11,2,2,3,3,10,10,11,11,
2,2,3,3,10,10,11,11,2,2,3,3,10,10,11,11,
4,4,5,5,12,12,13,13,4,4,5,5,12,12,13,13,
4,4,5,5,12,12,13,13,4,4,5,5,12,12,13,13,
6,6,7,7,14,14,15,15,6,6,7,7,14,14,15,15,
6,6,7,7,14,14,15,15,6,6,7,7,14,14,15,15
},
{
0,0,0,0,1,1,1,1,8,8,8,8,9,9,9,9,
0,0,0,0,1,1,1,1,8,8,8,8,9,9,9,9,
0,0,0,0,1,1,1,1,8,8,8,8,9,9,9,9,
0,0,0,0,1,1,1,1,8,8,8,8,9,9,9,9,
2,2,2,2,3,3,3,3,10,10,10,10,11,11,11,11,
2,2,2,2,3,3,3,3,10,10,10,10,11,11,11,11,
2,2,2,2,3,3,3,3,10,10,10,10,11,11,11,11,
2,2,2,2,3,3,3,3,10,10,10,10,11,11,11,11,
4,4,4,4,5,5,5,5,12,12,12,12,13,13,13,13,
4,4,4,4,5,5,5,5,12,12,12,12,13,13,13,13,
4,4,4,4,5,5,5,5,12,12,12,12,13,13,13,13,
4,4,4,4,5,5,5,5,12,12,12,12,13,13,13,13,
6,6,6,6,7,7,7,7,14,14,14,14,15,15,15,15,
6,6,6,6,7,7,7,7,14,14,14,14,15,15,15,15,
6,6,6,6,7,7,7,7,14,14,14,14,15,15,15,15,
6,6,6,6,7,7,7,7,14,14,14,14,15,15,15,15
},
{
0,8,0,8,0,8,0,8,1,9,1,9,1,9,1,9,
4,12,4,12,4,12,4,12,5,13,5,13,5,13,5,13,
0,8,0,8,0,8,0,8,1,9,1,9,1,9,1,9,
4,12,4,12,4,12,4,12,5,13,5,13,5,13,5,13,
0,8,0,8,0,8,0,8,1,9,1,9,1,9,1,9,
4,12,4,12,4,12,4,12,5,13,5,13,5,13,5,13,
0,8,0,8,0,8,0,8,1,9,1,9,1,9,1,9,
4,12,4,12,4,12,4,12,5,13,5,13,5,13,5,13,
2,10,2,10,2,10,2,10,3,11,3,11,3,11,3,11,
6,14,6,14,6,14,6,14,7,15,7,15,7,15,7,15,
2,10,2,10,2,10,2,10,3,11,3,11,3,11,3,11,
6,14,6,14,6,14,6,14,7,15,7,15,7,15,7,15,
2,10,2,10,2,10,2,10,3,11,3,11,3,11,3,11,
6,14,6,14,6,14,6,14,7,15,7,15,7,15,7,15,
2,10,2,10,2,10,2,10,3,11,3,11,3,11,3,11,
6,14,6,14,6,14,6,14,7,15,7,15,7,15,7,15
},
{
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,
13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,
11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,
6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,
15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
}
};

#define EDGE_A00 0
#define EDGE_1A0 1
#define EDGE_A10 2
#define EDGE_0A0 3
#define EDGE_A01 4
#define EDGE_1A1 5
#define EDGE_A11 6
#define EDGE_0A1 7
#define EDGE_00A 8
#define EDGE_10A 9
#define EDGE_11A 10
#define EDGE_01A 11

#define CORNER_000 0
#define CORNER_100 1
#define CORNER_110 2
#define CORNER_010 3
#define CORNER_001 4
#define CORNER_101 5
#define CORNER_111 6
#define CORNER_011 7

#define CONTOUR_ARRAY_BLOCK_SIZE 1048576

const int no_vertex = ~(int)0;

class Grid_Cell
{
public:
  int k0, k1;	// Cell position in xy plane.
  int vertex[20];	// Vertex numbers for 12 edges and 8 corners.
  bool boundary;	// Contour reaches boundary.
};

class Grid_Cell_List
{
public:
  Grid_Cell_List(int size0, int size1)
  	: cells(CONTOUR_ARRAY_BLOCK_SIZE)
  {
    this->cell_table_size0 = size0+2;	// Pad by one grid cell.
    int cell_table_size1 = size1+2;
    int size = cell_table_size0 * cell_table_size1;
    this->cell_count = 0;
    this->cell_base_index = 2;
    this->cell_table = new int64_t[size];
    for (int i = 0 ; i < size ; ++i)
      cell_table[i] = no_cell;
    for (int i = 0 ; i < cell_table_size0 ; ++i)
      cell_table[i] = cell_table[size-i-1] = out_of_bounds;
    for (int i = 0 ; i < size ; i += cell_table_size0)
      cell_table[i] = cell_table[i+cell_table_size0-1] = out_of_bounds;
  }

  ~Grid_Cell_List(){
		delete_cells();
		delete [] cell_table;
	}

  void set_edge_vertex(int k0, int k1, int e, int v){
    Grid_Cell *c = cell(k0,k1);
    if (c){
      c->vertex[e] = v;
		}
  }
  void set_corner_vertex(int k0, int k1, int corner, int v){
    Grid_Cell *c = cell(k0,k1);
    if (c){
			c->vertex[12+corner] = v;
			c->boundary = true;
		}
  }
  void finished_plane(){
		cell_base_index += cell_count;
		cell_count = 0;
	}

  int cell_count;		// Number of elements of cells currently in use.
  vector<Grid_Cell *> cells;

private:
  static const int64_t out_of_bounds = 0;
  static const int64_t no_cell = 1;
  int cell_table_size0;
  int64_t cell_base_index;	// Minimum valid cell index.
  int64_t *cell_table;		// Maps cell plane index to cell list index.

  // Get cell, initializing or allocating a new one if necessary.
  Grid_Cell *cell(int k0, int k1){
    int64_t i = k0+1 + (k1+1)*cell_table_size0;
    int64_t c = cell_table[i];
    if (c == out_of_bounds)
      return NULL;

    Grid_Cell *cp;
    if (c != no_cell && c >= cell_base_index){
			cp = cells[c-cell_base_index];
		} else {
			cell_table[i] = cell_base_index + cell_count;
			if (cell_count < cells.size()){
				cp = cells[cell_count];
			}else{
				cells.push_back(cp = new Grid_Cell);
			}
			cp->k0 = k0; cp->k1 = k1; cp->boundary = false;
			cell_count += 1;
		}
    return cp;
  }

  void delete_cells()
  {
    int64_t cc = cells.size();
    for (int64_t c = 0 ; c < cc ; ++c)
      delete cells[c];
  }
};



template <typename T>
class Contour_Surface
{
 public:
 	Contour_Surface(const vector<vector<vector<T>>> *grid,
					 float threshold, bool cap_faces, int64_t block_size)
			: grid(grid),        // Initialize grid member with the grid parameter
				threshold(threshold),  // Initialize threshold member with the threshold parameter
				cap_faces(cap_faces)  // Initialize cap_faces member with the cap_faces parameter
//				vxyz(3 * block_size),     // Initialize vxyz member with 3 times the block_size parameter
//				tvi(3 * block_size)       // Initialize tvi member with 3 times the block_size parameter
	{
		this->size[0] = grid[0].size();
		this->size[1] = grid[0][0].size();
		this->size[2] = grid[0][0][0].size();
		cout << "Init_test: Size of the grid " << this->size[0] << " " << this->size[1] << " " << this->size[2] << endl;
		cout << "Init_test: The size of the two initial arrays: " << vxyz.size() << " " << tvi.size() << endl;
		vxyz.reserve(3 * block_size);
		tvi.reserve(3 * block_size);

		for (auto &i : {1,2,3,4,5,6,7,8,9,10}){
			tvi.push_back(15);
			cout << "Init_test: test " << tvi[0] << " size " << tvi.size() << " capacity " << tvi.capacity() << endl;
		}

		for (auto &i : {1,2,3,4,5,6,7,8,9,10}){
			tvi.pop_back();
			cout << "Init_test: End of the test " << tvi[0] << " " << tvi.size() << " capacity " << tvi.capacity() << endl;
		}

	}

  ~Contour_Surface() {};

	void compute_surface(){
		// Initialize the grid cell list
		Grid_Cell_List gcp0(size[0]-1, size[1]-1);
		Grid_Cell_List gcp1(size[0]-1, size[1]-1);
		// Iterate each of the three dimensions from 0-2
    for (int k2 = 0; k2 < size[2]; ++k2){
			// Double buffering to avoid copying and communication of data ????
			auto& gp0 = (k2 % 2) ? gcp1 : gcp0;
			auto& gp1 = (k2 % 2) ? gcp0 : gcp1;
			cout << "Before computing " << k2 << endl;
			mark_plane_edge_cuts(gp0, gp1, k2);
			// TODO continue debugging the rest of the code.
			cout << "after computing" << endl;
//			if (k2 > 0){
//				make_triangles(gp0, k2);
//			}
//			gp0.finished_plane();
    }
	}

	const int64_t vertex_count() {
		return vxyz.size();
	}

	const int64_t triangle_count(){
		return tvi.size();
	}

	void geometry(vector<float>* vertex_xyz, vector<int64_t>* triangle_vertex_indices) {
		// Check what are vxyz, tvi
		// Give vxyz value to vertex_xyz and tvi to triangle_vertex_indices
		//		vxyz.array(vertex_xyz);   // TODO check what this function does
		//		tvi.array(triangle_vertex_indices);
		for (int i = 0; i < vxyz.size(); i++){
			vertex_xyz[i] = vxyz[i];
		}
		for (int i = 0; i < tvi.size(); i++){
			triangle_vertex_indices[i] = tvi[i];
		}
	}

	void normals(vector<float>* normals) {
		int64_t nVertices = vxyz.size();
		//		normals.resize(nVertices, std::vector<float>(3, 0.0f));
		for (int64_t vertex_idx = 0; vertex_idx < nVertices; ++vertex_idx) {
			float x[3] = {vxyz[vertex_idx*3+0], vxyz[vertex_idx*3+1], vxyz[vertex_idx*3+2]};  // Vertex position
			float g[3] = {0.0f, 0.0f, 0.0f};  // Gradient vector

			if (g[0] == 0 && g[1] == 0 && g[2] == 0) {
				// Here, you can compute g based on your specific use case.

				// Cast the vertice position to int64_t to get the grid index
				int64_t i[3] = {static_cast<int64_t>(x[0]), static_cast<int64_t>(x[1]), static_cast<int64_t>(x[2])};

				// Get the grid value at the index
				T ga = grid[i[0]][i[1]][i[2]];
				T gb = ga;

				int64_t off[3] = {0, 0, 0};
				float fb = 0;

				// fractional part of x
				for (int a = 0; a < 3; ++a) {
					// If fb (fractional part) is not on the grid, then update off and gb
					// offset + 1 means in that dimension, update the grid index by 1
					// meet one of the 3 dimensions, then break
					if ((fb = x[a] - static_cast<float>(i[a])) > 0) {
						off[a] = 1;
						gb = grid[i[0] + off[0]][i[1] + off[1]][i[2] + off[2]]; // update gb based on off
						break;
					}
				}

				float fa = 1 - fb;

				for (int a = 0; a < 3; ++a) {
					int64_t ia = i[a], ib = ia + off[a];
					T ga_val = (ia == 0) ? 2 * (ga - grid[ia + 1][i[1]][i[2]]) : ga - grid[ia - 1][i[1]][i[2]];
					T gb_val = (ib == 0) ? 2 * (gb - grid[ib + 1][i[1]][i[2]]) : gb - grid[ib - 1][i[1]][i[2]];
					if (ib == size[a] - 1) {
						gb_val = 2 * (grid[ib][i[1]][i[2]] - grid[ib - 1][i[1]][i[2]]);
					}
					g[a] = fa * ga_val + fb * gb_val;
				}

				float norm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
				if (norm > 0) {
					g[0] /= norm;
					g[1] /= norm;
					g[2] /= norm;
				}
			}

			normals[vertex_idx][0] = -g[0];
			normals[vertex_idx][1] = -g[1];
			normals[vertex_idx][2] = -g[2];
		}
	}

private:
	float threshold;
	bool cap_faces;
	int64_t block_size;

	const vector<vector<vector<T>>> *grid;
	vector<int64_t> size = vector<int64_t>(3);

	vector<float> vxyz;
	vector<int64_t> tvi;

	// The primary function to compute the surface
	// 1. Firstly mark the edge cuts
	// 2. Then make triangles


	void add_triangle_corner(int64_t v) {
//		vector<int64_t> vec = {v[0], v[1], v[2]};
		tvi.push_back(v);
	}   // TODO: Check this function

	int64_t create_vertex(float x, float y, float z){
//		vector<float> new_vertex{x,y,z};
		vxyz.push_back(x);
		vxyz.push_back(y);
		vxyz.push_back(z);
		return vertex_count()-1;
	}

	void make_cap_triangles(int face, int bits, int *cell_vertices){
		int fbits = face_corner_bits[face][bits];
		int *t = cap_triangle_table[face][fbits];
		for (int v = *t ; v != -1 ; ++t, v = *t){
			add_triangle_corner(cell_vertices[v]);
		}
	}

	//	????????????????????????

	void mark_plane_edge_cuts(Grid_Cell_List &gp0, Grid_Cell_List &gp1, int k2){
		int64_t k0_size = size[0], k1_size = size[1], k2_size = size[2];
		for (int k1 = 0; k1 < k1_size; ++k1) {
			if (k1 == 0 || k1 + 1 == k1_size || k2 == 0 || k2 + 1 == k2_size) {
				// Special treatment when the box in located at the boundary of the grid.
				for (int k0 = 0; k0 < k0_size; ++k0) {
					mark_boundary_edge_cuts(k0, k1, k2, gp0, gp1);
				}
			} else {
				if (k0_size > 0) {
					mark_boundary_edge_cuts(0, k1, k2, gp0, gp1);
				}
				mark_interior_edge_cuts(k1, k2, gp0, gp1);
				if (k0_size > 1) {
					mark_boundary_edge_cuts(k0_size - 1, k1, k2, gp0, gp1);
				}
			}
  	}
	}

	void mark_boundary_edge_cuts(int k0, int k1, int k2, Grid_Cell_List& gp0, Grid_Cell_List& gp1){
		// Get the dimensions of the 3D grid.
		int k0_size = size[0], k1_size = size[1], k2_size = size[2];
		// Calculate the 1D index for this 3D point in the grid.
		// Compute the value at the current point, shifted by the threshold.
		auto gridptr = *grid;
		float v0 = gridptr[k0][k1][k2] - threshold;
		// If the value is negative, there is no crossing, so exit.
		if (v0 < 0)
			return;
		// Initialize boundary vertex index to a value indicating 'no vertex'.
		int bv = no_vertex;
		// Check 6 neighboring vertices for edge crossings.
		float v1;

		// Axis 0 left (Left along x-axis)
		if (k0 > 0) {
			v1 = static_cast<float>((gridptr)[k0][k1][k2-1] - threshold);
			if (v1 < 0)
				add_vertex_axis_0(k0 - 1, k1, k2, k0 - v0 / (v0 - v1), gp0, gp1);
		} else if (cap_faces) { // boundary vertex for capping box faces.
			bv = add_cap_vertex_l0(bv, k0, k1, k2, gp0, gp1);
		}
//
		// Axis 0 right (Right along x-axis)
		if (k0 + 1 < k0_size) {
			v1 = static_cast<float>(gridptr[k0][k1][k2+1] - threshold);
			if (v1 < 0)
				add_vertex_axis_0(k0, k1, k2, k0 + v0 / (v0 - v1), gp0, gp1);
		} else if (cap_faces) {
			bv = add_cap_vertex_r0(bv, k0, k1, k2, gp0, gp1);
		}
//
		// Axis 1 left (Down along y-axis)
		if (k1 > 0) {
			v1 = static_cast<float>(gridptr[k0][k1-1][k2] - threshold);
			if (v1 < 0)
				add_vertex_axis_1(k0, k1 - 1, k2, k1 - v0 / (v0 - v1), gp0, gp1);
		} else if (cap_faces) {
			bv = add_cap_vertex_l1(bv, k0, k1, k2, gp0, gp1);
		}
//
		// Axis 1 right (Up along y-axis)
		if (k1 + 1 < k1_size) {
			v1 = static_cast<float>(gridptr[k0][k1+1][k2] - threshold);
			if (v1 < 0)
				add_vertex_axis_1(k0, k1, k2, k1 + v0 / (v0 - v1), gp0, gp1);
		} else if (cap_faces) {
			bv = add_cap_vertex_r1(bv, k0, k1, k2, gp0, gp1);
		}
//
		// Axis 2 left (Back along z-axis)
		if (k2 > 0) {
			v1 = static_cast<float>(gridptr[k0-1][k1][k2] - threshold);
			if (v1 < 0)
				add_vertex_axis_2(k0, k1, k2 - v0 / (v0 - v1), gp0);
		} else if (cap_faces) {
			bv = add_cap_vertex_l2(bv, k0, k1, k2, gp1);
		}
//
//		// Axis 2 right (Forward along z-axis)
//		if (k2 + 1 < k2_size) {
//			v1 = (float)(gridptr[k0+1][k1][k2] - threshold);
//			if (v1 < 0)
//				add_vertex_axis_2(k0, k1, k2 + v0 / (v0 - v1), gp1);
//		} else if (cap_faces) {
//			bv = add_cap_vertex_r2(bv, k0, k1, k2, gp0);
//		}
	}

	void mark_interior_edge_cuts(int k1, int k2, Grid_Cell_List &gp0, Grid_Cell_List &gp1){
		int k0_max = (size[0] > 0 ? size[0] - 1 : 0);
    for (int k0 = 1; k0 < k0_max; ++k0) {
			T v0 = (*grid)[k2][k1][k0] - threshold;

			if (!(v0 < static_cast<T>(0))) {
				// Grid point value is above threshold.
				// Look at 6 neighbors along x, y, z axes for values below threshold.
				T v1;
				if ((v1 = (*grid)[k2][k1][k0 - 1] - threshold) < 0)
					add_vertex_axis_0(k0 - 1, k1, k2, k0 - v0 / (v0 - v1), gp0, gp1);

				if ((v1 = (*grid)[k2][k1][k0 + 1] - threshold) < 0)
					add_vertex_axis_0(k0, k1, k2, k0 + v0 / (v0 - v1), gp0, gp1);

				if ((v1 = (*grid)[k2][k1 - 1][k0] - threshold) < 0)
					add_vertex_axis_1(k0, k1 - 1, k2, k1 - v0 / (v0 - v1), gp0, gp1);

				if ((v1 = (*grid)[k2][k1 + 1][k0] - threshold) < 0)
					add_vertex_axis_1(k0, k1, k2, k1 + v0 / (v0 - v1), gp0, gp1);

				if ((v1 = (*grid)[k2 - 1][k1][k0] - threshold) < 0)
					add_vertex_axis_2(k0, k1, k2 - v0 / (v0 - v1), gp0);

				if ((v1 = (*grid)[k2 + 1][k1][k0] - threshold) < 0)
					add_vertex_axis_2(k0, k1, k2 + v0 / (v0 - v1), gp1);
			}
    }
	}

	void add_vertex_axis_0(int k0, int k1, int k2, float x0, Grid_Cell_List &gp0, Grid_Cell_List &gp1){
	  int64_t v = create_vertex(x0,k1,k2);
		gp0.set_edge_vertex(k0, k1-1, EDGE_A11, v);
		gp0.set_edge_vertex(k0, k1, EDGE_A01, v);
		gp1.set_edge_vertex(k0, k1-1, EDGE_A10, v);
		gp1.set_edge_vertex(k0, k1, EDGE_A00, v);
	}
	void add_vertex_axis_1(int k0, int k1, int k2, float x1, Grid_Cell_List &gp0, Grid_Cell_List &gp1){
	  int64_t v = create_vertex(k0,x1,k2);
	  gp0.set_edge_vertex(k0-1, k1, EDGE_1A1, v);
		gp0.set_edge_vertex(k0, k1, EDGE_0A1, v);
		gp1.set_edge_vertex(k0-1, k1, EDGE_1A0, v);
		gp1.set_edge_vertex(k0, k1, EDGE_0A0, v);
	}
	void add_vertex_axis_2(int k0, int k1, float x2, Grid_Cell_List &gp){
		int64_t v = create_vertex(k0,k1,x2);
		gp.set_edge_vertex(k0, k1, EDGE_00A, v);
		gp.set_edge_vertex(k0-1, k1, EDGE_10A, v);
		gp.set_edge_vertex(k0, k1-1, EDGE_01A, v);
		gp.set_edge_vertex(k0-1, k1-1, EDGE_11A, v);
	}
	int add_cap_vertex_l0(int bv, int k0, int k1, int k2, Grid_Cell_List &gp0, Grid_Cell_List &gp1){
		if (bv == no_vertex)
    	bv = create_vertex(k0,k1,k2);
		gp0.set_corner_vertex(k0, k1-1, CORNER_011, bv);
		gp0.set_corner_vertex(k0, k1, CORNER_001, bv);
		gp1.set_corner_vertex(k0, k1-1, CORNER_010, bv);
		gp1.set_corner_vertex(k0, k1, CORNER_000, bv);
		return bv;
	}
	int add_cap_vertex_r0(int bv, int k0, int k1, int k2, Grid_Cell_List &gp0, Grid_Cell_List &gp1){
		if (bv == no_vertex)
			bv = create_vertex(k0,k1,k2);
		gp0.set_corner_vertex(k0-1, k1-1, CORNER_111, bv);
		gp0.set_corner_vertex(k0-1, k1, CORNER_101, bv);
		gp1.set_corner_vertex(k0-1, k1-1, CORNER_110, bv);
		gp1.set_corner_vertex(k0-1, k1, CORNER_100, bv);
		return bv;
	}
	int add_cap_vertex_l1(int bv, int k0, int k1, int k2, Grid_Cell_List &gp0, Grid_Cell_List &gp1){
		if (bv == no_vertex)
			bv = create_vertex(k0,k1,k2);
		gp0.set_corner_vertex(k0-1, k1, CORNER_101, bv);
		gp0.set_corner_vertex(k0, k1, CORNER_001, bv);
		gp1.set_corner_vertex(k0-1, k1, CORNER_100, bv);
		gp1.set_corner_vertex(k0, k1, CORNER_000, bv);
		return bv;
	}
	int add_cap_vertex_r1(int bv, int k0, int k1, int k2, Grid_Cell_List &gp0, Grid_Cell_List &gp1){
		if (bv == no_vertex)
			bv = create_vertex(k0,k1,k2);
		gp0.set_corner_vertex(k0-1, k1-1, CORNER_111, bv);
		gp0.set_corner_vertex(k0, k1-1, CORNER_011, bv);
		gp1.set_corner_vertex(k0-1, k1-1, CORNER_110, bv);
		gp1.set_corner_vertex(k0, k1-1, CORNER_010, bv);
		return bv;
	}
	int add_cap_vertex_l2(int bv, int k0, int k1, int k2, Grid_Cell_List &gp1){
		if (bv == no_vertex)
			bv = create_vertex(k0,k1,k2);
		gp1.set_corner_vertex(k0-1, k1-1, CORNER_110, bv);
		gp1.set_corner_vertex(k0-1, k1, CORNER_100, bv);
		gp1.set_corner_vertex(k0, k1-1, CORNER_010, bv);
		gp1.set_corner_vertex(k0, k1, CORNER_000, bv);
		return bv;
	}
	int add_cap_vertex_r2(int bv, int k0, int k1, int k2, Grid_Cell_List &gp0){
		if (bv == no_vertex)
			bv = create_vertex(k0,k1,k2);
		gp0.set_corner_vertex(k0-1, k1-1, CORNER_111, bv);
		gp0.set_corner_vertex(k0-1, k1, CORNER_101, bv);
		gp0.set_corner_vertex(k0, k1-1, CORNER_011, bv);
		gp0.set_corner_vertex(k0, k1, CORNER_001, bv);
		return bv;
	}

	void make_triangles(Grid_Cell_List &gp0, int k2){
		int64_t k0_size = size[0], k1_size = size[1], k2_size = size[2];
		std::vector<Grid_Cell*>& clist = gp0.cells;
	  int64_t cc = gp0.cell_count;
	  for (int64_t k = 0; k < cc; ++k) {
	  	Grid_Cell* c = clist[k];
			T gc = grid[c->k0][c->k1][k2-1];
			T gc2 = grid[c->k0][c->k1][k2];
			// Construct bitmask to see which vertices are below the threshold
			int bits = ((gc < threshold ? 0 : 1) |
									(grid[c->k0+1][c->k1][k2-1] < threshold ? 0 : 2) |
									(grid[c->k0+1][c->k1+1][k2-1] < threshold ? 0 : 4) |
									(grid[c->k0][c->k1+1][k2-1] < threshold ? 0 : 8) |
									(gc2 < threshold ? 0 : 16) |
									(grid[c->k0+1][c->k1][k2] < threshold ? 0 : 32) |
									(grid[c->k0+1][c->k1+1][k2] < threshold ? 0 : 64) |
									(grid[c->k0][c->k1+1][k2] < threshold ? 0 : 128));
			int64_t* cell_vertices = c->vertex;
			int* t = triangle_table[bits];
			for (int e = *t; e != -1; ++t, e = *t) {
				add_triangle_corner(cell_vertices[e]);
			}
			if (c->boundary && cap_faces) {
				if (c->k0 == 0) make_cap_triangles(4, bits, cell_vertices);
				if (c->k0 + 2 == k0_size) make_cap_triangles(2, bits, cell_vertices);
				if (c->k1 == 0) make_cap_triangles(1, bits, cell_vertices);
				if (c->k1 + 2 == k1_size) make_cap_triangles(3, bits, cell_vertices);
				if (k2 == 1) make_cap_triangles(0, bits, cell_vertices);
				if (k2 + 1 == k2_size) make_cap_triangles(5, bits, cell_vertices);
			}
	  }
	}

};


//void _contour_surface(vector<vector<vector<float>>> &grid_data, float threshold, bool cap_faces){
//	const int64_t grid_shape[3] = {grid_data.size(), grid_data[0].size(), grid_data[0][0].size()};
//	Contour_Surface<float> *c_surface = new Contour_Surface<float>(&grid_data, grid_shape, threshold, cap_faces, CONTOUR_ARRAY_BLOCK_SIZE);
//
//
//
//	vector<vector<float>> *vxyz, *nxyz;
//	int64_t *tvi;
//	c_surface -> geometry(vxyz, tvi);
//	c_surface -> normals(nxyz);
//	delete c_surface;
//
//
//	vector<float> _vxyz = flatten_2d_vector(vxyz);
//	vector<float> _nxyz = flatten_2d_vector(nxyz);
//	vector<int64_t> _tvi = flatten_2d_vector(tvi);
//
////	return py::make_tuple(
////		py::array_t<float>({vxyz.size(), 3}, _vxyz.data()),
////		py::array_t<float>({nxyz.size(), 3}, _nxyz.data()),
////		py::array_t<int64_t>({tvi.size(), 3}, _tvi.data())
////	);
//	return py::make_tuple(1,2,3);
//}


// In the original code, pass the address of a contour surface object to it and it in-place modifies the object
// put the
py::tuple contour_surface(py::array_t<float> dist_matrix, const int level, const bool cap_faces=false, const bool calculate_normals=true){
	py::buffer_info dist_matrix_info = dist_matrix.request();
	auto _dist_matrix = static_cast<float *>(dist_matrix_info.ptr);

	vector<vector<vector <float>>> v_dist_matrix = cast_vector_3d(_dist_matrix, dist_matrix_info.shape[0], dist_matrix_info.shape[1], dist_matrix_info.shape[2]);
	vector<int> v_shape = {
		static_cast<int>(dist_matrix_info.shape[0]),
		static_cast<int>(dist_matrix_info.shape[1]),
		static_cast<int>(dist_matrix_info.shape[2]),
	};

//	vector<float> *vxyz, *nxyz;
//	vector<int64_t> *tvi;

//	const int64_t grid_shape[3] = {v_dist_matrix.size(), v_dist_matrix[0].size(), v_dist_matrix[0][0].size()};
	Contour_Surface<float> *c_surface = new Contour_Surface<float>(&v_dist_matrix, level, cap_faces, CONTOUR_ARRAY_BLOCK_SIZE);
	c_surface->compute_surface();





//	py::tuple ret_tuple = _contour_surface(v_dist_matrix, level, cap_faces);
	py::tuple ret_tuple;
	return ret_tuple;
//	py::array_t<float> sas_va; // vertex array
//	py::array_t<float> sas_ta; // triangle array
//	py::array_t<float> sas_na; // normal array
//	return py::make_tuple(sas_va, sas_ta, sas_na);

}





PYBIND11_MODULE(test_surf, m) {
  m.def("sphere_surface_distance", &sphere_surface_distance,
  	py::arg("centers"),
  	py::arg("radii"),
  	py::arg("maxrange"),
  	py::arg("grid_shape")
  );
  m.def("contour_surface", &contour_surface,
		py::arg("dist_matrix"),
		py::arg("level"),
		py::arg("cap_faces")=false,
		py::arg("calculate_normals")=true
  );
}
