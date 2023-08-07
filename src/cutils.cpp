#include <iostream>
//#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "surface.h"
//#include "cutils.h"

using namespace std;
namespace py = pybind11;
#define CONTOUR_ARRAY_BLOCK_SIZE 1048576

//////////// Pure C++ functions ////////////



template <class Data_Type>
Contour_Surface *surface(const Data_Type *grid, const int size[3], const int stride[3],
float threshold, bool cap_faces){
  return cs;
}


//////////// Pybind C++ functions ////////////
py::tuple surface_py2(py::array_t<float> input_data, float threshold, bool cap_faces = true, bool return_normals = false){
  // Convert py::array to Numeric_Array
  py::buffer_info data = input_data.request();
  // You need to define the conversion from py::array to Numeric_Array
  // Call the C++ function for contour surface computation
  int size[3] = {static_cast<int>(data.shape[2]),
        static_cast<int>(data.shape[1]),
        static_cast<int>(data.shape[0])};
  int stride[3] = {data.stride(2), data.stride(1), data.stride(0)};
  int stride[3] = {(int)size[1]*size[0], (int)size[0], (int)1}
  // CSurface<Data_Type>(grid, size, stride,
  Contour_Surface* cs = new CSurface<double>(grid, size, stride,
                threshold, cap_faces, CONTOUR_ARRAY_BLOCK_SIZE);

  float* vxyz;
  float* nxyz;
  int* tvi;
  py::array_t<float> vertex_xyz = py::array_t<float>({cs->vertex_count(), 3});
  py::array_t<float> normals;
  if (return_normals){normals = py::array_t<float>({cs->vertex_count(), 3});};
  py::array_t<int> tv_indices = py::array_t<int>({cs->triangle_count(), 3});
  vxyz = static_cast<float*>(vertex_xyz.request().ptr);
  tvi = static_cast<int*>(tv_indices.request().ptr);
  if (return_normals){nxyz = static_cast<float*>(normals.request().ptr);};

  cs->geometry(vxyz, reinterpret_cast<int *>(tvi));
  if (return_normals){cs->normals(nxyz);};
  delete cs;
  if (return_normals){
    return py::make_tuple(vertex_xyz, tv_indices, normals);
  }else{
    return py::make_tuple(vertex_xyz, tv_indices);
  }
}



void sphere_surface_distance(py::array_t<float> centers, py::array_t<float> radii, float maxrange, py::array_t<float> matrix){
  py::buffer_info info_centers = centers.request();
  py::buffer_info info_radii = radii.request();
  py::buffer_info info_matrix = matrix.request();

  int64_t n = info_centers.shape[0];
  int64_t rs0 = info_radii.strides[0]/sizeof(float);
  int64_t cs0 = info_centers.strides[0]/sizeof(float), cs1 = info_centers.strides[1]/sizeof(float);
  int64_t ms0 = info_matrix.strides[0]/sizeof(float), ms1 = info_matrix.strides[1]/sizeof(float), ms2 = info_matrix.strides[2]/sizeof(float);

  float *ca = static_cast<float *>(info_centers.ptr);
  float *ra = static_cast<float *>(info_radii.ptr);
  float *ma = static_cast<float *>(info_matrix.ptr);

  for (int64_t c = 0 ; c < n ; ++c){
    float r = ra[c*rs0];
    if (r == 0) {
      continue;
    }

    float cijk[3];
    int ijk_min[3], ijk_max[3];
    for (int p = 0 ; p < 3 ; ++p){
      float x = ca[cs0*c+cs1*p];
      cijk[p] = x;
      ijk_min[p] = std::clamp((int)ceil(x-(r+maxrange)), 0, (int)info_matrix.shape[2-p]);
      ijk_max[p] = std::clamp((int)floor(x+(r+maxrange)), 0, (int)info_matrix.shape[2-p]);
    }

    for (int k = ijk_min[2] ; k <= ijk_max[2] ; ++k){
      float dk = (k-cijk[2]);
      float k2 = dk*dk;
      for (int j = ijk_min[1] ; j <= ijk_max[1] ; ++j){
        float dj = (j-cijk[1]);
        float jk2 = dj*dj + k2;
        for (int i = ijk_min[0] ; i <= ijk_max[0] ; ++i){
          float di = (i-cijk[0]);
          float ijk2 = di*di + jk2;
          float rd = sqrt(ijk2) - r;
          float *mijk = ma + (k*ms0+j*ms1+i*ms2);
          if (rd < *mijk){
            *mijk = rd;
          }
        }
      }
    }
  }
}


py::array_t<double> pairwise_distance(py::array_t<double> input_array){
  py::buffer_info buf = input_array.request();
  if (buf.ndim != 2) {
    throw runtime_error("Number of dimensions must be two");
  }
  auto data = static_cast<double *>(buf.ptr);
  int n = buf.shape[0];
  int d = buf.shape[1];
  vector<double> dist_mtx(n*n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
    double dist = 0.0;
    for (int k = 0; k < d; k++) {
      double diff = data[i*d + k] - data[j*d + k];
      dist += diff * diff;
    }
      dist = sqrt(dist);
      dist_mtx[i*n + j] = dist;
    }
  }
  py::array_t<double> result({n, n}, dist_mtx.data());
  return result;
}



py::array_t<double> cdist(py::array_t<double> arr1, py::array_t<double> arr2, int cutoff = 12){

  py::buffer_info buf1 = arr1.request();
  py::buffer_info buf2 = arr2.request();
  if (buf1.ndim != 2 || buf2.ndim != 2) {
    throw runtime_error("Number of dimensions must be two");
  }
  auto data1 = static_cast<double *>(buf1.ptr);
  auto data2 = static_cast<double *>(buf2.ptr);
  int n1 = buf1.shape[0];
  int n2 = buf2.shape[0];
  int d = buf1.shape[1];

  vector<double> dist_mtx(n1*n2);
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      double dist = 0.0;
      bool skip = false;
      for (int k = 0; k < d; k++) {
        double diff = data1[i*d + k] - data2[j*d + k];
        if (abs(diff) > cutoff){
          skip = true;
          break;
        } else {
          dist += diff * diff;
        }
      }
      if (! skip) {
        dist = sqrt(dist);
        dist_mtx[i*n2 + j] = dist;
      }
    }
  }
  py::array_t<double> result({n1, n2}, dist_mtx.data());
  return result;
}

PYBIND11_MODULE(cutils, m) {
  m.def("pairwise_distance", &pairwise_distance);
  m.def("cdist", &cdist, py::arg("arr1"), py::arg("arr2"), py::arg("cutoff") = 12);
  m.def("sphere_surface_distance", &sphere_surface_distance);
}
