#include <algorithm>
#include <cstring>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace {
using FloatInput = py::array_t<float, py::array::c_style | py::array::forcecast>;
// Coordinates arrive as float64 (pytraj's native dtype); keeping this distinct from
// FloatInput avoids forcing a whole-array cast before the loop even starts.
using DoubleInput = py::array_t<double, py::array::c_style | py::array::forcecast>;
using BoolInput = py::array_t<bool, py::array::c_style | py::array::forcecast>;

/**
 * @brief Boolean mask of which points fall strictly inside an axis-aligned box.
 *
 * @param points: Coordinates, shape (n, 3), float64 (untranslated)
 * @param lower: Per-axis lower bound, shape (3,)
 * @param upper: Per-axis upper bound, shape (3,)
 * @return py::array_t<bool>: True where the point is inside the box, shape (n,)
 */
py::array_t<bool> do_crop_mask(DoubleInput points, FloatInput lower, FloatInput upper) {
  py::buffer_info buf_points = points.request();
  py::buffer_info buf_lower = lower.request();
  py::buffer_info buf_upper = upper.request();

  const py::ssize_t n = buf_points.shape[0];
  const double *p = static_cast<double *>(buf_points.ptr);
  const float *lo = static_cast<float *>(buf_lower.ptr);
  const float *hi = static_cast<float *>(buf_upper.ptr);

  py::array_t<bool> mask(n);
  bool *out = mask.mutable_data();

  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i) {
      const double x = p[i * 3 + 0];
      const double y = p[i * 3 + 1];
      const double z = p[i * 3 + 2];
      out[i] =
          (x > lo[0]) && (x < hi[0]) && (y > lo[1]) && (y < hi[1]) && (z > lo[2]) && (z < hi[2]);
    }
  }
  return mask;
}

/**
 * @brief Gather the atoms selected by `mask` and translate them by `offset`, in one pass.
 *
 * @param points: Coordinates, shape (n, 3), float64
 * @param mask: Boolean selection, shape (n,)
 * @param offset: Per-axis translation, shape (3,)
 * @return py::array_t<float>: Selected, translated coordinates, shape (k, 3)
 */
py::array_t<float> do_gather_translate(DoubleInput points, BoolInput mask, FloatInput offset) {
  py::buffer_info buf_points = points.request();
  py::buffer_info buf_mask = mask.request();
  py::buffer_info buf_offset = offset.request();

  const py::ssize_t n = buf_points.shape[0];
  const double *p = static_cast<double *>(buf_points.ptr);
  const bool *m = static_cast<bool *>(buf_mask.ptr);
  const float *off = static_cast<float *>(buf_offset.ptr);

  py::ssize_t k = 0;
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < n; ++i)
      if (m[i])
        ++k;
  }

  py::array_t<float> out_arr({k, static_cast<py::ssize_t>(3)});
  float *out = out_arr.mutable_data();

  {
    py::gil_scoped_release release;
    py::ssize_t j = 0;
    for (py::ssize_t i = 0; i < n; ++i) {
      if (m[i]) {
        out[j * 3 + 0] = static_cast<float>(p[i * 3 + 0]) + off[0];
        out[j * 3 + 1] = static_cast<float>(p[i * 3 + 1]) + off[1];
        out[j * 3 + 2] = static_cast<float>(p[i * 3 + 2]) + off[2];
        ++j;
      }
    }
  }
  return out_arr;
}

/**
 * @brief Crop-result-gather + translate + weight-gather + pad, for every frame of a batch,
 * in one call - instead of the caller invoking this once per frame.
 *
 * @param frame_coords: Coordinates, shape (n_frames, n_atoms, 3), float64
 * @param mask: Boolean selection per frame, shape (n_frames, n_atoms)
 * @param offset: Per-axis translation shared by every frame, shape (3,)
 * @param cached_array: Per-atom weight, shape (n_atoms,)
 * @param max_allowed: Clamp on atoms kept per frame
 * @param default_coord: Padding value for coordinate slots with no atom
 * @return py::tuple: (coords[f, max_atom_nr, 3], weights[f, max_atom_nr], raw_counts[f],
 *   max_atom_nr) - raw_counts is the unclamped per-frame count, kept for the caller's
 *   MAX_ALLOWED_ATOMS-exceeded warning.
 */
py::tuple do_batched_gather(DoubleInput frame_coords, BoolInput mask, FloatInput offset,
                            FloatInput cached_array, const int max_allowed,
                            const float default_coord) {
  py::buffer_info buf_frames = frame_coords.request();
  py::buffer_info buf_mask = mask.request();
  py::buffer_info buf_offset = offset.request();
  py::buffer_info buf_cached = cached_array.request();

  const py::ssize_t n_frames = buf_frames.shape[0];
  const py::ssize_t n_atoms = buf_frames.shape[1];
  const double *frames = static_cast<double *>(buf_frames.ptr);
  const bool *m = static_cast<bool *>(buf_mask.ptr);
  const float *off = static_cast<float *>(buf_offset.ptr);
  const float *cached = static_cast<float *>(buf_cached.ptr);

  std::vector<int> raw_counts(n_frames);
  std::vector<int> counts(n_frames);
  int max_atom_nr = 0;

  {
    py::gil_scoped_release release;
    for (py::ssize_t f = 0; f < n_frames; ++f) {
      int c = 0;
      const bool *mf = m + f * n_atoms;
      for (py::ssize_t a = 0; a < n_atoms; ++a)
        if (mf[a])
          ++c;
      raw_counts[f] = c;
      const int clamped = c < max_allowed ? c : max_allowed;
      counts[f] = clamped;
      if (clamped > max_atom_nr)
        max_atom_nr = clamped;
    }
  }

  py::array_t<float> coords_out(
      {n_frames, static_cast<py::ssize_t>(max_atom_nr), static_cast<py::ssize_t>(3)});
  py::array_t<float> weights_out({n_frames, static_cast<py::ssize_t>(max_atom_nr)});
  py::array_t<int> raw_counts_out(n_frames);

  float *coords_ptr = coords_out.mutable_data();
  float *weights_ptr = weights_out.mutable_data();
  std::copy(raw_counts.begin(), raw_counts.end(), raw_counts_out.mutable_data());

  {
    py::gil_scoped_release release;
    std::fill(coords_ptr, coords_ptr + static_cast<size_t>(n_frames) * max_atom_nr * 3,
              default_coord);
    std::fill(weights_ptr, weights_ptr + static_cast<size_t>(n_frames) * max_atom_nr, 0.0f);

    for (py::ssize_t f = 0; f < n_frames; ++f) {
      const int k = counts[f];
      if (k == 0)
        continue;
      const bool *mf = m + f * n_atoms;
      const double *ff = frames + f * n_atoms * 3;
      float *cf = coords_ptr + static_cast<size_t>(f) * max_atom_nr * 3;
      float *wf = weights_ptr + static_cast<size_t>(f) * max_atom_nr;
      int j = 0;
      for (py::ssize_t a = 0; a < n_atoms; ++a) {
        if (mf[a]) {
          cf[j * 3 + 0] = static_cast<float>(ff[a * 3 + 0]) + off[0];
          cf[j * 3 + 1] = static_cast<float>(ff[a * 3 + 1]) + off[1];
          cf[j * 3 + 2] = static_cast<float>(ff[a * 3 + 2]) + off[2];
          wf[j] = cached[a];
          ++j;
          if (j == k)
            break;
        }
      }
    }
  }

  return py::make_tuple(coords_out, weights_out, raw_counts_out, max_atom_nr);
}

} // namespace

PYBIND11_MODULE(host_actions, m) {
  m.def("crop_mask", &do_crop_mask, py::arg("points"), py::arg("lower"), py::arg("upper"),
        "Boolean mask of which atoms fall inside an axis-aligned box");
  m.def("gather_translate", &do_gather_translate, py::arg("points"), py::arg("mask"),
        py::arg("offset"), "Gather the atoms selected by mask and translate them, in one pass");
  m.def("batched_gather", &do_batched_gather, py::arg("frame_coords"), py::arg("mask"),
        py::arg("offset"), py::arg("cached_array"), py::arg("max_allowed"),
        py::arg("default_coord"),
        "Crop-result-gather + translate + weight-gather + pad for every frame in one call");
}
