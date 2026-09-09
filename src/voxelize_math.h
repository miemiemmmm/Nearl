// Voxelization math shared by the CUDA kernels and the CPU reference.
//
// Compiled twice: nvcc sees __host__ __device__, g++ sees a plain function. The
// CPU path is a fallback, not a re-derivation -- keeping one definition is what
// stops the two backends from drifting apart.
//
// Results still differ in the last bits: the GPU accumulates the normalizer in a
// shared-memory tree and scatters with atomicAdd, while the CPU sums in order, so
// the two agree to float rounding rather than bitwise.

#ifndef NEARL_VOXELIZE_MATH_H
#define NEARL_VOXELIZE_MATH_H

#include <cmath>

#include "constants.h"

#ifdef __CUDACC__
#define NEARL_HD __host__ __device__
#else
#define NEARL_HD
#endif


// Distance-based Gaussian, single precision throughout.
NEARL_HD inline float nearl_gaussian(const float distance, const float sigma) {
  if (sigma == 0.0f)
    return 0.0f;
  const float z = distance / sigma;
  return expf(-0.5f * z * z) / (sigma * SQRT_2_PI);
}


// Inclusive index range of the atom's cutoff ball on one axis. A grid point at
// integer index i sits at i * spacing, so the ball spans
// [(c - cutoff) / spacing, (c + cutoff) / spacing]. The bound is tight; callers
// keep the dist_sq < cutoff_sq test inside the loop.
NEARL_HD inline void nearl_ball_bounds(const float c, const float cutoff, const float spacing,
                                       int *lo, int *hi) {
  *lo = static_cast<int>(ceilf((c - cutoff) / spacing));
  *hi = static_cast<int>(floorf((c + cutoff) / spacing));
}


NEARL_HD inline bool nearl_is_placeholder(const float *coord) {
  return coord[0] == DEFAULT_COORD_PLACEHOLDER && coord[1] == DEFAULT_COORD_PLACEHOLDER &&
         coord[2] == DEFAULT_COORD_PLACEHOLDER;
}


// Squared distance from an atom to the grid point at integer index (x, y, z).
NEARL_HD inline float nearl_dist_sq(const float *coord, const int x, const int y, const int z,
                                    const float spacing) {
  const float dx = coord[0] - x * spacing;
  const float dy = coord[1] - y * spacing;
  const float dz = coord[2] - z * spacing;
  return dx * dx + dy * dy + dz * dz;
}


// Flat index of grid point (x, y, z). Inverts the decoding the voxelization
// kernel has always used: z is the fastest axis and pairs with dims[0].
NEARL_HD inline int nearl_grid_index(const int x, const int y, const int z, const int *dims) {
  return x * dims[0] * dims[1] + y * dims[0] + z;
}


// ---------------------------------------------------------------------------
// Frame-wise aggregations, indexed by the type_agg codes in features.py.
// ---------------------------------------------------------------------------

NEARL_HD inline float nearl_agg_sum(const float *v, const int n) {
  float s = 0.0f;
  for (int i = 0; i < n; i++)
    s += v[i];
  return s;
}


NEARL_HD inline float nearl_agg_mean(const float *v, const int n) {
  return nearl_agg_sum(v, n) / n;
}


NEARL_HD inline float nearl_agg_variance(const float *v, const int n) {
  const float m = nearl_agg_mean(v, n);
  float s = 0.0f;
  for (int i = 0; i < n; i++)
    s += (v[i] - m) * (v[i] - m);
  return s / n;
}


NEARL_HD inline float nearl_agg_stddev(const float *v, const int n) {
  return sqrtf(nearl_agg_variance(v, n));
}


NEARL_HD inline float nearl_agg_max(const float *v, const int n) {
  float m = v[0];
  for (int i = 1; i < n; i++)
    if (v[i] > m)
      m = v[i];
  return m;
}


NEARL_HD inline float nearl_agg_min(const float *v, const int n) {
  float m = v[0];
  for (int i = 1; i < n; i++)
    if (v[i] < m)
      m = v[i];
  return m;
}


// Sorts in place, as the device version does.
NEARL_HD inline float nearl_agg_median(float *v, const int n) {
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (v[i] > v[j]) {
        const float t = v[i];
        v[i] = v[j];
        v[j] = t;
      }
    }
  }
  if (n % 2 == 0)
    return (v[n / 2 - 1] + v[n / 2]) / 2;
  return v[n / 2];
}


// Entropy of a fixed-width histogram over the observed range.
NEARL_HD inline float nearl_agg_entropy(const float *v, const int n) {
  if (n <= 1)
    return 0.0f;
  float lo = v[0], hi = v[0];
  for (int i = 1; i < n; ++i) {
    if (v[i] < lo)
      lo = v[i];
    if (v[i] > hi)
      hi = v[i];
  }
  const float range = hi - lo;
  if (range == 0.0f)
    return 0.0f;

  int hist[INFORMATION_ENTROPY_BINS] = {0};
  for (int i = 0; i < n; ++i) {
    int bin = static_cast<int>(((v[i] - lo) / range) * INFORMATION_ENTROPY_BINS);
    if (bin >= INFORMATION_ENTROPY_BINS)
      bin = INFORMATION_ENTROPY_BINS - 1;
    hist[bin] += 1;
  }

  float entropy = 0.0f;
  for (int i = 0; i < INFORMATION_ENTROPY_BINS; ++i) {
    if (hist[i] > 0) {
      const float p = static_cast<float>(hist[i]) / n;
      entropy -= p * log2f(p);
    }
  }
  return entropy;
}


// Drift: slope of the least-squares line through the series.
NEARL_HD inline float nearl_agg_slope(const float *v, const int n) {
  if (n <= 1)
    return 0.0f;
  float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
  for (int i = 0; i < n; i++) {
    sum_x += i;
    sum_y += v[i];
    sum_xy += i * v[i];
    sum_x2 += i * i;
  }
  const float denom = n * sum_x2 - sum_x * sum_x;
  if (denom == 0.0f)
    return 0.0f;
  return (n * sum_xy - sum_x * sum_y) / denom;
}


// Dispatch on the type_agg code. `v` is scratch and may be reordered.
NEARL_HD inline float nearl_aggregate(float *v, const int n, const int type_agg) {
  switch (type_agg) {
  case 1:
    return nearl_agg_mean(v, n);
  case 2:
    return nearl_agg_stddev(v, n);
  case 3:
    return nearl_agg_median(v, n);
  case 4:
    return nearl_agg_variance(v, n);
  case 5:
    return nearl_agg_max(v, n);
  case 6:
    return nearl_agg_min(v, n);
  case 7:
    return nearl_agg_entropy(v, n);
  case 8:
    return nearl_agg_slope(v, n);
  default:
    return 0.0f;
  }
}

#endif // NEARL_VOXELIZE_MATH_H
