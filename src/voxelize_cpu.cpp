// CPU reference for the voxelization kernels.
//
// voxelize_host_cpu mirrors frame_interp_global one step at a time: same ball
// bounds, same two clipped extents, same strict cutoff test, same flat index.
// The arithmetic itself lives in voxelize_math.h so both backends share it.

#include "voxelize_cpu.h"

#include <algorithm>
#include <vector>

#include "constants.h"
#include "voxelize_math.h"


void voxelize_host_cpu(float *interpolated, const float *coord, const float *weight,
                       const int *dims, const float spacing, const int atom_nr, const float cutoff,
                       const float sigma) {
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  std::fill(interpolated, interpolated + gridpoint_nr, 0.0f);
  if (atom_nr <= 0)
    return;

  const int buff_dim = static_cast<int>((cutoff + spacing) / spacing);
  const int buff_dims[3] = {dims[0] + 2 * buff_dim, dims[1] + 2 * buff_dim, dims[2] + 2 * buff_dim};
  const float cutoff_sq = cutoff * cutoff;

  for (int atom_idx = 0; atom_idx < atom_nr; ++atom_idx) {
    const float *c = coord + atom_idx * 3;
    const float w = weight[atom_idx];
    if (nearl_is_placeholder(c) || w == 0.0f)
      continue;

    int lo[3], hi[3];
    for (int d = 0; d < 3; ++d)
      nearl_ball_bounds(c[d], cutoff, spacing, &lo[d], &hi[d]);

    // Normalizer over the buffered grid: it reaches buff_dim points outside the
    // output on every face, so an atom near a face is still normalized over its
    // whole ball. Axis i pairs with buff_dims[2 - i], matching the kernel.
    float total = 0.0f;
    for (int x = std::max(lo[0], -buff_dim); x <= std::min(hi[0], buff_dims[2] - buff_dim - 1);
         ++x) {
      for (int y = std::max(lo[1], -buff_dim); y <= std::min(hi[1], buff_dims[1] - buff_dim - 1);
           ++y) {
        for (int z = std::max(lo[2], -buff_dim); z <= std::min(hi[2], buff_dims[0] - buff_dim - 1);
             ++z) {
          const float d2 = nearl_dist_sq(c, x, y, z, spacing);
          if (d2 < cutoff_sq)
            total += nearl_gaussian(sqrtf(d2), sigma);
        }
      }
    }
    if (total == 0.0f)
      continue;

    // Scatter over the output grid only, so an edge atom deposits just the
    // in-grid part of its weight. That asymmetry is deliberate in the kernel.
    const float inv_sum = w / total;
    for (int x = std::max(lo[0], 0); x <= std::min(hi[0], dims[2] - 1); ++x) {
      for (int y = std::max(lo[1], 0); y <= std::min(hi[1], dims[1] - 1); ++y) {
        for (int z = std::max(lo[2], 0); z <= std::min(hi[2], dims[0] - 1); ++z) {
          const float d2 = nearl_dist_sq(c, x, y, z, spacing);
          if (d2 < cutoff_sq)
            interpolated[nearl_grid_index(x, y, z, dims)] +=
                nearl_gaussian(sqrtf(d2), sigma) * inv_sum;
        }
      }
    }
  }
}


void trajectory_voxelization_host_cpu(float *voxelize_dynamics, const float *coord,
                                      const float *weight, const int *dims, const float spacing,
                                      const int frame_nr, const int atom_nr, const float cutoff,
                                      const float sigma, const int type_agg) {
  const int gridpoint_nr = dims[0] * dims[1] * dims[2];
  std::fill(voxelize_dynamics, voxelize_dynamics + gridpoint_nr, 0.0f);
  if (frame_nr <= 0)
    return;

  std::vector<float> frames(static_cast<size_t>(frame_nr) * gridpoint_nr, 0.0f);
  for (int f = 0; f < frame_nr; ++f) {
    voxelize_host_cpu(frames.data() + static_cast<size_t>(f) * gridpoint_nr,
                      coord + static_cast<size_t>(f) * atom_nr * 3,
                      weight + static_cast<size_t>(f) * atom_nr, dims, spacing, atom_nr, cutoff,
                      sigma);
  }

  // The kernel path aggregates at most MAX_FRAME_NUMBER frames; match it.
  const int used = frame_nr > MAX_FRAME_NUMBER ? MAX_FRAME_NUMBER : frame_nr;
  std::vector<float> column(used);
  for (int g = 0; g < gridpoint_nr; ++g) {
    for (int f = 0; f < used; ++f)
      column[f] = frames[static_cast<size_t>(f) * gridpoint_nr + g];
    voxelize_dynamics[g] = nearl_aggregate(column.data(), used, type_agg);
  }
}
