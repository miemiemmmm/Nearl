// Created by: Yang Zhang
// Description: The CUDA implementation of the marching observer algorithm

#include <algorithm>

#include <iostream>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "constants.h"  // For hard-coded variables: BLOCK_SIZE, MAX_FRAME_NUMBER
#include "gpuutils.cuh" // For hard-coded BLOCK_SIZE and device functions: mean_device, mean_device, standard_deviation_device
#include "marching_observers.cuh"


////////////////////////////////////////////////////////////////////////////////
// Spatial cell index over the atoms
////////////////////////////////////////////////////////////////////////////////
// Atoms are bucketed into a uniform grid of cells with cell size >= cutoff.
// Every atom within cutoff of an observer then lies in one of the 27 cells
// surrounding the observer's cell, so each observer only inspects those
// instead of the whole frame. Atoms are counting-sorted by (frame, cell) into
// sorted_atoms, a CSR layout: cell c's atoms occupy a contiguous run of
// sorted_atoms starting at cell_start[c] and ending just before cell_start[c+1],
// so walking a cell is a sequential, coalesced scan instead of following a
// scattered linked list. Padded placeholder atoms
// are never bucketed.
struct ObserverCells {
  const int *cell_start;
  const int *sorted_atoms;
  const float *coords;
  int3 dims;
  float3 min;
  float cell_size;
  int cell_count;
  int frame_base;
};


__device__ int3 observer_cell_index(const float *coord, const ObserverCells &cells) {
  return make_int3(static_cast<int>(floorf((coord[0] - cells.min.x) / cells.cell_size)),
                   static_cast<int>(floorf((coord[1] - cells.min.y) / cells.cell_size)),
                   static_cast<int>(floorf((coord[2] - cells.min.z) / cells.cell_size)));
}


// Shared by count_observer_cell_atoms and scatter_observer_cell_atoms: which
// (frame, cell) an atom belongs to, clamped so float rounding at the grid's
// edge can't push it out of range.
__device__ __forceinline__ int observer_atom_cell(const float *c, int idx, int atomnr,
                                                  int3 cell_dims, float3 cell_min, float cell_size,
                                                  int cell_total) {
  int cx = static_cast<int>(floorf((c[0] - cell_min.x) / cell_size));
  int cy = static_cast<int>(floorf((c[1] - cell_min.y) / cell_size));
  int cz = static_cast<int>(floorf((c[2] - cell_min.z) / cell_size));
  cx = min(max(cx, 0), cell_dims.x - 1);
  cy = min(max(cy, 0), cell_dims.y - 1);
  cz = min(max(cz, 0), cell_dims.z - 1);
  return (idx / atomnr) * cell_total + (cz * cell_dims.y + cy) * cell_dims.x + cx;
}


// Pass 1 of the counting sort: histogram of atoms per (frame, cell).
__global__ void count_observer_cell_atoms(int *cell_hist, const float *coord, int atom_total,
                                          int atomnr, int3 cell_dims, float3 cell_min,
                                          float cell_size, int cell_total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= atom_total)
    return;
  const float *c = coord + static_cast<size_t>(idx) * 3;
  if (c[0] == DEFAULT_COORD_PLACEHOLDER && c[1] == DEFAULT_COORD_PLACEHOLDER &&
      c[2] == DEFAULT_COORD_PLACEHOLDER)
    return;
  int cell = observer_atom_cell(c, idx, atomnr, cell_dims, cell_min, cell_size, cell_total);
  atomicAdd(&cell_hist[cell], 1);
}


// Pass 2: scatters each atom's index into its cell's slice of sorted_atoms.
// cell_cursor must start out holding a copy of cell_start's first cell_total entries.
__global__ void scatter_observer_cell_atoms(int *cell_cursor, int *sorted_atoms, const float *coord,
                                            int atom_total, int atomnr, int3 cell_dims,
                                            float3 cell_min, float cell_size, int cell_total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= atom_total)
    return;
  const float *c = coord + static_cast<size_t>(idx) * 3;
  if (c[0] == DEFAULT_COORD_PLACEHOLDER && c[1] == DEFAULT_COORD_PLACEHOLDER &&
      c[2] == DEFAULT_COORD_PLACEHOLDER)
    return;
  int cell = observer_atom_cell(c, idx, atomnr, cell_dims, cell_min, cell_size, cell_total);
  int pos = atomicAdd(&cell_cursor[cell], 1);
  sorted_atoms[pos] = idx;
}


namespace {

struct ObserverCellGrid {
  int3 dims;
  float3 min;
  float cell_size;
  int cell_count;
};


// Cell grid covering every non-placeholder atom padded by cutoff on each side,
// so any observer within cutoff of an atom lies inside the grid. The +1 cell
// per axis keeps floor() on the device inside the range despite float rounding.
ObserverCellGrid compute_observer_cell_grid(const float *coord, size_t atom_count, float cutoff) {
  ObserverCellGrid grid;
  grid.cell_size = cutoff > 0.0f ? cutoff : 1.0f;
  float lo[3] = {0.0f, 0.0f, 0.0f}, hi[3] = {0.0f, 0.0f, 0.0f};
  bool any = false;
  for (size_t i = 0; i < atom_count; ++i) {
    const float *c = coord + i * 3;
    if (c[0] == DEFAULT_COORD_PLACEHOLDER && c[1] == DEFAULT_COORD_PLACEHOLDER &&
        c[2] == DEFAULT_COORD_PLACEHOLDER)
      continue;
    if (!any) {
      // First real atom seeds the bounding box directly; every later atom
      // just widens it, so the per-dimension checks below don't need to keep
      // asking "have we started yet" on every iteration.
      lo[0] = hi[0] = c[0];
      lo[1] = hi[1] = c[1];
      lo[2] = hi[2] = c[2];
      any = true;
      continue;
    }
    for (int d = 0; d < 3; ++d) {
      if (c[d] < lo[d])
        lo[d] = c[d];
      if (c[d] > hi[d])
        hi[d] = c[d];
    }
  }
  if (!any) {
    grid.dims = make_int3(1, 1, 1);
    grid.min = make_float3(0.0f, 0.0f, 0.0f);
    grid.cell_count = 1;
    return grid;
  }
  grid.min = make_float3(lo[0] - cutoff, lo[1] - cutoff, lo[2] - cutoff);
  float mins[3] = {grid.min.x, grid.min.y, grid.min.z};
  int d[3];
  for (int k = 0; k < 3; ++k)
    d[k] =
        std::max(1, static_cast<int>(std::ceil((hi[k] + cutoff - mins[k]) / grid.cell_size)) + 1);
  grid.dims = make_int3(d[0], d[1], d[2]);
  grid.cell_count = d[0] * d[1] * d[2];
  return grid;
}


struct ObserverCellBuffers {
  int *cell_start = nullptr;
  int *sorted_atoms = nullptr;
};


// Builds the CSR cell index via a counting sort (histogram, prefix sum, scatter):
// cell_start[c]..cell_start[c+1] gives the range of sorted_atoms holding cell
// c's atom indices, contiguous per cell instead of a scattered linked list.
// ctx may be the persistent global context or a function-local one; either
// way it owns freeing the buffers, so there's no cudaFree here.
ObserverCellBuffers build_observer_cell_buffers(DeviceContext *ctx, cudaStream_t stream,
                                                const float *coord_device, const float *coord_host,
                                                int frame_number, int atomnr, float cutoff,
                                                ObserverCellGrid &grid) {
  grid = compute_observer_cell_grid(coord_host, static_cast<size_t>(frame_number) * atomnr, cutoff);
  ObserverCellBuffers bufs;
  size_t cell_total = std::max(static_cast<size_t>(frame_number) * grid.cell_count, size_t(1));
  size_t atom_cap = std::max(static_cast<size_t>(frame_number) * atomnr, size_t(1));

  bufs.cell_start = ctx->get_buffer_i(cell_total + 1, static_cast<size_t>(BufferSlot::CELL_HEAD));
  bufs.sorted_atoms = ctx->get_buffer_i(atom_cap, static_cast<size_t>(BufferSlot::ATOM_NEXT));
  int *cell_hist = ctx->get_buffer_i(cell_total, static_cast<size_t>(BufferSlot::SCRATCH));

  CUDA_CHECK(cudaMemsetAsync(cell_hist, 0, cell_total * sizeof(int), stream));
  int atom_total = frame_number * atomnr;
  if (atom_total > 0) {
    int blocks = (atom_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_observer_cell_atoms<<<blocks, BLOCK_SIZE, 0, stream>>>(
        cell_hist, coord_device, atom_total, atomnr, grid.dims, grid.min, grid.cell_size,
        grid.cell_count);
    CUDA_CHECK_KERNEL();
  }

  // cell_start[0] = 0; cell_start[c+1] = inclusive prefix sum of cell_hist up through c.
  CUDA_CHECK(cudaMemsetAsync(bufs.cell_start, 0, sizeof(int), stream));
  thrust::inclusive_scan(thrust::cuda::par.on(stream), cell_hist, cell_hist + cell_total,
                         bufs.cell_start + 1);

  if (atom_total > 0) {
    // Reuse cell_hist as the per-cell write cursor, seeded from cell_start.
    CUDA_CHECK(cudaMemcpyAsync(cell_hist, bufs.cell_start, cell_total * sizeof(int),
                               cudaMemcpyDeviceToDevice, stream));
    int blocks = (atom_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scatter_observer_cell_atoms<<<blocks, BLOCK_SIZE, 0, stream>>>(
        cell_hist, bufs.sorted_atoms, coord_device, atom_total, atomnr, grid.dims, grid.min,
        grid.cell_size, grid.cell_count);
    CUDA_CHECK_KERNEL();
  }
  return bufs;
}

} // namespace


////////////////////////////////////////////////////////////////////////////////
// Weighted center of mass of the atoms within cutoff of coord. Shared by
// eccentricity (15) and radius-of-gyration (16), which both need it first.
struct ComResult {
  float com[3] = {0.0f, 0.0f, 0.0f};
  float weight_sum = 0.0f;
  int count = 0;
};


__device__ ComResult center_of_mass_device(const float *coord, const ObserverCells &cells,
                                           const float *weight_framei, const float cutoff) {
  float cutoff_sq = cutoff * cutoff;
  ComResult result;
  int3 oc = observer_cell_index(coord, cells);
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = oc.x + dx, cy = oc.y + dy, cz = oc.z + dz;
        if (cx < 0 || cx >= cells.dims.x || cy < 0 || cy >= cells.dims.y || cz < 0 ||
            cz >= cells.dims.z)
          continue;
        int cell = cells.frame_base + (cz * cells.dims.y + cy) * cells.dims.x + cx;
        for (int idx = cells.cell_start[cell]; idx < cells.cell_start[cell + 1]; ++idx) {
          int j = cells.sorted_atoms[idx];
          if (square_distance_device(coord, cells.coords + j * 3) > cutoff_sq)
            continue;
          result.count += 1;
          result.com[0] += weight_framei[j] * cells.coords[j * 3];
          result.com[1] += weight_framei[j] * cells.coords[j * 3 + 1];
          result.com[2] += weight_framei[j] * cells.coords[j * 3 + 2];
          result.weight_sum += weight_framei[j];
        }
      }
  return result;
}


////////////////////////////////////////////////////////////////////////////////
// Direct count-based observables (types 1-3; unweighted)
////////////////////////////////////////////////////////////////////////////////
/// @brief 1.0 if any atom lies within cutoff of the observer, else 0.0.
__device__ __forceinline__ float existence_device(const float *coord, const ObserverCells &cells,
                                                  const float cutoff) {
  float cutoff_sq = cutoff * cutoff;
  int3 oc = observer_cell_index(coord, cells);
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = oc.x + dx, cy = oc.y + dy, cz = oc.z + dz;
        if (cx < 0 || cx >= cells.dims.x || cy < 0 || cy >= cells.dims.y || cz < 0 ||
            cz >= cells.dims.z)
          continue;
        int cell = cells.frame_base + (cz * cells.dims.y + cy) * cells.dims.x + cx;
        for (int idx = cells.cell_start[cell]; idx < cells.cell_start[cell + 1]; ++idx)
          if (square_distance_device(coord, cells.coords + cells.sorted_atoms[idx] * 3) <=
              cutoff_sq)
            return 1.0f;
      }
  return 0.0f;
}


/// @brief Count of atoms within cutoff of the observer.
__device__ __forceinline__ float direct_count_device(const float *coord, const ObserverCells &cells,
                                                     const float cutoff) {
  float cutoff_sq = cutoff * cutoff;
  float count = 0.0f;
  int3 oc = observer_cell_index(coord, cells);
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = oc.x + dx, cy = oc.y + dy, cz = oc.z + dz;
        if (cx < 0 || cx >= cells.dims.x || cy < 0 || cy >= cells.dims.y || cz < 0 ||
            cz >= cells.dims.z)
          continue;
        int cell = cells.frame_base + (cz * cells.dims.y + cy) * cells.dims.x + cx;
        for (int idx = cells.cell_start[cell]; idx < cells.cell_start[cell + 1]; ++idx)
          if (square_distance_device(coord, cells.coords + cells.sorted_atoms[idx] * 3) <=
              cutoff_sq)
            count += 1.0f;
      }
  return count;
}


/// @brief Count of atoms within cutoff with distinct weights, treating weight as a
/// discrete id (e.g. atom/residue index). Limited to DISTINCT_LIMIT unique values.
__device__ __forceinline__ float distinct_count_device(const float *coord,
                                                       const ObserverCells &cells,
                                                       const float *weight_framei,
                                                       const float cutoff) {
  float cutoff_sq = cutoff * cutoff;
  float encountered[DISTINCT_LIMIT]; // only the first `count` entries are ever read
  int count = 0;
  int3 oc = observer_cell_index(coord, cells);
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = oc.x + dx, cy = oc.y + dy, cz = oc.z + dz;
        if (cx < 0 || cx >= cells.dims.x || cy < 0 || cy >= cells.dims.y || cz < 0 ||
            cz >= cells.dims.z)
          continue;
        int cell = cells.frame_base + (cz * cells.dims.y + cy) * cells.dims.x + cx;
        for (int idx = cells.cell_start[cell]; idx < cells.cell_start[cell + 1]; ++idx) {
          int j = cells.sorted_atoms[idx];
          if (square_distance_device(coord, cells.coords + j * 3) > cutoff_sq)
            continue;
          float val = weight_framei[j];
          bool seen = false;
          for (int k = 0; k < count; ++k)
            if (encountered[k] == val) {
              seen = true;
              break;
            }
          if (!seen && count < DISTINCT_LIMIT)
            encountered[count++] = val;
        }
      }
  return static_cast<float>(count);
}


////////////////////////////////////////////////////////////////////////////////
// Weight-based observables (types 11-16)
////////////////////////////////////////////////////////////////////////////////
/// @brief Weighted mean distance from the observer to atoms within cutoff.
__device__ __forceinline__ float mean_distance_device(const float *coord,
                                                      const ObserverCells &cells,
                                                      const float *weight_framei,
                                                      const float cutoff) {
  float cutoff_sq = cutoff * cutoff;
  float sum = 0.0f, weight_sum = 0.0f;
  int count = 0;
  int3 oc = observer_cell_index(coord, cells);
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = oc.x + dx, cy = oc.y + dy, cz = oc.z + dz;
        if (cx < 0 || cx >= cells.dims.x || cy < 0 || cy >= cells.dims.y || cz < 0 ||
            cz >= cells.dims.z)
          continue;
        int cell = cells.frame_base + (cz * cells.dims.y + cy) * cells.dims.x + cx;
        for (int idx = cells.cell_start[cell]; idx < cells.cell_start[cell + 1]; ++idx) {
          int j = cells.sorted_atoms[idx];
          float dist_sq = square_distance_device(coord, cells.coords + j * 3);
          if (dist_sq > cutoff_sq)
            continue;
          sum += sqrtf(dist_sq) * weight_framei[j];
          weight_sum += weight_framei[j];
          count += 1;
        }
      }
  return count > 0 ? sum / weight_sum : 0.0f;
}


/// @brief Sum of weights of atoms within cutoff of the observer.
__device__ __forceinline__ float cumulative_weight_device(const float *coord,
                                                          const ObserverCells &cells,
                                                          const float *weight_framei,
                                                          const float cutoff) {
  float cutoff_sq = cutoff * cutoff;
  float weight_sum = 0.0f;
  int3 oc = observer_cell_index(coord, cells);
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = oc.x + dx, cy = oc.y + dy, cz = oc.z + dz;
        if (cx < 0 || cx >= cells.dims.x || cy < 0 || cy >= cells.dims.y || cz < 0 ||
            cz >= cells.dims.z)
          continue;
        int cell = cells.frame_base + (cz * cells.dims.y + cy) * cells.dims.x + cx;
        for (int idx = cells.cell_start[cell]; idx < cells.cell_start[cell + 1]; ++idx) {
          int j = cells.sorted_atoms[idx];
          if (square_distance_device(coord, cells.coords + j * 3) <= cutoff_sq)
            weight_sum += weight_framei[j];
        }
      }
  return weight_sum;
}


/// @brief Cumulative weight within cutoff divided by the cutoff sphere's volume.
__device__ __forceinline__ float density_device(const float *coord, const ObserverCells &cells,
                                                const float *weight_framei, const float cutoff) {
  float weight_sum = cumulative_weight_device(coord, cells, weight_framei, cutoff);
  float volume = (4.0f / 3.0f) * static_cast<float>(M_PI) * cutoff * cutoff * cutoff;
  return weight_sum / volume;
}


/// @brief Weighted pairwise dispersion: sum(w_j * w_k * dist(j,k)) / sum(w_j * w_k), j < k.
__device__ __forceinline__ float dispersion_device(const float *coord, const ObserverCells &cells,
                                                   const float *weight_framei, const float cutoff) {
  float cutoff_sq = cutoff * cutoff;
  float retval = 0.0f, weight_sum = 0.0f;
  int3 oc = observer_cell_index(coord, cells);
  for (int dzj = -1; dzj <= 1; ++dzj)
    for (int dyj = -1; dyj <= 1; ++dyj)
      for (int dxj = -1; dxj <= 1; ++dxj) {
        int cxj = oc.x + dxj, cyj = oc.y + dyj, czj = oc.z + dzj;
        if (cxj < 0 || cxj >= cells.dims.x || cyj < 0 || cyj >= cells.dims.y || czj < 0 ||
            czj >= cells.dims.z)
          continue;
        int cellj = cells.frame_base + (czj * cells.dims.y + cyj) * cells.dims.x + cxj;
        for (int idxj = cells.cell_start[cellj]; idxj < cells.cell_start[cellj + 1]; ++idxj) {
          int j = cells.sorted_atoms[idxj];
          if (square_distance_device(coord, cells.coords + j * 3) > cutoff_sq)
            continue;
          for (int dzk = -1; dzk <= 1; ++dzk)
            for (int dyk = -1; dyk <= 1; ++dyk)
              for (int dxk = -1; dxk <= 1; ++dxk) {
                int cxk = oc.x + dxk, cyk = oc.y + dyk, czk = oc.z + dzk;
                if (cxk < 0 || cxk >= cells.dims.x || cyk < 0 || cyk >= cells.dims.y || czk < 0 ||
                    czk >= cells.dims.z)
                  continue;
                int cellk = cells.frame_base + (czk * cells.dims.y + cyk) * cells.dims.x + cxk;
                for (int idxk = cells.cell_start[cellk]; idxk < cells.cell_start[cellk + 1];
                     ++idxk) {
                  int k = cells.sorted_atoms[idxk];
                  if (k <= j) // count each pair once
                    continue;
                  if (square_distance_device(coord, cells.coords + k * 3) > cutoff_sq)
                    continue;
                  float dist_sq =
                      square_distance_device(cells.coords + j * 3, cells.coords + k * 3);
                  retval += weight_framei[j] * weight_framei[k] * sqrtf(dist_sq);
                  weight_sum += weight_framei[j] * weight_framei[k];
                }
              }
        }
      }
  return weight_sum == 0.0f ? 0.0f : retval / weight_sum;
}


/// @brief Distance from the observer to the weighted center of mass of atoms within
/// cutoff. Returns cutoff itself (a high-dispersion placeholder) if none are within range.
__device__ __forceinline__ float eccentricity_device(const float *coord, const ObserverCells &cells,
                                                     const float *weight_framei,
                                                     const float cutoff) {
  ComResult com = center_of_mass_device(coord, cells, weight_framei, cutoff);
  if (com.count == 0 || com.weight_sum == 0.0f)
    return cutoff;
  float mean_com[3] = {com.com[0] / com.weight_sum, com.com[1] / com.weight_sum,
                       com.com[2] / com.weight_sum};
  return sqrtf(square_distance_device(coord, mean_com));
}


/// @brief Radius of gyration of atoms within cutoff about their weighted center of mass.
__device__ __forceinline__ float radius_of_gyration_device(const float *coord,
                                                           const ObserverCells &cells,
                                                           const float *weight_framei,
                                                           const float cutoff) {
  ComResult com = center_of_mass_device(coord, cells, weight_framei, cutoff);
  if (com.count <= 1 || com.weight_sum == 0.0f)
    return 0.0f;
  float mean_com[3] = {com.com[0] / com.weight_sum, com.com[1] / com.weight_sum,
                       com.com[2] / com.weight_sum};
  float cutoff_sq = cutoff * cutoff;
  float retval = 0.0f;
  int3 oc = observer_cell_index(coord, cells);
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = oc.x + dx, cy = oc.y + dy, cz = oc.z + dz;
        if (cx < 0 || cx >= cells.dims.x || cy < 0 || cy >= cells.dims.y || cz < 0 ||
            cz >= cells.dims.z)
          continue;
        int cell = cells.frame_base + (cz * cells.dims.y + cy) * cells.dims.x + cx;
        for (int idx = cells.cell_start[cell]; idx < cells.cell_start[cell + 1]; ++idx) {
          int j = cells.sorted_atoms[idx];
          if (square_distance_device(coord, cells.coords + j * 3) > cutoff_sq)
            continue;
          float com_dist_sq = square_distance_device(mean_com, cells.coords + j * 3);
          retval += weight_framei[j] * com_dist_sq;
        }
      }
  return sqrtf(retval / com.weight_sum);
}


////////////////////////////////////////////////////////////////////////////////
// Observable dispatch
////////////////////////////////////////////////////////////////////////////////
// Routes to the observable named by TypeObs (constants.h has the codes).
// TypeObs is a compile-time template param, not a runtime switch, so each
// instantiation compiles just one branch -- run_marching_observer_frames
// picks the instantiation to launch from the runtime type_obs.
template <int TypeObs>
__device__ __forceinline__ float
make_observation_device(const float *coord, const ObserverCells &cells, const float *weight_framei,
                        const float cutoff) {
  if constexpr (TypeObs == 1) {
    return existence_device(coord, cells, cutoff);
  } else if constexpr (TypeObs == 2) {
    return direct_count_device(coord, cells, cutoff);
  } else if constexpr (TypeObs == 3) {
    return distinct_count_device(coord, cells, weight_framei, cutoff);
  } else if constexpr (TypeObs == 11) {
    return mean_distance_device(coord, cells, weight_framei, cutoff);
  } else if constexpr (TypeObs == 12) {
    return cumulative_weight_device(coord, cells, weight_framei, cutoff);
  } else if constexpr (TypeObs == 13) {
    return density_device(coord, cells, weight_framei, cutoff);
  } else if constexpr (TypeObs == 14) {
    return dispersion_device(coord, cells, weight_framei, cutoff);
  } else if constexpr (TypeObs == 15) {
    return eccentricity_device(coord, cells, weight_framei, cutoff);
  } else if constexpr (TypeObs == 16) {
    return radius_of_gyration_device(coord, cells, weight_framei, cutoff);
  } else {
    return 0.0f; // unsupported TypeObs
  }
}


// Computes the observable at one grid point. Templated on TypeObs; run_marching_observer_frames
// picks which instantiation to launch from the runtime type_obs.
template <int TypeObs>
__global__ void
marching_observer_global(float *mobs_ret, const float *coord_frame, const float *weight_frame,
                         const int *dims, const float spacing, const int frame_number,
                         const int atomnr, const float cutoff, const int *cell_start,
                         const int *sorted_atoms, int3 cell_dims, float3 cell_min, float cell_size,
                         int cell_count) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int frame_idx = blockIdx.y;
  unsigned int grid_size = dims[0] * dims[1] * dims[2];
  if (index >= grid_size || frame_idx >= static_cast<unsigned int>(frame_number))
    return;

  ObserverCells cells;
  cells.cell_start = cell_start;
  cells.sorted_atoms = sorted_atoms;
  cells.coords = coord_frame;
  cells.dims = cell_dims;
  cells.min = cell_min;
  cells.cell_size = cell_size;
  cells.cell_count = cell_count;
  cells.frame_base = static_cast<int>(frame_idx) * cell_count;

  float *frame_output = mobs_ret + static_cast<size_t>(frame_idx) * grid_size;

  // Get the coordinate of the grid point (Observer) in real space
  float coord[3] = {static_cast<float>(index / (dims[0] * dims[1])) * spacing,
                    static_cast<float>((index / dims[0]) % dims[1]) * spacing,
                    static_cast<float>(index % dims[0]) * spacing};

  frame_output[index] = make_observation_device<TypeObs>(coord, cells, weight_frame, cutoff);
}


namespace {

// Uploaded coords/weights/dims plus the cell index and raw output grid,
// shared by marching_observer_host and observe_frame_host.
struct MarchingObserverBuffers {
  float *output_device;
  float *coords_device;
  float *weights_device;
  int *dims_device;
  ObserverCellBuffers cell_buffers;
};


MarchingObserverBuffers run_marching_observer_frames(DeviceContext *ctx, cudaStream_t stream,
                                                     const float *coord, const float *weights,
                                                     const int *dims, float spacing,
                                                     int frame_number, int atomnr, float cutoff,
                                                     int type_obs, unsigned int observer_number,
                                                     BufferSlot output_slot) {
  unsigned int grid_size = (observer_number + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t frame_output_count = static_cast<size_t>(frame_number) * observer_number;
  size_t frame_atom_count = static_cast<size_t>(frame_number) * atomnr;

  MarchingObserverBuffers bufs;
  bufs.output_device = ctx->get_buffer_f(frame_output_count, static_cast<size_t>(output_slot));
  bufs.coords_device =
      ctx->get_buffer_f(frame_atom_count * 3, static_cast<size_t>(BufferSlot::COORDS));
  bufs.weights_device =
      ctx->get_buffer_f(frame_atom_count, static_cast<size_t>(BufferSlot::WEIGHTS));
  bufs.dims_device = ctx->get_buffer_i(3, static_cast<size_t>(BufferSlot::DIMS));

  CUDA_CHECK(cudaMemsetAsync(bufs.output_device, 0, frame_output_count * sizeof(float), stream));
  copy_h2d_async(ctx, bufs.coords_device, coord, frame_atom_count * 3 * sizeof(float),
                 BufferSlot::COORDS, stream);
  copy_h2d_async(ctx, bufs.weights_device, weights, frame_atom_count * sizeof(float),
                 BufferSlot::WEIGHTS, stream);
  copy_h2d_async(ctx, bufs.dims_device, dims, 3 * sizeof(int), BufferSlot::DIMS, stream);

  ObserverCellGrid cell_grid;
  bufs.cell_buffers = build_observer_cell_buffers(ctx, stream, bufs.coords_device, coord,
                                                  frame_number, atomnr, cutoff, cell_grid);

  // One launch covers every frame (blockIdx.y picks the frame); pick the TypeObs
  // instantiation to launch here, once, from the runtime type_obs.
#define LAUNCH_MARCHING_OBSERVER(TYPE)                                                             \
  marching_observer_global<TYPE><<<dim3(grid_size, frame_number, 1), BLOCK_SIZE, 0, stream>>>(     \
      bufs.output_device, bufs.coords_device, bufs.weights_device, bufs.dims_device, spacing,      \
      frame_number, atomnr, cutoff, bufs.cell_buffers.cell_start, bufs.cell_buffers.sorted_atoms,  \
      cell_grid.dims, cell_grid.min, cell_grid.cell_size, cell_grid.cell_count)
  switch (type_obs) {
  case 1:
    LAUNCH_MARCHING_OBSERVER(1);
    break;
  case 2:
    LAUNCH_MARCHING_OBSERVER(2);
    break;
  case 3:
    LAUNCH_MARCHING_OBSERVER(3);
    break;
  case 11:
    LAUNCH_MARCHING_OBSERVER(11);
    break;
  case 12:
    LAUNCH_MARCHING_OBSERVER(12);
    break;
  case 13:
    LAUNCH_MARCHING_OBSERVER(13);
    break;
  case 14:
    LAUNCH_MARCHING_OBSERVER(14);
    break;
  case 15:
    LAUNCH_MARCHING_OBSERVER(15);
    break;
  case 16:
    LAUNCH_MARCHING_OBSERVER(16);
    break;
  default:
    break; // unsupported type_obs: output_device stays zero-initialized
  }
#undef LAUNCH_MARCHING_OBSERVER
  CUDA_CHECK_KERNEL();

  return bufs;
}

} // namespace


/**
 * @brief GPU entry point for the marching observer algorithm on a frame slice.
 *
 * Launches marching_observer_global across frames so that every grid point
 * computes an observable (e.g. density, count, eccentricity) from the atoms
 * within cutoff. The per-frame grids are stored, then reduced across frames
 * with gridwise_aggregation_global. Uses the global DeviceContext when active.
 *
 * @param mobs_dynamics Host output grid of size dims[0]*dims[1]*dims[2].
 * @param coord Host coordinates with shape (frame_number, atom_per_frame, 3).
 * @param weights Host weights with shape (frame_number, atom_per_frame).
 * @param dims Grid dimensions [x, y, z].
 * @param spacing Grid spacing.
 * @param frame_number Number of frames in the slice.
 * @param atom_per_frame Number of atoms in each frame.
 * @param cutoff Observer cutoff distance.
 * @param type_obs Observable type (see constants.h).
 * @param type_agg Aggregation type across frames (see constants.h).
 */
void marching_observer_host(float *mobs_dynamics, const float *coord, const float *weights,
                            const int *dims, const float spacing, const int frame_number,
                            const int atom_per_frame, const float cutoff, const int type_obs,
                            const int type_agg) {
  check_frame_count(frame_number);
  unsigned int observer_number = dims[0] * dims[1] * dims[2];
  unsigned int grid_size = (observer_number + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Fall back to a function-local context (auto-freed on return) if no global one is active.
  DeviceContext *ctx = get_global_device_context();
  DeviceContext local_ctx;
  if (!ctx || !ctx->valid()) {
    local_ctx.init();
    ctx = &local_ctx;
  }
  cudaStream_t stream = ctx->stream();

  MarchingObserverBuffers bufs = run_marching_observer_frames(
      ctx, stream, coord, weights, dims, spacing, frame_number, atom_per_frame, cutoff, type_obs,
      observer_number, BufferSlot::TRAJ_DYNAMICS);

  // The resultant aggregated mobs feature
  float *tmp_mobs_gpu =
      ctx->get_buffer_f(observer_number, static_cast<size_t>(BufferSlot::TMP_GRID));

  // Perform frame-wise aggregation on the voxelized trajectory
  CUDA_CHECK(cudaMemsetAsync(tmp_mobs_gpu, 0, observer_number * sizeof(float), stream));
  gridwise_aggregation_global<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      bufs.output_device, tmp_mobs_gpu, frame_number, observer_number, type_agg);
  CUDA_CHECK_KERNEL();

  // No normalization here: dividing the finished grid by its own sum would erase
  // the magnitude the observables measure, and the signed aggregations (drift)
  // can sum to ~0. Scale the channels at training time instead.

  // Copy the final result to the host memory
  CUDA_CHECK(cudaMemcpyAsync(mobs_dynamics, tmp_mobs_gpu, observer_number * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));

  if (!ctx->pending()) // local_ctx is never pending, so this always fires for it
    ctx->synchronize();
}


/**
 * @brief In-place marching observer algorithm writing into a caller-provided GPU buffer.
 *
 * Computes the requested observable for each frame, stores per-frame results,
 * aggregates across frames, and writes the final grid into `output`. Input
 * buffers are taken from the DeviceContext when active.
 */
void marching_observer_host_into(float *output, const float *coord, const float *weights,
                                 const int *dims, const float spacing, const int frame_number,
                                 const int atom_per_frame, const float cutoff, const int type_obs,
                                 const int type_agg) {
  check_frame_count(frame_number);
  unsigned int observer_number = dims[0] * dims[1] * dims[2];
  unsigned int grid_size = (observer_number + BLOCK_SIZE - 1) / BLOCK_SIZE;

  DeviceContext *ctx = get_global_device_context();
  DeviceContext local_ctx;
  if (!ctx || !ctx->valid()) {
    local_ctx.init();
    ctx = &local_ctx;
  }
  cudaStream_t stream = ctx->stream();

  MarchingObserverBuffers bufs = run_marching_observer_frames(
      ctx, stream, coord, weights, dims, spacing, frame_number, atom_per_frame, cutoff, type_obs,
      observer_number, BufferSlot::TRAJ_DYNAMICS);

  // output is caller-provided (e.g. a DLPack tensor's device buffer), so the
  // aggregation kernel writes straight into it -- no extra copy needed.
  CUDA_CHECK(cudaMemsetAsync(output, 0, observer_number * sizeof(float), stream));
  gridwise_aggregation_global<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      bufs.output_device, output, frame_number, observer_number, type_agg);
  CUDA_CHECK_KERNEL();

  if (!ctx->pending())
    ctx->synchronize();
}


/**
 * @brief GPU entry point for a single-frame marching-observer observation.
 *
 * Uploads one frame of coordinates and weights, launches marching_observer_global
 * over all grid points, and copies the observed grid back to the host. Uses the
 * global DeviceContext when active.
 */
void observe_frame_host(float *results, const float *coord_frame, const float *weight_frame,
                        const int *dims, const float spacing, const int atomnr, const float cutoff,
                        const int type_obs) {
  unsigned int observer_number = dims[0] * dims[1] * dims[2];

  DeviceContext *ctx = get_global_device_context();
  DeviceContext local_ctx;
  if (!ctx || !ctx->valid()) {
    local_ctx.init();
    ctx = &local_ctx;
  }
  cudaStream_t stream = ctx->stream();

  MarchingObserverBuffers bufs = run_marching_observer_frames(
      ctx, stream, coord_frame, weight_frame, dims, spacing, /*frame_number=*/1, atomnr, cutoff,
      type_obs, observer_number, BufferSlot::OUTPUT_GRID);

  CUDA_CHECK(cudaMemcpyAsync(results, bufs.output_device, observer_number * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));

  if (!ctx->pending())
    ctx->synchronize();
}


/**
 * @brief In-place single-frame marching-observer observation.
 *
 * Computes the requested observable for one frame and writes the grid directly
 * into the caller-provided CUDA pointer `output`.
 */
void observe_frame_host_into(float *output, const float *coord_frame, const float *weight_frame,
                             const int *dims, const float spacing, const int atomnr,
                             const float cutoff, const int type_obs) {
  unsigned int observer_number = dims[0] * dims[1] * dims[2];

  DeviceContext *ctx = get_global_device_context();
  DeviceContext local_ctx;
  if (!ctx || !ctx->valid()) {
    local_ctx.init();
    ctx = &local_ctx;
  }
  cudaStream_t stream = ctx->stream();

  MarchingObserverBuffers bufs = run_marching_observer_frames(
      ctx, stream, coord_frame, weight_frame, dims, spacing, /*frame_number=*/1, atomnr, cutoff,
      type_obs, observer_number, BufferSlot::OUTPUT_GRID);

  // output is caller-provided (e.g. a DLPack tensor's device buffer).
  CUDA_CHECK(cudaMemcpyAsync(output, bufs.output_device, observer_number * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));

  if (!ctx->pending())
    ctx->synchronize();
}
