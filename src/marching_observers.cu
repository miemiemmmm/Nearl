// Created by: Yang Zhang
// Description: The CUDA implementation of the marching observer algorithm

#include <algorithm>

#include <iostream>

#include "constants.h"  // For hard-coded variables: BLOCK_SIZE, MAX_FRAME_NUMBER
#include "gpuutils.cuh" // For hard-coded BLOCK_SIZE and device functions: mean_device, mean_device, standard_deviation_device
#include "marching_observers.cuh"


////////////////////////////////////////////////////////////////////////////////
// Spatial cell index over the atoms
////////////////////////////////////////////////////////////////////////////////
// Atoms are bucketed into a uniform grid of cells with cell size >= cutoff.
// Every atom within cutoff of an observer then lies in one of the 27 cells
// surrounding the observer's cell, so each observer only inspects those
// instead of the whole frame. The per-(frame, cell) buckets are linked lists
// built once per call: cell_head[frame * cell_count + cell] holds the head
// atom index (or -1) and atom_next[frame * atomnr + atom] the next atom of
// the same cell (or -1). Padded placeholder atoms are never bucketed.
struct ObserverCells {
  const int *cell_head;
  const int *atom_next;
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


__global__ void build_observer_cell_lists(int *cell_head, int *atom_next, const float *coord,
                                          int atom_total, int atomnr, int3 cell_dims,
                                          float3 cell_min, float cell_size, int cell_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= atom_total)
    return;
  const float *c = coord + static_cast<size_t>(idx) * 3;
  if (c[0] == DEFAULT_COORD_PLACEHOLDER && c[1] == DEFAULT_COORD_PLACEHOLDER &&
      c[2] == DEFAULT_COORD_PLACEHOLDER)
    return;
  int cx = static_cast<int>(floorf((c[0] - cell_min.x) / cell_size));
  int cy = static_cast<int>(floorf((c[1] - cell_min.y) / cell_size));
  int cz = static_cast<int>(floorf((c[2] - cell_min.z) / cell_size));
  cx = min(max(cx, 0), cell_dims.x - 1);
  cy = min(max(cy, 0), cell_dims.y - 1);
  cz = min(max(cz, 0), cell_dims.z - 1);
  int cell = (idx / atomnr) * cell_count + (cz * cell_dims.y + cy) * cell_dims.x + cx;
  atom_next[idx] = atomicExch(&cell_head[cell], idx);
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
    for (int d = 0; d < 3; ++d) {
      if (!any || c[d] < lo[d])
        lo[d] = c[d];
      if (!any || c[d] > hi[d])
        hi[d] = c[d];
    }
    any = true;
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
    d[k] = std::max(1, static_cast<int>(std::ceil((hi[k] + cutoff - mins[k]) / grid.cell_size)) + 1);
  grid.dims = make_int3(d[0], d[1], d[2]);
  grid.cell_count = d[0] * d[1] * d[2];
  return grid;
}


struct ObserverCellBuffers {
  int *cell_head = nullptr;
  int *atom_next = nullptr;
};


// Allocates the cell buckets (context slots or per-call cudaMalloc), clears
// the heads and fills the lists from the uploaded coordinates.
ObserverCellBuffers build_observer_cell_buffers(DeviceContext *ctx, bool use_ctx,
                                                cudaStream_t stream, const float *coord_device,
                                                const float *coord_host, int frame_number,
                                                int atomnr, float cutoff,
                                                ObserverCellGrid &grid) {
  grid = compute_observer_cell_grid(coord_host, static_cast<size_t>(frame_number) * atomnr, cutoff);
  ObserverCellBuffers bufs;
  size_t head_count = std::max(static_cast<size_t>(frame_number) * grid.cell_count, size_t(1));
  size_t next_count = std::max(static_cast<size_t>(frame_number) * atomnr, size_t(1));
  if (use_ctx) {
    bufs.cell_head = ctx->get_buffer_i(head_count, static_cast<size_t>(BufferSlot::CELL_HEAD));
    bufs.atom_next = ctx->get_buffer_i(next_count, static_cast<size_t>(BufferSlot::ATOM_NEXT));
  } else {
    CUDA_CHECK(cudaMalloc(&bufs.cell_head, head_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bufs.atom_next, next_count * sizeof(int)));
  }
  CUDA_CHECK(cudaMemsetAsync(bufs.cell_head, -1, head_count * sizeof(int), stream));
  int atom_total = frame_number * atomnr;
  if (atom_total > 0) {
    int blocks = (atom_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    build_observer_cell_lists<<<blocks, BLOCK_SIZE, 0, stream>>>(
        bufs.cell_head, bufs.atom_next, coord_device, atom_total, atomnr, grid.dims, grid.min,
        grid.cell_size, grid.cell_count);
    CUDA_CHECK_KERNEL();
  }
  return bufs;
}


void free_observer_cell_buffers(bool use_ctx, const ObserverCellBuffers &bufs) {
  if (!use_ctx) {
    CUDA_CHECK(cudaFree(bufs.cell_head));
    CUDA_CHECK(cudaFree(bufs.atom_next));
  }
}

} // namespace


////////////////////////////////////////////////////////////////////////////////
// Direct count-based observables
////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Check the existence of particles in a frame
 *
 * Determines if any atom within a specified cutoff distance exists relative to the observer's
 * position. This function iterates over all atoms in a given frame and checks if at least one
 * atom exists within the cutoff distance from the reference coordinate.
 *
 * @param coord The reference coordinate of the observer
 * @param coord_framei The coordinates of all atoms in the frame
 * @param atomnr The number of atoms in the frame
 * @param cutoff The cutoff distance
 *
 * @return 1.0 if at least one atom exists within the cutoff distance, 0.0 otherwise
 *
 * @note This direct count-based observation does not consider atom weights.
 */
__device__ float existence_device(const float *coord, const ObserverCells &cells,
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
        for (int j = cells.cell_head[cell]; j != -1; j = cells.atom_next[j])
          if (square_distance_device(coord, cells.coords + j * 3) <= cutoff_sq)
            return 1.0f;
      }
  return 0.0f;
}


/**
 * @brief Count atoms within a specified cutoff from an observer.
 *
 * This CUDA device function iterates over all atoms in a specified frame and accumulates
 * a count of those within a given cutoff distance from a reference coordinate.
 *
 * @param coord Pointer to the reference coordinate's float array (x, y, z).
 * @param coord_framei Pointer to the frame's atom coordinates float array, with each
 *        atom's coordinates stored consecutively as (x, y, z).
 * @param atomnr The total number of atoms in the frame.
 * @param cutoff The distance threshold for counting an atom. Only atoms within this
 *        distance from the reference coordinate are counted.
 *
 * @return The count of atoms within the cutoff distance from the reference coordinate.
 *
 * @note This direct count-based observation does not consider atom weights.
 */
__device__ float direct_count_device(const float *coord, const ObserverCells &cells,
                                     const float cutoff) {
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
        for (int j = cells.cell_head[cell]; j != -1; j = cells.atom_next[j])
          if (square_distance_device(coord, cells.coords + j * 3) <= cutoff_sq)
            retval += 1.0f;
      }
  return retval;
}


/**
 * @brief Count distinct weighted atoms within a cutoff distance from an observer.
 *
 * This CUDA device function iterates over atoms in a frame, checking each atom's
 * distance to a reference coordinate. It counts atoms with unique weights within
 * the specified cutoff distance. Assumes a fixed-size array for tracking encountered
 * values, with a limit of 1000 (hard coded) distinct weights.
 *
 * @param coord Pointer to the reference coordinate's float array (x, y, z).
 * @param coord_framei Pointer to the frame's atom coordinates float array (x, y, z for each atom).
 * @param weight_framei Pointer to the frame's atom weights float array, one per atom.
 * @param atomnr Total number of atoms in the frame.
 * @param cutoff Distance threshold for including an atom in the count.
 *
 * @return The count of unique-weight atoms within the cutoff distance.
 *
 * @note This direct count-based observation considers atom weights as discrete identification of
 * the particles. It is suitable for discrete atomic identities such as atom indices, residue
 * indices, or other discrete identifiers. The continuous weights (float numbers) are acceptable but
 * will be rounded to the nearest integer after multiplying by 10 (To avoid floating-point
 * comparison issues).
 *
 */
__device__ float distinct_count_device(const float *coord, const ObserverCells &cells,
                                       const float *weight_framei, const float cutoff) {
  float cutoff_sq = cutoff * cutoff;
  float distinct_count = 0.0f;

  // Initialize the encountered values with a default placeholder
  float encountered_values[DISTINCT_LIMIT];
  for (int i = 0; i < DISTINCT_LIMIT; ++i) {
    encountered_values[i] = DEFAULT_PLACEHOLDER;
  }

  float val_check;
  int3 oc = observer_cell_index(coord, cells);
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = oc.x + dx, cy = oc.y + dy, cz = oc.z + dz;
        if (cx < 0 || cx >= cells.dims.x || cy < 0 || cy >= cells.dims.y || cz < 0 ||
            cz >= cells.dims.z)
          continue;
        int cell = cells.frame_base + (cz * cells.dims.y + cy) * cells.dims.x + cx;
        for (int j = cells.cell_head[cell]; j != -1; j = cells.atom_next[j]) {
          if (square_distance_device(coord, cells.coords + j * 3) > cutoff_sq)
            continue;

          val_check = weight_framei[j];
          // Linear search unique values with early termination and insertion
          for (int k = 0; k < DISTINCT_LIMIT; ++k) {
            if (val_check == encountered_values[k]) {
              break;
            } else if (encountered_values[k] == DEFAULT_PLACEHOLDER) {
              encountered_values[k] = val_check;
              distinct_count += 1;
              break;
            }
          }
        }
      }
  return distinct_count;
}


////////////////////////////////////////////////////////////////////////////////
// Weight-based observables
////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Weighted mean distance of particles in frame i
 */
__device__ float mean_distance_device(const float *coord, const ObserverCells &cells,
                                      const float *weight_framei, const float cutoff) {
  float cutoff_sq = cutoff * cutoff;

  float retval = 0.0;
  float weight_sum = 0.0;
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
        for (int j = cells.cell_head[cell]; j != -1; j = cells.atom_next[j]) {
          float dist_sq = square_distance_device(coord, cells.coords + j * 3);
          if (dist_sq > cutoff_sq)
            continue;

          retval += sqrt(dist_sq) * weight_framei[j];
          weight_sum += weight_framei[j];
          count += 1;
        }
      }
  if (count > 0) {
    retval = retval / weight_sum;
    return retval;
  } else {
    return 0.0f;
  }
}


/**
 * @brief Calculate the cumulative weight of particles within a specified cutoff distance.
 */
__device__ float cumulative_weight_device(const float *coord, const ObserverCells &cells,
                                         const float *weight_framei, const float cutoff) {
  float cutoff_sq = cutoff * cutoff;
  float retval = 0.0;
  int3 oc = observer_cell_index(coord, cells);
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = oc.x + dx, cy = oc.y + dy, cz = oc.z + dz;
        if (cx < 0 || cx >= cells.dims.x || cy < 0 || cy >= cells.dims.y || cz < 0 ||
            cz >= cells.dims.z)
          continue;
        int cell = cells.frame_base + (cz * cells.dims.y + cy) * cells.dims.x + cx;
        for (int j = cells.cell_head[cell]; j != -1; j = cells.atom_next[j])
          if (square_distance_device(coord, cells.coords + j * 3) <= cutoff_sq)
            retval += weight_framei[j];
      }
  return retval;
}


/**
 * @brief Calculates the density of particles within a specified cutoff radius from a given point.
 */
__device__ float density_device(const float *coord, const ObserverCells &cells,
                                const float *weight_framei, const float cutoff) {
  float weight_sum = cumulative_weight_device(coord, cells, weight_framei, cutoff);
  float volume = (4.0 / 3.0) * M_PI * cutoff * cutoff * cutoff;
  return weight_sum / volume;
}


/**
 * @brief Computes the weighted dispersion of particles within a specified cutoff distance.
 *
 * This device function calculates the dispersion of particles based on their pairwise distances
 * and weights. It considers only those pairs of particles that are within a given cutoff distance
 * from each other. The dispersion is computed as a weighted sum of the the pairwise distance,
 * normalized by the total weight of all considered particle pairs.
 *
 * Parameters are the same as for the other observables.
 *
 * @return The calculated weighted dispersion value. If the weight sum of considered particle pairs
 * is zero, the function returns 0.0, indicating no dispersion or an invalid state.
 *
 */
__device__ float dispersion_device(const float *coord, const ObserverCells &cells,
                                   const float *weight_framei, const float cutoff) {
  float dist_sq, dist_sq_j, dist_sq_k;
  float cutoff_sq = cutoff * cutoff;

  float weight_sum = 0.0f;
  float retval = 0.0f;
  // Pairwise distance summation over the neighbourhood atoms; the k > j index
  // check enumerates each pair once regardless of the linked-list order.
  int3 oc = observer_cell_index(coord, cells);
  for (int dzj = -1; dzj <= 1; ++dzj)
    for (int dyj = -1; dyj <= 1; ++dyj)
      for (int dxj = -1; dxj <= 1; ++dxj) {
        int cxj = oc.x + dxj, cyj = oc.y + dyj, czj = oc.z + dzj;
        if (cxj < 0 || cxj >= cells.dims.x || cyj < 0 || cyj >= cells.dims.y || czj < 0 ||
            czj >= cells.dims.z)
          continue;
        int cellj = cells.frame_base + (czj * cells.dims.y + cyj) * cells.dims.x + cxj;
        for (int j = cells.cell_head[cellj]; j != -1; j = cells.atom_next[j]) {
          // Filter atom 1 outside the cutoff distance
          dist_sq_j = square_distance_device(coord, cells.coords + j * 3);
          if (dist_sq_j > cutoff_sq)
            continue;

          for (int dzk = -1; dzk <= 1; ++dzk)
            for (int dyk = -1; dyk <= 1; ++dyk)
              for (int dxk = -1; dxk <= 1; ++dxk) {
                int cxk = oc.x + dxk, cyk = oc.y + dyk, czk = oc.z + dzk;
                if (cxk < 0 || cxk >= cells.dims.x || cyk < 0 || cyk >= cells.dims.y || czk < 0 ||
                    czk >= cells.dims.z)
                  continue;
                int cellk = cells.frame_base + (czk * cells.dims.y + cyk) * cells.dims.x + cxk;
                for (int k = cells.cell_head[cellk]; k != -1; k = cells.atom_next[k]) {
                  if (k <= j)
                    continue;

                  // Filter atom 2 outside the cutoff distance
                  dist_sq_k = square_distance_device(coord, cells.coords + k * 3);
                  if (dist_sq_k > cutoff_sq)
                    continue;

                  dist_sq = square_distance_device(cells.coords + j * 3, cells.coords + k * 3);
                  // Need to clarify what is the formula it follows to characterize the dispersion
                  retval += weight_framei[j] * weight_framei[k] * sqrt(dist_sq);
                  weight_sum += weight_framei[j] * weight_framei[k];
                }
              }
        }
      }
  if (weight_sum == 0.0) {
    return 0.0f;
  } else {
    return retval / weight_sum;
  }
}


/**
 * @brief Compute the distance between center of mass to the observer
 *
 * This observable is inspired by the eccentricity of cable (distance between
 * the center of the conductor and the center of the insulation). It computes
 * the weighted center of mass (COM) for a collection of particles within a
 * specified cutoff distance from a reference point (observer) and then calculates
 * the distance from this COM to the reference point. This function can be used
 * to assess the dispersion or distribution of particles around the observer in
 * a three-dimensional space.
 *
 * Parameters are the same as for the other observables.
 *
 * @return The distance between the weighted center of mass of the particles (within
 * cutoff) and the observer. If no particles are within the cutoff distance, returns
 * the cutoff distance as an indication of high dispersion.
 *
 */
__device__ float eccentricity_device(const float *coord, const ObserverCells &cells,
                                     const float *weight_framei, const float cutoff) {
  float dist_sq;
  float cutoff_sq = cutoff * cutoff;

  float com[3] = {0.0, 0.0, 0.0};
  float weight_sum = 0.0f;
  int count = 0;
  // Get the center of mass
  int3 oc = observer_cell_index(coord, cells);
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = oc.x + dx, cy = oc.y + dy, cz = oc.z + dz;
        if (cx < 0 || cx >= cells.dims.x || cy < 0 || cy >= cells.dims.y || cz < 0 ||
            cz >= cells.dims.z)
          continue;
        int cell = cells.frame_base + (cz * cells.dims.y + cy) * cells.dims.x + cx;
        for (int j = cells.cell_head[cell]; j != -1; j = cells.atom_next[j]) {
          dist_sq = square_distance_device(coord, cells.coords + j * 3);
          if (dist_sq > cutoff_sq)
            continue;
          count += 1;
          com[0] += weight_framei[j] * cells.coords[j * 3];
          com[1] += weight_framei[j] * cells.coords[j * 3 + 1];
          com[2] += weight_framei[j] * cells.coords[j * 3 + 2];
          weight_sum += weight_framei[j];
        }
      }

  if (count > 0 && weight_sum != 0) {
    com[0] = com[0] / weight_sum;
    com[1] = com[1] / weight_sum;
    com[2] = com[2] / weight_sum;
  } else {
    return cutoff; // Return the cutoff distance if no particles within the cutoff
  }

  float retval = sqrt(square_distance_device(coord, com));
  return retval;
}


/**
 * @brief Radius of gyration of particles within a specified cutoff distance.
 *
 * This function calculates the radius of gyration for a collection of particles, considering only
 * those within a specified cutoff distance from a given reference point (`coord`).
 *
 * @result The radius of gyration of the particles within the cutoff distance from the reference
 * point, calculated relative to their center of mass. If no particles are within the cutoff
 * distance, the function returns 0.0.
 *
 * @note The signs of weights should be geater than 0 (otherwise Center of Mass will be wrong)
 */
__device__ float radius_of_gyration_device(const float *coord, const ObserverCells &cells,
                                           const float *weight_framei, const float cutoff) {
  float dist_sq, cutoff_sq = cutoff * cutoff;

  // Calculate the center of mass
  float com[3] = {0.0, 0.0, 0.0};
  float weight_sum = 0.0f;
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
        for (int j = cells.cell_head[cell]; j != -1; j = cells.atom_next[j]) {
          // Filter atoms outside of the cutoff distance
          dist_sq = square_distance_device(coord, cells.coords + j * 3);
          if (dist_sq > cutoff_sq)
            continue;

          com[0] += weight_framei[j] * cells.coords[j * 3];
          com[1] += weight_framei[j] * cells.coords[j * 3 + 1];
          com[2] += weight_framei[j] * cells.coords[j * 3 + 2];
          weight_sum += weight_framei[j];
          count += 1;
        }
      }

  if (count <= 1) {
    // If no particle or only 1 particle in the cutoff distance, return 0.0
    return 0.0f;
  } else if (weight_sum != 0) {
    com[0] /= weight_sum;
    com[1] /= weight_sum;
    com[2] /= weight_sum;
  } else {
    return 0.0f;
  }

  // Calculate the radius of gyration to the center of mass
  float retval = 0.0f;
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = oc.x + dx, cy = oc.y + dy, cz = oc.z + dz;
        if (cx < 0 || cx >= cells.dims.x || cy < 0 || cy >= cells.dims.y || cz < 0 ||
            cz >= cells.dims.z)
          continue;
        int cell = cells.frame_base + (cz * cells.dims.y + cy) * cells.dims.x + cx;
        for (int j = cells.cell_head[cell]; j != -1; j = cells.atom_next[j]) {
          dist_sq = square_distance_device(coord, cells.coords + j * 3);
          if (dist_sq > cutoff_sq)
            continue;

          dist_sq = square_distance_device(com, cells.coords + j * 3);
          retval += weight_framei[j] * dist_sq;
        }
      }
  return sqrt(retval / weight_sum);
}


////////////////////////////////////////////////////////////////////////////////
// Direct particle count-based observables
////////////////////////////////////////////////////////////////////////////////
/**
 * @brief The device function to calculate the observable in frame i
 */
__device__ float make_observation_device(const float *coord, const ObserverCells &cells,
                                         const float cutoff, const int type_obs) {
  // Does not consider the weight of the particles
  float ret_framei = 0.0f;
  if (type_obs == 1) {
    ret_framei = existence_device(coord, cells, cutoff);
  } else if (type_obs == 2) {
    ret_framei = direct_count_device(coord, cells, cutoff);
  }
  return ret_framei;
}


/**
 * @brief The device function to calculate the observable in frame i
 */
__device__ float make_observation_device(const float *coord, const ObserverCells &cells,
                                         const float *weight_framei, const float cutoff,
                                         const int type_obs) {
  float ret_framei = 0.0f;
  if (type_obs == 3) {
    ret_framei = distinct_count_device(coord, cells, weight_framei, cutoff);
  } else if (type_obs == 11) {
    ret_framei = mean_distance_device(coord, cells, weight_framei, cutoff);
  } else if (type_obs == 12) {
    ret_framei = cumulative_weight_device(coord, cells, weight_framei, cutoff);
  } else if (type_obs == 13) {
    ret_framei = density_device(coord, cells, weight_framei, cutoff);
  } else if (type_obs == 14) {
    ret_framei = dispersion_device(coord, cells, weight_framei, cutoff);
  } else if (type_obs == 15) {
    ret_framei = eccentricity_device(coord, cells, weight_framei, cutoff);
  } else if (type_obs == 16) {
    ret_framei = radius_of_gyration_device(coord, cells, weight_framei, cutoff);
  }
  return ret_framei;
}


/**
 * @brief The global kernel function to calculate the observable in a grid point
 */
__global__ void marching_observer_global(float *mobs_ret, const float *coord_frame,
                                         const float *weight_frame, const int *dims,
                                         const float spacing, const int frame_number,
                                         const int atomnr, const float cutoff,
                                         const int type_observable, const int *cell_head,
                                         const int *atom_next, int3 cell_dims, float3 cell_min,
                                         float cell_size, int cell_count) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int frame_idx = blockIdx.y;
  unsigned int grid_size = dims[0] * dims[1] * dims[2];
  if (index >= grid_size || frame_idx >= static_cast<unsigned int>(frame_number))
    return;

  ObserverCells cells;
  cells.cell_head = cell_head;
  cells.atom_next = atom_next;
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

  // Calculate the observable of grid point at index in the given frame
  if ((type_observable == 1) || (type_observable == 2)) {
    // Hard-coded for the direct count-based observables
    frame_output[index] = make_observation_device(coord, cells, cutoff, type_observable);
  } else {
    frame_output[index] =
        make_observation_device(coord, cells, weight_frame, cutoff, type_observable);
  }
}


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

  DeviceContext *ctx = get_global_device_context();
  const bool use_ctx = ctx && ctx->valid();
  cudaStream_t stream = use_ctx ? ctx->stream() : 0;

  // The (frame_number, observer_number) observed trajectory
  float *mobs_traj;
  // The resultant aggregated mobs feature (Initialize all digits in the return array to 0)
  float *tmp_mobs_gpu;
  // The atomic coordinates and weights of the frame i in the device memory
  float *coords_device;
  float *weights_device;
  int *dims_device;

  if (use_ctx) {
    mobs_traj = ctx->get_buffer_f(static_cast<size_t>(frame_number) * observer_number,
                                  static_cast<size_t>(BufferSlot::TRAJ_DYNAMICS));
    tmp_mobs_gpu = ctx->get_buffer_f(observer_number, static_cast<size_t>(BufferSlot::TMP_GRID));
    coords_device = ctx->get_buffer_f(static_cast<size_t>(frame_number) * atom_per_frame * 3,
                                      static_cast<size_t>(BufferSlot::COORDS));
    weights_device = ctx->get_buffer_f(static_cast<size_t>(frame_number) * atom_per_frame,
                                       static_cast<size_t>(BufferSlot::WEIGHTS));
    dims_device = ctx->get_buffer_i(3, static_cast<size_t>(BufferSlot::DIMS));
  } else {
    CUDA_CHECK(cudaMalloc(&mobs_traj, frame_number * observer_number * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp_mobs_gpu, observer_number * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&coords_device, frame_number * atom_per_frame * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weights_device, frame_number * atom_per_frame * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dims_device, 3 * sizeof(int)));
  }

  CUDA_CHECK(cudaMemsetAsync(mobs_traj, 0, frame_number * observer_number * sizeof(float), stream));
  CUDA_CHECK(cudaMemsetAsync(tmp_mobs_gpu, 0, observer_number * sizeof(float), stream));
  copy_h2d_async(ctx, coords_device, coord, frame_number * atom_per_frame * 3 * sizeof(float),
                 BufferSlot::COORDS, stream);
  copy_h2d_async(ctx, weights_device, weights, frame_number * atom_per_frame * sizeof(float),
                 BufferSlot::WEIGHTS, stream);
  copy_h2d_async(ctx, dims_device, dims, 3 * sizeof(int), BufferSlot::DIMS, stream);

  ObserverCellGrid cell_grid;
  ObserverCellBuffers cell_buffers = build_observer_cell_buffers(
      ctx, use_ctx, stream, coords_device, coord, frame_number, atom_per_frame, cutoff, cell_grid);

  // Process every frame in one launch; blockIdx.y selects the frame.
  marching_observer_global<<<dim3(grid_size, frame_number, 1), BLOCK_SIZE, 0, stream>>>(
      mobs_traj, coords_device, weights_device, dims_device, spacing, frame_number, atom_per_frame,
      cutoff, type_obs, cell_buffers.cell_head, cell_buffers.atom_next, cell_grid.dims,
      cell_grid.min, cell_grid.cell_size, cell_grid.cell_count);
  CUDA_CHECK_KERNEL();

  // Perform frame-wise aggregation on the voxelized trajectory
  CUDA_CHECK(cudaMemsetAsync(tmp_mobs_gpu, 0, observer_number * sizeof(float), stream));
  gridwise_aggregation_global<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      mobs_traj, tmp_mobs_gpu, frame_number, observer_number, type_agg);
  CUDA_CHECK_KERNEL();

  // No normalization here: dividing the finished grid by its own sum would erase
  // the magnitude the observables measure, and the signed aggregations (drift)
  // can sum to ~0. Scale the channels at training time instead.

  // Copy the final result to the host memory
  CUDA_CHECK(cudaMemcpyAsync(mobs_dynamics, tmp_mobs_gpu, observer_number * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));

  if (use_ctx) {
    if (!ctx->pending())
      ctx->synchronize();
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(mobs_traj));
    CUDA_CHECK(cudaFree(tmp_mobs_gpu));
    CUDA_CHECK(cudaFree(coords_device));
    CUDA_CHECK(cudaFree(weights_device));
    CUDA_CHECK(cudaFree(dims_device));
    free_observer_cell_buffers(use_ctx, cell_buffers);
  }
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
  const bool use_ctx = ctx && ctx->valid();
  cudaStream_t stream = use_ctx ? ctx->stream() : 0;

  float *mobs_traj;
  float *coords_device;
  float *weights_device;
  int *dims_device;

  if (use_ctx) {
    mobs_traj = ctx->get_buffer_f(static_cast<size_t>(frame_number) * observer_number,
                                  static_cast<size_t>(BufferSlot::TRAJ_DYNAMICS));
    coords_device = ctx->get_buffer_f(static_cast<size_t>(frame_number) * atom_per_frame * 3,
                                      static_cast<size_t>(BufferSlot::COORDS));
    weights_device = ctx->get_buffer_f(static_cast<size_t>(frame_number) * atom_per_frame,
                                       static_cast<size_t>(BufferSlot::WEIGHTS));
    dims_device = ctx->get_buffer_i(3, static_cast<size_t>(BufferSlot::DIMS));
  } else {
    CUDA_CHECK(cudaMalloc(&mobs_traj, frame_number * observer_number * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&coords_device, frame_number * atom_per_frame * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weights_device, frame_number * atom_per_frame * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dims_device, 3 * sizeof(int)));
  }

  CUDA_CHECK(cudaMemsetAsync(mobs_traj, 0, frame_number * observer_number * sizeof(float), stream));
  CUDA_CHECK(cudaMemcpyAsync(coords_device, coord,
                             frame_number * atom_per_frame * 3 * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(weights_device, weights, frame_number * atom_per_frame * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dims_device, dims, 3 * sizeof(int), cudaMemcpyHostToDevice, stream));

  // One launch for the whole slice; blockIdx.y selects the frame and the kernel
  // writes straight into its own slot of mobs_traj, so the per-frame staging
  // buffer and its device-to-device copy are both unnecessary.
  if (frame_number > 0)
    marching_observer_global<<<dim3(grid_size, frame_number, 1), BLOCK_SIZE, 0, stream>>>(
        mobs_traj, coords_device, weights_device, dims_device, spacing, frame_number,
        atom_per_frame, cutoff, type_obs);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemsetAsync(output, 0, observer_number * sizeof(float), stream));
  gridwise_aggregation_global<<<grid_size, BLOCK_SIZE, 0, stream>>>(mobs_traj, output, frame_number,
                                                                    observer_number, type_agg);
  CUDA_CHECK_KERNEL();

  if (use_ctx) {
    ctx->synchronize();
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(mobs_traj));
    CUDA_CHECK(cudaFree(coords_device));
    CUDA_CHECK(cudaFree(weights_device));
    CUDA_CHECK(cudaFree(dims_device));
  }
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
  unsigned int grid_size = (observer_number + BLOCK_SIZE - 1) / BLOCK_SIZE;

  int frame_nr = 1;

  DeviceContext *ctx = get_global_device_context();
  const bool use_ctx = ctx && ctx->valid();
  cudaStream_t stream = use_ctx ? ctx->stream() : 0;

  float *results_gpu;
  int *dims_gpu;
  float *coord_frame_gpu;
  float *weight_frame_gpu;
  if (use_ctx) {
    results_gpu = ctx->get_buffer_f(observer_number, static_cast<size_t>(BufferSlot::OUTPUT_GRID));
    dims_gpu = ctx->get_buffer_i(3, static_cast<size_t>(BufferSlot::DIMS));
    coord_frame_gpu = ctx->get_buffer_f(atomnr * 3, static_cast<size_t>(BufferSlot::COORDS));
    weight_frame_gpu = ctx->get_buffer_f(atomnr, static_cast<size_t>(BufferSlot::WEIGHTS));
  } else {
    CUDA_CHECK(cudaMalloc(&results_gpu, frame_nr * observer_number * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dims_gpu, 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&coord_frame_gpu, frame_nr * atomnr * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight_frame_gpu, frame_nr * atomnr * sizeof(float)));
  }

  CUDA_CHECK(
      cudaMemsetAsync(results_gpu, 0.0f, frame_nr * observer_number * sizeof(float), stream));
  copy_h2d_async(ctx, dims_gpu, dims, 3 * sizeof(int), BufferSlot::DIMS, stream);
  copy_h2d_async(ctx, coord_frame_gpu, coord_frame, frame_nr * atomnr * 3 * sizeof(float),
                 BufferSlot::COORDS, stream);
  copy_h2d_async(ctx, weight_frame_gpu, weight_frame, frame_nr * atomnr * sizeof(float),
                 BufferSlot::WEIGHTS, stream);

  ObserverCellGrid cell_grid;
  ObserverCellBuffers cell_buffers = build_observer_cell_buffers(
      ctx, use_ctx, stream, coord_frame_gpu, coord_frame, frame_nr, atomnr, cutoff, cell_grid);

  marching_observer_global<<<dim3(grid_size, frame_nr, 1), BLOCK_SIZE, 0, stream>>>(
      results_gpu, coord_frame_gpu, weight_frame_gpu, dims_gpu, spacing, frame_nr, atomnr, cutoff,
      type_obs, cell_buffers.cell_head, cell_buffers.atom_next, cell_grid.dims, cell_grid.min,
      cell_grid.cell_size, cell_grid.cell_count);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpyAsync(results, results_gpu, observer_number * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));

  if (use_ctx) {
    if (!ctx->pending())
      ctx->synchronize();
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(results_gpu));
    CUDA_CHECK(cudaFree(dims_gpu));
    CUDA_CHECK(cudaFree(coord_frame_gpu));
    CUDA_CHECK(cudaFree(weight_frame_gpu));
    free_observer_cell_buffers(use_ctx, cell_buffers);
  }
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
  unsigned int grid_size = (observer_number + BLOCK_SIZE - 1) / BLOCK_SIZE;

  int frame_nr = 1;

  DeviceContext *ctx = get_global_device_context();
  const bool use_ctx = ctx && ctx->valid();
  cudaStream_t stream = use_ctx ? ctx->stream() : 0;

  int *dims_gpu;
  float *coord_frame_gpu;
  float *weight_frame_gpu;
  if (use_ctx) {
    dims_gpu = ctx->get_buffer_i(3, static_cast<size_t>(BufferSlot::DIMS));
    coord_frame_gpu = ctx->get_buffer_f(atomnr * 3, static_cast<size_t>(BufferSlot::COORDS));
    weight_frame_gpu = ctx->get_buffer_f(atomnr, static_cast<size_t>(BufferSlot::WEIGHTS));
  } else {
    CUDA_CHECK(cudaMalloc(&dims_gpu, 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&coord_frame_gpu, frame_nr * atomnr * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight_frame_gpu, frame_nr * atomnr * sizeof(float)));
  }

  CUDA_CHECK(cudaMemsetAsync(output, 0.0f, frame_nr * observer_number * sizeof(float), stream));
  CUDA_CHECK(cudaMemcpyAsync(dims_gpu, dims, 3 * sizeof(int), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(coord_frame_gpu, coord_frame, frame_nr * atomnr * 3 * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(weight_frame_gpu, weight_frame, frame_nr * atomnr * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  marching_observer_global<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      output, coord_frame_gpu, weight_frame_gpu, dims_gpu, spacing, frame_nr, atomnr, cutoff,
      type_obs);
  CUDA_CHECK_KERNEL();

  if (use_ctx) {
    ctx->synchronize();
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dims_gpu));
    CUDA_CHECK(cudaFree(coord_frame_gpu));
    CUDA_CHECK(cudaFree(weight_frame_gpu));
  }
}
