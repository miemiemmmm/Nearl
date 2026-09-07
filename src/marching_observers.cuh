// Created by: Yang Zhang
// Description: Header for the CUDA implementation of the marching observers algorithm

#ifndef MARCHING_OBSERVERS_INCLUDE
#define MARCHING_OBSERVERS_INCLUDE

#include "gpuutils.cuh" // For AggregationType

/**
 * @brief The single source of truth for the supported observables.
 *
 * Every column is consumed somewhere, so a new observable is added by appending one line here:
 *   NAME  : the enumerator of ObservableType, also the name exposed to Python
 *   VALUE : the numeric value, kept stable for backwards compatibility of stored configurations
 *   FN    : the __device__ function implementing it, see marching_observers.cu. It must have the
 *           signature (coord, coord_frame, weight_frame, atomnr, cutoff); wrap it in
 *           marching_observers.cu if the underlying implementation ignores the weights.
 *
 * The list generates the enumeration below, the Python bindings in actions_py.cpp and the
 * runtime-to-template dispatch in marching_observers.cu.
 */
#define OBSERVABLE_TYPE_LIST(X)                                                                    \
  /* Particle count based observables */                                                           \
  X(EXISTENCE, 1, existence_device)                                                                \
  X(DIRECT_COUNT, 2, direct_count_device)                                                          \
  X(DISTINCT_COUNT, 3, distinct_count_device)                                                      \
  /* Weight based observables */                                                                   \
  X(MEAN_DISTANCE, 11, mean_distance_device)                                                       \
  X(CUMULATIVE_WEIGHT, 12, cumulative_weight_device)                                               \
  X(DENSITY, 13, density_device)                                                                   \
  X(DISPERSION, 14, dispersion_device)                                                             \
  X(ECCENTRICITY, 15, eccentricity_device)                                                         \
  X(RADIUS_OF_GYRATION, 16, radius_of_gyration_device)

/**
 * @brief The type of observable to be calculated by the marching observers algorithm.
 */
enum class ObservableType : int {
#define OBSERVABLE_TYPE_ENUMERATOR(NAME, VALUE, FN) NAME = VALUE,
  OBSERVABLE_TYPE_LIST(OBSERVABLE_TYPE_ENUMERATOR)
#undef OBSERVABLE_TYPE_ENUMERATOR
};

void marching_observer_host(float *mobs_dynamics, const float *coord, const float *weights,
                            const int *dims, const float spacing, const int frame_number,
                            const int atom_per_frame, const float cutoff,
                            const ObservableType type_obs, const AggregationType type_agg);

void observe_frame_host(float *results, const float *coord_frame, const float *weight_frame,
                        const int *dims, const float spacing, const int atomnr, const float cutoff,
                        const ObservableType type_obs);

#endif
