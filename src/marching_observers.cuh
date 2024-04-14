// This is the header file for the marching observer host function

// Hard-coded supported observable and aggregation types
#define OBSERVABLE_COUNT 9
#define SUPPORTED_OBSERVABLES {1, 2, 3, 11, 12, 13, 14, 15, 16}
#define AGGREGATION_COUNT 7
#define SUPPORTED_AGGREGATIONS {1, 2, 3, 4, 5, 6, 7}

// Direct Count-based Observables
// existence_device  1
// direct_count_device  2
// distinct_count  3

// Weight-based Observables
// mean_distance_device  11
// cumulative_weight_device  12
// density_device  13
// dispersion_device  14
// eccentricity_device  15
// radius_of_gyration_device  16

// Compile-time constants
#define MAX_FRAME_NUMBER 1000
#define DISTINCT_LIMIT 1000

void marching_observer_host(
  float *grid_return, const float *coord, const float *weights,
  const int *dims, const float spacing, 
  const int frame_number, const int atom_per_frame,
  const float cutoff, const int type_obs, const int type_agg
);
