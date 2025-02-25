// Created by: Yang Zhang
// Description: Constants used in the program

#ifndef CONSTANTS_INCLUDED
#define CONSTANTS_INCLUDED

// Miscellaneous Constants
#define BLOCK_SIZE 256
#define DEFAULT_COORD_PLACEHOLDER 99999.0f
#define DEFAULT_PLACEHOLDER 99999.0f
#define MAX_FRAME_NUMBER 512
#define DISTINCT_LIMIT 1000


// Obervable types
#define OBSERVABLE_COUNT 9
#define SUPPORTED_OBSERVABLES {1, 2, 3, 11, 12, 13, 14, 15, 16}

//////////////////////////////////////////////////////////////////////
// Hard-coded supported observable and aggregation types
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
//////////////////////////////////////////////////////////////////////

// Aggregation types
#define AGGREGATION_COUNT 8
#define SUPPORTED_AGGREGATIONS {1, 2, 3, 4, 5, 6, 7, 8}
#define INFORMATION_ENTROPY_BINS 16

//////////////////////////////////////////////////////////////////////
// mean  1
// standard_deviation 2
// median  3
// variance 4
// max 5
// min 6
// information_entropy 7
// drift 8 
//////////////////////////////////////////////////////////////////////

#endif 