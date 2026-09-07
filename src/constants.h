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


// NOTE: The observable types are defined by OBSERVABLE_TYPE_LIST in marching_observers.cuh

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