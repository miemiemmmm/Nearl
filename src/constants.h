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

// MATH
#define SQRT_2_PI 2.5066282746310002f


// NOTE: The observable types are defined by OBSERVABLE_TYPE_LIST in marching_observers.cuh

// NOTE: The aggregation types are defined by AGGREGATION_TYPE_LIST in gpuutils.cuh
#define INFORMATION_ENTROPY_BINS 16

#endif