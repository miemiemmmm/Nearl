#ifndef MARCHING_OBSERVERS_INCLUDE
#define MARCHING_OBSERVERS_INCLUDE

void marching_observer_host(
  float *grid_return, const float *coord, const float *weights,
  const int *dims, const float spacing, 
  const int frame_number, const int atom_per_frame,
  const float cutoff, const int type_obs, const int type_agg
);

#endif
