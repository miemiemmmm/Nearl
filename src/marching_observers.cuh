// This is the header file for the marching observer host function


void marching_observer_host(float *grid_return, const float *coord, 
  const int *dims, const float spacing, 
  const int frame_number, const int atom_per_frame,
  const float cutoff, const int type_obs, const int type_agg); 
