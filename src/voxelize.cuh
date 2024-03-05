void voxelize_host(
  float *interpolated, 
  const float *coord, 
  const float *weight, 
  const int *dims, 
  const int atom_nr, 
  const float spacing, 
  const float cutoff, 
  const float sigma
);

void trajectory_voxelization_host(
  float *voxelize_dynamics, 
  const float *coord, 
  const float *weight, 
  const int *dims, 
  const int frame_nr, 
  const int atom_nr, 
  const int interval, 
  const float spacing,
  const float cutoff, const float sigma
);



