// Function for voxelization of atomic coordinates and weights

#define DEFAULT_COORD_PLACEHOLDER 99999.0f

void voxelize_host(
  float *interpolated, 
  const float *coord, 
  const float *weight, 
  const int *dims, 
  const float spacing, 
  const int atom_nr, 
  const float cutoff, 
  const float sigma
);

void trajectory_voxelization_host(
  float *voxelize_dynamics, 
  const float *coord, 
  const float *weight, 
  const int *dims, 
  const float spacing,
  const int frame_nr, 
  const int atom_nr, 
  const float cutoff, 
  const float sigma,
  const int type_agg
);

