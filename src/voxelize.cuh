// Created by: Yang Zhang 
// Description: Header for the CUDA implementation of the property density flow algorithm

#ifndef VOXELIZE_INCLUDE
#define VOXELIZE_INCLUDE

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

void voxelize_host_cpu(
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

#endif
