// Created by: Yang Zhang 
// Description: Header for the CUDA implementation of the marching observers algorithm 

#ifndef MARCHING_OBSERVERS_INCLUDE
#define MARCHING_OBSERVERS_INCLUDE

void marching_observer_host(
  float *mobs_dynamics, const float *coord, const float *weights, const int *dims, 
  const float spacing, const int frame_number, const int atom_per_frame,
  const float cutoff, const int type_obs, const int type_agg
);

void observe_frame_host(float *results, const float *coord_frame, const float *weight_frame, const int *dims, const float spacing, const int atomnr, const float cutoff, const int type_obs); 

#endif
