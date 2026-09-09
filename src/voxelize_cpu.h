// CPU reference for the voxelization kernels, built without nvcc.
//
// Used when the CUDA extension is unavailable, and as the oracle the GPU path is
// tested against. The math comes from voxelize_math.h, which the kernels share.

#ifndef NEARL_VOXELIZE_CPU_H
#define NEARL_VOXELIZE_CPU_H

// Single frame: Gaussian splat of every atom, each normalized to its own weight.
// `interpolated` is zeroed first and holds dims[0]*dims[1]*dims[2] floats.
void voxelize_host_cpu(float *interpolated, const float *coord, const float *weight,
                       const int *dims, const float spacing, const int atom_nr, const float cutoff,
                       const float sigma);

// Frame slice: voxelize each frame, then reduce the frame axis with `type_agg`.
// `voxelize_dynamics` holds one grid, not one per frame.
void trajectory_voxelization_host_cpu(float *voxelize_dynamics, const float *coord,
                                      const float *weight, const int *dims, const float spacing,
                                      const int frame_nr, const int atom_nr, const float cutoff,
                                      const float sigma, const int type_agg);

#endif // NEARL_VOXELIZE_CPU_H
