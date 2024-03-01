import os

import numpy as np 


import nearl




tops = ["/tmp/ALIGN_END.pdb"]
trajs = ["/tmp/ALIGN_traj.nc"]


# Load trajectories 
# User is responsible for the trajectory preprocessing e.g. alignment, concatenation, etc.
FEATURIZER_PARMS = {
  "dims": [32, 32, 32],
  "boxsize": 16.0,
  "cutoff": 8.0,
  "sigma": 1.5
}

feat = nearl.features.Featurizer3D(FEATURIZER_PARMS)


# Register the dynamic feature 


# Setup featurization parameters



# Find the way to locate the points of interest if the reference frame 
# Anchor by position (COM of something)



# Interate the frames and compute the dynamic feature



# Dump the dynamic feature as well as the labels to the disk




