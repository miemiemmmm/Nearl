from BetaPose import  features, utils
from BetaPose import trajloader, data_io
from BetaPose import features, featurizer

import open3d as o3d
import time, builtins, tempfile, datetime, os
from BetaPose import utils, chemtools, representations;

import pytraj as pt
import numpy as np
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance_matrix

from BetaPose import utils, cluster

FEATURIZER_PARMS = {
  # Mask of components
  "MASK_INTEREST": ":LIG,MDL",
  "MASK_ENVIRONMENT": ":1-221",
  # POCKET SETTINGS
  "VOXEL_DIMENSION": [12, 12, 12],  # Unit: 1 (Number of lattice in one dimension)
  "CUBOID_LENGTH": [8, 8, 8],  # Unit: Angstorm (Need scaling)
  # SEARCH SETTINGS
  "UPDATE_INTERVAL": 1,
  "CUTOFF": 18,
}

# Load multiple trajectories
trajs = "/home/yzhang/zhang/MyTrajs/BFL-1/batch3/C209CsDJQucZ_job_001_traj.nc%/home/yzhang/zhang/MyTrajs/BFL-1/batch3/C209CsDJQucZ_job_002_traj.nc%/home/yzhang/zhang/MyTrajs/BFL-1/batch3/C209CsDJQucZ_job_003_traj.nc%/home/yzhang/zhang/MyTrajs/BFL-1/batch3/C209CsDJQucZ_job_004_traj.nc%/home/yzhang/zhang/MyTrajs/BFL-1/batch3/C209CsDJQucZ_job_005_traj.nc%/home/yzhang/zhang/MyTrajs/BFL-1/batch3/C209CsDJQucZ_job_006_traj.nc%/home/yzhang/zhang/MyTrajs/BFL-1/batch3/C209CsDJQucZ_job_007_traj.nc%/home/yzhang/zhang/MyTrajs/BFL-1/batch3/C209CsDJQucZ_job_008_traj.nc%/home/yzhang/zhang/MyTrajs/BFL-1/batch3/C209CsDJQucZ_job_009_traj.nc%/home/yzhang/zhang/MyTrajs/BFL-1/batch3/C209CsDJQucZ_job_010_traj.nc%"
trajs = trajs.strip("%").split("%")
tops = ["/home/yzhang/zhang/MyTrajs/BFL-1/batch3/C209CsDJQucZ_job_010_END.pdb"] * 10
tloader = trajloader.TrajectoryLoader(trajs, tops);
trajectories = trajloader.TrajectoryLoader(trajs, tops);

for traj in trajectories:
  # Complete the trajectory information
  traj.strip(":T3P");
  print("######################", traj.traj)
  #### TODO traj.addcharge();

  # Initialize the featurizer since different trajectory might have distinct parameters
  featurizer = featurizer.Featurizer3D(FEATURIZER_PARMS);
  feature_mass = features.MassFeature();
  # NOTE: in this step, the feature hooks back the feature and could access the featurizer by feat.featurer
  featurizer.register_feature(feature_mass)

  repr_traji, fpfh_traji, features_traji = featurizer.run(traj, range(1, 50, 10), traj.top.select("@CA&:50-70"))

  print(repr_traji.shape)
  print(fpfh_traji.shape)
  print(features_traji.shape)
  #   featurizer.dump("repr_form", repr_traji, "/tmp/test.h5");
  #   featurizer.dump("FPFH", np.array([(0,d) for d in fpfh_traji], dtype=object), "/tmp/test.h5");
  break

import pickle

thedict = {
  "repr": repr_traji,
  "fpfh": fpfh_traji,
  "features": features_traji
}

with open('/tmp/test.pkl', 'wb') as f:
  # Write the object to the file
  pickle.dump(thedict, f)







