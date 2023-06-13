import open3d as o3d
import time, builtins, tempfile, datetime, os
import pickle
from BetaPose import utils, chemtools, representations; 

import pytraj as pt 
import numpy as np 
from scipy.stats import entropy 
from scipy.ndimage import gaussian_filter 
from scipy.spatial import distance_matrix 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from BetaPose import utils
from BetaPose import trajloader, data_io
from BetaPose import features, featurizer



if __name__ == "__main__":
  print("Current working directory: ", os.getcwd())
  if os.path.exists("TEMP_DATA.pkl"):
    os.remove("TEMP_DATA.pkl")

  FEATURIZER_PARMS = {
    # Mask of components
    "MASK_INTEREST" : ":LIG,MDL",
    "MASK_ENVIRONMENT" : ":1-221",

    # POCKET SETTINGS
    "VOXEL_DIMENSION" : [12, 12, 12],    # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH" : [8,8,8],           # Unit: Angstorm (Need scaling)

    # SEARCH SETTINGS
    "UPDATE_INTERVAL" : 1,
    "CUTOFF": 18,
  }

  # Load multiple trajectories
  trajs = "/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_001_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_002_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_003_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_004_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_005_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_006_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_007_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_009_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_010_traj.nc%"
  tops = ["/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_END.pdb"] * 10
  trajs = trajs.strip("%").split("%")
  trajectories = trajloader.TrajectoryLoader(trajs, tops);

  for trajfile, topfile in zip(trajs, tops):
    # Complete the trajectory information
    st = time.perf_counter();
    # if traj.top.select(":T3P").__len__() > 0:
    #   traj.strip(":T3P")
    # Initialize the featurizer since different trajectory might have distinct parameters
    feat  = featurizer.Featurizer3D(FEATURIZER_PARMS);
    feature_mass = features.MassFeature();

    # NOTE: in this step, the feature hooks back the feature and could access the featurizer by feat.featurer
    feat.register_feature(feature_mass)   # i features

    resulti = ""
    reslst = list(range(5,145))
    for resi in reslst:  # TODO: change the range
      print(f"Processing residue {resi} of the trajectory {trajfile}");
      traj = trajloader.TrajectoryLoader(trajfile, topfile, mask=f":{resi}")[0];
      feat.register_traj(traj)
      # Fit the standardizer of the input features
      feat.register_frames(range(0, 1000, 20))  # TODO: change the range

      index_selected = traj.top.select("@CA,C,O,N,CB&:1")
      print(f"The number of atoms selected is {len(index_selected)}, Total generated molecule block is {feat.frameNr*len(index_selected)}")
      repr_traji, fpfh_traji, features_traji = feat.run_by_atom(index_selected, fbox_length=[6,6,6])

      print(f"The time used for the trajectory (Only residue {resi}) is {time.perf_counter() - st}");
      print(f"The featurization speed is {feat.frameNr * len(index_selected) / (time.perf_counter() - st)} blocks per "
            f"molecule block");
      if resi == reslst[0]:
        resulti = repr_traji
      else: 
        resulti = np.concatenate((resulti, repr_traji)); 

    # Save the data
    if os.path.exists("TEMP_DATA.pkl"):
      with open('TEMP_DATA.pkl', 'rb') as f:
        # Write the object to the file
        thedict = pickle.load(f)
        _repr_traji = thedict["repr"]
      resulti = np.concatenate([_repr_traji, resulti], axis=0)
      thedict = {
        "repr": resulti,
      }
      # Write the object to the file
      pickle.dump(thedict, open('TEMP_DATA.pkl', 'wb'))
    else:
      thedict = {
        "repr":resulti,
      }
      # Write the object to the file
      pickle.dump(thedict, open('TEMP_DATA.pkl', 'wb'))


