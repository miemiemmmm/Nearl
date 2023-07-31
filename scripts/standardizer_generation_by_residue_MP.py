import open3d as o3d
import time, builtins, tempfile, datetime, os, sys
import pickle
from BetaPose import utils, chemtools, representations;
import multiprocessing as mp
import pytraj as pt
import numpy as np
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from BetaPose import utils, trajloader
from BetaPose import features, featurizer

import dask
from dask.distributed import Client


omp_thread = os.environ.get('OMP_NUM_THREADS', -1);
print("OMP_NUM_THREADS ->" ,omp_thread)
if int(omp_thread) != 1:
  print("Warning: Please set the environment variable OMP_NUM_THREADS explicitly to 1: ")
  print('export OMP_NUM_THREADS=1; export OMP_PROC_BIND=CLOSE; export OMP_CHECK_AFFINITY=TRUE; export OMP_NESTED=TRUE; export OMP_WAIT_POLICY=ACTIVE; export PMI_MMAP_SYNC_WAIT_TIME=5; export PMI_NO_PREINITIALIZE=1; ')
#  sys.exit()

# For each residue,
def process_framei(trajfile, topfile, resi):
  print(f"Processing residue {resi} of the trajectory {trajfile}");
  # Initialize the featurizer
  feat = features.Featurizer3D(FEATURIZER_PARMS);
  # feature_mass = ;
  
  # NOTE: in this step, the feature hooks back the feature and could access the featurizer by feat.featurer
  feat.register_feature(features.MassFeature());  # i features
  feat.register_feature(features.BoxFeature()); 

  # Load the trajectory in need

  traj = trajloader.TrajectoryLoader(trajfile, topfile, mask=f":{resi}")[0];
  feat.register_traj(traj)
  # Fit the standardizer of the input features
  feat.register_frames(range(0, 1000, 20))  # TODO: change the range

  index_selected = traj.top.select("@CA,C,O,N,CB&:1")
  print(f"The number of atoms selected is {len(index_selected)}")
  # print(f"Total generated molecule block is {feat.frameNr * len(index_selected)}")

  repr_traji, features_traji = feat.run_by_atom(index_selected)
  return repr_traji



if __name__ == "__main__":
  print("Current working directory: ", os.getcwd())
  if os.path.exists("TEMP_DATA.pkl"):
    os.remove("TEMP_DATA.pkl")

  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION" : [12, 12, 12],    # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH" : [8,8,8],           # Unit: Angstorm (Need scaling)
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
    # # NOTE: in this step, the feature hooks back the feature and could access the featurizer by feat.featurer
    # feat.register_feature(feature_mass)   # i features

    # Process residues one by one
    reslst = list(range(5,145))     # TODO: change the range of residues

    with Client(processes=True, n_workers=16, threads_per_worker=2) as client:
      tasks = [dask.delayed(process_framei)(trajfile, topfile, resi) for resi in reslst]
      futures = client.compute(tasks)
      results = client.gather(futures)

    resulti = np.concatenate(results)
    print(f"Return from the multiprocessing: {resulti.shape}")

    # with mp.Pool(processes=24) as pool:
    #   result = pool.starmap(process_framei, [(trajfile, topfile, resi) for resi in reslst])
    # pool.join()




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

