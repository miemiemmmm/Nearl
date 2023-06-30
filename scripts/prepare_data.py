import subprocess, os, sys, time, h5py
import numpy as np
from BetaPose import representations, trajloader, features
import dask
from dask.distributed import Client

# Check the OMP thread number
omp_thread = os.environ.get('OMP_NUM_THREADS', -1);
if int(omp_thread) != 1:
  print("Warning: Please set the environment variable OMP_NUM_THREADS explicitly to 1: ")
  print('export OMP_NUM_THREADS=1; export OMP_PROC_BIND=CLOSE; export OMP_CHECK_AFFINITY=TRUE; export OMP_NESTED=TRUE; export OMP_WAIT_POLICY=ACTIVE; export PMI_MMAP_SYNC_WAIT_TIME=5; export PMI_NO_PREINITIALIZE=1; ')
  # sys.exit()



def parallelize_traj(trajectory):
  """
  Featurizer and the features initialized within the scope of this function

  """
  if trajectory.top.select(":T3P").__len__() > 0:
    trajectory.strip(":T3P")
  # Initialize the featurizer since different trajectory might have distinct parameters
  feat = features.Featurizer3D(FEATURIZER_PARMS);
  # NOTE: in this step, the feature hooks back the feature and could access the featurizer by feat.featurer
  feat.register_feature(features.BoxFeature());
  feat.register_feature(features.PaneltyFeature("(:LIG)&(!@H=)", "(:LIG<:5)&(!@H=)&(!:LIG)", ref=0));
  # feat.register_feature(features.FPFHFeature());

  ############## Done
  # feat.register_feature(features.MassFeature());
  ############## TODO
  # feat.register_feature(features.HydrophobicityFeature());
  ############## TODO
  # feat.register_feature(features.AromaticityFeature());
  ############## TODO
  # feat.register_feature(features.PartialChargeFeature());

  ############## TODO
  feat.register_feature(features.RFFeature1D(":LIG"));

  # Protein Dynamics realted features
  ############## DONE
  # feat.register_feature(features.EntropyResidueFeature());
  # feat.register_feature(features.EntropyAtomicFeature());

  ############## TODO
  # feat.register_feature(features.PartialChargeFeature(moi="(:LIG)"));
  # feat.register_feature(features.PartialChargeFeature(moi="(:T3P,HOH,WAT)"));
  # feat.register_feature(features.PartialChargeFeature(moi="(:1-100)"));

  feat.register_traj(trajectory)
  # Fit the standardizer of the input features
  feat.register_frames(range(0, 1000, 100))
  index_selected = trajectory.top.select(":LIG")
  print(f"The number of atoms selected is {len(index_selected)}, " +
        f"Total generated molecule block is {feat.FRAMENUMBER * len(index_selected)}")
  repr_traji, features_traji = feat.run_by_atom(index_selected, focus_mode="cog")
  print(features_traji)
  return features_traji;



if __name__ == "__main__":
  print("Current working directory: ", os.getcwd())
  if os.path.exists("../data/data.h5"):
    os.remove("../data/data.h5")

  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [48, 48, 48],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH":    [24, 24, 24],  # Unit: Angstorm (Need scaling)
  }

  # Load multiple trajectories
  # trajectories = sys.argv[1]
  # topologies = sys.argv[2]
  trajs = "/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_001_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_002_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_003_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_004_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_005_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_006_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_007_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_009_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_010_traj.nc%"
  tops = ["/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_END.pdb"] * 10
  trajs = trajs.strip("%").split("%")
  trajectories = trajloader.TrajectoryLoader(trajs, tops);

  # Top level iteration: iterate over trajectories

  with Client(processes=True, n_workers=16, threads_per_worker=2) as client:
    tasks = [dask.delayed(parallelize_traj)(traj) for traj in trajectories]
    futures = client.compute(tasks)
    results = client.gather(futures)

  print(results)
