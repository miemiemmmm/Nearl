import subprocess, os, sys, time, h5py

import numpy as np
import dask
from dask.distributed import Client

from BetaPose import representations, trajloader, features, data_io

# Check the OMP thread number
omp_thread = os.environ.get('OMP_NUM_THREADS', -1);
if int(omp_thread) != 1:
  print("Warning: Please set the environment variable OMP_NUM_THREADS explicitly to 1: ")
  print('export OMP_NUM_THREADS=1; export OMP_PROC_BIND=CLOSE; export OMP_CHECK_AFFINITY=TRUE; export OMP_NESTED=TRUE; export OMP_WAIT_POLICY=ACTIVE; export PMI_MMAP_SYNC_WAIT_TIME=5; export PMI_NO_PREINITIALIZE=1; ')
  sys.exit()

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
  feat.register_feature(features.TopFileNameFeature());
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
  feat.register_frames(range(0, 1000, 200))
  index_selected = trajectory.top.select(":LIG")
  print(f"The number of atoms selected is {len(index_selected)}, " +
        f"Total generated molecule block is {feat.FRAMENUMBER * len(index_selected)}")
  repr_traji, features_traji = feat.run_by_atom(index_selected, focus_mode="cog")
  # print(features_traji)
  return features_traji;



if __name__ == "__main__":
  print("Current working directory: ", os.getcwd());
  outputfile = "../data/data.h5";
  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [48, 48, 48],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH":    [24, 24, 24],  # Unit: Angstorm (Need scaling)
  }

  # Load multiple trajectories
  # trajectories = sys.argv[1]
  # topologies = sys.argv[2]
  trajectories = "/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_001_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_002_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_003_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_004_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_005_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_006_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_007_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_009_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_010_traj.nc%"
  topologies = ["/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_END.pdb"] * 10

  # Remove the output file if it exists
  if os.path.exists(outputfile):
    os.remove(outputfile)

  traj_list = trajectories.strip("%").split("%");
  top_list = [i for i in topologies];
  traj_loader = trajloader.TrajectoryLoader(traj_list, top_list);

  # Dask parallelization with 16 workers and 2 threads per worker;
  # Top level parallelization: parallelize over trajectories;
  with Client(processes=True, n_workers=16, threads_per_worker=2) as client:
    tasks = [dask.delayed(parallelize_traj)(traj) for traj in traj_loader];
    print("##################Tasks are generated##################");
    futures = client.compute(tasks);
    results = client.gather(futures);

  # Convert the results to numpy array
  box_array = np.array([[j[0] for j in i] for i in results]);
  panelty_array = np.array([[j[1] for j in i] for i in results]);
  name_array = np.array([[j[2] for j in i] for i in results]);
  RF_array = np.array([[j[3] for j in i] for i in results]);

  print("Tasks finished, start saving the data", name_array)
  name_array = np.array(name_array, dtype=h5py.string_dtype('utf-8'))

  # Save the data
  with data_io.hdf_operator(outputfile) as f_write:
    f_write.create_dataset("box", box_array);
    f_write.create_dataset("topo_name", name_array);
    f_write.create_dataset("panelty", panelty_array);
    f_write.create_dataset("RF", RF_array);
    f_write.draw_structure()

  # Check the data
  with data_io.hdf_operator(outputfile) as hdfile:
    hdfile.draw_structure()
    print(hdfile.data("box").shape)
    print(hdfile.data("topo_name").shape)
    print(hdfile.data("panelty").shape)
    print(hdfile.data("RF").shape)

