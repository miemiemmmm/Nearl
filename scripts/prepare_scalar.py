import time, tempfile, datetime, os, pickle, json, sys
import numpy as np
import multiprocessing as mp
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from BetaPose import trajloader, features, CONFIG

if __name__ == "__main__":
  omp_thread = os.environ.get('OMP_NUM_THREADS', -1);
  if int(omp_thread) != 1:
    print("Warning: Please set the environment variable OMP_NUM_THREADS explicitly to 1: ")
    print(
      'export OMP_NUM_THREADS=1; export OMP_PROC_BIND=CLOSE; export OMP_CHECK_AFFINITY=TRUE; export OMP_NESTED=TRUE; export OMP_WAIT_POLICY=ACTIVE; export PMI_MMAP_SYNC_WAIT_TIME=5; export PMI_NO_PREINITIALIZE=1; ')
    sys.exit()
  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [16, 16, 16],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH": [8, 8, 8],  # Unit: Angstorm (Need scaling)
  }

  # Load multiple trajectories
  trajs = "/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_001_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_002_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_003_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_004_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_005_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_006_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_007_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_009_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_010_traj.nc%"
  tops = ["/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_END.pdb"] * 10
  trajs = trajs.strip("%").split("%")
  trajectories = trajloader.TrajectoryLoader(trajs, tops);


  def process_traji(traj):
    print(f"Processing trajectory: {traj}");
    try:
      traj.strip(":T3P")
    except:
      pass
    # Initialize the featurizer since different trajectory might have distinct parameters
    featurizer = features.Featurizer3D(FEATURIZER_PARMS);
    featurizer.register_feature(features.BoxFeature())     # Register one simple feature
    featurizer.register_traj(traj)

    # Fit the standardizer of the input features
    # featurizer.register_frames(range(0, 1000, 50))
    # repr_traji, features_traji = featurizer.run_by_atom(traj.top.select("@CA,CB,N,O,C,CG,CD=,NE=,CE=&:2-145"))
    featurizer.register_frames(range(0, 1000, 100))
    print("====> before run_by_atom")
    repr_traji, features_traji = featurizer.run_by_atom(traj.top.select("@CA&:144-145"))
    print(f"Finished traj {traj}:", repr_traji.shape)
    return repr_traji


  with mp.Pool(processes=len(trajs)) as pool:
    result = pool.starmap(process_traji, [(traj,) for traj in trajectories])
  pool.join()

  repr_traji = np.concatenate(result)

  # for traj in trajectories:
  #   # Complete the trajectory information
  #   try:
  #     traj.strip(":T3P")
  #   except:
  #     pass
  #
  #   # Initialize the featurizer since different trajectory might have distinct parameters
  #
  #   featurizer = features.Featurizer3D(FEATURIZER_PARMS);
  #   featurizer.register_feature(features.BoxFeature())  # Register one simple feature
  #
  #   # NOTE: in this step, the feature hooks back the feature and could access the featurizer by feat.featurer
  #   featurizer.register_traj(traj)
  #
  #   # Fit the standardizer of the input features
  #   featurizer.register_frames(range(0, 1000, 50))
  #   # repr_traji, features_traji = featurizer.run_by_atom(traj.top.select("@CA,C,N,CB,CG,CD=,CE=&:2-145"))
  #   repr_traji, features_traji = featurizer.run_by_atom(traj.top.select("@CA,CB,N,O,C,CG,CD=,NE=,CE=&:2-145"))
  #
  #   # A lot of frames and centers
  #   # featurizer.register_frames(range(0, 1000, 50))
  #   # repr_traji, features_traji = featurizer.run_by_atom(traj.top.select("@CA&:1-145"))
  #
  #   # Fix the residues atom and iterate different frames
  #   # featurizer.register_frames(range(0, 1000, 10))
  #   # repr_traji, features_traji = featurizer.run_by_atom(traj.top.select("@CA&:45"))
  #
  #   # Fix the frame and iterate different resiudes
  #   # featurizer.register_frames([42])
  #   # repr_traji, features_traji = featurizer.run_by_atom(traj.top.select("@CA&:45-145"))
  #
  #   # Fix the frames, and iterated load one resdiue TODO
  #   # featurizer.register_frames([60])
  #   # repr_traji, fpfh_traji, features_traji = featurizer.run_by_atom(traj.top.select("@CA&:45-145"))
  #
  #   print(f"Finished traj {traj}:", repr_traji.shape)
  #   break


  slices = [(1 , 5), (5, 15),   (15, 35),   (35, 55),   (55, 9999)]
  repr_arr_flat_seg = repr_traji.reshape((-1, 12+CONFIG.get("VIEWPOINT_BINS")));
  print("flattened shape: ", repr_arr_flat_seg.shape)
  final_scalars = []
  for i in range(len(slices)):
    statelb = repr_arr_flat_seg[:,0] >= slices[i][0]
    stateub = repr_arr_flat_seg[:,0] < slices[i][1]
    state = statelb & stateub
    print("Slice: ", slices[i], "; Sample size: ", np.count_nonzero(state))
    if np.count_nonzero(state) > 0:
      scalari = {
        "means": np.mean(repr_arr_flat_seg[state], axis=0).tolist(),
        "stds": np.std(repr_arr_flat_seg[state], axis=0).tolist(),
        "sample_size": np.count_nonzero(state),
        "n_features": repr_arr_flat_seg.shape[1],
        "dims": list(FEATURIZER_PARMS["CUBOID_DIMENSION"]),
        "lengths": list(FEATURIZER_PARMS["CUBOID_LENGTH"]),
        "lower_bound": slices[0],
        "lower_mode" : "ge",
        "upper_bound": slices[1],
        "upper_mode" : "lt",
        "date" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      }
      final_scalars.append(scalari)
    else:
      scalari = {
        "means": [0] * repr_arr_flat_seg.shape[1],
        "stds": [0] * repr_arr_flat_seg.shape[1],
        "sample_size": 0,
        "n_features": repr_arr_flat_seg.shape[1],
        "dims": list(FEATURIZER_PARMS["CUBOID_DIMENSION"]),
        "lengths": list(FEATURIZER_PARMS["CUBOID_LENGTH"]),
        "lower_bound": slices[0],
        "lower_mode" : "ge",
        "upper_bound": slices[1],
        "upper_mode" : "lt",
        "date" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      }
      final_scalars.append(scalari)
    # print("Final shapes: ", repr_traji.shape)

  # Save the data as json
  np.save("/tmp/test.npy", repr_traji)
  with open('/tmp/TEMP_DATA.json', 'w') as fp:
    json.dump(final_scalars, fp)


# Write the object to the file
# pickle.dump(thedict, open('TEMP_DATA.pkl', 'wb'))
# write_scalar = False
# if write_scalar:
#   # scaler = StandardScaler()
#   scaler = RobustScaler();
#   scaler.fit(repr_traji);
#   pickle.dump(scaler, open("StandardScaler_model.pkl", "wb"))
