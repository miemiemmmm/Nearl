import time, os, sys

import numpy as np
import dask
from dask.distributed import Client, performance_report, LocalCluster

import nearl
from nearl import features, utils, _tempfolder
# Only needs the fingerprint of the

def parallelize_traj(traj_list, focused_res):
  traj_list = [i for i in traj_list]
  st_this = time.perf_counter()
  ret_array = np.array([])
  feat = features.Featurizer3D(FEATURIZER_PARMS)
  c=0
  for complex_idx, complex_file in enumerate(traj_list):
    print(f"Processing {complex_file} ({complex_idx + 1}/{len(traj_list)}, last complex took {(time.perf_counter() - st_this):.2f} seconds)")
    st_this = time.perf_counter()
    tmp_traj = nearl.io.Trajectory(complex_file[0], complex_file[1])
    if tmp_traj.top.select(focused_res+"&@CA").__len__() == 0:
      print(f"Complex file {complex_file} does not have residue {focused_res}")
      continue
    else:
      residues_traj = tmp_traj[focused_res]
      residues_traj_atoms = np.array(list(residues_traj.top.atoms))
      residues_CAs = residues_traj.top.select("@CA")
      c += len(residues_CAs)
      for atomidx in residues_CAs:
        ca_atom = residues_traj_atoms[atomidx]
        mask = f":{ca_atom.resid+1}"

        # Obtain the residue-based trajectory object
        theresidue = residues_traj[mask]
        index_selected = theresidue.top.select(focused_res+"&@CA")
        feat.register_traj(theresidue)
        feat.register_frames(range(0, theresidue.n_frames, 1000))

        # Compute the representation vector and concatenate it to the return array
        repr_traji, features_traji = feat.run_by_atom(index_selected)
        if len(ret_array) == 0:
          ret_array = np.array(repr_traji)
        else:
          ret_array = np.concatenate([ret_array, np.array(repr_traji)], axis=0)

    if len(ret_array) // 100 > 10 or complex_idx == len(traj_list) -1:
      tempfilename = os.path.join(_tempfolder, f"{focused_res.strip(':')}_{os.getpid()}.npy")
      print(f"Dumping the data to the temp file named {tempfilename}", file=sys.stderr)
      if os.path.exists(tempfilename) and os.path.getsize(tempfilename) > 0:
        # NOTE: Opening a zero-sized file with numpy causes unexpected EOF error
        prev_data = np.load(tempfilename, allow_pickle=True)
        # Convert the results to numpy array
        new_data = np.array(ret_array, dtype=np.float64)
        new_data = np.concatenate([prev_data, new_data], axis=0)
        print(f"Concated data shaped {new_data.shape}")
        new_data.astype(np.float64)
        nearl.io.temporary_dump(new_data, tempfilename)
        ret_array = np.array([])
      else:
        nearl.io.temporary_dump(ret_array, tempfilename)
        ret_array.astype(np.float64)
        ret_array = np.array([])

    print(f"This has {len(residues_CAs)} CA atoms")
  print(f"Total {c} CAs")



if __name__ == '__main__':
  st = time.perf_counter()
  trajs = "/MieT5/BetaPose_trajs/C209CsDJQucZ_job_001_traj.nc%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_002_traj.nc%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_003_traj.nc%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_004_traj.nc%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_005_traj.nc%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_006_traj.nc%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_007_traj.nc%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_traj.nc%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_009_traj.nc%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_010_traj.nc"
  tops = "/MieT5/BetaPose_trajs/C209CsDJQucZ_job_001_END.pdb%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_002_END.pdb%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_003_END.pdb%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_004_END.pdb%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_005_END.pdb%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_006_END.pdb%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_007_END.pdb%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_END.pdb%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_009_END.pdb%/MieT5/BetaPose_trajs/C209CsDJQucZ_job_010_END.pdb"

  traj_pairs = [(traj, top) for traj, top in zip(trajs.split("%"), tops.split("%"))]


  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [36, 36, 36],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH": [16, 16, 16],  # Unit: Angstorm (Need scaling)
  }

  do_test = False
  do_dasktest = False
  if do_test:
    FOCUSED_RESIDUE = ":ARG"
    result1 = parallelize_traj(traj_pairs, FOCUSED_RESIDUE)
    results = [result1]
    print(results)
  elif do_dasktest:
    cluster = LocalCluster(
      n_workers=1,
      threads_per_worker=4,
      processes=True,
      memory_limit='10GB',
    )
    with Client(cluster) as client:
      with performance_report(filename="dask-report.html"):
        FOCUSED_RESIDUE = ":ARG"
        testjob = [dask.delayed(parallelize_traj)(traj_pairs, FOCUSED_RESIDUE)]
        futures = client.compute(testjob)
        results = client.gather(futures)
        [i.release() for i in futures]
  else:
    # TODO: Change these settings before running for production
    worker_num = 8
    thread_per_worker = 4
    split_groups = np.array_split(traj_pairs, worker_num)
    cluster = LocalCluster(
      n_workers=worker_num,
      threads_per_worker=thread_per_worker,
      processes=True,
      memory_limit='10GB',
    )
    with Client(cluster) as client:
      FOCUSED_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                          'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                          'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                          'SER', 'THR', 'TRP', 'TYR', 'VAL']
      # FOCUSED_RESIDUES = ['ALA', 'ARG']
      with performance_report(filename="dask-report.html"):
        tasks = []
        c = 0
        for trajlist in split_groups:
          for residue in FOCUSED_RESIDUES:
            c += 1
            print(f"Task {c} | focused residue: {residue} | trajectories: {trajlist}")
            tasks.append(dask.delayed(parallelize_traj)(trajlist, f":{residue}"))

        # tasks = [dask.delayed(parallelize_traj)(traj_list) for traj_list in split_groups]
        print(f"# Task set contains {len(tasks)} jobs; Jobs are generated and ready to run")
        futures = client.compute(tasks)
        results = client.gather(futures)
        [i.release() for i in futures]
  print(f"# Task finished, time elapsed: {(time.perf_counter() - st)/60:.2f} min")


