import time, os, subprocess

import pytraj as pt
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
  # print(dir(nearl))
  nearl.CONFIG["tempfolder"] = f"/home/yzhang/Documents/tests/tempfolder_mlproj/single_{focused_res.replace(':', '')}"
  nearl.update_config()
  if (not os.path.isdir(nearl._tempfolder)):
    print("Creating the temp folder", nearl._tempfolder)
    subprocess.run(["mkdir", "-p", nearl._tempfolder])

  for complex_idx, complex_file in enumerate(traj_list):
    print(f"Processing {complex_file} ({complex_idx + 1}/{len(traj_list)}, last complex took {(time.perf_counter() - st_this):.2f} seconds)")
    st_this = time.perf_counter()
    tmp_traj = nearl.io.Trajectory(complex_file)
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
        feat.register_frames([0])

        # Compute the representation vector and concatenate it to the return array
        repr_traji, features_traji = feat.run_by_atom(index_selected)
        if len(ret_array) == 0:
          ret_array = np.array(repr_traji)
        else:
          ret_array = np.concatenate([ret_array, np.array(repr_traji)], axis=0)

    if len(ret_array) // 100 > 10 or complex_idx == len(traj_list) -1:
      tempfilename = os.path.join(_tempfolder, f"{focused_res.strip(':')}_{os.getpid()}.npy")
      if os.path.exists(tempfilename) and os.path.getsize(tempfilename) > 0:
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
      print(f"Dumped the data to the temp file named {tempfilename}")
    print(f"This has {len(residues_CAs)} CA atoms")
  print(f"Total {c} CAs")



if __name__ == '__main__':
  st = time.perf_counter()
  complex_folder = "/MieT5/BetaPose/data/complexes/"
  complex_file_list = "complex_filelist.txt"
  with open(os.path.join(complex_folder, complex_file_list), 'r') as f:
    complex_files = [i for i in f.read().strip("\n").split("\n") if i != ""]
  complex_files = [os.path.join(complex_folder, i) for i in complex_files]

  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [36, 36, 36],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH": [16, 16, 16],  # Unit: Angstorm (Need scaling)
  }

  do_test = False
  if do_test:
    st = time.perf_counter()
    FOCUSED_RESIDUE = ":HIS"
    found_PDB = complex_files[:100]
    result1 = parallelize_traj(found_PDB, FOCUSED_RESIDUE)
    results = [result1]
    print(f"# Task finished, time elapsed: {(time.perf_counter() - st) / 60:.2f} min")
  else:
    # TODO: Change these settings before running for production
    worker_num = 16
    thread_per_worker = 2
    found_PDB = complex_files[:50]
    split_groups = np.array_split(found_PDB, worker_num)
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
        st = time.perf_counter()
        futures = client.compute(tasks)
        results = client.gather(futures)
        [i.release() for i in futures]
        print(f"# Task finished, time elapsed: {(time.perf_counter() - st)/60:.2f} min")


