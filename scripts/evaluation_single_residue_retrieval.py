import time, os
import matplotlib.pyplot as plt
import pytraj as pt
import numpy as np
import dask
from dask.distributed import Client, performance_report, LocalCluster

import nearl
from nearl import features, utils, _tempfolder

from nearl.static import cos_sim

def evaluate_residues(traj_list, focused_res, results_array, resnames):
  means = np.mean(results_array, axis=0)
  stds = np.array([np.std(results_array[:, i]) if np.std(results_array[:, i]) != 0 else 1e9 for i in range(results_array.shape[1])])
  results_array_Zscore = (results_array - means) / stds
  print(results_array_Zscore.shape)

  traj_list = [i for i in traj_list]
  confusion_matrix = np.zeros((20, 20), dtype=int)
  feat = features.Featurizer3D(FEATURIZER_PARMS)

  for complex_idx, complex_file in enumerate(traj_list):
    # print(f"Processing {complex_file} ({complex_idx + 1}/{len(traj_list)}, last complex took {(time.perf_counter() - st_this):.2f} seconds)")
    # Reset the counter/feature container
    total_processed = 0
    total_correct = 0
    ret_array = np.array([])
    thedict = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
               'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18,
               'VAL': 19}
    def restoidx(resname):
      return thedict.get(resname)
    ref_resname = focused_res.strip(":")
    ref_residx = restoidx(ref_resname)


    tmp_traj = nearl.io.Trajectory(complex_file)

    if tmp_traj.top.select(focused_res+"&@CA").__len__() == 0:
      print(f"Complex file {complex_file} does not have residue {focused_res}")
      continue
    else:
      residues_traj = tmp_traj[focused_res]
      residues_traj_atoms = np.array(list(residues_traj.top.atoms))
      residues_CAs = residues_traj.top.select("@CA")

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

      ret_array_Zscore = (ret_array - means) / stds
      print("=====>", ret_array_Zscore)

      print(f"Means: {means[columns]}, stds: {stds[columns]}")
      print(f"Feature array original ({focused_res}): ", ret_array[:, columns].tolist())
      print(f"Feature array Z-score  ({focused_res}): ", np.array(ret_array_Zscore, dtype=np.float64).round(3)[:, columns].tolist())

      # similarities = cos_sim.cosine_similarity(ret_array_Zscore[:, columns], results_array_Zscore[:, columns])
      similarities = cos_sim.euclidean_similarity(ret_array_Zscore[:, columns], results_array_Zscore[:, columns])
      print("Similarities: ", np.mean(similarities), np.std(similarities))

      similarities = np.array(similarities)
      for i in similarities:
        total_processed += 1
        # bestmatch_resname = resnames[np.argmax(i)]
        bestmatch_resname = resnames[np.argmin(i)]
        bestmatch_residx = restoidx(bestmatch_resname)
        confusion_matrix[ref_residx, bestmatch_residx] += 1
        if bestmatch_resname == ref_resname:
          total_correct += 1
        else:
          print(f"Warning: Not correctly identified: {ref_resname} vs {bestmatch_resname}")
      print(f"Correct rate querying {focused_res:6s}: {total_correct/total_processed:6.3f} ({total_correct:4d}/{total_processed:<4d})")
  return confusion_matrix


if __name__ == '__main__':
  st = time.perf_counter()
  complex_folder = "/MieT5/BetaPose/data/complexes/"
  complex_file_list = "complex_filelist.txt"
  with open(os.path.join(complex_folder, complex_file_list), 'r') as f:
    complex_files = [i for i in f.read().strip("\n").split("\n") if i != ""]
  complex_files = [os.path.join(complex_folder, i) for i in complex_files]
  random_traj_list = np.random.choice(complex_files[6000:], 1000, replace=False)


  files = "/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3517805.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3517750.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TYR_3517769.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TRP_3517755.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3517769.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/THR_3517778.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520361.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520415.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TRP_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASN_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TRP_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLN_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520509.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/VAL_3520493.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ILE_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LYS_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TYR_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/VAL_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PHE_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ILE_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ALA_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/VAL_3520519.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLY_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/SER_3520517.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASN_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LYS_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520509.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASP_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/SER_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/CYS_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/CYS_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TRP_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TYR_3520493.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASN_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/VAL_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ALA_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLN_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LYS_3520509.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASN_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/SER_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520517.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLN_3520519.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLY_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TYR_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/THR_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ARG_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TRP_3520517.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PHE_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LEU_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ARG_3520519.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ILE_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ILE_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LYS_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ARG_3520509.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLN_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ILE_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TYR_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/CYS_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/THR_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ARG_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3520509.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ARG_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ILE_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ALA_3520493.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LEU_3520517.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LYS_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LYS_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/THR_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ALA_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LYS_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASN_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASP_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PHE_3520517.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ALA_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ILE_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520517.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PHE_3520509.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TYR_3520509.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/THR_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLY_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLN_3520493.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LYS_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LYS_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASP_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LYS_3520519.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/CYS_3520493.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LEU_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/THR_3520517.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASP_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASP_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ALA_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLY_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLN_3520509.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLY_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLN_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TYR_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLY_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/VAL_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLN_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ALA_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PHE_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/THR_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LEU_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/THR_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ARG_3520493.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/VAL_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/CYS_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ARG_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TRP_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ALA_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLY_3520493.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TYR_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASP_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/VAL_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TRP_3520493.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ARG_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TRP_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASP_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLY_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PHE_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/SER_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/SER_3520519.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ILE_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/SER_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TRP_3520519.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TYR_3520519.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TRP_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ALA_3520517.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ALA_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TRP_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/VAL_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASP_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/SER_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASN_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASP_3520493.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ILE_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ILE_3520517.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/HIS_3520519.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LEU_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ARG_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LEU_3520509.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ARG_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/CYS_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/THR_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASP_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/VAL_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASN_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3520517.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/MET_3520493.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/THR_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PRO_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LEU_3520528.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LYS_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/VAL_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520517.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ALA_3520524.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PHE_3520519.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LEU_3520493.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PHE_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASN_3520519.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLY_3520539.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/THR_3520509.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/SER_3520534.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/PHE_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASN_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ASN_3520509.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/TYR_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/SER_3520505.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLN_3520513.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLU_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLN_3520548.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLY_3520501.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/SER_3520546.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/VAL_3520490.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LEU_3520497.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/ILE_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/GLY_3520508.npy%/home/yzhang/Documents/tests/tempfolder_mlproj/LEU_3520513.npy"
  files = files.split("%")
  resname_list = []
  result_arr = []
  for i in files:
    name = os.path.split(i)[-1][:3]
    resi = np.load(i, allow_pickle=True)
    result_arr.append(resi)
    resname_list += [name]*resi.shape[0]
  result_arr = np.concatenate(result_arr, axis=0)
  print(len(resname_list), result_arr.shape)

  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [36, 36, 36],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH": [16, 16, 16],  # Unit: Angstorm (Need scaling)
  }
  columns = list(range(1,36))

  do_test = False
  if do_test:
    st = time.perf_counter()
    FOCUSED_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                        'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                        'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                        'SER', 'THR', 'TRP', 'TYR', 'VAL']
    PDB_todo = random_traj_list[:100]
    result1 = [evaluate_residues(PDB_todo, f":{i}", result_arr, resname_list) for i in FOCUSED_RESIDUES]
    results = [result1]
    print(f"# Task finished, time elapsed: {(time.perf_counter() - st) / 60:.2f} min")
  else:
    worker_num = 24
    thread_per_worker = 2
    PDB_todo = random_traj_list[:50]

    cluster = LocalCluster(n_workers=worker_num, threads_per_worker=thread_per_worker, processes=True, memory_limit='10GB')
    client = Client(cluster)
    FOCUSED_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                        'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                        'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                        'SER', 'THR', 'TRP', 'TYR', 'VAL']
    feature_vec_future = client.scatter(result_arr, broadcast=True)
    resname_future = client.scatter(resname_list, broadcast=True)

    with performance_report(filename="dask-report.html"):
      tasks = []
      c = 0
      for traj_list in np.array_split(PDB_todo, 2):
        for residue in FOCUSED_RESIDUES:
          c += 1
          print(f"Task {c} | focused residue: {residue}")
          tasks.append(dask.delayed(evaluate_residues)(traj_list, f":{residue}", feature_vec_future, resname_future))

      print(f"# Task set contains {len(tasks)} jobs; Jobs are generated and ready to run")
      st = time.perf_counter()
      futures = client.compute(tasks)
      results = client.gather(futures)
      print(f"# Task finished, time elapsed: {(time.perf_counter() - st) / 60:.2f} min")

      with open(f"confusion_matrix_single_residue_retrieval_{','.join([str(i) for i in columns])}.npy", "wb") as f:
        np.save(f, results)

