import os, argparse, time, json
import numpy as np
import pytraj as pt
from scipy.stats import entropy
from collections import OrderedDict

import nearl 
import nearl.data
from nearl.io import Trajectory



def parser(): 
  parser = argparse.ArgumentParser(description="Featurize the trajectories")
  parser.add_argument("-f", "--trajfiles", type=str, required=True, help="The file containing the list of pdb codes")
  parser.add_argument("-o", "--output_dir", type=str, default="",help="The output directory")
  parser.add_argument("--task_nr", type=int, default=1, help="The task number to run")
  parser.add_argument("--task_index", type=int, default=0, help="The task index to run")
  args = parser.parse_args()
  if not os.path.exists(args.output_dir):
    raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")
  if not os.path.exists(args.trajfiles):
    raise FileNotFoundError(f"Trajectory file {args.trajfiles} does not exist")
  return args


class customized_traj(Trajectory): 
  def __init__(self, *args, **kwargs): 
    super().__init__(*args, **kwargs)
    self.identity = os.path.basename(self.top_filename)[:4].lower()
    print(self.top_filename, self.traj_filename)
    print("Identity of the traj", self.identity)


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
               'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}
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
  nearl.update_config(
    verbose = False, 
    debug = False,
    # verbose = True, 
    # debug = True,
  )

  args = parser()
  args = vars(args)

  print(json.dumps(args, indent=2))

  task_nr = args.get("task_nr")
  task_index = args.get("task_index")
  outputfile = os.path.join(os.path.abspath(args["output_dir"]), f"InHouseOutput{task_index}.h5") 

  with open(args["trajfiles"], "r") as f:
    files = f.read().strip().split("\n")
    trajlists = [(i.split()[0], i.split()[1]) for i in files]

  print(trajlists)


  FEATURIZER_PARMS = {
    "dimensions": 32, 
    "lengths": 16, 
    "time_window": 16,    # Time window equal to 0.8 ns 

    # For default setting inference of registered features 
    "sigma": 1.5, 
    "cutoff": 2.55, 
    "outfile": outputfile, 
  }

  trajlists = np.array_split(trajlists, task_nr)[task_index]
  # trajlists = trajlists[:5]   # TODO: Remove this line for production run
  loader = nearl.io.TrajectoryLoader(trajlists, superpose=True, mask="!:T3P", trajtype=customized_traj)
  feat  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
  feat.register_trajloader(loader)
  feat.register_focus([":LIG"], "mask")
  
  features = OrderedDict()

  # Static features
  features["lig_annotation"] = nearl.features.Selection(selection=":LIG", selection_type="mask", outkey = "ligand_annotation")
  features["prot_annotation"] = nearl.features.Selection(selection="!:LIG", selection_type="mask", outkey = "protein_annotation")
  features["mass_lig"] = nearl.features.Mass( selection=":LIG", outkey="mass_lig" )
  features["mass_prot"] = nearl.features.Mass( selection="!:LIG", outkey="mass_prot" )

  # Static atom types 
  features["type_H_Lig"] = nearl.features.AtomType(selection=":LIG", focus_element=1, outkey="lig_type_H")
  features["type_C_Lig"] = nearl.features.AtomType(selection=":LIG", focus_element=6, outkey="lig_type_C")
  features["type_N_Lig"] = nearl.features.AtomType(selection=":LIG", focus_element=7, outkey="lig_type_N")
  features["type_O_Lig"] = nearl.features.AtomType(selection=":LIG", focus_element=8, outkey="lig_type_O")
  features["type_S_Lig"] = nearl.features.AtomType(selection=":LIG", focus_element=16, outkey="lig_type_S")

  features["type_H_Prot"] = nearl.features.AtomType(selection="!:LIG", focus_element=1, outkey="prot_type_H")
  features["type_C_Prot"] = nearl.features.AtomType(selection="!:LIG", focus_element=6, outkey="prot_type_C")
  features["type_N_Prot"] = nearl.features.AtomType(selection="!:LIG", focus_element=7, outkey="prot_type_N")
  features["type_O_Prot"] = nearl.features.AtomType(selection="!:LIG", focus_element=8, outkey="prot_type_O")
  features["type_S_Prot"] = nearl.features.AtomType(selection="!:LIG", focus_element=16, outkey="prot_type_S")
  ##############################################################################

  # Dynamic features
  features["obs_HCount_lig"] = nearl.features.MarchingObservers(selection=":LIG", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="lig_HCount_obs", element_type=1)
  features["obs_CCount_lig"] = nearl.features.MarchingObservers(selection=":LIG", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="lig_CCount_obs", element_type=6)
  features["obs_NCount_lig"] = nearl.features.MarchingObservers(selection=":LIG", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="lig_NCount_obs", element_type=7)
  features["obs_OCount_lig"] = nearl.features.MarchingObservers(selection=":LIG", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="lig_OCount_obs", element_type=8)
  features["obs_SCount_lig"] = nearl.features.MarchingObservers(selection=":LIG", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="lig_SCount_obs", element_type=16)

  features["obs_HCount_prot"] = nearl.features.MarchingObservers(selection="!:LIG", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="prot_HCount_obs", element_type=1)
  features["obs_CCount_prot"] = nearl.features.MarchingObservers(selection="!:LIG", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="prot_CCount_obs", element_type=6)
  features["obs_NCount_prot"] = nearl.features.MarchingObservers(selection="!:LIG", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="prot_NCount_obs", element_type=7)
  features["obs_OCount_prot"] = nearl.features.MarchingObservers(selection="!:LIG", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="prot_OCount_obs", element_type=8)
  features["obs_SCount_prot"] = nearl.features.MarchingObservers(selection="!:LIG", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="prot_SCount_obs", element_type=16)
  ##############################################################################
  

  # Labels
  features["pk_original"] = nearl.features.LabelAffinity(
    baseline_map=nearl.data.GENERAL_SET, 
    outkey="pk_original"
  )

  features["stepping"] = nearl.features.LabelStepping(
    baseline_map=nearl.data.GENERAL_SET, 
    outkey="label_stepping"
  )

  features["label_pcdt"] = nearl.features.LabelPCDT(
    selection=":LIG", 
    baseline_map=nearl.data.GENERAL_SET, 
    outkey="label_pcdt"
  )

  print(f"There are {len(features)} features registered: {features.keys()}")

  feat.register_features(features)
  
  feat.main_loop(8)
