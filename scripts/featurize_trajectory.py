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


if __name__ == '__main__':
  nearl.update_config(verbose = False, debug = False)
  # nearl.update_config(verbose = True, debug = True)

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
  
  feat.run()
