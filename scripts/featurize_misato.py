import time, json, os, argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

import nearl 
import nearl.data
from nearl.io.traj import MisatoTraj


# Define the way to generation labels for the misato trajectories

def parser(): 
  parser = argparse.ArgumentParser(description="Featurize the misato trajectories")
  parser.add_argument("--task_nr", type=int, default=1, help="The task number to run")
  parser.add_argument("--task_index", type=int, default=0, help="The task index to run")
  parser.add_argument("--output_dir", type=str, default="",help="The output directory")
  args = parser.parse_args()
  if not os.path.exists(args.output_dir):
    raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")
  return args

def get_trajlist(training_set, misatodir): 
  with open(training_set, "r") as f:
    pdbcodes = f.read().strip("\n").split("\n")
  trajlists = [(i, misatodir) for i in pdbcodes]
  return trajlists


if __name__ == "__main__":
  nearl.update_config(
    verbose = False, 
    debug = False,
    # verbose = True, 
    # debug = True,
  )

  args = parser()
  args = vars(args)
  print(args)
  task_nr = args.get("task_nr")
  task_index = args.get("task_index")


  # Load trajectories 
  misatodir = "/Matter/misato_database/"
  training_set = "/MieT5/BetaPose/data/misato_train.txt"
  # training_set = "/MieT5/BetaPose/data/misato_test.txt"
  outputfile = os.path.join(os.path.abspath(args["output_dir"]), f"MisatoOutput{task_index}.h5") 

  print(f"Input file: {training_set}, Output file: {outputfile}; Task {task_index} of {task_nr}")
  
  # Initialize featurizer object and register necessary components
  FEATURIZER_PARMS = {
    "dimensions": 32, 
    "lengths": 16, 
    "time_window": 10, 

    # For default setting inference of registered features
    "sigma": 1.5, 
    "cutoff": 2.55, 
    "outfile": outputfile, 

    # Other options
    "progressbar": False, 
  }

  trajlists = get_trajlist(training_set, misatodir)
  trajlists = np.array_split(trajlists, args.get("task_nr"))[args.get("task_index")]
  print(f"# Total number of trajectories {trajlists.__len__()}")

  # trajlists = trajlists[:10]   # TODO: Remove this line for production run
  loader = nearl.io.TrajectoryLoader(trajlists, trajtype=MisatoTraj, superpose=True)
  print(f"Performing the featurization on {len(loader)} trajectories")

  feat  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)

  feat.register_trajloader(loader)
  feat.register_focus([":MOL"], "mask")

  #############################################################################
  #############################################################################
  #############################################################################

  # !!! Change the keyword to different names to avoid conflict
  features = OrderedDict()

  # Use the default charge from the trajectory topology 
  features["feat_atomicnumber"] = nearl.features.AtomicNumber( outkey="atomic_number" )
  features["feat_mass"] = nearl.features.Mass( outkey="mass" )
  features["aromaticity"] = nearl.features.Aromaticity( outkey="aromaticity" )
  features["feat_pp"] = nearl.features.PartialCharge( outkey="partial_charge_positive", keep_sign="p" )
  features["feat_pn"] = nearl.features.PartialCharge( outkey="partial_charge_negative", keep_sign="n" )
  # features["feat_backbone"] = nearl.features.Backbone( outkey="backboness" )
  # features["feat_sidechain"] = nearl.features.Backbone( outkey="sidechain", reverse=True )
  # features["feat_donor"] = nearl.features.HBondDonor( outkey="hbond_donor" )
  # features["feat_acceptor"] = nearl.features.HBondAcceptor( outkey="hbond_acceptor" )
  # features["feat_heavyatom"] = nearl.features.HeavyAtom( outkey="heavy_atom" )
  # features["feat_ring"] = nearl.features.Ring( outkey="ring" )
  # features["feat_electronegativity"] = nearl.features.Electronegativity( outkey="electronegativity" )
         
  # # Atom types 
  features["feat_type_H"] = nearl.features.AtomType( focus_element=1, outkey="atomtype_hydrogen" )
  features["feat_type_C"] = nearl.features.AtomType( focus_element=6, outkey="atomtype_carbon" )
  features["feat_type_N"] = nearl.features.AtomType( focus_element=7, outkey="atomtype_nitrogen" )
  features["feat_type_O"] = nearl.features.AtomType( focus_element=8, outkey="atomtype_oxygen" )
  features["feat_type_S"] = nearl.features.AtomType( focus_element=16, outkey="atomtype_sulfur" )

  features["feat_dyn1.0.0"] = nearl.features.MarchingObservers(
    weight_type="atomic_number", obs="distinct_count", 
    agg = "mean", 
    selection=":MOL&!@H=",
    outkey="obs_distinct_atomic_number"
  )
  features["feat_dyn1.0.1"] = nearl.features.MarchingObservers(
    weight_type="atomic_number", obs="distinct_count", 
    agg = "standard_deviation", 
    selection=":MOL<:5&!@H=",
    outkey="obs_distinct_atomic_number"
  )

  features["feat_dyn1.1.0"] = nearl.features.MarchingObservers(
    weight_type="residue_id", obs="distinct_count", 
    agg = "mean", 
    outkey="obs_distinct_resid"
  )

  features["feat_dyn1.1.1"] = nearl.features.MarchingObservers(
    weight_type="mass", obs="density", 
    agg = "mean", 
    outkey="obs_density_mass"
  )

  # features["feat_dyn4"] = nearl.features.DensityFlow(
  #   weight_type="mass", agg="mean",
  #   outkey="df_mass"
  # )
  
  # features["feat_dyn5"] = nearl.features.DensityFlow(
  #   weight_type="atomic_number", agg="mean",
  #   outkey="df_atomic_number"
  # )

  # features["feat_dyn6"] = nearl.features.DensityFlow(
  #   weight_type="partial_charge", agg="mean", 
  #   keep_sign="p", 
  #   outkey="df_positive_charge"
  # )

  # features["feat_dyn7"] = nearl.features.DensityFlow(
  #   weight_type="partial_charge", agg="mean", 
  #   keep_sign="n", 
  #   outkey="df_negative_charge"
  # )

  features["feat_sel_lig"] = nearl.features.Selection(
    selection=":MOL", 
    selection_type="mask",
    outkey = "ligand_annotation"
  )

  features["feat_prot"] = nearl.features.Selection(
    selection="!:MOL", 
    selection_type="mask",
    outkey = "protein_annotation"
  )

  # No label
  features["pk_original"] = nearl.features.LabelAffinity(
    baseline_map=nearl.data.GENERAL_SET, 
    outkey="pk_original"
  )

  features["label"] = nearl.features.LabelPCDT(
    selection=":MOL", 
    baseline_map=nearl.data.GENERAL_SET, 
    outkey="label"
  )

  feat.register_features(features)
  
  feat.main_loop(20)

