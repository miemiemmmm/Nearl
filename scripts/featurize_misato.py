import os, argparse, time, json
import numpy as np
from collections import OrderedDict

import nearl 
import nearl.data
from nearl.io.traj import MisatoTraj


# Define the way to generation labels for the misato trajectories

def parser(): 
  parser = argparse.ArgumentParser(description="Featurize the misato trajectories")
  parser.add_argument("-f", "--pdbcodes", type=str, required=True, help="The file containing the list of pdb codes")
  parser.add_argument("-d", "--misato_dir", type=str, default="/Matter/misato_database/", help="The directory of the misato database")
  parser.add_argument("-o", "--output_dir", type=str, default="",help="The output directory")
  parser.add_argument("--task_nr", type=int, default=1, help="The task number to run")
  parser.add_argument("--task_index", type=int, default=0, help="The task index to run")
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
  outputfile = os.path.join(os.path.abspath(args["output_dir"]), f"MisatoOutput{task_index}.h5") 

  # Candidate trajectories 
  misatodir = args.get("misato_dir")
  training_set = args.get("pdbcodes")

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
  trajlists = np.array_split(trajlists, task_nr)[task_index]
  print(f"# Total number of trajectories {trajlists.__len__()}")

  # trajlists = trajlists[:5]   # TODO: Remove this line for production run
  loader = nearl.io.TrajectoryLoader(trajlists, trajtype=MisatoTraj, superpose=True)
  print(f"Performing the featurization on {len(loader)} trajectories")

  # for traj in loader:
  #   print(f"Selected {traj.top.select(':MOL').shape[0]} atoms from {traj.top.n_atoms} atoms")

  feat  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)

  feat.register_trajloader(loader)
  feat.register_focus([":MOL"], "mask")

  #############################################################################
  #############################################################################
  #############################################################################

  # !!! Change the keyword to different names to avoid conflict
  features = OrderedDict()

  # Static features
  features["lig_annotation"] = nearl.features.Selection(selection=":MOL", selection_type="mask", outkey = "ligand_annotation")
  features["prot_annotation"] = nearl.features.Selection(selection="!:MOL", selection_type="mask", outkey = "protein_annotation")
  features["mass_lig"] = nearl.features.Mass( selection=":MOL", outkey="mass_lig" )
  features["mass_prot"] = nearl.features.Mass( selection="!:MOL", outkey="mass_prot" )
  # features["arom_lig"] = nearl.features.Aromaticity( selection=":MOL", outkey="arom_lig" )
  # features["arom_prot"] = nearl.features.Aromaticity( selection="!:MOL", outkey="arom_prot" )
  # features["positive_lig"] = nearl.features.PartialCharge(selection=":MOL", keep_sign = "p", outkey="charge_positive_lig")
  # features["negative_lig"] = nearl.features.PartialCharge(selection=":MOL", keep_sign = "n", outkey="charge_negative_lig")
  # features["positive_prot"] = nearl.features.PartialCharge(selection="!:MOL", keep_sign = "p", outkey = "charge_positive_prot" )
  # features["negative_prot"] = nearl.features.PartialCharge(selection="!:MOL", keep_sign = "n", outkey = "charge_negative_prot" )
  # features["donor_prot"] = nearl.features.HBondDonor(selection="!:MOL", outkey="prot_donor" )
  # features["acceptor_prot"] = nearl.features.HBondAcceptor(selection="!:MOL", outkey="prot_acceptor" )
  # features["donor_lig"] = nearl.features.HBondDonor(selection=":MOL", outkey="prot_donor" )
  # features["acceptor_lig"] = nearl.features.HBondAcceptor(selection=":MOL", outkey="prot_acceptor" )
  # features["ring_lig"] = nearl.features.Ring(selection=":MOL", outkey="ring_lig")
  # features["ring_prot"] = nearl.features.Ring(selection="!:MOL", outkey="ring_prot")


  ##############################################################################
  # features["feat_atomicnumber"] = nearl.features.AtomicNumber( outkey="atomic_number" )
  # features["feat_backbone"] = nearl.features.Backbone( outkey="backboness" )
  # features["feat_sidechain"] = nearl.features.Backbone( selection=":MOL", outkey="sidechain", reverse=True )
  # features["feat_heavyatom"] = nearl.features.HeavyAtom( outkey="heavy_atom" )
  # features["feat_electronegativity"] = nearl.features.Electronegativity( outkey="electronegativity" )
  ##############################################################################

  
         
  # Static atom types 
  features["type_H_Lig"] = nearl.features.AtomType(selection=":MOL", focus_element=1, outkey="lig_type_H")
  features["type_C_Lig"] = nearl.features.AtomType(selection=":MOL", focus_element=6, outkey="lig_type_C")
  features["type_N_Lig"] = nearl.features.AtomType(selection=":MOL", focus_element=7, outkey="lig_type_N")
  features["type_O_Lig"] = nearl.features.AtomType(selection=":MOL", focus_element=8, outkey="lig_type_O")
  features["type_S_Lig"] = nearl.features.AtomType(selection=":MOL", focus_element=16, outkey="lig_type_S")

  features["type_H_Prot"] = nearl.features.AtomType(selection="!:MOL", focus_element=1, outkey="prot_type_H")
  features["type_C_Prot"] = nearl.features.AtomType(selection="!:MOL", focus_element=6, outkey="prot_type_C")
  features["type_N_Prot"] = nearl.features.AtomType(selection="!:MOL", focus_element=7, outkey="prot_type_N")
  features["type_O_Prot"] = nearl.features.AtomType(selection="!:MOL", focus_element=8, outkey="prot_type_O")
  features["type_S_Prot"] = nearl.features.AtomType(selection="!:MOL", focus_element=16, outkey="prot_type_S")
  ##############################################################################

  # Dynamic features
  features["obs_HCount_lig"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="lig_HCount_obs", element_type=1)
  features["obs_CCount_lig"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="lig_CCount_obs", element_type=6)
  features["obs_NCount_lig"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="lig_NCount_obs", element_type=7)
  features["obs_OCount_lig"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="lig_OCount_obs", element_type=8)
  features["obs_SCount_lig"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="lig_SCount_obs", element_type=16)

  features["obs_HCount_prot"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="prot_HCount_obs", element_type=1)
  features["obs_CCount_prot"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="prot_CCount_obs", element_type=6)
  features["obs_NCount_prot"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="prot_NCount_obs", element_type=7)
  features["obs_OCount_prot"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="prot_OCount_obs", element_type=8)
  features["obs_SCount_prot"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="prot_SCount_obs", element_type=16)
  ##############################################################################

  
  # features["feat_dyn1.1.1"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atomic_number", obs="distinct_count", agg = "mean", outkey="lig_atn_distinct_mean" )
  # features["feat_dyn1.1.2"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atomic_number", obs="distinct_count", agg = "standard_deviation", outkey="lig_density_mass_std")
  # features["feat_dyn1.1.3"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atomic_number", obs="distinct_count", agg = "mean", outkey="prot_atn_distinct_mean")
  # features["feat_dyn1.1.4"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atomic_number", obs="distinct_count", agg = "standard_deviation", outkey="prot_density_mass_std" )

  # features["feat_dyn1.1.5"] = nearl.features.MarchingObservers(weight_type="atomic_number", obs="distinct_count", agg = "mean", outkey="obs_density_mass_meam" )
  # features["feat_dyn1.1.6"] = nearl.features.MarchingObservers(weight_type="atomic_number", obs="distinct_count", agg = "standard_deviation", outkey="obs_density_mass_std")

  # features["feat_dyn1.2.1"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="mass", obs="mean_distance", agg = "mean", outkey="lig_mass_distance_mean")
  # features["feat_dyn1.2.2"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="mass", obs="mean_distance", agg = "standard_deviation", outkey="lig_mass_distance_std")
  # features["feat_dyn1.2.3"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="mass", obs="mean_distance", agg = "mean", outkey="prot_mass_distance_mean")
  # features["feat_dyn1.2.4"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="mass", obs="mean_distance", agg = "standard_deviation", outkey="prot_mass_distance_std")
  # features["feat_dyn1.2.5"] = nearl.features.MarchingObservers(weight_type="mass", obs="mean_distance", agg = "mean", outkey="obs_mass_distance_mean")
  
  # features["feat_dyn1.2.6"] = nearl.features.MarchingObservers(weight_type="mass", obs="mean_distance", agg = "standard_deviation", outkey="obs_mass_distance_std")

  # features["feat_dyn1.3.1"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="mass", obs="density", agg = "mean", outkey="lig_mass_density_mean")
  # features["feat_dyn1.3.2"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="mass", obs="density", agg = "standard_deviation", outkey="lig_mass_density_std")
  # features["feat_dyn1.3.3"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="mass", obs="density", agg = "mean",outkey="prot_mass_density_mean")
  # features["feat_dyn1.3.4"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="mass", obs="density", agg = "standard_deviation", outkey="prot_mass_density_std")
  # features["feat_dyn1.3.5"] = nearl.features.MarchingObservers(weight_type="mass", obs="density", agg = "mean", outkey = "obs_mass_density_mean")
  # features["feat_dyn1.3.6"] = nearl.features.MarchingObservers(weight_type="mass", obs="density", agg = "standard_deviation", outkey="obs_mass_density_std" )

  # features["feat_dyn1.4.1"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="mass", obs="dispersion", agg = "mean", outkey="lig_mass_dispersion_mean" )
  # features["feat_dyn1.4.2"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="mass", obs="dispersion", agg = "standard_deviation", outkey="lig_mass_dispersion_std" )
  # features["feat_dyn1.4.3"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="mass", obs="dispersion", agg = "mean", outkey="prot_mass_dispersion_mean")
  # features["feat_dyn1.4.4"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="mass", obs="dispersion", agg = "standard_deviation", outkey="prot_mass_dispersion_std")
  # features["feat_dyn1.4.5"] = nearl.features.MarchingObservers(weight_type="mass", obs="dispersion", agg = "mean",
  #   outkey="obs_mass_dispersion_mean")
  # features["feat_dyn1.4.6"] = nearl.features.MarchingObservers(weight_type="mass", obs="dispersion", agg = "standard_deviation", outkey="obs_mass_dispersion_std")

  # features["feat_dyn1.5.1"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atomic_number", obs="distinct_count", agg = "information_entropy", outkey="lig_atm_distinct_entropy" )
  # features["feat_dyn1.5.2"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atomic_number", obs="distinct_count", agg = "information_entropy", outkey="prot_atm_distinct_entropy" )
  # features["feat_dyn1.5.3"] = nearl.features.MarchingObservers(weight_type="atomic_number", obs="distinct_count", agg = "information_entropy", outkey="obs_atm_distinct_entropy" )

  # features["feat_dyn1.6.1"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="residue_id", obs="distinct_count", agg = "information_entropy", outkey="lig_resid_distinct_entropy" )
  # features["feat_dyn1.6.2"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="residue_id", obs="distinct_count", agg = "information_entropy", outkey="prot_resid_distinct_entropy")
  # features["feat_dyn1.6.3"] = nearl.features.MarchingObservers(weight_type="residue_id", obs="distinct_count", agg = "information_entropy", outkey="obs_resid_distinct_entropy")

  # Property density flow is too slow to compute
  # features["feat_dyn2.1.1"] = nearl.features.DensityFlow(
  #   selection=":MOL", weight_type="mass", agg="mean",
  #   outkey="df_lig_mass_mean"
  # )
  # features["feat_dyn2.1.2"] = nearl.features.DensityFlow(
  #   selection=":MOL", weight_type="mass", agg="standard_deviation",
  #   outkey="df_lig_mass_std"
  # )
  # features["feat_dyn2.1.3"] = nearl.features.DensityFlow(
  #   selection="!:MOL", weight_type="mass", agg="mean",
  #   outkey="df_prot_mass_mean"
  # )
  # features["feat_dyn2.1.4"] = nearl.features.DensityFlow(
  #   selection="!:MOL", weight_type="mass", agg="standard_deviation",
  #   outkey="df_prot_mass_std"
  # )
  # features["feat_dyn2.2.1"] = nearl.features.DensityFlow(
  #   selection=":MOL", weight_type="ring", agg="mean",
  #   outkey="df_lig_mass_mean"
  # )
  # features["feat_dyn2.2.2"] = nearl.features.DensityFlow(
  #   selection=":MOL", weight_type="ring", agg="standard_deviation",
  #   outkey="df_lig_mass_std"
  # )
  # features["feat_dyn2.2.3"] = nearl.features.DensityFlow(
  #   selection="!:MOL", weight_type="ring", agg="mean",
  #   outkey="df_prot_mass_mean"
  # )
  # features["feat_dyn2.2.4"] = nearl.features.DensityFlow(
  #   selection="!:MOL", weight_type="ring", agg="standard_deviation",
  #   outkey="df_prot_mass_std"
  # )



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
    selection=":MOL", 
    baseline_map=nearl.data.GENERAL_SET, 
    outkey="label_pcdt"
  )

  # features["coord_lig"] = nearl.features.Coords( selection=":MOL", outkey="coord_lig" )
  # features["coord_prot"] = nearl.features.Coords( selection="!:MOL", outkey="coord_prot" )
  # features["rffeatures"] = nearl.features.RFFeatures(selection=":MOL", search_cutoff=6, byres=True, outkey="rf_feature")

  print(f"There are {len(features)} features registered: {features.keys()}")

  feat.register_features(features)
  
  feat.main_loop(20)

