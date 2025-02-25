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
  parser.add_argument("-m", "--misato_dir", type=str, default="/Matter/misato_database/", help="The directory of the misato database")
  parser.add_argument("-o", "--output_dir", type=str, default="",help="The output directory")

  # Featurization settings
  parser.add_argument("-d", "--dimension", type=int, default=32, help="The dimension of the feature vector")
  parser.add_argument("-l", "--length", type=int, default=20, help="The length of the bounding box")
  parser.add_argument("-c", "--cutoff", type=float, default=2.5, help="The cutoff distance for the feature selection")
  parser.add_argument("-s", "--sigma", type=float, default=1.5, help="The sigma value for the feature selection")

  parser.add_argument("--h5prefix", type=str, default="Output", help="The prefix of the output h5 file")
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
  nearl.update_config(verbose = False, debug = False,)
  # nearl.update_config(verbose = True, debug = True)
  
  args = parser()
  args = vars(args)
  print(args)
  task_nr = args.get("task_nr")
  task_index = args.get("task_index")
  h5_prefix = args.get("h5prefix")
  outputfile = os.path.join(os.path.abspath(args["output_dir"]), f"{h5_prefix}{task_index}.h5") 

  # Candidate trajectories 
  misatodir = args.get("misato_dir")
  training_set = args.get("pdbcodes")
  VOX_cutoff = args.get("cutoff")
  VOX_sigma = args.get("sigma")

  print(f"Input file: {training_set}, Output file: {outputfile}; Task {task_index} of {task_nr}")

  # Initialize featurizer object and register necessary components
  FEATURIZER_PARMS = {
    "dimensions": args.get("dimension"), 
    "lengths": args.get("length"), 
    "time_window": 10, 
    "outfile": outputfile, 
  }

  trajlists = get_trajlist(training_set, misatodir)
  trajlists = np.array_split(trajlists, task_nr)[task_index]
  trajids = [i[0] for i in trajlists]
  print(f"Total number of trajectories {trajlists.__len__()}")

  # trajlists = trajlists[:5]   # TODO: Remove this line for production run
  # trajids = trajids[:5]       # TODO: Remove this line for production run
  loader = nearl.io.TrajectoryLoader(trajlists, trajtype=MisatoTraj, superpose=True, trajid = trajids)
  print(f"Performing the featurization on {len(loader)} trajectories")

  feat  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
  feat.register_trajloader(loader)
  feat.register_focus([":MOL"], "mask")

  #############################################################################
  #############################################################################
  #############################################################################

  # !!! Change the keyword to different names to avoid conflict
  features = OrderedDict()

  # Static features
  # features["lig_annotation"] = nearl.features.Selection(selection=":MOL", selection_type="mask", outkey = "ligand_annotation", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["prot_annotation"] = nearl.features.Selection(selection="!:MOL", selection_type="mask", outkey = "protein_annotation", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["mass_lig"] = nearl.features.Mass( selection=":MOL", outkey="mass_lig", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["mass_prot"] = nearl.features.Mass( selection="!:MOL", outkey="mass_prot", cutoff=VOX_cutoff, sigma=VOX_sigma)

  # features["arom_lig"] = nearl.features.Aromaticity(selection=":MOL", outkey="lig_arom", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["arom_prot"] = nearl.features.Aromaticity(selection="!:MOL", outkey="prot_arom", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["positive_lig"] = nearl.features.PartialCharge(selection=":MOL", charge_type="topology", keep_sign = "p", outkey="charge_positive_lig", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["negative_lig"] = nearl.features.PartialCharge(selection=":MOL", charge_type="topology", keep_sign = "n", outkey="charge_negative_lig", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["positive_prot"] = nearl.features.PartialCharge(selection="!:MOL", charge_type="topology", keep_sign = "p", outkey = "charge_positive_prot", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["negative_prot"] = nearl.features.PartialCharge(selection="!:MOL", charge_type="topology", keep_sign = "n", outkey = "charge_negative_prot", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["donor_prot"] = nearl.features.HBondDonor(selection="!:MOL", outkey="prot_donor", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["acceptor_prot"] = nearl.features.HBondAcceptor(selection="!:MOL", outkey="prot_acceptor", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["donor_lig"] = nearl.features.HBondDonor(selection=":MOL", outkey="lig_donor", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["acceptor_lig"] = nearl.features.HBondAcceptor(selection=":MOL", outkey="lig_acceptor", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["ring_lig"] = nearl.features.Ring(selection=":MOL", outkey="lig_ring", cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["ring_prot"] = nearl.features.Ring(selection="!:MOL", outkey="prot_ring", cutoff=VOX_cutoff, sigma=VOX_sigma)

  features["H"] = nearl.features.AtomType(selection="!:MOL", focus_element=1,  outkey="static_H_prot", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["C"] = nearl.features.AtomType(selection="!:MOL", focus_element=6,  outkey="static_C_prot", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["N"] = nearl.features.AtomType(selection="!:MOL", focus_element=7,  outkey="static_N_prot", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["O"] = nearl.features.AtomType(selection="!:MOL", focus_element=8,  outkey="static_O_prot", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["S"] = nearl.features.AtomType(selection="!:MOL", focus_element=16, outkey="static_S_prot", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["H_"] = nearl.features.AtomType(selection=":MOL", focus_element=1,  outkey="static_H_lig", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["C_"] = nearl.features.AtomType(selection=":MOL", focus_element=6,  outkey="static_C_lig", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["N_"] = nearl.features.AtomType(selection=":MOL", focus_element=7,  outkey="static_N_lig", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["O_"] = nearl.features.AtomType(selection=":MOL", focus_element=8,  outkey="static_O_lig", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["S_"] = nearl.features.AtomType(selection=":MOL", focus_element=16, outkey="static_S_lig", cutoff=VOX_cutoff, sigma=VOX_sigma)

  ##############################################################################
  # features["feat_atomicnumber"] = nearl.features.AtomicNumber( outkey="atomic_number" )
  # features["feat_backbone"] = nearl.features.Backbone( outkey="backboness" )
  # features["feat_sidechain"] = nearl.features.Backbone( selection=":MOL", outkey="sidechain", reverse=True )
  # features["feat_heavyatom"] = nearl.features.HeavyAtom( outkey="heavy_atom" )
  # features["feat_electronegativity"] = nearl.features.Electronegativity( outkey="electronegativity" )
  ##############################################################################
  # Dynamic features
  # features["obs_HCount_lig"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="mo_HCount_lig", element_type=1, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["obs_CCount_lig"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="mo_CCount_lig", element_type=6, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["obs_NCount_lig"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="mo_NCount_lig", element_type=7, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["obs_OCount_lig"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="mo_OCount_lig", element_type=8, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["obs_SCount_lig"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="mo_SCount_lig", element_type=16, cutoff=VOX_cutoff, sigma=VOX_sigma)

  # features["obs_HCount_prot"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="mo_HCount_prot", element_type=1, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["obs_CCount_prot"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="mo_CCount_prot", element_type=6, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["obs_NCount_prot"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="mo_NCount_prot", element_type=7, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["obs_OCount_prot"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="mo_OCount_prot", element_type=8, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["obs_SCount_prot"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="mo_SCount_prot", element_type=16, cutoff=VOX_cutoff, sigma=VOX_sigma)

  features["obs_lig_anno"]  = nearl.features.MarchingObservers(selection=":MOL", weight_type="uniformed", obs="distinct_count", agg = "standard_deviation", outkey="mo_PCount_lig", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["obs_prot_anno"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="uniformed", obs="distinct_count", agg = "standard_deviation", outkey="mo_PCount_prot", cutoff=VOX_cutoff, sigma=VOX_sigma)

  ##############################################################################
  # features["feat_CML"] = nearl.features.DensityFlow(selection=":MOL", weight_type="atom_type", agg="mean", outkey="pdf_CM_lig", element_type=6, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_CSL"] = nearl.features.DensityFlow(selection=":MOL", weight_type="atom_type", agg="standard_deviation", outkey="pdf_CS_lig", element_type=6, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_OML"] = nearl.features.DensityFlow(selection=":MOL", weight_type="atom_type", agg="mean", outkey="pdf_OM_lig", element_type=8, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_OSL"] = nearl.features.DensityFlow(selection=":MOL", weight_type="atom_type", agg="standard_deviation", outkey="pdf_OS_lig", element_type=8, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_NML"] = nearl.features.DensityFlow(selection=":MOL", weight_type="atom_type", agg="mean", outkey="pdf_NM_lig", element_type=7, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_NSL"] = nearl.features.DensityFlow(selection=":MOL", weight_type="atom_type", agg="standard_deviation", outkey="pdf_NS_lig", element_type=7, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_HML"] = nearl.features.DensityFlow(selection=":MOL", weight_type="atom_type", agg="mean", outkey="pdf_HM_lig", element_type=1, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_HSL"] = nearl.features.DensityFlow(selection=":MOL", weight_type="atom_type", agg="standard_deviation", outkey="pdf_HS_lig", element_type=1, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_SML"] = nearl.features.DensityFlow(selection=":MOL", weight_type="atom_type", agg="mean", outkey="pdf_SM_lig", element_type=16, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_SSL"] = nearl.features.DensityFlow(selection=":MOL", weight_type="atom_type", agg="standard_deviation", outkey="pdf_SS_lig", element_type=16, cutoff=VOX_cutoff, sigma=VOX_sigma)

  
  
  # features["feat_CMP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="atom_type", agg="mean", outkey="pdf_CM_prot" , element_type=6, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_CSP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="atom_type", agg="standard_deviation", outkey="pdf_CS_prot", element_type=6, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_OMP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="atom_type", agg="mean", outkey="pdf_OM_prot", element_type=8, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_OSP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="atom_type", agg="standard_deviation", outkey="pdf_OS_prot", element_type=8, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_NMP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="atom_type", agg="mean", outkey="pdf_NM_prot", element_type=7, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_NSP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="atom_type", agg="standard_deviation", outkey="pdf_NS_prot", element_type=7, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_HMP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="atom_type", agg="mean", outkey="pdf_HM_prot", element_type=1, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_HSP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="atom_type", agg="standard_deviation", outkey="pdf_HS_prot", element_type=1, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_SMP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="atom_type", agg="mean", outkey="pdf_SM_prot", element_type=16, cutoff=VOX_cutoff, sigma=VOX_sigma)
  # features["feat_SSP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="atom_type", agg="standard_deviation", outkey="pdf_SS_prot", element_type=16, cutoff=VOX_cutoff, sigma=VOX_sigma)

  features["feat_SML"] = nearl.features.DensityFlow(selection=":MOL", weight_type="uniformed", agg="mean", outkey="pdf_PM_lig", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["feat_SSL"] = nearl.features.DensityFlow(selection=":MOL", weight_type="uniformed", agg="standard_deviation", outkey="pdf_PS_lig", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["feat_SMP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="uniformed", agg="mean", outkey="pdf_PM_prot", cutoff=VOX_cutoff, sigma=VOX_sigma)
  features["feat_SSP"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="uniformed", agg="standard_deviation", outkey="pdf_PS_prot", cutoff=VOX_cutoff, sigma=VOX_sigma)


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

  
  

  # Labels
  features["pk_original"] = nearl.features.LabelAffinity(baseline_map=nearl.data.GENERAL_SET, outkey="pk_original")
  features["stepping"] = nearl.features.LabelStepping(baseline_map=nearl.data.GENERAL_SET, outkey="pk_stepping")
  features["label_pcdt"] = nearl.features.LabelPCDT(selection=":MOL", baseline_map=nearl.data.GENERAL_SET, outkey="label_pcdt")

  print(f"There are {len(features)} features registered: {features.keys()}")
  feat.register_features(features)
  feat.main_loop()

