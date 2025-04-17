import os, argparse, time, json
import numpy as np
from collections import OrderedDict

import nearl.featurizer, nearl.features
from nearl.io.traj import MisatoTraj


# Define the way to generation labels for the misato trajectories

def parser(): 
  parser = argparse.ArgumentParser(description="Featurize the misato trajectories")
  parser.add_argument("-f", "--pdbcodes", type=str, required=True, help="The file containing the list of pdb codes")
  parser.add_argument("-m", "--misato_dir", type=str, required=True, help="The directory of the misato database")
  parser.add_argument("-o", "--output_dir", type=str, default="",help="The output directory")
  parser.add_argument("-t", "--feature_type", type=int, default=1, help="Feature type to benchmark.")

  # Featurization settings
  parser.add_argument("-d", "--dimension", type=int, default=32, help="The dimension of the feature vector")
  parser.add_argument("-l", "--length", type=int, default=24, help="The length of the bounding box")
  parser.add_argument("-c", "--cutoff", type=float, default=5.0, help="The cutoff distance for the feature selection")
  parser.add_argument("-s", "--sigma", type=float, default=1.5, help="The sigma value for the feature selection")
  parser.add_argument("-w", "--windowsize", type=int, default=20, help="The time window for the feature selection")

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


def get_features(sigma): 
  features = OrderedDict()

  # Modify the feature type here. 
  features["feat_prot"] = nearl.features.Mass(selection="!:MOL", outkey="mass_prot", sigma=sigma)
  features["feat_lig"] = nearl.features.Mass(selection=":MOL", outkey="mass_lig", sigma=sigma)
  
  
  features["mo_prot2"] = nearl.features.MarchingObservers(weight_type="mass", selection="!:MOL", obs="mean_distance", agg="standard_deviation", outkey="mobs_prot")
  features["mo_lig2"] = nearl.features.MarchingObservers(weight_type="mass", selection=":MOL", obs="mean_distance", agg="standard_deviation", outkey="mobs_lig")
  
  features["pdf_prot2"] = nearl.features.DensityFlow(weight_type="mass", selection="!:MOL", agg="standard_deviation", outkey="pdb_prot", sigma=sigma)
  features["pdf_lig2"] = nearl.features.DensityFlow(weight_type="mass", selection=":MOL", agg="standard_deviation", outkey="pdb_lig", sigma=sigma)

  return features 


if __name__ == "__main__":
  """
  Usage: 
  python3 /MieT5/Nearl/scripts/benchmark_misatofeat.py -f /MieT5/Nearl/data/casf2016_test.txt -o /tmp/ -t 1 -d 32 -m /Matter/misato_database/ -c 2.5 -s 1.5 

  """
  nearl.update_config(verbose = False, debug = False,)
  # nearl.update_config(verbose = True, debug = True)
  
  args = parser()
  args = vars(args)
  print(json.dumps(args, indent=2))
  task_nr = args.get("task_nr")
  task_index = args.get("task_index")
  h5_prefix = args.get("h5prefix")
  feattype = args.get("feature_type")
  outputfile = os.path.join(os.path.abspath(args["output_dir"]), f"{h5_prefix}{task_index}.h5") 
  if os.path.exists(outputfile):
    raise ValueError(f"Output file {outputfile} exists. Please remove it first.")

  # Candidate trajectories 
  misatodir = args.get("misato_dir")
  training_set = args.get("pdbcodes")
  VOX_cutoff = args.get("cutoff")
  VOX_sigma = args.get("sigma")
  WINDOW_SIZE = args.get("windowsize")

  print(f"Input file: {training_set}, Output file: {outputfile}; Task {task_index} of {task_nr}")

  # Initialize featurizer object and register necessary components
  FEATURIZER_PARMS = {
    "dimensions": [args.get("dimension")]*3, 
    "lengths": args.get("length"), 
    "time_window": WINDOW_SIZE,               # TODO: Temporal setting for simple test 
    "outfile": outputfile, 
    "cutoff": VOX_cutoff,
    "padding": VOX_cutoff, 
    "frame_offset": 9, 
  }

  trajlists = get_trajlist(training_set, misatodir)
  trajlists = np.array_split(trajlists, task_nr)[task_index]
  trajids = [i[0] for i in trajlists]
  print(f"Total number of trajectories {trajlists.__len__()}")

  ##############################################################
  np.random.seed(0)
  np.random.shuffle(trajlists)
  #  ['6p85' '/Matter/misato_database/']
  #  ['3u6h' '/Matter/misato_database/']
  #  ['2wcx' '/Matter/misato_database/']
  #  ['2qwe' '/Matter/misato_database/']
  #  ['6rih' '/Matter/misato_database/']
  # trajlists, trajids = trajlists[:100], trajids[:100]   # TODO: Remove this line for production run
  ##############################################################
  loader = nearl.io.TrajectoryLoader(trajlists, trajtype=MisatoTraj, superpose=True, trajid = trajids)
  print(f"Performing the featurization on {len(loader)} trajectories")

  feat  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
  feat.register_trajloader(loader)
  feat.register_focus([":MOL"], "mask")
  
  features = get_features(VOX_sigma)  

  # Labels
  features["pk_original"] = nearl.features.LabelAffinity(baseline_map="/MieT5/Nearl/data/PDBBind_general_v2020.csv", outkey="pk_original")
  # features["label_pcdt"] = nearl.features.LabelPCDT(selection=":MOL", baseline_map="/MieT5/Nearl/data/PDBBind_general_v2020.csv", outkey="label_pcdt")
  print(f"There are {len(features)} features registered: {features.keys()}")

  feat.register_features(features)
  feat.run()
