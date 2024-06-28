"""
Extract static/dynamic featuresfor single residue classification 


"""
import os, sys, argparse
import numpy as np 

import nearl
import nearl.data

from collections import OrderedDict
from nearl.io.traj import MisatoTraj


def get_trajlist(training_set, misatodir): 
  with open(training_set, "r") as f:
    pdbcodes = f.read().strip("\n").split("\n")
  trajlists = [(i, misatodir) for i in pdbcodes]
  return trajlists

def parse_args():
  parser = argparse.ArgumentParser(description="Featurize the misato trajectories")
  parser.add_argument("--task_nr", type=int, default=1, help="The task number to run")
  parser.add_argument("--task_index", type=int, default=0, help="The task index to run")
  parser.add_argument("-o", "--output_dir", type=str, default="",help="The output directory")
  args = parser.parse_args()
  if not os.path.exists(args.output_dir):
    raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")
  return args


def generate_features():
  """
  Generate the dynamic residues for classification 
  """
  args = parse_args()
  args = vars(args)
  print(args)

  task_nr = args.get("task_nr")
  task_index = args.get("task_index")


  nearl.update_config(
    verbose = False, 
    debug = False,
    # verbose = True, 
    # debug = True,
  )

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
    "progressbar": True, 
  }

  # Load trajectories 
  trajlists = get_trajlist(training_set, misatodir)
  trajlists = np.array_split(trajlists, args.get("task_nr"))[args.get("task_index")]
  loader = nearl.io.TrajectoryLoader(trajlists, trajtype=MisatoTraj, superpose=True)
  

  feat = nearl.featurizer.Featurizer(FEATURIZER_PARMS)

  # Register trajectories 
  feat.register_trajloader(loader)
  # NOTE: In this case, no need to register focus 



  # Register the dynamic feature 
  features = OrderedDict()
  features["feat_backbone"] = nearl.features.Backbone(outkey="backboness")
  features["dyna_backbone"] = nearl.features.DynamicFeature(features["feat_backbone"], outkey="dyna_backbone")

  feat.register_features(features)

  # Interate the frames and compute the dynamic feature
  feat.loop_by_residue("dual", process_nr=4, tag_limit=4)


  # Dump the dynamic feature as well as the labels to the disk



if __name__ == "__main__":
  generate_features()

