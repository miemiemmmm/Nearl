"""
Extract static/dynamic featuresfor single residue classification 


"""
import os, sys, argparse
import numpy as np 

import nearl
import nearl.data

from collections import OrderedDict
from nearl.io.traj import MisatoTraj

nearl.update_config(
  verbose = False, 
  debug = False,
  # verbose = True, 
  # debug = True,
)

def parse_args():
  parser = argparse.ArgumentParser(description="Featurize the misato trajectories")
  # parser.add_argument("-f", "--pdbcodes", type=str, required=True, help="The file containing the list of pdb codes")
  parser.add_argument("-f", "--pdbcodes", type=str, help="The file containing the list of pdb codes")
  parser.add_argument("-d", "--misato_dir", type=str, default="/Matter/misato_database/", help="The directory of the misato database")
  parser.add_argument("-o", "--output_dir", type=str, default="",help="The output directory")
  parser.add_argument("--task_nr", type=int, default=1, help="The task number to run")
  parser.add_argument("--task_index", type=int, default=0, help="The task index to run")

  args = parser.parse_args()
  # TODO: check the files after coding of the function
  # if not os.path.exists(args.output_dir):
  #   raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")
  return args


def generate_features(settings):
  """
  Generate the dynamic residues for classification 
  """
  task_nr = args.get("task_nr")
  task_index = args.get("task_index")
  misatodir = settings.get("misato_dir")
  pdbcode_file = settings.get("pdbcodes")
  output_dir = settings.get("output_dir")
  outputfile = os.path.join(os.path.abspath(output_dir), f"MisatoOutput{task_index}.h5") 
  print(f"Input file: {pdbcode_file}, Output file: {outputfile}; Task {task_index} of {task_nr}")
  
  # Initialize featurizer object and register necessary components
  FEATURIZER_PARMS = {
    "dimensions": 32, 
    "lengths": 16, 
    "time_window": 50, 

    # For default setting inference of registered features
    "sigma": 1.5, 
    "cutoff": 2.55, 
    "outfile": outputfile, 
    "hdf_compress_level": 0,

    # Other options
    "progressbar": True, 
  }

  # Get the pair arguments for MisatoTraj
  with open(pdbcode_file, "r") as f:
    pdbcodes = f.read().strip("\n").split("\n")
  trajlists = [(i, misatodir) for i in pdbcodes]
  # Load trajectories 
  trajlists = np.array_split(trajlists, args.get("task_nr"))[args.get("task_index")]
  loader = nearl.io.TrajectoryLoader(trajlists, trajtype=MisatoTraj, superpose=True)
  

  feat = nearl.featurizer.Featurizer(FEATURIZER_PARMS)

  # Register trajectories 
  feat.register_trajloader(loader)
  # NOTE: In this case, no need to register focus 

  # Register the dynamic feature 
  features = OrderedDict()
  features["feat_backbone"] = nearl.features.Mass(outkey="mass")
  features["feat_dyna"] = nearl.features.MarchingObservers(weight_type="atom_type", obs="distinct_count", agg = "standard_deviation", outkey="lig_CCount_obs", element_type=6)

  # features[""]


  feat.register_features(features)

  # Interate the frames and compute the dynamic feature
  feat.loop_by_residue("dual", process_nr=4, tag_limit=4)

  # Dump the dynamic feature as well as the labels to the disk



if __name__ == "__main__":
  args = parse_args()
  args = vars(args)
  print(args)

  # misatodir = 
  # training_set = 
  # training_set = "/MieT5/BetaPose/data/misato_test.txt"

  settings = {
    "pdbcodes": "/MieT5/BetaPose/data/misato_train.txt",
    "misato_dir": "/Matter/misato_database/",
    "output_dir": "/tmp/",
  }
  args.update(settings)

  generate_features(args)

