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


def get_features(cutoff, sigma): 
  features = OrderedDict()

  # Modify the feature type here. 
  # print("Adding static features") 
  features["feat_prot"] = nearl.features.Mass(selection="!:MOL", outkey="protMassStatic", cutoff=cutoff, sigma=sigma)
  features["feat_lig"] = nearl.features.Mass(selection=":MOL", outkey="ligMassStatic", cutoff=cutoff, sigma=sigma)
  # outkey = "feat_static"
  
  # print("Adding marching observers") 
  features["mo_prot"] = nearl.features.MarchingObservers(selection="!:MOL", weight_type="mass", 
                                                          obs="mean_distance", agg="drift", outkey="protDistObsDrift", cutoff=cutoff)
  features["mo_lig"] = nearl.features.MarchingObservers(selection=":MOL", weight_type="mass", 
                                                          obs="mean_distance", agg="drift", outkey="ligDistObsDrift", cutoff=cutoff)
  # outkey = "feat_mo"
  
  # print("Adding probability density flow") 
  features["pdf_prot"] = nearl.features.DensityFlow(selection="!:MOL", weight_type="mass", 
                                                    outkey="protMassPropDrift", agg="drift", cutoff=cutoff, sigma=sigma) 
  features["pdf_lig"] = nearl.features.DensityFlow(selection=":MOL", weight_type="mass", 
                                                    outkey="ligMassPropDrift", agg="drift", cutoff=cutoff, sigma=sigma) 
  outkey = "feat_pdf" 

  return features, outkey 





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

  print(f"Input file: {training_set}, Output file: {outputfile}; Task {task_index} of {task_nr}")

  # Initialize featurizer object and register necessary components
  FEATURIZER_PARMS = {
    "dimensions": [args.get("dimension")]*3, 
    "lengths": args.get("length"), 
    "time_window": 100,               # TODO: Temporal setting for simple test 
    "outfile": outputfile, 
  }

  trajlists = get_trajlist(training_set, misatodir)
  trajlists = np.array_split(trajlists, task_nr)[task_index]
  trajids = [i[0] for i in trajlists]
  print(f"Total number of trajectories {trajlists.__len__()}")

  trajlists, trajids = trajlists[:50], trajids[:50]   # TODO: Remove this line for production run
  loader = nearl.io.TrajectoryLoader(trajlists, trajtype=MisatoTraj, superpose=True, trajid = trajids)
  print(f"Performing the featurization on {len(loader)} trajectories")

  feat  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
  feat.register_trajloader(loader)
  feat.register_focus([":MOL"], "mask")
  
  features, feat_key = get_features(VOX_cutoff, VOX_sigma)  

  # Labels
  features["pk_original"] = nearl.features.LabelAffinity(baseline_map="/MieT5/Nearl/data/PDBBind_general_v2020.csv", outkey="pk_original")
  # features["label_pcdt"] = nearl.features.LabelPCDT(selection=":MOL", baseline_map="/MieT5/Nearl/data/PDBBind_general_v2020.csv", outkey="label_pcdt")
  print(f"There are {len(features)} features registered: {features.keys()}")

  feat.register_features(features)
  feat.run()
