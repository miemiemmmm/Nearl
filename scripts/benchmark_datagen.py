import os, argparse, time, json

import h5py as h5 
import numpy as np
import pytraj as pt

from collections import OrderedDict

import nearl 
# import nearl.data
# from nearl.io import Trajectory

def parser(): 
  parser = argparse.ArgumentParser(description="Featurize the trajectories")
  parser.add_argument("-f", "--trajfiles", type=str, required=True, help="Input trajectory file path.")
  parser.add_argument("-t", "--feature-type", type=int, required=True, help="Feature type to benchmark.")
  parser.add_argument("-o", "--outputfile", type=str, default="/tmp/benchmarking_dataset.h5", help="Output file path.")
  parser.add_argument("-c", "--cpus", type=int, default=8, help="Number of CPUs.")
  parser.add_argument("-s", "--frame-span", type=int, default=20, help="The span for each frame-slice.")
  parser.add_argument("--force", type=int, default=0, help="Force to remove the output file.")
  args = parser.parse_args()

  if not os.path.exists(args.trajfiles): 
    raise ValueError(f"Input trajectory file not found: {args.trajfiles}") 
  if os.path.exists(args.outputfile) and args.force: 
    # remove the file
    print(f"Forcing to remove the output file {args.outputfile}.")
    os.remove(args.outputfile)
  elif os.path.exists(args.outputfile) and not args.force: 
    # ask user whether or not to remove the file
    print(f"Output file {args.outputfile} exists. Do you want to remove it? [y/n]")
    answer = input()
    if answer.lower() == "y":
      os.remove(args.outputfile)
    else:
      print(f"Output file {args.outputfile} exists. Please remove it first.")
      return None
  
  return args


def get_features(feattype): 
  features = OrderedDict()
  # nearl.features.AtomType(selection="!:LIG", focus_element=1, outkey="lig_type_H")
  if feattype == 1: 
    features["feat_static"] = nearl.features.AtomType(selection="!:LIG", focus_element=6, outkey="feat_static", cutoff=6)
    outkey = "feat_static"
  elif feattype == 2: 
    features["feat_mo"] = nearl.features.MarchingObservers(selection="!:LIG", weight_type="atom_type", element_type=6, obs="distinct_count", 
                                                           agg="standard_deviation", outkey="feat_mo", cutoff=2.55)
    outkey = "feat_mo"
  elif feattype == 3:
    features["feat_pdf"] = nearl.features.DensityFlow(selection="!:LIG", weight_type="atom_type", element_type=6, outkey="feat_pdf", 
                                                      agg="standard_deviation", cutoff=6)
    outkey = "feat_pdf" 
  return features, outkey 


def commandline_interface(args, outputdim): 
  # Variable parameters to pass: dimensions, time window
  outputfile = args.get("outputfile", "/tmp/benchmarking_dataset.h5")
  if os.path.exists(outputfile) and args["force"]:
    os.remove(outputfile)
  elif os.path.exists(outputfile): 
    raise ValueError(f"Output file {outputfile} exists. Please remove it first.") 
  
  feattype = args["feature_type"]
  trajfile = args["trajfiles"]
  cpu_nr = args.get("cpus", 8)
  frame_span = args.get("frame_span", 20)
  
  outputdim = int(outputdim)
  FEATURIZER_PARMS = {
    "dimensions": outputdim, 
    "lengths": 20, 
    "time_window": frame_span,    # Time window equal to 0.8 ns 
    # For default setting inference of registered features 
    "sigma": 1.5, 
    "outfile": outputfile, 
  }
  print(f"Job parameters: {FEATURIZER_PARMS}")
  featureset, feat_key = get_features(feattype)

  with open(trajfile, "r") as f:
    files = f.read().strip().split("\n")
    trajlists = [(i.split()[0], i.split()[1]) for i in files]

  # Prepare components for featurization 
  trajloader = nearl.io.TrajectoryLoader(trajlists, superpose=True, mask="!:T3P")  
  featurizer  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
  featurizer.register_trajloader(trajloader)
  featurizer.register_focus([":LIG"], "mask")
  featurizer.register_features(featureset)

  # Start the featurization 
  st = time.perf_counter()
  featurizer.main_loop(cpu_nr)
  ed = time.perf_counter()

  # Calculate the throughput 
  with h5.File(outputfile, "r") as f: 
    data_size = f[feat_key].shape[0]
    time_elapsed = ed - st
    print(f"Dimension: {outputdim}; Featurization throughput: {data_size / time_elapsed:6.3f}; Time elapse: {time_elapsed:6.3f}; Feature number: {data_size:3d};")


if __name__ == '__main__':
  """

  Example: 
  python /MieT5/BetaPose/scripts/benchmark_datagen.py -f /tmp/trajlist.txt -t 1 -c 8 -s 20 --force 1
  python /MieT5/BetaPose/scripts/benchmark_datagen.py -f /tmp/trajlist.txt -t 2 -c 8 -s 20 --force 1
  """
  nearl.update_config(verbose = False, debug = False)

  args = parser()
  args = vars(args)
  print(json.dumps(args, indent=2))
  

  DIMS_TO_TEST = [16, 24, 32, 48, 64]
  REPEAT = 10

  for dim in DIMS_TO_TEST: 
    for i in range(REPEAT): 
      print(f"Testing dimension: {dim} for {i+1}th time.") 
      args["outputfile"] = f"/tmp/benchmarking_{dim}_{i}.h5"
      commandline_interface(args, outputdim=dim)

