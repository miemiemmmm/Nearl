import os, argparse, time, json
import numpy as np
from collections import OrderedDict

import nearl 
import nearl.data


def parser(): 
  parser = argparse.ArgumentParser(description="Featurize the misato trajectories")
  parser.add_argument("-f", "--pdbcodes", type=str, required=True, help="The file containing the list of pdb codes")
  parser.add_argument("-t", "--complex_template", type=str, required=True, help="The template string to locate pdb file, e.g. {}_complex.pdb where {} is replaced by the pdb code listed in the --pdbcodes argument") 

  parser.add_argument("-o", "--output_dir", type=str, default="",help="The output directory")
  parser.add_argument("--task_nr", type=int, default=1, help="The task number to run")
  parser.add_argument("--task_index", type=int, default=0, help="The task index to run")
  parser.add_argument("--h5prefix", type=str, default="Output", help="The prefix of the output h5 file")
  args = parser.parse_args()
  if not os.path.exists(args.output_dir):
    raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")
  return args

def find_files(pdbcodes, template): 
  
  codes = pdbcodes
  complex_files = []
  for p in codes: 
    filename = template.format(p)
    if os.path.exists(filename): 
      complex_files.append([filename])
    else:
      # pass
      raise FileNotFoundError(f"Complex file {filename} does not exist.") 
  return complex_files

def commandline_interface(args, trajlists): 

  VOX_cutoff = 4 
  VOX_SIGMA = 0.25

  featureset = OrderedDict()
  featureset["H"] = nearl.features.AtomType(selection="!:LIG", focus_element=1, outkey="static_H_prot", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["C"] = nearl.features.AtomType(selection="!:LIG", focus_element=6, outkey="static_C_prot", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["N"] = nearl.features.AtomType(selection="!:LIG", focus_element=7, outkey="static_N_prot", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["O"] = nearl.features.AtomType(selection="!:LIG", focus_element=8, outkey="static_O_prot", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["S"] = nearl.features.AtomType(selection="!:LIG", focus_element=16, outkey="static_S_prot", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["H_"] = nearl.features.AtomType(selection=":LIG", focus_element=1, outkey="static_H_lig", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["C_"] = nearl.features.AtomType(selection=":LIG", focus_element=6, outkey="static_C_lig", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["N_"] = nearl.features.AtomType(selection=":LIG", focus_element=7, outkey="static_N_lig", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["O_"] = nearl.features.AtomType(selection=":LIG", focus_element=8, outkey="static_O_lig", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["S_"] = nearl.features.AtomType(selection=":LIG", focus_element=16, outkey="static_S_lig", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["pk"] = nearl.features.LabelAffinity(baseline_map=nearl.data.GENERAL_SET, outkey="pk_original")


  FEATURIZER_PARMS = {
    "dimensions": 32, 
    "lengths": 20, 
    "time_window": 1,    # Static structures only 
    # For default setting inference of registered features 
    # "sigma": 1.5, 
    "outfile": outputfile, 
  }
  print(pdbids)

  # Prepare components for featurization 
  trajloader = nearl.io.TrajectoryLoader(trajlists, superpose=True, mask="!:T3P", trajid=pdbids)
  featurizer  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
  featurizer.register_trajloader(trajloader)
  featurizer.register_focus([":LIG"], "mask")
  featurizer.register_features(featureset)

  featurizer.main_loop(8)


if __name__ == "__main__":
  nearl.update_config(verbose = False, debug = False)

  args = parser()
  args = vars(args)
  print(args)

  task_nr = args.get("task_nr")
  task_index = args.get("task_index")
  prefix = args.get("h5prefix")
  outputfile = os.path.join(os.path.abspath(args["output_dir"]), f"{prefix}{task_index}.h5") 


  template = args.get("complex_template")
  pdbcodes = args.get("pdbcodes") 
  with open(pdbcodes, "r") as f: 
    pdbcodes = f.read().strip("\n").split("\n")
  complex_files = find_files(pdbcodes, template)
  complex_files = np.array_split(complex_files, task_nr)[task_index]
  pdbids = np.array_split(pdbcodes, task_nr)[task_index]

  print(f"Found all complex files: {complex_files.__len__()}")
  if len(complex_files) != len(pdbids):
    raise ValueError("The number of pdbids does not match the number of complex files")
  

  commandline_interface(args, complex_files)
