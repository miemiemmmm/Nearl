import os, argparse, time, json
import numpy as np
from collections import OrderedDict

import nearl 
import nearl.data


def parser(): 
  parser = argparse.ArgumentParser(description="Featurize the misato trajectories")
  # Input and output settings
  parser.add_argument("-f", "--pdbcodes", type=str, required=True, help="The file containing the list of pdb codes")
  parser.add_argument("-t", "--complex_template", type=str, required=True, help="The template string to locate pdb file, e.g. {}_complex.pdb where {} is replaced by the pdb code listed in the --pdbcodes argument") 
  parser.add_argument("-o", "--output_dir", type=str, default="",help="The output directory")

  # Featurization settings
  parser.add_argument("-d", "--dimension", type=int, default=32, help="The dimension of the feature vector")
  parser.add_argument("-l", "--length", type=int, default=20, help="The length of the bounding box")
  parser.add_argument("-c", "--cutoff", type=float, default=2.5, help="The cutoff distance for the feature selection")
  parser.add_argument("-s", "--sigma", type=float, default=1.5, help="The sigma value for the feature selection")

  # Accessory settings 
  parser.add_argument("--task_nr", type=int, default=1, help="The task number to run")
  parser.add_argument("--task_index", type=int, default=0, help="The task index to run")
  parser.add_argument("--cpu_nr", type=int, default=6, help="The number of CPUs to use")
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

def commandline_interface(args, trajlist, pdbids): 
  VOX_length = args.get("length")
  VOX_dim = args.get("dimension")
  VOX_cutoff = args.get("cutoff")
  VOX_SIGMA = args.get("sigma")

  FEATURIZER_PARMS = {
    "dimensions": VOX_dim, 
    "lengths": VOX_length, 
    "time_window": 1,    # Static structures only 
    "outfile": outputfile, 
  }
  print(f"Featurization follows: Length: {VOX_length} | Dimension: {VOX_dim} | Cutoff: {VOX_cutoff} | Sigma: {VOX_SIGMA}")

  featureset = OrderedDict()
  featureset["all_ann"] = nearl.features.Selection(selection=":*", outkey="all_annotation", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["lig_ann"] = nearl.features.Selection(selection="!:LIG", outkey="lig_annotation", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["prot_ann"] = nearl.features.Selection(selection=":LIG", outkey="prot_annotation", cutoff=VOX_cutoff, sigma=VOX_SIGMA)


  featureset["aromatic"] = nearl.features.Aromaticity(selection=":*", outkey="aromatic", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["hb_donor"] = nearl.features.HBondDonor(selection=":*", outkey="hb_donor", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["hb_acceptor"] = nearl.features.HBondAcceptor(selection=":*", outkey="hb_acceptor", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["H"] = nearl.features.AtomType(selection=":*", focus_element=1, outkey="static_H", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["C"] = nearl.features.AtomType(selection=":*", focus_element=6, outkey="static_C", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["N"] = nearl.features.AtomType(selection=":*", focus_element=7, outkey="static_N", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["O"] = nearl.features.AtomType(selection=":*", focus_element=8, outkey="static_O", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["S"] = nearl.features.AtomType(selection=":*", focus_element=16, outkey="static_S", cutoff=VOX_cutoff, sigma=VOX_SIGMA)

  # Aromaticity
  featureset["aromatic"] = nearl.features.Aromaticity(selection="!:LIG", outkey="aromatic_prot", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["aromatic_"] = nearl.features.Aromaticity(selection=":LIG", outkey="aromatic_lig", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  # HB donor/acceptor
  featureset["hb_donor"] = nearl.features.HBondDonor(selection="!:LIG", outkey="hb_donor_prot", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["hb_donor_"] = nearl.features.HBondDonor(selection=":LIG", outkey="hb_donor_lig", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["hb_acceptor"] = nearl.features.HBondAcceptor(selection="!:LIG", outkey="hb_acceptor_prot", cutoff=VOX_cutoff, sigma=VOX_SIGMA)
  featureset["hb_acceptor_"] = nearl.features.HBondAcceptor(selection=":LIG", outkey="hb_acceptor_lig", cutoff=VOX_cutoff, sigma=VOX_SIGMA)

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


  # Simple discretization 
  # featureset["H"] = nearl.features.Discretize(selection=":*", focus_element=1, outkey="disc_H")
  # featureset["C"] = nearl.features.Discretize(selection=":*", focus_element=6, outkey="disc_C")
  # featureset["N"] = nearl.features.Discretize(selection=":*", focus_element=7, outkey="disc_N")
  # featureset["O"] = nearl.features.Discretize(selection=":*", focus_element=8, outkey="disc_O")
  # featureset["S"] = nearl.features.Discretize(selection=":*", focus_element=16, outkey="disc_S")

  # featureset["H_"] = nearl.features.Discretize(selection="!:LIG", focus_element=1, outkey="disc_H_prot")
  # featureset["C_"] = nearl.features.Discretize(selection="!:LIG", focus_element=6, outkey="disc_C_prot")
  # featureset["N_"] = nearl.features.Discretize(selection="!:LIG", focus_element=7, outkey="disc_N_prot")
  # featureset["O_"] = nearl.features.Discretize(selection="!:LIG", focus_element=8, outkey="disc_O_prot")
  # featureset["S_"] = nearl.features.Discretize(selection="!:LIG", focus_element=16, outkey="disc_S_prot")

  # featureset["H__"] = nearl.features.Discretize(selection=":LIG", focus_element=1, outkey="disc_H_lig")
  # featureset["C__"] = nearl.features.Discretize(selection=":LIG", focus_element=6, outkey="disc_C_lig")
  # featureset["N__"] = nearl.features.Discretize(selection=":LIG", focus_element=7, outkey="disc_N_lig")
  # featureset["O__"] = nearl.features.Discretize(selection=":LIG", focus_element=8, outkey="disc_O_lig")
  # featureset["S__"] = nearl.features.Discretize(selection=":LIG", focus_element=16, outkey="disc_S_lig")
  # featureset["pk"] = nearl.features.LabelAffinity(baseline_map=nearl.data.GENERAL_SET, outkey="pk_original")
  
  for i,j in zip(trajlist, pdbids): 
    print(f"Traj file: {i} | PDB code: {j}")
    if j not in i[0]: 
      raise ValueError(f"The pdb code {j} is not in the trajectory file {i}")

  # Prepare components for featurization 
  trajloader = nearl.io.TrajectoryLoader(trajlist, superpose=True, mask="!:T3P", trajid=pdbids)
  featurizer = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
  featurizer.register_trajloader(trajloader)
  featurizer.register_focus([":LIG"], "mask")
  featurizer.register_features(featureset)
  featurizer.main_loop(args.get("cpu_nr"))


if __name__ == "__main__":
  nearl.update_config(verbose = False, debug = False)
  # nearl.update_config(verbose = True, debug = True)

  args = parser()
  args = vars(args)
  print(args)

  task_nr = args.get("task_nr")
  task_index = args.get("task_index")
  prefix = args.get("h5prefix")
  outputfile = os.path.join(os.path.abspath(args["output_dir"]), f"{prefix}{task_index}.h5") 

  # Find the complex files and their PDB code
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
  
  # Run the featurization 
  commandline_interface(args, complex_files, pdbids)
