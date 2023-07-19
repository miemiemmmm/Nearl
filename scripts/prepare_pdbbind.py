import os, time
from itertools import chain

import numpy as np
import pytraj as pt
import pandas as pd

import dask 
from dask.distributed import Client

from BetaPose import chemtools, trajloader, features, data_io, utils, printit, savelog

class FeatureLabel(features.Feature):
  def __init__(self, affinity_file, delimiter="\t", header=None):
    """
    Use the topology name as the identifier of the molecule since it specific to the PDB ID of the protein
    Pre-generate the affinity table and use the topology name list to index the affinity
    Then apply the penalties to the affinity label
    Input:
      affinity_file: the file containing the affinity table (str, Path to the affinity table)
      delimiter: the delimiter of the affinity table (str, default: "\t")
    Output:
      label: the affinity label of the frame (float, baseline_affinity*penalty1*penalty2)
    """
    super().__init__();
    self.TABLE = pd.read_csv(affinity_file, delimiter=delimiter, header=None);
    # PDB in the first column and the affinity in the second column
    PLACEHOLDER = "#TEMPLATE#";
    self.TOPFILE_TEMPLATE = "/media/yzhang/MieT5/BetaPose/data/complexes/#TEMPLATE#_complex.pdb";
    self.FILES = [self.TOPFILE_TEMPLATE.replace(PLACEHOLDER, i) for i in self.TABLE[0].values];
    notfoundfile = 0;
    for i in self.FILES:
      if not os.path.exists(i):
        print(f"File {i} not found");
        notfoundfile += 1;
    if notfoundfile > 0:
      print(f"Total {notfoundfile} files not found");
    else:
      print(f"All files found !!!");

  def featurize(self):
    topfile = self.traj.top_filename;
    if topfile in self.FILES:
      idx = self.FILES.index(topfile);
      baseline_affinity = self.TABLE[1].values[idx];
    else:
      print(f"File {topfile} not found in the affinity table");
      print(f"Please check the file name template: {self.TOPFILE_TEMPLATE}");
      raise IOError("File not found");
    penalty1 = 1
    penalty2 = 1
    return baseline_affinity*penalty1*penalty2;

def combine_complex(idx, row):
  ligfile = os.path.join(ref_filedir, f"{row[0]}/{row[0]}_ligand.mol2")
  profile = os.path.join(ref_filedir, f"{row[0]}/{row[0]}_protein.pdb")
  if False not in [profile, ligfile]: 
    print(f"Processing Molecule {idx}: {row[0]}")
    try: 
      complex_str = chemtools.combine_molpdb(ligfile, profile, 
                                           outfile=os.path.join(out_filedir, f"{row[0]}_complex.pdb"))
      return True;
    except: 
      try: 
        ligfile = os.path.join(ref_filedir, f"{row[0]}/{row[0]}_ligand.sdf")
        complex_str = chemtools.combine_molpdb(ligfile, profile, 
                                             outfile=os.path.join(out_filedir, f"{row[0]}_complex.pdb"))
        return True;
      except: 
        print(f"Failed to process molecule {idx}: {row[0]}")
        return False
  else: 
    print("Not found input file: "); 
    print(profile, os.path.exists(profile)); 
    print(ligfile, os.path.exists(ligfile)); 
    return False; 

def parallelize_traj(traj_list):
  """
  Featurizer and the features initialized within the scope of this function
  """
  traj_list = [i for i in traj_list];
  print(traj_list)
  traj_loader = trajloader.TrajectoryLoader(traj_list, traj_list);
  ret_list = [];
  for trajectory in traj_loader:
    print(f"Processing trajectory {trajectory.top_filename}")
    if trajectory.top.select(":T3P,HOH,WAT").__len__() > 0:
      trajectory.strip(":T3P,HOH,WAT")

    # Initialize the featurizer since different trajectory might have distinct parameters
    feat = features.Featurizer3D(FEATURIZER_PARMS);
    # NOTE: in this step, the feature hooks back the feature and could access the featurizer by feat.featurer
    feat.register_feature(features.BoxFeature());
    feat.register_feature(features.PenaltyFeature("(:LIG)&(!@H=)", "(:LIG<:5)&(!@H=)&(!:LIG)", ref=0));
    # feat.register_feature(features.TopFileNameFeature());

    feat.register_feature(features.RFFeature1D(":LIG"));
    feat.register_feature(FeatureLabel(PDBBind_datafile));

    feat.register_traj(trajectory)
    # Fit the standardizer of the input features
    feat.register_frames(range(1))
    index_selected = trajectory.top.select(":LIG")
    print(f"The number of atoms selected is {len(index_selected)}, " +
          f"Total generated molecule block is {feat.FRAMENUMBER * len(index_selected)}")
    repr_traji, features_traji = feat.run_by_atom(index_selected, focus_mode="cog")
    ret_list.append(features_traji);
    if len(ret_list) % 25 == 0:
      printit(f"Processed {len(ret_list)}/({len(traj_list)}) trajectories; Time elapsed: {time.perf_counter() - st} seconds");
      savelog(f"/tmp/runtime_{os.getpid()}.log")
  return ret_list;


if __name__ == '__main__':
  st = time.perf_counter();

  ref_filedir = "/home/yzhang/Documents/Personal_documents/KDeep/dataset/refined-set-2016/";
  out_filedir = "/media/yzhang/MieT5/BetaPose/data/complexes/";
  PDBBind_datafile = "/media/yzhang/MieT5/KDeep/squeezenet/PDBbind_refined16.txt";
  output_hdffile = "/media/yzhang/MieT5/BetaPose/data/trainingdata/pdbbindrefined_v2016_randomforest.h5";

  table = pd.read_csv(PDBBind_datafile, delimiter="\t", header=None)
  PDBNUMBER = len(table)

  with Client(processes=True, n_workers=16, threads_per_worker=2) as client:
    tasks = [dask.delayed(combine_complex)(idx, row) for idx, row in table.iterrows() if not os.path.exists(os.path.join(out_filedir, f"{row[0]}_complex.pdb"))]
    futures = client.compute(tasks);
    results = client.gather(futures);

  printit(f"Complex combination finished. Used {time.perf_counter() - st:.2f} seconds.")
  printit(f"Success: {np.sum(results)}, Failed: {len(results) - np.sum(results)}");

  # Serial check the existence of the output complex files
  found_PDB = [];
  for idx, row in table.iterrows():
    filename = os.path.join(out_filedir, f"{row[0]}_complex.pdb")
    if os.path.exists(filename):
      found_PDB.append(filename)
    else:
      printit(f"Complex file not found: {filename}")

  if PDBNUMBER != len(found_PDB):
    printit(f"Found {len(found_PDB)} complexes, {PDBNUMBER - len(found_PDB)} complexes are missing")
    exit(0)
  else:
    printit(f"All complexes found; There are {len(found_PDB)} complexes in total")

  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [48, 48, 48],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH": [24, 24, 24],     # Unit: Angstorm (Need scaling)
  }

  # TODO;  for debug only select subset of complexes
  # found_PDB = found_PDB[:100];
  worker_num = 32;
  thread_per_worker = 1;

  split_groups = np.array_split(found_PDB, worker_num);
  with Client(processes=True, n_workers=worker_num, threads_per_worker=thread_per_worker) as client:
    tasks = [dask.delayed(parallelize_traj)(traj_list) for traj_list in split_groups];
    printit("##################Tasks are generated##################")
    futures = client.compute(tasks);
    results = client.gather(futures);

  printit("Tasks are finished, Collecting data")
  box_array = utils.data_from_tbagresults(results, 0);
  penalty_array = utils.data_from_tbagresults(results, 1);
  rf_array = utils.data_from_tbagresults(results, 2);
  label_array = utils.data_from_tbagresults(results, 3);

  with data_io.hdf_operator(output_hdffile, overwrite=True) as h5file:
    h5file.create_dataset("box", box_array)
    h5file.create_dataset("penalty", penalty_array)
    h5file.create_dataset("rf", rf_array)
    h5file.create_dataset("label", label_array)
    h5file.draw_structure()
  printit("##################Data are collected################")





