import time, json, os
import numpy as np
import pytraj as pt
import pandas as pd
from BetaPose import utils, trajloader, features, printit, savelog
from BetaPose import data_io

import dask
from dask.distributed import Client

SUPERSEDES = {
  "4dgo":"6qs5",
  "4otw":"6op9",
  "4v1c":"6iso",
  "5v8h":"6nfg",
  "5v8j":"6nfo",
  "6fim":"6fex",
  "6h7k":"6ibl",
  "3m8t":"5wcm",
  "4n3l":"6eo8",
  "4knz":"6nnr",
}

class FeatureLabel(features.Feature):
  def __init__(self, affinity_file, atom_of_interest, delimiter=",", header=0):
    super().__init__();
    self.ATOMOFINTEREST = atom_of_interest
    self.TABLE = pd.read_csv(affinity_file, delimiter=delimiter, header=header);
    self.PDBCODES = self.TABLE["pdbcode"].tolist();
    self.PKS = self.TABLE["pK"].tolist();
    self.FAIL_FLAG = False;

  def before(self):
    """
    Before is executed before every loop of centers, Meaning for different frame, it is run again
    """
    if isinstance(self.ATOMOFINTEREST, str):
      atom_select = self.traj.top.select(self.ATOMOFINTEREST);
    elif isinstance(self.ATOMOFINTEREST, (list, tuple, np.ndarray)):
      atom_select = np.array([int(i) for i in self.ATOMOFINTEREST]);
    self.traj.top.set_reference(self.traj[0]);
    reformed_mask = "@"+ ",".join([str(i + 1) for i in atom_select]);
    atom_counterpart = self.traj.top.select(f"(({reformed_mask})<:8)&(!@H=)&!({reformed_mask})");
    self.pdist, self.pdistinfo = utils.PairwiseDistance(self.traj, atom_select, atom_counterpart);
    if self.pdist is None or self.pdistinfo is None:
      self.FAIL_FLAG = True;
      return

    self.pdist_mean = self.pdist.mean(axis=1)
    if self.pdist.mean() > 8:
      printit("Warning: the mean distance between the atom of interest and its counterpart is larger than 8 Angstrom");
      printit("Please check the atom selection");
    elif np.percentile(self.pdist, 85) > 12:
      printit("Warning: the 85th percentile of the distance between the atom of interest and its counterpart is larger than 12 Angstrom");
      printit("Please check the atom selection");

    info_lengths = [len(self.pdistinfo[key]) for key in self.pdistinfo];
    if len(set(info_lengths)) != 1:
      printit("Warning: The length of the pdistinfo is not consistent", self.pdistinfo);

  def featurize(self):
    if self.FAIL_FLAG == True:
      return -1;
    pdbcode = self.traj.top_filename;
    pdbcode = pdbcode.lower();
    if pdbcode in self.PDBCODES:
      idx = self.PDBCODES.index(pdbcode);
      baseline_affinity = self.PKS[idx];
    elif pdbcode in SUPERSEDES:
      pdbcode = SUPERSEDES[pdbcode.lower()];
      idx = self.PDBCODES.index(pdbcode);
      baseline_affinity = self.PKS[idx];
    else:
      print(f"PDB code {pdbcode} not found in the affinity table");
      raise ValueError("PDB code not found");

    dists = np.zeros(self.pdist.shape[0], dtype=np.float64);
    c = 0;
    for i,j in zip(self.pdistinfo["indices_group1"], self.pdistinfo["indices_group2"]):
      disti = self.active_frame.xyz[i] - self.active_frame.xyz[j];
      disti = np.linalg.norm(disti);
      dists[c] = disti;
      c += 1;
    printit("Active frame index: ", self.active_frame_index)
    printit(dists, self.pdist[:,self.active_frame_index])

    # Panelty 1: the cosine similarity between the framei distance and the mean pairwise distance
    panelty1 = utils.cosine_similarity(dists, self.pdist_mean);
    # Panelty 2: the cosine similarity between the framei distance and the distance of the first frame
    panelty2 = utils.cosine_similarity(dists, self.pdist[:,0]);

    penalized = baseline_affinity * panelty1 * panelty2;
    print(f"Labeling the frame by score: {penalized}/{baseline_affinity}; Penalty 1:{panelty1}; Penalty 2:{panelty2};")
    return penalized

def parallelize_traj(traj_list):
  """
  Featurizer and the features initialized within the scope of this function
  """
  traj_list = [i for i in traj_list];
  print(traj_list)
  traj_loader = trajloader.TrajectoryLoader(traj_list, traj_list);
  traj_loader.set_outtype(misato_traj);   # Set the output type to the misato_traj

  ret_list = [];
  for trajectory in traj_loader:
    if trajectory.top.select(":T3P,HOH,WAT").__len__() > 0:
      trajectory.strip(":T3P,HOH,WAT")

    # Select the component of interest
    index_selected = np.array(ligand_indices_map[trajectory.top_filename.upper()])
    mask = "@" + ",".join([str(i+1) for i in index_selected])

    printit(f"Processing trajectory {trajectory.top_filename} | Selected atom number: {len(index_selected)}")

    # Initialize the featurizer since different trajectory might have distinct parameters
    feat = features.Featurizer3D(FEATURIZER_PARMS);
    # NOTE: in this step, the feature hooks back the feature and could access the featurizer by feat.featurer
    feat.register_feature(features.BoxFeature());
    feat.register_feature(features.PenaltyFeature(f"({mask})&(!@H=)", f"({mask}<:5)&(!@H=)&(!{mask})", ref=0));

    feat.register_feature(features.RFFeature1D(mask));
    feat.register_feature(FeatureLabel(pdbbind_csv, f"({mask})&(!@H=)"));

    feat.register_traj(trajectory)
    # NOTE: Register frame's starting index is 0
    feat.register_frames(range(0, 100, 10))
    print("In the featurizer function: ", trajectory.top_filename)

    # index_selected = trajectory.top.select(":LIG,MOL")
    print(f"The number of atoms selected is {len(index_selected)}, " +
          f"Total generated molecule block is {feat.FRAMENUMBER * len(index_selected)}")
    repr_traji, features_traji = feat.run_by_atom(index_selected, focus_mode="cog");
    ret_list.append(features_traji);
    if len(ret_list) % 25 == 0:
      printit(f"Processed {len(ret_list)}/({len(traj_list)}) trajectories; Time elapsed: {time.perf_counter() - st_total} seconds");
      savelog(f"/tmp/runtime_{os.getpid()}.log")
  return ret_list;

# The misato trajectory type, inherit from the trajloader.Trajectory class
# Trajectory class is primarily used for trajectory reading while featurization
# active-frame processing and meta-information retrieval
class misato_traj(trajloader.Trajectory):
  def __init__(self, pdbcode, _, *args, **kwarg):
    thetraj = utils.misato_traj(pdbcode, misato_md_hdf, misato_parm_dir, *args, **kwarg)
    super().__init__(thetraj, pdbcode);
    self.traj_filename = pdbcode;
    self.top_filename  = pdbcode;


if __name__ == "__main__":
  # Define the output file before running the script
  output_hdffile = "/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_randomforest_step10.h5"
  worker_num = 32;
  thread_per_worker = 1;


  st_total = time.perf_counter();
  # Check the PDB bind affinity table
  pdbbind_csv = "/media/yzhang/MieT5/BetaPose/data/PDBBind_general_v2020.csv";
  # Misato MD trajectory and QM reference
  misato_md_hdf = "/home/yzhang/Downloads/misato_database/MD.hdf5";
  misato_parm_dir = "/home/yzhang/Downloads/misato_database/parameter_restart_files_MD";

  # Misato MD trajectory index to featurize
  misato_md_index = "/media/yzhang/MieT5/BetaPose/data/misato_train_with_ligand.txt"
  with open(misato_md_index, "r") as f:
    trajlist = f.read().strip("\n").split("\n");
  print("The number of trajectories to be featurized is ", len(trajlist));

  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [48, 48, 48],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH": [24, 24, 24],  # Unit: Angstorm (Need scaling)
  }

  with open("/media/yzhang/MieT5/BetaPose/data/misato_ligand_indices.json", "r") as f:
    ligand_indices_map = json.load(f)

  production = True;
  if production == True:
    split_groups = np.array_split(trajlist, worker_num);
    with Client(processes=True, n_workers=worker_num, threads_per_worker=thread_per_worker) as client:
      tasks = [dask.delayed(parallelize_traj)(traj_list) for traj_list in split_groups];
      printit("##################Tasks are generated##################")
      futures = client.compute(tasks);
      results = client.gather(futures);
  else:
    # TODO This is the test code: to be removed
    trajlist = trajlist[:20]
    result1 = parallelize_traj(trajlist);
    results = [result1];

  printit("##################Tasks are finished################")
  box_array = utils.data_from_tbagresults(results, 0);
  penalty_array = utils.data_from_tbagresults(results, 1)
  rf_array = utils.data_from_tbagresults(results, 2)
  label_array = utils.data_from_tbagresults(results, 3)

  print(box_array.shape, penalty_array.shape, rf_array.shape, label_array.shape)

  with data_io.hdf_operator(output_hdffile, overwrite=True) as h5file:
    h5file.create_dataset("box", box_array)
    h5file.create_dataset("penalty", penalty_array)
    h5file.create_dataset("rf", rf_array)
    h5file.create_dataset("label", label_array)
    h5file.draw_structure()
  printit("##################Data are collected################")

  savelog("prodrun.log")


# loader = trajloader.TrajectoryLoader(trajlist, trajlist);
# loader.set_outtype(misato_traj);


