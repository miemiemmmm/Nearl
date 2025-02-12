import subprocess, os, sys, time, h5py

import numpy as np
import dask
from dask.distributed import Client

from Nearl import representations, trajloader, features, data_io
from Nearl import utils, printit, savelog


class FeatureLabel(features.Feature):
  def __init__(self, affinity_file, atom_of_interest, delimiter=",", header=0):
    """
    Use the topology name as the identifier of the molecule since it specific to the PDB ID of the protein
    Pre-generate the affinity table and use the topology name list to index the affinity
    Then apply the panelties to the affinity label
    Input:
      affinity_file: the file containing the affinity table (str, Path to the affinity table)
      delimiter: the delimiter of the affinity table (str, default: "\t")
    Output:
      label: the affinity label of the frame (float, baseline_affinity*panelty1*panelty2)
    """
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
      return -1
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
  printit(traj_list)
  trajfiles = [i[0] for i in traj_pair];
  topfiles = [i[1] for i in traj_pair];
  traj_loader = trajloader.TrajectoryLoader(trajfiles, topfiles);
  ret_list = [];
  for trajectory in traj_loader:
    print(f"Processing trajectory {trajectory.top_filename}")
    if trajectory.top.select(":T3P,HOH,WAT").__len__() > 0:
      trajectory.strip(":T3P,HOH,WAT")
    mask = ":LIG"

    # Initialize the featurizer since different trajectory might have distinct parameters
    feat = features.Featurizer3D(FEATURIZER_PARMS);
    # NOTE: in this step, the feature hooks back the feature and could access the featurizer by feat.featurer
    feat.register_feature(features.BoxFeature());
    feat.register_feature(features.PenaltyFeature(f"({mask})&(!@H=)", f"({mask}<:5)&(!@H=)&(!{mask})", ref=0));

    feat.register_feature(features.RFFeature1D(mask));
    feat.register_feature(FeatureLabel(pdbbind_csv, f"({mask})&(!@H=)"));


    feat.register_traj(trajectory)
    # Fit the standardizer of the input features
    feat.register_frames(range(0, 1000, 200))
    index_selected = trajectory.top.select(":LIG")
    print(f"The number of atoms selected is {len(index_selected)}, " +
          f"Total generated molecule block is {feat.FRAMENUMBER * len(index_selected)}")
    repr_traji, features_traji = feat.run_by_atom(index_selected, focus_mode="cog")
    ret_list.append(features_traji);
  return ret_list;


############## TODO: Write these featurizers in the future ################
# feat.register_feature(features.FPFHFeature());
# feat.register_feature(features.HydrophobicityFeature());
# feat.register_feature(features.AromaticityFeature());
# feat.register_feature(features.PartialChargeFeature());
# feat.register_feature(features.PartialChargeFeature(moi="(:LIG)"));
# feat.register_feature(features.PartialChargeFeature(moi="(:T3P,HOH,WAT)"));
# feat.register_feature(features.PartialChargeFeature(moi="(:1-100)"));

if __name__ == "__main__":
  print("Current working directory: ", os.getcwd());
  outputfile = "../data/data.h5";
  output_hdffile = "/MieT5/Nearl/data/trainingdata/mytraj_test.h5"
  worker_num = 24;
  thread_per_worker = 1;

  st_total = time.perf_counter();
  pdbbind_csv = "/MieT5/Nearl/data/PDBBind_general_v2020.csv";
  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [48, 48, 48],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH":    [24, 24, 24],  # Unit: Angstorm (Need scaling)
  }

  # Load multiple trajectories
  trajectories = "/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_001_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_002_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_003_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_004_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_005_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_006_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_007_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_009_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_010_traj.nc%"
  trajectories = trajectories.strip("%").split("%");
  topologies = ["/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_END.pdb"] * 10;

  # Check the existence of the trajectory before putting the trajectory array into the feature label index
  file_exist = [os.path.exists(traj) for traj in trajectories]
  if np.count_nonzero(file_exist) != len(file_exist):
    printit("Please check the file name template");
    raise IOError("File not found");
  else:
    printit("All trajectories are found");
  file_exist = [os.path.exists(top) for top in topologies]
  if np.count_nonzero(file_exist) != len(file_exist):
    printit("Please check the file name template");
    raise IOError("File not found");
  else:
    printit("All topologies are found");

  traj_loader = trajloader.TrajectoryLoader(trajectories, topologies);
  # Top level parallelization: parallelize over trajectories;
  # For result retrieval, use the trajectory bag level data-extraction
  traj_top_pairs = list(zip(trajectories, topologies));
  production = True;
  if production == True:
    split_groups = np.array_split(traj_top_pairs, worker_num);
    with Client(processes=True, n_workers=worker_num, threads_per_worker=thread_per_worker) as client:
      tasks = [dask.delayed(parallelize_traj)(traj_pairs) for traj_pairs in split_groups];
      print("##################Tasks are generated##################");
      futures = client.compute(tasks);
      results = client.gather(futures);
  else:
    # TODO This is the test code: to be removed
    result1 = parallelize_traj(traj_top_pairs[:5]);
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

