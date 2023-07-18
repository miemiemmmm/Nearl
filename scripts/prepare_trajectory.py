import subprocess, os, sys, time, h5py

import numpy as np
import dask
from dask.distributed import Client

from BetaPose import representations, trajloader, features, data_io


class FeatureLabel(features.Feature):
  def __init__(self, affinity_file, delimiter="\t", header=None):
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
    panelty1 = 1
    panelty2 = 1
    return baseline_affinity*panelty1*panelty2;

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
    feat.register_feature(features.PaneltyFeature("(:LIG)&(!@H=)", "(:LIG<:5)&(!@H=)&(!:LIG)", ref=0));
    # feat.register_feature(features.TopFileNameFeature());

    feat.register_feature(features.RFFeature1D(":LIG"));
    feat.register_feature(FeatureLabel("/media/yzhang/MieT5/KDeep/squeezenet/PDBbind_refined16.txt"));

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
  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [48, 48, 48],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH":    [24, 24, 24],  # Unit: Angstorm (Need scaling)
  }
  # Remove the output file if it exists
  if os.path.exists(outputfile):
    os.remove(outputfile)

  # Load multiple trajectories
  # trajectories = sys.argv[1]
  # topologies = sys.argv[2]

  trajectories = "/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_001_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_002_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_003_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_004_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_005_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_006_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_007_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_009_traj.nc%/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_010_traj.nc%"
  trajectories = trajectories.strip("%").split("%");

  topologies = ["/media/yzhang/MieT5/BetaPose_trajs/C209CsDJQucZ_job_008_END.pdb"] * 10;


  traj_loader = trajloader.TrajectoryLoader(trajectories, topologies);
  # Dask parallelization with 16 workers and 2 threads per worker;
  # Top level parallelization: parallelize over trajectories;
  with Client(processes=True, n_workers=16, threads_per_worker=2) as client:
    tasks = [dask.delayed(parallelize_traj)(traj) for traj in traj_loader];
    print("##################Tasks are generated##################");
    futures = client.compute(tasks);
    results = client.gather(futures);

  # Convert the results to numpy array
  box_array = np.array([[j[0] for j in i] for i in results]);
  panelty_array = np.array([[j[1] for j in i] for i in results]);
  name_array = np.array([[j[2] for j in i] for i in results]);
  RF_array = np.array([[j[3] for j in i] for i in results]);

  print("Tasks finished, start saving the data", name_array)
  name_array = np.array(name_array, dtype=h5py.string_dtype('utf-8'))

  # Save the data
  with data_io.hdf_operator(outputfile) as f_write:
    f_write.create_dataset("box", box_array);
    f_write.create_dataset("topo_name", name_array);
    f_write.create_dataset("panelty", panelty_array);
    f_write.create_dataset("RF", RF_array);
    f_write.draw_structure()

  # Check the data
  with data_io.hdf_operator(outputfile) as hdfile:
    hdfile.draw_structure()
    print(hdfile.data("box").shape)
    print(hdfile.data("topo_name").shape)
    print(hdfile.data("panelty").shape)
    print(hdfile.data("RF").shape)

