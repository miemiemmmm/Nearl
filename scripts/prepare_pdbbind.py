import os, time
import h5py as h5
from itertools import chain
import numpy as np
import pytraj as pt
import pandas as pd

import dask 
from dask.distributed import Client, performance_report, LocalCluster

# from BetaPose import chemtools,
import nearl
from nearl import features, utils, printit, savelog, _tempfolder

class FeatureLabel(features.Feature):
  def __init__(self, affinity_file, delimiter=",", header=0):
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
    self.TABLE = pd.read_csv(affinity_file, delimiter=delimiter, header=header);
    self.PDBCODES = self.TABLE["pdbcode"].tolist();
    self.PKS = self.TABLE["pK"].tolist();
    self.FAIL_FLAG = False;

  def featurize(self):
    topfile = self.traj.top_filename;
    topfile = os.path.basename(topfile)
    pdbcode = topfile[:4];
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
    # print("#############Penalty 1#############", baseline_affinity, type(baseline_affinity));
    return baseline_affinity;

def combine_complex(idx, row, ref_filedir):
  ligfile = os.path.join(ref_filedir, f"{row[0]}/{row[0]}_ligand.mol2")
  profile = os.path.join(ref_filedir, f"{row[0]}/{row[0]}_protein.pdb")
  if False not in [profile, ligfile]: 
    print(f"Processing Molecule {idx}: {row[0]}")
    try: 
      complex_str = utils.combine_molpdb(ligfile, profile,
                                           outfile=os.path.join(out_filedir, f"{row[0]}_complex.pdb"))
      return True;
    except: 
      try: 
        ligfile = os.path.join(ref_filedir, f"{row[0]}/{row[0]}_ligand.sdf")
        complex_str = utils.combine_molpdb(ligfile, profile,
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
  traj_loader = nearl.io.TrajectoryLoader(traj_list, traj_list);
  ret_list = [];
  st = time.perf_counter();

  for trajidx, trajectory in enumerate(traj_loader):
    print(f"Processing trajectory {trajectory.top_filename}, Last structure took {time.perf_counter()-st} seconds");
    st = time.perf_counter();
    if trajectory.top.select(":T3P,HOH,WAT").__len__() > 0:
      trajectory.strip(":T3P,HOH,WAT")
    trajectory.top.set_reference(trajectory[0]);
    mask = ":LIG"

    # Initialize the featurizer since different trajectory might have distinct parameters
    feat = features.Featurizer3D(FEATURIZER_PARMS);
    # NOTE: in this step, the feature hooks back the feature and could access the featurizer by feat.featurer
    # Following are the 1D RandomForest features
    feat.register_feature(features.BoxFeature());
    # feat.register_feature(features.PenaltyFeature(f"({mask})&(!@H=)", f"({mask}<:5)&(!@H=)&(!{mask})", ref=0));
    # feat.register_feature(features.RFFeature1D(mask));
    feat.register_feature(FeatureLabel(PDBBind_datafile));
    feat.register_feature(features.TopologySource());

    mask1 = ":LIG<:10&(!@H=)";
    mask2 = ":LIG&(!@H=)";

    feat.register_feature(features.XYZCoord(mask=mask1));
    feat.register_feature(features.XYZCoord(mask=mask2));

    # Following are the 3D based features
    feat.register_feature(features.Mass(mask=mask1));
    feat.register_feature(features.Mass(mask=mask2));

    feat.register_feature(features.EntropyResidueID(mask=mask1));
    feat.register_feature(features.EntropyResidueID(mask=mask2));

    feat.register_feature(features.EntropyAtomID(mask=mask1));
    feat.register_feature(features.EntropyAtomID(mask=mask2));

    feat.register_feature(features.Aromaticity(mask=mask1));
    feat.register_feature(features.Aromaticity(mask=mask2));

    feat.register_feature(features.PartialCharge(mask=mask1));
    feat.register_feature(features.PartialCharge(mask=mask2));

    feat.register_feature(features.HeavyAtom(mask=mask1));
    feat.register_feature(features.HeavyAtom(mask=mask2));

    feat.register_feature(features.HeavyAtom(mask=mask1, reverse=True));
    feat.register_feature(features.HeavyAtom(mask=mask2, reverse=True));

    feat.register_feature(features.Ring(mask=mask1));
    feat.register_feature(features.Ring(mask=mask2));

    feat.register_feature(features.HydrogenBond(mask=mask1, donor=True));
    feat.register_feature(features.HydrogenBond(mask=mask2, donor=True));

    feat.register_feature(features.HydrogenBond(mask=mask1, acceptor=True));
    feat.register_feature(features.HydrogenBond(mask=mask2, acceptor=True));

    feat.register_feature(features.Hybridization(mask=mask1));
    feat.register_feature(features.Hybridization(mask=mask2));

    # feat.register_feature(features.MassFeature(mask=":T3P"));
    # feat.register_feature(features.EntropyResidueIDFeature(mask=":T3P"));
    # feat.register_feature(features.EntropyAtomIDFeature(mask=":T3P"));

    # feat.register_feature(features.TopFileFeature());
    # feat.register_feature(FeatureLabel(PDBBind_datafile));

    feat.register_traj(trajectory)
    # Fit the standardizer of the input features
    feat.register_frames(range(1))
    index_selected = trajectory.top.select(":LIG")
    print(f"The number of atoms selected is {len(index_selected)}, " +
          f"Total generated molecule block is {feat.FRAMENUMBER * len(index_selected)}")
    repr_traji, features_traji = feat.run_by_atom(index_selected, focus_mode="cog")
    ret_list.append(features_traji);

    if ((trajidx+1) % 25 == 0):
      printit(f"Processed {len(ret_list)}/({len(traj_list)}) trajectories; Time elapsed: {time.perf_counter() - st} seconds");

    if ((trajidx+1) % 10 == 0 or trajidx == len(traj_list)-1):
      tempfilename =os.path.join(_tempfolder, f"temp{os.getpid()}.npy");
      if os.path.exists(tempfilename):
        prev_data = np.load(tempfilename, allow_pickle=True);
        # Convert the results to numpy array
        new_data = np.array(ret_list, dtype=object);
        new_data = np.concatenate([prev_data, new_data], axis=0)
        nearl.io.temporary_dump(new_data, tempfilename);
        ret_list = [];
      else:
        nearl.io.temporary_dump(ret_list, tempfilename);
        ret_list = [];
      repr_traji = None;
      features_traji = None;
  return str(tempfilename);


if __name__ == '__main__':
  st = time.perf_counter();
  #################################################################################
  ########### Part1: Combine protein-ligand to complex PDB file ###################
  #################################################################################
  ref_filedir1 = "/media/yzhang/MieT5/PDBbind_v2020_refined/";
  ref_filedir2 = "/media/yzhang/MieT5/PDBbind_v2020_other_PL/";
  out_filedir = "/media/yzhang/MieT5/BetaPose/data/complexes/";           # Output directory for the combined complex PDB file
  PDBBind_datafile = "/media/yzhang/MieT5/BetaPose/data/PDBBind_general_v2020.csv";   # PDBBind V2020
  SKIP_COMBINE = True

  # Read the PDBBind dataset
  table = pd.read_csv(PDBBind_datafile, delimiter=",", header=0);
  PDBNUMBER = len(table);

  if SKIP_COMBINE != True:
    refdirs = []
    for pdbcode in table.pdbcode.tolist():
      if os.path.exists(os.path.join(ref_filedir1, pdbcode)):
        refdirs.append(ref_filedir1);
      elif os.path.exists(os.path.join(ref_filedir2, pdbcode)):
        refdirs.append(ref_filedir2);
      else:
        print(f"Cannot find the reference directory for {pdbcode}");
        exit(1);

    if len(refdirs) == len(table.pdbcode.tolist()):
      print("All reference directories are found.")


    with Client(processes=True, n_workers=24, threads_per_worker=1) as client:
      tasks = [dask.delayed(combine_complex)(idx, row, refdir) for (idx, row),refdir in zip(table.iterrows(), refdirs) if not os.path.exists(os.path.join(out_filedir, f"{row[0]}_complex.pdb"))]
      futures = client.compute(tasks);
      results = client.gather(futures);

    printit(f"Complex combination finished. Used {time.perf_counter() - st:.2f} seconds.")
    printit(f"Success: {np.sum(results)}, Failed: {len(results) - np.sum(results)}");

  #################################################################################
  ########### Part2: Check the existence of required PDB complexes ################
  #################################################################################
  pdb_listfile = "/media/yzhang/MieT5/BetaPose/data/misato_original_index/train_MD.txt"
  output_hdffile = "/media/yzhang/MieT5/BetaPose/data/trainingdata/test_3d_data.h5";
  complex_dir = "/media/yzhang/MieT5/BetaPose/data/complexes/";

  SUPERSEDES = {
    "4dgo": "6qs5", "4otw": "6op9", "4v1c": "6iso",
    "5v8h": "6nfg", "5v8j": "6nfo", "6fim": "6fex",
    "6h7k": "6ibl", "3m8t": "5wcm", "4n3l": "6eo8",
    "4knz": "6nnr",
  }

  with open(pdb_listfile, "r") as f1:
    pdb_to_featurize = f1.read().strip("\n").split("\n");
    pdb_to_featurize = [SUPERSEDES[i.lower()] if i.lower() in SUPERSEDES else i for i in pdb_to_featurize];

  # Serial check the existence of the output complex files
  complex_files = [os.path.join(complex_dir, f"{pdbcode.lower()}_complex.pdb") for pdbcode in pdb_to_featurize];
  found_state = [os.path.exists(pdbfile) for pdbfile in complex_files];

  if np.count_nonzero(found_state) == len(found_state):
    print(f"Congretulations: All complexes found ({np.count_nonzero(found_state)}/{len(found_state)})");
  else:
    print(f"Error: {len(found_state) - np.count_nonzero(found_state)}/{len(found_state)} complexes are not found in the complex directory")
    print(np.array(pdb_to_featurize)[np.where(np.array(found_state) == False)[0]].tolist());
    exit(0);

  #################################################################################
  ########### Part3: Featurize the required PDB complexes #########################
  #################################################################################
  st_compute = time.perf_counter();
  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [48, 48, 48],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH": [24, 24, 24],     # Unit: Angstorm (Need scaling)
  }

  do_test = True;
  if do_test:
    found_PDB = complex_files[:50];
    result1 = parallelize_traj(found_PDB);
    results = [result1];
  else:
    # TODO: Change these settings before running for production
    worker_num = 2;
    thread_per_worker = 1;
    found_PDB = complex_files[:80];
    split_groups = np.array_split(found_PDB, worker_num);
    cluster = LocalCluster(
      n_workers=worker_num,
      threads_per_worker=thread_per_worker,
      memory_limit='10GB',
    );
    with Client(cluster) as client:
      with performance_report(filename="dask-report.html"):
        tasks = [dask.delayed(parallelize_traj)(traj_list) for traj_list in split_groups];
        printit("##################Tasks are generated##################")
        futures = client.compute(tasks);
        results = client.gather(futures);

  printit(f"Tasks are finished, Collecting data... Pure computation time: {time.perf_counter() - st_compute:.2f} seconds.")

  # Process the results from the computation
  os.remove(output_hdffile) if os.path.exists(output_hdffile) else None;

  keydict = {
    0: "box", 1: "label", 2: "pdbcode",
    3: "xyz_pro", 4: "xyz_lig",
    5: "mass_pro", 6: "mass_lig", 7: "entres_pro", 8: "entres_lig",
    9: "entatm_pro", 10: "entatm_lig", 11: "arom_pro", 12: "arom_lig",
    13: "pc_pro", 14: "pc_lig", 15: "ha_pro", 16: "ha_lig",
    17: "nha_pro", 18: "nha_lig", 19: "r_pro", 20: "r_lig",
    21: "donor_pro", 22: "donor_lig", 23: "acceptor_pro", 24: "acceptor_lig",
    25: "hyb_pro", 26: "hyb_lig",
  };

  for idx, key in keydict.items():
    for tmpfile in results:
      dataobj = np.load(tmpfile, allow_pickle=True);
      print(f"Loaded {tmpfile}, shape of it, {dataobj.shape}");
      thisdataset = utils.data_from_fbagresults(dataobj, idx);
      if tmpfile == results[0]:
        datai = thisdataset;
      else:
        datai = np.concatenate((datai, thisdataset), axis=0);
      if key in ["pdbcode"]:
        datai = np.array([s.encode('utf8') for s in datai.ravel()]).reshape((-1, 1));

    if key in ["xyz_pro", "xyz_lig"]:
      # The dimensions of coordinate array (X*3) is not aligned, Hence save separately
      datai = np.array(datai, dtype=object);
      # tmpfile = os.path.join(_tempfolder, f"dataset_{key}.npy");
      # np.save(tmpfile, datai);
      with nearl.io.hdf_operator(output_hdffile, append=True) as h5file:
        h5file.create_heterogeneous(key, datai)
    else:
      print(f"Shape of datai: {np.array(datai).shape}");
      with nearl.io.hdf_operator(output_hdffile, append=True) as h5file:
        h5file.dump_dataset(key, datai);

  # Finally show the structure of the output hdf5 file
  with nearl.io.hdf_operator(output_hdffile, append=True) as h5file:
    h5file.draw_structure();

  printit(f"##################Data are collected {time.perf_counter()-st:.3f} ################")
# 239.7 seconds for 80 complexes: 20 workers, OMP_NUM_THREADS=20
# 257.9 seconds for 80 complexes: 20 workers, OMP_NUM_THREADS=12
# 284.0 seconds for 80 complexes: 20 workers, OMP_NUM_THREADS=8
# 286.4 seconds for 80 complexes: 20 workers, OMP_NUM_THREADS=6
# 251.3 seconds for 80 complexes: 20 workers, OMP_NUM_THREADS=4
# 262.1 seconds for 80 complexes: 20 workers, OMP_NUM_THREADS=2
# 429.0 seconds for 80 complexes: 20 workers, OMP_NUM_THREADS=1




