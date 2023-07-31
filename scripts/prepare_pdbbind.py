import os, time
import h5py as h5
from itertools import chain

import numpy as np
import pytraj as pt
import pandas as pd

import dask 
from dask.distributed import Client

from BetaPose import chemtools, trajloader, features, data_io, utils, printit, savelog

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
  st = time.perf_counter();
  for trajectory in traj_loader:
    print(f"Processing trajectory {trajectory.top_filename}, Last structure took {time.perf_counter()-st} seconds");
    st = time.perf_counter();
    if trajectory.top.select(":T3P,HOH,WAT").__len__() > 0:
      trajectory.strip(":T3P,HOH,WAT")
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

    trajectory.top.set_reference(trajectory[0]);
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
  FEATURIZER_PARMS = {
    # POCKET SETTINGS
    "CUBOID_DIMENSION": [32, 32, 32],  # Unit: 1 (Number of lattice in one dimension)
    "CUBOID_LENGTH": [24, 24, 24],     # Unit: Angstorm (Need scaling)
  }

  worker_num = 64;
  thread_per_worker = 1;
  do_test = False;

  if do_test:
    found_PDB = complex_files[:10];
    result1 = parallelize_traj(found_PDB);
    results = [result1];
  else:
    found_PDB = complex_files[:20];
    split_groups = np.array_split(found_PDB, worker_num);
    with Client(processes=True, n_workers=worker_num, threads_per_worker=thread_per_worker) as client:
      tasks = [dask.delayed(parallelize_traj)(traj_list) for traj_list in split_groups];
      printit("##################Tasks are generated##################")
      futures = client.compute(tasks);
      results = client.gather(futures);

  printit("Tasks are finished, Collecting data")
  box_array = utils.data_from_tbagresults(results, 0);
  label_array = utils.data_from_tbagresults(results, 1);
  pdbcode_array = utils.data_from_tbagresults(results, 2);
  pdbcode_array = np.array([s.encode('utf8') for s in pdbcode_array.ravel()]).reshape((len(pdbcode_array), 1));

  mass_pro = utils.data_from_tbagresults(results, 3);
  mass_lig = utils.data_from_tbagresults(results, 4);
  entres_pro = utils.data_from_tbagresults(results, 5);
  entres_lig = utils.data_from_tbagresults(results, 6);
  entatm_pro = utils.data_from_tbagresults(results, 7);
  entatm_lig = utils.data_from_tbagresults(results, 8);


  arom_pro = utils.data_from_tbagresults(results, 9);
  arom_lig = utils.data_from_tbagresults(results, 10);

  pc_pro = utils.data_from_tbagresults(results, 11);
  pc_lig = utils.data_from_tbagresults(results, 12);

  ha_pro = utils.data_from_tbagresults(results, 13);
  ha_lig = utils.data_from_tbagresults(results, 14);

  nha_pro = utils.data_from_tbagresults(results, 15);
  nha_lig = utils.data_from_tbagresults(results, 16);

  r_pro = utils.data_from_tbagresults(results, 17);
  r_lig = utils.data_from_tbagresults(results, 18);

  donor_pro = utils.data_from_tbagresults(results, 19);
  donor_lig = utils.data_from_tbagresults(results, 20);

  acceptor_pro = utils.data_from_tbagresults(results, 21);
  acceptor_lig = utils.data_from_tbagresults(results, 22);

  hyb_pro = utils.data_from_tbagresults(results, 23);
  hyb_lig = utils.data_from_tbagresults(results, 24);

  with data_io.hdf_operator(output_hdffile, overwrite=True) as h5file:
    h5file.create_dataset("box", box_array)
    h5file.create_dataset("label", label_array)
    h5file.create_dataset("pdbcode", pdbcode_array);

    h5file.create_dataset("mass_pro", mass_pro)
    h5file.create_dataset("mass_lig", mass_lig)
    h5file.create_dataset("entres_pro", entres_pro)
    h5file.create_dataset("entres_lig", entres_lig)
    h5file.create_dataset("entatm_pro", entatm_pro)
    h5file.create_dataset("entatm_lig", entatm_lig)

    h5file.create_dataset("arom_pro", arom_pro)
    h5file.create_dataset("arom_lig", arom_lig)
    h5file.create_dataset("pc_pro", pc_pro)
    h5file.create_dataset("pc_lig", pc_lig)
    h5file.create_dataset("ha_pro", ha_pro)
    h5file.create_dataset("ha_lig", ha_lig)
    h5file.create_dataset("nha_pro", nha_pro)
    h5file.create_dataset("nha_lig", nha_lig)

    h5file.create_dataset("r_pro", r_pro)
    h5file.create_dataset("r_lig", r_lig)
    h5file.create_dataset("donor_pro", donor_pro)
    h5file.create_dataset("donor_lig", donor_lig)
    h5file.create_dataset("acceptor_pro", acceptor_pro)
    h5file.create_dataset("acceptor_lig", acceptor_lig)
    h5file.create_dataset("hyb_pro", hyb_pro)
    h5file.create_dataset("hyb_lig", hyb_lig)

    h5file.draw_structure()
  printit("##################Data are collected################")





