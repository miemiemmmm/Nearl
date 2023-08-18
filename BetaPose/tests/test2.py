import os, time

import numpy as np
import pytraj as pt
import pandas as pd

import dask
from dask.distributed import Client

from BetaPose import chemtools, trajloader, features, data_io

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
    feat.register_feature(features.PaneltyFeature("(:LIG)&(!@H=)", "(:LIG<:5)&(!@H=)&(!:LIG)", ref=0));
    feat.register_feature(features.TopFileNameFeature());
    feat.register_feature(features.RFFeature1D(":LIG"));


    feat.register_traj(trajectory)
    # Fit the standardizer of the input features
    feat.register_frames(range(1))
    index_selected = trajectory.top.select(":LIG")
    print(f"The number of atoms selected is {len(index_selected)}, " +
          f"Total generated molecule block is {feat.FRAMENUMBER * len(index_selected)}")
    repr_traji, features_traji = feat.run_by_atom(index_selected, focus_mode="cog")
    ret_list.append([features_traji]);
  return features_traji;


if __name__ == '__main__':
  st = time.perf_counter();
  table = pd.read_csv("/media/yzhang/MieT5/KDeep/squeezenet/PDBbind_refined16.txt",
                      delimiter="\t",
                      header=None)
  PDBNUMBER = len(table)
  ref_filedir = "/home/yzhang/Documents/Personal_documents/KDeep/dataset/refined-set-2016/"
  out_filedir = "/media/yzhang/MieT5/BetaPose/data/complexes/"

#   with Client(processes=True, n_workers=16, threads_per_worker=2) as client:
#     tasks = [dask.delayed(combine_complex)(idx, row) for idx, row in table.iterrows() if not os.path.exists(os.path.join(out_filedir, f"{row[0]}_complex.pdb"))]
#     futures = client.compute(tasks);
#     results = client.gather(futures);
#
#   print(f"Complex combination finished. Used {time.perf_counter() - st:.2f} seconds.")
#   print(f"Success: {np.sum(results)}, Failed: {len(results) - np.sum(results)}");

  # Serial check the existence of the output complex files
  found_PDB = [];
  for idx, row in table.iterrows():
    filename = os.path.join(out_filedir, f"{row[0]}_complex.pdb")
    if os.path.exists(filename):
      found_PDB.append(filename)
    else:
      print(f"Complex file not found: {filename}")

  print(f"Found {len(found_PDB)} complexes")
  if PDBNUMBER == len(found_PDB):
    print("All complexes found")
    FEATURIZER_PARMS = {
      # POCKET SETTINGS
      "CUBOID_DIMENSION": [48, 48, 48],  # Unit: 1 (Number of lattice in one dimension)
      "CUBOID_LENGTH": [24, 24, 24],  # Unit: Angstorm (Need scaling)
    }

    split_groups = np.array_split(found_PDB, 16);

    lst = found_PDB[3831:3835]
    print(lst)
    parallelize_traj(lst);
#     with Client(processes=True, n_workers=32, threads_per_worker=1) as client:
#     # with Client(processes=True) as client:
#       tasks = [dask.delayed(parallelize_traj)(traj_list) for traj_list in split_groups];
#       print("##################Tasks are generated##################")
#       futures = client.compute(tasks);
#       results = client.gather(futures);
#     box_array = np.array([[j[0] for j in i] for i in results]);
#     print(box_array.shape)

  else:
    print("Some complexes not found: ")
    print(f"Found {len(found_PDB)} complexes / Expected {PDBNUMBER} complexes")




