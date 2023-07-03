import os, time

import numpy as np
import pytraj as pt
import pandas as pd

import dask 
from dask.distributed import Client

from BetaPose import chemtools

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


if __name__ == '__main__':
  st = time.perf_counter();
  table = pd.read_csv("/home/yzhang/Documents/Personal_documents/KDeep/squeezenet/PDBbind_refined16.txt",
                      delimiter="\t",
                      header=None)
  ref_filedir = "/home/yzhang/Documents/Personal_documents/KDeep/dataset/refined-set-2016/"
  out_filedir = "/media/yzhang/MieT5/BetaPose/data/complexes/"

  with Client(processes=True, n_workers=16, threads_per_worker=2) as client:
    tasks = [dask.delayed(combine_complex)(idx, row) for idx, row in table.iterrows() if not os.path.exists(os.path.join(out_filedir, f"{row[0]}_complex.pdb"))]
    futures = client.compute(tasks);
    results = client.gather(futures);

  print(f"Complex combination finished. Used {time.perf_counter() - st:.2f} seconds.")
  print(f"Success: {np.sum(results)}, Failed: {len(results) - np.sum(results)}");

  # Serial check the existence of the output complex files
  for idx, row in table.iterrows():
    filename = f"/media/yzhang/MieT5/BetaPose/data/complexes/{row[0]}_complex.pdb"
    if not os.path.exists(filename):
      print(f"Complex file not found: {filename}")


  # Use Smiles to correct the corruped mol2 file
  # correctedmol = chemtools.CorrectMol2BySmiles(
  #   "/home/yzhang/Documents/Personal_documents/KDeep/dataset/refined-set-2016/1ksn/test.pdb",
  #   "COC(=O)[C@H](Cc1cccc(c1)C(N)=N)[C@@H](C)NC(=O)c1ccc(cc1)-c1cc[n+]([O-])cc1"
  # )
  # print("Corrected Molecule ===> ", correctedmol)
  #
  # complex_str = chemtools.combine_molpdb(
  #   correctedmol,
  #   "/home/yzhang/Documents/Personal_documents/KDeep/dataset/refined-set-2016/1ksn/1ksn_protein.pdb",
  #   outfile="/tmp/test.pdb"
  # )

