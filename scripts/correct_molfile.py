import os, time
from BetaPose import chemtools

if __name__ == '__main__':
  """
  This script is used to correct the corrupted mol file (mol2/sdf/pdb) and combine the ligand and protein into a complex file.
  """
  st = time.perf_counter();
  ligandfile = "/home/yzhang/Documents/Personal_documents/KDeep/dataset/refined-set-2016/1ksn/test.pdb"
  proteinfile = "/home/yzhang/Documents/Personal_documents/KDeep/dataset/refined-set-2016/1ksn/1ksn_protein.pdb"
  smiles_str = "COC(=O)[C@H](Cc1cccc(c1)C(N)=N)[C@@H](C)NC(=O)c1ccc(cc1)-c1cc[n+]([O-])cc1"
  output_complex_file = "/media/yzhang/MieT5/BetaPose/data/complexes/1ksn_complex.pdb"

  # Use Smiles to correct the corruped mol2 file
  correctedmol = chemtools.CorrectMolBySmiles(ligandfile, smiles_str)

  print("Corrected Molecule ===> ", correctedmol)
  if correctedmol is not None:
    complex_str = chemtools.combine_molpdb(correctedmol, proteinfile, outfile=output_complex_file)
    print(f"Complex combination finished. Used {time.perf_counter() - st:.2f} seconds.")
  else:
    print("Failed to correct the molecule.")
    exit(1)

