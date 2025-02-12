import os, time
from Nearl import chemtools

if __name__ == '__main__':
  """
  This script is used to correct the corrupted mol file (mol2/sdf/pdb) and combine the ligand and protein into a complex file.
  """
  st = time.perf_counter();
  ligand_info = [
    # "/MieT5/PDBbind_v2020_other_PL/2aoh/2aoh_ligand.sdf",
    # "/MieT5/PDBbind_v2020_other_PL/1a7x/1a7x_ligand.sdf",
    "/tmp/test.pdb",
    "OC1=CC=C(C(Cl)=C1)C1=NC2=C(C=CC=C2)C2=CC=NC3=C2C1=CN3",
    "/MieT5/PDBbind_v2020_other_PL/3kck/3kck_protein.pdb",
    "/MieT5/Nearl/data/complexes/3kck_complex.pdb"
  ]

  ligandfile = ligand_info[0]
  smiles_str = ligand_info[1]
  proteinfile = ligand_info[2]
  output_complex_file = ligand_info[3]
  # output_complex_file = "/MieT5/Nearl/data/complexes/1ksn_complex.pdb"

  # Use Smiles to correct the corruped mol2 file
  correctedmol = chemtools.CorrectMolBySmiles(ligandfile, smiles_str)

  print("Corrected Molecule ===> ", correctedmol)
  if correctedmol is not None:
    complex_str = chemtools.combine_molpdb(correctedmol, proteinfile, outfile=output_complex_file)
    print(f"Complex combination finished. Used {time.perf_counter() - st:.2f} seconds.")
  else:
    print("Failed to correct the molecule.")
    exit(1)

