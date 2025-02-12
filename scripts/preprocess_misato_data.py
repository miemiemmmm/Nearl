import json

import pytraj as pt
import numpy as np 
import pandas as pd

# from Nearl import utils, data_io

import nearl
from feater import io
#### Task1:
# 1. Check if the QM and MD data are present in the database
# 2. If NO, update the supercedes list

#### Task2:
# Find the ligand atom index in the MD trajectory
# 1. Initial selection is ligand
# 2. QM_AtomNumber is close to the last segment
# 3. QM_AtomNumber exactly matches a segment atom numbers
# 4. QM_AtomNumber is close to an arbitrary segment

#################### Summary ####################
# From train_MD.txt (13765 entries in total):
# 699 entries are without :MOL (primarily multimer peptides or amino acids)
# ----> 109    entries does not match the QM ligand
# ----> 13656  entries match the QM ligand or has the MOL segment
# From test_MD.txt (1612 entries in total):
# ----> 32     entries does not match the QM ligand
# ----> 1580   entries match the QM ligand or has the MOL segment
# From val_MD.txt (1595 entries in total):
# ----> 28     entries does not match the QM ligand
# ----> 1567   entries match the QM ligand or has the MOL segment
# For all_MD (16972 entries in total):
# ----> 169    entries does not match the QM ligand
# ---->     entries match the QM ligand or has the MOL segment
##################################################



# Update the supercedes list if necessary
# Some minor changes are made between different versions of PDBBind

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

misato_dir = "/home/yzhang/Downloads/misato_database/"
mdfile = "/MieT5/DataSets/misato_database/MD.hdf5"
qmfile = "/home/yzhang/Downloads/misato_database/QM.hdf5"
parmdir = "/home/yzhang/Downloads/misato_database/parameter_restart_files_MD"
listtocheck = "/home/yzhang/Downloads/misato_database/all_misato_MD.txt"
output_prefix = "/home/yzhang/Downloads/misato_database/all_misato_MD"

table = pd.read_csv(
  "/MieT5/Nearl/data/PDBBind_general_v2020.csv",
  delimiter=",", header=0
)
pdblist = table.pdbcode.tolist(); 

def misato_ligand_atomnr(pdbcode, misato_qmfile): 
  with io.hdffile(misato_qmfile) as f1: 
    # ['atom_properties' 'mol_properties']
    # ['atom_names' 'atom_properties_names' 'atom_properties_values' 'bonds']
    atomnr = f1.data(f"/{pdbcode.upper()}/atom_properties/atom_names").__len__()
  return atomnr

with io.hdffile(qmfile) as f1:
  count = 0
  for pdbcode in f1.keys():
    if pdbcode.lower() not in pdblist and pdbcode.lower() not in SUPERSEDES.keys():
      count += 1
  print(f"NOTE: {count} QM pdb entries not found in PDBBind v2020")

with io.hdffile(mdfile) as f1:
  count = 0
  for pdbcode in f1.keys():
    if pdbcode.lower() not in pdblist and pdbcode.lower() not in SUPERSEDES.keys():
      count += 1
  print(f"NOTE: {count} MD pdb entries not found in PDBBind v2020")

# thestr = "1B9J%3GJD%5MLO%2Q2A%6B5O%4U6X%2EZ5%5D0J%5NW8%1S9V%4ES0%2JDL%4UX9%4J8B%1HAA%4EZZ%3SW9%3UIH%1D8E%1PZ5%6G8J%1JET%3AVB%3O1D%3H52%1SJH%4K6Y%2C6G%1X7Q%5J7J%1LST%2QBW%1RGJ%6HOL%1R5W%6NU5%1RST%4GW1%1WKM%4HMK%6HOI%6BIY%4CY1%5IY4%1HC9%4K78%6HV2%4AA2%4FGY%4ERQ%4E81%6F55%3F3A%2MS4%3GHE%1NU8%3QS5%3ZMU%3FAS%4FYS%6MQE%4DMA%3C89%3U93%2DF6%3LNZ%3AVI%1SM3%4J46%4WY7%1RLQ%1OQP%6EH2%3NIK%1KLU%4K72%6MLG%5N7X%2Y8O%4JOK%6HZB%6D3Z%3ASL%5ZK5%5VWI%4B8P%1JEU%4I33%4PGD%4EZY%5JJM%5MXO%1HSL%2MWY%6BCR%5ZOP%4M1D%6ERU%4IN9%5TKJ%2BBA%4BA3%2FX7%6HHP%4J79%1CKB%1EJ4%4YMX%6P3W%2XS8%4PGE%5IAY%5E0L%2Q2C%5MK3%3DAB%5T6Z%5VQI%5UMZ%1B3F%3AVF%1GAG%1UJ0%1B46%3AVN%5AEI%3TPU%3LRH%1QWE%4J45%4U2W%1LAF%4JOE%2N9X%1H9L%4GXL%6G8L%2YNS%4DGB%4NUF%1B51%4J84%6KMJ%6G8P%1JD6%2LLQ%1CKA%2MNZ%3U0T%3F3D%2R02%4PRB%3L3X%1SLG%4O36%2LTW%2XXR%2Y07%6A9C%3UYR%4X1Q%6IQG%4RXZ%3ZMZ%2LTZ%4FMQ%6Q4Q%6MLJ%6B5M%3C3O%5GP7%5KSV%3C8A%5OK3%4J47%2JO9%2XRW%4TS1%6JJZ%4OEL%6CDO%2RQU%4JOF%2BJ4%6EGW%1QWF%5T6P%2QHR%3DIW%6NJZ%3G2W%1A30%1ZHK%4WK2%1FO0%5LGS%5LGP%6D40%2R03%2OLB%1UKH%6UYX%4IZM%3JZP%5SVZ%3OAP%3AVG%5N70%2M3O%4ZHL%1ZM6%3UVK%1Q5L%1QKA%2OQS%6O21%4U6Y%2Y9Q%6MT4%2W10%6HMT%6MT5%4H39%3RWD%6A8N%3RZ9%5TKK%2Y36%1KC5%2LTX%2R05%6MKU%4J73%4X2S%3B3S%2XXW%2A5S%5YC2%1R2B%4EJF%6B5R%3D25%4O3C%4X1S%2MPS%5YBA%4E3B%1JQ9%1QKB%1XB7%5I8C%6I4X%6N9T%1R5V%1OSV%3Q5U%4HFZ%5A2I%4FXZ%3AVA%2YNR%3UII%4ZHM%1UJJ%6K5R%6D07%3C88%3DS0%4J7I%6FZF%6QK8%6BAU%3G2U%1Y2A%4PRD%4ZV1%1AZE%4PRN%1KL5%3TIW%5LGQ%4Q6F%6CF6%2KRD%4ESG%4B4N%6HZC%1B5I%6MKW%5CIL%1I5H%6FRJ%3D3X%4O2E%4EZR%4RME%2NXM%3BTR%5WKH%5T70%3FT4%6CDP%3BIM%5KLT%3DVP%4K1E%4ONF%3IFL%4H3Q%5NIN%2LL6%3TJH%1XH3%4YK6%1IQ1%6QC0%6FKP%2D1X%6ML9%1ABT%3SOU%3EMH%4O2F%4QH7%2NXD%6BVH%1T37%3DS3%4EZT%1FTJ%1OXG%1S50%5XHZ%6HZD%6D3Y%2BR8%3C3Q%3QNJ%2IGW%2LGF%5GU4%3AVK%6MLI%2FYS%6GHJ%4J77%5E0M%2Z5S%3IVQ%3JZQ%6AM8%3C3R%5WA1%3FN0%3UVM%4RRV%4WV6%5T78%4G69%3RL8%1MQ1%6MQM%2NM1%6MLA%1ZHL%4ZV2%1LEK%5B56%3S9E%4EOY%5YY9%1H28%4RXH%1I8I%4MZ5%5V5O%5AZF%5E4W%4WCI%5CSZ%1XR9%6NKP%4O45%3AV9%6GY5%3UIK%6UYY%3AVH%1GWR%3TG5%4J8R%3BZE%4E34%2LL7%3ZVY%4G4P%4WHY%6HLB%3UEF%3LP4%2XL2%1BXL%5JM4%2Y06%2FO4%6GG4%1A3E%3F3E%3JZR%5OXN%4EWR%4KQP%3F48%4EZO%6GG5%3UIJ%2YNN%2ROL%3AVL%6BMI%6HY7%3FF3%6E49%5OXK%1F3J%2Y7I%4U0C%6MLO%4U0A%2KBS%5MLW%6G8I%2XZQ%3O6M%1H24%5OXL%3ZMT%1LEG%1WDN%3BZF%4EP2%4Y7R%1AZG%5VZY%6O9B%2Y6S%1MWN%3DLN%2I3H%4B2D%5KSU%4XC2%4HOD%3JZS%3AVM%5EPP%3UVL%3IVV%3L3Q%3EQS%1JUQ%2I3V%1DXP%5W4E%4J48%1INQ%6ERV%6MLN%6I5J%5LAX%3D1E%6ISO%1JWU%5HUY%1B52%3D1F%4ERZ%3DS4%4K3M%6EMA%3DS1%5KLR%1OXQ%3QG6%5ETF%4JOG%5IAW%4ZDU%6CDM%1F90%6FZP%6ERW%2XL3%1JWS%4HT6%2BGR%3ZN0%4ERY%2E7L%6AMI%5T8R%1KCS%6MQC%5YCO%6K5T%2LAS%3RSE%5NJX%4E35%6BYK%6HZA%3QS4%1JQ8%6QCG%5OXM%1OY7%4I31%2NXL%4J82%6GZL%3SHB%1RLP%1G7P%2OI9%6H8C%5A2K%4KMD%3AVJ%5AZG%3RWE%3C8B%3ZMV%6MM5%1H3H%4J44%4JOH%5N7B%4IS6%1B3G%5O0E%4JOJ%1JUF%3O6L%1B58%1B05%4N7G%3UVU%1N7T%1T7R%2PCU%1JP5%4X1N%6BD1%3TF7%6F8G%2L8J%5EAY%3JZO%1JBD%6MLP%1XHM%3NY3%1OXN%3N5U%3ERY%6HLE%6EM6%3EQY%3IFP%2ZNS%4X1R%5H1E%4DS1%4ZNX%4Z68%2IGV%5OWF%1JEV%2OR9%2X4S%3UVO%3NIJ%1JWM%4YJL%4PRA%1PRM%1B4Z%4J78%5IXF%3B3W%4GQ6%4PR5%4O3B%1MXL%6GGV%2NWN%3CS8%5I25%1LXH%6O9C%1MFG%4GW5%4EQJ%3RWJ%1IWQ%2Z5T%4J26%3S7F%5N1Y%2HUG%6BCY%1OK7%1B32%4EZX%1KL3%3NIN%6MLE%2EH8%1H25%3UI2%4XYN%5A2J%3G2S%5CIN%1B5J%5YGF%4RIS%1SJE%5ZOO%4MZ6%5V3R%3FUZ%3UVN%4EZQ%3NIM%4B8O%3HQH%2HRP%5OL3%1B40%2FLU%5B6C%1Y3A%5U06%1XR8%5OUA%3ZN1%1LAG%3NII%3T6R%4PRE%6EM7%1KLG%3RWI%4YJE%5ICK%4K75%4AA1%4LNP%5JIN%2PVU%3L3Z%4F14%3P4F%4PG9%2XS0%1UTI%1SFI%3O1E%1G7Q%6PIT%4J86%3RWG%2LTV%3RWF%3NIL%1G3F%4H3B%4I32%2KE1%6NCP%4N7H%2XXX%2W0Z%2VWF%6G6X%3FT3%4D2D%1JD5%2LLO%5FB1%3IQQ%2H9M%3QS6%1MF4%2RKM%2CV3%3D32%4OU3%"
# thelist = thestr.strip("%").split("%")
with io.hdffile(mdfile) as f1:
  with open(listtocheck,"r") as file_train:
    thelist = file_train.read().strip("\n").split("\n")

  final_lig_dict = {}
  c = 0
  for pdbcode in thelist:
    # Count total atom number and segmentation info
    atomnr = len(f1.data(f"/{pdbcode.upper()}/atoms_type"))
    bgatoms = f1.data(f"/{pdbcode.upper()}/molecules_begin_atom_index")
    segment_nrs = []
    for i in range(len(bgatoms)): 
      if i < len(bgatoms)-1: 
        leni = bgatoms[i+1] - bgatoms[i]
      else: 
        leni = atomnr - bgatoms[-1]
      segment_nrs.append(leni)
    segment_nrs =np.array(segment_nrs)
    
    # Check the ligand atom number in the QM. 
    try:
      qm_atomnr = misato_ligand_atomnr(pdbcode, qmfile)
    except:
      print(f"PDB code {pdbcode}: not found in the QM table");
      qm_atomnr = 0
    seg_diffs = np.abs(qm_atomnr-segment_nrs)
      
    source = "lig"
    # Check the last segment of the protein
    ligand_indices = np.where(f1.data(f"/{pdbcode.upper()}/atoms_residue") == 0)[0]
    # if the ligand is a peptide and not a small molecule
    if len(ligand_indices)==0: 
      source = "prot_last"
      ligand_indices = np.array([i for i in range(bgatoms[-1], atomnr)])
    
    if source == "lig":
      # Directly pass if the atoms are from the ligand
      final_indices = ligand_indices;

    elif abs(qm_atomnr - len(ligand_indices)) <= 4:
      # Last segment is close to the QM_reference
      final_indices = ligand_indices;
      
    elif qm_atomnr in segment_nrs: 
      # QM reference already in the segment number; 
      seg_idx = segment_nrs.tolist().index(qm_atomnr); 
      # NOTE: the first condition qm_reference does not equal to the last segment. 
      final_indices = np.arange(bgatoms[seg_idx], bgatoms[seg_idx+1]);

    elif seg_diffs.min() < 4:
      # QM reference close to one arbitrary segment; 
      seg_idx = seg_diffs.tolist().index(seg_diffs.min())
      if (seg_idx+1) > len(bgatoms): 
        end_idx = atomnr
      else: 
        end_idx = bgatoms[seg_idx+1]
      final_indices = np.arange(bgatoms[seg_idx], end_idx)
      # print("pathway 3", qm_atomnr, len(final_indices), f"({seg_idx+1}/{len(segment_nrs)})")
    
    else: 
      c += 1
      final_indices = []
      traj = nearl.io.traj.MisatoTraj(pdbcode, misato_dir); 
      print(f"{c} :: Not Found {source} :: MOL selection",traj.top.select(":MOL"), qm_atomnr, "-> Diff ", seg_diffs)
      # print(ligand_indices)
    if pdbcode.lower() in nearl.constants.PDBCODE_SUPERCEDES: 
      pdbidx = pdblist.index(SUPERSEDES[pdbcode.lower()].lower())
    else: 
      pdbidx = pdblist.index(pdbcode.lower())

    # comment = table.iloc[pdbidx][-1]
    if len(final_indices)> 0:
      final_lig_dict[pdbcode] = [int(i) for i in final_indices]
    
with open(f"{output_prefix}_indices.json", "w") as f1:
  json.dump(final_lig_dict, f1, indent=2)

with open(f"{output_prefix}_refined.txt", "w") as f1:
  pdblst_final = list(final_lig_dict.keys())
  print("Total number of PDBs: ", len(pdblst_final))
  f1.write("\n".join(pdblst_final)+"\n")


#     traj = utils.misato_traj(pdbcode, mdfile, parmdir);
#     pt.write_traj(f"/tmp/testmisato{pdbcode}.pdb", traj, frame_indices=[0], overwrite=True)
#     pt.write_traj(f"/tmp/testmisato{pdbcode}.nc", traj, overwrite=True)
