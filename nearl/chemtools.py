import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolTransforms

from openbabel import openbabel as ob

from . import printit, config

# TODO check this function
# Default charge partially from Charmm36 forcefield
DEFAULT_PARTIAL_CHARGE = {
  "H": 0.09,    # <Atom name="HB1" type="ALA-HB1" charge="0.09"/>
  "C": -0.18,   # <Atom name="CB" type="LEU-CB" charge="-0.18"/>
  "N": -0.47,   # <Atom name="N" type="ALA-N" charge="-0.47"/>
  "O": -0.51,   # <Atom name="O" type="ALA-O" charge="-0.51"/>
  "F": -0.22,   # <Atom name="F21" type="FETH-F21" charge="-0.22"/>  FLUOROETHANE
  "P": 1.5,     # <Atom name="PA" type="ATP-PA" charge="1.5"/>
  "S": -0.09,   # <Atom name="SD" type="MET-SD" charge="-0.09"/>  ADENOSINE TRIPHOSPHATE
  "NA": 1.0,  "K": 1.0,
  "MG": 2.0,  "ZN": 2.0,  "CA": 2.0,
  "CL": -0.04,  # <Atom name="CL" type="CALD-CL" charge="-0.04"/>  CHLOROACETALDEHYDE
  "BR": -0.1,   # <Atom name="BR11" type="BRET-BR11" charge="-0.1"/>  BROMOETHANE
  "B": 0.13,    # <Atom name="B1" type="BORE-B1" charge="-0.13"/>  ETHYL BORONIC ACID
}

def traj_to_rdkit(traj, atomidx, frameidx):
  """
  Convert a pytraj trajectory to rdkit mol object
  """
  pdbstr = write_pdb_block(traj, atomidx, frame_index=frameidx)
  mol = Chem.MolFromPDBBlock(pdbstr, sanitize=False, removeHs=False)
  mol = sanitize_bond(mol)
  try:
    mol = Chem.AddHs(mol, addCoords=True)
    # AllChem.UFFOptimizeMolecule(mol, maxIters=10)
    # AllChem.MMFFOptimizeMolecule(mol, maxIters=10)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    AllChem.ComputeGasteigerCharges(mol)
    # Deal with the NaN in the Gasteiger charges
    for atom in mol.GetAtoms():
      if np.isnan(atom.GetDoubleProp('_GasteigerCharge')):
        atomsymbol = atom.GetSymbol().upper()
        if atomsymbol in DEFAULT_PARTIAL_CHARGE.keys():
          printit(f"Warning: Found Nan in rdkit molecule; Setting the atom {atomsymbol} to its default partial charge {DEFAULT_PARTIAL_CHARGE[atomsymbol]}")
          atom.SetDoubleProp('_GasteigerCharge', DEFAULT_PARTIAL_CHARGE[atomsymbol])
        else:
          atom.SetDoubleProp('_GasteigerCharge', 0.0)
          printit(f"Warning: Found NaN in rdkit molecule; Atom {atomsymbol} not found in default preset. _GasteigerCharge set to 0.0")
    if True in np.isnan([atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]):
      printit("DEBUG Warning: Still found nan in the charge", [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()])
    return mol
  except Exception as e:
    printit(f"Caught exception during rdkit mol generation: ", e)
    if config.debug():
      print("Please check the following PDB string:")
      print(pdbstr)
    return None


def label_ring_status(molfile):
  obConversion = ob.OBConversion()
  mol = ob.OBMol()
  molformat = os.path.basename(molfile).split(".")[-1]
  obConversion.SetInFormat(molformat)
  obConversion.ReadFile(mol, molfile)
  ret_arr = np.array([atom.IsInRing() for atom in ob.OBMolAtomIter(mol)], dtype=np.float32)
  return ret_arr

def label_hybridization(molfile):
  obConversion = ob.OBConversion()
  mol = ob.OBMol()
  molformat = os.path.basename(molfile).split(".")[-1]
  obConversion.SetInFormat(molformat)
  obConversion.ReadFile(mol, molfile)
  ret_arr = np.array([atom.GetHyb() for atom in ob.OBMolAtomIter(mol)], dtype=np.float32)
  return ret_arr

def label_aromaticity(molfile):
  obConversion = ob.OBConversion()
  mol = ob.OBMol()
  molformat = os.path.basename(molfile).split(".")[-1]
  obConversion.SetInFormat(molformat)
  obConversion.ReadFile(mol, molfile)
  ret_arr = np.array([atom.IsAromatic() for atom in ob.OBMolAtomIter(mol)], dtype=np.float32)
  return ret_arr

def label_hbond_donor(molfile): 
  mol = ob.OBMol()
  conv = ob.OBConversion()
  molformat = os.path.basename(molfile).split(".")[-1]
  conv.SetInFormat(molformat)
  conv.ReadFile(mol, molfile)
  retarr = np.full(mol.NumAtoms(), 0)
  for i in range(mol.NumAtoms()):
    atom = mol.GetAtom(i+1)
    if atom.IsHbondDonor():
      retarr[i] = 1
  return retarr

def label_hbond_acceptor(molfile): 
  mol = ob.OBMol()
  conv = ob.OBConversion()
  molformat = os.path.basename(molfile).split(".")[-1]
  conv.SetInFormat(molformat)
  conv.ReadFile(mol, molfile)
  retarr = np.full(mol.NumAtoms(), 0)
  for i in range(mol.NumAtoms()):
    atom = mol.GetAtom(i+1)
    if atom.IsHbondAcceptor():
      retarr[i] = 1
  return retarr


def DACbytraj(traj, frameidx, themask, **kwargs):
  """
  Count Hydrogen bond donors and acceptors by trajectory and selection
  """
  acceptor_pattern = '[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'
  donor_pattern = "[!H0;#7,#8,#9]"
  selection = traj.top.select(themask)
  if len(selection) == 0:
    printit(f"No atom in the selected mask. Skipping it.")
    return np.array([]), np.array([])
  if (">" in themask) or ("<" in themask):
    printit(f"Detected distance based mash. Please make sure that the reference is set to the trajectory, otherwise, all of the atoms will be processed");
  tmp_traj = traj.copy_traj()
  if traj.top.select(f"!({themask})").__len__() > 0:
    tmp_traj.strip(f"!({themask})")
  if (tmp_traj.top.n_atoms == traj.top.n_atoms) and config.verbose():
    printit(f"All atoms are kept after applying the mask. Please make sure if this is wanted.")

  pdbstr = write_pdb_block(traj, selection, frame_index=frameidx)
  mol = Chem.MolFromPDBBlock(pdbstr, sanitize=False, removeHs=False)
  mol = sanitize_bond(mol)

  try:
    d_patt = Chem.MolFromSmarts(donor_pattern)
    d_hits = mol.GetSubstructMatches(d_patt)
    a_patt = Chem.MolFromSmarts(acceptor_pattern)
    a_hits = mol.GetSubstructMatches(a_patt)
    conf = mol.GetConformer()
    donors = np.zeros((len(d_hits),3))
    for idx, hit in enumerate(d_hits):
      atom = mol.GetAtomWithIdx(hit[0])
      donors[idx,:] = np.array(conf.GetAtomPosition(hit[0]))
    acceptors = np.zeros((len(a_hits),3))
    for idx,hit in enumerate(a_hits):
      atom = mol.GetAtomWithIdx(hit[0])
      acceptors[idx,:] = np.array(conf.GetAtomPosition(hit[0]))
    return donors, acceptors
  except:
    print("Error when reading the pdb file. Please check the following PDB string:")
    print(pdbstr)
    return 0, 0

def Chargebytraj(traj, frameidx, atomidx):
  """
  Count Hydrogen bond donors and acceptors by trajectory and selection
  """
  if len(atomidx) == 0:
    print(f"{Chargebytraj.__name__:15s}: No atom in the selected mask. Skipping it.")
    return np.array([]), np.array([])
  atomnr = len(atomidx)
  coord = np.zeros((atomnr, 3))
  charges = np.zeros(atomnr)
  pdbstr = write_pdb_block(traj, atomidx, frame_index=frameidx)

  try:
    mol = Chem.MolFromPDBBlock(pdbstr)
    AllChem.ComputeGasteigerCharges(mol)
  except Exception as e:
    print("#############################################")
    print(f"During processing the original file: {traj.top_filename}; ")
    print("Failed to read PDB/compute the Gasteiger charges. Caught exception: ")
    print(e)
    print("Please check the following PDB string:")
    print(pdbstr)
    print("#############################################")
    return charges, coord

  if not np.isclose(atomnr, mol.GetNumAtoms()):
    atomnr = mol.GetNumAtoms()
    coord = np.zeros((atomnr, 3))
    charges = np.zeros(atomnr)
    print(f"Warning: The atom number of given atom index and PDB does not match. {atomnr} vs {mol.GetNumAtoms()}")

  try:
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    for idx, atom in enumerate(mol.GetAtoms()):
      coord[idx,:] = np.asarray(conf.GetAtomPosition(idx))
      charges[idx] = float(atom.GetDoubleProp('_GasteigerCharge'))
    return charges, coord
  except Exception as e:
    print("#############################################")
    print(f"During processing the original file: {traj.top_filename}; ")
    print("Error when assigning charges. Caught exception: ")
    print(e)
    print("Please check the following PDB string:")
    print(pdbstr)
    print("#############################################")
    atomnr = mol.GetNumAtoms()
    coord = np.zeros((atomnr, 3))
    charges = np.zeros(atomnr)
    return charges, coord

def write_pdb_block(thetraj, idxs, pdbfile="", frame_index=0, marks=[], swap4char=False):
  # Loop over each residue and atom, and write to the PDB file
  idxs = np.asarray(idxs)
  if (len(marks) > 0) and (len(marks) == len(idxs)):
    marks = marks
  else:
    marks = ["ATOM"] * len(idxs)
  xyz_reduce = thetraj.xyz[frame_index].round(3)
  atom_arr = list(thetraj.top.atoms)
  res_arr = list(thetraj.top.residues)
  finalstr = ""

  # create a dictionary to map old indices to new
  old_to_new_idx = {old: new for new, old in enumerate(idxs)}

  for i, idx in enumerate(idxs):
    coord = xyz_reduce[idx]
    theatom = atom_arr[idx]
    res_id = theatom.resid
    theres = res_arr[res_id]
    _res_id = (res_id+1)%10000
    atom_name = theatom.name
    if atom_name[0].isnumeric():
      atom_name = f"{atom_name:<4}"
    elif (swap4char and len(atom_name) == 4):
      atom_name = atom_name[3]+atom_name[:3]
    elif (not atom_name[0].isnumeric() and len(atom_name) == 3):
      atom_name = f"{atom_name:>4}"
    elif (not atom_name[0].isnumeric() and len(atom_name) == 2):
      atom_name = f" {atom_name} "
    elif (not atom_name[0].isnumeric() and len(atom_name) == 1):
      atom_name = f" {atom_name}  "
    else:
      atom_name = f"{atom_name:4}"
    # Write the ATOM record to the PDB file
    finalstr += f"{marks[i]:<6s}{(i%99999)+1:>5} {atom_name:4} {theres.name:>3}  {_res_id:>4}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00\n";
  finalstr = atom_name_mapping(finalstr)
  # for bond in thetraj.top.bonds:
  #   if (bond.indices[0] in old_to_new_idx) and (bond.indices[1] in old_to_new_idx):
  #     finalstr += f"CONECT{old_to_new_idx[bond.indices[0]]+1:>5}{old_to_new_idx[bond.indices[1]]+1:>5}\n";
  # finalstr += "ENDMDL\n"
  finalstr += "TER\n"
  if len(pdbfile) > 0:
    with open(pdbfile, "w") as file1:
      file1.write(finalstr)
      return None
  else:
    return finalstr

def atom_name_mapping(pdbstr):
  pdbstr = pdbstr.replace("CL", "Cl").replace("BR", "Br")
  pdbstr = pdbstr.replace("NA", "Na").replace("MG", "Mg").replace("ZN", "Zn")  # .replace("CA", "Ca").replace("CO", "Co")
  pdbstr = pdbstr.replace("FE", "Fe").replace("CU", "Cu").replace("NI", "Ni").replace("MN", "Mn")
  return pdbstr

def combine_molpdb(molfile, pdbfile, outfile=""):
  """
  Combine a molecule file (either mol2/sdf/pdb) with a protein PDB file
  """
  # Read the ligand file
  if isinstance(molfile, Chem.rdchem.Mol):
    lig = molfile
  else:
    if "mol2" in molfile:
      lig = Chem.MolFromMol2File(molfile)
    elif "pdb" in molfile:
      lig = Chem.MolFromPDBFile(molfile)
    elif "sdf" in molfile:
      suppl = Chem.SDMolSupplier(molfile)
      lig = suppl[0]
    else:
      raise ValueError(f"Unrecognized ligand file extension: {molfile.split('.')[-1]}")
  # Write the ligand to a PDB file and combine the protein PDB file
  ligpdb = Chem.MolToPDBBlock(lig)
  atomlines = [i.replace("UNL", "LIG") for i in ligpdb.split("\n") if "HETATM" in i]
  with open(pdbfile, "r") as file1:
    pdborig = file1.read()
  linesorig = [i for i in pdborig.strip("\n").split("\n") if "HETATM" in i or "ATOM" in i]
  finallines = linesorig + atomlines
  finalstr = "\n".join(finallines) + "\nEND\n"
  # Check if the output file is specified
  if len(outfile) > 0:
    with open(outfile, "w") as file1:
      file1.write(finalstr)
  return finalstr

def correct_mol_by_smiles(refmol2, prob_smiles):
  """
  Correct the reference mol2 structure based on a probe smiles
  >>> molfile = "/storage006/yzhang/TMP_FOLDERS/w1SbtSzR/tmp_Sampling_target.mol2"
  >>> smi = "CNc1ncnc2c(C)n[nH]c12"
  >>> retmol = utils.correct_mol_by_smiles(molfile, smi)
  >>> modification.writeMOL2s([retmol], "/tmp/test.mol2") # The output mol2 file that you want to put
  """
  mol1 = Chem.MolFromSmiles(prob_smiles)
  if "mol2" in refmol2:
    mol2 = Chem.MolFromMol2File(refmol2)
  elif "pdb" in refmol2:
    mol2 = Chem.MolFromPDBFile(refmol2)
  elif "sdf" in refmol2:
    suppl = Chem.SDMolSupplier(refmol2)
    mol2 = suppl[0]
  else:
    raise ValueError(f"Unrecognized ligand file extension: {molfile.split('.')[-1]}")

  # Check the validity of the given molecules
  # NOTE: Ligand PDB format does not contain bond information; PDB might be the more robust format
  # NOTE: Smiles has to correctly represent the molecule structure
  if mol1 is None:
    printit("Failed to process the smiles. Please check the validity of the smiles")
    return None
  elif mol2 is None:
    printit("Failed to process the mol2 file. Please check the validity of the mol2 file")
    return None

  mol1 = Chem.AddHs(mol1, addCoords=True)
  AllChem.EmbedMolecule(mol1)
  # Find the maximum common subgraph (MCS) based on topology
  mcs = rdFMCS.FindMCS([mol1, mol2],
                       atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                       bondCompare=rdFMCS.BondCompare.CompareAny)
  # Get the MCS as an RDKit molecule
  mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
  # Get the atom indices of the MCS in the input molecules
  match1 = mol1.GetSubstructMatch(mcs_mol)
  match2 = mol2.GetSubstructMatch(mcs_mol)
  atom_map = [(i, j) for i, j in zip(match1, match2)]
  AllChem.AlignMol(mol1, mol2, atomMap=atom_map, maxIters=100)
  Chem.SanitizeMol(mol1, sanitizeOps=Chem.SANITIZE_ALL)
  return mol1

# Single Bonds:
# C-C: Approximately 1.52 Ångstrom (Å)
# C-N: Approximately 1.47 Å
# C-O: Approximately 1.43 Å
# C-S: Approximately 1.82 Å
# N-H: Approximately 1.01 Å
# C-H: Approximately 1.09 Å
# O-H: Approximately 0.96 Å
# Double Bonds:
# C=O (carbonyl group): Approximately 1.23 Å
# C=N: Approximately 1.28 Å
# Triple Bonds:
# C≡N (nitrile group): Approximately 1.16 Å
# Peptide Bonds:
# C-N (in a peptide bond): Approximately 1.33 Å
# C=O (in a peptide bond): Approximately 1.23 Å
# Disulphide Bonds:
# S-S: Approximately 2.05 Å

BOND_LENGTH_MAP = {
  "C-C": 1.51, # single bond C-C

  "C~C": 1.39, # Aromatic bond C-C
  "C~N": 1.35, # Aromatic bond C-N

  "C-N": 1.47,  "N-C": 1.47,
  "C-O": 1.43,  "O-C": 1.43,
  "C-S": 1.82,  "S-C": 1.82,
  "N-H": 1.01,  "H-N": 1.01,
  "C-H": 1.09,  "H-C": 1.09,
  "O-H": 0.96,  "H-O": 0.96,
  "C=C": 1.34,
  "C=O": 1.23,  "O=C": 1.23,
  "C=N": 1.28,  "N=C": 1.28,
  "C≡N": 1.16,  "N≡C": 1.16,
  # "C-N": 1.33,  "N-C": 1.33,
  # "C=O": 1.23,  "O=C": 1.23,
  "S-S": 2.05,
  "other": 1.5
}


def sanitize_bond(mol_raw):
  conf = mol_raw.GetConformer()
  emol = Chem.EditableMol(mol_raw)
  bonds_to_remove = []
  _avail_atom_types = ["C", "N", "O", "S", "H"]
  # Iterate over each bond in the molecule
  for bond in mol_raw.GetBonds():
    begin_atom_idx = bond.GetBeginAtomIdx()
    end_atom_idx = bond.GetEndAtomIdx()
    # Get the bond length for this bond.
    bond_length = rdMolTransforms.GetBondLength(conf, begin_atom_idx, end_atom_idx)
    elem1 = mol_raw.GetAtomWithIdx(begin_atom_idx).GetSymbol()
    elem2 = mol_raw.GetAtomWithIdx(end_atom_idx).GetSymbol()
    if (elem1 not in _avail_atom_types) or (elem2 not in _avail_atom_types):
      # Skip if the bond is not between C, N, O, S, H
      continue

    bond_order = bond.GetBondTypeAsDouble()
    if np.isclose(bond_order, 1.0):
      bond_rep = f"{elem1}-{elem2}"
    elif np.isclose(bond_order, 1.5):
      printit("################ found aromatic bond")
      bond_rep = f"{elem1}~{elem2}"
    elif np.isclose(bond_order, 2.0):
      bond_rep = f"{elem1}={elem2}"
    elif np.isclose(bond_order, 3.0):
      bond_rep = f"{elem1}≡{elem2}"
    else:
      # Unknown bond order, use the uniformed bond length threshold
      bond_rep = "other"
    # Unknown bond type, use the uniformed bond length threshold
    if bond_rep not in BOND_LENGTH_MAP:
      bond_rep = "other"

    bond_length_expected = BOND_LENGTH_MAP[bond_rep]
    if (bond_length > bond_length_expected) and (not np.isclose(bond_length, bond_length_expected, rtol=0.1)):
      printit(f"Removing abnormal bond {bond_rep} lengthed {bond_length:.2f}/{bond_length_expected} angstorm, formed by {begin_atom_idx}@{elem1} - {end_atom_idx}@{elem2}")
      bonds_to_remove.append((begin_atom_idx, end_atom_idx))
  # Remove the bonds outside of the iteration loop
  for bond in bonds_to_remove:
    emol.RemoveBond(*bond)
  new_mol = emol.GetMol()
  try:
    Chem.SanitizeMol(new_mol)
    return new_mol
  except Exception as e:
    printit(f"Caught exception during sanitizing the molecule: {e}")
    
    # raise ValueError("Failed to sanitize the molecule")
    # print("Failed to use the rdkit to sanitize the molecule, trying to use Biopython.")
    # with tempfile.NamedTemporaryFile() as temp:
    #   temp.write(pdb_string.encode())
    #   temp.close()
    #   parser = PDBParser(PERMISSIVE=1)
    #   structure = parser.get_structure("temp", temp.name)
    # print(structure)
    return new_mol


def molfile_to_rdkit(file_path, **kwarg):
  """
  Read a molecule file and return a list of rdkit molecule objects

  Notes
  -----
  The removeHs and sanitize flags are set False by default is to maximize the possibility to read molecules
  """
  rm_h = kwarg.get("removeHs", False)
  san = kwarg.get("sanitize", False)
  file_extension = os.path.splitext(file_path)[1]
  if file_extension == '.mol2':
    suppl = Mol2Supplier(file_path, removeHs=rm_h, sanitize=san)
  elif file_extension == '.sdf':
    suppl = Chem.SDMolSupplier(file_path, removeHs=rm_h, sanitize=san)
  elif file_extension == '.pdb':
    suppl = Chem.MolFromPDBFile(file_path, removeHs=rm_h, sanitize=san)
    return [suppl]
  elif file_extension == '.smi' or file_extension == '.smiles':
    suppl = Chem.SmilesMolSupplier(file_path, titleLine=False, sanitize=True)
  elif file_extension == '.inchi':
    with open(file_path, "r") as file1:
      mols = file1.read().strip("\n").split("\n")
    suppl = [Chem.MolFromInchi(m, sanitize=True) for m in mols]
  else:
    raise ValueError(f'Unsupported file format: {file_extension}')
  return [mol for mol in suppl]


class Mol2Supplier:
  def __init__(self, file_path, *args, **kwarg):
    self.file_path = file_path
    self.molecules = []
    self._parse_mol2_file(*args, **kwarg)

  def _parse_mol2_file(self, *args, **kwarg):
    with open(self.file_path, "r") as file:
      mol_strs = [f"@<TRIPOS>MOLECULE{i}" for i in file.read().split("@<TRIPOS>MOLECULE") if len(i)>0]
      for mol_str in mol_strs:
        mol = Chem.MolFromMol2Block(mol_str, *args, **kwarg)
        if mol != None:
          self.molecules.append(mol)
        else:
          printit("Failed to read MOL")

  def __iter__(self):
    return iter(self.molecules)

  def __len__(self):
    return len(self.molecules)

  def __getitem__(self, index):
    return self.molecules[index]

