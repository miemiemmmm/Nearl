from rdkit import Chem
from rdkit.Chem import AllChem
import pytraj as pt 
import numpy as np 
import tempfile

from . import CONFIG, _clear, _verbose, _tempfolder, printit

def DACbytraj(traj, frameidx, themask, **kwargs):
  """
  Count Hydrogen bond donors and acceptors by trajectory and selection
  """
  acceptor_pattern = '[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'
  donor_pattern = "[!H0;#7,#8,#9]"
  selection = traj.top.select(themask);
  if len(selection) == 0:
    print(f"{DACbytraj.__name__:15s}: No atom in the selected mask. Skipping it.")
    return np.array([]), np.array([])
  if (">" in themask) or ("<" in themask):
    print(f"{DACbytraj.__name__:15s}: Detected distance based mash. Please make sure that the reference is set to the trajectory, otherwise, all of the atoms will be processed");
  tmp_traj = traj.copy();
  if traj.top.select(f"!({themask})").__len__() > 0:
    tmp_traj.strip(f"!({themask})");
  if (tmp_traj.top.n_atoms == traj.top.n_atoms) and _verbose:
    print(f"{DACbytraj.__name__:15s}: All atoms are kept after applying the mask. Please make sure if this is wanted.")


  # with tempfile.NamedTemporaryFile(suffix=".pdb") as file1:
  #   pt.write_traj(file1.name, tmp_traj, overwrite=True, frame_indices=[frameidx])
  #   mol = Chem.MolFromPDBFile(file1.name);

  pdbstr = write_pdb_block(traj, selection, frame_index=frameidx)
  mol = Chem.MolFromPDBBlock(pdbstr);
  try:
    d_patt = Chem.MolFromSmarts(donor_pattern);
    d_hits = mol.GetSubstructMatches(d_patt);
    a_patt = Chem.MolFromSmarts(acceptor_pattern);
    a_hits = mol.GetSubstructMatches(a_patt);
    conf = mol.GetConformer()
    donors = np.zeros((len(d_hits),3));
    for idx, hit in enumerate(d_hits):
      atom = mol.GetAtomWithIdx(hit[0]);
      donors[idx,:] = np.array(conf.GetAtomPosition(hit[0]));
    acceptors = np.zeros((len(a_hits),3));
    for idx,hit in enumerate(a_hits):
      atom = mol.GetAtomWithIdx(hit[0])
      acceptors[idx,:] = np.array(conf.GetAtomPosition(hit[0]))
    return donors, acceptors
  except:
    print("Error when reading the pdb file. Please check the following PDB string:")
    print(pdbstr)
    return 0, 0

def Chargebytraj(traj, frameidx, themask):
  """
  Count Hydrogen bond donors and acceptors by trajectory and selection
  """
  selection = traj.top.select(themask);
  if len(selection) == 0:
    print(f"{Chargebytraj.__name__:15s}: No atom in the selected mask. Skipping it.")
    return np.array([]), np.array([])
  if (">" in themask) or ("<" in themask):
    print(f"{Chargebytraj.__name__:15s}: Detected distance based mash. Please make sure that the reference is set to the trajectory, otherwise, all of the atoms will be processed");
  tmp_traj = traj.copy();
  if traj.top.select(f"!({themask})").__len__() > 0:
    tmp_traj.strip(f"!({themask})");
  if (tmp_traj.top.n_atoms == traj.top.n_atoms) and _verbose:
    print(f"{Chargebytraj.__name__:15s}: All atoms are kept after applying the mask. Please make sure if this is wanted.")
  
  # with tempfile.NamedTemporaryFile() as file1:
  #   pt.write_traj(f"{file1.name}.pdb", tmp_traj, overwrite=True, frame_indices=[frameidx], options="model chainid A")
  #   with open(f"{file1.name}.pdb", "r") as f:
  #     pdbstr = f.read()
  #   if _clear:
  #     os.remove(f"{file1.name}.pdb");

  pdbstr = write_pdb_block(traj, selection, frame_index=frameidx)
  chargedict = {};
  try:
    mol = Chem.MolFromPDBBlock(pdbstr)
    AllChem.ComputeGasteigerCharges(mol);
    conf = mol.GetConformer();
    positions = conf.GetPositions();
    conf = mol.GetConformer();
    for idx, atom in enumerate(mol.GetAtoms()):
      key = tuple(np.array(conf.GetAtomPosition(idx)));
      chargedict[key] = float(atom.GetDoubleProp('_GasteigerCharge'));
  except:
    print("Error when reading the pdb file. Please check the following PDB string:")
    print(pdbstr)
  return chargedict

    # mol = Chem.MolFromPDBFile(f"{file1.name}.pdb");




# TODO: Why I used Pytraj to write PDB file and read by rdkit and finally write by rdkit?
# def write_pdb_block(traj, frameidx, themask):
#   selection = traj.top.select(themask);
#   if len(selection) == 0:
#     print(f"{write_pdb_block.__name__:15s}: No atom in the selected mask. Skipping it.")
#     return np.array([]), np.array([])
#   tmp_traj = traj.copy();
#   if traj.top.select(f"!({themask})").__len__() > 0:
#     tmp_traj.strip(f"!({themask})");
#   with tempfile.NamedTemporaryFile(suffix=".pdb") as file1:
#     pt.write_traj(file1.name, tmp_traj, overwrite=True, frame_indices=[frameidx])
#     mol = Chem.MolFromPDBFile(file1.name);
#   return Chem.MolToPDBBlock(mol);

def write_pdb_block(traj, frame_index=0, mask="*", write_pdb=False):
  selection = traj.top.select(mask);
  theframe = traj[frame_index];
  newxyz = np.asarray(theframe.xyz[selection]);
  newtop = traj.top._get_new_from_mask(mask);
  newtraj = pt.Trajectory(top=newtop, xyz=np.asarray([newxyz]));
  if write_pdb:
    with open(write_pdb, "w") as file1:
      pt.write_traj(file1.name, newtraj, overwrite=True)
    return None
  else:
    with tempfile.NamedTemporaryFile(suffix=".pdb") as file1:
      pt.write_traj(file1.name, newtraj, overwrite=True)
      with open(file1.name, "r") as file2:
        pdblines = [i for i in file2.read().split("\n") if "ATOM" in i or "HETATM" in i]
      pdbline = "\n".join(pdblines) + "\nEND\n"
    return pdbline


def write_pdb_block(thetraj, idxs, pdbfile = "", frame_index = 0, marks = [], swap4char=True):
  # Loop over each residue and atom, and write to the PDB file
  if (len(marks) > 0) and (len(marks) == len(idxs)):
    marks = marks;
  else:
    marks = ["ATOM"] * len(idxs);
  xyz_reduce = thetraj.xyz[frame_index].round(3);
  atom_arr = list(thetraj.top.atoms);
  res_arr = list(thetraj.top.residues);
  try:
    uc = thetraj.unitcells[index];
    spacegroup = "P 1";
    finalstr = f"TITLE    Topology Auto Generation : step =  {index}\n";
    finalstr += f"CRYST1 {uc[0]:8.3f} {uc[1]:8.3f} {uc[2]:8.3f} {uc[3]:6.2f} {uc[4]:6.2f} {uc[5]:6.2f} {spacegroup:<11}\n"
  except:
    finalstr = "";
  for i, idx in enumerate(idxs):
    coord = xyz_reduce[idx];
    theatom = atom_arr[idx];
    res_id = theatom.resid;
    theres = res_arr[res_id];
    _res_id = (res_id+1)%10000;
    atom_name = theatom.name;
    if atom_name[0].isnumeric():
      atom_name = f"{atom_name:<4}"
    elif (swap4char and len(atom_name) == 4):
      atom_name = atom_name[3]+atom_name[:3];
    elif (not atom_name[0].isnumeric() and len(atom_name) == 3):
      atom_name = f"{atom_name:>4}";
    elif (not atom_name[0].isnumeric() and len(atom_name) == 2):
      atom_name = f" {atom_name} ";
    elif (not atom_name[0].isnumeric() and len(atom_name) == 1):
      atom_name = f" {atom_name}  ";
    else:
      atom_name = f"{atom_name:4}"
    # Write the ATOM record to the PDB file
    finalstr += f"{marks[i]:<6s}{(i%99999)+1:>5} {atom_name:4} {theres.name:>3}  {_res_id:>4}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00\n";
  finalstr +="END"
  if len(pdbfile) > 0:
    with open(pdbfile, "w") as file1:
      file1.write(finalstr)
      return None
  else:
    return finalstr;


def combine_molpdb(molfile, pdbfile, outfile=""):
  """
  Combine a molecule file (either mol2/sdf/pdb) with a protein PDB file
  """
  # Read the ligand file
  if isinstance(molfile, Chem.rdchem.Mol):
    lig = molfile;
  else:
    if "mol2" in molfile:
      lig = Chem.MolFromMol2File(molfile);
    elif "pdb" in molfile:
      lig = Chem.MolFromPDBFile(molfile);
    elif "sdf" in molfile:
      suppl = Chem.SDMolSupplier(molfile);
      lig = suppl[0];
    else:
      raise ValueError(f"Unrecognized ligand file extension: {molfile.split('.')[-1]}");
  # Write the ligand to a PDB file and combine the protein PDB file
  ligpdb = Chem.MolToPDBBlock(lig);
  atomlines = [i.replace("UNL", "LIG") for i in ligpdb.split("\n") if "HETATM" in i];
  with open(pdbfile, "r") as file1:
    pdborig = file1.read();
  linesorig = [i for i in pdborig.strip("\n").split("\n") if "HETATM" in i or "ATOM" in i];
  finallines = linesorig + atomlines;
  finalstr = "\n".join(finallines) + "\nEND\n";
  # Check if the output file is specified
  if len(outfile) > 0:
    with open(outfile, "w") as file1:
      file1.write(finalstr);
  return finalstr

def CorrectMol2BySmiles(refmol2, prob_smiles):
  """
  Correct the reference mol2 structure based on a probe smiles
  >>> from utils_diverse import modification
  >>> molfile = "/storage006/yzhang/TMP_FOLDERS/w1SbtSzR/tmp_Sampling_target.mol2"
  >>> smi = "CNc1ncnc2c(C)n[nH]c12"
  >>> retmol = modification.CorrectMol2BySmiles(molfile, smi)
  >>> modification.writeMOL2s([retmol], "/tmp/test.mol2") # The output mol2 file that you want to put
  """
  from rdkit.Chem import rdFMCS;
  mol1 = Chem.MolFromSmiles(prob_smiles);
  print(refmol2)
  if "mol2" in refmol2:
    mol2 = Chem.MolFromMol2File(refmol2);
  elif "pdb" in refmol2:
    print("Reading pdb file")
    mol2 = Chem.MolFromPDBFile(refmol2);
  elif "sdf" in refmol2:
    suppl = Chem.SDMolSupplier(refmol2);
    mol2 = suppl[0];
  else:
    raise ValueError(f"Unrecognized ligand file extension: {molfile.split('.')[-1]}");

  # mol2 = Chem.MolFromMol2File(refmol2);
  if mol1 is None:
    print("Failed to process the smiles. Please check the validity of the smiles");
    return None;
  elif mol2 is None:
    print("Failed to process the mol2 file. Please check the validity of the mol2 file");
    return None;

  mol1 = Chem.AddHs(mol1, addCoords=True);
  AllChem.EmbedMolecule(mol1);
  # Find the maximum common subgraph (MCS) based on topology
  mcs = rdFMCS.FindMCS([mol1, mol2],
                       atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                       bondCompare=rdFMCS.BondCompare.CompareAny);
  # Get the MCS as an RDKit molecule
  mcs_mol = Chem.MolFromSmarts(mcs.smartsString);
  # Get the atom indices of the MCS in the input molecules
  match1 = mol1.GetSubstructMatch(mcs_mol);
  match2 = mol2.GetSubstructMatch(mcs_mol);
  atom_map = [(i, j) for i, j in zip(match1, match2)];
  AllChem.AlignMol(mol1, mol2, atomMap=atom_map, maxIters=100);
  Chem.SanitizeMol(mol1, sanitizeOps=Chem.SANITIZE_ALL);
  return mol1