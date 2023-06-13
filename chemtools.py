from rdkit import Chem
from rdkit.Chem import AllChem
import pytraj as pt 
import numpy as np 
import tempfile

from . import CONFIG

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
  if (tmp_traj.top.n_atoms == traj.top.n_atoms) and CONFIG["verbose"]:
    print(f"{DACbytraj.__name__:15s}: All atoms are kept after applying the mask. Please make sure if this is wanted.")
  with tempfile.NamedTemporaryFile(suffix=".pdb") as file1: 
    pt.write_traj(file1.name, tmp_traj, overwrite=True, frame_indices=[frameidx])
    mol = Chem.MolFromPDBFile(file1.name); 
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
  if (tmp_traj.top.n_atoms == traj.top.n_atoms) and CONFIG["verbose"]:
    print(f"{Chargebytraj.__name__:15s}: All atoms are kept after applying the mask. Please make sure if this is wanted.")
  
  with tempfile.NamedTemporaryFile(suffix=".pdb") as file1:
    pt.write_traj(file1.name, tmp_traj, overwrite=True, frame_indices=[frameidx])
    mol = Chem.MolFromPDBFile(file1.name);

  AllChem.ComputeGasteigerCharges(mol); 
  conf = mol.GetConformer(); 
  positions = conf.GetPositions(); 
  chargedict = {}; 
  conf = mol.GetConformer(); 
  for idx, atom in enumerate(mol.GetAtoms()): 
    key = tuple(np.array(conf.GetAtomPosition(idx))); 
    chargedict[key] = float(atom.GetDoubleProp('_GasteigerCharge')); 
  return chargedict; 


def writepdbs(traj, frameidx, themask):
  selection = traj.top.select(themask);
  if len(selection) == 0:
    print(f"{writepdbs.__name__:15s}: No atom in the selected mask. Skipping it.")
    return np.array([]), np.array([])
  tmp_traj = traj.copy();
  if traj.top.select(f"!({themask})").__len__() > 0:
    tmp_traj.strip(f"!({themask})");
  with tempfile.NamedTemporaryFile(suffix=".pdb") as file1:
    pt.write_traj(file1.name, tmp_traj, overwrite=True, frame_indices=[frameidx])
    mol = Chem.MolFromPDBFile(file1.name);
  return Chem.MolToPDBBlock(mol);


