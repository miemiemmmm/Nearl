import time, sys
import numpy as np 
from numpy.linalg import norm
from rdkit import Chem
from rdkit.Chem import AllChem

def _atom_matches_smarts(atom, smarts):
  idx = atom.GetIdx()
  patt = Chem.MolFromSmarts(smarts)
  for m in atom.GetOwningMol().GetSubstructMatches(patt):
    if idx in m:
      return True
  return False

def _sybyl_atom_type(atom):
  """ Asign sybyl atom type
  Reference #1: http://www.tripos.com/mol2/atom_types.html
  Reference #2: http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf
  """
  sybyl = None
  atom_symbol = atom.GetSymbol()
  atomic_num = atom.GetAtomicNum()
  hyb = atom.GetHybridization()-1  # -1 since 1 = sp, 2 = sp1 etc
  hyb = min(hyb, 3)
  degree = atom.GetDegree()
  aromtic = atom.GetIsAromatic()
  # define groups for atom types
  guanidine = '[NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])'  # strict
  if atomic_num == 6:
    if aromtic:
      sybyl = 'C.ar'
    elif degree == 3 and _atom_matches_smarts(atom, guanidine):
      sybyl = 'C.cat'
    else:
      sybyl = '%s.%i' % (atom_symbol, hyb)
  elif atomic_num == 7:
    if aromtic:
      sybyl = 'N.ar'
    elif _atom_matches_smarts(atom, 'C(=[O,S])-N'):
      sybyl = 'N.am'
    elif degree == 3 and _atom_matches_smarts(atom, '[$(N!-*),$([NX3H1]-*!-*)]'):
      sybyl = 'N.pl3'
    elif _atom_matches_smarts(atom, guanidine):  # guanidine has N.pl3
      sybyl = 'N.pl3'
    elif degree == 4 or hyb == 3 and atom.GetFormalCharge():
      sybyl = 'N.4'
    else:
      sybyl = '%s.%i' % (atom_symbol, hyb)
  elif atomic_num == 8:
    # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
    if degree == 1 and _atom_matches_smarts(atom, '[CX3](=O)[OX1H0-]'):
      sybyl = 'O.co2'
    elif degree == 2 and not aromtic:  # Aromatic Os are sp2
      sybyl = 'O.3'
    else:
      sybyl = 'O.2'
  elif atomic_num == 16:
    # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
    if degree == 3 and _atom_matches_smarts(atom, '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]'):
      sybyl = 'S.O'
    # https://github.com/rdkit/rdkit/blob/master/Data/FragmentDescriptors.csv
    elif _atom_matches_smarts(atom, 'S(=,-[OX1;+0,-1])(=,-[OX1;+0,-1])(-[#6])-[#6]'):
      sybyl = 'S.o2'
    else:
      sybyl = '%s.%i' % (atom_symbol, hyb)
  elif atomic_num == 15 and hyb == 3:
    sybyl = '%s.%i' % (atom_symbol, hyb)
  if not sybyl:
    sybyl = atom_symbol
  return sybyl
  
def writeMOL2(mol, filename):
  AllChem.ComputeGasteigerCharges(mol);
  # Get the Gasteiger charge of each atom in the molecule
  charges = [round(float(atom.GetProp("_GasteigerCharge")),6) for atom in mol.GetAtoms()]; 
  with open(filename, 'w') as f:
    # Write the Mol2 header
    f.write('@<TRIPOS>MOLECULE\n'); 
    f.write(mol.GetProp("_Name")+'\n'); 
    f.write('%d %d 0 0 0\n' % (mol.GetNumAtoms(), mol.GetNumBonds())); 
    # Write the atom block
    f.write('@<TRIPOS>ATOM\n'); 
    for i, atom in enumerate(mol.GetAtoms()):
      pos = mol.GetConformer().GetAtomPosition(i); 
      f.write(f'{i+1} {atom.GetSymbol()}{i+1} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f} {_sybyl_atom_type(atom)} 1 UNL {charges[i]}\n')
    # Write the bond block
    f.write('@<TRIPOS>BOND\n')
    for bond in mol.GetBonds():
      f.write('%d %d %d %s\n' % (bond.GetIdx()+1, bond.GetBeginAtomIdx()+1, bond.GetEndAtomIdx()+1, 'ar' if bond.GetIsAromatic() else int(bond.GetBondTypeAsDouble())))

def writeSDF(mol, filename): 
  writer = Chem.SDWriter(filename)
  writer.write(rdmol)
  writer.close()

def rotateMolByVector(mol, vector, ref_atom = 0): 
  conf = mol.GetConformer(); 
  coords = conf.GetPositions(); 
  hpartner = [i for i in mol.GetAtomWithIdx(ref_atom).GetNeighbors() if i.GetAtomicNum() == 1][0]; 
  reference_vector = np.array(conf.GetAtomPosition(hpartner.GetIdx())- conf.GetAtomPosition(ref_atom));
  # Normalize the required vectors; 
  reference_vector = reference_vector / norm(reference_vector); 
  vector = vector / norm(vector); 
  # Find the rotation axis and angle
  v = np.cross(reference_vector, vector); 
  c = np.dot(vector, reference_vector) / norm(vector); 
  angle = np.arccos(c); 
  # Create the rotation matrix
  s = np.sin(angle)
  skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  rot_matrix = np.eye(3) * c + (1 - c) * np.outer(v, v) + s * skew
  # Rotate the coordinates and update the coordinates 
  rotated_coords = np.dot(rot_matrix, coords.T).T
  for ai in range(mol.GetNumAtoms()):
    conf.SetAtomPosition(ai, rotated_coords[ai])
  return mol

def joinit(mol1, mol2, template, target): 
  for ai in range(mol1.GetNumAtoms()):
    atomi = mol1.GetAtomWithIdx(ai)
    if (atomi.GetAtomicNum() == 1) and (ai in target): 
      conf = mol1.GetConformer(); 
      hcoord = np.array(conf.GetAtomPosition(ai)); 
      atomi_neighbor = atomi.GetNeighbors()[0]; 
      pcoord = np.array(conf.GetAtomPosition(atomi_neighbor.GetIdx())); 
      # print(f"Index {ai}: Found the target H, replacing it", hcoord, "\nVector to align: ", hcoord - pcoord); 

      # Rotate the fragment molecule
      mol2 = rotateMolByVector(mol2, hcoord - pcoord); 
      conf_frag = mol2.GetConformer(); 
      rotated_coords = conf_frag.GetPositions(); 

      # Align the key atom to the hydrogen atoms
      frag_coords = rotated_coords + hcoord - rotated_coords[0]; 
      [conf_frag.SetAtomPosition(tmpi, frag_coords[tmpi]) for tmpi in range(mol2.GetNumAtoms())]; 
      hpartner = [i for i in mol2.GetAtomWithIdx(0).GetNeighbors() if i.GetAtomicNum() == 1][0]; 

      # Make molecules writable and Delete the unnecessary hydrogen atom
      rwmol1 = Chem.RWMol(mol1); 
      rwmol2 = Chem.RWMol(mol2); 
      rwmol1.RemoveAtom(ai); 
      rwmol1.UpdatePropertyCache()
      rwmol2.RemoveAtom(hpartner.GetIdx()); 
      rwmol2.UpdatePropertyCache(); 

      # Combine the two cleaned fragments, connect them and sanitize the new molecule
      mol1 = rwmol1.GetMol()
      mol2 = rwmol2.GetMol()
      combo = Chem.CombineMols(mol1, mol2); 
      rwmol = Chem.RWMol(combo); 
      rwmol.AddBond(atomi_neighbor.GetIdx(), mol1.GetNumAtoms(), order = Chem.rdchem.BondType.SINGLE); 
      rwmol.UpdatePropertyCache(); 
      Chem.SanitizeMol(rwmol); 

      # Optimize the conformation
      mollis = AllChem.ConstrainedEmbed(mol=rwmol, core=template, coreConfId=0, useTethers=True)
      # AllChem.UFFOptimizeMoleculeConfs(rwmol, numThreads=1)
      AllChem.MMFFOptimizeMoleculeConfs(rwmol, numThreads=1, maxIters=200)
      break
  # print(Chem.MolToMolBlock(rwmol))
  return rwmol.GetMol()
      
def replaceit(molfile, target, group,  format="mol2", num = 1): 
  """
    The molecule could be rdkit.Chem.rdchem.Mol class or a physical file; 
    
  """
  # Load the molecule to modify 
  if isinstance(molfile, Chem.rdchem.Mol): 
    rdmol = Chem.Mol(molfile.ToBinary());
    rdmol_template = Chem.RemoveHs(Chem.Mol(rdmol.ToBinary()));
    rdmol.UpdatePropertyCache();
  else: 
    with open(themol, "r") as file1: 
      if format == "mol2": 
        rdmol = Chem.MolFromMol2Block(file1.read(), removeHs=False, cleanupSubstructures=False, sanitize=False); 
      elif format == "sdf": 
        rdmol = Chem.MolFromMolBlock(file1.read(), removeHs=False, cleanupSubstructures=False, sanitize=False); 
      elif format == "pdb": 
        rdmol = Chem.MolFromPDBBlock(file1.read(), removeHs=False, cleanupSubstructures=False, sanitize=False); 
      else: 
        raise Exception("No valid format provided ")
      # Make the molecule writable
      rdmol_template = Chem.RemoveHs(Chem.Mol(rdmol.ToBinary()));
      rdmol.UpdatePropertyCache();
  
  all_h_idx = [int(ai) for ai in range(rdmol.GetNumAtoms()) if (rdmol.GetAtomWithIdx(ai).GetAtomicNum() == 1)]; 
  
  if isinstance(target, int):
    target = [int(target)]; 
    if not (target[0] in all_h_idx): 
      raise Exception(f"The target atom ({target[0]}) is not a hydrogen but a ({rdmol.GetAtomWithIdx(target[0]).GetSymbol()})"); 
  elif isinstance(target, list): 
    _target = target; 
    target = [int(i) for i in target if i in all_h_idx]; 
    if len(target) == 0: 
      errormsg = f"The target atoms " + ",".join([str(i) for i in _target]) + " has no hydrogen atoms"
      raise Exception(errormsg); 
  elif isinstance(target, str) and (target == "random"): 
    # Remember all the partners ???? 
    if num > len(all_h_idx):
      num = len(all_h_idx)
    target = np.random.choice(all_h_idx, num, replace=False); 
    if isinstance(target, int): 
      target = [target]; 
    else:
      """Note that numpy.int64 is not accepted by rdkit"""
      target = [int(i) for i in target]; 
    
  elif isinstance(target, str) and (target == "all"): 
    target = [int(ai) for ai in range(rdmol.GetNumAtoms()) if (rdmol.GetAtomWithIdx(ai).GetAtomicNum() == 1)]
    
  else: 
    raise Exception(f"Please define a integer to change a selected target"); 
  
  neighborlst = [rdmol.GetAtomWithIdx(ai).GetNeighbors()[0] for ai in target]; 
  if len(target) != len(neighborlst): 
    raise Exception(f"Hydrogen atom number ({len(target)}) and neighbor list does not match ({len(neighborlst)})"); 
  print(f"Found {len(target)} hydrogen atoms to modify;"); 
  # Initialize a fragment for modification
  frag = Chem.MolFromSmiles(group); 
  AllChem.EmbedMolecule(frag); 
  frag = AllChem.AddHs(frag, addCoords=True); 
  AllChem.MMFFOptimizeMolecule(frag, maxIters=50); 
  frag.UpdatePropertyCache(); 
  
  for idx, neighbor in enumerate(neighborlst): 
    st = time.perf_counter(); 
    tmptarget = [target[idx]]; 
    #     print("Target", t, atm.GetIdx()); 
    rdmol = joinit(rdmol, frag, rdmol_template, tmptarget); 
    # Update the target list in order to avoid possible index alteration problem
    _target = [i for i in target]; 
    target = [i if target[idx] > i else i-1 for i in _target]
    print(f"Processed the {idx+1:>3d}/{len(target):<3d} hydrogen atom (Used {time.perf_counter() - st:.3f} seconds)"); 
  return rdmol 

smilesdict={
  1: "C",
  2: "N",
  3: "O",
  4: "F",
  5: "P",
  6: "S",
  7: "Cl",
  8: "Br",
  9: "I",
  10: "C(F)(F)F",
  11: "C(F)F",
  12: "CF",
  13: "C(Cl)(Cl)Cl",
  14: "C(Cl)Cl",
  15: "CCl",
  16: "OC",
  17: "OC(=O)C"
} 

if __name__ == "__main__": 
  """ 
  Usage: 
    python3 Molecule_modification.py <input file> <outputfile> <mode of modification> <fragment selection> <other options>
  Example 
    python3 Molecule_modification.py /home/yzhang/Downloads/tmp.mol2 /tmp/test.mol2 random 1 4      # Randomly choose 4 hydrogens to replace with -CH3
    python3 Molecule_modification.py /home/yzhang/Downloads/tmp.mol2 /tmp/test.mol2 select 1 34%35  # Replace the hydrogen 34 and 35 with -CH3
    python3 Molecule_modification.py /home/yzhang/Downloads/tmp.mol2 /tmp/test.mol2 all    1        # Replace all of hydrogens by -CH3
  NOTE: 
    The replaceit function could use rdkit.Mol as input, meaning that replacement could be chained
    >>> rdmol = replaceit(themol, [33,34], "Cl");           # 1. replace hydrogens 33 and 34 with -Cl
    >>> rdmol = replaceit(rdmol, "random", "CC", num=2);   # 2. randomly replace two hydrogens with -CH2-CH3 (not including H33 and H34, replaced in the previous step)
    >>> rdmol = replaceit(rdmol, "all", "OC");              # 3. replace all hydrogens with -O-CH3 (including the hydrogens in the two -CH2-CH3 added in the previous step)
  """
  themol = sys.argv[1]; 
  outmol = sys.argv[2]; 
  mode = sys.argv[3]; 
  fragsel = sys.argv[4]; 
  if   mode == "random": 
    rdmol = replaceit(themol, "random", smilesdict[int(fragsel)], num=int(sys.argv[5])); 
  elif mode == "all": 
    rdmol = replaceit(themol, "all", smilesdict[int(fragsel)]); 
  elif mode == "select":
    rdmol = replaceit(themol, [int(i) for i in sys.argv[5].strip("%").split("%")], smilesdict[int(fragsel)]); 
  if len(outmol) > 0: 
    writeMOL2(rdmol, outmol); 


