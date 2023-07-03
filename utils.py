import tempfile
from hashlib import md5;

import numpy as np 
import pytraj as pt 

from scipy.interpolate import griddata, Rbf
from scipy.spatial import distance_matrix


def conflictfactor(pdbfile, ligname, cutoff=5):
  VDWRADII = {'1': 1.1, '2': 1.4, '3': 1.82, '4': 1.53, '5': 1.92, '6': 1.7, '7': 1.55, '8': 1.52,
    '9': 1.47, '10': 1.54, '11': 2.27, '12': 1.73, '13': 1.84, '14': 2.1, '15': 1.8,
    '16': 1.8, '17': 1.75, '18': 1.88, '19': 2.75, '20': 2.31, '28': 1.63, '29': 1.4,
    '30': 1.39, '31': 1.87, '32': 2.11, '34': 1.9, '35': 1.85, '46': 1.63, '47': 1.72,
    '48': 1.58, '50': 2.17, '51': 2.06, '53': 1.98, '54': 2.16, '55': 3.43, '56': 2.68,
    '78': 1.75, '79': 1.66, '82': 2.02, '83': 2.07
  }
  traj = pt.load(pdbfile, top=pdbfile);
  traj.top.set_reference(traj[0]);
  pocket_atoms = traj.top.select(f":{ligname}<:{cutoff}");
  atoms = np.array([*traj.top.atoms])[pocket_atoms];
  coords = traj.xyz[0][pocket_atoms];
  atomnr = len(pocket_atoms);
  cclash=0;
  ccontact = 0;
  for i, coord in enumerate(coords):
    partners = [atoms[i].index]
    for j in list(atoms[i].bonded_indices()):
      if j in pocket_atoms:
        partners.append(j)
    partners.sort()
    otheratoms = np.setdiff1d(pocket_atoms, partners)
    ret = distance_matrix([coord], traj.xyz[0][otheratoms])
    thisatom = atoms[i].atomic_number
    vdw_pairs = np.array([VDWRADII[str(i.atomic_number)] for i in np.array([*traj.top.atoms])[otheratoms]]) + VDWRADII[str(thisatom)]
    cclash += np.count_nonzero(ret < vdw_pairs - 1.25)
    ccontact += np.count_nonzero(ret < vdw_pairs + 0.4)

    st = (ret < vdw_pairs - 1.25)[0];
    if np.count_nonzero(st) > 0:
      partatoms = np.array([*traj.top.atoms])[otheratoms][st];
      thisatom = np.array([*traj.top.atoms])[atoms[i].index];
      for part in partatoms:
        dist = distance_matrix([traj.xyz[0][part.index]], [traj.xyz[0][thisatom.index]]);
        print(f"Found clash between: {thisatom.name}({thisatom.index}) and {part.name}({part.index}); Distance: {dist.squeeze().round(3)}")

  factor = 1 - ((cclash/2)/((ccontact/2)/atomnr))
  print(f"Clashing factor: {round(factor,3)}; Atom selected: {atomnr}; Contact number: {ccontact}; Clash number: {cclash}");
  return factor

def interpolate(points, weights, grid_dims):
  """
  Interpolate density from a set of weighted 3D points to an N x N x N mesh grid.

  Args:
  points (np.array): An array of shape (num_points, 3) containing the 3D coordinates of the points.
  weights (np.array): An array of shape (num_points,) containing the weights of the points.
  grid_size (int): The size of the output mesh grid (N x N x N).

  Returns:
  np.array: A 3D mesh grid of shape (grid_size, grid_size, grid_size) with the interpolated density.

  Note: 
    Interpolation box is in accordance with the bound box
  """
  # Compute the bounding box
  min_coords = np.min(points, axis=0)
  max_coords = np.max(points, axis=0)
  # Create the X x Y x Z grid
  grid_dims = [int(i) for i in grid_dims]; 
  grid_y, grid_x, grid_z = np.mgrid[min_coords[0]:max_coords[0]:grid_dims[0]*1j, 
                                    min_coords[1]:max_coords[1]:grid_dims[1]*1j, 
                                    min_coords[2]:max_coords[2]:grid_dims[2]*1j]; 
  grid_x, grid_y, grid_z = np.meshgrid(np.linspace(min_coords[0], max_coords[0], grid_dims[0]),
                                       np.linspace(min_coords[1], max_coords[1], grid_dims[1]),
                                       np.linspace(min_coords[2], max_coords[2], grid_dims[2]), indexing="xy")
  # Interpolate the density to the grid
  # grid_density = griddata(points, weights, (grid_x, grid_y, grid_z), method='linear', fill_value=0); 
  # Perform RBF interpolation
  rbf = Rbf(points[:, 0], points[:, 1], points[:, 2], weights)
  grid_density = rbf(grid_x, grid_y, grid_z)

  return grid_density, (grid_x, grid_y, grid_z)

def get_hash(thestr=""):
  if isinstance(thestr, str) and len(thestr)>0:
    return md5(bytes(thestr, "utf-8")).hexdigest()
  elif (isinstance(thestr, list) or isinstance(thestr, np.ndarray)) and len(thestr)>0:
    arrstr = ",".join(np.asarray(thestr).astype(str))
    return md5(bytes(arrstr, "utf-8")).hexdigest()
  else:
    import time;
    return md5(bytes(time.perf_counter().__str__(), "utf-8")).hexdigest()

# Two ways to get the mask of protein part
def GetProteinByTraj(traj):
  """
  Method 1: Get the mask of protein via a trajectory object
  """
  reslst = []
  for i in traj.top.atoms:
    if i.name=="CA":
      reslst.append(i.resid+1)
  mask = ":"+",".join([str(i) for i in reslst])
  return traj.top.select(mask)

def GetProteinMask(pdbfile):
  """
  Method 2: Get the mask of protein via a PDB file name 
  """
  reslst = [];
  traj = pt.load(pdbfile, top=pdbfile);
  for i in traj.top.atoms:
    if i.name=="CA":
      reslst.append(i.resid+1)
  mask = ":"+",".join([str(i) for i in reslst])
  return mask

def PairwiseDistance(traj, mask1, mask2, use_mean=False, ref_frame=None):
  """
  Get the pairwise distance between two masks
  Usually they are the heavy atoms of ligand and protein within the pocket
  """
  selmask1 = traj.top.select(mask1);
  selmask2 = traj.top.select(mask2);
  if len(selmask1) == 0:
    print("No target atom selected, please check the mask")
    return None, None;
  elif len(selmask2) == 0:
    print("No counter part atom selected, please check the mask")
    return None, None;
  atom_names = np.array([i.name for i in traj.top.atoms]);
  atom_ids = np.array([i.index for i in traj.top.atoms]);

  # Compute the distance matrix between the target and reference atoms
  if use_mean == True:
    frame_mean = np.mean(traj.xyz, axis=0);
    this_ligxyz = frame_mean[selmask1];
    this_proxyz = frame_mean[selmask2];
    ref_frame = distance_matrix(this_ligxyz, this_proxyz);
  else:
    this_ligxyz = traj.xyz[ref_frame if ref_frame is not None else 0][selmask1];
    this_proxyz = traj.xyz[ref_frame if ref_frame is not None else 0][selmask2];
    ref_frame = distance_matrix(this_ligxyz, this_proxyz);

  # Find closest atom pairs
  minindex = np.argmin(ref_frame, axis=1);
  selclosest = selmask2[minindex];

  # Compute the evolution of distance between closest-pairs
  # NOTE: Add 1 to pytraj index (Unlike Pytraj, in Cpptraj, Atom index starts from 1)
  distarr = np.array([pt.distance(traj, f"@{i + 1} @{j + 1}") for i, j in zip(selmask1, selclosest)]);

  gp1_names = atom_names[selmask1].tolist();
  gp1_ids   = atom_ids[selmask1].tolist();
  gp2_names = atom_names[selclosest].tolist();
  gp2_ids   = atom_ids[selclosest].tolist();
  return distarr, { "atom_name_group1":gp1_names, "index_group1":gp1_ids, "atom_name_group2":gp2_names, "index_group2":gp2_ids }



def ASALig(pdbfile, lig_mask):
  """
  Calculate the Ligand accessible surface area (SAS) contribution with respect to the protein-ligand complex.
  """
  import subprocess; 
  temp = tempfile.NamedTemporaryFile(suffix=".dat");
  tempname = temp.name;
  pro_mask = GetProteinMask(pdbfile);

  cpptraj_str = f"parm {pdbfile}\ntrajin {pdbfile}\nsurf {lig_mask} solutemask {pro_mask},{lig_mask} out {tempname}";
  p1 = subprocess.Popen(["echo", "-e", cpptraj_str], stdout=subprocess.PIPE) ;
  p2 = subprocess.Popen(["cpptraj"], stdin=p1.stdout, stdout=subprocess.DEVNULL);
  p1.wait();
  p2.wait();

  with open(tempname, "r") as file1: 
    lines = [i.strip() for i in file1.read().strip("\n").split("\n") if i.strip()[0]!="#"]; 
    f_val = float(lines[0].split()[1]); 
  temp.close();
  return f_val

def ASALigOnly(pdbfile, lig_mask): 
  """
  Calculate the LIGAND accessible surface area (ASA) only. (Other components are not loaded)
  """
  traj = pt.load(pdbfile, top=pdbfile, mask=lig_mask)
  sel = traj.top.select(lig_mask)
  surf = pt.surf(traj, lig_mask)
  return float(surf.round(3)[0].item())

def EmbeddingFactor(basepath, pdbcode, mask=":LIG"):
  """
  Embedding factor is measured by the accessible surface area (ASA) contribution of ligand in a complex
  to the pure ligand ASA
  """
  import os; 
  pdbcode = pdbcode.lower(); 
  basepath = os.path.abspath(basepath); 
  ligfile = os.path.join(basepath, f"{pdbcode}/{pdbcode}_ligand.mol2");
  pdbfile = os.path.join(basepath, f"{pdbcode}/{pdbcode}_protein.pdb");
  outfile = os.path.join(basepath, f"{pdbcode}/{pdbcode}_complex.pdb");

  if (os.path.isfile(ligfile)) and (os.path.isfile(pdbfile)):
    outfile = combineMOL2PDB(ligfile, pdbfile, outfile)
    slig_0 = ASALig(outfile, mask)
    slig_1 = ASALigOnly(outfile, mask)
  elif not os.path.isfile(ligfile):
    print(f"Cannot find the ligand file in the database {pdbcode} ({ligfile})")
  elif not os.path.isfile(pdbfile):
    print(f"Cannot find the protein file in the database {pdbcode} ({pdbfile})")
  # print(f"Surface contribution: {slig_0}; Surface pure: {slig_1}")
  return 1-slig_0/slig_1


def fetch(code):
  import requests; 
  pdb = code.lower();
  response = requests.post(f'http://files.rcsb.org/download/{pdb}.pdb'); 
  return response.text

def PRO_nha(pdbfile):
  traj = pt.load(pdbfile, top=pdbfile)
  atomic_numbers = np.array([i.atomic_number for i in traj.top.atoms])
  nha = np.count_nonzero(atomic_numbers > 1)
  return nha

def PRO_nhydrogen(pdbfile):
  traj = pt.load(pdbfile, top=pdbfile)
  atomic_numbers = np.array([i.atomic_number for i in traj.top.atoms])
  nh = np.count_nonzero(atomic_numbers == 1)
  return nh

def LIG_nha(pdbfile, mask=":LIG"):
  traj = pt.load(pdbfile, top=pdbfile, mask=mask)
  atomic_numbers = np.array([i.atomic_number for i in traj.top.atoms])
  nha = np.count_nonzero(atomic_numbers > 1)
  return nha

def cgenff2dic(filename):
  with open(filename) as file1:
    lst = list(filter(lambda i: re.match(r"^ATOM.*!", i), file1))
  theatom  = [i.strip("\n").split()[1] for i in lst]
  atomtype = [i.strip("\n").split()[2] for i in lst]
  charge   = [float(i.strip("\n").split()[3]) for i in lst]
  penalty  = [float(i.strip("\n").split()[-1]) for i in lst]
  return {"name":theatom, "type":atomtype, "charge":charge, "penalty":penalty}

def cgenff2xmls(cgenffname):
  cgenffdic = cgenff2dic(cgenffname); 
  root = ET.Element('ForceField')
  info = ET.SubElement(root, 'Info')
  info_date = ET.SubElement(info, "date")
  info_date.text = datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S');
  data_lig = ET.SubElement(root, 'LIG')
  for i in range(len(cgenffdic["name"])):
    tmpattrib={
      "name":cgenffdic["name"][i], 
      "type": cgenffdic["type"][i], 
      "charge": str(cgenffdic["charge"][i]), 
      'penalty': str(cgenffdic["penalty"][i]),
    }
    tmpatom = ET.SubElement(data_lig, 'ATOM', attrib = tmpattrib)
  ligxml_str = ET.tostring(root , encoding="unicode")
  ligxml_str = minidom.parseString(ligxml_str).toprettyxml(indent="  ")
  return ligxml_str

def cgenff2xml(cgenffname, outfile):
  xmlstr = cgenff2xmls(cgenffname); 
  with open(outfile, "w") as file1: 
    file1.write(xmlstr)
  return 

def DistanceLigPro(theid, mode="session", ligname="LIG"):
  """
    Calculate the COM distance from protein to ligand
    Protein uses CA atoms ; Ligand use ALL atoms
    Have 2 modes: session/file
  """
  import pytraj as pt
  from . import session_prep
  if mode == "session":
    from rdkit import Chem
    from scipy.spatial import distance_matrix
    import re
    import numpy as np 
    if len(theid) != 8: 
      print("Session ID length not equil to 8");
      return
    with tempfile.NamedTemporaryFile("w", suffix=".pdb") as file1, tempfile.NamedTemporaryFile("w", suffix=".mol2") as file2:
      session = session_prep.RecallSession(theid)
      file1.write(session["pdbfile"]); 
      protcom = pt.center_of_mass(pt.load(file1.name), "@CA"); 
      try:
        # Mol2 could successfully be parsed in pytraj
        file2.write(session["molfile"]); 
        traj = pt.load(file2.name)
        ligcom = pt.center_of_mass(pt.load(file2.name)); 
      except Exception as e: 
        # Directly calculate the COM of the ligand 
        # print(f"Error occurred while calculating the Ligand COM: {e}")
        atoms = session["molfile"].split("@<TRIPOS>ATOM\n")[1].split("@<TRIPOS>")[0]; 
        atoms = [i.strip().split() for i in atoms.strip("\n").split("\n")]; 
        coord = np.array([i[2:5] for i in atoms]).astype(np.float32); 
        atomtypes = [re.sub(r"[0-9]", "", i[1]) for i in atoms]; 
        masses = []; 
        for i in atomtypes:
          try: 
            m = Chem.Atom(i).GetMass()
            masses.append(m); 
          except: 
            masses.append(0); 
        com = np.average(coord, axis=0, weights=masses)
        ligcom = np.array([com])
      return distance_matrix(ligcom, protcom).item(); 
  elif mode == "file": 
    traj = pt.load(theid); 
    dist = pt.distance(traj, f"@CA  :{ligname}")
    return dist.item()
  else:
    return None

def getSeqCoord(filename):
  """
    Extract residue CA <coordinates> and <sequence> from PDB chain
  """
  from Bio.PDB.Polypeptide import three_to_one
  import pytraj as pt

  traj = pt.load(filename)
  resnames = [i.name for i in traj.top.residues];
  trajxyz = traj.xyz[0];
  retxyz = [];
  retseq = "";
  for atom in traj.top.atoms:
    if atom.name == "CA":
      try:
        resname = resnames[atom.resid]
        resxyz = trajxyz[atom.index]
        retseq += three_to_one(resname)
        retxyz.append(resxyz)
      except:
        pass
  return np.array(retxyz), retseq


def CompareStructures(tokens, modes, url="http://130.60.168.149/fcgi-bin/ACyang.fcgi"):
  """
    Compare the PDB structure before and after then session preparation
    Functions: 
      Extract residue <coordinates> and <sequence> from PDB chain
        Uses the coordinates of the CA atom as the center of the residue
        Skip unknown residues
  """
  from tmtools import tm_align;
  from BetaPose.test import ACGUIKIT_REQUESTS
  if isinstance(tokens, list) and isinstance(modes, list):
    results = []
    for token, mode in zip(tokens, modes): 
      if mode == "traj":
        acg_kit = ACGUIKIT_REQUESTS(url); 
        acg_kit.recall(token[4:]); 
        pdbstr = acg_kit.recallTraj(token)["PDBFile"]; 
      elif mode == "str":
        pdbstr = token; 
      elif mode == "fetch":
        assert len(token) == 4, "PDB with length of 4";
        pdbstr = fetch(token); 
      elif mode == "file":
        with open(token, "r") as file1: 
          pdbstr = file1.read(); 
      elif mode == "session":
        acg_kit = ACGUIKIT_REQUESTS(url); 
        pdbstr = acg_kit.recall(token)["pdbfile"]; 
      if token == tokens[0]: 
        with tempfile.NamedTemporaryFile("w", suffix=".pdb") as file1:
          file1.write(pdbstr); 
          coord_ref, seq_ref = getSeqCoord(file1.name);
        continue; 
      else: 
        with tempfile.NamedTemporaryFile("w", suffix=".pdb") as file1:
          file1.write(pdbstr); 
          coord_i, seq_i = getSeqCoord(file1.name);
        
        result = tm_align(coord_ref, coord_i, seq_ref, seq_i)
        results.append(max([result.tm_norm_chain1, result.tm_norm_chain2])); 
        # print(f"CoorSet 1 {coord_ref.shape}:{result.tm_norm_chain1:.3f} ; CoorSet 2 {coord_i.shape}:{result.tm_norm_chain2:.3f}; ")
  else: 
    print("Please provide a list of PDB structure of interest")
  return results  

def getPdbTitle(pdbcode):
  pdb = pdbcode.lower().strip().replace(" ", ""); 
  assert len(pdb) == 4, "Please enter a valid PDB name";
  pdbstr = fetch(pdb);
  title = " ".join([i.strip("TITLE").strip() for i in pdbstr.split("\n") if "TITLE" in i]); 
  return title
  
  
  
def getPdbSeq(pdbcode):
  from Bio.SeqUtils import seq1; 
  import re
  pdb = pdbcode.lower().strip().replace(" ", ""); 
  assert len(pdb) == 4, "Please enter a valid PDB name";
  pdbstr = fetch(pdb);
  
  chainids = [i[11] for i in pdbstr.split("\n") if re.search(r"SEQRES.*[A-Z].*[0-9]", i)];
  chainid = chainids[0];
  title = " ".join([i[19:] for i in pdbstr.split("\n") if re.search(f"SEQRES.*{chainid}.*[0-9]", i)]); 
  seqstr = "".join(title.split());
  seqstr = seq1(seqstr); 
  if len(seqstr) > 4:
    return seqstr
  else: 
    print("Not found a proper single chain")
    title = " ".join([i[19:] for i in pdbstr.split("\n") if re.search(r"SEQRES", i)])
    seqstr = "".join(title.split());
    seqstr = seq1(seqstr); 
    return seqstr

def colorgradient(mapname, gradient, cmin=0.1, cmax=0.9): 
  import matplotlib.pyplot as plt
  cmap = plt.get_cmap(mapname);
  # Define N equally spaced values between 0.1 and 0.9
  values = np.linspace(cmin, cmax, gradient);
  # Get the RGB values for each of the 6 values
  colors = cmap(values); 
  return colors[:,:3].tolist()

def getAxisIndex(idx, colnr):
  x = np.floor(idx/colnr).astype(int); 
  y = idx%colnr ; 
  return (x,y)

def smartsSupplier(smarts):
  from rdkit import Chem
  mols = []; 
  for idx, m in enumerate(smarts): 
    mol = Chem.MolFromSmarts(m); 
    mols.append(mol); 
  return mols

def DrawGridMols(axes, mols, colnr):
  from rdkit.Chem import Draw
  for axis in axes.reshape((-1,1)):
    axis[0].axis("off"); 
  for idx, mol in enumerate(mols): 
    figi = Draw.MolToImage(mol); 
    figi.thumbnail((100, 100)); 
    index = getAxisIndex(idx, colnr); 
    axes[index].imshow(figi); 
    axes[index].set_title(f"SubStruct {idx+1}"); 

def getmask(traj, mask): 
  selected = traj.top.select(mask)
  selected_str = [f"{i+1}," for i in selected]
  finalmask = "@"+"".join(selected_str).strip(",")
  return finalmask

def getmaskbyidx(traj, idxs):
  idxs = np.array(idxs);
  aids = [i.index for i in np.array(list(traj.top.atoms))[idxs]]; 
  aids = list(set(aids)); 
  aids.sort(); 
  selected_str = [f"{i+1}," for i in aids]; 
  finalmask = "@"+"".join(selected_str).strip(","); 
  return finalmask

def getresmask(traj, mask):
  selected = traj.top.select(mask); 
  rids = [i.resid for i in np.array(list(traj.top.atoms))[selected]]; 
  rids = list(set(rids)); 
  selected_str = [f"{i+1}," for i in rids]; 
  finalmask = ":"+"".join(selected_str).strip(","); 
  return finalmask

def filter_points_within_bounding_box(thearr, grid_center, grid_length, return_state=False):
  """
  Filter the coordinates array by a bounding box
  Args:
    thearr: array of coordinates
    grid_center: center of the box
    grid_length: length(s) of the box
    return_state: return the acceptance of the array, otherwise return the coordinates array
  """
  thearr = np.asarray(thearr);
  upperbound = np.asarray(grid_center) + np.asarray(grid_length)/2;
  lowerbound = np.asarray(grid_center) - np.asarray(grid_length)/2;
  ubstate = np.array([np.prod(i) for i in thearr < upperbound]); 
  lbstate = np.array([np.prod(i) for i in thearr > lowerbound]); 
  state = [bool(i) for i in ubstate*lbstate]; 
  if return_state: 
    return state 
  else: 
    return thearr[state]

def coordfilter(coord, refcoord): 
  refcoord = [tuple(i) for i in np.array(refcoord).round(2)];
  _coord   = [tuple(i) for i in np.array(coord).round(2)];
  ret = [];
  for idx, c in enumerate(_coord): 
    if c in refcoord: 
      ret.append(coord[idx]);
  return np.array(ret)

def ordersegments(lst):
  from collections import Counter
  counter = Counter(lst);
  sorted_elements = sorted(counter, key=lambda x: counter[x], reverse=True);
  if 0 in sorted_elements:
    sorted_elements.remove(0);
  return sorted_elements

def NormalizePDB(refpdb, testpdb, outpdb):
  """
  Priority, output all of the protein part and prefereably keep the cofactors in the reference PDB
  There might be mismatches between the reference and test PDB file 
  """
  trajref = pt.load(refpdb)
  trajtest = pt.load(testpdb)

  ref_prot_atoms = trajref.top.select("@CA,C,N,O,:FOR,NME,ACE,NH2")
  ref_prot_res = np.array([i.resid for i in trajref.top.atoms])[ref_prot_atoms]
  other_parts = [i.name for i in trajref.top.residues][max(ref_prot_res)+1:]
  test_other_res = [i for i in trajtest.top.residues][max(ref_prot_res)+1:]
  other_indexes = []
  for i in test_other_res:
    if len(other_parts) > 0 and i.name == other_parts[0]:
      other_parts.pop(0)
      other_indexes += [i for i in range(i.first,i.last)]
    elif len(other_parts) == 0:
      break
  other_indexes = [i+1 for i in other_indexes]
  prot_part_index = [i for i in trajref.top.residues][max(ref_prot_res)].last
  all_indexes = [i+1 for i in range(prot_part_index)] + other_indexes
  finalstr = ''
  with open(testpdb, "r") as file1:
    raw =[i for i in file1.read().split("\n") if len(i) > 0]
    for i in raw:
      if "ATOM" in i or "HETATM" in i:
        residx = int(i[6:11].strip())
        if residx in all_indexes:
          finalstr += i+"\n"
      else:
        finalstr += i+"\n"
  with open(outpdb, 'w') as file1:
    file1.write(finalstr)

def transform_by_euler_angle(roll, pitch, yaw, translate=[0, 0, 0]):
    # Precompute trigonometric functions
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    # Generate rotation matrices
    Rx = np.array([[1, 0, 0], [0, cos_roll, -sin_roll], [0, sin_roll, cos_roll]])
    Ry = np.array([[cos_pitch, 0, sin_pitch], [0, 1, 0], [-sin_pitch, 0, cos_pitch]])
    Rz = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

    # Combine rotations
    # R = Rx @ Ry @ Rz
    R = Rz @ Ry @ Rx

    # Create the final transformation matrix
    H = np.eye(4); 
    H[:3, :3] = R; 
    H[:3, 3] = np.array(translate).ravel()
    return H


def transform_pcd(pcd, trans_mtx): 
  # Homogenize the point cloud (add a row of ones)
  homogeneous_pcd = np.hstack((pcd, np.ones((pcd.shape[0], 1))));
  # Apply the transformation matrix to the point cloud
  transformed_pcd = np.dot(homogeneous_pcd, trans_mtx.T);
  # Remove the homogeneous coordinate (last column)
  transformed_pcd = transformed_pcd[:, :3];
  return transformed_pcd

def MSD(arr):
  """
    Mean Spacing Deviation
  """
  return np.array(arr).std(axis=1).mean(); 
def MSCV(arr):
  """
    Mean Spacing Coefficient of Variation
  """
  std = np.array(arr).std(axis=1); 
  mean = np.array(arr).mean(axis=1); 
  mscv = (std/mean).mean()
  return min(mscv, 1); 




