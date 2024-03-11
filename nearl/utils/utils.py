import os, time, re, hashlib

import numpy as np 
import pytraj as pt
from scipy.spatial import distance_matrix

from .. import printit

def get_hash(theinput="", mode="md5"):
  """
  Get the hash value of a string or a list or a numpy array
  """
  if mode=="md5":
    hash_func = hashlib.md5
  elif mode=="sha1":
    hash_func = hashlib.sha1
  elif mode=="sha256":
    hash_func = hashlib.sha256
  elif mode=="sha512":
    hash_func = hashlib.sha512
  else:
    print("Warning: Not found a valid hash function (md5, sha1, sha256, sha512), use md5 by default.")
    hash_func = hashlib.md5
  if isinstance(theinput, str) and len(theinput)>0:
    return hash_func(bytes(theinput, "utf-8")).hexdigest()
  elif isinstance(theinput, (list, tuple)) and len(theinput)>0:
    arrstr = ",".join(np.asarray(theinput).astype(str))
    return hash_func(bytes(arrstr, "utf-8")).hexdigest()
  elif isinstance(theinput, np.ndarray) and len(theinput)>0:
    return hash_func(theinput.tobytes()).hexdigest()
  elif isinstance(theinput, bytes) and len(theinput)>0:
    return hash_func(theinput).hexdigest()
  elif len(theinput) == 0:
    return hash_func(bytes(time.perf_counter().__str__(), "utf-8")).hexdigest()
  else:
    printit("Warning: Not a valid input, should be (string, tuple, list, np.ndarray, bytes). Using time.perf_counter() by default.")
    return hash_func(bytes(time.perf_counter().__str__(), "utf-8")).hexdigest()

def closest_partner_indices(coord1, coord2):
  dist_matrix = distance_matrix(coord1, coord2)
  minindex = np.argmin(dist_matrix, axis=1)
  return minindex


def dist_caps(traj, mask1, mask2, use_mean=False, ref_frame=None):
  """
  Get the pairwise distance between two masks
  Usually they are the heavy atoms of ligand and protein within the pocket
  """
  if isinstance(mask1, str):
    selmask1 = traj.top.select(mask1)
  elif isinstance(mask1, (list, tuple, np.ndarray)):
    selmask1 = np.asarray(mask1)

  if isinstance(mask2, str):
    selmask2 = traj.top.select(mask2)
  elif isinstance(mask2, (list, tuple, np.ndarray)):
    selmask2 = np.asarray(mask2)

  if len(selmask1) == 0:
    print("No target atom selected, please check the mask")
    return None, None
  elif len(selmask2) == 0:
    print("No counter part atom selected, please check the mask")
    return None, None

  # Compute the distance matrix between the target and reference atoms
  if use_mean == True:
    frame_mean = np.mean(traj.xyz, axis=0)
    mask1_xyz = frame_mean[selmask1]
    mask2_xyz = frame_mean[selmask2]
    ref_frame = distance_matrix(mask1_xyz, mask2_xyz)
  else:
    mask1_xyz = traj.xyz[ref_frame if ref_frame is not None else 0][selmask1]
    mask2_xyz = traj.xyz[ref_frame if ref_frame is not None else 0][selmask2]
    ref_frame = distance_matrix(mask1_xyz, mask2_xyz)

  # Find closest atom pairs
  minindex = np.argmin(ref_frame, axis=1)
  selclosest = selmask2[minindex]

  # Compute the evolution of distance between closest-pairs
  # NOTE: Add 1 to pytraj index (Unlike Pytraj, in Cpptraj, Atom index starts from 1)
  distarr = np.zeros((len(selmask1), traj.n_frames))
  c = 0
  for i, j in zip(selmask1, selclosest):
    distancei = traj.xyz[:,i,:] - traj.xyz[:,j,:]
    distarr[c, :] = np.sqrt(np.sum(distancei**2, axis=1))
    c+=1

  pdist_info = {"atom_name_group1": [], "indices_group1": [], "atom_name_group2": [], "indices_group2": []}

  if hasattr(traj, "atoms"):
    # With the nearl.io.traj class
    atoms = traj.atoms
  else:
    atoms = [i for i in traj.top.atoms]
  for idx in selmask1:
    pdist_info["atom_name_group1"].append(atoms[idx].name)
    pdist_info["indices_group1"].append(atoms[idx].index)
  for idx in selclosest:
    pdist_info["atom_name_group2"].append(atoms[idx].name)
    pdist_info["indices_group2"].append(atoms[idx].index)
  return distarr, pdist_info


def fetch(code):
  from requests import post
  pdb = code.lower()
  response = post(f'http://files.rcsb.org/download/{pdb}.pdb')
  return response.text


def get_mask(traj, mask):
  selected = traj.top.select(mask)
  selected_str = [f"{i+1}," for i in selected]
  finalmask = "@"+"".join(selected_str).strip(",")
  return finalmask


def get_residue_mask(traj, mask):
  """
  Get the residue mask by the atom mask
  """
  selected = traj.top.select(mask)
  if hasattr(traj, "atoms"):
    # With the nearl.io.traj class
    rids = [i.resid for i in traj.atoms[selected]]
  else:
    rids = [i.resid for i in np.array(list(traj.top.atoms))[selected]]
  rids = list(set(rids))
  selected_str = [f"{i+1}," for i in rids]
  finalmask = ":"+"".join(selected_str).strip(",")
  return finalmask


def get_mask_by_idx(traj, idxs):
  """
  Get the mask by the index of atoms(start from 0)
  """
  idxs = np.array(idxs)
  aids = [i.index for i in np.array(list(traj.top.atoms))[idxs]]
  aids = list(set(aids))
  aids.sort()
  selected_str = [f"{i+1}," for i in aids]
  finalmask = "@"+"".join(selected_str).strip(",")
  return finalmask


def get_protein_mask(_structure):
  """
  Get the protein mask for a given structure
  Args:
    _structure: a string or a pytraj trajectory object
  Returns:
    mask: a string of the protein mask
  """
  reslst = []
  if isinstance(_structure, str) and os.path.isfile(_structure):
    traj = pt.load(_structure, top=_structure)
  elif hasattr(_structure, "top"):
    traj = _structure
  else:
    raise ValueError("Invalid input structure (must be a file path or a pytraj trajectory object)")

  for i in traj.top.atoms:
    if (i.name=="CA" or i.name=="N") and (i.resid+1 not in reslst):
      reslst.append(i.resid+1)
  mask = ":"+",".join([str(i) for i in reslst])
  return mask


def color_steps(mapname, steps:int, cmin:float=0.1, cmax:float=0.9):
  """
  Get a list of colors from a matplotlib colormap
  Args:
    mapname: name of the colormap (importable to matplotlib.pyplot.get_cmap function)
    steps: number of colors to get
    cmin: minimum value of the colormap
    cmax: maximum value of the colormap
  Returns:
    colors: a list of RGB colors
  """
  from matplotlib.pyplot import get_cmap
  cmap = get_cmap(mapname)
  values = np.linspace(cmin, cmax, steps)
  colors = cmap(values)
  return colors[:,:3].tolist()


def msd(arr):
  """
  Mean Spacing Deviation
  """
  return np.array(arr).std(axis=1).mean()


def mscv(arr):
  """
  Mean Spacing Coefficient of Variation
  """
  std = np.array(arr).std(axis=1)
  mean = np.array(arr).mean(axis=1)
  mscv = (std/mean).mean()
  return min(mscv, 1)


def filter_points_within_bounding_box(thearr, grid_center, grid_length, return_state=False):
  """
  Filter the coordinates array by a bounding box
  Args:
    thearr: array of coordinates
    grid_center: center of the box
    grid_length: length(s) of the box
    return_state: return the acceptance of the array, otherwise return the coordinates array
  """
  thearr = np.asarray(thearr)
  upperbound = np.asarray(grid_center) + np.asarray(grid_length)/2
  lowerbound = np.asarray(grid_center) - np.asarray(grid_length)/2
  state = np.all((thearr < upperbound) & (thearr > lowerbound), axis=1)
  if return_state:
    return state
  else:
    return thearr[state]


def _entropy(x):
  """
  Calculate the entropy of a list/array of values
  Args:
    x (list/array): list of values
  Returns:
    entropy (float): entropy of x
  """
  x_set = list(set([i for i in x]))
  if len(x) <= 1:
    return 0
  counts = np.zeros(len(x_set))
  for xi in x:
    counts[x_set.index(xi)] += 1
  probs = counts/len(x)
  entropy = -np.sum(probs * np.log2(probs))
  return entropy


def cosine_similarity(vec1, vec2):
  """
  Compute cosine similarity between vec1 and vec2
  Args:
    vec1 (numpy array): vector 1
    vec2 (numpy array): vector 2
  Returns:
    similarity (float): cosine similarity between vec1 and vec2
  """
  # Compute the dot product of vec1 and vec2
  dot_product = np.dot(vec1, vec2)

  # Compute the L2 norms (a.k.a. Euclidean norms) of vec1 and vec2
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  # Compute cosine similarity and return it
  # We add a small number to the denominator for numerical stability,
  # just in case both norms are zero.
  similarity = dot_product / (norm_vec1 * norm_vec2 + 1e-9)
  return similarity


def penalties(cosine_sim, ref_val, scale_factor=0.1, max_val=0.95):
  # scale_factor represents the variability of the label
  # max_val represents cutoff of penalty
  return (max_val - np.array(cosine_sim)) * ref_val * scale_factor


def smarts_supplier(smarts):
  """
  Generate a list of RDKit mol object from a list of SMARTS strings
  Args:
    smarts: a list of SMARTS strings
  Returns:
    mols: a list of RDKit mol objects
  """
  from rdkit import Chem
  mols = []
  for idx, m in enumerate(smarts):
    mol = Chem.MolFromSmarts(m)
    mols.append(mol)
  return mols

def color_scale(point_nr, cmap="inferno"):
  from matplotlib.pyplot import get_cmap
  color_map = get_cmap(cmap)
  return [color_map(i)[:3] for i in np.linspace(int(0.1 * color_map.N), int(0.9 * color_map.N), point_nr).astype(int)] 

def get_pdb_title(pdbcode):
  pdb = pdbcode.lower().strip().replace(" ", "")
  assert len(pdb) == 4, "Please enter a valid PDB name"
  pdbstr = fetch(pdb)
  title = " ".join([i.strip("TITLE").strip() for i in pdbstr.split("\n") if "TITLE" in i])
  return title


def get_pdb_seq(pdbcode):
  from Bio.SeqUtils import seq1
  pdb = pdbcode.lower().strip().replace(" ", "")
  assert len(pdb) == 4, "Please enter a valid PDB name"
  pdbstr = fetch(pdb)

  chainids = [i[11] for i in pdbstr.split("\n") if re.search(r"SEQRES.*[A-Z].*[0-9]", i)]
  chainid = chainids[0]
  title = " ".join([i[19:] for i in pdbstr.split("\n") if re.search(f"SEQRES.*{chainid}.*[0-9]", i)])
  seqstr = "".join(title.split())
  seqstr = seq1(seqstr)
  if len(seqstr) > 4:
    return seqstr
  else:
    print("Not found a proper single chain")
    title = " ".join([i[19:] for i in pdbstr.split("\n") if re.search(r"SEQRES", i)])
    seqstr = "".join(title.split())
    seqstr = seq1(seqstr)
    return seqstr


def data_from_fbag(fbag, feature_idx):
  data = []
  for frame_data in fbag:
    if isinstance(frame_data[feature_idx], list):
      data.append(frame_data[feature_idx])
    elif isinstance(frame_data[feature_idx], np.ndarray):
      data.append(frame_data[feature_idx].tolist())
    elif isinstance(frame_data[feature_idx], (int, float, np.float32, np.float64)):
      data.append([frame_data[feature_idx]])
  return data


def data_from_tbag(bag, feature_idx):
  data = []
  for traj_data in bag:
    data += data_from_fbag(traj_data, feature_idx)
  return data


def data_from_tbagresults(results, feature_idx):
  data = []
  for tbag in results:
    data += data_from_tbag(tbag, feature_idx)
  return np.array(data)


def data_from_fbagresults(results, feature_idx):
  data = []
  for fbag in results:
    data += data_from_fbag(fbag, feature_idx)
  try:
    data = np.array(data)
  except:
    data = np.array(data, dtype=object)
  return data


def conflict_factor(pdbfile, ligname, cutoff=5):
  VDWRADII = {
    '1': 1.1, '2': 1.4, '3': 1.82, '4': 1.53, '5': 1.92, '6': 1.7, '7': 1.55, '8': 1.52,
    '9': 1.47, '10': 1.54, '11': 2.27, '12': 1.73, '13': 1.84, '14': 2.1, '15': 1.8,
    '16': 1.8, '17': 1.75, '18': 1.88, '19': 2.75, '20': 2.31, '28': 1.63, '29': 1.4,
    '30': 1.39, '31': 1.87, '32': 2.11, '34': 1.9, '35': 1.85, '46': 1.63, '47': 1.72,
    '48': 1.58, '50': 2.17, '51': 2.06, '53': 1.98, '54': 2.16, '55': 3.43, '56': 2.68,
    '78': 1.75, '79': 1.66, '82': 2.02, '83': 2.07
  }
  traj = pt.load(pdbfile, top=pdbfile)
  traj.top.set_reference(traj[0])
  pocket_atoms = traj.top.select(f":{ligname}<:{cutoff}")
  atoms = np.array([*traj.top.atoms])[pocket_atoms]
  coords = traj.xyz[0][pocket_atoms]
  atomnr = len(pocket_atoms)
  cclash = 0
  ccontact = 0
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

    st = (ret < vdw_pairs - 1.25)[0]
    if np.count_nonzero(st) > 0:
      partatoms = np.array([*traj.top.atoms])[otheratoms][st]
      thisatom = np.array([*traj.top.atoms])[atoms[i].index]
      for part in partatoms:
        dist = distance_matrix([traj.xyz[0][part.index]], [traj.xyz[0][thisatom.index]])
        print(f"Found clash between: {thisatom.name}({thisatom.index}) and {part.name}({part.index}); Distance: {dist.squeeze().round(3)}")

  factor = 1 - ((cclash/2)/((ccontact/2)/atomnr))
  print(f"Clashing factor: {round(factor,3)}; Atom selected: {atomnr}; Contact number: {ccontact}; Clash number: {cclash}")
  return factor
