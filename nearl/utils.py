import os, time, re, hashlib, sys

import h5py, torch
import numpy as np 
import pytraj as pt
from scipy.spatial import distance_matrix
from itertools import groupby

from . import constants
from . import printit, config


def get_hash(theinput="", mode="md5"):
  """
  Get the hash value of a string or a list or a numpy array

  Parameters
  ----------
  theinput : str or list-like or numpy-like (optional)
    The input to hash
  mode : str, default="md5"
    The hash function to use (md5, sha1, sha256, sha512)

  Returns
  -------
  hash_value : str
    The hash value of the input
    
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


def get_timestamp(): 
  """
  Get the current timestamp to nano-second

  Returns
  -------
  timestamp : str
    The timestamp in the format of %y%m%d_%H%M%S_ns

  """
  return time.strftime("%y%m%d_%H%M%S", time.localtime()) + "_" + time.time_ns().__str__()[-4:]


def compute_pcdt(traj, mask1, mask2, use_mean=False, ref=None, return_info=False):
  """
  Compute the Pairwise Closest DisTance between two atom sets in a trajectory
  Usually they are the heavy atoms of ligand and protein within the pocket

  Parameters
  ----------
  traj : trajectory_like
    The trajectory object
  mask1 : str or list or numpy array
    The mask of the first atom set
  mask2 : str or list or numpy array
    The mask of the second atom set
  use_mean : bool, default=False
    Use the mean structure of the trajectory as the reference
  ref : int or Frame, default=None
    The reference frame for computing the distance
  return_info : bool, default=False
    Return the information of the atom pairs

  Returns
  -------
  distarr : numpy array
    The distance array between the closest pairs of atoms
  pdist_info : dict (optional, if return_info=True)
    The information of the atom pairs
  """
  if ref is not None and isinstance(ref, (int, float, np.int32, np.int64, np.float32, np.float64)):
    traj.top.set_reference(traj[int(ref)])
  elif ref is not None and isinstance(ref, pt.Frame):
    traj.top.set_reference(ref)

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
  elif isinstance(ref, pt.Frame):
    mask1_xyz = ref.xyz[selmask1]
    mask2_xyz = ref.xyz[selmask2]
    ref_frame = distance_matrix(mask1_xyz, mask2_xyz)
  elif isinstance(ref, (int, float, np.int32, np.int64, np.float32, np.float64)):
    mask1_xyz = traj.xyz[int(ref)][selmask1]
    mask2_xyz = traj.xyz[int(ref)][selmask2]
    ref_frame = distance_matrix(mask1_xyz, mask2_xyz)
  else:
    # Otherwise, use the first frame as the reference
    mask1_xyz = traj.xyz[0][selmask1]
    mask2_xyz = traj.xyz[0][selmask2]
    ref_frame = distance_matrix(mask1_xyz, mask2_xyz)

  # Find closest atom pairs
  minindex = np.argmin(ref_frame, axis=1)
  selclosest = selmask2[minindex]

  # Compute the evolution of distance between closest-pairs
  distarr = np.zeros((len(selmask1), traj.n_frames))
  for c, (i, j) in enumerate(zip(selmask1, selclosest)):
    distancei = traj.xyz[:,i,:] - traj.xyz[:,j,:]
    distarr[c, :] = np.sqrt(np.sum(distancei**2, axis=1))

  if return_info:
    # NOTE: Add 1 to pytraj index (Unlike Pytraj, in Cpptraj, Atom index starts from 1)
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
  else:
    return distarr

def plot_pcdt(array):
  """
  Plot the Pairwise Closest Distance of a trajectory, the array should be the output of compute_pcdt with X-axis as time and Y-axis as atomic index

  Parameters
  ----------
  array : numpy array
    The distance array between the closest pairs of atoms

  Returns
  -------
  fig : matplotlib figure
    The figure object
  ax : matplotlib axis
    The axis object
  """
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(1)
  c = ax.pcolormesh(array, cmap="inferno")
  # Add color bar to the plot 
  fig.colorbar(c, ax=ax)
  
  ax.set_title("PCDT")
  ax.set_xlabel("Time")
  ax.set_ylabel("Atomic index")
  plt.show()
  return fig, ax

def get_pdbcode(pdbcode): 
  """
  Correct the PDB code to a standard format (lowercase) and replace with the superceded PDB code

  Parameters
  ----------
  pdbcode : str
    The PDB code

  Returns
  -------
  pdbcode : str
    The corrected PDB code

  """
  pdbcode = pdbcode.lower()
  return constants.PDBCODE_SUPERCEDES.get(pdbcode, pdbcode)

def fetch(code):
  """
  Fetch the PDB file from the RCSB PDB database

  Parameters
  ----------
  code : str
    The PDB code

  Returns
  -------
  response.text : str
    The content of the PDB file

  """
  from requests import post
  pdb = get_pdbcode(code)
  response = post(f'http://files.rcsb.org/download/{pdb}.pdb')
  return response.text


def get_mask(traj, mask):
  """
  Get the per-atom mask (start with @) from an atbitrary mask

  Parameters
  ----------
  traj : trajectory_like or topology_like
    The trajectory object or the topology object
  mask : str
    The mask of the atoms

  Returns
  -------
  finalmask : str
    The per-atom mask
  """
  if isinstance(traj, pt.Trajectory):
    selected = traj.top.select(mask)
  elif isinstance(traj, pt.Topology):
    selected = traj.select(mask)
  selected = [f"{i+1}," for i in selected]
  finalmask = "@"+"".join(selected).strip(",")
  return finalmask


def get_residue_mask(traj, mask):
  """
  Get the residue mask by the mask
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

def filter_points_within_bounding_box(thearr, grid_center, grid_length, return_coord=False):
  """
  Filter the coordinates array by a bounding box
  Parameters
  ----------
  thearr: array_like
    The array of coordinates
  grid_center: array_like
    The center of the box
  grid_length: array_like
    The length of the box
  return_coord: bool, default=False
    Return the coordinates within the box if True, otherwise return the boolean mask
    
  Returns
  -------
  thearr: array_like
    The filtered coordinates or the boolean mask
  """
  thearr = np.asarray(thearr)
  upperbound = np.asarray(grid_center) + np.asarray(grid_length)/2
  lowerbound = np.asarray(grid_center) - np.asarray(grid_length)/2
  state = np.all((thearr < upperbound) & (thearr > lowerbound), axis=1)
  if return_coord:
    return thearr[state]
  else:
    return state


def penalties(cosine_sim, ref_val, scale_factor=0.1, max_val=0.95):
  # scale_factor represents the variability of the label
  # max_val represents cutoff of penalty
  return (max_val - np.array(cosine_sim)) * ref_val * scale_factor


def generate_gridcoords(thecenter, thelengths, thedims):
  """
  Generate the 3D coordinates of a grid

  Parameters
  ----------
  thecenter: array_like
    The center of the grid
  thelengths: array_like
    The lengths of the grid
  thedims: array_like
    The dimensions of the grid

  Returns
  -------
  thegrid: tuple
    The grid coordinates
  thecoord: array_like
    The flattened coordinates

  """
  thegrid = np.meshgrid(
    np.arange(thedims[0]), np.arange(thedims[1]), np.arange(thedims[2]),
    indexing='ij'
  )
  thecoord = np.column_stack([thegrid[0].ravel(), thegrid[1].ravel(), thegrid[2].ravel()])
  thecoord = thecoord * thelengths / thedims + thecenter - thelengths / 2
  return thegrid, thecoord


def get_pdb_title(pdbcode):
  """
  Get the title of a PDB entry from the RCSB PDB database

  Parameters
  ----------
  pdbcode : str
    The PDB code

  Returns
  -------
  title : str
    The title of the PDB entry
  """
  pdb = get_pdbcode(pdbcode)
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


def conflict_factor(pdbfile, ligname, cutoff=5):
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
    vdw_pairs = np.array([constants.VDWRADII[i.atomic_number] for i in np.array([*traj.top.atoms])[otheratoms]]) + constants.VDWRADII[thisatom]
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


def append_hdf_data(hdffile, key, data, dtype, maxshape, **kwargs):
  """
  Append data to an existing HDF5 file

  Parameters
  ----------
  hdffile : str
    The path to the HDF5 file
  key : str
    The key to the dataset
  data : array_like
    The data to append
  dtype : str
    The data type of the dataset
  maxshape : tuple
    The maximum shape of the dataset
  kwargs : dict
    Additional keyword arguments for creating the dataset (If the dataset does not exist yet)

  """
  with h5py.File(hdffile, "a") as hdf: 
    if key in hdf: 
      dset = hdf[key]
      current_shape = dset.shape
      # Calculate the new shape after appending the new data
      new_shape = (current_shape[0] + data.shape[0], *current_shape[1:])
      dset.resize(new_shape)
      dset[current_shape[0]:new_shape[0]] = data
    else: 
      dset = hdf.create_dataset(key, data.shape, dtype=dtype, maxshape=maxshape, **kwargs)
      dset[:] = data


def update_hdf_data(hdffile, dataset_name:str, data:np.ndarray, hdf_slice, **kwargs):
  """
  Update the data of an existing HDF5 dataset with a slice of data

  The slice has to be explicitly defined by a normal slice object or np.s_[], so that it modifies the data in place

  Parameters
  ----------
  hdffile : str or h5py.File
    The path to the HDF5 file or the h5py.File object
  dataset_name : str
    The name of the dataset
  data : array_like
    The data to update
  hdf_slice : slice
    The slice of the dataset
  kwargs : dict
    Additional keyword arguments for creating the dataset (If the dataset does not exist yet)

  """
  if isinstance(hdffile, str):
    with h5py.File(hdffile, "a") as hdf:
      update_hdf_data(hdf, dataset_name, data, hdf_slice, **kwargs)
      # if dataset_name not in hdf.keys():
      #   hdf.create_dataset(dataset_name, data=data, **kwargs)
      # else:
      #   if hdf_slice.stop > hdf[dataset_name].shape[0]:
      #     hdf[dataset_name].resize(hdf_slice.stop, axis=0)
      #   hdf[dataset_name][hdf_slice] = data
  elif isinstance(hdffile, h5py.File):
    if dataset_name not in hdffile.keys():
        hdffile.create_dataset(dataset_name, data=data, **kwargs)
    else:
      if hdf_slice.stop > hdffile[dataset_name].shape[0]: 
        hdffile[dataset_name].resize(hdf_slice.stop, axis=0)
      hdffile[dataset_name][hdf_slice] = data


def dump_dict(outfile : str, groupname : str, dic : dict): 
  with h5py.File(outfile, "a") as hdf: 
    if groupname not in hdf.keys():
      hdf.create_group(groupname)
    else: 
      del hdf[groupname]
      hdf.create_group(groupname)
    for key, val in dic.items():
      hdf[groupname][key] = val


def find_block_single(traj, key): 
  """
  Find the single-residue blocks in a trajectory based on the keyword (residue name)

  Parameters
  ----------
  traj : trajectory_like
    The trajectory object
  key : str
    The residue name to search for
  
  Returns
  -------
  results : list
    The list of slices of each block
  """
  results = [] 
  for this_res in traj.top.residues: 
    if this_res.name != key:
      continue
    else:
      thisslice = np.s_[this_res.first_atom_index:this_res.last_atom_index]
      results.append(thisslice)
  return results


def find_block_dual(traj, key): 
  """
  Find the dual-residue blocks in a trajectory based on the keyword (residue name)

  Parameters
  ----------
  traj : trajectory_like
    The trajectory object
  key : str
    The residue name to search for

  Returns
  -------
  results : list
    The list of slices of each block (by atom)
  """
  iterable = traj.top.residues
  results = [] 
  previous_res = None 
  PIPTIDE_BOND_THRESHOLD = 1.32*1.25
  coord = traj.xyz[0]
  while True:
    try:
      this_res = next(iterable)
      if previous_res is not None: 
        # Check if the two residues are connected
        if f"{previous_res.name}{this_res.name}" != key:
          previous_res = this_res
          continue
        else: 
          top_res1 = traj.top[previous_res.first_atom_index:previous_res.last_atom_index]
          crd_res1 = coord[previous_res.first_atom_index:previous_res.last_atom_index, :]
          top_res2 = traj.top[this_res.first_atom_index:this_res.last_atom_index]
          crd_res2 = coord[this_res.first_atom_index:this_res.last_atom_index, :]
          atomc = top_res1.select("@C")
          atomn = top_res2.select("@N")
          
          if (previous_res.name not in constants.RES+[i for i in constants.RES_PATCH.keys()]) or (this_res.name not in constants.RES+[i for i in constants.RES_PATCH.keys()]):
            print(f"Residue pair {previous_res.name}-{this_res.name} is not supported yet", file=sys.stderr)
            previous_res = this_res
            continue

          if len(atomc) == 0 or len(atomn) == 0:
            previous_res = this_res
            continue
          elif len(atomc) > 1 or len(atomn) > 1:
            print("Warning: More than one C or N atom found in the residue, meaning there are some problem with the input structure.")
            previous_res = this_res
            continue
          else:
            peptide_bond_len = np.linalg.norm(crd_res1[atomc[0]] - crd_res2[atomn[0]])
            if peptide_bond_len > PIPTIDE_BOND_THRESHOLD:
              previous_res = this_res
              continue
            theslice = np.s_[previous_res.first_atom_index:this_res.last_atom_index]
            results.append(theslice)
            # print(f"Connected: {previous_res} -> {this_res}")
            previous_res = this_res 
      else:
        previous_res = this_res
    except StopIteration:
      break
  return results


def index_partition(input_seq, partition_nr): 
  """
  Return segments (consecutive True elements) computed from boolean mask ordered from largest to smallest segment

  Parameters
  ----------
  input_seq : array_like
    The input sequence of boolean mask
  partition_nr : int
    The number of partitions to return

  Returns
  -------
  ret_slices : list
    The list of slices of each segment
  """
  cidx = 0
  slices = []
  element_nrs = []
  for k, g in groupby(input_seq):
    lst = list(g)
    if k == True:
      element_nrs.append(len(lst))
      slices.append(np.s_[cidx:cidx+len(lst)])
    cidx += len(lst)

  ret_slices = [np.s_[0:0]] * partition_nr
  for c, i in enumerate(np.argsort(element_nrs)[::-1][:partition_nr]):
    ret_slices[c] = slices[i]
  return ret_slices 


def check_filelist(training_set): 
  with open(training_set, "r") as f:
    pdbcodes = f.read().strip("\n").split("\n")
  return pdbcodes


# Available models:
# Atom3D, DeepRank, Gnina2017, Gnina2017_, GninaDense, Gnina2018, KDeep, VoxNet, Pafnucy
def get_model(model_type:str, input_dim:int, output_dim:int, box_size, **kwargs):
  """
  A quick wrapper to get the model object

  Parameters
  ----------
  name : str
    The name of the model
  in_channels : int
    The number of input channels
  out_channels : int
    The number of output channels
  box_size : int
    The size of the box
  kwargs : dict
    Additional keyword arguments for the model

  Returns
  -------
  model : nn.Module
    The model object

  """
  # def get_model(model_type:str, input_dim:int, output_dim:int, box_size=None, **kwargs): 
  channel_nr = input_dim
  if model_type == "voxnet": 
    import nearl.models.model_voxnet
    voxnet_parms = {
      "input_channels": channel_nr,
      "output_dimension": output_dim,
      "input_shape": box_size,
      "dropout_rates" : [0.2, 0.3, 0.4],   # 0.25, 0.25, 0.25
    }
    return nearl.models.model_voxnet.VoxNet(**voxnet_parms)

  elif model_type == "deeprank":
    import nearl.models.model_deeprank
    return nearl.models.model_deeprank.DeepRankNetwork(channel_nr, output_dim, box_size)

  elif model_type == "gnina2017":
    import nearl.models.model_gnina
    return nearl.models.model_gnina.GninaNetwork2017(channel_nr, output_dim, box_size)

  elif model_type == "gnina2018":
    import nearl.models.model_gnina
    return nearl.models.model_gnina.GninaNetwork2018(channel_nr, output_dim, box_size)

  elif model_type == "kdeep":
    import nearl.models.model_kdeep
    return nearl.models.model_kdeep.KDeepNetwork(channel_nr, output_dim, box_size)

  elif model_type == "pafnucy":
    import nearl.models.model_pafnucy
    return nearl.models.model_pafnucy.PafnucyNetwork(channel_nr, output_dim, box_size, **kwargs)

  elif model_type == "atom3d":
    import nearl.models.model_atom3d
    return nearl.models.model_atom3d.Atom3DNetwork(channel_nr, output_dim, box_size)

  elif model_type == "resnet3d":
    import nearl.models.model_resnet3d
    return nearl.models.model_resnet3d.ResNet3D(channel_nr, output_dim, box_size, **kwargs)

  elif model_type == "resnet":
    from feater.models.resnet import ResNet
    model = ResNet(channel_nr, output_dim, "resnet18")
    return model

  elif model_type == "convnext_iso":
    from feater.models.convnext import ConvNeXtIsotropic
    model = ConvNeXtIsotropic(in_chans = channel_nr, num_classes=output_dim)
    return model

  elif model_type == "ViT":
    from transformers import ViTConfig, ViTForImageClassification
    configuration = ViTConfig(
      image_size = 128, 
      num_channels = channel_nr, 
      num_labels = output_dim, 
      window_size=4, 
    )
    model = ViTForImageClassification(configuration)
    return model

  else: 
    raise ValueError(f"Model type {model_type} is not supported")


def test_model(model, dataset, criterion, test_number, batch_size, use_cuda=1, process_nr=24): 
  tested_sample_nr = 0
  predictions = []
  targets = []
  losses = []

  with torch.no_grad():
    model.eval()
    for data, target in dataset.mini_batches(batch_size=batch_size, process_nr=process_nr):
      if tested_sample_nr >= test_number:
        break
      if use_cuda:
        data, target = data.cuda(), target.cuda()
      output = model(data)
      if "logits" in dir(output): 
        output = output.logits
      loss = criterion(output, target)
      tested_sample_nr += len(data)

      if output.shape[1] > 1: 
        pred_choice = torch.argmax(output, dim=1)
        correct = pred_choice.eq(target.data)
        predictions.append(correct)
      else: 
        predictions.append(output)
      
      losses.append(loss.item())
      targets.append(target)

  if config.verbose or config.debug: 
    printit(f"Tested {tested_sample_nr} samples")
  predictions = torch.cat(predictions, dim=0).cpu().numpy()
  targets = torch.cat(targets, dim=0).cpu().numpy()
  losses = np.array(losses)
  return predictions, targets, losses
