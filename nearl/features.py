import subprocess
import numpy as np

from rdkit import Chem
from scipy.spatial import KDTree
from open3d.pipelines.registration import compute_fpfh_feature
from open3d.geometry import KDTreeSearchParamHybrid

import feater

from .. import utils, commands

from .. import printit, draw_call_stack
from .. import _usegpu, _verbose

__all__ = [
  # Base class
  "Feature",
]

"""
When writing the customised feature, the following variables are automatically available:
  self.featurizer: The featurizer object
  self.top: The topology of the trajectory
  self.active_frame: The active frame of the trajectory
  self.grid: The grid for the feature
  self.center: The center of the box
  self.lengths: The lengths of the box
  self.dims: The dimensions of the box
  self.contents: The segmented moieties (mainly structural information from the fingerprint generator)
  

When registering the customised feature, the following needs to be defined by the user:
  self.featurize: The function to generate the feature from the trajectory

! IMPORTANT: 
  If there is difference from trajectory to trajectory, for example the name of the ligand, feature object has to behave 
  differently. However, the featurizer statically pipelines the feature.featurize() function and difference is not known
  in these featurization process. 
  This change has to take place in the featurizer object because the trajectories are registered into the featurizer. Add 
  the variable attribute to the featurizer object and use it in the re-organized feature.featurize() function.
    
"""

def crop(points, upperbound, padding):
  """
  Crop the points to the box defined by the center and lengths
  The mask is returned as a boolean array
  """
  # X within the bouding box
  x_state_0 = points[:, 0] < upperbound[0] + padding
  x_state_1 = points[:, 0] > 0 - padding
  # Y within the bouding box
  y_state_0 = points[:, 1] < upperbound[1] + padding
  y_state_1 = points[:, 1] > 0 - padding
  # Z within the bouding box
  z_state_0 = points[:, 2] < upperbound[2] + padding
  z_state_1 = points[:, 2] > 0 - padding
  # All states
  mask_inbox = x_state_0 * x_state_1 * y_state_0 * y_state_1 * z_state_0 * z_state_1
  return mask_inbox


class Feature:
  def __init__(self):
    self.__dims = None
    self.__spacing = None
    self.__center = None
    self.__lengths = None

  def __str__(self):
    ret_str = "Feature: "
    ret_str += "Dimensions: " + " ".join([str(i) for i in self.dims] + " ")
    ret_str += f"Spacing: {self.spacing} \n"
    return ret_str

  @property
  def dims(self):
    return self.__dims
  @dims.setter
  def dims(self, value):
    if isinstance(value, (int, float, np.int64, np.float64, np.int32, np.float32)): 
      self.__dims = np.array([int(value), int(value), int(value)])
    elif isinstance(value, (list, tuple, np.ndarray)):
      self.__dims = np.array([int(i) for i in value][:3])
    else:
      raise ValueError("The dimensions should be an integer or a list of integers")
    if self.__spacing is not None:
      self.__center = np.array(self.__dims * self.spacing, dtype=np.float32) / 2
    if self.__dims is not None and self.__spacing is not None:
      self.__lengths = self.__dims * self.__spacing

  @property
  def spacing(self):
    return self.__spacing
  @spacing.setter
  def spacing(self, value):
    self.__spacing = float(value)
    if self.__dims is not None:
      self.__center = np.array(self.__dims * self.spacing, dtype=np.float32) / 2
    if self.__dims is not None and self.__spacing is not None:
      self.__lengths = self.__dims * self.__spacing

  @property
  def center(self):
    return self.__center
  @property
  def lengths(self):
    return self.__lengths

  def hook(self, featurizer): 
    """
      Hook the feature generator back to the feature convolutor and obtain necessary attributes from the featurizer
      including the trajectory, active frame, convolution kernel etc
    """
    self.dims = featurizer.dims
    self.spacing = featurizer.spacing

  def cache(self, trajectory): 
    atoms = [i for i in trajectory.top.atoms]
    self.resids = np.array([i.resid for i in atoms])
    self.atomic_numbers = np.array([i.atomic_number for i in atoms])

  def query(self, topology, frame_coords, focal_point, byres=True):
    new_coords = frame_coords - focal_point + self.center 
    # No padding 
    mask = crop(new_coords, self.lengths, 0)
    # Get the boolean array of residues within the bounding box
    if byres:
      res_inbox = np.unique(self.resids[mask])
      final_mask = np.full(len(self.resids), False)
      for res in res_inbox:
        final_mask[np.where(self.resids == res)] = True
    else: 
      final_mask = mask
    return final_mask

  def run(self):
    pass

  def dump(self):
    pass
  


class MassStatic(Feature):
  """
    Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
    Parse of the Mask should not be in here, the input should be focal points in coordinates format
    Explicitly pass the cutoff and sigma while initializing the Feature object for the time being
    Atomic mass as a feature
  """
  def __init__(self, cutoff=None, sigma=None, outfile=None):
    super().__init__()
    self.cutoff = cutoff
    self.sigma = sigma
    self.outfile = outfile

  def hook(self, featurizer):
    super().hook(featurizer)
    if self.sigma is None:
      # In case the sigma is not set, inherit the sigma from the featurizer
      if "sigma" in dir(featurizer) and featurizer.sigma is not None:
        print(f"{self.__class__.__name__}: Inheriting the sigma from the featurizer: {featurizer.sigma}")
        self.sigma = featurizer.sigma
      else:
        print(f"{self.__class__.__name__}: Setting the sigma to {0.5 * self.spacing}")
        self.sigma = 0.5 * self.spacing
    if self.cutoff is None:
      if "cutoff" in dir(featurizer) and featurizer.cutoff is not None:
        print(f"{self.__class__.__name__}: Inheriting the cutoff from the featurizer: {featurizer.cutoff}")
        self.cutoff = featurizer.cutoff
      else: 
        print(f"{self.__class__.__name__}: Setting the cutoff to {2 * self.spacing}")
        self.cutoff = 2 * self.spacing

  def query(self, topology, frame_coords, focal_point): 
    """
      Get the atoms and weights within the bounding box
    """
    # TODO: The frame_coords should be a 2D array not based on multiple frames
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    # frame_coords = frame_coords - focal_point + self.center
    # print("Max/Min: ", np.max(frame_coords, axis=0), np.min(frame_coords, axis=0))
    # print("Mean: ", np.mean(frame_coords, axis=0))
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.atomic_numbers[idx_inbox]
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights

  def run(self, coords, weights): 
    """
      Voxelization of the atomic mass
      # Host-py function: voxelize_coords(coords, weights, grid_dims, spacing, cutoff, sigma):
    """
    if len(coords) == 0:
      printit(f"{self.__class__.__name__}: Warning: The coordinates are empty")
      return np.zeros(self.dims, dtype=np.float32)
    print(weights.tolist())
    printit(f"{self.__class__.__name__}: Center of the coordinates: {np.mean(coords, axis=0)}")
    ret = commands.voxelize_coords(coords, weights, self.dims, self.spacing, self.cutoff, self.sigma)
    printit(f"{self.__class__.__name__}: The sum of the returned array: {np.sum(ret)} VS {np.sum(weights)} from {len(weights)} atoms")
    return ret
  
  def dump(self, array):
    if self.outfile is not None:
      np.save(self.outfile, array)
    

class HeavyAtom(Feature):
  def __init__(self, cutoff=None, sigma=None, default_weight=1, outfile=None):
    super().__init__()
    self.cutoff = cutoff
    self.sigma = sigma
    self.outfile = outfile
    self.default_weight = default_weight

  def hook(self, featurizer):
    super().hook(featurizer)
    if self.sigma is None:
      # In case the sigma is not set, inherit the sigma from the featurizer
      if "sigma" in dir(featurizer) and featurizer.sigma is not None:
        print(f"{self.__class__.__name__}: Inheriting the sigma from the featurizer: {featurizer.sigma}")
        self.sigma = featurizer.sigma
      else:
        print(f"{self.__class__.__name__}: Setting the sigma to {0.5 * self.spacing}")
        self.sigma = 0.5 * self.spacing
    if self.cutoff is None:
      if "cutoff" in dir(featurizer) and featurizer.cutoff is not None:
        print(f"{self.__class__.__name__}: Inheriting the cutoff from the featurizer: {featurizer.cutoff}")
        self.cutoff = featurizer.cutoff
      else: 
        print(f"{self.__class__.__name__}: Setting the cutoff to {2 * self.spacing}")
        self.cutoff = 2 * self.spacing

  def cache(self, trajectory):
    super().cache(trajectory)
    # Prepare the heavy atom weights
    self.heavy_atoms = np.full(len(self.resids), 0, dtype=np.float32)
    self.heavy_atoms[np.where(self.atomic_numbers > 1)] = self.default_weight
    return 

  def query(self, topology, frame_coords, focal_point): 
    """
      Get the atoms and weights within the bounding box
    """
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    # frame_coords = frame_coords - focal_point + self.center

    # print("Max/Min: ", np.max(frame_coords, axis=0), np.min(frame_coords, axis=0))

    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.heavy_atoms[idx_inbox]
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights

  def run(self, coords, weights): 
    """
      Voxelization of the atomic mass
      # Host-py function: voxelize_coords(coords, weights, grid_dims, spacing, cutoff, sigma):
    """
    
    printit(f"{self.__class__.__name__}: Center of the coordinates: {np.mean(coords, axis=0)}")
    ret = commands.voxelize_coords(coords, weights, self.dims, self.spacing, self.cutoff, self.sigma)
    printit(f"{self.__class__.__name__}: The sum of the returned array: {np.sum(ret)} VS {np.sum(weights)} from {len(weights)} atoms")
    return ret
  
  def dump(self, array):
    if self.outfile is not None:
      np.save(self.outfile, array)



class PartialCharge(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Atomic charge feature for the structure of interest;
  Compute the charge based on the self.featurizer.boxed_pdb;
  """
  def __init__(self, mask="*", value=None):
    super().__init__()
    self.MASK = mask          # moiety of interest
    if value is not None:
      self.mode = "manual"
      self.charge_values = [i for i in value]
    else:
      self.mode = "gasteiger"
      self.charge_values = []
    self.ATOM_INDICES = np.array([])

  def before_frame(self):
    self.ATOM_INDICES = self.traj.top.select(self.MASK)
    if self.mode == "gasteiger":
      retmol = self.query_mol(self.ATOM_INDICES)
      if retmol is not None:
        self.charge_values = np.array([float(atom.GetProp("_GasteigerCharge")) for atom in retmol.GetAtoms()]).astype(float)
        self.charge_values = self.charge_values[:len(self.ATOM_INDICES)]
      else:
        self.charge_values = np.zeros(len(self.ATOM_INDICES)).astype(float)
    elif self.mode == "manual":
      self.charge_values = np.asarray(self.charge_values).astype(float)

    if len(self.charge_values) != len(self.ATOM_INDICES):
      printit("Warning: The number of atoms in PDB does not match the number of charge values")

  def featurize(self):
    """
    NOTE:
    The self.boxed_pdb is already cropped and atom are reindexed in the PDB block.
    Hence use the self.boxed_indices to get the original atom indices standing for the PDB block
    """
    # Get the atoms within the bounding box
    coord_candidates = self.active_frame.xyz[self.ATOM_INDICES]
    mask_inbox = self.crop_box(coord_candidates)
    coords = coord_candidates[mask_inbox]
    weights = self.charge_values[mask_inbox]
    feature_charge = self.interpolate(coords, weights)
    return feature_charge


class AM1BCCCharge(Feature):
  def __init__(self, moi="*", mode="auto", onetime=False):
    super().__init__()
    """AM1-BCC charges computed by antechamber for ligand molecules only"""
    self.cmd_template = ""    # command template for external charge computation programs
    self.computed = False     # whether self.charge_values is computed or not
    self.mode = "am1bcc"
    self.onetime = onetime
    self.cmd_template = "am1bcc -i LIGFILE -f ac -o OUTFILE -j 5"

  def featureize(self):
    if (not self.computed) or (not self.onetime):
      # run the am1bcc program
      self.charge_values = np.array(self.mode)
      cmd_final = self.cmd_template.replace("LIGFILE", self.featurizer.ligfile).replace("OUTFILE", self.featurizer.outfile)
      subprocess.run(cmd_final.split(), check=True)
      # TODO: Try out this function and read the output file into charge values
      self.charge_values = np.loadtxt(self.featurizer.outfile)
      self.computed = True
    rdmol = Chem.MolFromPDBBlock(self.featurizer.boxed_pdb)
    coords_pdb = np.array([i for i in rdmol.GetConformer().GetPositions()])
    if len(coords_pdb) != len(self.charge_values):
      printit("Warning: The number of atoms in PDB does not match the number of charge values")
    # Get the atoms within the bounding box
    upperbound = self.center + self.lengths / 2
    lowerbound = self.center - self.lengths / 2
    ubstate = np.all(coords_pdb < upperbound, axis=1)
    lbstate = np.all(coords_pdb > lowerbound, axis=1)
    mask_inbox = ubstate * lbstate

    coords = coords_pdb[mask_inbox]
    weights = charge_array[mask_inbox]
    feature_charge = self.interpolate(coords, weights)
    return feature_charge


class AtomTypeFeature(Feature):
  def __init__(self, aoi="*"):
    super().__init__()
    self.aoi = aoi

  def before_frame(self):
    if self.traj.top.select(self.aoi) == 0:
      raise ValueError("No atoms selected for atom type calculation")

  def featurize(self):
    pass


class HydrophobicityFeature(Feature):
  def __int__(self):
    super().__init__()

  def featurize(self):
    pass


class Aromaticity(Feature):
  def __init__(self, mask="*"):
    super().__init__()
    self.MASK = mask
    self.ATOM_INDICES = np.array([])
    self.AROMATICITY = np.array([])

  def before_frame(self):
    """
    Compute the aromaticity of each atom in the moiety of interest
    The atomic index matches between the
    """
    self.ATOM_INDICES = self.traj.top.select(self.MASK)
    retmol = self.query_mol(self.ATOM_INDICES)
    if retmol is not None:
      self.AROMATICITY = np.array([atom.GetIsAromatic() for atom in retmol.GetAtoms()]).astype(int)
      self.AROMATICITY = self.AROMATICITY[:len(self.ATOM_INDICES)]
    else:
      self.AROMATICITY = np.zeros(len(self.ATOM_INDICES)).astype(int)

  def featurize(self):
    # Interpolate the feature values to the grid points
    coord_candidate = self.active_frame.xyz[self.ATOM_INDICES]
    mask_inbox = self.crop_box(coord_candidate)
    coords = coord_candidate[mask_inbox]
    weights = self.AROMATICITY[mask_inbox]
    feature_arr = self.interpolate(coords, weights)
    return feature_arr





class Ring(Feature):
  def __init__(self, mask="*"):
    super().__init__()
    self.MASK = mask
    self.ATOM_INDICES = np.array([])
    self.RING = np.array([])

  def before_frame(self):
    self.ATOM_INDICES = self.traj.top.select(self.MASK)
    retmol = self.query_mol(self.ATOM_INDICES)
    if retmol is not None:
      self.RING = np.array([atom.IsInRing() for atom in retmol.GetAtoms()])
      self.RING = self.RING[:len(self.ATOM_INDICES)].astype(int)
    else:
      self.RING = np.zeros(len(self.ATOM_INDICES)).astype(int)
    if len(self.RING) != len(self.ATOM_INDICES):
      printit("Warning: The number of atoms in PDB does not match the number of aromaticity values")

  def featurize(self):
    coord_candidates = self.active_frame.xyz[self.ATOM_INDICES]
    mask_inbox = self.crop_box(coord_candidates)
    coords = coord_candidates[mask_inbox]
    weights = self.RING[mask_inbox]
    feature_arr = self.interpolate(coords, weights)
    return feature_arr


class Hybridization(Feature):
  def __init__(self, mask="*"):
    super().__init__()
    self.MASK = mask
    self.HYBRIDIZATION_DICT = {'SP': 1, 'SP2': 2, 'SP3': 3, "UNSPECIFIED": 0}
    self.ATOM_INDICES = np.array([])
    self.HYBRIDIZATION = np.array([])

  def before_frame(self):
    self.ATOM_INDICES = self.traj.top.select(self.MASK)
    retmol = self.query_mol(self.ATOM_INDICES)
    if retmol is not None:
      self.HYBRIDIZATION = [atom.GetHybridization() for atom in retmol.GetAtoms()]
      self.HYBRIDIZATION = np.array([self.HYBRIDIZATION_DICT[i] if i in self.HYBRIDIZATION_DICT else 0 for i in self.HYBRIDIZATION]).astype(int)
      self.HYBRIDIZATION = self.HYBRIDIZATION[:len(self.ATOM_INDICES)]
    else:
      self.HYBRIDIZATION = np.zeros(len(self.ATOM_INDICES)).astype(int)
    if len(self.HYBRIDIZATION) != len(self.ATOM_INDICES):
      printit("Warning: The number of atoms in PDB does not match the number of HYBRIDIZATION values")

  def featurize(self):
    coord_candidates = self.active_frame.xyz[self.ATOM_INDICES]
    mask_inbox = self.crop_box(coord_candidates)
    coords = coord_candidates[mask_inbox]
    weights = self.HYBRIDIZATION[mask_inbox]
    feature_arr = self.interpolate(coords, weights)
    return feature_arr


class HydrogenBond(Feature):
  def __init__(self, mask="*", donor=False, acceptor=False):
    super().__init__()
    self.MASK = mask
    if not donor and not acceptor:
      raise ValueError("Either donor or acceptor should be True")
    self.FIND_DONOR = bool(donor)
    self.FIND_ACCEPTOR = bool(acceptor)
    self.ATOM_INDICES = np.array([])
    self.HBP_STATE = np.array([])

  def before_frame(self):
    self.ATOM_INDICES = self.traj.top.select(self.MASK)
    retmol = self.query_mol(self.ATOM_INDICES)
    result = []
    if retmol is not None:
      for atom in retmol.GetAtoms():
        symbol = atom.GetSymbol()
        hydrogen_count = atom.GetTotalNumHs()
        if self.FIND_DONOR:
          is_hbp = symbol in ['N', 'O', 'F'] and hydrogen_count > 0
        elif self.FIND_ACCEPTOR:
          is_hbp = symbol in ['N', 'O', 'F']
        else:
          is_hbp = False
        result.append(is_hbp)
    else:
      atoms = np.array([i for i in self.traj.top.atoms])
      for atom in atoms[self.ATOM_INDICES]:
        symbol = atom.atomic_number
        partners = atom.bonded_indices()
        atom_partners = atoms[partners]
        atom_numbers = np.array([i.atomic_number for i in atom_partners])
        hydrogen_count = np.count_nonzero(atom_numbers == 1)
        if self.FIND_DONOR:
          is_hbp = symbol in [7, 8, 9] and hydrogen_count > 0
        elif self.FIND_ACCEPTOR:
          is_hbp = symbol in [7, 8, 9]
        else:
          is_hbp = False
        result.append(is_hbp)

    self.HBP_STATE = np.array(result)[:len(self.ATOM_INDICES)].astype(int)
    # if len(self.HBP_STATE) != len(self.ATOM_INDICES):
    #   printit("Warning: The number of atoms in PDB does not match the number of HBP_STATE values");

  def featurize(self):
    coord_candidates = self.active_frame.xyz[self.ATOM_INDICES]
    mask_inbox = self.crop_box(coord_candidates)
    coords = coord_candidates[mask_inbox]
    weights = self.HBP_STATE[mask_inbox]
    feature_arr = self.interpolate(coords, weights)
    return feature_arr


class BoxFeature(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Meta information of the box. Generally not used as a feature but the meta information for the box
  """
  def __init__(self):
    super().__init__()

  def featurize(self):
    """
    Get the box configuration (generally not used as a feature)
    """
    box_feature = np.array([self.center, self.lengths, self.dims]).ravel()
    return box_feature


class FPFHFeature(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Fast Point Feature Histograms
  """
  def __init__(self, down_sample_nr=600):
    super().__init__()
    self.down_sample_nr = down_sample_nr

  def featurize(self):
    themesh = self.featurizer.contents["final_mesh"]
    fpfh = compute_fpfh_feature(themesh.sample_points_uniformly(self.down_sample_nr),
                                KDTreeSearchParamHybrid(radius=1, max_nn=20))
    print("FPFH feature: ", fpfh.data.shape)
    return fpfh.data


class PenaltyFeature(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Deviation from the center of the box
  """
  def __init__(self, mask1, mask2, **kwargs):
    super().__init__()
    self.mask1 = mask1
    self.mask2 = mask2
    self.use_mean = kwargs.get("use_mean", False)
    self.ref_frame = kwargs.get("ref_frame", 0)
    self.FAIL_FLAG = False
    self.pdist_mean = 999
    self.pdist = np.array([])
    self.pdistinfo = np.array([])

  def before_frame(self):
    """
    Get the mean pairwise distance
    """
    if _verbose:
      print("Precomputing the pairwise distance between the closest atom pairs")

    if isinstance(self.mask1, str):
      atom_select = self.traj.top.select(self.mask1)
    elif isinstance(self.mask1, (list, tuple, np.ndarray)):
      atom_select = np.array([int(i) for i in self.mask1])
    if isinstance(self.mask2, str):
      atom_counterpart = self.traj.top.select(self.mask2)
    elif isinstance(self.mask2, (list, tuple, np.ndarray)):
      atom_counterpart = np.array([int(i) for i in self.mask2])
    if len(atom_select) == 0:
      self.FAIL_FLAG = True
      printit("Warning: PenaltyFeature: Mask1 is empty. Marked the FAIL_FLAG. Please check the atom selection")
      return
    elif len(atom_counterpart) == 0:
      self.FAIL_FLAG = True
      printit("Warning: PenaltyFeature: Mask2 is empty. Marked the FAIL_FLAG. Please check the atom selection")
      return

    traj_copy = self.traj.copy_traj()
    traj_copy.top.set_reference(traj_copy[self.ref_frame])
    self.pdist, self.pdistinfo = utils.dist_caps(traj_copy, atom_select, atom_counterpart,
                                                        use_mean=self.use_mean, ref_frame=self.ref_frame)
    self.pdist_mean = self.pdist.mean(axis=1)
    if self.pdist.mean() > 8:
      printit("Warning: the mean distance between the atom of interest and its counterpart is larger than 8 Angstrom")
      printit("Please check the atom selection")
    elif np.percentile(self.pdist, 85) > 12:
      printit("Warning: the 85th percentile of the distance between the atom of interest and its counterpart is larger than 12 Angstrom")
      printit("Please check the atom selection")

    info_lengths = [len(self.pdistinfo[key]) for key in self.pdistinfo]
    if len(set(info_lengths)) != 1:
      printit("Warning: The length of the pdistinfo is not consistent", self.pdistinfo)

  def featurize(self):
    """
    Get the deviation from the center of the box
    """
    if self.FAIL_FLAG is True:
      return 0
    coord_diff = self.active_frame.xyz[self.pdistinfo["indices_group1"]] - self.active_frame.xyz[self.pdistinfo["indices_group2"]]
    dists = np.linalg.norm(coord_diff, axis=1)
    cosine_sim = utils.cosine_similarity(dists, self.pdist_mean)
    return cosine_sim


class MSCVFeature(Feature):
  def __init__(self, mask1, mask2, window_size, **kwargs):
    super().__init__()
    self.mask1 = mask1
    self.mask2 = mask2
    self.use_mean = kwargs.get("use_mean", False)
    self.ref_frame = kwargs.get("ref_frame", 0)
    self.WINDOW_SIZE = int(window_size)

  def before_frame(self):
    """
    Get the mean pairwise distance
    """
    if _verbose:
      print("Precomputing the pairwise distance between the closest atom pairs")
    self.traj_copy = self.traj.copy()
    self.traj_copy.top.set_reference(self.traj_copy[self.ref_frame])
    self.pd_arr, self.pd_info = utils.dist_caps(self.traj_copy, self.mask1, self.mask2,
                                                       use_mean=self.use_mean, ref_frame=self.ref_frame)
    self.mean_pd = np.mean(self.pd_arr, axis=1)

  def featurize(self):
    """
    Get the mean square coefficient of variation of the segment
    """
    framenr = self.traj.n_frames
    if framenr < self.WINDOW_SIZE:
      # If the window size is larger than the number of frames, then use the whole trajectory
      frames = np.arange(0, framenr)
    elif (self.active_frame_index + self.WINDOW_SIZE > framenr):
      # If the last frame is not enough to fill the window, then use the last window
      frames = np.arange(framenr - self.WINDOW_SIZE, framenr)
    else:
      frames = np.arange(self.active_frame_index, self.active_frame_index + self.WINDOW_SIZE)

    # Store the pairwise distance for each frame
    pdists = np.zeros((self.pd_arr.shape[0], len(frames)))
    for fidx in frames:
      dists = np.linalg.norm(
        self.active_frame.xyz[self.pd_info["indices_group1"]] - self.active_frame.xyz[self.pd_info["indices_group2"]],
        axis=1)
      pdists[:, fidx] = dists
    mscv = utils.mscv(pdists)
    return mscv

class EntropyResidueID(Feature):
  def __init__(self, mask="*", window_size=10):
    super().__init__()
    self.WINDOW_SIZE = int(window_size)
    self.MASK = mask

  def before_focus(self):
    """
    Before processing focus points, get the correct slice of frames
    """
    # Determine the slice of frames from a trajectory
    framenr = self.traj.n_frames
    if framenr < self.WINDOW_SIZE:
      print("The slice is larger than the total frames, using the whole trajectory")
      frames = np.arange(0, framenr)
    elif (self.active_frame_index + self.WINDOW_SIZE > framenr):
      print("Unable to fill the last frame and unable not enough to fill the window, then use the last window")
      frames = np.arange(framenr - self.WINDOW_SIZE, framenr)
    else:
      frames = np.arange(self.active_frame_index, self.active_frame_index + self.WINDOW_SIZE)
    self.ENTROPY_CUTOFF = np.linalg.norm(self.lengths / self.dims)    # Automatically set the cutoff for each grid
    self.ATOM_INDICES = self.top.select(self.MASK)
    self._COORD = self.traj.xyz[frames, self.ATOM_INDICES].reshape((-1, 3))
    self.RESID_ENSEMBLE = np.array([i.resid for idx,i in enumerate(self.top.atoms) if idx in self.ATOM_INDICES] * len(frames))
    self.FRAME_NUMBER = len(frames)
    if len(self._COORD) != len(self.RESID_ENSEMBLE):
      raise Exception("Warning: The number of coordinates and the number of residue indices are not equal")

  def featurize(self):
    """
    Get the information entropy(Chaoticity in general) of the box
    """
    grid_coord = np.array(self.points3d, dtype=np.float64)
    atom_coord = np.array(self._COORD, dtype=np.float64)
    atom_info = np.array(self.RESID_ENSEMBLE, dtype=int)
    entropy_arr = interpolate.query_grid_entropy(grid_coord, atom_coord, atom_info,
                                                 cutoff=self.ENTROPY_CUTOFF)
    if len(entropy_arr) != len(self.featurizer.distances):
      draw_call_stack()
      raise Exception("Warning: The entropy array does not match the number of grid points")
    elif len(entropy_arr) == 0:
      raise Exception("Warning: The entropy array is empty")
    entropy_arr = entropy_arr.reshape(self.dims)
    return entropy_arr


class EntropyAtomID(Feature):
  def __init__(self, mask="*", window_size=10):
    super().__init__()
    self.WINDOW_SIZE = int(window_size)
    self.MASK = mask
    self.ENTROPY_CUTOFF = 0.01
    self._COORD = None
    self.ATOM_INDICES = None
    self.INDEX_ENSEMBLE = None
    self.FRAME_NUMBER = 0

  def before_focus(self):
    # Determine the slice of frames from a trajectory
    framenr = self.traj.n_frames
    if framenr < self.WINDOW_SIZE:
      print("The slice is larger than the total frames, using the whole trajectory")
      frames = np.arange(0, framenr)
    elif self.active_frame_index + self.WINDOW_SIZE > framenr:
      print("Unable to fill the last frame and unable not enough to fill the window, then use the last window")
      frames = np.arange(framenr - self.WINDOW_SIZE, framenr)
    else:
      frames = np.arange(self.active_frame_index, self.active_frame_index + self.WINDOW_SIZE)
    self.ENTROPY_CUTOFF = np.linalg.norm(self.lengths / self.dims)    # Automatically set the cutoff for each grid
    self.ATOM_INDICES = self.top.select(self.MASK)
    self._COORD = self.traj.xyz[frames, self.ATOM_INDICES].reshape((-1, 3))
    self.INDEX_ENSEMBLE = np.array([i.index for idx, i in enumerate(self.top.atoms) if idx in self.ATOM_INDICES] * len(frames))
    self.FRAME_NUMBER = len(frames)
    if len(self._COORD) != len(self.INDEX_ENSEMBLE):
      raise Exception("Warning: The number of coordinates and the number of residue indices are not equal")

  def featurize(self):
    """
    Get the information entropy(Chaoticity in general) of the box
    """
    entropy_arr = interpolate.query_grid_entropy(self.points3d, self._COORD, self.INDEX_ENSEMBLE, cutoff=self.ENTROPY_CUTOFF)
    if len(entropy_arr) != len(self.featurizer.distances):
      draw_call_stack()
      raise Exception("Warning: returned entropy array does not match the grid size")
    entropy_arr = entropy_arr.reshape(self.dims)
    return entropy_arr


class RFFeature1D(Feature):
  def __init__(self, moiety_of_interest, cutoff=12):
    super().__init__()
    self.moi = moiety_of_interest
    # For the 4*9 (36) features
    # Rows (protein)   : C, N, O, S
    # Columns (ligand) : C, N, O, F, P, S, Cl, Br, I
    self.cutoff = cutoff
    self.pro_atom_idx = {6: 0, 7: 1, 8: 2, 16: 3}
    self.lig_atom_idx = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}

  def featurize(self):
    """
    The features for the RFScore algorithm
    Nine atom types are considered: C, N, O, F, P, S, Cl, Br, I
    Output contains 36 features because of the lack of F, P, Cl, Br and I atoms in the PDBbind protein structures
    Return:
      rf_arr: (36, 1) array
    """
    atoms = np.asarray([i.atomic_number for i in self.top.atoms])
    moi_indices = self.top.select(self.moi)     # The indices of the moiety of interest
    if len(moi_indices) == 0:
      printit("Warning: The moiety of interest is not found in the topology")
      # Return a zero array if the moiety is not found
      return np.zeros(36)
    if _verbose:
      printit(f"The moiety of interest contains {len(moi_indices)} atoms")
    # build a kd-tree for interaction query
    other_indices = np.asarray([i for i in np.arange(len(self.active_frame.xyz)) if i not in moi_indices])
    kd_tree = KDTree(self.active_frame.xyz[other_indices])
    rf_arr = np.zeros((4, 9))
    for i, idx in enumerate(moi_indices):
      atom_number = atoms[idx]
      atom_coord = self.active_frame.xyz[idx]
      soft_idxs = kd_tree.query_ball_point(atom_coord, self.cutoff)
      hard_idxs = other_indices[soft_idxs]
      for idx_prot in hard_idxs:
        atom_number_prot = atoms[idx_prot]
        if atom_number in self.lig_atom_idx and atom_number_prot in self.pro_atom_idx:
          rf_arr[self.pro_atom_idx[atom_number_prot], self.lig_atom_idx[atom_number]] += 1
    return rf_arr.reshape(-1)


class TopologySource(Feature):
  def __init__(self):
    super().__init__()

  def featurize(self):
    top_source = str(self.traj.top_filename)
    return [top_source]


class XYZCoord(Feature):
  def __init__(self, mask="*"):
    super().__init__()
    self.ATOM_INDICES = np.array([])
    self.MASK = mask

  def before_frame(self):
    self.ATOM_INDICES = self.traj.top.select(self.MASK)

  def featurize(self):
    return self.active_frame.xyz[self.ATOM_INDICES]


class FeaturizationStatus(Feature):
  def __init__(self):
    super().__init__()

  def featurize(self):
    return


class Feature_Dynamic(Feature): 
  def __init__(self): 
    # dims and lengths are set in the parent class
    super().__init__()
    # Add the interval attribute for a slice of frames
    self.__interval = None

  def __str__(self):
    ret_str = "Dynamic Feature: "
    ret_str += "Dimensions: " + " ".join([str(i) for i in self.dims] + " ")
    ret_str += f"Spacing: {self.spacing} \n"
    return ret_str

  def hook(self, featurizer): 
    super().hook(featurizer)
    self.interval = featurizer.interval

  @property
  def interval(self):
    return self.__interval
  @interval.setter
  def interval(self, value):
    self.__interval = int(value)

  @property
  def interval(self):
    return self.__interval
  @interval.setter
  def interval(self, value):
    self.__interval = int(value)

  def cache(self): 
    pass

  def hook(self): 
    pass

  def query(self):
    pass

  def run(self):
    pass

  def dump(self):
    pass

class density_flow(Feature_Dynamic):
  def __init__(self):
    super().__init__()
    self.__sigma = 0.0
    self.__cutoff = 0.0

  def __str__(self):
    ret_str = super().__str__()
    ret_str = f"With class name {self.__class__.__name__} \n"
    ret_str += f"sigma: {self.sigma}, "
    ret_str += f"cutoff: {self.cutoff}, "
    return ret_str
  
  @property
  def sigma(self):
    return self.__sigma
  @sigma.setter
  def sigma(self, value):
    self.__sigma = float(value)

  @property
  def cutoff(self):
    return self.__cutoff
  @cutoff.setter
  def cutoff(self, value):
    self.__cutoff = float(value)
  
  def hook(self, featurizer): 
    """
      Necessary for the synchronization of the parameters with the featurizer
    """
    # Using pass-by-reference to update the density flow parameters
    updates = {
      "sigma": featurizer.sigma,
      "cutoff": featurizer.cutoff,
      "spacing": featurizer.spacing,
      "dims": featurizer.dims
    }
    if "sigma" in updates:
      self.sigma = float(updates["sigma"])
    if "cutoff" in updates:
      self.cutoff = float(updates["cutoff"])
    if "spacing" in updates:
      self.spacing = float(updates["spacing"])
    if "dims" in updates:
      self.dims = updates["dims"]

  def cache(self, topology): 

    pass

  def query(self, topology, coordinates, focus):
    """
      Query the coordinates and weights and feed for the following self.run function
    """
    return frames, weights

  def run(self, frames, weights):
    """
      Take frames of coordinates and weights and return the a feature array with the same dimensions as self.dims
    """
    # # Run the density flow algorithm
    # if (frames.shape[0] * frames.shape[1]) % len(weights) != 0:
    #   raise ValueError(f"The number of atoms in the frames is not divisible by the number of weights: {frames.shape[0] * frames.shape[1]} % {len(weights)}")
    # if self.interval != frames.shape[0]: 
    #   raise ValueError(f"The number of frames is not equal to the interval: {frames.shape[0]} != {self.interval}")
    
    # # TODO: finish this implementation later on. 
    # ret_arr = commands.voxelize_trajectory(frames, weights, self.dims, self.spacing, self.interval, self.cutoff, self.sigma)

    # if (np.prod(self.dims)) != len(frames):
    #   raise ValueError("The number of frames is not equal to the number of frames in the interval")
    # else: 
    #   ret_arr = ret_arr.reshape(self.dims)
    #   return ret_arr
    return np.full(self.dims, 4.0, dtype=np.float32)

  

  def dump(self, results, filename):
    # Dump the results to a file
    pass



class marching_observers(Feature_Dynamic): 
  def __init__(self, weight_type="mass", agg_type="mean", obs_type="particle_existance"): 
    super().__init__()
    self.__cutoff = 1.0
    self.__weight_type = "uniform"        #  "uniform" "mass" "charge" etc
    self.__agg_type = "mean"              # "mean" "sum" "std"
    self.__obs_type = "particle_count"    #  "particle_count", "particle_existance", "particle_density"
    self.agg, self.obs, self.weight_type

  @property
  def cutoff(self):
    return self.__cutoff
  @cutoff.setter
  def cutoff(self, value):
    self.__cutoff = float(value)  

  @property
  def agg(self): 
    if self.__agg_type == "mean": 
      return 1
    elif self.__agg_type == "sum":
      return 2
    elif self.__agg_type == "std":
      return 3
    else:
      raise ValueError("The aggregation type is not recognized")

  @property
  def obs(self):
    if self.__obs_type == "particle_count":
      return 1
    elif self.__obs_type == "particle_existance": 
      return 2
    elif self.__obs_type == "particle_density":
      return 3
    else:
      raise ValueError("The observation type is not recognized")
  
  @property
  def weight_type(self):
    if self.__weight_type == "uniform":
      return 0
    elif self.__weight_type == "mass":
      return 1
    elif self.__weight_type == "charge":
      return 2
    else:
      raise ValueError("The weight type is not recognized")

  def hook(self, featurizer): 
    # Synchronize the parameters to featurizer
    updates = {
      "sigma": featurizer.sigma,
      "cutoff": featurizer.cutoff,
      "spacing": featurizer.spacing,
      "dims": featurizer.dims
    }
    if "dims" in updates:
      self.dims = updates["dims"]
    if "spacing" in updates:
      self.spacing = float(updates["spacing"])
    if "cutoff" in updates:
      self.cutoff = float(updates["cutoff"])
    if "interval" in updates:
      self.interval = float(updates["interval"])

  def cache(self, topology): 
    # TODO: Cache the weights according to the topology, 
    cache = np.full((topology.n_atoms), 0.0, dtype=np.float32)
    if self.weight_type == 1:
      # Map the mass to atoms 
      pass
    elif self.weight_type == 2:
      # Map the radius to atoms
      pass
    elif self.weight_type == 3:
      # Resiude ID
      pass
    elif self.weight_type == 4:
      # Sidechainness
      pass  
    else: 
      # Uniformed weights
      cache = np.full(len(cache), 1.0, dtype=np.float32)
    self.cache_weights = cache

  def query(self, topology, coordinates, focus):
    """
      Query the coordinates and weights from a set of frames and return the feature array
    """
    # Get a fixed number of atoms for self.interval frames
    assert self.interval == len(coordinates), f"Warning from feature ({self.__str__()}): the number of frames is not equal to the expected interval"
    MAX_ALLOWED_ATOMS = 1000
    coords = np.full((self.interval, MAX_ALLOWED_ATOMS, 3), 0.0, dtype=np.float32)
    weights = np.full((self.interval, MAX_ALLOWED_ATOMS), 0.0, dtype=np.float32)
    max_len = 0
    for i in range(self.interval):
      # TODO Get the index of the atoms within the focal point
      indices = np.arange(100)
      
      # Crop the frame within the focal point from the coordinates
      atoms = coordinates[i, indices, :].astype(np.float32)
      
      # Determine the way to get the frame coordinates and copy the coordinates to the array
      atom_nr = min(len(atoms), MAX_ALLOWED_ATOMS)
      if atom_nr > max_len:
        max_len = len(atoms)
      coords[i, :atom_nr, :] = atoms
      weights[i, :atom_nr] = self.cached_weights[indices]

    if max_len == MAX_ALLOWED_ATOMS:
      print("Warning: The number of atoms within one frame exceeds the maximum allowed atoms")

    # Create a new array with the maximum allowed atoms
    coords = np.array(coords[:, :max_len, :], dtype=np.float32)
    return coords, weights
  
  def run(self, frames): 
    """
      Get several frames and 
    """
    # if self.interval != frames.shape[0]: 
    #   raise ValueError("The number of frames is not equal to the interval")

    # # TODO correct the function name and the parameters
    # ret_arr = commands.do_marching( 
    #   frames, self.dims, 
    #   self.spacing, self.interval, self.cutoff,
    #   self.agg, self.obs
    # ) 

    # if (np.prod(self.dims)) != len(frames):
    #   raise ValueError("The number of frames is not equal to the number of frames in the interval")
    # else: 
    #   ret_arr = ret_arr.reshape(self.dims)
    #   return ret_arr
    return np.full(self.dims, 4.0, dtype=np.float32)

  def dump(self, results, filename):
    # Dump the results to a file
    with feater.io.hdffile(filename, "a") as hdf: 
      feater.utils.add_data_to_hdf(hdf, "label")
      




class Label: 
  def __init__(): 
    pass

  def hook(self):
    pass

  def cache(self):
    pass

  def query(self):
    pass

  def run(self):
    pass

  def dump(self):
    pass


def label_from_rmsd(Label): 
  def cache(self, trajectory): 
    self.cached_array = trajectory.rmsd()

  def query(self, topology, frames, focus): 
    return 

  def run(self, frames): 
    # Mainbody for label computation.
    return 1.0

  def dump(self, results, filename): 
    """
      Dump the results to a file
    """
    with feater.io.hdffile(filename, "a") as hdf: 
      feater.utils.add_data_to_hdf(hdf, "label", [results], dtype=np.float32, chunks=True, maxshape=(None,), compression="gzip", compression_opts=4)
    





class Vectorizer(Feature):
  def __init__(self):
    super().__init__()
    self.QUERIED_MOLS = {}
    
  
  
  def hook(self, featurizer):

    pass

  def before_frame(self):
    """
    This function is called before the feature is computed, it does nothing by default
    NOTE: User should override this function, if there is one-time expensive computation
    """
    pass

  def after_frame(self):
    """
    This function is called after the feature is computed, it does nothing by default
    NOTE: User should override this function, if there is one-time expensive computation
    """
    pass

  def before_focus(self):
    pass

  def after_focus(self):
    pass

  def next(self):
    """
    Update the active frame of the trajectory
    """
    pass

  
  
  def query_mol(self, selection):
    sel_hash = utils.get_hash(selection)
    frame_hash = utils.get_hash(self.active_frame.xyz.tobytes())
    final_hash = sel_hash + frame_hash

    if final_hash in self.QUERIED_MOLS:
      return self.QUERIED_MOLS[final_hash]
    else:
      retmol = self.featurizer.selection_to_mol(selection)
      self.QUERIED_MOLS[final_hash] = retmol
    return retmol

  def run(self, points, weights):
    """
    Interpolate density from a set of weighted 3D points to an N x N x N mesh grid.

    Args:
    points (np.array): An array of shape (num_points, 3) containing the 3D coordinates of the points.
    weights (np.array): An array of shape (num_points,) containing the weights of the points.
    grid_size (int): The size of the output mesh grid (N x N x N).

    Returns:
    np.array: A 3D mesh grid of shape (grid_size, grid_size, grid_size) with the interpolated density.
    """
    weights = np.nan_to_num(weights, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    grid_coords = np.array(self.points3d.astype(np.float64), dtype=np.float64)
    atom_coords = np.array(points.reshape(-1, 3), dtype=np.float64)
    weights = np.array(weights.reshape(-1), dtype=np.float64)

    # Interpolate the density
    print(f"Shape of grid_coords: {grid_coords.shape}")
    print(f"Shape of atom_coords: {atom_coords.shape}")
    print(f"Shape of weights: {weights.shape}")
    grid_density = interpolate.interpolate(grid_coords, atom_coords, weights)
    grid_density = grid_density.reshape(self.dims)
    return grid_density
  


