import subprocess
import numpy as np

from scipy.spatial import KDTree
from rdkit import Chem
from open3d.pipelines.registration import compute_fpfh_feature
from open3d.geometry import KDTreeSearchParamHybrid

from ..utils import utils
from .. import CONFIG, printit
from .. import _usegpu, _verbose

# import time
# from open3d.io import write_triangle_mesh
# from numba import jit
# from .. import _clear, _debug
# import os, copy, datetime
# import builtins, json, tempfile, functools
# import pytraj as pt

"""
When writing the customised feature, the following variables are automatically available:
  self.featurizer: The featurizer object
  self.top: The topology of the trajectory
  self.active_frame: The active frame of the trajectory
  self.grid: The grid for the feature
  self.center: The center of the box
  self.lengths: The lengths of the box
  self.dims: The dimensions of the box
  

When registering the customised feature, the following needs to be defined by the user:
  self.featurize: The function to generate the feature from the trajectory

! IMPORTANT: 
  If there is difference from trajectory to trajectory, for example the name of the ligand, feature object has to behave 
  differently. However, the featurizer statically pipelines the feature.featurize() function and difference is not known
  in these featurization process. 
  This change has to take place in the featurizer object because the trajectories are registered into the featurizer. Add 
  the variable attribute to the featurizer object and use it in the re-organized feature.featurize() function.
    
"""


class Feature:
  """
  The base class for all features
  """
  def __init__(self):
    print(f"Initializing the feature base class {self.__class__.__name__}")
    self.featurizer = None

  def __str__(self):
    return self.__class__.__name__

  def set_featurizer(self, featurizer): 
    """
    Hook the feature generator back to the feature convolutor and obtain necessary attributes from the featurizer
    including the trajectory, active frame, convolution kernel etc
    """
    self.featurizer = featurizer
    if _verbose:
      printit(f"Hooking featurizer to {self.__class__.__name__}")

  # The Feature can ONLY READ the necessary attributes of the featurizer, but not udpate them.
  @property
  def active_frame(self):
    return self.featurizer.active_frame

  @property
  def active_frame_index(self):
    return self.featurizer.active_frame_index

  @property
  def traj(self):
    return self.featurizer.traj

  @property
  def top(self):
    return self.featurizer.traj.top

  @property
  def center(self):
    return np.asarray(self.featurizer.center)

  @property
  def lengths(self):
    return np.asarray(self.featurizer.lengths)

  @property
  def dims(self):
    return np.asarray(self.featurizer.dims)

  @property
  def grid(self):
    return np.array(self.featurizer.grid)

  @property
  def points3d(self):
    return np.array(self.featurizer.points3d)

  @property
  def status_flag(self):
    return self.featurizer.status_flag

  def crop_box(self, points):
    """
    Crop the points to the box defined by the center and lengths
    The mask is returned as a boolean array
    """
    thecoord = np.asarray(points)
    upperbound = self.center + self.lengths / 2
    lowerbound = self.center - self.lengths / 2
    ubstate = np.all(thecoord < upperbound, axis=1)
    lbstate = np.all(thecoord > lowerbound, axis=1)
    mask_inbox = ubstate * lbstate
    return mask_inbox
  
  def query_mol(self, selection):
    retmol = self.featurizer.boxed_to_mol(selection)
    return retmol

  # @profile
  def interpolate(self, points, weights):
    """
    Interpolate density from a set of weighted 3D points to an N x N x N mesh grid.

    Args:
    points (np.array): An array of shape (num_points, 3) containing the 3D coordinates of the points.
    weights (np.array): An array of shape (num_points,) containing the weights of the points.
    grid_size (int): The size of the output mesh grid (N x N x N).

    Returns:
    np.array: A 3D mesh grid of shape (grid_size, grid_size, grid_size) with the interpolated density.
    """
    from . import interpolate
    weights = np.nan_to_num(weights, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    atom_coords = points.reshape(-1, 3)
    # Interpolate the density
    if _usegpu:
      grid_density = interpolate.interpolate_gpu(self.points3d, atom_coords, weights)
    else:
      grid_density = interpolate.interpolate(self.points3d, atom_coords, weights)
    grid_density = grid_density.reshape(self.dims)
    return grid_density

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


class Mass(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Atomic mass as a feature
  """
  def __init__(self, mask="*"):
    super().__init__()
    self.MASK = mask
    self.ATOMIC_NUMBERS = None
    self.INDEX_CANDIDATES = np.array([])

  def before_frame(self):
    """
    Since topology does not change during the simulation, we can precompute the atomic mass
    """
    self.ATOMIC_NUMBERS = np.array([int(i.atomic_number) for i in self.traj.top.atoms])
    self.INDEX_CANDIDATES = self.traj.top.select(self.MASK)

  def featurize(self): 
    """
    1. Get the atomic feature
    2. Update the feature 
    """
    # Get the atoms within the bounding box
    coord_candidates = self.active_frame.xyz[self.INDEX_CANDIDATES]
    mass_candidates = self.ATOMIC_NUMBERS[self.INDEX_CANDIDATES]
    mask_inbox = self.crop_box(coord_candidates)

    # Get the coordinate/required atomic features within the bounding box
    coords = coord_candidates[mask_inbox]
    weights = mass_candidates[mask_inbox]

    feature_arr = self.interpolate(coords, weights)
    return feature_arr


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


class HeavyAtom(Feature):
  def __init__(self, mask="*", reverse=False):
    """
    Compute the heavy atom feature (or non-heavy atom feature) for the atoms of interest
    Args:
      mask: The mask for the atoms of interest
      reverse: Whether or not to reverse the function to compute the non-heavy atoms (hydrogen atoms)
    """
    super().__init__()
    self.MASK = mask
    self.REVERSE = bool(reverse)
    self.ATOM_INDICES = np.array([])
    self.ATOMIC_NUMBERS = np.array([])
    self.ATOM_STATE = np.array([])

  def before_frame(self):
    self.ATOM_INDICES = self.traj.top.select(self.MASK)
    self.ATOMIC_NUMBERS = np.array([int(i.atomic_number) for i in self.traj.top.atoms])
    if self.REVERSE:
      self.ATOM_STATE = np.array(self.ATOMIC_NUMBERS == 1, dtype=bool)
    else:
      self.ATOM_STATE = np.array(self.ATOMIC_NUMBERS > 1, dtype=bool)

  def featurize(self):
    coord_candidates = self.active_frame.xyz[self.ATOM_INDICES]
    mask_inbox = self.crop_box(coord_candidates)
    coords = coord_candidates[mask_inbox]
    weights = self.ATOM_STATE[self.ATOM_INDICES][mask_inbox]
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
  def __init__(self):
    super().__init__()

  def featurize(self):
    down_sample_nr = CONFIG.get("DOWN_SAMPLE_POINTS", 600)
    fpfh = compute_fpfh_feature(self.featurizer.mesh.sample_points_uniformly(down_sample_nr),
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
    self.pdist, self.pdistinfo = utils.PairwiseDistance(traj_copy, atom_select, atom_counterpart,
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
  def __init__(self, mask1, mask2, **kwargs):
    super().__init__()
    self.mask1 = mask1
    self.mask2 = mask2
    self.use_mean = kwargs.get("use_mean", False)
    self.ref_frame = kwargs.get("ref_frame", 0)
    self.WINDOW_SIZE = CONFIG.get("WINDOW_SIZE", 10)

  def before_frame(self):
    """
    Get the mean pairwise distance
    """
    if _verbose:
      print("Precomputing the pairwise distance between the closest atom pairs")
    self.traj_copy = self.traj.copy()
    self.traj_copy.top.set_reference(self.traj_copy[self.ref_frame])
    self.pd_arr, self.pd_info = utils.PairwiseDistance(self.traj_copy, self.mask1, self.mask2,
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
    mscv = utils.MSCV(pdists)
    return mscv

class EntropyResidueID(Feature):
  def __init__(self, mask="*"):
    super().__init__()
    self.WINDOW_SIZE = CONFIG.get("WINDOW_SIZE", 10)
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
    from . import interpolate
    entropy_arr = interpolate.grid4entropy(self.points3d, self._COORD, self.RESID_ENSEMBLE,
                                           cutoff=self.ENTROPY_CUTOFF)
    print(f"Check the entropy array: {entropy_arr.shape}")
    entropy_arr = entropy_arr.reshape(self.dims)
    return entropy_arr


class EntropyAtomID(Feature):
  def __init__(self, mask="*"):
    super().__init__()
    self.WINDOW_SIZE = CONFIG.get("WINDOW_SIZE", 10)
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
    from . import interpolate
    entropy_arr = interpolate.grid4entropy(self.points3d, self._COORD, self.INDEX_ENSEMBLE, cutoff=self.ENTROPY_CUTOFF)
    print(f"Check the entropy array: {entropy_arr.shape}")
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
