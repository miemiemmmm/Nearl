import subprocess, json, tempfile
import sys
import numpy as np

from rdkit import Chem
from scipy.spatial import KDTree
import pytraj as pt

from . import utils, commands, constants


from . import printit, draw_call_stack
from . import _verbose

__all__ = [
  # Base class
  "Feature",
  # Static Features
  "Mass",
  "HeavyAtom",

  # Dynamic features
  "DensityFlow",
  "MarchingObservers",
]

"""
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

  Parameters
  ----------
  points : np.ndarray
    The coordinates of the atoms
  upperbound : np.ndarray
    The upperbound of the box
  padding : float
    The padding of the box

  Returns
  -------
  mask_inbox : np.ndarray
    The boolean mask of the atoms within the box
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
  """
  Base class for the feature generator
  """
  # The input and the output of the query, run and dump function should be chained together.
  # The hook function should 
  #   Key methods
  # -----------
  # - hook: Hook the feature generator back to the feature convolutor and obtain necessary attributes from the featurizer
  # - cache: Cache the needed weights for each atom in the trajectory for further Feature.query function
  # - query: Query the atoms and weights within the bounding box
  # - run: Run the GPU-based feature generator based on the return of the Feature.query function, 
  # - dump: Dump the feature to the disk

  # Key parameters
  # --------------
  # - dims: The dimensions of the grid
  # - spacing: The spacing of the grid
  def __init__(self, dims=None, spacing=None, 
               outfile=None, outkey=None,
               cutoff=None, sigma=None,
               padding=0, byres=None, 
               outshape=None, force_requery=False,
               **kwargs):
    # Fundamental variables need setter callback 
    self.__dims = None
    self.__spacing = None
    self.dims = dims
    self.spacing = spacing
    self.__center = None
    self.__lengths = None

    # Simple variables, no need for setter callback, not all of them might be used in the feature definition
    self.cutoff = cutoff
    self._cutoff = False if cutoff is None else True
    self.sigma = sigma
    self._sigma = False if sigma is None else True
    self.outfile = outfile
    self._outfile = False if outfile is None else True
    self.outkey = outkey
    self._outkey = False if outkey is None else True
    self.padding = padding
    self._padding = False if padding == 0 else True
    self.byres = byres
    self._byres = False if byres is None else True

    # Simple variables, no need for setter callback and hook to featurizer
    self.outshape = outshape
    self._outshape = False if outshape is None else True
    self.force_requery = force_requery
    self._force_requery = False if force_requery is False else True

  def __str__(self):
    ret_str = "Feature: "
    if self.dims is not None:
      ret_str += "Dimensions: " + " ".join([str(i) for i in self.dims] + " ")
    if self.spacing is not None:
      ret_str += f"Spacing: {self.spacing} \n"
    return ret_str

  @property
  def dims(self):
    """
    Dimensions of the 3D grid
    """
    return self.__dims
  @dims.setter
  def dims(self, value):
    if isinstance(value, (int, float, np.int64, np.float64, np.int32, np.float32)): 
      self.__dims = np.array([int(value), int(value), int(value)])
    elif isinstance(value, (list, tuple, np.ndarray)):
      self.__dims = np.array([int(i) for i in value][:3])
    else:
      if _verbose:
        printit(f"Warning: The dims should be a number, list, tuple or a numpy array, not {type(value)}")
      self.__dims = None
    if self.__spacing is not None:
      self.__center = np.array(self.__dims * self.spacing, dtype=np.float32) / 2
    if self.__dims is not None and self.__spacing is not None:
      self.__lengths = self.__dims * self.__spacing

  @property
  def spacing(self):
    """
    The spacing (resolution) of the 3D grid
    """
    return self.__spacing
  @spacing.setter
  def spacing(self, value):
    if isinstance(value, (int, float, np.int64, np.float64, np.int32, np.float32)):
      self.__spacing = float(value)
    else: 
      if _verbose:
        printit(f"{self.__class__.__name__}: Warning: The spacing is not a valid number, setting to None")
      self.__spacing = None
    if self.__dims is not None:
      self.__center = np.array(self.__dims * self.spacing, dtype=np.float32) / 2
    if self.__dims is not None and self.__spacing is not None:
      self.__lengths = self.__dims * self.__spacing

  @property
  def center(self):
    """
    Center of the grid, read-only property calculated by dims and spacing
    """
    return self.__center
  @property
  def lengths(self):
    """
    Lengths of the grid, read-only property calculated by dims and spacing
    """
    return self.__lengths

  def hook(self, featurizer): 
    """
    Hook the feature generator back to the feature convolutor and obtain necessary attributes from the featurizer
    including the trajectory, active frame, convolution kernel etc

    Parameters
    ----------
    featurizer : nearl.featurizer.Featurizer
      The featurizer object describing the feature generation process

    Notes
    --------
    If the following attributes are not set manually, hook function will try to inherit them from the featurizer object: 
    sigma, cutoff, outfile, outkey, padding, byres
    """
    self.dims = featurizer.dims
    self.spacing = featurizer.spacing
    # TODO: Update this upon adding more variables to the feature
    for key in ["outfile", "outkey", "cutoff", "sigma", "padding", "byres"]:
      if getattr(self, f"_{key}") == False: 
        # Try to inherit the attributes from the featurizer if the attribute is not manually set
        if key in dir(featurizer) and getattr(featurizer, key) is not None:
          printit(f"{self.__class__.__name__}: Inheriting the sigma from the featurizer: {key} {getattr(featurizer, key)}")
          setattr(self, key, getattr(featurizer, key))

  def cache(self, trajectory): 
    """
    Cache the needed weights for each atom in the trajectory for further Feature.query() function

    Parameters
    ----------
    trajectory : nearl.io.traj.Trajectory
    """
    atoms = [i for i in trajectory.top.atoms]
    self.resids = np.array([i.resid for i in atoms], dtype=int)
    self.atomic_numbers = np.array([i.atomic_number for i in atoms], dtype=int)

  def query(self, topology, frame_coords, focal_point):
    """
    Base function to query the coordinates within the the bounding box near the focal point 

    Parameters
    ----------
    topology : pytraj.Topology
      The topology object
    frame_coords : np.ndarray
      The coordinates of the atoms
    focal_point : np.ndarray
      The focal points parsed from the your registered points see the featurizer object

    Returns
    -------
    final_mask : np.ndarray
      The boolean mask of the atoms within the bounding box

    Notes
    -----
    In the child feature, after croping the coordinates near the focus, move the coordinates to the center of the box before sending to the runner function
    """
    if focal_point.shape.__len__() > 1: 
      raise ValueError(f"The focal point should be a 1D array with length 3, not {focal_point.shape}")
    new_coords = frame_coords - focal_point + self.center 
    # No padding 
    mask = crop(new_coords, self.lengths, self.padding)
    # Get the boolean array of residues within the bounding box
    
    if (len(self.resids) != topology.n_atoms) or self.force_requery: 
      # Deal with inhomogeneous topology during iterating focal points
      if _verbose:
        printit(f"{self.__class__.__name__}: Dealing with inhomogeneous topology")
      if self.byres:
        res_ids = np.array([i.resid for i in topology.atoms])
        # print(f"Resids: {res_ids}, {mask}, {frame_coords.shape}")
        res_inbox = np.unique(res_ids[mask])
        final_mask = np.full(len(res_ids), False)
        for res in res_inbox:
          final_mask[np.where(res_ids == res)] = True
      else: 
        final_mask = mask
    else: 
      if self.byres:
        res_inbox = np.unique(self.resids[mask])
        final_mask = np.full(len(self.resids), False)
        for res in res_inbox:
          final_mask[np.where(self.resids == res)] = True
      else: 
        final_mask = mask
    return final_mask

  def run(self, coords, weights):
    """
    Take the output from 

    Parameters
    ----------
    coords : np.ndarray
      The coordinates of the atoms within the bounding box
    weights : np.ndarray
      The weights of the atoms within the bounding box

    Returns
    -------
    ret_arr : np.ndarray
      The result feature array

    """
    return np.zeros(self.dims, dtype=np.float32)

  def dump(self, result):
    """
    Dump the result feature to an HDF5 file, Feature.outfile and Feature.outkey should be set in its child class (Either via __init__ or hook function)

    Parameters
    ----------
    results : np.array
      The result feature array
    """
    if ("outfile" in dir(self)) and ("outkey" in dir(self)) and (len(self.outfile) > 0):
      if self._outshape is True:  
        # Explicitly set the shape of the output
        utils.append_hdf_data(self.outfile, self.outkey, np.asarray([result], dtype=np.float32), dtype=np.float32, maxshape=(None, *self.outshape), chunks=True, compression="gzip", compression_opts=4)
      elif len(self.dims) == 3: 
        utils.append_hdf_data(self.outfile, self.outkey, np.asarray([result], dtype=np.float32), dtype=np.float32, maxshape=(None, *self.dims), chunks=True, compression="gzip", compression_opts=4)
  


class Mass(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Parse of the Mask should not be in here, the input should be focal points in coordinates format
  Explicitly pass the cutoff and sigma while initializing the Feature object for the time being
  Atomic mass as a feature
  """
  def __init__(self, cutoff=None, sigma=None, **kwargs):
    super().__init__(cutoff=cutoff, sigma=sigma, **kwargs)

  def query(self, topology, frame_coords, focal_point): 
    """
    Get the atoms and weights within the bounding box

    Parameters
    ----------
    topology : pytraj.Topology
      The topology object
    frame_coords : np.ndarray
      The coordinates of the atoms
    focal_point : np.ndarray
      The focal points parsed from the your registered points see the featurizer object

    Returns
    -------
    coord_inbox : np.ndarray
      The coordinates of the atoms within the bounding box
    weights : np.ndarray
      The weights of the atoms within the bounding box

    Notes
    -----
    If a multiple frames are put to static feature, the frame_coords will take the first frame.

    Run the query method from the parent class to get the mask of the atoms within the bounding box. 

    Before sending the coordinates to the runner function, move the coordinates to the center of the box. 
    """
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    if (len(self.resids) != topology.n_atoms) or self.force_requery: 
      atomic_numbers = np.array([i.atomic_number for i in topology.atoms])
      weights = np.array([constants.ATOMICMASS[i] for i in atomic_numbers[idx_inbox]], dtype=np.float32)
    else:
      weights = np.array([constants.ATOMICMASS[i] for i in self.atomic_numbers[idx_inbox]], dtype=np.float32)
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights

  def run(self, coords, weights): 
    """
    Voxelization of the atomic mass
    """
    if len(coords) == 0:
      printit(f"{self.__class__.__name__}: Warning: The coordinates are empty")
      return np.zeros(self.dims, dtype=np.float32)
    # printit(f"{self.__class__.__name__}: Center of the coordinates: {np.mean(coords, axis=0)}")
    ret = commands.voxelize_coords(coords, weights, self.dims, self.spacing, self.cutoff, self.sigma)
    printit(f"{self.__class__.__name__}: The sum of the returned array: {np.sum(ret)} VS {np.sum(weights)} from {len(weights)} atoms")
    return ret
    

class HeavyAtom(Feature):
  def __init__(self, default_weight=1, 
               cutoff=None, sigma=None, **kwargs):
    super().__init__(cutoff=cutoff, sigma=sigma, **kwargs)
    self.default_weight = default_weight

  def cache(self, trajectory):
    """
    Prepare the heavy atom weights
    """
    if not hasattr(trajectory, "atoms") or not hasattr(trajectory, "residues"): 
      trajectory.make_index()

    super().cache(trajectory)
    self.heavy_atoms = np.full(len(self.resids), 0, dtype=np.float32)
    self.heavy_atoms[np.where(self.atomic_numbers > 1)] = self.default_weight

  def query(self, topology, frame_coords, focal_point): 
    """
    Get the atoms and weights within the bounding box
    """
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]  # NOTE: Get the first frame if multiple frames are given
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    if (len(self.resids) != topology.n_atoms) or self.force_requery: 
      weights = np.array([i.atomic_number > 1 for i in topology.atoms], dtype=np.float32)[idx_inbox]
    else:
      weights = self.heavy_atoms[idx_inbox]
    # Translate the result coordinates to the center of the box
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights

  def run(self, coords, weights): 
    """
    Voxelization of the atomic mass
    """
    # printit(f"{self.__class__.__name__}: Center of the coordinates: {np.mean(coords, axis=0)}")
    # Host-py function: voxelize_coords(coords, weights, grid_dims, spacing, cutoff, sigma):

    # from nearl.all_actions import do_voxelize
    
    # thefunc = copy.deepcopy(commands.voxelize_coords)
    # ret = thefunc(coords, weights, self.dims, self.spacing, self.cutoff, self.sigma)
    # coords = np.array(coords, dtype=np.float32)
    # ret = do_voxelize(coords, weights, self.dims, self.spacing, self.cutoff, self.sigma, auto_translate=0)
    ret = commands.voxelize_coords(coords, weights, self.dims, self.spacing, self.cutoff, self.sigma)
    # ret = tmpfunc("voxelize", coords, weights, self.dims, self.spacing, self.cutoff, self.sigma)
    
    printit(f"{self.__class__.__name__}: The sum of the returned array: {np.sum(ret)} VS {np.sum(weights)} from {len(weights)} atoms")
    return ret

def tmpfunc(functype, *args): 
  if functype == "voxelize": 
    ret = commands.voxelize_coords(*args)
  elif functype == "marching":
    ret = commands.marching_observers(*args)
  return ret

  


class PartialCharge(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Atomic charge feature for the structure of interest;
  Compute the charge based on the self.featurizer.boxed_pdb;
  """
  def __init__(self, type=None, parm=None, mode=None):
    super().__init__()
    [ "sqeqp",  "eem",  "abeem",  "sfkeem",  "qeq",
      "smpqeq",  "eqeq",  "eqeqc",  "delre",  "peoe",
      "mpeoe",  "gdac",  "sqe",  "sqeq0",  "mgc",
      "kcm",  "denr",  "tsef",  "charge2",  "veem",
      "formal"]
    # TODO: Change the way of using different modes 
    # /MieT5/BetaPose/nearl/data/charge_charmm36.json
    # /MieT5/BetaPose/nearl/data/charge_ff14sb.json
    # if mode == "manual":
    #   self.mode = "manual"
    # elif mode == "gasteiger": 
    #   self.mode = "gasteiger"
    # elif mode == "charmm36":
    #   self.mode = "charmm36"
    # elif mode == "ff14sb":
    #   self.mode = "ff14sb"
    # else:
    #   self.mode = "charmm36"

    self.charge_type = "eem"
    self.charge_parm = "EEM_00_NEEMP_ccd2016_npa"
    self.charge_values = None

  def cache(self, trajectory):
    super().cache(trajectory)
    if "charge" in trajectory.cached.keys(): 
      self.charge_values = np.array(trajectory.cached["charge"])
    else: 
      import chargefw2_python as cfw
      charges = None
      with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
        trajectory.write_frame(0, outfile=f.name)
        try: 
          printit(f"Calculating the molecular charge of the trajectory")
          mol = cfw.Molecules(f.name)
          charges = cfw.calculate_charges(mol, self.charge_type, self.charge_parm)
          printit(f"Finished the partial charge calculation")
        except Exception as e: 
          printit(f"Failed to calculate molecular charge: {e}")
      if charges is not None:
        thekey = charges.keys()[0]
        self.charge_values = np.array(charges[thekey])
      else: 
        print("Warning: The charge values are not set", file=sys.stderr)
        self.charge_values = np.zeros(len(trajectory.n_atoms))
    
    
    # self.ATOM_INDICES = self.traj.top.select(self.MASK)
    # if self.mode == "manual":
    #   self.charge_values = np.asarray(self.charge_values).astype(float)
    # elif self.mode == "gasteiger":
    #   retmol = self.query_mol(self.ATOM_INDICES)
    #   if retmol is not None:
    #     self.charge_values = np.array([float(atom.GetProp("_GasteigerCharge")) for atom in retmol.GetAtoms()]).astype(float)
    #     self.charge_values = self.charge_values[:len(self.ATOM_INDICES)]
    #   else:
    #     self.charge_values = np.zeros(len(self.ATOM_INDICES)).astype(float)
    # elif self.mode == "charmm36":
    #   # Use /MieT5/BetaPose/nearl/data/charge_charmm36.json
    #   atom_ids = [i.name for i in self.traj.top.atoms]
    #   with open("/MieT5/BetaPose/nearl/data/charge_charmm36.json", "r") as f:
    #     self.charge_values = json.load(f)
    # elif self.mode == "ff14sb":
    #   with open("/MieT5/BetaPose/nearl/data/charge_ff14sb.json", "r") as f:
    #     self.charge_values = json.load(f)
    # if len(self.charge_values) != len(self.ATOM_INDICES):
    #   printit("Warning: The number of atoms in PDB does not match the number of charge values")

  def query(self, topology, frame_coords, focal_point):
    """
    NOTE:
    The self.boxed_pdb is already cropped and atom are reindexed in the PDB block.
    Hence use the self.boxed_indices to get the original atom indices standing for the PDB block
    """
    # Get the atoms within the bounding box
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]  # NOTE: Get the first frame if multiple frames are given
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.charge_values[idx_inbox]   # TODO change this 
    # Translate the result coordinates to the center of the box
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights
  
  def run(self, coords, weights): 
    """
    Voxelization of the atomic mass
    """
    printit(f"{self.__class__.__name__}: Center of the coordinates: {np.mean(coords, axis=0)}")
    # Host-py function: voxelize_coords(coords, weights, grid_dims, spacing, cutoff, sigma):
    ret = commands.voxelize_coords(coords, weights, self.dims, self.spacing, self.cutoff, self.sigma)
    printit(f"{self.__class__.__name__}: The sum of the returned array: {np.sum(ret)} VS {np.sum(weights)} from {len(weights)} atoms")
    return ret



class AtomTypeFeature(Feature):
  def __init__(self, element=None):
    super().__init__()

    self.element = element

  def element_type(self, atomic_number):
    return 
  
  def cache(self, trajectory):
    super().cache(trajectory)
    self.istype = [1 if i == 12 else 0 for i in self.atomic_numbers]



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

  def featurize(self):
    coord_candidates = self.active_frame.xyz[self.ATOM_INDICES]
    mask_inbox = self.crop_box(coord_candidates)
    coords = coord_candidates[mask_inbox]
    weights = self.HBP_STATE[mask_inbox]
    feature_arr = self.interpolate(coords, weights)
    return feature_arr


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


class DensityFlow(Feature):
  """
  Dynamic feature: Density flow,

  Aggregation type: mean, std, sum

  Weight type: mass, radius, residue_id, sidechainness, uniform
    
  """
  def __init__(self, agg = "mean", weight_type="mass", 
               cutoff=None, sigma=None, outfile=None, outkey=None, **kwargs):
    super().__init__(cutoff=cutoff, sigma=sigma, outfile=outfile, outkey=outkey, **kwargs)
    self.__agg_type = agg
    self.__weight_type = weight_type

  @property
  def agg(self):
    if self.__agg_type == "mean":
      return 0
    elif self.__agg_type == "median":
      return 1
    elif self.__agg_type == "std":
      return 2
    elif self.__agg_type == "variance":
      return 3
    elif self.__agg_type == "max":
      return 4
    elif self.__agg_type == "min":
      return 5
    else:
      raise ValueError("The aggregation type is not recognized")
  @agg.setter
  def agg(self, value):
    assert isinstance(value, str), "The aggregation type should be a string"
    self.__agg_type = value
  
  @property
  def weight_type(self):
    if self.__weight_type == "atomic_number":
      return 0
    elif self.__weight_type == "mass":
      return 1
    elif self.__weight_type == "radius":
      return 2
    elif self.__weight_type == "residue_id":
      return 3
    elif self.__weight_type == "sidechainness":
      return 4
    else:
      raise ValueError("The weight type is not recognized")
  @weight_type.setter
  def weight_type(self, value):
    assert isinstance(value, str), "The weight type should be a string"
    self.__weight_type = value
  

  def cache(self, trajectory): 
    super().cache(trajectory)
    cache = np.full((trajectory.n_atoms), 0.0, dtype=np.float32)
    if self.weight_type == 0:
      cache = np.array(self.atomic_numbers, dtype=np.float32)
    elif self.weight_type == 1:
      # Map the mass to atoms 
      cache = np.array([constants.ATOMICMASS[i] for i in self.atomic_numbers], dtype=np.float32)
    elif self.weight_type == 2:
      # Map the radius to atoms
      cache = np.array([utils.VDWRADII[str(i)] for i in self.atomic_numbers], dtype=np.float32)
    elif self.weight_type == 3:
      # Resiude ID
      cache = np.array([i.resid for i in self.top.atoms], dtype=np.float32)
    elif self.weight_type == 4:
      # Sidechainness
      cache = np.array([1 if i.name in ["C", "N", "O", "CA"] else 0 for i in self.top.atoms], dtype=np.float32)
    else: 
      # Uniformed weights
      cache = np.full(len(cache), 1.0, dtype=np.float32)
    self.cached_weights = cache

  def query(self, topology, frame_coords, focal_point):
    """
    Query the coordinates and weights and feed for the following self.run function
    """
    assert len(frame_coords.shape) == 3, f"Warning from feature ({self.__str__()}): The coordinates should follow the convention (frames, atoms, 3); "
    # NOTE: Depend on the GPU cache size for each thread
    MAX_ALLOWED_ATOMS = 1000 
    max_atom_nr = 0
    coord_list = []
    weight_list = []

    if self.force_requery:
      self.cache(pt.Trajectory(top=topology, xyz=frame_coords))

    # Count the maximum number of atoms in the box across all frames and get the coordinates and weights
    for frame in frame_coords:
      idx_inbox = super().query(topology, frame, focal_point)
      coord_list.append(frame[idx_inbox] - focal_point + self.center)
      weight_list.append(self.cached_weights[idx_inbox])
      max_atom_nr = max(max_atom_nr, np.count_nonzero(idx_inbox))
    printit(f"{self.__class__.__name__}: The maximum number of atoms in the box: {max_atom_nr}, averaged number {np.mean([len(i) for i in coord_list])}.")
    
    if max_atom_nr > MAX_ALLOWED_ATOMS:
      printit(f"Warning: the maximum allowed atom slice is {MAX_ALLOWED_ATOMS} but the maximum atom number is {max_atom_nr}")
    
    # After processing each frames (Python list of weights and coordinates), build fixed sized numpy arrays for runner function
    max_atom_nr = min(MAX_ALLOWED_ATOMS, max_atom_nr)
    ret_coord = np.full((len(frame_coords), max_atom_nr, 3), 99999.0, dtype=np.float32)
    ret_weight = np.full((len(frame_coords), max_atom_nr), 0.0, dtype=np.float32)
    # NOTE: The weights array is flattened to a 1D array
    ret_weight = np.full(len(frame_coords) * max_atom_nr, 0.0, dtype=np.float32)
    for idx, (coord, weight) in enumerate(zip(coord_list, weight_list)):
      therange = min(len(coord), max_atom_nr)
      ret_coord[idx, :therange] = coord[:therange]
      ret_weight[idx*max_atom_nr:idx*max_atom_nr+therange] = weight[:therange]
    return ret_coord, ret_weight

  def run(self, frames, weights):
    """
    Take frames of coordinates and weights and return the a feature array with the same dimensions as self.dims
    """
    # Run the density flow algorithm
    if (frames.shape[0] * frames.shape[1]) != len(weights): 
      raise ValueError(f"The number of atoms in the frames is not divisible by the number of weights: {frames.shape[0] * frames.shape[1]} % {len(weights)}")
    
    frames = np.array(frames, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)
    ret_arr = commands.voxelize_trajectory(frames, weights, self.dims, self.spacing, self.cutoff, self.sigma, self.agg)
    return ret_arr.reshape(self.dims)  


class MarchingObservers(DensityFlow): 
  """
  Perform the marching observers algorithm to get the dynamic feature. 

  Inherit from the DensityFlow class since there are common ways to query the coordinates and weights. 

  Observation types: particle_count, particle_existance, mean_distance, radius_of_gyration

  Weight types: mass, radius, residue_id, sidechainness, uniform

  Aggregation type: mean, std, sum
  
  """
  def __init__(self, obs="particle_existance", 
               agg="mean", weight_type="mass", 
               cutoff=None, outfile=None, outkey=None, **kwargs): 
    # Just omit the sigma parameter while the inheritance. 
    # while initialization of the parent class, weight_type, cutoff, agg are set
    super().__init__(agg=agg, weight_type=weight_type, cutoff=cutoff, outfile=outfile, outkey=outkey, **kwargs)
    self.__obs_type = obs

  @property
  def obs(self):
    if self.__obs_type == "particle_count":
      return 1
    elif self.__obs_type == "particle_existance": 
      return 2
    elif self.__obs_type == "mean_distance":
      return 3
    elif self.__obs_type == "radius_of_gyration":
      return 4
    else:
      raise ValueError("The observation type is not recognized")
  @obs.setter
  def obs(self, value):
    assert isinstance(value, str), "The observation type should be a string"
    self.__obs_type = value

  def cache(self, trajectory): 
    """
    Notes
    -----
    Use the same method to cache the weights as the parent class
    """
    super().cache(trajectory)

  def query(self, topology, coordinates, focus):
    """
    Query the coordinates and weights from a set of frames and return the feature array

    Notes
    -----
    Use the same method to query the coordinates and weights as the parent class
    """
    ret_coord, ret_weight = super().query(topology, coordinates, focus)
    return ret_coord, ret_weight
  
  def run(self, frames, weights): 
    """
    Get several frames and perform the marching observers algorithm

    Parameters
    ----------
    frames : np.array
      The coordinates of the atoms in the frames.
    weights : np.array
      The weights of the atoms in the frames.

    Returns
    -------
    ret_arr : np.array
      The feature array with the same dimensions as self.dims
    """
    # Check the frame cooridnates and weights
    if (frames.shape[0] * frames.shape[1]) == len(weights): 
      pass 
    elif frames.shape[1] == len(weights):
      weights = np.tile(weights, frames.shape[0])
    else:
      raise ValueError(f"The number of atoms in the frames is not divisible by the number of weights: {frames.shape[0] * frames.shape[1]} % {len(weights)}")

    # TODO: correct the function name and the parameters. 
    # TODO: Since there is placeholder atoms to align the dimensions of the coordinates, remove them. 
    ret_arr = commands.marching_observers( 
      frames, self.dims, 
      self.spacing, self.cutoff, 
      self.agg, self.obs
    ) 
    return ret_arr.reshape(self.dims)


class Label_RMSD(Feature): 
  def __init__(self, 
               outkey=None, outfile=None, outshape=(None,), 
               selection=None, selection_type=None, base_value=0, 
               **kwargs): 
    super().__init__(outkey=outkey, outfile=outfile, outshape = outshape, **kwargs) 
    self.selection = selection
    self.selection_type = selection_type   # "mask" or (list, tuple, np.ndarray) for atom indices
    self.base_value = float(base_value)

  def cache(self, trajectory): 
    # Cache the address of the trajectory for further query of the RMSD array based on focused points
    if isinstance(self.selection, str):
      selected = trajectory.top.select(self.selection)
    elif isinstance(self.selection, (list, tuple, np.ndarray)):
      selected = np.array([int(i) for i in self.selection])
    else: 
      raise ValueError("The selection should be either a string or a list of atom indices")
    RMSD_CUTOFF = 8
    self.refframe = trajectory[0]

    self.cached_array = pt.rmsd_nofit(trajectory, mask=selected, ref=self.refframe)
    self.cached_array = np.minimum(self.cached_array, RMSD_CUTOFF)  # Set a cutoff RMSD for limiting the z-score
    

  def query(self, topology, frames, focus): 
    # Query points near the area around the focused point
    # Major problem, How to use the cached array to query the label based on the focused point
    # Only need the topology and the frames. 
    tmptraj = pt.Trajectory(xyz=frames, top=topology)
    rmsd_arr = pt.rmsd_nofit(tmptraj, mask=self.selection, ref=self.refframe)
    z_score = (np.mean(rmsd_arr) - np.mean(self.cached_array)) / np.std(self.cached_array)   
    return z_score

  def run(self, z_score): 
    # Greater than 0 meaning RMSD is larger than the average and should apply a penalty
    # Less than 0 meaning RMSD is smaller than the average and should apply a reward
    # correction factor is 0.1 * base_value * z_score
    final_value = self.base_value - self.base_value * 0.1 * z_score
    return final_value

  def dump(self, result): 
    #Dump the results to a file
    utils.append_hdf_data(self.outfile, self.outkey, np.array([result], dtype=np.float32), dtype=np.float32, maxshape=(None,), chunks=True, compression="gzip", compression_opts=4)
    
class Label_PCDT(Feature): 
  def __init__(self, 
               outkey=None, outfile=None, outshape=(None,),
               selection=None, selection_type=None, base_value=0, 
               **kwargs): 
    super().__init__(outkey=outkey, outfile=outfile, outshape = outshape, **kwargs) 
    self.selection = selection
    self.selection_type = selection_type   # "mask" or (list, tuple, np.ndarray) for atom indices
    self.selection_counterpart = None
    self.base_value = float(base_value)

  def cache(self, trajectory): 
    # Cache the address of the trajectory for further query of the RMSD array based on focused points
    if isinstance(self.selection, str):
      selected = trajectory.top.select(self.selection)
    elif isinstance(self.selection, (list, tuple, np.ndarray)):
      selected = np.array([int(i) for i in self.selection])
    else: 
      raise ValueError("The selection should be either a string or a list of atom indices")
    RMSD_CUTOFF = 8
    self.refframe = trajectory[0]
    self.refframe.xyz[self.selection]
    self.selection_counterpart = []

    self.cached_array = utils.compute_pcdt(trajectory, mask=selected, ref=self.refframe, return_info=False)
    self.cached_array = np.minimum(self.cached_array, RMSD_CUTOFF)  # Set a cutoff distance for limiting the z-score

  def query(self, topology, frames, focus):
    tmptraj = pt.Trajectory(xyz=frames, top=topology)
    pdist_arr = utils.compute_pcdt(tmptraj, mask=self.selection, ref=self.refframe)
    pdist_mean = np.mean(pdist_arr, axis=1)
    pdist_mean_cached = np.mean(self.cached_array, axis=1)
    cosine_sim = (np.dot(pdist_mean, pdist_mean_cached) / (np.linalg.norm(pdist_mean) * np.linalg.norm(pdist_mean_cached)))
    return cosine_sim
  
  def run(self, cosine_sim): 
    # Cosine similarity between 1 and -1
    # Hence use penalty factor as 0.1 * base_value * (1 - cosine_sim)
    final_value = self.base_value - self.base_value * 0.1 * (1 - cosine_sim)
    return final_value

  def dump(self, result): 
    #Dump the results to a file
    utils.append_hdf_data(self.outfile, self.outkey, np.array([result], dtype=np.float32), dtype=np.float32, maxshape=(None,), chunks=True, compression="gzip", compression_opts=4)


class Label_ResType(Feature): 
  def __init__(self,
    outkey=None, outfile=None, outshape=(None,),
    restype="single", byres=True,
    **kwargs
  ): 
    """
    Check the labeled single/dual residue labeling based on the residue type.

    Notes
    -----
    This is the special label-type feature that returns the label based on the residue type.
    """
    super().__init__(outkey=outkey, outfile=outfile, outshape = outshape, byres=True, **kwargs)
    self.restype = restype
  
  def query(self, topology, frames, focus):
    """
    When querying the single-residue types, the topology and frames has to be cropped to the focused residue (COG of the residue). 
    Hence only the cropped residues will be retured. 
    """
    if frames.shape.__len__() == 3: 
      frames = frames[0]
    returned = super().query(topology, frames, focus)
    resnames = [i.name for i in topology.residues]
    resids = [i.resid for i in topology.atoms]

    final_resname = ""
    processed = []
    for i in range(len(returned)): 
      if returned[i] and resids[i] not in processed:
        final_resname += resnames[resids[i]]
        processed.append(resids[i])
    return (final_resname, )

  def run(self, resname):
    if self.restype == "single" and resname in constants.RES2LAB:
      retval = constants.RES2LAB[resname]
    elif self.restype == "single" and resname not in constants.RES2LAB:
      printit(f"DEBUG: The residue type {resname} is not recognized, there might be some problem in the trajectory cropping")
      retval = 0 
    elif self.restype == "dual" and resname in constants.RES2LAB_DUAL:
      retval = constants.RES2LAB_DUAL[resname]
    elif self.restype == "dual" and resname not in constants.RES2LAB_DUAL:
      printit(f"DEBUG: The residue type {resname} is not recognized, there might be some problem in the trajectory cropping")
      retval = 0
    return retval
  
  def dump(self, result): 
    #Dump the results to a file
    utils.append_hdf_data(self.outfile, self.outkey, np.array([result], dtype=np.float32), dtype=np.float32, maxshape=(None,), chunks=True, compression="gzip", compression_opts=4)



# import siesta 
import pytraj as pt
import open3d as o3d

def selection_to_mol(traj, frameidx, selection):
  atom_sel = np.asarray(selection)
  try: 
    rdmol = utils.traj_to_rdkit(traj, atom_sel, frameidx)
    if rdmol is None:
      with tempfile.NamedTemporaryFile(suffix=".pdb") as temp:
        outmask = "@"+",".join((atom_sel+1).astype(str))
        _traj = traj[outmask].copy_traj()
        pt.write_traj(temp.name, _traj, frame_indices=[frameidx], overwrite=True)
        with open(temp.name, "r") as tmp:
          print(tmp.read())
        rdmol = Chem.MolFromMol2File(temp.name, sanitize=False, removeHs=False)
    return rdmol
  except:
    return None


def viewpoint_histogram_xyzr(xyzr_arr, viewpoint, bin_nr): 
  # Generate the 
  thearray = np.asarray(xyzr_arr, dtype=np.float32)
  vertices, faces = siesta.xyzr_to_surf(thearray)  
  c_vertices = np.mean(vertices, axis=0)
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(vertices)
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.compute_vertex_normals()
  mesh.compute_triangle_normals()

  v_view = viewpoint - c_vertices
  v_view = v_view / np.linalg.norm(v_view)
  # Get the normal of each vertex
  normals = np.array(mesh.vertex_normals)
  # Get the cosine angle and split to bins 
  cos_angle = np.dot(normals, v_view)
  bins = np.linspace(-1, 1, bin_nr)
  hist, _ = np.histogram(cos_angle, bins)
  return hist / np.sum(hist)


class Viewpoint_Vectorizer(Feature):
  def __init__(self):
    super().__init__(byres=False)
    self.QUERIED_SEGMENTS = []
    self.SEGMENT_NUMBER = 6
    self.VIEW_BINNR = 10
  
  
  def hook(self, featurizer):
    super().hook(featurizer)

  def cache(self, trajectory): 
    super().cache(trajectory)

  def query(self, topology, frames, focus): 
    # Query the molecular block at the focused point? 
    # Also only needs residue-based segments
    #### Previously query the coordinates and weights from the frames 
    idx_inbox = super().query(topology, frames, focus)

    from itertools import groupby
    therange = np.arange(len(idx_inbox))
    split_indices = [list(g) for k, g in groupby(idx_inbox)]
    lens = [len(i) for i in split_indices]
    slices =[np.s_[lens[i-1]:lens[i]] if i > 0 else np.s_[:lens[i]] for i in range(len(lens))]
    arg_rank = np.argsort([np.count_nonzero(i) for i in split_indices])

    ret_arr = []
    # TODO : Check the segment number
    for i in arg_rank[:self.SEGMENT_NUMBER]:
      if arg_rank[i] > 0:
        ret_arr.append(therange[slices[i]])

    # Process the inbox status and return the segment
    viewpoint = np.array([999, 0, 0], dtype=float)

    return ret_arr, viewpoint, self.VIEW_BINNR

  def run(self, segments, viewpoint, bin_nr): 
    # Update the coordinates and weights
    # Objective: Formulate the 1D feature array
    #### Since voxelization is the main objective of the normal 3D feature, this only needs to return the 1D feature array
    ret_arr = np.full((self.SEGMENT_NUMBER, self.VIEW_BINNR), 0.0, dtype=float) 
    for seg_idx in range(self.SEGMENT_NUMBER): 
      print(f"Processing the segment {seg_idx}")
      # Put the xyzr array to the GPU code
      viewpoint_feature = viewpoint_histogram_xyzr(segments[seg_idx], viewpoint, bin_nr)
      ret_arr[seg_idx] = viewpoint_feature
    return ret_arr

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

