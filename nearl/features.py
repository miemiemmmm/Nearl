import sys, tempfile, os, subprocess, time, logging

import h5py
import numpy as np
import pytraj as pt

from . import utils, commands, constants, chemtools   # local modules 
from . import printit, config   # local static methods/objects

# TODO: 
# - Add description of each features in the docstring 

logger = logging.getLogger(__name__)

__all__ = [
  # Base class
  "Feature",

  # Static Features
  "AtomicNumber",
  "Mass",
  "HeavyAtom",
  "Aromaticity",
  "Ring",
  "Selection",
  "HBondDonor",
  "HBondAcceptor",
  "Hybridization",
  "Backbone",
  "AtomType",
  "PartialCharge",
  "Electronegativity",
  "Hydrophobicity",

  # Dynamic features
  "DynamicFeature",
  "DensityFlow",
  "MarchingObservers",

  # Label-type features
  "LabelIdentity",
  "LabelAffinity",
  "LabelPCDT",
  "LabelRMSD",

  # Other features
  # "VectorizerViewpoint", 

  # Function to get properties for dynamic features
  "cache_properties", 
  
]


# Hardcoded maps in dynamic feature preparation
SUPPORTED_FEATURES = {
  "atomic_id": 1,       "residue_id": 2,  "atomic_number": 3, "hybridization": 4,
  "mass": 11, "radius": 12, "electronegativity": 13, "hydrophobicity": 14,
  "partial_charge": 15, "uniformed" : 16, 
  "heavy_atom": 21, "aromaticity": 22, "ring": 23, "hbond_donor": 24,
  "hbond_acceptor": 25, "backboneness": 26, "sidechainness": 27,  "atom_type": 28
}


# Hardcoded maps in the C++ level code
SUPPORTED_AGGREGATION = {
  "mean": 1,
  "standard_deviation": 2,
  "median": 3,
  "variance": 4,
  "max": 5,
  "min": 6,
  "information_entropy": 7, 
  "drift": 8,
}


# Hardcoded maps in the C++ level code
SUPPORTED_OBSERVATION = {
  "existence": 1,
  "direct_count": 2,
  "distinct_count": 3,
  "mean_distance": 11,
  "cumulative_weight": 12,
  "density": 13,
  "dispersion": 14,
  "eccentricity": 15,
  "radius_of_gyration": 16
}


def crop(points, upperbound, padding, spacing):
  """
  Crop the points to the box defined by the center and lengths. 

  Parameters
  ----------
  points : np.ndarray
    The coordinates of the atoms
  upperbound : np.ndarray
    The upperbound of the box
  padding : float
    The padding of the box
  spacing : float
    The spacing of the box for half grid offset

  Returns
  -------
  mask_inbox : np.ndarray
    The boolean mask of the atoms within the box
  """
  # X within the bouding box
  x_state_0 = points[:, 0] < upperbound[0] + padding - spacing/2
  x_state_1 = points[:, 0] > 0 - padding - spacing/2
  # Y within the bouding box
  y_state_0 = points[:, 1] < upperbound[1] + padding - spacing/2
  y_state_1 = points[:, 1] > 0 - padding - spacing/2
  # Z within the bouding box
  z_state_0 = points[:, 2] < upperbound[2] + padding - spacing/2
  z_state_1 = points[:, 2] > 0 - padding - spacing/2
  # All states
  mask_inbox = np.array(x_state_0 * x_state_1 * y_state_0 * y_state_1 * z_state_0 * z_state_1, dtype=bool)
  return mask_inbox


def selection_to_mol(traj, frameidx, selection):
  from rdkit import Chem
  atom_sel = np.array(selection)
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


class Feature:
  """
  Base class for the feature generator

  Attributes
  ----------
  dims : np.ndarray
    The dimensions of the grid
  spacing : float
    The spacing of the grid
  lengths : np.ndarray
    The lengths of the grid
  cutoff : float
    The cutoff distance for voxelization or marching observers
  padding : float 
    The padding of the box when query the atoms within the grid
  byres : bool 
    The boolean flag to get the residues within the bounding box (default is by atoms)
  outfile : str, path-like
    The path of HDF file to store the result features
  outkey : str
    To which key/tag the result features will be dumped 
  sigma : float
    The sigma of the Gaussian-based voxelization, applies to static and PDF features 

  Methods
  -------
  hook(featurizer)
    Hook the feature generator back to the feature convolutor and obtain necessary attributes from the featurizer
  cache(trajectory)
    Cache the needed weights for each atom in the trajectory for further Feature.query function
  query(topology, frame_coords, focal_point)
    Query the atoms and weights within the bounding box
  run(coords, weights)
    Voxelize the coordinates and weights by default 
  dump(result)
    Dump the result feature to an HDF5 file, Feature.outfile and Feature.outkey should be set in its child class (Either via __init__ or hook function)

  Notes
  -----
  The input and the output of the query, run and dump function should be chained together.
  
  """
  
  # Individual parameters
  # - outshape: The shape of the output array
  # - force_recache: The boolean flag to force recache the weights
  def __init__(self, 
    dims=None, spacing=None, 
    outfile=None, outkey=None,
    cutoff=None, sigma=None,
    padding=None, byres=None, 
    outshape=None, force_recache=None,
    selection=None, translate="origin", 
    frame_offset=None,
    **kwargs
  ):
    # Fundamental variables need setter callback 
    self.__dims = None
    self.__spacing = None
    if dims is not None:
      self.dims = dims
    if spacing is not None:
      self.spacing = spacing
    self.__center = None
    self.__lengths = None
    self.__padding = padding # cutoff if padding is None else padding 
    if padding is not None:
      self.padding = padding

    # General variables for featurization process 
    self.cutoff = cutoff
    self.sigma = sigma
    self.outfile = outfile
    self.byres = byres
    self.cached_array = None
    self.__frame_offset = frame_offset

    # To be used in .query() function
    self.selection = selection
    self.selected = None
    self.classname = self.__class__.__name__
    self.force_recache = False if force_recache is None else True
    if self.force_recache:
      printit(f"{self.classname}: Forcing recache the weights")
    self.translate = translate
    if self.translate == "center": 
      printit(f"{self.classname}: Will translate the coordinates of focused substructure to center of the bounding box. ")
    elif self.translate == "origin":
      printit(f"{self.classname}: Will translate the coordinates of focused substructure to align the origin of the bounding box. ")
    else: 
      logger.warning(f"{self.classname}: The translate parameter is not recognized. ") 
    
    # To be used in .dump() function
    self.outkey = outkey        # Has to be specific to the feature
    self.outshape = outshape 
    self.hdf_compress_level = kwargs.get("hdf_compress_level", 0)
    self.hdf_dump_opts = {}

    self.PARAMSPACE = {
      "dims": self.dims, 
      "spacing": self.spacing, 
      "cutoff": self.cutoff, 
      "sigma": self.sigma,
      "outfile": self.outfile, 
      "padding": self.padding, 
      "byres": self.byres, 
      "selection": self.selection,
      "translate": self.translate,
      "force_recache": self.force_recache,
      "outshape": self.outshape,
      "HDF_OPTS": self.hdf_dump_opts
    }

  def __str__(self, detailed=False):
    ret_str = f"{self.classname} " 
    if detailed:
      keys = list(self.PARAMSPACE.keys()) 
      for k,val in self.PARAMSPACE.items():
        if k == keys[-1]:
          ret_str += f"{k}:{val}"
        else:
          ret_str += f"{k}:{val}|"
    else: 
      ret_str += f"<dims:{self.dims}|lengths:{self.__lengths}|spacing:{self.spacing}|cutoff:{self.cutoff}|sigma:{self.sigma}|selection:{self.selection}>" 
    return ret_str

  @property
  def params(self): 
    """
    Obtain the parameters of the feature object
    """
    return self.PARAMSPACE

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
      logger.warning(f"{self.classname} The dimension should be a number, list, tuple or a numpy array, not {type(value)}")
      self.__dims = None
    if self.__spacing is not None:
      self.__center = self.lengths / 2
    if self.__dims is not None and self.__spacing is not None:
      self.__lengths = self.dims * self.__spacing

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
      logger.warning(f"{self.classname}: The spacing is not a valid number, setting to None") 
      self.__spacing = None
    if self.__dims is not None and self.__spacing is not None:
      self.__center = np.array(self.__dims * self.spacing, dtype=np.float32) / 2
    if self.__dims is not None and self.__spacing is not None:
      self.__lengths = self.dims * self.__spacing

  @property
  def center(self):
    """
    Center of the grid, read-only property calculated by dims and spacing
    """
    return self.__center
  @property
  def lengths(self):
    """
    Lengths of the grid. 
    """
    return self.__lengths

  @property
  def padding(self):
    """
    The padding of the box when query the atoms within the grid. The default padding is set to be the cutoff distance. 
    """
    return self.__padding
  @padding.setter
  def padding(self, value):
    self.__padding = float(value)

  @property
  def frame_offset(self): 
    """
    The offset of the frame to be used in each frame-slice. This property only applies for static features. 
    """
    return self.__frame_offset
  @frame_offset.setter
  def frame_offset(self, value):
    self.__frame_offset = int(value)

  def hook(self, featurizer): 
    """
    Hook the feature generator back to the feature convolutor and obtain necessary attributes from the featurizer
    including the trajectory, active frame, convolution kernel etc

    Parameters
    ----------
    featurizer : nearl.featurizer.Featurizer
      The featurizer object describing the feature generation process

    Notes
    -----
    If the following attributes are not set manually, hook function will try to inherit them from the featurizer object: 
    sigma, cutoff, outfile, outkey, padding, byres
    """
    printit(f"{self}: Hooking feature ({self.__str__()}) to the featurizer. ")
    if self.dims is None and featurizer.dims is not None:
      self.dims = featurizer.dims
    if self.spacing is None and featurizer.spacing is not None:
      self.spacing = featurizer.spacing
    # Update this upon adding more variables to the feature
    for key in constants.COMMON_FEATURE_PARMS:
      if getattr(self, key, None) is None: 
        # If the variable is not manually set, try to inherit the attributes from the featurizer
        if key in featurizer.FEATURE_PARMS.keys() and featurizer.FEATURE_PARMS[key] is not None:
          keyval = featurizer.FEATURE_PARMS[key]
          setattr(self, key, keyval)
          printit(f"{self}: Inheriting the parameter from the featurizer: {key} {keyval}")
          self.PARAMSPACE[key] = keyval
    
    # Setting coupled parameters or defaults if not set manually 
    if self.padding is None: 
      logger.warning(f"{self}: The padding is not set, setting to cutoff")
      self.padding = self.cutoff 

    if self.frame_offset is None: 
      logger.info(f"{self}: Setting the frame offset to 0")
      self.frame_offset = 0

    if getattr(self, "hdf_compress_level", 0):
      self.hdf_dump_opts["compression_opts"] = self.hdf_compress_level
      self.hdf_dump_opts["compression"] = "gzip"
      self.PARAMSPACE["HDF_OPTS"] = {
        "compression_opts": self.hdf_compress_level, 
        "compression": "gzip"
      }
    

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
    if self.selection is None:
      # If the selection is not set, select all the atoms
      self.selected = np.full(len(self.atomic_numbers), True, dtype=bool)
    elif isinstance(self.selection, str):
      # If the selection is a string, select the atoms based on the selection string
      selected = trajectory.top.select(self.selection)
      self.selected = np.full(len(self.atomic_numbers), False, dtype=bool)
      self.selected[selected] = True
      printit(f"{self}: Selected {np.count_nonzero(self.selected)} atoms based on the selection string")
    elif isinstance(self.selection, (list, tuple, np.ndarray)):
      if len(self.selection) != len(self.atomic_numbers):
        logger.warning(f"{self.classname}: The length of selection array does not match the number of atoms {trajectory.n_atoms} in the topology. ")
      selected = np.array(self.selection)
      self.selected = np.full(len(self.atomic_numbers), False, dtype=bool)
      self.selected[selected] = True
      printit(f"{self}: Selected {np.count_nonzero(self.selected)} atoms based on the selection string")


  def query(self, topology, frame_coords, focal_point):
    """
    Base function to query the coordinates within the the bounding box near the focal point and translate the coordinates to the center of the box

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
      The boolean mask of the atoms within the bounding box; shape=(n_atoms,)
    final_coords : np.ndarray 
      The coordinates within the bounding box; shape=(queried_n_atoms, 3)

    Notes
    -----
    The frame coordinates are explicitly tranlated to match the focued part to the center of the box. 
    """
    assert focal_point.shape.__len__() == 1, f"The focal point should be a 1D array with length 3, rather than the given {focal_point.shape}"
    
    if (len(self.resids) != topology.n_atoms) or self.force_recache: 
      logger.info(f"{self}: Dealing with inhomogeneous topology")
      if len(frame_coords.shape) == 2:
        self.cache(pt.Trajectory(xyz=np.array([frame_coords]), top=topology))
      else:
        self.cache(pt.Trajectory(xyz=frame_coords, top=topology))

    if self.center is None or self.lengths is None or self.padding is None:
      logger.warning(f"{self} Skipping the coordinates cropping due to the missing center, lengths or padding information") 
      return np.full(topology.n_atoms, True, dtype=bool), frame_coords
    else: 
      # Align the coordinates to the center of the bounding box (with focal point being the center) 
      frame_coords = frame_coords - focal_point + self.center - self.spacing/2
      mask = crop(frame_coords, self.lengths, self.padding, self.spacing)
      
      if np.count_nonzero(mask) == 0:
        logger.warning(f"{self}: Did not find atoms in the bounding box near the focal point {np.round(focal_point,1)}")

      # Get the boolean array of residues within the bounding box
      if self.byres:
        res_inbox = np.unique(self.resids[mask])
        final_mask = np.full(len(self.resids), False)
        for res in res_inbox:
          final_mask[np.where(self.resids == res)] = True
      else: 
        final_mask = mask
      # Apply the selected atoms
      if self.selection is not None:
        final_mask = final_mask * self.selected
      final_coords = np.ascontiguousarray(frame_coords[final_mask])
      logger.debug(f"Returned {np.count_nonzero(final_mask)}; Selected {np.count_nonzero(self.selected)}; Total {len(final_mask)}. ")
      return final_mask, final_coords

  def run(self, coords, weights):
    """
    By default voxelize the coordinates and weights 

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
    if len(coords) == 0:
      logger.warning(f"{self.classname}: The coordinates from query method are empty, returning an all-zero array") 
      return np.zeros(self.dims, dtype=np.float32)
    st = time.perf_counter()
    ret = commands.frame_voxelize(coords, weights, self.dims, self.spacing, self.cutoff, self.sigma)
    print(f"{'Timing_STAT':15} {time.perf_counter() - st:10f} seconds for {coords.shape[0]} atoms")  # TODO
    return ret

  def dump(self, result):
    """
    Dump the result feature to an HDF5 file, Feature.outfile and Feature.outkey should be set in its child class (Either via __init__ or hook function)

    Parameters
    ----------
    results : np.array
      The result feature array
    """
    logger.debug(f"{self.classname}: Dumping the result to the HDF5 file {self.outfile} with the key {self.outkey}") 
    if ("outfile" in dir(self)) and ("outkey" in dir(self)) and (len(self.outfile) > 0):
      if np.isnan(result).sum() > 0:
        logger.warning(f"{self.classname}: Found {np.isnan(result).sum()} NaN values in the result array")
        result = np.nan_to_num(result).astype(result.dtype)     # Replace NaN with 0
      if self.outshape is not None: 
        # Output shape is explicitly set (Usually heterogeneous data like coordinates)
        utils.append_hdf_data(self.outfile, self.outkey, np.array([result], dtype=np.float32), dtype=np.float32, maxshape=self.outshape, chunks=True, compression="gzip", compression_opts=self.hdf_compress_level)
      elif len(self.dims) == 3: 
        # For homogeneous features, set the chunks to match their actual shape 
        if self.hdf_compress_level == 0: 
          utils.append_hdf_data(self.outfile, self.outkey, np.array([result], dtype=np.float32), dtype=np.float32, maxshape=(None, *self.dims), chunks=(1, *self.dims))
        else: 
          utils.append_hdf_data(self.outfile, self.outkey, np.array([result], dtype=np.float32), dtype=np.float32, maxshape=(None, *self.dims), chunks=(1, *self.dims), **self.hdf_dump_opts)
    else: 
      logger.warning(f"{self.classname}: The outfile and/or outkey are not set, the result is not dumped into file. ")
      


class AtomicNumber(Feature):
  """
  Annotate each atoms with their atomic number. 
  """
  def query(self, topology, frame_coords, focal_point): 
    """
    Query the atomic coordinates and the atomic numbers as weights within the bounding box

    Notes
    -----
    By default, a slice of frame coordinates is passed to the querier function (typically 3 dimension shaped by [frames_number, atom_number, 3])
    However, for static feature, only one frame is needed. 
    Hence, the querier function by default uses the first frame, Set the frame_offset to change this behavior. 

    The focused part of the coords needs to be translated to the center of the box before sending to the runner function.
    """
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    self.cached_array = np.array(self.atomic_numbers, dtype=np.float32)
    weights = self.cached_array[idx_inbox]
    logger.info(f"{self.classname}: Found {len(weights)} atoms in the bounding box and total weight is {np.sum(weights)}")
    return coord_inbox, weights


class Mass(Feature):
  """
  Annotate each atoms with their atomic mass. 
  """
  def cache(self, trajectory):
    super().cache(trajectory)
    self.cached_array = np.array([constants.ATOMICMASS[i] for i in self.atomic_numbers], dtype=np.float32)

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
      frame_coords = frame_coords[self.frame_offset]
    
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]
    logger.debug(f"Query: Sum of weights: {np.sum(weights)}, {weights.shape}, {weights[:5]}, {np.mean(weights)}") 
    return coord_inbox, weights


class HeavyAtom(Feature):
  """
  Annotate each atoms as heavy atom or not (heavy atoms are encoded 1, otherwise 0). 

  Parameters
  ----------
  default_weight : int, default 1
    The default weight of the heavy atoms. 
  """
  def __init__(self, default_weight=1, **kwargs):
    super().__init__(**kwargs)
    self.default_weight = default_weight

  def cache(self, trajectory):
    """
    Prepare the heavy atom weights
    """
    super().cache(trajectory)
    self.cached_array = np.full(len(self.resids), 0, dtype=np.float32)
    self.cached_array[np.where(self.atomic_numbers > 1)] = self.default_weight

  def query(self, topology, frame_coords, focal_point): 
    """
    Get the atoms and weights within the bounding box
    """
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset] 
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]
    return coord_inbox, weights


class Radius(Feature):
  def cache(self, trajectory):
    super().cache(trajectory)
    self.cached_array = np.array([constants.ATOMICRADIUS[i] for i in self.atomic_numbers], dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]
    return coord_inbox, weights


class Aromaticity(Feature):
  """
  Annotate each atoms as aromatic atom or not (aromatic atoms are encoded 1, otherwise 0). 

  Parameters
  ----------
  reverse : bool, default False
    Set to True if you want to get the non-aromatic atoms to be 1; The aromaticity is calculated by OpenBabel. 
  """
  def __init__(self, reverse=None, **kwargs): 
    super().__init__(**kwargs)
    self.reverse = reverse 

  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_aromatic = chemtools.label_aromaticity(fpt.name)

    if len(atoms_aromatic) != trajectory.n_atoms:
      logger.warning(f"{self.classname}: The length of feature array {len(atoms_aromatic)} does not match the number of atoms {trajectory.n_atoms} in the topology. ") 

    if self.reverse: 
      self.cached_array = np.array([1 if i == 0 else 0 for i in atoms_aromatic], dtype=np.float32)
    else: 
      self.cached_array = np.array(atoms_aromatic, dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]
    return coord_inbox, weights
    

class Ring(Feature):
  """
  Annotate each atoms as ring atom or not (ring atoms are encoded 1, otherwise 0). 

  Parameters
  ----------
  reverse : bool, default False
    Set to True if you want to get the non-ring atoms to be 1; The ring status is calculated by OpenBabel. 
  """
  def __init__(self, reverse=False, **kwargs):
    super().__init__(**kwargs)
    # Set reverse to True if you want to get the non-ring atoms to be 1
    self.reverse = reverse 

  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_in_ring = chemtools.label_ring_status(fpt.name)

    if len(atoms_in_ring) != trajectory.n_atoms:
      logger.warning(f"{self.classname}: The length of feature array {len(atoms_in_ring)} does not match the number of atoms {trajectory.n_atoms} in the topology. ") 

    if self.reverse: 
      self.cached_array = np.array([1 if i == 0 else 0 for i in atoms_in_ring], dtype=np.float32)
    else: 
      self.cached_array = np.array(atoms_in_ring, dtype=np.float32)
  
  def query(self, topology, frame_coords, focal_point): 
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]
    return coord_inbox, weights


class Selection(Feature):
  """
  Annotate each atoms with the selection of the user (selected atoms are encoded 1, otherwise 0). 
  
  Parameters
  ----------
  default_value : int, default 1
    The default value of the weights for the selected atoms.

  Notes
  -----
  The argument `selection` is required to be set in the constructor. 
  """
  def __init__(self, default_value=1, **kwargs):
    if "selection" not in kwargs.keys():
      raise ValueError(f"{self.classname}: The selection parameter should be set")
    super().__init__(**kwargs)
    self.default_value = default_value

  def cache(self, trajectory):
    super().cache(trajectory)
    self.cached_array = np.full(trajectory.top.n_atoms, 0, dtype=np.float32)
    self.cached_array[self.selected] = self.default_value

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox] 
    assert len(weights) == len(coord_inbox), f"{self.classname}: The length of weights {len(weights)} does not match the length of coordinates {len(coord_inbox)}"
    return coord_inbox, weights
  

class HBondDonor(Feature):
  """
  Annotate each atoms with the hydrogen bond donor atoms (donor atoms are encoded 1, otherwise 0).
  """
  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_hbond_donor = chemtools.label_hbond_donor(fpt.name)

    if len(atoms_hbond_donor) != trajectory.n_atoms:
      logger.warning(f"{self.classname}: The length of feature array {len(atoms_hbond_donor)} does not match the number of atoms {trajectory.n_atoms} in the topology. ") 
    
    self.cached_array = np.array(atoms_hbond_donor, dtype=np.float32)
  
  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]
    return coord_inbox, weights


class HBondAcceptor(Feature):
  """
  Annotate each atoms with the hydrogen bond acceptor atoms (acceptor atoms are encoded 1, otherwise 0). 
  """
  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_hbond_acceptor = chemtools.label_hbond_acceptor(fpt.name)

    if len(atoms_hbond_acceptor) != trajectory.n_atoms:
      logger.warning(f"{self.classname}: The length of feature array {len(atoms_hbond_acceptor)} does not match the number of atoms {trajectory.n_atoms} in the topology. ") 

    self.cached_array = np.array(atoms_hbond_acceptor, dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]
    return coord_inbox, weights


class Hybridization(Feature):
  """
  Annotate each atoms with the hybridization of the atoms (integer range from 0 to 3).
  """
  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_hybridization = chemtools.label_hybridization(fpt.name)

    if len(atoms_hybridization) != trajectory.n_atoms:
      logger.warning(f"{self.classname}: The length of feature array {len(atoms_hybridization)} does not match the number of atoms {trajectory.n_atoms} in the topology. ") 

    self.cached_array = np.asarray(atoms_hybridization, dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]
    return coord_inbox, weights


class Backbone(Feature):
  """
  Annotate each atoms with the backbone atoms (backbone atoms are encoded 1, otherwise 0). 

  Parameters
  ----------
  reverse : bool, default False
    Set to True if you want to get the non-backbone atoms (sidechain) to be 1 
  
  Notes
  -----
  Backbone atoms are defined as the atoms with the name "C", "O", "CA", "HA", "N" and "HN".
  The reverse parameter can be set to True to get the non-backbone atoms (sidechain) to be 1.
  """
  def __init__(self, reverse=False, **kwargs):
    super().__init__(**kwargs)
    self.reverse = reverse

  def cache(self, trajectory):
    super().cache(trajectory)
    backboneness = [1 if i.name in ["C", "O", "CA", "HA", "N", "HN"] else 0 for i in trajectory.top.atoms]
    if self.reverse: 
      self.cached_array = np.array([1 if i == 0 else 0 for i in backboneness], dtype=np.float32)
    else: 
      self.cached_array = np.array(backboneness, dtype=np.float32)
    
  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]
    return coord_inbox, weights
  

class AtomType(Feature):
  """
  Annotate each atoms as the selected atom type (focus_element) or not (selected atoms are encoded 1, otherwise 0).

  Parameters
  ----------
  focus_element : int
    The atomic number of the atom type to be selected. 
  """
  def __init__(self, focus_element, **kwargs):
    super().__init__(**kwargs)
    self.focus_element = int(focus_element)

  def __str__(self):
    ret_str = f"{self.classname} <focus:{constants.ATOMICNR2SYMBOL[self.focus_element]}>" 
    return ret_str

  def cache(self, trajectory):
    super().cache(trajectory)
    self.cached_array = np.full(trajectory.top.n_atoms, 0, dtype=np.float32)
    self.cached_array[np.where(self.atomic_numbers == self.focus_element)] = 1

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]

    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]

    if np.sum(weights) == 0:
      logger.warning(f"{self.classname}: No atoms of the type {self.focus_element} is found in the bounding box")
    
    return coord_inbox, weights


class PartialCharge(Feature):
  """
  Annotate each atom with their partial charge. The charges could be derived from multiple sources, including its own topology, manually set, external functions, or recomputed using ChargeFW2. 

  Parameters
  ----------
  charge_type : str, "topology" by default 
    The supported charge types can be "topology", "manual", "chargefw" and "external". 
  charge_parm : str
    The charge parameter. 
    If "topology", Nearl will refer to the charge values in the topology and this parameter will be ignored. 
    In case of "manual", the charge_parm should be a dictionary with its trajectory identity as the key and the charge values as the value. 
    The "external" type allows the user to pass the reference to an external function to calculate the charge values. 
    In case of "chargefw", the charge_parm is the name of the charge method to be used in `ChargeFW2 <https://github.com/sb-ncbr/ChargeFW2>`_. Note that the computation of charge could be very expensive depending on the size of the structure. For more information about the charge types and parameters, please refer to its documentation.  
  """
  def __init__(self, charge_type="topology", charge_parm=None, keep_sign="both", **kwargs):
    super().__init__(**kwargs)
    self.charge_type = charge_type
    self.charge_parm = charge_parm
    if charge_type not in ["topology", "manual", "chargefw", "external"]: 
      raise ValueError(f"{self.classname}: The charge type should be either 'topology', 'manual', 'chargefw' or 'external' rather than {charge_type}")
    if keep_sign not in ["positive", "negative", "both", "p", "n", "b"]:
      raise ValueError(f"{self.classname}: The keep_sign parameter should be either positive/p, negative/n or both/b rather than {keep_sign}")

    if keep_sign in ["positive", "p"]:
      self.keep_sign = "positive"
    elif keep_sign in ["negative", "n"]:
      self.keep_sign = "negative"
    elif keep_sign in ["both", "b"]:
      self.keep_sign = "both"

  def __str__(self):
    ret_str = f"{self.classname} <type:{self.charge_type}|sign:{self.keep_sign}>"
    return ret_str

  def cache(self, trajectory):
    super().cache(trajectory)
    if self.charge_type == "topology":
      if np.allclose(trajectory.top.charge, 0): 
        raise ValueError(f"{self.classname}: The charge values in the topology are all zero meaning the topology of the trajectory does not contain necessary charge information. ")
      else: 
        self.cached_array = trajectory.top.charge

    elif self.charge_type == "manual": 
      # If the charge type is manual, the charge values should be set manually
      assert len(self.charge_parm) == trajectory.top.n_atoms, "The number of charge values does not match the number of atoms in the trajectory"
      self.cached_array = self.charge_parm[trajectory.identity]
      self.cached_array = np.array(self.cached_array, dtype=np.float32)

    elif self.charge_type == "external": 
      # Use external function to calculate the charge values
      self.cached_array = self.charge_parm(trajectory)

    elif self.charge_type == "chargefw":
      # Otherwise, compute the charges using the ChargeFW2
      # The following types are supported by ChargeFW: 
      # "sqeqp",  "eem",  "abeem",  "sfkeem",  "qeq", "smpqeq",  "eqeq",  "eqeqc",  "delre",  "peoe", 
      # "mpeoe",  "gdac",  "sqe",  "sqeq0",  "mgc", "kcm",  "denr",  "tsef",  "charge2",  "veem", "formal", 
      try: 
        import chargefw2_python as cfw
      except ImportError as e:
        raise ImportError(f"{self.classname}: The ChargeFW2 is not installed. Check the https://github.com/sb-ncbr/ChargeFW2 for more information. ") from e
      charges = None
      charge_values = None
      mol = None
      with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
        pt.write_traj(f.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
        filename = os.path.basename(f.name).split(".")[0]

        # Step 1: Load the molecule
        try: 
          mol = cfw.Molecules(f.name)
        except Exception as e: 
          logger.warning(f"{self.classname}: Failed to load the molecule because of the following error:\n{e}")
        
        # Step 2: Calculate the charges
        try: 
          charges = cfw.calculate_charges(mol, self.charge_type, self.charge_parm)
          if filename not in charges.keys():
            raise ValueError(f"{self.classname}: The charges are not calculated for the file {filename}")
          else: 
            charge_values = np.array(charges[filename], dtype=np.float32)
            if len(charge_values) != trajectory.n_atoms:
              raise ValueError(f"{self.classname}: The number of charge values does not match the number of atoms in the trajectory")
            else: 
              self.cached_array = charge_values
        except Exception as e: 
          # Step 3 (optional): If the default charge method fails, try alternative methods
          subprocess.call(["cp", f.name, "/tmp/chargefailed.pdb"])
          logger.warning(f"{self.classname}: Default charge method {self.charge_type} failed due to the following error:\n{e}") 
          for method, parmfile in cfw.get_suitable_methods(mol): 
            if method == "formal":
              continue
            if len(parmfile) == 0:
              parmfile = ""
            else: 
              parmfile = parmfile[0].split(".")[0]
            if len(parmfile) > 0:
              printit(f"{self.classname}: Trying alternative charge methods {method} with {parmfile} parameter")
            else: 
              printit(f"{self.classname}: Trying alternative charge methods {method} without parameter file")
            try:
              charges = cfw.calculate_charges(mol, method, parmfile)
              if filename not in charges.keys():
                continue
              else:
                charge_values = np.array(charges[filename], dtype=np.float32)
                if len(charge_values) != trajectory.n_atoms:
                  continue
                else: 
                  printit(f"{self.classname}: Finished the charge computation with {method} method" + (" without parameter file" if len(parmfile) == 0 else " with parameter file"))
                  self.cached_array = charge_values
                  break
            except Exception as e:
              logger.warning(f"{self.classname}: Failed to calculate molecular charge (Alternative charge types) because of the following error:\n {e}")
              continue

      if charges is None or charge_values is None:
        logger.warning(f"{self.classname}: The charge computation fails. Setting all charge values to 0. ", file=sys.stderr)
        self.cached_array = np.zeros(trajectory.n_atoms)
    
    assert self.cached_array is not None, f"{self.classname}: The charge values are not set. Please check the charge type and parameters. " 
    assert len(self.cached_array) == trajectory.n_atoms, f"{self.classname}: The number of charge values does not match the number of atoms in the trajectory. " 

    # Final check of the needed sign of the charge values
    if self.keep_sign in ["positive", "p"]:
      self.cached_array = np.maximum(self.cached_array, 0)
    elif self.keep_sign in ["negative", "n"]:
      self.cached_array = np.minimum(self.cached_array, 0)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox] 
    return coord_inbox, weights


class Electronegativity(Feature):
  """
  Annotate each atoms with the electronegativity of the atoms. The electronegativity is from the https://periodictable.com/.
  """
  def cache(self, trajectory):
    super().cache(trajectory)
    self.cached_array = np.array([constants.ELECNEG[i] for i in self.atomic_numbers], dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]
    return coord_inbox, weights
  

class Hydrophobicity(Feature):
  """
  Annotate each atoms with the hydrophobicity of the atoms.

  Notes
  -----
  The hydrophobicity is calculated by the absolute difference between the electronegativity of the atom and the electronegativity of the carbon atom. 
  """
  def cache(self, trajectory):
    super().cache(trajectory)
    elecnegs = np.array([constants.ELECNEG[i] for i in self.atomic_numbers], dtype=np.float32)
    self.cached_array = np.abs(elecnegs - constants.ELECNEG[6])
  
  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[self.frame_offset]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    weights = self.cached_array[idx_inbox]
    return coord_inbox, weights


###############################################################################
# Label-Tyep Features
###############################################################################
def cache_properties(trajectory, property_type, **kwargs): 
  """
  Cache the required atomic properties for the trajectory (called in the :func:`nearl.features.DynamicFeature.cache`). 
  Re-implement the previously cached properties in the Feature classes
  
  Notes
  -----
  .. note::

    **Direct count-based weights**:

    +------------------------+------------------+
    | Property Name          | Property Type    |
    +========================+==================+
    | atomic_id              | int              |
    +------------------------+------------------+
    | residue_id             | int              |
    +------------------------+------------------+

    **Atom properties-based weights**:

    +------------------------+------------------+
    | Property Name          | Property Type    |
    +========================+==================+
    | atomic_number          | int              |
    +------------------------+------------------+
    | hybridization          | int              |
    +------------------------+------------------+
    | mass                   | float            |
    +------------------------+------------------+
    | radius                 | float            |
    +------------------------+------------------+
    | electronegativity      | float            |
    +------------------------+------------------+
    | hydrophobicity         | float            |
    +------------------------+------------------+
    | partial_charge         | float            |
    +------------------------+------------------+
    | uniformed              | float            |
    +------------------------+------------------+
    | heavy_atom             | boolean          |
    +------------------------+------------------+
    | aromaticity            | boolean          |
    +------------------------+------------------+
    | ring                   | boolean          |
    +------------------------+------------------+
    | hbond_donor            | boolean          |
    +------------------------+------------------+
    | hbond_acceptor         | boolean          |
    +------------------------+------------------+
    | sidechainness          | boolean          |
    +------------------------+------------------+
    | backboneness           | boolean          |
    +------------------------+------------------+
    | atom_type              | boolean          |
    +------------------------+------------------+
  
    The **atom_type** needs the extra argument **element_type** (Atomic number of the element of interest). 
  """
  
  atoms = [i for i in trajectory.top.atoms]
  atom_numbers = np.array([i.atomic_number for i in atoms], dtype=int)
    
  if property_type == 1:  
    # Atomic ID 
    cached_arr = np.array([i.index for i in atoms], dtype=np.float32)

  elif property_type == 2:  
    # Residue ID 
    cached_arr = np.array([i.resid for i in atoms], dtype=np.float32)
    
  elif property_type == 3: 
    # Atomic number
    cached_arr = np.array([i.atomic_number for i in atoms], dtype=np.float32)

  elif property_type == 4:
    # Hybridization 
    # Directly borrow the Hybridization feature class
    tmp_feat = Hybridization()
    tmp_feat.cache(trajectory)
    cached_arr = tmp_feat.atoms_hybridization

  elif property_type == 11:
    # Mass
    cached_arr = np.array([constants.ATOMICMASS[i] for i in atom_numbers], dtype=np.float32)

  elif property_type == 12:
    # Radius
    cached_arr = np.array([utils.VDWRADII[str(i)] for i in atom_numbers], dtype=np.float32)

  elif property_type == 13:
    # Electronegativity
    cached_arr = np.array([constants.ELECNEG[i] for i in atom_numbers], dtype=np.float32)

  elif property_type == 14:
    # Hydrophobicity
    elecnegs = np.array([constants.ELECNEG[i] for i in atom_numbers], dtype=np.float32)
    cached_arr = np.abs(elecnegs - constants.ELECNEG[6])

  elif property_type == 15:
    # Partial charge
    tmp_feat = PartialCharge()
    tmp_feat.cache(trajectory)
    cached_arr = tmp_feat.charge_values

  elif property_type == 16:
    # Manual uniformed weight
    weight_val = kwargs.get("manual_weight", 1)
    cached_arr = np.full(len(atom_numbers), weight_val, dtype=np.float32)

  elif property_type == 21:
    # Heavy atom
    cached_arr = np.full(len(atom_numbers), 0, dtype=np.float32)
    cached_arr[np.where(atom_numbers > 1)] = 1

  elif property_type == 22:
    # Aromaticity
    tmp_feat = Aromaticity()
    tmp_feat.cache(trajectory)
    cached_arr = tmp_feat.atoms_aromatic

  elif property_type == 23:
    # Ring
    tmp_feat = Ring()
    tmp_feat.cache(trajectory)
    cached_arr = tmp_feat.atoms_in_ring

  elif property_type == 24:
    # HBondDonor
    tmp_feat = HBondDonor()
    tmp_feat.cache(trajectory)
    cached_arr = tmp_feat.atoms_hbond_donor

  elif property_type == 25:
    # HBondAcceptor
    tmp_feat = HBondAcceptor()
    tmp_feat.cache(trajectory)
    cached_arr = tmp_feat.atoms_hbond_acceptor

  elif property_type == 26:
    # Backboneness 
    backboneness = [1 if i.name in ["C", "O", "CA", "HA", "N", "HN"] else 0 for i in atoms]
    cached_arr = np.array([1 if i == 0 else 0 for i in backboneness], dtype=np.float32)

  elif property_type == 27:
    # Sidechainness 
    sidechainness = [0 if i.name in ["C", "O", "CA", "HA", "N", "HN"] else 1 for i in atoms]
    cached_arr = np.array(sidechainness, dtype=np.float32)

  elif property_type == 28:
    # Atom type 
    # Needs the focus_element to be set in the kwargs
    if "element_type" not in kwargs.keys():
      raise ValueError("The focus element should be set for the atom type")
    focus_element = kwargs["element_type"]  
    cached_arr = np.array([1 if i == focus_element else 0 for i in atom_numbers], dtype=np.float32)
  
  return np.array(cached_arr, dtype=np.float32)


class DynamicFeature(Feature):
  """
  Base class for the dynamic features. 
  
  Parameters
  ----------
  weight_type : str, default="mass"
    The weight type for the dynamic feature. Check :func:`nearl.features.cache_properties` for the available weight types
  agg : str, default="mean"
    The aggregation function for the dynamic feature. 

  Notes
  -----
  Visit this function :func:`nearl.features.cache_properties` to get the available properties for the dynamic features.

  After processing each frames, an aggregation function is applied to the [F,D,D,D] tensor to reduce the tensor to a [D,D,D] tensor.
  The following aggregation functions are supported:

  +------------------------+------------------+
  | Aggregation Type       | Aggregation Type |
  +========================+==================+
  | mean                   | 1                |
  +------------------------+------------------+
  | standard_deviation     | 2                |
  +------------------------+------------------+
  | median                 | 3                |
  +------------------------+------------------+
  | variance               | 4                |
  +------------------------+------------------+
  | max                    | 5                |
  +------------------------+------------------+
  | min                    | 6                |
  +------------------------+------------------+
  | information_entropy    | 7                |
  +------------------------+------------------+
  | drift                  | 8                |
  +------------------------+------------------+
    
  """

  def __init__(self, agg="mean", weight_type="mass", **kwargs):
    super().__init__(**kwargs)
    self._agg_type = agg
    self._weight_type = weight_type
    # Need to manually handle the kwargs for specific features instances
    self.feature_args = kwargs
    self.MAX_ALLOWED_ATOMS = 9999     # Maximum allowed atoms in the box 
    self.DEFAULT_COORD = 99999.0

  def __str__(self):
    ret = f"{self.classname} <agg:{self._agg_type}|weight:{self._weight_type}|cutoff:{self.cutoff}>"
    # for key, value in self.feature_args.items():
    #   ret += f"|{key}:{value}"
    # ret += ">"
    return ret

  @property
  def agg(self):
    """
    Accepted aggregation functions for the dynamic feature:     
    """
    if self._agg_type in SUPPORTED_AGGREGATION.keys():
      return SUPPORTED_AGGREGATION[self._agg_type]
    else:
      raise ValueError("The aggregation type is not recognized")
  @agg.setter
  def agg(self, value):
    assert isinstance(value, str), "The aggregation type should be a string"
    self._agg_type = value

  @property
  def weight_type(self):
    """
    Check SUPPORTED_FEATURES for the available weight types. 

    Notes
    -----
    Be cautious about the partial charge (which contains both positive and negative values). A lot of aggregation functions compute weighted average (e.g. weighted center). 
    Make sure that the weight and aggregation function valid in physical sense. 
    """
    if self._weight_type in SUPPORTED_FEATURES.keys():
      return SUPPORTED_FEATURES[self._weight_type]
    else:
      raise ValueError(f"The weight type is not recognized {self._weight_type}; Available types are {SUPPORTED_FEATURES.keys()}")
  @weight_type.setter
  def weight_type(self, value):
    assert isinstance(value, str), "The weight type should be a string"
    self._weight_type = value
  
  def cache(self, trajectory): 
    """
    Take the required weight type (self.weight_type) and cache the weights for each atom in the trajectory
    """
    super().cache(trajectory)   # Obtain the atomic number and residue IDs
    self.cached_array = cache_properties(trajectory, self.weight_type, **self.feature_args)

  def query(self, topology, frame_coords, focal_point):
    """
    Query the coordinates and weights and feed for the following self.run function

    Notes
    -----

    - self.MAX_ALLOWED_ATOMS: Depends on the GPU cache size for each thread
    - self.DEFAULT_COORD: The coordinates for padding of atoms in the box across required frames. Also hard coded in GPU code. 

    The return weight array should be flattened to a 1D array

    """
    assert len(frame_coords.shape) == 3, f"{self.classname}::Warning: from feature ({self.__str__()}): The coordinates should follow the convention (frames, atoms, 3); "

    coords = np.full((len(frame_coords), self.MAX_ALLOWED_ATOMS, 3), self.DEFAULT_COORD, dtype=np.float32)
    weights = np.full((len(frame_coords), self.MAX_ALLOWED_ATOMS), 0.0, dtype=np.float32)
    max_atom_nr = 0
    zero_count = 0
    for idx, frame in enumerate(frame_coords):
      # Operation on each frame (Frame is modified inplace)
      idx_inbox, coord_inbox = super().query(topology, frame, focal_point)

      atomnr_inbox = np.count_nonzero(idx_inbox)
      if atomnr_inbox > self.MAX_ALLOWED_ATOMS:
        logger.warning(f"{self.classname}: The maximum allowed atom slice is {self.MAX_ALLOWED_ATOMS} but the maximum atom number is {atomnr_inbox}")
      zero_count += 1 if atomnr_inbox == 0 else 0
      atomnr_inbox = min(atomnr_inbox, self.MAX_ALLOWED_ATOMS)

      coords[idx, :atomnr_inbox] = coord_inbox[:atomnr_inbox]
      weights[idx, :atomnr_inbox] = self.cached_array[idx_inbox][:atomnr_inbox]
      max_atom_nr = max(max_atom_nr, atomnr_inbox)
    
    if zero_count > 0 and config.verbose(): 
      logger.warning(f"{self.classname}: {zero_count} out of {len(frame_coords)} frames has no atoms in the box. The coordinates will be padded with {self.DEFAULT_COORD} and 0.0 for the weights.")

    # Prepare the return arrays
    ret_coord  = np.ascontiguousarray(coords[:, :max_atom_nr], dtype=np.float32)
    ret_weight = np.ascontiguousarray(weights[:, :max_atom_nr].flatten(), dtype=np.float32)
    return ret_coord, ret_weight
  
  def run(self, frames, weights):
    """
    Run the dynamic feature algorithm
    """
    raise NotImplementedError("The run function should be implemented in the child class")
  

class DensityFlow(DynamicFeature):
  """
  Perform Density flow algorithm on the registered trajectories. Each frame is firstly voxelized into 3D grid (sized [F,D,D,D]), and then aggregated the time dimension to a 3D grid (sized [D,D,D]). 
  
  Notes
  -----
  .. note::

    For weight types, please refer to the :func:`nearl.features.cache_properties` function.

    For aggregation types, please refer to the :class:`nearl.features.DynamicFeature` .

  """
  def __str__(self):
    ret = f"{self.classname} <agg:{self._agg_type}|weight:{self._weight_type}|cutoff:{self.cutoff}|sigma:{self.sigma}>"
    return ret
  
  def query(self, topology, frame_coords, focal_point):
    """
    """
    ret_coord, ret_weight = super().query(topology, frame_coords, focal_point)
    return ret_coord, ret_weight

  def run(self, frames, weights):
    """
    Take frames of coordinates and weights to run the density flow algorithm. 

    Parameters
    ----------
    frames : np.array
      The coordinates of the atoms in the frames.
    weights : np.array
      The weights of the atoms in the frames. NOTE: The weight has to match the number of atoms in the frames.

    Return: 
    -------
    ret_arr : np.array
      The dynamic feature array with the dimensions of self.dims

    """
    st = time.perf_counter()
    ret_arr = commands.density_flow(frames, weights, self.dims, self.spacing, self.cutoff, self.sigma, self.agg)
    print(f"{'Timing_PDF':15} {time.perf_counter() - st:10f} seconds for {frames.shape[1]} atoms")  # TODO
    
    return ret_arr.reshape(self.dims)  


class MarchingObservers(DynamicFeature): 
  """
  Perform the marching observers algorithm on the registered trajectories. Each 3D voxel serves as a observer and the algorithm marches through the voxels to observe the motion of the atoms in the trajectory (sized [F,D,D,D]). The marching observers algorithm is then aggregated to a 3D grid (sized [D,D,D]). 

  Notes
  -----
  Direct Count-based Observables

  +------------------------+------------------+
  | Property Name          | Property Type    |
  +========================+==================+
  | existence              | 1                |
  +------------------------+------------------+
  | direct_count           | 2                |
  +------------------------+------------------+
  | distinct_count         | 3                |
  +------------------------+------------------+

  Weight-based Observables

  +------------------------+------------------+
  | Property Name          | Property Type    |
  +========================+==================+
  | mean_distance          | 11               |
  +------------------------+------------------+
  | cumulative_weight      | 12               |
  +------------------------+------------------+
  | density                | 13               |
  +------------------------+------------------+
  | dispersion             | 14               |
  +------------------------+------------------+
  | eccentricity           | 15               |
  +------------------------+------------------+
  | radius_of_gyration     | 16               |
  +------------------------+------------------+
  
  For weight types, please refer to the :func:`nearl.features.cache_properties` function.

  For aggregation types, please refer to the :class:`nearl.features.DynamicFeature` .
  
  """
  def __init__(self, obs="existence", **kwargs): 
    # Just omit the sigma parameter while the inheritance. 
    # while initialization of the parent class, weight_type, cutoff, agg are set
    super().__init__(**kwargs)
    self._obs_type = obs     # Directly pass to the CUDA voxelizer
    
  def __str__(self): 
    return f"{self.classname} <obs:{self._obs_type}|type:{self._weight_type}|agg:{self._agg_type}|cutoff:{self.cutoff}>"
  @property
  def obs(self):
    """
    The observation type for the marching observers algorithm
    """
    if self._obs_type in SUPPORTED_OBSERVATION.keys():
      return SUPPORTED_OBSERVATION[self._obs_type]
    else:
      raise ValueError(f"The observation type is not recognized {self._obs_type}; Available types are {SUPPORTED_OBSERVATION.keys()}")
  @obs.setter
  def obs(self, value):
    assert isinstance(value, str), "The observation type should be a string"
    self._obs_type = value

  def query(self, topology, coordinates, focus):
    """
    Query the coordinates and weights from a set of frames and return the feature array

    Notes
    -----
    Use the same method to query the coordinates and weights as the parent class

    """
    ret_coord, ret_weight = super().query(topology, coordinates, focus)
    return ret_coord, ret_weight
  
  def run(self, coords, weights): 
    """
    Get several frames and perform the marching observers algorithm

    Parameters
    ----------
    coords : np.array
      The coordinates of the atoms in the frames.
    weights : np.array
      The weights of the atoms in the frames.

    Returns
    -------
    ret_arr : np.array
      The feature array with the dimensions of self.dims 
    """
    st = time.perf_counter()
    ret_arr = commands.marching_observer(
      coords, weights, 
      self.dims, self.spacing, 
      self.cutoff, 
      self.obs, self.agg
    ) 
    print(f"{'Timing_OBS':15} {time.perf_counter() - st:10f} seconds for {coords.shape[1]} atoms")  # TODO 
    return ret_arr.reshape(self.dims)


###############################################################################
# Label-Tyep Features
###############################################################################
class LabelIdentity(Feature):
  """
  Return the identity attribute of the trajectory as meta-data.

  Notes
  -----
  In default Trajectory type, the identity is the file name

  In MisatoTraj Trajectory type, the identity is the PDB code
  
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    """
    Cache the trajectory identity for the feature query. 

    """
    super().cache(trajectory)
    self.identity = trajectory.identity

  def query(self, *args):
    """
    Return the identity of the trajectory and there is no dependency on the frame slices. 

    """
    return self.identity


class LabelAffinity(Feature):
  """
  Read the PDBBind table and return the affinity values according to the pdbcode (identity from the trajectory) 

  Parameters
  ----------
  baseline_map : str
    The path to the baseline map for the affinity values
  **kwargs : dict
    The additional arguments for the parent class :class:`nearl.features.Feature`
  
  Attributes
  ----------
  baseline_table : pd.DataFrame
    The baseline table for the affinity values. 
  base_value : float
    The base value for the affinity values, searched in the :func:`cache <nearl.features.LabelAffinity.cache>` and 
    :func:`search_baseline <nearl.features.LabelAffinity.search_baseline>` functions.
  
  """
  def __init__(self, baseline_map, colname="pK", **kwargs):
    super().__init__(outshape=(None,), **kwargs)
    try: 
      import pandas as pd
    except ImportError as e:
      raise ImportError("The pandas module is required for the LabelAffinity feature")
    self.baseline_table = pd.read_csv(baseline_map, header=0, delimiter=",")
    self.base_value = None
    self.colname = colname
    if self.colname not in self.baseline_table.columns:
      raise ValueError(f"The column name {self.colname} is not found in the baseline table")

  def search_baseline(self, pdbcode):
    """
    Search the baseline value based on the PDB code. 

    Notes
    -----
    The default implementation here is to read the csv from PDBBind dataset and search through the PDB code. 
    Override this function to customize the search method for baseline affinity.

    We recommend to use a map from the trajectory identity to the affinity values. 

    """
    pdbcode = utils.get_pdbcode(pdbcode) # This returns the pdbcode in lower case 
    if pdbcode in self.baseline_table["pdbcode"].values: 
      row = self.baseline_table.loc[self.baseline_table["pdbcode"] == pdbcode]
      return row[self.colname].values[0]
    else:
      raise ValueError(f"Cannot find the baseline value for {pdbcode}")
  
  def cache(self, trajectory):
    """
    Loop up the baseline values from the designated table and cache the pairwise closest distances. 

    Notes
    -----
    In this base type, it does not 
    the super.cache() is not needed.

    """
    # IMPORTANT: Retrieve the base value based on the trajectory identity
    self.base_value = self.search_baseline(trajectory.identity)
    if config.verbose():
      printit(f"{self.classname}: Affinity value of {trajectory.identity} is {self.base_value}")

  def query(self, *args):
    """
    Return the for the baseline affinity based on the :func:`nearl.io.traj.Trajectory.identity`. No extra trajectory information is needed. 
    """
    return (self.base_value, )

  def run(self, affinity_val):
    if config.verbose() or config.debug():
      printit(f"{self.classname}: The affinity value is {affinity_val:4.2f}")
    return affinity_val


class LabelStepping(LabelAffinity):
  """
  Convert the base class affinity value to a steping function. Could convert the regression problem to a classification problem. 
  """
  def query(self, *args):
    assert self.base_value is not None, "The base value should be set before the query"
    final_label = self.base_value // 2
    final_label = min(final_label, 5)
    return (final_label, )


class LabelRMSD(LabelAffinity): 
  def __init__(self, 
    base_value=0, 
    outshape=(None,), 
    **kwargs
  ): 
    if "selection" not in kwargs.keys():
      raise ValueError("The selection should be set for the RMSD label feature")
    super().__init__(outshape = outshape, **kwargs) 
    self.base_value = float(base_value)

  def cache(self, trajectory): 
    # Cache the address of the trajectory for further query of the RMSD array based on focused points
    super().cache(trajectory)
    RMSD_CUTOFF = 8 
    self.refframe = trajectory[0] 
    self.cached_array = pt.rmsd_nofit(trajectory, mask=self.selection, ref=self.refframe) 
    self.cached_array = np.minimum(self.cached_array, RMSD_CUTOFF)  # Set a cutoff RMSD for limiting the z-score

  def query(self, topology, frames, focus): 
    # Query points near the area around the focused point
    # Major problem, How to use the cached array to query the label based on the focused point
    # Only need the topology and the frames. 
    tmptraj = pt.Trajectory(xyz=frames, top=topology)
    rmsd_arr = pt.rmsd_nofit(tmptraj, mask=self.selection, ref=self.refframe)
    # refcoord = self.refframe.xyz[self.selected]
    rmsd = np.mean(rmsd_arr)
    result = self.base_value * (1 - 0.1 * (rmsd / 8.0) )
    return (result, )


class LabelPCDT(LabelAffinity): 
  """
  Label the feature based on the cosine similarity between the PCDT of the focused point and the cached PCDT array.

  Parameters
  ----------
  selection : str or list, tuple, np.ndarray
    The selection of the atoms for the PCDT calculation
  search_cutoff : float, default=None
    The cutoff distance for limiting the outliers in the PCDT calculation
  **kwargs : dict
    The additional arguments for the parent class :class:`nearl.features.LabelAffinity`
  
  """
  def __init__(self, selection=None, search_cutoff = None, **kwargs): 
    # Initialize the self.base_value in the parent class
    super().__init__(**kwargs) 
    # Pairs of atoms for the PCDT calculation 
    self.selection_prototype = selection
    self.selected = None
    self.selected_counterpart = None

    self.refframe = None
    self.cached_array = None
    self.search_cutoff = search_cutoff

  def cache(self, trajectory): 
    """
    Cache the Pairwise Closest DisTance (PCDT) array for the conformation-based penalty calculation
    """
    # IMPORTANT: Retrieve the base value based on the trajectory identity (self.base_value) in the parent class
    super().cache(trajectory)

    # Parse the selection and the counterpart selection
    if isinstance(self.selection_prototype, str):
      selected = trajectory.top.select(self.selection_prototype)
    elif isinstance(self.selection_prototype, (list, tuple, np.ndarray)):
      selected = np.array([int(i) for i in self.selection_prototype])
    else: 
      raise ValueError("The selection should be either a string or a list of atom indices")

    # IMPORTANT: Set the global reference frame for the trajectory
    # This allows a fixed pair of atoms for the PCDT calculation
    self.refframe = trajectory[0]
    self.selected = selected
    self.selected_counterpart = np.full(len(selected), 0, dtype=int)
    backbone_cb = trajectory.top.select("@C, @N, @CA, @O, @CB")
    for i in range(len(selected)):
      dist = np.linalg.norm(self.refframe.xyz[selected[i]] - self.refframe.xyz[backbone_cb], axis=1)
      self.selected_counterpart[i] = backbone_cb[np.argmin(dist)]

    self.cached_array = utils.compute_pcdt(trajectory, self.selected, self.selected_counterpart, ref=self.refframe, return_info=False)

    # Set a cutoff distance for limiting outliers in the PCDT
    if self.search_cutoff is not None:
      self.cached_array = np.minimum(self.cached_array, self.search_cutoff)  

  def query(self, topology, frames, focus):
    """
    Compute the PCDT of the frame slice and return the final value based on the cached PCDT array

    Notes
    -----
    This function applies a Z-score-based penalty to the original base pK value. 

    """
    tmptraj = pt.Trajectory(xyz=frames, top=topology)
    pdist_arr = utils.compute_pcdt(tmptraj, self.selected, self.selected_counterpart, ref=self.refframe, return_info=False)
    if self.search_cutoff is not None:
      pdist_arr = np.minimum(pdist_arr, self.search_cutoff)
    
    # Per atom Z-score calculation
    mean_0 = np.mean(self.cached_array, axis=1)
    mean_1 = np.mean(pdist_arr, axis=1)
    std_0 = np.std(self.cached_array, axis=1)

    # Avoid the potential division by zero
    if np.any(std_0 == 0):
      # printit(f"DEBUG: The standard deviation of the cached PCDT array is zero for some atoms")
      if self.search_cutoff is not None:
        std_0[std_0 == 0] = self.search_cutoff
      else: 
        std_0[std_0 == 0] = 6.0
    
    # Z-score makes the feature too noisy
    z_scores = (mean_0 - mean_1) / std_0
    z_score = np.mean(z_scores)
    penalty = abs(np.mean(z_score))
    # try with cosine similarity
    # penalty = 1 - max(np.dot(mean_0, mean_1) / ((np.linalg.norm(mean_0) * np.linalg.norm(mean_1))), 0)
    # result = penalty
    # print("====>>>>", penalty)
    result = self.base_value * (1 - 0.1 * penalty)

    return (result, )


###############################################################################
# Reporter-Tyep Features (for debugging and testing purposes) 
###############################################################################
# class LabelResType(Feature): 
#   def __init__(self, restype, byres=True, outshape=(None,), **kwargs): 
#     """
#     Report the residue type as the result feature. 

#     Notes
#     -----
#     This is the special label-type feature that returns the label based on the residue type.
#     """
#     super().__init__(outshape = outshape, byres=byres, **kwargs)
#     self.restype = restype
  
#   def query(self, topology, frames, focus):
#     """
#     When querying the single-residue types, the topology and frames has to be cropped to the focused residue (COG of the residue). 
#     Hence only the cropped residues will be retured. 
#     """
#     if frames.shape.__len__() == 3: 
#       frames = frames[self.frame_offset]
#     returned, _ = super().query(topology, frames, focus)
#     resnames = [i.name for i in topology.residues]
#     resids = [i.resid for i in topology.atoms]

#     final_resname = ""
#     processed = []
#     for i in range(len(returned)): 
#       if returned[i] and resids[i] not in processed:
#         final_resname += resnames[resids[i]]
#         processed.append(resids[i])
#     return (final_resname, )

#   def run(self, resname):
#     """
#     Look up the residue name in the dictionary and return its label
#     """
#     if self.restype == "single" and resname in constants.RES2LAB:
#       retval = constants.RES2LAB[resname]
#     elif self.restype == "single" and resname not in constants.RES2LAB:
#       printit(f"DEBUG: The residue type {resname} is not recognized, there might be some problem in the trajectory cropping")
#       retval = 0 
#     elif self.restype == "dual" and resname in constants.RES2LAB_DUAL:
#       retval = constants.RES2LAB_DUAL[resname]
#     elif self.restype == "dual" and resname not in constants.RES2LAB_DUAL:
#       printit(f"DEBUG: The residue type {resname} is not recognized, there might be some problem in the trajectory cropping")
#       retval = 0
#     return retval


# class Coords(Feature):
#   """
#   Report the coordinates as the result feature. 
#   """
#   def __init__(self, **kwargs): 
#     super().__init__(outshape=(None,4), **kwargs)

#   def query(self, topology, frames, focus):
#     if frames.shape.__len__() == 3: 
#       frames = frames[self.frame_offset]
#     idx_inbox, coord_inbox = super().query(topology, frames, focus)
#     nr_atom = np.count_nonzero(idx_inbox)
#     ret = np.full((nr_atom, 4), 0.0, dtype=np.float32)
#     ret[:, :3] = coord_inbox
#     ret[:, 3] = self.atomic_numbers[idx_inbox]
#     return (ret,)
  
#   def run(self, arr):
#     return arr
  
#   def dump(self, result): 
#     with h5py.File(self.outfile, "r") as f:
#       if self.outkey in f.keys():
#         start_idx = f[self.outkey].shape[0]
#       else:
#         start_idx = 0
#       end_idx = start_idx + result.shape[0]
#     utils.append_hdf_data(self.outfile, f"{self.outkey}_", np.array([[start_idx, end_idx]], dtype=int), dtype=int, maxshape=(None,2), chunks=True, compression="gzip", compression_opts=4)
#     utils.append_hdf_data(self.outfile, self.outkey, np.array(result, dtype=np.float32), dtype=np.float32, maxshape=self.outshape, chunks=True, compression="gzip", compression_opts=4)  

