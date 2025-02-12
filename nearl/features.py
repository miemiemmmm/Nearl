import sys, tempfile, os, subprocess

import h5py
import numpy as np
import pytraj as pt

from . import utils, commands, constants, chemtools   # local modules 
from . import printit, config   # local static methods/objects

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
  "LabelResType",
  

  # Other features
  # "VectorizerViewpoint", 

  # Function to get properties for dynamic features
  "cache_properties", 
  
]

"""
! IMPORTANT: 
  If there is difference from trajectory to trajectory, for example the name of the ligand, feature object has to behave 
  differently. However, the featurizer statically pipelines the feature.featurize() function and difference is not known
  in these featurization process. 
  This change has to take place in the featurizer object because the trajectories are registered into the featurizer. Add 
  the variable attribute to the featurizer object and use it in the re-organized feature.featurize() function. 
"""


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
  "information_entropy": 7
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


def crop(points, upperbound, padding):
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

  Methods
  -------
  hook(featurizer)
    Hook the feature generator back to the feature convolutor and obtain necessary attributes from the featurizer
  cache(trajectory)
    Cache the needed weights for each atom in the trajectory for further Feature.query function
  query(topology, frame_coords, focal_point)
    Query the atoms and weights within the bounding box
  run(coords, weights)
    By default use voxelize_coords to voxelize the coordinates and weights
  dump(result)
    Dump the result feature to an HDF5 file, Feature.outfile and Feature.outkey should be set in its child class (Either via __init__ or hook function)

  Notes
  -----
  The input and the output of the query, run and dump function should be chained together.
  
  """
  # The input and the output of the query, run and dump function should be chained together.
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
  # Hookable to featurizer
  # - sigma: The sigma of the Gaussian kernel
  # - cutoff: The cutoff distance
  # - outfile: The output file
  # - outkey: The output key
  # - padding: The padding of the box
  # - byres: The boolean flag to get the residues within the bounding box
  # Individual parameters
  # - outshape: The shape of the output array
  # - force_recache: The boolean flag to force recache the weights
  def __init__(self, 
    dims=None, spacing=None, 
    outfile=None, outkey=None,
    cutoff=None, sigma=None,
    padding=0, byres=None, 
    outshape=None, force_recache=None,
    selection=None,
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

    # General variables for featurization process 
    self.cutoff = cutoff
    self.sigma = sigma
    self.outfile = outfile
    self.padding = padding
    self.byres = byres

    # To be used in .query() function
    self.selection = selection
    self.selected = None
    self.force_recache = False if force_recache is None else True
    
    # To be used in .dump() function
    self.outkey = outkey        # Has to be specific to the feature
    self.outshape = outshape 
    self.hdf_compress_level = kwargs.get("hdf_compress_level", 0)
    self.hdf_dump_opts = {}

  def __str__(self):
    ret_str = f"{self.__class__.__name__}"
    if config.verbose():
      ret_str += f"<dims:{self.dims}|spacing:{self.spacing}>"
      # ret_str += f"<> "
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
      if config.verbose():
        printit(f"{self.__class__.__name__} Warning: The dims should be a number, list, tuple or a numpy array, not {type(value)}")
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
      if config.verbose():
        printit(f"{self.__class__.__name__}: Warning: The spacing is not a valid number, setting to None")
      self.__spacing = None
    if self.__dims is not None and self.__spacing is not None:
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
    -----
    If the following attributes are not set manually, hook function will try to inherit them from the featurizer object: 
    sigma, cutoff, outfile, outkey, padding, byres
    """
    printit(f"{self.__class__.__name__}: Using features: ", featurizer.dims, featurizer.spacing)
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
          printit(f"{self.__class__.__name__}: Inheriting the sigma from the featurizer: {key} {keyval}")
    
    if getattr(self, "hdf_compress_level", 0):
      self.hdf_dump_opts["compression_opts"] = self.hdf_compress_level
      self.hdf_dump_opts["compression"] = "gzip"
    

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
        printit(f"{self.__class__.__name__} Warning: The number of atoms in structure does not match the number of aromaticity values")
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
      The boolean mask of the atoms within the bounding box

    Notes
    -----
    The frame coordinates are explicitly tranlated to match the focued part to the center of the box. 
    """
    assert focal_point.shape.__len__() == 1, f"The focal point should be a 1D array with length 3, rather than the given {focal_point.shape}"
    
    if (len(self.resids) != topology.n_atoms) or self.force_recache: 
      if config.verbose():
        printit(f"{self.__class__.__name__}: Dealing with inhomogeneous topology")
      if len(frame_coords.shape) == 2:
        self.cache(pt.Trajectory(xyz=np.array([frame_coords]), top=topology))
      else:
        self.cache(pt.Trajectory(xyz=frame_coords, top=topology))

    if self.center is None or self.lengths is None or self.padding is None:
      printit(f"{self} Skipping the coordinates cropping due to the missing center, lengths or padding information") 
      return np.full(topology.n_atoms, True, dtype=bool), frame_coords
    else: 
      coord_trans = frame_coords - focal_point + self.center  
      mask = crop(coord_trans, self.lengths, self.padding)
      
      if np.count_nonzero(mask) == 0:
        printit(f"{self} Warning: Found 0 atoms in the bounding box near the focal point {np.round(focal_point,1)}")
      elif config.verbose() or config.debug(): 
        printit(f"{self}: Found {np.count_nonzero(mask)} atoms in the bounding box near the focal point {np.round(focal_point,1)}")

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
      final_coords = np.ascontiguousarray(coord_trans[final_mask])
      return final_mask, final_coords

  def run(self, coords, weights):
    """
    By default use voxelize_coords to voxelize the coordinates and weights

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
      printit(f"{self.__class__.__name__}: Warning: The coordinates are empty")
      return np.zeros(self.dims, dtype=np.float32)
    
    ret = commands.voxelize_coords(coords, weights, self.dims, self.spacing, self.cutoff, self.sigma)
    
    if np.sum(np.isnan(ret)) > 0:
      printit(f"{self} Warning: The returned array has {np.isnan(ret).sum()} NaN values")
    # Check the sum of the absolute values of the returned array
    if config.verbose() or config.debug():
      ret_sum = np.sum(np.abs(ret))
      if np.isclose(ret_sum, 0):
        printit(f"{self} Warning: The sum of the returned array is zero")
    
    return ret

  def dump(self, result):
    """
    Dump the result feature to an HDF5 file, Feature.outfile and Feature.outkey should be set in its child class (Either via __init__ or hook function)

    Parameters
    ----------
    results : np.array
      The result feature array
    """
    if ("outfile" in dir(self)) and ("outkey" in dir(self)) and (len(self.outfile) > 0):
      if np.isnan(result).sum() > 0:
        printit(f"{self.__class__.__name__}: Warning: Found {np.isnan(result).sum()} NaN values in the result array")
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
      printit(f"{self.__class__.__name__}: Warning: The outfile and outkey are not set, the result is not dumped to the disk")
      


class AtomicNumber(Feature):
  """
  Weight type atomic number. 
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    super().cache(trajectory)
  
  def query(self, topology, frame_coords, focal_point): 
    """
    Query the atomic coordinates and weights within the bounding box
    

    Notes
    -----
    By default, a slice of frame coordinates is passed to the querier function (typically 3 dimension shaped by [frames_number, atom_number, 3])
    However, for static feature, only one frame is needed. 
    Hence, the querier function by default uses the first frame (frame_coords[0], the default bahavior is customizable by the user). 

    The focused part of the coords needs to be translated to the center of the box before sending to the runner function.
    """
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = np.array([self.atomic_numbers[i] for i in self.atomic_numbers[idx_inbox]], dtype=np.float32)
    return coord_inbox, weights


class Mass(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Parse of the Mask should not be in here, the input should be focal points in coordinates format
  Explicitly pass the cutoff and sigma while initializing the Feature object for the time being
  Atomic mass as a feature
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    super().cache(trajectory)
    self.mass = np.array([constants.ATOMICMASS[i] for i in self.atomic_numbers], dtype=np.float32)


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
    
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = np.array([self.mass[i] for i in self.atomic_numbers[idx_inbox]], dtype=np.float32)
    return coord_inbox, weights


class HeavyAtom(Feature):
  def __init__(self, default_weight=1, **kwargs):
    super().__init__(**kwargs)
    self.default_weight = default_weight

  def cache(self, trajectory):
    """
    Prepare the heavy atom weights
    """
    super().cache(trajectory)
    self.heavy_atoms = np.full(len(self.resids), 0, dtype=np.float32)
    self.heavy_atoms[np.where(self.atomic_numbers > 1)] = self.default_weight

  def query(self, topology, frame_coords, focal_point): 
    """
    Get the atoms and weights within the bounding box
    """
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]  # NOTE: Get the first frame if multiple frames are given
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = self.heavy_atoms[idx_inbox]
    return coord_inbox, weights


class Aromaticity(Feature):
  def __init__(self, reverse=None, **kwargs): 
    super().__init__(**kwargs)
    # Set reverse to True if you want to get the non-aromatic atoms to be 1
    self.reverse = reverse 

  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_aromatic = chemtools.label_aromaticity(fpt.name)

    if len(atoms_aromatic) != trajectory.n_atoms:
      printit(f"{self.__class__.__name__} Warning: The number of atoms in PDB does not match the number of aromaticity values")
    if self.reverse: 
      self.atoms_aromatic = np.array([1 if i == 0 else 0 for i in atoms_aromatic], dtype=np.float32)
    else: 
      self.atoms_aromatic = np.array(atoms_aromatic, dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = self.atoms_aromatic[idx_inbox]
    return coord_inbox, weights


class Ring(Feature):
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
      printit(f"{self.__class__.__name__} Warning: The number of atoms in PDB does not match the number of aromaticity values")

    if self.reverse: 
      self.atoms_in_ring = np.array([1 if i == 0 else 0 for i in atoms_in_ring], dtype=np.float32)
    else: 
      self.atoms_in_ring = np.array(atoms_in_ring, dtype=np.float32)
  
  def query(self, topology, frame_coords, focal_point): 
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = self.atoms_in_ring[idx_inbox]
    return coord_inbox, weights


class Selection(Feature):
  """

  """
  def __init__(self, default_value=1, **kwargs):
    if "selection" not in kwargs.keys():
      raise ValueError(f"{self.__class__.__name__}: The selection parameter should be set")
    super().__init__(**kwargs)
    self.default_value = default_value

  def __str__(self): 
    return f"{self.__class__.__name__}({self.selection}:{self.default_value})"

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = np.full(len(coord_inbox), self.default_value, dtype=np.float32)
    return coord_inbox, weights
  

class HBondDonor(Feature):
  """

  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_hbond_donor = chemtools.label_hbond_donor(fpt.name)

    if len(atoms_hbond_donor) != trajectory.n_atoms:
      printit(f"{self.__class__.__name__} Warning: The number of atoms in PDB does not match the number of aromaticity values")
    self.atoms_hbond_donor = np.array(atoms_hbond_donor, dtype=np.float32)
  
  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = self.atoms_hbond_donor[idx_inbox]
    return coord_inbox, weights


class HBondAcceptor(Feature):
  """

  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_hbond_acceptor = chemtools.label_hbond_acceptor(fpt.name)

    if len(atoms_hbond_acceptor) != trajectory.n_atoms:
      printit(f"{self.__class__.__name__} Warning: The number of atoms in PDB does not match the number of aromaticity values")
    self.atoms_hbond_acceptor = np.array(atoms_hbond_acceptor, dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = self.atoms_hbond_acceptor[idx_inbox]
    return coord_inbox, weights


class Hybridization(Feature):
  """

  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_hybridization = chemtools.label_hybridization(fpt.name)

    if len(atoms_hybridization) != trajectory.n_atoms:
      printit(f"{self.__class__.__name__} Warning: The number of atoms in PDB does not match the number of aromaticity values")
    self.atoms_hybridization = np.array(atoms_hybridization, dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = self.atoms_hybridization[idx_inbox]
    return coord_inbox, weights


class Backbone(Feature):
  """

  """
  def __init__(self, reverse=False, **kwargs):
    super().__init__(**kwargs)
    self.reverse = reverse

  def cache(self, trajectory):
    super().cache(trajectory)
    backboneness = [1 if i.name in ["C", "O", "CA", "HA", "N", "HN"] else 0 for i in trajectory.top.atoms]
    if self.reverse: 
      self.backboneness = np.array([1 if i == 0 else 0 for i in backboneness], dtype=np.float32)
    else: 
      self.backboneness = np.array(backboneness, dtype=np.float32)
    
  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = self.backboneness[idx_inbox]
    return coord_inbox, weights
  

class AtomType(Feature):
  """

  """
  def __init__(self, focus_element, **kwargs):
    super().__init__(**kwargs)
    self.focus_element = int(focus_element)

  def __str__(self):
    return f"{self.__class__.__name__} <focus:{constants.ATOMICNR2SYMBOL[self.focus_element]}>"

  def cache(self, trajectory):
    super().cache(trajectory)
    self.atom_type = np.array([1 if i.atomic_number == self.focus_element else 0 for i in trajectory.top.atoms], dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]

    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = self.atom_type[idx_inbox]

    if np.sum(weights) == 0:
      printit(f"{self} Warning: No atoms of the type {self.focus_element} found in the bounding box")
    
    return coord_inbox, weights


class PartialCharge(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Atomic charge feature for the structure of interest;
  The charges are calculated using the ChargeFW package

  Parameters
  ----------
  charge_type : str, Using 'qeq' by default
    The charge type. If 'manual' is used, the charge_parm should be set manually via an array of charge values of each atom
  charge_parm : str, Using 'QEq_00_original' by default 
    The charge parameter
  force_compute : bool, Using False by default
    The partial charge will be prioritized from the charges are already in the trajectory/topology. If True, the charge values will be recomputed anyway. 

  Notes
  -----
  For more information about the charge types and parameters, please refer to the ChargeFW documentation with the url: 
  https://github.com/sb-ncbr/ChargeFW2

  """
  # "qeq" -> "QEq_00_original"
  # The following types are supported by ChargeFW: 
  # [ "sqeqp",  "eem",  "abeem",  "sfkeem",  "qeq", "smpqeq",  "eqeq",  "eqeqc",  "delre",  "peoe", 
  #   "mpeoe",  "gdac",  "sqe",  "sqeq0",  "mgc", "kcm",  "denr",  "tsef",  "charge2",  "veem", "formal" ]
  def __init__(self, charge_type="topology", charge_parm=None, force_compute=False, keep_sign="both", **kwargs):
    super().__init__(**kwargs)
    self.charge_type = charge_type
    self.charge_parm = charge_parm
    self.force_compute = force_compute
    if keep_sign in ["positive", "p"]:
      self.keep_sign = "positive"
    elif keep_sign in ["negative", "n"]:
      self.keep_sign = "negative"
    elif keep_sign in ["both", "b"]:
      self.keep_sign = "both"
    else:
      raise ValueError(f"{self.__class__.__name__} Warning: The keep_sign parameter should be either 'positive', 'negative' or 'both' rather than {keep_sign}")

  def __str__(self):
    return f"{self.__class__.__name__} <type:{self.charge_type}|sign:{self.keep_sign}>"

  def cache(self, trajectory):
    super().cache(trajectory)
    if np.abs(trajectory.top.charge).sum() > 0 and not self.force_compute:
      # Inherit the charge values from the trajectory if the charges are already computed
      self.charge_type = "topology"
      self.charge_values = trajectory.top.charge

    elif self.charge_type == "manual": 
      # If the charge type is manual, the charge values should be set manually
      assert len(self.charge_parm) == trajectory.top.n_atoms, f"The number of charge values does not match the number of atoms in the trajectory"
      self.charge_values = np.array(self.charge_parm, dtype=np.float32)

    elif self.charge_type == "function": 
      # Use some precalculated charge values
      self.charge_values = self.charge_parm(trajectory.identity)
      assert len(self.charge_values) == trajectory.top.n_atoms, f"The number of charge values does not match the number of atoms in the trajectory"

    else: 
      # Otherwise, compute the charges using the ChargeFW2
      import chargefw2_python as cfw
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
          printit(f"{self.__class__.__name__} Warning: Failed to load the molecule because of the following error:\n{e}")
        
        # Step 2: Calculate the charges
        try: 
          charges = cfw.calculate_charges(mol, self.charge_type, self.charge_parm)
          if filename not in charges.keys():
            raise ValueError(f"{self.__class__.__name__}: The charges are not calculated for the file {filename}")
          else: 
            charge_values = np.array(charges[filename], dtype=np.float32)
            if len(charge_values) != trajectory.n_atoms:
              raise ValueError(f"{self.__class__.__name__}: The number of charge values does not match the number of atoms in the trajectory")
            else: 
              self.charge_values = charge_values
        except Exception as e: 
          # Step 3 (optional): If the default charge method fails, try alternative methods
          subprocess.call(["cp", f.name, "/tmp/chargefailed.pdb"])
          printit(f"{self.__class__.__name__} Warning: Default charge method {self.charge_type} failed due to the following error:\n{e}") 
          for method, parmfile in cfw.get_suitable_methods(mol): 
            if method == "formal":
              continue
            if len(parmfile) == 0:
              parmfile = ""
            else: 
              parmfile = parmfile[0].split(".")[0]
            if len(parmfile) > 0:
              printit(f"{self.__class__.__name__}: Trying alternative charge methods {method} with {parmfile} parameter")
            else: 
              printit(f"{self.__class__.__name__}: Trying alternative charge methods {method} without parameter file")
            try:
              charges = cfw.calculate_charges(mol, method, parmfile)
              if filename not in charges.keys():
                continue
              else:
                charge_values = np.array(charges[filename], dtype=np.float32)
                if len(charge_values) != trajectory.n_atoms:
                  continue
                else: 
                  printit(f"{self.__class__.__name__}: Finished the charge computation with {method} method" + (" without parameter file" if len(parmfile) == 0 else " with parameter file"))
                  self.charge_values = charge_values
                  break
            except Exception as e:
              printit(f"{self.__class__.__name__} Warning: Failed to calculate molecular charge (Alternative charge types) because of the following error:\n {e}")
              continue

      if charges is None or charge_values is None:
        printit(f"{self.__class__.__name__} Warning: The charge computation fails. Setting all charge values to 0. ", file=sys.stderr)
        self.charge_values = np.zeros(trajectory.n_atoms)
    
    # Final check of the needed sign of the charge values
    if self.keep_sign in ["positive", "p"]:
      self.charge_values = np.maximum(self.charge_values, 0)
    elif self.keep_sign in ["negative", "n"]:
      self.charge_values = np.minimum(self.charge_values, 0)
    elif self.keep_sign in ["both", "b"]:
      pass
    else: 
      raise ValueError(f"{self.__class__.__name__} Warning: The keep_sign parameter should be either 'positive', 'negative' or 'both' rather than {self.keep_sign}")
        

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = self.charge_values[idx_inbox] 
    return coord_inbox, weights


class Electronegativity(Feature):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    super().cache(trajectory)
    self.electronegativity = np.array([constants.ELECNEG[i] for i in self.atomic_numbers], dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = self.electronegativity[idx_inbox]
    return coord_inbox, weights
  

class Hydrophobicity(Feature):
  def __int__(self):
    super().__init__()

  def cache(self, trajectory):
    super().cache(trajectory)
    elecnegs = np.array([constants.ELECNEG[i] for i in self.atomic_numbers], dtype=np.float32)
    self.hydrophobicity = np.abs(elecnegs - constants.ELECNEG[6])
  
  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox, coord_inbox = super().query(topology, frame_coords, focal_point)
    # coord_inbox = frame_coords[idx_inbox]
    weights = self.hydrophobicity[idx_inbox]
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
    # atomic_id
    cached_arr = np.array([i.index for i in atoms], dtype=np.float32)

  elif property_type == 2:  
    # residue_id
    cached_arr = np.array([i.resid for i in atoms], dtype=np.float32)
    
  elif property_type == 3: 
    # atomic_number
    cached_arr = np.array([i.atomic_number for i in atoms], dtype=np.float32)

  elif property_type == 4:
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
    # backboneness 
    backboneness = [1 if i.name in ["C", "O", "CA", "HA", "N", "HN"] else 0 for i in atoms]
    cached_arr = np.array([1 if i == 0 else 0 for i in backboneness], dtype=np.float32)

  elif property_type == 27:
    # sidechainness
    sidechainness = [0 if i.name in ["C", "O", "CA", "HA", "N", "HN"] else 1 for i in atoms]
    cached_arr = np.array(sidechainness, dtype=np.float32)

  elif property_type == 28:
    # Needs the focus_element to be set in the kwargs
    if "element_type" not in kwargs.keys():
      raise ValueError("The focus element should be set for the atom type")
    focus_element = kwargs["element_type"]  
    cached_arr = np.array([1 if i == focus_element else 0 for i in atom_numbers], dtype=np.float32)
  
  return np.array(cached_arr, dtype=np.float32)


class DynamicFeature(Feature):
  """
  Visit this function :func:`nearl.features.cache_properties` to get the available properties for the dynamic features.

  Parameters
  ----------
  weight_type : str, default="mass"
    The weight type for the dynamic feature. Check :func:`nearl.features.cache_properties` for the available weight types
  agg : str, default="mean"
    The aggregation function for the dynamic feature. 

  Notes
  -----    
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
    
  """

  def __init__(self, agg="mean", weight_type="mass", **kwargs):
    super().__init__(**kwargs)
    self._agg_type = agg
    self._weight_type = weight_type
    # Need to manually handle the kwargs for specific features instances
    self.feature_args = kwargs
    self.MAX_ALLOWED_ATOMS = 1000
    self.DEFAULT_COORD = 99999.0

  def __str__(self):
    ret = f"{self.__class__.__name__} <agg:{self.agg}|weight:{self.weight_type}"
    for key, value in self.feature_args.items():
      ret += f"|{key}:{value}"
    ret += ">"
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
    self.cached_weights = cache_properties(trajectory, self.weight_type, **self.feature_args)

  def query(self, topology, frame_coords, focal_point):
    """
    Query the coordinates and weights and feed for the following self.run function

    Notes
    -----

    - self.MAX_ALLOWED_ATOMS: Depends on the GPU cache size for each thread
    - self.DEFAULT_COORD: The coordinates for padding of atoms in the box across required frames. Also hard coded in GPU code. 

    The return weight array should be flattened to a 1D array

    """
    assert len(frame_coords.shape) == 3, f"Warning from feature ({self.__str__()}): The coordinates should follow the convention (frames, atoms, 3); "

    coords = np.full((len(frame_coords), self.MAX_ALLOWED_ATOMS, 3), self.DEFAULT_COORD, dtype=np.float32)
    weights = np.full((len(frame_coords), self.MAX_ALLOWED_ATOMS), 0.0, dtype=np.float32)
    max_atom_nr = 0
    zero_count = 0
    for idx, frame in enumerate(frame_coords):
      # Operation on each frame (Frame is modified inplace)
      idx_inbox, coord_inbox = super().query(topology, frame, focal_point)

      atomnr_inbox = np.count_nonzero(idx_inbox)
      if atomnr_inbox > self.MAX_ALLOWED_ATOMS:
        printit(f"{self.__class__.__name__} Warning: The maximum allowed atom slice is {self.MAX_ALLOWED_ATOMS} but the maximum atom number is {atomnr_inbox}")
      zero_count += 1 if atomnr_inbox == 0 else 0
      atomnr_inbox = min(atomnr_inbox, self.MAX_ALLOWED_ATOMS)

      coords[idx, :atomnr_inbox] = coord_inbox[:atomnr_inbox]
      weights[idx, :atomnr_inbox] = self.cached_weights[idx_inbox][:atomnr_inbox]
      max_atom_nr = max(max_atom_nr, atomnr_inbox)
    
    if zero_count > 0 and config.verbose(): 
      printit(f"{self.__class__.__name__} Warning: {zero_count} out of {len(frame_coords)} frames has no atoms in the box. The coordinates will be padded with {self.DEFAULT_COORD} and 0.0 for the weights.")

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
  Dynamic feature: Density flow algorithm
  
  Notes
  -----
  .. note::

    For weight types, please refer to the :func:`nearl.features.cache_properties` function.

    For aggregation types, please refer to the :class:`nearl.features.DynamicFeature` .

  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs) 

  def query(self, topology, frame_coords, focal_point):
    """
    """
    ret_coord, ret_weight = super().query(topology, frame_coords, focal_point)
    return ret_coord, ret_weight

  def run(self, frames, weights):
    """
    Take frames of coordinates and weights and return the a feature array with the same dimensions as self.dims
    """
    # Run the density flow algorithm
    assert frames.shape[0] * frames.shape[1] == len(weights), f"The production of frame_nr and atom_nr has to be equal to the number of weights: {frames.shape[0] * frames.shape[1]}/{len(weights)}"
    
    ret_arr = commands.voxelize_trajectory(frames, weights, self.dims, self.spacing, self.cutoff, self.sigma, self.agg)

    # print(frames.shape, weights.shape)
    # atom_nr = frames.shape[1]
    # printit(f"{self}: Expected feature sum {np.sum(weights[:atom_nr])} and the actual sum {np.sum(ret_arr)}")
    return ret_arr.reshape(self.dims)  


class MarchingObservers(DynamicFeature): 
  """
  Perform the marching observers algorithm to get the dynamic feature. 

  Inherit from the DynamicFeature class since there are common ways to query the coordinates and weights. 

  Notes
  -----
  Observation types: 
  Direct Count-based Observables

  +------------------------+------------------+
  | Property Name          | Property Type    |
  +========================+==================+
  | particle_existance     | 1                |
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
  def __init__(self, obs="particle_existance", **kwargs): 
    # Just omit the sigma parameter while the inheritance. 
    # while initialization of the parent class, weight_type, cutoff, agg are set
    super().__init__(**kwargs)
    self._obs_type = obs     # Directly pass to the CUDA voxelizer
    
  def __str__(self): 
    return f"{self.__class__.__name__} <obs:{self._obs_type}|type:{self._weight_type}|agg:{self._agg_type}>"
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
      The feature array with the same dimensions as self.dims
    """
    # if np.count_nonzero(coords != self.DEFAULT_COORD) == 0:
    #   print("============> ALL COORDINATES ARE DEFAULT <============")
    #   return np.zeros(self.dims, dtype=np.float32)
    # else: 
    ret_arr = commands.marching_observers(
      coords, weights, 
      self.dims, self.spacing, 
      self.cutoff, 
      self.obs, self.agg
    ) 
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
      printit(f"{self.__class__.__name__}: Affinity value of {trajectory.identity} is {self.base_value}")

  def query(self, *args):
    """
    Return the for the baseline affinity based on the :func:`nearl.io.traj.Trajectory.identity`. No extra trajectory information is needed. 
    """
    return (self.base_value, )

  def run(self, affinity_val):
    if config.verbose() or config.debug():
      printit(f"{self.__class__.__name__}: The affinity value is {affinity_val:4.2f}")
    return affinity_val

class LabelStepping(LabelAffinity):
  """
  Convert the base class affinity value to a steping function. Could convert the regression problem to a classification problem. 
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def query(self, *args):
    assert not (self.base_value is None), "The base value should be set before the query"
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


class LabelResType(Feature): 
  def __init__(self, restype="single", byres=True, outshape=(None,), **kwargs): 
    """
    Check the labeled single/dual residue labeling based on the residue type.

    Notes
    -----
    This is the special label-type feature that returns the label based on the residue type.
    """
    super().__init__(outshape = outshape, byres=byres, **kwargs)
    self.restype = restype
  
  def query(self, topology, frames, focus):
    """
    When querying the single-residue types, the topology and frames has to be cropped to the focused residue (COG of the residue). 
    Hence only the cropped residues will be retured. 
    """
    if frames.shape.__len__() == 3: 
      frames = frames[0]
    returned, _ = super().query(topology, frames, focus)
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
    """
    Look up the residue name in the dictionary and return its label
    """
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


class Coords(Feature):
  """
  A test feature type to store the 
  """
  def __init__(self, **kwargs): 
    super().__init__(outshape=(None,4), **kwargs)

  def query(self, topology, frames, focus):
    if frames.shape.__len__() == 3: 
      frames = frames[0]
    idx_inbox, coord_inbox = super().query(topology, frames, focus)
    nr_atom = np.count_nonzero(idx_inbox)
    ret = np.full((nr_atom, 4), 0.0, dtype=np.float32)
    ret[:, :3] = coord_inbox
    ret[:, 3] = self.atomic_numbers[idx_inbox]
    return (ret,)
  
  def run(self, arr):
    return arr
  
  def dump(self, result): 
    with h5py.File(self.outfile, "r") as f:
      if self.outkey in f.keys():
        start_idx = f[self.outkey].shape[0]
      else:
        start_idx = 0
      end_idx = start_idx + result.shape[0]
    utils.append_hdf_data(self.outfile, f"{self.outkey}_", np.array([[start_idx, end_idx]], dtype=int), dtype=int, maxshape=(None,2), chunks=True, compression="gzip", compression_opts=4)
    utils.append_hdf_data(self.outfile, self.outkey, np.array(result, dtype=np.float32), dtype=np.float32, maxshape=self.outshape, chunks=True, compression="gzip", compression_opts=4)



class VectorizerViewpoint(Feature):
  def __init__(self, 
    write_block= False, write_segment = False,
    **kwargs
  ):
    super().__init__(**kwargs)
    self.QUERIED_SEGMENTS = []
    self.SEGMENT_NUMBER = 6
    self.VIEW_BINNR = 10
    self.viewpoint_mode = "far"
    # Manually set the outshape to change the maxshape during dumping the feature hdf file
    self.outshape = (self.SEGMENT_NUMBER, self.VIEW_BINNR)
    self.write_block = write_block
    self.write_segment = write_segment
  
  
  def hook(self, featurizer):
    super().hook(featurizer)

  def query(self, topology, frames, focus): 
    """
    After query the molecular block at the focused point, perform segmentation and viewpoint calculation
    """
    if len(frames.shape) == 3: 
      frames = frames[0]
    idx_inbox, _ = super().query(topology, frames, focus)
    segments = utils.index_partition(idx_inbox, self.SEGMENT_NUMBER)

    xyzr_ret = []
    for seg in segments:
      coords = frames[seg]
      rs = [constants.ATOMICRADII[i] for i in self.atomic_numbers[seg]]
      xyzr = np.full((len(coords), 4), 0.0, dtype=np.float32)
      xyzr[:, :3] = coords
      xyzr[:, 3] = rs
      xyzr_ret.append(xyzr)
    
    # Process the inbox status and return the segment
    if self.viewpoint_mode == "center":
      center = self.center
      viewpoints = np.tile(center, (self.SEGMENT_NUMBER, 1))
    elif self.viewpoint_mode == "self": 
      viewpoints = np.full((self.SEGMENT_NUMBER, 3), 0.0, dtype=np.float32)
      for i in range(self.SEGMENT_NUMBER): 
        print(segments[i], frames[segments[i]])
        if len(frames[segments[i]]) == 0: 
          viewpoints[i] = np.full(3, 999, dtype=np.float32)
        else:
          viewpoints[i] = np.mean(frames[segments[i]], axis=0)
    elif self.viewpoint_mode == "far":
      viewpoints = np.full((self.SEGMENT_NUMBER, 3), 999999.0, dtype=np.float32)
    else:
      raise ValueError("The viewpoint mode is not recognized")
    
    if self.write_block:
      self.write_block_ply(xyzr_ret)
    return xyzr_ret, viewpoints

  def run(self, xyzrs, viewpoints): 
    """
    Take XYZR arrays and viewpoints to calculate the viewpoint histogram
    """
    ret_arr = np.full((self.SEGMENT_NUMBER, self.VIEW_BINNR), 0.0, dtype=float) 
    for seg_idx in range(self.SEGMENT_NUMBER): 
      if len(xyzrs[seg_idx]) > 0: 
        viewpoint_feature = commands.viewpoint_histogram_xyzr(xyzrs[seg_idx], viewpoints[seg_idx], self.VIEW_BINNR, write_ply=self.write_segment)
        ret_arr[seg_idx] = viewpoint_feature
    return ret_arr

  def write_block_ply(self, xyzrs):
    """
    Write all segments in a molecular block to a ply file

    Notes
    -----
    This function is mainly for development purpose. Add this to the query method for each molecular block processing.
    """
    import open3d as o3d
    result_mesh = o3d.geometry.TriangleMesh()
    colors = utils.color_steps("inferno", steps=self.SEGMENT_NUMBER)
    for seg_idx in range(self.SEGMENT_NUMBER): 
      if len(xyzrs[seg_idx]) > 0: 
        _, mesh = commands.viewpoint_histogram_xyzr(xyzrs[seg_idx], np.array([0,0,0]), self.VIEW_BINNR, return_mesh=True)
        mesh.paint_uniform_color(colors[seg_idx])
        result_mesh += mesh
    o3d.io.write_triangle_mesh(f"block_{utils.get_timestamp()}.ply", result_mesh, write_ascii=True,   write_vertex_colors=True)
        

class RFFeatures(Feature):
  """
  The RFScore features for the protein-ligand interaction.
  For the 4*9 (36) features
  Rows (protein)   : C, N, O, S
  Columns (ligand) : C, N, O, F, P, S, Cl, Br, I
  """
  def __init__(self, search_cutoff=12, **kwargs):
    if "selection" not in kwargs.keys():
      raise ValueError("The selection should be set for the RFFeatures ")
    # NOTE: Hardcoded for the RFScore algorithm and correct dump of the feature array
    super().__init__(outshape=(None, 36, ), **kwargs)
    self.search_cutoff = search_cutoff
    self.pro_atom_idx = {6: 0, 7: 1, 8: 2, 16: 3}
    self.lig_atom_idx = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}

  def cache(self, trajectory):
    """
    Find the selection and its counterpart. 
    """
    super().cache(trajectory)
    if trajectory.top.select(self.selection).__len__() == 0:
      raise ValueError(f"The moiety of interest {self.selection} is not found in the topology")
    
    selected = trajectory.top.select(self.selection)
    counter_part = []
    for atom in trajectory.top.atoms: 
      if (atom.atomic_number in self.pro_atom_idx.keys()) and (atom.index not in selected):
        counter_part.append(atom.index)
    self.counter_part = np.array(counter_part, dtype=int)
    self.selected = np.array(selected, dtype=int)

  def query(self, topology, frames, focus):
    """
    Generate the features for the RFScore algorithm. This method requires the selection and its counterpart during cacheing. 
    
    Parameters
    ----------
    topology: traj-like
    frames: np.ndarray
    focus: np.ndarray

    Returns
    -------
    ret_arr: np.ndarray shaped (36, )

    Notes
    -----
    Nine atom types are considered: C, N, O, F, P, S, Cl, Br, I in the ligand part, and four atom types are considered in the protein part: C, N, O, S. Hence the output array contains explicitly 36 features. 
    """
    from scipy.spatial import KDTree
    if len(frames.shape) == 3:
      frames = frames[0]

    rf_arr = np.full((4, 9), 0.0, dtype=np.float32)
    kd_tree = KDTree(frames[self.counter_part])
    processed_atoms = []
    for idx in self.selected: 
      atomic_nr_sel = self.atomic_numbers[idx]
      coord_sel = frames[idx]
      soft_idxs = kd_tree.query_ball_point(coord_sel, self.search_cutoff)
      hard_idxs = self.counter_part[soft_idxs]
      for idx_prot in hard_idxs: 
        if idx_prot not in processed_atoms:  # NOTE: Make sure each atom is processed only once
          processed_atoms.append(idx_prot)
          atomic_nr_prot = self.atomic_numbers[idx_prot]
          if atomic_nr_sel in self.lig_atom_idx.keys() and atomic_nr_prot in self.pro_atom_idx.keys():
            rf_arr[self.pro_atom_idx[atomic_nr_prot], self.lig_atom_idx[atomic_nr_sel]] += 1
        else: 
          continue
    ret_arr = rf_arr.reshape(-1)
    return (ret_arr, )

  def run(self, ret_arr):
    """
    Just pass the feature array to the next dump function
    """
    return ret_arr
  

class Discretize(AtomType): 
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    super().cache(trajectory)
    
  def run(self, coords, weights):
    if len(coords) == 0:
      printit(f"{self.__class__.__name__}: Warning: The coordinates are empty")
      return np.zeros(self.dims, dtype=np.float32)
    
    ret = commands.discretize_coord(coords, weights, self.dims, self.spacing)
    
    if np.sum(np.isnan(ret)) > 0:
      printit(f"{self} Warning: The returned array has {np.isnan(ret).sum()} NaN values")
    # Check the sum of the absolute values of the returned array
    if config.verbose() or config.debug():
      ret_sum = np.sum(np.abs(ret))
      if np.isclose(ret_sum, 0):
        printit(f"{self} Warning: The sum of the returned array is zero")
    return ret
  