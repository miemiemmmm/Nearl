import sys, tempfile, os
import subprocess, json

import numpy as np
import pytraj as pt
from rdkit import Chem
from scipy.spatial import KDTree

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
  "HBondDonor",
  "HBondAcceptor",
  "Hybridization",
  "Backbone",
  "AtomType",
  "PartialCharge",

  # Dynamic features
  "DensityFlow",
  "MarchingObservers",

  # Label-type features
  "Label_RMSD",
  "Label_PCDT",
  "Label_ResType",

  # Other features
  "VectorizerViewpoint", 
  
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
  mask_inbox = np.asarray(x_state_0 * x_state_1 * y_state_0 * y_state_1 * z_state_0 * z_state_1, dtype=bool)
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
  def __init__(self, 
    dims=None, spacing=None, 
    outfile=None, outkey=None,
    cutoff=None, sigma=None,
    padding=0, byres=None, 
    outshape=None, force_recache=False,
    **kwargs
  ):
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
    self.force_recache = force_recache
    self._force_recache = False if force_recache is False else True

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
      if config.verbose():
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
      if config.verbose():
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
    
    if (len(self.resids) != topology.n_atoms) or self.force_recache: 
      if config.verbose():
        printit(f"{self.__class__.__name__}: Dealing with inhomogeneous topology")
      if len(frame_coords.shape) == 2:
        self.cache(pt.Trajectory(xyz=np.array([frame_coords]), top=topology))
      else:
        self.cache(pt.Trajectory(xyz=frame_coords, top=topology))
    
    # Apply the padding to the box croppring
    new_coords = frame_coords - focal_point + self.center 
    mask = crop(new_coords, self.lengths, self.padding)

    # Get the boolean array of residues within the bounding box
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

    # Check the sum of the absolute values of the returned array
    ret_sum = np.sum(np.abs(ret))
    if np.isclose(ret_sum, 0):
      printit(f"{self.__class__.__name__} Warning: The sum of the returned array is zero")
    elif np.isnan(ret_sum):
      printit(f"{self.__class__.__name__} Warning: The returned array has {np.isnan(ret).sum()} NaN values")
    # Near the boundary, the coordinates might not sum up to the weights
    # elif not np.isclose(ret_sum, np.sum(weights)):
    #   printit(f"{self.__class__.__name__} Warning: The sum of the returned array {ret_sum} is not equal to the sum of the weights {np.sum(weights)}")
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
      if self.outshape is not None or self._outshape != False: 
        # Explicitly set the shape of the output
        utils.append_hdf_data(self.outfile, self.outkey, np.asarray([result], dtype=np.float32), dtype=np.float32, maxshape=(None, *self.outshape), chunks=True, compression="gzip", compression_opts=4)
      elif len(self.dims) == 3: 
        utils.append_hdf_data(self.outfile, self.outkey, np.asarray([result], dtype=np.float32), dtype=np.float32, maxshape=(None, *self.dims), chunks=True, compression="gzip", compression_opts=4)


class AtomicNumber(Feature):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    super().cache(trajectory)
  
  def query(self, topology, frame_coords, focal_point): 
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = np.array([self.atomic_numbers[i] for i in self.atomic_numbers[idx_inbox]], dtype=np.float32)
    coord_inbox = coord_inbox - focal_point + self.center
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
    
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = np.array([self.mass[i] for i in self.atomic_numbers[idx_inbox]], dtype=np.float32)
    coord_inbox = coord_inbox - focal_point + self.center
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
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.heavy_atoms[idx_inbox]
    # Translate the result coordinates to the center of the box
    coord_inbox = coord_inbox - focal_point + self.center
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
      self.atoms_aromatic = np.asarray([1 if i == 0 else 0 for i in atoms_aromatic], dtype=np.float32)
    else: 
      self.atoms_aromatic = np.asarray(atoms_aromatic, dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.atoms_aromatic[idx_inbox]
    coord_inbox = coord_inbox - focal_point + self.center
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
      self.atoms_in_ring = np.asarray([1 if i == 0 else 0 for i in atoms_in_ring], dtype=np.float32)
    else: 
      self.atoms_in_ring = np.asarray(atoms_in_ring, dtype=np.float32)
  
  def query(self, topology, frame_coords, focal_point): 
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.atoms_in_ring[idx_inbox]
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights


class HBondDonor(Feature):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_hbond_donor = chemtools.label_hbond_donor(fpt.name)

    if len(atoms_hbond_donor) != trajectory.n_atoms:
      printit(f"{self.__class__.__name__} Warning: The number of atoms in PDB does not match the number of aromaticity values")
    self.atoms_hbond_donor = np.asarray(atoms_hbond_donor, dtype=np.float32)
  
  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.atoms_hbond_donor[idx_inbox]
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights


class HBondAcceptor(Feature):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_hbond_acceptor = chemtools.label_hbond_acceptor(fpt.name)

    if len(atoms_hbond_acceptor) != trajectory.n_atoms:
      printit(f"{self.__class__.__name__} Warning: The number of atoms in PDB does not match the number of aromaticity values")
    self.atoms_hbond_acceptor = np.asarray(atoms_hbond_acceptor, dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.atoms_hbond_acceptor[idx_inbox]
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights


class Hybridization(Feature):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def cache(self, trajectory):
    super().cache(trajectory)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fpt:
      pt.write_traj(fpt.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      atoms_hybridization = chemtools.label_hybridization(fpt.name)

    if len(atoms_hybridization) != trajectory.n_atoms:
      printit(f"{self.__class__.__name__} Warning: The number of atoms in PDB does not match the number of aromaticity values")
    self.atoms_hybridization = np.asarray(atoms_hybridization, dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.atoms_hybridization[idx_inbox]
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights

class Backbone(Feature):
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
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.backboneness[idx_inbox]
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights
  

class AtomType(Feature):
  def __init__(self, focus_element, **kwargs):
    super().__init__(**kwargs)
    self.focus_element = focus_element

  def cache(self, trajectory):
    super().cache(trajectory)
    self.atom_type = np.array([1 if i == self.focus_element else 0 for i in trajectory.top.atoms], dtype=np.float32)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.atom_type[idx_inbox]
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights


# /MieT5/BetaPose/nearl/data/charge_charmm36.json   # self.mode = "charmm36"
# /MieT5/BetaPose/nearl/data/charge_ff14sb.json     # self.mode = "ff14sb"
# self.charge_type = "eem" # self.charge_parm = "EEM_00_NEEMP_ccd2016_npa"
# self.charge_type = "peoe" # self.charge_parm = "PEOE_00_original"
class PartialCharge(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Atomic charge feature for the structure of interest;
  """
  def __init__(self, charge_type="qeq", charge_parm="QEq_00_original", **kwargs):
    super().__init__(**kwargs)
    # The following types are supported by ChargeFW: 
    [ "sqeqp",  "eem",  "abeem",  "sfkeem",  "qeq",
      "smpqeq",  "eqeq",  "eqeqc",  "delre",  "peoe",
      "mpeoe",  "gdac",  "sqe",  "sqeq0",  "mgc",
      "kcm",  "denr",  "tsef",  "charge2",  "veem",
      "formal"
    ]

    self.charge_type = charge_type
    self.charge_parm = charge_parm

  def cache(self, trajectory):
    super().cache(trajectory)

    import chargefw2_python as cfw
    charges = None
    charge_values = None
    with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
      pt.write_traj(f.name, trajectory, format="pdb", frame_indices=[0], overwrite=True)
      try: 
        mol = cfw.Molecules(f.name)
        charges = cfw.calculate_charges(mol, self.charge_type, self.charge_parm)
        # File name is the keyword of the returned charge values
        filename = os.path.basename(f.name).split(".")[0]
        if filename not in charges.keys():
          raise ValueError(f"{self.__class__.__name__} The charges are not calculated for the file {filename}")
        else: 
          charge_values = np.array(charges[filename], dtype=np.float32)
          if len(charge_values) != trajectory.n_atoms:
            raise ValueError(f"{self.__class__.__name__} The number of charge values does not match the number of atoms in the trajectory")
          else: 
            self.charge_values = charge_values
      except Exception as e: 
        subprocess.call(["cp", f.name, "/tmp/chargefailed.pdb"])
        printit(f"{self.__class__.__name__} Warning: Failed to calculate molecular charge because of the following error:\n {e}")

    if charges is None or charge_values is None:
      printit(f"{self.__class__.__name__} Warning: The charge values are not set", file=sys.stderr)
      self.charge_values = np.zeros(trajectory.n_atoms)

  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]  
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.charge_values[idx_inbox] 
    # Translate the result coordinates to the center of the box
    coord_inbox = coord_inbox - focal_point + self.center
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
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.electronegativity[idx_inbox]
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights
  

class HydrophobicityFeature(Feature):
  def __int__(self):
    super().__init__()

  def cache(self, trajectory):
    super().cache(trajectory)
    elecnegs = np.array([constants.ELECNEG[i] for i in self.atomic_numbers], dtype=np.float32)
    self.hydrophobicity = np.abs(elecnegs - constants.ELECNEG[6])
  
  def query(self, topology, frame_coords, focal_point):
    if frame_coords.shape.__len__() == 3: 
      frame_coords = frame_coords[0]
    idx_inbox = super().query(topology, frame_coords, focal_point)
    coord_inbox = frame_coords[idx_inbox]
    weights = self.hydrophobicity[idx_inbox]
    coord_inbox = coord_inbox - focal_point + self.center
    return coord_inbox, weights

class DynamicFeature(Feature):
  def __init__(self, agg="mean", weight_type="mass", **kwargs):
    super().__init__(**kwargs)
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
    super().cache(trajectory)   # Obtain the atomic number and residue IDs
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

    Notes
    -----
    - MAX_ALLOWED_ATOMS: Depends on the GPU cache size for each thread

    - DEFAULT_COORD: The coordinates for padding of atoms in the box across required frames. Also hard coded in GPU code. 

    The return weight array should be flattened to a 1D array
    """
    assert len(frame_coords.shape) == 3, f"Warning from feature ({self.__str__()}): The coordinates should follow the convention (frames, atoms, 3); "
    MAX_ALLOWED_ATOMS = 1000 
    DEFAULT_COORD = 99999.0

    coords = np.full((len(frame_coords), MAX_ALLOWED_ATOMS, 3), DEFAULT_COORD, dtype=np.float32)
    weights = np.full((len(frame_coords), MAX_ALLOWED_ATOMS), 0.0, dtype=np.float32)
    max_atom_nr = 0
    for idx, frame in enumerate(frame_coords):
      idx_inbox = super().query(topology, frame, focal_point)
      atomnr_inbox = np.count_nonzero(idx_inbox)
      if atomnr_inbox > MAX_ALLOWED_ATOMS:
        printit(f"{self.__class__.__name__} Warning: the maximum allowed atom slice is {MAX_ALLOWED_ATOMS} but the maximum atom number is {atomnr_inbox}")
      atomnr_inbox = min(atomnr_inbox, MAX_ALLOWED_ATOMS)
      coord_inbox = frame[idx_inbox] - focal_point + self.center
      coords[idx, :atomnr_inbox] = coord_inbox[:atomnr_inbox]
      weights[idx, :atomnr_inbox] = self.cached_weights[idx_inbox][:atomnr_inbox]
      max_atom_nr = max(max_atom_nr, atomnr_inbox)

    # Prepare the return arrays
    ret_coord = np.array(coords[:, :max_atom_nr], dtype=np.float32)
    ret_weight = np.array(weights[:, :max_atom_nr].flatten(), dtype=np.float32)
    return ret_coord, ret_weight
  
  def run(self, frames, weights):
    """
    Run the dynamic feature algorithm
    """
    raise NotImplementedError("The run function should be implemented in the child class")
  

class DensityFlow(DynamicFeature):
  """
  Dynamic feature: Density flow,

  Aggregation type: mean, std, sum

  Weight type: mass, radius, residue_id, sidechainness, uniform, atomid
  """
  # agg = "mean", # weight_type="mass", 
  def __init__(self, **kwargs):
    super().__init__(**kwargs) 

  def query(self, topology, frame_coords, focal_point):
    ret_coord, ret_weight = super().query(topology, frame_coords, focal_point)
    return ret_coord, ret_weight

  def run(self, frames, weights):
    """
    Take frames of coordinates and weights and return the a feature array with the same dimensions as self.dims
    """
    # Run the density flow algorithm
    assert frames.shape[0] * frames.shape[1] == len(weights), f"The production of frame_nr and atom_nr has to be equal to the number of weights: {frames.shape[0] * frames.shape[1]}/{len(weights)}"
    ret_arr = commands.voxelize_trajectory(frames, weights, self.dims, self.spacing, self.cutoff, self.sigma, self.agg)
    return ret_arr.reshape(self.dims)  


class MarchingObservers(DynamicFeature): 
  """
  Perform the marching observers algorithm to get the dynamic feature. 

  Inherit from the DynamicFeature class since there are common ways to query the coordinates and weights. 

  Observation types: particle_count, particle_existance, mean_distance, radius_of_gyration

  Weight types: mass, radius, residue_id, sidechainness, uniform

  Aggregation type: mean, std, sum
  
  """
  def __init__(self, obs="particle_existance", **kwargs): 
    # Just omit the sigma parameter while the inheritance. 
    # while initialization of the parent class, weight_type, cutoff, agg are set
    super().__init__(**kwargs)
    self.__obs_type = obs     # Directly pass to the CUDA voxelizer

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
    selection=None, selection_type=None, base_value=0, 
    outshape=(None,), 
    **kwargs
  ): 
    super().__init__(outshape = outshape, **kwargs) 
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
    

class Label_PCDT(Feature): 
  def __init__(self, 
    outkey=None, outfile=None, outshape=(None,),
    selection=None, selection_type=None, base_value=0, 
    **kwargs
  ): 
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
    idx_inbox = super().query(topology, frames, focus)
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
    if config.verbose():
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

