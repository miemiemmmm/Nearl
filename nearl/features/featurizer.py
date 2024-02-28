import tempfile, copy

import numpy as np
import pytraj as pt
from rdkit import Chem

from open3d.geometry import TriangleMesh

from nearl import utils
from nearl.io import hdf5

from . import fingerprint
from .. import printit, CONFIG, _verbose, _debug, draw_call_stack

__all__ = [
  "initialize_grid",
  "Featurizer3D",

]

def initialize_grid(thecenter, thelengths, thedims):
  thegrid = np.meshgrid(
    np.linspace(thecenter[0] - thelengths[0] / 2, thecenter[0] + thelengths[0] / 2, thedims[0]),
    np.linspace(thecenter[1] - thelengths[1] / 2, thecenter[1] + thelengths[1] / 2, thedims[1]),
    np.linspace(thecenter[2] - thelengths[2] / 2, thecenter[2] + thelengths[2] / 2, thedims[2]),
    indexing='ij'
  )
  thecoord = np.column_stack([thegrid[0].ravel(), thegrid[1].ravel(), thegrid[2].ravel()])
  return thegrid, thecoord

class Featurizer3D:
  def __init__(self, parms):
    """
    Initialize the featurizer with the given parameters
    """
    # Check the essential parameters for the featurizer
    self.parms = parms
    # parms_to_check = ["CUBOID_DIMENSION", "CUBOID_LENGTH", "MASK_INTEREST", "MASK_ENVIRONMENT"]
    parms_to_check = ["CUBOID_DIMENSION", "CUBOID_LENGTH"]
    for parm in parms_to_check:
      if parm not in parms:
        printit(f"Warning: Not found required parameter: {parm}. Please define the keyword <{parm}> in your parameter set. ")
        return
    # Basic parameters for the featurizer
    self.__dims = np.array([int(i) for i in parms["CUBOID_DIMENSION"]])
    self.__lengths = np.array([float(i) for i in parms["CUBOID_LENGTH"]])
    self.__resolutions = self.__lengths / self.__dims

    # Box related variables for feature mapping and generation
    self.__distances = np.arange(np.prod(self.__dims)).astype(int)
    self.__boxcenter = np.array([0, 0, 0]).astype(float)
    self.grid, self.__points3d = initialize_grid(self.__boxcenter, self.__lengths, self.__dims)

    # Identity vector generation related variables
    self.mesh = TriangleMesh()
    self.BOX_MOLS = {}
    self.SEGMENTNR = CONFIG.get("SEGMENT_LIMIT", 6)
    # Hard coded structural features (12)
    self.VPBINS = 12 + CONFIG.get("VIEWPOINT_BINS", 30)

    # Initialize the attributes for featurization

    self._fp_generator = None
    self._traj = None
    self._top = None
    self._active_frame_index = 0
    self._active_frame = None
    self.__status_flag = []

    self.FEATURESPACE = []
    self.FEATURENUMBER = 0
    self.trajloader = None
    self.FRAMES = []
    self.FRAMENUMBER = 0

    self.boxed_pdb = ""
    self.boxed_ply = ""
    self.boxed_indices = []
    self.contents = {}

    if _verbose:
      printit("Featurizer is initialized successfully")
      print("With Center at: ", self.__boxcenter)
      print("With Length at: ", self.__lengths)
      print("With Dimensions at: ", self.__dims)

  def __str__(self):
    finalstr = f"Feature Number: {self.FEATURENUMBER}; \n"
    for i in self.FEATURESPACE:
      finalstr += f"Feature: {i.__str__()}\n"
    return finalstr

  # Writable attributes
  @property
  def origin(self):
    return self.__points3d - self.__lengths / 2

  @origin.setter
  def origin(self, neworigin):
    assert len(neworigin) == 3, "The new origin should be a vector of length 3"
    diff = np.asarray(neworigin) - self.origin
    self.__boxcenter += diff
    self.__points3d += diff

  @property
  def center(self):
    return self.__boxcenter

  @center.setter
  def center(self, newcenter):
    assert len(newcenter) == 3, "The new center should be a vector of length 3"
    diff = np.asarray(newcenter) - self.__boxcenter
    # self.__boxcenter = np.asarray(newcenter)
    self.__boxcenter += diff
    self.__points3d += diff

  @property
  def lengths(self):
    return self.__lengths

  @lengths.setter
  def lengths(self, new_length):
    assert len(new_length) == 3, "The new lengths should be a vector of length 3"
    if isinstance(new_length, int) or isinstance(new_length, float):
      self.__lengths = np.array([new_length] * 3)
    elif isinstance(new_length, list) or isinstance(new_length, np.ndarray):
      assert len(new_length) == 3, "length should be 3"
      self.__lengths = np.array(new_length)
    else:
      raise Exception("Unexpected data type")

  @property
  def dims(self):
    return np.array(self.__dims)

  @property
  def distances(self):
    return self.__distances

  @dims.setter
  def dims(self, newdims):
    assert len(newdims) == 3, "The new dims should be a vector of length 3"
    if isinstance(newdims, int) or isinstance(newdims, float):
      self.__dims = np.array([newdims] * 3)
    elif isinstance(newdims, list) or isinstance(newdims, np.ndarray):
      assert len(newdims) == 3, "length should be 3"
      self.__dims = np.array(newdims)
    else:
      raise Exception("Unexpected data type")

  @property
  def traj(self):
    return self._traj

  @traj.setter
  def traj(self, the_traj):
    self._traj = the_traj
    self._top = self._traj.top.copy()

  @property
  def active_frame(self):
    return self._active_frame

  @active_frame.setter
  def active_frame(self, new_frame):
    self._active_frame = new_frame

  @property
  def active_frame_index(self):
    return self._active_frame_index

  @active_frame_index.setter
  def active_frame_index(self, new_index):
    self._active_frame_index = int(new_index)
    self._active_frame = self._traj[self._active_frame_index]
    # self._active_frame_index = np.where(np.isclose(self.traj.time - self._active_frame.time, 0))[0][0]

  @property
  def fp_generator(self):
    return self._fp_generator

  @fp_generator.setter
  def fp_generator(self, new_fp_generator):
    self._fp_generator = new_fp_generator

  @property
  def status_flag(self):
    return self.__status_flag

  @status_flag.setter
  def status_flag(self, newstatus):
    self.__status_flag.append(bool(newstatus))

  # Read-only attributes
  @property
  def points3d(self):
    return self.__points3d

  @property
  def resolutions(self):
    return self.__resolutions

  @property
  def top(self):
    return self._top

  def reset_status(self):
    self.__status_flag = []

  def translate(self, offsets, relative=True):
    """
    Apply a translational movement to the cell box
    Args:
      offsets: the offsets to be applied
      relative: whether the offsets are relative to the current position; if absolute, move the center to the offsets
    """
    if relative:
      self.__boxcenter += offsets
    else:
      self.__boxcenter = np.array(offsets)
    self.update_box()

  def update_box(self, center=None, lengths=None, dims=None):
    """
    Avoid frequent use of the update_box function because it generates new point set
    Only needed when changing the box parameter <CUBOID_DIMENSION> and <CUBOID_LENGTH>
    Basic variables: self.__boxcenter, self.__lengths, self.__dims
    """
    if center is not None:
      assert len(center) == 3, "The new center should be a vector of length 3"
      self.__boxcenter = np.array(center)
    if lengths is not None:
      assert len(lengths) == 3, "The new lengths should be a vector of length 3"
      self.__lengths = np.array(lengths)
    if dims is not None:
      assert len(dims) == 3, "The new dims should be a vector of length 3"
      self.__dims = np.array(dims)

    self.__resolutions = self.__lengths / self.__dims
    self.__distances = np.arange(np.prod(self.__dims)).astype(int)
    self.grid, self.__points3d = initialize_grid(self.__boxcenter, self.__lengths, self.__dims)


  def distance_to_index(self, point):
    """
    Convert distance to matrix coordinate
    """
    k0 = self.__dims[1] * self.__dims[2]
    k1 = self.__dims[0]
    d0 = int(point / k0)
    d1 = int((point - d0 * k0) / k1)
    d2 = int(point - d0 * k0 - d1 * k1)
    return d0, d1, d2

  def update_box_length(self, length=None, scale_factor=1.0):
    if length is not None:
      self.__lengths = float(length)
    else:
      self.__lengths *= scale_factor
    self.update_box()

  def map_to_grid(self, thearray, dtype=float):
    """
    Map the a 1D array (N*N*N, 1) to a 3D matrix (N, N, N)
    Args:
      thearray: A 1D array sized by (N*N*N, 1)
      dtype: The data type of the array
    """
    if len(self.__distances) != len(thearray):
      printit("Cannot match the length of the array to the 3D cuboid")
      return np.array([0])
    template = np.zeros(self.__dims).astype(dtype)
    for ind in self.__distances:
      _3d_idx = tuple(self.distance_to_index(ind))
      template[_3d_idx] = thearray[ind]
    return template

  def register_feature(self, feature):
    """
    Register a feature to the featurizer
    Args:
      feature: A feature object
    """
    self.FEATURESPACE.append(feature)
    for feature in self.FEATURESPACE:
      feature.set_featurizer(self)
    self.FEATURENUMBER = len(self.FEATURESPACE)

  def register_trajloader(self, trajloader):
    self.trajloader = trajloader

  def register_frames(self, theframes):
    """
    Register the frames to the featurizer for futher iteration
    Args:
      theframes: A list of frame indexes
    """
    self.FRAMES = theframes
    self.FRAMENUMBER = len(self.FRAMES)
    if self.traj is not None:
      self.active_frame_index = self.FRAMES[0]
      self.active_frame = self.traj[self.active_frame_index]

  def register_traj(self, thetraj):
    """
    Register a trajectory to the featurizer
    Args:
      thetraj: A trajectory object
    """
    # TODO: Make sure the trajectory related parameters are updated when the trajectory is changed
    self._traj = thetraj
    self._top = thetraj.top.copy()
    if self.FRAMENUMBER > 0:
      self.active_frame_index = self.FRAMES[0]
      self.active_frame = self.traj[self.active_frame_index]

  # DATABASE operation functions
  def connect(self, dataset):
    """
    Connect to a dataset
    Args:
      dataset: File path of the HDF file;
    """
    self.dataset = hdf5.hdf_operator(dataset, "a")

  def disconnect(self):
    """
    Disconnect the active dataset
    """
    self.dataset.close()

  def dump(self, key, data, dataset):
    """
    Dump the cached data to the active dataset
    """
    self.connect(dataset)
    try:
      dtypes = self.dataset.dtype(key)
      if not all([isinstance(i, float) for i in data[0]]):
        # A list of compound data types
        print("Using float format")
        converted_data = hdf5.array2dataset(data, dtypes)
      else:
        print("Using void format")
        converted_data = data
      self.dataset.append_entry(key, converted_data)
    except Exception as e:
      print(f"Error: {e}")
    self.disconnect()

  def selection_to_mol(self, selection):
    if isinstance(selection, str):
      atom_sel = self.traj.top.select(selection)
    elif isinstance(selection, (list, tuple, np.ndarray)):
      atom_sel = np.asarray(selection)
    else:
      raise ValueError(f"{__file__}: Unexpected selection type")

    try:
      rdmol = utils.traj_to_rdkit(self.traj, atom_sel, self.active_frame_index)
      if rdmol is None:
        with tempfile.NamedTemporaryFile(suffix=".pdb") as temp:
          outmask = "@"+",".join((atom_sel+1).astype(str))
          _traj = self.traj[outmask].copy_traj()
          pt.write_traj(temp.name, _traj, frame_indices=[self.active_frame_index], overwrite=True)
          with open(temp.name, "r") as tmp:
            print(tmp.read())
          rdmol = Chem.MolFromMol2File(temp.name, sanitize=False, removeHs=False)
      return rdmol
    except:
      return None


  def write_box(self, pdbfile="", elements=None, bfactors=None, write_pdb=False):
    """
    Write the 3D grid box with or without protein structure to a PDB file
    Args:
      pdbfile: str, optional, the output PDB file name. If not provided, the PDB formatted string is returned.
      elements: list, optional, a list of element symbols for each point. Default is a dummy atom "Du".
      bfactors: list, optional, a list of B-factor values for each point. Default is 0.0 for all points.
      write_pdb: bool, optional, if False, avoid writing PDB structure.
    Return:
      None if pdbfile is provided, otherwise a PDB formatted string representing the 3D points.
    """
    if elements is None:
      elements = ["Du"] * len(self.__distances)
    if bfactors is None:
      bfactors = [0.00] * len(self.__distances)
    template = "ATOM      1  Du  TMP     1       0.000   0.000   0.000  1.00  0.00"
    if write_pdb and len(self.traj) > 0:
      with tempfile.NamedTemporaryFile(suffix=".pdb") as file1:
        newxyz = np.array([self.traj[self.trajloader.activeframe].xyz])
        newtraj = pt.Trajectory(xyz=newxyz, top=self.traj.top)
        pt.write_traj(file1.name, newtraj, overwrite=True)
        with open(file1.name, "r") as file2:
          pdblines = [i for i in file2.read().split("\n") if "ATOM" in i or "HETATM" in i]
        pdbline = "\n".join(pdblines) + "\n"
    else:
      pdbline = ""
    for i in self.__distances:
      point = self.__points3d[i]
      elem = elements[i]
      bfval = bfactors[i]
      tmpstr = "".join([f"{i:>8.3f}" for i in point])
      thisline = f"ATOM  {i:>5}  {elem:<3}{template[16:30]}{tmpstr}{template[54:60]}{round(bfval, 2):>6}\n"
      pdbline += thisline
    if len(pdbfile) > 0:
      with open(pdbfile, "w") as file1:
        file1.write(pdbline)
    else:
      return pdbline

  # Primary function to pipeline computation streamline
  def run_by_atom(self, atoms, focus_mode=None):
    """
    Iteratively compute the features for each selected atoms (atom index) in the trajectory
    Args:
      atoms: list, a list of atom indexes
      focus_mode: str, optional, the focus mode. If None, use the default focus mode. Default is None.
    """
    # Step2: Initialize the feature array
    if focus_mode == "cog":
      repr_processed = np.zeros((self.FRAMENUMBER, self.SEGMENTNR * self.VPBINS))
      feat_processed = np.zeros((self.FRAMENUMBER, self.FEATURENUMBER)).tolist()
    else:
      repr_processed = np.zeros((self.FRAMENUMBER * len(atoms), self.SEGMENTNR * self.VPBINS))
      feat_processed = np.zeros((self.FRAMENUMBER * len(atoms), self.FEATURENUMBER)).tolist()

    # Compute the one-time-functions of each feature
    for feature in self.FEATURESPACE:
      feature.before_frame()

    # Step3: Iterate registered frames
    c_0 = 0       # For each focus, c_0 is the starting indexand c_1 is the ending index of the feature array
    c_total = 0   # Totally worked vectors from 0 to c_total
    fp_generator = fingerprint.generator(self.traj)
    for frame in self.FRAMES:
      self.active_frame_index = frame
      self.active_frame = self.traj[frame]
      if focus_mode == "cog":
        # Only focus on the center of geometry of the selected atoms
        focuses = np.mean(self.active_frame.xyz[atoms], axis=0).reshape((-1,3))
      else:
        # Focus on each selected atoms
        focuses = self.active_frame.xyz[atoms].reshape((-1,3))

      if _verbose or _debug:
        printit(f"Frame {frame}: Generated {len(focuses)} centers")
      # For each frame, run number of atoms times to compute the features/segmentations
      repr_vec, feat_vec = self.run_frame(focuses, fp_generator)

      c_1 = c_0 + len(repr_vec)
      c_total += len(repr_vec)
      repr_processed[c_0:c_1] = repr_vec
      feat_processed[c_0:c_1] = feat_vec
      c_0 = c_1

    for feature in self.FEATURESPACE:
      feature.after_frame()

    return repr_processed[:c_total], feat_processed[:c_total]

  def run_by_center(self, center):
    """
    Iteratively compute the features for each center (absolute coordinates) in the 3D space
    Args:
      center: list, a list of 3D coordinates
    """
    center = np.array(center)
    if (center.shape.__len__() < 2) and (center.shape[-1] != 3):
      raise ValueError("Center must be a list of 3 numbers")
    else:
      _centers = center[:3, :]
      center_number = len(_centers)
      # Step1: Initialize the MolBlock representation generator(required by the run_frame method)
      self.fp_generator = fingerprint.generator(self.traj)

      # Step2: Initialize the feature array
      id_processed = np.zeros((self.FRAMENUMBER * center_number, self.SEGMENTNR * self.VPBINS))
      feat_processed = np.zeros((self.FRAMENUMBER * center_number, self.FEATURENUMBER)).tolist()

      # Some run-time variables

      # Step3: Iterate registered frames
      c = 0
      c_total = 0
      for frame in self.FRAMES:
        self.active_frame_index = frame
        self.active_frame = self.traj[frame]
        self.fp_generator.frame = frame
        printit(f"Frame {frame}: Generated {center_number} centers")
        # For each frame, run number of centers times to compute the features/segmentations
        repr_vec, feat_vec = self.run_frame(_centers)

        c_1 = c + len(repr_vec)
        c_total += len(repr_vec)
        id_processed[c:c_1] = repr_vec
        feat_processed[c:c_1] = feat_vec
        c = c_1

      return id_processed[:c_total], feat_processed[:c_total]

  def run_frame(self, centers, fp_generator):
    """
    Generate the feature vectors for each center in the current frame
    Explicitly transfer the generator object to the function
    Needs to correctly set the box of the fingerprint.generator by desired center and lengths
    Args:
      centers: list, a list of 3D coordinates
      fp_generator: fingerprint.generator, the generator object
    """
    # Step1: Initialize the identity vector and feature vector
    centernr = len(centers)
    repr_vector = np.zeros((centernr, 6 * self.VPBINS))
    feat_vector = np.zeros((centernr, self.FEATURENUMBER)).tolist()
    mask = np.ones(centernr).astype(bool)  # Mask failed centers for run time rubustness

    fp_generator.frame = self.active_frame_index
    if _verbose:
      printit(f"Expected to generate {centernr} fingerprint anchors")

    # Compute the one-time-functions of each feature
    for feature in self.FEATURESPACE:
      feature.before_focus()

    # Step2: Iterate each center
    for idx, center in enumerate(centers):
      # Reset the focus of representation generator
      self.center = center
      fp_generator.set_box(self.center, self.lengths)
      # Segment the box and generate feature vectors for each segment
      segments = fp_generator.query_segments()

      # DEBUG ONLY
      if _verbose or _debug:
        seg_set = set(segments)
        seg_set.discard(0)
        printit(f"Found {len(seg_set)} non-empty segments", {i: j for i, j in zip(*np.unique(segments, return_counts=True)) if i != 0})

      """
      Compute the identity vector for the molecue block
      Identity generation is compulsory because it is the only hint to retrieve the feature block
      """
      feature_vector = fp_generator.vectorize()

      try:
        feature_vector.sum()
      except:
        print("Error: Feature vector is not generated correctly")
        print(feature_vector)
        print(feature_vector[0].shape)

      if np.count_nonzero(feature_vector) == 0 or None in fp_generator.mols:
        # Returned feature vector is all-zero, the featurization is most likely failed
        mask[idx] = False
        continue

      # Collect the results before processing the features
      # Use the dictionary as the result container to improve the flexibility (rather than attributes)
      self.contents = {
        "meshes": fp_generator.meshes,
        "final_mesh": fp_generator.final_mesh,
        "vertices": fp_generator.vertices,
        "faces": fp_generator.faces,
        "normals": fp_generator.normals,

        "segments": segments,
        "indices": fp_generator.indices,
        "indices_res": fp_generator.indices_res,
        "mols": fp_generator.mols,

        "pdb_result": fp_generator.get_pdb_string(),
        "ply_result": fp_generator.get_ply_string(),

      }

      if len(feature_vector) == 0:
        if _verbose:
          printit(f"Center {center} has no feature vector")
        mask[idx] = False
        continue
      repr_vector[idx] = feature_vector

      # Step2.5: Iterate different features
      for fidx, feature in enumerate(self.FEATURESPACE):
        feat_arr = feature.featurize()
        if isinstance(feat_arr, np.ndarray) and isinstance(feat_arr.dtype, (int, np.int32, np.int64,
          float, np.float32, np.float64, complex, np.complex64, np.complex128)):
          feat_arr = np.nan_to_num(feat_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        feat_vector[idx][fidx] = feat_arr

      # Clear the result contents
      self.contents = {}
    """
    Step3: Remove the masked identity vector and feature vector
    Final size of the feature vector: (number of centers, number of features)
    """
    ret_repr_vector = repr_vector[mask]
    ret_feat_vector = [item for item, use in zip(feat_vector, mask) if use]

    # Compute the one-time-functions of each feature
    for feature in self.FEATURESPACE:
      feature.after_focus()

    # DEBUG ONLY: After the iteration, check the shape of the feature vectors
    if _verbose:
      printit(f"Result identity vector: {ret_repr_vector.shape} ; Feature vector: {ret_feat_vector.__len__()} - {ret_feat_vector[0].__len__()}")
    return ret_repr_vector, ret_feat_vector

