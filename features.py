import os, subprocess, copy
import time, datetime
import builtins, json, tempfile, functools
import numpy as np
import pytraj as pt

from numba import jit
from scipy.interpolate import griddata, Rbf
from scipy.spatial import KDTree;
from scipy.spatial.distance import cdist;
from scipy.stats import entropy;

from rdkit import Chem
from rdkit.Chem import AllChem;

# open3d related modules
from open3d.io import write_triangle_mesh
from open3d.pipelines.registration import compute_fpfh_feature
from open3d.geometry import KDTreeSearchParamHybrid, TriangleMesh, PointCloud

from . import utils, CONFIG, printit, representations
from BetaPose import chemtools

_clear = CONFIG.get("clear", False);
_verbose = CONFIG.get("verbose", False);

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

@jit(nopython=True)
def entropy_jit(x):
  x_set = list(set(x))
  counts = np.zeros(len(x_set))
  for xi in x:
    counts[x_set.index(xi)] += 1;
  probs = counts/len(x);
  entropy = -np.sum(probs * np.log2(probs + 1e-10))  # add small constant to avoid log2(0)
  return entropy

class Featurizer3D:
  def __init__(self, parms):
    """
    Initialize the featurizer with the given parameters
    parms: a dictionary of parameters
    """
    # Check the essential parameters for the featurizer
    self.parms = parms;
    # parms_to_check = ["CUBOID_DIMENSION", "CUBOID_LENGTH", "MASK_INTEREST", "MASK_ENVIRONMENT"]
    parms_to_check = ["CUBOID_DIMENSION", "CUBOID_LENGTH"];
    for parm in parms_to_check:
      if parm not in parms:
        printit(f"Warning: Not found required parameter: {parm}. Please define the keyword <{parm}> in your parameter set. ")
        return
    # Basic parameters for the featurizer
    self.__dims = np.array([int(i) for i in parms["CUBOID_DIMENSION"]]);
    self.__lengths = np.array([float(i) for i in parms["CUBOID_LENGTH"]]);

    # Zero feature space
    self.FEATURESPACE = [];
    self.FEATURENUMBER = 0;

    # Box related variables for feature mapping and generation
    self.__distances = np.arange(np.prod(self.__dims)).astype(int);
    self.__boxcenter = np.array([0, 0, 0]);
    self.__points3d = self.get_points();
    self.__grid = np.arange(np.prod(self.__dims)).reshape(self.__dims);

    # Identity vector generation related variables
    self.mesh = TriangleMesh();
    self.BOX_MOLS = {};
    self.active_frame_index = 0;
    self.SEGMENTNR = CONFIG.get("SEGMENT_LIMIT", 6);
    self.VPBINS = 12 + CONFIG.get("VIEWPOINT_BINS", 30);

    ##########################################################
    if _verbose:
      printit("Featurizer is initialized successfully")
      print("With Center at: ", self.__boxcenter);
      print("With Length at: ", self.__lengths);
      print("With Dimensions at: ", self.__dims);

  def __str__(self):
    finalstr = f"Feature Number: {len(self.FEATURENUMBER)}; \n"
    for i in self.FEATURESPACE:
      finalstr += f"Feature: {i.__str__()}\n"
    return finalstr

  @property
  def origin(self):
    return np.array(self.__points3d[0]);

  @origin.setter
  def origin(self, neworigin):
    diff = np.array(neworigin) - np.array(neworigin);
    self.__boxcenter += diff;
    self.__points3d += diff;

  @property
  def center(self):
    return np.array(self.__boxcenter);

  @center.setter
  def center(self, newcenter):
    diff = np.array(newcenter) - np.mean(self.__points3d, axis=0);
    self.__boxcenter = np.array(newcenter);
    self.__points3d += diff;

  @property
  def lengths(self):
    return np.array(self.__lengths);

  @lengths.setter
  def lengths(self, new_length):
    if isinstance(new_length, int) or isinstance(new_length, float):
      self.__lengths = np.array([new_length] * 3);
    elif isinstance(new_length, list) or isinstance(new_length, np.ndarray):
      assert len(new_length) == 3, "length should be 3"
      self.__lengths = np.array(new_length);
    else:
      raise Exception("Unexpected data type")

  @property
  def dims(self):
    return np.array(self.__dims)
  @property
  def points3d(self):
    return self.__points3d

  @property
  def resolutions(self):
    return self.__resolutions

  def translate(self, offsets, relative=True, **kwarg):
    """
    Apply a translational movement to the cell box;
    """
    if relative:
      self.__boxcenter += offsets;
    else:
      self.__boxcenter = np.array(offsets);
    self.updatebox();
    return

  def updatebox(self):
    """
    Avoid frequent use of the updatebox function because it generates new point set
    Only needed when changing the box parameter <CUBOID_DIMENSION> and <CUBOID_LENGTH>
    Basic variables: self.__lengths, self.__dims
    """
    self.__resolutions = self.__lengths / self.__dims;
    self.__distances = np.arange(np.prod(self.__dims)).astype(int);
    self.__points3d = self.get_points();
    self.__boxcenter = np.mean(self.__points3d, axis=0);

  def get_points(self):
    # Generate grid points
    self.grid = np.mgrid[self.center[0] - self.lengths[0] / 2:self.center[0] + self.lengths[0] / 2:self.dims[0] * 1j,
                self.center[1] - self.lengths[1] / 2:self.center[1] + self.lengths[1] / 2:self.dims[1] * 1j,
                self.center[2] - self.lengths[2] / 2:self.center[2] + self.lengths[2] / 2:self.dims[2] * 1j]
    coord3d = np.column_stack([self.grid[0].ravel(), self.grid[1].ravel(), self.grid[2].ravel()])
    return coord3d

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
    self.updatebox()

  def points_to_3D(self, thearray, dtype=float):
    """
    Convert a 1D array to a 3D cuboid
    Args:
      thearray: A 1D array
    """
    if len(self.__distances) != len(thearray):
      printit("Cannot match the length of the array to the 3D cuboid");
      return np.array([0])
    template = np.zeros((self.__pointnr, self.__pointnr, self.__pointnr)).astype(dtype);
    for ind in self.__distances:
      array_3Didx = tuple(self.__indexes3d[ind]);
      template[array_3Didx] = thearray[ind]
    return template

  def register_feature(self, feature):
    """
    Register a feature to the featurizer
    Args:
      feature: A feature object
    """
    self.FEATURESPACE.append(feature);
    for feature in self.FEATURESPACE:
      feature.set_featurizer(self)
    self.FEATURENUMBER = len(self.FEATURESPACE);

  def register_frames(self, theframes):
    """
    Register the frames to the featurizer for futher iteration
    Args:
      theframes: A list of frame indexes
    """
    self.FRAMES = theframes;
    self.FRAMENUMBER = len(self.FRAMES);

  def register_traj(self, thetraj):
    """
    Register a trajectory to the featurizer
    Args:
      thetraj: A trajectory object
    """
    # TODO: Make sure the trajectory related parameters are updated when the trajectory is changed
    self.traj = thetraj;
    self.top = thetraj.top.copy();

  ####################################################################################################
  ######################################## DATABASE operation ########################################
  ####################################################################################################
  def connect(self, dataset):
    """
    Connect to a dataset
    Args:
      dataset: File path of the HDF file;
    """
    self.dataset = data_io.hdf_operator(dataset)

  def disconnect(self):
    """
    Disconnect the active dataset
    """
    self.dataset.close()

  def dump(self, key, data, dataset):
    """
    Dump the cached data to the active dataset
    """
    self.connect(dataset);
    try:
      dtypes = self.dataset.dtype(key);
      if not all([isinstance(i, float) for i in data[0]]):
        ################################## A list of compound data types ###################################
        print("Using float format")
        converted_data = data_io.array2dataset(data, dtypes);
      else:
        print("Using void format")
        converted_data = data
      self.dataset.append_entry(key, converted_data);
    except Exception as e:
      print(f"Error: {e}")
    self.disconnect();

  def boxed_to_mol(self, selection):
    if isinstance(selection, str):
      atom_sel = self.traj.top.select(selection);
    elif isinstance(selection, (list, tuple, np.ndarray)):
      atom_sel = selection;

    string_prep = f"{self.traj.top_filename}%{len(atom_sel)}%{atom_sel}";
    string_hash = utils.get_hash(string_prep);
    if string_hash in self.BOX_MOLS.keys():
      return self.BOX_MOLS[string_hash];

    # try:
    if True:
      print("======>>>> Generating a new boxed molecule <<<<======")
      rdmol = chemtools.traj_to_rdkit(self.traj, atom_sel, self.active_frame_index);
      if rdmol is None:
        with tempfile.NamedTemporaryFile(suffix=".pdb") as temp:
          outmask = "@"+",".join((atom_sel+1).astype(str));
          print(self.traj[outmask])
          _traj = self.traj[outmask].copy_traj();
          print(f"======>>>> Generating a new trajectory object: {_traj.n_frames} frames <<<<======")
          pt.write_traj(temp.name, _traj, frame_indices=[self.active_frame_index], overwrite=True);

          with open(temp.name, "r") as f:
            print(f.read());

          rdmol = Chem.MolFromMol2File(temp.name, sanitize=False, removeHs=False);
          print("RdMol ==>> ", rdmol)
      self.BOX_MOLS[string_hash] = rdmol;
      return rdmol;
      # pdbstr = chemtools.write_pdb_block(self.traj, atom_sel, frame_index=self.active_frame_index);
      # rdmol = Chem.MolFromPDBBlock(pdbstr, sanitize=False, removeHs=False);
      # rdmol = chemtools.sanitize_bond(rdmol);
      # Chem.SanitizeMol(rdmol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS);
      # rdmol = Chem.AddHs(rdmol, addCoords=True);
      # AllChem.EmbedMolecule(rdmol);
      # AllChem.ComputeGasteigerCharges(rdmol);
      # # Double check the number of atoms and the match of atom names
      # if len(atom_sel) != rdmol.GetNumAtoms():
      #   printit("Warning: The number of atoms in the box does not match the number of atoms in the trajectory");
      # traj_atoms = [i for i in self.traj.top.atoms];
      # for atomidx, atom in zip(atom_sel, rdmol.GetAtoms()):
      #   atomname = atom.GetPDBResidueInfo().GetName();
      #   if np.isnan(atom.GetDoubleProp('_GasteigerCharge')):
      #     if atomsymbol in chemtools.DEFAULT_PARTIAL_CHARGE.keys():
      #       atom.SetDoubleProp('_GasteigerCharge', chemtools.DEFAULT_PARTIAL_CHARGE[atomsymbol]);
      #     else:
      #       printit("Warning: Cannot determine the partial charge of the atom {atomname}, settting to 0");
      #       atom.SetDoubleProp('_GasteigerCharge', 0.0);
      #   if atomname.strip() != traj_atoms[atomidx].name:
      #     printit(f"Warning: Atom name mismatch between pytraj {traj_atoms[atomidx]} and rdkit{atomname}");
    # except Exception as e:
    #   printit(f"The following Exception occurred while reading pdb into rdkit mol: {e}")
    #   return None


  def write_box(self, pdbfile="", elements=[], bfactors=[], write_pdb=False):
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
    if len(elements) == 0:
      elements = ["Du"] * len(self.__distances);
    if len(bfactors) == 0:
      bfactors = [0.00] * len(self.__distances);
    template = "ATOM      1  Du  TMP     1       0.000   0.000   0.000  1.00  0.00";
    if write_pdb and len(self.traj) > 0:
      with tempfile.NamedTemporaryFile(suffix=".pdb") as file1:
        newxyz = np.array([self.traj[self.trajloader.activeframe].xyz])
        newtraj = pt.Trajectory(xyz=newxyz, top=self.traj.top)
        pt.write_traj(file1.name, newtraj, overwrite=True)
        with open(file1.name, "r") as file2:
          pdblines = [i for i in file2.read().split("\n") if "ATOM" in i or "HETATM" in i]
        pdbline = "\n".join(pdblines) + "\n"
    else:
      pdbline = "";
    coordinates = np.round(self.__points3d, decimals=3);
    for i in self.__distances:
      point = self.__points3d[i];
      elem = elements[i];
      bfval = bfactors[i];
      tmpstr = "".join([f"{i:>8.3f}" for i in point]);
      thisline = f"ATOM  {i:>5}  {elem:<3}{template[16:30]}{tmpstr}{template[54:60]}{round(bfval, 2):>6}\n"
      pdbline += thisline
    if len(pdbfile) > 0:
      with open(pdbfile, "w") as file1:
        file1.write(pdbline)
    else:
      return pdbline

  ####################################################################################################
  ####################################### Perform Computation ########################################
  ####################################################################################################
  def run_by_atom(self, atoms, fbox_length="same", focus_mode=None):
    """
    Iteratively compute the features for each selected atoms (atom index) in the trajectory
    Args:
      atoms: list, a list of atom indexes
      fbox_length: str or list, optional, the length of the 3D grid box. If "same", use the same length as the
      trajectory. If a list of 3 numbers, use the provided length. Default is "same".
    """
    # Step1: Initialize the MolBlock representation generator(required by the runframe method)
    self.repr_generator = representations.generator(self.traj);
    if fbox_length == "same":
      self.repr_generator.length = [i for i in self.__lengths];
    elif (not isinstance(fbox_length, str)) and len(fbox_length) == 3:
      self.repr_generator.length = [i for i in fbox_length];

    # Step2: Initialize the feature array
    repr_processed = np.zeros((self.FRAMENUMBER * len(atoms), self.SEGMENTNR * self.VPBINS));
    feat_processed = np.zeros((self.FRAMENUMBER * len(atoms), self.FEATURENUMBER)).tolist();

    # Compute the one-time-functions of each feature
    for feature in self.FEATURESPACE:
      feature.before_frame();

    # Step3: Iterate registered frames
    c = 0;
    c_total = 0;
    for frame in self.FRAMES:
      self.active_frame_index = frame;
      self.active_frame = self.traj[frame];
      if focus_mode == "cog":
        focuses = np.array([self.active_frame.xyz[atoms].mean(axis=0)]);
      else:
        focuses = self.active_frame.xyz[atoms];
      self.repr_generator.frame = frame;
      printit(f"Frame {frame}: Generated {len(focuses)} centers");
      # For each frame, run number of atoms times to compute the features/segmentations
      repr_vec, feat_vec = self.runframe(focuses);

      c_1 = c + len(repr_vec);
      c_total += len(repr_vec);
      repr_processed[c:c_1] = repr_vec;
      feat_processed[c:c_1] = feat_vec;
      c = c_1;
    for feature in self.FEATURESPACE:
      feature.after_frame();
    return repr_processed[:c_total], feat_processed[:c_total]

  def run_by_center(self, center):
    """
    Iteratively compute the features for each centers (absolute coordinates) in the 3D space
    Args:
      center: list, a list of 3D coordinates
    """
    center = np.array(center)
    if (center.shape.__len__() < 2) and (center.shape[-1] != 3):
      raise ValueError("Center must be a list of 3 numbers");
    else:
      _centers = center[:3,: ];
      center_number = len(_centers);
      # Step1: Initialize the MolBlock representation generator(required by the runframe method)
      self.repr_generator = representations.generator(self.traj);

      # Step2: Initialize the feature array
      id_processed = np.zeros((self.FRAMENUMBER * center_number, self.SEGMENTNR * self.VPBINS));
      feat_processed = np.zeros((self.FRAMENUMBER * len(atoms), self.FEATURENUMBER)).tolist();

      # Step3: Iterate registered frames
      c = 0;
      c_total = 0;
      for frame in self.FRAMES:
        self.active_frame_index = frame;
        self.active_frame = self.traj[frame];
        self.repr_generator.frame = frame;
        printit(f"Frame {frame}: Generated {center_number} centers");
        # For each frame, run number of centers times to compute the features/segmentations
        repr_vec, feat_vec = self.runframe(_centers);

        c_1 = c + len(repr_vec);
        c_total += len(repr_vec);
        id_processed[c:c_1] = repr_vec;
        feature_processed[c:c_1] = feat_vec;
        c = c_1;
      return id_processed[:c_total], feature_processed[:c_total]


  def runframe(self, centers):
    """
    Generate the feature vectors for each center in the current frame
    Trajectory already loaded in self.repr_generator

    Needs to correctly set the self.repr_generator.center and self.repr_generator.lengths
    Args:
      centers: list, a list of 3D coordinates
    """
    """Step1: Initialize the identity vector and feature vector"""
    repr_vector = np.zeros((len(centers), 6 * (12 + CONFIG.get("VIEWPOINT_BINS", 30))));
    feat_vector = np.zeros((len(centers), self.FEATURENUMBER)).tolist();
    # By default, all centers are valid; Mask failed centers in the following steps
    mask = np.ones(len(centers), dtype=bool);
    if _verbose:
      printit(f"Expected to generate {len(centers)} identity vectors");
    # Compute the one-time-functions of each feature
    for feature in self.FEATURESPACE:
      feature.before_focus();

    """Step2: Iterate each center"""
    for idx, center in enumerate(centers):
      # Reset the focus of representation generator
      self.center = center;
      self.repr_generator.center = self.center;
      self.repr_generator.length = self.lengths;
      # Segment the box and generate feature vectors for each segment
      slices, segments = self.repr_generator.slicebyframe();

      # DEBUG ONLY
      if _verbose:
        printit(f"Found {len(set(segments))-1} non-empty segments", {i:j for i,j in zip(*np.unique(segments, return_counts=True)) if i != 0});

      """
      Compute the identity vector for the molecue block
      Identity generation is compulsory because it is the only hint to retrieve the feature block
      """
      feature_vector, mesh_objs = self.repr_generator.vectorize(segments);
      if np.count_nonzero(feature_vector) == 0 or len(mesh_objs) == 0:
        # Returned feature vector is all-zero, the featurization is most likely failed
        mask[idx] = False
        continue
      final_mesh = functools.reduce(lambda a, b: a + b, mesh_objs);
      # Keep the intermediate information as metainformation if further review/featurization is needed
      if _verbose:
        printit("Final mesh generated", final_mesh);
      self.mesh = copy.deepcopy(final_mesh);
      self.boxed_pdb = self.repr_generator.active_pdb;
      self.boxed_ply = self.repr_generator.active_ply;
      self.boxed_indices = self.repr_generator.active_indices;
      if len(feature_vector) == 0:
        if _verbose:
          printit(f"Center {center} has no feature vector");
        mask[idx] = False
        continue
      repr_vector[idx] = feature_vector;

      # Identity vector computation time only occupies 10% of the total time
      # E.G. 1.686 (clock time) vs 0.174 (cpu time)

      """Step2.5: Iterate different features"""
      for fidx, feature in enumerate(self.FEATURESPACE):
        feat_arr = feature.featurize();
        if isinstance(feat_arr, np.ndarray):
          if isinstance(feat_arr.dtype, (int, float, complex, np.float32, np.float64,
                                         np.int32, np.int64, np.complex64, np.complex128)):
            feat_arr = np.nan_to_num(feat_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0);
        feat_vector[idx][fidx] = feat_arr

    """
    Step3: Remove the masked identity vector and feature vector
    Final size of the feature vector: (number of centers, number of features)
    """
    ret_repr_vector = repr_vector[mask];
    ret_feat_vector = [item for item, use in zip(feat_vector, mask) if use];
    # Compute the one-time-functions of each feature
    for feature in self.FEATURESPACE:
      feature.after_focus();
    # DEBUG ONLY: After the iteration, check the shape of the feature vectors
    if _verbose:
      printit(f"Result identity vector: {ret_repr_vector.shape} ; Feature vector: {ret_feat_vector.__len__()} - {ret_feat_vector[0].__len__()}");
    return ret_repr_vector, ret_feat_vector


class Feature:
  """
  The base class for all features
  """
  def __init__(self):
    print(f"Initializing the feature base class {self.__class__.__name__}");
  def __str__(self):
    return self.__class__.__name__

  def set_featurizer(self, featurizer): 
    """
    Hook the feature generator back to the feature convolutor and obtain necessary attributes from the featurizer
    including the trajectory, active frame, convolution kernel etc
    """
    self.featurizer = featurizer
    if _verbose:
      printit(f"Hooking featurizer to {self.__class__.__name__}");

  """
  The Feature can ONLY READ the necessary attributes of the featurizer, but not udpate them.
  """
  @property
  def active_frame(self):
    return self.featurizer.active_frame;
  @property
  def active_frame_index(self):
    return self.featurizer.active_frame_index;
  @property
  def traj(self):
    return self.featurizer.traj;
  @property
  def top(self):
    return self.featurizer.traj.top;
  @property
  def center(self):
    return np.asarray(self.featurizer.center);
  @property
  def lengths(self):
    return np.asarray(self.featurizer.lengths);
  @property
  def dims(self):
    return np.asarray(self.featurizer.dims);
  @property
  def grid(self):
    return np.array(self.featurizer.grid);
  @property
  def points3d(self):
    return np.array(self.featurizer.points3d);

  def crop_box(self, points):
    """
    Crop the points to the box defined by the center and lengths
    The mask is returned as a boolean array
    """
    thecoord = np.asarray(points);
    upperbound = self.center + self.lengths / 2;
    lowerbound = self.center - self.lengths / 2;
    ubstate = np.all(thecoord < upperbound, axis=1);
    lbstate = np.all(thecoord > lowerbound, axis=1);
    mask_inbox = ubstate * lbstate;
    return mask_inbox
  
  def query_mol(self, selection):
    retmol = self.featurizer.boxed_to_mol(selection);
    return retmol

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
    weights = np.nan_to_num(weights, copy=False, nan=0.0, posinf=0.0, neginf=0.0);
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a grid for interpolation
    grid_x, grid_y, grid_z = self.grid

    # Perform 3D Gaussian RBF interpolation
    rbf = Rbf(x, y, z, weights, function='gaussian')
    grid_density = rbf(grid_x, grid_y, grid_z)

    # thegrid = tuple(self.grid);
    # grid_density = griddata(points, weights, thegrid, method='linear', fill_value=0);
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

  def run(self, trajectory):
    """
    update interval
    self.traj.superpose arguments. 
    updatesearchlist arguments. 
    """
    sttime = time.perf_counter();
    self.traj = trajectory.traj; 
    self.feature_array = [];
    # Iterate through the frames in one trajectory
    for index, frame in enumerate(self.traj):
      # Update search list and shift the box
      #######################??????????????????????
      if index % self.featurizer.interval == 0:
      #######################??????????????????????
        self.featurizer.translate()
        self.searchlist = trajectory.updatesearchlist(index, self.featurizer.keymask , 18); 
      feature_i = self.forward(self.traj[index]); 
      self.feature_array.append(feature_i); 
    self.feature_array = np.array(self.feature_array); 
    print(f"Feature {self.__class__.__name__}: {round(time.perf_counter()-sttime, 3)} seconds")
    return self.feature_array; 

class Mass(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Atomic mass as a feature
  """
  def __init__(self, mask="*"):
    super().__init__()
    self.MASK = mask;

  def before_frame(self):
    """
    Since topology does not change during the simulation, we can precompute the atomic mass
    """
    self.ATOMIC_NUMBERS = np.array([int(i.atomic_number) for i in self.traj.top.atoms]);
    self.INDEX_CANDIDATES = self.traj.top.select(self.MASK);

  def featurize(self): 
    """
    1. Get the atomic feature
    2. Update the feature 
    """
    # Get the atoms within the bounding box
    COORD_CANDIDATES = self.active_frame.xyz[self.INDEX_CANDIDATES];
    MASS_CANDIDATES = self.ATOMIC_NUMBERS[self.INDEX_CANDIDATES];
    mask_inbox = self.crop_box(COORD_CANDIDATES);

    # Get the coordinate/required atomic features within the bounding box
    coords  = COORD_CANDIDATES[mask_inbox]
    weights = MASS_CANDIDATES[mask_inbox]

    feature_arr = self.interpolate(coords, weights)
    return feature_arr;

class PartialCharge(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Atomic charge feature for the structure of interest;
  Compute the charge based on the self.featurizer.boxed_pdb;
  """
  def __init__(self, mask="*", value=[]):
    super().__init__()
    self.MASK = mask;          # moiety of interest
    if len(value) > 0:
      self.mode = "manual";
      self.charge_values = [i for i in value];
    else:
      self.mode = "gasteiger";
      self.charge_values = [];

  def before_frame(self):
    self.ATOM_INDICES = self.traj.top.select(self.MASK);
    if (self.mode == "gasteiger"):
      retmol = self.query_mol(self.ATOM_INDICES);
      if retmol is not None:
        self.charge_values = np.array([float(atom.GetProp("_GasteigerCharge")) for atom in retmol.GetAtoms()]).astype(float);
        self.charge_values = self.charge_values[:len(self.ATOM_INDICES)];
      else:
        self.charge_values = np.zeros(len(self.ATOM_INDICES)).astype(float);
    elif self.mode == "manual":
      self.charge_values = np.asarray(self.charge_values).astype(float);

    if len(self.charge_values) != len(self.ATOM_INDICES):
      printit("Warning: The number of atoms in PDB does not match the number of charge values");

  def featurize(self):
    """
    NOTE:
    The self.boxed_pdb is already cropped and atom are reindexed in the PDB block.
    Hence use the self.boxed_indices to get the original atom indices standing for the PDB block
    """
    # Get the atoms within the bounding box
    COORD_CANDIDATES = self.active_frame.xyz[self.ATOM_INDICES];
    mask_inbox = self.crop_box(COORD_CANDIDATES);
    coords  = COORD_CANDIDATES[mask_inbox]
    weights = self.charge_values[mask_inbox]
    feature_charge = self.interpolate(coords, weights)
    return feature_charge;


def AM1BCCChargeFeature(Feature):
  def __init__(self, moi="*", mode="auto", onetime=False):
    super().__init__()
    """AM1-BCC charges computed by antechamber for ligand molecules only"""
    self.cmd_template = "";  # command template for external charge computation programs
    self.computed = False;   # whether self.charge_values is computed or not
    self.mode = "am1bcc";
    self.onetime = onetime;
    self.cmd_template = "am1bcc -i LIGFILE -f ac -o OUTFILE -j 5";

  def featureize(self):
    if (not self.computed) or (not self.onetime):
      # run the am1bcc program
      self.charge_values = np.array(mode);
      cmd_final = self.cmd_template.replace("LIGFILE", self.featurizer.ligfile).replace("OUTFILE", self.featurizer.outfile);
      subprocess.run(cmd_final.split(), check=True);
      # TODO: Try out this function and read the output file into charge values
      self.charge_values = np.loadtxt(self.featurizer.outfile);
      self.computed = True;
    rdmol = Chem.MolFromPDBBlock(self.featurizer.boxed_pdb);
    coords_pdb = np.array([i for i in rdmol.GetConformer().GetPositions()]);
    if len(coords_pdb) != len(self.charge_values):
      printit("Warning: The number of atoms in PDB does not match the number of charge values");
    # Get the atoms within the bounding box
    upperbound = self.center + self.lengths / 2;
    lowerbound = self.center - self.lengths / 2;
    ubstate = np.all(coords_pdb < upperbound, axis=1);
    lbstate = np.all(coords_pdb > lowerbound, axis=1);
    mask_inbox = ubstate * lbstate;

    coords = coords_pdb[mask_inbox]
    weights = charge_array[mask_inbox];
    feature_charge = self.interpolate(coords, weights)
    return feature_charge


class AtomTypeFeature(Feature):
  def __init__(self, aoi="*"):
    super().__init__()
    self.aoi = aoi;
  def before_frame(self):
    if (self.traj.top.select(self.aoi) == 0):
      raise ValueError("No atoms selected for atom type calculation");

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
    self.MASK = mask;

  def before_frame(self):
    """
    Compute the aromaticity of each atom in the moiety of interest
    The atomic index matches between the
    """
    from rdkit import Chem
    self.ATOM_INDICES = self.traj.top.select(self.MASK);
    retmol = self.query_mol(self.ATOM_INDICES);
    if retmol is not None:
      self.AROMATICITY = np.array([atom.GetIsAromatic() for atom in retmol.GetAtoms()]).astype(int);
      self.AROMATICITY = self.AROMATICITY[:len(self.ATOM_INDICES)];
    else:
      self.AROMATICITY = np.zeros(len(self.ATOM_INDICES)).astype(int);

  def featurize(self):
    # Interpolate the feature values to the grid points
    COORD_CANDIDATES = self.active_frame.xyz[self.ATOM_INDICES];
    mask_inbox = self.crop_box(COORD_CANDIDATES);
    coords  = COORD_CANDIDATES[mask_inbox];
    weights = self.AROMATICITY[mask_inbox];
    feature_arr = self.interpolate(coords, weights);
    return feature_arr;

class HeavyAtom(Feature):
  def __init__(self, mask="*", reverse=False):
    """
    Compute the heavy atom feature (or non-heavy atom feature) for the atoms of interest
    Args:
      mask: The mask for the atoms of interest
      reverse: Whether or not to reverse the function to compute the non-heavy atoms (hydrogen atoms)
    """
    super().__init__()
    self.MASK = mask;
    self.REVERSE = bool(reverse);

  def before_frame(self):
    self.ATOM_INDICES = self.traj.top.select(self.MASK);
    self.ATOMIC_NUMBERS = np.array([int(i.atomic_number) for i in self.traj.top.atoms]);
    if self.REVERSE:
      self.ATOM_STATE = np.array(self.ATOMIC_NUMBERS == 1, dtype=bool);
    else:
      self.ATOM_STATE = np.array(self.ATOMIC_NUMBERS > 1, dtype=bool);

  def featurize(self):
    COORD_CANDIDATES = self.active_frame.xyz[self.ATOM_INDICES];
    mask_inbox = self.crop_box(COORD_CANDIDATES);
    coords  = COORD_CANDIDATES[mask_inbox];
    weights = self.ATOM_STATE[self.ATOM_INDICES][mask_inbox];
    feature_arr = self.interpolate(coords, weights);
    return feature_arr;

class Ring(Feature):
  def __init__(self, mask="*"):
    super().__init__()
    self.MASK = mask;

  def before_frame(self):
    self.ATOM_INDICES = self.traj.top.select(self.MASK);
    retmol = self.query_mol(self.ATOM_INDICES);
    if retmol is not None:
      self.RING = np.array([atom.IsInRing() for atom in retmol.GetAtoms()])
      self.RING = self.RING[:len(self.ATOM_INDICES)].astype(int);
    else:
      self.RING = np.zeros(len(self.ATOM_INDICES)).astype(int);
    if len(self.RING) != len(self.ATOM_INDICES):
      printit("Warning: The number of atoms in PDB does not match the number of aromaticity values");

  def featurize(self):
    COORD_CANDIDATES = self.active_frame.xyz[self.ATOM_INDICES];
    mask_inbox = self.crop_box(COORD_CANDIDATES);
    coords  = COORD_CANDIDATES[mask_inbox];
    weights = self.RING[mask_inbox];
    feature_arr = self.interpolate(coords, weights);
    return feature_arr;
class Hybridization(Feature):
  def __init__(self, mask="*"):
    super().__init__()
    self.MASK = mask;
    self.HYBRIDIZATION_DICT = {'SP': 1, 'SP2': 2, 'SP3': 3, "UNSPECIFIED": 0}

  def before_frame(self):
    self.ATOM_INDICES = self.traj.top.select(self.MASK);
    retmol = self.query_mol(self.ATOM_INDICES);
    if retmol is not None:
      self.HYBRIDIZATION = [atom.GetHybridization() for atom in retmol.GetAtoms()]
      self.HYBRIDIZATION = np.array([self.HYBRIDIZATION_DICT[i] if i in self.HYBRIDIZATION_DICT else 0 for i in self.HYBRIDIZATION]).astype(int);
      self.HYBRIDIZATION = self.HYBRIDIZATION[:len(self.ATOM_INDICES)];
    else:
      self.HYBRIDIZATION = np.zeros(len(self.ATOM_INDICES)).astype(int);
    if len(self.HYBRIDIZATION) != len(self.ATOM_INDICES):
      printit("Warning: The number of atoms in PDB does not match the number of HYBRIDIZATION values");
  def featurize(self):
    COORD_CANDIDATES = self.active_frame.xyz[self.ATOM_INDICES];
    mask_inbox = self.crop_box(COORD_CANDIDATES);
    coords  = COORD_CANDIDATES[mask_inbox];
    weights = self.HYBRIDIZATION[mask_inbox];
    feature_arr = self.interpolate(coords, weights);
    return feature_arr;

class HydrogenBond(Feature):
  def __init__(self, mask="*", donor=False, acceptor=False):
    super().__init__()
    self.MASK = mask;
    if not donor and not acceptor:
      raise ValueError("Either donor or acceptor should be True");
    self.FIND_DONOR = bool(donor);
    self.FIND_ACCEPTOR = bool(acceptor);
  def before_frame(self):
    self.ATOM_INDICES = self.traj.top.select(self.MASK);
    retmol = self.query_mol(self.ATOM_INDICES);
    result = [];
    if retmol is not None:
      for atom in retmol.GetAtoms():
        symbol = atom.GetSymbol()
        hydrogen_count = atom.GetTotalNumHs()
        if self.FIND_DONOR:
          is_hbp = symbol in ['N', 'O', 'F'] and hydrogen_count > 0
        elif self.FIND_ACCEPTOR:
          is_hbp = symbol in ['N', 'O', 'F']
        result.append(is_hbp)
    else:
      atoms = np.array([i for i in self.traj.top.atoms]);
      for atom in atoms[self.ATOM_INDICES]:
        symbol = atom.atomic_number;
        partners = atom.bonded_indices();
        atom_partners = atoms[partners];
        atom_numbers = np.array([i.atomic_number for i in atom_partners]);
        hydrogen_count = np.count_nonzero(atom_numbers == 1);
        if self.FIND_DONOR:
          is_hbp = symbol in [7, 8, 9] and hydrogen_count > 0;
        elif self.FIND_ACCEPTOR:
          is_hbp = symbol in [7, 8, 9];
        result.append(is_hbp)

    self.HBP_STATE = np.array(result)[:len(self.ATOM_INDICES)].astype(int);
    # if len(self.HBP_STATE) != len(self.ATOM_INDICES):
    #   printit("Warning: The number of atoms in PDB does not match the number of HBP_STATE values");

  def featurize(self):
    COORD_CANDIDATES = self.active_frame.xyz[self.ATOM_INDICES];
    mask_inbox = self.crop_box(COORD_CANDIDATES);
    coords  = COORD_CANDIDATES[mask_inbox];
    weights = self.HBP_STATE[mask_inbox];
    feature_arr = self.interpolate(coords, weights);
    return feature_arr;



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
    box_feature = np.array([self.center, self.lengths, self.dims]).ravel();
    return box_feature

class FPFHFeature(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Fast Point Feature Histograms
  """
  def __init__(self):
    super().__init__();

  def featurize(self):
    down_sample_nr = CONFIG.get("DOWN_SAMPLE_POINTS", 600)
    fpfh = compute_fpfh_feature(self.featurizer.mesh.sample_points_uniformly(down_sample_nr),
                                KDTreeSearchParamHybrid(radius=1, max_nn=20));
    print("FPFH feature: ", fpfh.data.shape)
    return fpfh.data

class PenaltyFeature(Feature):
  """
  Auxiliary class for featurizer. Needs to be hooked to the featurizer after initialization.
  Deviation from the center of the box
  """
  def __init__(self, mask1, mask2, **kwargs):
    super().__init__()
    self.mask1 = mask1;
    self.mask2 = mask2;
    self.use_mean = kwargs.get("use_mean", False);
    self.ref_frame = kwargs.get("ref_frame", 0);
    self.FAIL_FLAG = False;

  def before_frame(self):
    """
    Get the mean pairwise distance
    """
    if _verbose:
      print("Precomputing the pairwise distance between the closest atom pairs")

    if isinstance(self.mask1, str):
      atom_select = self.traj.top.select(self.mask1);
    elif isinstance(self.mask1, (list, tuple, np.ndarray)):
      atom_select = np.array([int(i) for i in self.mask1]);
    if isinstance(self.mask2, str):
      atom_counterpart = self.traj.top.select(self.mask2);
    elif isinstance(self.mask2, (list, tuple, np.ndarray)):
      atom_counterpart = np.array([int(i) for i in self.mask2]);
    if len(atom_select) == 0:
      self.FAIL_FLAG = True;
      printit("Warning: PenaltyFeature: Mask1 is empty. Marked the FAIL_FLAG. Please check the atom selection");
      return
    elif len(atom_counterpart) == 0:
      self.FAIL_FLAG = True;
      printit("Warning: PenaltyFeature: Mask2 is empty. Marked the FAIL_FLAG. Please check the atom selection");
      return

    traj_copy = self.traj.copy_traj();
    traj_copy.top.set_reference(traj_copy[self.ref_frame]);
    self.pdist, self.pdistinfo = utils.PairwiseDistance(traj_copy, atom_select, atom_counterpart, use_mean=self.use_mean, ref_frame=self.ref_frame);
    self.pdist_mean = self.pdist.mean(axis=1)
    if self.pdist.mean() > 8:
      printit("Warning: the mean distance between the atom of interest and its counterpart is larger than 8 Angstrom");
      printit("Please check the atom selection");
    elif np.percentile(self.pdist, 85) > 12:
      printit("Warning: the 85th percentile of the distance between the atom of interest and its counterpart is larger than 12 Angstrom");
      printit("Please check the atom selection");

    info_lengths = [len(self.pdistinfo[key]) for key in self.pdistinfo];
    if len(set(info_lengths)) != 1:
      printit("Warning: The length of the pdistinfo is not consistent", self.pdistinfo);


  def featurize(self):
    """
    Get the deviation from the center of the box
    """
    if self.FAIL_FLAG == True:
      return 0;
    coord_diff = self.active_frame.xyz[self.pdistinfo["indices_group1"]] - self.active_frame.xyz[self.pdistinfo["indices_group2"]]
    dists = np.linalg.norm(coord_diff, axis=1)
    cosine_sim = utils.cosine_similarity(dists, self.pdist_mean);
    return cosine_sim;

class MSCVFeature(Feature):
  def __init__(self, mask1, mask2, **kwargs):
    super().__init__();
    self.mask1 = mask1;
    self.mask2 = mask2;
    self.use_mean = kwargs.get("use_mean", False);
    self.ref_frame = kwargs.get("ref_frame", 0);
    self.WINDOW_SIZE = CONFIG.get("WINDOW_SIZE", 10);
  def before_frame(self):
    """
    Get the mean pairwise distance
    """
    if _verbose:
      print("Precomputing the pairwise distance between the closest atom pairs")
    self.traj_copy = self.traj.copy();
    self.traj_copy.top.set_reference(self.traj_copy[self.ref_frame]);
    self.pd_arr, self.pd_info = utils.PairwiseDistance(self.traj_copy, self.mask1, self.mask2, use_mean=self.use_mean, ref_frame=self.ref_frame);
    self.mean_pd = np.mean(self.pd_arr, axis=1);

  def featurize(self):
    """
    Get the mean square coefficient of variation of the segment
    """
    framenr = self.traj.n_frames;
    if framenr < self.WINDOW_SIZE:
      # If the window size is larger than the number of frames, then use the whole trajectory
      frames = np.arange(0, framenr);
    elif (self.active_frame_index + self.WINDOW_SIZE > framenr):
      # If the last frame is not enough to fill the window, then use the last window
      frames = np.arange(framenr - self.WINDOW_SIZE, framenr);
    else:
      frames = np.arange(self.active_frame_index, self.active_frame_index + self.WINDOW_SIZE);

    # Store the pairwise distance for each frame
    pdists = np.zeros()
    for fidx in frames:
      dists = np.linalg.norm(
        self.active_frame.xyz[self.pd_info["indices_group1"]] - self.active_frame.xyz[self.pd_info["indices_group2"]],
        axis=1);
      pdists[:, fidx] = dists;
    mscv = utils.MSCV(pdists);
    return mscv;

class EntropyResidueID(Feature):
  def __init__(self, mask="*"):
    super().__init__();
    self.WINDOW_SIZE = CONFIG.get("WINDOW_SIZE", 10);
    self.MASK = mask;

  def before_focus(self):
    """
    Before processing focus points, get the correct slice of frames
    """
    # Determine the slice of frames from a trajectory
    framenr = self.traj.n_frames;
    if framenr < self.WINDOW_SIZE:
      print("The slice is larger than the total frames, using the whole trajectory");
      frames = np.arange(0, framenr);
    elif (self.active_frame_index + self.WINDOW_SIZE > framenr):
      print("Unable to fill the last frame and unable not enough to fill the window, then use the last window");
      frames = np.arange(framenr - self.WINDOW_SIZE, framenr);
    else:
      frames = np.arange(self.active_frame_index, self.active_frame_index + self.WINDOW_SIZE);
    self.ENTROPY_CUTOFF = np.linalg.norm(self.lengths / self.dims);  # Automatically set the cutoff for each grid
    self.ATOM_INDICES = self.top.select(self.MASK);
    self._COORD = self.traj.xyz[frames, self.ATOM_INDICES].reshape((-1, 3));
    self.RESID_ENSEMBLE = np.array([i.resid for idx,i in enumerate(self.top.atoms) if idx in self.ATOM_INDICES] * len(frames));
    self.FRAME_NUMBER = len(frames);
    if len(self._COORD) != len(self.RESID_ENSEMBLE):
      raise Exception("Warning: The number of coordinates and the number of residue indices are not equal")


  # @profile
  def featurize(self):
    """
    Get the information entropy(Chaoticity in general) of the box
    """
    entropy_arr = np.zeros(tuple(self.dims));
    distances = cdist(self.points3d, self._COORD)
    distances = distances.reshape(self.dims[0], self.dims[1], self.dims[2], self.FRAME_NUMBER, len(self.ATOM_INDICES));
    for I in range(self.dims[0]):
      for J in range(self.dims[1]):
        for K in range(self.dims[2]):
          _distances_status = distances[I, J, K, :] < self.ENTROPY_CUTOFF;
          if True in _distances_status:
            _distances_status = _distances_status.ravel();
            entropy_arr[I,J,K] = entropy_jit(self.RESID_ENSEMBLE[_distances_status])
          else:   # All False
            continue
    return entropy_arr

class EntropyAtomID(Feature):
  def __init__(self, mask="*"):
    super().__init__();
    self.WINDOW_SIZE = CONFIG.get("WINDOW_SIZE", 10);
    self.MASK = mask;

  def before_focus(self):
    # Determine the slice of frames from a trajectory
    framenr = self.traj.n_frames;
    if framenr < self.WINDOW_SIZE:
      print("The slice is larger than the total frames, using the whole trajectory");
      frames = np.arange(0, framenr);
    elif (self.active_frame_index + self.WINDOW_SIZE > framenr):
      print("Unable to fill the last frame and unable not enough to fill the window, then use the last window");
      frames = np.arange(framenr - self.WINDOW_SIZE, framenr);
    else:
      frames = np.arange(self.active_frame_index, self.active_frame_index + self.WINDOW_SIZE);
    self.ENTROPY_CUTOFF = np.linalg.norm(self.lengths / self.dims);  # Automatically set the cutoff for each grid
    self.ATOM_INDICES = self.top.select(self.MASK);
    self._COORD = self.traj.xyz[frames, self.ATOM_INDICES].reshape((-1, 3));
    self.INDEX_ENSEMBLE = np.array(
      [i.index for idx, i in enumerate(self.top.atoms) if idx in self.ATOM_INDICES] * len(frames));
    self.FRAME_NUMBER = len(frames);
    if len(self._COORD) != len(self.INDEX_ENSEMBLE):
      raise Exception("Warning: The number of coordinates and the number of residue indices are not equal")

  def featurize(self):
    """
    Get the information entropy(Chaoticity in general) of the box
    """
    entropy_arr = np.zeros(tuple(self.dims));
    distances = cdist(self.points3d, self._COORD)
    distances = distances.reshape(self.dims[0], self.dims[1], self.dims[2], self.FRAME_NUMBER, len(self.ATOM_INDICES));
    for I in range(self.dims[0]):
      for J in range(self.dims[1]):
        for K in range(self.dims[2]):
          _distances_status = distances[I, J, K, :] < self.ENTROPY_CUTOFF;
          if True in _distances_status:
            _distances_status = _distances_status.ravel();
            entropy_arr[I, J, K] = entropy_jit(self.INDEX_ENSEMBLE[_distances_status])
          else:  # All False"""
            continue
    return entropy_arr


class RFFeature1D(Feature):
  def __init__(self, moiety_of_interest, cutoff=12):
    super().__init__();
    self.moi = moiety_of_interest;
    # For the 4*9 (36) features
    # Rows (protein)   : C, N, O, S
    # Columns (ligand) : C, N, O, F, P, S, Cl, Br, I
    self.cutoff = cutoff;
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
    atoms = np.asarray([i.atomic_number for i in self.top.atoms]);
    moi_indices = self.top.select(self.moi); # The indices of the moiety of interest
    if len(moi_indices) == 0:
      printit("Warning: The moiety of interest is not found in the topology");
      # Return a zero array if the moiety is not found
      return np.zeros(36);
    if _verbose:
      printit(f"The moiety of interest contains {len(moi_indices)} atoms");
    # build a kd-tree for interaction query
    other_indices = np.asarray([i for i in np.arange(len(self.active_frame.xyz)) if i not in moi_indices]);
    kd_tree = KDTree(self.active_frame.xyz[other_indices]);
    rf_arr = np.zeros((4,9));
    for i, idx in enumerate(moi_indices):
      atom_number = atoms[idx];
      atom_coord  = self.active_frame.xyz[idx];
      soft_idxs = kd_tree.query_ball_point(atom_coord, self.cutoff);
      hard_idxs = other_indices[soft_idxs];
      for idx_prot in hard_idxs:
        atom_number_prot = atoms[idx_prot];
        if atom_number in self.lig_atom_idx and atom_number_prot in self.pro_atom_idx:
          rf_arr[self.pro_atom_idx[atom_number_prot], self.lig_atom_idx[atom_number]] += 1;
    return rf_arr.reshape(-1);


class TopologySource(Feature):
  def __init__(self):
    super().__init__();
  def featurize(self):
    top_source = str(self.traj.top_filename);
    return [top_source];



