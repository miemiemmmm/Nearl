import tempfile, copy

import numpy as np
import pytraj as pt
from rdkit import Chem

from open3d.geometry import TriangleMesh

from nearl import utils
from nearl.io import hdf5

from . import fingerprint
from .. import printit, _verbose, _debug

import multiprocessing as mp

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
  def __init__(self, parms, segment_limit=6, viewpoint_bins=30):
    """
    Initialize the featurizer with the given parameters
    """
    # Check the essential parameters for the featurizer
    self.parms = parms
    parms_to_check = ["DIMENSIONS", "LENGTHS"]
    for parm in parms_to_check:
      if parm not in parms:
        printit(f"Warning: Not found required parameter: {parm}. Please define the keyword <{parm}> in your parameter set. ")
        return
    # Basic parameters for the featurizer to communicate with cuda code
    self.__dims = None
    self.__lengths = None
    self.dims = parms.get("DIMENSIONS", 32)   # Set the dimensions of the 3D grid
    self.lengths = parms.get("LENGTHS", 16)   # Set the lengths of the 3D grid
    self.__spacing = np.mean(self.__lengths / self.__dims)
    self.__boxcenter = np.array(self.dims/2, dtype=float)
    self.__sigma = 1.0 
    self.__cutoff = 4.0
    self.__interval = 1 

    # Derivative parameters
    self._traj = None
    self.FRAMENUMBER = 0
    self.FRAMESLICENUMBER = 0
    self.FRAMESLICES = []

    # Component I: Feature space
    self.FEATURESPACE = []
    self.FEATURENUMBER = 0

    # Component II: Focal point space 
    self.FOCALPOINTS = []
    self.FOCALNUMBER = 0

    # Component III: Trajectory space
    self.TRAJLOADER = None
    self.TRAJECTORYNUMBER = 0
    

    # Initialize the attributes for featurization
    # TODO
    self.__status_flag = []
    self.contents = {}

    # TODO: Identity vector generation related variables
    self.mesh = TriangleMesh()
    self.BOX_MOLS = {}
    self.SEGMENTNR = int(segment_limit)
    # Hard coded structural features (12)
    self.VPBINS = 12 + int(viewpoint_bins)
    
    if _verbose:
      printit("Featurizer is initialized successfully")
      print("With Dimensions: ", self.__dims)
      print("With Lengths: ", self.__lengths)

  def __str__(self):
    finalstr = f"Feature Number: {self.FEATURENUMBER}; \n"
    for feat in self.FEATURESPACE:
      finalstr += f"Feature: {feat.__str__()}\n"
    return finalstr

  # The most important attributes to determine the size of the 3D grid
  @property
  def dims(self):
    return np.array(self.__dims)
  @dims.setter
  def dims(self, newdims):
    if isinstance(newdims, (int, float, np.float32, np.float64, np.int32, np.int64)): 
      self.__dims = np.array([newdims, newdims, newdims], dtype=int)
    elif isinstance(newdims, (list, tuple, np.ndarray)):
      assert len(newdims) == 3, "length should be 3"
      self.__dims = np.array(newdims, dtype=int)
    else:
      raise Exception("Unexpected data type, either be a number or a list of 3 integers")
    if self.__lengths is not None:
      self.__spacing = np.mean(self.__lengths / self.__dims)
    self.__boxcenter = np.array(self.dims/2, dtype=float)
    
  # The most important attributes to determine the size of the 3D grid
  @property
  def lengths(self):
    return self.__lengths
  @lengths.setter
  def lengths(self, new_length):
    if isinstance(new_length, (int, float, np.float32, np.float64, np.int32, np.int64)):
      self.__lengths = np.array([new_length] * 3, dtype=float)
    elif isinstance(new_length, (list, tuple, np.ndarray)):
      assert len(new_length) == 3, "length should be 3"
      self.__lengths = np.array(new_length, dtype=float)
    else:
      raise Exception("Unexpected data type, either be a number or a list of 3 floats")
    if self.__dims is not None:
      self.__spacing = np.mean(self.__lengths / self.__dims)
    
  # READ-ONLY because it is determined by the DIMENSIONS and LENGTHS
  @property
  def spacing(self):
    return self.__spacing
  @property
  def boxcenter(self):
    return self.__boxcenter

  # Attributes important for computing of features (for CUDA part)
  @property
  def cutoff(self):
    return self.__cutoff
  @cutoff.setter
  def cutoff(self, new_cutoff):
    self.__cutoff = float(new_cutoff)

  @property
  def interval(self):
    return self.__interval
  @interval.setter
  def interval(self, new_interval):
    self.__interval = int(new_interval)

  @property
  def sigma(self):
    return self.__sigma
  @sigma.setter
  def sigma(self, newsigma):
    self.__sigma = float(newsigma)
  
  @property
  def traj(self):
    return self._traj
  @traj.setter
  def traj(self, the_traj):
    """
      Set the trajectory and its related parameters
    """
    self._traj = the_traj
    self.FRAMENUMBER = the_traj.n_frames
    self.SLICENUMBER = self.FRAMENUMBER // self.interval
    if self.FRAMENUMBER % self.interval != 0:
      printit("Warning: the number of frames is not divisible by the interval. The last few frames will be ignored.")
    printit(f"Registered {self.SLICENUMBER} slices for the trajectory ({self.FRAMENUMBER}) with {self.interval} interval.")
    frame_array = np.array([0] + np.cumsum([self.__interval] * self.SLICENUMBER).tolist())
    self.FRAMESLICES = [np.s_[frame_array[i]:frame_array[i+1]] for i in range(self.SLICENUMBER)]

  @property
  def top(self):
    return self.traj.top


  # TODO: check how to use this property
  @property
  def status_flag(self):
    return self.__status_flag
  @status_flag.setter
  def status_flag(self, newstatus):
    self.__status_flag.append(bool(newstatus))
  

  def reset_status(self):
    self.__status_flag = []


  def update_box_length(self, length=None, scale_factor=1.0):
    if length is not None:
      self.__lengths = float(length)
    else:
      self.__lengths *= scale_factor
    self.update_box()

  def register_feature(self, feature):
    """
    Register a feature to the featurizer
    Args:
      feature: A feature object
    """
    feature.hook(self)  # Hook the featurizer to the feature
    self.FEATURESPACE.append(feature)
    self.FEATURENUMBER = len(self.FEATURESPACE)
  
  def register_features(self, features):
    """
    Register a list of features to the featurizer
    Args:
      features: A list of feature objects
    """
    for feature in features:
      self.register_feature(feature)

  def register_trajloader(self, trajloader):
    self.TRAJLOADER = trajloader
    self.TRAJECTORYNUMBER = len(trajloader)
    print(f"Registered {self.TRAJECTORYNUMBER} trajectories")


  def register_focus(self, focus, format):
    # Define the focus points to process
    # Formats includes: 
    # "cog": provide a masked selection of atoms
    # "absolute": provide a list of 3D coordinates
    # "index": provide a list of atom indexes (int)
    # NOTE: before running the featurizer, the focus should be registered and it is specific to the trajectory
    # for each interval, there is one focus point
    if format == "cog":
      self.FOCALPOINTS_PROTOTYPE = focus
      self.FOCALPOINTS_TYPE = "cog"
      self.FOCALPOINTS = None
    elif format == "absolute":
      assert len(focus.shape) == 2, "The focus should be a 2D array"
      assert focus.shape[1] == 3, "The focus should be a 2D array with 3 columns"
      self.FOCALPOINTS_PROTOTYPE = focus
      self.FOCALPOINTS = focus
      self.FOCALPOINTS_TYPE = "absolute"
    elif format == "index":
      self.FOCALPOINTS_PROTOTYPE = focus
      self.FOCALPOINTS_TYPE = "index"
      self.FOCALPOINTS = None
    else: 
      raise ValueError(f"Unexpected focus format: {format}")

  def parse_focus(self): 
    # Parse the focus points to the correct format
    self.FOCALPOINTS = np.full((self.SLICENUMBER, len(self.FOCALPOINTS_PROTOTYPE), 3), 99999, dtype=np.float32)
    self.FOCALNUMBER = len(self.FOCALPOINTS_PROTOTYPE)
    if self.FOCALPOINTS_TYPE == "cog":
      # Get the center of geometry for the frames with self.interval
      for midx, mask in enumerate(self.FOCALPOINTS_PROTOTYPE): 
        selection = self.traj.top.select(mask)
        for fidx in range(self.SLICENUMBER):
          frame = self.traj.xyz[fidx*self.interval]
          self.FOCALPOINTS[fidx, midx] = np.mean(frame[selection], axis=0)

    elif self.FOCALPOINTS_TYPE == "index":
      for midx, mask in enumerate(self.FOCALPOINTS_PROTOTYPE): 
        for idx, frame in enumerate(self.traj.xyz[::self.interval]):
          self.FOCALPOINTS[idx, midx] = np.mean(frame[mask], axis=0)

    elif self.FOCALPOINTS_TYPE == "absolute":
      for focusidx, focus in enumerate(self.FOCALPOINTS_PROTOTYPE): 
        assert len(focus) == 3, "The focus should be a 3D coordinate"
        for idx, frame in enumerate(self.traj.xyz[::self.interval]):
          self.FOCALPOINTS[idx, focusidx] = focus
      
    else:
      raise ValueError(f"Unexpected focus format: {self.FOCALPOINTS_TYPE}")


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

  def main_loop(self, process_nr=20): 
    pool = mp.Pool(process_nr)
    for tid in range(self.TRAJECTORYNUMBER):
      # Setup the trajectory and its related parameters such as slicing of the trajectory
      self.traj = self.TRAJLOADER[tid]

      # Cache the weights for each atoms in the trajectory
      for feat in self.FEATURESPACE:
        feat.cache(self.traj)

      printit(f"{self.__class__.__name__}: Start processing the trajectory {tid} with {self.SLICENUMBER} frames")

      tasks = []
      feature_map = []
      # Pool the actions for each trajectory
      for bid in range(self.SLICENUMBER): 
        frames = self.traj.xyz[self.FRAMESLICES[bid]]

        # Update the focus points for each bin  
        self.parse_focus()    # generate the self.FOCALPOINTS as (self.FOCALNUMBER, 3) array
        print("Focal points shape: ", self.FOCALPOINTS.shape)

        # After determineing each focus point, run the featurizer for each focus point
        for pid in range(self.FOCALNUMBER):
          printit(f"Processing the focal point {pid} at the bin {bid}")
          focal_point = self.FOCALPOINTS[bid, pid]
          assert len(focal_point) == 3, "The focal point should be a 3D coordinate"   # TODO temporary check for debugging

          # Crop the trajectory and send the coordinates/trajectory to the featurizer
          for fidx in range(self.FEATURENUMBER):
            # Explicitly transfer the topology and frames to get the queried coordinates for the featurizer
            queried = self.FEATURESPACE[fidx].query(self.top, frames, focal_point)
            tasks.append([self.FEATURESPACE[fidx].run, queried])
            feature_map.append((tid, bid, pid, fidx))
      printit(f"Tasks are ready for the trajectory {tid} with {len(tasks)} tasks")
      
      # Run the actions in the process pool
      _tasks = [pool.apply_async(wrapper_runner, task) for task in tasks]
      results = [task.get() for task in _tasks]
      # print(results)
      print("Nan in return", [(np.nan in i) for i in results])
      print([i.shape for i in results])


      print("Dumping the results to the feature space...")
      # TODO: dump to file for each feature
      # for feat_meta, result in zip(feature_map, results):
      #   tid, bid, pid, fidx = feat_meta
      #   self.FEATURESPACE[fidx].dump(result)
      print(f"Finished the trajectory {tid} with {len(tasks)} tasks")
      break
    pool.close()
    pool.join()
    print("All trajectories and tasks are finished")

def wrapper_runner(func, args):
  """
    Take the feature.run methods and its input arguments for multiprocessing
  """
  return func(*args)

def selection_to_mol(traj, frameidx, selection):
  if isinstance(selection, str):
    atom_sel = traj.top.select(selection)
  elif isinstance(selection, (list, tuple, np.ndarray)):
    atom_sel = np.asarray(selection)
  else:
    raise ValueError(f"{__file__}: Unexpected selection type")

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
