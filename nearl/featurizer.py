import time, json, logging 

import numpy as np

from . import utils, constants
from . import printit, config

__all__ = [
  "Featurizer",
]

def wrapper_runner(func, args):
  """
  Take the feature.run methods and its input arguments for multiprocessing

  Parameters
  ----------
  func : function
    The function to be run
  args : list
    The arguments for the function
  
  """
  return func(*args)  


class Featurizer:
  """
  Featurizer aims to automate the process of featurization of multiple Features for a batch of structures or trajectories

  Parameters
  ----------
  parms : dict
    A dictionary of parameters for the featurizer
    
  Attributes
  ----------
  traj : :class:`nearl.Trajectory <nearl.io.traj.Trajectory>` or pytraj.Trajectory
    The trajectory to be processed
  top : pytraj.Topology
    The topology of the trajectory
  dims : np.ndarray
    The dimensions of the 3D grid
  lengths : np.ndarray
    The lengths of the 3D grid
  spacing : float
    The spacing of the 3D grid
  time_window : int, default = 1
    The time window for the trajectory (default is 1), Simple integer.
  frame_slice : slice
    The frame-slice being processed in the trajectory 

  
  FRAMENUMBER : int
    The number of frames in the trajectory to be processed
  FRAMESLICENUMBER : int
    The number of slices of frames in the trajectory
  FRAMESLICES : list
    A list of slices of frames in the trajectory to be processed

  FEATURESPACE : list
    A list of features to be processed
  FEATURENUMBER : int
    The number of features to be processed

  FOCALPOINTS_PROTOTYPE
    The prototype of the focal points
  FOCALPOINTS : np.ndarray
    The focal points to be processed
  FOCALNUMBER : int
    The number of focal points for each frame slice

  Notes
  -----
  Required parameters:

  - **dimensions**: the dimensions of the 3D grid
  - lengths: the lengths of the 3D grid (optional)
  - spacing: the spacing of the 3D grid (optional)
  - time_window: the time window for the trajectory (default is 1), Simple integer.
  
  The following are optional parameters for features. 
  If the initialization of the feature did not explicitly define the following parameters, the following parameters will be inherited from the featurizer: 

  - outfile: The output file to dump the parameters and the results
  - sigma: The smoothness of the Gaussian-based feature distribution
  - cutoff: The cutoff distance for the grid-based feature calculation

  """
  def __init__(self, parms={}, **kwargs):
    """
    Initialize the featurizer with the given parameters
    """
    # Check the essential parameters for the featurizer
    # assert "dimensions" in parms, "Please define the 'dimensions' in the parameter set"
    # assert ("lengths" in parms) or ("spacing" in parms), "Please define the 'lengths' or 'spacing' in the parameter set"

    # Basic parameters for the featurizer to communicate with cuda code
    self.__dims = parms.get("dimensions", None)   # Set the dimensions of the 3D grid
    self.__lengths = None
    self.__spacing = None
    if "lengths" in parms:
      self.__lengths = parms.get("lengths", 16)   # Set the lengths of the 3D grid
      self.__spacing = np.mean(self.lengths / self.dims)
    elif "spacing" in parms:
      self.__spacing = parms.get("spacing", 1.0)
      self.__lengths = self.dims * self.__spacing  # Directly assignment avoid the re-calculation of the spacing

    self.time_window = int(parms.get("time_window", 1))   # The time window for the trajectory (default is 1), Simple integer.

    # Get common feature parameters to hook the features
    self.FEATURE_PARMS = {}
    for key in constants.COMMON_FEATURE_PARMS:
      if key in parms.keys():
        self.FEATURE_PARMS[key] = parms[key]
      elif key in kwargs.keys():
        self.FEATURE_PARMS[key] = kwargs[key]
      else: 
        self.FEATURE_PARMS[key] = None

    self.OTHER_PARMS = {}
    for key in parms.keys():
      if key not in constants.COMMON_FEATURE_PARMS:
        self.OTHER_PARMS[key] = parms[key]
      else:
        continue
    for key in kwargs.keys():
      if key not in constants.COMMON_FEATURE_PARMS:
        self.OTHER_PARMS[key] = kwargs[key]
      else:
        continue

    # Derivative parameters from trajectory 
    self._traj = None
    self.frame_slice = None 
    self.FRAMENUMBER = 0
    self.FRAMESLICENUMBER = 0
    self.FRAMESLICES = []

    # Component I: Feature space
    self.FEATURESPACE = []
    self.FEATURENUMBER = 0

    # Component II: Focal point space 
    self.FOCALPOINTS = []
    self.FOCALPOINTS_PROTOTYPE = None
    self.FOCALNUMBER = 0

    # Component III: Trajectory space
    self.TRAJLOADER = None
    self.TRAJECTORYNUMBER = 0
    
    self.classname = self.__class__.__name__  
    if config.verbose():
      printit(f"{self.classname}: Featurizer is initialized successfully with dimensions: {self.dims} and lengths: {self.lengths}")

    if "outfile" in parms.keys():
      # Dump the parm dict to that hdf file 
      printit(f"{self.classname}: Dumping the parameters to {parms['outfile']} : {self.parms}")
      utils.dump_dict(parms["outfile"], "featurizer_parms", self.parms)

  def __str__(self):
    finalstr = f"Feature Number: {self.FEATURENUMBER}; \n"
    for feat in self.FEATURESPACE:
      finalstr += f"Feature: {feat.__str__()}\n"
    return finalstr

  @property
  def parms(self):
    """
    Return the parameters of the featurizer
    """
    return {k:v if v is not None else 0 for k,v in {**self.FEATURE_PARMS, **self.OTHER_PARMS}.items()} 

  # The most important attributes to determine the size of the 3D grid
  @property
  def dims(self):
    """
    The 3 dimensions of the 3D grid
    """
    return np.array(self.__dims) if self.__dims is not None else None
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
      self.__spacing = np.mean(self.lengths / self.dims)
    
  # The most important attributes to determine the size of the 3D grid
  @property
  def lengths(self):
    """
    The lengths of the 3D grid in Angstrom 
    """
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
      self.__spacing = np.mean(self.lengths / self.dims)

  @property
  def spacing(self):
    """
    The spacing (also the resolution) between grid points of the 3D grid
    """
    return self.__spacing  
  
  @property
  def traj(self):
    """
    The trajectory being processed. 

    """
    return self._traj
  @traj.setter
  def traj(self, the_traj):
    self._traj = the_traj
    self.FRAMENUMBER = the_traj.n_frames
    self.SLICENUMBER = self.FRAMENUMBER // self.time_window
    if self.SLICENUMBER == 0: 
      logging.warning(f"{self.classname}: No frame slice is available. The trajectory have {self.FRAMENUMBER} frames and the time window is {self.time_window}.")
    if self.FRAMENUMBER % self.time_window != 0 and self.FRAMENUMBER != 1:
      logging.warning(f"{self.classname}: the number of frames ({self.FRAMENUMBER}) is not divisible by the time window ({self.time_window}). The last few frames will be ignored.")
    logging.info(f"{self.classname}: Registered {self.SLICENUMBER} slices of frames with {self.time_window} as the time window (frames-per-slice).")
    logging.debug(f"Having {self.SLICENUMBER} frame slices in the trajectory ") 
    frame_array = np.array([0] + np.cumsum([self.time_window] * self.SLICENUMBER).tolist())
    self.FRAMESLICES = [np.s_[frame_array[i]:frame_array[i+1]] for i in range(self.SLICENUMBER)]

  @property
  def top(self):
    """
    The topology of the trajectory being processed. 
    """
    return self.traj.top

  def register_feature(self, feature):
    """
    Register a :class:`nearl.features.Feature` to the featurizer
    
    Parameters
    ----------
    feature : :class:`nearl.features.Feature`
    
    """
    feature.hook(self)  # Hook the featurizer to the feature
    self.FEATURESPACE.append(feature)
    self.FEATURENUMBER = len(self.FEATURESPACE)
    output_keys = [i.outkey for i in self.FEATURESPACE]
    if len(set(output_keys)) != len(output_keys): 
      print(np.unique(output_keys, return_counts=True))
      raise ValueError("The output keys for the features should be unique.")
  
  def register_features(self, features):
    """
    Register multiple :class:`nearl.features.Feature` in a list or dictionary to the featurizer 

    Parameters
    ----------
    features : list-like or dict-like 
    
    """
    if isinstance(features, (list, tuple)):
      for feature in features:
        self.register_feature(feature)
    elif isinstance(features, dict):
      for _, feature in features.items():
        if config.verbose() or config.debug():
          printit(f"{self.classname}: Registering the feature named: {_} from {feature.classname} class")
        self.register_feature(feature)

  def register_trajloader(self, trajloader):
    """
    Register a trajectory loader to the featurizer for further processing 

    Parameters
    ----------
    trajloader : :class:`nearl.io.trajloader.TrajectoryLoader`
    
    """
    self.TRAJLOADER = trajloader
    self.TRAJECTORYNUMBER = len(trajloader)
    printit(f"{self.classname}: Registered {self.TRAJECTORYNUMBER} trajectories")


  def register_focus(self, focus, format):
    """
    Register a set of focal points to the featurizer for further processing

    Parameters
    ----------
    focus : list_like
      The focal points to process
    format : str
      The format of the focal points

    Notes
    -----
    The following 4 formats of focuses are supported:

    - **mask**: provide a selection of atoms (`Amber's selection convention <https://amberhub.chpc.utah.edu/atom-mask-selection-syntax/>`_)
    - **index**: provide a list of atom indices (int)
    - **absolute**: provide a list of 3D coordina tes
    - **json**: provide a json file containing the indexes of the atoms for each trajectory (the key for each trajectory should match the ``feat.identity`` attribute)

    """
    if format == "mask":
      self.FOCALPOINTS_PROTOTYPE = focus
      self.FOCALPOINTS_TYPE = "mask"
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

    elif format == "json": 
      self.FOCALPOINTS_PROTOTYPE = focus
      self.FOCALPOINTS_TYPE = "json"
      self.FOCALPOINTS = None

    else: 
      raise ValueError(f"Unexpected focus format: {format}. Please choose from 'mask', 'absolute', 'index', 'function'")

  def parse_focus(self): 
    """
    After registering the active trajectory, parse the focal points for each frame-slice in the ``run`` method. 
    The resulting shape will be a 3D array with the shape ``(slice_number, focus_number, 3)``  

    """
    # Parse the focus points to the correct format
    self.FOCALPOINTS = np.full((self.SLICENUMBER, len(self.FOCALPOINTS_PROTOTYPE), 3), 99999, dtype=np.float32)
    logging.debug(f"Shape of the focal points prototype: {self.FOCALPOINTS.shape}")
    self.FOCALNUMBER = len(self.FOCALPOINTS_PROTOTYPE)
    if self.FOCALPOINTS_TYPE == "mask":
      # Get the center of geometry for the frames with self.interval
      for midx, mask in enumerate(self.FOCALPOINTS_PROTOTYPE): 
        selection = self.traj.top.select(mask)
        if len(selection) == 0:
          printit(f"{self.classname} Warning: The trajectory {self.traj.identity} does not have any atoms in the selection {mask}")
          return False
        for fidx in range(self.SLICENUMBER):
          frame = self.traj.xyz[fidx*self.time_window]
          self.FOCALPOINTS[fidx, midx] = np.mean(frame[selection], axis=0)
      return 1
    elif self.FOCALPOINTS_TYPE == "index":
      for midx, mask in enumerate(self.FOCALPOINTS_PROTOTYPE): 
        for idx, frame in enumerate(self.traj.xyz[::self.time_window]):
          if idx >= self.SLICENUMBER: 
            break
          self.FOCALPOINTS[idx, midx] = np.mean(frame[mask], axis=0)
      return 1

    elif self.FOCALPOINTS_TYPE == "absolute":
      for focusidx, focus in enumerate(self.FOCALPOINTS_PROTOTYPE): 
        assert len(focus) == 3, "The focus should be a 3D coordinate"
        logging.debug(f"Shape of the focus: {focus.shape}") 
        for idx, frame in enumerate(self.traj.xyz[::self.time_window]):
          logging.warning(f"Processing the frame {idx} with the focus {focus}") 
          if idx >= self.SLICENUMBER: 
            break
          self.FOCALPOINTS[idx, focusidx] = focus
      return 1
      
    elif self.FOCALPOINTS_TYPE == "json":
      with open(self.FOCALPOINTS_PROTOTYPE, "r") as f:
        focus = json.load(f)
        indices = focus[utils.get_pdbcode(self.traj.identity)]
      indices = np.array(indices, dtype=int)
      for idx, frame in enumerate(self.traj.xyz[::self.time_window]):
        if idx >= self.SLICENUMBER:
          break
        focus = np.mean(frame[indices], axis=0)
        self.FOCALPOINTS[0, idx] = focus
      return 1

    else:
      raise ValueError(f"Unexpected focus format: {self.FOCALPOINTS_TYPE}")

  def run(self): 
    """
    Run the featurization for each iteration over trajectory, frame-slice, focal-point, and feature. 
    """
    for tid in range(self.TRAJECTORYNUMBER):
      # Setup the trajectory and its related parameters such as slicing of the trajectory
      self.traj = self.TRAJLOADER[tid]
      msg = f"Processing the trajectory {tid+1} ({self.traj.identity}) with {self.SLICENUMBER} frame slices"
      printit(f"{self.classname}: {msg:=^80}")
      st = time.perf_counter()

      if self.FOCALPOINTS_PROTOTYPE is not None:
        # NOTE: Re-parse the focal points for each trajectory
        # Expected output shape is (self.SLICENUMBER, self.FOCALNUMBER, 3) array 
        focus_state = self.parse_focus()
        if focus_state == 0:
          printit(f"{self.classname} Warning: Skipping the trajectory {self.traj.identity}(index {tid+1}) because focal points parsing is failed. ")
          continue 
        if config.verbose() or config.debug():
          printit(f"{self.classname}: Parsing of focal points on trajectory ({tid+1}/{self.traj.identity}) yeield the shape: {self.FOCALPOINTS.shape}. ")

      # Cache the weights for each atoms in the trajectory (run once for each trajectory)
      for feat in self.FEATURESPACE:
        if config.verbose(): 
          printit(f"{self.classname}: Caching the weights of feature {feat.classname} for the trajectory {tid+1}")
        feat.cache(self.traj)

      tasks = []
      feature_map = []
      # Pool the actions for each trajectory
      for bid in range(self.SLICENUMBER): 
        self.frame_slice = self.FRAMESLICES[bid]
        frames = self.traj.xyz[self.FRAMESLICES[bid]]
        if self.FOCALNUMBER > 0:
          # After determineing each focus point, run the featurizer for each focus point
          for pid in range(self.FOCALNUMBER):
            focal_point = self.FOCALPOINTS[bid, pid]
            # Crop the trajectory and send the coordinates/trajectory to the featurizer
            for fidx in range(self.FEATURENUMBER):
              # NOTE: Isolate the effect on the calculation of the next feature 
              queried = self.FEATURESPACE[fidx].query(self.top, frames, focal_point)
              tasks.append([self.FEATURESPACE[fidx].run, queried])
              feature_map.append((tid, bid, fidx))
        else:
          # Without registeration of focal points: focal-point independent features such as label-generation
          for fidx in range(self.FEATURENUMBER):
            # Explicitly transfer the topology and frames to get the queried coordinates for the featurizer
            queried = self.FEATURESPACE[fidx].query(self.top, frames, [0, 0, 0])
            tasks.append([self.FEATURESPACE[fidx].run, queried])
            feature_map.append((tid, bid, fidx))

      printit(f"{self.classname}: Trajectory {tid+1} yields {len(tasks)} frame-slices (tasks) for the featurization. ")
      
      # Remove the dependency on the multiprocessing due to high overhead
      results = [wrapper_runner(*task) for task in tasks] 
      printit(f"{self.classname}: Tasks are finished, dumping the results to the feature space...")

      if config.verbose() or config.debug():
        printit(f"{self.classname}: Dumping the results to the feature space...")
        
      # Dump to file for each feature
      for feat_meta, result in zip(feature_map, results):
        tid, bid, fidx = feat_meta
        self.FEATURESPACE[fidx].dump(result)

      msg = f"Finished the trajectory {tid+1} / {self.TRAJECTORYNUMBER} with {len(tasks)} tasks in {time.perf_counter() - st:.6f} seconds"
      msg = f"{msg:=^80}"
      if tid < self.SLICENUMBER - 1:
        msg += "\n"
      printit(f"{self.classname}: {msg}")
    printit(f"{self.classname}: All trajectories and tasks are finished. \n")

  def loop_by_residue(self, restype, tag_limit=0): 
    """
    TO BE ADDED
    """
    for tid in range(self.TRAJECTORYNUMBER):
      # Setup the trajectory and its related parameters such as slicing of the trajectory
      self.traj = self.TRAJLOADER[tid]
      printit(f"{self.classname}: Start processing the trajectory {tid+1} with {self.SLICENUMBER} frames")

      # Cache the weights for each atoms in the trajectory (run once for each trajectory)
      for feat in self.FEATURESPACE:
        feat.cache(self.traj)
      
      # Calculate the slices to pro cess based on the single / dual residue tag
      tasks = []
      feature_map = []
      for bid in range(self.SLICENUMBER): 
        frames = self.traj.xyz[self.FRAMESLICES[bid]]
        if restype == "single":  
          for single_resname in (constants.RES + [i for i in constants.RES_PATCH.keys()]): 
            if single_resname in constants.RES_PATCH.keys():
              label = constants.RES2LAB[constants.RES_PATCH[single_resname]]
            else: 
              label = constants.RES2LAB[single_resname]
            # Find all of the residue block in the sequence and iterate them 
            slices = utils.find_block_single(self.traj, single_resname) 
            for sidx, s_ in enumerate(slices):
              if tag_limit > 0 and sidx >= tag_limit:
                break
              sliced_top = self.traj.top[s_]
              sliced_coord = frames[:, s_, :] 
              focal_point = np.mean(sliced_coord[0], axis=0)
              for fidx in range(self.FEATURENUMBER):
                queried = self.FEATURESPACE[fidx].query(sliced_top, sliced_coord.copy(), focal_point)
                tasks.append([self.FEATURESPACE[fidx].run, queried])
                feature_map.append((tid, bid, fidx, label))

        elif restype == "dual":
          # for label, dual_resname in constants.LAB2RES_DUAL.items(): 
          for res1 in (constants.RES + [i for i in constants.RES_PATCH.keys()]): 
            for res2 in (constants.RES + [i for i in constants.RES_PATCH.keys()]): 
              tmp_key = ""
              if res1 in constants.RES_PATCH.keys():
                tmp_key += constants.RES_PATCH[res1]
              else:
                tmp_key += res1
              if res2 in constants.RES_PATCH.keys():
                tmp_key += constants.RES_PATCH[res2]
              else:
                tmp_key += res2
              label = constants.RES2LAB_DUAL.get(tmp_key, "Unknown")
              dual_resname = res1 + res2
              # Find the residue block in the sequence.
              slices = utils.find_block_dual(self.traj, dual_resname)
              for s_ in slices:
                sliced_top = self.traj.top[s_]
                sliced_coord = frames[:, s_, :] 
                focal_point = np.mean(sliced_coord[0], axis=0)
                for fidx in range(self.FEATURENUMBER):
                  queried = self.FEATURESPACE[fidx].query(sliced_top, sliced_coord.copy(), focal_point)
                  tasks.append([self.FEATURESPACE[fidx].run, queried])
                  feature_map.append((tid, bid, fidx, label))
      
      printit(f"{self.classname}: Task set containing {len(tasks)} tasks are created for the trajectory {tid+1}; ")
      results = [wrapper_runner(*task) for task in tasks]

      printit(f"{self.classname}: Tasks are finished, dumping the results to the feature space...")

      # Dump to file for each feature
      for feat_meta, result in zip(feature_map, results):
        tid, bid, fidx, label = feat_meta
        self.FEATURESPACE[fidx].dump(result)
      
      if self.FEATURE_PARMS.get("outfile", None) is not None: 
        # Dump the label to the file
        labels = np.array([i[-1] for i in feature_map], dtype=int) 
        if self.FEATURE_PARMS.get("hdf_compress_level", 0) > 0:
          utils.append_hdf_data(self.FEATURE_PARMS["outfile"], "label", labels[:int(len(feature_map)/len(self.FEATURESPACE))], dtype=int, maxshape=(None, ), chunks=True, compress_level=self.FEATURE_PARMS.get("hdf_compress_level", 0))
        else: 
          utils.append_hdf_data(self.FEATURE_PARMS["outfile"], "label", labels[:int(len(feature_map)/len(self.FEATURESPACE))], dtype=int, maxshape=(None, ), chunks=True)
      
      if config.verbose() or config.debug():
        printit(f"{self.classname}: Finished the trajectory {tid+1} with {len(tasks)} tasks")
    printit(f"{self.classname}: All trajectories and tasks are finished")

