import time

import numpy as np
from tqdm import tqdm

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
  **kwargs : dict
    A dictionary of parameters for the featurizer
  
    
  Attributes
  ----------
  dims : np.ndarray, default = [32, 32, 32]
    The dimensions of the 3D grid
  lengths : np.ndarray, default = [16, 16, 16]
    The lengths of the 3D grid
  spacing : float, default = 0.5
    The spacing of the 3D grid
  time_window : int, default = 1
    The time window for the trajectory (default is 1), Simple integer.

  traj : :class:`nearl.Trajectory <nearl.io.traj.Trajectory>` or pytraj.Trajectory
    The trajectory to be processed
  FRAMENUMBER : int
    The number of frames in the trajectory to be processed
  FRAMESLICENUMBER : int
    The number of slices of frames in the trajectory
  FRAMESLICES : list
    A list of slices of frames in the trajectory to be processed

  TRAJLOADER : nearl.io.trajloader.TrajectoryLoader
    A trajectory iterator for the featurizer
  TRAJECTORYNUMBER : int
    The number of trajectories to be processed

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

  .. note::

    Required parameters:

    - **dimensions**: the dimensions of the 3D grid
    - lengths: the lengths of the 3D grid (optinal)
    - spacing: the spacing of the 3D grid (optinal)
    - time_window: the time window for the trajectory (default is 1), Simple integer.
    
    The following are optional parameters for features. 
    If the initialization of the feature did not explicit define the following parameters, the following parameters will be inherited from the featurizer: 

    - outfile: The output file to dump the parameters and the results
    - sigma: The smoothness of the Gaussian-based feature distribution
    - cutoff: The cutoff distance for the grid-based feature calculation

  """
  def __init__(self, parms={}, **kwargs):
    """
    Initialize the featurizer with the given parameters
    """
    # Check the essential parameters for the featurizer
    assert "dimensions" in parms, "Please define the 'dimensions' in the parameter set"
    assert ("lengths" in parms) or ("spacing" in parms), "Please define the 'lengths' or 'spacing' in the parameter set"

    # Basic parameters for the featurizer to communicate with cuda code
    self.__lengths = None
    self.dims = parms.get("dimensions", 32)   # Set the dimensions of the 3D grid
    if "lengths" in parms:
      self.lengths = parms.get("lengths", 16)   # Set the lengths of the 3D grid
      self.__spacing = np.mean(self.__lengths / self.__dims)
    elif "spacing" in parms:
      self.__spacing = parms.get("spacing", 1.0)
      self.__lengths = self.__dims * self.__spacing  # Directly assignment avoid the re-calculation of the spacing

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
    
    if config.verbose():
      printit(f"{self.__class__.__name__}: Featurizer is initialized successfully with dimensions: {self.__dims} and lengths: {self.__lengths}")

    if "outfile" in parms.keys():
      # Dump the parm dict to that hdf file
      tmpdict = {**parms, **kwargs}
      {'dimensions': 10, 'lengths': 16, 'time_window': 10, 'sigma': 1.5, 'cutoff': 2.55, 'outfile': '/tmp/test.h5', 'progressbar': True}
      tmpdict["dimensions"] = [int(i) for i in self.dims]
      tmpdict["lengths"] = [float(i) for i in self.lengths]
      print(tmpdict) # TODO
      utils.dump_dict(parms["outfile"], "featurizer_parms", tmpdict)

  def __str__(self):
    finalstr = f"Feature Number: {self.FEATURENUMBER}; \n"
    for feat in self.FEATURESPACE:
      finalstr += f"Feature: {feat.__str__()}\n"
    return finalstr

  # The most important attributes to determine the size of the 3D grid
  @property
  def dims(self):
    """
    The 3 dimensions of the 3D grid
    """
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
    
  # The most important attributes to determine the size of the 3D grid
  @property
  def lengths(self):
    """
    The lengths of the grid in 3 dimensions
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
      self.__spacing = np.mean(self.__lengths / self.__dims)

  @property
  def spacing(self):
    """
    The spacing between grid points, could also be understood as the **resolution** of the 3D grid
    """
    return self.__spacing  
  
  @property
  def traj(self):
    """
    Structures are regarded as trajectories (static PDB is view as a trajectory with only 1 frame). 
    """
    return self._traj
  @traj.setter
  def traj(self, the_traj):
    """
    Set the trajectory and its related parameters
    """
    self._traj = the_traj
    self.FRAMENUMBER = the_traj.n_frames
    self.SLICENUMBER = self.FRAMENUMBER // self.time_window
    if self.FRAMENUMBER % self.time_window != 0 and self.FRAMENUMBER != 1:
      printit(f"{self.__class__.__name__} Warning: the number of frames ({self.FRAMENUMBER}) is not divisible by the time window ({self.time_window}). The last few frames will be ignored.")
    printit(f"{self.__class__.__name__}: Registered {self.SLICENUMBER} slices of frames with {self.time_window} as the time window (frames-per-slice).")
    frame_array = np.array([0] + np.cumsum([self.time_window] * self.SLICENUMBER).tolist())
    self.FRAMESLICES = [np.s_[frame_array[i]:frame_array[i+1]] for i in range(self.SLICENUMBER)]

  @property
  def top(self):
    return self.traj.top

  def register_feature(self, feature):
    """
    Register a feature to the featurizer
    
    Parameters
    ----------
    feature : nearl.features.Feature
      A feature object
    """
    feature.hook(self)  # Hook the featurizer to the feature
    self.FEATURESPACE.append(feature)
    self.FEATURENUMBER = len(self.FEATURESPACE)
    output_keys = [i.outkey for i in self.FEATURESPACE]
    if len(set(output_keys)) != len(output_keys): 
      raise ValueError("The output keys for the features should be unique")
  
  def register_features(self, features):
    """
    Register a list of features to the featurizer

    Parameters
    ----------
    features : list_like or dict_like 
      A list or dictionary like object of a set of features (nearl.features.Feature) 
    """
    if isinstance(features, (list, tuple)):
      for feature in features:
        self.register_feature(feature)
    elif isinstance(features, dict):
      for _, feature in features.items():
        if config.verbose() or config.debug():
          printit(f"{self.__class__.__name__}: Registering the feature named: {_} from {feature.__class__.__name__} class")
        self.register_feature(feature)

  def register_trajloader(self, trajloader):
    """
    Register a trajectory iterator for future featurization

    Parameters
    ----------
    trajloader : :class:`nearl.io.trajloader.TrajectoryLoader`
      A trajectory iterator
    
    """
    self.TRAJLOADER = trajloader
    self.TRAJECTORYNUMBER = len(trajloader)
    print(f"{self.__class__.__name__}: Registered {self.TRAJECTORYNUMBER} trajectories")


  def register_focus(self, focus, format):
    """
    Register the focal points to process in the featurization

    Parameters
    ----------
    focus : list_like
      The focal points to process
    format : str
      The format of the focal points

    Notes
    -----
    .. note::

      Definition of focus follows the following formats:

      - "mask": provide a selection of atoms (Amber's selection convention)
      - "absolute": provide a list of 3D coordinates
      - "index": provide a list of atom indexes (int)

    Focal points are applied to each slice of the trajectory

    For each trajectory, the parse of focal points should be re-done to match the trajectory
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

    elif format == "function": 
      if not callable(focus):
        raise ValueError("The focus should be a callable function when the format is 'function'")
      self.FOCALPOINTS_PROTOTYPE = focus
      self.FOCALPOINTS_TYPE = "function"
      self.FOCALPOINTS = None

    else: 
      raise ValueError(f"Unexpected focus format: {format}. Please choose from 'mask', 'absolute', 'index', 'function'")

  def parse_focus(self): 
    """
    Parse the focal points to the correct format after the active trajectories are registered. 
    This will be run in the main_loop method. 

    Notes
    -----
    Prepare the focal points for each slice of the trajectory by slice_number * focus number 

    """
    if self.FOCALPOINTS_TYPE == "function":
      self.FOCALPOINTS = self.FOCALPOINTS_PROTOTYPE(self.traj)
      assert self.FOCALPOINTS.shape.__len__() == 3, "The focal points should be a 3D array"
      self.FOCALNUMBER = self.FOCALPOINTS.shape[1]
      return 1

    # Parse the focus points to the correct format
    self.FOCALPOINTS = np.full((self.SLICENUMBER, len(self.FOCALPOINTS_PROTOTYPE), 3), 99999, dtype=np.float32)
    self.FOCALNUMBER = len(self.FOCALPOINTS_PROTOTYPE)
    if self.FOCALPOINTS_TYPE == "mask":
      # Get the center of geometry for the frames with self.interval
      for midx, mask in enumerate(self.FOCALPOINTS_PROTOTYPE): 
        selection = self.traj.top.select(mask)
        for fidx in range(self.SLICENUMBER):
          frame = self.traj.xyz[fidx*self.time_window]
          self.FOCALPOINTS[fidx, midx] = np.mean(frame[selection], axis=0)

    elif self.FOCALPOINTS_TYPE == "index":
      for midx, mask in enumerate(self.FOCALPOINTS_PROTOTYPE): 
        for idx, frame in enumerate(self.traj.xyz[::self.time_window]):
          self.FOCALPOINTS[idx, midx] = np.mean(frame[mask], axis=0)

    elif self.FOCALPOINTS_TYPE == "absolute":
      for focusidx, focus in enumerate(self.FOCALPOINTS_PROTOTYPE): 
        assert len(focus) == 3, "The focus should be a 3D coordinate"
        for idx, frame in enumerate(self.traj.xyz[::self.time_window]):
          self.FOCALPOINTS[idx, focusidx] = focus
      
    else:
      raise ValueError(f"Unexpected focus format: {self.FOCALPOINTS_TYPE}")

  def main_loop(self, process_nr=20): 
    """

    """
    for tid in range(self.TRAJECTORYNUMBER):
      # Setup the trajectory and its related parameters such as slicing of the trajectory
      self.traj = self.TRAJLOADER[tid]
      msg = f"Processing traj {tid} ({self.traj.identity}) with {self.SLICENUMBER} frame slices"
      printit(f"{self.__class__.__name__}: {msg:=^80}")
      st = time.perf_counter()

      # Cache the weights for each atoms in the trajectory (run once for each trajectory)
      for feat in self.FEATURESPACE:
        if config.verbose(): 
          printit(f"{self.__class__.__name__}: Caching the weights of feature {feat.__class__.__name__} for the trajectory {tid}")
        feat.cache(self.traj)

      if self.FOCALPOINTS_PROTOTYPE is not None:
        # NOTE: Re-parse the focal points for each trajectory
        # Expected output shape is (self.SLICENUMBER, self.FOCALNUMBER, 3) array 
        if config.verbose() or config.debug():
          printit(f"{self.__class__.__name__}: Re-parsing focal points for the trajectory {tid}")
        self.parse_focus() 
        

      tasks = []
      feature_map = []
      # Pool the actions for each trajectory
      for bid in range(self.SLICENUMBER): 
        frames = self.traj.xyz[self.FRAMESLICES[bid]]
        if self.FOCALNUMBER > 0:
          # After determineing each focus point, run the featurizer for each focus point
          for pid in range(self.FOCALNUMBER):
            focal_point = self.FOCALPOINTS[bid, pid]

            # Crop the trajectory and send the coordinates/trajectory to the featurizer
            for fidx in range(self.FEATURENUMBER):
              # Explicitly transfer the topology and frames to get the queried coordinates for the featurizer
              # NOTE: pass a copy of frames to the querier function to avoid in-place modification of the frames 
              # NOTE: Isolate the effect on the calculation of the next feature 
              queried = self.FEATURESPACE[fidx].query(self.top, frames.copy(), focal_point)
              tasks.append([self.FEATURESPACE[fidx].run, queried])
              feature_map.append((tid, bid, fidx))
        else:
          # Without registeration of focal points: focal-point independent features such as label-generation
          for fidx in range(self.FEATURENUMBER):
            # Explicitly transfer the topology and frames to get the queried coordinates for the featurizer
            queried = self.FEATURESPACE[fidx].query(self.top, frames.copy(), [0, 0, 0])
            tasks.append([self.FEATURESPACE[fidx].run, queried])
            feature_map.append((tid, bid, fidx))

      printit(f"{self.__class__.__name__}: Tasks are ready for the trajectory {tid} with {len(tasks)} tasks")
      
      if self.OTHER_PARMS.get("progressbar", False):
        results = [wrapper_runner(*task) for task in tqdm(tasks)]
      else:
        results = [wrapper_runner(*task) for task in tasks]
      # Run the actions in the process pool
      # _tasks = [pool.apply_async(wrapper_runner, task) for task in tasks]
      # results = [task.get() for task in _tasks]

      printit(f"{self.__class__.__name__}: Tasks are finished, dumping the results to the feature space...")

      if config.verbose() or config.debug():
        printit(f"{self.__class__.__name__}: Dumping the results to the feature space...")
        
      # Dump to file for each feature
      for feat_meta, result in zip(feature_map, results):
        tid, bid, fidx = feat_meta
        self.FEATURESPACE[fidx].dump(result)

      msg = f"Finished the trajectory {tid+1} / {self.TRAJECTORYNUMBER} with {len(tasks)} tasks in {time.perf_counter() - st:.2f} seconds"
      printit(f"{self.__class__.__name__}: {msg:^^80}\n")
    printit(f"{self.__class__.__name__}: All trajectories and tasks are finished")

  def loop_by_residue(self, restype, process_nr=20, tag_limit=0): 
    """
    """
    for tid in range(self.TRAJECTORYNUMBER):
      # Setup the trajectory and its related parameters such as slicing of the trajectory
      self.traj = self.TRAJLOADER[tid]
      printit(f"{self.__class__.__name__}: Start processing the trajectory {tid} with {self.SLICENUMBER} frames")

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
      
      printit(f"{self.__class__.__name__}: Task set containing {len(tasks)} tasks are created for the trajectory {tid}; ")
      ######################################################
      # TODO: Find a proper way to parallelize the CUDA function. 
      if self.OTHER_PARMS.get("progressbar", False):
        results = [wrapper_runner(*task) for task in tqdm(tasks)]
      else:
        results = [wrapper_runner(*task) for task in tasks]
      
      ######################################################
      # Run the actions in the process pool 
      # Not working
      # pool = mp.Pool(process_nr)
      # _tasks  = [pool.apply_async(wrapper_runner, task) for task in tasks]
      # results = [task.get() for task in _tasks]
      # OR results = pool.starmap(wrapper_runner, tasks) 
      ######################################################
      # Not working
      # with dask.config.set(scheduler='processes', num_workers=process_nr):
      #   results = dask.compute(*[dask.delayed(wrapper_runner)(func, args) for func, args in tasks])
      ######################################################

      printit(f"{self.__class__.__name__}: Tasks are finished, dumping the results to the feature space...")

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
        printit(f"{self.__class__.__name__}: Finished the trajectory {tid} with {len(tasks)} tasks")
    printit("f{self.__class__.__name__}: All trajectories and tasks are finished")

