import tempfile

import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from . import utils, constants
from . import printit, config

__all__ = [
  "Featurizer",
]


class Featurizer:
  """
  Featurizer aims to automate the process of featurization of multiple Features for a batch of structures or trajectories
  """
  def __init__(self, parms):
    """
    Initialize the featurizer with the given parameters
    Parameters
    ----------
    parms: dict
      A dictionary of parameters for the featurizer
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
    self.FOCALNUMBER = 0

    # Component III: Trajectory space
    self.TRAJLOADER = None
    self.TRAJECTORYNUMBER = 0
    
    if config.verbose():
      printit(f"Featurizer is initialized successfully with dimensions: {self.__dims} and lengths: {self.__lengths}")

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
    self.__boxcenter = np.array(self.dims/2, dtype=float)
    
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
    self.SLICENUMBER = self.FRAMENUMBER // self.interval
    if self.FRAMENUMBER % self.interval != 0:
      printit("Warning: the number of frames is not divisible by the interval. The last few frames will be ignored.")
    printit(f"Registered {self.SLICENUMBER} slices for the trajectory ({self.FRAMENUMBER}) with {self.interval} interval.")
    frame_array = np.array([0] + np.cumsum([self.__interval] * self.SLICENUMBER).tolist())
    self.FRAMESLICES = [np.s_[frame_array[i]:frame_array[i+1]] for i in range(self.SLICENUMBER)]

  @property
  def top(self):
    return self.traj.top

  def update_box_length(self, length=None, scale_factor=1.0):
    if length is not None:
      self.__lengths = float(length)
    else:
      self.__lengths *= scale_factor
    self.update_box()

  def register_feature(self, feature):
    """
    Register a feature to the featurizer
    
    Parameters
    ----------
    feature: nearl.features.Feature
      A feature object
    """
    feature.hook(self)  # Hook the featurizer to the feature
    self.FEATURESPACE.append(feature)
    self.FEATURENUMBER = len(self.FEATURESPACE)
  
  def register_features(self, features):
    """
    Register a list of features to the featurizer

    Parameters
    ----------
    features: list_like
      A list of feature objects
    """
    for feature in features:
      self.register_feature(feature)

  def register_trajloader(self, trajloader):
    """
    Register a trajectory iterator for future featurization

    Parameters
    ----------
    trajloader: nearl.io.TrajectoryLoader
      A trajectory iterator
    """
    self.TRAJLOADER = trajloader
    self.TRAJECTORYNUMBER = len(trajloader)
    print(f"Registered {self.TRAJECTORYNUMBER} trajectories")


  def register_focus(self, focus, format):
    """
    Register the focal points to process in the featurization

    Parameters
    ----------
    focus: 
      The focal points to process
    format: string
      The format of the focal points

    Notes
    -----
    Formats includes:
    - "cog": provide a masked selection of atoms
    - "absolute": provide a list of 3D coordinates
    - "index": provide a list of atom indexes (int)

    Focal points are applied to each slice of the trajectory

    For each trajectory, the parse of focal points should be re-done to match the trajectory
    """
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
    """
    Parse the focal points to the correct format after the active trajectories are registered. 
    This will be run in the main_loop method. 

    Notes
    -----
    Prepare the focal points for each slice of the trajectory by slice_number * focus number 

    """
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

  def main_loop(self, process_nr=20): 
    for tid in range(self.TRAJECTORYNUMBER):
      # Setup the trajectory and its related parameters such as slicing of the trajectory
      self.traj = self.TRAJLOADER[tid]
      printit(f"{self.__class__.__name__}: Start processing the trajectory {tid} with {self.SLICENUMBER} frames")

      # Cache the weights for each atoms in the trajectory (run once for each trajectory)
      for feat in self.FEATURESPACE:
        feat.cache(self.traj)

      # Update the focus points for each bin as (self.SLICENUMBER, self.FOCALNUMBER, 3) array (run once for each trajectory)
      self.parse_focus() 
      print("Focal points shape: ", self.FOCALPOINTS.shape)

      tasks = []
      feature_map = []
      # Pool the actions for each trajectory
      for bid in range(self.SLICENUMBER): 
        frames = self.traj.xyz[self.FRAMESLICES[bid]]

        # After determineing each focus point, run the featurizer for each focus point
        for pid in range(self.FOCALNUMBER):
          # printit(f"Processing the focal point {pid} at the bin {bid}")
          focal_point = self.FOCALPOINTS[bid, pid]

          # Crop the trajectory and send the coordinates/trajectory to the featurizer
          for fidx in range(self.FEATURENUMBER):
            # Explicitly transfer the topology and frames to get the queried coordinates for the featurizer
            queried = self.FEATURESPACE[fidx].query(self.top, frames, focal_point)
            tasks.append([self.FEATURESPACE[fidx].run, queried])
            feature_map.append((tid, bid, pid, fidx))
      printit(f"Tasks are ready for the trajectory {tid} with {len(tasks)} tasks")
      
      results = [wrapper_runner(*task) for task in tqdm(tasks)]
      # Run the actions in the process pool
      # _tasks = [pool.apply_async(wrapper_runner, task) for task in tasks]
      # results = [task.get() for task in _tasks]

      printit("Tasks are finished, dumping the results to the feature space...")

      if config.verbose() or config.debug():
        printit("Dumping the results to the feature space...")
        
      # Dump to file for each feature
      for feat_meta, result in zip(feature_map, results):
        tid, bid, pid, fidx = feat_meta
        self.FEATURESPACE[fidx].dump(result)

      if config.verbose() or config.debug():
        printit(f"Finished the trajectory {tid} with {len(tasks)} tasks")
      break
    printit("All trajectories and tasks are finished")

  def loop_by_residue(self, process_nr=20, restype="single"): 
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
          for label, single_resname in constants.LAB2RES.items(): 
            # Find all of the residue block in the sequence and iterate them 
            slices = utils.find_block_single(self.traj, single_resname) 
            for s_ in slices:
              sliced_top = self.traj.top[s_]
              sliced_coord = frames[:, s_, :] 
              focal_point = np.mean(sliced_coord[0], axis=0)
              for fidx in range(self.FEATURENUMBER):
                queried = self.FEATURESPACE[fidx].query(sliced_top, sliced_coord, focal_point)
                tasks.append([self.FEATURESPACE[fidx].run, queried])
                feature_map.append((tid, bid, fidx, label))

        elif restype == "dual":
          for label, dual_resname in constants.LAB2RES_DUAL.items(): 
            # Find the residue block in the sequence.
            slices = utils.find_block_dual(self.traj, dual_resname)
            for s_ in slices:
              sliced_top = self.traj.top[s_]
              sliced_coord = frames[:, s_, :] 
              focal_point = np.mean(sliced_coord[0], axis=0)
              for fidx in range(self.FEATURENUMBER):
                queried = self.FEATURESPACE[fidx].query(sliced_top, sliced_coord, focal_point)
                tasks.append([self.FEATURESPACE[fidx].run, queried])
                feature_map.append((tid, bid, fidx, label))

      printit(f"Task set containing {len(tasks)} tasks are created for the trajectory {tid}; ")
      
      ######################################################
      # TODO: Find a proper way to parallelize the CUDA function. 
      results = [wrapper_runner(*task) for task in tqdm(tasks)] 
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
      printit(f"Tasks are finished, dumping the results to the feature space...")

      # Dump to file for each feature
      for feat_meta, result in zip(feature_map, results):
        tid, bid, fidx, label = feat_meta
        self.FEATURESPACE[fidx].dump(result)
      if config.verbose() or config.debug():
        printit(f"Finished the trajectory {tid} with {len(tasks)} tasks")
      break
    printit("All trajectories and tasks are finished")

def wrapper_runner(func, args):
  """
  Take the feature.run methods and its input arguments for multiprocessing

  Parameters
  ----------
  func: function
    The function to be run
  args: list
    The arguments for the function
  """
  return func(*args)  

