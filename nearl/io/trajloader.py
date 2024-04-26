import numpy as np
from .traj import  Trajectory


__all__ = [
  "TrajectoryLoader",
]


class TrajectoryLoader: 
  """
  A class to load multiple trajectories for further processing

  Attributes
  ----------
  trajs : list
    A list of trajectory arguments
  trajtype : trajectory_like
    The trajectory type to be used while loading the trajectory
  loading_options : dict
    The loading options (stride, frame_indices, mask)

  """
  def __init__(self, trajs = None, trajtype = None, **kwarg):
    """
    Systematically load trajectories for further processing

    Parameters
    ----------
    trajs : str or list
      Trajectory file names
    trajtype: trajectory_like
      The trajectory type to be used while loading the trajectory
    """
    self.trajs = []
    if isinstance(trajs, (list, tuple)):
      for traj in trajs:
        self.trajs.append(tuple(traj))
    elif hasattr(trajs, "__iter__"):
      # Check if it is iterable
      for traj in trajs:
        self.trajs.append(tuple(traj))
    else: 
      raise ValueError(f"The input should be a list or tuple of trajectory arguments rather than {type(trajs)}")

    if len(self.trajs) > 0: 
      if trajtype is None:
        # If the user does not specify the output type, use Trajectory as default
        self.OUTPUT_TYPE = [Trajectory] * len(self.trajs)
      else:
        self.OUTPUT_TYPE = [trajtype] * len(self.trajs)
    else: 
      self.OUTPUT_TYPE = []

    # Remember the user's configuration
    self.__loading_options = {"stride": None, "frame_indices": None, "mask": None, "superpose": False }
    self.loading_options = kwarg

  @property
  def loading_options(self): 
    """
    Get the loading options (stride, frame_indices, mask)
    """
    return {key: value for key, value in self.__loading_options.items()}
  @loading_options.setter
  def loading_options(self, kwargs):
    """
    Update the loading options (stride, frame_indices, mask)
    """
    for key, value in kwargs.items():
      if key in kwargs:
        self.__loading_options[key] = value

  def __str__(self):
    outstr = ""
    for i in self.__iter__(): 
      outstr += i.__str__()
    return outstr.strip("\n")

  def __iter__(self):
    """
    Iterate through the trajectories in the loader

    Yields
    ------
    trajectory_like
      The trajectory object
    """
    options = self.loading_options
    for i in range(len(self)): 
      yield self.OUTPUT_TYPE[i](*self.trajs[i], **options)

  def __len__(self):
    """
    Get the number of trajectories in the loader
    """
    return len(self.trajs)

  def __getitem__(self, index):
    """
    Get the trajectory object by index

    Parameters
    ----------
    index : int, list, tuple, slice
      The index of the trajectory object to be retrieved

    Returns
    -------
    trajectory_like or list
      The trajectory object or a list of trajectory objects
    """
    options = self.loading_options
    if isinstance(index, int):
      ret = self.OUTPUT_TYPE[index](*self.trajs[index], **options)
    elif isinstance(index, (list, tuple)):
      tmpindices = np.array(index, dtype=int)
      ret = [self.OUTPUT_TYPE[i](*self.trajs[i], **options) for i in tmpindices]
    elif isinstance(index, (slice, np.ndarray)):
      tmpindices = np.arange(self.__len__())[index]
      ret = [self.OUTPUT_TYPE[i](*self.trajs[i], **options) for i in tmpindices]
    else: 
      raise IndexError("Index must be either an integer or a slice")
    return ret
  
  def append(self, trajs = None, trajtype = None): 
    """
    Append a trajectory or a list of trajectories to the trajectory loader

    Parameters
    ----------
    trajs : str or list
      Trajectory file names
    trajtype: trajectory_like
      The trajectory type to be used while loading the trajectory
    """
    # determine how many trajectories will be appended
    traj_nr = len(trajs) if isinstance(trajs, (list, tuple)) else 0
    if traj_nr == 0: 
      raise ValueError(f"The input should be a list or tuple of trajectory arguments rather than {type(trajs)}")
    # Append to self.trajs, self.tops and self.OUTPUT_TYPE
    for i in range(traj_nr):
      self.trajs.append(trajs)
      # self.tops.append(None)
      self.OUTPUT_TYPE.append(trajtype)

