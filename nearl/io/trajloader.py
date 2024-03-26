import numpy as np
from .traj import  Trajectory


__all__ = [
  "TrajectoryLoader",
]


class TrajectoryLoader: 
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
    self.__loading_options = {"stride": None, "frame_indices": None, "mask": None }
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
    Update the loading options
    """
    for key, value in kwargs.items():
      if key in kwargs:
        self.__loading_options[key] = value

  def __str__(self):
    outstr = ""
    for i in self.__iter__(): 
      outstr += i.__str__()
      # outstr += i.traj.__str__().replace("\n", "\t")+"\n"
    return outstr.strip("\n")

  def __iter__(self):
    options = self.loading_options
    for i in range(len(self)): 
      yield self.OUTPUT_TYPE[i](*self.trajs[i], **options)

  def __len__(self):
    return len(self.trajs)

  def __getitem__(self, index):
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

