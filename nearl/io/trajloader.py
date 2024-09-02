import sys
import numpy as np
from .traj import  Trajectory


__all__ = [
  "TrajectoryLoader",
]


class TrajectoryLoader: 
  """
  A class to systematically load trajectories for further processing in the pipeline. 

  Parameters
  ----------
  trajs : str or list
    A list of trajectories to be loaded
  trajtype : trajectory_like
    The trajectory type to be used while loading the trajectory
  **kwarg : dict
    The loading options (stride, frame_indices, mask)
  
  Attributes
  ----------
  trajs : list
    A list of trajectories to be loaded
  trajtype : trajectory_like, optional, default = :class:`nearl.io.traj.Trajectory`
    The trajectory type to be used while loading the trajectory
  loading_options : dict, optional
    The loading options (stride, frame_indices, mask)
    
  Examples
  --------
  .. code-block:: python

    import nearl
    import nearl.data
    trajs = [
      nearl.data.MINI_TRAJ,
      nearl.data.MINI_TRAJ,
      nearl.data.MINI_TRAJ,
      nearl.data.MINI_TRAJ,
    ]
    loader = nearl.TrajectoryLoader(trajs)
    print(f"{len(loader)} trajectories detected")  # 4 trajectories detected
    loader[3].visualize()  # Visualize the last trajectory
    print(loader.loading_options)  # Print the loading options
    for traj in loader:
      print(traj)  # Print the trajectory information

  """
  def __init__(self, trajs = None, trajtype = None, trajid = None, **kwarg):
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

    self.i_ = 0
    if trajid is not None:
      self.trajids = trajid
      if len(trajid) != len(self.trajs):
        raise ValueError(f"The number of trajectory ids {len(trajid)} does not match the number of trajectories {len(self.trajs)}")
    else: 
      self.trajids = None

    # Remember the user's configuration
    self.__loading_options = {"stride": None, "frame_indices": None, "mask": None, "superpose": False }
    self.loading_options = kwarg

  @property
  def loading_options(self): 
    """
    Get the loading options (stride, frame_indices, mask)
    """
    options = {key: value for key, value in self.__loading_options.items() if value is not None}
    if self.trajids is not None: 
      options["identity"] = self.trajids[self.i_]
    return options
  
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
      self.i_ = i
      print(f"Loading trajectory {i+1}/{len(self)}", file=sys.stderr)
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
      self.i_ = index
      ret = self.OUTPUT_TYPE[index](*self.trajs[index], **options)
    elif isinstance(index, (list, tuple)):
      self.i_ = index[0]
      tmpindices = np.array(index, dtype=int)
      ret = [self.OUTPUT_TYPE[i](*self.trajs[i], **options) for i in tmpindices]
    elif isinstance(index, (slice, np.ndarray)):
      self.i_ = index.start
      tmpindices = np.arange(self.__len__())[index]
      ret = [self.OUTPUT_TYPE[i](*self.trajs[i], **options) for i in tmpindices]
    else: 
      raise IndexError("Index must be either an integer or a slice")
    return ret
  
  def append(self, trajs = None, trajtype = None): 
    """
    Append a trajectory or a list of trajectories to the trajectory loader. 

    Parameters
    ----------
    trajs : str or list
      Trajectory file names
    trajtype : trajectory_like, optional, default = :class:`nearl.io.traj.Trajectory`
      The trajectory type to be used while loading the trajectory
      
    """
    # determine how many trajectories will be appended
    traj_nr = len(trajs) if isinstance(trajs, (list, tuple)) else 0
    if traj_nr == 0: 
      raise ValueError(f"The input should be a list or tuple of trajectory arguments rather than {type(trajs)}")
    # Append to self.trajs, self.tops and self.OUTPUT_TYPE
    for i in range(traj_nr):
      self.trajs.append(trajs)
      self.OUTPUT_TYPE.append(trajtype)

