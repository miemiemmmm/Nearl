import sys
import numpy as np
from .traj import  Trajectory
from .. import config, printit


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
  def __init__(self, trajs = None, trajtype = None, trajids = None, **kwarg):
    # Initialize the trajectory container 
    self.trajs = []
    if isinstance(trajs, (list, tuple)):
      for traj in trajs:
        self.trajs.append(tuple(traj))
    elif hasattr(trajs, "__iter__"):
      for traj in trajs:
        self.trajs.append(tuple(traj))
    else: 
      raise ValueError(f"The input should be a list or tuple of trajectory arguments rather than {type(trajs)}")

    # Set the trajectory type
    if len(self.trajs) > 0: 
      if trajtype is None:
        # Default trajectory type
        self.OUTPUT_TYPE = [Trajectory] * len(self.trajs)
      else:
        # User-defined trajectory type 
        self.OUTPUT_TYPE = [trajtype] * len(self.trajs)
    else: 
      self.OUTPUT_TYPE = []
    
    # Set the trajectory identities
    if trajids is not None:
      if len(trajids) != len(self.trajs):
        raise ValueError(f"The number of trajectory ids {len(trajids)} does not match the number of trajectories {len(self.trajs)}")
      self.trajids = trajids
    else: 
      self.trajids = [None] * len(self.trajs)
      # self.trajids = [i[0] if isinstance(i, (list, tuple)) else None for i in trajs]

    # Setup loading configurations
    self.i_ = 0              # The index of the trajectory being processed 
    self.__loading_options = {"stride": None, "frame_indices": None, "mask": None, "superpose": False}
    self.loading_options = kwarg
    
    # Update the loading options if they are individially set for each trajectory 
    if kwarg.get("strides", None) is None:
      self.strides = [None] * len(self.trajs)
    else:
      if len(self.strides) != len(self.trajs):
        raise ValueError(f"The manually defined number of strides {len(self.strides)} does not match the number of trajectories {len(self.trajs)}")
      self.strides = [int(i) for i in kwarg.get("strides")]

    if kwarg.get("masks", None) is None: 
      self.masks = [None] * len(self.trajs)
    else:
      if len(self.masks) != len(self.trajs):
        raise ValueError(f"The manually defined number of masks {len(self.masks)} does not match the number of trajectories {len(self.trajs)}") 
      self.masks = kwarg.get("masks")

  @property
  def loading_options(self): 
    """
    Get the loading options (stride, frame_indices, mask)
    """
    options = {key: value for key, value in self.__loading_options.items() if value is not None}
    if config.verbose or config.debug:
      printit(f"{self.__class__.__name__}: Loading the trajectory {self.i_} whose identity is {self.trajids[self.i_]}")
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

  def get_loading_options(self, index): 
    """
    Get the loading options if the user manually defines the loading options for specific trajectories 

    Parameters
    ----------
    index : int
      The index of the trajectory object to be retrieved

    """
    ret_options = {k:v for k,v in self.__loading_options.items()}
    if self.strides[index] is not None:
      ret_options["stride"] = self.strides[index]
    if self.masks[index] is not None:
      ret_options["mask"] = self.masks[index]
    return ret_options

  def __str__(self):
    outstr = ""
    for i in self.__iter__(): 
      outstr += i.__str__()
    return outstr.strip("\n")

  def __iter__(self):
    """
    Iterate through the trajectories in the trajectory loader

    Yields
    ------
    trajectory_like
      The trajectory object
    """
    for i in range(len(self)): 
      options = self.get_loading_options(i)
      self.i_ = i
      printit(f"Loading the trajectory {i+1}/{len(self.trajs)}")
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
    if isinstance(index, int):
      if config.verbose or config.debug:
        printit(f"{self.__class__.__name__}: Setting the trajectory index to {index}")
      self.i_ = index
      options = self.get_loading_options(index)
      ret = self.OUTPUT_TYPE[index](*self.trajs[index], **options)

    elif isinstance(index, (list, tuple)):
      tmpindices = np.array(index, dtype=int)
      ret = []
      for i in tmpindices:
        self.i_ = i
        options = self.get_loading_options(i)
        ret.append(self.OUTPUT_TYPE[i](*self.trajs[i], **options))
      
    elif isinstance(index, slice):
      tmpindices = np.arange(self.__len__())[index]
      ret = []
      for i in tmpindices:
        self.i_ = i
        options = self.get_loading_options(i)
        ret.append(self.OUTPUT_TYPE[i](*self.trajs[i], **options))
      
    else: 
      raise IndexError("Index must be either an integer or a slice")
    
    return ret
  
  def append(self, trajs = None, trajtype = None, **kwarg): 
    """
    Append a trajectory or a list of trajectories to the trajectory loader. 

    Parameters
    ----------
    trajs : str or list
      Trajectory file names
    trajtype : trajectory_like, optional, default = :class:`nearl.io.traj.Trajectory`
      The trajectory type to be used while loading the trajectory
      
    """
    # Determine the number of trajectories to add 
    if not isinstance(trajs, (list, tuple)):
      raise ValueError(f"The input should be a list or tuple of trajectory arguments rather than {type(trajs)}") 
    traj_nr = len(trajs) 
    if traj_nr == 0: 
      raise ValueError(f"No trajectory detected in the input")  
    
    # Append to self.trajs, self.OUTPUT_TYPE and misc. loading options
    for idx, traj_arg in enumerate(trajs):
      self.trajs.append(traj_arg)
      self.OUTPUT_TYPE.append(trajtype)
      if kwarg.get("trajids", None) is not None:
        self.trajids.append(kwarg.get("trajids")[idx])
      else:
        self.trajids.append(None)
      if kwarg.get("strides", None) is not None: 
        self.strides.append(kwarg.get("strides")[idx])
      else:
        self.strides.append(None)
      if kwarg.get("masks", None) is not None:
        self.masks.append(kwarg.get("masks")[idx])
      else:
        self.masks.append(None)


