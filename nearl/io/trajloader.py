from .traj import  Trajectory


__all__ = [
  "TrajectoryLoader",
]


class TrajectoryLoader: 
  def __init__(self, trajs, tops, **kwarg):
    """
    Systematically load trajectories for further processing
    Args:
        trajs (str or list): Trajectory file names
        tops (str or list): Topology file names
        **kwarg: Keyword arguments for pytraj.load
    """
    if isinstance(trajs, str):
      self.trajs = [trajs]
      self.tops = [tops]
    elif isinstance(trajs, list):
      self.trajs = trajs
      self.tops = tops
    self.kwargs = kwarg
    # If the user does not specify the output type, use Trajectory
    self.OUTPUT_TYPE = Trajectory

  def __str__(self):
    outstr = ""
    for i in self.__iter__(): 
      outstr += i.traj.__str__().replace("\n", "\t")+"\n"
    return outstr.strip("\n")

  def __iter__(self):
    return self.__loadtrajs(self.trajs, self.tops)

  def __len__(self):
    return len(self.trajs)

  def __getitem__(self, index):
    used_kwargs = self.__desolvekwargs()
    if isinstance(index, int):
      ret = self.OUTPUT_TYPE(self.trajs[index], self.tops[index], **used_kwargs)
    elif isinstance(index, slice):
      ret = [self.OUTPUT_TYPE(traj, top, **used_kwargs) for traj, top in zip(self.trajs[index], self.tops[index])]
    return ret

  def set_outtype(self, outtype):
    self.OUTPUT_TYPE = outtype

  def __loadtrajs(self, trajs, tops):
    used_kwargs = self.__desolvekwargs()
    for traj, top in zip(trajs, tops):
      yield self.OUTPUT_TYPE(traj, top, **used_kwargs)

  def __desolvekwargs(self):
    ret_kwargs = {}
    ret_kwargs["stride"] = self.kwargs.get("stride", None)
    ret_kwargs["mask"] = self.kwargs.get("mask", None)
    ret_kwargs["frame_indices"] = self.kwargs.get("frame_indices", None)
    return ret_kwargs

