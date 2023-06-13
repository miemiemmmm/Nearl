import time;
import pytraj as pt;
import numpy as np;
from . import utils, CONFIG

# Loader of trajectories for iteration
# Basic way of using: 

class TrajectoryLoader: 
  def __init__(self, trajs, tops, **kwarg):
    """
    Systematically load trajectories for further processing
    Example:
    >>> for traj in trajloader:
    >>>   # Do something with traj
    Args:
        trajs (str or list): Trajectory file names
        tops (str or list): Topology file names
        **kwarg: Keyword arguments for pytraj.load
    """
    if isinstance(trajs, str):
      self.trajs = [trajs]; 
      self.tops  = [tops]; 
    elif isinstance(trajs, list):
      self.trajs = trajs; 
      self.tops  = tops;
    self.kwargs = kwarg;
  def __str__(self):
    outstr = ""
    for i in self.__iter__(): 
      outstr += i.traj.__str__().replace("\n", "\t")+"\n"
    return outstr.strip("\n")

  def __iter__(self):
    return self.__loadtrajs(self.trajs, self.tops); 
  def __getitem__(self, index):
    used_kwargs = self.__desolvekwargs();
    if isinstance(index, int):
      ret = Trajectory(self.trajs[index], self.tops[index], **used_kwargs);
    elif isinstance(index, slice):
      ret = [Trajectory(traj, top, **used_kwargs) for traj, top in zip(self.trajs[index], self.tops[index])]
    return ret
  def __loadtrajs(self, trajs, tops):
    used_kwargs = self.__desolvekwargs();
    for traj, top in zip(trajs, tops):
      yield Trajectory(traj, top, **used_kwargs)
  def __desolvekwargs(self):
    ret_kwargs = {}
    ret_kwargs["stride"] = self.kwargs.get("stride", None)
    ret_kwargs["mask"] = self.kwargs.get("mask", None)
    ret_kwargs["frame_indices"] = self.kwargs.get("frame_indices", None)
    return ret_kwargs



# Trajectory object
class Trajectory(pt.Trajectory):
  def __init__(self, trajfile, pdbfile, **kwarg):
    """
    Initialize the trajectory object from pytraj.Trajectory
    Add more customizable functions
    Args:
      trajfile: trajectory file
      pdbfile:  pdb file
      **kwarg:  keyword arguments
    """
    st = time.perf_counter();
    super().__init__(top=pdbfile);
    # Set the keyword arguments for slicing/masking trajectory;
    stride = kwarg.get("stride", None)
    frame_indices = kwarg.get("frame_indices", None)
    mask = kwarg.get("mask", None)
    if CONFIG.get("verbose", False):
      print(f"Module {self.__class__.__name__}: Loading trajectory {trajfile} with topology {pdbfile}");
      print(f"Module {self.__class__.__name__}: stride: {stride}, frame_indices: {frame_indices}, mask: {mask}")
    traj = pt.io.load_traj(trajfile, self.top, stride=stride);
    # Do the slicing and Masking if needed
    if stride is not None:
      if mask is None:
        buffer_traj = traj[:]
      else:
        buffer_traj = traj[mask]
    else:
      frame_indices_ = frame_indices
      # Convert the frame_indices to list if it is tuple
      if isinstance(frame_indices_, tuple):
        frame_indices_ = list(frame_indices_)
      # Load all frames if frame_indices_ is None
      if frame_indices_ is None and mask is None:
        buffer_traj = traj[:]
      elif frame_indices_ is None and mask is not None:
        # load all frames with given mask
        # eg: traj['@CA']
        buffer_traj = traj[mask]
      elif frame_indices_ is not None and mask is None:
        # eg. traj[[0, 3, 7]]
        buffer_traj = traj[frame_indices_]
      else:
        # eg. traj[[0, 3, 7], '@CA']
        buffer_traj = traj[frame_indices_, mask]
    # Reassign the trajectory if there is slicing or masking of the trajectory
    if (self.n_frames != buffer_traj.n_frames) or (self.n_atoms != buffer_traj.n_atoms):
      if CONFIG.get("verbose", False):
        print(f"Module {self.__class__.__name__}: Reassigning the trajectory; ");
      self.top = buffer_traj.top;
      self._xyz = buffer_traj._xyz;
      self._boxes = buffer_traj._boxes;
      self.forces = buffer_traj.forces;
      self.velocities = buffer_traj.velocities;
      self.time = buffer_traj.time;
      self._life_holder = buffer_traj._life_holder;
      self._frame_holder = buffer_traj._frame_holder;

    self.activeframe = 0;
    if CONFIG.get("verbose", False):
      print(f"Module {self.__class__.__name__}: Trajectory loaded in {time.perf_counter() - st:.2f} seconds");
  def copy(self):
    thecopy = pt.Trajectory();
    thecopy.top = self.top.copy()
    thecopy.xyz = self._xyz.copy()
    return thecopy

  def cluster_pairwise(self, mask, **kwarg): 
    if "countermask" in kwarg.keys():
      countermask = kwarg["countermask"]
      pdist, y = utils.PairwiseDistance(self.traj, f"{mask}&!@H=", f"{countermask}&!@H=", use_mean=True);
    else: 
      pdist, y = utils.PairwiseDistance(self.traj, f"{mask}&!@H=", f"{mask}<@6&!{mask}&@C,CA,CB,N,O", use_mean=True);
    clusters = cluster.ClusterAgglomerative(pdist, 10); 
    cluster_rand = cluster.RandomPerCluster(clusters, number=1); 
    self.frames = cluster_rand; 
    return self.frames
    
  def cluster_rmsd(self):
    pass
    
  def slicetraj(self, frameindices):
    frameindices = np.array(frameindices).astype(int)

    
  def cluster(self, method="", **kwarg):
    if len(method) == 0:
      self.cluster_pairwise(**kwarg); 
    elif (method == "distance"): 
      pass
    return self.frames
  
  def updatesearchlist(self, frame, mask, cutoff):
    refframe = pt.Frame(self.traj[frame]); 
    self.traj.top.set_reference(refframe); 
    self.traj.superpose(ref=refframe, mask="@CA")
    self.searchlist = self.traj.top.select(f"{mask}<@{cutoff}"); 
    self.activeframe = frame; 
    return self.searchlist;
  
  def test(self, coordinates, elements=[]):
    if len(elements) == 0: 
      elements = ["H"] * len(coordinates); 
    newframe = pt.Frame(self.traj[0])
    newtop = pt.Topology(self.traj.top)
    resids = [i.index for i in self.traj.top.residues]
    maxid = resids[-1]
    print(f"Before: {newframe.xyz.shape}")
    for i, c in enumerate(coordinates): 
      therid = (maxid+i+2); 
      print(therid, c)
      theatom = pt.Atom(name='CL', charge=0.04, mass=17.0, resname="BOX",type="H", resid=therid);
      theres  = pt.Residue(name='BOX', resid=therid)
      newtop.add_atom(theatom, theres)
      newframe.append_xyz(np.array([c]).astype(np.float64))
    print(f"After addition newframe {newframe.xyz.shape}, {newtop.n_atoms}")
    thexyz = np.array([newframe.xyz])
    newtraj = pt.Trajectory(xyz=thexyz, top=newtop)
    self.testtraj = newtraj; 
    pt.write_traj("/tmp/test.pdb", newtraj, overwrite=True)





