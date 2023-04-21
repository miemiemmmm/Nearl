import pytraj as pt 
from . import utils

# Loader of trajectories for iteration
# Basic way of using: 
# for traj in trajloader: 
#   doit()
class TrajectoryLoader: 
  def __init__(self, trajs, tops):
    if isinstance(trajs, str):
      self.trajs = [trajs]; 
      self.tops  = [tops]; 
    elif isinstance(trajs, list):
      self.trajs = trajs; 
      self.tops  = tops; 
  def __str__(self):
    outstr = ""
    for i in self.__iter__(): 
      outstr += i.traj.__str__().replace("\n", "\t")+"\n"
    return outstr.strip("\n")
  def __iter__(self):
    return self.__loadtrajs(self.trajs, self.tops); 
  def __getitem__(self, index):
    if isinstance(index, int):
      ret = TRAJ(self.trajs[index], self.tops[index]); 
    elif isinstance(index, slice):
      ret = [TRAJ(traj, top) for traj, top in zip(self.trajs[index], self.tops[index])]
    return ret
  def __loadtrajs(self, trajs, tops):
    for traj, top in zip(trajs, tops):
      yield TRAJ(traj, top)


# Trajectory object
class TRAJ: 
  def __init__(self, trajfile, pdbfile):
    self.traj = pt.load(trajfile, top=pdbfile); 
    self.traj.top.set_reference(self.traj[0]);
    self.activeframe = 0; 
  def __getitem__(self, key):
    return self.traj[key]
  @property
  def top(self):
    return self.traj.top
  @property
  def xyz(self): 
    return self.traj.xyz

  def copy(self):
    return self.traj.copy()

  def strip(self, mask): 
    self.traj.strip(mask)

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
    self.traj = self.traj; 
    
  def slicetraj(self, frameindices):
    frameindices = np.array(frameindices).astype(int)
    self.traj = pt.Trajectory(self.traj[frameindices])
    
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





