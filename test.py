import pytraj as pt 
import numpy as np 
from BetaPose import utils, cluster

class TRAJ:
  def __init__(self, trajfile, pdbfile):
    # print("Generate a new traj")
    self.traj = pt.load(trajfile, top=pdbfile);
    self.traj.top.set_reference(self.traj[0])

  def cluster_pairwise(self, mask, **kwarg):
    if "countermask" in kwarg.keys():
      countermask = kwarg["kwarg"]
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

  def updatesearchlist(self, mask, cutoff):
    self.searchlist = self.traj.top.select(f"{mask}<@{cutoff}");
    return self.searchlist;


class TrajectoryLoader: 
  def __init__(self, trajs, tops):
    if isinstance(trajs, str):
      self.trajs = [trajs]; 
      self.tops  = [tops]; 
    elif isinstance(trajs, list):
      self.trajs = trajs; 
      self.tops  = tops; 
  def __iter__(self):
    return self.__loadtrajs(self.trajs, self.tops); 
  
  def __loadtrajs(self, trajs, tops):
    for traj, top in zip(trajs, tops):
      yield TRAJ(traj, top)


class feature:
  def __init__(self):
    print("here")

  def __display__(self):
    print(self.traj);

  def __str__(self):
    return self.__class__.__name__

  def hook(self, featurizer):
    self.featurizer = featurizer
    self.top = featurizer.traj.top

  def run(self, trajectory):
    """
      update interval
      self.traj.superpose arguments.
      updatesearchlist arguments.
    """
    self.traj = trajectory.traj;
    self.feature_array=[];
    for index, frame in enumerate(self.traj):
      theframe = self.traj[index];
      if index % 1 == 0:
        refframe = pt.Frame(theframe);
        self.searchlist = trajectory.updatesearchlist(":MDL" , 18);
        self.traj.top.set_reference(theframe);
        self.traj.superpose(ref=refframe, mask="@CA")

      feature_frame = self.forward(self.traj[index]);
      self.feature_array.append(feature_frame);
    self.feature_array = np.array(self.feature_array);
    return self.feature_array;



