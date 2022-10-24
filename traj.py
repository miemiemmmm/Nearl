import pytraj as pt 
import numpy as np 
from BetaPose import utils, cluster

class TRAJ:
  def __init__(self, trajfile, pdbfile):
    print("Generate a new traj")
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
    print(cluster_rand)

  def cluster_rmsd(self):
    self.traj = self.traj;

  def cluster(self, method="", **kwarg):
    print(kwarg.keys())
    print(type(kwarg))
    if len(method) == 0:
      self.cluster_pairwise(**kwarg);
    elif (method == "distance"):
      pass


########################################################
class TRAJ:
  def __init__(self, trajfile, pdbfile):
    print("Generate a new traj")
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
    print(cluster_rand)

  def cluster_rmsd(self):
    self.traj = self.traj;

  def cluster(self, method="", **kwarg):
    print(kwarg.keys())
    print(type(kwarg))
    if len(method) == 0:
      self.cluster_pairwise(**kwarg);
    elif (method == "distance"):
      pass
