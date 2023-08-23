import time

import pytraj as pt
import numpy as np

from nearl import utils
from nearl import printit, _verbose


# Trajectory object
class Trajectory(pt.Trajectory):
  def __init__(self, trajfile=None, pdbfile=None, **kwarg):
    """
    Initialize the trajectory object from pytraj.Trajectory
    Add more customizable functions
    Args:
      trajfile: trajectory file
      pdbfile:  pdb file
      **kwarg:  keyword arguments
    """
    if _verbose:
      printit(f"Module {self.__class__.__name__}: Loading trajectory {trajfile} with topology {pdbfile}")
    st = time.perf_counter()
    # Set the keyword arguments for slicing/masking trajectory;
    stride = kwarg.get("stride", None)
    frame_indices = kwarg.get("frame_indices", None)
    mask = kwarg.get("mask", None)

    # NOTE: If both stride and frame_indices are given, stride will be respected;
    # NOTE: If none of stride or frame_indices are given, all frames will be loaded;
    if isinstance(trajfile, str) and isinstance(pdbfile, str):
      # Initialize the trajectory object
      tmptraj = pt.load(trajfile, pdbfile, stride=stride, frame_indices=frame_indices)
      timeinfo = tmptraj.time
      boxinfo = tmptraj._boxes
    elif isinstance(trajfile, pt.Trajectory):
      # Initialize the trajectory object
      tmptraj = trajfile
      timeinfo = tmptraj.time
      boxinfo = tmptraj._boxes
    elif trajfile is None and pdbfile is None:
      super().__init__()
      return

    # NOTE: Adding mask in the first pt.load function causes lose of time information
    if mask is not None:
      tmptraj = tmptraj[mask]
    top = tmptraj.top
    xyz = tmptraj.xyz

    # Set basic attributes for pytraj.Trajectory;
    super().__init__(xyz=xyz, top=top, velocity=tmptraj.velocities, force=tmptraj.forces)
    self._boxes = boxinfo
    self.time = timeinfo
    self._life_holder = tmptraj._life_holder
    self._frame_holder = tmptraj._frame_holder

    # Personalized attributes to facilitate further trajectory processing;
    self.top_filename = pdbfile
    self.traj_filename = trajfile
    self.mask = mask

    self._active_index = 0
    self._active_frame = self[0]

    if _verbose:
      printit(f"Module {self.__class__.__name__}: stride: {stride}; frame_indices: {frame_indices}; mask: {mask}")
      printit(f"Module {self.__class__.__name__}: Trajectory loaded in {time.perf_counter() - st:.2f} seconds")

  @property
  def active_frame(self):
    return self._active_index, self._active_frame

  @active_frame.setter
  def active_frame(self, index):
    try:
      self._active_index = index
      self._active_frame = self[index]
    except IndexError:
      self._active_index = 0
      self._active_frame = self[0]
      printit(f"Module {self.__class__.__name__}: Index {index} out of range ({len(self)}). Reset active to frame 0.")

  def copy_traj(self):
    xyzcopy = self.xyz.copy()
    topcopy = self.top.copy()
    thecopy = pt.Trajectory(xyz=xyzcopy, top=topcopy,
                            velocity=self.velocities.copy() if self.velocities else None,
                            force=self.forces.copy() if self.velocities else None)
    thecopy._boxes = self._boxes
    thecopy.time = self.time
    thecopy._life_holder = self._life_holder
    thecopy._frame_holder = self._frame_holder
    return thecopy

  def compute_closest_pairs_distance(self, mask, **kwarg):
    if "countermask" in kwarg.keys():
      countermask = kwarg["countermask"]
      pdist, pdist_info = utils.PairwiseDistance(self, f"{mask}&!@H=", f"{countermask}&!@H=", use_mean=True)
    else:
      pdist, pdist_info = utils.PairwiseDistance(self, f"{mask}&!@H=", f"{mask}<@6&!{mask}&@C,CA,CB,N,O",
                                                 use_mean=True)
    self.pdist = pdist
    self.pdist_info = pdist_info
    return pdist, pdist_info

  ############################################
  ############################################
  ############################################
  ############################################
  def cluster_pairwise(self, cluster_nr=10, **kwarg):
    clusters = cluster.ClusterAgglomerative(pdist, cluster_nr)
    cluster_rand = cluster.RandomPerCluster(clusters, number=1)
    self.frames = cluster_rand
    return self.frames

  def slicetraj(self, frameindices):
    frameindices = np.array(frameindices).astype(int)

  def cluster(self, method="", **kwarg):
    if len(method) == 0:
      self.cluster_pairwise(**kwarg)
    elif (method == "distance"):
      pass
    return self.frames

  def updatesearchlist(self, frame, mask, cutoff):
    refframe = pt.Frame(self.traj[frame])
    self.traj.top.set_reference(refframe)
    self.traj.superpose(ref=refframe, mask="@CA")
    self.searchlist = self.traj.top.select(f"{mask}<@{cutoff}")
    self.activeframe = frame
    return self.searchlist

  def test(self, coordinates, elements=[]):
    if len(elements) == 0:
      elements = ["H"] * len(coordinates)
    newframe = pt.Frame(self.traj[0])
    newtop = pt.Topology(self.traj.top)
    resids = [i.index for i in self.traj.top.residues]
    maxid = resids[-1]
    print(f"Before: {newframe.xyz.shape}")
    for i, c in enumerate(coordinates):
      therid = (maxid + i + 2)
      print(therid, c)
      theatom = pt.Atom(name='CL', charge=0.04, mass=17.0, resname="BOX", type="H", resid=therid)
      theres = pt.Residue(name='BOX', resid=therid)
      newtop.add_atom(theatom, theres)
      newframe.append_xyz(np.array([c]).astype(np.float64))
    print(f"After addition newframe {newframe.xyz.shape}, {newtop.n_atoms}")
    thexyz = np.array([newframe.xyz])
    newtraj = pt.Trajectory(xyz=thexyz, top=newtop)
    self.testtraj = newtraj
    pt.write_traj("/tmp/test.pdb", newtraj, overwrite=True)