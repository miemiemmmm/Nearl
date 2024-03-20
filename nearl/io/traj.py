import os, tempfile

import h5py
import pytraj as pt
import numpy as np

from .. import utils
from .. import printit, _verbose


__all__ = [
  "Trajectory",
  "MisatoTraj",
]

# Trajectory object
class Trajectory(pt.Trajectory):
  def __init__(self, traj_src = None, pdb_src = None, **kwarg):
    """
    This class inherits from pytraj.Trajectory for adding more customizable functions. 
    It could take the trajectory and topology as file names or pytraj.Trajectory for initialization.

    Parameters
    ----------
    traj_src : trajectory_like
      The trajectory like or filename to be loaded
    pdb_src : topology_like
      The topology like or filename to be loaded


    Examples
    --------
    >>> from nearl.io import Trajectory
    >>> traj = Trajectory("traj.nc", "top.pdb")
    """
    # Set the keyword arguments for slicing/masking trajectory;
    stride = kwarg.get("stride", None)
    frame_indices = kwarg.get("frame_indices", None)
    mask = kwarg.get("mask", None)

    if _verbose:
      printit(f"{self.__class__.__name__}: Loading trajectory {traj_src} with topology {pdb_src}")
      printit(f"{self.__class__.__name__}: stride: {stride}; frame_indices: {frame_indices}; mask: {mask}")

    # NOTE: If both stride and frame_indices are given, stride will be respected;
    # NOTE: If none of stride or frame_indices are given, all frames will be loaded;
    if isinstance(traj_src, str) and isinstance(pdb_src, str):
      # Initialize the trajectory object
      tmptraj = pt.load(traj_src, pdb_src, stride=stride, frame_indices=frame_indices)
      timeinfo = tmptraj.time
      boxinfo = tmptraj._boxes
    elif isinstance(traj_src, (pt.Trajectory, self.__class__)):
      # Initialize the trajectory object
      tmptraj = traj_src
      timeinfo = tmptraj.time
      boxinfo = tmptraj._boxes
    elif (traj_src is None) and (pdb_src is None):
      super().__init__()
      return
    elif isinstance(traj_src, str) and (pdb_src is None):
      # In the case that the trajectory is self-contained
      tmptraj = pt.load(traj_src)
      timeinfo = tmptraj.time
      boxinfo = tmptraj._boxes
    else:
      printit(type(traj_src), type(pdb_src))
      raise ValueError("Invalid input for traj_src and pdb_src")

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

    # Non-pytraj attributes to facilitate further trajectory processing;
    self.top_filename = pdb_src
    self.traj_filename = traj_src
    self.mask = mask

    # Prepare the per-atom/per-residue index for the further trajectory processing;
    self.atoms = None
    self.residues = None
    self.make_index()
    self.cached = {}

  def __getitem__(self, index):
    # Get the return from its parent pt.Trajectory;
    self._life_holder = super().__getitem__(index)
    if isinstance(self._life_holder, pt.Frame):
      pass
    else:
      self._life_holder.top_filename = self.top_filename
      self._life_holder.traj_filename = self.traj_filename
      self._life_holder.mask = self.mask
      self._life_holder.make_index()
    return self._life_holder
  

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

  def make_index(self):
    """
    Prepare the per-atom/per-residue index for the further trajectory processing;
    """
    self.atoms = np.array([i for i in self.top.atoms])
    self.residues = np.array([i for i in self.top.residues])

  def write_frame(self, frames, outfile="", mask=""): 
    """
    Save the trajectory to the file

    Parameters
    ----------
    frames : int, list_list or slice
      The frame indices to be saved
    outfile : str
      The file name to save the trajectory
    mask : str
      The mask to select the atoms to be saved

    Examples
    --------
    >>> from nearl.io import Trajectory
    >>> traj = Trajectory("traj.nc", "top.pdb")
    >>> traj.write_frame(0, "frame0.pdb", mask=":1-10")
    """
    if isinstance(frames, int):
      # Write one frame to the file
      frame_coords = np.array([self.xyz[frames]])
    elif isinstance(frames, (list, tuple)):
      tmp_indices = np.array(frames, dtype=int)
      frame_coords = self.xyz[tmp_indices]
    elif isinstance(frames, (slice, np.ndarray)):
      tmp_indices = np.arange(self.n_frames)[frames]
      frame_coords = self.xyz[tmp_indices]
    
    if len(mask) > 0: 
      atom_sel = self.top.select(mask)
      frame_coords = frame_coords[:, atom_sel, :]
      top = self.top[atom_sel]
    else:
      top = self.top
    
    if len(outfile) > 0:

      tmp_traj = pt.Trajectory(xyz=frame_coords, top=top)
      pt.save(outfile, tmp_traj, overwrite=True)

  def add_dummy_points(self, coordinates, elements = None, frame_idx=0, outfile=""):
    """
    Add additional points to a frame for visual inspection
    """
    if elements is None:
      elements = ["H"] * len(coordinates)
    elif isinstance(elements, str):
      elements = [elements] * len(coordinates)
    elif isinstance(elements, (list, tuple)):
      assert len(elements) == len(coordinates), "The length of elements should be the same as the coordinates"
      elements = list(elements)
    
    newframe = pt.Frame(self[frame_idx])
    newtop = pt.Topology(self.top)
    if self.residues is not None:
      maxid = self.residues[-1].index
    else: 
      maxid = [i.index for i in self.traj.top.residues][-1]

    print(f"Before: {newframe.xyz.shape}")
    for i, c in enumerate(coordinates):
      therid = (maxid + i + 2)
      theatom = pt.Atom(name='CL', charge=0.04, mass=17.0, resname="BOX", type="H", resid=therid)
      theres = pt.Residue(name='BOX', resid=therid, chainID=2)
      newtop.add_atom(theatom, theres)
      newframe.append_xyz(np.array([c]).astype(np.float64))
    print(f"After addition newframe {newframe.xyz.shape}, {newtop.n_atoms}")
    thexyz = np.array([newframe.xyz])

    # Write the new trajectory to the file
    if len(outfile) > 0:
      newtraj = pt.Trajectory(xyz=thexyz, top=newtop)
      pt.save(outfile, newtraj, overwrite=True)

  def compute_closest_pairs_distance(self, mask, **kwarg):
    if "countermask" in kwarg.keys():
      countermask = kwarg["countermask"]
      pdist, pdist_info = utils.dist_caps(self, f"{mask}&!@H=", f"{countermask}&!@H=", use_mean=True)
    else:
      pdist, pdist_info = utils.dist_caps(self, f"{mask}&!@H=", f"{mask}<@6&!{mask}&@C,CA,CB,N,O",
                                                 use_mean=True)
    self.pdist = pdist
    self.pdist_info = pdist_info
    return pdist, pdist_info

  def cache_rdkit(self, mask="", **kwarg):
    """
    Convert the trajectory to RDKit molecule object
    """
    
    from rdkit import Chem
    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmpfile:
      self.write_frame(0, outfile=tmpfile.name, mask=mask)
      mol = Chem.MolFromPDBFile(tmpfile.name, sanitize=False, removeHs=False)
      mol = utils.sanitize_bond(mol)
    return mol


    # rdmol = utils.traj_to_rdkit(self, "*", )
    # if rdmol is not None:
    #   self.rdmol = rdmol
    # else: 
    #   self.rdmol = None    
  ############################################
  ############################################
  # def cluster_pairwise(self, cluster_nr=10, **kwarg):
  #   clusters = cluster.ClusterAgglomerative(pdist, cluster_nr)
  #   cluster_rand = cluster.RandomPerCluster(clusters, number=1)
  #   self.frames = cluster_rand
  #   return self.frames

  # def cluster(self, method="", **kwarg):
  #   if len(method) == 0:
  #     self.cluster_pairwise(**kwarg)
  #   elif (method == "distance"):
  #     pass
  #   return self.frames




class MisatoTraj(Trajectory): 
  """
  Takes the Misato HDF5 trajectory to initialize the Trajectory object compatible with the Nearl package

  Parameters
  ----------
  pdbcode : str
    The PDB code of the trajectory to be loaded
  misatodir : str
    The directory of the Misato MD simulation output

  Notes
  -----
  Before pushing the trajectory list to the trajectory loader, define the corresponding trajectory type (see example). 

  This module uses relative path to the `misatodir` to retrieve the trajectory. 

  Due to the fact that the trajectory stored in the HDF does not contain the time information, solvents and ions are stripped
  for the alignment of the coordinates with the topology. 

  Examples
  --------
  >>> from nearl.io import TrajectoryLoader, MisatoTraj
  >>> trajs = [('5WIJ', '/MieT5/DataSets/misato_database/'), 
       ('4ZX0', '/MieT5/DataSets/misato_database/'), 
       ('3EOV', '/MieT5/DataSets/misato_database/'), 
       ('4K6W', '/MieT5/DataSets/misato_database/'), 
       ('1KTI', '/MieT5/DataSets/misato_database/')
      ]
  >>> trajloader = TrajectoryLoader(trajs, trajtype=MisatoTraj)
  >>> for i in trajloader: print(i.xyz.shape)

  """
  def __init__(self, pdbcode, misatodir, **kwarg): 
    # Needs dbfile and parm_folder;
    topfile = f"{misatodir}/parameter_restart_files_MD/{pdbcode.lower()}/production.top.gz"
    if not os.path.exists(topfile):
      # print(f"Error: The topology file of PDB {pdbcode} is not found", file=sys.stderr)
      raise FileNotFoundError(f"The topology file of PDB {pdbcode} is not found ({topfile})")

    top = pt.load_topology(topfile)
    # ! IMPORTANT: Remove water and ions to align the coordinates with the topology
    res = set([i.name for i in top.residues])
    if "WAT" in res:
      top.strip(":WAT")
    if "Cl-" in res:
      top.strip(":Cl-")
    if "Na+" in res:
      top.strip(":Na+")

    with h5py.File(f"{misatodir}/MD.hdf5", "r") as hdf:
      keys = hdf.keys()
      if pdbcode.upper() in keys:
        coord = hdf[f"/{pdbcode.upper()}/trajectory_coordinates"]
        # Parse frames (Only one from stride and frame_indices will take effect) and masks
        if "stride" in kwarg.keys():
          slice_frame = np.s_[::int(kwarg["stride"])]
        elif "frame_indices" in kwarg.keys():
          slice_frame = np.s_[kwarg["frame_indices"]]
        else: 
          slice_frame = np.s_[:]
        if "mask" in kwarg.keys():
          slice_atom = np.s_[top.select(kwarg["mask"])]
          top = top[slice_atom]
        else: 
          slice_atom = np.s_[:]

        ret_traj = pt.Trajectory(xyz=coord[slice_frame, slice_atom, :], top=top)
      else:
        raise ValueError(f"Not found the key for PDB code {pdbcode.upper()} in the HDF5 trajectory file.")
    super().__init__(ret_traj)

