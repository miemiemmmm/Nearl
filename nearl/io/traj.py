import os, sys

import h5py
import numpy as np
import pytraj as pt

from .. import utils
from .. import printit, config


__all__ = [
  "Trajectory",
  "MisatoTraj",
]


class Trajectory(pt.Trajectory):
  """
  This class represents the base-class for trajectory handling in the Nearl package.
  This class inherits from pytraj.Trajectory. 

  Attributes
  ----------
  top_filename : str
    The topology file of the trajectory
  traj_filename : str
    The trajectory file of the trajectory
  mask : str
    The mask to select the atoms to be saved
  mask_indices : np.ndarray
    The indices of the atoms selected by the mask
  atoms : np.ndarray
    The per-atom index of the trajectory
  residues : np.ndarray
    The per-residue index of the trajectory
  
  Methods
  -------
  identity()
    Return the identity of the trajectory
  copy_traj()
    Return a copy of the trajectory object
  make_index()
    Prepare the per-atom/per-residue index for the further trajectory processing
  write_frame()
    Save the trajectory to the file
  add_dummy_points()
    Add additional points to a frame for visual inspection  

  Notes
  -----
  Three types of trajectory initialization are supported:

  1. File-based trajectory initialization (traj_src and pdb_src are strings)
  2. Pytraj-based trajectory initialization (traj_src is pytraj.Trajectory)
  3. Self-based trajectory initialization (traj_src is self)

  Examples
  --------
  >>> from nearl.io import Trajectory
  >>> traj = Trajectory("traj.nc", "top.pdb")
  
  """
  def __init__(self, traj_src = None, pdb_src = None, **kwarg):
    """
    Initialize the trajectory object with the trajectory and topology files

    Parameters
    ----------
    traj_src : trajectory_like
      The trajectory like or filename to be loaded
    pdb_src : topology_like
      The topology like or filename to be loaded
    """
    # Set the keyword arguments for slicing/masking trajectory;
    stride = kwarg.get("stride", None)
    frame_indices = kwarg.get("frame_indices", None)
    mask = kwarg.get("mask", None)

    if config.verbose():
      printit(f"{self.__class__.__name__}: Loading trajectory {traj_src} with topology {pdb_src}")
      printit(f"{self.__class__.__name__}: stride: {stride}; frame_indices: {frame_indices}; mask: {mask}")

    # NOTE: If both stride and frame_indices are given, stride will be respected;
    # NOTE: If none of stride or frame_indices are given, all frames will be loaded;
    if isinstance(traj_src, str) and isinstance(pdb_src, str):
      # File name-based trajectory initialization
      tmptraj = pt.load(traj_src, pdb_src, stride=stride, frame_indices=frame_indices, mask=mask)
      timeinfo = tmptraj.time
      boxinfo = tmptraj._boxes

    elif isinstance(traj_src, str) and (pdb_src is None):
      # In the case that the trajectory is self-consistent e.g. PDB file
      tmptraj = pt.load(traj_src, mask=mask)
      timeinfo = tmptraj.time
      boxinfo = tmptraj._boxes

    elif isinstance(traj_src, (pt.Trajectory, self.__class__)):
      # Pytraj or self-based trajectory initialization
      if mask is not None: 
        tmptraj = traj_src[mask]
      else:
        tmptraj = traj_src
      timeinfo = tmptraj.time
      boxinfo = tmptraj._boxes

    elif (traj_src is None) and (pdb_src is None):
      # Initialize an empty object
      super().__init__() 
      return 
    
    else:
      raise ValueError(f"Invalid input for traj source ({type(traj_src)}) and pdb_src ({type(pdb_src)})")

    # NOTE: Adding mask in the first pt.load function causes lose of time information
    # if mask is not None:
    #   tmptraj = tmptraj[mask]
    top = tmptraj.top
    xyz = tmptraj.xyz
    printit(tmptraj.xyz.shape)
    assert tmptraj.top.n_atoms == tmptraj.xyz.shape[1], f"The number of atoms in the topology and the coordinates should be the same, rather than {tmptraj.top.n_atoms} and {tmptraj.xyz.shape[1]}"

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
    if "identity" in kwarg.keys():
      self.identity_ = kwarg["identity"]
    else:
      self.identity_ = None

    print("ideneity is ", self.identity, file=sys.stderr)

    # Prepare the per-atom/per-residue index for the further trajectory processing;
    self.atoms = None
    self.residues = None
    self.make_index()

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
  
  @property
  def identity(self):
    """
    Return the identity of the trajectory used for metadata retrieval. 

    Returns
    -------
    str
      By default, it returns the trajectory file name
    """
    if self.identity_ is not None: 
      return self.identity_
    else: 
      return self.traj_filename

  def copy_traj(self):
    """
    Return a copy of the trajectory object
    """
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

  def write_frame(self, frames, outfile="", mask="", **kwarg): 
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
    kwarg : dict
      Additional keyword arguments for the pytraj.save function

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
      pt.save(outfile, tmp_traj, overwrite=True, **kwarg)

  def add_dummy_points(self, coordinates, elements = None, frame_idx=0, outfile=""):
    """
    Add additional points to a frame for visual inspection

    Parameters
    ----------
    coordinates : list
      The list of coordinates to be added
    elements : str or list
      The list of element symbols to be added
    frame_idx : int
      The frame index to add the dummy points
    outfile : str
      The file name to save the new trajectory
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
  

class MisatoTraj(Trajectory): 
  """
  Built-in implementation of the Misato HDF5 trajectory. 

  Original paper:

  Siebenmorgen, T., Menezes, F., Benassou, S., Merdivan, E., Kesselheim, S., Piraud, M., Theis, F.J., Sattler, M. and Popowicz, G.M., 2023. MISATO-Machine learning dataset of protein-ligand complexes for structure-based drug discovery. bioRxiv, pp.2023-05.

  Attributes
  ----------
  pdbcode : str
    The PDB code as the identity of the trajectory 
  topfile : str
    The topology file of the trajectory
  trajfile : str
    The trajectory file of the trajectory

  Notes
  -----
  The trajectory type has to be manually defined when pushing to the trajectory loader (see example). 
  Solvents and ions are stripped for the alignment of the coordinates with the topology. 

  This module uses relative path to the `misatodir` to retrieve the trajectory. The following files are required to load the trajectory:

  1. The topology file ({misatodir}/parameter_restart_files_MD/{pdbcode}/production.top.gz)
  2. The trajectory file ({misatodir}/MD.hdf5)
  

  .. tip::
    Since there is no explicit annotation for the ligand part, we use a ligand indices map to 
    extract the ligand part of the protein. 


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
  

  """
  def __init__(self, pdbcode, misatodir, **kwarg): 
    """
    Initialize the MisatoTraj object with the PDB code and the Misato directory. 

    Parameters
    ----------
    pdbcode : str
      The PDB code of the trajectory to be loaded
    misatodir : str
      The directory of the Misato MD simulation output

    Notes
    -----
    The trajectory is firstly read as a pytraj.Trajectory object and then converted to the Nearl Trajectory object.

    """
    # Needs dbfile and parm_folder;
    self.topfile = f"{misatodir}/parameter_restart_files_MD/{pdbcode.lower()}/production.top.gz"
    if not os.path.exists(self.topfile):
      raise FileNotFoundError(f"The topology file of PDB {pdbcode} is not found ({self.topfile})")
    
    self.trajfile = os.path.join(misatodir, f"MD.hdf5")
    if not os.path.exists(self.trajfile):
      raise FileNotFoundError(f"The trajectory file is not found ({self.trajfile})")
    
    # NOTE: Get the PDB code in the standard format, lowercase and replace superceded PDB codes
    self.pdbcode = pdbcode
    if config.verbose():
      printit(f"{self.__class__.__name__}: Loading trajectory {pdbcode} with topology {self.topfile}")

    top = pt.load_topology(self.topfile)
    # ! IMPORTANT: Remove water and ions to align the coordinates with the topology
    res = set([i.name for i in top.residues])
    if "WAT" in res:
      top.strip(":WAT")
    if "Cl-" in res:
      top.strip(":Cl-")
    if "Na+" in res:
      top.strip(":Na+")

    if config.verbose():
      printit(f"{self.__class__.__name__}: Topology loaded with {top.n_atoms} atoms")

    with h5py.File(self.trajfile, "r") as hdf:
      keys = hdf.keys()
      if pdbcode.upper() in keys:
        coord = hdf[f"/{pdbcode.upper()}/trajectory_coordinates"]
        if config.verbose():
          printit(f"{self.__class__.__name__}: Trajectory loaded with {coord.shape[0]} frames and {coord.shape[1]} atoms")
        # Parse frames (Only one from stride and frame_indices will take effect) and masks
        if "stride" in kwarg.keys() and kwarg["stride"] is not None:
          slice_frame = np.s_[::int(kwarg["stride"])]
        elif "frame_indices" in kwarg.keys() and kwarg["frame_indices"] is not None:
          slice_frame = np.s_[kwarg["frame_indices"]]
        else: 
          slice_frame = np.s_[:]
        if "mask" in kwarg.keys() and kwarg["mask"] is not None:
          slice_atom = np.s_[top.select(kwarg["mask"])]
          top = top[slice_atom]
        else: 
          slice_atom = np.s_[:]
        ret_traj = pt.Trajectory(xyz=coord[slice_frame, slice_atom, :], top=top)
      else:
        raise ValueError(f"Not found the key for PDB code {pdbcode.upper()} in the HDF5 trajectory file.")

    if kwarg.get("superpose", False): 
      if kwarg.get("mask", None) is not None:
        printit(f"{self.__class__.__name__}: Superpose the trajectory with mask {kwarg['mask']}")
        pt.superpose(ret_traj, mask="@CA")
      else:
        printit(f"{self.__class__.__name__}: Superpose the trajectory with default mask @CA")
        pt.superpose(ret_traj, mask="@CA")
    
    
    printit(ret_traj.xyz.shape)   # DEBUG
    assert ret_traj.xyz.shape.__len__() == 3 , f"What? Shape of the trajectory is {ret_traj.xyz.shape}"
    printit("Result traj: ", ret_traj)

    # Pytraj trajectory-based initialization
    super().__init__(ret_traj)

  @property
  def identity(self):
    """
    Return the PDB code as the identity of the trajectory

    Returns
    -------
    str
      The PDB code of the trajectory
    """
    return utils.get_pdbcode(self.pdbcode)

