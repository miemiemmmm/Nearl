Load customized trajectories
===============================

General concept
---------------

topology, 
coordinates, 
identity, 
are the major components of a trajectory.


Suppose a simple PDB file is given, the trajectory can be loaded with the following code:


.. code-block:: python

  from nearl.io import Trajectory

  class NewTrajectory(Trajectory):
    def __init__(self, path):
      self.path = path
      super().__init__(path)

Key attributes
^^^^^^^^^^^^^^

- top (pytraj.Topology): The topology of the trajectory.
- xyz (np.ndarray): The coordinates of the trajectory.
- identity (any type): could be manually defined


Case study: MISATO trajectory dataset
-------------------------------------

The original paper of `MISATO <https://doi.org/10.1101/2023.05.24.542082>`_ trajectory dataset: 
*Siebenmorgen, T., Menezes, F., Benassou, S., Merdivan, E., Kesselheim, S., Piraud, M., Theis, F.J., Sattler, M. and Popowicz, G.M., 2023. MISATO-Machine learning dataset of protein-ligand complexes for structure-based drug discovery. bioRxiv, pp.2023-05.* 

The MISATO data files are freely downloadable from `Zenodo <https://zenodo.org/records/7711953>`_. 
In the test case, the MISATO dataset is organized in the following structure:

.. code-block:: bash 

  $ tree /path/to/misato_dataset
  # Outputs: 
  /path/to/misato_dataset
  ├── MD.hdf5
  ├── parameter_restart_files_MD
  │   ├── 10gs
  │   │   ├── production.rst
  │   │   └── production.top.gz
  │   ├── 11gs
  │   │   ├── production.rst
  │   │   └── production.top.gz
  │   ├── 13gs
  │   │   ├── production.rst
  │   │   └── production.top.gz
  ......


Definition of the MISATO trajectory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``__init__`` method is expected to take the arguments that is able to guide the loading of the trajectory. 
In this case, a PDB code and the directory of the MISATO trajectory dataset are enough to correctly retrieve the trajectory information. 

- Topology file: ``f"{misatodir}/parameter_restart_files_MD/{pdbcode.lower()}/production.top.gz"``
- HDF trajectory dataset: ``f"{misatodir}/MD.hdf5"``
- Coordinates: ``f"/{pdbcode.upper()}/trajectory_coordinates"`` tag in the HDF file

The ``identity`` property is optional, but it is recommended to provide in the case that the identity of the trajectory is needed. 
In this case, the protein identity is needed to connect to the pK values as labels for the following training. 

.. code-block:: python

  from nearl.io import Trajectory
  class MisatoTraj(Trajectory): 
    def __init__(self, pdbcode, misatodir, **kwarg): 
      # Needs dbfile and parm_folder;
      self.topfile = f"{misatodir}/parameter_restart_files_MD/{pdbcode.lower()}/production.top.gz"
      if not os.path.exists(self.topfile):
        raise FileNotFoundError(f"The topology file of PDB {pdbcode} is not found ({self.topfile})")
      
      self.trajfile = os.path.join(misatodir, f"MD.hdf5")
      if not os.path.exists(self.trajfile):
        raise FileNotFoundError(f"The trajectory file is not found ({self.trajfile})")
      
      # NOTE: Get the PDB code in the standard format, lowercase and replace superceded PDB codes
      self.pdbcode = pdbcode

      top = pt.load_topology(self.topfile)
      # ! IMPORTANT: Remove water and ions to align the coordinates with the topology
      res = set([i.name for i in top.residues])
      if "WAT" in res:
        top.strip(":WAT")
      if "Cl-" in res:
        top.strip(":Cl-")
      if "Na+" in res:
        top.strip(":Na+")

      with h5py.File(self.trajfile, "r") as hdf:
        keys = hdf.keys()
        if pdbcode.upper() in keys:
          coord = hdf[f"/{pdbcode.upper()}/trajectory_coordinates"]
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
      
      # Pytraj trajectory-based initialization
      super().__init__(ret_traj)

    @property
    def identity(self):
      return utils.get_pdbcode(self.pdbcode)

To view the trajectory, try with the following commands in Jupyter Notebook: 

.. code-block:: python

  traj = MisatoTraj("10gs", "/path/to/misato")
  print(traj)
  traj.visualize()





.. TODO
.. Add the tutorial index when appropriate
.. Add script download link when appropriate