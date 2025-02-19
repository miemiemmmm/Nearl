Load customized trajectories
===============================

This tutorial demonstrates how to load custom trajectories using the `MISATO <https://doi.org/10.1038/s43588-024-00627-2>`_ dataset as an example. 
You will learn to handle non-standard topology/coordinate formats and integrate trajectories into machine learning workflows.


General concepts
----------------

A canonical trajectory consists of three core components:

1. Topology: Definition of the system
2. Coordinates: Evolving 3D positions of atoms over time
3. Identity: Optional metadata if fine-grained control (*e.g.* labeling frame-slices) is needed

A simple trajectory could be loaded using the following code. `Pytraj <https://amber-md.github.io/pytraj/latest/index.html>`_ is used as the backend for trajectory handling.
They are stored in the following class attributes:

- top (pytraj.Topology)
- xyz (np.ndarray)
- identity (any type): Relies on manually definition and requirement 


.. code-block:: python

  from nearl.io import Trajectory

  traj = Trajectory("path/to/trajectory.nc", top="path/to/topology.pdb")
  print(traj.top)         # pytraj.Topology:  21375 atoms, 6474 residues, 6324 mols
  print(traj.xyz.shape)   # numpy.ndarray: (1001, 21375, 3) -> 1001 frames, 21375 atoms
  print(traj.identity)    # str: path/to/trajectory.nc -> trajectory file path by default

The definition of a **customized trajectory** class follows the logic that customizable part loads the trajectory, coordinates, potentially set identity and then pass the information to the parent class.

.. code-block:: python

  from nearl.io import Trajectory

  class NewTrajectory(Trajectory):
    def __init__(self, arg1, arg2, ..., **kwargs):
      traj = load_trajectory(arg1, arg2, ...)  # Dummy function to guide the loading of the trajectory
      super().__init__(traj, **kwargs)


Case study: MISATO trajectory dataset
-------------------------------------

File structure
^^^^^^^^^^^^^^

The `MISATO <https://doi.org/10.1038/s43588-024-00627-2>`_ dataset (*Siebenmorgen, Till, et al. Nat. Comput. Sci. (2024): 1-12.*) contains over 13,000 short MD trajectories of protein-ligand complexes in the `PDBBind <http://www.pdbbind.org.cn/>`_ dataset. 
The associated trajectories are hosted on `Zenodo <https://zenodo.org/records/7711953>`_. 
Assuming the MISATO dataset is downloaded into the folder ``/directory/of/misato_dataset``, files are organized as: 

.. code-block:: bash 

  /directory/of/misato_dataset
  ├── MD.hdf5                           # The non-canonical HDF5 trajectory file
  ├── parameter_restart_files_MD
  │   ├── 10gs
  │   │   ├── production.rst
  │   │   └── production.top.gz         # AMBER-style topology 
  ......

  # The structure of the HDF trajectory  
  $ h5ls -r /directory/of/misato_dataset/MD.hdf5
  /                        Group          # Root of the HDF5 file
  /10GS                    Group          # The trajectory of the PDB code 10gs 
  /10GS/atoms_element      Dataset {6593}
  /10GS/atoms_number       Dataset {6593}
  /10GS/atoms_residue      Dataset {6593}
  /10GS/atoms_type         Dataset {6593}
  /10GS/frames_bSASA       Dataset {100}
  /10GS/frames_distance    Dataset {100}
  /10GS/frames_interaction_energy Dataset {100}
  /10GS/frames_rmsd_ligand Dataset {100}
  /10GS/molecules_begin_atom_index Dataset {3}
  /10GS/trajectory_coordinates Dataset {100, 6593, 3}     # Evolving coordinates 
  ......

Definition of trajectory
^^^^^^^^^^^^^^^^^^^^^^^^

To correctly retrieve the trajectory, only the PDB code (``pdbcode``) and the directory of the MISATO dataset (``misatodir``) are needed. 

- Topology: ``<misatodir>/parameter_restart_files_MD/<pdbcode>/production.top.gz``
- Evolving coordinate: ``<misatodir>/MD.hdf5`` >> ``/<pdbcode>/trajectory_coordinates`` tag
- Identity: PDB code in lowercase format for finding the binding constant pK in the PDBBind dataset


.. code-block:: python

  import os  
  import h5py                # Read the HDF trajectory file
  import numpy as np
  import pytraj as pt
  from nearl.io import Trajectory
  
  class MisatoTraj(Trajectory): 
    def __init__(self, pdbcode, misatodir, **kwarg): 
      # Locate the topology and trajectory files based on the directory of MISATO dataset 
      self.topfile = f"{misatodir}/parameter_restart_files_MD/{pdbcode.lower()}/production.top.gz"
      self.trajfile = os.path.join(misatodir, f"MD.hdf5")

      # IMPORTANT: Original topolgy contains water and ions 
      # IMPORTANT: Remove them to align the coordinates with the topology 
      top = pt.load_topology(self.topfile)
      top.strip(":WAT")
      try: top.strip(":Cl-") 
      except: pass
      try: top.strip(":Na+")
      except: pass

      with h5py.File(self.trajfile, "r") as hdf:
        if pdbcode.upper() in hdf.keys():
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

      # NOTE: Get the PDB code in the standard format, lowercase and replace superceded PDB codes
      self.pdbcode = pdbcode.lower()
      self.traj = ret_traj
      pt.superpose(ret_traj, mask="@CA")
      
      # Initialization the Trajectory object with Pytraj trajectory 
      super().__init__(ret_traj)

    @property
    def identity(self):
      return self.pdbcode

In Jupyter notebook, `NGLview <https://github.com/nglviewer/nglview>`_ could visualize the trajectory as follows:   

.. code-block:: python

  traj = MisatoTraj("10gs", "/path/to/misato")
  print(traj)
  traj.traj.visualize()    # To visualize the trajectory 


Featurize MISATO trajectories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Similar to tutorial 1, the following code converts the 


.. code-block:: python

  import nearl.io, nearl.featurizer, nearl.features 

  misato_dir = "/directory/of/misato_dataset"
  pdbs = ['1gpk', '1h23', ..., '4qac']
  trajlist = [(pdb, misato_dir) for pdb in pdbs]     # List of tuples for the trajectory and misato directory

  # Explicitly set the trajectory type to be the customized MisatoTraj
  loader = nearl.io.TrajectoryLoader(trajlist, trajtype=MisatoTraj, superpose=True, trajids = pdbs) 

  # Initialize featurizer, register the trajectory loader and focus on the ligand 
  FEATURIZER_PARMS = {"dimensions": [32, 32, 32], "lengths": 20, "time_window": 10, "outfile": "/tmp/example.h5"} 
  feat  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
  feat.register_trajloader(loader)
  feat.register_focus([":MOL"], "mask")

  # Register the feature and run the featurization
  feature = nearl.features.Mass(selection="!:MOL", outkey="feat_static", cutoff=2.5, sigma=1.0)
  feat.register_feature(feature)
  print(len(feat.FEATURESPACE))
  feat.run(8)



.. note:: 

  :download:`Download Python source code for local execution <_static/tutorial2_customize_traj.py>` 

