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
- identity (any type): Could be manually defined. Useful for manual cooperation with other types e.g. labeling of frame-slices. 


Case study: MISATO trajectory dataset
-------------------------------------

File structure
^^^^^^^^^^^^^^

The `MISATO <https://doi.org/10.1038/s43588-024-00627-2>`_ dataset (*Siebenmorgen, Till, et al. Nat. Comput. Sci. (2024): 1-12.*), is a massive simulation over PDBBind general set, containing over 13,000 short trajectories. 
In this tutorial, we will use the ``MISATO`` dataset as a case study to demonstrate how to load a customized trajectory. 
The associated data files are hosted on `Zenodo <https://zenodo.org/records/7711953>`_. 
Let's assume that the ``MISATO`` dataset is stored in the directory ``/directory/of/misato_dataset`` and the organization of the topologies and trajectories are as follows: 

.. code-block:: bash 

  $ tree /directory/of/misato_dataset
  /directory/of/misato_dataset
  ├── MD.hdf5                           # The non-canonical HDF5 trajectory file
  ├── parameter_restart_files_MD
  │   ├── 10gs
  │   │   ├── production.rst
  │   │   └── production.top.gz         # AMBER-style topology file 
  │   ├── 11gs
  │   │   ├── production.rst
  │   │   └── production.top.gz
  ......

  # The structure of the HDF trajectory is as follows: 
  $ h5ls -r /directory/of/misato_dataset/MD.hdf5
  /                        Group        # Root group
  /10GS                    Group
  /10GS/atoms_element      Dataset {6593}
  /10GS/atoms_number       Dataset {6593}
  /10GS/atoms_residue      Dataset {6593}
  /10GS/atoms_type         Dataset {6593}
  /10GS/frames_bSASA       Dataset {100}
  /10GS/frames_distance    Dataset {100}
  /10GS/frames_interaction_energy Dataset {100}
  /10GS/frames_rmsd_ligand Dataset {100}
  /10GS/molecules_begin_atom_index Dataset {3}
  /10GS/trajectory_coordinates Dataset {100, 6593, 3}     # The trajectory of the PDB code 10gs 
  /11GS                    Group
  /11GS/atoms_element      Dataset {6600}
  /11GS/atoms_number       Dataset {6600}
  /11GS/atoms_residue      Dataset {6600}
  /11GS/atoms_type         Dataset {6600}
  /11GS/frames_bSASA       Dataset {100}
  /11GS/frames_distance    Dataset {100}
  /11GS/frames_interaction_energy Dataset {100}
  /11GS/frames_rmsd_ligand Dataset {100}
  /11GS/molecules_begin_atom_index Dataset {3}
  /11GS/trajectory_coordinates Dataset {100, 6600, 3}
  ......

Definition of trajectory
^^^^^^^^^^^^^^^^^^^^^^^^

With the PDB code (``pdbcode``) and the directory of the ``MISATO`` dataset (``misatodir``), we are able to correctly retrieve the trajectory information. 
The topology can be found in ``f"{misatodir}/parameter_restart_files_MD/{pdbcode.lower()}/production.top.gz"`` and the coordinates can be found in the ``f"/{pdbcode.upper()}/trajectory_coordinates"`` tag in the HDF trajectory file ``f"{misatodir}/MD.hdf5"``. 


The ``identity`` attribute is optional, but it is recommended to manually set in the case that the identity of the trajectory is needed. 
In this case, the protein identity is needed to connect to the pK values as labels for the following training. 

.. The ``__init__`` method is expected to take the arguments that is able to guide the loading of the trajectory. 

.. code-block:: python

  import os, h5py
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

In Jupyter Notebook, the following commands could be used to visualize the trajectory: 

.. code-block:: python

  traj = MisatoTraj("10gs", "/path/to/misato")
  print(traj)
  traj.traj.visualize()    # To visualize the trajectory 


Featurize MISATO trajectories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  import nearl.io, nearl.featurizer, nearl.features 

  misato_dir = "/directory/of/misato_dataset"
  pdbs = ['1gpk', '1h23', '1k1i', '1nc3', '1o3f', '1p1q', '1pxn', '1r5y', '1ydr', '2c3i',
          '2p4y', '2qbr', '2vkm', '2wn9', '2wvt', '2zcr', '3ag9', '3b1m', '3cj4', '3coz',
          '3dxg', '3fv2', '3gbb', '3gc5', '3gnw', '3gr2', '3n86', '3nq9', '3pww', '3pxf',
          '3qgy', '3ryj', '3u8n', '3uew', '3uex', '3uo4', '3wz8', '3zsx', '4cr9', '4crc',
          '4ddh', '4de3', '4e5w', '4e6q', '4gkm', '4jia', '4k77', '4mme', '4ogj', '4qac']
  trajlist = [(pdb, misato_dir) for pdb in pdbs]

  FEATURIZER_PARMS = {"dimensions": [32, 32, 32], "lengths": 20, "time_window": 10, "outfile": "/tmp/example.h5"} 
  loader = nearl.io.TrajectoryLoader(trajlist, trajtype=MisatoTraj, superpose=True, trajids = pdbs)
  feat  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
  feat.register_trajloader(loader)
  feat.register_focus([":MOL"], "mask")

  feat = nearl.features.Mass(selection="!:MOL", outkey="feat_static", cutoff=2.5, sigma=1.0)
  feat.register_feature(feat)
  print(len(feat.FEATURESPACE))
  feat.run(8)



.. note:: 

  :download:`Download Python source code for local execution <_static/tutorial2_customize_traj.py>` 



.. TODO
.. Add the tutorial index when appropriate
.. Add script download link when appropriate