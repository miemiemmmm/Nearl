Batch PDB structure loading
==============================
Since meaningful large-scale MD trajectories are not always easily accessible, 
Nearl provides a way to use static structures for the generation of various pre-defined static features. 
In this tutorial, we will go through the steps to use the static structures (like PDBBind dataset) for the generation of static features. 



Load PDB structures
-------------------

In the package, there is 100 protein-ligand complexes from PDBBind dataset for demonstration purposes. 
The PDB paths are stored in the `nearl.data.MINI_SET` variable.
In Nearl, each static structure are treated as a trajectory with only one frame. 
If this is the case, you only needs to load that these PDB files for trajectory initialization. 

.. code-block:: python

  import nearl
  import nearl.data

  # Make up the list of tuple with the PDB files
  trajs = [(i,) for i in nearl.data.MINI_SET]
  loader = nearl.TrajectoryLoader(trajs)

  # The protein and ligands are combined and ligands are named as LIG
  for traj in loader:
    print(traj.top.select(":LIG"))


Initialize the featurizer
-------------------------

The following code use the featurizer parameters like in the :ref:`Quick Start <ref_quick_start_featurizer>` section. 
The major difference between this example and the previous one is that the `time_window` parameter is set to 1. 
This correctly leads the featurizer to treat the static structures as a trajectory with only one frame.

.. code-block:: python

  FEATURIZER_PARMS = {
    "dimensions": 32,       # Dimension of the 3D grid
    "lengths": 16,          # Length of the 3D grid in Angstrom, it yields 0.5 resolution
    "time_window": 1,       # !! Number of frames has to be 1 when using static structures
    "sigma": 1.5,
    "cutoff": 3.5,
    "outfile": "/tmp/features.h5",
  }
  featurizer = nearl.featurizer.Featurizer(FEATURIZER_PARMS)


Register features and run
-------------------------


.. code-block:: python

  # Register two simple features
  featurizer.register_feature(nearl.features.Mass(selection=":LIG", outkey="lig_annotation"))
  featurizer.register_feature(nearl.features.Mass(selection="!:LIG,T3P", outkey="prot_annotation"))  # Append another feature

  # Register the trajectory loader, focus and run the featurization
  featurizer.register_trajloader(loader)  # Register the trajectory loader in the first step
  featurizer.register_focus([":LIG"], "mask")  # focus on the ligand
  featurizer.main_loop()
  

.. Result visualization
.. --------------------

.. .. code-block:: bash 

..   pass


.. note::

  :download:`Download Python source code for local execution <_static/static_pdb_featurization.py>`

.. TODO
.. Add the tutorial index when appropriate
.. Add script download link when appropriate
