Start your first featurization
==============================



.. TODO: Check other formats e.g. XTC and TRR

After installing Nearl, the following steps will guide you through the process of loading a trajectory, initializing a featurizer, registering features, and starting the featurization.
For the consistent import convention, all Nearl-related modules will be imported as shown in the following code: 

.. code-block:: python

  import nearl
  import nearl.data

You could follow the following code blocks one-by-one or download the :download:`script <_static/quick_start.py>` and directly run it locally.


Load trajectories
-----------------

The `pytraj.Trajectory <https://amber-md.github.io/pytraj/latest/index.html>`_ in `Pytraj <https://amber-md.github.io/pytraj/latest/trajectory.html>`_ is the backend for trajectory processing. 
Trajectory loader currently supports the following formats: **NetCDF**, **PDB**, **DCD**.
By default, the `Trajectory` object one or two arguments marking the trajectory and/or topology as the first and second arguments. 
In the distribution of Nearl, there is a small trajectory for testing purposes in ``nearl.data.MINI_TRAJ``. 
Here is a simple example of how to load the trajectory and visualize it in Jupyter notebook.

.. code-block:: python

  print(type(nearl.data.MINI_TRAJ))  #  <class 'tuple'>
  traj = nearl.Trajectory(nearl.data.MINI_TRAJ[0], nearl.data.MINI_TRAJ[1]) 
  traj.visualize() 



The :class:`TrajectoryLoader<nearl.io.trajloader.TrajectoryLoader>` could take arbitrary number of trajectories for the following featurization.
This can be done by providing a list of tuples, where each tuple contains the path to the trajectory and the topology file respectively. 
A trajectory list of tuples, which contains the correct way to initialize the `nearl.Trajectory` object. 
As a quick start, the following code initializes a simple  trajectory loader and repeatatively put the short demo trajectory in the trajectory loader.

.. code-block:: python

  trajs = [
    nearl.data.MINI_TRAJ, 
    nearl.data.MINI_TRAJ, 
    nearl.data.MINI_TRAJ, 
    nearl.data.MINI_TRAJ, 
  ]
  loader = nearl.TrajectoryLoader(trajs)
  print(f"{len(loader)} trajectories detected")  # 4 trajectories detected


.. tip:: 

  The input tuple only describes how to initialize the ``nearl.Trajectory`` object and does not load the trajectory into memory until it is needed. 
  This avoids loading all the trajectories into memory at once, which is useful when dealing with large trajectories or a large number of trajectories. 
  **Direct trajectory-based loader initialization is doable but strongly discouraged.** 

  .. code-block:: python

    traj_list = [
      (traj1, top1),
      (traj2, top2),
      (traj3, top3), 
      ..., 
      (trajn, topn)
    ]
    trajloader = nearl.TrajectoryLoader(traj_list, **kwarg)


.. See :class:`nearl.io.trajloader` for more details.
.. see :ref:`nearl.io.trajloader` here
.. see :class:`This is a class <nearl.features.DensityFlow>` here
.. see :func:`here <nearl.features.DensityFlow>` for feature i


Initialize a featurizer
-----------------------
Featurizer is the core component of Nearl which store the blueprint of the featurization, and is responsible for controlling the featurization process. 

(mainly features and trajectories)

The following code initializes a simple featurizer with the following parameters: 

.. _ref_quick_start_featurizer:

.. code-block:: python

  FEATURIZER_PARMS = {
    "dimensions": 32,       # Dimension of the 3D grid
    "lengths": 16,          # Length of the 3D grid in Angstrom, it yields 0.5 resolution
    "time_window": 10,      # Number of frames to slice each trajectory
    "sigma": 1.5,
    "cutoff": 3.5,
    "outfile": "/tmp/features.h5",
  }
  featurizer = nearl.featurizer.Featurizer(FEATURIZER_PARMS)


Register featurizers
--------------------
For more featurizer settings, check the class methods of :class:`nearl.featurizer.Featurizer`. 
The argument ``outkey`` should be defined separately for the dump of the features.

.. tip::
  There are 3 major ways to register one or more features into the featurizer.

  1. Register a list: Typical way to register features
  2. Register individually: Convenient when the number of features is small 
  3. Register an ordered dictionary: Useful when there are many similar features and the tag could be used to distinguish them

.. code-block:: python

  from collections import OrderedDict
  
  # Use a simple list of features
  features_list = [
    nearl.features.Aromaticity(selection=":LIG", outkey="arom_lig"),
    nearl.features.Aromaticity(selection=":LIG", outkey="arom_prot"),
  ]
  featurizer.register_features(features_list)

  # Register features individually
  featurizer.register_feature(nearl.features.Mass(selection=":LIG", outkey="lig_annotation"))
  featurizer.register_feature(nearl.features.Mass(selection="!:LIG,T3P", outkey="prot_annotation"))  # Append another feature

  # Use a dictionary of features
  feature_dict = OrderedDict()
  feature_dict["obs_density_lig"] = nearl.features.MarchingObservers(selection=":LIG", obs="density", agg="mean", weight_type="mass", outkey="obs_density_lig")
  feature_dict["obs_density_prot"] = nearl.features.MarchingObservers(selection="!(:LIG,T3P)", obs="density", agg="mean", weight_type="mass", outkey="obs_density_prot")
  feature_dict["df_mass_std_lig"] = nearl.features.DensityFlow(selection=":LIG", agg="standard_deviation", weight_type="mass", outkey="df_mass_std_lig")
  feature_dict["df_mass_std_prot"] = nearl.features.DensityFlow(selection="!(:LIG,T3P)", agg="standard_deviation", weight_type="mass", outkey="df_mass_std_prot")

  featurizer.register_features(feature_dict)


Start featurization
-------------------
After registering the features, trajectory loader and substructure of interest has to be registered before starting the featurization. 
All of the features will be put into the output file defined in ``FEATURIZER_PARMS["outfile"]``

.. code-block:: python

  featurizer.register_trajloader(loader)  # Register the trajectory loader in the first step
  featurizer.register_focus([":LIG"], "mask")  # focus on the ligand
  featurizer.main_loop()

Check output file
-----------------
The ``ncdump`` program (might need to install netcdf-bin) could be used to check the output file. 

.. code-block:: bash 

  $ ncdump -h /tmp/features.h5  # outfile defined in FEATURIZER_PARMS
  netcdf features {
  dimensions:
    phony_dim_0 = UNLIMITED ; // (40 currently)
    phony_dim_1 = 32 ;
    phony_dim_2 = 32 ;
    phony_dim_3 = 32 ;
  variables:
    float arom_lig(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
    float arom_prot(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
    float df_mass_std_lig(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
    float df_mass_std_prot(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
    float lig_annotation(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
    float obs_density_lig(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
    float obs_density_prot(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;
    float prot_annotation(phony_dim_0, phony_dim_1, phony_dim_2, phony_dim_3) ;

  group: featurizer_parms {
    variables:
      double cutoff ;
      int64 dimensions ;
      int64 lengths ;
      string outfile ;
      double sigma ;
      int64 time_window ;
    } // group featurizer_parms
  }

