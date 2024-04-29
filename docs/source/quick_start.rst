Start your first featurization
==============================

.. TODO: Check other formats e.g. XTC and TRR

Load trajectories
-----------------

The `pytraj.Trajectory <https://amber-md.github.io/pytraj/latest/index.html>`_ in `Pytraj <https://amber-md.github.io/pytraj/latest/trajectory.html>`_ is the backend for trajectory processing. 
Trajectory loader currently supports the following formats: **NetCDF**, **PDB**, **DCD**.
By default, the `Trajectory` object one or two arguments marking the trajectory and/or topology as the first and second arguments. 
In the distribution of Nearl, there is a small trajectory for testing purposes in ``nearl.data.MINI_TRAJ``. 
Here is a simple example of how to load the trajectory and visualize it in Jupyter notebook.

.. code-block:: python

  import nearl
  import nearl.data
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


See :class:`nearl.io.trajloader` for more details.

see :ref:`nearl.io.trajloader` here

see :class:`This is a class <nearl.features.DensityFlow>` here

see :func:`here <nearl.features.DensityFlow>` for feature i


Initialize a featurizer
-----------------------
Featurizer is the core component of Nearl which store the blueprint of the featurization, and is responsible for controlling the featurization process. 

(mainly features and trajectories)

The following code initializes a simple featurizer with the following parameters: 

.. code-block:: python

  import nearl
  FEATURIZER_PARMS = {
    "dimensions": 32,       # Dimension of the 3D grid
    "lengths": 16,          # Length of the 3D grid in Angstrom, it yields 0.5 resolution
    "time_window": 10,      # Number of frames to slice each trajectory 
    "outfile": "/tmp/features.h5", 
  }
  featurizer = nearl.featurizer.Featurizer(FEATURIZER_PARMS)


Register featurizers
--------------------
There are 3 major ways to register one or more features into the featurizer.

- via a list: Typical way to register features
- individually: Convenient when the number of features is small 
- via an ordered dictionary: Useful when there are many similar features and the tag could be used to distinguish them

.. code-block:: python

  from collections import OrderedDict
  
  # Use a simple list of features
  features_2 = [
    nearl.features.Mass(selection=":T3P"),
    nearl.features.PartialCharge(),
  ]
  featurizer.register_features(features_2)

  # Register features individually
  featurizer.register_feature(nearl.features.Mass(selection="!:LIG,T3P"))  # Append another feature

  # Use a dictionary of features
  features = OrderedDict()    
  features["mass"] = nearl.features.Mass(selection=":LIG")
  features["charge"] = nearl.features.Charge()
  featurizer.register_features(features)



Start featurization
-------------------
After registering the features, trajectory loader and substructure of interest has to be registered before starting the featurization. 
All of the features will be put into the output file defined in ``FEATURIZER_PARMS["outfile"]``

.. code-block:: python

  featurizer.register_focus([":LIG"], "mask")
  featurizer.featurize(loader)
  feat.main_loop()


