.. _ref_quick_start:

Start your first featurization
==============================

.. TODO: Check other formats e.g. XTC and TRR

After installing Nearl, the following steps will guide you through the key stages of feature extraction, including: 
  - Construct a trajectory loader
  - Build a featurizer 
  - Initialize and register features to the featurizer  
  - Start the featurization process

Run the following code to automatically download the example data used in this documentation. 
You can set the target folder (e.g. ``/tmp/nearl_test``) to any writable directory on your system. 

.. code-block:: python

  import nearl
  EXAMPLE_DATA = nearl.get_example_data("/tmp/nearl_test") 
  print(EXAMPLE_DATA.keys()) 
  # Outputs: dict_keys(['MINI_TRAJSET', 'MINI_PDBSET', 'PDBBIND_REFINED', 'PDBBIND_GENERAL'])

.. _ref_quick_start_trajloader: 

Load trajectories
-----------------

The Nearl trajectory loader supports commonly used MD trajectory formats, such as **NetCDF**, **XTC**, and **DCD** (also **PDB**, though multi-model PDB is space-inefficient). 
It uses `PyTraj <https://amber-md.github.io/pytraj/latest/_api/pytraj.trajectory.html>`_ as the backend for trajectory handling.  
For more details about supported trajectory formats, refer to the `CPPTRAJ documentation <https://amberhub.chpc.utah.edu/cpptraj/trajectory-file-commands/>`_. 
The default :class:`Trajectory <nearl.io.traj.Trajectory>` object requires the correct trajectory and topology arguments for loading. 
The ``MINI_TRAJSET`` keyword from the ``EXAMPLE_DATA`` contains a minimal set of trajectories and topologies to get started.
Here’s how to load a trajectory and visualize it in a Jupyter Notebook. 

.. code-block:: python

  import nearl.io
  print(type(EXAMPLE_DATA["MINI_TRAJSET"])) 
  # Outputs: <class 'list'> -> a list of tuples 

  traj = nearl.io.Trajectory(*EXAMPLE_DATA["MINI_TRAJSET"][0])  
  traj.visualize()    # Display the trajectory 


The :class:`TrajectoryLoader<nearl.io.trajloader.TrajectoryLoader>` is the major component for batch processing of large-scale MD datasets and allows you to register multiple trajectories. 
It accepts a list of tuples, where each tuple specifies the paths to a trajectory file and its corresponding topology file. 
Each tuple should contain the correct sequence to initialize the :class:`Trajectory <nearl.io.traj.Trajectory>` object properly. 
Below is an example of initializing a simple trajectory loader: 

.. code-block:: python

  loader = nearl.io.TrajectoryLoader(EXAMPLE_DATA["MINI_TRAJSET"])
  print(f"{len(loader)} trajectories detected") 


.. tip:: 

  The ``TrajectoryLoader`` does not immediately load trajectories upon initialization. 
  Instead, the list of tuples (input arguments for :class:`Trajectory <nearl.io.traj.Trajectory>` instantiation) is registered and trajectories are loaded only when needed
  This behavior is useful when dealing with large datasets, as it avoids overloading memory. 
  While it is possible to directly use a list of trajectories for trajectory loader initialization, it is not recommended unless you fully understand the implications. 

  .. code-block:: python

    traj_list = [
      (traj1, top1),
      (traj2, top2),
      (traj3, top3), 
      ..., 
      (trajn, topn)
    ]
    trajloader = nearl.TrajectoryLoader(traj_list)


.. Some examples of link to API document
.. See :class:`nearl.io.trajloader` for more details.
.. see :ref:`nearl.io.trajloader` here
.. see :class:`This is a class <nearl.features.DensityFlow>` here
.. see :func:`here <nearl.features.DensityFlow>` for feature i


.. _ref_quick_start_featurizer:

Initialize a featurizer
-----------------------

Featurizer is the core component to control the featurization process, namely coordinate the information flow between trajectories and features. 
The following code initializes a simple featurizer with the following parameters: 

.. code-block:: python

  FEATURIZER_PARMS = {
    "dimensions": 32,       # Dimension of the 3D grid 
    "lengths": 16,          # Length of the 3D grid in Angstrom, it yields 0.5 resolution 
    "time_window": 10,      # Number of frames to slice each trajectory 
    "sigma": 1.5, 
    "cutoff": 3.5, 
    "outfile": "/tmp/features.h5",   # Output structured HDF file 
  }
  featurizer = nearl.featurizer.Featurizer(FEATURIZER_PARMS)

For more featurizer configurations, check the class methods of :class:`nearl.featurizer.Featurizer`. 

.. _ref_quick_start_features:

Register features
-----------------
The following code demonstrates the 3 major ways to register one or more features into the featurizer. 
All resulting features will be put to the ``FEATURIZER_PARMS["outfile"]`` to align the features in the same HDF5 file. 
The argument ``outkey`` for each individual feauture should be defined separately because it is used to identify the feature tag when supplying the desired feature during training. 

1. **Register a list of features:** Typical way to register features

.. code-block:: python
  
  # Use a simple list of features
  features_list = [
    nearl.features.Aromaticity(selection=":LIG", outkey="arom_lig"),
    nearl.features.Aromaticity(selection=":LIG", outkey="arom_prot"),
  ]
  featurizer.register_features(features_list)

2. **Register features individually:** Convenient when the number of features is small

.. code-block:: python

  # Register features individually
  featurizer.register_feature(nearl.features.Mass(selection=":LIG", outkey="lig_annotation"))
  featurizer.register_feature(nearl.features.Mass(selection="!:LIG,T3P", outkey="prot_annotation"))  # Append another feature


3. **Register via ordered dictionary:** Useful when there are many similar features and setting tags helps the readability of the code

.. code-block:: python

  from collections import OrderedDict
  # Use a dictionary of features
  feature_dict = OrderedDict()
  feature_dict["obs_density_lig"] = nearl.features.MarchingObservers(selection=":LIG", obs="density", agg="mean", weight_type="mass", outkey="obs_density_lig")
  feature_dict["obs_density_prot"] = nearl.features.MarchingObservers(selection="!(:LIG,T3P)", obs="density", agg="mean", weight_type="mass", outkey="obs_density_prot")
  feature_dict["df_mass_std_lig"] = nearl.features.DensityFlow(selection=":LIG", agg="standard_deviation", weight_type="mass", outkey="df_mass_std_lig")
  feature_dict["df_mass_std_prot"] = nearl.features.DensityFlow(selection="!(:LIG,T3P)", agg="standard_deviation", weight_type="mass", outkey="df_mass_std_prot")
  featurizer.register_features(feature_dict)


.. _ref_quick_start_featurization:

Start featurization
-------------------
After registering the features, trajectory loader and substructure of interest has to be registered before starting the featurization. 


.. code-block:: python

  # Register the trajectory loader in the first step
  featurizer.register_trajloader(loader) 
  # focus on the protein-ligand binding site 
  featurizer.register_focus([":LIG"], "mask")  
  featurizer.main_loop()


.. _ref_quick_start_viewoutput:

Check output file
-----------------
If the featurization process is successful, all features are stored in the HDF5 output file defined by ``FEATURIZER_PARMS["outfile"]``. 
The ``ncdump`` program (requires the installation of `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_) is a simple tool to check the output. 
Running the command `ncdump -h /tmp/features.h5``, you should see the following output: 

.. code-block:: bash 

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


.. note:: 

  :download:`Download Python source code for local execution <_static/quick_start.py>` 
