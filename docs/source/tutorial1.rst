Feature customization
=====================

Feature engineering is a crucial step in molecular science, utilizing the domain expertise to extract meaningful features from raw data using certain data mining techniques. 
Properly defining and selecting these features can significantly enhance the performance of machine learning algorithms. 
In this tutorial, we will delve into the process of feature customization within Nearl's framework and present a case study on ``RFScore``.



General concept
---------------
When defining a new feature, you need to inherit the base class ``Features`` and implement the feature function. 
The ``cache`` method is designed to store the structural information (``self.cached_props``) that will be used during the following steps. 
It is designed to be called only once when processing each trajectory, unless the trajectory is reloaded from time to time. 
The returned variables from the ``query`` method are directly put to the ``run`` method. 
The following is a simple implementation of a feature. 

.. code-block:: python

  import numpy as np
  from nearl import commands, utils
  from nearl.features import Features
  
  class MyFirstFeature(Features): 
    def __init__(self, **kwargs):
      super().__init__(**kwargs)
      # Add your own initialization

    def cache(self, trajectory):
      super().cache(trajectory)
      # this example is to generate a random number for each atom
      self.cached_props = np.full(trajectory.top.n_atoms, 1)

    def query(self, topology, frames, focus): 
      # Retrieve the concerned substructure for further feature calculation 
      mask_inbox, coords = super().query(topology, frame, focus)
      weights = self.cached_props[mask_inbox]
      return coords, weights 

    def run(self, coords, weights): 
      # Transform the frame-slice into a feature vector 
      feature_vector = commands.voxelize_coords(coords, weights, self.dims, self.spacing, self.cutoff, self.sigma)
      return feature_vector

    def dump(self, feature_vector):
      # Put the output feature ``feature_vector`` into HDF file 
      utils.append_hdf_data(self.outfile, self.outkey, 
                            np.asarray([result], dtype=np.float32), 
                            dtype=np.float32, maxshape=(None, *self.dims), 
                            chunks=True, compression="gzip", compression_opts=4)


Case study: **RFScore** features
--------------------------------
The following code-block, we will customize the featurization process using the ``RFScore`` features (Ballester *et al.*). 
``RFScore`` is a 36-dimensional feature that each element represents the count of specific protein–ligand atom type pairs within a certain distance ``cutoff``. 
The original paper and the detailed description of ``RFScore`` is available at `Bioinformatics <https://doi.org/10.1093/bioinformatics/btq112>`_.


Feature implementation
^^^^^^^^^^^^^^^^^^^^^^

The selected components (``mask_interest``) are designated as the *ligand part* (moiety of interest) and their counterparts (``!mask_interest``) are considered as the *receptor part*. 
In the ``cache`` method, the indices of ligand and receptor parts are stored in ``self.idx_interest`` and ``self.idx_counterpart``. 
The resulting feature vector is a ``4×9`` matrix, where the rows represent the protein atoms (C, N, O, S) and the columns represent the ligand atoms (C, N, O, F, P, S, Cl, Br, I). 
Consequently, the hash map that maps all possible contacts to their internal indices if the resulting vector is stored in ``self.idx_hashmap``. 
Since applying a bounding box is unnecessary in this example, the ``query`` method directly returns the frame-slice (``frame_coords``). 
The ``run`` method then receives the output coordinates from the ``query`` method and computes the contact map based on the cached properties and the ``cutoff``. 

Note that ``RFScore`` is a static feature, and we use only the first frame of each frame-slice. 
Since this is not a voxel-based feature, the ``outshape`` for feature initialization is manually set to ``[1, 36]``.
Hydrogen atoms are included during the structure processing, although they do not affect the final count of contacts. 

.. code-block:: python

  import numpy as np
  from scipy.spatial import KDTree
  from nearl.features import Feature

  # Result is a 4*9 (36 dimensions) contact map
  # Rows (protein)   : C, N, O, S
  # Columns (ligand) : C, N, O, F, P, S, Cl, Br, I
  PROT_MAP = {6: 0, 7: 1, 8: 2, 16: 3}
  LIG_MAP = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}

  class RFScoreFeat(Feature): 
    def __init__(self, moiety_of_interest, cutoff, **kwargs):
      super().__init__(outshape = [1, 36], **kwargs)
      self.moi = moiety_of_interest
      self.cutoff = cutoff
    
    def cache(self, trajectory): 
      super().cache(trajectory)
      # Build the map of parts of interest and the counterpart 
      self.idx_interest = trajectory.top.select(self.moi)  # The indices of the moiety of interest
      self.idx_counterpart = np.setdiff1d(np.arange(trajectory.top.n_atoms), self.idx_interest)
      self.atom_numbers = np.array([i.atomic_number for i in trajectory.top.atoms], dtype=int)

      # Construct a hashmap for fast lookup of all possible contacts 
      self.idx_hashmap = {}
      for p, p_idx in PROT_MAP.items(): 
        for l, l_idx in LIG_MAP.items(): 
          self.idx_hashmap[f"{p}_{l}"] = (p_idx, l_idx) 

    def query(self, topology, frame_coords, focal_point): 
      return (frame_coords,)

    def run(self, coords): 
      # Build a kd-tree for the counterpart coordinates
      kd_tree = KDTree(coords[0][self.idx_counterpart]) 
      # Initialize the feature vector 
      rf_feature = np.zeros((4, 9), dtype=int)

      # Process atoms in the moiety of interest 
      for idx in self.idx_interest: 
        atom_number = self.atom_numbers[idx] 
        atom_crd = coords[0][idx] 
        inner_idxs = kd_tree.query_ball_point(atom_crd, self.cutoff) 
        counterpart_indices = self.idx_counterpart[inner_idxs]
        for idx_prot in counterpart_indices: 
          iidx = self.idx_hashmap.get(f"{self.atom_numbers[idx_prot]}_{atom_number}", None) 
          if iidx is not None: 
            rf_feature[iidx] += 1
      return rf_feature.reshape(-1)


Feature generation
^^^^^^^^^^^^^^^^^^
As in the :ref:`Quick Start <ref_quick_start>`, we will use the simple trajectory in the example dataset for demonstration. 

``RFScore`` feature will focus on the ligand part (annotated as ``:LIG``) and the cutoff distance is set to ``5.5 Å``. 
The resulting features will be dumped to the ``/tmp/rf_data.h5`` file and the resulting vectors will be stored in the ``rf_feature`` key in the HDF5 file. 

.. code-block:: python

  import nearl
  import nearl.featurizer, nearl.io
  # Initialize the trajectory loader and featurizer 
  EXAMPLE_DATA = nearl.get_example_data("/tmp/nearl_test") 
  loader = nearl.io.TrajectoryLoader(EXAMPLE_DATA["MINI_TRAJSET"])
  featurizer = nearl.featurizer.Featurizer({ 
    "time_window": 10,
    "outfile": "/tmp/rf_data.h5",
  })

  # Register the feature and start the featurization 
  featurizer.register_feature(RFScoreFeat(":LIG", 5.5, outkey="rf_feature")) 
  featurizer.register_trajloader(loader) 
  featurizer.register_focus([":LIG"], "mask") 
  featurizer.run() 


Result inspection
^^^^^^^^^^^^^^^^^
The following code snippet retrieves the feature vectors under ``rf_feature`` we just generated. 

.. code-block:: python
  
  import h5py
  with h5py.File("/tmp/rf_data.h5", "r") as hdf:
    x_train = hdf["rf_feature"][:]
    print(x_train.shape)


.. Using random forest regressor as am example with our implementation of ``RFScore`` features. 

.. note:: 

  :download:`Download Python source code for local execution <_static/tutorial1_feature_customize.py>` 


.. TODO
.. Add the tutorial index when appropriate
