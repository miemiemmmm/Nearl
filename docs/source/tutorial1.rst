1. Feature customization
========================

Feature engineering is a crucial step in machine learning. 
It is the process of using domain knowledge to extract features from raw data via data mining techniques. 
These features can be used to improve the performance of machine learning algorithms. 
In this tutorial, we will go through the process of feature customization in Nearl's framework and do a case study on RFScore.




General concept
---------------

When defining a new feature, you need to inherit the base class Features and implement the feature function.

.. code-block:: python

  from nearl import commands, utils
  from nearl.features import Features
  
  class MyFirstFeature(Features): 
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      # your own initialization

    def cache(self, trajectory):
      # this example is to generate a random number for each atom
      self.cached_props = np.random.rand(trajectory.n_atoms)

    def query(self, topology, frames, focual_point):
      frame = frames[0]
      mask_inbox = super().query(topology, frame, focual_point)
      coords = frame.xyz[mask_inbox]
      weights = self.cached_props[mask_inbox]
      return coords, weights

    def run(self, coords, weights):
      feature_vector = commands.voxelize_coords(coords, weights,  self.dims, self.spacing, self.cutoff, self.sigma )
      return feature_vector

    def dump(self, feature_vector):
      # your own output
      utils.append_hdf_data(self.outfile, self.outkey, 
                            np.asarray([result], dtype=np.float32), 
                            dtype=np.float32, maxshape=(None, *self.dims), 
                            chunks=True, compression="gzip", compression_opts=4)




.. _RFScore: https://doi.org/10.1093/bioinformatics/btq112

Case study: RFScore features
----------------------------

The orignal paper of `RFScore`_ is as follows:  
Ballester, P.J. and Mitchell, J.B., 2010. A machine learning approach to predicting protein–ligand binding affinity with applications to molecular docking. Bioinformatics, 26(9), pp.1169-1175.


Feature implementation
^^^^^^^^^^^^^^^^^^^^^^

In this tutorial, we will see how to customize the featurization used in RFScore.

.. code-block:: python

  from nearl.features import Feature
  from nearl.features import RFScore




Model training
^^^^^^^^^^^^^^

.. code-block:: python
  
  import h5py
  from sklearn.ensemble import RandomForestRegressor

  rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)

  with h5py.File("rf_data.h5", "r") as f:
    x_train = f["features"][:]
    y_train = f["labels"][:]
  
  rf_regressor.fit(x_train, y_train)










