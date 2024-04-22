.. Tutorials
.. =========


RFScore featurization
---------------------
In this tutorial, we will see how to customize the featurization used in RFScore.


>>> from nearl.features import Feature
>>> from nearl.features import RFScore


Load non-canonical trajectories
-------------------------------

>>> from nearl.data import Trajectory
>>> class NewTrajectory(self, ):
...     def __init__(self, path):
...         self.path = path





.. Register a feature
.. ------------------
.. View the example project featurizing a small subset of the [PDBbind](http://www.pdbbind.org.cn/) dataset
.. [in this script](https://github.com/miemiemmmm/BetaPose/blob/master/scripts/prepare_pdbbind.py)


Register focused substructure
-----------------------------

# Register the area of interest

>>> feat.register_focus([":MOL"],  "mask")


Manual focus parser
-------------------

If the built-in focus parser does not meet your needs, you could define your own focus parser. 
The following is an example to read a standalone json file recording the atom indices of the structure of interest. 

# Initialize the proper shaped array to store the focal points

# There has to be three dimensions: number of frame slices, number of focal points, and 3

>>> def manual_focal_parser(traj): 
...     # The reference json file to map the indices to track
...     ligandmap = "/MieT5/BetaPose/data/misato_ligand_indices.json"
...     # The sliding time window has to match the one put in the featurizer
...     timewindow = 10
...     with open(ligandmap, "r") as f:
...         LIGAND_INDICE_MAP = json.load(f)
...         ligand_indices = np.array(LIGAND_INDICE_MAP[traj.identity.upper()])
...     FOCALPOINTS = np.full((traj.n_frames // timewindow, 1, 3), 99999, dtype=np.float32)
...     for i in range(traj.n_frames // timewindow):
...         FOCALPOINTS[i] = np.mean(traj.xyz[i*timewindow][ligand_indices], axis=0)
...     return FOCALPOINTS

To use the manually defined focal point parser, you need to register that function to the featurizer

>>> feat.register_focus(manual_focal_parser, "function")




Define your own feature
-----------------------

When defining a new feature, you need to inherit the base class Features and implement the feature function.


>>> from nearl import commands, utils
>>> from nearl.features import Features
>>> class MyFirstFeature(Features): 
...     def __init__(self, *args, **kwargs):
...         super().__init__(*args, **kwargs)
...         # your own initialization
...
...     def cache(self, trajectory):
...         # this example is to generate a random number for each atom
...         self.cached_props = np.random.rand(trajectory.n_atoms)
...
...     def query(self, topology, frames, focual_point):
...         frame = frames[0]
...         mask_inbox = super().query(topology, frame, focual_point)
...         coords = frame.xyz[mask_inbox]
...         weights = self.cached_props[mask_inbox]
...         return coords, weights
...
...     def run(self, coords, weights):
...         feature_vector = commands.voxelize_coords(coords, weights,  self.dims, self.spacing, self.cutoff, self.sigma )
...         return feature_vector
...
...     def dump(self, feature_vector):
...         # your own output
...         utils.append_hdf_data(self.outfile, self.outkey, 
...                               np.asarray([result], dtype=np.float32), 
...                               dtype=np.float32, maxshape=(None, *self.dims), 
...                               chunks=True, compression="gzip", compression_opts=4)

