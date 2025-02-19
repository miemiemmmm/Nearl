import numpy as np
from scipy.spatial import KDTree

import nearl
from nearl.features import Feature
import nearl.io, nearl.featurizer

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



EXAMPLE_DATA = nearl.get_example_data("/tmp/nearl_test") 

# Initialize the trajectory loader and featurizer 
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


import h5py
with h5py.File("/tmp/rf_data.h5", "r") as hdf:
  x_train = hdf["rf_feature"][:]
  print(x_train.shape)
  