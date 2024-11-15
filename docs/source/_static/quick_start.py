# Start your first Nearl featurization
######################################

import nearl
import nearl.io, nearl.featurizer, nearl.features
from collections import OrderedDict

if __name__ == "__main__": 
  # Get the example data 
  EXAMPLE_DATA = nearl.get_example_data("/tmp/nearl_test")

  # Initialize the trajectory loader
  print(EXAMPLE_DATA["MINI_TRAJSET"])
  loader = nearl.io.TrajectoryLoader(EXAMPLE_DATA["MINI_TRAJSET"], mask="!:T3P")
  print(f"Loading {len(loader)} trajectories detected")  # 4 trajectories detected

  # Initialize the featurizer
  FEATURIZER_PARMS = {
    "dimensions": 32,       # Dimension of the 3D grid
    "lengths": 16,          # Length of the 3D grid in Angstrom, it yields 0.5 resolution
    "time_window": 10,      # Number of frames to slice each trajectory
    "sigma": 1.5,
    "cutoff": 3.5,
    "outfile": "/tmp/features.h5",
  }
  featurizer = nearl.featurizer.Featurizer(FEATURIZER_PARMS)


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

  # Register the trajectory loader, focused selection and start the main loop
  featurizer.register_trajloader(loader) 
  featurizer.register_focus([":LIG"], "mask") 
  featurizer.main_loop()
