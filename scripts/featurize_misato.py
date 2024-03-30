import time, json, os, argparse
import numpy as np
import pandas as pd

import nearl 
import nearl.data
from nearl.io.traj import MisatoTraj, Trajectory

# Define the way to generation labels for the misato trajectories


class MisatoLabels(nearl.features.Label_PCDT):
  def __init__(self, baseline_map, **kwargs):
    """
    Since there is no explicit annotation for the ligand part, we use a ligand indices map to 
    extract the ligand part of the protein.
    """
    super().__init__(outshape=(None,), **kwargs)
    self.baseline_table = pd.read_csv(baseline_map, header=0, delimiter=",")

  def search_baseline(self, pdbcode):
    pdbcode = nearl.utils.get_pdbcode(pdbcode)
    if pdbcode.lower() in self.baseline_table["pdbcode"].values:
      return self.baseline_table.loc[self.baseline_table["pdbcode"] == pdbcode.lower()]["pK"].values[0]
    else:
      raise ValueError(f"Cannot find the baseline value for {pdbcode}")
  
  def cache(self, traj):
    """
    Loop up the baseline values from a table and cache the pairwise closest distances. 
    """
    super().cache(traj)
    nearl.printit(f"{self.__class__.__name__}: Base value is {self.base_value}")


def parser(): 
  parser = argparse.ArgumentParser(description="Featurize the misato trajectories")
  parser.add_argument("--task_nr", type=int, default=1, help="The task number to run")
  parser.add_argument("--task_index", type=int, default=0, help="The task index to run")
  return parser.parse_args()

def get_trajlist(training_set, misatodir): 
  with open(training_set, "r") as f:
    pdbcodes = f.read().strip("\n").split("\n")
  trajlists = [(i, misatodir) for i in pdbcodes]
  return trajlists


if __name__ == "__main__":
  nearl.update_config(
    verbose = False, 
    debug = False,
    # verbose = True, 
    # debug = True,
  )

  args = parser()
  args = vars(args)
  print(args)
  task_nr = args.get("task_nr")
  task_index = args.get("task_index")


  # Load trajectories 
  misatodir = "/Matter/misato_database/"
  training_set = "/MieT5/BetaPose/data/misato_train.txt"
  outputfile = f"/Matter/nearl_training_data/test_jobs/testoutput{task_index}.h5"

  # Initialize featurizer object and register necessary components
  FEATURIZER_PARMS = {
    "dimensions": 32, 
    "lengths": 16, 
    "time_window": 10, 

    # For default setting inference of registered features
    "sigma": 2.0, 
    "cutoff": 2.0, 
    "outfile": outputfile, 

    # Other options
    "progressbar": False, 
  }


  trajlists = get_trajlist(training_set, misatodir)
  trajlists = np.array_split(trajlists, args.get("task_nr"))[args.get("task_index")]
  print(f"# Total number of trajectories {trajlists.__len__()}")

  # trajlists = trajlists[247:248]
  loader = nearl.io.TrajectoryLoader(trajlists, trajtype=MisatoTraj, superpose=True)
  print(f"Performing the featurization on {len(loader)} trajectories")

  feat  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)

  feat.register_trajloader(loader)
  feat.register_focus([":MOL"], "mask")
  # feat.register_focus(manual_focal_parser, "function")

  #############################################################################
  # Initialize required features
  feat_mass = nearl.features.Mass(outkey="mass")

  # Using the default charge from the trajectory topology 
  feat_pcp = nearl.features.PartialCharge(outkey="partial_charge_positive", keep_sign="p")
  feat_pcn = nearl.features.PartialCharge(outkey="partial_charge_negative", keep_sign="n")


  # !!! Change the keyword to different names to avoid conflict
  feat_mo1 = nearl.features.MarchingObservers(
    weight_type="mass", obs="distinct_count", 
    agg = "mean", 
    outkey="mass_distinct_count"
  )

  feat_mo2 = nearl.features.MarchingObservers(
    weight_type="mass", obs="density", 
    agg = "mean", 
    outkey="mass_density"
  )

  feat_mo3 = nearl.features.MarchingObservers(
    weight_type="mass", obs="existence", 
    agg = "standard_deviation", 
    outkey="mass_density"
  )

  feat_label = MisatoLabels(
    selection=":MOL", 
    baseline_map=nearl.data.GENERAL_SET, 
    outkey="label"
  )


  # feat.register_features([mass_feat, pc_feat, marchingobs_feat, label_feat])
  feat.register_features([feat_mass, feat_pcp, feat_pcn, feat_mo1, feat_mo2, feat_mo3,  feat_label])
  
  feat.main_loop(20)

