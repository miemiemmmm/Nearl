from nearl.io.trajloader import TrajectoryLoader


def test_loading_options_filters_out_unknown_keys():
  loader = object.__new__(TrajectoryLoader)
  loader._TrajectoryLoader__loading_options = {"stride": None, "frame_indices": None, "mask": None, "superpose": False}
  loader.loading_options = {"stride": 2, "trajs": ["a.nc"], "tops": ["a.pdb"], "mask": ":LIG"}
  opts = loader._TrajectoryLoader__loading_options
  assert opts["stride"] == 2
  assert opts["mask"] == ":LIG"
  assert "trajs" not in opts
  assert "tops" not in opts
