import pytest

from nearl.io.trajloader import TrajectoryLoader

DUMMY_TRAJS = [("a.nc", "a.pdb"), ("b.nc", "b.pdb")]


class _DummyTraj:
  """Stand-in trajtype so loader tests don't need real trajectory files on disk."""
  def __init__(self, traj_file, top_file, **kwargs):
    self.traj_file = traj_file
    self.top_file = top_file
    self.kwargs = kwargs


def test_loading_options_filters_out_unknown_keys():
  loader = object.__new__(TrajectoryLoader)
  loader._TrajectoryLoader__loading_options = {"stride": None, "frame_indices": None, "mask": None, "superpose": False}
  loader.loading_options = {"stride": 2, "trajs": ["a.nc"], "tops": ["a.pdb"], "mask": ":LIG"}
  opts = loader._TrajectoryLoader__loading_options
  assert opts["stride"] == 2
  assert opts["mask"] == ":LIG"
  assert "trajs" not in opts
  assert "tops" not in opts


def test_matching_strides_length_is_accepted():
  loader = TrajectoryLoader(DUMMY_TRAJS, strides=[1, 2])
  assert loader.strides == [1, 2]


def test_mismatched_strides_length_raises_value_error():
  with pytest.raises(ValueError):
    TrajectoryLoader(DUMMY_TRAJS, strides=[1, 2, 3])


def test_matching_masks_length_is_accepted():
  loader = TrajectoryLoader(DUMMY_TRAJS, masks=[":LIG", "!:LIG"])
  assert loader.masks == [":LIG", "!:LIG"]


def test_mismatched_masks_length_raises_value_error():
  with pytest.raises(ValueError):
    TrajectoryLoader(DUMMY_TRAJS, masks=[":LIG"])


def test_non_iterable_trajs_raises_value_error():
  with pytest.raises(ValueError):
    TrajectoryLoader(42)


def test_trajids_length_mismatch_raises_value_error():
  with pytest.raises(ValueError):
    TrajectoryLoader(DUMMY_TRAJS, trajids=["only_one"])


def test_iteration_yields_trajectories_in_order():
  loader = TrajectoryLoader(DUMMY_TRAJS, trajtype=_DummyTraj)
  results = list(loader)
  assert [r.traj_file for r in results] == ["a.nc", "b.nc"]


def test_getitem_int_returns_single_trajectory():
  loader = TrajectoryLoader(DUMMY_TRAJS, trajtype=_DummyTraj)
  result = loader[1]
  assert result.traj_file == "b.nc"


def test_getitem_list_returns_selected_trajectories_in_order():
  loader = TrajectoryLoader(DUMMY_TRAJS, trajtype=_DummyTraj)
  result = loader[[1, 0]]
  assert [r.traj_file for r in result] == ["b.nc", "a.nc"]


def test_getitem_slice_returns_trajectory_range():
  loader = TrajectoryLoader(DUMMY_TRAJS, trajtype=_DummyTraj)
  result = loader[0:1]
  assert [r.traj_file for r in result] == ["a.nc"]
