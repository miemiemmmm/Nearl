import os
import nglview as nv
import numpy as np
import pytraj as pt
from ipywidgets import Box

display_config = {
  "*": "cartoon",
  "*": "ball+stick",
}


def random_color():
  r = lambda: np.random.randint(0,255)
  return '#%02X%02X%02X' % (r(),r(),r())


def nglview_mask(traj, mask):
  return "@"+",".join([str(i) for i in traj.top.select(mask)])


class TrajectoryViewer: 
  def __init__(self, thetraj, top=None):
    if hasattr(thetraj, "xyz"):
      self.traj = thetraj
    elif (isinstance(thetraj, str) and os.path.isfile(thetraj)):
      self.traj = pt.load(thetraj, top=top)
    self.viewer = nv.show_pytraj(self.traj)
    self.background_color = "#f9fcfd"
    self.center_indices = None
    self.auto_focus = True
    self.config_display(display_config)
    self.resize_stage()

  def center(self, selection):
    self.auto_focus = False
    if selection is str:
      self.center_indices = nglview_mask(self.traj, selection)
    elif selection is (list, np.ndarray):
      self.center_indices = selection
  @property
  def bg(self):
    return self.background_color

  @bg.setter
  def bg(self, color):
    self.background_color = color

  def config_display(self, viewer_dic):
    self.viewer[0].clear_representations()
    if self.auto_focus:
      self.center(list(viewer_dic.keys())[0])

    # Add representations to the representation stack
    for rep in viewer_dic:
      mask = nglview_mask(self.traj, rep)
      self.viewer[0].add_representation(viewer_dic[rep], selection=mask)

    # Center the stage
    if self.center_indices is not None:
      self.viewer.center(selection = self.center_indices)
    else:
      self.viewer.center(selection = mask)
    self.viewer.stage.set_parameters(backgroundColor = self.background_color)
    return self.viewer

  def resize_stage(self, width = 800, height = 800):
    box = Box([self.viewer])
    box.layout.width = f"{width}px"
    box.layout.height = f"{height}px"
    self.viewer._view_height = f"{height}px"
    self.viewer._view_width = f"{width}px"
    return box

  def add_caps(self, partner1, partner2):
    for i, j in zip(partner1, partner2):
      self.viewer[0].add_representation("distance", atomPair=[[f"@{i}", f"@{j}"]],
                                   color="blue", label_color="black", label_fontsize=0)
    return self.viewer


if __name__ == "__main__":
  # Example usage of the TrajectoryViewer class
  import sys

  if len(sys.argv) == 2:
    traj = pt.load(sys.argv[1])
  elif len(sys.argv) == 3:
    traj = pt.load(sys.argv[1], sys.argv[2])
  else:
    print("Usage: python view.py traj.pdb traj.nc")
    sys.exit(1)
  viewer = TrajectoryViewer(traj)
  viewer.config_display(display_config)
  viewer.add_distance([0,1,2], [3,4,5])
  viewer.viewer
