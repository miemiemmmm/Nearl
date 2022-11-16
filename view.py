import nglview as nv 
import pytraj as pt


def rand_color():
  r = lambda: random.randint(0,255)
  return '#%02X%02X%02X' % (r(),r(),r())
def nglmask(traj, mask):
  return "@"+",".join([str(i) for i in traj.top.select(mask)])
def show_complex(traj, viewer_dic):
  self.viewer = nv.show_pytraj(traj)
  self.viewer[0].clear_representations()
  for i in viewer_dic:
    mask = nglmask(traj, i)
    self.viewer[0].add_representation(viewer_dic[i], selection=mask)
  viewer.center(selection=mask)
  viewer.stage.set_parameters(backgroundColor="#f9fcfd")
  return viewer

def add_distance(viewer, partner1, partner2):
  for i, j in zip(partner1, partner2):
    viewer[0].add_representation("distance", atomPair=[[f"@{i}", f"@{j}"]], 
                                 color="blue", label_color="black", label_fontsize=0); 
  return viewer

class TrajectoryViewer: 
  def __init__(self, traj):
    self.traj = traj; 
    self.viewer = nv.show_pytraj(traj); 
    
  def show_complex(self, viewer_dic):
    self.viewer[0].clear_representations(); 
    for rep in viewer_dic:
      mask = nglmask(self.traj, rep); 
      self.viewer[0].add_representation(viewer_dic[rep], selection=mask); 
    self.viewer.center(selection = mask); 
    self.viewer.stage.set_parameters(backgroundColor="#f9fcfd"); 
    return self.viewer
  def add_distance(self, partner1, partner2):
    for i, j in zip(partner1, partner2):
      self.viewer[0].add_representation("distance", atomPair=[[f"@{i}", f"@{j}"]],
                                   color="blue", label_color="black", label_fontsize=0);
    return self.viewer


