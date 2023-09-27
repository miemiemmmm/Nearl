import os, copy, re

import open3d as o3d
import nglview as nv
import numpy as np
import pytraj as pt
from ipywidgets import Box

from nearl.features import features

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


# TODO: HERE are the primary migration from features.fingerprint module to utils.view module
# TODO: Not yet test if the code work correctly
def displayfiles(plyfiles, add=[]):
  """
  Display a list of ply files (trangle mesh) in the same window
  """
  objs = []
  finalobj = None
  for obji, plyfile in enumerate(plyfiles):
    color = features.SEGMENT_CMAPS[obji]
    mesh = o3d.io.read_triangle_mesh(plyfile)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    objs.append(mesh)
    if obji == 0:
      finalobj = mesh
    else:
      finalobj += mesh
  display(objs, add=add)
  return objs


def display(objects, add=[]):
  """
  Display a list of objects in the same window
  Args:
    objects: list of open3d.geometry.TriangleMesh
    add: list of additional objects for accessary
  """
  if len(objects) == 0 and len(add)==0:
    return []
  else:
    objs = copy.deepcopy(objects)
    for i in range(1, len(objs)):
      color = features.SEGMENT_CMAPS[i]
      objs[i].paint_uniform_color(color)
      if isinstance(objs[i], o3d.geometry.TriangleMesh):
        objs[i].compute_vertex_normals()
    o3d.visualization.draw_geometries(add+objs, width=1200, height=1000)


def display_registration(source, target, transformation):
  """
  Apply the transformation metrix to the source point cloud and display it with the target point cloud
  Args:
    source: open3d.geometry.PointCloud
    target: open3d.geometry.PointCloud
    transformation: transformation matrix, np.array sized (4,4)
  """
  source_temp = copy.deepcopy(source)
  target_temp = copy.deepcopy(target)
  source_temp.paint_uniform_color(features.SEGMENT_CMAPS[1])
  target_temp.paint_uniform_color(features.SEGMENT_CMAPS[-1])
  source_temp.transform(transformation)
  display([source_temp, target_temp])


def display_convex(obj, n_points=600):
  pcd, hulls = features.compute_convex(obj, n_points=n_points)
  display([pcd, hulls])


def voxelize(obj, show=True):
  if isinstance(obj, o3d.geometry.TriangleMesh):
    pcd = obj.sample_points_uniformly(600)
  elif isinstance(obj, o3d.geometry.PointCloud):
    pcd = obj.voxel_down_sample(0.01)
  else:
    print(f"Please provide a o3d.geometry.TriangleMesh or o3d.geometry.PointCloud object rather than {type(obj)}")
    return False
  pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(600, 3)))
  # fit to unit cube
  pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
          center=pcd.get_center())
  voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
  if show:
    display([voxel_grid])
  return voxel_grid


def NewCuboid(center=[0,0,0], length=6):
  """
  Accessory function to create a cuboid formed by 8 points and 12 lines
  Args:
    center: center of the cuboid
    length: length of the cuboid
  """
  points = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
  ])
  points = points * length
  points = points + np.array(center) - (length/2)
  lines = np.array([
    [0, 1], [0, 2], [1, 3], [2, 3],
    [4, 5], [4, 6], [5, 7], [6, 7],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ])
  colors = [[0, 0, 0] for i in range(len(lines))]
  line_set = o3d.geometry.LineSet(
    points = o3d.utility.Vector3dVector(points),
    lines = o3d.utility.Vector2iVector(lines),
  )
  line_set.colors = o3d.utility.Vector3dVector(colors)
  return line_set


def NewCoordFrame(center=[0,0,0], scale=1):
  """
  Accessory function to create a coordinate frame
  Args:
    center: center of the coordinate frame
    scale: scale of the coordinate frame
  """
  coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
  coord_frame.scale(scale=scale, center=center)
  coord_frame.translate(center, relative=False)
  return coord_frame


ATOM_PATTERNS = {0: '^[0-9]*H.*$', 1: '^[0-9]*D.*$', 2: '^O.*$', 3: '^CA$', 4: '^CD$', 5: '^CD  $', 6: '^CA$',
                 7: '^N$', 8: '^CA$', 9: '^C$', 10: '^O$', 11: '^P$', 12: '^CB$', 13: '^CB$', 14: '^CB$', 15: '^CG$',
                 16: '^CG$', 17: '^CG$', 18: '^CG$', 19: '^O1$', 20: '^O2$', 21: '^CH3$', 22: '^CD$', 23: '^NE$',
                 24: '^RE$', 25: '^CZ$', 26: '^NH[12][AB]?$', 27: '^RH[12][AB]?$', 28: '^OD1$', 29: '^ND2$',
                 30: '^AD1$', 31: '^AD2$', 32: '^OD[12][AB]?$', 33: '^ED[12][AB]?$', 34: '^OD1[AB]?$', 35: '^ND2$',
                 36: '^AD1$', 37: '^AD2$', 38: '^OD2$', 39: '^LP[12]$', 40: '^SG$', 41: '^SG$', 42: '^OE[12][AB]?$',
                 43: '^EE[12][AB]?$', 44: '^CD$', 45: '^OE1$', 46: '^NE2$', 47: '^AE[12]$', 48: '^CE1|CD2$',
                 49: '^ND1$', 50: '^ND1$', 51: '^RD1$', 52: '^NE2$', 53: '^RE2$', 54: '^NE2$', 55: '^RE2$',
                 56: '^A[DE][12]$', 57: '^CG1$', 58: '^CG2$', 59: '^CD|CD1$', 60: '^CD1$', 61: '^CD2$',
                 62: '^C[GDE]$', 63: '^NZ$', 64: '^KZ$', 65: '^SD$', 66: '^CE$', 67: '^C[DE][12]$', 68: '^CZ$',
                 69: '^C[GD]$', 70: '^SE$', 71: '^SEG$', 72: '^OD1$', 73: '^OD2$', 74: '^OG$', 75: '^OG1$',
                 76: '^CG2$', 77: '^CD1$', 78: '^CD2$', 79: '^CE2$', 80: '^NE1$', 81: '^CE3$', 82: '^CZ2$',
                 83: '^CZ3$', 84: '^CH2$', 85: '^C[DE][12]$', 86: '^CZ$', 87: '^OH$', 88: '^CG1$', 89: '^CG2$',
                 90: '^CD$', 91: '^CE$', 92: '^FE[1-7]$', 93: '^S[1-7]$', 94: '^OXO$', 95: '^FE1$', 96: '^FE2$',
                 97: '^O1$', 98: '^O2$', 99: '^FE$', 100: '^CH[A-D]$', 101: '^N[A-D]$', 102: '^N [A-D]$',
                 103: '^C[1-4][A-D]$', 104: '^CM[A-D]$', 105: '^C[AB][AD]$', 106: '^CG[AD]$', 107: '^O[12][AD]$',
                 108: '^C[AB][BC]$', 109: '^OH2$', 110: '^N[123]$', 111: '^C1$', 112: '^C2$', 113: '^C3$',
                 114: '^C4$', 115: '^C5$', 116: '^C6$', 117: '^O7$', 118: '^O8$', 119: '^S$', 120: '^O[1234]$',
                 121: '^O[1234]$', 122: '^O4$', 123: '^P1$', 124: '^O[123]$', 125: '^C[12]$', 126: '^N1$',
                 127: '^C[345]$', 128: '^BAL$', 129: '^POI$', 130: '^DOT$', 131: '^CU$', 132: '^ZN$', 133: '^MN$',
                 134: '^FE$', 135: '^MG$', 136: '^MN$', 137: '^CO$', 138: '^SE$', 139: '^YB$', 140: '^N1$',
                 141: '^C[2478]$', 142: '^O2$', 143: '^N3$', 144: '^O4$', 145: '^C[459]A$', 146: '^N5$',
                 147: '^C[69]$', 148: '^C[78]M$', 149: '^N10$', 150: '^C10$', 151: '^C[12345]\\*$',
                 152: '^O[234]\\*$', 153: '^O5\\*$', 154: '^OP[1-3]$', 155: '^OT1$', 156: '^C01$', 157: '^C16$',
                 158: '^C14$', 159: '^C.*$', 160: '^SEG$', 161: '^OXT$', 162: '^OT.*$', 163: '^E.*$', 164: '^S.*$',
                 165: '^C.*$', 166: '^A.*$', 167: '^O.*$', 168: '^N.*$', 169: '^R.*$', 170: '^K.*$', 171: '^P[A-D]$',
                 172: '^P.*$', 173: '^.O.*$', 174: '^.N.*$', 175: '^.C.*$', 176: '^.P.*$', 177: '^.H.*$'}
RESIDUE_PATTERNS = {0: '^.*$', 1: '^.*$', 2: '^WAT|HOH|H2O|DOD|DIS$', 3: '^CA$', 4: '^CD$', 5: '^.*$', 6: '^ACE$',
                    7: '^.*$', 8: '^.*$', 9: '^.*$', 10: '^.*$', 11: '^.*$', 12: '^ALA$', 13: '^ILE|THR|VAL$',
                    14: '^.*$', 15: '^ASN|ASP|ASX|HIS|HIP|HIE|HID|HISN|HISL|LEU|PHE|TRP|TYR$',
                    16: '^ARG|GLU|GLN|GLX|MET$', 17: '^LEU$', 18: '^.*$', 19: '^GLN$', 20: '^GLN$', 21: '^ACE$',
                    22: '^ARG$', 23: '^ARG$', 24: '^ARG$', 25: '^ARG$', 26: '^ARG$', 27: '^ARG$', 28: '^ASN$',
                    29: '^ASN$', 30: '^ASN$', 31: '^ASN$', 32: '^ASP$', 33: '^ASP$', 34: '^ASX$', 35: '^ASX$',
                    36: '^ASX$', 37: '^ASX$', 38: '^ASX$', 39: '^CYS|MET$', 40: '^CY[SXM]$', 41: '^CYH$',
                    42: '^GLU$', 43: '^GLU$', 44: '^GLU|GLN|GLX$', 45: '^GLN$', 46: '^GLN$', 47: '^GLN|GLX$',
                    48: '^HIS|HID|HIE|HIP|HISL$', 49: '^HIS|HIE|HISL$', 50: '^HID|HIP$', 51: '^HID|HIP$',
                    52: '^HIS|HIE|HIP$', 53: '^HIS|HIE|HIP$', 54: '^HID|HISL$', 55: '^HID|HISL$',
                    56: '^HIS|HID|HIP|HISD$', 57: '^ILE$', 58: '^ILE$', 59: '^ILE$', 60: '^LEU$', 61: '^LEU$',
                    62: '^LYS$', 63: '^LYS$', 64: '^LYS$', 65: '^MET$', 66: '^MET$', 67: '^PHE$', 68: '^PHE$',
                    69: '^PRO|CPR$', 70: '^CSO$', 71: '^CSO$', 72: '^CSO$', 73: '^CSO$', 74: '^SER$', 75: '^THR$',
                    76: '^THR$', 77: '^TRP$', 78: '^TRP$', 79: '^TRP$', 80: '^TRP$', 81: '^TRP$', 82: '^TRP$',
                    83: '^TRP$', 84: '^TRP$', 85: '^TYR$', 86: '^TYR$', 87: '^TYR$', 88: '^VAL$', 89: '^VAL$',
                    90: '^.*$', 91: '^.*$', 92: '^FS[34]$', 93: '^FS[34]$', 94: '^FS3$', 95: '^FEO$', 96: '^FEO$',
                    97: '^HEM$', 98: '^HEM$', 99: '^HEM$', 100: '^HEM$', 101: '^HEM$', 102: '^HEM$', 103: '^HEM$',
                    104: '^HEM$', 105: '^HEM$', 106: '^HEM$', 107: '^HEM$', 108: '^HEM$', 109: '^HEM$', 110: '^AZI$',
                    111: '^MPD$', 112: '^MPD$', 113: '^MPD$', 114: '^MPD$', 115: '^MPD$', 116: '^MPD$', 117: '^MPD$',
                    118: '^MPD$', 119: '^SO4|SUL$', 120: '^SO4|SUL$', 121: '^PO4|PHO$', 122: '^PC$', 123: '^PC$',
                    124: '^PC$', 125: '^PC$', 126: '^PC$', 127: '^PC$', 128: '^BIG$', 129: '^POI$', 130: '^DOT$',
                    131: '^.*$', 132: '^.*$', 133: '^.*$', 134: '^.*$', 135: '^.*$', 136: '^.*$', 137: '^.*$',
                    138: '^.*$', 139: '^.*$', 140: '^FMN$', 141: '^FMN$', 142: '^FMN$', 143: '^FMN$', 144: '^FMN$',
                    145: '^FMN$', 146: '^FMN$', 147: '^FMN$', 148: '^FMN$', 149: '^FMN$', 150: '^FMN$', 151: '^FMN$',
                    152: '^FMN$', 153: '^FMN$', 154: '^FMN$', 155: '^ALK|MYR$', 156: '^ALK|MYR$', 157: '^ALK$',
                    158: '^MYR$', 159: '^ALK|MYR$', 160: '^.*$', 161: '^.*$', 162: '^.*$', 163: '^.*$', 164: '^.*$',
                    165: '^.*$', 166: '^.*$', 167: '^.*$', 168: '^.*$', 169: '^.*$', 170: '^.*$', 171: '^.*$',
                    172: '^.*$', 173: '^FAD|NAD|AMX|APU$', 174: '^FAD|NAD|AMX|APU$', 175: '^FAD|NAD|AMX|APU$',
                    176: '^FAD|NAD|AMX|APU$', 177: '^FAD|NAD|AMX|APU$'}
ATOM_NUM = {0: 15, 1: 15, 2: 2, 3: 18, 4: 22, 5: 22, 6: 9, 7: 4, 8: 7, 9: 10, 10: 1, 11: 13, 12: 9, 13: 7, 14: 8,
            15: 10, 16: 8, 17: 7, 18: 8, 19: 3, 20: 3, 21: 9, 22: 8, 23: 4, 24: 4, 25: 10, 26: 5, 27: 5, 28: 1,
            29: 5, 30: 3, 31: 3, 32: 3, 33: 3, 34: 1, 35: 5, 36: 3, 37: 3, 38: 3, 39: 13, 40: 13, 41: 12, 42: 3,
            43: 3, 44: 10, 45: 1, 46: 5, 47: 3, 48: 11, 49: 14, 50: 4, 51: 4, 52: 4, 53: 4, 54: 14, 55: 14, 56: 4,
            57: 8, 58: 9, 59: 9, 60: 9, 61: 9, 62: 8, 63: 6, 64: 6, 65: 13, 66: 9, 67: 11, 68: 11, 69: 8, 70: 9,
            71: 9, 72: 3, 73: 3, 74: 2, 75: 2, 76: 9, 77: 11, 78: 10, 79: 10, 80: 4, 81: 11, 82: 11, 83: 11, 84: 11,
            85: 11, 86: 10, 87: 2, 88: 9, 89: 9, 90: 8, 91: 8, 92: 21, 93: 13, 94: 1, 95: 21, 96: 21, 97: 1, 98: 1,
            99: 21, 100: 11, 101: 14, 102: 14, 103: 10, 104: 9, 105: 8, 106: 10, 107: 3, 108: 11, 109: 2, 110: 14,
            111: 9, 112: 10, 113: 8, 114: 7, 115: 9, 116: 9, 117: 2, 118: 2, 119: 13, 120: 3, 121: 3, 122: 3,
            123: 13, 124: 3, 125: 8, 126: 14, 127: 9, 128: 17, 129: 23, 130: 23, 131: 20, 132: 19, 133: 24, 134: 25,
            135: 26, 136: 27, 137: 28, 138: 29, 139: 31, 140: 4, 141: 10, 142: 1, 143: 14, 144: 1, 145: 10, 146: 4,
            147: 11, 148: 9, 149: 4, 150: 10, 151: 8, 152: 2, 153: 3, 154: 3, 155: 3, 156: 10, 157: 9, 158: 9,
            159: 8, 160: 9, 161: 3, 162: 3, 163: 3, 164: 13, 165: 7, 166: 11, 167: 1, 168: 4, 169: 4, 170: 6,
            171: 13, 172: 13, 173: 1, 174: 4, 175: 7, 176: 13, 177: 15}

# 15 -> H
ELEMENT_NAME ={
  1: "O",
  2: "O",
  3: "O",
  4: "N",
  5: "N",
  6: "N",  # Also K
  7: "C",
  8: "C",
  9: "C",
  10: "C",
  11: "C",
  12: "S",
  13: "S",  # Also P
  14: "N",
  15: "H",
  17: "UNK",
  18: "C",
  19: "ZN",
  20: "CU",
  21: "FE",
  22: "C",
  23: "UNK",
  24: "MN",
  25: "FE",
  26: "MG",
  27: "MN",
  28: "C",
  29: "S",
  31: "UNK",
}

element_color_map = {
  # BASIC ELEMENTS
  "C": [0.5, 0.5, 0.5],
  "H": [1,1,1],
  "N": [0,0,1],
  "O": [1,0,0],
  "S": [1,1,0],
  "P": [1,0.6,0.4],

  # METALS
  "NA": [0.7, 0.7, 0.1],
  "MG": [0.7, 0.7, 0.1],
  "CA": [0.7, 0.7, 0.1],
  "K" : [0, 0.5, 1],
  "ZN": [0.8, 0.4, 0.1],
  "CU": [0.8, 0.4, 0.1],
  "FE": [0.8, 0.4, 0.1],
  "MN": [0.6, 0, 0.4],

  # UNKNOWNS
  "UNK": [0.5, 0.5, 0.5],
  "U": [0.5, 0.5, 0.5],
}


# Generate Open3D readable object
def get_atom_num(atom="", residue=""):
  """
  Get the atomic number of an atom based on its atom name and residue name.
  Args:
    atom (str): atom name
    residue (str): residue name
  Returns:
    (int): atomic number
  """
  atom = atom.replace(" ", "")
  residue = residue.replace(" ", "")
  for pat in range(len(ATOM_NUM)):
    if re.match(ATOM_PATTERNS[pat], atom) and re.match(RESIDUE_PATTERNS[pat], residue):
      break
  if pat == len(ATOM_NUM):
    print(f"Warning: Atom {atom} in {residue} not found in the available patterns. Using default radius of 0.01")
    return "U"
  else:
    return ELEMENT_NAME.get(ATOM_NUM[pat], "U")



def rotation_matrix_from_vectors(vec1, vec2):
  """
  Find the rotation matrix that aligns vec1 to vec2
  Args:
    vec1: A 3d "source" vector
    vec2: A 3d "destination" vector
  Returns:
    mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
  """
  a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
  v = np.cross(a, b)
  c = np.dot(a, b)
  s = np.linalg.norm(v)
  kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
  return rotation_matrix


def transform_coord(coord, matrix):
  """
  Directly transform the <N,3> coordinate with the given transformation matrix
  TODO: Support more modes other than just 4*4 transformation matrix
  """
  if matrix.shape != (4, 4):
    raise ValueError("Expecting a 4x4 transformation matrix")
  coord_4 = np.ones((len(coord), 4))
  coord_4[:, :3] = coord
  newcoord = matrix @ coord_4.T
  newcoord = newcoord.T[:, :3]
  return newcoord

def create_sphere(center, radius=0.5, color=[0, 0, 1]):
  """
  Create a sphere with the given center and radius
  Args:
    center (np.array): center of the sphere
    radius (float): radius of the sphere
    color (list): color of the sphere
  Returns:
    sphere (o3d.geometry.TriangleMesh): open3d sphere object
  """
  sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
  sphere.paint_uniform_color(color)
  sphere.translate(center)
  sphere.compute_vertex_normals()
  return sphere


def create_cylinder(start, end, radius=0.2, color=[0.4275, 0.2941, 0.0745]):
  """
  Create a cylinder with the given start and end points and radius
  Args:
    start (np.array): start point of the cylinder
    end (np.array): end point of the cylinder
    radius (float): radius of the cylinder
    color (list): color of the cylinder
  Returns:
    cylinder (o3d.geometry.TriangleMesh): open3d cylinder object
  """
  vec = end - start
  length = np.linalg.norm(vec)
  cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, length)
  cylinder.paint_uniform_color(color)

  direction = vec / length
  rot = rotation_matrix_from_vectors([0, 0, 1], direction)  # Change to z-axis
  cylinder.rotate(rot, center=[0, 0, 0])  # Rotate around the origin

  mid = (start + end) / 2
  cylinder.translate(mid - cylinder.get_center())
  cylinder.compute_vertex_normals()
  return cylinder


def molecule_to_o3d(pdb_path):
  """
  Convert a molecule to an open3d object
  Args:
    pdb_path (str): path to the pdb file
  Returns:
    geometries (list): list of open3d objects
  """
  # Load PDB structure
  structure = pt.load(pdb_path)
  atoms = list(structure.top.atoms)
  residues = list(structure.top.residues)
  coords = list(structure.xyz[0])

  # Add spheres as each atom
  geometries = []
  for idx, c in enumerate(coords):
    theatom = atoms[idx]
    resname = residues[theatom.resid].name
    atomtype = get_atom_num(theatom.name, resname)
    print(f"Atom Name: {theatom.name:8s} | Res Name: {resname:8s} | ---> {atomtype} |")
    color = element_color_map.get(atomtype, [0.5, 0.5, 0.5])
    geometries.append(create_sphere(c, radius=0.5, color=color))

  # Add cylinders as bonds
  for bond in list(structure.top.bonds):
    n_i, n_j = bond.indices
    pos_1 = coords[n_i]
    pos_2 = coords[n_j]
    if np.linalg.norm(pos_1 - pos_2) < 3:  # Simple condition to check if there is a bond
      geometries.append(create_cylinder(pos_1, pos_2, radius=0.15))
  return geometries

def display_icp(static_objs, dynamic_objs, outfile="", resetbb1=True, resetbb2=False):
    """
    Dynamics object are processed via their deep copies
    """
    import subprocess, tempfile, os

    getobj = lambda p: o3d.io.read_triangle_mesh(p) if o3d.io.read_triangle_mesh(p).triangles.__len__() > 0 else o3d.io.read_point_cloud(p)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    
    for sobj in static_objs:
        if isinstance(sobj, str) and sobj.endswith('.ply'):
            sobj = getobj(sobj)
        elif isinstance(sobj, (o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, o3d.geometry.LineSet)):
            sobj = copy.deepcopy(sobj)
        else:
            raise ValueError("Expecting an object file or an open3d object")
        vis.add_geometry(sobj, reset_bounding_box=bool(resetbb1))
        vis.update_geometry(sobj)
        vis.poll_events()
        vis.update_renderer()
    
    # For each object in the dynamic_objs, Added, render and then remove it. 
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Working on tempfolder {temp_dir}")
        for i, obj in enumerate(dynamic_objs):
            if isinstance(obj, str) and obj.endswith('.ply'):
                geom = getobj(obj)
            elif isinstance(obj, (o3d.geometry.TriangleMesh, o3d.geometry.PointCloud)):
                geom = copy.deepcopy(obj)
            else:
                raise ValueError("Expecting an object file or an open3d object")
            # Add point cloud to the visualizer
            vis.add_geometry(geom, reset_bounding_box=bool(resetbb2))
            vis.update_geometry(geom)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(os.path.join(temp_dir, f"frame_{i:04d}.png"))
            # Remove point cloud from the visualizer for the next iteration
            vis.remove_geometry(geom, reset_bounding_box=bool(resetbb2))
        # Destroy the visualizer
        vis.destroy_window()

        if len(outfile) > 0: 
            # Compile frames into a movie using FFmpeg
            ffmpeg_command = f"ffmpeg -r 24 -i {temp_dir}/frame_%04d.png -vcodec libx264 -pix_fmt yuv422p -y {outfile}"
            subprocess.run(ffmpeg_command, shell=True)

def xyzr_to_o3d(xyzr_path, radius_factor=1.0):
  # Load PDB structure
  with open(xyzr_path, "r") as f:
    lines = f.readlines()
  coords = []
  radii = []
  for line in lines:
    line = line.strip().split()
    coords.append([float(line[0]), float(line[1]), float(line[2])])
    radii.append(float(line[3]))
  coords = np.array(coords)
  radii = np.array(radii) * radius_factor
  geometries = []
  for idx, c in enumerate(coords):
    color = [0.5,0.5,0.5]
    geometries.append(create_sphere(c, radius=radii[idx], color=color))
  return geometries

def get_objs(objfiles, show_wire=1):
  final_geometries = []
  for file in objfiles:
    if ".pdb" in file or ".mol2" in file:
      geometries = molecule_to_o3d(file)
      final_geometries += geometries
    elif ".ply" in file:
      mesh = o3d.io.read_triangle_mesh(file)
      mesh.paint_uniform_color([0.5, 0.1, 0.1])
      if bool(show_wire):
        lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        final_geometries += [lineset]
      else:
        mesh.compute_vertex_normals()
        final_geometries += [mesh]
    elif ".obj" in file:
      mesh = o3d.io.read_triangle_mesh(file)
      mesh.paint_uniform_color([0.5, 0.1, 0.1])
      mesh.compute_vertex_normals()
      final_geometries += [mesh]
    elif ".xyzr" in file:
      xyzr_geoms = xyzr_to_o3d(file, radius_factor=1)
      final_geometries += xyzr_geoms
    else:
      print(f"Warning: {file} is not a supported file type. Skipping...")
  return final_geometries

def get_coord(source_file):
  """
  Get the coordinates of atoms from a molecule or XYZR file
  """
  if ".pdb" in source_file or ".mol2" in source_file:
    coord = np.asarray(pt.load(source_file).xyz[0], dtype=np.float64)
  elif ".xyzr" in source_file:
    with open(source_file, "r") as f1:
      coord = [i.split()[:3] for i in f1.read().strip("\n").split("\n")]
      coord = np.asarray(coord, dtype=np.float64)
  return coord

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
