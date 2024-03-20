import os, sys, re, subprocess, tempfile, copy
import time
from itertools import combinations

import pytraj as pt
import numpy as np 

from scipy import spatial
from scipy.spatial.distance import cdist, pdist, squareform

import open3d as o3d
from rdkit import Chem

import nearl
from nearl import utils, io
from .. import printit

__all__ = [
  "generator",
  "PointFeature",
  "compute_convex",
  "SEGMENT_CMAPS"
]

ACCEPTOR_PATTERN = '[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'
DONOR_PATTERN = "[!H0;#7,#8,#9]"
RD_DONOR_PATTERN = Chem.MolFromSmarts(DONOR_PATTERN)
RD_ACCEPTOR_PATTERN = Chem.MolFromSmarts(ACCEPTOR_PATTERN)

ATOM_PATTERNS = {0: '^[0-9]*H.*$', 1: '^[0-9]*D.*$', 2: '^O.*$', 3: '^CA$', 4: '^CD$', 5: '^CD  $', 6: '^CA$', 7: '^N$', 8: '^CA$', 9: '^C$', 10: '^O$', 11: '^P$', 12: '^CB$', 13: '^CB$', 14: '^CB$', 15: '^CG$', 16: '^CG$', 17: '^CG$', 18: '^CG$', 19: '^O1$', 20: '^O2$', 21: '^CH3$', 22: '^CD$', 23: '^NE$', 24: '^RE$', 25: '^CZ$', 26: '^NH[12][AB]?$', 27: '^RH[12][AB]?$', 28: '^OD1$', 29: '^ND2$', 30: '^AD1$', 31: '^AD2$', 32: '^OD[12][AB]?$', 33: '^ED[12][AB]?$', 34: '^OD1[AB]?$', 35: '^ND2$', 36: '^AD1$', 37: '^AD2$', 38: '^OD2$', 39: '^LP[12]$', 40: '^SG$', 41: '^SG$', 42: '^OE[12][AB]?$', 43: '^EE[12][AB]?$', 44: '^CD$', 45: '^OE1$', 46: '^NE2$', 47: '^AE[12]$', 48: '^CE1|CD2$', 49: '^ND1$', 50: '^ND1$', 51: '^RD1$', 52: '^NE2$', 53: '^RE2$', 54: '^NE2$', 55: '^RE2$', 56: '^A[DE][12]$', 57: '^CG1$', 58: '^CG2$', 59: '^CD|CD1$', 60: '^CD1$', 61: '^CD2$', 62: '^C[GDE]$', 63: '^NZ$', 64: '^KZ$', 65: '^SD$', 66: '^CE$', 67: '^C[DE][12]$', 68: '^CZ$', 69: '^C[GD]$', 70: '^SE$', 71: '^SEG$', 72: '^OD1$', 73: '^OD2$', 74: '^OG$', 75: '^OG1$', 76: '^CG2$', 77: '^CD1$', 78: '^CD2$', 79: '^CE2$', 80: '^NE1$', 81: '^CE3$', 82: '^CZ2$', 83: '^CZ3$', 84: '^CH2$', 85: '^C[DE][12]$', 86: '^CZ$', 87: '^OH$', 88: '^CG1$', 89: '^CG2$', 90: '^CD$', 91: '^CE$', 92: '^FE[1-7]$', 93: '^S[1-7]$', 94: '^OXO$', 95: '^FE1$', 96: '^FE2$', 97: '^O1$', 98: '^O2$', 99: '^FE$', 100: '^CH[A-D]$', 101: '^N[A-D]$', 102: '^N [A-D]$', 103: '^C[1-4][A-D]$', 104: '^CM[A-D]$', 105: '^C[AB][AD]$', 106: '^CG[AD]$', 107: '^O[12][AD]$', 108: '^C[AB][BC]$', 109: '^OH2$', 110: '^N[123]$', 111: '^C1$', 112: '^C2$', 113: '^C3$', 114: '^C4$', 115: '^C5$', 116: '^C6$', 117: '^O7$', 118: '^O8$', 119: '^S$', 120: '^O[1234]$', 121: '^O[1234]$', 122: '^O4$', 123: '^P1$', 124: '^O[123]$', 125: '^C[12]$', 126: '^N1$', 127: '^C[345]$', 128: '^BAL$', 129: '^POI$', 130: '^DOT$', 131: '^CU$', 132: '^ZN$', 133: '^MN$', 134: '^FE$', 135: '^MG$', 136: '^MN$', 137: '^CO$', 138: '^SE$', 139: '^YB$', 140: '^N1$', 141: '^C[2478]$', 142: '^O2$', 143: '^N3$', 144: '^O4$', 145: '^C[459]A$', 146: '^N5$', 147: '^C[69]$', 148: '^C[78]M$', 149: '^N10$', 150: '^C10$', 151: '^C[12345]\\*$', 152: '^O[234]\\*$', 153: '^O5\\*$', 154: '^OP[1-3]$', 155: '^OT1$', 156: '^C01$', 157: '^C16$', 158: '^C14$', 159: '^C.*$', 160: '^SEG$', 161: '^OXT$', 162: '^OT.*$', 163: '^E.*$', 164: '^S.*$', 165: '^C.*$', 166: '^A.*$', 167: '^O.*$', 168: '^N.*$', 169: '^R.*$', 170: '^K.*$', 171: '^P[A-D]$', 172: '^P.*$', 173: '^.O.*$', 174: '^.N.*$', 175: '^.C.*$', 176: '^.P.*$', 177: '^.H.*$'}
RESIDUE_PATTERNS = {0: '^.*$', 1: '^.*$', 2: '^WAT|HOH|H2O|DOD|DIS$', 3: '^CA$', 4: '^CD$', 5: '^.*$', 6: '^ACE$', 7: '^.*$', 8: '^.*$', 9: '^.*$', 10: '^.*$', 11: '^.*$', 12: '^ALA$', 13: '^ILE|THR|VAL$', 14: '^.*$', 15: '^ASN|ASP|ASX|HIS|HIP|HIE|HID|HISN|HISL|LEU|PHE|TRP|TYR$', 16: '^ARG|GLU|GLN|GLX|MET$', 17: '^LEU$', 18: '^.*$', 19: '^GLN$', 20: '^GLN$', 21: '^ACE$', 22: '^ARG$', 23: '^ARG$', 24: '^ARG$', 25: '^ARG$', 26: '^ARG$', 27: '^ARG$', 28: '^ASN$', 29: '^ASN$', 30: '^ASN$', 31: '^ASN$', 32: '^ASP$', 33: '^ASP$', 34: '^ASX$', 35: '^ASX$', 36: '^ASX$', 37: '^ASX$', 38: '^ASX$', 39: '^CYS|MET$', 40: '^CY[SXM]$', 41: '^CYH$', 42: '^GLU$', 43: '^GLU$', 44: '^GLU|GLN|GLX$', 45: '^GLN$', 46: '^GLN$', 47: '^GLN|GLX$', 48: '^HIS|HID|HIE|HIP|HISL$', 49: '^HIS|HIE|HISL$', 50: '^HID|HIP$', 51: '^HID|HIP$', 52: '^HIS|HIE|HIP$', 53: '^HIS|HIE|HIP$', 54: '^HID|HISL$', 55: '^HID|HISL$', 56: '^HIS|HID|HIP|HISD$', 57: '^ILE$', 58: '^ILE$', 59: '^ILE$', 60: '^LEU$', 61: '^LEU$', 62: '^LYS$', 63: '^LYS$', 64: '^LYS$', 65: '^MET$', 66: '^MET$', 67: '^PHE$', 68: '^PHE$', 69: '^PRO|CPR$', 70: '^CSO$', 71: '^CSO$', 72: '^CSO$', 73: '^CSO$', 74: '^SER$', 75: '^THR$', 76: '^THR$', 77: '^TRP$', 78: '^TRP$', 79: '^TRP$', 80: '^TRP$', 81: '^TRP$', 82: '^TRP$', 83: '^TRP$', 84: '^TRP$', 85: '^TYR$', 86: '^TYR$', 87: '^TYR$', 88: '^VAL$', 89: '^VAL$', 90: '^.*$', 91: '^.*$', 92: '^FS[34]$', 93: '^FS[34]$', 94: '^FS3$', 95: '^FEO$', 96: '^FEO$', 97: '^HEM$', 98: '^HEM$', 99: '^HEM$', 100: '^HEM$', 101: '^HEM$', 102: '^HEM$', 103: '^HEM$', 104: '^HEM$', 105: '^HEM$', 106: '^HEM$', 107: '^HEM$', 108: '^HEM$', 109: '^HEM$', 110: '^AZI$', 111: '^MPD$', 112: '^MPD$', 113: '^MPD$', 114: '^MPD$', 115: '^MPD$', 116: '^MPD$', 117: '^MPD$', 118: '^MPD$', 119: '^SO4|SUL$', 120: '^SO4|SUL$', 121: '^PO4|PHO$', 122: '^PC$', 123: '^PC$', 124: '^PC$', 125: '^PC$', 126: '^PC$', 127: '^PC$', 128: '^BIG$', 129: '^POI$', 130: '^DOT$', 131: '^.*$', 132: '^.*$', 133: '^.*$', 134: '^.*$', 135: '^.*$', 136: '^.*$', 137: '^.*$', 138: '^.*$', 139: '^.*$', 140: '^FMN$', 141: '^FMN$', 142: '^FMN$', 143: '^FMN$', 144: '^FMN$', 145: '^FMN$', 146: '^FMN$', 147: '^FMN$', 148: '^FMN$', 149: '^FMN$', 150: '^FMN$', 151: '^FMN$', 152: '^FMN$', 153: '^FMN$', 154: '^FMN$', 155: '^ALK|MYR$', 156: '^ALK|MYR$', 157: '^ALK$', 158: '^MYR$', 159: '^ALK|MYR$', 160: '^.*$', 161: '^.*$', 162: '^.*$', 163: '^.*$', 164: '^.*$', 165: '^.*$', 166: '^.*$', 167: '^.*$', 168: '^.*$', 169: '^.*$', 170: '^.*$', 171: '^.*$', 172: '^.*$', 173: '^FAD|NAD|AMX|APU$', 174: '^FAD|NAD|AMX|APU$', 175: '^FAD|NAD|AMX|APU$', 176: '^FAD|NAD|AMX|APU$', 177: '^FAD|NAD|AMX|APU$'}
EXP_RADII = {1: 1.4, 2: 1.4, 3: 1.4, 4: 1.54, 5: 1.54, 6: 1.54, 7: 1.74, 8: 1.74, 9: 1.74, 10: 1.74, 11: 1.74, 12: 1.8, 13: 1.8, 14: 1.54, 15: 1.2, 16: 0.0, 17: 5.0, 18: 1.97, 19: 1.4, 20: 1.4, 21: 1.3, 22: 1.49, 23: 0.01, 24: 1.2, 25: 1.24, 26: 1.6, 27: 1.24, 28: 1.25, 29: 2.15, 30: 3.0, 31: 1.15, 38: 1.8}
UNITED_RADII = {1: 1.4, 2: 1.6, 3: 1.4, 4: 1.7, 5: 1.8, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0, 10: 1.74, 11: 1.86, 12: 1.85, 13: 1.8, 14: 1.54, 15: 1.2, 16: 1.5, 17: 5.0, 18: 1.97, 19: 1.4, 20: 1.4, 21: 1.3, 22: 1.49, 23: 0.01, 24: 0.0, 25: 1.24, 26: 1.6, 27: 1.24, 28: 1.25, 29: 2.15, 30: 3.0, 31: 1.15, 38: 1.8}
ATOM_NUM = {0: 15, 1: 15, 2: 2, 3: 18, 4: 22, 5: 22, 6: 9, 7: 4, 8: 7, 9: 10, 10: 1, 11: 13, 12: 9, 13: 7, 14: 8, 15: 10, 16: 8, 17: 7, 18: 8, 19: 3, 20: 3, 21: 9, 22: 8, 23: 4, 24: 4, 25: 10, 26: 5, 27: 5, 28: 1, 29: 5, 30: 3, 31: 3, 32: 3, 33: 3, 34: 1, 35: 5, 36: 3, 37: 3, 38: 3, 39: 13, 40: 13, 41: 12, 42: 3, 43: 3, 44: 10, 45: 1, 46: 5, 47: 3, 48: 11, 49: 14, 50: 4, 51: 4, 52: 4, 53: 4, 54: 14, 55: 14, 56: 4, 57: 8, 58: 9, 59: 9, 60: 9, 61: 9, 62: 8, 63: 6, 64: 6, 65: 13, 66: 9, 67: 11, 68: 11, 69: 8, 70: 9, 71: 9, 72: 3, 73: 3, 74: 2, 75: 2, 76: 9, 77: 11, 78: 10, 79: 10, 80: 4, 81: 11, 82: 11, 83: 11, 84: 11, 85: 11, 86: 10, 87: 2, 88: 9, 89: 9, 90: 8, 91: 8, 92: 21, 93: 13, 94: 1, 95: 21, 96: 21, 97: 1, 98: 1, 99: 21, 100: 11, 101: 14, 102: 14, 103: 10, 104: 9, 105: 8, 106: 10, 107: 3, 108: 11, 109: 2, 110: 14, 111: 9, 112: 10, 113: 8, 114: 7, 115: 9, 116: 9, 117: 2, 118: 2, 119: 13, 120: 3, 121: 3, 122: 3, 123: 13, 124: 3, 125: 8, 126: 14, 127: 9, 128: 17, 129: 23, 130: 23, 131: 20, 132: 19, 133: 24, 134: 25, 135: 26, 136: 27, 137: 28, 138: 29, 139: 31, 140: 4, 141: 10, 142: 1, 143: 14, 144: 1, 145: 10, 146: 4, 147: 11, 148: 9, 149: 4, 150: 10, 151: 8, 152: 2, 153: 3, 154: 3, 155: 3, 156: 10, 157: 9, 158: 9, 159: 8, 160: 9, 161: 3, 162: 3, 163: 3, 164: 13, 165: 7, 166: 11, 167: 1, 168: 4, 169: 4, 170: 6, 171: 13, 172: 13, 173: 1, 174: 4, 175: 7, 176: 13, 177: 15}


# Color map for the segments of the molecule block
# TODO removed the CONFIG from the nearl module, figure how to fix the passing of necessary variable 
_SEGMENT_LIMIT = 6
_SEGMENT_CMAP = None

if _SEGMENT_CMAP is not None and _SEGMENT_CMAP != "inferno":
  # Not the default color map;
  from matplotlib.pyplot import get_cmap
  cmap = get_cmap(_SEGMENT_CMAP)
  SEGMENT_CMAPS = [cmap(i)[:3] for i in range(int(0.1 * cmap.N), int(0.9 * cmap.N), int(0.9 * cmap.N) // 10)]
elif _SEGMENT_CMAP == "inferno" and _SEGMENT_LIMIT == 6:
  SEGMENT_CMAPS = [
    [0.087411, 0.044556, 0.224813],
    [0.354032, 0.066925, 0.430906],
    [0.60933, 0.159474, 0.393589],
    [0.841969, 0.292933, 0.248564],
    [0.974176, 0.53678, 0.048392],
    [0.964394, 0.843848, 0.273391]
  ]
  if nearl._verbose:
    printit("Using the default color map/gradient")
else:
  # Default color map -> inferno
  from matplotlib.pyplot import get_cmap
  cmap = get_cmap('inferno')
  SEGMENT_CMAPS = [cmap(i)[:3] for i in range(int(0.1 * cmap.N), int(cmap.N * 0.9), int(cmap.N * 0.9) // 10)]


# Generate Open3D readable object
def getRadius(atom="", residue="", exp=True):
  """
  Get the radius of an atom based on its atom name and residue name.
  Args:
    atom (str): atom name
    residue (str): residue name
    exp (bool): whether to use explicit radii or united atom radii
  """
  atom = atom.replace(" ", "")
  residue = residue.replace(" ", "")
  for pat in range(len(ATOM_NUM)):
    if re.match(ATOM_PATTERNS[pat], atom) and re.match(RESIDUE_PATTERNS[pat], residue):
      break
  if pat == len(ATOM_NUM):
    print(f"Warning: Atom {atom} in {residue} not found in the available patterns. Using default radius of 0.01")
    rad = 0.01
  else:
    # Map the pattern to the atom number and get its radius
    rad = UNITED_RADII[ATOM_NUM[pat]] if exp != True else EXP_RADII[ATOM_NUM[pat]]
  return rad


def pdb2xyzr(thepdb, write="", exp=True):
  """
  Convert atoms in a PDB file to xyzr format.
  Args:
    thepdb (str): path to the PDB file or the PDB string
    write (str): path to write the xyzr file
    exp (bool): whether to use explicit radii or united atom radii
  """
  if os.path.isfile(thepdb): 
    with open(thepdb, "r") as f:
      pdblines = f.read().strip("\n").split("\n")
  elif ("ATOM" in thepdb) or ("HETATM" in thepdb): 
    pdblines = thepdb.strip("\n").split("\n")
  else: 
    raise Exception(f"{__name__:15s}: Please provide a valid PDB file path or PDB string.")
  finallines = ""
  for line in pdblines:
    line = line.strip()
    if not line.startswith("ATOM") and not line.startswith("HETATM"):
      continue
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    resname = line[17:20].strip()
    aname = line[12:16].strip()
    # special handling for hydrogens (start with digits, not the letter "H")
    if re.match(r"\d[HhDd]", aname):
      aname = "H"
    # However, some bogus PDP files have the H in column 13 so we allow those too, 
    # which means we will treat as Hydrogen helium and hafnium but we protect HG
    if re.match(r"[Hh][^Gg]", aname):
      aname = "H"
    resnum = line[22:26].strip()
    # Spaces in atom / residue name will be removed in getRadius function
    rad = getRadius(atom=aname, residue=resname, exp=exp)
    finallines += f"{x:10.3f}{y:10.3f}{z:10.3f}{rad:6.2f}\n"
  if len(write) > 0: 
    if "xyzr" not in write:
      filename = write+".xyzr"
    else:
      filename = write
    with open(filename, "w") as file1:
      file1.write(finallines)
  else: 
    return finallines


################################# Representation vector generator ##################################
def pseudo_energy(coord, mode, charges=[]):
  # Using the following code for distance calculation, speeds up a bit
  mode = mode.lower()
  pairs = list(combinations(np.arange(len(coord)), 2))
  energy_final = 0
  if mode == "lj": 
    for p in pairs: 
      p1 = coord[p[0]]
      p2 = coord[p[1]]
      dist = np.linalg.norm(p1 - p2)
      energy_final += pseudo_lj(dist)
  elif mode == "elec": 
    if len(charges) == 0:
      charges = np.zeros(len(coord))
    else: 
      if len(charges) != len(coord):
        raise Exception(f"The length of the charge ({len(charges)}) does not equal to the dimension of atoms({len(coord)})")
      else: 
        charges = np.array(charges)
    for p in pairs: 
      q1 = charges[p[0]]
      q2 = charges[p[1]]
      p1 = coord[p[0]]
      p2 = coord[p[1]]
      dist = np.linalg.norm(p1-p2)
      energy_final += pseudo_elec(q1, q2, dist)
  else: 
    raise Exception(f"{__name__:15s}: Only two pseudo-energy evaluation modes are supported: Lenar-jones (lj) and Electrostatic (elec)")
  return energy_final


def pseudo_lj(r, epsilon=1, sigma=1):
  """
  Calculates the Lennard-Jones potential for a given distance
  """
  return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)


def pseudo_elec(q1, q2, r):
  """
  Calculates the Coulombic interaction energy between two atoms
  """
  k = 8.98
  return k*q1*q2/r

def chargedict2array(traj, frame, charges):
  """
  TODO: Objective ????
  """
  _charge = {tuple(np.array(k).round(2)):v for k,v in charges.items()}
  chargelst = np.zeros(len(traj.xyz[frame]))
  for idx, coord in enumerate(traj.xyz[frame]):
    coord = tuple(coord.round(2))
    if coord in _charge.keys(): 
      chargelst[idx] = _charge[coord]
  return chargelst


def fpfh_similarity(fp1, fp2): 
  """
  Calculate the FPFH similarity between fpfh features 
  """
  dist_matrix = cdist(fp1, fp2, 'euclidean')
  similarity = 1 / (1 + np.mean(dist_matrix))
  return similarity


def fpfh_similarity2(fp1, fp2):
  """
  Calculate the FPFH similarity between fpfh features 
  """
  dist_matrix = fp1 - fp2
  similarity = 1 / (1 + np.abs(np.mean(dist_matrix)))
  return similarity


def write_ply(coords, normals=[], triangles=[], filename=""):
  """
  Write the PLY file for further visualization
  Args:
    coords: the coordinates of the vertices
    normals: the normals of the vertices
    triangles: the triangles of the mesh
    filename: the filename of the PLY file
  """
  header = ["ply", "format ascii 1.0", "comment author: Yang Zhang (y.zhang@bioc.uzh.ch)", f"element vertex {len(coords)}"]
  header.append("property float x")
  header.append("property float y")
  header.append("property float z")
  if len(normals) > 0: 
    header.append("property float nx")
    header.append("property float ny")
    header.append("property float nz")
  if len(triangles) > 0: 
    header.append(f"element face {len(triangles)}")
    header.append("property list uchar int vertex_indices")
  header.append("end_header")
  finalstr = ""
  # Write the PLY header
  for line in header:
    finalstr += (line + "\n")
  # Write the vertex data
  if len(normals) > 0: 
    for xyz, normal in zip(coords, normals):
      finalstr += f"{xyz[0]:<8.3f} {xyz[1]:<8.3f} {xyz[2]:<8.3f} {normal[0]:8.3f} {normal[1]:8.3f} {normal[2]:8.3f}\n"
  else: 
    for xyz in coords:
      finalstr += f"{xyz[0]:<8.3f} {xyz[1]:<8.3f} {xyz[2]:<8.3f}\n"
  if len(triangles) > 0: 
    triangles = np.asarray(triangles).astype(int)
    # Write the triangle data
    for tri in triangles:
      finalstr += f"3 {tri[0]:4d} {tri[1]:4d} {tri[2]:4d}\n"
  if len(filename) == 0:
    return finalstr
  else:
    with open(filename, "w") as file1:
      file1.write(finalstr)
      return True


class generator: 
  """
  NOTE: Test function of the generator
  >>> traj = pt.load(traj, top=top, mask=":1-151,:LIG", stride=100)
  >>> repres = repr_generator(traj)
  >>> repres.set_box([30,35,30], [8,8,8])
  >>> repres.set_frame(5)
  >>> segments = repres.query_segments()
  >>> feature_vector, mesh_obj, fpfh = repres.vectorize(segments)

  NOTE: voxel_down_sample might be a better solution to keep most feature during down-sampling
  >>> pcd_new = o3d.geometry.PointCloud( points=o3d.utility.Vector3dVector(finalobj.vertices))
  >>> pcd_new.normals = o3d.utility.Vector3dVector(mesh.vertex_normals)
  >>> finalobj_down = pcd_new.voxel_down_sample(0.8)
  >>> finalobj_down.estimate_normals(ktree)
  """
  def __init__(self, traj):
    """
    Initialize the molecule block representation generator class
    Register the trajectory and atoms information
    Args:
      traj: nearl.io.trajectory object
    """
    self.traj = traj
    self._center = np.zeros(3)
    self._length = np.ones(3)
    self._lower_bound = self.center - self.length / 2
    self._upper_bound = self.center + self.length / 2

    # Load parameters from the configuration file
    self.SEGMENT_LIMIT = _SEGMENT_LIMIT
    self.FPFH_DOWN_SAMPLES = 600  # TODO: removed config, how to pass this variable
    self.VIEWPOINTBINS = 125      # TODO: removed config, how to pass this variable

    self._SEGMENTS = np.zeros(self.traj.n_atoms)
    self._SEGMENTS_ORDER = None
    self._SEGMENTS_NUMBER = 0     # Not include the empty segment
    self._RDMOLS = []
    self._COMBINED_MESH = None
    self._INDICES = None
    self._INDICES_RES = None
    self._PDB_STRING = ""


    # Runtime-only properties
    self._SEGMENT = None
    self._SEGMENT_INDEX = 0
    self._SEGRES_MOL = None
    self._SEGRES_INDICES = None
    self._MESH = None
    self._VP_HIST = None
    self._ATOM_COUNT = None
    self._TEMP_PREFIX = None
    self._STANDPOINT = "next"  # TODO: removed config, how to pass this variable
    self._STANDPOINT_COORD = None

    if nearl._verbose:
      # Summary the configuration of the identity generator
      printit("Parameters are loaded")
      printit(f"SEGMENT_LIMIT: {self.SEGMENT_LIMIT}", end=" | ")
      print(f"DOWN_SAMPLE_POINTS: {self.FPFH_DOWN_SAMPLES}", end=" | ")
      print(f"VIEWPOINT_BINS: {self.VIEWPOINTBINS}", end=" | ")

  # Writable properties
  @property
  def center(self):
    return self._center

  @center.setter
  def center(self, new_center):
    """
    Set the center of the box
    Args:
      new_center: the new center (a list of number) of the box
    """
    assert len(new_center) == 3, "Length should be 3"
    the_new_center = np.array(new_center)
    self.set_box(the_new_center, self._length)

  @property
  def length(self):
    return self._length

  @length.setter
  def length(self, new_length):
    """
    Set the length of the box
    Args:
      new_length: the new length (A number or a list of number) of the box
    """
    if isinstance(new_length, (int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
      the_new_length = np.array([new_length, new_length, new_length])
      self.set_box(self._center, the_new_length)
    elif isinstance(new_length, (list, tuple, np.ndarray)):
      assert len(new_length) == 3, "length should be 3"
      the_new_length = np.array([i for i in new_length])
      self.set_box(self._center, the_new_length)
    else: 
      raise Exception("Unexpected data type for length (should be int, float, list, tuple or np.ndarray)")

  @property
  def frame(self):
    return self._frame

  @frame.setter
  def frame(self, framei: int):
    assert isinstance(framei, int), "Frame index should be int"
    self._frame = framei

  @property
  def segment_index(self):
    """
    The atomic index of the segment under vectorization
    """
    return self._SEGMENT_INDEX

  @segment_index.setter
  def segment_index(self, new_index: int):
    assert isinstance(new_index, int), "Segment index should be int"
    self._SEGMENT_INDEX = new_index


  # READ ONLY PROPERTIES
  @property
  def lower_bound(self):
    return self._lower_bound

  @property
  def upper_bound(self):
    return self._upper_bound

  @property
  def mesh(self):
    return self._MESH

  @property
  def mol(self):
    return self._SEGRES_MOL

  @property
  def segment_residue(self):
    """
    The residue-based atomic index of the segment under vectorization
    """
    return self._SEGRES_INDICES

  @property
  def segment(self):
    """
    The atomic index of the segment under vectorization i
    """
    return self._SEGMENT

  @property
  def atom_count(self):
    """
    Number of different atoms types in the segment i
    """
    return self._ATOM_COUNT

  @property
  def vpc(self):
    """
    Viewpoint component of the segment i
    """
    return self._VP_HIST

  @property
  def temp_prefix(self):
    if self._TEMP_PREFIX is None:
      prefix = self.set_tempprefix()
    else:
      prefix = self._TEMP_PREFIX
    return prefix

  @property
  def standpoint(self):
    return self._STANDPOINT

  @standpoint.setter
  def standpoint(self, new_standpoint):
    """
    Set the standpoint mode of the viewpoint
    Args:
      new_standpoint: the new standpoint mode str(self, next, first, end, previous) or an absolute coordinate (tuple, list or np.ndarray)
    """
    if isinstance(new_standpoint, str):
      if new_standpoint.lower() not in ["self", "next", "previous", "first", "end"]:
        raise Exception("Unexpected standpoint (should be self, next, first, end or previous)")
      self._STANDPOINT = new_standpoint.lower()
      self._STANDPOINT_COORD = None
    elif isinstance(new_standpoint, (tuple, list, np.ndarray)):
      self._STANDPOINT = "absolute"
      self._STANDPOINT_COORD = np.array(new_standpoint)
    else:
      raise Exception("Unexpected data type for standpoint (should be str(self, next, first, end, previous), or an absolute coordinate (tuple, list or np.ndarray))")

  @property
  def standpoint_coord(self):
    return self._STANDPOINT_COORD

  @standpoint_coord.setter
  def standpoint_coord(self, new_coord):
    if isinstance(new_coord, (tuple, list, np.ndarray)):
      self._STANDPOINT_COORD = np.array(new_coord)
    else:
      raise Exception("Unexpected data type for standpoint (should be str(self, next, first, end, previous), or an absolute coordinate (tuple, list or np.ndarray))")

  def set_frame(self, frameidx):
    self._frame = frameidx

  def set_box(self, center=None, length=None):
    """
    Configure the box information (self._center, self._length, self._lower_bound, self._upper_bound) of the generator
    Args:
      center: center of the box
      length: length of the box
    """
    if center is not None:
      self._center = np.asarray(center)
    if length is not None:
      self._length = np.asarray(length)
    self._lower_bound = self.center - self.length / 2
    self._upper_bound = self.center + self.length / 2
    assert len(self.center) == 3, "Length of center vector should be 3"
    assert len(self.length) == 3, "Length of length vector should be 3"
    assert len(self.lower_bound) == 3, "Length of lower bound vector should be 3"
    assert len(self.upper_bound) == 3, "Length of upper bound vector should be 3"
    assert np.all(self.lower_bound < self.upper_bound), "Lower bound should be smaller than upper bound"
    if not nearl._clear:
      # For each update of the box, update the temporary file prefix if not clear temporary files
      self.set_tempprefix()

  def set_tempprefix(self, tempprefix="", inplace=True):
    """
    Get an intermediate file prefix for the generator
    Args:
      tempprefix: manually set the prefix of the intermediate files
      inplace: if True, set the prefix to self._TEMP_PREFIX, otherwise return the prefix
    Returns:
      _tempfile_prefix: the prefix of the intermediate files
    """
    if len(tempprefix) > 0:
      _tempfile_prefix = os.path.join(nearl._tempfolder, f"tmp_{tempprefix}_")
    else:
      pid = os.getpid()
      temphash = utils.get_hash()[-10:]
      temphash = utils.get_hash(temphash + str(pid))[-10:]
      _tempfile_prefix = os.path.join(nearl._tempfolder, f"p{pid}_{temphash}_")
    if inplace:
      self._TEMP_PREFIX = _tempfile_prefix
    return _tempfile_prefix

  def close(self):
    """
    Remove the reference to the trajectory, mesh and rdkit molecule
    """
    del self.traj, self._MESH, self._SEGRES_MOL


  # Result variables
  @property
  def indices(self):
    """
    The atomic indices of all segments under vectorization
    """
    return np.array(self._INDICES)

  @property
  def indices_res(self):
    return np.array(self._INDICES_RES)

  @property
  def segments(self):
    """
    Segment group number series for each atom in the molecular system
    E.G. [X_i] * N where segment number of atom i (0-N) is X_i, N is the total number of atoms
    """
    return np.array(self._SEGMENTS)

  @property
  def segments_number(self):
    """
    Number of unique segment groups number
    """
    return int(self._SEGMENTS_NUMBER)

  @property
  def segments_order(self):
    """
    Segment group number from most abundant to least abundant
    E.G. [5, 1, 8, 2, 4, 9] means the most abundant segment is 5, the second most abundant segment is 1, etc.
    """
    return self._SEGMENTS_ORDER

  @property
  def mols(self):
    return [Chem.Mol(x) for x in self._RDMOLS]

  @property
  def meshes(self):
    """
    The meshes of all segments under vectorization
    """
    return [copy.deepcopy(i) for i in self._MESHES]

  @property
  def final_mesh(self):
    """
    The meshes of all segments under vectorization
    """
    return copy.deepcopy(self._COMBINED_MESH)

  @property
  def vertices(self):
    return np.array(self._COMBINED_MESH.vertices)

  @property
  def faces(self):
    return np.array(self._COMBINED_MESH.triangles)

  @property
  def normals(self):
    return np.array(self._COMBINED_MESH.vertex_normals)

  def get_ply_string(self):
    """
    Convert the mesh to a ply string
    """
    return write_ply(self._COMBINED_MESH.vertices,
                     normals=self._COMBINED_MESH.vertex_normals,
                     triangles=self._COMBINED_MESH.triangles)

  def get_pdb_string(self):
    """
    Obtain the pdb string the structures of all segments under vectorization
    """
    return str(self._PDB_STRING)

  def query_segments(self, threshold=2):
    """
    Generate a slice from all frame of a trajectory (Each frame takes one dimension)
    Returned results are the atomic coordinates and the atomic indices
    Args:
      threshold: the threshold of the distance between two segments
    Returns:
      xyz: atomic coordinates (N*3) within the bounding box (N atoms are in the bounding box)
      s_final: segment index (N*1) of each atom: like the following list [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4,
      4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    Note:
      The mask `states` indicates the atoms that are within the bounding box to the total number of atoms in the system
      Only returns the non-empty segments back to the main function
    """
    xyz = self.traj.xyz[self._frame]
    states = utils.filter_points_within_bounding_box(xyz, self._center, self._length, return_state=True)
    s_final = np.zeros(len(states), dtype=int)
    lastres = -999
    seg_counter = 0
    for idx, _state in enumerate(states):
      if _state:
        # Atom that is within the bounding box, Check if it belongs to the same segment
        if self.traj.atoms[idx].resid - lastres > threshold:
          seg_counter += 1
        s_final[idx] = seg_counter
        lastres = self.traj.atoms[idx].resid

    if not nearl._clear:
      with io.hdf_operator(f"{self.temp_prefix}segments.h5", "w") as hdf:
        hdf.create_dataset("xyz", data=xyz)
        hdf.create_dataset("xyz_b", data=xyz[s_final > 0])
        hdf.create_dataset("segments", data=s_final)
        hdf.create_dataset("segments_b", data=s_final[s_final > 0])
        hdf.create_dataset("box", data=np.array([*self._center, *self._length]))

    self._SEGMENTS = s_final
    # Empty segment with ID 0 not included
    self._SEGMENTS_NUMBER = min(len([i for i in set(s_final) if i != 0]), self.SEGMENT_LIMIT)
    # Order the segments from the most abundant to least ones (unique segments, 1 * self._SEGMENTS_NUMBER)
    self._SEGMENTS_ORDER = order_segments(s_final)[:self._SEGMENTS_NUMBER]
    return s_final

  def compute_segment_i(self, seg_indices):
    """
    Compute the segment information of a given segment
    Args:
      seg_indices: the segment indices of the desired segment
    """
    # Reset the segment information and embed all computation into the class
    self._SEGMENT = np.asarray(seg_indices)
    self._SEGRES_INDICES = np.array([])
    self._ATOM_COUNT = None
    self._SEGRES_MOL = None
    self._MESH = None
    self._VP_HIST = None

    # Generate the residue-based segment indices
    seg_res_mask = utils.get_residue_mask(self.traj, utils.get_mask_by_idx(self.traj, seg_indices))
    self._SEGRES_INDICES = self.traj.top.select(seg_res_mask)

    self._ATOM_COUNT = self.atom_type_count(self._SEGMENT)
    if len(self._ATOM_COUNT) >= 0:
      SUCCESS_COUNT = True
    else:
      SUCCESS_COUNT = False

    # Segment conversion to rdkit molecule
    self._SEGRES_MOL = self.segment_to_rdkit()
    if (self._SEGRES_MOL is None) or (not self._SEGRES_MOL):
      SUCCESS_MOLE = False
    else:
      SUCCESS_MOLE = True

    # Segment conversion to triangle mesh
    self._MESH = self.segment_to_mesh(seg_indices)
    if (self._MESH is None) or self._MESH.is_empty():
      SUCCESS_MESH = False
    else:
      SUCCESS_MESH = True

    self._VP_HIST = self.segment_to_vpc([1000,1000,1000])
    if self._VP_HIST is None:
      SUCCESS_VP = False
    else:
      SUCCESS_VP = True

    return [SUCCESS_COUNT, SUCCESS_MESH, SUCCESS_MOLE, SUCCESS_VP]

  def atom_type_count(self, theidxi=None):
    """
    Descriptor 1 and 2:
    Args:
      theidxi: the indices of the atoms in the segment
    Return:
      Atom counts as a dictionary
    """
    ATOM_DICT = {
      'H': 1, 'C': 6, 'N': 7, 'O': 8,
      'F': 9, 'P': 15, 'S': 16, 'Cl': 17,
      'Br': 35, 'I': 53
    }
    if theidxi is None:
      atomic_numbers = np.array([i.atomic_number for i in self.traj.atoms[self._SEGMENT]]).astype(int)
    else:
      atomic_numbers = np.array([i.atomic_number for i in self.traj.atoms[np.array(theidxi)]]).astype(int)
    counts = {}
    # Iterate through every atom type in the dictionary
    # Return every type of available atom types for method robustness
    for atom, atom_num in ATOM_DICT.items():
      at_count = np.count_nonzero(np.isclose(atomic_numbers, atom_num))
      counts[atom] = at_count
    return counts

  def segment_to_rdkit(self, theidxi=None):
    """
    Generate a rdkit molecule based on the residue-based segment
    """
    if theidxi is None:
      rdmol = utils.traj_to_rdkit(self.traj, self._SEGRES_INDICES, self.frame)
    else:
      rdmol = utils.traj_to_rdkit(self.traj, np.array(theidxi), self.frame)
    if rdmol is None:
      return None
    else:
      return rdmol

  def segment_to_mesh(self, theidxi=None, d=4, r=1.5):
    """
    Generate a mesh object from a segment of atoms
    Use ChimeraX's method to generate molecular surface
    Avoids the use of MSMS to generate intermediate .xyzr, .vert, .face files
    Args:
      d: density of the surface
      r: radius of the atoms
    """
    if theidxi is None:
      segment_indices = self._SEGMENT
    else:
      segment_indices = np.array(theidxi)

    resnames = np.array([a.name for a in self.traj.residues])
    rads = [getRadius(i, j) for i, j in [(a.name, resnames[a.resid]) for a in self.traj.atoms[segment_indices]]]

    # # TODO: integrate the density of points and radius of probe into the surface generation function
    # vertices, normals, faces = ses_surface_geometry(self.traj.xyz[self.frame][segment_indices], rads)
    # if nearl._debug:
    #   printit(f"From {len(rads)} atoms, returned vertices/normals/triangles: ", vertices.shape, normals.shape, faces.shape)
    #
    # if len(vertices) == 0:
    #   tmphash = utils.get_hash()[0:10]
    #   with open(nearl._tempfolder + f"/{tmphash}.xyzr", "w") as f:
    #     for (xyz, r) in zip(self.traj.xyz[self.frame][segment_indices], rads):
    #       f.write(f"{xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f} {r:.3f}\n")
    #
    # # Retrieve and post-process the mesh: vertices, normals, faces
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    # mesh.triangles = o3d.utility.Vector3iVector(faces)
    # mesh.remove_degenerate_triangles()
    # mesh.compute_vertex_normals()

    mesh = ses_surface_geometry(self.traj.xyz[self.frame][segment_indices], rads)

    if not mesh.is_empty():
      return mesh
    else:
      printit(f"Failed to convert the XYZR to triangle mesh.")
      return o3d.geometry.TriangleMesh()

  def segment_to_vpc(self, viewpoint=None, bins=None):
    """
    Compute the viewpoint feature of a segment
    """
    pf_gen = PointFeature(self._MESH)

    if self.standpoint == "next":
      if self.segment_index == (self._SEGMENTS_NUMBER - 1):
        standpoint_coord = self.center
      else:
        theidx = np.where(self._SEGMENTS == self._SEGMENTS_ORDER[self.segment_index + 1])[0]
        standpoint_coord = self.traj.xyz[self.frame][theidx].mean(axis=0)

    elif self.standpoint == "previous":
      if self.segment_index == (self._SEGMENTS_NUMBER - 1):
        standpoint_coord = self.center
      else:
        theidx = np.where(self._SEGMENTS == self._SEGMENTS_ORDER[self.segment_index - 1])[0]
        standpoint_coord = self.traj.xyz[self.frame][theidx].mean(axis=0)

    elif self.standpoint == "first":
      theidx = np.where(self._SEGMENTS == self._SEGMENTS_ORDER[1])[0]
      standpoint_coord = self.traj.xyz[self.frame][theidx].mean(axis=0)

    elif self.standpoint == "end":
      theidx = np.where(self._SEGMENTS == self._SEGMENTS_ORDER[-1])[0]
      standpoint_coord = self.traj.xyz[self.frame][theidx].mean(axis=0)

    elif self.standpoint == "self":
      theidx = np.where(self._SEGMENTS == self._SEGMENTS_ORDER[self.segment_index - 1])[0]
      standpoint_coord = self.traj.xyz[self.frame][theidx].mean(axis=0)

    elif self.standpoint == "absolute":
      standpoint_coord = self.standpoint_coord

    elif viewpoint is not None:
      standpoint_coord = np.array(viewpoint)
    self.standpoint_coord = standpoint_coord

    if bins is None:
      bin_nr = self.VIEWPOINTBINS
    else:
      bin_nr = bins

    if nearl._verbose:
      printit(f"Center mode: {self.standpoint}; Center coord: {standpoint_coord}; Bin number: {bin_nr}")

    vpc = pf_gen.compute_vpc(self.standpoint_coord, bins=bin_nr)
    if vpc is None or np.sum(vpc) == 0:
      return None
    else:
      return vpc

  def vectorize(self):
    """
    Vectorize the segments (at maximum 6) of a frame
    Two major steps:
      1. Generate the fingerprint for each segment (including PDB generation/Molecular surface generation)
      2. Collect all the results, combine surfaces and return it to the run_frame function
    """
    # Initialize the identity feature vector
    frame_feature = np.zeros((self.SEGMENT_LIMIT, 12 + self.VIEWPOINTBINS))
    pdb_final = ""
    segment_objects = []   # 3D objects for each segment
    atom_indices = []      # Atom indices for each segment
    atom_indices_res = []  # Atom indices for each segment (residue-based)
    segment_rdmols = []  # RDKit molecules for each segment

    # ITERATE the segments from the most abundant to least ones
    for seg_index, segi in enumerate(self.segments_order):
      # From the segment series to segment indices
      theidxi = np.where(self.segments == segi)[0]
      self.segment_index = seg_index
      if nearl._verbose:
        printit(f"{__name__:15s}: Processing the segment {seg_index + 1}/{self.segments_number} ...")
        printit(f"{__name__:15s}: Atom number by segment index: {len(theidxi)}; residue-based index: {len(self.segment_residue)}")
      ret_status = self.compute_segment_i(theidxi)

      if False in ret_status:
        printit(f"{__name__:15s}: Failed processing the segment {seg_index + 1}/{self.segments_number} of frame {self.frame} in {self.traj.top_filename}")
        printit(f"{__name__:15s}: Skip this segment ...")
        frame_feature[seg_index, :] = 0
        continue

      # ATOM types counts
      carbon_number = self.atom_count.get("C", 0)
      total_number = sum(self.atom_count.values())
      # For each segment, computer the partial charge, hydrogen bond, pseudo energy separately
      positive_charge, negative_charge = self.partial_charge()
      donor_number, acceptor_number = self.hbp_count()
      PE_lj, PE_el = self.pseudo_energy()
      # Point cloud-based descriptors
      surface_area = self.surface(self.mesh)
      enclosed_volume = self.volume(self.mesh)
      mean_radius = self.mean_radius(self.mesh)
      hull_ratio = self.convex_hull_ratio(self.mesh)

      frame_feature[seg_index, :12] = [
        total_number, carbon_number, donor_number, acceptor_number, positive_charge, negative_charge,
        PE_lj, PE_el, surface_area, enclosed_volume, mean_radius, hull_ratio
      ]
      # TODO:
      # Candidates for the structural descriptors:
      # 1. Longest path of the segment from one atom to another atom

      frame_feature[seg_index, -self.VIEWPOINTBINS:] = self.vpc

      if nearl._verbose:
        printit(f"{__name__:15s}: Segment {seg_index + 1}/{self.segments_number} computation succeeded.")
        printit(f"{__name__:15s}: Carbon number: {self.atom_count.get('C', 0)}; Total number: {sum(self.atom_count.values())}")
        printit(f"{__name__:15s}: Partial charge: (Positive: {positive_charge}/Negative: {negative_charge})")
        printit(f"{__name__:15s}: Hydrogen bond: (Donor: {donor_number}/Acceptor: {acceptor_number})")
        printit(f"{__name__:15s}: Pseudo energy: (LJ: {PE_lj}, Electrostatic: {PE_el})")
        printit(f"{__name__:15s}: Surface area: {surface_area}, Enclosed volume: {enclosed_volume}, Mean radius: {mean_radius}, Hull ratio: {hull_ratio}")
        printit(f"{__name__:15s}: VPC info -> Sum: {self.vpc.sum()}; Max: {self.vpc.max()}; Min: {self.vpc.min()}")

      # Mesh post-processing, pass a deep copy to the segment_objects because Python only passes the reference
      self.mesh.paint_uniform_color(SEGMENT_CMAPS[seg_index])
      segment_objects.append(copy.deepcopy(self.mesh))

      # PDB string processing
      atom_indices += self.segment_residue.tolist()
      pdbstr = utils.write_pdb_block(self.traj, self.segment_residue, frame_index=self.frame)
      # new_lines = [line for line in pdbstr.strip("\n").split('\n') if ("^END$" not in line and "CONECT" not in line)]
      new_lines = [line for line in pdbstr.strip("\n").split("\n")]
      pdb_final += ('\n'.join(new_lines) + "\n")
      segment_rdmols.append(self.mol)
      atom_indices_res += self.segment_residue.tolist()

    # END of the segment iteration
    try:
      for idx, mesh in enumerate(segment_objects):
        if idx == 0:
          combined_mesh = mesh
        else:
          combined_mesh += mesh
      if nearl._verbose:
        printit("Final 3D object: ", combined_mesh)
      self._COMBINED_MESH = combined_mesh
    except:
      printit(f"Warning: {self.traj.top_filename} -> Failed to combine the segment meshes for frame {self.frame} in {self.traj.top_filename}")
      printit(f"Warning: {self.traj.top_filename} -> Non-zero feature number: {np.count_nonzero(frame_feature)} ; frame index: {self.frame}; ")
      printit(f"Warning: {self.traj.top_filename} -> PDB_string: {pdb_final} ; Mesh objects: {segment_objects}")
      printit(f"Warning: {self.traj.top_filename} -> Skip this frame ...")
      with open(f"{self.temp_prefix}frame{self.frame}_FAILED.pdb", "w") as f:
        f.write(pdb_final)
      return np.zeros((self.SEGMENT_LIMIT, 12 + self.VIEWPOINTBINS)).reshape(-1)

    # Keep the final PDB and PLY files in memory for further use
    self._RDMOLS = segment_rdmols
    self._PDB_STRING = pdb_final
    self._INDICES = atom_indices
    self._INDICES_RES = atom_indices_res
    self._MESHES = segment_objects

    if not nearl._clear:
      # Write out the final mesh if the intermediate output is required for debugging purpose
      # Reset the file prefix to make the temporary output file organized
      with open(f"{self.temp_prefix}frame{self.frame}.pdb", "w") as f:
        f.write(self.get_pdb_string())
      with open(f"{self.temp_prefix}frame{self.frame}.ply", "w") as f:
        f.write(self.get_ply_string())

    return frame_feature.reshape(-1)

  def hbp_count(self):
    """
    Descriptor 3 and 4: Counter hydrogen bond donor and acceptors.
    Args:
      theidxi: the indices of the atoms in the segment
    Return: 
      number_d: Number of hydrogen bond donor
      number_a: Number of hydrogen bond acceptor
    """
    conf = self.mol.GetConformer()
    d_matches = self.mol.GetSubstructMatches(RD_DONOR_PATTERN)
    a_matches = self.mol.GetSubstructMatches(RD_ACCEPTOR_PATTERN)

    # Get all atom positions and store in a numpy array outside the loop
    all_positions = np.array(conf.GetPositions())

    # Convert bounds checks to numpy operations
    d_coords = all_positions[[d[0] for d in d_matches]]  # Gather all d coordinates at once
    a_coords = all_positions[[a[0] for a in a_matches]]  # Gather all a coordinates at once
    d_within = np.sum(np.all((d_coords > self.lower_bound) & (d_coords < self.upper_bound), axis=1))
    a_within = np.sum(np.all((a_coords > self.lower_bound) & (a_coords < self.upper_bound), axis=1))
    return d_within, a_within

  def partial_charge(self):
    """
    Descriptor 5 and 6: Counter positive partial charge and negative partial charge.
    Return: 
      charge_p: Number of positive partial charge
      charge_n: Number of negative partial charge
    """
    conf = self.mol.GetConformer()

    all_positions = np.array(conf.GetPositions())
    all_charges = np.array([atom.GetDoubleProp('_GasteigerCharge') for atom in self.mol.GetAtoms()])

    mask = np.all((all_positions > self.lower_bound) & (all_positions < self.upper_bound), axis=1)
    # NOTE: there might be NaN values in the partial charge returned by RDKit
    charge_p = np.sum(all_charges[mask & (all_charges > 0)])
    charge_n = np.sum(all_charges[mask & (all_charges < 0)])
    return charge_p, charge_n

  def pseudo_energy(self):
    """
    Descriptor 7 and 8: Compute the pseudo-lj and pseudo-elec potential
    Return: 
      pp_lj: Pseudo Lenar-Jones potential
      pp_elec: Pseudo Electrostatic potential
    """
    # Compute the pseudo-lj and pseudo-elec potential constants
    K_ELEC = 8.98
    EPSILON_LJ = 1
    SIGMA_LJ = 1
    atomnr = self.mol.GetNumAtoms()
    conf = self.mol.GetConformer()
    positions = conf.GetPositions()
    charges = np.array([atom.GetDoubleProp('_GasteigerCharge') for atom in self.mol.GetAtoms()])

    if atomnr != len(positions) or atomnr != len(charges):
      # Returns 0 for both pp_lj and pp_elec if the atom number mismatch
      print(f"DEBUG: Atom number mismatch in pseudo_energy for frame {self.frame}")
      return 0, 0

    # Now use these pair-wise values to calculate pseudo potentials
    distances = pdist(positions)                  # Pair-wise distances
    distances = squareform(distances)             # Square-form the distance to make it compatible with charge pairs
    charge_products = np.outer(charges, charges)  # Calculate all pair-wise charges product
    mask = np.triu_indices_from(distances, k=1)   # Only upper triangle of the matrix is needed
    pp_lj = 4 * EPSILON_LJ * np.sum(((SIGMA_LJ / distances[mask]) ** 12 - (SIGMA_LJ / distances[mask]) ** 6))
    pp_elec = K_ELEC * np.sum(charge_products[mask] / distances[mask])
    return pp_lj, pp_elec

  # Descriptor 9 and 10: Surface area and volume of the mesh
  def volume(self, mesh): 
    """
    Volume computation is not robust enough
    """
    try:
      VOL = mesh.get_volume()
    except:
      import  nearl.static.interpolate_c as interpolate_cpu
      vertices = np.asarray(mesh.vertices)
      bound = np.max(vertices, axis=0) - np.min(vertices, axis=0)
      voxel_size = np.max(bound) / 36
      for i in range(10):
        volume_info = interpolate_cpu.compute_volume(vertices, voxel_size=voxel_size)
        VOL_estimate = volume_info[0] + (0.5*volume_info[1])
        if volume_info[3] > 0 and volume_info[3] > volume_info[4]:
          if nearl._verbose or nearl._debug:
            printit(f"Computation success")
          break
        elif volume_info[7] > 72:
          if nearl._verbose or nearl._debug:
            printit("Grid too dense")
          voxel_size = voxel_size * 1.5
        elif volume_info[7] < 12:
          if nearl._verbose or nearl._debug:
            printit("Grid too sparse")
          voxel_size = voxel_size * 0.5
        elif volume_info[3] == 0:
          if nearl._verbose or nearl._debug:
            printit("No voxel is inside the mesh (usually voxel size too small or mesh resolution too low)")
          voxel_size = voxel_size * 1.1 + 0.1
        elif volume_info[3] > 0 and volume_info[3] < volume_info[4]:
          if nearl._verbose or nearl._debug:
            printit("Majority of voxels are surface voxels (usually voxel size is too large)")
          voxel_size = voxel_size * 0.9 - 0.1
        else:
          voxel_size *= 1.1
      if volume_info[0] == 0:
        VOL_estimate = 4 * np.pi * self.mean_radius(mesh) ** 3 / 3
      VOL = VOL_estimate
    return VOL
    
  def surface(self, mesh):
    """
    Surface area computation
    Args:
      mesh: open3d.geometry.TriangleMesh
    """
    return mesh.get_surface_area()
  
  def mean_radius(self, mesh):
    """
    Down sample the mesh uniformly and compute the mean radius from the point cloud to the geometric center
    Args:
      mesh: open3d.geometry.TriangleMesh
    """
    pcd = mesh.sample_points_uniformly(self.FPFH_DOWN_SAMPLES)
    mean_radius = np.linalg.norm(np.asarray(pcd.points) - pcd.get_center(), axis=1).mean()
    return mean_radius
  
  def convex_hull_ratio(self, mesh):
    """
    Down sample the mesh uniformly and compute the convex hull ratio
    Args:
      mesh: open3d.geometry.TriangleMesh
    """
    pcd = mesh.sample_points_uniformly(self.FPFH_DOWN_SAMPLES)
    hull, _ = pcd.compute_convex_hull()
    hull_ratio = len(hull.vertices)/self.FPFH_DOWN_SAMPLES
    return hull_ratio
  
  def fpfh_down(self, mesh, origin=True):
    """
    Down sample the mesh uniformly and compute the fpfh feature
    TODO: add support for the voxel-base down sampling 
    """
    relative = bool(not origin)
    mesh_copy = copy.deepcopy(mesh)
    mesh_copy.translate([0,0,0], relative=relative)
    ktree = o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=20)
    mesh_down = mesh_copy.sample_points_uniformly(self.FPFH_DOWN_SAMPLES)
    mesh_down.estimate_normals(ktree)
    fpfh_down = o3d.pipelines.registration.compute_fpfh_feature(mesh_down, ktree)
    return fpfh_down.data


def object_meta(obj):
  points = np.asarray(obj.vertices).round(3)
  normals = np.asarray(obj.vertex_normals).round(3)
  colors = np.asarray(obj.vertex_colors)*256
  colors = colors.astype(int)
  triangles = np.asarray(obj.triangles).astype(int)
  return points.reshape(-1), normals.reshape(-1), colors.reshape(-1), triangles.reshape(-1)


def order_segments(lst):
  """
  Process the segments and order them by the number of atoms
  """
  from collections import Counter
  counter = Counter(lst)
  sorted_elements = sorted(counter, key=lambda x: counter[x], reverse=True)
  if 0 in sorted_elements:
    sorted_elements.remove(0)
  return sorted_elements


class PointFeature(object):
  # TODO: setup the points to down sample
  def __init__(self, obj_3d):
    self._obj = obj_3d
    if "vertices" in dir(self._obj):
      self._pcd = np.array(self._obj.vertices)
    elif "points" in dir(self._obj):
      self._pcd = np.array(self._obj.points)
    self._norm = np.array(self._obj.vertex_normals)
    self._kdtree = spatial.KDTree(self._pcd)

  @property
  def pcd(self):
    return self._pcd
  @property
  def norm(self):
    return self._norm

  def self_vpc(self, bins=128):
    cos_angles = [np.dot(n, d/np.linalg.norm(d)) for n,d in zip(self.norm, self.pcd-self.pcd.mean(axis=0))]
    angles = np.arccos(cos_angles)
    hist, _ = np.histogram(angles, bins=self.VIEWPOINTBINS, range=(0, np.pi))
    hist_normalized = hist / np.sum(hist)
    hist_normalized = np.asarray([i if not np.isnan(i) else 0 for i in hist_normalized])
    return hist_normalized

  def compute_vpc(self, viewpoint, bins=128):
    # Compute the relative position of the viewpoint to the center of the point cloud
    rel_vp = np.asarray(viewpoint) - self.pcd.mean(axis=0)
    rel_vp_normalized = rel_vp / np.linalg.norm(rel_vp)

    # Calculate the angle between the normals and the relative viewpoint vectors
    cos_angles = np.dot(self.norm, rel_vp_normalized)
    angles = np.arccos(cos_angles)

    # Create the viewpoint component histogram
    hist, _ = np.histogram(angles, bins=bins, range=(0, np.pi))
    hist_normalized = hist / np.sum(hist)
    hist_normalized = np.asarray([i if not np.isnan(i) else 0 for i in hist_normalized])
    return hist_normalized


def compute_convex(obj, n_points=600):
  if isinstance(obj, o3d.geometry.TriangleMesh):
    pcd = obj.sample_points_uniformly(n_points)
  elif isinstance(obj, o3d.geometry.PointCloud):
    pcd = obj.voxel_down_sample(0.01)
  hull, _ = pcd.compute_convex_hull()
  hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
  hull_ls.paint_uniform_color([0.77, 0, 1])
  return (pcd, hull_ls)



def cosine_similarity(a, b):
  """
  Compute the cosine similarity between two vectors
  Args:
    a: np.array
    b: np.array
  """
  if np.linalg.norm(a) * np.linalg.norm(b) < 0.0001:
    return 0.0
  else:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_similarity(array1, array2, weights=[1, 0.5, 0.15, 0.1, 0.1, 0.05]):
  """
  Compute the similarity between two arrays
  Args:
    array1: np.array
    array2: np.array
  """
  if array1.shape != array2.shape:
    print(f"Two arrays have different shapes: {array1.shape} and {array2.shape}")
    return False
  else:
    array1 = array1.ravel().reshape(6, -1)
    array2 = array2.ravel().reshape(6, -1)
    weights = np.asarray(weights)
    contrib_chem = np.array([cosine_similarity(array1[i, :12].ravel(), array2[i, :12].ravel()) for i in range(6)])
    contrib_viewpoint = np.array([cosine_similarity(array1[i, 12:].ravel(), array2[i, 12:].ravel()) for i in range(6)])

    # for i in range(6):
    #   _cos = cossim(array1[i, :12].ravel(), array2[i, :12].ravel())
    #   print(f"====> session {i}", _cos, array1[i, :12].ravel(), array2[i, :12].ravel())

    similarities = (contrib_chem + contrib_viewpoint) / 2 * weights / sum(weights[np.nonzero(contrib_chem)])
    # print(similarities, contrib_viewpoint.round(2), contrib_chem.round(2))

    print(f"{'Chem contribution':20s}: ", ''.join(f'{i:6.2f}' for i in contrib_chem))
    print(f"{'VP contribution':20s}: ", ''.join(f'{i:6.2f}' for i in contrib_viewpoint))
    print(f"{'Weights':20s}: ", ''.join(f'{i:6.2f}' for i in weights))
    print(f"{'Contribution(real)':20s}: ", ''.join(f'{i:6.2f}' for i in similarities))
    print(f"{'Final Similarity':20s}: ", sum(similarities), "\n")
    return sum(similarities)

def weight(array1):
  array1 = array1.ravel().reshape(6, -1)
  return array1[:, 0].ravel()

######################################################################
########## ChimeraX's function to compute surface from XYZR ##########
######################################################################

def invert_matrix(tf):
  tf = np.asarray(tf)
  r = tf[:, :3]
  t = tf[:, 3]
  tfinv = np.zeros((3, 4), np.float64)
  rinv = tfinv[:, :3]
  tinv = tfinv[:, 3]
  rinv[:, :] = np.linalg.inv(r)
  tinv[:] = np.dot(rinv, -t)
  return tfinv

def affine_transform_vertices(vertex_positions, tf):
  tf_rot = tf[:, :3]
  tf_trans = tf[:, 3]
  # Use NumPy's broadcasting capabilities to perform the matrix multiplication
  vertex_positions_transformed = np.dot(vertex_positions, tf_rot.T) + tf_trans
  return vertex_positions_transformed

def reduce_geometry(va, na, ta, vi, ti):
  vmap = np.zeros(len(va), np.int32)
  rva = va.take(vi, axis=0)
  rna = na.take(vi, axis=0)
  rta = ta.take(ti, axis=0)
  # Remap triangle vertex indices to use shorter vertex list.
  vmap[vi] = np.arange(len(vi), dtype=vmap.dtype)
  rta = vmap.take(rta.ravel()).reshape((len(ti), 3))
  return rva, rna, rta

def ses_surface_geometry(xyz, radii, grid_spacing=0.3):
  # TODO need to implement the new SiESTA-based surface generation
  from nearl.static import surface
  xyzr_array = np.zeros((len(xyz), 4), np.float32)
  xyzr_array[:, :3] = xyz
  xyzr_array[:, 3] = radii
  result_tuple = surface.get_surf(xyzr_array, grid_size=grid_spacing, smooth_step = 3, slice_number = 400)

  if len(result_tuple) != 2:
    print("Error: surface.get_surf() returned unexpected number of values")
    return o3d.geometry.TriangleMesh()
  newmesh = o3d.geometry.TriangleMesh()
  newmesh.vertices = o3d.utility.Vector3dVector(result_tuple[0])
  newmesh.triangles = o3d.utility.Vector3iVector(result_tuple[1])
  newmesh.remove_degenerate_triangles()
  newmesh.compute_vertex_normals()
  return newmesh





  ######################################################################
  # Used to be in the featurizer for fingerprint generation
  # def run_frame(self, centers, fp_generator):
  #   """
  #   Generate the feature vectors for each center in the current frame
  #   Explicitly transfer the generator object to the function
  #   Needs to correctly set the box of the fingerprint.generator by desired center and lengths
  #   Args:
  #     centers: list, a list of 3D coordinates
  #     fp_generator: fingerprint.generator, the generator object
  #   """
  #   # Step1: Initialize the identity vector and feature vector
  #   centernr = len(centers)
  #   repr_vector = np.zeros((centernr, 6 * self.VPBINS))
  #   feat_vector = np.zeros((centernr, self.FEATURENUMBER)).tolist()
  #   mask = np.ones(centernr).astype(bool)  # Mask failed centers for run time rubustness

  #   fp_generator.frame = self.active_frame_index
  #   if _verbose:
  #     printit(f"Expected to generate {centernr} fingerprint anchors")

  #   # Compute the one-time-functions of each feature
  #   for feature in self.FEATURESPACE:
  #     feature.before_focus()

  #   # Step2: Iterate each center
  #   for idx, center in enumerate(centers):
  #     # Reset the focus of representation generator
  #     self.center = center
  #     fp_generator.set_box(self.center, self.lengths)
  #     # Segment the box and generate feature vectors for each segment
  #     segments = fp_generator.query_segments()

  #     # DEBUG ONLY
  #     if _verbose or _debug:
  #       seg_set = set(segments)
  #       seg_set.discard(0)
  #       printit(f"Found {len(seg_set)} non-empty segments", {i: j for i, j in zip(*np.unique(segments, return_counts=True)) if i != 0})

  #     """
  #     Compute the identity vector for the molecue block
  #     Identity generation is compulsory because it is the only hint to retrieve the feature block
  #     """
  #     feature_vector = fp_generator.vectorize()

  #     try:
  #       feature_vector.sum()
  #     except:
  #       print("Error: Feature vector is not generated correctly")
  #       print(feature_vector)
  #       print(feature_vector[0].shape)

  #     if np.count_nonzero(feature_vector) == 0 or None in fp_generator.mols:
  #       # Returned feature vector is all-zero, the featurization is most likely failed
  #       mask[idx] = False
  #       continue

  #     # Collect the results before processing the features
  #     # Use the dictionary as the result container to improve the flexibility (rather than attributes)
  #     self.contents = {
  #       "meshes": fp_generator.meshes,
  #       "final_mesh": fp_generator.final_mesh,
  #       "vertices": fp_generator.vertices,
  #       "faces": fp_generator.faces,
  #       "normals": fp_generator.normals,

  #       "segments": segments,
  #       "indices": fp_generator.indices,
  #       "indices_res": fp_generator.indices_res,
  #       "mols": fp_generator.mols,

  #       "pdb_result": fp_generator.get_pdb_string(),
  #       "ply_result": fp_generator.get_ply_string(),
  #     }

  #     if len(feature_vector) == 0:
  #       if _verbose:
  #         printit(f"Center {center} has no feature vector")
  #       mask[idx] = False
  #       continue
  #     repr_vector[idx] = feature_vector

  #     # Step2.5: Iterate different features
  #     for fidx, feature in enumerate(self.FEATURESPACE):
  #       feat_arr = feature.featurize()
  #       if isinstance(feat_arr, np.ndarray) and isinstance(feat_arr.dtype, (int, np.int32, np.int64,
  #         float, np.float32, np.float64, complex, np.complex64, np.complex128)):
  #         feat_arr = np.nan_to_num(feat_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
  #       feat_vector[idx][fidx] = feat_arr

  #     # Clear the result contents
  #     self.contents = {}
  #   """
  #   Step3: Remove the masked identity vector and feature vector
  #   Final size of the feature vector: (number of centers, number of features)
  #   """
  #   ret_repr_vector = repr_vector[mask]
  #   ret_feat_vector = [item for item, use in zip(feat_vector, mask) if use]

  #   # Compute the one-time-functions of each feature
  #   for feature in self.FEATURESPACE:
  #     feature.after_focus()

  #   # DEBUG ONLY: After the iteration, check the shape of the feature vectors
  #   if _verbose:
  #     printit(f"Result identity vector: {ret_repr_vector.shape} ; Feature vector: {ret_feat_vector.__len__()} - {ret_feat_vector[0].__len__()}")
  #   return ret_repr_vector, ret_feat_vector


