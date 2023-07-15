import os, sys, re, time, subprocess, tempfile, datetime, copy, functools
import pytraj as pt
import numpy as np 
import open3d as o3d
from scipy import spatial
import multiprocessing as mp
from itertools import combinations
from scipy.spatial.distance import cdist, pdist, squareform
from rdkit import Chem
from . import utils, chemtools
from . import CONFIG, printit

_clear = CONFIG.get("clear", False);
_verbose = CONFIG.get("verbose", False);
_tempfolder = CONFIG.get("tempfolder", "/tmp");
_debug = CONFIG.get("debug", False);

_usegpu = CONFIG.get("usegpu", False);
if _usegpu:
  import cupy as cp

ACCEPTOR_PATTERN = '[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'
DONOR_PATTERN = "[!H0;#7,#8,#9]"
RD_DONOR_PATTERN = Chem.MolFromSmarts(DONOR_PATTERN);
RD_ACCEPTOR_PATTERN = Chem.MolFromSmarts(ACCEPTOR_PATTERN);

ATOM_PATTERNS = {0: '^[0-9]*H.*$', 1: '^[0-9]*D.*$', 2: '^O.*$', 3: '^CA$', 4: '^CD$', 5: '^CD  $', 6: '^CA$', 7: '^N$', 8: '^CA$', 9: '^C$', 10: '^O$', 11: '^P$', 12: '^CB$', 13: '^CB$', 14: '^CB$', 15: '^CG$', 16: '^CG$', 17: '^CG$', 18: '^CG$', 19: '^O1$', 20: '^O2$', 21: '^CH3$', 22: '^CD$', 23: '^NE$', 24: '^RE$', 25: '^CZ$', 26: '^NH[12][AB]?$', 27: '^RH[12][AB]?$', 28: '^OD1$', 29: '^ND2$', 30: '^AD1$', 31: '^AD2$', 32: '^OD[12][AB]?$', 33: '^ED[12][AB]?$', 34: '^OD1[AB]?$', 35: '^ND2$', 36: '^AD1$', 37: '^AD2$', 38: '^OD2$', 39: '^LP[12]$', 40: '^SG$', 41: '^SG$', 42: '^OE[12][AB]?$', 43: '^EE[12][AB]?$', 44: '^CD$', 45: '^OE1$', 46: '^NE2$', 47: '^AE[12]$', 48: '^CE1|CD2$', 49: '^ND1$', 50: '^ND1$', 51: '^RD1$', 52: '^NE2$', 53: '^RE2$', 54: '^NE2$', 55: '^RE2$', 56: '^A[DE][12]$', 57: '^CG1$', 58: '^CG2$', 59: '^CD|CD1$', 60: '^CD1$', 61: '^CD2$', 62: '^C[GDE]$', 63: '^NZ$', 64: '^KZ$', 65: '^SD$', 66: '^CE$', 67: '^C[DE][12]$', 68: '^CZ$', 69: '^C[GD]$', 70: '^SE$', 71: '^SEG$', 72: '^OD1$', 73: '^OD2$', 74: '^OG$', 75: '^OG1$', 76: '^CG2$', 77: '^CD1$', 78: '^CD2$', 79: '^CE2$', 80: '^NE1$', 81: '^CE3$', 82: '^CZ2$', 83: '^CZ3$', 84: '^CH2$', 85: '^C[DE][12]$', 86: '^CZ$', 87: '^OH$', 88: '^CG1$', 89: '^CG2$', 90: '^CD$', 91: '^CE$', 92: '^FE[1-7]$', 93: '^S[1-7]$', 94: '^OXO$', 95: '^FE1$', 96: '^FE2$', 97: '^O1$', 98: '^O2$', 99: '^FE$', 100: '^CH[A-D]$', 101: '^N[A-D]$', 102: '^N [A-D]$', 103: '^C[1-4][A-D]$', 104: '^CM[A-D]$', 105: '^C[AB][AD]$', 106: '^CG[AD]$', 107: '^O[12][AD]$', 108: '^C[AB][BC]$', 109: '^OH2$', 110: '^N[123]$', 111: '^C1$', 112: '^C2$', 113: '^C3$', 114: '^C4$', 115: '^C5$', 116: '^C6$', 117: '^O7$', 118: '^O8$', 119: '^S$', 120: '^O[1234]$', 121: '^O[1234]$', 122: '^O4$', 123: '^P1$', 124: '^O[123]$', 125: '^C[12]$', 126: '^N1$', 127: '^C[345]$', 128: '^BAL$', 129: '^POI$', 130: '^DOT$', 131: '^CU$', 132: '^ZN$', 133: '^MN$', 134: '^FE$', 135: '^MG$', 136: '^MN$', 137: '^CO$', 138: '^SE$', 139: '^YB$', 140: '^N1$', 141: '^C[2478]$', 142: '^O2$', 143: '^N3$', 144: '^O4$', 145: '^C[459]A$', 146: '^N5$', 147: '^C[69]$', 148: '^C[78]M$', 149: '^N10$', 150: '^C10$', 151: '^C[12345]\\*$', 152: '^O[234]\\*$', 153: '^O5\\*$', 154: '^OP[1-3]$', 155: '^OT1$', 156: '^C01$', 157: '^C16$', 158: '^C14$', 159: '^C.*$', 160: '^SEG$', 161: '^OXT$', 162: '^OT.*$', 163: '^E.*$', 164: '^S.*$', 165: '^C.*$', 166: '^A.*$', 167: '^O.*$', 168: '^N.*$', 169: '^R.*$', 170: '^K.*$', 171: '^P[A-D]$', 172: '^P.*$', 173: '^.O.*$', 174: '^.N.*$', 175: '^.C.*$', 176: '^.P.*$', 177: '^.H.*$'}
RESIDUE_PATTERNS = {0: '^.*$', 1: '^.*$', 2: '^WAT|HOH|H2O|DOD|DIS$', 3: '^CA$', 4: '^CD$', 5: '^.*$', 6: '^ACE$', 7: '^.*$', 8: '^.*$', 9: '^.*$', 10: '^.*$', 11: '^.*$', 12: '^ALA$', 13: '^ILE|THR|VAL$', 14: '^.*$', 15: '^ASN|ASP|ASX|HIS|HIP|HIE|HID|HISN|HISL|LEU|PHE|TRP|TYR$', 16: '^ARG|GLU|GLN|GLX|MET$', 17: '^LEU$', 18: '^.*$', 19: '^GLN$', 20: '^GLN$', 21: '^ACE$', 22: '^ARG$', 23: '^ARG$', 24: '^ARG$', 25: '^ARG$', 26: '^ARG$', 27: '^ARG$', 28: '^ASN$', 29: '^ASN$', 30: '^ASN$', 31: '^ASN$', 32: '^ASP$', 33: '^ASP$', 34: '^ASX$', 35: '^ASX$', 36: '^ASX$', 37: '^ASX$', 38: '^ASX$', 39: '^CYS|MET$', 40: '^CY[SXM]$', 41: '^CYH$', 42: '^GLU$', 43: '^GLU$', 44: '^GLU|GLN|GLX$', 45: '^GLN$', 46: '^GLN$', 47: '^GLN|GLX$', 48: '^HIS|HID|HIE|HIP|HISL$', 49: '^HIS|HIE|HISL$', 50: '^HID|HIP$', 51: '^HID|HIP$', 52: '^HIS|HIE|HIP$', 53: '^HIS|HIE|HIP$', 54: '^HID|HISL$', 55: '^HID|HISL$', 56: '^HIS|HID|HIP|HISD$', 57: '^ILE$', 58: '^ILE$', 59: '^ILE$', 60: '^LEU$', 61: '^LEU$', 62: '^LYS$', 63: '^LYS$', 64: '^LYS$', 65: '^MET$', 66: '^MET$', 67: '^PHE$', 68: '^PHE$', 69: '^PRO|CPR$', 70: '^CSO$', 71: '^CSO$', 72: '^CSO$', 73: '^CSO$', 74: '^SER$', 75: '^THR$', 76: '^THR$', 77: '^TRP$', 78: '^TRP$', 79: '^TRP$', 80: '^TRP$', 81: '^TRP$', 82: '^TRP$', 83: '^TRP$', 84: '^TRP$', 85: '^TYR$', 86: '^TYR$', 87: '^TYR$', 88: '^VAL$', 89: '^VAL$', 90: '^.*$', 91: '^.*$', 92: '^FS[34]$', 93: '^FS[34]$', 94: '^FS3$', 95: '^FEO$', 96: '^FEO$', 97: '^HEM$', 98: '^HEM$', 99: '^HEM$', 100: '^HEM$', 101: '^HEM$', 102: '^HEM$', 103: '^HEM$', 104: '^HEM$', 105: '^HEM$', 106: '^HEM$', 107: '^HEM$', 108: '^HEM$', 109: '^HEM$', 110: '^AZI$', 111: '^MPD$', 112: '^MPD$', 113: '^MPD$', 114: '^MPD$', 115: '^MPD$', 116: '^MPD$', 117: '^MPD$', 118: '^MPD$', 119: '^SO4|SUL$', 120: '^SO4|SUL$', 121: '^PO4|PHO$', 122: '^PC$', 123: '^PC$', 124: '^PC$', 125: '^PC$', 126: '^PC$', 127: '^PC$', 128: '^BIG$', 129: '^POI$', 130: '^DOT$', 131: '^.*$', 132: '^.*$', 133: '^.*$', 134: '^.*$', 135: '^.*$', 136: '^.*$', 137: '^.*$', 138: '^.*$', 139: '^.*$', 140: '^FMN$', 141: '^FMN$', 142: '^FMN$', 143: '^FMN$', 144: '^FMN$', 145: '^FMN$', 146: '^FMN$', 147: '^FMN$', 148: '^FMN$', 149: '^FMN$', 150: '^FMN$', 151: '^FMN$', 152: '^FMN$', 153: '^FMN$', 154: '^FMN$', 155: '^ALK|MYR$', 156: '^ALK|MYR$', 157: '^ALK$', 158: '^MYR$', 159: '^ALK|MYR$', 160: '^.*$', 161: '^.*$', 162: '^.*$', 163: '^.*$', 164: '^.*$', 165: '^.*$', 166: '^.*$', 167: '^.*$', 168: '^.*$', 169: '^.*$', 170: '^.*$', 171: '^.*$', 172: '^.*$', 173: '^FAD|NAD|AMX|APU$', 174: '^FAD|NAD|AMX|APU$', 175: '^FAD|NAD|AMX|APU$', 176: '^FAD|NAD|AMX|APU$', 177: '^FAD|NAD|AMX|APU$'}
EXP_RADII = {1: 1.4, 2: 1.4, 3: 1.4, 4: 1.54, 5: 1.54, 6: 1.54, 7: 1.74, 8: 1.74, 9: 1.74, 10: 1.74, 11: 1.74, 12: 1.8, 13: 1.8, 14: 1.54, 15: 1.2, 16: 0.0, 17: 5.0, 18: 1.97, 19: 1.4, 20: 1.4, 21: 1.3, 22: 1.49, 23: 0.01, 24: 1.2, 25: 1.24, 26: 1.6, 27: 1.24, 28: 1.25, 29: 2.15, 30: 3.0, 31: 1.15, 38: 1.8}
UNITED_RADII = {1: 1.4, 2: 1.6, 3: 1.4, 4: 1.7, 5: 1.8, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0, 10: 1.74, 11: 1.86, 12: 1.85, 13: 1.8, 14: 1.54, 15: 1.2, 16: 1.5, 17: 5.0, 18: 1.97, 19: 1.4, 20: 1.4, 21: 1.3, 22: 1.49, 23: 0.01, 24: 0.0, 25: 1.24, 26: 1.6, 27: 1.24, 28: 1.25, 29: 2.15, 30: 3.0, 31: 1.15, 38: 1.8}
ATOM_NUM = {0: 15, 1: 15, 2: 2, 3: 18, 4: 22, 5: 22, 6: 9, 7: 4, 8: 7, 9: 10, 10: 1, 11: 13, 12: 9, 13: 7, 14: 8, 15: 10, 16: 8, 17: 7, 18: 8, 19: 3, 20: 3, 21: 9, 22: 8, 23: 4, 24: 4, 25: 10, 26: 5, 27: 5, 28: 1, 29: 5, 30: 3, 31: 3, 32: 3, 33: 3, 34: 1, 35: 5, 36: 3, 37: 3, 38: 3, 39: 13, 40: 13, 41: 12, 42: 3, 43: 3, 44: 10, 45: 1, 46: 5, 47: 3, 48: 11, 49: 14, 50: 4, 51: 4, 52: 4, 53: 4, 54: 14, 55: 14, 56: 4, 57: 8, 58: 9, 59: 9, 60: 9, 61: 9, 62: 8, 63: 6, 64: 6, 65: 13, 66: 9, 67: 11, 68: 11, 69: 8, 70: 9, 71: 9, 72: 3, 73: 3, 74: 2, 75: 2, 76: 9, 77: 11, 78: 10, 79: 10, 80: 4, 81: 11, 82: 11, 83: 11, 84: 11, 85: 11, 86: 10, 87: 2, 88: 9, 89: 9, 90: 8, 91: 8, 92: 21, 93: 13, 94: 1, 95: 21, 96: 21, 97: 1, 98: 1, 99: 21, 100: 11, 101: 14, 102: 14, 103: 10, 104: 9, 105: 8, 106: 10, 107: 3, 108: 11, 109: 2, 110: 14, 111: 9, 112: 10, 113: 8, 114: 7, 115: 9, 116: 9, 117: 2, 118: 2, 119: 13, 120: 3, 121: 3, 122: 3, 123: 13, 124: 3, 125: 8, 126: 14, 127: 9, 128: 17, 129: 23, 130: 23, 131: 20, 132: 19, 133: 24, 134: 25, 135: 26, 136: 27, 137: 28, 138: 29, 139: 31, 140: 4, 141: 10, 142: 1, 143: 14, 144: 1, 145: 10, 146: 4, 147: 11, 148: 9, 149: 4, 150: 10, 151: 8, 152: 2, 153: 3, 154: 3, 155: 3, 156: 10, 157: 9, 158: 9, 159: 8, 160: 9, 161: 3, 162: 3, 163: 3, 164: 13, 165: 7, 166: 11, 167: 1, 168: 4, 169: 4, 170: 6, 171: 13, 172: 13, 173: 1, 174: 4, 175: 7, 176: 13, 177: 15}

# Color map for the segments of the molecule block
__color_number = CONFIG.get("SEGMENT_LIMIT", 6);
if CONFIG.get("segment_colormap", None):
  from matplotlib.pyplot import get_cmap
  cmap = get_cmap(CONFIG.get("segment_colormap"));
  SEGMENT_CMAPS = [cmap(i)[:3] for i in range(int(0.1 * cmap.N), int(0.9 * cmap.N), int(0.9 * cmap.N) // 10)];
else:
  # Default color map -> inferno
  if CONFIG.get("SEGMENT_LIMIT", 6) == 6:
    SEGMENT_CMAPS = [
      [0.087411, 0.044556, 0.224813],
      [0.354032, 0.066925, 0.430906],
      [0.60933,  0.159474, 0.393589],
      [0.841969, 0.292933, 0.248564],
      [0.974176, 0.53678,  0.048392],
      [0.964394, 0.843848, 0.273391]
    ];
    printit("Using default color map");
  else:
    from matplotlib.pyplot import get_cmap
    cmap = get_cmap('inferno');
    SEGMENT_CMAPS = [cmap(i)[:3] for i in range(int(0.1 * cmap.N), int(cmap.N * 0.9), int(cmap.N * 0.9) // 10)];

####################################################################################################
################################# Generate Open3D readable object ##################################
####################################################################################################
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
    rad = 0.01;
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
      pdblines = f.read().strip("\n").split("\n"); 
  elif ("ATOM" in thepdb) or ("HETATM" in thepdb): 
    pdblines = thepdb.strip("\n").split("\n");
  else: 
    raise Exception(f"{pdb2xyzr.__name__:15s}: Please provide a valid PDB file path or PDB string.")
  finallines = ""; 
  for line in pdblines:
    line = line.strip(); 
    if not line.startswith("ATOM") and not line.startswith("HETATM"):
      continue
    x = float(line[30:38]); 
    y = float(line[38:46]); 
    z = float(line[46:54]); 
    resname = line[17:20].strip(); 
    aname = line[12:16].strip(); 
    # special handling for hydrogens (start with digits, not the letter "H")
    if re.match(r"\d[HhDd]", aname):
      aname = "H"
    # However, some bogus PDP files have the H in column 13 so we allow those too, 
    # which means we will treat as Hydrogen helium and hafnium but we protect HG
    if re.match(r"[Hh][^Gg]", aname):
      aname = "H"
    resnum = line[22:26].strip(); 
    # Spaces in atom / residue name will be removed in getRadius function
    rad = getRadius(atom=aname, residue=resname, exp=exp); 
    finallines += f"{x:10.3f}{y:10.3f}{z:10.3f}{rad:6.2f}\n"
  if len(write) > 0: 
    if "xyzr" not in write:
      filename = write+".xyzr";
    else:
      filename = write;
    with open(filename, "w") as file1:
      file1.write(finallines);
  else: 
    return finallines

def runmsms(msms, inputxyzr, outprefix, d = 4, r = 1.2):
  """
  Run MSMS to generate the surface of a set of atoms
  Args:
    msms (str): path to the MSMS executable
    inputxyzr (str): path to the input file
    outprefix (str): path to the output file
    d (float): density of the surface
    r (float): probe radius
  """
  subprocess.run([msms, "-if", inputxyzr, "-of", outprefix, "-density", str(d), "-probe_radius", str(r), "-all"], stdout=subprocess.DEVNULL);
  if os.path.isfile(f"{outprefix}.vert") and os.path.isfile(f"{outprefix}.face"):
    with open(f"{outprefix}.vert", "r") as f:
      for i in range(10):
        line = f.readline()
        if "#" in line:
          continue
        else:
          npoints = int(line.strip("\n").strip().split()[0]);
          break
    if npoints == 0:
      print("Problematic files: ", inputxyzr, outprefix)
      raise Exception(f"{runmsms.__name__:15s}: Debug: No vertices generated.")

    with open(f"{outprefix}.face", "r") as f:
      for i in range(10):
        line = f.readline()
        if "#" in line:
          continue
        else:
          nfaces = int(line.strip("\n").strip().split()[0]);
          break
    if nfaces == 0:
      raise Exception(f"{runmsms.__name__:15s}: Debug: No faces generated.")

    return True
  else:
    # If default parameters fail, try again with some other probe radius
    if _verbose:
      print(f"{runmsms.__name__:15s}: Failed to generate corresponding vertex and triangle file with default setting. "
            f"Trying other parameters");
    for r in np.arange(1.0, 2.0, 0.01):
      subprocess.run([msms, "-if", inputxyzr, "-of", outprefix, "-density", str(d), "-probe_radius", str(r)], stdout=subprocess.DEVNULL);
      if os.path.isfile(f"{outprefix}.vert") and os.path.isfile(f"{outprefix}.face"):
        break
    if os.path.isfile(f"{outprefix}.vert") and os.path.isfile(f"{outprefix}.face"):
      return True
    else:
      print(f"{runmsms.__name__:15s}: Failed to generate corresponding vertex and triangle file");
      return False

def pdb2msms(msms, pdbfile, outprefix):
  """
  Generate the MSMS output for a PDB file
  Args:
    msms (str): path to the MSMS executable
    pdbfile (str): path to the PDB file
    outprefix (str): prefix of the output file
  """
  xyzrfile = pdb2xyzr(pdbfile, write=outprefix)
  ret = runmsms(msms, xyzrfile, outprefix, d = 4, r = 1.4)
  if ret and _verbose:
    print(f"{pdb2msms.__name__:15s}: Successfully generated the MSMS output")
  elif not ret:
    print(f"{pdb2msms.__name__:15s}: Failed to generate the MSMS output")

def traj2msms(msms, traj, frame, indice, force=False, d=4, r=1.5):
  """
  Generate the MSMS output for a trajectory
  ???? TODO: Objective ????
  Args:
    msms (str): path to the MSMS executable
    traj (mdtraj.Trajectory): the trajectory object
    frame (int): the frame number
    indice (list): the list of atom indices
    force (bool): whether to overwrite existing files
    d (float): density of the surface
    r (float): probe radius
  """
  out_prefix = os.path.join(_tempfolder, "msms_")
  if os.path.isfile(f"{out_prefix}.vert") or os.path.isfile(f"{out_prefix}.face"): 
    if force != True: 
      raise Exception(f"{traj2msms.__name__:15s}: {out_prefix}.vert or {out_prefix}.face already exists, please add argument force=True to enable overwriting of existing files.")
  with tempfile.NamedTemporaryFile(prefix=out_prefix) as file1:
    atoms = np.array([a for a in traj.top.atoms]);
    resnames = np.array([a.name for a in traj.top.residues])
    indice = np.array(indice);
    rads = [getRadius(i,j) for i,j in [(a.name,resnames[a.resid]) for a in atoms[indice]]]
    xyzrline = ""
    for (x,y,z),rad in zip(traj.xyz[frame][indice], rads):
      xyzrline += f"{x:10.3f}{y:10.3f}{z:10.3f}{rad:6.2f}\n"
    with open(file1.name, "w") as file1:
      file1.write(xyzrline);
    ret = runmsms(msms, f"{file1.name}.xyzr", file1.name, d=d, r=r);
    if not ret:
      print(f"{traj2msms.__name__:15s}: Failed to generate the MSMS output")

def msms2pcd(vertfile, filename=""):
  """
  Convert the MSMS output (vertex file) to a point cloud readable by Open3D
  Args:
    vertfile (str): path to the MSMS vertex file
    filename (str): path to the output file
  """
  if not os.path.isfile(vertfile):
    raise Exception(f"{msms2pcd.__name__:15s}: Cannot find the MSMS output files (.vert");
  # Read MSMS vertice file
  with open(vertfile, "r") as file1:
    c = 0;
    verticenr = 0;
    xyzs = [];
    normals = [];
    for line in file1:
      if "#" in line:
        # Skip comment lines; 
        continue
      elif verticenr > 0 and c <= verticenr:
        verti = [float(i) for i in line.strip().split()]
        xyzs.append(verti[:3]);
        normals.append(verti[3:6]);
        c += 1
      elif c == 0:
        # Read the title of the vertice file (First line)
        verticenr = int(line.strip().split()[0]);
  pcd = o3d.geometry.PointCloud();
  pcd.points  = o3d.utility.Vector3dVector(xyzs);
  pcd.normals = o3d.utility.Vector3dVector(normals);
  if len(filename) > 0:
    write_ply(xyzs, normals=normals, filename=filename);
  if not pcd.is_empty():
    return pcd
  else:
    print(f"{msms2pcd.__name__:15s}: Failed to convert the MSMS output files to triangle mesh, please check the MSMS output files");
    return o3d.geometry.TriangleMesh();

def msms2mesh(vertfile, facefile, filename=""):
  """
  Convert the MSMS output (vertex and triangle faces) to a triangle mesh readable by Open3D
  Args:
    vertfile (str): path to the MSMS vertex file
    facefile (str): path to the MSMS face file
    filename (str): path to the output file
  """
  if not os.path.isfile(vertfile):
    raise Exception(f"{msms2mesh.__name__:15s}: Cannot find the MSMS output files (.vert"); 
  elif not os.path.isfile(facefile): 
    raise Exception(f"{msms2mesh.__name__:15s}: Cannot find the MSMS output files (.face)");
  ################## Convert the MSMS vertex file to numpy array (xyz and normals) ###################
  with open(vertfile, "r") as file1:
    lines = (line.rstrip() for line in file1 if "#" not in line)  # Generator expression to reduce memory
    verticenr = int(next(lines).split()[0])
    try:
      thearray = np.array([float(i) for i in " ".join([_ for _ in lines]).strip().split()]).reshape((verticenr, -1));
    except ValueError:
      print(f"debug: Check the Vert file {vertfile}")
      print(f"debug: Check the Face file {vertfile}")
      raise Exception("vert fails");
    xyzs = thearray[:, :3];
    normals = thearray[:, 3:6];
  with open(facefile, "r") as file1:
    lines = (line.rstrip() for line in file1 if "#" not in line)  # Generator expression to reduce memory
    facenr = int(next(lines).split()[0])
    try:
      thearray = np.array([int(i) for i in " ".join([_ for _ in lines]).strip().split()]).reshape((facenr, -1));
    except ValueError:
      print(f"debug: Check the Vert file {vertfile}")
      print(f"debug: Check the Face file {vertfile}")
      raise Exception("Face fails");

    faces = thearray[:, :3]-1;
  if len(filename)>0: 
    write_ply(xyzs, normals, faces, filename=filename); 
  mesh = o3d.geometry.TriangleMesh();
  mesh.vertices       = o3d.utility.Vector3dVector(xyzs);
  mesh.vertex_normals = o3d.utility.Vector3dVector(normals);
  mesh.triangles      = o3d.utility.Vector3iVector(faces);
  mesh.remove_degenerate_triangles();
  mesh.compute_vertex_normals();
  if not mesh.is_empty():
    return mesh
  else:
    print(f"{msms2mesh.__name__:15s}: Failed to convert the MSMS output files to triangle mesh, please check the MSMS output files"); 
    return o3d.geometry.TriangleMesh();

####################################################################################################
################################# Representation vector generator ##################################
####################################################################################################


def pseudo_energy_gpu(coord, mode, charges=[]):
  mode = mode.lower();
  _coord = cp.array(coord);
  pairs = list(combinations(cp.arange(len(_coord)), 2));
  energy_final = cp.array(0, dtype=cp.float64);
  if mode == "lj":
    for p in pairs:
      p1 = _coord[p[0]];
      p2 = _coord[p[1]];
      dist = cp.linalg.norm(p1-p2);
      energy_final += pseudo_lj(dist);
  elif mode == "elec":
    if len(charges) == 0:
      charges = cp.zeros(len(_coord));
    else:
      if len(charges) != len(_coord):
        raise Exception(f"The length of the charge ({len(charges)}) does not equal to the dimension of atoms({len(coord)})")
      else:
        charges = cp.array(charges);
    for p in pairs:
      p1 = _coord[p[0]];
      p2 = _coord[p[1]];
      dist = cp.linalg.norm(p1-p2);
      energy_final += pseudo_elec(charges[p[0]], charges[p[1]], dist);
  else:
    raise Exception(f"{pseudo_energy.__name__:15s}: Only two pseudo-energy evaluation modes are supported: Lenar-jones (lj) and Electrostatic (elec)")
  return energy_final.get();



def pseudo_energy(coord, mode, charges=[]):
  # Using the following code for distance calculation, speeds up a bit
  mode = mode.lower(); 
  pairs = list(combinations(np.arange(len(coord)), 2));
  energy_final = 0; 
  if mode == "lj": 
    for p in pairs: 
      p1 = coord[p[0]];
      p2 = coord[p[1]];
      dist = np.linalg.norm(p1 - p2);
      energy_final += pseudo_lj(dist);
  elif mode == "elec": 
    if len(charges) == 0:
      charges = np.zeros(len(coord));
    else: 
      if len(charges) != len(coord):
        raise Exception(f"The length of the charge ({len(charges)}) does not equal to the dimension of atoms({len(coord)})")
      else: 
        charges = np.array(charges); 
    for p in pairs: 
      q1 = charges[p[0]];
      q2 = charges[p[1]];
      p1 = coord[p[0]];
      p2 = coord[p[1]];
      dist = np.linalg.norm(p1-p2);
      energy_final += pseudo_elec(q1, q2, dist);
  else: 
    raise Exception(f"{pseudo_energy.__name__:15s}: Only two pseudo-energy evaluation modes are supported: Lenar-jones (lj) and Electrostatic (elec)")
  return energy_final

def chargedict2array(traj, frame, charges):
  """
  TODO: Objective ????
  """
  _charge = {tuple(np.array(k).round(2)):v for k,v in charges.items()}; 
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
  header.append("property float x");
  header.append("property float y");
  header.append("property float z");
  if len(normals) > 0: 
    header.append("property float nx");
    header.append("property float ny");
    header.append("property float nz");
  if len(triangles) > 0: 
    header.append(f"element face {len(triangles)}");
    header.append("property list uchar int vertex_indices");
  header.append("end_header");
  finalstr  = "";
  # Write the PLY header
  for line in header:
    finalstr += (line + "\n")
  # Write the vertex data
  if len(normals) > 0: 
    for xyz, normal in zip(coords, normals):
      finalstr += (f"{xyz[0]:<8.3f} {xyz[1]:<8.3f} {xyz[2]:<8.3f} {normal[0]:8.3f} {normal[1]:8.3f} {normal[2]:8.3f}\n")
  else: 
    for xyz in coords:
      finalstr += (f"{xyz[0]:<8.3f} {xyz[1]:<8.3f} {xyz[2]:<8.3f}\n")
  if len(triangles) > 0: 
    triangles = np.asarray(triangles).astype(int)
    # Write the triangle data
    for tri in triangles:
      finalstr += (f"3 {tri[0]:4d} {tri[1]:4d} {tri[2]:4d}\n")
  if len(filename) == 0:
    return finalstr
  else:
    with open(filename, "w") as file1:
      file1.write(finalstr)
      return True

class generator: 
  """
  NOTE: Test function of the generator
  >>> traj = pt.load(traj, top=top, mask=":1-151,:LIG", stride=100); 
  >>> repres = repr_generator(traj)
  >>> repres.center = [30,35,30]
  >>> repres.length = [8,8,8]
  >>> repres.frame = 5
  >>> slices, segments = repres.slicebyframe(); 
  >>> feature_vector, mesh_obj, fpfh = repres.vectorize(segments); 

  NOTE: voxel_down_sample might be a better solution to keep most feature during down-sampling
  >>> pcd_new = o3d.geometry.PointCloud( points=o3d.utility.Vector3dVector(finalobj.vertices));
  >>> pcd_new.normals = o3d.utility.Vector3dVector(mesh.vertex_normals);
  >>> finalobj_down = pcd_new.voxel_down_sample(0.8); 
  >>> finalobj_down.estimate_normals(ktree); 
  """
  def __init__(self, traj):
    """
    Initialize the molecule block representation generator class
    Register the trajectory and atoms information
    Args:
      traj: trajectory object
    """
    self.traj = traj;
    self.atoms = np.asarray(list(self.traj.top.atoms));
    self._center = np.zeros(3);
    self._length = np.ones(3);

    # Load parameters from the configuration file
    self.SEGMENT_LIMIT = CONFIG.get("SEGMENT_LIMIT", 6);
    self.FPFH_DOWN_SAMPLES = CONFIG.get("DOWN_SAMPLE_POINTS", 600);
    self.VIEWPOINTBINS = CONFIG.get("VIEWPOINT_BINS", 8);

    # Check the availability of the MSMS executable
    self.MSMS_EXE = CONFIG.get("msms", "");
    if len(self.MSMS_EXE) == 0:
      self.MSMS_EXE = os.environ.get("MSMS_EXE", "");
    if (not self.MSMS_EXE) or (len(self.MSMS_EXE) == 0):
      print(f"Warning: Cannot find the executable for msms program. Use the following ways to find the MSMS executable:\n\"msms\": \"/path/to/MSMS/executable\" in configuration file\nor\nexport MSMS_EXE=/path/to/MSMS/executable", file=sys.stderr)
    elif not os.path.isfile(self.MSMS_EXE):
      print(f"Warning: Designated MSMS executable({self.MSMS_EXE}) not found. Please check the following path: {self.MSMS_EXE}", file=sys.stderr)

    if _verbose:
      # Summary the configuration of the identity generator
      printit("Parameters are loaded")
      printit(f"SEGMENT_LIMIT: {self.SEGMENT_LIMIT}", end=" | ")
      print(f"DOWN_SAMPLE_POINTS: {self.FPFH_DOWN_SAMPLES}", end=" | ")
      print(f"VIEWPOINT_BINS: {self.VIEWPOINTBINS}", end=" | ")
      print(f"MSMS_EXE: {self.MSMS_EXE}")

    # Create a temporary name for intermediate files
    if (not _clear):
      self.set_tempprefix();

  @property
  def center(self):
    return self._center
  @center.setter
  def center(self, new_center):
    assert len(new_center) == 3, "Length should be 3"
    self._center = np.array(new_center)
  @property
  def length(self):
    return self._length
  @length.setter
  def length(self, new_length):
    if isinstance(new_length, int) or isinstance(new_length, float): 
      self._length = np.array([new_length, new_length, new_length]);
    elif isinstance(new_length, list) or isinstance(new_length, np.ndarray): 
      assert len(new_length) == 3, "length should be 3"
      self._length = np.array([i for i in new_length]);
    else: 
      raise Exception("Unexpected data type")
  @property
  def frame(self):
    return self._frame
  @frame.setter
  def frame(self, framei):
    assert isinstance(framei, int), "Frame index should be int"
    self._frame = framei

  def set_tempprefix(self, tempprefix=""):
    if len(tempprefix) > 0:
      self.tempprefix = os.path.join(_tempfolder, f"tmp_{tempprefix}_");
    else:
      temphash = utils.get_hash()[-10:];
      self.tempprefix = os.path.join(_tempfolder, f"tmp_{temphash}_");


  def slicebyframe(self, threshold=2): 
    """
    Generate a slice from all frame of a trajectory (Each frame takes one dimension)
    Returned results are the atomic coordinates and the atomic indices
    """
    xyz = self.traj.xyz[self._frame];
    idx_arr = [];
    state = utils.filter_points_within_bounding_box(xyz, self._center, self._length, return_state=True);
    s_final = np.zeros(len(state), dtype=int);
    lastres = -999;
    seg_counter = 0;
    # for idx, state in zip(range(len(state)), state):
    for idx, state in enumerate(state):
      if state:  # atom with idx within the bounding box
        # Check if the atom is in the same segment
        if self.atoms[idx].resid - lastres > threshold:
          seg_counter += 1;
        s_final[idx] = seg_counter;
        lastres = self.atoms[idx].resid;
      # else case not needed, s_final[idx] is already 0
    return xyz[s_final > 0], s_final[s_final > 0]

  def segment2mesh(self, theidxi, force=False, d=4, r=1.5):
    """
    Use ChimeraX's method to generate molecular surface
    Avoids the use of MSMS to generate intermediate .xyzr, .vert, .face files
    """
    indice = np.asarray(theidxi);
    resnames = np.array([a.name for a in self.traj.top.residues])
    rads = [getRadius(i, j) for i, j in [(a.name, resnames[a.resid]) for a in self.atoms[indice]]]
    vertices, normals, faces = ses_surface_geometry(self.traj.xyz[self.frame][indice], rads)

    mesh = o3d.geometry.TriangleMesh();
    mesh.vertices = o3d.utility.Vector3dVector(vertices);
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals);
    mesh.triangles = o3d.utility.Vector3iVector(faces);
    mesh.remove_degenerate_triangles();
    mesh.compute_vertex_normals();
    if not mesh.is_empty():
      return mesh
    else:
      print(
        f"{msms2mesh.__name__:15s}: Failed to convert the MSMS output files to triangle mesh, please check the MSMS output files");
      return o3d.geometry.TriangleMesh();

  def _segment2mesh(self, theidxi, force=False, d=4, r=1.5):
    """
    Generate a mesh object from a segment of atoms
    Args:
      theidxi: the index of atoms in the segment
      force: force to generate the mesh object even if it is already generated
      d: the density of the mesh for the MSMS program
      r: the radius of probe for the MSMS program
    """
    indice = np.asarray(theidxi);
    self.set_tempprefix();
    if mp.current_process().name == 'MainProcess':
      filename = self.tempprefix + "msms";
    else:
      filename = self.tempprefix + "msms_" + str(mp.current_process().pid);

    # Prepare the xyzr file for MSMS
    resnames = np.array([a.name for a in self.traj.top.residues])
    rads = [getRadius(i, j) for i, j in [(a.name, resnames[a.resid]) for a in self.atoms[indice]]]
    xyzrline = "";
    for (x, y, z), rad in zip(self.traj.xyz[self.frame][indice], rads):
      xyzrline += f"{x:10.3f}{y:10.3f}{z:10.3f}{rad:6.2f}\n"
    with open(f"{filename}.xyzr", "w") as file1:
      file1.write(xyzrline);

    # Generate the vertices and faces file by program MSMS
    ret = runmsms(self.MSMS_EXE, f"{filename}.xyzr", filename, d=d, r=r);

    # Already check the existence of output files
    if ret:
      if _clear:
        mesh = msms2mesh(f"{filename}.vert", f"{filename}.face", filename="");
      else:
        mesh = msms2mesh(f"{filename}.vert", f"{filename}.face", filename=f"{filename}.ply");
      if (_clear and os.path.isfile(f"{filename}.vert")):
        os.remove(f"{filename}.vert")
      if (_clear and os.path.isfile(f"{filename}.face")):
        os.remove(f"{filename}.face")
      if (_clear and os.path.isfile(f"{filename}.xyzr")):
        os.remove(f"{filename}.xyzr")
      if mesh.is_empty() and _verbose:
        raise Exception(f"{self.segment2mesh.__name__:15s}:Failed to generate the 3d object");
      return mesh
    else:
      print(f"{self.segment2mesh.__name__:15s}: Failed to generate the mesh object for segment {theidxi}")
      return False


  def vectorize(self, segment):
    """
    Vectorize the segments (at maximum 6) of a frame
    Args:
      segment: the segment to be vectorized
    """
    # Initialize the identity feature vector
    framefeature = np.zeros((self.SEGMENT_LIMIT, 12 + self.VIEWPOINTBINS));
    pdb_final = "";
    self.__mesh = None;     # Load the lastest mesh object to the object
    segment_objects = [];   # 3D objects for each segment
    atom_indices = [];      # Atom indices for each segment


    # Order the segments from the most abundant to least ones
    segcounter = 0;
    nrsegments = min(len(set(segment)) - 1, self.SEGMENT_LIMIT);
    ordered_segs = utils.ordersegments(segment)[:nrsegments];
    """ ITERATE the at maximum 6 segments """
    for segidx, segi in enumerate(ordered_segs):
      if _verbose:
        printit(f"{self.vectorize.__name__:15s}: Processing segment {segidx + 1}/{nrsegments} ...")
      # ATOM types counts
      theidxi = np.where(segment == segi)[0];
      atomdict = self.atom_type_count(theidxi);
      C_Nr = atomdict.get("C", 0); 
      N_Nr = atomdict.get("N", 0); 
      O_Nr = atomdict.get("O", 0); 
      H_Nr = atomdict.get("H", 0); 
      T_Nr = sum(atomdict.values())
      if _verbose:
        printit(f"{self.vectorize.__name__:15s}: {T_Nr} net atoms in segment {segidx + 1}/{nrsegments} ...")

      # Generate the rdkit molecule here for Residue-based descriptors
      self.seg_mask = utils.getresmask(self.traj, utils.getmaskbyidx(self.traj, theidxi));
      self.seg_indices = self.traj.top.select(self.seg_mask);
      self.seg_mol = chemtools.traj_to_rdkit(self.traj, self.seg_indices, self.frame);
      if (self.seg_mol == None) or (not self.seg_mol):
        framefeature[segcounter - 1, :] = 0;
        print(f"Failed to generate the rdkit molecule for segment {segidx + 1}/{nrsegments} of frame {self.frame} in {self.traj.top_filename}");
        print("Skip this segment ...")
        continue;
      if _verbose:
        printit(f"{self.vectorize.__name__:15s}: Atom number selected by residue: {len(self.seg_indices)} : {self.seg_mask}");

      ########################################################
      # For each segment, computer the partial charge and hydrogen bond information separately
      ########################################################
      C_p, C_n = self.partial_charge();
      N_d, N_a = self.hbp_count();
      PE_lj, PE_el = self.pseudo_energy();
      if _verbose:
        print(f"{self.vectorize.__name__:15s}: Residue-based descriptors of segment {segidx + 1}/{nrsegments} ...")

      atom_indices += self.seg_indices.tolist();
      pdbstr = chemtools.write_pdb_block(self.traj, self.seg_indices, frame_index = self.frame);
      # new_lines = [line for line in pdbstr.strip("\n").split('\n') if ("^END$" not in line and "CONECT" not in line)]
      new_lines = [line for line in pdbstr.strip("\n").split('\n')];
      pdb_final += ('\n'.join(new_lines) + "\n");

      # Segment conversion to triangle mesh
      self.mesh = self.segment2mesh(theidxi);
      if self.mesh == False or self.mesh.is_empty():
        print(f"Failed to generate the surface mesh for segment {segidx + 1}/{nrsegments} of frame {self.frame} in {self.traj.top_filename}");
        print("Skip this segment ...")
        framefeature[segcounter - 1, :] = 0;
        continue

      # Point cloud-based descriptors
      SA = self.surface(self.mesh)
      VOL = self.volume(self.mesh)
      rad = self.mean_radius(self.mesh)
      h_ratio = self.convex_hull_ratio(self.mesh);
      self.mesh.paint_uniform_color(SEGMENT_CMAPS[segcounter]);

      framefeature[segcounter, :12] = [
        T_Nr, C_Nr, N_d, N_a, C_p, C_n, PE_lj, PE_el, SA, VOL, rad, h_ratio
      ]

      # Try fixed viewpoint
      pf_gen = PointFeature(self.mesh);
      if segidx != (nrsegments - 1):
        idx_next = np.where(segment == ordered_segs[segidx+1])[0];
        cog_next = self.traj.xyz[self.frame][idx_next].mean(axis=0);
      else:
        cog_next = self.center;
      vpc = pf_gen.compute_vpc(cog_next, bins = self.VIEWPOINTBINS);
      framefeature[segcounter, -self.VIEWPOINTBINS:] = vpc;
      if (_verbose):
        printit(f"Viewpoint: [1000, 0, 0]; Sum of VPC is: {sum(vpc)}");
      self.__mesh = copy.deepcopy(self.mesh);
      segcounter += 1;
      segment_objects.append(self.mesh);
      if _verbose:
        printit(f"Segment {segcounter} / {nrsegments}: {self.mesh}")
    ########################################################
    # END of the segment iteration
    ########################################################
    if _verbose:
      printit("Final 3D object: ", functools.reduce(lambda a, b: a+b, segment_objects))
    if (not _clear):
      # Write out the final mesh if the intermediate output is required for debugging purpose
      # Reset the file prefix to make the temporary output file organized
      self.set_tempprefix()
      with open(f"{self.tempprefix}frame{self.frame}.pdb", "w") as f:
        f.write(pdb_final);
      final_mesh = functools.reduce(lambda a, b: a + b, segment_objects);
      o3d.io.write_triangle_mesh(f"{self.tempprefix}frame{self.frame}.ply", final_mesh, write_ascii=True);
    # Keep the final PDB and PLY files in memory for further use
    self.active_pdb = pdb_final;
    self.active_indices = atom_indices;

    o3d.io.write_triangle_mesh(f"{self.tempprefix}frame{self.frame}.ply", final_mesh, write_ascii=True);
    with open(f"{self.tempprefix}frame{self.frame}.ply", "r") as f:
      self.active_ply = f.read();
    if _verbose and (len(self.active_pdb) == 0 or len(self.active_ply) == 0):
      printit(f"DEBUG: Failed to correctly generate the intermediate PDB and PLY files for frame {self.frame}");
    return framefeature.reshape(-1), segment_objects
  
  def atom_type_count(self, theidxi):
    """
    Descriptor 1 and 2:
    Args:
      theidxi: the indices of the atoms in the segment
    Return: 
      Atom counts as a dictionary
    """
    ATOM_DICT = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
    atomic_numbers = np.array([i.atomic_number for i in self.atoms[theidxi]]); 
    atom_number = len(atomic_numbers);
    count={}
    for atom, atom_num in ATOM_DICT.items(): 
      if np.count_nonzero(atomic_numbers - atom_num == 0) > 0: 
        count[atom] = np.count_nonzero(atomic_numbers - atom_num == 0); 
    return count

  def hbp_count(self):
    """
    Descriptor 3 and 4: Counter hydrogen bond donor and acceptors.
    Args:
      theidxi: the indices of the atoms in the segment
    Return: 
      number_d: Number of hydrogen bond donor
      number_a: Number of hydrogen bond acceptor
    """
    conf = self.seg_mol.GetConformer()
    d_matches = self.seg_mol.GetSubstructMatches(RD_DONOR_PATTERN);
    a_matches = self.seg_mol.GetSubstructMatches(RD_ACCEPTOR_PATTERN);
    d_within = 0;
    a_within = 0;

    # Get all atom positions and store in a numpy array outside the loop
    all_positions = np.array(conf.GetPositions());

    # Convert bounds checks to numpy operations
    lower_bound = self.center - self.length / 2
    upper_bound = self.center + self.length / 2
    d_coords = all_positions[[d[0] for d in d_matches]]  # Gather all d coordinates at once
    a_coords = all_positions[[a[0] for a in a_matches]]  # Gather all a coordinates at once
    d_within = np.sum(np.all((d_coords > lower_bound) & (d_coords < upper_bound), axis=1))
    a_within = np.sum(np.all((a_coords > lower_bound) & (a_coords < upper_bound), axis=1))
    return d_within, a_within

  def partial_charge(self):
    """
    Descriptor 5 and 6: Counter positive partial charge and negative partial charge.
    Return: 
      charge_p: Number of positive partial charge
      charge_n: Number of negative partial charge
    """
    charge_p = 0;
    charge_n = 0;
    conf = self.seg_mol.GetConformer();
    lower_bound = self.center - self.length / 2
    upper_bound = self.center + self.length / 2

    all_positions = np.array(conf.GetPositions())
    all_charges = np.array([atom.GetDoubleProp('_GasteigerCharge') for atom in self.seg_mol.GetAtoms()])
    # if True in np.isnan(all_charges) and _debug: # Check for NaN values
    #   printit(f"Warning: NaN value found in partial charge for frame {self.frame}")
    #   raise ValueError("NaN value found in partial charge");
    # print(all_charges)

    mask = np.all((all_positions > lower_bound) & (all_positions < upper_bound), axis=1)
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

    # Compute the pseudo-lj and pseudo-elec potential
    pp_elec = 0;
    pp_lj = 0;
    # Constants
    K_ELEC = 8.98
    EPSILON_LJ = 1
    SIGMA_LJ = 1

    atomnr = self.seg_mol.GetNumAtoms();

    conf = self.seg_mol.GetConformer();
    positions = conf.GetPositions();
    charges = np.array([atom.GetDoubleProp('_GasteigerCharge') for atom in self.seg_mol.GetAtoms()])

    if atomnr != len(positions) or atomnr != len(charges):
      print(f"DEBUG: Atom number mismatch in pseudo_energy for frame {self.frame}")
      return 0, 0  # Returns 0 for both pp_lj and pp_elec

    # Now use these pair-wise values to calculate pseudo potentials
    distances = pdist(positions);                 # Pair-wise distances
    distances = squareform(distances);            # Square-form the distance to make it compatible with charge pairs
    charge_products = np.outer(charges, charges)  # Calculate all pair-wise charges product
    mask = np.triu_indices_from(distances, k=1)   # Only upper triangle of the matrix is needed
    pp_lj = 4 * EPSILON_LJ * np.sum(((SIGMA_LJ / distances[mask]) ** 12 - (SIGMA_LJ / distances[mask]) ** 6))
    pp_elec = K_ELEC * np.sum(charge_products[mask] / distances[mask])
    return pp_lj, pp_elec;

  # Descriptor 9 and 10: Surface area and volume of the mesh
  def volume(self, mesh): 
    """
    Volume computation is not robust enough
    """
    try:
      print("DEBUG here: after the geometrical descriptors calculation", self.mesh, VOL)
      VOL = mesh.get_volume();
    except: 
      VOL  = 1.5 * mesh.get_surface_area();
    return VOL
    
  def surface(self, mesh):
    """
    Surface area computation
    Args:
      mesh: open3d.geometry.TriangleMesh
    """
    return mesh.get_surface_area();
  
  def mean_radius(self, mesh):
    """
    Down sample the mesh uniformly and compute the mean radius from the point cloud to the geometric center
    Args:
      mesh: open3d.geometry.TriangleMesh
    """
    pcd = mesh.sample_points_uniformly(CONFIG.get("DOWN_SAMPLE_POINTS", 600));
    mean_radius = np.linalg.norm(np.asarray(pcd.points) - pcd.get_center(), axis=1).mean()
    return mean_radius
  
  def convex_hull_ratio(self, mesh):
    """
    Down sample the mesh uniformly and compute the convex hull ratio
    Args:
      mesh: open3d.geometry.TriangleMesh
    """
    samples = CONFIG.get("DOWN_SAMPLE_POINTS", 600);
    pcd = mesh.sample_points_uniformly(samples);
    hull, _ = pcd.compute_convex_hull();
    hull_ratio = len(hull.vertices)/samples;
    return hull_ratio
  
  def fpfh_down(self, mesh, origin=True):
    """
    Down sample the mesh uniformly and compute the fpfh feature
    TODO: add support for the voxel-base down sampling 
    """
    samples = CONFIG.get("DOWN_SAMPLE_POINTS", 600);
    relative = bool(not origin); 
    mesh_copy = copy.deepcopy(mesh); 
    mesh_copy.translate([0,0,0], relative=relative);
    ktree = o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=20);
    mesh_down = mesh_copy.sample_points_uniformly(samples);
    mesh_down.estimate_normals(ktree);
    fpfh_down = o3d.pipelines.registration.compute_fpfh_feature(mesh_down, ktree);
    return fpfh_down.data

def object_meta(obj):
  points = np.asarray(obj.vertices).round(3);
  normals = np.asarray(obj.vertex_normals).round(3);
  colors = np.asarray(obj.vertex_colors)*256;
  colors = colors.astype(int); 
  triangles = np.asarray(obj.triangles).astype(int); 
  return points.reshape(-1), normals.reshape(-1), colors.reshape(-1), triangles.reshape(-1)


class PointFeature(object):
  def __init__(self, obj_3d):
    # self._neighbors = nneighbors;
    # self._radius = rad;
    self._obj = obj_3d;
    # TODO: setup the points to down sample
    if "vertices" in dir(self._obj):
      self._pcd = np.array(self._obj.vertices);
    elif "points" in dir(self._obj):
      self._pcd = np.array(self._obj.points);
    self._norm = np.array(self._obj.vertex_normals);
    self._kdtree = spatial.KDTree(self._pcd);

  def self_vpc(self, bins=128):
    cos_angles = [np.dot(n, d/np.linalg.norm(d)) for n,d in zip(self._norm, self._pcd-self._pcd.mean(axis=0))];
    angles = np.arccos(cos_angles);
    hist, _ = np.histogram(angles, bins=self.VIEWPOINTBINS, range=(0, np.pi))
    hist_normalized = hist / np.sum(hist)
    hist_normalized = np.asarray([i if not np.isnan(i) else 0 for i in hist_normalized]);
    return hist_normalized

  def compute_vpc(self, viewpoint, bins=128):
    # Compute the relative position of the viewpoint to the center of the point cloud
    rel_vp = np.asarray(viewpoint) - self._pcd.mean(axis=0);

    # Normalize the relative viewpoint vectors
    rel_vp_normalized = rel_vp / np.linalg.norm(rel_vp);

    # Calculate the angle between the normals and the relative viewpoint vectors
    cos_angles = np.dot(self._norm, rel_vp_normalized);
    angles = np.arccos(cos_angles);

    # Create the viewpoint component histogram
    hist, _ = np.histogram(angles, bins=bins, range=(0, np.pi))

    # Normalize the histogram
    hist_normalized = hist / np.sum(hist)

    hist_normalized = np.asarray([i if not np.isnan(i) else 0 for i in hist_normalized]);
    return hist_normalized


####################################################################################################
###################################### Open3D object display #######################################
####################################################################################################
def displayfiles(plyfiles, add=[]):
  """
  Display a list of ply files (trangle mesh) in the same window
  """
  objs = []; 
  finalobj = None;
  for obji, plyfile in enumerate(plyfiles): 
    color = SEGMENT_CMAPS[obji];
    mesh = o3d.io.read_triangle_mesh(plyfile); 
    mesh.compute_vertex_normals(); 
    mesh.paint_uniform_color(color); 
    objs.append(mesh);
    if obji == 0: 
      finalobj = mesh; 
    else: 
      finalobj += mesh;
  display(objs, add=add); 
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
    objs = copy.deepcopy(objects);
    for i in range(1, len(objs)):
      color = SEGMENT_CMAPS[i];
      objs[i].paint_uniform_color(color);
      if isinstance(objs[i], o3d.geometry.TriangleMesh): 
        objs[i].compute_vertex_normals();
    o3d.visualization.draw_geometries(add+objs, width=1200, height=1000);

def display_registration(source, target, transformation):
  """
  Apply the transformation metrix to the source point cloud and display it with the target point cloud
  Args:
    source: open3d.geometry.PointCloud
    target: open3d.geometry.PointCloud
    transformation: transformation matrix, np.array sized (4,4)
  """
  source_temp = copy.deepcopy(source); 
  target_temp = copy.deepcopy(target); 
  source_temp.paint_uniform_color(SEGMENT_CMAPS[1]);
  target_temp.paint_uniform_color(SEGMENT_CMAPS[-1]);
  source_temp.transform(transformation)
  display([source_temp, target_temp])

def displayconvex(obj, n_points=600):
  pcd, hulls = computeconvex(obj, n_points=600)
  display([pcd, hulls])
  
def computeconvex(obj, n_points=600):
  if isinstance(obj, o3d.geometry.TriangleMesh):
    pcd = obj.sample_points_uniformly(n_points); 
  elif isinstance(obj, o3d.geometry.PointCloud):
    pcd = obj.voxel_down_sample(0.01); 
  hull, _ = pcd.compute_convex_hull(); 
  hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull); 
  hull_ls.paint_uniform_color([0.77, 0, 1])
  return (pcd, hull_ls)

def voxelize(obj, show=True):
  if isinstance(obj, o3d.geometry.TriangleMesh):
    pcd = obj.sample_points_uniformly(600)
  elif isinstance(obj, o3d.geometry.PointCloud):
    pcd = obj.voxel_down_sample(0.01);
  else: 
    print(f"Please provide a o3d.geometry.TriangleMesh or o3d.geometry.PointCloud object rather than {type(obj)}")
    return False
  pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(600, 3)))
  # fit to unit cube
  pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
          center=pcd.get_center())  
  voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
  if show: 
    display([voxel_grid]); 
  return voxel_grid 

####################################################################################################
####################################### Open3D accessory div #######################################
####################################################################################################
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
  points = points * length;
  points = points + np.array(center) - (length/2);
  lines = np.array([
    [0, 1], [0, 2], [1, 3], [2, 3],
    [4, 5], [4, 6], [5, 7], [6, 7],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ]); 
  colors = [[0, 0, 0] for i in range(len(lines))]
  line_set = o3d.geometry.LineSet(
    points = o3d.utility.Vector3dVector(points),
    lines = o3d.utility.Vector2iVector(lines),
  ); 
  line_set.colors = o3d.utility.Vector3dVector(colors); 
  return line_set

def NewCoordFrame(center=[0,0,0], scale=1):
  """
  Accessory function to create a coordinate frame
  Args:
    center: center of the coordinate frame
    scale: scale of the coordinate frame
  """
  coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(); 
  coord_frame.scale(scale=scale, center=center)
  coord_frame.translate(center, relative=False)
  return coord_frame


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
    array1 = array1.ravel().reshape(6, -1);
    array2 = array2.ravel().reshape(6, -1);
    weights = np.asarray(weights);
    contrib_chem = np.array([cosine_similarity(array1[i, :12].ravel(), array2[i, :12].ravel()) for i in range(6)])
    contrib_viewpoint = np.array([cosine_similarity(array1[i, 12:].ravel(), array2[i, 12:].ravel()) for i in range(6)])

    # print("TESTHERE");
    # print(array1.tolist(), array2.tolist());
    # for i in range(6):
    #   _cos = cossim(array1[i, :12].ravel(), array2[i, :12].ravel())
    #   print(f"====> session {i}", _cos, array1[i, :12].ravel(), array2[i, :12].ravel());
    # print("TESTEND")

    similarities = (contrib_chem + contrib_viewpoint) / 2 * weights / sum(weights[np.nonzero(contrib_chem)]);
    # print(similarities, contrib_viewpoint.round(2), contrib_chem.round(2))

    print(f"{'Chem contribution':20s}: ", ''.join(f'{i:6.2f}' for i in contrib_chem))
    print(f"{'VP contribution':20s}: ", ''.join(f'{i:6.2f}' for i in contrib_viewpoint))
    print(f"{'Weights':20s}: ", ''.join(f'{i:6.2f}' for i in weights))
    print(f"{'Contribution(real)':20s}: ", ''.join(f'{i:6.2f}' for i in similarities));
    print(f"{'Final Similarity':20s}: ", sum(similarities), "\n")
    return sum(similarities)

def weight(array1):
  array1 = array1.ravel().reshape(6, -1);
  # array2 = array2.ravel().reshape(6, -1);
  # print(array1)
  # print("reweight", array1[:, 0].ravel())
  return array1[:, 0].ravel()

######################################################################
########## ChimeraX's function to compute surface from XYZR ##########
######################################################################
from numba import jit

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
  tf_rot = tf[:, :3];
  tf_trans = tf[:, 3];
  # Use NumPy's broadcasting capabilities to perform the matrix multiplication
  vertex_positions_transformed = np.dot(vertex_positions, tf_rot.T) + tf_trans;
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

def ses_surface_geometry(xyz, radii, probe_radius=1.4, grid_spacing=0.5, sas=False):
  '''
  Calculate a solvent excluded molecular surface using a distance grid
  contouring method.  Vertex, normal and triangle arrays are returned.
  If sas is true then the solvent accessible surface is returned instead.
  This avoid generating the
  '''
  sys.path.insert(0, "/media/yzhang/MieT5/BetaPose/static")
  import _geometry, _surface, _map

  radii = np.asarray(radii, np.float32)
  # Compute bounding box for atoms
  xyz_min, xyz_max = xyz.min(axis=0), xyz.max(axis=0)
  pad = 2 * probe_radius + radii.max() + grid_spacing
  origin = [x - pad for x in xyz_min]

  # Create 3d grid for computing distance map
  s = grid_spacing
  shape = [int(np.ceil((xyz_max[a] - xyz_min[a] + 2 * pad) / s))
           for a in (2, 1, 0)]

  try:
    matrix = np.empty(shape, np.float32)
  except (MemoryError, ValueError):
    raise MemoryError('Surface calculation out of memory trying to allocate a grid %d x %d x %d '
                      % (shape[2], shape[1], shape[0]) +
                      'to cover xyz bounds %.3g,%.3g,%.3g ' % tuple(xyz_min) +
                      'to %.3g,%.3g,%.3g ' % tuple(xyz_max) +
                      'with grid size %.3g' % grid_spacing)

  max_index_range = 2
  matrix[:, :, :] = max_index_range

  # Transform centers and radii to grid index coordinates
  tf_matrix34 = np.array(((1.0 / s, 0, 0, -origin[0] / s),
                          (0, 1.0 / s, 0, -origin[1] / s),
                          (0, 0, 1.0 / s, -origin[2] / s)))

  """transforms the atomic coordinates to grid index coordinates"""
  ijk = affine_transform_vertices(xyz, tf_matrix34);

  ri = radii.astype(np.float32)
  ri += probe_radius
  ri /= s

  # Compute distance map from surface of spheres, positive outside.
  _map.sphere_surface_distance(ijk, ri, max_index_range, matrix)

  # Get the SAS surface as a contour surface of the distance map
  level = 0
  sas_va, sas_ta, sas_na = _map.contour_surface(matrix, level, cap_faces=False,
                                                calculate_normals=True)
  if sas:
    invert_m34 = invert_matrix(tf_matrix34);
    ses_va = affine_transform_vertices(ses_va, invert_m34);
    return sas_va, sas_na, sas_ta

  # Compute SES surface distance map using SAS surface vertex
  # points as probe sphere centers.
  matrix[:, :, :] = max_index_range
  rp = np.empty((len(sas_va),), np.float32)
  rp[:] = float(probe_radius) / s
  _map.sphere_surface_distance(sas_va, rp, max_index_range, matrix)
  ses_va, ses_ta, ses_na = _map.contour_surface(matrix, level, cap_faces=False,
                                                calculate_normals=True)

  # Transform surface from grid index coordinates to atom coordinates
  invert_m34 = invert_matrix(tf_matrix34);
  ses_va = affine_transform_vertices(ses_va, invert_m34);

  # Delete connected components more than 1.5 probe radius from atom spheres.
  kvi = []
  kti = []
  vtilist = _surface.connected_pieces(ses_ta)

  for vi, ti in vtilist:
    v0 = ses_va[vi[0], :]
    d = xyz - v0
    d2 = (d * d).sum(axis=1)
    adist = (np.sqrt(d2) - radii).min()
    if adist < 1.5 * probe_radius:
      kvi.append(vi)
      kti.append(ti)
  keepv = np.concatenate(kvi) if kvi else []
  keept = np.concatenate(kti) if kti else []
  va, na, ta = reduce_geometry(ses_va, ses_na, ses_ta, keepv, keept)
  return va, na, ta




