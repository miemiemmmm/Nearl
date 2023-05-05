import os, sys, re, time, subprocess, tempfile, datetime, copy
import pytraj as pt
import numpy as np 
import open3d as o3d

from itertools import combinations
from scipy.spatial.distance import cdist
from . import utils, chemtools, CONFIG

ATOM_PATTERNS = {0: '^[0-9]*H.*$', 1: '^[0-9]*D.*$', 2: '^O.*$', 3: '^CA$', 4: '^CD$', 5: '^CD  $', 6: '^CA$', 7: '^N$', 8: '^CA$', 9: '^C$', 10: '^O$', 11: '^P$', 12: '^CB$', 13: '^CB$', 14: '^CB$', 15: '^CG$', 16: '^CG$', 17: '^CG$', 18: '^CG$', 19: '^O1$', 20: '^O2$', 21: '^CH3$', 22: '^CD$', 23: '^NE$', 24: '^RE$', 25: '^CZ$', 26: '^NH[12][AB]?$', 27: '^RH[12][AB]?$', 28: '^OD1$', 29: '^ND2$', 30: '^AD1$', 31: '^AD2$', 32: '^OD[12][AB]?$', 33: '^ED[12][AB]?$', 34: '^OD1[AB]?$', 35: '^ND2$', 36: '^AD1$', 37: '^AD2$', 38: '^OD2$', 39: '^LP[12]$', 40: '^SG$', 41: '^SG$', 42: '^OE[12][AB]?$', 43: '^EE[12][AB]?$', 44: '^CD$', 45: '^OE1$', 46: '^NE2$', 47: '^AE[12]$', 48: '^CE1|CD2$', 49: '^ND1$', 50: '^ND1$', 51: '^RD1$', 52: '^NE2$', 53: '^RE2$', 54: '^NE2$', 55: '^RE2$', 56: '^A[DE][12]$', 57: '^CG1$', 58: '^CG2$', 59: '^CD|CD1$', 60: '^CD1$', 61: '^CD2$', 62: '^C[GDE]$', 63: '^NZ$', 64: '^KZ$', 65: '^SD$', 66: '^CE$', 67: '^C[DE][12]$', 68: '^CZ$', 69: '^C[GD]$', 70: '^SE$', 71: '^SEG$', 72: '^OD1$', 73: '^OD2$', 74: '^OG$', 75: '^OG1$', 76: '^CG2$', 77: '^CD1$', 78: '^CD2$', 79: '^CE2$', 80: '^NE1$', 81: '^CE3$', 82: '^CZ2$', 83: '^CZ3$', 84: '^CH2$', 85: '^C[DE][12]$', 86: '^CZ$', 87: '^OH$', 88: '^CG1$', 89: '^CG2$', 90: '^CD$', 91: '^CE$', 92: '^FE[1-7]$', 93: '^S[1-7]$', 94: '^OXO$', 95: '^FE1$', 96: '^FE2$', 97: '^O1$', 98: '^O2$', 99: '^FE$', 100: '^CH[A-D]$', 101: '^N[A-D]$', 102: '^N [A-D]$', 103: '^C[1-4][A-D]$', 104: '^CM[A-D]$', 105: '^C[AB][AD]$', 106: '^CG[AD]$', 107: '^O[12][AD]$', 108: '^C[AB][BC]$', 109: '^OH2$', 110: '^N[123]$', 111: '^C1$', 112: '^C2$', 113: '^C3$', 114: '^C4$', 115: '^C5$', 116: '^C6$', 117: '^O7$', 118: '^O8$', 119: '^S$', 120: '^O[1234]$', 121: '^O[1234]$', 122: '^O4$', 123: '^P1$', 124: '^O[123]$', 125: '^C[12]$', 126: '^N1$', 127: '^C[345]$', 128: '^BAL$', 129: '^POI$', 130: '^DOT$', 131: '^CU$', 132: '^ZN$', 133: '^MN$', 134: '^FE$', 135: '^MG$', 136: '^MN$', 137: '^CO$', 138: '^SE$', 139: '^YB$', 140: '^N1$', 141: '^C[2478]$', 142: '^O2$', 143: '^N3$', 144: '^O4$', 145: '^C[459]A$', 146: '^N5$', 147: '^C[69]$', 148: '^C[78]M$', 149: '^N10$', 150: '^C10$', 151: '^C[12345]\\*$', 152: '^O[234]\\*$', 153: '^O5\\*$', 154: '^OP[1-3]$', 155: '^OT1$', 156: '^C01$', 157: '^C16$', 158: '^C14$', 159: '^C.*$', 160: '^SEG$', 161: '^OXT$', 162: '^OT.*$', 163: '^E.*$', 164: '^S.*$', 165: '^C.*$', 166: '^A.*$', 167: '^O.*$', 168: '^N.*$', 169: '^R.*$', 170: '^K.*$', 171: '^P[A-D]$', 172: '^P.*$', 173: '^.O.*$', 174: '^.N.*$', 175: '^.C.*$', 176: '^.P.*$', 177: '^.H.*$'}
RESIDUE_PATTERNS = {0: '^.*$', 1: '^.*$', 2: '^WAT|HOH|H2O|DOD|DIS$', 3: '^CA$', 4: '^CD$', 5: '^.*$', 6: '^ACE$', 7: '^.*$', 8: '^.*$', 9: '^.*$', 10: '^.*$', 11: '^.*$', 12: '^ALA$', 13: '^ILE|THR|VAL$', 14: '^.*$', 15: '^ASN|ASP|ASX|HIS|HIP|HIE|HID|HISN|HISL|LEU|PHE|TRP|TYR$', 16: '^ARG|GLU|GLN|GLX|MET$', 17: '^LEU$', 18: '^.*$', 19: '^GLN$', 20: '^GLN$', 21: '^ACE$', 22: '^ARG$', 23: '^ARG$', 24: '^ARG$', 25: '^ARG$', 26: '^ARG$', 27: '^ARG$', 28: '^ASN$', 29: '^ASN$', 30: '^ASN$', 31: '^ASN$', 32: '^ASP$', 33: '^ASP$', 34: '^ASX$', 35: '^ASX$', 36: '^ASX$', 37: '^ASX$', 38: '^ASX$', 39: '^CYS|MET$', 40: '^CY[SXM]$', 41: '^CYH$', 42: '^GLU$', 43: '^GLU$', 44: '^GLU|GLN|GLX$', 45: '^GLN$', 46: '^GLN$', 47: '^GLN|GLX$', 48: '^HIS|HID|HIE|HIP|HISL$', 49: '^HIS|HIE|HISL$', 50: '^HID|HIP$', 51: '^HID|HIP$', 52: '^HIS|HIE|HIP$', 53: '^HIS|HIE|HIP$', 54: '^HID|HISL$', 55: '^HID|HISL$', 56: '^HIS|HID|HIP|HISD$', 57: '^ILE$', 58: '^ILE$', 59: '^ILE$', 60: '^LEU$', 61: '^LEU$', 62: '^LYS$', 63: '^LYS$', 64: '^LYS$', 65: '^MET$', 66: '^MET$', 67: '^PHE$', 68: '^PHE$', 69: '^PRO|CPR$', 70: '^CSO$', 71: '^CSO$', 72: '^CSO$', 73: '^CSO$', 74: '^SER$', 75: '^THR$', 76: '^THR$', 77: '^TRP$', 78: '^TRP$', 79: '^TRP$', 80: '^TRP$', 81: '^TRP$', 82: '^TRP$', 83: '^TRP$', 84: '^TRP$', 85: '^TYR$', 86: '^TYR$', 87: '^TYR$', 88: '^VAL$', 89: '^VAL$', 90: '^.*$', 91: '^.*$', 92: '^FS[34]$', 93: '^FS[34]$', 94: '^FS3$', 95: '^FEO$', 96: '^FEO$', 97: '^HEM$', 98: '^HEM$', 99: '^HEM$', 100: '^HEM$', 101: '^HEM$', 102: '^HEM$', 103: '^HEM$', 104: '^HEM$', 105: '^HEM$', 106: '^HEM$', 107: '^HEM$', 108: '^HEM$', 109: '^HEM$', 110: '^AZI$', 111: '^MPD$', 112: '^MPD$', 113: '^MPD$', 114: '^MPD$', 115: '^MPD$', 116: '^MPD$', 117: '^MPD$', 118: '^MPD$', 119: '^SO4|SUL$', 120: '^SO4|SUL$', 121: '^PO4|PHO$', 122: '^PC$', 123: '^PC$', 124: '^PC$', 125: '^PC$', 126: '^PC$', 127: '^PC$', 128: '^BIG$', 129: '^POI$', 130: '^DOT$', 131: '^.*$', 132: '^.*$', 133: '^.*$', 134: '^.*$', 135: '^.*$', 136: '^.*$', 137: '^.*$', 138: '^.*$', 139: '^.*$', 140: '^FMN$', 141: '^FMN$', 142: '^FMN$', 143: '^FMN$', 144: '^FMN$', 145: '^FMN$', 146: '^FMN$', 147: '^FMN$', 148: '^FMN$', 149: '^FMN$', 150: '^FMN$', 151: '^FMN$', 152: '^FMN$', 153: '^FMN$', 154: '^FMN$', 155: '^ALK|MYR$', 156: '^ALK|MYR$', 157: '^ALK$', 158: '^MYR$', 159: '^ALK|MYR$', 160: '^.*$', 161: '^.*$', 162: '^.*$', 163: '^.*$', 164: '^.*$', 165: '^.*$', 166: '^.*$', 167: '^.*$', 168: '^.*$', 169: '^.*$', 170: '^.*$', 171: '^.*$', 172: '^.*$', 173: '^FAD|NAD|AMX|APU$', 174: '^FAD|NAD|AMX|APU$', 175: '^FAD|NAD|AMX|APU$', 176: '^FAD|NAD|AMX|APU$', 177: '^FAD|NAD|AMX|APU$'}
EXP_RADII = {1: 1.4, 2: 1.4, 3: 1.4, 4: 1.54, 5: 1.54, 6: 1.54, 7: 1.74, 8: 1.74, 9: 1.74, 10: 1.74, 11: 1.74, 12: 1.8, 13: 1.8, 14: 1.54, 15: 1.2, 16: 0.0, 17: 5.0, 18: 1.97, 19: 1.4, 20: 1.4, 21: 1.3, 22: 1.49, 23: 0.01, 24: 1.2, 25: 1.24, 26: 1.6, 27: 1.24, 28: 1.25, 29: 2.15, 30: 3.0, 31: 1.15, 38: 1.8}
UNITED_RADII = {1: 1.4, 2: 1.6, 3: 1.4, 4: 1.7, 5: 1.8, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0, 10: 1.74, 11: 1.86, 12: 1.85, 13: 1.8, 14: 1.54, 15: 1.2, 16: 1.5, 17: 5.0, 18: 1.97, 19: 1.4, 20: 1.4, 21: 1.3, 22: 1.49, 23: 0.01, 24: 0.0, 25: 1.24, 26: 1.6, 27: 1.24, 28: 1.25, 29: 2.15, 30: 3.0, 31: 1.15, 38: 1.8}
ATOM_NUM = {0: 15, 1: 15, 2: 2, 3: 18, 4: 22, 5: 22, 6: 9, 7: 4, 8: 7, 9: 10, 10: 1, 11: 13, 12: 9, 13: 7, 14: 8, 15: 10, 16: 8, 17: 7, 18: 8, 19: 3, 20: 3, 21: 9, 22: 8, 23: 4, 24: 4, 25: 10, 26: 5, 27: 5, 28: 1, 29: 5, 30: 3, 31: 3, 32: 3, 33: 3, 34: 1, 35: 5, 36: 3, 37: 3, 38: 3, 39: 13, 40: 13, 41: 12, 42: 3, 43: 3, 44: 10, 45: 1, 46: 5, 47: 3, 48: 11, 49: 14, 50: 4, 51: 4, 52: 4, 53: 4, 54: 14, 55: 14, 56: 4, 57: 8, 58: 9, 59: 9, 60: 9, 61: 9, 62: 8, 63: 6, 64: 6, 65: 13, 66: 9, 67: 11, 68: 11, 69: 8, 70: 9, 71: 9, 72: 3, 73: 3, 74: 2, 75: 2, 76: 9, 77: 11, 78: 10, 79: 10, 80: 4, 81: 11, 82: 11, 83: 11, 84: 11, 85: 11, 86: 10, 87: 2, 88: 9, 89: 9, 90: 8, 91: 8, 92: 21, 93: 13, 94: 1, 95: 21, 96: 21, 97: 1, 98: 1, 99: 21, 100: 11, 101: 14, 102: 14, 103: 10, 104: 9, 105: 8, 106: 10, 107: 3, 108: 11, 109: 2, 110: 14, 111: 9, 112: 10, 113: 8, 114: 7, 115: 9, 116: 9, 117: 2, 118: 2, 119: 13, 120: 3, 121: 3, 122: 3, 123: 13, 124: 3, 125: 8, 126: 14, 127: 9, 128: 17, 129: 23, 130: 23, 131: 20, 132: 19, 133: 24, 134: 25, 135: 26, 136: 27, 137: 28, 138: 29, 139: 31, 140: 4, 141: 10, 142: 1, 143: 14, 144: 1, 145: 10, 146: 4, 147: 11, 148: 9, 149: 4, 150: 10, 151: 8, 152: 2, 153: 3, 154: 3, 155: 3, 156: 10, 157: 9, 158: 9, 159: 8, 160: 9, 161: 3, 162: 3, 163: 3, 164: 13, 165: 7, 166: 11, 167: 1, 168: 4, 169: 4, 170: 6, 171: 13, 172: 13, 173: 1, 174: 4, 175: 7, 176: 13, 177: 15}

CMAP6 = [
  [0.087411, 0.044556, 0.224813], [0.354032, 0.066925, 0.430906],
  [0.60933, 0.159474, 0.393589], [0.841969, 0.292933, 0.248564],
  [0.974176, 0.53678, 0.048392], [0.964394, 0.843848, 0.273391]
];

####################################################################################################
################################# Generate Open3D readable object ##################################
####################################################################################################
def getRadius(atom="", residue="", exp=False):
  atom = atom.replace(" ", "")
  residue = residue.replace(" ", "")
  for pat in range(len(ATOM_NUM)):
    if re.match(ATOM_PATTERNS[pat], atom) and re.match(RESIDUE_PATTERNS[pat], residue):
      break
  if pat == len(ATOM_NUM):
    rad = 0.01; 
  else:
    rad = UNITED_RADII[ATOM_NUM[pat]] if exp != True else EXP_RADII[ATOM_NUM[pat]]
  return rad

def pdb2xyzr(thepdb, write="", exp=False): 
  if os.path.isfile(thepdb): 
    with open(thepdb, "r") as f:
      pdblines = f.read().strip("\n").split("\n"); 
  elif ("ATOM" in thepdb) or ("HETATM" in thepdb): 
    pdblines = thepdb.strip("\n").split("\n");
  else: 
    raise Exception(f"{pdb2xyzr.__name__:15s}: Please provide a valid PDB path or pdb string.")
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
    resname = resname.replace(" ", ""); 
    aname = aname.replace(" ", ""); 
    rad = getRadius(atom=aname, residue=resname, exp=exp); 
    finallines += f"{x:10.3f}{y:10.3f}{z:10.3f}{rad:6.2f}\n"
  if len(write) > 0: 
    if "xyzr" not in write:
      write = write+".xyzr"; 
    with open(write, "w") as file1: 
      file1.write(finallines); 
    return write
  else: 
    return finallines

def runmsms(msms, inputfile, outfile, d = 4, r = 1.5): 
  subprocess.run([msms, "-if", inputfile, "-of", outfile, "-density", str(d), "-probe_radius", str(r), "-all"], stdout=subprocess.DEVNULL); 
  if os.path.isfile(f"{outfile}.vert") and os.path.isfile(f"{outfile}.face"): 
    return True
  else: 
    subprocess.run([msms, "-if", inputfile, "-of", outfile, "-density", str(d), "-probe_radius", str(r)], stdout=subprocess.DEVNULL);
    if os.path.isfile(f"{outfile}.vert") and os.path.isfile(f"{outfile}.face"):
      return True
    else: 
      # Very low possibility of failing
      print(f"{runmsms.__name__:15s}: Failed to generate corresponding vertex and triangle file"); 
      return False 

def pdb2msms(msms, pdbfile, outprefix): 
  xyzrfile = pdb2xyzr(pdbfile, write=outprefix)
  ret = runmsms(msms, xyzrfile, outprefix, d = 4, r = 1.4)
  if ret:
    print(f"{pdb2msms.__name__:15s}: Successfully generated the MSMS output")
  else:
    print(f"{pdb2msms.__name__:15s}: Failed to generate the MSMS output")

def traj2msms(msms, traj, frame, indice, out_prefix="/tmp/test", force=False, d=4, r=1.5):
  if os.path.isfile(f"{out_prefix}.vert") or os.path.isfile(f"{out_prefix}.face"): 
    if force != True: 
      raise Exception(f"{traj2msms.__name__:15s}: {out_prefix}.vert or {out_prefix}.face already exists, please add argument force=True to enable overwriting of existing files.")
  with tempfile.NamedTemporaryFile(suffix=".xyzr") as file1:
    atoms = np.array([a for a in traj.top.atoms]);
    resnames = np.array([a.name for a in traj.top.residues])
    indice = np.array(indice);
    rads = [getRadius(i,j) for i,j in [(a.name,resnames[a.resid]) for a in atoms[indice]]]
    xyzrline = ""
    for (x,y,z),rad in zip(traj.xyz[frame][indice], rads):
      xyzrline += f"{x:10.3f}{y:10.3f}{z:10.3f}{rad:6.2f}\n"
    with open(file1.name, "w") as file1:
      file1.write(xyzrline);
    ret = runmsms(msms, file1.name, out_prefix, d=d, r=r);
    if ret: 
      pass
    else: 
      print(f"{traj2msms.__name__:15s}: Failed to generate the MSMS output")

def msms2pcd(vertfile, filename=""):
  """
  Convert the MSMS output (vertex and triangle faces) to a point cloud readable by Open3D
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
  if len(filename)>0:
    write_ply(xyzs, normals=normals, filename=filename);
  if not pcd.is_empty():
    return pcd
  else:
    print(f"{msms2pcd.__name__:15s}: Failed to convert the MSMS output files to triangle mesh, please check the MSMS output files");
    return o3d.geometry.TriangleMesh();

def msms2mesh(vertfile, facefile, filename=""):
  """
  Convert the MSMS output (vertex and triangle faces) to a triangle mesh readable by Open3D
  """
  if not os.path.isfile(vertfile):
    raise Exception(f"{msms2mesh.__name__:15s}: Cannot find the MSMS output files (.vert"); 
  elif not os.path.isfile(facefile): 
    raise Exception(f"{msms2mesh.__name__:15s}: Cannot find the MSMS output files (.face)");
  ################## Convert the MSMS vertex file to numpy array (xyz and normals) ###################
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
        # Read the title of the vertice file 
        verticenr = int(line.strip().split()[0]); 
  ############## Read the MSMS triangle faces file for the generation of triangle mesh ###############
  with open(facefile, "r") as file1:
    c = 0;
    facenr = 0; 
    faces = [];
    for line in file1:
      if "#" in line:
        # Skip comment lines; 
        continue
      elif facenr > 0 and c <= facenr:
        # NOTE: open3d index start from 0 and hence minus 1; 
        facei = [float(i)-1 for i in line.strip().split()]
        faces.append(facei[:3]);
        c += 1
      elif c == 0:
        # Read the title of the face file 
        facenr = int(line.strip().split()[0]); 
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

def pseudo_energy(traj, frame, idxs, mode, charges=[]): 
  # Using the following code for distance calculation, speeds up a bit
  mode = mode.lower(); 
  pairs = list(combinations(idxs, 2)); 
  xyz = traj.xyz[frame]; 
  energy_final = 0; 
  if mode == "lj": 
    for p in pairs: 
      p1 = xyz[p[0]];
      p2 = xyz[p[1]];
      dist = np.linalg.norm(p1-p2);   
      energy_final += pseudo_lj(dist);
  elif mode == "elec": 
    if len(charges) == 0:
      charges = np.zeros(len(xyz)); 
    else: 
      if len(charges) != len(xyz): 
        raise Exception(f"The length of the charge ({len(charges)}) does not equal to the dimension of atoms({len(xyz)})")
      else: 
        charges = np.array(charges); 
    for p in pairs: 
      q1 = charges[p[0]];
      q2 = charges[p[1]];
      p1 = xyz[p[0]];
      p2 = xyz[p[1]];
      dist = np.linalg.norm(p1-p2);
      energy_final += pseudo_elec(q1, q2, dist);
  else: 
    raise Exception(f"{pseudo_energy.__name__:15s}: Only two pseudo-energy evaluation modes are supported: Lenar-jones (lj) and Electrostatic (elec)")
  return energy_final

def chargedict2array(traj, frame, charges): 
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
    self.atoms = np.asarray(list(traj.top.atoms)); 
    self.traj = traj; 
    self.SEGMENT_LIMIT = 6
    self.FPFH_DOWN_SAMPLES = 600; 
    
    if CONFIG.get("msms", False): 
      self.MSMS_EXE = CONFIG["msms"]
    else: 
      self.MSMS_EXE = os.environ.get("MSMS_EXE", ""); 
    if not self.MSMS_EXE or len(self.MSMS_EXE) == 0: 
      print("Warning: Cannot find the executable for msms program. Use the following command to set up msms: export MSMS_EXE=/your/path/to/msms", file=sys.stderr)
    elif not os.path.isfile(self.MSMS_EXE): 
      print(f"Warning: Designated MSMS executable not found. Please check the following path: {self.MSMS_EXE}", file=sys.stderr)
    else: 
      print(f"Found the MSMS executable {self.MSMS_EXE}")
      
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
      self._length = np.array([new_length] * 3);
    elif isinstance(new_length, list) or isinstance(new_length, np.ndarray): 
      assert len(new_length) == 3, "length should be 3"
      self._length = np.array(new_length); 
    else: 
      raise Exception("Unexpected data type")
  @property
  def frame(self):
    return self._frame
  @frame.setter
  def frame(self, framei):
    assert isinstance(framei, int), "Frame index should be int"
    self._frame = framei

    
  def slicebyframe(self, threshold=2): 
    """
    Generate a slice from all frame of a trajectory (Each frame takes one dimension)
    Returned results are the atomic coordinates and the atomic indices
    """
    xyz = self.traj.xyz[self._frame]; 
    idx_arr = []; 
    s_final = [];
    state = utils.boxfilter(xyz, self._center, self._length, return_state=True); 
    lastres = -999;
    seg_counter = 0;
    for idx, state in enumerate(state):
      if state:
        if self.atoms[idx].resid - lastres > threshold:
          seg_counter += 1;
        s_final.append(seg_counter)
        lastres = self.atoms[idx].resid;
      else:
        s_final.append(0);
    return xyz[np.array([bool(i) for i in s_final])], np.array(s_final)
    
  def segment2mesh(self, theidxi, force=False, clear=True, d=4, r=1.5): 
    indice = np.array(theidxi);
    with tempfile.NamedTemporaryFile(suffix=".xyzr") as file1:
      filename = file1.name; 
      resnames = np.array([a.name for a in self.traj.top.residues])
      rads = [getRadius(i,j) for i,j in [(a.name,resnames[a.resid]) for a in self.atoms[indice]]]
      xyzrline = ""; 
      for (x,y,z),rad in zip(self.traj.xyz[self.frame][indice], rads):
        xyzrline += f"{x:10.3f}{y:10.3f}{z:10.3f}{rad:6.2f}\n"
      with open(file1.name, "w") as file1:
        file1.write(xyzrline);
      out_prefix = file1.name.replace(".xyzr", "")
      ret = runmsms(self.MSMS_EXE, file1.name, out_prefix, d=d, r=r);
      # Already check the existence of output files
      if ret:
        if clear: 
          mesh = msms2mesh(f"{out_prefix}.vert", f"{out_prefix}.face", filename="");
        else: 
          mesh = msms2mesh(f"{out_prefix}.vert", f"{out_prefix}.face", filename=f"{out_prefix}.ply");
        if mesh.is_empty(): 
          try: 
            mesh = o3d.io.read_triangle_mesh(f"{out_prefix}.ply"); 
            mesh.remove_degenerate_triangles(); 
            mesh.compute_vertex_normals(); 
          except:
            raise Exception(f"{self.segment2mesh.__name__:15s}:Failed to generate the 3d object"); 
        if (clear and os.path.isfile(f"{out_prefix}.vert")): 
          os.remove(f"{out_prefix}.vert")
        if (clear and os.path.isfile(f"{out_prefix}.face")): 
          os.remove(f"{out_prefix}.face")
        return mesh
      else:
        print(f"{self.segment2mesh.__name__:15s}: Failed to generate the MSMS output")
        print(theidxi)
        return False

    
  def vectorize(self, segment, clear=True, msms=""): 
    """
    """
    if (not clear):
      with tempfile.NamedTemporaryFile(suffix="_final") as file1:
        tempname = file1.name;
    else:
      tempname = ""
    framefeature = np.zeros((self.SEGMENT_LIMIT,12)) #12 * self.SEGMENT_LIMIT).reshape(); 
    # Order the segments from the most abundant to least ones
    segcounter = 0;
    pdb_final = ""
    """ ITERATE the at maximum 6 segments """
    for segi in utils.ordersegments(segment)[:self.SEGMENT_LIMIT]:
      # ATOM types counts
      theidxi = np.where(segment == segi)[0];
      atomdict = self.atom_type_count(theidxi)
      C_Nr = atomdict.get("C", 0); 
      N_Nr = atomdict.get("N", 0); 
      O_Nr = atomdict.get("O", 0); 
      H_Nr = atomdict.get("H", 0); 
      T_Nr = sum(atomdict.values())
      
      #################################### Residue-based descriptors #####################################
      self.resmask = utils.getresmask(self.traj, utils.getmaskbyidx(self.traj, theidxi));
      self.charges = chemtools.Chargebytraj(self.traj, self.frame, self.resmask);
      N_d, N_a = self.hbp_count(theidxi);
      C_p, C_n = self.partial_charge(theidxi);
      PE_lg, PE_el = self.pseudo_energy(theidxi)

      if (not clear):
        pdb_final += chemtools.writepdbs(self.traj, self.frame, self.resmask);
      else:
        pdb_final += ""
      
      ############################### Segment conversion to triangle mesh ################################
      self.mesh = self.segment2mesh(theidxi, clear=clear);
      if self.mesh == False or self.mesh.is_empty(): 
        return [], [], []
      ################################## Point cloud-based descriptors ###################################
      SA = self.surface(self.mesh)
      VOL = self.volume(self.mesh)
      
      rad = self.mean_radius(self.mesh)
      h_ratio = self.convex_hull_ratio(self.mesh);
      self.mesh.paint_uniform_color(CMAP6[segcounter]); 
      if segcounter == 0:
        finalobj = copy.deepcopy(self.mesh);
      else:
        finalobj += copy.deepcopy(self.mesh);
        
      framefeature[segcounter, :] = [
        T_Nr, C_Nr, N_d, N_a, C_p, C_n, PE_lg, PE_el, SA, VOL, rad, h_ratio
      ]
      segcounter += 1
    ################################# Generate the final fpfh features #################################
    if (not clear):
      o3d.io.write_triangle_mesh(f"{tempname}.ply", finalobj, write_ascii=True);
      with open(f"{tempname}.pdb", "w") as file1:
        file1.write(pdb_final)

    fpfh_final = self.fpfh_down(finalobj);
    return framefeature.reshape(-1), finalobj, fpfh_final
    
  
  def atom_type_count(self, theidxi):
    """
    Descriptor 1 and 2:
    Return: 
      Atom counts as a dictionary
    """
    ATOM_DICT = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
    atomic_numbers = np.array([i.atomic_number for i in self.atoms[theidxi]]); 
    atom_number   = len(atomic_numbers); 
    count={}
    for atom, atom_num in ATOM_DICT.items(): 
      if np.count_nonzero(atomic_numbers - atom_num == 0) > 0: 
        count[atom] = np.count_nonzero(atomic_numbers - atom_num == 0); 
    return count
  
  def hbp_count(self, theidxi, **kwarg):
    """
    Descriptor 3 and 4: Counter hydrogen bond donor and acceptors. 
    Return: 
      number_d: Number of hydrogen bond donor
      number_a: Number of hydrogen bond acceptor
    """
    coord_d, coord_a = chemtools.DACbytraj(self.traj, self.frame, self.resmask, **kwarg);
    withinbox_d = utils.boxfilter(coord_d, self.center, self.length);
    withinbox_a = utils.boxfilter(coord_a, self.center, self.length);
    number_d = len(withinbox_d);
    number_a = len(withinbox_a);
    return number_d, number_a
  
  def partial_charge(self, theidxi):
    """
    Descriptor 5 and 6: Counter positive partial charge and negative partial charge.
    Return: 
      charge_p: Number of positive partial charge
      charge_n: Number of negative partial charge
    """
    charges = {i:j for i,j in self.charges.items()}
    withinbox = utils.boxfilter(np.array([i for i in charges.keys()]), self.center, self.length);
    charge_p = sum([charges[tuple(i)] for i in withinbox if charges[tuple(i)]>0]);
    charge_n = sum([charges[tuple(i)] for i in withinbox if charges[tuple(i)]<0]);
    return charge_p, charge_n
  
  def pseudo_energy(self, theidxi):
    """
    Descriptor 7 and 8: Compute the pseudo-lj and pseudo-elec potential
    Return: 
      pp_lj: Pseudo Lenar-Jones potential
      pp_elec: Pseudo Electrostatic potential
    """
    charges = {i:j for i,j in self.charges.items()}; 
    charge_arr = chargedict2array(self.traj, self.frame, charges);
    pp_lj = pseudo_energy(self.traj, self.frame, theidxi, "lj");
    pp_elec = pseudo_energy(self.traj, self.frame, theidxi, "elec", charges=charge_arr);
    return pp_lj, pp_elec

  # Descriptor 9 and 10: Surface area and volume of the mesh
  def volume(self, mesh): 
    """
    Volume computation is not robust enough
    """
    try: 
      VOL = mesh.get_volume();
    except: 
      VOL  = 1.5 * mesh.get_surface_area();
    return VOL
    
  def surface(self, mesh): 
    return mesh.get_surface_area();
  
  def mean_radius(self, mesh, samples=600): 
    pcd = mesh.sample_points_uniformly(samples);
    mean_radius = np.linalg.norm(np.asarray(pcd.points) - pcd.get_center(), axis=1).mean()
    return mean_radius
  
  def convex_hull_ratio(self, mesh, samples=600): 
    pcd = mesh.sample_points_uniformly(samples);
    hull, _ = pcd.compute_convex_hull();
    hull_ratio = len(hull.vertices)/samples;
    return hull_ratio
  
  def fpfh_down(self, mesh, samples=600, origin=True): 
    """
    TODO: add support for the voxel-base down sampling 
    """
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


####################################################################################################
###################################### Open3D object display #######################################
####################################################################################################
def displayfiles(plyfiles, outfile="", add=[]): 
  cmap = CMAP6;
  objs = []; 
  finalobj = 0; 
  for obji, plyfile in enumerate(plyfiles): 
    color = cmap[obji]; 
    mesh = o3d.io.read_triangle_mesh(plyfile); 
    mesh.compute_vertex_normals(); 
    mesh.paint_uniform_color(color); 
    objs.append(mesh);
    if obji == 0: 
      finalobj = mesh; 
    else: 
      finalobj += mesh;
  write(finalobj, outfile=outfile); 
  display(objs, add=add); 
  return objs

def write(objs, outfile=""):
  if len(outfile) > 0:
    o3d.io.write_triangle_mesh(outfile, objs, write_ascii=True); 
    if os.path.isfile(outfile): 
      return True 
  else: 
    return False

def display(objects, add=[]):
  if len(objects) == 0 and len(add)==0:
    return []
  else: 
    objs = copy.deepcopy(objects); 
    cmap = CMAP6; 
    for i in range(1, len(objs)):
      color = cmap[i];
      objs[i].paint_uniform_color(color);
      if isinstance(objs[i], o3d.geometry.TriangleMesh): 
        objs[i].compute_vertex_normals();
    o3d.visualization.draw_geometries(add+objs, width=1200, height=1000);

def display_registration(source, target, transformation=np.eye(4)):
  source_temp = copy.deepcopy(source); 
  target_temp = copy.deepcopy(target); 
  source_temp.paint_uniform_color(CMAP6[1]); 
  target_temp.paint_uniform_color(CMAP6[-1]); 
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
  points = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
  ])
  points = points*length; 
  points = points + np.array(center)-(length/2); 
  lines = np.array([
    [0, 1], [0, 2], [1, 3], [2, 3],
    [4, 5], [4, 6], [5, 7], [6, 7],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ]); 
  colors = [[0, 0, 0] for i in range(len(lines))]
  line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
  ); 
  line_set.colors = o3d.utility.Vector3dVector(colors); 
  return line_set

def NewCoordFrame(center=[0,0,0], scale=1): 
  coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(); 
  coord_frame.scale(scale=scale, center=center)
  coord_frame.translate(center, relative=False)
  return coord_frame

