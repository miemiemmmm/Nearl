import os, sys, re, time, subprocess, tempfile
import pytraj as pt
import numpy as np 
import open3d as o3d
from itertools import combinations

ATOM_PATTERNS = {0: '^[0-9]*H.*$', 1: '^[0-9]*D.*$', 2: '^O.*$', 3: '^CA$', 4: '^CD$', 5: '^CD  $', 6: '^CA$', 7: '^N$', 8: '^CA$', 9: '^C$', 10: '^O$', 11: '^P$', 12: '^CB$', 13: '^CB$', 14: '^CB$', 15: '^CG$', 16: '^CG$', 17: '^CG$', 18: '^CG$', 19: '^O1$', 20: '^O2$', 21: '^CH3$', 22: '^CD$', 23: '^NE$', 24: '^RE$', 25: '^CZ$', 26: '^NH[12][AB]?$', 27: '^RH[12][AB]?$', 28: '^OD1$', 29: '^ND2$', 30: '^AD1$', 31: '^AD2$', 32: '^OD[12][AB]?$', 33: '^ED[12][AB]?$', 34: '^OD1[AB]?$', 35: '^ND2$', 36: '^AD1$', 37: '^AD2$', 38: '^OD2$', 39: '^LP[12]$', 40: '^SG$', 41: '^SG$', 42: '^OE[12][AB]?$', 43: '^EE[12][AB]?$', 44: '^CD$', 45: '^OE1$', 46: '^NE2$', 47: '^AE[12]$', 48: '^CE1|CD2$', 49: '^ND1$', 50: '^ND1$', 51: '^RD1$', 52: '^NE2$', 53: '^RE2$', 54: '^NE2$', 55: '^RE2$', 56: '^A[DE][12]$', 57: '^CG1$', 58: '^CG2$', 59: '^CD|CD1$', 60: '^CD1$', 61: '^CD2$', 62: '^C[GDE]$', 63: '^NZ$', 64: '^KZ$', 65: '^SD$', 66: '^CE$', 67: '^C[DE][12]$', 68: '^CZ$', 69: '^C[GD]$', 70: '^SE$', 71: '^SEG$', 72: '^OD1$', 73: '^OD2$', 74: '^OG$', 75: '^OG1$', 76: '^CG2$', 77: '^CD1$', 78: '^CD2$', 79: '^CE2$', 80: '^NE1$', 81: '^CE3$', 82: '^CZ2$', 83: '^CZ3$', 84: '^CH2$', 85: '^C[DE][12]$', 86: '^CZ$', 87: '^OH$', 88: '^CG1$', 89: '^CG2$', 90: '^CD$', 91: '^CE$', 92: '^FE[1-7]$', 93: '^S[1-7]$', 94: '^OXO$', 95: '^FE1$', 96: '^FE2$', 97: '^O1$', 98: '^O2$', 99: '^FE$', 100: '^CH[A-D]$', 101: '^N[A-D]$', 102: '^N [A-D]$', 103: '^C[1-4][A-D]$', 104: '^CM[A-D]$', 105: '^C[AB][AD]$', 106: '^CG[AD]$', 107: '^O[12][AD]$', 108: '^C[AB][BC]$', 109: '^OH2$', 110: '^N[123]$', 111: '^C1$', 112: '^C2$', 113: '^C3$', 114: '^C4$', 115: '^C5$', 116: '^C6$', 117: '^O7$', 118: '^O8$', 119: '^S$', 120: '^O[1234]$', 121: '^O[1234]$', 122: '^O4$', 123: '^P1$', 124: '^O[123]$', 125: '^C[12]$', 126: '^N1$', 127: '^C[345]$', 128: '^BAL$', 129: '^POI$', 130: '^DOT$', 131: '^CU$', 132: '^ZN$', 133: '^MN$', 134: '^FE$', 135: '^MG$', 136: '^MN$', 137: '^CO$', 138: '^SE$', 139: '^YB$', 140: '^N1$', 141: '^C[2478]$', 142: '^O2$', 143: '^N3$', 144: '^O4$', 145: '^C[459]A$', 146: '^N5$', 147: '^C[69]$', 148: '^C[78]M$', 149: '^N10$', 150: '^C10$', 151: '^C[12345]\\*$', 152: '^O[234]\\*$', 153: '^O5\\*$', 154: '^OP[1-3]$', 155: '^OT1$', 156: '^C01$', 157: '^C16$', 158: '^C14$', 159: '^C.*$', 160: '^SEG$', 161: '^OXT$', 162: '^OT.*$', 163: '^E.*$', 164: '^S.*$', 165: '^C.*$', 166: '^A.*$', 167: '^O.*$', 168: '^N.*$', 169: '^R.*$', 170: '^K.*$', 171: '^P[A-D]$', 172: '^P.*$', 173: '^.O.*$', 174: '^.N.*$', 175: '^.C.*$', 176: '^.P.*$', 177: '^.H.*$'}
RESIDUE_PATTERNS = {0: '^.*$', 1: '^.*$', 2: '^WAT|HOH|H2O|DOD|DIS$', 3: '^CA$', 4: '^CD$', 5: '^.*$', 6: '^ACE$', 7: '^.*$', 8: '^.*$', 9: '^.*$', 10: '^.*$', 11: '^.*$', 12: '^ALA$', 13: '^ILE|THR|VAL$', 14: '^.*$', 15: '^ASN|ASP|ASX|HIS|HIP|HIE|HID|HISN|HISL|LEU|PHE|TRP|TYR$', 16: '^ARG|GLU|GLN|GLX|MET$', 17: '^LEU$', 18: '^.*$', 19: '^GLN$', 20: '^GLN$', 21: '^ACE$', 22: '^ARG$', 23: '^ARG$', 24: '^ARG$', 25: '^ARG$', 26: '^ARG$', 27: '^ARG$', 28: '^ASN$', 29: '^ASN$', 30: '^ASN$', 31: '^ASN$', 32: '^ASP$', 33: '^ASP$', 34: '^ASX$', 35: '^ASX$', 36: '^ASX$', 37: '^ASX$', 38: '^ASX$', 39: '^CYS|MET$', 40: '^CY[SXM]$', 41: '^CYH$', 42: '^GLU$', 43: '^GLU$', 44: '^GLU|GLN|GLX$', 45: '^GLN$', 46: '^GLN$', 47: '^GLN|GLX$', 48: '^HIS|HID|HIE|HIP|HISL$', 49: '^HIS|HIE|HISL$', 50: '^HID|HIP$', 51: '^HID|HIP$', 52: '^HIS|HIE|HIP$', 53: '^HIS|HIE|HIP$', 54: '^HID|HISL$', 55: '^HID|HISL$', 56: '^HIS|HID|HIP|HISD$', 57: '^ILE$', 58: '^ILE$', 59: '^ILE$', 60: '^LEU$', 61: '^LEU$', 62: '^LYS$', 63: '^LYS$', 64: '^LYS$', 65: '^MET$', 66: '^MET$', 67: '^PHE$', 68: '^PHE$', 69: '^PRO|CPR$', 70: '^CSO$', 71: '^CSO$', 72: '^CSO$', 73: '^CSO$', 74: '^SER$', 75: '^THR$', 76: '^THR$', 77: '^TRP$', 78: '^TRP$', 79: '^TRP$', 80: '^TRP$', 81: '^TRP$', 82: '^TRP$', 83: '^TRP$', 84: '^TRP$', 85: '^TYR$', 86: '^TYR$', 87: '^TYR$', 88: '^VAL$', 89: '^VAL$', 90: '^.*$', 91: '^.*$', 92: '^FS[34]$', 93: '^FS[34]$', 94: '^FS3$', 95: '^FEO$', 96: '^FEO$', 97: '^HEM$', 98: '^HEM$', 99: '^HEM$', 100: '^HEM$', 101: '^HEM$', 102: '^HEM$', 103: '^HEM$', 104: '^HEM$', 105: '^HEM$', 106: '^HEM$', 107: '^HEM$', 108: '^HEM$', 109: '^HEM$', 110: '^AZI$', 111: '^MPD$', 112: '^MPD$', 113: '^MPD$', 114: '^MPD$', 115: '^MPD$', 116: '^MPD$', 117: '^MPD$', 118: '^MPD$', 119: '^SO4|SUL$', 120: '^SO4|SUL$', 121: '^PO4|PHO$', 122: '^PC$', 123: '^PC$', 124: '^PC$', 125: '^PC$', 126: '^PC$', 127: '^PC$', 128: '^BIG$', 129: '^POI$', 130: '^DOT$', 131: '^.*$', 132: '^.*$', 133: '^.*$', 134: '^.*$', 135: '^.*$', 136: '^.*$', 137: '^.*$', 138: '^.*$', 139: '^.*$', 140: '^FMN$', 141: '^FMN$', 142: '^FMN$', 143: '^FMN$', 144: '^FMN$', 145: '^FMN$', 146: '^FMN$', 147: '^FMN$', 148: '^FMN$', 149: '^FMN$', 150: '^FMN$', 151: '^FMN$', 152: '^FMN$', 153: '^FMN$', 154: '^FMN$', 155: '^ALK|MYR$', 156: '^ALK|MYR$', 157: '^ALK$', 158: '^MYR$', 159: '^ALK|MYR$', 160: '^.*$', 161: '^.*$', 162: '^.*$', 163: '^.*$', 164: '^.*$', 165: '^.*$', 166: '^.*$', 167: '^.*$', 168: '^.*$', 169: '^.*$', 170: '^.*$', 171: '^.*$', 172: '^.*$', 173: '^FAD|NAD|AMX|APU$', 174: '^FAD|NAD|AMX|APU$', 175: '^FAD|NAD|AMX|APU$', 176: '^FAD|NAD|AMX|APU$', 177: '^FAD|NAD|AMX|APU$'}
EXP_RADII = {1: 1.4, 2: 1.4, 3: 1.4, 4: 1.54, 5: 1.54, 6: 1.54, 7: 1.74, 8: 1.74, 9: 1.74, 10: 1.74, 11: 1.74, 12: 1.8, 13: 1.8, 14: 1.54, 15: 1.2, 16: 0.0, 17: 5.0, 18: 1.97, 19: 1.4, 20: 1.4, 21: 1.3, 22: 1.49, 23: 0.01, 24: 1.2, 25: 1.24, 26: 1.6, 27: 1.24, 28: 1.25, 29: 2.15, 30: 3.0, 31: 1.15, 38: 1.8}
UNITED_RADII = {1: 1.4, 2: 1.6, 3: 1.4, 4: 1.7, 5: 1.8, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0, 10: 1.74, 11: 1.86, 12: 1.85, 13: 1.8, 14: 1.54, 15: 1.2, 16: 1.5, 17: 5.0, 18: 1.97, 19: 1.4, 20: 1.4, 21: 1.3, 22: 1.49, 23: 0.01, 24: 0.0, 25: 1.24, 26: 1.6, 27: 1.24, 28: 1.25, 29: 2.15, 30: 3.0, 31: 1.15, 38: 1.8}
ATOM_NUM = {0: 15, 1: 15, 2: 2, 3: 18, 4: 22, 5: 22, 6: 9, 7: 4, 8: 7, 9: 10, 10: 1, 11: 13, 12: 9, 13: 7, 14: 8, 15: 10, 16: 8, 17: 7, 18: 8, 19: 3, 20: 3, 21: 9, 22: 8, 23: 4, 24: 4, 25: 10, 26: 5, 27: 5, 28: 1, 29: 5, 30: 3, 31: 3, 32: 3, 33: 3, 34: 1, 35: 5, 36: 3, 37: 3, 38: 3, 39: 13, 40: 13, 41: 12, 42: 3, 43: 3, 44: 10, 45: 1, 46: 5, 47: 3, 48: 11, 49: 14, 50: 4, 51: 4, 52: 4, 53: 4, 54: 14, 55: 14, 56: 4, 57: 8, 58: 9, 59: 9, 60: 9, 61: 9, 62: 8, 63: 6, 64: 6, 65: 13, 66: 9, 67: 11, 68: 11, 69: 8, 70: 9, 71: 9, 72: 3, 73: 3, 74: 2, 75: 2, 76: 9, 77: 11, 78: 10, 79: 10, 80: 4, 81: 11, 82: 11, 83: 11, 84: 11, 85: 11, 86: 10, 87: 2, 88: 9, 89: 9, 90: 8, 91: 8, 92: 21, 93: 13, 94: 1, 95: 21, 96: 21, 97: 1, 98: 1, 99: 21, 100: 11, 101: 14, 102: 14, 103: 10, 104: 9, 105: 8, 106: 10, 107: 3, 108: 11, 109: 2, 110: 14, 111: 9, 112: 10, 113: 8, 114: 7, 115: 9, 116: 9, 117: 2, 118: 2, 119: 13, 120: 3, 121: 3, 122: 3, 123: 13, 124: 3, 125: 8, 126: 14, 127: 9, 128: 17, 129: 23, 130: 23, 131: 20, 132: 19, 133: 24, 134: 25, 135: 26, 136: 27, 137: 28, 138: 29, 139: 31, 140: 4, 141: 10, 142: 1, 143: 14, 144: 1, 145: 10, 146: 4, 147: 11, 148: 9, 149: 4, 150: 10, 151: 8, 152: 2, 153: 3, 154: 3, 155: 3, 156: 10, 157: 9, 158: 9, 159: 8, 160: 9, 161: 3, 162: 3, 163: 3, 164: 13, 165: 7, 166: 11, 167: 1, 168: 4, 169: 4, 170: 6, 171: 13, 172: 13, 173: 1, 174: 4, 175: 7, 176: 13, 177: 15}

def genslices(traj, center, cube_length, threshold=2): 
  """
  Generate a slice from all frame of a trajectory (Each frame takes one dimension)
  Returned results are the atomic coordinates and the atomic indices
  """
  center = np.array(center); 
  upperbound = np.array(center) + cube_length/2; 
  lowerbound = np.array(center) - cube_length/2; 
  ret_arr = []; 
  idx_arr = []; 
  atoms = [a for a in traj.top.atoms]; 
  for xyz in traj.xyz: 
    ubstate = np.array([np.prod(i) for i in xyz < upperbound]); 
    lbstate = np.array([np.prod(i) for i in xyz > lowerbound]); 
    state = [bool(i) for i in ubstate*lbstate]; 
    s_final = []; 
    lastres = -999; 
    seg_counter = 0; 
    for idx, state in enumerate(state): 
      if state: 
        if atoms[idx].resid - lastres > threshold: 
          seg_counter += 1; 
        s_final.append(seg_counter)
        lastres = atoms[idx].resid; 
        # print("resid", atoms[idx].resid, state, "; Diff:",atoms[idx].resid-lastres, "; segment: ", seg_counter)
      else: 
        s_final.append(0); 
    # print(np.count_nonzero(s_final))
    ret_arr.append(xyz[np.array([bool(i) for i in s_final])])
    idx_arr.append(s_final)
  return ret_arr, np.array(idx_arr)


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
  subprocess.run([msms, "-if", inputfile, "-of", outfile, "-density", str(d), "-probe_radius", str(r)], stdout=subprocess.DEVNULL); 
  if os.path.isfile(f"{outfile}.vert") and os.path.isfile(f"{outfile}.face"): 
    return True
  else: 
    print(f"{runmsms.__name__:15s}: Failed to generate corresponding vertex and triangle file"); 
    return False

def pdb2msms(msms, pdbfile, outprefix): 
  xyzrfile = pdb2xyzr(pdbfile, write=outprefix)
  ret = runmsms(msms, xyzrfile, outprefix, d = 4, r = 1.4)
  if ret:
    print(f"{pdb2msms.__name__:15s}: Successfully generated the MSMS output")
  else:
    print(f"{pdb2msms.__name__:15s}: Failed to generate the MSMS output")

def traj2msms(msms, traj, indice, frame=0, out_prefix="/tmp/test", force=False, d=4, r=1.5):
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
      print(f"{traj2msms.__name__:15s}: Successfully generated the MSMS output")
    else: 
      print(f"{traj2msms.__name__:15s}: Failed to generate the MSMS output")

####################################################################################################
################## 3D object processing after the MSMS vertex/triangle generation ##################
####################################################################################################
def vert2xyzn(filename, outfilename): 
  """
  Convert the MSMS vertex file to Open3D readable point cloud file (.xyzn, coordinate X,Y,Z and their normal)
  """
  with open(filename, "r") as file1: 
    c = 0; 
    xyzns = []; 
    for line in file1: 
      if "#" in line: 
        continue
      elif len(line) < 9: 
        continue
      else: 
        c += 1
        if c == 1:
          pass
        else: 
          xyzns.append([float(i) for i in line.strip().split()[:6]]); 
  print(f"{vert2xyzn.__name__:15s}: The 3D object contains {len(xyzns)} points")
  with open(outfilename, "w") as file2: 
    for i in xyzns: 
      file2.write(f"{i[0]:10.3f}{i[1]:10.3f}{i[2]:10.3f}{i[3]:10.3f}{i[4]:10.3f}{i[5]:10.3f}\n")

def readtriangle(filename): 
  """
  Read the MSMS triangle faces file for the generation of triangle mesh
  """
  with open(filename, "r") as file1: 
    c = 0; 
    triangles = []; 
    for line in file1: 
      if "#" in line: 
        continue
      elif len(line) < 9: 
        continue
      else: 
        c += 1
        if c == 1:
          pass
        else: 
          triangles.append([int(i) for i in line.strip().split()[:3]]); 
  print(f"{readtriangle.__name__:15s}: The 3D object contains {len(triangles)} triangles ")
  return np.array(triangles)

def msms2obj(prefix, outfile): 
  """
  Convert the MSMS output (vertex and triangle faces) to a triangle mesh readable by Open3D
  """
  vertfile = f"{prefix}.vert"; 
  facefile = f"{prefix}.face"; 
  if not os.path.isfile(vertfile) or not os.path.isfile(facefile): 
    raise Exception(f"{msms2obj.__name__:15s}: Cannot find the MSMS output files (.vert/.face) based on the given prefix")
  with tempfile.NamedTemporaryFile(suffix=".xyzn") as xyznfile: 
    vert2xyzn(vertfile, xyznfile.name); 
    pcd = o3d.io.read_point_cloud(xyznfile.name); 
  triangles = readtriangle(facefile); 
  mesh = o3d.geometry.TriangleMesh(); 
  mesh.vertices = pcd.points; 
  mesh.vertex_normals = pcd.normals; 
  mesh.triangles = o3d.utility.Vector3iVector(triangles-1); 
  mesh.compute_vertex_normals(); 
  ret = o3d.io.write_triangle_mesh(outfile, mesh);
  if ret: 
    print(f"{msms2obj.__name__:15s}: Successfully saved the object as triangle mesh")
  else: 
    print(f"{msms2obj.__name__:15s}: Failed to save the object, please check the MSMS output")

# Pseudo LJ
def pseudo_lj(r, epsilon=1, sigma=1):
  """
  Calculates the Lennard-Jones potential for a given distance
  """
  return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
# Pseudo Elec
def pseudo_elec(q1, q2, r):
  """
  Calculates the Coulombic interaction energy between two atoms
  """
  k = 8.9875517923e9   # Coulomb constant in N·m^2·C^-2
  return k*q1*q2/r

def pseudo_energy(traj, idxs, mode): 
  # Using the following code for distance calculation, speeds up a bit
  # dist = math.sqrt(sum((p1-p2)**2)); 
  mode = mode.lower(); 
  pairs = list(combinations(idxs, 2)); 
  #   print(f"{pseudo_energy.__name__:15s}: Obtained {len(pairs)} atom pairs from the following list({len(theidxi)} elements): {list(theidxi)}")
  energy_final = 0; 
  if mode == "lj": 
    for p in pairs: 
      p1 = traj.xyz[0][p[0]];
      p2 = traj.xyz[0][p[1]];
      dist = np.linalg.norm(p1-p2);   
      energy_final += pseudo_lj(dist);
  elif mode == "elec": 
    atmlst = list(traj.top.atoms); 
    for p in pairs: 
      q1 = traj.top.charge[p[0]];
      q2 = traj.top.charge[p[1]];
      p1 = traj.xyz[0][p[0]];
      p2 = traj.xyz[0][p[1]];
      dist = np.linalg.norm(p1-p2);
      energy_final += pseudo_elec(q1, q2, dist);
  else: 
    raise Exception(f"{pseudo_energy.__name__:15s}: Only two pseudo-energy evaluation modes are supported: Lenar-jones (lj) and Electrostatic (elec)")
  print(f"{pseudo_energy.__name__:15s}: Energy {mode:>4s} -> {energy_final:>10.6f} (based on {len(pairs)} atom pairs)")
  return energy_final

def displayit(plyfiles): 
  cmap = [[0.087411, 0.044556, 0.224813], [0.354032, 0.066925, 0.430906], [0.60933, 0.159474, 0.393589], [0.841969, 0.292933, 0.248564], [0.974176, 0.53678, 0.048392], [0.964394, 0.843848, 0.273391]]
  objs = []; 
  for obji, plyfile in enumerate(plyfiles): 
    color = cmap[obji]; 
    mesh = o3d.io.read_triangle_mesh(plyfile); 
    mesh.compute_vertex_normals(); 
    mesh.paint_uniform_color(color); 
    objs.append(mesh);
  o3d.visualization.draw_geometries(objs); 
  return objs
 

