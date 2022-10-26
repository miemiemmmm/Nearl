import pytraj as pt 
import numpy as np 
from scipy.spatial import distance_matrix
import tempfile

def conflictfactor(pdbfile, ligname, cutoff=5):
  VDWRADII = {'1': 1.1, '2': 1.4, '3': 1.82, '4': 1.53, '5': 1.92, '6': 1.7, '7': 1.55, '8': 1.52,
    '9': 1.47, '10': 1.54, '11': 2.27, '12': 1.73, '13': 1.84, '14': 2.1, '15': 1.8,
    '16': 1.8, '17': 1.75, '18': 1.88, '19': 2.75, '20': 2.31, '28': 1.63, '29': 1.4,
    '30': 1.39, '31': 1.87, '32': 2.11, '34': 1.9, '35': 1.85, '46': 1.63, '47': 1.72,
    '48': 1.58, '50': 2.17, '51': 2.06, '53': 1.98, '54': 2.16, '55': 3.43, '56': 2.68,
    '78': 1.75, '79': 1.66, '82': 2.02, '83': 2.07
  }
  traj = pt.load(pdbfile, top=pdbfile);
  traj.top.set_reference(traj[0]);
  pocket_atoms = traj.top.select(f":{ligname}<:{cutoff}");
  atoms = np.array([*traj.top.atoms])[pocket_atoms];
  coords = traj.xyz[0][pocket_atoms];
  atomnr = len(pocket_atoms);
  cclash=0;
  ccontact = 0;
  for i, coord in enumerate(coords):
    partners = [atoms[i].index]
    for j in list(atoms[i].bonded_indices()):
      if j in pocket_atoms:
        partners.append(j)
    partners.sort()
    otheratoms = np.setdiff1d(pocket_atoms, partners)
    ret = distance_matrix([coord], traj.xyz[0][otheratoms])
    thisatom = atoms[i].atomic_number
    vdw_pairs = np.array([VDWRADII[str(i.atomic_number)] for i in np.array([*traj.top.atoms])[otheratoms]]) + VDWRADII[str(thisatom)]
    cclash += np.count_nonzero(ret < vdw_pairs - 1.25)
    ccontact += np.count_nonzero(ret < vdw_pairs + 0.4)

    st = (ret < vdw_pairs - 1.25)[0];
    if np.count_nonzero(st) > 0:
      partatoms = np.array([*traj.top.atoms])[otheratoms][st];
      thisatom = np.array([*traj.top.atoms])[atoms[i].index];
      for part in partatoms:
        dist = distance_matrix([traj.xyz[0][part.index]], [traj.xyz[0][thisatom.index]]);
        print(f"Found clash between: {thisatom.name}({thisatom.index}) and {part.name}({part.index}); Distance: {dist.squeeze().round(3)}")

  factor = 1 - ((cclash/2)/((ccontact/2)/atomnr))
  print(f"Clashing factor: {round(factor,3)}; Atom selected: {atomnr}; Contact number: {ccontact}; Clash number: {cclash}");
  return factor

# Two ways to get the mask of protein part
def GetProteinByTraj(traj):
  """
  Method 1: Get the mask of protein via a trajectory object
  """
  reslst = []
  for i in traj.top.atoms:
    if i.name=="CA":
      reslst.append(i.resid+1)
  mask = ":"+",".join([str(i) for i in reslst])
  return traj.top.select(mask)

def GetProteinMask(pdbfile):
  """
  Method 2: Get the mask of protein via a PDB file name 
  """
  reslst = [];
  traj = pt.load(pdbfile, top=pdbfile);
  for i in traj.top.atoms:
    if i.name=="CA":
      reslst.append(i.resid+1)
  mask = ":"+",".join([str(i) for i in reslst])
  return mask

def PairwiseDistance(traj, mask1, mask2, use_mean=False):
  """
  Get the pairwise distance between two masks
  Usually they are the heavy atoms of ligand and protein within the pocket
  """
  print("Calculating pairwise distance")
  selmask1 = traj.top.select(mask1);
  selmask2 = traj.top.select(mask2);
  atom_names = np.array([i.name for i in traj.top.atoms])
  atom_ids = np.array([i.index for i in traj.top.atoms])
  
  if use_mean == True:
    frame_mean = np.mean(traj.xyz, axis=0);
    this_ligxyz = frame_mean[selmask1];
    this_proxyz = frame_mean[selmask2];
    ref_frame = distance_matrix(this_ligxyz, this_proxyz);
  else:
    pdist = pt.pairwise_distance(traj, mask_1=mask1, mask_2=mask2);
    ref_frame = pdist[0][0];

  minindex = [np.where(ref_frame[i] == np.min(ref_frame[i]))[0][0] for i in range(len(ref_frame))]
  absolute_index = [selmask2[i] for i in minindex]
  min_dists = np.min(ref_frame, axis=1)
  # For mask selection, remember to add 1 because it is the read index number
  distlist = []
  for i,j in zip(selmask1, absolute_index):
    # print(f"Pairing: {atom_names[i]:>4}({atom_ids[i]:>6}) - {atom_names[j]:>4}({atom_ids[j]:>6})")
    dist_tmp = pt.distance(traj, f"@{i+1} @{j+1}")
    distlist.append(dist_tmp)

  distarr = np.array(distlist).astype(np.float32)
  
  gp1_names = list(atom_names[selmask1])
  gp1_ids  = list(atom_ids[selmask1])
  gp2_names = list(atom_names[absolute_index])
  gp2_ids  = list(atom_ids[absolute_index])

  return distarr, { "gp1_names":gp1_names, "gp1_ids":gp1_ids, "gp2_names":gp2_names, "gp2_ids":gp2_ids }

def combineMOL2PDB(mol2file, pdbfile, outfile):
  from rdkit import Chem; 
  lig = Chem.MolFromMol2File(mol2file)
  ligpdb = Chem.MolToPDBBlock(lig)
  
  atomlines = [i.replace("UNL", "LIG") for i in ligpdb.split("\n") if "HETATM" in i]
  with open(pdbfile, "r") as file1: 
    pdborig = file1.read(); 
  linesorig = [i for i in pdborig.split("\n") if "HETATM" in i or "ATOM" in i]
  finallines = linesorig + atomlines; 
  finalstr = "\n".join(finallines); 

  temp = tempfile.NamedTemporaryFile(suffix=".pdb")
  temp.write(bytes(finalstr, "utf-8"))
  
  traj = pt.load(temp.name)
  
  temp.close()
  pt.save(outfile, traj, overwrite=True)
  return outfile

def ASALig(pdbfile, lig_mask):
  """
  Calculate the Ligand accessible surface area (SAS) contribution with respect to the protein-ligand complex.
  """
  import subprocess; 
  temp = tempfile.NamedTemporaryFile(suffix=".dat");
  tempname = temp.name;
  pro_mask = GetProteinMask(pdbfile);

  cpptraj_str = f"parm {pdbfile}\ntrajin {pdbfile}\nsurf {lig_mask} solutemask {pro_mask},{lig_mask} out {tempname}";
  p1 = subprocess.Popen(["echo", "-e", cpptraj_str], stdout=subprocess.PIPE) ;
  p2 = subprocess.Popen(["cpptraj"], stdin=p1.stdout, stdout=subprocess.DEVNULL);
  p1.wait();
  p2.wait();

  with open(tempname, "r") as file1: 
    lines = [i.strip() for i in file1.read().strip("\n").split("\n") if i.strip()[0]!="#"]; 
    f_val = float(lines[0].split()[1]); 
  temp.close();
  return f_val

def ASALigOnly(pdbfile, lig_mask): 
  """
  Calculate the LIGAND accessible surface area (ASA) only. (Other components are not loaded)
  """
  traj = pt.load(pdbfile, top=pdbfile, mask=lig_mask)
  sel = traj.top.select(lig_mask)
  surf = pt.surf(traj, lig_mask)
  return float(surf.round(3)[0].item())

def EmbeddingFactor(basepath, pdbcode, mask=":LIG"):
  """
  Embedding factor is measured by the accessible surface area (ASA) contribution of ligand in a complex
  to the pure ligand ASA
  """
  import os; 
  pdbcode = pdbcode.lower(); 
  basepath = os.path.abspath(basepath); 
  ligfile = os.path.join(basepath, f"{pdbcode}/{pdbcode}_ligand.mol2");
  pdbfile = os.path.join(basepath, f"{pdbcode}/{pdbcode}_protein.pdb");
  outfile = os.path.join(basepath, f"{pdbcode}/{pdbcode}_complex.pdb");

  if (os.path.isfile(ligfile)) and (os.path.isfile(pdbfile)):
    outfile = combineMOL2PDB(ligfile, pdbfile, outfile)
    slig_0 = ASALig(outfile, mask)
    slig_1 = ASALigOnly(outfile, mask)
  elif not os.path.isfile(ligfile):
    print(f"Cannot find the ligand file in the database {pdbcode} ({ligfile})")
  elif not os.path.isfile(pdbfile):
    print(f"Cannot find the protein file in the database {pdbcode} ({pdbfile})")
  print(f"Surface contribution: {slig_0}; Surface pure: {slig_1}")
  return 1-slig_0/slig_1


def cgenff2dic(filename):
  with open(filename) as file1:
    lst = list(filter(lambda i: re.match(r"^ATOM.*!", i), file1))
  theatom  = [i.strip("\n").split()[1] for i in lst]
  atomtype = [i.strip("\n").split()[2] for i in lst]
  charge   = [float(i.strip("\n").split()[3]) for i in lst]
  penalty  = [float(i.strip("\n").split()[-1]) for i in lst]
  return {"name":theatom, "type":atomtype, "charge":charge, "penalty":penalty}

def cgenff2xmls(cgenffname):
  cgenffdic = cgenff2dic(cgenffname); 
  root = ET.Element('ForceField')
  info = ET.SubElement(root, 'Info')
  info_date = ET.SubElement(info, "date")
  info_date.text = datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S');
  data_lig = ET.SubElement(root, 'LIG')
  for i in range(len(cgenffdic["name"])):
    tmpattrib={
      "name":cgenffdic["name"][i], 
      "type": cgenffdic["type"][i], 
      "charge": str(cgenffdic["charge"][i]), 
      'penalty': str(cgenffdic["penalty"][i]),
    }
    tmpatom = ET.SubElement(data_lig, 'ATOM', attrib = tmpattrib)
  ligxml_str = ET.tostring(root , encoding="unicode")
  ligxml_str = minidom.parseString(ligxml_str).toprettyxml(indent="  ")
  return ligxml_str

def cgenff2xml(cgenffname, outfile):
  xmlstr = cgenff2xmls(cgenffname); 
  with open(outfile, "w") as file1: 
    file1.write(xmlstr)
  return 

