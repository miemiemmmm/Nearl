import tempfile, re

import numpy as np
import pytraj as pt
from scipy.spatial import distance_matrix

def DistanceLigPro(theid, mode="session", ligname="LIG"):
  """
    Calculate the COM distance from protein to ligand
    Protein uses CA atoms ; Ligand use ALL atoms
    Have 2 modes: session/file
  """
  from . import session_prep
  if mode == "session":
    from rdkit import Chem
    if len(theid) != 8:
      print("Session ID length not equil to 8")
      return
    with tempfile.NamedTemporaryFile("w", suffix=".pdb") as file1, tempfile.NamedTemporaryFile("w", suffix=".mol2") as file2:
      session = session_prep.RecallSession(theid)
      file1.write(session["pdbfile"])
      protcom = pt.center_of_mass(pt.load(file1.name), "@CA")
      try:
        # Mol2 could successfully be parsed in pytraj
        file2.write(session["molfile"])
        traj = pt.load(file2.name)
        ligcom = pt.center_of_mass(pt.load(file2.name))
      except Exception as e:
        # Directly calculate the COM of the ligand
        # print(f"Error occurred while calculating the Ligand COM: {e}")
        atoms = session["molfile"].split("@<TRIPOS>ATOM\n")[1].split("@<TRIPOS>")[0]
        atoms = [i.strip().split() for i in atoms.strip("\n").split("\n")]
        coord = np.array([i[2:5] for i in atoms]).astype(np.float32)
        atomtypes = [re.sub(r"[0-9]", "", i[1]) for i in atoms]
        masses = []
        for i in atomtypes:
          try:
            m = Chem.Atom(i).GetMass()
            masses.append(m)
          except:
            masses.append(0)
        com = np.average(coord, axis=0, weights=masses)
        ligcom = np.array([com])
      return distance_matrix(ligcom, protcom).item()
  elif mode == "file":
    traj = pt.load(theid)
    dist = pt.distance(traj, f"@CA  :{ligname}")
    return dist.item()
  else:
    return None



def ASALig(pdbfile, lig_mask):
  """
  Calculate the Ligand accessible surface area (SAS) contribution with respect to the protein-ligand complex.
  """
  import subprocess
  temp = tempfile.NamedTemporaryFile(suffix=".dat")
  tempname = temp.name
  pro_mask = get_protein_mask(pdbfile)

  cpptraj_str = f"parm {pdbfile}\ntrajin {pdbfile}\nsurf {lig_mask} solutemask {pro_mask},{lig_mask} out {tempname}"
  p1 = subprocess.Popen(["echo", "-e", cpptraj_str], stdout=subprocess.PIPE)
  p2 = subprocess.Popen(["cpptraj"], stdin=p1.stdout, stdout=subprocess.DEVNULL)
  p1.wait()
  p2.wait()

  with open(tempname, "r") as file1:
    lines = [i.strip() for i in file1.read().strip("\n").split("\n") if i.strip()[0]!="#"]
    f_val = float(lines[0].split()[1])
  temp.close()
  return f_val

def ASALigOnly(pdbfile, lig_mask):
  """
  Calculate the LIGAND accessible surface area (ASA) only. (Other components are not loaded)
  """
  traj = pt.load(pdbfile, top=pdbfile, mask=lig_mask)
  sel = traj.top.select(lig_mask)
  surf = pt.surf(traj, lig_mask)
  return float(surf.round(3)[0].item())

def embedding_factor(basepath, pdbcode, mask=":LIG"):
  """
  Embedding factor is measured by the accessible surface area (ASA) contribution of ligand in a complex
  to the pure ligand ASA
  """
  pdbcode = pdbcode.lower()
  basepath = os.path.abspath(basepath)
  ligfile = os.path.join(basepath, f"{pdbcode}/{pdbcode}_ligand.mol2")
  pdbfile = os.path.join(basepath, f"{pdbcode}/{pdbcode}_protein.pdb")
  outfile = os.path.join(basepath, f"{pdbcode}/{pdbcode}_complex.pdb")

  if (os.path.isfile(ligfile)) and (os.path.isfile(pdbfile)):
    outfile = combine_molpdb(ligfile, pdbfile, outfile)
    slig_0 = ASALig(outfile, mask)
    slig_1 = ASALigOnly(outfile, mask)
  elif not os.path.isfile(ligfile):
    print(f"Cannot find the ligand file in the database {pdbcode} ({ligfile})")
  elif not os.path.isfile(pdbfile):
    print(f"Cannot find the protein file in the database {pdbcode} ({pdbfile})")
  # print(f"Surface contribution: {slig_0}; Surface pure: {slig_1}")
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
  cgenffdic = cgenff2dic(cgenffname)
  root = ET.Element('ForceField')
  info = ET.SubElement(root, 'Info')
  info_date = ET.SubElement(info, "date")
  info_date.text = datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S')
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
  xmlstr = cgenff2xmls(cgenffname)
  with open(outfile, "w") as file1:
    file1.write(xmlstr)
  return