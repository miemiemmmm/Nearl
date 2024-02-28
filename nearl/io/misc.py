import os
import numpy as np
import pytraj as pt
from nearl.io import hdf5

def misato_traj(thepdb, mdfile, parmdir, *args, **kwargs):

  # Needs dbfile and parm_folder;
  topfile = f"{parmdir}/{thepdb.lower()}/production.top.gz"
  if not os.path.exists(topfile):
    print(f"The topology file of PDB:{thepdb} not found")
    return pt.Trajectory()

  top = pt.load_topology(topfile)
  res = set([i.name for i in top.residues])
  if "WAT" in res:
    top.strip(":WAT")
  if "Cl-" in res:
    top.strip(":Cl-")
  if "Na+" in res:
    top.strip(":Na+")

  with hdf5.hdf_operator(mdfile, "r") as f1:
    keys = f1.hdffile.keys()
    if thepdb.upper() in keys:
      coord = f1.data(f"/{thepdb.upper()}/trajectory_coordinates")
      coord = np.array(coord)
      ret_traj = pt.Trajectory(xyz=coord, top=top)
      return ret_traj
    else:
      print(f"Not found the key for PDB code {thepdb.upper()}")
      return pt.Trajectory()

