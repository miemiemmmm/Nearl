import time
import pytraj as pt

from BetaPose import utils, trajloader
from BetaPose import data_io

class misato_traj(trajloader.Trajectory): 
  def __init__(self, pdbcode, _, *args, **kwarg): 
    mdfile = "/home/yzhang/Downloads/misato_database/MD.hdf5"; 
    parmdir = "/home/yzhang/Downloads/misato_database/parameter_restart_files_MD"; 
    thetraj = utils.misato_traj(pdbcode, mdfile, parmdir, *args, **kwarg); 
    super().__init__(thetraj, pdbcode); 


mdlist = "/home/yzhang/Downloads/misato_database/train_MD.txt"
with open(mdlist, "r") as f:
  trajlist = f.read().strip("\n").split("\n");

print(f"# Total number of trajectories {trajlist.__len__()}");
# print(trajlist.__len__());
# trajlist = trajlist[:200]

loader = trajloader.TrajectoryLoader(trajlist, trajlist)
loader.set_outtype(misato_traj)
st = time.perf_counter()
for idx, traj in enumerate(loader):
  coord = traj.xyz;
  sel_mol = traj.top.select(":MOL")
  if sel_mol.__len__() > 0:
    print(f"{traj.top_filename},FOUND_MOLATOM_{len(sel_mol)}")
  else: 
    print(f"{traj.top_filename},NOT_FOUND_MOL")
print(f"# Time elapsed {time.perf_counter() - st}");


