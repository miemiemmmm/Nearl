import os, h5py, sys
import numpy as np
import pytraj as pt

from nearl.io import Trajectory
import nearl.io, nearl.featurizer, nearl.features 


class MisatoTraj(Trajectory): 
  def __init__(self, pdbcode, misatodir, **kwarg): 
    # Locate the topology and trajectory files based on the directory of MISATO dataset 
    self.topfile = f"{misatodir}/parameter_restart_files_MD/{pdbcode.lower()}/production.top.gz"
    self.trajfile = os.path.join(misatodir, f"MD.hdf5")

    # IMPORTANT: Original topolgy contains water and ions 
    # IMPORTANT: Remove them to align the coordinates with the topology 
    top = pt.load_topology(self.topfile)
    top.strip(":WAT")
    try: top.strip(":Cl-") 
    except: pass
    try: top.strip(":Na+")
    except: pass

    with h5py.File(self.trajfile, "r") as hdf:
      if pdbcode.upper() in hdf.keys():
        coord = hdf[f"/{pdbcode.upper()}/trajectory_coordinates"]
        # Parse frames (Only one from stride and frame_indices will take effect) and masks
        if "stride" in kwarg.keys() and kwarg["stride"] is not None:
          slice_frame = np.s_[::int(kwarg["stride"])]
        elif "frame_indices" in kwarg.keys() and kwarg["frame_indices"] is not None:
          slice_frame = np.s_[kwarg["frame_indices"]]
        else: 
          slice_frame = np.s_[:]
        if "mask" in kwarg.keys() and kwarg["mask"] is not None:
          slice_atom = np.s_[top.select(kwarg["mask"])]
          top = top[slice_atom]
        else: 
          slice_atom = np.s_[:]
        ret_traj = pt.Trajectory(xyz=coord[slice_frame, slice_atom, :], top=top)
      else:
        raise ValueError(f"Not found the key for PDB code {pdbcode.upper()} in the HDF5 trajectory file.")

    # NOTE: Get the PDB code in the standard format, lowercase and replace superceded PDB codes
    self.pdbcode = pdbcode.lower()
    self.traj = ret_traj
    pt.superpose(ret_traj, mask="@CA")
    
    # Initialization the Trajectory object with Pytraj trajectory 
    super().__init__(ret_traj)

  @property
  def identity(self):
    return self.pdbcode


if __name__ == "__main__":
  """
  The script demonstrates how to customize the trajectory loader to load the MISATO dataset

  Usage: 
    python tutorial2_customize_traj.py /path/to/misato_dataset
  
  """
  if len(sys.argv) < 2: 
    print(f"Usage: {sys.argv[0]} /path/to/misato_dataset", file=sys.stderr)
    sys.exit(1)
  if not os.path.exists(sys.argv[1]):
    print(f"Error: The directory {sys.argv[1]} does not exist.", file=sys.stderr)
    print(f"Usage: {sys.argv[0]} /path/to/misato_dataset", file=sys.stderr)
    sys.exit(1)
  if not os.path.exists(f"{sys.argv[1]}/MD.hdf5"):
    print(f"Error: The directory {sys.argv[1]} does not contain the MD.hdf5 file.", file=sys.stderr)
    print(f"Usage: {sys.argv[0]} /path/to/misato_dataset", file=sys.stderr)
    sys.exit(1)

  misato_dir = sys.argv[1]
  pdbs = ['1gpk', '1h23', '1k1i', '1nc3', '1o3f', '1p1q', '1pxn', '1r5y', '1ydr', '2c3i',
          '2p4y', '2qbr', '2vkm', '2wn9', '2wvt', '2zcr', '3ag9', '3b1m', '3cj4', '3coz',
          '3dxg', '3fv2', '3gbb', '3gc5', '3gnw', '3gr2', '3n86', '3nq9', '3pww', '3pxf',
          '3qgy', '3ryj', '3u8n', '3uew', '3uex', '3uo4', '3wz8', '3zsx', '4cr9', '4crc',
          '4ddh', '4de3', '4e5w', '4e6q', '4gkm', '4jia', '4k77', '4mme', '4ogj', '4qac']
  trajlist = [(pdb, misato_dir) for pdb in pdbs]

  FEATURIZER_PARMS = {"dimensions": [32, 32, 32], "lengths": 20, "time_window": 10, "outfile": "/tmp/example.h5"} 
  loader = nearl.io.TrajectoryLoader(trajlist, trajtype=MisatoTraj, superpose=True, trajids = pdbs)
  feat  = nearl.featurizer.Featurizer(FEATURIZER_PARMS)
  feat.register_trajloader(loader)
  feat.register_focus([":MOL"], "mask")

  feature = nearl.features.Mass(selection="!:MOL", outkey="feat_static", cutoff=2.5, sigma=1.0)
  feat.register_feature(feature)
  print(len(feat.FEATURESPACE))
  feat.run(8)


