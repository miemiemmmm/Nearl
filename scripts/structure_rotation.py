import numpy as np
import pytraj as pt
from BetaPose import utils
import tempfile, sys

if __name__ == "__main__":
  """
  Usage: python structure_rotation.py <PDB File> <Rotation Steps>
  Systematic rotation of the given PDB file
  """
  # Check the input arguments
  if len(sys.argv) != 3:
    print("Usage: python structure_rotation.py <PDB File> <Rotation Steps>")
    print("If translation of the structure is needed while coordinate transformation, please modify the script")
    sys.exit(1)
  # Load the PDB structure to for further coordinate rotation
  file = sys.argv[1];
  # For each degree of freedom, how many steps to take for one full rotation
  rotation_steps = int(sys.argv[2]);
  # By default, there is no translation, modify the script if translation is needed
  translate = [0, 0, 0]

  # Print out the parameters for this run
  print(f"{'Input file':15s} : {file}")
  print(f"{'Rotation steps':15s} : {rotation_steps}")
  print(f"{'Translation':15s} : {translate}")
  print(f"{'Total steps':15s} : {rotation_steps ** 3}")

  traj = pt.load(file)
  coord = traj.xyz[0]
  c = 0
  all_steps = rotation_steps ** 3
  print("Start rotating the structure")
  for r in np.linspace(0, np.pi * 2, rotation_steps):
    for p in np.linspace(0, np.pi * 2, rotation_steps):
      for z in np.linspace(0, np.pi * 2, rotation_steps):
        # Compute the transformation matrix and apply it to the coordinates
        trans_matrix = utils.transform_by_euler_angle(r, p, z, translate);
        coord_transformed = utils.transform_pcd(coord, trans_matrix);

        # Create a new frame and append the new frame to the trajectory
        new_frame = pt.Frame(traj.top.n_atoms);
        new_frame.top = traj.top;
        new_frame.xyz = coord_transformed;
        traj.append(new_frame);
        c += 1
        if c % 100 == 0:
          print(f"Processed {c}/{all_steps} frames")
  pt.write_traj("transformed.nc", traj, overwrite=True)
  print("Done")
