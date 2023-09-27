import sys, time

import numpy as np
import pytraj as pt
from BetaPose import utils
import dask
from dask.distributed import Client
from dask.diagnostics import ProgressBar


def parallelize_transform(tasks):
  """
  This function is fast and there is not much demand to multi-process it
  """
  new_frames = []
  for r, p, z in tasks:
    # Compute the transformation matrix and apply it to the coordinates
    trans_matrix = utils.TM_euler(r, p, z, translate);
    coord_transformed = utils.transform_pcd(coord, trans_matrix);
    # Create a new frame and append the new frame to the trajectory
    new_frame = pt.Frame(traj.top.n_atoms);
    new_frame.top = traj.top;
    new_frame.xyz = coord_transformed;
    new_frames.append(new_frame);
  return new_frames


if __name__ == "__main__":
  """
  Usage: python structure_rotation.py <PDB File> <Rotation Steps>
  Systematic rotation of the given PDB file
  """
  # Check the input arguments
  # if len(sys.argv) != 3:
  #   print("Usage: python structure_rotation.py <PDB File> <Rotation Steps>")
  #   print("If translation of the structure is needed while coordinate transformation, please modify the script")
  #   sys.exit(1)
  # Load the PDB structure to for further coordinate rotation
  # file = sys.argv[1];
  # For each degree of freedom, how many steps to take for one full rotation
  # rotation_steps = int(sys.argv[2]);

  file = "/media/yzhang/MieT5/BetaPose/data/complexes/2rkm_complex.pdb"
  rotation_steps = 20    # For each degree of freedom, how many steps to take for one full rotation
  translate = [0, 0, 0]  # By default, there is no translation, modify the script if translation is needed
  to_origin = True       # Put the center of the structure to the origin point

  # Print out the parameters for this run
  print(f"{'Input file':15s} : {file}")
  print(f"{'Rotation steps':15s} : {rotation_steps}")
  print(f"{'Translation':15s} : {translate}")
  print(f"{'Total steps':15s} : {rotation_steps ** 3}")

  st = time.perf_counter()
  mesh = np.meshgrid(np.linspace(-np.pi, np.pi, rotation_steps),
                     np.linspace(-np.pi, np.pi, rotation_steps),
                     np.linspace(-np.pi, np.pi, rotation_steps)
                    )
  euler_angles = np.column_stack([mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()])

  traj = pt.load(file);
  coord = traj.xyz[0];
  if to_origin:
    print(f"Move the system to the origin point, offset is {np.mean(coord, axis=0)}");
    coord -= np.mean(coord, axis=0);

  results = parallelize_transform(euler_angles);
  print(f"Finished computing new frames, start writing to trajectory file, time elapsed: {time.perf_counter() - st}");

  # Put the new frames to a pytraj trajectory
  st_write = time.perf_counter()
  new_traj = pt.Trajectory(xyz=np.array([frame.xyz for frame in results]), top=traj.top);
  print(f"Finished putting frames to pytraj trajectory, time elapsed: {time.perf_counter() - st_write}");

  # Write the new trajectory to a file
  pt.write_traj("transformed.nc", new_traj, overwrite=True)
  print("Finished, Total time elapsed: ", time.perf_counter() - st_write);


