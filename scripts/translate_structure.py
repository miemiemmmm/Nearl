import time, sys
import numpy as np
import pytraj as pt
from BetaPose import utils

if __name__ == "__main__":
  # Load the PDB structure to for further coordinate rotation
  file = sys.argv[1];
  # For each degree of freedom, how many steps to take for one full rotation
  translation_struct = sys.argv[2];
  translation_length = int(sys.argv[3]);
  translation_steps  = int(sys.argv[4]);
  # By default, there is no translation, modify the script if translation is needed
  translate = [int(translation_length)] * 3;

  print("Summary of the input parameters");
  print(f"{'Input file':15s} : {file}")
  print(f"{'Translation steps':15s} : {translation_steps}")
  print(f"{'Translation length':15s} : {translation_length}")
  print(f"{'Translation':15s} : {translate}")

  # Pre-comute the necessary offsets for partial coordinate translation
  st = time.perf_counter();
  mesh = np.meshgrid(np.linspace(-translate[0], translate[0], translation_steps),
                      np.linspace(-translate[1], translate[1], translation_steps),
                      np.linspace(-translate[2], translate[2], translation_steps)
                      )
  offsets = np.column_stack([mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()])

  traj = pt.load(file);
  coord = traj.xyz[0];
  selected_atoms = traj.top.select(translation_struct);
  new_frames = [];
  for idx, offset in enumerate(offsets):
    new_coord = coord.copy();
    new_coord[selected_atoms] =  new_coord[selected_atoms] + offset;
    new_frame = pt.Frame(traj.top.n_atoms);
    new_frame.top = traj.top;
    new_frame.xyz = new_coord;
    new_frames.append(new_frame);
    if (idx+1) % 500 == 0:
      print(f"Finished {idx+1}/{offsets.shape[0]} frames")

  print("Generating a new pytraj trajectory")
  new_traj = pt.Trajectory(xyz = np.array([f.xyz for f in new_frames]), top = traj.top);

  print("Writing to the trajectory file transformed.nc")
  pt.write_traj("transformed.nc", new_traj, overwrite=True)
  print(f"Done, Total time elapsed: {time.perf_counter() - st} seconds");


