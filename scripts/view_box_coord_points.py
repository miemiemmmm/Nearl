import os
import numpy as np
import open3d as o3d
import nearl

h5file = "/home/yzhang/Documents/tests/tempfolder_mlproj/p66596_1b8f1192db_segments.h5"


if os.path.isfile(h5file.replace("segments.h5", "frame0.pdb")):
  mol_objs = nearl.utils.view.molecule_to_o3d(h5file.replace("segments.h5", "frame0.pdb"))
else:
  print("Not found pdbfile", h5file.replace("segments.h5", "frame0.pdb"))

with nearl.io.hdf_operator(h5file, "r") as hdf:
  xyz_coord = hdf.data("xyz")
  box_info = hdf.data("box")
  segment_info = hdf.data("s_final")
  radii = np.ones(len(xyz_coord))*0.25
  allcolors = nearl.utils.color_steps("bwr", max(segment_info)+1)
  xyz_colors = [allcolors[i] for i in segment_info]

print(xyz_coord.shape)
print(box_info.shape)

vis = o3d.visualization.Visualizer()
vis.create_window()

for i in range(len(xyz_coord)):
  coord = xyz_coord[i]
  radius = radii[i]
  sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
  sphere.compute_vertex_normals()
  # set the color
  sphere.paint_uniform_color(xyz_colors[i])
  sphere.translate(coord)
  vis.add_geometry(sphere)

cube = nearl.utils.view.NewCuboid(box_info[:3], box_info[3:6])
vis.add_geometry(cube)
cf = nearl.utils.view.NewCoordFrame(np.min(xyz_coord, axis=0), 8)
vis.add_geometry(cf)


for obj in mol_objs:
  vis.add_geometry(obj)

vis.run()
# vis.destroy_window()


