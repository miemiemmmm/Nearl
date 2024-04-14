import os, time, random, argparse

import numpy as np
import open3d as o3d
from pytraj import Trajectory

from matplotlib import colormaps
from feater import io, constants
from siesta.scripts import view_obj


element_color_map = {
  # BASIC ELEMENTS
  12: [0.5, 0.5, 0.5],
  1:  [1, 1, 1],
  14: [0, 0, 1],
  15: [1, 0, 0],     # TODO: Need to correct the data in the hdf file. Round to closest interger not the floor
  16: [1, 0, 0],
  32: [1, 1, 0],
  31: [1, 0.6, 0.4],

  # METALS
  23: [0.7, 0.7, 0.1],
  24: [0.7, 0.7, 0.1],
  40: [0.7, 0.7, 0.1],
  39 : [0, 0.5, 1],
  65: [0.8, 0.4, 0.1],
  63: [0.8, 0.4, 0.1],
  56: [0.8, 0.4, 0.1],
  55: [0.6, 0, 0.4],

  # UNKNOWNS
  "UNK": [0.5, 0.5, 0.5],
  "U": [0.5, 0.5, 0.5],
}


def get_coordi(hdf, index:int) -> np.ndarray:
  st = hdf["coord_starts"][index]
  ed = hdf["coord_ends"][index]
  coord = hdf["coordinates"][st:ed]
  return np.asarray(coord, dtype=np.float64)

def get_elemi(hdf, index:int) -> np.ndarray:
  st = hdf["start"][index]
  ed = hdf["end"][index]
  elemi = hdf["elems"][st:ed]
  retcolor = np.zeros((elemi.shape[0], 3), dtype=np.float32)
  for idx in range(len(elemi)):
    mass = elemi[idx]
    print("mass is : ", mass)
    if mass not in element_color_map:
      print(f"Warning: Element {mass} is not in the element color map")
      retcolor[idx] = element_color_map["UNK"]
    else:
      retcolor[idx] = element_color_map[mass]
  return retcolor

def get_voxeli(hdf, index:int) -> np.ndarray:
  
  dims = np.asarray(hdf["shape"])
  index_st  = index * dims
  index_end = index_st + dims
  voxeli = hdf["voxel"][index_st[0]:index_end[0]]
  return voxeli


def get_geo_voxeli(voxel, cmap="inferno", percentile=95, hide=1, scale_factor=[1,1,1]) -> list:
  dims = np.asarray(voxel.shape)
  cmap = colormaps.get_cmap(cmap)
  vmax = np.max(voxel)
  # Hide the zero voxels
  # Empty voxels are not shown in the visualization
  if hide: 
    mask = (voxel - 1e-6) > 0 
    vcutoff = np.percentile(voxel[mask], percentile)
  else:
    vcutoff = np.percentile(voxel, percentile)
  ret = []
  for x in range(dims[0]):
    for y in range(dims[1]):
      for z in range(dims[2]):
        if voxel[x,y,z] < vcutoff:
          continue
        box = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
        # box.translate(np.asarray([x,y,z], dtype=np.float64))
        box.translate(np.asarray([x * scale_factor[0], y * scale_factor[1], z * scale_factor[2]], dtype=np.float64))
        color = cmap(voxel[x,y,z]/vmax)[:3]
        box.paint_uniform_color(color)
        ret.append(box)
  return ret

def get_geo_coordi(coordi) -> list:
  ret = []
  for i in range(coordi.shape[0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    sphere.translate(coordi[i])
    sphere.paint_uniform_color([1,0,0])
    ret.append(sphere)
  return ret


# TODO: Already put this to the SiESTA
def add_bounding_box(dims): 
  boxpoints = np.array([
    [0,0,0],
    [dims[0],0,0],
    [0, dims[1], 0],
    [0,0,dims[2]],
    [dims[0], dims[1], 0],
    [dims[0], 0, dims[2]],
    [0, dims[1], dims[2]],
    [dims[0], dims[1], dims[2]],
  ])
  lines = [
    [0,1], [0,2], [0,3], [1,4],
    [1,5], [2,4], [2,6], [3,5],
    [3,6], [4,7], [5,7], [6,7],
  ]
  ret = []
  for line in lines:
    cylinder = view_obj.create_cylinder(boxpoints[line[0]], boxpoints[line[1]], radius=0.1, color=[0,0,1])
    ret.append(cylinder)
  print(len(ret))
  for geo in ret:
    print(geo)
    geo.compute_vertex_normals()
  return ret


def main_render(inputfile:str, index:int, args):
  print("Reading the voxel")
  st = time.perf_counter()
  with io.hdffile(inputfile, "r") as hdf:
    voxeli = hdf[args.tagname][index]
    # dims = np.asarray(hdf["dimensions"])
    dims = np.array([32, 32, 32])
    # boxsize = np.asarray(hdf["boxsize"])
    boxsize = np.array([16, 16, 16])
    scale_factor = boxsize / dims
  print("Reading the coordinate")
  print(f"The scale factor is {scale_factor}")
  print(f"The sum of the voxel is {np.sum(voxeli):6.3f}: Max {np.max(voxeli):6.3f}, Min {np.min(voxeli):6.3f}")

  # Reading the reference file
  if args.reference:
    with io.hdffile(args.reference, "r") as hdf:  
      coordi = get_coordi(hdf, index)
      top = hdf.get_top(hdf["topology_key"][index])
      traj = Trajectory(top=top, xyz=np.array([coordi]))
      coord_t = coordi 
      coord_cog = np.mean(coord_t, axis=0)
      diff = coord_cog - np.asarray([8, 8, 8])
      coord_t -= diff
      traj.xyz[0] = coord_t
      geoms_coord = view_obj.traj_to_o3d(traj)
  else: 
    geoms_coord = []
  
  print(f"Coodinate reading took {time.perf_counter() - st:6.2f}s")
  # Main rendering functions
  vis = o3d.visualization.Visualizer()
  vis.create_window(window_name="HDF viewer", width=600, height=600)

  geoms_voxel = get_geo_voxeli(voxeli, cmap=args.cmap, percentile=args.percentile, hide=args.hide, scale_factor=scale_factor)
  for geo in geoms_voxel:
    vis.add_geometry(geo)

  for geo in geoms_coord:
    geo.compute_vertex_normals()
    vis.add_geometry(geo)

  if args.boundingbox:
    geoms_box = view_obj.create_bounding_box(dims * scale_factor)
    for geo in geoms_box:
      vis.add_geometry(geo)

  # Mark the center of the voxel
  if args.markcenter:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    sphere.translate(np.asarray(dims * scale_factor)/2)
    sphere.paint_uniform_color([0, 1, 0])
    vis.add_geometry(sphere)

  vis.poll_events()
  vis.update_renderer()
  vis.run()
  vis.destroy_window()
  # Save the objects to a ply object file 

  final_obj = o3d.geometry.TriangleMesh()
  for geo in geoms_voxel:
    geo.compute_vertex_normals()
    final_obj += geo
  for geo in geoms_coord:
    geo.compute_vertex_normals()
    final_obj += geo
  if args.boundingbox:
    for geo in geoms_box:
      geo.compute_vertex_normals()
      final_obj += geo
  o3d.io.write_triangle_mesh("/home/yzhang/Desktop/test.ply", 
    final_obj,
    write_ascii=True,
    write_vertex_normals=True, 
    write_vertex_colors=True 
  )

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--fileinput", type=str, help="The input HDF5 file")
  parser.add_argument("-i", "--index", type=int, default=0, help="The index of the molecule to be viewed")
  parser.add_argument("-p", "--percentile", type=int, default=90, help="The percentile of the voxel to be viewed")
  parser.add_argument("-c", "--cmap", type=str, default="jet", help="The colormap to be used")
  parser.add_argument("-t", "--tagname", type=str, default="voxel", help="The tag name of the voxel")
  parser.add_argument("-hide", "--hide", default=1, type=int, help="Hide the zero voxels. Default: 1.")
  parser.add_argument("-m", "--markcenter", default=1, type=int, help="Mark the center of the voxel (Marked by a green sphere). Default: 1")
  parser.add_argument("-r", "--reference", type=str, default=None, help="The reference file to be compared with")
  parser.add_argument("-b", "--boundingbox", type=int, default=1, help="Add bounding box to the voxel. Default: 1")
  args = parser.parse_args()
  if not os.path.exists(args.fileinput):
    raise ValueError(f"Input file {args.fileinput} does not exist")
  return args


def console_interface():
  args = parse_args()
  print(args)
  main_render(args.fileinput, args.index, args)

if __name__ == "__main__":
  console_interface()

