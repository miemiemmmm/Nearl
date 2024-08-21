import os, time, argparse

import h5py

import numpy as np
import open3d as o3d
from pytraj import Trajectory

from matplotlib import colormaps

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
        trans = np.asarray([x * scale_factor[0], y * scale_factor[1], z * scale_factor[2]], dtype=np.float64)
        box.translate(trans)
        color = cmap(voxel[x,y,z]/vmax)[:3]
        box.paint_uniform_color(color)
        ret.append(box)
  return ret

def get_geo_slices(voxel, slices:dict, cmap="inferno", scale_factor=[1,1,1]) -> list:
  dims = np.asarray(voxel.shape)
  vmax = np.max(voxel)
  ret = []
  for key in slices.keys():
    if key not in ["x", "y", "z"]:
      raise ValueError(f"Invalid key {key} for the slice")
    cmap = colormaps.get_cmap(cmap)
    if key == "x": 
      theslice = voxel[slices[key],:,:]
    elif key == "y":
      theslice = voxel[:,slices[key],:]
    else:
      theslice = voxel[:,:,slices[key]]
    if key == "x": 
      for y in range(dims[1]):
        for z in range(dims[2]):
          box = o3d.geometry.TriangleMesh.create_box(width=0.001, height=0.45, depth=0.45)
          box.translate(np.asarray([slices[key] * scale_factor[0], y * scale_factor[1], z * scale_factor[2]], dtype=np.float64))
          color = cmap(theslice[y,z]/vmax)[:3]
          # box.compute_vertex_normals()
          box.paint_uniform_color(color)
          ret.append(box)
    elif key == "y":
      for x in range(dims[0]):
        for z in range(dims[2]):
          box = o3d.geometry.TriangleMesh.create_box(width=0.45, height=0.001, depth=0.45)
          box.translate(np.asarray([x * scale_factor[0], slices[key] * scale_factor[1], z * scale_factor[2]], dtype=np.float64))
          color = cmap(theslice[x,z]/vmax)[:3]
          # box.compute_vertex_normals()
          box.paint_uniform_color(color)
          ret.append(box)
    else:
      for x in range(dims[0]):
        for y in range(dims[1]):
          box = o3d.geometry.TriangleMesh.create_box(width=0.45, height=0.45, depth=0.001)
          box.translate(np.asarray([x * scale_factor[0], y * scale_factor[1], slices[key] * scale_factor[2]], dtype=np.float64))
          color = cmap(theslice[x,y]/vmax)[:3]
          # box.compute_vertex_normals()
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


def render_slices(inputfile:str, index:int, args):
  print("Reading the voxel and meta-data from the hdf file")
  st = time.perf_counter()
  with h5py.File(inputfile, "r") as hdf:
    voxeli = hdf[args.tagname][index]
    
    # Get dimensions metadata from the hdf file
    dim = hdf["featurizer_parms"]["dimensions"][()]
    dims = np.array([dim, dim, dim])
    
    # Get lengths metadata from the hdf file
    length = hdf["featurizer_parms"]["lengths"][()]
    if hasattr(length, "__len__"):
      boxsize = np.array(length)
    else:
      boxsize = np.array([length, length, length])
    print(boxsize.shape)
    scale_factor = boxsize / dims
    print(f"Dim: {dim}; Length: {length}; Scale factor: {scale_factor}")
    print(f"Summary of the voxel: \nSum: {np.sum(voxeli):6.3f}; Max {np.max(voxeli):6.3f}; Min {np.min(voxeli):6.3f}")
  print(f"Reading the voxel took {time.perf_counter() - st:8.6f}s")

  
  # Reading the reference file
  # if args.reference:
  #   with h5py.File(args.reference, "r") as hdf:  
  #     coordi = get_coordi(hdf, index)
  #     top = hdf.get_top(hdf["topology_key"][index])
  #     traj = Trajectory(top=top, xyz=np.array([coordi]))
  #     coord_t = coordi 
  #     coord_cog = np.mean(coord_t, axis=0)
  #     diff = coord_cog - boxsize / 2
  #     coord_t -= diff
  #     traj.xyz[0] = coord_t
  #     geoms_coord = view_obj.traj_to_o3d(traj)
  # else: 
  #   geoms_coord = []
  
  
  # Main rendering functions
  st = time.perf_counter()
  vis = o3d.visualization.Visualizer()
  vis.create_window(window_name="HDF viewer", width=1000, height=1000)

  slice_dict = {"x": args.x, "y": args.y, "z": args.z}
  for i in ["x", "y", "z"]:
    if slice_dict[i] < 0:
      slice_dict.pop(i)
  print(f"Slices to be viewed: {slice_dict}")
  geoms_voxel = get_geo_slices(voxeli, 
                               slices=slice_dict, 
                               cmap=args.cmap, scale_factor=scale_factor)
  for geo in geoms_voxel:
    vis.add_geometry(geo)

  # for geo in geoms_coord:
  #   geo.compute_vertex_normals()
  #   vis.add_geometry(geo)

  if args.boundingbox:
    geoms_box = view_obj.create_bounding_box(dims * scale_factor)
    for geo in geoms_box:
      vis.add_geometry(geo)

  print(f"Object generation took: {time.perf_counter() - st:8.6f}s")

  vis.poll_events()
  vis.update_renderer()
  vis.run()
  vis.destroy_window()
  # Save the objects to a ply object file 

  if args.saveply:
    print(f"Saving the objects to {args.saveply}")
    final_obj = o3d.geometry.TriangleMesh()
    for geo in geoms_voxel:
      geo.compute_vertex_normals()
      final_obj += geo
    # for geo in geoms_coord:
    #   geo.compute_vertex_normals()
    #   final_obj += geo
    if args.boundingbox:
      for geo in geoms_box:
        geo.compute_vertex_normals()
        final_obj += geo
    o3d.io.write_triangle_mesh(args.saveply, 
      final_obj,
      write_ascii=True,
      write_vertex_normals=True, 
      write_vertex_colors=True 
    )


def parse_view_slice():
  parser = argparse.ArgumentParser("View the slice of the voxel for each axes")
  parser.add_argument("-f", "--fileinput", type=str, help="The input HDF5 file")
  parser.add_argument("-i", "--index", type=int, default=0, help="The index of the molecule to be viewed, default: 0")
  parser.add_argument("-t", "--tagname", type=str, default="voxel", help="The tag name of the voxel, default: voxel")
  parser.add_argument("-c", "--cmap", type=str, default="inferno", help="The colormap to be used, default: inferno")
  # parser.add_argument("-r", "--reference", type=str, default=None, help="The reference file to be compared with")
  parser.add_argument("-b", "--boundingbox", type=int, default=1, help="Whether or not to add a bounding box of the voxel. Default: 1")
  parser.add_argument("-s", "--saveply", type=str, default=None, help="The output file to save the ply object")
  parser.add_argument("-x", "--x", type=int, default=-1, help="The x slice to be viewed, default: -1 (hide)")
  parser.add_argument("-y", "--y", type=int, default=15, help="The y slice to be viewed, default: 15")
  parser.add_argument("-z", "--z", type=int, default=15, help="The z slice to be viewed, default: 15")
  args = parser.parse_args()
  if not os.path.exists(args.fileinput):
    raise ValueError(f"Input file {args.fileinput} does not exist")
  return args


def parse_view_voxel():
  parser = argparse.ArgumentParser("View 3D voxels ")
  parser.add_argument("-f", "--fileinput", type=str, help="The input HDF5 file")
  parser.add_argument("-i", "--index", type=int, default=0, help="The index of the molecule to be viewed, default: 0")
  parser.add_argument("-t", "--tagname", type=str, default="voxel", help="The tag name of the voxel, default: voxel")
  parser.add_argument("-c", "--cmap", type=str, default="inferno", help="The colormap to be used, default: inferno")
  parser.add_argument("-p", "--percentile", type=int, default=95, help="The percentile to be used for the cutoff, default: 95")
  parser.add_argument("--hide", type=int, default=1, help="Whether or not to hide the zero voxels, default: 1")
  parser.add_argument("-b", "--boundingbox", type=int, default=1, help="Whether or not to add a bounding box of the voxel, default: 1")
  parser.add_argument("-so", "--saveobj", type=str, default=None, help="The output file to save the obj object, default: None")                           # TODO: Implement this later 
  parser.add_argument("-si", "--saveimg", type=str, default=None, help="The output file to save the image upon closing the 3D viewer, default: None")     # TODO: Implement this later 

  args = parser.parse_args()
  if not os.path.exists(args.fileinput):
    raise ValueError(f"Input file {args.fileinput} does not exist")
  return args
  

def render_voxels(inputfile:str, index:int, args): 
  print("Reading the voxel and meta-data from the hdf file")
  st = time.perf_counter()
  with h5py.File(inputfile, "r") as hdf:
    voxeli = hdf[args.tagname][index]

    if np.count_nonzero(voxeli) == 0:
      raise ValueError("The voxel is full of zeros, not able to visualize")
    
    # Get dimensions metadata from the hdf file
    dim = hdf["featurizer_parms"]["dimensions"][()]
    if hasattr(dim, "__len__"):
      dims = np.array(dim)
    else:
      dims = np.array([dim, dim, dim])
    
    # Get lengths metadata from the hdf file
    length = hdf["featurizer_parms"]["lengths"][()]
    if hasattr(length, "__len__"):
      boxsize = np.array(length)
    else:
      boxsize = np.array([length, length, length])
    scale_factor = boxsize / dims
  
  print(f"Dim: {dim}; Length: {length}; Scale factor: {scale_factor}")
  print(f"Summary of the voxel: \nSum: {np.sum(voxeli):6.3f}; Max {np.max(voxeli):6.3f}; Min {np.min(voxeli):6.3f}")
  print(f"Reading the voxel took {time.perf_counter() - st:8.6f}s")

  # Initialize the visualization
  st = time.perf_counter()
  vis = o3d.visualization.Visualizer()
  vis.create_window(window_name="HDF viewer", width=1000, height=1000)

  # Add the voxel to the visualization
  geoms_voxel = get_geo_voxeli(voxeli, cmap=args.cmap, percentile=args.percentile, hide=args.hide, scale_factor=scale_factor)
  for geo in geoms_voxel:
    vis.add_geometry(geo)

  # Add the bounding box to the visualization
  if args.boundingbox:
    geoms_box = view_obj.create_bounding_box(dims * scale_factor)
    for geo in geoms_box:
      vis.add_geometry(geo)

  vis.poll_events()
  vis.update_renderer()
  vis.run()
  vis.destroy_window()

def CLI_view_slice(): 
  args = parse_view_slice()
  print("Visualization settings: ", vars(args))
  render_slices(args.fileinput, args.index, args)

def CLI_view_voxel(): 
  args = parse_view_voxel()
  print("Visualization settings: ", vars(args))
  render_voxels(args.fileinput, args.index, args)

def CLI_open_obj():
  """
  Open the obj file in the viewer
  """
  pass

# def CLI_view_voxel(): 
