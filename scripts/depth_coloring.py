import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import time

def get_depth(point_cloud, camera_position):
  # Compute distances to camera
  distances = np.linalg.norm(point_cloud - camera_position, axis=1)

  # Normalize distances
  min_dist, max_dist = np.min(distances), np.max(distances)
  normalized_distances = (distances - min_dist) / (max_dist - min_dist)

  # Map to colors
  colormap = plt.cm.get_cmap("jet")  # or any other colormap
  colors = colormap(normalized_distances)
  return colors

def show_maplotlib(point_cloud, colors):
  if len(point_cloud) != len(colors):
    raise ValueError("Point cloud and colors must have the same length")

  # Visualization using Matplotlib
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=colors)
  plt.show()
def show_open3d(point_cloud, camera_position, colors, model="pointcloud"):
  if len(point_cloud) != len(colors):
    raise ValueError("Point cloud and colors must have the same length")

  # Visualization using Open3D

  if model == "pointcloud":
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

  elif model == "ball":
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in range(len(point_cloud)):
      mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
      mesh_sphere.compute_vertex_normals()
      mesh_sphere.paint_uniform_color(colors[i][:3])
      mesh_sphere.translate(point_cloud[i])
      vis.add_geometry(mesh_sphere)
    vis.run()
  elif model == "box":
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for i in range(len(point_cloud)):
      mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
      mesh_box.compute_vertex_normals()
      mesh_box.paint_uniform_color(colors[i][:3])
      mesh_box.translate(point_cloud[i])
      # geometries.append(mesh_box)
      vis.add_geometry(mesh_box)


    view_control = vis.get_view_control()
    camera = view_control.convert_to_pinhole_camera_parameters()
    print(camera)
    print(dir(camera.extrinsic))
    print(camera.intrinsic.intrinsic_matrix)
    print(camera.intrinsic.get_focal_length())
    print(camera.intrinsic.get_principal_point())
    print(camera.intrinsic.get_skew())
    print(camera.intrinsic.get_matrix())

    # vis.reset_view_point([100,100,100])
    print(dir(vis))
    print(dir(view_control))

    # Camera parameters: (zoom, front, lookat, up)
    zoom = 10.8
    front = [1000.50, -0.50, -0.50]
    lookat = [-110, 1000, 0]
    up = [0, 1, 0]

    # camera.

    # Set parameters
    # view_control.set_zoom(zoom)
    # view_control.set_front(front)
    # view_control.set_lookat([0,0,0], lookat, up)
    # view_control.set_uqp(up)


    vis.poll_events()
    # vis.run()
    time.sleep(1)
    vis.destroy_window()

# Generate random point cloud data for demonstration
N = 1000
point_cloud = np.random.rand(N, 3) * 10  # Nx3 array
# Define camera position
camera_position = np.array([10.0, 0.0, 10.0])
depths = get_depth(point_cloud, camera_position)
show_maplotlib(point_cloud, depths)
show_open3d(point_cloud, camera_position, depths, model="box")



