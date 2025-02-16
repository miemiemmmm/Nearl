import time
import numpy as np 
import open3d as o3d

from nearl import commands

view_time = 3

def test_view_ret():
  ret = np.random.normal(size=(32, 32, 32), loc=5, scale=2).astype(np.float32)
  dims = np.array([32, 32, 32], dtype=np.int32)

  points = []
  colors = []
  max_val = np.max(ret)
  for i in range(dims[0]):
    for j in range(dims[1]):
      for k in range(dims[2]):
        if ret[i, j, k] > 0.1:
          points.append([i, j, k])
          colors.append([ret[i, j, k] / max_val, 0, 0])

  print(f"Open3D viewer will automatically exit in {view_time} seconds...") 
  p3d = o3d.geometry.PointCloud()
  p3d.points = o3d.utility.Vector3dVector(np.array(points))
  p3d.colors = o3d.utility.Vector3dVector(np.array(colors))
  # Draw the point cloud for 5 seconds and then close the window
  viewer = o3d.visualization.Visualizer()
  viewer.create_window()
  viewer.add_geometry(p3d)
  st = time.perf_counter()
  while (time.perf_counter()-st) < view_time: 
    viewer.poll_events()
    viewer.update_renderer()
    time.sleep(0.1)
  viewer.destroy_window()






