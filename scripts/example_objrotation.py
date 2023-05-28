import numpy as np 
import open3d as o3d
import time
from BetaPose import utils

from matplotlib.cm import inferno
  

pcd = np.arange(90*3, 102*3).reshape(-1,3)
# pcd = np.asarray([np.zeros(102),np.zeros(102), np.arange(102)]).T
# pcd_out = np.array(pcd)

rot_steps = 20

t1 = time.perf_counter()
c = 0
for r in np.linspace(0, np.pi*2, rot_steps):
  for p in np.linspace(0, np.pi*2, rot_steps):
    for z in np.linspace(0, np.pi*2, rot_steps):
      TransMatrix = utils.transform_by_euler_angle(r, p , z)
      pcd2 = utils.transform_pcd(pcd, TransMatrix)
      if c == 0:
        pcd_out = pcd2
      else:
        pcd_out = np.vstack((pcd_out, pcd2))
      c += 1

print("Total rotation time: ", time.perf_counter() - t1)

# Create the resulting point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pcd_out)
thecmap = inferno(np.linspace(0,1,len(pcd_out)))[:, :3]
point_cloud.colors = o3d.utility.Vector3dVector(thecmap)

# Visualize the point cloud and save it
o3d.visualization.draw_geometries([point_cloud])
o3d.io.write_point_cloud("example_objrotation.ply", point_cloud)


