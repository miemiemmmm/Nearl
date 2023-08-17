import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Extended Gaussian Image


# %matplotlib qt
def compute_egi(mesh, num_bins):
  if isinstance(mesh, o3d.geometry.TriangleMesh): 
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
  else: 
    normals = np.asarray(mesh.normals)
  theta = np.arctan2(normals[:, 1], normals[:, 0])
  phi = np.arccos(normals[:, 2])

  theta_bins = np.linspace(-np.pi, np.pi, num_bins + 1)
  phi_bins = np.linspace(0, np.pi, num_bins // 2 + 1)

  # egi_hist, _, _ = np.histogram2d(theta, phi, bins=[theta_bins, phi_bins])
  egi_hist, _, _ = np.histogram2d(
    theta, phi, bins=(num_bins, num_bins // 2),
    range=[[-np.pi, np.pi], [0, np.pi]])
  return egi_hist

def visualize_egi(mesh, num_bins):
  egi_hist = compute_egi(mesh, num_bins)
  num_bins_theta, num_bins_phi = egi_hist.shape
  theta_bins = np.linspace(-np.pi, np.pi, num_bins_theta+1)
  phi_bins = np.linspace(0, np.pi, num_bins_phi+1)
  theta, phi = np.meshgrid(theta_bins[:-1] + np.diff(theta_bins) / 2,
                           phi_bins[:-1] + np.diff(phi_bins) / 2)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  x = np.sin(phi) * np.cos(theta) 
  y = np.sin(phi) * np.sin(theta) 
  z = np.cos(phi) 
  
   # Create a unit sphere
  sphere_phi, sphere_theta = np.mgrid[0:np.pi:100j, 0:2 * np.pi:100j]
  sphere_x = np.sin(sphere_phi) * np.cos(sphere_theta)
  sphere_y = np.sin(sphere_phi) * np.sin(sphere_theta)
  sphere_z = np.cos(sphere_phi)

  ax.plot_surface(sphere_x, sphere_y, sphere_z, color="c", alpha=0.1, linewidth=10, antialiased=False); 

  arrow_scale = 2/np.max(egi_hist); 
  arrow_cutoff = np.quantile(egi_hist.reshape(-1), 0.8)
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      if egi_hist[j, i] > arrow_cutoff:
        ax.quiver(0, 0, 0, x[i, j], y[i, j], z[i, j], 
                  length = egi_hist[j, i] * arrow_scale, 
                  color='b', alpha=0.6)

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_box_aspect([1, 1, 1])

  plt.show()

# Load a 3D mesh
bunny = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(bunny.path)
# mesh = o3d.geometry.TriangleMesh.create_box()
# mesh = mesh.sample_points_uniformly(1000)

if isinstance(mesh, o3d.geometry.TriangleMesh): 
  mesh.compute_vertex_normals()
else: 
  mesh.estimate_normals() #

# Set the number of bins for EGI calculation
num_bins = 10

# Visualize the EGI of the mesh
visualize_egi(mesh, num_bins)
