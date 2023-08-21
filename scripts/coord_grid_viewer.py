import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from BetaPose import data_io
import numpy as np 

# %matplotlib qt

def density(coords, weights, showall=False):
  """
  Create a 3D scatter plot for an N x 3 array of 3D points.

  Args:
  points (np.array): An array of shape (num_points, 3) containing the 3D coordinates of the points.

  Returns:
  None
  """
  # Extract the x, y, and z coordinates of the points
  x_coords = coords[:, 0];
  y_coords = coords[:, 1];
  z_coords = coords[:, 2];
  weights = weights.ravel(); 
  if showall: 
    deviation = np.percentile(weights[weights-1e-6 > 0], 15)
    weight_final = np.asarray(weights)+deviation
  else: 
    weight_final = np.asarray(weights).round(3)

  # Create a 3D scatter plot
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x_coords, y_coords, z_coords, 
             cmap='Reds', 
             c=weights*1000, 
             s=weight_final, 
             alpha=0.3, 
             edgecolor='k')

  # Set axis labels
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  
  # Switch off display of axis and display the plot
  ax.axis('off')
  return fig, ax

def add_coord(thefig, coords):
  x_coords = coords[:, 0];
  y_coords = coords[:, 1];
  z_coords = coords[:, 2];
  fig = thefig[0]; 
  ax = thefig[1]; 
  ax.scatter(x_coords, y_coords, z_coords, s=50)
  return fig, ax


def boxinfo_to_coord(theboxinfo): 
  dim1 = np.linspace(theboxinfo[0] - theboxinfo[3]/2, theboxinfo[0] + theboxinfo[3]/2, int(theboxinfo[6]))
  dim2 = np.linspace(theboxinfo[1] - theboxinfo[4]/2, theboxinfo[1] + theboxinfo[4]/2, int(theboxinfo[7]))
  dim3 = np.linspace(theboxinfo[2] - theboxinfo[5]/2, theboxinfo[2] + theboxinfo[5]/2, int(theboxinfo[8]))
  mesh = np.meshgrid(dim1, dim2, dim3, indexing='ij'); 
  return np.column_stack((mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()));

idx = 17
with data_io.hdf_operator("/media/yzhang/MieT5/BetaPose/data/trainingdata/test_3d_data.h5", read_only=True) as f: 
#   f.draw_structure()
  boxinfo = f.data("box")[idx]
  coord = boxinfo_to_coord(boxinfo)
  
  thegrid = f.data("mass_lig")[idx]
  thexyz = f.data(f"/xyz_lig/xyz_lig_{idx}")
  thefig = density(coord, thegrid, showall=True); 
  thefig = add_coord(thefig, thexyz)
  
plt.show()
  

