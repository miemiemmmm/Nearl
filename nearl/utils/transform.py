import numpy as np

def transform_pcd(pcd, trans_mtx):
  # Homogenize the point cloud (add a row of ones)
  homogeneous_pcd = np.hstack((pcd, np.ones((pcd.shape[0], 1))))
  # Apply the transformation matrix to the point cloud
  transformed_pcd = np.dot(homogeneous_pcd, trans_mtx.T)
  # Remove the homogeneous coordinate (last column)
  transformed_pcd = transformed_pcd[:, :3]
  return transformed_pcd


def tm_euler(roll, pitch, yaw, translate=[0, 0, 0]):
  """
  Generate a transformation matrix from Euler angles
  NOTE: Could also use this function to do xyz rotation
  >>> from scipy.spatial.transform import Rotation
  >>> R = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
  """
  # Precompute trigonometric functions
  cos_roll, sin_roll = np.cos(roll), np.sin(roll)
  cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
  cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
  # Generate rotation matrices
  Rx = np.array([[1, 0, 0], [0, cos_roll, -sin_roll], [0, sin_roll, cos_roll]])
  Ry = np.array([[cos_pitch, 0, sin_pitch], [0, 1, 0], [-sin_pitch, 0, cos_pitch]])
  Rz = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
  # Combine rotations
  R = Rz @ Ry @ Rx
  # Create the final transformation matrix
  H = np.eye(4)
  H[:3, :3] = R
  H[:3, 3] = np.array(translate).ravel()
  return H


def tm_quaternion(q, translate=[0, 0, 0]):
  from scipy.spatial.transform import Rotation
  # Generate rotation matrix
  R = Rotation.from_quat(q).as_matrix()
  # Create the final transformation matrix
  H = np.eye(4)
  H[:3, :3] = R
  H[:3, 3] = np.array(translate).ravel()
  return H