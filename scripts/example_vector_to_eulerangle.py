import numpy as np
from BetaPose import utils


def euler_angles_from_vector0(v1, v2):
  # Normalize both vectors
  v1_unit = v1 / np.linalg.norm(v1)
  v2_unit = v2 / np.linalg.norm(v2)

  # Calculate the cross product and dot product of the vectors
  cross = np.cross(v2_unit, v1_unit)
  dot = np.dot(v2_unit, v1_unit)

  # Calculate the angle between the vectors
  angle = np.arccos(np.clip(dot, -1.0, 1.0))

  # Calculate the axis of rotation
  axis = cross / np.linalg.norm(cross)

  # Calculate the quaternion
  qw = np.cos(angle / 2)
  qx, qy, qz = axis * np.sin(angle / 2)

  # Convert quaternion to Euler angles
  roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
  pitch = np.arcsin(2 * (qw * qy - qz * qx))
  yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
  return roll, pitch, yaw

def euler_angles_from_vector(v1, v2):
  # Normalize vectors
  v1_norm = v1 / np.linalg.norm(v1)
  v2_norm = v2 / np.linalg.norm(v2)
  # Compute the cross product and dot product of the two vectors
  cross = np.cross(v2_norm, v1_norm)
  dot = np.dot(v2_norm, v1_norm)
  angle = np.arccos(dot)

  # Compute the rotation matrix around the cross product vector
  if np.linalg.norm(cross) > 0:
    cross_norm = cross / np.linalg.norm(cross)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_product_matrix = np.array([[0, -cross_norm[2], cross_norm[1]],
                                     [cross_norm[2], 0, -cross_norm[0]],
                                     [-cross_norm[1], cross_norm[0], 0]])
    R = cos_angle * np.eye(3) + (1 - cos_angle) * np.outer(cross_norm, cross_norm) + sin_angle * cross_product_matrix
  else:
    R = np.eye(3)

  # Compute the roll, pitch, and yaw angles from the rotation matrix
  yaw = np.arctan2(R[1, 0], R[0, 0])
  pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
  roll = np.arctan2(R[2, 1], R[2, 2])

  return roll, pitch, yaw

if __name__ == "__main__":
  v1 = [0,0,1]
  v2 = [1,5,3]
  eas = euler_angles_from_vector(v1,v2);


  tm = utils.TM_euler(*eas);
  ret = utils.transform_pcd(np.array([v2]), tm);
  print(f"Target vector: {v1} | Vector to be aligned: {v2}");
  print(f"Roll: {eas[0]:.4f}; Pitch: {eas[1]:.4f}; Yaw: {eas[2]:.4f}");
  print(f"Aligned vector (after applying the euler angles): {ret}");
