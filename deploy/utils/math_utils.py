import numpy as np
from scipy.spatial.transform import Rotation

def rotatepoint(q, v):
    # q_v = [v[0], v[1], v[2], 0]
    # return quatmultiply(quatmultiply(q, q_v), quatconj(q))[:-1]
    #
    # https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
    q_r = q[3:4]
    q_xyz = q[:3]
    t = 2 * np.cross(q_xyz, v)
    return v + q_r * t + np.cross(q_xyz, t)


def heading_zup(quat):
    ref_dir = np.zeros_like(quat[:3])
    ref_dir[0] = 1
    ref_dir = rotatepoint(quat, ref_dir)
    return np.arctan2(ref_dir[1], ref_dir[0])

def heading_quat(quat):
    yaw = heading_zup(quat)
    return Rotation.from_euler("z", yaw).as_quat()

def normalize(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a batch of quaternions."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (norm + eps)

def yaw_quat(quat: np.ndarray) -> np.ndarray:
    """
    Extract the yaw component of a quaternion (w, x, y, z).
    
    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)
    
    Returns:
        A quaternion with only yaw component. Same shape as input.
    """
    shape = quat.shape
    quat_yaw = quat.reshape(-1, 4)

    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]

    # yaw angle
    yaw = np.arctan2(2 * (qw * qz + qx * qy),
                     1 - 2 * (qy * qy + qz * qz))

    # construct yaw-only quaternion
    quat_yaw_out = np.zeros_like(quat_yaw)
    quat_yaw_out[:, 0] = np.cos(yaw / 2.0)
    quat_yaw_out[:, 3] = np.sin(yaw / 2.0)

    quat_yaw_out = normalize(quat_yaw_out)
    return quat_yaw_out.reshape(shape)
