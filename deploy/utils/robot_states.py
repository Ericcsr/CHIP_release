import numpy as np


class G1RobotState:
    def __init__(self):
        # Proprioception (MuJoCo joint order)
        self.q = np.zeros(29)
        self.dq = np.zeros(29)
        self.omega = np.zeros(3)  # body-frame angular velocity
        self.imu_quat = np.array([0, 0, 0, 1])  # xyzw

        # Root orientation = IMU orientation with the initial heading removed (xyzw)
        self.root_orn = np.array([0, 0, 0, 1])

        # Last policy action (Isaac joint order)
        self.last_action = np.zeros(29)
