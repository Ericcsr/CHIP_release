"""MuJoCo sim2sim backend with the same interface as real_env.UnitreeRobot.

The physics runs in a separate process at `simulation_dt` and exchanges joint
targets / proprioception with the controller through shared memory.
"""
import multiprocessing as mp
import time
from multiprocessing.shared_memory import SharedMemory

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from utils.robot_utils import Rate

ZERO_TORQUE = -np.ones(29, dtype=np.float32) * 200.0
DAMPING = np.ones(29, dtype=np.float32) * 200.0

_shm_buffer = []  # keep SharedMemory handles alive


class ElasticBand:
    """Virtual gantry that holds the torso up until the policy starts.

    ref: https://github.com/unitreerobotics/unitree_mujoco
    Viewer keys: 7 / 8 lower / raise the band, 9 toggles it.
    """

    def __init__(self):
        self.kp_pos = 10000
        self.kd_pos = 1000
        self.kp_ang = 1000
        self.kd_ang = 10
        self.point = np.array([0, 0, 1])
        self.length = 0
        self.enable = True

    def Advance(self, pose):
        """pose: [pos(3), quat wxyz(4), lin_vel(3), ang_vel(3)] in world frame -> wrench [f(3), tau(3)]."""
        pos, quat, lin_vel, ang_vel = pose[0:3], pose[3:7], pose[7:10], pose[10:13]
        f = self.kp_pos * (self.point - pos + np.array([0, 0, self.length])) - self.kd_pos * lin_vel
        rotvec = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_rotvec()
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel
        return np.concatenate([f, torque])

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable

    def disable_band(self):
        self.enable = False


def shared_np(size, name, dtype=np.float32):
    nbytes = int(np.prod(size) * np.dtype(dtype).itemsize)
    try:
        shm = SharedMemory(create=True, size=nbytes, name=name)
    except FileExistsError:
        shm = SharedMemory(create=False, name=name)
    arr = np.ndarray(size, dtype=dtype, buffer=shm.buf)
    arr[:] = 0.0
    _shm_buffer.append(shm)
    return arr


def pd_control(target_q, data, kp, kd):
    return (target_q - data.qpos[7:36]) * kp - data.qvel[6:35] * kd


def damping_control(data, kd):
    return -data.qvel[6:35] * kd


def run_simulation(control_lock, data_lock, xml_path, config, headless=False):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = config.get("simulation_dt", 0.005)
    mujoco.mj_forward(model, data)

    elastic_band = ElasticBand()
    band_attached_link = model.body("torso_link").id

    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=elastic_band.MujuocoKeyCallback)

    control = shared_np(30, "control", np.float32)
    q = shared_np(29, "q", np.float32)
    dq = shared_np(29, "dq", np.float32)
    omega = shared_np(3, "omega", np.float32)
    imu_quat = shared_np(4, "imu_quat", np.float32)
    base_pos = shared_np(3, "base_pos", np.float32)  # for logging/debugging only, never fed to the policy

    kp = np.array(config["joint_stiffness"], dtype=np.float32)
    kd = np.array(config["joint_damping"], dtype=np.float32)
    torque_limit = np.array(config["torque_limit"], dtype=np.float32)

    rate = Rate(1 / (model.opt.timestep * config.get("slow_down", 1.0)))
    while viewer is None or viewer.is_running():
        with control_lock:
            if control[0] > 180:
                tau = damping_control(data, kd)
            elif control[0] < -180:
                tau = np.zeros(29, dtype=np.float32)
            else:
                tau = pd_control(control[:29], data, kp, kd)
            if control[29] > 100:
                elastic_band.disable_band()
        data.ctrl[:] = np.clip(tau, -torque_limit, torque_limit)

        if elastic_band.enable:
            pose = np.concatenate([data.xpos[band_attached_link], data.xquat[band_attached_link], np.zeros(6)])
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, band_attached_link, pose[7:13], 0)
            pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()  # [ang, lin] -> [lin, ang]
            data.xfrc_applied[band_attached_link] = elastic_band.Advance(pose)
        else:
            data.xfrc_applied[band_attached_link] = np.zeros(6)

        mujoco.mj_step(model, data)

        with data_lock:
            q[:] = data.qpos[7:36]
            dq[:] = data.qvel[6:35]
            imu_quat[:] = data.qpos[3:7][[1, 2, 3, 0]]  # xyzw
            omega[:] = data.qvel[3:6]  # body-frame angular velocity (gyroscope)
            base_pos[:] = data.qpos[:3]
        if viewer is not None:
            viewer.sync()
        rate.sleep()


class MujocoRobot:
    def __init__(self, xml_path, config, headless=False, auto_start=False):
        self.xml_path = xml_path
        self.config = config
        self.auto_start = auto_start
        self.control_dt = config.get("control_dt", 0.02)

        self.control_var = shared_np(30, "control", np.float32)
        self.q_var = shared_np(29, "q", np.float32)
        self.dq_var = shared_np(29, "dq", np.float32)
        self.omega_var = shared_np(3, "omega", np.float32)
        self.imu_quat_var = shared_np(4, "imu_quat", np.float32)
        self.base_pos_var = shared_np(3, "base_pos", np.float32)

        self.control_lock = mp.Lock()
        self.data_lock = mp.Lock()
        self.process = mp.Process(
            target=run_simulation,
            args=(self.control_lock, self.data_lock, xml_path, config, headless),
            daemon=True,
        )
        self.process.start()

        # Wait for the simulation to publish a valid IMU quaternion.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            with self.data_lock:
                if np.linalg.norm(self.imu_quat_var) > 0.5:
                    break
            if not self.process.is_alive():
                raise RuntimeError("MuJoCo simulation process exited during startup")
            time.sleep(0.005)
        else:
            raise RuntimeError("MuJoCo simulation failed to start within 10s")

    def pd_control(self, target_q):
        with self.control_lock:
            self.control_var[:29] = target_q

    def release_robot(self):
        """Remove the elastic band so the robot stands on its own."""
        with self.control_lock:
            self.control_var[29] = 200.0

    def set_robot_state(self, q):
        """Interpolate joint targets to `q` over 2 s."""
        num_step = int(2.0 / self.control_dt)
        with self.data_lock:
            init_q = self.q_var.copy()
        target_q = np.asarray(q, dtype=np.float32)
        for i in range(num_step):
            alpha = i / num_step
            self.pd_control(init_q * (1 - alpha) + target_q * alpha)
            time.sleep(self.control_dt)

    def maintain_state(self, q):
        self.pd_control(q)
        if self.auto_start:
            time.sleep(1.0)
        else:
            input("Press Enter to start the policy...")

    def damping_state(self):
        self.pd_control(DAMPING)

    def zero_torque_state(self):
        self.pd_control(ZERO_TORQUE)

    def get_robot_state(self):
        """Returns q, dq (MuJoCo joint order), IMU quaternion (xyzw), body angular velocity."""
        with self.data_lock:
            return self.q_var.copy(), self.dq_var.copy(), self.imu_quat_var.copy(), self.omega_var.copy()

    def get_base_height(self):
        """Debug only: simulated pelvis height."""
        with self.data_lock:
            return float(self.base_pos_var[2])

    def step_robot(self, action):
        self.pd_control(action)

    def get_start_ticker(self):
        # In simulation the reference motion starts playing right after the policy starts.
        return True

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join()
        for shm in _shm_buffer:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        _shm_buffer.clear()
