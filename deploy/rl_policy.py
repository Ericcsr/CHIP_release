import sys
from threading import Lock

import numpy as np
import onnxruntime
from scipy.spatial.transform import Rotation

from utils.params import MUJOCO_TO_ISAAC, ISAAC_TO_MUJOCO
from utils.math_utils import yaw_quat
from utils.storage_utils import ObsQueue, HistoryBuffer

# Allow loading .npz files pickled with numpy>=2 when running numpy<2.
if int(np.__version__.split(".")[0]) < 2:
    import numpy.core.multiarray as multiarray

    sys.modules.setdefault("numpy._core.multiarray", multiarray)


class KeyboardController:
    """Sets the hand compliance command from the keyboard.

    g: stiff (0, 0, 0)   h: both hands compliant (0.5, 0.5, 0)   j: right hand compliant (0, 0.5, 0)

    Falls back to a fixed compliance value if no keyboard backend is available
    (e.g. running headless without an X display).
    """

    PRESETS = {
        "g": np.array([0.0, 0.0, 0.0]),
        "h": np.array([0.5, 0.5, 0.0]),
        "j": np.array([0.0, 0.5, 0.0]),
    }

    def __init__(self, init_compliance=(0.5, 0.5, 0.0)):
        self.lock = Lock()
        self.compliance = np.array(init_compliance, dtype=np.float64)
        try:
            from pynput import keyboard

            self.listener = keyboard.Listener(on_release=self.on_release)
            self.listener.start()
        except Exception as e:  # no display / no input backend
            print(f"[KeyboardController] keyboard disabled ({e}); compliance fixed at {self.compliance}")
            self.listener = None

    def on_release(self, key):
        char = getattr(key, "char", None)
        if char in self.PRESETS:
            with self.lock:
                self.compliance = self.PRESETS[char].copy()
            print(f"[KeyboardController] compliance set to {self.compliance}")

    def get_compliance(self):
        with self.lock:
            return self.compliance.copy()


class RLBasePolicy:
    def __init__(self, onnx_model_path, obs_names, use_sim=False):
        self.onnx_model_path = onnx_model_path
        self.obs_names = obs_names
        if use_sim:
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = tuple(self.session.get_inputs()[0].shape)
        self.meta_data = self.session.get_modelmeta().custom_metadata_map
        self.ticker = 0

    def get_q_init(self):
        return np.zeros(29)

    def prepare_control_signals(self, robot_state):
        raise NotImplementedError

    def prepare_obs(self, robot_state, control_signals):
        raise NotImplementedError

    def get_action(self, obs, start_ticker=False):
        raise NotImplementedError


class RLCHIPPolicy(RLBasePolicy):
    """CHIP policy: lower-body joint command + local 3-point (wrists, head) targets + hand compliance."""

    # Fallbacks used only if the ONNX file carries no metadata (Isaac joint order).
    DEFAULT_Q_POSE = [-0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.669, 0.669, 0.2, 0.2, -0.363, -0.363, 0.2,
                      -0.2, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0]
    ACTION_SCALE = [0.548, 0.548, 0.548, 0.351, 0.351, 0.439, 0.548, 0.548, 0.439,
                    0.351, 0.351, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439,
                    0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075,
                    0.075, 0.075]

    # Body indices (Isaac body order) of left wrist, right wrist and torso in the motion file.
    VR_3POINT_INDICES = [28, 29, 9]
    # Offsets of the 3 tracked points in the corresponding body frames.
    VR_3POINT_OFFSETS = np.array([[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]])

    def __init__(
        self,
        onnx_model_path,
        obs_names,
        ref_motion_path,
        use_sim=False,
        lookahead_steps=1,
        lookahead_frame_skips=1,
        hist_names=(),
        hist_length=1,
        ref_motion_start_index=0,
        init_compliance=(0.5, 0.5, 0.0),
    ):
        super().__init__(onnx_model_path, obs_names, use_sim=use_sim)

        def _meta_array(key, fallback):
            if key in self.meta_data:
                return np.array([float(x) for x in self.meta_data[key].split(",")])
            return np.array(fallback, dtype=np.float64)

        default_joint_pos = _meta_array("default_joint_pos", self.DEFAULT_Q_POSE)  # Isaac order
        action_scale = _meta_array("action_scale", self.ACTION_SCALE)  # Isaac order
        self.default_value = {"q": default_joint_pos, "dq": np.zeros(29)}
        if len(action_scale) != 29:
            self.action_scale = np.ones(29) * action_scale
        else:
            self.action_scale = action_scale[ISAAC_TO_MUJOCO]  # MuJoCo order

        self.lower_joint_indices = ISAAC_TO_MUJOCO[:12]

        self.ref_motion = np.load(ref_motion_path)
        self.motion_length = self.ref_motion["joint_pos"].shape[0]
        self.ref_motion_start_index = int(np.clip(ref_motion_start_index, 0, self.motion_length - 1))
        self.ticker = self.ref_motion_start_index

        body_pos_w = self.ref_motion["body_pos_w"]
        body_quat_w = self.ref_motion["body_quat_w"]  # wxyz
        self.ref_q_pos = self.ref_motion["joint_pos"].copy()
        self.ref_q_vel = self.ref_motion["joint_vel"].copy()
        self.ref_anchor_poses = body_pos_w[:, 0].copy()
        self.ref_anchor_orns = body_quat_w[:, 0][:, [1, 2, 3, 0]].copy()  # xyzw
        idx = self.VR_3POINT_INDICES
        self.ref_vr_3point_poses = body_pos_w[:, idx].copy()  # [T, 3, 3]
        self.ref_vr_3point_orns = body_quat_w[:, idx][:, :, [1, 2, 3, 0]].copy()  # [T, 3, 4] xyzw
        for i in range(3):
            self.ref_vr_3point_poses[:, i] += Rotation.from_quat(self.ref_vr_3point_orns[:, i]).apply(
                self.VR_3POINT_OFFSETS[None, i, :]
            )

        si = self.ref_motion_start_index
        self.init_root_heading_inv = Rotation.from_quat(yaw_quat(body_quat_w[si, 0])[[1, 2, 3, 0]]).inv()

        self.keyboard_controller = KeyboardController(init_compliance)
        print("Motion length:", self.motion_length)

        if lookahead_steps != 1:
            self.obs_queue = ObsQueue(max_size=lookahead_steps, stride=lookahead_frame_skips)
            self.delay_queue = ObsQueue(max_size=lookahead_steps, stride=lookahead_frame_skips)
        if hist_length > 1:
            self.history_buffer = HistoryBuffer(hist_length, obs_names=hist_names, flatten=True)

    def get_q_init(self):
        return self.ref_q_pos[self.ref_motion_start_index, ISAAC_TO_MUJOCO]

    def _control_signals_from_motion(self):
        mid = min(self.ticker, self.motion_length - 1)
        return (
            self.ref_q_pos[mid],
            self.ref_q_vel[mid],
            self.ref_anchor_poses[mid],
            self.ref_anchor_orns[mid],
            self.ref_vr_3point_poses[mid],
            self.ref_vr_3point_orns[mid],
        )

    def prepare_control_signals(self, robot_state):
        (ref_joint_pos, ref_joint_vel, ref_anchor_pos, ref_anchor_orn,
         ref_vr_3point_poses, ref_vr_3point_orns) = self._control_signals_from_motion()
        # Reference anchor orientation relative to the initial reference heading.
        rel_anchor_orn = (self.init_root_heading_inv * Rotation.from_quat(ref_anchor_orn)).as_quat()

        if hasattr(self, "obs_queue"):
            self.obs_queue.push(np.stack([ref_joint_pos[self.lower_joint_indices],
                                          ref_joint_vel[self.lower_joint_indices]]))
            lower_cmd = self.obs_queue.get_traj()
            self.delay_queue.push(np.hstack([ref_anchor_pos, ref_anchor_orn]))
            ref_anchor_pos, ref_anchor_orn = self.delay_queue[1][:3], self.delay_queue[1][3:7]
        else:
            lower_cmd = [ref_joint_pos[self.lower_joint_indices], ref_joint_vel[self.lower_joint_indices]]

        # 3-point targets expressed in the reference anchor (pelvis) frame.
        ref_anchor_rot_inv = Rotation.from_quat(ref_anchor_orn).inv()
        vr_3point_pos_l = ref_anchor_rot_inv.apply(ref_vr_3point_poses - ref_anchor_pos[None, :])
        vr_3point_orn_l = ref_anchor_rot_inv * Rotation.from_quat(ref_vr_3point_orns)

        # Robot orientation comes from the IMU only (heading-aligned at start); no root position is used.
        anchor_rot_inv = Rotation.from_quat(robot_state.root_orn).inv()
        return {
            "lower_command": np.hstack(lower_cmd).flatten(),
            "vr_3point_pos": vr_3point_pos_l.flatten(),
            "vr_3point_ori": vr_3point_orn_l.as_quat(scalar_first=True).flatten(),
            "motion_anchor_ori_b": (anchor_rot_inv * Rotation.from_quat(rel_anchor_orn)).as_matrix()[:, :2].flatten(),
            "projected_gravity": anchor_rot_inv.apply(np.array([0, 0, -1])),
            "compliance": self.keyboard_controller.get_compliance(),
        }

    def prepare_obs(self, robot_state, control_signals):
        obs = []
        robot_state_keys = robot_state.__dict__.keys()
        for key in self.obs_names:
            if key in robot_state_keys:
                if key in ["q", "dq"]:
                    item = robot_state.__dict__[key][MUJOCO_TO_ISAAC] - self.default_value[key]
                else:
                    item = robot_state.__dict__[key]
            else:
                item = control_signals[key]
            if hasattr(self, "history_buffer") and key in self.history_buffer.buffer_dict:
                self.history_buffer.add(key, item)
                item = self.history_buffer.get_history(key)
            obs.append(item)
        return np.concatenate(obs).reshape(1, -1)

    def get_action(self, obs, start_ticker=False):
        assert obs.shape == self.input_shape, f"Obs shape: {obs.shape}, input shape: {self.input_shape}"
        action = self.session.run(None, {self.input_name: obs.astype(np.float32)})[0].flatten()
        if start_ticker:
            self.ticker += 1
        elif self.ticker > self.ref_motion_start_index:
            self.ticker = self.ref_motion_start_index
        return action
