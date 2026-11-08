"""
MuJoCo visualizer that shows robot motion alongside reference motion.
Runs in a separate process (run_controller.py --use_sim --show_ref) and shows the
reference motion in its own viewer, synchronized with the policy's motion ticker.
"""
import multiprocessing as mp
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation
from utils.math_utils import yaw_quat
from utils.params import ISAAC_TO_MUJOCO


def _run_ref_visualizer(
    xml_path: str,
    ref_motion_path: str,
    control_dt: float,
    ticker_value,
    ref_motion_start_index: int = 0,
):
    """Run the visualizer showing both robot and reference motion."""
    # Load reference motion
    ref_motion = np.load(ref_motion_path)
    motion_length = ref_motion["joint_pos"].shape[0]
    si = int(ref_motion_start_index)
    if si < 0:
        si = 0
    if si >= motion_length:
        si = motion_length - 1
    init_root_pos = ref_motion["body_pos_w"][si, 0].copy()
    init_root_pos[2] = 0
    init_root_heading_inv = Rotation.from_quat(
        yaw_quat(ref_motion["body_quat_w"][si, 0])[[1, 2, 3, 0]]
    ).inv()
    ref_q_pos = ref_motion["joint_pos"].copy()
    ref_anchor_poses = ref_motion["body_pos_w"][:, 0].copy()
    ref_anchor_orns = ref_motion["body_quat_w"][:, 0][:, [1, 2, 3, 0]].copy()

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # MuJoCo quat: wxyz. Scipy/our format: xyzw
    def to_mujoco_quat(xyzw):
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    viewer = mujoco.viewer.launch_passive(model, data)
    # Run at 60 Hz for smooth display
    display_rate_hz = 60.0
    last_update = time.time()

    # Sync with policy: only advance when policy ticker advances (no auto-play).
    # Use same mid indexing as rl_policy._control_signals_from_motion()
    while viewer.is_running():
        now = time.time()
        if now - last_update < 1.0 / display_rate_hz:
            time.sleep(0.001)
            continue
        last_update = now

        # Get ticker from controller - only advances when policy executes get_action
        ticker = int(ticker_value.value)
        # Same mid logic as policy: mid = ticker if ticker < motion_length else motion_length - 1
        mid = ticker if ticker < motion_length else motion_length - 1
        ref_joint_pos = ref_q_pos[mid][ISAAC_TO_MUJOCO]
        ref_anchor_pos = ref_anchor_poses[mid]
        ref_anchor_orn = ref_anchor_orns[mid]

        # Transform to common frame (relative to ref init)
        ref_rel_pos = init_root_heading_inv.apply(ref_anchor_pos - init_root_pos)
        ref_rel_orn = (
            init_root_heading_inv * Rotation.from_quat(ref_anchor_orn)
        ).as_quat()

        # Show the reference motion
        data.qpos[:3] = ref_rel_pos
        data.qpos[3:7] = to_mujoco_quat(ref_rel_orn)
        data.qpos[7:36] = ref_joint_pos

        mujoco.mj_forward(model, data)
        viewer.sync()

    viewer.close()


def start_ref_visualizer_process(
    xml_path: str,
    ref_motion_path: str,
    control_dt: float = 0.02,
    ticker_value=None,
    ref_motion_start_index: int = 0,
) -> mp.Process:
    """Start the reference motion visualizer in a separate process.
    ticker_value: multiprocessing.Value('f') shared with run_controller for sync.
    """
    process = mp.Process(
        target=_run_ref_visualizer,
        args=(xml_path, ref_motion_path, control_dt, ticker_value, ref_motion_start_index),
    )
    process.start()
    return process
