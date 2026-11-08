"""Run the CHIP policy in MuJoCo (sim2sim) or on a real Unitree G1 (sim2real).

The policy only uses proprioception (joint states + IMU); no root position / velocity estimate is needed.
"""
import os
from argparse import ArgumentParser
from multiprocessing import Value

import yaml
from scipy.spatial.transform import Rotation

from rl_policy import RLCHIPPolicy
from utils.math_utils import heading_zup
from utils.params import ISAAC_TO_MUJOCO
from utils.robot_states import G1RobotState
from utils.robot_utils import Rate


def main(env, policy, config, ticker_value=None, max_steps=None, on_step=None):
    env.set_robot_state(policy.get_q_init())
    env.maintain_state(policy.get_q_init())

    rate = Rate(1 / (config.get("control_dt", 0.02) * config.get("slow_down", 1.0)))
    robot_state = G1RobotState()

    env.release_robot()  # let the robot move
    init_heading_rot = None
    step = 0
    while max_steps is None or step < max_steps:
        robot_state.q, robot_state.dq, robot_state.imu_quat, robot_state.omega = env.get_robot_state()
        # Root orientation from the IMU, with the initial heading removed.
        if init_heading_rot is None:
            init_heading_rot = Rotation.from_euler("z", heading_zup(robot_state.imu_quat))
        robot_state.root_orn = (init_heading_rot.inv() * Rotation.from_quat(robot_state.imu_quat)).as_quat()

        control_signals = policy.prepare_control_signals(robot_state)
        obs = policy.prepare_obs(robot_state, control_signals)
        action = policy.get_action(obs, start_ticker=env.get_start_ticker())
        if ticker_value is not None:
            ticker_value.value = float(policy.ticker)

        robot_state.last_action = action.copy()
        scaled_action = action[ISAAC_TO_MUJOCO] * policy.action_scale + policy.default_value["q"][ISAAC_TO_MUJOCO]
        env.step_robot(scaled_action)
        if on_step is not None:
            on_step(step, env, policy)
        step += 1
        rate.sleep()


def build_policy(config, use_sim):
    return RLCHIPPolicy(
        config["onnx_model_path"],
        config["obs_names"],
        config["ref_motion_path"],
        use_sim=use_sim,
        lookahead_steps=config.get("lookahead_steps", 1),
        lookahead_frame_skips=config.get("lookahead_frame_skips", 1),
        hist_names=config.get("history_names", []),
        hist_length=config.get("history_length", 1),
        ref_motion_start_index=config.get("ref_motion_start_index", 0),
        init_compliance=config.get("init_compliance", [0.5, 0.5, 0.0]),
    )


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)["rl_policy"]
    # Resolve relative paths against the deploy root so the script works from any cwd.
    root = os.path.dirname(os.path.abspath(__file__))
    for key in ["onnx_model_path", "ref_motion_path", "mujoco_xml_path"]:
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.join(root, config[key])
    return config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment yaml.")
    parser.add_argument("--use_sim", action="store_true", help="Run in MuJoCo instead of on the real robot.")
    parser.add_argument("--net", type=str, default=None, help="Network interface connected to the robot (real robot only).")
    parser.add_argument("--headless", action="store_true", help="Sim only: run without the MuJoCo viewer and skip the Enter prompt.")
    parser.add_argument("--show_ref", action="store_true", help="Sim only: show the reference motion in a second viewer.")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many control steps.")
    parser.add_argument("--slow_down", type=float, default=1.0, help="Run simulation and policy x times slower.")
    args = parser.parse_args()

    config = load_config(args.config)
    config["slow_down"] = args.slow_down

    ticker_value = None
    ref_vis_process = None
    if args.use_sim:
        from mujoco_env import MujocoRobot

        env = MujocoRobot(config["mujoco_xml_path"], config, headless=args.headless, auto_start=args.headless)
        if args.show_ref and not args.headless:
            from ref_motion_visualizer import start_ref_visualizer_process

            ticker_value = Value("f", float(config.get("ref_motion_start_index", 0)))
            ref_vis_process = start_ref_visualizer_process(
                config["mujoco_xml_path"],
                config["ref_motion_path"],
                ticker_value=ticker_value,
                ref_motion_start_index=config.get("ref_motion_start_index", 0),
            )
    else:
        from real_env import UnitreeRobot

        env = UnitreeRobot(args.net, config)

    policy = build_policy(config, use_sim=args.use_sim)

    try:
        main(env, policy, config, ticker_value=ticker_value, max_steps=args.max_steps)
    finally:
        if ref_vis_process is not None:
            ref_vis_process.terminate()
            ref_vis_process.join()
        if args.use_sim:
            env.close()
