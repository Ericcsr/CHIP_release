"""Play a trained CHIP checkpoint in Isaac Lab and export it to ONNX for deployment."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--motion_dir", type=str, required=True, help="Directory containing the motion clips (*.npz) to play."
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to a local model_*.pt checkpoint. If omitted, --wandb_path or --load_run/--checkpoint are used.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _resolve_checkpoint(agent_cfg: RslRlOnPolicyRunnerCfg) -> str:
    if args_cli.checkpoint_path:
        return os.path.abspath(args_cli.checkpoint_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path
        if "model" in run_path:
            run_path = "/".join(run_path.split("/")[:-1])
        wandb_run = wandb.Api().run(run_path)
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            files = [f.name for f in wandb_run.files() if "model" in f.name and f.name.endswith(".pt")]
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        wandb_run.file(str(file)).download("./logs/rsl_rl/temp", replace=True)
        print(f"[INFO]: Downloaded model checkpoint from: {run_path}/{file}")
        return os.path.abspath(f"./logs/rsl_rl/temp/{file}")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # Start all motions from their first frame for consistent evaluation.
    env_cfg.commands.motion.start_from_initial_frame = True

    motion_files = sorted(
        os.path.join(args_cli.motion_dir, x) for x in os.listdir(args_cli.motion_dir) if x.endswith(".npz")
    )
    if not motion_files:
        raise FileNotFoundError(f"No .npz motion files found in {args_cli.motion_dir}")
    env_cfg.commands.motion.motion_files = motion_files

    resume_path = _resolve_checkpoint(agent_cfg)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx (consumed by ../deploy)
    export_model_dir = os.path.join(log_dir, "exported")
    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path or resume_path, export_model_dir)
    print(f"[INFO]: Exported ONNX policy to {os.path.join(export_model_dir, 'policy.onnx')}")

    obs, _ = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
