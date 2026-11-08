# CHIP Training

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

Isaac Lab training code for the CHIP whole-body tracking policy on the Unitree G1
(task `Tracking-Flat-chip-G1-v0`).

The policy tracks a lower-body joint reference plus three sparse "VR" keypoints (both
hands and the head). Random external forces are applied to these keypoints during
training. The keypoint goals the policy observes are shifted in hindsight by
`stiffness × applied_force`, and the stiffness is also given to the policy as an
observation. This lets one policy behave anywhere from stiff tracking to compliant
interaction, while the dense tracking rewards still use the original reference.

## Hardware

The paper's policy was trained on **8 GPUs × 8192 environments** (one node with 8× L40/L40S-class GPUs).
A single GPU with a small `--num_envs` is enough to check that training runs.

## Installation

From this `train/` directory:

```bash
bash setup.sh            # creates conda env `chip` (override with ENV_NAME=...)
conda activate chip
```

`setup.sh` installs Isaac Sim 4.5.0 and Isaac Lab v2.2.0 (cloned next to this repository), downloads the
Unitree G1 description into `source/whole_body_tracking/whole_body_tracking/assets/`, and installs this
package in editable mode. If you want W&B logging (`--logger wandb`), run `wandb login`.

## Data

Download the retargeted G1 training motions (1750 clips at 50 FPS: AMASS and OMOMO motions retargeted
with GMR, plus LAFAN1 clips) and unzip:

**[chip_training.zip (Google Drive)](https://drive.google.com/file/d/1S3FNFCd85MH4TYUq1rv16OWLDr99Xdus/view?usp=sharing)**

```bash
pip install gdown
gdown 1S3FNFCd85MH4TYUq1rv16OWLDr99Xdus
unzip chip_training.zip -d /path/to/data
# -> /path/to/data/twist_lafan_compliance/*.npz   (1750 clips)
```

Each `.npz` holds `fps`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and
`body_ang_vel_w`, with joints and bodies in Isaac Lab order. To convert your own retargeted motions
(BeyondMimic / LAFAN-style CSV), use `scripts/csv_to_npz.py`. To preview a clip, use
`scripts/replay_npz.py --motion_file <clip.npz>`.

## Training

Full-scale training (8 GPUs × 8192 envs, as in the paper):

```bash
bash train.sh /path/to/data/twist_lafan_compliance my_chip_run
```

which runs

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 scripts/rsl_rl/train.py \
    --task=Tracking-Flat-chip-G1-v0 --headless --distributed \
    --num_envs 8192 \
    --motion_dir /path/to/data/twist_lafan_compliance \
    --run_name my_chip_run \
    --logger wandb --log_project_name chip \
    --max_iterations 60000
```

Quick single-GPU smoke test:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-chip-G1-v0 --headless \
    --num_envs 512 --max_iterations 10 \
    --motion_dir /path/to/data/twist_lafan_compliance
```

Checkpoints go to `logs/rsl_rl/g1_flat/<timestamp>_<run_name>/model_*.pt`. An ONNX policy is
exported next to each checkpoint. To resume, use `--resume True --load_run <run_folder> --checkpoint model_XXXX.pt`.

## Evaluation and ONNX Export

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-chip-G1-v0 --num_envs 2 \
    --motion_dir /path/to/data/twist_lafan_compliance \
    --checkpoint_path logs/rsl_rl/g1_flat/<run>/model_XXXX.pt
# or: --wandb_path <entity>/<project>/<run_id>
```

`play.py` writes `<run>/exported/policy.onnx`, with joint names, default joint positions, action scale,
and PD gains stored in the ONNX metadata. This file is what [`../deploy`](../deploy) runs. To export
every checkpoint of a run without launching Isaac Sim, use `scripts/rsl_rl/export_run_pt_to_onnx.py <run_dir>`.

## Code Structure

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`: base scene, observations, rewards, events, and terminations
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`: the CHIP G1 task (`G1FlatChipEnvCfg`): compliance observation, hindsight-compliant keypoint targets, stiffness randomization, and observation history
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/`
  - `commands.py`: multi-clip motion command with adaptive sampling, keypoint targets, and external-force/stiffness buffers
  - `events.py`: `force_based_push` (random keypoint forces) and `change_compliance` (stiffness randomization)
  - `observations.py`: `vr_3point_local_compliant_target`, `compliance`, and other observation terms
  - `rewards.py`, `terminations.py`: tracking rewards and termination terms
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`: PPO hyperparameters
- `scripts/rsl_rl/`: `train.py`, `play.py`, and ONNX export utilities

## Acknowledgments

Built on [BeyondMimic](https://beyondmimic.github.io/) / [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) and [Isaac Lab](https://isaac-sim.github.io/IsaacLab).
