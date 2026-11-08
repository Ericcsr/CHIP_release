#!/bin/bash
# Full-scale CHIP training: 8 GPUs x 8192 envs per GPU.
# Usage: bash train.sh /path/to/twist_lafan_compliance [run_name]
set -e
MOTION_DIR=${1:?"usage: bash train.sh <motion_dir> [run_name]"}
RUN_NAME=${2:-chip}
NUM_GPUS=${NUM_GPUS:-8}
NUM_ENVS=${NUM_ENVS:-8192}

python -m torch.distributed.run --nnodes=1 --nproc_per_node="$NUM_GPUS" scripts/rsl_rl/train.py \
    --task=Tracking-Flat-chip-G1-v0 \
    --headless \
    --distributed \
    --num_envs "$NUM_ENVS" \
    --motion_dir "$MOTION_DIR" \
    --run_name "$RUN_NAME" \
    --logger wandb --log_project_name chip \
    --max_iterations 60000
