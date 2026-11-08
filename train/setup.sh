#!/bin/bash
# Set up the CHIP training environment (Isaac Sim 4.5.0 + Isaac Lab v2.2.0).
# Run from the `train/` directory:  bash setup.sh
set -e

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=${ENV_NAME:-chip}

eval "$(conda shell.bash hook)"
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Isaac Lab v2.2.0 (cloned next to this repo)
pip install pip==23 setuptools==65
cd "$TRAIN_DIR/../.."
if [ ! -d IsaacLab ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git
fi
cd IsaacLab
git checkout v2.2.0
# only the rsl_rl learning framework is needed
./isaaclab.sh -i rsl_rl
# Isaac Lab v2.2.0 expects torch 2.7.0 (cu128); re-pin in case a dependency upgraded it
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Unitree G1 robot description
cd "$TRAIN_DIR"
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/
rm unitree_description.tar.gz

# This package
pip install -e source/whole_body_tracking

echo "Done. Activate with: conda activate $ENV_NAME"
echo "Optional (for --logger wandb): wandb login"
