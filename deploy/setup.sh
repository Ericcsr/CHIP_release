#!/bin/bash
# Creates the `chip_deploy` conda environment for sim2sim and sim2real deployment.
#   bash setup.sh          # MuJoCo sim2sim only
#   bash setup.sh --real   # also builds CycloneDDS and installs unitree_sdk2_python (real G1)
set -e

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=${ENV_NAME:-chip_deploy}

eval "$(conda shell.bash hook)"
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"
pip install -r "$DEPLOY_DIR/requirements.txt"

if [ "$1" == "--real" ]; then
    # Requires: git, cmake, a C compiler (sudo apt install git cmake build-essential)
    THIRD_PARTY="$DEPLOY_DIR/third_party"
    mkdir -p "$THIRD_PARTY"

    # CycloneDDS 0.10.x (DDS middleware used by the Unitree SDK)
    if [ ! -d "$THIRD_PARTY/cyclonedds/install" ]; then
        git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x "$THIRD_PARTY/cyclonedds"
        mkdir -p "$THIRD_PARTY/cyclonedds/build"
        cd "$THIRD_PARTY/cyclonedds/build"
        cmake .. -DCMAKE_INSTALL_PREFIX="$THIRD_PARTY/cyclonedds/install" -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
        cmake --build . --target install -j"$(nproc)"
    fi
    export CYCLONEDDS_HOME="$THIRD_PARTY/cyclonedds/install"

    # Unitree SDK2 python bindings
    if [ ! -d "$THIRD_PARTY/unitree_sdk2_python" ]; then
        git clone https://github.com/unitreerobotics/unitree_sdk2_python.git "$THIRD_PARTY/unitree_sdk2_python"
    fi
    pip install -e "$THIRD_PARTY/unitree_sdk2_python"
    python -c "import unitree_sdk2py; print('unitree_sdk2py installed')"
fi

echo "Done. Activate with: conda activate $ENV_NAME"
