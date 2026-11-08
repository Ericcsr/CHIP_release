# CHIP Deployment

This directory deploys a trained CHIP policy on the Unitree G1 (29 DoF). You can run it in MuJoCo (sim2sim) or on the real robot (sim2real).
The policy reads only proprioception: joint positions and velocities, IMU orientation, and IMU angular velocity. It does **not** need a root position or velocity estimate.
It tracks a reference motion, given as lower-body joint commands plus wrist and head 3-point targets. A runtime command sets the hand compliance.

## Installation

```bash
bash setup.sh            # MuJoCo sim2sim only
bash setup.sh --real     # also builds CycloneDDS and installs the Unitree SDK (real robot)
conda activate chip_deploy
```

For sim2sim only, you can instead run `pip install -r requirements.txt` (Python 3.10).

### Unitree SDK (real robot only)

`bash setup.sh --real` installs [CycloneDDS 0.10.x](https://github.com/eclipse-cyclonedds/cyclonedds) and
[unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) into `third_party/`. It needs
`git`, `cmake` and a C compiler (`sudo apt install git cmake build-essential`). To install them manually
into an existing environment:

```bash
mkdir -p third_party && cd third_party

# 1. CycloneDDS 0.10.x
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build . --target install
cd ../..
export CYCLONEDDS_HOME="$(pwd)/cyclonedds/install"

# 2. Unitree SDK2 python bindings
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
pip install -e unitree_sdk2_python

# check
python -c "import unitree_sdk2py; print('ok')"
```

`CYCLONEDDS_HOME` is only needed while installing, not at runtime. The simulation path never imports the SDK.

## Sim2Sim (MuJoCo)

```bash
python run_controller.py --use_sim --config exported_policies/local_chip/experiment.yaml
```

1. The robot hangs from a virtual elastic band and moves to the first frame of the reference motion.
2. Press **Enter** in the terminal. The band is released, the policy starts, and the reference motion begins playing.

Options:
- `--show_ref`: open a second viewer that plays the reference motion in sync with the policy.
- `--headless`: no viewer and no Enter prompt. Combine with `--max_steps N` for a quick check.
- `--slow_down X`: run the simulation and the policy X times slower.

Elastic band keys (in the MuJoCo viewer): `7` / `8` lower or raise the band, `9` toggles it.

## Sim2Real (Unitree G1)

> **Warning:** Hang the robot from a gantry before launching. The robot moves to the initial pose immediately when the script starts.

Put the robot in debug mode (damping), connect over Ethernet, then run:

```bash
python run_controller.py --net <network_interface> --config exported_policies/local_chip/experiment.yaml
```

| Remote controller | Action |
|-------------------|--------|
| `L1 + A` | Start the policy. It balances at the first reference frame. |
| `start`  | Start playing the reference motion. |
| `B`      | Reset the reference motion to its first frame. |
| `select` | Emergency stop (damping mode) and exit. |

## Hand compliance (keyboard, sim and real)

The keys below are read from the keyboard of the machine running `run_controller.py` (via `pynput`).
If no display or keyboard backend is available, compliance stays fixed at `init_compliance`.

| Key | Compliance (left, right, -) |
|-----|-----------------------------|
| `g` | `0.0, 0.0, 0.0` (stiff) |
| `h` | `0.5, 0.5, 0.0` (both hands compliant, default) |
| `j` | `0.0, 0.5, 0.0` (right hand compliant) |

## Experiment folder

```
exported_policies/local_chip/
├── experiment.yaml   # deployment config
├── policy.onnx       # exported CHIP policy (obs dim 984, 29 actions)
└── ref_motion.npz    # example reference motion (first 60 s of LAFAN walk1_subject1 from the training set)
```

`experiment.yaml` fields (under `rl_policy`, paths relative to this directory):

| Field | Meaning |
|-------|---------|
| `onnx_model_path` | Exported policy. Default joint pose and action scale are read from the ONNX metadata. |
| `ref_motion_path` | Reference motion `.npz` with `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w` (same format as the training motions, 50 Hz, Isaac joint/body order). |
| `ref_motion_start_index` | First reference frame to track. |
| `mujoco_xml_path` | MuJoCo scene (sim only). |
| `obs_names`, `history_names`, `history_length` | Observation layout. Must match the training config of the exported policy. |
| `init_compliance` | Initial hand compliance command. |
| `control_dt`, `simulation_dt` | Policy rate (50 Hz) and MuJoCo physics step. |
| `joint_stiffness`, `joint_damping`, `torque_limit` | PD gains and torque limits (MuJoCo / Unitree joint order). |

To track another motion, point `ref_motion_path` at any motion `.npz` from the training data.

## Deploying your own policy

Export the trained checkpoint to ONNX (see `../train/README.md`). Copy it into a new folder under `exported_policies/` together with an `experiment.yaml` adapted from `local_chip`.
Make sure `obs_names`, `history_names` and `history_length` match the training environment.

## Code structure

- `run_controller.py`: control loop (50 Hz) and entry point
- `rl_policy.py`: `RLCHIPPolicy`, which builds observations from the reference motion and robot state and runs ONNX inference
- `mujoco_env.py`: MuJoCo backend (physics runs in a separate process at `simulation_dt`)
- `real_env.py`: Unitree G1 low-level backend (`unitree_sdk2_python`)
- `ref_motion_visualizer.py`: optional reference motion viewer
- `utils/`: joint order maps, math helpers, history buffers
