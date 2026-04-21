# unitree_sim_isaaclab Local Run Guide (No Docker)

This guide is for running `unitree_sim_isaaclab` directly on your local Linux machine (no Docker), including:

- Running simulation tasks
- Sending keyboard movement commands
- Replaying existing episodes
- Generating `data.json` episodes for training
- (Optional) Converting episodes for `h1_mimic_tasks`

## 1. Prerequisites

- Ubuntu 22.04+ recommended
- NVIDIA driver + CUDA-capable GPU (CPU mode is also supported but slower)
- Isaac Sim + Isaac Lab already installed in your Python environment
- This repo cloned at:
  - `/home/{USER}/mj_ws/unitree_sim_isaaclab`

## 2. Environment Setup (Local)

From your host machine terminal:

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
pip install -r requirements.txt
```

Download assets once:

```bash
sudo apt update
sudo apt install -y git-lfs
. fetch_assets.sh
```

If you hit `ModuleNotFoundError: teleimager.image_server`, export:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src
```

## 3. Run Simulation Locally

Example (GPU):

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src
python sim_main.py --device cuda --enable_cameras --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint --enable_dex1_dds --robot_type g129
```

Notes:

- If `DISPLAY` is not set, Isaac Sim runs headless automatically.
- Headless `GLFW` warnings are common and not fatal by themselves.

## 4. Move Robot with Keyboard (Wholebody Tasks Only)

Keyboard control publishes DDS run commands and is intended for tasks containing `Wholebody`.

Terminal A (run sim):

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
python sim_main.py --device cuda --enable_cameras --task Isaac-Move-Cylinder-G129-Dex1-Wholebody --enable_dex1_dds --robot_type g129

python sim_main.py --device cuda --enable_cameras --task Isaac-Move-Cylinder-H12-WholeBody --enable_inspire_dds --robot_type h1_2

```

Terminal B (keyboard publisher):

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
python send_commands_keyboard.py --backend stdin --channel 1
```

H1-2 dedicated keyboard publisher (includes reset keys):

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
python nontask_control/h12_walk_keyboard.py --backend stdin --channel 1
```

Default keys:

- `W/S`: forward/backward
- `A/D`: left/right
- `Z/X`: rotate left/right
- `C`: crouch
- `Q`: quit keyboard publisher
- `U`: reset object (`reset category=1`, `h12_walk_keyboard.py`)
- `P`: reset all (`reset category=2`, `h12_walk_keyboard.py`)

Important:
- Keyboard control in `Wholebody` tasks sends high-level run commands (`x/y/yaw/height`) to the RL policy.
- If the loaded policy is not trained for your robot/task pair, the robot may not move even though commands are being published.

## 4.1 Train H1-2 Wholebody Policy (PPO)

This repo now includes a local RSL-RL training entrypoint for:
- `Isaac-Move-Cylinder-H12-WholeBody`

Run training:

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Move-Cylinder-H12-WholeBody \
  --device cuda \
  --headless \
  --num_envs 64 \
  --max_iterations 3000
```

Outputs:
- checkpoints/logs: `logs/rsl_rl/h12_move_cylinder_wholebody/<run_timestamp>/`
- exported policy: `logs/rsl_rl/h12_move_cylinder_wholebody/<run_timestamp>/exported/policy.onnx`

Run sim with your trained ONNX:

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-Move-Cylinder-H12-WholeBody \
  --enable_inspire_dds \
  --robot_type h1_2 \
  --model_path logs/rsl_rl/h12_move_cylinder_wholebody/<run_timestamp>/exported/policy.onnx
```

## 5. H1-2 Locomotion (Velocity Walking) Training

### Available tasks

| Task ID | Terrain | Notes |
|---|---|---|
| `Isaac-H12-Velocity-ManagerBased-v0` | Flat | Main locomotion task, 21-DOF control |

### Train

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-H12-Velocity-ManagerBased-v0 \
  --device cuda --headless \
  --num_envs 4096 \
  --max_iterations 5000
```

Use `--num_envs 1024` if VRAM is limited. Logs and checkpoints go to:
```
logs/rsl_rl/h12_velocity_flat/<timestamp>/
logs/rsl_rl/h12_velocity_flat/<timestamp>/exported/policy.onnx   ← auto-exported
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/rsl_rl/h12_velocity_flat
```

### Play

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-H12-Velocity-ManagerBased-v0 \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/h12_velocity_flat/<timestamp>/model_5000.pt
```

Pass `--num_envs 1` for cleaner single-robot debugging.

### Key config files

| File | What to tune |
|---|---|
| `tasks/h1-2_tasks/h12_velocity/rough_env_cfg.py` | Rewards, scene, actuator gains, command ranges |
| `tasks/h1-2_tasks/h12_velocity/flat_env_cfg.py` | Flat-terrain overrides, play config |
| `tasks/h1-2_tasks/h12_velocity/agents/rsl_rl_ppo_cfg.py` | Network size, learning rate, iterations |

### Reward composition (current)

Defined in `H12Rewards` inside `rough_env_cfg.py`, with `__post_init__` overrides:

| Term | Weight | Purpose |
|---|---|---|
| `track_lin_vel_xy_exp` | +1.5 | Forward/lateral velocity tracking |
| `track_ang_vel_z_exp` | +1.0 | Yaw tracking |
| `feet_air_time_positive_biped` | +0.75, thr=0.35s | Forces clear foot lift |
| `feet_slide` | -0.4 | Penalizes foot slip during contact |
| `base_height_l2` | -0.5, target=0.98m | Prevents excessive crouching |
| `flat_orientation_l2` | -1.0 | Keeps torso upright |
| `lin_vel_z_l2` | -0.5 | Penalizes vertical bouncing |
| `ang_vel_xy_l2` | -0.05 | Penalizes roll/pitch rate |
| `dof_torques_l2` | -1e-7 | Energy efficiency (lower body) |
| `dof_acc_l2` | -2.5e-7 | Joint smoothness (lower body) |
| `action_rate_l2` | -0.005 | Smooth action changes |
| `joint_deviation_hip` | -0.2 | Hip yaw/roll near zero |
| `joint_deviation_arms` | -0.1 | Arms near default (allow swing) |
| `joint_deviation_torso` | -0.1 | Torso joint near zero |
| `stand_still` | -0.5 | No jitter at zero command |
| `dof_pos_limits` | -1.0 | All lower-body joint limits |
| `termination_penalty` | -200.0 | Discourages falls |

### PPO config (current)

- Network: `[512, 256, 128]` ELU (actor + critic)
- Action scale: `0.75` (applied to delta from default joint positions)
- `entropy_coef`: `0.01`, `learning_rate`: `1e-3`, `desired_kl`: `0.01`
- 21 controlled joints: 12 leg + 1 torso + 4 left arm + 4 right arm (no wrists)

### Checkpoint compatibility

Older checkpoints trained with `log_std` (pre-rsl_rl API change) are auto-migrated on load. No manual patching needed.

### Common issues

- **Shuffling / stiff legs**: increase `feet_air_time` weight, check `action_scale` is `0.75`, remove any knee joint deviation penalty.
- **Falls backward**: ankle pitch too positive; try moving `ankle_pitch` default toward `-0.3` in `robot_configs.py`.
- **Falls forward**: ankle pitch too negative; try `-0.22`.
- **Architecture mismatch on load**: the PPO config network shape must match the checkpoint. Old checkpoints used `[128, 128, 128]`.

---

## 6. Replay Existing Dataset

Use replay mode to load existing `data.json` episodes:

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-Stack-RgyBlock-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --replay_data \
  --file_path /path/to/episode_root_or_data_json
```

Important:

- Use `--replay_data` (current code flag), not `--replay`.
- `--file_path` can be:
  - One `data.json`
  - A directory containing `episode_*/data.json`

## 6. Generate New `data.json` Episodes

In this repo, generation is wired through replay mode. Run replay + generation together:

```bash
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-Stack-RgyBlock-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --replay_data \
  --file_path /path/to/source_dataset \
  --generate_data \
  --generate_data_dir ./data_gen
```

Optional flags:

- `--modify_light`
- `--modify_camera`
- `--rerun_log` (visualization only; not required for data generation)

Output layout:

```text
data_gen/
  episode_0000/
    data.json
    colors/
    depths/
    audios/
  episode_0001/
    ...
```

Quick check:

```bash
find ./data_gen -name data.json | sort
```

## 7. Convert Generated Episodes for `h1_mimic_tasks` (Optional)

If you want to feed these episodes into your mimic workflow:

```bash
cd /home/{USER}/mj_ws/IsaacLab_Humanoid/h1_mimic_tasks
conda activate {unitree_sim condaenv}
python scripts/mimic/import_unitree_reference.py \
  --input_path /home/{USER}/mj_ws/unitree_sim_isaaclab/data_gen \
  --output outputs/mimic/unitree_reference_raw.hdf5
```

## 8. H1-2 Wholebody -> Mimic Reference Pipeline

If your goal is to use H1-2 wholebody trajectories from this repo as Mimic reference data, use this exact flow.

Step 1: generate Unitree episodes (`data.json`) from replay.

```bash
cd /home/{USER}/mj_ws/unitree_sim_isaaclab
conda activate rical_unitree
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src

python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-Move-Cylinder-H12-WholeBody \
  --enable_inspire_dds \
  --robot_type h1_2 \
  --replay_data \
  --file_path /path/to/source_data_json_or_episode_dir \
  --generate_data \
  --generate_data_dir ./data_gen_h12
```

Step 2: verify generated episodes.

```bash
find ./data_gen_h12 -name data.json | sort
```

Step 3: convert to HDF5 reference in `h1_mimic_tasks`.

```bash
cd /home/{USER}/mj_ws/IsaacLab_Humanoid/h1_mimic_tasks
conda activate rical_unitree

python scripts/mimic/import_unitree_reference.py \
  --input_path /home/{USER}/mj_ws/unitree_sim_isaaclab/data_gen_h12 \
  --output outputs/mimic/unitree_reference_raw.hdf5 \
  --write_states
```

Step 4: run Mimic annotation and dataset generation.

```bash
export ISAACLAB_ROOT=/home/{USER}/RICAL_IsaacLab

python scripts/mimic/annotate_demos.py \
  --task H1-Pick-Block-Mimic-v0 \
  --input_file outputs/mimic/unitree_reference_raw.hdf5 \
  --output_file outputs/mimic/unitree_reference_annotated.hdf5 \
  --auto

python scripts/mimic/generate_dataset.py \
  --task H1-Pick-Block-Mimic-v0 \
  --input_file outputs/mimic/unitree_reference_annotated.hdf5 \
  --output_file outputs/mimic/unitree_reference_generated.hdf5
```

Important caveat:
- `import_unitree_reference.py` stores actions as `[left_arm, right_arm, left_ee, right_ee]`.
- `H1-Pick-Block-Mimic-v0` currently uses a 6D IK delta-pose action.
- So this is a reference-motion bridge, not guaranteed plug-and-play training data without action-space mapping.

## 9. Common Issues

- `No module named rerun.blueprint`:
  - `rerun` is optional unless you enable `--rerun_log`.
  - Data generation does not require rerun.
- `task_name ... is different from ...`:
  - Replay loader checks dataset task name. Match `--task` to source data.
- Headless warnings (`GLFW`, `MESA`, `left-click sim window`):
  - Usually expected on servers without display.
