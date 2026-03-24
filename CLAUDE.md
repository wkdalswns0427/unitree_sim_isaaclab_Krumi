# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab simulation environment for Unitree humanoid robots (G1-29dof, H1-2). Supports teleoperation data collection, episode replay/generation, and PPO reinforcement learning training. Communication with real robots and external systems uses CycloneDDS via the Unitree SDK.

## Environment Setup

```bash
conda activate rical_unitree
pip install -r requirements.txt

# Required for teleimager module
export PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src

# Fetch USD assets (requires git-lfs, run once)
. fetch_assets.sh
```

## Key Commands

**Run simulation (teleoperation):**
```bash
python sim_main.py --device cuda --enable_cameras --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint --enable_dex1_dds --robot_type g129
```

**Run wholebody task (requires keyboard publisher in a second terminal):**
```bash
# Terminal A
python sim_main.py --device cuda --enable_cameras --task Isaac-Move-Cylinder-H12-WholeBody --enable_inspire_dds --robot_type h1_2

# Terminal B
python nontask_control/h12_walk_keyboard.py --backend stdin --channel 1
```

**Replay existing dataset:**
```bash
python sim_main.py --device cuda --enable_cameras --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint --enable_dex1_dds --robot_type g129 --replay_data --file_path /path/to/episode_dir
```

**Generate new episodes (replay + generate):**
```bash
python sim_main.py --device cuda ... --replay_data --file_path /path/to/source --generate_data --generate_data_dir ./data_gen
```

**Train H1-2 wholebody policy (RSL-RL PPO):**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Move-Cylinder-H12-WholeBody \
  --device cuda --headless --num_envs 64 --max_iterations 3000
```
Training logs and ONNX export go to `logs/rsl_rl/<experiment_name>/<timestamp>/`.

**Run sim with trained ONNX policy:**
```bash
python sim_main.py --device cuda --enable_cameras --task Isaac-Move-Cylinder-H12-WholeBody \
  --enable_inspire_dds --robot_type h1_2 \
  --model_path logs/rsl_rl/h12_move_cylinder_wholebody/<timestamp>/exported/policy.onnx
```

## Architecture

### Entry Points
- **[sim_main.py](sim_main.py)** — Main simulation loop. Parses args, creates Isaac Lab environment, sets up DDS, creates an `ActionProvider`, and runs the step loop.
- **[scripts/reinforcement_learning/rsl_rl/train.py](scripts/reinforcement_learning/rsl_rl/train.py)** — RSL-RL PPO training. Loads env/agent configs from gym registry, trains, then exports `policy.pt` and `policy.onnx`.

### Task Registration (`tasks/`)
Tasks are gymnasium environments registered in each task's `__init__.py`. All tasks are auto-imported by [tasks/\_\_init\_\_.py](tasks/__init__.py) via `import_packages`.

Task naming convention: `Isaac-<Action>-<Object>-<Robot>-<Hand>-<ControlMode>`

**Robot types:**
- `g1_tasks/` — G1-29dof robot with Dex1 gripper, Dex3 hand, or Inspire hand
- `h1-2_tasks/` — H1-2 robot with Inspire hand (27-dof)

**Control modes:**
- `Joint` — Joint-space teleoperation (DDS action provider)
- `Wholebody` — Whole-body locomotion + manipulation (RL policy action provider)

Each task folder contains:
- `__init__.py` — gym.register call with env and agent config entry points
- `*_env_cfg.py` — Environment config (scene, observations, rewards, terminations, actions)
- `mdp/` — Task-specific observations, rewards, terminations
- `agents/rsl_rl_ppo_cfg.py` — PPO runner config (for trainable tasks)

**Shared task components (`tasks/common_*/`):**
- `common_scene/` — Base scene configs (object placements, cameras)
- `common_rewards/` — Shared reward functions per task type
- `common_termination/` — Shared termination conditions per task type
- `common_observations/` — Robot state observers (joint positions/velocities for each robot/hand variant), camera state

### DDS Communication (`dds/`)
CycloneDDS-based pub/sub system using `unitree_sdk2py`. A central `dds_manager` registers named DDS objects.

Key DDS objects registered at startup (via [dds/dds_create.py](dds/dds_create.py)):
- `G1RobotDDS` — Main robot arm joint states/commands
- `GripperDDS` / `Dex3DDS` / `InspireDDS` — Hand joint states/commands
- `RunCommandDDS` — High-level locomotion velocity commands (Wholebody tasks)
- `ResetPoseDDS` — Reset object/scene commands
- `SimStateDDS` — Publishes current env state as JSON
- `RewardsDDS` — Publishes reward signals

### Action Providers (`action_provider/`)
Implement `ActionProvider` ABC with `get_action(env) -> Tensor`:
- `DDSActionProvider` — Reads joint targets from DDS (teleoperation)
- `DDSRLActionProvider` — Runs ONNX RL policy with high-level run commands from DDS (Wholebody)
- `FileActionProviderReplay` — Replays from `data.json` episodes

### Layered Control (`layeredcontrol/`)
`RobotController` holds an `ActionProvider` and calls `controller.step()` each iteration, which internally calls `action_provider.get_action(env)` and steps the Isaac Lab environment.

### Tools (`tools/`)
Utilities for data I/O, episode writing, reward reading, augmentation (lights/cameras), USD editing, and rerun visualization.

## Task-Specific Notes

- `--task` must exactly match the `--task` used when the source dataset was recorded (replay loader validates this).
- Wholebody tasks auto-set `--action_source dds_wholebody`; the RL policy runs on-board and accepts high-level run commands from the keyboard publisher.
- `--enable_dex1_dds` / `--enable_dex3_dds` / `--enable_inspire_dds` are mutually exclusive.
- If `DISPLAY` is not set, Isaac Sim runs headless automatically; GLFW warnings are expected and non-fatal.

## Common Issues

- `ModuleNotFoundError: teleimager.image_server` → export `PYTHONPATH=$PYTHONPATH:$(pwd)/teleimager/src`
- `task_name ... is different from ...` → `--task` flag must match the task name embedded in the source dataset
- `No module named rerun.blueprint` → `rerun` is optional; only needed with `--rerun_log`
