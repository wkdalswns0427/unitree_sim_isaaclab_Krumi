#!/usr/bin/env python3

# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Train a local Unitree task with RSL-RL (without editing IsaacLab core scripts)."""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)


parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL for local Unitree tasks.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Move-Cylinder-H12-WholeBody",
    help="Task name.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent config entry point key from gym task registration.",
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max iterations from agent cfg.")
parser.add_argument("--experiment_name", type=str, default=None, help="Override experiment name.")
parser.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval (in steps).")
parser.add_argument("--export_onnx", action="store_true", default=True, help="Export ONNX after training.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Default to non-headless unless the user explicitly asks for headless mode.
if "--headless" not in sys.argv:
    args_cli.headless = False

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from torch.distributions import Normal

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import tasks  # noqa: F401


def _patch_rsl_policy_distribution_safety():
    """Patch rsl_rl ActorCritic distribution update to avoid invalid std crashes."""
    try:
        import rsl_rl.modules.actor_critic as actor_critic_mod
    except Exception:
        return

    actor_critic_cls = getattr(actor_critic_mod, "ActorCritic", None)
    if actor_critic_cls is None or getattr(actor_critic_cls, "_unitree_safe_std_patch", False):
        return

    def _safe_update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

        # Keep distribution numerically valid even if upstream updates explode transiently.
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        std = torch.clamp(std, min=1.0e-6, max=10.0)
        self.distribution = Normal(mean, std)

    actor_critic_cls.update_distribution = _safe_update_distribution
    actor_critic_cls._unitree_safe_std_patch = True
    print("[INFO] Applied safe-std patch for rsl_rl ActorCritic distribution.")


def _extract_policy_and_normalizer(runner):
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None
    return policy_nn, normalizer


def main():
    _patch_rsl_policy_distribution_safety()

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    # Guard against negative/invalid policy std in rsl_rl ActorCritic.
    if hasattr(agent_cfg, "policy"):
        if getattr(agent_cfg.policy, "noise_std_type", None) != "log":
            agent_cfg.policy.noise_std_type = "log"
        if hasattr(agent_cfg.policy, "init_noise_std") and agent_cfg.policy.init_noise_std <= 0.0:
            agent_cfg.policy.init_noise_std = 1.0
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if hasattr(agent_cfg, "policy"):
        print(
            "[INFO] Policy noise config:",
            f"noise_std_type={getattr(agent_cfg.policy, 'noise_std_type', None)}",
            f"init_noise_std={getattr(agent_cfg.policy, 'init_noise_std', None)}",
        )

    # Keep training lightweight and avoid camera shared-memory paths unless video recording is requested.
    if not args_cli.video and hasattr(env_cfg, "scene"):
        for camera_name in ("front_camera", "left_wrist_camera", "right_wrist_camera", "robot_camera", "world_camera"):
            if hasattr(env_cfg.scene, camera_name):
                setattr(env_cfg.scene, camera_name, None)
        # Some custom tasks include image terms inside the policy observations.
        # RSL-RL MLP runners expect 1D observations, so drop camera term in non-video training.
        if (
            hasattr(env_cfg, "observations")
            and hasattr(env_cfg.observations, "policy")
            and hasattr(env_cfg.observations.policy, "camera_image")
        ):
            env_cfg.observations.policy.camera_image = None
    if hasattr(env_cfg, "rewards") and hasattr(env_cfg.rewards, "reward"):
        reward_params = getattr(env_cfg.rewards.reward, "params", None)
        if reward_params is None:
            env_cfg.rewards.reward.params = {}
        env_cfg.rewards.reward.params["enable_dds"] = False

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logging to: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    if args_cli.export_onnx:
        policy_nn, normalizer = _extract_policy_and_normalizer(runner)
        export_dir = os.path.join(log_dir, "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
        print(f"[INFO] Exported policy to: {export_dir}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        print(f"[ERROR] Training failed: {exc}")
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
