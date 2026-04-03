from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from tasks.common_observations.inspire_state import (
    get_robot_inspire_joint_states,
    get_robot_girl_joint_names,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_vel_cache: dict = {}


def get_inspire_joint_pos(
    env: ManagerBasedRLEnv,
    enable_dds: bool = False,
) -> torch.Tensor:
    """Inspire proximal joint positions (12 dims)."""
    return get_robot_inspire_joint_states(env, enable_dds=enable_dds)


def get_inspire_joint_vel(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Inspire proximal joint velocities (12 dims)."""
    global _vel_cache

    joint_vel = env.scene["robot"].data.joint_vel
    device = joint_vel.device
    batch = joint_vel.shape[0]
    joint_names = env.scene["robot"].data.joint_names
    joint_key = id(joint_names)

    if _vel_cache.get("joint_key") != joint_key or _vel_cache.get("device") != device:
        inspire_names = get_robot_girl_joint_names()
        joint_to_idx = {name: i for i, name in enumerate(joint_names)}
        idx = torch.tensor([joint_to_idx[n] for n in inspire_names], dtype=torch.long, device=device)
        _vel_cache = {"joint_key": joint_key, "device": device, "idx": idx, "idx_batch": None}

    idx = _vel_cache["idx"]
    if _vel_cache.get("idx_batch") is None or _vel_cache["idx_batch"].shape[0] != batch:
        _vel_cache["idx_batch"] = idx.unsqueeze(0).expand(batch, -1)

    return torch.gather(joint_vel, 1, _vel_cache["idx_batch"])


# Fist target angles for the 12 proximal joints (same order as get_robot_girl_joint_names):
#   R_pinky, R_ring, R_middle, R_index, R_thumb_pitch, R_thumb_yaw,
#   L_pinky, L_ring, L_middle, L_index, L_thumb_pitch, L_thumb_yaw
_FIST_CLOSED = torch.tensor([
    1.5, 1.5, 1.5, 1.5, 0.4, 1.0,   # right hand
    1.5, 1.5, 1.5, 1.5, 0.4, 1.0,   # left hand
])
_FIST_OPEN = torch.zeros(12)

_target_cache: dict = {}


def get_fist_target(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Return the current fist target (12 dims): 0=open, fist angles=closed.

    Alternates between open and closed every 2.5 seconds based on episode step count.
    """
    global _target_cache
    device = env.device
    batch = env.num_envs

    if _target_cache.get("device") != device:
        _target_cache = {
            "device": device,
            "closed": _FIST_CLOSED.to(device),
            "open": _FIST_OPEN.to(device),
        }

    # Toggle period: 2.5 seconds. dt per policy step = sim_dt * decimation
    dt = env.step_dt
    toggle_steps = int(2.5 / dt) if dt > 0 else 125
    if toggle_steps < 1:
        toggle_steps = 1

    # Use episode_length_buf to determine phase within episode.
    step = env.episode_length_buf  # [N] integer step count
    phase = (step // toggle_steps) % 2  # 0 or 1

    closed = _target_cache["closed"].unsqueeze(0).expand(batch, -1)
    opened = _target_cache["open"].unsqueeze(0).expand(batch, -1)

    mask = (phase == 1).unsqueeze(-1).float()  # [N, 1]
    return mask * closed + (1.0 - mask) * opened


__all__ = [
    "get_inspire_joint_pos",
    "get_inspire_joint_vel",
    "get_fist_target",
]