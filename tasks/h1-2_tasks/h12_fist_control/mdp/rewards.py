from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from tasks.common_observations.inspire_state import get_robot_girl_joint_names

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_reward_cache: dict = {}

# Same target angles as observations.py
_FIST_CLOSED = torch.tensor([
    1.5, 1.5, 1.5, 1.5, 0.4, 1.0,
    1.5, 1.5, 1.5, 1.5, 0.4, 1.0,
])
_FIST_OPEN = torch.zeros(12)


def _get_inspire_pos(env):
    """Extract current inspire proximal joint positions."""
    global _reward_cache
    joint_pos = env.scene["robot"].data.joint_pos
    device = joint_pos.device
    batch = joint_pos.shape[0]
    joint_names = env.scene["robot"].data.joint_names
    joint_key = id(joint_names)

    if _reward_cache.get("joint_key") != joint_key or _reward_cache.get("device") != device:
        inspire_names = get_robot_girl_joint_names()
        joint_to_idx = {name: i for i, name in enumerate(joint_names)}
        idx = torch.tensor([joint_to_idx[n] for n in inspire_names], dtype=torch.long, device=device)
        _reward_cache = {"joint_key": joint_key, "device": device, "idx": idx, "idx_batch": None}

    idx = _reward_cache["idx"]
    if _reward_cache.get("idx_batch") is None or _reward_cache["idx_batch"].shape[0] != batch:
        _reward_cache["idx_batch"] = idx.unsqueeze(0).expand(batch, -1)

    return torch.gather(joint_pos, 1, _reward_cache["idx_batch"])


def _get_target(env):
    """Compute the current fist target matching the observation."""
    device = env.device
    batch = env.num_envs

    closed = _FIST_CLOSED.to(device)
    opened = _FIST_OPEN.to(device)

    dt = env.step_dt
    toggle_steps = max(int(2.5 / dt), 1) if dt > 0 else 125
    step = env.episode_length_buf
    phase = (step // toggle_steps) % 2

    mask = (phase == 1).unsqueeze(-1).float()
    return mask * closed.unsqueeze(0) + (1.0 - mask) * opened.unsqueeze(0)


def fist_tracking_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
) -> torch.Tensor:
    """exp(-||pos - target||^2 / std^2): peaks at 1.0 when fingers match target."""
    pos = _get_inspire_pos(env)
    target = _get_target(env)
    sq_err = ((pos - target) ** 2).sum(dim=-1)
    return torch.exp(-sq_err / (std ** 2))


def fist_tracking_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Raw L2 distance from current finger positions to target (for penalty shaping)."""
    pos = _get_inspire_pos(env)
    target = _get_target(env)
    return ((pos - target) ** 2).sum(dim=-1)


__all__ = [
    "fist_tracking_reward",
    "fist_tracking_l2",
]