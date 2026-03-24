from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from tasks.common_rewards.base_reward_pickplace_cylindercfg import compute_reward

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Cache wrist body indices so the search over body_names runs only once.
_wrist_cache: dict = {}


def wrist_to_object_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.15,
) -> torch.Tensor:
    """Dense shaped reward: exp(-dist/std) for the closer of the two wrists to the cylinder.

    This provides a gradient signal that teaches the policy to move its arms
    toward the object before the sparse cylinder-placement reward fires.
    std=0.15 m means the reward reaches 0.5 when the wrist is ~10 cm away.
    """
    global _wrist_cache

    robot = env.scene["robot"]
    body_names = robot.data.body_names
    body_key = id(body_names)

    if _wrist_cache.get("body_key") != body_key:
        wrist_indices = [i for i, name in enumerate(body_names) if "wrist" in name.lower()]
        if not wrist_indices:
            _wrist_cache = {"body_key": body_key, "indices": None}
        else:
            _wrist_cache = {
                "body_key": body_key,
                "indices": torch.tensor(wrist_indices, dtype=torch.long, device=robot.data.body_pos_w.device),
            }

    indices = _wrist_cache.get("indices")
    if indices is None:
        return torch.zeros(env.num_envs, device=env.device)

    obj_pos = env.scene[object_cfg.name].data.root_pos_w          # [N, 3]
    wrist_pos = robot.data.body_pos_w[:, indices, :]               # [N, W, 3]
    dists = torch.norm(wrist_pos - obj_pos.unsqueeze(1), dim=-1)   # [N, W]
    min_dist = dists.min(dim=1).values                             # [N]
    return torch.exp(-min_dist / std)


def base_to_object_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 2.0,
) -> torch.Tensor:
    """Dense approach reward: exp(-dist/std) for the robot base to the cylinder.

    Provides a gradient signal that teaches the policy to walk toward the object.
    Uses a wide std so the reward is meaningful even at the starting distance (~1.3 m).
    """
    obj_pos = env.scene[object_cfg.name].data.root_pos_w[:, :2]   # [N, 2] XY only
    base_pos = env.scene["robot"].data.root_pos_w[:, :2]           # [N, 2]
    dist = torch.norm(obj_pos - base_pos, dim=-1)                  # [N]
    return torch.exp(-dist / std)


__all__ = [
    "compute_reward",
    "wrist_to_object_reward",
    "base_to_object_reward",
]

