from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from tasks.common_rewards.base_reward_pickplace_cylindercfg import compute_reward
from .curriculum import get_curriculum_stage, STAGE_GRASP, STAGE_HOLD

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

    Always active — provides the directional signal that makes staying upright
    worthwhile.  Uses the configured std directly.
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

    obj_pos   = env.scene[object_cfg.name].data.root_pos_w          # [N, 3]
    wrist_pos = robot.data.body_pos_w[:, indices, :]                 # [N, W, 3]
    dists     = torch.norm(wrist_pos - obj_pos.unsqueeze(1), dim=-1) # [N, W]
    min_dist  = dists.min(dim=1).values                              # [N]
    return torch.exp(-min_dist / std)


_fingertip_cache: dict = {}


def fingertip_grasp_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    proximity_threshold: float = 0.15,
    grasp_std: float = 0.05,
) -> torch.Tensor:
    """Dense grasp reward: fingertips closing around cylinder.

    Active from STAGE_GRASP onward.  Gated by wrist proximity so the policy
    first learns to bring the arm close before squeezing the fingers.
    """
    stage = get_curriculum_stage(env)
    active = stage >= STAGE_GRASP
    if not active.any():
        return torch.zeros(env.num_envs, device=env.device)

    global _fingertip_cache
    robot      = env.scene["robot"]
    body_names = robot.data.body_names
    body_key   = id(body_names)

    if _fingertip_cache.get("body_key") != body_key:
        tip_keywords = ("distal", "intermediate")
        tip_indices   = [
            i for i, name in enumerate(body_names)
            if any(k in name.lower() for k in tip_keywords)
        ]
        wrist_indices = [i for i, name in enumerate(body_names) if "wrist" in name.lower()]
        dev = robot.data.body_pos_w.device
        _fingertip_cache = {
            "body_key":  body_key,
            "tip_idx":   torch.tensor(tip_indices,   dtype=torch.long, device=dev) if tip_indices   else None,
            "wrist_idx": torch.tensor(wrist_indices, dtype=torch.long, device=dev) if wrist_indices else None,
        }

    tip_idx   = _fingertip_cache.get("tip_idx")
    wrist_idx = _fingertip_cache.get("wrist_idx")
    if tip_idx is None or wrist_idx is None:
        return torch.zeros(env.num_envs, device=env.device)

    obj_pos = env.scene[object_cfg.name].data.root_pos_w  # [N, 3]

    # Wrist proximity gate
    wrist_pos  = robot.data.body_pos_w[:, wrist_idx, :]                                  # [N, W, 3]
    wrist_dist = torch.norm(wrist_pos - obj_pos.unsqueeze(1), dim=-1).min(dim=1).values  # [N]
    gate       = (wrist_dist < proximity_threshold).float()                               # [N]

    # Fingertip closeness
    tip_pos       = robot.data.body_pos_w[:, tip_idx, :]               # [N, T, 3]
    tip_dists     = torch.norm(tip_pos - obj_pos.unsqueeze(1), dim=-1) # [N, T]
    mean_tip_dist = tip_dists.mean(dim=1)                              # [N]

    result = gate * torch.exp(-mean_tip_dist / grasp_std)
    return torch.where(active, result, torch.zeros_like(result))


def base_to_object_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 2.0,
) -> torch.Tensor:
    """Dense approach reward: exp(-dist/std) for the robot base to the cylinder.

    Always active — the wide std provides gradient even at the starting distance.
    """
    obj_pos  = env.scene[object_cfg.name].data.root_pos_w[:, :2]  # [N, 2] XY only
    base_pos = env.scene["robot"].data.root_pos_w[:, :2]           # [N, 2]
    dist     = torch.norm(obj_pos - base_pos, dim=-1)              # [N]
    return torch.exp(-dist / std)


def placement_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    enable_dds: bool = True,
    dense_xy_weight: float = 0.0,
    dense_z_weight: float = 0.0,
    dense_xy_scale: float = 4.0,
    dense_z_scale: float = 10.0,
    target_x: float | None = None,
    target_y: float | None = None,
    target_z: float | None = None,
    min_x: float = -0.42,
    max_x: float = 1.0,
    min_y: float = 0.2,
    max_y: float = 0.7,
    min_height: float = 0.5,
    post_min_x: float = 0.28,
    post_max_x: float = 0.96,
    post_min_y: float = 0.24,
    post_max_y: float = 0.57,
    post_min_height: float = 0.81,
    post_max_height: float = 0.9,
) -> torch.Tensor:
    """Sparse + dense placement reward gated on STAGE_HOLD.

    Only fires once the policy has learned to grasp the cylinder.
    """
    stage  = get_curriculum_stage(env)
    active = stage >= STAGE_HOLD
    if not active.any():
        return torch.zeros(env.num_envs, device=env.device)

    raw = compute_reward(
        env, object_cfg=object_cfg, enable_dds=enable_dds,
        dense_xy_weight=dense_xy_weight, dense_z_weight=dense_z_weight,
        dense_xy_scale=dense_xy_scale, dense_z_scale=dense_z_scale,
        target_x=target_x, target_y=target_y, target_z=target_z,
        min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
        min_height=min_height,
        post_min_x=post_min_x, post_max_x=post_max_x,
        post_min_y=post_min_y, post_max_y=post_max_y,
        post_min_height=post_min_height, post_max_height=post_max_height,
    )
    return torch.where(active, raw, torch.zeros_like(raw))


__all__ = [
    "compute_reward",
    "placement_reward",
    "wrist_to_object_reward",
    "fingertip_grasp_reward",
    "base_to_object_reward",
]
