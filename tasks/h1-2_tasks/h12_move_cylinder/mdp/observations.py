
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from tasks.common_observations.h12_27dof_state import get_robot_boy_joint_states
from tasks.common_observations.inspire_state import (
    get_robot_inspire_joint_states,
    get_robot_girl_joint_names,
)
from tasks.common_observations.camera_state import get_camera_image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Per-environment index cache so joint lookup is done only once.
_inspire_vel_cache: dict = {}


def get_robot_inspire_joint_vel(
    env: ManagerBasedRLEnv,
    enable_dds: bool = False,
) -> torch.Tensor:
    """Return Inspire proximal joint velocities (12 dims).

    Provides closed-loop velocity feedback so the policy can learn to control
    finger speed, not just target position.
    """
    global _inspire_vel_cache

    joint_vel = env.scene["robot"].data.joint_vel
    device = joint_vel.device
    batch = joint_vel.shape[0]
    joint_names = env.scene["robot"].data.joint_names
    joint_key = id(joint_names)  # joint list is stable; use id as cheap key

    if _inspire_vel_cache.get("joint_key") != joint_key or _inspire_vel_cache.get("device") != device:
        inspire_names = get_robot_girl_joint_names()
        joint_to_idx = {name: i for i, name in enumerate(joint_names)}
        missing = [n for n in inspire_names if n not in joint_to_idx]
        if missing:
            raise ValueError(f"[inspire_vel_obs] Missing joints: {missing}")
        idx = torch.tensor([joint_to_idx[n] for n in inspire_names], dtype=torch.long, device=device)
        _inspire_vel_cache = {"joint_key": joint_key, "device": device, "idx": idx, "idx_batch": None}

    idx = _inspire_vel_cache["idx"]
    if _inspire_vel_cache.get("idx_batch") is None or _inspire_vel_cache["idx_batch"].shape[0] != batch:
        _inspire_vel_cache["idx_batch"] = idx.unsqueeze(0).expand(batch, -1)

    return torch.gather(joint_vel, 1, _inspire_vel_cache["idx_batch"])


# Per-environment cache for object position computation.
_obj_pos_cache: dict = {}


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of quaternion (w, x, y, z)."""
    return torch.cat([q[:, 0:1], -q[:, 1:4]], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (w, x, y, z)."""
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def get_object_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the cylinder position in the robot body frame (3 dims).

    Properly rotates the world-frame offset by the inverse of the robot base
    orientation so the observation is invariant to robot heading.
    """
    obj_pos_w = env.scene[object_cfg.name].data.root_pos_w      # [N, 3]
    robot_pos_w = env.scene["robot"].data.root_pos_w             # [N, 3]
    robot_quat_w = env.scene["robot"].data.root_quat_w           # [N, 4] (w,x,y,z)
    offset_w = obj_pos_w - robot_pos_w                           # [N, 3]
    return _quat_rotate(_quat_conjugate(robot_quat_w), offset_w) # [N, 3]


__all__ = [
    "get_robot_boy_joint_states",
    "get_robot_inspire_joint_states",
    "get_robot_inspire_joint_vel",
    "get_object_pos_in_robot_frame",
    "get_camera_image",
]
