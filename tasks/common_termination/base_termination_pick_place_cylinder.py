# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_object_estimate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_x: float = -0.42,                # minimum x position threshold
    max_x: float = 1.0,                # maximum x position threshold
    min_y: float = 0.2,                # minimum y position threshold
    max_y: float = 0.7,                # maximum y position threshold
    min_height: float = 0.5,
) -> torch.Tensor:
   # when the object is not in the set return, reset
    # Get object entity from the scene
    # 1. get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]
    
    # 2. get object position in world coordinates
    wheel_x = object.data.root_pos_w[:, 0]         # x position
    wheel_y = object.data.root_pos_w[:, 1]        # y position
    wheel_height = object.data.root_pos_w[:, 2]   # z position (height)

    # Convert per-env bounds to world frame using env origin offsets.
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        min_x_w = min_x + env_origins[:, 0]
        max_x_w = max_x + env_origins[:, 0]
        min_y_w = min_y + env_origins[:, 1]
        max_y_w = max_y + env_origins[:, 1]
    else:
        min_x_w = torch.full_like(wheel_x, min_x)
        max_x_w = torch.full_like(wheel_x, max_x)
        min_y_w = torch.full_like(wheel_y, min_y)
        max_y_w = torch.full_like(wheel_y, max_y)

    done_x = (wheel_x < max_x_w) & (wheel_x > min_x_w)
    done_y = (wheel_y < max_y_w) & (wheel_y > min_y_w)
    done_height = wheel_height > min_height
    done = done_x & done_y & done_height

    return ~done
