# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
# global variable to cache the DDS instance
_rewards_dds = None
_dds_retry_interval_s = 1.0
_dds_next_retry_time = 0.0
_dds_cleanup_registered = False
import sys
import os
import time
def _get_rewards_dds_instance():
    """get the DDS instance, delay initialization"""
    global _rewards_dds, _dds_next_retry_time, _dds_cleanup_registered

    if _rewards_dds is not None:
        return _rewards_dds

    now = time.monotonic()
    if now < _dds_next_retry_time:
        return None

    try:
        # dynamically import the DDS module
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
        from dds.dds_master import dds_manager

        _rewards_dds = dds_manager.objects.get("rewards")
        if _rewards_dds is None:
            _dds_next_retry_time = now + _dds_retry_interval_s
            return None

        print("[Observations Rewards] DDS communication instance obtained")

        # register the cleanup function
        if not _dds_cleanup_registered:
            import atexit

            def cleanup_dds():
                try:
                    if _rewards_dds:
                        dds_manager.unregister_object("rewards")
                        print("[rewards_dds] DDS communication closed correctly")
                except Exception as e:
                    print(f"[rewards_dds] Error closing DDS: {e}")

            atexit.register(cleanup_dds)
            _dds_cleanup_registered = True

    except Exception as e:
        print(f"[Observations Rewards] Failed to get DDS instances: {e}")
        _rewards_dds = None
        _dds_next_retry_time = now + _dds_retry_interval_s

    return _rewards_dds
def compute_reward(
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
    min_x: float = -0.42,                # minimum x position threshold
    max_x: float = 1.0,                # maximum x position threshold
    min_y: float = 0.2,                # minimum y position threshold
    max_y: float = 0.7,                # maximum y position threshold
    min_height: float = 0.5,
    post_min_x: float = 0.28,
    post_max_x: float = 0.96,
    post_min_y: float = 0.24,
    post_max_y: float = 0.57,
    post_min_height: float = 0.81,
    post_max_height: float = 0.9,
) -> torch.Tensor:
   # when the object is not in the set return, reset

    interval = getattr(env, "_reward_interval", 1) or 1
    counter = getattr(env, "_reward_counter", 0)
    last = getattr(env, "_reward_last", None)
    if interval > 1 and last is not None and counter % interval != 0:
        env._reward_counter = counter + 1
        return last

    # 1. get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]
    
    # 2. get object position in world coordinates
    wheel_x = object.data.root_pos_w[:, 0]         # x position
    wheel_y = object.data.root_pos_w[:, 1]        # y position
    wheel_height = object.data.root_pos_w[:, 2]   # z position (height)

    # Convert per-env bounds to world frame using env origins.
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        min_x_w = min_x + env_origins[:, 0]
        max_x_w = max_x + env_origins[:, 0]
        min_y_w = min_y + env_origins[:, 1]
        max_y_w = max_y + env_origins[:, 1]

        post_min_x_w = post_min_x + env_origins[:, 0]
        post_max_x_w = post_max_x + env_origins[:, 0]
        post_min_y_w = post_min_y + env_origins[:, 1]
        post_max_y_w = post_max_y + env_origins[:, 1]

        tx = (0.5 * (post_min_x + post_max_x) if target_x is None else target_x) + env_origins[:, 0]
        ty = (0.5 * (post_min_y + post_max_y) if target_y is None else target_y) + env_origins[:, 1]
    else:
        min_x_w = torch.full_like(wheel_x, min_x)
        max_x_w = torch.full_like(wheel_x, max_x)
        min_y_w = torch.full_like(wheel_y, min_y)
        max_y_w = torch.full_like(wheel_y, max_y)

        post_min_x_w = torch.full_like(wheel_x, post_min_x)
        post_max_x_w = torch.full_like(wheel_x, post_max_x)
        post_min_y_w = torch.full_like(wheel_y, post_min_y)
        post_max_y_w = torch.full_like(wheel_y, post_max_y)

        tx = torch.full_like(wheel_x, 0.5 * (post_min_x + post_max_x) if target_x is None else target_x)
        ty = torch.full_like(wheel_y, 0.5 * (post_min_y + post_max_y) if target_y is None else target_y)

    # element-wise operations
    done_x = (wheel_x < max_x_w) & (wheel_x > min_x_w)
    done_y = (wheel_y < max_y_w) & (wheel_y > min_y_w)
    done_height = (wheel_height > min_height)
    done = done_x & done_y & done_height

    # 3. get post position conditions
    done_post_x = (wheel_x < post_max_x_w) & (wheel_x > post_min_x_w)
    done_post_y = (wheel_y < post_max_y_w) & (wheel_y > post_min_y_w)
    done_post_height = (wheel_height > post_min_height) & (wheel_height < post_max_height)
    done_post = done_post_x & done_post_y & done_post_height

    # Create reward tensor for all environments
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    
    # Set rewards based on conditions
    reward[~done] = -1.0  # Not in valid area
    reward[done_post] = 1.0  # In target post area
    reward[done & ~done_post] = 0.0  # In valid area but not target

    # Optional dense shaping around target center for easier learning.
    if dense_xy_weight > 0.0 or dense_z_weight > 0.0:
        tz = 0.5 * (post_min_height + post_max_height) if target_z is None else target_z

        dist_xy = torch.sqrt((wheel_x - tx) ** 2 + (wheel_y - ty) ** 2)
        dist_z = torch.abs(wheel_height - tz)
        dense_reward = (
            dense_xy_weight * torch.exp(-dense_xy_scale * dist_xy)
            + dense_z_weight * torch.exp(-dense_z_scale * dist_z)
        )
        reward = reward + dense_reward

    if enable_dds:
        rewards_dds = _get_rewards_dds_instance()
        if rewards_dds:
            rewards_dds.write_rewards_data(reward)
    env._reward_last = reward
    env._reward_counter = counter + 1
    return reward
