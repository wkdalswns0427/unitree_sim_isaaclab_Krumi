# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Two-stage curriculum for the H1-2 cylinder task.

Balance and approach rewards are always active — the robot needs the
approach signal (walk toward cylinder, extend arm) to give it a *reason*
to stay upright.  The curriculum only gates the harder sub-tasks:

Stages (per-environment, independently tracked):
  0 - APPROACH : Balance + walk toward cylinder + reach with arm.
                 All stability and approach rewards are active.
  1 - GRASP    : + fingertip grasp reward activates.
  2 - HOLD     : + placement reward activates (full task).

Stage progress is preserved across episode resets.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ── Stage identifiers ────────────────────────────────────────────────────────
STAGE_APPROACH = 0  # Balance + approach (always active from the start)
STAGE_GRASP = 1     # + fingertip grasp
STAGE_HOLD = 2      # + placement (full task)

# Consecutive successful policy steps before advancing.
# Policy dt = sim_dt * decimation = 0.005 * 4 = 0.02 s.
_APPROACH_STEPS = 50   # ~1 s with wrist near cylinder
_GRASP_STEPS    = 50   # ~1 s with fingertips near cylinder

# Distance thresholds (metres)
_APPROACH_WRIST_DIST = 0.35   # wrist within 35 cm → approach complete
_GRASP_TIP_DIST      = 0.20   # mean fingertip within 20 cm → grasp complete

# ── Body-index cache ─────────────────────────────────────────────────────────
_body_cache: dict = {}


def _get_body_indices(env: "ManagerBasedRLEnv"):
    """Return cached (wrist_idx, tip_idx) tensors for the current robot."""
    global _body_cache
    robot = env.scene["robot"]
    body_names = robot.data.body_names
    key = id(body_names)
    if _body_cache.get("key") != key:
        dev = robot.data.body_pos_w.device
        wrist_idx = [i for i, n in enumerate(body_names) if "wrist" in n.lower()]
        tip_kw = ("distal", "intermediate")
        tip_idx = [
            i for i, n in enumerate(body_names)
            if any(k in n.lower() for k in tip_kw)
        ]
        _body_cache = {
            "key": key,
            "wrist": torch.tensor(wrist_idx, dtype=torch.long, device=dev) if wrist_idx else None,
            "tips":  torch.tensor(tip_idx,   dtype=torch.long, device=dev) if tip_idx  else None,
        }
    return _body_cache["wrist"], _body_cache["tips"]


# ── Public API ───────────────────────────────────────────────────────────────

def get_curriculum_stage(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return per-environment curriculum stage (long tensor, shape [N])."""
    if not hasattr(env, "_curriculum_stage"):
        env._curriculum_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._curriculum_success_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env._curriculum_stage


def advance_curriculum_stage(env: "ManagerBasedRLEnv", env_ids) -> dict:
    """Check advancement conditions each step and promote stages.

    Called by :class:`~isaaclab.managers.CurriculumManager` every step.
    Returns a logging dict with mean stage and per-stage fractions.
    """
    stage   = get_curriculum_stage(env)   # initialises tensors on first call
    success = env._curriculum_success_steps

    # Reset success counter for environments that just finished an episode.
    if isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
        success[env_ids] = 0

    robot   = env.scene["robot"]
    obj_pos = env.scene["object"].data.root_pos_w   # [N, 3]
    wrist_idx, tip_idx = _get_body_indices(env)

    # ── Stage 0 → 1: wrist close to cylinder for N steps ───────────────────
    s0 = stage == STAGE_APPROACH
    if s0.any() and wrist_idx is not None:
        wrist_pos  = robot.data.body_pos_w[:, wrist_idx, :]          # [N, W, 3]
        wrist_dist = torch.norm(
            wrist_pos - obj_pos.unsqueeze(1), dim=-1
        ).min(dim=1).values                                           # [N]
        near = wrist_dist < _APPROACH_WRIST_DIST
        success[s0 &  near] += 1
        success[s0 & ~near]  = 0
        promote = s0 & (success >= _APPROACH_STEPS)
        stage[promote]   = STAGE_GRASP
        success[promote] = 0

    # ── Stage 1 → 2: mean fingertip near cylinder for N steps ──────────────
    s1 = stage == STAGE_GRASP
    if s1.any() and tip_idx is not None:
        tip_pos       = robot.data.body_pos_w[:, tip_idx, :]          # [N, T, 3]
        mean_tip_dist = torch.norm(
            tip_pos - obj_pos.unsqueeze(1), dim=-1
        ).mean(dim=1)                                                  # [N]
        grasped = mean_tip_dist < _GRASP_TIP_DIST
        success[s1 &  grasped] += 1
        success[s1 & ~grasped]  = 0
        promote = s1 & (success >= _GRASP_STEPS)
        stage[promote]   = STAGE_HOLD
        success[promote] = 0

    # ── Logging ──────────────────────────────────────────────────────────────
    counts = torch.bincount(stage, minlength=3).float()
    total  = float(env.num_envs)
    return {
        "mean_stage":   stage.float().mean().item(),
        "pct_approach": (counts[0] / total).item(),
        "pct_grasp":    (counts[1] / total).item(),
        "pct_hold":     (counts[2] / total).item(),
    }


__all__ = [
    "STAGE_APPROACH",
    "STAGE_GRASP",
    "STAGE_HOLD",
    "get_curriculum_stage",
    "advance_curriculum_stage",
]
