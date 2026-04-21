#!/usr/bin/env python3
"""Open warehouse scene and hold H1-2 in Pose 1 (bricklaying stance).

Joint order and gains match the real-robot C++ SDK script exactly.

Usage:
    python scene/pose1_scene.py
    python scene/pose1_scene.py --scene /path/to/other.usd
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


# ── Joint name mapping (matches C++ H1JointIndex enum order) ─────────────────
JOINT_NAMES = [
    "left_hip_yaw_joint",         # 0  LeftHipYaw
    "left_hip_pitch_joint",       # 1  LeftHipPitch
    "left_hip_roll_joint",        # 2  LeftHipRoll
    "left_knee_joint",            # 3  LeftKnee
    "left_ankle_pitch_joint",     # 4  LeftAnklePitch
    "left_ankle_roll_joint",      # 5  LeftAnkleRoll
    "right_hip_yaw_joint",        # 6  RightHipYaw
    "right_hip_pitch_joint",      # 7  RightHipPitch
    "right_hip_roll_joint",       # 8  RightHipRoll
    "right_knee_joint",           # 9  RightKnee
    "right_ankle_pitch_joint",    # 10 RightAnklePitch
    "right_ankle_roll_joint",     # 11 RightAnkleRoll
    "torso_joint",                # 12 WaistYaw
    "left_shoulder_pitch_joint",  # 13 LeftShoulderPitch
    "left_shoulder_roll_joint",   # 14 LeftShoulderRoll
    "left_shoulder_yaw_joint",    # 15 LeftShoulderYaw
    "left_elbow_joint",           # 16 LeftElbow
    "left_wrist_roll_joint",      # 17 LeftWristRoll
    "left_wrist_pitch_joint",     # 18 LeftWristPitch
    "left_wrist_yaw_joint",       # 19 LeftWristYaw
    "right_shoulder_pitch_joint", # 20 RightShoulderPitch
    "right_shoulder_roll_joint",  # 21 RightShoulderRoll
    "right_shoulder_yaw_joint",   # 22 RightShoulderYaw
    "right_elbow_joint",          # 23 RightElbow
    "right_wrist_roll_joint",     # 24 RightWristRoll
    "right_wrist_pitch_joint",    # 25 RightWristPitch
    "right_wrist_yaw_joint",      # 26 RightWristYaw
]

# ── Per-joint PD gains (from C++ kKp / kKd arrays) ───────────────────────────
KP = [
    200, 200, 200, 300, 100, 100,           # left leg
    200, 200, 200, 300, 100, 100,           # right leg
    200,                                    # torso
    120, 120,  80,  80,  40,  40,  40,     # left arm
    120, 120,  80,  80,  40,  40,  40,     # right arm
]
KD = [
     5,   5,   5,   8,   3,   3,
     5,   5,   5,   8,   3,   3,
     5,
     3,   3,   2,   2,   1,   1,   1,
     3,   3,   2,   2,   1,   1,   1,
]

# ── Pose 1 — bricklaying / wall-work stance ───────────────────────────────────
POSE1 = [
     0.00,  # left_hip_yaw
    -0.20,  # left_hip_pitch
     0.05,  # left_hip_roll
     0.37,  # left_knee
    -0.18,  # left_ankle_pitch
     0.00,  # left_ankle_roll
     0.00,  # right_hip_yaw
    -0.20,  # right_hip_pitch
    -0.05,  # right_hip_roll
     0.52,  # right_knee
    -0.18,  # right_ankle_pitch
     0.00,  # right_ankle_roll
     0.00,  # torso
     0.30,  # left_shoulder_pitch
     0.15,  # left_shoulder_roll
     0.00,  # left_shoulder_yaw
     0.79,  # left_elbow
     0.00,  # left_wrist_roll
     0.00,  # left_wrist_pitch
     0.00,  # left_wrist_yaw
    -0.10,  # right_shoulder_pitch
    -1.23,  # right_shoulder_roll
     0.00,  # right_shoulder_yaw
     0.55,  # right_elbow
     0.00,  # right_wrist_roll
     0.00,  # right_wrist_pitch
     0.00,  # right_wrist_yaw
]

POSE1_MAP = dict(zip(JOINT_NAMES, POSE1))
KP_MAP    = dict(zip(JOINT_NAMES, KP))
KD_MAP    = dict(zip(JOINT_NAMES, KD))


def _find_articulation_root(stage, search_path: str):
    from pxr import UsdPhysics
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith(search_path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="H1-2 Pose 1 scene viewer.")
    parser.add_argument(
        "--scene",
        type=str,
        default=os.path.join(PROJECT_ROOT, "scene", "warehouse_h1_sc.usd"),
        help="Path to scene USD.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import omni.timeline
    import omni.usd
    from omni.isaac.dynamic_control import _dynamic_control

    scene_path = os.path.abspath(os.path.expanduser(args.scene))
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    usd_ctx = omni.usd.get_context()
    if not usd_ctx.open_stage(scene_path):
        raise RuntimeError(f"Failed to open: {scene_path}")

    simulation_app.update()
    simulation_app.update()

    stage = usd_ctx.get_stage()
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    dc = _dynamic_control.acquire_dynamic_control_interface()

    art_root = _find_articulation_root(stage, "/World/h1_2_with_FTP_hand")
    if art_root is None:
        print("[ERROR] Articulation root not found")
        simulation_app.close()
        return

    art = dc.get_articulation(art_root)
    if art == _dynamic_control.INVALID_HANDLE:
        print(f"[ERROR] Invalid articulation at {art_root}")
        simulation_app.close()
        return

    n = dc.get_articulation_dof_count(art)
    print(f"[INFO] {art_root}  DOFs: {n}")

    # Build name->dof map and apply per-joint gains
    name_to_dof = {}
    for i in range(n):
        dof = dc.get_articulation_dof(art, i)
        name = dc.get_dof_name(dof)
        name_to_dof[name] = dof
        props = dc.get_dof_properties(dof)
        props.stiffness = float(KP_MAP.get(name, 100.0))
        props.damping   = float(KD_MAP.get(name, 5.0))
        dc.set_dof_properties(dof, props)

    # Teleport to pose immediately
    dof_states = dc.get_articulation_dof_states(art, _dynamic_control.STATE_POS)
    for idx, name in enumerate(name_to_dof.keys()):
        if name in POSE1_MAP:
            dof_states["pos"][idx] = POSE1_MAP[name]
    dc.set_articulation_dof_states(art, dof_states, _dynamic_control.STATE_POS)

    # Set position targets so drives hold the pose
    for name, q in POSE1_MAP.items():
        if name in name_to_dof:
            dc.set_dof_position_target(name_to_dof[name], q)

    print("[INFO] Pose 1 (bricklaying stance) applied. Close window to exit.")

    while simulation_app.is_running():
        simulation_app.update()

    omni.timeline.get_timeline_interface().stop()
    usd_ctx.close_stage()
    simulation_app.close()


if __name__ == "__main__":
    main()
