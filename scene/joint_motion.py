#!/usr/bin/env python3
"""H1-2 joint motion demo — opens scene USD and drives joints via dynamic_control.

Usage:
    python scene/joint_motion_demo.py
    python scene/joint_motion_demo.py --scene /path/to/scene.usd --torso_amp 0.0
"""

import argparse
import math
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


# kReady from real-robot C++ SDK — confirmed balanced on hardware
# Balance condition: hip_pitch + ankle_pitch + knee ≈ 0  (-0.4 + 0.8 - 0.4 = 0)
H12_STANCE = {
    "left_hip_yaw_joint":          0.0,
    "left_hip_pitch_joint":       -0.4,
    "left_hip_roll_joint":         0.0,
    "left_knee_joint":             0.8,
    "left_ankle_pitch_joint":     -0.23,
    "left_ankle_roll_joint":       0.0,
    "right_hip_yaw_joint":         0.0,
    "right_hip_pitch_joint":      -0.4,
    "right_hip_roll_joint":        0.0,
    "right_knee_joint":            0.8,
    "right_ankle_pitch_joint":    -0.23,
    "right_ankle_roll_joint":      0.0,
    "torso_joint":                 0.0,
    "left_shoulder_pitch_joint":  -0.3,
    "left_shoulder_roll_joint":    0.2,
    "left_shoulder_yaw_joint":     0.0,
    "left_elbow_joint":            0.3,
    "left_wrist_roll_joint":       0.0,
    "left_wrist_pitch_joint":      0.0,
    "left_wrist_yaw_joint":        0.0,
    "right_shoulder_pitch_joint": -0.3,
    "right_shoulder_roll_joint":  -0.2,
    "right_shoulder_yaw_joint":    0.0,
    "right_elbow_joint":           0.3,
    "right_wrist_roll_joint":      0.0,
    "right_wrist_pitch_joint":     0.0,
    "right_wrist_yaw_joint":       0.0,
}

# Per-joint PD gains matching C++ kKp/kKd (order: left leg, right leg, torso, left arm, right arm)
_KP = {
    "left_hip_yaw_joint": 200,   "left_hip_pitch_joint": 200,  "left_hip_roll_joint": 200,
    "left_knee_joint": 300,      "left_ankle_pitch_joint": 400, "left_ankle_roll_joint": 200,
    "right_hip_yaw_joint": 200,  "right_hip_pitch_joint": 200, "right_hip_roll_joint": 200,
    "right_knee_joint": 300,     "right_ankle_pitch_joint": 400,"right_ankle_roll_joint": 200,
    "torso_joint": 200,
    "left_shoulder_pitch_joint": 120, "left_shoulder_roll_joint": 120,
    "left_shoulder_yaw_joint": 80,    "left_elbow_joint": 80,
    "left_wrist_roll_joint": 40,      "left_wrist_pitch_joint": 40, "left_wrist_yaw_joint": 40,
    "right_shoulder_pitch_joint": 120,"right_shoulder_roll_joint": 120,
    "right_shoulder_yaw_joint": 80,   "right_elbow_joint": 80,
    "right_wrist_roll_joint": 40,     "right_wrist_pitch_joint": 40,"right_wrist_yaw_joint": 40,
}
_KD = {
    "left_hip_yaw_joint": 5,    "left_hip_pitch_joint": 5,   "left_hip_roll_joint": 5,
    "left_knee_joint": 8,       "left_ankle_pitch_joint": 10, "left_ankle_roll_joint": 5,
    "right_hip_yaw_joint": 5,   "right_hip_pitch_joint": 5,  "right_hip_roll_joint": 5,
    "right_knee_joint": 8,      "right_ankle_pitch_joint": 10,"right_ankle_roll_joint": 5,
    "torso_joint": 5,
    "left_shoulder_pitch_joint": 3, "left_shoulder_roll_joint": 3,
    "left_shoulder_yaw_joint": 2,   "left_elbow_joint": 2,
    "left_wrist_roll_joint": 1,     "left_wrist_pitch_joint": 1, "left_wrist_yaw_joint": 1,
    "right_shoulder_pitch_joint": 3,"right_shoulder_roll_joint": 3,
    "right_shoulder_yaw_joint": 2,  "right_elbow_joint": 2,
    "right_wrist_roll_joint": 1,    "right_wrist_pitch_joint": 1,"right_wrist_yaw_joint": 1,
}


def _find_articulation_root(stage, search_path: str) -> str | None:
    from pxr import UsdPhysics
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith(search_path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return p
    return None


def _setup_drives(dc, art) -> dict:
    """Apply per-joint PD gains, return name->dof_handle mapping."""
    n = dc.get_articulation_dof_count(art)
    name_to_dof = {}
    for i in range(n):
        dof = dc.get_articulation_dof(art, i)
        name = dc.get_dof_name(dof)
        name_to_dof[name] = dof
        props = dc.get_dof_properties(dof)
        props.stiffness = float(_KP.get(name, 100.0))
        props.damping   = float(_KD.get(name, 5.0))
        dc.set_dof_properties(dof, props)
    return name_to_dof


def main() -> None:
    parser = argparse.ArgumentParser(description="H1-2 joint motion demo.")
    parser.add_argument(
        "--scene",
        type=str,
        default=os.path.join(PROJECT_ROOT, "scene", "warehouse_h1_sc.usd"),
        help="Path to scene USD.",
    )
    parser.add_argument("--rate",       type=float, default=120.0, help="Update rate (Hz).")
    parser.add_argument("--duration",   type=float, default=0.0,   help="Run duration seconds (0=forever).")
    parser.add_argument("--torso_amp",  type=float, default=0.20,  help="Torso oscillation amplitude (rad).")
    parser.add_argument("--torso_freq", type=float, default=0.35,  help="Torso oscillation frequency (Hz).")
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

    # Write stance + per-joint drive gains to USD BEFORE play so PhysX picks them up at startup.
    h12_joints_base = "/World/h1_2_with_FTP_hand/joints"
    for joint_name, q in H12_STANCE.items():
        prim = stage.GetPrimAtPath(f"{h12_joints_base}/{joint_name}")
        if prim.IsValid():
            for attr_name, val in [
                ("drive:angular:physics:targetPosition", float(q)),
                ("drive:angular:physics:stiffness",      float(_KP.get(joint_name, 100.0))),
                ("drive:angular:physics:damping",        float(_KD.get(joint_name, 5.0))),
            ]:
                attr = prim.GetAttribute(attr_name)
                if attr.IsValid():
                    attr.Set(val)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()  # one frame to let PhysX create the articulation

    dc = _dynamic_control.acquire_dynamic_control_interface()

    art_root = _find_articulation_root(stage, "/World/h1_2_with_FTP_hand")
    if art_root is None:
        print("[ERROR] ArticulationRootAPI not found under /World/h1_2_with_FTP_hand")
        simulation_app.close()
        return

    art = dc.get_articulation(art_root)
    if art == _dynamic_control.INVALID_HANDLE:
        print(f"[ERROR] Invalid articulation handle at {art_root}")
        simulation_app.close()
        return

    n_dofs = dc.get_articulation_dof_count(art)
    print(f"[INFO] Articulation: {art_root}  DOFs: {n_dofs}")

    name_to_dof = _setup_drives(dc, art)

    # Teleport joints to stance before gravity has time to act.
    dof_states = dc.get_articulation_dof_states(art, _dynamic_control.STATE_POS)
    for idx, name in enumerate(name_to_dof.keys()):
        if name in H12_STANCE:
            dof_states["pos"][idx] = H12_STANCE[name]
    dc.set_articulation_dof_states(art, dof_states, _dynamic_control.STATE_POS)

    # Apply position targets immediately
    for name, q in H12_STANCE.items():
        if name in name_to_dof:
            dc.set_dof_position_target(name_to_dof[name], q)

    dt = 1.0 / max(args.rate, 1e-3)
    t0 = time.time()
    print("[INFO] Running. Close window or Ctrl+C to stop.")

    try:
        while simulation_app.is_running():
            now = time.time()
            t = now - t0
            if args.duration > 0.0 and t >= args.duration:
                break

            torso_q = args.torso_amp * math.sin(2.0 * math.pi * args.torso_freq * t)
            targets = {**H12_STANCE, "torso_joint": torso_q}

            for name, q in targets.items():
                if name in name_to_dof:
                    dc.set_dof_position_target(name_to_dof[name], q)

            simulation_app.update()

            sleep_time = dt - (time.time() - now)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("[INFO] Stopped.")
    finally:
        timeline.stop()
        usd_ctx.close_stage()
        simulation_app.close()


if __name__ == "__main__":
    main()
