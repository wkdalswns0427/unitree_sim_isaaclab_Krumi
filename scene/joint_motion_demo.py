#!/usr/bin/env python3

"""H1/H1-2 joint motion controller — attaches to an already-open scene.

No simulator is launched. Run this inside an existing Isaac Sim session:

  Option 1 — Script Editor (Window > Script Editor in Isaac Sim):
      exec(open('/path/to/scene/joint_motion_demo.py').read())

  Option 2 — import from another script that already has AppLauncher running:
      from scene.joint_motion_demo import run_controller
      run_controller(torso_amp=0.0)   # static stance
      run_controller()                 # with torso sway
"""

import math
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _set_attr(prim, name: str, value) -> None:
    attr = prim.GetAttribute(name)
    if attr.IsValid():
        attr.Set(value)


def _set_joint_target(
    stage, joint_path: str, target: float, kp: float | None = None, kd: float | None = None
) -> None:
    prim = stage.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        return
    _set_attr(prim, "drive:angular:physics:targetPosition", float(target))
    if kp is not None:
        _set_attr(prim, "drive:angular:physics:stiffness", float(kp))
    if kd is not None:
        _set_attr(prim, "drive:angular:physics:damping", float(kd))


def _disable_gravity_under(stage, root_path: str) -> int:
    from pxr import UsdPhysics

    count = 0
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(root_path):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateDisableGravityAttr(True)
                count += 1
    return count


def run_controller(
    rate: float = 120.0,
    duration: float = 0.0,
    torso_amp: float = 0.20,
    torso_freq: float = 0.35,
    keep_upright: bool = True,
) -> None:
    """Run joint motion controller in the currently open stage.

    Assumes a scene is already loaded and the simulation timeline is (or will be) playing.
    Call from inside an Isaac Sim session — Script Editor or embedded in another launcher script.
    """
    import omni.kit.app
    import omni.timeline
    import omni.usd

    usd_ctx = omni.usd.get_context()
    stage = usd_ctx.get_stage()
    if stage is None:
        print("[ERROR] No stage loaded. Open a scene first, then run the controller.")
        return

    # Robot A: Isaac H1 in /World
    h1_base = "/World/h1/Joints"
    h1_targets = {
        "left_hip_pitch": -0.28,
        "right_hip_pitch": -0.28,
        "left_knee":        0.79,
        "right_knee":       0.79,
        "left_ankle":      -0.52,
        "right_ankle":     -0.52,
    }
    h1_torso = f"{h1_base}/torso"

    # Robot B: H1-2 with FTP hand
    h12_base = "/h1_2_with_FTP_hand/joints"
    h12_targets = {
        "left_hip_pitch_joint":  -0.20,
        "right_hip_pitch_joint": -0.20,
        "left_knee_joint":        0.42,
        "right_knee_joint":       0.42,
        "left_ankle_pitch_joint": -0.23,
        "right_ankle_pitch_joint":-0.23,
    }
    h12_torso = f"{h12_base}/torso_joint"

    if keep_upright:
        h1_count  = _disable_gravity_under(stage, "/World/h1")
        h12_count = _disable_gravity_under(stage, "/h1_2_with_FTP_hand")
        print(f"[INFO] Disabled gravity: /World/h1={h1_count}, /h1_2_with_FTP_hand={h12_count}")

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    app = omni.kit.app.get_app()
    dt  = 1.0 / max(rate, 1e-3)
    t0  = time.time()
    print("[INFO] Joint motion controller running. Ctrl+C to stop.")

    try:
        while True:
            now = time.time()
            t   = now - t0
            if duration > 0.0 and t >= duration:
                break

            torso_offset = torso_amp * math.sin(2.0 * math.pi * torso_freq * t)

            for joint_name, q in h1_targets.items():
                _set_joint_target(stage, f"{h1_base}/{joint_name}", q, kp=140.0, kd=10.0)
            _set_joint_target(stage, h1_torso, torso_offset, kp=120.0, kd=10.0)

            for joint_name, q in h12_targets.items():
                _set_joint_target(stage, f"{h12_base}/{joint_name}", q, kp=140.0, kd=10.0)
            _set_joint_target(stage, h12_torso, torso_offset, kp=120.0, kd=10.0)

            app.update()

            sleep_time = dt - (time.time() - now)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("[INFO] Controller stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="H1/H1-2 joint motion controller (attach to running scene).")
    parser.add_argument("--rate",       type=float, default=120.0, help="Update rate (Hz).")
    parser.add_argument("--duration",   type=float, default=0.0,   help="Run duration in seconds (0=forever).")
    parser.add_argument("--torso_amp",  type=float, default=0.20,  help="Torso oscillation amplitude (rad).")
    parser.add_argument("--torso_freq", type=float, default=0.35,  help="Torso oscillation frequency (Hz).")
    parser.add_argument("--no-keep-upright", action="store_true",  help="Allow gravity on robot bodies.")
    args = parser.parse_args()

    run_controller(
        rate=args.rate,
        duration=args.duration,
        torso_amp=args.torso_amp,
        torso_freq=args.torso_freq,
        keep_upright=not args.no_keep_upright,
    )
