#!/usr/bin/env python3

"""Launch Isaac Sim and open a USD scene."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a USD scene in Isaac Sim.")
    parser.add_argument(
        "--scene",
        type=str,
        default=os.path.join(PROJECT_ROOT, "scene", "warehouse_h1_sc.usd"),
        help="Path or URI to a USD file.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = _build_parser()
    args_cli = parser.parse_args()

    # Keep GUI enabled by default for scene viewing.
    if "--headless" not in sys.argv:
        args_cli.headless = False

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    usd_ctx = None
    try:
        import omni.usd

        scene_path = os.path.abspath(os.path.expanduser(args_cli.scene))
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene USD not found: {scene_path}")

        import omni.timeline

        usd_ctx = omni.usd.get_context()
        ok = usd_ctx.open_stage(scene_path)
        if not ok:
            raise RuntimeError(f"Failed to open scene: {scene_path}")

        stage = usd_ctx.get_stage()
        omni.timeline.get_timeline_interface().play()
        simulation_app.update()  # one frame so PhysX creates the articulation

        from omni.isaac.dynamic_control import _dynamic_control
        from pxr import UsdPhysics

        dc = _dynamic_control.acquire_dynamic_control_interface()

        # Find articulation root
        art_root = None
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith("/World/h1_2_with_FTP_hand") and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                art_root = str(prim.GetPath())
                break

        h12_stance = {
            "left_hip_pitch_joint":    -0.28,
            "right_hip_pitch_joint":   -0.28,
            "left_hip_roll_joint":      0.0,
            "right_hip_roll_joint":     0.0,
            "left_hip_yaw_joint":       0.0,
            "right_hip_yaw_joint":      0.0,
            "left_knee_joint":          0.70,
            "right_knee_joint":         0.70,
            "left_ankle_pitch_joint":  -0.42,
            "right_ankle_pitch_joint": -0.42,
            "left_ankle_roll_joint":    0.0,
            "right_ankle_roll_joint":   0.0,
            "torso_joint":              0.0,
            "left_shoulder_pitch_joint":  0.30,
            "right_shoulder_pitch_joint": 0.30,
            "left_shoulder_roll_joint":   0.20,
            "right_shoulder_roll_joint": -0.20,
            "left_elbow_joint":           0.50,
            "right_elbow_joint":          0.50,
        }

        if art_root:
            art = dc.get_articulation(art_root)
            if art != _dynamic_control.INVALID_HANDLE:
                n = dc.get_articulation_dof_count(art)
                name_to_dof = {}
                for i in range(n):
                    dof = dc.get_articulation_dof(art, i)
                    name = dc.get_dof_name(dof)
                    name_to_dof[name] = dof
                    props = dc.get_dof_properties(dof)
                    props.stiffness = 300.0
                    props.damping = 20.0
                    dc.set_dof_properties(dof, props)

                # Teleport to stance
                dof_states = dc.get_articulation_dof_states(art, _dynamic_control.STATE_POS)
                for idx, name in enumerate(name_to_dof.keys()):
                    if name in h12_stance:
                        dof_states["pos"][idx] = h12_stance[name]
                dc.set_articulation_dof_states(art, dof_states, _dynamic_control.STATE_POS)

                for name, q in h12_stance.items():
                    if name in name_to_dof:
                        dc.set_dof_position_target(name_to_dof[name], q)
                print(f"[INFO] Stance applied to {art_root}")

        print(f"[INFO] Opened scene: {scene_path}")
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        if usd_ctx is not None:
            usd_ctx.close_stage()
        simulation_app.close()


if __name__ == "__main__":
    main()
