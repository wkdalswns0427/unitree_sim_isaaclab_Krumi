#!/usr/bin/env python3
"""
H1-2 all-joint keyboard controller via DDS low-level commands.

Lets you select any joint on the H1-2 + Inspire hand and nudge it with
keyboard keys.  Useful for verifying USD joint ranges and actuator behaviour.

Usage:
  python3 nontask_control/h12_joint_keyboard.py --backend stdin --channel 1

Controls:
  UP / W      : increase selected joint position
  DOWN / S    : decrease selected joint position
  LEFT / A    : select previous joint
  RIGHT / D   : select next joint
  [ / ]       : select previous / next joint group
  R           : reset selected joint to default
  0 (zero)    : reset ALL joints to default
  +/=  / -    : increase / decrease step size
  SPACE       : emergency stop (zero velocity, hold position)
  Q           : quit
"""

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import OrderedDict

# Disable iceoryx SHM transport — prevents dds_write.c assertion when ROS
# iceoryx is installed. ChannelFactory.Init() passes config directly to
# Domain(id, config), so CYCLONEDDS_URI env var is ignored; patch the SDK
# config strings directly before any DDS domain is created.
try:
    import unitree_sdk2py.core.channel as _ch
    _shm_off = "<SharedMemory><Enable>false</Enable></SharedMemory>"
    for _attr in ("ChannelConfigAutoDetermine", "ChannelConfigHasInterface"):
        _cfg = getattr(_ch, _attr, None)
        if _cfg and _shm_off not in _cfg:
            setattr(_ch, _attr, _cfg.replace("</Domain>", f"{_shm_off}</Domain>"))
    del _ch, _shm_off, _attr, _cfg
except Exception:
    pass

import numpy as np

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HGLowCmd
    DDS_AVAILABLE = True
except ImportError:
    DDS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Joint definitions (matching H1-2 27-DOF + Inspire hand)
# ---------------------------------------------------------------------------

JOINT_GROUPS = OrderedDict([
    ("Left Leg", [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ]),
    ("Right Leg", [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]),
    ("Left Arm", [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]),
    ("Right Arm", [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]),
    ("Left Hand (Inspire)", [
        "L_index_proximal_joint",
        "L_middle_proximal_joint",
        "L_ring_proximal_joint",
        "L_pinky_proximal_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_proximal_yaw_joint",
    ]),
    ("Right Hand (Inspire)", [
        "R_index_proximal_joint",
        "R_middle_proximal_joint",
        "R_ring_proximal_joint",
        "R_pinky_proximal_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_proximal_yaw_joint",
    ]),
    ("Torso", [
        "torso_joint",
    ]),
])

ALL_JOINTS = []
JOINT_TO_GROUP = {}
GROUP_START_IDX = {}
idx = 0
for grp, joints in JOINT_GROUPS.items():
    GROUP_START_IDX[grp] = idx
    for j in joints:
        ALL_JOINTS.append(j)
        JOINT_TO_GROUP[j] = grp
        idx += 1

GROUP_NAMES = list(JOINT_GROUPS.keys())

# Default positions (standing pose from H12_CFG_WITH_INSPIRE_HAND)
DEFAULT_POS = {
    "left_hip_pitch_joint": -0.05,
    "left_knee_joint": 0.2,
    "left_ankle_pitch_joint": -0.15,
    "right_hip_pitch_joint": -0.05,
    "right_knee_joint": 0.2,
    "right_ankle_pitch_joint": -0.15,
}


class JointKeyboardController:
    def __init__(self, backend: str = "stdin"):
        self.running = True
        self.lock = threading.Lock()

        self.joint_positions = {j: DEFAULT_POS.get(j, 0.0) for j in ALL_JOINTS}
        self.selected_idx = 0
        self.step_size = 0.05  # radians per key press

        self._stdin_old_settings = None
        self._start_stdin_listener()
        self._print_status()

    def _start_stdin_listener(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Requires an interactive terminal (TTY)")
        fd = sys.stdin.fileno()
        self._stdin_fd = fd
        self._stdin_old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._input_thread = threading.Thread(target=self._stdin_loop, daemon=True)
        self._input_thread.start()

    def _stdin_loop(self):
        while self.running:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if not ch:
                continue

            # Handle escape sequences (arrow keys)
            if ch == "\x1b":
                seq = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0.05)[0] else ""
                if seq == "[A":  # Up
                    ch = "w"
                elif seq == "[B":  # Down
                    ch = "s"
                elif seq == "[D":  # Left
                    ch = "a"
                elif seq == "[C":  # Right
                    ch = "d"
                else:
                    continue

            with self.lock:
                self._handle_key(ch)

    def _handle_key(self, ch):
        if ch in ("w", "W"):
            self._nudge(+1)
        elif ch in ("s", "S"):
            self._nudge(-1)
        elif ch in ("a", "A"):
            self.selected_idx = (self.selected_idx - 1) % len(ALL_JOINTS)
            self._print_status()
        elif ch in ("d", "D"):
            self.selected_idx = (self.selected_idx + 1) % len(ALL_JOINTS)
            self._print_status()
        elif ch == "[":
            self._prev_group()
        elif ch == "]":
            self._next_group()
        elif ch in ("r", "R"):
            jn = ALL_JOINTS[self.selected_idx]
            self.joint_positions[jn] = DEFAULT_POS.get(jn, 0.0)
            self._print_status()
        elif ch == "0":
            for jn in ALL_JOINTS:
                self.joint_positions[jn] = DEFAULT_POS.get(jn, 0.0)
            self._print_status()
        elif ch in ("+", "="):
            self.step_size = min(self.step_size + 0.01, 0.5)
            self._print_status()
        elif ch == "-":
            self.step_size = max(self.step_size - 0.01, 0.01)
            self._print_status()
        elif ch == " ":
            # Emergency: zero all velocities (keep current positions)
            self._print_status()
            print("  >>> HOLD POSITION <<<")
        elif ch in ("q", "Q"):
            self.running = False

    def _nudge(self, direction: int):
        jn = ALL_JOINTS[self.selected_idx]
        self.joint_positions[jn] += direction * self.step_size
        self.joint_positions[jn] = round(self.joint_positions[jn], 4)
        self._print_status()

    def _prev_group(self):
        cur_group = JOINT_TO_GROUP[ALL_JOINTS[self.selected_idx]]
        gi = GROUP_NAMES.index(cur_group)
        gi = (gi - 1) % len(GROUP_NAMES)
        self.selected_idx = GROUP_START_IDX[GROUP_NAMES[gi]]
        self._print_status()

    def _next_group(self):
        cur_group = JOINT_TO_GROUP[ALL_JOINTS[self.selected_idx]]
        gi = GROUP_NAMES.index(cur_group)
        gi = (gi + 1) % len(GROUP_NAMES)
        self.selected_idx = GROUP_START_IDX[GROUP_NAMES[gi]]
        self._print_status()

    def _print_status(self):
        jn = ALL_JOINTS[self.selected_idx]
        grp = JOINT_TO_GROUP[jn]
        val = self.joint_positions[jn]
        # Clear line and print
        sys.stdout.write(f"\r\033[K[{grp}] {jn} = {val:+.4f}  (step={self.step_size:.3f})  ")
        sys.stdout.flush()

    def get_positions(self) -> dict:
        with self.lock:
            return self.joint_positions.copy()

    def stop(self):
        self.running = False
        if self._stdin_old_settings is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_settings)
            except Exception:
                pass


def _build_lowcmd(positions: dict) -> "HGLowCmd":
    """Build a HG LowCmd message from the joint position dict."""
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    cmd = unitree_hg_msg_dds__LowCmd_()
    for i, jn in enumerate(ALL_JOINTS):
        if i < len(cmd.motor_cmd):
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].q = float(positions.get(jn, 0.0))
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = 60.0
            cmd.motor_cmd[i].kd = 5.0
    cmd.crc = CRC().Crc(cmd)
    return cmd


def main():
    parser = argparse.ArgumentParser(description="H1-2 all-joint keyboard controller")
    parser.add_argument("--backend", choices=["stdin"], default="stdin")
    parser.add_argument("--channel", type=int, default=1, help="DDS channel id")
    parser.add_argument("--publish_rate", type=float, default=50.0, help="Command publish rate (Hz)")
    parser.add_argument("--no_dds", action="store_true", help="Dry-run: print commands without DDS")
    args = parser.parse_args()

    print("=" * 70)
    print("H1-2 ALL-JOINT KEYBOARD CONTROLLER")
    print("=" * 70)
    print("W/S or UP/DOWN   : increase / decrease joint position")
    print("A/D or LEFT/RIGHT: select previous / next joint")
    print("[  /  ]          : jump to previous / next joint group")
    print("R                : reset selected joint to default")
    print("0                : reset ALL joints to default")
    print("+  /  -          : increase / decrease step size")
    print("SPACE            : hold position")
    print("Q                : quit")
    print("=" * 70)
    print()
    print("Joint groups:")
    for grp, joints in JOINT_GROUPS.items():
        print(f"  {grp}: {', '.join(joints)}")
    print()

    pub = None
    if not args.no_dds:
        if not DDS_AVAILABLE:
            print("[WARN] unitree_sdk2py not available, running in dry-run mode")
        else:
            ChannelFactoryInitialize(args.channel)
            pub = ChannelPublisher("rt/lowcmd", HGLowCmd)
            pub.Init()
            print("[joint-kbd] DDS publisher initialized on rt/lowcmd")

    controller = JointKeyboardController(backend=args.backend)
    interval = 1.0 / args.publish_rate

    try:
        while controller.running:
            time.sleep(interval)
            positions = controller.get_positions()
            if pub is not None:
                cmd = _build_lowcmd(positions)
                pub.Write(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        # Send zero-velocity hold on exit
        if pub is not None:
            positions = controller.get_positions()
            cmd = _build_lowcmd(positions)
            pub.Write(cmd)
        print("\n[joint-kbd] stopped")


if __name__ == "__main__":
    main()