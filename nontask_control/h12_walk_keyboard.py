#!/usr/bin/env python3
"""
H1-2 wholebody keyboard walk command publisher.

Use with:
  python3 sim_main.py --device cuda --enable_cameras \
    --task Isaac-Move-Cylinder-H12-WholeBody \
    --enable_inspire_dds --robot_type h1_2 --wait_for_keyboard_start

Then run this script in another terminal:
  python3 nontask_control/h12_walk_keyboard.py --backend stdin --channel 1
"""

import argparse
import os
import select
import sys

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
import termios
import threading
import time
import tty
from typing import Optional

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

try:
    from pynput import keyboard as pynput_keyboard

    PYNPUT_AVAILABLE = True
    PYNPUT_IMPORT_ERROR = ""
except Exception as exc:
    pynput_keyboard = None
    PYNPUT_AVAILABLE = False
    PYNPUT_IMPORT_ERROR = str(exc)


class KeyboardWalkController:
    def __init__(self, backend: str = "auto"):
        self.backend = self._resolve_backend(backend)
        self.running = True
        self.lock = threading.Lock()

        self.control = {
            "x_vel": 0.0,
            "y_vel": 0.0,
            "yaw_vel": 0.0,
            "height_delta": 0.0,
        }
        self.ranges = {
            "x_vel": (-0.6, 1.0),
            "y_vel": (-0.5, 0.5),
            "yaw_vel": (-1.57, 1.57),
            "height_delta": (-0.3, 0.1),
        }
        self.step = 0.05

        self.key_states = {k: False for k in ("w", "s", "a", "d", "z", "x", "c", "v")}
        self._last_key_event_ts = {k: 0.0 for k in self.key_states}

        self._reset_request: Optional[int] = None
        self._stdin_old_settings = None

        self._loop_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._loop_thread.start()
        self._start_keyboard_listener()

    def _resolve_backend(self, backend: str) -> str:
        if backend not in ("auto", "pynput", "stdin"):
            raise ValueError(f"unsupported backend: {backend}")
        if backend == "pynput":
            if not PYNPUT_AVAILABLE:
                raise RuntimeError(f"pynput not available: {PYNPUT_IMPORT_ERROR}")
            return "pynput"
        if backend == "stdin":
            return "stdin"
        if PYNPUT_AVAILABLE and os.environ.get("DISPLAY"):
            return "pynput"
        return "stdin"

    def _set_key(self, key_char: str, pressed: bool):
        if key_char in self.key_states:
            self.key_states[key_char] = pressed
            if pressed:
                self._last_key_event_ts[key_char] = time.time()

    def _start_keyboard_listener(self):
        if self.backend == "pynput":
            self._start_pynput_listener()
        else:
            self._start_stdin_listener()

    def _start_pynput_listener(self):
        def on_press(key):
            key_char = key.char.lower() if hasattr(key, "char") and key.char else None
            if key_char is None:
                return
            with self.lock:
                if key_char in self.key_states:
                    self._set_key(key_char, True)
                elif key_char == "u":
                    self._reset_request = 1
                elif key_char == "p":
                    self._reset_request = 2
                elif key_char == " ":
                    for k in self.key_states:
                        self._set_key(k, False)
                    for k in self.control:
                        self.control[k] = 0.0
                elif key_char == "q":
                    self.running = False
                    return False

        def on_release(key):
            key_char = key.char.lower() if hasattr(key, "char") and key.char else None
            if key_char is None:
                return
            with self.lock:
                if key_char in self.key_states:
                    self._set_key(key_char, False)

        self.listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()
        print("[walk-kbd] using pynput backend")

    def _start_stdin_listener(self):
        if not sys.stdin.isatty():
            raise RuntimeError("stdin backend requires an interactive terminal (TTY)")
        fd = sys.stdin.fileno()
        self._stdin_fd = fd
        self._stdin_old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)
        self._stdin_thread.start()
        print("[walk-kbd] using stdin backend")

    def _stdin_loop(self):
        while self.running:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if not ch:
                continue
            key_char = ch.lower()
            with self.lock:
                if key_char in self.key_states:
                    self._set_key(key_char, True)
                elif key_char == "u":
                    self._reset_request = 1
                elif key_char == "p":
                    self._reset_request = 2
                elif key_char == " ":
                    for k in self.key_states:
                        self._set_key(k, False)
                    for k in self.control:
                        self.control[k] = 0.0
                elif key_char == "q":
                    self.running = False
                    return

    def _update_loop(self):
        while self.running:
            with self.lock:
                if self.backend == "stdin":
                    now = time.time()
                    for key_char in self.key_states:
                        if self.key_states[key_char] and (now - self._last_key_event_ts[key_char] > 0.15):
                            self._set_key(key_char, False)

                self._update_axis("x_vel", "w", "s")
                self._update_axis("y_vel", "d", "a")
                self._update_axis("yaw_vel", "x", "z")
                self._update_axis("height_delta", "v", "c", neutral_to=0.0)

            time.sleep(0.02)

    def _update_axis(self, axis: str, pos_key: str, neg_key: str, neutral_to: float = 0.0):
        if self.key_states[pos_key]:
            self.control[axis] = min(self.control[axis] + self.step, self.ranges[axis][1])
        elif self.key_states[neg_key]:
            self.control[axis] = max(self.control[axis] - self.step, self.ranges[axis][0])
        else:
            if self.control[axis] > neutral_to:
                self.control[axis] = max(neutral_to, self.control[axis] - self.step * 2)
            elif self.control[axis] < neutral_to:
                self.control[axis] = min(neutral_to, self.control[axis] + self.step * 2)
        self.control[axis] = float(np.round(self.control[axis], 3))

    def take_snapshot(self):
        with self.lock:
            cmd = self.control.copy()
            reset = self._reset_request
            self._reset_request = None
        return cmd, reset

    def stop(self):
        self.running = False
        if hasattr(self, "listener"):
            self.listener.stop()
        if self._stdin_old_settings is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_settings)
            except Exception:
                pass


def publish_string(pub: ChannelPublisher, value: str):
    pub.Write(String_(data=value))


def main():
    parser = argparse.ArgumentParser(description="H1-2 keyboard walking command publisher")
    parser.add_argument("--backend", choices=["auto", "pynput", "stdin"], default="auto")
    parser.add_argument("--channel", type=int, default=1, help="DDS channel id")
    parser.add_argument("--base_height", type=float, default=0.8, help="neutral commanded body height")
    args = parser.parse_args()

    print("=" * 64)
    print("H1-2 WALK KEYBOARD")
    print("W/S: forward/backward   A/D: strafe left/right")
    print("Z/X: yaw left/right     C/V: lower/raise height")
    print("SPACE: stop command     U: reset object   P: reset all")
    print("Q: quit")
    print("=" * 64)

    ChannelFactoryInitialize(args.channel)
    run_pub = ChannelPublisher("rt/run_command/cmd", String_)
    run_pub.Init()
    reset_pub = ChannelPublisher("rt/reset_pose/cmd", String_)
    reset_pub.Init()

    controller = KeyboardWalkController(backend=args.backend)
    last = None

    try:
        while controller.running:
            time.sleep(0.01)
            cmd, reset = controller.take_snapshot()
            cmd_list = [
                float(cmd["x_vel"]),
                -float(cmd["y_vel"]),
                -float(cmd["yaw_vel"]),
                float(args.base_height + cmd["height_delta"]),
            ]
            publish_string(run_pub, str(cmd_list))
            if reset is not None:
                publish_string(reset_pub, str(reset))
                print(f"[walk-kbd] sent reset category={reset}")

            if cmd_list != last:
                print(f"[walk-kbd] {cmd_list}")
                last = cmd_list
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        publish_string(run_pub, str([0.0, 0.0, 0.0, float(args.base_height)]))
        print("[walk-kbd] stopped")


if __name__ == "__main__":
    main()
