# H1-2 Balance & Locomotion Controller Guide

Two methods to get stable, continuous balance on H1-2 — either by running the trained RL policy, or building a lightweight IMU-based balance controller using unitree_sdk2py.

---

## Method A — Trained Walking Policy (Recommended)

The RL policy already knows how to balance. The infrastructure (`DDSRLActionProvider`) already exists in this repo.

### Step 1: Train and export

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-H12-Velocity-ManagerBased-v0 \
  --device cuda --headless --num_envs 4096 --max_iterations 5000
```

The training script auto-exports `policy.onnx` and `policy.pt` to:
```
logs/rsl_rl/h12_velocity_flat/<timestamp>/exported/
```

### Step 2: Verify the exported policy

```bash
python3 - <<'EOF'
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("logs/rsl_rl/h12_velocity_flat/<timestamp>/exported/policy.onnx")
obs = np.zeros((1, 75), dtype=np.float32)
action = sess.run(None, {sess.get_inputs()[0].name: obs})[0]
print("Output shape:", action.shape)  # expect (1, 21)
EOF
```

### Step 3: Run in sim with keyboard velocity commands

**Terminal A — simulation:**
```bash
python sim_main.py \
  --device cuda --enable_cameras \
  --task Isaac-Move-Cylinder-H12-WholeBody \
  --enable_inspire_dds --robot_type h1_2 \
  --model_path logs/rsl_rl/h12_velocity_flat/<timestamp>/exported/policy.onnx
```

**Terminal B — keyboard:**
```bash
python send_commands_keyboard.py --backend stdin --channel 1
# WASD = forward/back/strafe, QE = turn, Space = stop
```

### Step 4: Run against a standalone scene (no DDS)

Use `scene/joint_motion.py` as the base. Add policy inference in the loop:

```python
import onnxruntime as ort
import numpy as np

POLICY_PATH = "logs/rsl_rl/h12_velocity_flat/<timestamp>/exported/policy.onnx"
DEFAULT_POS  = np.array([...])   # 21 values from LOCOMOTION_JOINT_NAMES defaults
ACTION_SCALE = 0.75              # must match training config

sess = ort.InferenceSession(POLICY_PATH)
prev_action = np.zeros(21, dtype=np.float32)
cmd_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # vx, vy, wz

while simulation_app.is_running():
    # --- collect observations (shape: 75) ---
    root_state = dc.get_articulation_root_body_states(art, _dynamic_control.STATE_ALL)
    dof_states = dc.get_articulation_dof_states(art, _dynamic_control.STATE_ALL)

    # root_state: [pos(3), rot_quat(4), lin_vel(3), ang_vel(3)]
    lin_vel_w  = root_state[7:10]
    ang_vel_w  = root_state[10:13]
    quat       = root_state[3:7]               # (w, x, y, z) — check sdk convention

    # rotate world gravity [0,0,-9.81] into body frame using quaternion inverse
    gravity_w  = np.array([0.0, 0.0, -1.0])
    proj_grav  = quat_rotate_inverse(quat, gravity_w)  # implement or use scipy

    joint_pos  = dof_states["pos"][:21]        # first 21 = locomotion joints
    joint_vel  = dof_states["vel"][:21]

    obs = np.concatenate([
        lin_vel_w,
        ang_vel_w,
        proj_grav,
        cmd_vel,
        joint_pos - DEFAULT_POS,
        joint_vel,
        prev_action,
    ], dtype=np.float32)[None]  # (1, 75)

    # --- policy inference ---
    action = sess.run(None, {sess.get_inputs()[0].name: obs})[0][0]  # (21,)
    targets = DEFAULT_POS + ACTION_SCALE * action
    prev_action = action

    # --- send targets ---
    for i, name in enumerate(LOCOMOTION_JOINT_NAMES):
        if name in name_to_dof:
            dc.set_dof_position_target(name_to_dof[name], float(targets[i]))

    simulation_app.update()
```

**Quaternion helper (body-frame gravity projection):**
```python
from scipy.spatial.transform import Rotation

def quat_rotate_inverse(quat_wxyz, vec):
    # quat: [w, x, y, z]
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return r.inv().apply(vec)
```

---

## Method B — IMU/CoM Feedback Balance Controller (Real Robot)

A minimal whole-body balance controller using unitree_sdk2py. Runs on the real robot or in sim via DDS.

### Architecture

```
LowState (500 Hz)              Controller                  LowCmd (500 Hz)
─────────────────              ──────────────              ─────────────────
imu.rpy[1]       ──pitch──►   CoM estimator  ──targets──► ankle_pitch joints
imu.gyroscope[1] ──dω/dt──►   ankle strategy             hip_pitch joints
motor_state[i].q ──q_leg──►   LIP model
motor_state[i].dq ──dq──►
```

### Step 1: Install SDK

```bash
pip install unitree_sdk2py
```

Disable iceoryx shared memory (causes crash without real robot hardware):
```python
import unitree_sdk2py.core.channel as _ch
_shm_off = "<SharedMemory><Enable>false</Enable></SharedMemory>"
for attr in ("ChannelConfigAutoDetermine", "ChannelConfigHasInterface"):
    cfg = getattr(_ch, attr, None)
    if cfg and _shm_off not in cfg:
        setattr(_ch, attr, cfg.replace("</Domain>", f"{_shm_off}</Domain>"))
```

### Step 2: Subscribe to robot state

```python
from unitree_sdk2py.core.channel import ChannelFactory, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_

ChannelFactory.Instance().Init(0)

state = {"rpy": [0,0,0], "gyro": [0,0,0], "q": [0]*27, "dq": [0]*27}

def on_low_state(msg: LowState_):
    state["rpy"]  = list(msg.imu_state.rpy)
    state["gyro"] = list(msg.imu_state.gyroscope)
    for i in range(27):
        state["q"][i]  = msg.motor_state[i].q
        state["dq"][i] = msg.motor_state[i].dq

sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(on_low_state, 10)
pub = ChannelPublisher("rt/lowcmd", LowCmd_)
pub.Init()
```

### Step 3: Ankle strategy balance loop

The simplest effective approach. Treats the robot as an inverted pendulum; counters forward lean with ankle plantarflexion.

```python
import time
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

# Joint indices (H1-2 HG protocol order)
LEFT_ANKLE_PITCH  = 4
RIGHT_ANKLE_PITCH = 10
LEFT_HIP_PITCH    = 1
RIGHT_HIP_PITCH   = 7

# Stance defaults
ANKLE_DEFAULT = -0.23   # rad — confirmed balanced in sim
HIP_DEFAULT   = -0.4

# Gains — balance loop only, keep weak to not fight leg pose
KP_ANKLE = 80.0
KD_ANKLE = 5.0
KP_HIP   = 150.0
KD_HIP   = 8.0

# Balance gains (tuned empirically; start small)
PITCH_KP = 0.6    # ankle correction per rad of lean
PITCH_KD = 0.04   # damping

def balance_loop():
    rate = 0.002  # 500 Hz
    while True:
        t0 = time.time()

        pitch   = state["rpy"][1]          # forward lean (rad)
        pitch_d = state["gyro"][1]         # pitch rate (rad/s)

        ankle_corr = PITCH_KP * pitch + PITCH_KD * pitch_d
        # Clamp correction to safe range
        ankle_corr = max(-0.15, min(0.15, ankle_corr))

        cmd = unitree_hg_msg_dds__LowCmd_()

        for idx, default in [
            (LEFT_ANKLE_PITCH,  ANKLE_DEFAULT),
            (RIGHT_ANKLE_PITCH, ANKLE_DEFAULT),
        ]:
            cmd.motor_cmd[idx].q   = default - ankle_corr
            cmd.motor_cmd[idx].kp  = KP_ANKLE
            cmd.motor_cmd[idx].kd  = KD_ANKLE
            cmd.motor_cmd[idx].tau = 0.0
            cmd.motor_cmd[idx].dq  = 0.0

        for idx, default in [
            (LEFT_HIP_PITCH,  HIP_DEFAULT),
            (RIGHT_HIP_PITCH, HIP_DEFAULT),
        ]:
            cmd.motor_cmd[idx].q   = default
            cmd.motor_cmd[idx].kp  = KP_HIP
            cmd.motor_cmd[idx].kd  = KD_HIP
            cmd.motor_cmd[idx].tau = 0.0
            cmd.motor_cmd[idx].dq  = 0.0

        pub.write(cmd)
        elapsed = time.time() - t0
        time.sleep(max(0.0, rate - elapsed))

balance_loop()
```

### Step 4: Add CoM velocity estimation (LIP model)

For stronger disturbance rejection, estimate horizontal CoM velocity from IMU:

```python
import numpy as np

COM_HEIGHT = 0.98   # m — approximate pelvis height
GRAVITY    = 9.81

class LIPBalance:
    """Linear Inverted Pendulum balance controller."""
    def __init__(self, h=COM_HEIGHT, dt=0.002):
        self.h  = h
        self.dt = dt
        self.com_vel = 0.0      # estimated CoM horizontal velocity
        self.com_pos = 0.0      # estimated CoM position offset

    def update(self, pitch_rad, pitch_rate_rad_s):
        # Small-angle: horizontal accel ≈ g * pitch
        com_accel = GRAVITY * pitch_rad
        self.com_vel += com_accel * self.dt
        self.com_pos += self.com_vel * self.dt

        # LIP capture point: x_cp = com_pos + com_vel / omega_0
        omega_0 = np.sqrt(GRAVITY / self.h)
        capture_pt = self.com_pos + self.com_vel / omega_0

        # Ankle torque to drive capture point to zero
        ankle_correction = 0.8 * capture_pt + 0.05 * self.com_vel
        return np.clip(ankle_correction, -0.15, 0.15)

lip = LIPBalance()

# In balance loop, replace ankle_corr with:
ankle_corr = lip.update(state["rpy"][1], state["gyro"][1])
```

### Step 5: Test in sim first

Before running on real hardware, test via DDS bridge in sim:

```bash
# Terminal A: sim with DDS enabled
python sim_main.py --task Isaac-Move-Cylinder-H12-WholeBody \
  --enable_inspire_dds --robot_type h1_2 --device cuda

# Terminal B: your balance controller (targets go through DDS to sim)
python your_balance_controller.py
```

---

## Comparison

| | Method A (RL Policy) | Method B (WBC) |
|---|---|---|
| Balance quality | High — end-to-end trained | Depends on tuning |
| Walking | Yes, velocity-commanded | Add separately (CPG / stepping logic) |
| Real robot | Via `DDSRLActionProvider` | Directly via unitree_sdk2py |
| Sim-to-real gap | Managed by domain randomization | None (physics-based) |
| Implementation effort | Low — export + run | Medium — tune gains |
| Handles perturbations | Well (trained on push events) | Only if LIP model is fast enough |

**Recommended**: Start with Method A in sim to validate walking, then optionally layer Method B on top as a safety fallback for standing balance when velocity command = 0.
