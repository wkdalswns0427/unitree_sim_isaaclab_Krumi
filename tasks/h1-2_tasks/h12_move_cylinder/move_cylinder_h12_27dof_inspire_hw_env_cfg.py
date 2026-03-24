# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
import torch
import os

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from . import mdp
from tasks.common_config import H12RobotPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_pickplace_cylindercfg_wholebody import TableCylinderSceneCfgWH

# Task target: pick the cylinder and place it 1 ft (0.3048 m) to the +x direction.
_OBJECT_INIT_X = -2.58514
_OBJECT_INIT_Y = -2.78975
_ONE_FOOT_M = 0.3048
_TARGET_X = _OBJECT_INIT_X + _ONE_FOOT_M
_TARGET_Y = _OBJECT_INIT_Y
_TARGET_Z = 0.855

# Workspace and goal-zone bounds for termination/reward.
_MIN_X = _OBJECT_INIT_X - 0.20
_MAX_X = _TARGET_X + 0.25
_MIN_Y = _OBJECT_INIT_Y - 0.25
_MAX_Y = _OBJECT_INIT_Y + 0.25
_MIN_H = 0.5
_POST_HALF_X = 0.08
_POST_HALF_Y = 0.10
_POST_MIN_X = _TARGET_X - _POST_HALF_X
_POST_MAX_X = _TARGET_X + _POST_HALF_X
_POST_MIN_Y = _TARGET_Y - _POST_HALF_Y
_POST_MAX_Y = _TARGET_Y + _POST_HALF_Y
_POST_MIN_H = 0.81
_POST_MAX_H = 0.9

@configclass
class ObjectTableSceneCfg(TableCylinderSceneCfgWH):
    """Object-table scene config for H1-2 wholebody move task."""

    # Floating-base robot: the base is free to move so the policy actually
    # experiences gravity and must learn to balance. The non-floating variant
    # has its root anchored to the world and cannot walk.
    robot: ArticulationCfg = H12RobotPresets.h12_27dof_inspire_wholebody_floating(
        init_pos=(-3.9, -2.81811, 1.00),
        init_rot=(1, 0, 0, 0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    # Cameras are not present in the official H1-2 USD used by the floating-base
    # variant, so they are omitted here. The RL policy only uses the `policy`
    # observation group which has no camera terms.
    front_camera = None
    left_wrist_camera = None
    right_wrist_camera = None
    robot_camera = None
    world_camera = None  # inherited from base scene; disable for headless RL


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        # Expose legs (balance), arms (reaching), and proximal finger joints only.
        # Distal (intermediate/distal) joints are passive and mirror the proximals
        # via fixed ratios in hardware. Including them would double the hand action
        # space and force the policy to re-learn the mirroring relationship from
        # scratch, making convergence much harder.
        joint_names=[
            # Legs — needed for balance while arms move
            ".*_hip_.*_joint",
            ".*_knee_joint",
            ".*_ankle_.*_joint",
            # Arms — reaching and wrist orientation
            ".*_shoulder_.*_joint",
            ".*_elbow_joint",
            ".*_wrist_.*_joint",
            # Inspire proximal joints — the 12 directly actuated finger joints
            ".*_proximal_joint",
            ".*_proximal_pitch_joint",
            ".*_proximal_yaw_joint",
        ],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states, params={"enable_dds": False})
        # Finger positions — where the joints currently are
        robot_inspire_state = ObsTerm(func=mdp.get_robot_inspire_joint_states, params={"enable_dds": False})
        # Finger velocities — closed-loop feedback so the policy can regulate
        # finger speed, not just endpoint position
        robot_inspire_vel = ObsTerm(func=mdp.get_robot_inspire_joint_vel)
        # Balance signals: the policy needs these to know whether it is upright
        # and how fast it is rotating. Without them the policy has no gradient
        # signal to resist falling even if balance rewards are present.
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, scale=1.0)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity, scale=1.0)
        # Object position relative to robot root — without this the policy has
        # no spatial awareness of where the cylinder is and cannot learn to reach.
        object_pos = ObsTerm(func=mdp.get_object_pos_in_robot_frame)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_out_of_workspace = DoneTerm(
        func=mdp.reset_object_estimate,
        params={
            "min_x": _MIN_X,
            "max_x": _MAX_X,
            "min_y": _MIN_Y,
            "max_y": _MAX_Y,
            "min_height": _MIN_H,
        },
    )
    # Terminate when the robot's torso hits the ground so episodes reset
    # quickly instead of the policy learning to flail on the floor.
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 10.0,
        },
    )


@configclass
class RewardsCfg:
    reward = RewTerm(
        func=mdp.compute_reward,
        weight=1.0,
        params={
            # Disable DDS reward publishing for RL training runs.
            "enable_dds": False,
            # Goal: place 1 ft to the right (+x) from the object default location.
            "min_x": _MIN_X,
            "max_x": _MAX_X,
            "min_y": _MIN_Y,
            "max_y": _MAX_Y,
            "min_height": _MIN_H,
            "post_min_x": _POST_MIN_X,
            "post_max_x": _POST_MAX_X,
            "post_min_y": _POST_MIN_Y,
            "post_max_y": _POST_MAX_Y,
            "post_min_height": _POST_MIN_H,
            "post_max_height": _POST_MAX_H,
            "target_x": _TARGET_X,
            "target_y": _TARGET_Y,
            "target_z": _TARGET_Z,
            "dense_xy_weight": 0.4,
            "dense_z_weight": 0.2,
            "dense_xy_scale": 4.0,
            "dense_z_scale": 10.0,
        },
    )
    # Heavy penalty for falling; creates a strong gradient to stay upright.
    termination_penalty = RewTerm(
        func=base_mdp.is_terminated,
        weight=-50.0,
    )
    # Penalise tilt away from upright (projected gravity deviates from [0,0,-1]).
    flat_orientation = RewTerm(
        func=base_mdp.flat_orientation_l2,
        weight=-2.0,
    )
    # Penalise vertical root velocity to discourage bouncing / falling.
    lin_vel_z = RewTerm(
        func=base_mdp.lin_vel_z_l2,
        weight=-1.0,
    )
    # Penalise rapid joint acceleration to encourage smooth, stable motion.
    action_rate = RewTerm(
        func=base_mdp.action_rate_l2,
        weight=-0.005,
    )
    # Dense reaching reward: pulls the policy toward the cylinder before the
    # sparse placement reward fires. joint_deviation_arms is intentionally
    # omitted — penalising arm movement would prevent the policy from ever
    # reaching the object.
    wrist_to_object = RewTerm(
        func=mdp.wrist_to_object_reward,
        weight=1.0,
        params={"std": 1.5},
    )
    base_to_object = RewTerm(
        func=mdp.base_to_object_reward,
        weight=0.8,
        params={"std": 2.0},
    )
    # === Balance rewards ===
    # Positive reward each step the robot stays upright — counteracts
    # the termination penalty so the policy learns to survive.
    alive = RewTerm(
        func=base_mdp.is_alive,
        weight=2.0,
    )
    # Keep hip joints near default standing pose for stability.
    joint_deviation_hip = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )


def _reset_object_self(env) -> None:
    base_mdp.reset_root_state_uniform(
        env,
        torch.arange(env.num_envs, device=env.device),
        pose_range={"x": [-0.05, 0.05], "y": [0.0, 0.05]},
        velocity_range={},
        asset_cfg=SceneEntityCfg("object"),
    )


def _reset_all_self(env) -> None:
    base_mdp.reset_scene_to_default(
        env,
        torch.arange(env.num_envs, device=env.device),
    )


@configclass
class EventCfg:
    # Reset the robot base pose and velocity on episode reset so it
    # returns to the standing position instead of staying underground.
    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Reset all joint positions and velocities to defaults.
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Reset object to its default position (with small random offset).
    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (0.0, 0.05)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class MoveCylinderH1227dofInspireWholebodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(func=_reset_object_self),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(func=_reset_all_self),
        )
