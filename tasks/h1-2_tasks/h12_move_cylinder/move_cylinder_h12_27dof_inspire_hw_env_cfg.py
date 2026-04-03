# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
import torch
import os

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
        # Moved ~0.6 m closer to the cylinder (was -3.9).  The robot still needs
        # to walk a short distance, which is appropriate for curriculum stage 1+,
        # but the distance is no longer so large that the approach reward saturates.
        init_pos=(-3.3, -2.81811, 1.00),
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
    # ── Curriculum stage 2 (HOLD): placement reward ─────────────────────────
    # Gated — zero until the policy has learned to grasp.
    placement = RewTerm(
        func=mdp.placement_reward,
        weight=1.0,
        params={
            "enable_dds": False,
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
    # ── Curriculum stage 1 (GRASP): fingertip reward ────────────────────────
    # Gated — zero until wrist is reliably near the cylinder.
    fingertip_grasp = RewTerm(
        func=mdp.fingertip_grasp_reward,
        weight=2.0,
        params={"proximity_threshold": 0.40, "grasp_std": 0.10},
    )
    # ── Always active: balance + approach ───────────────────────────────────
    # base_to_object (std=1.0): steep enough that the policy gets real gradient
    # to walk forward.  At 0.72 m → reward 0.49, at 0.3 m → 0.74 (Δ=0.25).
    base_to_object = RewTerm(
        func=mdp.base_to_object_reward,
        weight=1.0,
        params={"std": 1.0},
    )
    # wrist_to_object (std=0.3): nearly zero when far away, so the policy
    # keeps arms relaxed while walking.  Strong gradient only within ~40 cm
    # so the arm extends naturally once the base is close.
    wrist_to_object = RewTerm(
        func=mdp.wrist_to_object_reward,
        weight=1.5,
        params={"std": 0.3},
    )
    alive = RewTerm(
        func=base_mdp.is_alive,
        weight=2.0,
    )
    termination_penalty = RewTerm(
        func=base_mdp.is_terminated,
        weight=-50.0,
    )
    flat_orientation = RewTerm(
        func=base_mdp.flat_orientation_l2,
        weight=-2.0,
    )
    lin_vel_z = RewTerm(
        func=base_mdp.lin_vel_z_l2,
        weight=-1.0,
    )
    action_rate = RewTerm(
        func=base_mdp.action_rate_l2,
        weight=-0.005,
    )
    joint_deviation_hip = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    # Penalise arm joints deviating from default pose — prevents the policy
    # from twisting arms into unnatural configurations for marginal reward.
    joint_deviation_arm = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint"])},
    )


@configclass
class CurriculumCfg:
    """Four-stage curriculum: balance → approach → grasp → hold.

    The :func:`~mdp.advance_curriculum_stage` function is called every step by
    the :class:`~isaaclab.managers.CurriculumManager`.  It advances each
    environment's stage independently when the success condition for that stage
    is met for a sustained number of consecutive steps.  Stage progress is
    preserved across episode resets.
    """
    stage_advance = CurrTerm(func=mdp.advance_curriculum_stage)


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
    curriculum: CurriculumCfg = CurriculumCfg()

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
