# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""H1-2 fist open/close training task.

The robot stands still (base-fixed) and learns to close its fingers into a fist
then open them, alternating every 2.5 seconds.  Only the 12 Inspire proximal
joints are in the action space.
"""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp
from tasks.common_config import H12RobotPresets


@configclass
class FistSceneCfg(InteractiveSceneCfg):
    """Minimal scene: base-fixed H1-2 with Inspire hand, no objects."""

    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )

    robot: ArticulationCfg = H12RobotPresets.h12_27dof_inspire_wholebody(
        init_pos=(0.0, 0.0, 0.80),
        init_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        super().__post_init__()
        # Fix the root link so the robot base doesn't move during finger-only training.
        self.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
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
        # Current finger joint positions (12)
        finger_pos = ObsTerm(func=mdp.get_inspire_joint_pos, params={"enable_dds": False})
        # Current finger joint velocities (12)
        finger_vel = ObsTerm(func=mdp.get_inspire_joint_vel)
        # Target: open (zeros) or closed (fist angles) — alternates every 2.5s (12)
        fist_target = ObsTerm(func=mdp.get_fist_target)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # Main tracking reward: exp(-error^2 / std^2).
    fist_tracking = RewTerm(
        func=mdp.fist_tracking_reward,
        weight=5.0,
        params={"std": 0.3},
    )
    # Small L2 penalty for additional gradient.
    fist_l2 = RewTerm(
        func=mdp.fist_tracking_l2,
        weight=-0.5,
    )
    # Smooth actions.
    action_rate = RewTerm(
        func=base_mdp.action_rate_l2,
        weight=-0.01,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class FistControlH12EnvCfg(ManagerBasedRLEnvCfg):
    scene: FistSceneCfg = FistSceneCfg(
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
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation