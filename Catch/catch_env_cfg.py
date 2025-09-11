# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=GroundPlaneCfg(),
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(usd_path="/home/digitaltwins/Project/5GSwarmRobot/V2025_4/AirSimDigitalTwins-/Model/Catch_scene/object2_1_copy.usd", scale=(0.01, 0.01, 0.01)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.9, 0.0, 0.0)),
    )


##
# MDP settings
##
#===========================================================XXXXXXXXXXXXXXXXXX
@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_6",           # 原先是 MISSING，这里明确给末端
        resampling_time_range=(15, 15),   
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-1.35, -1.35), pos_y=(0.0, 0.0), pos_z=(1.5, 1.5),
            roll=(-3.14 ,3.14), pitch=(-3.14, 3.14), yaw=(-3.14,3.14)
        ),
    )
#============================================================√√√√√√√√√√√√√√√√√√√√√√
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    # agv_action: mdp.JointPositionActionCfg = MISSING
#============================================================√√√√√√√√√√√√√√√√√√√√√√
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        #机器人关节位置+角速度
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        #目标物体相对于机器人坐标系下
        object_pose = ObsTerm(func=mdp.object_position_in_robot_root_frame_pos)
        object_quat = ObsTerm(func=mdp.object_position_in_robot_root_frame_quat)
        #command中的目标位姿
        # target_object_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()
#============================================================√√√√√√√√√√√√√√√√√√√√√√
@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
#============================================================√√√√√√√√√√√√√√√√√√√√√√
@configclass
class RewardsCfg:
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-3e-2)
    joint_vel   = RewTerm(func=mdp.joint_vel_l2,  weight=-3e-2, params={"asset_cfg": SceneEntityCfg("robot")})
    # 成功抬升奖励
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 1.3}, weight=12)
    ########末端接近物体##########
    approach_ee = RewTerm(func=mdp.object_ee_distance, weight=2.0,params={"threshold": 0.3 })
    orientation_correct = RewTerm(
        func=mdp.grasp_object_orientation,
        weight=0.5,
    )
    #########抓住物体##########
    approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)

    #掉落惩罚
    object_dropping = RewTerm(func=mdp.object_is_dropped, params={"minimal_height": 0.8}, weight=-15.0)
#============================================================√√√√√√√√√√√√√√√√√√√√√√
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 任务超时终止条件
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # # 物体掉落终止条件（如果物体低于最低高度，则任务终止）
    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": 0.05, "asset_cfg": SceneEntityCfg("object")},  # 目标物体是cube
    # )

# #============================================================√√√√√√√√√√√√√√√√√√√√√√
@configclass
class CurriculumCfg:
###############阶段0：关节速度以及加速度抑制，以及粗调抓取的位置与姿态
###############阶段1：细化预抓取位置姿态
    ##初步稳定增加速度惩罚
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -3e-1, "num_steps": 5000}
    )
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -3e-1, "num_steps": 5000}
    )
#     #增大靠近奖励
#     approach_ee_modify = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "approach_ee_exactly", "weight": 4.5, "num_steps": 5000}
#     )
#     #细化姿势调整
#     increase_orientation = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "orentation_correct_exactly", "weight": -4, "num_steps": 10000}
#     )
# ###############阶段2：进入实际抓取
# ###删除eew奖励，改换为eew_real与obj_cube的距离差值（相对大权重），保持increaseorientation不变###
#     approach_real = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "approach_ee_real", "weight": 3.5, "num_steps": 10000}
#     )
#     decrease_approach_ee_modify = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "approach_ee_exactly", "weight": 1.5, "num_steps": 10000}
#     )
#     turn_off_approach_ee = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "approach_ee", "weight": 0, "num_steps": 10000}
#     )
#     turn_off_approach_ee_modify = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "approach_ee_exactly", "weight": 0, "num_steps": 15000}
#     )
#     increase_approach_real = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "approach_ee_real", "weight": 5.5, "num_steps": 15000}
#     )
#     Lift_Object = CurrTerm(
#         func=mdp.modify_reward_weight,params={"term_name": "lifting_object", "weight": 18, "num_steps": 30000}
#     )
# ###############阶段3：追踪command的位置
@configclass
class CatchEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=6)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
