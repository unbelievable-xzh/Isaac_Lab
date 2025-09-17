# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
import isaaclab_tasks.manager_based.manipulation.Catch.mdp as mdp

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.h2017_mass import h2017_CONFIG

##
# Rigid object lift environment.
##


@configclass
class CatchCubeEnvCfg(joint_pos_env_cfg.CatchCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = h2017_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            body_name="link_6",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.205]),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["PrismaticJoint.*"],
            open_command_expr={"PrismaticJoint.*": -0.02},
            close_command_expr={"PrismaticJoint.*": 0.02},
        )

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.35, 0.0, 1.5], rot=[0.70711, 0.0, -0.70711, 0.0]),
            spawn=sim_utils.CuboidCfg(
                size=(0.01, 1.8, 0.05),  
                rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                
                max_depenetration_velocity=5.0,
                disable_gravity=False,),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),  # 蓝色
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
        )

                # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/h2017nomass/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/h2017nomass/link_6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.200],
                    ),
                ),
            ],
        )
        marker_cfg_obj = FRAME_MARKER_CFG.copy()
        marker_cfg_obj.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_obj.prim_path = "/Visuals/FrameTransformer"
        self.scene.object_frame = FrameTransformerCfg(

            prim_path="{ENV_REGEX_NS}/Object",  # 物体 root prim 路径
            debug_vis=True,
            visualizer_cfg=marker_cfg_obj,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object",  # 物体 body prim
                    name="object_center",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],  # 如果需要可在物体中心偏移
                    ),
                ),
            ],
        )


@configclass
class CatchCubeEnvCfg_PLAY(CatchCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 10
        # disable randomization for play
        self.observations.policy.enable_corruption = False


##
# Deformable object lift environment.
##


# @configclass
# class FrankaTeddyBearLiftEnvCfg(FrankaCubeLiftEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         self.scene.object = DeformableObjectCfg(
#             prim_path="{ENV_REGEX_NS}/Object",
#             init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.05), rot=(0.707, 0, 0, 0.707)),
#             spawn=UsdFileCfg(
#                 usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
#                 scale=(0.01, 0.01, 0.01),
#             ),
#         )

#         # Make the end effector less stiff to not hurt the poor teddy bear
#         self.scene.robot.actuators["panda_hand"].effort_limit_sim = 50.0
#         self.scene.robot.actuators["panda_hand"].stiffness = 40.0
#         self.scene.robot.actuators["panda_hand"].damping = 10.0

#         # Disable replicate physics as it doesn't work for deformable objects
#         # FIXME: This should be fixed by the PhysX replication system.
#         self.scene.replicate_physics = False

#         # Set events for the specific object type (deformable cube)
#         self.events.reset_object_position = EventTerm(
#             func=mdp.reset_nodal_state_uniform,
#             mode="reset",
#             params={
#                 "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
#                 "velocity_range": {},
#                 "asset_cfg": SceneEntityCfg("object"),
#             },
#         )

#         # Remove all the terms for the state machine demo
#         # TODO: Computing the root pose of deformable object from nodal positions is expensive.
#         #       We need to fix that part before enabling these terms for the training.
#         self.terminations.object_dropping = None
#         self.rewards.reaching_object = None
#         self.rewards.lifting_object = None
#         self.rewards.object_goal_tracking = None
#         self.rewards.object_goal_tracking_fine_grained = None
#         self.observations.policy.object_position = None
