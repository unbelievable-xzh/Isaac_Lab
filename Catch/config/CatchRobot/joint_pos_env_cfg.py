# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.h2017_mass import h2017_CONFIG
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.Catch.catch_env_cfg import CatchEnvCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.h2017_mass import h2017_CONFIG


@configclass
class CatchCubeEnvCfg(CatchEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set H2017 as robot
        self.scene.robot = h2017_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (h2017)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            scale=1,
            use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["PrismaticJoint_1"],
            open_command_expr={"PrismaticJoint_1": 0.04},
            close_command_expr={"PrismaticJoint_1": 0.0},
        )

        self.commands.object_pose.body_name = "link_6"

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.78, 0.0, 1.06], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 1.2, 0.05),  
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
#         self.scene.object = RigidObjectCfg(
#             prim_path="{ENV_REGEX_NS}/Object",
#             init_state=RigidObjectCfg.InitialStateCfg(
#                 pos=[-1.78, 0.0, 1.06],
#                 rot=[1.0, 0.0, 0.0, 0.0],
#             ),
#             spawn=sim_utils.CuboidCfg(
#                 size=(0.08, 1.8, 0.01),   
#                 mass_props=sim_utils.MassPropertiesCfg(mass=0.30,),
#                 physics_material=sim_utils.RigidBodyMaterialCfg(
#                     static_friction=1.2,    # 前期高摩擦更稳，后期降到 0.6~0.8
#                     dynamic_friction=1.0,
#                     restitution=0.0,        # 近似无弹跳
#                     friction_combine_mode="average",     # 或 "min"
#                     restitution_combine_mode="min",),
#                 collision_props=sim_utils.CollisionPropertiesCfg(
#                     contact_offset=0.003,   # 接触膨胀，避免穿透
#                     rest_offset=0.0, ),
#                 rigid_props=RigidBodyPropertiesCfg(
#                     solver_position_iteration_count=16,   # 你已设置
#                     solver_velocity_iteration_count=4,    # 建议 >1，薄物+摩擦接触更稳
#                     max_angular_velocity=1000.0,
#                     max_linear_velocity=1000.0,
#                     max_depenetration_velocity=3.0,       # 可略降，避免“弹出”
#                     disable_gravity=False,
#                     enable_gyroscopic_forces=True,),
# )
# )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/h2017nomass/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/h2017nomass/link_6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.15],
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
        marker_cfg_gripper = FRAME_MARKER_CFG.copy()
        marker_cfg_gripper.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_gripper.prim_path = "/Visuals/FrameTransformer"
        self.scene.gripper_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/catch/GeoContainer",
            debug_vis=False,
            visualizer_cfg=marker_cfg_gripper,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/catch/Link_2",
                    name="finger_1",
                    offset=OffsetCfg(
                        pos=[0.01, 0.57, 0.075],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/catch/Link_3",
                    name="finger_3",
                    offset=OffsetCfg(
                        pos=[0.01, 0.0, 0.075],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/catch/Link_5",
                    name="finger_4",
                    offset=OffsetCfg(
                        pos=[-0.01, 0.57, 0.075],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/catch/Link_6",
                    name="finger_6",
                    offset=OffsetCfg(
                        pos=[-0.01, 0.0, 0.075],
                    ),
                ),
            ],
        )
        ########
        # Set the body name for the end effector
        self.rewards.approach_gripper_handle.params["offset"] = 0.1
        self.rewards.grasp_object.params["open_joint_pos"] = 0.04
        self.rewards.grasp_object.params["asset_cfg"].joint_names = ["PrismaticJoint_1"]
@configclass
class CatchCubeEnvCfg_PLAY(CatchCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 6
        # disable randomization for play
        self.observations.policy.enable_corruption = False
