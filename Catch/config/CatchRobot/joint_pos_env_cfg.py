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
            joint_names=["PrismaticJoint.*"],
            open_command_expr={"PrismaticJoint.*": 0.02},
            close_command_expr={"PrismaticJoint.*": -0.02},
        )

        self.commands.object_pose.body_name = "link_6"

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.67, 0.0, 1.09], rot=[0.70711, 0.0, -0.70711, 0.0]),
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
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/h2017nomass/link_6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.30],
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
        marker_cfg_real = FRAME_MARKER_CFG.copy()
        marker_cfg_real.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_real.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame_real = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/h2017nomass/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/h2017nomass/link_6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.20],
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
