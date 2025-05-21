from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import format_path
from mani_skill.utils.structs import Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import assets, download_asset

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

import os
import tempfile
from xml.etree import ElementTree as ET
from pathlib import Path

def generate_modified_urdf(template_urdf_path: str, output_path: str, z_scale: float) -> tuple:
    """
    Load the original URDF, modify the scale of link5, and save to a temporary file.
    Scales link5's z-dimension and adjusts joint positions to maintain proper connections.
    
    Args:
        template_urdf_path: Path to the original URDF file
        output_path: Path where the modified URDF should be saved
        z_scale: Scale factor for the z dimension of link5
        
    Returns:
        tuple: (path to modified URDF, original URDF data)
    """
    # Parse the URDF file
    print(f"Generating modified URDF for {template_urdf_path} with z-scale of {z_scale}...")
    tree = ET.parse(template_urdf_path)
    root = tree.getroot()
    
    # Get the directory of the original URDF to resolve relative paths
    base_dir = os.path.dirname(template_urdf_path)
    
    # Find link5 in the URDF and apply scaling if needed
    if z_scale != 1.0:
        found = False
        for link in root.findall(".//link"):
            if link.attrib.get("name") == "panda_link5":
                found = True
                # For each visual and collision element in link5
                for element in link.findall("visual") + link.findall("collision"):
                    # Find geometry element
                    geometry = element.find("geometry")
                    if geometry is not None:
                        # Find mesh element
                        mesh = geometry.find("mesh")
                        if mesh is not None:
                            # Add scale attribute to scale in Z direction only
                            mesh.attrib["scale"] = f"1.0 1.0 {z_scale}"
                            
                            # Make mesh path absolute
                            if "filename" in mesh.attrib:
                                filename = mesh.attrib["filename"]
                                if not os.path.isabs(filename):
                                    abs_path = os.path.normpath(os.path.join(base_dir, filename))
                                    mesh.attrib["filename"] = abs_path
                                    
                                # Handle both STL and GLB files
                                if filename.endswith(('.stl', '.glb')):
                                    print(f"Scaling {filename} with z-scale of {z_scale}")
        
        if not found:
            print("Warning: panda_link5 not found in URDF file.")
        else:
            print(f"Successfully modified link5 with z-scale of {z_scale}")
        
        # Calculate joint5 offset (between link4 and link5)
        # Move joint5 upward relative to link4 to reduce overlap
        joint5_offset = 0.2 * (z_scale - 1.0)
        
        # Find and adjust joint5
        for joint in root.findall(".//joint"):
            if joint.attrib.get("name") == "panda_joint5":
                origin = joint.find("origin")
                if origin is not None and "xyz" in origin.attrib:
                    # Parse the current xyz values
                    xyz = origin.attrib["xyz"].split()
                    x, y, z = map(float, xyz)
                    
                    # Apply offset to y-coordinate (up/down in the robot's frame)
                    y += joint5_offset
                    
                    # Set the new xyz values
                    origin.attrib["xyz"] = f"{x} {y} {z}"
                    print(f"Adjusted joint5 origin: added {joint5_offset} to y coordinate")
    
    # Fix all mesh paths - make them absolute
    for link in root.findall(".//link"):
        for element in link.findall(".//visual") + link.findall(".//collision"):
            geometry = element.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None and "filename" in mesh.attrib:
                    filename = mesh.attrib["filename"]
                    if not os.path.isabs(filename):
                        abs_path = os.path.normpath(os.path.join(base_dir, filename))
                        mesh.attrib["filename"] = abs_path
    
    # Save the modified URDF
    tree.write(output_path)
    
    print(f"Modified URDF saved to {output_path}")
    
    return output_path, tree
    

@register_agent()
class ContextualPandaStick(BaseAgent):
    uid = "contextual_panda_stick"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_stick.urdf"
    urdf_config = dict()
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    
    ee_link_name = "panda_hand_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    def __init__(
        self,
        scene: sapien.Scene,
        control_freq: int,
        control_mode: str,
        agent_idx: Optional[int] = None,
        initial_pose: Optional[sapien.Pose] = None,
    #     build_separate: bool = False,
        link5_z_scale: float = 1.0,
    ):
        self.link5_z_scale = link5_z_scale
        super().__init__(
            scene=scene,
            control_freq=control_freq,
            control_mode=control_mode,
            agent_idx=agent_idx,
            initial_pose=initial_pose,
        #     build_separate=build_separate,
        )
        print("Initializing ContextualPandaStick...")

    def _load_articulation(self, initial_pose: Optional[Union[sapien.Pose, Pose]] = None):
        def build_articulation(scene_idxs: Optional[List[int]] = None):
            loader = self.scene.create_urdf_loader()
            asset_path = format_path(str(self.urdf_path))

            temp_urdf_path = None
            if self.link5_z_scale != 1.0:
                temp_dir = tempfile.gettempdir()
                temp_urdf_path = os.path.join(temp_dir, f"panda_stick_scaled_{self.link5_z_scale}.urdf")
                asset_path, _ = generate_modified_urdf(asset_path, temp_urdf_path, self.link5_z_scale)
                _, _ = generate_modified_urdf(asset_path, "/home/swarajh/ContextualManiskill/contextual_maniskill/temp.urdf", self.link5_z_scale)

            loader.name = self.uid
            if self._agent_idx is not None:
                loader.name = f"{self.uid}-agent-{self._agent_idx}"
            loader.fix_root_link = self.fix_root_link
            loader.load_multiple_collisions_from_file = self.load_multiple_collisions
            loader.disable_self_collisions = self.disable_self_collisions

            if self.urdf_config:
                urdf_config = sapien_utils.parse_urdf_config(self.urdf_config)
                sapien_utils.check_urdf_config(urdf_config)
                sapien_utils.apply_urdf_config(loader, urdf_config)

            builder = loader.parse(asset_path)["articulation_builders"][0]
            builder.initial_pose = initial_pose
            if scene_idxs is not None:
                builder.set_scene_idxs(scene_idxs)
                builder.set_name(f"{self.uid}-agent-{self._agent_idx}-{scene_idxs}")
            robot = builder.build()
            assert robot is not None, f"Fail to load URDF from {asset_path}"

            if temp_urdf_path and os.path.exists(temp_urdf_path):
                try:
                    os.remove(temp_urdf_path)
                except:
                    pass

            return robot

        self.robot = build_articulation()
        self.robot_link_names = [link.name for link in self.robot.get_links()]

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        arm_pd_ee_delta_pose_align = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_delta_pose_align.frame = "ee_align"

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos),
            pd_joint_pos=dict(arm=arm_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose),
            pd_ee_delta_pose_align=dict(arm=arm_pd_ee_delta_pose_align),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos),
            pd_ee_target_delta_pos=dict(arm=arm_pd_ee_target_delta_pos),
            pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel),
            pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel),
            pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold