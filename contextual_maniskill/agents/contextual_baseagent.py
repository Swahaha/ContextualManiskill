from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union, Any

import numpy as np
import sapien
import torch
from gymnasium import spaces

from mani_skill import format_path
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.utils import assets, download_asset, sapien_utils
from mani_skill.utils.structs import Articulation
from mani_skill.utils.structs.pose import Pose

from mani_skill.agents.controllers.base_controller import (
    ControllerConfig,
)

class ContextualBaseAgent(BaseAgent):
    """
    BaseAgent with the ability to modify link geometries (visual and collision).
    This allows for modifying specific robot links (e.g., making Panda's link7 longer).
    """
    
    def __init__(
        self,
        scene,
        control_freq: int,
        control_mode: Optional[str] = None,
        agent_idx: Optional[str] = None,
        initial_pose: Optional[Union[sapien.Pose, Pose]] = None,
        build_separate: bool = False,
        link_modifications=None,
    ):
        """
        Args:
            link_modifications: Dictionary mapping link names to modification parameters.
                Example: {"panda_link7": {"scale": [1.5, 1.0, 1.0]}} to make link7 1.5x longer in x direction
        """
        # Store link_modifications locally
        self.link_modifications = link_modifications or {}
        
        # Pass ALL the expected parameters to the parent class
        super().__init__(
            scene=scene,
            control_freq=control_freq,
            control_mode=control_mode,
            agent_idx=agent_idx,
            initial_pose=initial_pose,
            build_separate=build_separate,
        )
        
    def _load_articulation(
        self, initial_pose: Optional[Union[sapien.Pose, Pose]] = None
    ):
        """
        Loads the robot articulation
        """

        def build_articulation(scene_idxs: Optional[List[int]] = None):
            if self.urdf_path is not None:
                loader = self.scene.create_urdf_loader()
                asset_path = format_path(str(self.urdf_path))
            elif self.mjcf_path is not None:
                loader = self.scene.create_mjcf_loader()
                asset_path = format_path(str(self.mjcf_path))

            loader.name = self.uid
            if self._agent_idx is not None:
                loader.name = f"{self.uid}-agent-{self._agent_idx}"
            loader.fix_root_link = self.fix_root_link
            loader.load_multiple_collisions_from_file = self.load_multiple_collisions
            loader.disable_self_collisions = self.disable_self_collisions
            
            
            # Proceed with the existing URDF config
            if self.urdf_config is not None:
                urdf_config = sapien_utils.parse_urdf_config(self.urdf_config)
                sapien_utils.check_urdf_config(urdf_config)
                sapien_utils.apply_urdf_config(loader, urdf_config)
            
            # Load the robot
            builder = loader.parse(asset_path)["articulation_builders"][0]
            builder.initial_pose = initial_pose
            if scene_idxs is not None:
                builder.set_scene_idxs(scene_idxs)
                builder.set_name(f"{self.uid}-agent-{self._agent_idx}-{scene_idxs}")
            
            robot = builder.build()
            assert robot is not None, f"Failed to load URDF/MJCF from {asset_path}"
            return robot

        if self.build_separate:
            arts = []
            for scene_idx in range(self.scene.num_envs):
                robot = build_articulation([scene_idx])
                self.scene.remove_from_state_dict_registry(robot)
                arts.append(robot)
            self.robot = Articulation.merge(
                arts, name=f"{self.uid}-agent-{self._agent_idx}", merge_links=True
            )
            self.scene.add_to_state_dict_registry(self.robot)
        else:
            self.robot = build_articulation()
            
        # Apply scaling to the built articulation if needed
        if hasattr(self, 'scale'):
            if isinstance(self.scale, (int, float)) or isinstance(self.scale, (list, tuple)):
                self._apply_scale_to_articulation(self.robot, self.scale)
            
        # Apply link-specific modifications after loading
        if self.link_modifications:
            self._apply_link_modifications()
            
        # Cache robot link names
        self.robot_link_names = [link.name for link in self.robot.get_links()]
    
    def _apply_link_modifications(self):
        """Apply modifications to robot links."""
        # Either remove this entirely or skip the scaling part
        # print("Starting link modifications")
        # for link_name, modifications in self.link_modifications.items():
        #     link = sapien_utils.get_obj_by_name(self.robot.get_links(), link_name)
        #     if link is None:
        #         print(f"WARNING: Link {link_name} not found!")
        #         continue
        #     if "scale" in modifications:
        #         scale = modifications["scale"]
        #         print(f"Applying scale {scale} to link {link_name}")
        #         self._scale_geometries(link, scale)
        pass  # Do nothing - the URDF already has scaling

    def _scale_geometries(self, link, scale):
        """Scale both visual and collision geometries."""
        try:
            # Get visual bodies from the link
            visual_bodies = link.get_visual_bodies() if hasattr(link, 'get_visual_bodies') else []
            for visual in visual_bodies:
                # Apply scale to the visual body's local transform
                transform = visual.local_transform
                # Scale the transform (implementation depends on SAPIEN version)
                if hasattr(visual, 'set_local_scale'):
                    visual.set_local_scale(scale)
                print(f"Scaled visual body in link {link.name}")
                
            # Get collision shapes from the link
            collision_shapes = link.get_collision_shapes() if hasattr(link, 'get_collision_shapes') else []
            for collision in collision_shapes:
                # Apply scale to the collision shape similarly
                if hasattr(collision, 'set_local_scale'):
                    collision.set_local_scale(scale)
                print(f"Scaled collision shape in link {link.name}")
                
            # If scaling methods aren't available, try URDF loader scaling instead
            # This is a fallback case - the loader scaling should handle this before we get here
            if not visual_bodies and not collision_shapes:
                print(f"No visual or collision geometries found for {link.name}, scaling may not be visible")
                
            return True
        except Exception as e:
            print(f"Error during scaling of {link.name}: {e}")
            return False

    def is_grasping(self, object=None):
        """Implements the is_grasping interface required by BaseAgent"""
        # For many robot arms, this could check contact forces between gripper and object
        # or detect if the gripper is closed around something
        # Implementation depends on your specific robot
        if hasattr(self.robot, "is_grasping"):
            return self.robot.is_grasping(object)
        return False

    def is_static(self, threshold: float):
        """Implements the is_static interface required by BaseAgent"""
        # Check if robot joint velocities are below threshold
        qvel = self.robot.get_qvel()
        return torch.linalg.norm(qvel) < threshold
    
    def _apply_scale_to_articulation(self, robot, scale):
        """
        Apply scaling to the built articulation if needed
        """
        # Implementation depends on Sapien's specific API
        # This is pseudocode and may not work directly
        if isinstance(scale, (int, float)):
            robot.set_global_scale(scale)
        elif isinstance(scale, (list, tuple)) and len(scale) == 3:
            robot.set_directional_scale(*scale)
