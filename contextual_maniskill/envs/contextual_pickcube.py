from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
# from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


# from contextual_maniskill.agents.base_agent import ContextualBaseAgent
from contextual_maniskill.utils.scene_builder.table.contexutual_scene_builder import ContextualTableSceneBuilder
from contextual_maniskill.agents.contextual_panda import ContextualPanda

@register_env("ContextualPickCube-v1", max_episode_steps=50)
class ContextualPickCubeEnv(BaseEnv):
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="contextual_panda", robot_init_qpos_noise=0.02, **kwargs):
        self.panda_link5_z_scale = kwargs.pop("panda_link5_z_scale", 1.0)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        print("Initializing ContextualPickCubeEnv...")
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.8, 0.9, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict, initial_agent_poses: Optional[Union[sapien.Pose, Pose]] = sapien.Pose(p=[-0.615, 0, 0]), build_separate: bool = False):
        """
        Loads a Panda or ContextualPanda robot with the specified link5_z_scale.
        """
        if self.robot_uids == "panda":
            self.agent = Panda(
                self.scene,
                self._control_freq,
                self._control_mode,
                agent_idx=None,
                initial_pose=initial_agent_poses,
                build_separate=build_separate,
            )
        elif self.robot_uids == "contextual_panda":
            self.agent = ContextualPanda(
                self.scene,
                self._control_freq,
                self._control_mode,
                agent_idx=None,
                initial_pose=initial_agent_poses,
                build_separate=build_separate,
                link5_z_scale=self.panda_link5_z_scale,
            )
        else:
            raise ValueError(f"Unsupported robot_uids: {self.robot_uids}")

    def _load_scene(self, options: dict):
        self.table_scene = ContextualTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        if self.robot_uids == "panda":
            qvel = qvel[..., :-2]
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel, axis=1)
        )
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
