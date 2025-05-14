from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import contextual_maniskill.envs.contextual_pickcube
import contextual_maniskill.envs.contextual_push_t

def combine_successful_episodes(run_name):
    """
    Combine all successful episodes into a single dataset
    """
    episodes_dir = f"runs/{run_name}/successful_episodes"
    if not os.path.exists(episodes_dir):
        return
    
    all_params = []
    all_trajectories = []
    all_tasks = []
    
    for episode_file in os.listdir(episodes_dir):
        if episode_file.endswith('.pt'):
            episode_data = torch.load(os.path.join(episodes_dir, episode_file))
            all_params.append(episode_data['param'])
            all_trajectories.append(episode_data['traj'])
            all_tasks.append(episode_data['task'])
    
    if all_params:
        dataset = {
            'param': torch.stack(all_params),
            'traj': torch.stack(all_trajectories),
            'task': torch.stack(all_tasks)
        }
        
        # Save combined dataset
        dataset_path = f"runs/{run_name}/successful_episodes_dataset.pt"
        torch.save(dataset, dataset_path)
        print(f"Combined dataset saved to {dataset_path}")

combine_successful_episodes("ContextualPickCube-v1__make-an-agent_ppo__1__1747081709")
