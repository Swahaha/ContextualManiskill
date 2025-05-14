import torch
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from hyperppo import HyperAgent

import contextual_maniskill.envs.contextual_pickcube


# === Config ===
arch_to_eval = [64,64,64]
num_envs = 8
max_steps = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "final_cpt.pt"

# === Environment ===
env = gym.make(
    "ContextualPickCube-v1",
    num_envs=num_envs,
    obs_mode="state",
    render_mode="rgb_array",
    sim_backend="physx_cuda",
    robot_uids="contextual_panda",
    panda_link5_z_scale=2.0
)

env = RecordEpisode(
    env,
    output_dir="",  # no saving here
    save_trajectory=False,
    save_video=True,
    max_steps_per_video=max_steps,
    video_fps=30,
)

env = RecordEpisode(
    env,
    output_dir="",  # no saving here
    save_trajectory=False,
    save_video=True,
    max_steps_per_video=max_steps,
    video_fps=30,
)

if isinstance(env.action_space, Dict):
    env = FlattenActionSpaceWrapper(env)

env = ManiSkillVectorEnv(env, num_envs=num_envs, ignore_terminations=False, record_metrics=True)

# === Agent ===
agent = HyperAgent(env).to(device)
agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
agent.eval()

# === Architecture Index ===
arc_list = agent.hyper_actor.list_of_arcs
try:
    idx = arc_list.index(tuple(arch_to_eval))
except ValueError:
    raise ValueError(f"Architecture {arch_to_eval} not found in list_of_arcs!")

# === Setup GHN with chosen architecture ===
shape_ind = agent.hyper_actor.list_of_shape_inds[idx].unsqueeze(0)
agent.hyper_actor.set_graph(indices_vector=np.array([idx]), shape_ind_vec=shape_ind)

# === Run Batched Evaluation ===
obs, _ = env.reset(seed=42)
total_rewards = torch.zeros(num_envs, device=device)
done_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)

for _ in range(max_steps):
    with torch.no_grad():
        action = agent.hyper_actor.get_det_action(obs.to(device))
    obs, reward, terminated, truncated, info = env.step(action)

    active = ~(terminated | truncated)
    reward[~done_flags] *= active[~done_flags]  # mask post-done rewards
    total_rewards += reward.to(device)
    done_flags |= terminated | truncated

    if done_flags.all():
        break

env.close()
torch.cuda.empty_cache()

# === Report Mean Reward ===
mean_reward = total_rewards.mean().item()
print(f"\n✅ Architecture {arch_to_eval} | Reward Mean across {num_envs} envs: {mean_reward:.2f}")
