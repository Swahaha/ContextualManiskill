# ContextualManiSkill: HyperPPO for Tabletop Tasks

This repository implements **HyperPPO** (a Graph HyperNetwork-based PPO agent) in the [ManiSkill](https://github.com/haosulab/ManiSkill) framework, focusing on **tabletop manipulation tasks**. It is designed for research in robot learning, meta-architecture RL, and environment/robot randomization.

## Key Features
- **HyperPPO Integration**: Meta-architecture RL agent using Graph HyperNetworks for policy learning.
- **Dynamic URDF Configuration**: Scale robot joints and arm lengths (e.g., Panda link5) at runtime for rapid testing and ablation of RL policies on different robot morphologies.
- **Contextual Environments**: Custom tabletop tasks (e.g., PickCube, PushT) with scene and robot randomization.
- **ManiSkill Compatibility**: Leverages ManiSkill's simulation, robot models, and environment registration.

## Installation
First, install ManiSkill (required):

```bash
pip install --upgrade mani_skill
```

## Conda Environment (Recommended)
You can also set up all dependencies using the provided `environment.yml` file. This will create a conda environment named `hyper`:

```bash
conda env create -f environment.yml
conda activate hyper
```

## How to Run
Example command to train a policy (from `Notes.txt`):

```bash
# (Optional) Set CUDA device -> important for multi GPU setups
export CUDA_VISIBLE_DEVICES=3

# Run training (example for PickCube)
nohup python Contextual_hyperppo.py --env_id="ContextualPickCube-v1" > training_log.txt 2>&1 &
```
- Replace `--env_id` with your desired environment (e.g., `ContextualPushT-v1`).
- Adjust other arguments as needed for your experiment.

## Project Structure
- `contextual_maniskill/`
  - `envs/`: Contextual tabletop environments (e.g., PickCube, PushT)
  - `agents/`: Custom agents (e.g., Panda, PandaStick) with dynamic URDF scaling
  - `utils/`: Scene builders and utilities for environment setup
  - `panda/`: Robot model files (URDF, SRDF, assets) -> just for context, not directly used
- `hyperppo_model/`: HyperPPO agent and supporting modules
- `Contextual_hyperppo.py`, `mani_hyperppo.py`, etc.: Training and evaluation scripts
- `Context_envs/`: The original push_t benchmark code for context

## Notes
- Training logs, checkpoints, and videos are saved in the project directory.
- The dynamic URDF scaling is controlled via script arguments (e.g., `--panda_link5_z_scale`).
- See the code and scripts for more details on available options and customization.
