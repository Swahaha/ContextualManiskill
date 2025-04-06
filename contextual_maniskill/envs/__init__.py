"""
Registering custom environments so they can be instantiated with gym.make
"""

# Import the environment classes to trigger the @register_env decorator
from contextual_maniskill.envs.contextual_pickcube import ContextualPickCubeEnv 