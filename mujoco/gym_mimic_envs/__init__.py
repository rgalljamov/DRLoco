from gym.envs.registration import register

register(
    id='MimicWalker2d-v0',
    entry_point='gym_mimic_envs.mujoco:MimicWalker2dEnv'
)

register(
    id='MimicWalker3d-v0',
    entry_point='gym_mimic_envs.mujoco:MimicWalker3dEnv'
)

"""
import gym

# Solves the bug: " Cannot re-register id: MimicWalker2d-v0"
# Thanks @rhalbersma from https://github.com/mpSchrader/gym-sokoban/issues/29#issuecomment-612801318
def register(id, entry_point, force=True):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.envs.register(
        id=id,
        entry_point=entry_point,
    )

# Register modified versions of existing environments
register(
    id='MimicWalker2d-v0',
    entry_point='gym_mimic_envs.mujoco:MimicWalker2d'
)
"""