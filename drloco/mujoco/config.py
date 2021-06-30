from drloco.mujoco.mimic_walker3d import MimicWalker3dEnv
from drloco.mujoco.mimic_walker_165cm_65kg import MimicWalker165cm65kgEnv

# map environment ids to the corresponding classes
env_map = {'StraightMimicWalker': MimicWalker3dEnv,
           'MimicWalker165cm65kg': MimicWalker165cm65kgEnv}