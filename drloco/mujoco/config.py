from drloco.mujoco.mimic_walker3d import MimicWalker3dEnv
from drloco.mujoco.mimic_walker_165cm_65kg import MimicWalker165cm65kgEnv

# specify environment ids
straight_walker = 'StraightMimicWalker'
hip3d_2seg_upper_body_walker = 'MimicWalker165cm65kg'

# map environment ids to the corresponding classes
env_map = {straight_walker: MimicWalker3dEnv,
           hip3d_2seg_upper_body_walker: MimicWalker165cm65kgEnv}

# specify the frequency the simulation is running at [Hz]
sim_freqs = {straight_walker: 1000,
             hip3d_2seg_upper_body_walker: 1000}