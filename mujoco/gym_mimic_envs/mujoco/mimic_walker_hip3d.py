from os.path import join, dirname
from gym_mimic_envs.mimic_env import MimicEnv
from scripts.ref_trajecs.straight_walk_hip3d_trajecs import StraightWalking3dHipTrajectories
from gym_mimic_envs.mujoco.mimic_walker3d import MimicWalker3dEnv, qpos_indices, qvel_indices

class MimicWalker3dHipEnv(MimicWalker3dEnv):
    def __init__(self):
        # super(MimicWalker3dHipEnv, self).__init__()
        # specify the name of the environment XML file
        walker_xml = 'walker3d_hip3d.xml'
        # init reference trajectories
        # by specifying the indices in the mocap data to use for qpos and qvel
        reference_trajectories = StraightWalking3dHipTrajectories(qpos_indices, qvel_indices)
        # specify absolute path to the MJCF file
        mujoco_xml_file = join(dirname(__file__), "assets", walker_xml)
        # init the mimic environment
        MimicEnv.__init__(self, mujoco_xml_file, reference_trajectories)