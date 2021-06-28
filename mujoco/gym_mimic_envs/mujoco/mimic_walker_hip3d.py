from os.path import join, dirname
from gym_mimic_envs.mimic_env import MimicEnv
import drloco.ref_trajecs.loco3d_trajecs as refs
from drloco.ref_trajecs.straight_walk_hip3d_trajecs import StraightWalking3dHipTrajectories
from gym_mimic_envs.mujoco.mimic_walker3d import MimicWalker3dEnv, qpos_indices, qvel_indices

# specify which joint trajectories are required for the current walker
walker_qpos_indices = [refs.PELVIS_TX, refs.PELVIS_TZ, refs.PELVIS_TY,
                       refs.LUMBAR_BENDING, refs.LUMBAR_EXTENSION, refs.PELVIS_ROTATION,
                       refs.HIP_FLEXION_R, refs.HIP_ADDUCTION_R, refs.HIP_ROTATION_R,
                       refs.KNEE_ANG_R, refs.ANKLE_ANG_R,
                       refs.HIP_FLEXION_L, refs.HIP_ADDUCTION_L, refs.HIP_ROTATION_L,
                       refs.KNEE_ANG_L, refs.ANKLE_ANG_L]

adaptations = {
    refs.PELVIS_TY: -1,
    refs.PELVIS_TY: 1.65 / 1.75,
    refs.PELVIS_ROTATION: -1,
    refs.KNEE_ANG_R: -1,
    refs.KNEE_ANG_L: -1,
    refs.HIP_FLEXION_R: -1,
    refs.HIP_FLEXION_L: -1,
    refs.HIP_ADDUCTION_L: -1,
    refs.HIP_ROTATION_L: -1
}

# the indices in the joint position and joint velocity matrices are the same for all joints
walker_qvel_indices = walker_qpos_indices

class MimicWalker3dHipEnv(MimicEnv):
    def __init__(self):
        # specify the name of the environment XML file
        walker_xml = 'walker3d_hip3d.xml'
        # init reference trajectories
        # by specifying the indices in the mocap data to use for qpos and qvel
        ref_trajecs = refs.Loco3dReferenceTrajectories(
            walker_qpos_indices, walker_qvel_indices, adaptations)
        # specify absolute path to the MJCF file
        mujoco_xml_file = join(dirname(__file__), "assets", walker_xml)
        # init the mimic environment
        MimicEnv.__init__(self, mujoco_xml_file, ref_trajecs)

    # ----------------------------
    # Methods we override:
    # ----------------------------

    def _get_COM_indices(self):
        return [0,1,2]

    def _get_trunk_rot_joint_indices(self):
        return [3, 4, 5]

    def _get_not_actuated_joint_indices(self):
        return self._get_COM_indices() + [3,4,5]