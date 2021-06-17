from os.path import join, dirname
from gym_mimic_envs.mimic_env import MimicEnv
import scripts.ref_trajecs.loco3d_trajecs as refs
from scripts.ref_trajecs.straight_walk_hip3d_trajecs import StraightWalking3dHipTrajectories
from gym_mimic_envs.mujoco.mimic_walker3d import MimicWalker3dEnv, qpos_indices, qvel_indices

# specify which joint trajectories are required for the current walker
walker_qpos_indices = [refs.PELVIS_TX, refs.PELVIS_TZ, refs.PELVIS_TY,
                       refs.PELVIS_LIST, refs.PELVIS_TILT, refs.PELVIS_ROTATION,
                       refs.LUMBAR_BENDING, refs.LUMBAR_EXTENSION, refs.LUMBAR_ROTATION,
                       refs.HIP_FLEXION_R, refs.HIP_ADDUCTION_R, refs.HIP_ROTATION_R,
                       refs.KNEE_ANG_R, refs.ANKLE_ANG_R,
                       refs.HIP_FLEXION_L, refs.HIP_ADDUCTION_L, refs.HIP_ROTATION_L,
                       refs.KNEE_ANG_L, refs.ANKLE_ANG_L]

# empty, as model representing the same subject the mocap data was collected from!
adaptations = {}

# the indices in the joint position and joint velocity matrices are the same for all joints
walker_qvel_indices = walker_qpos_indices

class MimicWalker165cm65kgEnv(MimicEnv):
    def __init__(self):
        # specify the name of the environment XML file
        walker_xml = 'walker_165cm_65kg.xml'
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

    def get_joint_indices_for_phase_estimation(self):
        # return both hip joint indices in the saggital plane
        # and both knee joint indices in the saggital plane
        return [9,12,14,17]

    def _get_not_actuated_joint_indices(self):
        return self._get_COM_indices() + [3,4,5]