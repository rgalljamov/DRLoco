import numpy as np
import gym
from os.path import join, dirname
from gym.envs.mujoco import mujoco_env
from gym_mimic_envs.mimic_env import MimicEnv

from scripts import config_light as cfgl
from scripts.common import config as cfg
from scripts.mocap import ref_trajecs as refs

# pause sim on startup to be able to change rendering speed, camera perspective etc.
pause_mujoco_viewer_on_start = True

# qpos and qvel indices for quick access to the reference trajectories
qpos_indices = [refs.COM_POSX, refs.COM_POSY, refs.COM_POSZ,
                refs.TRUNK_ROT_X, refs.TRUNK_ROT_Y, refs.TRUNK_ROT_Z,
                refs.HIP_SAG_ANG_R, refs.HIP_FRONT_ANG_R,
                refs.KNEE_ANG_R, refs.ANKLE_ANG_R,
                refs.HIP_SAG_ANG_L, refs.HIP_FRONT_ANG_L,
                refs.KNEE_ANG_L, refs.ANKLE_ANG_L]

qvel_indices = [refs.COM_VELX, refs.COM_VELY, refs.COM_VELZ,
                refs.TRUNK_ANGVEL_X, refs.TRUNK_ANGVEL_Y, refs.TRUNK_ANGVEL_Z,
                refs.HIP_SAG_ANGVEL_R, refs.HIP_FRONT_ANGVEL_R,
                refs.KNEE_ANGVEL_R, refs.ANKLE_ANGVEL_R,
                refs.HIP_SAG_ANGVEL_L, refs.HIP_FRONT_ANGVEL_L,
                refs.KNEE_ANGVEL_L, refs.ANKLE_ANGVEL_L]

ref_trajec_adapts = {}

class MimicWalker3dEnv(MimicEnv):
    '''
    The 2D Mujoco Walker from OpenAI Gym extended to match
    the 3D bipedal walker model from Guoping Zhao.
    '''

    def __init__(self):
        # specify the name of the environment XML file
        walker_xml = cfgl.WALKER_MJC_XML_FILE
        # init reference trajectories
        # by specifying the indices in the mocap data to use for qpos and qvel
        global qpos_indices, qvel_indices
        reference_trajectories = refs.ReferenceTrajectories(qpos_indices, qvel_indices)
        # specify absolute path to the MJCF file
        mujoco_xml_file = join(dirname(__file__), "assets", walker_xml)
        # init the mimic environment
        MimicEnv.__init__(self, mujoco_xml_file, reference_trajectories)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

        # ----------------------------
        # Methods we override:
        # ----------------------------

    def _get_COM_indices(self):
        """
        Needed to distinguish between joint and COM kinematics.

        Returns a list of indices pointing at COM joint position/index
        in the considered robot model, e.g. [0,1,2]
        """
        return [0,1,2]

    def _get_trunk_rot_joint_indices(self):
        return [3, 4, 5]

    def _get_not_actuated_joint_indices(self):
        """
        Needed for playing back reference trajectories
        by using position servos in the actuated joints.

        @returns a list of indices specifying indices of
        joints in the considered robot model that are not actuated.
        Example: return [0,1,2]
        """
        return self._get_COM_indices() + [3,4,5]

    def _get_max_actuator_velocities(self):
        """Maximum joint velocities approximated from the reference data."""
        return np.array([5, 1, 10, 10, 5, 1, 10, 10])

    def has_ground_contact(self):
        has_contact = [False, False]
        for contact in self.data.contact[:self.data.ncon]:
            if contact.geom1 == 0 and contact.geom2 == 4:
                # right foot has ground contact
                has_contact[1] = True
            elif contact.geom1 == 0 and contact.geom2 == 7:
                # left foot has ground contact
                has_contact[0] = True

        if cfg.is_mod(cfg.MOD_3_PHASES):
            double_stance = all(has_contact)
            if cfg.is_mod(cfg.MOD_GRND_CONTACT_ONE_HOT):
                if double_stance:
                    return [False, False, True]
                else:
                    has_contact += [False]
            else: has_contact + [double_stance]

        # when both feet have no ground contact
        if cfg.is_mod(cfg.MOD_GROUND_CONTACT_NNS) and not any(has_contact):
            # print('Both feet without ground contact!')
            # let the left and right foot network handle this situation
            has_contact = np.array(has_contact)
            has_contact[:2] = True

        return has_contact