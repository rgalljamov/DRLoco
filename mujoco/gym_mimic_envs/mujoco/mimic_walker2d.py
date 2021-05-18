import numpy as np
from gym import utils
from os.path import join, dirname
from gym.envs.mujoco import mujoco_env
from gym_mimic_envs.mimic_env import MimicEnv
from scripts.common.utils import is_remote
from scripts.mocap import ref_trajecs as refs
import scripts.common.config as cfg
from scripts.mocap.ref_trajecs import ReferenceTrajectories

# qpos and qvel indices for quick access to the reference trajectories
qpos_indices = [refs.COM_POSX, refs.COM_POSZ, refs.TRUNK_ROT_Y,
                refs.HIP_SAG_ANG_R, refs.KNEE_ANG_R, refs.ANKLE_ANG_R,
                refs.HIP_SAG_ANG_L, refs.KNEE_ANG_L, refs.ANKLE_ANG_L]

qvel_indices = [refs.COM_VELX, refs.COM_VELZ, refs.TRUNK_ANGVEL_Y,
                refs.HIP_SAG_ANGVEL_R,
                refs.KNEE_ANGVEL_R, refs.ANKLE_ANGVEL_R,
                refs.HIP_SAG_ANGVEL_L,
                refs.KNEE_ANGVEL_L, refs.ANKLE_ANGVEL_L]

# adaptations needed to account for different body shape
# and axes definitions in the reference trajectories
ref_trajec_adapts = {}

class MimicWalker2dEnv(MimicEnv, mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Building upon the Walker2d-v2 Environment with the id: Walker2d-v2
    """
    def __init__(self):
        walker_xml = {'mim2d': 'walker2pd.xml',
                      'mim_trq2d': 'walker2d.xml'}[cfg.env_abbrev]
        mujoco_env.MujocoEnv.__init__(self,
                                      join(dirname(__file__), "assets", walker_xml), 4)
        utils.EzPickle.__init__(self)
        # init the mimic environment, automatically loads and inits ref trajectories
        MimicEnv.__init__(self,
                          ReferenceTrajectories(qpos_indices, qvel_indices, ref_trajec_adapts))

    @staticmethod
    def get_refs(reset=False):
        refs = ReferenceTrajectories(qpos_indices, qvel_indices, ref_trajec_adapts)
        if reset: refs.reset()
        return refs

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    # ----------------------------
    # Methods we override:
    # ----------------------------

    def _get_COM_indices(self):
        return [0,1] # x, z

    def _get_trunk_rot_joint_indices(self):
        return [2]

    def _get_not_actuated_joint_indices(self):
        return [0,1,2]

    def _get_max_actuator_velocities(self):
        """Maximum joint velocities approximated from the reference data."""
        return np.array([5, 10, 10, 5, 10, 10])

    def has_ground_contact(self):
        has_contact = [False, False]
        for contact in self.data.contact[:10]:
            if contact.geom1 == 0 and contact.geom2 == 4:
                # right foot has ground contact
                has_contact[1] = True
            elif contact.geom1 == 0 and contact.geom2 == 7:
                # left foot has ground contact
                has_contact[0] = True
        if cfg.is_mod(cfg.MOD_3_PHASES):
            has_contact += [all(has_contact)]
        return has_contact


