import numpy as np
import scipy.io as spio
from scripts.common.utils import get_absolute_project_path
from scripts.ref_trajecs.base_ref_trajecs import BaseReferenceTrajectories

# create index constants for easier access to specific reference trajectory parts
PELVIS_TILT, PELVIS_LIST, PELVIS_ROTATION = range(0,3)
PELVIS_TX, PELVIS_TY, PELVIS_TZ = range(3,6)
HIP_FLEXION_R, HIP_ADDUCTION_R, HIP_ROTATION_R = range(6, 9)
KNEE_ANG_R, ANKLE_ANG_R = range(9, 11)
SUBTALAR_ANG_R, MTP_ANG_R = range(11, 13)
HIP_FLEXION_L, HIP_ADDUCTION_L, HIP_ROTATION_L = range(13, 16)
KNEE_ANG_L, ANKLE_ANG_L = range(16, 18)
SUBTALAR_ANG_L, MTP_ANG_L = range(18, 20)
LUMBAR_EXTENSION, LUMBAR_BENDING, LUMBAR_ROTATION = range(20, 23)
ARM_FLEX_R, ARM_ADD_R, ARM_ROT_R = range(23, 26)
ELBOW_FLEX_R, PRO_SUP_R, WRIST_FLEX_R, WRIST_DEV_R = range(26, 30)
ARM_FLEX_L, ARM_ADD_L, ARM_ROT_L = range(30, 33)
ELBOW_FLEX_L, PRO_SUP_L, WRIST_FLEX_L, WRIST_DEV_L = range(33, 37)


class Loco3dReferenceTrajectories(BaseReferenceTrajectories):
    def __init__(self, qpos_indices, qvel_indices, adaptations):
        rel_mocap_data_path = 'assets/mocaps/loco3d/loco3d_guoping.mat'
        super(Loco3dReferenceTrajectories, self).__init__(rel_mocap_data_path, 500, 100,
                                                          qpos_indices, qvel_indices,
                                                          adaptations=adaptations)
        self._pos = 1000

    def _load_ref_trajecs(self):
        dir_path = get_absolute_project_path()
        file_path = 'assets/mocaps/loco3d/loco3d_guoping.mat'

        data = spio.loadmat(dir_path + file_path, squeeze_me=True)

        # labels of the individual dimensions/rows in the mocap data matrix
        kin_labels = data['rowNameIK']
        angles = data['angJoi']
        ang_vels = data['angDJoi']
        return angles, ang_vels

    def _get_COM_Z_pos_index(self):
        return PELVIS_TY

    def get_desired_walking_velocity_vector(self, do_eval):
        # during evaluation, let the agent walk just straight.
        # This way, we can retain our current evaluation metrics.
        return [1.2, 0] if do_eval else self.get_qvel()[:2]
