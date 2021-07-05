import numpy as np
import scipy.io as spio
from drloco.common.utils import get_project_path
from drloco.ref_trajecs.base_ref_trajecs import BaseReferenceTrajectories

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
        # the mocaps were sampled with 500Hz
        sampling_frequency = 500
        # for control frequency, use the one specified in the config file
        from drloco.config.config import CTRL_FREQ
        control_frequency = CTRL_FREQ
        # initialize the base class
        super(Loco3dReferenceTrajectories, self).__init__(sampling_frequency,
                                                          control_frequency,
                                                          qpos_indices, qvel_indices,
                                                          adaptations=adaptations)

    def _load_ref_trajecs(self):
        dir_path = get_project_path()
        file_path = 'mocaps/loco3d/loco3d_guoping.mat'

        data = spio.loadmat(dir_path + file_path, squeeze_me=True)

        # labels of the individual dimensions/rows in the mocap data matrix
        # todo: angles and ang_vels are returned, but qlabels are set in the function
        self._qlabels = data['rowNameIK']
        angles = data['angJoi']
        ang_vels = data['angDJoi']
        return angles, ang_vels

    def _get_COM_Z_pos_index(self):
        return PELVIS_TY

    def get_desired_walking_velocity_vector(self, do_eval, debug=False):
        if False and do_eval:
            # during evaluation, let the agent walk just straight.
            # This way, we can retain our current evaluation metrics.
            return [1.2, 0.0]

        # get the average velocities in x and z directions
        # average over n seconds
        n_seconds = 0.5
        n_timesteps = int(n_seconds * self._sample_freq)
        # consider the reference trajectory has a maximum length
        end_pos = min(self._pos + n_timesteps, self._trajec_len-1)
        qvels_x = self._qvel_full[PELVIS_TX, self._pos: end_pos]
        # NOTE: y direction in the simulation corresponds to z direction in the mocaps
        qvels_y = self._qvel_full[PELVIS_TZ, self._pos: end_pos]
        # get the mean velocities
        mean_x_vel = np.mean(qvels_x)
        mean_y_vel = np.mean(qvels_y)
        if debug:
            try:
                self.des_vels_x += [mean_x_vel]
                self.des_vels_y += [mean_y_vel]
                # calculate the walker position by integrating the velocity vector
                self.xpos += mean_x_vel * 1/self._control_freq
                self.ypos += mean_y_vel * 1/self._control_freq
                self.xposs += [self.xpos]
                self.yposs += [self.ypos]
            except:
                self.des_vels_x = [mean_x_vel]
                self.des_vels_y = [mean_y_vel]
                self.xpos, self.ypos = 0, 0
                self.xposs = [self.xpos]
                self.yposs = [self.ypos]

            if len(self.des_vels_x) > 1000:
                from matplotlib import pyplot as plt
                fig, subs = plt.subplots(1,4)
                subs[0].plot(self.des_vels_x)
                subs[1].plot(self.des_vels_y)
                subs[2].plot(self.des_vels_x, self.des_vels_y)
                subs[3].plot(self.xposs, self.yposs)
                for i in range(3):
                    subs[i].set_title('Desired Velocity\nin {} direction'.format(['X', 'Y', 'X and Y'][i]))
                subs[3].set_title('Walkers position\n(from mean des_vel)')
                plt.show()
                exit(33)
        return [mean_x_vel, mean_y_vel]
