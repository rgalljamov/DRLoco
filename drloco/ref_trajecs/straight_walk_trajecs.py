'''
Script to handle reference trajectories.
- Get trajectory for a single step
- Identify most appropriate next step
- Get only specified trajectory parts

- handle COM and Joint Kinematics separately
'''
import random
import numpy as np
import scipy.io as spio
from drloco.ref_trajecs.base_ref_trajecs import BaseReferenceTrajectories
from drloco.config.config import EVAL_N_TIMES
from drloco.common.utils import log, is_remote, config_pyplot, smooth_exponential, \
    get_project_path

# execute on my private PC or on the remote Lauflabor PC
REMOTE = is_remote()

# relative paths to the two available trajectories
PATH_CONSTANT_SPEED = 'mocaps/straight_walking/Trajecs_Constant_Speed_400Hz.mat'
PATH_SPEED_RAMP = 'mocaps/straight_walking/Trajecs_Ramp_Slow_400Hz_EulerTrunkAdded.mat'
# path to the reference trajectories to use
PATH_REF_TRAJECS = PATH_SPEED_RAMP

# is the trajectory with the constant speed chosen?
_is_constant_speed = PATH_CONSTANT_SPEED in PATH_REF_TRAJECS

# label every trajectory in the mocap data with the corresponding name
labels = ['COM Pos (X)', 'COM Pos (Y)', 'COM Pos (Z)',
          'Trunk Rot (quat,w)', 'Trunk Rot (quat,x)', 'Trunk Rot (quat,y)', 'Trunk Rot (quat,z)',
          'Ang Hip Frontal R', 'Ang Hip Sagittal R',
          'Ang Knee R', 'Ang Ankle R',
          'Ang Hip Frontal L', 'Ang Hip Sagittal L',
          'Ang Knee L', 'Ang Ankle L',

          'COM Vel (X)', 'COM Vel (Y)', 'COM Vel (Z)',
          'Trunk Ang Vel (X)', 'Trunk Ang Vel (Y)', 'Trunk Ang Vel (Z)',
          'Vel Hip Frontal R', 'Vel Hip Sagittal R',
          'Vel Knee R', 'Vel Ankle R',
          'Vel Hip Frontal L', 'Vel Hip Sagittal L',
          'Vel Knee L', 'Vel Ankle L',

          'Foot Pos L (X)', 'Foot Pos L (Y)', 'Foot Pos L (Z)',
          'Foot Pos R (X)', 'Foot Pos R (Y)', 'Foot Pos R (Z)',

          'GRF R', 'GRF L',

          'Trunk Rot (euler,x)', 'Trunk Rot (euler,y)', 'Trunk Rot (euler,z)',
          ]

if _is_constant_speed:
    labels.remove('GRF R')
    labels.remove('GRF L')

labels = np.array(labels)

# reference trajectory: joint position indices
COM_POSX, COM_POSY, COM_POSZ = range(0,3)
TRUNK_ROT_Q1, TRUNK_ROT_Q2, TRUNK_ROT_Q3, TRUNK_ROT_Q4 = range(3,7)
HIP_FRONT_ANG_R, HIP_SAG_ANG_R, KNEE_ANG_R, ANKLE_ANG_R = range(7,11)
HIP_FRONT_ANG_L, HIP_SAG_ANG_L, KNEE_ANG_L, ANKLE_ANG_L = range(11,15)


# reference trajectory: joint velocity indices
COM_VELX, COM_VELY, COM_VELZ = range(15,18)
TRUNK_ANGVEL_X, TRUNK_ANGVEL_Y, TRUNK_ANGVEL_Z = range(18,21)
HIP_FRONT_ANGVEL_R, HIP_SAG_ANGVEL_R, KNEE_ANGVEL_R, ANKLE_ANGVEL_R = range(21,25)
HIP_FRONT_ANGVEL_L, HIP_SAG_ANGVEL_L, KNEE_ANGVEL_L, ANKLE_ANGVEL_L = range(25,29)

# mirror right step to get left step
mirred_indices = [COM_POSX, COM_POSY, COM_POSZ,
                  TRUNK_ROT_Q1, TRUNK_ROT_Q2, TRUNK_ROT_Q3, TRUNK_ROT_Q4,
                  HIP_FRONT_ANG_L, HIP_SAG_ANG_L, KNEE_ANG_L, ANKLE_ANG_L,
                  HIP_FRONT_ANG_R, HIP_SAG_ANG_R, KNEE_ANG_R, ANKLE_ANG_R,
                  COM_VELX, COM_VELY, COM_VELZ,
                  TRUNK_ANGVEL_X, TRUNK_ANGVEL_Y, TRUNK_ANGVEL_Z,
                  HIP_FRONT_ANGVEL_L, HIP_SAG_ANGVEL_L, KNEE_ANGVEL_L, ANKLE_ANGVEL_L,
                  HIP_FRONT_ANGVEL_R, HIP_SAG_ANGVEL_R, KNEE_ANGVEL_R, ANKLE_ANGVEL_R]


# reference trajectory: foot position and GRF indices
FOOT_POSX_L, FOOT_POSY_L, FOOT_POSZ_L, FOOT_POSX_R, FOOT_POSY_R, FOOT_POSZ_R = range(29,35)
mirred_indices += [FOOT_POSX_R, FOOT_POSY_R, FOOT_POSZ_R, FOOT_POSX_L, FOOT_POSY_L, FOOT_POSZ_L]
if _is_constant_speed:
    TRUNK_ROT_X, TRUNK_ROT_Y, TRUNK_ROT_Z = range(35, 38)
    mirred_indices += [TRUNK_ROT_X, TRUNK_ROT_Y, TRUNK_ROT_Z]
else:
    GRF_R, GRF_L = range(35, 37)
    TRUNK_ROT_X, TRUNK_ROT_Y, TRUNK_ROT_Z = range(37, 40)
    mirred_indices += [GRF_L, GRF_R, TRUNK_ROT_X, TRUNK_ROT_Y, TRUNK_ROT_Z]

negate_indices = [COM_POSY, TRUNK_ROT_X, TRUNK_ROT_Z, HIP_FRONT_ANG_R, HIP_FRONT_ANG_L,
                  COM_VELY, TRUNK_ANGVEL_X, TRUNK_ANGVEL_Z, HIP_FRONT_ANGVEL_R, HIP_FRONT_ANGVEL_L]


class StraightWalkingTrajectories(BaseReferenceTrajectories):
    def __init__(self, qpos_indices, q_vel_indices, adaptations={}, mirror_refs=False):
        """
            :param mirror_refs: if True, mirror the right step trajectories
            and use them for the left steps to get perfectly symmetric walking.
        """
        super(StraightWalkingTrajectories, self).__init__(
            400, 200, qpos_indices, q_vel_indices, data_labels=labels)

        # setup pyplot
        self.plt = config_pyplot(fig_size=True)
        self.reset()
        # calculate walking speeds for each step
        self.step_velocities = self._calculate_walking_speed()
        # calculated and added trunk euler rotations
        # self._add_trunk_euler_rotations()
        # some steps are done with left, some with right foot
        self.left_step_indices = self._determine_left_steps_indices()
        # mirror right step and use it as left step
        if mirror_refs: self._mirror_refs()
        # current step
        self._step = np.concatenate([self._qpos_full, self._qvel_full], axis=0)
        # distance walked so far (COM X Position)
        self.dist = 0
        # flag to indicate the last step in the refs was reached
        self.has_reached_last_step = False
        # count how many steps were taken without skipping steps
        self.count_steps_same_vel = 1
        # during evaluation we want our agent to start from different positions
        self.n_deterministic_inits = 0

    def _mirror_refs(self):
        # print('Mirroring the mocap data to have symmetric walking!')
        right_step_indices = np.array(self.left_step_indices) - 1
        # replace left steps with right steps
        self.data[self.left_step_indices] = self.data[right_step_indices]
        for i_left_step in self.left_step_indices:
            # steps at left index are right steps, but not mirrored yet
            right_step = self.data[i_left_step]
            mirred_right_step = right_step[mirred_indices, :]
            # some trajectories maintain their value but have to be negated
            mirred_right_step[negate_indices, :] *= -1
            self.data[i_left_step] = mirred_right_step

    def next(self):
        """
        Increases the internally managed position
                on the current step trajectory by a specified amount.
        :param increment number of points to proceed on the ref trajecs
                         increment=2 corresponds to 200Hz sample frequency
        """
        self._pos += self._increment
        self._ep_dur += 1
        # when we reached the trajectory's end of the current step
        dif = self._pos - self._trajec_len + 1
        if dif > 0:
            # choose the next step
            self._qpos_full = self._get_next_step()
            self._qvel_full = self._qpos_full
            # update the trajec length as it is different for each step
            self._trajec_len = self._qpos_full.shape[1]
            # make sure to do the required increment
            self._pos = dif

    def reset(self):
        """ Set all indices and counters to zero."""
        self._i_step = 0
        self._step = self.data[0]
        self._qpos_full = self._step
        self._qvel_full = self._step
        self._trajec_len = self._step.shape[1]
        self._pos = 0
        self.dist = 0
        self._ep_dur = 0
        self.has_reached_last_step = False

    def get_phase_variable(self):
        phase = self._pos / self._trajec_len
        if not (phase >= 0 and phase <= 1):
           print(f'Phase Variable should be between 0 and 1 but was {phase}')
        return phase

    def get_kinematic_ranges(self):
        '''Returns the maximum range of qpos and qvel in reference trajecs.'''
        return self.ranges[self.qpos_is], self.ranges[self.qvel_is]

    def get_labels_by_model_index(self, pos_rel_is, vel_rel_is):
        '''@returns: the names/labels of the corresponding kinematics
           given their relative index.
           @params: both index lists are relative to qpos_is and qvel_is'''
        global labels
        pos_is = np.array(self.qpos_is)[pos_rel_is]
        vel_is = np.array(self.qvel_is)[vel_rel_is]
        pos_labels = labels[pos_is]
        vel_labels = labels[vel_is]
        return pos_labels, vel_labels

    def get_kinematics_labels(self, concat=True):
        """
        Returns a list of all kinematic labels used with the current model.
        @param: concat: if true, return a single list containing qpos and qvel labels,
                        if false, return two lists qpos_labels and qvel_labels
        """
        global labels
        qpos_labels = labels[self._qpos_indices]
        qvel_labels = labels[self._qvel_indices]
        if concat:
            return np.concatenate([qpos_labels, qvel_labels]).flatten()
        else:
            return qpos_labels, qvel_labels

    def _adapt_trajecs_to_other_body(self, adapts: dict):
        '''The trajectories were collected from a single reference person.
           They have to be adjusted when used with a model
           with different body properties compared to the reference person.'''
        indices = adapts.keys()
        for index in indices:
            scalar = adapts[index]
            # also adapt the kinematic ranges
            self.ranges[index] *= np.abs(scalar)
            for i_step in range(len(self.data)):
                self.data[i_step][index,:] *= scalar


    def _determine_left_steps_indices(self):
        """
        The dataset contains steps with right and left legs.
        The side of the swing leg is the side of the step.
        The swing leg has a higher knee angle velocity compared to the stance leg.
        :return: the indices of steps taken with the left leg.
        """
        indices = [i for (i, step) in enumerate(self.data)
                   if np.max(step[KNEE_ANGVEL_L]) > np.max(step[KNEE_ANGVEL_R])]
        return indices


    def is_step_left(self):
        return self._i_step in self.left_step_indices


    def _get_deterministic_init_state(self, i_step = 0):
        ''' Deterministic State Initialization.
            @returns: qpos and qvel on a predefined position on the ref trajecs
                      but choosing another step each time. '''
        self.reset()

        # choose another reference step each time
        self._i_step = self.n_deterministic_inits
        self._step = self.data[self._i_step]
        # desired init position: mid stance
        self._pos = int(0.75 * len(self._step[0]))

        self.n_deterministic_inits += 1
        # print(f'{self.n_deterministic_inits} deterministic inits (pos {self._pos}).')

        if self.n_deterministic_inits >= EVAL_N_TIMES:
            self.n_deterministic_inits = 0

        # initialize the eval episodes always in the same state
        # (iterate between left and right only)
        SAME_INIT = False
        if SAME_INIT:
            self._i_step = self.n_deterministic_inits % 2
            self._step = self.data[self._i_step]
            self._pos = int(0.85 * len(self._step[0]))

        qpos, qvel = self.get_qpos(), self.get_qvel()
        # print(qpos, qvel)
        return qpos, qvel

    def get_com_kinematics_full(self):
        """:returns com kinematics for the current steps."""
        com_pos = self._step[:3, :]
        com_vel = self._step[15:18, :]
        return com_pos, com_vel

    def get_com_height(self):
        return self._step[COM_POSZ, self._pos]

    def get_trunk_ang_saggit(self):
        return self._step[TRUNK_ROT_Y, self._pos]

    def get_trunk_rotation(self):
        ''':returns trunk_rot: in quaternions (4D)
                    trun_ang_vel: corresponding angular velocities (3D)'''
        trunk_rot = self._step[3:7, :]
        trunk_ang_vel = self._step[18:21, :]
        return trunk_rot, trunk_ang_vel

    def get_hip_kinematics(self):
        indices_angs = [7,8, 11,12]
        indices_vels = [21,22, 25,26]
        ang_front_r, ang_sag_r, ang_front_l, ang_sag_l = self._step[indices_angs]
        vel_front_r, vel_sag_r, vel_front_l, vel_sag_l = self._step[indices_vels]
        return ang_front_r, ang_sag_r, ang_front_l, ang_sag_l, \
               vel_front_r, vel_sag_r, vel_front_l, vel_sag_l

    def get_knee_kinematics(self):
        indices = [9, 13, 23,27]
        ang_r, ang_l, vel_r, vel_l = self._step[indices]
        return ang_r, ang_l, vel_r, vel_l

    def get_ankle_kinematics(self):
        indices = [10,14, 24,28]
        ang_r, ang_l, vel_r, vel_l = self._step[indices]
        return ang_r, ang_l, vel_r, vel_l

    def _load_ref_trajecs(self):
        """ In this class, the data is split into individual steps.
            Shape of data is: (n_steps, data_dims, traj_len). """
        # load matlab data, containing trajectories of 250 steps
        path = get_project_path() + PATH_REF_TRAJECS
        data = spio.loadmat(path, squeeze_me=True)
        # 250 steps, shape (250,1), where 1 is an array with kinematic data
        data = data['Data']
        # flatten the array to have dim (steps,)
        data = data.flatten()
        # contains the data of all steps
        self.data = data
        # todo: check if that is an error of our base class
        #  or the reason for that lies in the mocap data
        # return all the data for both qpos and qvel.
        # the correct data will be fetched by the corresponding indices.
        return data[0], data[0]

    def _get_next_step(self):
        """
        The steps are sorted. To get the next step, we just have to increase the index.
        However, the COM X Position is zero'ed for each step.
        Thus, we need to add the so far traveled distance to COM X Position.
        """

        # increase the step index, reset if last step was reached
        if self._i_step >= len(self.data) - 1:
            self.has_reached_last_step = True
            # reset to the step with the correct foot
            self._i_step = 0 if self._i_step in self.left_step_indices else 1
        else:
            self._i_step += 1
            self.count_steps_same_vel += 1

        # update the so far traveled distance
        self.dist = self._step[COM_POSX, -1]
        # choose the next step
        # copy to add the com x position only of the current local step variable
        step = np.copy(self.data[self._i_step])
        assert step[COM_POSX, 0] < 0.005, \
            "The COM X Position on each new step trajectory should start with 0.0 " \
            f"but started with {step[COM_POSX, 0]}"
        # add the so far traveled distance to the x pos of the COM
        step[COM_POSX,:] += self.dist
        return step

    def _add_trunk_euler_rotations(self):
        '''Used to extend reference data with euler rotations of the trunk.
           Before, trunk rotations were only given in unit quaternions.'''
        from scipy.spatial.transform import Rotation as Rot

        data_dict = spio.loadmat(self.path)
        data = data_dict['Data']
        data = data.flatten()
        # save only the first 30 steps (constant speed)
        data = data[:30]
        new_data = np.ndarray(data.shape, dtype=np.object)


        # iterate over all steps and add three more dimensions containing trunk euler rotations
        for i, step in enumerate(data):
            # get trunk rotation in quaternions: q1...q4
            q1, q2 = step[TRUNK_ROT_Q1,:], step[TRUNK_ROT_Q2,:]
            q3, q4 = step[TRUNK_ROT_Q3,:], step[TRUNK_ROT_Q4,:]
            # quaternion in scalar-last (x, y, z, w) format.
            old_rot_quat = np.array([q2, q3, q4, q1])
            rot_quat = Rot.from_quat(old_rot_quat.transpose())
            # convert to euler rotations
            euler_x, euler_y, euler_z = rot_quat.as_euler('xyz').transpose()
            # save the new angles in the reference trajectory data
            dims, dur = step.shape
            new_step = np.ndarray((dims + 3, dur), dtype=np.object)
            new_step[0:dims,:] = step
            new_step[dims:,:] = [euler_x, euler_y, euler_z]

            new_data[i] = new_step

        self.data = new_data.flatten()

        data_dict['Data'] = new_data
        print('BEFORE saved final')
        # spio.savemat('Trajecs_Ramp_Slow_400Hz_EulerTrunkAdded.mat', data_dict, do_compression=True)
        spio.savemat('Trajecs_Constant_Speed_400Hz.mat', data_dict, do_compression=True)
        print('saved final')
        raise SystemExit('This function was only used to transform '
                         'the Trunk Quaternion to Euler Rotations.\n'
                         'The transformed data was saved.\n'
                         'This method is now only required for documentation.')

    def _calculate_walking_speed(self):
        step_speeds = []
        for step in self.data:
            com_vels = step[COM_VELX,:]
            walk_speed = np.mean(com_vels)
            step_speeds.append(walk_speed)

            # filter speeds as are too noisy
            speeds_filtered = smooth_exponential(step_speeds, alpha=0.2)

        PLOT = False
        if PLOT:
            plt = self.plt
            plt.plot(step_speeds)
            plt.plot(speeds_filtered)
            plt.xlabel('Step Nr. [ ]')
            plt.ylabel('Mean COM Forward Velocity [m/s]')
            plt.title('Changes in the walking speed of individual steps over time')
            plt.legend([r'Original Mean Velocities', r'Exponentially Smoothed ($\alpha$=0.2)'])
            plt.show()
            # exit(33)

        return speeds_filtered

    def get_step_velocity(self):
        """
        Returns the mean COM forward velocity of the current step
        which is a rough estimation of the walking speed
        """
        return self.step_velocities[max(0, self._i_step-self.count_steps_same_vel+1)]

    def _determine_trajectory_ranges(self, print_ranges=False):
        '''Needed for early termination. We terminate an episode when the agent
           deviated too much from the reference trajectories. How much deviation is allowed
           depends on the maximum range of a joint position or velocity.'''
        # load already determined and saved ranges or calculate and save if not yet happened
        PATH_TRAJEC_RANGES = ''
        try:
            global labels
            npz = np.load(PATH_TRAJEC_RANGES)
            ranges = npz['ranges']
            if print_ranges:
                for label, range in zip(labels, ranges):
                    print(f'{label}\t\t{range}')
            return ranges
        except FileNotFoundError:
            print('COULD NOT LOAD TRAJEC RANGES, (RE)CALCULATING THEM!')
            pass

        mins = np.ones((len(self.data),self.data[0].shape[0]))
        maxs = np.ones_like(mins)
        for i_step, step in enumerate(self.data):
            for i_traj, traj in enumerate(step):
                min = np.min(traj)[0][0]
                max = np.max(traj)[0][0]
                mins[i_step, i_traj] = min
                maxs[i_step, i_traj] = max
        mins = np.min(mins, axis=0)
        maxs = np.max(maxs, axis=0)
        ranges = maxs - mins
        np.savez(PATH_TRAJEC_RANGES, mins=mins, maxs=maxs, ranges=ranges)
        self.ranges = ranges

    # ----------------------------
    # Methods we override:
    # ----------------------------

    def get_random_init_state(self):
        # reset episode duration and so far traveled distance
        self._ep_dur = 0
        self.dist = 0

        # which of the 250 steps are we looking at
        self._i_step = random.randint(0, len(self.data) - 1)
        self._step = self.data[self._i_step]
        self._qpos_full = self._step
        self._qvel_full = self._step
        self._trajec_len = self._step.shape[1]
        self._pos = random.randint(0, self._trajec_len - 1)
        init_kinematics = self.get_qpos(), self.get_qvel()

        return init_kinematics

    def get_deterministic_init_state(self):
        return self._get_deterministic_init_state(i_step=0)

    def get_desired_walking_velocity_vector(self, do_eval, debug=False):
        return [self.get_step_velocity()]

    def _get_COM_Z_pos_index(self):
        return COM_POSZ




if __name__ == '__main__':
    refs = StraightWalkingTrajectories([], [])
