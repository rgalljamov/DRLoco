import numpy as np
from collections import Iterable
from drloco.common.utils import get_project_path


class BaseReferenceTrajectories:
    """
    Base class for providing reference trajectories to the RL environment during training
    within the DeepMimic Framework (Peng et. al, 2018).

    :param sample_freq:     frequency the reference trajectory (e.g. mocap) data was collected with
    :param control_freq:    frequency the policy is sampled at.
    :param qpos_indices:    indices describing where in the data matrix to find the joint positions
    :param qvel_indices:    indices describing where in the data matrix to find the joint velocities
    :param data_labels:     names of the data in each row (used for plotting, monitoring and debugging)
    :param adaptations:     dict mapping indices to float values describing scalar multiplications
                            of the trajectories at the specified index. Often required to
                            adapt the reference trajectories to a walker environment.
    """
    def __init__(self, sample_freq, control_freq,
                 qpos_indices, qvel_indices, data_labels=[], adaptations={}):

        self._sample_freq = sample_freq
        self._control_freq = control_freq
        self._qpos_indices = qpos_indices
        self._qvel_indices = qvel_indices
        self._qlabels = data_labels

        # containers of the joint positions and velocities of the whole reference trajectory
        self._qpos_full, self._qvel_full = self._load_ref_trajecs()
        self._n_joints, self._trajec_len = self._qpos_full.shape
        # adapt reference trajectories to the considered walker model
        self.adapt_trajectories(adaptations)
        # check if all data rows have a label
        assert len(data_labels) == 0 \
               or len(data_labels) in [self._n_joints, len(self._qpos_indices)], \
            "Please provide a label for each row in the data matrix.\n" \
            f"You provided {len(data_labels)} labels for a matrix of shape {self._qpos_full.shape}.\n"
        # position on the reference trajectories
        self._pos = 0
        # how many points to "jump over" when next() is called
        self._set_increment()

    def get_qpos(self):
        """
        Returns the joint positions of the considered robot model
        at the current timestep/(time)position.
        """
        return self._qpos_full[self._qpos_indices, self._pos]

    def get_qvel(self):
        """
        Returns the joint velocities of the considered robot model
        at the current timestep/(time)position.
        """
        return self._qvel_full[self._qvel_indices, self._pos]

    def get_reference_trajectories(self):
        """
        Returns the joint positions and velocities
        of the considered robot model
        at the current timestep/(time)position.
        """
        return self.get_qpos(), self.get_qvel()

    def reset(self):
        """ Set all indices, counters and containers to zero."""
        self._pos = 0

    def get_deterministic_init_state(self, pos_in_percent=0):
        """ 
        Returns the data on a certain position on the reference trajectory.
        :param pos_in_percent:  specifies the position on the reference trajectory
                                by skipping the percentage (in [0:100]) of points
        """
        self._pos = int(self._trajec_len * pos_in_percent / 100)
        return self.get_qpos(), self.get_qvel()

    def get_random_init_state(self):
        '''
        Random State Initialization (cf. DeepMimic Paper).
        :returns qpos and qvel of a random position on the reference trajectories
        '''
        self._pos = np.random.randint(0, self._trajec_len)
        return self.get_qpos(), self.get_qvel()

    def _set_increment(self):
        increment = self._sample_freq / self._control_freq
        assert increment.is_integer(), \
            f'Please check your control frequency and the sample frequency of the reference data!' \
            f'The sampling frequency ({self._sample_freq}) of the reference data should be equal to ' \
            f'or an integer multiple of the control frequency which is set to {self._control_freq}.'
        self._increment = int(increment)

    def next(self):
        """
        Increase the internally managed position on the reference trajectory.
        This method should be called in the step() function of the RL environment.
        """
        self._pos += self._increment
        # reset the position to zero again when it reached the max possible value
        if self._pos >= self._trajec_len - 1:
            self._pos = 0

    def adapt_trajectories(self, adaptations_dict):
        """
        Adapt reference trajectories to the considered walker model.
        The trajectories were collected from a single reference person.
        They have to be adjusted when used with a walker model that has
        different body properties compared to the reference person.
        :param adaptations_dict:    mapping from joint_indices to scalars. The scalars are used
                                    to scale the trajectories at the specified index.
        """
        joint_indices = adaptations_dict.keys()
        for index in joint_indices:
            scalar = adaptations_dict[index]
            self._qpos_full[index, :] *= scalar
            self._qvel_full[index, :] *= scalar

    def get_kinematic_label_at_pos(self, pos):
        return self._qlabels[self._qpos_indices][pos]

    def get_kinematics_labels(self):
        return self._qlabels

    def adjust_COM_Z_pos(self, offset):
        self._qpos_full[self._get_COM_Z_pos_index(), :] -= offset

    # ----------------------------
    # Methods to override:
    # ----------------------------

    def _load_ref_trajecs(self):
        """
        Populates self._qpos_full and self._qvel_full with the reference trajectories.
        The expected shape for each container/variable is (n_trajecs, n_points)
        where n_trajecs is the number of trajectories and n_points their length (in samples).
        :returns    two numpy matrices containing the reference joint position
                    and joint velocity trajectories in shape (n_trajecs, n_points).
        """
        raise NotImplementedError

    def _get_COM_Z_pos_index(self):
        """
        Returns the index of the COM Z position in the considered reference trajectories.
        """
        raise NotImplementedError

    def get_desired_walking_velocity_vector(self, do_eval: bool = False) -> Iterable:
        """
        Returns the desired velocity vector as an iterable (e.g. [vx, vy])
        for the current position on the reference trajectories.
        This vector is used as part of the state observations. During training, the vector
        is derived from the reference trajectories. When running the trained model, the desired
        velocity is specified by the user.
        :param do_eval: indicates if the model is currently evaluated (True) or training (False)
        """
        raise NotImplementedError
