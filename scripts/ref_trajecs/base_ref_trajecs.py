import numpy as np
from scripts.common.utils import get_absolute_project_path


class BaseReferenceTrajectories:
    def __init__(self, rel_data_path, sample_freq, control_freq, 
                 qpos_indices, qvel_indices, data_labels=None):
        """
        :param rel_data_path:   relative path to the .npz file containing the reference trajectories
        :param sample_freq:     frequency the reference trajectory (e.g. mocap) data was collected with
        :param control_freq:    frequency the policy is sampled at.
        :param qpos_indices:    indices describing where in the data matrix to find the joint positions
        :param qvel_indices:    indices describing where in the data matrix to find the joint velocities
        :param data_labels:     names of the data in each row (used for plotting, monitoring and debugging)
        """
        self._data_path = get_absolute_project_path() + rel_data_path
        self._sample_freq = sample_freq
        self._control_freq = control_freq
        self._qpos_indices = qpos_indices
        self._qvel_indices = qvel_indices
        self._data_labels = data_labels

        # container of the trajectories (see _load_ref_trajecs() for details)
        self._data = self._load_ref_trajecs()
        self._data_dims, self._trajec_len = self._data.shape
        # check if all data rows have a label
        assert len(data_labels) == self._data_dims, \
            "Please provide a label for each row in the data matrix.\n" \
            f"You provided {len(data_labels)} labels for a matrix of shape {self._data.shape}.\n"
        # position on the reference trajectories
        self._pos = 0
        # how many points to "jump over" when next() is called
        self._set_increment()

    def _get_by_indices(self, indices, position=None):
        """
        Get specified reference trajectories at a specified position.
        If position is None, just use the current position on the trajectories: self._pos.
        """
        if position is None:
            position = self._pos
        data = self._data[indices, int(position)]
        return data

    def get_qpos(self):
        return self._get_by_indices(self._qpos_indices)

    def get_qvel(self):
        return self._get_by_indices(self._qvel_indices)

    def reset(self):
        """ Set all indices, counters and containers to zero."""
        self._pos = 0

    def get_deterministic_init_state(self, pos_in_percent):
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
            f'The sampling frequency of the reference data should be equal to ' \
            f'or an integer multiple of the control frequency.'
        self._increment = increment

    def next(self):
        """
        Increase the internally managed position on the reference trajectory.
        This method should be called in the step() function of the RL environment.
        """
        self._pos += self._increment
        # reset the position to zero again when it reached the max possible value
        if self._pos >= self._trajec_len - 1:
            self._pos = 0

    # ----------------------------
    # Methods to override:
    # ----------------------------

    def _load_ref_trajecs(self):
        """
        Populates self.data with the reference trajectories.
        The expected shape is (n_dims, n_points)
        where n_dims is the number of trajectories and n_points their length.
        """
        raise NotImplementedError

    def _get_desired_walking_velocity_vector(self):
        """
        Returns the desired velocity vector for the current position on the reference trajectories.
        This vector is used as part of the state observations. During training, the vector
        is derived from the reference trajectories. When running the trained model, the desired
        velocity is specified by the user.
        """
        raise NotImplementedError
