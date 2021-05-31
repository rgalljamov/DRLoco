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
        self._qpos_is = qpos_indices
        self._qvel_is = qvel_indices
        self._data_labels = data_labels

        # container of the trajectories (see _load_ref_trajecs() for details)
        self._data = self._load_ref_trajecs()
        # check if all data rows have a label
        assert len(data_labels) == self._data.shape[0], \
            "Please provide a label for each row in the data matrix.\n"
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
        data = self._data[indices, position]
        return data

    def get_qpos(self):
        return self._get_by_indices(self._qpos_is)

    def get_qvel(self):
        return self._get_by_indices(self._qvel_is)

    def reset(self):
        """ Set all indices, counters and containers to zero."""
        self._pos = 0

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
        if self._pos >= self._data.shape[1]:
            self._pos = 0

    # ----------------------------
    # Methods to override:
    # ----------------------------

    def _load_ref_trjaecs(self):
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
