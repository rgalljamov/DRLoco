import numpy as np
from drloco.config.hypers import lr_scale

class Schedule(object):
    def value(self, fraction_timesteps_left):
        """
        Value of the schedule for a given timestep

        :param fraction_timesteps_left:
            (float) PPO2 does not pass a step count in to the schedule functions
             but instead a number between 0 to 1.0 indicating how much timesteps are left
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError

class LinearDecay(Schedule):
    def __init__(self, start_value, final_value):
        self.start = start_value
        self.end = final_value
        self.slope = lr_scale * (final_value - start_value)

    def value(self, fraction_timesteps_left):
        fraction_passed = 1 - fraction_timesteps_left
        val = self.start + fraction_passed * self.slope
        # value should not be smaller then the minimum specified
        val = np.max([val, self.end])
        return val

    def __str__(self):
        return f'LinearSchedule: {self.start} -> {self.end}'

    def __repr__(self):
        return f'LinearSchedule: {self.start} -> {self.end}'


class LinearSchedule(LinearDecay):
    """This class is just required to be able to load models trained with the LinearSchedule
       which we later renamed to LinearDecay."""
    pass


class ExponentialSchedule(Schedule):
    def __init__(self, start_value, final_value, slope=5):
        """@param slope: determines how fast the scheduled value decreases.
           The higher the slope, the stronger is the exponential decay."""
        self.start = start_value
        self.end = final_value
        self.slope = slope
        self.difference = start_value - final_value

    def value(self, fraction_timesteps_left):
        fraction_passed = 1 - fraction_timesteps_left
        val = self.end + np.exp(-self.slope * fraction_passed) * self.difference
        return val
