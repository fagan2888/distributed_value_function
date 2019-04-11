"""
A simple grid environment to develop distributed value functions
"""

import gym
import numpy as np
from gym.spaces import MultiDiscrete


def compare_two_array(left, right):
    """ Compare two arrays and set 0: same, 1 less, 2: greater。
        If the type is float, we will add a tolerant value.

    Args:
        left: left array
        right: right array

    Returns: a numpy array with 0: same, 1 less, 2: greater

    """
    assert left.shape == right.shape, 'Left shape and right shape must be the same.'
    equal_index = np.where(np.allclose(left, right))
    greater_index = np.where(left > right)
    less_index = np.where(left < right)
    output = np.zeros_like(left)
    output[greater_index] = 2
    output[less_index] = 1
    output[equal_index] = 0
    return output


class LineGrid(gym.Env):
    """
    We try type C grid because the performance of various algorithms differs much.
    The style of the grid is a straight line with multiple resistors.
    Each resistor has a base resistance and can double/halve/stay. It has a maximum and minimum resistance.
    This type of circuit has an easy close-form solution to compare with q learning performance.
    The cities are randomly chosen to place at each distributor.
    This environment is also a perfect example to develop meta-reinforcement learning algorithm.

    The state of the global environment is the state of the resistance.
    The local state is
        distributor-distributor: 1) Whether the left voltage increased, lowered or remain the same in last iteration.
                                   (0: remain, 1: decreased, 2: increased)
                                 2) Whether the right voltage increased, lowered or remain the same in last iteration.
                                 3) Whether the resistance is at minimum, maximum or in between
                                   (0: between, 1: minimum, 2: maximum)
        distributor-city: We assume cities has no resistance and thus the voltage of city is the same as voltage in
                          distributor. We add one more state: whether the voltage meet requirement or not.
                          To make the state consistent, we assume a virtual city connected to those which doesn't
                          connect to any cities that always meet the requirement.
        distributor-provider: Whether the voltage at provider is higher or not.

    The action space is
        0: half, 1: remain, 2: double

    State space is (num_resistance, 3 * 3 * 3 * 2 = 54)
    """

    def __init__(self, num_resistance, voltage_range_dict, max_steps=100):
        """

        Args:
            num_resistance: number of resistance.
            voltage_range: a dictionary map from city location to a tuple of range.
            max_steps: we expected to solve the problem in 100 steps.
        """
        self.source_voltage = 110  # to make every simple
        self.num_resistance = num_resistance
        self.resistance_level = [0, 1, 2, 3, 4, 5, 6]
        self.resistance = np.array([1, 2, 4, 8, 16, 32, 64])

        self.action_space = MultiDiscrete([3] * self.num_resistance)
        self.observation_space = MultiDiscrete([54] * self.num_resistance)

        # true state of the system
        self.current_level = None
        self.previous_voltage = None

        self.voltage_range = np.array([-np.inf, np.inf])
        self.voltage_range = np.tile(self.voltage_range, (self.num_resistance, 1))
        for location in voltage_range_dict:
            self.voltage_range[location] = np.array(voltage_range_dict[location])

        self.max_steps = max_steps

    def _get_obs(self):
        """

        Returns: a list of encoded observations

        """
        # calculate current voltage
        voltages = self._calculate_voltage()
        # left voltage state
        if self.previous_voltage is None:
            left_voltage_state = np.zeros_like(voltages[:-1], dtype=np.int)
        else:
            left_voltage_state = compare_two_array(voltages[:-1], self.previous_voltage[:-1])
        # right voltage state
        if self.previous_voltage is None:
            right_voltage_state = np.zeros_like(voltages[1:], dtype=np.int)
        else:
            right_voltage_state = compare_two_array(voltages[1:], self.previous_voltage[1:])

        # resistance state
        min_index = np.where(self.current_level == self.resistance_level[0])
        max_index = np.where(self.current_level == self.resistance_level[-1])
        resistance_state = np.zeros_like(self.current_level, dtype=np.int)
        resistance_state[min_index] = 1
        resistance_state[max_index] = 2

        # whether meet requirement
        requirement_state = np.logical_and(voltages[1:] > self.voltage_range[:, 0],
                                           voltages[1:] < self.voltage_range[:, 1]).astype(np.int)

        # get encoded discrete state
        encode_state = self._encode_obs(left_voltage_state, right_voltage_state, resistance_state, requirement_state)

        self.previous_voltage = voltages

        return encode_state

    def _encode_obs(self, left_voltage_state, right_voltage_state, resistance_state, requirement_state):
        """ Encode multiple discrete state into a single discrete state.
        We concatenate (left_voltage_state, right_voltage_state, resistance_state, requirement_state)
        and treat it as a binary-like number and we turn it into decimal. The weights for each digit is
        18, 6, 2, 1

        Args:
            left_voltage_state: 3 discrete states
            right_voltage_state: 3 discrete states
            resistance_state: 3 discrete states
            requirement_state: 2 discrete states

        Returns:

        """
        final_state = 18 * left_voltage_state + 6 * right_voltage_state + 2 * resistance_state + requirement_state
        assert final_state.shape == (self.num_resistance,)
        return final_state

    def _get_reward(self):
        voltages = self.previous_voltage
        requirement_state = np.logical_and(voltages[1:] > self.voltage_range[:, 0],
                                           voltages[1:] < self.voltage_range[:, 1]).astype(np.int)
        number_not_meet_requirement = self.num_resistance - np.sum(requirement_state)
        return -number_not_meet_requirement

    def _encode_action(self, raw_action):
        """ Encode the raw action into action space that can directly compute the next state
            Raw action space: 0: half, 1: remain, 2: double
            Encoded actions space: 0 -> -1. 1 -> 0. 2 -> 1. In this way, we can directly add action to current level

        Args:
            raw_action: shape (num_resistance,)

        Returns: action shape (num_resistance,)

        """
        return raw_action - 1

    def _get_info(self):
        return {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action {}".format(action)
        self._elapsed_steps += 1
        action = self._encode_action(action)
        self.current_level = np.clip(self.current_level + action, self.resistance_level[0], self.resistance_level[-1])
        obs = self._get_obs()
        reward = self._get_reward()
        if self.max_steps <= self._elapsed_steps:
            done = True
        else:
            done = False
        info = self._get_info()
        return obs, reward, done, info

    def reset(self):
        """ Random set each resistance level """
        self.current_level = np.random.randint(0, len(self.resistance_level), size=(self.num_resistance))
        self._elapsed_steps = 0
        return self._get_obs()

    def _calculate_voltage(self):
        """ Calculate the voltage at each node based on the current level """
        actual_resistance = self.resistance[self.current_level]
        actual_resistance = np.append(actual_resistance, [0], axis=0)
        actual_resistance = np.flip(actual_resistance, axis=0)
        proportional = np.cumsum(actual_resistance) / np.sum(actual_resistance)
        proportional = np.flip(proportional, axis=0)
        voltages = self.source_voltage * proportional
        return voltages

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


def make_type_c_grid():
    num_resistance = 10
    voltage_range_dict = {
        0: (100, 110),
        3: (70, 80),
        6: (40, 50),
        8: (10, 20)
    }
    env = LineGrid(num_resistance, voltage_range_dict)
    return env
