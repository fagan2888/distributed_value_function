"""
A simple grid environment to develop distributed value functions
"""

import gym


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
        distributor-distributor: 1) Whether the neighbor voltage is higher or not.
                                 2) Whether the neighbor voltage increased, lowered or remain the same in last iteration.
                                 3) Whether the resistance is at minimum, maximum or in between
        distributor-city: We assume cities has no resistance and thus the voltage of city is the same as voltage in
                          distributor. We add one more state: whether the voltage meet requirement or not.
        distributor-provider: Whether the voltage at provider is higher or not.
    """
    def __init__(self, num_distributors, num_cities, city_locations=None):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass
