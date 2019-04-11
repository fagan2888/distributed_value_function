"""
Defines the API of the agent
"""


class BaseAgent(object):
    def predict(self, observation):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample()


class QLearningAgent(BaseAgent):
    def __init__(self):
        """
        Initialize data structures for Q leaner. You may need to change the constructor to pass more arguments.
        """
        pass

    def predict(self, observation):
        pass
