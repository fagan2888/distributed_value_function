"""
Defines the API of the agent
"""


class BaseAgent(object):
    def learn(self, obs, next_obs, action, reward):
        pass

    def predict(self, obs):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs):
        return self.action_space.sample()


class QLearningAgent(BaseAgent):
    def __init__(self):
        """
        Initialize data structures for Q leaner. You may need to change the constructor to pass more arguments.
        """
        pass

    def predict(self, obs):
        pass

    def learn(self, obs, next_obs, action, reward):
        pass


class DistributedQLearningAgent(BaseAgent):
    def __init__(self):
        pass

    def predict(self, obs):
        pass

    def learn(self, obs, next_obs, action, reward):
        pass
