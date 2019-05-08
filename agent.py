"""
Defines the API of the agent
"""
import numpy as np

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
        # Action Space (local): double, stay the same, or half
        self.action_space = 3
        # Distributor-distributor state space: 18, from paper. TODO
        self.obs_space = 18

        # initialize Q table with 0
        # What should I set Initial conditions (Q0)?
        # self.QTable = [[0 for i in range(state_space)] for j in range(action_space)]
        self.QTable = np.zeros(shape=(self.obs_space, self.action_space))
        self.learningRate = 0.1
        self.discountFactor = 0.1

    def predict(self, obs):
        maxQ = 0
        maxAction = -1
        for action in self.action_space:
            if maxQ < self.QTable[obs][action]:
                maxQ = self.QTable[obs][action]
                maxAction = action

        return maxAction

    def learn(self, obs, next_obs, action, reward):
        maxFutureQ = 0
        for action in self.action_space:
            if maxFutureQ < self.QTable[next_obs][action]:
                maxFutureQ = self.QTable[next_obs][action]

        self.QTable[obs][action] = (1 - self.learningRate) * self.QTable[obs][action] + self.learningRate * (reward + self.discountFactor * maxFutureQ)

class DistributedQLearningAgent(BaseAgent):
    def __init__(self):
        pass

    def predict(self, obs):
        pass

    def learn(self, obs, next_obs, action, reward):
        pass
