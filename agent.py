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
        self.learningRate = 0.2
        self.discountFactor = 0.99

    def predict(self, obs):
        maxQ = 0
        maxAction = -1
        for action in self.action_space:
            if maxQ < self.QTable[obs][action]:
                maxQ = self.QTable[obs][action]
                maxAction = action

        # use np.argmax

        return maxAction

    def getMaxQ(self, obs):
        """ Get the max Q value given obs. Use np.max

        Args:
            obs: integer

        Returns: max Q value

        """
        pass

    def learn(self, obs, next_obs, action, reward, maxFutureQ):
        """

        Args:
            obs: local obs. Type: int
            next_obs: local obs of next state. int
            action: local action. int
            reward: local reward. float
            maxFutureQ: average of neighbor maxQ, including itself.

        Returns:

        """
        # maxFutureQ = 0
        # for action in self.action_space:
        #     if maxFutureQ < self.QTable[next_obs][action]:
        #         maxFutureQ = self.QTable[next_obs][action]

        self.QTable[obs][action] = (1 - self.learningRate) * self.QTable[obs][action] + self.learningRate * (
                reward + self.discountFactor * maxFutureQ)


class DistributedQLearningAgent(BaseAgent):
    def __init__(self, num_local_agents):
        self.num_local_agents = num_local_agents
        self.local_agents = []
        # create a list of local agents
        for _ in range(self.num_local_agents):
            self.local_agents.append(QLearningAgent())

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """

        Args:
            obs: np array of shape (10,)

        Returns:

        """
        # look at the Q table of each local agent and return their action. Concatenate.
        action = []
        for i, agent in enumerate(self.local_agents):
            action.append(agent.predict(obs[:, i]))
        action = np.array(action)
        return action

    def learn(self, obs, next_obs, action, reward) -> None:
        """ 1. Compute neighbor maxQ.
            2. learn using local info

        Args:
            obs: shape (10,)
            next_obs: shape (10,)
            action: (10,)
            reward: (10,)

        Returns: None

        """
        neighbor_maQ = ...
        for i, local_agent in enumerate(self.local_agents):
            local_agent.learn(obs[i], next_obs[i], action[i], reward[i], neighbor_maQ[i])
