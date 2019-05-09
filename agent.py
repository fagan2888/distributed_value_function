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
        self.action_count = 3
        # Distributor-distributor state space: 54
        self.obs_count = 54

        # initialize Q table with 0
        # What should I set Initial conditions (Q0)?
        # self.QTable = [[0 for i in range(state_space)] for j in range(action_space)]
        self.QTable = np.zeros(shape=(self.obs_count, self.action_count))
        self.learningRate = 0.2
        self.discountFactor = 0.99

    def predict(self, obs):
        """ Get the max Q value given obs. Use np.argmax

                Args:
                    obs: integer

                Returns: max Q value's index (which is the action to take).

        """
        rowInQ = self.QTable[obs]
        return np.argmax(rowInQ)

    def getMaxQ(self, obs):
        """ Get the max Q value given obs. Use np.max

        Args:
            obs: integer

        Returns: max Q value

        """
        rowInQ = self.QTable[obs]
        return np.max(rowInQ)

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
        self.QTable[obs][action] = (1 - self.learningRate) * self.QTable[obs][action] + \
                                   self.learningRate * (reward + self.discountFactor * maxFutureQ)


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
        action = np.ndarray(action)
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
        # Gather each agent's max Q given observation
        agent_maxQ = []
        for i, agent in enumerate(self.local_agents):
            agent_maxQ.append(self.local_agents.getMaxQ(self, obs[i]))

        # Compute each agent's neighboring max Q
        neighbor_maxQ = []
        neighbor_maxQ.append((agent_maxQ[0] + agent_maxQ[1]) / 2)
        for i in range(1, self.local_agents.count() - 1):
            neighbor_maxQ.append((agent_maxQ[i-1] + agent_maxQ[i] + agent_maxQ[i+1]) / 3)
        neighbor_maxQ.append((agent_maxQ[self.local_agents.count() - 2] + agent_maxQ[self.local_agents.count() - 1]) / 2)

        # Call each agent's learn function
        for i, local_agent in enumerate(self.local_agents):
            local_agent.learn(obs[i], next_obs[i], action[i], reward[i], neighbor_maxQ[i])
