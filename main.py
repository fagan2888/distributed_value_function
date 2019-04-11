"""
Defines the main training loop. Now it's just take random action
"""

import numpy as np

from agent import RandomAgent
from env import make_type_c_grid


def epsilon_greedy(random_action, agent_action, epsilon):
    if np.random.rand() < epsilon:
        action = random_action
    else:
        action = agent_action
    return action


def q_learn(env, agent):
    """ A typical q learning looks something like this. You probably want to pass a list of agents
    for distributed value functions.

    Args:
        env: environment
        agent: q learning agent

    Returns:

    """
    epsilon = 0.2  # you may want to decrease epsilon as training goes.
    learning_iteration = 1000000
    obs = env.reset()
    for _ in range(learning_iteration):
        # get an action using epilon-greedy
        action = epsilon_greedy(env.action_space.sample(), agent.predict(obs), epsilon)
        next_obs, reward, done, _ = env.step(action)
        agent.learn(obs, next_obs, reward)  # update q table using obs, next_obs, reward
        if done:
            obs = env.reset()
        else:
            obs = next_obs


if __name__ == '__main__':
    env = make_type_c_grid()
    agent = RandomAgent(env.action_space)
    done = False
    obs = env.reset()
    reward_lst = []
    while not done:
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        reward_lst.append(reward)

    print('Total reward {}'.format(np.sum(reward_lst)))
    print(reward_lst)
