import gym
import ptan
import numpy as np
import torch.nn as nn

env = gym.make("CartPole-v0")
net = nn.Sequential(nn.Linear(env.observation_space.shape[0], 256), nn.ReLU(), nn.Linear(256, env.action_space.n))

action_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.1)
agent = ptan.agent.DQNAgent(net, action_selector)

