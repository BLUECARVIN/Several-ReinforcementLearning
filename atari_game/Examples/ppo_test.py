import sys
sys.path.append("..")

import Agent
import gym
import numpy as np

env = gym.make('LunarLander-v2')
agent = Agent.PPOAgent(env, 12, "./")
agent.train(is_render=False)