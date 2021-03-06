import sys
sys.path.append("..")

import Agent
import gym 
import numpy as np

env = gym.make('Breakout-v0')
agent = Agent.DoubleDQNAgent(env, 12, None, memory_size=1000)
agent.test(is_render=True, path='./Log2/DQNtest.pt', epoch=100)