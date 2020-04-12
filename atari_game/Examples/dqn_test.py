import sys
sys.path.append("..")

import Agent
import gym
import numpy as np

env = gym.make('Asteroids-v0')
# a = np.empty([1000000, 84, 84], dtype=np.uint8)
# agent = Agent.DoubleDQNAgent(env, 1626, '/home/szchen/Several-ReinforcentLearning/atari_game/Examples/Log/')
agent = Agent.DoubleDQNAgent(env, 12, None, learning_start=1000, memory_size=1000)
agent.train(is_render=False)