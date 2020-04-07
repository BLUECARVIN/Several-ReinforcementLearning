import sys
sys.path.append("..")

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import gym
import numpy as np
import random
from Utils import hard_update
from MLP import QNet
import ReplayBuffer


class DQNAgent(object):
	def __init(self, 
		env,
		random_seed,
		save_path,
		q_net=QNet,
		gamma=0.99,
		batch_size=64,
		initial_e = 0.5,
		end_e = 0.01,
		lr=1e-3,
		learning_starts=50000,
		learning_freq=4,
		frame_history_len=4,
		target_update_freq=10000,
		memory_size=1e6):
		"""
		DQN Agent

		paras:
			env: the gym environment
			seed: the random seed
			save_path: the path to save model parameters
			q_net: the Q learning network function
			gamma: the reward's decrease parameter
			initial_e: the initial prob to choose random action
			end_e: the end prob to choose random action
			lr: the optimizer's learning rate
			target_update_freq: the target netwok's update frequency
			learning_start: begin to learn after learning_start steps
			learning_freq: the training frequency 
			frame_history_len: how much frames should be feed to the model as one data
			memory_size: the maxmium size of replay buffer
		"""
		assert type(env.observation_space) == gym.spaces.Box
		assert type(env.action_space) == gym.spaces.Discrete

		# fix random seed
		torch.manual_seed(random_seed)
		np.random.seed(random_seed)
		random.seed(random_seed)

		# set env
		self.env = env
		# get observation dim
		if len(env.observation_space.shape) == 1: # running on low-dimension observation(RAM)
			self.observation_dim = env.observation_space.shape[0]
		else:
			img_h, img_w, img_c = env.observation_space.shape
			self.observation_dim = frame_history_len * img_c
		# get action dim
		self.action_dim = env.action_space.n

		# set Q network
		self.learning_Q = q_net(self.observation_dim, self.action_dim)
		self.target_Q = q_net(self.observation_dim, self.action_dim)
		# sync two networks' parameter
		hard_update(self.target_Q, self.learning_Q)

		# set replay buffer
		replaybuffer = 
