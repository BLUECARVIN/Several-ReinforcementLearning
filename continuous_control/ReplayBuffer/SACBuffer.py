import numpy as np
from .utils import *

class SAC_ReplayBuffer:
	def __init__(self, observation_dim, action_dim, max_size):
		"""
		A FIFO replay buffer for SAC.

		Paras:
			observation_dim: the dimention of observation
			action_dim:	the dimention of action
			max_size: the maximum size of the buffer
		"""
		self.observation_buffer = np.zeros(combined_shape(max_size, observation_dim), dtype=np.float32)
		self.next_observation_buffer = np.zeros(combined_shape(max_size, observation_dim), dtype=np.float32)
		self.action_buffer = np.zeros(combined_shape(max_size, action_dim), dtype=np.float32)
		self.reward_buffer = np.zeros(max_size, dtype=np.float32)
		self.done_buffer = np.zeros(max_size, dtype=np.float32)

		self.index = 0
		self.size = 0
		self.max_size = max_size

	def add(self, observation, action, reward, next_observation, done):
		# Add the data to the buffer and update the index and the size
		self.observation_buffer[self.index] = observation
		self.next_observation_buffer[self.index] = next_observation
		self.reward_buffer[self.index] = reward
		self.action_buffer[self.index] = action
		self.done_buffer[self.index] = done

		self.index = (self.index + 1) % self.max_size
		self.size = min(self.max_size, self.size + 1)

	def sample_batch(self, batchSize=32):
		batchSize = min(batchSize, self.size)
		idx = np.random.randint(0, self.size, size=batchSize)
		batch = dict(observation=self.observation_buffer[idx], 
			next_observation=self.next_observation_buffer[idx], 
			action=self.action_buffer[idx],
			reward=self.reward_buffer[idx],
			done=self.done_buffer[idx])
		return batch
