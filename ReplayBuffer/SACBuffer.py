import numpy as np
from .utils import *

class SAC_ReplayBuffer:
	def __init__(self, observationDim, actionDim, maxSize):
		"""
		A FIFO replay buffer for SAC.

		Paras:
			observationDim: the dimention of observation
			actionDim:	the dimention of action
			maxSize: the maximum size of the buffer
		"""
		self.observationBuffer = np.zeros(combined_shape(maxSize, observationDim), dtype=np.float32)
		self.nextObservationBuffer = np.zeros(combined_shape(maxSize, observationDim), dtype=np.float32)
		self.actionBuffer = np.zeros(combined_shape(maxSize, actionDim), dtype=np.float32)
		self.rewardBuffer = np.zeros(maxSize, dtype=np.float32)
		self.doneBuffer = np.zeros(maxSize, dtype=np.float32)

		self.index = 0
		self.size = 0
		self.maxSize = maxSize

	def add(self, observation, action, reward, nextObservation, done):
		# Add the data to the buffer and update the index and the size
		self.observationBuffer[self.index] = observation
		self.nextObservationBuffer[self.index] = nextObservation
		self.rewardBuffer[self.index] = reward
		self.actionBuffer[self.index] = action
		self.doneBuffer[self.index] = done

		self.index = (self.index + 1) % self.maxSize
		self.size = min(self.maxSize, self.size + 1)

	def sample_batch(self, batchSize=32):
		batchSize = min(batchSize, self.size)
		idx = np.random.randint(0, self.size, size=batchSize)
		batch = dict(observation=self.observationBuffer[idx], 
			nextObservation=self.nextObservationBuffer[idx], 
			action=self.actionBuffer[idx],
			reward=self.rewardBuffer[idx],
			done=self.doneBuffer[idx])
		return batch
