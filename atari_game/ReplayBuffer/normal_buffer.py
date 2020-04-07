"""
    This file references from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import numpy as np
import random


class ReplayBuffer(object):
	"""
	A memory buffer for atari game

	The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.
	
	paras:
		max_size: the max number transitions can be stored in the buffer
		frame_history_len:  number of memories to be retired for each observation
	"""
	def __init__(self, max_size, frame_history_len):
		self.max_size = max_size
		self.frame_history_len = frame_history_len

		self.next_index = 0
		self.size = 0	# how many transitions in buffer now

		self.obs = None
		self.action = None
		self.reward = None
		self.done = None


	def can_sample(self, batch_size):
		if self.size >= batch_size:
			return True
		else:
			return False


	def _encode_observation(self, index):
		end_index = index + 1 
		start_index = end_index - self.frame_history_len

		# if using low-dimensional observations, like RAM state, directly return the latest RAM
		if len(self.obs.shape) == 2:
			return self.obs[end_index - 1]

		# if there were not enough frames ever in the buffer for context
		if start_index < 0 and self.size != self.max_size:
			start_index = 0
		for idx in range(start_index, end_index - 1):
			if self.done[idx % self.max_size]:
				start_index = index + 1
		missing_context = self.frame_history_len - (end_index - start_index)
		# if zero padding is need for missing context
		# or we are on the boundry of the buffer
		if start_index < 0 or missing_context > 0:
			frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
			for idx in range(start_index, end_index):
				frames.append(self.obs[idx % self.max_size])
			return np.concatenate(frames, 0)
		else:
			img_h, img_w = self.obs.shape[2], self.obs.shape[3]
			return self.obs[start_index:end_index].reshape(-1, img_h, img_w)


	def _encode_sample(self, index):
		obs_batch = np.concatenate([self._encode_observation(idx)[np.newaxis, :] for idx in index], 0)
		act_batch = self.action[index]
		rew_batch = self.reward[index]
		next_obs_batch = np.concatenate([self._encode_observation(idx+1)[np.newaxis, :] for idx in index], 0)
		done_mask = np.array([1.0 if self.done[idx] else 0. for idx in index], dtype=np.float32)
		return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


	def _sample_n_unique(sampling_f, n):
		"""
		Helper function:
			given a function and returns comparable objects, sample n such unique objects.
		"""
		res = []
		while len(res) < n:
			candidate = sampling_f()
			if candidate not in res:
				res.append(candidate)
		return res


	def sample_batch(self, batch_size):
		"""
		Sample a batch of memory frames 
		
		paras:
			batch_size: how many transitions to samples

		return:
			obs_batch: (batch_size, img_c*frame_history_len, img_h, img_w) -> np.array with dtype np.unit8
			act_batch: (batch_size, ) -> with dtype np.int32
			rew_batch: (batch_size, ) -> with dtype np.float32
			next_obs_batch: (batch_size, img_c*frame_history_len, img_h, img_w) -> with dtype np.float32'
			done_mask: (batch_size, ) -> with dtype np.float32
		"""
		assert self.can_sample(batch_size), "there are no enough tansitions in buffer"
		index = self._sample_n_unique(lambda: random.randint(0, self.size-2), batch_size)
		return self._encode_sample(index)


	def encoder_recent_observation(self):
		"""
		Return the most recent "frame_history_len" frames

		return:
			observation: (img_h, img_w, img_c*frame_history_len)
		"""
		assert self.size > 0, "The frames in buffer is 0"
		return self._encode_observation((self.next_index - 1) % self.max_size)


	def store_frame(self, frame):
		"""
		store a single frame in the buffer at the next available index
		if buffer is full, rewrite the old one

		paras:
			frame: (img_h, img_w, img_c)

		return:
			index: int
		"""
		if len(frame.shape) > 1:
			# transpose image frame into (c, h, w)
			frame = frame.transpose(2, 0, 1)

		if self.obs is None:
			self.obs = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
			self.action = np.empty([self.size], dtype=np.int32)
			self.reward = np.empty([self.size], dtype=np.float32)
			self.done = np.empty([self.size], dtype=np.bool)

		self.obs[self.next_index] = frame

		ret = self.next_index
		self.next_index  = (self.next_index + 1) % self.max_size
		self.size = min(self.max_size, self.size + 1)
		return ret


	def store_effct(self, index, action, reward, done):	
		"""
		store effects of action taken after observing frame stored at index.

		paras:
			index: index in buffer of recently observed frame (returend by store_frame) -> int
			action: performed action -> int
			reward: recived reward after performing the action -> float
			done: if episode was finised, true, else false -> bool
		"""
		self.action[index] = action
		self.reward[index] = reward
		self.done[index] = done