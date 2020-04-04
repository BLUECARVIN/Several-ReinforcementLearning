import random
from collections import deque
import numpy as np


class MemoryBuffer:
	"""
	A simple FIFO memory replay buffer 
	"""
	def __init__(self, max_size):
		self.buffer = deque(maxlen=max_size)