import numpy as np


def combined_shape(size, shape=None):
	"""
	combine the given size(scalar) and shape(scalar or turple)
	and return a turple (size, *shape)

	paras: 
		size[np.int]: the maxmium size of the replay buffer
		shape[np.int / turple]: the dimention of the storaged data

	return:
		[turple]
	"""
	if shape == None:
		return (size, )
	else:
		return (size, shape) if np.isscalar(shape) else (size, *shape)
