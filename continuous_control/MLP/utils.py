import numpy as np
from torch import nn


def count_vars(module):
	return sum([np.prod(p.shape) for p in module.parameters()])


def create_mlp(size, activation, output_activation=None):
	layers = []
	for j in range(len(size) - 2):
		layers += [nn.Linear(size[j], size[j+1]), activation()]
	layers += [nn.Linear(size[-2], size[-1])]
	return nn.Sequential(*layers)