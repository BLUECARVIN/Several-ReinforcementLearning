import torch
from torch import nn
from .utils import *


class MLP_QFunction(nn.Module):
	def __init__(self, observationDim, actionDim, hiddenSizes, activation):
		super().__init__()
		self.q = create_mlp(
			[observationDim+actionDim] + list(hiddenSizes) + [1],
			activation)

	def forward(self, obs, act):
		q = self.q(torch.cat([obs, act], dim=-1))
		return torch.squeeze(q, -1)

