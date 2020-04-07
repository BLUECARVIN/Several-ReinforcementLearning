import torch
from torch import nn
from torch.nn import functional as F
from .utils import *


class MLP_QFunction(nn.Module):
	def __init__(self, observation_dim, action_dim, hidden_sizes, activation):
		super().__init__()
		self.q = create_mlp(
			[observation_dim+action_dim] + list(hidden_sizes) + [1],
			activation)

	def forward(self, obs, act):
		q = self.q(torch.cat([obs, act], dim=-1))
		return torch.squeeze(q, -1)


class Conv_QFunction(nn.Module):
	def __init__(self, in_channels, action_dim):
		"""
		initialize a Q Network with 2D Convolutions kernel
		"""
		super(Conv_QFunction, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.fc4 = nn.Linear(7*7*64, 512)
		self.fc5 = nn.Linear(512, aciton_dim)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.fc4(x.view(x.size(0), -1)))
		x = self.fc5(x)
		return x