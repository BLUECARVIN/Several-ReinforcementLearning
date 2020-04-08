from torch import nn
from torch.nn import functional as F


class QNet(nn.Module):
	def __init__(self, input_channel=4, num_actions=18):
		"""
		Create a MLP Q network as described in DQN paper
		"""
		super(QNet, self).__init__()
		self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.fc1 = nn.Linear(7*7*64, 512)
		self.fc2 = nn.Linear(512, num_actions)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.fc1(x.flatten(start_dim=1)))
		x = self.fc2(x)
		return x