from .utils import *
from .QFunction import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class MLP_SquashedGaussianActor(nn.Module):
	def __init__(self,
		observation_dim,
		action_dim, 
		hidden_sizes,
		activation,
		act_limit):
		super().__init__()

		self.log_std_max = 2
		self.log_std_min = -20

		self.net = create_mlp([observation_dim] + list(hidden_sizes),
			activation,
			activation)
		self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
		self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
		self.act_limit = act_limit

	def forward(self, observation, deterministic=False, with_log_prob=True):
		net_out = self.net(observation)
		# computer the \mu and \sigma of the gaussian
		mu = self.mu_layer(net_out)

		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		std = torch.exp(log_std)

		# Pre-squash distribution and sample
		pi_distribution = Normal(mu, std)

		if deterministic:
			# only used for evaluating policy at test time.
			pi_action = mu
		else:
			pi_action = pi_distribution.rsample()

		if with_log_prob:
			# Appendix C
			log_pro_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
			log_pro_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(dim=-1)
		else:
			log_pro_pi = None

		pi_action = torch.tanh(pi_action)
		pi_action = self.act_limit * pi_action
		return pi_action, log_pro_pi