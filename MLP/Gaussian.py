from .utils import *
from .QFunction import *
import torh
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class MLP_SquashedGaussianActor(nn.Module):
	def __init__(self,
		observationDim,
		actionDim, 
		hiddenSizes,
		activation,
		actLimit):
		super().__init__()

		self.logSTDMax = 2
		self.logSTDMin = -20

		self.net = create_mlp([observationDim] + list(hiddenSizes),
			activation,
			activation)
		self.muLayer = nn.Linear(hiddenSizes[-1], actionDim)
		self.logSTDLayer = nn.Linear(hiddenSizes[-1], actionDim)
		self.actLimit = actLimit

	def forward(self, observation, deterministic=False, withLogProb=True):
		netOut = self.net(obs)
		# computer the \mu and \sigma of the gaussian
		mu = self.muLayer(netOut)

		logSTD = self.logSTDLayer(netOut)
		logSTD = torch.clamp(logSTD, logSTDMin, logSTDMax)
		std = torch.exp(logSTD)

		# Pre-squash distribution and sample
		piDistribution = Normal(mu, std)

		if deterministic:
			# only used for evaluating policy at test time.
			piAction = mu
		else:
			piAction = piDistribution.rsample()

		if withLogProb:
			# Appendix C
			logProPi = piDistribution.log_prob(piAction).sum(axis=-1)
			logProPi -= (2 * (np.log(2) - piAction - F.softplus(-2*piAction))).sum(axis=-1)
		else:
			logProPi = None

		piAction = torch.tanh(piAction)
		piAction = self.actLimit * piAction
		return piAction, logProPi