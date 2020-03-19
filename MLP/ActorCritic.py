from .utils import *
from .QFunction import *
from .Gaussian import *

class MLP_ActorCritic(nn.Module):
	def __init__(self, 
		observationSpace, 
		actionSpace, 
		hiddenSizes=(256, 256),
		activation=nn.ReLU):
		super().__init__()

		observationDim = observationSpace.shape[0]
		actionDim = actionSpace.shape[0]
		actionLimit = actionSpace.high[0]

		# build policy and value functions
		self.pi = MLP_SquashedGaussianActor(observationDim, actionDim, hiddenSizes, activation, actionLimit)
		self.q1 = MLP_QFunction(observationDim, actionDim, hiddenSizes, activation)
		self.q2 = MLP_QFunction(observationDim, actionDim, hiddenSizes, activation)

	def act(self, observation, deterministic=False):
		with torch.no_grad():
			action, _ = self.pi(observation, deterministic, False)
			return action.detach().cpu().numpy()