from .utils import *
from .QFunction import *


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
		self.q1 = 
