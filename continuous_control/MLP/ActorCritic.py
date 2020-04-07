from .utils import *
from .QFunction import *
from .Gaussian import *

class MLP_ActorCritic(nn.Module):
	def __init__(self, 
		observation_space, 
		action_space, 
		hidden_sizes=(256, 256),
		activation=nn.ReLU):
		super().__init__()

		observation_dim = observation_space.shape[0]
		action_dim = action_space.shape[0]
		action_limit = action_space.high[0]

		# build policy and value functions
		self.pi = MLP_SquashedGaussianActor(observation_dim, action_dim, hidden_sizes, activation, action_limit)
		self.q1 = MLP_QFunction(observation_dim, action_dim, hidden_sizes, activation)
		self.q2 = MLP_QFunction(observation_dim, action_dim, hidden_sizes, activation)

	def act(self, observation, deterministic=False):
		with torch.no_grad():
			action, _ = self.pi(observation, deterministic, False)
			return action.detach().cpu().numpy()