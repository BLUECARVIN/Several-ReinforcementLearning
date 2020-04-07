import numpy as np


def hard_update(target, source):
	'''
	Copies the parameters from source network to target network

	paras:
	-----------------------------------------
	target: the target pytorch model which needs to be updated
	source: the source pytorch model is used to update the target network
	'''
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)

def soft_update(target, source, tau):
	'''
	Update the parameters from source network to target network softly
	y = τ*x + (1 - τ)*y

	paras:
	-----------------------------------------
	target: the target pytorch model which needs to be updated
	source: the source pytorch model is used to update the target network
	tau: the coefficient of soft update
	'''
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data*(1.0-tau) + param.data*tau)


def one_hot(actions, numActions):
	'''
	Given the actions and the number of action space, return the one-hot code

	paras:
	--------------------------------------------
	actions: the numpy array of action index --> np.array
	numActions: the number of action space
	'''
	labels = actions.reshape(1, -1)[0]
	oneHotCode = np.eye(numActions)[labels]
	return oneHotCode