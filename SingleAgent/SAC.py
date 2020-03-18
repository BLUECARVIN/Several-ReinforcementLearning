import sys
sys.path.append("..")
import numpy as np
from copy import deepcopy
import itertools
import time

import gym
import ReplayBuffer

import torch
import MLP
from torch.optim import Adam

class SAC:
	def __init__(self,
		env,
		actorCritic=MLP.ActorCritic.MLP_ActorCritic,
		ACKwargs=dict(),
		randomSeed=0,
		stepsPerEpoch=4000,
		maxEpochs=100,
		replayBufferSize=int(1e6),
		gamma=0.99,
		polyak=0.995,
		lr=1e-3,
		alpha=0.2,
		batchSize=100,
		startSteps=10000,
		updateAfter=1000,
		updateEvery=50,
		numTestEpisodes=10,
		maxEpochLen=1000,
		saveFreq=1,
		**kwargs):

		# set random seed
		torch.manual_seed(randomSeed)
		np.random.seed(randomSeed)

		# set self parameter
		self.ACKwargs = ACKwargs
		self.stepsPerEpoch = stepsPerEpoch
		self.maxEpochs = maxEpochs
		self.replayBufferSize = replayBufferSize
		self.gamma = gamma
		self.polyak = polyak
		self.lr = lr
		self.alpha = alpha
		self.batchSize = batchSize
		self.startSteps = startSteps
		self.updateAfter = updateAfter
		self.updateEvery= updateEvery
		self.numTestEpisodes = numTestEpisodes
		self.maxEpochLen = maxEpochLen

		self.ac = actorCritic()