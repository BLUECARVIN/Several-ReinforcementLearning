import sys
sys.path.append("..")
import numpy as np
from copy import deepcopy
import itertools
import time

import gym
import ReplayBuffer
from Utils import *

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

		# set env
		self.env = env
		self.testEnv = env
		self.observationDim = env.observation_space.shape
		self.actionDim = env.action_space.shape[0]

		# Action limit for clamping, assumes all dimension share the same bound
		self.actionLimit = env.action_space.high[0]

		# build AC learning network and target network
		self.ac = actorCritic(env.observation_space, env.action_space, **ACKwargs)
		self.acTarget = actorCritic(env.observation_space, env.action_space, **ACKwargs)

		# sync the parameters between two networks by hard-update
		hard_update(self.acTarget, self.ac)
		# Freeze target networks with respect to optimizers
		for paras in acTarget.parameters():
			paras.requires_grad = False

		# List of parameters for both Q-Networks (for convenience)
		self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

		# Set Memory ReplayBuffer
		self.replayBuffer = ReplayBuffer.SAC_ReplayBuffer(observationDim=self.observationDim,
			actionDim=self.actionDim, maxSize=replayBufferSize)

	# calculate loss
	def compute_loss_q(self, data):
