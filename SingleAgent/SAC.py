import sys
sys.path.append("..")

import numpy as np
import itertools
import time
from pathlib import Path
import tqdm

import ReplayBuffer
from Utils import *

import torch
from MLP import ActorCritic
from torch.optim import Adam

class SAC:
	def __init__(self,
		env,
		savePath,
		actorCritic=ActorCritic.MLP_ActorCritic,
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
		self.savePath = Path(savePath)
		self.saveFreq = saveFreq

		# set env
		self.env = env
		self.testEnv = env
		self.observationDim = env.observation_space.shape
		self.actionDim = env.action_space.shape[0]

		# Action limit for clamping, assumes all dimension share the same bound
		self.actionLimit = env.action_space.high[0]

		# build AC learning network and target network
		self.ac = actorCritic(env.observation_space, env.action_space, **ACKwargs).cuda()
		self.acTarget = actorCritic(env.observation_space, env.action_space, **ACKwargs).cuda()

		# sync the parameters between two networks by hard-update
		hard_update(self.acTarget, self.ac)

		# Freeze target networks with respect to optimizers
		for paras in self.acTarget.parameters():
			paras.requires_grad = False

		# List of parameters for both Q-Networks (for convenience)
		self.qParams = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

		# Set Memory ReplayBuffer
		self.replayBuffer = ReplayBuffer.SAC_ReplayBuffer(observationDim=self.observationDim,
			actionDim=self.actionDim, maxSize=replayBufferSize)

		# set opimizers
		self.piOptimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
		self.qOptimizer = Adam(self.qParams, lr=self.lr)

	# ===================================== Loss ========================================
	# calculate loss of Q function
	def compute_loss_q(self, data):
		observation = torch.tensor(data['observation'], dtype=torch.float32).cuda()
		action = torch.tensor(data['action'], dtype=torch.float32).cuda()
		reward = torch.tensor(data['reward'], dtype=torch.float32).cuda()
		nextObervation = torch.tensor(data['nextObservation'], dtype=torch.float32).cuda()
		done = torch.tensor(data['done'], dtype=torch.float32).cuda()

		q1 = self.ac.q1(observation, action)
		q2 = self.ac.q2(observation, action)

		# Bellman backup for Q function
		with torch.no_grad():
			# target actions come from current policy
			nextAction, logProbAction2 = self.ac.pi(nextObervation)

			#Target Q-values
			q1PiTarget = self.acTarget.q1(nextObervation, nextAction)
			q2PiTarget = self.acTarget.q2(nextObervation, nextAction)
			qPiTarget = torch.min(q1PiTarget, q2PiTarget)
			backup = reward + self.gamma * (1 - done) * (qPiTarget - self.alpha * logProbAction2)

		# MSE loss against bellman backup
		lossQ1 = ((q1 - backup)**2).mean()
		lossQ2 = ((q2 - backup)**2).mean()
		lossQ = lossQ1 + lossQ2

		# wait to be loged
		qInfo = dict(Q1Vals=q1.detach().cpu().numpy(),
			Q2Vals=q2.detach().cpu().numpy())
		return lossQ, qInfo

	# calculate SAC pi loss
	def compute_loss_pi(self, data):
		observation = torch.tensor(data['observation'], dtype=torch.float32).cuda()
		pi, logProbPi = self.ac.pi(observation)
		q1Pi = self.ac.q1(observation, pi)
		q2Pi = self.ac.q2(observation, pi)
		qPi = torch.min(q1Pi, q2Pi)

		# Entropy-regularized policy loss
		lossPi = (self.alpha * logProbPi - qPi).mean()

		# wait to be logged
		piInfo = dict(LogPi=logProbPi.detach().cpu().numpy())

		return lossPi, piInfo

	# update the parameters of network
	def update(self,data):
		# first run one gradient descent step for Q1 and Q2
		self.qOptimizer.zero_grad()
		lossQ, _ = self.compute_loss_q(data) # _ is the log info 
		lossQ.backward()
		self.qOptimizer.step()

		# wait for record
		# ......

		# Freeze Q networks, save resource
		for p in self.qParams:
			p.requires_grad = False

		# run one gradient descent step for pi.
		self.piOptimizer.zero_grad()
		lossPi, _ = self.compute_loss_pi(data) # _ is the log info
		lossPi.backward()
		self.piOptimizer.step()

		# Unfreeze Q-network, optimize it at next DDPG step
		for p in self.qParams:
			p.requires_grad = True

		# wait for record
		# ......

		# Finally, update target networks by polyak averageing
		soft_update(self.acTarget, self.ac, self.polyak)

 	# ============================ test ======================================
	def get_action(self, observation, deterministic=False):
 		observation = torch.tensor(observation, dtype=torch.float32).cuda()
 		return self.ac.act(observation, deterministic)

	def test_agent(self, step=-1):
		# print("begin to test at {} step".format(step))
		for j in range(self.numTestEpisodes):
 			observation = self.testEnv.reset()
 			done = False
 			epochReward = 0
 			epochLength = 0

 			while not(done or (epochLength == self.maxEpochLen)):
 				# take deterministic actions at test time
 				observation, reward, done, _ = self.testEnv.step(self.get_action(observation, True))
 				epochReward += reward
 				epochLength += 1
		# print("The epoch reward is {}, the epoch length is {}".format(epochReward, epochLength))
		return epochReward

	# ============================== train ====================================
	def train(self):
		totalSteps = self.stepsPerEpoch * self.maxEpochs
		# startTime = time.time()
		# init env
		observation = self.env.reset()
		epochReward = 0
		epochLength = 0

		# for log
		testReward = np.array([])

		for t in tqdm.tqdm(range(totalSteps)):
			# make actions
			if t > self.startSteps:
				action = self.get_action(observation)
			else:
				action = self.env.action_space.sample()

			# step the env
			nextObervation, reward, done, _ = self.env.step(action)
			epochReward += reward
			epochLength += 1

			done = False if epochLength==self.maxEpochLen else done

			# Store the state into replay buffer
			self.replayBuffer.add(observation, action, reward, nextObervation, done)

			observation = nextObervation

			# end trajectory and init the env
			if done or (epochLength == self.maxEpochLen):
				# waite to be loged
				# .....
				observation = self.env.reset()
				epochReward = 0
				epochLength = 0

			# update the network
			if t >= self.updateAfter and t % self.updateEvery == 0:
				for j in range(self.updateEvery):
					batch = self.replayBuffer.sample_batch(self.batchSize)
					self.update(data=batch)

			# enc of epoch
			if (t+1) % self.stepsPerEpoch == 0:
				epoch = (t+1) // self.stepsPerEpoch

				# save model
				if (epoch % self.saveFreq == 0) or (epoch == self.maxEpochs):
					torch.save(self.acTarget.state_dict(), self.savePath/'modelStateDict.pt')

				# test
				testReward = np.append(testReward, self.test_agent(t))
				np.savez(self.savePath/'testReward.npz', hist=testReward)