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
		save_path,
		actor_critic=ActorCritic.MLP_ActorCritic,
		ac_kwargs=dict(),
		random_seed=0,
		steps_per_epoch=4000,
		max_epochs=100,
		replay_buffer_size=int(1e6),
		gamma=0.99,
		polyak=0.995,
		lr=1e-3,
		alpha=0.2,
		batch_size=100,
		start_steps=10000,
		update_after=1000,
		update_every=50,
		num_test_episodes=10,
		max_epoch_len=1000,
		save_freq=1,
		**kwargs):

		# set random seed
		torch.manual_seed(random_seed)
		np.random.seed(random_seed)

		# set self parameter
		self.ac_kwargs = ac_kwargs
		self.steps_per_epoch = steps_per_epoch
		self.max_epochs = max_epochs
		self.replay_buffer_size = replay_buffer_size
		self.gamma = gamma
		self.polyak = polyak
		self.lr = lr
		self.alpha = alpha
		self.batch_size = batch_size
		self.start_steps = start_steps
		self.update_after = update_after
		self.update_every= update_every
		self.num_test_episodes = num_test_episodes
		self.max_epoch_len = max_epoch_len
		self.save_path = Path(save_path)
		self.save_freq = save_freq

		# set env
		self.env = env
		self.test_env = env
		self.observation_dim = env.observation_space.shape
		self.action_dim = env.action_space.shape[0]

		# Action limit for clamping, assumes all dimension share the same bound
		self.action_limit = env.action_space.high[0]

		# build AC learning network and target network
		self.ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).cuda()
		self.ac_target = actor_critic(env.observation_space, env.action_space, **ac_kwargs).cuda()

		# sync the parameters between two networks by hard-update
		hard_update(self.ac_target, self.ac)

		# Freeze target networks with respect to optimizers
		for paras in self.ac_target.parameters():
			paras.requires_grad = False

		# List of parameters for both Q-Networks (for convenience)
		self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

		# Set Memory ReplayBuffer
		self.replayBuffer = ReplayBuffer.SAC_ReplayBuffer(observation_dim=self.observation_dim,
			action_dim=self.action_dim, maxSize=replay_buffer_size)

		# set opimizers
		self.piOptimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
		self.qOptimizer = Adam(self.q_params, lr=self.lr)

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
			q1PiTarget = self.ac_target.q1(nextObervation, nextAction)
			q2PiTarget = self.ac_target.q2(nextObervation, nextAction)
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
		for p in self.q_params:
			p.requires_grad = False

		# run one gradient descent step for pi.
		self.piOptimizer.zero_grad()
		lossPi, _ = self.compute_loss_pi(data) # _ is the log info
		lossPi.backward()
		self.piOptimizer.step()

		# Unfreeze Q-network, optimize it at next DDPG step
		for p in self.q_params:
			p.requires_grad = True

		# wait for record
		# ......

		# Finally, update target networks by polyak averageing
		soft_update(self.ac_target, self.ac, self.polyak)

 	# ============================ test ======================================
	def get_action(self, observation, deterministic=False):
 		observation = torch.tensor(observation, dtype=torch.float32).cuda()
 		return self.ac.act(observation, deterministic)

	def test_agent(self, step=-1):
		# print("begin to test at {} step".format(step))
		for j in range(self.num_test_episodes):
 			observation = self.test_env.reset()
 			done = False
 			epochReward = 0
 			epochLength = 0

 			while not(done or (epochLength == self.max_epoch_len)):
 				# take deterministic actions at test time
 				observation, reward, done, _ = self.test_env.step(self.get_action(observation, True))
 				epochReward += reward
 				epochLength += 1
		# print("The epoch reward is {}, the epoch length is {}".format(epochReward, epochLength))
		return epochReward

	# ============================== train ====================================
	def train(self):
		totalSteps = self.steps_per_epoch * self.max_epochs
		# startTime = time.time()
		# init env
		observation = self.env.reset()
		epochReward = 0
		epochLength = 0

		# for log
		testReward = np.array([])

		for t in tqdm.tqdm(range(totalSteps)):
			# make actions
			if t > self.start_steps:
				action = self.get_action(observation)
			else:
				action = self.env.action_space.sample()

			# step the env
			nextObervation, reward, done, _ = self.env.step(action)
			epochReward += reward
			epochLength += 1

			done = False if epochLength==self.max_epoch_len else done

			# Store the state into replay buffer
			self.replayBuffer.add(observation, action, reward, nextObervation, done)

			observation = nextObervation

			# end trajectory and init the env
			if done or (epochLength == self.max_epoch_len):
				# waite to be loged
				# .....
				observation = self.env.reset()
				epochReward = 0
				epochLength = 0

			# update the network
			if t >= self.update_after and t % self.update_every == 0:
				for j in range(self.update_every):
					batch = self.replayBuffer.sample_batch(self.batch_size)
					self.update(data=batch)

			# enc of epoch
			if (t+1) % self.steps_per_epoch == 0:
				epoch = (t+1) // self.steps_per_epoch

				# save model
				if (epoch % self.save_freq == 0) or (epoch == self.max_epochs):
					torch.save(self.ac_target.state_dict(), self.save_path/'modelStateDict.pt')

				# test
				testReward = np.append(testReward, self.test_agent(t))
				np.savez(self.save_path/'testReward.npz', hist=testReward)