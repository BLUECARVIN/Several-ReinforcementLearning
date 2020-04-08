import sys
sys.path.append("..")

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import os
import copy
import pickle
import gym
import numpy as np
import random
from PIL import Image

from Utils import hard_update
from MLP import QNet
import ReplayBuffer


class DQNAgent(object):
	def __init__(self, 
		env,
		random_seed,
		save_path,
		q_net=QNet,
		gamma=0.99,
		batch_size=64,
		initial_eps = 0.5,
		end_eps = 0.01,
		eps_plan = 500000,
		lr=1e-3,
		learning_start=50000,
		learning_freq=4,
		frame_history_len=4,
		target_update_freq=10000,
		memory_size=1000000,
		max_steps = 10000000,
		**kwargs):
		"""
		DQN Agent

		paras:
			env: the gym environment
			seed: the random seed
			save_path: the path to save model parameters
			q_net: the Q learning network function
			gamma: the reward's decrease parameter
			initial_e: the initial prob to choose random action
			end_e: the end prob to choose random action
			lr: the optimizer's learning rate
			target_update_freq: the target netwok's update frequency
			test_freq: the test frequency
			learning_start: begin to learn after learning_start steps
			learning_freq: the training frequency 
			frame_history_len: how much frames should be feed to the model as one data
			memory_size: the maxmium size of replay buffer
		"""
		assert type(env.observation_space) == gym.spaces.Box
		assert type(env.action_space) == gym.spaces.Discrete

		# fix random seed
		torch.manual_seed(random_seed)
		np.random.seed(random_seed)
		random.seed(random_seed)

		# set env
		self.env = env
		self.test_env = copy.deepcopy(env)
		# get observation dim
		if len(env.observation_space.shape) == 1: # running on low-dimension observation(RAM)
			self.observation_dim = env.observation_space.shape[0]
		else:
			img_h, img_w, img_c = env.observation_space.shape
			self.observation_dim = frame_history_len 
		# get action dim
		self.action_dim = env.action_space.n

		# set Q network
		self.learning_Q = q_net(self.observation_dim, self.action_dim).cuda()
		self.target_Q = q_net(self.observation_dim, self.action_dim).cuda()
		# sync two networks' parameter
		hard_update(self.target_Q, self.learning_Q)

		# set replay buffer
		self.replay_buffer = ReplayBuffer.ReplayBuffer(memory_size, frame_history_len)

		# define learning Q network's optimizer
		self.optimizer = torch.optim.Adam(self.learning_Q.parameters(), lr=lr)
		# define loss function
		self.loss_func = nn.MSELoss()

		# initial other parameters
		self.gamma = gamma
		self.batch_size = batch_size
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.eps_plan = eps_plan
		self.learning_start = learning_start
		self.learning_freq = learning_freq
		self.frame_history_len = frame_history_len
		self.max_steps = max_steps
		self.target_update_freq = target_update_freq
		self.steps = 0
		self.save_path = save_path

		# set the eps
		self.eps = self.initial_eps

	# ============================ save and load model ===============================
	def save_model(self, name, path=None):
		if path:
			self.save_path = path
		if not os.path.isdir(self.save_path):
			os.makedirs(self.save_path)
		torch.save(self.target_Q.state_dict(), self.save_path + name + '.pt')
		print("The target model's parameters have been saved sucessfully!")

	def load_model(self, file_path):
		self.target_Q.load_state_dict(torch.load(file_path))
		hard_update(self.learning_Q, self.target_Q)
		print("The models' parameters have been loaded sucessfully!")


	# ============================ utils ==========================================
	def cal_eps(self):
		self.eps = self.initial_eps - (self.initial_eps - self.end_eps) / self.eps_plan * self.steps
		if self.eps < self.end_eps:
			self.eps = self.end_eps


	def pre_process(self, observation):
		img = np.reshape(observation, [210, 160, 3]).astype(np.float32)
		img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
		img = Image.fromarray(img)
		resized_screen = img.resize((84, 110), Image.BILINEAR)
		resized_screen = np.array(resized_screen)
		x_t = resized_screen[18:102, :]
		x_t = np.reshape(x_t, [84, 84, 1])
		return x_t.astype(np.uint8)

	# ============================= evaluate ======================================
	def get_exploration_action(self, state):
		sample = random.random()
		self.cal_eps()
		if sample > self.eps:
			state = torch.from_numpy(state).type(torch.float32).unsqueeze(0) / 255.0
			state = Variable(state).cuda()
			# action = torch.argmax(self.learning_Q(state)).detach().cpu()
			action = torch.argmax(self.learning_Q(state), dim=1).cpu()
			# action = self.learning_Q(state).data.max(1)[1].cpu()
		else:
			# action = int(np.random.uniform() * self.action_dim)
			# action = torch.from_numpy(action)
			action = torch.IntTensor([random.randrange(self.action_dim)])
		return action


	def get_exploitation_action(self, state):
		state = torch.from_numpy(state).type(torch.float32).unsqueeze(0) / 255.0
		state = Variable(state).cuda()
		action = torch.argmax(self.target_Q(state)).detach().cpu()
		return action

	# ============================= train ======================================
	def train(self, is_render=False, path=None):
		last_observation = self.env.reset()
		last_observation = self.pre_process(last_observation)
		mean_episode_reward = -float('nan')
		best_mean_episode_reward = -float('inf')
		log = {'mean_episode_reward':[], 'best_mean_episode_reward':[]}
		num_param_updates = 0
		episode_rewards = []
		one_episode_reward = []
		while self.steps < self.max_steps:
			# store lastest observation 
			last_index = self.replay_buffer.store_frame(last_observation)

			recent_observation = self.replay_buffer.encoder_recent_observation()
			# choose a random action if not state learning yet
			if self.steps < self.learning_start:
				action = random.randrange(self.action_dim)
			else:
				action = self.get_exploration_action(recent_observation)[0].numpy()

			# make a step
			observation, reward, done, _ = self.env.step(action)
			observation = self.pre_process(observation)
			one_episode_reward.append(reward)
			if is_render:
				self.env.render()

			# clip rewards between -1 and 1
			reward = max(-1.0, min(reward, 1.0))
			# store ohter info in replay memory
			self.replay_buffer.store_effct(last_index, action, reward, done)
			# if done, restat env
			if done:
				observation = self.env.reset()
				observation = self.pre_process(observation)
				episode_rewards.append(np.sum(one_episode_reward))
				one_episode_reward = []
			last_observation = observation


			# perform experience replay and train the network
			if ((self.steps > self.learning_start) and 
				(self.steps % self.learning_freq == 0) and 
				self.replay_buffer.can_sample):
				# get batch from replay buffer
				obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample_batch(self.batch_size)
				# turn all data to tensor
				obs_batch = Variable(torch.from_numpy(obs_batch).type(torch.float32) / 255.0).cuda()
				act_batch = Variable(torch.from_numpy(act_batch).long()).cuda()
				rew_batch = Variable(torch.from_numpy(rew_batch).type(torch.float32)).cuda()
				next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(torch.float32) / 255.).cuda()
				not_done_mask = Variable(torch.from_numpy(1 - done_mask).type(torch.float32)).cuda()


				# ================================ calculate bellman =========================================
				# get current Q value
				current_q_value = self.learning_Q(obs_batch).gather(1, act_batch.unsqueeze(1))
				# compute next q value based on which action gives max Q values
				next_max_q = self.target_Q(next_obs_batch).max(1)[0]
				next_q_values = not_done_mask * next_max_q
				# compute the target of the current q values
				target_q_values = rew_batch + (self.gamma * next_q_values)
				# compute bellman error
				bellman_error = target_q_values.view(-1, 1) - current_q_value
				# clip bellman error between [-1, 1]
				clipped_bellman_error = bellman_error.clamp(-1, 1)
				# * -1
				bellman_loss = -1. * clipped_bellman_error

				# optimize
				self.optimizer.zero_grad()
				current_q_value.backward(bellman_loss.data)
				self.optimizer.step()

				# update steps
				num_param_updates += 1
				# update network
				if num_param_updates % self.target_update_freq == 0:
					hard_update(self.target_Q, self.learning_Q)

			if len(episode_rewards) > 0:
				mean_episode_reward = np.mean(episode_rewards[-100:])
			if len(episode_rewards) > 100:
				best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

			log['mean_episode_reward'].append(mean_episode_reward)
			log['best_mean_episode_reward'].append(best_mean_episode_reward)

			if self.steps % 5000 == 0 and self.steps > self.learning_start:
				print("Steps: {}".format(self.steps))
				print("mean reward (lastest 100 episodes): {:.4f}".format(mean_episode_reward))
				print("best mean reward: {:.4f}".format(best_mean_episode_reward))
				print("episodes: {}".format(len(episode_rewards)))
				print("exploration: {:.4f}".format(self.eps))
				sys.stdout.flush()

				with open(self.save_path + 'log.pkl', 'wb') as f:
					pickle.dump(log, f)

				self.save_model('DQNtest', path=path)

			self.steps += 1