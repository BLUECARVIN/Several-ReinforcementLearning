import sys
sys.path.append("..")

from Agent.AC import ActorCritic
from ReplayBuffer.ppo_buffer import PPOSimplestBuffer

import torch 
from torch import nn
from torch.optim import optimizer
import os 

import gym
import copy
import numpy as np
import random


class PPOAgent(object):
    def __init__(self,
                 env,
                 save_path,
                 random_seed=None,
                 AC=ActorCritic,
                 gamma=0.99,
                 lr=2e-3,
                 betas=(0.9, 0.999),
                 learning_epoch=4,
                 eps_clip=0.2,
                 log_interval=20,
                 device='cuda:0',
                 max_episodes=50000, 
                 max_timestep=300,
                 latent_dim=[64, 32],
                 learning_freq=2000,
                 solved_reward=145,
                 **kwargs):
        """
        PPOAgent

        paras:
            solved_reward： stop training if avg_reward > sloved_reward
        """

        # set env
        self.env = env
        self.test_env = copy.deepcopy(env)
        
        # value type observation environment
        assert type(env.observation_space) == gym.spaces.Box

        # fix random seed
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        if isinstance(env.observation_space, gym.spaces.Box):
            self.state_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            print(1)
            self.action_dim = env.action_space.n
            self.env_type = 'discrete'
        elif isinstance(env.action_space, gym.spaces.Box):
            print(2)
            self.action_dim = env.action_space.shape[0]
            self.env_type = 'continuous'


        # training parameters
        self.lr = lr
        self.betas = betas
        self.eps_clip = eps_clip
        self.log_interval = log_interval
        self.max_episodes = max_episodes
        self.max_timestep = max_timestep
        self.latent_dim = latent_dim
        self.learning_freq = learning_freq
        self.gamma = gamma
    
        self.solved_reward = solved_reward
        self.learning_epoch = learning_epoch

        self.steps = 0
        self.save_path = save_path
        self.device = device

        print(self.env_type)

        self.learning_policy = AC(self.state_dim, 
                                  self.action_dim,
                                  self.latent_dim, 
                                  device=self.device, 
                                  env_type=self.env_type)
        self.target_policy = AC(self.state_dim, 
                                self.action_dim, 
                                self.latent_dim, 
                                device=self.device,
                                env_type=self.env_type)
        self.target_policy.load_state_dict(self.learning_policy.state_dict())

        self.optimizer = torch.optim.Adam(self.learning_policy.parameters(), lr=self.lr, betas=self.betas)

        self.mseloss = nn.MSELoss()

        # replay buffer
        self.replay_buffer = PPOSimplestBuffer()

    # =========================== utils ================
    def save_model(self, name, path=None):
        if path:
            self.save_path = path     
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.target_policy.state_dict(), self.save_path + name + '.pt')
        print("The target model's parameters have been saved sucessfully!")

    def load_model(self, file_path):
        self.target_policy.load_state_dict(torch.load(file_path))
        hard_update(self.learning_policy, self.target_policy)
        print("The models' parameters have been loaded sucessfully!")
    
    def select_action(self, state, memory):
        if self.env_type == 'continuous':
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            return self.target_policy.act(state, memory).cpu().data.numpy().flatten()
        if self.env_type == 'discrete':
            return self.target_policy.act(state, memory)

    # ======================== training ================
    def update(self):
        # Monte Carlo Estimate of State Rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.replay_buffer.rewards), reversed(self.replay_buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        if self.env_type == 'discrete':
            old_states = torch.stack(self.replay_buffer.states).to(self.device).detach()
            old_actions = torch.stack(self.replay_buffer.actions).to(self.device).detach()
            old_logprobs = torch.stack(self.replay_buffer.logprobs)
        elif self.env_type == 'continuous':
            old_states = torch.squeeze(torch.stack(self.replay_buffer.states).to(self.device), 1).detach()
            old_actions = torch.squeeze(torch.stack(self.replay_buffer.actions).to(self.device), 1).detach()
            old_logprobs = torch.squeeze(torch.stack(self.replay_buffer.logprobs), 1).to(self.device).detach()

        # optimize policy
        for _ in range(self.learning_epoch):
            # Evaluateing old actions and values
            logprobs, state_values, dist_entropy = self.learning_policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mseloss(state_values, rewards) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.target_policy.load_state_dict(self.learning_policy.state_dict())

    def train(self, is_render=False, path=None):
        # Monte Carlo Estimate of State Rewards
        running_reward = 0
        avg_length = 0
        timestep = 0
        print("begin to train")
        for episode in range(self.max_episodes):
            state = self.env.reset()
            # print('now is {} episode'.format(episode))
            for t in range(self.max_timestep):
                timestep += 1
                # Runing target policy
                action = self.select_action(state, self.replay_buffer)
                state, reward, done, _ = self.env.step(action)

                # add to buffer
                self.replay_buffer.rewards.append(reward)
                self.replay_buffer.is_terminals.append(done)

                # update 
                if timestep % self.learning_freq == 0:
                    self.update()
                    self.replay_buffer.clear_buffer()
                    timestep = 0
                
                running_reward += reward
                if is_render:
                    self.env.render()
                
                if done:
                    break
            avg_length += t

            # stop training condition
            if running_reward > (self.log_interval * self.solved_reward):
                self.save_model("best_model.py", self.save_path)
                print("Training is done")
                break

            # logging
            if episode % self.log_interval == 0:
                avg_length = int(avg_length / self.log_interval)
                running_reward = int((running_reward / self.log_interval))

                print('At {} episode, the avg lengh is {}, the mean_reward is {}'.format(episode, avg_length, running_reward))
                running_reward = 0
                avg_length = 0


