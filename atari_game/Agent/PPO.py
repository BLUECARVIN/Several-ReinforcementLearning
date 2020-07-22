import sys
sys.path.append("..")

from Agent.AC import ActorCritic
from ReplayBuffer.ppo_buffer import PPOSimplestBuffer

import torch 
from torch import nn
from torch.optim import optimizer

import gym
import copy
import numpy as np
import random


class PPOAgent(object):
    def __init__(self,
                 env,
                 random_seed,
                 save_path,
                 AC=ActorCritic,
                 gamma=0.99,
                 lr=2.e-3,
                 betas=(0.9, 0.999),
                 learning_epoch=4,
                 eps_clip=0.2,
                 log_interval=20,
                 device='cuda:0',
                 max_episodes=50000,
                 max_timestep=300,
                 latent_dim=64,
                 learning_freq=2000,
                 solved_reward=230,
                 **kwargs):
        """
        PPOAgent

        paras:
            solved_reward： stop training if avg_reward > sloved_reward
        """

        # set env
        self.env = env
        self.test_env = copy.deepcopy(env)
        
        # discrete environment
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        # fix random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # To DO 
        # now is just for LunarLander-v2
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

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

        self.learning_policy = AC(self.state_dim, self.action_dim, self.latent_dim, device=self.device)
        self.target_policy = AC(self.state_dim, self.action_dim, self.latent_dim, device=self.device)
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
        torch.save(self.target_Q.state_dict(), self.save_path + name + '.pt')
        print("The target model's parameters have been saved sucessfully!")

    def load_model(self, file_path):
        self.target_Q.load_state_dict(torch.load(file_path))
        hard_update(self.learning_Q, self.target_Q)
        print("The models' parameters have been loaded sucessfully!")

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
        rewards = (rewards - rewards) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.replay_buffer.states).to(self.device).detach()
        old_actions = torch.stack(self.replay_buffer.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.replay_buffer.logprobs)

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
        for episode in range(self.max_episodes):
            state = self.env.reset()
            for t in range(self.max_timestep):
                timestep += 1

                # Runing target policy
                action = self.target_policy.act(state, self.replay_buffer)
                state, reward, done, _ = self.env.step(action)

                # add to buffer
                self.replay_buffer.rewards.append(reward)
                self.replay_buffer.is_terminals.append(done)

                # update 
                if timestep % self.learning_freq == 0:
                    self.update()
                    self.replay_buffer.clear_memory()
                    timestep = 0
                
                running_reward += reward
                if is_render:
                    self.env.render()
                
                if done:
                    break
            avg_length += t

        # stop training condition
        if running_reward > (self.log_interval * solved_reward):
            self.save_model("best_model.py", self.save_path)
            print("Training is done")

        # logging
        if episode % self.log_interval == 0:
            avg_length = int(avg_length / self.log_interval)
            running_reward = int((running_reward / self.log_interval))

            print('At {} episode, the avg lengh is {}, the mean_reward is {}'.format(episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

