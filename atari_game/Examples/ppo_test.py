import sys
sys.path.append("..")

import Agent
import gym
import numpy as np

import argparse

def main(args):
    env = gym.make(args.env)
    solved_reward = args.solved_reward

    if isinstance(env.action_space, gym.spaces.Box):
        solved_reward = args.solved_reward
        latent_dim = [64, 32]
        agent = Agent.PPOAgent(env, 
                               "./", 
                               seed=args.seed,
                               latent_dim=latent_dim,
                               lr=3e-4,
                               learning_epoch=80,
                               max_episodes=10000,
                               max_timestep=1500,
                               learning_freq=4000,
                               solved_reward=230,
                               )

    elif isinstance(env.action_space, gym.spaces.Discrete):
        config = {"solved_reward":145}
        # latent_dim = [64, 64]
        agent = Agent.PPOAgent(env, 12, "./",)

    agent.train(is_render=args.render)


if __name__ == "__main__":
    my_args = argparse.ArgumentParser()
    my_args.add_argument('--env', type=str, default='LunarLander-v2')
    my_args.add_argument('--render', type=bool, default=False)
    my_args.add_argument('--solved_reward', type=float, default=230)
    my_args.add_argument('--seed', type=int, default=None)
    args = my_args.parse_args()

    main(args)
