from SingleAgent import SAC
import gym

env = gym.make('HalfCheetah-v2')
savePath = ''
sac = SAC.SAC(env, savePath)

sac.train()