from SingleAgent import SAC
import gym

env = gym.make('Walker2d-v2')
savePath = ''
sac = SAC.SAC(env, savePath)

sac.train()