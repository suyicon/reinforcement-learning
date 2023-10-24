import time
import gym
import numpy as np
import turtle
from Gridworld.CliffWalkingWapper import CliffWalkingWapper

env = gym.make("CliffWalking-v0")
env = CliffWalkingWapper(env)
env.reset()
while True:
    action = np.random.randint(0,4)
    (obs,reward,done,_,info) = env.step(action)
    env.render()
    if done:
        break