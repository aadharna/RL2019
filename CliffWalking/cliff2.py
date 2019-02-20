import gym
import time
import random

env = gym.make('CliffWalking-v0')
for i_episode in range(1):
    observation = env.reset()
    act=1
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(act)
        print(observation,reward,done,info)
        act = random.randint(0,action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
