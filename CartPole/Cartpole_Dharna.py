#!/usr/bin/env python
# coding: utf-8

# ### Aaron Dharna
# #### 2/15/2019
# ##### MC Prediction and Control on Cartpole. 

# In[1]:


import sys
import gym
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# In[2]:


env = gym.make('CartPole-v0')


# `Observation`: 
# 
#         Type: Box(4)
#         Num	Observation                 Min         Max
#         0	Cart Position             -4.8            4.8
#         1	Cart Velocity             -Inf            Inf
#         2	Pole Angle                 -24 deg        24 deg
#         3	Pole Velocity At Tip      -Inf            Inf
#         
# `Actions`:
# 
#         Type: Discrete(2)
#         Num	Action
#         0	Push cart to the left
#         1	Push cart to the right
#         
# Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
#     
# Reward:
# 
#         Reward is 1 for every step taken, including the termination step
#         
# Starting State:
# 
#         All observations are assigned a uniform random value in [-0.05..0.05]
#         
# Episode Termination:
# 
#         Pole Angle is more than 12 degrees
#         Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
#         Episode length is greater than 200
#         
#         Solved Requirements:
#         Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

# In[4]:


for i_episode in range(1):
    observation = env.reset()
    act=1
    for t in range(200):
        env.render()
        action = env.action_space.sample()
        #print(action)
        observation, reward, done, info = env.step(act)
        #print(observation, reward, done)
        time.sleep(0.1)
        act = random.randint(0,1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


# In[5]:


print(env.observation_space)
print(env.action_space)


# In[124]:


def discritize(CVB=(-3, 3), PVB=(-3, 3)):
    """
    CVB: default cart velocity bounds = -3, 3
    PVB: default pole velocity bounds = -3, 3
    
    For Cartpole, we have:

        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
    """
    assert(CVB[0] < CVB[1] and PVB[0] < PVB[1])
    
    cart_position_space = np.arange(-2.5, 2.7, .3)
    cart_velocity_space = np.linspace(CVB[0], CVB[1], 5)
    pole_angle_space = np.arange(-13, 14, 2)
    #pole_velocity_space = np.linspace(PVB[0], PVB[1], 5)
    
    return cart_position_space, cart_velocity_space, pole_angle_space


# In[125]:


cps, cvs, pas = discritize()


# In[3]:


def toDegrees(radians):
    return radians*(180/np.pi)

def getDiscreteStateFromObs(obs):
    poleAngleDeg = toDegrees(obs[2])
    cartPos = None
    poleAng = None
    
    #if the position of the cart is in a fail state: 
    # indicate so with a state 0 or len(cps) - 1. 
    # the other states in cps are valid and useful because certain states
    # will have value based on how close they are to failure conditions.
    if obs[0] <= -2.4:
        cartPos = 0
    elif obs[0] >= 2.4:
        cartPos = len(cps) - 1
    else:
        cartPos = np.argmin(abs(obs[0] - cps))
    
    #same as above, but now for the pole degree. 
    if poleAngleDeg <= -12:
        poleAng = 0
    elif poleAngleDeg >= 12:
        poleAng = len(pas) - 1
    else:
        poleAng = np.argmin(abs(poleAngleDeg - pas))
    
    #since there is not a terminating velocity condition, 
    # we do not need to collect failure states into a single 
    # accumulating index. 
    cartVel = np.argmin(abs(obs[1] - cvs))
    
    #I believe we do not need to use poleVelocity 
    # since this indicates the amount of force needed (something we do not have control over)
    # to save the pole from continuing along it's given trajectory.
    #poleVel = np.argmin(abs(obs[3] - pvs))
    return (cartPos, cartVel, poleAng)


# In[10]:


state = getDiscreteStateFromObs(env.reset())
state


# ## MC control
# 
# Greedy in the Limit with Infinite Exploration (GLIE)

# In[90]:


def e_greedy_policy_creation(Q, epsilon, nA):
    """
    Q: Our Q table. 
      Q[state] = numpy.array
      Q[state][action] = float.
    epsilon: small value that controls exploration.
    nA: the number of actions available in this environment
    
    return: an epsilon-greedy policy specific to each state.
    """
    #policy[state][action] -> probability that the action should be chosen
    policy = defaultdict(lambda: np.ones(nA))
    
    #for each state in Q, build an epsilon-greedy policy
    for state in Q.keys():
        policy[state] *= epsilon/nA
        bestAction = np.argmax(Q[state])
        policy[state][bestAction] = 1 - epsilon + (epsilon/nA)
        
    return policy

def generate_episode_from_policy(env, policy):
    """
    env: the openAi gym environemnt.
    policy: epsilon greedy policy specified for each state we have seen.
    
    return: a episode of the environment playing out according to our given policy.
    """
    nA = env.action_space.n
    episode = []
    state = env.reset()
    state = getDiscreteStateFromObs(state)
    
    #continue until we are told to stop.
    while True:
        #random choice if the state has not been seen before
        # It will get a value update on the next pass. 
        if state not in policy:
            policy[state] *= 1/nA
            
        #chose an action following the policy specified for the given state we are in.
        action = np.random.choice(np.arange(nA), p=policy[state])
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        #becuase of the nature of for loops, state will persist at the start of next loop
        # until it is overwritten by the next_state.
        state = getDiscreteStateFromObs(next_state)

        if done:
            break
            
    return episode


# In[118]:


def mc_control_GLIE(env, num_episodes, alpha=1.0, gamma=1.0):
    """
    env: the openAi gym environemnt.
    num_episodes: the number of episodes to train for
    alpha: step size by which to move the Q table
    gamma: discount factor for future rewards. 
    
    return:
    Optimal Q* table
    Optimal Policy* (greedy creation from Q*)
    """
    nA = env.action_space.n
    # initialize empty dictionaries of arrays
    #Q[state][action]
    Q = defaultdict(lambda: np.zeros(nA))
    #N = defaultdict(lambda: defaultdict(int))  
    
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        epsilon = max(1.0/((i_episode/8000)+1), 0.2)

        policy = e_greedy_policy_creation(Q, epsilon, nA)
        
        episode = generate_episode_from_policy(env, policy)
        
        seen = []
        
        #If I had written every-visit MC control, I could have actually 
        # vectorized some of this code to make it faster. However, I wrote this for 
        # first visit, therefore, we need the if conditional, and cannot always count on the 
        # further enumeration. 
        
        #for each state/action/reward triple in the episode
        for t, (state, action, reward) in enumerate(episode):
            
            #if the state has not been seen before, update its value estimate
            if state not in seen:
                seen.append(state)
                G = 0
                
                #Go to the end of the episode to get the value of the state according to current policy
                for fi, (fstate, faction, freward) in enumerate(episode[t:]):
                    G += (gamma**fi)*freward
                
                Q[state][action] += alpha*(G - Q[state][action])
                
    #create our policy, piPrime based on our final Q table. 
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
            
    return policy, Q


# In[ ]:


# start = time.time()
# policy_glie, Q_glie = mc_control_GLIE(env, 150000, 0.01)
# print("learning time: ", time.time() - start)


# In[ ]:


for i_episode in range(1):
    observation = env.reset()
    for t in range(250):
        env.render()
        state = getDiscreteStateFromObs(observation)
        action = policy_glie[state]
        observation, reward, done, info = env.step(action)
        time.sleep(0.1)
        if done:
            print("final state: {}".format(getDiscreteStateFromObs(observation)))
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


# In[123]:


if __name__ == "__main__":
    
    start = time.time()
    policy_glie, Q_glie = mc_control_GLIE(env, 150000, 0.01)
    print("learning time: ", time.time() - start) 
    
    for i_episode in range(1):
        observation = env.reset()
        for t in range(250):
            env.render()
            state = getDiscreteStateFromObs(observation)
            action = policy_glie[state]
            observation, reward, done, info = env.step(action)
            time.sleep(0.1)
            if done:
                print("final state: {}".format(getDiscreteStateFromObs(observation)))
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

