# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:15:37 2018

@author: voldemort
"""

import numpy as np
import pprint
import sys
from lib.envs.gridworld import GridworldEnv
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()
def value_iteration(env,theta=0.0001, discount=1.0):
    def one_step_lookahead(state,V):
        A=np.zeros(env.nA)
        for a in range(env.nA):
            for prob,next_state ,reward,done in env.P[state][a]:
                A[a]+=prob*reward(+discount*V[next_state])
            return A
    V = np.zeros(env.nS)
    while True:
        delta=0
        for s in range(env.nS):
            A=one_step_lookahead(s,V)
            bestaction=np.max(A)
            delta=max(np.abs(bestaction-V[s]))
            V[s]=bestaction
        if delta<theta:
            break
    policy=np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        A=one_step_lookahead(s,V)
        bestaction=np.argmax(A)
        policy[s,bestaction]=1.0
    return policy,V
policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")