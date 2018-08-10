# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:02:26 2018

@author: voldemort
"""

import numpy as np
import pprint
import sys
from lib.envs.gridworld import GridworldEnv
pp=pprint.PrettyPrinter(indent=2)
env=GridworldEnv()
def policy_eval(policy,env,discount=1.0,theta=0.00001):
    V=np.zeros(env.nS)
    while True:
        delta=0
        for s in range(env.nS):
            v=0
            for a,action_prob in enumerate(policy[s]):
                for prob,next_state,reward,done in env.P[s][a]:
                    v+=action_prob*prob*(reward+discount*V[next_state])
            delta=max(delta,np.abs(v-V[s]))
            V[s]=v
        if delta<theta:
            break
    return np.array(V)

randompolicy = np.ones([env.nS, env.nA]) / env.nA
v=policy_eval(randompolicy,env)
print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
