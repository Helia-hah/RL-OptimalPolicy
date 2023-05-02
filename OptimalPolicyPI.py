
import numpy as np
from scipy.linalg import block_diag

def diagonalization(A, n_states, n_actions):
    
    A = A.reshape([n_states, n_actions])
    return block_diag(*list(A))

"""Iterative Policy Evaluation"""

def iterativePolicyEvaluation(env, Pi, gamma, tol=1e-10):
    """
        Iterative policy evaluation for a given policy
    """
    v = np.random.rand(env.n_states, 1)
    v[env.goal, 0] = 0.
    diff = float('inf')
    while diff > tol:
        v_new = Pi @ (env.r + gamma * env.P @ v)
        diff = np.amax(np.abs(v_new - v))
        v = v_new
    return v

"""Policy Iteration"""

import random
def policyIteration(env, gamma):
    

    #create a policy
    pi = np.zeros((env.n_states, env.n_actions))

    for i in range (0, env.n_states):

      column = random.randint(0,env.n_actions-1)
      pi[i][column] = 1
   

    policy_stable = False
    while policy_stable == False:

      # Diagonalization for pi  
      Pi = diagonalization(pi, env.n_states, env.n_actions)
   
      # iterativePolicyEvaluation
      v = iterativePolicyEvaluation(env, Pi, gamma, tol=1e-10)



      policy_stable = True

      for s in range (0,env.n_states):
        
        old_action = np.argmax(pi[s])
        new_action= np.argmax(env.r[4*s:4*(s+1)] + gamma * env.P[4*s:4*(s+1)] @ v)

        if old_action != new_action:
          policy_stable =False
          pi[s][old_action]=0
          pi[s][new_action]=1
     

    return  Pi , v
