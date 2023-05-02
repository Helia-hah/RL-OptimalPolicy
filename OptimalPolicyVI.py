
import numpy as np
from scipy.linalg import block_diag

def diagonalization(A, n_states, n_actions):
    
    A = A.reshape([n_states, n_actions])
    return block_diag(*list(A))


"""Value Iteration"""

def valueIteration(env, gamma):

    tol=1e-10

    # create v
    v = np.random.rand(env.n_states, 1)
    v[env.goal, 0] = 0


    diff= 10
    while diff > tol:

      diff=0
      v_vec = np.zeros((env.n_states,1))
      for s in range (0,env.n_states):
        new_v = np.amax(env.r[4*s:4*(s+1)] + gamma * env.P[4*s:4*(s+1)] @ v)
        diff = max(diff,abs(v[s]-new_v))
        v_vec[s] = new_v
      v = v_vec


    pi = np.zeros((env.n_states, env.n_actions))
    for s in range (0,env.n_states):
      new_a= np.argmax(env.r[4*s:4*(s+1)] + gamma * env.P[4*s:4*(s+1)] @ v)
      pi[s][new_a]=1

    # Diagonalization for pi
    Pi = diagonalization(pi, env.n_states, env.n_actions)


    return Pi , v