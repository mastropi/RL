# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:36:13 2020

@author: Daniel Mastropietro
@description: Definition of Monte Carlo algorithms
"""

import numpy as np

from . import Learner


class LeaMCLambda(Learner):
    """
    Monte Carlo learning algorithm using step size `alpha`, discount `gamma`, and decay parameter `lmbda`  
    applied to a discrete environment defined with the DiscreteEnv class of openAI's gym module.

    Args:
        env (gym.envs.toy_text.discrete.DiscreteEnv): the environment where the learning takes place.
    """

    def __init__(self, env, alpha=0.1, gamma=0.9, lmbda=0.8):
        # Attributes that MUST be presented for all TD methods
        self.env = env
        self.V = ValueFunctionApprox(self.env.getNumStates())
        self.Q = None
        self.alpha = alpha
        self.gamma = gamma
        
        # Attributes specific to the current TD method
        self.lmbda = lmbda
        
        # Reset the variables that store information about the episode 
        self.reset()      

    def reset(self):
        "Resets the variables that store information about the episode"
        # Store the states visited in the episode
        self.states = []
        
        ### All the attributes that follow are used to store information about the
        ### n-step returns and the lambda returns (i.e. the lambda-weighted average
        ### n-step returns for ALL time steps in the episode, from t=0, ..., T-1,
        ### where T is the time at which the episode terminates).
        ### Note that this information is stored in lists as opposed to numpy arrays
        ### because we don't know their size in advance (as we don't know when the episode
        ### will terminate) and increasing the size of numpy arrays is apparently less efficient
        ### than increasing the size of lists
        ### (Ref: https://stackoverflow.com/questions/568962/how-do-i-create-an-empty-array-matrix-in-numpy)
        # n-step return (G(t:t+n) 0 <= t <= T-1, 1 <= n <= T-t-1)
        # The array is a 1d array indexed by t only because the n's are accumulated in the sum
        # that then is used to compute G(t,lambda))
        self._G_list = []
        # lambda-return (G(t,lambda))
        self._Glambda_list = []

    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        "Learn the prediction problem: estimate the state value function"
        self.states += [state]
        self._updateG(t, reward, done)

        if done:
            # This means t+1 is the terminal time, T (recall that we WERE in time t and we STEPPED INTO time t+1)
            T = t + 1
            Glambda = np.array(self._Glambda_list)

            # Update the weights recursively from time 0 to T-1
            for tt in np.arange(T):
                # Error, where the lambda-return is used as the current value function estimate
                delta = Glambda[tt] - self.V.getValue(self.states[tt])

                # Gradient of V: it must have the same size as the weights
                # In the linear case the gradient is equal to the feature associated to state s,
                # which is stored at column s of the feature matrix X.
                gradient_V = self.V.X[:, self.states[tt]]  
                    ## The above returns a ROW vector which is good because the weights are stored as a ROW vector

                # Update the weights based on the error observed at each time step and the gradient of the value function
                self.V.setWeights( self.V.getWeights() + self.alpha * delta * gradient_V )

            # Reset the variables in the object that contain information about the episode
            self.reset()  

    def _updateG(self, t, reward, done):
        times_reversed = np.arange(t, -1, -1)  # This is t, t-1, ..., 0

        # Add the latest available return G(t:t+1), corresponding to the current time t, to the list
        # of n-step returns and initialize it to 0 (before it is updated with the currently observed reward
        # immediately below)
        self._G_list += [0]
        # Update all the n-step G(tt:tt+n) returns, from time tt = 0, ..., t-1, using the new observed reward.
        # The gamma discount on the observed reward is stronger (i.e. more discount) as we move from time
        # t-1 down to time 0, that's why the exponent of gamma is the array of reversed times constructed above.
        assert len(self._G_list) == len(times_reversed), \
                "Length of _G_list ({}) coincides with length of times_reversed ({})" \
                    .format(len(self._G_list), len(times_reversed))
        self._G_list = [g + self.gamma**n * reward for (g, n) in zip(self._G_list, times_reversed)]

        # Update the estimates of the lambda-returns G(tt,lambda), for time tt = 0, ..., t-1
        self._Glambda_list += [0]
        if not done:
            # This implies adding the corresponding n-step return weighted by lambda**(n-1)
            # Since these updates are done in a cumulative way, n can actually be computed from t as done below.
            ## We also include time = 0 because we have already added the current reward to _G_list
            assert len(self._Glambda_list) == len(self._G_list), \
                    "Length of _Glambda_list ({}) coincides with length of _G_list ({})" \
                        .format(len(self._G_list), len(times_reversed))
            self._Glambda_list = [glambda + self.lmbda**n * g
                                  for (glambda, g, n) in zip(self._Glambda_list, self._G_list, times_reversed)]
        else:
            # We finalize the computation of G(t,lambda) by scaling the sum so far with (lambda - 1)
            # and adding the final contribution from the latest reward (with no (lambda-1)-scaling)
            self._Glambda_list = [(self.lmbda - 1) * glambda + self.lmbda**n * g
                                  for (glambda, g, n) in zip(self._Glambda_list, self._G_list, times_reversed)]


class ValueFunctionApprox:
    "Class that contains information about the estimation of the state value function"

    def __init__(self, nS):
        "nS is the number of states"
        self.nS = nS
        self.weights = np.zeros(nS)
        # The features are dummy or indicator functions, i.e. each column of the
        # feature matrix X represents a feature and each row represents a state. Assuming we
        # order the states in columns in the same way we order them in rows, the X matrix
        # is a diagonal matrix
        self.X = np.eye(nS)

    def getValue(self, state):
        # TODO: Apply the generalization to any set of features X (i.e. uncomment the line below)
        #v = np.dot(self.weights[state], self.X[:,state])
        v = self.weights[state]
        return v

    def getValues(self):
        # TODO: Apply the generalization to any set of features X (i.e. uncomment the line below)
        #return np.dot(self.weights, self.X)
        return self.weights

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights
