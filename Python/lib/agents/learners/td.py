# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:57:16 2020

@author: Daniel Mastropetro
@description: Definition of temporal difference algorithms.
Each class should:
a) Implement the following attributes:
    - env --> the environment on which the learning takes place
    - V --> an object containing the information on how the state value function is ESTIMATED.
    - Q --> an object containing the information on how the state-action value function is ESTIMATED.
    - alpha --> the learning rate
    - gamma --> the reward discount parameter
b) Implement the following methods:
    - learn_pred_V() --> prediction problem: learn the state value function under the current policy (V(s))
    - learn_pred_Q() --> prediction problem: learn the action-value function under the currenty policy (Q(s,a))
    - learn_ctrl_policy() --> control problem: learn the optimal policy
"""

import numpy as np

from . import Learner


#__all__ = [ "LeaTDLambda" ]

class LeaTDLambda(Learner):
    """
    TD(Lambda) learning algorithm using step size `alpha`, discount `gamma`, and decay parameter `lmbda`
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
        # Eligibility traces
        self.z = np.zeros(self.env.getNumStates())

    def setParams(self, alpha=None, gamma=None, lmbda=None):
        self.alpha = alpha if alpha else self.alpha
        self.gamma = gamma if gamma else self.gamma
        self.lmbda = lmbda if lmbda else self.lmbda

    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        self._updateZ(state)
        delta = reward + self.gamma * self.V.getValue(next_state) - self.V.getValue(state)
        delta *= self.alpha
        self.V.setWeights( self.V.getWeights() + delta * self.z )

    def _updateZ(self, state):
        dev = 1
        self.z *= self.gamma * self.lmbda
        self.z[state] += dev

    def getZ(self):
        return self.z


# TODO: (2020/04/10) This class should cover ALL value functions whose estimation is done via approximation
# (i.e. using a parameterized expression whose parameters are materialized as a vector of weights)
# So the constructor should receive:
# - the dimension of the weights
# - the features x that are affected by the weights (possibly in a linear manner
# or perhaps in a nonlinear manner as well!)
class ValueFunctionApprox:
    "Class that contains information about the estimation of the state value function"

    def __init__(self, nS):
        "nS is the number of states"
        self.nS = nS
        self.weights = np.zeros(nS)

    def getValue(self, state):
        v = self.weights[state]
        return v

    def getValues(self):
        return self.weights

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights
