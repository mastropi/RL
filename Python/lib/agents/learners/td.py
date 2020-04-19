# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:57:16 2020

@author: Daniel Mastropetro
@description: Definition of Temporal Difference algorithms.
"""

import numpy as np

from . import Learner
from .value_functions import ValueFunctionApprox


class LeaTDLambda(Learner):
    """
    TD(Lambda) learning algorithm using step size `alpha`, discount `gamma`, and decay parameter `lmbda`
    applied to a discrete environment defined with the DiscreteEnv class of openAI's gym module.

    Args:
        env (gym.envs.toy_text.discrete.DiscreteEnv): the environment where the learning takes place.
    """

    def __init__(self, env, alpha=0.1, gamma=0.9, lmbda=0.8, debug=False):
        super().__init__()
        self.debug = debug

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

        # Reset the variables that store information about the episode
        self.reset()

    def setParams(self, alpha=None, gamma=None, lmbda=None):
        self.alpha = alpha if alpha else self.alpha
        self.gamma = gamma if gamma else self.gamma
        self.lmbda = lmbda if lmbda else self.lmbda

    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        self._updateZ(state)
        delta = reward + self.gamma * self.V.getValue(next_state) - self.V.getValue(state)
        self.V.setWeights( self.V.getWeights() + self.alpha * delta * self.z )

    def _updateZ(self, state):
        dev = 1
        self.z *= self.gamma * self.lmbda
        self.z[state] += dev


class LeaTDLambdaAdaptive(LeaTDLambda):
    
    def __init__(self, env, alpha=0.1, gamma=0.9, lmbda=0.8, debug=False):
        super().__init__(env, alpha, gamma, lmbda, debug)

    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        # Adaptive lambda
        state_value = self.V.getValue(state)
        delta = reward + self.gamma * self.V.getValue(next_state) - state_value
        # Define the relative state value change by dividing the change by the current state value
        # (if the current state value is 0, then set the change to infinite unless the change is 0 as well)
        delta_relative = delta / state_value if state_value != 0 \
                                         else 0. if delta == 0. \
                                         else np.Inf
        lambda_adaptive = 1 - np.exp( -np.abs(delta_relative) )
        # Update elegibility trace
        self._updateZ(state, lambda_adaptive)
        # Update the weights
        self.V.setWeights( self.V.getWeights() + self.alpha * delta * self.z )

        if self.debug:
            print("t: {}, delta = {:.3g} --> lambda = {:.3g}".format(t, delta, lambda_adaptive))
            print("\tV(s={}->{}) = {}".format(state, next_state, self.V.getValue(state)))
            if done:
                import pandas as pd
                pd.options.display.float_format = '{:,.2f}'.format
                print(pd.DataFrame( np.c_[self.z, self.V.getValues()].T, index=['z', 'V'] ))
    
            #input("Press Enter...")

    def _updateZ(self, state, lambda_adaptive):
        dev = 1
        # Multiply the PREVIOUS z, z(t-1), by gamma*lambda
        # IMPORTANT: lambda is computed with the information at time t!! (not at time t-1)
        self.z *= self.gamma * lambda_adaptive
        self.z[state] += dev
