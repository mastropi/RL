# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:34:27 2021

@author: Daniel Mastropietro
@description: Monte-Carlo learners on continuing tasks
"""

from enum import unique, Enum

import numpy as np

from .. import Learner

@unique  # Unique enumeration values (i.e. on the RHS of the equal sign)
class AlphaUpdateType(Enum):
    FIRST_STATE_VISIT = 1
    EVERY_STATE_VISIT = 2


class LeaMC(Learner):
    """
    Monte Carlo learning algorithm using step size `alpha` and discount `gamma`
    applied to a generic environment defined in the gym.Env class of openAI.

    Arguments:
    env: gym.Env
        The environment where learning takes place.
    """

    def __init__(self, env, alpha=0.1, gamma=1.0,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                 alpha_min=0.,
                 V_start=0.0, Q_start=0.0,
                 # TODO: (2021/10/25) Decide the purpose of V_start and Q_start... currently they are not used and I am not sure how they would be used for the estimation of V or the calculation of G
                 debug=False):
        super().__init__(env, alpha, adjust_alpha, alpha_update_type, alpha_min)
        self.debug = debug

        self.gamma = gamma
        self.dict_state_counts = dict()

        # Value functions and related quantities
        self.V_start = V_start
        self.Q_start = Q_start
        self.V = self.V_start
        self.Q = self.Q_start
        self.G = 0.0    # Average reward during the queue evolution until max simulation time

        # Historic values of the value functions
        self.V_hist = []
        self.Q_hist = []

        super().reset(reset_value_functions=True)

    # Overrides superclass method
    def reset_supporting_attributes(self):
        "Resets the dictionary containing the number of times a state is visited"
        self.dict_state_counts = dict()

    # Overrides superclass method
    def reset_return(self):
        "Resets the observed return during the simulation period"
        self.G = 0.0

    # Overrides superclass method
    def reset_value_functions(self):
        print("MC Learner of value functions: Resetting value functions...")
        # Value functions
        self.V = self.V_start
        self.Q = self.Q_start
        # Historic values
        self.V_hist = []
        self.Q_hist = []

    def update_state_counts(self, state):
        "Updates the number of times the given state has been visited during the learning process"
        # Create a string version of the state, so that we can store lists as dictionary keys
        # which are actually not accepted as dictionary keys (with the error message "list type is unhashable").
        # e.g. [3, 1, 5] becomes '[3, 1, 5]'
        state_str = str(state)
        if state_str in self.dict_state_counts.keys():
            self.dict_state_counts[state_str] += 1
        else:
            self.dict_state_counts[state_str] = 1

    def updateG(self, t, reward):
        """
        Updates the average return G with the observed reward

        Arguments:
        t: int
            Current simulation time (discrete).
            It is assumed that the time for the first iteration is 0.

        reward: float
            R(t+1): reward received by the agent when transition to state S(t+1).
        """
        self.G = (t * self.G + reward) / (t + 1)

    def learn(self, t, state, reward):
        """
        Updates the value function with the observed average return G

        Arguments:
        t: float
            Discrete time at which learning takes place.
        """
        self.V = self.G
        self.record_learning(state, reward)
        self._record_history()

    def _record_history(self):
        self.V_hist += [self.V]
        #self.Q_hist += [self.Q]

    #----- GETTERS -----#
    def getV(self):
        return self.V

    def getQ(self):
        return self.Q

    def getVStart(self):
        return self.V_start

    def getQStart(self):
        return self.Q_start

    def getVHist(self):
        return self.V_hist

    def getQHist(self):
        return self.Q_hist

    #----- SETTERS -----#
    def setVStart(self, V):
        self.V_start = V

    def setQStart(self, Q):
        self.Q_start = Q

