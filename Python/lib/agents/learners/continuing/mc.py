# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:34:27 2021

@author: Daniel Mastropietro
@description: Monte-Carlo learners on continuing tasks
"""
import copy
from enum import unique, Enum

import numpy as np

from .. import GenericLearner, AlphaUpdateType


class LeaMC(GenericLearner):
    """
    Monte Carlo learning algorithm using step size `alpha` and discount `gamma`
    applied to a generic environment defined in the gym.Env class of openAI.

    Arguments:
    env: gym.Env
        The environment where learning takes place.
    """

    def __init__(self, env, alpha=1.0, gamma=1.0,
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
        self.V = self.V_start   # Average reward over the simulation time
        self.Q = self.Q_start
        self.G = None           # Baseline-corrected cumulative reward for every time step (Trunk Reservation paper, pag. 4)
        self.averageReward = 0.0     # Average return (used as baseline to correct the reward in G)

        # Historic values of the value functions
        self.V_hist = []
        self.Q_hist = []

        super().reset()

    # Overrides superclass method
    def reset_supporting_attributes(self):
        "Resets the dictionary containing the number of times a state is visited"
        self.dict_state_counts = dict()

    # Overrides superclass method
    def reset_return(self):
        "Resets the observed return during the simulation period"
        self.G = None
        self.averageReward = 0.0

    # Overrides superclass method
    def reset_value_functions(self):
        #print("MC Learner of value functions: Resetting value functions...")
        # Value functions
        self.V = self.V_start
        self.Q = self.Q_start
        # Historic values
        self.V_hist = []
        self.Q_hist = []

    def learn(self, T):
        """
        Computes the average reward V and the return G(t) for every simulation time t as the sum of the difference of\
        each reward at t and the baseline value, chosen as the average reward V, observed over the whole simulation period.

        Arguments:
        T: int
            Time corresponding to the end of the simulation at which Monte-Carlo learning occurs.
        """
        # The value function is set as the average reward over the whole simulation time (constant for all states)
        self.V = self.averageReward

        rewards_history = copy.deepcopy(self.getRewards())
        #print("Rewards history: {}".format(rewards_history))
        rewards_history.reverse()
        #print("Rewards history reversed: {}".format(rewards_history))
        delta_rewards_history = np.array(rewards_history) - self.V
        #print("Delta rewards history reversed: {}".format(delta_rewards_history))
        self.G = np.nancumsum(delta_rewards_history)
        # Reverse the G just computed because we want to have t indexing the values of G from left to right of the array
        self.G = self.G[::-1]

        #print("Value function (average reward in simulation): {}".format(self.V))
        #print("Baseline-corrected return: {}".format(self.G))

        # Assert that G(0) is "0". Note that we compare with the maximum G(t) value because the "0" value is relative to this value
        # (o.w. the assertion may fail... e.g. when G(0) ~ 1E-8... but in that case Gmax ~ 1E7... so G(0) is really "0"!
        Gmax = np.max( np.abs(self.G) )
        if self.__class__ == LeaMC.__class__:   # Note: isinstance(self, LeaMC) gives also True when the type is LeaFV!! (I guess it's because LeaFV inherits from LeaMC)
            assert np.isclose(self.G[0] / (Gmax + 1), 0.0), "G(0) is almost 0 ({})".format(self.G[0])
        else:
            print("The value of G(0) is {}".format(self.G[0]))

    def updateAverageReward(self, t, reward):
        """
        Updates the average reward observed until t, as long as the reward is not NaN

        Arguments:
        t: int
            Current simulation time (discrete).
            It is assumed that the time for the first iteration is 0.

        reward: float
            R(t): reward received by the agent when transition to state S(t+1).
            If the reward is NaN, the average reward is not updated.
            This is equivalent to considering that the reward in that case is the same to the average reward observed
            so far.
        """
        if not np.isnan(reward):
            self.averageReward = (t * self.averageReward + reward) / (t + 1)

    def _record_history(self):
        self.V_hist += [self.V]
        #self.Q_hist += [self.Q]

    #----- GETTERS -----#
    def getAverageReward(self):
        return self.averageReward

    def getReturn(self):
        return self.G

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
