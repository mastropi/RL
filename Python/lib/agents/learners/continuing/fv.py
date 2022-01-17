# -*- coding: utf-8 -*-
"""
Created on Mon Dec 06 23:31:27 2021

@author: Daniel Mastropietro
@description: Fleming-Viot learners on continuing tasks
"""
import copy

import numpy as np

from .. import AlphaUpdateType
from .mc import LeaMC


class LeaFV(LeaMC):
    """
    Fleming-Viot learning algorithm using step size `alpha` and discount `gamma`
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
        super().__init__(env, alpha, gamma, adjust_alpha, alpha_update_type, alpha_min, V_start, Q_start, debug)
        # Dictionary that stores the stationary probability of states of interest, which are the keys of the dictionary
        self.probas_stationary = dict()

    #----- SETTERS -----#
    def setAverageReward(self, averageReward):
        self.averageReward = averageReward

    def setProbasStationary(self, probas_stationary):
        self.probas_stationary = probas_stationary

    #----- GETTERS -----#
    def getAverageReward(self):
        return self.averageReward

    def getProbasStationary(self):
        return self.probas_stationary
