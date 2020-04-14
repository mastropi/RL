# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:35:13 2020

@author: Daniel Mastropietro
@description: Definition of classes that are common to all environments.
"""

import numpy as np

from gym.envs.toy_text import discrete

#__all__ = [ 'Environment' ]

# TODO: 2020/04/11: Rename this class EnvDiscrete
class Environment(discrete.DiscreteEnv):
    """
    Class defining methods that are generic to ALL environments.
    """

    #--- Getters
    def getInitialStateDistribution(self):
        return np.copy(self.isd)

    def getState(self):
        return self.s

    def getNumActions(self):
        # This nA attribute is in the super class, but using
        # super().nA
        # super(EnvGridworld2D, self).nA
        # do NOT work... with the error "super() does not have attribute nA"... WHY???@?!?!@?#!@?#!?
        return self.nA

    def getNumStates(self):
        # Same comment as for getNumActions() and super()
        return self.nS

    #--- Setters
    def setInitialStateDistribution(self, isd):
        self.isd = isd
