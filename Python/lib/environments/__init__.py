# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:35:13 2020

@author: Daniel Mastropietro
@description: Definition of classes that are common to all environments.
"""

import numpy as np

from gym.envs.toy_text import discrete
from sympy.physics.mechanics.tests.test_system import states

#__all__ = [ 'EnvironmentDiscrete' ]

class EnvironmentDiscrete(discrete.DiscreteEnv):
    """
    Class defining methods that are generic to ALL environments.
    """
    def __init__(self, nS, nA, P, isd, terminal_states=set()):
        """
        Calls the constructor of the super class which requires:
        - nS: number of possible states
        - nA: number of possible actions
        - P: probability of each state-action pair
        - isd: initial state distribution
        - terminal_states: set containing the terminal states of the environment
        """
        super().__init__(nS, nA, P, isd)

        # All states
        self.all_states = np.arange(self.getNumStates())
        self.terminal_states = terminal_states

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

    def getTerminalStates(self):
        return self.terminal_states

    #--- Setters
    def setInitialStateDistribution(self, isd):
        self.isd = isd
