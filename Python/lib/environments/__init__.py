# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:35:13 2020

@author: Daniel Mastropietro
@description: Definition of classes that are common to all environments.
"""

import warnings

import numpy as np

from gym.envs.toy_text import discrete

#__all__ = [ 'EnvironmentDiscrete' ]

class EnvironmentDiscrete(discrete.DiscreteEnv):
    """
    Class defining methods that are generic to ALL environments.

    Constructor parameters:
    - nS: number of possible states
    - nA: number of possible actions
    - P: 2D dictionary with entries all possible states of the environment and for each state all possible actions
      for that state. Each dictionary entry value is a list of tuples of the form:
      (prob_going_to_next_state, next_state, reward(next_state), is_terminal(next_state))
    - isd: initial state distribution
    - dim: environment dimension (e.g. 1D gridworld, 2D gridworld, etc.), which is just informative 
    - terminal_states: set containing the terminal states of the environment
    - terminal_rewards: list containing the values of the rewards at the terminal states
    """

    def __init__(self, nS, nA, P, isd, dim=1, terminal_states=set(), terminal_rewards=[]):
        super().__init__(nS, nA, P, isd)

        # Dimension of the environment (e.g. 1D (gridworld), 2D (gridworld), etc.)
        self.dim = dim

        # All states
        self.all_states = list( range(self.getNumStates()) )
        self.terminal_states = terminal_states
        self.non_terminal_states = set(self.all_states).difference( set( self.terminal_states ) )
        if len(terminal_rewards) != len(terminal_states):
            warnings.warn("The number of rewards given for the terminal states ({}) is" \
                          " different than the number of terminal states ({}). The rewards are left undefined." \
                          .format(len(terminal_rewards), len(terminal_states)))
            self.terminal_rewards = []
        else:
            self.terminal_rewards = terminal_rewards

    #--- Getters
    def getDimension(self):
        return self.dim

    def getInitialStateDistribution(self):
        return np.copy(self.isd)

    def getState(self):
        return self.s

    def getNumActions(self):
        # This nA attribute is in the super class, but using
        # super().nA
        # or
        # super(EnvGridworld2D, self).nA
        # do NOT work... with the error "super() does not have attribute nA"... WHY???@?!?!@?#!@?#!?
        return self.nA

    def getNumStates(self):
        # Same comment as for getNumActions() and super()
        return self.nS

    def getTerminalStates(self):
        "Returns the list of terminal states"
        return list(self.terminal_states)

    def getTerminalStatesAndRewards(self):
        return zip( self.terminal_states, self.terminal_rewards )

    def getNonTerminalStates(self):
        "Returns the list of non terminal states"
        return list(self.non_terminal_states)

    def isTerminalState(self, state):
        return set([state]).issubset( set(self.terminal_states) )

    #--- Setters
    def setInitialStateDistribution(self, isd):
        self.isd = isd
