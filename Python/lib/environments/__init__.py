# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:35:13 2020

@author: Daniel Mastropietro
@description: Definition of classes that are common to all discrete-state / discrete-action environments.

This class inherits from the discrete.DiscreteEnv class, where the state and action spaces are defined as discrete
spaces by:
    self.action_space = spaces.Discrete(self.nA)
    self.observation_space = spaces.Discrete(self.nS)
The state is stored in attribute:
    self.s

See more details in gym/envs/toy_text/discrete.py.
"""

import warnings

import numpy as np

from gym.envs.toy_text import discrete

#__all__ = [ 'EnvironmentDiscrete' ]


class EnvironmentDiscrete(discrete.DiscreteEnv):
    """
    Class defining methods that are generic to ALL environments with discrete state and actions

    Constructor parameters:
    - nS: number of possible states
    - nA: number of possible actions
    - P: 2D dictionary with entries all possible states of the environment and for each state all possible actions
      for that state. Each dictionary entry value is a list of tuples of the form:
      (prob_going_to_next_state, next_state, reward(next_state), is_terminal(next_state))
    - isd: initial state distribution
    - dim: environment dimension (e.g. 1D gridworld, 2D gridworld, etc.), which is just informative 
    - terminal_states: set containing the terminal states of the environment
    - terminal_rewards: dictionary indexed by the terminal states containing the rewards as values.
        The number of elements in this dictionary does NOT need to coincide with the number of terminal states.
        If a terminal state is not found among the keys of the dictionary, its associated reward is returned as 0
        by the corresponding method.
    """

    def __init__(self, nS, nA, P, isd, dim=1, terminal_states=set(), terminal_rewards=dict()):
        super().__init__(nS, nA, P, isd)

        # Dimension of the environment (e.g. 1D (gridworld), 2D (gridworld), etc.)
        self.dim = dim

        # All states
        self.all_states = list( range(self.getNumStates()) )
        self.terminal_states = terminal_states
        self.non_terminal_states = set(self.all_states).difference( set( self.terminal_states ) )
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

    def getTerminalRewards(self):
        "Returns the terminal rewards (only their values, not the states where they occur, for this use getTerminalRewardsDict())"
        return self.getTerminalRewardsDict().values()

    def getTerminalRewardsDict(self):
        "Returns the dictionary containing the terminal rewards, indexed by the terminal state"
        return self.terminal_rewards

    def getTerminalReward(self, s):
        "Returns the reward for the given terminal state"
        if s not in self.terminal_rewards.keys():
            warnings.warn(f"The given state s={s} is not present in the terminal rewards dictionary. Have you forgotten to include it? A zero reward is returned.")
        return self.terminal_rewards.get(s, 0.0)

    def getNonTerminalStates(self):
        "Returns the list of non terminal states"
        return list(self.non_terminal_states)

    def isTerminalState(self, state):
        return set([state]).issubset( set(self.terminal_states) )

    #--- Setters
    def setInitialStateDistribution(self, isd):
        self.isd = isd

    def setState(self, s):
        self.s = s


