# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:52:18 2020

@author: Daniel Mastropietro
@description: Definition of gridworld environments (used the gridworld defined by Denny Britz in
https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py as a
starting point)
Every environment should define the following methods:
    - getNumActions()
    - getNumStates()
"""

import io
import numpy as np
import sys

# Imports a class defined in __init__.py
from . import EnvironmentDiscrete

#__all__ = [ "EnvGridworld2D",
#            "EnvGridworld1D" ]

class EnvGridworld1D(EnvironmentDiscrete):
    """
    1D Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an M-long grid and your goal is to reach the terminal
    state on the right.
    You receive a reward of 0 at each step until you reach a terminal state
    in which case you receive reward -1 on the left state, and +1 on the right cell.

    For example, a 10-long grid looks as follows:
    T  o  o  o  o  x  o  o  o  T
    x is your position and T are the two terminal states.
    """

    def __init__(self, length=21):
        if length < 3:
            raise ValueError('the length of the grid must be at least 3')

        nS = length
        nA = 2

        # Possible actions (i.e. we simply NAME the action values (indices) by defining constants)
        LEFT = 0
        RIGHT = 1

        # We substract 1 because the state are base 0 (i.e. they represent indices)
        MAX_S = nS - 1

        # Terminal states
        terminal_states = set([0, nS-1])
        rewards_dict = dict({0: -1.0, nS-1: +1.0})

        # True value functions (for algorithm evaluation purposes)
        # Note that terminal states have value = 0
        self.V = -1.0 + 2/(nS-1) * np.arange(nS)    # Note that this is the same as np.arange(-(nS-1), nS+1, 2) / (nS-1)
                                                    # (note that stop=nS+1, because the end value is NOT included!)
        self.V[0] = self.V[nS-1] = 0.0
        self.Q = None 

        # Define the possible actions based on the geometry
        P = {}
        grid = np.arange(nS)
        it = np.nditer(grid)

        # Function that checks whether we arrived at a terminal state
        num_nonterminal_states = nS - len(terminal_states)
        is_terminal = lambda s: s in terminal_states
        reward = lambda s: rewards_dict[s] if s in rewards_dict.keys() else 0.0

        while not it.finished:
            s = it.iterindex

            # P[s][a] = (prob, next_state, reward, is_terminal)
            # Given that we are state 's' and take action 'a':
            # prob: probability of going to state 'next_state'
            # next_state: a possible next_state when taking action 'a'
            # reward: reward received when going to state 'next_state'
            # is_terminal: whether 'next_state' is a terminal state
            P[s] = {a : [] for a in range(nA)}

            if is_terminal(s):
                # Terminal states are assigned value 0 (i.e. they receive 0 reward by convention)
                P[s][LEFT] = [(1.0, s, 0.0, True)]
                P[s][RIGHT] = [(1.0, s, 0.0, True)]
            else:
                # Not a terminal state
                # ns = Next State!
                ns_left = np.max([0, s-1])
                ns_right = np.min([s+1, MAX_S])
                P[s][LEFT] = [(1.0, ns_left, reward(ns_left), is_terminal(ns_left))]
                P[s][RIGHT] = [(1.0, ns_right, reward(ns_right), is_terminal(ns_right))]

            it.iternext()

        # Initial state distribution is uniform, excluding terminal state which have zero probability
        isd = np.ones(nS) / num_nonterminal_states
        isd[0] = 0.0
        isd[nS-1] = 0.0
        assert np.isclose(np.sum(isd), 1.0), \
                "The initial state probabilities sum up to 1 ({})".format(np.sum(isd))

        self.P = P
        #print("P")
        #print(P)

        super(EnvGridworld1D, self).__init__(nS, nA, P, isd,
                                             dim=1,
                                             terminal_states=set([i for i, s in enumerate(range(nS)) if is_terminal(s)]),
                                             terminal_rewards=[reward(s) for s in terminal_states])

    def getV(self):
        "Returns the true state value function"
        return self.V
    
    def getQ(self):
        "Returns the true state-action value function"
        return self.Q


class EnvGridworld2D(EnvironmentDiscrete):
    """
    2D Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    The agent's goal on an MxN grid is to reach the terminal
    state at the top left or the bottom right corner.
    
    The grid cells are indexed as a 0-based matrix, i.e. by indices (i,j) == (y,x)
    where y indicates the rows and x indicates the columns.
  
    For example, a 5x4 grid looks like:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  o
    o  o  o  T
    where 'x' is the position of the agent and T are the two terminal states.
    The top-left corner has position (y,x) = (0,0).
    The bottom-right corner has position (y,x) = (4,3).

    States are 1D indexed by row starting at index = 0 at the top-left corner
    up to nS-1 at the bottom-right corner, where nS = #rows * #cols.

    The agent can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    The agent receives a reward of 0 at each step until it reaches a terminal state
    in which case it receives reward -1 at the top-left cell, and +1 at the bottom-right cell.

    This can be specialized to a 1D Grid world by setting one of the dimensions of `shape` to 1.
    For example, a 10-long grid looks as follows:
    T  o  o  o  o  x  o  o  o  T
    x is your position and T are the two terminal states,
    where the left T receives reward -1 and the right T receives reward +1.

    The constructor receives the following parameters:
    
    shape: list/tuple
        Size of the 2D gridworld given as #rows x #cols.
        default: [5, 5]

    terminal_states: set
        Set with the 1D state numbers considered as terminal states.
        default: set([0, 5*5-1])

    rewards_dict: dict
        Reward values for a subset of states in the grid. The 1D state number is used as dictionary key.
        default: {0: -1.0, 5*5-1: +1.0}
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[5,5], terminal_states=None, rewards_dict=None):
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("Shape argument must be a list/tuple of length 2")
        if terminal_states is not None and len(terminal_states) == 0:
            raise ValueError("There must be at least one terminal state")
        if rewards_dict is not None and not isinstance(rewards_dict, dict):
            raise ValueError("The rewards information must be a dictionary indexed by the 1D state number")

        # Environment geometry
        self.shape = shape
        nS = np.prod(shape)         # Number of states
        nA = 4                      # Number of actions (one in each cardinal point)

        # Terminal states
        if terminal_states is None:
            terminal_states = set([0, nS-1])
            rewards_dict = dict({0: -1.0, nS-1: 1.0})
        elif not isinstance(terminal_states, set):
            terminal_states = set(terminal_states)

        # Possible actions (i.e. we simply NAME the action values (indices) by defining constants)
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

        # We substract 1 because the x and y values are base 0 (as they represent indices)
        MAX_Y = shape[0] - 1
        MAX_X = shape[1] - 1

        # Define the possible actions based on the geometry
        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        # Function that checks whether we arrived at a terminal state
        num_nonterminal_states = nS - len(terminal_states)
        is_terminal = lambda s: s in terminal_states
        reward = lambda s: rewards_dict[s] if s in rewards_dict.keys() else 0.0

        # Initialize the initial state distribution matrix to 0
        # This matrix will be updated as we go along constructing the environment below
        # and will be set to a uniform distribution on the NON-terminal states.
        isd = np.zeros(nS)
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index   # 0 <= y <= shape[0]-1, 0 <= x <= shape[1]-1

            # Initial state distribution
            if not is_terminal(s):
                isd[s] = 1 / num_nonterminal_states

            # Transition matrix and rewards: P[s][a] = (prob, next_state, reward, is_terminal)
            # Given that we are state 's' and take action 'a':
            # prob: probability of going to state 'next_state'
            # next_state: a possible next_state when taking action 'a'
            # reward: reward received when going to state 'next_state'
            # is_terminal: whether 'next_state' is a terminal state
            P[s] = {a : [] for a in range(nA)}

            # We're stuck in a terminal state
            if is_terminal(s):
                P[s][UP] = [(1.0, s, reward(s), True)]
                P[s][RIGHT] = [(1.0, s, reward(s), True)]
                P[s][DOWN] = [(1.0, s, reward(s), True)]
                P[s][LEFT] = [(1.0, s, reward(s), True)]
            # Not a terminal state
            else:
                # ns = Next State!
                ns_up = s if y == 0 else s - (MAX_X + 1)
                ns_right = s if x == MAX_X else s + 1
                ns_down = s if y == MAX_Y else s + (MAX_X + 1)
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward(ns_up), is_terminal(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward(ns_right), is_terminal(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward(ns_down), is_terminal(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward(ns_left), is_terminal(ns_left))]

            it.iternext()

        assert np.isclose(np.sum(isd), 1.0), \
                "The initial state probabilities sum up to 1 ({})".format(np.sum(isd))

        self.P = P
        #print("P")
        #print(P)

        super(EnvGridworld2D, self).__init__(nS, nA, P, isd,
                                             dim=2,
                                             terminal_states=terminal_states,
                                             terminal_rewards=[reward(s) for s in terminal_states])

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = [idx + 1 for idx in it.multi_index]

            if self.s == s:
                output = " ({:2d}) x ".format(s)
            elif self.isTerminalState(s):
                output = " ({:2d}) T ".format(s)
            else:
                output = " ({:2d}) o ".format(s)

            if x == 1:
                output = output.lstrip()
            if x == self.shape[1]:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1]:
                outfile.write("\n")

            it.iternext()
    
    #--- Getters
    def getShape(self):
        return self.shape

    def getV(self):
        "Returns the true state value function"
        return None
    
    def getQ(self):
        "Returns the true state-action value function"
        return None
