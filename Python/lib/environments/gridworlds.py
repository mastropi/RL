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

Note that the action space is defined ALWAYS as an integer action space because this is how gym defines the
`action_space` attribute of their DiscreteEnv environment in envs/toy_text/discrete.py, i.e. by instantiating action_space
as a Discrete object of type int (the default type). If we wanted to instantiate the `action_space` of the gridworld environments
defined here as an action space of type e.g. Direciont2D (defined below), we would need to re-write the constructor
of DiscreteEnv (i.e. override it via a subclass) and change the instantiation of the `action_space` attribute...
and I find this too much of a hassle for gaining almost nothing (and probably for losing performance, as indexing
a dictionary with an Enum is probably slower than indexing it with an integer).
"""

import io
import numpy as np
import sys

from enum import Enum, unique

# Imports a class defined in __init__.py
from . import EnvironmentDiscrete

from Python.lib.utils.basic import index_linear2multi, index_multi2linear

#__all__ = [ "EnvGridworld2D",
#            "EnvGridworld1D" ]


@unique
class Direction1D(Enum):
    # The values should be defined so that summing len(enum)/2 to any of the actions MOD len(enum) gives the action of the opposite direction
    LEFT = 0
    RIGHT = 1

@unique
class Direction2D(Enum):
    # The values should be defined so that summing len(enum)/2 to any of the actions MOD len(enum) gives the action of the opposite direction
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

def get_adjacent_states(shape: tuple, s: int):
    """
    Returns, in a 2D gridworld of the given shape, all the adjacent states to a state of interest s and the direction from s to which each is located

    Arguments:
    shape: tuple
        Shape of the 2D gridworld on which the adjacent states to `s` should be retrieved.

    s: int
        1D index of the state of interest in the 2D gridworld.

    Return: set
    Set of the tuples (t, d) containing:
    - t: an adjacent state to s in direction `d`. If there is no adjacent state because `s` is at the border, t = None.
    - d: the direction from s to which state `t` is located, even if t is None.
    The considered directions are all those indicated in the `Direction` enum.
    """
    adjacent_states = []    # List of tuples (s_adj, direction) where s_adj is the 1D index of the state adjacent to s and direction is the direction w.r.t. s where the adjacent state was found
    for direction in Direction2D.__members__.values():
        adjacent_states += [(state_adjacent(shape, s, direction), direction)]

    return set(adjacent_states)

def state_adjacent(shape: tuple, state: int, direction: Direction2D):
    "Computes the 1D state that is adjacent to the given 1D `state` towards `direction`"
    y, x = index_linear2multi(state, shape)

    ymax = shape[0] - 1
    xmax = shape[1] - 1

    if direction == Direction2D.UP:
        s_adjacent = None if y == 0 else (y-1, x)
    elif direction == Direction2D.RIGHT:
        s_adjacent = None if x == xmax else (y, x+1)
    elif direction == Direction2D.DOWN:
        s_adjacent = None if y == ymax else (y+1, x)
    elif direction == Direction2D.LEFT:
        s_adjacent = None if x == 0 else (y, x-1)
    else:
        raise ValueError(f"Invalid direction {direction}")

    if s_adjacent is None:
        return None
    else:
        return index_multi2linear(s_adjacent, shape)

def get_opposite_direction(d: Direction2D):
    """
    The direction opposite to a direction of interest d

    Arguments:
    d: Direction
        Direction of interest.

    Return: Direction2D
    The direction that is opposite to `d` which is computed as (d + #directions/2) mod #directions.
    Ex: when there are 4 directions: UP (0), DOWN (2), RIGHT (1), LEFT (3), it is computed as (d + 2) mod 4 so that
    the opposite of UP is DOWN and the opposite of RIGHT is LEFT, and viceversa.
    """
    opposite_d = (d.value + len(Direction2D) // 2) % len(Direction2D)
    return Direction2D(opposite_d)


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

    Arguments:
    length: (opt) int
        Number of states in the gridworld.
        default: 21

    initial_state_distribution: (opt) 1D list or array
        See definition in super class.
        default: None
    """

    def __init__(self, length=21, initial_state_distribution=None):
        if length < 3:
            raise ValueError('the length of the grid must be at least 3')

        nS = length
        nA = 2

        # We substract 1 because the state are base 0 (i.e. they represent indices)
        MAX_S = nS - 1

        # Terminal states and reward obtained when reaching each possible state
        terminal_states = set([0, nS-1])
        rewards_dict = dict({0: -1.0, nS-1: +1.0})

        #-- True value functions (for algorithm evaluation purposes)
        #-- Note that terminal states have value = 0
        # Value functions for the RANDOM walk policy ONLY!
        self.V = -1.0 + 2/(nS-1) * np.arange(nS)    # Note that this is the same as np.arange(-(nS-1), nS+1, 2) / (nS-1)
                                                    # (note that stop=nS+1, because the end value is NOT included!)
        self.V[0] = self.V[nS-1] = 0.0              # The value of terminal states is defined as 0.0.
                                                    # This is VERY IMPORTANT, meaning that the estimated value of
                                                    # terminal states should be ALWAYS 0, o.w. learning diverges...
                                                    # (in fact, if this were not the case, when transitioning to a
                                                    # terminal state, the TD error would receive twice the reward, as
                                                    # it is computed (when gamma = 1) as:
                                                    # delta(T) = R(T) + V(S(T)) - V(S(T-1)) = 2 R(T) - V(S(T-1)),
                                                    # where the 2 multiplying R(T) doesn't look right.
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

            # P[s] is a dictionary indexed by the action "a" and its value a list of tuples for each possible next state that can be reached from state 's' taking action 'a':
            # P[s][a] = [(prob_1, next_state_1, reward_1, is_terminal_1), (prob_2, next_state_2, reward_2, is_terminal_2), ...]
            # Given that we are state 's' and take action 'a':
            # prob: probability of going to state 'next_state'
            # next_state: a possible next_state when taking action 'a'
            # reward: reward received when going to state 'next_state'
            # is_terminal: whether 'next_state' is a terminal state
            P[s] = {a : [] for a in range(nA)}

            if is_terminal(s):
                # Terminal states are assigned value 0 (i.e. they receive 0 reward by convention)
                # Also, with probability 1 the next state is the same state `s`.
                P[s][Direction1D.LEFT.value] = [(1.0, s, 0.0, True)]
                P[s][Direction1D.RIGHT.value] = [(1.0, s, 0.0, True)]
            else:
                # Not a terminal state
                # ns = Next State!
                ns_left = np.max([0, s-1])
                ns_right = np.min([s+1, MAX_S])
                P[s][Direction1D.LEFT.value] = [(1.0, ns_left, reward(ns_left), is_terminal(ns_left))]
                P[s][Direction1D.RIGHT.value] = [(1.0, ns_right, reward(ns_right), is_terminal(ns_right))]

            it.iternext()

        # Initial state distribution is uniform, excluding terminal state which have zero probability
        if initial_state_distribution is None:
            isd = np.ones(nS) / num_nonterminal_states
            isd[0] = 0.0
            isd[nS-1] = 0.0
        else:
            if initial_state_distribution is not None:
                if not isinstance(initial_state_distribution, (list, tuple, np.ndarray)) or len(initial_state_distribution) != length:
                    raise ValueError("The initial state distribution must be a list, tuple or numpy array with as many elements as the number of states in the environment")
                if not all(np.array(initial_state_distribution) >= 0) or not np.isclose(np.sum(initial_state_distribution), 1.0):
                    raise ValueError(f"The values in the initial state distribution must sum up to 1 (sum = {np.sum(initial_state_distribution)})")
                if not all([p == 0 for s, p in enumerate(initial_state_distribution) if s in terminal_states]):
                    raise ValueError(f"The initial state distribution must be 0 at the terminal states ({terminal_states}):\ngiven distribution = {initial_state_distribution}")
            isd = initial_state_distribution
        assert np.isclose(np.sum(isd), 1.0), "The initial state probabilities sum up to 1 ({})".format(np.sum(isd))

        self.P = P
        #print("P")
        #print(P)

        super(EnvGridworld1D, self).__init__(nS, nA, P, isd,
                                             dim=1,
                                             terminal_states=set([i for i, s in enumerate(range(nS)) if is_terminal(s)]),
                                             terminal_rewards=dict([(s, reward(s)) for s in terminal_states]))


class EnvGridworld1D_OneTerminalState(EnvironmentDiscrete):
    """
    1D Grid World environment with one terminal state (right one) and one transient state at state 0
    (the leftmost state) for which the probability of going right is 1.
    The goal is to analyze the conjecture that lambda should start near 1 and decrease towards 0 as more reliable
    estimates of the value function are obtained, in particular to see if this happens in our adaptive TD(lambda)
    algorithm.

    This example is mentioned in Bertsekas, pag. 200-201 and, similar to this also in Sutton,
    chapter 7 when analyzing the advantages of TD(0) (more model-based) over Monte Carlo (more experience-based)
    through a two non-terminal states A and B and two terminal states reachable in one step from B.

    The default reward landscape is +1 for all states except at the terminal state.

    The initial state of the environment is the transient state.

    For example, a 10-long grid looks as follows:
    t  o  o  o  o  x  o  o  o  T
    where t = "transient, T = "Terminal", and "x" is the current state of the environment.

    Arguments:
    length: (opt) int
        Number of states in the gridworld.
        default: 21

    terminal_states: (opt) set or list or array-like
        State indices considered as terminal states.
        default: set([`length` - 1])

    rewards_dict: dict
        Reward values for a subset of states in the grid. The state index is used as dictionary key.
        default: {`length`: 0.0}

    reward_default: float
        Default value for the state reward.
        This value is used as reward delivered by a state when the state does not appear in `rewards_dict`.
        default: +1.0

    initial_state_distribution: (opt) list or array
        Probability distribution defining how to choose the initial state when the environment is initialized or reset.
        It should have the same length as the number of states in the environment and terminal states
        must have probability 0, while the sum of the probability on all other states should equal 1.
        This value is passed to the super class constructor as parameter `isd`.
        default: None
    """

    def __init__(self, length=21, terminal_states=None, rewards_dict=None, reward_default=1.0, initial_state_distribution=None):
        if length < 3:
            raise ValueError('the length of the grid must be at least 3')
        if terminal_states is not None and len(terminal_states) == 0:
            raise ValueError("There must be at least one terminal state")
        if rewards_dict is not None and not isinstance(rewards_dict, dict):
            raise ValueError("The rewards information must be a dictionary indexed by the 1D state number")
        if initial_state_distribution is not None:
            if not isinstance(initial_state_distribution, (list, tuple, np.ndarray)) or len(initial_state_distribution) != length:
                raise ValueError(f"The initial state distribution must be a list, tuple or numpy array with as many elements as the number of states in the environment ({length}): {initial_state_distribution}")
            if not all(np.array(initial_state_distribution) >= 0) or not np.isclose(np.sum(initial_state_distribution), 1.0):
                raise ValueError(f"The values in the initial state distribution must sum up to 1 (sum = {np.sum(initial_state_distribution)})")

        nS = length
        nA = 2

        # We substract 1 because the state are base 0 (i.e. they represent indices)
        MAX_S = nS - 1

        # Terminal states and reward obtained when reaching each possible state
        terminal_states = set([nS - 1]) if terminal_states is None else set(terminal_states)
        rewards_dict = dict({nS - 1: 0.0}) if rewards_dict is None else rewards_dict

        if initial_state_distribution is not None and not all([p == 0 for s, p in enumerate(initial_state_distribution) if s in terminal_states]):
            raise ValueError(f"The initial state distribution must be 0 at the terminal states ({terminal_states}):\ngiven distribution = {initial_state_distribution}")

        # Define the possible actions based on the geometry
        P = {}
        grid = np.arange(nS)
        it = np.nditer(grid)

        # Function that checks whether we arrived at a terminal state
        is_terminal = lambda s: s in terminal_states
        reward = lambda s: rewards_dict[s] if s in rewards_dict.keys() else reward_default

        while not it.finished:
            s = it.iterindex

            # P[s] is a dictionary indexed by the action "a" and its value a list of tuples for each possible next state that can be reached from state 's' taking action 'a':
            # P[s][a] = [(prob_1, next_state_1, reward_1, is_terminal_1), (prob_2, next_state_2, reward_2, is_terminal_2), ...]
            # Given that we are state 's' and take action 'a':
            # prob: probability of going to state 'next_state'
            # next_state: a possible next_state when taking action 'a'
            # reward: reward received when going to state 'next_state'
            # is_terminal: whether 'next_state' is a terminal state
            P[s] = {a: [] for a in range(nA)}

            if is_terminal(s):
                # Terminal states are assigned value 0 (i.e. they receive 0 reward by convention)
                # Also, with probability 1 the next state is the same state `s`.
                P[s][Direction1D.LEFT.value] = [(1.0, s, 0.0, True)]
                P[s][Direction1D.RIGHT.value] = [(1.0, s, 0.0, True)]
            else:
                if s > 0:
                    # Not a terminal state, nor the transient state
                    # ns = Next State!
                    ns_left = np.max([0, s - 1])
                    ns_right = np.min([s + 1, MAX_S])
                    P[s][Direction1D.LEFT.value] = [(1.0, ns_left, reward(ns_left), is_terminal(ns_left))]
                    P[s][Direction1D.RIGHT.value] = [(1.0, ns_right, reward(ns_right), is_terminal(ns_right))]
                else:
                    # Transient state: the state transitions to the right with probability 1
                    # i.e. the LEFT action is not possible (transition probability = 0.0)
                    P[s][Direction1D.LEFT.value] = [(0.0, 0, 0.0, False)]
                    P[s][Direction1D.RIGHT.value] = [(1.0, 1, 0.0, False)]

            it.iternext()

        # Initial state distribution: the environment always starts at the transient state
        if initial_state_distribution is None:
            # Define the initial state distribution as starting always at state 0.
            isd = np.zeros(nS)
            isd[0] = 1.0
        else:
            isd = initial_state_distribution
        assert np.isclose(np.sum(isd), 1.0), \
            "The initial state probabilities sum up to 1 ({})".format(np.sum(isd))

        self.P = P
        # print("P")
        # print(P)

        super(EnvGridworld1D_OneTerminalState, self).__init__(nS, nA, P, isd,
                                                              dim=1,
                                                              terminal_states=set([i for i, s in enumerate(range(nS)) if is_terminal(s)]),
                                                              terminal_rewards=dict([(s, reward(s)) for s in terminal_states]))


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

    The agent can take actions in each direction (UP, RIGHT, DOWN, LEFT) as defined in enum `Direction`.
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

    initial_state_distribution: (opt) 1D list or array
        Probability distribution defining how to choose the initial state when the environment is
        initialized or reset.
        It should have the same length as the number of states in the environment and terminal states
        must have probability 0, while the sum of the probability on all other states should equal 1.
        This value is passed to the super class constructor as parameter `isd`.
        default: None
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[5,5], terminal_states=None, rewards_dict=None, initial_state_distribution=None):
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("Shape argument must be a list/tuple of length 2")
        if terminal_states is not None and len(terminal_states) == 0:
            raise ValueError("There must be at least one terminal state")
        if rewards_dict is not None and not isinstance(rewards_dict, dict):
            raise ValueError("The rewards information must be a dictionary indexed by the 1D state number")
        if initial_state_distribution is not None:
            if not isinstance(initial_state_distribution, (list, tuple, np.ndarray)) or len(initial_state_distribution) != np.prod(shape):
                raise ValueError(f"The initial state distribution must be a list, tuple or numpy array with as many elements as the number of states in the environment ({np.prod(shape)}): {initial_state_distribution}")
            if not all(np.array(initial_state_distribution) >= 0) or not np.isclose(np.sum(initial_state_distribution), 1.0):
                raise ValueError(f"The values in the initial state distribution must sum up to 1 (sum = {np.sum(initial_state_distribution)})")

        # Environment geometry
        self.shape = shape
        nS = np.prod(shape)         # Number of states
        nA = 4                      # Number of actions (one in each cardinal point)

        # Terminal states and reward obtained when reaching any possible state
        if terminal_states is None:
            terminal_states = set([0, nS-1])
            rewards_dict = dict({0: -1.0, nS-1: 1.0})
        elif not isinstance(terminal_states, set):
            terminal_states = set(terminal_states)

        if initial_state_distribution is not None and not all([p == 0 for s, p in enumerate(initial_state_distribution) if s in terminal_states]):
            raise ValueError(f"The initial state distribution must be 0 at the terminal states ({terminal_states}):\ngiven distribution = {initial_state_distribution}")

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

        if initial_state_distribution is None:
            # Initialize the initial state distribution matrix to 0
            # This matrix will be updated as we go along constructing the environment below
            # and will be set to a uniform distribution on the NON-terminal states.
            isd = np.zeros(nS)
        else:
            isd = initial_state_distribution
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index   # 0 <= y <= shape[0]-1, 0 <= x <= shape[1]-1

            # Initial state distribution
            if initial_state_distribution is None:
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
                P[s][Direction2D.UP.value] = [(1.0, s, reward(s), True)]
                P[s][Direction2D.RIGHT.value] = [(1.0, s, reward(s), True)]
                P[s][Direction2D.DOWN.value] = [(1.0, s, reward(s), True)]
                P[s][Direction2D.LEFT.value] = [(1.0, s, reward(s), True)]
            # Not a terminal state
            else:
                # ns = Next State!
                ns_up = s if y == 0 else s - (MAX_X + 1)
                ns_right = s if x == MAX_X else s + 1
                ns_down = s if y == MAX_Y else s + (MAX_X + 1)
                ns_left = s if x == 0 else s - 1
                P[s][Direction2D.UP.value] = [(1.0, ns_up, reward(ns_up), is_terminal(ns_up))]
                P[s][Direction2D.RIGHT.value] = [(1.0, ns_right, reward(ns_right), is_terminal(ns_right))]
                P[s][Direction2D.DOWN.value] = [(1.0, ns_down, reward(ns_down), is_terminal(ns_down))]
                P[s][Direction2D.LEFT.value] = [(1.0, ns_left, reward(ns_left), is_terminal(ns_left))]

            it.iternext()

        assert np.isclose(np.sum(isd), 1.0), "The initial state probabilities must sum up to 1 ({})".format(np.sum(isd))

        self.P = P
        #print("P")
        #print(P)

        super(EnvGridworld2D, self).__init__(nS, nA, P, isd,
                                             dim=2,
                                             terminal_states=terminal_states,
                                             terminal_rewards=dict([(s, reward(s)) for s in terminal_states]))

    def _render(self, mode='human', close=False):
        """
        Renders the current gridworld layout
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


class EnvGridworld2D_WithObstacles(EnvGridworld2D):
    """
    2D Grid World environment with obstacles that inherits from class EnvGridworld2D.

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

    obstacles_set: set
        Set with the 1D state numbers (cells) where the agent cannot go.
        Normally a "good" number of obstacles for a grid of size N is int(log(N)).
        default: None

    initial_state_distribution: (opt) 1D list or array
        See definition in super class.
        default: None
    """

    def __init__(self, shape=[5, 5], terminal_states=None, rewards_dict=None, obstacles_set=None, initial_state_distribution=None):
        super().__init__(shape=shape, terminal_states=terminal_states, rewards_dict=rewards_dict, initial_state_distribution=initial_state_distribution)

        # Update the `isd` attribute defined by the super constructor (containing the initial state distribution)
        # (which does NOT know of obstacles in the gridworld) to ensure that the agent does NOT start at an obstacle state.
        if initial_state_distribution is None and obstacles_set is not None and len(obstacles_set) > 0:
            set_terminal_and_obstacles_states = set.union(terminal_states, obstacles_set)
            set_nonterminal_and_nonobstacles_states = set(range(self.nS)).difference(set_terminal_and_obstacles_states)
            num_terminal_states = len(terminal_states)
            num_nonterminal_and_nonobstacles_states = len(set_nonterminal_and_nonobstacles_states)
            for s in set_terminal_and_obstacles_states:
                self.isd[s] = 0.0
            # Uplift the probability of non-obstacle and non-terminal states after we have set to 0 the probability of obstacle and terminal states
            assert num_nonterminal_and_nonobstacles_states < self.nS - num_terminal_states
            for s in set_nonterminal_and_nonobstacles_states:
                self.isd[s] *= (self.nS - num_terminal_states) / num_nonterminal_and_nonobstacles_states
        elif initial_state_distribution is not None:
            for s in obstacles_set:
                if self.isd[s] != 0.0:
                    raise ValueError(f"The initial state distribution must be 0 for obstacles states ({obstacles_set}):\ngiven distribution = {self.isd}")
        assert np.isclose(np.sum(self.isd), 1.0), "The initial state probabilities must sum up to 1 ({})".format(np.sum(self.isd))

        # Treat the obstacles as such by setting the transition probability of the adjacent cells when moving towards the obstacle to 0
        self.obstacles_set = obstacles_set
        for s_obstacle in self.obstacles_set:
            # Get the 4 states that are adjacent to the obstacle set, in each of the geographical directions (N, S, E, W)
            # so that we can set their transition probabilities to 0 and the reward of the obstacle cell
            # when taking the action that intends to go to the obstacle cell.
            set_adjacent_states = get_adjacent_states(shape, s_obstacle)
            for ss, d in set_adjacent_states:
                if ss is not None:
                    opposite_d = get_opposite_direction(d)
                    # We state that the probability of staying in state ss when the agent wants to move in the opposite direction
                    # to where ss is w.r.t. s_obstacle (d) is 1, i.e. the agent cannot move from ss in the opposite direction to `d`.
                    # Ex: d = UP => the agent cannot come from the state ss that is above s_obstacle when going DOWN (`minusd`)
                    self.P[ss][opposite_d.value] = [(1, ss, rewards_dict.get(ss, 0.0), self.isTerminalState(ss))]

    def isObstacleState(self, s):
        return s in self.obstacles_set

    def _render(self, mode='human', close=False):
        """
        Renders the current gridworld with obstacles layout
        For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  P  P  o
            o  o  o  T
        where x is the agent's position, P is the position of the obstacles (Prohibited) and T represents the terminal states.
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
            elif self.isObstacleState(s):
                # Check that the transition probabilities of every adjacent state going to the obstacle state is 0
                set_adjacent_states = get_adjacent_states(self.getShape(), s)
                for ss, d in set_adjacent_states:
                    if ss is not None:
                        opposite_d = get_opposite_direction(d)
                        assert self.P[ss][opposite_d.value][0][0] == 1 and self.P[ss][opposite_d.value][0][1] == ss, \
                            "The transition probability going {} to obstacle state s={} from the adjacent state {} ss={} is 0: P[{}][{}] = {}, P(ss->s) = {}" \
                            .format(opposite_d.name, s, d.name, ss, ss, opposite_d.name, self.P[ss][opposite_d.value][0], 1 - self.P[ss][opposite_d.value][0][0])
                output = " ({:2d}) P ".format(s)
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

        return outfile

    def getAllValidStates(self):
        "Returns the list of valid states (i.e. states that are not obstacles)"
        return [s for s in self.getAllStates() if not self.isObstacleState(s)]
