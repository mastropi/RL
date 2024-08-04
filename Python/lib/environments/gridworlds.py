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
import warnings

from enum import Enum, unique
import matplotlib.pyplot as plt
import random

# Imports a class defined in __init__.py
from . import EnvironmentDiscrete

from Python.lib.utils.basic import index_linear2multi, index_multi2linear


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
    1D Gridworld environment with any number of terminal states and any rewards landscape

    The default is a 21-state gridworld with deterministic initial state distribution on the left state (s=0)
    and one terminal state at the rightmost state (s=20) with an all-zero reward landscape except at the right state where it is +1.

    Arguments:
    length: (opt) int
        Number of states in the gridworld.
        default: 21

    terminal_states: (opt) set or list or array-like
        Indices of the terminal states.
        default: set([`length` - 1])

    rewards_dict: dict
        Reward values for a subset of states in the grid. The state index is used as dictionary key.
        default: {`length`: 1.0}

    reward_default: float
        Default value for the reward.
        This value is used as the reward obtained when visiting a state that does not appear among the keys of `rewards_dict`.
        default: 0.0

    initial_state_distribution: (opt) list or array
        Probability distribution defining how to choose the initial state when the environment is initialized or reset.
        It should have the same length as the number of states in the environment. Terminal states must have probability 0,
        and the sum of the probability on all other states must equal 1.
        This value is passed to the super class constructor as parameter `isd`.
        default: None, in which case state s=0 has mass 1 and the rest mass 0
    """

    # NOTE: This class was originally created as EnvGridworld1D_OneTerminalState with the goal of analyzing the conjecture by Urtzi
    # that an adaptive lambda should start near 1 and decrease towards 0 as more reliable estimates of the value function are obtained,
    # in particular to see if this happens in our adaptive TD(lambda) algorithm.
    #
    # This example is mentioned in Bertsekas, pag. 200-201 and, similar to this also in Sutton,
    # chapter 6 when analyzing the advantages of TD(0) (more model-based) over Monte Carlo (more experience-based)
    # through a two non-terminal states A and B and two terminal states reachable in one step from B.
    #
    # The initial state of the environment is the leftmost state which, under the example mentioned in Bertsekas and in Sutton,
    # is a transient state, meaning that the Markov chain goes right with probability 1.
    #
    # Note that this environment can still be created using the new version of the class.

    def __init__(self, length=21, terminal_states=None, rewards_dict=None, reward_default=0.0, initial_state_distribution=None):
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
        terminal_states = set([nS-1]) if terminal_states is None else set(terminal_states)
        rewards_dict = dict({nS-1: 1.0}) if rewards_dict is None else rewards_dict

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
                # Terminal states are assigned value 0 (i.e. they receive 0 reward when moving to another state by convention)
                # Also, with probability 1 the next state is the same state `s`.
                P[s][Direction1D.LEFT.value] = [(1.0, s, 0.0, True)]
                P[s][Direction1D.RIGHT.value] = [(1.0, s, 0.0, True)]
            else:
                # Not a terminal state
                # ns = Next State!
                ns_left = np.max([0, s - 1])
                ns_right = np.min([s + 1, MAX_S])
                P[s][Direction1D.LEFT.value] = [(1.0, ns_left, reward(ns_left), is_terminal(ns_left))]
                P[s][Direction1D.RIGHT.value] = [(1.0, ns_right, reward(ns_right), is_terminal(ns_right))]

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
        #print("P")
        #print(P)

        super(EnvGridworld1D, self).__init__(nS, nA, P, isd,
                                             dim=1,
                                             terminal_states=set([i for i, s in enumerate(range(nS)) if is_terminal(s)]),
                                             rewards=rewards_dict)

    def getShape(self):
        return (self.nS,)


class EnvGridworld1D_Classic(EnvGridworld1D):
    """
    1D Gridworld environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an M-long grid and your goal is to reach the terminal state on the right.
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
        super().__init__(length, terminal_states=set({0, length-1}), rewards_dict=dict({0: -1.0, length-1: +1.0}), reward_default=0.0, initial_state_distribution=initial_state_distribution)
        nS = length

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


class EnvGridworld2D(EnvironmentDiscrete):
    """
    2D Gridworld environment from Sutton's Reinforcement Learning book chapter 4.
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

    This can be specialized to a 1D gridworld by setting one of the dimensions of `shape` to 1.
    For example, a 10-long grid looks as follows:
    T  o  o  o  o  x  o  o  o  T
    x is your position and T are the two terminal states,
    where the left T receives reward -1 and the right T receives reward +1.

    The constructor receives the following parameters:
    
    shape: (opt) list/tuple
        Size of the 2D gridworld given as #rows x #cols.
        default: [5, 5]

    terminal_states: (opt) set, list, tuple
        Set with the 1D state numbers considered as terminal states.
        default: set([0, shape[0]*shape[1]-1])

    obstacle_states: (opt) set, list, tuple
        Set with the 1D state numbers (cells) where the agent cannot go.
        Normally a "good" number of obstacles for a grid of size N is int(log(N)).
        default: None

    rewards_dict: (opt) dict
        Reward values for a subset of states in the grid. The 1D state number is used as dictionary key.
        default: {0: -1.0, shape[0]*shape[1]-1: +1.0}

    wind_dict: (opt) dict
        Wind direction, if any. It must contain the following two keys:
        - 'direction': Direction2D object (e.g. Direction2D.UP, Direction2D.RIGHT, etc.)
        - 'intensity': value in [0, 1] indicating the probability that the agent is deviated towards the direction of the wind when it choose to move (to any direction).
        Ex: wind_dict = {'direction': Direction2D.LEFT, 'intensity': 0.1}
        default: None

    initial_state_distribution: (opt) 1D list or array
        Probability distribution defining how to choose the initial state when the environment is
        initialized or reset.
        It should have the same length as the number of states in the environment and terminal states
        must have probability 0, while the sum of the probability on all other states should equal 1.
        This value is passed to the super class constructor as parameter `isd`.
        default: None
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[5, 5], terminal_states=None, obstacle_states=None, rewards_dict=None, wind_dict=None, initial_state_distribution=None):
        #----- Parse input parameters 1
        # Environment shape
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("Shape argument must be a list/tuple of length 2")

        # Environment geometry
        self.shape = shape
        nS = np.prod(shape)         # Number of states
        nA = 4                      # Number of actions (one in each cardinal point)

        # Maximum values for the X and Y dimensions: note that we substract 1 because the x and y values are base 0 (as they represent indices)
        MAX_Y = self.shape[0] - 1
        MAX_X = self.shape[1] - 1

        #----- Parse input parameters 2
        # Terminal states
        if terminal_states is not None and len(terminal_states) == 0:
            raise ValueError("There must be at least one terminal state")
        # Rewards
        if rewards_dict is not None:
            if not isinstance(rewards_dict, dict):
                raise ValueError("The rewards information must be a dictionary indexed by the 1D state number")
            if not set(rewards_dict.keys()).issubset(np.arange(nS)):
                warnings.warn(f"Some keys of the rewards dictionary `rewards_dict` are NOT a valid state of the environment (valid keys are between 0 and {nS-1}). The reward for those states will be ignored.")
        # Initial state distribution
        if initial_state_distribution is not None:
            if not isinstance(initial_state_distribution, (list, tuple, np.ndarray)) or len(initial_state_distribution) != np.prod(self.shape):
                raise ValueError(f"The initial state distribution must be a list, tuple or numpy array with as many elements as the number of states in the environment ({np.prod(self.shape)}): {initial_state_distribution}")
            if not all(np.array(initial_state_distribution) >= 0) or not np.isclose(np.sum(initial_state_distribution), 1.0):
                raise ValueError(f"The values in the initial state distribution must sum up to 1 (sum = {np.sum(initial_state_distribution)})")

        #----- Parse input parameters 3
        # Terminal states and their rewards
        if terminal_states is None:
            set_terminal_states = set([0, nS-1])
            if rewards_dict is None:
                rewards_dict = dict({0: -1.0, nS-1: 1.0})
        else:
            set_terminal_states = set(terminal_states)
        # Obstacle states
        if obstacle_states is not None:
            if not set(obstacle_states).issubset(np.arange(nS)):
                raise ValueError(f"The obstacle states must be a subset of the environment states, i.e. they must be integer values between 0 and {nS-1}: {obstacle_states}")
        else:
            # Convert None to an emtpy set so that we can more easily handle the obstacle states below
            obstacle_states = set({})

        # Terminal and obstacle states (which cannot be chosen as start states => the corresponding value of the isd array for those states should be set to 0)
        set_terminal_and_obstacle_states = set.union(set_terminal_states, obstacle_states)
        num_nonterminal_and_nonobstacle_states = nS - len(set_terminal_and_obstacle_states)

        # Check the initial state distribution if given
        if initial_state_distribution is not None and not all([p == 0 for s, p in enumerate(initial_state_distribution) if s in set_terminal_and_obstacle_states]):
            raise ValueError(f"The initial state distribution must be 0 at the terminal and obstacle states ({set_terminal_states}):\ngiven distribution = {initial_state_distribution}")

        #----- Auxiliary functions
        # Function that checks whether the state is a terminal state
        # (Note: we cannot use the superclass method because we haven't called the superclass constructor yet)
        is_terminal = lambda s: s in set_terminal_states
        # Function that checks whether the state is an obstacle state
        is_obstacle = lambda s: s in obstacle_states
        # Function that returns the reward received when visiting a state
        reward = lambda s: rewards_dict[s] if s in rewards_dict.keys() else 0.0
        #----- Auxiliary functions

        # Define the initial state distribution to pass to the superclass
        if initial_state_distribution is None:
            # Initialize the initial state distribution matrix to 0
            # This matrix will be updated as we go along constructing the environment below
            # and will be set to a uniform distribution on the NON-terminal and NON-obstacle states.
            isd = np.zeros(nS)
        else:
            isd = initial_state_distribution

        # Obstacles set given in 2D-coordinate states
        # This is used to define the next state based on the action chosen by the agent and on the eventual wind present in the environment.
        # Note that we define the outer border of the gridworld as part of the obstacles set so that computing the next state is easier.
        # Recall that below the 2D coordinates are referred to as (y, x) where y varies downwards and x varies rightwards.
        # All natural obstacles (i.e. gridworld outer borders) are out of range for the given `shape` of the environment.
        set_obstacle_states_2d = set.union( set([(y,            -1) for y in np.arange(-1, self.shape[0]+1)]),
                                            set([(y, self.shape[1]) for y in np.arange(-1, self.shape[0]+1)]),
                                            set([(-1,            x) for x in np.arange(-1, self.shape[1]+1)]),
                                            set([(self.shape[0], x) for x in np.arange(-1, self.shape[1]+1)]),
                                            set([np.unravel_index(o, self.shape) for o in obstacle_states]))

        # Obstacles set given in 1D-coordinate states, which is stored as an attribute of the class
        self.set_obstacle_states = set(obstacle_states)

        # Define the transition probabilities
        P = {}
        grid = np.arange(nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index   # 0 <= y <= shape[0]-1, 0 <= x <= shape[1]-1

            # Initial state distribution
            if initial_state_distribution is None:
                if not is_terminal(s) and not self.is_obstacle_state(s):
                    isd[s] = 1 / num_nonterminal_and_nonobstacle_states

            # Transition dictionary containing the rewards information:
            # Given that the system is at state 's' and the agent takes action 'a', it has the form:
            # P[s][a] = (prob, next_state, reward, is_terminal), where:
            # - prob: probability of going to state 'next_state'
            # - next_state: a possible next_state when taking action 'a'
            # - reward: reward received when going to state 'next_state'
            # - is_terminal: whether 'next_state' is a terminal state
            P[s] = {a : [] for a in range(nA)}

            if is_terminal(s) or is_obstacle(s):
                # The system has reached a terminal state or is at an obstacle state
                # => Set the transition information to indicate that the system stays there
                # Note that this is not really what happens with obstacle states, where the system can NEVER go, but we still set the transition probability to 1 of staying there
                # because this facilitates the treatment an checks done on the transition probability.
                P[s][Direction2D.UP.value]    = [(1.0, s, reward(s), is_terminal(s))]
                P[s][Direction2D.RIGHT.value] = [(1.0, s, reward(s), is_terminal(s))]
                P[s][Direction2D.DOWN.value]  = [(1.0, s, reward(s), is_terminal(s))]
                P[s][Direction2D.LEFT.value]  = [(1.0, s, reward(s), is_terminal(s))]
            else:
                # Not a terminal state
                # ns = Next State

                # CONVENTIONS FOR WHAT FOLLOWS:
                # - All 2D-coordinate states may be INVALID states, i.e. either obstacle or out of range (they will become valid when they are converted to 1D coordinates).
                # - All 1D-coordinate states must be VALID states, i.e. neither obstacle nor out of range.
                # - All UP, RIGHT, DOWN, LEFT directions are abbreviated as UP, RT, DN, LT when given as 2D coordinates, but written in full when given in 2D coordinates.
                ns_up_2d = (y - 1, x)
                ns_rt_2d = (y    , x + 1)
                ns_dn_2d = (y + 1, x)
                ns_lt_2d = (y    , x - 1)

                # 1D states that are VALID states of the environment, i.e. none of these states must be part of the obstacle states
                ns_up    = s if ns_up_2d in set_obstacle_states_2d else np.ravel_multi_index(ns_up_2d, self.shape)
                ns_right = s if ns_rt_2d in set_obstacle_states_2d else np.ravel_multi_index(ns_rt_2d, self.shape)
                ns_down  = s if ns_dn_2d in set_obstacle_states_2d else np.ravel_multi_index(ns_dn_2d, self.shape)
                ns_left  = s if ns_lt_2d in set_obstacle_states_2d else np.ravel_multi_index(ns_lt_2d, self.shape)
                assert not set({ns_up, ns_right, ns_down, ns_left}).issubset(obstacle_states)
                if wind_dict is None:
                    # Deterministic movement of the agent given the selected action
                    P[s][Direction2D.UP.value]    = [(1.0, ns_up,    reward(ns_up),    is_terminal(ns_up))]
                    P[s][Direction2D.RIGHT.value] = [(1.0, ns_right, reward(ns_right), is_terminal(ns_right))]
                    P[s][Direction2D.DOWN.value]  = [(1.0, ns_down,  reward(ns_down),  is_terminal(ns_down))]
                    P[s][Direction2D.LEFT.value]  = [(1.0, ns_left,  reward(ns_left),  is_terminal(ns_left))]
                else:
                    # There is WIND (for now only ONE wind direction is allowed)
                    # => The transition is no longer deterministic when the agent takes an action on a given state

                    # Define the (objective) diagonal cells for the current state in 2D coordinates, regardless of whether those states are obstacles ore out of range
                    ns_ne_2d = (y - 1, x + 1)
                    ns_se_2d = (y + 1, x + 1)
                    ns_sw_2d = (y + 1, x - 1)
                    ns_nw_2d = (y - 1, x - 1)

                    # Define the next state given the direction the agent decides to move and the direction of the wind
                    # Note that each next state is given as a 1D-coordinate state and that they are VALID states
                    # (because they are determined after analyzing whether each possible next state (based on movement direction and wind direction) is valid).
                    dict_next_state_dictated_by_wind_for_each_direction = dict({
                        # Directions (of movement)
                        Direction2D.UP: {
                                        # Possible wind directions
                                        Direction2D.UP: ns_up,
                                        Direction2D.RIGHT: np.ravel_multi_index(ns_ne_2d, self.shape) if ns_ne_2d not in set_obstacle_states_2d
                                                    else   np.ravel_multi_index(ns_up_2d, self.shape) if ns_up_2d not in set_obstacle_states_2d
                                                    else   s,
                                        Direction2D.DOWN: s,
                                        Direction2D.LEFT: np.ravel_multi_index(ns_nw_2d, self.shape) if ns_nw_2d not in set_obstacle_states_2d
                                                    else  np.ravel_multi_index(ns_up_2d, self.shape) if ns_up_2d not in set_obstacle_states_2d
                                                    else  s
                                        },
                        Direction2D.RIGHT: {
                                        # Possible wind directions
                                        Direction2D.UP: np.ravel_multi_index(ns_ne_2d, self.shape) if ns_ne_2d not in set_obstacle_states_2d
                                                else    np.ravel_multi_index(ns_rt_2d, self.shape) if ns_rt_2d not in set_obstacle_states_2d
                                                else    s,
                                        Direction2D.RIGHT: ns_right,
                                        Direction2D.DOWN: np.ravel_multi_index(ns_se_2d, self.shape) if ns_se_2d not in set_obstacle_states_2d
                                                    else  np.ravel_multi_index(ns_rt_2d, self.shape) if ns_rt_2d not in set_obstacle_states_2d
                                                    else  s,
                                        Direction2D.LEFT: s
                                        },
                        Direction2D.DOWN: {
                                        # Possible wind directions
                                        Direction2D.UP: s,
                                        Direction2D.RIGHT: np.ravel_multi_index(ns_se_2d, self.shape) if ns_se_2d not in set_obstacle_states_2d
                                                    else   np.ravel_multi_index(ns_dn_2d, self.shape) if ns_dn_2d not in set_obstacle_states_2d
                                                    else   s,
                                        Direction2D.DOWN: ns_down,
                                        Direction2D.LEFT: np.ravel_multi_index(ns_sw_2d, self.shape) if ns_sw_2d not in set_obstacle_states_2d
                                                    else  np.ravel_multi_index(ns_dn_2d, self.shape) if ns_dn_2d not in set_obstacle_states_2d
                                                    else  s
                                        },
                        Direction2D.LEFT: {
                                        # Possible wind directions
                                        Direction2D.UP: np.ravel_multi_index(ns_nw_2d, self.shape) if ns_nw_2d not in set_obstacle_states_2d
                                                else    np.ravel_multi_index(ns_lt_2d, self.shape) if ns_lt_2d not in set_obstacle_states_2d
                                                else    s,
                                        Direction2D.RIGHT: s,
                                        Direction2D.DOWN: np.ravel_multi_index(ns_sw_2d, self.shape) if ns_sw_2d not in set_obstacle_states_2d
                                                    else  np.ravel_multi_index(ns_lt_2d, self.shape) if ns_lt_2d not in set_obstacle_states_2d
                                                    else  s,
                                        Direction2D.LEFT: ns_left
                                        }
                    })

                    # Use the WIND information to define the transition probabilities dictionary values for the current state s
                    wind_direction = wind_dict['direction']
                    wind_intensity = wind_dict['intensity']
                    assert wind_direction in Direction2D.__members__.values()
                    assert 0 <= wind_intensity <= 1, f"The wind intensity must be in [0, 1]: {wind_intensity}"

                    # Start with the deterministic next state for each action if there were no wind
                    P[s][Direction2D.UP.value]    = [(1.0 - wind_intensity, ns_up,    reward(ns_up),    is_terminal(ns_up))]
                    P[s][Direction2D.RIGHT.value] = [(1.0 - wind_intensity, ns_right, reward(ns_right), is_terminal(ns_right))]
                    P[s][Direction2D.DOWN.value]  = [(1.0 - wind_intensity, ns_down,  reward(ns_down),  is_terminal(ns_down))]
                    P[s][Direction2D.LEFT.value]  = [(1.0 - wind_intensity, ns_left,  reward(ns_left),  is_terminal(ns_left))]

                    # Now add the other possible next state where the agent can end up given the wind
                    for direction in Direction2D.__members__.values():
                        ns_by_wind = dict_next_state_dictated_by_wind_for_each_direction[direction][wind_direction]
                        P[s][direction.value] += [(wind_intensity, ns_by_wind, reward(ns_by_wind), is_terminal(ns_by_wind))]

            # Check that probabilities of moving to any possible next state for the current state s sum up to 1
            for direction in Direction2D.__members__.values():
                assert np.isclose(sum([e[0] for e in P[s][direction.value]]), 1), f"The probabilities of the next state given action {direction} must sum up to 1: {sum([e[0] for e in P[s][direction.value]])}"

            it.iternext()

        self.P = P
        #print("P")
        #print(P)

        # For obstacle states, check that the transition probabilities of every adjacent state of going to the obstacle state is 0
        for s_obstacle in obstacle_states:
            set_adjacent_states = get_adjacent_states(self.getShape(), s_obstacle)
            for s_adjacent, d in set_adjacent_states:
                if s_adjacent is not None:
                    opposite_d = get_opposite_direction(d)
                    proba_going_to_obstacle_from_adjacent_state = sum([e[0] for e in self.P[s_adjacent][opposite_d.value] if e[1] == s_obstacle])
                    assert proba_going_to_obstacle_from_adjacent_state == 0, \
                        "The transition probability going {} to obstacle state s_obstacle={} from its {} adjacent state (s_adjacent={}) must be 0: P[{}][{}] = {}, P(s_adjacent -> s_obstacle) = {}" \
                            .format(opposite_d.name, s_obstacle, d.name, s_adjacent, s_adjacent, opposite_d.name, self.P[s_adjacent][opposite_d.value], proba_going_to_obstacle_from_adjacent_state)

        assert np.isclose(np.sum(isd), 1.0), "The initial state probabilities must sum up to 1 ({})".format(np.sum(isd))

        super(EnvGridworld2D, self).__init__(nS, nA, P, isd,
                                             dim=2,
                                             terminal_states=set_terminal_states,
                                             rewards=rewards_dict)

    def isObstacleState(self, state):
        return state in self.set_obstacle_states

    def is_obstacle_state(self, state):
        return self.isObstacleState(state)

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
                output = " ({:2d}) P ".format(s)    # "P" stands for "Prohibited"
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

    #--- Getters
    def getShape(self):
        return self.shape

    def getAllValidStates(self):
        "Returns the list of valid states (i.e. states that are not obstacles)"
        return [s for s in self.getAllStates() if not self.isObstacleState(s)]

    def getObstacleStates(self):
        "Returns the list of obstacle states"
        return list(self.set_obstacle_states)


class EnvGridworld2D_Random(EnvGridworld2D):
    def __init__(self, shape=[5, 5], terminal_states=None, seed=None , rewards_dict=None, wind_dict=None, initial_state_distribution=None):
        nRows, nColumns = shape
        nObstacles = int(nRows*nColumns*0.4)
        start = nColumns*(nRows-1)
        end = nColumns-1

        list_of_possible_obstacles = [*range(nColumns*nRows)]
        list_of_possible_obstacles.pop(start)
        list_of_possible_obstacles.pop(end)

        random.seed(seed)
        no_path = True
        directions = [-nColumns, 1, nColumns, -1]
        obstacle_list = []
        while no_path:
            queue = [start]
            visited = set()
            obstacle_list = random.sample(list_of_possible_obstacles, nObstacles)
            while queue:
                state = queue.pop()

                if state == end:
                    no_path = False

                for direction in directions:
                    new_state = state + direction
                    out_of_bounds = False
                    if new_state not in obstacle_list and new_state not in visited:
                        #Checking if we don't go out of bounds
                        if not 0 <= new_state <= nColumns*nRows - 1:
                            out_of_bounds = True
                        if direction == 1 and new_state%nColumns == 0:
                            out_of_bounds = True
                        if direction == -1 and new_state%nColumns == -1%nColumns:
                            out_of_bounds = True

                        if not out_of_bounds:
                            visited.add(new_state)
                            queue = queue + [new_state]

        super().__init__(shape=shape, terminal_states=terminal_states, obstacle_states=obstacle_list , rewards_dict=rewards_dict, wind_dict=wind_dict, initial_state_distribution=initial_state_distribution)

    def plot_labyrinth(self):
        nRows, nColumns = self.shape
        grid = np.zeros((nRows, nColumns))

        for obs in self.set_obstacle_states:
            raw, column = divmod(obs, nColumns)
            grid[raw][column] = 1

        start = nColumns * (nRows - 1)
        end = nColumns - 1
        start_row, start_column = divmod(start, nColumns)
        end_row, end_column = divmod(end, nColumns)

        grid[start_row][start_column] = 0.5
        grid[end_row][end_column] = 0.5

        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='Greys')
        plt.text(start_column, start_row, 'Start', ha='center', va='center', fontsize=15)
        plt.text(end_column, end_row, 'Finish', ha='center', va='center', fontsize=15)
        plt.title('Generated Labyrinth')
        plt.tight_layout()
        plt.show()