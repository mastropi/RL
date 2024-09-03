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
import pandas as pd

from gym.envs.toy_text import discrete
from gym import spaces

#__all__ = [ 'EnvironmentDiscrete' ]


class EnvironmentDiscrete(discrete.DiscreteEnv):
    """
    Class defining methods that are generic to ALL environments with discrete state and actions
    optionally having terminal states which are suitable for episodic learning tasks. Rewards are only accepted
    at terminal states.

    Classes inheriting from this class should implement:
    - render(): a method that displays the environment.

    Constructor parameters:
    - nS: number of possible states
    - nA: number of possible actions
    - P: 2D dictionary with entries all possible states of the environment and for each state all possible actions
      for that state. Each dictionary entry value is a list of tuples of the form:
      (prob_going_to_next_state, next_state, reward(next_state), is_terminal(next_state))
    - isd: (array-like, i.e. list is also possible) initial state distribution
    - dim: environment dimension (e.g. 1D gridworld, 2D gridworld, etc.), which is just informative 
    - terminal_states: set containing the terminal states of the environment
    - rewards: dictionary indexed by the states in the environment containing the rewards of visiting them as values.
        The number of elements in this dictionary does NOT need to coincide with the number of states in the environment.
        If a state is not found among the keys of the dictionary, its associated reward is returned as 0 by the corresponding method.
    - store_trajectory: (bool) whether to store the trajectory observed during a simulation in the object.
    This is useful for instance when running a Fleming-Viot simulation where N particles evolve separately
    and we want to keep track of the state and actions visited by each particle.
    """

    def __init__(self, nS, nA, P, isd, dim=1, terminal_states=set(), rewards=dict(), store_trajectory=False):
        super().__init__(nS, nA, P, isd)

        # Dimension of the environment (e.g. 1D (gridworld), 2D (gridworld), etc.)
        self.dim = dim

        #--- 2023/10/04 ---
        # State and action space (possibly used sometimes as I migrate to defining states and actions as integers to defining them as gym.spaces.Discrete
        # which ***are still integers*** but encapsulated as part of the gym.spaces.Discrete class.
        # Particularly useful are methods contains(x) to check whether element x is in the space (e.g. `state_space.contains(3)`) and sample() that retrieves a random sample
        # according to the internal random number generator whose seed can be set with seed() (e.g. `action_space.seed(1313)`).
        # Note that in order to *iterate* on all possible states or actions, we CANNOT do `for s in state_space` as it says that state_space is not iterable!!!
        # We need to do instead: `for s in range(state_space.n)`... I can't believe it.
        self.state_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)
        #--- 2023/10/04 ---

        self.all_states = list( np.arange(self.getNumStates()) )
        self.terminal_states = set(terminal_states)
        self.non_terminal_states = set(self.all_states).difference(self.terminal_states)
        self.rewards = rewards

        self.store_trajectory = store_trajectory
        self.df_trajectory = None
        self.reset_trajectory()

        # Define the true value functions as all zeros when they are not defined already by the actual environment (inheriting from this class)
        # The goal is to be able to do analysis of the *estimated* value functions even when the true state value functions are unknonwn
        # because some of these analyses require that the true value function be defined (e.g. when computing the RMSE value of the estimated value function)
        # Ex: It is possible to request the Simulator._run_single() method to analyze how the value of a particular state (state_observe)
        # evolves during the learning process, but this requires that the attributes self.V be defined in the environment object.
        attributes_in_object = dir(self)
        if "V" not in attributes_in_object:
            self.V = np.array([0.0]*self.getNumStates())
        if "Q" not in attributes_in_object:
            self.Q = np.array([0.0]*self.getNumStates()*self.getNumActions())

    def reset_trajectory(self, reset_state=None):
        """
        Resets the trajectory of the environment to a data frame having just ONE record containing the current state of the environment

        This is only done when the self.store_trajectory flag is True.

        This first record has the following characteristics:
        a) The 'time' column is set to 0 (and not e.g. -1, which could be another possibility).
        This allows for:
        - working with both discrete-time and continuous-time Markov chains
        - making the time column represent the state TO WHICH the Markov chain transitions,
        which is consistent with the convention normally used that considers the Markov chain trajectory
        to be continuous from the right, see e.g. Pierre Bremaud's book in particular the graph of the jump process
        presented in Chapter 13, "Continuous-time Markov chains".

        b) The 'action' column is set to -1 because there is no particular action leading to the start state.
        *** Recall that actions are assumed to be indices starting at 0. ***

        c) The 'next_state' column is set to the start state, so it has the same value as the 'state' column.

        d) The 'reward' column is set to the reward yielded by the environment when the agent visits the start state.
        """
        if self.store_trajectory:
            if reset_state is None:
                reset_state = self.getState()
            self.df_trajectory = pd.DataFrame([[0, reset_state, -1, reset_state, self.getReward(reset_state)]],
                                               columns=['time', 'state', 'action', 'next_state', 'reward'])

    def update_trajectory(self, t, state, action, next_state, reward):
        """
        Updates the trajectory information by adding a new row in the data frame containing the
        times at which a state is visited, the action taken on the state, the state to which the process
        transitions to after taking the action, and the reward received by such transition.

        The new row is added to the end of the data frame using the number of rows already present in
        the data frame as row index.
        """
        assert self.store_trajectory, "The `store_trajectory` flag must be True when updating the trajectory stored in the environment"
        self.df_trajectory = pd.concat([self.df_trajectory,
                                        pd.DataFrame([[t, state, action, next_state, reward]],
                                                     index=[len(self.df_trajectory)],
                                                     columns=self.df_trajectory.columns)],
                                       axis=0)

    def isTerminalState(self, state):
        return set([state]).issubset(self.terminal_states)

    def is_terminal_state(self, state):
        return self.isTerminalState(state)

    def render(self, mode='human'):
        "Renders the environment. See details in the Env class defined in gym/core.py"
        raise NotImplementedError

    #--- Getters
    def isStateContinuous(self):
        return False

    def getDimension(self):
        return self.dim

    def getInitialStateDistribution(self):
        return np.copy(self.isd)

    def getStoreTrajectoryFlag(self):
        return self.store_trajectory

    def getState(self):
        "Returns the 1D index representing the environment state"
        return self.s

    def getStateFromIndex(self, s, simulation=True):
        "Returns the environment state from the given 1D state index. See more details in documentation for getIndexFromState()"
        return s

    def getStateIndicesFromIndex(self, s):
        "Returns the multidimensional state indices representing the actual environment state associated to the 1D state index"
        raise NotImplementedError

    def getIndexFromState(self, state):
        """
        Returns the 1D index representation of the given environment state

        In this class the two state representations coincide
        because states in the EnvironmentDiscrete class are defined by their 1D index representation

        This method must be defined because it is used by simulators in order to go from the actual representation of the environment
        (which could be a physical continuous-valued state, as in the Mountain Car) to the index representation of a discretized version of the environment,
        which is needed for instance to run Fleming-Viot simulations (which requires the definition of an absorption set A that is based on
        an environment with a countable state space.
        """
        return state

    def getStateSpace(self):
        return self.state_space

    def getActionSpace(self):
        return self.action_space

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

    def getAllStates(self):
        "Returns the sorted list of all states in the environment"
        return self.all_states

    def getAllValidStates(self):
        "Returns the list of valid states (i.e. states where the system can be at). Unless this method is overridden, it returns all states"
        return self.getAllStates()

    def getReward(self, s):
        "Returns the reward received when visiting the given state"
        return self.rewards.get(s, 0.0)

    def getTerminalStates(self):
        "Returns the list of terminal states"
        return list(self.terminal_states)

    def getRewards(self):
        "Returns the rewards (only their values, not the states where they occur; for the latter, use getRewardsDict())"
        return self.getRewardsDict().values()

    def getRewardsDict(self):
        "Returns the dictionary containing the rewards, indexed by the state"
        return self.rewards

    def getNonTerminalStates(self):
        "Returns the list of non terminal states"
        return list(self.non_terminal_states)

    def getTrajectory(self):
        return self.df_trajectory

    def getV(self):
        "Returns the true state value function stored in the object"
        return self.V

    def getQ(self):
        "Returns the true action value function in the object"
        return self.Q

    #--- Setters
    def setInitialStateDistribution(self, isd):
        self.isd = isd

    def setStoreTrajectoryFlag(self, store_trajectory):
        "Sets the store_trajectory flag to the given value"
        self.store_trajectory = store_trajectory

    def setSeed(self, seed=None):
        """
        Sets the seed of the environment which defines the seed and the np_random() generator of random numbers
        of the DiscreteEnv class defined in gym/envs/toy_text/discrete.py.

        The random number generator np_random() is used by the reset() and the step() methods of the DiscreteEnv class.

        seed: (opt) int
            Seed value to set.
            When None, a new np_random() generator is defined with a random seed
            (see the seed() method defined in the DiscreteEnv class defined in gym/envs/toy_text/discrete.py).
            default: None

        Return: int
        The seed value set.
        """
        return self.seed(seed)

    def setState(self, s):
        "Sets the value of the 1D index representing the environment state"
        self.s = s

    def setV(self, state_values: np.ndarray(1)):
        "Sets the true state value function for a particular policy (not shown explicitly)"
        self.V = state_values
