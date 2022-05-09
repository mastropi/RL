# -*- coding: utf-8 -*-
"""
Created on Wed May 4 19:35

@author: Daniel Mastropietro
@description: Definition of the mountain car environment with discrete actions,
inheriting from the environment defined in the gym package by openAI.
Ref: https://gym.openai.com/docs/
Source code and description of environment: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
"""

import copy
import numpy as np

from gym.envs.toy_text.discrete import categorical_sample
from gym.envs.classic_control import MountainCarEnv

from Python.lib.utils.basic import discretize, index_linear2multi, index_multi2linear
from Python.lib.environments import EnvironmentDiscrete


class MountainCarDiscrete(MountainCarEnv, EnvironmentDiscrete):
    """
    Discrete Mountain Car environment defining the methods used by my learners, such as LeaTDLambda, for instance
    to get the number of states in the environment, getNumStates().

    The constructor receives the number of points in which we would like to discretize the state space, in each of
    its dimensions, namely the position and the velocity, which are allowed to vary between [-max_position, +max_position]
    and [-max_speed, +max_speed], respectively, where the two names are attributes of the MountainCarEnv environment
    from which this class derives and are normally equal to max_position = 0.6, max_speed = 0.07.
    """

    def __init__(self, nx, nv):
        super().__init__()
        # Attributes defined in super classes
        self.dim = 2                    # From my EnvironmentDiscrete: dimension of the environment
                                        # In principle this is used only for plotting purposes, e.g. when plotting
                                        # the estimated value function as a 2D image for `dim = 2` environments.
        self.isd = None                 # From my EnvironmentDiscrete class which in turn is inherited from the discrete environment in toy_text
        self.terminal_states = set()    # TDOO-2022/05/06 From my EnvironmentDiscrete class
        self.terminal_rewards = []      # TDOO-2022/05/06 From my EnvironmentDiscrete class

        # Shape information of the 2D state information
        self.nx = nx
        self.nv = nv
        self.shape = (self.nx, self.nv)     # Number of rows x number of columns
                                            # We put the positions on the rows because the state returned by the MountainCarEnv is (x, v),
                                            # i.e. the first dimension is the position, so that we do a direct mapping
                                            # from "first dimension" to "row" in the 2D shape.

        # Minimum and maximum values for the position and velocity
        # Note: In the documentation of the source code in GitHub (mentioned above) it says that:
        # -1.2 <= x <= 0.6
        # -0.07 <= v <= 0.07
        # that's why we set xmin to -2*self.max_position, because self.max_position = 0.6
        self.xmin, self.xmax = -2.0 * self.max_position, self.max_position
        self.vmin, self.vmax = -self.max_speed, self.max_speed

        # Discretization sizes
        self.dx = (self.xmax - self.xmin) / self.nx
        self.dv = (self.vmax - self.vmin) / self.nv

        # Number of states and actions
        self.nS = self.nx * self.nv
        self.nA = self.action_space.n
        self.all_states = list(range(self.nS))

        # In case we need to have all possible (x, v) indices indexing the discrete position and velocity
        # We can go from 2D indices to 1D indices and viceversa using respectively the following functions
        # defined in the `basic` module:
        # index_multi2linear()
        # index_linear2multi()
        # Note: the reshaping is done by row (e.g. [[0, 1, 2], [3, 4, 5]] when nx = 2, nv = 3)
        # self.all_states_2d = np.arange(self.nS).reshape(self.nx, self.nv)
        # In case we need to iterate on all possible states
        #it = np.nditer(states2d, flags=['multi_index'])             # Retrieve the original 2D layout with `x, v = it.multi_index` when iterating with `while not it.finished`.

    # (2022/05/05) In the current version gym-0.12.1 that I have installed, the MountainCarEnv environment does not accept
    # a seed when resetting the environment. I should update the gym installation.
    #def reset(self, seed=None):
        #super().reset(seed=seed)
    def reset(self, seed=None):
        if self.isd is not None:    # isd = Initial State Distribution (defined in the environments.__init__.EnvironmentDiscrete class)
            # Set an initial random state using the EnvironmentDiscrete procedure, which is actually the gym.Env procedure
            # that uses the Initial State Distribution information to pick the initial state
            #print("ISD: {} (which={})".format(self.isd, np.where(self.isd==1.0)))
            idx_state = categorical_sample(self.isd, self.np_random)
            state = self.get_state_from_index(idx_state)
            self.state = state
        else:
            # Set an initial random state using the MountainCarEnv procedure
            self.seed(seed)         # This method is defined in gym.Env
            state = super().reset()
        return state

    def step(self, action, return_continuous_observation=False):
        """
        Performs one step in the environment for the given action

        Arguments:
        action: int
            Action taken by the agent on the environment.

        return_continuous_observation: boolean
            Whether to include the continuous observation value in the returned tuple.
            It should be set to False when using step() on simulations, because simulations normally expect an 1D index
            as next state of the environment (following the action taken).
            default: False

        Return: either a 5-tuple or a 4-tuple depending on the value of return_continuous_observation
        If return_continuous_observation = False:
            It returns a 4-tuple with the 1D state index representing the current state of the environment:
            idx_state, reward, done, info
        If return_continuous_observation = True:
            It returns a 5-tuple with the actual 2D state representing the current state of the environment:
            "continuous observation", "discretized observation", reward, done, info
        """
        observation, reward, done, info = super().step(action)
        #print("\nStep: action = {} -> observation = {}, done = {}".format(action, observation, done))

        # Discretize position and velocity
        # These are integer values in [0, nx-1] and [0, nv-1] respectively for position and velocity
        observation_discrete = self.discretize(observation)

        if return_continuous_observation:
            return observation, observation_discrete, reward, done, info
        else:
            idx_state = self.getState()
            return idx_state, reward, done, info

    def discretize(self, state):
        """
        Discretizes the given environment state using the discretization parameters defined in the object

        Arguments:
        state: array, list or tuple
            The continued-value state (x, v) to discretize in the x and in the v direction, respectively.

        Return: numpy.ndarray
        2D-array containing the discretized x and v values represented by integer values indexing the respective
        discretizing interval to which x and v belong.
        """
        state_discrete = copy.deepcopy(np.array(state))
        state_discrete[0] = discretize(state[0], self.nx, self.xmin, self.xmax)
        state_discrete[1] = discretize(state[1], self.nv, self.vmin, self.vmax)
        return state_discrete

    def undiscretize(self, state_discrete):
        """
        Returns the continuous position and velocity values based on the discrete state which contains indices
        of a (hypothetical) 2D matrix indexing all the possible discrete positions and velocities.

        Arguments:
        state: array, list or tuple
            The discrete-valued state (xd, vd) to undiscretize in the x and in the v direction, respectively.

        Return: array
        2D array containing the continuous position in the first element and the continuous velocity at the second
        element.
        The continuous-valued position and velocity are chosen as the left bound of the discretizing interval
        corresponding to the given discrete state. It's important to keep the left bound as the value of the continuous
        value because the goal is on the right of the mountain, meaning that the isTerminalState() method defined
        below does not contradict the `done` condition returned by the continuous-state MountainCarEnv environment,
        which is based on the position PRIOR to discretization.
        """
        return np.array([self.xmin + state_discrete[0] * self.dx, self.vmin + state_discrete[1] * self.dv])

    def get_index_from_state(self, state):
        "Returns the 0, ..., nS-1 index corresponding to the given continuous-valued state"
        state_discrete = self.discretize(state)
        idx_state = index_multi2linear(state_discrete, self.shape, order='F')
        return idx_state

    def get_state_from_index(self, idx_state: int):
        idx_position, idx_velocity = index_linear2multi(idx_state, self.shape, order='F')   # order='F' means sweep the 2D matrix by column
        state = self.undiscretize(np.array([idx_position, idx_velocity]))
        return state

    def get_positions(self):
        "Returns the left bound of all discretizing intervals of the car's position"
        return self.xmin + np.arange(self.nx) * self.dx

    def get_velocities(self):
        "Returns the left bound of all discretizing intervals of the car's velocity"
        return self.vmin + np.arange(self.nv) * self.dv

    def get_indices_for_non_terminal_states(self):
        idx_positions_non_terminal = [idx for idx, x in enumerate(self.get_positions()) if x < self.goal_position]

        # Add all the linear indices associated to the above positions coming from all the different possible velocities
        idx_states_non_terminal = []
        for idx_vel in range(self.nv):
            idx_states_non_terminal += [idx + idx_vel*self.nx for idx in idx_positions_non_terminal]

        return idx_states_non_terminal

    #--- Getters
    def getShape(self):
        return self.shape

    def getState(self):
        """
        Returns the state index associated to the current environment observation

        Return: int
        Index between 0 and nS-1, where nS is the number of states in the environment.
        """
        state_discrete = self.discretize(np.array(self.state))
            ## WARNING: The `state` attribute is stored as a tuple inside the MountainCarEnv environment,
            ## but returned as numpy array by the reset() method and by the step() method!!
            ## So, here we should convert self.state to a numpy array, otherwise we get the error in our discretize()
            ## method that tuple values cannot be updated (when trying to set the value of state_discrete)
        idx_state = index_multi2linear(state_discrete, self.shape, order='F')
        assert 0 <= idx_state < self.nS, "The current state index is between 0 and nS-1 = {} ({})".format(self.nS-1, idx_state)
        return idx_state

    def isTerminalState(self, idx_state: int):
        state = self.get_state_from_index(idx_state)
        x = state[0]
        #print("Check terminal state: {} (dx={})".format(state, self.dx))
        return x >= self.goal_position

    #--- Setters
    def setState(self, state):
        "The state should be a duple (x, v) with the car's position and velocity to set"
        self.env.state = state
