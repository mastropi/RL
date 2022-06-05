# -*- coding: utf-8 -*-
"""
Created on Wed May 4 19:35

@author: Daniel Mastropietro
@description: Definition of the mountain car environment with discrete actions,
inheriting from the environment defined in the gym package by openAI.
Ref: https://gym.openai.com/docs/
Source code and description of environment: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
Example of use on the mountain car environment: https://mpatacchiola.github.io/blog/2017/08/14/dissecting-reinforcement-learning-6.html
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
        # Shape information of the 2D state information
        self.nx = nx
        self.nv = nv
        self.shape = (self.nx, self.nv)     # Number of rows * number of columns
                                            # We put the positions on the rows because the state returned
                                            # by the MountainCarEnv is the tuple (x, v)
                                            # (i.e. the first dimension is the position),
                                            # and by doing so we do a direct and natural mapping
                                            # from "first dimension" of the tuple to "row" in the 2D shape.
        self.shape_names = ("position", "velocity")
        # Minimum and maximum values for the position and velocity
        # Note: In the documentation of the source code in GitHub (mentioned above) it says that:
        # -1.2 <= x <= 0.6
        # -0.07 <= v <= 0.07
        self.xmin, self.xmax = self.min_position, self.max_position
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

        # Attributes defined in super classes
        self.dim = 2                    # From my EnvironmentDiscrete: dimension of the environment
                                        # In principle this is used only for plotting purposes, e.g. when plotting
                                        # the estimated value function as a 2D image for `dim = 2` environments.
        self.isd = None                 # From my EnvironmentDiscrete class which in turn is inherited from the discrete environment in toy_text

        self.non_terminal_states = set(self.get_indices_for_non_terminal_states())
        self.terminal_states = set(self.all_states).difference( set( self.non_terminal_states ) )
        self.terminal_rewards = []      # TDOO-2022/05/06 From my EnvironmentDiscrete class

        #-- True value functions (for algorithm evaluation purposes)
        #-- Note that terminal states have value = 0
        self.V = None

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
        if done:
            # Fix the reward when the car reaches the goal!
            # In the original implementation (see gym/envs/classic_control/mountain_car.py) the reward is ALWAYS -1.0!!
            # (although the documentation reads:
            # "The goal is to reach the flag placed on top of the right hill as quickly as possible,
            # as such the agent is penalised with a reward of -1 for each timestep it isn't at the goal and
            # is not penalised (reward = 0) for when it reaches the goal."
            # IN ANY CASE, THIS "bug" IS NOT A BIG DEAL, because the total reward observed when the car reaches the goal
            # no longer increases (negatively) making the rewards closer to the goal more valuable. So, it's ok.
            reward = 0.0
            #print("\nStep: action = {} -> observation = {}, reward = {}, done = {}".format(action, observation, reward, done))

        # Reduce the reward amplitude, just to try whether smaller rewards make TD algorithms converge faster
        # when the initial value function guess is all zeros
        #reward = reward / 1000

        # Make the environment work as in gridworld, i.e. a non-zero reward is received at the terminal states
        # In the original MountainCar environment, all rewards are -1.0 except at the terminal state which is 0,
        # therefore we reverse these rewards here.
        #reward = 1.0 + reward

        # Discretize position and velocity
        # These are integer values in [0, nx-1] and [0, nv-1] respectively for position and velocity
        observation_discrete = self.discretize(observation)

        if return_continuous_observation:
            return observation, observation_discrete, reward, done, info
        else:
            idx_state = self.get_index_from_state_discrete(observation_discrete)
            return idx_state, reward, done, info

    # NOTE: There is a numpy function to discretize called numpy.digitize()
    # See https://mpatacchiola.github.io/blog/2017/08/14/dissecting-reinforcement-learning-6.html
    # for an example of use in the Mountain Car environment.
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

    def reshape_from_2d_to_1d(self, values_2d):
        """
        Converts a 2D representation of the states to a 1D representation

        The 2D representation is swept by column during the conversion to 1D representation,
        i.e. we fix the column of the 2D representation, we sweep the rows, we change the column, we seep the rows, etc.,
        which is what is done by order='F' (Fortran-like) in the call to reshape().

        In terms of the mountain car states, each row represents a fixed position, and each column a fixed velocity,
        which means that in the 1D representation we will see first the different positions for the smallest velocity,
        up to the different positions for the largest velocity.
        """
        return np.squeeze( values_2d.reshape(1, np.prod(values_2d.shape), order='F') )

    def reshape_from_1d_to_2d(self, values_1d):
        """
        Converts a 1D representation of the states to a 2D representation

        The values laid out in the 1D representation are arranged in the 2D representation by column first,
        i.e. we fix the column, we fill the rows, we change the column, we fill the rows, etc.,
        which is what is done by order='F' (Fortran-like) in the call to reshape().

        In terms of the mountain car states, each row represents a fixed position, and each column a fixed velocity.
        """
        return values_1d.reshape(self.nx, self.nv, order='F')

    def get_index_from_state(self, state):
        "Returns the 0, ..., nS-1 index corresponding to the given continuous-valued state"
        state_discrete = self.discretize(state)
        idx_state = self.get_index_from_state_discrete(state_discrete)
        return idx_state

    def get_index_from_state_discrete(self, state_discrete):
        """
        Returns the 1D state index from the 2D state indices

        Arguments:
        state_discrete: list, tuple or numpy array
            2D indices of the state.

        Return: int
        1D index of the state in the 1D representation of the 2D shape indexing the states.
        Different positions come first in the 1D representation of the states for the smallest velocity.
        """
        return index_multi2linear(state_discrete, self.shape, order='F')     # order='F' means sweep the 2D matrix by column

    def get_state_from_index(self, idx_state: int):
        """
        Returns the continuous-valued position and velocity from the given 1D state index.
        The conversion to continuous values is done by the undiscretize() method.
        """
        state_discrete = self.get_state_discrete_from_index(idx_state)
        state = self.undiscretize(state_discrete)
        return state

    def get_state_from_discrete_state(self, state_discrete):
        """
        Returns the continuous-valued position and velocity from the given 2D state indices.
        This is an alias for the undiscretize() method.
        """
        return self.undiscretize(state_discrete)

    def get_state_discrete_from_index(self, idx_state: int):
        return index_linear2multi(idx_state, self.shape, order='F')   # order='F' means sweep the 2D matrix by column

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

    def getShapeNames(self):
        """
        Returns the names of the measures stored in each dimension of the 2D representation of the states

        Return: tuple
        Duple with the following elements:
        - name of the row labels of the 2D shape storing the states (e.g. "position")
        - name of the column labels of the 2D shape storing the states (e.g. "velocity")
        """
        return self.shape_names

    def getPositionDimension(self):
        "Returns the dimension in the 2D shape of states along which the different positions are stored"
        return self.shape_names.index("position")

    def getVelocityDimension(self):
        "Returns the dimension in the 2D shape of states along which the different velocities are stored"
        return self.shape_names.index("velocity")

    def getPositionColor(self):
        "Returns the color to use in plots about the position"
        return "red"

    def getVelocityColor(self):
        "Returns the color to use in plots about the velocity"
        return "blue"

    def getV(self):
        return self.V

    def getState(self):
        """
        Returns the state index associated to the current environment observation

        Return: int
        Index between 0 and nS-1, where nS is the number of states in the environment.
        """
        idx_state = self.get_index_from_state(self.state)
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

    def setV(self, state_values: np.ndarray(1)):
        "Sets the state value function for a particular policy (not shown explicitly)"
        self.V = state_values
