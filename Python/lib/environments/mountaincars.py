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

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from gym.envs.toy_text.discrete import categorical_sample
from gym.envs.classic_control import MountainCarEnv

from Python.lib.utils.basic import index_linear2multi, index_multi2linear
from Python.lib.environments import EnvironmentDiscrete


class MountainCarDiscrete(MountainCarEnv, EnvironmentDiscrete):
    """
    Discrete Mountain Car environment defining the methods used by my learners, such as LeaTDLambda.

    The constructor receives the number of points in which we would like to discretize the possible velocity values,
    which are in the range [-max_speed, +max_speed]. The max_speed value is defined in the MountainCarEnv environment
    from which this class derives and is normally equal to max_speed = 0.07.

    The discretization grid for position is automatically computed from the velocity discretization grid in order to:
    - guarantee that the goal on the right is reached
    - guarantee that the car leaves every possible interval at the smallest discretized non-zero velocity.
    The latter condition is applied to the interval with the littlest help from gravity, namely the valley,
    and the condition to leave the interval is established when the original velocity is 0.0, namely when the new
    velocity is equal to `force`.
    Note that the smaller the smallest non-zero velocity (dv), the larger the number of discretized intervals for
    position. This implies that, the larger the number of velocity intervals, the larger the number of position intervals.

    When stepping to the next state using the step() function of the super class MountainCarEnv, the continuous (x, v)
    values assigned to each discrete state are the left bounds of the x- and v- interval to which the state belongs.
    Ex: if there are 6 intervals for x: [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8], and
    5 intervals for v: [-0.07, -0.035, 0.00, 0.035, 0.07],
    then each possible state is represented by all possible combinations of the 6x5 intervals making 30 possible states,
    and e.g. the continuous-valued position and velocity for state [0.4, -0.035] which are input to the step() method
    of the super class MountainCarEnv are x = 0.4 and v = -0.035.
    Note that the state (x, v) = (0.8, 0.07) represents the cell with x values in [0.8, 1.2) and v values in [0.007, 0.105).

    The `force` and `gravity` values are multiplied 20 fold in this class from their original values defined in the
    super class MountainCarEnv, so that we don't require so many discretized intervals to guarantee the aforementioned
    condition on the position intervals, which would make the learning process very slow.
    For instance, the above logic of defining the number of discretized intervals for position would give about 1000
    intervals when we use the original `force` and `gravity` values defined in the super class MountainCarEnv,
    respectively 0.001 and 0.0025. In our class, their values become 0.02 and 0.05 respectively.

    Arguments:
    nv: int
        Number of discretization intervals for velocity.
        An odd number is enforced by adding 1 when even, so that v = 0.0 is always part of the velocity grid.

    seed_reset: (opt) int
        Seed to use when choosing the start state when resetting the environment.
        default: None
    """

    # NAMING CONVENTIONS FOR ATTRIBUTES AND VARIABLES CONTAINING STATE INFORMATION:
    # In this class we call:
    # - idx_state: 1D index representing the state (used in the discrete.DiscreteEnv environment from which the super
    # class EnvironmentDiscrete derives (it's used in its attribute `self.s`).
    # - state: 2D discretized state as tuple (x, v) = (position, velocity) representing the LOWER LIMIT of each discretized cell.
    # IMPORTANT: this `state` does NOT contain the same information as the `state` attribute in MountainCarEnv, which
    # is a non-discretized real-valued state.
    # Ex: (0.4, 0.07), which in the example given in the class documentation, it represents the 2D cell [0.4, 0.8) x [0.07, 0.105)
    # - state_discrete: 2D index of the continuous-valued 2D state.
    # Ex: (3, 5)
    # HOWEVER, despite the above naming convention, it is currently a pity that the methods getState() and setState()
    # return the state 1D INDEX (and NOT the 2D continuous state as a tuple, as one would expect from their names)
    # because these methods override those defined in the super class EnvironmentDiscrete.
    # TODO: (2022/06/06) Rename the methods in EnvironmentDiscrete to get and set the state to getStateIndex() and setStateIndex()
    # (so that when we talk about `state` we think of the human-understandable state (e.g. (x, v) in the Mountain Car environment)
    # Note that the super class EnvironmentDiscrete should define methods called getState() and setState() which should raise NotImplementedError
    # to indicate that those methods cannot be implemented in such class (EnvironmentDiscrete) because we need information
    # about the environment itself, namely about how the index state self.s defined in discrete.DiscreteEnv translates
    # into the human-understandable state.
    def __init__(self, nv, seed_reset=None, debug=False):
        MountainCarEnv.__init__(self)
        self.debug = debug
        self.setSeed(seed_reset)

        # Number of intervals  information of the 2D state information
        self.nv = nv
        # Make sure that nv is odd so that v=0.0 is part of the grid (required when defining the grid for positions below)
        # and we get symmetric values for v.
        self.nv = self.nv + 1 if self.nv % 2 == 0 else self.nv

        # Minimum and maximum values for position and velocity
        # Note: In the documentation of the source code in GitHub (mentioned above) it says that:
        # -1.2 <= x <= 0.5 (which is the goal position): note that we do not need to allow x to go past the goal position because we discretize the value of x using the left bound of each interval
        # -0.07 <= v <= 0.07
        self.xmin, self.xmax = self.min_position, self.goal_position #self.max_position
        self.vmin, self.vmax = -self.max_speed, self.max_speed

        # Discretization size for velocity
        self.dv = (self.vmax - self.vmin) / (self.nv - 1)

        # Discretization intervals in each dimension
        # It is important to make sure that:
        # a) The car can get out of the interval with the minimum possible force, which happens at the valley of the path,
        # namely when x = pi/6 ~= -0.5, where gravity = 0, therefore the force when accelerating is just self.force
        # In this case, we should consider the worst case scenario which is when the car is at velocity 0, in which case
        # the new velocity will be simply force, and this new velocity should be enough to take the car out of
        # that interval, namely, since delta(x) = velocity*1 and velocity = force, we should have:
        #   dx > force
        # In other positions, the new velocity will have also the help from gravity, therefore we are in business if
        # we satisfy the above condition for dx, i.e. the mesh grid density for the positions.
        # Note that gravity is largest at two positions, cos(3*x) = cos(-pi) and cos(0),
        # which happens when x = -pi/3 ~ 1.05 and x = 0.0
        #
        # b) The car can reach the goal on the right. This implies that the maximum velocity that can be used by the car
        # when it is at the rightmost position interval before reaching the goal should be enough to take the car to
        # the goal. This means that the rightmost position interval should start at no less than:
        #   self.goal_position - self.max_speed
        # (since the new position is determined by `position + velocity`)
        # I would even dare to say that it should reach the goal by applying also the minimum discretized velocity,
        # which means that the rightmost position should be no less than at:
        #   self.goal_position - self.dv
        # (assuming that 0.0 is a possible velocity, which is guaranteed by calling np.linspace() below with an odd
        # number of points and endpoint=True (the default)
        # Note that the number of points returned by linspace is the number of points specified as third argument,
        # regardless of the value of endpoint.
        self.force = 20*self.force
        self.gravity = 20*self.gravity
        velocity_achieved_from_zero_speed_at_zero_gravity = self.force
        position_max_leftmost = self.xmin + 0.8*velocity_achieved_from_zero_speed_at_zero_gravity
        position_min_rightmost = self.goal_position - self.vmax # This is not used but computed for informational purposes if needed
        position_rightmost = self.goal_position - self.dv
        self.dx = position_max_leftmost - self.xmin
        self.nx = int((self.xmax - self.xmin) / self.dx) + 1
        print("position_rightmost = {:.3f} > position_min_righmost (for max speed = {}) = {:.3f}".format(position_rightmost, self.vmax, position_min_rightmost))
        print("Parameters for linspace for position: min=xmin={:.3f}, max=position_rightmost={:.3f}, n=nx-1={}, with interval width=dx={:.3f}" \
              .format(self.xmin, position_rightmost, self.nx-1, self.dx))
        #self.positions = np.r_[self.xmin,
        #                       np.linspace(position_max_leftmost, position_min_rightmost, self.nx-2, endpoint=True),
        #                       self.goal_position]
        self.positions = np.r_[np.linspace(self.xmin, position_rightmost, self.nx-1, endpoint=True),
                               self.goal_position]
        self.velocities = np.linspace(self.vmin, self.vmax, self.nv, endpoint=True)
        # The above calculations assume the following
        assert self.xmax == self.goal_position
        assert 0.0 in self.velocities
        assert len(self.positions) == self.nx
        assert len(self.velocities) == self.nv
        assert np.abs(self.velocities[1] - self.velocities[0] - self.dv) < 1E-6, "v0={:.3f}, v1={:.3f}, v1-v0={:.6f}, dv={:.6f}".format(self.velocities[0], self.velocities[1], self.velocities[1] - self.velocities[0], self.dv)

        # Shape of the 2D internal representation of the state
        # IMPORTANT: shape and shape_names should be consistently defined
        # (e.g. shape = (nx, nv) and shape_names = ("position", "velocity",
        # or shape = (nv, nx) and shape_names = ("velocity", "position")
        self.shape = (self.nx, self.nv)     # Number of rows * number of columns
                                            # We put the positions on the rows because the state returned
                                            # by the MountainCarEnv is the tuple (x, v)
                                            # (i.e. the first dimension is the position),
                                            # and by doing so we do a direct and natural mapping
                                            # from "first dimension" of the tuple to "row" in the 2D shape.
        self.shape_names = ("position", "velocity")

        # Number of states and actions
        self.nS = self.nx * self.nv
        self.nA = self.action_space.n
        self.all_states = list(np.arange(self.nS))

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
        # We need to define the initial state distribution that is required by the constructor of EnvironmentDiscrete called below,
        # which in turn inherits from the `discrete` environment in toy_text
        # However, this piece of information is NOT used because the reset of the environment is done by MountainCarEnvironment, NOT by EnvironmentDiscrete
        self.isd = np.zeros(self.nS)
        self.isd[0] = 1

        self.non_terminal_states = set(self.get_indices_for_non_terminal_states())
        self.terminal_states = set(self.getAllStates()).difference( set( self.non_terminal_states ) )
        self.terminal_rewards = dict()  # TODO: (2022/05/06) Define the terminal rewards, as done in my EnvironmentDiscrete class, from which this class derives

        # Initialize the other class from which this class inherits, so that we can use its attributes (e.g. rewards, etc)
        EnvironmentDiscrete.__init__(self, self.nS, self.nA, None, self.isd, dim=self.dim, rewards=self.terminal_rewards, terminal_states=self.terminal_states)

        # 2D discrete-valued state of the environment (self.state), which gives the LOWER LIMIT of the corresponding (x, v) cell in the discretized space
        # (e.g. state = (0.8, 0.07))
        # Note however that the self.reset() method called here defines both:
        # - the 2D discretized `state` = duple (x, v) containing the discretized car's position and velocity of the MountainCarEnv environment
        # - the 1D state `s` of the discrete.DiscreteEnv of the gym library, from which EnvironmentDiscrete inherits,
        # because we need these two state attributes to be ALIGNED.
        # Otherwise, the Mountain Car environment will be at a different state than we think we are at (via the self.s attribute of discrete.DiscreteEnv)!
        # Note that the MountainCarEnv environment does not have any getter or setter to access `state`.
        self.reset()

    def setSeed(self, seed=None):
        "Sets the seed of the environment for generating random numbers such as those performed by reset() and step()"
        # Set the seed for the reset() method
        self.seed_reset = seed
        # Set the seed of the environment (e.g. used to generate the steps)
        super().seed(seed)

    # NOTE: (2022/05/05) In the current version gym-0.12.1 that I have installed, the MountainCarEnv environment does not accept
    # a seed when resetting the environment. I should update the gym installation, as the newer version does accept a seed
    # (as seen in the GitHub repository of the MountainCarEnv environment)
    def reset(self, isd=None):
        """
        Resets the environment's state, optionally using the initial state distribution given in parameter `isd`,
        otherwise the reset step of the MountainCarEnv environment is used
        """
        # Select the 1D index representing the 2D state of the mountain car
        if isd is not None:    # isd = Initial State Distribution to use, if any desired
            # Set an initial random state using the EnvironmentDiscrete procedure, which is actually the gym.Env procedure
            # that uses the Initial State Distribution information to pick the initial state
            print("ISD: {} (which={})".format(isd, np.where(isd==1.0)))
            idx_state = categorical_sample(isd, self.np_random)
        else:
            # Set an initial random state using the MountainCarEnv procedure
            self.seed(self.seed_reset)         # This method is defined in MountainCarEnv included in gym/envs/classic_control
            # Set the (x, v) state of the Mountain Car (continuous-valued 2D state)
            state_continuous_2d = super().reset() # reset() through the MountainCarEnv environment (since this is the first super class appearing in the list of super classes in the definition of this class)
            # Set the 1D index associated to the continuous-valued (x, v) state just set (this 1D index is computed using the position and velocity discretization)
            idx_state = self.get_index_from_state(state_continuous_2d)

        # Set the 2D discretized state (self.state) and its 1D index version (state of the EnvironmentDiscrete environment)
        self.setState(idx_state)

        # We need to return a state because the environment reset() performed by simulators assign the returned state from the reset() method to `next_state`
        # Otherwise, next_state will be None!
        return idx_state

    def step(self, action, return_continuous_state=False):
        """
        Performs one step in the environment for the given action

        Arguments:
        action: int
            Action taken by the agent on the environment.

        return_continuous_state: boolean
            Whether to include the continuous observation value in the returned tuple.
            It should be set to False when using step() on simulations, because simulations normally expect an 1D index
            as next state of the environment (following the action taken).
            default: False

        Return: either a 5-tuple or a 4-tuple depending on the value of return_continuous_state
        If return_continuous_state = False:
            It returns a 4-tuple with the 1D state index representing the current state of the environment:
            idx_state, reward, done, info
        If return_continuous_state = True:
            It returns a 5-tuple with the actual 2D state representing the current state of the environment:
            "continuous state", "discretized state", reward, done, info
        """
        observation, reward, done, info = super().step(action)
            ## This is the step() method defined in MountainCarEnv => observation is the next state given as (x, v)
            ## Note that this method changes the 2D continuous-valued `state` attribute of the MountainCarEnv environment
            ## which is stored as a tuple, although by the reset() method it is stored as an array! (it is also called `state`)
            ## Nevertheless, `observation` above is an array because the step() method returns an array as first element
            ## of the output tuple.
            ## We should change the 1D state index attribute self.s of the discrete.DiscreteEnv environment whose value
            ## should be consistent with the new environment's 2D state.
            ## We do this NOW.
        # Set both the 1D state index in discrete.DiscreteEnv AND the `state` attribute in this object
        # which DIFFERS from the `state` attribute stored in the MountainCarEnv object which contains the
        # real-valued state, i.e. without any sort of discretization. Here, instead, the `state` attribute is
        # real-valued but can only store a discrete number of real value pairs, as they represent the left bound
        # of the discretized interval corresponding to the 1D state index self.s stored in discrete.DiscreteEnv.
        self.setState(self.get_index_from_state(observation))
        if done:
            # FIX a bug in the original Mountain Car environment about the reward when the car reaches the goal!
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
        # The interpretation of the state value with this definition is "the probability of reaching the terminal state
        # given we start at the state", whereas the original interpretation is "the expected number of steps to reach
        # the terminal state given we start at the state". Note that the random walk is a null recurrent chain, so the
        # latter should be infinite(!), but the former is a positive probability.
        reward = 1.0 + reward
        assert reward == done*1.0

        # Discretize position and velocity
        # These are integer values in [0, nx-1] and [0, nv-1] respectively for position and velocity
        # First we convert the `observation` received from the MountainCarEnv.step() method into a 2D continuous-valued
        # state (with infinite number of possible values, i.e. this is NOT the same as the `state` attribute defined
        # in the class, because the latter can only take finitely many different value pairs (x, v), namely those
        # corresponding to the left bound of the discretized intervals in each direction, x and v)
        # We do this conversion from observation to state_cont because the order in which position and velocity are
        # stored in each environment (our and MountainCarEnv from which we inherit) COULD be different
        # (however, normally we are storing it in the same order in the state array, so that we make our life easier).
        state_cont = np.array([np.nan, np.nan])
        state_cont[self.getPositionDimension()] = observation[0]  # Position is stored in observation's index 0 (see MountainCarEnv)
        state_cont[self.getVelocityDimension()] = observation[1]  # Position is stored in observation's index 1 (see MountainCarEnv)
        state_discrete = self.discretize(state_cont)

        if return_continuous_state:
            return state_cont, state_discrete, reward, done, info
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
        discretized interval to which x and v belong.
        """
        state_discrete = copy.deepcopy(np.array(state))
        dim_x, x = self.getPositionDimension(), state[self.getPositionDimension()]
        dim_v, v = self.getVelocityDimension(), state[self.getVelocityDimension()]

        # We now use np.digitize() instead of my basic.discretize() function because my original discretize didn't
        # always work properly (border problems, e.g. when giving exactly the left bound of an interval, it returned the
        # interval to the left of the right one (x = -1.02 when -1.02 was the left bound of the interval index 1).
        # Also, even if I have deprecated my discretize() function --and replaced with a discretize() method that
        # calls np.digitize() under the hood-- we still call np.digitize() here in order to avoid the generation of
        # the intervals each time we need to discretize a value. Instead we store these intervals as part of the object.
        # IMPORTANT: np.digitize() returns a value between 1 and the length of bins (second parameter),
        # NOT between 0 and len-1 as one would have expected (considering how Python defines indices)!!!!!!
        state_discrete[dim_x] = np.digitize(x, self.positions) - 1
        state_discrete[dim_v] = np.digitize(v, self.velocities) - 1
        assert 0 <= state_discrete[dim_x] <= self.nx - 1, "{}, {}".format(state_discrete[dim_x], self.nx-1)
        assert 0 <= state_discrete[dim_v] <= self.nv - 1, "{}, {}".format(state_discrete[dim_v], self.nv-1)
        #state_discrete[dim_x] = discretize(x, self.nx, self.xmin, self.xmax)
        #state_discrete[dim_v] = discretize(v, self.nv, self.vmin, self.vmax)

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
        The continuous-valued position and velocity are chosen as the LOWER LIMIT of the discretized interval
        corresponding to the given discrete state. It's important to keep the lower limit as the value of the continuous
        value because the goal is on the right of the mountain, meaning that the isTerminalState() method defined
        below does not contradict the `done` condition returned by the continuous-state MountainCarEnv environment,
        which is based on the position PRIOR to discretization.
        """
        #return np.array([self.xmin + state_discrete[self.getPositionDimension()] * self.dx, self.vmin + state_discrete[self.getVelocityDimension()] * self.dv])
        dim_x = self.getPositionDimension()
        dim_v = self.getVelocityDimension()
        return np.array([self.positions[state_discrete[dim_x]],
                         self.velocities[state_discrete[dim_v]]])

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

    def plot(self, statelist):
        "Generates a GIF showing the states through which the mountain car has been during learning"
        positions = [state[0] for state in statelist]
        def mountain_car_track(x):
            return np.sin(3*x)

        fig, ax = plt.subplots()
        x = np.linspace(-1.2, 0.6, 1000)
        y = mountain_car_track(x)
        ax.plot(x, y, 'k')

        ax.set_xlim(-1.2, 0.6)
        ax.set_ylim(-1.2, 1.2)  # Maybe change this to see properly

        cmap = plt.cm.get_cmap('coolwarm')
        norm = plt.Normalize(0, len(positions))

        points, = ax.plot([], [], 'o', markersize=5)

        def init():
            points.set_data([], [])
            return points,

        def update(frame):
            current_positions = positions[:frame + 1]
            current_y = mountain_car_track(np.array(current_positions))
            for i in range(frame + 1):
                ax.plot(current_positions[i], current_y[i], 'o', color=cmap(norm(i)), markersize=5)
            return points,

        ani = animation.FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True)

        # Save animation
        ani.save('./mountain_car.gif', writer='pillow', fps=10)

        print("GIF saved in: ./mountain_car.gif")

    def get_from_adjacent_states(self, state):
        " Returns the set of 2D-discrete-states adjacent to the given continuous-valued state (x, v) whose probability of transitioning to (x, v) is > 0"
        idx_state = self.get_index_from_state(state)
        adjacent_states_list = []
        actions_from_adjacent_states_list = []
        for x in self.positions:
            for v in self.velocities:
                idx_test_state = self.get_index_from_state((x, v))
                for a in np.arange(self.getNumActions()):
                    self.setState(idx_test_state)
                    idx_next_state = self.step(a)[0]
                    if idx_next_state == idx_state:
                        adjacent_states_list += [(x, v)]
                        actions_from_adjacent_states_list += [a - 1]    # This converts the actions 0, 1, 2 to -1, 0, 1 (accelerate left, do not accelerate, accelerate right)

        # Set the state of the environment at the state where it was before calling this method
        self.setState(idx_state)

        if self.debug:
            print(f"Adjacent states for state idx={idx_state}, {state}:")
            print([str(s) + " (a=" + str(a) + ")" for s, a in zip(adjacent_states_list, actions_from_adjacent_states_list)])

        return adjacent_states_list

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
        "Returns the 2D index from the 1D index of the state"
        return index_linear2multi(idx_state, self.shape, order='F')   # order='F' means sweep the 2D matrix by column

    def get_position(self, state):
        """
        Returns the position of a given 2D state

        Arguments:
        state: array, list or tuple
            The continued-value state (x, v) for which the position is wished.

        Return: float
        The continuous-valued position associated to the given state.
        """
        return state[self.getPositionDimension()]

    def get_velocity(self, state):
        """
        Returns the velocity of a given 2D state

        Arguments:
        state: array, list or tuple
            The continued-value state (x, v) for which the velocity is wished.

        Return: float
        The continuous-valued velocity associated to the given state.
        """
        return state[self.getVelocityDimension()]

    def get_positions(self):
        "Returns the lower limit of all discretized intervals of the car's position"
        return self.positions

    def get_velocities(self):
        "Returns the lower limit of all discretized intervals of the car's velocity"
        return self.velocities

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

    def getState(self):
        """
        Returns the state index associated to the current environment observation

        Return: int
        Index between 0 and nS-1, where nS is the number of states in the environment.
        """
        idx_state = super().getState()  # This calls the getState() method in the discrete.DiscreteEnv class (since the MountainCarEnv class does not have any getState() method defined)
        assert self.get_index_from_state(self.state) == idx_state, \
            "The 2D continuous-valued state stored in the Mountain Car object ({}) falls in the cell represented by the 1D state index stored in the discrete.DiscreteEnv object ({})." \
            .format(self.state, idx_state, self.get_index_from_state(self.state))
            # Note that we assert that the 2D continuous-valued state converted to 1D state index match the current 1D state index stored in the environment
            # and NOT the other way round because an INFINITE number of 2D continuous-valued states are mapped to the same 1D state index, whereas only ONE
            # 2D continuous-valued state is mapped to the 1D state index conversion to 2D, namely the 2D continuous-valued state that is exactly at one of the
            # grid points defined by the discretization of the 2D state landscape (position, velocity) that is linearly indexed (column first) by the 1D state index.
            # Note that the assertion that is based on the 1D to 2D conversion fails soon after the environment is reset by the self.reset() method defined above
            # become in such reset() method, the 2D state is first chosen at random and a 1D state index mapped to it: if we converted this 1D state index back to
            # a 2D state, the result would NOT be the 2D state selected at random in the 2D continuous-valued space!
        return idx_state

    def getReward(self, s):
        """
        Returns the reward received when visiting a state

        It returns always 0, as the method has been defined only to comply with the processes working with environments that require the definition of this method,
        such as value function learners inheriting from discrete.Learner.
        """
        return 0.0

    def isTerminalState(self, idx_state: int):
        state = self.get_state_from_index(idx_state)
        x = self.get_position(state)
        #print("Check terminal state: {} (dx={})".format(state, self.dx))
        return x >= self.goal_position

    #--- Setters
    def setState(self, idx_state):
        """
        Sets the state of the environment, both the state index and
        the internal 2D `state` (the discretized continuous-valued state) representing the discretized position and velocity

        Both the state index in the super class discrete.DiscreteEnv (which is superclass of the super class on which
        this class derives, i.e. EnvironmentDiscrete) is set and the internal state (self.state) containing the (x, v) duple
        with the car's position and velocity.

        Arguments:
        idx_state: int
            State index.
        """
        # Set the state (1D index) in the EnvironmentDiscrete environment
        super().setState(idx_state)
        self.state = self.get_state_from_index(idx_state)

    def getNumStates(self):
        return self.nS

    def getNumActions(self):
        return self.nA
