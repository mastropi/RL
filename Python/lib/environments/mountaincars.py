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

from typing import Union
import numpy as np

from matplotlib import animation, pyplot as plt, cm
from matplotlib.ticker import MaxNLocator

from gym.envs.toy_text.discrete import categorical_sample
from gym.envs.classic_control import MountainCarEnv

from Python.lib.utils.basic import index_linear2multi, index_multi2linear
from Python.lib.environments import EnvironmentDiscrete

from Python.lib.utils.basic import is_integer



class MountainCarDiscrete(MountainCarEnv, EnvironmentDiscrete):
    """
    Discrete-action Mountain Car environment defining the methods used by my learners, such as LeaTDLambda

    Discrete here means: discrete actions (-1, 0, +1) for accelerate left, do not accelerate, and accelerate right, respectively.
    The states can be either continuous or discrete, depending on how the object is constructed.

    The meaning of continuous state and discrete state lies on how the car's dynamics are computed, whether using continuous-valued
    position and velocity, or using discrete-valued position and velocity, i.e. position and velocity values that have been discretized
    into a 2D grid.

    For the discrete-state case, some tuning on the force, gravity and max_speed is needed in order to make the problem interesting,
    e.g. in order to guarantee transitioning to other states when an action is applied for every (or most of) the states.
    The discrete-state representation has the advantage of allowing a tabular representation of the state and action value functions.

    In the continuous-state case, there is still a discretization carried out on the position and velocity because some of the processes
    we run require such a discretization, e.g. the definition of the absorption set A in the Fleming-Viot learning process, which requires
    that the state space be finite.

    Following are the main differences between the continuous-state case and the discrete state-case:
    - continuous-state case: (the default)
        - the constructor receives the number of points into which to discretize the position (nx) and the velocity (nv).
        - a factor of the maximum speed parameter (factor_max_speed) is used to adjust the max_speed parameter of the MountainCarEnv super class.
    - discrete-state case:
        - the constructor receives ONLY the number of points into which to discretize the *velocity* (nv),
    whose values are in the range [-max_speed, +max_speed].
        The discretization grid for the position is automatically computed from the velocity discretization grid so as to:
            - guarantee that the goal on the right is reached
            - guarantee that the car leaves (at least almost always) every possible cell (discrete state) at the smallest discretized non-zero velocity.
        The latter condition is applied to the interval with the smallest help from gravity, namely the valley,
        and the condition to leave the interval is established when the original velocity is 0.0, namely when the new velocity is equal to `force`.
        Note that the smaller the interval of the discretized velocity space (dv), the larger the number of discretized intervals for
        position. This implies that, the larger the number of velocity intervals, the larger the number of position intervals as well.
        - The max_speed value is defined by multiplying the original `max_speed` parameter in the MountainCarEnv super class
        (normally equal to 0.07) by `0.25*factor_for_force_and_gravity`, which means that any value passed to the `factor_max_speed` parameter is ignored.
        This adjustment has been found useful to make the dynamics of the discrete-state environment interesting, as described above
        for any adjustement of the car's force and gravity. It implies for instance that the car is able to go faster by an larger force applied
        than the original one and thus be able to transition to other discrete states when an action is taken.
        - Car's dynamics: when stepping to the next state using the step() function of the super class MountainCarEnv, the continuous (x, v)
        values assigned to each discrete state are the left bounds of the x- and v-interval to which the state belongs.
        Ex: if there are 6 grid points for x and 5 grid points for v, i.e.:
            xd in [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8]  (dx = 0.4)
            vd in [-0.07, -0.035, 0.00, 0.035, 0.07] (dv = 0.035)
        then each possible discrete-valued state is represented by all possible combinations of the 6x5 intervals making up 30 possible states
        to form the tuple (xd, vd).
        If the continuous-valued state in the MountainCarEnv super class is (x, v) = (0.43, 0.068),
        its corresponding discrete-valued state is (xd, vd) = (0.4, 0.035).
        This discrete-valued state is the state input to the step() method in this class that is responsible for moving the car
        after an action is taken (0, 1 or 2, which correspond to a discrete acceleration value, -1, 0, 1), by calling the step() method of the super class
        on the discrete-valued state (xd, vd). When the new continuous-valued position and velocity are obtained corresponding
        to the next continuous-valued state, the next state is discretized into one of the (xd, vd) combinations,
        using the LOWER LEFT limit of the 2D cell where the (x, v) point falls.
        This means that the Mountain Car dynamics, although based on continuous-valued equations, are always based on its discrete-valued position and velocity,
        as opposed to its continuous-valued position and velocity.
        Note also that the discrete-valued state with largest possible values for position and velocity,
        (xd, vd) = (0.8, 0.07) represents the cell with x values in [0.8, 1.2) and v values in [0.07, 0.105).

    In both continuous-state and discrete-state cases, the original `force` and `gravity` values defined in the MountainCarEnv super class
    are multiplied by constructor parameter `factor_for_force_and_gravity` which can be used to control how easy it is for the mountain car to reach the goal.
    In the discrete-state case, this factor also helps reduce the number of discretized intervals of the position that guarantees
    transitioning to another discrete state at (almost) every step, which otherwise could make the learning process very slow (because of a too large
    number of states in the system to represent by tabular value functions).

    For example, the above logic of defining the number of discretized intervals for position in the discrete-state case would give about 1000
    intervals when we use the original `force` and `gravity` values defined in the super class MountainCarEnv, which are respectively equal to 0.001 and 0.0025.
    With the following values of factor_for_force_and_gravity, we get:
    - 20 => "new force" = 0.02, "new gravity" = 0.05 => 107 discretized position intervals
    - 100 => "new force" = 0.1, "new gravity" = 0.25 => 21 discretized position intervals

    Arguments:
    nv: int
        Number of discretization intervals for velocity.
        An odd number is enforced by adding 1 when even, so that v = 0.0 is always part of the velocity grid.

    nx: (opt) int
        Number of discretization intervals for velocity.
        It cannot be None when `discrete_state = False` (the default).
        default: None

    factor_for_force_and_gravity: (opt) positive float
        Multiplier of the original `force` and `gravity` values defined in the MountainCarEnv super class
        affecting the number of discretized intervals for the car's position, as described above.
        Increase this value to decrease the number of discrete positions in the environment.
        20 => 107 discrete positions
        100 => 21 discrete positions
        default: 20, which gives 107 discretized position intervals

    factor_force: (opt) positive float
        Factor to further multiply the force by, in addition to the already applied `factor_for_force_and_gravity`.
        This would typically be used to increase the car's force in order to control how easily the car is able to reach the goal by overcoming gravity.
        Use a value larger than 1 to make reaching the goal easier and smaller than 1 to make reaching the goal harder
        than in the original mountain car environment.
        default: 1.0

    factor_max_speed: (opt) positive float
        Factor to multiply the `max_speed` parameter of the MountainCarEnv super class by, in order to limit the maximum velocity further or
        in order to allow a larger maximum velocity when the car's force has been increased by a positive factor, or simply to allow the car to move faster
        under the original car's force value to perhaps make it reach the goal more easily.
        This value is ONLY used for the continuous-state environment, i.e. when discrete_state = False. Otherwise, for the discrete-state environment,
        the `max_speed` parameter is multiplied by `0.25*factor_for_force_and_gravity` because this factor has proven to work well in the discrete-state
        dynamics.
        default: 1.0

    discrete_state: (opt) bool
        Whether the state space used to run the environment dynamics on (i.e. the logic used to compute the next environment state with self.step()
        when an action is taken) is based on discrete-valued or continuous-valued states.
        default: False, meaning that the mountain car environment is by default a continuous-state environment

    shape_display_names: (opt) tuple
        2D tuple with the name of concept plotted in each axis of the 2D representation of the environment states.
        Possible values are: ("velocity", "position"), ("position", "velocity")
        default: ("velocity", "position") which means that the car's position is plotted on the horizontal axis of 2D plots by the plotting functions defined in the class

    seed_reset: (opt) int
        Seed to use when choosing the start state at environment reset.
        default: None
    """

    # NAMING CONVENTIONS FOR ATTRIBUTES AND VARIABLES CONTAINING STATE INFORMATION:
    # In this class we call:
    # - idx_state: (int) 1D index representing the state, used in the discrete.DiscreteEnv environment from which the super
    # class EnvironmentDiscrete derives (stored as attribute `s`), and is thus an attribute of the class.
    # The way this 1D index is computed from the state depends on the shape used to display the 2D environment, either positions x velocities or velocities x positions,
    # as the conversion from `state` to `idx_state` is done by first converting `state` to `idx_state_2d` and then converting `idx_state_2d` to `idx_state` using the
    # C-like order for conversions from multidimensional indices to unidimensional indices. See method get_index_from_state() and the documentation for reshape_from_2d_to_1d()
    # for more conceptual details.
    # - idx_state_2d: (tuple) 2D index to access the state in the 2D DISPLAY of the environment states corresponding to `idx_state`.
    # This concept is an internal representation of which the user doesn't have to know about. In fact, it is NOT stored as an attribute.
    # Its value can be retrieved from `idx_state` using the get_index_2d_from_index()` method.
    # Ex: (3, 5)
    # - state: (tuple) 2D continuous-valued state as a tuple (x, v) defined in the super class MountainCarEnv. It is an attribute of the super class and thus of this class.
    # Ex: (0.43, 0.091), which in the example given in the class documentation, it falls inside the 2D cell [0.4, 0.8) x [0.07, 0.105) of the discretized state space
    # - state_discrete: (tuple) 2D discretized continuous-valued state as a tuple (xd, vd) = (position, velocity) representing the LOWER LIMIT of each discretized cell
    # of the continuous state space. It is NOT an attribute of the class and its value can be retrieved from `idx_state` by calling the `get_state_discrete_from_index()` method.
    # It is also possible to get this discretized-value state from the state by calling `get_state_discrete_from_state()` method.
    # Ex: (0.4, 0.07), which in the example given in the class documentation, it represents the 2D cell [0.4, 0.8) x [0.07, 0.105) of the discretized state space
    # - state_simulation: (int) the value representing the state during simulations. This coincides with `idx_state` because of the discrete-state nature of this environment.
    # Here, it is only used conceptually.
    #
    # HOWEVER, despite the above naming convention, it is currently a pity that the methods getState() and setState()
    # return the state 1D INDEX (and NOT the 2D continuous state as a tuple, as one would expect from their names)
    # because these methods override those defined in the super class EnvironmentDiscrete.
    # TODO: (2022/06/06) Rename the methods in EnvironmentDiscrete to get and set the state to getStateIndex() and setStateIndex()
    # (so that when we talk about `state` we think of the human-understandable state (e.g. (x, v) in the Mountain Car environment)
    # Note that the super class EnvironmentDiscrete should define methods called getState() and setState() which should raise NotImplementedError
    # to indicate that those methods cannot be implemented in such class (EnvironmentDiscrete) because we need information
    # about the environment itself, namely about how the index state self.s defined in discrete.DiscreteEnv translates
    # into the human-understandable state.
    def __init__(self, nv, nx=None, factor_for_force_and_gravity=20, factor_force=1.0, factor_max_speed=1.0, discrete_state=False, shape_display_names=("velocity", "position"), seed_reset=None):
        MountainCarEnv.__init__(self)
        self.setSeed(seed_reset)

        # Store whether the state of this environment is DISCRETE-valued or CONTINUOUS-valued
        # i.e. whether the result of taking an action on the car is computed on the discrete-valued or on the continuous-valued state
        self.state_is_continuous = not discrete_state

        # Adjust the the force, the gravity and the max speed based on the given factor
        # This is done to reduce the number of discrete positions (and thus make the state space smaller)
        # Note that max_speed is defined in the original mountain car environment to clip the velocity which is a direct function of "force - gravity" (because the time step is t=1)
        factor_max_speed = 0.25*factor_for_force_and_gravity if discrete_state else factor_max_speed
        self.max_speed = factor_max_speed * self.max_speed
        self.force = factor_force * factor_for_force_and_gravity * self.force
        self.gravity = factor_for_force_and_gravity * self.gravity

        # Minimum and maximum values for position and velocity
        # Note: In the documentation of the source code in GitHub (mentioned above) it says that:
        # -1.2 <= x <= 0.5 (which is the goal position): note that we do not need to allow x to go past the goal position because we discretize the value of x using the left bound of each interval
        # -0.07 <= v <= 0.07
        # and these are ranges for force = 0.001 and gravity = 0.0025.
        # TODO: (2024/08/31) Homogenize the min and max values of position so that the max position is always self.max_position (NOT self.goal_position)
        if self.state_is_continuous:
            self.xmin, self.xmax = self.min_position, self.max_position
        else:
            self.xmin, self.xmax = self.min_position, self.goal_position
        self.vmin, self.vmax = -self.max_speed, self.max_speed

        #-- Discretization of velocity
        # Number of discrete intervals to use
        self.nv = nv
        # Make sure that nv is odd so that v=0.0 is part of the grid (required when defining the grid for positions below)
        # and we get symmetric values for v.
        self.nv = self.nv + 1 if self.nv % 2 == 0 else self.nv
        self.dv = (self.vmax - self.vmin) / (self.nv - 1)
        # Discrete velocity values (lower limit of each grid interval)
        self.velocities = np.linspace(self.vmin, self.vmax, self.nv, endpoint=True)
        # Enforce the value of 0.0 velocity for the assertion below
        for i, v in enumerate(self.velocities):
            if np.isclose(v, 0.0):
                self.velocities[i] = 0.0

        #-- Discretization of position
        if self.state_is_continuous:
            # Number of discrete intervals to use
            if nx is None:
                raise ValueError("Parameter `nx` defining the number of discretization intervals of the position cannot be None when `discrete_state` is False."
                                 "\nThis piece of information is needed to deal with processes that require a finite number of states, such as the absorption set A of the Fleming-Viot process.")
            self.nx = nx
            self.dx = (self.xmax - self.xmin) / (self.nx - 1)
            # Discrete position values (lower limit of each grid interval)
            self.positions = np.linspace(self.xmin, self.xmax, self.nx, endpoint=True)
        else:
            # The state is discrete
            # => It is important to make sure that:
            # a) The car can get out of the interval with the minimum possible force, which happens at the valley of the path,
            # namely when x = pi/6 ~= -0.5, where gravity = 0, therefore the force when accelerating is just self.force
            # In this case, we should consider the worst case scenario which is when the car is at velocity 0, in which case
            # the new velocity will be simply force, and this new velocity should be enough to take the car out of
            # that interval, namely, since delta(x) = velocity*1 and velocity = force, we should have:
            #   dx > force
            # In other positions, the new velocity will have also the help from gravity, therefore we are in business if
            # we satisfy the above condition for dx, i.e. the mesh grid density for the positions.
            # Note that gravity is largest at two positions, cos(3*x) = cos(-pi) and cos(0),
            # which happens when x = -pi/3 ~ -1.05 and x = 0.0
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
            # number of points and endpoint=True (the default))
            # Note that the number of points returned by np.linspace() is the number of points specified as third argument,
            # regardless of the value of endpoint.
            velocity_achieved_from_zero_speed_at_zero_gravity = self.force
            position_max_leftmost = self.xmin + 0.8*velocity_achieved_from_zero_speed_at_zero_gravity
                ## NOTE: (2024/08/15) I don't recall what the 0.8 factor means... HOWEVER, if factor_for_force_and_gravity = 100, thanks to the 0.8, the value of dx becomes 0.08,
                ## making the discrete position closest to the rightmost position at x = 0.6 be 0.52, i.e. larger than the goal at 0.5, thus making the goal have at least one point.
            position_min_rightmost = self.goal_position - self.vmax  # This is not used but computed for informational purposes if needed
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

        # A few checks: the above calculations assume the following
        assert self.xmax >= self.goal_position if self.state_is_continuous else self.xmax == self.goal_position
        assert 0.0 in self.velocities
        assert len(self.positions) == self.nx
        assert len(self.velocities) == self.nv
        if self.state_is_continuous:
            assert np.isclose(self.positions[1] - self.positions[0], self.dx, atol=1E-6), "v0={:.3f}, v1={:.3f}, v1-v0={:.6f}, dv={:.6f}".format(self.positions[0], self.positions[1], self.positions[1] - self.positions[0], self.dx)
        assert np.isclose(self.velocities[1] - self.velocities[0], self.dv,atol=1E-6), "v0={:.3f}, v1={:.3f}, v1-v0={:.6f}, dv={:.6f}".format(self.velocities[0], self.velocities[1], self.velocities[1] - self.velocities[0], self.dv)

        # Shape of the environment for the outside world
        # i.e. number of intervals in the position values x number of intervals in the velocity values obtained by discretization
        # that is, this shape respects the order in which position and velocity are stored in the continuous-valued state of the super class.
        # It differs from the display shape coming below, which furthermore is a private attribute, as the user doesn't need to know about it, in principle
        self.shape = (self.nx, self.nv)

        # Shape of the environment for display purposes, which in turn corresponds to the 2D internal representation of the state
        # (of which the user doesn't have to know; the user should know only of the actual state (or discretized continuous-valued state --e.g. (0.4, 0.07))
        # and of the 1D state index. In fact these two state representations may be used in simulations).
        # Note that the shape defines HOW the 2D states (x, v) are displayed as a matrix, whether positions represent rows and velocities columns or the other way round.
        # The shape has NOTHING TO DO with the order of the tuple representing the state of the mountain car which is ALWAYS (x, v).
        if shape_display_names not in [("velocity", "position"), ("position", "velocity")]:
            raise ValueError(f"""Parameter `shape_display_names` must be either equal to ("velocity", "position") or ("position", "velocity"): {shape_display_names}""")
        self._shape_display_names = shape_display_names
        self._shape_display = (self.nx, self.nv) if self._shape_display_names == ("position", "velocity") else (self.nv, self.nx)

        # Number of states and actions
        self.nS = self.nx * self.nv
        self.nA = self.action_space.n
        self.all_states = list(np.arange(self.nS))
        # In case we need to store the 2D states, we can use this
        #self.all_states_2d = [(x, v) for x in self.positions for v in self.velocities]

        # In case we need to have all possible (x, v) indices indexing the discrete position and velocity
        # We can go from 2D indices to 1D indices and the other way round using respectively the following functions
        # defined in the `basic` module:
        #   index_multi2linear()
        #   index_linear2multi()
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

        # Define other pieces of information required by processes that rely on the information stored in the EnvironmentDiscrete super class,
        # most importantly the reward received by the terminal states, which is used when calling the EnvironmentDiscrete.getReward() method, used extensively
        # in processes involving Fleming-Viot!!
        # Reward landscape
        self.reward_at_goal = 1.0  # Set this to 1.0 for a sparse reward landscape and zero for a reward of -1 at every step
        if self.state_is_continuous:
            # Define the reward received by the terminal states
            # This is used when computing the expected reward on a FINITE state space using sum{x} p(x)r(x), where r(x) is the reward received by the DISCRETE state x,
            # for instance when the compute_expected_reward() function in computing.py is called to compute the expected reward where the environment's method
            # getRewardsDict() retrieving the reward stored in the self.rewards attribute is used as r(x) value.
            # Below we define self.rewards which assigns a non-zero reward to the terminal states.
            self.terminal_states = self.get_indices_terminal_states()
            # Do of the terminal states
            for s in self.terminal_states:
                # Check that some states in the discrete states classified as terminal states contain some continuous-valued terminal states (i.e. having position x >= self.goal_position = 0.5)
                assert self.get_state_from_index(s)[0] + self.dx >= self.goal_position, \
                    f"SOME of the states in the set of terminal states must have position >= goal_position (= {self.goal_position}).\nCondition failed for ({s, self.get_state_from_index(s)})."
            # Check that none of the states in discrete states NOT classified as terminal states contain continuous-valued terminal states
            assert len(
                [self.get_state_from_index(s) for s in self.all_states if s not in self.terminal_states and self.get_state_from_index(s)[0] + self.dx >= self.goal_position]) == 0, \
                f"NONE of the states in the set of non-terminal states have position >= goal_position (= {self.goal_position})"

            # Rewards dictionary
            # Note that a non-zero reward between 0 and self.reward_at_goal is assigned to PARTIAL discrete terminal states, i.e. to discrete terminal states that contain both
            # states that are NOT continuous-valued terminal states and states that are continuous-valued terminal states.
            self.rewards = dict([(s, self.reward_at_goal * min(1.0, (self.getPosition(self.get_state_from_index(s)) + self.dx - self.goal_position) / self.dx)) for s in self.terminal_states])
        else:
            self.non_terminal_states = set(self.get_indices_non_terminal_states())
            self.terminal_states = set(self.getAllStates()).difference( set( self.non_terminal_states ) )
            self.rewards = dict([(s, self.reward_at_goal) for s in self.terminal_states])

        # Initialize the other class from which this class inherits, so that we can use its attributes (e.g. rewards, etc)
        EnvironmentDiscrete.__init__(self, self.nS, self.nA, None, self.isd, dim=self.dim, rewards=self.rewards, terminal_states=self.terminal_states)

        # Reset the environment
        # The following reset() call defines both:
        # - the 2D continuous-valued `state` = (x, v) containing the car's position and velocity of the MountainCarEnv environment
        # - the 1D state `s` of the discrete.DiscreteEnv of the gym library, from which EnvironmentDiscrete inherits.
        # The 1D state `s` is only important in the discrete-state case, because this is the attribute defining the state of the environment at every simulation step,
        # and clearly these two state attributes need to be ALIGNED.
        # Note that the MountainCarEnv super class environment does not have any getter or setter to access `state`.
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
        Resets the environment state, including the continuous-valued state of the super class and
        the 1D state index representing the state in the EnvironmentDiscrete super class (`s`)

        The reset can optionally be done using the initial state distribution (of the 1D state indices) given in parameter `isd`,
        otherwise the reset step of the MountainCarEnv environment is used.
        """
        if isd is not None:    # isd = Initial State Distribution to use, if any desired
            # Select the 1D index representing the 2D state of the mountain car
            # The reset of the state is done using the EnvironmentDiscrete reset process, which actually uses the gym.Env reset process
            # which selects a state following the Initial State Distribution information stored in the environment.
            print("ISD: {} (which={})".format(isd, np.where(isd==1.0)))
            idx_state = categorical_sample(isd, self.np_random)

            # Set the 2D continuous-valued state (state of the MountainCarEnv environment) and its 1D index version (state of the EnvironmentDiscrete environment)
            self.setState(idx_state)
        else:
            # Set an initial random state using the MountainCarEnv reset process
            self.seed(self.seed_reset)  # This method is defined in MountainCarEnv included in gym/envs/classic_control
            # Set the continuous-valued (x, v) state of the Mountain Car super class environment
            # Notes:
            # - This super().reset() calls the reset() method of the MountainCarEnv environment because this is the first super class
            # appearing in the list of super classes in the definition of this class.
            # - The state returned by the super class is converted to a tuple because this is how it is stored by their step() method
            # although their reset() method returns an array!
            state = tuple(super().reset())
            if self.state_is_continuous:
                # Set the state of the MountainCarEnv super class because we need it to be a tuple (instead of an array which is the type used by the MoutainCarEnv.reset() method!)
                # (storing a tuple as the state as opposed to an array is necessary to be able to use it as key of the rewards dictionary
                # --o.w. we get the error "unhashable type: 'numpy.ndarray' when trying to retrieve the dictionary key when the key is given as an array...)
                self.state = state
            else:
                # Set the 1D index associated to the continuous-valued (x, v) state just set (this 1D index is computed using the position and velocity discretization process)
                idx_state = self.get_index_from_state(state)
                self.setState(idx_state)

        # We need to return a state because the environment reset() performed by simulators assign the returned state from the reset() method to `next_state`
        # Otherwise, next_state will be None!
        if self.state_is_continuous:
            return self.state
        else:
            # Return the state represented by the index
            return self.getStateFromIndex(idx_state)

    def step(self, action, simulation=True):
        """
        Performs one step in the environment for the given action

        The next state is calculated using the continuous-valued state as input for the Physics dynamics defined in the MountainCarEnv super class
        if self.state_is_continuouse = True, or using the discrete-valued state if self.state_is_continuous = False, instead.

        Arguments:
        action: int
            Action taken by the agent on the environment.
            Valid values are 0 (accelerate left), 1 (do not accelerate), 2 (accelerate right).

        Return: tuple
        A 4-tuple with the following elements:
        - state: int or tuple; the next state of the environment after taking the given action,
        which is int when self.state_is_continuous = False and simulation = False, or tuple o.w. containing the continuous-valued next state.
        - reward: the reward received after visiting `state`.
        - done: whether the car reached the goal (based on the continuous-valued state).
        - info: additional info returned by the step() method of the MountainCarEnv super class environment.
        """
        # Note: the following step() method sets the self.state attribute of the MountainCarEnv super class as a tuple, but `observation` is an array
        observation, reward, done, info = super().step(action)
        assert np.allclose(self.state, observation), "The state of the environment must have been set when calling step(action)"

        if not self.state_is_continuous:
            # We now change the 1D state index in discrete.DiscreteEnv AND the continuous-valued state of the MountainCarEnv super class environment
            # so that it represents the discrete-valued state (xd, vd).
            self.setState(self.get_index_from_state(self.state))

            # IMPORTANT: Check DONE in the DISCRETE state space!!!
            # This allows us to be "independent" of the way we discretize the continuous-valued state, which could be ROUNDing instead of TRUNCating (see self.discretize()).
            # If TRUNCating, this step is not necessary, but when ROUNDing it is necessary because in that case the discrete state may be at the goal already,
            # even if the continuous state is not (because of the "ROUNDING" step).
            idx_state = self.getState()
            done = idx_state in self.terminal_states

        # Adapt the reward structure defined in the original Mountain Car environment to our reward structure in two ways:
        # - The reward structure of the original environment is designed as the reward received
        # after LEAVING the state (i.e. after taking an action, because they want to count the number of actions taken
        # and minimize that number as optimization objective), whereas our reward structure is that the reward
        # is received when VISITING the state.
        # - The reward landscape of the original environment is to assign -1 to each step performed until the goal is reached.
        # Our usual reward landscape is to assign +1 when reaching the goal and 0 elsewhere, but we can set the attribute self.reward_at_goal = 0.0
        # and we recover the reward landscape of the original environment.
        if done:
            reward = self.reward_at_goal
            #print("\nStep: action = {} -> observation = {}, reward = {}, done = {}".format(action, observation, reward, done))
        else:
            reward = self.reward_at_goal + reward   # This is ZERO reward when reward_at_goal = 1.0 (since the reward yielded by the original environment is -1 at every step

        if not self.state_is_continuous and simulation:
            return idx_state, reward, done, info
        else:
            return self.state, reward, done, info

    def discretize(self, state):
        """
        Discretizes into 2D indices the given continuous-valued state using the discretization parameters defined in the object (e.g. dx, dv, etc.)
        and self._shape_display_names, which defines the layout of position and velocity in the 2D environment layout used to visually represent the environment.

        Arguments:
        state: tuple
            The continuous-valued state = (x, v) to discretize.

        Return: numpy.ndarray
        Tuple containing the 2D indices representing the cell in the 2D environment layout where the given `state` falls.
        Whether the 2D indices represent first the position index and then the velocity index or the other way round depends on the 2D layout
        used to represent the environment, defined by self._shape_display_names.
        """
        idx_state_2d = np.array((-1, -1))
        # The state is ALWAYS (x, v) REGARDLESS of the value of self._shape_display_names
        x, v = state

        # We now use np.digitize() instead of my basic.discretize() function because my original discretize didn't
        # always work properly (border problems, e.g. when giving exactly the left bound of an interval, it returned the
        # interval to the left of the right one (x = -1.02 when -1.02 was the left bound of the interval index 1).
        # Also, even if I have deprecated my discretize() function --and replaced with a discretize() method that
        # calls np.digitize() under the hood-- we still call np.digitize() here in order to avoid the generation of
        # the intervals each time we need to discretize a value. Instead we store these intervals as part of the object.
        # IMPORTANT: np.digitize() returns a value between 1 and the length of bins (second parameter),
        # NOT between 0 and len-1 as one would have expected (considering how Python defines indices)!!!!!!

        # Use this if we want to ROUND the continuous-valued position and velocity
        #idx_state_2d[self.getPositionDimension()] = max(0, int(np.digitize(x + self.dx/2, self.positions) - 1))
        #idx_state_2d[self.getVelocityDimension()] = max(0, int(np.digitize(v + self.dv/2, self.velocities) - 1))

        # Use this if we want to TRUNCATE the continuous-valued position and velocity
        # Note that we bound any values going out of bound from the left to the leftmost position and velocity values
        idx_state_2d[self.getPositionDimension()] = max(0, int(np.digitize(x, self.positions) - 1))
        idx_state_2d[self.getVelocityDimension()] = max(0, int(np.digitize(v, self.velocities) - 1))

        assert 0 <= idx_state_2d[self.getPositionDimension()] <= self.nx - 1, "{}, {}".format(idx_state_2d[self.getPositionDimension()], self.nx-1)
        assert 0 <= idx_state_2d[self.getVelocityDimension()] <= self.nv - 1, "{}, {}".format(idx_state_2d[self.getVelocityDimension()], self.nv-1)

        return tuple(idx_state_2d)

    def undiscretize(self, idx_state_2d):
        """
        Returns the discrete-valued state (xd, vd) associated to the given 2D index of the 2D environment layout (defined by self._shape_display_names)

        Arguments:
        idx_state_2d: tuple, list or array
            The 2D index of the state to convert to a discrete-valued state, with the position index first or the velocity index first
            depending on the value of self._shape_display_names which defines precisely that order as either
            ("position", "velocity") or ("velocity", "position") and whose value is returned by self.getShapeDisplayNames().

        Return: tuple
        Tuple (xd, vd) containing the discrete-valued position at the first element and the discrete-valued velocity at the second element.
        These values represent the undiscretized state defined as the LOWER LIMIT of the discretized cell represented by the given 2D index
        discretizing the continuous state space.
        """
        dim_x = self.getPositionDimension()
        dim_v = self.getVelocityDimension()
        state_discrete = tuple([self.positions[idx_state_2d[dim_x]],
                                self.velocities[idx_state_2d[dim_v]]])
        return state_discrete

    def reshape_from_2d_to_1d(self, values_2d):
        """
        Converts the given 2D representation of the values to a 1D representation using C-like ordering of the 2D shape

        This means that the 1D representation of the values will map the 2D cells of the 2D environment layout in the order in which
        the quantity (either position or velocity) placed along the HORIZONTAL direction (columns) varies first.
        This depends on the value of self._shape_display_names. More precisely:
        - when self._shape_display_names = ("position", "velocity") => the 1D representation of the state indexes the state by varying first the velocity,
        as velocity is laid out along the horizontal direction (columns).
        - when self._shape_display_names = ("velocity", "position") => the 1D representation of the state indexes the state by varying first the position,
        as position is laid out along the horizontal direction (columns).

        Arguments:
        values_2d: 2D numpy array
            2D numpy array with as many elements as the number of discrete states in the environment, containing values to reshape to 1D.
            Normally the shape of `values_2d` corresponds to the same 2D shape used to represent the environment states for display purposes,
            which is defined by the value of the self._shape_display_names attribute that can be retrieved with self.getShapeDisplayNames().

        Return: 1D numpy array
        1D numpy array containing the input values originally laid out in a 2D shape corresponding to the 2D environment layout of the discrete states,
        as explained in the description above.
        """
        return np.squeeze( values_2d.reshape(1, np.prod(values_2d.shape)) )

    def reshape_from_1d_to_2d(self, values_1d):
        "Converts a 1D representation of the values to a 2D representation using C-like ordering. See details in documentation for self.reshape_from_2d_to_1d()"
        return values_1d.reshape(self._shape_display)

    #-----------
    def get_index_from_state(self, state: tuple):
        "Returns the 1D state index corresponding to the given continuous-valued state"
        idx_state_2d = self.discretize(state)
        idx_state = self.get_index_from_index_2d(idx_state_2d)
        return idx_state

    def get_state_from_index(self, idx_state: int):
        "Returns the continuous-valued state from the given 1D state index. The conversion to continuous values is done by the undiscretize() method."
        idx_state_2d = self.get_index_2d_from_index(idx_state)
        state = self.undiscretize(idx_state_2d)
        return state

    def get_random_state_from_index(self, idx_state: int):
        "Returns a randomly chosen continuous-valued state from the 2D cell referenced by the given 1D state index"
        state_discrete = self.get_state_discrete_from_index(idx_state)

        # Choose a random position and a random velocity in the interval in order to make the continuous state
        position_lower = state_discrete[0]
        position_upper = position_lower + self.dx
        velocity_lower = state_discrete[1]
        velocity_upper = velocity_lower + self.dv
        # Note that we use self.np_random() and NOT np.random() because self.np_random contains the random number generator whose seed is the seed set as self.seed_reset by the reset() method
        state = (self.np_random.uniform(position_lower, position_upper), self.np_random.uniform(velocity_lower, velocity_upper))

        return state
    #-----------

    #-----------
    def get_index_from_index_2d(self, idx_state_2d: tuple):
        """
        Returns the 1D state index from the 2D state indices

        Arguments:
        idx_state_2d: tuple (but also list and numpy array are valid)
            2D indices of the state.

        Return: int
        1D index of the state in the 1D representation of the 2D shape indexing the states.
        What each 1D index represents depends on the value of self._shape_display_names which can be either ("position", "velocity") or ("velocity", "position").
        If ("position", "velocity"), all the different positions will come first in the 1D representation of the 2D state space.
        """
        return index_multi2linear(idx_state_2d, self._shape_display)

    def get_index_2d_from_index(self, idx_state: int):
        "Returns the 2D index from the 1D index of the state"
        return index_linear2multi(idx_state, self._shape_display)
    #-----------

    #-----------
    def get_state_from_index_2d(self, idx_state_2d: tuple):
        """
        Returns the discrete-valued state (xd, vd) from the given 2D state index.
        This is an alias for the undiscretize() method.
        """
        return self.undiscretize(idx_state_2d)

    def get_index_2d_from_state(self, state: tuple):
        idx_state_2d = self.discretize(state)
        return idx_state_2d
    #-----------

    #-----------
    def get_state_discrete_from_state(self, state: tuple):
        """
        Returns the 2D discrete-valued state (xd, vd) from the given 2D continuous-valued state

        Arguments:
        state: tuple (but also list and numpy array are valid)
            Continuous-valued state (x, v) to convert to a discrete-valued state (xd, vd).

        Return: tuple
        Duple representing the discrete-valued state associated to `state` in the form (xd, vd).
        """
        idx_state = self.get_index_from_state(state)
        state_discrete = self.get_state_discrete_from_index(idx_state)
        return state_discrete

    def get_state_discrete_from_index(self, idx_state: int):
        """
        Returns the discrete-valued state (xd, vd) from the given 1D state index.
        The conversion to continuous values is done by the undiscretize() method.

        Arguments:
        idx_state: int
            1D index representing the discretized state.

        Return: tuple
        Duple representing the discrete-valued state associated to `idx_state` in the form (xd, vd).
        """
        idx_state_2d = self.get_index_2d_from_index(idx_state)
        state_discrete = self.undiscretize(idx_state_2d)
        return state_discrete
    #-----------

    #-----------
    def get_indices_terminal_states(self):
        """
        Returns the list of 1D state indices containing (x, v) states that are terminal. Note that normally terminal states will contain also states (x, v)
        that are NOT terminal, just because the grid into which the (x, v) state space is divided will most likely no grid line will pass exactly
        by the goal position of x = self.goal_position.
        """
        idx_states_terminal = [self.get_index_from_state((xd, vd))  for xd in self.positions for vd in self.velocities
                                                                    if xd >= self.goal_position or xd < self.goal_position and xd + self.dx >= self.goal_position]
        return idx_states_terminal

    def get_indices_non_terminal_states(self):
        "Returns the 1D indices of the non-terminal discrete states, defined as those (xd, vd) cells that do NOT contain any continuous-state (x, v) having x >= self.goal_position"
        idx_states_non_terminal = [self.get_index_from_state((xd, vd))  for xd in self.positions for vd in self.velocities
                                                                        if xd < self.goal_position and xd + self.dx < self.goal_position]
        return idx_states_non_terminal
    #-----------

    def get_adjacent_coming_from_states(self, state: tuple):
        "Returns the set of discrete-valued states (xd, vd) adjacent to the given continuous-valued state (x, v) whose probability of transitioning to (x, v) is > 0"
        idx_state = self.get_index_from_state(state)
        adjacent_states_list = []
        actions_from_adjacent_states_list = []

        # Iterate on all discrete positions and discrete velocities and check whether they can transition to the given state
        for xd in self.positions:
            for vd in self.velocities:
                test_state = (xd, vd)
                idx_test_state = self.get_index_from_state(test_state)
                for a in np.arange(self.getNumActions()):
                    self.setState(idx_test_state)
                    idx_next_state = self.step(a)[0]
                    if idx_next_state == idx_state:
                        adjacent_states_list += [test_state]
                        actions_from_adjacent_states_list += [a - 1]    # This converts the actions 0, 1, 2 to -1, 0, 1 (accelerate left, do not accelerate, accelerate right)

        # Set the state of the environment at the state where it was before calling this method
        self.setState(state)

        #print(f"Adjacent states for state idx={idx_state}, {state}:")
        #print([str(s) + " (a=" + str(a) + ")" for s, a in zip(adjacent_states_list, actions_from_adjacent_states_list)])

        return adjacent_states_list

    #--- Getters
    def isTerminalState(self, idx_state: int):
        state = self.get_state_from_index(idx_state)
        x = self.getPosition(state)
        #print("Check terminal state: {} (dx={})".format(state, self.dx))
        return x >= self.goal_position

    def getPosition(self, state: tuple):
        """
        Returns the position of the given 2D continuous-valued state

        Arguments:
        state: tuple (but also list and numpy array are valid)
            The state (x, v) for which the position is wished.

        Return: float
        The continuous-valued position associated to the given state.
        """
        return state[0]

    def getVelocity(self, state: tuple):
        """
        Returns the velocity of the given 2D continuous-valued state

        Arguments:
        state: tuple, list or array
            The state (x, v) for which the velocity is wished.

        Return: float
        The continuous-valued velocity associated to the given state.
        """
        return state[1]

    def getPositions(self):
        "Returns the lower limit of all discretized intervals of the car's position"
        return self.positions

    def getVelocities(self):
        "Returns the lower limit of all discretized intervals of the car's velocity"
        return self.velocities

    def isStateContinuous(self):
        return self.state_is_continuous

    def getStateDimensions(self):
        """
        Returns the dimension of the discretized (x, v) states, i.e. first the number of discrete positions, and then the number of discrete velocities,
        REGARDLESS of the shape used to represent the state space (which is given by self.getShape()).
        """
        return (self.nx, self.nv)

    def getShape(self):
        return self.shape

    def getShapeDisplayNames(self):
        """
        Returns the names of the measures stored in each dimension of the 2D representation of the states

        Return: tuple
        Duple with the following elements:
        - name of the row labels of the 2D shape storing the states (e.g. "position")
        - name of the column labels of the 2D shape storing the states (e.g. "velocity")
        """
        return self._shape_display_names

    def getPositionDimension(self):
        "Returns the dimension in the 2D shape representation of the environment states containing the position."
        return self._shape_display_names.index("position")

    def getVelocityDimension(self):
        "Returns the dimension in the 2D shape representation of the environment states containing the velocity."
        return self._shape_display_names.index("velocity")

    def getPositionColor(self):
        "Returns the color to use in plots about the position"
        return "red"

    def getVelocityColor(self):
        "Returns the color to use in plots about the velocity"
        return "blue"

    # MERGED!
    def getState(self, simulation=True):
        """
        Returns the current state of the environment, whose value (either an integer or a tuple) depends on parameter `simulation`

        Arguments:
        simulation: (opt) bool
            Whether to return the current state of the environment as the value used in simulations or return the state as a tuple representing the
            continuous-valued state of the environment (when self.state_is_continuous = True) or the discrete-valued state (when self.state_is_continuous = False)
            default: True

        Return: int or tuple
        When simulation = False: tuple (either continuous-valued state (x, v) when self.state_is_continuous = True, or discrete-valued state (xd, vd) o.w.)
        When simulation = True: tuple when self.state_is_continuous = True, o.w. int between 0 and nS-1, where nS is the number of discrete states in the environment.
        """
        if self.state_is_continuous or not simulation:
            return self.state
        else:
            # The environment is represented by discrete states and the method is called by a simulation process (or the caller wants to know what the simulation state would be)
            # => Get the 1D state index representing the current state of the discrete-state environment and return it
            idx_state = super().getState()  # This calls the getState() method in the discrete.DiscreteEnv class (since the MountainCarEnv class does not have any getState() method defined)

            # Now we assert that the 1D index and the discrete-valued state (xd, vd) stored in the MountainCarEnv super class are aligned.
            # Note that an eventual assertion of the 1D to 2D conversion (i.e. the opposite of the above assertion) would fail soon after the environment is reset,
            # because in the default process of the self.reset() method, a continuous-valued 2D state is first chosen at random and then a 1D state index mapped to it:
            # if we converted this 1D state index back to a 2D state, the result would NOT be the 2D state selected at random in the 2D continuous-valued space!
            assert self.get_index_from_state(self.state) == idx_state, \
                "The 2D continuous-valued state (x, v) stored in the MountainCarEnv super class ({}) falls in the cell represented by the 1D state index stored in the discrete.DiscreteEnv object ({})." \
                .format(self.state, idx_state, self.get_index_from_state(self.state))
            return idx_state

    # MERGED!
    def getStateFromIndex(self, idx_state: int, simulation=True):
        """
        Returns the state associated to the given 1D state index, whose value (either an integer or a tuple) depends on parameter `simulation`

        Arguments:
        simulation: (opt) bool
            Whether to return state associated to the given 1D index as the value used in simulations or return the state as a tuple representing the
            continuous-valued state associated to `idx_state` (when self.state_is_continuous = True) or the discrete-valued state (when self.state_is_continuous = False)
            default: True

        Return: int or tuple
        When simulation = False: tuple (either continuous-valued state (x, v) when self.state_is_continuous = True, or discrete-valued state (xd, vd) o.w.)
        When simulation = True: tuple when self.state_is_continuous = True, o.w. int between 0 and nS-1, where nS is the number of discrete states in the environment.
        """
        if self.state_is_continuous or not simulation:
            return self.get_state_from_index(idx_state)
        else:
            return idx_state

    # MERGED!
    def getIndexFromState(self, state, simulation=True):
        """
        Returns the 1D index representation of the given state. when simulation=True, `state` must be an integer value representing a 1D index as well,
        as the state used for simulations is a 1D index (as the state space is discrete, so we can do that).
        When simulation=False, `state` is expected to be a tuple containing the continuous-valued state.
        """
        if self.state_is_continuous or not simulation:
            return self.get_index_from_state(state)
        else:
            assert is_integer(state), f"The simulation state must be integer: {state}"
            return state

    def getStateIndicesFromIndex(self, idx_state: int):
        "Returns the 2D state indices associated to the 1D state index"
        return self.get_index_2d_from_index(idx_state)

    def getDiscreteStateFromState(self, state: tuple):
        "Returns the discrete-valued state (xd, vd) associated to the continuous-valued state (x, v)"
        return self.get_state_discrete_from_state(state)

    #--- Setters
    # MERGED!
    def setState(self, state_or_state_index: Union[int, tuple]):
        """
        Sets the state of the environment, both the 1D state index (attribute `s` of EnvironmentDiscrete super class)
        and the continuous-valued state (x, v) representing the position and velocity of the mountain car (attribute `state` of MountainCarEnv super class).

        Arguments:
        state_or_state_index: int or array-like
            When int, it represents a 1D state index.
            When array, it represents a 2D continuous-valued state of the original environment (super class).
        """
        if is_integer(state_or_state_index):
            # The input state is a 1D state index
            # => set the continuous-valued state using the get_state_from_index() method which normally selects the state at the lower limit of the 2D cell represented by the 1D state index
            idx_state = state_or_state_index
            # Compute the discrete-valued state associated to the given 1D state index
            state = self.get_state_from_index(idx_state)
        else:
            state = state_or_state_index
            idx_state = self.get_index_from_state(state)

        # Set the state (1D index `s`) in the EnvironmentDiscrete environment
        super().setState(idx_state)
        # Set the state (continuous-valued position and velocity (x, v)) in the MountainCarEnv environment
        # NOTE that this will most likely CHANGE the state stored in the super class environment to the discrete-valued state (xd, vd) if the input argument is a 1D index.
        self.state = state

    #--- Plot methods
    def plot_trajectory_gif(self, trajectory):
        "Generates a GIF showing the states through which the mountain car has been during learning"
        positions = [state[0] for state in trajectory]
        def mountain_car_track(x):
            return np.sin(3*x)

        fig, ax = plt.subplots()
        x = np.linspace(-1.2, 0.6, 1000)
        y = mountain_car_track(x)
        ax.plot(x, y, 'k')

        ax.set_xlim(-1.2, 0.6)
        ax.set_ylim(-1.2, 1.2)  # Maybe change this to see properly

        cmap = cm.get_cmap('coolwarm')
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
        print("GIF saved to: ./mountain_car.gif")

    def plot_values(self, values, ax=None, figsize=(8, 8), cmap="Blues", vmin=None, vmax=None, alpha=1.0):
        "Plots values (optionally on the given `ax` axis), defined for EVERY state and given as a 1D array, on the 2D shape used to represent the environment as an image. Returns the axis object and the image object."
        values_2d = self.reshape_from_1d_to_2d(values)
        if ax is None:
            new_figure = True
            ax = plt.figure(figsize=figsize).subplots(1, 1)
        else:
            new_figure = False

        # Plot the values by starting from the lower left corner and going first horizontally showing the values on the first row of `values_2d`,
        # then move up one line in the image and plot the values on the second row, etc.
        img = ax.imshow(values_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
        plt.colorbar(img)

        if new_figure:
            self._finalize_plot(ax)

        return ax, img

    def plot_points(self, points, ax=None, style=".-", figsize=(8, 8), color="blue", cmap=None, markersize=5):
        """
        Plots a set of points representing simulation states (i.e. recorded by simulations) in the 2D discrete-state shape representation of the environment

        Points are connected by lines if requested by the code in `style`.

        Arguments:
        points: array-like
            Points to plot representing states obtained from simulations.

        ax: (opt) Axes object
            Existing axes object on which the points should be added.
            default: None, in which case a new axes object is created

        style: (opt) str
            Style of the plotted points, it's the symbol used in matplotlib.pyplot.plot(), e.g. ".-" or "." or "x".
        """
        if ax is None:
            new_figure = True
            ax = plt.figure(figsize=figsize).subplots(1, 1)
        else:
            new_figure = False

        n_points = len(points)
        if cmap is not None:
            colors = cm.get_cmap(cmap, lut=n_points)
        for i, point in enumerate(points[:-1]):
            # Convert the point to a 2D index representing the entry of the point in the 2D shape representation of the environment
            idx_point = self.getIndexFromState(point)
            idx_point_next = self.getIndexFromState(points[i+1])
            idx_point_2d = self.get_index_2d_from_index(idx_point)
            idx_point_2d_next = self.get_index_2d_from_index(idx_point_next)
            _color = colors(i / n_points) if cmap is not None else color
            if self.getShapeDisplayNames() == ("velocity", "position"):
                # Velocity is on the VERTICAL axis and position on the HORIZONTAL axis
                ax.plot( [idx_point_2d[self.getPositionDimension()], idx_point_2d_next[self.getPositionDimension()]],
                         [idx_point_2d[self.getVelocityDimension()], idx_point_2d_next[self.getVelocityDimension()]],
                         style, color=_color, markersize=markersize)
            else:
                # Velocity is on the HORIZONTAL axis and position on the VERTICAL axis
                ax.plot( [idx_point_2d[i - 1, self.getVelocityDimension()], idx_point_2d[i, self.getVelocityDimension()]],
                         [idx_point_2d[i - 1, self.getPositionDimension()], idx_point_2d[i, self.getPositionDimension()]],
                         style, color=_color, markersize=markersize)

        if new_figure:
            self._finalize_plot(ax)

        return ax

    def add_labels(self, ax, labels, color="black", fontsize=14):
        "Adds `labels`, given as a 1D array, to an existing plot, given in `ax` which is assumed to contain an image, e.g. generated by self.plot_values()"
        labels_2d = labels.reshape(self._shape_display)
        for y in range(self._shape_display[0]):
            for x in range(self._shape_display[1]):
                # Recall the y value indexes the row of the matrix shown in the image (vertical axis of the plot)
                # and the x value indexes the column of the matrix shown in the image (horizontal axis of the plot)
                # That's why the text coordinates (of state_counts) are (y, x) to be placed at coordinates (x, y).
                ax.text(x, y, "{:.0f}".format(labels_2d[y, x]), color=color, fontsize=fontsize, horizontalalignment="center", verticalalignment="center")

    def _finalize_plot(self, ax):
        "Finalizes the plot created with self.plot_values() or self.plot_points() with more informed axes"
        # Recall that _shape_display_names indicates a tuple with the concept plotted on the Y axis and the concept plotted on the X axis.
        ax.set_xlabel(self._shape_display_names[1])
        ax.set_ylabel(self._shape_display_names[0])

        # Make the X-axis and Y-axis cover the whole spectrum of positions and velocities
        # Recall that the X and Y axis values are the INDICES of the positions and velocities respectively, NOT their actual position and velocity values.
        # We comment this out because it actually doesn't really do what we want, namely to have e.g. the velocity axis values symmetrical...
        if self._shape_display_names[0] == "velocity":
            # Velocity is plotted on the VERTICAL axis
            ax.set_xlim((ax.get_xlim()[0], len(self.positions) - 0.5))
            ax.set_ylim((ax.get_ylim()[0], len(self.velocities) - 0.5))
        else:
            # Velocity is plotted on the HORIZONTAL axis
            ax.set_xlim((ax.get_xlim()[0], len(self.velocities) - 0.5))
            ax.set_ylim((ax.get_ylim()[0], len(self.positions) - 0.5))

        # Show the actual position and velocity values as ticklabels
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        xticks = ax.get_xticks()[1:-1].astype(int)
        yticks = ax.get_yticks()[1:-1].astype(int)
        if self._shape_display_names[0] == "velocity":
            # Velocity is plotted on the VERTICAL axis
            xticklabels = ["{:.3f}".format(s) for s in self.positions[xticks]]
            yticklabels = ["{:.3f}".format(s) for s in self.velocities[yticks]]
        else:
            # Velocity is plotted on the HORIZONTAL axis
            xticklabels = ["{:.3f}".format(s) for s in self.velocities[xticks]]
            yticklabels = ["{:.3f}".format(s) for s in self.positions[yticks]]
        ax.xaxis.set_ticklabels([ax.xaxis.get_ticklabels()[0]] + xticklabels + [ax.xaxis.get_ticklabels()[-1]]);
        ax.tick_params(axis='x', labelrotation=90)
        ax.yaxis.set_ticklabels([ax.yaxis.get_ticklabels()[0]] + yticklabels + [ax.yaxis.get_ticklabels()[-1]])




class deprecated_MountainCarDiscrete(MountainCarEnv):
    """
    Continuous-state Mountain Car environment defining the methods used by my learners (e.g. , such as LeaTDLambda.

    Discrete here means: discrete actions (-1, 0, +1) but continuous states, which can be discretized if needed by learning processes such as tabular
    value functions learning.

    The constructor receives the number of points in which we would like to discretize the position and velocity values.

    Arguments:
    nx: int
        Number of grid points for the car's position.

    nv: int
        Number of grid points for the car's velocity.
        An odd number is enforced by adding 1 when even, so that v = 0.0 is always part of the velocity grid.

    seed_reset: (opt) int
        Seed to use when choosing the start state when resetting the environment.
        default: None
    """

    # NAMING CONVENTIONS FOR ATTRIBUTES AND VARIABLES CONTAINING STATE INFORMATION:
    # In this class we call:
    # - idx_state: (int) 1D index representing the state (used in the discrete.DiscreteEnv environment from which the super
    # class EnvironmentDiscrete derives (it's used in its attribute `self.s`)).
    # The way this 1D index is computed from the state depends on the shape used to display the 2D environment, weither positions x velocities or velocities x positions,
    # as the conversion from `state` to `idx_state` is done by first converting `state` to `idx_state_2d` and then converting `idx_state_2d` to `idx_state` using the
    # C-order for conversions from multidimensional indices to unidimensional indices. See method get_index_from_state().
    # - idx_state_2d: (tuple) 2D index of the continuous-valued state that indexes the 2D environment layout used in plots.
    # Normally plots will have the position x plotted on the horizontal axis and the velocity v on the vertical axis,
    # as this better represents the shape of the mountain car track, where the goal is on the right.
    # Ex: (3, 5)
    # - state_discrete: (tuple) 2D discretized state as tuple (xd, vd) = (position, velocity) representing the LOWER LIMIT of each discretized cell.
    # Ex: (0.4, 0.07)
    # - state: (tuple) 2D continuous-valued state as a tuple (x, v), i.e. the state stored in the super class MountainCarEnv.
    # - state_simulation: (tuple) the value representing the state during simulations. This coincides with `state`. It is only used here as a concept,
    # as there is no information stored about it.
    def __init__(self, nx, nv, factor_for_force_and_gravity=1.0, factor_force=1.0, factor_max_speed=1.0, seed_reset=None, debug=False):
        MountainCarEnv.__init__(self)
        self.debug = debug
        self.setSeed(seed_reset)

        # Define that the state of this environment is CONTINUOUS-valued
        self.state_is_continuous = True

        # Adapt the max_speed, force and gravity of the original environment so that the movement is interesting in terms of changing states in the discretized problem
        # and/or to increase the probability of reaching the goal at a reasonable simulation time.
        self.max_speed = factor_max_speed * self.max_speed
        self.force = factor_force * factor_for_force_and_gravity * self.force
        self.gravity = factor_for_force_and_gravity * self.gravity

        # Number of grid points for each dimension (x, v)
        self.nx = nx
        self.nv = nv
        # Make sure that nv is odd so that v=0.0 is part of the grid and we get symmetric values for v.
        self.nv = self.nv + 1 if self.nv % 2 == 0 else self.nv

        self.xmin, self.xmax = self.min_position, self.max_position
        self.vmin, self.vmax = -self.max_speed, self.max_speed

        # Discretization sizes
        self.dx = (self.xmax - self.xmin) / (self.nx - 1)
        self.dv = (self.vmax - self.vmin) / (self.nv - 1)

        # Discrete position values (lower limit of grid interval)
        self.positions = np.linspace(self.xmin, self.xmax, self.nx, endpoint=True)
        self.velocities = np.linspace(self.vmin, self.vmax, self.nv, endpoint=True)
        # Enforce the value of 0.0 velocity for the assertion below
        for i, v in enumerate(self.velocities):
            if np.isclose(v, 0.0):
                self.velocities[i] = 0.0

        # The above calculations assume the following
        assert self.xmax >= self.goal_position
        assert 0.0 in self.velocities
        assert len(self.positions) == self.nx
        assert len(self.velocities) == self.nv
        assert np.isclose(self.positions[1] - self.positions[0], self.dx, atol=1E-6), "v0={:.3f}, v1={:.3f}, v1-v0={:.6f}, dv={:.6f}".format(self.positions[0], self.positions[1], self.positions[1] - self.positions[0], self.dx)
        assert np.isclose(self.velocities[1] - self.velocities[0], self.dv,atol=1E-6), "v0={:.3f}, v1={:.3f}, v1-v0={:.6f}, dv={:.6f}".format(self.velocities[0], self.velocities[1], self.velocities[1] - self.velocities[0], self.dv)

        # Shape of the environment for the outside world
        # (see details in MountainCarDiscrete2)
        self.shape = (self.nx, self.nv)

        # Shape of the environment for display purposes, which in turn corresponds to the 2D internal representation of the state
        # (see details in MountainCarDiscrete2)
        self._shape_display_names = ("velocity", "position")  # Here we choose x to go along the horizontal direction, i.e. its values are represented on the X axis.
        self._shape_display = (self.nx, self.nv) if self._shape_display_names == ("position", "velocity") else (self.nv, self.nx)

        # Reward landscape
        self.reward_at_goal = 1.0   # Set this to 1.0 for a sparse reward landscape and zero for a reward of -1 at every step

        # Number of states and actions
        self.nS = self.nx * self.nv
        self.nA = self.action_space.n
        self.all_states = list(np.arange(self.nS))
        # In case we need to store the 2D states, we can use this
        # self.all_states_2d = [(x, v) for x in self.positions for v in self.velocities]
        # In case we need to iterate on all possible states
        # it = np.nditer(self.all_states_2d, flags=['multi_index'])  # Retrieve the original 2D layout with `x, v = it.multi_index` when iterating with `while not it.finished`

        # Attributes defined in super classes
        self.dim = 2                    # From my EnvironmentDiscrete: dimension of the environment
                                        # In principle this is used only for plotting purposes, e.g. when plotting
                                        # the estimated value function as a 2D image for `dim = 2` environments.
        # We need to define the initial state distribution that is required by the constructor of EnvironmentDiscrete called below,
        # which in turn inherits from the `discrete` environment in toy_text.
        # However, this piece of information is NOT used because the reset of the environment is done by MountainCarEnvironment, NOT by EnvironmentDiscrete
        self.isd = np.zeros(self.nS)
        self.isd[0] = 1

        # Define the reward received by the terminal states
        # This is used when computing the expected reward on a FINITE state space using sum{x} p(x)r(x), where r(x) is the reward received by the DISCRETE state x,
        # for instance when the compute_expected_reward() function in computing.py is called to compute the expected reward where the environment's method
        # getRewardsDict() retrieving the reward stored in the self.rewards attribute is used as r(x) value.
        # Below we define self.rewards which assigns a non-zero reward to the terminal states.
        self.terminal_states = self.get_indices_terminal_states()
        # Do of the terminal states
        for s in self.terminal_states:
            # Check that some states in the discrete states classified as terminal states contain some continuous-valued terminal states (i.e. having position x >= self.goal_position = 0.5)
            assert self.get_state_from_index(s)[0] + self.dx >= self.goal_position, \
                f"SOME of the states in the set of terminal states must have position >= goal_position (= {self.goal_position}).\nCondition failed for ({s, self.get_state_from_index(s)})."
        # Check that none of the states in discrete states NOT classified as terminal states contain continuous-valued terminal states
        assert len([self.get_state_from_index(s) for s in self.all_states if s not in self.terminal_states and self.get_state_from_index(s)[0] + self.dx >= self.goal_position]) == 0, \
                f"NONE of the states in the set of non-terminal states have position >= goal_position (= {self.goal_position})"

        # Rewards dictionary
        # Note that a non-zero reward between 0 and self.reward_at_goal is assigned to PARTIAL discrete terminal states, i.e. to discrete terminal states that contain both
        # states that are NOT continuous-valued terminal states and states that are continuous-valued terminal states.
        self.rewards = dict([(s, self.reward_at_goal * min(1.0, (self.get_position(self.get_state_from_index(s)) + self.dx - self.goal_position) / self.dx)) for s in self.terminal_states])

        # Reset the environment
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
    def reset(self):
        "Resets the environment's continuous-valued state"
        # Set an initial random state using the MountainCarEnv procedure
        self.seed(self.seed_reset)           # This method is defined in MountainCarEnv included in gym/envs/classic_control
        # Set the (x, v) state of the Mountain Car (continuous-valued 2D state)
        # Note that the reset() method sets the value of self.state of the super class as an array (NOT a tuple, so here we convert it to a tuple because of the next comment)
        self.state = tuple(super().reset())  # Convert the state in the super class to a tuple because this is how it is stored by their step() method!

        return self.state

    def step(self, action):
        """
        Performs one step in the environment for the given action in the continuous-valued state

        Arguments:
        action: int
            Action taken by the agent on the environment.

        Return: tuple
        A 4-tuple with the following elements:
        - state: the next continuous-valued state after taking `action`.
        - reward: the reward received after visiting `state`.
        - done: whether the car reached the goal (based on the continuous-valued state).
        - info: additional info returned by the step() method of the original mountain car environment.
        """
        observation, reward, done, info = super().step(action)  # This step() method sets the self.state attribute of the MountainCarEnv super class
        assert np.allclose(self.state, observation), "The state of the environment must have been set when calling step(action)"

        if done:
            # Adapt the reward structure defined in the original Mountain Car environment to our reward structure in two ways:
            # - The reward structure of the original environment is designed as the reward received
            # after LEAVING the state (i.e. after taking an action, because they want to count the number of actions taken
            # and minimize that number as optimization objective), whereas our reward structure is that the reward
            # is received when VISITING the state.
            # - The reward structure of the original environment is to assign -1 to each step performed until the goal is reached.
            # Our reward structure is to assign +1 when reaching the goal and 0 elsewhere.
            reward = self.reward_at_goal
            #print("\nStep: action = {} -> observation = {}, reward = {}, done = {}".format(action, observation, reward, done))
        else:
            reward = self.reward_at_goal + reward   # This is ZERO reward when reward_at_goal = 1.0 (since the reward yielded by the original environment is -1 at every step

        return self.state, reward, done, info

    def discretize(self, state):
        """
        Discretizes into 2D indices the given continuous-valued state using the discretization parameters defined in the object (e.g. dx, dv, etc.)
        and self._shape_display_names, which defines the layout of position and velocity in the 2D environment layout used to visually represent the environment.

        Arguments:
        state: tuple
            The continuous-valued state = (x, v) to discretize.

        Return: tuple
        Tuple containing the 2D indices representing the cell in the 2D environment layout where the given `state` falls.
        Whether the 2D indices represent first the position index and then the velocity index or the other way round depends on the 2D layout
        used to represent the environment, defined by self._shape_display_names.
        """
        idx_state_2d = np.array((-1, -1))
        x, v = state

        # We now use np.digitize() instead of my basic.discretize() function because my original discretize didn't
        # always work properly (border problems, e.g. when giving exactly the left bound of an interval, it returned the
        # interval to the left of the right one (x = -1.02 when -1.02 was the left bound of the interval index 1).
        # Also, even if I have deprecated my discretize() function --and replaced with a discretize() method that
        # calls np.digitize() under the hood-- we still call np.digitize() here in order to avoid the generation of
        # the intervals each time we need to discretize a value. Instead we store these intervals as part of the object.
        # IMPORTANT: np.digitize() returns a value between 1 and the length of bins (second parameter),
        # NOT between 0 and len-1 as one would have expected (considering how Python defines indices)!!!!!!

        # Use this if we want to ROUND the continuous-valued position and velocity
        #idx_state_2d[self.getPositionDimension()] = max(0, int(np.digitize(x + self.dx/2, self.positions) - 1))
        #idx_state_2d[self.getVelocityDimension()] = max(0, int(np.digitize(v + self.dv/2, self.velocities) - 1))

        # Use this if we want to TRUNCATE the position and velocity
        # Note that we bound any values going out of bound from the left to the leftmost position and velocity values
        idx_state_2d[self.getPositionDimension()] = max(0, int(np.digitize(x, self.positions) - 1))
        idx_state_2d[self.getVelocityDimension()] = max(0, int(np.digitize(v, self.velocities) - 1))

        assert 0 <= idx_state_2d[self.getPositionDimension()] <= self.nx - 1, "{}, {}".format(idx_state_2d[self.getPositionDimension()], self.nx-1)
        assert 0 <= idx_state_2d[self.getVelocityDimension()] <= self.nv - 1, "{}, {}".format(idx_state_2d[self.getVelocityDimension()], self.nv-1)

        return tuple(idx_state_2d)

    def undiscretize(self, idx_state_2d):
        """
        Returns the discrete-valued state (xd, vd) associated to the given 2D index of the 2D environment layout (defined by self._shape_display_names)

        Arguments:
        idx_state_2d: tuple, list or array
            The 2D index of the state to convert to a discrete-valued state, with the position index first or the velocity index first
            depending on the value of self._shape_display_names which defines precisely that order as either
            ("position", "velocity") or ("velocity", "position") and whose value is returned by self.getShapeDisplayNames().

        Return: tuple
        Tuple (xd, vd) containing the discrete-valued position at the first element and the discrete-valued velocity at the second element.
        These values represent the undiscretized state defined as the LOWER LIMIT of the discretized cell represented by the given 2D index
        discretizing the continuous state space.
        """
        dim_x = self.getPositionDimension()
        dim_v = self.getVelocityDimension()
        state_discrete = tuple([self.positions[idx_state_2d[dim_x]],
                                self.velocities[idx_state_2d[dim_v]]])
        return state_discrete

    def reshape_from_2d_to_1d(self, values_2d):
        """
        Converts the given 2D representation of the values to a 1D representation using C-like ordering of the 2D shape.
        This means that the 1D representation of the values will map the 2D cells of the 2D environment layout in the order in which
        the quantity (either position or velocity) placed along the HORIZONTAL direction (columns) varies first.
        This depends on the value of self._shape_display_names. More precisely:
        - when self._shape_display_names = ("position", "velocity") => the 1D representation of the state indexes the state by varying first the velocity,
        as velocity is laid out along the horizontal direction (columns).
        - when self._shape_display_names = ("velocity", "position") => the 1D representation of the state indexes the state by varying first the position,
        as position is laid out along the horizontal direction (columns).
        """
        return np.squeeze( values_2d.reshape(1, np.prod(values_2d.shape)) )

    def reshape_from_1d_to_2d(self, values_1d):
        "Converts a 1D representation of the values to a 2D representation using C-like ordering. See details in documentation for self.reshape_from_2d_to_1d()"
        return values_1d.reshape(self._shape_display)

    #-----------
    def get_index_from_state(self, state):
        "Returns the 1D state index corresponding to the given continuous-valued state"
        idx_state_2d = self.discretize(state)
        idx_state = self.get_index_from_index_2d(idx_state_2d)
        return idx_state

    def get_state_from_index(self, idx_state: int):
        """
        Returns the 2D continuous state associated to the given 1D state index

        The 2D continuous state is actually the lower limit (xd, vd) of the 2D interval, of the grid into which the (position, velocity) state space
        has been discretized to, containing the set of all continuous states (x, v) represented by the 1D index.
        """
        return self.get_state_discrete_from_index(idx_state)

    def get_random_state_from_index(self, idx_state: int):
        "Returns a randomly chosen continuous-valued state (a state from the original Mountain Car environment) from the position-velocity interval referenced by the 1D state index"
        state_discrete = self.get_state_discrete_from_index(idx_state)

        # Choose a random position and a random velocity in the interval in order to make the continuous state
        position_lower = state_discrete[0]
        position_upper = position_lower + self.dx
        velocity_lower = state_discrete[1]
        velocity_upper = velocity_lower + self.dv
        # Note that we use self.np_random() and NOT np.random() because self.np_random contains the random number generator whose seed is the seed set as self.seed_reset by the reset() method
        state = (self.np_random.uniform(position_lower, position_upper), self.np_random.uniform(velocity_lower, velocity_upper))

        return state
    #----------

    #-----------
    def get_state_discrete_from_index_2d(self, idx_state_2d):
        """
        Returns the discrete-valued state (xd, vd) from the given 2D state index.
        This is an alias for the undiscretize() method.
        """
        return self.undiscretize(idx_state_2d)

    def get_index_2d_from_state(self, state):
        idx_state_2d = self.discretize(state)
        return idx_state_2d
    #-----------

    #-----------
    def get_index_from_index_2d(self, idx_state_2d):
        """
        Returns the 1D state index corresponding to the 2D state index associated to the 2D environment layout

        Arguments:
        idx_state_2d: list, tuple or numpy array
            2D index representing the state in the 2D environment layout, which is defined by self._shape_display_names.

        Return: int
        1D index of the state following C-like order of the 2D index representation of the state according to the 2D environment layout,
        defined by self._shape_display_names, either ("position", "velocity") or=("velocity", "position").
        """
        return index_multi2linear(idx_state_2d, self._shape_display)

    def get_index_2d_from_index(self, idx_state: int):
        "Returns the 2D index state from the given 1D index state using C-like order (the default in Python)"
        return index_linear2multi(idx_state, self._shape_display)
    #-----------

    #-----------
    def get_state_discrete_from_state(self, state):
        """
        Returns the 2D discrete-valued state (xd, vd), from the given 2D continuous-valued state

        Arguments:
        state: tuple
            Continuous-valued state (x, v) to convert to a discrete-valued state (xd, vd).

        Return: tuple
        Duple representing the discrete-valued state associated to `state` in the form (xd, vd).
        """
        idx_state = self.get_index_from_state(state)
        state_discrete = self.get_state_discrete_from_index(idx_state)
        return state_discrete

    def get_state_discrete_from_index(self, idx_state: int):
        """
        Returns the discrete-valued state (xd, vd) from the given 1D state index.
        The conversion to continuous values is done by the undiscretize() method.

        Arguments:
        idx_state: int
            1D index representing the discretized state.

        Return: tuple
        Duple representing the discrete-valued state associated to `idx_state` in the form (xd, vd).
        """
        idx_state_2d = self.get_index_2d_from_index(idx_state)
        state_discrete = self.undiscretize(idx_state_2d)
        return state_discrete
    #-----------

    def get_positions(self):
        "Returns the lower limit of all discretized intervals of the car's position"
        return self.positions

    def get_velocities(self):
        "Returns the lower limit of all discretized intervals of the car's velocity"
        return self.velocities

    def get_indices_terminal_states(self):
        """
        Returns the list of 1D state indices containing (x, v) states that are terminal. Note that normally terminal states will contain also states (x, v)
        that are NOT terminal, just because the grid into which the (x, v) state space is divided will most likely no grid line will pass exactly
        by the goal position of x = self.goal_position.
        """
        idx_positions_terminal = [idx for idx, x in enumerate(self.get_positions()) if x + self.dx >= self.goal_position]
        # All velocities at terminal positions still make a terminal state
        idx_states_terminal = []
        for idx_v in range(self.nv):
            for idx_x in idx_positions_terminal:
                idx_state_2d = (idx_x, idx_v) if self._shape_display_names == ("position", "velocity") else (idx_v, idx_x)
                idx_states_terminal += [self.get_index_from_index_2d(idx_state_2d)]
        return idx_states_terminal

    #--- Getters
    def isStateContinuous(self):
        return self.state_is_continuous

    def getShape(self):
        return self.shape

    def getShapeDisplayNames(self):
        """
        Returns the names of the measures stored in each dimension of the 2D representation of the states

        Return: tuple
        Duple with the following elements:
        - name of the row labels of the 2D shape storing the states (e.g. "position")
        - name of the column labels of the 2D shape storing the states (e.g. "velocity")
        """
        return self._shape_display_names

    def get_position(self, state):
        """
        Returns the position of the given 2D continuous-valued state

        Arguments:
        state: tuple, list or array
            The state (x, v) for which the position is wished.

        Return: float
        The continuous-valued position associated to the given state.
        """
        return state[0]

    def get_velocity(self, state):
        """
        Returns the velocity of the given 2D continuous-valued state

        Arguments:
        state: tuple, list or array
            The state (x, v) for which the velocity is wished.

        Return: float
        The continuous-valued velocity associated to the given state.
        """
        return state[1]

    def getPosition(self, state):
        return self.get_position(state)

    def getVelocity(self, state):
        return self.get_velocity(state)

    def getPositionDimension(self):
        "Returns the dimension in the 2D shape of states along which the different positions are stored"
        return self._shape_display_names.index("position")

    def getVelocityDimension(self):
        "Returns the dimension in the 2D shape of states along which the different velocities are stored"
        return self._shape_display_names.index("velocity")

    def getPositionColor(self):
        "Returns the color to use in plots about the position"
        return "red"

    def getVelocityColor(self):
        "Returns the color to use in plots about the velocity"
        return "blue"

    def getState(self):
        """
        Returns the 2D continuous-valued state

        Return: numpy array
        2D array containing the position and velocity of the car in this order.
        """
        return self.state

    def getStateFromIndex(self, idx_state: int, simulation=True):
        """
        Returns the state associated to the given 1D state index

        Regardless of parameter `simulation` (which is present to be consistent with the other getStateFromIndex() methods defined in all other environments)
        the state returned is the value returned by get_state_from_index().
        """
        return self.get_state_from_index(idx_state)

    def getStateIndicesFromIndex(self, idx_state: int):
        "Returns the 2D state indices associated to the 1D state index"
        return self.get_index_2d_from_index(idx_state)

    def getIndexFromState(self, state, simulation=True):
        """
        Returns the 1D state index corresponding to the given continuous-valued state

        Regardless of parameter `simulation` (which is present to be consistent with the other getIndexFromState() methods defined in all other environments),
        `state` must always be a tuple representing the continuous-valued state
        """
        return self.get_index_from_state(state)

    def getDiscreteStateFromState(self, state):
        "Returns the discrete-valued state (xd, vd) associated to the continuous-valued state (x, v)"
        return self.get_state_discrete_from_state(state)

    def getReward(self, state):
        "Returns the reward received when visiting the continuous-valued state"
        return self.reward_at_goal if self.getPosition(state) >= self.goal_position else self.reward_at_goal - 1.0  # -1.0 is the reward yielded by the original environment at every step of the car

    def getNumStates(self):
        return self.nS

    def getNumActions(self):
        return self.nA

    def getAllStates(self):
        return self.all_states

    #-- Getters that are normally read from EnvironmentDiscrete (but this class does not inherit from it)
    def getV(self):
        return None

    def getQ(self):
        return None

    def getAllValidStates(self):
        return self.getAllStates()

    def getTerminalStates(self):
        "Returns the 1D indices representing the terminal states. Defined because it is required by simulators for INFO purposes (e.g. discrete.Simulator())"
        return self.terminal_states

    def getRewardsDict(self):
        return self.rewards

    def getInitialStateDistribution(self):
        "Returns a delta on the center state of the discretized position and velocity. This method is needed by simulators to treat extreme cases"
        isd = np.zeros(self.getNumStates())
        idx_state_center = self.getNumStates() // 2
        return isd
    #-- Getters that are normally read from EnvironmentDiscrete (but this class does not inherit from it)

    #--- Setters
    # (2024/08/11) THIS METHOD IS CURRENTLY NOT CALLED because at this point I intend to write this class only to handle the continuous-valued state
    # The discrete-valued state should be retrieved ad-hoc as needed by the process being run, e.g. in order to identify the states that should go to the absorption set A,
    # which requires that the state space be finite or countable.
    def setState(self, _state):
        """
        Sets the state of the 2D continuous-valued state of the environment

        Arguments:
        state: int or array-like
            When int, it represents a 1D state index.
            When array, it represents a 2D continuous-valued state of the original environment (super class).
        """
        if is_integer(_state):
            # The input state is a 1D state index
            # => convert it to a 2D continuous state by uniformly sampling in the interval associated to the 1D state index
            idx_state = _state
            state = self.get_random_state_from_index(idx_state)
        else:
            state = _state
        self.state = state

    def plot_trajectory_gif(self, trajectory):
        """
        Generates a GIF showing the states through which the mountain car has been during learning

        Arguments:
        trajectory: list
            List of 2D states (x, v) to plot. Their values can represent the continuous-valued states or the discrete-valued states,
            but always in the order (x, v), NOT (v, x).
        """
        positions = [state[0] for state in trajectory]
        def mountain_car_track(x):
            return np.sin(3*x)

        fig, ax = plt.subplots()
        x = np.linspace(-1.2, 0.6, 1000)
        y = mountain_car_track(x)
        ax.plot(x, y, 'k')

        ax.set_xlim(-1.2, 0.6)
        ax.set_ylim(-1.2, 1.2)  # Maybe change this to see properly

        cmap = cm.get_cmap('coolwarm')
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
        print("GIF saved to: ./mountain_car.gif")

    def plot_values(self, values, figsize=(8, 8), cmap="Blues", vmin=None, vmax=None, ax=None):
        "Plots values, defined for EVERY state, given as a 1D array on the 2D shape used to represent the environment as an image. Returns the axis object and the image object."
        values_2d = self.reshape_from_1d_to_2d(values)
        if ax is None:
            new_figure = True
            ax = plt.figure(figsize=figsize).subplots(1, 1)
        else:
            new_figure = False

        # Plot the values by starting from the lower left corner and going first horizontally showing the values on the first row of `values_2d`,
        # then move up one line in the image and plot the values on the second row, etc.
        img = ax.imshow(values_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(img)

        if new_figure:
            self._finalize_plot(ax)

        return ax, img

    def plot_points(self, points, style=".-", figsize=(8, 8), color="blue", cmap=None, markersize=5, ax=None):
        """
        Plots a set of points representing simulation states (i.e. recorded by simulations) in the 2D discrete-state shape representation of the environment.

        Points are connected by lines if requested by the code in `style`.

        Arguments:
        points: array-like
            Points to plot representing states obtained from simulations.

        style: (opt) str
            Style of the plotted points, it's the symbol used in matplotlib.pyplot.plot(), e.g. ".-" or "." or "x".
        """
        if ax is None:
            new_figure = True
            ax = plt.figure(figsize=figsize).subplots(1, 1)
        else:
            new_figure = False

        n_points = len(points)
        if cmap is not None:
            colors = cm.get_cmap(cmap, lut=n_points)
        for i, point in enumerate(points[:-1]):
            # Convert the point to a 2D index representing the entry of the point in the 2D shape representation of the environment
            idx_point = self.getIndexFromState(point)
            idx_point_next = self.getIndexFromState(points[i+1])
            idx_point_2d = self.get_index_2d_from_index(idx_point)
            idx_point_2d_next = self.get_index_2d_from_index(idx_point_next)
            _color = colors(i / n_points) if cmap is not None else color
            if self.getShapeDisplayNames() == ("velocity", "position"):
                # Velocity is on the VERTICAL axis and position on the HORIZONTAL axis
                ax.plot( [idx_point_2d[self.getPositionDimension()], idx_point_2d_next[self.getPositionDimension()]],
                         [idx_point_2d[self.getVelocityDimension()], idx_point_2d_next[self.getVelocityDimension()]],
                         style, color=_color, markersize=markersize)
            else:
                # Velocity is on the HORIZONTAL axis and position on the VERTICAL axis
                ax.plot( [idx_point_2d[i - 1, self.getVelocityDimension()], idx_point_2d[i, self.getVelocityDimension()]],
                         [idx_point_2d[i - 1, self.getPositionDimension()], idx_point_2d[i, self.getPositionDimension()]],
                         style, color=_color, markersize=markersize)

        if new_figure:
            self._finalize_plot(ax)

        return ax

    def add_labels(self, ax, labels, color="black", fontsize=14):
        "Adds labels to an existing plot assumed to contain an image, e.g. generated by self.plot_values()"
        labels_2d = labels.reshape(self._shape_display)
        for y in range(self._shape_display[0]):
            for x in range(self._shape_display[1]):
                # Recall the y value indexes the row of the matrix shown in the image (vertical axis of the plot)
                # and the x value indexes the column of the matrix shown in the image (horizontal axis of the plot)
                # That's why the text coordinates (of state_counts) are (y, x) to be placed at coordinates (x, y).
                ax.text(x, y, "{:.0f}".format(labels_2d[y, x]), color=color, fontsize=fontsize, horizontalalignment="center", verticalalignment="center")

    def _finalize_plot(self, ax):
        "Finalizes the plot created with self.plot_values() or self.plot_points() with more informed axes"
        # Recall that _shape_display_names indicates a tuple with the concept plotted on the Y axis and the concept plotted on the X axis.
        ax.set_xlabel(self._shape_display_names[1])
        ax.set_ylabel(self._shape_display_names[0])

        # Make the X-axis and Y-axis cover the whole spectrum of positions and velocities
        # Recall that the X and Y axis values are the INDICES of the positions and velocities respectively, NOT their actual position and velocity values.
        # We comment this out because it actually doesn't really do what we want, namely to have e.g. the velocity axis values symmetrical...
        if self._shape_display_names[0] == "velocity":
            # Velocity is plotted on the VERTICAL axis
            ax.set_xlim((ax.get_xlim()[0], len(self.get_positions()) - 0.5))
            ax.set_ylim((ax.get_ylim()[0], len(self.get_velocities()) - 0.5))
        else:
            # Velocity is plotted on the HORIZONTAL axis
            ax.set_xlim((ax.get_xlim()[0], len(self.get_velocities()) - 0.5))
            ax.set_ylim((ax.get_ylim()[0], len(self.get_positions()) - 0.5))

        # Show the actual position and velocity values as ticklabels
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        xticks = ax.get_xticks()[1:-1].astype(int)
        yticks = ax.get_yticks()[1:-1].astype(int)
        if self._shape_display_names[0] == "velocity":
            # Velocity is plotted on the VERTICAL axis
            xticklabels = ["{:.3f}".format(s) for s in self.get_positions()[xticks]]
            yticklabels = ["{:.3f}".format(s) for s in self.get_velocities()[yticks]]
        else:
            # Velocity is plotted on the HORIZONTAL axis
            xticklabels = ["{:.3f}".format(s) for s in self.get_velocities()[xticks]]
            yticklabels = ["{:.3f}".format(s) for s in self.get_positions()[yticks]]
        ax.xaxis.set_ticklabels([ax.xaxis.get_ticklabels()[0]] + xticklabels + [ax.xaxis.get_ticklabels()[-1]]);
        ax.tick_params(axis='x', labelrotation=90)
        ax.yaxis.set_ticklabels([ax.yaxis.get_ticklabels()[0]] + yticklabels + [ax.yaxis.get_ticklabels()[-1]])


class deprecated_MountainCarDiscrete2(MountainCarEnv, EnvironmentDiscrete):
    """
    Discrete-state Mountain Car environment defining the methods used by my learners, such as LeaTDLambda.

    Discrete here means: discrete actions (-1, 0, +1) and discrete states!
    Note that this may not be the most sensible way to approach this problem, as the dynamics are affected by this discretization
    and some tuning on the force, gravity and max_speed needs to be done in order to guarantee transitioning to other states
    when an action is applied for every (or most of) the states. However, this is a reasonable first approach to the problem
    which allows for a tabular representation of the state and action value functions.

    The constructor receives the number of points in which we would like to discretize the possible *velocity* values,
    which are in the range [-max_speed, +max_speed]. The max_speed value is defined by multiplying the original `max_speed` parameter
    in the MountainCarEnv super class environment (normally equal to 0.07) by 0.25*factor_for_force_and_gravity,
    in order to adapt it to an eventual adjustment of the car's force and gravity which for instance would imply that the car
    is able to go faster and thus be able to transition to other discrete states when an action is taken.

    The discretization grid for position is automatically computed from the velocity discretization grid so as to:
    - guarantee that the goal on the right is reached
    - guarantee that the car leaves (at least almost always) every possible interval (discrete state) at the smallest discretized non-zero velocity.
    The latter condition is applied to the interval with the smallest help from gravity, namely the valley,
    and the condition to leave the interval is established when the original velocity is 0.0, namely when the new
    velocity is equal to `force`.
    Note that the smaller the interval of the discretized velocity space (dv), the larger the number of discretized intervals for
    position. This implies that, the larger the number of velocity intervals, the larger the number of position intervals.

    When stepping to the next state using the step() function of the super class MountainCarEnv, the continuous (x, v)
    values assigned to each discrete state are the left bounds of the x- and v-interval to which the state belongs.
    Ex: if there are 6 grid points for x and 5 grid points for v, i.e.:
        xd in [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8]  (dx = 0.4)
        vd in [-0.07, -0.035, 0.00, 0.035, 0.07] (dv = 0.035)
    then each possible discrete-valued state is represented by all possible combinations of the 6x5 intervals making up 30 possible states
    to form the tuple (xd, vd).
    If the continuous-valued state in the MountainCarEnv super class is (x, v) = (0.43, 0.068),
    its corresponding discrete-valued state is (xd, vd) = (0.4, 0.035).
    This discrete-valued state is the state input to the step() method in this class that is responsible for moving the car
    after an action is taken (i.e. a discrete acceleration value, 0, 1 or 2), by calling the step() method of the super class
    on the discrete-valued state (xd, vd). When the new continuous-valued position and velocity are obtained corresponding
    to the next continuous-valued state, the next state is discretized into one of the (xd, vd) combinations,
    using the LOWER LEFT limit of the 2D cell where the (x, v) point falls.
    This means that the Mountain Car dynamics, although based on continuous-valued equations, are always based on its discrete-valued position and velocity,
    as opposed to its continuous-valued position and velocity.
    Note also that the discrete-valued state with largest possible values for position and velocity,
    (xd, vd) = (0.8, 0.07) represents the cell with x values in [0.8, 1.2) and v values in [0.07, 0.105).

    The original `force` and `gravity` values defined in the MountainCarEnv super class are multiplied by parameter `factor_for_force_and_gravity`
    so that we don't require so many discretized intervals of the position to guarantee the aforementioned
    condition on the position intervals, which would make the learning process very slow.
    For instance, the above logic of defining the number of discretized intervals for position would give about 1000
    intervals when we use the original `force` and `gravity` values defined in the super class MountainCarEnv,
    respectively 0.001 and 0.0025.
    With the default value of factor_for_force_and_gravity of:
    - 20 => "new force" = 0.02, "new gravity" = 0.05 => 107 discretized position intervals
    - 100 => "new force" = 0.1, "new gravity" = 0.25 => 21 discretized position intervals

    Arguments:
    nv: int
        Number of discretization intervals for velocity.
        An odd number is enforced by adding 1 when even, so that v = 0.0 is always part of the velocity grid.

    factor_for_force_and_gravity: (opt) positive float
        Multiplier of the original `force` and `gravity` values defined in the MountainCarEnv super class
        affecting the number of discretized intervals for the car's position, as described above.
        Increase this value to decrease the number of discrete positions in the environment.
        20 => 107 discrete positions
        100 => 21 discrete positions
        default: 20, which gives 107 discretized position intervals

    factor_force: (opt) positive float
        A factor to further multiply the force by, in addition to the already applied `factor_for_force_and_gravity`.
        This would typically be used to increase the car's force in order to control how easily the car is able to reach the goal
        by overcoming gravity. Use a value larger than 1 to make reaching the goal easier and smaller than 1 to make reaching the goal harder.
        default: 1.0

    seed_reset: (opt) int
        Seed to use when choosing the start state at environment reset.
        default: None
    """

    # NAMING CONVENTIONS FOR ATTRIBUTES AND VARIABLES CONTAINING STATE INFORMATION:
    # In this class we call:
    # - idx_state: (int) 1D index representing the state, used in the discrete.DiscreteEnv environment from which the super
    # class EnvironmentDiscrete derives (stored as attribute `s`), and is thus an attribute of the class.
    # The way this 1D index is computed from the state depends on the shape used to display the 2D environment, weither positions x velocities or velocities x positions,
    # as the conversion from `state` to `idx_state` is done by first converting `state` to `idx_state_2d` and then converting `idx_state_2d` to `idx_state` using the
    # C-order for conversions from multidimensional indices to unidimensional indices. See method get_index_from_state().
    # - idx_state_2d: (tuple) 2D index to access the state in the 2D DISPLAY of the environment states corresponding to `idx_state`.
    # This concept is an internal representation of which the user doesn't have to know about. In fact, it is NOT stored as an attribute.
    # Its value can be retrieved from `idx_state` using the get_index_2d_from_index()` method.
    # Ex: (3, 5)
    # - state: (tuple) 2D continuous-valued state as a tuple (x, v) defined in the super class MountainCarEnv. It is an attribute of the super class and thus of this class.
    # Ex: (0.43, 0.091), which in the example given in the class documentation, it falls inside the 2D cell [0.4, 0.8) x [0.07, 0.105) of the discretized state space
    # - state_discrete: (tuple) 2D discretized continuous-valued state as a tuple (xd, vd) = (position, velocity) representing the LOWER LIMIT of each discretized cell
    # of the continuous state space. It is NOT an attribute of the class and its value can be retrieved from `idx_state` by calling the `get_state_discrete_from_index()` method.
    # It is also possible to get this discretized-value state from the state by calling `get_state_discrete_from_state()` method.
    # Ex: (0.4, 0.07), which in the example given in the class documentation, it represents the 2D cell [0.4, 0.8) x [0.07, 0.105) of the discretized state space
    # - state_simulation: (int) the value representing the state during simulations. This coincides with `idx_state` because of the discrete-state nature of this environment.
    # Here, it is only used conceptually.
    #
    # HOWEVER, despite the above naming convention, it is currently a pity that the methods getState() and setState()
    # return the state 1D INDEX (and NOT the 2D continuous state as a tuple, as one would expect from their names)
    # because these methods override those defined in the super class EnvironmentDiscrete.
    # TODO: (2022/06/06) Rename the methods in EnvironmentDiscrete to get and set the state to getStateIndex() and setStateIndex()
    # (so that when we talk about `state` we think of the human-understandable state (e.g. (x, v) in the Mountain Car environment)
    # Note that the super class EnvironmentDiscrete should define methods called getState() and setState() which should raise NotImplementedError
    # to indicate that those methods cannot be implemented in such class (EnvironmentDiscrete) because we need information
    # about the environment itself, namely about how the index state self.s defined in discrete.DiscreteEnv translates
    # into the human-understandable state.
    def __init__(self, nv, factor_for_force_and_gravity=20, factor_force=1.0, seed_reset=None, debug=False):
        MountainCarEnv.__init__(self)
        self.debug = debug
        self.setSeed(seed_reset)

        # Define that the state of this environment is DISCRETE-valued, i.e. the result of taking an action on the car is computed from the discretized continuous-valued state
        self.state_is_continuous = False

        # Number of intervals  information of the 2D state information
        self.nv = nv
        # Make sure that nv is odd so that v=0.0 is part of the grid (required when defining the grid for positions below)
        # and we get symmetric values for v.
        self.nv = self.nv + 1 if self.nv % 2 == 0 else self.nv

        # Adjust the the force, the gravity and the max speed based on the given factor
        # This is done to reduce the number of discrete positions (and thus make the state space smaller)
        # Note that max_speed is defined in the original mountain car environment to clip the velocity which is a direct function of "force - gravity" (because the time step is t=1)
        factor_for_max_speed = 0.25*factor_for_force_and_gravity
        self.max_speed = factor_for_max_speed * self.max_speed
        self.force = factor_force * factor_for_force_and_gravity * self.force
        self.gravity = factor_for_force_and_gravity * self.gravity

        # Minimum and maximum values for position and velocity
        # Note: In the documentation of the source code in GitHub (mentioned above) it says that:
        # -1.2 <= x <= 0.5 (which is the goal position): note that we do not need to allow x to go past the goal position because we discretize the value of x using the left bound of each interval
        # -0.07 <= v <= 0.07
        # and these are ranges for force = 0.001 and gravity = 0.0025.
        self.xmin, self.xmax = self.min_position, self.goal_position
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
        # which happens when x = -pi/3 ~ -1.05 and x = 0.0
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
        # number of points and endpoint=True (the default))
        # Note that the number of points returned by np.linspace() is the number of points specified as third argument,
        # regardless of the value of endpoint.

        velocity_achieved_from_zero_speed_at_zero_gravity = self.force
        position_max_leftmost = self.xmin + 0.8*velocity_achieved_from_zero_speed_at_zero_gravity
            ## NOTE: (2024/08/09) I don't recall what the 0.8 factor means...
            ## NOTE 2: (2024/08/15) Note that if factor_for_force_and_gravity = 100, thanks to the 0.8, the value of dx becomes 0.08,
            ## making the discrete position closest to the rightmost position at x = 0.6 be 0.52, i.e. larger than the goal at 0.5,
            ## thus making the goal have at least one point.
        position_min_rightmost = self.goal_position - self.vmax  # This is not used but computed for informational purposes if needed
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
        # Enforce the value of 0.0 velocity for the assertion below
        for i, v in enumerate(self.velocities):
            if np.isclose(v, 0.0):
                self.velocities[i] = 0.0
        # The above calculations assume the following
        assert self.xmax == self.goal_position
        assert 0.0 in self.velocities
        assert len(self.positions) == self.nx
        assert len(self.velocities) == self.nv
        assert np.isclose(self.velocities[1] - self.velocities[0], self.dv,atol=1E-6), "v0={:.3f}, v1={:.3f}, v1-v0={:.6f}, dv={:.6f}".format(self.velocities[0], self.velocities[1], self.velocities[1] - self.velocities[0], self.dv)

        # Shape of the environment for the outside world
        # i.e. number of intervals in the position values x number of intervals in the velocity values obtained by discretization
        # that is, this shape respects the order in which position and velocity are stored in the continuous-valued state of the super class.
        # It differs from the display shape coming below, which furthermore is a private attribute, as the user doesn't need to know about it, in principle
        self.shape = (self.nx, self.nv)

        # Shape of the environment for display purposes, which in turn corresponds to the 2D internal representation of the state
        # (of which the user doesn't have to know; the user should know only of the actual state (or discretized continuous-valued state --e.g. (0.4, 0.07))
        # and of the 1D state index. In fact these two state representations may be used in simulations).
        # Note that the shape defines HOW we display the 2D state (x, v) as a matrix,
        # whether positions represent rows and velocities columns or viceversa.
        # The shape has NOTHING TO DO with the order of the tuple representing the state of the mountain car which is ALWAYS (x, v).
        self._shape_display_names = ("velocity", "position")  # Here we choose x to go along the horizontal direction, i.e. its values are represented on the X axis.
        self._shape_display = (self.nx, self.nv) if self._shape_display_names == ("position", "velocity") else (self.nv, self.nx)

        # Number of states and actions
        self.nS = self.nx * self.nv
        self.nA = self.action_space.n
        self.all_states = list(np.arange(self.nS))
        # In case we need to store the 2D states, we can use this
        #self.all_states_2d = [(x, v) for x in self.get_positions() for v in self.get_velocities()]

        # In case we need to have all possible (x, v) indices indexing the discrete position and velocity
        # We can go from 2D indices to 1D indices and viceversa using respectively the following functions
        # defined in the `basic` module:
        #   index_multi2linear()
        #   index_linear2multi()
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

        # Define other pieces of information required by processes that rely on the information stored in the EnvironmentDiscrete super class,
        # most importantly the reward received by the terminal states, which is used when calling the EnvironmentDiscrete.getReward() method, used extensively
        # in processes involving Fleming-Viot!!
        self.non_terminal_states = set(self.get_indices_for_non_terminal_states())
        self.terminal_states = set(self.getAllStates()).difference( set( self.non_terminal_states ) )
        self.rewards = dict([(s, 1.0) for s in self.terminal_states])

        # Initialize the other class from which this class inherits, so that we can use its attributes (e.g. rewards, etc)
        EnvironmentDiscrete.__init__(self, self.nS, self.nA, None, self.isd, dim=self.dim, rewards=self.rewards, terminal_states=self.terminal_states)

        # The following reset() call defines both:
        # - the 2D continuous-valued `state` = (x, v) containing the car's position and velocity of the MountainCarEnv environment
        # - the 1D state `s` of the discrete.DiscreteEnv of the gym library, from which EnvironmentDiscrete inherits, because we need these two state attributes to be ALIGNED.
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
        Resets the environment's state, including the continuous-valued state of the super class, the discrete-valued state representing the state in this class,
        and the 1D state index representing the state in the EnvironmentDiscrete super class (`s`)

        The reset can optionally be done using the initial state distribution (of the 1D state indices) given in parameter `isd`,
        otherwise the reset step of the MountainCarEnv environment is used.
        """
        if isd is not None:    # isd = Initial State Distribution to use, if any desired
            # Select the 1D index representing the 2D state of the mountain car
            # The reset of the state is done using the EnvironmentDiscrete reset process, which actually uses the gym.Env reset process
            # which selects a state following the Initial State Distribution information stored in the environment.
            print("ISD: {} (which={})".format(isd, np.where(isd==1.0)))
            idx_state = categorical_sample(isd, self.np_random)
        else:
            # Set an initial random state using the MountainCarEnv reset process
            self.seed(self.seed_reset)  # This method is defined in MountainCarEnv included in gym/envs/classic_control
            # Set the (x, v) state of the Mountain Car (continuous-valued 2D state)
            state = super().reset()     # reset() through the MountainCarEnv environment (since this is the first super class appearing in the list of super classes in the definition of this class)
            # Set the 1D index associated to the continuous-valued (x, v) state just set (this 1D index is computed using the position and velocity discretization process)
            idx_state = self.get_index_from_state(state)

        # Set the 2D continuous-valued state (state of the MountainCarEnv environment) and its 1D index version (state of the EnvironmentDiscrete environment)
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
            Whether to include the continuous-valued observation value in the returned tuple.
            It should be set to False when using step() on simulations, because simulations normally expect a 1D index as next state of the environment.
            default: False

        Return: either a 5-tuple or a 4-tuple depending on the value of return_continuous_state
        If return_continuous_state = False:
            It returns a 4-tuple with the 1D state index representing the current discrete state of the environment:
                `idx_state`, reward, done, info
        If return_continuous_state = True:
            It returns a 5-tuple with the actual 2D state representing the current continuous state of the environment:
                `state`, `idx_state_2d`, reward, done, info
            where `state` is the continuous-valued state of the MountainCarEnv super class
            and `idx_state_2d` is the 2D index referencing the 2D environment layout.
        """
        observation, reward, done, info = super().step(action)
            ## This is the step() method defined in MountainCarEnv => observation is the next state given as (x, v)
            ## Note that this method changes the 2D continuous-valued `state` attribute of the MountainCarEnv super class environment
            ## which is stored as a tuple --although by the reset() method it is stored as an array!
            ## That state is also called `state` but differs from the `state` defined in THIS object because the `state` in THIS object
            ## can only take finitely many values, namely the lower limits of the 2D cells into which the (x, v) space is discretized into.
            ## Nevertheless, `observation` above is an array because this is how the step() method returns the next state!

        # Call `observation` by its name, i.e. the continuous-valued `state`
        # (and take the opportunity to convert the array in `observation` to tuple, which is how we store the states in this class)
        state = tuple(observation)

        # We now change the 1D state index in discrete.DiscreteEnv AND the `state` attribute in this object
        # to match the new state of the super class environment, defined in `observation`.
        # which DIFFERS from the `state` attribute stored in the MountainCarEnv super class, as described in the above comment.
        # Note that the 1D state indexes the 2D finite-valued `state` defined in THIS object.
        self.setState(self.get_index_from_state(observation))

        # IMPORTANT: Check DONE in the DISCRETE state space!!!
        # This allows us to be "independent" of the way we discretize the continuous-valued state, which could be ROUNDing instead of TRUNCating
        # (see the self.discretize() method). If TRUNCating, this step is not necessary, but when ROUNDing it is because in that case the discrete state may be at the goal already
        # even if the continuous state is not (because of the "ROUNDING" approach that is now implemented (since Aug-2024).
        done = self.getState() in self.terminal_states

        if done:
            # Set the reward when reaching the goal to 0.0 so that we can more easily adjust the reward landscape below
            # (e.g. in order to assign a reward of +1 when reaching the goal and 0 everywhere else)
            # Note that in the original implementation (see gym/envs/classic_control/mountain_car.py) the reward is -1.0 at every step taken.
            # The documentation reads:
            # "The goal is to reach the flag placed on top of the right hill as quickly as possible,
            # as such the agent is penalised with a reward of -1 for each timestep it isn't at the goal and
            # is not penalised (reward = 0) for when it reaches the goal."
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
        # These are integer values in [0, nx-1] and [0, nv-1] respectively for position and velocity.
        # The order in which they are stored in idx_state_2d depends on the 2D layout used to represent the environment in plots,
        # i.e. on the value of the self._shape_display_names attribute (see self.discretize() for details).
        idx_state_2d = self.discretize(state)

        if return_continuous_state:
            return state, idx_state_2d, reward, done, info
        else:
            idx_state = self.getState()
            return idx_state, reward, done, info

    def discretize(self, state):
        """
        Discretizes into 2D indices the given continuous-valued state using the discretization parameters defined in the object (e.g. dx, dv, etc.)
        and self._shape_display_names, which defines the layout of position and velocity in the 2D environment layout used to visually represent the environment.

        Arguments:
        state: tuple
            The continuous-valued state = (x, v) to discretize.

        Return: numpy.ndarray
        Tuple containing the 2D indices representing the cell in the 2D environment layout where the given `state` falls.
        Whether the 2D indices represent first the position index and then the velocity index or the other way round depends on the 2D layout
        used to represent the environment, defined by self._shape_display_names.
        """
        idx_state_2d = np.array((-1, -1))
        # The state is ALWAYS (x, v) REGARDLESS of the value of self._shape_display_names
        x, v = state

        # We now use np.digitize() instead of my basic.discretize() function because my original discretize didn't
        # always work properly (border problems, e.g. when giving exactly the left bound of an interval, it returned the
        # interval to the left of the right one (x = -1.02 when -1.02 was the left bound of the interval index 1).
        # Also, even if I have deprecated my discretize() function --and replaced with a discretize() method that
        # calls np.digitize() under the hood-- we still call np.digitize() here in order to avoid the generation of
        # the intervals each time we need to discretize a value. Instead we store these intervals as part of the object.
        # IMPORTANT: np.digitize() returns a value between 1 and the length of bins (second parameter),
        # NOT between 0 and len-1 as one would have expected (considering how Python defines indices)!!!!!!

        # Use this if we want to ROUND the continuous-valued position and velocity
        #idx_state_2d[self.getPositionDimension()] = max(0, int(np.digitize(x + self.dx/2, self.positions) - 1))
        #idx_state_2d[self.getVelocityDimension()] = max(0, int(np.digitize(v + self.dv/2, self.velocities) - 1))

        # Use this if we want to TRUNCATE the continuous-valued position and velocity
        # Note that we bound any values going out of bound from the left to the leftmost position and velocity values
        idx_state_2d[self.getPositionDimension()] = max(0, int(np.digitize(x, self.positions) - 1))
        idx_state_2d[self.getVelocityDimension()] = max(0, int(np.digitize(v, self.velocities) - 1))

        assert 0 <= idx_state_2d[self.getPositionDimension()] <= self.nx - 1, "{}, {}".format(idx_state_2d[self.getPositionDimension()], self.nx-1)
        assert 0 <= idx_state_2d[self.getVelocityDimension()] <= self.nv - 1, "{}, {}".format(idx_state_2d[self.getVelocityDimension()], self.nv-1)

        return tuple(idx_state_2d)

    def undiscretize(self, idx_state_2d):
        """
        Returns the discrete-valued state (xd, vd) associated to the given 2D index of the 2D environment layout (defined by self._shape_display_names)

        Arguments:
        idx_state_2d: tuple, list or array
            The 2D index of the state to convert to a discrete-valued state, with the position index first or the velocity index first
            depending on the value of self._shape_display_names which defines precisely that order as either
            ("position", "velocity") or ("velocity", "position") and whose value is returned by self.getShapeDisplayNames().

        Return: tuple
        Tuple (xd, vd) containing the discrete-valued position at the first element and the discrete-valued velocity at the second element.
        These values represent the undiscretized state defined as the LOWER LIMIT of the discretized cell represented by the given 2D index
        discretizing the continuous state space.
        """
        dim_x = self.getPositionDimension()
        dim_v = self.getVelocityDimension()
        state_discrete = tuple([self.positions[idx_state_2d[dim_x]],
                                self.velocities[idx_state_2d[dim_v]]])
        return state_discrete

    def reshape_from_2d_to_1d(self, values_2d):
        """
        Converts the given 2D representation of the values to a 1D representation using C-like ordering of the 2D shape.
        This means that the 1D representation of the values will map the 2D cells of the 2D environment layout in the order in which
        the quantity (either position or velocity) placed along the HORIZONTAL direction (columns) varies first.
        This depends on the value of self._shape_display_names. More precisely:
        - when self._shape_display_names = ("position", "velocity") => the 1D representation of the state indexes the state by varying first the velocity,
        as velocity is laid out along the horizontal direction (columns).
        - when self._shape_display_names = ("velocity", "position") => the 1D representation of the state indexes the state by varying first the position,
        as position is laid out along the horizontal direction (columns).
        """
        return np.squeeze( values_2d.reshape(1, np.prod(values_2d.shape)) )

    def reshape_from_1d_to_2d(self, values_1d):
        "Converts a 1D representation of the values to a 2D representation using C-like ordering. See details in documentation for self.reshape_from_2d_to_1d()"
        return values_1d.reshape(self._shape_display)

    #-----------
    def get_index_from_state(self, state):
        "Returns the 1D state index corresponding to the given continuous-valued state"
        idx_state_2d = self.discretize(state)
        idx_state = self.get_index_from_index_2d(idx_state_2d)
        return idx_state

    def get_state_from_index(self, idx_state):
        "Returns the continuous-valued state from the given 1D state index. The conversion to continuous values is done by the undiscretize() method."
        idx_state_2d = self.get_index_2d_from_index(idx_state)
        state = self.undiscretize(idx_state_2d)
        return state
    #-----------

    #-----------
    def get_index_from_index_2d(self, idx_state_2d):
        """
        Returns the 1D state index from the 2D state indices

        Arguments:
        idx_state_2d: list, tuple or numpy array
            2D indices of the state.

        Return: int
        1D index of the state in the 1D representation of the 2D shape indexing the states.
        What each 1D index represents depends on the value of self._shape_display_names which can be either ("position", "velocity") or ("velocity", "position").
        If ("position", "velocity"), all the different positions will come first in the 1D representation of the 2D state space.
        """
        return index_multi2linear(idx_state_2d, self._shape_display)

    def get_index_2d_from_index(self, idx_state: int):
        "Returns the 2D index from the 1D index of the state"
        return index_linear2multi(idx_state, self._shape_display)
    #-----------

    #-----------
    def get_state_from_index_2d(self, idx_state_2d):
        """
        Returns the discrete-valued state (xd, vd) from the given 2D state index.
        This is an alias for the undiscretize() method.
        """
        return self.undiscretize(idx_state_2d)

    def get_index_2d_from_state(self, state):
        idx_state_2d = self.discretize(state)
        return idx_state_2d
    #-----------

    #-----------
    def get_state_discrete_from_state(self, state):
        """
        Returns the 2D discrete-valued state (xd, vd) from the given 2D continuous-valued state

        Arguments:
        state: tuple
            Continuous-valued state (x, v) to convert to a discrete-valued state (xd, vd).

        Return: tuple
        Duple representing the discrete-valued state associated to `state` in the form (xd, vd).
        """
        idx_state = self.get_index_from_state(state)
        state_discrete = self.get_state_discrete_from_index(idx_state)
        return state_discrete

    def get_state_discrete_from_index(self, idx_state: int):
        """
        Returns the discrete-valued state (xd, vd) from the given 1D state index.
        The conversion to continuous values is done by the undiscretize() method.

        Arguments:
        idx_state: int
            1D index representing the discretized state.

        Return: tuple
        Duple representing the discrete-valued state associated to `idx_state` in the form (xd, vd).
        """
        idx_state_2d = self.get_index_2d_from_index(idx_state)
        state_discrete = self.undiscretize(idx_state_2d)
        return state_discrete
    #-----------

    def get_position(self, state):
        """
        Returns the position of the given 2D continuous-valued state

        Arguments:
        state: tuple, list or array
            The state (x, v) for which the position is wished.

        Return: float
        The continuous-valued position associated to the given state.
        """
        return state[0]

    def get_velocity(self, state):
        """
        Returns the velocity of the given 2D continuous-valued state

        Arguments:
        state: tuple, list or array
            The state (x, v) for which the velocity is wished.

        Return: float
        The continuous-valued velocity associated to the given state.
        """
        return state[1]

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

    def get_adjacent_coming_from_states(self, state):
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

    #--- Getters
    def isStateContinuous(self):
        return self.state_is_continuous

    def getStateDimensions(self):
        """
        Returns the dimension of the discretized (x, v) states, i.e. first the number of discrete positions, and then the number of discrete velocities,
        REGARDLESS of the shape used to represent the state space (which is given by self.getShape()).
        """
        return (self.nx, self.nv)

    def getShape(self):
        return self.shape

    def getShapeDisplayNames(self):
        """
        Returns the names of the measures stored in each dimension of the 2D representation of the states

        Return: tuple
        Duple with the following elements:
        - name of the row labels of the 2D shape storing the states (e.g. "position")
        - name of the column labels of the 2D shape storing the states (e.g. "velocity")
        """
        return self._shape_display_names

    def getPositionDimension(self):
        "Returns the dimension in the 2D shape representation of the environment states containing the position."
        return self._shape_display_names.index("position")

    def getVelocityDimension(self):
        "Returns the dimension in the 2D shape representation of the environment states containing the velocity."
        return self._shape_display_names.index("velocity")

    def getPositionColor(self):
        "Returns the color to use in plots about the position"
        return "red"

    def getVelocityColor(self):
        "Returns the color to use in plots about the velocity"
        return "blue"

    def getState(self):
        """
        Returns the 1D state index associated to the current environment observation

        Return: int
        Index between 0 and nS-1, where nS is the number of states in the environment.
        """
        idx_state = super().getState()  # This calls the getState() method in the discrete.DiscreteEnv class (since the MountainCarEnv class does not have any getState() method defined)
        assert self.get_index_from_state(self.state) == idx_state, \
            "The 2D continuous-valued state (x, v) stored in the MountainCarEnv super class ({}) falls in the cell represented by the 1D state index stored in the discrete.DiscreteEnv object ({})." \
            .format(self.state, idx_state, self.get_index_from_state(self.state))
            # Note that we assert that the 2D continuous-valued state converted to 1D state index match the current 1D state index stored in the environment
            # and NOT the other way round because an INFINITE number of 2D continuous-valued states are mapped to the same 1D state index, whereas only ONE
            # 2D continuous-valued state is mapped to the 1D state index conversion to 2D, namely the 2D continuous-valued state that is exactly at one of the
            # grid points defined by the discretization of the 2D state landscape (position, velocity), a.k.a. discrete-valued state `state_discrete`,
            # that is linearly indexed by the 1D state index.
            # Note that an eventual assertion of the 1D to 2D conversion (i.e. the opposite of the above assertion) would fail soon after the environment is reset,
            # because in the default process of the self.reset() method, a continuous-valued 2D state is first chosen at random and then a 1D state index mapped to it:
            # if we converted this 1D state index back to a 2D state, the result would NOT be the 2D state selected at random in the 2D continuous-valued space!
        return idx_state

    def getStateFromIndex(self, idx_state: int, simulation=True):
        "Returns the state associated to the given 1D state index. When simulation=True, such state coincides with the 1D index, because the environment is a discrete-state environment"
        state_simulation = idx_state
        if simulation:
            state = state_simulation
        else:
            state = self.get_state_from_index(idx_state)
        return state

    def getStateIndicesFromIndex(self, idx_state: int):
        "Returns the 2D state indices associated to the 1D state index"
        return self.get_index_2d_from_index(idx_state)

    def getIndexFromState(self, state, simulation=True):
        """
        Returns the 1D index representation of the given state. when simulation=True, `state` must be an integer value representing a 1D index as well,
        as the state used for simulations is a 1D index (as the state space is discrete, so we can do that).
        When simulation=False, `state` is expected to be a tuple containing the continuous-valued state.
        """
        if simulation:
            assert is_integer(state), f"The simulation state must be an integer: {state}"
            idx_state = state
        else:
            idx_state = self.get_index_from_state(state)
        return idx_state

    def getReward(self, s):
        "Returns the reward received when visiting a state given as a 1D index representation of the state"
        return super().getReward(s)  # This calls the getReward() method in the EnvironmentDiscrete class (since the MountainCarEnv class does not have any getReward() method defined)

    def getNumStates(self):
        return self.nS

    def getNumActions(self):
        return self.nA

    def isTerminalState(self, idx_state: int):
        state = self.get_state_from_index(idx_state)
        x = self.get_position(state)
        #print("Check terminal state: {} (dx={})".format(state, self.dx))
        return x >= self.goal_position

    #--- Setters
    def setState(self, idx_state):
        """
        Sets the state of the environment, both the 1D state index (attribute `s` of EnvironmentDiscrete super class)
        and the continuous-valued 2D state (x, v) representing the position and velocity of the mountain car (attribute `state` of MountainCarEnv super class).

        Arguments:
        idx_state: int
            1D state index to set.
        """
        # Set the state (1D index) in the EnvironmentDiscrete environment
        super().setState(idx_state)
        # Set the state (continuous-valued position and velocity (x, v)) in the MountainCarEnv environment
        self.state = self.get_state_from_index(idx_state)

    #--- Plot methods
    def plot_trajectory_gif(self, trajectory):
        "Generates a GIF showing the states through which the mountain car has been during learning"
        positions = [state[0] for state in trajectory]
        def mountain_car_track(x):
            return np.sin(3*x)

        fig, ax = plt.subplots()
        x = np.linspace(-1.2, 0.6, 1000)
        y = mountain_car_track(x)
        ax.plot(x, y, 'k')

        ax.set_xlim(-1.2, 0.6)
        ax.set_ylim(-1.2, 1.2)  # Maybe change this to see properly

        cmap = cm.get_cmap('coolwarm')
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
        print("GIF saved to: ./mountain_car.gif")

    def plot_values(self, values, ax=None, figsize=(8, 8), cmap="Blues", vmin=None, vmax=None, alpha=1.0):
        "Plots values, defined for EVERY state, given as a 1D array on the 2D shape used to represent the environment as an image. Returns the axis object and the image object."
        values_2d = self.reshape_from_1d_to_2d(values)
        if ax is None:
            new_figure = True
            ax = plt.figure(figsize=figsize).subplots(1, 1)
        else:
            new_figure = False

        # Plot the values by starting from the lower left corner and going first horizontally showing the values on the first row of `values_2d`,
        # then move up one line in the image and plot the values on the second row, etc.
        img = ax.imshow(values_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
        plt.colorbar(img)

        if new_figure:
            self._finalize_plot(ax)

        return ax, img

    def plot_points(self, points, style=".-", figsize=(8, 8), color="blue", cmap=None, markersize=5, ax=None):
        """
        Plots a set of points representing simulation states (i.e. recorded by simulations) in the 2D discrete-state shape representation of the environment.

        Points are connected by lines if requested by the code in `style`.

        Arguments:
        points: array-like
            Points to plot representing states obtained from simulations.

        style: (opt) str
            Style of the plotted points, it's the symbol used in matplotlib.pyplot.plot(), e.g. ".-" or "." or "x".
        """
        if ax is None:
            new_figure = True
            ax = plt.figure(figsize=figsize).subplots(1, 1)
        else:
            new_figure = False

        n_points = len(points)
        if cmap is not None:
            colors = cm.get_cmap(cmap, lut=n_points)
        for i, point in enumerate(points[:-1]):
            # Convert the point to a 2D index representing the entry of the point in the 2D shape representation of the environment
            idx_point = self.getIndexFromState(point)
            idx_point_next = self.getIndexFromState(points[i+1])
            idx_point_2d = self.get_index_2d_from_index(idx_point)
            idx_point_2d_next = self.get_index_2d_from_index(idx_point_next)
            _color = colors(i / n_points) if cmap is not None else color
            if self.getShapeDisplayNames() == ("velocity", "position"):
                # Velocity is on the VERTICAL axis and position on the HORIZONTAL axis
                ax.plot( [idx_point_2d[self.getPositionDimension()], idx_point_2d_next[self.getPositionDimension()]],
                         [idx_point_2d[self.getVelocityDimension()], idx_point_2d_next[self.getVelocityDimension()]],
                         style, color=_color, markersize=markersize)
            else:
                # Velocity is on the HORIZONTAL axis and position on the VERTICAL axis
                ax.plot( [idx_point_2d[i - 1, self.getVelocityDimension()], idx_point_2d[i, self.getVelocityDimension()]],
                         [idx_point_2d[i - 1, self.getPositionDimension()], idx_point_2d[i, self.getPositionDimension()]],
                         style, color=_color, markersize=markersize)

        if new_figure:
            self._finalize_plot(ax)

        return ax

    def add_labels(self, ax, labels, color="black", fontsize=14):
        "Adds labels to an existing plot assumed to contain an image, e.g. generated by self.plot_values()"
        labels_2d = labels.reshape(self._shape_display)
        for y in range(self._shape_display[0]):
            for x in range(self._shape_display[1]):
                # Recall the y value indexes the row of the matrix shown in the image (vertical axis of the plot)
                # and the x value indexes the column of the matrix shown in the image (horizontal axis of the plot)
                # That's why the text coordinates (of state_counts) are (y, x) to be placed at coordinates (x, y).
                ax.text(x, y, "{:.0f}".format(labels_2d[y, x]), color=color, fontsize=fontsize, horizontalalignment="center", verticalalignment="center")

    def _finalize_plot(self, ax):
        "Finalizes the plot created with self.plot_values() or self.plot_points() with more informed axes"
        # Recall that _shape_display_names indicates a tuple with the concept plotted on the Y axis and the concept plotted on the X axis.
        ax.set_xlabel(self._shape_display_names[1])
        ax.set_ylabel(self._shape_display_names[0])

        # Make the X-axis and Y-axis cover the whole spectrum of positions and velocities
        # Recall that the X and Y axis values are the INDICES of the positions and velocities respectively, NOT their actual position and velocity values.
        # We comment this out because it actually doesn't really do what we want, namely to have e.g. the velocity axis values symmetrical...
        if self._shape_display_names[0] == "velocity":
            # Velocity is plotted on the VERTICAL axis
            ax.set_xlim((ax.get_xlim()[0], len(self.get_positions()) - 0.5))
            ax.set_ylim((ax.get_ylim()[0], len(self.get_velocities()) - 0.5))
        else:
            # Velocity is plotted on the HORIZONTAL axis
            ax.set_xlim((ax.get_xlim()[0], len(self.get_velocities()) - 0.5))
            ax.set_ylim((ax.get_ylim()[0], len(self.get_positions()) - 0.5))

        # Show the actual position and velocity values as ticklabels
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        xticks = ax.get_xticks()[1:-1].astype(int)
        yticks = ax.get_yticks()[1:-1].astype(int)
        if self._shape_display_names[0] == "velocity":
            # Velocity is plotted on the VERTICAL axis
            xticklabels = ["{:.3f}".format(s) for s in self.get_positions()[xticks]]
            yticklabels = ["{:.3f}".format(s) for s in self.get_velocities()[yticks]]
        else:
            # Velocity is plotted on the HORIZONTAL axis
            xticklabels = ["{:.3f}".format(s) for s in self.get_velocities()[xticks]]
            yticklabels = ["{:.3f}".format(s) for s in self.get_positions()[yticks]]
        ax.xaxis.set_ticklabels([ax.xaxis.get_ticklabels()[0]] + xticklabels + [ax.xaxis.get_ticklabels()[-1]]);
        ax.tick_params(axis='x', labelrotation=90)
        ax.yaxis.set_ticklabels([ax.yaxis.get_ticklabels()[0]] + yticklabels + [ax.yaxis.get_ticklabels()[-1]])
