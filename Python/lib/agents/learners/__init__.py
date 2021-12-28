# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:37:08 2021

@author: Daniel Mastropietro
@description: Definition of classes that are common to all generic learners, where 'generic' means that they do not
belong to e.g. a continuing or episodic learning task.

This class stores the following attributes as part of the object:
- env: the environment on which the learning takes place
- alpha: the learning rate

Classes inheriting from this class should define the following methods:
- reset_supporting_attributes()
- reset_value_functions()
"""

from enum import Enum, unique

import numpy as np

MIN_COUNT = 1  # Minimum state count to start shrinking alpha


@unique  # Unique enumeration values (i.e. on the RHS of the equal sign)
class AlphaUpdateType(Enum):
    FIRST_STATE_VISIT = 1
    EVERY_STATE_VISIT = 2


class GenericLearner:
    """
    Class defining methods that are generic to ALL learners.

    NOTE: Before using any learner the simulation program should call the reset() method!
    Otherwise, the simulation process will most likely fail (because variables that are
    defined in the reset method to track the simulation process will not be defined).
    In addition, the *specific* Learner constructor should NOT call the reset() method
    because the reset method would then be called twice: once when the learner is constructed
    and once prior to the first simulation.
    """

    def __init__(self, env, alpha :float,
                 adjust_alpha=False,
                 min_count_to_update_alpha=0, min_time_to_update_alpha=0,
                 alpha_min=0.):
        """
        Parameters:
        env: gym.Env
            Environment where learning takes place.
            The environment needs not be "discrete" in the sense the gym package uses discrete, namely that there is
            a pre-defined number of states (as is the case in the EnvironmentDiscrete environment of gym).

        alpha: positive float
            Initial learning rate.

        min_count_to_update_alpha: int
            Minimum count of a state-action pair at which alpha starts to be updated by the update_learning_rate(s,a) method.

        min_time_to_update_alpha: int
            Minimum learning time step at which alpha starts to be updated by the update_learning_rate_by_episode() method.
        """
        self.env = env
        self.alpha = alpha          # Initial and maximum learning rate
        self.adjust_alpha = adjust_alpha
        self.min_count_to_update_alpha = min_count_to_update_alpha
        self.min_time_to_update_alpha = min_time_to_update_alpha
        self.alpha_min = alpha_min  # Used when adjust_alpha=True

        # Observed state, action, and reward at the latest learning time
        self.state = None           # S(t): state BEFORE taking the action A(t) and receiving reward R(t+1)
        self.action = None          # A(t): action taken at state S(t)
        self.reward = None          # R(t+1): reward received by the agent when taking action A(t) on state S(t)

        # Trajectory history: information of the times, states, actions, and rewards stored as part of the learning process
        # (so that we can be retrieve it for e.g. analysis or plotting)
        self.times = []         # This time is expected to be a *discrete* time, indexing the respective states, actions and rewards
        self.states = []        # Stores S(t), the state at each time t stored in `times`
        self.actions = []       # Stores A(t), the action taken at state S(t)
        self.rewards = []       # Stores R(t) = R(S(t), A(t)), the reward received after taking action A(t) on state S(t)

        self.dict_state_action_counts = dict()

        # Learning time and learning rate at that time
        self._time = 0              # Int: this value is expected to be updated by the user, whenever learning takes place
        self._alpha = self.alpha    # Float: current learning rate (at the learning time `self._time`)

        # Information about the historical learning rates
        self.alpha_mean = []    # Average alpha
        self.alphas = []        # List of alphas used during the learning process.

    def reset(self, reset_value_functions=False, reset_trajectory=False, reset_counts=False):
        """
        Resets the variables that store information about the learning process

        Parameters:
        reset_value_functions: bool, optional
            Whether to reset all the value functions to their initial estimates.

        reset_counts: bool, optional
            Whether to reset the counters (of e.g. the states or the action-states).
        """
        self.alpha_mean = []
        self.alphas = []

        # Reset the latest learning state, action, and reward
        self.state = None
        self.action = None
        self.reward = None

        # Reset the attributes that keep track of visited states
        self.reset_supporting_attributes()
        self.reset_return()

        if reset_value_functions:
            # Only reset the initial estimates of the value functions when requested
            self.reset_value_functions()

        if reset_trajectory:
            self.reset_trajectory()

        if reset_counts:
            self.dict_state_action_counts = dict()

    def reset_trajectory(self):
        self.times = []
        self.states = []
        self.actions = []
        self.rewards = []

    def reset_supporting_attributes(self):
        # Reset supporting learning attributes (e.g. alphas, state counts, etc.). Method to be overridden by subclasses
        pass

    def reset_return(self):
        # Reset the observed return. Method to be overridden by subclasses
        pass

    def reset_value_functions(self):
        # Method to be overridden by subclasses
        pass

    def store_learning_rate(self):
        """
        Stores the current learning rate alpha that is supposedly used for learning when this method is called

        Note that the values of the self.alphas list may be indexed by many different things...
        For instance, if learning takes place at every visited state-action, then there will probably be an alpha
        for each visited state-action. But if learning happens at the end of an episode, the stored alphas
        will probably be indexed by episode number.
        It is however difficult to store the index together with the alpha, because the index structure may change
        from one situation to the other, as just described.
        """
        self.alphas += [self._alpha]

    def update_learning_time(self):
        """
        Increments the count of learning time by one.

        This method is expected to be called by the user whenever they want to record that learning took place.
        It can be used to update the learning rate alpha by the number of learning episodes
        (see update_learning_rate_by_episode()).
        """
        self._time += 1

    def update_learning_rate(self, state, action):
        """
        Updates the learning rate at the current learning time, given the visit count of the given state and action

        The update is done ONLY when the attribute self.adjust_alpha is True.

        Arguments:
        state: Environment dependent
            State visited by the environment used to retrieve the visited count to update alpha.

        action: Environment dependent
            Action received by the environment used to retrieve the visited count to update alpha.

        Return: float
        The updated value of alpha based on the visited count of the state and action.
        """
        #with np.printoptions(precision=4):
        #    print("Before updating alpha: state {}: state_action_count={:.0f}, alpha>={}: alpha={}\n{}" \
        #          .format(state, self.dict_state_action_count[str(state)], self.alpha_min, self.alphas[str(state)]))
        if self.adjust_alpha:
            state_action_count = self.getCount(state, action)
            time_divisor = np.max([1, state_action_count - self.min_count_to_update_alpha])
                ## Note: We don't sum anything to the difference count - min_count because we want to have the same
                ## time_divisor value when min_count = 0, in which case the first update occurs when count = 2
                ## (yielding time_divisor = 2), as when min_count > 0 (e.g. if min_count = 5, then we would have the
                ## first update at count = 7, with time_divisor = 2 as well (= 7 - 5)).
            print("\tUpdating alpha... state={}, action={}: time divisor = {}".format(state, action, time_divisor))
            alpha_prev = self._alpha
            self._alpha = np.max( [self.alpha_min, self.alpha / time_divisor] )
            print("\t\tstate {}: alpha: {} -> {}".format(state, alpha_prev, self._alpha))
        else:
            self._alpha = self.alpha

        return self._alpha

    def update_learning_rate_by_episode(self):
        """
        Updates the learning rate at the current learning time by the number of times learning took place already,
        independently of the visit count of each state and action.

        The update is done ONLY when the attribute self.adjust_alpha is True.

        Return: float
        The updated value of alpha.
        """
        if self.adjust_alpha:
            time_divisor = np.max([1, self._time - self.min_time_to_update_alpha])
                ## See comment in method update_learning_rate() about not adding any constant to the difference time - min_time
            print("\tUpdating alpha by learning episode: time divisor = {}".format(time_divisor))
            alpha_prev = self._alpha
            self._alpha = np.max( [self.alpha_min, self.alpha / time_divisor] )
            print("\t\talpha: {} -> {}".format(alpha_prev, self._alpha))
        else:
            self._alpha = self.alpha

        return self._alpha

    def update_trajectory(self, t, state, action, reward):
        """
        Records the state S(t) on which an action is taken, the taken action A(t), and the observed reward R(t)
        for going to state S(t+1), and updates the history of this trajectory.

        Arguments:
        t: int
            Time at which the given, state, action, and reward occur.
        """
        self.state = state
        self.action = action
        self.reward = reward
        self._update_trajectory_history(t)

    def _update_trajectory_history(self, t):
        """
        Updates the history of the visited states, actions taken, and observed rewards at the given simulation time

        Arguments:
        t: int
            Time to which the stored state, action, and reward are associated.
        """
        self.times += [t]
        self.states += [self.state]
        self.actions += [self.action]
        self.rewards += [self.reward]

    def update_counts(self, state, action):
        "Updates the number of times the given state and action have been visited during the learning process"
        # Create a string version of the state, so that we can store lists as dictionary keys
        # which are actually not accepted as dictionary keys (with the error message "list type is unhashable").
        # e.g. [3, 1, 5] becomes '[3, 1, 5]'
        state_action_str = str( (state, action) )
        if state_action_str in self.dict_state_action_counts.keys():
            self.dict_state_action_counts[state_action_str] += 1
        else:
            self.dict_state_action_counts[state_action_str] = 1

    def getAverageLearningRates(self):
        return self.alpha_mean

    def getLearningRates(self):
        return self.alphas

    def getLearningRate(self):
        return self._alpha

    def getLearningTime(self):
        "Returns the number of times learning took place according to the learning time attribute"
        return self._time

    def getCount(self, state, action):
        "Returns the number of times the given state and action has been visited during the learning process"
        key = str( (state, action) )
        return self.dict_state_action_counts.get(key, 0)

    def getTimes(self):
        return self.times

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions

    def getRewards(self):
        return self.rewards
