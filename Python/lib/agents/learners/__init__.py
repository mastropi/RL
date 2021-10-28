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


class Learner:
    """
    Class defining methods that are generic to ALL learners.

    NOTE: Before using any learner the simulation program should call the reset() method!
    Otherwise, the simulation process will most likely fail (because variables that are
    defined in the reset method to track the simulation process will not be defined).
    In addition, the *specific* Learner constructor should NOT call the reset() method
    because the reset method would then be called twice: once when the learner is constructed
    and once prior to the first simulation.
    """

    def __init__(self, env, alpha,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                 alpha_min=0.):
        """
        Parameters:
        env: gym.Env
            Environment where learning takes place.
            The environment needs not be "discrete" in the sense the gym package uses discrete, namely that there is
            a pre-defined number of states (as is the case in the EnvironmentDiscrete environment of gym).

        alpha: positive float
            Learning rate.

        alpha_update_type: AlphaUpdateType
            How alpha is updated, e.g. AlphaUpdateType.FIRST_STATE_VISIT, AlphaUpdateType.EVERY_STATE_VISIT
            This value defines the denominator when updating alpha for each state as alpha/n, where alpha
            is the initial learning rate (passed as parameter alpha) and n is the number of FIRST or EVERY visit
            to the state, depending on the value of the alpha_update_type parameter.
        """
        self.env = env
        self.alpha = alpha                          # Starting and maximum learning rate
        self.adjust_alpha = adjust_alpha
        self.alpha_update_type = alpha_update_type
        self.alpha_min = alpha_min  # Used when adjust_alpha=True

        # Observed state, action, and reward at the latest learning time
        self.state = None           # S(t): state BEFORE taking the action A(t) and receiving reward R(t+1)
        self.action = None          # A(t): action taken at state S(t)
        self.reward = None          # R(t+1): reward received by the agent when taking action A(t) on state S(t)

        # Trajectory history: information of the states, actions, and rewards on which learning takes place
        # (so that it can be retrieved by the user if needed as a piece of information)
        self.states = []
        self.actions = []
        self.rewards = []

        self.dict_state_action_counts = dict()

        # Information about the historical learning rates
        self.alpha_mean = []    # Average alpha
        self.alphas = []        # List of alphas used during the learning process (for now it's one alpha per state, but these may vary in the future)

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

    def update_alpha(self, state, action):
        """
        Updates the learning rate given the visited count of the given state and action

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
            time_divisor = np.max([1, state_action_count])
            print("\tUpdating alpha... state={}, action={}: time divisor = {}".format(state, action, time_divisor))
            alpha = np.max( [self.alpha_min, self.alpha / time_divisor] )
            print("\t\tstate {}: alpha: {}".format(state, alpha))
        else:
            alpha = self.alpha

        return alpha

    def update_trajectory(self, state, action, reward):
        """
        Records the state S(t) on which an action is taken, the taken action A(t), and the observed reward R(t+1),
        and updates the history of this trajectory.
        """
        self.state = state
        self.action = action
        self.reward = reward
        self._update_trajectory_history()

    def _update_trajectory_history(self):
        "Updates the history of the visited states and observed rewards during the simulation"
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

    def getCount(self, state, action):
        "Returns the number of times the given state and action has been visited during the learning process"
        key = str( (state, action) )
        return self.dict_state_action_counts.get(key, 0)

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions

    def getRewards(self):
        return self.rewards
