# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:34:27 2021

@author: Daniel Mastropietro
@description: Monte-Carlo learners on continuing tasks
"""

from enum import unique, Enum

from .. import Learner

@unique  # Unique enumeration values (i.e. on the RHS of the equal sign)
class AlphaUpdateType(Enum):
    FIRST_STATE_VISIT = 1
    EVERY_STATE_VISIT = 2


class LeaMC(Learner):
    """
    Monte Carlo learning algorithm using step size `alpha` and discount `gamma`
    applied to a generic environment defined in the gym.Env class of openAI.

    Arguments:
    env: gym.Env
        The environment where learning takes place.
    """

    def __init__(self, env, alpha=0.1, gamma=1.0,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                 alpha_min=0.,
                 debug=False):
        super().__init__(env, alpha, adjust_alpha, alpha_update_type, alpha_min)
        self.debug = debug

        self.gamma = gamma
        self.dict_state_counts = dict()
        # Value functions and related quantities
        self.G = 0.0    # Average reward during the queue evolution until max simulation time
        self.V = 0.0
        self.Q = 0.0

        super().reset(reset_value_functions=True)

    def reset_state_counts(self):
        "Resets the dictionary containing the number of times a state is visited"
        self.dict_state_counts = dict()

    def reset_value_functions(self):
        # Average reward during the queue evolution until max simulation time
        self.G = 0.0

        # Value functions
        self.V = 0.0
        self.Q = 0.0

    def update_state_counts(self, state):
        "Updates the number of times the given state has been visited during the learning process"
        # Create a string version of the state, so that we can store lists as dictionary keys
        # which are actually not accepted as dictionary keys (with the error message "list type is unhashable").
        # e.g. [3, 1, 5] becomes '[3, 1, 5]'
        state_str = str(state)
        if state_str in self.dict_state_counts.keys():
            self.dict_state_counts[state_str] += 1
        else:
            self.dict_state_counts[state_str] = 1

    def updateAverageG(self, t, reward):
        if t == 0:
            self.G = reward
        else:
            self.G = ( (t - 1) * self.G + reward) / t

    def learn(self, t):
        """
        Updates the value function with the observed average return G

        Arguments:
        t: float
            Discrete time at which learning takes place.
        """
        self.V = self.G

    def getV(self):
        return self.V

    def getQ(self):
        return self.Q
