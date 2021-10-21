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
- reset_state_counts()
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
        self.alpha = alpha
        self.adjust_alpha = adjust_alpha
        self.alpha_update_type = alpha_update_type
        self.alpha_min = alpha_min  # Used when adjust_alpha=True

        # Information of the states and rewards on which learning takes place
        # (so that it can be retrieved by the user if needed as a piece of information)
        self.states = []
        self.rewards = []

        # Information about the historical learning rates
        self.alpha_mean = []    # Average alpha
        self._alphas_used = []  # List of alphas used during the learning process (for now it's one alpha per state, but these may vary in the future)

    def reset(self, reset_value_functions=False):
        """
        Resets the variables that store information about the learning process

        Parameters:
        reset_value_functions: bool, optional
            Whether to reset all the value functions to their initial estimates.
        """
        self.alpha_mean = []
        self._alphas_used = []

        # Reset the attributes that keep track of visited states
        self.reset_state_counts()

        if reset_value_functions:
            # Only reset the initial estimates of the value functions when requested
            self.reset_value_functions()
