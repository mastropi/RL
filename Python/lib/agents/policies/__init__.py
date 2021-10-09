# -*- coding: utf-8 -*-
"""
Created on 09 Oct 2021

@author: Daniel Mastropetro
@description: Definition of the attributes and methods that are needed for any parameterized policy with two possible actions
"""

import numpy as np


class GenericParameterizedPolicyTwoActions:
    """
    Class that holds the generic attributes and methods of a parameterized policy with two possible actions

    It is assumed that subclasses define the following methods:
    - getPolicy(a): returns the probability of taking action `a`.
    - getGradient(a): returns the gradient of the policy at action `a`.

    Arguments:
    env: a generic environment object
        The object must have the following methods defined:
        - np_random() which generates a random value between 0 and 1.

    theta: list
        List of the parameters of the parameterized policy.
    """

    def __init__(self, env, theta: list):
        self.env = env
        self.theta = theta

    def choose_action(self):
        """
        Choose an action, either 0 (reject) or 1 (accept) based on the policy
        evaluated at the current state of the environment.
        """
        prob_action_1 = self.getPolicyForAction(1)
        if prob_action_1 == 1.0:
            return 1
        elif prob_action_1 == 0.0:
            return 0
        else:
            # Choose action 1 with probability equal to prob_action_1
            return self.env.np_random() < prob_action_1

    def getGradientLog(self, action):
        "Returns the gradient of the log policy at the given action and current state of the environment"
        policy_value = self.getPolicyForAction(action)
        if policy_value > 0:
            return self.getGradient(action) / policy_value
        elif self.getGradient(action) == 0.0:
            return 0.0
        else:
            return np.nan
