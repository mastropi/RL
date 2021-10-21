# -*- coding: utf-8 -*-
"""
Created on 09 Oct 2021

@author: Daniel Mastropetro
@description: Definition of the attributes and methods that are needed for any parameterized policy with two possible actions
"""

import numpy as np

from utils.basic import as_array
from environments.queues import Actions


class GenericParameterizedPolicyTwoActions:
    """
    Class that holds the generic attributes and methods of a parameterized policy with two possible actions

    It is assumed that subclasses define the following methods:
    - getPolicyForAction(a): returns the probability of taking action `a`.
    - getGradient(a): returns the gradient of the policy at action `a`.

    Arguments:
    env: a generic environment object
        The object must have the following methods defined:
        - np_random() which generates a random value between 0 and 1.

    theta: float or list or array
        Initial values for the parameters of the parameterized policy.
        The parameter is converted ALWAYS to an numpy array (i.e. instance of numpy.ndarray)
    """

    def __init__(self, env, theta: float or list or np.ndarray):
        self.env = env

        self.setThetaParameter(theta)
        # We store the start value of theta so that we can reset it when resetting the policy
        self.theta_start = self.theta

    def reset(self):
        self.setThetaParameter(self.theta_start)

    def choose_action(self):
        """
        Choose an action, either ActionTypes.REJECT or ActionTypes.ACCEPT based on the policy
        evaluated at the current state of the environment.

        Return: ActionTypes
            Action chosen by the policy.
        """
        prob_action_1 = self.getPolicyForAction(1)
        if prob_action_1 == 1.0:
            return Actions(1)
        elif prob_action_1 == 0.0:
            return Actions(0)
        else:
            # Choose action 1 with probability equal to prob_action_1
            action_num = int( self.env.np_random() < prob_action_1 )
            return Actions(action_num)

    def getGradientLog(self, action):
        "Returns the gradient of the log policy at the given action and current state of the environment"
        policy_value = self.getPolicyForAction(action)
        if policy_value > 0:
            return self.getGradient(action) / policy_value
        elif self.getGradient(action) == 0.0:
            return 0.0
        else:
            return np.nan

    #----- GETTERS -----#
    def getThetaParameter(self):
        return self.theta

    #----- SETTERS -----#
    def setThetaParameter(self, theta):
        "Sets the theta parameter of the parameterized policy"
        # Convert theta to a numpy array
        self.theta = as_array(theta)
