# -*- coding: utf-8 -*-
"""
Created on 09 Oct 2021

@author: Daniel Mastropietro
@description: Definition of the attributes and methods that are needed for any parameterized policy with two possible actions
"""

from enum import Enum, unique

import numpy as np

from Python.lib.utils.basic import is_scalar


@unique
class PolicyTypes(Enum):
    ACCEPT = 'accept'
    ASSIGN = 'assign'


class GenericParameterizedPolicyTwoActions:
    """
    Class that holds the generic attributes and methods of a parameterized policy with two possible actions

    It is assumed that subclasses define the following methods:
    - getPolicyForAction(a,s): returns the probability of taking action `a` when the environment is at state `s`.
    - getGradient(a,s): returns the gradient of the policy for action `a` when the environment is at state `s`.

    Arguments:
    env: Environment
        A generic environment object accepting two actions which must have the following methods defined:
        - np_random() which generates a random value between 0 and 1.
        - getActions() which returns all possible actions to be done when interacting with the environment.

    theta: float
        Initial value for the parameter of the parameterized policy, which should be a real unidimensional value
        because it is a threshold value between two possible actions (as the name of the current class indicates is the case).
        The policy should define two possible actions via the Actions enum.
    """

    def __init__(self, env, theta: float):
        self.env = env

        # Store the given theta as start theta value so that we can reset it to that value when resetting the policy
        if not is_scalar(theta):
            raise ValueError("The theta parameter must be scalar ({})".format(theta))
        self.theta_start = theta

        # Store the start theta in the object
        self.setThetaParameter(self.theta_start)

    def reset(self):
        self.setThetaParameter(self.theta_start)

    def choose_action(self, state):
        """
        Choose an action given the state: either ActionTypes.REJECT or ActionTypes.ACCEPT based on the policy
        evaluated at the given state.

        state: State(?)
            State of the environment stored in the object on which the policy acts.
            The state type is highly dependent on the environment type.

        Return: Actions
            Action chosen by the policy.
        """
        actions = self.env.getActions()
        prob_action_1 = self.getPolicyForAction(actions(1), state)
        if prob_action_1 == 1.0:
            return actions(1)
        elif prob_action_1 == 0.0:
            return actions(0)
        else:
            # Choose action 1 with probability equal to prob_action_1
            action_num = int( self.env.np_random() < prob_action_1 )
            return actions(action_num)

    def getGradientLog(self, action, state):
        """
        Returns the gradient of the log policy at the given action and given state of the environment" \

        action: Environment dependent
            Action in the action space of the environment defined in the object on which the computation
            of the policy gradient's log is requested.

        state: Environment dependent
            State of the environment defined in the object on which the computation
            of the policy gradient's log is requested.
        """
        gradient = self.getGradient(action, state)
        policy_value = self.getPolicyForAction(action, state)
        if policy_value > 0:
            return gradient / policy_value
        elif gradient == 0.0:
            return 0.0
        else:
            # Division by 0 would have occurred!
            return np.nan

    #----- GETTERS -----#
    def getEnv(self):
        return self.env

    def getThetaParameter(self):
        return self.theta

    #----- SETTERS -----#
    def setThetaParameter(self, theta):
        "Sets the theta parameter of the parameterized policy"
        if not is_scalar(theta):
            raise ValueError("Parameter theta to set must be scalar: {}".format(theta))
        self.theta = theta

    #----- ABSTRACT METHODS (to be defined by subclasses) -----#
    def getGradient(self, action, state):
        raise NotImplementedError

    def getPolicyForAction(self, action, state):
        raise NotImplementedError
