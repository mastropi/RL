# -*- coding: utf-8 -*-
"""
Created on 09 Oct 2021

@author: Daniel Mastropietro
@description: Definition of the generic classes for policies.
"""

from enum import Enum, unique

import numpy as np

from Python.lib.utils.basic import is_scalar


@unique
class PolicyTypes(Enum):
    ACCEPT = 1
    ASSIGN = 2

@unique
class AcceptPolicyType(Enum):
    THRESHOLD = 1
    TRUNK_RESERVATION = 2


class GenericParameterizedPolicyTwoActions:
    """
    Class that holds the generic attributes and methods of a parameterized policy with two possible actions

    It is assumed that subclasses define the following methods:
    - getPolicyForAction(a,s): returns the probability of taking action `a` when the environment is at state `s`.
    - getGradient(a,s): returns the gradient of the policy for action `a` when the environment is at state `s`.

    Arguments:
    env: Environment
        A generic environment object accepting two actions which must have the following methods defined:
        - np_random() which generates a random value between 0 and 1, used to choose an action for the policy.
        - getActions() which returns all possible actions to be done when interacting with the environment.

    theta: float
        Initial value for the parameter of the parameterized policy, which should be a real unidimensional value
        because it is a threshold value between two possible actions (as the name of the current class indicates is the case).
        The policy to which this parameter applies should define two possible actions via the Actions enum.
    """

    def __init__(self, env, theta: float):
        self.env = env
        assert set([0, 1]).issubset(set(env.getActions().__members__.values())), \
            "The environment on which the two-action policy acts must have action-index 0 and action-index 1 as two possible actions ({})".format(env.getActions())

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

        state: Undefined
            State of the environment stored in the object on which the policy acts.
            The state type is highly dependent on the environment type.
            If the environment on which this policy acts is a queue environment, the state is most likely a tuple
            with two elements: the queue state (which can be a scalar for single-server systems or a tuple for
            multi-server systems or loss networks) and the class of the arriving job.

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
