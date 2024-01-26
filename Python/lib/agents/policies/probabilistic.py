# -*- coding: utf-8 -*-
"""
Created on Sat Dec  02 21:37:07 2023

@author: Daniel Mastropietro
@description: Generic probabilistic policies constructed on environments defined in the environments module.
Each policy should have the following methods:
- choose_action(state) --> it returns an index on the action space with the action to take by the agent at a given state.
"""

import numpy as np

from Python.lib.environments import EnvironmentDiscrete


class PolGenericDiscrete():
    """
    Random walk policy for a discrete environment defined with the DiscreteEnv class
    of openAI's gym module.

    Arguments:
    env: EnvironmentDiscrete
        The environment where the policy acts.

    policy: dict of list
        Dictionary indexed by states in the environment (which are indexed as 0, 1, ...)
        whose value is a list containing the probability of each possible action in the environment
        associated to the given state.

    policy_default: (opt) list
        List indexed by all possible actions in the environment containing the default probability of each action
        to be assigned for a state not present in the `policy` dictionary.
        When given, the states
        default: None
    """

    def __init__(self, env: EnvironmentDiscrete, policy: dict, policy_default: list=None):
        if not issubclass(env.__class__, EnvironmentDiscrete):
            raise TypeError("The environment must be of type {} from the {} module ({})" \
                            .format(EnvironmentDiscrete.__name__, EnvironmentDiscrete.__module__, env.__class__))
        if not isinstance(policy, dict):
            raise TypeError(f"The given policy must be a dictionary (type={type(policy)})")

        self.env = env

        # Check correctness of policy
        states_given_in_policy = policy.keys()
        for s in states_given_in_policy:
            if not np.isclose(sum(policy[s]), 1.0):
                raise ValueError(f"The policy values given for state {s} in `policy` do not sum up to 1 ({sum(policy[s])})")
        self.policy = policy

        # Update the policy values if parameter `policy_default` is given
        if policy_default is not None:
            if len(policy_default) != self.env.getNumActions():
                raise ValueError(f"The length of the list given in `policy_default` ({len(policy_default)}) must be equal to the number of possible actions in the environment ({self.env.getNumActions()})")
            if not np.isclose(sum(policy_default), 1.0):
                raise ValueError(f"The policy values given in `policy_default` must sum up to 1 ({sum(policy_default)})")

            for s in set(self.env.getAllStates()).difference(states_given_in_policy):
                self.policy[s] = policy_default

    def choose_action(self, state):
        """
        Choose an action based on the policy stored in the object

        Arguments:
        state: int
            State index for which an action should be chosen.

        Return: int
        The index of the chosen action among the action indices defined in the environment stored in the object.
        """
        # Note: we should NOT use np.random.choice() to select the random value but instead the np_random.choice() method
        # from the mtrand.RandomState object np_random stored in the environment (coming from the gym DiscreteEnv class)
        # which is already used when resetting and stepping through the environment).
        # This is important for repeatability, because if the environment uses np_random.rand() to step
        # and here we use np.random.choice() to choose the action, we would be using two different seeds!
        # (and if we only set the np_random seed but not the np.random seed then the choice here
        # with np.random would NOT be deterministic...)
        action = self.env.np_random.choice(self.env.getNumActions(), p=self.getPolicyForState(state))
        return action

    def isDeterministic(self):
        return all([np.sum(1.0 in self.policy[s]) for s in self.policy])

    def getPolicyForState(self, state):
        return self.policy[state]

    def getPolicy(self):
        return self.policy
