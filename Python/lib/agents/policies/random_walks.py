# -*- coding: utf-8 -*-
"""
Created on Mon Mar  30 19:40:52 2020

@author: Daniel Mastropietro
@description: Random walk policies constructed on environments defined in the environments module.
Each policy should have the following methods:
- choose_action(state) --> it returns an index on the action space with the action to take by the agent at a given state.
"""


#import numpy as np

from Python.lib.environments import EnvironmentDiscrete


#__all__ = [ "PolRandomWalkDiscrete" ]

class PolRandomWalkDiscrete():
    """
    Random walk policy for a discrete environment defined with the DiscreteEnv class
    of openAI's gym module.

    Arguments:
    env: EnvironmentDiscrete
        The environment where the random walk takes place.
    """

    def __init__(self, env: EnvironmentDiscrete):
#        if not isinstance(env, EnvironmentDiscrete):
#            raise TypeError("The environment must be of type {} from the {} module ({})" \
#                            .format(EnvironmentDiscrete.__name__, EnvironmentDiscrete.__module__, env.__class__))
        self.env = env

    def choose_action(self, state):
        """
        Choose an action of the random walk

        Note: Even if the state is not used, actions in general depend on the state of the system,
        therefore conceptually this method should receive an state on which the action will act,
        and also to have a signature that is compatible with the other choose_action() methods
        that are defined by other policies.
        """
        # Note: we should NOT use np.random.choice() to select the random value but instead the np_random.choice() method
        # from the mtrand.RandomState object np_random stored in the environment (coming from the gym DiscreteEnv class)
        # which is already used when resetting and stepping through the environment).
        # This is important for repeatability, because if the environment uses np_random.rand() to step
        # and here we use np.random.choice() to choose the action, we would be using two different seeds!
        # (and if we only set the np_random seed but not the np.random seed then the choice here
        # with np.random would NOT be deterministic...)
        action = self.env.np_random.choice(self.env.getNumActions())
        return action
