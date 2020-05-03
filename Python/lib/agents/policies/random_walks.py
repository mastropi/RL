# -*- coding: utf-8 -*-
"""
Created on Mon Mar  30 19:40:52 2020

@author: Daniel Mastropietro
@description: Random walk policies constructed on environments defined in the environments package.
Each policy should have the following methods:
    - choose_action() --> it returns an index on the action space with the action to take by the agent
"""


#import numpy as np

from Python.lib.environments import EnvironmentDiscrete


#__all__ = [ "PolRandomWalkDiscrete" ]

class PolRandomWalkDiscrete():
    """
    Random walk policy for a discrete environment defined with the DiscreteEnv class
    of openAI's gym module.

    Args:
        env(gym.envs.toy_text.discrete.DiscreteEnv) the environment where the random walk takes
        place.
    """

#    def __init__(self, env: DiscreteEnv):
    def __init__(self, env):
#        if not isinstance(env, EnvironmentDiscrete):
#            raise TypeError("The environment must be of type {} from the {} module ({})" \
#                            .format(EnvironmentDiscrete.__name__, EnvironmentDiscrete.__module__, env.__class__))
        self.env = env

    def choose_action(self):
        # Use np.random.choice() if the seed has been set using np.random.seed()
        # Otherwise, use the np_random.choice() method from the mtrand.RandomState object np_random stored in the environment
        # (coming from the gym DiscreteEnv class), which is already used when resetting and stepping the environment)
        # This is important for repeatability, because if the environment uses np_random.rand() to step
        # and here we use np.random.choice() to choose the action, we would be using two different seeds!
        # (and if we only set the np_random seed but not the np.random seed then the choice here
        # with np.random would NOT be deterministic...)
        #action = np.random.choice(self.env.getNumActions())
        action = self.env.np_random.choice(self.env.getNumActions())
        return action
