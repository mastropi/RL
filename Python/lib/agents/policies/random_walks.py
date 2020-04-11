# -*- coding: utf-8 -*-
"""
Created on Mon Mar  30 19:40:52 2020

@author: Daniel Mastropietro
@description: Random walk policies constructed on environments defined in the environments package.
Each policy should have the following methods:
    - choose_action() --> it returns an index on the action space with the action to take by the agent
"""


import numpy as np

#from gym.envs.toy_text.discrete import DiscreteEnv


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
        self.env = env

    def choose_action(self):
        action = np.random.choice(self.env.getNumActions())
        return action
