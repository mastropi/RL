# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:15:04 2020

@author: Daniel Mastropietro
@description: Definition of classes that are common to all learners.
"""


class Learner:
    """
    Class defining methods that are generic to ALL environments.
    """

    def getV(self):
        "Returns the object containing information about the state value function estimation"
        return self.V

    def getQ(self):
        "Returns the object containing information about the state-action value function estimation"
        return self.Q
