# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:57:16 2020

@author: Daniel Mastropietro
@description: Definition of value function estimators.

The value functions used by the learner classes are assumed to:
a) Be defined in terms of weights, e.g. the state value function for state s is V(s,w),
where w is the vector of weights applied to a set of features X.
Note that this assumption does NOT offer any limitation, since a tabular value function
can be defined using binary/dummy features. 

b) Have the following methods defined:
- reset(): resets the vector w of weigths to their initial estimates 
- getWeights(): reads the vector w of weights
- setWeights(): updates the vector w of weights
- setWeight(): updates the value of the weight for a particular state
- getValue(): reads the value function for a particular state or state-action
- getValues(): reads the value function for ALL states or state-actions
"""

import warnings

import numpy as np

# TODO: (2020/04/10) This class should cover ALL value functions whose estimation is done via approximation
# (i.e. using a parameterized expression whose parameters are materialized as a vector of weights)
# So the constructor should receive:
# - the dimension of the weights
# - the features x that are affected by the weights (possibly in a linear manner
# or perhaps in a nonlinear manner as well!)
class ValueFunctionApprox:
    "Class that contains information about the estimation of the state value function"

    def __init__(self, nS):
        "nS is the number of states"
        self.nS = nS
        self.weights = np.zeros(nS)
        # The features are dummy or indicator functions, i.e. each column of the
        # feature matrix X represents a feature and each row represents a state. Assuming we
        # order the _states in columns in the same way we order them in rows, the X matrix
        # is a diagonal matrix
        self.X = np.eye(self.nS)

        # Reset the weights to their initial estimation (e.g. all zeros) 
        self.reset()

    def reset(self):
        "Resets the weights to their initial estimates (i.e. erase all learning memory!)"
        self.weights[:] = 0

    def getWeights(self):
        return self.weights

    def setWeight(self, state, weight):
        if not (0 <= state < self.nS):
            warnings.warn("Invalid state ({}). It should be between 0 and {}. Nothing to do.".format(state, self.nS-1))
            return -1
        self.weights[state] = weight

    def setWeights(self, weights):
        self.weights = weights

    def getValue(self, state):
        if not (0 <= state < self.nS):
            warnings.warn("Invalid state ({}). It should be between 0 and {}. None is returned.".format(state, self.nS-1))
            return None
        return np.dot(self.weights, self.X[:,state])

    def getValues(self):
        return np.dot(self.weights, self.X)
