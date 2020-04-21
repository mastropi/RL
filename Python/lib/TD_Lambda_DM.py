# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:40:59 2020

@author: Daniel Mastropietro
"""

import numpy as np

from .GitHub.MJeremy2017.RL.RandomWalk_Lambda.TD_Lambda import ValueFunction
from .GitHub.MJeremy2017.RL.RandomWalk_Lambda.TD_Lambda import RandomWalk
from .GitHub.MJeremy2017.RL.RandomWalk_Lambda.TD_Lambda import ValueFunctionTD


class ValueFunction_DM(ValueFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getValues(self):
        return self.weights


class RandomWalk_DM(RandomWalk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Adaptive off-line learning
    def learnOffLine(self, valueFunc):
        T = len(self.states) - 1
        for t in range(T):
            # start from time t
            state = self.states[t]
            gtlambda = 0
            for n in range(1, T - t):
                # compute G_t:t+n
                gttn = self.gt2tn(valueFunc, t, t + n)
                
                lambda_power = np.power(self.lmbda, n - 1)

                gtlambda += lambda_power * gttn
                if lambda_power < self.rate_truncate:
                    break

            gtlambda *= 1 - self.lmbda
            if lambda_power >= self.rate_truncate:
                gtlambda += lambda_power * self.reward

            delta = gtlambda - valueFunc.value(state)
            valueFunc.learn(state, delta)

# 2020/03/07: INCOMPLETE! Still need to think about the implementation
class ValueFunctionTD_DM(ValueFunctionTD):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def updateZAdaptive(self, state, nxtState, reward):
        dev = 1
        state_value_estimate = reward + self.gamma * self.value(nxtState)
        state_value = self.value(state)
        delta_state_value = state_value_estimate - state_value 
        #delta_state_value_relative = delta_state_value - 
        lambda_adaptive = self.lmbda * np.exp( delta_state_value_relative )
        self._z *= self.gamma * self.lmbda
        self._z[state] += dev

    def learnAdaptive(self, state, nxtState, reward):
        delta = reward + self.gamma * self.value(nxtState) - self.value(state)
        self.weights += self.alpha * delta * self._z

    def getValues(self):
        return self.weights


