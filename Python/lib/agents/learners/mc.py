# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:36:13 2020

@author: Daniel Mastropietro
@description: Definition of Monte Carlo algorithms
"""

import numpy as np

from . import Learner


class LeaMCLambda(Learner):
    """
    Monte Carlo learning algorithm using step size `alpha`, discount `gamma`, and decay parameter `lmbda`  
    applied to a discrete environment defined with the DiscreteEnv class of openAI's gym module.

    Args:
        env (gym.envs.toy_text.discrete.DiscreteEnv): the environment where the learning takes place.
    """

    def __init__(self, env, alpha=0.1, gamma=0.9, lmbda=0.8, debug=False):
        self.debug = debug

        # Attributes that MUST be presented for all TD methods
        self.env = env
        self.V = ValueFunctionApprox(self.env.getNumStates())
        self.Q = None
        self.alpha = alpha
        self.gamma = gamma
        
        # Attributes specific to the current TD method
        self.lmbda = lmbda
        
        # Reset the variables that store information about the episode 
        self.reset()      

    def reset(self):
        "Resets the variables that store information about the episode"
        # Store the states visited in the episode
        self.states = []
        # Store the rewards obtained after each action
        # (these are needed to compute G(t:t+n) at the end of the episode)
        # We initialize the rewards with one element equal to 0 for ease of notation
        # when retrieving the state and the reward received at each time t
        # because the state is defined for t = 0, ..., T-1 
        # while the reward is defined for t = 1, ..., T
        self.rewards = [0]
        
        ### All the attributes that follow are used to store information about the
        ### n-step returns and the lambda returns (i.e. the lambda-weighted average
        ### n-step returns for ALL time steps in the episode, from t=0, ..., T-1,
        ### where T is the time at which the episode terminates).
        ### Note that this information is stored in lists as opposed to numpy arrays
        ### because we don't know their size in advance (as we don't know when the episode
        ### will terminate) and increasing the size of numpy arrays is apparently less efficient
        ### than increasing the size of lists
        ### (Ref: https://stackoverflow.com/questions/568962/how-do-i-create-an-empty-array-matrix-in-numpy)
        # n-step return (G(t:t+n) 0 <= t <= T-1, 1 <= n <= T-t-1)
        # The array is a 1d array indexed by t only because the n's are accumulated in the sum
        # that then is used to compute G(t,lambda))
        self._G_list = []
        # lambda-return (G(t,lambda))
        self._Glambda_list = []

    def setParams(self, alpha=None, gamma=None, lmbda=None):
        self.alpha = alpha if alpha else self.alpha
        self.gamma = gamma if gamma else self.gamma
        self.lmbda = lmbda if lmbda else self.lmbda

    def learn_pred_V_slow(self, t, state, action, next_state, reward, done, info):
        # This learner updates the estimate of the value function V ONLY at the end of the episode
        self.states += [state]
        self.rewards += [reward]
        if done:
            # SHOULDN'T THIS BE T = t + 1?
            #T = len(self.states) - 1
            T = t + 1
            assert len(self.states) == T and len(self.rewards) == T + 1, \
                    "The number of states visited ({}) is equal to T ({}) " \
                    .format(len(self.states), T) + \
                    "and the number of rewards ({}) is T+1 ({})" \
                    .format(len(self.rewards), T)
            for t in range(T):
                # start from time t
                state = self.states[t]
                gtlambda = 0.
                for n in range(1, T - t):
                    # Compute G(t:t+n)
                    gttn = self.gt2tn(t, t + n)
                    lambda_power = self.lmbda**(n - 1)
                    # Update G(t,lambda)
                    gtlambda += lambda_power * gttn
                    #if lambda_power < self.rate_truncate:
                    #    break
    
                # ARE WE MISSING THE LAST TERM IN G(t,lambda)??
                #gtlambda *= 1 - self.lmbda 
                gtlambda = (1 - self.lmbda) * gtlambda + self.lmbda**(T - t - 1) * self.gt2tn(t, T)
                #if lambda_power >= self.rate_truncate:
                #    gtlambda += lambda_power * self.reward

                delta = gtlambda - self.V.getValue(state)
                self.updateV(state, delta)
            
            # Reset the episode information
            self.reset()

    def gt2tn(self, start, end):
        """
        @param start:       start time, t
        @param end:         end time, t+n
        @return:            The n-step gamma-discounted return starting at time 'start'.
        """
        G = 0.
        for t in range(start, end):     # Recall that the last value is excluded from the range
            reward = self.rewards[t+1]  # t+1 is fine because `reward` has one more element than `states`
            G += self.gamma**(t - start) * reward

        if end < len(self.states): 
            # The end time is NOT the end of the episode
            # => Add all return coming after the final time considered here (`end`) as the current estimate of
            # the value function at the end state.
            G += self.gamma**(end - start) * self.getV().getValue(self.states[end])

        return G

    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        "Learn the prediction problem: estimate the state value function"
        self.states += [state]
        self._updateG(t, state, next_state, reward, done)

        if done:
            # This means t+1 is the terminal time T
            # (recall we WERE in time t and we STEPPED INTO time t+1, so T = t+1)
            self.learn(t)
            # Reset the variables in the object that contain information about the episode
            self.reset()  

    def _updateG(self, t, state, next_state, reward, done):
        times_reversed = np.arange(t, -1, -1)  # This is t, t-1, ..., 0

        # Add the latest available return G(t:t+1), corresponding to the current time t, to the list
        # of n-step returns and initialize it to 0 (before it is updated with the currently observed reward
        # immediately below)
        #self._G_list += [0.0]
        # Update all the n-step G(tt:tt+n) returns, from time tt = 0, ..., t-1, using the new observed reward.
        # The gamma discount on the observed reward is stronger (i.e. more discount) as we move from time
        # t-1 down to time 0, that's why the exponent of gamma is the array of REVERSED times constructed above.
        # Note that each G is updated using the G(tt:tt+n-1) value from the previous iteration as INPUT
        # and the meaning of the updated G is G(tt:tt+n) for each n=1, .., t-tt+1)
        # (see my hand-written notes for better understanding) 
        Vns = self.V.getValue(next_state)   # Vns = V(next_state)        
        Vs = self.V.getValue(state)         #  Vs = V(state)
        assert not done or done and Vns == 0, "Terminal states have value 0 ({:.2g})".format(Vns)
        delta = reward + self.gamma*Vns - Vs
            ## This delta is the change to add to the G value at the previous iteration
            ## for all G's EXCEPT the new one corresponding to G(t:t+1) corresponding to
            ## the current time t (which does NOT have a previous value). 
        self._G_list = [g + self.gamma**n * delta for (g, n) in zip(self._G_list, times_reversed[:-1])]
        # Add the latest available return G(t:t+1), corresponding to the current time t, to the list of n-step returns
        self._G_list += [reward + self.gamma*Vns] 
        assert len(self._G_list) == len(times_reversed), \
                "Length of _G_list ({}) coincides with length of times_reversed ({})" \
                .format(len(self._G_list), len(times_reversed))

        # Update the estimates of the lambda-returns G(tt,lambda), for time tt = 0, ..., t-1
        self._Glambda_list += [0.0]
        if not done:
            # This implies adding the corresponding n-step return weighted by lambda**(n-1)
            # Since these updates are done in a cumulative way, n can actually be computed from t as done below.
            # We also include time = 0 because we have already added the current reward to _G_list above
            assert len(self._Glambda_list) == len(self._G_list), \
                    "Length of _Glambda_list ({}) coincides with length of _G_list ({})" \
                        .format(len(self._G_list), len(times_reversed))
            self._Glambda_list = [glambda + self.lmbda**n * g
                                  for (glambda, g, n) in zip(self._Glambda_list, self._G_list, times_reversed)]
            #print("t: {} \tG(t:t+n): {} \n\tG(t,lambda): {}".format(t, self._G_list, self._Glambda_list)) 
        else:
            # We finalize the computation of G(t,lambda) by scaling the sum so far with (lambda - 1)
            # and adding the final contribution from the latest reward (with no (lambda-1)-scaling)
            self._Glambda_list = [(1 - self.lmbda) * glambda + self.lmbda**n * g
                                  for (glambda, g, n) in zip(self._Glambda_list, self._G_list, times_reversed)]
            if self.debug:
                print("[DONE] t: {} \tG(t:t+n): {} \n\tG(t,lambda): {}".format(t, self._G_list, self._Glambda_list)) 

    def learn(self, t):
        "Updates the value function based on the new observed episode"
        # Terminal time
        T = t + 1
        
        # Store the list of G(t,lambda) values into an array 
        Glambda = np.array(self._Glambda_list)

        # Update the weights recursively from time 0 to T-1
        if self.debug:
            print("DONE:")
        for tt in np.arange(T):
            state = self.states[tt]
            # Error, where the lambda-return is used as the current value function estimate
            delta = Glambda[tt] - self.V.getValue(state)
            if self.debug:
                print("t: {} \tG(t,lambda): {} \tV({}): {} \tdelta: {}" \
                      .format(tt, Glambda[tt], state, self.V.getValue(state), delta))

            self.updateV(state, delta)

    def updateV(self, state, delta):
        # Gradient of V: it must have the same size as the weights
        # In the linear case the gradient is equal to the feature associated to state s,
        # which is stored at column s of the feature matrix X.
        gradient_V = self.V.X[:,state]  
            ## The above returns a ROW vector which is good because the weights are stored as a ROW vector

        # Update the weights based on the error observed at each time step and the gradient of the value function
        self.V.setWeights( self.V.getWeights() + self.alpha * delta * gradient_V )


class ValueFunctionApprox:
    "Class that contains information about the estimation of the state value function"

    def __init__(self, nS):
        "nS is the number of states"
        self.nS = nS
        self.V = 0
        self.weights = np.zeros(nS)
        # The features are dummy or indicator functions, i.e. each column of the
        # feature matrix X represents a feature and each row represents a state. Assuming we
        # order the states in columns in the same way we order them in rows, the X matrix
        # is a diagonal matrix
        self.X = np.eye(nS)

    def getValue(self, state):
        # TODO: Apply the generalization to any set of features X (i.e. uncomment the line below)
        #v = np.dot(self.weights[state], self.X[:,state])
        v = self.weights[state]
        return v

    def getValues(self):
        # TODO: Apply the generalization to any set of features X (i.e. uncomment the line below)
        #return np.dot(self.weights, self.X)
        return self.weights

    def setValues(self):
        self.V = np.dot(self.weights, self.X) 

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights
