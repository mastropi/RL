# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:57:16 2020

@author: Daniel Mastropetro
@description: Definition of Temporal Difference algorithms.
"""

import numpy as np

from . import Learner, AlphaUpdateType
from .value_functions import ValueFunctionApprox


class LeaTDLambda(Learner):
    """
    TD(Lambda) learning algorithm using step size `alpha`, discount `gamma`, and decay parameter `lmbda`
    applied to a discrete environment defined with the DiscreteEnv class of openAI's gym module.

    Args:
        env (gym.envs.toy_text.discrete.DiscreteEnv): the environment where the learning takes place.
    """

    def __init__(self, env, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=False, alpha_min=0.,
                 debug=False):
        super().__init__(env, alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.debug = debug

        # Attributes that MUST be presented for all TD methods
        self.V = ValueFunctionApprox(self.env.getNumStates())
        self.Q = None
        self.alpha = alpha
        self.gamma = gamma
        
        # Attributes specific to the current TD method
        self.lmbda = lmbda
        # Eligibility traces
        self._z = np.zeros(self.env.getNumStates())
        self._z_all = np.zeros((0,self.env.getNumStates()))

    def _reset_at_start_of_episode(self):
        super()._reset_at_start_of_episode()
        self._z[:] = 0.
        self._z_all = np.zeros((0,self.env.getNumStates()))

    def setParams(self, alpha=None, gamma=None, lmbda=None, adjust_alpha=None, alpha_update_type=None,
                  adjust_alpha_by_episode=None, alpha_min=0.):
        super().setParams(alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.gamma = gamma if gamma is not None else self.gamma
        self.lmbda = lmbda if lmbda is not None else self.lmbda

    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        self._update_trajectory(t, state, reward)
        self._updateZ(state, self.lmbda)
        delta = reward + self.gamma * self.V.getValue(next_state) - self.V.getValue(state)
        #print("episode {}, state {}: count = {}, alpha = {}".format(self.episode, state, self._state_counts_overall[state], self._alphas[state]))
        self.V.setWeights( self.V.getWeights() + self._alphas[state] * delta * self._z )

        # Update alpha for the next iteration for "by state counts" update
        if not self.adjust_alpha_by_episode:
            self._update_alphas(state)

        if done:
            if self.debug:
                self._plotZ()
            self.store_trajectory(next_state)
            self._update_state_counts(t+1, next_state)            

            # Update alpha for the next iteration for "by episode" updates
            if self.adjust_alpha_by_episode:
                for state in range(self.env.getNumStates()):
                    if not self.env.isTerminalState(state):
                        self._update_alphas(state)

            self.final_report(t)

    def _updateZ(self, state, lmbda):
        # Gradient of V: it must have the same size as the weights
        # In the linear case the gradient is equal to the feature associated to state s,
        # which is stored at column s of the feature matrix X.
        gradient_V = self.V.X[:,state]
            ## Note: the above returns a ROW vector which is good because the weights are stored as a ROW vector
        self._z = self.gamma * lmbda * self._z + gradient_V
        self._z_all = np.r_[self._z_all, self._z.reshape(1,len(self._z))]

    def _plotZ(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self._z_all[:,6:12], '.-')
        plt.legend(np.arange(6,12,1))
        ax = plt.gca()
        ax.set_title("Episode {}".format(self.episode))


class LeaTDLambdaAdaptive(LeaTDLambda):
    
    def __init__(self, env, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=True, alpha_min=0.,
                 lambda_min=0., burnin=False, debug=False):
        super().__init__(env, alpha, gamma, lmbda, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min, debug)
        self._lambdas = []
        # Minimum lambda for the adaptive lambda so that there is still some impact
        # in past states at the beginning when all state values are equal and equal to 0
        self.lambda_min = lambda_min
        # Whether to perform a burn-in learning using a constant lambda
        # at the beginning to accelerate learning
        self.burnin = burnin

        # Counter of state visits WITHOUT resetting the count after each episode
        # (This is used to decide whether we should use the adaptive or non-adaptive lambda
        # based on whether the delta information from which the agent learns already contains
        # bootstrapping information about the value function at the next state)  
        self.state_counts_noreset = np.zeros(self.env.getNumStates())

        # Keep track of lambdas sums by state in order to compute statistics of lambda by state
        self._all_lambdas_n = np.zeros(self.env.getNumStates(), dtype=int)
        self._all_lambdas_sum = np.zeros(self.env.getNumStates(), dtype=float)
        self._all_lambdas_sum2 = np.zeros(self.env.getNumStates(), dtype=float)

        # Keep track of the average lambda by episode
        self.lambda_mean_by_episode = []

    def _reset_at_start_of_episode(self):
        super()._reset_at_start_of_episode()
        self._lambdas = []

    def reset(self, reset_episode=False, reset_value_functions=False):
        super().reset(reset_episode=reset_episode, reset_value_functions=reset_value_functions)
        if reset_episode:
            self.lambda_mean_by_episode = []
            del self._all_lambdas_n, self._all_lambdas_sum, self._all_lambdas_sum2
            self._all_lambdas_n = np.zeros(self.env.getNumStates(), dtype=int)
            self._all_lambdas_sum = np.zeros(self.env.getNumStates(), dtype=float)
            self._all_lambdas_sum2 = np.zeros(self.env.getNumStates(), dtype=float)

    def setParams(self, alpha=None, gamma=None, lmbda=None, adjust_alpha=None, alpha_update_type=None,
                  adjust_alpha_by_episode=None, alpha_min=0.,
                  lambda_min=0., burnin=False):
        super().setParams(alpha, gamma, lmbda, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.lambda_min = lambda_min if lambda_min is not None else self.lambda_min
        self.burnin = burnin if burnin is not None else self.burnin

    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        self._update_trajectory(t, state, reward)

        self.state_counts_noreset[state] += 1
        state_value = self.V.getValue(state)
        delta = reward + self.gamma * self.V.getValue(next_state) - state_value

        # Decide whether we do adaptive or non-adaptive lambda at this point
        # (depending on whether there is bootstrap information available or not)
        # TODO: (2020/04/21) Adapt the check on whether a value function has been modified at least once
        # that works also for the case when the initial state value function is not 0 (e.g. it is random).
        if not done and self.burnin and self.V.getValue(next_state) == 0: # self.state_counts_noreset[next_state] == 0:
            # The next state is non terminal and there is still no bootstrap information
            # about the state value function coming from the next state
            # => Do NOT do an adaptive lambda yet... so that some learning still happens.
            # In fact, the adaptive lambda may suggest NO learning due to the absence of innovation
            # in case the reward of the current step is 0 (e.g. in gridworlds receiving reward only at terminal states
            lambda_adaptive = self.lmbda
            #print("episode: {}, t: {}, lambda (fixed) = {}".format(self.episode, t, lambda_adaptive))
            #print("episode: {}, t: {}, next_state: {}, state_counts[{}] = {} \n\tstate counts: {}\n\tlambda(adap)={}" \
            #      .format(self.episode, t, next_state, next_state, self.state_counts_noreset[next_state], self.state_counts_noreset, lambda_adaptive))
        else:
            # Adaptive lambda
            # Define the relative state value change by dividing the change by the current state value
            # (if the current state value is 0, then set the change to infinite unless the change is 0 as well)
            delta_relative = delta / state_value if state_value != 0 \
                                                 else 0. if delta == 0. \
                                                 else np.Inf
            #delta_relative = np.exp( np.abs(delta) - np.abs(state_value) )
            lambda_adaptive = 1 - (1 - self.lambda_min) * np.exp( -np.abs(delta_relative) )
            #lambda_adaptive = 1 - (1 - self.lambda_min) * np.exp( -np.abs(delta) )
            #print("episode: {}, t: {}, lambda (adaptive) = {}".format(self.episode, t, lambda_adaptive))
        self._lambdas += [lambda_adaptive]
        # Update the sum of lambdas for the final lambda statistics by state
        self._all_lambdas_n[state] += 1 
        self._all_lambdas_sum[state] += lambda_adaptive 
        self._all_lambdas_sum2[state] += lambda_adaptive**2 

        # Update the eligibility trace
        self._updateZ(state, lambda_adaptive)
        # Update the weights
        self.V.setWeights( self.V.getWeights() + self._alphas[state] * delta * self._z )

        # Update alpha for the next iteration for "by state counts" update
        if not self.adjust_alpha_by_episode:
            self._update_alphas(state)

        if done:
            if self.debug:
                self._plotZ()
            self.store_trajectory(next_state)
            self._update_state_counts(t+1, next_state)            

            # Update alpha for the next iteration for "by episode" updates
            if self.adjust_alpha_by_episode:
                for state in range(self.env.getNumStates()):
                    if not self.env.isTerminalState(state):
                        self._update_alphas(state)
                
            self.final_report(t)

        if self.debug:
            print("t: {}, delta = {:.3g} --> lambda = {:.3g}".format(t, delta, lambda_adaptive))
            print("\tV(s={}->{}) = {}".format(state, next_state, self.V.getValue(state)))
            if done:
                import pandas as pd
                pd.options.display.float_format = '{:,.2f}'.format
                print(pd.DataFrame( np.c_[self._z, self.V.getValues()].T, index=['_z', 'V'] ))
    
            #input("Press Enter...")

    def final_report(self, T):
        super().final_report(T)
        # Store the (average) _lambdas by episode
        #print("lambdas in episode {}".format(self.episode))
        #with np.printoptions(precision=3, suppress=True):
        #    print(np.c_[self.states[:-1], self._lambdas]) 
        self.lambda_mean_by_episode += [np.mean(self._lambdas)]

    def compute_lambda_statistics_by_state(self):
        "Computes the number, mean, and standard deviation of historical values of lambda by state"
        lambdas_n = self._all_lambdas_n
        lambdas_mean = [S/n for n, S in zip(self._all_lambdas_n, self._all_lambdas_sum)]
        lambdas_std = [np.sqrt( (S2 - S**2/n) / (n - 1) )
                                for n, S, S2 in zip(self._all_lambdas_n, self._all_lambdas_sum, self._all_lambdas_sum2)]
        return lambdas_n, lambdas_mean, lambdas_std
