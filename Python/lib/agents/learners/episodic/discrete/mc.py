# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:36:13 2020

@author: Daniel Mastropietro
@description: Definition of Monte Carlo algorithms
"""

import numpy as np
import pandas as pd

from agents.learners.episodic.discrete import Learner, AlphaUpdateType
from .value_functions import ValueFunctionApprox

DEFAULT_NUMPY_PRECISION = np.get_printoptions().get('precision')
DEFAULT_NUMPY_SUPPRESS = np.get_printoptions().get('suppress')


class LeaMCLambda(Learner):
    """
    Monte Carlo learning algorithm using step size `alpha`, discount `gamma`, and decay parameter `lmbda`  
    applied to a discrete-state environment defined with the DiscreteEnv class of openAI's gym module.

    Args:
        env (gym.envs.toy_text.discrete.DiscreteEnv): the environment where the learning takes place.
    """

    def __init__(self, env, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                 adjust_alpha_by_episode=False, alpha_min=0.,
                 debug=False):
        super().__init__(env, alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.debug = debug

        # Attributes that MUST be presented for all MC methods
        self.V = ValueFunctionApprox(self.env.getNumStates())
        self.Q = None
        self.alpha = alpha
        self.gamma = gamma
        
        # Attributes specific to the current MC method
        self.lmbda = lmbda

    def _reset_at_start_of_episode(self):
        """
        Resets internal structures used during learning specific to this learning algorithm
        (all attributes reset here should start with an underscore, i.e. they should be private)
        """
        super()._reset_at_start_of_episode()

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

    def setParams(self, alpha=None, gamma=None, lmbda=None, adjust_alpha=None, alpha_update_type=None,
                  adjust_alpha_by_episode=None, alpha_min=0.):
        super().setParams(alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.gamma = gamma if gamma is not None else self.gamma
        self.lmbda = lmbda if lmbda is not None else self.lmbda

    def learn_pred_V_slow(self, t, state, action, next_state, reward, done, info):
        # This learner updates the estimate of the value function V ONLY at the end of the episode
        self._update_trajectory(t, state, reward)
        if done:
            # Store the trajectory and rewards
            self.store_trajectory(next_state)
            self._update_state_counts(t+1, next_state)            

            # Terminal time
            T = t + 1
            assert len(self._states) == T and len(self._rewards) == T + 1, \
                    "The number of _states visited ({}) is equal to T ({}) " \
                    .format(len(self._states), T) + \
                    "and the number of _rewards ({}) is T+1 ({})" \
                    .format(len(self._rewards), T)
            for t in range(T):
                # start from time t
                state = self._states[t]
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

            self.final_report(t)

    def gt2tn(self, start, end):
        """
        @param start:       start time, t
        @param end:         end time, t+n
        @return:            The n-step gamma-discounted return starting at time 'start'.
        """
        G = 0.
        for t in range(start, end):     # Recall that the last value is excluded from the range
            reward = self._rewards[t+1]  # t+1 is fine because `reward` has one more element than `_states`
            G += self.gamma**(t - start) * reward

        if end < len(self._states): 
            # The end time is NOT the end of the episode
            # => Add all return coming after the final time considered here (`end`) as the current estimate of
            # the value function at the end state.
            G += self.gamma**(end - start) * self.getV().getValue(self._states[end])

        return G

    def learn_pred_V_mc(self, t, state, action, next_state, reward, done, info):
        "Learn the prediction problem (estimate the state value function) using explicitly MC"
        self._update_trajectory(t, state, reward)

        if done:
            # Store the trajectory and rewards
            self.store_trajectory(next_state)
            self._update_state_counts(t+1, next_state)            

            # This means t+1 is the terminal time T
            # (recall we WERE in time t and we STEPPED INTO time t+1, so T = t+1)
            T = t + 1
            self.learn_mc(T)

            self.final_report(T)

    def learn_mc(self, T):
        """
        Updates the value function based on a new observed EPISODE using first-visit Monte Carlo.
        That is, this function is expected to be called when the episode ends.

        Arguments:
        T: int
            Length of the episode, i.e. the time step at which the episode ends.
        """
        #-- Compute the observed return for each state in the trajectory for EVERY visit to it
        # NOTE: we start at the LATEST state (as opposed to the first) so that we don'tt
        # need to have a data structure that stores the already visited states in the episode;
        # we trade data structure creation and maintenance with easier algorithmic implementation of
        # first visit that does NOT require a special data structure storage.
        G = 0
        # Keep track of the number of updates to the value function at each state
        # so that we can assert that there is at most one update for the first-visit MC algorithm
        nupdates = np.zeros(self.env.getNumStates())
        for tt in np.arange(T,0,-1) - 1:     # This is T-1, T-2, ..., 0
            state = self._states[tt]
            G = self.gamma*G + self._rewards[tt+1]
            # First-visit MC: We only update the value function estimation at the first visit of the state
            if self._states_first_visit_time[state] == tt:
                delta = G - self.V.getValue(state)
                self.updateV(state, delta)
                nupdates[state] += 1

        assert all(nupdates <= 1), "Each state has been updated at most once"

    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        "Learn the prediction problem: estimate the state value function"
        self._update_trajectory(t, state, reward)
        self._updateG(t, state, next_state, reward, done)

        if done:
            # Store the trajectory and rewards
            self.store_trajectory(next_state)
            self._update_state_counts(t+1, next_state)            

            # This means t+1 is the terminal time T
            # (recall we WERE in time t and we STEPPED INTO time t+1, so T = t+1)
            T = t + 1
            self.learn(T)

            self.final_report(T)

    def _updateG(self, t, state, next_state, reward, done):
        times_reversed = np.arange(t, -1, -1)  # This is t, t-1, ..., 0

        #-- Expression for delta: what we learn at this time step
        # Value of next state and current state needed to compute delta
        Vns = self.V.getValue(next_state)   # Vns = V(next_state)        
        Vs = self.V.getValue(state)         #  Vs = V(state)
        
        self._values_next_state += [Vns]

        
        assert not done or done and Vns == 0, "Terminal _states have value 0 ({:.2g})".format(Vns)
        fn_delta = lambda n: reward + self.gamma*Vns - (n>0)*Vs
            ## This fn_delta(n) is the change to add to the G value at the previous iteration
            ## for all G's EXCEPT the new one corresponding to G(t:t+1) corresponding to
            ## the current time t (which does NOT have a previous value).
            ## Note that n=1 in the expression G(t:t+1) corresponds to n=0 in the code
            ## (because of how the times_reversed array is defined.

        # Add the latest available return G(t:t+1), corresponding to the current time t, to the list
        # of n-step returns and initialize it to 0 (before it is updated with the currently observed reward
        # immediately below)
        self._G_list += [0.0]
        # Update all the n-step G(tt:tt+n) returns, from time tt = 0, ..., t-1, using the new observed reward.
        # The gamma discount on the observed reward is stronger (i.e. more discount) as we move from time
        # t-1 down to time 0, that's why the exponent of gamma (n) reads its value from the array of
        # REVERSED times constructed above.
        # Note that each G is updated using the G(tt:tt+n-1) value from the previous iteration as INPUT
        # and the meaning of the updated G is G(tt:tt+n) for each n=1, .., t-tt+1)
        # This is true for ALL G(tt:tt+n) except for n=1 which does NOT have a previous value for G
        # (note that in the code the theoretical "n=1" corresponds to n=0, which is why we multiply Vs
        # with the condition (n>0) in function fn_delta(n)   
        # (see my hand-written notes for better understanding) 
        self._G_list = [g + self.gamma**n * fn_delta(n) for (g, n) in zip(self._G_list, times_reversed)]
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
                        .format(len(self._Glambda_list), len(self._G_list))
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
            if True:
                # DM-2020/07/20: Check whether the statement in paper "META-Learning state-based eligibility traces
                # by Zhao et al. (2020) is true, namely that G(t,lambda) can be computed recursively as:
                #     G(t,lambda) = R(t+1) + gamma * ( (1 - lambda) * V(S(t+1)) + lambda * G(t,lambda) )
                # AND IT IS VERIFIED!!!
                # The problem: I don't know how to prove it... I tried it on my A4 sheets when travelling to Valais
                # on 19-Jul-2020 but did not succeed...
                print("[DONE] t={}, Check G(t,lambda): [t, R(t+1), V(t+1), G(t,lambda), check_G(t,lambda), diff]".format(t))
                check = [R + self.gamma * ( (1 - self.lmbda)*Vns + self.lmbda*G )
                         for (R, Vns, G) in zip(self._rewards[1:-1], self._values_next_state[1:-1], self._Glambda_list[1:])] + [np.nan]
                diff = [c - G for (c, G) in zip(check, self._Glambda_list)]
                np.set_printoptions(precision=3, suppress=True)
                print(np.c_[np.arange(t+1), self._rewards[1:], self._values_next_state[1:], self._Glambda_list, check, diff])
                np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

    def learn(self, T):
        """
        Updates the value function based on a new observed EPISODE using first-visit Monte Carlo.
        That is, this function is expected to be called when the episode ends.

        Arguments:
        T: int
            Length of the episode, i.e. the time step at which the episode ends.
        """
        # Store the list of G(t,lambda) values into an array 
        Glambda = np.array(self._Glambda_list)

        # Update the weights recursively from time 0 to T-1
        if self.debug:
            print("DONE:")
        for tt in np.arange(T):
            state = self._states[tt]
            # First-visit MC: We only update the value function estimation at the first visit of the state
            if self._states_first_visit_time[state] == tt:
                # Error, where the lambda-return is used as the current value function estimate,
                # i.e. as the TARGET value --to which we want to take V(S(t))-- 
                # which we consider estimated by G(t,lambda)
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
            ## Note: the above returns a ROW vector which is good because the weights are stored as a ROW vector

        # Update the weights based on the error observed at each time step and the gradient of the value function
        self.V.setWeights( self.V.getWeights() + self._alphas[state] * delta * gradient_V )

        # Update alpha for the next iteration, once we have already updated the value function for the current state
        self._update_alphas(state)


class LeaMCLambdaAdaptive(LeaMCLambda):
    
    def __init__(self, env, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                 adjust_alpha_by_episode=True, alpha_min=0.,
                 debug=False):
        super().__init__(env, alpha, gamma, lmbda, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min, debug)

        # Arrays that keep track of previous _rewards for each state
        self.all_states = np.arange(self.env.getNumStates())
        self.state_rewards = np.zeros(self.env.getNumStates())
        self.state_counts = np.zeros(self.env.getNumStates())
        self.state_lambdas = self.lmbda * np.ones(self.env.getNumStates())

    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        "Learn the prediction problem: estimate the state value function"
        self._update_trajectory(t, state, reward)
        self._updateG(t, state, next_state, reward, done)

        if done:
            # Store the trajectory and rewards
            self.store_trajectory(next_state)
            self._update_state_counts(t+1, next_state)            
            
            # Compute the gamma-discounted _rewards for each state visited in the episode
            state_rewards_prev = self.state_rewards.copy()
            self._computeStateRewards(next_state)
            visited_states = self.state_counts > 0

            # Delta discounted _rewards (w.r.t. previous discounted _rewards)
            delta_state_rewards = self.state_rewards[ visited_states ] - state_rewards_prev[ visited_states ]
            abs_delta_state_rewards = np.abs( delta_state_rewards )
            mean_abs_delta_state_rewards = np.mean( abs_delta_state_rewards )
            self.state_lambdas[ visited_states ] = self.lmbda * np.exp( abs_delta_state_rewards - mean_abs_delta_state_rewards )
            LambdaAdapt = pd.DataFrame.from_items([
                                ('visited', self.all_states[ visited_states ]),
                                ('count', self.state_counts[ visited_states ]),
                                ('rewards_prev_epi', state_rewards_prev[ visited_states ]),
                                ('rewards_curr_epi', self.state_rewards[ visited_states ]),
                                ('delta_rewards', delta_state_rewards),
                                ('delta_average_abs', np.repeat(mean_abs_delta_state_rewards, np.sum(visited_states))),
                                ('lambda', self.state_lambdas[ visited_states ])])
            print("\n\nSummary of lambda adaptation (Episode {}):".format(self.episode))
            print(LambdaAdapt)
            #input("Press Enter to continue...")

            # This means t+1 is the terminal time T
            # (recall we WERE in time t and we STEPPED INTO time t+1, so T = t+1)
            T = t + 1
            self.learn(T)

    def _computeStateRewards(self, terminal_state):
        # Length of the episode
        T = len(self._states)
        
        # Reset the state rewards and the state counts to 0
        # so that we can compute them fresh for the current episode
        self.state_rewards[:] = 0.
        self.state_counts[:] = 0

        # The following loop computes the gamma-discounted _rewards
        # (iteratively from the end of the episode to its beginning) 
        next_state = terminal_state     # Recall that the state reward for the terminal state is defined as 0
        for t in np.arange(T, 0, -1) - 1: # this is T-1, T-2, ..., 0 (the first element is included, the last one is not)
            state = self._states[t]

            # In the computation of the discounted reward for the current state
            # we use _rewards[t+1] and this is ok even for t = T-1 because we have constructed
            # the _rewards array to have one more element than the _states array.
            discounted_reward_for_state = self._rewards[t+1] + self.gamma * self.state_rewards[next_state]

            # Update the stored state _rewards which is set to the average discounted _rewards
            self._updateStateRewards(state, discounted_reward_for_state)

            # Prepare for next iteration
            # (note that this is called next_state but is actually
            # the state visited *previously* (i.e. at time t-1) in the episode) 
            next_state = state

    def _updateStateRewards(self, state, reward):
        # Regular average update
        self.state_rewards[state] = ( self.state_counts[state] * self.state_rewards[state] + reward ) \
                                    / (self.state_counts[state] + 1)   
        self.state_counts[state] += 1
