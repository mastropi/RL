# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:36:13 2020

@author: Daniel Mastropietro
@description: Definition of Monte Carlo algorithms that are allowed to depend on the lambda parameter of lambda-returns.
                Here we use Monte Carlo to mean that the update of the value functions is done AT THE END OF THE EPISODE,
                i.e. OFFLINE, as opposed to at every step of the trajectory (or nearly), i.e. ONLINE, which would be
                the case for the TD(0) algorithm or an n-step TD algorithm.
                The classes defined here contain for instance:
                - an implementation of traditional Monte Carlo, where the target value function in the TD error (V_hat(s))
                is the observed return for the visited state at which the TD error is calculated.
                In this update strategy, there is NO mention to lambda whatsoever:
                See methods learn_pred_V_mc() and learn_mc() in LeaMCLambda.
                - an implementation of generalized Monte Carlo, which we call MC(lambda), where the target value function
                in the TD error (V_hat(s)) is equal to the lambda-return, for ANY value of lambda between 0 and 1.
                See methods learn_pred_V_lambda() and learn_lambda_return() in LeaMCLambda.
"""

from enum import Enum, unique
import numpy as np
import pandas as pd

from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners.episodic.discrete import Learner, AlphaUpdateType
from Python.lib.agents.learners.value_functions import ValueFunctionApprox

DEFAULT_NUMPY_PRECISION = np.get_printoptions().get('precision')
DEFAULT_NUMPY_SUPPRESS = np.get_printoptions().get('suppress')


@unique
class LearnerType(Enum):
    MC = 1
    LAMBDA_RETURN = 2


class LeaMCLambda(Learner):
    """
    Monte Carlo learning algorithm using step size `alpha`, discount `gamma`, and decay parameter `lmbda`  
    applied to a discrete-state environment defined with the DiscreteEnv class of openAI's gym module.

    Arguments:
    env: EnvironmentDiscrete
        The discrete-(state, action) environment where the learning takes place.
    """

    def __init__(self, env, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                 adjust_alpha_by_episode=False, alpha_min=0.,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 learner_type=LearnerType.MC,
                 debug=False):
        super().__init__(env, alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min,
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed)
        self.debug = debug

        # Attributes that MUST be presented for all MC methods
        self.V = ValueFunctionApprox(self.env.getNumStates(), self.env.getTerminalStates())
        self.Q = None
        self.alpha = alpha
        self.gamma = gamma
        
        #-- Attributes specific to the current MC method
        # Type of learner, whether use the traditional MC or the lambda-return leading to the MC(Lambda) learner,
        # which should be equivalent to MC as long as all episodes reach the terminal state.
        self.learner_type = learner_type
        # Value of lambda in the MC(lambda) learner
        if self.learner_type == LearnerType.LAMBDA_RETURN:
            self.lmbda = lmbda
        else:
            self.lmbda = None

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
                  adjust_alpha_by_episode=None, alpha_min=None):
        super().setParams(alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.gamma = gamma if gamma is not None else self.gamma
        self.lmbda = lmbda if lmbda is not None else self.lmbda


    #------------- Method that is called by the outside world to learn V(s) ----------------------#
    # Here we choose what method of this class to call internally
    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        if self.learner_type == LearnerType.LAMBDA_RETURN:
            self.learn_pred_V_lambda_return(t, state, action, next_state, reward, done, info)
        else:
            self.learn_pred_V_mc(t, state, action, next_state, reward, done, info)
    #------------- Method that is called by the outside world to learn V(s) ----------------------#


    #--------------------- MC(lambda): lambda-return Monte Carlo (SLOW) --------------------------#
    # NOTE: This is a slow implementation of MC(lambda) because it does not take advantage of the recursive
    # calculation of the return as G(t) = R(t+1) + G(t+1).
    # This piece of code was initially taken from MJeremy's GitHub.
    # For more information, see the entry on 13-Apr-2022 in my Tasks-Projects.xlsx file.
    def deprecated_learn_pred_V_slow(self, t, state, action, next_state, reward, done, info):
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
                    gttn = self.deprecated_gt2tn(t, t + n)
                    lambda_power = self.lmbda**(n - 1)
                    # Update G(t,lambda)
                    gtlambda += lambda_power * gttn
                    #if lambda_power < self.rate_truncate:
                    #    break

                # ARE WE MISSING THE LAST TERM IN G(t,lambda)??
                #gtlambda *= 1 - self.lmbda
                gtlambda = (1 - self.lmbda) * gtlambda + self.lmbda**(T - t - 1) * self.deprecated_gt2tn(t, T)
                #if lambda_power >= self.rate_truncate:
                #    gtlambda += lambda_power * self.reward

                delta = gtlambda - self.V.getValue(state)
                self.updateV(state, delta)

            self.final_report(t)

    def deprecated_gt2tn(self, start, end):
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
    #--------------------- MC(lambda): lambda-return Monte Carlo (SLOW) --------------------------#


    #----------------------------- Traditional Monte Carlo ---------------------------------------#
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

        This is the straight implementation of first-visit Monte Carlo, i.e. it does NOT use the equivalence of
        Monte Carlo with the lambda-return with lambda = 1, which is what method learn_lambda_return() does.
        In fact, in this function there is not evene mention to lambda. 

        Arguments:
        T: int
            Length of the episode, i.e. the time step at which the episode ends.
        """
        #-- Compute the observed return for each state in the trajectory for EVERY visit to it
        # Initialize the observed return to the value of the end state
        # (in case it is not a terminal state which has value 0 --this would happen if the episode is terminated early)
        # Note that we don't affect the initial G with gamma, because gamma comes into play in the recursive formula
        # used below to compute the final G)
        G = self.V.getValue(self.states[-1])

        # Keep track of the number of updates to the value function at each state
        # so that we can assert that there is at most one update for the first-visit MC algorithm
        nupdates = np.zeros(self.env.getNumStates())

        # NOTE: we start at the LATEST state (as opposed to the first) so that we don't
        # need to have a data structure that stores the already visited states in the episode;
        # we trade data structure creation and maintenance with easier algorithmic implementation of
        # first visit that does NOT require a special data structure storage.
        for tt in np.arange(T,0,-1) - 1:     # This is T-1, T-2, ..., 0
            state = self.states[tt]
            G = self.gamma*G + self._rewards[tt+1]
            # First-visit MC: We only update the value function estimation at the first visit of the state
            if self._states_first_visit_time[state] == tt:
                delta = G - self.V.getValue(state)
                self.updateV(state, delta)
                nupdates[state] += 1

        assert all(nupdates <= 1), "Each state has been updated at most once"
    #----------------------------- Traditional Monte Carlo -----------------------------------------#


    #---------------------- MC(lambda): lambda-return Monte Carlo ----------------------------------#
    def learn_pred_V_lambda_return(self, t, state, action, next_state, reward, done, info):
        """
        Learn the prediction problem: estimate the state value function.

        This function is expected to be called ONLINE, i.e. at each visit to a state, as opposed to OFFLINE,
        i.e. at the end of an episode.
        HOWEVER, learning, i.e. the update of the value function happens ONLY at the end of the episode.
        This means that every time this function is called before the end of the episode, the value function remains
        constant.
        """
        self._update_trajectory(t, state, reward)
        self._updateG(t, state, next_state, reward, done)

        if done:
            #-- Learn the new value of the state value function
            # Store the trajectory and rewards
            self.store_trajectory(next_state)
            self._update_state_counts(t+1, next_state)            

            # This means t+1 is the terminal time T
            # (recall we WERE in time t and we STEPPED INTO time t+1, so T = t+1)
            T = t + 1
            self.learn_lambda_return(T)

            self.final_report(T)

    def _updateG(self, t, state, next_state, reward, done):
        times_reversed = np.arange(t, -1, -1)  # This is t, t-1, ..., 0

        #-- Expression for delta: what we learn at this time step
        # Value of next state and current state needed to compute delta
        Vns = self.V.getValue(next_state)   # Vns = V(next_state)        
        Vs = self.V.getValue(state)         #  Vs = V(state)
        
        self._values_next_state += [Vns]

        # 2022/05/17: This assertion is no longer valid because we have implemented in the simulator (Simulator.run())
        # the option that the episode might end because of a maximum number of steps and NOT because a terminal state
        # has been reached.
        #assert not done or done and Vns == 0, "Terminal states have value 0 ({:.2g})".format(Vns)
        fn_delta = lambda n: reward + self.gamma*Vns - (n>0)*Vs
            ## This fn_delta(n) is the change to add to G(tt:tt+n) (actually we add fn_delta(n) *multiplied* by gamma^n)
            ## to construct G(tt:tt+n+1) (see the recursive updating expression in a comment below).
            ## Note that we need to remove `Vs` when n=0 (i.e. we have `(n>0)*Vs` because for n=0, G(tt:tt+n)
            ## degenerates to G(tt:tt), but this does NOT mean anything, i.e. when deriving the recursive expression
            ## that gives the computation of G(tt:tt+n) recursively, we must define G(tt:tt) = 0.
            ## And this definition implies that the substraction of `Vs` above should NOT be part of fn_delta(n)
            ## when n=0.

        # Add the latest available return G(t:t+1), corresponding to the current time t, to the list
        # of n-step returns and initialize it to 0 (before it is updated with the currently observed reward
        # immediately below)
        self._G_list += [0.0]
        # Update all the n-step G(tt:tt+n) returns, from time tt = 0, ..., t-1, based on the new observed reward.
        # This follows from the relation:
        # G(tt:tt+n+1)  = G(tt:tt+n) + gamma**n * [R(tt+n+1) + gamma*V(S(tt+n+1)) - V(S(tt+n))] =
        #               = G(tt:tt+n) + gamma^n * delta(tt+n+1)
        # where V(.) is the value function estimated at the end of the previous episode.
        # This update has to be done for every tt <= t-1
        # The gamma discount on the observed reward is stronger (i.e. more discount) as we move from time
        # t-1 down to time 0, that's why the exponent of gamma (n) reads its value from the array of
        # REVERSED times constructed above.
        # Note that each G is updated using the G(tt:tt+n-1) value from the previous iteration as INPUT
        # and the meaning of the updated G is G(tt:tt+n) for each n=1, .., t-tt+1)
        # This is true for ALL G(tt:tt+n) except for n=1 which does NOT have a previous value for G
        # (i.e. there is no such thing as G(tt:tt))
        # Note that in the code the theoretical "n=1" corresponds to n=0, which is why we multiply Vs
        # with the condition (n>0) in function fn_delta(n)
        # (see my hand-written notes for better understanding).
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
            if False:
                # DM-2020/07/20: Check whether the statement in paper "META-Learning state-based eligibility traces
                # by Zhao et al. (2020) is true, namely that G(t,lambda) can be computed recursively as:
                #     G(t,lambda) = R(t+1) + gamma * ( (1 - lambda) * V(S(t+1)) + lambda * G(t,lambda) )
                # AND IT IS VERIFIED!!!
                # The problem: I don't know how to prove it... I tried it on my A4 sheets when travelling to Valais
                # on 19-Jul-2020 but did not succeed...
                # OK, finally I was able to prove it and wrote it down on 31-Jul-2022 in my SPSS notebook.
                print("[DONE] t={}, Check G(t,lambda): [t, R(t+1), V(t+1), G(t,lambda), check_G(t,lambda), diff]".format(t))
                check = [R + self.gamma * ( (1 - self.lmbda)*Vns + self.lmbda*G )
                         for (R, Vns, G) in zip(self._rewards[1:-1], self._values_next_state[1:-1], self._Glambda_list[1:])] + [np.nan]
                diff = [c - G for (c, G) in zip(check, self._Glambda_list)]
                np.set_printoptions(precision=3, suppress=True)
                print(np.c_[np.arange(t+1), self._rewards[1:], self._values_next_state[1:], self._Glambda_list, check, diff])
                np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

    def learn_lambda_return(self, T):
        """
        Updates the value function based on a new observed EPISODE using first-visit Monte Carlo on any value of lambda.
        That is, this function is expected to be called when the episode ends.

        It differs from learn_mc() in that this method:
        - uses the lambda return values G(t,lambda) stored in the object, self._Glambda_list,
        which is a list because there is an entry for each time step in the trajectory.
        - can handle any value of lambda, not only lambda = 1, which is the actual definition of Monte Carlo update,
        strictly speaking. 

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
            state = self.states[tt]
            # First-visit MC: We only update the value function estimation at the first visit of the state
            if self._states_first_visit_time[state] == tt:
                # Value of the error (delta) where the lambda-return is used as the current value function estimate,
                # i.e. as the TARGET value --to which we want to take V(S(t))-- 
                # which we consider estimated by G(t,lambda)
                delta = Glambda[tt] - self.V.getValue(state)
                if self.debug:
                    print("t: {} \tG(t,lambda): {} \tV({}): {} \tdelta: {}" \
                          .format(tt, Glambda[tt], state, self.V.getValue(state), delta))

                self.updateV(state, delta)
    #---------------------- MC(lambda): lambda-return Monte Carlo --------------------------------#


    #------------------- Auxiliary function: value function udpate -------------------------------#
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
    #------------------- Auxiliary function: value function udpate -------------------------------#


class LeaMCLambdaAdaptive(LeaMCLambda):
    
    def __init__(self, env, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                 adjust_alpha_by_episode=True, alpha_min=0.,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 debug=False):
        super().__init__(env, alpha, gamma, lmbda, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min,
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed,
                         debug=debug)

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
        T = len(self.states)
        
        # Reset the state rewards and the state counts to 0
        # so that we can compute them fresh for the current episode
        self.state_rewards[:] = 0.
        self.state_counts[:] = 0

        # The following loop computes the gamma-discounted _rewards
        # (iteratively from the end of the episode to its beginning) 
        next_state = terminal_state     # Recall that the state reward for the terminal state is defined as 0
        for t in np.arange(T, 0, -1) - 1: # this is T-1, T-2, ..., 0 (the first element is included, the last one is not)
            state = self.states[t]

            # In the computation of the discounted reward for the current state
            # we use _rewards[t+1] and this is ok even for t = T-1 because we have constructed
            # the rewards array to have one more element than the states array.
            discounted_reward_for_state = self.rewards[t+1] + self.gamma * self.state_rewards[next_state]

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
