# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:36:13 2020

@author: Daniel Mastropietro
@description:   Definition of Monte Carlo algorithms that are allowed to depend on the lambda parameter of lambda-returns.
                Here we use Monte Carlo to mean that the update of the value functions is done AT THE END OF THE EPISODE,
                i.e. OFFLINE, as opposed to at every step of the trajectory (or nearly), i.e. ONLINE, which would be
                the case for the TD(0) algorithm or an n-step TD algorithm.
                The classes defined here contain for instance:
                - an implementation of traditional Monte Carlo, where the target value function in the TD error (V_hat(s))
                is the observed return for the visited state at which the TD error is calculated.
                In this update strategy, there is NO mention to lambda whatsoever:
                See methods learn_mc() and learn_mc_at_episode_end() in LeaMCLambda.
                - an implementation of generalized Monte Carlo, which we call MC(lambda), where the target value function
                in the TD error (V_hat(s)) is equal to the lambda-return, for ANY value of lambda between 0 and 1.
                See methods learn_lambda_return() and learn_lambda_return_at_episode_end() in LeaMCLambda.
"""

from enum import Enum, unique
import numpy as np
import pandas as pd

from Python.lib.agents.learners import LearningCriterion, LearningTask, ResetMethod
from Python.lib.agents.learners.episodic.discrete import Learner, AlphaUpdateType
from Python.lib.agents.learners.value_functions import ActionValueFunctionApprox, StateValueFunctionApprox


@unique
class LearnerType(Enum):
    MC = 1
    LAMBDA_RETURN = 2


class LeaMCLambda(Learner):
    """
    Monte Carlo learning algorithm using step size `alpha`, discount `gamma`, and decay parameter `lmbda`  
    applied to a discrete-state environment defined with the DiscreteEnv class of openAI's gym module.

    The decay parameter `lmbda` is only used when the learner_type is LearnerType.LAMBDA_RETURN, o.w. if
    learner_type is LearnerType.MC, the decay parameter `lmbda` is set to None. In this case, the learner is the
    traditional Monte-Carlo method.

    Arguments:
    env: EnvironmentDiscrete
        The discrete-(state, action) environment where the learning takes place.

    criterion: (opt) LearningCriterion
        The learning criterion in terms of how observed rewards are propagated to the return G(t).
        (e.g. average reward criterion or discounted reward criterion).
        default: LearningCriterion.DISCOUNTED

    task: (opt) LearningTask
        The type of learning task, whether it is based on episodes or it is based on a continuing Markov process that never ends
        (or that it ends after a pre-specified number of observed steps).
        IMPORTANT: It is not yet clear whether the class is already prepared to deal with the CONTINUING learning task, as it was originally
        created to deal with the DISCOUNTED learning task.
        default: LearningTask.EPISODIC
    """

    def __init__(self, env, criterion=LearningCriterion.DISCOUNTED, task=LearningTask.EPISODIC, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                 adjust_alpha_by_episode=False, alpha_min=0., func_adjust_alpha=None,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 store_history_over_all_episodes=False,
                 learner_type=LearnerType.MC,
                 debug=False):
        super().__init__(env, criterion=criterion, task=task, alpha=alpha, adjust_alpha=adjust_alpha, alpha_update_type=alpha_update_type,
                         adjust_alpha_by_episode=adjust_alpha_by_episode, alpha_min=alpha_min, func_adjust_alpha=func_adjust_alpha,
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed,
                         store_history_over_all_episodes=True if task == LearningTask.CONTINUING else store_history_over_all_episodes)
        self.debug = debug

        # Attributes that MUST be presented for all MC methods
        if task == LearningTask.CONTINUING:
            # For continuing learning tasks, there are NO terminal states, i.e. their value should NOT be set to 0 by the learner,
            # as they have their own value too!
            self.V = StateValueFunctionApprox(self.env.getNumStates(), {})
            self.Q = ActionValueFunctionApprox(self.env.getNumStates(), self.env.getNumActions(), {})
            self.A = ActionValueFunctionApprox(self.env.getNumStates(), self.env.getNumActions(), {})
        else:
            self.V = StateValueFunctionApprox(self.env.getNumStates(), self.env.getTerminalStates())
            self.Q = ActionValueFunctionApprox(self.env.getNumStates(), self.env.getNumActions(), self.env.getTerminalStates())
            self.A = ActionValueFunctionApprox(self.env.getNumStates(), self.env.getNumActions(), self.env.getTerminalStates())
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
    def learn(self, t, state, action, next_state, reward, done, info):
        "Learn the state value function V(s) with information collected at discrete time t"
        if self.learner_type == LearnerType.LAMBDA_RETURN:
            self.learn_lambda_return(t, state, action, next_state, reward, done, info)
        else:
            self.learn_mc(t, state, action, next_state, reward, done, info)
    #------------- Method that is called by the outside world to learn V(s) ----------------------#


    #--------------------- MC(lambda): lambda-return Monte Carlo (SLOW) --------------------------#
    # NOTE: This is a slow implementation of MC(lambda) because it does not take advantage of the recursive
    # calculation of the return as G(t) = R(t+1) + G(t+1).
    # This piece of code was initially taken from MJeremy's GitHub.
    # For more information, see the entry on 13-Apr-2022 in my Tasks-Projects.xlsx file.
    def deprecated_learn_slow(self, t, state, action, next_state, reward, done, info):
        # This learner updates the estimate of the value function V ONLY at the end of the episode
        if info.get('update_trajectory', True):
            self._update_trajectory(t, state, action, reward)
        if info.get('update_counts', True):
            self._update_state_counts(t, state)
        if done:
            # Terminal time
            T = t + 1

            # Store the trajectory
            self.store_trajectory_at_episode_end(T, next_state, debug=self.debug)
            self._update_state_counts(t+1, next_state)

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
                self._updateV(state, delta)
                self._updateQ(state, action, delta)
                self._updateA(state, action, delta)
                # Update the learning rate alpha for the next iteration
                self._update_alphas(state)

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
            G += self.gamma**(end - start) * self.V.getValue(self._states[end])

        return G
    #--------------------- MC(lambda): lambda-return Monte Carlo (SLOW) --------------------------#


    #----------------------------- Traditional Monte Carlo ---------------------------------------#
    def learn_mc(self, t, state, action, next_state, reward, done, info):
        "Learn the prediction problem (estimate the state value function) using explicitly MC"
        if info.get('update_trajectory', True):
            # We may not want to update the trajectory when learning the value function
            # (e.g. when using episodes under a continuing learning task context: in that case, the value functions
            # of the terminal state are normally learned at the start of the next episode (before updating the state)
            # and thus the state of the environment and action taken should NOT be recorded because they had already
            # been recorded at the previous step, when the episode ended (by the `done` block below)
            # --see also discrete.Simulator._run_single() and search for 'LearningTask.CONTINUING')
            self._update_trajectory(t, state, action, reward)
        if info.get('update_counts', True):
            self._update_state_counts(t, state)

        if done:
            # This means t+1 is the terminal time T
            # (recall we WERE in time t and we STEPPED INTO time t+1, so T = t+1)
            T = t + 1

            # Learn the value functions!
            # First store the alphas to be used in the value functions update
            self.store_learning_rate(self.getAlphasByState())
            self.learn_mc_at_episode_end(T, next_state)

            # Store the trajectory and update the state count of the end state
            # IMPORTANT: we need to store the trajectory AFTER the learning step performed above
            # because the method called next computes and stores the average learning rate by episode
            # (i.e. the average of the alpha's used at the different learning step of the value functions carried out by the learn method above,
            # i.e. the alpha used at every call to self._updateV() and self._updateQ() performed --typically at the first visit of the state only, not at every visit--
            # while traversing the states visited in the trajectory),
            self.store_trajectory_at_episode_end(T, next_state, debug=self.debug)
            if info.get('update_counts', True):
                self._update_state_counts(T, next_state)

    def learn_mc_at_episode_end(self, T, state_end):
        """
        Updates the value function based on a new observed EPISODE using first-visit Monte Carlo.
        That is, this function is expected to be called when the episode ends.

        This is the straight implementation of first-visit Monte Carlo, i.e. it does NOT use the equivalence of
        Monte Carlo with the lambda-return with lambda = 1, which is what method learn_lambda_return_at_episode_end() does.
        In fact, in this function there is not even a mention to lambda.

        Arguments:
        T: int
            Length of the episode, i.e. the time step at which the episode ends.

        state_end: int
            Index of the state at which the episode ends, whose value is retrieved as an initial value for the return G.
            This is needed for the cases where the episode ends NOT because the agent reached a terminal
            state but because the maximum simulation time has been reached (e.g. in continuing learning tasks
            or in episodic tasks where the agent takes a long time to reach a terminal state --because of a bad policy).
        """
        # Update the average reward (over all episodes, if required, typically when learning under the AVERAGE reward criterion)
        self.update_average_reward(T, state_end)

        #-- Compute the observed return for each state in the trajectory for EVERY visit to it
        # Initialize the observed return to the value of the end state
        # This is ONLY relevant when the end state is NOT a terminal state (which by definition has value 0)
        # in which case the state value is an estimate of the return that would be observed should the trajectory
        # continue until a terminal state is reached.
        # A non-terminal state as end state happens only when the episode is terminated early because the maximum
        # number of steps has been reached (this is set by e.g. parameter max_time_steps_per_episode in Simulator.run() in order
        # to avoid that a simulation runs forever without reaching a terminal state which might be very rare
        # --as in the Mountain Car example when the next action is chosen uniformly at random)
        # Note that we don't affect the initial G with gamma, because gamma comes into play in the recursive formula
        # used below to compute the final G)
        G = self.V.getValue(state_end)

        # Keep track of the number of updates to the value function at each state
        # so that we can assert that there is at most one update for the first-visit MC algorithm
        n_updates = np.zeros(self.env.getNumStates())

        # Keep track of the accumulated discount which is 1 for the previous to last state,
        # gamma for the previous to previous-to-last,
        # gamma^2 for the previous to previous to previous-to-last, and so forth.
        # This means that the accumulated discount for a time step that is s time steps before the end of the episode
        # is (1 + gamma + gamma^2 + ... + gamma^(s-1))
        # This is used as division of the return for the episodic learning task under the average reward criterion
        discount = 0.0  # We initialize at 0 because we want the discount for the previous to last state S(T-1) to be 1.0 (see update of `discount` in the FOR loop below)

        # NOTE: we start at the LATEST state (as opposed to the first) so that we don't
        # need to have a data structure that stores the already visited states in the episode;
        # we trade data structure creation and maintenance with easier algorithmic implementation of
        # first visit that does NOT require a special data structure storage.
        assert len(self._states) == len(self._actions) and len(self._rewards) == len(self._states) + 1
        for tt in np.arange(T, 0, -1) - 1:     # This is T-1, T-2, ..., 0
            state = self._states[tt]
            action = self._actions[tt]
            discount = 1 + self.gamma*discount      # The `1 +` corresponds to the discount associated to the reward observed at the currently processed time, when going from tt to tt+1.
            G = self._rewards[tt+1] + self.gamma*G  # The reward used in this sum is the reward of going from state `state` observed at time tt to the state observed at the next time tt+1
            if self.criterion == LearningCriterion.AVERAGE:
                if self.task == LearningTask.CONTINUING:
                    # Compute the differential return which is the one used for continuing learning tasks under the average reward criterion
                    # Ref: Sutton (2018), pag. 250
                    G -= self.getAverageReward()
                else:
                    # The return is defined as the discount-adjusted return (where "adjustment" in this case means "division")
                    # When the discount factor gamma = 1, this becomes the average reward observed from the current time tt to the end of the episode T.
                    G /= discount
            # First-visit MC: We only update the value function estimation at the first visit of the state
            if self._states_first_visit_time[state] == tt:
                delta = G - self.V.getValue(state)
                if self.criterion == LearningCriterion.AVERAGE and self.task == LearningTask.CONTINUING:
                    # Compute the differential delta which should be used to update the differential value function
                    # Ref: Sutton (2018), pag. 250
                    delta -= self.getAverageReward()
                self._updateV(state, delta)
                self._updateQ(state, action, delta)
                self._updateA(state, action, delta)
                # Update the learning rate alpha for the next iteration
                self._update_alphas(state)
                n_updates[state] += 1

        assert all(n_updates <= 1), "Each state has been updated at most once"
    #----------------------------- Traditional Monte Carlo -----------------------------------------#


    #---------------------- MC(lambda): lambda-return Monte Carlo ----------------------------------#
    # TODO: (2023/10/21) Implement the correction of the lambda-return needed in the average reward criterion case, as done above in learn_mc_at_episode_end()
    # In principle the values in _G_lambda_list (computed in _updateG()) should be corrected with the average reward.
    def learn_lambda_return(self, t, state, action, next_state, reward, done, info):
        """
        Learn the prediction problem: estimate the state value function.

        This function is expected to be called ONLINE, i.e. at each visit to a state, as opposed to OFFLINE,
        i.e. at the end of an episode.
        HOWEVER, learning, i.e. the update of the value function, happens ONLY at the end of the episode.
        This means that every time this function is called before the end of the episode, the value function remains
        constant.
        """
        if info.get('update_trajectory', True):
            # We may not want to update the trajectory when using this call just to learn the value functions
            # (e.g. when using episodes under a continuing learning task context: in that case, the value functions
            # of the terminal state are normally learned at the start of the next episode (before updating the state)
            # and thus the state of the environment and action taken should NOT be recorded because they had already
            # been recorded at the previous step, when the episode ended (by the `done` block below)
            # --see also discrete.Simulator._run_single() and search for 'LearningTask.CONTINUING')
            self._update_trajectory(t, state, action, reward)
        if info.get('update_counts', True):
            self._update_state_counts(t, state)
        self._updateG(t, state, next_state, reward, done)

        if done:
            # This means t+1 is the terminal time T
            # (recall we WERE in time t and we STEPPED INTO time t+1, so T = t+1)
            T = t + 1

            # Learn the value functions!
            # First store the alphas to be used in the value functions update
            self.store_learning_rate(self.getAlphasByState())
            self.learn_lambda_return_at_episode_end(T, next_state)

            # Store the trajectory and update the state count of the end state
            # IMPORTANT: we need to store the trajectory AFTER the learning step performed above
            # because the method called next computes and stores the average learning rate by episode
            # (i.e. the average of the alpha's used at the different learning step of the value functions carried out by the learn method above,
            # i.e. the alpha used at every call to self._updateV() and self._updateQ() performed --typically at the first visit of the state only, not at every visit--
            # while traversing the states visited in the trajectory),
            self.store_trajectory_at_episode_end(T, next_state, debug=self.debug)
            if info.get('update_counts', True):
                self._update_state_counts(T, next_state)

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
                print(np.c_[np.arange(t+1), self._rewards[1:], self._values_next_state[1:], self._Glambda_list, check, diff])

    def learn_lambda_return_at_episode_end(self, T, state_end):
        """
        Updates the value function based on a new observed EPISODE using first-visit Monte Carlo on any value of lambda.
        That is, this function is expected to be called when the episode ends.

        It differs from learn_mc_at_episode_end() in that this method:
        - uses the lambda return values G(t,lambda) stored in the object, self._Glambda_list,
        which is a list because there is an entry for each time step in the trajectory.
        - can handle any value of lambda, not only lambda = 1, which is the actual definition of Monte Carlo update,
        strictly speaking. 

        Arguments:
        T: int
            Length of the episode, i.e. the time step at which the episode ends.

        state_end: int
            Index of the state at which the episode ends.
        """
        # Update the average reward (over all episodes, if required, typically when learning under the AVERAGE reward criterion)
        self.update_average_reward(T, state_end)

        # Store the list of G(t,lambda) values into an array
        Glambda = np.array(self._Glambda_list)

        # Update the weights recursively from time 0 to T-1
        if self.debug:
            print("DONE:")
        for tt in np.arange(T):
            state = self._states[tt]
            action = self._actions[tt]
            # First-visit MC: We only update the value function estimation at the first visit of the state
            if self._states_first_visit_time[state] == tt:
                # Value of the error (delta) where the lambda-return is used as the current value function estimate,
                # i.e. as the TARGET value --to which we want to take V(S(t))-- 
                # which we consider estimated by G(t,lambda)
                delta = Glambda[tt] - self.V.getValue(state)
                if self.debug:
                    print("t: {} \tG(t,lambda): {} \tV({}): {} \tdelta: {}" \
                          .format(tt, Glambda[tt], state, self.V.getValue(state), delta))

                self._updateV(state, delta)
                self._updateQ(state, action, delta)
                self._updateA(state, action, delta)

                # Update the learning rate alpha for the next iteration
                # Note that the update is based ONLY on the state visit frequency, NOT on the state-action visit frequency...
                # This may not be the best approach for the estimation of Q(s,a) if, for instance, certain actions of a given state
                # are not visited often, their learning rate will be decreased even if the action was never visited!
                # TODO: (2023/11/08) We might need to update alphas ALSO by the ACTION visit frequency in order to have an appropriate estimation of Q(s,a)...
                self._update_alphas(state)
    #---------------------- MC(lambda): lambda-return Monte Carlo --------------------------------#


    #------------------- Auxiliary function: value function udpate -------------------------------#
    def _updateV(self, state, delta):
        "Updates the state value function V(s) for the given state using the given delta on the gradient computed assuming a linear approximation function"
        gradient_V = self.V.X[:, state] # row vector
        self.V.setWeights( self.V.getWeights() + self.getAlphasByState() * delta * gradient_V )    # The alpha value used in learning each state (given in self.getAlphasByState()) depends on the state being learned

    def _updateQ(self, state, action, delta):
        "Updates the action value function Q(s,a) for the given state and action using the given delta on the gradient computed assuming a linear approximation function"
        gradient_Q = self.Q.X[:, self.Q.getLinearIndex(state, action)]  # row vector
        _alphas = np.repeat(self.getAlphasByState(), self.env.getNumActions()) # We use the same alpha on all the actions associated to each state, but the alpha depends on the state
        self.Q.setWeights( self.Q.getWeights() + _alphas * delta * gradient_Q )

    def _updateA(self, state, action, advantage):
        """
        Sets the value of the Advantage function to the given value for the given state and action.
        An unbiased estimation of the advantage is the delta(V) observed when taking the given action at the given state.
        """
        # Recall that _setWeight() assumes that the features are dummy features (so, at some point this would need to be updated)
        self.A._setWeight(state, action, advantage)
    #------------------- Auxiliary function: value function udpate -------------------------------#

    #-- Getters
    def getV(self):
        return self.V

    def getQ(self):
        return self.Q

    def getA(self):
        return self.A


class LeaMCLambdaAdaptive(LeaMCLambda):
    
    def __init__(self, env, criterion=LearningCriterion.DISCOUNTED, task=LearningTask.EPISODIC, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                 adjust_alpha_by_episode=True, alpha_min=0., func_adjust_alpha=None,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 store_history_over_all_episodes=False,
                 debug=False):
        super().__init__(env, criterion=criterion, task=task, alpha=alpha, gamma=gamma, lmbda=lmbda, adjust_alpha=adjust_alpha, alpha_update_type=alpha_update_type,
                         adjust_alpha_by_episode=adjust_alpha_by_episode, alpha_min=alpha_min, func_adjust_alpha=func_adjust_alpha,
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed, store_history_over_all_episodes=store_history_over_all_episodes,
                         debug=debug)

        # Arrays that keep track of previous rewards for each state
        self.all_states = np.arange(self.env.getNumStates())
        self.state_rewards = np.zeros(self.env.getNumStates())
        self.state_counts = np.zeros(self.env.getNumStates())
        self.state_lambdas = self.lmbda * np.ones(self.env.getNumStates())

    def learn(self, t, state, action, next_state, reward, done, info):
        "Learn the prediction problem: estimate the state value function"
        if info.get('update_trajectory', True):
            self._update_trajectory(t, state, action, reward)
        if info.get('update_counts', True):
            self._update_state_counts(t, state)
        self._updateG(t, state, next_state, reward, done)

        if done:
            # This means t+1 is the terminal time T
            # (recall we WERE in time t and we STEPPED INTO time t+1, so T = t+1)
            T = t + 1

            # Store the trajectory
            self.store_trajectory_at_episode_end(T, next_state, debug=self.debug)
            if info.get('update_counts', True):
                self._update_state_counts(T, next_state)
            
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
                                ('visited', self.env.getAllStates()[ visited_states ]),
                                ('count', self.state_counts[ visited_states ]),
                                ('rewards_prev_epi', state_rewards_prev[ visited_states ]),
                                ('rewards_curr_epi', self.state_rewards[ visited_states ]),
                                ('delta_rewards', delta_state_rewards),
                                ('delta_average_abs', np.repeat(mean_abs_delta_state_rewards, np.sum(visited_states))),
                                ('lambda', self.state_lambdas[ visited_states ])])
            print("\n\nSummary of lambda adaptation (Episode {}):".format(self.episode))
            print(LambdaAdapt)
            #input("Press Enter to continue...")

            # Store the alphas to be used in the value functions update
            self.store_learning_rate(self.getAlphasByState())
            # done-2024/01/29: (2023/10/21) Fix this call, because currently this learn() method is not found
            # TODO: (2024/01/29) Check if the following piece of code is correct, or whether we should move it BEFORE the call to self.store_trajectory_at_episode_end() done above, as is the case with the non-adaptive MC learner
            if self.learner_type == LearnerType.LAMBDA_RETURN:
                self.learn_lambda_return_at_episode_end(T, next_state)
            else:
                self.learn_mc_at_episode_end(T, next_state)
            self.store_learning_rate(self.getAlphasByState())

    def _computeStateRewards(self, terminal_state):
        "Computes the gamma discounted reward for each state visited during a trajectory (stored in the self._states attribute)"
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
            # the rewards array to have one more element than the states array.
            discounted_reward_for_state = self._rewards[t+1] + self.gamma * self.state_rewards[next_state]

            # Update the stored state _rewards which is set to the average discounted _rewards
            self._updateStateRewards(state, discounted_reward_for_state)

            # Prepare for next iteration
            # (note that this is called next_state but is actually
            # the state visited *previously* (i.e. at time t-1) in the episode) 
            next_state = state

    def _updateStateRewards(self, state, discounted_reward):
        "Updates the gamma discounted average reward currently stored in the object for the given state using the given newly observed discounted reward value"
        # Regular average update
        self.state_rewards[state] = ( self.state_counts[state] * self.state_rewards[state] + discounted_reward ) \
                                    / (self.state_counts[state] + 1)   
        self.state_counts[state] += 1
