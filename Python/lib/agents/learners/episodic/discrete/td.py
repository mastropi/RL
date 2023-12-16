# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:57:16 2020

@author: Daniel Mastropietro
@description: Definition of Temporal Difference algorithms.
"""


from enum import Enum, unique

import numpy as np
from matplotlib import pyplot as plt, cm

from Python.lib.agents.learners import LearningCriterion, ResetMethod
from Python.lib.agents.learners.episodic.discrete import Learner, AlphaUpdateType
from Python.lib.agents.learners.value_functions import ActionValueFunctionApprox, StateValueFunctionApprox
import Python.lib.utils.plotting as plotting

@unique  # Unique enumeration values (i.e. on the RHS of the equal sign)
class AdaptiveLambdaType(Enum):
    ATD = 1     # (full) Adaptive TD(lambda)
    HATD = 2    # Homogeneously Adaptive TD(lambda)

DEFAULT_NUMPY_PRECISION = np.get_printoptions().get('precision')
DEFAULT_NUMPY_SUPPRESS = np.get_printoptions().get('suppress')


class LeaTDLambda(Learner):
    """
    TD(Lambda) learning algorithm using step size `alpha`, discount `gamma`, and decay parameter `lmbda`
    applied to a discrete environment defined with the DiscreteEnv class of openAI's gym module.

    Arguments:
    env: gym.envs.toy_text.discrete.DiscreteEnv
        The environment where the learning takes place.

    store_history_over_all_episodes: (opt) bool
        Whether to store in the attributes of the generic super class storing the trajectory
        (e.g. states, actions, rewards) the whole trajectory history, over all episodes.
        If this is requested, the trajectory is stored in lists of lists, where each sublist is
        the trajectory observed in an episode.
        This is useful if we need to compute something using the whole trajectory history or whether
        we want to know what the trajectories were on all episodes.
        This value is set to True if the learning criterion is the average reward because we need to
        access the average reward value estimated over ALL the episodes run as the average reward
        criterion is used on continuing learning tasks.
        default: False
    """

    def __init__(self, env, criterion=LearningCriterion.DISCOUNTED, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=False, alpha_min=0., func_adjust_alpha=None,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 store_history_over_all_episodes=False,
                 debug=False):
        super().__init__(env, criterion=criterion, alpha=alpha, adjust_alpha=adjust_alpha, alpha_update_type=alpha_update_type,
                         adjust_alpha_by_episode=adjust_alpha_by_episode, alpha_min=alpha_min, func_adjust_alpha=func_adjust_alpha,
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed,
                         store_history_over_all_episodes=True if criterion == LearningCriterion.AVERAGE else store_history_over_all_episodes)
        self.debug = debug

        # Attributes that MUST be present for all TD methods
        if criterion == LearningCriterion.AVERAGE:
            # Under the average reward criterion, there are NO terminal states
            # This is important because under the average reward criterion, the value of the terminal state should NOT be set to 0 by the learner.
            # as it has its own value too!
            self.V = StateValueFunctionApprox(self.env.getNumStates(), {})
            self.Q = ActionValueFunctionApprox(self.env.getNumStates(), self.env.getNumActions(), {})
        else:
            self.V = StateValueFunctionApprox(self.env.getNumStates(), self.env.getTerminalStates())
            self.Q = ActionValueFunctionApprox(self.env.getNumStates(), self.env.getNumActions(), self.env.getTerminalStates())
        self.gamma = gamma
        
        # Attributes specific to the current TD method
        self.lmbda = lmbda
        # Eligibility traces for learning V
        self._z_V = np.zeros(self.env.getNumStates())
        self._z_V_all = np.zeros((0, self.env.getNumStates()))                            # Historic information
        # Eligibility traces for learning Q
        self._z_Q = np.zeros(self.env.getNumStates() * self.env.getNumActions())
        self._z_Q_all = np.zeros((0, self.env.getNumStates() * self.env.getNumActions())) # Historic information

        # (Nov-2020) Product of alpha and z (the eligibility trace)
        # which gives the EFFECTIVE alpha value of the Stochastic Approximation algorithm
        # Goal: Compare the rate of convergence of the non-adaptive vs. the adaptive TD(lambda)
        # by keeping track of the effective alpha as a FUNCTION of the episode number for EACH STATE
        # Each episode is a different row of the _alphas_effective array and the states are across the columns.
        self._times_nonzero_update = [[] for _ in self.env.getAllStates()]
        self._alphas_effective = np.zeros((0, self.env.getNumStates()))

    def _reset_at_start_of_episode(self):
        super()._reset_at_start_of_episode()
        self._z_V[:] = 0.
        self._z_V_all = np.zeros((0, self.env.getNumStates()))
        self._z_Q[:] = 0.
        self._z_Q_all = np.zeros((0, self.env.getNumStates() * self.env.getNumActions()))

        # The effective alphas are only computed for the learning of V, not of Q
        # (as this is only stored for information purposes --e.g. plots of the eligibility traces to check if things are working properly)
        self._alphas_effective = np.zeros((0, self.env.getNumStates()))

    def setParams(self, alpha=None, gamma=None, lmbda=None, adjust_alpha=None, alpha_update_type=None,
                  adjust_alpha_by_episode=None, alpha_min=None):
        super().setParams(alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.gamma = gamma if gamma is not None else self.gamma
        self.lmbda = lmbda if lmbda is not None else self.lmbda

    def learn(self, t, state, action, next_state, reward, done, info):
        if info.get('update_trajectory', True):
            # We may not want to update the trajectory when learning the value function
            # (e.g. when learning an episodic task using the average reward criterion, where the episodic task is represented
            # as a continuing task: in that case, the value functions of the terminal state should be learned but its
            # state NOT recorded when learning the value functions as it is recorded as the end of the episode below
            # inside the `done` block --see also discrete.Simulator._run_single() and search for 'LearningCriterion.AVERAGE')
            self._update_trajectory(t, state, action, reward)  # This method belongs to the Learner super class defined in learners.episodic.discrete

        # Compute the delta values used for the update of each value function
        # NOTE: We compute the delta separately, and NOT inside the functions that update the value functions,
        # because the delta information is needed by the adaptive TD(lambda) learner and implementing a specific
        # function that computes the delta values increases DRY implementation.
        delta_V, delta_Q = self._compute_deltas(state, action, next_state, reward, info)

        #print("episode {}, state {}: count = {}, alpha = {}".format(self.episode, state, self._state_counts_over_all_episodes[state], self._alphas[state]))
        self._updateZ(state, action, self.lmbda)
        self._updateV(delta_V)
        self._updateQ(delta_Q)

        # We store the effective learning rates alpha
        # (effective in terms of  the eligibility trace that affects the delta values used when updating V and Q above)
        self._alphas_effective = np.r_[self._alphas_effective, (self._alphas * self._z_V).reshape(1, len(self._z_V))]
            ## NOTE: We need to reshape the product alpha*z because _alphas_effective is a 2D array with as many rows as
            ## the number of episodes run so far and as many columns as the number of states. The length of alpha*z
            ## is the number of states which should be laid out across the columns when appending a new row to
            ## _alphas_effective using np.r_[].

        # Update alpha for the next iteration for "by state counts" update
        #print("Learn: state = {}, next_state = {}, done = {}".format(state, next_state, done))
        if not self.adjust_alpha_by_episode:
            self._update_alphas(state)

        if done:
            # Terminal time (recall that we were at time t and stepped into time t+1 when reaching the terminal state)
            T = t + 1

            if self.debug: #and self.episode > 45: # Use the condition on `episode` in order to plot just the last episodes
                self._plotZ()
                self._plotAlphasEffective()
            self.store_trajectory_at_end_of_episode(T, next_state, debug=self.debug)
            self._update_state_counts(t+1, next_state)

            # Update alpha for the next iteration for "by episode" updates
            if self.adjust_alpha_by_episode:
                for state in range(self.env.getNumStates()):
                    if not self.env.isTerminalState(state):
                        self._update_alphas(state)

    def _compute_deltas(self, state, action, next_state, reward, info):
        """
        Computes the delta values to be used for the state value and action value functions update

        The delta value of the action value function is computed using the Expected SARSA approach,
        i.e. the expected Q-value over all possible next actions is used. This is theoretically equal to
        the state value function of the next state and thus its current estimate is used.

        Note that, since we use an *estimate* of the state value function V, as opposed to the true state value function,
        the result will most likely NOT be the same as computing the expected value of the estimated Q-values over all actions,
        as such computation does not necessarily produce a value that coincides with the estimated state value function.
        """
        # Note that for the Q value of the next state and next action we use its expected value
        # (over all possible actions) as it is done by the Expected SARSA learning of the Q function.
        # This avoids having to choose a particular next action for which we would require a new parameter
        # such as the epsilon value of the epsilon-greedy next action strategy.
        delta_V = reward + self.gamma * self.V.getValue(next_state) - self.V.getValue(state)
        delta_Q = reward + self.gamma * self._expected_next_Q(next_state) - self.Q.getValue(state, action)

        # Check whether we are learning the differential value function (average reward criterion) and adjust delta accordingly
        # Ref: Sutton, pag. 250
        if self.criterion == LearningCriterion.AVERAGE:
            if info.get('average_reward') is not None:
                average_reward_correction = info.get('average_reward')
            else:
                # The average reward that is used for the correction of the value functions to obtain the differential value functions
                # should be the average reward observed over ALL episodes.
                # This information should be stored by the learner because the constructor parameter store_history_over_all_episodes
                # is set to True when the learning criterion passed to the constructor is the average reward.
                average_reward_correction = self.getAverageReward()
            delta_V -= average_reward_correction
            delta_Q -= average_reward_correction

        return delta_V, delta_Q

    def _updateZ(self, state, action, lmbda):
        "Updates the eligibility traces used for learning V and those used for learning Q"
        # The gradients of V and Q are computed assuming the function approximation is linear, where
        # the gradient is equal to the feature associated to state s or state-action (s,a),
        # stored at the corresponding column of the feature matrix X.
        gradient_V = self.V.X[:, state]
        gradient_Q = self.Q.X[:, self.Q.getLinearIndex(state, action)]

        self._z_V = self.gamma * lmbda * self._z_V + \
                    gradient_V                                    # For every-visit TD(lambda)
                    #gradient_V * (self._state_counts[state] == 1)  # For first-visit TD(lambda)
        self._z_V_all = np.r_[self._z_V_all, self._z_V.reshape(1, len(self._z_V))]

        self._z_Q = self.gamma * lmbda * self._z_Q + \
                    gradient_Q
        self._z_Q_all = np.r_[self._z_Q_all, self._z_Q.reshape(1, len(self._z_Q))]

    def _updateV(self, delta):
        if delta != 0.0:
            # IMPORTANT: (2020/11/11) if we use _alphas[state] as the learning rate alpha in the following update,
            # we are using the SAME learning rate alpha for the update of ALL states, namely the
            # learning rate value associated to the state that is being visited now.
            # This is NOT how the alpha value should be updated for each state
            # (as we should apply the alpha associated to the state that decreases with the number of visits
            # to EACH state --which happens differently). However, this seems to give slightly faster convergence
            # than the theoretical alpha strategy just mentioned, at least in the gridworld environment!
            # If we wanted to use the strategy that should be applied in theory, we should simply
            # replace `self._alphas[state]` with `self._alphas` in the below expression,
            # as self._alphas is an array containing the alpha value to be used for each state.
            #self.V.setWeights( self.V.getWeights() + self._alphas[state] * delta * self._z_V )
            self.V.setWeights( self.V.getWeights() + self._alphas * delta * self._z_V )

    def _updateQ(self, delta):
        if delta != 0.0:
            # Repeat the alpha for each state as many times as the number of possible actions in the environment
            # Note that this repeat each value as we need it based on how state and actions are stored in the feature matrix used in self.Q,
            # namely grouped by state (e.g. if alphas = [2.5, 4.1, 3.0], the repeat by 2 generates [2.5, 2.5, 4.1, 4.1, 3.0, 3.0]
            # i.e. the same alpha for all actions associated to the same state (which is what we want, i.e. alphas on different actions grouped by state).
            _alphas = np.repeat(self._alphas, self.env.getNumActions())
            self.Q.setWeights( self.Q.getWeights() + _alphas * delta * self._z_Q )

    def _expected_next_Q(self, next_state):
        """
        Computes the expected Q value for the next state and next action.
        The estimated value V of the next state is used for this, which is similar to computing the expected Q value.
        However, to compute the expected Q value of the next state and next action (which is what is done
        by the Expected SARSA learner of the Q function) we need to have access to the policy...
        but we don't have access to it here (because it is not passed to the constructor of the class).
        """
        return self.V.getValue(next_state)

    def _plotZ(self):
        states2plot = self._choose_states2plot()
        plt.figure()
        plt.plot(self._z_V_all[:,states2plot], '.-')
        plt.legend(states2plot)
        ax = plt.gca()
        ax.set_xlabel("time step")
        ax.set_ylabel("z")
        start_state = self._states[0]
        ax.set_title("Eligibility trace by time step (Episode {} - start state = {})".format(self.episode, start_state))

    def _plotAlphasEffective(self):
        states2plot = self._choose_states2plot()
        plt.figure()
        #for s in states2plot:
        #   plt.plot(self._times_nonzero_update[s], self._alphas_effective[s], '.-')
        #plt.plot(self._times_nonzero_update[self.env.getNumStates()-1], self._alphas_effective[self.env.getNumStates()-1], '.-')
        plt.plot(self._alphas_effective[:,states2plot], '.-')
        plt.legend(states2plot)
        ax = plt.gca()
        #ax.set_xlim([0,len(self._states)-1])
        ax.set_ylim([0,1])
        ax.set_xlabel("time step")
        ax.set_ylabel("alpha*z")
        start_state = self._states[0]
        ax.set_title("Effective learning rate (alpha*z) by time step (Episode {} - start state = {})".format(self.episode, start_state))

    def _choose_states2plot(self):
        from Python.lib.environments.gridworlds import EnvGridworld1D
        from Python.lib.environments.mountaincars import MountainCarDiscrete
        if isinstance(self.env, EnvGridworld1D):
            states2plot = list(range(9,12))
        elif isinstance(self.env, MountainCarDiscrete):
            # Assuming MountainCar environment
            states2plot = list(range(100, 104))
        else:
            # Choose the states around the middle state
            states2plot = list(range(int(self.env.getNumStates()/2)-1, int(self.env.getNumStates()/2)+2))

        return states2plot

    #-- Getters
    def getV(self):
        return self.V

    def getQ(self):
        return self.Q


class LeaTDLambdaAdaptive(LeaTDLambda):
    
    def __init__(self, env, criterion=LearningCriterion.DISCOUNTED, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=True, alpha_min=0., func_adjust_alpha=None,
                 lambda_min=0., lambda_max=0.99, adaptive_type=AdaptiveLambdaType.ATD,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 store_history_over_all_episodes=False,
                 burnin=False, plotwhat="boxplots", fontsize=15, debug=False):
        super().__init__(env, criterion=criterion, alpha=alpha, gamma=gamma, lmbda=lmbda, adjust_alpha=adjust_alpha, alpha_update_type=alpha_update_type,
                         adjust_alpha_by_episode=adjust_alpha_by_episode, alpha_min=alpha_min, func_adjust_alpha=func_adjust_alpha,
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed,
                         store_history_over_all_episodes=True if criterion == LearningCriterion.AVERAGE else store_history_over_all_episodes,
                         debug=debug)
        
        # List that keeps the history of ALL lambdas used at EVERY TIME STEP
        # (i.e. all states are mixed up here and if we want to identify which state the lambda corresponds to
        # we need to look at the history of states provided by the learner)
        self._lambdas = []
        # Minimum lambda for the adaptive lambda so that there is still some impact
        # in past states at the beginning when all state values are equal and equal to 0
        self.lambda_min = lambda_min
        # Maximum lambda for the adaptive lambda which guarantees convergence
        self.lambda_max = lambda_max
        # Type of adaptive lambda (FULLY adaptive or HOMOGENEOUS)
        self.adaptive_type = adaptive_type
        # Whether to perform a burn-in learning using a constant lambda
        # at the beginning to accelerate learning
        self.burnin = burnin
        # Type of plot to generate to analyze the adaptive lambdas by state
        self.plotwhat = plotwhat
        # Reference font size to use in 2D image plot of the lambdas by state
        # to show the number of cases and/or the lambda value
        self.fontsize = fontsize

        # Counter of state visits WITHOUT resetting the count after each episode
        # (This MIGHT be used to decide whether we should use the adaptive or non-adaptive lambda
        # based on whether the delta information from which the agent learns already contains
        # bootstrapping information about the value function at the next state)  
        self.state_counts_noreset = np.zeros(self.env.getNumStates())

        #-- Variables used in the HOMOGENEOUS adaptive type case
        self._gradient_V_all = np.zeros((0, self.env.getNumStates()))

        #-- Variables for lambda statistics over all episodes
        # List of lists to store the lambdas used for each state in each episode
        self._lambdas_in_episode = [[] for _ in self.env.getAllStates()]
        # Store all the lambdas over all episodes
        # This is a 3D list indexed by:
        # - episode number
        # - state
        # - visit to the state in the episode
        self._all_lambdas_by_episode = []
        # For count, mean and std by state
        self._all_lambdas_n = np.zeros(self.env.getNumStates(), dtype=int)
        self._all_lambdas_sum = np.zeros(self.env.getNumStates(), dtype=float)
        self._all_lambdas_sum2 = np.zeros(self.env.getNumStates(), dtype=float)

        # Keep track of the average lambda by episode
        # computed over the lambdas observed during the TRAJECTORY
        # (i.e. it's not an average over the lambdas by state!)
        self.lambda_mean_by_episode = []

    def reset(self, reset_episode=False, reset_value_functions=False):
        super().reset(reset_episode=reset_episode, reset_value_functions=reset_value_functions)
        if reset_episode:
            self.lambda_mean_by_episode = []
            self._all_lambdas_by_episode = []
            del self._all_lambdas_n, self._all_lambdas_sum, self._all_lambdas_sum2
            self._all_lambdas_n = np.zeros(self.env.getNumStates(), dtype=int)
            self._all_lambdas_sum = np.zeros(self.env.getNumStates(), dtype=float)
            self._all_lambdas_sum2 = np.zeros(self.env.getNumStates(), dtype=float)

    def _reset_at_start_of_episode(self):
        super()._reset_at_start_of_episode()
        self._gradient_V_all = np.zeros((0, self.env.getNumStates()))
        self._gradient_Q_all = np.zeros((0, self.env.getNumStates() * self.env.getNumActions()))
        self._lambdas = []
        self._lambdas_in_episode = [[] for _ in self.env.getAllStates()]

    def setParams(self, alpha=None, gamma=None, lmbda=None, adjust_alpha=None, alpha_update_type=None,
                  adjust_alpha_by_episode=None, alpha_min=None,
                  lambda_min=None, lambda_max=None, adaptive_type=None,
                  burnin=False):
        super().setParams(alpha, gamma, lmbda, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.lambda_min = lambda_min if lambda_min is not None else self.lambda_min
        self.lambda_max = lambda_max if lambda_max is not None else self.lambda_max
        self.adaptive_type = adaptive_type if adaptive_type is not None else self.adaptive_type
        self.burnin = burnin if burnin is not None else self.burnin

    def learn(self, t, state, action, next_state, reward, done, info):
        if info.get('update_trajectory', True):
            # We may not want to update the trajectory when learning the value function
            # See the comment in the learn() method of the super class (normally LeaTDLambda) for an use case.
            self._update_trajectory(t, state, action, reward)  # This method belongs to the Learner super class defined in learners.episodic.discrete

        # See comment in the constructor of the meaning of this attribute, which is exclusively used in the adaptive lambda learner
        self.state_counts_noreset[state] += 1

        delta_V, delta_Q = self._compute_deltas(state, action, next_state, reward, info)

        # Decide whether we do adaptive or non-adaptive lambda at this point
        # (depending on whether there is bootstrap information available or not)
        # TODO: (2020/04/21) Adapt the check on whether a value function has been modified at least once that works also for the case when the initial state value function is not 0 (e.g. it is random).
        # (2023/11/12) The implementation of the above TO-DO would be in the line of using the state_counts_noreset attribute, which for some reason is now commented out.
        delta = delta_V
        if not done and self.burnin and self.V.getValue(next_state) == 0: # self.state_counts_noreset[next_state] == 0:
            # The next state is non terminal and there is still no bootstrap information
            # about the state value function coming from the next state
            # => Do NOT do an adaptive lambda yet... so that some learning still happens.
            # In fact, the adaptive lambda may suggest NO learning due to the absence of innovation
            # in case the reward of the current step is 0 (e.g. in gridworlds receiving reward only at terminal states)
            lambda_adaptive = self.lmbda
            #print("episode: {}, t: {}, lambda (fixed) = {}".format(self.episode, t, lambda_adaptive))
            #print("episode: {}, t: {}, next_state: {}, state_counts[{}] = {} \n\tstate counts: {}\n\tlambda(adap)={}" \
            #      .format(self.episode, t, next_state, next_state, self.state_counts_noreset[next_state], self.state_counts_noreset, lambda_adaptive))
        else:
            #-- Adaptive lambda
            # Define the relative target error delta by dividing the bootstrap delta (for now only for V, not for Q) to a reference value defined below
            #ref_value = self.V.getValue(state)                                         # reference value is the value of the current state
            #ref_value = np.mean( np.abs(self.V.getValues()) )                          # reference value is the average value over all states
            ref_value = np.mean( np.abs(self.V.getValues()[self.env.getNonTerminalStates()]) )     # reference value is the average value over NON-TERMINAL states (whose value is always 0, so they should no te included in the average)
            delta_relative = delta / ref_value if ref_value != 0 \
                                               else 0. if delta == 0. \
                                               else np.Inf
            # Relative delta that prevents division by 0 (i.e. delta_relative = exp(|delta|) / exp(|value|))
            #delta_relative = np.exp( np.abs(delta) - np.abs(self.V.getValue(state)) )

            # Compute lambda as a function of delta or relative delta
            lambda_adaptive = min( 1 - (1 - self.lambda_min) * np.exp( -np.abs(delta_relative) ), self.lambda_max )
            #lambda_adaptive = min( 1 - (1 - self.lambda_min) * np.exp( -np.abs(delta) ), self.lambda_max )

            if self.debug:
                print("episode: {}, t: {}: TD ERROR != 0 => delta = {}, delta_rel = {}, lambda (adaptive) = {}\n" \
                        "trajectory so far: {}" \
                        "--> V(s): {}".format(self.episode, t, delta, delta_relative, lambda_adaptive, self._states, self.V.getValues()))

        # Update the eligibility trace
        self._updateZ(state, action, lambda_adaptive)
        self._updateV(delta_V)
        self._updateQ(delta_Q)

        # The effective alphas are only computed for the learning of V, not of Q
        # (as this is only stored for information purposes --e.g. plots of the eligibility traces to check if things are working properly)
        self._alphas_effective = np.r_[self._alphas_effective, (self._alphas * self._z_V).reshape(1, len(self._z_V))]
            ## NOTE: We need to reshape the product alpha*z because _alphas_effective is a 2D array with as many rows as
            ## the number of episodes run so far and as many columns as the number of states. The length of alpha*z
            ## is the number of states which should be laid out across the columns when appending a new row to
            ## _alphas_effective using np.r_[].

        # Keep history of used lambdas
        self._lambdas += [lambda_adaptive]
        # Update the history of lambdas used
        self._lambdas_in_episode[state] += [lambda_adaptive]
        self._all_lambdas_n[state] += 1
        self._all_lambdas_sum[state] += lambda_adaptive
        self._all_lambdas_sum2[state] += lambda_adaptive**2

        # Update alpha for the next iteration for "by state counts" update
        if not self.adjust_alpha_by_episode:
            self._update_alphas(state)

        if done:
            # Terminal time (recall that we were at time t and stepped into time t+1 when reaching the terminal state)
            T = t + 1

            if self.debug: # and self.episode > 45:
                self._plotZ()
                self._plotAlphasEffective()
            self.store_trajectory_at_end_of_episode(T, next_state, debug=self.debug)
            self._update_state_counts(t+1, next_state)
            self._store_lambdas_in_episode()

            # Update alpha for the next episode for "by episode" updates
            if self.adjust_alpha_by_episode:
                for state in range(self.env.getNumStates()):
                    if not self.env.isTerminalState(state):
                        self._update_alphas(state)

        if self.debug:
            print("t: {}, delta = {:.3g} --> lambda = {:.3g}".format(t, delta, lambda_adaptive))
            print("\tV(s={}->{}) = {}".format(state, next_state, self.V.getValue(state)))
            if done:
                import pandas as pd
                pd.options.display.float_format = '{:,.2f}'.format
                print(pd.DataFrame( np.c_[self._z_V, self.V.getValues()].T, index=['_z', 'V'] ))
    
            #input("Press Enter...")

    def _updateZ(self, state, action, lmbda):
        if self.debug and False:
            print("")
            print("state = {}: lambda = {:.2f}".format(state, lmbda))
        if self.adaptive_type == AdaptiveLambdaType.ATD:
            super()._updateZ(state, action, lmbda)
        else:
            # In the HOMOGENEOUS adaptive lambda we need to store the HISTORY of the gradient
            # (because we need to retroactively apply the newly computed lambda to previous eligibility traces)
            gradient_V = self.V.X[:, state]      # Note: this is returned as a ROW vector, even when we retrieve the `state` COLUMN of matrix X
            gradient_Q = self.V.X[:, self.Q.getLinearIndex(state, action)]
            # Use the following calculation of the gradient for FIRST-VISIT TD(lambda)
            # (i.e. the gradient is set to 0 if the current visit of `state` is not the first one)
            #gradient_V * (self._state_counts[state] == 1)  # For first-visit TD(lambda)
            self._gradient_V_all = np.r_[self._gradient_V_all, gradient_V.reshape(1, len(gradient_V))]
            self._gradient_Q_all = np.r_[self._gradient_Q_all, gradient_Q.reshape(1, len(gradient_Q))]

            if self.debug and False:
                print("Gradients:")
                print(self._gradient_V_all)

            # Compute the exponents of gamma*lambda, which go from n_trace_length-1 down to 0
            # starting with the oldest gradient.
            # Note that the trace length is the same for both V and Q,
            # as it is simply the number of rows in _gradient_V_all, which coincides with the number of rows in _gradient_Q_all.
            assert self._gradient_V_all.shape[0] == self._gradient_Q_all.shape[0]
            n_trace_length = self._gradient_V_all.shape[0]
            exponents = np.array( range(n_trace_length-1, -1, -1) ).reshape(n_trace_length, 1)
                ## The exponents are e.g. (3, 2, 1, 0) when n_trace_length = 4
                ## Note that we reshape the exponents as a column vector because we need to use it
                ## as the exponents of gamma*lambda which will multiply the respective row of the historical gradients
                ## i.e.:
                ## (gamma*lambda)**exponents[0] multiplies row 0 of the historical gradients
                ## (gamma*lambda)**exponents[1] multiplies row 1 of the historical gradients
                ## and so forth...

            # New eligibility trace using the latest computed lambda as weight for ALL past time steps
            self._z_V = np.sum( (self.gamma * lmbda)**exponents * self._gradient_V_all, axis=0 )
            self._z_V_all = np.r_[self._z_V_all, self._z_V.reshape(1, len(self._z_V))]
            self._z_Q = np.sum( (self.gamma * lmbda)**exponents * self._gradient_Q_all, axis=0 )
            self._z_Q_all = np.r_[self._z_Q_all, self._z_Q.reshape(1, len(self._z_Q))]

            if self.debug and False:
                print("Exponents: {}".format((self.gamma * lmbda)**exponents))

        if self.debug and False:
            print("Z's & lambda's by state:")
            print(self._z_V_all)
            print(self._lambdas_in_episode)

    def _store_lambdas_in_episode(self):
        if self.debug:
            print("lambdas in episode {}:".format(self.episode))
            print(self._lambdas_in_episode)
        self._all_lambdas_by_episode += [[self._lambdas_in_episode[s] for s in self.env.getAllStates()]]

        # Store the (average) _lambdas by episode
        if self.debug:
            print("lambdas in episode {}".format(self.episode))
            np.set_printoptions(precision=3, suppress=True)
            print(np.c_[self.states[:-1], self._lambdas])
            np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)
        self.lambda_mean_by_episode += [np.mean(self._lambdas)]

    def compute_lambda_statistics_by_state(self):
        "Computes the number, mean, and standard deviation of historical values of lambda by state"
        lambdas_n = self._all_lambdas_n
        lambdas_mean = [S/n for n, S in zip(self._all_lambdas_n, self._all_lambdas_sum)]
        lambdas_std = [np.sqrt( (S2 - S**2/n) / (n - 1) )
                                for n, S, S2 in zip(self._all_lambdas_n, self._all_lambdas_sum, self._all_lambdas_sum2)]
        return lambdas_n, lambdas_mean, lambdas_std

    # GETTERS
    def getLambdasByEpisodeAndState(self):
        return self._all_lambdas_by_episode

    # PLOTTING FUNCTIONS
    def plot_info(self, episode, nepisodes):
        """
        Plots selected information about the experiment run.

        Arguments:
        episode: int
            Number of episode of interest for the plot.

        nepisodes: int
            Number of total episodes to run or already run

        what: str
            What to plot. Possible values are:
            - "boxplots": for a boxplot of lambdas by state for different ranges of episode number
                (e.g. early learning episodes vs. late learning episodes) divided into 4 groups).
            - "average": for a plot of average lambda by state over all the episodes run.
        """

        # Finalize setup of all possible graphs created here
        def finalize_plot_1D(ax, state_counts, title="Lambda by state for selected episodes / State count distribution"):
            ax.set_title(title)

            ax.set_xticks(self.env.getAllStates())
            ax.set_xlim((0, self.env.getNumStates()-1))
            ax.set_ylim((0, 1.02))
            ax.set_ylabel("Lambda")
            ax.tick_params(axis='y', colors="orange")
            ax.yaxis.label.set_color("orange")

            # State count distribution
            ax1sec = ax.twinx()
            ax1sec.bar(self.env.getAllStates(), state_counts, color="blue", alpha=0.2)
            ax1sec.tick_params(axis='y', colors="blue")
            ax1sec.yaxis.label.set_color("blue")
            ax1sec.set_ylabel("State count")
            plt.sca(ax) # Go back to the primary axis

        def finalize_plot_2D(ax, state_lambdas_2D, state_counts_2D, fontsize=12, title="Lambda by state for selected episodes / State count distribution"):
            ax.set_title(title)

            (nx, ny) = state_counts_2D.shape
            # Adjust the font size according to the image shape
            # It is assumed that the unmodified fontsize works fine for a 5x5 grid
            fontsize = int( np.min((5/nx, 5/ny)) * fontsize )
            for y in range(ny):
                for x in range(nx):
                    if state_lambdas_2D is None:
                        ax.text(y, x, "N={}".format(int(state_counts_2D[x,y])),
                                fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
                    elif state_counts_2D is None:
                        ax.text(y, x, "{:.3f}".format(state_lambdas_2D[x,y]),
                                fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
                    else:
                        ax.text(y, x, "{:.3f}\n (N={})".format(state_lambdas_2D[x,y], int(state_counts_2D[x,y])),
                                fontsize=fontsize, horizontalalignment='center', verticalalignment='center')

        #---- Plot of state-dependent lambdas
        # Option 1: Compute the mean and standard deviation of lambdas by state for last episode 
        #nstates = self.env.getNumStates()
        #states = np.arange(nstates)
        #lambda_sum = np.zeros_like(states, dtype=float)
        #lambda2_sum = np.zeros_like(states, dtype=float)
        #n = np.zeros_like(states)
        #for i, lmbda in enumerate(self._lambdas):
        #    s = self.states[i]
        #    lambda_sum[s] += lmbda
        #    lambda2_sum[s] += lmbda**2
        #    n[s] += 1
        #    print("i: {}, state: {}, lambda: {}, lambda_sum: {}, n: {}".format(i, s, lmbda, lambda_sum[s], n[s]))
        #lambda_mean = np.nan * np.ones_like(states, dtype=float)
        #lambda_std = np.nan * np.ones_like(states, dtype=float)
        #for s in range(self.env.getNumStates()):
        #    if n[s] > 0:
        #        lambda_mean[s] = lambda_sum[s] / n[s]
        #        if n[s] > 1:
        #            lambda_std[s] = np.sqrt( ( lambda2_sum[s] - n[s] * lambda_mean[s]**2 ) / (n[s] - 1) )
        #ax1.plot(states, lambda_mean, '.-', color="orange")
        #ax1.errorbar(states, lambda_mean, yerr=lambda_std, capsize=4, color="orange")
        #ax1.set_title("Mean and StdDev of state-dependent lambdas at the last episode")

        # Option 2: Use the statistics tracked during experiment to plot lambda statistics by state 
        #(lambdas_n, lambdas_mean, lambdas_std) = self.compute_lambda_statistics_by_state()
        #print("lambda statistics by state:")
        #with np.printoptions(precision=3, suppress=True):
        #    print(np.c_[self.env.getAllStates(), lambdas_n, lambdas_mean, lambdas_std])
        #ax1.plot(self.env.getAllStates(), lambdas_mean, '.-', color="orange")
        #ax1.errorbar(self.env.getAllStates(), lambdas_mean, yerr=lambdas_std, capsize=4, color="orange")

        if nepisodes < 4:
            print("NOTE: No plots by episode number group are generated: at least 4 episodes are needed.")
            return

        # Option 3: Plot distribution of lambdas by range of episode numbers
        # (to see how lambdas by state evolve with time)
        # Define 4 groups of episode numbers range 
        episodes_to_show = lambda nepisodes, nplots: set( list( range(0, nepisodes, max(1, int( (nepisodes-1)/(nplots-1) ))) ) + [nepisodes-1] )
        episodes_to_consider = lambda start, stop: range(start, stop+1)

        #for e in episodes_to_show(nepisodes, nplots):
        #    print("Plotting lambdas for episode {}:".format(e))
        #    print(self._all_lambdas_by_episode[e])
        #    ax1.violinplot(self._all_lambdas_by_episode[e], showmeans=True)
        lambdas_by_state = [[] for _ in self.env.getAllStates()]
        for e in episodes_to_consider(0, nepisodes-1):
            for s in self.env.getAllStates():
                lambdas_by_state[s] += self._all_lambdas_by_episode[e][s]
        states2plot = [s for s in self.env.getNonTerminalStates() if self._state_counts_over_all_episodes[s] > 0] 
        #print("lambdas_by_state for plotting:")
        #print([lambdas_by_state[s] for s in states2plot])

        #plt.figure()
        #ax1 = plt.gca()
        # Note: the violinplot() function does NOT accept an empty list for plotting nor NaN values
        # (when at least a NaN value is present, nothing is shown for the corresponding group!)   
        #plotting.violinplot(ax1, [lambdas_by_state[s] for s in states2plot], positions=states2plot,
                            #color_body="orange", color_lines="orange", color_means="red")
        #finalize_plot(ax1, self.env.getStateCounts())
 
        fig = plt.figure()
        nplots = 4
        axes = fig.subplots(2, int(nplots/2))
        if self.plotwhat == "average":
            # Create a new figure to show the distribution of state counts
            # as we cannot plot them in the same figure as the lambda values
            fig_counts = plt.figure()
            axes_counts = fig_counts.subplots(2, int(nplots/2))

        lambda_mean = np.sum( [S for S in self._all_lambdas_sum] ) / np.sum( [n for n in self._all_lambdas_n] )
        nvisits_min = np.min( [n for n in self._all_lambdas_n] )
        nvisits_mean = np.mean( [n for n in self._all_lambdas_n] )
        nvisits_max = np.max( [n for n in self._all_lambdas_n] )
        print("Average lambda over all steps and episodes: {:.2f}".format(lambda_mean))
        print("# visits: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})".format(nvisits_min, nvisits_mean, nvisits_max))

        episode_step = max(1, int(nepisodes / nplots))      # Ex: int(50/4) = 48/4 = 12 
        for idx_ax, ax in enumerate(axes.reshape(nplots)):
            episode_begin = idx_ax*episode_step             # Ex: 0, 12, 24, 36
            episode_end = episode_begin + episode_step - 1 if idx_ax < nplots - 1 else nepisodes - 1
                ## Ex: 11, 23, 35, 49 (for the last block of episodes, include ALL remaining episodes)
            #print("Plotting lambdas for episodes from {} to {}...".format(episode_begin+1, episode_end+1))
            lambdas_by_state = [[] for _ in self.env.getAllStates()]
            nvisits_by_state = np.zeros(self.env.getNumStates())
            for e in episodes_to_consider(episode_begin, episode_end):
                for s in self.env.getAllStates():
                    lambdas_by_state[s] += self._all_lambdas_by_episode[e][s]
                    nvisits_by_state[s] += len(self._all_lambdas_by_episode[e][s])
            states2plot = [s for s in self.env.getNonTerminalStates() if nvisits_by_state[s] > 0]
            #print("lambdas_by_state for plotting:")
            #print([lambdas_by_state[s] for s in states2plot])
            lambda_min_episodes = np.min( [np.min(lambdas_by_state[s]) for s in states2plot] )
            lambda_mean_episodes = np.sum( [np.sum(lambdas) for lambdas in lambdas_by_state] ) / np.sum( [n for n in nvisits_by_state] )
            lambda_max_episodes = np.max( [np.max(lambdas_by_state[s]) for s in states2plot] )
            nvisits_min_episodes = np.min( [n for n in nvisits_by_state] )
            nvisits_mean_episodes = np.mean( [n for n in nvisits_by_state] )
            nvisits_max_episodes = np.max( [n for n in nvisits_by_state] )
            print("\nLambdas over episodes {} thru {}: (min, mean, max) = ({:.2f}, {:.2f}, {:.2f})" \
                  .format(episode_begin+1, episode_end+1, lambda_min_episodes, lambda_mean_episodes, lambda_max_episodes))
            print("# visits over episodes {} thru {}: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})" \
                   .format(episode_begin+1, episode_end+1, nvisits_min_episodes, nvisits_mean_episodes, nvisits_max_episodes))

            if self.plotwhat == "boxplots":
                # Note: the violinpot() function does NOT accept an empty list for plotting nor NaN values
                # (when at least a NaN value is present, nothing is shown for the corresponding group!)
                plotting.violinplot(ax, [lambdas_by_state[s] for s in states2plot], positions=states2plot,
                                    color_body="orange", color_lines="orange", color_means="red")
                finalize_plot_1D(ax, nvisits_by_state, title="Episodes {} thru {}\nlambdas: (min, mean, max) = ({:.2f}, {:.2f}, {:.2f})" \
                                 .format(episode_begin+1, episode_end+1, lambda_min_episodes, lambda_mean_episodes, lambda_max_episodes))
                fig.suptitle("Lambdas distribution by state for different time periods of the experiment")
            elif self.plotwhat == "average":
                # Average lambda values by state
                lambda_mean_by_state = [np.mean(lambdas_by_state[s]) if s in states2plot
                                                                     else np.nan
                                                                     for s in self.env.getAllStates()]
                lambda_se_by_state = [np.std(lambdas_by_state[s]) / np.sqrt(nvisits_by_state[s]) if s in states2plot
                                                                                                 else np.nan
                                                                                                 for s in self.env.getAllStates()]
                if self.env.getDimension() == 2:
                    # Prepare data to plot 
                    shape = self.env.getShape()
                    lambda_mean_by_state_2D = np.asarray(lambda_mean_by_state).reshape(shape)
                    nvisits_by_state_2D = np.asarray(nvisits_by_state).reshape(shape)

                    # Display the 2D images
                    # --lambdas distribution
                    colormap = cm.get_cmap("Oranges")
                    colornorm = plt.Normalize(vmin=0.0, vmax=1.0)
                    ax.imshow(lambda_mean_by_state_2D, cmap=colormap, norm=colornorm)
                    finalize_plot_2D(ax, lambda_mean_by_state_2D, nvisits_by_state_2D, fontsize=self.fontsize, title="Episodes {} thru {}\nlambdas: (min, mean, max) = ({:.2f}, {:.2f}, {:.2f})" \
                                 .format(episode_begin+1, episode_end+1, lambda_min_episodes, lambda_mean_episodes, lambda_max_episodes))

                    # --state count distribution
                    colormap = cm.get_cmap("Blues")
                    colornorm = plt.Normalize(vmin=np.min(nvisits_by_state), vmax=np.max(nvisits_by_state))
                    axc = axes_counts[int(idx_ax/2), idx_ax%2]
                    axc.imshow(nvisits_by_state_2D, cmap=colormap, norm=colornorm)
                    finalize_plot_2D(axc, None, nvisits_by_state_2D, fontsize=int(self.fontsize*1.5), title="Episodes {} thru {}\nstate count: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})" \
                                 .format(episode_begin+1, episode_end+1, nvisits_min_episodes, nvisits_mean_episodes, nvisits_max_episodes))
                else:
                    # For 1D or dimensions higher than 2, just plot lambdas in terms of the 1D state numbers
                    ax.errorbar(states2plot, lambda_mean_by_state, yerr=lambda_se_by_state, capsize=4, color="orange")
                    finalize_plot_1D(ax, nvisits_by_state, title="Episodes {} thru {}\nlambdas: (min, mean, max) = ({:.2f}, {:.2f}, {:.2f})" \
                                     .format(episode_begin+1, episode_end+1, lambda_min_episodes, lambda_mean_episodes, lambda_max_episodes))

                fig.suptitle("Lambda by state averaged over episodes in different time periods of the experiment" \
                             "\n(overall lambda average = {:.2f})".format(lambda_mean))
                fig_counts.suptitle("State count distribution over episodes in different time periods of the experiment")
