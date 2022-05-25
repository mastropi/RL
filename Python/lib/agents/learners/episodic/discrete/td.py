# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:57:16 2020

@author: Daniel Mastropietro
@description: Definition of Temporal Difference algorithms.
"""


from enum import Enum, unique

import numpy as np
from matplotlib import pyplot as plt, cm

from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners.episodic.discrete import Learner, AlphaUpdateType
from Python.lib.agents.learners.value_functions import ValueFunctionApprox
import Python.lib.utils.plotting as plotting

@unique  # Unique enumeration values (i.e. on the RHS of the equal sign)
class AdaptiveLambdaType(Enum):
    ATD = 1     # (full) Adaptive TD(lambda)
    HATD = 2    # Homogeneously Adaptive TD(lambda)


class LeaTDLambda(Learner):
    """
    TD(Lambda) learning algorithm using step size `alpha`, discount `gamma`, and decay parameter `lmbda`
    applied to a discrete environment defined with the DiscreteEnv class of openAI's gym module.

    Arguments:
    env: gym.envs.toy_text.discrete.DiscreteEnv
        The environment where the learning takes place.
    """

    def __init__(self, env, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=False, alpha_min=0.,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 debug=False):
        super().__init__(env, alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min,
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed)
        self.debug = debug

        # Attributes that MUST be present for all TD methods
        self.V = ValueFunctionApprox(self.env.getNumStates())
        self.Q = None
        self.alpha = alpha
        self.gamma = gamma
        
        # Attributes specific to the current TD method
        self.lmbda = lmbda
        # Eligibility traces
        self._z = np.zeros(self.env.getNumStates())
        self._z_all = np.zeros((0, self.env.getNumStates()))
        
        # (Nov-2020) Product of alpha and z (the eligibility trace)
        # which gives the EFFECTIVE alpha value of the Stochastic Approximation algorithm
        # Goal: Compare the rate of convergence of the non-adaptive vs. the adaptive TD(lambda)
        # by keeping track of the effective alpha as a FUNCTION of the episode number for EACH STATE
        # Each episode is a different row of the _alphas_effective array and the states are across the columns.
        self._times_nonzero_update = [[] for _ in self.env.all_states]
        self._alphas_effective = np.zeros((0, self.env.getNumStates()))

    def _reset_at_start_of_episode(self):
        super()._reset_at_start_of_episode()
        self._z[:] = 0.
        self._z_all = np.zeros((0, self.env.getNumStates()))
        self._alphas_effective = np.zeros((0, self.env.getNumStates()))

    def setParams(self, alpha=None, gamma=None, lmbda=None, adjust_alpha=None, alpha_update_type=None,
                  adjust_alpha_by_episode=None, alpha_min=None):
        super().setParams(alpha, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.gamma = gamma if gamma is not None else self.gamma
        self.lmbda = lmbda if lmbda is not None else self.lmbda

    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        self._update_trajectory(t, state, reward)
        self._updateZ(state, self.lmbda)
        delta = reward + self.gamma * self.V.getValue(next_state) - self.V.getValue(state)
        #print("episode {}, state {}: count = {}, alpha = {}".format(self.episode, state, self._state_counts_overall[state], self._alphas[state]))
        self._alphas_effective = np.r_[self._alphas_effective, self._alphas[state] * self._z.reshape(1,len(self._z))]
        if delta != 0.0:
            # IMPORTANT: (2020/11/11) if we use _alphas[state] as the learning rate alpha in the following update,
            # we are using the SAME learning rate alpha for the update of ALL states, namely the
            # learning rate value associated to the state that is being visited now.
            # This is NOT how the alpha value should be updated for each state
            # (as we should apply the alpha associated to the state that decreases with the number of visits
            # to EACH state --which happens differently). However, this seems to give slightly faster convergence
            # than the theoretical alpha strategy just mentioned.
            # If we wanted to use the strategy that should be applied in theory, we should simply
            # replace `self._alphas[state]` with `self._alphas` in the below expression. 
            self.V.setWeights( self.V.getWeights() + self._alphas[state] * delta * self._z )

        # Update alpha for the next iteration for "by state counts" update
        if not self.adjust_alpha_by_episode:
            self._update_alphas(state)

        if done:
            if self.debug: #and self.episode > 45: # Use the condition on `episode` in order to plot just the last episodes 
                self._plotZ()
                self._plotAlphasEffective()
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
        self._z = self.gamma * lmbda * self._z + \
                    gradient_V                                    # For every-visit TD(lambda)
                    #gradient_V * (self._state_counts[state] == 1)  # For first-visit TD(lambda)
        self._z_all = np.r_[self._z_all, self._z.reshape(1, len(self._z))]

    def _plotZ(self):
        states2plot = self._choose_states2plot()
        plt.figure()
        plt.plot(self._z_all[:,states2plot], '.-')
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
            states2plot = list(range(int(self.nS/2)-1, int(self.nS/2)+2))

        return states2plot

class LeaTDLambdaAdaptive(LeaTDLambda):
    
    def __init__(self, env, alpha=0.1, gamma=1.0, lmbda=0.8,
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                 adjust_alpha_by_episode=True, alpha_min=0.,
                 lambda_min=0., lambda_max=0.99, adaptive_type=AdaptiveLambdaType.ATD,
                 reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=None,
                 burnin=False, plotwhat="boxplots", fontsize=15, debug=False):
        super().__init__(env, alpha, gamma, lmbda, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min,
                         reset_method=reset_method, reset_params=reset_params, reset_seed=reset_seed,
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
        # (This is used to decide whether we should use the adaptive or non-adaptive lambda
        # based on whether the delta information from which the agent learns already contains
        # bootstrapping information about the value function at the next state)  
        self.state_counts_noreset = np.zeros(self.env.getNumStates())

        #-- Variables used in the HOMOGENEOUS adaptive type case
        self._gradient_V_all = np.zeros((0, self.env.getNumStates()))

        #-- Variables for lambda statistics over all episodes
        # List of lists to store the lambdas used for each state in each episode
        self._lambdas_in_episode = [[] for _ in self.env.all_states]
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
        self._lambdas = []
        self._lambdas_in_episode = [[] for _ in self.env.all_states]

    def setParams(self, alpha=None, gamma=None, lmbda=None, adjust_alpha=None, alpha_update_type=None,
                  adjust_alpha_by_episode=None, alpha_min=None,
                  lambda_min=None, lambda_max=None, adaptive_type=None,
                  burnin=False):
        super().setParams(alpha, gamma, lmbda, adjust_alpha, alpha_update_type, adjust_alpha_by_episode, alpha_min)
        self.lambda_min = lambda_min if lambda_min is not None else self.lambda_min
        self.lambda_max = lambda_max if lambda_max is not None else self.lambda_max
        self.adaptive_type = adaptive_type if adaptive_type is not None else self.adaptive_type
        self.burnin = burnin if burnin is not None else self.burnin

    def learn_pred_V(self, t, state, action, next_state, reward, done, info):
        self._update_trajectory(t, state, reward)

        self.state_counts_noreset[state] += 1
        state_value = self.V.getValue(state)
        delta = reward + self.gamma * self.V.getValue(next_state) - state_value

        # Decide whether we do adaptive or non-adaptive lambda at this point
        # (depending on whether there is bootstrap information available or not)
        # TODO: (2020/04/21) Adapt the check on whether a value function has been modified at least once that works also for the case when the initial state value function is not 0 (e.g. it is random).
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
            #-- Adaptive lambda
            # Define the relative target error delta by dividing delta to a reference value defined below
            #ref_value = state_value        # reference value is the value of the current state
            ref_value = np.mean( np.abs(self.V.getValues()) )   # reference value is the average value over all states
            delta_relative = delta / ref_value if ref_value != 0 \
                                               else 0. if delta == 0. \
                                               else np.Inf
            # Relative delta that prevents division by 0 (i.e. delta_relative = exp(|delta|) / exp(|value|))
            #delta_relative = np.exp( np.abs(delta) - np.abs(state_value) )

            # Compute lambda as a function of delta or relative delta
            lambda_adaptive = min( 1 - (1 - self.lambda_min) * np.exp( -np.abs(delta_relative) ), self.lambda_max )
            #lambda_adaptive = min( 1 - (1 - self.lambda_min) * np.exp( -np.abs(delta) ), self.lambda_max )

        # Keep history of used lambdas
        self._lambdas += [lambda_adaptive]
        # Update the history of lambdas used
        self._lambdas_in_episode[state] += [lambda_adaptive]
        self._all_lambdas_n[state] += 1 
        self._all_lambdas_sum[state] += lambda_adaptive 
        self._all_lambdas_sum2[state] += lambda_adaptive**2 

        # Update the eligibility trace
        self._updateZ(state, lambda_adaptive)
        # Update the weights
        self._alphas_effective = np.r_[self._alphas_effective, self._alphas[state] * self._z.reshape(1, len(self._z))]
        if delta != 0.0:
            self.V.setWeights( self.V.getWeights() + self._alphas[state] * delta * self._z )
            if self.debug:
                print("episode: {}, t: {}: TD ERROR != 0 => delta = {}, delta_rel = {}, lambda (adaptive) = {}\n" \
                        "trajectory so far: {}" \
                        "--> V(s): {}".format(self.episode, t, delta, delta_relative, lambda_adaptive, self._states, self.V.getValues()))

        # Update alpha for the next iteration for "by state counts" update
        if not self.adjust_alpha_by_episode:
            self._update_alphas(state)

        if done:
            if self.debug: # and self.episode > 45:
                self._plotZ()
                self._plotAlphasEffective()
            self.store_trajectory(next_state)
            self._update_state_counts(t+1, next_state)
            self._store_lambdas_in_episode()            

            # Update alpha for the next episode for "by episode" updates
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

    def _updateZ(self, state, lmbda):
        if self.debug and False:
            print("")
            print("state = {}: lambda = {:.2f}".format(state, lmbda))
        if self.adaptive_type == AdaptiveLambdaType.ATD:
            super()._updateZ(state, lmbda)
        else:
            # In the HOMOGENEOUS adaptive lambda we need to store the HISTORY of the gradient
            # (because we need to retroactively apply the newly computed lambda to previous eligibility traces)
            gradient_V = self.V.X[:,state]      # Note: this is returned as a ROW vector, even when we retrieve the `state` COLUMN of matrix X
            # Use the following calculation of the gradient for FIRST-VISIT TD(lambda)
            # (i.e. the gradient is set to 0 if the current visit of `state` is not the first one)
            #gradient_V * (self._state_counts[state] == 1)  # For first-visit TD(lambda)
            self._gradient_V_all = np.r_[self._gradient_V_all, gradient_V.reshape(1, len(gradient_V))]

            if self.debug and False:
                print("Gradients:")
                print(self._gradient_V_all)

            # Compute the exponents of gamma*lambda, which go from n_trace_length-1 down to 0
            # starting with the oldest gradient.
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
            self._z = np.sum( (self.gamma * lmbda)**exponents * self._gradient_V_all, axis=0 )
            self._z_all = np.r_[self._z_all, self._z.reshape(1, len(self._z))]

            if self.debug and False:
                print("Exponents: {}".format((self.gamma * lmbda)**exponents))

        if self.debug and False:
            print("Z's & lambda's by state:")
            print(self._z_all)
            print(self._lambdas_in_episode)

    def _store_lambdas_in_episode(self):
        #print("lambdas in episode {}:".format(self.episode))
        #print(self._lambdas_in_episode)
        self._all_lambdas_by_episode += [[self._lambdas_in_episode[s] for s in self.env.all_states]]

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

            ax.set_xticks(self.env.all_states)
            ax.set_xlim((0, self.env.getNumStates()-1))
            ax.set_ylim((0, 1.02))
            ax.set_ylabel("Lambda")
            ax.tick_params(axis='y', colors="orange")
            ax.yaxis.label.set_color("orange")

            # State count distribution
            ax1sec = ax.twinx()
            ax1sec.bar(self.env.all_states, state_counts, color="blue", alpha=0.2)
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
        #    print(np.c_[self.env.all_states, lambdas_n, lambdas_mean, lambdas_std])
        #ax1.plot(self.env.all_states, lambdas_mean, '.-', color="orange")
        #ax1.errorbar(self.env.all_states, lambdas_mean, yerr=lambdas_std, capsize=4, color="orange")

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
        lambdas_by_state = [[] for _ in self.env.all_states]
        for e in episodes_to_consider(0, nepisodes-1):
            for s in self.env.all_states:
                lambdas_by_state[s] += self._all_lambdas_by_episode[e][s]
        states2plot = [s for s in self.env.getNonTerminalStates() if self._state_counts_overall[s] > 0] 
        #print("lambdas_by_state for plotting:")
        #print([lambdas_by_state[s] for s in states2plot])

        #plt.figure()
        #ax1 = plt.gca()
        # Note: the violinpot() function does NOT accept an empty list for plotting nor NaN values
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
            lambdas_by_state = [[] for _ in self.env.all_states]
            nvisits_by_state = np.zeros(self.env.getNumStates())
            for e in episodes_to_consider(episode_begin, episode_end):
                for s in self.env.all_states:
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
                                                                     for s in self.env.all_states]
                lambda_se_by_state = [np.std(lambdas_by_state[s]) / np.sqrt(nvisits_by_state[s]) if s in states2plot
                                                                                                 else np.nan
                                                                                                 for s in self.env.all_states]
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
                    finalize_plot_2D(axc, None, nvisits_by_state_2D, fontsize=self.fontsize*1.5, title="Episodes {} thru {}\nstate count: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})" \
                                 .format(episode_begin+1, episode_end+1, nvisits_min_episodes, nvisits_mean_episodes, nvisits_max_episodes))
                else:
                    # For 1D or dimensions higher than 2, just plot lambdas in terms of the 1D state numbers
                    ax.errorbar(states2plot, lambda_mean_by_state, yerr=lambda_se_by_state, capsize=4, color="orange")
                    finalize_plot_1D(ax, nvisits_by_state, title="Episodes {} thru {}\nlambdas: (min, mean, max) = ({:.2f}, {:.2f}, {:.2f})" \
                                     .format(episode_begin+1, episode_end+1, lambda_min_episodes, lambda_mean_episodes, lambda_max_episodes))

                fig.suptitle("Lambda by state averaged over episodes in different time periods of the experiment" \
                             "\n(overall lambda average = {:.2f})".format(lambda_mean))
                fig_counts.suptitle("State count distribution over episodes in different time periods of the experiment")
