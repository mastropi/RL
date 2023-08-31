# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:13:36 2021

@author: Daniel Mastropietro
@description: Policy Learners
"""
import copy

import numpy as np
import pandas as pd

from Python.lib.agents.learners import GenericLearner
from Python.lib.utils.basic import find, is_scalar
#from Python.lib.agents.policies.parameterized import AcceptPolicyType   # NEW: TO IMPLEMENT IN ORDER TO DECIDE WHETHER THE ACCEPTANCE POLICY IS OF THRESHOLD TYPE OR TRUNK-RESERVATION

# Minimum value allowed for the theta parameter of a parameterized policy
# NOTE: Avoid an integer value as lower bound of theta because the estimated gradient of the average state value
# may be zero at integer values of theta (depending on the method used to compute it ).
# We use a negative theta close to -1.0 (and not a theta that is close to 0, such as 0.1)
# so that the deterministic blocking size of the linear step policy can be K=0 (as K = ceiling(theta + 1) in that case)
# TODO: (2023/03/02) Probably this definition should go in parameterized.py and should be defined for each parameterized policy defined there.
THETA_MIN = -1.0 + 0.1


# TODO: (2023/02/27) Create a subclass of LeaPolicyGradient called LeaPolicyGradientOnQueues that should be used for policy gradient learners applied on queues
# In such policy gradient learner applied on queues, we would be able to confidently talk about acceptance policies (as these are natural policies of agents interacting with queues),
# which in a generic policy gradient learner as the one defined here, an acceptance policy may NOT be natural (or required).
# Once we define the new LeaPolicyGradientOnQueues class, we should probably move the different learn() methods currently defined in this class
# because they refer to acceptance policies for incoming jobs to queues.
class LeaPolicyGradient(GenericLearner):
    """
    Policy gradient learner using step size `alpha` and `learnerV` as learner of the state value function

    Arguments:
    env: gym.Env
        The environment where the learning takes place.

    policy: Parameterized policy
        Parameterized policy whose parameter theta is learned.
        This policy should be an object instantiated to one of the policy classes defined in agent.learners.policies,
        so that when the agent learns the policy, that object is also updated with the learned theta parameter value.

    learnerV: GenericLearner
        Learner of the state value function.

    alpha: (opt) float
        Learning rate. This is the initial learning rate when the value of alpha is adjusted as simulation progresses.
        default: 1.0

    fixed_window: (opt) bool
        Whether to use a fixed window to estimate the return G(t) for the update of parameter theta.
        In that case G(t) is estimated for t=0, 1, ..., int(T/2) (or similar).
        default: False

    clipping: (opt) bool
        Whether to clip the absolute change of the delta parameter at each learning step to a specified maximum.
        default: False

    clipping_value: (opt) float
        Maximum absolute change of the delta parameter at each learning step.
        Only used when clipping = True.
        default: +1.0
    """

    def __init__(self, env, policy, learnerV, alpha=1.0,
                 adjust_alpha=False,
                 func_adjust_alpha=None,
                 min_count_to_update_alpha=0,
                 min_time_to_update_alpha=0,
                 alpha_min=0.,
                 fixed_window=False,
                 clipping=False, clipping_value=+1.0,
                 debug=False):
        super().__init__(env, alpha=alpha, adjust_alpha=adjust_alpha, func_adjust_alpha=func_adjust_alpha, min_count_to_update_alpha=min_count_to_update_alpha, min_time_to_update_alpha=min_time_to_update_alpha, alpha_min=alpha_min)
        self.debug = debug

        self.fixed_window = fixed_window is None and False or fixed_window
        self.clipping = clipping
        self.clipping_value = clipping_value

        self.policy = policy
        if isinstance(self.policy, list):
            self.is_multi_policy = True
        else:
            self.is_multi_policy = False

        self.learnerV = learnerV
        self.averageRewardUnderPolicy = 0.0

        # Attributes that store the history of the learning process
        self.thetas = []            # History of theta parameters
        self.gradients = []         # History of gradients
        self.log_gradients = []     # History of log gradients

    def reset_value_functions(self):
        self.reset_policy()
        self.learnerV.reset()

    def is_theta_unidimensional(self, theta):
        return not self.is_multi_policy or self.is_multi_policy and len(theta) == 1

    def store_theta(self, theta):
        "Stores the theta among the stored historical theta values"
        # TODO: (2021/11/04) Having renamed this method from `update_thetas` to `store_theta()`, make the same standardization change in `discrete.Learner._update_alphas()`
        # NOTE: We store a COPY of the parameter passed here because otherwise any changes that are done at theta in the outside world
        # will also be done in the element that is added here to self.thetas!!
        # It already happened... when learning theta, the learn*() method that updates theta makes the value of theta
        # initially stored here be updated when theta is updated!
        # Note that if theta is a scalar, copy.deepcopy() still works fine.
        self.thetas += [copy.deepcopy(theta)]

    def store_gradient(self, state, action, gradient):
        """
        Stores a new gradient into the historical gradients seen by the learning process

        Arguments:
        state: Environment dependent (e.g. if the environment is a queue, it may be the tuple (queue_state, job_class))
            State of the environment on which the policy gradient value given in `gradient` was computed.
            This is only used to compute the log(gradient).

        action: Environment dependent
            Action on which the policy gradient value given in `gradient` was computed.
            This is only used to compute the log(gradient).

        gradient: float
            Policy gradient value to store.
        """
        self.gradients += [gradient]

        if self.is_multi_policy:
            if gradient is None or len(gradient) != len(self.policy):
                raise ValueError("The `gradient` parameter must not be None and must have the same length as the number of learner policies defined in the object (gradient={})".format(gradient))
            log_gradient = [np.nan]*len(self.policy)
            for i, pol in enumerate(self.policy):
                state_i = self.getUnidimensionalStateFromMultidimensionalState(state, i)
                policy_value_i = pol.getPolicyForAction(action, state_i)
                log_gradient[i] = gradient[i] / policy_value_i if policy_value_i != 0.0 else 0.0 if gradient[i] == 0.0 else np.nan
        else:
            policy_value = self.policy.getPolicyForAction(action, state)
            log_gradient = gradient / policy_value if policy_value != 0.0 else 0.0 if gradient == 0.0 else np.nan
        self.log_gradients += [log_gradient]

    def estimateAverageRewardUnderPolicy(self):
        """
        Estimates the average reward received by the agent under the policy used *throughout the simulation*
        (i.e. the states, actions and rewards are only those seen in the CURRENT simulation
        --as opposed to those seen throughout the WHOLE policy learning process--
        because we need to consider only the CURRENT policy (responsible for generating the
        CURRENT simulation), which is parameterized by the CURRENT theta value).

        This is the `rho` baseline value used in the Trunk Reservation paper, pag. 4.
        """
        # Not needed
        #T = len([a for a in self.getActions() if a is not None])
        self.averageRewardUnderPolicy = np.nanmean( [r # * self.policy.getPolicyForAction(a, s)
                                                        for a, r in zip(self.learnerV.getActions(), self.learnerV.getRewards())
                                                        if a is not None] )

    def computeCorrectedReturn(self, baseline):
        """
        Computes the return G(t) where rewards are corrected by the given baseline, rho.
        Only the time steps where the agent interacts with the environment are taken into consideration.

        Arguments:
        baseline: float
            Baseline used to correct each reward in the computation of G(t) as sum_{s>=t}{ reward(s) - rho }

        window_fixed: (opt) bool
            Whether to compute G(t) on a fixed time window, which is appropriate when G(t) is an estimate of the
            state value function defined for non-discounted non-episodic environments, where the value function
            is defined w.r.t. the average reward observed on the simulation process.
            default: False

        Return: numpy array
        Array containing the value of the return G(t) at each time step t in the simulation history, either whether
        an action is taken by the agent or not. However, only rewards associated to actions are taken into consideration.
        """
        def comment_VersionUsingExplicitIterationOnTimeProvedToGiveTheSameResultAsTheOtherImplementationWithCumsum():
            rewards_history = self.learnerV.getRewards()
            actions_history = self.learnerV.getActions()
            T = len(rewards_history)
            assert T > 0

            # Initialize the output array
            G = np.zeros(T)
            if actions_history[T-1] is not None:
                G[T-1] = rewards_history[T-1] - baseline
            for t in range(T-2, -1, -1):  # This is T-2, T-3, ..., 0
                reward = rewards_history[t]
                action = actions_history[t]
                if action is not None:
                    delta_reward = reward - baseline
                    G[t] = delta_reward + G[t+1]
                else:
                    G[t] = G[t+1]

            return G

        def not_needed_because_G_is_already_computed_in_learnerV():
            rewards_history = [r for r in self.learnerV.getRewards() if not np.isnan(r)]
            actions_history = [a for a in self.learnerV.getActions() if a is not None]
            rewards_history.reverse()
            actions_history.reverse()
            delta_rewards_history = np.array(rewards_history) - baseline
                ## 2021/11/28: We must consider only the historical records where there is an action.
                ## Note that as of today we are also storing in the history the first DEATH following a BIRTH
                ## (for which the action is None) with the purpose of plotting the transition of the latest BIRTH
                ## and avoid any suspicion that the implementation is correct just because in the trajectory plot
                ## we observe that the buffer size does not change between two consecutive time steps.
            G = np.cumsum(delta_rewards_history)

            # Reverse the G just computed because we want to have t indexing the values of G from left to right of the array
            return G[::-1]

        # Get the return from the learner of the value functions
        G = not_needed_because_G_is_already_computed_in_learnerV()
        #G = [g for g, a in zip(self.learnerV.getReturn(), self.learnerV.getActions()) if a is not None]
        return G

    def computeCorrectedReturnOnFixedWindow(self, baseline, W):
        """
        Computes the return G(t) where rewards are corrected by the given baseline, rho, on a fixed window W.
        Only the time steps where the agent interacts with the environment are taken into consideration.

        Arguments:
        baseline: float
            Baseline used to correct each reward in the computation of G(t) as sum_{s=t}^{s=t+T-1}{ reward(s) - rho }.

        W: int
            Size of the window on which each value of G(t) is computed.

        Return: numpy array
        Array containing the value of the return G(t) at each time step t in the simulation history from 0 to T-W+1.
        """
        if not (W > 0 and W < len(self.learnerV.getRewards())):
            raise ValueError("The window size W ({}) must be between 1 and the length of the episode on which the return is computed ({})." \
                             .format(W, len(self.learnerV.getRewards())))
        # Make sure W is integer
        W = int(W)

        rewards_history = [r for r in self.learnerV.getRewards() if not np.isnan(r)]
        actions_history = [a for a in self.learnerV.getActions() if a is not None]
            ## 2021/11/28: We must consider only the historical records where there is an action and consequently a non-NaN reward.
            ## Note that as of today we are also storing in the history the first DEATH following a BIRTH
            ## (for which the action is None) with the purpose of plotting the transition of the latest BIRTH
            ## and avoid any suspicion that the implementation is correct just because in the trajectory plot
            ## we observe that the buffer size does not change between two consecutive time steps.
        assert len(rewards_history) == len(actions_history)
        T = len(actions_history)
        assert T > 0

        # Initialize the output array
        G = np.zeros(T - W + 1)
        G[0] = np.sum(np.array(rewards_history[:W]) - baseline)
        for t in range(1, T - W + 1):
            # Update G(t) by removing the first corrected reward value at t and summing the new corrected reward value at t+W-1
            # (as long as the agent did an action at those times)
            G[t] = G[t-1] - (rewards_history[t-1] - baseline) + (rewards_history[t+W-1] - baseline)
            #print("t={}, G(t-1)={:.1f}, G(t)={:.1f}".format(t, G[t-1], G[t]))
        assert t == T - W

        return G

    def learn_update_theta_at_end_of_episode(self, T):
        """
        Learns the policy by updating the theta parameter at the end of the episode using the average gradient
        observed throughout the episode, i.e. as the average of G(t) * grad( log(Pol(A(t)|S(t),theta) ) over all t.

        When the fixed_window attribute is True, a fixed window is used to compute G(t) for each t.
        In that case, the size of the window is T/2 and G(t) is estimated for t = 0, 1, ..., int(T/2).
        Otherwise, G(t) is estimated for t = 0, 1, ..., T with a window that shrinks with t.

        Arguments:
        T: int
            Time corresponding to the end of simulation at which learning takes place.

        Return: tuple
        Tuple with the following elements:
        - theta: theta value before its update.
        - theta_next: theta value after its update.
        - None: this None element is added so that the return tuple is of the same size as the tuple returned by
        learn_linear_theoretical_from_estimated_values(), where it contains the percent coverage of visit to blocking states.
        - V: the average reward observed in the episodic simulation using the theta value before its update.
        - gradV: gradient of the value function that generated the update on theta.
        - deltas: array with the delta innovation values for each simulation time step which is obtained by bootstrapping.
            This is normally equal to G(t). For the case when attribute fixed_window = True, this array is equal to NaN
            for int((T+1)/2) < t <= T+1
        """
        theta = self.getThetaParameter()
        theta_prev = copy.deepcopy(theta)

        # Compute the baseline-corrected return where the baseline is the estimation of the average reward under the policy
        self.estimateAverageRewardUnderPolicy()

        # Observed return for every t: this should be the return already adjusted by the baseline
        # NOTE: (2021/11/04) This baseline adjustment is essential to the convergence of theta, i.e. to observing
        # negative Delta(theta) when there are a few large negative rewards making theta shrink to a smaller value
        # (as the reward is so negative because theta is too large and in our model the cost of blocking increases
        # exponentially with theta for large theta values...)
        if self.fixed_window:
            G = self.computeCorrectedReturnOnFixedWindow(self.getAverageRewardUnderPolicy(), int((T+1)/2))
        else:
            G = self.computeCorrectedReturn(self.getAverageRewardUnderPolicy())
        # Construct the array of delta values that provide the innovation information
        # (in general from bootstrapping, but in this case delta(t) is simply G(t))
        # as the values of G(t) for as long as G(t) is defined... then complete with NaN.
        # Note that no NaN completion is necessary when we are NOT computing G(t) on a fixed window
        # because in that case t goes from 0 to T+1, which is also len(G).
        deltas = np.r_[G, np.nan*np.ones(T+1-len(G))]

        # Trajectory used to compute the gradient of the policy used in the update of theta
        states = [s for s, a in zip(self.learnerV.getStates(), self.learnerV.getActions()) if a is not None]
        actions = [a for a in self.learnerV.getActions() if a is not None]
        rewards = [s for s, a in zip(self.learnerV.getRewards(), self.learnerV.getActions()) if a is not None]
        print("\n--- POLICY LEARNING ---")
        print("[LeaPolicyGradient.learn_update_theta_at_end_of_episode] theta = {}".format(self.getThetaParameter()))
        print("[LeaPolicyGradient.learn_update_theta_at_end_of_episode] Average reward (from V learner) = {}".format(self.learnerV.getAverageReward()))
        print("[LeaPolicyGradient.learn_update_theta_at_end_of_episode] Average reward under policy (baseline for G(t)) rho = {}".format(self.getAverageRewardUnderPolicy()))
        print("[LeaPolicyGradient.learn_update_theta_at_end_of_episode] TRAJECTORY (t, state, action, reward, reward - rho, delta = G, gradient(log(Pol)))")
        df_trajectory = pd.DataFrame( np.c_[range(T+1), [self.env.getBufferSizeFromState(s) for s in states], actions,
                                            rewards, [r - self.getAverageRewardUnderPolicy() for r in rewards], deltas,
                                            [self.getGradientLog(a, s) for a, s in zip(actions, states)],
                                            [delta * self.getGradientLog(a, s) for a, s, delta in zip(actions, states, deltas)]] )
        df_trajectory.columns = ['t', 'state', 'action', 'reward', 'reward - rho', 'G(t)', 'grad(log(Pol))', 'grad(V)']
        print(df_trajectory[ [grad != 0.0 and grad is not None or r != 0.0 for r, grad in zip(df_trajectory['reward'], df_trajectory['grad(log(Pol))']) ]])
        #print(df_trajectory)

        # Estimate the gradient of the value function V(theta), called J(theta) in Sutton, pag. 327
        gradV = 0.0
        nactions = 0                # Number of actions taken by the agent during the trajectory (used to estimate gradV)
        for t in range(T+1):
            # Store the current theta value
            # NOTES:
            # a) Theta is stored BEFORE ITS UPDATE (if any), so that when we plot the historic theta values
            # together with the historic states and rewards, we compare the theta value BEFORE any possible
            # update of theta coming from the action and reward plotted for the corresponding same time step.
            # b) Theta is stored as part of the history even if it has not been updated in the previous iteration,
            # so that its values are aligned with the history of rewards received and actions taken which make it
            # possible to have an informative plot.
            self.store_theta(theta)

            state = states[t]
            action = actions[t]
            reward = rewards[t]
            delta = deltas[t]

            if not np.isnan(delta) and action is not None:
                # Increase the count of actions taken by the agent, which is the denominator used when estimating grad(V)
                # Note: the action may be None if for instance the environment experienced a completed service
                # in which case there is no action to take... (in terms of Accept or Reject)
                # Note also that we check that delta is not NaN because NaN indicates that the current simulation time step
                # is NOT part of the calculation of deltas (which happens when attribute fixed_window = True)
                nactions += 1

                gradLogPol = self.getGradientLog(action, state)
                if gradLogPol != 0.0:
                    # Update the estimated gradient
                    print("[LeaPolicyGradient.learn_update_theta_at_end_of_episode] Learning at simulation time t={}, state={}, action={}, reward={}...".format(t, state, action, reward))
                    print("\t[LeaPolicyGradient.learn_update_theta_at_end_of_episode] t={}: Delta(t) = corrected G(t) = {:.3f}".format(t, delta))
                    print("\t[LeaPolicyGradient.learn_update_theta_at_end_of_episode] t={}: Log policy gradient for Pol(A(t)={}/S(t)={} ) = {}" \
                          .format(t, action, state, gradLogPol))
                    print("\t[LeaPolicyGradient.learn_update_theta_at_end_of_episode] #actions taken by the agent so far = {}".format(nactions))
                    gradV += delta * gradLogPol
                    #gradV += (reward - self.getAverageRewardUnderPolicy()) * gradLogPol    # This computation of the gradient is NOT useful for the learning process

        # Estimated grad(V) as the average of G(t) * grad(log(Pol(t)))
        gradV /= nactions

        # Note that we bound the delta theta to avoid too large changes!
        # We use "strange" numbers to avoid having theta fall always on the same distance from an integer (e.g. 4.1, 5.1, etc.)
        # The lower and upper bounds are asymmetric (larger lower bound) so that a large negative reward can lead to a large reduction of theta.
        bound_delta_theta_upper = +np.Inf if not self.clipping else self.clipping_value  #+1.131
        bound_delta_theta_lower = -np.Inf if not self.clipping else -self.clipping_value  #-1.131 #-5.312314 #-1.0
        delta_theta = np.max([ bound_delta_theta_lower, np.min([self.getLearningRate() * gradV, bound_delta_theta_upper]) ])
        print("[LeaPolicyGradient.learn_update_theta_at_end_of_episode] Estimated grad(V(theta)) = {}".format(gradV))
        print("[LeaPolicyGradient.learn_update_theta_at_end_of_episode] Delta(theta) = alpha * grad(V) = {}".format(delta_theta))

        theta_lower = THETA_MIN
        theta = np.max([theta_lower, theta + delta_theta])

        # Update the policy on the new theta and record it into the history of its updates
        self.setThetaParameter(theta)

        return theta_prev, theta, None, self.getAverageRewardUnderPolicy(), gradV, deltas

    def learn_update_theta_at_each_time_step(self, T):
        """
        Learns the policy by updating its theta parameter at each time step of the Markov chain.
        Nevertheless, the policy is updated (i.e. the value of the theta parameter is set) ONLY at the end of the episode.
        This is the way learning happens in the Trunk Reservation paper by Massaro et al. (2019).
        This learning is supposed to give the same results as when the theta parameter is updated only at the end of
        the episode, using the average gradient observed throughout the time steps of the Markov chain, as done by
        learn_update_theta_at_end_of_episode(). However, the update of theta done here at every time step could be
        bounded at each time step (e.g. when self.clipping = True), therefore results may not be exactly the same.

        When the fixed_window attribute is True, a fixed window is used to compute G(t) for each t.
        In that case, the size of the window is T/2 and G(t) is estimated for t = 0, 1, ..., int(T/2).
        Otherwise, G(t) is estimated for t = 0, 1, ..., T with a window that shrinks with t.

        Arguments:
        T: int
            Time corresponding to the end of simulation at which learning takes place.
            The simulation is assumed to start at time = 0, meaning there are T+1 simulation steps.

        Return: tuple
        Tuple with the following elements:
        - theta: theta value before its final update.
        - theta_next: theta value after its final update.
        - None: this None element is added so that the return tuple is of the same size as the tuple returned by
        learn_linear_theoretical_from_estimated_values(), where it contains the percent coverage of visit to blocking states.
        - V: the average reward observed in the episodic simulation using the theta value before its update.
        - gradV: the average of the observed gradients of the value function throughout the episode.
        - deltas: array with the delta innovation values for each simulation time step which is obtained by bootstrapping.
            This is normally equal to G(t). For the case when attribute fixed_window = True, this array is equal to NaN
            for int((T+1)/2) < t <= T+1
        """
        theta = self.getThetaParameter()
        theta_prev = copy.deepcopy(theta)

        # Compute the baseline-corrected return where the baseline is the estimation of the average reward under the policy
        self.estimateAverageRewardUnderPolicy()

        # Observed return for every t: this should be the return already adjusted by the baseline
        if self.fixed_window:
            G = self.computeCorrectedReturnOnFixedWindow(self.getAverageRewardUnderPolicy(), int((T+1)/2))
        else:
            G = self.computeCorrectedReturn(self.getAverageRewardUnderPolicy())
        # Complete delta with NaN values at the end after the last calculated value of G
        deltas = np.r_[G, np.nan*np.ones(T+1-len(G))]

        # Trajectory used to compute the gradient of the policy used in the update of theta
        states = [s for s, a in zip(self.learnerV.getStates(), self.learnerV.getActions()) if a is not None]
        actions = [a for a in self.learnerV.getActions() if a is not None]
        rewards = [s for s, a in zip(self.learnerV.getRewards(), self.learnerV.getActions()) if a is not None]
        print("\n--- POLICY LEARNING ---")
        print("[LeaPolicyGradient.learn_update_theta_at_each_time_step] theta = {}".format(self.getThetaParameter()))
        print("[LeaPolicyGradient.learn_update_theta_at_each_time_step] Average reward (from V learner) = {}".format(self.learnerV.getAverageReward()))
        print("[LeaPolicyGradient.learn_update_theta_at_each_time_step] Average reward under policy (baseline for G(t)) = {}".format(self.getAverageRewardUnderPolicy()))
        print("[LeaPolicyGradient.learn_update_theta_at_each_time_step] TRAJECTORY (t, state, action, reward, reward - rho, delta = G)")
        df_trajectory = pd.DataFrame( np.c_[range(T+1), [self.env.getBufferSizeFromState(s) for s in states], actions,
                                            rewards, [r - self.getAverageRewardUnderPolicy() for r in rewards],
                                            deltas,
                                            [self.getGradientLog(a, s) for a, s in zip(actions, states)],
                                            [delta * self.getGradientLog(a, s) for a, s, delta in zip(actions, states, deltas)]] )
        df_trajectory.columns = ['t', 'state', 'action', 'reward', 'reward - rho', 'G(t)', 'grad(log(Pol))', 'grad(V)']
        print(df_trajectory[ [grad != 0.0 and grad is not None or r != 0.0 for r, grad in zip(df_trajectory['reward'], df_trajectory['grad(log(Pol))']) ]])
        #print(df_trajectory)

        # Go over every time step of the trajectory
        gradV_mean = 0.0
        n_theta_updates = 0
        for t in range(T+1):
            # Store the current theta value
            # NOTES:
            # a) Theta is stored BEFORE ITS UPDATE (if any), so that when we plot the historic theta values
            # together with the historic states and rewards, we compare the theta value BEFORE any possible
            # update of theta coming from the action and reward plotted for the corresponding same time step.
            # b) Theta is stored as part of the history even if it has not been updated in the previous iteration,
            # so that its values are aligned with the history of rewards received and actions taken which make it
            # possible to have an informative plot.
            self.store_theta(theta)

            state = states[t]
            action = actions[t]
            reward = rewards[t]
            delta = deltas[t]

            if delta != 0.0 and not np.isnan(delta) and action is not None:
                # Note: the action may be None if for instance the environment experienced a completed service
                # in which case there is no action to take... (in terms of Accept or Reject)
                # Note also that we check that delta is not NaN because NaN indicate that the current simulation time step
                # is NOT part of the calculation of deltas (which happens when attribute fixed_window = True)

                if self.getGradientLog(action, state) != 0.0:
                    # [DONE-2021/11/28] DM-2021/11/25: STOP AT LEARNING STEP 24 (starting at theta = 11.3) FOR METHOD 2 TO INVESTIGATE WHY A REJECTION OCCURS WHEN STATE = 2 AND THETA = 4...
                    # R: The reason is that the accept/reject policy is updated ONLY AT THE END of the episode!! which means that the value of theta is not always related to the occurrence of rejections.
                    print("[LeaPolicyGradient.learn_update_theta_at_each_time_step] Learning at simulation time t={}, theta={}, state={}, action={}, reward={}...".format(t, theta, state, action, reward))
                    print("\t[LeaPolicyGradient.learn_update_theta_at_each_time_step] t={}: Delta(t) = G(t) = {:.3f}".format(t, delta))
                    print("\t[LeaPolicyGradient.learn_update_theta_at_each_time_step] t={}: Log-policy gradient for Pol(A(t)={}/S(t)={} ) = {:.6f}" \
                          .format(t, action, state, self.getGradientLog(action, state)))

                    # Store the gradient (for analysis purposes)
                    gradV = delta * self.getGradientLog(action, state)
                    gradV_mean += gradV

                    # Update alpha based on the action-state visit count
                    # Normally the update ONLY happens when the adjust_alpha attribute of the learner is True,
                    # and in that case, it is computed as the initial alpha defined in the learner divided by
                    # the number of visits to the state-action.
                    # This is why we need to update it BEFORE learning, because the
                    # alpha currently stored in the learner may be an alpha for a completely different state-action
                    # to the one on which we are learning now.
                    alpha = self.update_learning_rate(state, action)
                    # Store the alpha that is going to be used for learning next
                    self.store_learning_rate()

                    print("\t[LeaPolicyGradient.learn_update_theta_at_each_time_step] t={}: alpha(state={}, action={} (n={})) = {}".format(t, state, action, self.getCount(state, action), alpha))

                    # Note that we bound the delta theta to avoid too large changes!
                    #theta += self.getLearningRate() * delta * self.getGradientLog(action, state)
                    bound_delta_theta_upper = +np.Inf if not self.clipping else self.clipping_value  #+1.131
                    bound_delta_theta_lower = -np.Inf if not self.clipping else -self.clipping_value  #-1.131
                    delta_theta = np.max([ bound_delta_theta_lower, np.min([alpha * gradV, bound_delta_theta_upper]) ])
                    print("\t[LeaPolicyGradient.learn_update_theta_at_each_time_step] t={}: delta(theta) = {}".format(t, delta_theta))

                    # Only update the visit count of the state and action when delta(theta) != 0.0 because this means that
                    # the state and action have actually been used to learn the policy.
                    if delta_theta != 0.0:
                        self.update_counts(state, action)
                    theta_lower = THETA_MIN
                    theta = np.max([theta_lower, theta + delta_theta])

                    n_theta_updates += 1

        # Update the policy on the new theta only at the END of the simulation time
        # (this is what the Trunk Reservation paper does: "The new policy is generated at the end of the episode")
        self.setThetaParameter(theta)

        # Compute the average grad(V) observed in the episodic simulation
        if n_theta_updates > 0:
            gradV_mean /= n_theta_updates
        else:
            gradV_mean = None

        return theta_prev, theta, None, self.getAverageRewardUnderPolicy(), gradV_mean, deltas

    def learn_linear_theoretical_from_estimated_values(self, T, probas_stationary, Q_values):
        """
        Learns the policy by updating the theta parameter using gradient ascent and estimating the gradient
        from the theoretical expression of the gradient in the linear step policy (only applies for this type of policy!),
        namely:

        grad(V) = sum_x{ Pr(x) * ( Q(x,1) - Q(x,0) )}

        where the sum is over all states x or buffer sizes at which blocking can occur.
        In the case of a single-buffer queue system, the sum is over just ONE buffer size value, namely K-1, where K
        is the buffer size having deterministic blocking.
        Pr(x) is the stationary distribution of state x or buffer size K-1.

        Arguments:
        probas_stationary: dict
            Dictionary containing (state, stationary probability estimate) pairs for each `state` of interest
            (i.e. every state necessary to learn the theta parameter of the policy).

        Q_values: dict
            Dictionary containing (state, [Q(state, a=0), Q(state, a=1)]) pairs for each `state` of interest
            present in the `probas_stationary` dictionary, where Q(state, a) is the state-action value of action `a`,
            either reject (a=0) or accept (a=1).

        Return: tuple
        Tuple with the following elements:
        - theta: theta value before its update.
        - theta_next: theta value after its update.
        - coverage: percent of blocking states or blocking buffer sizes that has been visited by the simulation.
        There is one percent coverage value per policy threshold, since each threshold has an associated blocking size,
        be it a buffer size (when the acceptance policy blocks based on buffer size) or a job occupancy
        (when the acceptance policy blocks based on job-class occupancy of the arriving job class).
        - Q_mean: dictionary indexed by each state of interest where the estimated stationary probability is > 0 with
        the average of the two state-action values (just to return something as it is not really meaningful).
        - gradV: gradient of the value function responsible for the update on theta.
        - Q_diff: same as Q_mean but the values stored are the difference in the state-action values for each state.
        """
        #---- Auxiliary functions -----
        # TODO: (2023/02/01) Try to remove the `isinstance()` calls here on `x`... if possible. They are intended to accomodate both a 1D queue state represented as a scalar and an n-D queue state represented as a tuple
        # I guess the best solution would be to change the whole code base so that 1D states are ALSO represented as a tuple
        ## NOTE that we impose the condition to compute Qdiff ONLY when x is NOT at the full capacity of the system (np.sum(x) < self.env.getCapacity())
        ## because if this is the case the probability of rejecting an incoming job is 1,
        ## hence the derivative of the acceptance policy is 0 and thus no Qdiff needs to be computed.
        does_state_contribute_to_gradient_component = lambda x, i, K: (not isinstance(x, tuple) and x == K - 1 or isinstance(x, tuple) and x[i] == K - 1) and np.sum(x) < self.env.getCapacity()
        #---- Auxiliary functions -----

        theta = self.getThetaParameter()    # For multi-policy learners, theta is an array (one scalar theta value for each policy being learned) o.w. theta is a scalar
        theta_prev = copy.deepcopy(theta) if self.is_multi_policy else theta

        # Store the value of theta (BEFORE its update) --> for plotting purposes
        # Note that we store just ONE theta, as opposed to repeating theta as many times as simulation steps
        # have been used in each queue simulation episode, because this methodology can be used in either
        # Monte-Carlo mode (where we can follow the trajectory of the single particle being used in the estimation) or
        # Fleming-Viot mode (where there is no one single trajectory to plot that is responsible for learning theta).
        self.store_theta(theta)

        # Get the deterministic blocking values for each policy to be learned
        if self.is_multi_policy:
            Ks = [0]*self.getNumPolicies()
            for i, pol in enumerate(self.policy):
                Ks[i] = pol.getDeterministicBlockingValue()
        else:
            Ks = [self.policy.getDeterministicBlockingValue()]

        # Estimated grad(V)
        gradV = np.zeros(self.getNumPolicies()) if self.is_multi_policy else np.array([0.0])
            ## Note: For single policy learners we still define the gradient as an array in order to define a common code below
            ## that works both for the multidimensional theta case and the unidimensional theta case.
        # Create the dictionaries that will contain the information about the average Q and the Q difference between the two possible actions for each state or buffer size
        Q_mean = dict()
        Q_diff = dict()
        num_states_contributing_to_gradient_component = [0]*len(Ks)
        coverage_states_contributing_to_gradient_component = [0.0]*len(Ks)  # Proportion of states contributing to the i-th gradient component that are used in the calculation of the gradient component (a state is not used if its estimated probability is = 0.0)
        for x, p in probas_stationary.items():
            if not np.isnan(p) and not (0.0 <= p <= 1.0):
                raise ValueError("If not missing, the stationary probability estimate must be between 0 and 1 ({})".format(p))

            # Find the component(s) of the gradient to which the current analyzed state x contributes,
            # namely ALL components i whose occupancy in the system is K(i) - 1, for which the derivative of the linear-step acceptance policy is not zero.
            # NOTE that, for a given x, there may be more than one component satisfying this condition,
            # which happens when more than one component has its value at the blocking value - 1, e.g. x = (0, 3, 5) and the blocking values are Ks = [8, 4, 6]
            # meaning that components 1 and 2 (with values 3 and 5) will contribute to the respective gradient component value.
            for i, K in enumerate(Ks):
                if does_state_contribute_to_gradient_component(x, i, K):
                    # Non-deterministic blocking occurs for the i-th component of the state x when a new job of class i arrives
                    # => The product p(x) * Delta(Q[x]) contributes to the i-th component of the gradient
                    num_states_contributing_to_gradient_component[i] += 1
                    if p > 0.0 and x in Q_values.keys():
                        # The currently analyzed state x is a state for which the Q values have been computed
                        if not isinstance(Q_values[x], (list, np.ndarray)) or len(Q_values[x]) != 2:
                            raise ValueError("Each entry of parameter Q_values (a dictionary) must be a list or numpy array of length 2 ({})".format(Q_values[x]))
                        coverage_states_contributing_to_gradient_component[i] += 1
                        Q_mean[x] = 0.5 * (Q_values[x][1] + Q_values[x][0])
                        Q_diff[x] = Q_values[x][1] - Q_values[x][0]
                        gradV[i] += p * Q_diff[x]
                    # DM-2022/10/30: This is a test to use the estimated probability as 1.0 (idea suggested by Urtzi/Matt on 19-Oct-2022 for the single-server queue system stating that the
                    # resulting algorithm is a stochastic approximation algorithm and therefore should converge if alpha is decreased)
                    #if x in Q_values.keys():
                    #    coverage_states_contributing_to_gradient_component[i] += 1
                    #    Q_mean[x] = 0.5 * (Q_values[x][1] + Q_values[x][0])
                    #    Q_diff[x] = Q_values[x][1] - Q_values[x][0]
                    #    gradV[i] += 1.0 * Q_diff[x]
        # Finalize the computation of the state coverage by computing the *proportion* of covered states
        for i in range(len(Ks)):
            coverage_states_contributing_to_gradient_component[i] = coverage_states_contributing_to_gradient_component[i] / max(1, num_states_contributing_to_gradient_component[i])
        print("")
        print(f"[LeaPolicyGradient.learn_linear_theoretical_from_estimated_values] Number of states contributing to each gradient component: {num_states_contributing_to_gradient_component}")
        print(f"[LeaPolicyGradient.learn_linear_theoretical_from_estimated_values] Coverage:" + "[" + ", ".join(["{:.1f}%".format(c*100) for c in coverage_states_contributing_to_gradient_component]) + "]")

        if self.is_theta_unidimensional(theta):
            # Unidimensional theta case
            if not np.isnan(gradV[0]) and gradV[0] != 0.0:
                # Note that we bound the delta theta to avoid too large changes!
                # We use "strange" numbers to avoid having theta fall always on the same distance from an integer (e.g. 4.1, 5.1, etc.)
                # The lower and upper bounds are asymmetric (larger lower bound) so that a large negative reward can lead to a large reduction of theta.
                # Use +1.0 and -1.0 respectively for CLIPPING delta(theta) to +/- 1.
                bound_delta_theta_upper = +np.Inf if not self.clipping else self.clipping_value #+1.0 #+np.Inf #+1.1234 #+np.Inf #+2.1310  # +1.131
                bound_delta_theta_lower = -np.Inf if not self.clipping else -self.clipping_value #-1.0 #-np.Inf #-1.1234 #-np.Inf #-5.3123  # -1.131 #-5.312314 #-1.0

                #alpha = self.getLearningRate()
                # DM-2021/12/24: An attempt to make alpha proportional to the INVERSE of the second derivative of V,
                # which is a multiple of grad(V), making then grad(V) disappear from alpha*gradV below!
                # In fact, this relationship makes sense when we see that the graph of the theoretical grad(V)
                # And this would mean that we only need to know the SIGN of grad(V).
                # This expression is taken from the computation of Laplacian(V) following the approach outlined in Sutton
                # Chapter 13 (Policy Gradient), which gives:
                #   Laplacian(V) = Pr(K-1) * sum_{x} { P(x | K-1, a=1) - P(x | K-1, a=0) } * grad(V)
                # When we compute the sum over x (using the Poisson-like distribution used in the Trunk-Reservation paper
                # (page 4) of going from K to K-u (for the first term in the sum) and
                # from K-1 to K-1-u (for the second term in the sum), i.e. of serving u jobs between the current job arrival
                # and the next job arrival, and then "trivially" upper bounding that sum.
                # The PROBLEM with this approach is that alpha is proportional to the inverse of grad(V)... and then
                # to compute the new theta, we multiply again by grad(V), so the grad(V) cancels out...
                # (except for the sign, which of course is a crucial ingredient of the theta update!)
                # In addition, the value of alpha happens to be inversely proportional to the stationary probability
                # (which makes its value very big when the stationary probability is very small... thus making theta
                # oscillate largely around the optimum theta...)
                # Ref: handwritten notes written on 21-Dec-2021 thru 24-Dec-2021.
                #alpha = self.getLearningRate() / theta[0] / probas_stationary / np.abs(gradV)
                alpha = self.getLearningRate()
                delta_theta = np.max([bound_delta_theta_lower, np.min([alpha * gradV[0], bound_delta_theta_upper])])
                print("[LeaPolicyGradient.learn_linear_theoretical_from_estimated_values] Estimated grad(V(theta)) = {}".format(gradV[0]))
                print("[LeaPolicyGradient.learn_linear_theoretical_from_estimated_values] Delta(theta) = alpha * grad(V) = {:.3f} * {:e} = {:.6f}".format(alpha, gradV[0], delta_theta))

                theta_lower = THETA_MIN
                theta = np.max([theta_lower, theta + delta_theta])  # NOTE: When theta is an array, this returns a scalar!

                # Update the policy on the new theta and record it into the history of its updates
                if self.is_multi_policy:
                    # Convert back theta to an array
                    theta = np.array([theta])
                self.setThetaParameter(theta)
        else:
            # theta and gradV are multidimensional arrays
            if not any(np.isnan([g for g in gradV])):
                assert isinstance(theta, np.ndarray), "The theta parameter must be an array for multidimensional theta parameters: {}".format(theta)
                assert isinstance(gradV, np.ndarray), "The gradV object storing the policy gradient must be an array for multidimensional theta parameters: {}".format(gradV)
                alpha = self.getLearningRate()

                gradV_norm = np.linalg.norm(gradV)
                nonzero_gradV_components = [np.abs(v) for v in gradV if v != 0.0]
                #if gradV_norm == 0.0:
                if len(nonzero_gradV_components) == 0:
                    # All gradV components are 0! => nothing to update
                    alpha = np.nan
                    delta_theta = 0.0
                else:
                    #-- When adjusting alpha by the gradient magnitude at each learning step
                    #gradV_abs_min_nonzero_component_value = np.min(nonzero_gradV_components)
                    #alpha0 = self.getInitialLearningRate()
                    #shrinkage = alpha0 / alpha
                    #alpha = 1 / gradV_abs_min_nonzero_component_value / shrinkage
                    #-- When adjusting alpha by the gradient magnitude

                    delta_theta = alpha * gradV
                theta += delta_theta
                # Lower bound theta so that it doesn't go negative or to zero, values which do not make sense for the policy
                theta = np.max([theta, np.repeat(THETA_MIN, len(theta))], axis=0) # This computes the maximum between the two arrays, say theta = [3, -5, 0] and [0.1, 0.1, 0.1] by column giving [3, 0.1, 0.1]
                print("[LeaPolicyGradient.learn_linear_theoretical_from_estimated_values] Estimated grad(V(theta)) = {}".format(gradV))
                print("[LeaPolicyGradient.learn_linear_theoretical_from_estimated_values] Delta(theta) = alpha * grad(V) = {:.3f} * {} = {}".format(alpha, gradV, delta_theta))
                print("[LeaPolicyGradient.learn_linear_theoretical_from_estimated_values] theta: {} -> {}".format(theta_prev, theta))

                # Upper bound theta according to the system's capacity (which is fixed)
                # Either for threshold policies or for Trunk Reservation policies, the value of each theta dimension cannot exceed a value that would make
                # the blocking size K(i) for job class i be larger than the system's capacity K.
                K = self.env.getCapacity()
                _theta_orig = copy.deepcopy(theta)
                for i, theta_i in enumerate(theta):
                    # We choose to upper bound theta by K-1 MINUS a small epsilon.
                    # We consider this epsilon so that theta is NOT an integer and learning can continue
                    # --o.w. the derivative of the linear step policy we are dealing with in this method will always be 0 at theta.
                    # This upper bound guarantees that the blocking states defined by theta are *valid states* of the Markov Chain.
                    # Note that we bound theta by K-1 MINUS epsilon, instead of PLUS epsilon so that the deterministic blocking size,
                    # that is equal to ceiling(theta+1), is equal to K.
                    theta[i] = min(theta_i, K - 1 - 0.1)
                print("[LeaPolicyGradient.learn_linear_theoretical_from_estimated_values] theta BOUNDED by system's capacity K={}: {} -> {}".format(K, _theta_orig, theta))

                self.setThetaParameter(theta)

        return theta_prev, theta, coverage_states_contributing_to_gradient_component, Q_mean, gradV, Q_diff

    def learn_linear_theoretical_from_estimated_values_IGA(self, T, probas_stationary: dict, Q_values: dict):
        """
        Learns the policy by updating an integer-valued theta parameter using Integer Gradient Ascent (IGA)

        IGA estimates the left gradient and the right gradient w.r.t. theta from the theoretical expression of
        the gradient in the linear step policy (only applies here!), namely:

        grad_left(V) = Pr(K-1) * ( Q(K-1,1) - Q(K-1,0) )
        grad_right(V) = Pr(K) * ( Q(K,1) - Q(K,0) )

        where Pr(.) is the stationary distribution for the given buffer sizes.

        Arguments:
        probas_stationary: (opt) dict of floats between 0 and 1 (used when estimating the theoretical grad(V))
            Dictionary with the estimated stationary probability for each buffer size defined in its keys.
            The dictionary keys are expected to be K-1 and K.
            default: None

        Q_values: (opt) dict of lists or numpy arrays (used when estimating the theoretical grad(V))
            Dictionary with the estimated state-action values for each buffer size defined in its keys.
            Each element in the list is indexed by the initial action taken when estimating the corresponding Q value.
            The dictionary keys are expected to be K-1 and K.
            default: None

        Return: tuple
        Tuple with the following elements:
        - theta: theta value before its update
        - theta_next: theta value after its update
        - V: the average of the two state-action values (just to return something as it is not really meaningful)
        - gradV: gradient of the value function that generated the update on theta
        - Q_diff: the difference in the state-action values between the accept action and the reject action
        """
        theta = self.getThetaParameter()
        # WE ASSUME THAT theta IS SCALAR!
        theta = theta[0]
        theta_prev = copy.deepcopy(theta)

        # Store the value of theta (BEFORE its update) --> for plotting purposes
        # Note that we store just ONE theta, as opposed to repeating theta as many times as simulation steps
        # have been used in each queue simulation episode, because this methodology can be used in either
        # Monte-Carlo mode (where we can follow the trajectory of the single particle being used in the estimation) or
        # Fleming-Viot mode (where there is no one single trajectory to plot that is responsible for learning theta).
        self.store_theta(theta)

        # Estimated grad(V), left and right
        assert not self.is_multi_policy
        K = self.policy.getDeterministicBlockingValue()

        # Check input parameters
        keys_probas = probas_stationary.keys()
        keys_Q_values = Q_values.keys()
        assert [K - 1, K] == sorted(keys_probas) and [K - 1, K] == sorted(keys_Q_values), \
            "Keys K-1={} and K={} are present in the dictionaries containing the stationary probability estimations ({}) and the Q values ({})" \
                .format(K - 1, K, probas_stationary, Q_values)

        # Left gradient
        Q_mean_Km1 = 0.5 * (Q_values[K-1][1] + Q_values[K-1][0])
        Q_diff_Km1 = Q_values[K-1][1] - Q_values[K-1][0]
        gradV_left = probas_stationary[K-1] * Q_diff_Km1

        # Right gradient
        Q_mean_K = 0.5 * (Q_values[K][1] + Q_values[K][0])
        Q_diff_K = Q_values[K][1] - Q_values[K][0]
        gradV_right = probas_stationary[K] * Q_diff_K

        print("Estimated grad(V(theta)) (left) = {}".format(gradV_left))
        print("Estimated grad(V(theta)) (right) = {}".format(gradV_right))

        if np.sign(gradV_left) == np.sign(gradV_right):
            delta_theta = np.sign(gradV_right)
            print("Delta(theta) = {}".format(delta_theta))
            theta = np.max([1, theta + delta_theta])    # theta lower bounded by 1

            # Update the policy on the new theta and record it into the history of its updates
            self.setThetaParameter(theta)

        return theta_prev, theta, Q_mean_Km1, gradV_left, Q_diff_Km1

    #----- GETTERS -----#
    def getIsMultiPolicy(self):
        return self.is_multi_policy

    def getNumPolicies(self):
        return len(self.policy)

    def getPolicy(self):
        return self.policy

    def getPolicyClass(self):
        if self.is_multi_policy:
            return type(self.policy[0])
        else:
            return type(self.policy)

    def getLearnerV(self):
        return self.learnerV

    def getThetaParameter(self):
        "Returns the unidimensional or multidimensional theta of the parameterized policy"
        if self.is_multi_policy:
            theta = np.nan * np.ones(self.getNumPolicies())
            for i, pol in enumerate(self.policy):
                theta[i] = pol.getThetaParameter()
        else:
            theta = self.policy.getThetaParameter()
        return theta

    def getGradient(self, action, state):
        "Returns the unidimensional or multidimensional gradient for the given action applied on the given state"
        if self.is_multi_policy:
            gradient = [np.nan] * self.getNumPolicies()
            for i, pol in enumerate(self.policy):
                # Extract the real-valued state for the current policy, as the gradient of each paramterized policy
                # is based on a real-valued parameter (as opposed to an array parameter)
                state_i = self.getUnidimensionalStateFromMultidimensionalState(state, i)
                gradient[i] = pol.getGradient(action, state_i)
        else:
            gradient = self.policy.getGradient(action, state)
        return gradient

    def getGradientLog(self, action, state):
        "Returns the log of the unidimensional or multidimensional gradient for the given action applied on the given state"
        if self.is_multi_policy:
            log_gradient = [np.nan] * self.getNumPolicies()
            for i, pol in enumerate(self.policy):
                # Extract the real-valued state for the current policy, as the gradient of each paramterized policy
                # is based on a real-valued parameter (as opposed to an array parameter)
                state_i = self.getUnidimensionalStateFromMultidimensionalState(state, i)
                log_gradient[i] = pol.getGradientLog(action, state_i)
        else:
            log_gradient = self.policy.getGradientLog(action, state)
        return log_gradient

    def getThetas(self):
        return self.thetas

    def getGradients(self):
        "Returns the historical gradient values stored in the object"
        return self.gradients

    def getGradientsLog(self):
        "Returns the historical log-gradient values stored in the object"
        return self.log_gradients

    def getAverageRewardUnderPolicy(self):
        "Returns the observed average reward under the current theta parameter of the policy"
        return self.averageRewardUnderPolicy

    def getUnidimensionalStateFromMultidimensionalState(self, state, dim):
        """
        Extracts the real-valued state corresponding to the given dimension of a multidimensional state,
        based on the state structure defined by the queue environment stored in the object.

        This assumes that the queue environment defines a state based on the queue state itself and on the class
        of the arriving job (if any), which are extracted using methods getQueueStateFromState() and getJobClassFromState().

        Note: The unidimensional and multidimensional characteristic of a state refers to the unidimensional and
        multidimensional characteristic of the QUEUE state itself.
        Ex:
        - unidimensional state: (3, None)           => `3` is unidimensional
        - multidimensional state: ((0, 5, 2), None) => `(0, 5, 2)` is multidimensional
        where in both cases `None` refers to the class of the arriving job, which is None because no job is considered
        to have arrived at the moment of analysis of the Markov chain whose state is given by the queue environment state.

        NOTE: The multidimensional state could also be a scalar value, in which case the output unidimensional state
        coincides with the input unidimensional state.

        IMPORTANT: This method further ASSUMES that the way in which the queue environment stores states is using
        a 2D tuple, where the first element is the QUEUE state itself and the second element is the job class.
        This is reflected in the fact that the returned value of the method is a 2D tuple containing these pieces of information.

        Arguments:
        state: Queue-environment dependent
            Multidimensional queue environment state from which the unidimensional state should be created.
            The multidimensional state could also be unidimensional nd scalar.

        dim: int
            Dimension of interest that should be extracted from the given multidimensional state.

        Return: tuple
        Duple of the form (unidimensional-queue-state, job-class), where `unidimensional-queue-state` is extracted
        from the `dim`-th dimension of the multidimensional QUEUE state in the input `state`. Ex: `(3, None)`.
        """
        queue_state, job_class = self.env.getQueueStateFromState(state), self.env.getJobClassFromState(state)
        queue_state_dim = queue_state if is_scalar(queue_state) else queue_state[dim]
        return (queue_state_dim, job_class)

    #----- SETTERS -----#
    def setThetaParameter(self, theta):
        if self.is_multi_policy:
            if theta is None or is_scalar(theta) or len(theta) != self.getNumPolicies():
                raise ValueError("The parameter value to set `theta` must not be None or scalar and must have the same length as the number of multi-policy learners defined in the object")
            for i, pol in enumerate(self.policy):
                pol.setThetaParameter(theta[i])
        else:
            self.policy.setThetaParameter(theta)

    #----- RESETTERS -----#
    def reset_policy(self):
        if self.is_multi_policy:
            for pol in self.policy:
                pol.reset()
        else:
            self.policy.reset()


if __name__ == "__main__":
    #------------------ Test of computeCorrectedReturn() -------------------#
    # (2023/01/30) Outdated test: it doesn't work any more and it is no longer relevant, as for now we don't use the corrected return in FVRL nor the learnerV learner
    def do_not_run():
        from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses, rewardOnJobRejection_ExponentialCost
        from Python.lib.agents.learners import LearnerTypes
        from Python.lib.agents.learners.continuing.mc import LeaMC
        from Python.lib.agents.learners.policies import LeaPolicyGradient
        from Python.lib.agents.policies import PolicyTypes
        from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep
        from Python.lib.agents.queues import AgeQueue
        from Python.lib.queues import QueueMM
        from Python.lib.simulators.queues import SimulatorQueue, LearningMode

        capacity = np.Inf
        nservers = 1
        job_class_rates = [0.7]
        service_rates = [1.0]
        queue = QueueMM(job_class_rates, service_rates, nservers, capacity)
        env_queue_mm = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, rewardOnJobRejection_ExponentialCost, None)

        # Acceptance/Reject Policy
        theta = 3.9
        policies = dict({PolicyTypes.ACCEPT: PolQueueTwoActionsLinearStep(env_queue_mm, theta), PolicyTypes.ASSIGN: None})
        policy = policies[PolicyTypes.ACCEPT]

        # Policy learner
        alpha = 1.0
        gamma = 1.0
        learnerV = LeaMC(env_queue_mm, gamma=gamma)
        learner = LeaPolicyGradient(env_queue_mm, policy, learnerV)
        learners = dict({LearnerTypes.V: learnerV,
                         LearnerTypes.Q: None,
                         LearnerTypes.P: LeaPolicyGradient(env_queue_mm, policies[PolicyTypes.ACCEPT], learnerV, alpha=alpha)})
        agent_gradient_mc = AgeQueue(env_queue_mm, policies, learners)

        # Simulate the queue one time
        print("Test #1: LeaPolicyGradient.computeCorrectedReturn()")
        # Need a few dummy parameters that are required by the simul.run() method:
        # I set a value for each of them that is not expected to have an impact in the test results.
        theta_true_dummy = 5.0
        buffer_size_activation_factor_dummy = 0.5
        t_sim = 20
        t_learn = 1
        dict_params_learning = dict({'t_learn': t_learn, 'mode': LearningMode.REINFORCE_TRUE})
        start_queue_state = [ np.ceil(theta) ]
        simul = SimulatorQueue(env_queue_mm, agent_gradient_mc, dict_params_learning, debug=False)
        simul.run(dict_params_simul={'t_sim': t_sim, 'theta_start': theta,
                                     'theta_true': theta_true_dummy,
                                     'buffer_size_activation_factor': buffer_size_activation_factor_dummy,
                                     'nparticles': 1}, start_state=(start_queue_state, None), seed=1717, verbose=False)

        learner.estimateAverageRewardUnderPolicy()
        G = learner.computeCorrectedReturn(learner.getAverageRewardUnderPolicy())
        G_expected = np.array([ 0.0,           -9.52380952e-02, 8.09523810e-01, 7.14285714e-01,
                                6.19047619e-01, 5.23809524e-01, 4.28571429e-01, 1.33333333e+00,
                                1.23809524e+00, 1.14285714e+00, 1.04761905e+00, 9.52380952e-01,
                                8.57142857e-01, 7.61904762e-01, 6.66666667e-01, 5.71428571e-01,
                                4.76190476e-01, 3.80952381e-01, 2.85714286e-01, 1.90476190e-01,
                                9.52380952e-02])
        # 2021/11/25: Previous version, before fixing the transition times of the discrete-time embedded (skeleton) Markov Chain
        # We observe that the value of G(t) stays constants for some time steps and this is because at those time steps
        # jobs were served, so no action was taken, therefore no contribution to G(t) is considered.
        #G_expected = np.array([  0.0, -0.2,  0.6, 0.6,
        #                         0.6,  0.6,  0.4, 0.2,
        #                         0.0,  0.0, -0.2, 0.6,
        #                         0.6,  0.6,  0.6, 0.6,
        #                         0.4,  0.4,  0.2, 0.0, 0.0])
        print("Observed G:")
        print(G)
        assert np.allclose(G, G_expected)

        W = 3       # The window size can be up to t_sim + 1
        print("Test #2: LeaPolicyGradient.computeCorrectedReturnOnFixedWindow(W={})".format(W))
        G = learner.computeCorrectedReturnOnFixedWindow(learner.getAverageRewardUnderPolicy(), W)
        G_expected = np.array([ -0.71428571, -0.71428571, 0.28571429, 0.28571429, -0.71428571, -0.71428571,
                                -0.71428571, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429,
                                0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429, 0.28571429,
                                0.28571429])
        # 2021/11/25: Previous version, before fixing the transition times of the discrete-time embedded (skeleton) Markov Chain
        # We observe that the value of G(t) stays constants for some time steps and this is because at those time steps
        # jobs were served, so no action was taken, therefore no contribution to G(t) is considered.
        #G_expected = np,array([-0.6, -0.8,  0.,   0.2,  0.4,  0.6,  0.4,  0.4, -0.6, -0.6, -0.8,  0.,   0.,   0.2, 0.2,  0.4,  0.4,  0.4,  0.2])
        print("Observed G:")
        print(G)
        assert len(G) == (t_sim + 1) - W + 1    # t_sim+1 because the simulation starts at t=0, therefore there are t_sim+1 simulation steps
        assert np.allclose(G, G_expected)
