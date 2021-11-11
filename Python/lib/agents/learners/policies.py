# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:13:36 2021

@author: Daniel Mastropietro
@description: Policy Learners
"""
import copy

import numpy as np
import pandas as pd

from . import Learner, AlphaUpdateType


class LeaPolicyGradient(Learner):
    """
    Policy gradient learner using step size `alpha` and `learnerV` as learner of the state value function

    Arguments:
    env: gym.Env
        The environment where the learning takes place.

    policy: Parameterized policy
        Parameterized policy that is used to learn the parameter theta of the policy.

    learnerV: Learner
        Learner of the state value function.

    alpha: float
        Learning rate.
    """

    def __init__(self, env, policy, learnerV, alpha=0.1,
                 adjust_alpha=False,
                 min_count_to_update_alpha=0,
                 min_time_to_update_alpha=0,
                 alpha_min=0.,
                 debug=False):
        super().__init__(env, alpha, adjust_alpha, min_count_to_update_alpha, min_time_to_update_alpha, alpha_min)
        self.debug = debug

        self.policy = policy
        self.learnerV = learnerV
        self.averageRewardUnderPolicy = 0.0

        # Attributes that store the history of the learning process
        self.gradients = []
        self.log_gradients = []

    def reset_supporting_attributes(self):
        "Resets the counts of the states visited during the learning process"
        self.gradients = []
        self.log_gradients = []

    def reset_value_functions(self):
        self.policy.reset()
        self.learnerV.reset()

    def record_gradient(self, state, action, gradient):
        self.gradients += [gradient]

        policy_value = self.policy.getPolicyForAction(action, state)
        log_gradient = gradient / policy_value if policy_value != 0.0 else 0.0 if gradient == 0.0 else np.nan
        self.log_gradients += [log_gradient]

    def estimateAverageRewardUnderPolicy(self):
        """
        Estimates the average reward received by the agent under the policy *throughout the queue simulation*
        (i.e. the states, actions and rewards are only those seen in the CURRENT queue simulation
        --as opposed to those seen throughout the WHOLE policy learning process--
        because we need to consider only the CURRENT policy (responsible for generating the
        current queue simulation), which is parameterized by the CURRENT theta).

        This is the `rho` baseline value used in the Trunk Reservation paper, pag. 4.
        """
        # Not needed
        #T = len([a for a in self.getActions() if a is not None])
        self.averageRewardUnderPolicy = np.mean( [r # * self.policy.getPolicyForAction(a, s)
                                                        for a, r in zip(self.learnerV.getActions(), self.learnerV.getRewards())
                                                        if a is not None] )

    def computeCorrectedReturn(self, baseline):
        """
        Computes the return G(t) where rewards are corrected by the given baseline, rho.
        Only the time steps where the agent interacts with the environment are taken into consideration.

        Arguments:
        baseline: float
            Baseline used to correct each reward in the computation of G(t) as sum_{s>=t}{ reward(s) - rho }

        Return: numpy array
        Array containing the value of the return G(t) at each time step t in the history of actions taken by the agent
        (i.e. when the action is not None).
        """
        rewards_history = copy.deepcopy(self.learnerV.getRewards())
        actions_history = copy.deepcopy(self.learnerV.getActions())
        rewards_history.reverse()
        actions_history.reverse()
        delta_rewards_history = [r - baseline if a is not None else 0.0 for a, r in zip(actions_history, rewards_history)]
        G = np.cumsum(delta_rewards_history)
        # Reverse the G just computed because we want to have t indexing the values of G from left to right of the array
        return G[::-1]

    def learn(self, T):
        """
        Learns the policy by updating the theta parameter using gradient ascent and estimating the gradient as the
        average of G(t) * grad( log(Pol(A(t)/S(t),theta) ).

        Arguments:
        T: int
            Time corresponding to the end of simulation at which learning takes place.

        Return: tuple
        Tuple with the following elements:
        - theta: theta value before its update
        - theta_next: theta value after its update
        - V: the average reward observed for the theta before its update
        - gradV: gradient of the value function that generated the update on theta
        """
        theta = self.policy.getThetaParameter()
        theta_prev = copy.deepcopy(theta)

        # Compute the baseline-corrected return where the baseline is the estimation of the average reward under the policy
        self.estimateAverageRewardUnderPolicy()

        # Observed return for every t: this should be the return already adjusted by the baseline
        # NOTE: (2021/11/04) This baseline adjustment is essential to the convergence of theta, i.e. to observing
        # negative Delta(theta) when there are a few large negative rewards making theta shrink to a smaller value
        # (as the reward is so negative because the theta is too large and the cost of blocking increases
        # exponentially with theta for large theta values...)
        G = self.computeCorrectedReturn(self.getAverageRewardUnderPolicy())
        deltas = G

        # Trajectory used to compute the gradient of the policy used in the update of theta
        states = self.learnerV.getStates()
        actions = self.learnerV.getActions()
        rewards = self.learnerV.getRewards()
        print("\n--- POLICY LEARNING ---")
        print("theta = {}".format(self.policy.getThetaParameter()))
        print("Average reward (from V learner) = {}".format(self.learnerV.getAverageReward()))
        print("Average reward under policy (baseline for G(t)) rho = {}".format(self.getAverageRewardUnderPolicy()))
        print("TRAJECTORY (t, state, action, reward, reward - rho, delta = G, gradient(log(Pol)))")
        df_trajectory = pd.DataFrame( np.c_[range(T+1), [self.env.getBufferSizeFromState(s) for s in states], actions, rewards, [r - self.getAverageRewardUnderPolicy() for r in rewards], deltas, [self.policy.getGradientLog(a, s) if a is not None else None for a, s in zip(actions, states)]] )
        df_trajectory.columns = ['t', 'state', 'action', 'reward', 'reward - rho', 'G(t)', 'grad']
        print(df_trajectory[ [grad != 0.0 and grad is not None or r != 0.0 for r, grad in zip(df_trajectory['reward'], df_trajectory['grad']) ]])
        #print(df_trajectory)

        # Estimate the gradient of the value function V(theta), called J(theta) in Sutton, pag. 327
        gradV = 0.0
        nactions = 0                # Number of actions taken by the agent during the trajectory (used to estimate gradV)
        for t in range(T+1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            delta = deltas[t]

            if action is not None:
                # Increase the count of actions taken by the agent, which is the denominator used when estimating grad(V)
                # Note: the action may be None if for instance the environment experienced a completed service
                # in which case there is no action to take... (in terms of Accept or Reject)
                nactions += 1

                gradLogPol = self.policy.getGradientLog(action, state)
                if gradLogPol != 0.0:
                    # Update the estimated gradient
                    print("Learning at simulation time t={}, state={}, action={}, reward={}...".format(t, state, action, reward))
                    print("\tt={}: Delta(t) = corrected G(t) = {:.3f}".format(t, delta))
                    print("\tt={}: Log policy gradient for Pol(A(t)={}/S(t)={} ) = {}" \
                          .format(t, action, state, gradLogPol))
                    print("\t#actions taken by the agent so far = {}".format(nactions))
                    gradV += delta * gradLogPol
                    #gradV += (reward - self.getAverageRewardUnderPolicy()) * gradLogPol    # This computation of the gradient is NOT useful for the learning process

            # Update the history of thetas (even if theta has not been updated) so that it is aligned with the
            # history of rewards received and actions taken which make it possible to have an informative plot.
            self.policy.store_theta(theta)

        # Estimated grad(V) as the average of G(t) * grad(log(Pol(t)))
        gradV /= nactions

        # Note that we bound the delta theta to avoid too large changes!
        # We use "strange" numbers to avoid having theta fall always on the same distance from an integer (e.g. 4.1, 5.1, etc.)
        # The lower and upper bounds are asymmetric (larger lower bound) so that a large negative reward can lead to a large reduction of theta.
        bound_delta_theta_upper = +1.131
        bound_delta_theta_lower = -1.131 #-5.312314 #-1.0
        delta_theta = np.max([ bound_delta_theta_lower, np.min([self.alpha * gradV, bound_delta_theta_upper]) ])
        print("Estimated grad(J(theta)) = {}".format(gradV))
        print("Delta(theta) = alpha * grad(V) = {}".format(delta_theta))

        theta_lower = 0.1       # Do NOT use an integer value as lower bound of theta because the gradient is never non-zero at integer-valued thetas
        theta = np.max([theta_lower, theta + delta_theta])

        # Update the policy on the new theta and record it into the history of its updates
        self.policy.setThetaParameter(theta)

        return theta_prev, theta, self.getAverageRewardUnderPolicy(), gradV

    def learn_TR(self, T):
        """
        Learns the policy in the Trunk-Reservation (TR) manner, i.e. by updating the theta parameter
        at every time step of each simulated trajectory but WITHOUT updating the policy until the last time step
        has been reached.

        Arguments:
        T: int
            Time corresponding to the end of simulation at which learning takes place.
        """
        theta = self.policy.getThetaParameter()

        # Compute the baseline-corrected return where the baseline is the estimation of the average reward under the policy
        self.estimateAverageRewardUnderPolicy()

        # Observed return for every t: this should be the return already adjusted by the baseline
        G = self.computeCorrectedReturn(self.getAverageRewardUnderPolicy())
        #G = self.computeCorrectedReturn(0.0)
        deltas = G

        # Trajectory used to compute the gradient of the policy used in the update of theta
        states = self.learnerV.getStates()
        actions = self.learnerV.getActions()
        rewards = self.learnerV.getRewards()
        print("\n--- POLICY LEARNING ---")
        print("theta = {}".format(self.policy.getThetaParameter()))
        print("Average reward (from V learner) = {}".format(self.learnerV.getAverageReward()))
        print("Average reward under policy (baseline for G(t)) = {}".format(self.getAverageRewardUnderPolicy()))
        print("TRAJECTORY (t, state, action, reward, reward - rho, delta = G)")
        print(np.c_[range(T+1), states, actions, rewards, [r - self.getAverageRewardUnderPolicy() for r in rewards], deltas])

        # Go over every time step of the trajectory
        for t in range(T+1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            delta = deltas[t]

            if delta != 0.0 and action is not None:
                ## Note: the action may be None if for instance the environment experienced a completed service
                ## in which case there is no action to take... (in terms of Accept or Reject)

                if self.policy.getGradientLog(action, state) != 0.0:
                    print("Learning at simulation time t={}, state={}, action={}, reward={}...".format(t, state, action, reward))
                    print("\tt={}: Delta(t) = G(t) = {:.3f}".format(t, delta))
                    print("\tt={}: Log-policy gradient for Pol(A(t)={}/S(t)={} ) = {}" \
                          .format(t, action, state, self.policy.getGradientLog(action, state)))

                    # Update alpha based on the action-state visit count
                    alpha = self.update_alpha(state, action)

                    print("\tt={}: alpha(state={}, action={} (n={})) = {}".format(t, state, action, self.getCount(state, action), alpha))

                    # Note that we bound the delta theta to avoid too large changes!
                    #theta += self.alpha * delta * self.policy.getGradientLog(action, state)
                    bound_delta_theta_upper = +0.5
                    bound_delta_theta_lower = -0.5
                    delta_theta = np.max([ bound_delta_theta_lower, np.min([alpha * delta * self.policy.getGradientLog(action, state), bound_delta_theta_upper]) ])
                    print("\tt={}: delta(theta) = {}".format(t, delta_theta))

                    # Only update the visit count of the state and action when delta(theta) != 0.0 because this means that
                    # the state and action have actually been used to learn the policy.
                    if delta_theta != 0.0:
                        self.update_counts(state, action)
                    theta_lower = 0.1       # Do NOT use an integer value as lower bound of theta because the gradient is never non-zero at integer-valued thetas
                    theta = np.max([theta_lower, theta + delta_theta])

            # Update the historic theta values generated during the learning process
            # (even if theta was not changed at this iteration t, as this guarantees that the length
            # of the historic theta values is the same as the length of the historic states, actions and rewards)
            self.policy.store_theta(theta)

        # Update the policy on the new theta only at the END of the simulation time
        # (this is what the Trunk Reservation paper does: "The new policy is generated at the end of the episode")
        self.policy.setThetaParameter(theta)

    #----- GETTERS -----#
    def getPolicy(self):
        return self.policy

    def getLearnerV(self):
        return self.learnerV

    def getGradients(self):
        return self.gradients

    def getGradientsLog(self):
        return self.log_gradients

    def getAverageRewardUnderPolicy(self):
        return self.averageRewardUnderPolicy
