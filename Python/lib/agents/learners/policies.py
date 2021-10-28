# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:13:36 2021

@author: Daniel Mastropietro
@description: Policy Learners
"""
import copy

import numpy as np

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
                 adjust_alpha=False, alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                 alpha_min=0.,
                 debug=False):
        super().__init__(env, alpha, adjust_alpha, alpha_update_type, alpha_min)
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
        (i.e. the states, actions and rewards are only those seen in the last queue simulation, NOT in the whole
        policy learning steps, because we need to consider only the CURRENT policy, which is parameterized by the
        CURRENT theta).
        This is the `rho` baseline value used in the Trunk Reservation paper, pag. 4.
        """
        T = len([a for a in self.getActions() if a is not None])
        self.averageRewardUnderPolicy = 1.0 / T * np.sum( [r * self.policy.getPolicyForAction(a, s)
                                                        for s, a, r in zip(self.learnerV.getStates(), self.learnerV.getActions(), self.learnerV.getRewards())
                                                        if a is not None] )

    def computeCorrectedReturn(self, baseline):
        "Computes the estimated "
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
        Learns the policy by updating the theta parameter based on the state and the action taken on that state

        Arguments:
        T: int
            Time corresponding to the end of simulation at which learning takes place.
        """
        theta = self.policy.getThetaParameter()

        # Compute the baseline-corrected return where the baseline is the estimation of the average reward under the policy
        self.estimateAverageRewardUnderPolicy()

        # Observed return for every t: this should be the return already adjusted by the baseline
        G = self.computeCorrectedReturn(self.getAverageRewardUnderPolicy())
        deltas = G

        # Trajectory used to compute the gradient of the policy used in the update of theta
        states = self.learnerV.getStates()
        actions = self.learnerV.getActions()
        rewards = self.learnerV.getRewards()
        print("\n--- POLICY LEARNING ---")
        print("theta = {}".format(self.policy.getThetaParameter()))
        print("Average reward = {}".format(self.learnerV.getAverageReward()))
        print("Average reward under policy (baseline for G(t)) = {}".format(self.getAverageRewardUnderPolicy()))
        print("TRAJECTORY (t, state, action, reward, reward - rho, delta = G)")
        print(np.c_[range(T+1), states, actions, rewards, [r - self.getAverageRewardUnderPolicy() for r in rewards], deltas])

        # Learning rates for the policy
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
                    print("\tt={}: Log policy gradient for Pol(A(t)={}/S(t)={} ) = {}" \
                          .format(t, action, state, self.policy.getGradientLog(action, state)))

                    # Update alpha based on the action-state visit count
                    alpha = self.update_alpha(state, action)

                    print("\tt={}: alpha(state={}, action={} (n={})) = {}".format(t, state, action, self.getCount(state, action), alpha))

                    # Note that we bound the delta theta to avoid too large changes!
                    #theta += self.alpha * delta * self.policy.getGradientLog(action, state)
                    bound_delta_theta_upper = +0.5
                    bound_delta_theta_lower = -0.5
                    delta_theta = np.max([np.min([bound_delta_theta_upper, alpha * delta * self.policy.getGradientLog(action, state)]), bound_delta_theta_lower])
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
            self.policy.update_thetas(theta)

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
