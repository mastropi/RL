# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:13:36 2021

@author: Daniel Mastropietro
@description: Policy Learners
"""
import copy

import numpy as np
import pandas as pd

from Python.lib.agents.learners import Learner


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

    alpha: (opt) float
        Learning rate. This is the initial learning rate when the value of alpha is adjusted as simulation progresses.
        default: 0.1

    fixed_window: (opt) bool
        Whether to use a fixed window to estimate the return G(t) for the update of parameter theta.
        In that case G(t) is estimated for t=0, 1, ..., int(T/2) (or similar).
        default: False
    """

    def __init__(self, env, policy, learnerV, alpha=0.1,
                 adjust_alpha=False,
                 min_count_to_update_alpha=0,
                 min_time_to_update_alpha=0,
                 alpha_min=0.,
                 fixed_window=False,
                 debug=False):
        super().__init__(env, alpha, adjust_alpha, min_count_to_update_alpha, min_time_to_update_alpha, alpha_min)
        self.debug = debug

        self.fixed_window = fixed_window

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

    def learn(self, T):
        """
        Learns the policy by updating the theta parameter using gradient ascent and estimating the gradient as the
        average of G(t) * grad( log(Pol(A(t)/S(t),theta) ).

        When the fixed_window attribute is True, a fixed window is used to compute G(t) for each t.
        In that case, the size of the window is T/2 and G(t) is estimated for t = 0, 1, ..., int(T/2).
        Otherwise, G(t) is estimated for t = 0, 1, ..., T with a window that shrinks with t.

        Arguments:
        T: int
            Time corresponding to the end of simulation at which learning takes place.

        Return: tuple
        Tuple with the following elements:
        - theta: theta value before its update
        - theta_next: theta value after its update
        - V: the average reward observed in the episodic simulation using the theta value before its update
        - gradV: gradient of the value function that generated the update on theta
        - deltas: array with the delta innovation values for each simulation time step which is obtained by bootstrapping.
            This is normally equal to G(t). For the case when attribute fixed_window = True, this array is equal to NaN
            for int((T+1)/2) < t <= T+1
        """
        theta = self.policy.getThetaParameter()
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
        print("theta = {}".format(self.policy.getThetaParameter()))
        print("Average reward (from V learner) = {}".format(self.learnerV.getAverageReward()))
        print("Average reward under policy (baseline for G(t)) rho = {}".format(self.getAverageRewardUnderPolicy()))
        print("TRAJECTORY (t, state, action, reward, reward - rho, delta = G, gradient(log(Pol)))")
        df_trajectory = pd.DataFrame( np.c_[range(T+1), [self.env.getBufferSizeFromState(s) for s in states], actions,
                                            rewards, [r - self.getAverageRewardUnderPolicy() for r in rewards], deltas,
                                            [self.policy.getGradientLog(a, s) for a, s in zip(actions, states)],
                                            [delta * self.policy.getGradientLog(a, s) for a, s, delta in zip(actions, states, deltas)]] )
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
            self.policy.store_theta(theta)

            state = states[t]
            action = actions[t]
            reward = rewards[t]
            delta = deltas[t]

            if not np.isnan(delta) and action is not None:
                # Increase the count of actions taken by the agent, which is the denominator used when estimating grad(V)
                # Note: the action may be None if for instance the environment experienced a completed service
                # in which case there is no action to take... (in terms of Accept or Reject)
                # Note also that we check that delta is not NaN because NaN indicate that the current simulation time step
                # is NOT part of the calculation of deltas (which happens when attribute fixed_window = True)
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

        # Estimated grad(V) as the average of G(t) * grad(log(Pol(t)))
        gradV /= nactions

        # Note that we bound the delta theta to avoid too large changes!
        # We use "strange" numbers to avoid having theta fall always on the same distance from an integer (e.g. 4.1, 5.1, etc.)
        # The lower and upper bounds are asymmetric (larger lower bound) so that a large negative reward can lead to a large reduction of theta.
        bound_delta_theta_upper = +np.Inf   #+1.131
        bound_delta_theta_lower = -np.Inf   #-1.131 #-5.312314 #-1.0
        delta_theta = np.max([ bound_delta_theta_lower, np.min([self.getLearningRate() * gradV, bound_delta_theta_upper]) ])
        print("Estimated grad(V(theta)) = {}".format(gradV))
        print("Delta(theta) = alpha * grad(V) = {}".format(delta_theta))

        theta_lower = 0.1       # Do NOT use an integer value as lower bound of theta because the gradient is never non-zero at integer-valued thetas
        theta = np.max([theta_lower, theta + delta_theta])

        # Update the policy on the new theta and record it into the history of its updates
        self.policy.setThetaParameter(theta)

        return theta_prev, theta, self.getAverageRewardUnderPolicy(), gradV, deltas

    def learn_TR(self, T):
        """
        Learns the policy in the Trunk-Reservation (TR) manner, i.e. by updating the theta parameter
        at every time step of each simulated trajectory but WITHOUT updating the policy until the last time step
        has been reached.

        When the fixed_window attribute is True, a fixed window is used to compute G(t) for each t.
        In that case, the size of the window is T/2 and G(t) is estimated for t = 0, 1, ..., int(T/2).
        Otherwise, G(t) is estimated for t = 0, 1, ..., T with a window that shrinks with t.

        Arguments:
        T: int
            Time corresponding to the end of simulation at which learning takes place.
            The simulation is assumed to start at time = 0, meaning there are T+1 simulation steps.

        Return: tuple
        Tuple with the following elements:
        - theta: theta value before its final update
        - theta_next: theta value after its final update
        - V: the average reward observed in the episodic simulation using the theta value before its update
        - gradV: the average of the observed gradients of the value function throughout the episode
        - deltas: array with the delta innovation values for each simulation time step which is obtained by bootstrapping.
            This is normally equal to G(t). For the case when attribute fixed_window = True, this array is equal to NaN
            for int((T+1)/2) < t <= T+1
        """
        theta = self.policy.getThetaParameter()
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
        print("theta = {}".format(self.policy.getThetaParameter()))
        print("Average reward (from V learner) = {}".format(self.learnerV.getAverageReward()))
        print("Average reward under policy (baseline for G(t)) = {}".format(self.getAverageRewardUnderPolicy()))
        print("TRAJECTORY (t, state, action, reward, reward - rho, delta = G)")
        df_trajectory = pd.DataFrame( np.c_[range(T+1), [self.env.getBufferSizeFromState(s) for s in states], actions,
                                            rewards, [r - self.getAverageRewardUnderPolicy() for r in rewards],
                                            deltas,
                                            [self.policy.getGradientLog(a, s) for a, s in zip(actions, states)],
                                            [delta * self.policy.getGradientLog(a, s) for a, s, delta in zip(actions, states, deltas)]] )
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
            self.policy.store_theta(theta)

            state = states[t]
            action = actions[t]
            reward = rewards[t]
            delta = deltas[t]

            if delta != 0.0 and not np.isnan(delta) and action is not None:
                # Note: the action may be None if for instance the environment experienced a completed service
                # in which case there is no action to take... (in terms of Accept or Reject)
                # Note also that we check that delta is not NaN because NaN indicate that the current simulation time step
                # is NOT part of the calculation of deltas (which happens when attribute fixed_window = True)

                if self.policy.getGradientLog(action, state) != 0.0:
                    # [DONE-2021/11/28] DM-2021/11/25: STOP AT LEARNING STEP 24 (starting at theta = 11.3) FOR METHOD 2 TO INVESTIGATE WHY A REJECTION OCCURS WHEN STATE = 2 AND THETA = 4...
                    # R: The reason is that the accept/reject policy is updated ONLY AT THE END of the episode!! which means that the value of theta is not always related to the occurrence of rejections.
                    print("Learning at simulation time t={}, theta={}, state={}, action={}, reward={}...".format(t, theta, state, action, reward))
                    print("\tt={}: Delta(t) = G(t) = {:.3f}".format(t, delta))
                    print("\tt={}: Log-policy gradient for Pol(A(t)={}/S(t)={} ) = {:.6f}" \
                          .format(t, action, state, self.policy.getGradientLog(action, state)))

                    # Store the gradient (for analysis purposes)
                    gradV = delta * self.policy.getGradientLog(action, state)
                    gradV_mean += gradV

                    # Update alpha based on the action-state visit count
                    alpha = self.update_alpha(state, action)

                    print("\tt={}: alpha(state={}, action={} (n={})) = {}".format(t, state, action, self.getCount(state, action), alpha))

                    # Note that we bound the delta theta to avoid too large changes!
                    #theta += self.getLearningRate() * delta * self.policy.getGradientLog(action, state)
                    bound_delta_theta_upper = +np.Inf   #+1.131
                    bound_delta_theta_lower = -np.Inf   #-1.131
                    delta_theta = np.max([ bound_delta_theta_lower, np.min([alpha * gradV, bound_delta_theta_upper]) ])
                    print("\tt={}: delta(theta) = {}".format(t, delta_theta))

                    # Only update the visit count of the state and action when delta(theta) != 0.0 because this means that
                    # the state and action have actually been used to learn the policy.
                    if delta_theta != 0.0:
                        self.update_counts(state, action)
                    theta_lower = 0.1       # Do NOT use an integer value as lower bound of theta because the gradient is never non-zero at integer-valued thetas
                    theta = np.max([theta_lower, theta + delta_theta])

                    n_theta_updates += 1

        # Update the policy on the new theta only at the END of the simulation time
        # (this is what the Trunk Reservation paper does: "The new policy is generated at the end of the episode")
        self.policy.setThetaParameter(theta)

        # Compute the average grad(V) observed in the episodic simulation
        if n_theta_updates > 0:
            gradV_mean /= n_theta_updates
        else:
            gradV_mean = None

        return theta_prev, theta, self.getAverageRewardUnderPolicy(), gradV_mean, deltas

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


if __name__ == "__main__":
    from environments.queues import EnvQueueSingleBufferWithJobClasses, rewardOnJobRejection_ExponentialCost
    from agents.learners.continuing.mc import LeaMC
    from agents.learners.policies import LeaPolicyGradient
    from agents.policies.parameterized import PolQueueTwoActionsLinearStep
    from agents.queues import AgeQueue, PolicyTypes, LearnerTypes
    from queues import QueueMM
    from simulators import SimulatorQueue

    #------------------ Test of computeCorrectedReturn() -------------------#
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
    t_sim = 20
    dict_nsteps = dict({'queue': t_sim, 'learn': 1})
    start_state = [ np.ceil(theta) ]
    simul = SimulatorQueue(env_queue_mm, agent_gradient_mc, dict_nsteps, debug=False)
    simul.run(start_state, seed=1717, verbose=False)

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
