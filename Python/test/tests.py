# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:00:15 2021

@author: Daniel Mastropietro
"""

#import runpy
#runpy.run_path('../../setup.py')


########
# 2023/12/03: Compute the stationary probability of each state of an environment using the policy and the environment's transition probabilities for each possible action in each state
# Goal: Compute the true stationary probability of 1D gridworld where we want to test the FV estimation of the stationary probabilities
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from Python.lib.agents.policies import probabilistic
from Python.lib.environments.gridworlds import EnvGridworld1D_OneTerminalState
import Python.lib.agents as agents
from Python.lib.agents.learners import LearningCriterion, LearningTask, ResetMethod
from Python.lib.agents.learners.episodic.discrete import td, fv
from Python.lib.simulators.discrete import Simulator as DiscreteSimulator
import Python.lib.utils.plotting as plotting

# Environment
nS = 5
env1d = EnvGridworld1D_OneTerminalState(length=nS, rewards_dict={nS-1: +1.0}, reward_default=0.0)
print(f"Policy in {nS}-state gridworld:")
for k, v in env1d.P.items():
    print(f"State {k}: {v}")
print(f"Terminal states in {nS}-state gridworld: {env1d.getTerminalStates()}")
print(f"Terminal rewards: {env1d.getTerminalRewards()}")

# Policy
policy_probabilities = [0.1, 0.9]   #[0.5, 0.5]) #[0.9, 0.1])
policy = probabilistic.PolGenericDiscrete(env1d, dict({0: [0.0, 1.0], nS-1: [0.0, 1.0]}), policy_default=policy_probabilities)

# Transition probability matrix under the given policy
# It is based on the environment transition probabilities (given the action at each state) and the defined probabilistic policy
# i.e. P[x,y] = sum_{a}{ p(x,y|a) * policy(a|x) }, where p(x,y|a) is the transition probability of going from x -> y when taking action a at state x.
P = np.matrix(np.zeros((nS, nS)))
for s in range(nS):
    for a in range(env1d.getNumActions()):
        # In the environment, P[s][a] is a list of tuples of the form (prob, next_state, reward, is_terminal) (see the DiscreteEnv environment defined in envs/toy_text/discrete.py)
        # and in each tuple, the transition probability to each possible next state for the GIVEN action 'a' is given at index 0 and the next state is given at index 1.
        for info_transition in env1d.P[s][a]:
            # Next state to be updated in the transition probability matrix
            ns = info_transition[1]
            prob_transition_from_s_to_ns_given_action_a = info_transition[0]
            prob_taking_action_a_at_state_s = policy.policy[s][a]
            P[s,ns] += prob_taking_action_a_at_state_s * prob_transition_from_s_to_ns_given_action_a
P_epi = P
P_con = P.copy()
# For the continuing learning task, update the transition probability of the terminal state, because when reached we want it to start again from state 0
# so that the learning task is continuing (recall that `-1` means "the last element of the array", i.e. the "last state").
P_con[-1,0] = 1.0; P_con[-1,-1] = 0.0
# Check that P is a valid transition probability matrix
for s in range(nS):
    assert np.isclose(np.sum(P[s]), 1.0)

# We can estimate the stationary distribution for the continuing learning task by computing P^t for t sufficiently large
# See below another computation of the stationary distribution, based on the eigendecomposition of P.
# HOWEVER, THIS ONLY WORKS WHEN THERE IS NO SPECIFIC PERIODICITY IN TERMS OF HOW WE CAN GET TO A PARTICULAR STATE!!
# (which doesn't mean that the Markov chain is periodic...)
# For instance: when nS is an EVEN number in this 1D gridworld environment, elevating P to the an EVEN power gives a different result
# than elevating P to an ODD number! (because the probability of being at certain states after an even number of steps is 0;
# example: if nS = 4 and we start at state 0, the probability of being again at state 0 after 1, 3, 5, 7, ... steps is 0 because
# even if we come back to state 0 after reaching the terminal state, it takes always an EVEN number of steps to return to 0!
# However, if nS = 5 and we start at state 0, the probability of being again at state 0 after 1 or 3 steps is 0 but the probability
# of being at state 0 after 5 steps is NOT zero because it suffices the system to go through the terminal state back to the initial state 0.
# And this trajectory automatically makes the probability of being again at state 0 after 7, 9, ... steps no longer 0 as is the case when nS is even!
# *** The SOLUTION is compute mu as 0.5 * (P^t + P^(t+1)) for t sufficiently large, and this works for both nS odd and even. ***
mu = 0.5 * (np.squeeze(np.array((P_con**9999)[0,:])) + np.squeeze(np.array((P_con**10000)[0,:])) )
print(f"Stationary probability distribution for cont. learning task (P^Inf): {mu}")
# When probs = [0.9, 0.1], we get 0.00054319 for the "terminal" state with reward +1, i.e. a reasonably small occupation probability to showcase FV
# Check that this is the statioary distribution
if not policy.isDeterministic():
    assert np.allclose(mu*P_con, mu, atol=1E-4)
## OK

# Note that we can also find the stationary distribution as the eigenvector of P' that corresponds to eigenvalue = 1
# Note that:
# - the eigenvalues are NOT sorted in a particular order (see help(np.linalg.eig))
# - the eigenvectors are given as columns
eigenvalues, eigenvectors = np.linalg.eig(P_con.T)
idx_eigenvalue_one = np.where( np.abs(eigenvalues - 1.0) < 1E-6 )[0][0]
assert np.isclose(eigenvalues[ idx_eigenvalue_one ], 1.0)
eigenvector_one = eigenvectors[:, idx_eigenvalue_one]
prob_stationary = np.squeeze(np.array(np.abs(eigenvector_one) / np.sum(np.abs(eigenvector_one))))
print(f"Stationary probability distribution for cont. learning task (eigen): {prob_stationary}")
if not policy.isDeterministic():
    assert np.allclose(prob_stationary, mu, 1E-4)
## OK!!

# True differential state values for the average reward criterion, under the given policy
# Solving the Bellman equation for the differential value function and using gamma -> 1
# (Note that we cannot use gamma = 1 because (I - P) is singular; in fact, the largest eigenvalue of P is 1.0
# which means that (I - P) has an eigenvalue equal to 0.0 (in fact we can decompose P as Q*D*Q^-1 and therefore
# we can write (I - P) = Q * (I - D) * Q^-1 which means that one of the elements of the diagonal matrix (I - D) is 0
# (i.e. the element associated to the eigenvalue 1.0 in D).
# HOWEVER, if gamma = 1, using np.linalg.pinv() we can compute the pseudo-inverse, i.e. the Moore-Penrose generalized inverse,
# which provides a matrix (I - P)^+ such as (I - P)^+ * b gives V in (I - P)*V = b where V has minimum norm among
# all possible V solutions of the system of equation (I - P)*V = b.
# Recall that `b` is the vector containing the EXPECTED one-step reward received by the agent under the given policy,
# i.e. b[x] = sum_{y,a}{ p(y,a|x) * r(y) } = sum_{y,a}{ p(y|a,x) * policy(a|x) * r(y) } = sum_{y}{ P[x,y] * r(y) }
# In our case, where the reward landscape is r(y) = 1.0 * Ind{y=4}, we have that b[x] = 0 for all x except x = 3,
# at which b[x=3] = P[x=3,2]*0.0 + P[x=3,4]*1.0 = P[x=3,4]*r (= P[-2,-1]*r in Python extended notation that allows us to be independent of the gridworld size)
# Finally, in the continuing learning task case, the Bellman equation is: (I - P)*V = b - g*1,
# where g is the average reward (i.e. the average reward observed over all states under stationarity), and `1` is the vector of ones.
r = env1d.getTerminalReward(nS-1)     # Reward received at the terminal state (assumed only one)
gamma = 0.9
gamma_almost1 = 0.999999
# True value functions under the different learning tasks and learning criteria, namely:
# V_true_epi_disc: learning task = EPISODIC; learning criterion = DISCOUNTED
# V_true_epi_avg: UNDEFINED
#   --> IN FACT, in this case the state value function V(s) is defined as the expected average reward observed in the episode,
#       but this is a complicated quantity because its calculation implies dividing by the length of the episode, T, which is a random variable!
#       We could perhaps talk about the TOTAL expected reward, i.e. V_true_epi_total = E[ sum-of-rewards-observed-in-episode-of-length-T ]
#       but in our case where there is only one non-zero reward at the rightmost state, such learning criterion (TOTAL) is not sensible
#       because for every non-terminal state x, the total observed rewards in ANY episode under ANY policy is ALWAYS 1, i.e. the reward received at termination!
#       (this is confirmed by the calculation below that gives V_true_epi_tot)
# V_true_con_avg: learning task = CONTINUING; learning criterion = AVERAGE
# V_true_con_disc: learning task = CONTINUING; learning criterion = DISCOUNTED
# WE SHOULD COMPARE THE LEARNED STATE VALUE FUNCTIONS WITH THE CORRESPONDING TRUE VALUE FUNCTIONS
# depending on the Learning Task and on the Learning Criterion.
b = np.array([0]*(nS-2) + [P_epi[-2, -1]*r] + [0])
I = np.eye(nS)
V_true_epi_disc = np.linalg.solve(I - gamma*P_epi, b)
# For the state value functions defined by the total or by the average reward, we use gamma_almost1 because using 1 instead gives infinite because I - P is singular
V_true_epi_tot = np.linalg.solve(I - gamma_almost1*P_epi, b)   # This is not a sensible criterion for the episodic learning task for this reward's landscape (see reason in above comment)
V_true_con_avg = np.linalg.solve(I - gamma_almost1*P_con, b - mu[-1]*r) # Note: mu[-1]*r is the average reward (i.e. the average observed reward over all states under stationarity, since avg.reward = sum_{x} mu[x] * r[x])
V_true_con_disc = np.linalg.solve(I - gamma*P_con, b)
# Generalized inverse (minimum norm solution)
V_true_con_avg_minnorm = np.dot( np.linalg.pinv(np.eye(len(P_con)) - P_con), b - mu[-1]*r )
# Check the expected results
if policy_probabilities == [0.5, 0.5] and nS == 5:
#if policy.policy == {0: [0.0, 1.0], 1: [0.5, 0.5], 2: [0.5, 0.5], 3: [0.5, 0.5], 4: [0.0, 1.0]} :
    assert  gamma == 0.9 and \
            np.allclose(V_true_epi_disc, np.array([0.33500299, 0.37222554, 0.49216488, 0.7214742 , 0.0]))
    assert  gamma_almost1 == 0.999999 and \
            np.allclose(V_true_epi_tot, np.array([1.0, 1.0, 1.0, 1.0, 0.0]), atol=1E-5)
    assert  gamma_almost1 == 0.999999 and \
            np.allclose(V_true_con_avg, np.array([-0.13494804, -0.07612458,  0.10034589,  0.39446375, -0.19377143]))
    assert np.allclose(V_true_con_avg_minnorm, np.matrix(([[-0.15294118, -0.09411765,  0.08235294,  0.37647059, -0.21176471]])))
    assert np.linalg.norm(V_true_con_avg_minnorm) <= np.linalg.norm(V_true_con_avg)
    assert  gamma == 0.9 and \
            np.allclose(V_true_con_disc, np.array([0.45975997, 0.51084441, 0.67544983, 0.99015521, 0.41378397]))
    # The difference between consecutive values of V should be the same in both solutions computed above
    assert np.allclose(np.diff(V_true_con_avg), np.diff(V_true_con_avg_minnorm))
## ALL OK!

avg_reward_true = np.sum([mu[s]*r for s, r in env1d.getTerminalRewardsDict().items()])
print(f"True average reward for the cont. learning task: {avg_reward_true}")


#-- Learn the average reward and the state value function
learning_task = LearningTask.CONTINUING
#learning_task = LearningTask.EPISODIC

#learning_criterion = LearningCriterion.AVERAGE; gamma = 1.0
learning_criterion = LearningCriterion.DISCOUNTED

# Store in the environment the true state value function of the learning task we want to analyze so that we can use it to compute the RMSE when estimating the value function
if learning_task == LearningTask.EPISODIC:
    env1d.setV(V_true_epi_disc)
    print(f"True state value function (EPISODIC / DISCOUNTED): {env1d.getV()}")
else:
    assert learning_task == LearningTask.CONTINUING
    if learning_criterion == LearningCriterion.AVERAGE:
        env1d.setV(V_true_con_avg)
        print(f"True differential state value function (CONTINUING / AVERAGE): {env1d.getV()}")
    elif learning_criterion == LearningCriterion.DISCOUNTED:
        env1d.setV(V_true_con_disc)
        print(f"True state value function (CONTINUING / DISCOUNTED): {env1d.getV()}")

# Simulation parameters
N = 50 #50 #500 #200 #20                                 # Number of particles in FV simulation
T = 100 #200 #500                                     # Number of time steps for the estimation of E(T_A)
max_time_steps_fv_per_particle = 50 #10 #50         # Max average number of steps allowed for each particle in the FV simulation
max_time_steps_fv_for_expectation = T
max_time_steps_fv_for_all_particles = N * max_time_steps_fv_per_particle
# Traditional method learning parameters
# They are set for a fair comparison with FV learning
# The maximum time steps to be observed in the benchmark methods is set to the sum of:
# - the max number of time steps allowed to estimate E(T_A)
# - the max number of time steps allowed over all the FV particles
M = 1   # Number of normal particles created during the FV simulation to explore using the underlying Markov process
        # This value MUST coincide with n_normal_max variable defined in Simulator._run_simulation_fv_learnvaluefunctions()
max_time_steps_benchmark = max_time_steps_fv_for_expectation + (1 + M) * max_time_steps_fv_for_all_particles

# Parameters common for all learners
R = 20 #20 #50             # Replications of the estimation process
seed = 1717
nepisodes = 100 #1000
max_time_steps_per_episode = nS*10   #1000 #100
start_state = None  # The start state is defined by the Initial State Distribution (isd) of the environment
plot = False

alpha = 1.0
#gamma = 1.0    # gamma is defined when defining the learning criterion above or when computing the discounted state value for episodic learning tasks (normally `V_true_epi_disc`)
lmbda = 0.0     # We should NOT use lambda > 0 because FV does NOT have the logic implemented to handle this case (currently the trajectories of all particles are mixed into the same trajectory stored in the self._states list.
alpha_min = 0.01 #0.01 #0.1 #0.0
adjust_alpha = True

time_start_all = timer()
if R > 1:
    counts_td = np.nan * np.ones((R, nS), dtype=int)
    counts_fv = np.nan * np.ones((R, nS), dtype=int)
    time_steps_td = np.zeros(R, dtype=int)
    time_steps_fv = np.zeros(R, dtype=int)
    estimates_V_td = np.nan * np.ones((R, nS))
    estimates_V_fv = np.nan * np.ones((R, nS))
    estimates_Q_td = np.nan * np.ones((R, nS*env1d.getNumActions()))
    estimates_Q_fv = np.nan * np.ones((R, nS*env1d.getNumActions()))
    estimates_avg_td = np.nan * np.ones(R)
    estimates_avg_fv = np.nan * np.ones(R)
for rep in range(R):
    print(f"\n******* Running replication {rep+1} of {R} (TD + FV)..." )
    print(f"\nTD(lambda={lmbda})")

    seed_this = seed + rep*13
    #-- Learning the average reward using TD(lambda)
    # Learner and agent definition
    params = dict({'alpha': alpha,
                   'gamma': gamma,
                   'lambda': lmbda,
                   'alpha_min': alpha_min,
                   })
    learner_td = td.LeaTDLambda(env1d,
                                criterion=learning_criterion,
                                task=learning_task,
                                alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                                adjust_alpha=adjust_alpha, adjust_alpha_by_episode=False, func_adjust_alpha=None, #np.sqrt,
                                alpha_min=params['alpha_min'],
                                #reset_method=ResetMethod.RANDOM_UNIFORM, reset_seed=1713,
                                debug=False)
    agent_td = agents.GenericAgent(policy, learner_td)

    # Simulation
    time_start = timer()
    sim = DiscreteSimulator(env1d, agent_td, debug=False)
    state_values_td, action_values_td, state_counts_td, RMSE_by_episode, MAPE_by_episode, learning_info = \
        sim.run(nepisodes=nepisodes,
                max_time_steps=max_time_steps_benchmark,
                max_time_steps_per_episode=+np.Inf, #max_time_steps_per_episode, #+np.Inf
                start_state_first_episode=start_state, seed=seed_this,
                compute_rmse=True, state_observe=nS-2,  # This is the state just before the terminal state
                verbose=True, verbose_period=10,
                plot=plot, pause=0.1)
    time_td = timer() - time_start
    avg_reward_td = learner_td.getAverageReward()
    if R > 1:
        counts_td[rep] = state_counts_td
        time_steps_td[rep] = learning_info['t']
        estimates_V_td[rep] = state_values_td
        estimates_Q_td[rep] = action_values_td
        estimates_avg_td[rep] = avg_reward_td
        print(f"\nNumber of time steps used by TD(lambda): {learning_info['t']}")
        print(f"Average reward estimated by TD(lambda): {avg_reward_td}")
        print(f"True average reward: {avg_reward_true}")
        print("Relative error: {:.1f}%".format((avg_reward_td / avg_reward_true - 1)*100))
        print("TD(lambda) took {:.1f} minutes".format(time_td / 60))


    #-- Learning the average reward using FV(lambda)
    print(f"\nFV(lambda={lmbda})")

    seed_this = seed_this + 1
    # Learner and agent definition
    params = dict({'N': N,  # NOTE: We should use N = 500 if we use N = 200, T = 200 above to compute the benchmark time steps because if here we use the N defined above, the number of time steps used by FV is much smaller than the number of time steps used by TD (e.g. 1500 vs. 5000) --> WHY??
                   'T': T,                           # Max simulation time over ALL episodes run when estimating E(T_A). This should be sufficiently large to obtain an initial non-zero average reward estimate (if possible)
                   'absorption_set': set(np.arange(2)), #set({1}),  # Note: The absorption set must include ALL states in A (see reasons in the comments in the LeaFV constructor)
                   'activation_set': set({2}),
                   'alpha': alpha,
                   'gamma': gamma,
                   'lambda': lmbda,
                   'alpha_min': alpha_min,
                   })
    learner_fv = fv.LeaFV(env1d, params['N'], params['T'], params['absorption_set'], params['activation_set'],
                          probas_stationary_start_state_et=None,
                          probas_stationary_start_state_fv=None,
                          criterion=learning_criterion,
                          alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                          adjust_alpha=adjust_alpha, adjust_alpha_by_episode=False, func_adjust_alpha=None, #np.sqrt,
                          #reset_method=ResetMethod.RANDOM_UNIFORM, reset_params={'min': -1, 'max': +1}, reset_seed=1713,
                          alpha_min=params['alpha_min'],
                          debug=False)
    agent_fv = agents.GenericAgent(policy, learner_fv)

    # Simulation
    time_start = timer()
    sim = DiscreteSimulator(env1d, agent_fv, debug=False)
    state_values_fv, action_values_fv, state_counts_fv, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, n_events_et, n_events_fv = \
        sim.run(nepisodes=nepisodes,
                max_time_steps_per_episode=+np.Inf, #max_time_steps_per_episode,  #+np.Inf
                max_time_steps_fv=max_time_steps_fv_for_all_particles,
                min_num_cycles_for_expectations=0,
                seed=seed_this,
                verbose=True, verbose_period=1,
                plot=plot)
    time_fv = timer() - time_start
    avg_reward_fv = learner_fv.getAverageReward()
    if R == 1:
        # Show the results of TD lambda so that we can compare it easily with FV shown below
        print(f"\nNumber of time steps used by TD(lambda): {learning_info['t']}")
        print(f"Average reward estimated by TD(lambda): {avg_reward_td}")
        print(f"True average reward: {avg_reward_true}")
        print("Relative error: {:.1f}%".format((avg_reward_td / avg_reward_true - 1)*100))
        print("TD(lambda) took {:.1f} minutes".format(time_td / 60))
    else:
        counts_fv[rep] = state_counts_fv
        time_steps_fv[rep] = n_events_et + n_events_fv
        estimates_V_fv[rep] = state_values_fv
        estimates_Q_fv[rep] = action_values_fv
        estimates_avg_fv[rep] = avg_reward_fv
    print(f"\nNumber of time steps used by FV(lambda): {n_events_et + n_events_fv} (= {n_events_et} for E(T) + {n_events_fv} for FV)")
    print(f"Average reward estimated by FV(lambda): {avg_reward_fv}")
    print(f"True average reward: {avg_reward_true}")
    print("Relative error: {:.1f}%".format((avg_reward_fv / avg_reward_true - 1)*100))
    print("FV(lambda) took {:.1f} minutes".format(time_fv / 60))
print("\n\nAll replications took {:.1f} minutes\n\n".format((timer() - time_start_all) / 60))

#-- Plot results
# Replications
if R > 1:
    # Plot the estimates of the value function
    states = np.arange(1, nS + 1)

    #np.diff(estimates_V_td, axis=1)
    #np.diff(estimates_V_fv, axis=1)
    min_V_td, max_V_td, median_V_td, mean_V_td, std_V_td, avg_events_td = estimates_V_td.min(axis=0), estimates_V_td.max(axis=0), np.median(estimates_V_td, axis=0), estimates_V_td.mean(axis=0), estimates_V_td.std(axis=0), time_steps_td.mean(axis=0)
    min_V_fv, max_V_fv, median_V_fv, mean_V_fv, std_V_fv, avg_events_fv = estimates_V_fv.min(axis=0), estimates_V_fv.max(axis=0), np.median(estimates_V_fv, axis=0), estimates_V_fv.mean(axis=0), estimates_V_fv.std(axis=0), time_steps_fv.mean(axis=0)

    # Compute percentiles
    # Function that returns a function to compute percentiles using the agg() aggregation method in pandas `groupby` data frames
    # Ref: https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function
    def percentile(n):
        def percentile_(x):
            return x.quantile(n / 100)
        percentile_.__name__ = 'percentile_{:.0f}'.format(n)
        return percentile_
    percentiles_low = [10, 25]
    percentiles_upp = [90, 75]
    alphas = [0.10, 0.15, 0.20]
    percentiles_td = pd.DataFrame({'replication': np.array([np.repeat(r, nS) for r in range(1, R+1)]).reshape(-1),
                                    'state': np.array([np.repeat(s, R) for s in states]).T.reshape(-1),
                                    'V': estimates_V_td.reshape(-1)}, columns=['replication', 'state', 'V'])[['state', 'V']] \
                            .groupby('state') \
                            .agg(['count', 'mean', 'min', 'median', 'max', 'std'] + [percentile(p) for p in percentiles_low] + [percentile(p) for p in percentiles_upp])
    percentiles_fv = pd.DataFrame({'replication': np.array([np.repeat(r, nS) for r in range(1, R+1)]).reshape(-1),
                                    'state': np.array([np.repeat(s, R) for s in states]).T.reshape(-1),
                                    'V': estimates_V_fv.reshape(-1)}, columns=['replication', 'state', 'V'])[['state', 'V']] \
                            .groupby('state') \
                            .agg(['count', 'mean', 'min', 'median', 'max', 'std'] + [percentile(p) for p in percentiles_low] + [percentile(p) for p in percentiles_upp])

    plot_what = ["td", "fv"] #["fv"] #["td", "fv"]
    legend = [] #if plot_what != ["td", "fv"] else [f"True V(s)", f"V(s): TD(avg. #events={avg_events_td})", f"V(s): FV(avg. #events={avg_events_fv})"] + \
                                                  #["TD min-max"] + [f"TD {p}%-{q}%" for p, q in zip(percentiles_low, percentiles_upp)] + \
                                                  #["FV min-max"] + [f"FV {p}%-{q}%" for p, q in zip(percentiles_low, percentiles_upp)]
    ax, ax_diff = plt.figure().subplots(1, 2)
    ax.plot(states, env1d.getV(), color="blue")
    legend += [f"True V(s)"]
    ax.plot(states, median_V_td, color="red", linewidth=2) if "td" in plot_what else None
    ax.plot(states, median_V_fv, color="green", linewidth=2) if "fv" in plot_what else None
    legend += ["TD Median (avg. #events={:.0f})".format(int(avg_events_td))] if "td" in plot_what else []
    legend += ["FV Median (avg. #events={:.0f})".format(int(avg_events_fv))] if "fv" in plot_what else []
    ax.plot(states, mean_V_td, color="red", linewidth=2, linestyle="dashed") if "td" in plot_what else None
    ax.plot(states, mean_V_fv, color="green", linewidth=2, linestyle="dashed") if "fv" in plot_what else None
    legend += ["TD Mean (avg. #events={:.0f})".format(int(avg_events_td))] if "td" in plot_what else []
    legend += ["FV Mean (avg. #events={:.0f})".format(int(avg_events_fv))] if "fv" in plot_what else []
    ax.fill_between(states,
                    max_V_td,
                    min_V_td,
                    color="red",
                    alpha=0.1) if "td" in plot_what else None
    legend += ["TD min-max"] if "td" in plot_what else []
    for alpha, p, q in zip(alphas, percentiles_low, percentiles_upp):
        ax.fill_between(states,
                        percentiles_td['V']['percentile_' + str(q)],
                        percentiles_td['V']['percentile_' + str(p)],
                        color="red",
                        alpha=alpha) if "td" in plot_what else None
        legend += [f"TD {p}%-{q}%"] if "td" in plot_what else []
    ax.fill_between(states,
                    max_V_fv,
                    min_V_fv,
                    color="green",
                    alpha=0.1) if "fv" in plot_what else None
    legend += ["FV min-max"] if "fv" in plot_what else []
    for alpha, p, q in zip(alphas, percentiles_low, percentiles_upp):
        ax.fill_between(states,
                        percentiles_fv['V']['percentile_' + str(q)],
                        percentiles_fv['V']['percentile_' + str(p)],
                        color="green",
                        alpha=alpha) if "fv" in plot_what else None
        legend += [f"FV {p}%-{q}%"] if "fv" in plot_what else []
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("States")
    ax.set_ylabel("V(s)")
    ax.legend(legend, loc="upper left")
    ax.set_title("V(s)")
    # NOTE: The horizontal reference line must come AFTER the legend is added because o.w. the line gets infiltrated in the middle and the legend gets mixed up!! (WHY??)
    ax.axhline(0, color="gray", linestyle="dashed")
    # Plot each estimate separately in order to check the behaviour of each estimate (which is not reflected in the percentiles by state!)
    for rep in range(R):
        ax.plot(states, estimates_V_td[rep], color="red", linewidth=0.1) if "td" in plot_what else None
        ax.plot(states, estimates_V_fv[rep], color="green", linewidth=0.1) if "fv" in plot_what else None
    # Plot of the difference of consecutive V(s)
    ax_diff.plot(states[:-1], np.diff(env1d.getV()), color="blue")
    ax_diff.plot(states[:-1], np.diff(median_V_td), color="red")
    ax_diff.plot(states[:-1], np.diff(median_V_fv), color="green")
    ax_diff.axhline(0, color="gray")
    ax_diff.set_xlabel("Left states")
    ax_diff.set_ylabel("DIFF V(s)")
    ax_diff.legend([f"True diff V(s)", f"MEDIAN Diff V(s): TD(lambda={lmbda})", f"MEDIAN Diff V(s): FV(lambda={lmbda})"])
    ax_diff.set_title("V(s) difference among contiguous states")
    plt.suptitle(f"1D gridworld ({nS} states) with one terminal state (r=+1), policy probabilities = {policy_probabilities}, # particles = {N}, # replications = {R}, lambda={lmbda}, alpha_min={alpha_min}, adjust_alpha={adjust_alpha}"
                 f"\n{learning_task.name} learning task, {learning_criterion.name} reward criterion\nDistribution of state values V(s) by state")

    # Plot the average reward estimate for each replication
    plt.figure()
    ax = plt.gca()
    ax.plot(np.arange(1, R+1), estimates_avg_td, color="red")
    ax.plot(np.arange(1, R+1), estimates_avg_fv, color="green")
    ax.axhline(avg_reward_true, color="gray", linestyle="dashed")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax.twinx()
    ax2.bar(np.arange(1, R+1), height=time_steps_td, color="red", alpha=0.3)
    ax2.bar(np.arange(1, R+1), height=time_steps_fv, color="green", alpha=0.3)

    # Violin plots of the average reward estimate by method
    ax = plt.figure().gca()
    plotting.violinplot(ax, estimates_avg_td, positions=[1], color_body="red", color_lines="red", color_means="black")
    plotting.violinplot(ax, estimates_avg_fv, positions=[2], color_body="green", color_lines="green", color_means="black")
    ax.plot(1 + np.random.normal(0, 0.01, R), estimates_avg_td, '.', marker=".", markersize=3, color="black")
    ax.plot(2 + np.random.normal(0, 0.01, R), estimates_avg_fv, '.', marker=".", markersize=3, color="black")
    ax.axhline(avg_reward_true, color="gray", linestyle="dashed")
    ax.xaxis.set_ticks([1, 2])
    ax.xaxis.set_ticklabels([f"TD(lambda={lmbda})", f"FV(lambda={lmbda})"])
    ax.set_ylabel("Average reward")
    plt.title(f"Distribution of average reward estimation on {R} replications: TD (red) vs. FV (green)")

# Estimated value functions for a given replication
# Note that, under the average reward learning criterion, we plot the value function as the difference of the V(s) (either true or estimated)
# w.r.t. V(0) (either true or estimated) because under the average reward criterion, the value function V(s) is NOT unique.
# Ref: Gast paper at https://www.sigmetrics.org/mama/abstracts/Gast.pdf
# or Ger Koole's lecture notes at https://www.scribd.com/document/177118281/Lecture-Notes-Stochastic-Optimization-Koole
# and available at Urtzi's ownCloud as of Dec-2023: https://cloud.irit.fr/s/xRmIQolbiWagiSf
rep = R - 1
ref_value_true = ref_value_td = ref_value_fv = 0.0
if learning_criterion == LearningCriterion.AVERAGE:
    ref_value_true = env1d.getV()[0]
    ref_value_td = estimates_V_td[rep][0]
    ref_value_fv = estimates_V_fv[rep][0]
ax_V, ax_Vd = plt.figure().subplots(1,2)    # ax_Vd: for plotting the difference among state values (which is what really interest us for learning an optimal policy!)
states = np.arange(1, nS+1)
ax_V.plot(states, env1d.getV() - ref_value_true, color="blue")
ax_V.plot(states, estimates_V_td[rep] - ref_value_td, color="red")
ax_V.plot(states, estimates_V_fv[rep] - ref_value_fv, color="green")
ax_V.plot(states, estimates_Q_td[rep].reshape(nS, env1d.getNumActions())[:,0], color="orange", linestyle="dashed")
ax_V.plot(states, estimates_Q_td[rep].reshape(nS, env1d.getNumActions())[:,1], color="red", linestyle="dashed")
ax_V.plot(states, estimates_Q_fv[rep].reshape(nS, env1d.getNumActions())[:,0], color="lightgreen", linestyle="dashed")
ax_V.plot(states, estimates_Q_fv[rep].reshape(nS, env1d.getNumActions())[:,1], color="green", linestyle="dashed")
ax_V.axhline(0, color="gray", linestyle="dashed")
ax_V.set_xlabel("States")
ax_V.set_ylabel("V(s)")
ax_V.set_title("V(s)")
ax_V.legend([f"True V(s)", f"V(s): TD(lambda={lmbda})", f"V(s): FV(lambda={lmbda})", "Q(s,LEFT): TD", "Q(s,RIGHT): TD", "Q(s,LEFT): FV", "Q(s,RIGHT): FV"], loc="upper left")
ax2_V = ax_V.twinx()
ax2_V.plot(states, mu, color="violet", marker=".", markersize=20)
ax2_V.bar(states, counts_td[rep] / sum(counts_td[rep]), color="red", alpha=0.3)
ax2_V.bar(states, counts_fv[rep] / sum(counts_fv[rep]), color="green", alpha=0.3)
ax2_V.set_ylabel("Relative state frequency")
ax2_V.legend(["stationary probability", f"Rel. Freq.(s): TD(#events={learning_info['t']})", f"Rel. Freq.(s): FV(#events={n_events_et + n_events_fv})"], loc="upper right")
ax_Vd.plot(states[:-1], np.diff(env1d.getV() - ref_value_true), color="blue")
ax_Vd.plot(states[:-1], np.diff(estimates_V_td[rep] - ref_value_td), color="red")
ax_Vd.plot(states[:-1], np.diff(estimates_V_fv[rep] - ref_value_fv), color="green")
ax_Vd.axhline(0, color="gray")
ax_Vd.set_xlabel("Left states")
ax_Vd.set_ylabel("DIFF V(s)")
ax_Vd.legend([f"True diff V(s)", f"Diff V(s): TD(lambda={lmbda})", f"Diff V(s): FV(lambda={lmbda})"])
ax_Vd.set_title("V(s) difference among contiguous states")
plt.suptitle(f"REPLICATION {rep+1} of {R}: 1D gridworld ({nS} states) with one terminal state (r=+1). Policy probabilities = {policy_probabilities}. # particles = {N}"
             f"\n{learning_task.name} learning task, {learning_criterion.name} reward criterion\nState values V(s) and visit frequency (count) for TD and FV")

# To show the different replications one after the other
# Goal: Check the Q values, if V is comprised in between them
plt.figure()
plt.pause(5)
for r in range(R):
    if r > 0:
        plt.plot(states, estimates_V_fv[r-1], color="white")
        plt.plot(states, estimates_Q_fv[r-1].reshape(nS, env1d.getNumActions())[:, 0], color="white", linestyle="dashed")
        plt.plot(states, estimates_Q_fv[r-1].reshape(nS, env1d.getNumActions())[:, 1], color="white", linestyle="dashed")
    plt.plot(states, estimates_V_fv[r], color="green")
    plt.plot(states, estimates_Q_fv[r].reshape(nS, env1d.getNumActions())[:, 0], color="red", linestyle="dashed")
    plt.plot(states, estimates_Q_fv[r].reshape(nS, env1d.getNumActions())[:, 1], color="orange", linestyle="dashed")
    plt.title(f"Replication {r+1} of {R}")
    plt.draw()
    plt.pause(5)
## OK: In most of the cases, the estimates are correct (in ~ 17 out of 20 replications)

# Check the proportion of actions taken at each state
states_history = np.concatenate(learner_td.states)
actions_history = np.concatenate(learner_td.actions)
for s in range(nS):
    ind_s = states_history == s
    states_history[ind_s]
    print(f"Distribution of actions for s={s} (TD): visit count = {np.sum(ind_s)}")
    print(pd.Series(actions_history[ind_s]).value_counts() / np.sum(ind_s))
## OK! 50% for all states except state 0 and 4

raise KeyboardInterrupt


########
# 2023/10/12: Learn an actor-critic policy using neural networks (with the torch package)
# Learning happens with the ActorCriticNN learner which defines a loss of type `tensor` which can be minimized using the backward() method of torch Tensors
# IT WORKS!
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt, cm
from matplotlib.ticker import MaxNLocator
from scipy.special import rel_entr

from Python.lib.agents.learners import ResetMethod
from Python.test.test_optimizers_discretetime import InputLayer, Test_EstPolicy_EnvGridworldsWithObstacles
from Python.lib.agents.learners import LearningCriterion, LearningTask
from Python.lib.agents.learners.policies import LeaActorCriticNN

#--- Auxiliary functions
KL_THRESHOLD = 0.005
policy_changed_from_previous_learning_step = lambda KL_distance, num_states: np.abs(KL_distance) / num_states > KL_THRESHOLD
#--- Auxiliary functions

# Learning task and learning criterion are used by the constructor of the test class below
learning_task = LearningTask.CONTINUING
learning_criterion = LearningCriterion.DISCOUNTED; gamma = 0.9
#learning_criterion = LearningCriterion.AVERAGE; gamma = 1.0

seed = 1317
test_ac = Test_EstPolicy_EnvGridworldsWithObstacles()
size_vertical = 3; size_horizontal = 4
#size_vertical = 6; size_horizontal = 8
#size_vertical = 8; size_horizontal = 12
#size_vertical = 9; size_horizontal = 13
#size_vertical = 10; size_horizontal = 14
#size_vertical = 10; size_horizontal = 30
env_shape = (size_vertical, size_horizontal) #(3, 4)

# (2023/12/25) It's IMPORTANT that the start state be part of the absorption set if we want the value functions estimated by FV
# converge to a constant, as opposed to diverging negatively. Need to investigate why this happens.
start_state_in_absorption_set = True   #False #True

if False and env_shape == (3, 4):
    obstacles_set = {} #{2, 5, 11} #{} #{5} #{2,5,11}
    absorption_set = {0, 1} #{0, 4, 8} #{0, 1, 8}
    nn_hidden_layer_sizes = [8]
else:
    # The path to the terminal state is just a corridor through the last row right and then the last column up
    # So at the upper left corner there is a rectangle that brings to nowhere
    rectangle_to_nowhere_width = size_horizontal - 2
    rectangle_to_nowhere_height = size_vertical - 2
    states_previous_to_last_row = np.ravel_multi_index([np.repeat(rectangle_to_nowhere_height, rectangle_to_nowhere_width), [y for y in range(1, rectangle_to_nowhere_width+1)]], env_shape)
    states_previous_to_last_column = np.ravel_multi_index([[x for x in range(0, rectangle_to_nowhere_height+1)], np.repeat(rectangle_to_nowhere_width, rectangle_to_nowhere_height+1)], env_shape)
    obstacles_set = set(np.concatenate([list(states_previous_to_last_row) + list(states_previous_to_last_column)]))

    # The absorption set is a rectangular area at the upper left corner of the grid + (possibly) the lower left corner
    #lower_left_state = (size_vertical-1) * size_horizontal
    #absorption_set = set(np.concatenate([list(range(x*size_horizontal, x*size_horizontal + size_horizontal-2)) for x in range(size_vertical-2)]))
    # The absorption set is a rectangular area that touches the right and bottom walls of the big rectangular area that leads to nowhere
    left_margin = 0 #0 #int(rectangle_to_nowhere_width/2)
    top_margin = rectangle_to_nowhere_height
    absorption_set = set(np.concatenate([list(range(x * size_horizontal + left_margin, x * size_horizontal + rectangle_to_nowhere_width)) for x in range(0, top_margin)]))

    # Add the environment's start state to the absorption set
    # (ASSUMED TO BE AT THE LOWER LEFT OF THE LABYRINTH)
    if start_state_in_absorption_set:
        _start_state = np.ravel_multi_index((size_vertical-1, 0), env_shape)
        absorption_set.add(_start_state)

    # Number of hidden layers in the neural network model
    # Using multiple layers whose size is proportional to the gridworld size... however this tends to be counterproductive...
    # i.e. learning is slower and may fail (e.g. it usually converges to a non-optimal policy where the advantage function is 0), presumably because of the larger number of parameters.
    # Perhaps the architecture would work if the learning step of the neural network parameters optimizer is smaller and
    # a larger number of policy learning steps is used (I tried with optimizer_learning_rate = 0.01 instead of 0.1 and
    # the parameters started to be learned (i.e. the average reward went up --as opposed to 0-- although with large oscillations)
    # but still 30 learning steps did not suffice to learn completely.
    # This was tried with the 6x8 gridworld with a large rectangle of states going nowhere at the upper-left part
    # with adaptive TD(lambda), where the hidden layer sizes were set to [38, 19].
    #nn_hidden_layer_sizes = [int( 0.8*np.prod(env_shape) ), int( 0.4*np.prod(env_shape) )]
    # Keep the neural network rather small
    nn_hidden_layer_sizes = [12]
print(f"Neural Network architecture:\n{len(nn_hidden_layer_sizes)} hidden layers of sizes {nn_hidden_layer_sizes}")

test_ac.setUpClass(shape=env_shape, nn_input=InputLayer.ONEHOT, nn_hidden_layer_sizes=nn_hidden_layer_sizes,
                    # General learning parameters
                    learning_task=learning_task,
                    learning_criterion=learning_criterion,
                    alpha=1.0, gamma=gamma, lmbda=0.0,
                    alpha_min=0.1,
                    # Fleming-Viot parameters
                    # Small N and T are N=50, T=1000 for the 8x12 labyrinth with corridor
                    N=50,   #50 #20 #100
                    T=100, #1000, #3000,  # np.prod(env_shape) * 10  #100 #1000
                    obstacles_set=obstacles_set, absorption_set=absorption_set,
                    reset_method_value_functions=ResetMethod.ALLZEROS,
                    seed=seed, debug=False)
test_ac.setUp()
print(test_ac.policy_nn.nn_model)
test_ac.env2d._render()
print(f"True state value function (dummy values for now):\n{test_ac.env2d.getV()}")
alpha_initial = test_ac.agent_nn_td.getLearner().getInitialLearningRate()

# Maximum average reward (to use as reference in information and plots)
# It is computed as the inverse of the shortest path from Start to Terminal in Manhattan-like movements,
# INCLUDING THE START STATE for continuing learning tasks (just think about it, we restart every time we reach the terminal state with reward = 0)
# EXCLUDING THE START STATE for episodic learning tasks (just think about it)
# *** This value is problem dependent and should be adjusted accordingly ***
max_avg_reward_continuing = 1 / (np.sum(test_ac.env2d.shape) - 1)   # In this case the start state counts! (we subtract 1 because the bottom-right state must NOT count twice!)
max_avg_reward_episodic = 1 / (np.sum(test_ac.env2d.shape) - 2) # In this case the start state does not count (we subtract 2 because, besides not counting twice the bottom-right state, the start state does not count in the episodic setting)


# Learning method (of the value functions and the policy)
# Both value functions and policy are learned online using the same simulation
learning_method = "all_online"
# Value functions are learned separately from the policy
# Policy learning can happen online or OFFLINE
learning_method = "values_td"; simulator_value_functions = test_ac.sim_td       # Normally TD(0), but could be TD(lambda)
#learning_method = "values_tdl"; simulator_value_functions = test_ac.sim_td     # TD(lambda)
#learning_method = "values_tda"; simulator_value_functions = test_ac.sim_tda    # Adaptive TD(lambda)
#learning_method = "values_fv"; simulator_value_functions = test_ac.sim_fv

# FV learning parameters (which are used to define parameters of the other learners analyzed so that their comparison with FV is fair)
max_time_steps_fv_per_particle = 50                            # Max average number of steps allowed for each particle in the FV simulation
max_time_steps_fv_for_expectation = test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()                         # This is parameter T
max_time_steps_fv_for_all_particles = test_ac.agent_nn_fv.getLearner().getNumParticles() * max_time_steps_fv_per_particle    # This is parameter N * max_time_steps_fv_per_particle which defines the maximum number of steps to run the FV system when learning the value functions that are used as critic of the loss function at each policy learning step

# Traditional method learning parameters
# They are set for a fair comparison with FV learning
# The maximum time steps to be observed in the benchmark methods is set to the sum of:
# - the max number of time steps allowed to estimate E(T_A)
# - the max number of time steps allowed over all the FV particles
# Values previously used: #2000 #test_ac.env2d.getNumStates()*10
max_time_steps_benchmark = max_time_steps_fv_for_expectation + max_time_steps_fv_for_all_particles
print("******")
print(f"A MAXIMUM of {max_time_steps_benchmark} steps will be allowed during the simulation of either FV or any benchmark method used.")
print("******")

# Common learning parameters
n_learning_steps = 100 #200 #50 #100 #30
n_episodes_per_learning_step = 50   #100 #30   # This parameter is used as the number of episodes to run both for the value functions learning AND the policy learning, and both for the traditional and the FV methods method (for FV, this number of episodes is used to learn the expected reabsorption time E(T_A))
max_time_steps_per_episode = test_ac.env2d.getNumStates()*10    # This parameter is just set as a SAFEGUARD against being blocked in an episode at some state of which the agent could be liberated by restarting to a new episode (when this max number of steps is reached)
max_time_steps_per_policy_learning_episode = test_ac.env2d.getNumStates() #np.prod(env_shape) * 10 #max_time_steps_benchmark // n_episodes_per_learning_step   # Maximum number of steps per episode while LEARNING THE *POLICY* ONLINE (NOT used for the value functions (critic) learning)
adjust_alpha_initial_by_learning_step = False; t_learn_min_to_adjust_alpha = 30
policy_learning_mode = "online"     # Whether the policy is learned online or OFFLINE (only used when value functions are learned separately from the policy)
optimizer_learning_rate = 0.01 #0.01 #0.1
reset_value_functions_at_every_learning_step = False #(learning_method == "values_fv")     # Reset the value functions when learning with FV, o.w. the learning can become too unstable due to the oversampling of the states with high value... (or something like that)
verbose_period = n_episodes_per_learning_step // 10

time_start = timer()

# Initialize objects that will contain the results by learning step
break_when_no_change = False    # Whether to stop the learning process when the average reward doesn't change from one step to the next
break_when_goal_reached = False  # Whether to stop the learning process when the average reward is close enough to the maximum average reward (by a relative tolerance of 0.1%)
state_counts_all = np.zeros((n_learning_steps, test_ac.env2d.getNumStates()))
V_all = np.zeros((n_learning_steps, test_ac.env2d.getNumStates()))
Q_all = np.zeros((n_learning_steps, test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions()))
R_all = np.nan * np.ones((n_learning_steps,))       # Average reward (EPISODIC learning task)
R_long_all = np.nan * np.ones((n_learning_steps,))  # Long-run Average reward (CONTINUING learning task). It does NOT converge to the same value as the episodic average reward because there is one more reward value per episode!! (namely the reward going from the terminal state to the start state)
loss_all = np.nan * np.ones((n_learning_steps,))
nsteps_all = np.nan * np.ones((n_learning_steps,))  # Number of value function time steps run per every policy learning step
KL_all = np.nan * np.ones((n_learning_steps,))      # K-L divergence between two consecutive policies
alpha_all = alpha_initial * np.ones((n_learning_steps,))   # Initial alpha used at each policy learning step
if learning_method == "all_online":
    # Online Actor-Critic policy learner with TD(lambda) as value functions learner and value functions learning happens at the same time as policy learning
    learner_ac = LeaActorCriticNN(test_ac.env2d, test_ac.agent_nn_td.getPolicy(), test_ac.agent_nn_td.getLearner(), reset_value_functions=reset_value_functions_at_every_learning_step, optimizer_learning_rate=optimizer_learning_rate, seed=test_ac.seed, debug=True)
    for t_learn in range(1, n_learning_steps+1):
        print(f"\n\n*** Running learning step {t_learn} of {n_learning_steps} (AVERAGE REWARD at previous step = {R_all[max(0, t_learn-1)]}) of MAX={max_avg_reward_episodic} using {nsteps_all[max(0, t_learn-1)]} time steps for Critic estimationn)...")
        print("Learning the VALUE FUNCTIONS and POLICY simultaneously...")
        loss_all[t_learn-1] = learner_ac.learn(n_episodes_per_learning_step, max_time_steps_per_episode=max_time_steps_per_policy_learning_episode, prob_include_in_train=1.0) # prob_include_in_train=0.5)

        state_counts_all[t_learn-1, :] = learner_ac.learner_value_functions.getStateCounts()
        V_all[t_learn-1, :] = learner_ac.learner_value_functions.getV().getValues()
        Q_all[t_learn-1, :, :] = learner_ac.learner_value_functions.getQ().getValues().reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())
        R_all[t_learn-1] = learner_ac.learner_value_functions.getAverageReward()
        # Could also retrieve the average reward from the Actor-Critic learner (if store_trajectory_history=False in the constructor of the value functions learner)
        #R_all[t_learn-1] = learner_ac.average_reward_over_episodes
else:
    # Reset the initial alpha of the TD learner in case it has been updated by a previous run below during the policy learning steps
    simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial)
    simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial)

    # Value functions (Critic) are learned separately from the application of the policy and the policy (Actor) may be learned OFFLINE or online
    learner_ac = LeaActorCriticNN(test_ac.env2d, simulator_value_functions.getAgent().getPolicy(), simulator_value_functions.getAgent().getLearner(), reset_value_functions=reset_value_functions_at_every_learning_step, optimizer_learning_rate=optimizer_learning_rate, seed=test_ac.seed, debug=True)

    # Keep track of the policy learned so that we can analyze how much it changes after each learning step w.r.t. the previous learning step
    policy_prev = None
    for t_learn in range(1, n_learning_steps+1):
        print(f"\n\n*** Running learning step {t_learn} of {n_learning_steps} (AVERAGE REWARD at previous step = {R_all[max(0, t_learn-2)]} of MAX={max_avg_reward_episodic})...")
        reset_value_functions_at_this_step = reset_value_functions_at_every_learning_step if t_learn > 1 else True  # ALWAYS RESET THE VALUE FUNCTIONS WHEN IT'S THE VERY FIRST LEARNING STEP (becaue we don't want to keep histroy from a earlier learning process on the same learner!)
        # Update the initial learning rate for the value functions at each learning step to a smaller value than the previous learning step
        # SHOULD WE SET IT TO THE AVERAGE LEARNING RATE FROM THE PREVIOUS LEARNING STEP?? (so that we start off where we left at the last learning moment)
        alpha_initial_at_current_learning_step = alpha_initial / t_learn

        #-- Optionally adjust the initial learning rate alpha
        # Compute the K-L divergence of the current policy w.r.t. the previous policy to decide whether to update the initial learning rate or not
        # If the new policy is too different from the previous policy we should NOT update the learning rate because the value functions learned
        # under the previous policy is most likely far away from the true values functions under the new policy.
        # Note: rel_entr(p, q) computes the term p * log(p/q) contributing to the relative entropy or Kullback-Leibler divergence
        # of the "new" distribution p(x) w.r.t. the "old" distribution q(x), which is defined as the expectation (measured over p(x)) of the
        # difference between log(p(x)) and log(q(x)). Hence if p(x) is larger than q(x) the contribution is positive and
        # if it is smaller, the contribution is negative. Therefore a positive K-L divergence typically corresponds to an increase
        # in the probability from q(x) at the values of x having a large "new" probability p(x) value.
        # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html
        # Note that we could also use the function scipy.stats.entropy() to compute directly the K-L divergence (i.e. without summing over the rel_entr() values, as done here)
        policy = test_ac.policy_nn.get_policy_values()
        KL_distance = np.sum(rel_entr(policy, policy_prev)) if t_learn > 1 else 0.0
        print(f"K-L distance with previous policy: {KL_distance} (should be < {KL_THRESHOLD} to reduce initial alpha)")
        KL_all[t_learn - 1] = KL_distance
        policy_prev = policy.copy()

        if adjust_alpha_initial_by_learning_step:
            # Update the initial learning rate alpha ONLY when:
            # - the learning step is larger than or equal to a minimum
            # - the policy did NOT change significantly from the previous learning step
            #   (because in that case we can consider the current estimate to be a reliable estimator of the value functions)
            if t_learn >= t_learn_min_to_adjust_alpha:
                if policy_changed_from_previous_learning_step(KL_distance, test_ac.env2d.getNumStates()): #np.abs(loss_all[t_learn-2]) > 1.0
                    simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial)
                else:
                    simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial / 10)
            alpha_all[t_learn - 1] = simulator_value_functions.getAgent().getLearner().getInitialLearningRate()
        print(
            f"*** INITIAL learning rate alpha = {simulator_value_functions.getAgent().getLearner().getInitialLearningRate()} " +
            (f"(adjustment happens starting at t_learn >= {t_learn_min_to_adjust_alpha}) " if adjust_alpha_initial_by_learning_step else "***\n"))

        # Pass a different seed (for the simulator) for each learning step... o.w. we will be using the same seed for them at every learning step!!
        seed_this = test_ac.seed + t_learn - 1

        print(f"Learning the CRITIC (at the current policy) using {learning_method.upper()}...")
        # Learn the value functions using the FV simulator
        if learning_method == "values_fv":
            V, Q, state_counts, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, n_events_et, n_events_fv = \
                simulator_value_functions.run(nepisodes=n_episodes_per_learning_step, max_time_steps_per_episode=max_time_steps_per_episode,
                                              max_time_steps_fv=max_time_steps_fv_for_all_particles, min_num_cycles_for_expectations=0,
                                                ## Note: We set the minimum number of cycles for the estimation of E(T_A) to 0 because we do NOT need
                                                ## the estimation of the average reward to learn the optimal policy, as it cancels out in the advantage function Q(s,a) - V(s)!!
                                              reset_value_functions=reset_value_functions_at_this_step, seed=seed_this, verbose=True, verbose_period=verbose_period)
            #average_reward = test_ac.agent_nn_fv.getLearner().getAverageReward()  # This average reward should not be used because it is inflated by the FV process that visits the states with rewards more often
            average_reward = expected_reward
            nsteps_all[t_learn-1] = n_events_et + n_events_fv
            print(f"Learning step {t_learn}: Learning of value functions COMPLETED using {learning_method} method on {nsteps_all[t_learn-1]} time steps")
        else:
            # TD learners
            learning_episodes_observe = [7, 20, 30, 40]
            V, Q, state_counts, _, _, info = \
                simulator_value_functions.run(nepisodes=n_episodes_per_learning_step, max_time_steps=max_time_steps_benchmark, max_time_steps_per_episode=max_time_steps_per_episode, #max_time_steps_benchmark // n_episodes_per_learning_step,
                                             reset_value_functions=reset_value_functions_at_this_step, seed=seed_this,
                                             state_observe=np.ravel_multi_index((1, env_shape[1]-1), env_shape), compute_rmse=True if t_learn in learning_episodes_observe else False, plot=False if t_learn in learning_episodes_observe else False,
                                             verbose=True, verbose_period=verbose_period)
            average_reward = simulator_value_functions.getAgent().getLearner().getAverageReward()
            nsteps_all[t_learn - 1] = info['t'] + 1     # +1 because the t value when just one time step is run is 0 (see the run() method called above)
            print(f"Learning step {t_learn}: Learning of value functions COMPLETED using {learning_method} method on {nsteps_all[t_learn-1]} time steps")

        state_counts_all[t_learn-1, :] = state_counts
        V_all[t_learn-1, :] = V
        Q_all[t_learn-1, :, :] = Q.reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())

        print(f"Learning the POLICY {policy_learning_mode.upper()}...")
        # Policy learning
        if policy_learning_mode == "online":
            # ONLINE with critic provided by the action values learned above using the TD simulator
            loss_all[t_learn - 1] = learner_ac.learn(n_episodes_per_learning_step, max_time_steps_per_episode=max_time_steps_per_policy_learning_episode, prob_include_in_train=1.0,
                                                     action_values=Q)
            R_all[t_learn - 1] = learner_ac.average_reward_over_episodes
            R_long_all[t_learn - 1] = average_reward  # This is the FV-based estimated average reward
        else:
            # OFFLINE learner where ALL states and actions are swept and the loss computed on all of them using the state distribution as weights
            loss_all[t_learn-1] = learner_ac.learn_offline_from_estimated_value_functions(V, Q, state_counts)
            R_all[t_learn-1] = average_reward

        # Check if we need to stop learning because the average reward didn't change a bit
        if  break_when_no_change and t_learn > 1 and R_all[t_learn-1] - R_all[t_learn-2] == 0.0 or \
            break_when_goal_reached and np.isclose(R_all[t_learn-1], max_avg_reward_episodic, rtol=0.001):
            print(f"*** Policy learning process stops at t_learn={t_learn} because the average reward didn't change a bit from previous learning step! ***")
            break

time_end = timer()
print("{} learning process took {:.1f} minutes".format(learning_method.upper(), (time_end - time_start) / 60))

# Make a copy of the measures that we would like to compare (on the same plot)
if learning_method in ["all_online", "values_tdl"]:
    loss_all_online = loss_all.copy()
    R_all_online = R_all.copy()
    R_long_all_online = R_long_all.copy()
    Q_all_online = Q_all.copy()
    V_all_online = V_all.copy()
    state_counts_all_online = state_counts_all.copy()
    nsteps_all_online = nsteps_all.copy()
    KL_all_online = KL_all.copy()
    alpha_all_online = alpha_all.copy()
elif learning_method == "values_td":
    loss_all_td = loss_all.copy()
    R_all_td = R_all.copy()
    R_long_all_td = R_long_all.copy()
    Q_all_td = Q_all.copy()
    V_all_td = V_all.copy()
    state_counts_all_td = state_counts_all.copy()
    nsteps_all_td = nsteps_all.copy()
    KL_all_td = KL_all.copy()
    alpha_all_td = alpha_all.copy()
elif learning_method == "values_tda":
    loss_all_tda = loss_all.copy()
    R_all_tda = R_all.copy()
    R_long_all_tda = R_long_all.copy()
    Q_all_tda = Q_all.copy()
    V_all_tda = V_all.copy()
    state_counts_all_tda = state_counts_all.copy()
    nsteps_all_tda = nsteps_all.copy()
    KL_all_tda = KL_all.copy()
    alpha_all_tda = alpha_all.copy()
elif learning_method == "values_fv":
    loss_all_fv = loss_all.copy()
    R_all_fv = R_all.copy()
    R_long_all_fv = R_long_all.copy()
    Q_all_fv = Q_all.copy()
    V_all_fv = V_all.copy()
    state_counts_all_fv = state_counts_all.copy()
    nsteps_all_fv = nsteps_all.copy()
    KL_all_fv = KL_all.copy()
    alpha_all_fv = alpha_all.copy()

# Plot loss and average reward for the currently analyzed learner
print("\nPlotting...")
ax_loss = plt.figure().subplots(1,1)
ax_loss.plot(range(1, n_learning_steps+1), loss_all[:n_learning_steps], marker='.', color="red")
ax_loss.plot(range(1, n_learning_steps+1), alpha_all[:n_learning_steps], '--', color="cyan")
ax_loss.set_xlabel("Learning step")
ax_loss.set_ylabel("Loss")
ax_loss.axhline(0, color="red", linewidth=1, linestyle='dashed')
ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_loss.legend(["Loss", "alpha0"], loc='upper left')
ax_R = ax_loss.twinx()
ax_R.plot(range(1, n_learning_steps+1), R_all[:n_learning_steps], marker='.', color="green")
ax_R.axhline(max_avg_reward_episodic, color="green", linewidth=1)
ax_R.plot(range(1, n_learning_steps+1), R_long_all[:n_learning_steps], marker='.', color="greenyellow")
ax_R.axhline(max_avg_reward_continuing, color="greenyellow", linewidth=1)
ax_R.set_ylabel("Average reward")
ax_R.plot(range(1, n_learning_steps+1), KL_all[:n_learning_steps], color="blue", linewidth=1)
ax_R.axhline(KL_THRESHOLD, color="blue", linestyle="dashed")
ax_R.axhline(0, color="green", linewidth=1, linestyle='dashed')
ax_R.legend(["Average reward (episodic)", "Max. average reward (episodic)",
            "Long-run average reward estimated by value functions learner", "Max. average reward (continuing)",
            "K-L divergence with previous policy", "K-L threshold for reduced alpha0", "K-L divergence between consecutive policies"],
            loc="upper right")
plt.title(f"{learning_method.upper()} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={simulator_value_functions.getAgent().getLearner().gamma}) - Labyrinth {env_shape}\n"
          f"Evolution of the LOSS (left, red) and Average Reward (right, green) with the learning step")

#-- Plot the value functions for the state next to the terminal state
marker = ''
Q_all_baseline = Q_all[:n_learning_steps, :, :] - np.tile(V_all[:n_learning_steps, :].T, (test_ac.env2d.getNumActions(), 1, 1)).T
ax_Q, ax_Q_baseline = plt.figure().subplots(1, 2)
state_of_interest = np.ravel_multi_index((1, env_shape[1]-1), env_shape)
ax_Q.plot(range(1, n_learning_steps + 1), V_all[:n_learning_steps, state_of_interest], marker=marker, color="black")
ax_Q.plot(range(1, n_learning_steps + 1), Q_all[:n_learning_steps, state_of_interest, :], marker=marker)
ax_Q.legend(["V(s)"] + ["Q(s," + str(a) + ")" for a in range(Q_all.shape[2])], loc='upper left')
ax_Q_baseline.plot(range(1, n_learning_steps + 1), V_all[:n_learning_steps, state_of_interest] - V_all[:n_learning_steps, state_of_interest], marker=marker, color="white") # We plot this constant value 0 so that the legend is correct
ax_Q_baseline.plot(range(1, n_learning_steps + 1), Q_all_baseline[:n_learning_steps, state_of_interest, :], marker=marker)
ax_Q_baseline.legend(["V(s) - V(s)"] + ["Q(s," + str(a) + ") - V(s)" for a in range(Q_all.shape[2])], loc='upper left')

# Optimum Q-values (for the optimum deterministic policy)
# This assumes that there is a reward of 1 at the terminal state which is one step away
if learning_criterion == LearningCriterion.AVERAGE:
    assert learning_task == LearningTask.CONTINUING
    svalue  = 0.5 * (1.0 - max_avg_reward_continuing)     # Differential state value V(s) under the optimal policy (since the policy tells the agent to always go up, the agent receives reward 1.0, which is corrected (subtracted) by the max (because we are following the OPTIMAL policy) average reward; the 0.5 factor is explained by the calculations on my Mas de Canelles notebook sheet)
    qvalue0 = 0.5 * (1.0 - max_avg_reward_continuing)     # Differential optimal action value Q(s,a) of going up (a=0) when we start at s = state_of_interest, one cell away from the terminal state, which gives reward 1.0 (we subtract the max average reward because we are following the OPTIMAL policy; the 0.5 factor is explained by the calculations on my Mas de Canelles notebook sheet)
    qvalue1 = qvalue0 - max_avg_reward_continuing
    qvalue2 = qvalue0 - 2*max_avg_reward_continuing
    qvalue3 = qvalue0 - max_avg_reward_continuing
else:   # DISCOUNTED reward criterion
    gamma = test_ac.gamma
    reward_at_terminal = 1
    # State value V(s):
    # - Under the optimal policy we go always up and observe the terminal reward right-away
    # - If the learning task is continuing, we still keep observing the terminal reward discounted by the length of the optimal path (= np.sum(test_ac.env2d.shape) - 1)
    # (the `-1` at the end of the parenthesis cancels the `1+` at the beginning of the parenthesis when the learning task is CONTINUING)
    svalue  = reward_at_terminal * (1 + int(learning_task == LearningTask.CONTINUING) * (1 / (1 - gamma**(np.sum(test_ac.env2d.shape)-1))) - 1)
    qvalue0 = svalue
    qvalue1 = gamma * svalue
    qvalue2 = gamma**2 * svalue
    qvalue3 = gamma * svalue
ax_Q.axhline(qvalue0, linestyle='dashed', color="blue")
ax_Q.axhline(qvalue1, linestyle='dashed', color="orange")
ax_Q.axhline(qvalue2, linestyle='dashed', color="green")
ax_Q.axhline(qvalue3, linestyle='dashed', color="red")
ax_Q.set_xlabel("Learning step")
ax_Q.set_ylabel("Q values and state values")
ax_Q_baseline.axhline(qvalue0 - svalue, linestyle='dashed', color="blue")
ax_Q_baseline.axhline(qvalue1 - svalue, linestyle='dashed', color="orange")
ax_Q_baseline.axhline(qvalue2 - svalue, linestyle='dashed', color="green")
ax_Q_baseline.axhline(qvalue3 - svalue, linestyle='dashed', color="red")
ax_Q_baseline.set_xlabel("Learning step")
ax_Q_baseline.set_ylabel("Q values w.r.t. baseline")
plt.suptitle(f"{learning_method.upper()} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={simulator_value_functions.getAgent().getLearner().gamma}) - Labyrinth {env_shape}\nQ(s,a) and V(s) for state previous to the terminal state under the optimal policy, i.e. s={state_of_interest}\nMax average reward (continuing) = {max_avg_reward_continuing}")


# Same plot for all states
axes = plt.figure().subplots(test_ac.env2d.shape[0], test_ac.env2d.shape[1])
first_learning_step = 0 #n_learning_steps * 3 // 4  #0
y2max = int(round(np.max(state_counts_all)*1.1)) # For a common Y2-axis showing the state counts
min_V, max_V = np.min(V_all), np.max(V_all)      # For a common Y-axis showing the value functions
min_Q, max_Q = np.min(Q_all), np.max(Q_all)      # For a common Y-axis showing the value functions
Q_all_baseline = Q_all[first_learning_step:n_learning_steps, :, :] - np.tile(V_all[first_learning_step:n_learning_steps, :].T, (test_ac.env2d.getNumActions(), 1, 1)).T
min_Q_baseline, max_Q_baseline = np.min(Q_all_baseline), np.max(Q_all_baseline)      # For a common Y-axis showing the value functions
ymin, ymax = min(min_V, min_Q), max(max_V, max_Q)
ymin_baseline, ymax_baseline = min(0, min_Q_baseline), max(0, max_Q_baseline)

plot_baseline = False
ylim = (ymin_baseline, ymax_baseline) if plot_baseline else (ymin, ymax)     # Use this for common Y-axis limits
#ylim = (None, None)     # Use this for unequal Y-axis limits
marker = ''
for i, ax in enumerate(axes.reshape(-1)):
    # Value functions on the left axis
    if plot_baseline:
        # Q values with baseline (so that we can better see the difference in value among the different actions)
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), V_all[first_learning_step:n_learning_steps, i] - V_all[first_learning_step:n_learning_steps, i], marker=marker, color="black")  # We plot this so that the legend is fine
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), Q_all_baseline[first_learning_step:n_learning_steps, i, :], marker=marker)
    else:
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), V_all[first_learning_step:n_learning_steps, i], marker=marker, color="black")
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), Q_all[first_learning_step:n_learning_steps, i, :], marker=marker)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(ylim)

    # State counts on the right axis
    ax2 = ax.twinx()
    ax2.plot(np.arange(1+first_learning_step, n_learning_steps+1), state_counts_all[first_learning_step:n_learning_steps, i], color="violet", linewidth=1)
    ax2.set_ylim((0, y2max))
    y2ticks = [0, int(min(state_counts_all[first_learning_step:n_learning_steps, i])), int(max(state_counts_all[first_learning_step:n_learning_steps, i])), int(np.max(state_counts_all))]
    ax2.set_yticks(y2ticks)
    ax2.yaxis.set_ticklabels(y2ticks, fontsize=7)
# Only show the x and y labels on the bottom-right plot (to avoid making the plot too cloggy)
ax.set_xlabel("Learning step")
ax2.set_ylabel("State count")
ax.legend(["V(s)"] + ["Q(s," + str(a) + ")" for a in range(Q_all.shape[2])], loc='upper left')
ax2.legend(["State count"], loc='upper right')
plt.suptitle(f"{learning_method.upper()} - {learning_task.name} learning task - {learning_criterion.name} reward criterion - Labyrinth {env_shape}\nEvolution of the value functions V(s) and Q(s,a) with the learning step by state\nMaximum average reward (continuing): {max_avg_reward_continuing}")


#-- Final policy parameters and policy distribution by state
env = test_ac.env2d
gridworld_shape = env.shape
policy = learner_ac.getPolicy()
learner = learner_ac.getValueFunctionsLearner()
print("Final network parameters:")
print(list(policy.getThetaParameter()))

# Plot the evolution of a few weights of the neural network
# TBD

colormap = cm.get_cmap("rainbow")  # useful colormaps are "jet", "rainbow", seismic"
colornorm = None

# Policy for each action at each state
axes = plt.figure().subplots(*gridworld_shape)
probs_actions_toplot = np.nan*np.ones((3, 3))
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        state_1d = np.ravel_multi_index((i, j), gridworld_shape)
        print("")
        for action in range(env.getNumActions()):
            print(f"Computing policy Pr(a={action}|s={(i,j)})...", end= " ")
            idx_2d = (0, 1) if action == 0 else (1, 2) if action == 1 else (2, 1) if action == 2 else (1, 0)
            probs_actions_toplot[idx_2d] = policy.getPolicyForAction(action, state_1d)
            print("p = {:.3f}".format(probs_actions_toplot[idx_2d]))
        img = axes[i,j].imshow(probs_actions_toplot, cmap=colormap, vmin=0, vmax=1)
plt.colorbar(img, ax=axes)  # This adds a colorbar to the right of the FIGURE. However, the mapping from colors to values is taken from the last generated image! (which is ok because all images have the same range of values.
                            # Otherwise see answer by user10121139 in https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
plt.suptitle(f"{learning_method.upper()}\nPolicy at each state")

print("{} learning process took {:.1f} minutes".format(learning_method.upper(), (time_end - time_start) / 60))

#-- ALTOGETHER
# Plot all average rewards together (to compare methods)
# Color names are listed here: https://matplotlib.org/stable/gallery/color/named_colors.html
# We normalize the average reward plots so that they converge to 1.0 (easier interpretation of the plot)
ax_loss, ax_R = plt.figure().subplots(1,2)
legend = []
if 'loss_all_online' in locals():
    ax_loss.plot(range(1, n_learning_steps + 1), loss_all_online[:n_learning_steps], '-', marker='x', color="darkred")
    legend += ["TD ALL online"] if learning_method == "all_online" else [f"TD(Lambda={test_ac.agent_nn_td.getLearner().lmbda})"]
if 'loss_all_td' in locals():
    ax_loss.plot(range(1, n_learning_steps+1), loss_all_td[:n_learning_steps], marker='.', color="red")
    legend += [f"TD {policy_learning_mode}"]
if 'loss_all_tda' in locals():
    ax_loss.plot(range(1, n_learning_steps+1), loss_all_tda[:n_learning_steps], '-.', marker='.', color="red")
    legend += [f"TD Adaptive {policy_learning_mode}"]
if 'loss_all_fv' in locals():
    from Python.lib.agents.learners.episodic.discrete.td import LeaTDLambdaAdaptive
    ax_loss.plot(range(1, n_learning_steps+1), loss_all_fv[:n_learning_steps], marker='.', color="orangered")
    legend += [f"FV {policy_learning_mode} (adaptive lambda)"] if issubclass(test_ac.agent_nn_fv.getLearner().__class__, LeaTDLambdaAdaptive) else [f"FV {policy_learning_mode}"]
ax_loss.set_xlabel("Learning step")
ax_loss.set_ylabel("Loss")
ax_loss.axhline(0, color="gray")
ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_loss.set_title("Evolution of the LOSS with the learning step by learning method")
ax_loss.legend(legend)
ax_R.plot(range(1, n_learning_steps+1), R_all_online[:n_learning_steps] / max_avg_reward_episodic, '-', marker='x', color="forestgreen") if 'R_all_online' in locals() else None
ax_R.plot(range(1, n_learning_steps+1), R_all_td[:n_learning_steps] / max_avg_reward_episodic, marker='.', color="green") if 'R_all_td' in locals() else None
ax_R.plot(range(1, n_learning_steps+1), R_all_tda[:n_learning_steps] / max_avg_reward_episodic, '-.', marker='.', color="green") if 'R_all_tda' in locals() else None
ax_R.plot(range(1, n_learning_steps+1), R_all_fv[:n_learning_steps] / max_avg_reward_episodic, marker='.', color="greenyellow") if 'R_all_fv' in locals() else None
ax_R.set_xlabel("Learning step")
ax_R.set_ylabel("Average reward (normalized by the MAX average reward = {:.2g})".format(max_avg_reward_episodic))
ax_R.axhline(1, color="gray")
ax_R.axhline(0, color="gray")
ax_R.set_title("Evolution of the NORMALIZED Average Reward with the learning step by learning method")
ax_R.legend(legend, loc="center left")
plt.suptitle(f"ALL LEARNING METHODS: Labyrinth {env_shape} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={simulator_value_functions.getAgent().getLearner().gamma})")

# If we want to add the ratio between number of steps used by two methods compared
df_ratio_nsamples = pd.DataFrame({'td': nsteps_all_td, 'fv': nsteps_all_fv, 'ratio_fv_td': nsteps_all_fv / nsteps_all_td})
ax_R_nsamples = ax_R.twinx()
ax_R_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'][:n_learning_steps], color="blue", linewidth=0.5)
ax_R_nsamples.axhline(1.0, color="blue", linewidth=0.5, linestyle="dashed")
ax_R_nsamples.set_ylim((ax_R.get_ylim()[0], None))
ax_R_nsamples.legend(["Sample size ratio (FV/TD)", "Reference line showing equal sample size ratio"], loc="center right")



# Distribution of state counts at last learning step run
state_counts = learner.getStateCounts()
state_counts_2d = np.array(state_counts).reshape(*gridworld_shape)
print(state_counts_2d)
print(state_counts_2d / np.sum(state_counts_2d.reshape(-1)))
ax = plt.figure().subplots(1,1)
img = ax.imshow(state_counts_2d, cmap=colormap, norm=colornorm)
plt.colorbar(img)

# Let's look at the trajectories of the learner (it works when constructing the learner with store_history_over_all_episodes=True)
print(len(learner.getStates()))
print([len(trajectory) for trajectory in learner.getStates()])

raise KeyboardInterrupt


########
# 2023/03/08: Test the package optparse to parse arguments when calling a script from the command prompt, specially
# its capabilities of parsing an argument that should be interpreted as a list.
# Goal: Run several replications of the same simulation (using different seeds)
# Ref:
# https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
# https://docs.python.org/3.7/library/optparse.html#option-callbacks

import optparse

# ---------------------------- Auxiliary functions ---------------------------#
def convert_str_to_list_of_floats(option, opt, value, parser):
    print(f"option: {type(option)}, {dir(option)}")
    print(f"opt: {opt}")
    print(f"value: {value}")
    print(f"parser: {parser}")
    print(f"parser.values: {parser.values}")
    if isinstance(value, str):
        setattr(parser.values, option.dest, [float(s) for s in value.split(",")])

def parse_input_parameters(argv):
    # Written for uugot.it project in Apr-2021
    # Parse input parameters
    # Ref: https://docs.python.org/3.7/library/optparse.html
    # Main steps:
    # 1) The option parser is initialized with optparse.OptionParser(), where we can specify the usage= and version=,
    # as e.g. `optparse.OptionParser(usage="%prog [-v] [-p]", version="%prog 1.0")`
    # 2) New options to parse are added with parser.add_option(), where the metavar= argument (e.g. `metavar="FILE"`)
    # is used to indicate the option expects a value to be specified (e.g. `--filename="file.txt"` as opposed to `--verbose`, which expects no value).
    # We can also define:
    #    a) the default value of the option (although this is more clearly done with parser.set_defaults().
    #    b) the action to take with the option value read with the action= argument, e.g. "store_true", "store_false",
    #       which are actually needed for FLAG options that do NOT require any option value (e.g. -v for verbose, etc.),
    #       and ***whose default value (i.e. when the flag is not given) is specified by the default= parameter***.
    #       The default action is "store" which is used for options accepting a value as in `--file="file.txt".
    #       --> NOTE that the action can be "callback" meaning that a callback function with the signature callback(option, opt, value, parser)
    #       is called to parse the argument value. In this case, if the value of the argument needs to be updated
    #       (e.g. a string converted to a list) we need to:
    #       - define the name of the argument to set with the `dest=` option of the parser.add_option() method.
    #       - set the value of the argument in the callback by calling `setattr(parser.values, option.dest, <value>)`.
    #       Ref: https://docs.python.org/3.7/library/optparse.html#option-callbacks
    #    b) the type of the option value expected with the type= argument (e.g. type="int"), which defaults to "string".
    #    c) the store destination with the dest= argument defining the attribute name of the `options` object
    #       created when running parser.parse_args() (see next item) where the option value is stored.
    #       See more details about the default value of dest= below.
    # 3) Options are parsed with parser.parse_args() into a tuple (options, args), where `options` is an object
    # that contains all the name-value pair options and `args` is an object containing the positional parameters
    # that come after all other options have been passed (e.g. `-v --file="file.txt" arg1 arg2`).
    # 4) Every option read is stored as an attribute of the `options` object created by parser.parse_args()
    # whose name is the value specified by the dest= parameter of the parser.add_option() method, or its
    # (intelligent) default if none is specified (e.g. the option '--model_pos' is stored as options.model_pos by default)
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage="%prog [--method] [--t_learn] [--replications] [--theta_ref] [--theta_start] [-N] [-T] [--J_factor] [-d] [-v]")
    parser.add_option("--method",
                      metavar="METHOD",
                      help="Learning method [default: %default]")
    parser.add_option("--t_learn",
                      type="int",
                      metavar="# Learning steps",
                      help="Number of learning steps [default: %default]")
    parser.add_option("--replications",
                      type="int",
                      metavar="# Replications", default=1,
                      help="Number of replications to run [default: %default]")
    parser.add_option("--theta_start", dest="theta_start",
                      type="str",
                      action="callback",
                      callback=convert_str_to_list_of_floats,
                      metavar="Initial theta",
                      help="Initial theta parameter for the learning process [default: %default]")
    parser.add_option("-N",
                      type="int",
                      metavar="# particles", default=1,
                      help="Number of Fleming-Viot particles [default: %default]")
    parser.add_option("-T",
                      type="int",
                      metavar="# arrival events", default=100,
                      help="Number of arrival events to observe before ending the simulation [default: %default]")
    parser.add_option("-d", "--debug", dest="debug", default=False,
                      action="store_true",
                      help="debug mode")
    parser.add_option("-v", "--verbose", dest="verbose", default=False,
                      action="store_true",
                      help="verbose: show relevant messages in the log")

    parser.set_defaults(method="MC",
                        t_learn=10,
                        theta_start=0.1)

    (options, args) = parser.parse_args(argv)

    print("Parsed command line options: " + repr(options))

    # options: dictionary with name-value pairs
    # args: argument values (which do not require an argument name
    return options, args
# ---------------------------- Auxiliary functions ---------------------------#

# Parse input parameters
options, args = parse_input_parameters(sys.argv[1:])
print(f"options: {options}")
print(f"args: {args}")

raise KeyboardInterrupt


########
# 2021/02/03: Test a generator of value combinations.
# Goal: Generate all possible states of a multidimensional Markov chain.
import numpy as np
from math import factorial
from time import time

from Python.lib.utils.computing import all_combos_with_sum, comb

def prob(n, const, rho):
    return np.prod( [(1- r)*r**nr for r, nr in zip(rho, n)] ) / const

C = 20
R = 3
rho = [0.5]*R
#rho = [0.2, 0.875, 0.833]

time_start = time()
const = 0
ncases_total = 0
prod = [0]*(C+1)   # Array to store the contributions to the normalizing for each 1 <= c <= C
for c in range(C+1):
    print("Computing normalizing constant for R={}, c={}...".format(R, c), end=" ")
    ncases = comb(c+R-1,c)
    combos = all_combos_with_sum(R, c)
    count = 0
    while True:
        try:
            v = next(combos)
            #print(v, end=" ")
            assert len(v) == len(rho), "The length of v and rho coincide ({}, {})".format(len(v), len(rho))
            prod[c] += np.prod( [(1- r)*r**nr for r, nr in zip(rho, v)] )
            count += 1
        except StopIteration as e:
            #print("END!")
            break
    combos.close()
    const += prod[c]
    print("--> generated combinations: {}".format(count))
    #print("prod: {}".format(prod))
    #print("const: {}".format(const))
    assert count == ncases
    ncases_total += ncases
assert const <= 1, "The normalizing constant is <= 1"
assert abs(sum(prod)/const - 1.0) < 1E-6

# Blocking probability
pblock = prod[C] / const
time_end = time()

print("\nExecution time: {} sec".format(time_end - time_start))
print("Total number of cases: {}".format(ncases_total))
print("Normalizing constant for rho={}: {}".format(rho, const))
print("Blocking probability: Pr(C)={:.5f}%".format(pblock*100))
