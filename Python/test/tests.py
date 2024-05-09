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
from Python.lib.environments.gridworlds import EnvGridworld1D
import Python.lib.agents as agents
from Python.lib.agents.learners import LearningCriterion, LearningTask, ResetMethod
from Python.lib.agents.learners.episodic.discrete import td, fv
from Python.lib.simulators.discrete import Simulator as DiscreteSimulator
from Python.lib.utils.computing import percentile
import Python.lib.utils.plotting as plotting

# Environment
nS = 5
start_states = {0}
# Use the following for a random selection of the start state outside A
#start_states = set(np.arange(2, nS-1))
isd = [1/len(start_states) if s in start_states else 0.0 for s in range(nS)]
env1d = EnvGridworld1D(length=nS, rewards_dict={nS-1: +1.0}, reward_default=0.0, initial_state_distribution=isd)
print(f"Policy in {nS}-state gridworld:")
for k, v in env1d.P.items():
    print(f"State {k}: {v}")
print(f"Terminal states in {nS}-state gridworld: {env1d.getTerminalStates()}")
print(f"Rewards: {env1d.getRewardsDict()}")
print(f"Start state distribution: {env1d.isd}")

# Policy
policy_probabilities = [0.7, 0.3] #[0, 1] #[0.7, 0.3] #[0.1, 0.9]   #[0.5, 0.5]) #[0.9, 0.1])
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
# For the continuing learning task, update the transition probability of the terminal state, because when reached we want it to start again from the environment's start state
# (which can be a distribution over states, not necessarily a fixed state)
# so that the learning task is continuing (recall that `-1` in the row means "transition from the last element of the array", i.e. the "last state").
P_con[-1, :] = env1d.getInitialStateDistribution()
# Check that P is a valid transition probability matrix
for s in range(nS):
    assert np.isclose(np.sum(P[s]), 1.0)

# We can estimate the stationary distribution for the continuing learning task by computing P^t for t sufficiently large
# See below another computation of the stationary distribution, based on the eigendecomposition of P.
# HOWEVER, THIS ONLY WORKS WHEN THERE IS NO SPECIFIC PERIODICITY IN TERMS OF HOW WE CAN GET TO A PARTICULAR STATE!!
# (i.e. when the Markov chain is aperiodic...)
# For instance: when nS is an EVEN number in this 1D gridworld environment, elevating P to an EVEN power gives a different result
# than elevating P to an ODD number! (because the probability of being at certain states after an even number of steps is 0;
# example: if nS = 4 and we start at state 0, the probability of being again at state 0 after 1, 3, 5, 7, ... steps is 0 because
# even if we come back to state 0 after reaching the terminal state, it takes always an EVEN number of steps to return to 0!
# However, if nS = 5 and we start at state 0, the probability of being again at state 0 after 1 or 3 steps is 0 but the probability
# of being at state 0 after 5 steps is NOT zero because it suffices the system to go through the terminal state back to the initial state 0.
# And this trajectory automatically makes the probability of being again at state 0 after 7, 9, ... steps no longer 0 as is the case when nS is even!
# *** The SOLUTION is to compute mu as 0.5 * (P^t + P^(t+1)) for t sufficiently large, and this works for both nS odd and even. ***
# *** Note that the general expression is to compute (P^t + P^(t+1) + ... P^(t+d-1)) / d, where d is the periodicity of the Markov chain. ***
# *** See also Asmussen, pag. 171, expression (1.6), where he talks about regenerative processes. ***
if policy.isDeterministic():
    # In the deterministic policy case, we assume that the policy is always going to the right or always going to the left.
    # In such case, the period is equal to the number of states in the gridworld and hence, a stationary distribution does not really exist, because it is NOT unique.
    # as the distribution at each state DEPENDS ON THE START AND END STATES. However, we can compute the "stationary" distribution (which would be obtained as the eigvenvector
    # of the largest eigenvalue of the transition matrix, as the "period" average of P^d, where d is the period.
    evector = (P_con**0)[0,:]   # This means: we consider that the start state is state s=0 (because at time 0 we choose the vector [1, 0, ..., 0] as the initial distribution of the states
    for k in range(1, nS):
        evector += (P_con**k)[0, :]
    mu = evector / nS
else:
    # There is periodicity ONLY when the number of states in the environment is EVEN, in which case the period is 2
    mu = 0.5 * (np.squeeze(np.array((P_con**9999)[0, :])) + np.squeeze(np.array((P_con**10000)[0, :])) )
print(f"Stationary probability distribution for cont. learning task estimated as P^Inf: {mu}")
# When probs = [0.9, 0.1], we get 0.00054319 for the "terminal" state with reward +1, i.e. a reasonably small occupation probability to showcase FV
# Check that this is the statioary distribution
if not policy.isDeterministic():
    assert np.allclose(mu*P_con, mu, atol=1E-4)
## OK

# We can also find the stationary distribution as the eigenvector of P' that corresponds to eigenvalue = 1
# Note that:
# - the eigen-decomposition is computed on the transpose of P (as we are interested in the left-eigen-decomposition of P, which is equal to the right-eigen-decomposition of P')
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
# Assign mu to prob_stationary, as this value already takes care of the possible periodicity of the Markov chain
mu = prob_stationary
## OK!!


#-- Define learning task and learning criterion to test
learning_task = LearningTask.CONTINUING
#learning_task = LearningTask.EPISODIC

learning_criterion = LearningCriterion.AVERAGE; gamma = 1.0
#learning_criterion = LearningCriterion.DISCOUNTED; gamma = 0.9


# We now compute the true state value functions under the given policy, against which we will compare our state value function estimates.
# To find the true state value function V(s) we solve the Bellman equation using gamma -> 1
# (Note that we cannot use gamma = 1 because (I - P) is singular; in fact, the largest eigenvalue of P is 1.0
# which means that (I - P) has an eigenvalue equal to 0.0 (in fact we can decompose P as Q*D*Q^-1 and therefore
# we can write (I - P) = Q * (I - D) * Q^-1 which means that one of the elements of the diagonal matrix (I - D) is 0
# (i.e. the element associated to the eigenvalue 1.0 in D).
# HOWEVER, if gamma = 1, using np.linalg.pinv() we can compute the pseudo-inverse, i.e. the Moore-Penrose generalized inverse,
# which provides a matrix (I - P)^+ such as (I - P)^+ * b gives V in (I - P)*V = b where V has minimum norm among
# all possible V solutions of the system of equation (I - P)*V = b.
# Recall that `b` is the vector (indexed by each state x) containing the EXPECTED one-step reward (from x) received by the agent under the given policy,
# i.e. b[x] = sum_{y,a}{ p(y,a|x) * r(y) } = sum_{y,a}{ p(y|a,x) * policy(a|x) * r(y) } = sum_{y}{ P[x,y] * r(y) }
# In our case, where the reward landscape is r(y) = 1.0 * Ind{y=4}, we have that b[x] = 0 for all x except x = 3,
# at which b[x=3] = P[x=3,2]*0.0 + P[x=3,4]*1.0 = P[x=3,4]*r (= P[-2,-1]*r in Python extended notation that allows us to be independent of the gridworld size)
# Finally, in the continuing learning task case, the Bellman equation is: (I - P)*V = b - g*1,
# where g is the average reward (i.e. the average reward observed over all states under stationarity), and `1` is the vector of ones.
r = env1d.getReward(nS-1)     # Reward received at the terminal state (assumed there is only one terminal state)
gamma_almost1 = 0.999999
# True value functions under the different learning tasks and learning criteria, namely:
# V_true_epi_disc: learning task = EPISODIC; learning criterion = DISCOUNTED
# V_true_epi_avg: UNDEFINED
#   --> IN FACT, in this case the state value function V(s) is defined as the expected average reward observed in the episode,
#       but this is a complicated quantity because its calculation implies dividing by the length of the episode, T, which is a random variable!
#       We could perhaps talk about the TOTAL expected reward, as done next.
# V_true_epi_tot: V(s) = E[ sum-of-rewards-observed-in-episode-of-length-T ]
#       However in our case, where there is only one non-zero reward at the rightmost state, such learning criterion (TOTAL) is not sensible
#       because for every non-terminal state x, the total observed rewards for ANY episode length under ANY policy is ALWAYS 1, i.e. the reward received at termination!
#       (this is confirmed by the value of V_true_epi_tot computed below, where V(s) ~= 1 for all states other than the terminal!)
#       The total reward criterion would be ok if we had reward = -1 at the leftmost state and +1 at the rightmost state because in that case the rewards are different
#       at the different terminal states and this is precisely the example widely presented in e.g. Sutton where they use widely the total reward criterion as they
#       tend to use gamma = 1 for that example.
# V_true_con_avg: learning task = CONTINUING; learning criterion = AVERAGE
# V_true_con_disc: learning task = CONTINUING; learning criterion = DISCOUNTED
# WE SHOULD COMPARE THE LEARNED STATE VALUE FUNCTIONS WITH THE CORRESPONDING TRUE VALUE FUNCTIONS
# depending on the Learning Task and on the Learning Criterion.
b_epi = np.array([np.sum([P_epi[x,y]*env1d.getReward(y) for y in range(nS)]) for x in range(nS)])
b_con = np.array([np.sum([P_con[x,y]*env1d.getReward(y) for y in range(nS)]) for x in range(nS)])
# Average reward, i.e. the average observed reward over all states under stationarity, since avg.reward = sum_{x} mu[x] * r[x]
# Recall that the average reward is the one that makes the system of equations satisfied by V(s) (the Bellman equations) feasible (i.e. consistent, as opposed to inconsistent).
# And recall that the average reward g appears in the independent term of the Bellman equations which is `b - g*1`.
# For more details, see the very good notes by Ger Koole: https://www.scribd.com/document/177118281/Lecture-Notes-Stochastic-Optimization-Koole
g = sum([mu[x]*env1d.getReward(x) for x in range(nS)])
I = np.eye(nS)
if gamma < 1:
    V_true_epi_disc = np.linalg.solve(I - gamma*P_epi, b_epi)
else:
    V_true_epi_disc = None
# For the state value functions defined by the total or by the average reward, we use gamma_almost1 because using 1 instead gives infinite because I - P is singular
V_true_epi_tot = np.linalg.solve(I - gamma_almost1*P_epi, b_epi)        # This "TOTAL reward" criterion is not a sensible criterion for the episodic learning task for this reward's landscape (see reason in above comment)
V_true_con_avg = np.linalg.solve(I - (gamma - 1E-6)*P_con, b_con - g)   # Note that we use (gamma - 1E-6) and NOT gamma_almost1 because gamma could still be < 1 in the average reward criterion (this is NOT the case for the total reward criterion, and this is why we use gamma_almost1 to compute V_true_epi_tot.
if gamma < 1:
    V_true_con_disc = np.linalg.solve(I - gamma*P_con, b_con)
else:
    V_true_con_disc = None
# Generalized inverse (minimum norm solution), applied for the continuing learning task case using the average reward criterion with gamma = 1
V_true_con_avg_minnorm = np.asarray(np.dot( np.linalg.pinv(np.eye(len(P_con)) - P_con), b_con - g))[0]
# Check the expected results
if policy_probabilities == [0.5, 0.5] and nS == 5 and list(env1d.getRewardsDict().keys()) == [nS-1] and gamma == 0.9:
#if policy.policy == {0: [0.0, 1.0], 1: [0.5, 0.5], 2: [0.5, 0.5], 3: [0.5, 0.5], 4: [0.0, 1.0]} :
    assert  gamma == 0.9 and \
            np.allclose(V_true_epi_disc, np.array([0.33500299, 0.37222554, 0.49216488, 0.7214742, 0.0]))
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

avg_reward_true = np.sum([mu[s]*r for s, r in env1d.getRewardsDict().items()])
print(f"True average reward for the CONTINUING learning task under policy {policy_probabilities}: {avg_reward_true}")

# Store the true state value function in the V attribute of the environment so that we can compare the estimated state value function with the true one
# Note that, under the average reward learning criterion with gamma = 1, we reference the state value function V(s) to V(0) because in that case,
# the value function V(s) is NOT unique.
# (Reason: the system of linear equations that V(s) satisfies (the Bellman equations, (I - P) V = R - g*1) is underdetermined, because one of the eigenvalues of (I - P) is 0
# AND g is the average reward (i.e. the expected reward observed under stationarity), which makes the system of equations be consistent, with one equation exactly equal to
# a linear combination of the other equations. If g were not the average reward, the system of equations would be inconsistent and no V(s) would satisfy it.)
# Ref: Gast paper at https://www.sigmetrics.org/mama/abstracts/Gast.pdf
# or Ger Koole's lecture notes at https://www.scribd.com/document/177118281/Lecture-Notes-Stochastic-Optimization-Koole
# and available at Urtzi's ownCloud as of Dec-2023: https://cloud.irit.fr/s/xRmIQolbiWagiSf
# We do this so that we can UNAMBIGUOUSLY compare the state value function estimated by the learning methods below with the true state value function!
if learning_task == LearningTask.EPISODIC and gamma < 1:
    env1d.setV(V_true_epi_disc)
    print(f"True state value function (EPISODIC / DISCOUNTED): {env1d.getV()}")
else:
    assert learning_task == LearningTask.CONTINUING
    if learning_criterion == LearningCriterion.AVERAGE:
        # Choose the reference value to correct the state value function V(s), which is non-zero ONLY when gamma = 1 (under the average reward criterion)
        V_ref_value = 0.0 if gamma < 1 else V_true_con_avg_minnorm[0]
        env1d.setV(V_true_con_avg_minnorm - V_ref_value)
        print(f"True differential state value function (CONTINUING / AVERAGE): {env1d.getV()}")
    elif learning_criterion == LearningCriterion.DISCOUNTED and gamma < 1:
        env1d.setV(V_true_con_disc)
        print(f"True state value function (CONTINUING / DISCOUNTED): {env1d.getV()}")

#-- Learn the average reward and the state value function
# Simulation parameters
N = 200 #50 #500 #200 #20                            # Number of particles in FV simulation
T = 2000 #200 #500                                   # Number of time steps of the single Markov chain simulation run as first step of the FV learning process
                                                    # This initial simulation is done to:
                                                    # - obtain an initial exploration of the environment that would help determine the absorption set A
                                                    #   (e.g. based on state visit frequency going beyond a given threshold --not yet implemented though)
                                                    # - under the AVERAGE reward criterion:
                                                    #   estimate the expected reabsorption cycle time, E(T_A) needed to compute the FV estimator of the average reward.
                                                    # - under the DISCOUNTED reward criterion:
                                                    #   obtain an initial estimation of the value functions of the states in the absorption set A
                                                    #   (which is needed by the FV simulation to bootstrap the state and action values for the states
                                                    #   at the boundary of A).
estimate_on_fixed_sample_size = True                # This is only used for the DISCOUNTED reward criterion
max_time_steps_fv_per_particle = 100 #50 #10 #50         # Max average number of steps allowed for each particle in the FV simulation
max_time_steps_fv_for_expectation = T
max_time_steps_fv_for_all_particles = max_time_steps_fv_per_particle * N
# Traditional method learning parameters
# They are set for a fair comparison with FV learning
# The maximum time steps to be observed in the benchmark methods is set to the sum of:
# - the max number of time steps allowed to estimate E(T_A)
# - the max number of time steps allowed over all the FV particles
M = 1   # Number of normal particles created during the FV simulation to explore using the underlying Markov process
        # This value MUST coincide with n_normal_max variable defined in Simulator._run_simulation_fv_learnvaluefunctions()
max_time_steps_benchmark = max_time_steps_fv_for_expectation + (1 + M) * max_time_steps_fv_for_all_particles

# Parameters common for all learners
R = 1 #20 #50             # Replications of the estimation process
seed = 1717
nepisodes = 100 #1000       # Note: this parameter is ONLY used in EPISODIC learning tasks
max_time_steps_per_episode = nS*10   #1000 #100
start_state = None  # The start state is defined by the Initial State Distribution (isd) of the environment
plot = True if R == 1 else False    # True
plot_value_functions = True
colormap = "rainbow"

alpha = 1.0
#gamma = 1.0    # gamma is defined when defining the learning criterion above or when computing the discounted state value for episodic learning tasks (normally `V_true_epi_disc`)
lmbda = 0.0     # We should NOT use lambda > 0 because FV does NOT have the logic implemented to handle this case (currently the trajectories of all particles are mixed into the same trajectory stored in the self._states list.
alpha_min = 0.01 #0.01 #0.1 #0.0
adjust_alpha = True

suptitle =  f"1D gridworld ({nS} states) with one terminal state (r=+1), policy probabilities = {policy_probabilities}, # particles = {N}, # replications = {R}, lambda={lmbda}, alpha_min={alpha_min}, adjust_alpha={adjust_alpha}" + \
            f"\n{learning_task.name} learning task, {learning_criterion.name} reward criterion"

time_start_all = timer()
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
    print(f"\n******* Running replication {rep+1} of {R} (FV + TD)..." )


    #-- Learning the average reward using FV(lambda)
    # This simulation should come BEFORE TD(lambda) so that we know for how many steps to run TD, i.e. as many steps as transitions seen by FV
    seed_this = seed + rep*13

    print(f"\nFV(lambda={lmbda})")
    # Learner and agent definition
    params = dict({'N': N,  # NOTE: We should use N = 500 if we use N = 200, T = 200 above to compute the benchmark time steps because if here we use the N defined above, the number of time steps used by FV is much smaller than the number of time steps used by TD (e.g. 1500 vs. 5000) --> WHY??
                   'T': T,                           # Max simulation time over ALL episodes run when estimating E(T_A). This should be sufficiently large to obtain an initial non-zero average reward estimate (if possible)
                   'absorption_set': set({1}), #set(np.arange(2)), #set({1}),  # Note: The absorption set must include ALL states in A (see reasons in the comments in the LeaFV constructor)
                   'activation_set': set({0, 2}),
                   'alpha': alpha,
                   'gamma': gamma,
                   'lambda': lmbda,
                   'alpha_min': alpha_min,
                   })
    learner_fv = fv.LeaFV(env1d, params['N'], params['T'], params['absorption_set'], params['activation_set'],
                          probas_stationary_start_state_et=None,
                          probas_stationary_start_state_fv=None, #dict({0: policy_probabilities[0], 2: policy_probabilities[1]}), #None,
                          criterion=learning_criterion,
                          alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                          adjust_alpha=adjust_alpha, adjust_alpha_by_episode=False, func_adjust_alpha=None, #np.sqrt,
                          #reset_method=ResetMethod.RANDOM_UNIFORM, reset_params={'min': -1, 'max': +1}, reset_seed=1713,
                          alpha_min=params['alpha_min'],
                          estimate_on_fixed_sample_size=estimate_on_fixed_sample_size,
                          debug=False)
    if False:
        learner_fv.setAverageReward(avg_reward_true)
    agent_fv = agents.GenericAgent(policy, learner_fv)

    # Simulation
    time_start = timer()
    sim = DiscreteSimulator(env1d, agent_fv, debug=False)
    state_values_fv, action_values_fv, advantage_values_fv, state_counts_fv, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, n_events_et, n_events_fv = \
        sim.run(max_time_steps_fv=max_time_steps_fv_for_all_particles,
                use_average_reward_stored_in_learner=False,
                seed=seed_this,
                verbose=True, verbose_period=50,
                plot=plot, colormap=colormap, pause=0.1)  # Use plot=True to create plots that show how learning progresses; use pause=+np.Inf to "press ENTER to continue" between plots
    time_fv = timer() - time_start
    avg_reward_fv = learner_fv.getAverageReward()
    print(f"Estimated average reward (FV): {avg_reward_fv}")
    print(f"True average reward: {avg_reward_true}")
    if R > 1:
        # Only show the results when R > 1 because when R = 1, the FV results are shown below, together with the TD results
        print(f"\nNumber of time steps used by FV(lambda): {n_events_et + n_events_fv} (= {n_events_et} for E(T) + {n_events_fv} for FV)")
        print(f"Average reward estimated by FV(lambda): {avg_reward_fv}")
        print(f"True average reward: {avg_reward_true}")
        print("Relative error: {:.1f}%".format((avg_reward_fv / avg_reward_true - 1)*100))
        print("FV(lambda) took {:.1f} minutes".format(time_fv / 60))
    counts_fv[rep] = state_counts_fv
    time_steps_fv[rep] = n_events_et + n_events_fv
    # Note that we reference the value functions to the state value of state 0, V(0), in the average reward criterion with gamma = 1, because in that case the value functions are NOT unique (see explanation above)
    estimates_V_fv[rep] = state_values_fv - (learning_criterion == LearningCriterion.AVERAGE and gamma == 1 and state_values_fv[0] or 0)
    estimates_Q_fv[rep] = action_values_fv - (learning_criterion == LearningCriterion.AVERAGE and gamma == 1 and state_values_fv[0] or 0)
    estimates_avg_fv[rep] = avg_reward_fv


    #-- Learning the average reward using TD(lambda)
    # Comment out to use same seed as for the FV case
    #seed_this = seed_this + 1

    print(f"\nTD(lambda={lmbda})")
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
    state_values_td, action_values_td, advantage_values_td, state_counts_td, RMSE_by_episode, MAPE_by_episode, learning_info = \
        sim.run(nepisodes=nepisodes,
                max_time_steps=time_steps_fv[rep], #max_time_steps_benchmark,
                max_time_steps_per_episode=+np.Inf, #max_time_steps_per_episode, #+np.Inf
                start_state_first_episode=start_state, seed=seed_this,
                estimated_average_reward=None, #avg_reward_true,
                compute_rmse=True, state_observe=None, #nS-2,  # This is the state just before the terminal state
                verbose=True, verbose_period=50,
                plot=plot, colormap=colormap, pause=0.1)    # Use plot=True to create plots that show how learning progresses; use pause=+np.Inf to "press ENTER to continue" between plots
    time_td = timer() - time_start
    avg_reward_td = learner_td.getAverageReward()
    print(f"Estimated average reward (TD): {avg_reward_td}")
    print(f"True average reward: {avg_reward_true}")
    if R == 1:
        # Show the results of FV lambda so that we can compare it easily with TD shown below
        print(f"\nNumber of time steps used by FV(lambda): {n_events_et + n_events_fv} (= {n_events_et} for E(T) + {n_events_fv} for FV)")
        print(f"Average reward estimated by FV(lambda): {avg_reward_fv}")
        print(f"True average reward: {avg_reward_true}")
        print("Relative error: {:.1f}%".format((avg_reward_fv / avg_reward_true - 1)*100))
        print("FV(lambda) took {:.1f} minutes".format(time_fv / 60))
    print(f"\nNumber of time steps used by TD(lambda): {learning_info['nsteps']}")
    print(f"Average reward estimated by TD(lambda): {avg_reward_td}")
    print(f"True average reward: {avg_reward_true}")
    print("Relative error: {:.1f}%".format((avg_reward_td / avg_reward_true - 1)*100))
    print("TD(lambda) took {:.1f} minutes".format(time_td / 60))
    counts_td[rep] = state_counts_td
    time_steps_td[rep] = learning_info['nsteps']
    # Note that we reference the value functions to the state value of state 0, V(0), in the average reward criterion with gamma = 1, because in that case the value functions are NOT unique (see explanation above)
    estimates_V_td[rep] = state_values_td - (learning_criterion == LearningCriterion.AVERAGE and gamma == 1 and state_values_td[0] or 0)
    estimates_Q_td[rep] = action_values_td - (learning_criterion == LearningCriterion.AVERAGE and gamma == 1 and state_values_td[0] or 0)
    estimates_avg_td[rep] = avg_reward_td

    if learning_task == LearningTask.CONTINUING:
        # Only for CONTINUING learning task is the number of learning steps performed by each method guaranteed to be the same
        # because in EPISODIC learning task, TD learning uses the number of learning steps defined by each episode (which we cannot control)...
        assert time_steps_td[rep] == time_steps_fv[rep], f"The number of learning steps used by FV ({time_steps_fv[rep]}) and by TD ({time_steps_td[rep]}) must be the same"
print("\n\nAll replications took {:.1f} minutes\n\n".format((timer() - time_start_all) / 60))

# Plot the learning rates by state for the last replication
ax_td, ax_fv = plt.figure().subplots(1,2)
ax_td.plot(learner_td.alphas); ax_td.legend(np.arange(nS), title="States"); ax_td.set_title("Alphas in TD learning by state"); ax_td.set_ylim((0, alpha))
ax_fv.plot(learner_fv.alphas); ax_fv.legend(np.arange(nS), title="States"); ax_fv.set_title("Alphas in FV learning by state"); ax_fv.set_ylim((0, alpha))
plt.suptitle(suptitle)

if False:   # NEED TO FIX THE CORRECT STORAGE OF THE TRAJECTORY HISTORY IN THE FV LEARNER as explained in the WARNING message below when retrieving the FV trajectory history
    # Check the state and action distribution for the last replication
    # TD
    states_history_td = np.concatenate(learner_td.states)
    print(f"Distribution of state visits (TD):\n{pd.Series(states_history_td).value_counts(sort=False)}")
    actions_history_td = np.concatenate(learner_td.actions)
    for s in range(nS):
        ind_s = states_history_td == s
        states_history_td[ind_s]
        print(f"Distribution of actions taken for s={s} (TD): count = \n{np.sum(ind_s)}")
        print(pd.Series(actions_history_td[ind_s]).value_counts() / np.sum(ind_s))
    ## OK! For the [0.5, 0.5] case, 50% for all states except state 0 and 4
    # FV
    # WARNING: The trajectory history recovered here most likely includes ONLY the exploration by the "normal" particle (this is the case when running the FV simulation with _run_simulation_fv_learnvaluefunctions())
    # The reason is that, in that case, the trajectories taken by the FV particles is stored in their respective environment.
    states_history_fv = np.concatenate(learner_fv.states)
    print(f"Distribution of state visits (FV):\n{pd.Series(states_history_fv).value_counts(sort=False)}")
    actions_history_fv = np.concatenate(learner_fv.actions)
    for s in range(nS):
        ind_s = states_history_fv == s
        states_history_fv[ind_s]
        print(f"Distribution of actions for s={s} (TD): visit count = {np.sum(ind_s)}")
        print(pd.Series(actions_history_fv[ind_s]).value_counts() / np.sum(ind_s))


#-- Plot results
# Replications
if R > 1:
    # Plot the estimates of the value function
    states = np.arange(1, nS + 1)

    if plot_value_functions:
        #np.diff(estimates_V_td, axis=1)
        #np.diff(estimates_V_fv, axis=1)
        min_V_td, max_V_td, median_V_td, mean_V_td, std_V_td, avg_events_td = estimates_V_td.min(axis=0), estimates_V_td.max(axis=0), np.median(estimates_V_td, axis=0), estimates_V_td.mean(axis=0), estimates_V_td.std(axis=0), time_steps_td.mean(axis=0)
        min_V_fv, max_V_fv, median_V_fv, mean_V_fv, std_V_fv, avg_events_fv = estimates_V_fv.min(axis=0), estimates_V_fv.max(axis=0), np.median(estimates_V_fv, axis=0), estimates_V_fv.mean(axis=0), estimates_V_fv.std(axis=0), time_steps_fv.mean(axis=0)

        # Compute percentiles
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
        plt.suptitle(suptitle + "\nDistribution of state values V(s) by state")

    # Plot the average reward estimate for each replication
    plt.figure()
    ax = plt.gca()
    ax.plot(np.arange(1, R+1), estimates_avg_td, color="red")
    ax.plot(np.arange(1, R+1), estimates_avg_fv, color="green")
    ax.axhline(avg_reward_true, color="gray", linestyle="dashed")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim((0, None))
    ax.set_xlabel("Replication")
    ax.set_ylabel("Average reward")
    ax2 = ax.twinx()
    ax2.bar(np.arange(1, R+1), height=time_steps_td, color="red", alpha=0.3)
    ax2.bar(np.arange(1, R+1), height=time_steps_fv, color="green", alpha=0.3)
    ax2.set_ylabel("# time steps")
    plt.title("Average reward estimates for each replication")
    plt.legend(["TD", "FV"])

    # Violin plots of the average reward estimate by method
    ax = plt.figure().gca()
    plotting.violinplot(ax, estimates_avg_td, positions=[1], color_body="red", color_lines="red", color_means="black")
    plotting.violinplot(ax, estimates_avg_fv, positions=[2], color_body="green", color_lines="green", color_means="black")
    ax.plot(1 + np.random.normal(0, 0.01, R), estimates_avg_td, '.', marker=".", markersize=3, color="black")
    ax.plot(2 + np.random.normal(0, 0.01, R), estimates_avg_fv, '.', marker=".", markersize=3, color="black")
    ax.axhline(avg_reward_true, color="gray", linestyle="dashed")
    ax.xaxis.set_ticks([1, 2])
    ax.xaxis.set_ticklabels([f"TD(lambda={lmbda})", f"FV(lambda={lmbda})"])
    ax.set_ylim((0, None))
    ax.set_ylabel("Average reward")
    plt.title(f"Distribution of average reward estimation on {R} replications: TD (red) vs. FV (green)")


# Estimated value functions for a given replication
rep = R - 1
ax_V, ax_Vd = plt.figure().subplots(1,2)    # ax_Vd: for plotting the difference among state values (which is what really interest us for learning an optimal policy!)
states = np.arange(1, nS+1)
ax_V.plot(states, env1d.getV(), color="blue")
ax_V.plot(states, estimates_V_td[rep], color="red")
ax_V.plot(states, estimates_V_fv[rep], color="green")
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
ax2_V.legend(["stationary probability", f"Rel. Freq.(s): TD(#events={learning_info['nsteps']})", f"Rel. Freq.(s): FV(#events={n_events_et + n_events_fv})"], loc="upper right")
ax_Vd.plot(states[:-1], np.diff(env1d.getV()), color="blue")
ax_Vd.plot(states[:-1], np.diff(estimates_V_td[rep]), color="red")
ax_Vd.plot(states[:-1], np.diff(estimates_V_fv[rep]), color="green")
ax_Vd.axhline(0, color="gray")
ax_Vd.set_xlabel("Left states")
ax_Vd.set_ylabel("DIFF V(s)")
ax_Vd.legend([f"True diff V(s)", f"Diff V(s): TD(lambda={lmbda})", f"Diff V(s): FV(lambda={lmbda})"])
ax_Vd.set_title("V(s) difference among contiguous states")
plt.suptitle(f"REPLICATION {rep+1} of {R}: " + suptitle + "\nState values V(s) and visit frequency (count) for TD and FV")

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
    plt.pause(1)
## OK: In most of the cases, the estimates are correct (in ~ 17 out of 20 replications)

raise KeyboardInterrupt


########
# 2023/10/12: Learn an actor-critic policy using neural networks (with the torch package)
# Learning happens with the ActorCriticNN learner which defines a loss of type `tensor` which can be minimized using the backward() method of torch Tensors
# IT WORKS!
from timeit import default_timer as timer
import os
import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt, cm
from matplotlib.ticker import MaxNLocator
from scipy.special import rel_entr

from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners import LearningCriterion, LearningTask
from Python.lib.agents.learners.policies import LeaActorCriticNN
from Python.lib.agents.policies import probabilistic

from Python.lib.environments.gridworlds import Direction2D
from Python.lib.estimators.fv import estimate_expected_reward
from Python.lib.utils.computing import compute_transition_matrices, compute_state_value_function_from_transition_matrix

from Python.lib.utils.basic import get_current_datetime_as_string, load_objects_from_pickle, save_objects_to_pickle
from Python.test.test_optimizers_discretetime import InputLayer, Test_EstPolicy_EnvGridworldsWithObstacles

# When saving results or reading previously saved results
resultsdir = "./RL-003-Classic/results"

#--- Auxiliary functions
KL_THRESHOLD = 0.005
policy_changed_from_previous_learning_step = lambda KL_distance, num_states: np.abs(KL_distance) / num_states > KL_THRESHOLD

def compute_true_state_value_function(env, policy, learning_task, learning_criterion, gamma=1.0):
    #*********************************** IMPORTANT **********************************#
    # This function may give a bad value of the true state value function for policies close to deterministic.
    # This would be most likely due to the ill-conditioning of the transition matrix P derived from the policy and the transition probabilities of the environment
    # when eigenvalues are computed.
    # For more details and possible fix see function computing.compute_state_value_function_from_transition_matrix().
    #*********************************** IMPORTANT **********************************#
    # TODO: (2024/05/07) Try to solve the above problem of instability in the calculation of the state value function
    """
    Computes the true state value function for the given policy applied on the given environment
    under the given learning task, learning criterion and discount factor gamma.

    It also computes the expected reward under stationarity for the CONTINUING learning task (with no discount applied of course).

    The state value function is stored in the environment object, so that it can be used as reference for comparing the estimated state value function.

    Arguments:
    env: EnvironmentDiscrete
        Environment with discrete states and discrete actions with ANY initial state distribution.
        Rewards can be anywhere.

    policy: policy object with method getPolicyForAction(a, s) defined, returning Prob(action | state)
        Policy object acting on a discrete-state / discrete-action environment.
        This could be of class e.g. random_walks.PolRandomWalkDiscrete, probabilistic.PolGenericDiscrete, PolNN.

    gamma: (opt) float in (0, 1]
        Discount factor for the observed rewards.
        default: 1.0

    Return: tuple
    Tuple containing the following 3 elements:
    - V_true: the true state value function for the given learning task, learning criterion and discount gamma.
    - expected_reward: the expected reward for the CONTINUING learning task.
    - mu: the stationary probability for the CONTINUING learning task.
    """
    P_epi, P_con, b_epi, b_con, g, mu = compute_transition_matrices(env, policy)
    P = P_con if learning_task == LearningTask.CONTINUING else P_epi
    b = b_con if learning_task == LearningTask.CONTINUING else b_epi
    bias = g if learning_criterion == LearningCriterion.AVERAGE else None
    V_true = compute_state_value_function_from_transition_matrix(P, b, bias=bias, gamma=gamma)
    env.setV(V_true)
    dict_proba_stationary = dict(zip(np.arange(len(mu)), mu))
    avg_reward_true = estimate_expected_reward(env, dict_proba_stationary)
    return V_true, avg_reward_true, mu
#--- Auxiliary functions

# Learning task and learning criterion are used by the constructor of the test class below
learning_task = LearningTask.CONTINUING
#learning_criterion = LearningCriterion.DISCOUNTED; gamma = 0.9
learning_criterion = LearningCriterion.AVERAGE; gamma = 1.0    # gamma could be < 1 in the average reward criterion in order to take the limit as gamma -> 1 as presented in Sutton, pag. 251/252.

seed = 1317
test_ac = Test_EstPolicy_EnvGridworldsWithObstacles()

problem_2d = True
exit_state_at_bottom = False

if problem_2d:
    # 2D labyrinth
    size_vertical = 3; size_horizontal = 4
    #size_vertical = 6; size_horizontal = 8
    size_vertical = 8; size_horizontal = 12
    #size_vertical = 9; size_horizontal = 13
    #size_vertical = 10; size_horizontal = 14
    #size_vertical = 10; size_horizontal = 30

    # Whether the active set in FV should be connected (in order to avoid isolation of the two activation states and reduce possible problems)
    # When this is the case, the connectedness of the active set is achieved by removing the left-most obstacle in the previous-to-bottom row.
    connected_active_set = True
    initial_policy = None #[0.4, 0.15, 0.05, 0.4] #None   # Random walk as initial policy when None
else:
    # 1D gridworld: the interesting dimension is the vertical dimension, with terminal state at the top (this is to unify the structure of 2D labyrinth and 1D gridworld)
    size_vertical = 21; size_horizontal = 1     # We choose a value like K = 20 or 40 in the M/M/1/K queue system
    initial_policy = [0.4, 0.0, 0.6, 0.0]       # We choose a probability of going down which is similar to mu/(lambda + mu) in the queue system where lambda/mu = 0.7, i.e. a value close to 1 / 1.7 = 0.588
env_shape = (size_vertical, size_horizontal)
nS = size_vertical * size_horizontal

# Environment's entry and exit states
entry_state = np.ravel_multi_index((size_vertical - 1, 0), env_shape)
exit_state = entry_state + env_shape[1] - 1 if exit_state_at_bottom else None   # None means that the EXIT state is set at the top-right of the labyrinth
entry_state_in_absorption_set = True   #False #True

# Presence of wind: direction and probability of deviation in that direction when moving
if problem_2d:
    wind_dict = None
    wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.6})
    wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.7})
    #wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.8})
else:
    # WIND is currently not allowed in 1D gridworlds
    wind_dict = None

# Obstacles and absorption set
if problem_2d:
    #-- 2D labyrinth
    # The path to the terminal state is just a corridor through the last row right and then the last column up
    # So at the upper left corner there is a rectangle that brings to nowhere
    rectangle_to_nowhere_width = size_horizontal - 2
    rectangle_to_nowhere_height = size_vertical - 2
    states_previous_to_last_row = np.ravel_multi_index([np.repeat(rectangle_to_nowhere_height, rectangle_to_nowhere_width), [y for y in range(1, rectangle_to_nowhere_width+1)]], env_shape)
    states_previous_to_last_column = np.ravel_multi_index([[x for x in range(0, rectangle_to_nowhere_height+1)], np.repeat(rectangle_to_nowhere_width, rectangle_to_nowhere_height+1)], env_shape)
    obstacles_set = set(np.concatenate([list(states_previous_to_last_row) + list(states_previous_to_last_column)]))

    if connected_active_set:
        obstacles_set = obstacles_set.difference({min(states_previous_to_last_row)})

    # The absorption set is a rectangular area at the upper left corner of the grid + (possibly) the lower left corner
    #lower_left_state = (size_vertical-1) * size_horizontal
    #absorption_set = set(np.concatenate([list(range(x*size_horizontal, x*size_horizontal + size_horizontal-2)) for x in range(size_vertical-2)]))
    # The absorption set is a rectangular area that touches the right and bottom walls of the big rectangular area that leads to nowhere
    left_margin = 0 #0 #int(rectangle_to_nowhere_width/2)
    top_margin = rectangle_to_nowhere_height
    absorption_set = set(np.concatenate([list(range(x * size_horizontal + left_margin, x * size_horizontal + rectangle_to_nowhere_width)) for x in range(0, top_margin)]))
else:
    #-- 1D gridworld
    obstacles_set = set()
    absorption_set = set(np.arange(np.prod(env_shape) * 2 // 3 + 1, env_shape[0] - 1))  # We choose an absorption set size J ~ K/3, like in the M/M/1/K queue system (recall that numbering of states here is the opposite than in M/M/1/K

# Add the environment's start state to the absorption set
# (ASSUMED TO BE AT THE LOWER LEFT OF THE LABYRINTH)
if entry_state_in_absorption_set:
    absorption_set.add(entry_state)

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

test_ac.setUpClass( shape=env_shape, exit_state=exit_state, wind_dict=wind_dict,
                    nn_input=InputLayer.ONEHOT, nn_hidden_layer_sizes=nn_hidden_layer_sizes, initial_policy=initial_policy,
                    # General learning parameters
                    learning_task=learning_task,
                    learning_criterion=learning_criterion,
                    alpha=1.0, gamma=gamma, lmbda=0.0,
                    alpha_min=0.1,
                    # Fleming-Viot parameters
                    # Small N and T are N=50, T=1000 for the 8x12 labyrinth with corridor
                    N=200, #50 #20 #100
                    T=1000, #1000, #3000,  # np.prod(env_shape) * 10  #100 #1000
                    obstacles_set=obstacles_set, absorption_set=absorption_set,
                    start_states_set={entry_state}, #{nS-1}, #None,
                    reset_method_value_functions=ResetMethod.ALLZEROS,
                    estimate_on_fixed_sample_size=True,
                    seed=seed, debug=False)
test_ac.setUp()
print(test_ac.policy_nn.nn_model)
test_ac.env2d._render()
print(f"Absorption set (1D):\n{absorption_set}")
print(f"Absorption set (2D):\n{[np.unravel_index(s, env_shape) for s in absorption_set]}")

# Average reward of random policy
policy_random = probabilistic.PolGenericDiscrete(test_ac.env2d, policy=dict(), policy_default=[0.25, 0.25, 0.25, 0.25])
V_true, avg_reward_true, mu = compute_true_state_value_function(test_ac.env2d, policy_random, learning_task, learning_criterion)
print(f"Average reward under RANDOM policy: {avg_reward_true}")
if initial_policy is not None:
    policy_initial = probabilistic.PolGenericDiscrete(test_ac.env2d, policy=dict(), policy_default=initial_policy)
    V_true, avg_reward_true, mu = compute_true_state_value_function(test_ac.env2d, policy_random, learning_task, learning_criterion)
    print(f"Average reward under INITIAL policy: {avg_reward_true}")

print(f"True state value function under initial policy:\n{test_ac.env2d.getV()}")

# Maximum average reward (to use as reference in information and plots)
# It is computed as the inverse of the shortest path from Start to Terminal in Manhattan-like movements,
# INCLUDING THE START STATE for continuing learning tasks (just think about it, we restart every time we reach the terminal state with reward = 0)
# EXCLUDING THE START STATE for episodic learning tasks (just think about it)
# *** This value is problem dependent and should be adjusted accordingly ***
if wind_dict is None:
    print("\nComputing MAX average in DETERMINISTIC environment...")
    if exit_state == entry_state + env_shape[1] - 1:
        # This is the case when the exit state is at the bottom-right of the labyrinth
        max_avg_reward_continuing = 1 / env_shape[1]          # In this case the start state counts! (we subtract 1 because the bottom-right state must NOT count twice!)
        max_avg_reward_episodic = 1 / (env_shape[1] - 1)      # In this case the start state does not count (we subtract 2 because, besides not counting twice the bottom-right state, the start state does not count in the episodic setting)
    else:
        # This is the case when the exit state is at the top-right of the labyrinth (the default)
        assert exit_state is None
        max_avg_reward_continuing = 1 / (np.sum(env_shape) - 1)   # In this case the start state counts! (we subtract 1 because the bottom-right state must NOT count twice!)
        max_avg_reward_episodic = 1 / (np.sum(env_shape) - 2)     # In this case the start state does not count (we subtract 2 because, besides not counting twice the bottom-right state, the start state does not count in the episodic setting)
else:
    # There is wind in the environment
    # => Compute the average reward using the transition matrix information of the environment (as the wind makes things more complicated)
    # together with the optimal policy used by the agent (which we know)
    print(f"\nComputing MAX average in STOCHASTIC environment with WIND: {wind_dict}...")
    if exit_state == entry_state + env_shape[1] - 1:
        policy_optimal = probabilistic.PolGenericDiscrete(test_ac.env2d, policy=dict(), policy_default=[0.0, 1.0, 0.0, 0.0])
        V_true, avg_reward_true, mu = compute_true_state_value_function(test_ac.env2d, policy_optimal, learning_task, learning_criterion)
        max_avg_reward_continuing = avg_reward_true
        # Compute the approximated episodic average reward (it is approximated because of the randomness generated by the wind.
        # In order to compute the exact episodic average reward, we need to think more.
        # Here, we use the formula that relates the continuing average reward with the episodic one in the deterministic environment case.
        max_avg_reward_episodic = avg_reward_true * env_shape[1] / (env_shape[1] - 1)
    else:
        # This is the case when the exit state is at the top-right of the labyrinth (the default)
        rightmost_states = [np.ravel_multi_index((r, env_shape[1]-1), env_shape) for r in np.arange(env_shape[0])]
        policy_optimal = probabilistic.PolGenericDiscrete(test_ac.env2d, policy=dict(zip(rightmost_states, [[1.0, 0.0, 0.0, 0.0]]*len(rightmost_states))), policy_default=[0.0, 1.0, 0.0, 0.0])
        V_true, avg_reward_true, mu = compute_true_state_value_function(test_ac.env2d, policy_optimal, learning_task, learning_criterion)
        max_avg_reward_continuing = avg_reward_true
        # Compute the approximated episodic average reward (it is approximated because of the randomness generated by the wind.
        # In order to compute the exact episodic average reward, we need to think more.
        # Here, we use the formula that relates the continuing average reward with the episodic one in the deterministic environment case.
        max_avg_reward_episodic = avg_reward_true * np.sum(env_shape) / (np.sum(env_shape) - 1)
print(f"MAX CONTINUING average reward: {max_avg_reward_continuing}")
print(f"MAX EPISODIC average reward (approximate when wind present): {max_avg_reward_episodic}")

# State to observe and e.g. plot Q values, etc
if exit_state == entry_state + env_shape[1] - 1:
    state_observe = np.ravel_multi_index((env_shape[0] - 1, env_shape[1] - 2), env_shape)
else:
    state_observe = np.ravel_multi_index((1, env_shape[1] - 1), env_shape)

# Learning methods: define characteristics to use when plotting results for each method
dict_colors = dict(); dict_linestyles = dict(); dict_legends = dict()
dict_colors['all_online'] = "darkred"; dict_linestyles['all_online'] = "solid"; dict_legends['all_online'] = "ALL Online"
dict_colors['values_td'] = "red"; dict_linestyles['values_td'] = "solid"; dict_legends['values_td'] = "TD"
# For a second trial of TDAC (which sometimes is useful --e.g. to compare "using the same budget as FV for every policy learning step" vs. "using an average budget at each policy learning step")
dict_colors['values_td2'] = "darkorange"; dict_linestyles['values_td2'] = "solid"; dict_legends['values_td2'] = "TD2"
dict_colors['values_tdl'] = "orange"; dict_linestyles['values_tdl'] = "dashed"; dict_legends['values_tdl'] = "TD(lambda)"
dict_colors['values_fv'] = "green"; dict_linestyles['values_fv'] = "solid"; dict_legends['values_fv'] = "FV"
# For a second trial of FVAC (which sometimes is useful --e.g. to compare "allowing infinite budget for FV" vs. "limited budget")
dict_colors['values_fv2'] = "cyan"; dict_linestyles['values_fv2'] = "solid"; dict_legends['values_fv2'] = "FV2"
dict_colors['values_fvos'] = "lightgreen"; dict_linestyles['values_fvos'] = "solid"; dict_legends['values_fvos'] = "FV OverSampling"

# Dictionaries to store the results for the different methods (for comparison purposes)
dict_loss = dict()
dict_R = dict()
dict_R_long = dict()
dict_V = dict()
dict_Q = dict()
dict_A = dict()
dict_state_counts = dict()
dict_nsteps = dict()
dict_KL = dict()
dict_alpha = dict()
dict_time_elapsed = dict()
# Statistics about the results across replications
dict_stats_R = dict()

# Number of replications to run on each method
nrep = 5 #10

# Learning method (of the value functions and the policy)
# Both value functions and policy are learned online using the same simulation
learning_method = "all_online"; simulator_value_functions = None
# Value functions are learned separately from the policy
# Policy learning can happen online or OFFLINE
learning_method = "values_td"; simulator_value_functions = test_ac.sim_td       # Normally TD(0), but could be TD(lambda)
#learning_method = "values_tdl"; simulator_value_functions = test_ac.sim_td     # TD(lambda)
#learning_method = "values_tda"; simulator_value_functions = test_ac.sim_tda    # Adaptive TD(lambda)
learning_method = "values_fv"; simulator_value_functions = test_ac.sim_fv
#learning_method = "values_fvos"; simulator_value_functions = test_ac.sim_fv     # FV used just as an oversampling method

learning_method_type = learning_method[:9]  # This makes e.g. "values_fvos" become "values_fv"

# FV learning parameters (which are used to define parameters of the other learners analyzed so that their comparison with FV is fair)
max_time_steps_fv_per_particle = 50 #100 #50                            # Max average number of steps allowed for each particle in the FV simulation
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

#-- Common learning parameters
# Parameters about policy learning (Actor)
n_learning_steps = 50 #200 #50 #100 #30
n_episodes_per_learning_step = 50   #100 #30   # This parameter is used as the number of episodes to run the policy learning process for and, if the learning task is EPISODIC, also as the number of episodes to run the simulators that estimate the value functions
max_time_steps_per_policy_learning_episode = test_ac.env2d.getNumStates() if problem_2d else 2*test_ac.env2d.getNumStates() #np.prod(env_shape) * 10 #max_time_steps_benchmark // n_episodes_per_learning_step   # Maximum number of steps per episode while LEARNING THE *POLICY* ONLINE (NOT used for the value functions (critic) learning)
policy_learning_mode = "online"     # Whether the policy is learned online or OFFLINE (only used when value functions are learned separately from the policy)
allow_deterministic_policy = True #False
use_average_reward_from_previous_step = True #learning_method_type == "values_fv" #False #True            # Under the AVERAGE reward crtierion, whether to use the average reward estimated from the previous policy learning step as correction of the value functions (whenever it is not 0), at least as an initial estimate
use_advantage = not (learning_method == "values_fvos") # Set this to True if we want to use the advantage function learned as the TD error, instead of using the advantage function as the difference between the estimated Q(s,a) and the estimated V(s) (where the average reward cancels out)
optimizer_learning_rate = 0.01 #0.01 #0.1
reset_value_functions_at_every_learning_step = False #(learning_method == "values_fv")     # Reset the value functions when learning with FV, o.w. the learning can become too unstable due to the oversampling of the states with high value... (or something like that)

# Parameters about value function learning (Critic)
alpha_initial = simulator_value_functions.getAgent().getLearner().getInitialLearningRate()
adjust_alpha_initial_by_learning_step = False; t_learn_min_to_adjust_alpha = 30 # based at 1 (regardless of the base value used for t_learn)
#max_time_steps_per_episode = test_ac.env2d.getNumStates()*10  # (2024/05/02) NO LONGER USED!  # This parameter is just set as a SAFEGUARD against being blocked in an episode at some state of which the agent could be liberated by restarting to a new episode (when this max number of steps is reached)
epsilon_random_action = 0.1 #0.1 #0.05 #0.0 #0.01
use_average_max_time_steps_in_td_learner = False #learning_method == "values_td2" #True #False
learning_steps_observe = [50, 90] #[2, 30, 48] #[2, 10, 11, 30, 31, 49, 50] #[7, 20, 30, 40]  # base at 1, regardless of the base value used for t_learn
verbose_period = max_time_steps_fv_for_all_particles // 10
plot = False         # Whether to plot the evolution of value function and average reward estimation
colormap = "seismic"  # "Reds"  # Colormap to use in the plot of the estimated state value function V(s)

save = True

time_start = timer()
datetime_start_str = get_current_datetime_as_string(format="filename")

# A few further parameters for the policy learning process
break_when_no_change = False    # Whether to stop the learning process when the average reward doesn't change from one step to the next
break_when_goal_reached = False  # Whether to stop the learning process when the average reward is close enough to the maximum average reward (by a relative tolerance of 0.1%)

# Initialize objects that will contain the results by learning step
state_counts_all = np.zeros((nrep, n_learning_steps, test_ac.env2d.getNumStates()))
V_all = np.zeros((nrep, n_learning_steps, test_ac.env2d.getNumStates()))
Q_all = np.zeros((nrep, n_learning_steps, test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions()))
A_all = np.zeros((nrep, n_learning_steps, test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions()))
R_all = np.nan * np.ones((nrep, n_learning_steps))       # Average reward (EPISODIC learning task)
R_long_all = np.nan * np.ones((nrep, n_learning_steps))  # Long-run Average reward (CONTINUING learning task). It does NOT converge to the same value as the episodic average reward because there is one more reward value per episode!! (namely the reward going from the terminal state to the start state)
loss_all = np.nan * np.ones((nrep, n_learning_steps))
nsteps_all = np.nan * np.ones((nrep, n_learning_steps))  # Number of value function time steps run per every policy learning step
KL_all = np.nan * np.ones((nrep, n_learning_steps))      # K-L divergence between two consecutive policies
alpha_all = alpha_initial * np.ones((nrep, n_learning_steps))   # Initial alpha used at each policy learning step
time_elapsed_all = np.nan * np.ones(nrep)   # Execution time for each replication
if learning_method_type == "values_fv":
    # Define the object where we will store the number of steps used by the FV learner of value functions at each policy learning step
    # so that we use the same number for the TD learner of value functions when the TDAC policy learning process is run afterwards.
    # I.e. this assumes that FVAC is run BEFORE TDAC!
    max_time_steps_benchmark_all = np.nan * np.ones((nrep, n_learning_steps))

    # Store the definition of the start state distribution for the E(T_A) simulation for the FV simulation, so that we can reset them to their original ones
    # when starting a new replication (as these distributions may be updated by the process)
    probas_stationary_start_state_et = test_ac.agent_nn_fv.getLearner().getProbasStationaryStartStateET()
    probas_stationary_start_state_fv = test_ac.agent_nn_fv.getLearner().getProbasStationaryStartStateFV()
if learning_method == "all_online":
    # Online Actor-Critic policy learner with TD as value functions learner and value functions learning happens at the same time as policy learning
    learner_ac = LeaActorCriticNN(test_ac.env2d, simulator_value_functions.getAgent().getPolicy(), simulator_value_functions.getAgent().getLearner(),
                                  allow_deterministic_policy=allow_deterministic_policy,
                                  reset_value_functions=reset_value_functions_at_every_learning_step, initial_policy=initial_policy, optimizer_learning_rate=optimizer_learning_rate, seed=test_ac.seed, debug=True)
else:
    # Value functions (Critic) are learned separately from the application of the policy and the policy (Actor) may be learned OFFLINE or online
    # IMPORTANT: We pass the policy of the agent stored in the value functions simulator as policy for the Actor-Critic learner so that when the Actor-Critic learner
    # updates the policy, the policy of the agent stored in the value functions simulator is ALSO updated. This is crucial for using the updated policy
    # when learning the value functions at the next policy learning step.
    learner_ac = LeaActorCriticNN(test_ac.env2d, simulator_value_functions.getAgent().getPolicy(), simulator_value_functions.getAgent().getLearner(),
                                  allow_deterministic_policy=allow_deterministic_policy,
                                  reset_value_functions=reset_value_functions_at_every_learning_step, initial_policy=initial_policy, optimizer_learning_rate=optimizer_learning_rate, seed=test_ac.seed, debug=True)

seed_base = test_ac.seed
for rep in range(nrep):
    seed_rep = seed_base + rep*1317
    print(f"\n->>>>>>>>>>>\nRunning replication {rep+1} of {nrep} (seed={seed_rep})... @{format(get_current_datetime_as_string())}")

    # Reset the policy actor and the critic every time a new replication starts
    print("Resetting the policy learner and the critic (if any)...")
    learner_ac.reset(reset_value_functions=True, reset_policy=True, initial_policy=initial_policy)

    if simulator_value_functions is not None:
        # Whenever there is a critic, do a couple of resets
        # (recall that no critic is defined for the ALL-online learning method)
        _learner_value_functions_for_critic = simulator_value_functions.getAgent().getLearner()

        # Reset the value functions and average reward that may have been obtained at the end of the previous replication
        # Note that these may not be reset at the start of the first policy learning step because execution parameter use_average_reward_from_previous_step may be set to True.
        _learner_value_functions_for_critic.reset(reset_episode=True, reset_value_functions=True, reset_average_reward=True)

        # Reset the initial alpha of the TD learner (just in case)
        _learner_value_functions_for_critic.setInitialLearningRate(alpha_initial)

        # For FV learners, reset the start distribution for the E(T_A) simulation and for FV to their original values defined in the learners defined in test_ac
        if learning_method_type == "values_fv":
            _learner_value_functions_for_critic.setProbasStationaryStartStateET(probas_stationary_start_state_et)
            _learner_value_functions_for_critic.setProbasStationaryStartStateFV(probas_stationary_start_state_fv)

    time_start_rep = timer()
    if learning_method == "all_online":
        for t_learn in range(n_learning_steps):
            print(f"\n\n*** Running learning step {t_learn+1} of {n_learning_steps} (AVERAGE REWARD at previous step = {R_all[rep, max(0, t_learn-1)]}) of MAX={max_avg_reward_episodic} using {nsteps_all[rep, max(0, t_learn-1)]} time steps for Critic estimation)... (seed={seed_learn}) @{get_current_datetime_as_string()}")
            print("Learning the VALUE FUNCTIONS and POLICY simultaneously...")
            loss_all[rep, t_learn] = learner_ac.learn(n_episodes_per_learning_step, start_state=entry_state, max_time_steps_per_episode=max_time_steps_per_policy_learning_episode, prob_include_in_train=1.0) # prob_include_in_train=0.5)
                ## Note that we make sure that the start state when learning the policy is the entrance state to the labyrinth, `entry_state`, because the environment may have defined
                ## a different initial state distribution (e.g. a random start in the states outside the absorption set used by the FV learner.

            state_counts_all[rep, t_learn, :] = learner_ac.learner_value_functions.getStateCounts()
            V_all[rep, t_learn, :] = learner_ac.learner_value_functions.getV().getValues()
            Q_all[rep, t_learn, :, :] = learner_ac.learner_value_functions.getQ().getValues().reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())
            A_all[rep, t_learn, :, :] = learner_ac.learner_value_functions.getA().getValues().reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())
            R_all[rep, t_learn] = learner_ac.learner_value_functions.getAverageReward()
            # Could also retrieve the average reward from the Actor-Critic learner (if store_trajectory_history=False in the constructor of the value functions learner)
            #R_all[rep, t_learn] = learner_ac.average_reward_over_episodes
    else:
        # Keep track of the policy learned so that we can analyze how much it changes after each learning step w.r.t. the previous learning step
        policy_prev = None
        for t_learn in range(n_learning_steps):
            # Pass a different seed (for the simulator) for each learning step... o.w. we will be using the same seed for them at every learning step!!
            seed_learn = seed_rep + t_learn
            print(f"\n\n*** Running learning step {t_learn+1} of {n_learning_steps} (AVERAGE REWARD at previous step = {R_all[rep, max(0, t_learn-1)]} of MAX={max_avg_reward_episodic})... (seed={seed_learn}) @{get_current_datetime_as_string()}")
            reset_value_functions_at_this_step = reset_value_functions_at_every_learning_step if t_learn > 0 else True  # ALWAYS RESET THE VALUE FUNCTIONS WHEN IT'S THE VERY FIRST LEARNING STEP (becaue we don't want to keep histroy from a earlier learning process on the same learner!)
            # Update the initial learning rate for the value functions at each learning step to a smaller value than the previous learning step
            # SHOULD WE SET IT TO THE AVERAGE LEARNING RATE FROM THE PREVIOUS LEARNING STEP?? (so that we start off where we left at the last learning moment)
            alpha_initial_at_current_learning_step = alpha_initial / (t_learn + 1)

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
            policy = learner_ac.getPolicy().get_policy_values()
            KL_distance = np.sum(rel_entr(policy, policy_prev)) if t_learn > 0 else 0.0
            print(f"K-L distance with previous policy: {KL_distance} (should be < {KL_THRESHOLD} to reduce initial alpha)")
            KL_all[rep, t_learn] = KL_distance
            policy_prev = policy.copy()

            if adjust_alpha_initial_by_learning_step:
                # Update the initial learning rate alpha ONLY when:
                # - the learning step is larger than or equal to a minimum
                # - the policy did NOT change significantly from the previous learning step
                #   (because in that case we can consider the current estimate to be a reliable estimator of the value functions)
                if t_learn + 1 >= t_learn_min_to_adjust_alpha:
                    if policy_changed_from_previous_learning_step(KL_distance, test_ac.env2d.getNumStates()): #np.abs(loss_all[rep, t_learn-1]) > 1.0
                        simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial)
                    else:
                        simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial / 10)
                alpha_all[rep, t_learn] = simulator_value_functions.getAgent().getLearner().getInitialLearningRate()
            print(
                f"*** INITIAL learning rate alpha = {simulator_value_functions.getAgent().getLearner().getInitialLearningRate()} " +
                (f"(adjustment happens starting at learning step (base 1) >= {t_learn_min_to_adjust_alpha}) " if adjust_alpha_initial_by_learning_step else "***\n"))

            print(f"Learning the CRITIC (at the current policy) using {learning_method.upper()}...")
            # Learn the value functions using the FV simulator
            if learning_method_type == "values_fv":
                # Reset to None the start state distribution for the E(T_A) excursion at the very first learning step so that
                # we do NOT carry over whatever this distribution was at the end of the previous replication or at the end of the previous execution of the learning process
                # using this same learner.
                if t_learn == 0:
                    simulator_value_functions.getAgent().getLearner().setProbasStationaryStartStateET(None)
                V, Q, A, state_counts, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, n_events_et, n_events_fv = \
                    simulator_value_functions.run(t_learn=t_learn,
                                                  max_time_steps_fv=max_time_steps_fv_for_all_particles,
                                                  min_num_cycles_for_expectations=0,
                                                      ## Note: We set the minimum number of cycles for the estimation of E(T_A) to 0 because we do NOT need
                                                      ## the estimation of the average reward to learn the optimal policy, as it cancels out in the advantage function Q(s,a) - V(s)!!
                                                  use_average_reward_stored_in_learner=use_average_reward_from_previous_step,
                                                  reset_value_functions=reset_value_functions_at_this_step,
                                                  plot=plot if t_learn+1 in learning_steps_observe else False, colormap=colormap,
                                                  epsilon_random_action=epsilon_random_action,
                                                  seed=seed_learn, verbose=False, verbose_period=verbose_period)
                #average_reward = simulator_value_functions.getAgent().getLearner().getAverageReward()  # This average reward should not be used because it is inflated by the FV process that visits the states with rewards more often
                average_reward = expected_reward
                nsteps_all[rep, t_learn] = n_events_et + n_events_fv
                max_time_steps_benchmark_all[rep, t_learn] = n_events_et + n_events_fv  # Number of steps to use when running TDAC at the respective learning step
                print(f"Learning step {t_learn+1}: Learning of value functions COMPLETED using {learning_method} method on {nsteps_all[rep, t_learn]} time steps")
            else:
                # TD learners
                if 'max_time_steps_benchmark_all' in locals() and max_time_steps_benchmark_all[rep, t_learn] != np.nan:
                    # The FV learner was run before running this TD learner
                    if use_average_max_time_steps_in_td_learner:
                        _max_time_steps = int( np.mean(max_time_steps_benchmark_all[rep, :]) )
                    else:
                        # => Use the number of steps used by the FV learner at the current policy learning step (t_learn) as maximum number of steps to allow for the TD learner
                        _max_time_steps = max_time_steps_benchmark_all[rep, t_learn]
                else:
                    # When max_time_steps_benchmark_all is not defined, it means that the number of steps to run the TD learner for is calculated above
                    # and may not be exactly equal to the number of steps the FV learner took at each policy learning step.
                    _max_time_steps = max_time_steps_benchmark
                print(f"*** TD learning will use {_max_time_steps} simulation steps. ***")
                if False:
                    # Use this when the discrete.Simulator.run() calls the _run_single() method to run the simulation, which is based on the existence of episodes
                    V, Q, A, state_counts, _, _, learning_info = \
                        simulator_value_functions.run(nepisodes=n_episodes_per_learning_step,
                                                      t_learn=t_learn,
                                                      max_time_steps=max_time_steps_benchmark,
                                                      max_time_steps_per_episode=max_time_steps_per_episode,  # max_time_steps_benchmark // n_episodes_per_learning_step,
                                                      reset_value_functions=reset_value_functions_at_this_step,
                                                      seed=seed_learn,
                                                      state_observe=state_observe, compute_rmse=True if t_learn+1 in learning_steps_observe else False,
                                                      epsilon_random_action=epsilon_random_action,
                                                      plot=plot if t_learn+1 in learning_steps_observe else False, colormap=colormap,
                                                      verbose=True, verbose_period=verbose_period)
                else:
                    # Use this when the discrete.Simulator.run() calls the _run_single_continuing_task() method to run the simulation (where there are no episodes)
                    V, Q, A, state_counts, _, _, learning_info = \
                        simulator_value_functions.run(t_learn=t_learn,
                                                      max_time_steps=_max_time_steps,
                                                      estimated_average_reward=simulator_value_functions.getAgent().getLearner().getAverageReward() if use_average_reward_from_previous_step else None,
                                                      reset_value_functions=reset_value_functions_at_this_step,
                                                      seed=seed_learn,
                                                      state_observe=state_observe,
                                                      compute_rmse=True if t_learn+1 in learning_steps_observe else False,
                                                      plot=plot if t_learn+1 in learning_steps_observe else False, colormap=colormap,
                                                      epsilon_random_action=epsilon_random_action,
                                                      verbose=True, verbose_period=verbose_period)
                average_reward = simulator_value_functions.getAgent().getLearner().getAverageReward()
                nsteps_all[rep, t_learn] = learning_info['nsteps']
                print(f"Learning step {t_learn+1}: Learning of value functions COMPLETED using {learning_method} method on {nsteps_all[rep, t_learn]} time steps")

            state_counts_all[rep, t_learn, :] = state_counts
            V_all[rep, t_learn, :] = V
            Q_all[rep, t_learn, :, :] = Q.reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())
            A_all[rep, t_learn, :, :] = A.reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())

            print(f"Learning the POLICY {policy_learning_mode.upper()} using estimated {use_advantage and 'ADVANTAGE A(s,a) values' or 'ACTION Q(s,a) values'} "
                  f"on {n_episodes_per_learning_step} episodes starting at state s={entry_state}, using MAX {max_time_steps_per_policy_learning_episode} steps per episode...")
            # Policy learning
            if policy_learning_mode == "online":
                # ONLINE with critic provided by the action values learned above using the TD simulator
                loss_all[rep, t_learn] = learner_ac.learn(n_episodes_per_learning_step, start_state=entry_state, max_time_steps_per_episode=max_time_steps_per_policy_learning_episode, prob_include_in_train=1.0,
                                                         use_advantage=use_advantage,
                                                         advantage_values=A,
                                                         action_values=Q)       # This parameter is not used when use_advantage=False
                    ## Note that we make sure that the start state when learning the policy is the entrance state to the labyrinth, `entry_state`,
                    ## because the environment may have defined a different initial state distribution, which is used during the learning of the value functions,
                    ## for instance, any randomly selected state outside the absorption set A used by the FV learner.
                    ## Note that the learned value functions are passed as critic to the Actor-Critic policy learner via the `action_values` parameter.
                R_all[rep, t_learn] = learner_ac.average_reward_over_episodes
                R_long_all[rep, t_learn] = average_reward  # This is the average reward estimated by the value functions learner used above
            else:
                # OFFLINE learner where ALL states and actions are swept and the loss computed on all of them using the state distribution as weights
                loss_all[rep, t_learn] = learner_ac.learn_offline_from_estimated_value_functions(V, A, Q, state_counts, use_advantage=use_advantage)
                R_all[rep, t_learn] = average_reward

            # Check if we need to stop learning because the average reward didn't change a bit
            if  break_when_no_change and t_learn > 0 and R_all[rep, t_learn] - R_all[rep, t_learn-1] == 0.0 or \
                break_when_goal_reached and np.isclose(R_all[rep, t_learn], max_avg_reward_episodic, rtol=0.001):
                print(f"*** Policy learning process stops at learning step t_learn+1={t_learn+1} because the average reward didn't change a bit from previous learning step! ***")
                break
    time_elapsed_rep = timer() - time_start_rep
    print(f"<<<<<<<<<< FINISHED replication {rep+1} of {nrep}... (@{format(get_current_datetime_as_string())}" + ", took {:.1f} min)".format(time_elapsed_rep / 60))
    time_elapsed_all[rep] = time_elapsed_rep

time_end = timer()
time_elapsed = time_end - time_start
print("{} learning process took {:.1f} minutes ({:.1f} hours)".format(learning_method.upper(), time_elapsed / 60, time_elapsed / 3600))


############# Store the measures that we would like to compare
dict_loss[learning_method] = loss_all.copy()
dict_R[learning_method] = R_all.copy()
dict_R_long[learning_method] = R_long_all.copy()
dict_V[learning_method] = V_all.copy()
dict_Q[learning_method] = Q_all.copy()
dict_A[learning_method] = A_all.copy()
dict_state_counts[learning_method] = state_counts_all.copy()
dict_nsteps[learning_method] = nsteps_all.copy()
dict_KL[learning_method] = KL_all.copy()
dict_alpha[learning_method] = alpha_all.copy()
dict_time_elapsed[learning_method] = time_elapsed_all.copy()
############# Store the measures that we would like to compare


# Plot loss and average reward for the currently analyzed learner
print("\nPlotting...")
ax_loss = plt.figure().subplots(1,1)
ax_loss.plot(range(1, n_learning_steps+1), loss_all[rep, :n_learning_steps], marker='.', color="red")
ax_loss.plot(range(1, n_learning_steps+1), alpha_all[rep, :n_learning_steps], '--', color="cyan")
ax_loss.set_xlabel("Learning step")
ax_loss.set_ylabel("Loss")
ax_loss.axhline(0, color="red", linewidth=1, linestyle='dashed')
ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_loss.legend(["Loss", "alpha0"], loc='upper left')
ax_R = ax_loss.twinx()
ax_R.plot(range(1, n_learning_steps+1), R_all[rep, :n_learning_steps], marker='.', color="green")
ax_R.axhline(max_avg_reward_episodic, color="green", linewidth=1)
ax_R.plot(range(1, n_learning_steps+1), R_long_all[rep, :n_learning_steps], marker='.', color="greenyellow")
ax_R.axhline(max_avg_reward_continuing, color="greenyellow", linewidth=1)
ax_R.set_ylabel("Average reward")
ax_R.plot(range(1, n_learning_steps+1), KL_all[rep, :n_learning_steps], color="blue", linewidth=1)
ax_R.axhline(KL_THRESHOLD, color="blue", linestyle="dashed")
ax_R.axhline(0, color="green", linewidth=1, linestyle='dashed')
ax_R.legend(["Average reward (episodic)", "Max. average reward (episodic)",
            "Long-run average reward estimated by value functions learner", "Max. average reward (continuing)",
            "K-L divergence with previous policy", "K-L threshold for reduced alpha0", "K-L divergence between consecutive policies"],
            loc="upper right")
plt.title(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={simulator_value_functions.getAgent().getLearner().gamma}) - Labyrinth {env_shape}"
          f"\nN={test_ac.agent_nn_fv.getLearner().getNumParticles()}, T={test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
          f"\nEvolution of the LOSS (left, red) and Average Reward (right, green) with the learning step")



#-- Plot the value functions for the state next to the terminal state
# ONLY VALID WHEN THE EXIT STATE IS AT THE TOP RIGHT OF THE LABYRINTH
marker = ''
Q_all_baseline = Q_all[rep, :n_learning_steps, :, :] - np.tile(V_all[rep, :n_learning_steps, :].T, (test_ac.env2d.getNumActions(), 1, 1)).T
ax_Q, ax_Q_baseline = plt.figure().subplots(1,2)
ax_Q.plot(range(1, n_learning_steps + 1), V_all[rep, :n_learning_steps, state_observe], marker=marker, color="black")
ax_Q.plot(range(1, n_learning_steps + 1), Q_all[rep, :n_learning_steps, state_observe, :], marker=marker)
ax_Q.legend(["V(s)"] + ["Q(s," + str(a) + ")" for a in range(Q_all.shape[2])], loc='upper left')
ax_Q_baseline.plot(range(1, n_learning_steps + 1), V_all[rep, :n_learning_steps, state_observe] - V_all[rep, :n_learning_steps, state_observe], marker=marker, color="white") # We plot this constant value 0 so that the legend is correct
ax_Q_baseline.plot(range(1, n_learning_steps + 1), Q_all_baseline[:n_learning_steps, state_observe, :], marker=marker)
ax_Q_baseline.legend(["V(s) - V(s)"] + ["Q(s," + str(a) + ") - V(s)" for a in range(Q_all.shape[2])], loc='upper left')

# Optimum Q-values (for the optimum deterministic policy)
# This assumes that there is a reward of 1 at the terminal state which is one step away
if learning_criterion == LearningCriterion.AVERAGE:
    assert learning_task == LearningTask.CONTINUING
    svalue  = 0.5 * (1.0 - max_avg_reward_continuing)     # Differential state value V(s) under the optimal policy (since the policy tells the agent to always go up, the agent receives reward 1.0, which is corrected (subtracted) by the max (because we are following the OPTIMAL policy) average reward; the 0.5 factor is explained by the calculations on my Mas de Canelles notebook sheet)
    qvalue0 = 0.5 * (1.0 - max_avg_reward_continuing)     # Differential optimal action value Q(s,a) of going up (a=0) when we start at s = state_observe, one cell away from the terminal state, which gives reward 1.0 (we subtract the max average reward because we are following the OPTIMAL policy; the 0.5 factor is explained by the calculations on my Mas de Canelles notebook sheet)
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
plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={simulator_value_functions.getAgent().getLearner().gamma}) - Labyrinth {env_shape}"
             f"\nN={test_ac.agent_nn_fv.getLearner().getNumParticles()}, T={test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
             f"\nQ(s,a) and V(s) for state previous to the terminal state under the optimal policy, i.e. s={state_observe}\nMax average reward (continuing) = {max_avg_reward_continuing}")


# Same plot for all states
axes = plt.figure().subplots(test_ac.env2d.shape[0], test_ac.env2d.shape[1])
first_learning_step = 0 #n_learning_steps * 3 // 4  #0
y2max = int(round(np.max(state_counts_all)*1.1)) # For a common Y2-axis showing the state counts
min_V, max_V = np.min(V_all), np.max(V_all)      # For a common Y-axis showing the value functions
min_Q, max_Q = np.min(Q_all), np.max(Q_all)      # For a common Y-axis showing the value functions
Q_all_baseline = Q_all[rep, first_learning_step:n_learning_steps, :, :] - np.tile(V_all[rep, first_learning_step:n_learning_steps, :].T, (test_ac.env2d.getNumActions(), 1, 1)).T
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
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), V_all[rep, first_learning_step:n_learning_steps, i] - V_all[rep, first_learning_step:n_learning_steps, i], marker=marker, color="black")  # We plot this so that the legend is fine
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), Q_all_baseline[first_learning_step:n_learning_steps, i, :], marker=marker)
    else:
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), V_all[rep, first_learning_step:n_learning_steps, i], marker=marker, color="black")
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), Q_all[rep, first_learning_step:n_learning_steps, i, :], marker=marker)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(ylim)

    # State counts on the right axis
    ax2 = ax.twinx()
    ax2.plot(np.arange(1+first_learning_step, n_learning_steps+1), state_counts_all[rep, first_learning_step:n_learning_steps, i], color="violet", linewidth=1)
    ax2.set_ylim((0, y2max))
    y2ticks = [0, int(min(state_counts_all[rep, first_learning_step:n_learning_steps, i])), int(max(state_counts_all[rep, first_learning_step:n_learning_steps, i])), int(np.max(state_counts_all))]
    ax2.set_yticks(y2ticks)
    ax2.yaxis.set_ticklabels(y2ticks, fontsize=7)
# Only show the x and y labels on the bottom-right plot (to avoid making the plot too cloggy)
ax.set_xlabel("Learning step")
ax2.set_ylabel("State count")
ax.legend(["V(s)"] + ["Q(s," + str(a) + ")" for a in range(Q_all.shape[2])], loc='upper left')
ax2.legend(["State count"], loc='upper right')
plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion - Labyrinth {env_shape}"
             f"\nN={test_ac.agent_nn_fv.getLearner().getNumParticles()}, T={test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
             f"\nEvolution of the value functions V(s) and Q(s,a) with the learning step by state\nMaximum average reward (continuing): {max_avg_reward_continuing}")


# Plot the ADVANTAGE function
axes = plt.figure().subplots(test_ac.env2d.shape[0], test_ac.env2d.shape[1])
first_learning_step = 0 #n_learning_steps * 3 // 4  #0
y2max = int(round(np.max(state_counts_all)*1.1)) # For a common Y2-axis showing the state counts
min_A, max_A = np.min(A_all), np.max(A_all)      # For a common Y-axis showing the value functions
ymin, ymax = min_A, max_A

#ylim = (ymin, ymax)     # Use this for common Y-axis limits
ylim = (None, None)     # Use this for unequal Y-axis limits
marker = ''
for i, ax in enumerate(axes.reshape(-1)):
    # Value functions on the left axis
    ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), A_all[rep, first_learning_step:n_learning_steps, i, :], marker=marker)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(ylim)

    # State counts on the right axis
    ax2 = ax.twinx()
    ax2.plot(np.arange(1+first_learning_step, n_learning_steps+1), state_counts_all[rep, first_learning_step:n_learning_steps, i], color="violet", linewidth=1)
    ax2.set_ylim((0, y2max))
    y2ticks = [0, int(min(state_counts_all[rep, first_learning_step:n_learning_steps, i])), int(max(state_counts_all[rep, first_learning_step:n_learning_steps, i])), int(np.max(state_counts_all))]
    ax2.set_yticks(y2ticks)
    ax2.yaxis.set_ticklabels(y2ticks, fontsize=7)
# Only show the x and y labels on the bottom-right plot (to avoid making the plot too cloggy)
ax.set_xlabel("Learning step")
ax2.set_ylabel("State count")
ax.legend(["A(s," + str(a) + ")" for a in range(A_all.shape[2])], loc='upper left')
ax2.legend(["State count"], loc='upper right')
plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion - Labyrinth {env_shape}"
             f"\nN={test_ac.agent_nn_fv.getLearner().getNumParticles()}, T={test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
             f"\nEvolution of the Advantage function A(s,a) with the learning step by state\nMaximum average reward (continuing): {max_avg_reward_continuing}")


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
proba_actions_toplot = np.nan*np.ones((3, 3))
if problem_2d:
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            state_1d = np.ravel_multi_index((i, j), gridworld_shape)
            print("")
            for action in range(env.getNumActions()):
                print(f"Computing policy Pr(a={action}|s={(i,j)})...", end= " ")
                idx_2d = (0, 1) if action == 0 else (1, 2) if action == 1 else (2, 1) if action == 2 else (1, 0)
                proba_actions_toplot[idx_2d] = policy.getPolicyForAction(action, state_1d)
                print("p = {:.3f}".format(proba_actions_toplot[idx_2d]))
            img = axes[i,j].imshow(proba_actions_toplot, cmap=colormap, vmin=0, vmax=1)
else:
    for i in range(len(axes)):
        state = i
        for action in range(env.getNumActions()):
            print(f"Computing policy Pr(a={action}|s={state})...", end=" ")
            idx_2d = (0, 1) if action == 0 else (1, 2) if action == 1 else (2, 1) if action == 2 else (1, 0)
            proba_actions_toplot[idx_2d] = policy.getPolicyForAction(action, state)
            print("p = {:.3f}".format(proba_actions_toplot[idx_2d]))
        img = axes[i].imshow(proba_actions_toplot, cmap=colormap, vmin=0, vmax=1)
plt.colorbar(img, ax=axes)  # This adds a colorbar to the right of the FIGURE. However, the mapping from colors to values is taken from the last generated image! (which is ok because all images have the same range of values.
                            # Otherwise see answer by user10121139 in https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
plt.suptitle(f"{learning_method.upper()}\nPolicy at each state")

print("{} learning process took {:.1f} minutes ({:.1f} hours)".format(learning_method.upper(), time_elapsed / 60, time_elapsed / 3600))


############## SAVE
if save:
    _size = f"{env_shape[0]}x{env_shape[1]}"
    _learning_method = str.replace(learning_method, "values_", "")
    _filename = f"ActorCritic_labyrinth_{_size}_{datetime_start_str}_{_learning_method.upper()}AC.pkl"
    objects_to_save = ["dict_loss", "dict_R", "dict_R_long", "dict_R_long_true", "dict_V", "dict_Q", "dict_A", "dict_state_counts", "dict_nsteps", "dict_KL", "dict_alpha", "dict_time_elapsed"]
    if "max_time_steps_benchmark_all" in locals():
        objects_to_save += ["max_time_steps_benchmark_all"]
    else:
        objects_to_save += ["max_time_steps_benchmark"]
    _filepath = os.path.join(resultsdir, _filename)
    # Save the original object names plus those with the prefix so that, when loading the data back,
    # we have also the objects without the prefix which are need to generate the plots right-away.
    # HOWEVER, when reading the saved data, make sure that any object with the original name (without the suffix) has been copied to another object
    # (e.g. loss_all_td could be a copy of loss_all before reading the data previously saved for the FV learning method)
    object_names_to_save = objects_to_save + [f"{obj_name}" for obj_name in objects_to_save]
    save_objects_to_pickle(object_names_to_save, _filepath, locals())
    print(f"Results for ALL Actor-Critic methods: {[str.replace(meth, 'values_', '').upper() for meth in dict_loss.keys()]} saved to '{_filepath}'")
############## SAVE ALL RESULTS TOGETHER



############## LOAD RESULTS
resultsdir = "./RL-003-Classic/results"
_datetime = "20240428_092427" #"20240428_205959" #"20240421_140558" #"20240405_094608" #"20240322_095848" #"20240219_230301" #"20240219_142206"          # Use format yyymmdd_hhmmss
_size_str = "6x8" #"21x1" #"10x14" #"3x4" #"10x14"
_learning_method = "ALL" #"td" #"fv"
_suffix = "_LabyrinthWithConnectedActiveSetAndFinishAtBottomRightLostPolicy_TDAC,FVAC,N=20,T=500,LimitedTime5xFVWorksTDFails" #"_LabyrinthWithConnectedActiveSetAndFinishAtBottomRightRandomPolicy_TDAC,FVAC,N=20,T=100,LimitedTime5xFVSlightlyBetterThanTD" #"_N=50,T=5000,InfiniteTimeFVisBetterLessVariance" #"_FV&TD_FVIsWorse" #""
_N = 50
_T = 5000
if _learning_method == "ALL":
    _filename = f"{prefix}{_datetime}_{_learning_method.upper()}{_suffix}.pkl"
else:
    _filename = f"{prefix}{_datetime}_{_learning_method.upper()}AC_N={_N}.pkl"
_filepath = os.path.join(resultsdir, _filename)
object_names = load_objects_from_pickle(_filepath, globals())
print(f"The following objects were loaded from '{_filepath}':\n{object_names}")
# The following reward values are used in the ALLTOGETHER plotting below
env_shape = (int(_size[:_size.index("x")]), int(_size[_size.index("x")+1:]))
_methods = list(dict_loss.keys())
max_avg_reward_continuing = 1 / (np.sum(env_shape) - 1)
max_avg_reward_episodic = 1 / (np.sum(env_shape) - 2)
nrep = len(dict_loss[_methods[0]])
n_learning_steps = len(dict_loss[_methods[0]][0])
############## LOAD RESULTS


#-- ALTOGETHER PLOT
# Plot all average rewards together (to compare methods)
# Color names are listed here: https://matplotlib.org/stable/gallery/color/named_colors.html
# We normalize the average reward plots so that they converge to 1.0 (easier interpretation of the plot)
if "n_learning_steps" not in locals():
    n_learning_steps = len(dict_loss[list(dict_loss.keys())[0]][0])
if "policy_learning_mode" not in locals():
    policy_learning_mode = "online"
ax_loss, ax_R = plt.figure().subplots(1,2)
legend = []
for meth in dict_loss.keys():
    ax_loss.plot(np.arange(1, n_learning_steps + 1), dict_loss[meth][nrep-1, :n_learning_steps], '-', marker='.', color=dict_colors[meth])
    legend += [f"{dict_legends[meth]} (average reward)"]
    ax_R.plot(np.arange(1, n_learning_steps+1), dict_R[meth][nrep-1, :n_learning_steps] / max_avg_reward_episodic, '-', marker='.', color=dict_colors[meth])
ax_loss.set_xlabel("Learning step")
ax_loss.set_ylabel("Loss")
ax_loss.axhline(0, color="gray")
ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_loss.set_title(f"Evolution of the LOSS with the learning step by learning method (last replication {nrep})")
ax_loss.legend(legend)
ax_R.set_xlabel("Learning step")
ax_R.set_ylabel("Average reward (normalized by the MAX average reward = {:.2g})".format(max_avg_reward_episodic))
ax_R.axhline(1, color="gray")
ax_R.axhline(0, color="gray")
ax_R.set_title(f"Evolution of the NORMALIZED Average Reward with the learning step by learning method (last replication {nrep})")
ax_R.legend(legend, loc="center left")
plt.suptitle(f"ALL LEARNING METHODS: Labyrinth {env_shape} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={gamma})"
             f"\nN={test_ac.agent_nn_fv.getLearner().getNumParticles()}, T={test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()}, MAX budget={max_time_steps_benchmark} steps per policy learning step")

# If we want to add the ratio between number of steps used by two methods compared
if "values_td" in dict_nsteps.keys() and "values_fv" in dict_nsteps.keys():
    df_ratio_nsamples = pd.DataFrame({'td': dict_nsteps['values_td'][nrep-1,:], 'fv': dict_nsteps['values_fv'][nrep-1,:], 'ratio_fv_td': dict_nsteps['values_fv'][nrep-1,:] / dict_nsteps['values_td'][nrep-1,:]})
    ax_R_nsamples = ax_R.twinx()
    ax_R_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'][:n_learning_steps], color="blue", linewidth=0.5)
    ax_R_nsamples.axhline(1.0, color="blue", linewidth=0.5, linestyle="dashed")
    ax_R_nsamples.set_ylim((ax_R.get_ylim()[0], None))
    ax_R_nsamples.legend(["Sample size ratio (FV/TD)", "Reference line showing equal sample size ratio"], loc="center right")

# Plot results on several replications
if nrep > 1:
    dict_stats_R = dict.fromkeys(dict_loss.keys())
    ax = plt.figure().subplots(1,1)
    lines = []
    legend = []
    _xshift = -0.1 # shift on the X axis to avoid overlap of vertical error bars
    for meth in dict_loss.keys():
        _xshift += 0.1
        # Compute distribution of values to plot
        dict_stats_R[meth] = dict()
        dict_stats_R[meth]['min'], \
        dict_stats_R[meth]['max'], \
        dict_stats_R[meth]['median'], \
        dict_stats_R[meth]['mean'], \
        dict_stats_R[meth]['std'], \
        dict_stats_R[meth]['n'] = dict_R[meth].min(axis=0), dict_R[meth].max(axis=0), np.median(dict_R[meth], axis=0), dict_R[meth].mean(axis=0), dict_R[meth].std(axis=0), len(dict_R[meth])
        # Percentiles (if needed)
        # percentiles_low = [10, 25]
        # percentiles_upp = [90, 75]
        # alphas = [0.10, 0.15, 0.20]
        # percentiles = pd.DataFrame({'replication': np.array([np.repeat(r, n_learning_steps) for r in range(1, nrep+1)]).reshape(-1),
        #                             'step': np.array([np.repeat(s, R) for s in range(n_learning_steps)]).T.reshape(-1),
        #                             'R': R_all.reshape(-1)}, columns=['replication', 'state', 'V'])[['state', 'V']] \
        #     .groupby('state') \
        #     .agg(['count', 'mean', 'min', 'median', 'max', 'std'] + [percentile(p) for p in percentiles_low] + [percentile(p) for p in percentiles_upp])

        # Plot
        _xvalues = np.arange(1, n_learning_steps+1) + _xshift
        #line = ax.plot(_xvalues, dict_stats_R[meth]['mean'] / max_avg_reward_episodic, color=dict_colors[meth], linestyle=dict_linestyles[meth], linewidth=2)[0]
        line = ax.errorbar(_xvalues, dict_stats_R[meth]['mean'] / max_avg_reward_episodic, yerr=dict_stats_R[meth]['std'] / np.sqrt(dict_stats_R[meth]['n']) / max_avg_reward_episodic, color=dict_colors[meth], linestyle=dict_linestyles[meth], linewidth=2, marker=".", markersize=12)[0]
        lines += [line]
        legend += [f"{dict_legends[meth]} (average +/- SE)"]
        line = ax.plot(_xvalues, dict_stats_R[meth]['median'] / max_avg_reward_episodic, color=dict_colors[meth], linestyle="dashed", linewidth=2, marker="x", markersize=12)[0]
        lines += [line]
        legend += [f"{dict_legends[meth]} (median)"]
        line = ax.plot(_xvalues, dict_stats_R[meth]['max'] / max_avg_reward_episodic, color=dict_colors[meth], linestyle="dashed")[0]
        lines += [line]
        legend += [f"{dict_legends[meth]} (min/max)"]
        ax.plot(_xvalues, dict_stats_R[meth]['min'] / max_avg_reward_episodic, color=dict_colors[meth], linestyle="dashed")
        ax.fill_between(_xvalues,
                        dict_stats_R[meth]['max'] / max_avg_reward_episodic,
                        dict_stats_R[meth]['min'] / max_avg_reward_episodic,
                        color=dict_colors[meth],
                        alpha=0.1)
    ax.axhline(1, color="gray")
    ax.legend(lines, legend, loc="center left")
    ax.set_ylim((-0.01, 1.01))
    ax.set_xlabel("Learning step")
    ax.set_ylabel("Average reward (normalized by the MAX average reward = {:.2g})".format(max_avg_reward_episodic))
    plt.suptitle(f"ALL LEARNING METHODS: Labyrinth {env_shape} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={gamma})\n{nrep} replications"
                 f"\nN={'test_ac' in locals() and test_ac.agent_nn_fv.getLearner().getNumParticles() or _N}, T={'test_ac' in locals() and test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation() or _T}, MAX budget={'max_time_steps_benchmark' in locals() and max_time_steps_benchmark or 'N/A'} steps per policy learning step")

    # Plot of number of samples ratios between FV learnings and TD learning
    legend_nsamples = []
    if "values_td" in dict_nsteps.keys() and "values_fv" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.mean(dict_nsteps['values_td'], axis=0),
                                          'fv': np.mean(dict_nsteps['values_fv'], axis=0)})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'][:n_learning_steps], color="blue", linewidth=0.5)
        ref_line = ax_nsamples.axhline(1.0, color="blue", linewidth=0.5, linestyle="dashed")
        legend_nsamples += ["Sample size ratio (FV/TD)", "Reference line showing equal sample size ratio"]
    if "values_td2" in dict_nsteps.keys() and "values_fv" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.mean(dict_nsteps['values_td2'], axis=0),
                                          'fv': np.mean(dict_nsteps['values_fv'], axis=0)})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        if "ax_nsamples" not in locals():
            ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'][:n_learning_steps], color="cyan", linewidth=0.5)
        legend_nsamples += ["Sample size ratio (FV/TD2)"]
        if "ref_line" not in locals():
            ref_line = ax_nsamples.axhline(1.0, color="cyan", linewidth=0.5, linestyle="dashed")
            legend_nsamples += ["Reference line showing equal sample size ratio"]
    if "values_td" in dict_nsteps.keys() and "values_fvos" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.mean(dict_nsteps['values_td'], axis=0),
                                          'fv': np.mean(dict_nsteps['values_fvos'], axis=0)})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        if "ax_nsamples" not in locals():
            ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps + 1), df_ratio_nsamples['ratio_fv_td'][:n_learning_steps], color="magenta", linewidth=0.5)
        legend_nsamples += ["Sample size ratio (FVOS/TD)"]
        if "ref_line" not in locals():
            ax_nsamples.axhline(1.0 - 1E-6, color="magenta", linewidth=0.5, linestyle="dashed")
            legend_nsamples += ["Reference line showing equal sample size ratio"]
        ax_nsamples.set_ylim((ax.get_ylim()[0], None))

    if "ax_nsamples" in locals():
        ax_nsamples.set_ylim((ax.get_ylim()[0], None))
        ax_nsamples.set_ylabel("Average sample Ratio FV/TD across replications")
        ax_nsamples.legend(legend_nsamples, loc="lower right")
#-- ALTOGETHER PLOT


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
