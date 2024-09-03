# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:02:17 2024 (from tests.py in order to isolate the tests on 1D environments and be able to run 2D environments faster by selecting what to run from the top of the file!)

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

# We can estimate the stationary distribution for the CONTINUING learning task by computing P^t for t sufficiently large
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
    # as the distribution at each state DEPENDS ON THE START AND END STATES. However, we can compute the "stationary" distribution as how often the Markov chain is at each state
    # under the periodicity situation. E.g. if the environment is a 1D environment with 5 states {0, 1, ..., 4}, and the Markov chain starts at state s=0,
    # and the deterministic policy indicates to always go RIGHT, then the period is d = 5 (as, recall that when the Markov chain reaches s=4, it goes back to s=0, so 1 every
    # 5 steps the Markov chain se retrouve at s=0). Under periodicity, the power matrix of P is periodic, i.e. P^t = P^(t+d) for all t >= 0. Thus, mu'*P^d = mu'*P^0 = mu'
    # and hence mu (the "stationary" probability) is a left-eigenvector of P^d. Note that P^0 = I (the identity matrix), hence P^d = I as well,
    # hence each column of I is an eigenvector mu. There are nS such eigenvectors, as the eigenvalue 1 of I has multiplicity nS since I is of size nS x nS.
    # We can define an "average" eigenvector by computing the average of all those nS eigenvectors, which are the state vectors of the system starting at each of the nS states.
    evector = (P_con**0)[0, :]   # This means: we consider that the start state is state s=0 (because at time 0 we choose the vector [1, 0, ..., 0] as the initial distribution of the states)
    for k in range(1, nS):
        evector += (P_con**k)[0, :] # This means, to the first eigenvector associated to starting the Markov chain at [1, 0, ..., 0], we add the eigenvector associated to starting the Markov chain at state s=k, i.e. where the state of the Markov chain is all zeros except 1 at position k.
    mu = evector / nS   # Here we compute the average eigenvector over all nS eigenvectors of P^d, which is equal to P^0 = I, the identity matrix of size nS x nS
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
max_time_steps_fv_per_particle = 100 #50 #10 #50         # Max average number of steps allowed for each particle in the FV simulation
max_time_steps_fv_for_expectation = T
max_time_steps_fv_for_all_particles = max_time_steps_fv_per_particle * N
# Traditional method learning parameters
# They are set for a fair comparison with FV learning
# The maximum time steps to be observed in the benchmark methods is set to the sum of:
# - the max number of time steps allowed to estimate E(T_A)
# - the max number of time steps allowed over all the FV particles
M = 1   # Number of normal particles created during the FV simulation to explore using the underlying Markov process
        # This value MUST coincide with n_normal_max variable defined in Simulator._run_simulation_fv_discounted()
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
    # WARNING: The trajectory history recovered here most likely includes ONLY the exploration by the "normal" particle (this is the case when running the FV simulation with _run_simulation_fv_discounted())
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

