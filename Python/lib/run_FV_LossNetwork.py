# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:17:48 2023

@author: Daniel Mastropietro
@description:   Runs the FV estimation of the blocking probability in a Loss Network system receiving multi-class jobs.
                Can be used to run the process in batch mode.
                In batch mode execution, parameters should be defined using equal sign `=` ONLY when the parameter is str or numeric, but NOT when the
                parameter is expecting a boolean value or something else that requires special parsing (e.g. a "list", which is the case with a couple
                of parameters in this script --see example below).
                In case of boolean arguments, they are EITHER included among the script arguments or NOT, which defines whether their value is set to True
                (if INCLUDED and option action="store_true" when defining the parameter by the script) or False
                (if not included or INCLUDED and option action="store_false" when defining the parameter by the script). See parse_input_parameters() below.
                Ex:
                --analysis_type=T
                --burnin_time_steps=10
                --save_results
                --values_parameter_analyzed "[64, 128, 256]"    --> This parameter is expecting a list which is parsed by calling
                                                                    the `convert_str_argument_to_list_of_type()` function and thus must receive a str type.
                For more info see the documentation of the optparse module used to parse input arguments: https://docs.python.org/3/library/optparse.html
"""

if __name__ == "__main__":
    # Only run this when running the script, o.w. it may give an error when importing functions if setup.py is not found
    import runpy
    runpy.run_path('../../setup.py')

import os
import sys
import copy
from warnings import warn
from timeit import default_timer as timer
from enum import Enum, unique
import optparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Python.lib.agents.learners import LearnerTypes
from Python.lib.agents.learners.continuing.mc import LeaMC
from Python.lib.agents.learners.continuing.fv import LeaFV
from Python.lib.agents.policies import PolicyTypes
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep
from Python.lib.agents.queues import AgeQueue

from Python.lib.environments.queues import rewardOnJobRejection_ByClass

from Python.lib.simulators import SetOfStates
from Python.lib.simulators.queues import define_queue_environment_and_agent, estimate_blocking_fv, estimate_blocking_mc, \
    estimate_expected_reward, SurvivalProbabilityEstimation

from Python.lib.utils.basic import aggregation_bygroups, convert_str_argument_to_list_of_type, get_current_datetime_as_string, \
    is_scalar, measure_exec_time, set_pandas_options, reset_pandas_options
from Python.lib.utils.computing import compute_blocking_probability_knapsack, \
    compute_blocking_probability_knapsack_from_probabilities_and_job_arrival_rates, \
    compute_stationary_probability_knapsack_when_blocking_by_class

from Python.lib.run_FV import closeLogFile, createLogFileHandleAndResultsFileNames, plot_results_fv_mc, save_dataframes


#------------------- Functions --------------------
def define_absorption_and_activation_sets(Js):
    """
    Defines the absorption and the activation set for a multi-class loss network system

    Arguments:
    Js: list or array-like
        Size of the absorption set A of the Fleming-Viot process in each dimension, where each dimension represents a different arriving job class.

    Return: Tuple
    Duple with the following two elements:
    - absorption_set: SetOfStates object defining the absorption set, defined as the hyper-rectangle [0, Js[0] - 1] x [0, Js[1] - 1] x ... x [0, Js[I-1] - 1]
    where I is the number of job classes of the loss network.
    - activation_set: SetOfStates object defining the activation set defined as the set of states from where the Markov chain representing the system
    can transition to the multidimensional absorption set A just described in ONE step.
    Note that, if e.g. Js = [2, 5, 3], the activation set contains those states that are at a distance of 1 in EXACTLY ONE dimension.
    E.g. the state (3, 4, 2) is part of the activation set but (3, 5, 2) (where the first TWO dimensions are one value larger than the original tuple)
    is NOT part of the activation set, because we cannot transition in ONE step from (3, 5, 2) to (2, 4, 2), which is the closest state to (3, 5, 2)
    that is part of the absorption set.
    """
    absorption_set = SetOfStates(set_boundaries=tuple([J - 1 for J in Js]))
    activation_set = SetOfStates(set_boundaries=tuple(Js))
    # TODO: (2024/07/16) Define the activation_set using the following line because it is much more clear
    # In this way, the activation_set is the CORRECT activation set, o.w. in the currently implemented approach, the `activation_set` set includes also the absorption set and
    # the selection of a state from the ACTUAL activation set is done ad-hoc by e.g. the SetOfStates.random_choice() method or by choosing a state that satisfies the condition
    # of having exactly one dimension at the boundary (see such implementations in run_simulation_fv() in simulators/queues.py, when particles (`envs`) are initialized).
    #activation_set = SetOfStates(states=SetOfStates(set_boundaries=tuple(Js)).getStates(exactly_one_dimension_at_boundary=True))
    return absorption_set, activation_set


@measure_exec_time
def run_mc_estimation_loss_network( env_queue, Ks, Js, K, T,
                                    burnin_time_steps, min_num_cycles_for_expectations,
                                    seed=1717):
    """
    Runs the Monte-Carlo estimation of the blocking probability of a loss network system (usually used for testing)

    This function essentially sets up the loss-network environment, the agent, the admission control policy (accept/reject) and learners
    and calls estimate_blocking_mc().

    Arguments:
    env_queue: EnvQueueLossNetworkWithJobClasses
        The queue environment that handles the loss network to simulate.

    Ks: list of int
        Blocking sizes for each job class accepted by the loss network system.

    Js: list of int
        Smallest number of jobs of each class where the "equivalent" FV particle is still active (i.e. not absorbed).

    K: int
        The server's total capacity.

    T: int
        Number of arrivals at which the simulation stops.

    seed: int
        Seed to use for the pseudo-random number generation.
    """
    print("\n--> Running Monte-Carlo estimation on a multi-job-class loss network system...")

    assert len(Ks) == len(Js), "The lengths of Ks and Js coincide: Ks={}, Js={}".format(Ks, Js)

    # Queue environments to use
    # Set the total server's capacity
    env_queue.setCapacity(K)

    # Agent interacting with the environment (which defines whether to accept or reject a job)
    # Define the agent acting on the queue environment
    # The agent blocks an incoming job of class c when the number of jobs of the class being served is K(c)
    # where K(c) is defined by parameter theta(c) of the parameterized policy used for that job class,
    # where theta(c) = K(c) - 1 which guarantees a deterministic blocking at K(c).
    thetas = [K - 1 for K in Ks]
    # We define a separate 1-D acceptance policy for each job class
    # All policies have the same structure, what changes is their real-valued theta parameter
    policies = dict({PolicyTypes.ACCEPT: [PolQueueTwoActionsLinearStep(env_queue, thetas[c])
                                          for c in range(env_queue.getNumJobClasses())],
                     PolicyTypes.ASSIGN: None})
    learners = dict({LearnerTypes.V: LeaMC(env_queue, gamma=1.0),
                     LearnerTypes.Q: None,
                     LearnerTypes.P: None})
    agent_accept_reject = AgeQueue(env_queue, policies, learners)

    # Simulation parameters
    absorption_set, activation_set = define_absorption_and_activation_sets(Js)
    dict_params_simul = dict({'absorption_set': absorption_set,
                              'activation_set': activation_set,
                              'T': T,  # number of arrivals at which the simulation stops
                              'burnin_time_steps': burnin_time_steps,
                              'min_num_cycles_for_expectations': min_num_cycles_for_expectations,
                              'seed': seed})
    dict_params_info = dict()

    # Run the simulation!
    return estimate_blocking_mc(env_queue, agent_accept_reject,
                                dict_params_simul, dict_params_info)


@measure_exec_time
def run_fv_estimation_loss_network( env_queue, Ks, Js, K, N, T,
                                    burnin_time_steps, min_num_cycles_for_expectations, method_survival_probability_estimation,
                                    seed=1717):
    """
    Runs the Fleming-Viot estimation of the blocking probability of a loss network system (usually used for testing)

    This function essentially sets up the loss-network environment, the agent, the admission control policy (accept/reject) and learners
    and calls estimate_blocking_fv().

    Arguments:
    env_queue: EnvQueueLossNetworkWithJobClasses
        The queue environment that handles the loss network to simulate.

    Ks: list of int
        Blocking sizes for each job class accepted by the loss network system.

    Js: list of int
        Smallest number of jobs of each class where the FV particle is still active (i.e. not absorbed).

    K: int
        The server's total capacity.

    N: int
        Number of particles of the Fleming-Viot system.

    T: int
        Number of arrivals at which the simulation stops.

    seed: int
        Seed to use for the pseudo-random number generation.
    """
    print("\n--> Running Fleming-Viot estimation on a multi-job-class loss network system...")

    assert len(Ks) == len(Js), "The lengths of Ks and Js coincide: Ks={}, Js={}".format(Ks, Js)

    # Queue environments to use as FV particles
    # Set the total server's capacity
    env_queue.setCapacity(K)
    envs_queue = [env_queue if i == 0 else copy.deepcopy(env_queue) for i in range(N)]

    # Agent interacting with the environment (which defines whether to accept or reject a job)
    # Define the agent acting on the queue environment
    # The agent blocks an incoming job of class c when the number of jobs of the class being served is K(c)
    # where K(c) is defined by parameter theta(c) of the parameterized policy used for that job class,
    # where theta(c) = K(c) - 1 which guarantees a deterministic blocking at K(c).
    thetas = [K - 1 for K in Ks]
    # We define a separate 1-D acceptance policy for each job class
    # All policies have the same structure, what changes is their real-valued theta parameter
    policies = dict({PolicyTypes.ACCEPT: [PolQueueTwoActionsLinearStep(env_queue, thetas[c])
                                          for c in range(env_queue.getNumJobClasses())],
                     PolicyTypes.ASSIGN: None})
    learners = dict({LearnerTypes.V: LeaFV(env_queue, gamma=1.0),
                     LearnerTypes.Q: None,
                     LearnerTypes.P: None})
    agent_accept_reject = AgeQueue(env_queue, policies, learners)

    # Simulation parameters
    absorption_set, activation_set = define_absorption_and_activation_sets(Js)
    dict_params_simul = dict({'absorption_set': absorption_set,
                              'activation_set': activation_set,
                              'T': T,  # number of arrivals at which the simulation stops
                              'burnin_time_steps': burnin_time_steps,
                              'min_num_cycles_for_expectations': min_num_cycles_for_expectations,
                              'method_survival_probability_estimation': method_survival_probability_estimation,
                              'seed': seed})
    dict_params_info = dict()

    # Run the simulation!
    return estimate_blocking_fv(envs_queue, agent_accept_reject,
                                dict_params_simul, dict_params_info,
                                probas_stationary=None)


def analyze_convergence(capacity=10, job_class_rates=[0.7], service_rates=[1.0],
                        blocking_sizes=[5], blocking_costs=None,
                        N=[20, 40, 80],
                        T=500,
                        J=None,
                        J_factor=[0.3],
                        use_stationary_probability_for_start_state=False,
                        burnin_time_steps=20, min_num_cycles_for_expectations=5,
                        method_proba_surv=SurvivalProbabilityEstimation.FROM_N_PARTICLES,
                        replications=5,
                        run_mc=False,
                        seed=1717):
    """
    Analyze convergence of the FV estimation of the blocking probability as the number of particles N OR
    the number of arrival times (T) increases.

    Arguments:
    blocking_sizes: (opt) list
        List of blocking sizes for the different job classes.
        default: [5]

    blocking_costs: (opt) list
        List of blocking closts for the different job classes.
        Blocking costs should be non-negative values.
        When none, a blocking cost of 1 is assumed for each class.
        default: None

    N: (opt) int or list or numpy array
        List giving the number of particles to consider in each simulation.
        default: [20, 40, 80]

    T: (opt) int or list or numpy array
        Number of discrete *time steps* to run the queue to estimate the blocking probability in Monte-Carlo
        or the number of *arrival events* to estimate E(T_A) for the FV estimator.
        If not scalar, either N should be a scalar or be a list with just one element, o.w. it should have
        the same number of elements as the number of particles that are tried in this simulation.
        default: 500

    J: (opt) list
        List of J values to consider for the different job classes that define the absorption size for each class
        in the Fleming-Viot simulation.
        If given, it has priority over J_factor.
        default: None

    J_factor: (opt) list
        List of J factors to consider for the different job classes that define the absorption size for each class
        in the Fleming-Viot simulation.
        This value is not used if J is given.
        default: [0.3]

    burnin_time_steps: (opt) int
        Number of burn-in time steps to consider before a return cycle and a reabsorption cycle qualifies for
        the estimation of expectations.
        default: 20

    min_num_cycles_for_expectations: (opt) int
        Minimum number of observed cycles in order to estimate expectations (instead of setting their values to NaN).
        default: 5

    method_proba_surv: (opt) SurvivalProbabilityEstimation
        Method used to estimate the survival probability P(T>t), whether "from M cycles observed by a single particle"
        or "from the N particles used in the FV simulation".
        default: SurvivalProbabilityEstimation.FROM_N_PARTICLES
    """
    #---------------------------- Auxiliary functions --------------------------
    def compute_blocking_state_coverages(probas_stationary, probas_stationary_true, expected_costs):
        """
        Computes the coverage of blocking states by the simulation, computing quantities related to entropy
        and cost-entropy.

        probas_stationary: dict
            Dictionary indexed by state with the estimated stationary probability of blocking states.

        probas_stationary_true: dict
            Dicionary indexed by state with the true stationary probability of states (blocking and non-blocking).
            But all states in probas_stationary should be present in this dictionary.

        expected_costs: dict
            Dictionary indexed by state with the expected blocking cost of states (blocking and non-blocking).
            But all states in probas_stationary should be present in this dictionary.
        """
        blocking_states = sorted(probas_stationary.keys())
        dict_prob_blocking_states = dict([(x, p) for x, p in probas_stationary_true.items() if x in blocking_states])
        dict_info_blocking_states = dict([(x, -np.log10(p)) for x, p in probas_stationary_true.items() if x in blocking_states])
        dict_expcosts_blocking_states = dict([(x, c) for x, c in expected_costs.items() if x in blocking_states])

        # Data for the coverage computation
        blocking_states_observed = [x for x in blocking_states if probas_stationary[x] > 0]
        prob_blocking_states_observed = [p for x, p in dict_prob_blocking_states.items() if x in blocking_states_observed]
        info_blocking_states_observed = [logp for x, logp in dict_info_blocking_states.items() if x in blocking_states_observed]

        # Coverages
        n_blocking_states_observed = len(blocking_states_observed)
        coverage_prob_blocking_states = np.sum(prob_blocking_states_observed) / sum(dict_prob_blocking_states.values())    # CANNOT np.sum() on the output of dict.values()!!!
        coverage_info_blocking_states = np.sum(info_blocking_states_observed) / sum(dict_info_blocking_states.values())    # CANNOT np.sum() on the output of dict.values()!!!
        # Entropy coverage
        coverage_entropy_blocking_states = np.sum([probas_stationary[x] * dict_info_blocking_states[x] for x in blocking_states_observed]) / \
                                           np.sum([probas_stationary_true[x] * dict_info_blocking_states[x] for x in blocking_states])
        coverage_costentropy_blocking_states = np.sum([expected_costs[x] * probas_stationary[x] * dict_info_blocking_states[x] for x in blocking_states_observed]) / \
                                               np.sum([expected_costs[x] * probas_stationary_true[x] * dict_info_blocking_states[x] for x in blocking_states])

        # Estimated and True distribution of cost-entropy contribution
        contribution_costentropy_blocking_states = np.array([expected_costs[x] * probas_stationary[x] * dict_info_blocking_states[x] for x in blocking_states]) / \
                                               np.sum([expected_costs[x] * probas_stationary_true[x] * dict_info_blocking_states[x] for x in blocking_states])
        contribution_true_costentropy_blocking_states = np.array([expected_costs[x] * probas_stationary_true[x] * dict_info_blocking_states[x] for x in blocking_states]) / \
                                               np.sum([expected_costs[x] * probas_stationary_true[x] * dict_info_blocking_states[x] for x in blocking_states])

        return coverage_prob_blocking_states, coverage_info_blocking_states, coverage_entropy_blocking_states, coverage_costentropy_blocking_states, \
                    contribution_costentropy_blocking_states, contribution_true_costentropy_blocking_states, n_blocking_states_observed
    #---------------------------- Auxiliary functions --------------------------

    nservers = len(service_rates)

    #--- System setup
    dict_params = dict({'environment': {'queue_system': "loss-network",
                                        'capacity': capacity,
                                        'nservers': nservers,
                                        'job_class_rates': job_class_rates,  # [0.8, 0.7]
                                        'service_rates': service_rates,  # [1.0, 1.0, 1.0]
                                        'policy_assignment_probabilities': None, # [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]] )
                                        'reward_func': rewardOnJobRejection_ByClass,
                                        'reward_func_params': None if blocking_costs is None else {'reward_at_rejection': [-c for c in blocking_costs]},
                                        'rewards_accept_by_job_class': None
                                        },
                        'policy': { 'parameterized_policy': [PolQueueTwoActionsLinearStep for _ in blocking_sizes],
                                    'theta': [float(K-1) for K in blocking_sizes]  # This means that blocking is deterministic at blocking_sizes and otherwise there is acceptance.
                                   },
                        'learners': {'V': {'learner': LeaFV,
                                           'params': {'gamma': 1}
                                           },
                                     'Q': {'learner': None,
                                           'params': {}},
                                     'P': {'learner': None,
                                           'params': {}}
                                     },
                        'agent': {'agent': AgeQueue}
                        })
    env_queue, rhos, agent = define_queue_environment_and_agent(dict_params)

    # The test of the Fleming-Viot convergence can be focused on either the convergence in terms of increasing
    # number of particles (N), or in terms of an increasing number of arrival events (T), or both.
    # For example, if an analysis on incrcarring number of particles is done, then we can:
    # - Set K to a small value (e.g. K=5)
    # - Consider increasing values of N (e.g. N=[200, 400, 800])
    # - Set a large simulation time (e.g. T=500)
    # We should check that the error between the estimated and true blocking probability decreases as 1/sqrt(N)
    # (according to Propostion 2.1 in Matt's draft, although this proposition applies to the estimation of Phi(t,K)
    # NOT to the estimation of the blocking probability... In fact, we prove in our paper sent and accepted at EWRL-2022)
    # that the blocking probability converges to the true blocking probability as 1/sqrt(M) + 1/sqrt(N) where M is the
    # number of cycles observed and used for the estimation of E(T_A), the denominator of the blocking probability.
    # This is why also an convergence analysis on increasing number of arrival events (T) is sensible.

    #--------------------------- Parse input parameters -----------------------
    if J is None:
        Js = [max(1, int( round(JF * K) )) for JF, K in zip(J_factor, blocking_sizes)]
    else:
        Js = J
    absorption_set, activation_set = define_absorption_and_activation_sets(Js)

    if is_scalar(N):
        if is_scalar(T):
            # There is only one value of N and T to try
            N_values = [N]
        else:
            # The analysis is done for different T values
            # => Replicate the single N value as many times as the number of T values to try
            N_values = [N] * len(T)
    else:
        # The analysis is done for different N values
        N_values = list(N)

    if is_scalar(T):
        T_values = [T] * len(N_values)
    else:
        # The analysis is done for different T values
        T_values = list(T)

    # Check if either of the input parameters is a list of length 1, in which case we just replicate the single value
    # as many times as values in the other parameter
    if len(N_values) == 1 and len(T_values) > 1:
        N_values = [N_values[0]] * len(T_values)
    if len(T_values) == 1 and len(N_values) > 1:
        T_values = [T_values[0]] * len(N_values)

    # Check
    if len(T_values) != len(N_values):
        raise ValueError("When both parameters N and T are lists they must have the same length:\nN={}\nT={}" \
                         .format(N, T))

    # Info parameters
    dict_params_info = {'plot': True, 'log': False}
    #--------------------------- Parse input parameters -----------------------

    # Modify pandas display options so that data frames are not truncated
    pandas_options = set_pandas_options()

    df_results = pd.DataFrame(columns=[ # System characteristics
                                        'capacity',
                                        'blocking_costs',
                                        'lambdas',
                                        'rhos',
                                        # Simulation parameters
                                        'J',
                                        'N',
                                        'T',
                                        'burnin_time_steps',
                                        'min_n_cycles',
                                        'method_proba_surv',
                                        # Replication
                                        'replication',
                                        'seed',
                                        # Monte-Carlo results
                                        'Probas(MC)',        # Estimated stationary probabilities for each blocking state
                                        'ProbasRatio(MC)',   # Estimated stationary probabilities for each blocking state / True stationary probability
                                        'Pr(MC)',            # Estimated blocking probability (it doesn't take into account the cost of blocking each job class)
                                        'ExpCost(MC)',       # Estimated expected blocking cost for the system
                                        'Time(MC)',          # Last continuous time value observed in the MC simulation
                                        'EMC(T)',            # Estimated expected return time E(T) used in the MC probability estimator
                                        '#Events(MC)',
                                        '#Cycles(MC)',       # Number of return cycles on which E(T) is estimated
                                        '#BlockingStates(MC)',# Number of blocking states observed during the simulation
                                        'Coverage(MC)',      # Proportion of blocking states coverage (where estimated stationary p(x) > 0)
                                        'CoverageProb(MC)',  # Probability-weighted proportion of blocking states coverage (where estimated stationary p(x) > 0, weighted by p)
                                        'CoverageInfo(MC)',  # Information-weighted proportion of blocking states coverage (where estimated stationary p(x) > 0, weighted by -log10(p), so the states with smaller p receive larger weight)
                                        'CoverageEntropy(MC)',      # Entropy coverage (-p*log10(p))
                                        'CoverageCostEntropy(MC)',  # Cost-Entropy coverage (-c*p*log(p))
                                        'ContributionCostEntropy(MC)',# Contribution to the cost entropy by each blocking state (i.e. c(x) p(x) log(p(x)) / sum(of above over all blocking states)
                                        # Fleming-Viot results
                                        'Probas(FV)',        # Estimated stationary probabilities for each blocking state
                                        'ProbasRatio(FV)',   # Estimated stationary probabilities for each blocking state / True stationary probability
                                        'Pr(FV)',            # Estimated blocking probability (it doesn't take into account the cost of blocking each job class)
                                        'ExpCost(FV)',       # Estimated expected blocking cost for the system
                                        'Time(FV)',          # Last continuous time value observed in the FV simulation
                                        'E(T)',              # Estimated expected re-absorption time E(T_A), denominator of the FV probability estimator
                                        '#Cycles(E(T))',
                                        'MaxSurvTime',       # Max survival time observed when estimating S(t) = P(T>t)
                                        '#Events(ET)',
                                        '#Events(FV-Only)',
                                        '#Events(FV)',
                                        '#Samples(S(t))',
                                        '#BlockingStates(FV)',# Number of blocking states observed during the simulation
                                        'Coverage(FV)',      # Proportion of blocking states coverage (where estimated stationary p(x) > 0)
                                        'CoverageProb(FV)',  # Probability-weighted proportion of blocking states coverage (where estimated stationary p(x) > 0, weighted by p)
                                        'CoverageInfo(FV)',  # Information-weighted proportion of blocking states coverage (where estimated stationary p(x) > 0, weighted by -log10(p), so the states with smaller p receive larger weight)
                                        'CoverageEntropy(FV)',      # Entropy coverage (-p*log10(p))
                                        'CoverageCostEntropy(FV)',  # Cost-Entropy coverage (-c*p*log(p))
                                        'ContributionCostEntropy(FV)',# Contribution to the cost entropy by each blocking state (i.e. c(x) p(x) log(p(x)) / sum(of above over all blocking states)
                                        # True probability
                                        'BlockingStates',    # Blocking states
                                        'Probas(True)',      # True stationary probabilities for each blocking state
                                        'Pr(True)',
                                        'Costs',             # Costs c(x) associated to each state x (prior to multiplying by p(x))
                                        'ExpectedCosts',     # Expected blocking cost for each blocking state: p(x) c(x)
                                        'ContributionCostEntropy',# Contribution of each expected cost to the cost-based entropy
                                        'ExpCost',           # Expected blocking cost (one single value associated to the cost of blocking on any possible state by the arrival of any possible job)
                                        'RelEntropy',        # Entropy of the blocking system relative to its maximum possible entropy
                                        'RelCostEntropy',    # Entropy of the blocking cost relative to its maximum possible entropy
                                        # Execution times
                                        'exec_time_mc(sec)',
                                        'exec_time_fv(sec)',
                                        'exec_time(sec)'])
    #ncases = int( np.log(nparticles_max / nparticles_min) / np.log(1 + nparticles_step_prop)) + 1
    ncases = len(N_values)

    # Compute the true blocking probability that is used to analyze the estimator quality
    print("Computing stationary probability of blocking states and TRUE blocking probability for capacity={}, rhos={}...".format(capacity, rhos))
    proba_blocking_true = compute_blocking_probability_knapsack(capacity, rhos, job_class_rates, blocking_sizes=blocking_sizes)

    # True stationary probability (this is used to:
    # - choose the start state in the FV simulation when use_stationary_probability_for_start_state = True
    # - compute the probability-weighted coverage of blocking states:
    effective_capacity, x, dist = compute_stationary_probability_knapsack_when_blocking_by_class(capacity, rhos, blocking_sizes)
    probas_stationary_true = dict([(tuple(xx), dd) for xx, dd in zip(x, dist)])
    assert proba_blocking_true == compute_blocking_probability_knapsack_from_probabilities_and_job_arrival_rates(probas_stationary_true, effective_capacity, job_class_rates, blocking_sizes)

    # Blocking states
    _set_of_valid_states = SetOfStates(set_boundaries=blocking_sizes)
    blocking_states = sorted(_set_of_valid_states.getStates(at_least_one_dimension_at_boundary=True).intersection(probas_stationary_true.keys()) \
                             .union([x for x in probas_stationary_true.keys() if np.sum(x) == effective_capacity]))
    n_blocking_states = len(blocking_states)
    print("Blocking states ({}): {}".format(n_blocking_states, blocking_states))

    # Expected blocking cost
    expected_blocking_cost = -estimate_expected_reward(env_queue, agent, probas_stationary_true)
    # Expected costs for each blocking state x
    probas_arrival = [l / np.sum(job_class_rates) for l in job_class_rates]
    expected_costs_values = [probas_stationary_true[x] * \
                             np.sum([blocking_costs[i] * probas_arrival[i] * int(x[i] == blocking_sizes[i] or np.sum(x) >= effective_capacity) for i, _ in enumerate(job_class_rates)])
                                for x in blocking_states]
    expected_costs = dict([(tuple(x), c) for x, c in zip(blocking_states, expected_costs_values)])

    # Entropy of blocking and entropy of blocking cost relative to the maximum possible entropy
    # The maximum possible entropy is log(#blocking-states) which is derived when all probabilities (of each blocking state) are the same,
    # since in that case, letting B = #blocking-states, the entropy becomes H = -1/B*log(1/B) * B = log(B)
    # In the cost-based entropy, the maximum entropy is assumed to also happen when all probabilities are the same,
    # which yield equal-probability costs computed below as expected_costs_uniform_values
    # which makes the maximum cost-based entropy be equal to HC = sum_x{ -c_unif(x)*1/B*log(1/B) } = sum_x{c_unif(x)} * log(B) / B
    expected_costs_uniform_values = [1/n_blocking_states * \
                                     np.sum([blocking_costs[i] * probas_arrival[i] * int(x[i] == blocking_sizes[i] or np.sum(x) >= effective_capacity) for i, _ in enumerate(job_class_rates)])
                                        for x in blocking_states]
    entropy_blocking_rel = np.sum([-probas_stationary_true[x] * np.log10(probas_stationary_true[x]) for x in blocking_states]) / np.log10(n_blocking_states)
    entropy_blocking_cost_rel = np.sum([-expected_costs[x] * probas_stationary_true[x] * np.log10(probas_stationary_true[x]) for x in blocking_states]) / \
                            (np.sum(expected_costs_uniform_values) * np.log10(n_blocking_states) / n_blocking_states)

    print("System: capacity={}, lambdas={}, mus={}, rhos={}, buffer_size_activation={}, # burn-in time steps={}, min # cycles for expectations={}" \
          .format(capacity, job_class_rates, service_rates, rhos, Js, burnin_time_steps, min_num_cycles_for_expectations))
    time_start_all = timer()
    for case, (N, T) in enumerate(zip(N_values, T_values)):
        print("\n*** Running simulation for N={} ({} of {}) on {} replications...".format(N, case+1, ncases, replications))

        dict_params_simul = {
            'T': T,
            'absorption_set': absorption_set,
            'activation_set': activation_set,
            'burnin_time_steps': burnin_time_steps,
            'min_num_cycles_for_expectations': min_num_cycles_for_expectations,
            'method_survival_probability_estimation': method_proba_surv,
            'seed': seed,
                }

        for r in range(1, replications+1):
            print("\n\n\n\n\tReplication {} of {}...".format(r, replications))
            seed_rep = seed + 10*(r - 1)
                ## We multiply by 10 to leave enough "space" to assign seeds in between
                ## two consecutive replications to assign to the different FV steps
                ## (est_surv, est_abs, est_fv)

            time_start_fv = timer()

            print("\n\t--> Running Fleming-Viot estimation... {}".format(get_current_datetime_as_string()))
            dict_params_simul['maxevents'] = np.Inf
            dict_params_simul['seed'] = seed_rep
            envs_queue = [env_queue if i == 0 else copy.deepcopy(env_queue) for i in range(N)]
            proba_blocking_fv, expected_reward_fv, probas_stationary_fv, \
                expected_absorption_time, n_absorption_time_observations, \
                    time_last_absorption, time_end_simulation_et, max_survival_time, time_end_simulation_fv, \
                        n_events_et, n_events_fv_only = estimate_blocking_fv(envs_queue, agent,
                                                                             dict_params_simul, dict_params_info,
                                                                             probas_stationary=probas_stationary_true if use_stationary_probability_for_start_state else None)
            # If we need to check the calculation of the blocking probability done by the above estimate_blocking_*() call
            if False:
                proba_blocking_from_stationary_probabilities = compute_blocking_probability_knapsack_from_probabilities_and_job_arrival_rates(
                    probas_stationary_fv,
                    capacity, job_class_rates,
                    blocking_sizes)
                #print(f"Pr(FV)={proba_blocking_fv}, from stationary probs: {proba_blocking_from_stationary_probabilities}")
                assert np.isclose(proba_blocking_fv, proba_blocking_from_stationary_probabilities)
            n_events_fv = n_events_et + n_events_fv_only
            n_survival_curve_observations = n_absorption_time_observations

            exec_time_fv = timer() - time_start_fv

            if run_mc:
                time_start_mc = timer()
                print("\t--> Running Monte-Carlo estimation... {}".format(get_current_datetime_as_string()))
                # Make a copy of the simulation parameters so that we do not alter the simulation parameters that are in use for the FV simulation which defines the benchmark
                dict_params_simul_mc = copy.deepcopy(dict_params_simul)
                dict_params_simul_mc['maxevents'] = n_events_fv
                dict_params_simul_mc['T'] = n_events_fv
                proba_blocking_mc, expected_reward_mc, probas_stationary_mc, n_cycles, \
                    expected_return_time_mc, _, \
                        time_mc, n_events_mc = estimate_blocking_mc(env_queue, agent, dict_params_simul_mc, dict_params_info=dict_params_info)
                # Note: (2022/10/24) The above number n_cycles is the number of cycles used for the estimation of the stationary probabilities
                # It may NOT coincide with the number of return cycles used to estimate expected_return_time_mc
                # (whose value is stored in returned object `_`) when a burn-in period is left at the beginning
                # of the simulation (see dict_params_simul['burnin_time_steps']).
                # In fact, the cycle used in the estimation of the stationary probabilities is defined by
                # the return to the state at which the Markov process is found *after* the burn-in period is over
                # (which may NOT coincide with the start state) (and cycles are counted from that moment over,
                # so whatever happened during the burn-in period is discarded), whereas the cycle used in the estimation
                # of the expected return time is defined by the return to the *initial* state of the Markov chain,
                # and cycles are counted starting from the time the first return to the initial state happens *after*
                # the burn-in period.
                # Here we store the former, i.e. the number of cycles used to estimate the stationary probabilities
                # because the probabilities are our quantity of interest to evaluate the Monte-Carlo estimator.
                n_return_observations = n_cycles

                # If we need to check the calculation of the blocking probability done by the above estimate_blocking_*() call
                if False:
                    proba_blocking_from_stationary_probabilities = compute_blocking_probability_knapsack_from_probabilities_and_job_arrival_rates(
                        probas_stationary_mc,
                        capacity, job_class_rates,
                        blocking_sizes)
                    # print(f"Pr(MC)={proba_blocking_mc}, from stationary probs: {proba_blocking_from_stationary_probabilities}")
                    assert np.isclose(proba_blocking_mc, proba_blocking_from_stationary_probabilities)

                # Check comparability in terms of # events in each simulation (MC vs. FV)
                if n_events_mc != n_events_fv:
                    message = "!!!! #events(MC) != #events(FV) ({}, {}) !!!! (Comparison between the Monte-Carlo and Fleming-Viot may not be a fair comparison)".format(n_events_mc, n_events_fv)
                    print(message)  # Shown in the log
                    warn(message)   # Shown in the console

                exec_time_mc = timer() - time_start_mc
            else:
                proba_blocking_mc, expected_reward_mc, expected_return_time_mc, n_return_observations, dict_stats_mc = np.nan, None, None, {}
                time_mc = 0.0
                n_events_mc = 0
                exec_time_mc = 0.0

            exec_time = exec_time_fv + exec_time_mc

            if run_mc:
                print("Total execution time FV & MC: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))
            else:
                print("Total execution time FV: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))

            # -- Results
            if run_mc:
                # MC results
                # Compute the frequence-based and probability-based coverage of the blocking states where the estimated stationary probability > 0
                # The probability-based coverage is computed using the TRUE stationary probabilities as (inverse) weights of the observed states
                # so that we know the observation of the most important states to
                coverage_prob_blocking_states_mc, coverage_info_blocking_states_mc, \
                    coverage_entropy_blocking_states_mc, coverage_costentropy_blocking_states_mc, \
                        contribution_costentropy_blocking_states_mc, contribution_true_costentropy_blocking_states, n_blocking_states_observed_mc = \
                            compute_blocking_state_coverages(probas_stationary_mc, probas_stationary_true, expected_costs)
                print("\tP(K) by MC: {:.6f}% (#events: {}, coverage of blocking states: {} of {} {:.1f}%, prob-weighted coverage: {:.1f}%, inverse-prob-weighted coverage: {:.1f}%)" \
                      .format(proba_blocking_mc*100, n_events_mc, n_blocking_states_observed_mc, n_blocking_states, n_blocking_states_observed_mc / n_blocking_states *100, coverage_prob_blocking_states_mc*100, coverage_info_blocking_states_mc*100))

            # FV results
            coverage_prob_blocking_states_fv, coverage_info_blocking_states_fv, \
                coverage_entropy_blocking_states_fv, coverage_costentropy_blocking_states_fv, \
                    contribution_costentropy_blocking_states_fv, _, n_blocking_states_observed_fv = \
                compute_blocking_state_coverages(probas_stationary_fv, probas_stationary_true, expected_costs)
            print("\tP(K) by FV: {:.6f}%, E(T) = {:.1f} (simulation time for E(T) = {:.1f} ({} steps) (complete cycles span {:.1f}%),"
                    " max survival time = {:.1f}, #events = {} (ET) + {} (FV) = {}), coverage of blocking states = {} of {} {:.1f}%, prob-weighted coverage = {:.1f}%, inverse-prob-weighted coverage = {:.1f}%" \
                  .format(  proba_blocking_fv * 100, expected_absorption_time, time_end_simulation_et, n_events_et,
                            time_last_absorption/time_end_simulation_et*100,
                            max_survival_time,
                            n_events_et, n_events_fv_only, n_events_fv,
                            n_blocking_states_observed_fv, n_blocking_states, n_blocking_states_observed_fv / n_blocking_states *100, coverage_prob_blocking_states_fv*100, coverage_info_blocking_states_fv*100))
            print("\tTrue P(K): {:.6f}%".format(proba_blocking_true*100))

            # Store the results
            df_append = pd.DataFrame([[ # System characteristics
                                        capacity,
                                        blocking_costs,
                                        job_class_rates,
                                        format(rhos),
                                        # Simulation parameters
                                        Js,
                                        N,
                                        T,
                                        dict_params_simul['burnin_time_steps'],
                                        dict_params_simul['min_num_cycles_for_expectations'],
                                        method_proba_surv.name,
                                        # Replication
                                        r,
                                        dict_params_simul['seed'],
                                        # MC results
                                        [probas_stationary_mc[x] for x in blocking_states],
                                        [probas_stationary_mc[x] / probas_stationary_true[x] for x in blocking_states],
                                        proba_blocking_mc,
                                        -expected_reward_mc,
                                        time_mc,
                                        expected_return_time_mc,
                                        n_events_mc,
                                        n_return_observations,
                                        n_blocking_states_observed_mc,
                                        n_blocking_states_observed_mc / n_blocking_states,
                                        coverage_prob_blocking_states_mc,
                                        coverage_info_blocking_states_mc,
                                        coverage_entropy_blocking_states_mc,
                                        coverage_costentropy_blocking_states_mc,
                                        contribution_costentropy_blocking_states_mc,
                                        # FV results
                                        [probas_stationary_fv[x] for x in blocking_states],
                                        [probas_stationary_fv[x] / probas_stationary_true[x] for x in blocking_states],
                                        proba_blocking_fv,
                                        -expected_reward_fv,
                                        time_end_simulation_fv,
                                        expected_absorption_time,
                                        n_absorption_time_observations,
                                        max_survival_time,
                                        n_events_et,
                                        n_events_fv_only,
                                        n_events_fv,
                                        n_survival_curve_observations,
                                        n_blocking_states_observed_fv,
                                        n_blocking_states_observed_fv / n_blocking_states,
                                        coverage_prob_blocking_states_fv,
                                        coverage_info_blocking_states_fv,
                                        coverage_entropy_blocking_states_fv,
                                        coverage_costentropy_blocking_states_fv,
                                        contribution_costentropy_blocking_states_fv,
                                        # True probability
                                        blocking_states,
                                        [probas_stationary_true[x] for x in blocking_states],
                                        proba_blocking_true,
                                        [expected_costs[x] / probas_stationary_true[x] for x in blocking_states],
                                        [expected_costs[x] for x in blocking_states],
                                        contribution_true_costentropy_blocking_states,
                                        expected_blocking_cost,
                                        entropy_blocking_rel,
                                        entropy_blocking_cost_rel,
                                        # Execution times
                                        exec_time_mc,
                                        exec_time_fv,
                                        exec_time]],
                                     columns=df_results.columns, index=[case+1])
            df_results = pd.concat([df_results, df_append], axis=0)

        # Fill NaN values that would impede the generation of plots
        for col in ['Pr(MC)', 'ExpCost(MC)', 'Pr(FV)', 'ExpCost(FV)']:
            df_results[col].fillna(0, inplace=True)

        print("Results:")
        print(df_results)
    time_end_all = timer()

    print("Total execution time: {:.1f} min".format((time_end_all - time_start_all) / 60))
    title = "Simulation results for capacity={}, lambdas={}, mus={}, rhos={}, ({}<=N<={}), ({}<=T<={}), #Events<={}, Rep={}" \
        .format(capacity, job_class_rates, service_rates, rhos, N_values[0], N_values[-1], T_values[0], T_values[-1], n_events_fv, replications)
    print(title)
    print("Raw results:")
    print(df_results)

    df_results_agg_by_NT = aggregation_bygroups(df_results, ['N', 'T'], ['#Events(MC)', '#Cycles(MC)', 'Pr(MC)', '#Events(FV)', 'Pr(FV)'])
    print("Aggregated results by N & T:")
    print(df_results_agg_by_NT)

    # Reset pandas display options modified above
    reset_pandas_options(pandas_options)

    # Add back the average of # events to the full data frame
    df_results = pd.merge(df_results, df_results_agg_by_NT.xs('mean', axis=1, level=1)[['#Events(MC)', '#Cycles(MC)', '#Events(FV)']],
                          left_on=['N', 'T'], right_index=True, suffixes=["", "_mean"])
    # Convert average to integer
    if run_mc:
        df_results = df_results.astype({'#Events(MC)_mean': np.int})
        df_results = df_results.astype({'#Cycles(MC)_mean': np.int})
    df_results = df_results.astype({'#Events(FV)_mean': np.int})

    return df_results, df_results_agg_by_NT
#------------------- Functions --------------------


#------------------- Functions to parse input arguments ---------------------#
def parse_input_parameters(argv):
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage="%prog "
                                         "[--queue_system] "
                                         "[--method] "
                                         "[--analysis_type] "
                                         "[--values_parameter_analyzed] "
                                         "[--value_parameter_not_analyzed] "
                                         "[--J] "
                                         "[--J_factor] "
                                         "[--use_stationary_probability_for_start_states] "
                                         "[--burnin_time_steps] "
                                         "[--min_num_cycles_for_expectations] "
                                         "[--replications] "
                                         "[--seed] "
                                         "[--save_results] "
                                         "[--save_with_dt] "
                                         "[--plot] "
                                         )
    parser.add_option("--queue_system",
                      type="str",
                      metavar="Queue system",
                      help="Queue system defining the environment on which learning takes place [default: %default]")
    parser.add_option("--method",
                      type="str",
                      metavar="Learning method",
                      help="Learning method [default: %default]")
    parser.add_option("--analysis_type",
                      type="str",
                      metavar="Type of analysis",
                      help="Type of analysis, either 'N' (#particles is increased) or 'T' (#arrival events is increased) [default: %default]")
    parser.add_option("--values_parameter_analyzed", dest="values_parameter_analyzed",
                      type="str",
                      action="callback",
                      callback=convert_str_argument_to_list_of_type,
                      metavar="Values of the analysis parameter",
                      help="List of values of the parameter against which the convergence of the method is analyzed [default: %default]")
    parser.add_option("--value_parameter_not_analyzed",
                      type="int",
                      metavar="Parameter not analyzed",
                      help="Fixed value of the parameter against which the convergence of the method is NOT analyzed (i.e. parameter 'T' if analysis_type='N' or 'N' if analysis_type='T') [default: %default]")
    parser.add_option("--J", dest="J",
                      type="str",
                      action="callback",
                      callback=convert_str_argument_to_list_of_type,
                      metavar="J values for the different job classes",
                      help="Absorption set sizes J for each class [default: %default]")
    parser.add_option("--J_factor", dest="J_factor",
                      type="str",
                      action="callback",
                      callback=convert_str_argument_to_list_of_type,
                      metavar="J/K factors",
                      help="Factors that define the absorption set size J (only used when J is not given) [default: %default]")
    parser.add_option("--use_stationary_probability_for_start_states",
                      action="store_true",
                      help="Whether to use the stationary probability distribution for the start states in Fleming-Viot simulation [default: %default]")
    parser.add_option("--burnin_time_steps",
                      type="int",
                      metavar="Burn-in time steps",
                      help="Number of MC steps to use as burn-in before starting to estimate quantities of interest [default: %default]")
    parser.add_option("--min_num_cycles_for_expectations",
                      type="int",
                      metavar="Minimum # cycles for expectations",
                      help="Minimum # cycles to observe in order to compute expectations (e.g. E(T_A) the expected re-absorption time) [default: %default]")
    parser.add_option("--replications",
                      type="int",
                      metavar="# Replications",
                      help="Number of replications to run [default: %default]")
    parser.add_option("--run_mc",
                      action="store_true",
                      help="Whether to run the Monte-Carlo estimation against which the FV estimator is compared [default: %default]")
    parser.add_option("--seed",
                      type="int",
                      metavar="Seed",
                      help="Seed value to use for the simulations [default: %default]")
    parser.add_option("--save_results",
                      action="store_true",
                      help="Whether to save the results into a CSV file [default: %default]")
    parser.add_option("--save_with_dt",
                      action="store_true",
                      help="Whether to use the execution datetime as suffix of output file names [default: %default]")
    parser.add_option("--plot",
                      action="store_true",
                      help="Whether to plot the learning process (e.g. theta estimates and objective function) [default: %default]")
    if False:
        parser.add_option("-d", "--debug", dest="debug", default=False,
                          action="store_true",
                          help="debug mode")
        parser.add_option("-v", "--verbose", dest="verbose", default=False,
                          action="store_true",
                          help="verbose: show relevant messages in the log")

    # IMPORTANT: The default values of parameters processed by callbacks should define the value using the type of the input argument of the callback (e.g. str)
    # E.g. `--values_parameter_analyzed "[64, 128, 256]"` --> Note that there is NO equal sign following the parameter name as is the case with the other numeric or str parameters
    # as in `--analysis_type=T` (str type) or `--burnin_time_steps=10` (int type).
    # Also, boolean parameters are EITHER included or NOT included among the arguments. If included, the action defined in the `action` argument of parser.add_option() above
    # defines what value is assigned to the variable given in the `dest=` argument of said add_option() method.
    # When action="store_true", the default action set for the argument just below should be `False`, o.w. the parameter will always be set to True,
    # as there is no way to set it to False (since False is set when the argument is NOT included among the script arguments...!)
    # The contrary is true if action="store_false", i.e. the parameter's default value should be set to True.
    # NOTE however that this may not always be the case, if e.g. the script is called from PyCharm where the easy way
    # to pass parameter values is to set their default with parser.defaults(). If this is the case, WHENEVER we use
    # the script to run a process on the server is by setting the default value of boolean variables to False.
    parser.set_defaults(queue_system="loss-network",
                        method="FV",
                        analysis_type="T",
                        values_parameter_analyzed=[400, 2000, 4000], #[640, 1280, 2560], #[400, 4000], #[640, 1280, 2560], #[400, 800, 1600], #[64, 128, 256], #[160, 320, 640], #[640, 1280, 2560], #[80, 160, 320, 640], #[1280, 2560], #[80, 160, 320, 640],
                        value_parameter_not_analyzed=200, #200, #50, #400 #500 #100
                        J=[2, 3], #[2, 3], #[1, 2], #[1, 1], #None,
                        J_factor=None, #[0.5, 0.5], #[0.2, 0.4],    # [0.1, 0.6]
                        use_stationary_probability_for_start_states=False,
                        burnin_time_steps=10,
                        min_num_cycles_for_expectations=5,
                        replications=10,
                        run_mc=True,
                        seed=1313,
                        save_results=True,
                        save_with_dt=True,
                        plot=True)

    (options, args) = parser.parse_args(argv)

    print("Parsed command line options: " + repr(options))

    # options: `Values` object whose parameters are referred with the dot notation (e.g. options.x)
    # args: argument values (which do not require an argument name
    return options, args

def show_execution_parameters(options):
    print(get_current_datetime_as_string())
    print("Execution parameters:")
    print("nservers={}".format(nservers))
    print("job class rates={}".format(job_class_rates))
    print("service rates={}".format(service_rates))
    print("class loads={}".format(rhos))
    print("blocking sizes={}".format(blocking_sizes))
    print("Type of analysis: '{}'".format(options.analysis_type))
    print("Capacity K={}".format(capacity))
    print("Activation sizes J={}".format(options.J))
    print("Activation size factors J/K={}".format(options.J_factor))
    print("# particles N={}".format(N_values))
    print("# arrival events for E(T_A) estimation T={}".format(T_values))
    print("use_stationary_probability_for_start_states={}".format(options.use_stationary_probability_for_start_states))
    print("Burn-in time steps BITS={}".format(options.burnin_time_steps))
    print("Min #cycles to estimate expectations MINCE={}".format(options.min_num_cycles_for_expectations))
    print("Replications={}".format(options.replications))
    print("run_mc={}".format(options.run_mc))
    print("save_results={}".format(options.save_results))
    print("seed={}".format(options.seed))
#------------------- Functions to parse input arguments ---------------------#


if __name__ == "__main__":
    #--------------------------- Parse user arguments -----------------------#
    options, args = parse_input_parameters(sys.argv[1:])

    print("Parsed user arguments:")
    print(f"Options: {options}")
    print(f"Arguments: {args}")
    print("")

    resultsdir = "../../RL-002-QueueBlocking/results/FV/" + options.queue_system

    if options.queue_system == "loss-network":
        capacity = 6 #7 #6 #10
        job_class_rates = [1, 5]
        rhos = [0.5, 0.3] #[0.3, 0.1] #[0.8, 0.6] #[0.3, 0.1] #[0.8, 0.6] #[0.2, 0.1] #[0.5, 0.3]
        service_rates = [l / r for l, r in zip(job_class_rates, rhos)]
        blocking_sizes = [4, 6] #[1, 2] #[4, 6] #[5, 7] #[4, 6]

        #************ BLOCKING COSTS ************
        # See explanation details in run_FVRL.py
        adjustment_constant_for_large_enough_cost = 20 # This constant is normally > 1 and is adjusted by eye-ball as a function of rhos (e.g. 20 for rhos = [0.5, 0.3], 2 for rhos = [0.2, 0.1]), based on NOT obtaining costs that are too large (but the final decision comes from observing the magnitude of the obtained gradient, which should not be too large nor too small)
        blocking_costs = [adjustment_constant_for_large_enough_cost * sum(job_class_rates) * 1/r**(capacity - 1) / l for l, r in zip(job_class_rates, rhos)] #[5000, 1000, 100] (for rho_average = 0.8) #[2E5, 1.5E5, 1E5] #[2E5, 1E2, 1E0] #[1E1, 1E2, 1E4] #[20.0, 10.0, 5.0]
        # Perturb the costs so that we don't get the trivial optimum at e.g. [6, 6, 6]
        np.random.seed(1)
        multiplicative_noise_values = [0.5, 2]
        #blocking_costs = [int(round(c*n)) if c*n > 1 else c*n for c, n in zip(blocking_costs, multiplicative_noise_values)]
        blocking_costs = [2E3, 2E4] #[1, 1] #[2.5E3, 4.9E6] #[2000, 20000] ([0.5, 0.3]) #[180, 600] ([0.8. 0.6]) # Costs computed in run_FVRL.py from p(x), lambdas and rhos
        #************ BLOCKING COSTS ************
    else:
        raise ValueError(f"The queue system ({options.queue_system}) is invalid. Valid queue systems are: 'loss-network'")

    nservers = len(service_rates)
    J = options.J
    J_factor = options.J_factor

    # Set the values of N and T
    if not options.analysis_type in ["N", "T"]:
        raise ValueError("Parameter 'analysis_type' must be either 'N' or 'T'")
    if options.analysis_type == "N":
        other_variable = "T"
        N_values = options.values_parameter_analyzed
        T_values = [options.value_parameter_not_analyzed]
        resultsfile_prefix = "estimates_vs_N"
        params_str = "K={},block={},costs={},lambdas={},rho={},J={},T={},N={},Prob={}" \
            .format(capacity, blocking_sizes, 1 if blocking_costs is None else blocking_costs, job_class_rates, rhos, J if J is not None else J_factor, T_values, N_values, options.use_stationary_probability_for_start_states)
    else:
        other_variable = "N"
        N_values = [options.value_parameter_not_analyzed]
        T_values = options.values_parameter_analyzed
        resultsfile_prefix = "estimates_vs_T"
        params_str = "K={},block={},costs={},lambdas={},rhos={},J={},N={},T={},Prob={}" \
                .format(capacity, blocking_sizes, 1 if blocking_costs is None else blocking_costs, job_class_rates, rhos, J if J is not None else J_factor, N_values, T_values, options.use_stationary_probability_for_start_states) \
                .replace(" ", "")

    show_execution_parameters(options)

    if options.save_results:
        dt_start, stdout_sys, fh_log, logfile, resultsfile, resultsfile_agg, proba_functions_file, figfile = \
            createLogFileHandleAndResultsFileNames(subdir=f"FV/{options.queue_system}", prefix=resultsfile_prefix, suffix=params_str, use_dt_suffix=options.save_with_dt)
        # Show the execution parameters again in the log file
        show_execution_parameters(options)
    else:
        fh_log = None; resultsfile = None; resultsfile_agg = None; proba_functions_file = None; figfile = None
    #--------------------------- Parse user arguments -----------------------#

    start_time = timer()
    results, results_agg = analyze_convergence( capacity=capacity, job_class_rates=job_class_rates, service_rates=service_rates,
                                                blocking_sizes=blocking_sizes, blocking_costs=blocking_costs,
                                                N=N_values, #[N, 2*N, 4*N, 8*N, 16*N],  # [800, 1600, 3200], #[10, 20, 40], #[24, 66, 179],
                                                T=T_values, #50, #[170, 463, 1259],
                                                J=J,
                                                J_factor=J_factor,
                                                use_stationary_probability_for_start_state=options.use_stationary_probability_for_start_states,
                                                burnin_time_steps=options.burnin_time_steps, min_num_cycles_for_expectations=options.min_num_cycles_for_expectations,
                                                method_proba_surv=SurvivalProbabilityEstimation.FROM_N_PARTICLES,
                                                replications=options.replications, run_mc=options.run_mc,
                                                seed=options.seed)
    end_time = timer()
    elapsed_time = end_time - start_time
    print("\n+++ OVERALL execution time: {:.1f} min, {:.1f} hours".format(elapsed_time / 60, elapsed_time / 3600))

    # Save results
    save_dataframes([{'df': results, 'file': resultsfile},
                     {'df': results_agg, 'file': resultsfile_agg}])

    # Plot results
    fontsize = 20
    show_title = False
    savefig = False
    figmaximize = True
    if options.plot:
        title = (resultsfile if resultsfile is not None else "") + "\n" + params_str + " (exec. time = {:.0f} min)".format(elapsed_time / 60) if show_title else None
        if options.analysis_type == "N":
            for T in T_values:
                # Note: the columns defined in parameters `x` and `x2` are grouping variables that define each violin plot
                figfile = os.path.join(resultsdir, f"FV-Conv{options.analysis_type}-{params_str}-Proba-Violin-2Xaxis.png") if savefig else None
                axes1 = plot_results_fv_mc( results, x="N", x2="#Cycles(MC)_mean",
                                            prob_true="Pr(True)",
                                            xlabel="# particles", xlabel2="# Return Cycles to absorption set",
                                            ymin=0.0, plot_mc=options.run_mc, splines=False,
                                            fontsize=fontsize,
                                            figfile=figfile,
                                            figmaximize=figmaximize,
                                            title=title)
                figfile = os.path.join(resultsdir, f"FV-Conv{options.analysis_type}-{params_str}-Cost-Violin-2Xaxis.png") if savefig else None
                axes2 = plot_results_fv_mc( results, x="N", x2="#Cycles(MC)_mean",
                                            prob_mc="ExpCost(MC)", prob_fv="ExpCost(FV)", prob_true="ExpCost",
                                            multiplier=1,
                                            xlabel="# particles", xlabel2="# Return Cycles to absorption set", ylabel="Expected cost",
                                            ymin=0.0, plot_mc=options.run_mc, splines=False,
                                            fontsize=fontsize,
                                            figfile=figfile,
                                            figmaximize=figmaximize,
                                            title=title)
        elif options.analysis_type == "T":
            for N in N_values:
                # Note: the columns defined in parameters `x` and `x2` are grouping variables that define each violin plot
                figfile = os.path.join(resultsdir, f"FV-Conv{options.analysis_type}-{params_str}-Proba-Violin-2Xaxis.png") if savefig else None
                axes1 = plot_results_fv_mc( results, x="T", x2="#Cycles(MC)_mean",
                                            prob_true="Pr(True)",
                                            xlabel="# arrival events", xlabel2="# Return Cycles to absorption set",
                                            ymin=0.0, plot_mc=options.run_mc, splines=False,
                                            fontsize=fontsize,
                                            figfile=figfile,
                                            figmaximize=figmaximize,
                                            title=title)
                figfile = os.path.join(resultsdir, f"FV-Conv{options.analysis_type}-{params_str}-Cost-Violin-2Xaxis.png") if savefig else None
                axes2 = plot_results_fv_mc( results, x="T", x2="#Cycles(MC)_mean",
                                            prob_mc="ExpCost(MC)", prob_fv="ExpCost(FV)", prob_true="ExpCost",
                                            multiplier=1,
                                            xlabel="# arrival events", xlabel2="# Return Cycles to absorption set", ylabel="Expected cost",
                                            ymin=0.0, plot_mc=options.run_mc, splines=False,
                                            fontsize=fontsize,
                                            figfile=figfile,
                                            figmaximize=figmaximize,
                                            title=title)

    # Close log file, if any was opened
    if fh_log is not None:
        closeLogFile(fh_log, stdout_sys, dt_start)

    if True:
        # Plot the estimated probability and expected cost by blocking state, so that we can analyze how well each blocking state contributing to the expected cost is estimated

        # Split relevant columns that are stored as lists into separate columns
        # NOTE: 'ExpectedCosts' contains the TRUE expected cost by blocking state, and all estimated expected costs (MC and FV) are computed by multiplying this 'ExpectedCosts'
        # by the probability estimate ratio (e.g. ProbasRatio(MC), which is equal to Probas(MC) / Probas(True))
        columns_to_convert = ['BlockingStates', 'Probas(MC)', 'ProbasRatio(MC)', 'Probas(FV)', 'ProbasRatio(FV)', 'Probas(True)', 'ExpectedCosts']
        all_columns_converted = []
        for col in columns_to_convert:
            n_blocking_states = len(results[col].iloc[0])
            column_names = [col + '_' + str(j) for j in range(1, n_blocking_states+1)]
            results[column_names] = pd.DataFrame(results[col].tolist(), index=results.index)
            all_columns_converted += column_names
        # Compute statistics on the results associated to the largest parameter value (which is more reliable) (e.g. max T value if analysis by T)
        msk = results.index == min(results.index)
        msk = results.index == max(results.index)
        param_value = results[options.analysis_type][msk].iloc[0]
        results_n = results[all_columns_converted][msk].agg(['count'])
        results_mean = results[all_columns_converted][msk].agg(['mean'])
        results_std = results[all_columns_converted][msk].agg(['std'])
        results_n['id'] = 1  # Needed for the transposition
        results_mean['id'] = 1  # Needed for the transposition
        results_std['id'] = 1  # Needed for the transposition
        results_toplot_n = pd.wide_to_long(results_n, columns_to_convert, i='id', j='idx', sep="_")
        results_toplot_n.reset_index(inplace=True)
        results_toplot_n['BlockingStates'] = results['BlockingStates'].iloc[0]
        results_toplot_mean = pd.wide_to_long(results_mean, columns_to_convert, i='id', j='idx', sep="_")
        results_toplot_mean.reset_index(inplace=True)
        results_toplot_mean['BlockingStates'] = results['BlockingStates'].iloc[0]
        results_toplot_std = pd.wide_to_long(results_std, columns_to_convert, i='id', j='idx', sep="_")
        results_toplot_std.reset_index(inplace=True)
        results_toplot_std['BlockingStates'] = results['BlockingStates'].iloc[0]
        # Sort the values by decreasing true contribution to the expected cost
        results_toplot_mean['ExpectedCostsRel'] = results_toplot_mean['ExpectedCosts'] / results['ExpCost'].iloc[0]
        results_toplot_mean['ExpectedCosts(MC)'] = results_toplot_mean['ExpectedCosts'] * results_toplot_mean['ProbasRatio(MC)']
        results_toplot_mean['ExpectedCostsRel(MC)'] = results_toplot_mean['ExpectedCosts(MC)'] / results['ExpCost'].iloc[0]
        results_toplot_mean['ExpectedCosts(FV)'] = results_toplot_mean['ExpectedCosts'] * results_toplot_mean['ProbasRatio(FV)']
        results_toplot_mean['ExpectedCostsRel(FV)'] = results_toplot_mean['ExpectedCosts(FV)'] / results['ExpCost'].iloc[0]
        results_toplot_std['ExpectedCosts(FV)'] = results_toplot_mean['ExpectedCosts'] * results_toplot_std['ProbasRatio(FV)']
        results_toplot_std['ExpectedCostsRel(FV)'] = results_toplot_std['ExpectedCosts(FV)'] / results['ExpCost'].iloc[0]
        results_toplot_std['ExpectedCosts(MC)'] = results_toplot_mean['ExpectedCosts'] * results_toplot_std['ProbasRatio(MC)']
        results_toplot_std['ExpectedCostsRel(MC)'] = results_toplot_std['ExpectedCosts(MC)'] / results['ExpCost'].iloc[0]

        # If sorting directly on the data frame, use the following sort_values() method
        #results_toplot_mean.sort_values(['ExpectedCosts'], ascending=False, inplace=True)
        #results_toplot_mean.reset_index(inplace=True)
        ord = np.argsort(results_toplot_mean['ExpectedCosts'])[::-1]
        ax_probas, ax_costs = plt.figure().subplots(1, 2)
        blocking_state_indices = np.arange(len(results_toplot_mean))
        ax_probas.plot(blocking_state_indices, results_toplot_mean['Probas(True)'][ord], 'b.--')
        ax_probas.errorbar(blocking_state_indices, results_toplot_mean['Probas(FV)'][ord], yerr=results_toplot_std['Probas(FV)'][ord], capsize=4, marker='x', color='green', markersize=10)
        ax_probas.errorbar(blocking_state_indices+0.05, results_toplot_mean['Probas(MC)'][ord], yerr=results_toplot_std['Probas(MC)'][ord], capsize=4, marker='x', color='red', markersize=10)
        ax_costs.plot(blocking_state_indices, results_toplot_mean['ExpectedCostsRel'][ord], 'b.--')
        ax_costs.errorbar(blocking_state_indices, results_toplot_mean['ExpectedCostsRel(FV)'][ord], yerr=results_toplot_std['ExpectedCostsRel(FV)'][ord], capsize=4, marker='x', color='green', markersize=10)
        ax_costs.errorbar(blocking_state_indices+0.05, results_toplot_mean['ExpectedCostsRel(MC)'][ord], yerr=results_toplot_std['ExpectedCostsRel(MC)'][ord], capsize=4, marker='x', color='red', markersize=10)
        #ax_probas.set_yscale('log')    # Use log-scale when probabilities have very different scales
        #ax_costs.set_yscale('log')     # Use log-scale when expected costs have very different scales
        for ax in [ax_probas, ax_costs]:
            ax.set_xlabel("Blocking state (sorted by decreasing True Expected Cost)")
            ax.xaxis.set_ticklabels([ax.xaxis.get_ticklabels()[0]] + list(results_toplot_mean['BlockingStates'][ord]) + [ax.xaxis.get_ticklabels()[-1]])
        ax_probas.set_ylabel("Probability")
        ax_costs.set_ylabel("Proportion of expected cost")
        ax_probas.set_title("Probability by blocking state")
        ax_costs.set_title("Proportion of Expected Cost by blocking state over Total Expected Cost")
        ax_probas.legend(["Pr(True)", "Pr(FV)", "Pr(MC)"])
        ax_costs.legend(["True proportion", "Estimated proportion (FV)", "Estimated proportion (MC)"])
        plt.suptitle(f"Probabilities and Expected Cost by blocking state for largest analysis variable value ({options.analysis_type}={param_value}, {other_variable}={options.value_parameter_not_analyzed})\n(sorted by decreasing True Expected Cost, over {results_toplot_n['ExpectedCosts'].iloc[0]} replications)")
