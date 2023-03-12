# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:17:48 2023

@author: Daniel Mastropietro
@description: Runs the FV estimation of the blocking probability in a Loss Network system receiving multi-class jobs.
"""

import runpy
runpy.run_path('../../setup.py')

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

from Python.lib.environments.queues import rewardOnJobRejection_Constant

from Python.lib.simulators import SetOfStates
from Python.lib.simulators.queues import define_queue_environment_and_agent, estimate_blocking_fv, estimate_blocking_mc, SurvivalProbabilityEstimation

from Python.lib.utils.basic import aggregation_bygroups, convert_str_argument_to_list_of_type, get_current_datetime_as_string, \
    is_scalar, measure_exec_time, set_pandas_options, reset_pandas_options
from Python.lib.utils.computing import compute_blocking_probability_knapsack

from Python.lib.run_FV import createLogFileHandleAndResultsFileNames, plot_results_fv_mc, save_dataframes, show_execution_parameters


#------------------- Functions --------------------
@measure_exec_time
def run_mc_estimation_loss_network( env_queue, Ks, Js, K, T,
                                    burnin_time_steps, min_num_cycles_for_expectations,
                                    seed=1717):
    """
    Test the Monte-Carlo estimation of the blocking probability of a loss network system

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
    dict_params_simul = dict({'absorption_set': SetOfStates(state_boundaries=tuple([J - 1 for J in Js])),
                              'activation_set': SetOfStates(state_boundaries=tuple(Js)),
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
    Test the Fleming-Viot estimation of the blocking probability of a loss network system

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
    dict_params_simul = dict({'absorption_set': SetOfStates(state_boundaries=tuple([J - 1 for J in Js])),
                              'activation_set': SetOfStates(state_boundaries=tuple(Js)),
                              'T': T,  # number of arrivals at which the simulation stops
                              'burnin_time_steps': burnin_time_steps,
                              'min_num_cycles_for_expectations': min_num_cycles_for_expectations,
                              'method_survival_probability_estimation': method_survival_probability_estimation,
                              'seed': seed})
    dict_params_info = dict()

    # Run the simulation!
    return estimate_blocking_fv(envs_queue, agent_accept_reject,
                                dict_params_simul, dict_params_info)


def analyze_convergence(capacity=10, job_class_rates=[0.7], service_rates=[1.0],
                        blocking_sizes=[5],
                        N=[20, 40, 80],
                        T=500,
                        J_factor=[0.3],
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

    N: (opt) int or list or numpy array
        List giving the number of particles to consider in each simulation.
        default: [20, 40, 80]

    T: (opt) int or list or numpy array
        Number of discrete *time steps* to run the queue to estimate the blocking probability in Monte-Carlo
        or the number of *arrival events* to estimate E(T_A) for the FV estimator.
        If not scalar, either N should be a scalar or be a list with just one element, o.w. it should have
        the same number of elements as the number of particles that are tried in this simulation.
        default: 500

    J_factor: (opt) list
        List of J factors to consider for the different job classes that define the absorption size for each class
        in the Fleming-Viot simulation.
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
    nservers = len(service_rates)

    #--- System setup
    dict_params = dict({'environment': {'queue_system': "single-server",
                                        'capacity': capacity,
                                        'nservers': nservers,
                                        'job_class_rates': job_class_rates,  # [0.8, 0.7]
                                        'service_rates': service_rates,  # [1.0, 1.0, 1.0]
                                        'policy_assignment_probabilities': [[1.0]], # [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]] )
                                        'reward_func': rewardOnJobRejection_Constant,
                                            # Note: this reward function is defined here just for informational purposes
                                            # (e.g. when showing debug messages when DEBUG_ESTIMATORS = True in the simulator class)
                                            # but the process actually simply estimates the blocking probability,
                                            # i.e. it does NOT estimate the average reward (although with this definition
                                            # of reward, it could).
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
    Js = [int( round(J_factor * K) ) for J_factor, K in zip(J_factor, blocking_sizes)]

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

    df_results = pd.DataFrame(columns=['capacity',
                                       'rhos',
                                       'J',
                                       'N',
                                       'T',
                                       'burnin_time_steps',
                                       'min_n_cycles',
                                       'method_proba_surv',
                                       'replication',
                                       'Pr(MC)',
                                       'EMC(T)',
                                       'Time(MC)',          # Last continuous time value observed in the MC simulation
                                       '#Events(MC)',
                                       '#Cycles(MC)',
                                       'E(T)',
                                       '#Cycles(E(T))',
                                       'MaxSurvTime',
                                       'Pr(FV)',
                                       'Time(FV)',          # Last continuous time value observed in the FV simulation
                                       '#Events(ET)',
                                       '#Events(FV-Only)',
                                       '#Events(FV)',
                                       '#Samples(S(t))',
                                       'Pr(K)',
                                       'seed',
                                       'exec_time_mc(sec)',
                                       'exec_time_fv(sec)',
                                       'exec_time(sec)'])
    #ncases = int( np.log(nparticles_max / nparticles_min) / np.log(1 + nparticles_step_prop)) + 1
    ncases = len(N_values)

    # Compute the true blocking probability that is used to analyze the estimator quality
    print("Computing TRUE blocking probability for capacity={}, rhos={}...".format(capacity, rhos))
    proba_blocking_true = compute_blocking_probability_knapsack(capacity, rhos, job_class_rates, blocking_sizes=blocking_sizes)

    print("System: capacity={}, lambdas={}, mus={}, rhos={}, buffer_size_activation={}, # burn-in time steps={}, min # cycles for expectations={}" \
          .format(capacity, job_class_rates, service_rates, rhos, Js, burnin_time_steps, min_num_cycles_for_expectations))
    time_start_all = timer()
    for case, (N, T) in enumerate(zip(N_values, T_values)):
        print("\n*** Running simulation for N={} ({} of {}) on {} replications...".format(N, case+1, ncases, replications))

        dict_params_simul = {
            'T': T,
            'absorption_set': SetOfStates(state_boundaries=tuple([J - 1 for J in Js])),
            'activation_set': SetOfStates(state_boundaries=tuple(Js)),
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
            proba_blocking_fv, expected_reward, probas_stationary, \
                expected_absorption_time, n_absorption_time_observations, \
                    time_last_absorption, time_end_simulation_et, max_survival_time, time_end_simulation_fv, \
                        n_events_et, n_events_fv_only = estimate_blocking_fv(envs_queue, agent,
                                                                             dict_params_simul, dict_params_info)
            n_events_fv = n_events_et + n_events_fv_only
            n_survival_curve_observations = n_absorption_time_observations

            exec_time_fv = timer() - time_start_fv

            if run_mc:
                time_start_mc = timer()
                print("\t--> Running Monte-Carlo estimation... {}".format(get_current_datetime_as_string()))
                dict_params_simul['maxevents'] = n_events_fv
                dict_params_simul['seed'] = seed_rep + 2  # This is the same seed used in the FV simulation in estimate_blocking_fv(), so we can compare better
                dict_params_simul['T'] = n_events_fv
                proba_blocking_mc, expected_reward_mc, probas_stationary, n_cycles, \
                    expected_return_time_mc, _, \
                        n_events_mc = estimate_blocking_mc(env_queue, agent, dict_params_simul, dict_params_info=dict_params_info)
                time_mc = np.nan
                # Note: (2022/10/24) The above number n_cycles is the number of cycles observed for the estimation of the stationary probabilities
                # It may NOT coincide with the number of return cycles observed and used to estimate expected_return_time_mc
                # (whose value is stored in returned object `_`)
                # when a burn-in period is left at the beginning of the simulation (see dict_params_simul['burnin_period']),
                # since in that case, the cycle used in the estimation of the stationary probabilities is defined by
                # the return to the buffer size at which the Markov process is found *after* the burn-in period is over
                # (which may NOT coincide with the start buffer size).
                # Here we store the former, i.e. the number of cycles used to estimate the stationary probabilities
                # because those are the values we are interested in collecting to analyze the Monte-Carlo estimator.
                n_return_observations = n_cycles

                # Check comparability in terms of # events in each simulation (MC vs. FV)
                if n_events_mc != n_events_fv:
                    message = "!!!! #events(MC) != #events(FV) ({}, {}) !!!! (Comparison between the Monte-Carlo and Fleming-Viot may not be a fair comparison)".format(n_events_mc, n_events_fv)
                    print(message)  # Shown in the log
                    warn(message)   # Shown in the console

                exec_time_mc = timer() - time_start_mc
            else:
                proba_blocking_mc, expected_return_time_mc, n_return_observations, dict_stats_mc = np.nan, None, None, {}
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
                print("\tP(K) by MC: {:.6f}% (#events {})".format(proba_blocking_mc*100, n_events_mc))

            # FV results
            print("\tP(K) by FV: {:.6f}%, E(T) = {:.1f} (simulation time for E(T) = {:.1f} ({} steps) (complete cycles span {:.1f}%),"
                    " max survival time = {:.1f}, #events = {} (ET) + {} (FV) = {})" \
                  .format(  proba_blocking_fv * 100, expected_absorption_time, time_end_simulation_et, n_events_et,
                            time_last_absorption/time_end_simulation_et*100,
                            max_survival_time,
                            n_events_et, n_events_fv_only, n_events_fv))
            print("\tTrue P(K): {:.6f}%".format(proba_blocking_true*100))

            # Store the results
            df_append = pd.DataFrame([[ capacity,
                                        format(rhos),
                                        Js,
                                        N,
                                        T,
                                        # We use the dict_params_simul parameters here (as opposed to the input parameters to the function (e.g. `burnin_time_steps`)
                                        # because these parameters may be changed above depending on the actual simulator+estimator used to run the experiments.
                                        dict_params_simul['burnin_time_steps'],
                                        dict_params_simul['min_num_cycles_for_expectations'],
                                        method_proba_surv.name,
                                        r,
                                        proba_blocking_mc,
                                        expected_return_time_mc,
                                        time_mc,
                                        n_events_mc,
                                        n_return_observations,
                                        expected_absorption_time,
                                        n_absorption_time_observations,
                                        max_survival_time,
                                        proba_blocking_fv,
                                        time_end_simulation_fv,
                                        n_events_et,
                                        n_events_fv_only,
                                        n_events_fv,
                                        n_survival_curve_observations,
                                        proba_blocking_true,
                                        dict_params_simul['seed'],
                                        exec_time_mc,
                                        exec_time_fv,
                                        exec_time]],
                                     columns=df_results.columns, index=[case+1])
            df_results = pd.concat([df_results, df_append], axis=0)

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
                                         "[--J_factor] "
                                         "[--burnin_time_steps] "
                                         "[--min_num_cycles_for_expectations] "
                                         "[--replications] "
                                         "[--seed] "
                                         "[--create_log] "
                                         "[--save_results] "
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
    parser.add_option("--values_parameter_analyzed",
                      action="callback",
                      callback=convert_str_argument_to_list_of_type,
                      metavar="Values of the analysis parameter",
                      help="List of values of the parameter against which the convergence of the method is analyzed [default: %default]")
    parser.add_option("--parameter_not_analyzed",
                      type="int",
                      metavar="Parameter not analyzed",
                      help="Fixed value of the parameter against which the convergence of the method is NOT analyzed (i.e. parameter 'T' if analysis_type='N' or 'N' if analysis_type='T') [default: %default]")
    parser.add_option("--J_factor", dest="J_factor",
                      type="str",
                      action="callback",
                      callback=convert_str_argument_to_list_of_type,
                      metavar="J/K factors", default="0.3, 0.3, 0.3",
                      help="Factors that define the absorption set size J [default: %default]")
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
    parser.add_option("--seed",
                      type="int",
                      metavar="Seed",
                      help="Seed value to use for the simulations [default: %default]")
    parser.add_option("--create_log",
                      type="bool",
                      action="store_true",
                      help="Whether to create a log file [default: %default]")
    parser.add_option("--save_results",
                      type="bool",
                      action="store_true",
                      help="Whether to save the results into a CSV file [default: %default]")
    parser.add_option("--save_with_dt",
                      type="bool",
                      action="store_true",
                      help="Whether to use the execution datetime as suffix of output file names [default: %default]")
    parser.add_option("--plot",
                      type="bool",
                      action="store_true",
                      help="Whether to plot the learning process (e.g. theta estimates and objective function) [default: %default]")
    if False:
        parser.add_option("-d", "--debug", dest="debug", default=False,
                          action="store_true",
                          help="debug mode")
        parser.add_option("-v", "--verbose", dest="verbose", default=False,
                          action="store_true",
                          help="verbose: show relevant messages in the log")

    parser.set_defaults(queue_system="loss-network",
                        method="MC",
                        analysis_type="N",
                        values_parameter_analyzed="10, 20, 40, 80",
                        value_parameter_not_analyzed=100,
                        J_factor=0.3,
                        burnin_time_steps=20,
                        min_num_cycles_for_expectations=5,
                        replications=5,
                        seed=1313,
                        create_log=False,
                        save_results=False,
                        save_with_dt=True,
                        plot=True)

    (options, args) = parser.parse_args(argv)

    print("Parsed command line options: " + repr(options))

    # options: dictionary with name-value pairs
    # args: argument values (which do not require an argument name
    return options, args
#------------------- Functions to parse input arguments ---------------------#


if __name__ == "__main__":
    #--------------------------- Parse user arguments -----------------------#
    options, args = parse_input_parameters(sys.argv[1:])

    print("Parsed user arguments:")
    print(f"Options: {options}")
    print(f"Arguments: {args}")
    print("")

    if options['queue_system'] == "loss-network":
        capacity = 6  # 6 #10
        job_class_rates = [1, 5]
        rhos = [0.8, 0.6]
        service_rates = [l / r for l, r in zip(job_class_rates, rhos)]
    else:
        raise ValueError(f"The queue system ({options['queue_system']}) is invalid. Valid queue systems are: 'loss-network'")

    nservers = len(service_rates)
    J_factor = options['J_factor']
    #J = int(np.round(options['J_factor'] * K))    # We need the float() because the input arguments are read as strings when running the program from the command line

    # Set the values of N and T
    if not options['analysis_type'] in ["N", "T"]:
        raise ValueError("Parameter 'analysis_type' must be either 'N' or 'T'")
    if options['analysis_type'] == "N":
        N = options['values_parameter_analyzed']
        T = options['value_parameter_not_analyzed']
        N_values = []
        resultsfile_prefix = "estimates_vs_N"
        resultsfile_suffix = "K={},J={},T={}".format(capacity, J_factor, T)
    else:
        N = options['value_parameter_not_analyzed']
        T = options['values_parameter_analyzed']
        T_values = []
        resultsfile_prefix = "estimates_vs_T"
        resultsfile_suffix = "K={},J={},N={}".format(capacity, J_factor, N)

    seed = options['seed']
    show_execution_parameters()

    if options['save_results']:
        dt_start, stdout_sys, fh_log, logfile, resultsfile, resultsfile_agg, proba_functions_file, figfile = \
            createLogFileHandleAndResultsFileNames(prefix=resultsfile_prefix, suffix=resultsfile_suffix, use_dt_suffix=options['save_with_dt'])
        # Show the execution parameters again in the log file
        show_execution_parameters()
    else:
        fh_log = None; resultsfile = None; resultsfile_agg = None; proba_functions_file = None; figfile = None
    #--------------------------- Parse user arguments -----------------------#

    results, results_agg = analyze_convergence( capacity=capacity, job_class_rates=job_class_rates, service_rates=service_rates,
                                                N=N, #[N, 2*N, 4*N, 8*N, 16*N],  # [800, 1600, 3200], #[10, 20, 40], #[24, 66, 179],
                                                T=T, #50, #[170, 463, 1259],
                                                J_factor=J_factor,
                                                burnin_time_steps=options['burnin_time_steps'], min_num_cycles_for_expectations=options['min_num_cycles_for_expectations'],
                                                method_proba_surv=SurvivalProbabilityEstimation.FROM_N_PARTICLES,
                                                replications=options['replications'], run_mc=False,
                                                seed=seed)

    # Save results
    save_dataframes([{'df': results, 'file': resultsfile},
                     {'df': results_agg, 'file': resultsfile_agg}])

    # Plot results
    if options['plot']:
        if options['analysis_type'] == "N":
            for T in T_values:
                # Note: the columns defined in parameters `x` and `x2` are grouping variables that define each violin plot
                axes = plot_results_fv_mc(results, x="N", x2="#Cycles(MC)_mean",
                                          xlabel="# particles", xlabel2="# Return Cycles to absorption set",
                                          ymin=0.0, plot_mc=False, splines=False,
                                          title="nservers={}, K={}, J={}, T={}, BITS={}, MINCE={}" \
                                                .format(nservers, capacity, J_factor, T, options['burnin_time_steps'], options['min_num_cycles_for_expectations']))
        elif options['analysis_type'] == "T":
            for N in N_values:
                # Note: the columns defined in parameters `x` and `x2` are grouping variables that define each violin plot
                axes = plot_results_fv_mc(results, x="T", x2="#Cycles(MC)_mean",
                                          xlabel="# arrival events", xlabel2="# Return Cycles to absorption set",
                                          ymin=0.0, plot_mc=False, splines=False,
                                          title="nservers={}, K={}, J={}, N={}, BITS={}, MINCE={}" \
                                                .format(nservers, capacity, J_factor, N, options['burnin_time_steps'], options['min_num_cycles_for_expectations']))
