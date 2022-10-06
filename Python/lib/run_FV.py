# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 12:15:39 2022

@author: Daniel Mastropietro
@description: Runs the FV estimation of the average reward.
For queues, and when the reward (cost) for blocking is set equal to 1, it estimates the blocking probability.
"""

import os
import sys
import re
import copy
from warnings import warn
from timeit import default_timer as timer
from enum import Enum, unique

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Python.lib.agents.learners.continuing.fv import LeaFV
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep
from Python.lib.agents.queues import AgeQueue

import Python.lib.estimators as estimators

from Python.lib.simulators.queues import estimate_blocking_fv, estimate_blocking_mc
from Python.lib.simulators import define_queue_environment_and_agent

from Python.lib.utils.basic import aggregation_bygroups, get_current_datetime_as_string, get_datetime_from_string, is_scalar
from Python.lib.utils.computing import compute_blocking_probability_birth_death_process
import Python.lib.utils.plotting as plotting

@unique
class Process(Enum):
    "What process is used by Fleming-Viot, the one defined in estimators.py or the one defined in simulators/queues.py"
    Estimators = 1
    Simulators = 2


#------------------- Functions --------------------
from agents.policies import PolicyTypes
from environments.queues import EnvQueueSingleBufferWithJobClasses


def analyze_convergence(estimation_process=Process.Estimators,
                        nservers=1, K=5, buffer_size_activation=1, burnin_cycles_absorption=5,
                        nparticles=[],
                        nparticles_min=40, nparticles_max=80, nparticles_step_prop=1,
                        nmeantimes=500,
                        replications=5,
                        run_mc=True,
                        seed=1717):
    """
    2021/04/19: Analyze convergence of the FV algorithm as the number of particles N increases

    Arguments:
    estimation_process: (opt) Process
        The estimation process that should be used to estimate the blocking probability, whether the estimator
        defined in estimators.py (Process.Estimators) or the estimator defined in simulators/queues.py
        (Process.Simulators).
        default: Process.Estimators

    nparticles: (opt) list
        List giving the number of particles to considered in each simulation.
        If given, it takes precedence over parameters nparticles_min and nparticles_max.
        default: []

    nparticles_step_prop: (opt) positive float
        Step proportion: N(n+1) = (1 + prop)*N(n),
        so that we scale the step as the number of particles increases.

    nmeantimes: (opt) int or list
        Number of discrete time steps to run the queue to estimate either the blocking probability (for MC)
        or to estimate P(T>t) and E(T_A) for the FV estimator.
        If not scalar, it should have the same number of elements as the number of particles that are tried
        in this simulation.
        default: 500
    """

    #--- System setup
    dict_params = dict({'environment': {'capacity': K,
                                        'nservers': nservers,
                                        'job_class_rates': [0.7],  # [0.8, 0.7]
                                        'service_rates': [1.0],  # [1.0, 1.0, 1.0]
                                        'policy_assignment_probabilities': [[1.0]], # [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]] )
                                        'reward_func': None, # Note: we should normally define a reward function when going from state S(t) -> S(t+1) by taking action A(t),
                                                             # but here we avoid defining it because we directly estimate the blocking probability,
                                                             # which is tantamount to receiving reward +1 when blocking.
                                        'rewards_accept_by_job_class': None
                                        },
                        'policy': {'parameterized_policy': PolQueueTwoActionsLinearStep,
                                   'theta': float(K-1)  # This means that there blocking is deterministic at K and otherwise there is acceptance.
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

    # The test of the Fleming-Viot implementation is carried out as follows:
    # - Set K to a small value (e.g. K=5)
    # - Increase the number of particles N
    # - Set a large simulation time (e.g. nmeantimes=50)
    # - Increase the number of particles N check that the error
    # between the estimated and true blocking probability decreases as 1/sqrt(N)
    # (according to Propostion 2.1 in Matt's draft)

    # Simulation parameters
    if buffer_size_activation < 1:
        buffer_size_activation_value = int( round(buffer_size_activation*K) )
    else:
        buffer_size_activation_value = buffer_size_activation

    if nparticles is None or isinstance(nparticles, list) and len(nparticles) == 0:
        # Create the list of nparticles values from the min and max nparticles to consider
        nparticles = nparticles_min
        nparticles_list = [nparticles]
        while nparticles < nparticles_max:
           nparticles += int(nparticles_step_prop * nparticles)
           nparticles_list += [nparticles]
    elif is_scalar(nparticles):
        nparticles_list = [nparticles]
    else:
        # nparticles is a list
        nparticles_list = list(nparticles)

    if not is_scalar(nmeantimes) and len(nmeantimes) > 1 and len(nmeantimes) != len(nparticles):
        raise ValueError("Parameter nmeantimes must be either scalar or have the same length as the number of particles to try ({}): {}" \
                         .format(len(nparticles), nmeantimes))
    if is_scalar(nmeantimes):
        nmeantimes_list = [nmeantimes] * len(nparticles_list)
    else:
        nmeantimes_list = nmeantimes

    # Info parameters
    dict_params_info = {'plot': True, 'log': False}

    df_results = pd.DataFrame(columns=['K',
                                       'BSA',
                                       'N',
                                       'burnin_cycles',
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
    ncases = len(nparticles_list)

    # Compute the true blocking probability that is used to analyze the estimator quality
    print("Computing TRUE blocking probability for nservers={}, K={}, rhos={}...".format(nservers, K, rhos))
    proba_blocking_true = compute_blocking_probability_birth_death_process(rhos, K)

    print("System: # servers={}, K={}, rhos={}, buffer_size_activation={}, #burn-in absorption cycles={}" \
          .format(nservers, K, rhos, buffer_size_activation_value, burnin_cycles_absorption))
    time_start_all = timer()
    for case, (nparticles, nmeantimes) in enumerate(zip(nparticles_list, nmeantimes_list)):
        print("\n*** Running simulation for nparticles={} ({} of {}) on {} replications...".format(nparticles, case+1, ncases, replications))

        dict_params_simul = {
            'nparticles': nparticles,
            'nmeantimes': nmeantimes,
            'buffer_size_activation': buffer_size_activation_value,
            'burnin_cycles_absorption': burnin_cycles_absorption,
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
            if estimation_process == Process.Estimators:
                proba_blocking_fv, integral, expected_absorption_time, \
                    n_survival_curve_observations, n_absorption_time_observations, \
                        est_fv, est_abs, dict_stats_fv = estimators.estimate_blocking_fv(env_queue, agent,
                                                                                         dict_params_simul,
                                                                                         dict_params_info=dict_params_info)
                time_end_simulation_fv = dict_stats_fv['time']
                n_events_fv_only = dict_stats_fv['nevents']  # TOTAL number of events: _abs + (properly) _fv
                n_events_et = dict_stats_fv['nevents_abs']
                n_events_fv = n_events_et + n_events_fv_only
                max_survival_time = dict_stats_fv['time_max_survival']
            elif estimation_process == Process.Simulators:
                envs_queue = [env_queue if i == 0 else copy.deepcopy(env_queue) for i in range(nparticles)]
                dict_params_simul['T'] = dict_params_simul['nmeantimes']
                proba_blocking_fv, expected_reward, probas_stationary, \
                    expected_absorption_time, n_absorption_time_observations, \
                        time_last_absorption, time_end_simulation_et, max_survival_time, time_end_simulation_fv, \
                            n_events_et, n_events_fv_only = estimate_blocking_fv(envs_queue, agent,
                                                                                            dict_params_simul, dict_params_info)
                integral = np.nan
                n_events_fv = n_events_et + n_events_fv_only
                n_survival_curve_observations = n_absorption_time_observations

            exec_time_fv = timer() - time_start_fv

            if run_mc:
                time_start_mc = timer()
                print("\t--> Running Monte-Carlo estimation... {}".format(get_current_datetime_as_string()))
                dict_params_simul['maxevents'] = n_events_fv
                dict_params_simul['seed'] = seed_rep + 2  # This is the same seed used in the FV simulation in estimate_blocking_fv(), so we can compare better
                if estimation_process == Process.Estimators:
                    proba_blocking_mc, \
                        expected_return_time_mc, \
                            n_return_observations, \
                                est_mc, dict_stats_mc = estimators.estimate_blocking_mc(env_queue, agent, dict_params_simul,
                                                                                        dict_params_info=dict_params_info)
                    time_mc = dict_stats_mc.get('time'),
                    n_events_mc = dict_stats_mc.get('nevents', 0)
                elif estimation_process == Process.Simulators:
                    dict_params_simul['T'] = n_events_fv
                    proba_blocking_mc, expected_reward_mc, probas_stationary, \
                        expected_return_time_mc, n_return_observations, \
                            n_events_mc = estimate_blocking_mc(env_queue, agent, dict_params_simul, dict_params_info=dict_params_info)
                    time_mc = np.nan

                # Check comparability in terms of # events in each simulation (MC vs. FV)
                if n_events_mc != n_events_fv:
                    message = "!!!! #events(MC) != #events(FV) ({}, {}) !!!!".format(n_events_mc, n_events_fv)
                    print(message)  # Shown in the log
                    warn(message)   # Shown in the console

                exec_time_mc = timer() - time_start_mc
            else:
                proba_blocking_mc, expected_return_time_mc, n_return_observations, est_mc, dict_stats_mc = np.nan, None, None, None, {}
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
                if estimation_process == Process.Estimators:
                    print("\tP(K) by MC: {:.6f}% (simulation time = {:.1f} out of max={:.1f}, #events {} out of {})" \
                      .format(proba_blocking_mc*100, est_mc.get_simulation_time(), est_mc.getMaxSimulationTime(), est_mc.getNumberOfEvents(), est_mc.getMaxNumberOfEvents()))
                elif estimation_process == Process.Simulators:
                    print("\tP(K) by MC: {:.6f}% (#events {})" \
                      .format(proba_blocking_mc*100, n_events_mc))

            # FV results
            if estimation_process == Process.Estimators:
                print("\tP(K) by FV: {:.6f}%, E(T) = {:.1f} (simulation time for E(T) = {:.1f} ({} steps) (complete cycles span {:.1f}%),"
                        " max survival time = {:.1f}, #events = {} (ET) + {} (FV) = {})" \
                      .format(  proba_blocking_fv*100, expected_absorption_time, dict_stats_fv['time_abs'], dict_stats_fv['nevents_abs'],
                                expected_absorption_time*n_absorption_time_observations / dict_stats_fv['time_abs']*100,
                                dict_stats_fv['time_max_survival'],
                                dict_stats_fv['nevents_abs'], dict_stats_fv['nevents_fv'], dict_stats_fv['nevents']))
            elif estimation_process == Process.Simulators:
                print("\tP(K) by FV: {:.6f}%, E(T) = {:.1f} (simulation time for E(T) = {:.1f} ({} steps) (complete cycles span {:.1f}%),"
                        " max survival time = {:.1f}, #events = {} (ET) + {} (FV) = {})" \
                      .format(  proba_blocking_fv * 100, expected_absorption_time, time_end_simulation_et, n_events_et,
                                time_last_absorption/time_end_simulation_et*100,
                                max_survival_time,
                                n_events_et, n_events_fv_only, n_events_fv))
            print("\tTrue P(K): {:.6f}%".format(proba_blocking_true*100))

            # Store the results
            df_append = pd.DataFrame([[K,
                                       buffer_size_activation_value,
                                       nparticles,
                                       burnin_cycles_absorption,
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
            df_results = df_results.append(df_append)

        print("Results:")
        print(df_results)
    time_end_all = timer()

    print("Total execution time: {:.1f} min".format((time_end_all - time_start_all) / 60))
    if estimation_process == Process.Estimators:
        title = "Simulation results for #servers={}, K={}, rhos={}, ({}<=N<={}), T<={}, #Events<={}, Rep={}" \
            .format(nservers, K, rhos, nparticles_min, nparticles_max, est_fv.getMaxSimulationTime(), est_fv.getMaxNumberOfEvents(), replications)
    elif estimation_process == Process.Simulators:
        title = "Simulation results for #servers={}, K={}, rhos={}, ({}<=N<={}), T={}, #Events<={}, Rep={}" \
            .format(nservers, K, rhos, nparticles_min, nparticles_max, dict_params_simul['T'], n_events_fv, replications)
    print(title)
    print("Raw results by N:")
    print(df_results)

    df_results_agg_by_N = aggregation_bygroups(df_results, ['N'], ['#Events(MC)', '#Cycles(MC)', 'Pr(MC)', '#Events(FV)', 'Pr(FV)'])
    print("Aggregated results by N:")
    print(df_results_agg_by_N)

    # Add back the average of # events to the full data frame
    df_results = pd.merge(df_results, df_results_agg_by_N.xs('mean', axis=1, level=1)[['#Events(MC)', '#Cycles(MC)', '#Events(FV)']],
                          left_on='N', right_index=True, suffixes=["", "_mean"])
    # Convert average to integer
    if run_mc:
        df_results = df_results.astype({'#Events(MC)_mean': np.int})
        df_results = df_results.astype({'#Cycles(MC)_mean': np.int})
    df_results = df_results.astype({'#Events(FV)_mean': np.int})

    if estimation_process == Process.Estimators:
        return df_results, df_results_agg_by_N, est_fv, est_mc
    else:
        return df_results, df_results_agg_by_N, None, None

def analyze_absorption_size(nservers=1,
                            replications=5,
                            K_values=[10, 20, 30, 40],
                            nparticles_values=[200, 400, 800, 1600],
                            nmeantimes_values=[50, 50, 50, 50],
                            buffer_size_activation_values=[1],
                            burnin_cycles_absorption_values=[5],
                            seed=1717,
                            run_mc=True,
                            dict_params_info={'plot': True, 'log': False}):
    #--- Parse input parameters
    if is_scalar(burnin_cycles_absorption_values):
        burnin_cycles_absorption_values = [burnin_cycles_absorption_values] * len(buffer_size_activation_values)
    assert len(nparticles_values) == len(K_values), "The number of values in the nparticles parameter is the same as in K_values."
    assert len(nmeantimes_values) == len(K_values), "The number of values in the nmeantimes parameter is the same as in K_values."
    assert len(burnin_cycles_absorption_values) == len(buffer_size_activation_values), "The number of values in the burnin_cycles_absorption_values parameter is the same as in buffer_size_activation_values."
    #--- Parse input parameters

    #--- System setup
    dict_params = dict({'environment': {'capacity': np.Inf, # Set to Inf because the capacity is still undefined. This value will be updated below when analyzing each specific K.
                                        'nservers': nservers,
                                        'job_class_rates': [0.7],  # [0.8, 0.7]
                                        'service_rates': [1.0],  # [1.0, 1.0, 1.0]
                                        'policy_assignment_probabilities': [[1.0]], # [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]] )
                                        'reward_func': None, # Note: we should normally define a reward function when going from state S(t) -> S(t+1) by taking action A(t),
                                                             # but here we avoid defining it because we directly estimate the blocking probability,
                                                             # which is tantamount to receiving reward +1 when blocking.
                                        'rewards_accept_by_job_class': None
                                        },
                        'policy': {'parameterized_policy': PolQueueTwoActionsLinearStep,
                                   'theta': np.Inf      # Set to Inf because the capacity has been set to Inf. This value will be updated below when analyzing each specific K
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

    # Upper bound for the number of simulations to run
    # (it's an upper bound because when buffer_size_activation is a proportion, there may be repeated values for the actual buffer_size_activation used)
    nsimul = int(   len(K_values) * \
                    len(buffer_size_activation_values))

    print("System: # servers = {}, rhos = {}".format(env_queue.getNumServers(), rhos))

    np.random.seed(seed)
    time_start = timer()
    case = 0
    idx_K = -1
    for K in K_values:
        idx_K += 1

        # Set maximum capacity of queue to K
        env_queue.setCapacity(K)
        # Set deterministic acceptance policy that accepts up to K-1
        agent.getAcceptancePolicy().setThetaParameter(K-1)

        nparticles = nparticles_values[idx_K]
        nmeantimes = nmeantimes_values[idx_K]
        print("\n\n---> NEW K (Queue's capacity = {})".format(env_queue.getCapacity()))
        print("---> (nparticles={}, nmeantimes={})" \
              .format(nparticles, nmeantimes_values))

        print("Computing TRUE blocking probability...", end=" --> ")
        time_pr_start = timer()
        proba_blocking_true = compute_blocking_probability_birth_death_process(rhos, K)
        time_pr_end = timer()
        print("{:.1f} sec".format(time_pr_end - time_pr_start))
        print("Pr(K)={:.6f}%".format(proba_blocking_true*100))

        buffer_size_activation_value_prev = None
        for idx_bsa, buffer_size_activation in enumerate(buffer_size_activation_values):
            # When the buffer size for activation parameter is smaller than 1 it is considered a proportion of the queue's capacity
            if buffer_size_activation < 1:
                buffer_size_activation_value = max(1, int( round( buffer_size_activation * K ) ))
            else:
                buffer_size_activation_value = buffer_size_activation
                # Convert the buffer size activation into a proportion so that it appears
                # in the right place in the PLOT w.r.t. the other buffer_size_activation proportions
                # (and not e.g. to the RIGHT of all other proportion values,
                # which makes interpretation of the plot more difficult)
                buffer_size_activation = buffer_size_activation_value / K
            # Do not repeat the previous buffer size activation value (which may happen when the parameter is given as a proportion)
            if buffer_size_activation_value == buffer_size_activation_value_prev:
                continue
            burnin_cycles_absorption = burnin_cycles_absorption_values[idx_bsa]
            print("\n\t---> NEW BUFFER SIZE (BSA={}, BIC={})".format(buffer_size_activation_value, burnin_cycles_absorption))

            case += 1
            print("******************!!!!!!! Simulation {} of {} !!!!!!*****************\n\tK={}, nparticles={}, nmeantimes={}, buffer_size_activation={}, #burn-in absorption cycles={}" \
                  .format(case, nsimul, K, nparticles, nmeantimes, buffer_size_activation_value, burnin_cycles_absorption))
            for rep in range(replications):
                time_start_rep = timer()
                # NOTE THAT THE FIRST REPLICATION (rep=0) HAS THE SAME SEED FOR ALL PARAMETER SETTINGS
                # This is nice because we can a little bit better compare the effect of the different parameter settings
                # (but not so much anyway, because the values that are generated as the parameter settings change
                # impact whatever is generated next --e.g. the initial events for particle 2
                # will change if more events are generated for particle 1 when the simulation time increases...)
                print("\n\t\t### Replication {} of {} ###".format(rep+1, replications))
                seed_rep = seed + 10*rep
                    ## We multiply by 10 to leave enough "space" to assign seeds in between
                    ## two consecutive replications to assign to the different FV steps
                    ## (est_surv, est_abs, est_fv)

                dict_params_simul = {
                    'nparticles': nparticles,
                    'nmeantimes': nmeantimes,
                    'buffer_size_activation': buffer_size_activation_value,
                    'burnin_cycles_absorption': burnin_cycles_absorption,
                    'seed': seed_rep,
                        }

                print("\t\t*** FLEMING-VIOT ESTIMATION ***")
                dict_params_simul['maxevents'] = np.Inf
                proba_blocking_fv, integral, expected_absorption_time, \
                    n_survival_curve_observations, n_absorption_time_observations, \
                        est_fv, est_abs, dict_stats_fv = estimators.estimate_blocking_fv(env_queue, agent,
                                                                                         dict_params_simul,
                                                                                         dict_params_info=dict_params_info)

                if run_mc:
                    print("\t\t*** MONTE-CARLO ESTIMATION ***")
                    dict_params_simul['maxevents'] = dict_stats_fv['nevents']
                    dict_params_simul['seed'] = 1327*seed_rep
                    proba_blocking_mc, \
                        expected_return_time_mc, \
                            n_return_observations, \
                                est_mc, dict_stats_mc = estimators.estimate_blocking_mc(env_queue, agent, dict_params_simul, dict_params_info=dict_params_info)
                else:
                    proba_blocking_mc, expected_return_time_mc, n_return_observations, est_mc, dict_stats_mc = np.nan, None, None, None, {}

                # Show estimations
                if run_mc:
                    print("\t\tP(K) by MC: {:.6f}%".format(proba_blocking_mc*100))
                print("\t\tP(K) by FV (integral={:g} (n={}), E(T)={:.1f} (n={})): {:.6f}%".format(integral, n_survival_curve_observations, expected_absorption_time, n_absorption_time_observations, proba_blocking_fv*100))
                print("\t\tTrue P(K): {:.6f}%".format(proba_blocking_true*100))

                # Analyze the fairness of the comparison of results based on simulation time number of observed events
                print("-- FAIRNESS --")
                print("FV simulation:")
                #print("- time = {:.1f} (surv={:.1f} ({:.1f}%), abs={:.1f} ({:.1f}%), fv={:.1f} ({:.1f}%))" \
                #      .format(dict_stats_fv['time'], dict_stats_fv['time_surv'], dict_stats_fv['time_surv_prop']*100, \
                #                                     dict_stats_fv['time_abs'], dict_stats_fv['time_abs_prop']*100,
                #                                     dict_stats_fv['time_fv'], dict_stats_fv['time_fv_prop']*100))
                #print("- #events = {} (surv={} ({:.1f}%), abs={} ({:.1f}%), fv={} ({:.1f}%))" \
                #      .format(dict_stats_fv['nevents'], dict_stats_fv['nevents_surv'], dict_stats_fv['nevents_surv_prop']*100,
                #                                        dict_stats_fv['nevents_abs'], dict_stats_fv['nevents_abs_prop']*100,
                #                                        dict_stats_fv['nevents_fv'], dict_stats_fv['nevents_fv_prop']*100))
                print("- time = {:.1f} (avg = {:.1f} per particle)".format(dict_stats_fv['time'], dict_stats_fv['time'] / nparticles))
                print("- #events = {} (avg = {:.0f} per particle)".format(dict_stats_fv['nevents'], dict_stats_fv['nevents'] / nparticles))
                if run_mc:
                    print("MC simulation:\n- time = {:.1f}\n- #events = {}".format(dict_stats_mc['time'], dict_stats_mc['nevents']))
                    print("Ratio MC / FV: time={:.1f}, nevents={:.1f}".format(dict_stats_mc['time'] / dict_stats_fv['time'], dict_stats_mc['nevents'] / dict_stats_fv['nevents']))
                    if dict_stats_mc['nevents'] != dict_stats_fv['nevents']:
                        message = "!!!! #events(MC) != #events(FV) ({}, {}) !!!!".format(dict_stats_mc['nevents'], dict_stats_fv['nevents'])
                        print(message)  # Shown in the log
                        warn(message)   # Shown in the console

                # Add the observed measure to the output data frame with the results
                # Notes:
                # - We use the from_items() method as opposed to injecting  the data from a dictionary
                # in order to preserve the order of the columns!
                # - We need to enclose the value set for each column in [] in order to avoid the error message
                # "when passing all scalar values, you must pass an index",
                # which is NOT solved by using the parameter orient='columns', or at least it is not always solved...
                # I have no clue when that would work!
                # - There is no `index` parameter in the from_items() method so we need to set the index
                # value afterwards by calling the set_index() method in conjunction with a pd.Index() call.
                # Using set_index([1]) does NOT work because it says that the value 1 is an invalid key!!
                df_new_estimates = pd.DataFrame.from_items([
                                                            ('rhos', [len(rhos) == 1 and rhos or len(rhos) > 1 and format(rhos)]),
                                                            ('K', [K]),
                                                            ('N', [nparticles]),
                                                            ('nmeantimes', [nmeantimes]),
                                                            ('buffer_size_activation', [buffer_size_activation]),
                                                            ('buffer_size_activation_value', [buffer_size_activation_value]),
                                                            ('burnin_cycles_absorption', [burnin_cycles_absorption]),
                                                            ('rep', [rep+1]),
                                                            ('seed', [seed_rep]),
                                                            ('Pr(MC)', [proba_blocking_mc]),
                                                            ('EMC(T)', [expected_return_time_mc]),
                                                            ('time(MC)', [dict_stats_mc.get('time')]),
                                                            ('n(MC)', [dict_stats_mc.get('nevents')]),
                                                            ('n(RT)', [n_return_observations]),
                                                            ('Pr(FV)', [proba_blocking_fv]),
                                                            ('integral', [integral]),
                                                            ('E(T)', [expected_absorption_time]),
                                                            ('n(FV)', [dict_stats_fv['nevents']]),
                                                            ('n(PT)', [n_survival_curve_observations]),
                                                            ('n(ET)', [n_absorption_time_observations]),
                                                            ('Pr(K)', [proba_blocking_true]),
                                                            ('ratio_mc_fv_time', [dict_stats_mc.get('time') is None and np.nan or dict_stats_mc.get('time') / dict_stats_fv['time']]),
                                                            ('ratio_mc_fv_events', [dict_stats_mc.get('nevents') is None and np.nan or dict_stats_mc.get('nevents') / dict_stats_fv['nevents']]),
                                                        ]) #, orient='columns')

                # Survival probability and Phi(t) functions
                df_new_functions = est_fv.estimate_proba_survival_and_blocking_conditional()
                nobs = df_new_functions.shape[0]
                df_new_functions = pd.concat([  pd.DataFrame({'K': [K]*nobs,
                                                             'J': [buffer_size_activation_value]*nobs,
                                                             'rep': [rep+1]*nobs}),
                                                df_new_functions ],
                                                axis=1)
                # Plot the blue and red curves contributing to the integral used in the FV estimation
                if rep < 0: #<= 2:
                    estimators.plot_curve_estimates(df_new_functions,
                                                    dict_params={
                                                        'birth_rates': est_fv.queue.getBirthRates(),
                                                        'death_rates': est_fv.queue.getDeathRates(),
                                                        'K': est_fv.queue.getCapacity(),
                                                        'nparticles': dict_params_simul['nparticles'],
                                                        'nmeantimes': dict_params_simul['nmeantimes'],
                                                        'maxtime_mc': est_mc is not None and est_mc.getMaxSimulationTime() or 0.0,
                                                        'maxtime_fv': est_fv.getMaxSimulationTime(),
                                                        'buffer_size_activation': buffer_size_activation_value,
                                                        'mean_lifetime': expected_absorption_time,
                                                        'n_survival_curve_observations': n_survival_curve_observations,
                                                        'n_absorption_time_observations': n_absorption_time_observations,
                                                        'proba_blocking_fv': proba_blocking_fv,
                                                        'finalize_type': est_fv.getFinalizeType(),
                                                        'seed': seed_rep
                                                        })

                # Append results to output data frames
                if case == 1 and rep == 0:
                    # First loop iteration
                    # => Create the output data frames
                    df_proba_blocking_estimates = df_new_estimates
                    df_proba_survival_and_blocking_conditional = df_new_functions
                else:
                    # Update the output data frames with new rows
                    df_proba_blocking_estimates = pd.concat([df_proba_blocking_estimates,
                                                             df_new_estimates],
                                                             axis=0)
                    df_proba_survival_and_blocking_conditional = pd.concat([df_proba_survival_and_blocking_conditional,
                                                                            df_new_functions],
                                                                            axis=0)

                time_end_rep = timer()
                exec_time = time_end_rep - time_start_rep
                print("\n---> Execution time MC + FV: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))

            buffer_size_activation_value_prev = buffer_size_activation_value

        # Show the results obtained for the current K
        print("Simulation results for #servers={}, rhos={}, K={}, N={}, T={} ({}x), max #events={}" \
              .format(env_queue.getNumServers(), rhos, K, nparticles, est_fv.getMaxSimulationTime(), nmeantimes, est_fv.getMaxNumberOfEvents()))
        print(df_proba_blocking_estimates)

    # Correctly define the row indices from 0 to the number of records in each data frame
    df_proba_blocking_estimates.set_index( pd.Index(range(df_proba_blocking_estimates.shape[0])), inplace=True )
    df_proba_survival_and_blocking_conditional.set_index( pd.Index(range(df_proba_survival_and_blocking_conditional.shape[0])), inplace=True )

    time_end = timer()
    time_elapsed = time_end - time_start
    print("Execution time: {:.1f} sec, {:.1f} min, {:.1f} hours".format(time_elapsed, time_elapsed / 60, time_elapsed / 3600))

    print("Top and bottom 5 records in the results data frame:")
    print(df_proba_blocking_estimates.head())
    print(df_proba_blocking_estimates.tail())

    # Aggregate results
    df_proba_blocking_estimates_agg = aggregation_bygroups(df_proba_blocking_estimates,
                                                           ['K', 'buffer_size_activation', 'N'],
                                                           ['E(T)', 'Pr(MC)', 'Pr(FV)', 'Pr(K)'])

    return df_proba_blocking_estimates, df_proba_blocking_estimates_agg, \
                df_proba_survival_and_blocking_conditional, \
                    est_fv, est_mc

def plot_results_fv_mc(df_results, x, x2=None, xlabel=None, xlabel2=None, y2=None, ylabel2=None,
                       prob_fv="Pr(FV)", prob_mc="Pr(MC)", prob_true="Pr(K)",
                       splines=True, use_weights_splines=False,
                       smooth_params={'bias': None, 'variability': None, 'mse': None},
                       xmin=None, xmax=None, ymin=None, ymax=None,
                       subset=None,
                       plot_mc=True,
                       figfile=None):
    """
    Plots the estimated blocking probability by number of particles (FV) and average #Cycles (MC)

    Return: List of AxesSubplot objects of matplotlib.axes._subplots
    List of objects containing the axes of the error bar plot and of the violin plot.

    Arguments:
    df_results:

    x: str
        Name of the variable for the X-axis of the first plot (FV case).

    x2: str
        Name of the variable for the X-axis of the second plot (MC case).

    xlabel: str
        Label for the X-axis of the first plot.

    xlabel2: str
        Label for the X-axis of the second plot.

    y2: str
        Name of the variable for the secondary axis of the last variability plot (plot #6), which is only plotted
        when y2 is not None.
        The typical case is to show the complexity of the algorithm by plotting the average number of events seen
        by the algorithm.

    ylabel2: str
        Label for the secondary axis of the last variability plot (plot #6), which is ONLY plotted when y2 is not None.

    Example: This example uses a secondary axis on the last variability plot (#6) to show the complexity of the algorithm
    [Aug-2021]
    results['log(n(FV))'] = np.log10(results['n(FV)'])
    df_plotted, axes_error, axes_violin, axes_variability, axes_bias, axes_mse = \
        plot_results_fv_mc(results, "buffer_size_activation", xlabel="J as fraction of K",
                       y2="log(n(FV))", ylabel2="Avg. #events (log)",
                       plot_mc=False,
                       smooth_params={'bias': [1E2], 'variability': 1E3, 'mse': 1E-22},
                       xmin=0, xmax=1, ymin=0)
    """

    # --- Parse input parameters
    # What to plot
    if plot_mc:
        if x2 is None:
            x2 = x
        xvars = [x, x2]
        yvars = [prob_fv, prob_mc]
        figsize = (8, 4)
        subplots = (1, 2)
        nsubplots = 2
        colors = ["green", "red"]
    else:
        xvars = [x]
        yvars = [prob_fv]
        figsize = (4, 4)
        subplots = (1, 1)
        nsubplots = 1
        colors = ["green"]

    # Variables to plot
    if y2 == "":
        y2 = None

    # Axis limits
    axis_properties = {'limits': {}}
    if xmin is not None:
        axis_properties['limits']['xmin'] = xmin
    if xmax is not None:
        axis_properties['limits']['xmax'] = xmax
    if ymin is not None:
        axis_properties['limits']['ymin'] = ymin
    if ymax is not None:
        axis_properties['limits']['ymax'] = ymax

    # Axis labels
    if xlabel is None:
        xlabel = x
    if xlabel2 is None:
        xlabel2 = x2
    if ylabel2 is None:
        ylabel2 = y2

    if splines:
        plot_func_summarize = plotting.plot_splines
        points_properties = {'color': "black", 'color_line': colors}

        # Smoothing parameters for each of the 3 spline plots below
        assert isinstance(smooth_params, dict)
        for k in smooth_params.keys():
            if not isinstance(smooth_params[k], list):
                smooth_params[k] = [smooth_params[k]] * nsubplots
    else:
        plot_func_summarize = plotting.plot_errorbars
        points_properties = {'color': "black", 'marker': ".",
                             'color_center': "black", 'marker_center': "."}
        smooth_params = {'bias': [None] * nsubplots, 'variability': [None] * nsubplots, 'mse': [None] * nsubplots}

    # Rows to plot
    if subset is not None:
        df_results = df_results[subset]
    # --- Parse input parameters

    # 1) Average P(K) + error bars
    axes_error = plotting.plot(plotting.plot_errorbars,
                               df_results, xvars, yvars,
                               yref=prob_true, yref_legend="True value",
                               figsize=figsize, subplots=subplots,
                               dict_options={'axis': axis_properties,
                                             'multipliers': {'x': 1, 'y': 100, 'error': 2},
                                             'labels': {'x': [xlabel, xlabel2], 'y': "Blocking probability (%)"},
                                             'properties': {'color': "black", 'color_center': colors}})

    # 2) Violin plots
    axes_violin = plotting.plot(plotting.plot_violins,
                                df_results, xvars, yvars,
                                yref=prob_true, yref_legend="True value",
                                figsize=figsize, subplots=subplots,
                                dict_options={'axis': axis_properties,
                                              'multipliers': {'x': 1, 'y': 100},
                                              'labels': {'x': [xlabel, xlabel2], 'y': "Blocking probability (%)"},
                                              'properties': {'color': colors, 'color_center': colors}})

    # -- Compute variability and bias
    df2plot = pd.DataFrame()
    nvars = []
    weightvars = []
    biasvars = []
    bias2vars = []
    variancevars = []
    msevars = []
    madvars = []
    cvbiasvars = []
    cvmadvars = []
    vars2plot_bias = []
    vars2plot_variability = []
    vars2plot_mse = []
    for idx, (x, y) in enumerate(zip(xvars, yvars)):
        # Variability and bias
        if y2 is not None:
            yvars2agg = [y, y2]
        else:
            yvars2agg = [y]
        summary_stats = aggregation_bygroups(df_results, [x], yvars2agg,
                                             stats=["count", "mean", "median", "mad", "std", "var"])

        # Store the results in the data frame for plotting
        nvar = "n({})".format(y)
        weightvar = "w({})".format(y)
        biasvar = "b({})".format(y)
        bias2var = "b2({})".format(y)
        variancevar = "Var({})".format(y)
        msevar = "MSE({})".format(y)
        madvar = "MAD({})".format(y)
        cvbiasvar = "CVBIAS({})".format(y)
        cvstdvar = "CVSTD({})".format(y)
        cvmadvar = "CVMAD({})".format(y)
        cvrmsevar = "CVRMSE({})".format(y)
        if y2 is not None:
            y2meanvar = "mean({})".format(y2)
        df2plot[x] = summary_stats.index
        df2plot.set_index(summary_stats.index, inplace=True)
        df2plot[nvar] = summary_stats[y]["count"]
        df2plot[weightvar] = df2plot[nvar] / np.sum(df2plot[nvar])
        df2plot[biasvar] = summary_stats[y]["mean"] - df_results.iloc[0][prob_true]
        df2plot[bias2var] = df2plot[biasvar] ** 2
        df2plot[variancevar] = summary_stats[y]["var"]
        df2plot[msevar] = df2plot[biasvar] ** 2 + df2plot[variancevar]
        df2plot[madvar] = summary_stats[y]["mad"]
        df2plot[cvbiasvar] = np.abs(df2plot[biasvar]) / df_results.iloc[0][prob_true] * 100
        df2plot[cvstdvar] = np.sqrt(df2plot[variancevar]) / df_results.iloc[0][prob_true] * 100
        df2plot[cvmadvar] = df2plot[madvar] / df_results.iloc[0][prob_true] * 100
        df2plot[cvrmsevar] = np.sqrt(df2plot[msevar]) / df_results.iloc[0][prob_true] * 100
        if y2 is not None:
            df2plot[y2meanvar] = summary_stats[y2]["mean"]

        var2plot_bias = cvbiasvar  # bias2var
        var2plot_variability = cvstdvar  # cvmadvar # variancevar
        var2plot_mse = cvrmsevar

        # Weighted squared sum of Y values --> used when choosing a good smoothing parameter
        # The calculation is based on the documentation for parameter 's' in help(scipy.interpolate.slprep)
        if splines and use_weights_splines:
            # Normalized weights
            weights = df2plot[weightvar]
            weightvars += [weightvar]
        else:
            weights = 1
            weightvars += [None]
        if splines and smooth_params['bias'][idx] is None:
            smooth_params['bias'][idx] = np.mean(np.abs(df2plot[var2plot_bias])) * np.sum(
                (weights * df2plot[var2plot_bias]) ** 2)
        if splines and smooth_params['variability'][idx] is None:
            smooth_params['variability'][idx] = np.mean(np.abs(df2plot[var2plot_variability])) * np.sum(
                (weights * df2plot[var2plot_variability]) ** 2)  # / \
            # np.sum( df2plot[nvar]**2 ) * np.var( df2plot[var2plot_variability] )
        if splines and smooth_params['mse'][idx] is None:
            smooth_params['mse'][idx] = np.mean(np.abs(df2plot[var2plot_mse])) * np.sum(
                (weights * df2plot[var2plot_mse]) ** 2)
        if splines:
            print("Smoothing parameter for bias, variability, MSE for subplot {}:\n{}".format(idx + 1, smooth_params))
        nvars += [nvar]
        biasvars += [biasvar]
        bias2vars += [bias2var]
        variancevars += [variancevar]
        msevars += [msevar]
        madvars += [madvar]
        cvbiasvars += [cvbiasvar]
        cvmadvars += [cvmadvar]
        vars2plot_bias += [var2plot_bias]
        vars2plot_variability += [var2plot_variability]
        vars2plot_mse += [var2plot_mse]

    # 3) Variability plot
    if splines:
        splines_opt = {'weights': weightvars, 'smooth_par': smooth_params['variability']}
    else:
        splines_opt = {}
    axes_variability = plotting.plot(plot_func_summarize,
                                     df2plot, xvars, vars2plot_variability,
                                     subplots=subplots,
                                     dict_params={'pointlabels': nvars, 'splines': splines_opt},
                                     dict_options={'axis': axis_properties,
                                                   'multipliers': {'x': 1, 'y': 1},
                                                   'labels': {'x': [xlabel, xlabel2], 'y': "CV w.r.t. true Pr(K) (%)"},
                                                   'properties': points_properties,
                                                   'texts': {'title': "Relative variability of {}".format(y)}
                                                   })

    # 4) Bias plot
    if splines:
        splines_opt = {'weights': weightvars, 'smooth_par': smooth_params['bias']}
    else:
        splines_opt = {}
    axes_bias = plotting.plot(plot_func_summarize,
                              df2plot, xvars, vars2plot_bias,
                              subplots=subplots,
                              dict_params={'pointlabels': nvars, 'splines': splines_opt},
                              dict_options={'axis': axis_properties,
                                            'multipliers': {'x': 1, 'y': 1},
                                            'labels': {'x': [xlabel, xlabel2], 'y': "CV w.r.t. true Pr(K) (%)"},
                                            'properties': points_properties,
                                            'texts': {'title': "Relative bias of {}".format(y)}
                                            })

    # 5) RMSE = sqrt( Variance + Bias^2 )
    if splines:
        splines_opt = {'weights': weightvars, 'smooth_par': smooth_params['mse']}
    else:
        splines_opt = {}
    axes_mse = plotting.plot(plot_func_summarize,
                             df2plot, xvars, vars2plot_mse,
                             subplots=subplots,
                             dict_params={'pointlabels': nvars, 'splines': splines_opt},
                             dict_options={'axis': axis_properties,
                                           'multipliers': {'x': 1, 'y': 1},
                                           'labels': {'x': [xlabel, xlabel2], 'y': "RMSE"},
                                           'properties': points_properties,
                                           'texts': {'title': "Root Mean Squared Error of {}".format(y)}
                                           })

    # 6) Variability with the plot of a secondary axis (e.g. showing the "complexity" of the algorithm)
    if y2 is not None:
        axes = plt.figure().subplots(subplots[0], subplots[1])
        if not isinstance(axes, list):
            axes = [axes]
        for (ax, x, y, n, var2plot, w, s, color) in zip(axes, xvars, yvars, nvars, vars2plot_variability, weightvars,
                                                        smooth_params['variability'], colors):
            if splines:
                legend_obj, legend_txt = plotting.plot_splines(ax, df2plot, x, var2plot, w=w, s=s,
                                                               dict_options={'properties': {'color': "black",
                                                                                            'color_line': color}})
            else:
                points = plotting.pointsplot(ax, df2plot, x, var2plot, dict_options={'properties': points_properties})
                legend_obj = [points[0]]
                legend_txt = [var2plot]
            # Point labels showing sample size
            for (xx, yy, nn) in zip(df2plot[x], df2plot[var2plot], df2plot[n]):
                ax.text(xx, yy, nn)
            ax.set_title("Relative variability of {}".format(y))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("CV w.r.t. true Pr(K) (%)")
            ax2 = ax.twinx()
            sizes = ax2.plot(df2plot[x], df2plot[y2meanvar], 'r.-')
            ax2.set_ylabel(ylabel2)
            ax2.set_ylim([0, ax2.get_ylim()[1]])
            legend_obj += sizes
            legend_txt += [ylabel2]
            ax.legend(legend_obj, legend_txt)

    if figfile is not None:
        plt.gcf().subplots_adjust(left=0.15, top=0.75)
        plt.savefig(figfile)

    return df2plot, axes_error[0], axes_violin[0], axes_variability[0], axes_bias[0], axes_mse[0]

def createLogFileHandleAndResultsFileNames(path="../../RL-002-QueueBlocking", prefix="run"):
    """
    Redirects the standard output to a file which is used to log messages.
    Creates output filenames for raw results and aggregated results.

    Ref: https://www.stackabuse.com/writing-to-a-file-with-pythons-print-function/
    """

    dt_start = get_current_datetime_as_string()
    dt_suffix = get_current_datetime_as_string(format="suffix")
    logfile = "{}/logs/{}_{}.log".format(path, prefix, dt_suffix)
    resultsfile = "{}/results/{}_{}_results.csv".format(path, prefix, dt_suffix)
    resultsfile_agg = "{}/results/{}_{}_results_agg.csv".format(path, prefix, dt_suffix)
    proba_functions_file = "{}/results/{}_{}_proba_functions.csv".format(path, prefix, dt_suffix)
    figfile = re.sub("\.[a-z]*$", ".png", resultsfile)

    fh_log = open(logfile, "w")
    print("Log file '{}' has been open for output.".format(logfile))
    print("Started at: {}".format(dt_start))
    stdout_sys = sys.stdout
    sys.stdout = fh_log

    print("Started at: {}".format(dt_start))

    return dt_start, stdout_sys, fh_log, logfile, resultsfile, resultsfile_agg, proba_functions_file, figfile

def closeLogFile(fh_log, stdout_sys, dt_start):
    dt_end = get_current_datetime_as_string()
    print("Ended at: {}".format(dt_end))
    datetime_diff = get_datetime_from_string(dt_end) - get_datetime_from_string(dt_start)
    time_elapsed = datetime_diff.total_seconds()
    print("Execution time: {:.1f} min, {:.1f} hours".format(time_elapsed / 60, time_elapsed / 3600))

    fh_log.close()

    # Reset the standard output
    sys.stdout = stdout_sys
    print("Ended at: {}".format(dt_end))
    print("Execution time: {:.1f} min, {:.1f} hours".format(time_elapsed / 60, time_elapsed / 3600))

def save_dataframes(list_of_dataframes):
    i = -1
    for df2save in list_of_dataframes:
        i += 1
        df = df2save['df']
        if not isinstance(df, pd.DataFrame):
            print("Skipping variable {} in the list because it is not a data frame".format(i))
        else:
            filename = df2save['file']
            if filename is not None:
                df.to_csv(filename)
                print("Data frame saved to {}".format(os.path.abspath(filename)))
#------------------- Functions --------------------


#------------------- Execution starts -------------
# Default execution arguments when no arguments are given
# Example of execution from the command line:
# python run_FV.py 1 N 5 10 0.5 8 1
print("User arguments: {}".format(sys.argv))
if len(sys.argv) == 1:    # Only the execution file name is contained in sys.argv
    sys.argv += [1]       # Number of servers in the system to simulate
    sys.argv += ["N"]     # Type of analysis: either "N" for the impact of number of particles or "J" for the impact of buffer size"
    sys.argv += [40]      # K: capacity of the system
    sys.argv += [200]     # N: number of particles in the FV system.
    sys.argv += [0.5]     # J factor: factor such that J = factor*K.
    sys.argv += [3]       # Number of replications
    sys.argv += [1]       # Test number to run: only one is accepted
if len(sys.argv) == 8:    # Only the 6 required arguments are given by the user (recall that the first argument is the program name)
    sys.argv += ['None']  # T: number of arrival events to consider in the estimation of Pr(T>t) and E(T) in the FV approach. When 'None' it is chosen as 50*N.
    sys.argv += [2]       # Number of methods to run: 1 (only FV), 2 (FV & MC)
    sys.argv += ["nosave"]# Either "nosave" or anything else for saving the results and log
print("Parsed user arguments: {}".format(sys.argv))
print("")

#-- Parse user arguments
nservers = int(sys.argv[1])
analysis_type = sys.argv[2]
K = int(sys.argv[3])
N = int(sys.argv[4])
J = int(np.round(sys.argv[5] * K))
replications = int(sys.argv[6])
tests2run = [int(v) for v in [sys.argv[
                                  7]]]  # NOTE: It's important to enclose sys.argv[5] in brackets because o.w., from the command line, a number with more than one digit is interpreted as a multi-element list!! (e.g. 10 is interpreted as a list with elements [1, 0])
T = sys.argv[8]
if T.lower() == 'none':
    T = 50 * N
run_mc = int(sys.argv[9]) == 2
save_results = sys.argv[10] != "nosave"

if len(tests2run) == 0:
    print("No tests have been specified to run. Please specify the test number as argument 4.")
    sys.exit()

print(get_current_datetime_as_string())
print("Execution parameters:")
print("nservers={}".format(nservers))
print("Type of analysis: analysis_type={}".format(analysis_type))
print("Capacity K={}".format(K))
print("# particles N={}".format(N))
print("Activation size J={}".format(J))
print("Replications={}".format(replications))
print("tests2run={}".format(tests2run))
print("# arrival events T={}".format(T))
print("run_mc={}".format(run_mc))
print("save_results={}".format(save_results))

if analysis_type not in ["N", "J"]:
    raise ValueError("Valid values for the second parameter are 'N' or 'J'. Given: {}".format(analysis_type))
elif analysis_type == "N":
    resultsfile_prefix = "estimates_vs_N"
else:
    resultsfile_prefix = "estimates_vs_J"
#-- Parse user arguments

if save_results:
    dt_start, stdout_sys, fh_log, logfile, resultsfile, resultsfile_agg, proba_functions_file, figfile = \
        createLogFileHandleAndResultsFileNames(prefix=resultsfile_prefix)
else:
    fh_log = None;
    resultsfile = None;
    resultsfile_agg = None;
    proba_functions_file = None;
    figfile = None

if analysis_type == "N":
    # -- Single-server
    results, results_agg, est_fv, est_mc = analyze_convergence(
                                                estimation_process=Process.Simulators,
                                                nservers=nservers, K=40, buffer_size_activation=J,
                                                nparticles=[N],  # [800, 1600, 3200], #[10, 20, 40], #[24, 66, 179],
                                                nmeantimes=T,  # 50, #[170, 463, 1259],
                                                replications=replications, run_mc=run_mc,
                                                seed=1313)

    # results, results_agg, est_fv, est_mc = analyze_convergence(nservers=1, K=20, buffer_size_activation=0.25)
    # results, results_agg, est_fv, est_mc = analyze_convergence(nservers=1, K=20, buffer_size_activation=0.5)
    # results, results_agg, est_fv, est_mc = analyze_convergence(nservers=1, K=20, buffer_size_activation=0.75)
    # results, results_agg, est_fv, est_mc = analyze_convergence(nservers=1, K=20, buffer_size_activation=0.9)

    # results, results_agg, est_fv, est_mc = analyze_convergence(nservers=1, K=40, buffer_size_activation=0.5)

    # -- Multi-server
    # results, results_agg, est_fv, est_mc = analyze_convergence(nservers=3, K=5, buffer_size_activation=0.5, burnin_cycles_absorption=3, run_mc=False)
    # results, results_agg, est_fv, est_mc = analyze_convergence(nservers=3, K=10, buffer_size_activation=0.5)
    # results, results_agg, est_fv, est_mc = \
    #    analyze_convergence(nservers=3, K=20, buffer_size_activation=0.5, burnin_cycles_absorption=4,
    #                                             nparticles_min=800, nparticles_max=1600, nparticles_step_prop=1,
    #                                             nmeantimes=1400, replications=5,
    #                                             run_mc=run_mc, plotFlag=True)
    # results, results_agg, est_fv, est_mc = \
    #    analyze_convergence(nservers=3, K=20, buffer_size_activation=0.2, burnin_cycles_absorption=1,
    #                                             nparticles_min=400, nparticles_max=1200, nparticles_step_prop=1,
    #                                             nmeantimes=500, replications=5,
    #                                             run_mc=run_mc, plotFlag=True)
    # results, results_agg, est_fv, est_mc = analyze_convergence(nservers=3, K=30, buffer_size_activation=0.5)

    # results, results_agg, est_fv, est_mc = analyze_convergence(nservers=3, K=40, buffer_size_activation=0.25)
    # results, results_agg, est_fv, est_mc = analyze_convergence(nservers=3, K=40, buffer_size_activation=0.3, burnin_cycles_absorption=1)
    # results, results_agg, est_fv, est_mc = \
    #    analyze_convergence(nservers=3, K=40, buffer_size_activation=0.5, burnin_cycles_absorption=2,
    #                                             nparticles_min=400, nparticles_max=1600, nparticles_step_prop=1,
    #                                             nmeantimes=100000, replications=5,
    #                                             run_mc=run_mc, plotFlag=True)
    # results, results_agg, est_fv, est_mc = analyze_convergence(nservers=3, K=40, buffer_size_activation=0.7)

    # Save results
    save_dataframes([{'df': results, 'file': resultsfile},
                     {'df': results_agg, 'file': resultsfile_agg}])

    # Plot results
    axes = plot_results_fv_mc(results, x="N", x2="#Cycles(MC)_mean",
                              xlabel="# particles", xlabel2="# Cycles",
                              ymin=0.0, plot_mc=run_mc, splines=False)

    if fh_log is not None:
        closeLogFile(fh_log, stdout_sys, dt_start)
elif analysis_type == "J":
    # Info for plotting...
    x = "buffer_size_activation";
    xlabel = "J as fraction of K"
    if 1 in tests2run:
        K_values = [5, 5]  # [10, 20, 30, 40]
        results, results_agg, proba_functions, est_fv, est_mc = analyze_absorption_size(
                                                                    nservers=nservers,
                                                                    replications=replications,
                                                                    K_values=K_values,
                                                                    nparticles_values=[20, 40],  # [200, 400, 800, 1600],
                                                                    nmeantimes_values=[5000, 5000],  # [50, 50, 50, 50],
                                                                    buffer_size_activation_values=[0.25, 0.5],  # [1, 0.2, 0.4, 0.6, 0.8],
                                                                    burnin_cycles_absorption_values=[5, 3],
                                                                    seed=1313,
                                                                    run_mc=run_mc)
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg},
                         {'df': proba_functions, 'file': proba_functions_file}])
        for K in K_values:
            axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K']==K, plot_mc=run_mc)
    if 2 in tests2run:
        K_values = [10, 20]
        results, results_agg, proba_functions, est_fv, est_mc = analyze_absorption_size(
                                                                    nservers=nservers,
                                                                    replications=replications,
                                                                    K_values=K_values,
                                                                    nparticles_values=[200, 400],
                                                                    nmeantimes_values=[50, 50],
                                                                    buffer_size_activation_values=[1, 0.2, 0.4, 0.6, 0.8],
                                                                    burnin_cycles_absorption_values=[5, 5],
                                                                    seed=1313,
                                                                    run_mc=run_mc)
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg},
                         {'df': proba_functions, 'file': proba_functions_file}])
        for K in K_values:
            axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K'] == K, plot_mc=run_mc)
    if 3 in tests2run:
        K_values = [30, 40]
        results, results_agg, proba_functions, est_fv, est_mc = analyze_absorption_size(
                                                                    nservers=nservers,
                                                                    replications=replications,
                                                                    K_values=K_values,
                                                                    nparticles_values=[800, 1600],
                                                                    nmeantimes_values=[50, 50],
                                                                    buffer_size_activation_values=[1, 0.2, 0.4, 0.5, 0.7],
                                                                    burnin_cycles_absorption_values=[5, 5],
                                                                    seed=1313,
                                                                    run_mc=run_mc)
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg},
                         {'df': proba_functions, 'file': proba_functions_file}])
        for K in K_values:
            axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K'] == K, plot_mc=run_mc)
    if 4 in tests2run:
        K_values = [10]
        results, results_agg, proba_functions, est_fv, est_mc = analyze_absorption_size(
                                                                    nservers=nservers,
                                                                    replications=replications,
                                                                    K_values=K_values,
                                                                    nparticles_values=[400],
                                                                    nmeantimes_values=[50],
                                                                    buffer_size_activation_values=[0.1, 0.25, 0.5],
                                                                    burnin_cycles_absorption_values=[5],
                                                                    seed=1313,
                                                                    run_mc=run_mc)
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg},
                         {'df': proba_functions, 'file': proba_functions_file}])
        for K in K_values:
            axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K'] == K, plot_mc=run_mc)
    if 5 in tests2run:
        K_values = [20]
        results, results_agg, proba_functions, est_fv, est_mc = analyze_absorption_size(
                                                                    nservers=nservers,
                                                                    replications=replications,
                                                                    K_values=K_values,
                                                                    nparticles_values=[3200],
                                                                    nmeantimes_values=[50],
                                                                    buffer_size_activation_values=[0.2, 0.4, 0.5, 0.6, 0.8],
                                                                    burnin_cycles_absorption_values=[5],
                                                                    seed=1313,
                                                                    run_mc=run_mc)
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg},
                         {'df': proba_functions, 'file': proba_functions_file}])
        for K in K_values:
            axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K'] == K, plot_mc=run_mc)
    if 6 in tests2run:
        K_values = [30]
        results, results_agg, proba_functions, est_fv, est_mc = analyze_absorption_size(
                                                                    nservers=nservers,
                                                                    replications=replications,
                                                                    K_values=K_values,
                                                                    nparticles_values=[800],
                                                                    nmeantimes_values=[50],
                                                                    buffer_size_activation_values=[0.1, 0.25, 0.5],
                                                                    burnin_cycles_absorption_values=[5],
                                                                    seed=1313,
                                                                    run_mc=run_mc)
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg},
                         {'df': proba_functions, 'file': proba_functions_file}])
        for K in K_values:
            axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K'] == K, plot_mc=run_mc)
    if 7 in tests2run:
        K_values = [40]
        results, results_agg, proba_functions, est_fv, est_mc = analyze_absorption_size(
                                                                    nservers=nservers,
                                                                    replications=replications,
                                                                    K_values=K_values,
                                                                    nparticles_values=[3200],
                                                                    nmeantimes_values=[8E6],
                                                                    buffer_size_activation_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                                                    burnin_cycles_absorption_values=[4, 3, 3, 3, 2, 1, 1, 1, 1],
                                                                    seed=1313,
                                                                    run_mc=run_mc)
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg},
                         {'df': proba_functions, 'file': proba_functions_file}])
        for K in K_values:
            axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K'] == K, plot_mc=run_mc)
    if 8 in tests2run:
        # Same as 7 but for small J values (to see if the variance of the estimator increases first and then decreases)
        K_values = [40]
        results, results_agg, proba_functions, est_fv, est_mc = analyze_absorption_size(
                                                                    nservers=nservers,
                                                                    replications=replications,
                                                                    K_values=K_values,
                                                                    nparticles_values=[3200],
                                                                    nmeantimes_values=[8E6],
                                                                    buffer_size_activation_values=[1, 2, 3, 4, 5, 6, 7, 8],
                                                                    burnin_cycles_absorption_values=[4, 4, 4, 4, 4, 4, 4, 4],
                                                                    seed=1313,
                                                                    run_mc=run_mc)
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg},
                         {'df': proba_functions, 'file': proba_functions_file}])
        for K in K_values:
            axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K'] == K, plot_mc=run_mc)
    if 9 in tests2run:
        # Larger K value for multi-server, where MC is expected to fail
        K_values = [60]
        results, results_agg, proba_functions, est_fv, est_mc = analyze_absorption_size(
                                                                    nservers=nservers,
                                                                    replications=replications,
                                                                    K_values=K_values,
                                                                    nparticles_values=[6400],
                                                                    nmeantimes_values=[1000],  # [8E7],
                                                                    buffer_size_activation_values=[1, 3, 5, 7, 12, 15, 21, 24],  # [1, 3, 5, 7, 9, 12, 15, 18, 21, 24],
                                                                    burnin_cycles_absorption_values=[3, 3, 3, 3, 2, 2, 2, 2],  # [3, 3, 3, 3, 3, 2, 2, 2, 2, 2]
                                                                    seed=1313,
                                                                    run_mc=run_mc)
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg},
                         {'df': proba_functions, 'file': proba_functions_file}])
        for K in K_values:
            axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K'] == K, plot_mc=run_mc)
    if 10 in tests2run:
        # Large K value for multi-server, with limited simulation time before
        K_values = [40]
        results, results_agg, proba_functions, est_fv, est_mc = analyze_absorption_size(
                                                                    nservers=nservers,
                                                                    replications=replications,
                                                                    K_values=K_values,
                                                                    nparticles_values=[3200],  # [1000],   #[5]
                                                                    nmeantimes_values=[1000],  # [1000],   #[100]
                                                                    buffer_size_activation_values=[0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.35, 0.4, 0.5, 0.6],
                                                                    # [0.1, 0.3, 0.5, 0.7], #[0.1, 0.3, 0.4, 0.5, 0.6, 0.8],
                                                                    burnin_cycles_absorption_values=[0],  # [3, 3, 2, 1],#[3, 3, 2, 2, 1, 1], #[3, 3, 3, 2, 2, 1],
                                                                    seed=1313,
                                                                    run_mc=run_mc,
                                                                    dict_params_info={'plot': False, 'log': False})
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg},
                         {'df': proba_functions, 'file': proba_functions_file}])
        # for K in K_values:
        #    axes = plot_results_fv_mc(results, "buffer_size_activation", xlabel="J as fraction of K",
        #                              subset=results['K']==K,
        #                              plot_mc=run_mc,
        #                              smooth_params={'bias': [1E2], 'variability': 1E3, 'mse': 1E-22})

    if fh_log is not None:
        closeLogFile(fh_log, stdout_sys, dt_start)
