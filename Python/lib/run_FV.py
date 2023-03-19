# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 12:15:39 2022

@author: Daniel Mastropietro
@description: Runs the FV estimation of the average reward.
For queues, and when the reward (cost) for blocking is set equal to 1, it estimates the blocking probability.
"""

import runpy
runpy.run_path('../../setup.py')

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

from Python.lib.agents.learners import LearnerTypes
from Python.lib.agents.learners.continuing.mc import LeaMC
from Python.lib.agents.learners.continuing.fv import LeaFV
from Python.lib.agents.policies import PolicyTypes
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep
from Python.lib.agents.queues import AgeQueue

from Python.lib.environments.queues import rewardOnJobRejection_Constant

import Python.lib.estimators as estimators

from Python.lib.simulators.queues import compute_nparticles_and_narrivals_for_fv_process, \
    compute_nparticles_and_nsteps_for_fv_process_many_settings, compute_rel_errors_for_fv_process, \
    define_queue_environment_and_agent, estimate_blocking_fv, estimate_blocking_mc, SurvivalProbabilityEstimation

from Python.lib.utils.basic import aggregation_bygroups, get_current_datetime_as_string, get_datetime_from_string, \
    is_scalar, measure_exec_time, set_pandas_options, reset_pandas_options
from Python.lib.utils.computing import compute_blocking_probability_birth_death_process
import Python.lib.utils.plotting as plotting


@unique
class Process(Enum):
    "What process is used by Fleming-Viot, the one defined in estimators.py or the one defined in simulators/queues.py"
    Estimators = 1
    Simulators = 2


#------------------- Functions --------------------
@measure_exec_time
def run_mc_estimation_single_server(env_queue, K, J, T,
                                    burnin_time_steps, min_num_cycles_for_expectations,
                                    seed=1717):
    """
    Test the Monte-Carlo estimation of the blocking probability of a single-server system

    Arguments:
    env_queue: environment
        A queue environment out of those defined in environments/queues.py.

    K: int
        Capacity of the queue.

    J: int
        Smallest buffer size of the queue where the queue is still active (i.e. not absorbed).
        Absorption happens at buffer size = J - 1.

    N: int
        Number of particles of the Fleming-Viot system.

    T: int
        Number of arrivals at which the simulation stops.

    seed: int
        Seed to use for the pseudo-random number generation.
    """
    print("\n--> Running Monte-Carlo estimation on a single server system...")

    # Agent interacting with the environment (which defines whether to accept or reject a job)
    # Define the agent acting on the queue environment
    # The agent blocks an incoming job when the buffer size of the queue is at its capacity K.
    # This is achieved by setting the parameter theta of the parameterized linear step acceptance policy
    # to the integer value K-1.
    policies = dict({PolicyTypes.ACCEPT: PolQueueTwoActionsLinearStep(env_queue, float(K - 1)),
                     PolicyTypes.ASSIGN: None})
    learners = dict({LearnerTypes.V: LeaMC(env_queue, gamma=1.0),
                     LearnerTypes.Q: None,
                     LearnerTypes.P: None})
    agent_accept_reject = AgeQueue(env_queue, policies, learners)

    # Simulation parameters
    dict_params_simul = dict({'buffer_size_activation': J,  # J-1 is the absorption buffer size
                              'T': T,  # number of arrivals at which the simulation stops
                              'burnin_time_steps': burnin_time_steps,
                              'min_num_cycles_for_expectations': min_num_cycles_for_expectations,
                              'seed': seed})
    dict_params_info = dict()

    # Run the simulation!
    return estimate_blocking_mc(env_queue, agent_accept_reject,
                                dict_params_simul, dict_params_info)


@measure_exec_time
def run_fv_estimation_single_server(  env_queue, K, J, N, T,
                                      burnin_time_steps, min_num_cycles_for_expectations, method_survival_probability_estimation,
                                      seed=1717):
    """
    Test the Fleming-Viot estimation of the blocking probability of a single-server system

    Arguments:
    env_queue: environment
        A queue environment out of those defined in environments/queues.py.

    K: int
        Capacity of the queue.

    J: int
        Smallest buffer size of the queue where the queue is still active (i.e. not absorbed).
        Absorption happens at buffer size = J - 1.

    N: int
        Number of particles of the Fleming-Viot system.

    T: int
        Number of arrivals at which the simulation stops.

    seed: int
        Seed to use for the pseudo-random number generation.
    """
    print("\n--> Running Fleming-Viot estimation on a single server system...")

    # Queue environments to use as FV particles
    envs_queue = [env_queue if i == 0 else copy.deepcopy(env_queue) for i in range(N)]

    # Agent interacting with the environment (which defines whether to accept or reject a job)
    # Define the agent acting on the queue environment
    # The agent blocks an incoming job when the buffer size of the queue is at its capacity K.
    # This is achieved by setting the parameter theta of the parameterized linear step acceptance policy
    # to the integer value K-1.
    policies = dict({PolicyTypes.ACCEPT: PolQueueTwoActionsLinearStep(env_queue, float(K - 1)),
                     PolicyTypes.ASSIGN: None})
    learners = dict({LearnerTypes.V: LeaFV(env_queue, gamma=1.0),
                     LearnerTypes.Q: None,
                     LearnerTypes.P: None})
    agent_accept_reject = AgeQueue(env_queue, policies, learners)

    # Simulation parameters
    dict_params_simul = dict({'buffer_size_activation': J,  # J-1 is the absorption buffer size
                              'T': T,  # number of arrivals at which the simulation stops
                              'burnin_time_steps': burnin_time_steps,
                              'min_num_cycles_for_expectations': min_num_cycles_for_expectations,
                              'method_survival_probability_estimation': method_survival_probability_estimation,
                              'seed': seed})
    dict_params_info = dict()

    # Run the simulation!
    return estimate_blocking_fv(envs_queue, agent_accept_reject,
                                dict_params_simul, dict_params_info)


def analyze_convergence(estimation_process=Process.Simulators,
                        nservers=1, job_class_rates=[0.7], service_rates=[1.0],
                        K=5, buffer_size_activation=1,
                        burnin_time_steps=20, burnin_cycles_absorption=5,
                        min_num_cycles_for_expectations=5,
                        method_proba_surv=SurvivalProbabilityEstimation.FROM_N_PARTICLES,
                        nparticles=[],
                        nparticles_min=40, nparticles_max=80, nparticles_step_prop=1,
                        nmeantimes=500,
                        replications=5,
                        run_mc=True,
                        seed=1717):
    """
    2021/04/19: Analyze convergence of the FV estimation of the blocking probability as the number of particles N OR
    the number of arrival times (T = `nmeantimes`) increases.

    Arguments:
    estimation_process: (opt) Process
        The estimation process that should be used to estimate the blocking probability, whether the estimator
        defined in estimators.py (Process.Estimators) or the estimator defined in simulators/queues.py
        (Process.Simulators).
        default: Process.Simulators

    burnin_time_steps: (opt) int
        Number of burn-in time steps to consider before a return cycle and a reabsorption cycle qualifies for
        the estimation of expectations. This is used when the estimation is performed by the estimators
        defined in simulators.queues (NEW implementation) (e.g. estimate_blocking_mc() and estimate_blocking_fv()).
        default: 20

    burnin_cycles_absorption: (opt) int
        Number of burn-in absorption cycles to consider before an absorption cycle qualifies for the estimation of
        expectations. This is used when the estimation is performed by the estimator EstimatorQueueBlockingFlemingViot
        (OLD implementation).
        default: 5

    min_num_cycles_for_expectations: (opt) int
        Minimum number of observed cycles in order to estimate expectations (instead of setting their values to NaN).
        default: 5

    method_proba_surv: (opt) SurvivalProbabilityEstimation
        Method used to estimate the survival probability P(T>t), whether "from M cycles observed by a single particle"
        or "from the N particles used in the FV simulation".
        default: SurvivalProbabilityEstimation.FROM_N_PARTICLES

    nparticles: (opt) int or list or numpy array
        List giving the number of particles to consider in each simulation.
        If given, it takes precedence over parameters nparticles_min and nparticles_max.
        default: []

    nparticles_step_prop: (opt) positive float
        Step proportion: N(n+1) = (1 + prop)*N(n),
        so that we scale the step as the number of particles increases.
        default: 1

    nmeantimes: (opt) int or list or numpy array
        Number of discrete *time steps* to run the queue to estimate the blocking probability in Monte-Carlo
        or the number of *arrival events* to estimate E(T_A) for the FV estimator.
        If not scalar, either nparticles should be a scalar or be a list with just one element, o.w. it should have
        the same number of elements as the number of particles that are tried in this simulation.
        default: 500
    """
    assert nservers == len(service_rates), "The number of servers coincides with the number of service rates given"

    #--- System setup
    dict_params = dict({'environment': {'queue_system': "single-server",
                                        'capacity': K,
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
                        'policy': { 'parameterized_policy': PolQueueTwoActionsLinearStep if is_scalar(K) else [PolQueueTwoActionsLinearStep for _ in K],
                                    'theta': float(K-1) if is_scalar(K) else [float(KK-1) for KK in K]  # This means that there blocking is deterministic at K and otherwise there is acceptance.
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
    # number of particles (nparticles), or in terms of an increasing number of arrival events (nmeantimes), or both.
    # For example, if an analysis on incrcarring number of particles is done, then we can:
    # - Set K to a small value (e.g. K=5)
    # - Consider increasing values of nparticles (e.g. nparticles=[200, 400, 800])
    # - Set a large simulation time (e.g. nmeantimes=500)
    # We should check that the error between the estimated and true blocking probability decreases as 1/sqrt(N)
    # (according to Propostion 2.1 in Matt's draft, although this proposition applies to the estimation of Phi(t,K)
    # NOT to the estimation of the blocking probability... In fact, we prove in our paper sent and accepted at EWRL-2022)
    # that the blocking probability converges to the true blocking probability as 1/sqrt(M) + 1/sqrt(N) where M is the
    # number of cycles observed and used for the estimation of E(T_A), the denominator of the blocking probability.
    # This is why also an convergence analysis on increasing number of arrival events (nmeantimes) is sensible.

    #--------------------------- Parse input parameters -----------------------
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
        if is_scalar(nmeantimes):
            # There is only one value of nparticles and nmeantimes to try
            nparticles_list = [nparticles]
        else:
            # The analysis is done for different nmeantimes values
            # => Replicate the single nparticles value as many times as the number of nmeantimes values to try
            nparticles_list = [nparticles] * len(nmeantimes)
    else:
        # nparticles is a list or array
        nparticles_list = list(nparticles)

    if is_scalar(nmeantimes):
        nmeantimes_list = [nmeantimes] * len(nparticles_list)
    else:
        # nmeantimes is a list or array
        nmeantimes_list = list(nmeantimes)

    # Check if either of the input parameters is a list of length 1, in which case we just replicate the single value
    # as many times as values in the other parameter
    if len(nparticles_list) == 1 and len(nmeantimes_list) > 1:
        nparticles_list = [nparticles_list[0]] * len(nmeantimes_list)
    if len(nmeantimes_list) == 1 and len(nparticles_list) > 1:
        nmeantimes_list = [nmeantimes_list[0]] * len(nparticles_list)

    # Check
    if len(nmeantimes_list) != len(nparticles_list):
        raise ValueError("When both parameters nmeantimes and nparticles are lists they must have the same length:\nnparticles={}\nnmeantimes={}" \
                         .format(nparticles, nmeantimes))

    # Info parameters
    dict_params_info = {'plot': True, 'log': False}
    #--------------------------- Parse input parameters -----------------------

    # Modify pandas display options so that data frames are not truncated
    pandas_options = set_pandas_options()

    df_results = pd.DataFrame(columns=['nservers',
                                       'rhos',
                                       'K',
                                       'J',
                                       'N',
                                       'T',
                                       'burnin_time_steps',
                                       'burnin_cycles',
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
    ncases = len(nparticles_list)

    # Compute the true blocking probability that is used to analyze the estimator quality
    print("Computing TRUE blocking probability for nservers={}, K={}, rhos={}...".format(nservers, K, rhos))
    proba_blocking_true = compute_blocking_probability_birth_death_process(rhos, K)

    print("System: # servers={}, K={}, rhos={}, buffer_size_activation={}, #burn-in time steps={}, #burn-in absorption cycles={}" \
          .format(nservers, K, rhos, buffer_size_activation_value, burnin_time_steps, burnin_cycles_absorption))
    time_start_all = timer()
    for case, (nparticles, nmeantimes) in enumerate(zip(nparticles_list, nmeantimes_list)):
        print("\n*** Running simulation for nparticles={} ({} of {}) on {} replications...".format(nparticles, case+1, ncases, replications))

        dict_params_simul = {
            'nparticles': nparticles,
            'nmeantimes': nmeantimes,
            'buffer_size_activation': buffer_size_activation_value,
            'burnin_time_steps': estimation_process == Process.Estimators and np.nan or \
                                 estimation_process == Process.Simulators and burnin_time_steps,
            'burnin_cycles_absorption': estimation_process == Process.Estimators and burnin_cycles_absorption or \
                                        estimation_process == Process.Simulators and np.nan,
            'min_num_cycles_for_expectations': estimation_process == Process.Estimators and np.nan or \
                                               estimation_process == Process.Simulators and min_num_cycles_for_expectations,
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
                # Make a copy of the simulation parameters so that we do not alter the simulation parameters that are in use for the FV simulation which defines the benchmark
                dict_params_simul_mc = copy.deepcopy(dict_params_simul)
                dict_params_simul_mc['maxevents'] = n_events_fv
                if estimation_process == Process.Estimators:
                    proba_blocking_mc, \
                        expected_return_time_mc, \
                            n_return_observations, \
                                est_mc, dict_stats_mc = estimators.estimate_blocking_mc(env_queue, agent, dict_params_simul_mc,
                                                                                        dict_params_info=dict_params_info)
                    time_mc = dict_stats_mc.get('time'),
                    n_events_mc = dict_stats_mc.get('nevents', 0)
                elif estimation_process == Process.Simulators:
                    dict_params_simul_mc['T'] = n_events_fv
                    proba_blocking_mc, expected_reward_mc, probas_stationary, n_cycles, \
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
            df_append = pd.DataFrame([[ nservers,
                                        len(rhos) == 1 and rhos or len(rhos) > 1 and format(rhos),
                                        K,
                                        buffer_size_activation_value,
                                        nparticles,
                                        nmeantimes,
                                        dict_params_simul['burnin_time_steps'],
                                        dict_params_simul['burnin_cycles_absorption'],
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
    if estimation_process == Process.Estimators:
        title = "Simulation results for #servers={}, K={}, rhos={}, ({}<=N<={}), T<={}, #Events<={}, Rep={}" \
            .format(nservers, K, rhos, nparticles_min, nparticles_max, est_fv.getMaxSimulationTime(), est_fv.getMaxNumberOfEvents(), replications)
    elif estimation_process == Process.Simulators:
        title = "Simulation results for #servers={}, K={}, rhos={}, ({}<=N<={}), T={}, #Events<={}, Rep={}" \
            .format(nservers, K, rhos, nparticles_min, nparticles_max, dict_params_simul['T'], n_events_fv, replications)
    print(title)
    print("Raw results by N:")
    print(df_results)

    df_results_agg_by_NT = aggregation_bygroups(df_results, ['N', 'T'], ['#Events(MC)', '#Cycles(MC)', 'Pr(MC)', '#Events(FV)', 'Pr(FV)'])
    print("Aggregated results by N:")
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

    if estimation_process == Process.Estimators:
        return df_results, df_results_agg_by_NT, est_fv, est_mc
    else:
        return df_results, df_results_agg_by_NT, None, None

# TODO: (2022/10/17) Adapt the process to the new (simpler) implementation of the FV estimator done in simulators.queues
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
    dict_params = dict({'environment': {'queue_sytem': "single-server",
                                        'capacity': np.Inf, # Set to Inf because the capacity is still undefined. This value will be updated below when analyzing each specific K.
                                        'nservers': nservers,
                                        'job_class_rates': [0.7],  # [0.8, 0.7]
                                        'service_rates': [1.0],  # [1.0, 1.0, 1.0]
                                        'policy_assignment_probabilities': [[1.0]], # [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]] )
                                        'reward_func': rewardOnJobRejection_Constant,
                                            # Note: this reward function is defined here just for informational purposes
                                            # (e.g. when showing debug messages when DEBUG_ESTIMATORS = True in the simulator class)
                                            # but the process actually simply estimates the blocking probability,
                                            # i.e. it does NOT estimate the average reward (although with this definition
                                            # of reward, it could).
                                        'rewards_accept_by_job_class': None
                                        },
                        'policy': { 'parameterized_policy': PolQueueTwoActionsLinearStep if is_scalar(K_values[0]) else [PolQueueTwoActionsLinearStep for _ in K_values[0]],
                                    'theta': np.Inf if is_scalar(K_values[0]) else [np.Inf for KK in K_values[0]]  # Set to Inf because the capacity has been set to Inf. This value will be updated below when analyzing each specific K
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
        agent.setAcceptancePolicyThresholds(K-1)

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
            print("\n\t---> NEW BUFFER SIZE (J={}, BIC={})".format(buffer_size_activation_value, burnin_cycles_absorption))

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
                # TODO: (2022/11/04) Change the way the data frame is created to the method used in analyze_convergence(), which does NOT use from_items()
                df_new_estimates = pd.DataFrame.from_items([('nservers', nservers),
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

def plot_results_fv_mc(df_results, x, x2=None, xlabel=None, xlabel2=None, y2=None, ylabel="Blocking probability (%)", ylabel2=None,
                       prob_fv="Pr(FV)", prob_mc="Pr(MC)", prob_true="Pr(K)",
                       splines=True, use_weights_splines=False,
                       smooth_params={'bias': None, 'variability': None, 'mse': None},
                       xmin=None, xmax=None, ymin=None, ymax=None,
                       title=None,
                       subset=None,
                       plot_mc=True,
                       plot_violin_only=True,
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

    title: str
        Title to show in the first plot with the estimated values by each method.
        The other subsequent plots have their title already set.

    subset: boolean list or numpy array
        Subset of records in `df_results` to include in the plot.

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
    if not plot_violin_only:
        axes_error = plotting.plot(plotting.plot_errorbars,
                                   df_results, xvars, yvars,
                                   yref=prob_true, yref_legend="True value",
                                   figsize=figsize, subplots=subplots,
                                   dict_options={'axis': axis_properties,
                                                 'multipliers': {'x': 1, 'y': 100, 'error': 2},
                                                 'labels': {'x': [xlabel, xlabel2], 'y': ylabel},
                                                 'properties': {'color': "black", 'color_center': colors},
                                                 'texts': {'title': title}})

    # 2) Violin plots
    axes_violin = plotting.plot(plotting.plot_violins,
                                df_results, xvars, yvars,
                                yref=prob_true, yref_legend="True value",
                                figsize=figsize, subplots=subplots,
                                dict_options={'axis': axis_properties,
                                              'multipliers': {'x': 1, 'y': 100},
                                              'labels': {'x': [xlabel, xlabel2], 'y': ylabel},
                                              'properties': {'color': colors, 'color_center': colors},
                                              'texts': {'title': title}})

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
    for idx, (xx, y) in enumerate(zip(xvars, yvars)):
        # Variability and bias
        if y2 is not None:
            yvars2agg = [y, y2]
        else:
            yvars2agg = [y]
        if idx == 0:
            # We are plotting the FV results
            # => xx = x and we should group just by that variable
            assert x == xx
            groupvars_ = [xx]
        else:
            # This is the case when plotting the MC results
            # => We should group by x and xx in case the average number of return cycles plotted on the x-axis
            # for the MC results is the same for more than one value of the analyzed x value shown on the FV results
            # plot (e.g.  # particles). This already happened once for N = 10 and 22, we got #Cycles(MC)_mean = 53
            # (namely today 15-Dec-2022).
            groupvars_ = [x, xx]

        summary_stats = aggregation_bygroups(df_results, groupvars_, yvars2agg,
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

        # The following call to reset_index() removes the possibly multi-index (when processing the MC results)
        # in summary_stats and stores those values as COLUMNS of the data frame.
        # This is done in order to store the values of the currently analyzed variable `xx`
        # (to be shown on the x-axis) as a column in the df2plot data frame.
        summary_stats.reset_index(inplace=True)
        df2plot[xx] = summary_stats[xx]
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
    if not plot_violin_only:
        axes_variability = plotting.plot(plot_func_summarize,
                                         df2plot, xvars, vars2plot_variability,
                                         subplots=subplots,
                                         dict_params={'pointlabels': nvars, 'splines': splines_opt},
                                         dict_options={'axis': axis_properties,
                                                       'multipliers': {'x': 1, 'y': 1},
                                                       'labels': {'x': [xlabel, xlabel2], 'y': "CV w.r.t. true " + prob_true + " (%)"},
                                                       'properties': points_properties,
                                                       'texts': {'title': "Relative variability of {}".format(y)}
                                                       })

    # 4) Bias plot
    if splines:
        splines_opt = {'weights': weightvars, 'smooth_par': smooth_params['bias']}
    else:
        splines_opt = {}
    if not plot_violin_only:
        axes_bias = plotting.plot(plot_func_summarize,
                                  df2plot, xvars, vars2plot_bias,
                                  subplots=subplots,
                                  dict_params={'pointlabels': nvars, 'splines': splines_opt},
                                  dict_options={'axis': axis_properties,
                                                'multipliers': {'x': 1, 'y': 1},
                                                'labels': {'x': [xlabel, xlabel2], 'y': "CV w.r.t. true " + prob_true + " (%)"},
                                                'properties': points_properties,
                                                'texts': {'title': "Relative bias of {}".format(y)}
                                                })

    # 5) RMSE = sqrt( Variance + Bias^2 )
    if splines:
        splines_opt = {'weights': weightvars, 'smooth_par': smooth_params['mse']}
    else:
        splines_opt = {}
    if not plot_violin_only:
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
    if not plot_violin_only:
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
                ax.set_ylabel("CV w.r.t. true " + prob_true + " (%)")
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

    if not plot_violin_only:
        return df2plot, axes_error[0], axes_violin[0], axes_variability[0], axes_bias[0], axes_mse[0]
    else:
        return df2plot, axes_violin[0]

def createLogFileHandleAndResultsFileNames(path="../../RL-002-QueueBlocking", prefix="run", suffix="", use_dt_suffix=True):
    """
    Redirects the standard output to a file which is used to log messages.
    Creates output filenames for raw results and aggregated results.

    Ref: https://www.stackabuse.com/writing-to-a-file-with-pythons-print-function/

    Arguments:
    use_dt_suffix: bool
        Whether to add a suffix showing the execution datetime in the filenames.
    """
    if suffix is not None and suffix != "":
        suffix = "_" + suffix
    dt_start = get_current_datetime_as_string()
    if use_dt_suffix:
        dt_suffix = "_" + get_current_datetime_as_string(format="filename")
    else:
        dt_suffix = ""
    logfile = "{}/logs/{}{}{}.log".format(path, prefix, dt_suffix, suffix)
    resultsfile = "{}/results/{}{}{}_results.csv".format(path, prefix, dt_suffix, suffix)
    resultsfile_agg = "{}/results/{}{}{}_results_agg.csv".format(path, prefix, dt_suffix, suffix)
    proba_functions_file = "{}/results/{}{}{}_proba_functions.csv".format(path, prefix, dt_suffix, suffix)
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

    # Reset the standard output and show the execution time
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

def update_values_to_analyze(values, value_required, value_min, value_max, discard_smaller_than_minimum=True, factor_uplift=np.nan):
    "Updates the list of parameter values to analyze (e.g. the list of N or of T values to analyze for convergence)"
    if value_required <= 0:
        raise ValueError("Parameter `value_required` must be positive: {}".format(value_required))
    if value_required < value_min and discard_smaller_than_minimum:
        return values, np.nan

    # We add the required value to the list first, because we want to uplift ALL subsequent values,
    # regardless of whether they are smaller than the minimum or not
    # We do this for ALL parameter values, even those whose value is larger than the minimum,
    # because if we uplift the smaller values, the larger value will most likely be smaller than the uplifted values
    # leaving us with a parameter sequence that is not increasing, e.g.: N=[50, 75, 125, 275, 87]

    # NOTE: This assumes that the values to add to the list are in increasing order
    if len(values) == 0 and value_required < value_min:
        factor_uplift = value_min / value_required
    if not np.isnan(factor_uplift):
        value_required = int(value_required * factor_uplift)
    if value_required <= value_max:
        values += [value_required]

    return values, factor_uplift

def show_execution_parameters():
    print(get_current_datetime_as_string())
    print("Execution parameters:")
    print("nservers={}".format(nservers))
    print("Type of analysis: '{}'".format(analysis_type))
    print("Capacity K={}".format(K))
    print("Activation size J={}".format(J))
    if analysis_type == "N":
        print("# particles N={} (rel. errors = {}%)".format(N_values, errors_rel*100))
        print("# events T={} (actually required for error in E(T_A) ~ {:.1f}%: T={})".format(T_values, error_rel*100, T_required))
    elif analysis_type == "T":
        print("# arrival events T={} (rel. errors = {}%)".format(T_values, errors_rel*100))
        print("# particles N={} (actually required for error in phi(t,K) ~ {:.1f}%: N={})".format(N_values, error_rel*100, N_required))
    elif analysis_type == "J":
        print("# particles N={} (actually required for error in phi(t,K) ~ {:.1f}%: N={})".format(N, error_rel*100, N_required))
        print("# events T={} (actually required for error in E(T_A) ~ {:.1f}%: T={})".format(T, error_rel * 100, T_required))
    print("Burn-in time steps BITS={}".format(burnin_time_steps))
    print("Min #cycles to estimate expectations MINCE={}".format(min_num_cycles_for_expectations))
    print("Replications={}".format(replications))
    print("tests2run={}".format(tests2run))
    print("run_mc={}".format(run_mc))
    print("save_results={}".format(save_results))
    print("seed={}".format(seed))
#------------------- Functions --------------------


if __name__ == "__main__":
    #------------------- Execution starts -------------
    # Default execution arguments when no arguments are given
    # Example of execution from the command line:
    # python run_FV.py 1 N 5 10 0.5 8 1
    print("User arguments: {}".format(sys.argv))
    nargs_required = 7
    counter_opt_args = 0
    if len(sys.argv) == 1:    # Only the execution file name is contained in sys.argv
        sys.argv += [1]       # Number of servers in the system to simulate
        sys.argv += ["N"]     # Type of analysis: either "N" for the impact of number of particles, "T" for the impact of the number of events, or "J" for the impact of buffer size
        sys.argv += [10]      # K: capacity of the system
        sys.argv += [0.4]     # J factor: factor such that J = round(factor*K)
        sys.argv += [0.2]     # Either the value of the parameter that is NOT analyzed (according to parameter "type of analysis", e.g. "N" if "type of analysis" = "T") (if value >= 10), or the relative expected error from which such value is computed (if value < 10).
        sys.argv += [3]       # Number of replications
        sys.argv += [1]       # Test number to run: only one is accepted
    if len(sys.argv) == nargs_required + counter_opt_args + 1:
        sys.argv += [False]   # Whether to discard the values to analyze that are less than the minimum specified (e.g. N < N_min) or uplift the values to be larger so that we have enough samples to compute estimates
    counter_opt_args += 1
    if len(sys.argv) == nargs_required + counter_opt_args + 1:
        sys.argv += [10]      # BITS: Burn-in time steps to consider until stationarity can be assumed for the estimation of expectations (e.g. E(T) (return cycle time) in Monte-Carlo, E(T_A) (reabsorption cycle time) in Fleming-Viot)
    counter_opt_args += 1
    if len(sys.argv) == nargs_required + counter_opt_args + 1:
        sys.argv += [5]       # MINCE: Minimum number of cycles to be used for the estimation of expectations (e.g. E(T) in Monte-Carlo and E(T_A) in Fleming-Viot)
    counter_opt_args += 1
    if len(sys.argv) == nargs_required + counter_opt_args + 1:
        sys.argv += [1]       # Number of methods to run: 1 (only FV), 2 (FV & MC)
    counter_opt_args += 1
    if len(sys.argv) == nargs_required + counter_opt_args + 1:
        sys.argv += ["nosave"]  # Either "nosave" or anything else for saving the results and log
    counter_opt_args += 1
    if len(sys.argv) == nargs_required + counter_opt_args + 1:
        sys.argv += [True]   # Whether to use the execution date and time in the output file name containing the results
    counter_opt_args += 1
    if len(sys.argv) == nargs_required + counter_opt_args + 1:
        sys.argv += ["ID1"]  # Identifier to use for this run (e.g. "ID1"), in addition to the execution datetime (if requested)
    counter_opt_args += 1
    if len(sys.argv) == nargs_required + counter_opt_args + 1:
        sys.argv += [True]   # Whether to generate plots with the results
    counter_opt_args += 1
    print("Parsed user arguments: {}".format(sys.argv))
    print("")

    #-- Parse user arguments
    # This function parses a boolean input parameter which, depending on where this script is called from, may be a boolean
    # value already (if run from e.g. PyCharm) or may be a string value (if run from the command line).
    parse_boolean_parameter = lambda x: isinstance(x, bool) and x or isinstance(x, str) and x == "True"

    nservers = int(sys.argv[1]); lmbda = 0.7; mu = 1.0  # Queue system characteristics
    analysis_type = sys.argv[2]
    K = int(sys.argv[3])
    J = max(1, int(np.round(float(sys.argv[4]) * K)))    # We need the float() because the input arguments are read as strings when running the program from the command line

    #---------------------- Definition of parameters N and T ----------------------
    # Min and Max values are defined for N and T in order to:
    # - (min) not have a too small values making estimates not work at all (they would most likely be set to NaN!)
    # - (max) not incur into a simulation that takes forever.
    # When the analysis is done by one of these parameters, any value that is above its max value is NOT considered for simulation.
    # Otherwise, the parameter is clipped to the [min, max] interval.
    N_min = 10; N_max = 6000
    T_min = 50; T_max = 10000
    # We define the relative errors to consider which define the values of the parameter to analyze.
    # Errors are defined in *decreasing* order so that the parameter values are in increasing order.
    errors_rel = np.r_[np.arange(1.0, 0.0, -0.2), 0.1, 0.05]

    other_parameter = float(sys.argv[5])
    discard_smaller_than_minimum = parse_boolean_parameter(sys.argv[8])
    if other_parameter >= 10:
        # The parameter defines "the number of..." particles or arrivals (instead of an expected relative error value)
        # => Just assign the given value to the respective parameter
        if analysis_type == "N":
            T = int(other_parameter)
            error_rel = compute_rel_errors_for_fv_process([lmbda/mu], K, J/K, N=None, T=T, constant_proportionality=1)
        elif analysis_type == "T":
            N = int(other_parameter)
            error_rel = compute_rel_errors_for_fv_process([lmbda/mu], K, J/K, N=N, T=None)
        elif analysis_type == "J":
            N = int(other_parameter)
            T = int(other_parameter)
    else:
        # The parameter defines the expected relative error for the estimation of Phi(t,K) or E(T_A)
        # => Compute N or T based on this expected relative error
        error_rel = other_parameter
        # This expected relative error is used to define the value of the other parameter, but if this parameter value
        # falls out of the [min, max] value defined for the parameter, then no experiments are run.
        if analysis_type == "N":
            T_required = compute_nparticles_and_narrivals_for_fv_process([lmbda / mu], K, J / K, error_rel_phi=None, error_rel_et=error_rel, constant_proportionality=1)
            if T_required == 0:
                T_required = 1
            factor_uplift = np.nan
            T_values, factor_uplift = update_values_to_analyze( [], T_required, T_min, T_max,
                                                                discard_smaller_than_minimum=discard_smaller_than_minimum,
                                                                factor_uplift=factor_uplift)
            ## Note: T_values is used when calling the analyze_convergence() function
        elif analysis_type == "T":
            N_required = compute_nparticles_and_narrivals_for_fv_process([lmbda / mu], K, J / K, error_rel_phi=error_rel, error_rel_et=None)
            if N_required == 0:
                # This happens when J = K, in which case we don't need any particle because all the particles of the Fleming-Viot process will be always blocked! (since the only active state is x=K)                N_required = 1
                N_required = 1
            factor_uplift = np.nan
            N_values, factor_uplift = update_values_to_analyze( [], N_required, N_min, N_max,
                                                                discard_smaller_than_minimum=discard_smaller_than_minimum,
                                                                factor_uplift=factor_uplift)
            ## Note: N_values is used when calling the analyze_convergence() function
        elif analysis_type == "J":
            N_required, T_required = compute_nparticles_and_narrivals_for_fv_process([lmbda / mu], K, J / K, error_rel_phi=error_rel, error_rel_et=error_rel, constant_proportionality=1)
            N = int(min(max(N_min, N_required), N_max))
            T = int(min(max(T_min, T_required), T_max))

    # Define the values to use on the parameter to analyze based on the expected relative error for the estimation
    # affected by the parameter (Phi(t,K) for parameter N and E(T_A) for parameter T)
    if analysis_type == "N":
        N_values = []
        factor_uplift = np.nan
        for err_rel in errors_rel:
            N_required = compute_nparticles_and_narrivals_for_fv_process([lmbda / mu], K, J / K, error_rel_phi=err_rel, error_rel_et=None)
            N_values, factor_uplift = update_values_to_analyze(N_values, N_required, N_min, N_max,
                                                               discard_smaller_than_minimum=discard_smaller_than_minimum,
                                                               factor_uplift=factor_uplift)
    elif analysis_type == "T":
        T_values = []
        factor_uplift = np.nan
        for err_rel in errors_rel:
            T_required = compute_nparticles_and_narrivals_for_fv_process([lmbda / mu], K, J / K, error_rel_phi=None, error_rel_et=err_rel, constant_proportionality=1)
            T_values, factor_uplift = update_values_to_analyze(T_values, T_required, T_min, T_max,
                                                               discard_smaller_than_minimum=discard_smaller_than_minimum,
                                                               factor_uplift=factor_uplift)
    # Use the following if we want to analyze a fixed value of N and T (e.g. when debugging)
    #N_values = [50]
    #T_values = [50]
    #---------------------- Definition of parameters N and T ----------------------

    # Computing the minimum required N and T values for different expected relative errors for different K and J/K values
    df_NT_required = compute_nparticles_and_nsteps_for_fv_process_many_settings(   rhos=[lmbda/mu],
                                                                                   K_values=[5, 10, 20, 40],
                                                                                   JF_values=np.arange(0.1, 0.9, 0.1),
                                                                                   error_rel=np.arange(0.1, 1.1, 0.1))
    #df_NT_required.to_csv("../../RL-002-QueueBlocking/NT_required.csv")

    replications = int(sys.argv[6])
    tests2run = [int(v) for v in [sys.argv[7]]]  # NOTE: It's important to enclose sys.argv[7] in brackets because o.w.,
                                                 # from the command line, a number with more than one digit is interpreted
                                                 # as a multi-element list!! (e.g. 10 is interpreted as a list with elements [1, 0])
    burnin_time_steps = int(sys.argv[9])
    min_num_cycles_for_expectations = int(sys.argv[10])
    run_mc = int(sys.argv[11]) == 2
    save_results = sys.argv[12] != "nosave"
    save_with_dt = parse_boolean_parameter(sys.argv[13])
    id_run = sys.argv[14]
    plot = parse_boolean_parameter(sys.argv[15])

    if len(tests2run) == 0:
        print("No tests have been specified to run. Please specify the test number as one of the arguments.")
        sys.exit()

    seed = 1313

    show_execution_parameters()

    if len(N_values) == 0 or len(T_values) == 0:
        print("The parameter settings (K, J, error_rel) = {} are such that no N values or no T values are eligible",
              " for running the simulation, because they would be too large making the simulation prohibitive in terms of execution time.",
              "\nTry other set of parameters. The program stops.".format([K, J, error_rel]))
        sys.exit()

    if analysis_type not in ["N", "T", "J"]:
        raise ValueError("Valid values for the second parameter are 'N', 'T' or 'J'. Given: {}".format(analysis_type))
    elif analysis_type == "N":
        resultsfile_prefix = "estimates_vs_N"
        resultsfile_suffix = "{}K={},J={},T={}".format((id_run == "" and "") or (id_run + "_"), K, J, T_values)
    elif analysis_type == "T":
        resultsfile_prefix = "estimates_vs_T"
        resultsfile_suffix = "{}K={},J={},N={}".format((id_run == "" and "") or (id_run + "_"), K, J, N_values)
    else:
        resultsfile_prefix = "estimates_vs_J"
        resultsfile_suffix = "{}K={},N={},T={}".format((id_run == "" and "") or (id_run + "_"), K, N_values, T_values)
    #-- Parse user arguments

    if save_results:
        dt_start, stdout_sys, fh_log, logfile, resultsfile, resultsfile_agg, proba_functions_file, figfile = \
            createLogFileHandleAndResultsFileNames(prefix=resultsfile_prefix, suffix=resultsfile_suffix, use_dt_suffix=save_with_dt)
        # Show the execution parameters again in the log file
        show_execution_parameters()
    else:
        fh_log = None; resultsfile = None; resultsfile_agg = None; proba_functions_file = None; figfile = None

    if analysis_type in ["N", "T"]:
        # -- Single-server
        results, results_agg, est_fv, est_mc = analyze_convergence(
                                                    estimation_process=Process.Simulators,
                                                    nservers=nservers, job_class_rates=[lmbda], service_rates=[mu], K=K, buffer_size_activation=J,
                                                    burnin_time_steps=burnin_time_steps, min_num_cycles_for_expectations=min_num_cycles_for_expectations,
                                                    method_proba_surv=SurvivalProbabilityEstimation.FROM_N_PARTICLES,
                                                    nparticles=N_values, #[N, 2*N, 4*N, 8*N, 16*N],  # [800, 1600, 3200], #[10, 20, 40], #[24, 66, 179],
                                                    nmeantimes=T_values, #50, #[170, 463, 1259],
                                                    replications=replications, run_mc=run_mc,
                                                    seed=seed)

        # -- Multi-server
        # results, results_agg, est_fv, est_mc = \
        #    analyze_convergence(nservers=3, K=20, buffer_size_activation=0.5, burnin_cycles_absorption=4,
        #                                             nparticles_min=800, nparticles_max=1600, nparticles_step_prop=1,
        #                                             nmeantimes=1400, replications=5,
        #                                             run_mc=run_mc, plotFlag=True)

        # Save results
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg}])

        # Plot results
        if plot:
            if analysis_type == "N":
                for T in T_values:
                    # Note: the columns defined in parameters `x` and `x2` are grouping variables that define each violin plot
                    axes = plot_results_fv_mc(results, x="N", x2="#Cycles(MC)_mean",
                                              xlabel="# particles", xlabel2="# Return Cycles to {}".format(J-1),
                                              ymin=0.0, plot_mc=run_mc, splines=False,
                                              title="nservers={}, K={}, J={}, T={}, BITS={}, MINCE={}" \
                                                    .format(nservers, K, J, T, burnin_time_steps, min_num_cycles_for_expectations))
            elif analysis_type == "T":
                for N in N_values:
                    # Note: the columns defined in parameters `x` and `x2` are grouping variables that define each violin plot
                    axes = plot_results_fv_mc(results, x="T", x2="#Cycles(MC)_mean",
                                              xlabel="# arrival events", xlabel2="# Return Cycles to {}".format(J-1),
                                              ymin=0.0, plot_mc=run_mc, splines=False,
                                              title="nservers={}, K={}, J={}, N={}, BITS={}, MINCE={}" \
                                                    .format(nservers, K, J, N, burnin_time_steps, min_num_cycles_for_expectations))
    elif analysis_type == "J":
        # Info for plotting...
        x = "buffer_size_activation"
        xlabel = "J as fraction of K"
        J_values = []
        multiplier = 1          # Multiplier (float >= 1) that defines the J values to consider
                                # (e.g. if multiplier = 2 we consider the values J, 2*J, 4*J, as long as k*J does not go over K (where k = 1, 2, 4, ... in the example given))
                                # The number of J values considered is more than one only when multiplier > 1
        J_value = J             # This is an integer value
        J_values = [J_value]
        while multiplier > 1 and np.round(multiplier * J_values[-1]) < K:
            J_values += [np.round(multiplier * J_values[-1])]
        results, results_agg, proba_functions, est_fv, est_mc = analyze_absorption_size(
                                                                    nservers=nservers,
                                                                    replications=replications,
                                                                    K_values=[K],
                                                                    nparticles_values=[N],
                                                                    nmeantimes_values=[T],
                                                                    buffer_size_activation_values=J_values,
                                                                    burnin_cycles_absorption_values=0,
                                                                    seed=1313,
                                                                    run_mc=run_mc,
                                                                    dict_params_info={'plot': False, 'log': False})
        save_dataframes([{'df': results, 'file': resultsfile},
                         {'df': results_agg, 'file': resultsfile_agg},
                         {'df': proba_functions, 'file': proba_functions_file}])

        axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K'] == K, plot_mc=run_mc)

    # Close log file, if any was opened
    if fh_log is not None:
        closeLogFile(fh_log, stdout_sys, dt_start)


    def deprecated_analysis_by_J_several_tests():
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

