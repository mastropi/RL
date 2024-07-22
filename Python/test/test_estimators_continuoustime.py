# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:18:27 2022

@author: Daniel Mastropietro
@description: Unit tests for estimators (V, Q) on continuous-time MDPs.
@details: Naming conventions follow the instructions given in test_conventions.txt.
"""

import runpy
runpy.run_path('../../setup.py')

import unittest
from unittest_data_provider import data_provider

import numpy as np

from Python.lib.environments import queues as env_queues
from Python.lib.environments.queues import rewardOnJobRejection_Constant
import Python.lib.queues as queues
from Python.lib.estimators.fv import SurvivalProbabilityEstimation

from Python.lib.utils.basic import get_current_datetime_as_string
from Python.lib.utils.computing import stationary_distribution_product_form, func_prod_birthdeath, func_prod_knapsack

from Python.lib.run_FV import run_mc_estimation_single_server, run_fv_estimation_single_server
from Python.lib.run_FV_LossNetwork import run_mc_estimation_loss_network, run_fv_estimation_loss_network


class Test_EstAverageValueV_EnvQueueSingleServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Environment 1: single server queue with Poisson arrivals and departures
        nservers = 1
        rates_birth = [0.7]
        rates_death = [1.0]
        capacities = [5, 20, 40]
        dict_queue_mm_single_server = dict()
        cls.dict_env_queue_mm_single_server = dict()
        for idx_K, K in enumerate(capacities):
            dict_queue_mm_single_server[K] = queues.QueueMM(rates_birth, rates_death, nservers, capacities[idx_K],
                                                            origin=0.0)
            # We consider there is only one job type arriving at the single server queue
            job_class_rates = dict_queue_mm_single_server[K].getBirthRates()
            reward_func = None
            rewards_accept_by_job_class = [1.0]
            dict_params_reward_func = None
            cls.dict_env_queue_mm_single_server[K] = env_queues.EnvQueueSingleBufferWithJobClasses(
                dict_queue_mm_single_server[K],
                job_class_rates,
                reward_func,
                rewards_accept_by_job_class,
                dict_params_reward_func)

    # -------- DATA -------
    # Case number, description, expected value, parameters
    data_test_EnvQueueSingleServer_MetMCFV_SeveralCapacities = lambda DEFAULT_EXECUTION = True: (
        # Note: True blocking probability for different K values when rho = 0.7
        # Pr(K) = rho^K * (1 - rho) / (1 - rho^(K+1))
        # K = 5: Pr(K) = 5.71%
        # K = 20: Pr(K) = 0.0239% = 2.39E-4
        # K = 40: Pr(K) = 0.000019% = 1.9E-7

        # First set of results
        # Not so good estimation because:
        # (a) P(T>t) is estimated from M reabsorption cycles of the single particle simulation (instead of from the N particles, which gives a good estimate for any size of the absorption set, J)
        # (b) No burn-in period is used.
        # (c) No minimum number of cycles are required for the estimation of expectations.

        # Currently (30-Aug-2023), on 9 tests for the FV approach in 3 tests of 3 tests with SMALL, MODERATE and LARGE K values,
        # the execution takes about 5 minutes.
        (1, DEFAULT_EXECUTION, 'Small K',
         {'K': 5, 'J': 2, 'N': 5, 'T': 20,
          'burnin_time_steps': 0, 'min_num_cycles_for_expectations': 0, 'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(MC)': 0.0, 'E(T) (MC)': 1.98926, '#E(T) (MC)': 44, '#events_MC': 100,
          'Pr(FV)': 0.0103065, 'E(T)': 7.06, '#E(T)': 4, 'Tmax': 28.2, 'Tend': 41.2, 'Smax': 4.6,
          '#events_ET': 39, '#events_FV': 44}),
        (2, DEFAULT_EXECUTION, 'Moderate K',
         {'K': 20, 'J': 10, 'N': 100, 'T': 2000,
          'burnin_time_steps': 0, 'min_num_cycles_for_expectations': 0, 'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(MC)': 0.00031823, 'E(T) (MC)': 50.02, '#E(T) (MC)': 2866, '#events_MC': 200E3,
          'Pr(FV)': 0.00047179, 'E(T)': 243.54, '#E(T)': 11, 'Tmax': 2678.9, 'Tend': 2900.6, 'Smax': 47.3,
          '#events_ET': 4004, '#events_FV': 8164}),
        (3, False, 'Large K',   # (2023/11/12) We do NOT execute the simulation for large K because it takes too long and it is essentially the same as for moderate K... The only difference would be to consider cases where there are not enough samples to estimate quantities (but this already pass as of today)
         {'K': 40, 'J': 35, 'N': 400, 'T': 2000,
          'burnin_time_steps': 0, 'min_num_cycles_for_expectations': 0, 'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(MC)': 0.0, 'E(T) (MC)': 3.9979, '#E(T) (MC)': 1, '#events_MC': 800E3,
          'Pr(FV)': 9.619e-05, 'E(T)': 3.99, '#E(T)': 1, 'Tmax': 4.0, 'Tend': 2809.3, 'Smax': 1.1,
          '#events_ET': 4029, '#events_FV': 792}),

        # Second set of results
        # Better estimation than above because items (b) and (c) are no longer true
        (4, DEFAULT_EXECUTION, 'Small K',
         {'K': 5, 'J': 2, 'N': 5, 'T': 20,
          'burnin_time_steps': 10, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(MC)': 0.0, 'E(T) (MC)': 1.99389, '#E(T) (MC)': 40, '#events_MC': 100,
          'Pr(FV)': np.nan, 'E(T)': np.nan, '#E(T)': 2, 'Tmax': 28.2, 'Tend': 41.2, 'Smax': 4.6,
          '#events_ET': 39, '#events_FV': 0}),
        (5, DEFAULT_EXECUTION, 'Moderate K',
         {'K': 20, 'J': 10, 'N': 100, 'T': 2000,
          'burnin_time_steps': 10, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(MC)': 0.00031819, 'E(T) (MC)': 49.99, '#E(T) (MC)': 2864, '#events_MC': 200E3,
          'Pr(FV)': 0.000418716, 'E(T)': 274.41, '#E(T)': 9, 'Tmax': 2678.9, 'Tend': 2900.6, 'Smax': 47.3,
          '#events_ET': 4004, '#events_FV': 8164}),
        (6, False, 'Large K',   # (2023/11/12) We do NOT execute the simulation for large K because it takes too long and it is essentially the same as for moderate K... The only difference would be to consider cases where there are not enough samples to estimate quantities (but this already pass as of today)
         {'K': 40, 'J': 12, 'N': 400, 'T': 2000,
          'burnin_time_steps': 10, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(MC)': 0.0, 'E(T) (MC)': 100.71, '#E(T) (MC)': 5681, '#events_MC': 800E3,
          'Pr(FV)': 0.0, 'E(T)': 188.35, '#E(T)': 13, 'Tmax': 2676.9, 'Tend': 2899.0, 'Smax': 21.9,
          '#events_ET': 4005, '#events_FV': 14758}),

        # Third set of results
        # Better estimation than above because in addition item (a) is no longer true, i.e. now P(T>t) is estimated from the *N particles* used in the FV simulation
        # (and in some cases because the parameter values are larger, e.g. larger number of particles)
        (7, DEFAULT_EXECUTION, 'Small K',
         {'K': 5, 'J': 3, 'N': 80, 'T': 200,
          'burnin_time_steps': 20, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_N_PARTICLES},
            # NOTE: (2022/11/01) The estimated probability is too small compared to the true value...
            # And this is most likely due to a large estimated E(T) value... which should have been ~ 5 (see above tests),
            # but it turned out to be 8.65... why??
         {'Pr(MC)': 0.0617052, 'E(T) (MC)': 3.49, '#E(T) (MC)': 3330, '#events_MC': 16000,
          'Pr(FV)': 0.0408603, 'E(T)': 6.98, '#E(T)': 32, 'Tmax': 287.8, 'Tend': 288.9, 'Smax': 12.7,
          '#events_ET': 382, '#events_FV': 1723}),
        (8, DEFAULT_EXECUTION, 'Moderate K',
         {'K': 20, 'J': 10, 'N': 200, 'T': 2000,
          'burnin_time_steps': 10, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_N_PARTICLES},
         {'Pr(MC)': 0.000242275, 'E(T) (MC)': 48.66, '#E(T) (MC)': 5865, '#events_MC': 400E3,
          'Pr(FV)': 0.000163620, 'E(T)': 268.36, '#E(T)': 9, 'Tmax': 2678.9, 'Tend': 2900.6, 'Smax': 74.5,
          '#events_ET': 4004, '#events_FV': 25386}),
        (9, False, 'Large K',   # (2023/11/12) We do NOT execute the simulation for large K because it takes too long and it is essentially the same as for moderate K... The only difference would be to consider cases where there are not enough samples to estimate quantities (but this already pass as of today)
         {'K': 40, 'J': 12, 'N': 400, 'T': 2000,    # The value of J was tuned here in order to obtain sufficient samples for the estimation of E(T) so that the estimated probability is NOT NaN
          'burnin_time_steps': 10, 'min_num_cycles_for_expectations': 4,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_N_PARTICLES},
         {'Pr(MC)': 0.0, 'E(T) (MC)': 100.71, '#E(T) (MC)': 5681, '#events_MC': 800E3,
          'Pr(FV)': 9.83671e-08, 'E(T)': 188.10, '#E(T)': 13, 'Tmax': 2676.9, 'Tend': 2899.0, 'Smax': 65.0,
          '#events_ET': 4005, '#events_FV': 44191}),
    )
    # -------- DATA -------

    @data_provider(data_test_EnvQueueSingleServer_MetMCFV_SeveralCapacities)
    def test_EnvQueueSingleServer_MetMC_SeveralCapacities(self, casenum, run, desc, dict_params, dict_expected):
        "Test the Monte-Carlo implementation of the blocking probability of a single-server queue system"
        if run:
            print("\n*** Testing {}, case number {}: '{}' ***".format(self.id(), casenum, desc))
            K = dict_params['K']
            nservers = self.dict_env_queue_mm_single_server[K].getNumServers()
            job_rates = self.dict_env_queue_mm_single_server[K].getJobClassRates()
            service_rates = self.dict_env_queue_mm_single_server[K].getServiceRates()
            proba_blocking_mc, expected_reward, probas_stationary, n_cycles_used_for_probas_estimation, \
                expected_return_time, n_return_cycles, time_mc, n_events_mc = run_mc_estimation_single_server(   self.dict_env_queue_mm_single_server[K],
                                                                                                       dict_params['K'],
                                                                                                       dict_params['J'],
                                                                                                       dict_params['N'] * dict_params['T'],
                                                                                                       dict_params['burnin_time_steps'],
                                                                                                       dict_params['min_num_cycles_for_expectations'],
                                                                                                       seed=1313)

            rhos = list(np.array(job_rates) / np.array(service_rates))
            print(get_current_datetime_as_string())
            print("EXECUTION PARAMETERS:")
            print("- # servers: {}".format(nservers))
            print("- job arrival rates at servers: {}".format(job_rates))
            print("- service rates: {}".format(service_rates))
            print("- capacity: {}".format(K))
            print("- loads: {}".format(rhos))
            print("- absorption set size: {}".format(dict_params['J']))
            print("- # arrival time steps: {}".format(dict_params['T']))
            print("")

            print("ESTIMATION RESULTS:")
            print("- Blocking probability Pr(K={}) = {}".format(K, proba_blocking_mc))
            print("- Stationary probability Pr(K={}) = {}".format(K, probas_stationary[K]))
            assert np.isnan(proba_blocking_mc) and np.isnan(probas_stationary[K]) or \
                   np.isclose(proba_blocking_mc, probas_stationary[K])
            print("- Expected return time to the start state = {}".format(expected_return_time))
            print("- Number of observed cycles to the start state = {}".format(n_return_cycles))
            print("- Number of cycles used in the estimation of the stationary probability (using renewal theory) = {}".format(n_cycles_used_for_probas_estimation))
            print("- Number of observed events = {}".format(n_events_mc))

            # True stationary probability of blocking
            x, dist = stationary_distribution_product_form(K, rhos, func_prod_birthdeath)
            proba_blocking_true = np.sum([v for xx, v in zip(x, dist) if xx[0] in probas_stationary.keys()])
            print("")
            print("- Stationary blocking probability = {}".format(proba_blocking_mc))
            print("- TRUE stationary blocking probability = {}".format(proba_blocking_true))

            # Check system setup is the one required to obtain the expected results
            assert nservers == 1, "Number of servers is 1"
            assert list(job_rates) == [0.7], "Arrival rate is 0.7"
            assert list(service_rates) == [1.0], "Service rate is 1.0"

            # Assertions
            assert np.isnan(proba_blocking_mc) and np.isnan(dict_expected['Pr(MC)']) or \
                   np.isclose(proba_blocking_mc, dict_expected['Pr(MC)'])
            assert np.isnan(expected_return_time) and np.isnan(dict_expected['E(T) (MC)']) or \
                   np.isclose(expected_return_time, dict_expected['E(T) (MC)'], atol=0.01)
            self.assertEqual(n_return_cycles, dict_expected['#E(T) (MC)'])
            self.assertEqual(n_events_mc, dict_expected['#events_MC'])
            # Consistency assertions
            assert np.isnan(proba_blocking_mc) and np.isnan(probas_stationary[K]) or \
                   np.isclose(proba_blocking_mc, probas_stationary[K])

    @data_provider(data_test_EnvQueueSingleServer_MetMCFV_SeveralCapacities)
    def test_EnvQueueSingleServer_MetFV_SeveralCapacities(self, casenum, run, desc, dict_params, dict_expected):
        "Test the Fleming-Viot implementation of the blocking probability of a single-server queue system"
        if run:
            print("\n*** Testing {}, case number {}: '{}' ***".format(self.id(), casenum, desc))
            K = dict_params['K']
            nservers = self.dict_env_queue_mm_single_server[K].getNumServers()
            job_rates = self.dict_env_queue_mm_single_server[K].getJobClassRates()
            service_rates = self.dict_env_queue_mm_single_server[K].getServiceRates()
            proba_blocking_fv, expected_reward, probas_stationary, \
                expected_absorption_time, n_absorption_time_observations, \
                    time_last_absorption, time_end_simulation_et, max_survival_time, time_end_simulation_fv, \
                        n_events_et, n_events_fv_only = run_fv_estimation_single_server(   self.dict_env_queue_mm_single_server[K],
                                                                                           dict_params['K'],
                                                                                           dict_params['J'],
                                                                                           dict_params['N'],
                                                                                           dict_params['T'],
                                                                                           dict_params['burnin_time_steps'],
                                                                                           dict_params['min_num_cycles_for_expectations'],
                                                                                           dict_params['method_survival_probability_estimation'],
                                                                                           seed=1313)

            rhos = list(np.array(job_rates) / np.array(service_rates))
            print(get_current_datetime_as_string())
            print("EXECUTION PARAMETERS:")
            print("- # servers: {}".format(nservers))
            print("- job arrival rates at servers: {}".format(job_rates))
            print("- service rates: {}".format(service_rates))
            print("- capacity: {}".format(K))
            print("- loads: {}".format(rhos))
            print("- absorption set size: {}".format(dict_params['J']))
            print("- # particles: {}".format(dict_params['N']))
            print("- # arrival time steps: {}".format(dict_params['T']))
            print("")

            print("ESTIMATION RESULTS:")
            print("- Blocking probability Pr(K={}) = {}".format(K, proba_blocking_fv))
            print("- Stationary probability Pr(K={}) = {}".format(K, probas_stationary[K]))
            assert np.isnan(proba_blocking_fv) and np.isnan(probas_stationary[K]) or \
                   np.isclose(proba_blocking_fv, probas_stationary[K])
            print("- Expected reabsorption time E(T) = {}".format(expected_absorption_time))
            print("- Number of reabsorption cycles = {}".format(n_absorption_time_observations))
            print("- Time of last absorption = {:.1f}".format(time_last_absorption))
            print("- Last observed time of the MC simulation for E(T) = {:.1f}".format(time_end_simulation_et))
            print("- Max observed survival time = {:.1f}".format(max_survival_time))
            print("- Time FV simulation ends = {:.1f}".format(time_end_simulation_fv))
            print("- Number of events in MC simulation for E(T) = {}".format(n_events_et))
            print("- Number of events in FV simulation for Phi(t) = {}".format(n_events_fv_only))

            # True stationary probability of blocking
            x, dist = stationary_distribution_product_form(K, rhos, func_prod_birthdeath)
            proba_blocking_true = np.sum([v for xx, v in zip(x, dist) if xx[0] in probas_stationary.keys()])
            print("")
            print("- Stationary blocking probability = {}".format(proba_blocking_fv))
            print("- TRUE stationary blocking probability = {}".format(proba_blocking_true))

            # Check system setup is the one required to obtain the expected results
            assert nservers == 1, "Number of servers is 1"
            assert list(job_rates) == [0.7], "Arrival rate is 0.7"
            assert list(service_rates) == [1.0], "Service rate is 1.0"

            # Assertions
            assert np.isnan(proba_blocking_fv) and np.isnan(dict_expected['Pr(FV)']) or \
                   np.isclose(proba_blocking_fv, dict_expected['Pr(FV)'])
            assert np.isnan(expected_absorption_time) and np.isnan(dict_expected['E(T)']) or \
                   np.isclose(expected_absorption_time, dict_expected['E(T)'], atol=0.01)  # places = decimal places
            self.assertEqual(n_absorption_time_observations, dict_expected['#E(T)'])
            self.assertAlmostEqual(time_last_absorption, dict_expected['Tmax'], places=1)
            self.assertAlmostEqual(time_end_simulation_et, dict_expected['Tend'], places=1)
            self.assertAlmostEqual(max_survival_time, dict_expected['Smax'], places=1)
            self.assertEqual(n_events_et, dict_expected['#events_ET'])
            self.assertEqual(n_events_fv_only, dict_expected['#events_FV'])
            # Consistency assertions
            assert np.isnan(proba_blocking_fv) and np.isnan(probas_stationary[K]) or \
                   np.isclose(proba_blocking_fv, probas_stationary[K])
            assert time_end_simulation_fv == 0.0 or \
                   time_end_simulation_fv >= max_survival_time


class Test_EstAverageValueV_EnvQueueLossNetworkWithJobClasses(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rates_birth = [0.1, 0.5, 0.8]
        rates_death = [1.0, 1.0, 1.0]
        nservers = len(rates_death)
        queue_mm_loss = queues.QueueMM(rates_birth, rates_death, nservers, +np.Inf, origin=0.0)
        reward_func = rewardOnJobRejection_Constant
        rewards_accept_by_job_class = [0.0] * len(rates_birth)
        dict_params_reward_func = None
        cls.env_queue_mm_loss = env_queues.EnvQueueLossNetworkWithJobClasses(
            queue_mm_loss,
            reward_func,
            rewards_accept_by_job_class,
            dict_params_reward_func)

    def test_EnvQueueLossNetwork_MetMC_SingleCapacity(self):
        blocking_occupancies_by_jobclass = [4, 2, 5]
        activation_occupancies_by_jobclass = [2, 1, 3]
        #activation_occupancies_by_jobclass = [int(0.3*occup) for occup in blocking_occupancies_by_jobclass]
        nservers = 10
        T = 2000 #200000 # 2000
        J = activation_occupancies_by_jobclass
        K = blocking_occupancies_by_jobclass
        burnin_time_steps = 20
        min_num_cycles_for_expectations = 5

        proba_blocking_mc, expected_reward, probas_stationary, n_cycles_used_for_probas_estimation, \
            expected_return_time, n_return_cycles, time_mc, n_events_mc = run_mc_estimation_loss_network(self.env_queue_mm_loss,
                                                                                                blocking_occupancies_by_jobclass,
                                                                                                activation_occupancies_by_jobclass,
                                                                                                nservers,
                                                                                                T,
                                                                                                burnin_time_steps,
                                                                                                min_num_cycles_for_expectations,
                                                                                                seed=1313)

        print(get_current_datetime_as_string())
        job_class_rates = self.env_queue_mm_loss.getJobClassRates()
        service_rates = self.env_queue_mm_loss.getServiceRates()
        rhos = list(np.array(job_class_rates) / np.array(service_rates))
        print("EXECUTION PARAMETERS:")
        print("- # servers (system's capacity): {}".format(nservers))
        print("- job arrival rates: {}".format(job_class_rates))
        print("- service rates: {}".format(service_rates))
        print("- loads: {}".format(rhos))
        print("- blocking occupancies per job class: {}".format(blocking_occupancies_by_jobclass))
        print("- activation occupancies per job class: {}".format(activation_occupancies_by_jobclass))
        print("- # arrival time steps: {}".format(T))
        print("")

        print("ESTIMATION RESULTS:")
        print("- Blocking probability = {}".format(proba_blocking_mc))
        proba_stationary = np.sum([v for k, v in probas_stationary.items()])
        print("- Stationary probability of ALL blocking states = {}".format(proba_stationary))
        print("- Stationary probabilities:")
        n_states_nonzero_probability = 0
        for s in probas_stationary.keys():
            n_states_nonzero_probability += probas_stationary[s] > 0
            print("state s = {}: Pr(s) = {}".format(s, probas_stationary[s]))
        print("- Number of states with positive estimated probability: {} of {} ({:.1f}%)".format(n_states_nonzero_probability, len(probas_stationary), n_states_nonzero_probability / len(probas_stationary)*100))
        print("- Estimated blocking probability by MC = {:.6f}%".format(proba_blocking_mc*100))
        print("- Estimated expected reward by MC = {:.6f}".format(expected_reward))
        print("- Expected return time to the start state = {}".format(expected_return_time))
        print("- Number of observed cycles to the start state = {}".format(n_return_cycles))
        print("- Number of cycles used in the estimation of the stationary probabilities (using renewal theory) = {}".format(n_cycles_used_for_probas_estimation))
        print("- Number of observed events = {}".format(n_events_mc))

        # True stationary probability of blocking
        x, dist = stationary_distribution_product_form(nservers, rhos, func_prod_knapsack)
        proba_stationary_true = np.sum([v for xx, v in zip(x, dist) if tuple(xx) in probas_stationary.keys()])
        print("")
        print("- Stationary probability of ALL blocking states (does NOT depend on the job arrival rates) = {}".format(proba_stationary))
        print("- TRUE Stationary probability of ALL blocking states = {}".format(proba_stationary_true))
        print("- Blocking probability (it depends on the job arrival rates) = {}".format(proba_blocking_mc))

        assert np.isclose(proba_stationary, 0.069825, atol=1E-6)
        assert np.isclose(proba_blocking_mc, 0.025118, atol=1E-6)
        # Check system setup is the one required to obtain the expected results
        assert nservers == 10, "Number of servers is 10"
        assert list(self.env_queue_mm_loss.getJobClassRates()) == [0.1, 0.5, 0.8], "Job arrival rates are [0.1, 0.5, 0.8]"
        assert list(self.env_queue_mm_loss.getServiceRates()) == [1.0, 1.0, 1.0], "Service rates are [1.0, 1.0, 1.0]"
        assert T == 2000, "Number of arrival events is 2000"
        assert K == [4, 2, 5], "Blocking sizes are [4, 2, 5]"
        assert J == [2, 1, 3], "Activation sizes are [1, 1, 1]"
        assert burnin_time_steps == 20, "The burn-in time is 20 time steps (discrete value)"
        assert min_num_cycles_for_expectations == 5, "The min # cycles for expectation calculations is 5"

    def test_EnvQueueLossNetwork_MetFV_SingleCapacity(self):
        blocking_occupancies_by_jobclass = [4, 2, 5]
        activation_occupancies_by_jobclass = [2, 1, 3]
        #activation_occupancies_by_jobclass = [1, 1, 1]
        nservers = 10
        N = 100 # 1000
        T = 200 # 2000
        K = blocking_occupancies_by_jobclass
        J = activation_occupancies_by_jobclass
        # (2023/01/26) Note about burnin and min cycles: Using (5, 5) or (50, 10) as (burnin_time_steps, min_cycles) didn't change much the estimated probability (e.g. from 14.454% to 14.453%!)
        burnin_time_steps = 20 #10
        min_num_cycles_for_expectations = 5
        method_survival_probability_estimation = SurvivalProbabilityEstimation.FROM_N_PARTICLES

        proba_blocking_fv, expected_reward, probas_stationary, \
            expected_absorption_time, n_absorption_time_observations, \
                time_last_absorption, time_end_simulation_et, max_survival_time, time_end_simulation_fv, \
                    n_events_et, n_events_fv_only = run_fv_estimation_loss_network( self.env_queue_mm_loss,
                                                                                    blocking_occupancies_by_jobclass,
                                                                                    activation_occupancies_by_jobclass,
                                                                                    nservers,
                                                                                    N,
                                                                                    T,
                                                                                    burnin_time_steps,
                                                                                    min_num_cycles_for_expectations,
                                                                                    method_survival_probability_estimation,
                                                                                    seed=1713)

        print(get_current_datetime_as_string())
        job_class_rates = self.env_queue_mm_loss.getJobClassRates()
        service_rates = self.env_queue_mm_loss.getServiceRates()
        rhos = list(np.array(job_class_rates) / np.array(service_rates))
        print("EXECUTION PARAMETERS:")
        print("- # servers (system's capacity): {}".format(nservers))
        print("- job arrival rates: {}".format(job_class_rates))
        print("- service rates: {}".format(service_rates))
        print("- loads: {}".format(rhos))
        print("- blocking occupancies per job class: {}".format(blocking_occupancies_by_jobclass))
        print("- activation occupancies per job class: {}".format(activation_occupancies_by_jobclass))
        print("- # particles: {}".format(N))
        print("- # arrival time steps: {}".format(T))
        print("")

        print("ESTIMATION RESULTS:")
        print("- Blocking probability = {}".format(proba_blocking_fv))
        proba_stationary = np.sum([v for k, v in probas_stationary.items()])
        print("- Stationary probability of ALL blocking states = {}".format(proba_stationary))
        print("- Stationary probabilities:")
        n_states_nonzero_probability = 0
        for s in probas_stationary.keys():
            n_states_nonzero_probability += probas_stationary[s] > 0
            print("state s = {}: Pr(s) = {}".format(s, probas_stationary[s]))
        print("- Number of states with positive estimated probability: {} of {} ({:.1f}%)".format(n_states_nonzero_probability, len(probas_stationary), n_states_nonzero_probability / len(probas_stationary)*100))
        print("- Estimated blocking probability by FV = {:.6f}%".format(proba_blocking_fv*100))
        print("- Estimated expected reward by FV = {:.6f}".format(expected_reward))
        print("- Expected reabsorption time E(T) = {}".format(expected_absorption_time))
        print("- Number of reabsorption cycles = {}".format(n_absorption_time_observations))
        print("- Time of last absorption = {:.1f}".format(time_last_absorption))
        print("- Last observed time of the MC simulation for E(T) = {:.1f}".format(time_end_simulation_et))
        print("- Max observed survival time = {:.1f}".format(max_survival_time))
        print("- Time FV simulation ends = {:.1f}".format(time_end_simulation_fv))
        print("- Number of events in MC simulation for E(T) = {}".format(n_events_et))
        print("- Number of events in FV simulation for Phi(t) = {}".format(n_events_fv_only))

        # True stationary probability of blocking
        x, dist = stationary_distribution_product_form(nservers, rhos, func_prod_knapsack)
        proba_stationary_true = np.sum([v for xx, v in zip(x, dist) if tuple(xx) in probas_stationary.keys()])
        print("")
        print("- Stationary probability of ALL blocking states (does NOT depend on the job arrival rates) = {}".format(proba_stationary))
        print("- TRUE Stationary probability of ALL blocking states = {}".format(proba_stationary_true))
        print("- Blocking probability (it depends on the job arrival rates) = {}".format(proba_blocking_fv))

        assert n_states_nonzero_probability == 16
        assert len(probas_stationary) == 49
        assert n_absorption_time_observations == 57
        assert np.isclose(expected_absorption_time, 2.12, atol=1E-2)
        assert n_events_et == 397
        assert n_events_fv_only == 2308
        assert np.isclose(max_survival_time, 6.0, atol=1E-1)
        assert np.isclose(proba_stationary, 0.048536, atol=1E-6)
            ## The above result is a rather far from the true stationary probability = 0.076922.
            ## If we increase the number of particles from 100 to 500, the result gets closer,
            ## but the estimation process increases from 15 sec to ~ 7 min...
            ## Note also that if the activation sizes are decreased from [2, 1, 3] to [1, 1, 1],
            ## the stationary probability increases... the opposite from what I thought... although we are looking at just one sample here!
        assert np.isclose(proba_blocking_fv, 0.017514, atol=1E-6)
        # Check system setup is the one required to obtain the expected results
        assert nservers == 10, "Number of servers is 10"
        assert list(self.env_queue_mm_loss.getJobClassRates()) == [0.1, 0.5, 0.8], "Job arrival rates are [0.1, 0.5, 0.8]"
        assert list(self.env_queue_mm_loss.getServiceRates()) == [1.0, 1.0, 1.0], "Service rates are [1.0, 1.0, 1.0]"
        assert N == 100, "Number of particles is 100"
        assert T == 200, "Number of arrival events for E(T_A) estimation is 200"
        assert K == [4, 2, 5], "Blocking sizes are [4, 2, 5]"
        assert J == [2, 1, 3], "Activation sizes are [2, 1, 3]"
        assert burnin_time_steps == 20, "The burn-in time is 20 time steps (discrete value)"
        assert min_num_cycles_for_expectations == 5, "The min # cycles for expectation calculations is 5"


if __name__ == '__main__':
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line

    runner = unittest.TextTestRunner()

    # Run all tests
    # unittest.main()

    # Create the test suites
    test_suite = unittest.TestSuite()
    test_suite.addTest(Test_EstAverageValueV_EnvQueueSingleServer("test_EnvQueueSingleServer_MetMC_SeveralCapacities"))
    test_suite.addTest(Test_EstAverageValueV_EnvQueueSingleServer("test_EnvQueueSingleServer_MetFV_SeveralCapacities"))
    test_suite.addTest(Test_EstAverageValueV_EnvQueueLossNetworkWithJobClasses("test_EnvQueueLossNetwork_MetMC_SingleCapacity"))
    test_suite.addTest(Test_EstAverageValueV_EnvQueueLossNetworkWithJobClasses("test_EnvQueueLossNetwork_MetFV_SingleCapacity"))

    # Run the test suites
    runner.run(test_suite)
