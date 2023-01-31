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

import copy

import numpy as np

from Python.lib.agents.learners import LearnerTypes
from Python.lib.agents.learners.continuing.fv import LeaFV
from Python.lib.agents.policies import PolicyTypes
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep
from Python.lib.agents.queues import AgeQueue
from Python.lib.environments import queues
from Python.lib.environments.queues import rewardOnJobRejection_Constant
from Python.lib.simulators import SetOfStates
from Python.lib.simulators.queues import estimate_blocking_fv, SurvivalProbabilityEstimation

from Python.lib.utils.basic import get_current_datetime_as_string, measure_exec_time
from Python.lib.utils.computing import stationary_distribution_product_form, func_prod_knapsack


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
            cls.dict_env_queue_mm_single_server[K] = queues.EnvQueueSingleBufferWithJobClasses(
                dict_queue_mm_single_server[K],
                job_class_rates,
                reward_func,
                rewards_accept_by_job_class,
                dict_params_reward_func)

    @measure_exec_time
    def run_fv_estimation(self, env_queue, K, J, N, T,
                          burnin_time_steps, min_num_cycles_for_expectations, method_survival_probability_estimation,
                          seed=1717):
        """
        Test the Fleming-Viot estimation of the blocking probability of a queue

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
        policies = dict({PolicyTypes.ACCEPT: PolQueueTwoActionsLinearStep(env_queue,
                                                                          float(K - 1)),
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

    # -------- DATA -------
    # Case number, description, expected value, parameters
    # These are the same tests 1, 2 and 3 from data_test_lambda_return_random_walk
    data_test_EnvQueueSingleServer_MetFV_TestSeveralCapacities = lambda DEFAULT_EXECUTION = True: (
        # Note: True blocking probability for different K values when rho = 0.7
        # Pr(K) = rho^K * (1 - rho) / (1 - rho^(K+1))
        # K = 5: Pr(K) = 5.71%
        # K = 20: Pr(K) = 0.0239% = 2.39E-4
        # K = 40: Pr(K) = 0.000019% = 1.9E-7
        # Not so good estimation because:
        # (a) P(T>t) is estimated from M reabsorption cycles of the single particle simulation
        # (b) No burn-in period is used.
        # (c) No minimum number of cycles are required for the estimation of expectations.
        (1, DEFAULT_EXECUTION, 'Small K',
         {'K': 5, 'J': 2, 'N': 5, 'T': 20,
          'burnin_time_steps': 0, 'min_num_cycles_for_expectations': 0, 'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(FV)': 0.0598497, 'E(T)': 5.70, '#E(T)': 8, 'Tmax': 45.6, 'Tend': 50.3, 'Smax': 5.6,
          '#events_ET': 40, '#events_FV': 47}),
        (2, DEFAULT_EXECUTION, 'Moderate K',
         {'K': 20, 'J': 10, 'N': 100, 'T': 2000,
          'burnin_time_steps': 0, 'min_num_cycles_for_expectations': 0, 'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(FV)': 0.00040698, 'E(T)': 40.14, '#E(T)': 70, 'Tmax': 2809.5, 'Tend': 2840.9, 'Smax': 37.1,
          '#events_ET': 4003, '#events_FV': 6370}),
        (3, DEFAULT_EXECUTION, 'Large K',
         {'K': 40, 'J': 35, 'N': 400, 'T': 2000,
          'burnin_time_steps': 0, 'min_num_cycles_for_expectations': 0, 'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(FV)': 0.0059430, 'E(T)': 6.09, '#E(T)': 1, 'Tmax': 6.1, 'Tend': 2849.5, 'Smax': 3.2,
          '#events_ET': 4033, '#events_FV': 2239}),

        # Better estimation than above because items (b) and (c) are no longer true
        (4, DEFAULT_EXECUTION, 'Small K',
         {'K': 5, 'J': 2, 'N': 5, 'T': 20,
          'burnin_time_steps': 10, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(FV)': 0.0604309, 'E(T)': 5.65, '#E(T)': 7, 'Tmax': 45.6, 'Tend': 50.3, 'Smax': 5.6,
          '#events_ET': 40, '#events_FV': 47}),
        (5, DEFAULT_EXECUTION, 'Moderate K',
         {'K': 20, 'J': 10, 'N': 100, 'T': 2000,
          'burnin_time_steps': 10, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(FV)': 0.00040203, 'E(T)': 40.63, '#E(T)': 69, 'Tmax': 2809.5, 'Tend': 2840.9, 'Smax': 37.1,
          '#events_ET': 4003, '#events_FV': 6370}),
        (6, DEFAULT_EXECUTION, 'Large K',
         {'K': 40, 'J': 35, 'N': 400, 'T': 2000,
          'burnin_time_steps': 10, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_M_CYCLES},
         {'Pr(FV)': np.nan, 'E(T)': np.nan, '#E(T)': 0, 'Tmax': 6.1, 'Tend': 2849.5, 'Smax': 3.2,
          '#events_ET': 4033, '#events_FV': 0}),

        # Better estimation than above because in addition P(T>t) is estimated from the N particles used in the FV simulation
        (7, DEFAULT_EXECUTION, 'Small K',
         {'K': 5, 'J': 3, 'N': 80, 'T': 200,
          'burnin_time_steps': 20, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_N_PARTICLES},
            # NOTE: (2022/11/01) The estimated probability is too small compared to the true value...
            # And this is most likely due to a large estimated E(T) value... which should have been ~ 5 (see above tests),
            # but it turned out to be 8.65... why??
         {'Pr(FV)': 0.02774982, 'E(T)': 8.65, '#E(T)': 25, 'Tmax': 269.5, 'Tend': 291.2, 'Smax': 8.2,
          '#events_ET': 387, '#events_FV': 1126}),
        (8, DEFAULT_EXECUTION, 'Moderate K',
         {'K': 20, 'J': 10, 'N': 200, 'T': 2000,
          'burnin_time_steps': 10, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_N_PARTICLES},
         {'Pr(FV)': 0.000880075, 'E(T)': 40.26, '#E(T)': 69, 'Tmax': 2809.5, 'Tend': 2840.9, 'Smax': 77.0,
          '#events_ET': 4003, '#events_FV': 26250}),
        (9, DEFAULT_EXECUTION, 'Large K',
         {'K': 40, 'J': 20, 'N': 400, 'T': 2000,
          'burnin_time_steps': 10, 'min_num_cycles_for_expectations': 4,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_N_PARTICLES},
         {'Pr(FV)': 2.47624615E-06, 'E(T)': 585.21, '#E(T)': 4, 'Tmax': 2339.7, 'Tend': 2842.5, 'Smax': 100.2,
          '#events_ET': 4017, '#events_FV': 68224}),
    )

    # -------- DATA -------

    @data_provider(data_test_EnvQueueSingleServer_MetFV_TestSeveralCapacities)
    def test_EnvQueueSingleServer_MetFV_TestSeveralCapacities(self, casenum, run, desc, dict_params, dict_expected):
        "Test the Fleming-Viot implementation of the blocking probability of a single-server queue system"
        if run:
            print("\n*** Testing {}, case number {}: '{}' ***".format(self.id(), casenum, desc))
            K = dict_params['K']
            nservers = self.dict_env_queue_mm_single_server[K].getNumServers()
            rates_job = self.dict_env_queue_mm_single_server[K].getJobClassRates()
            rates_service = self.dict_env_queue_mm_single_server[K].getServiceRates()
            proba_blocking_fv, expected_reward, probas_stationary, \
                expected_absorption_time, n_absorption_time_observations, \
                    time_last_absorption, time_end_simulation_et, max_survival_time, time_end_simulation_fv, \
                        n_events_et, n_events_fv_only = self.run_fv_estimation(self.dict_env_queue_mm_single_server[K],
                                                                   dict_params['K'],
                                                                   dict_params['J'],
                                                                   dict_params['N'],
                                                                   dict_params['T'],
                                                                   dict_params['burnin_time_steps'],
                                                                   dict_params['min_num_cycles_for_expectations'],
                                                                   dict_params['method_survival_probability_estimation'],
                                                                   seed=1313)

            print(get_current_datetime_as_string())
            print("EXECUTION PARAMETERS:")
            print("- # servers: {}".format(nservers))
            print("- job arrival rates at servers: {}".format(rates_job))
            print("- service rates: {}".format(rates_service))
            print("- capacity: {}".format(K))
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

            # Check system setup is the one required to obtain the expected results
            assert nservers == 1, "Number of servers is 1"
            assert list(rates_job) == [0.7], "Arrival rate is 0.7"
            assert list(rates_service) == [1.0], "Service rate is 1.0"

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
        rates_birth = [0.5, 0.7, 0.8]
        rates_death = [1.0, 1.0, 1.0]
        nservers = len(rates_death)
        queue_mm_loss = queues.QueueMM(rates_birth, rates_death, nservers, +np.Inf, origin=0.0)
        reward_func = rewardOnJobRejection_Constant
        rewards_accept_by_job_class = None #[1.0] * len(rates_birth)
        dict_params_reward_func = None
        cls.env_queue_mm_loss = queues.EnvQueueLossNetworkWithJobClasses(
            queue_mm_loss,
            reward_func,
            rewards_accept_by_job_class,
            dict_params_reward_func)

    @measure_exec_time
    def run_fv_estimation(self, env_queue, Ks, Js, K, N, T,
                          burnin_time_steps, min_num_cycles_for_expectations, method_survival_probability_estimation,
                          seed=1717):
        """
        Test the Fleming-Viot estimation of the blocking probability of a queue

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
                                              for c in range(self.env_queue_mm_loss.getNumJobClasses())],
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

    def test_EnvQueueLossNetwork_MetFV_SingleCapacity(self):
        blocking_occupancies_per_jobclass = [4, 2, 5]
        #activation_occupancies_per_jobclass = [2, 1, 3]
        activation_occupancies_per_jobclass = [1, 1, 1]
        nservers = 10
        N = 1000
        T = 2000
        # (2023/01/26) Note about burnin and min cycles: Using (5, 5) or (50, 10) as (burnin_time_steps, min_cycles) didn't change much the estimated probability (e.g. from 14.454 to 14.453!)
        burnin_time_steps = 10
        min_num_cycles_for_expectations = 5
        method_survival_probability_estimation = SurvivalProbabilityEstimation.FROM_N_PARTICLES

        proba_blocking_fv, expected_reward, probas_stationary, \
            expected_absorption_time, n_absorption_time_observations, \
                time_last_absorption, time_end_simulation_et, max_survival_time, time_end_simulation_fv, \
                    n_events_et, n_events_fv_only = self.run_fv_estimation( self.env_queue_mm_loss,
                                                                            blocking_occupancies_per_jobclass,
                                                                            activation_occupancies_per_jobclass,
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
        rhos = np.array(job_class_rates) / np.array(service_rates)
        print("EXECUTION PARAMETERS:")
        print("- # servers (system's capacity): {}".format(nservers))
        print("- job arrival rates: {}".format(job_class_rates))
        print("- service rates: {}".format(service_rates))
        print("- loads: {}".format(rhos))
        print("- blocking occupancies per job class: {}".format(blocking_occupancies_per_jobclass))
        print("- activation occupancies per job class: {}".format(activation_occupancies_per_jobclass))
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

        # Check system setup is the one required to obtain the expected results
        #assert nservers == 10, "Number of servers is 10"
        #assert list(self.env_queue_mm_loss.getJobClassRates()) == [0.5, 0.7, 0.8], "Job arrival rates are [0.5, 0.7, 0.8]"
        #assert list(self.env_queue_mm_loss.getServiceRates()) == [1.0, 1.0, 1.0], "Service rates are [1.0, 1.0, 1.0]"

        # True stationary probability of blocking
        x, dist = stationary_distribution_product_form(nservers, rhos, func_prod_knapsack)
        proba_stationary_true = np.sum([v for xx, v in zip(x, dist) if tuple(xx) in probas_stationary.keys()])
        print("")
        print("- Stationary probability of ALL blocking states = {}".format(proba_stationary))
        print("- TRUE Stationary probability of ALL blocking states = {}".format(proba_stationary_true))
        assert np.isclose(proba_stationary, proba_stationary_true)


if __name__ == '__main__':
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line

    runner = unittest.TextTestRunner()

    # Run all tests
    # unittest.main()

    # Create the test suites
    test_suite = unittest.TestSuite()
    test_suite.addTest(Test_EstAverageValueV_EnvQueueLossNetworkWithJobClasses("test_EnvQueueLossNetwork_MetFV_SingleCapacity"))
    test_suite.addTest(Test_EstAverageValueV_EnvQueueSingleServer("test_EnvQueueSingleServer_MetFV_TestSeveralCapacities"))

    # Run the test suites
    runner.run(test_suite)
