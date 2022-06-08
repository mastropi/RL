# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:19:38 2022

@author: Daniel Mastropietro
@description: Write unit tests on the estimators.py file
"""

import runpy
runpy.run_path('../../setup.py')

import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt
import unittest

from Python.lib import estimators
from Python.lib import simulators
from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners.continuing.fv import LeaFV
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep
from Python.lib.agents.queues import AgeQueue, PolicyTypes as QueuePolicyTypes, LearnerTypes
from Python.lib.environments import gridworlds, mountaincars, queues
from Python.lib.utils.basic import measure_exec_time


class Test_EstimatorValueFunctionOfflineDeterministicNextState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_grid = gridworlds.EnvGridworld1D(length=19+2)
        cls.env_mountain = mountaincars.MountainCarDiscrete(5)
            ## Note: we use more density to discretize the positions because we need to make sure that the car reaches the terminal state
            ## and this is not given if the discretization on the right side of the ramp is not dense enough because
            ## the new position is determined by x + v and vmax = 0.07, so the closest non-terminal x should 0.5 - 0.07 = 0.43
            ## where 0.5 = self.goal_position defined in mountain_cars.py in the MountainCarEnv class.

    def test_estimator_gridworld_random_walk_onecase(self):
        max_iter = 1000
        estimator = estimators.EstimatorValueFunctionOfflineDeterministicNextState(self.env_grid, gamma=1.0)
        niter, max_deltaV_abs, max_deltaV_rel_abs = estimator.estimate_state_values_random_walk(synchronous=True, max_delta=1E-6, max_delta_rel=np.nan, max_iter=max_iter)

        state_values = estimator.getV().getValues()
        print("Estimated state values on {}-state 1D gridworld environment after {} iterations out of {}, with max|delta(V)| = {}, max|delta_rel(V)| = {}:\n{}" \
              .format(self.env_grid.getNumStates(), niter, max_iter, max_deltaV_abs, max_deltaV_rel_abs, estimator.getV().getValues()))
        print("Average V = {}, Average |V| = {}".format(np.mean(state_values), np.mean(np.abs(state_values))))
        plt.plot(state_values, 'r.-')
        plt.title("Estimated value function V(s)")

    def test_estimator_mountaincar_random_walk_onecase(self):
        max_iter = 100
        estimator = estimators.EstimatorValueFunctionOfflineDeterministicNextState(self.env_mountain, gamma=1.0)
        niter, max_deltaV_abs, max_deltaV_rel_abs = estimator.estimate_state_values_random_walk(synchronous=True, max_delta=np.nan, max_delta_rel=1E-5, max_iter=max_iter,
                                                            reset_method=ResetMethod.ALLZEROS, reset_params=dict({'min': 0.2, 'max': 0.8}), reset_seed=1713)

        state_values = estimator.getV().getValues()
        state_values_2d = self.env_mountain.reshape_from_1d_to_2d(state_values)
        non_terminal_states = np.array(self.env_mountain.getNonTerminalStates())
        min_state_value_non_terminal, max_state_value_non_terminal =  np.min(state_values[non_terminal_states]), np.max(state_values[non_terminal_states])

        print("Estimated state values on {}-state Mountain Car environment after {} iterations out of {}, with max|delta(V)| = {}, max|delta_rel(V)| = {} (mean(V) = {:.6f}, mean|V| = {:.6f}):\n{}" \
              .format(self.env_mountain.getNumStates(), niter, max_iter, max_deltaV_abs, max_deltaV_rel_abs, np.mean(state_values), np.mean(np.abs(state_values)), state_values))

        plt.figure(figsize=(20,10))
        plt.plot(state_values, 'r.-')
        plt.title("Estimated value function (mean|V| = {})".format(np.mean(np.abs(state_values))))

        fig = plt.figure(figsize=(20,10))
        ax = fig.subplots(1,1)
        ax.plot(state_values, 'r.-')
        for x in range(0, np.prod(self.env_mountain.shape), self.env_mountain.shape[0]):
            plt.axvline(x, color="gray")
        ax.set_ylim((min_state_value_non_terminal, max_state_value_non_terminal))
        ax.set_xlabel("linear state index (grouped by {})".format(self.env_mountain.getShapeNames()[1]))
        ax.set_title("Estimated value function (scaled to non-terminal states")

        positions, velocities = self.env_mountain.get_positions(), self.env_mountain.get_velocities()
        fig = plt.figure(figsize=(20,10))
        ax = fig.subplots(1,1)
        ax.plot(positions, np.mean(state_values_2d,
                axis=self.env_mountain.getVelocityDimension()), '.-', color=self.env_mountain.getPositionColor()) # Sum along the different velocities
        ax.plot(velocities, np.mean(state_values_2d,
                axis=self.env_mountain.getPositionDimension()), '.-', color=self.env_mountain.getVelocityColor()) # Sum along the different positions
        ax.legend(["mean(V) on positions", "mean(V) on velocities"])
        ax.set_ylim((np.min(state_values[non_terminal_states]), np.max(state_values[non_terminal_states])))
        ax.set_xlabel("Position / Velocity")
        ax.set_ylabel("Average V")
        ax.set_title("Average V by dimension (position / velocity) (scaled on non-terminal states range)")


class Test_EstimatorQueueBlockingFlemingViot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Environment 1: single server queue with Poisson arrivals and departures
        nservers = 1
        capacity = 20 #5
        rates_birth = [0.7]
        rates_death = [1.0]
        queue_mm_single_server = queues.QueueMM(rates_birth, rates_death, nservers, capacity, origin=0.0)
        # We consider there is only one job arriving at the single server queue
        job_class_rates = queue_mm_single_server.getBirthRates()
        reward_func = None
        rewards_accept_by_job_class = [1.0]
        dict_params_reward_func = None
        cls.env_queue_mm_single_server = queues.EnvQueueSingleBufferWithJobClasses( queue_mm_single_server,
                                                                                    job_class_rates,
                                                                                    reward_func,
                                                                                    rewards_accept_by_job_class,
                                                                                    dict_params_reward_func)

        # Execution parameters that do not define the system
        cls.exec_params = { 'J': int(capacity/2),     # Size of the absorption set
                            'N': 100, #5,                   # Number of particles
                            'T': 2000, #20,                  # Number of arrival events
                            }

    @measure_exec_time
    def run_fv_estimation(self, env_queue, J, N, T, seed=1717):
        """
        Test the Fleming-Viot estimation of the blocking probability of a queue

        Arguments:
        env_queue: environment
            A queue environment out of those defined in environments/queues.py.

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
        # Environment
        envs_queue = [env_queue if i == 0 else copy.deepcopy(env_queue) for i in range(N)]
        # Agent interacting with the environment (which defines whether to accept or reject a job)
        # Define the agent acting on the queue environment
        # The agent blocks an incoming job when the buffer size of the queue is at its capacity K.
        # This is achieved by setting the parameter theta of the parameterized linear step acceptance policy
        # to the integer value K-1.
        K = self.env_queue_mm_single_server.getCapacity()
        policies = dict({QueuePolicyTypes.ACCEPT: PolQueueTwoActionsLinearStep(env_queue,
                                                                               float(K-1)),
                         QueuePolicyTypes.ASSIGN: None})
        learners = dict({LearnerTypes.V: LeaFV(self.env_queue_mm_single_server, gamma=1.0),
                         LearnerTypes.Q: None,
                         LearnerTypes.P: None})
        agent_accept_reject = AgeQueue(env_queue, policies, learners)
        dict_params_simul = dict({'buffer_size_activation': J,  # J-1 is the absorption buffer size
                                  'T': T,                       # number of arrivals at which the simulation stops
                                  'seed': seed})
        dict_params_info = dict()

        return simulators.estimate_blocking_fv( envs_queue, agent_accept_reject,
                                                dict_params_simul, dict_params_info)

    def test_fv_implementation_single_server(self):
        "Test the Fleming-Viot implementation of the blocking probability of a single-server queue system"
        nservers = self.env_queue_mm_single_server.getNumServers()
        rates_job = self.env_queue_mm_single_server.getJobClassRates()
        rates_service = self.env_queue_mm_single_server.getServiceRates()
        K = self.env_queue_mm_single_server.getCapacity()
        J = self.exec_params['J']
        N = self.exec_params['N']
        T = self.exec_params['T']
        proba_blocking_fv, expected_reward, probas_stationary, \
            expected_absorption_time, n_absorption_time_observations, \
                time_last_absorption, time_end_simulation_et, max_survival_time, \
                    n_events_et, n_events_fv_only = self.run_fv_estimation(self.env_queue_mm_single_server, J, N, T, seed=1313)

        print("EXECUTION PARAMETERS:")
        print("- # servers: {}".format(nservers))
        print("- job arrival rates at servers: {}".format(rates_job))
        print("- service rates: {}".format(rates_service))
        print("- capacity: {}".format(K))
        print("- absorption set size: {}".format(J))
        print("- # particles: {}".format(N))
        print("- # arrival time steps: {}".format(T))
        print("")

        print("ESTIMATION RESULTS:")
        print("- Blocking probability Pr(K={}) = {}".format(K, proba_blocking_fv))
        print("- Stationary probability Pr(K={}) = {}".format(K, probas_stationary[K]))
        self.assertAlmostEqual(proba_blocking_fv, probas_stationary[K])
        print("- Expected re-absorption time E(T) = {}".format(expected_absorption_time))
        print("- Number of re-absorption cycles = {}".format(n_absorption_time_observations))
        print("- Time of last absorption = {:.1f}".format(time_last_absorption))
        print("- Last observed time of the MC simulation for E(T) = {:.1f}".format(time_end_simulation_et))
        print("- Max observed survival time = {:.1f}".format(max_survival_time))
        print("- Number of events in MC simulation for E(T) = {}".format(n_events_et))
        print("- Number of events in FV simulation for Phi(t) = {}".format(n_events_fv_only))

        # Check results
        if nservers == 1 and K == 5 and rates_job == [0.7] and rates_service == [1.0] and \
            J == int(K/2) and N == 5 and T == 20:
            self.assertAlmostEqual(proba_blocking_fv, 0.0598497)
            self.assertAlmostEqual(expected_absorption_time, 5.704832, places=6)    # places = decimal places
            self.assertEqual(n_absorption_time_observations, 8)
            self.assertAlmostEqual(time_last_absorption, 45.6, places=1)
            self.assertAlmostEqual(time_end_simulation_et, 50.3, places=1)
            self.assertAlmostEqual(max_survival_time, 5.6, places=1)
            self.assertEqual(n_events_et, 40)
            self.assertEqual(n_events_fv_only, 47)
            # Execution time is about 0.15-0.20 sec (when manually averaged over 4-5 runs on a not charged PC --i.e. recently restarted)
        elif nservers == 1 and K == 20 and rates_job == [0.7] and rates_service == [1.0] and \
            J == int(K/2) and N == 100 and T == 2000:
            self.assertAlmostEqual(proba_blocking_fv, 0.00040698)
            self.assertAlmostEqual(expected_absorption_time, 40.14, places=2)    # places = decimal places
            self.assertEqual(n_absorption_time_observations, 70)
            self.assertAlmostEqual(time_last_absorption, 2809.5, places=1)
            self.assertAlmostEqual(time_end_simulation_et, 2840.9, places=1)
            self.assertAlmostEqual(max_survival_time, 37.1, places=1)
            self.assertEqual(n_events_et, 4003)
            self.assertEqual(n_events_fv_only, 6370)
        else:
            raise warnings.warn("No assertion has been executed. Check that the execution parameters fall into one of the asserted cases!")


if __name__ == '__main__':
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line

    #tests2run = ["value_function"]
    #tests2run = ["value_function", "fv"]
    tests2run = ["fv"]

    runner = unittest.TextTestRunner()

    # Run all tests
    #unittest.main()

    # Create a test suite for the value function estimator
    suite_value_function = unittest.TestSuite()
    suite_value_function.addTest(Test_EstimatorValueFunctionOfflineDeterministicNextState("test_estimator_gridworld_random_walk_onecase"))
    suite_value_function.addTest(Test_EstimatorValueFunctionOfflineDeterministicNextState("test_estimator_mountaincar_random_walk_onecase"))

    # Create a test suite for the FV estimator
    # Ref: https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
    suite_fv = unittest.TestSuite()
    suite_fv.addTest(Test_EstimatorQueueBlockingFlemingViot("test_fv_implementation_single_server"))

    if "value_function" in tests2run:
        runner.run(suite_value_function)
    
    if "fv" in tests2run:
        runner.run(suite_fv)
