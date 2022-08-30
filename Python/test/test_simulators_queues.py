# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:55:27 2022

@author: Daniel Mastropietro
@description: Unit tests for functions and methods defined in simulators.queues.
@details: Naming conventions follow the instructions given in test_conventions.txt.
"""
import unittest

import numpy as np

from Python.lib.agents.policies.job_assignment import PolJobAssignmentProbabilistic

from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses
import Python.lib.queues as queues
from Python.lib.queues import Event

from Python.lib.simulators.queues import compute_job_rates_by_server, generate_event, SimulatorQueue


class Test_Support_EnvQueueMultiServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Define the queue environment
        capacity = 10
        nservers = 4
        job_class_rates = [0.7, 0.5]
        service_rates = [1.0, 0.8, 1.3, 0.75]

        # Job class to server assignment probabilities so that we can compute the job rates by server (needed to create the QueueMM object)
        policy_assignment_probabilities = [[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.4, 0.1]]
        policy_assign = PolJobAssignmentProbabilistic(policy_assignment_probabilities)
        job_rates_by_server = compute_job_rates_by_server(job_class_rates, nservers, policy_assignment_probabilities)

        # Queue M/M/nservers/capacity object
        queue = queues.QueueMM(job_rates_by_server, service_rates, nservers, capacity)
        cls.env_queue_mm = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, None, None)

    def test_function_generate_event(self):
        print("\n")
        print("".join(np.repeat("*", 20)))
        print("Testing generate_event() method on a multi-server queue system defined in simulators.queues...")

        # The simulation object
        simul = SimulatorQueue(self.env_queue_mm, None, None, N=1)

        np.random.seed(1717)
        time, event, job_class_or_server, idx_env = generate_event(simul.envs)
        print("Observed minimum time {} for an {} associated to job class (if BIRTH) or server (if DEATH) = {}".format(time, event, job_class_or_server))
        time_expected, event_expected, job_class_or_server_expected, idx_env_expected = 0.36603, Event.BIRTH, 0, 0
        assert np.allclose(time, time_expected, atol=1E-5)
        assert event == event_expected
        assert job_class_or_server == job_class_or_server_expected
        assert idx_env == idx_env_expected

        # Repeat the realization to check that we always get a BIRTH event because by default the queue sizes of the servers are initialized at 0
        N = 10
        for i in range(N):
            print("Generating event number {} out of {}...".format(i+1, N))
            _, event, _, _ = generate_event(simul.envs)
            assert event == Event.BIRTH

        #-- Repeat the realization to check the distribution of event types and selected job classes or servers
        # We set a random seed so that we embrace many more test cases as every time we run this test we get new realizations!
        np.random.seed(None)
        # Set two of the servers to a positive queue size
        simul.envs[0].setState(([0, 1, 0, 2], None))

        # Keep track of the number of events observed for each rate (either a job class rate or a service rate)
        rates = list(self.env_queue_mm.getJobClassRates()) + list(self.env_queue_mm.getServiceRates())
        valid_rates = list(self.env_queue_mm.getJobClassRates()) + [r if s > 0 else np.nan for r, s in zip(simul.envs[0].getServiceRates(), simul.envs[0].getQueueState())]
        n_events_by_rate = np.zeros(len(rates))
        N = 100
        # Expected proportion of events for each exponential
        p = np.array( [r / np.nansum(valid_rates) if r > 0 else 0.0 for r in valid_rates] )
        for _ in range(N):
            time, event, job_class_or_server, _ = generate_event(simul.envs)
            if event == Event.BIRTH:
                # The BIRTH events are associated to indices at the beginning of the rates list
                n_events_by_rate[job_class_or_server] += 1
            elif event == Event.DEATH:
                # The DEATH event are associated to indices at the end of the rates list
                # so we sum the returned index to the number of job class rates.
                n_events_by_rate[len(self.env_queue_mm.getJobClassRates()) + job_class_or_server] += 1
        # Observed proportions of events by server and type of event
        phat = 1.0 * n_events_by_rate / N
        se_phat = np.sqrt(p * (1 - p) / N)
        print("EXPECTED / OBSERVED / SE proportions of events by rate on N={}:\n{}".format(N, np.c_[p, phat, se_phat]))
        print("NOTE: If the test fails, run it again, because it may have failed just by chance due to the random seed chosen when running the test.")
        assert np.allclose(phat, p, atol=3 * se_phat)  # true probabilities should be contained in +/- 3 SE(phat) from phat



if __name__ == "__main__":
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
    runner = unittest.TextTestRunner()

    # Run all tests
    # unittest.main()

    # Multi-server tests
    test_suite_multiserver = unittest.TestSuite()
    test_suite_multiserver.addTest(Test_Support_EnvQueueMultiServer("test_function_generate_event"))
    runner.run(test_suite_multiserver)
