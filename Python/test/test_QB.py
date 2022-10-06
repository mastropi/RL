# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:16:02 2020

@author: Daniel Mastropietro
@description: Test the blocking probability estimation in queues.
"""

import runpy
runpy.run_path('../../setup.py')

import os
import sys
import numpy as np
import pandas as pd
import copy
import re
from enum import Enum, unique

from warnings import warn
from timeit import default_timer as timer
import unittest
import matplotlib
from matplotlib import pyplot as plt, cm, ticker as mtick

from Python.lib.utils.basic import aggregation_bygroups, is_scalar, get_current_datetime_as_string, get_datetime_from_string
import Python.lib.utils.plotting as plotting

from Python.lib.agents.learners import LearnerTypes
from Python.lib.agents.learners.continuing.fv import LeaFV

from Python.lib.agents.policies import PolicyTypes
from Python.lib.agents.policies.job_assignment import PolJobAssignmentProbabilistic
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep

from Python.lib.agents.queues import AgeQueue

import Python.lib.queues as queues  # The keyword `queues` is used in the code
from Python.lib.queues import Event
from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses, rewardOnJobRejection_ExponentialCost

import Python.lib.estimators as estimators
from Python.lib.estimators import FinalizeType, plot_curve_estimates

from Python.lib.simulators.queues import estimate_blocking_fv, estimate_blocking_mc

from Python.lib.utils.computing import get_server_loads, compute_job_rates_by_server, compute_blocking_probability_birth_death_process

DEFAULT_NUMPY_PRECISION = np.get_printoptions().get('precision')
DEFAULT_NUMPY_SUPPRESS = np.get_printoptions().get('suppress')


@unique
class Process(Enum):
    Estimators = 1
    Simulators = 2


class Test_QB_Particles(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.log = False
        self.capacity = 5
        self.rate_birth = 0.7
        self.nservers = kwargs.pop('nservers', 1)
        super().__init__(*args, **kwargs)

        if self.nservers == 1:
            # One server
            self.job_class_rates = [self.rate_birth]
            self.rate_death = [1]
            self.policy_assign = PolJobAssignmentProbabilistic([[1]])
        elif self.nservers == 3:
            # Multiple servers
            self.job_class_rates = [0.8, 0.7]
            self.rate_death = [1, 1, 1]
            self.policy_assign = PolJobAssignmentProbabilistic([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
        else:
            raise ValueError("Given Number of servers ({}) is invalid. Valid values are: 1, 3".format(nservers))
        # rho rates for each server based on arrival rates and assignment probabilities
        self.rate_birth = compute_job_rates_by_server(self.job_class_rates, self.nservers, self.policy_assign.getProbabilisticMap())
        self.rhos = get_server_loads(self.rate_birth, self.rate_death)

        self.queue = queues.QueueMM(self.rate_birth, self.rate_death, self.nservers, self.capacity)

        self.plotFlag = True

    def tests_on_one_queue(self):
        print("\nRunning test " + self.id())
        # Cannot remove non-existing job
        self.queue.reset()  
        self.queue.remove()
        assert self.queue.size == 0

        # Add a job increases the queue size by one in non-limit situations
        self.queue.reset()
        self.queue.add()
        assert self.queue.size == 1

        # Cannot add a job in a full queue
        self.queue.reset()
        for i in range(self.capacity):
            self.queue.add()
        assert self.queue.size == self.capacity
        self.queue.add()
        assert self.queue.size == self.capacity

    def deprecated_test_simulation_of_events_on_a_system_with_one_particle(self):
        print("\nRunning test " + self.id())
        nmeantimes = 100
        est = estimators.EstimatorQueueBlockingFlemingViot(1, self.queue, [0.5],
                                                           nmeantimes=nmeantimes,
                                                           reactivate=False)
        
        # The state udpate process is correctly done. Let aNET = "array with Next Event Times",
        # then the following assertions hold.
        est.reset()
        print("The test runs on a simulation with {} iterations" \
              "\n(no pre-specified seed, so that different test runs produce even more tests!)".format(nmeantimes))
        time_event_prev = 0.0
        for it in range(nmeantimes):
            # aNET = Array of Next Event Times
            aNET = est.get_times_next_events(0)
            aNET_prev = aNET.copy()
            est._update_one_particle(0)
            
            # Assertions about the time of next event compared to the POSSIBLE times for the next event
            #print("Array of Next Event Times that gives rise to the ACTUAL next event: {}".format(aNET_prev))
            #print("Time of ACTUAL next event: {}".format(est.get_time_last_event(0)))
            assert len(aNET) == 2, \
                    "There are always two elements in the array of Next Event Times"
            assert np.allclose(est.get_time_last_event(0) - time_event_prev, np.min(aNET_prev)), \
                    "The time elapsed from the previous event is equal to the smallest time stored" \
                    " in the array of Next Event Times"

            # Assertions about the time of the event NOT applied
            type_event_applied = est.get_type_last_event(0)
            type_event_not_applied = Event(1 - type_event_applied.value) 
            assert  aNET[type_event_not_applied.value] > 0 and \
                    aNET[type_event_not_applied.value] == aNET_prev[type_event_not_applied.value] - aNET_prev[type_event_applied.value], \
                    "The next event time of the event type NOT applied is updated" \
                    " to the its original time-to-event minus the time to the ACTUAL event, and is POSITIVE"

            # Assertions about the list that stores the order of the possible next event times
            time_next_birth = aNET[Event.BIRTH.value]
            time_next_death = aNET[Event.DEATH.value]
            assert est._order_times_next_events[0] == [Event.BIRTH.value, Event.DEATH.value] and time_next_birth < time_next_death \
                or est._order_times_next_events[0] == [Event.DEATH.value, Event.BIRTH.value] and time_next_birth > time_next_death, \
                    "The list that stores the order of next event times reflects the true order"

            time_event_prev = est.get_time_last_event(0)

    def run_test_compute_counts(self,   reactivate,
                                        finalize_type=FinalizeType.ABSORB_CENSORED,
                                        nparticles=5,
                                        nmeantimes=20,
                                        seed=1713,
                                        log=False): 
        est = estimators.EstimatorQueueBlockingFlemingViot(nparticles, self.queue, [0.5],
                                                           nmeantimes=nmeantimes,
                                                           reactivate=reactivate,
                                                           finalize_type=finalize_type,
                                                           plotFlag=True,
                                                           seed=seed, log=log)        
        est.reset()
        for it in range(nmeantimes):
            est.generate_one_iteration(it+1)
            N1 = n_active_particles = est.get_number_active_particles()
            if np.mod(it, int(nmeantimes/10)) == 0:
                print("Iteration {} of {}... ({} active particles)".format(it+1, nmeantimes, N1))
                
        # Recompute the counts and check they are the same as the online calculation
        counts_alive_orig = est.counts_alive.copy()
        counts_blocked_orig = est.counts_blocked.copy()
        est.compute_counts()

        print("\tAsserting counts...")
        if  True or len(est.counts_alive) != len(counts_alive_orig) or \
            len(est.counts_blocked) != len(counts_blocked_orig) or \
            est.counts_alive != counts_alive_orig or \
            est.counts_blocked != counts_blocked_orig:
            print("ALIVE:")
            print("orig: \n{}".format(counts_alive_orig))
            print("new: \n{}".format(est.counts_alive))
            print("BLOCKED:")
            print("orig: \n{}".format(counts_blocked_orig))
            print("new: \n{}".format(est.counts_blocked))

        assert len(est.counts_alive) == len(counts_alive_orig), \
                "The length of the two counts of ALIVE particles list are the same (orig={}, new={})" \
                .format(len(counts_alive_orig), len(est.counts_alive))
        assert len(est.counts_blocked) == len(counts_blocked_orig), \
                "The length of the two counts of blocked particles list are the same (orig={}, new={})" \
                .format(len(counts_blocked_orig), len(est.counts_blocked))

        assert est.counts_alive == counts_alive_orig, \
                "The counts of ALIVE particles coincides with the online calculation" \
                "\n{}".format(np.c_[counts_alive_orig, est.counts_alive])
        assert est.counts_blocked == counts_blocked_orig, \
                "The counts of BLOCKED particles coincides with the online calculation" \
                "\n{}".format(np.c_[counts_blocked_orig, est.counts_blocked])

    def no_test_compute_counts(self, log=False):
        print("\nRunning test " + self.id())
        seed = 1713
        all_nparticles = [5, 10, 20, 30]
        all_nmeantimes = [20, 50]


        reactivate = True
        finalize_type = FinalizeType.ABSORB_CENSORED
        print("\n***** START OF reactivate={}, finalize_type={}...".format(reactivate, finalize_type))
        for nparticles in all_nparticles:
            for nmeantimes in all_nmeantimes:
                print("\nRunning with reactivate={}, finalize_type={}...".format(reactivate, finalize_type))
                print("\tnparticles={}, nmeantimes={}...".format(nparticles, nmeantimes))
                self.run_test_compute_counts(reactivate=reactivate,
                                            finalize_type=finalize_type,
                                            nparticles=nparticles,
                                            nmeantimes=nmeantimes,
                                            seed=seed,
                                            log=log)

        reactivate = True
        finalize_type = FinalizeType.REMOVE_CENSORED
        print("\n***** START OF reactivate={}, finalize_type={}...".format(reactivate, finalize_type))
        for nparticles in all_nparticles:
            for nmeantimes in all_nmeantimes:
                print("\nRunning with reactivate={}, finalize_type={}...".format(reactivate, finalize_type))
                print("\tnparticles={}, nmeantimes={}...".format(nparticles, nmeantimes))
                self.run_test_compute_counts(reactivate=reactivate,
                                            finalize_type=finalize_type,
                                            nparticles=nparticles,
                                            nmeantimes=nmeantimes,
                                            seed=seed,
                                            log=log)

    def no_tests_on_n_particles(self,   reactivate=False,
                                     finalize_type=FinalizeType.ABSORB_CENSORED,
                                     nparticles=5,
                                     nmeantimes=20,
                                     seed=1713,
                                     log=False): 
        print("\nRunning test " + self.id())
        #nparticles = 30
        #nmeantimes = 200
        #reactivate = True
        #finalize_type = FinalizeType.ABSORB_CENSORED
        #finalize_type = FinalizeType.REMOVE_CENSORED
        #seed = 1713
        est = estimators.EstimatorQueueBlockingFlemingViot(nparticles, self.queue, [0.5],
                                                           nmeantimes=nmeantimes,
                                                           reactivate=reactivate,
                                                           finalize_type=finalize_type,
                                                           plotFlag=True,
                                                           seed=seed, log=log)
        print("Simulation setup:")
        print(est.setup())
        
        est.reset()
        for it in range(nmeantimes):
            est.update_state(it+1)
            N1 = n_active_particles = est.get_number_active_particles()
            if np.mod(it, int(nmeantimes/10)) == 0:
                print("Iteration {} of {}... ({} active particles)".format(it+1, nmeantimes, N1))

            if self.log:
                print("------ END OF ITER {} of {} ------".format(it+1, nmeantimes))

            # The list storing the times when particles became active is sorted increasingly
            particles, activation_times = est.get_all_activation_times()
            if self.log:
                np.set_printoptions(precision=3)
                print("activation times: {}".format(np.array(activation_times)))
                np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION)
            assert sorted(activation_times) == activation_times, \
                    "The absolute times of activation are sorted: {}".format(np.array(activation_times))
            assert len(activation_times) >= N1, \
                    "The number of activation times ({}) is at least equal to the number of active particles (N1 = {})" \
                    .format(len(activation_times), N1)

            particles, absorption_times = est.get_all_absorption_times()
            if self.log:
                np.set_printoptions(precision=3)
                print("absorption times: {}".format(np.array(absorption_times)))
                np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION)
            assert sorted(absorption_times) == absorption_times, \
                    "The (relative) absorption times are sorted: {}".format(np.array(absorption_times))            

            # The list storing the time segments where statistics are computed is sorted
            survival_time_segments = est.get_survival_time_segments()
            if self.log:
                np.set_printoptions(precision=3)
                print("time segments: {}".format(np.array(survival_time_segments)))
                np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION)
            assert sorted(survival_time_segments) == survival_time_segments, \
                    "The time segments are sorted: {}".format(np.array(survival_time_segments))
            assert survival_time_segments[0] == 0.0, \
                    "The first element of the time segments list ({}) is equal to 0" \
                    .format(survival_time_segments[0])
            #assert survival_time_segments[1:] == absorption_times, \
            #        "The ending time segments ({}) are equal to the absorption times ({})" \
            #        .format(survival_time_segments, absorption_times)

            counts_particles_alive_by_elapsed_time = est.get_counts_particles_alive_by_elapsed_time()
            if self.log:
                print("Counts by segment: {}".format(counts_particles_alive_by_elapsed_time))
            assert np.all( counts_particles_alive_by_elapsed_time == np.arange(len(survival_time_segments)-1, -1, -1) ), \
                    "Since the time segments are defined by jumps in the count of active particles" \
                    " they should decrease linearly to 0 starting at {} ({})" \
                    .format(len(survival_time_segments)-1, counts_particles_alive_by_elapsed_time)
            if self.log:
                print("------ END OF ITER {} of {} ------".format(it+1, nmeantimes))

        print("\n\n****** SIMULATION SUMMARY ({} iterations) ******".format(nmeantimes))

        #************************************************************************
        # GRAPHICAL RENDER
        if True:
            print("\nSIMULATION RENDER (before treatment of censored values):")
            print(est.render())
        #************************************************************************


        #************************************************************************
        # FINALIZE THE SIMULATION PROCESS BY DEALING WITH ACTIVE PARTICLES (censoring)
        est.finalize()
        print("\n\n****** ESTIMATION PROCESS ({} iterations) ******".format(nmeantimes))
        #************************************************************************


        #************************************************************************
        # GRAPHICAL RENDER
        if True:
            print("\nSIMULATION RENDER (after treatment of censored values):")
            print(est.render())
        #************************************************************************



        survival_time_segments = est.get_survival_time_segments()
        if False:
            np.set_printoptions(precision=3, suppress=True)
            print("Survival time segments: {}".format(np.array(survival_time_segments)))
            np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)
        assert sorted(survival_time_segments) == survival_time_segments, \
                "The survival time segments are sorted: {}".format(np.array(survival_time_segments))
        assert survival_time_segments[0] == 0.0, \
                "The first element of the survival time segments list ({}) is equal to 0" \
                .format(survival_time_segments[0])

        counts_particles_alive_by_elapsed_time = est.get_counts_particles_alive_by_elapsed_time()
        if False:
            print("Counts by survival segment: {}".format(counts_particles_alive_by_elapsed_time))
        assert np.all( counts_particles_alive_by_elapsed_time == np.arange(len(survival_time_segments)-1, -1, -1) ), \
                "Since the time segments are defined by jumps in the count of active particles" \
                " they should decrease linearly to 0 starting at {} ({})" \
                .format(len(survival_time_segments)-1, counts_particles_alive_by_elapsed_time)

        blocking_time_segments = est.get_blocking_time_segments()
        if False:
            np.set_printoptions(precision=3, suppress=True)
            print("Blocking time segments: {}".format(np.array(blocking_time_segments)))
            np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)
        assert sorted(blocking_time_segments) == blocking_time_segments, \
                "The blocking time segments are sorted: {}".format(np.array(blocking_time_segments))
        assert blocking_time_segments[0] == 0.0, \
                "The first element of the blocking time segments list ({}) is equal to 0" \
                .format(blocking_time_segments[0])

        counts_particles_blocked_by_elapsed_time = est.get_counts_particles_blocked_by_elapsed_time()
        if False:
            print("Counts by blocking segment: {}".format(counts_particles_blocked_by_elapsed_time))
        assert counts_particles_blocked_by_elapsed_time[0] == 0, \
                "The first element of the counts of blocked particles list ({}) is equal to 0" \
                .format(counts_particles_blocked_by_elapsed_time[0])
        assert len([c for c in counts_particles_blocked_by_elapsed_time if c < 0]) == 0, \
                "All counts in the counts of blocked particles list are non-negative ({})" \
                .format(counts_particles_blocked_by_elapsed_time)

        print("\nLatest time the system was updated to a known state: {:.3f}".format( est.get_time_latest_known_state() ))
        times_last_event_by_particle = est.get_all_times_last_event()
        print("Range of latest event times in all {} particles in the system: [{:.3f}, {:.3f}]" \
              .format(est.N, np.min(times_last_event_by_particle), np.max(times_last_event_by_particle)))
        print("Latest particle positions: {}".format(est.positions))
        np.set_printoptions(precision=3, suppress=True)
        print("Latest event times: {}".format(np.array(times_last_event_by_particle)))
        particles, elapsed_times_since_activation = est.get_all_elapsed_times()
        print("Latest elapsed times since activation: {}".format(np.array(elapsed_times_since_activation)))
        print("Particles associated to these times  : {}".format(particles))
        np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

        #print("\nESTIMATIONS *** METHOD 1 ***:")
        df_proba_survival_and_blocking_conditional_BF = est.estimate_proba_survival_and_blocking_conditional()
        #with np.printoptions(precision=3, suppress=True):
        #    print(df_proba_survival_and_blocking_conditional_BF)

        print("\nESTIMATIONS *** METHOD 2: FROM observed TIMES ***:")

        ##### IMPORTANT: THE FOLLOWING RECOMPUTATION OF COUNTS SHOULD ALWAYS BE DONE WHEN finalize_type = REMOVE!!
        ##### In fact, the finalize() process by REMOVE changes the counts since it removes time segments!!   
        #est.compute_counts()
        ##### IMPORTANT

        df_proba_survival_and_blocking_conditional = est.estimate_proba_survival_and_blocking_conditional()
        np.set_printoptions(precision=3, suppress=True)
        print("Estimated probabilities by time:")
        print(df_proba_survival_and_blocking_conditional)
        np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

        if False and est.finalize_type != FinalizeType.REMOVE_CENSORED:
            # Only compare probabilities when censored times are NOT removed
            # (because when censored times are removed, the information stored in
            # the counts_alive and counts_blocked lists is quite different between the two different methods) 
            print("\nEqual survivals and blocking probs?")
            df_comparison = (df_proba_survival_and_blocking_conditional_BF == df_proba_survival_and_blocking_conditional)
            equal = np.min(np.min(df_comparison))
            print(equal)
            if not equal:
                print(df_comparison)

        if self.log:
            print("Total blocking time by particle:")
            for p in range(est.N):
                df_blocking_periods = est.get_blocking_periods(p)
                print("\n\nParticle {}".format(p))
                print("Block periods:")
                print(df_blocking_periods)
                
                df_survival_periods = est.get_survival_periods(p)
                print("\nSurvival periods:")
                print(df_survival_periods)
        
        
                total_blocking_time = est.get_total_blocking_time(p)
                total_survival_time = est.get_total_survival_time(p)
                print("")
                print("Total blocking time = {:.3f}".format(total_blocking_time))
                print("Total survival time = {:.3f}".format(total_survival_time))
                print("% blocking time = {:.1f}%".format(total_blocking_time / total_survival_time * 100))
                print("\n")

        print("\nTotal blocking time for ALL particles:")
        all_total_blocking_time = est.get_all_total_blocking_time()
        all_total_survival_time = est.get_all_total_survival_time()
        print("Total blocking time (all particles) = {:.3f}".format(all_total_blocking_time))
        print("Total survival time (all particles) = {:.3f}".format(all_total_survival_time))
        print("% blocking time = {:.1f}%".format(all_total_blocking_time / all_total_survival_time * 100))
        print("\n")

        proba_blocking = est.estimate_proba_blocking()
        K = self.capacity
        print("\nBlocking probability estimate: {:.1f}%".format(proba_blocking*100))
        proba_blocking_K = compute_blocking_probability_birth_death_process(self.rhos, K)
        if proba_blocking_K is not None:
            print("Theoretical value: {:.1f}%".format(proba_blocking_K*100))

        print("Simulation setup:")
        print(est.setup())

        return  df_proba_survival_and_blocking_conditional, \
                (reactivate, finalize_type, nparticles, nmeantimes)

    @classmethod
    def test_fv_implementation(cls,
                               estimation_process=Process.Estimators,
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
        rate_birth = 0.7
        if nservers == 1:
            job_class_rates = [rate_birth]
            rate_death = [1.0]
            policy_assign = PolJobAssignmentProbabilistic([[1]])
            rewards_accept_by_job_class = None # [1]
        elif nservers == 3:
            job_class_rates = [0.8, 0.7]
            rate_death = [1.0, 1.0, 1.0]
            policy_assign = PolJobAssignmentProbabilistic([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
            rewards_accept_by_job_class = None # [1, 1]
        else:
            raise ValueError("Given Number of servers ({}) is invalid. Valid values are: 1, 3".format(nservers))

        # Create the queue, the queue environment, and the agent acting on it
        queue = queues.QueueMM(rate_birth, rate_death, nservers, K)
        env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates=job_class_rates,
                                                       reward_func=rewardOnJobRejection_ExponentialCost,
                                                       rewards_accept_by_job_class=rewards_accept_by_job_class)
        # Define the agent acting on the queue environment
        policies = dict({PolicyTypes.ACCEPT: PolQueueTwoActionsLinearStep(env_queue, float(K-1)), PolicyTypes.ASSIGN: policy_assign})
        learners = dict({LearnerTypes.V: LeaFV(env_queue, gamma=1.0),
                         LearnerTypes.Q: None,
                         LearnerTypes.P: None})
        agent = AgeQueue(queue, policies, learners)
        job_rates_by_server = compute_job_rates_by_server(job_class_rates, nservers, policy_assign.getProbabilisticMap())

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

        print("System: # servers={}, K={}, rhos={}, buffer_size_activation={}, #burn-in absorption cycles={}" \
              .format(nservers, K, get_server_loads(job_rates_by_server, rate_death), buffer_size_activation_value, burnin_cycles_absorption))
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

                if case + 1 == 1 and r == 1:
                    # Compute the true blocking probability only ONCE, namely for the first replication of the first N value case
                    rhos = [l / m for l, m in zip(job_rates_by_server, env_queue.getServiceRates())]
                    print("Computing TRUE blocking probability for nservers={}, K={}, rhos={}...".format(nservers, K, rhos))
                    proba_blocking_true = compute_blocking_probability_birth_death_process(rhos, K)

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

    def analyze_estimates(  self,
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

        # Upper bound for the number of simulations to run
        # (it's an upper bound because when buffer_size_activation is a proportion, there may be repeated values for the actual buffer_size_activation used) 
        nsimul = int(   len(K_values) * \
                        len(buffer_size_activation_values))
    
        np.random.seed(seed)
        case = 0
        idx_K = -1
        print("System: # servers = {}, rhos = {}".format(self.nservers, self.rhos))

        time_start = timer()
        queue = copy.deepcopy(self.queue)
        for K in K_values:
            idx_K += 1
            queue.K = K
            nparticles = nparticles_values[idx_K]
            nmeantimes = nmeantimes_values[idx_K]
            print("\n\n---> NEW K (Queue's capacity = {})".format(queue.getCapacity()))
            print("---> (nparticles={}, nmeantimes={})" \
                  .format(nparticles, nmeantimes_values))

            print("Computing TRUE blocking probability...", end=" --> ")
            time_pr_start = timer()
            proba_blocking_true = compute_blocking_probability_birth_death_process(self.rhos, K)
            time_pr_end = timer()
            print("{:.1f} sec".format(time_pr_end - time_pr_start))
            print("Pr(K)={:.6f}%".format(proba_blocking_true*100))

            # Create the queue environment that is simulated below
            env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates=self.job_class_rates, reward_func=rewardOnJobRejection_ExponentialCost, rewards_accept_by_job_class=None) #[1] * len(self.job_class_rates))
            # Define the agent acting on the queue environment
            policies = dict({PolicyTypes.ACCEPT: None, PolicyTypes.ASSIGN: self.policy_assign})
            learners = None
            agent = AgeQueue(queue, policies, learners)

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
                                                                ('rhos', [len(self.rhos) == 1 and self.rhos or len(self.rhos) > 1 and format(self.rhos)]),
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
                        plot_curve_estimates(df_new_functions,
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
                  .format(self.nservers, self.rhos, K, nparticles, est_fv.getMaxSimulationTime(), nmeantimes, est_fv.getMaxNumberOfEvents()))
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

    @classmethod
    def plot_aggregated_convergence_results(cls, results_convergence):

        def plot(df, df_agg,
                 grp_K, grp_axis, grp_legend,
                 replications):
            """
            The input data frame should be aggregated by K and two other variables that define
            the grp_axis and the grp_legend, IN THIS ORDER and be created via the pd.groupby()
            method using as_index=True.
            """
            K_values, _, _ = df_agg.index.levels

            # Get the index position of each "index variable"
            # IMPORTANT: We CANNOT retrieve these values from the above df_agg.index.levels
            # call because we don't know the position of the `axis` variable and of the `legend`
            # variable among the index columns.
            grp_axis_idx = df_agg.index.names.index(grp_axis)
            grp_legend_idx = df_agg.index.names.index(grp_legend)
            axis_values = df_agg.index.levels[grp_axis_idx]
            legend_values = df_agg.index.levels[grp_legend_idx]
            se_mult = 2
            axes = plt.figure(figsize=(9,4*len(K_values))).subplots(len(K_values),2)
            cmap_red = cm.get_cmap('Reds')
            cmap_green = cm.get_cmap('Greens')
            color_norm = matplotlib.colors.Normalize(vmin=0.8*legend_values[0], vmax=legend_values[-1])
            for idx, K in enumerate(K_values):
                ind_K = (df[grp_K] == K)
                if len(K_values) == 1:
                    (ax_mc, ax_fv) = axes
                else:
                    (ax_mc, ax_fv) = axes[idx]
                legend_lines_mc  = []
                legend_lines_fv  = []
                legend_labels_mc = []
                legend_labels_fv = []
                for idx_legend, legend_value in enumerate(legend_values):
                    print("Plotting for K={}, legend value {}...".format(K, legend_value))
                    try:
                        # Subset the values to plot for the current K value and legend
                        # The subsetting step depends on the position of the legend variable
                        # among the index variables.

                        # NOTE: We need to `try` this process because the legend values
                        # may NOT happen in all grouping levels, in which case we would have
                        # an index error.
                        if grp_legend_idx == 1:
                            # The legend variable is the second of the index columns
                            # Note: this subsetting can yield an error when the legend value
                            # is NOT a possible value for the current K value.
                            df2plot = df_agg.loc[(K, legend_value)]
                        else:
                            # The legend variable is the third of the index columns
                            # => we need to list all possible values of the second index columns to subset the data for plotting
                            # Note: this subsetting CANNOT yield an error even when some of the
                            # axis_values do NOT take place at the current K value, because
                            # Python filters the data frame to the axis_values occurring for the
                            # current K.
                            df2plot = df_agg.loc[(K, list(axis_values), legend_value)]

                        # Color for the lines for each legend
                        # (the line for the last legend --which corresponds to the largest parameter value
                        # expected to be the best setting, because it has more particles/longer-simulation-time
                        # is plotted with the HIGHEST COLOR INTENSITY, while the rest are plotted with lighter colors
                        # selected from the respective color map)
                        if idx_legend == len(legend_values) - 1:
                            color_red = "red"
                            color_green = "green"
                        else:
                            color_strength = color_norm(legend_value)
                            color_red = cmap_red(color_strength)
                            color_green = cmap_green(color_strength)

                        # Lines in each plot (shown with error bars)
                        yerr_mc = se_mult*df2plot[y_mc]['SE']
                        yerr_fv = se_mult*df2plot[y_fv]['SE']
                        yerr_ref_mc = se_mult*df2plot[proba_true]['SE']       # Reference value is the blocking probability
                        line_mc = ax_mc.errorbar(axis_values, df2plot[y_mc]['mean']*100, yerr=yerr_mc*100,
                                    capsize=4, color=color_red, marker='.')
                        line_fv = ax_fv.errorbar(axis_values, df2plot[y_fv]['mean']*100, yerr=yerr_fv*100,
                                    capsize=4, color=color_green, marker='.')
                        line_ref_mc = ax_mc.errorbar(axis_values, df2plot[proba_true]['mean']*100, yerr=yerr_ref_mc*100,
                                    capsize=4, color="black", marker='.', linestyle='dashed')

                        # Legend lines and labels in each plot (MC and FV)
                        legend_lines_mc  += [line_mc, line_ref_mc]
                        legend_lines_fv  += [line_fv, line_ref_mc]
                        legend_labels_mc += ["MC: {}={}".format(grp_legend, int(legend_value)), "True Pr(K)"]
                        legend_labels_fv += ["FV: {}={}".format(grp_legend, int(legend_value)), "True Pr(K)"]

                        # Add violinplots for the LARGEST legend value
                        # (which is expected to give the best results, as mentioned above)
                        if idx_legend == len(legend_values) - 1:
                            ind = (df[grp_K] == K) & (df[grp_legend] == legend_value)
                            plotting.violinplot(ax_mc,  [df[ind & (df[grp_axis]==x)][y_mc]*100 for x in axis_values],
                                                        positions=axis_values, showmeans=True, showmedians=False, linewidth=4,
                                                        color_body="red", color_lines="red", color_means="red")            
                            plotting.violinplot(ax_fv,  [df[ind & (df[grp_axis]==x)][y_fv]*100 for x in axis_values],
                                                        positions=axis_values, showmeans=True, showmedians=False, linewidth=4,
                                                        color_body="green", color_lines="green", color_means="green")            
                            for ax in (ax_mc, ax_fv):
                                ax.hlines([df[ind & (df[grp_axis]==x)][proba_true]*100 for x in axis_values], ax.get_xlim()[0], ax.get_xlim()[1], color='black', linestyles='dashed')

                        # Add labels     
                        for ax in (ax_mc, ax_fv):
                            ax.set_xlabel(grp_axis)
                            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                        ax_mc.set_ylabel('K = {:.0f}'.format(K), fontsize=12)
                    except:
                        print("Legend value SKIPPED because it does not happen for this K value!")
                        pass

                # Define Y-axis limits: the Y-axis upper limit is set to the maximum observed in the original containing the results (non-aggregated for plotting)
                ymax = np.nanmax(np.r_[df[ind_K][y_mc], df[ind_K][y_fv]])*100
                for ax in (ax_mc, ax_fv):
                    ax.set_ylim((0, ymax*1.1))

                # Add the legend
                ax_mc.legend(legend_lines_mc,
                             legend_labels_mc,
                             ncol=2,
                             fontsize='x-small')
                ax_fv.legend(legend_lines_fv,
                             legend_labels_fv,
                             ncol=2,
                             fontsize='x-small')

                # Add the title
                plt.suptitle("Estimated blocking probability P(K) using a particle system" \
                             " (Mean +/-{}SE + Boxplots of largest legend value, replications={})" \
                             "\nMonte Carlo (red) vs. Fleming Viot (green)" \
                             .format(se_mult, replications))

        # Define the analysis data frame and the name of the variables involved in the plots
        df = results_convergence
        # Grouping variables
        grp_buffer = 'buffer_size_activation'
        grp_part = 'N'
        grp_time = 'nmeantimes'
        grp_K = 'K'
        # The grouping variables should be EXACTLY THREE, which define:
        # - the page (e.g. K)
        # - the axis (e.g. nparticles)
        # - the legend (e.g. buffer size)
        groupvars = [grp_K, grp_buffer, grp_part]
        # Analysis variables
        y_mc = 'Pr(MC)'
        y_fv = 'Pr(FV)'      # Approximation 1
        y_fv2 = 'Pr(FV2)'     # Approximation 2
        proba_true = 'Pr(K)'  # Reference value for comparison of estimated values
        analvars = [y_mc, y_fv, y_fv2, proba_true]
        replications = int(np.max(df['rep']))

        # Analysis by group (mean, std, min, max)
        df_agg = aggregation_bygroups(df, groupvars, analvars)

        # Plot convergence analysis for ALL parameter values
        # first for grp_part, then for grp_time
        grp_axis = grp_part
        grp_legend = grp_buffer
        print("1:\n{}".format(df_agg))
        plot(df, df_agg,
             grp_K, grp_axis, grp_legend,
             replications)

        grp_axis = grp_buffer
        grp_legend = grp_part
        #print("2:\n{}".format(df_agg))
        #plot(df, df_agg,
        #     grp_K, grp_axis, grp_legend,
        #     replications)

        return df_agg

    @classmethod
    def plot_aggregated_convergence_results_maxvalues(cls, results_convergence):

        def plot(df2plot, grp_K, grp_axis, legend, se_mult=2):
            K_values = df2plot.index.levels[0]

            legend_label_mc = "MC: {}".format(legend)
            legend_label_fv = "FV: {}".format(legend)
            axes = plt.figure(figsize=(9,4*len(K_values))).subplots(len(K_values),1)
            for idx, K in enumerate(K_values):
                if len(K_values) == 1:
                    ax = axes
                else:
                    ax = axes[idx]
                ax.errorbar(df2plot.loc[K].index, df2plot.loc[K][y_mc]['mean'], yerr=se_mult*df2plot.loc[K][y_mc]['SE'],
                            capsize=4, color='red', marker='.')
                ax.errorbar(df2plot.loc[K].index, df2plot.loc[K][y_fv]['mean'], yerr=se_mult*df2plot.loc[K][y_fv]['SE'],
                            capsize=4, color='green', marker='.')
                ax.errorbar(df2plot.loc[K].index, df2plot.loc[K][proba_true]['mean'], yerr=se_mult*df2plot.loc[K][proba_true]['SE'],
                            capsize=4, color='black', marker='.', linestyle='dashed')
                #ax.axhline(y=P_true, color='black', linestyle='dashed')
                ax.set_xlabel(grp_axis)
                ax.set_ylabel('K = {:.0f}'.format(K))
                ymax = np.max(np.r_[df2plot.loc[K][y_mc]['mean'] + se_mult*df2plot.loc[K][y_mc]['SE'],
                                    df2plot.loc[K][y_fv]['mean'] + se_mult*df2plot.loc[K][y_fv]['SE'],
                                    df2plot.loc[K][proba_true]['mean'] + se_mult*df2plot.loc[K][proba_true]['SE']])
                ax.set_ylim((0, ymax*1.1))
                ax.legend([legend_label_mc,
                           legend_label_fv,
                           "Blocking rate values"])
                plt.suptitle("P(K) vs. {} in the Fleming Viot system (+/-{}SE, replications={})" \
                             .format(grp_axis.upper(), se_mult, replications))

        # Define the analysis data frame and the name of the variables involved in the plots
        df = results_convergence
        grp_part = 'N'
        grp_time = 'nmeantimes'
        grp_K = 'K'
        y_mc = 'Pr(MC)'
        y_fv = 'Pr(FV)'  # Approximation 1
        proba_true = 'Pr(K)'  # Reference values for comparing estimated values against: observed blocking time rate (calculated from the MC simulation)
        analvars = [y_mc, y_fv, proba_true]
        replications = int(np.max(df['rep']))

        # Filter each analysis by group by the largest value of the other group variable
        # (because the largest value of the parameter is supposed to be best in the sense that
        # more particles are used in the simulation or the simulation is run for longer)
        all_nmeantimes = np.unique(df[grp_time])
        nmeantimes_max = int(all_nmeantimes[-1])
        legend = "{} = {}".format(grp_time, nmeantimes_max)
        df4part = df[ df[grp_time] == nmeantimes_max ]
        df_agg_byparam = aggregation_bygroups(df4part, [grp_K, grp_part], analvars)
        plot(df_agg_byparam, grp_K, grp_part, legend, se_mult=2)

        all_nparticles = np.unique(df[grp_part])
        nparticles_max = int(all_nparticles[-1])
        legend = "{} = {}".format(grp_part, nparticles_max)
        df4iter = df[ df[grp_part] == nparticles_max ]
        df_byiter = aggregation_bygroups(df4iter, [grp_K, grp_time], analvars)
        plot(df_byiter, grp_K, grp_time, legend, se_mult=2)

#------------------- Functions --------------------
def deprecated_test_mc_implementation(nservers, K, paramsfile, nmeantimes=None, repmax=None, figfile=None):
    "2021/05/14: Run the MC simulation with simulation parameters read from a CSV file"

    #--- Test one server
    rate_birth = 0.7
    if nservers == 1:
        job_class_rates = [rate_birth]
        rate_death = [1]
        policy_assign = PolJobAssignmentProbabilistic([[1]])
        rewards_accept_by_job_class = None  # [1]
    elif nservers == 3:
        job_class_rates = [0.8, 0.7]
        rate_death = [1, 1, 1]
        policy_assign = PolJobAssignmentProbabilistic([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
        rewards_accept_by_job_class = None  # [1, 1]
    else:
        raise ValueError("Given Number of servers ({}) is invalid. Valid values are: 1, 3".format(nservers))

    # Create the queue, the queue environment, and the agent acting on it
    queue = queues.QueueMM(rate_birth, rate_death, nservers, K)
    env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates=job_class_rates,
                                                   reward_func=rewardOnJobRejection_ExponentialCost,
                                                   rewards_accept_by_job_class=rewards_accept_by_job_class)
    # Define the agent acting on the queue environment
    policies = dict({PolicyTypes.ACCEPT: None, PolicyTypes.ASSIGN: policy_assign})
    learners = None
    agent = AgeQueue(queue, policies, learners)

    rhos = get_server_loads(queue.getBirthRates(), queue.getDeathRates())

    # Info parameters 
    dict_params_info = {'plot': True, 'log': False}

    # Read data with parameter settings for the run
    df_parameters = pd.read_csv(paramsfile)
    df_parameters.rename(index=str, columns={'Unnamed: 0': 'setcase'}, inplace=True)
    print("System: # servers={}".format(nservers))
    print("Simulation parameters read from {} for Monte-Carlo simulation:\n{}".format(paramsfile, df_parameters))
    df_results = pd.DataFrame(columns=['K',
                                       'BSA',
                                       'N',
                                       'replication',
                                       'Pr(MC)',
                                       'EMC(T)',
                                       'Time(MC)',
                                       '#Events(MC)',
                                       '#Cycles(MC)',
                                       'Pr(K)',
                                       'seed',
                                       'exec_time'])
    ncases = df_parameters.shape[0]
    max_fv = 0  # Maximum P(K) estimation by FV read from the parameters file in order to properly scale the MC plot
    time_start_all = timer()
    for case in range(ncases):
        paramK = df_parameters['K'][case]
        # Only run the simulation for the value of K given in the input parameter of the function
        if K == paramK:
            max_fv = max( max_fv, df_parameters['Pr(FV)'][case] )
            setcase = df_parameters['setcase'][case]
            buffer_size_activation = df_parameters['BSA'][case]
            nparticles = df_parameters['N'][case]
            nevents = df_parameters['#Events(FV)'][case]
            rep = df_parameters['replication'][case]
            seed = df_parameters['seed'][case]
            proba_blocking_true = df_parameters['Pr(K)'][case]
            if rep == 1:
                print("\n*** Running replications for set case #{} on T={}x, #events={}: K={}, BSA={}, N={}, seed={}".format(setcase, nmeantimes, nevents, K, buffer_size_activation, nparticles, seed))
            print("\n\tReplication {}...".format(rep), end=" ")

            dict_params_simul = {
                'nparticles': nparticles,
                'nmeantimes': nmeantimes,
                'buffer_size_activation': buffer_size_activation,
                'seed': seed,
                    }

            if repmax is None or rep <= repmax:
                time_start = timer()
    
                print("--> Running Monte-Carlo simulation...")
                dict_params_simul['maxevents'] = nevents
                proba_blocking_mc, \
                    expected_return_time_mc, \
                        n_return_observations, \
                            est_mc, dict_stats_mc = estimators.estimate_blocking_mc(env_queue, agent, dict_params_simul, dict_params_info=dict_params_info)

                time_end = timer()
                exec_time = time_end - time_start
                print("execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))
                print("\tP(K) by MC: {:.6f}% (simulation time = {:.1f} out of max={}, #events {} out of {})" \
                      .format(proba_blocking_mc*100, est_mc.get_simulation_time(), est_mc.getMaxSimulationTime(), est_mc.getNumberOfEvents(), est_mc.getMaxNumberOfEvents()))
                print("\tTrue P(K): {:.6f}%".format(proba_blocking_true*100))
    
                # Store the results
                df_append = pd.DataFrame([[K,
                                           buffer_size_activation,
                                           nparticles,
                                           rep,
                                           proba_blocking_mc,
                                           expected_return_time_mc,
                                           dict_stats_mc['time'],
                                           dict_stats_mc['nevents'],
                                           n_return_observations,
                                           proba_blocking_true,
                                           dict_params_simul['seed'],
                                           exec_time]],
                                         columns=df_results.columns, index=[case])
                df_results = df_results.append(df_append)

    time_end_all = timer()

    print("Total execution time: {:.1f} min".format((time_end_all - time_start_all) / 60))
    title = "Monte-Carlo simulation results for #servers={}, rhos={}, K={}:".format(nservers, rhos, K)
    print(title)
    print(df_results)
    showtitle = False
    
    df_results_agg_by_N = aggregation_bygroups(df_results, ['N'], ['#Events(MC)', 'Pr(MC)'])
    print("Aggregated results by N:")
    print(df_results_agg_by_N)

    # Add back the average of # events to the full data frame      
    df_results = pd.merge(df_results, df_results_agg_by_N['#Events(MC)']['mean'],
                          left_on='N', right_index=True, suffixes=["", "_mean"])
    # Convert average to integer
    df_results = df_results.astype({'#Events(MC)_mean': np.int})

    plt.figure()
    legend_lines_mc = []
    legend_lines_ref = []
    ax = plt.gca()
    plt.plot(df_results['N'], df_results['Pr(MC)']*100, 'r.', markersize=2)
    line_mc = plt.errorbar(list(df_results_agg_by_N.index), df_results_agg_by_N['Pr(MC)']['mean']*100, yerr=2*df_results_agg_by_N['Pr(MC)']['SE']*100, capsize=4, color='red', marker='x')
    legend_lines_mc += [line_mc]
    line_ref = ax.hlines(df_results.iloc[0]['Pr(K)']*100, df_results.iloc[0]['N'], df_results.iloc[-1]['N'], color='gray', linestyles='dashed')
    legend_lines_ref += [line_ref]
    plt.title(title, fontsize=10)
    #ax.set_xlim([0, ax.get_xlim()[1]])
    ymax = max( ax.get_ylim()[1], max_fv )
    ax.set_ylim([0, ymax])
    ax.set_xlabel("N (number of particles)")
    ax.set_ylabel("Blocking probability (%)")
    ax.legend(legend_lines_mc + legend_lines_ref, ['MC +/- 2SE', 'True'], fontsize='x-small')

    # Violin plots
    (ax_mc) = plt.figure(figsize=(8,4)).subplots(1,1)
    nevents_values = np.unique(df_results['#Events(MC)_mean'])
    violin_widths_mc = (nevents_values[-1] - nevents_values[0]) / 10
    plotting.violinplot(ax_mc,  [df_results[ df_results['#Events(MC)_mean']==x ]['Pr(MC)']*100 for x in nevents_values],
                                positions=nevents_values, showmeans=True, showmedians=False, linewidth=2, widths=violin_widths_mc,
                                color_body="red", color_lines="red", color_means="red")
    # Add the observed points
    ax_mc.plot(df_results['#Events(MC)'], df_results['Pr(MC)']*100, 'r.', markersize=2)
    
    ax_mc.hlines(df_results.iloc[0]['Pr(K)']*100, df_results.iloc[0]['#Events(MC)_mean'], df_results.iloc[-1]['#Events(MC)_mean'], color='gray', linestyles='dashed')
    if showtitle:
        plt.suptitle(title, fontsize=10)
    # Set a common vertical axis
    ymax = max( ax_mc.get_ylim()[1], max_fv )
    ax_mc.set_ylim([0, ymax])
    ax_mc.set_xlabel("Average number of events")
    ax_mc.set_ylabel("Blocking probability (%)")

    if figfile is not None:
        plt.gcf().subplots_adjust(left=0.15)
            ## To avoid cut off of vertical axis label!!
            ## Ref: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot 
        plt.savefig(figfile)
        
    return df_results, df_results_agg_by_N, est_mc 

def deprecated_compute_errors(df_results,
                   y_mc="Pr(MC)", y_fv="Pr(FV)", y_true="Pr(K)"):
    """
    Computes estimation errors on a set of results w.r.t. to a true value
   
    Arguments:
    df_results: pandas DataFrame
        DataFrame with the results of the simulation which should contain
        at least the following columns:
        - `y_mc`: variable containing the estimated of `y_true` by Monte-Carlo
        - `y_fv`: variable containing the estimate of `y_true` by Fleming-Viot
        - `y_true`: variable containing the true value estimated by MC and FV (used as a reference line)

    Return: pandas DataFrame
        Input DataFrame with 4 new columns:
        - "error_mc": absolute error of the MC estimation
        - "error_fv": absolute error of the FV estimation
        - "error_rel_mc": relative error of the MC estimation
        - "error_rel_fv": relative error of the FV estimation
    """
    df = copy.deepcopy(df_results)
    
    df["error_mc"] = df[y_mc] - df[y_true]
    df["error_fv"] = df[y_fv] - df[y_true]
    df["error_rel_mc"] = (df[y_mc] - df[y_true]) / df[y_true]
    df["error_rel_fv"] = (df[y_fv] - df[y_true]) / df[y_true]
    #print(df[[y_mc, y_fv, "error_rel_mc", "error_rel_fv"]])

    return df

def deprecated_plot_estimates1(df_results, x, y, subset=None,
                   grp_K="K", y_true="Pr(K)", rep="rep",
                   violin_width_factor=0.2, color="green", xlabel=None,
                   markersize=7, fontsize=13, showtitle=False, figfile=None):
    """
    Plots the distribution of estimates as violin plots for the MC and the FV methods
    against a variable of interest, x.

    Arguments:
    df_results: pandas DataFrame
        DataFrame with the results of the simulation which should contain
        at least the following columns:
        - `grp_K`: grouping variable by which a separate plot is made
        - `x`: variable of interest to plot on the X-axis
        - `y`: variable containing the estimate of `y_true` to plot
        - `y_true`: variable containing the true value estimated by MC and FV (used as a reference line)
        - `rep`: index identifying the simulation replication

    subset: expression
        An expression to filter the rows to plot.
        Ex: df_results['nparticles']==400

    Return: pandas DataFrame
        DataFrame containing the observations in the input data frame used for plotting.
    """
    #--- Parse input parameters
    # Rows to plot
    if subset is not None:
        df2plot = df_results[subset]
    else:
        df2plot = df_results

    if xlabel is None:
        xlabel = x
    #--- Parse input parameters

    # Get the values of the group variable on which we generate a SEPARATE plot
    # by computing its frequency distribution
    K_values = df2plot[grp_K].value_counts(sort=True).index
    replications = int(np.max(df2plot[rep]))    # For informational purposes
    for K in K_values:
        ind = (df2plot[grp_K] == K)
        # Compute the frequency of occurrence of the x values
        # so that we get the values from the index of the frequency table
        x_values = df2plot[ind][x].value_counts(sort=True).index
        # Widths of violins as 10% of the x value range (so that violins are visible --as opposed to just seeing a vertical line)
        proba_blocking_K = df2plot[ind][y_true].iloc[0]
        #print("K={:.0f}: {}".format(K, x_values))
        plt.figure()
        ax = plt.gca()
        #ax = plt.gca()
        #plt.plot([x_values[0], x_values[-1]], [0.0, 0.0], color="gray")
        # We create a violin plot for each set of not NaN errors (MC & FV), one set per x value
        y_values = [df2plot[ind & (df2plot[x]==x_value)][y].dropna()*100 for x_value in x_values]
        widths = violin_width_factor * (x_values[-1] - x_values[0])
        plotting.violinplot(ax,  y_values,
                            positions=x_values, showmeans=True, showmedians=False, linewidth=2, widths=widths,
                            color_body=color, color_lines=color, color_means=color)

        # Plot points
        npoints = df2plot[ind].shape[0]
        jitter = 1 + 0.1*(np.random.random(npoints) - 0.5)
        ax.plot(df2plot[ind][x]*jitter, df2plot[ind][y]*100, 'k.', markersize=markersize)

        # Reference line with the true value
        ax.axhline(proba_blocking_K*100, color="gray", linestyle="dashed")

        ymax = 1.1*np.max(np.r_[df2plot[ind][y]*100, proba_blocking_K*100])
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.05*ymax, ymax])
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel("Blocking probability (%)", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
        plt.gcf().subplots_adjust(left=0.3)
        if showtitle:
            plt.suptitle("Distribution of blocking probability estimates of Pr(K={:.0f}) = {:.6f}% on {} replications" \
                         .format(K, proba_blocking_K*100, replications) +
                         "\nMonte Carlo (red) vs. Fleming Viot (green)")

    return df2plot

def deprecated_plot_estimates(df_results, x, subset=None,
                   grp_K="K", y_mc="Pr(MC)", y_fv="Pr(FV)", y_true="Pr(K)", rep="rep",
                   xlabel=None, markersize=7, fontsize=13, showtitle=False, figfile=None):
    """
    Plots the distribution of estimates as violin plots for the MC and the FV methods
    against a variable of interest, x.

    Arguments:
    df_results: pandas DataFrame
        DataFrame with the results of the simulation which should contain
        at least the following columns:
        - `grp_K`: grouping variable by which a separate plot is made
        - `x`: variable of interest to plot on the X-axis
        - `y_mc`: variable containing the estimated of `y_true` by Monte-Carlo
        - `y_fv`: variable containing the estimate of `y_true` by Fleming-Viot
        - `y_true`: variable containing the true value estimated by MC and FV (used as a reference line)
        - `rep`: index identifying the simulation replication

    subset: expression
        An expression to filter the rows to plot.
        Ex: df_results['nparticles']==400

    Return: pandas DataFrame
        DataFrame containing the observations in the input data frame used for plotting.
    """
    #--- Parse input parameters
    # Rows to plot
    if subset is not None:
        df2plot = df_results[subset]
    else:
        df2plot = df_results

    if xlabel is None:
        xlabel = x
    #--- Parse input parameters

    # Get the values of the group variable on which we generate a SEPARATE plot
    # by computing its frequency distribution
    K_values = df2plot[grp_K].value_counts(sort=False).index
    replications = int(np.max(df2plot[rep]))    # For informational purposes
    for K in K_values:
        ind = (df2plot[grp_K] == K)
        # Compute the frequency of occurrence of the x values
        # so that we get the values from the index of the frequency table
        x_values = df2plot[ind][x].value_counts(sort=True).index
        # Widths of violins as 10% of the x value range (so that violins are visible --as opposed to just seeing a vertical line)
        widths = (x_values[-1] - x_values[0]) / 10
        proba_blocking_K = df2plot[ind][y_true].iloc[0]
        #print("K={:.0f}: {}".format(K, x_values))
        (fig, axes) = plt.subplots(1, 2, figsize=(12,4))
        ax1 = axes[0]
        ax2 = axes[1]
        #ax = plt.gca()
        #plt.plot([x_values[0], x_values[-1]], [0.0, 0.0], color="gray")
        # We create a violin plot for each set of not NaN errors (MC & FV), one set per x value
        y1 = [df2plot[ind & (df2plot[x]==x_value)][y_mc].dropna()*100 for x_value in x_values]
        y2 = [df2plot[ind & (df2plot[x]==x_value)][y_fv].dropna()*100 for x_value in x_values]
        plotting.violinplot(ax1,  y1,
                            positions=x_values, showmeans=True, showmedians=False, linewidth=2, widths=widths,
                            color_body="red", color_lines="red", color_means="red")
        plotting.violinplot(ax2,  y2,
                            positions=x_values, showmeans=True, showmedians=False, linewidth=2, widths=widths,
                            color_body="green", color_lines="green", color_means="green")

        # Plot points
        npoints = df2plot.shape[0]
        jitter = 1 + 0.1*(np.random.random(npoints) - 0.5)
        ax1.plot(df2plot[x]*jitter, df2plot[y_mc]*100, 'k.', markersize=markersize)
        ax2.plot(df2plot[x]*jitter, df2plot[y_fv]*100, 'k.', markersize=markersize)

        ymax = np.max([ np.max(np.r_[y1, y2]), proba_blocking_K ])
        for ax in axes:
            ax.set_xlim([0, 1])
            ax.set_ylim([0, ymax])
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel("Estimated blocking probability (%)", fontsize=fontsize)
            ax.axhline(proba_blocking_K*100, color="gray", linestyle="dashed")
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if showtitle:
            plt.suptitle("Distribution of blocking probability estimates of Pr(K={:.0f}) = {:.6f}% on {} replications" \
                         .format(K, proba_blocking_K*100, replications) +
                         "\nMonte Carlo (red) vs. Fleming Viot (green)")

    return df2plot

def deprecated_plot_errors(df_results, x, subset=None, widths=0.1,
                grp_K="K", error_mc="error_rel_mc", error_fv="error_rel_fv", rep="rep",
                xlabel=None, showtitle=True):
    """
    Plots the distribution of estimation errors as violin plots for the MC and the FV methods
    against a variable of interest, x.

    Arguments:
    df_results: pandas DataFrame
        DataFrame with the results of the simulation which should contain
        at least the following columns:
        - `grp_K`: grouping variable by which a separate plot is made
        - `x`: variable of interest to plot on the X-axis
        - `error_mc`: variable to plot containing the error of Monte-Carlo method
        - `error_fv`: variable to plot containing the error of the Fleming-Viot method
        - `rep`: index identifying the simulation replication

    subset: expression
        An expression to filter the rows to plot.
        Ex: df_results['nparticles']==400

    widths: float
        `widths` parameter of the pyplot.violinplot() function that defines the width of each violin.
        This should be set to something that makes the violin plots visible w.r.t. to the scale of the x axis.

    Return: pandas DataFrame
        DataFrame containing the observations in the input data frame used for plotting.
    """
    #--- Parse input parameters
    # Rows to plot
    if subset is not None:
        df2plot = df_results[subset]
    else:
        df2plot = df_results

    if xlabel is None:
        xlabel = x
    #--- Parse input parameters

    # Get the values of the group variable on which we generate a SEPARATE plot
    # by computing its frequency distribution
    K_values = df2plot[grp_K].value_counts(sort=False).index
    replications = int(np.max(df2plot[rep]))    # For informational purposes

    # Compute the max absolute values of all Y's to plot, so we can visually compare the plots
    yabsmax = np.nanmax(np.r_[ np.abs(df2plot[error_mc]), np.abs(df2plot[error_fv])] )*100
    # The above max value may be too large, so we set the vertical scale to -100% -- +100%
    yabsmax = 1.0*100
    for K in K_values:
        ind = (df2plot[grp_K] == K)
        # Compute the frequency of occurrence of the x values
        # so that we get the values from the index of the frequency table
        x_values = df2plot[ind][x].value_counts(sort=False).index
        #print("K={:.0f}: {}".format(K, x_values))
        (fig, axes) = plt.subplots(1, 2, figsize=(12,4))
        ax1 = axes[0]
        ax2 = axes[1]
        #ax = plt.gca()
        #plt.plot([x_values[0], x_values[-1]], [0.0, 0.0], color="gray")
        # We create a violin plot for each set of not NaN errors (MC & FV), one set per x value
        plotting.violinplot(ax1,  [df2plot[ind & (df2plot[x]==x_value)][error_mc].dropna()*100 for x_value in x_values],
                                    positions=x_values, showmeans=True, showmedians=False, linewidth=2, widths=widths,
                                    color_body="red", color_lines="red", color_means="red") 
        plotting.violinplot(ax2,  [df2plot[ind & (df2plot[x]==x_value)][error_fv].dropna()*100 for x_value in x_values],
                                    positions=x_values, showmeans=True, showmedians=False, linewidth=2, widths=widths,
                                    color_body="green", color_lines="green", color_means="green")
        for ax in axes:
            ax.set_ylim((-yabsmax*1.1, yabsmax*1.1))
            ax.set_xlabel(xlabel)
            ax.set_ylabel("K={:.0f} -- Relative Error (%)".format(K))
            ax.axhline(0.0, color="gray")
        if showtitle:
            plt.suptitle("Error distribution of blocking probability estimation Pr(K={:.0f}) on {} replications" \
                         .format(K, replications) +
                         "\nMonte Carlo (red) vs. Fleming Viot (green)")

    return df2plot

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

    #--- Parse input parameters
    # What to plot
    if plot_mc:
        if x2 is None:
            x2 = x
        xvars = [x, x2]
        yvars = [prob_fv, prob_mc]
        figsize = (8,4)
        subplots = (1,2)
        nsubplots = 2
        colors = ["green", "red"]
    else:
        xvars = [x]
        yvars = [prob_fv]
        figsize = (4,4)
        subplots = (1,1)
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
                smooth_params[k] = [ smooth_params[k] ] * nsubplots
    else:
        plot_func_summarize = plotting.plot_errorbars
        points_properties = {'color': "black", 'marker': ".",
                             'color_center': "black", 'marker_center': "."}
        smooth_params = {'bias': [None]*nsubplots, 'variability': [None]*nsubplots, 'mse': [None]*nsubplots}

    # Rows to plot
    if subset is not None:
        df_results = df_results[subset]
    #--- Parse input parameters

    # 1) Average P(K) + error bars
    axes_error = plotting.plot( plotting.plot_errorbars,
                                df_results, xvars, yvars,
                                yref=prob_true, yref_legend="True value",
                                figsize=figsize, subplots=subplots,
                                dict_options={'axis': axis_properties,
                                              'multipliers': {'x': 1, 'y': 100, 'error': 2},
                                              'labels': {'x': [xlabel, xlabel2], 'y': "Blocking probability (%)"},
                                              'properties': {'color': "black", 'color_center': colors}})
  
    # 2) Violin plots
    axes_violin = plotting.plot( plotting.plot_violins,
                                 df_results, xvars, yvars,
                                 yref=prob_true, yref_legend="True value",
                                 figsize=figsize, subplots=subplots,
                                 dict_options={'axis': axis_properties,
                                               'multipliers': {'x': 1, 'y': 100},
                                               'labels': {'x': [xlabel, xlabel2], 'y': "Blocking probability (%)"},
                                               'properties': {'color': colors, 'color_center': colors}})

    #-- Compute variability and bias
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
    for idx, (x, y) in enumerate( zip(xvars, yvars) ):
        # Variability and bias
        if y2 is not None:
            yvars2agg = [y, y2]
        else:
            yvars2agg = [y]
        summary_stats = aggregation_bygroups(df_results, [x], yvars2agg, stats=["count", "mean", "median", "mad", "std", "var"])
        
        # Store the results in the data frame for plotting
        nvar = "n({})".format(y)
        weightvar = "w({})".format(y)
        biasvar = "b({})".format(y)
        bias2var = "b2({})".format(y)
        variancevar = "Var({})".format(y)
        msevar = "MSE({})".format(y)
        madvar = "MAD({})".format(y)
        cvbiasvar = "CVBIAS({})".format(y)
        cvstdvar =  "CVSTD({})".format(y)
        cvmadvar = "CVMAD({})".format(y)
        cvrmsevar = "CVRMSE({})".format(y)
        if y2 is not None:
            y2meanvar = "mean({})".format(y2)
        df2plot[x] = summary_stats.index
        df2plot.set_index(summary_stats.index, inplace=True)
        df2plot[nvar] = summary_stats[y]["count"]
        df2plot[weightvar] = df2plot[nvar] / np.sum(df2plot[nvar])
        df2plot[biasvar] = summary_stats[y]["mean"] - df_results.iloc[0][prob_true]
        df2plot[bias2var] = df2plot[biasvar]**2
        df2plot[variancevar] = summary_stats[y]["var"]
        df2plot[msevar] = df2plot[biasvar]**2 + df2plot[variancevar]
        df2plot[madvar] = summary_stats[y]["mad"]
        df2plot[cvbiasvar] = np.abs( df2plot[biasvar] ) / df_results.iloc[0][prob_true] * 100
        df2plot[cvstdvar] = np.sqrt(df2plot[variancevar]) / df_results.iloc[0][prob_true] * 100
        df2plot[cvmadvar] = df2plot[madvar] / df_results.iloc[0][prob_true] * 100
        df2plot[cvrmsevar] = np.sqrt(df2plot[msevar]) / df_results.iloc[0][prob_true] * 100
        if y2 is not None:
            df2plot[y2meanvar] = summary_stats[y2]["mean"]

        var2plot_bias = cvbiasvar  # bias2var 
        var2plot_variability = cvstdvar # cvmadvar # variancevar 
        var2plot_mse = cvrmsevar

        # Weighted squared sum of Y values --> used when choosing a good smoothing parameter
        # The calculation is based on the documentation for parameter 's' in help(scipy.interpolate.slprep)
        if splines and use_weights_splines:
            # Normalized weights
            weights = df2plot[weightvar]
            weightvars += [ weightvar ]
        else:
            weights = 1
            weightvars += [ None ]
        if splines and smooth_params['bias'][idx] is None:
            smooth_params['bias'][idx] = np.mean( np.abs( df2plot[var2plot_bias] ) ) * np.sum( (weights * df2plot[var2plot_bias]) **2)
        if splines and smooth_params['variability'][idx] is None:
            smooth_params['variability'][idx] = np.mean( np.abs( df2plot[var2plot_variability] ) ) * np.sum( (weights * df2plot[var2plot_variability]) **2) #/ \
                                                 #np.sum( df2plot[nvar]**2 ) * np.var( df2plot[var2plot_variability] )
        if splines and smooth_params['mse'][idx] is None:
            smooth_params['mse'][idx] = np.mean( np.abs( df2plot[var2plot_mse] ) ) * np.sum( (weights * df2plot[var2plot_mse]) **2)
        if splines:
            print("Smoothing parameter for bias, variability, MSE for subplot {}:\n{}".format(idx+1, smooth_params))
        nvars += [ nvar ]
        biasvars += [ biasvar ]
        bias2vars += [ bias2var ]
        variancevars += [ variancevar ]
        msevars += [ msevar ]
        madvars += [ madvar ]
        cvbiasvars += [ cvbiasvar ]
        cvmadvars += [ cvmadvar ]
        vars2plot_bias += [ var2plot_bias ]
        vars2plot_variability += [ var2plot_variability ]
        vars2plot_mse += [ var2plot_mse ]
    
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
        for (ax, x, y, n, var2plot, w, s, color) in zip(axes, xvars, yvars, nvars, vars2plot_variability, weightvars, smooth_params['variability'], colors):
            if splines:
                legend_obj, legend_txt = plotting.plot_splines(ax, df2plot, x, var2plot, w=w, s=s,
                                                               dict_options={'properties': {'color': "black", 'color_line': color}})
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


if __name__ == "__main__":
    # Default execution arguments when no arguments are given
    # Example of execution from the command line:
    # python test_QB.py 1 N 5 10 0.5 8 1
    print("User arguments: {}".format(sys.argv))
    if len(sys.argv) == 1:    # Only the execution file name is contained in sys.argv
        sys.argv += [1]       # Number of servers in the system to simulate
        sys.argv += ["N"]     # Type of analysis: either "N" for the impact of number of particles or "J" for the impact of buffer size"
        sys.argv += [40]      # K: capacity of the system
        sys.argv += [200]     # N: number of particles in the FV system.
        sys.argv += [0.5]     # J factor: factor such that J = factor*K.
        sys.argv += [2]       # Number of replications
        sys.argv += [1]       # Test number to run: only one is accepted
    if len(sys.argv) == 8:    # Only the 6 required arguments are given by the user (recall that the first argument is the program name)
        sys.argv += ['None']  # T: number of arrival events to consider in the estimation of Pr(T>t) and E(T) in the FV approach. When 'None' it is chosen as 50*N.
        sys.argv += [2]       # Number of methods to run: 1 (only FV), 2 (FV & MC)
        sys.argv += ["nosave"]  # Either "nosave" or anything else for saving the results and log
    print("Parsed user arguments: {}".format(sys.argv))
    print("")

    #-- Parse user arguments
    nservers = int(sys.argv[1])
    analysis_type = sys.argv[2]
    K = int(sys.argv[3])
    N = int(sys.argv[4])
    J = int(np.round(sys.argv[5]*K))
    replications = int(sys.argv[6])
    tests2run = [int(v) for v in [sys.argv[7]]] # NOTE: It's important to enclose sys.argv[5] in brackets because o.w., from the command line, a number with more than one digit is interpreted as a multi-element list!! (e.g. 10 is interpreted as a list with elements [1, 0])
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
        dt_start, stdout_sys, fh_log, logfile, resultsfile, resultsfile_agg, proba_functions_file, figfile = createLogFileHandleAndResultsFileNames(prefix=resultsfile_prefix)
    else:
        fh_log = None; resultsfile = None; resultsfile_agg = None; proba_functions_file = None; figfile = None

    if analysis_type == "N":
        # DM-2020/08/24: Instead of using unittest.main(), use what comes below to test the FV system
        # because there is a very weird error generated by the fact that queue.size inside
        # the EstimatorQueueBlockingFlemingViot class is interpreted of being of class
        # unittest.runner.TextTestResult (or similar)!! instead of simply returning the size
        # attribute of the `queue` object!!!
        #suite = unittest.TestSuite()
        #suite.addTest(Test_QB_Particles("test_fv_implementation"))
        #runner = unittest.TextTestRunner()
        #runner.run(suite)
        
        #******************* ACTUAL EXECUTION ***************
        #-- Single-server
        results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(estimation_process=Process.Simulators,
                                                                                        nservers=nservers, K=40, buffer_size_activation=J,
                                                                                        nparticles=[N], #[800, 1600, 3200], #[10, 20, 40], #[24, 66, 179],
                                                                                        nmeantimes=T, #50, #[170, 463, 1259],
                                                                                        replications=replications, run_mc=run_mc,
                                                                                        seed=1313)

        #results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(nservers=1, K=20, buffer_size_activation=0.25)
        #results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(nservers=1, K=20, buffer_size_activation=0.5)
        #results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(nservers=1, K=20, buffer_size_activation=0.75)
        #results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(nservers=1, K=20, buffer_size_activation=0.9)
    
        #results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(nservers=1, K=40, buffer_size_activation=0.5)
    
        #-- Multi-server
        #results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(nservers=3, K=5, buffer_size_activation=0.5, burnin_cycles_absorption=3, run_mc=False)
        #results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(nservers=3, K=10, buffer_size_activation=0.5)
        #results, results_agg, est_fv, est_mc = \
        #    Test_QB_Particles.test_fv_implementation(nservers=3, K=20, buffer_size_activation=0.5, burnin_cycles_absorption=4,
        #                                             nparticles_min=800, nparticles_max=1600, nparticles_step_prop=1,
        #                                             nmeantimes=1400, replications=5,
        #                                             run_mc=run_mc, plotFlag=True)
        #results, results_agg, est_fv, est_mc = \
        #    Test_QB_Particles.test_fv_implementation(nservers=3, K=20, buffer_size_activation=0.2, burnin_cycles_absorption=1,
        #                                             nparticles_min=400, nparticles_max=1200, nparticles_step_prop=1,
        #                                             nmeantimes=500, replications=5,
        #                                             run_mc=run_mc, plotFlag=True)
        #results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(nservers=3, K=30, buffer_size_activation=0.5)
    
        #results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(nservers=3, K=40, buffer_size_activation=0.25)
        #results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(nservers=3, K=40, buffer_size_activation=0.3, burnin_cycles_absorption=1)
        #results, results_agg, est_fv, est_mc = \
        #    Test_QB_Particles.test_fv_implementation(nservers=3, K=40, buffer_size_activation=0.5, burnin_cycles_absorption=2,
        #                                             nparticles_min=400, nparticles_max=1600, nparticles_step_prop=1,
        #                                             nmeantimes=100000, replications=5,
        #                                             run_mc=run_mc, plotFlag=True)
        #results, results_agg, est_fv, est_mc = Test_QB_Particles.test_fv_implementation(nservers=3, K=40, buffer_size_activation=0.7)
        #******************* ACTUAL EXECUTION ***************
    
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
        test = Test_QB_Particles(nservers=nservers)
    
        # Info for plotting...
        x = "buffer_size_activation"; xlabel = "J as fraction of K"
        if 1 in tests2run:
            K_values = [5, 5]   #[10, 20, 30, 40]
            results, results_agg, proba_functions, est_fv, est_mc = test.analyze_estimates(
                                            replications=replications,
                                            K_values=K_values,
                                            nparticles_values=[20, 40], #[200, 400, 800, 1600],
                                            nmeantimes_values=[5000, 5000], #[50, 50, 50, 50],
                                            buffer_size_activation_values=[0.25, 0.5], #[1, 0.2, 0.4, 0.6, 0.8],
                                            burnin_cycles_absorption_values=[5, 3],
                                            seed=1313,
                                            run_mc=run_mc)
            save_dataframes([{'df': results, 'file': resultsfile},
                             {'df': results_agg, 'file': resultsfile_agg},
                             {'df': proba_functions, 'file': proba_functions_file}])
            #for K in K_values:
            #    axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K']==K, plot_mc=run_mc)
        if 2 in tests2run:
            K_values = [10, 20]
            results, results_agg, proba_functions, est_fv, est_mc = test.analyze_estimates(
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
                axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K']==K, plot_mc=run_mc)
        if 3 in tests2run:
            K_values = [30, 40]
            results, results_agg, proba_functions, est_fv, est_mc = test.analyze_estimates(
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
                axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K']==K, plot_mc=run_mc)
        if 4 in tests2run:
            K_values = [10]
            results, results_agg, proba_functions, est_fv, est_mc = test.analyze_estimates(
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
                axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K']==K, plot_mc=run_mc)
        if 5 in tests2run:
            K_values = [20]
            results, results_agg, proba_functions, est_fv, est_mc = test.analyze_estimates(
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
                axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K']==K, plot_mc=run_mc)
        if 6 in tests2run:
            K_values = [30]
            results, results_agg, proba_functions, est_fv, est_mc = test.analyze_estimates(
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
                axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K']==K, plot_mc=run_mc)
        if 7 in tests2run:
            K_values = [40]
            results, results_agg, proba_functions, est_fv, est_mc = test.analyze_estimates(
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
                axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K']==K, plot_mc=run_mc)
        if 8 in tests2run:
            # Same as 7 but for small J values (to see if the variance of the estimator increases first and then decreases)
            K_values = [40]
            results, results_agg, proba_functions, est_fv, est_mc = test.analyze_estimates(
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
                axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K']==K, plot_mc=run_mc)
        if 9 in tests2run:
            # Larger K value for multi-server, where MC is expected to fail
            K_values = [60]
            results, results_agg, proba_functions, est_fv, est_mc = test.analyze_estimates(
                                            replications=replications,
                                            K_values=K_values,
                                            nparticles_values=[6400],
                                            nmeantimes_values=[1000],#[8E7],
                                            buffer_size_activation_values=[1, 3, 5, 7, 12, 15, 21, 24],#[1, 3, 5, 7, 9, 12, 15, 18, 21, 24],
                                            burnin_cycles_absorption_values=[3, 3, 3, 3, 2, 2, 2, 2],#[3, 3, 3, 3, 3, 2, 2, 2, 2, 2]
                                            seed=1313,
                                            run_mc=run_mc)
            save_dataframes([{'df': results, 'file': resultsfile},
                             {'df': results_agg, 'file': resultsfile_agg},
                             {'df': proba_functions, 'file': proba_functions_file}])
            for K in K_values:
                axes = plot_results_fv_mc(results, x, xlabel=xlabel, subset=results['K']==K, plot_mc=run_mc)
        if 10 in tests2run:
            # Large K value for multi-server, with limited simulation time before 
            K_values = [40]
            results, results_agg, proba_functions, est_fv, est_mc = test.analyze_estimates(
                                            replications=replications,
                                            K_values=K_values,
                                            nparticles_values=[3200], #[1000],   #[5]
                                            nmeantimes_values=[1000], #[1000],   #[100]
                                            buffer_size_activation_values=[0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.35, 0.4, 0.5, 0.6],#[0.1, 0.3, 0.5, 0.7], #[0.1, 0.3, 0.4, 0.5, 0.6, 0.8],
                                            burnin_cycles_absorption_values=[0], #[3, 3, 2, 1],#[3, 3, 2, 2, 1, 1], #[3, 3, 3, 2, 2, 1],
                                            seed=1313,
                                            run_mc=run_mc,
                                            dict_params_info={'plot': False, 'log': False})
            save_dataframes([{'df': results, 'file': resultsfile},
                             {'df': results_agg, 'file': resultsfile_agg},
                             {'df': proba_functions, 'file': proba_functions_file}])
            #for K in K_values:
            #    axes = plot_results_fv_mc(results, "buffer_size_activation", xlabel="J as fraction of K",
            #                              subset=results['K']==K,
            #                              plot_mc=run_mc,
            #                              smooth_params={'bias': [1E2], 'variability': 1E3, 'mse': 1E-22})

        if fh_log is not None:
            closeLogFile(fh_log, stdout_sys, dt_start)
