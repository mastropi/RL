# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:16:02 2020

@author: Daniel Mastropietro
@description: Estimators are defined for problems of interest in Reinforcement Learning.
"""

import runpy
runpy.run_path('../../setup.py')

import copy
import numpy as np
import pandas as pd

from timeit import default_timer as timer
from datetime import timedelta 
import unittest
from unittest_data_provider import data_provider
import matplotlib
from matplotlib import pyplot as plt, cm, ticker as mtick

import Python.lib.queues as queues
import Python.lib.estimators as estimators
from Python.lib.queues import Event
from Python.lib.estimators import FinalizeType

#from importlib import reload
#reload(estimators)
#from Python.lib.estimators import EstimatorQueueBlockingFlemingViot

#import test_utils


class Test_QB_Particles(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = False
        self.rate_birth = 1.2
        self.rate_death = 1.4
        self.capacity = 5
        
        # One server
        #self.nservers = 1
        #self.job_rates = [1.2]
        #self.policy = [[1]]
        
        # Multiple servers
        self.nservers = 3
        self.job_rates = [2, 5]
        self.policy = [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]

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

    def test_simulation_of_events_on_a_system_with_one_particle(self):
        print("\nRunning test " + self.id())
        niter = 100
        est = estimators.EstimatorQueueBlockingFlemingViot(1, niter, self.queue,
                                                           reactivate=False)
        
        # The state udpate process is correctly done. Let aNET = "array with Next Event Times",
        # then the following assertions hold.
        est.reset()
        print("The test runs on a simulation with {} iterations" \
              "\n(no pre-specified seed, so that different test runs produce even more tests!)".format(niter))
        time_event_prev = 0.0
        for it in range(niter):
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
                                        niter=20,
                                        seed=1713,
                                        log=False): 
        est = estimators.EstimatorQueueBlockingFlemingViot(nparticles, niter, self.queue,
                                                           reactivate=reactivate,
                                                           finalize_type=finalize_type,
                                                           plotFlag=True,
                                                           seed=seed, log=log)        
        est.reset()
        for it in range(niter):
            est.generate_one_iteration(it+1)
            N1 = n_active_particles = est.get_number_active_particles()
            if np.mod(it, int(niter/10)) == 0:
                print("Iteration {} of {}... ({} active particles)".format(it+1, niter, N1))
                
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
        all_niters = [20, 50]


        reactivate = True
        finalize_type = FinalizeType.ABSORB_CENSORED
        print("\n***** START OF reactivate={}, finalize_type={}...".format(reactivate, finalize_type))
        for nparticles in all_nparticles:
            for niter in all_niters:
                print("\nRunning with reactivate={}, finalize_type={}...".format(reactivate, finalize_type))
                print("\tnparticles={}, niter={}...".format(nparticles, niter))
                self.run_test_compute_counts(reactivate=reactivate,
                                            finalize_type=finalize_type,
                                            nparticles=nparticles,
                                            niter=niter,
                                            seed=seed,
                                            log=log)

        reactivate = True
        finalize_type = FinalizeType.REMOVE_CENSORED
        print("\n***** START OF reactivate={}, finalize_type={}...".format(reactivate, finalize_type))
        for nparticles in all_nparticles:
            for niter in all_niters:
                print("\nRunning with reactivate={}, finalize_type={}...".format(reactivate, finalize_type))
                print("\tnparticles={}, niter={}...".format(nparticles, niter))
                self.run_test_compute_counts(reactivate=reactivate,
                                            finalize_type=finalize_type,
                                            nparticles=nparticles,
                                            niter=niter,
                                            seed=seed,
                                            log=log)

    def no_tests_on_n_particles(self,   reactivate=False,
                                     finalize_type=FinalizeType.ABSORB_CENSORED,
                                     nparticles=5,
                                     niter=20,
                                     seed=1713,
                                     log=False): 
        print("\nRunning test " + self.id())
        #nparticles = 30
        #niter = 200
        #reactivate = True
        #finalize_type = FinalizeType.ABSORB_CENSORED
        #finalize_type = FinalizeType.REMOVE_CENSORED
        #seed = 1713
        est = estimators.EstimatorQueueBlockingFlemingViot(nparticles, niter, self.queue,
                                                           reactivate=reactivate,
                                                           finalize_type=finalize_type,
                                                           plotFlag=True,
                                                           seed=seed, log=log)
        print("Simulation setup:")
        print(est.setup())
        
        est.reset()
        for it in range(niter):
            est.update_state(it+1)
            N1 = n_active_particles = est.get_number_active_particles()
            if np.mod(it, int(niter/10)) == 0:
                print("Iteration {} of {}... ({} active particles)".format(it+1, niter, N1))

            if self.log:
                print("------ END OF ITER {} of {} ------".format(it+1, niter))

            # The list storing the times when particles became active is sorted increasingly
            particles, activation_times = est.get_all_activation_times()
            if self.log:
                with np.printoptions(precision=3):
                    print("activation times: {}".format(np.array(activation_times)))
            assert sorted(activation_times) == activation_times, \
                    "The absolute times of activation are sorted: {}".format(np.array(activation_times))
            assert len(activation_times) >= N1, \
                    "The number of activation times ({}) is at least equal to the number of active particles (N1 = {})" \
                    .format(len(activation_times), N1)

            particles, absorption_times = est.get_all_absorption_times()
            if self.log:
                with np.printoptions(precision=3):
                    print("absorption times: {}".format(np.array(absorption_times)))
            assert sorted(absorption_times) == absorption_times, \
                    "The (relative) absorption times are sorted: {}".format(np.array(absorption_times))            

            # The list storing the time segments where statistics are computed is sorted
            survival_time_segments = est.get_survival_time_segments()
            if self.log:
                with np.printoptions(precision=3):
                    print("time segments: {}".format(np.array(survival_time_segments)))
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
                print("------ END OF ITER {} of {} ------".format(it+1, niter))

        print("\n\n****** SIMULATION SUMMARY ({} iterations) ******".format(niter))

        #************************************************************************
        # GRAPHICAL RENDER
        if True:
            print("\nSIMULATION RENDER (before treatment of censored values):")
            print(est.render())
        #************************************************************************


        #************************************************************************
        # FINALIZE THE SIMULATION PROCESS BY DEALING WITH ACTIVE PARTICLES (censoring)
        est.finalize()
        print("\n\n****** ESTIMATION PROCESS ({} iterations) ******".format(niter))
        #************************************************************************


        #************************************************************************
        # GRAPHICAL RENDER
        if True:
            print("\nSIMULATION RENDER (after treatment of censored values):")
            print(est.render())
        #************************************************************************



        survival_time_segments = est.get_survival_time_segments()
        if False:
            with np.printoptions(precision=3, suppress=True):
                print("Survival time segments: {}".format(np.array(survival_time_segments)))
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
            with np.printoptions(precision=3, suppress=True):
                print("Blocking time segments: {}".format(np.array(blocking_time_segments)))
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

        print("\nLatest time the system changed: {:.3f}".format(est.get_time_last_change_of_system()))
        times_last_event_by_particle = est.get_all_times_last_event()
        print("Range of latest event times in all {} particles in the system: [{:.3f}, {:.3f}]" \
              .format(est.N, np.min(times_last_event_by_particle), np.max(times_last_event_by_particle)))
        print("Latest particle positions: {}".format(est.positions))
        with np.printoptions(precision=3, suppress=True):
            print("Latest event times: {}".format(np.array(times_last_event_by_particle)))
            particles, elapsed_times_since_activation = est.get_all_elapsed_times()
            print("Latest elapsed times since activation: {}".format(np.array(elapsed_times_since_activation)))
            print("Particles associated to these times  : {}".format(particles))


        #print("\nESTIMATIONS *** METHOD 1 ***:")
        #df_proba_survival_and_blocking_conditional_BF = est.estimate_proba_survival_and_blocking_conditional()
        #with np.printoptions(precision=3, suppress=True):
        #    print(df_proba_survival_and_blocking_conditional_BF)

        # TODO: (2020/06/14) Move the call to compute_counts() inside estimate_proba_blocking()
        print("\nESTIMATIONS *** METHOD 2: FROM observed TIMES ***:")

        ##### IMPORTANT: THE FOLLOWING RECOMPUTATION OF COUNTS SHOULD ALWAYS BE DONE WHEN finalize_type = REMOVE!!
        ##### In fact, the finalize() process by REMOVE changes the counts since it removes time segments!!   
        #est.compute_counts()
        ##### IMPORTANT

        df_proba_survival_and_blocking_conditional = est.estimate_proba_survival_and_blocking_conditional()
        with np.printoptions(precision=3, suppress=True):
            print("Estimated probabilities by time:")
            print(df_proba_survival_and_blocking_conditional)

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

        print("Total blocking time for ALL particles:")
        all_total_blocking_time = est.get_all_total_blocking_time()
        all_total_survival_time = est.get_all_total_survival_time()
        print("")
        print("Total blocking time = {:.3f}".format(all_total_blocking_time))
        print("Total survival time = {:.3f}".format(all_total_survival_time))
        print("% blocking time = {:.1f}%".format(all_total_blocking_time / all_total_survival_time * 100))
        print("\n")

        proba_blocking = est.estimate_proba_blocking()
        rho = self.rate_birth / self.rate_death
        K = self.capacity
        print("\nBlocking probability estimate: {:.1f}%".format(proba_blocking*100))
        print("Theoretical value (rho^K / sum(rho^i)) (rho={:.3f}, K={}): {:.1f}%" \
              .format(rho, K,
                      rho**K / np.sum([ rho**i for i in range(K+1) ]) *100))

        print("Simulation setup:")
        print(est.setup())

        self.plot_results(df_proba_survival_and_blocking_conditional, 
                          reactivate,
                          finalize_type,
                          nparticles,
                          niter)

        return  df_proba_survival_and_blocking_conditional, \
                (reactivate, finalize_type, nparticles, niter)

    def test_simpler_algorithm(self):
        print("\nRunning test " + self.id())
        start = 1
        reactivate = True
        finalize_type = FinalizeType.REMOVE_CENSORED
        nparticles = 3
        nmeantimes = 3
        seed = 1717
        self.run(start=start,
            mean_lifetime=None,
            reactivate=reactivate,
            finalize_type=finalize_type,
            nparticles=nparticles,
            nmeantimes=nmeantimes,
            seed=seed,
            plotFlag=True,
            log=True)

    def run(self,   start=1,
                    mean_lifetime=None,
                    reactivate=True,
                    finalize_type=FinalizeType.REMOVE_CENSORED,
                    nparticles=100,
                    nmeantimes=1000,
                    seed=1717,
                    plotFlag=True,
                    log=False):
        # DM-2020/12/23: We make a copy of the queue so that we do NOT change the queue defined in the constructor of this object
        # For the reactivation case, we start the buffer size of the queue at 1,
        # a situation that is obtained by setting the first server to 1 and the rest to 0.
        initial_server_sizes = np.repeat(0, self.queue.getNServers())
        initial_server_sizes[0] = start
        queue = copy.deepcopy(self.queue)
        queue.setServerSizes(initial_server_sizes)
        job_rates = self.job_rates
        policy = self.policy
        est = estimators.EstimatorQueueBlockingFlemingViot(nparticles, queue,
                                                           nmeantimes=nmeantimes,
                                                           job_rates=job_rates, policy=policy,
                                                           mean_lifetime=mean_lifetime,
                                                           reactivate=reactivate,
                                                           finalize_type=finalize_type,
                                                           plotFlag=plotFlag,
                                                           seed=seed, log=log)
        if log:
            print("Simulation setup:")
            print(est.setup())

        proba_blocking_integral, proba_blocking_laplacian, integral, gamma, expected_survival_time = est.simulate()

        # Estimate the probability of blocking in the Monte Carlo case for benchmarking
        # as the proportion of time blocked over all elapsed time (= total survival time).
        if not reactivate:
            all_total_blocking_time = est.get_all_total_blocking_time()
            all_total_survival_time = est.get_all_total_survival_time()
            prop_blocking_time = all_total_blocking_time / all_total_survival_time
        else:
            prop_blocking_time = None

        if False:
            print("\n\nRelevant events for each particle ID:")
            for P in range(len(est.info_particles)):
                print("Particle ID P={} (P={} --> Q={} (q={})" \
                      .format(P,
                              est.info_particles[P]['particle number'],
                              est.info_particles[P]['reactivated number'], 
                              est.info_particles[P]['reactivated ID']),
                      end="")
                if est.info_particles[P]['t0'] is not None:
                    print(" -> x={} @t={:.3f} @iter={})" \
                          .format(est.info_particles[P]['x'], est.info_particles[P]['t0'], est.info_particles[P]['iter']))
                else:
                    print(")")
                print(np.c_[est.info_particles[P]['t'], est.info_particles[P]['E']])
                print("\n")

        if log and not reactivate:
            if False:
                print("Total blocking time by particle:")
                blocking_periods = est.get_all_blocking_periods()
                survival_periods = est.get_all_survival_periods()
                for P in range(est.N):
                    print("\n\nParticle {}".format(P))
                    print("Block periods:")
                    print(blocking_periods[P])
    
                    print("\nSurvival periods:")
                    print(survival_periods[P])
    
                    total_blocking_time = est.get_total_blocking_time(P)
                    total_survival_time = est.get_total_survival_time(P)
                    print("")
                    print("Total blocking time = {:.3f}".format(total_blocking_time))
                    print("Total survival time = {:.3f}".format(total_survival_time))
                    print("% blocking time = {:.1f}%".format(total_blocking_time / total_survival_time * 100))
                    print("\n")
            print("Total blocking time for ALL particles:")
            print("")
            print("Total blocking time = {:.3f}".format(all_total_blocking_time))
            if reactivate:
                survival_times_for_particles = [ dict_info['t'][-1] for dict_info in est.info_particles ]
                total_survival_time = np.sum( survival_times_for_particles )
                print("Total survival time = {:.3f}".format(total_survival_time))
                print("% blocking time = {:.1f}%".format(all_total_blocking_time / total_survival_time * 100))
            else:
                print("Total survival time = {:.3f}".format(all_total_survival_time))
                print("% blocking time = {:.1f}%".format(all_total_blocking_time / all_total_survival_time * 100))
            print("\n")

        if log:
            rho = self.rate_birth / self.rate_death
            K = self.capacity
            print("\nEstimation of blocking probability via Approximation 1 & 2:")
            print("Integral = {:.6f}".format(integral))
            print("Gamma = {:.6f}".format(gamma))
            print("Expected Survival Time = {:.3f}".format(expected_survival_time))
            print("Blocking probability estimate (Approx. 1): {:.1f}%".format(proba_blocking_integral*100))
            print("Blocking probability estimate (Approx. 2): {:.1f}%".format(proba_blocking_laplacian*100))
            print("Theoretical value (rho^K / sum(rho^i)) (rho={:.3f}, K={}): {:.3f}%" \
                  .format(rho, K,
                          rho**K / np.sum([ rho**i for i in range(K+1) ]) *100))
            print("Last time the system of particle was updated: {:.1f}".format(est.get_time_last_change_of_system()))
    
            print("\nSimulation setup:")
            print(est.setup())

        if plotFlag:
            df_proba_survival_and_blocking_conditional = est.estimate_proba_survival_and_blocking_conditional()
            self.plot_results(df_proba_survival_and_blocking_conditional,
                              mean_lifetime,
                              reactivate,
                              finalize_type,
                              nparticles,
                              niter,
                              seed)

        return  proba_blocking_integral, \
                proba_blocking_laplacian, \
                integral, \
                gamma, \
                expected_survival_time, \
                prop_blocking_time, \
                (mean_lifetime, reactivate, finalize_type, nparticles, niter)

    def analyze_convergence(self,   finalize_type=FinalizeType.REMOVE_CENSORED,
                                    replications=5,
                                    # For the following range specifications,
                                    # each new value is the previous value TIMES the third element of the tuple
                                    # Both min and max are included in the values considered for the simulation
                                    K_range=(5, 40, 2),
                                    nparticles_range=(10, 80, 2),
                                    niter_range=(50, 800, 2), 
                                    seed=1717,
                                    log=False):
        K_min, K_max, K_mult = K_range 
        nparticles_min, nparticles_max, nparticles_mult = nparticles_range 
        niter_min, niter_max, niter_mult = niter_range
        nsimul = int(   (np.ceil( np.log10(K_max / K_min) / np.log10(K_mult)) + 1) * \
                        (np.ceil( np.log10(nparticles_max / nparticles_min) / np.log10(nparticles_mult)) + 1) * \
                        (np.ceil( np.log10(niter_max / niter_min) / np.log10(niter_mult)) + 1) ) 

        df_proba_blocking_estimates = pd.DataFrame.from_items([
                                                               ('K', []),
                                                               ('nparticles', []),
                                                               ('niter', []),
                                                               ('rep', []),
                                                               ('seed', []),
                                                               ('integral', []),
                                                               ('E(T)', []),
                                                               ('PMC(K)', []),
                                                               ('PFV1(K)', []),
                                                               ('PFV2(K)', []),
                                                               ('P(K)', [])
                                                               ])
        rho = self.queue.getBirthRate() / self.queue.getDeathRate()
        np.random.seed(seed)
        i = 0
        K = K_min
        while K <= K_max:
            self.queue.K = K
            print("\n---> NEW K (Queue's capacity = {})".format(self.queue.getCapacity()))
            proba_blocking_K = rho**K / np.sum([ rho**i for i in range(K+1) ])
            nparticles = nparticles_min
            while nparticles <= nparticles_max:
                print("\n\t---> NEW NPARTICLES ({})".format(nparticles))
                niter = niter_min
                while niter <= niter_max:
                    i += 1
                    print("Simulation {} of {}: \n\tK={}, particles={}, iterations={}" \
                          .format(i, nsimul, K, nparticles, niter))
                    for rep in range(replications):
                        print("\tReplication {} of {}".format(rep+1, replications))
                        print("\t1) Estimating the expected survival time...")
                        seed_rep = seed + rep * int(np.round(100*np.random.random()))
                        reactivate = False
                        start = 0
                        mean_lifetime = None
                        _, _, _, _, mean_lifetime_mc, proba_blocking_mc, params_mc = self.run(start,
                                                                        mean_lifetime,
                                                                        reactivate,
                                                                        finalize_type,
                                                                        nparticles,
                                                                        niter,
                                                                        seed_rep,
                                                                        plotFlag=False,
                                                                        log=log)

                        print("\t2) Estimating blocking probability using Fleming-Viot (E(T) = {:.1f})..." \
                              .format(mean_lifetime_mc))
                        reactivate = True
                        start = 1
                        proba_blocking_integral, proba_blocking_laplacian, integral, gamma, expected_survival_time, _, params_fv = \
                                                                self.run(   start,
                                                                            mean_lifetime_mc,
                                                                            reactivate,
                                                                            finalize_type,
                                                                            nparticles,
                                                                            niter,
                                                                            seed_rep,
                                                                            plotFlag=False,
                                                                            log=log)
        
                        df_proba_blocking_estimates = pd.concat([df_proba_blocking_estimates,
                                                                 pd.DataFrame.from_items([
                                                                            ('K', [K]),
                                                                            ('nparticles', [nparticles]),
                                                                            ('niter', [niter]),
                                                                            ('rep', [rep+1]),
                                                                            ('seed', [seed_rep]),
                                                                            ('integral', [integral]),
                                                                            ('E(T)', [expected_survival_time]),
                                                                            ('PMC(K)', [proba_blocking_mc]),
                                                                            ('PFV1(K)', [proba_blocking_integral]),
                                                                            ('PFV2(K)', [proba_blocking_laplacian]),
                                                                            ('P(K)', [proba_blocking_K])]
                                                                                         )],
                                                                 axis=0)
                                                                
                        print("\t--> PMC(K)={:.6f}% vs. PFV1(K)={:.6f}% vs. PFV2(K)={:.6f}% vs. P(K)={:.6f}%" \
                              .format(proba_blocking_mc*100, proba_blocking_integral*100, proba_blocking_laplacian*100, proba_blocking_K*100))

                    if False:
                        # DM-2020/08/23: Stop at a simulation showing large values of PFV2(K) (e.g. 72% when the actual P(K) = 10%!)
                        if K == 5 and nparticles == 10 and niter == 400:
                            import sys
                            sys.exit()

                    if niter < niter_max:
                        # Give the opportunity to reach niter_max (in case niter*niter_mult goes beyond niter_max)
                        niter = np.min([niter*niter_mult, niter_max])
                    else:
                        # We should stop iterating on niter at the next iteration
                        niter = niter * niter_mult
                if nparticles < nparticles_max:
                    # Give the opportunity to reach nparticles_max (in case nparticles*nparticles_mult goes beyond nparticles_max)
                    nparticles = np.min([nparticles*nparticles_mult, nparticles_max])
                else:
                    # We should stop iterating on nparticles at the next iteration
                    nparticles = nparticles * nparticles_mult
            if K < K_max:
                # Give the opportunity to reach K_max (in case K + K_mult goes beyond K_max)
                K = np.min([K*K_mult, K_max])
            else:
                # We should stop iterating on K at the next iteration
                K = K * K_mult

        return df_proba_blocking_estimates, rho

    def plot_results(self, df_proba_survival_and_blocking_conditional, *args):
        mean_lifetime = args[0]
        reactivate = args[1]
        finalize_type = args[2]
        nparticles = args[3]
        niter = args[4]
        seed = args[5]
        print(mean_lifetime)
        print(reactivate)
        print(finalize_type.name)
        print(nparticles)
        print(niter)
        print(seed)

        plt.figure()
        color1 = 'blue'
        color2 = 'red'
        #y2max = 0.05
        y2max = 1.0
        #y2max = 1.1*np.max(df_proba_survival_and_blocking_conditional['P(BLOCK / T>t,s=1)'])
        plt.step(df_proba_survival_and_blocking_conditional['t'], df_proba_survival_and_blocking_conditional['P(T>t / s=1)'],
                 'b-', where='post')
        ax = plt.gca()
        ax.set_xlabel('t')
        ax.spines['left'].set_color(color1)
        ax.tick_params(axis='y', color=color1)
        ax.yaxis.label.set_color(color1)
        ax2 = ax.twinx()
        ax2.step(df_proba_survival_and_blocking_conditional['t'], df_proba_survival_and_blocking_conditional['P(BLOCK / T>t,s=1)'],
                 'r-', where='post')
        ax2.set_ylim(0, y2max)
        ax.spines['right'].set_color(color2)
        ax2.tick_params(axis='y', color=color2)
        ax2.yaxis.label.set_color(color2)
        plt.sca(ax)
        ax.legend(['P(T>t / s=1)'], loc='upper left')
        ax2.legend(['P(BLOCK / T>t,s=1)'], loc='upper right')
        plt.title("K={}, rate(B)={:.1f}, rate(D)={:.1f}, rho={:.3f}, mean_lifetime={:.1f}" \
                  ", reactivate={}, finalize={}, N={}, #iter={}, seed={}" \
                  .format(self.capacity,
                          self.queue.getBirthRate(),
                          self.queue.getDeathRate(),
                          self.queue.getBirthRate() / self.queue.getDeathRate(),

                          mean_lifetime or np.nan, reactivate, finalize_type.name[0:3],
                          nparticles, niter, seed
                          ))
        ax.title.set_fontsize(9)

    def plot_convergence_analysis_allvalues(self, results_convergence):

        def plot(convergence_results_agg_by_K_nparticles_niter,
                 grp_K, grp_axis, grp_legend,
                 replications):
            """
            The input data frame should be aggregated by K, nparticles, niter IN THIS ORDER and
            be created via the pd.groupby() method using as_index=True.
            """
            df_agg = convergence_results_agg_by_K_nparticles_niter
            all_K, all_nparticles, all_niters = df_agg.index.levels
            
            grp_legend_idx = df_agg.index.names.index(grp_legend)
            grp_axis_idx = df_agg.index.names.index(grp_axis)
            grp_legend_values = df_agg.index.levels[grp_legend_idx]
            grp_axis_values = df_agg.index.levels[grp_axis_idx]
            se_mult = 2
            axes = plt.figure(figsize=(9,16)).subplots(len(all_K),2)
            cmap_red = cm.get_cmap('Reds')
            cmap_green = cm.get_cmap('Greens')
            color_norm = matplotlib.colors.Normalize(vmin=0.8*grp_legend_values[0], vmax=grp_legend_values[-1])
            for idx, K in enumerate(all_K):
                ind_K = df[grp_K] == K
                P_true = df[ind_K].iloc[0]['P(K)']
                (ax_mc, ax_fv) = axes[idx]
                legend_lines_mc  = []
                legend_lines_fv  = []
                legend_labels_mc = []
                legend_labels_fv = []
                for idx_legend, legend_value in enumerate(grp_legend_values):
                    if grp_legend_idx == 1:
                        # This tells us how 
                        df2plot = df_agg.loc[(K, legend_value)]
                    else:
                        df2plot = df_agg.loc[(K, list(grp_axis_values), legend_value)]
                    if idx_legend == len(grp_legend_values) - 1:
                        color_red = "red"
                        color_green = "green"
                    else:
                        color_strength = color_norm(legend_value)
                        color_red = cmap_red(color_strength)
                        color_green = cmap_green(color_strength)
                    yerr_mc = se_mult*df2plot['SE'][y_mc]*100
                    yerr_fv = se_mult*df2plot['SE'][y_fv]*100
                    line_mc = ax_mc.errorbar(grp_axis_values, df2plot['mean'][y_mc]*100, yerr=yerr_mc,
                                capsize=4, color=color_red, marker='.')
                    line_fv = ax_fv.errorbar(grp_axis_values, df2plot['mean'][y_fv]*100, yerr=yerr_fv,
                                capsize=4, color=color_green, marker='.')
                    legend_lines_mc  += [line_mc]
                    legend_lines_fv  += [line_fv]
                    legend_labels_mc += ["MC: {}={}".format(grp_legend, int(legend_value))]
                    legend_labels_fv += ["FV: {}={}".format(grp_legend, int(legend_value))]
        
                    # Add violinplots for the largest parameter
                    if idx_legend == len(grp_legend_values) - 1:
                        import Python.lib.utils.plotting  as plotting
                        ind = (df[grp_K] == K) & (df[grp_legend] == legend_value)
                        plotting.violinplot(ax_mc,  [df[ind & (df[grp_axis]==x)][y_mc]*100 for x in grp_axis_values],
                                                    positions=grp_axis_values, showmedians=False, linewidth=4,
                                                    color_body="red", color_lines="red", color_means="red")            
                        plotting.violinplot(ax_fv,  [df[ind & (df[grp_axis]==x)][y_fv]*100 for x in grp_axis_values],
                                                    positions=grp_axis_values, showmedians=False, linewidth=4,
                                                    color_body="green", color_lines="green", color_means="green")            
        
                    for ax in (ax_mc, ax_fv):
                        refline = ax.axhline(y=P_true*100, color='black', linestyle='dashed')
                        ax.set_xlabel(grp_axis)
                        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                    ax_mc.set_ylabel('K = {:.0f}'.format(K), fontsize=12)
        
                ymax = np.max(np.r_[P_true, df[ind_K][y_mc], df[ind_K][y_fv]])*100
                for ax in (ax_mc, ax_fv):
                    ax.set_ylim((0, ymax*1.1))
                ax_mc.legend((refline, *legend_lines_mc),
                             ("True value (P={:.3f}%)".format(P_true*100), *legend_labels_mc),
                             ncol=2,
                             fontsize='x-small')
                ax_fv.legend((refline, *legend_lines_fv),
                             ("True value (P={:.3f}%)".format(P_true*100), *legend_labels_fv),
                             ncol=2,
                             fontsize='x-small')
                plt.suptitle("Estimated blocking probability P(K) using a particle system" \
                             " (Mean +/-{}SE + Boxplots of largest legend value, replications={})" \
                             "\nMonte Carlo (red) vs. Fleming Viot (green)" \
                             "\n\nDependency with {}" \
                             .format(se_mult, replications, grp_axis.upper()))
        # Define the analysis data frame and the name of the variables involved in the plots
        df = results_convergence
        grp_part = 'nparticles'
        grp_iter = 'niter'
        grp_K = 'K'
        y_mc = 'PMC(K)'
        y_fv = 'PFV1(K)'   # Approximation 1
        replications = int(np.max(df['rep']))

        # Analysis by group (mean, std, min, max)
        grouped = df.groupby([grp_K, grp_part, grp_iter], as_index=True, group_keys=False)
        agg_functions = {'n': 'count', 'mean': np.mean, 'std': np.std, 'min': 'min', 'max': 'max'}
        df_agg = grouped[y_mc, y_fv].agg(agg_functions)
        df_agg_se = df_agg['std'][[y_mc, y_fv]] / np.sqrt(df_agg['n'][[y_mc, y_fv]])
        df_agg_se.set_axis([['SE','SE'],[y_mc, y_fv]], axis=1, inplace=True)
        df_agg = pd.concat([df_agg, df_agg_se], axis=1)

        # Plot convergence analysis for ALL parameter values
        # first for grp_part, then for grp_iter
        grp_axis = grp_part
        grp_legend = grp_iter
        plot(df_agg,
             grp_K, grp_axis, grp_legend,
             replications)

        grp_axis = grp_iter
        grp_legend = grp_part
        plot(df_agg,
             grp_K, grp_axis, grp_legend,
             replications)

    def plot_convergence_analysis_maxvalues(self, convergence_results):

        def df_agg_param(df4param, grp_K, grp_param):
            """
            Computes summary statistics in the input data frame by K and by the values
            of the parameter given in grp_param (e.g. 'nparticles', 'niter').
            """
            grouped_param = df4param.groupby([grp_K, grp_param], as_index=True, group_keys=False)
            agg_functions = {'n': 'count', 'mean': np.mean, 'std': np.std}
            df_agg_byparam = grouped_param[y_mc, y_fv].agg(agg_functions)
            df_agg_byparam_se = df_agg_byparam['std'][[y_mc,y_fv]] / np.sqrt(df_agg_byparam['n'][[y_mc, y_fv]])
            df_agg_byparam_se.set_axis([['SE','SE'],[y_mc,y_fv]], axis=1, inplace=True)
            df_agg_byparam = pd.concat([df_agg_byparam, df_agg_byparam_se], axis=1)

            return df_agg_byparam

        def plot(df2plot, grp_K, grp_axis, legend, se_mult=2):
            all_K = df2plot.index.levels[0]

            legend_label_mc = "MC: {}".format(legend)
            legend_label_fv = "FV: {}".format(legend)
            axes = plt.figure(figsize=(9,16)).subplots(len(all_K),1)
            for idx, K in enumerate(all_K):
                P_true = df[df[grp_K] == K].iloc[0]['P(K)']
                ax = axes[idx]
                ax.errorbar(df2plot.loc[K].index, df2plot.loc[K]['mean'][y_mc], yerr=se_mult*df2plot.loc[K]['SE'][y_mc],
                            capsize=4, color='red', marker='.')
                ax.errorbar(df2plot.loc[K].index, df2plot.loc[K]['mean'][y_fv], yerr=se_mult*df2plot.loc[K]['SE'][y_fv],
                            capsize=4, color='green', marker='.')
                ax.axhline(y=P_true, color='black', linestyle='dashed')
                ax.set_xlabel(grp_axis)
                ax.set_ylabel('K = {:.0f}'.format(K))
                ymax = np.max(np.r_[P_true,
                                    df2plot.loc[K]['mean'][y_mc] + se_mult*df2plot.loc[K]['SE'][y_mc],
                                    df2plot.loc[K]['mean'][y_fv] + se_mult*df2plot.loc[K]['SE'][y_fv]])
                ax.set_ylim((0, ymax*1.1))
                ax.legend(["True value (K={}, P={:.2g}%)".format(K, P_true*100),
                           legend_label_mc,
                           legend_label_fv])
                plt.suptitle("P(K) vs. {} in the Fleming Viot system (+/-{}SE, replications={})" \
                             .format(grp_axis.upper(), se_mult, replications))

        # Define the analysis data frame and the name of the variables involved in the plots
        df = results_convergence
        grp_part = 'nparticles'
        grp_iter = 'niter'
        grp_K = 'K'
        y_mc = 'PMC(K)'
        y_fv = 'PFV1(K)'  # Approximation 1
        replications = int(np.max(df['rep']))
        
        # Filter each analysis by group by the largest value of the other group variable
        all_niters = np.unique(df[grp_iter])
        niters_max = int(all_niters[-1])
        legend = "{} = {}".format(grp_iter, niters_max)
        df4part = df[ df[grp_iter] == niters_max ]
        df_agg_byparam = df_agg_param(df4part, grp_K, grp_part)
        plot(df_agg_byparam, grp_K, grp_part, legend, se_mult=2)

        all_nparticles = np.unique(df[grp_part])
        nparticles_max = int(all_nparticles[-1])
        legend = "{} = {}".format(grp_part, nparticles_max)
        df4iter = df[ df[grp_part] == nparticles_max ]
        df_byiter = df_agg_param(df4iter, grp_K, grp_iter)
        plot(df_byiter, grp_K, grp_iter, legend, se_mult=2)


# DM-2020/12/23: To change which portion of the below code to run, change the IF condition
# to `== "__main__"` or to `!= "__main__"` accordingly
if __name__ == "__main__":
    #unittest.main()

    # DM-2020/08/24: Instead of using unittest.main(), use the following to test the FV system
    # because there is a very weird error generated by the fact that queue.size inside
    # the EstimatorQueueBlockingFlemingViot class is interpreted of being of class
    # unittest.runner.TextTestResult (or similar)!! instead of simply returning the size
    # attribute of the `queue` object!!!
    test = Test_QB_Particles()
    test.test_simpler_algorithm()

else:
    # Lines to execute "by hand" (i.e. at the Python prompt)
    test = Test_QB_Particles()
    
    time_start = timer()
    results_convergence, rho = test.analyze_convergence(
                                    finalize_type=FinalizeType.REMOVE_CENSORED,
                                    replications=5,
                                    K_range=(5, 40, 2),
                                    nparticles_range=(10, 40, 2),
                                    niter_range=(50, 400, 2), 
                                    seed=1717,
                                    log=False)
    time_end = timer()

    print("Top and bottom 5 records in the results data frame:")
    print(results_convergence.head())
    print(results_convergence.tail())

    print("Execution time: {:.1f} min".format(timedelta(time_end - time_start).seconds / 60))

    #filename = "../../RL-002-QueueBlocking/results/fv_approx1_convergence_rho={:.3f}.csv"
    #results_convergence.to_csv(filename.format(rho))
    #print("Results of simulation saved to {}".format(filename))    

    # Plots
    test.plot_convergence_analysis_allvalues(results_convergence)
    test.plot_convergence_analysis_maxvalues(results_convergence)
