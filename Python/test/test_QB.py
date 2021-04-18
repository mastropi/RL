# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:16:02 2020

@author: Daniel Mastropietro
@description: Estimators are defined for problems of interest in Reinforcement Learning.
"""

import runpy
runpy.run_path('../../setup.py')

import os
import numpy as np
import pandas as pd
import copy

from timeit import default_timer as timer
#from datetime import timedelta 
import unittest
from unittest_data_provider import data_provider
import matplotlib
from matplotlib import pyplot as plt, cm, ticker as mtick
import Python.lib.utils.plotting  as plotting

import Python.lib.queues as queues
import Python.lib.estimators as estimators
from Python.lib.queues import Event
from Python.lib.estimators import FinalizeType
from Python.lib.utils.computing import compute_blocking_probability_birth_death_process

#from importlib import reload
#reload(estimators)
#from Python.lib.estimators import EstimatorQueueBlockingFlemingViot

#import test_utils


class Test_QB_Particles(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = False
        self.rate_birth = 0.5
        self.capacity = 5
        
        #self.nservers = 1
        self.nservers = 3

        if self.nservers == 1:
            # One server
            self.job_rates = [self.rate_birth]
            self.rate_death = [1]
            self.policy = [[1]]
        elif self.nservers == 3:
            # Multiple servers
            self.job_rates = [0.8, 0.7]
            self.rate_death = [1, 1, 1]
            self.policy = [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]

        # rho rates for each server based on arrival rates and assignment probabilities
        self.rhos = self.compute_rhos()

        self.queue = queues.QueueMM(self.rate_birth, self.rate_death, self.nservers, self.capacity)

        self.plotFlag = True

    def compute_rhos(self):
        "Computes the rho rates for each server based on arrival rates, service rates, and assignment probabilities"
        R = self.nservers
        J = len(self.job_rates) # Number of job classes
        rhos = [0]*self.nservers
        for r in range(R):
            for c in range(J):
                rhos[r] += self.policy[c][r] * self.job_rates[c] / self.rate_death[r]
        
        return rhos

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
        nmeantimes = 100
        est = estimators.EstimatorQueueBlockingFlemingViot(1, self.queue, 0.5,
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
        est = estimators.EstimatorQueueBlockingFlemingViot(nparticles, self.queue, 0.5,
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
        est = estimators.EstimatorQueueBlockingFlemingViot(nparticles, self.queue, 0.5,
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

        print("\nLatest time the system was updated to a known state: {:.3f}".format( est.get_time_latest_known_state() ))
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
        df_proba_survival_and_blocking_conditional_BF = est.estimate_proba_survival_and_blocking_conditional()
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
        proba_blocking_K = compute_blocking_probability_birth_death_process(self.nservers, K, self.rhos)
        if proba_blocking_K is not None:
            print("Theoretical value: {:.1f}%".format(proba_blocking_K*100))

        print("Simulation setup:")
        print(est.setup())

        self.plot_curve_estimates(df_proba_survival_and_blocking_conditional,
                          est.queue.getBirthRates(),
                          est.queue.getDeathRates(),
                          K,
                          nparticles,
                          nmeantimes,
                          buffer_size_activation,
                          None,
                          reactivate,
                          finalize_type,
                          seed)

        return  df_proba_survival_and_blocking_conditional, \
                (reactivate, finalize_type, nparticles, nmeantimes)

    @classmethod
    def test_fv_implementation(cls):
        #--- Test one server
        nservers = 1
        K = 5
        rate_birth = 0.5
        job_rates = [rate_birth]
        rate_death = [1]
        policy = [[1]]
        queue = queues.QueueMM(rate_birth, rate_death, nservers, K)
    
        # The test of the Fleming-Viot implementation is carried out as follows:
        # - Set K to a small value (e.g. K=5)
        # - Increase the number of particles N
        # - Set a large simulation time (e.g. nmeantimes=50)
        # - Increase the number of particles N check that the error
        # between the estimated and true blocking probability decreases as 1/sqrt(N)
        # (according to Propostion 2.1 in Matt's draft)

        # Simulation parameters
        buffer_size_activation = 1 #int(test.capacity/2)
        finalize_type = FinalizeType.ABSORB_CENSORED
        nmeantimes = 20
        seed = 1717
        plotFlag = False
        log = False

        replications = 10
        nparticles_min = 50
        nparticles_max = 600
        nparticles_step_prop = 0.5  # STEP proportion: N(n+1) = (1 + prop)*N(n), so that we scale the step as the number of particles increases
        nparticles = nparticles_min
        df_results = pd.DataFrame(columns=['N', 'replication', 'Pr(MC)', 'Time(MC)', 'E(T)', 'Pr(FV)', 'Time(FV)', 'Pr(K)', 'exec_time(s)'])
        time_start_all = timer()
        case = 0
        ncases = int( np.log(nparticles_max / nparticles_min) / np.log(1 + nparticles_step_prop)) + 1
        while nparticles <= nparticles_max:
            case += 1 
            print("Running simulation for nparticles={} on {} replications ({} of {})...".format(nparticles, replications, case, ncases))
        
            for r in range(1, replications+1):
                print("\tReplication {} of {}...".format(r, replications), end=" ")
                seed_rep = seed + r - 1

                # 1) MC: Run simulation to estimate expected return time to absorption set via Monte-Carlo 
                # Note that we use only ONE particle, as we need to guarantee that we observe enough
                # time for the system to be absorbed several times.
                # In order to make the comparison between the MC estimate of the blocking probability
                # (which is done as well!) and the FV estimate, i.e. to have an honest comparison,
                # we use the same N*T value in both situations, which, since N=1 in MC we choose
                # T(MC) = T(FV)*N(FV).
                time_start = timer()
                est_mc = estimators.EstimatorQueueBlockingFlemingViot(1, queue, job_rates,
                                                           service_rates=None,
                                                           buffer_size_activation=buffer_size_activation,
                                                           nmeantimes=nmeantimes*nparticles,
                                                           policy_assign=policy,
                                                           mean_lifetime=None,
                                                           reactivate=False,
                                                           finalize_type=FinalizeType.REMOVE_CENSORED,
                                                           plotFlag=plotFlag,
                                                           seed=seed_rep, log=log)
                proba_blocking_mc, total_blocking_time, total_survival_time, expected_survival_time = est_mc.simulate()
                proba_survival_given_activation = est_mc.estimate_proba_survival_given_activation()
                if nparticles == nparticles_min:
                    # Compute the theoretical blocking probability only at the first iteration
                    rhos = est_mc.rhos
                    proba_blocking_true = compute_blocking_probability_birth_death_process(nservers, K, rhos)
    
                # b) FV: Fleming-Viot
                # The estimated expected return time to the absorption set is used as input
                est_fv = estimators.EstimatorQueueBlockingFlemingViot(nparticles, queue, job_rates,
                                                           service_rates=None,
                                                           buffer_size_activation=buffer_size_activation,
                                                           nmeantimes=nmeantimes,
                                                           policy_assign=policy,
                                                           mean_lifetime=expected_survival_time,
                                                           proba_survival_given_activation=proba_survival_given_activation,
                                                           reactivate=True,
                                                           finalize_type=finalize_type,
                                                           plotFlag=plotFlag,
                                                           seed=seed_rep, log=log)
                proba_blocking_fv_integral, proba_blocking_fv_laplacian, integral, gamma = est_fv.simulate()
                time_end = timer()
                exec_time = time_end - time_start
                print("execution time: {:.1f} sec".format(exec_time))
            
                # Results
                assert est_mc.maxtime == est_fv.maxtime*nparticles, \
                    "The simulation time of the MC ({:.1f}) and the FV ({:.1f}) runs are comparable" \
                    .format(est_mc.maxtime, est_fv.maxtime*nparticles)
                print("\tP(K) by MC: {:.3f}%, E(T) = {:.1f} (simulation time = {:.1f})".format(proba_blocking_mc*100, expected_survival_time, est_mc.maxtime))
                print("\tP(K) estimated by FV1: {:.3f}% (simulation time = {:.1f})".format(proba_blocking_fv_integral*100, est_fv.maxtime))
                print("\tP(K) estimated by FV2: {:.3f}% (simulation time = {:.1f})".format(proba_blocking_fv_laplacian*100, est_fv.maxtime))
                print("\tTrue P(K): {:.3f}%".format(proba_blocking_true*100))
                
                if r == 1:
                    # Plot the survival curve and condition blocking probability for the first replication
                    df_proba_survival_and_blocking_conditional_mc = est_mc.estimate_proba_survival_and_blocking_conditional()
                    df_proba_survival_and_blocking_conditional = est_fv.estimate_proba_survival_and_blocking_conditional()
                    cls.plot_curve_estimates(df_proba_survival_and_blocking_conditional,
                                      est_mc.queue.getBirthRates(),
                                      est_mc.queue.getDeathRates(),
                                      K,
                                      nparticles,
                                      nmeantimes,
                                      buffer_size_activation,
                                      expected_survival_time,
                                      True,
                                      finalize_type,
                                      seed_rep)
                    
                    if False:
                        # Compare the survival probability given activation between MC and FV estimations
                        # (assuming FV does NOT receive the estimate from MC when constructing the object above, clearly!)
                        killing_rates = df_proba_survival_and_blocking_conditional_mc['Killing Rate']
                        killing_rates_notnan = killing_rates[ ~np.isnan(killing_rates) ]
                        gamma = np.mean( killing_rates_notnan.iloc[-20:-1] )   # Take the estimate of gamma from the tail of the killing rate values, as its estimate is more valid as t -> Infinity
                        gamma_std = np.std( killing_rates_notnan.iloc[-20:-1] )
                        t = df_proba_survival_and_blocking_conditional_mc['t']
                        n_killings_mc = len(est_mc.sk)
                        n_killings_fv = len(est_fv.sk)
                        plt.figure()
                        ax = plt.gca()
                        ax.step(t, df_proba_survival_and_blocking_conditional_mc['P(T>t / s=1)'], 'r-', where='post')
                        ax.step(df_proba_survival_and_blocking_conditional['t'], df_proba_survival_and_blocking_conditional['P(T>t / s=1)'], 'g-', where='post')
                        ax.plot(t, np.exp(-gamma*t), 'k--')
                        ax.set_xlabel("t")
                        ax.set_ylabel("P(T>t, s=1)")
                        ax.set_xlim([0, 20]) #np.max(t)])
                        plt.title("Comparison between the MC estimate and the FV estimate of P(T>t / s=1)")
                        ax.legend(["MC (based on {} killings)".format(n_killings_mc),
                                   "FV (based on {} killings)".format(n_killings_fv),
                                   "exp(-gamma*t) (avg. gamma={:.3f} +/- {:.3f})".format(gamma, gamma_std)])
    
                # Store the results
                df_append = pd.DataFrame([[nparticles, r, proba_blocking_mc, est_mc.maxtime, expected_survival_time, proba_blocking_fv_integral, est_fv.maxtime, proba_blocking_true, exec_time]], columns=df_results.columns, index=[case])
                df_results = df_results.append(df_append)
            
            nparticles += int( nparticles_step_prop*nparticles )
        time_end_all = timer()

        print("Total execution time: {:.1f} min".format((time_end_all - time_start_all) / 60))
        title = "Simulation results for c={}, K={}, rhos={}, ({}<=N<={}), T<={:.0f}".format(nservers, K, rhos, nparticles_min, nparticles_max, est_fv.maxtime)
        print(title)
        print("Raw results by N:")
        print(df_results)
        
        df_results_agg_by_N = cls.aggregation_bygroups(df_results, ['N'], ['Pr(MC)', 'Pr(FV)'])
        print("Aggregated results by N:")
        print(df_results_agg_by_N)

        plt.figure()
        legend_lines_mc = []
        legend_lines_fv = []
        legend_lines_ref = []
        ax = plt.gca()
        plt.plot(df_results['N'], df_results['Pr(MC)']*100, 'r.', markersize=2)
        line_mc = plt.errorbar(list(df_results_agg_by_N.index), df_results_agg_by_N['mean']['Pr(MC)']*100, yerr=2*df_results_agg_by_N['SE']['Pr(FV)']*100, capsize=4, color='red', marker='x')
        legend_lines_mc += [line_mc]
        plt.plot(df_results['N'], df_results['Pr(FV)']*100, 'g.', markersize=2)
        line_fv = plt.errorbar(list(df_results_agg_by_N.index), df_results_agg_by_N['mean']['Pr(FV)']*100, yerr=2*df_results_agg_by_N['SE']['Pr(FV)']*100, capsize=4, color='green', marker='x')
        legend_lines_fv += [line_fv]
        line_ref = ax.hlines(df_results.iloc[0]['Pr(K)']*100, df_results.iloc[0]['N'], df_results.iloc[-1]['N'], color='gray', linestyles='dashed')
        legend_lines_ref += [line_ref]
        # Ref: https://matplotlib.org/3.3.3/gallery/statistics/boxplot_color.html
        #bplot_mc = plt.boxplot(df_results['Pr(MC)']*100, vert=True, notch=True, patch_artist=True)
        #for patch in bplot_mc['boxes']:
        #    patch.set_facecolor('lightred')
        #bplot_fv = plt.boxplot(df_results['Pr(FV)']*100, vert=True, notch=True, patch_artist=True)
        #for patch in bplot_fv['boxes']:
        #    patch.set_facecolor('lightgreen')
        plt.title(title)
        #ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xlabel("N (number of particles)")
        ax.set_ylabel("Blocking probability (%)")
        ax.legend(legend_lines_mc + legend_lines_fv + legend_lines_ref, ['MC +/- 2SE', 'FV +/- 2SE', 'True'], fontsize='x-small')

        return df_results, df_results_agg_by_N

    def run_simulation(self,   buffer_size_activation=1,
                    mean_lifetime=None,
                    reactivate=True,
                    finalize_type=FinalizeType.ABSORB_CENSORED,
                    nparticles=200,
                    nmeantimes=50,
                    seed=1717,
                    plotFlag=True,
                    log=False):
        queue = self.queue
        job_rates = self.job_rates
        policy = self.policy
        est = estimators.EstimatorQueueBlockingFlemingViot(nparticles, queue, job_rates,
                                                           service_rates=None,
                                                           buffer_size_activation=buffer_size_activation,
                                                           nmeantimes=nmeantimes,
                                                           policy_assign=policy,
                                                           mean_lifetime=mean_lifetime,
                                                           reactivate=reactivate,
                                                           finalize_type=finalize_type,
                                                           plotFlag=plotFlag,
                                                           seed=seed, log=log)
        if log:
            print("\n(START) Simulation setup: (REACTIVATE={})".format(reactivate))
            print(est.setup())
            print("Simulating...")

        if reactivate:
            proba_blocking_integral, proba_blocking_laplacian, integral, gamma = est.simulate()
        else:
            proba_blocking_mc, total_blocking_time, total_survival_time, expected_survival_time = est.simulate()

        if False:
            # Estimate the probability of blocking in the Monte Carlo case for benchmarking
            # as the proportion of time blocked over all elapsed time (= total survival time).
            print("Estimating blocking time rate over all {} particles (K={})...".format(nparticles, self.capacity))

            print("\tComputing survival time over all {} particles...".format(nparticles))
            ts1 = timer()
            all_total_survival_time = est.get_all_total_survival_time()
            ts2 = timer()
            print("\t\ttook {:.1f} sec".format((ts2 - ts1)))

            print("\tComputing blocking time over all {} particles...".format(nparticles))
            tb1 = timer()
            all_total_blocking_time = est.get_all_total_blocking_time()
            tb2 = timer()
            print("\t\ttook {:.1f} sec".format((tb2 - tb1)))

            if not reactivate:
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
    
            if True: #log and not reactivate:
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
    
                        total_blocking_time_P = est.get_total_blocking_time(P)
                        total_survival_time_P = est.get_total_survival_time(P)
                        print("")
                        print("Total blocking time = {:.3f}".format(total_blocking_time_P))
                        print("Total survival time = {:.3f}".format(total_survival_time_P))
                        print("% blocking time = {:.1f}%".format(total_blocking_time_P / total_survival_time_P * 100))
                        print("\n")
                print("\nTotal blocking time for ALL particles:")
                print("Total blocking time (all particles) = {:.3f}".format(all_total_blocking_time))
                print("Total survival time (all particles) = {:.3f}".format(all_total_survival_time))
                print("% blocking time = {:.1f}%".format(all_total_blocking_time / all_total_survival_time * 100))
                print("\n")
        else:
            prop_blocking_time = None

        K = self.capacity
        if log:
            if reactivate:
                print("\nEstimation of blocking probability via Fleming-Viot (Approximation 1 & 2):")
                print("Integral = {:.6f}".format(integral))
                print("Gamma = {:.6f}".format(gamma))
                print("Blocking probability estimate (Approx. 1): {:.6f}%".format(proba_blocking_integral*100))
                print("Blocking probability estimate (Approx. 2): {:.6f}%".format(proba_blocking_laplacian*100))
            else:
                print("\nEstimation of blocking probability via Monte-Carlo:")
                print("Expected Survival Time = {:.3f}".format(expected_survival_time))
                print("Blocking probability estimate (proportion of blocking time / survival time): {:.3f}%".format(proba_blocking_mc*100))
            print("Last time the system has a known state: {:.3f}".format( est.get_time_latest_known_state() ))
    
            print("\n(END) Simulation setup: (REACTIVATE={})".format(reactivate))
            print(est.setup())

        if plotFlag:
            print("Plotting...")
            df_proba_survival_and_blocking_conditional = est.estimate_proba_survival_and_blocking_conditional()
            self.plot_curve_estimates(df_proba_survival_and_blocking_conditional,
                              est.queue.getBirthRates(),
                              est.queue.getDeathRates(),
                              K,
                              nparticles,
                              nmeantimes,
                              buffer_size_activation,
                              mean_lifetime,
                              reactivate,
                              finalize_type,
                              seed)

        # Note: `est` is the object used for the estimation process (so that we can do further calculations outside if needed)
        if reactivate:
            return  est, \
                    proba_blocking_integral, \
                    proba_blocking_laplacian, \
                    integral, \
                    gamma, \
                    (mean_lifetime, reactivate, finalize_type, nparticles, nmeantimes)
        else:
            return  est, \
                    proba_blocking_mc, \
                    total_blocking_time, \
                    total_survival_time, \
                    expected_survival_time, \
                    (mean_lifetime, reactivate, finalize_type, nparticles, nmeantimes)
                    
    def analyze_convergence(self,   finalize_type=FinalizeType.REMOVE_CENSORED,
                                    replications=5,
                                    # For the following range specifications,
                                    # each new value is the previous value TIMES the third element of the tuple
                                    # Both min and max are included in the values considered for the simulation
                                    K_range=(5, 40, 2),
                                    buffer_size_activation_values=[1],
                                    nparticles_range=(10, 80, 2),
                                    nmeantimes_range=(50, 800, 2), 
                                    seed=1717,
                                    log=False):
        K_min, K_max, K_mult = K_range 
        nparticles_min, nparticles_max, nparticles_mult = nparticles_range 
        nmeantimes_min, nmeantimes_max, nmeantimes_mult = nmeantimes_range
        # Number of simulations to run based on the input parameters defining simulation characteristics
        nsimul = int(   (np.ceil( np.log10(K_max / K_min) / np.log10(K_mult)) + 1) * \
                        len(buffer_size_activation_values) * \
                        (np.ceil( np.log10(nparticles_max / nparticles_min) / np.log10(nparticles_mult)) + 1) * \
                        (np.ceil( np.log10(nmeantimes_max / nmeantimes_min) / np.log10(nmeantimes_mult)) + 1) )

        df_proba_blocking_estimates = pd.DataFrame.from_items([
                                                               ('K', []),
                                                               ('nparticles', []),
                                                               ('nmeantimes', []),
                                                               ('buffer_size_activation', []),
                                                               ('buffer_size_activation_value', []),
                                                               ('rep', []),
                                                               ('seed', []),
                                                               ('integral', []),
                                                               ('E(T)', []),
                                                               ('PMC(K)', []),
                                                               ('PFV1(K)', []),
                                                               ('PFV2(K)', []),
                                                               ('Pr(K)', []),
                                                               ])
        np.random.seed(seed)
        case = 0
        K = K_min
        while K <= K_max:
            self.queue.K = K
            print("\n---> NEW K (Queue's capacity = {})".format(self.queue.getCapacity()))

            buffer_size_activation_value_prev = None
            for buffer_size_activation in buffer_size_activation_values:
                # When the buffer size for activation parameter is smaller than 1 it is considered a proportion of the queue's capacity
                if buffer_size_activation < 1:
                    buffer_size_activation_value = max(1, int( buffer_size_activation * K ))
                else:
                    buffer_size_activation_value = buffer_size_activation
                # Do not repeat the previous buffer size activation value (which may happen when the parameter is given as a proportion)
                if buffer_size_activation_value == buffer_size_activation_value_prev:
                    continue
                print("\n\t---> NEW BUFFER SIZE({})".format(buffer_size_activation_value))
    
                print("Computing TRUE blocking probability...", end=" --> ")
                time_pr_start = timer()
                proba_blocking_K = compute_blocking_probability_birth_death_process(self.nservers, K, self.rhos)
                time_pr_end = timer()
                print("{:.1f} sec".format(time_pr_end - time_pr_start))
                print("Pr(K)={:.6f}%".format(proba_blocking_K*100))
    
                nparticles = nparticles_min
                while nparticles <= nparticles_max:
                    print("\n\t\t---> NEW NPARTICLES ({})".format(nparticles))
                    nmeantimes = nmeantimes_min
                    while nmeantimes <= nmeantimes_max:
                        case += 1
                        print("******************!!!!!!! Simulation {} of {} !!!!!!*****************\n\tK={}, buffer_size_activation={}, particles={}, nmeantimes={}" \
                              .format(case, nsimul, K, buffer_size_activation_value, nparticles, nmeantimes))
                        for rep in range(replications):
                            print("\n\tReplication {} of {}".format(rep+1, replications))
                            print("\t1) Estimating the expected survival time using NO reactivation...", end=" --> ")
                            # NOTE THAT THE FIRST REPLICATION (rep=0) HAS THE SAME SEED FOR ALL PARAMETER SETTINGS
                            # This is nice because we can a little bit better compare the effect of the different parameter settings
                            # (but not so much anyway, because the values that are generated as the parameter settings
                            # change impact whatever is generated next --e.g. the initial events for particle 2
                            # will change if more events are generated for particle 1 when the simulation time increases...)
                            seed_rep = seed + rep # * int(np.round(100*np.random.random()))
                            reactivate = False
                            mean_lifetime = None
                            time_mc_start = timer()
                            est_mc, proba_blocking_mc, total_blocking_time_mc, total_survival_time_mc, mean_lifetime_mc, params_mc = \
                                self.run_simulation(
                                        buffer_size_activation_value,
                                        mean_lifetime,
                                        reactivate,
                                        FinalizeType.REMOVE_CENSORED,
                                        1,
                                        nparticles*nmeantimes,
                                        seed_rep,
                                        plotFlag=False,
                                        log=log)
                            time_mc_end = timer()
                            print("{:.1f} sec".format(time_mc_end - time_mc_start))
    
                            print("\t2) Estimating blocking probability using Fleming-Viot (E(T) = {:.1f})..." \
                                  .format(mean_lifetime_mc), end=" --> ")
                            reactivate = True
                            time_fv_start = timer()
                            est_fv, proba_blocking_integral, proba_blocking_laplacian, integral, gamma, params_fv = \
                                self.run_simulation(
                                        buffer_size_activation_value,
                                        mean_lifetime_mc,
                                        reactivate,
                                        finalize_type,
                                        nparticles,
                                        nmeantimes,
                                        seed_rep,
                                        plotFlag=False,
                                        log=log)
                            time_fv_end = timer()
                            print("{:.1f} sec".format(time_fv_end - time_fv_start))
    
                            # Compute the rate of blocking time w.r.t. total simulation time
                            # in order to have a rough estimation of the blocking probability
                            assert est_mc.maxtime == nparticles*est_fv.maxtime, "The simulation time of the MC process and the FV process are comparable"
                            #rate_blocking_time_mc = est_mc.get_all_total_blocking_time() / (est_mc.maxtime * nparticles)
                            #rate_blocking_time_fv = est_fv.get_all_total_blocking_time() / (est_fv.maxtime * nparticles)
                            #print("Blocking time rate MC: {:.3f}%".format(rate_blocking_time_mc*100))
                            #print("Blocking time rate FV: {:.3f}%".format(rate_blocking_time_fv*100))
    
                            df_proba_blocking_estimates = pd.concat([df_proba_blocking_estimates,
                                                                     pd.DataFrame({
                                                                                'K': K,
                                                                                'nparticles': nparticles,
                                                                                'nmeantimes': nmeantimes,
                                                                                'buffer_size_activation': buffer_size_activation,
                                                                                'buffer_size_activation_value': buffer_size_activation_value,
                                                                                'rep': rep+1,
                                                                                'seed': seed_rep,
                                                                                'integral': integral,
                                                                                'E(T)': mean_lifetime_mc,
                                                                                'PMC(K)': proba_blocking_mc,
                                                                                'PFV1(K)': proba_blocking_integral,
                                                                                'PFV2(K)': proba_blocking_laplacian,
                                                                                'Pr(K)': proba_blocking_K,
                                                                                }, index=[(case-1)*replications + rep+1])],
                                                                     axis=0)
                                                                    
                            if proba_blocking_K is not None:
                                print("\t--> PMC(K)={:.6f}% vs. Pr(K)={:.6f}% vs. PFV1(K)={:.6f}% vs. PFV2(K)={:.6f}% vs. Pr(K)={:.6f}%" \
                                      .format(proba_blocking_mc*100, proba_blocking_K*100, proba_blocking_integral*100, proba_blocking_laplacian*100, proba_blocking_K*100))
                            else:
                                print("\t--> PMC(K)={:.6f}% vs. PFV1(K)={:.6f}% vs. PFV2(K)={:.6f}%" \
                                      .format(proba_blocking_mc*100, proba_blocking_integral*100, proba_blocking_laplacian*100))
                            df_proba_survival_and_blocking_conditional = est_fv.estimate_proba_survival_and_blocking_conditional()
                            if rep == 0:
                                # Plot the blue and red curves contributing to the integral used in the FV estimation
                                self.plot_curve_estimates(df_proba_survival_and_blocking_conditional,
                                                  est_mc.queue.getBirthRates(),
                                                  est_mc.queue.getDeathRates(),
                                                  K,
                                                  nparticles,
                                                  nmeantimes,
                                                  buffer_size_activation_value,
                                                  mean_lifetime_mc,
                                                  reactivate,
                                                  finalize_type,
                                                  seed)
    
                        if False:
                            # DM-2020/08/23: Stop at a simulation showing large values of PFV2(K) (e.g. 72% when the true P(K) = 10%!)
                            if K == 5 and nparticles == 10 and True: # niter == 400:
                                import sys
                                sys.exit()
    
                        if nmeantimes < nmeantimes_max:
                            # Give the opportunity to reach nmeantimes_max (in case nmeantimes*nmeantimes_mult goes beyond nmeantimes_max)
                            nmeantimes = np.min([nmeantimes*nmeantimes_mult, nmeantimes_max])
                        else:
                            # We should stop iterating on nmeantimes at the next iteration
                            nmeantimes = nmeantimes * nmeantimes_mult
                    if nparticles < nparticles_max:
                        # Give the opportunity to reach nparticles_max (in case nparticles*nparticles_mult goes beyond nparticles_max)
                        nparticles = np.min([nparticles*nparticles_mult, nparticles_max])
                    else:
                        # We should stop iterating on nparticles at the next iteration
                        nparticles = nparticles * nparticles_mult
                buffer_size_activation_value_prev = buffer_size_activation_value
            if K < K_max:
                # Give the opportunity to reach K_max (in case K + K_mult goes beyond K_max)
                K = np.min([K*K_mult, K_max])
            else:
                # We should stop iterating on K at the next iteration
                K = K * K_mult

        return df_proba_blocking_estimates

    @classmethod
    # Note that a class method can be invoked from an instance as well!
    # Ref: https://stackoverflow.com/questions/30190061/python-why-can-i-call-a-class-method-with-an-instance
    def plot_curve_estimates(cls, df_proba_survival_and_blocking_conditional, *args, log=False):
        birth_rates = args[0]
        death_rates = args[1]
        rhos = [b/d for b, d in zip(birth_rates, death_rates)]
        K = args[2]
        nparticles = args[3]
        nmeantimes = args[4]
        buffer_size_activation = args[5]
        mean_lifetime = args[6]
        reactivate = args[7]
        finalize_type = args[8]
        seed = args[9]
        if log:
            print("arrival rates={}".format(birth_rates))
            print("service rates={}".format(death_rates))
            print("rhos={}".format(rhos))
            print("K={}".format(K))
            print("nparticles={}".format(nparticles))
            print("nmeantimes={}".format(nmeantimes))
            print("buffer_size_activation={}".format(buffer_size_activation))
            print("mean_lifetime={}".format(mean_lifetime))
            print("reactivate={}".format(reactivate))
            print("finalize_type={}".format(finalize_type.name))
            print("seed={}".format(seed))

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
        ax.hlines(0.0, 0, ax.get_xlim()[1], color='gray')
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
        plt.title("K={}, N={}, activation size={}, nmeantimes={}, rhos={}, mean_lifetime={:.1f}" \
                  ", reactivate={}, finalize={}, seed={}" \
                  .format(K, nparticles, buffer_size_activation, nmeantimes,
                          rhos,
                          mean_lifetime or np.nan, reactivate, finalize_type.name[0:3],
                          seed
                          ))
        ax.title.set_fontsize(9)

    @classmethod
    def aggregation_bygroups(cls, df, groupvars, analvars,
                             dict_stats={'n': 'count', 'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max}):
        """
        Computes the given summary statistics in the input data frame on the analysis variables
        by the given group variables.
        
        Arguments:
        groupvars: list
            List of grouping variables.

        analvars: list
            List of analysis variables whose statistics is of interest.
    
        dict_stats: dict
            Dictionary with the summary statistics names and functions to compute them.
            default: {'n': 'count', 'mean': np.mean, 'std': np.std, 'min': 'min', 'max': 'max'}
    
        Return: data frame
        Data frame containing the summary statistics results.
        If the 'std' and 'n' (count) summary statistics is part of the `dict_stats` dictionary,
        the Standard Error (SE) is also computed as 'std' / sqrt('n')
        """
        df_grouped = df.groupby(groupvars, as_index=True, group_keys=False)
        df_agg = df_grouped[analvars].agg(dict_stats)
        stat_names = dict_stats.keys() 
        if 'std' in stat_names and 'n' in stat_names:
            df_agg_se = df_agg['std'][analvars] / np.sqrt(df_agg['n'][analvars])
            # Rename the column names to reflect the column represents the Standard Error (as opposed to the Standard Deviation)
            df_agg_se.set_axis([['SE']*len(analvars), analvars], axis=1, inplace=True)
            df_agg = pd.concat([df_agg, df_agg_se], axis=1)
    
        return df_agg

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
                        yerr_mc = se_mult*df2plot['SE'][y_mc]
                        yerr_fv = se_mult*df2plot['SE'][y_fv]
                        yerr_ref_mc = se_mult*df2plot['SE'][proba_true]       # Referece value are the blocking rates
                        line_mc = ax_mc.errorbar(axis_values, df2plot['mean'][y_mc]*100, yerr=yerr_mc*100,
                                    capsize=4, color=color_red, marker='.')
                        line_fv = ax_fv.errorbar(axis_values, df2plot['mean'][y_fv]*100, yerr=yerr_fv*100,
                                    capsize=4, color=color_green, marker='.')
                        line_ref_mc = ax_mc.errorbar(axis_values, df2plot['mean'][proba_true]*100, yerr=yerr_ref_mc*100,
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
                                                        positions=axis_values, showmedians=False, linewidth=4,
                                                        color_body="red", color_lines="red", color_means="red")            
                            plotting.violinplot(ax_fv,  [df[ind & (df[grp_axis]==x)][y_fv]*100 for x in axis_values],
                                                        positions=axis_values, showmedians=False, linewidth=4,
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
        grp_part = 'nparticles'
        grp_time = 'nmeantimes'
        grp_K = 'K'
        # The grouping variables should be EXACTLY THREE, which define:
        # - the page (e.g. K)
        # - the axis (e.g. nparticles)
        # - the legend (e.g. buffer size)
        groupvars = [grp_K, grp_buffer, grp_part]
        # Analysis variables
        y_mc = 'PMC(K)'
        y_fv = 'PFV1(K)'      # Approximation 1
        y_fv2 = 'PFV2(K)'     # Approximation 2
        proba_true = 'Pr(K)'  # Reference value for comparison of estimated values
        analvars = [y_mc, y_fv, y_fv2, proba_true]
        replications = int(np.max(df['rep']))

        # Analysis by group (mean, std, min, max)
        df_agg = cls.aggregation_bygroups(df, groupvars, analvars)

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
                ax.errorbar(df2plot.loc[K].index, df2plot.loc[K]['mean'][y_mc], yerr=se_mult*df2plot.loc[K]['SE'][y_mc],
                            capsize=4, color='red', marker='.')
                ax.errorbar(df2plot.loc[K].index, df2plot.loc[K]['mean'][y_fv], yerr=se_mult*df2plot.loc[K]['SE'][y_fv],
                            capsize=4, color='green', marker='.')
                ax.errorbar(df2plot.loc[K].index, df2plot.loc[K]['mean'][proba_true], yerr=se_mult*df2plot.loc[K]['SE'][proba_true],
                            capsize=4, color='black', marker='.', linestyle='dashed')
                #ax.axhline(y=P_true, color='black', linestyle='dashed')
                ax.set_xlabel(grp_axis)
                ax.set_ylabel('K = {:.0f}'.format(K))
                ymax = np.max(np.r_[df2plot.loc[K]['mean'][y_mc] + se_mult*df2plot.loc[K]['SE'][y_mc],
                                    df2plot.loc[K]['mean'][y_fv] + se_mult*df2plot.loc[K]['SE'][y_fv],
                                    df2plot.loc[K]['mean'][proba_true] + se_mult*df2plot.loc[K]['SE'][proba_true]])
                ax.set_ylim((0, ymax*1.1))
                ax.legend([legend_label_mc,
                           legend_label_fv,
                           "Blocking rate values"])
                plt.suptitle("P(K) vs. {} in the Fleming Viot system (+/-{}SE, replications={})" \
                             .format(grp_axis.upper(), se_mult, replications))

        # Define the analysis data frame and the name of the variables involved in the plots
        df = results_convergence
        grp_part = 'nparticles'
        grp_time = 'nmeantimes'
        grp_K = 'K'
        y_mc = 'PMC(K)'
        y_fv = 'PFV1(K)'  # Approximation 1
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
        df_agg_byparam = cls.aggregation_bygroups(df4part, [grp_K, grp_part], analvars)
        plot(df_agg_byparam, grp_K, grp_part, legend, se_mult=2)

        all_nparticles = np.unique(df[grp_part])
        nparticles_max = int(all_nparticles[-1])
        legend = "{} = {}".format(grp_part, nparticles_max)
        df4iter = df[ df[grp_part] == nparticles_max ]
        df_byiter = cls.aggregation_bygroups(df4iter, [grp_K, grp_time], analvars)
        plot(df_byiter, grp_K, grp_time, legend, se_mult=2)

    @classmethod
    def plot_errors(cls, results_convergence, x, subset=None, widths=0.1,
                    grp_K="K", y_mc="PMC(K)", y_fv = "PFV1(K)", y_true="Pr(K)", rep="rep"):
        """
        Plots the distribution of estimation errors for the MC and the FV methods
        against a variable of interest, x.

        Arguments:
        results_convergence: pandas DataFrame
            DataFrame with the results of the simulation which should contain
            at least the following columns:
            - `grp_K`: variable with the capacity of the queue
            - `x`: variable of interest to plot on the X-axis
            - `y_mc`: estimate of `y_true` by Monte-Carlo
            - `y_fv`: estimate of `y_true` by Fleming-Viot
            - `rep`: index identifying the simulation replication

        Return: pandas DataFrame
            Input DataFrame with 4 new columns:
            - "error_mc": absolute error of the MC estimation
            - "error_fv": absolute error of the FV estimation
            - "error_rel_mc": relative error of the MC estimation
            - "error_rel_fv": relative error of the FV estimation
        """

        # Define the analysis data frame and the name of the variables involved in the plots
        df = copy.deepcopy(results_convergence)

        replications = int(np.max(df[rep]))

        K_values = df[grp_K].value_counts(sort=False).index
        df["error_mc"] = df[y_mc] - df[y_true]
        df["error_fv"] = df[y_fv] - df[y_true]
        df["error_rel_mc"] = (df[y_mc] - df[y_true]) / df[y_true]
        df["error_rel_fv"] = (df[y_fv] - df[y_true]) / df[y_true]
        #print(df[[grp_K, x, y_mc, y_fv, "error_rel_mc", "error_rel_fv"]])

        # Columns to plot
        yvar2plot_mc = "error_rel_mc"
        yvar2plot_fv = "error_rel_fv"

        # Rows to plot
        if subset is not None:
            df2plot = df[subset]

        # Compute the max absolute values of all Y's to plot, so we can visually compare the plots
        yabsmax = np.nanmax(np.r_[ np.abs(df[yvar2plot_mc]), np.abs(df[yvar2plot_fv])] )*100
        # The above max value may be too large, so we set the vertical scale to -100% -- +100%
        yabsmax = 1.0*100
        for K in K_values:
            ind = (df2plot[grp_K] == K)
            axis_values = df2plot[ind][x].value_counts(sort=False).index
            #print("K={:.0f}: {}".format(K, axis_values))
            (fig, axes) = plt.subplots(1, 2, figsize=(12,4))
            ax1 = axes[0]
            ax2 = axes[1]
            #ax = plt.gca()
            #plt.plot([axis_values[0], axis_values[-1]], [0.0, 0.0], color="gray")
            # We create a violin plot for each set of not NaN errors (MC & FV), one set per x value
            plotting.violinplot(ax1,  [df2plot[ind & (df2plot[x]==x_value)][yvar2plot_mc].dropna()*100 for x_value in axis_values],
                                        positions=axis_values, showmeans=True, showmedians=False, linewidth=2, widths=widths,
                                        color_body="red", color_lines="red", color_means="red")
            plotting.violinplot(ax2,  [df2plot[ind & (df2plot[x]==x_value)][yvar2plot_fv].dropna()*100 for x_value in axis_values],
                                        positions=axis_values, showmeans=True, showmedians=False, linewidth=2, widths=widths,
                                        color_body="green", color_lines="green", color_means="green")
            for ax in axes:
                ax.set_ylim((-yabsmax*1.1, yabsmax*1.1))
                ax.set_xlabel(x)
                ax.set_ylabel("K={:.0f} -- Relative Error (%)".format(K))
                ax.axhline(0.0, color="gray")
            plt.suptitle("Error distribution of blocking probability estimation Pr(K={:.0f}) on {} replications" \
                     .format(K, replications) +
                     "\nMonte Carlo (red) vs. Fleming Viot (green)")

        return df

# DM-2020/12/23: To change which portion of the below code to run, change the IF condition
# to `== "__main__"` or to `!= "__main__"` accordingly, taking into account that when running
# this file as a script (F5) __name__ is equal to "__main__".
if __name__ != "__main__":
    run_unit_tests = True
    if run_unit_tests:
        #suite = unittest.TestSuite()
        #suite.addTest(Test_QB_Particles("test_fv_implementation"))
        #runner = unittest.TextTestRunner()
        #runner.run(suite)

        results, results_agg = Test_QB_Particles.test_fv_implementation()
    else:
        # DM-2020/08/24: Instead of using unittest.main(), use the following to test the FV system
        # because there is a very weird error generated by the fact that queue.size inside
        # the EstimatorQueueBlockingFlemingViot class is interpreted of being of class
        # unittest.runner.TextTestResult (or similar)!! instead of simply returning the size
        # attribute of the `queue` object!!!
        test = Test_QB_Particles()
        
        time_start = timer()
        buffer_size_activation = 1
        #buffer_size_activation = int(test.capacity/2)
        finalize_type = FinalizeType.ABSORB_CENSORED
        nparticles = 20
        nmeantimes = 50
        seed = 1717
        plotFlag = True
        log = False
        est_mc,     proba_blocking_mc, \
                    total_blocking_time, \
                    total_survival_time, \
                    expected_survival_time, \
                    params_mc = test.run_simulation(   buffer_size_activation=buffer_size_activation,
                                            mean_lifetime=None,
                                            reactivate=False,
                                            finalize_type=FinalizeType.REMOVE_CENSORED,
                                            nparticles=1,
                                            nmeantimes=nparticles*nmeantimes,
                                            seed=seed,
                                            plotFlag=plotFlag,
                                            log=log)
        est_fv,     proba_blocking_integral_fv, \
                    proba_blocking_laplacian_fv, \
                    integral_fv, \
                    gamma_fv, \
                    params_fv = test.run_simulation(   buffer_size_activation=buffer_size_activation,
                                            mean_lifetime=expected_survival_time,
                                            reactivate=True,
                                            finalize_type=finalize_type,
                                            nparticles=nparticles,
                                            nmeantimes=nmeantimes,
                                            seed=seed,
                                            plotFlag=plotFlag,
                                            log=log)
    
        print("\n**** Summary of estimation ****")
        assert est_mc.maxtime == nparticles*est_fv.maxtime, "The simulation times of the MC and the FV process are comparable"
        #blocking_time = est_mc.get_all_total_blocking_time()
        #survival_time = est_mc.get_all_total_survival_time()
        #rate_blocking_time_mc = blocking_time / survival_time
        #print("Blocking time (MC --slow method): {:.3f}%".format(rate_blocking_time_mc*100))
        print("P(K) by MC: {:.3f}%".format(proba_blocking_mc*100))
        print("P(K) estimated by FV1: {:.6f}%".format(proba_blocking_integral_fv*100))
        print("P(K) estimated by FV2: {:.6f}%".format(proba_blocking_laplacian_fv*100))
        proba_blocking_true = compute_blocking_probability_birth_death_process(test.nservers, test.capacity, test.rhos)
        if proba_blocking_true is not None:
            print("True P(K): {:.6f}%".format(proba_blocking_true*100))
    
        time_end = timer()
        print("Execution time: {:.1f} min".format((time_end - time_start) / 60))
else:
    # Lines to execute "by hand" (i.e. at the Python prompt)
    test = Test_QB_Particles()

    time_start = timer()
    results_convergence = test.analyze_convergence(
                                    finalize_type=FinalizeType.ABSORB_CENSORED,
                                    replications=8,
                                    K_range=(5, 40, 2), 
                                    buffer_size_activation_values=[1, 0.2, 0.3, 0.4, 0.5],
                                    nparticles_range=(400, 400, 2),
                                    nmeantimes_range=(50, 50, 2),
                                    seed=1717,
                                    log=False)
    time_end = timer()

    print("Top and bottom 5 records in the results data frame:")
    print(results_convergence.head())
    print(results_convergence.tail())

    print("Execution time: {:.1f} min".format((time_end - time_start) / 60))

    # Aggregate results
    results_convergence_agg = test.aggregation_bygroups(results_convergence,
                                                        ["K", "buffer_size_activation", "nparticles"],
                                                        ["PMC(K)", "PFV1(K)", "Pr(K)"])

    # Plots
    nparticles_max = 400
    results_convergence_errors = test.plot_errors(results_convergence, "buffer_size_activation", subset=results_convergence["nparticles"]==nparticles_max, widths=0.05)

    #results_convergence_agg = test.plot_aggregated_convergence_results(results_convergence)
    #test.plot_aggregated_convergence_results_maxvalues(results_convergence)

    # OLD filename save
    #filename = "../../RL-002-QueueBlocking/results/fv_approx1_convergence_rho={:.3f}.csv"
    #results_convergence.to_csv(filename.format(rho))
    #print("Results of simulation saved to {}".format(filename))  

    #filename = "../../RL-002-QueueBlocking/results/RL-QB-20210205-SimulationMultiServer3-JobClass2-K=5&10&20,N=100&200&400&800,T=20&40.csv"
    #filename = "../../RL-002-QueueBlocking/results/RL-QB-20210211-SimulationSingleServer-K=5&10,N=100&200,T=20&40.csv"
    #filename = "../../RL-002-QueueBlocking/results/RL-QB-20210301-SimulationSingleServer-K=5&10&20&40,B=1&0.2&0.4&0.6,N=200&400,T=30.csv"
    #filename = "../../RL-002-QueueBlocking/results/RL-QB-20210301-SimulationSingleServer-K=5&10&20&40,B=1&0.2&0.4&0.6,N=400,T=50.csv"
    filename = "../../RL-002-QueueBlocking/results/RL-QB-20210301-SimulationSingleServer-K=40,B=0.2&0.25&0.3&0.35&0.4,N=400,T=50.csv"
    #results_convergence.to_csv(filename)
    #print("Results of simulation saved to {}".format(os.path.abspath(filename)))

    #filename = "../../RL-002-QueueBlocking/results/RL-QB-20210205-SimulationMultiServer3-JobClass2-K=5&10&20,N=100&200&400&800,T=20&40 (AGG).csv"
    #filename = "../../RL-002-QueueBlocking/results/RL-QB-20210211-SimulationSingleServer-K=5&10,N=100&200,T=20&40 (AGG).csv"
    #filename = "../../RL-002-QueueBlocking/results/RL-QB-20210301-SimulationSingleServer-K=5&10&20&40,B=1&0.2&0.4&0.6,N=200&400,T=30 (AGG).csv"
    #filename = "../../RL-002-QueueBlocking/results/RL-QB-20210301-SimulationSingleServer-K=5&10&20&40,B=1&0.2&0.4&0.6,N=400,T=50 (AGG).csv"
    filename = "../../RL-002-QueueBlocking/results/RL-QB-20210301-SimulationSingleServer-K=40,B=0.2&0.25&0.3&0.35&0.4,N=400,T=50 (AGG).csv"
    #results_convergence_agg.to_csv(filename)
    #print("Results of simulation saved to {}".format(os.path.abspath(filename)))
