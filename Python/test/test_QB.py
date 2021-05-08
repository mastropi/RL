# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:16:02 2020

@author: Daniel Mastropietro
@description: Estimators are defined for problems of interest in Reinforcement Learning.
"""

import runpy
runpy.run_path('../../setup.py')

import os
import sys
import numpy as np
import pandas as pd
import copy

from timeit import default_timer as timer
from datetime import datetime
import unittest
from unittest_data_provider import data_provider
import matplotlib
from matplotlib import pyplot as plt, cm, ticker as mtick
import Python.lib.utils.plotting  as plotting

import Python.lib.queues as queues
import Python.lib.estimators as estimators
from Python.lib.queues import Event
from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses
from Python.lib.estimators import EventType, FinalizeType, FinalizeCondition, plot_curve_estimates
from Python.lib.utils.computing import compute_blocking_probability_birth_death_process

#from importlib import reload
#reload(estimators)
#from Python.lib.estimators import EstimatorQueueBlockingFlemingViot

#import test_utils


class Test_QB_Particles(unittest.TestCase):

    def __init__(self, nservers=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = False
        self.capacity = 5
        self.rate_birth = 0.7
        self.nservers = nservers

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
        else:
            raise ValueError("Given Number of servers ({}) is invalid. Valid values are: 1, 3".format(nservers))
        # rho rates for each server based on arrival rates and assignment probabilities
        self.rate_birth, self.rhos = self.compute_rhos()

        self.queue = queues.QueueMM(self.rate_birth, self.rate_death, self.nservers, self.capacity)

        self.plotFlag = True

    def compute_rhos(self):
        "Computes the rho rates for each server based on arrival rates, service rates, and assignment probabilities"
        R = self.nservers
        J = len(self.job_rates) # Number of job classes
        lambdas = [0]*self.nservers
        rhos = [0]*self.nservers
        for r in range(R):
            for c in range(J):
                lambdas[r] += self.policy[c][r] * self.job_rates[c]
            rhos[r] = lambdas[r] / self.rate_death[r]

        return lambdas, rhos

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
        proba_blocking_K = compute_blocking_probability_birth_death_process(self.rhos, K)
        if proba_blocking_K is not None:
            print("Theoretical value: {:.1f}%".format(proba_blocking_K*100))

        print("Simulation setup:")
        print(est.setup())

        return  df_proba_survival_and_blocking_conditional, \
                (reactivate, finalize_type, nparticles, nmeantimes)

    @classmethod
    def test_fv_implementation(cls, nservers=1, K=5, buffer_size_activation=1):
        "2021/04/19: Analyze convergence of the FV algorithm as number of particles N increases"

        #--- Test one server
        rate_birth = 0.7
        if nservers == 1:
            job_rates = [rate_birth]
            rate_death = [1]
            policy = [[1]]
            queue = queues.QueueMM(rate_birth, rate_death, nservers, K)
            # Queue environment (to pass to the simulation functions)
            env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_rates=job_rates, rewards=[1], policy_assign=policy)
        elif nservers == 3:
            job_rates = [0.8, 0.7]
            rate_death = [1, 1, 1]
            policy = [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
            queue = queues.QueueMM(rate_birth, rate_death, nservers, K)
            # Queue environment (to pass to the simulation functions)
            env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_rates=job_rates, rewards=[1, 1], policy_assign=policy)
        else:
            raise ValueError("Given Number of servers ({}) is invalid. Valid values are: 1, 3".format(nservers))
            
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
        nmeantimes = 50
        seed = 1717

        # Info parameters 
        dict_params_info = {'plot': True, 'log': False}

        replications = 8
        nparticles_min = 800
        nparticles_max = 1600
        nparticles_step_prop = 1  # STEP proportion: N(n+1) = (1 + prop)*N(n), so that we scale the step as the number of particles increases
        nparticles = nparticles_min
        df_results = pd.DataFrame(columns=['N', 'replication', 'Pr(MC)', 'Time(MC)', 'E(T)', 'Pr(FV)', 'Time(FV)', 'Pr(K)', 'exec_time(s)'])
        case = 0
        ncases = int( np.log(nparticles_max / nparticles_min) / np.log(1 + nparticles_step_prop)) + 1
        print("System: # servers={}, K={}, rhos={}, buffer_size_activation={}".format(nservers, K, env_queue.getIntensities(), buffer_size_activation_value))
        time_start_all = timer()
        while nparticles <= nparticles_max:
            case += 1 
            print("\n*** Running simulation for nparticles={} ({} of {}) on {} replications...".format(nparticles, case, ncases, replications))

            dict_params_simul = {
                'nparticles': nparticles,
                'nmeantimes': nmeantimes,
                'buffer_size_activation': buffer_size_activation_value,
                'multiplier_for_extended_simulation_time': 1, #10 * (K/10),
                'multiplier_adjust_for_activation': False,
                'seed': seed,
                    }

            for r in range(1, replications+1):
                print("\n\n\n\n\tReplication {} of {}...".format(r, replications))
                dict_params_simul['seed'] = seed + r - 1

                time_start = timer()
                print("\t--> Running Monte-Carlo estimation...")
                proba_blocking_mc, est_mc = estimators.estimate_blocking_mc(env_queue, dict_params_simul, dict_params_info=dict_params_info)

                print("\n\t--> Running Fleming-Viot estimation...")
                proba_blocking_fv, integral, expected_survival_time, \
                    n_survival_curve_observations, n_survival_time_observations, \
                        est_fv, est_abs, est_surv = estimators.estimate_blocking_fv(env_queue, dict_params_simul, dict_params_info=dict_params_info)
                time_end = timer()
                exec_time = time_end - time_start
                print("execution time MC + FV: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))

                if nparticles == nparticles_min and r == 1:
                    rhos = est_mc.rhos
                    print("Computing TRUE blocking probability for nservers={}, K={}, rhos={}...".format(nservers, K, rhos))
                    proba_blocking_true = compute_blocking_probability_birth_death_process(rhos, K)

                # Results
                assert est_mc.maxtime - est_fv.maxtime*nparticles >= -0.001*est_fv.maxtime, \
                    "The simulation time of the MC ({:.1f}) is longer than that of FV ({:.1f})" \
                    .format(est_mc.maxtime, est_fv.maxtime*nparticles)
                print("\tP(K) by MC: {:.6f}% (simulation time = {:.1f})".format(proba_blocking_mc*100, est_mc.maxtime))
                print("\tP(K) estimated by FV: {:.6f}%, E(T) = {:.1f} (simulation time = {:.1f})".format(proba_blocking_fv*100, est_fv.mean_lifetime, est_fv.maxtime))
                print("\tTrue P(K): {:.6f}%".format(proba_blocking_true*100))

                # Store the results
                df_append = pd.DataFrame([[nparticles, r, proba_blocking_mc, est_mc.maxtime, est_fv.mean_lifetime, proba_blocking_fv, est_fv.maxtime, proba_blocking_true, exec_time]],
                                         columns=df_results.columns, index=[case])
                df_results = df_results.append(df_append)

            print("Results:")
            print(df_results)
            nparticles += int( nparticles_step_prop*nparticles )
        time_end_all = timer()

        print("Total execution time: {:.1f} min".format((time_end_all - time_start_all) / 60))
        title = "Simulation results for #servers={}, K={}, rhos={}, ({}<=N<={}), T<={:.0f}, Rep={}".format(nservers, K, rhos, nparticles_min, nparticles_max, est_fv.maxtime, replications)
        print(title)
        print("Raw results by N:")
        print(df_results)
        
        df_results_agg_by_N = aggregation_bygroups(df_results, ['N'], ['Pr(MC)', 'Pr(FV)'])
        print("Aggregated results by N:")
        print(df_results_agg_by_N)

        plt.figure()
        legend_lines_mc = []
        legend_lines_fv = []
        legend_lines_ref = []
        ax = plt.gca()
        plt.plot(df_results['N'], df_results['Pr(MC)']*100, 'r.', markersize=2)
        line_mc = plt.errorbar(list(df_results_agg_by_N.index), df_results_agg_by_N['mean']['Pr(MC)']*100, yerr=2*df_results_agg_by_N['SE']['Pr(MC)']*100, capsize=4, color='red', marker='x')
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
        plt.title(title, fontsize=10)
        #ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xlabel("N (number of particles)")
        ax.set_ylabel("Blocking probability (%)")
        ax.legend(legend_lines_mc + legend_lines_fv + legend_lines_ref, ['MC +/- 2SE', 'FV +/- 2SE', 'True'], fontsize='x-small')
    
        # Violin plots
        (ax_mc, ax_fv) = plt.figure(figsize=(8,4)).subplots(1,2)
        N_values = np.unique(df_results['N'])
        violin_widths = (N_values[-1] - N_values[0]) / 10
        plotting.violinplot(ax_mc,  [df_results[ df_results['N']==x ]['Pr(MC)']*100 for x in N_values],
                                    positions=N_values, showmeans=True, showmedians=False, linewidth=2, widths=violin_widths,
                                    color_body="red", color_lines="red", color_means="red")
        plotting.violinplot(ax_fv,  [df_results[ df_results['N']==x ]['Pr(FV)']*100 for x in N_values],
                                    positions=N_values, showmeans=True, showmedians=False, linewidth=2, widths=violin_widths,
                                    color_body="green", color_lines="green", color_means="green")            
        # Add the observed points
        # (THIS DOES NOT WORK, AS NOW POINTS ARE ADDED! However, note that the violin plots extend to the min and max values in the data, because showextrema=True by default in the violinplot() call)
        ax_mc.plot(df_results['N'], df_results['Pr(MC)']*100, 'r.', markersize=2)
        ax_fv.plot(df_results['N'], df_results['Pr(FV)']*100, 'g.', markersize=2)
        
        ax_mc.hlines(df_results.iloc[0]['Pr(K)']*100, df_results.iloc[0]['N'], df_results.iloc[-1]['N'], color='gray', linestyles='dashed')
        ax_fv.hlines(df_results.iloc[0]['Pr(K)']*100, df_results.iloc[0]['N'], df_results.iloc[-1]['N'], color='gray', linestyles='dashed')
        plt.suptitle(title, fontsize=10)
        # Set a common vertical axis
        ymax = max([ax_mc.get_ylim()[1], ax_fv.get_ylim()[1]])
        ax_mc.set_ylim([0, ymax])
        ax_mc.set_xlabel("N (number of particles)")
        ax_mc.set_ylabel("Blocking probability (%)")
        ax_fv.set_ylim([0, ymax])
        ax_fv.set_xlabel("N (number of particles)")

        return df_results, df_results_agg_by_N, est_mc, est_fv, est_abs, est_surv

    def analyze_estimates(self,
                                    replications=5,
                                    K_values=[10, 20, 30, 40],
                                    nparticles_values=[200, 400, 800, 1600],
                                    nmeantimes_values=[50, 50, 50, 50],
                                    multiplier_values=[10, 15, 20, 25],
                                    multiplier_adjust_with_activation=False,
                                    buffer_size_activation_values=[1],
                                    seed=1717,
                                    dict_params_out=None,
                                    dict_params_info={'plot': True, 'log': False}):
        assert len(nparticles_values) == len(K_values), "The number of values in the nparticles parameter is the same as in K_values."
        assert len(nmeantimes_values) == len(K_values), "The number of values in the nmeantimes parameter is the same as in K_values."
        assert len(multiplier_values) == len(K_values), "The number of values in the multiplier parameter is the same as in K_values."
    
        # Upper bound for the number of simulations to run
        # (it's an upper bound because when buffer_size_activation is a proportion, there may be repeated values for the actual buffer_size_activation used) 
        nsimul = int(   len(K_values) * \
                        len(buffer_size_activation_values))
    
        df_proba_blocking_estimates = pd.DataFrame.from_items([
                                                               ('rhos', []),
                                                               ('K', []),
                                                               ('nparticles', []),
                                                               ('nmeantimes', []),
                                                               ('multiplier', []),
                                                               ('buffer_size_activation', []),
                                                               ('buffer_size_activation_value', []),
                                                               ('rep', []),
                                                               ('seed', []),
                                                               ('integral', []),
                                                               ('E(T)', []),
                                                               ('PMC(K)', []),
                                                               ('PFV(K)', []),
                                                               ('Pr(K)', []),
                                                               ('maxtime(MC)', []),
                                                               ('maxtime(FV)', []),
                                                               ('ratio_maxtime_mc_fv', []),
                                                               ])
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
            multiplier0 = multiplier_values[idx_K]
            print("\n\n---> NEW K (Queue's capacity = {})".format(queue.getCapacity()))
            print("---> (nparticles={}, nmeantimes={}, multiplier for extended simulation ONE particle={}, adjust multiplier={})" \
                  .format(nparticles, nmeantimes_values, multiplier_values, multiplier_adjust_with_activation))
    
            print("Computing TRUE blocking probability...", end=" --> ")
            time_pr_start = timer()
            proba_blocking_true = compute_blocking_probability_birth_death_process(self.rhos, K)
            time_pr_end = timer()
            print("{:.1f} sec".format(time_pr_end - time_pr_start))
            print("Pr(K)={:.6f}%".format(proba_blocking_true*100))

            # Create the queue environment that is simulated below
            env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_rates=self.job_rates, rewards=[1]*len(self.job_rates), policy_assign=self.policy)
                ## The rewards are not used at this point, so I just set them to 1 for all job classes.

            buffer_size_activation_value_prev = None
            for buffer_size_activation in buffer_size_activation_values:
                # When the buffer size for activation parameter is smaller than 1 it is considered a proportion of the queue's capacity
                if buffer_size_activation < 1:
                    buffer_size_activation_value = max(1, int( round( buffer_size_activation * K ) ))
                else:
                    buffer_size_activation_value = buffer_size_activation
                    # Convert the buffer size activatoin into a proportion so that it appears
                    # in the right place in the PLOT w.r.t. the other buffer_size_activation proportions
                    # (and not e.g. to the RIGHT of all other proportion values,
                    # which makes interpretatoin of the plot more difficult)
                    buffer_size_activation = buffer_size_activation_value / K
                # Do not repeat the previous buffer size activation value (which may happen when the parameter is given as a proportion)
                if buffer_size_activation_value == buffer_size_activation_value_prev:
                    continue
                print("\n\t---> NEW BUFFER SIZE({})".format(buffer_size_activation_value))

                if multiplier_adjust_with_activation:
                    # Make simulation time of single particle longer when the buffer size for activation is larger
                    # so that we get better estimates of the survival curve and the expected survival time
                    # (while guaranteeing a minimum of multiplier = 1)
                    # e.g. when BSA = 0.5*K, we set multiplier = 5 * "original multiplier"
                    multiplier = max([ 1, multiplier0  *  10 * buffer_size_activation_value / K ] )
                else:
                    multiplier = multiplier0

                case += 1
                print("******************!!!!!!! Simulation {} of {} !!!!!!*****************\n\tK={}, nparticles={}, nmeantimes={}, multiplier={}, buffer_size_activation={}" \
                      .format(case, nsimul, K, nparticles, nmeantimes, multiplier, buffer_size_activation_value))
                for rep in range(replications):
                    # NOTE THAT THE FIRST REPLICATION (rep=0) HAS THE SAME SEED FOR ALL PARAMETER SETTINGS
                    # This is nice because we can a little bit better compare the effect of the different parameter settings
                    # (but not so much anyway, because the values that are generated as the parameter settings change
                    # impact whatever is generated next --e.g. the initial events for particle 2
                    # will change if more events are generated for particle 1 when the simulation time increases...)
                    print("\n\t\t### Replication {} of {} ###".format(rep+1, replications))
                    seed_rep = seed + rep
    
                    dict_params_simul = {
                        'nparticles': nparticles,
                        'nmeantimes': nmeantimes,
                        'buffer_size_activation': buffer_size_activation_value,
                        'multiplier_for_extended_simulation_time': multiplier,
                        'seed': seed_rep,
                            }

                    print("\t\t*** MONTE-CARLO ESTIMATION ***")
                    proba_blocking_mc, est_mc = estimators.estimate_blocking_mc(env_queue, dict_params_simul, dict_params_info=dict_params_info)
    
                    print("\t\t*** FLEMING-VIOT ESTIMATION ***")
                    proba_blocking_fv, integral, expected_survival_time, \
                        n_survival_curve_observations, n_survival_time_observations, \
                            est_fv, est_abs, est_surv = estimators.estimate_blocking_fv(env_queue, dict_params_simul, dict_params_info=dict_params_info)

                    print("\t\tP(K) by MC: {:.6f}%".format(proba_blocking_mc*100))
                    print("\t\tP(K) estimated by FV (E(T)={:.1f}): {:.6f}%".format(expected_survival_time, proba_blocking_fv*100))
                    print("\t\tTrue P(K): {:.6f}%".format(proba_blocking_true*100))

                    # Check the results are comparable
                    assert est_mc.maxtime - nparticles*est_fv.maxtime > -0.001*est_fv.maxtime, \
                        "The simulation time of the MC process ({:.1f}) is equal or longer than N times the simulation time of the FV process ({:.1f})" \
                        .format(est_mc.maxtime, nparticles*est_fv.maxtime)

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
                                                                ('nparticles', [nparticles]),
                                                                ('nmeantimes', [nmeantimes]),
                                                                ('multiplier', [multiplier]),
                                                                ('buffer_size_activation', [buffer_size_activation]),
                                                                ('buffer_size_activation_value', [buffer_size_activation_value]),
                                                                ('rep', [rep+1]),
                                                                ('seed', [seed_rep]),
                                                                ('integral', [integral]),
                                                                ('E(T)', [expected_survival_time]),
                                                                ('PMC(K)', [proba_blocking_mc]),
                                                                ('PFV(K)', [proba_blocking_fv]),
                                                                ('Pr(K)', [proba_blocking_true]),
                                                                ('maxtime(MC)', [est_mc.maxtime]),
                                                                ('maxtime(FV)', [est_fv.maxtime]),
                                                                ('ratio_maxtime_mc_fv', [est_mc.maxtime / (nparticles*est_fv.maxtime)]),
                                                            ]) #, orient='columns')
                    df_new_estimates.set_index( pd.Index([(case-1)*replications + rep+1]), inplace=True )
                    df_proba_blocking_estimates = pd.concat([df_proba_blocking_estimates,
                                                             df_new_estimates],
                                                             axis=0)

                    df_proba_survival_and_blocking_conditional = est_fv.estimate_proba_survival_and_blocking_conditional()

                    # Plot the blue and red curves contributing to the integral used in the FV estimation
                    if rep <= 2:
                        plot_curve_estimates(df_proba_survival_and_blocking_conditional,
                                             dict_params={
                                                'birth_rates': est_fv.queue.getBirthRates(),
                                                'death_rates': est_fv.queue.getDeathRates(),
                                                'K': est_fv.queue.getCapacity(),
                                                'nparticles': dict_params_simul['nparticles'],
                                                'nmeantimes': dict_params_simul['nmeantimes'],
                                                'maxtime_mc': est_mc.maxtime,
                                                'maxtime_fv': est_fv.maxtime,
                                                'buffer_size_activation': buffer_size_activation_value,
                                                'mean_lifetime': expected_survival_time,
                                                'n_survival_curve_observations': n_survival_curve_observations,
                                                'n_survival_time_observations': n_survival_time_observations,
                                                'proba_blocking_fv': proba_blocking_fv,
                                                'finalize_type': est_fv.getFinalizeType(),
                                                'seed': seed
                                                })
                buffer_size_activation_value_prev = buffer_size_activation_value

            # Show the results obtained for the current K
            print("Simulation results for #servers={}, rhos={}, K={}, N={}, T={:.1f} ({}x), multiplier={}" \
                  .format(self.nservers, self.rhos, K, nparticles, est_fv.maxtime, nparticles, nmeantimes, multiplier))
            print(df_proba_blocking_estimates)

            if dict_params_info['plot']:
                # Estimates themselves
                plot_estimates(df_proba_blocking_estimates, "buffer_size_activation_value", widths=0.5, subset=df_proba_blocking_estimates["K"]==K)
                plot_estimates(df_proba_blocking_estimates, "buffer_size_activation", widths=0.05, subset=df_proba_blocking_estimates["K"]==K)
                # Errors
                df_proba_blocking_estimates_with_errors = compute_errors(df_proba_blocking_estimates)
                plot_errors(df_proba_blocking_estimates_with_errors, "buffer_size_activation_value", widths=0.5, subset=df_proba_blocking_estimates["K"]==K)
                plot_errors(df_proba_blocking_estimates_with_errors, "buffer_size_activation", widths=0.05, subset=df_proba_blocking_estimates["K"]==K)

        time_end = timer()
        time_elapsed = time_end - time_start
        print("Execution time: {:.1f} sec, {:.1f} min, {:.1f} hours".format(time_elapsed, time_elapsed / 60, time_elapsed / 3600))

        print("Top and bottom 5 records in the results data frame:")
        print(df_proba_blocking_estimates.head())
        print(df_proba_blocking_estimates.tail())

        # Aggregate results
        df_proba_blocking_estimates_agg = aggregation_bygroups(df_proba_blocking_estimates,
                                                               ["K", "buffer_size_activation", "nparticles"],
                                                               ["PMC(K)", "PFV(K)", "Pr(K)"])

        df_proba_blocking_estimates.to_csv(resultsfile)
        print("Results of simulation saved to {}".format(os.path.abspath(resultsfile)))

        df_proba_blocking_estimates_agg.to_csv(resultsfile_agg)
        print("Aggregated results of simulation saved to {}".format(os.path.abspath(resultsfile_agg)))

        return df_proba_blocking_estimates, df_proba_blocking_estimates_agg

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
        y_fv = 'PFV(K)'      # Approximation 1
        y_fv2 = 'PFV2(K)'     # Approximation 2
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
        y_fv = 'PFV(K)'  # Approximation 1
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

def aggregation_bygroups(df, groupvars, analvars,
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

def compute_errors(df_results,
                   y_mc="PMC(K)", y_fv="PFV(K)", y_true="Pr(K)"):
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

def plot_estimates(df_results, x, subset=None, widths=0.1,
                   grp_K="K", y_mc="PMC(K)", y_fv="PFV(K)", y_true="Pr(K)", rep="rep"):
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
        Ex: df_results["nparticles"]==400

    Return: pandas DataFrame
        DataFrame containing the observations in the input data frame used for plotting.
    """
    # Rows to plot
    if subset is not None:
        df2plot = df_results[subset]
    else:
        df2plot = df_results

    # Get the values of the group variable on which we generate a SEPARATE plot
    # by computing its frequency distribution
    K_values = df2plot[grp_K].value_counts(sort=False).index
    replications = int(np.max(df2plot[rep]))    # For informational purposes
    for K in K_values:
        ind = (df2plot[grp_K] == K)
        # Compute the frequency of occurrence of the x values
        # so that we get the values from the index of the frequency table
        x_values = df2plot[ind][x].value_counts(sort=False).index
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
        ymax = np.max([ np.max(np.r_[y1, y2]), proba_blocking_K])
        for ax in axes:
            ax.set_ylim([0, ymax])
            ax.set_xlabel(x)
            ax.set_ylabel("K={:.0f} Estimated blocking probability (%)".format(K))
            ax.axhline(proba_blocking_K*100, color="gray")
        plt.suptitle("Distribution of blocking probability estimates of Pr(K={:.0f}) = {:.6f}% on {} replications" \
                 .format(K, proba_blocking_K*100, replications) +
                 "\nMonte Carlo (red) vs. Fleming Viot (green)")

    return df2plot

def plot_errors(df_results, x, subset=None, widths=0.1,
                grp_K="K", error_mc="error_rel_mc", error_fv="error_rel_fv", rep="rep"):
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
        Ex: df_results["nparticles"]==400

    widths: float
        `widths` parameter of the pyplot.violinplot() function that defines the width of each violin.
        This should be set to something that makes the violin plots visible w.r.t. to the scale of the x axis.

    Return: pandas DataFrame
        DataFrame containing the observations in the input data frame used for plotting.
    """
    # Rows to plot
    if subset is not None:
        df2plot = df_results[subset]
    else:
        df2plot = df_results

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
            ax.set_xlabel(x)
            ax.set_ylabel("K={:.0f} -- Relative Error (%)".format(K))
            ax.axhline(0.0, color="gray")
        plt.suptitle("Error distribution of blocking probability estimation Pr(K={:.0f}) on {} replications" \
                 .format(K, replications) +
                 "\nMonte Carlo (red) vs. Fleming Viot (green)")

    return df2plot

def createLogFileHandleAndResultsFileNames(path="../../RL-002-QueueBlocking", prefix="run"):
    """
    Redirects the standard output to a file which is used to log messages.
    Creates output filenames for raw results and aggregated results.

    Ref: https://www.stackabuse.com/writing-to-a-file-with-pythons-print-function/
    """

    dt_start = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    dt_suffix = datetime.today().strftime("%Y%m%d_%H%M%S")
    logfile = "{}/logs/{}_{}.log".format(path, prefix, dt_suffix)
    resultsfile = "{}/results/{}_{}_results.csv".format(path, prefix, dt_suffix)
    resultsfile_agg = "{}/results/{}_{}_results_agg.csv".format(path, prefix, dt_suffix)

    fh_log = open(logfile, "w")
    print("Log file '{}' has been open for output.".format(logfile))
    print("Started at: {}".format(dt_start))
    stdout_sys = sys.stdout
    sys.stdout = fh_log

    print("Started at: {}".format(dt_start))

    return dt_start, stdout_sys, fh_log, logfile, resultsfile, resultsfile_agg

def closeLogFile(fh_log, stdout_sys, dt_start):
    dt_end = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    print("Ended at: {}".format(dt_end))
    datetime_diff = datetime.strptime(dt_end, "%Y-%m-%d %H:%M:%S") - datetime.strptime(dt_start, "%Y-%m-%d %H:%M:%S")
    time_elapsed = datetime_diff.days*86400 + datetime_diff.seconds
    print("Execution time: {:.1f} min, {:.1f} hours".format(time_elapsed / 60, time_elapsed / 3600))

    fh_log.close()

    # Reset the standard output
    sys.stdout = stdout_sys
    print("Ended at: {}".format(dt_end))
    print("Execution time: {:.1f} min, {:.1f} hours".format(time_elapsed / 60, time_elapsed / 3600))


# DM-2020/12/23: To change which portion of the below code to run, change the IF condition
# to `== "__main__"` or to `!= "__main__"` accordingly, taking into account that when running
# this file as a script (F5) __name__ is equal to "__main__".
if __name__ == "__main__":
    run_unit_tests = True
    if run_unit_tests:
        #suite = unittest.TestSuite()
        #suite.addTest(Test_QB_Particles("test_fv_implementation"))
        #runner = unittest.TextTestRunner()
        #runner.run(suite)

        dt_start, stdout_sys, fh_log, logfile, resultsfile, resultsfile_agg = createLogFileHandleAndResultsFileNames(prefix="test_fv_implementation")

        #******************* ACTUAL EXECUTION ***************
        #results, results_agg = Test_QB_Particles.test_fv_implementation(K=20, buffer_size_activation=8)
        #results, results_agg = Test_QB_Particles.test_fv_implementation(K=10, buffer_size_activation=0.5)
        results, results_agg, est_mc, est_fv, est_abs, est_surv = Test_QB_Particles.test_fv_implementation(nservers=3, K=20, buffer_size_activation=0.5)
        #results, results_agg = Test_QB_Particles.test_fv_implementation(K=30, buffer_size_activation=0.5)
        #results, results_agg = Test_QB_Particles.test_fv_implementation(nservers=3, K=40, buffer_size_activation=0.25)
        #results, results_agg, est_mc, est_fv, est_abs, est_surv = Test_QB_Particles.test_fv_implementation(nservers=3, K=40, buffer_size_activation=0.5)
        #results, results_agg = Test_QB_Particles.test_fv_implementation(nservers=3, K=40, buffer_size_activation=0.7)
        #******************* ACTUAL EXECUTION ***************

        results.to_csv(resultsfile)
        print("Results of simulation saved to {}".format(os.path.abspath(resultsfile)))

        results_agg.to_csv(resultsfile_agg)
        print("Aggregated results of simulation saved to {}".format(os.path.abspath(resultsfile_agg)))

        closeLogFile(fh_log, stdout_sys, dt_start)
    else:
        # DM-2020/08/24: Instead of using unittest.main(), use the following to test the FV system
        # because there is a very weird error generated by the fact that queue.size inside
        # the EstimatorQueueBlockingFlemingViot class is interpreted of being of class
        # unittest.runner.TextTestResult (or similar)!! instead of simply returning the size
        # attribute of the `queue` object!!!
        test = Test_QB_Particles(nservers=3)
        
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
                    params_mc = test.run_simulation(
                                            buffer_size_activation=buffer_size_activation,
                                            mean_lifetime=None,
                                            proba_survival_given_activation=None,
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
                    params_fv = test.run_simulation(
                                            buffer_size_activation=buffer_size_activation,
                                            mean_lifetime=expected_survival_time,
                                            proba_survival_given_activation=None,
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
        proba_blocking_true = compute_blocking_probability_birth_death_process(test.rhos, test.capacity)
        if proba_blocking_true is not None:
            print("True P(K): {:.6f}%".format(proba_blocking_true*100))
    
        time_end = timer()
        print("Execution time: {:.1f} min".format((time_end - time_start) / 60))
else:
    # Lines to execute "by hand" (i.e. at the Python prompt)
    test = Test_QB_Particles(nservers=3)

    dt_start, stdout_sys, fh_log, logfile, resultsfile, resultsfile_agg = createLogFileHandleAndResultsFileNames(prefix="analyze_estimates")

    # NOTES on the below calls to simulation execution:
    # - Use larger N values to improve the estimation of Phi(t)
    # - When larger N values are used smaller T values (simulation time) can be used
    # because the larger particle N already guarantees a large simulation time for
    # the 1-particle system that estimates P(T>t).
    tests2run = [4]
    if 1 in tests2run:
        results_convergence = test.analyze_estimates(
                                        replications=12,
                                        K_values=[10, 20, 30, 40],
                                        nparticles_values=[200, 400, 800, 1600],
                                        nmeantimes_values=[50, 50, 50, 50],
                                        multiplier_values=[10, 15, 20, 25],
                                        buffer_size_activation_values=[1, 0.2, 0.4, 0.6, 0.8],
                                        seed=1313,
                                        dict_params_out={'logfilehandle': fh_log,
                                                         'resultsfile': resultsfile,
                                                         'resultsfile_agg': resultsfile_agg})
    if 2 in tests2run:
        results_convergence = test.analyze_estimates(
                                        replications=12,
                                        K_values=[10, 20],
                                        nparticles_values=[200, 400],
                                        nmeantimes_values=[50, 50],
                                        multiplier_values=[1.2, 1.2],
                                        buffer_size_activation_values=[1, 0.2, 0.4, 0.6, 0.8],
                                        seed=1313,
                                        dict_params_out={'logfilehandle': fh_log,
                                                         'resultsfile': resultsfile,
                                                         'resultsfile_agg': resultsfile_agg})
    if 3 in tests2run:
        results_convergence = test.analyze_estimates(
                                        replications=12,
                                        K_values=[30, 40],
                                        nparticles_values=[800, 1600],
                                        nmeantimes_values=[50, 50],
                                        multiplier_values=[1.2, 5],
                                        buffer_size_activation_values=[1, 0.2, 0.4, 0.5, 0.7],
                                        seed=1313,
                                        dict_params_out={'logfilehandle': fh_log,
                                                         'resultsfile': resultsfile,
                                                         'resultsfile_agg': resultsfile_agg})
    if 4 in tests2run:
        results_convergence = test.analyze_estimates(
                                        replications=12,
                                        K_values=[10],
                                        nparticles_values=[400],
                                        nmeantimes_values=[50],
                                        multiplier_values=[1],
                                        multiplier_adjust_with_activation=False,
                                        buffer_size_activation_values=[0.1, 0.25, 0.5],
                                        seed=1313,
                                        dict_params_out={'logfilehandle': fh_log,
                                                         'resultsfile': resultsfile,
                                                         'resultsfile_agg': resultsfile_agg})
    if 5 in tests2run:
        results_convergence = test.analyze_estimates(
                                        replications=12,
                                        K_values=[20],
                                        nparticles_values=[600],
                                        nmeantimes_values=[50],
                                        multiplier_values=[1],
                                        multiplier_adjust_with_activation=False,
                                        buffer_size_activation_values=[0.1, 0.25, 0.5],
                                        seed=1313,
                                        dict_params_out={'logfilehandle': fh_log,
                                                         'resultsfile': resultsfile,
                                                         'resultsfile_agg': resultsfile_agg})
    if 6 in tests2run:
        results_convergence = test.analyze_estimates(
                                        replications=12,
                                        K_values=[30],
                                        nparticles_values=[800],
                                        nmeantimes_values=[50],
                                        multiplier_values=[1],
                                        multiplier_adjust_with_activation=False,
                                        buffer_size_activation_values=[0.1, 0.25, 0.5],
                                        seed=1313,
                                        dict_params_out={'logfilehandle': fh_log,
                                                         'resultsfile': resultsfile,
                                                         'resultsfile_agg': resultsfile_agg})
    if 7 in tests2run:
        results_convergence = test.analyze_estimates(
                                        replications=12,
                                        K_values=[40],
                                        nparticles_values=[1200],
                                        nmeantimes_values=[50],
                                        multiplier_values=[1],
                                        multiplier_adjust_with_activation=False,
                                        buffer_size_activation_values=[0.1, 0.25, 0.5],
                                        seed=1313,
                                        dict_params_out={'logfilehandle': fh_log,
                                                         'resultsfile': resultsfile,
                                                         'resultsfile_agg': resultsfile_agg})

    closeLogFile(fh_log, stdout_sys, dt_start)
