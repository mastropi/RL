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
import unittest
from unittest_data_provider import data_provider
import matplotlib.pyplot as plt

import Python.lib.queues as queues
import Python.lib.estimators as estimators
from Python.lib.queues import Event
from Python.lib.estimators import FinalizeType, RenderType

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
        self.nservers = 1
        self.capacity = 5
        self.queue = queues.QueueMM(self.rate_birth, self.rate_death, self.nservers, self.capacity)

        self.plotFlag = True

    def no_tests_on_one_queue(self):
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

    def no_test_simulation_on_one_particle(self):
        print("\nRunning test " + self.id())
        niter = 100
        est = estimators.EstimatorQueueBlockingFlemingViot(1, niter, self.queue,
                                                           reactivate=True)
        
        # The state udpate process is correctly done. Let aNET = "array with Next Event Times",
        # then the following assertions hold.
        est.reset()
        print("The test runs on a simulation with {} iterations" \
              "\n(no pre-specified seed, so that different test runs produce even more tests!)".format(niter))
        for it in range(niter):
            aNET = est.get_times_next_events(0)
            aNET_prev = aNET.copy()
            est.update_state(it+1)
            #print("aNET_prev: {}".format(aNET_prev))
            #print("time last change: {}".format(est.get_time_last_event(0)))
            
            assert len(aNET) == 2, \
                    "There are always two elements in the aNET array containing the next event times"
            
            assert est.get_time_last_event(0) == np.min(aNET_prev), \
                    "The applied event has the smallest time among those stored in the previous value of aNET"
    
            type_event_applied = est.get_type_last_event(0)
            type_event_not_applied = 1 - type_event_applied 
            assert  aNET[type_event_not_applied] > 0 and aNET[type_event_not_applied] == \
                    aNET_prev[type_event_not_applied] - aNET_prev[type_event_applied], \
                    "The updated time-to-event of the event type NOT applied is equal to the previous time-to-event " + \
                    "minus its previous time-to-event value, and is POSITIVE"
    
            time_next_birth = aNET[queues.BIRTH]
            time_next_death = aNET[queues.DEATH]
            assert est._order_times_next_events[0] == [queues.BIRTH, queues.DEATH] and time_next_birth < time_next_death \
                or est._order_times_next_events[0] == [queues.DEATH, queues.BIRTH] and time_next_birth > time_next_death, \
                    "The list that stores the order of next event times reflects the true order"

    def run_test_compute_counts(self,   reactivate,
                                        finalize_type=FinalizeType.ABSORB_CENSORED,
                                        nparticles=5,
                                        niter=20,
                                        seed=1713,
                                        log=False): 
        est = estimators.EstimatorQueueBlockingFlemingViot(nparticles, niter, self.queue,
                                                           reactivate=reactivate,
                                                           finalize_type=finalize_type,
                                                           render_type=RenderType.GRAPH,
                                                           seed=seed, log=log)        
        est.reset()
        for it in range(niter):
            est.update_state(it+1)
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
            est.render()

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
                                                           render_type=RenderType.GRAPH,
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

    def test_simpler_algorithm(self, start=1,
                                     mean_lifetime=None,
                                     reactivate=True,
                                     finalize_type=FinalizeType.REMOVE_CENSORED,
                                     nparticles=5,
                                     niter=40,
                                     seed=1717,
                                     log=False):
        print("\nRunning test " + self.id())
        #nparticles = 30
        #niter = 200
        #reactivate = True
        #finalize_type = FinalizeType.ABSORB_CENSORED
        #finalize_type = FinalizeType.REMOVE_CENSORED
        #seed = 1713
        queue = copy.deepcopy(self.queue) 
        queue.size = start
        est = estimators.EstimatorQueueBlockingFlemingViot(nparticles, niter, queue,
                                                           mean_lifetime=mean_lifetime,
                                                           reactivate=reactivate,
                                                           finalize_type=finalize_type,
                                                           render_type=RenderType.GRAPH,
                                                           seed=seed, log=log)
        print("Simulation setup:")
        print(est.setup())

        est.simulate()
        est.render()

        print("\n\nRelevant events for each particle ID:")
        if False:
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

        est.finalize()
        est.compute_counts()

        if False and not reactivate:
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

        proba_blocking, integral, expected_survival_time = est.estimate_proba_blocking()
        
        print("Total blocking time for ALL particles:")
        all_total_blocking_time = est.get_all_total_blocking_time()
        all_total_survival_time = est.get_all_total_survival_time()
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


        rho = self.rate_birth / self.rate_death
        K = self.capacity
        print("\nEstimation of blocking probability via Approximation 1:")
        print("Integral = {:.3f}".format(integral))
        print("Expected Survival Time = {:.3f}".format(expected_survival_time))
        print("Blocking probability estimate: {:.1f}%".format(proba_blocking*100))
        print("Theoretical value (rho^K / sum(rho^i)) (rho={:.3f}, K={}): {:.1f}%" \
              .format(rho, K,
                      rho**K / np.sum([ rho**i for i in range(K+1) ]) *100))

        print("Simulation setup:")
        print(est.setup())

        df_proba_survival_and_blocking_conditional = est.estimate_proba_survival_and_blocking_conditional()
        self.plot_results(df_proba_survival_and_blocking_conditional, 
                          reactivate,
                          finalize_type,
                          nparticles,
                          niter,
                          seed)

        return  df_proba_survival_and_blocking_conditional, \
                expected_survival_time, \
                (reactivate, finalize_type, nparticles, niter)

    def plot_results(self, df_proba_survival_and_blocking_conditional, *args):
        reactivate = args[0]
        finalize_type = args[1]
        nparticles = args[2]
        niter = args[3]
        seed = args[4]
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
        plt.title("K={}, rate(B)={:.1f}, rate(D)={:.1f}, rho={:.3f}, reactivate={}, finalize={}, N={}, #iter={}, seed={}" \
                  .format(self.capacity,
                          self.queue.rates[Event.BIRTH.value],
                          self.queue.rates[Event.DEATH.value],
                          self.queue.rates[Event.BIRTH.value] / self.queue.rates[Event.DEATH.value],
                          reactivate, finalize_type.name[0:3],
                          nparticles, niter, seed
                          ))
        ax.title.set_fontsize(9)


if __name__ == "__main__":
    unittest.main()

else:
    # Lines to execute "by hand" (i.e. at the Python prompt)
    test = Test_QB_Particles()

    nparticles = 5
    niter = 100
    seed = 1717

    # Estimate the expected survival time using simulation without reactivation
    test_mc = Test_QB_Particles()
    reactivate = False
    finalize_type = FinalizeType.REMOVE_CENSORED
    start = 0
    mean_lifetime = None
    df_resultsFR0, EST_FR0, paramsFR0 = test_mc.test_simpler_algorithm(start,
                                                                       mean_lifetime,
                                                                 reactivate,
                                                                 finalize_type,
                                                                 nparticles,
                                                                 niter,
                                                                 seed)

    test_fv = Test_QB_Particles()
    reactivate = True
    finalize_type = FinalizeType.REMOVE_CENSORED
    start = 1
    mean_lifetime = EST_FR0
    df_resultsTR1, EST_TR1, paramsTR1 = test_fv.test_simpler_algorithm(start,
                                                                 mean_lifetime,
                                                                 reactivate,
                                                                 finalize_type,
                                                                 nparticles,
                                                                 niter,
                                                                 seed)

    if False:
        #####
        reactivate = False
        finalize_type = FinalizeType.ABSORB_CENSORED
        df_resultsFA, paramsFA = test.tests_on_n_particles(reactivate,
                                                        finalize_type,
                                                        nparticles,
                                                        niter,
                                                        seed)
        test.plot_results(df_resultsFA, *paramsFA)
    
        #####
        reactivate = True
        finalize_type = FinalizeType.ABSORB_CENSORED
        df_resultsTA, paramsTA = test.tests_on_n_particles(reactivate,
                                                        finalize_type,
                                                        nparticles,
                                                        niter,
                                                        seed)
        test.plot_results(df_resultsTA, *paramsTA)
    
        #####
        reactivate = False
        finalize_type = FinalizeType.REMOVE_CENSORED
        df_resultsFR, paramsFR = test.tests_on_n_particles(reactivate,
                                                        finalize_type,
                                                        nparticles,
                                                        niter,
                                                        seed)
        test.plot_results(df_resultsFR, *paramsFR)
    
        #####
        reactivate = True
        finalize_type = FinalizeType.REMOVE_CENSORED
        df_resultsTR, paramsTR = test.tests_on_n_particles(reactivate,
                                                        finalize_type,
                                                        nparticles,
                                                        niter,
                                                        seed)
        test.plot_results(df_resultsTR, *paramsTR)
