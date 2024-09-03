# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:16:02 2020

@author: Daniel Mastropietro
@description: Test the blocking probability estimation in queues.
"""

import runpy
runpy.run_path('../../setup.py')

from enum import Enum, unique
import copy
from timeit import default_timer as timer
import unittest

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt, cm, ticker as mtick

from Python.lib.utils.basic import aggregation_bygroups, set_numpy_options, reset_numpy_options
import Python.lib.utils.plotting as plotting

from Python.lib.agents.policies import PolicyTypes
from Python.lib.agents.policies.job_assignment import PolJobAssignmentProbabilistic

from Python.lib.agents.queues import AgeQueue

import Python.lib.queues as queues  # The keyword `queues` is used in the code
from Python.lib.queues import Event
from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses, rewardOnJobRejection_ExponentialCost

import Python.lib.deprecated_estimators as estimators
from Python.lib.deprecated_estimators import FinalizeType

from Python.lib.utils.computing import get_server_loads, compute_job_rates_by_server, compute_blocking_probability_birth_death_process


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
                _dict_numpy_options = set_numpy_options()
                print("activation times: {}".format(np.array(activation_times)))
                reset_numpy_options(_dict_numpy_options)
            assert sorted(activation_times) == activation_times, \
                    "The absolute times of activation are sorted: {}".format(np.array(activation_times))
            assert len(activation_times) >= N1, \
                    "The number of activation times ({}) is at least equal to the number of active particles (N1 = {})" \
                    .format(len(activation_times), N1)

            particles, absorption_times = est.get_all_absorption_times()
            if self.log:
                _dict_numpy_options = set_numpy_options()
                print("absorption times: {}".format(np.array(absorption_times)))
                reset_numpy_options(_dict_numpy_options)
            assert sorted(absorption_times) == absorption_times, \
                    "The (relative) absorption times are sorted: {}".format(np.array(absorption_times))            

            # The list storing the time segments where statistics are computed is sorted
            survival_time_segments = est.get_survival_time_segments()
            if self.log:
                _dict_numpy_options = set_numpy_options()
                print("time segments: {}".format(np.array(survival_time_segments)))
                reset_numpy_options(_dict_numpy_options)
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
            _dict_numpy_options = set_numpy_options()
            print("Survival time segments: {}".format(np.array(survival_time_segments)))
            reset_numpy_options(_dict_numpy_options)
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
            _dict_numpy_options = set_numpy_options()
            print("Blocking time segments: {}".format(np.array(blocking_time_segments)))
            reset_numpy_options(_dict_numpy_options)
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
        _dict_numpy_options = set_numpy_options()
        print("Latest event times: {}".format(np.array(times_last_event_by_particle)))
        particles, elapsed_times_since_activation = est.get_all_elapsed_times()
        print("Latest elapsed times since activation: {}".format(np.array(elapsed_times_since_activation)))
        print("Particles associated to these times  : {}".format(particles))
        reset_numpy_options(_dict_numpy_options)

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
        _dict_numpy_options = set_numpy_options()
        print("Estimated probabilities by time:")
        print(df_proba_survival_and_blocking_conditional)
        reset_numpy_options(_dict_numpy_options)

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


# DM-2022/10/06: Function currently not in use by the project but could be useful as it has a lot of logic implemented
def plot_aggregated_convergence_results(results_convergence):

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


# DM-2022/10/06: Function currently not in use by the project but could be useful as it has a lot of logic implemented
def plot_aggregated_convergence_results_maxvalues(results_convergence):

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
    agent = AgeQueue(env_queue, policies, learners)

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
#------------------- Functions --------------------
