# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:16:02 2020

@author: Daniel Mastropietro
@description: Estimators are defined for problems of interest in Reinforcement Learning.
"""

import runpy
runpy.run_path('../../setup.py')

import numpy as np
import unittest
from unittest_data_provider import data_provider
import matplotlib.pyplot as plt

import Python.lib.queues as queues
import Python.lib.estimators as estimators

#from importlib import reload
#reload(estimators)
#from Python.lib.estimators import EstimatorQueueBlockingFlemingViot

#import test_utils

class Test_QB_Particles(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = 10
        self.rate_birth = 1.2
        self.rate_death = 1.4
        self.nservers = 1
        self.capacity = 5
        self.queue = queues.QueueMM(self.rate_birth, self.rate_death, self.nservers, self.capacity)

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
        est = estimators.EstimatorQueueBlockingFlemingViot(1, queue=self.queue)
        
        # The state udpate process is correctly done. Let aNET = "array with Next Event Times",
        # then the following assertions hold.
        est.reset()
        niter = 100
        print("The test is run on {} different simulations (no pre-specified seed)".format(niter))
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
                    "The array that stores the order of next event times reflects the true order"

    def tests_on_n_particles(self):
        print("\nRunning test " + self.id())
        nparticles = 5
        seed = 1717
        est = estimators.EstimatorQueueBlockingFlemingViot(nparticles, queue=self.queue, seed=seed)

        # Changes on the last time the system changed when a new event occurs
        # OLD
        #t = est.generate_event_time(self.rate_birth, size=1)
        #est.time_last_change_of_system = est.time_last_change_of_system + t
        
        est.reset()
        niter = 10
        print("The test is run on {} different simulations (no pre-specified seed)".format(niter))
        for it in range(niter):
            # The list storing the times when particles became active is sorted increasingly
            est.update_state(it+1)
            
            print("------ END OF ITER {} of {} ------".format(it+1, niter))
            N1 = n_active_particles = est.get_number_active_particles()
            particles, activation_times = est.get_all_activation_times()
            with np.printoptions(precision=3):
                print("activation times: {}".format(np.array(activation_times)))
                assert sorted(activation_times) == activation_times, \
                        "The absolute times of activation are sorted: {}".format(np.array(activation_times))
            assert len(activation_times) >= N1, \
                    "The number of activation times ({}) is at least equal to the number of active particles (N1 = {})" \
                    .format(len(activation_times), N1)

            particles, absorption_times = est.get_all_absorption_times()
            with np.printoptions(precision=3):
                print("absorption times: {}".format(np.array(absorption_times)))
                assert sorted(absorption_times) == absorption_times, \
                        "The (relative) absorption times are sorted: {}".format(np.array(absorption_times))            

            # The list storing the time segments where statistics are computed is sorted
            time_segments = est.get_time_segments()
            with np.printoptions(precision=3):
                print("time segments: {}".format(np.array(time_segments)))
                assert sorted(time_segments) == time_segments, \
                        "The time segments are sorted: {}".format(np.array(time_segments))
            assert time_segments[0] == 0.0, \
                    "The first element of the time segments array ({}) is equal to 0" \
                    .format(time_segments[0])
            #assert time_segments[1:] == absorption_times, \
            #        "The ending time segments ({}) are equal to the absorption times ({})" \
            #        .format(time_segments, absorption_times)

            counts_particles_alive_by_elapsed_time = est.get_counts_particles_alive_by_elapsed_time()
            print("Counts by segment: {}".format(counts_particles_alive_by_elapsed_time))
            assert np.all( counts_particles_alive_by_elapsed_time == np.arange(len(time_segments)-1, -1, -1) ), \
                    "Since the time segments are defined by jumps in the count of active particles" \
                    " they should decrease linearly to 0 starting at {} ({})" \
                    .format(len(time_segments)-1, counts_particles_alive_by_elapsed_time)
            print("------ END OF ITER {} of {} ------".format(it+1, niter))

        # At the last iteration we should finalize the counts of active praticles
        # by counting all the particles that are still active 
        est.finalize()

        time_segments = est.get_time_segments()
        with np.printoptions(precision=3):
            print("time segments: {}".format(np.array(time_segments)))
            assert sorted(time_segments) == time_segments, \
                    "The time segments are sorted: {}".format(np.array(time_segments))
        assert time_segments[0] == 0.0, \
                "The first element of the time segments array ({}) is equal to 0" \
                .format(time_segments[0])

        counts_particles_alive_by_elapsed_time = est.get_counts_particles_alive_by_elapsed_time()
        print("Counts by segment: {}".format(counts_particles_alive_by_elapsed_time))
        assert np.all( counts_particles_alive_by_elapsed_time == np.arange(len(time_segments)-1, -1, -1) ), \
                "Since the time segments are defined by jumps in the count of active particles" \
                " they should decrease linearly to 0 starting at {} ({})" \
                .format(len(time_segments)-1, counts_particles_alive_by_elapsed_time)

        print("\nLatest time the system changed: {:.3f}".format(est.get_time_last_change_of_system()))
        times_last_event_by_particle = est.get_all_times_last_event()
        print("Range of latest event times in all {} particles in the system: [{:.3f}, {:.3f}]" \
              .format(est.N, np.min(times_last_event_by_particle), np.max(times_last_event_by_particle)))
        print("Latest particle positions: {}".format(est.positions))
        with np.printoptions(precision=3):
            print("Latest event times: {}".format(np.array(times_last_event_by_particle)))
            particles, elapsed_times_since_activation = est.get_all_elapsed_times()
            print("Latest elapsed times since activation: {}".format(np.array(elapsed_times_since_activation)))
            print("Particles associated to these times  : {}".format(particles))
        print("\nSIMULATION RENDER:")
        print(est.render())
        print("\nSeed used: {}".format(est.seed))

if __name__ == "__main__":
    unittest.main()
