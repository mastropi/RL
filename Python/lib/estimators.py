# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:12:23 2020

@author: Daniel Mastropietro
@description: Estimators are defined for problems of interest in Reinforcement Learning.
"""

import copy     # Used to generate different instances of the queue under analysis
import bisect   # To insert elements in order in a list (bisect.insort(list, element)

import numpy as np
from matplotlib import pyplot as plt, cm    # cm is for colormaps (e.g. cm.get_cmap())

from . import queues
from .utils.basic import find

class EstimatorQueueBlockingFlemingViot:
    
    def __init__(self, nparticles, queue, seed=None):
        self.N = nparticles
        self.queue = queue          # This variable is used to create the particles as replications of the given queue
        self.seed = seed

        # Positions or state of each particle
        self.positions = np.zeros(self.N, dtype=int)

        # Times of last change
        #self.times_last_change = np.zeros(self.N)
        self.time_last_change_of_system = 0
        
        self.reset()
        
    def reset(self):
        np.random.seed(self.seed)

        # Reset the iteration counter
        # (which starts at 1 when simulation runs,
        # and its value is set by the simulation process when calling update_state()) 
        self.iter = 0

        # 1) Initialize the particles in the system
        # (after deleting the variable in case it exists, so that we do not use more memory)
        try:
            del self.particles
        except:
            pass
        # Initialize the first particle to the queue stored in the object
        self.particles = [self.queue]
        # Add copies of the given queue if N > 1, so that we can simulate several particles
        self.particles += [ copy.deepcopy(self.queue) for _ in np.arange(self.N-1) ]

        # 2) Compute the times of the next birth and death for each queue/particle, and their order
        self.times_next_events = [ [self.queue.generate_next_birth_time(), self.queue.generate_next_death_time()]
                                    for _ in np.arange(self.N) ]
        self._order_times_next_events = [ self._compute_order( self.times_next_events[p] )
                                            for p in np.arange(self.N) ]

        # 3) Attributes used directly in the computation of the estimated measures that contribute to the probability of blocking.
        self.t = []     # Absolute times when each particle touches 1 (either for the first time or not)
                        # and thus is added as a particle that contributes to the estimation of the probability of blocking.
                        # The times are sorted in ascending order.
                        # Goal: compute the relative times 's' defined below.
                        # Size: N1 = variable number of particles being considered for the estimation of the probability of blocking.
        self.p = []     # Particle number that corresponds to the time reported in the t array.
                        # We keep this information so that we can find the activation time of a particle when
                        # the particle is blocked or stops being blocked.
                        # Size: same as the activation times array
        self.iact = []  # Array that keeps track of the iteration at which each activation took place.
                        # Size: same as the activation times array
        self.active = []# Whether the "particle" in the corresponding position of the t array is active or alive,
                        # i.e. if it is part of the set of N1 trajectories that are used to compute the statistics
                        # for the blocking probability.
                        # We keep this information so that we can:
                        # - access ALL the history of the times when particles became active
                        # - avoid spending time in removing a particle from the self.t array  
                        # Size: same as the activation times array
        self.s = [0]     # Times when a change happens in the system affecting the statistics that depend on
                        # the number of particles that are ALIVE or ACTIVE (i.e. have not been absorbed)
                        # These times are measured RELATIVE to the activation moment of each particle,
                        # i.e. to the moment when they touched position 1.
                        # The list is updated whenever a new particle is activated (i.e. touches position 1,
                        # either for the first time or not), and when this happens a set of N1 values is added
                        # to the list, in a way that the list's natural order is preserved.
                        # Each of the N1 times is computed as the difference between the absolute time
                        # the particle is ACTIVATED and each of the activation time of every particle already
                        # alive in the system.
                        # Goal: define the disjoint time segments on which the measures to be estimated are constant.
                        # Size: N1*(N1-1)/2 = variable number of time points defining the said disjoint time segments.
        self.sb = []    # Times when a change happens in the system affecting the statistics that depend on
                        # the number of particles that are BLOCKED.
                        # Size: As many elements as trajectories beginning at position 1 that come from particles
                        # becoming either blocked or unblocked.
        self.ktimes = []# List of sorted absolute absorption times (for checking purposes). k = kill
                        # Size: As many as number absorptions occurred
        self.pkt = []   # Particle number associated to each absorption time (pkt = Particle Kill Time).
                        # Size: Same as ktimes
        self.ikt = []   # Array that keeps track of the iteration at which each absorption took place.
                        # Size: same as the absorption times array
        # Particle counts used in the computation of the estimated measures 
        self.counts_alive = [0]     # Number of particles that are alive in each time segment defined by s
                                    # => have not been absorbed by position 0.
                                    # The time segment associated to the count has the time value in the s array
                                    # as its lower bound.
                                    # Size: same as self.s
        self.counts_blocked = []    # Number of particles that are blocked in each time segment defined by sb
                                    # => have reached position K, the maximum capacity of the queue represented by the particle. 
                                    # Size: same as self.sb

        #print("Particle system with {} particles has been reset:".format(self.N))
        #print("times_next_events = {}".format(self.times_next_events))
        #print("_order_times_next_events = {}".format(self._order_times_next_events))
        #print("")

    def _compute_order(self, arr):
        return list( np.argsort(arr) )

    def _update_one_particle(self, p):
        """
        Updates the position/state of the particle based on the current "next times-to-event array"
        for the particle, and updates this array with new times-to-event.
       
        The times-to-event array is a two-element array that represents the times of occurrence of the
        two possible events on a particle, namely birth and death in this order. The times are measured
        from the occurrence of the latest event taking place on the particle.
      
        In practice, this method does the following: it takes the current value of the said two-element array
        --e.g. [3.2, 2.7]-- and selects the smallest time between the two (2.7 in this case, which corresponds
        to a death event). The goal is to apply to the particle the event that is happening *next*, and update
        its state accordingly (i.e. increase its position if it's a birth event or decrease its position
        if it's a death event, as long as the position restrictions are satisfied). 
        
        The two-element array is then updated as follows:
        - a new random value is generated to fill the time-to-event for the (death) event just chosen;
        - the time corresponding to the non-chosen event (birth) is decreased so that the new value represents
        the time-to-event for the (birth) event *as measured from the time in which the chosen (death) event
        just took place*.
        
        For example, the two-element array may be updated to [0.5, 1.2],  where 0.5 is a deterministic value
        (= 3.2 - 2.7) while 1.2 is a randomly generated value --e.g. following an exponential distribution with
        parameter equal to the death rate.
        
        The new array stores the information of *when* the next event will take place on this particle and
        of what *type* it will be. In this case, the next event will take place 0.5 time units later than
        the (death) event just selected, and it will be a birth event --since the said time value (0.5) is
        stored at the first position of the array, which represents a birth event. 

        Arguments:
        p: int
            Index of the particle to update, from 0 to N-1, where N is the number of particles in the system.
        
        Return: list
        A two-element list containing:
        - the time of the selected event, measured in absolute terms, i.e. from the common origin of all particles.
        - the change in position after applying the event, which can be -1, 0 or +1.
        """
        assert isinstance(p, int) and 0 <= p and p < self.N, \
                "The selected particle ({}) is valid (its index is an integer between 0 and {})".format(p, len(self.N)-1)
        
        # Find the event type to apply (BIRTH or DEATH, whichever occurs first)
        event_type_to_apply = self._order_times_next_events[p][0]
        #print("times_next_events[{}]: {}".format(p, self.times_next_events[p]))
        #print("event_type_to_apply: {}".format(event_type_to_apply))
        time_since_previous_event = self.times_next_events[p][event_type_to_apply]
        time_of_event = self.particles[p].getTimeLastEvent() + time_since_previous_event

        # Update the particle's state by applying the event type just selected and record the position change
        position_change = self._apply_event(p, event_type_to_apply, time_of_event)
        event_type_applied = event_type_to_apply
        event_type_not_applied = self._order_times_next_events[p][1]

        # Generate the next time for the event type just applied from the list of event times for this particle
        event_rate = self.particles[p].rates[event_type_applied]
        self.times_next_events[p][event_type_applied] = self.particles[p].generate_event_times(event_rate, size=1)

        # Update the time-to-event for the event type not applied so that it is now relative to the event just applied
        self.times_next_events[p][event_type_not_applied] -= time_since_previous_event

        # Update the order of the next event times
        self._order_times_next_events[p] = self._compute_order( self.times_next_events[p] )

        #print("UPDATED times_next_events[{}]: {}".format(p, self.times_next_events[p]))

        return time_of_event, position_change

    def _apply_event(self, p, event_type_to_apply, time_to_event):
        position_change = self.particles[p].apply_event(event_type_to_apply, time_to_event)
        self.positions[p] = self.particles[p].size
        return position_change

    def update_state(self, iter):
        """
        The state of the particle system is updated based on the time of occurrence of the
        next event on each particle.
        
        Goal: extract the information of when the next event happens on each particle, update
        the position of each particle and the times of the upcoming next events,
        which will be used the next time the system state is updated.  
        
        Return: float numpy array
        The times of occurrence of the events on each particle leading to the update of the system state.
        The times of occurrence are measured w.r.t. the common origin for all particles.
        """
        self.iter = iter

        times_of_events = np.nan*np.ones(self.N)
        positions_change = np.zeros(self.N, dtype=int)
        for p in range(self.N):
            times_of_events[p], positions_change[p] = self._update_one_particle(p)
        with np.printoptions(precision=3, suppress=True):
            print("\ntimes to event: {}".format(times_of_events))
        print("\tparticle positions: {}".format(self.positions))
        print("\tpositions change: {}".format(positions_change))
        print("\tactivation times by particle:")
        for p in range(self.N):
            _, activation_times_p = self.get_activation_times(p)
            with np.printoptions(precision=3):
                print("\t\tp={}: {}".format(p, np.array(activation_times_p)))
        print("\ttime elapsed since activation times:")
        for p in range(self.N):
            _, activation_times_p = self.get_activation_times(p)
            with np.printoptions(precision=3):
                print("\t\tp={}: {}".format(p, np.array([times_of_events[p] - u for u in activation_times_p])))
        order_times_to_event = self._compute_order(times_of_events)

        self.update_times_record(times_of_events, order_times_to_event, positions_change)

        particle_with_latest_event = order_times_to_event[-1]
        self.time_last_change_of_system = times_of_events[ particle_with_latest_event ]

        return times_of_events

    def update_times_record(self, times_of_events, order_times_to_event, positions_change):
        for o in order_times_to_event:
            # Absolute time to event on the current particle
            t = times_of_events[o]

            # Possible status change of a particle 
            is_activated = lambda p: self.positions[p] == 1 # Note that a particle can be activated even if it was ALREADY at position 1 => a new sub-trajectory becomes active
            is_absorbed = lambda p: positions_change[p] < 0 and self.positions[p] == 0
            is_blocked = lambda p: self.positions[p] == self.particles[p].getCapacity()
            is_nolonger_blocked = lambda p: positions_change[p] < 0 and self.positions[p] == self.particles[p].getCapacity() - positions_change[p] 

            if is_activated(o):
                self._add_new_activation_time(o, t)

            elif is_absorbed(o):
                self._add_new_time_segment(o, t)

            elif is_blocked(o) or is_nolonger_blocked(o):
                self._add_new_blocking_time(o, t)

            with np.printoptions(precision=3, suppress=True):
                particles, times_from_activation = self.get_all_elapsed_times()
                print("event time: {:.3f}, active in the system (at different times): {}" \
                      "\n\tt: {}" \
                      "\n\ta: {}" \
                      "\n\te: {}" \
                      "\n\tp: {}" \
                      "\n\ts: {}" \
                      "\n\tcounts_alive: {}" \
                      .format(t, sum(self.active),
                             np.array(self.t),
                             self.active,
                             np.array(times_from_activation),
                             np.array(particles),
                             np.array(self.s),
                             self.counts_alive))

    def _add_new_activation_time(self, o, time_of_event):
        "Adds a new time to the array of activation times"
        # Add this time to the list of activation times and to the list of active flags
        # NOTE: we need to EXPLICITLY insert the new time IN ORDER in the array,
        # despite the fact the time_of_event values are SORTED in the CURRENT call to this method,
        # because we do not know anything about the order of these time_of_event values in relation to
        # the time values ALREADY PRESENT in the array.
        idx_insort = self.insort(self.t, time_of_event)
        self.p.insert(idx_insort, o) # Store the particle number associated to the time just added
        self.iact.insert(idx_insort, self.iter)
        self.active.insert(idx_insort, True)
        with np.printoptions(precision=3, suppress=True):
            print("\n>>>>> ACTIVATION: p={}: a={:.3f}, time_of_event={}".format(o, time_of_event, np.array(self.t)))

    def _add_new_time_segment(self, p, time_of_event):
        "Adds a new time segment to the array of time segments"
        #-- Insert the current time in the array of absorption times
        idx_insort = self.insort(self.ktimes, time_of_event)
        self.pkt.insert(idx_insort, p)
        self.ikt.insert(idx_insort, self.iter)

        #-- Insert the current time in the array of time-segment endings
        #-- and update counts of each segment accordingly
        # Get the activation times for all trajectories associated to this absorbed particle
        # and iterate on each of them to compute the relative times to obsorbtion to be
        # added as new time-segment endings.
        idx_activation_times, activation_times = self.get_activation_times(p)
        relative_times_to_killed = [time_of_event - self.t[i] for i in idx_activation_times]
        with np.printoptions(precision=3, suppress=True):
            print("\nXXXXXXXXXXXXXXX ABSORPTION: p={}: time_of_event={:.3f}, a: {}, s: {}".format(p, time_of_event, np.array(activation_times), np.array(relative_times_to_killed)))
        for idx_activation_time, s in zip(idx_activation_times, relative_times_to_killed):
            idx_insort = self.insort(self.s, s)
            self.update_counts_alive(idx_insort)
            # De-activate the trajectory associated to the absorption
            self.active[idx_activation_time] = False

    def _add_new_blocking_time(self, o, t):
        # Find all the sub-trajectories that can be associated to the current particle's blocking
        # There may be more than one if the particle has touched 1 more than once without being absorbed.
        idx_activation_times = find(self.p, o)
        relative_times_to_blocked_or_unblocked = [t - self.t[i] for i in idx_activation_times]
        pass

    def insort(self, target_list, s):
        """
        Inserts a value in an array in order and returns the index of insertion
        (something that the bisect.insort() function does not do)
         """
        idx = bisect.bisect(target_list, s)  # Index where to insert the new relative time in order
        target_list.insert(idx, s)           # Insert the new relative time in order
        return idx

    def update_counts_alive(self, idx_to_insert):
        """
        Updates the counts of particles alive in each time segment where the count can change
        following the activation of a new particle.
        """
        assert 1 <= idx_to_insert and idx_to_insert <= len(self.counts_alive), \
                "The index where a new counter needs to be inserted ({}) is at least 1" \
                " and at most the current number of elements in the counts_alive array ({})" \
                .format(idx_to_insert, len(self.counts_alive))

        # Insert a new element in the array whose value is equal to the count of the segment
        # before split (in fact, the particle just absorbed had not yet been counted in the segment
        # and thus we should not count it on the right part of the split, since the particle no longer
        # exists at that time)
        self.counts_alive.insert(idx_to_insert, self.counts_alive[idx_to_insert - 1])

        # Increase by 1 ALL counts to the LEFT of the insert index
        # (because the absorption indicates that all time segments that are smaller
        # than the elapsed time to absorption should count the particle just absorbed as active)
        for idx in range(idx_to_insert):
            self.counts_alive[idx] += 1

    def update_counts_blocked(self, idx_to_insert, n_time_insertions):
        """
        Updates the counts of blocked particles in each time segment where the count can change
        following the blocking of a new particle.
        """
        assert idx_to_insert <= len(self.counts_blocked), \
                "The index where a new counter needs to be inserted ({}) is at most" \
                " the current number of elements in the counts_blocked array ({})" \
                .format(idx_to_insert, len(self.counts_blocked))

        # Insert a new element in the array that counts the number of particles blocked in the corresponding time segment
        self.counts_blocked.insert(idx_to_insert, 0)
        for idx in range(idx_to_insert + 1):
            self.counts_blocked[idx] += 0

    def finalize(self):
        for i, active_particle in enumerate(self.active):
            if active_particle:
                # => We add the time segment corresponding to the NEXT event happening for each active particle
                # Note that this time is fully acknowledgeable as if the particle was active at the latest event
                # then it will still be active just before the next event takes place, which may be considered
                # as the last time we observe the particle.
                p = self.p[i]
                next_event_type = self._order_times_next_events[p][0]
                time_of_next_event = self.particles[p].getTimeLastEvent() + self.times_next_events[p][next_event_type]
                self._add_new_time_segment(p, time_of_next_event)

    def estimate_proba_absortion(self, n, t):
        return np.mean([ self.positions[p] for p in self.particles ])

    #-------------- Getters
    def get_number_active_particles(self):
        return sum(self.active)

    def get_all_activation_times(self):
        return self.p, self.t

    def get_all_absorption_times(self):
        return self.pkt, self.ktimes

    def get_all_elapsed_times(self):
        """
        Returns the elapsed time from activation of each ACTIVE sub-trajectory of each particle in the system
        
        Return: list
        Two-element list containing:
        - a list with the particle numbers corresponding to each element in the elapsed times array
        - a list with the elapsed times of each trajectory at the current point of the simulation 
        """
        particles = []
        times_from_activation = []
        for p in range(self.N):
            idx_activation_times_p, activation_times_p = self.get_activation_times(p)
            t = self.get_time_last_event(p)
            particles += [self.p[idx] for idx in idx_activation_times_p]
            times_from_activation += [t - u for u in activation_times_p]

        order = self._compute_order(times_from_activation)
        return [particles[o] for o in order], [times_from_activation[o] for o in order]

    def get_activation_times(self, p):
        """
        Returns the activation times for all ACTIVE trajectories of particle p,
        that is for all sub-trajectories of the trajectory of particle p that start at position 1.
        
        Return: list
        Two-element list containing:
        - a list with the indices in the activation times array from where the activation times are taken
        - a list with the activation times 
        """
        idx_activation_times_p = [idx for idx in find(self.p, p) if self.active[idx]]
        activation_times_p = [self.t[i] for i in idx_activation_times_p]        
        return idx_activation_times_p, activation_times_p

    def get_absorption_times(self, p):
        """
        Returns the absorption times (relative to the trajectory activation) for all trajectories
        historically observed for particle p, that is for all sub-trajectories of all trajectories
        observed for particle p that start at position 1.
        
        Return: list
        Two-element list containing:
        - a list with the particle numbers corresponding to each element in the absorption times array
        - a list with the absorption times of each sub-trajectory measured w.r.t. the sub-trajectory activation
        """
        idx_absorption_times_p = [idx for idx in find(self.pkt, p)]
        absorption_times_p = [self.ktimes[i] for i in idx_absorption_times_p]
        return idx_absorption_times_p, absorption_times_p

    def get_time_segments(self):
        return self.s

    def get_type_last_event(self, p):
        return self.particles[p].getTypeLastEvent()

    def get_all_times_last_event(self):
        return [self.get_time_last_event(p) for p in range(self.N)]

    def get_time_last_event(self, p):
        return self.particles[p].getTimeLastEvent()

    def get_all_times_next_events(self):
        return self.times_next_events

    def get_times_next_events(self, p):
        try:
            return self.times_next_events[p]
        except:
            self.raiseErrorInvalidParticle(p)
            return np.nan
    
    def get_time_last_change_of_system(self):
        return self.time_last_change_of_system

    def get_counts_particles_alive_by_elapsed_time(self):
        return self.counts_alive

    def get_counts_particles_blocked_by_elapsed_time(self):
        return self.counts_blocked

    def render(self):
        all_types = [1]*len(self.t) + [-1]*len(self.ktimes)
        all_particles = self.p + self.pkt
        all_times = self.t + self.ktimes
        all_iters = self.iact + self.ikt

        all_order = np.argsort( np.abs(all_times) )

        all_types = [all_types[o] for o in all_order]
        all_particles = [all_particles[o] for o in all_order]
        all_times = [all_times[o] for o in all_order]
        all_iters = [all_iters[o] for o in all_order]

        render_str = ""
        activation_times_shown = [False]*len(self.t)
        iter_width = len(str(self.iter))    # Display width for the iteration number so that all lines are aligned in the output string 
        for i, t in enumerate(all_times):
            if all_types[i] > 0:
                render_str += "it={:{width}}: [A] ".format(all_iters[i], width=iter_width) + "{:.3f}".format(t) \
                                + " ({})\n".format(all_particles[i])
            elif all_types[i] < 0:
                # Get all activation times for the absorbed particle prior to the absorption time
                p = all_particles[i]
                absorption_time = t
                idx_activation_times_for_absorption = [idx for idx in find(self.p, p) if self.t[idx] < absorption_time and not activation_times_shown[idx]]
                activation_times_for_absorption = [self.t[idx] for idx in idx_activation_times_for_absorption]
                str_relative_absorption_times = ",".join(["{:.3f}".format(t - u) for u in activation_times_for_absorption])
                render_str += "it={:{width}}: [X] ".format(all_iters[i], width=iter_width) + "{:.3f}".format(t) \
                                + " ({};{})\n".format(p, str_relative_absorption_times)

                # Update the list of activation times already shown (so that they are not shown again in future absorption events!)
                for idx in idx_activation_times_for_absorption:
                    activation_times_shown[idx] = True

        return render_str

    #-------------- Helper functions
    def raiseErrorInvalidParticle(self, particle):
        raise ValueError("Wrong value for particle: {}\n".format(particle))
