# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:12:23 2020

@author: Daniel Mastropietro
@description: Estimators are defined for problems of interest in Reinforcement Learning.
"""

from enum import Enum, unique

import copy     # Used to generate different instances of the queue under analysis
import bisect   # To insert elements in order in a list (bisect.insort(list, element)

import numpy as np
from matplotlib import pyplot as plt, cm    # cm is for colormaps (e.g. cm.get_cmap())

from . import queues
from .queues import Event
from .utils.basic import find

@unique # Unique enumeration values (i.e. on the RHS of the equal sign)
class EventType(Enum):
    ABSORPTION = 0
    BLOCK = +1
    UNBLOCK = -1

class EstimatorQueueBlockingFlemingViot:
    
    def __init__(self, nparticles, queue, seed=None, log=False):
        self.N = nparticles
        self.queue = queue          # This variable is used to create the particles as replications of the given queue
        self.seed = seed
        self.LOG = log

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

        # Positions or state of each particle
        self.positions = np.zeros(self.N, dtype=int)

        # 2) Compute the times of the next birth and death for each queue/particle, and their order
        self.times_next_events = [ [self.queue.generate_next_birth_time(), self.queue.generate_next_death_time()]
                                    for _ in np.arange(self.N) ]
        self._order_times_next_events = [ self._compute_order( self.times_next_events[p] )
                                            for p in np.arange(self.N) ]

        # 3) Attributes used directly in the computation of the estimated measures that contribute to the probability of blocking.
        self.active = []# Whether the particle in the corresponding index of the activation times array is active,
                        # i.e. if it has not yet been absorbed.
                        # GOAL: find all the activation times for an absorption event. 
                        # SIZE: same as the activation times array

        self.atimes = []# Absolute times when a particle becomes active, i.e. touches position 1.
                        # If this happens for an active particle, a fictitious particle is created
                        # and their position followed as well.
                        # These times are called activation times and they are sorted increasingly in the array.
                        # GOAL: use as reference time to compute the relative absorption and blocking times.
                        # SIZE AT THE END OF SIMULATION: N1 = number of particles considered for the estimation
                        # of the probability of blocking.
        self.pat = []   # Particle number associated to the corresponding index in the activation times array.
                        # GOAL: find the activation time of a particle when the particle is absorbed/blocked/unblocked.
                        # SIZE: same as the activation times array
        self.iat = []   # Array that keeps track of the iteration at which each activation takes place.
                        # SIZE: same as the activation times array

        self.ktimes = []# List of sorted absolute absorption times (for checking purposes). k = kill
                        # Size: As many as number of absorptions occurred
        self.pkt = []   # Particle number associated to each absorption time (pkt = Particle Kill Time).
                        # Size: Same as the absorption times array
        self.ikt = []   # Array that keeps track of the iteration at which each absorption took place.
                        # Size: same as the absorption times array

        self.btimes = []# List of sorted absolute BLOCK times (for checking purposes).
                        # Size: As many as number of blocks occurred
        self.pbt = []   # Particle number associated to each BLOCK time (pbt = Particle Block Time).
                        # Size: Same as the blocking times array
        self.ibt = []   # Array that keeps track of the iteration at which each BLOCK takes place.
                        # Size: same as the block times array

        self.utimes = []# List of sorted absolute UNBLOCK times (for checking purposes).
                        # Size: As many as number of unblocks occurred
        self.put = []   # Particle number associated to each UNBLOCK time (put = Particle Unblock Time).
                        # Size: Same as the unblock times array
        self.iut = []   # Array that keeps track of the iteration at which each UNBLOCK takes place.
                        # Size: same as the unblock times array

        self.sk = [0]   # Times of an ABSORPTION (Killing) relative to ALL the activation times of the absorbed particle.
                        # The list is updated whenever an active particle is absorbed (i.e. touches position 0)
                        # GOAL: define the disjoint time segments on which the estimated survival probability is constant.
                        # SIZE AT THE END OF SIMULATION: as many as particles have been activated during the simulation.
        self.sbu = [0]  # Times of a BLOCK/UNBLOCK relative to the activation times of the particle.
                        # Size: As many elements as trajectories beginning at position 1 that come from particles
                        # becoming either blocked or unblocked.

        # Particle counts used in the computation of the estimated measures 
        self.counts_alive = [0]     # Number of particles that are alive in each time segment defined by sk
                                    # => have not been absorbed by position 0.
                                    # The time segment associated to the count has the time value in the sk array
                                    # as its lower bound.
                                    # Size: same as the times of ABSORPTION array
        self.counts_blocked = [0]   # Number of particles that are blocked in each time segment defined by sb
                                    # => have reached position K, the maximum capacity of the queue represented by the particle. 
                                    # Size: same as the times of BLOCK/UNBLOCK array

        # Times of last change
        self.time_last_change_of_system = 0.0

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
        pat: int
            Index of the particle to update, from 0 to N-1, where N is the number of particles in the system.
        
        Return: list
        A two-element list containing:
        - the time of the selected event, measured in absolute terms, i.e. from the common origin of all particles.
        - the change in position after applying the event, which can be -1, 0 or +1.
        """
        assert isinstance(p, int) and 0 <= p and p < self.N, \
                "The selected particle ({}) is valid (its index is an integer between 0 and {})".format(p, len(self.N)-1)
        
        # Find the event type to apply (whichever occurs first)
        earliest_event = self._order_times_next_events[p][0]
        #print("times_next_events[{}]: {}".format(pat, self.times_next_events[pat]))
        #print("earliest_event: {}".format(earliest_event))
        time_since_previous_event = self.times_next_events[p][earliest_event]
        time_of_event = self.particles[p].getTimeLastEvent() + time_since_previous_event

        # Update the particle's state by applying the event type just selected and record the position change
        position_change = self._apply_event(p, Event(earliest_event), time_of_event)
        latest_event = self._order_times_next_events[p][1]

        # Generate the next time for the event type just applied from the list of event times for this particle
        event_rate = self.particles[p].rates[earliest_event]
        self.times_next_events[p][earliest_event] = self.particles[p].generate_event_times(event_rate, size=1)

        # Update the time-to-event for the event type not applied so that it is now relative to the event just applied
        self.times_next_events[p][latest_event] -= time_since_previous_event

        # Update the order of the next event times
        self._order_times_next_events[p] = self._compute_order( self.times_next_events[p] )

        #print("UPDATED times_next_events[{}]: {}".format(pat, self.times_next_events[pat]))

        return time_of_event, position_change

    def _apply_event(self, p, event_to_apply, time_to_event):
        position_change = self.particles[p].apply_event(event_to_apply, time_to_event)
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
        The times of occurrence are measured w.r.atimes. the common origin for all particles.
        """
        self.iter = iter

        times_of_events = np.nan*np.ones(self.N)
        positions_change = np.zeros(self.N, dtype=int)
        for p in range(self.N):
            times_of_events[p], positions_change[p] = self._update_one_particle(p)
        if self.LOG:
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
            is_blocked = lambda p: positions_change[p] > 0 and self.positions[p] == self.particles[p].getCapacity()
            is_unblocked = lambda p: positions_change[p] < 0 and self.positions[p] == self.particles[p].getCapacity() + positions_change[p] 

            if is_activated(o):
                self._add_new_activation_time(o, t)

            elif is_absorbed(o):
                self._add_new_survival_time_segment(o, t)

            elif is_blocked(o):
                self._add_new_blocking_time_segment(o, t, EventType.BLOCK)
            elif is_unblocked(o):
                self._add_new_blocking_time_segment(o, t, EventType.UNBLOCK)

            assert len(self.active) ==  len(self.atimes) == len(self.pat) == len(self.iat) \
                    and                 len(self.btimes) == len(self.pbt) == len(self.ibt) \
                    and                 len(self.utimes) == len(self.put) == len(self.iut),\
                    "Iter {}: The length of arrays keeping track of the same type of information is the same" \
                    .format(self.iter)

            if self.LOG:
                with np.printoptions(precision=3, suppress=True):
                    particles, times_from_activation = self.get_all_elapsed_times()
                    print("event time: {:.3f} ({}), p={}, pos={}, active in the system (at different times): {}" \
                          "\n\ta: {}" \
                          "\n\tt: {}" \
                          "\n\te: {}" \
                          "\n\tp: {}" \
                          "\n\ts: {}" \
                          "\n\tb: {}" \
                          "\n\tcounts_alive: {}" \
                          "\n\tcounts_blocked: {}" \
                          .format(t, self.particles[o].getTypeLastEvent().name, o, self.positions[o], sum(self.active),
                                 self.active,
                                 np.array(self.atimes),
                                 np.array(times_from_activation),
                                 np.array(particles),
                                 np.array(self.sk),
                                 np.array(self.sbu),
                                 self.counts_alive,
                                 self.counts_blocked))

    def _add_new_activation_time(self, p, time_of_event):
        "Adds a new time to the array of activation times"
        # Add this time to the list of activation times and to the list of active flags
        # NOTE: we need to EXPLICITLY insert the new time IN ORDER in the array,
        # despite the fact the time_of_event values are SORTED in the CURRENT call to this method,
        # because we do not know anything about the order of these time_of_event values in relation to
        # the time values ALREADY PRESENT in the array.
        idx_insort = self._insert_new_time_in_order(p, time_of_event, self.pat, self.atimes, self.iat)
        self.active.insert(idx_insort, True)
        if self.LOG:
            with np.printoptions(precision=3, suppress=True):
                print("\n>>>>> ACTIVATION: p={}: a={:.3f}, time_of_event={}".format(p, time_of_event, np.array(self.atimes)))

    def _add_new_survival_time_segment(self, p, time_of_event):
        "Adds a new segment-start time to the array of survival time segments"
        # Insert the current time in the array of absorption times
        self._insert_new_time_in_order(p, time_of_event, self.pkt, self.ktimes, self.ikt)

        # Insert the current time in the array of time-segment starts and update counts of each segment accordingly
        self._insert_new_time(p, time_of_event, EventType.ABSORPTION)

    def _add_new_blocking_time_segment(self, p, time_of_event, type_of_event):
        """
        Adds a new block/unblock time in the respective absolute time array
        and a segment-start time to the array of block/unblock time segments.
        """
        #-- Insert the current time in the array of blocking times
        if type_of_event == EventType.BLOCK:
            self._insert_new_time_in_order(p, time_of_event, self.pbt, self.btimes, self.ibt)
        elif type_of_event == EventType.UNBLOCK:
            self._insert_new_time_in_order(p, time_of_event, self.put, self.utimes, self.iut)

        # Insert the current time in the array of time-segment starts and update counts of each segment accordingly
        self._insert_new_time(p, time_of_event, type_of_event)

    def _insert_new_time(self, p, time_of_event, type_of_event):
        idx_activation_times, activation_times = self.get_activation_times(p)
        times_since_activation = [time_of_event - t for t in activation_times]
        if self.LOG:
            with np.printoptions(precision=3, suppress=True):
                print("\nXXXXXXXXXXXXXXX {}: p={}: time_of_event={:.3f}, a: {}, s: {}" \
                      .format(type_of_event.name.upper(), p, time_of_event, np.array(activation_times), np.array(times_since_activation)))

        for idx_activation_time, s in zip(idx_activation_times, times_since_activation):
            if type_of_event == EventType.ABSORPTION:
                idx_insort = self.insort(self.sk, s)
                self.update_counts_alive(idx_insort)
                # De-activate the trajectory associated to the absorption
                self.active[idx_activation_time] = False
            elif type_of_event == EventType.BLOCK or type_of_event == EventType.UNBLOCK:
                idx_insort = self.insort(self.sbu, s)
                self.update_counts_blocked(idx_insort, type_of_event)
            else:
                raise ValueError("The type of event is invalid: {}".format(type_of_event))

    def _insert_new_time_in_order(self, p, time, lparticles, ltimes, liter):
        """
        Inserts new elements in the lists related to absolute time storage.
        
        Arguments:
        p: int
            Index of the particle affected by the insert.
        time: positive float
            Time to insert.
        
        lparticles: list
            List containing the particle numbers associated to each time in `ltimes` where `p` is inserted.
            
        ltimes: list
            List containing the times where `time` is inserted in order.

        liter: list
            List containing the simulation iteration numbers at which the insert takes place.
        """
        # Main update: insert new element in array of times and in array of particles associated to those times
        idx_insort = self.insort(ltimes, time)
        lparticles.insert(idx_insort, p)

        # Aux update: insert new element in array of iteration number at which the insert happens
        liter.insert(idx_insort, self.iter)

        return idx_insort

    def insort(self, ltarget, s):
        """
        Inserts a value in a list in order and returns the index of insertion
        (something that the bisect.insort() function does not do)
         """
        idx = bisect.bisect(ltarget, s)  # Index where to insert the new relative time in order
        ltarget.insert(idx, s)           # Insert the new relative time in order
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

    def update_counts_blocked(self, idx_to_insert, type_of_event):
        """
        Updates the counts of blocked particles in each time segment where the count can change
        following the blocking or unblocking of a new particle.
        """
        assert 1 <= idx_to_insert <= len(self.counts_blocked), \
                "The index where a new counter needs to be inserted ({}) is at least 1" \
                " and at most the current number of elements in the counts_alive array ({})" \
                .format(idx_to_insert, len(self.counts_blocked))

        # Insert a new element in the array whose value is equal to the count of the segment before split
        self.counts_blocked.insert(idx_to_insert, self.counts_blocked[idx_to_insert - 1])

        # INCREASE/DECREASE by 1 ALL counts to the RIGHT of the insert index
        # (because the block/unblock indicates that all time segments that are larger
        # than the elapsed time to block/unblock should count the change of blocking status of the particle)
        if type_of_event == EventType.BLOCK:
            delta_count = +1
        elif type_of_event == EventType.UNBLOCK:
            delta_count = -1
        for idx in range(idx_to_insert, len(self.counts_blocked)):
            self.counts_blocked[idx] += delta_count

    def finalize(self):
        "Finalize the statistics based on the particles still active"
        self.iter += 1
        for i, active_particle in enumerate(self.active):
            if active_particle:
                # => We add the time segment corresponding to the NEXT event happening for each active particle
                # Note that this time is fully acknowledgeable as if the particle was active at the latest event
                # then it will still be active just before the next event takes place, which may be considered
                # as the last time we observe the particle.
                p = self.pat[i]
                earliest_event = self._order_times_next_events[p][0]
                time_of_next_event = self.particles[p].getTimeLastEvent() + self.times_next_events[p][earliest_event]
                self._add_new_survival_time_segment(p, time_of_next_event)

    def estimate_proba_absortion_given_position_one(self, n, t):
        assert len(self.counts_alive) == len(self.sk), \
                "The number of elements in the counts array ({})" \
                " is the same as the number of elements in the time segments array ({})" \
                .format(len(self.counts_alive), len(self.sk))
        return np.c_[self.sk,     self.counts_alive / len(self.sk)]

    #-------------- Getters
    def get_number_active_particles(self):
        return sum(self.active)

    def get_all_activation_times(self):
        return self.pat, self.atimes

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
            particles += [self.pat[idx] for idx in idx_activation_times_p]
            times_from_activation += [t - u for u in activation_times_p]

        order = self._compute_order(times_from_activation)
        return [particles[o] for o in order], [times_from_activation[o] for o in order]

    def get_activation_times(self, p):
        """
        Returns the activation times for all ACTIVE trajectories of particle pat,
        that is for all sub-trajectories of the trajectory of particle pat that start at position 1.
        
        Return: list
        Two-element list containing:
        - a list with the indices in the activation times array from where the activation times are taken
        - a list with the activation times 
        """
        idx_activation_times_p = [idx for idx in find(self.pat, p) if self.active[idx]]
        activation_times_p = [self.atimes[i] for i in idx_activation_times_p]        
        return idx_activation_times_p, activation_times_p

    def get_absorption_times(self, p):
        """
        Returns the absorption times (relative to the trajectory activation) for all trajectories
        historically observed for particle pat, that is for all sub-trajectories of all trajectories
        observed for particle pat that start at position 1.
        
        Return: list
        Two-element list containing:
        - a list with the particle numbers corresponding to each element in the absorption times array
        - a list with the absorption times of each sub-trajectory measured w.r.t. the sub-trajectory activation
        """
        idx_absorption_times_p = [idx for idx in find(self.pkt, p)]
        absorption_times_p = [self.ktimes[i] for i in idx_absorption_times_p]
        return idx_absorption_times_p, absorption_times_p

    def get_survival_time_segments(self):
        return self.sk

    def get_blocking_time_segments(self):
        return self.sbu

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
        
        def update_render_string_with_event(t, iter, p, char):
            global render_str;
            global activation_times_shown;
            global iter_width;
            idx_activation_times_for_event = [idx for idx in find(self.pat, p)
                                              if self.atimes[idx] < t and not activation_times_shown[idx]]
            activation_times_for_event = [self.atimes[idx] for idx in idx_activation_times_for_event]
            #with np.printoptions(precision=3):
            #    print("iter: {}".format(iter))
            #    print("\tp: {} ({})".format(p, char))
            #    print("\tt: {:.3f}".format(t))
            #    print("\tatimes: {}".format(np.array(self.atimes)))
            #    print("\tpat: {}".format(self.pat))
            #    print("\tatimes_shown: {}".format(activation_times_shown))
            #    print("\tidx_activation_times: {}".format(idx_activation_times_for_event))
            #    print("\tactivation_times: {}".format(np.array(activation_times_for_event)))
            str_relative_event_times = ",".join(["{:.3f}".format(t - u) for u in activation_times_for_event])

            render_str += "it={:{width}}: [{}] ".format(iter, char, width=iter_width) + "{:.3f}".format(t) \
                            + " ({};{})\n".format(p, str_relative_event_times)
            
            return idx_activation_times_for_event

        #--------------------------------------------------------------------------------------
        all_types = [1]*len(self.atimes) + [0]*len(self.ktimes) + [9]*len(self.btimes) + [-9]*len(self.utimes)
        all_particles = self.pat + self.pkt + self.pbt + self.put
        all_times = self.atimes + self.ktimes + self.btimes + self.utimes
        all_iters = self.iat + self.ikt + self.ibt + self.iut

        all_order = np.argsort( np.abs(all_times) )

        all_types = [all_types[o] for o in all_order]
        all_particles = [all_particles[o] for o in all_order]
        all_times = [all_times[o] for o in all_order]
        all_iters = [all_iters[o] for o in all_order] 
        last_iter = max(all_iters)

        global render_str;
        global activation_times_shown;
        global iter_width;
        render_str = ""
        activation_times_shown = [False]*len(self.atimes)
        iter_width = len(str(self.iter))    # Display width for the iteration number so that all lines are aligned in the output string 
        for i, t in enumerate(all_times):
            if all_types[i] == 1:
                # ACTIVATION TIME
                render_str += "it={:{width}}: [A] ".format(all_iters[i], width=iter_width) + "{:.3f}".format(t) \
                                + " ({})\n".format(all_particles[i])
            elif all_types[i] == 0:
                # ABSORPTION TIME OR FINAL ITERATION
                if all_iters[i] == last_iter:
                    char = "F"  # Final iteration where we count also the non-absorbed particles by applying an absorption
                else:
                    char = "X"  # Actual absorption
                idx_activation_times_for_absorption = update_render_string_with_event(t, all_iters[i], all_particles[i], char)
                # Update the list of activation times already shown
                # (so that they are not shown again in future events!)
                for idx in idx_activation_times_for_absorption:
                    activation_times_shown[idx] = True
            elif all_types[i] == +9:
                # BLOCK TIME
                update_render_string_with_event(t, all_iters[i], all_particles[i], "B")
            elif all_types[i] == -9:
                # UNBLOCK TIME
                update_render_string_with_event(t, all_iters[i], all_particles[i], "U")

        return render_str

    #-------------- Helper functions
    def raiseErrorInvalidParticle(self, particle):
        raise ValueError("Wrong value for particle: {}\n".format(particle))
