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
import pandas as pd
from matplotlib import pyplot as plt, cm    # cm is for colormaps (e.g. cm.get_cmap())

from . import queues
from .queues import Event
from .utils.basic import find

@unique # Unique enumeration values (i.e. on the RHS of the equal sign)
class EventType(Enum):
    ACTIVATION = +1
    REACTIVATION = +2
    ABSORPTION = 0
    BLOCK = +9
    UNBLOCK = -9
    REMOVED = -99   # This is used to mark events that are removed from the statistics calculation
                    # when finalize_type = FinalizeType.REMOVED_CENSORED 

@unique # Unique enumeration values (i.e. on the RHS of the equal sign)
class FinalizeType(Enum):
    "Types of FINALIZE step of simulation to use in the finalize() method"
    ESTIMATE_CENSORED = 1
    REMOVE_CENSORED = 2
    ABSORB_CENSORED = 3

class EstimatorQueueBlockingFlemingViot:
    """Estimator of the blocking probability of queues using the Fleming Viot particle system
    
    Arguments:
    nparticles: int
        Number of particles to consider in the Fleming Viot system.

    niter: int
        Number of iterations on which the estimation is based.
        This is important for the finalization process carried out at the end of the simulation,
        so that absorbed particles at the last iteration are NOT re-activated but considered as
        absorbed, and thus included in the estimation of the expected survival time, regardless
        of the `finalize_type` parameter value.

    queue: subclass of GenericQueue
        The object should have the following attributes:
        - rates: a list with two elements corresponding to the birth and death events
        at the positions specified by the Event.BIRTH and Event.DEATH values of the Event Enum.

        The object should have the following methods implemented:
        - generate_birth_time()
        - generate_death_time()
        - generate_event_times()
        - getTimeLastEvent()
        - getTypeLastEvent()
        - apply_event()

    reactivate: (opt) bool
        Whether to reactivate a particle after absorption to a positive position.

    finalize_type: (opt) FinalizeType
        Indicates what type of finalization of the simulation should be done on the active particles, either:
        - FinalizeType.ESTIMATE_CENSORED
        - FinalizeType.REMOVE_CENSORED
        - FinalizeType.ABSORB_CENSORED
        of which only "remove" and "absorb" are currently implemented.

    seed: (opt) int
        Random seed to use for the random number generation by numpy.random.

    log: (opt) bool
        Whether to show messages of what is happening with the particles.
    """
    def __init__(self, nparticles, niter, queue,
                 reactivate=True, finalize_type=FinalizeType.REMOVE_CENSORED,
                 seed=None, log=False):
        self.N = nparticles
        self.niter = niter
        self.queue = queue                  # This variable is used to create the particles as replications of the given queue
        self.reactivate = reactivate        # Whether the particle is reactivated to a positive position after absorption
        self.finalize_type = finalize_type  # How to finalize the simulation
        self.seed = seed
        self.LOG = log

        self.reset()
        
    def reset(self):
        np.random.seed(self.seed)

        # Reset the iteration counter
        # (which starts at 1 when simulation runs,
        # and its value is set by the simulation process when calling update_state()) 
        self.iter = 0
        
        # Variable the keeps track whether the finalize process (at the end of the simulation)
        # has been carried out or not
        self.finalize_status = False

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
        self.times_next_events = [ [self.queue.generate_birth_time(), self.queue.generate_death_time()]
                                    for _ in np.arange(self.N) ]
        self._order_times_next_events = [ self._compute_order( self.times_next_events[p] )
                                            for p in np.arange(self.N) ]

        # Arrays that are used to estimate the expected survival time (time to killing from position 0)
        self.times0 = np.zeros(self.N, dtype=float)        # Latest times the particles changed to position 0
        self.ktimes0_sum = np.zeros(self.N, dtype=float)   # Times to absorption from latest time it changed to position 0
        self.ktimes0_n = np.zeros(self.N, dtype=int)       # Number of times the particles were absorbed

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
        
        Return: tuple
        The tuple contains:
        - the time of the selected event, measured in absolute terms, i.e. from the common origin of all particles.
        - the change in position after applying the event, which can be -1, 0 or +1.
        """
        assert self.isValidParticle(p), \
                "The selected particle ({}) is valid (its index is an integer between 0 and {})".format(p, len(self.N)-1)
        
        # Find the event type to apply (whichever occurs first)
        time_of_event, time_since_previous_event, earliest_event = self.get_time_next_event(p)

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
            # Update the particle's position based on the time and type of the next event
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
        for o in order_times_to_event:  # o represents a particle
            # Absolute time to event on the current particle
            t = times_of_events[o]

            # Possible status change of a particle 
            is_activated = lambda p: self.positions[p] == 1 # Note that a particle can be activated even if it was ALREADY at position 1 => a new sub-trajectory becomes active
            is_absorbed = lambda p: positions_change[p] < 0 and self.positions[p] == 0
            is_blocked = lambda p: positions_change[p] > 0 and self.positions[p] == self.particles[p].getCapacity()
            is_unblocked = lambda p: positions_change[p] < 0 and self.positions[p] == self.particles[p].getCapacity() + positions_change[p] 

            if is_activated(o):
                self._add_new_activation_time(o, t, EventType.ACTIVATION)

            elif is_absorbed(o):
                self._add_new_time_segment(o, t, EventType.ABSORPTION)
                self._update_survival_time_from_zero(o, t)
                if self.reactivate and not self.isLastIteration():
                    # Note: for the last iteration, we should NOT reactivate the particle
                    # so that we can consider the particle as absorbed (i.e. not censored time)
                    # for the calculation of the statistics (basically the expected survival time)
                    self._reactivate(o, t)

            elif is_blocked(o):
                self._add_new_time_segment(o, t, EventType.BLOCK)
            elif is_unblocked(o):
                self._add_new_time_segment(o, t, EventType.UNBLOCK)

            particles, times_from_activation = self.get_all_elapsed_times()
            assert len(self.active) ==  len(self.atimes) == len(self.pat) == len(self.iat) \
                    and                 len(self.ktimes) == len(self.pkt) == len(self.ikt) \
                    and                 len(self.btimes) == len(self.pbt) == len(self.ibt) \
                    and                 len(self.utimes) == len(self.put) == len(self.iut),\
                    "Iter {}: The length of arrays keeping track of the same type of information is the same" \
                    .format(self.iter)

            if self.LOG:
                with np.printoptions(precision=3, suppress=True):
                    print("event time: {:.3f} ({}), p={}, pos={}, active in the system (at different times): {}" \
                          "\n\ta: {}" \
                          "\n\tt: {}" \
                          "\n\te: {}" \
                          "\n\tp: {}" \
                          "\n\ts: {}" \
                          "\n\tb: {}" \
                          "\n\tcounts_alive: {}" \
                          "\n\tcounts_blocked: {}" \
                          "\n\tsurvival_time_from_zero: {}" \
                          .format(t, self.particles[o].getTypeLastEvent().name, o, self.positions[o], sum(self.active),
                                 self.active,
                                 np.array(self.atimes),
                                 np.array(times_from_activation),
                                 np.array(self.pat),
                                 np.array(self.sk),
                                 np.array(self.sbu),
                                 self.counts_alive,
                                 self.counts_blocked,
                                 self.ktimes0_sum))

            assert sum(np.array(times_from_activation) < 0) == 0 \
                    and sum(np.array(self.atimes) < 0) == 0 and sum(np.array(self.ktimes) < 0) == 0 \
                    and sum(np.array(self.btimes) < 0) == 0 and sum(np.array(self.utimes) < 0) == 0, \
                    "Iter {}: All times are non-negative" \
                    .format(self.iter)

    def _add_new_activation_time(self, p, time_of_event, type_of_event):
        "Adds a new time to the array of activation times"
        # Add this time to the list of activation times and to the list of active flags
        # NOTE: we need to EXPLICITLY insert the new time IN ORDER in the array,
        # despite the fact the time_of_event values are SORTED in the CURRENT call to this method,
        # because we do not know anything about the order of these time_of_event values in relation to
        # the time values ALREADY PRESENT in the array.
        idx_insort = self._insert_new_time_in_order(p, time_of_event, EventType.ACTIVATION)
        self.active.insert(idx_insort, True)
        if self.LOG:
            with np.printoptions(precision=3, suppress=True):
                print("\n>>>>> {}: iter={}, p={}: time_of_activation={:.3f}:" \
                        "\n\tall_activation_times={}" \
                        "\n\tall_particles={}" \
                        "\n\tall_active={}" \
                      .format(type_of_event.name.upper(), self.iter, p, time_of_event,
                              np.array(self.atimes), self.pat, self.active))

    def _update_survival_time_from_zero(self, p, time_of_event):
        # Refer the survival time to the latest time the particle was at position 0
        # and add it to the previously measured survival time
        self.ktimes0_sum[p] += time_of_event - self.times0[p]
        self.ktimes0_n[p] += 1

        if False:
            with np.printoptions(precision=3, suppress=True):
                print(">>>> Last time at 0: p={}, {:.3f}".format(p, self.times0[p]))
                print(">>>> Total Survival times: p={}, {}".format(p, np.array(self.ktimes0_sum)))
                print(">>>> Total Survival measures: p={}, {}".format(p, np.array(self.ktimes0_n)))

        # Update the latest time the particle was at position 0
        self.times0[p] = time_of_event

        if False:
            print(">>>> UPDATED Last time at 0: p={}, {:.3f}".format(p, self.times0[p]))

    def _add_new_time_segment(self, p, time_of_event, type_of_event):
        """
        Adds a new time in the absolute time array of the given event type
        and a segment-start time of the given event type
        to the array of corresponding time segments.
        
        Valid event types are absorption/block/unblock.
        """
        #-- Insert the current time in the array of blocking times
        self._insert_new_time_in_order(p, time_of_event, type_of_event)

        # Insert the current time in the array of time-segment starts and update counts of each segment accordingly
        self._insert_new_time(p, time_of_event, type_of_event)

    def _insert_new_time_in_order(self, p, time, type_of_event):
        """
        Inserts new elements in the lists related to absolute time storage.
        
        Arguments:
        p: int
            Index of the particle affected by the insert.
        time: positive float
            Time to insert.
        type_of_event: EventType
            Type of the event associated to the time to insert.
            
        Return:
        The index of the insertion in the lists that contain information about the events
        of the given type for the given particle.
        """
        ltimes, lparticles, liter, _ = self.get_lists_with_event_type_info(type_of_event)

        idx_insort = self.insort(ltimes, time)
        lparticles.insert(idx_insort, p)
        liter.insert(idx_insort, self.iter)

        return idx_insort

    def _insert_new_time(self, p, time_of_event, type_of_event):
        idx_activation_times, activation_times = self.get_activation_times(p)
        times_since_activation = [time_of_event - u for u in activation_times]
        if self.LOG:
            with np.printoptions(precision=3, suppress=True):
                print("\nXXXXXXXXXXXXXXX {}: iter={}, p={}: time_of_event={:.3f}, a: {}, s: {}" \
                      .format(type_of_event.name.upper(), self.iter, p, time_of_event, np.array(activation_times), np.array(times_since_activation)))

        self._insert_new_relative_times(idx_activation_times, times_since_activation, type_of_event)

    def _insert_new_relative_times(self, idx_activation_times, times_since_activation, type_of_event):
        for idx_activation_time, s in zip(idx_activation_times, times_since_activation):
            if type_of_event == EventType.ABSORPTION:
                idx_insort = self.insort(self.sk, s)
                self._update_counts_alive(idx_insort)
                # De-activate the trajectory associated to the absorption
                self.active[idx_activation_time] = False
            elif type_of_event == EventType.BLOCK or type_of_event == EventType.UNBLOCK:
                idx_insort = self.insort(self.sbu, s)
                self._update_counts_blocked(idx_insort, type_of_event)
            else:
                raise ValueError("The type of event is invalid: {}".format(type_of_event))

    def insort(self, ltarget, t):
        """
        Inserts a value in a list in order and returns the index of insertion
        (something that the bisect.insort() function does not do)
         """
        idx = bisect.bisect(ltarget, t)  # Index where to insert the new time in order
        ltarget.insert(idx, t)           # Insert the new time in order
        return idx

    def _update_counts_alive(self, idx_to_insert):
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

    def _update_counts_blocked(self, idx_to_insert, type_of_event):
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

    def _reactivate(self, p, t):
        """
        Reactivate a particle at the given time to an active particle chosen at random
        
        Arguments:
        p: int
            Particle number to reactivate.

        t: float
            Absolute time of absorption of the particle.

        Return: int
        The assigned particle number.
        """
        active_particles = self.get_active_particles()

        # Keep just the active particles that have at least an activation time that is SMALLER
        # than the current time t of the particle that we are re-activating.
        # Otherwise, the particle would be re-activated to a particle that at the moment (t)
        # has not yet been activated.
        # If no particles satisfy this condition, then the particle to re-activate is NOT re-activated.
        eligible_active_particles = []
        for q in active_particles:
            _, activation_times_q = self.get_activation_times(q)
            # Compare the minimum activation time
            # (which is at position 0 since the activation times returned by the method are sorted)
            if activation_times_q[0] < t:
                eligible_active_particles += [q]

        if len(eligible_active_particles) > 0:
            p_assigned = eligible_active_particles[ np.random.randint(0, len(eligible_active_particles)) ]

            # Assign to this particle ALL the activation times for the selected particle that are SMALLER
            # than the current particle's time.
            # I.e. we assume that the particle has the same sub-trajectories of the
            # selected particle starting at position 1, as long as they were activated before
            # the particle's time t.
            _, activation_times_for_assigned_particle = self.get_activation_times(p_assigned)
            if self.LOG:
                print("Eligible active particles: {}".format(eligible_active_particles))
                with np.printoptions(precision=3, suppress=True):
                    print("Valid activation times for selected particle (p_assigned={}): {}" \
                          .format(p_assigned, np.array([atime for atime in activation_times_for_assigned_particle if atime < t])))
            for atime in activation_times_for_assigned_particle:
                if atime < t:
                    self._add_new_activation_time(p, (1+1E-6)*atime, EventType.REACTIVATION)
                        # We insert a time that is a little larger than the existing time,
                        # so that the value is inserted AFTER the pre-existing time
                        # making "repeated" times be sorted according to the iteration number
                        # when they were inserted.
                    # Set the latest time the particle was at position 0 to the time
                    # the assigned particle was at position 0
                    # DM-2020/06/14: NOO!! We should NOT do this because we are NOT counting the
                    # blocking periods of the reactivated particles BEFORE reactivation.
                    # If we do so, the estimate of the survival time (from position 0)
                    # would be too large...
                    #self.times0[p] = self.times0[p_assigned] 
        else:
            p_assigned = None

        return p_assigned

    def compute_counts(self):
        """
        Computes the survival time segments and blocking time segments needed to
        compute the survival probability from position 1 and the probability of blocking
        for active particles. 
        """
       
        all_event_types =   [EventType.ABSORPTION]*len(self.ktimes) + \
                            [EventType.BLOCK]*len(self.btimes) + \
                            [EventType.UNBLOCK]*len(self.utimes)
        all_event_particles = self.pkt + self.pbt + self.put
        all_event_times = self.ktimes + self.btimes + self.utimes
        all_event_iters = self.ikt + self.ibt + self.iut

        #with np.printoptions(precision=3, suppress=True):
        #    print("activation_times: {}".format(np.array(self.atimes)))
        #    print("particles: {}".format(self.pat))
        #    print("iters: {}".format(self.iat))

        #with np.printoptions(precision=3, suppress=True):
        #    print("BEFORE SORTING:")
        #    print("all_event_times: {}".format(np.array(all_event_times)))
        #    print("all_event_particles: {}".format(all_event_particles))
        #    print("all_event_types: {}".format(all_event_types))
        #    print("all_event_iters: {}".format(np.array(all_event_iters)))

        # Sort by iteration number, as this is the order in which the events were added
        all_events_order = np.argsort(all_event_iters)

        all_event_types = [all_event_types[o] for o in all_events_order]
        all_event_particles = [all_event_particles[o] for o in all_events_order]
        all_event_times = [all_event_times[o] for o in all_events_order]
        all_event_iters = [all_event_iters[o] for o in all_events_order] 
        last_iter = max(all_event_iters)

        #with np.printoptions(precision=3, suppress=True):
        #    print("AFTER SORTING:")
        #    print("all_event_times: {}".format(np.array(all_event_times)))
        #    print("all_event_particles: {}".format(all_event_particles))
        #    print("all_event_types: {}".format(all_event_types))
        #    print("all_event_iters: {}".format(np.array(all_event_iters)))

        # Reset the time segments array in case they were already calculated
        # and make all particles active (i.e. not yet observed)        
        self.sk = [0.0]
        self.sbu = [0.0]
        self.counts_alive = [0]
        self.counts_blocked = [0]
        self.active = [True]*len(self.atimes)
        for idx, t in enumerate(all_event_times):
            type_of_event = all_event_types[idx]
            p = all_event_particles[idx]
            iter = all_event_iters[idx]

            # Get the activation times that are needed to compute the relative times to the event
            idx_activation_times, _ = self.get_activation_times(p)
                # NOTE: These indices are indices in the array of ALL activation times
                # not only in those retrieved by the get_activation_times() method.

            # Choose only the activation times that happened BEFORE the current time
            # AND at an iteration number that is EARLIER than the current iteration number
            # These two conditions guarantee that:
            # - the activation time is SMALLER than the current event time
            # - the activation time is NOT actually the activation time of a particle after RE-ACTIVATION
            # (which happened LATER than the currently analyzed time)
            idx_eligible_activation_times = []
            eligible_activation_times = []
            times_since_activation = []
            for i in idx_activation_times:
                if self.iat[i] < iter and self.atimes[i] < t:
                    idx_eligible_activation_times += [i]
                    eligible_activation_times += [self.atimes[i]]
                    times_since_activation += [t - self.atimes[i]]

            if self.LOG:
                with np.printoptions(precision=3, suppress=True):
                    print("\nXXXXXXXXXXXXXXX {}: iter={}, p={}: time_of_event={:.3f}, a: {}, s: {}" \
                          .format(type_of_event.name.upper(), iter, p, t, np.array(eligible_activation_times), np.array(times_since_activation)))

            self._insert_new_relative_times(idx_eligible_activation_times, times_since_activation, type_of_event)

        #with np.printoptions(precision=3, suppress=True):
        #    print("Relative absorption times: sk={}".format(np.array(self.sk)))
        #    print("Relative blocking times: sbu={}".format(np.array(self.sbu)))

    def finalize(self):
        """
        Finalize the simulation process by treating particles that are active,
        i.e. those particles providing CENSORED survival time values.
        
        The type of finalize process is defined by the object's attribute `finalize_type`.
        """
        self.iter += 1
        active_particles = self.get_active_particles()
        
        if self.LOG:
            if self.finalize_type == FinalizeType.ABSORB_CENSORED:
                finalize_process = "absorbing"
            elif self.finalize_type == FinalizeType.REMOVE_CENSORED:
                finalize_process = "removing"
            print("\n****** FINALIZING the simulation by {} the currently active particles... ******".format(finalize_process))

        if self.finalize_type == FinalizeType.ABSORB_CENSORED:
            # We absorb the active particle at the end of the simulation
            # MOTIVATION: get a lower bound as an estimate of the actual survival time.
            # PROBLEM: not all particles are likely to be absorbed soon (e.g. if the particle's position is large)
            for p in active_particles:
                self.finalize_absorb_particle(p)
        elif self.finalize_type == FinalizeType.REMOVE_CENSORED:
            # We remove all time records associated to the active particle that happen after the latest absorption.
            # MOTIVATION: the expected survival time (denominator for the blocking probability)
            # is not easily measured in the presence of censored observations
            # (although it is still possible as long as the largest censored time is smaller
            # than the largest absorption time observed).
            # PROBLEM: we lose information, which may not be so valuable if the number of active particles
            # is small compared to the number of particles that have been absorbed during the simulation.
            for p in active_particles:
                # We first mark for removal to avoid nightmares regarding updating list indices after removal!!
                self.finalize_mark_for_removal(p)
            self.finalize_remove()

        self.finalize_status = True

    def finalize_absorb_particle(self, p):
        """
        The given particle is absorbed using the next event time as absorption time,
        and the absorption time is recorded as a valid absorption time to be used in the calculation
        of the expected survival time
        """
        # Time to the next event that would have come
        time_next_event, _, _ = self.get_time_next_event(p)
      
        # Unblock particle if it is blocked
        if self.positions[p] == self.queue.getCapacity():
            self._add_new_time_segment(p, time_next_event, EventType.UNBLOCK)

        # Add the absorption
        self._add_new_time_segment(p, time_next_event, EventType.ABSORPTION)

        # Add the survival time information
        self._update_survival_time_from_zero(p, time_next_event)

    def finalize_mark_for_removal(self, p):
        "Remove all time records associated to the particle that happen after its latest absorption time"
        # TODO: (2020/06/14) Modify the program so that censored values are NOT removed but just marked as "NOT USE"
        # The most affected part is the calculation of the statistics for the blocking probability. 

        def mark_times_for_removal(p, idx_last_absorption_time, last_absorption_time, type_of_event):
            idx_times_p, _ = self.get_times_for_particle(p, type_of_event)
            ltimes, lparticles, liters, lactive = self.get_lists_with_event_type_info(type_of_event)

            is_activation_time_a_reactivation_at_latest_absorption = \
                lambda idx_time, idx_absorption_time: type_of_event == EventType.ACTIVATION and idx_last_absorption_time is not None and liters[idx_time] == self.ikt[idx_absorption_time]
            is_activation_time_after_latest_absorption = lambda idx_time: ltimes[idx_time] > last_absorption_time

            for idx_time in idx_times_p:
                if self.LOG:
                    print("\n{}".format(type_of_event))
                    if idx_last_absorption_time is not None:
                        with np.printoptions(precision=3, suppress=True):
                            print("\tp={}, idx_time={}, idx_last_absorb_time={}, last_absorb_time={:.3f}:" \
                                  "\n\tevent_time_analyzed: {}, compared to absorption time: {:.3f}" \
                                  "\n\titer_num_analyzed: {}, iter_num_absorption: {}" \
                                  .format(p, idx_time, idx_last_absorption_time, last_absorption_time,
                                          ltimes[idx_time], last_absorption_time,
                                          liters[idx_time], self.ikt[idx_last_absorption_time]))
                    print(pd.DataFrame({'activation_time': np.array(ltimes), 'particle': lparticles, 'iter': liters}))

                if  is_activation_time_a_reactivation_at_latest_absorption(idx_time, idx_last_absorption_time) or \
                    is_activation_time_after_latest_absorption(idx_time):
                    if self.LOG:
                        print(">>>>> MARKED FOR REMOVAL! p={}, idx_time={}, event_time={}" \
                              .format(p, idx_time, ltimes[idx_time]))
                    ltimes[idx_time] = EventType.REMOVED.value
                    lparticles[idx_time] = EventType.REMOVED.value
                    liters[idx_time] = EventType.REMOVED.value
                    if type_of_event == EventType.ACTIVATION:
                        lactive[idx_time] = EventType.REMOVED.value
                else:
                    if self.LOG:
                        print("***** NOT MARKED FOR REMOVAL! p={}, idx_time={}, event_time={}" \
                              .format(p, idx_time, ltimes[idx_time]))

        # Compute the latest absorption time for the particle
        idx_absorption_times_p, absorption_times_p = self.get_times_for_particle(p, EventType.ABSORPTION)
        try:
            assert sorted(absorption_times_p) == absorption_times_p, \
                    "The absorption times are sorted ({})".format(absorption_times_p)
            idx_last_absorption_time = idx_absorption_times_p[-1]
            last_absorption_time = absorption_times_p[-1]
        except:
            # The particle was never absorbed
            idx_last_absorption_time = None
            last_absorption_time = 0.0

        # Remove all absolute event times (activation/block/unblock)
        # that are larger than the latest absorption time for the particle
        # AND
        # all activation times that are associated to the iteration when
        # the latest absorption happened (which have activation time SMALLER than the latest absorption!) 
        if self.LOG:
            print("\tRemoving times happening after last absorption: p={}, idx_last_absorb_time={}, last_absorb_time={:.3f}" \
                  .format(p, idx_last_absorption_time, last_absorption_time))
            with np.printoptions(precision=3, suppress=True):
                idx_activation_times_p, activation_times_p = self.get_times_for_particle(p, EventType.ACTIVATION)
                _, block_times_p = self.get_times_for_particle(p, EventType.BLOCK)
                _, unblock_times_p = self.get_times_for_particle(p, EventType.UNBLOCK)
                print("\tBEFORE")
                print("\t\tActivation times: {}".format(np.array(activation_times_p)))
                print("\t\tActive?: {}".format([self.active[idx] for idx in idx_activation_times_p]))
                print("\t\tIter: {}".format([self.iat[idx] for idx in idx_activation_times_p]))
                print("\t\tAbsorption times: {}".format(np.array(self.ktimes)))
                print("\t\tParticles: {}".format(self.pkt))
                print("\t\tIter: {}".format(self.ikt))
                print("\t\tBlock times: {}".format(np.array(block_times_p)))
                print("\t\tUnblock times: {}".format(np.array(unblock_times_p)))
        mark_times_for_removal(p, idx_last_absorption_time, last_absorption_time, EventType.ACTIVATION)
        mark_times_for_removal(p, idx_last_absorption_time, last_absorption_time, EventType.BLOCK)
        mark_times_for_removal(p, idx_last_absorption_time, last_absorption_time, EventType.UNBLOCK)
        if self.LOG:
            with np.printoptions(precision=3, suppress=True):
                idx_activation_times_p, activation_times_p = self.get_times_for_particle(p, EventType.ACTIVATION)
                _, block_times_p = self.get_times_for_particle(p, EventType.BLOCK)
                _, unblock_times_p = self.get_times_for_particle(p, EventType.UNBLOCK)
                print("\tAFTER")
                print("\t\tActivation times: {}".format(np.array(activation_times_p)))
                print("\t\tActive?: {}".format([self.active[idx] for idx in idx_activation_times_p]))
                print("\t\tIter: {}".format([self.iat[idx] for idx in idx_activation_times_p]))
                print("\t\tAbsorption times: {}".format(np.array(self.ktimes)))
                print("\t\tParticles: {}".format(self.pkt))
                print("\t\tIter: {}".format(self.ikt))
                print("\t\tBlock times: {}".format(np.array(block_times_p)))
                print("\t\tUnblock times: {}".format(np.array(unblock_times_p)))
                print("")

        # NOTE: No action is needed for the lists containing the survival times record
        # as these times are not added for active particles, as long as they are not absorbed.

    def finalize_remove(self):
        "Removes the events that were marked for removal by the finalize_mark_for_removal(p) method"        

        def remove_events(type_of_event):
            def remove_marked_for_removal(alist):
                alist_orig = alist.copy()
                for element in alist_orig:
                    if element == EventType.REMOVED.value:
                        alist.remove(element)

            ltimes, lparticles, liters, lactive = self.get_lists_with_event_type_info(type_of_event)

            remove_marked_for_removal(ltimes)
            remove_marked_for_removal(lparticles)
            remove_marked_for_removal(liters)
            if type_of_event == EventType.ACTIVATION:
                remove_marked_for_removal(lactive)

        remove_events(EventType.ACTIVATION)
        remove_events(EventType.BLOCK)
        remove_events(EventType.UNBLOCK)

    def estimate_proba_survival_given_position_one(self):
        assert len(self.sk) > 1, "The length of the survival times array is at least 2 ({})".format(self.sk)
        assert len(self.counts_alive) == len(self.sk), \
                "The number of elements in the survival counts array ({})" \
                " is the same as the number of elements in the survival time segments array ({})" \
                .format(len(self.counts_alive), len(self.sk))

        self.proba_surv = [c / (len(self.sk)-1) for c in self.counts_alive]
        return pd.DataFrame.from_items([('t', self.sk), ('P(T>t / s=1)', self.proba_surv)])

    def estimate_proba_blocking_given_alive(self):
        assert len(self.counts_blocked) == len(self.sbu), \
                "The number of elements in the blocked counts array ({})" \
                " is the same as the number of elements in the blocking time segments array ({})" \
                .format(len(self.counts_blocked), len(self.sbu))
        
        if  len(self.sbu) > 1:
            self.proba_block = [c / (len(self.sbu)-1) for c in self.counts_blocked]
        else:
            self.proba_block = [0.0]

        return pd.DataFrame.from_items([('t', self.sbu), ('P(BLOCK / T>t,s=1)', self.proba_block)])

    def estimate_expected_survival_time_given_position_zero(self):
        # The following disregards the particles that were still alive when the simulation stopped
        # These measurements are censored, so we could improve the estimate of the expected value
        # by using the Kaplan-Meier estimator of the survival curve, that takes censoring into account.
        # However, note that the KM estimator (https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator)
        # assumes that the censoring times are deterministic (which may not be the case).
        # Also, from the above Wiki entry, it seems it considers only integer time values where S(t) is updated...??
        # My intuition is that we could divide the censored time by (1 - S(t)), where S(t) is the survival function
        # at the time of censoring t (which is available as long as S(t) was estimated, i.e. if t < at least one
        # other actual observed (non-censored) survival time.
        # In our case, the censoring time is NOT fixed as it depends on the next-event time simulated by the
        # exponential distribution for each of the N particles.
        # For me, the probability of censoring could be calculated for the last iteration of the simulation by:
        # P(censoring) = P(not dying at last iteration) = 1 - P(dying at last iteration)
        # P(dying at last iteration) = P(dying in general of the particle) = P(Death & position=1) =
        #     P(Death/position=1) * P(position=1) = P(Death < Birth) * P(position=1)
        # Now:
        # - P(position=1) would be the stationary probability for x=1 => ~ rho = lambda/mu
        # - P(Death < Birth) should be easily computed since Death and Birth are two independent exponential random variables
        #
        # SO WE COULD INVESTIGATE THIS!
        # (if needed to improve the estimation of the expected survival time, but I don't think will be necessary
        # as the censoring times are a VERY SMALL percentage if the simulation runs for a long time, so the effect
        # of their values is negligible!!)
        # 
        # Ref: (from publications available via inp-toulouse.fr)
        # Three methods for the estimation of the expected survival time (mean lifetime) with censoring observations: 
        # https://pdf-sciencedirectassets-com.gorgone.univ-toulouse.fr/273257/1-s2.0-S1572312705X00041/1-s2.0-S1572312704000309/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCXEaeh8EmEXM1GJ0AsPnDBoBRxxhD%2FRNMBZvlT89cGUQIgcCH1%2BLxWNKNBtkbfTcMI3q7js3xvf6DZC%2BmcYLAQ0WgqtAMIWhADGgwwNTkwMDM1NDY4NjUiDP3BNfcD2wIdDaF7JyqRA63Ej%2BdVb77iAoLsdnwyYCgwc4ooUNe%2BeHIpNFKWVG8IyWP17l2JSt%2FhAcBzMvmgm42UU7DiXNLupV6BLinQ6XHHiw1TJcJSwFkm3haZAdBx40UOkLvjedfmoLTTQ7sIXnsuPyCjqhnMmAs3Y5gCkFMrFcoquAad7YB72NCAX8ySrBORh4DgNWdd6OmHUA475tyoWMPtQ9Wb4TtPLW7415y7vsR82Yas8FgXvCvn2DH5xaJSaCBF90U8XD8c3a6AmBgKYbG7iUAWEvA9z0Tftor6QHztDFCZnyjQA%2FGgby1fcHA5FvgjgYOAoIjyvT%2BqKpC341LrVwnDigQxLNzazsBboO8x1KuAu8KSjX%2BEH662HJXpaZRqF22ufh879BhOwyvvKQwcIvJg%2BqFzx89I5L3dbwZhg8sneWRWcr7DqZSa%2BlHh7Q3Zc3gt4%2Fx2XWKpIn4EKkDLcYG12%2BH5c5XE8B4xbRPQLtTlGvUMctPMrR9YmRhssNHYJd%2BQBf%2BK2wAjBBylHgAWC6B9FDdALhFRUpOBMOeLjfcFOusBakizCxQRio4MQp3YHVpJPuRkzm5RuKsuLCVffL3FbbALxprtCRWWxZll3EkIE6wcHlreR%2FrKCe1Hm%2FPbeSa1U4uCflD2QKaS8CKV1um7mXhZgRpVQuupA453Jh3klzlclFKg6pabJwcjJbr%2BC1MtVBZJcJjmUkmMX98%2FuuQ2BYR6NR8pZ1Y6Y8vvmmIWYSl561kP44cTBZfKGjRrJMWHrtsjX9Y9bOxwGQkVqN4SCr5%2B8GP6%2FAoYT3aBrVxyfb2BAeH5RvA%2FIpnWzLdj0qbMgtV%2FtmXg4W%2BqgCESAuq2IRR0HPK2ucmwbQxXUQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200612T095802Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY34E3LPN3%2F20200612%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ebcd7148040e44a9ed31b0a67e08c4100ddf771f8a22fa79d9a5e9232ceae7ce&hash=e36d3366c70cb42bd700b8069e70049ff7709d3c7d834f05b64f4d59d4ea0b6b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1572312704000309&tid=spdf-93f8c8ba-b315-425c-b497-e0d86f6bb7c8&sid=c3f22fac2ce5f24a791b80d7da0081506107gxrqb&type=client
        # (saved to RL-002-QueueBlocking/200503-JournalStatisticalMethodology-Datta-MeanLifetimeEstimationWithCensoredData.pdf)  

        if self.finalize_type == FinalizeType.ESTIMATE_CENSORED:
            # Add an estimation of the survival time for the censored (i.e. still active) particles
            active_particles = self.get_active_particles()
            for p in active_particles:
                survival_time = self.particles[p].getTimeLastEvent() - self.times0[p]
                ### TODO: (2020/06/13) Write the get_survival_function() method that returns the estimate of S(t)
                self.ktimes0_sum[p] = survival_time / (1 - self.get_survival_function(survival_time))
                self.ktimes0_n[p] += 1
        else:
            if self.LOG:
                active_particles = self.get_active_particles()
                print("active particles: {}".format(active_particles))
                print("activation times: {}".format(self.atimes))
                print("all particles: {}".format(self.pat))
                print("active?: {}".format(self.active))
            assert sum(self.active) == 0, "The number of active particles is 0 ({})".format(sum(self.active))

        if np.sum(self.ktimes0_n) > 0 :
            expected_survival_time = np.sum(self.ktimes0_sum) / np.sum(self.ktimes0_n)
        else:
            # All particles are alive (censored), so we estimate the expected survival time
            # as the average time each particle survived so far.
            # This should be a VERY RARE SITUATION, because it means that absolutely NO particle
            # has been absorbed in the whole simulation process!
            expected_survival_time = np.mean([self.particles[p].getTimeLastEvent() for p in range(self.N)])

        return expected_survival_time

    def estimate_proba_blocking(self):
        # Insert times in the survival times array and the P(T>t/s=1) array
        # so that we can merge the two arrays consistently on time
        sk = self.sk.copy()
        proba_surv = self.proba_surv.copy()
        sbu = self.sbu.copy()
        proba_block = self.proba_block.copy()

        for s in self.sbu:
            if (s > 0):
                idx_insort = self.insort(sk, s)
                assert idx_insort > 0, "The index to insert the new time is larger than 0 ({})".format(idx_insort)
                proba_surv.insert(idx_insort, proba_surv[idx_insort-1])

        # Insert times in the blocking times array and the P(BLOCK / T>t, s=1) array
        for s in self.sk:
            if (s > 0):
                idx_insort = self.insort(sbu, s)
                assert idx_insort > 0, "The index to insert the new time is larger than 0 ({})".format(idx_insort)
                proba_block.insert(idx_insort, proba_block[idx_insort-1])

        assert len(proba_surv) == len(proba_block), \
                "The length of the survival probabilities array and the block probabilities array to multiply are the same ({}, {})" \
                .format(len(proba_surv), len(proba_block))

        # Integrate => Multiply the two quantities and sum
        integral = 0
        for i in range(len(proba_surv)):
            integral += proba_surv[i] * proba_block[i]
        if self.LOG:
            print("integral = {:.3f}".format(integral))

        # Estimate the blocking probability!
        self.proba_blocking = integral / self.estimate_expected_survival_time_given_position_zero()

        return pd.DataFrame.from_items([('t', sk), ('P(T>t / s=1)', proba_surv), ('P(BLOCK / T>t,s=1)', proba_block)])


    #-------------- Getters
    def get_active_particles(self):
        "Returns the list of particle numbers whose current position is > 0"
        return np.unique([self.pat[idx] for idx in range(len(self.active)) if self.active[idx]])

    def get_number_active_particles(self):
        return sum(self.active)

    def get_all_activation_times(self):
        return self.pat, self.atimes

    def get_all_absorption_times(self):
        return self.pkt, self.ktimes

    def get_all_elapsed_times(self):
        """
        Returns the elapsed time from activation of each ACTIVE sub-trajectory of each particle in the system
        
        Return: tuple
        The tuple contains:
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
            with np.printoptions(precision=3, suppress=True):
                assert np.sum([t < u for u in activation_times_p]) == 0, \
                        "All the activation times of particle {} ({}) are smaller than the time of the latest event ({:.3f})" \
                        .format(p, np.array(activation_times_p), t)

        order = self._compute_order(times_from_activation)
        return [particles[o] for o in order], [times_from_activation[o] for o in order]

    def get_activation_times(self, p):
        """
        Returns the activation times for all ACTIVE trajectories of particle p,
        that is for all sub-trajectories of the trajectory of particle p that start at position 1.
        
        Return: tuple
        The tuple contains:
        - a list with the indices in the activation times array from where the activation times are taken
        - a list with the activation times 
        """
        if self.isValidParticle(p):
            idx_activation_times_p = [idx for idx in find(self.pat, p) if self.active[idx]]
            activation_times_p = [self.atimes[i] for i in idx_activation_times_p]
    
            assert sorted(activation_times_p) == activation_times_p, \
                    "The returned activation times of particle {} are sorted ({}" \
                    .format(p, activation_times_p)
    
            return idx_activation_times_p, activation_times_p
        else:
            self.raiseWarningInvalidParticle(p)
            return [], []

    def get_times_for_particle(self, p, type_of_event):
        """
        Returns the absolute times for particle p associated to the given event type
        
        Return: tuple
        The tuple contains:
        - a list with the indices in the indexed attributes of the class (normally lists)
        that store information about the given event type that contain the times of the events
        for particle p.
        - a list with the event times for particle p.
        """
        if self.isValidParticle(p):
            ltimes, lparticles, _, _ = self.get_lists_with_event_type_info(type_of_event)
            idx_times_p = [idx for idx in find(lparticles, p)]
            times_p = [ltimes[i] for i in idx_times_p]
            return idx_times_p, times_p
        else:
            self.raiseWarningInvalidParticle(p)
            return [], []

    def get_survival_time_segments(self):
        return self.sk

    def get_blocking_time_segments(self):
        return self.sbu

    def get_all_total_blocking_time(self):
        "Returns the total blocking time for all particles"
        blocking_time = 0.0
        for p in range(self.N):
            blocking_time += self.get_total_blocking_time(p)
        return blocking_time

    def get_total_blocking_time(self, p):
        "Returns the total blocking time for a particle"
        df_blocking_periods = self.get_blocking_periods(p)
        return np.sum(df_blocking_periods['Unblock Time'] - df_blocking_periods['Block Time'])

    def get_blocking_periods(self, p):
        """
        Returns the blocking periods (block / unblock) for a particle
        
        Return: pandas DataFrame
        DataFrame with two columns:
        - 'Block Time': start time of blocking
        - 'Unblock Time': end time of blocking
        """
        if not self.finalize_status:
            raise Warning("The simulation has not been finalized..." \
                          "\nThe result of the total blocking time may be incorrect or give an error." \
                          "\nRun first the finalize() method and rerun")

        # Blocking periods
        _, block_times = self.get_times_for_particle(p, EventType.BLOCK)
        _, unblock_times = self.get_times_for_particle(p, EventType.UNBLOCK)

        assert len(block_times) == len(unblock_times), \
                "Particle {}: The number of blocking times ({}) is the same as the number of unblocking times ({})" \
                .format(p, block_times, unblock_times)

        df_blocking_times = pd.DataFrame.from_items([('Block Time', block_times), ('Unblock Time', unblock_times)])

        for r in df_blocking_times.index:
            assert df_blocking_times.loc[r]['Block Time'] < df_blocking_times.loc[r]['Unblock Time'], \
                    "The block time ({:.3f}) is always smaller than the unblock time ({:.3f})" \
                    .format(df_blocking_times.loc[r]['Block Time'], df_blocking_times.loc[r]['Unblock Time'])

        return df_blocking_times

    def get_all_total_survival_time(self):
        "Returns the total blocking time for all particles"
        survival_time = 0.0
        for p in range(self.N):
            df_survival_periods_p = self.get_survival_periods(p)
            survival_time += np.sum(df_survival_periods_p['Survival Period Span'])

        assert np.allclose(survival_time, np.sum(self.ktimes0_sum)), \
                "The sum of the survival period spans ({:.3f}) equals" \
                " the sum of the total survival times per particle computed during the simulation ({:.3f})" \
                .format(survival_time, np.sum(self.ktimes0_sum))

        return survival_time

    def get_total_survival_time(self, p):
        "Returns the total survival time for a particle (from position 0 back to position 0)"
        df_survival_periods = self.get_survival_periods(p)
        return np.sum(df_survival_periods['Survival Period Span'])

    def get_survival_periods(self, p):
        """
        Returns all the survival periods (period in which the particle goes from position 0
        back to position 0) for particle p.
        
        Return: tuple
        - list with the end time of each survival period
        - list with the time span of each survival period
        """
        _, absorption_times = self.get_times_for_particle(p, EventType.ABSORPTION)

        survival_times = []
        if len(absorption_times) > 0:
            survival_times = [absorption_times[0]]
            for idx, t in enumerate(absorption_times[1:]):
                survival_times += [t - absorption_times[idx]]

        assert len(survival_times) == len(absorption_times)
        assert np.all(np.array(survival_times) > 0), \
                "The survival period span is always positive ({})" \
                .format(survival_times)

        # Survival periods
        return pd.DataFrame.from_items([('Survival Period End', absorption_times), ('Survival Period Span', survival_times)])

    def get_survival_times(self):
        """
        Returns the survival times from position 0 for all particles
        
        Return: tuple
        The tuple contains information about the survival periods (i.e. the contiguous time
        during which a particle goes from position 0 back to position 0 for the first time)
        - a list with the number of survival periods for all particles
        - a list with the time span of the survival periods
        """
        return self.ktimes_n, self.ktimes_sum

    def get_type_last_event(self, p):
        try:
            return self.particles[p].getTypeLastEvent()
        except:
            self.raiseErrorInvalidParticle(p)
            return None

    def get_all_times_last_event(self):
        return [self.get_time_last_event(p) for p in range(self.N)]

    def get_time_last_event(self, p):
        try:
            return self.particles[p].getTimeLastEvent()
        except:
            self.raiseErrorInvalidParticle(p)
            return None

    def get_all_times_next_events(self):
        return self.times_next_events

    def get_time_next_event(self, p):
        """
        Returns the absolute time of the next event for the given particle, which is chosen
        as the earliest time of all possible next events the particle can undergo.
        
        Return: tuple
        The tuple contains:
        - the absolute time of the next event
        - the relative time of the next event w.r.t. the last event time
        - the index associated to the time of the next event from the list of all possible events for the particle
        """
        try:
            earliest_event = self._order_times_next_events[p][0]
            time_since_last_event = self.times_next_events[p][earliest_event]
            time_next_event = self.particles[p].getTimeLastEvent() + time_since_last_event
            return time_next_event, time_since_last_event, earliest_event
        except:
            self.raiseErrorInvalidParticle(p)
            return None, None, None

    def get_times_next_events(self, p):
        "Returns the times for all possible next events the particle can undergo" 
        try:
            return self.times_next_events[p]
        except:
            self.raiseErrorInvalidParticle(p)
            return None

    def get_time_last_change_of_system(self):
        return self.time_last_change_of_system

    def get_counts_particles_alive_by_elapsed_time(self):
        return self.counts_alive

    def get_counts_particles_blocked_by_elapsed_time(self):
        return self.counts_blocked

    def get_lists_with_event_type_info(self, type_of_event):
        "Returns the object's lists that contain information about the given event type"
        if type_of_event == EventType.ACTIVATION:
            ltimes = self.atimes
            lparticles = self.pat
            liters = self.iat
            lactive = self.active
        elif type_of_event == EventType.ABSORPTION:
            ltimes = self.ktimes
            lparticles = self.pkt
            liters = self.ikt
            lactive = None 
        elif type_of_event == EventType.BLOCK:
            ltimes = self.btimes
            lparticles = self.pbt
            liters = self.ibt
            lactive = None 
        elif type_of_event == EventType.UNBLOCK:
            ltimes = self.utimes
            lparticles = self.put
            liters = self.iut
            lactive = None 

        return ltimes, lparticles, liters, lactive

    def render(self):
        "Renders the simulation process in a semi-graphical way"

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
        all_types = [EventType.ACTIVATION.value]*len(self.atimes) + [EventType.ABSORPTION.value]*len(self.ktimes) + [EventType.BLOCK.value]*len(self.btimes) + [EventType.UNBLOCK.value]*len(self.utimes)
        all_particles = self.pat + self.pkt + self.pbt + self.put
        all_times = self.atimes + self.ktimes + self.btimes + self.utimes
        all_iters = self.iat + self.ikt + self.ibt + self.iut

        all_order = np.argsort( np.abs(all_times) )

        all_types = [all_types[o] for o in all_order]
        all_particles = [all_particles[o] for o in all_order]
        all_times = [all_times[o] for o in all_order]
        all_iters = [all_iters[o] for o in all_order] 
        last_iter = self.niter

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
    def setup(self):
        params_str = "***********************" \
                    "\nK = {}" \
                    "\nlambda = {:.3f}" \
                    "\nmu = {:.3f}" \
                    "\nrho = {:.3f}" \
                    "\nnparticles = {}" \
                    "\nreactivate = {}" \
                    "\nfinalize_type = {}" \
                    "\nniter = {}" \
                    "\nseed = {}" \
                    "\n***********************" \
                    .format(self.queue.getCapacity(),
                            self.queue.rates[Event.BIRTH.value], self.queue.rates[Event.DEATH.value],
                            self.queue.rates[Event.BIRTH.value] / self.queue.rates[Event.DEATH.value],
                            self.N, self.reactivate, self.finalize_type.name, self.niter, self.seed)
        return params_str

    def isLastIteration(self):
        return self.iter == self.niter

    def isValidParticle(self, p):
        return (isinstance(p, int) or isinstance(p, np.int32) or isinstance(p, np.int64)) and 0 <= p < self.N

    def raiseErrorInvalidParticle(self, p):
        raise ValueError("Wrong particle number: {}. Valid values are integers between 0 and {}.\n".format(p, self.N-1))

    def raiseWarningInvalidParticle(self, p):
        raise Warning("Wrong particle number: {}. Valid values are integers between 0 and {}.\n".format(p, self.N-1))
