# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:12:23 2020

@author: Daniel Mastropietro
@description: Estimators are defined for problems of interest in Reinforcement Learning.
"""

from enum import Enum, unique

import copy     # Used to generate different instances of the queue under analysis
import bisect

import numpy as np
from numpy import printoptions
import pandas as pd
from matplotlib import pyplot as plt, cm    # cm is for colormaps (e.g. cm.get_cmap())

from . import queues
from .queues import Event, GenericQueue
from .utils.basic import find, find_first, find_last, insort, merge_values_in_time

@unique # Unique enumeration values (i.e. on the RHS of the equal sign)
class EventType(Enum):
    ACTIVATION = +1
    REACTIVATION = +2
    ABSORPTION = 0
    BLOCK = +9
    UNBLOCK = -9
    CENSORING = +99 # To mark an event as censoring, i.e. for censored observations a new event of this type
                    # is added to the event history of the particle when using FinalizeType.ESTIMATE_CENSORED.  
    REMOVED = -99   # This is used to mark events that are removed from the statistics calculation
                    # when finalize_type = FinalizeType.REMOVED_CENSORED 

@unique # Unique enumeration values (i.e. on the RHS of the equal sign)
class FinalizeType(Enum):
    "Types of FINALIZE step of simulation to use in the finalize() method"
    ESTIMATE_CENSORED = 1
    REMOVE_CENSORED = 2
    ABSORB_CENSORED = 3

@unique
class RenderType(Enum):
    "Types of rendering the evolution of particles in the simulated process"
    GRAPH = 1
    TEXT = 0

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
    def __init__(self, nparticles: int, niter: int, queue: GenericQueue,
                 reactivate=True, finalize_type=FinalizeType.REMOVE_CENSORED, render_type=RenderType.GRAPH,
                 seed=None, log=False):
        if reactivate and nparticles < 2:
            raise ValueError("The number of particles must be at least 2 when reactivate=True ({})".format(nparticles))
            import sys
            sys.exit(-1)
        self.N = nparticles
        self.niter = niter
        self.queue = queue                  # This variable is used to create the particles as replications of the given queue

        if queue.size not in [0, 1]:
            raise ValueError("The start position of the particles must be either 0 or 1 ({})".format(queue.size))
        if queue.getCapacity() <= 1:
            # This condition must be satisfied because ONLY ONE EVENT is accepted for each event time
            # i.e. ACTIVATION, BLOCK, UNBLOCK or ABSORPTION.
            # If we allowed a queue capacity of 1, when the position of the particle goes to 1, the particle
            # is both ACTIVATED and BLOCKED, a situation that cannot be handled with the code as is written now.
            # In any case, a capacity of 1 is NOT of interest.   
            raise ValueError("The maximum position of the particles must be larger than 1 ({})".format(queue.getCapacity()))
        self.START = queue.size             # Initial start position for each particle

        if mean_lifetime is not None and (np.isnan(mean_lifetime) or mean_lifetime <= 0):
            raise ValueError("The mean life time must be positive ({})".format(mean_lifetime))
        self.mean_lifetime = mean_lifetime  # A possibly separate estimation of the expected survival time

        self.reactivate = reactivate        # Whether the particle is reactivated to a positive position after absorption
        self.finalize_type = finalize_type  # How to finalize the simulation
        self.render_type = render_type
        self.seed = seed
        self.LOG = log

        # Epsilon time --> used to avoid repeition of times when reactivating a particle
        # This epsilon time takes into account the scale of the problem by making it 1E6 times smaller
        # than the minimum of the event rates.
        # I.e. we cannot simply set it to 1E-6... what if the event rate IS 1E-6??
        self.EPSILON_TIME = 1E-6 * np.min([ self.queue.rates[Event.BIRTH.value], self.queue.rates[Event.DEATH.value] ])

        self.reset()
        
    def reset(self):
        # NAMING CONVENTIONS:
        # - We use identifiers `P`, `Q` and `p`, `q` respectively as follows:
        #   - UPPERCASE: particle number from 0 to N-1 
        #   - lowercase: particle ID from 0 to M >= N-1, which indexes the info_particles list
        #                of original and reactivated particles.
        #   Both lowercase and UPPERCASE indices have the same variation range when no reactivation is carried out.
        # - We use identifiers `E` and `e` as follows:
        #   - `E`: (uppercase E) event of interest for the calculation of statistics,
        #          needed for the blocking probability, such as ACTIVATION, ABSORPTION, etc.,
        #          namely those listed in the EventTypes Enum class.
        #   - `e`: (lowercase e) event of the types valid in the Event enum defined for queues
        #          (e.g. BIRTH and DEATH)
        #

        np.random.seed(self.seed)

        # Reset the iteration counter
        # (which starts at 1 when simulation runs,
        # and its value is set by the simulation process when calling update_state()) 
        self.iter = 0
        
        # Variable the keeps track whether the finalize process (at the end of the simulation)
        # has been carried out or not
        self.is_simulation_finalized = False

        # Reset results of the simulation
        self.t = None                           # List containing the times at which the probabilities by time are estimated
        self.proba_surv_by_t = None             # List the same length as t containing S(t / s=1) = P(T>t / s=1) 
        self.proba_block_by_t = None            # List the same length as t containing P(BLOCK / T>t,s=1)
        self.expected_survival_time = np.nan    # Expected survival time starting from s=0
        self.proba_blocking = np.nan            # Blocking probability (the ultimate goal of this simulation!)
                                                # It's computed as Integral( proba_surv_by_t * proba_block_by_t ) / expected_survival_time

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

        #-- Attributes used in the SIMPLER implementation
        # Trajectories of each particle (always N in total)
        self.trajectories = []
        # Information about the particles used to compute the statistics needed to estimate the blocking probability
        self.info_particles = []
        assert self.START in [0, 1], "The start position for the particle is 0 or 1 ({})".format(self.START)
        if self.START == 0:
            start_event_type = EventType.ABSORPTION
        elif self.START == 1:
            start_event_type = EventType.ACTIVATION
        for P in range(self.N):
            self.trajectories += [ dict({
                                           't': [0.0],          # event times at which the particle changes position
                                           'x': [self.START],   # positions taken by the particle after each event
                                           'e': [Event.RESET]   # events (birth, death) associated to the event times
                                           }) ]
            self.info_particles += [ dict({
                                           't': [0.0],                  # times at which the events of interest take place
                                           'E': [start_event_type],     # events of interest associated to the times 't'
                                                                        # (the events of interest are those affecting the
                                                                        # estimation of the blocking probability:
                                                                        # activation, absorption, block, unblock)
                                           't0': None,                  # Absorption /reactivation time
                                           'x': None,                   # Position after reactivation
                                           'iter': None,                # Iteration number of absorption / reactivation
                                           'particle number': P,        # Source particle (queue) from which reactivation happened (only used when reactivate=True)
                                           'reactivated number': None,  # Reactivated particle (queue) to which the source particle was reactivated (only used when reactivate=True)
                                           'reactivated ID': None       # Reactivated particle ID (index of this info_particle list) 
                                           }) ]
            if self.START == 0:
                # Remove the first event added above when particles start at 0
                # in order to avoid failing the assertions stating that the first
                # event in the info_particles dictionaries is an ACTIVATION. 
                self.info_particles[P]['t'].pop(0)
                self.info_particles[P]['E'].pop(0)
        # List stating whether the corresponding particle number has been absorbed already at least once
        # This is used when updating the dict_info_absorption_times dictionary, as this should contain
        # only one absorption time per particle.
        self.particle_already_absorbed = [False]*self.N
        # List of reactivation IDs which are the indices in the info_particles list
        # where we should look for the latest information (relevant events and their times)
        # about each particle.
        self.particle_reactivation_ids = list(range(self.N))

        # Information about the absorption times, which is used for reactivating an absorbed particle
        self.dict_info_absorption_times = dict({'t': [],
                                                'P': [],
                                                'iter': []})

        # Active particle numbers at the end of the simulation (it may contain values between 0 and N-1 
        self.is_particle_active = [False]*self.N
        #-- Attributes used in the SIMPLER implementation


        # Positions or state of each particle
        self.positions = np.zeros(self.N, dtype=int)

        # History of all positions and event times for each particle 
        self.all_positions = []
        self.all_times = []
        for P in range(self.N):
            self.all_positions += [ [self.START] ]
            self.all_times += [ [0.0] ]

        # 2) Compute the times of the next birth and death for each queue/particle, and their order
        self.times_next_events = [ [self.queue.generate_birth_time(), self.queue.generate_death_time()]
                                    for _ in np.arange(self.N) ]
        self._order_times_next_events = [ self._compute_order( self.times_next_events[p] )
                                            for p in np.arange(self.N) ]

        # Arrays that are used to estimate the expected survival time (time to killing from position 0)
        if self.START == 0:
            # Latest times the particles changed to position 0 is 0.0 as they start at position 0
            self.times0 = np.zeros(self.N, dtype=float)
        else:
            # Latest times the particles changed to position 0 is unknown as they don't start at position 0
            self.times0 = np.nan * np.ones(self.N, dtype=float)
        self.ktimes0_sum = np.zeros(self.N, dtype=float)   # Times to absorption from latest time it changed to position 0
        self.ktimes0_n = np.zeros(self.N, dtype=int)       # Number of times the particles were absorbed

        # 3) Attributes used directly in the computation of the estimated measures that contribute to the probability of blocking.
        # TODO: (2020/06/16) Store the information on each particle in separate lists (FOR SPEED UP!)
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
        # Note that these initial values are ALWAYS 0, regardless of the initial position of the particles
        # (whether 0 or 1). For more details see the comments in the compute_counts() method.
        self.counts_alive = [0] # Number of particles that are alive in each time segment defined by sk
                                # => have not been absorbed by position 0.
                                # The time segment associated to the count has the time value in the sk array
                                # as its lower bound.
                                # Size: same as the times of ABSORPTION array
        self.counts_blocked = [0]# Number of particles that are blocked in each time segment defined by sb
                                # => have reached position K, the maximum capacity of the queue represented by the particle. 
                                # Size: same as the times of BLOCK/UNBLOCK array

        # Times of last change
        self.time_last_change_of_system = 0.0

        if self.LOG:
            print("Particle system with {} particles has been reset:".format(self.N))
            print("")

    def simulate(self):
        self.reset()
        self.generate_trajectories()
        #self.estimate_proba_blocking()

    def generate_trajectories(self):
        # IMPORTANT: We start at iteration 1 because we consider 0 the iteration for the starting positions
        for it in range(1, self.niter+1):
            self.iter = it
            for P in range(self.N):
                time_of_event, position_change = self._update_one_particle(P)
                assert -1 <= position_change <= 1, \
                        "The change of position of particle P={} is -1, 0, or +1 ({})" \
                        .format(P, position_change)
                self._update_trajectories(P)
                self._update_info_particles(P, position_change)
                if not self.reactivate and self._is_particle_absorbed(P, position_change) or \
                   self.reactivate and self._is_particle_absorbed(P, position_change) and not self.particle_already_absorbed[P]:
                    self.particle_already_absorbed[P] = True
                    self._update_info_absorption_times(P, time_of_event, it)
                    self._update_survival_time_from_zero(P, time_of_event)

                # TODO: (2020/06/22) Add assertions about the information stored in the above objects
                # (see notes on loose sheets of paper)
                assert sorted(self.dict_info_absorption_times['t']) == self.dict_info_absorption_times['t'], \
                        "The absorption times in the dict_info_absorption_times dictionary are sorted" \
                        "\n{}".format(self.dict_info_absorption_times[P]['t'])
                if self.reactivate:
                    assert  len(self.dict_info_absorption_times['P']) == 0 or \
                            list(np.unique(self.dict_info_absorption_times['P'])) == sorted(self.dict_info_absorption_times['P']), \
                            "The particles associated to the absorption times in the dict_info_absorption_times dictionary are unique" \
                            "\n{}".format(self.dict_info_absorption_times['P'])

        if self.LOG: #True:
            print("\n******** Info PARTICLES before REACTIVATION:")
            for P in range(self.N):
                with printoptions(precision=3, suppress=True):
                    print("Particle {}".format(P))
                    print(np.c_[np.array(self.info_particles[P]['t']), np.array(self.info_particles[P]['E'])])

        if self.reactivate:
            # Helper function for an assertion below
            has_been_absorbed_at_time_t = lambda P, t: len(self.trajectories[P]['x']) == self.niter + 1 and self.trajectories[P]['t'][-1] < t and self.trajectories[P]['x'][-1] == 0
                ## NOTE: For this function to return what the name says, the 'x' attribute of the dictionary
                ## stored for each particle in self.trajectories must increase its size as the simulation progresses
                ## (in fact, the function checks whether the length of this attribute is equal to the number of iterations)
                ## (as opposed to fill it up up front when generating all the events for each particle
                ## at the beginning of this method generate_trajectories())    

            # Re-generate the trajectories based on reactivation of particles after absorption
            initial_number_absorption_times = len(self.dict_info_absorption_times['t'])
            while len(self.dict_info_absorption_times['t']) > 0:
                if self.LOG:
                    with printoptions(precision=3, suppress=True):
                        print("\n******** Trajectories:")
                        for P in range(self.N):
                            print("Particle {}".format(P))
                            print(np.r_[self.trajectories[P]['t'], self.trajectories[P]['x']])
                            pass
                        print("Absorption times:")
                        print(self.dict_info_absorption_times['t'])
                        print(self.dict_info_absorption_times['P'])
                        print(self.dict_info_absorption_times['iter'])
                t0 = self.dict_info_absorption_times['t'].pop(0)
                P = self.dict_info_absorption_times['P'].pop(0)
                iter = self.dict_info_absorption_times['iter'].pop(0)
                
                if iter < self.niter:
                    # Only reactivate the particle if the absorption didn't happen at the last iteration
                    # If this is the case, we should take the last position of the particle as absorbed
                    # which would avoid us further work when finalizing the simulation
                    # (which needs to deal with all non-absorbed --i.e. still alive-- particles).  

                    # Choose a particle and copy the relevant info from it to the reactivated particle
                    Q = self._choose_assigned_particle(P)
                    position_Q_at_t0 = self._copy_info_from_particle_at_given_time(P, Q, t0, iter)

                    # Since the particle was reactivated state that it is no longer absorbed
                    self.particle_already_absorbed[P] = False

                    # Update the trajectory of the reactivated particle to new positions
                    # using the already generated event times and starting at the position
                    # of the assigned particle Q.
                    iter_final, position_change = self._regenerate_trajectory(P, position_Q_at_t0, iter)

                    if self._is_particle_absorbed(P, position_change):
                        # Update the lists used to compute the expected survival time
                        self._update_survival_time_from_zero(P, self.trajectories[P]['t'][iter_final])
                        if iter_final < self.niter:
                            # The particle was absorbed and we have not yet reached the last iteration
                            # => Add the absorption time to the dictionary of absorption times
                            assert not self.particle_already_absorbed[P], \
                                    "The flag indicating that particle {} has been absorbed is set to False" \
                                    .format(P)
                            self.particle_already_absorbed[P] = True
                            self._update_info_absorption_times(P, self.trajectories[P]['t'][iter_final], iter_final)
    
                            assert len(self.info_particles[ self.particle_reactivation_ids[P] ]) >= 2, \
                                    "The number of relevant events of the reactivated particle P={}" \
                                    " to particle ID {} is at least two (an activation and an absorption at the end)." \
                                    .format(P, self.particle_reactivation_ids[P])
                            assert self.info_particles[ self.particle_reactivation_ids[P] ]['E'][-2] == EventType.ACTIVATION and \
                                   self.info_particles[ self.particle_reactivation_ids[P] ]['E'][-1] == EventType.ABSORPTION, \
                                    "The last two relevant events of the reactivated particle (p={}) for particle P={}" \
                                    " are ACTIVATION and ABSORPTION ({}, {})" \
                                    .format(self.particle_reactivation_ids[P], P,
                                            self.particle_reactivation_ids[P]['E'][-2].value,
                                            self.particle_reactivation_ids[P]['E'][-1].value)
                            if not has_been_absorbed_at_time_t(Q, t0):
                                # The following assertion only makes sense when the particle Q to which
                                # particle P has been assigned to is still active.
                                # It is NOT necessarily true ONLY when particle Q has become inactive, and
                                # this is only the case when ALL of the following conditions take place:
                                # - the particle Q has reached the last iteration of simulation
                                # - the particle Q has been absorbed at the last iteration of simulation
                                # - the absorption time of particle Q is SMALLER than the reactivation time of particle P.
                                assert np.sum( [1   if E == EventType.ABSORPTION
                                                    else 0
                                                    for E in self.info_particles[ self.particle_reactivation_ids[P] ]['E'] ] ) == 1, \
                                        "There is exactly one ABSORPTION event for the reactivated particle (P={}, p={}):" \
                                        "\n(events={})\n(times={})" \
                                        .format(P, self.particle_reactivation_ids[P],
                                                [E.value for E in self.info_particles[ self.particle_reactivation_ids[P] ]['E']],
                                                self.info_particles[ self.particle_reactivation_ids[P] ]['t'])

                    # Assertions for ALL cases, i.e. not only for the case when the particle is absorbed
                    # at the end of the trajectory regeneration process 
                    assert self.particle_reactivation_ids[P] == len(self.info_particles) - 1, \
                            "The reactivated particle ID (p={}) for particle P={}" \
                            " is the last one in the info_particles list (len={})" \
                            .format(self.particle_reactivation_ids[P], P, len(self.info_particles))
                    assert len(self.info_particles[ self.particle_reactivation_ids[P] ]) >= 2, \
                            "The number of relevant events of the reactivated particle P={}" \
                            " to particle ID {} is at least two (an activation and an absorption at the end)." \
                            .format(P, self.particle_reactivation_ids[P])
                    assert self.info_particles[ self.particle_reactivation_ids[P] ]['E'][0] == EventType.ACTIVATION, \
                            "The first relevant event of the reactivated particle (p={}) for particle P={}" \
                            " is an ACTIVATION ({})" \
                            .format(self.particle_reactivation_ids[P], P,
                                    self.particle_reactivation_ids[P]['E'][0].value)

                assert len(self.dict_info_absorption_times['t']) <= initial_number_absorption_times, \
                        "The length of the list containing the absorption times ({}) is" \
                        " <= the initial number of absorption times ({})" \
                        .format(len(self.dict_info_absorption_times['t']), initial_number_absorption_times)

    def _update_active_particles(self, P):
        """
        Updates the flag that states whether the given particle number P is active
        at the current iteration of the simulation in the list that stores this information.
        """
        # IMPORTANT: For the reactive=True case it is CRUCIAL that the 'x' attribute in the dictionary
        # storing the trajectory of each particle number increases its size as the simulation progresses
        # (as opposed to being all filled up with values (up to element self.niter) at the very beginning
        # by the process that generates all the event times for each particle number
        # --typically done by generate_trajectories()).
        # For the reactivate=False, the 'x' attribute is fully filled at the beginning and this is ok
        # because we are only interested in checking the active condition at the end of the simulation
        # i.e. at the last iteration, which is in effect retrieved by the -1 element of the list stored 
        # in the 'x' attribute, as done here.
        self.is_particle_active[P] = self.trajectories[P]['x'][-1] > 0

    def _update_trajectories(self, P):
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        self.assertTimeInsertedInTrajectories(P)
        self.trajectories[P]['t'] += [self.particles[P].getTimeLastEvent()]
        if not self.reactivate or \
           self.reactivate and not self.particle_already_absorbed[P]:
            self.trajectories[P]['x'] += [self.particles[P].size]
        self.trajectories[P]['e'] += [self.particles[P].getTypeLastEvent()]
        
        # Update the list stating whether each particle is active
        self._update_active_particles(P)

    def _update_info_absorption_times(self, P, time_of_absorption, it):
        "Inserts a new absorption time in order, together with its particle number and iteration number"
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        idx_insort, found = insort(self.dict_info_absorption_times['t'], time_of_absorption)
        self.dict_info_absorption_times['P'].insert(idx_insort, P)
        self.dict_info_absorption_times['iter'].insert(idx_insort, it)

    def _is_particle_absorbed(self, P, last_position_change):
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        return last_position_change < 0 and self.particles[P].size == 0

    def _update_info_particles(self, P, last_position_change):
        assert self.isValidParticle(P), "The particle to update is valid (0<=P<{}) ({})".format(self.N, P)

        #--------------- Helper functions -------------------
        # Possible status change of a particle 
        is_activated = lambda P: self.particles[P].size == 1 # Note that a particle can be activated even if it was ALREADY at position 1 => a new sub-trajectory becomes active
        is_absorbed = lambda P: self._is_particle_absorbed(P, last_position_change)
        is_blocked = lambda P: last_position_change > 0 and self.particles[P].size == self.particles[P].getCapacity()
        is_unblocked = lambda p: last_position_change < 0 and self.particles[P].size == self.particles[P].getCapacity() + last_position_change 

        # Type of event (relevant for the blocking probability estimation)
        type_of_event = lambda P: is_activated(P) and EventType.ACTIVATION or is_absorbed(P) and EventType.ABSORPTION or is_blocked(P) and EventType.BLOCK or is_unblocked(P) and EventType.UNBLOCK or None
        #--------------- Helper functions -------------------

        if type_of_event(P) is not None and \
           (not self.reactivate or
            self.reactivate and not self.particle_already_absorbed[P]):
            # When reactivate=True, only add the information about the event types
            # as long as the particle has not yet been absorbed. If the latter is the case
            # we should not add this information because it belongs to a new particle ID
            # that will be created once the particle is reactivated.
            p = self.particle_reactivation_ids[P]
            assert p < len(self.info_particles), \
                    "The particle ID (p={}) exists in the info_particles list (0 <= p < {})" \
                    .format(p, len(self.info_particles))
            self.info_particles[p]['t'] += [self.particles[P].getTimeLastEvent()]
            self.info_particles[p]['E'] += [type_of_event(P)]

    def _choose_assigned_particle(self, P):
        list_of_particles_to_choose_from = list(range(P)) + list(range(P+1, self.N))
        chosen_particle_number = list_of_particles_to_choose_from[ np.random.randint(0, self.N-1) ]

        assert self.isValidParticle(chosen_particle_number) and chosen_particle_number != P, \
                "The chosen particle is valid (0 <= Q < {} and different from {}) ({})" \
                .format(self.N, P, chosen_particle_number)
        return chosen_particle_number

    def _copy_info_from_particle_at_given_time(self, P, Q, t, iter):
        position_Q = self.get_position(Q, t)
        self.particles[P].resize(position_Q, t)
        self._copy_info_particles(P, Q, t, position_Q, iter)

        # Update the particle ID associated to the particle number P
        # to the particle ID just added to the info_particles list,
        # i.e. set it to the largest index of the info_particles list.
        self.particle_reactivation_ids[P] = len(self.info_particles) - 1

        return position_Q

    def _copy_info_particles(self, P, Q, t, position_Q, iter):
        """
        Copies the information about the relevant events happening on the given particle Q
        up to the given time t.
        The information is copied into a new entry in the info_particles list
        representing the ID of a newly reactivated particle from absorbed particle P.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) corresponding to the absorbed particle.

        Q: non-negative int
            Particle number (index of the list of particle queues) corresponding to the particle
            assigned to the absorbed particle to be reactivated.

        t: positive float
            Absorption time, which determines the maximum time up to which the information from particle Q
            is copied to P.

        position_Q: non-negative int
            Position of the Q particle at the time of reactivation.

        iter: non-negative int
            Iteration number at which the absorption happens.
        """

        def copy_event_times_and_types_up_to_time_t(q, t):
            for idx, s in enumerate(self.info_particles[q]['t']):
                if s <= t:
                    self.info_particles[-1]['t'] += [s]
                    self.info_particles[-1]['E'] += [ self.info_particles[q]['E'][idx] ]
                else:
                    break

        assert self.isValidParticle(P) and self.isValidParticle(Q), \
                "The particle numbers representing the absorbed (P) and reactivated (Q) particles are valid" \
                "\n(0<=P<{}, 0<=Q<{}) (P={}, Q={})" \
                .format(self.N, self.N, P, Q)
                
        # Particle ID in the info_particles list from where we should copy
        # the information of relevant events for the assigned particle Q up to time t.
        q = self.particle_reactivation_ids[Q]
        assert self.isValidParticleId(q), "The particle ID is valid (0<=q<{}) ({})" \
                .format(len(self.info_particles), q)

        self.info_particles += [ dict({'t': [],
                                       'E': [],
                                       't0': t,
                                       'x': position_Q,
                                       'iter': iter,
                                       'particle number': P,
                                       'reactivated number': Q,
                                       'reactivated ID': q}) ]
        copy_event_times_and_types_up_to_time_t(q, t)

        if self.LOG: #True:
            with printoptions(precision=3, suppress=True):
                print("\n******** Info PARTICLES after particle {} reactivated to particle {} at time {:.3f}:".format(P, Q, t))
                print("Particle ID {}".format(q))
                print(np.r_[self.info_particles[q]['t'], self.info_particles[q]['E']])
                print("NEW Particle ID")
                print(np.r_[self.info_particles[-1]['t'], self.info_particles[-1]['E']])

    def _regenerate_trajectory(self, P, position_start, iter):
        """
        Regenerates the trajectory of particle P from the given start position
        up until the last iteration is reached or the particle is absorbed.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) corresponding to the reactivated particle.

        position_start: non-negative int
            Start position of the reactivated particle from which the trajectory continues.

        iter: non-negative int
            Iteration number at which the reactivation takes place from the given start position.

        Return: tuple
        The tuple contains:
        - the last iteration number at which the trajectory was updated
        - the last change of position experienced by the particle.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        assert 0 <= position_start <= self.particles[P].getCapacity(), \
                "The start position of reactivated particle {} ({}) is between 0 and {}" \
                " (the max. position allowed for the particle)" \
                .format(P, position_start, self.particles[P].getCapacity())
        assert self.particles[P].size == position_start, \
                "The start position ({}) of the reactivated particle {} is equal to the given start position ({})" \
                .format(self.particles[P].size, P, position_start)
        assert iter < self.niter, \
                "The iteration number at which reactivation takes place ({}) is NOT the last iteration ({})" \
                .format(iter, self.niter)

        self.trajectories[P]['x'][iter] = position_start
        self.all_positions[P][iter] = position_start
        for it in range(iter+1, self.niter+1):
            position_change = self._update_particle_trajectory(P, it)
            self._update_info_particles(P, position_change)

            assert  self.particles[P].getTypeLastEvent() == self.trajectories[P]['e'][it] and \
                    self.particles[P].getTimeLastEvent() == self.trajectories[P]['t'][it], \
                    "The type and time of the last event for particle {}" \
                    " have not changed during reactivation at iteration {}: (new={}, {} vs. orig={}, {})" \
                    .format(P, it,
                            self.particles[P].getTypeLastEvent(), self.particles[P].getTimeLastEvent(),
                            self.trajectories[P]['e'][it], self.trajectories[P]['t'][it])

            if self._is_particle_absorbed(P, position_change):
                # => Stop the generation of the trajectory as this particle will need to be reactivated
                break

        # Update the list stating whether each particle is active
        self._update_active_particles(P) 

        return it, position_change

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

    def _apply_event(self, P, event_to_apply, time_of_event):
        position_change = self.particles[P].apply_event(event_to_apply, time_of_event)
        self._update_particle_position_info(P, self.particles[P].size, time_of_event)
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
            assert -1 <= positions_change[p] <= 1, \
                    "The change of position of particle p={} is -1, 0, or +1 ({})" \
                    .format(p, positions_change[p])
        if self.LOG:
            with printoptions(precision=3, suppress=True):
                print("\ntimes to event: {}".format(times_of_events))
            print("\tparticle positions: {}".format(self.positions))
            print("\tpositions change: {}".format(positions_change))
            print("\tactivation times by particle:")
            for p in range(self.N):
                _, activation_times_p = self.get_activation_times_for_active_subtrajectories(p)
                with printoptions(precision=3):
                    print("\t\tp={}: {}".format(p, np.array(activation_times_p)))
            print("\ttime elapsed since activation times:")
            for p in range(self.N):
                _, activation_times_p = self.get_activation_times_for_active_subtrajectories(p)
                with printoptions(precision=3):
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
                with printoptions(precision=3, suppress=True):
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

            self.assertSystemConsistency()

    def _add_new_activation_time(self, p, time_of_event, type_of_event):
        "Adds a new time to the array of activation times"
        # Add this time to the list of activation times and to the list of active flags
        # NOTE: we need to EXPLICITLY insert the new time IN ORDER in the array,
        # despite the fact the time_of_event values are SORTED in the CURRENT call to this method,
        # because we do not know anything about the order of these time_of_event values in relation to
        # the time values ALREADY PRESENT in the array.
        idx_insort = self._insert_new_time_in_order(p, time_of_event, EventType.ACTIVATION)
        self.active.insert(idx_insort, True)
        if type_of_event == EventType.REACTIVATION: # self.LOG:
            with printoptions(precision=3, suppress=True):
                print("\n>>>>> {}: p={}, iter={}: time_of_activation={:.3f}:" \
                        "\n\tall_activation_times={}" \
                        "\n\tall_particles={}" \
                        "\n\tall_active={}" \
                      .format(type_of_event.name.upper(), p, self.iter, time_of_event,
                              np.array(self.atimes), self.pat, self.active))

    def _update_survival_time_from_zero(self, P, time_of_absorption):
        ref_time = 0.0 if np.isnan(self.times0[P]) else self.times0[P]

        # Refer the survival time to the latest time the particle was at position 0
        # and add it to the previously measured survival time
        assert time_of_absorption > ref_time, \
                "The time of absorption ({}) contributing to the mean survival time calculation" \
                " is larger than the latest time of absorption of the particle ({})" \
                .format(time_of_absorption, ref_time)
        self.ktimes0_sum[P] += time_of_absorption - ref_time
        self.ktimes0_n[P] += 1

        if False:
            with printoptions(precision=3, suppress=True):
                print("\n>>>> Particle P={}: absorption @t={:.3f}".format(P, time_of_absorption))
                print(">>>> Previous times at position 0: {:.3f}".format(self.times0[P]))
                print(">>>> Total Survival times for ALL particles: {}".format(np.array(self.ktimes0_sum)))
                print(">>>> Total Survival units for ALL particles: {}".format(np.array(self.ktimes0_n)))

        # Update the latest time the particle was at position 0
        self.times0[P] = time_of_absorption

        if False:
            print(">>>> UPDATED Last time at 0: P={}, {:.3f}".format(P, self.times0[P]))

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

    def _insert_new_time_in_order(self, p, time_of_event, type_of_event):
        """
        Inserts new elements in the lists related to absolute time_of_event storage.
        
        Arguments:
        p: int
            Index of the particle affected by the insert.
        time_of_event: positive float
            Time to insert.
        type_of_event: EventType
            Type of the event associated to the time_of_event to insert.
            
        Return:
        The index of the insertion in the lists that contain information about the events
        of the given type for the given particle.
        """
        ltimes, lparticles, liter, _ = self.get_lists_with_event_type_info(type_of_event)

        idx_insort, found = insort(ltimes, time_of_event)
        lparticles.insert(idx_insort, p)
        liter.insert(idx_insort, self.iter)

        return idx_insort

    def _insert_new_time(self, p, time_of_event, type_of_event):
        # Get the activation times so that we can compute the RELATIVE time to the event
        idx_activation_times, activation_times = self.get_activation_times_for_active_subtrajectories(p)
        times_since_activation = [time_of_event - u for u in activation_times if u < time_of_event]
            ## Need to ask whether the activation time u < time_of_event because
            ## when we reactivate a particle (in method _reactivate()),
            ## we do the insertion of times ASYNCHRONOUSLY, i.e. we FIRST insert all activation times
            ## and THEN insert all block and unblock times. This would lead to the situation where
            ## some block times, AT THE TIME OF THEIR INSERTION IN THE RELATIVE TIMES OF BLOCK TIMES LIST,
            ## would happen EARLIER than some of the activation times of the particle... which does not make sense.
            ## So we should NOT insert that relative time to the list of relative block times list.
            ## This is precisely what we accomplish by asking u < time_of_event.  
        if self.LOG:
            with printoptions(precision=3, suppress=True):
                print("\nXXXXXXXXXXXXXXX {}: p={}, iter={}: time_of_event={:.3f}, a: {}, s: {}" \
                      .format(type_of_event.name.upper(), p, self.iter, time_of_event, np.array(activation_times), np.array(times_since_activation)))

        self._insert_new_relative_times(idx_activation_times, times_since_activation, type_of_event)

    def _insert_new_relative_times(self, idx_activation_times, times_since_activation, type_of_event):
        for idx_activation_time, s in zip(idx_activation_times, times_since_activation):
            if type_of_event == EventType.ABSORPTION:
                idx_insort, found = insort(self.sk, s)
                self._update_counts_alive(idx_insort)
                # De-activate the trajectory associated to the absorption
                self.active[idx_activation_time] = False
            elif type_of_event == EventType.BLOCK or type_of_event == EventType.UNBLOCK:
                idx_insort, found = insort(self.sbu, s)
                self._update_counts_blocked(idx_insort, type_of_event)
            else:
                raise ValueError("The type of event is invalid: {}".format(type_of_event))

    def _update_counts_alive(self, idx_to_insert: int, event_type: EventType):
        """
        Updates the counts of particles alive in each time segment where the count can change
        following the activation of a new particle.

        Arguments:
        idx_to_insert: non-negative int
            Index in the counts_alive list where a new element will be inserted to count
            the number of alive particles in the time segment defined by
            [ self.sk[idx_to_insert], self.sk[idx_to_insert+1] )
            If the new element is added at the end of the list, the the upper bound of
            the time segment if Inf.

        event_type: EventType
            Whether the particle has been absorbed (EventType.ABSORPTION) or
            censored (EventType.CENSORING).
            This affects how the counts_alive list is updated:
            - in the absorption case, all the counts to the LEFT of the newly inserted element
            to the list are increased by +1.
            - in the censoring case ALSO the count of the newly inserted element is increased by +1
            (from the original value copied from the element to the left).
            This is stating that the particle associated to this count and the corresponding
            time segment in self.sk is still alive, at least up to the end of the newly inserted
            "time segment" in the self.sk list --which could be infinite if the new element is
            inserted at the end of the list.
        """
        assert 1 <= idx_to_insert and idx_to_insert <= len(self.counts_alive), \
                "The index where a new counter needs to be inserted ({}) is at least 1" \
                " and at most the current number of elements in the counts_alive array ({})" \
                .format(idx_to_insert, len(self.counts_alive))
        assert event_type in [EventType.ABSORPTION, EventType.CENSORING], \
                "The event type is either ABSORPTION or CENSORING ({})" \
                .format(event_type.name)

        # Insert a new element in the list and assign it a count value
        # equal to the count of the segment before split
        # (in fact, the particle just absorbed had not yet been counted in the segment
        # and thus we should not count it on the right part of the split, since the particle no longer
        # exists at that time)
        self.counts_alive.insert(idx_to_insert, self.counts_alive[idx_to_insert - 1])

        # Increase by 1 ALL counts to the LEFT of the insert index
        # (because the absorption indicates that all time segments that are smaller
        # than the elapsed time to absorption should count the particle just absorbed as active)
        for idx in range(idx_to_insert):
            self.counts_alive[idx] += 1

        if event_type == EventType.CENSORING:
            # Increase by 1 the count corresponding to the inserted index
            # (i.e. the count corresponding to the time segment that STARTS at the new inserted
            # time in the self.sk list of time segments, which means that the particle
            # is considered alive at least up to the end of such time segment --which could be
            # INFINITE if this new element is inserted at the end of the list; in such case
            # we are assuming that the particle is still alive up to Infinity.)  
            self.counts_alive[idx_to_insert] += 1

    def _update_counts_blocked(self, idx_to_insert_or_update, event_type, new=True):
        """
        Updates the counts of blocked particles in each time segment where the count can change
        following the blocking or unblocking of a new particle.

        Arguments:
        new: bool
            Whether the index to insert corresponds to a new split of an existing time segment
            or whether the time segment already exists and it should be updated. 
        """
        assert 1 <= idx_to_insert_or_update <= len(self.counts_blocked), \
                "The index where a new counter needs to be inserted ({}) is at least 1" \
                " and at most the current number of elements in the counts_blocked array ({})" \
                .format(idx_to_insert_or_update, len(self.counts_blocked))
        assert event_type in [EventType.BLOCK, EventType.UNBLOCK], \
                "The event type is either BLOCK or UNBLOCK ({})" \
                .format(event_type.name)

        if new:
            # Insert a new element in the list and assign it a count value
            # equal to the count of the segment before split.
            self.counts_blocked.insert(idx_to_insert_or_update, self.counts_blocked[idx_to_insert_or_update - 1])

        # INCREASE/DECREASE by 1 ALL counts to the RIGHT of the insert index
        # (because the block/unblock indicates that all time segments that are larger
        # than the elapsed time to block/unblock should count the change of blocking status of the particle)
        if event_type == EventType.BLOCK:
            delta_count = +1
        elif event_type == EventType.UNBLOCK:
            delta_count = -1
        for idx in range(idx_to_insert_or_update, len(self.counts_blocked)):
            self.counts_blocked[idx] += delta_count

    def _update_particle_position_info(self, P, new_position, time_at_new_position):
        self.positions[P] = new_position
        self.all_positions[P] += [self.positions[P]]
        self.all_times[P] += [time_at_new_position]

    def _update_particle_trajectory(self, P, it):
        """
        Updates the trajectory of the given particle number at the given iteration
        and returns the change in position.
        """
        # Retrieve the event type and time as this information has been already generated
        # at the very beginning of the simulation by the call to the generate_trajectories() method.
        next_event = self.trajectories[P]['e'][it]
        next_event_time = self.trajectories[P]['t'][it]

        # Apply the event to the particle (queue)
        position_change = self.particles[P].apply_event(next_event, next_event_time)

        # Update the trajectory position at the retrieved event time
        self.trajectories[P]['x'] += [self.particles[P].size]

        # Update the `positions` lists
        self.positions[P] = self.particles[P].size
        self.all_positions[P][it] = self.particles[P].size

        return position_change

    def finalize(self):
        """
        Finalize the simulation process by treating particles that are active,
        i.e. those particles providing CENSORED survival time values.
        
        The type of finalize process is defined by the object's attribute `finalize_type`.
        """
        active_particle_numbers = self.get_active_particle_numbers()
        active_particle_ids = list( np.sort([ self.particle_reactivation_ids[P] for P in active_particle_numbers ]) )

        if True: #self.LOG:
            if self.finalize_type == FinalizeType.ABSORB_CENSORED:
                finalize_process = "absorbing"
            elif self.finalize_type == FinalizeType.REMOVE_CENSORED:
                finalize_process = "removing"
            elif self.finalize_type == FinalizeType.ESTIMATE_CENSORED:
                finalize_process = "using"
            print("\n****** FINALIZING the simulation by {} the currently active particles: ******".format(finalize_process))
            print("Active particles: {}" \
                  .format(", ".join(["(p={}, P={})".format(p, self.info_particles[p]['particle number']) for p in active_particle_ids])))

        # Make a copy of the list because this may be updated by removing elements
        if self.finalize_type == FinalizeType.REMOVE_CENSORED:
            assert sorted(active_particle_ids) == active_particle_ids, \
                    "The list of active particle IDs is sorted," \
                    " which is crucial for the correct removal of censored particles ({})" \
                    .format(active_particle_ids)
        nremoved = 0
        for p in active_particle_ids:
            dict_info = self.info_particles[p - nremoved]
            P = dict_info['particle number'] 
            print("Processing particle ID p={}, P={} out of {} particles".format(p, P, len(active_particle_ids)))
            assert len(dict_info['E']) > 0, \
                    "There is at least one event in the info_particles dictionary for particle ID p={}" \
                    .format(p)
            assert dict_info['E'][0] == EventType.ACTIVATION, \
                    "The first event for particle ID p={} is an ACTIVATION ({})" \
                    .format(p, dict_info['E'][0].value)

            if dict_info['E'][-1] != EventType.ABSORPTION:
                if self.finalize_type == FinalizeType.REMOVE_CENSORED:
                    if self.reactivate:
                        # Simply remove the particle from the analysis
                        self.info_particles.pop(p - nremoved)
                        nremoved += 1
                            # NOTE that no update needs to be done to the quantities
                            # used to estimate E(T) (e.g. ktimes0_sum) because these quantities are updated
                            # ONLY when there is an absorption, and this never happened
                            # for this particle (recall that, when reactivate=True,
                            # there is at most ONE absorption per particle,
                            # and when it occurs, it's the last event to occur. 
                    else:
                        # Find the latest absorption time and remove all the events since then
                        idx_last_absorption = find_last(dict_info['E'], EventType.ABSORPTION)
                        if idx_last_absorption >= 0:
                            # Remove all the elements AFTER the index of the last absorption to the end 
                            while len(dict_info['E']) > idx_last_absorption + 1:
                                dict_info['t'].pop(idx_last_absorption+1)
                                dict_info['E'].pop(idx_last_absorption+1)
                            # NOTE that no update needs to be done to the quantities
                            # used to estimate E(T) (e.g. ktimes0_sum) because these quantities are updated
                            # ONLY when there is an absorption, and this already happened
                            # at the latest absorption time. Since we are removing everything
                            # that happened after that, we do not need to update neither
                            # ktimes0_sum, nor ktimes0_n, nor times0.   
                    # Flag the particle as inactive (as the latest censored trajector was removed)
                    self.is_particle_active[P] = False

                elif self.finalize_type in [FinalizeType.ABSORB_CENSORED, FinalizeType.ESTIMATE_CENSORED]:
                    # Increase slightly the time to insert the new "fictitious" event
                    # in order to avoid repeated times in the list of time segments
                    # (which messes up the process that merges the two list of times:
                    # the one defining the alive time segments, and the one defining
                    # the block/unblock time segments)
                    time_to_insert = dict_info['t'][-1] + self.EPSILON_TIME
                    if dict_info['E'][-1] == EventType.BLOCK:
                        # If the last event was a BLOCK, unblock it in order to avoid a potential
                        # value for P(BLOCK / T>t, s=1) > 1, which would happen in the case 
                        # the end of the block/unblock time segment starting at such BLOCK event
                        # ends LATER than the end of the alive time segment *AND* the alive time segment
                        # is NOT the last one among the alive time segments (which would imply that
                        # the alive count associated to that time is valid until Infinity
                        # --so its count will never decrease)  
                        dict_info['t'] += [time_to_insert]
                        dict_info['E'] += [EventType.UNBLOCK]
                        # Increase the time to insert for the next insertion of the censoring "event" next
                        time_to_insert += self.EPSILON_TIME

                    # Final event which can be either an absorption or a censoring time
                    dict_info['t'] += [time_to_insert]
                    if self.finalize_type == FinalizeType.ABSORB_CENSORED:
                        dict_info['E'] += [EventType.ABSORPTION]
                        # Flag the particle as inactive (as it was just absorbed)
                        self.is_particle_active[P] = False
                    elif self.finalize_type == FinalizeType.ESTIMATE_CENSORED:
                        dict_info['E'] += [EventType.CENSORING]  

                    # Survival time from s=0
                    self._update_survival_time_from_zero(P, time_to_insert)

        self.is_simulation_finalized = True

    def compute_counts(self):
        """
        Computes the survival time segments and blocking time segments needed for
        the calculation of P(T>t / s=1) and P(BLOCK / T>t, s=1). 
        """
        print("Computing counts for the calculation of the survival function and conditional blocking probability...")
        if not self.is_simulation_finalized:
            raise ValueError("The simulation has not been finalized..." \
                          "\nThe computation of the counts by time segment cannot proceed." \
                          "\nRun first the finalize() method and rerun.")

        # Initialize the lists that will store information about the counts by time
        self.sk = [0.0]
        self.sbu = [0.0]
        # The following counts are initialized to 0 because:
        # - they correspond to the number of particles counted in the time segment 
        # defined by the value in sk or sbu and Infinity, and they are updated ONLY when
        # a new time is observed, in which case the non-zero count goes to the segment
        # just inserted (if it's the first one, the segment is [0, t) where t is the
        # event time triggering the update of the counts).
        # - the capacity of the queue is assumed larger than 1, as the status of a particle
        # CANNOT be both ACTIVE and BLOCKED, because if we allowed this we would need to
        # change the way events are stored in the info_particles list of dictionaries,
        # as now only one event type is accepted at each event time. 
        self.counts_alive = [0]
        self.counts_blocked = [0]
        # Go over each particle
        for p, dict_info in enumerate(self.info_particles):
            P = dict_info['particle number']
            print("Processing particle ID p={}, P={} out of {} particles".format(p, P, len(self.info_particles)))
            assert dict_info['E'][0] == EventType.ACTIVATION, \
                    "The first event for particle ID p={} is an ACTIVATION ({})" \
                    .format(p, dict_info['E'][0].value)
            assert dict_info['E'][-1] in [EventType.ABSORPTION, EventType.CENSORING], \
                    "The last event for particle ID p={} is an ABSORPTION or a CENSORING observation ({})" \
                    .format(p, dict_info['E'][-1].value)

            activation_times = []
            event_type_prev = None
            for t, event_type in zip(dict_info['t'], dict_info['E']):
                if event_type == EventType.ACTIVATION:
                    assert event_type_prev != EventType.CENSORING, \
                            "There must be NO event after a CENSORING observation ({})".format(event_type)
                    # 2020/06/25: This assertion is not true under very rare BORDERLINE circumstances
                    # happening towards the end of the simulation, but they are valid. In any case
                    # the situation happens all the time when reactivate=False, so the situation is dealt with
                    # appropriately below by reinitializing the activation_times list to empty after
                    # the occurrence of an absorption.
                    # For more details about the conditions under which this can happen, see the
                    # generate_trajectories() method where it is explained when calling the lambda function
                    # has_been_absorbed_at_time_t().    
                    #if self.reactivate:
                    #    assert event_type_prev != EventType.ABSORPTION, \
                    #            "There must be NO event after an absorption (p={}, P={}, {})" \
                    #            .format(p, P, event_type)
                    assert len(activation_times) == 0 and event_type_prev in [None, EventType.ABSORPTION] or \
                           len(activation_times)  > 0, \
                            "The activation_times list is empty only when it's the first event" \
                            " or when the previous event was an ABSORPTION (prev={}, atimes={})" \
                            .format(event_type_prev, activation_times)  
                    activation_times += [t]
                elif event_type in [EventType.ABSORPTION, EventType.BLOCK, EventType.UNBLOCK]:
                    if event_type == EventType.UNBLOCK:
                        assert event_type_prev == EventType.BLOCK, \
                                "The event coming before an UNBLOCK event is a BLOCK event ({})" \
                                .format(event_type_prev.value)

                    self.insert_relative_time(t, activation_times, event_type)
                    if event_type == EventType.ABSORPTION:
                        activation_times = []

                    if False:
                        # These assertions are disabled because they take time
                        assert sorted(self.sk) == list(np.unique(self.sk)), \
                                "The list of survival time segments contains unique values" \
                                " after insertion of event {} for particle p={}, P={}" \
                                .format(event_type.name, p, P)
                        assert sorted(self.sbu) == list(np.unique(self.sbu)), \
                                "The list of block/unblock time segments contains unique values" \
                                " after insertion of event {} for particle p={}, P={}" \
                                .format(event_type.name, p, P)
                event_type_prev = event_type

        assert len(self.counts_alive) == len(self.sk), \
                "The length of counts_alive ({}) is the same as the length of self.sk ({})" \
                .format(len(self.counts_alive), len(self.sk))
        assert len(self.counts_blocked) == len(self.sbu), \
                "The length of counts_alive ({}) is the same as the length of self.sk ({})" \
                .format(len(self.counts_blocked), len(self.sbu))
        if self.START == 0 and self.finalize_type in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
            assert self.counts_alive[-1] == 0, "The last element of the counts_alive list is 0 ({})".format(self.counts_alive[-1])
        if self.START < self.queue.getCapacity():
            assert self.counts_blocked[0] == 0, "The first element of the counts_blocked list is 0 ({})".format(self.counts_blocked[0])

        if self.LOG:
            with printoptions(precision=3, suppress=True):
                print("Relative absorption times and counts:\n{}".format(np.c_[np.array(self.sk), np.array(self.counts_alive)]))
                print("Relative blocking times:\n{}".format(np.c_[np.array(self.sbu), np.array(self.counts_blocked)]))

    def insert_relative_time(self, t: float, activation_times: list, event_type: EventType):
        """
        Inserts a new time segment in the list containing the times where a change in counts
        associated to the given event type happens.
        
        Only the activation times that are smaller than the given time t are considered.
        """
        for a in activation_times:
            s = t - a
            if s < 0:
                break
            if event_type in [EventType.ABSORPTION, EventType.CENSORING]:
                idx_insort, found = insort(self.sk, s)
                assert not found, "The time value is NOT found in the list of survival time segments ({})".format(s)
                self._update_counts_alive(idx_insort, event_type)
            elif event_type in [EventType.BLOCK, EventType.UNBLOCK]:
                idx_insort, found = insort(self.sbu, s, unique=True)
                self._update_counts_blocked(idx_insort, event_type, new=not found)
            else:
                raise ValueError("The event type is invalid: {}".format(event_type))

    def estimate_proba_survival_given_position_one(self):
        assert len(self.sk) > 1, "The length of the survival times array is at least 2 ({})".format(self.sk)
        assert len(self.counts_alive) == len(self.sk), \
                "The number of elements in the survival counts array ({})" \
                " is the same as the number of elements in the survival time segments array ({})" \
                .format(len(self.counts_alive), len(self.sk))

        self.proba_surv_by_t = [c / (len(self.sk)-1) for c in self.counts_alive]
        return pd.DataFrame.from_items([('t', self.sk), ('P(T>t / s=1)', self.proba_surv_by_t)])

    def estimate_proba_survival_and_blocking_conditional(self):
        "Computes P(BLOCK / T>t,s=1)"

        # Since we must compute this probability CONDITIONED to the event T > t,
        # we first need to merge the measurements of the block times and of the survival times
        # into one single set of measurement time points.
        assert len(self.counts_blocked) == len(self.sbu), \
                "The number of elements in the blocked counts array ({})" \
                " is the same as the number of elements in the blocking time segments array ({})" \
                .format(len(self.counts_blocked), len(self.sbu))

        if self.START == 0 and self.finalize_type in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
            assert self.counts_alive[-1] == 0, \
                    "The number of particles alive at the last measured time is 0" \
                    " since particles are assumed to have been all absorbed or removed ({})" \
                    .format(self.counts_alive)
        if self.START < self.queue.getCapacity() and self.finalize_type in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
            assert self.counts_blocked[-1] == 0, \
                    "The number of particles blocked at the last measured time is 0" \
                    " since particles are assumed to have been all absorbed or removed ({})" \
                    .format(self.counts_blocked)

        self.t, counts_alive, counts_blocked = merge_values_in_time(self.sk, self.counts_alive, self.sbu, self.counts_blocked)

        if  len(self.t) > 1:
            self.proba_surv_by_t = [n_survived / counts_alive[0] for n_survived in counts_alive]
            self.proba_block_by_t = [n_blocked / n_survived if n_survived > 0
                                                            else 0.0 if n_blocked == 0
                                                            else np.inf 
                                                            for n_blocked, n_survived in zip(counts_blocked, counts_alive)]
        else:
            self.proba_surv_by_t = [1.0]
            self.proba_block_by_t = [0.0]

        # Assertions
        idx_inf = find(self.proba_block_by_t, np.inf)
        assert len(idx_inf) == 0, \
                "The conditional blocking probability is never INFINITE" \
                " (which would mean that the # blocked particles > 0 while # survived particles = 0)" \
                "\nTimes where this happens: {}".format([self.t[idx] for idx in idx_inf])
        assert np.all( 0 <= np.array(self.proba_block_by_t) ) and np.all( np.array(self.proba_block_by_t) <= 1 ), \
                "The conditional blocking probabilities take values between 0 and 1 ([{:.3f}, {:.3f}])" \
                .format(np.min(self.proba_block_by_t), np.max(self.proba_block_by_t))
        assert self.proba_surv_by_t[0] == 1.0, "The survival function at t = 0 is 1.0 ({:.3f})".format(self.proba_surv_by_t[0])
        if self.START == 0 and self.finalize_type in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
            assert self.proba_surv_by_t[-1] == 0.0, "The survival function at the last measured time is 0 ({})".format(self.proba_surv_by_t[-1])
        if self.START < self.queue.getCapacity() and self.finalize_type in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
            assert self.proba_block_by_t[-1] == 0.0, "The conditional blocking probability at the last measured time is 0 ({})".format(self.proba_block_by_t[-1])

        return pd.DataFrame.from_items([('t', self.t),
                                        ('P(T>t / s=1)', self.proba_surv_by_t),
                                        ('P(BLOCK / T>t,s=1)', self.proba_block_by_t)])

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

        #if self.finalize_type == FinalizeType.ESTIMATE_CENSORED:
        #    # Add an estimation of the survival time for the censored (i.e. still active) particles
        #    for P in self.active_particles:
        #        survival_time = self.particles[P].getTimeLastEvent() - self.times0[P]
        #        ### TODO: (2020/06/13) Write the get_survival_function() method that returns the estimate of S(t)
        #        self.ktimes0_sum[P] += survival_time # / (1 - self.get_survival_function(survival_time))
        #        self.ktimes0_n[P] += 1
        #else:
        if self.finalize_type != FinalizeType.ESTIMATE_CENSORED:
            assert sum(self.is_particle_active) == 0, "The number of active particles is 0 ({})".format(sum(self.is_particle_active))

        assert np.sum(self.ktimes0_n) > 0, "At least one particle has been absorbed"
        
        if self.reactivate:
            # When reactivation is used, ALL survival times are measured with respect to time 0
            # (assuming at that time the particle started at position 0
            # Therefore we estimate the expected survival time as the average of the latest
            # time measured by each particle.
            self.expected_survival_time = np.mean([self.particles[P].getTimeLastEvent() for P in range(self.N)])
        else:
            self.expected_survival_time = np.sum(self.ktimes0_sum) / np.sum(self.ktimes0_n)

        return self.expected_survival_time

    def estimate_proba_blocking(self):
        # Compute self.t, self.proba_surv_by_t, self.proba_block_by_t
        if not self.is_simulation_finalized:
            raise ValueError("The simulation has not been finalized..." \
                          "\nThe estimation of the blocking probability cannot proceed." \
                          "\nRun first the finalize() method and rerun.")
            
        self.estimate_proba_survival_and_blocking_conditional()

        # Integrate => Multiply the survival, the conditional blocking probabilities, and delta(t) and sum 
        integral = 0.0
        for i in range(0, len(self.proba_surv_by_t)-1):
            integral += (self.proba_surv_by_t[i] * self.proba_block_by_t[i]) * (self.t[i+1] - self.t[i])
        if self.LOG:
            print("integral = {:.3f}".format(integral))

        # Expected survival time
        expected_survival_time = self.estimate_expected_survival_time_given_position_zero()

        # Estimate the blocking probability!
        self.proba_blocking = integral / expected_survival_time

        return self.proba_blocking, integral, expected_survival_time

    #-------------------------------------- Getters -------------------------------------------
    def get_position(self, P, t):
        "Returns the position of the given particle at the given time"
        if self.isValidParticle(P):
            idx = bisect.bisect(self.trajectories[P]['t'], t)
            return idx > 0 and self.trajectories[P]['x'][idx-1] or 0
        else:
            self.raiseWarningInvalidParticle(P)
            return np.nan

    def get_active_particle_numbers(self):
        "Returns the list of particle particle numbers P whose current position is > 0"
        return [P for P in range(len(self.is_particle_active)) if self.is_particle_active[P]]

    def get_active_particles_at_time(self, t):
        """
        Returns the list of particle numbers that are active at time t
        and a dictionary with their activation times after their latest absorption time
        """
        activation_times_smaller_than_t, particles_with_activation_times_smaller_than_t = \
            self.get_times_and_particles_smaller_than_t(t, EventType.ACTIVATION)
        absorption_times_smaller_than_t, particles_with_absorption_times_smaller_than_t = \
            self.get_times_and_particles_smaller_than_t(t, EventType.ABSORPTION)

        # Construct the list of active particles at time t
        # and the dictionary containing the activation times for each active particle
        # (which might be necessary during reactivation to copy the activation times
        # of the assigned particle to the reactivated particle)
        active_particles_at_time_t = []
        dict_position_by_particle = dict()
        dict_activation_times_by_particle = dict()
        for p in range(self.N):
            assert sorted(absorption_times_smaller_than_t) == absorption_times_smaller_than_t, \
                    "The absorption times list is sorted increasingly ({})" \
                    .format(absorption_times_smaller_than_t)
            idx_last_absorption_time_for_p = find_last(particles_with_absorption_times_smaller_than_t, p)
            if idx_last_absorption_time_for_p < 0:
                # The particle was never absorbed
                # => t = 0 is the last time the particle was at position 0
                last_absorption_time = 0.0
            else:
                last_absorption_time = absorption_times_smaller_than_t[ idx_last_absorption_time_for_p ]
            # Find the activation times between the last absorption time and the current time t 
            activation_times_later_than_last_absorption_and_smaller_than_t_for_p = \
                    [activation_times_smaller_than_t[idx]   for idx in range(len(activation_times_smaller_than_t))
                                                            if  particles_with_activation_times_smaller_than_t[idx] == p and
                                                                last_absorption_time < activation_times_smaller_than_t[idx]]
            if len(activation_times_later_than_last_absorption_and_smaller_than_t_for_p) > 0:
                active_particles_at_time_t += [p]
                # Get the latest position of the particle before time t
                for idx in range(len(self.all_times[p])-1, -1, -1):
                    if self.all_times[p][idx] < t:
                        dict_position_by_particle[p] = self.all_positions[p][idx]
                        break
                dict_activation_times_by_particle[p] = activation_times_later_than_last_absorption_and_smaller_than_t_for_p

        return active_particles_at_time_t, dict_position_by_particle, dict_activation_times_by_particle

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
            idx_activation_times_p, activation_times_p = self.get_activation_times_for_active_subtrajectories(p)
            t = self.get_time_last_event(p)
            particles += [self.pat[idx] for idx in idx_activation_times_p]
            times_from_activation += [t - u for u in activation_times_p]
            with printoptions(precision=3, suppress=True):
                assert np.sum([t < u for u in activation_times_p]) == 0, \
                        "All the activation times of particle {} ({}) are smaller than the time of the latest event ({:.3f})" \
                        .format(p, np.array(activation_times_p), t)

        order = self._compute_order(times_from_activation)
        return [particles[o] for o in order], [times_from_activation[o] for o in order]

    def get_activation_times_for_active_subtrajectories(self, p):
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

    def get_times_and_particles_smaller_than_t(self, t, type_of_event):
        """
        Returns the times smaller than t and their respective particle numbers
        for the given event type.

        This function is useful to return at once both the times AND the particles
        associated to those times.

        Argument:
        t: positive float
            Time of interest.

        type_of_event: EventType
            Type of event associated to the times to search in relation to t.

        Return: tuple
        The tuple contains:
        - a list with the times that are smaller than t for the given event type
        - a list with the particles associated to those times. 
        """
        ltimes, lparticles, _, _ = self.get_lists_with_event_type_info(type_of_event)
        zipped_values = list( zip( *[(ltimes[idx], lparticles[idx]) for idx in range(len(ltimes)) if ltimes[idx] < t] ) )
        
        if len(zipped_values) > 0:
            # Extract the tuples from the zipped values list
            # and convert each to a list.
            # The conversion to a list is IMPORTANT because that allows us to
            # e.g. check whether the list is sorted (done for instance via assertions)
            # because sorted( tuple ) returns a list!
            times = list( zipped_values[0] )
            particles = list( zipped_values[1] ) 
            return times, particles
        else:
            return [], []

    def get_survival_time_segments(self):
        return self.sk

    def get_blocking_time_segments(self):
        return self.sbu

    def get_all_total_blocking_time(self):
        "Returns the total blocking time for all particles"
        blocking_time = 0.0
        for P in range(self.N):
            blocking_time += self.get_total_blocking_time(P)
        return blocking_time

    def get_total_blocking_time(self, P):
        "Returns the total blocking time for a particle"
        all_blocking_periods = self.get_all_blocking_periods()
        return np.sum( all_blocking_periods[P]['Unblock Time'] - all_blocking_periods[P]['Block Time'] )

    def get_all_blocking_periods(self):
        """
        Returns the blocking periods (block / unblock) for each particle
        
        Return: list of pandas DataFrame
        The list has one element per particle P (0 <= P <= N-1) and the DataFrame has two columns:
        - 'Block Time': start time of blocking
        - 'Unblock Time': end time of blocking
        """
        if not self.is_simulation_finalized:
            raise Warning("The simulation has not been finalized..." \
                          "\nThe result of the total blocking time may be incorrect or give an error." \
                          "\nRun first the finalize() method and rerun.")

        # Blocking periods
        # NOTE: It's important to create SEPARATE data frames for each particle and NOT use [ pd.DataFrame ] * N
        # because the latter will generate just ONE data frame N times!! 
        blocking_times = []
        for P in range(self.N):
            blocking_times += [ pd.DataFrame.from_items([('Block Time', []), ('Unblock Time', [])]) ]
        for p, dict_info in enumerate(self.info_particles):
            P = dict_info['particle number']
            block_times_p = [t    for idx, t in enumerate(dict_info['t'])
                                if dict_info['E'][idx] == EventType.BLOCK]
            unblock_times_p = [t  for idx, t in enumerate(dict_info['t'])
                                if dict_info['E'][idx] == EventType.UNBLOCK]

            assert len(block_times_p) == len(unblock_times_p), \
                    "Particle {}: The number of blocking times ({}) is the same as the number of unblocking times ({})" \
                    .format(p, len(block_times_p), len(unblock_times_p))
            for b, u in zip(block_times_p, unblock_times_p):
                assert b < u, \
                        "The block time ({:.3f}) is always smaller than the unblock time ({:.3f})" \
                        .format(b, u)

            blocking_times[P] = pd.concat( [ blocking_times[P],
                                             pd.DataFrame({'Block Time': block_times_p,
                                                           'Unblock Time': unblock_times_p}) ], axis=0 )

        return blocking_times

    def get_all_total_survival_time(self):
        "Returns the total survival time for all particles"
        survival_time = 0.0
        for P in range(self.N):
            survival_time += self.get_total_survival_time(P)

        assert np.allclose(survival_time, np.sum(self.ktimes0_sum)), \
                "The sum of the survival period spans ({:.3f}) equals" \
                " the sum of the total survival times per particle computed during the simulation ({:.3f})" \
                .format(survival_time, np.sum(self.ktimes0_sum))

        return survival_time

    def get_total_survival_time(self, P):
        "Returns the total survival time for a particle (from position 0 back to position 0)"
        all_survival_periods = self.get_all_survival_periods()
        return np.sum(all_survival_periods[P]['Survival Period Span'])

    def get_all_survival_periods(self):
        """
        Returns the survival periods for each particle number P
        (these are the periods in which the particle goes from position 0 back to position 0).
        Censoring times are included as valid ending survival period times.
        
        Return: list of pandas DataFrame
        The list has one element per particle P (0 <= P <= N-1) and the DataFrame has two columns:
        - 'Survival Period End': end time of survival period
        - 'Survival Period Span': duration of survival period
        """

        # Survival periods for each particle number

        absorption_and_censoring_times = []
        survival_times = []
        # NOTE: It's important to create SEPARATE lists and data frames for each particle
        # and NOT use [ pd.DataFrame ] * N because the latter will generate just ONE data frame N times!! 
        for P in range(self.N):
            absorption_and_censoring_times += [ [0.0] ]
            survival_times += [ pd.DataFrame.from_items([('Survival Period End', []), ('Survival Period Span', [])]) ]
        for p, dict_info in enumerate(self.info_particles):
            P = dict_info['particle number']
            #with printoptions(precision=3):
            #    print("\np={}, P={}:\n{}".format(p, P, np.c_[ np.array(dict_info['t']), np.array(dict_info['E']) ]))
            absorption_and_censoring_times_p = [t   for idx, t in enumerate(dict_info['t'])
                                                    if dict_info['E'][idx] in [EventType.ABSORPTION, EventType.CENSORING]]
            absorption_and_censoring_times[P] += absorption_and_censoring_times_p

        for P in range(self.N):
            survival_times_P = np.diff(absorption_and_censoring_times[P])
            if len(survival_times_P) > 0:
                assert np.all(survival_times_P > 0), \
                        "The survival period spans for particle P={} are all positive ({})" \
                        .format(P, survival_times_P)

            survival_times[P] = pd.concat( [ survival_times[P],
                                             pd.DataFrame({'Survival Period End': absorption_and_censoring_times[P][1:],
                                                           'Survival Period Span': survival_times_P}) ], axis=0 )

        return survival_times

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
            #with printoptions(precision=3):
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
        if self.render_type == RenderType.GRAPH:
            K = self.queue.getCapacity()
            colormap = cm.get_cmap("jet")
            reflines = range(0, (K+1)*self.N, K+1)
            reflines_block = range(K, (K+(K+1))*self.N, K+1)
            max_time = 0.0
            particle_numbers = list(range(self.N))
            for p in particle_numbers:
                max_time = np.max([max_time, np.nanmax(self.all_times[p])])
            plt.figure()
            ax = plt.gca()
            ax.set_xlabel("t")
            ax.set_yticks(reflines)
            ax.yaxis.set_ticklabels(particle_numbers)
            ax.xaxis.set_ticks(np.arange(0, round(max_time)+1) )
            ax.set_ylim((0, (K+1)*self.N))
            ax.hlines(reflines_block, 0, max_time, color='gray', linestyles='dashed')
            ax.hlines(reflines, 0, max_time, color='gray')
            for p in particle_numbers:
                color = colormap( (p+1) / self.N )
                # Non-overlapping step plots at vertical positions (K+1)*p
                plt.step(self.all_times[p], [(K+1)*p + pos for pos in self.all_positions[p]], 'x-',
                         where='post', color=color, markersize=3)
            plt.title("K={}, rate(B)={:.1f}, rate(D)={:.1f}, reactivate={}, finalize={}, N={}, #iter={}, seed={}" \
                      .format(self.queue.getCapacity(),
                          self.queue.rates[Event.BIRTH.value],
                          self.queue.rates[Event.DEATH.value],
                          self.reactivate, self.finalize_type.name[0:3], self.N, self.niter, self.seed
                          ))
            ax.title.set_fontsize(9)
            plt.show()
        else:
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
    
            # Default rendering is TEXTUAL
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
                    "\nstart = {}" \
                    "\nreactivate = {}" \
                    "\nfinalize_type = {}" \
                    "\nniter = {}" \
                    "\nseed = {}" \
                    "\n***********************" \
                    .format(self.queue.getCapacity(),
                            self.queue.rates[Event.BIRTH.value], self.queue.rates[Event.DEATH.value],
                            self.queue.rates[Event.BIRTH.value] / self.queue.rates[Event.DEATH.value],
                            self.N, self.reactivate, self.finalize_type.name, self.niter, self.seed)
                            self.reactivate, self.finalize_type.name, self.niter, self.seed)
        return params_str

    def isLastIteration(self):
        return self.iter == self.niter

    def isValidParticle(self, P):
        "Returns whether the given particle is a valid particle number indexing the list of particle queues"
        return (isinstance(P, int) or isinstance(P, np.int32) or isinstance(P, np.int64)) and 0 <= P < self.N

    def isValidParticleId(self, p):
        "Returns whether the given particle is a valid particle ID indexing the info_particles attribute"
        return (isinstance(p, int) or isinstance(p, np.int32) or isinstance(p, np.int64)) and 0 <= p < len(self.info_particles)

    def assertTimeInsertedInTrajectories(self, P):
        assert  len(self.trajectories[P]['t']) == 0 and self.particles[P].getTimeLastEvent() > 0 or \
                len(self.trajectories[P]['t'])  > 0 and self.particles[P].getTimeLastEvent() > self.trajectories[P]['t'][-1], \
                "The time to insert in the trajectories dictionary ({}) for particle {}" \
                " is larger than the latest inserted time ({})" \
                .format(self.particles[P].getTimeLastEvent(),
                        len(self.trajectories[P]['t']) == 0 and 0.0 or self.trajectories[P]['t'][-1])

    def assertSystemConsistency(self, particle_number=None, time=None):
        is_state_consistent_particle = \
            lambda p: self.positions[p] == self.particles[p].size and self.positions[p] == self.all_positions[p][-1] and self.particles[p].getTimeLastEvent() == self.all_times[p][-1]

        if time is None:
            time_str = "time last change={:.3f}".format(self.time_last_change_of_system)
        else:
            time_str = "time={:.3f}".format(time)

        consistent_state = True
        if particle_number is None:
            for p in range(self.N):
                consistent_state = is_state_consistent_particle(p)              
                if not consistent_state:
                    particle_number = p
                    break
            if not consistent_state:
                self.render()
            assert consistent_state, \
                    "Iter {}, {}: The system is NOT in a consistent state at least for particle {}" \
                    .format(self.iter, time_str, particle_number)           
        else:
            consistent_state = is_state_consistent_particle(particle_number)
            if not consistent_state:
                self.render()
            assert consistent_state, \
                    "Iter {}, {}: The system is NOT in a consistent state for particle {}" \
                    "\n\tpositions[{}]={} vs. particles[{}].size={}" \
                    "\n\tpositions[{}]={} vs. all_positions[{}][-1]={}" \
                    "\n\tparticles[{}].TimeLastEvent={} vs. all_times[{}][-1]={}" \
                    .format(self.iter, time_str, particle_number,
                            particle_number, self.positions[particle_number], particle_number, self.particles[particle_number].size,
                            particle_number, self.positions[particle_number], particle_number, self.all_positions[particle_number][-1],
                            particle_number, self.particles[particle_number].getTimeLastEvent(), particle_number, self.all_times[particle_number][-1])         

    def raiseErrorInvalidParticle(self, p):
        raise ValueError("Wrong particle number: {}. Valid values are integers between 0 and {}.\n".format(p, self.N-1))

    def raiseWarningInvalidParticle(self, p):
        raise Warning("Wrong particle number: {}. Valid values are integers between 0 and {}.\n".format(p, self.N-1))
