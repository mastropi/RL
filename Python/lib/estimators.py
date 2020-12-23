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

class EstimatorQueueBlockingFlemingViot:
    """
    Estimator of the blocking probability of queues using the Fleming Viot particle system
    
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
        - apply_events()

    mean_lifetime: (opt) positive float
        Mean particle lifetime to be used as the expected survival time.
        This is useful when reactivate=True and start=1, as in that case
        the particles never touch position 0, which means that the mean lifetime
        cannot be computed.

    reactivate: (opt) bool
        Whether to reactivate a particle after absorption to a positive position.
        Note that absorption occurs when the particle goes to position (0, 0, ..., 0)
        where the array of zeros has length equal to the number of servers.
        That is, the particle is an ARRAY of server sizes, which should ALL be idle
        in order for the particle to be considered absorbed.

    finalize_type: (opt) FinalizeType
        Indicates what type of finalization of the simulation should be done on the active particles, either:
        - FinalizeType.ESTIMATE_CENSORED
        - FinalizeType.REMOVE_CENSORED
        - FinalizeType.ABSORB_CENSORED
        of which only "remove" and "absorb" are currently implemented.

    seed: (opt) int
        Random seed to use for the random number generation by numpy.random.

    plotFlag: (opt) bool
        Whether to plot the trajectories of the particles.

    log: (opt) bool
        Whether to show messages of what is happening with the particles.
    """
    def __init__(self, nparticles: int, niter: int, queue: GenericQueue,
                 mean_lifetime=None, reactivate=True, finalize_type=FinalizeType.REMOVE_CENSORED,
                 seed=None, plotFlag=False, log=False):
        if reactivate and nparticles < 2:
            raise ValueError("The number of particles must be at least 2 when reactivate=True ({})".format(nparticles))
            import sys
            sys.exit(-1)
        self.N = nparticles
        self.niter = niter
        self.queue = queue                  # This variable is used to create the particles as replications of the given queue
        self.nservers = self.queue.getNServers()

        #-- Parameter checks
        if queue.getCapacity() <= 1:
            # This condition must be satisfied because ONLY ONE EVENT is accepted for each event time
            # i.e. ACTIVATION, BLOCK, UNBLOCK or ABSORPTION.
            # If we allowed a queue capacity of 1, when the position of the particle goes to 1, the particle
            # is both ACTIVATED and BLOCKED, a situation that cannot be handled with the code as is written now.
            # In any case, a capacity of 1 is NOT of interest.   
            raise ValueError("The maximum position of the particles must be larger than 1 ({})".format(queue.getCapacity()))

        self.START = queue.getServerSizes() # Initial server size for each simulated particle (should be a numpy array)
        if np.sum(self.START) not in [0, 1]:
            raise ValueError("The start position for the particle (buffer size) should be either 0 or 1. These are the initial server sizes: {}" \
                             "\nProcess aborts." \
                             .format(self.START))
        #-- Parameter checks

        if mean_lifetime is not None and (np.isnan(mean_lifetime) or mean_lifetime <= 0):
            raise ValueError("The mean life time must be positive ({})".format(mean_lifetime))
        self.mean_lifetime = mean_lifetime  # A possibly separate estimation of the expected survival time

        self.reactivate = reactivate        # Whether the particle is reactivated to a positive position after absorption
        self.finalize_type = finalize_type  # How to finalize the simulation
        self.plotFlag = plotFlag
        self.seed = seed
        self.LOG = log

        # Epsilon time --> used to avoid repetition of times when reactivating a particle
        # This epsilon time takes into account the scale of the problem by making it 1E6 times smaller
        # than the minimum of the event rates.
        # I.e. we cannot simply set it to 1E-6... what if the event rate IS 1E-6??
        self.EPSILON_TIME = 1E-6 * np.min([ self.queue.getBirthRate(), self.queue.getDeathRate() ])

        #self.reset()    # Reset is called whenever a new simulation starts (see method simulate() below)
        
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
        # Its size is normally MUCH LARGER than N (the number of particles in the system)
        # because every reactivation adds a new element to this list!
        self.info_particles = []
        assert np.sum(self.START) in [0, 1], "The start position for the particle (buffer size) is either 0 or 1 ({})".format(self.START)
        if np.sum(self.START) == 0:
            start_event_type = EventType.ABSORPTION
        elif np.sum(self.START) == 1:
            start_event_type = EventType.ACTIVATION
        for P in range(self.N):
            self.trajectories += [ dict({
                                           't': [np.repeat(0.0, self.nservers)],          # event times at which the particle changes position (in any of the servers)
                                           'x': [self.START],                             # positions taken by the particle after each event (recall that self.START is an array having the size of each server at start time)
                                           'e': [np.repeat(Event.RESET, self.nservers)]   # events (birth, death) associated to the event times
                                           }) ]
            self.info_particles += [ dict({
                                           't': [np.repeat(0.0, self.nservers)],                  # times at which the events of interest take place
                                           'E': [np.repeat(start_event_type, self.nservers)],     # events of interest associated to the times 't'
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
            if start_event_type == EventType.ABSORPTION:
                # Remove the first event added above when particles start at 0
                # in order to avoid failing the assertions stating that the first
                # event in the info_particles dictionaries is an ACTIVATION. 
                self.info_particles[P]['t'].pop(0)
                self.info_particles[P]['E'].pop(0)
        # List stating whether the corresponding particle number has been absorbed already at least once
        # This is used when updating the dict_info_absorption_times dictionary, as this should contain
        # only one absorption time per particle.
        self.particle_already_absorbed = [False]*self.N
        # List of reactivation IDs indexing the info_particles list
        # where we should look for the information (relevant events and their times)
        # about each particle number.
        # These IDs start each being in the range [0, N-1], but as the simulation progresses
        # and reactivation occurs, start increasing beyond N-1, because each reactivation
        # implies a new ID. So the first reactivated particle will have ID = N, the second reactivate
        # particle will have ID = N+1, and so forth.
        # *** These IDs are represented with lower letters in the code (e.g. `q`) ***
        self.particle_reactivation_ids = list(range(self.N))    # This is [0, 1, ..., N-1]

        # Information about the absorption times, which is used to regenerate the trajectory positions
        # (but not trajectory event times!) of an absorbed particle.
        # This dictionary initially (i.e. before every re-activating one particle) contains the
        # FIRST absorption times of each particle, once their trajectories are fully generated for
        # a pre-defined number of simulation iterations. Once the regeneration of trajectory positions
        # starts, the dictionary starts containing the next absorption time for the regenerated particle.
        # For more information see the regenerate_trajectories_iteratively_until_all_absorption_times_are_exhausted()
        # function or similar.
        self.dict_info_absorption_times = dict({'t': [],        # List of absorption times 
                                                'P': [],        # List of particle numbers associated to each absorption time in 't'
                                                'iter': []})    # List of iterations at which the absorption times occur

        # Active particle numbers at the end of the simulation (it may contain values between 0 and N-1 
        self.is_particle_active = [False]*self.N
        #-- Attributes used in the SIMPLER implementation


        # Positions or state of each particle
        self.positions = np.zeros((self.N, self.nservers), dtype=int)

        # History of all positions and event times for each particle 
        self.all_positions = []
        self.all_times = []
        self.times_next_events = np.zeros((self.N, self.nservers, 2), dtype=float)  # 2 is the number of events: birth and death
        for P in range(self.N):
            self.all_positions += [ [self.START] ]  # Recall: self.START is an ARRAY with the size of each server in the system
            self.all_times += [ [np.repeat(0.0, self.nservers)] ]
            self.times_next_events[P] = np.c_[self.queue.generate_event_times(Event.BIRTH), self.queue.generate_event_times(Event.DEATH)]
        # The following returns an array the same dimension as its input array
        # having as content the indices that order the values (when indices are read from left to right)
        # along EACH 1D array in the last dimension (in this case third dimension),
        # which contain precisely what we want to sort (the birth-death event times of each server).
        # So for instance, if the data are:
        # x = [3.2, 9.7, 0.1, 5.3]
        # the order array is:
        # o = [2, 0, 3, 1]
        # to NOT be confused with the ranks which give the rank of each value in x:
        # r = [1, 3, 0, 2]
        # Therefore:
        # x[o] = sort(x)
        # r[o] = sort(r)
        self._order_times_next_events = self._compute_order( self.times_next_events )

        # 2) Compute the times of the next birth and death for each queue/particle, and their order
        #self.times_next_events = [ [self.queue.generate_birth_time(), self.queue.generate_death_time()]
        #                            for _ in np.arange(self.N) ]
        #self._order_times_next_events = [ self._compute_order( self.times_next_events[p] )
        #                                    for p in np.arange(self.N) ]

        # Arrays that are used to estimate the expected survival time (time to killing from position 0)
        if start_event_type == EventType.ABSORPTION:
            # Latest times the particles changed to position 0 is 0.0 as they start at position 0
            self.times0 = np.zeros(self.N, dtype=float)
        else:
            # Latest times the particles changed to position 0 is unknown as they don't start at position 0
            self.times0 = np.nan * np.ones(self.N, dtype=float)
        self.ktimes0_sum = np.zeros(self.N, dtype=float)   # Times to absorption from latest time it changed to position 0
        self.ktimes0_n = np.zeros(self.N, dtype=int)       # Number of times the particles were absorbed

        # 3) Attributes used directly in the computation of the estimated measures that contribute to the probability of blocking.
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
        # TODO: Update this value as the simulation goes forward
        self.time_last_change_of_system = 0.0

        if self.LOG:
            print("Particle system with {} particles has been reset:".format(self.N))
            print("")

    #--------------------------------- Functions to simulate ----------------------------------
    def simulate(self):
        """
        Simulates the system of particles and estimates the blocking probability
        via Approximation 1 and Approximation 2 of Matt's draft on queue blocking dated Apr-2020.

        Return: tuple
        The tuple contains the following elements:
        - the estimated blocking probability
        - the estimated expected blocking time given the particle starts at position 1 and is not absorbed
          (this is the numerator of the expression giving the blocking probability estimate)
        - the estimated expected survival time (time to first absorption) given the particle starts at position 0.
          (this is the denominator of the expression giving the blocking probability estimate)
        """
        self.reset()

        self.generate_trajectories()

        self.finalize()
        self.compute_counts()

        proba_blocking_integral, proba_blocking_laplacian, integral, gamma, expected_survival_time = \
            self.estimate_proba_blocking()
        self.proba_blocking_integral = proba_blocking_integral
        self.proba_blocking_laplacian = proba_blocking_laplacian

        if self.plotFlag:
            self.plot_trajectories()

        return self.proba_blocking_integral, self.proba_blocking_laplacian, integral, gamma, expected_survival_time

    def generate_trajectories(self):
        # NOTE: We start at iteration 1 because we consider 0 to be the iteration for the starting particle positions
        for it in range(1, self.niter+1):
            self.generate_one_iteration(it)

        if self.LOG: #True:
            print("\n******** Info PARTICLES before REACTIVATION:")
            for P in range(self.N):
                with printoptions(precision=3, suppress=True):
                    print("Particle {}".format(P))
                    print(np.c_[np.array(self.info_particles[P]['t']), np.array(self.info_particles[P]['E'])])

        if self.reactivate:
            self.regenerate_trajectories_iteratively_until_all_absorption_times_are_exhausted()

    def generate_one_iteration(self, it):
        "Generates the next iteration, i.e. the next change of state in the system"
        self.iter = it
        for P in range(self.N):
            #time_of_event, position_change = self._update_one_particle(P)
            time_of_event, position_change = self._update_one_particle_by_server(P)
            self._update_trajectories(P)
            self._update_info_particles(P, position_change)
            if not self.reactivate and self._is_particle_absorbed(P, position_change) or \
               self.reactivate and self._is_particle_absorbed(P, position_change) and not self.particle_already_absorbed[P]:
                # In the REACTIVATE case (i.e. Fleming-Viot algorithm), only record the absorption time
                # if it is the first absorption of the current particle (in the non-reactivate case, record all absorption times).
                # In fact, when applying Fleming-Viot, we are going to re-generate the trajectory positions
                # (but NOT their occurrence times) after the first absorption occurs for a particle
                # (see below function regenerate_trajectories_iteratively_until_all_absorption_times_are_exhausted() or similar).
                self.particle_already_absorbed[P] = True
                self._update_info_absorption_times(P, time_of_event, it)
                self._update_survival_time_from_zero(P, time_of_event)

            self.time_last_change_of_system = np.max(time_of_event)
            self.assertSystemConsistency(P, self.time_last_change_of_system)

            # TODO: (2020/06/22) Add assertions about the information stored in the above objects
            # (see notes on loose sheets of paper)
            assert sorted(self.dict_info_absorption_times['t']) == self.dict_info_absorption_times['t'], \
                    "The absorption times in the dict_info_absorption_times dictionary are sorted" \
                    "\n{}".format(self.dict_info_absorption_times['t'])
            if self.reactivate:
                assert  len(self.dict_info_absorption_times['P']) == 0 or \
                        list(np.unique(self.dict_info_absorption_times['P'])) == sorted(self.dict_info_absorption_times['P']), \
                        "The particles associated to the absorption times in the dict_info_absorption_times dictionary are unique" \
                        "\n{}".format(self.dict_info_absorption_times['P'])

    def regenerate_trajectories_iteratively_until_all_absorption_times_are_exhausted(self):
        """
        Start with the particle having the smallest FIRST absorption time in the already generated trajectories
        and regenerate the trajectory positions (but NOT its times, so that we don't need to generate the times again!)
        from that time onwards., until the particle is absorbed again, recording its new absorbing time
        in the dictionary of "first absorbing times for each particle" (self.dict_info_absorption_times or similar).

        Continue until all absorption times (original first ones and newly observed during this process)
        are exhausted (i.e. until the self.dict_info_absorption_times dictionary becomes empty). 
        """
        # Helper function for an assertion below
        has_been_absorbed_at_time_t = lambda P, t: len(self.trajectories[P]['x']) == self.niter + 1 and self.trajectories[P]['t'][-1] < t and self.trajectories[P]['x'][-1] == 0
            ## NOTE: For this function to return what the name says, the 'x' attribute of the dictionary
            ## stored for each particle in self.trajectories must increase its size as the simulation progresses
            ## (in fact, the function checks whether the length of this attribute is equal to the number of iterations)
            ## (as opposed to filling it up upfront when generating all the events for each particle
            ## at the beginning of this method regenerate_trajectories_iteratively_until_all_absorption_times_are_exhausted())    

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
            # Remove the first element of the dictionary which contains all the absorption times
            # and respective particle numbers
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
                #assert self.info_particles[ self.particle_reactivation_ids[P] ]['E'][0] == EventType.ACTIVATION, \
                #        "The first relevant event of the reactivated particle (p={}) for particle P={}" \
                #        " is an ACTIVATION ({})" \
                #        .format(self.particle_reactivation_ids[P], P,
                #                self.info_particles[ self.particle_reactivation_ids[P] ]['E'][0].value)

            assert len(self.dict_info_absorption_times['t']) <= initial_number_absorption_times, \
                    "The length of the list containing the absorption times ({}) is" \
                    " <= the initial number of absorption times ({})" \
                    .format(len(self.dict_info_absorption_times['t']), initial_number_absorption_times)

    def _update_one_particle(self, P):
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
        P: non-negative int
            Index of the particle to update, from 0 to N-1, where N is the number of particles in the system.
        
        Return: tuple
        The tuple contains:
        - the time of the selected event, measured in absolute terms, i.e. from the common origin of all particles.
        - the change in position after applying the event, which can be -1, 0 or +1.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        
        # Find the event type to apply (whichever occurs first)
        time_of_event, time_since_previous_event, earliest_event = self.get_time_next_event(P)

        # Update the particle's state by applying the event type just selected and record the position change
        position_change = self._apply_event(P, Event(earliest_event), time_of_event)
        latest_event = self._order_times_next_events[P][1]

        # Generate the next time for the event type just applied from the list of event times for this particle
        self.times_next_events[P][earliest_event] = self.particles[P].generate_event_times(Event(earliest_event), size=1)

        # Update the time-to-event for the event type not applied so that it is now relative to the event just applied
        self.times_next_events[P][latest_event] -= time_since_previous_event

        # Update the order of the next event times
        self._order_times_next_events[P] = self._compute_order( self.times_next_events[P] )

        #print("UPDATED times_next_events[{}]: {}".format(P, self.times_next_events[P]))

        return time_of_event, position_change

    def _update_one_particle_by_server(self, P):
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
        P: non-negative int
            Index of the particle to update, from 0 to N-1, where N is the number of particles in the system.
        
        Return: tuple
        The tuple contains:
        - the time of the selected event, measured in absolute terms, i.e. from the common origin of all particles.
        - the change in position after applying the event, which can be -1, 0 or +1.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        
        # Find the event type to apply (whichever occurs first)
        time_of_event_by_server, time_since_previous_event_by_server, earliest_event_by_server = self.get_time_next_event_by_server(P)
        type_of_event_by_server = [Event(e) for e in earliest_event_by_server]

        # Update the particle's state by applying the event types to each server and record the position change
        position_change_by_server = self._apply_event_by_server(P, type_of_event_by_server, time_of_event_by_server)
        latest_event_by_server = self._order_times_next_events[P,:,1] # `:` indicates ALL servers

        # Generate the times of the next events of each applied event type to each server and
        # update the time-to-event of the respective event types not applied
        # so that they are now relative to the events just applied.
        next_event_times_by_server = self.particles[P].generate_event_times(type_of_event_by_server)
        for server in range(self.nservers):
            # Generate the next event times for the event type just applied for each server
            self.times_next_events[P][server][earliest_event_by_server[server]] = next_event_times_by_server[server]
            # Update the time-to-events for the event type not applied for each server
            self.times_next_events[P][server][latest_event_by_server[server]] -= time_since_previous_event_by_server[server]

        # Update the order of the next event times
        self._order_times_next_events[P] = self._compute_order( self.times_next_events[P] )

        #print("UPDATED times_next_events by server [P={}]: {}".format(P, self.times_next_events[P]))
        #print("POSITION change by server [P={}]: {}".format(P, position_change_by_server))
        #print("\n")

        return time_of_event_by_server, position_change_by_server

    def _compute_order(self, arr, axis=-1):
        # axis=-1 means "the last dimension of the array"
        return np.argsort(arr, axis=axis)

    def _apply_event(self, P, event_to_apply, time_of_event):
        position_change = self.particles[P].apply_event(event_to_apply, time_of_event)
        self._update_particle_position_info(P, self.particles[P].getServerSizes(), time_of_event)
        return position_change

    def _apply_event_by_server(self, P, event_types, event_times):
        position_change = self.particles[P].apply_events(event_types, event_times)
        self._update_particle_position_info(P, self.particles[P].getServerSizes(), event_times)
        return position_change

    def _update_particle_position_info(self, P, new_position :np.ndarray, time_at_new_position :np.ndarray):
        self.positions[P] = new_position
        self.all_positions[P] += [self.positions[P]]
        self.all_times[P] += [time_at_new_position]

    def _update_trajectories(self, P):
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        self.assertTimeInsertedInTrajectories(P)
        self.trajectories[P]['t'] += [self.particles[P].getTimeLastEvent()]
        if not self.reactivate or \
           self.reactivate and not self.particle_already_absorbed[P]:
            self.trajectories[P]['x'] += [self.particles[P].getServerSizes()]
        self.trajectories[P]['e'] += [self.particles[P].getTypeLastEvent()]
        
        # Update the list stating whether each particle is active
        self._update_active_particles(P)

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
        self.is_particle_active[P] = all(self.trajectories[P]['x'][-1] > 0)

    def _update_info_particles(self, P, last_position_change):
        assert self.isValidParticle(P), "The particle to update is valid (0<=P<{}) ({})".format(self.N, P)

        #--------------- Helper functions -------------------
        # Possible status change of a particle 
        is_activated = lambda P: self._is_particle_activated(P)
        is_absorbed = lambda P: self._is_particle_absorbed(P, last_position_change)
        is_blocked = lambda P: self._is_particle_blocked(P, last_position_change)
        is_unblocked = lambda p: self._is_particle_unblocked(P, last_position_change) 

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

    def _is_particle_activated(self, P):
        # Note that a particle can be activated even if it was ALREADY at position 1 at some earlier event time => a new sub-trajectory becomes active
        return self.particles[P].getBufferSize() == 1

    def _is_particle_absorbed(self, P, last_position_change):
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        return any(last_position_change < 0) and self.particles[P].getBufferSize() == 0

    def _is_particle_blocked(self, P, last_position_change):
        return any(last_position_change > 0) and self.particles[P].getBufferSize() == self.particles[P].getCapacity()        

    def _is_particle_unblocked(self, P, last_position_change):
        return any(last_position_change < 0) and self.particles[P].getBufferSize() == self.particles[P].getCapacity() + np.sum(last_position_change)

    def _update_info_absorption_times(self, P, time_of_absorption, it):
        "Inserts a new absorption time in order, together with its particle number and iteration number"
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        idx_insort, found = insort(self.dict_info_absorption_times['t'], time_of_absorption)
        self.dict_info_absorption_times['P'].insert(idx_insort, P)
        self.dict_info_absorption_times['iter'].insert(idx_insort, it)

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

    def _choose_assigned_particle(self, P):
        list_of_particles_to_choose_from = list(range(P)) + list(range(P+1, self.N))
        chosen_particle_number = list_of_particles_to_choose_from[ np.random.randint(0, self.N-1) ]

        assert self.isValidParticle(chosen_particle_number) and chosen_particle_number != P, \
                "The chosen particle is valid (0 <= Q < {} and different from {}) ({})" \
                .format(self.N, P, chosen_particle_number)
        return chosen_particle_number

    def _copy_info_from_particle_at_given_time(self, P, Q, t, iter):
        """
        Copies the information from particle P at time t to particle Q.
        
        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) corresponding to the absorbed particle.

        Q: non-negative int
            Particle number (index of the list of particle queues) corresponding to the particle
            assigned to the absorbed particle to be reactivated.
        
        t: positive float
            Time at which the information of particle Q is retrieved (e.g. its positions)
            and copied to particle P.

        Return: numpy array of non-negative int
        Position of particle Q at the given time t.
        """
        position_Q = self.get_position(Q, t)
        self.particles[P].resize(position_Q, t)
        self._copy_info_particle(P, Q, t, position_Q, iter)

        # Update the particle ID associated to the particle number P
        # to the particle ID just added by _copy_info_particle() to the info_particles list.
        # This ID is the last index added to the info_particles list
        # and represents the new particle ID associated to the particle number P
        # since its associated particle ID prior to reactivation (i.e. prior to the copy done here)
        # is no longer active as that particle ID is associated to an ABSORBED particle.
        self.particle_reactivation_ids[P] = len(self.info_particles) - 1

        return position_Q

    def _copy_info_particle(self, P, Q, t, position_Q, iter):
        """
        Assigns the given position of particle Q to the absorbed particle P.
        The information is stored in a new entry in the info_particles list
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

        assert self.isValidParticle(P) and self.isValidParticle(Q), \
                "The particle numbers representing the absorbed (P) and reactivated (Q) particles are valid" \
                "\n(0<=P<{}, 0<=Q<{}) (P={}, Q={})" \
                .format(self.N, self.N, P, Q)
                
        # Particle ID in the info_particles list associated to particle number Q
        # to which the absorbed particle P is reactivated.
        q = self.particle_reactivation_ids[Q]
        assert self.isValidParticleId(q), "The particle ID is valid (0<=q<{}) ({})" \
                .format(len(self.info_particles), q)

        # Add a new entry to the info_particles list that represents the reactivated particle
        # Its position is assigned to the position of the reassigned particle Q at time t.
        self.info_particles += [ dict({'t': [],
                                       'E': [],
                                       't0': t,
                                       'x': position_Q,
                                       'iter': iter,
                                       'particle number': P,
                                       'reactivated number': Q,
                                       'reactivated ID': q}) ]

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

        position_start: array of non-negative int
            Start position of the reactivated particle from which the trajectory continues.
            To accommodate the general case of a particle representing c servers, the position is an array
            whose content is the queue size on each server. 

        iter: non-negative int
            Iteration number at which the reactivation takes place from the given start position.

        Return: tuple
        The tuple contains:
        - the last iteration number at which the trajectory was updated
        - the last change of position experienced by the particle.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        assert 0 <= np.sum(position_start) <= self.particles[P].getCapacity(), \
                "The start positions of the reactivated particle {} ({}) do not exceed the capacity of the buffer ({})" \
                .format(P, position_start, self.particles[P].getCapacity())
        assert all(self.particles[P].size == position_start), \
                "The start position ({}) of the reactivated particle {} is equal to the given start position ({})" \
                .format(self.particles[P].size, P, position_start)
        assert iter < self.niter, \
                "The iteration number at which reactivation takes place ({}) is NOT the last iteration ({})" \
                .format(iter, self.niter)

        self.trajectories[P]['x'][iter] = position_start
        self.all_positions[P][iter] = position_start
        self._update_info_particles(P, position_start)
        for it in range(iter+1, self.niter+1):
            #position_change = self._update_particle_trajectory(P, it)
            position_change = self._update_particle_trajectory_by_server(P, it)
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
        self.trajectories[P]['x'] += [self.particles[P].getServerSizes()]

        # Update the `positions` lists
        self.positions[P] = self.particles[P].getServerSizes()
        self.all_positions[P][it] = self.particles[P].getServerSizes()

        return position_change

    def _update_particle_trajectory_by_server(self, P, it):
        """
        Updates the trajectory of the given particle number at the given iteration
        and returns the change in position.
        """
        # Retrieve the event type and time as this information has been already generated
        # at the very beginning of the simulation by the call to the generate_trajectories() method.
        next_event = self.trajectories[P]['e'][it]
        next_event_time = self.trajectories[P]['t'][it]

        # Apply the event to the particle (queue)
        position_change_by_server = self._apply_event_by_server(P, next_event, next_event_time)

        # Update the trajectory position at the retrieved event time
        # IMPORTANT: Note that the 'x' values (i.e. the positions of the trajectory)
        # need to be generated SEPARATELY from the 't' and 'e' values because these latter two values
        # have ALREADY BEEN GENERATED when generating one trajectory for each particle
        # (by the initial part of generate_trajectories()).
        # Thus, here we are adding the POSITION ('x') of the particle after it has been absorbed.
        self.trajectories[P]['x'] += [self.particles[P].getServerSizes()]

        # Update the `positions` lists
        # TODO: (2020/12/23) In principle we should be able to get rid of these updates
        # because their values are set by the call to _apply_event_by_server() above.
        # In order to check this, we should run a unit test and see of the test still passes after removing these two lines.
        self.positions[P] = self.particles[P].getServerSizes()
        self.all_positions[P][it] = self.particles[P].getServerSizes()

        return position_change_by_server

    def finalize(self):
        """
        Finalizes the simulation process by treating particles that are active,
        i.e. those particles providing CENSORED survival time values.
        
        The type of finalize process is defined by the object's attribute `finalize_type`.
        """

        active_particle_numbers = self.get_active_particle_numbers()
        active_particle_ids = list( np.sort([ self.particle_reactivation_ids[P] for P in active_particle_numbers ]) )

        if self.LOG:
            if self.finalize_type == FinalizeType.ABSORB_CENSORED:
                finalize_process = "absorbing"
            elif self.finalize_type == FinalizeType.REMOVE_CENSORED:
                finalize_process = "removing"
            elif self.finalize_type == FinalizeType.ESTIMATE_CENSORED:
                finalize_process = "censoring"
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
            #print("Processing particle ID p={}, P={} out of {} particles".format(p, P, len(active_particle_ids)))
            if self.finalize_type == FinalizeType.REMOVE_CENSORED:
                if self.reactivate:
                    if self.LOG:
                        print("...{} particle p={} (P={})".format(finalize_process, p, P))
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
                    if self.LOG:
                        print("...{} the active tail of particle P={}".format(finalize_process, P))
                    # Find the latest absorption time and remove all the events since then
                    idx_last_absorption = find_last(dict_info['E'], EventType.ABSORPTION)
                    # Remove all the elements AFTER the index of the last absorption to the end
                    # Note that if no absorption event is found, ALL the trajectory is removed
                    # and this works fine because find_last() returns -1 when the searched value
                    # is not found in the list.
                    while len(dict_info['E']) > idx_last_absorption + 1:
                        dict_info['t'].pop(idx_last_absorption+1)
                        dict_info['E'].pop(idx_last_absorption+1)
                    # NOTE that no update needs to be done to the quantities
                    # used to estimate E(T) (e.g. ktimes0_sum) because these quantities are updated
                    # ONLY when there is an absorption, and this already happened
                    # at the latest absorption time. Since we are removing everything
                    # that happened after that, we do not need to update neither
                    # ktimes0_sum, nor ktimes0_n, nor times0.   
                # Flag the particle as inactive (as the latest censored trajectory was removed)
                self.is_particle_active[P] = False

            elif self.finalize_type in [FinalizeType.ABSORB_CENSORED, FinalizeType.ESTIMATE_CENSORED]:
                if self.LOG:
                    print("...{} particle p={}".format(finalize_process, p))
                # Increase slightly and randomly the time to insert the new "fictitious" event
                # in order to avoid repeated times in the list of time segments
                # (the random component is especially important here as it would affect
                # the insertion of a survival time segment which is done RELATIVE to
                # valid earlier activation events)
                # Note that having repeated time values messes up the process that merges the
                # two list of times needed to compute the blocking probability, namely:
                # - the list of alive time segments
                # - the list of block/unblock time segments
                time_to_insert = self.trajectories[P]['t'][-1] + self.EPSILON_TIME*np.random.random()

                # Final event which can be either an absorption or a censoring time
                if self.finalize_type == FinalizeType.ABSORB_CENSORED:
                    # To avoid assertion failures... make sure that, at the end of the simulation:
                    # - the particle is unblocked if it was blocked
                    #   (adding this preserves the assertion that the number of
                    #   blocked particles at the very last time is 0 --since the particles are all absorbed!)
                    # - the particle is activated if it was not in position 1
                    #   (this makes things logical --i.e. before absorption the particle passes by position 1)
                    if self.trajectories[P]['x'][-1] == self.queue.getCapacity():
                        dict_info['t'] += [time_to_insert]
                        dict_info['E'] += [EventType.UNBLOCK]
                        # Increase the time to insert for the insertion of the next event
                        time_to_insert += self.EPSILON_TIME*np.random.random()
                    if self.trajectories[P]['x'][-1] != 1:
                        dict_info['t'] += [time_to_insert]
                        dict_info['E'] += [EventType.ACTIVATION]
                        # Increase the time to insert for the insertion of the
                        # fictitious absorption event coming next (to avoid time value repetitions)
                        time_to_insert += self.EPSILON_TIME*np.random.random()
                    dict_info['t'] += [time_to_insert]
                    dict_info['E'] += [EventType.ABSORPTION]
                    # Flag the particle as inactive (as it was just absorbed)
                    self.is_particle_active[P] = False
                elif self.finalize_type == FinalizeType.ESTIMATE_CENSORED:
                    dict_info['t'] += [time_to_insert]
                    dict_info['E'] += [EventType.CENSORING]  

                # Survival time from s=0
                self._update_survival_time_from_zero(P, time_to_insert)

        self.is_simulation_finalized = True

    def compute_counts(self):
        """
        Computes the survival time segments and blocking time segments needed for
        the calculation of P(T>t / s=1) and P(BLOCK / T>t, s=1). 
        """
        if self.LOG:
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
            #print("Processing particle ID p={}, P={} out of {} particles".format(p, P, len(self.info_particles)))
            assert len(dict_info['E']) == 0 or dict_info['E'][-1] in [EventType.ABSORPTION, EventType.CENSORING], \
                    "Either particle p={} has NO events" \
                    " or the last event is an ABSORPTION or a CENSORING ({})" \
                    .format(p, dict_info['E'][-1])

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
                    #assert len(activation_times) == 0 and event_type_prev in [None, EventType.ABSORPTION] or \
                    #       len(activation_times)  > 0, \
                    #        "The activation_times list is empty only when it's the first event" \
                    #        " or when the previous event was an ABSORPTION (prev={}, atimes={})" \
                    #        .format(event_type_prev, activation_times)  
                    activation_times += [t]
                elif event_type == EventType.ABSORPTION:
                    assert event_type_prev == EventType.ACTIVATION and len(activation_times) > 0, \
                            "The previous event of an ABSORPTION is always an ACTIVATION ({}, p={}, P={}, t={}))" \
                            .format(event_type.name, p, P, t) 
                    self.insert_relative_time(t, activation_times, event_type)
                    activation_times = []                    
                    if True:
                        # We may want to disable this assertion if they take too long
                        assert sorted(self.sk) == list(np.unique(self.sk)), \
                                "The list of survival time segments contains unique values" \
                                " after insertion of event {} for particle p={}, P={}" \
                                .format(event_type.name, p, P)
                elif event_type in [EventType.BLOCK, EventType.UNBLOCK]:
                    # We insert the ABSOLUTE time for BLOCK and UNBLOCK events
                    # since this is used to compute the empirical distribution at the absolute time t
                    self.insert_absolute_time(t, event_type)
                    if event_type == EventType.UNBLOCK:
                        assert event_type_prev == EventType.BLOCK, \
                                "The event coming before an UNBLOCK event is a BLOCK event ({})" \
                                .format(event_type_prev.value)
                    if True:
                        # We may want to disable this assertion if they take too long
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
        if self.finalize_type in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
            assert self.counts_alive[-1] == 0, "The last element of the counts_alive list is 0 ({})".format(self.counts_alive[-1])
            assert self.counts_blocked[0] == 0, "The first element of the counts_blocked list is 0 ({})".format(self.counts_blocked[0])

        if self.LOG:
            with printoptions(precision=3, suppress=True):
                print("Relative absorption times and counts:\n{}".format(np.c_[np.array(self.sk), np.array(self.counts_alive)]))
                print("Relative blocking times:\n{}".format(np.c_[np.array(self.sbu), np.array(self.counts_blocked)]))

    def insert_absolute_time(self, t: float, event_type: EventType):
        assert event_type in [EventType.BLOCK, EventType.UNBLOCK], \
                "The event type when inserting the relative time for time t={}" \
                " is either BLOCK or UNBLOCK ({})" \
                .format(t, event_type.name)
        idx_insort, found = insort(self.sbu, t, unique=True)
        self._update_counts_blocked(idx_insort, event_type, new=not found)
        assert not found, "The absolute time value is NOT found in the list of block/unblock time  ({})".format(t)

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
            assert event_type in [EventType.ABSORPTION, EventType.CENSORING], \
                    "The event type when inserting the relative time for time t={}" \
                    " is either ABSORPTION or CENSORING ({})" \
                    .format(t, event_type.name)
            idx_insort, found = insort(self.sk, s)
            assert not found, "The relative time value is NOT found in the list of survival time segments ({})".format(s)
            self._update_counts_alive(idx_insort, event_type)

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
            assert 0 <= self.counts_blocked[idx] <= self.N, \
                    "The number of blocked particles in time segment with index idx={} is between 0 and {} ({}) \n({})" \
                    .format(idx, self.N, self.counts_blocked[idx], self.counts_blocked) 
    #--------------------------------- Functions to simulate ----------------------------------


    #----------------------------- Functions to analyze the simulation ------------------------
    def compute_survival_probability_from_counts(self, counts_alive: list):
        """
        Computes the survival probability (Pr(T>t)) at each time step where counts change.
        The input data is the array containing the counts of alive particle by time step.
        """
        if len(self.t) <= 1:
            return [1.0]
        
        assert counts_alive is not None and len(counts_alive) > 0, \
                    "The input list is not None and has at least 1 element".format(counts_alive)
        # I comment out the following assertion because sorting takes time...
        #assert sorted(counts_alive, reverse=True) == counts_alive, \
        #            "The input array with the number of particles alive measured at every death time is sorted non-increasingly"
        assert counts_alive[0] > 0, "The number of particles alive at t=0 is positive ({})".format(counts_alive[0])

        return [n_survived / counts_alive[0] for n_survived in counts_alive]        

    def compute_probability_from_counts(self, counts):
        """
        Computes an empirical probability of an event by time using an array of event counts
        at each time step the count changes.
        """
        if len(self.t) <= 1:
            return [0.0]
        return [n / self.N for n in counts]        

    def compute_killing_rate(self):
        """
        Computes the killing rate of the survival process stored in the object
        at each time step where the number of particles alive change.
        """
        if len(self.t) <= 1:
            return np.nan
        return [-np.log(p)/t    if t > 0 and p > 0
                                else np.nan
                                for t, p in zip(self.t, self.proba_surv_by_t)]        

    def estimate_QSD_at_block(self):
        """
        Smoothes the estimate of the Quasi-Stationary distribution (QSD), nu(K), given by phi(t,K)
        --the estimated probability of blocking given survival-- according to Matt's draft describing
        Approximation 2 for the estimation of the blocking probability,
        by averaging a set of values of phi(t,K) over a specified time interval for large times 
        (i.e. over times towards the end of the simulation). 
        The goal is to make that value phi(t,K) more stable, which, if we didn't average, would be
        measure at one single time t, which may oscillate quite a lot for different simulations or
        slightly different t values.
        For details, see graphs produced by the algorithm and shown at the meeting minutes document.
        
        The average is computed over a range of times defined by hard-coded parameters which define
        two the start and end time of the range as a proportion of the number of time steps where
        a change happens during the simulation, as e.g.:
        proportion of start = 0.25 
        proportion of stop = 0.75 
        """ 
        proportion_of_length_to_start_integrating = 0.25
        proportion_of_length_to_stop_integrating = 0.75
        idx_tfirst = int( proportion_of_length_to_start_integrating * len(self.proba_surv_by_t) )
        idx_tlast = int( proportion_of_length_to_stop_integrating * len(self.proba_surv_by_t) )
        pblock_integral = 0.0
        for i in range(idx_tfirst, idx_tlast-1):
            pblock_integral += self.proba_block_by_t[i] * (self.t[i+1] - self.t[i])
        timespan = self.t[idx_tlast] - self.t[idx_tfirst]
        pblock_mean = pblock_integral / timespan

        return pblock_mean

    def compute_blocking_time_estimate(self):
        """
        Computes Pr(K) * (Expected Survival time given start at position 0) where
        Pr(K) is the calculation of the blocking probability using Approximation 2 in Matt's draft.
        
        A modification is done on this estimate to increase robustness or stability of the estimation,
        namely the phi(t,K) probability of blocking given survival is averaged over a time interval
        picked over large times (i.e. the times towards the end of the simulation). 
        """
        if  len(self.t) <= 1:
            return np.nan

        pblock_mean = self.estimate_QSD_at_block()

        return [1/gamma * np.exp(gamma*t) * psurv * pblock_mean
                for t, gamma, psurv in zip(self.t, self.gamma, self.proba_surv_by_t)]        

    def estimate_proba_survival_given_position_one(self):
        """
        Computes the survival probability given the particle started at position 1
        
        A data frame is returned containing the times at which the probability is estimated
        and the probability estimate itself.
        """
        assert len(self.sk) > 1, "The length of the survival times array is at least 2 ({})".format(self.sk)
        assert self.sk[-1], "The last value of the survival time segments array is 0.0 ({:.2f})".format(self.sk[-1])
        assert len(self.counts_alive) == len(self.sk), \
                "The number of elements in the survival counts array ({})" \
                " is the same as the number of elements in the survival time segments array ({})" \
                .format(len(self.counts_alive), len(self.sk))
        assert self.counts_alive[0] == len(self.sk) - 1, \
                "The number of particles alive at t=0 (counts_alive[0]={})" \
                " is equal to S-1, where S is the number of elements in the survival time segments array (len(sk)-1={})" \
                .format(self.counts_alive[0], len(self.sk)-1)

        # NOTE: The list containing the survival probability is NOT stored in the object
        # because the one we store in the object is the list that is aligned with the times
        # at which the conditional blocking probability is measured.
        proba_surv_by_t = self.compute_survival_probability_from_counts(self.counts_alive)
        return pd.DataFrame.from_items([('t', self.sk), ('P(T>t / s=1)', proba_surv_by_t)])

    def estimate_proba_survival_and_blocking_conditional(self):
        """
        Computes the following quantities which are returned in a data frame:
        - time at which a change in any quantity happens
        - P(T>t / s=1): survival probability given start position = 1
        - P(BLOCK / T>t,s=1): blocking probability given survived and start position = 1
        - gamma: killing rate
        - blocking time estimate = Pr(BLOCK) * (Expected Survival Time starting at position = 0)
        """

        # Since we must compute this probability CONDITIONED to the event T > t,
        # we first need to merge the measurements of the block times and of the survival times
        # into one single set of measurement time points.
        assert len(self.counts_blocked) == len(self.sbu), \
                "The number of elements in the blocked counts array ({})" \
                " is the same as the number of elements in the blocking time segments array ({})" \
                .format(len(self.counts_blocked), len(self.sbu))

        if self.finalize_type in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
            assert self.counts_alive[-1] == 0, \
                    "The number of particles alive at the last measured time is 0" \
                    " since particles are assumed to have been all absorbed or removed ({})" \
                    .format(self.counts_alive)
            assert self.counts_blocked[-1] == 0, \
                    "The number of particles blocked at the last measured time is 0" \
                    " since particles are assumed to have been all absorbed or removed ({})" \
                    .format(self.counts_blocked)

        self.t, counts_alive, counts_blocked = merge_values_in_time(self.sk, self.counts_alive, self.sbu, self.counts_blocked)

        self.proba_surv_by_t = self.compute_survival_probability_from_counts(counts_alive)
        self.proba_block_by_t = self.compute_probability_from_counts(counts_blocked)
        self.gamma = self.compute_killing_rate()
        self.blocking_time_estimate = self.compute_blocking_time_estimate()
            ## Note: blocking time estimate = Pr(K) * Expected survival time
            ## This is the Approximation 2 proposed by Matt where we use an estimate
            ## of phi(t,K), the probability of blocking given survival to time t, at
            ## just ONE (large) time.

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
        if self.finalize_type in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
            assert self.proba_surv_by_t[-1] == 0.0, "The survival function at the last measured time is 0 ({})".format(self.proba_surv_by_t[-1])
            assert self.proba_block_by_t[-1] == 0.0, "The conditional blocking probability at the last measured time is 0 ({})".format(self.proba_block_by_t[-1])

        return pd.DataFrame.from_items([('t', self.t),
                                        ('P(T>t / s=1)', self.proba_surv_by_t),
                                        ('P(BLOCK / T>t,s=1)', self.proba_block_by_t),
                                        ('Killing Rate', self.gamma),
                                        ('Blocking Time Estimate', self.blocking_time_estimate)])

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

    def estimate_proba_blocking_via_integral(self, expected_survival_time):
        "Computes the blocking probability via Approximation 1 in Matt's draft"
        # Integrate => Multiply the survival, the conditional blocking probabilities, and delta(t) and sum
        integral = 0.0
        for i in range(0, len(self.proba_surv_by_t)-1):
            integral += (self.proba_surv_by_t[i] * self.proba_block_by_t[i]) * (self.t[i+1] - self.t[i])
        if self.LOG:
            print("integral = {:.3f}".format(integral))

        assert expected_survival_time > 0, "The expected survival time is positive ({})".format(expected_survival_time)
        proba_blocking_integral = integral / expected_survival_time

        return proba_blocking_integral, integral

    def estimate_proba_blocking_via_laplacian(self, expected_survival_time):
        "Computes the blocking probability via Approximation 2 in Matt's draft"
        try:
            # Find the index with the last informed (non-NaN) gamma
            # (we exclude the first element as gamma = NaN, since t = 0 at the first element)
            # (note that we don't need to substract one to the result of index() because
            # the exclusion of the first element does this "automatically") 
            idx_tmax = self.gamma[1:].index(np.nan)
        except:
            # The index with the last informed gamma is the last index
            idx_tmax = -1

        proba_blocking_laplacian = self.blocking_time_estimate[idx_tmax] / expected_survival_time
        gamma = self.gamma[idx_tmax]
        proba_block_mean = self.estimate_QSD_at_block()

        if False:
            if self.mean_lifetime is not None:
                print("Building blocks of estimation:")
                print("idx_tmax = {}".format(idx_tmax))
                print("tmax = {:.3f}".format(self.t[idx_tmax]))
                print("gamma = {:.3f}".format(gamma))
                print("Pr(T>t / s=1) = {:.3f}".format(self.proba_surv_by_t[idx_tmax]))
                print("Pr(K / T>t,s=1) = {:.3f}".format(self.proba_block_by_t[idx_tmax]))
                print("Average blocking probability = {:.3f}".format(proba_block_mean))

        return proba_blocking_laplacian, gamma

    def estimate_proba_blocking(self):
        """
        Estimates the blocking probability

        Return: tuple
        The tuple contains the following elements:
        - the estimated blocking probability
        - the estimated expected blocking time given the particle starts at position 1 and is not absorbed
          (this is the numerator of the expression giving the blocking probability estimate)
        - the estimated expected survival time (time to first absorption) given the particle starts at position 0.
          (this is the denominator of the expression giving the blocking probability estimate)
        """
        if not self.is_simulation_finalized:
            raise ValueError("The simulation has not been finalized..." \
                          "\nThe estimation of the blocking probability cannot proceed." \
                          "\nRun first the finalize() method and rerun.")
            
        #-- Compute the building blocks of the blocking probability estimate which are stored in the object
        # This includes:
        # - self.t: time at which an event occurs
        # - self.proba_surv_by_t: survival probability given start position = 1
        # - self.proba_block_by_t: blocking probability given alive and start position = 1
        # - self.gamma: killing rate, eigenvalue of the laplacian
        # - self.blocking_time_estimate: estimated blocking time = Pr(block) * (Expected Survival Time)
        df = self.estimate_proba_survival_and_blocking_conditional()
        if False:
            if self.mean_lifetime is not None:
                plt.plot(df['t'], df['Killing Rate'], 'b.-')
                ax = plt.gca()
                ax.set_ylim((0,2))
                ax.set_xlabel("t")
                ax.set_ylabel("Killing Rate (gamma)")
                plt.title("K={}, particles={}, iterations={}".format(self.queue.getCapacity(), self.N, self.niter))
                plt.show()

        #-- Expected survival time
        if self.mean_lifetime is None:
            expected_survival_time = self.estimate_expected_survival_time_given_position_zero()
        else:
            expected_survival_time = self.mean_lifetime

        #-- Blocking probability estimate via Approximation 1: estimate the integral
        proba_blocking_integral, integral = self.estimate_proba_blocking_via_integral(expected_survival_time)

        #-- Blocking probability estimate via Approximation 2: estimate the Laplacian eigenvalue (gamma) and eigenvector (h(1))
        proba_blocking_laplacian, gamma = self.estimate_proba_blocking_via_laplacian(expected_survival_time)

        return proba_blocking_integral, proba_blocking_laplacian, integral, gamma, expected_survival_time
    #----------------------------- Functions to analyze the simulation ------------------------


    #-------------------------------------- Getters -------------------------------------------
    def get_position(self, P, t):
        """
        Returns the position of the given particle at the given time.
        
        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) whose position is of interest.

        t: float
            Time at which the position of the particle is of interest.

        Return: numpy array
        The position is given by the size of each server that is serving the queue represented by the particle. 
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        assert t >= 0.0, "The queried time is non-negative, implying that there is always a valid position retrieved for the particle (t={})".format(t)
        particle_position = -1 * np.ones(self.nservers, dtype=int)  # initialize the particle position to an invalid value in case its value is not set below (so we can identify an error occurred)
        for server in range(self.nservers):
            # Construct ALL the times at which the analyzed server changed position in its trajectory.
            # These are the times on which the given time t should be searched for in order to
            # retrieve the server's position (i.e. its queue size).
            # We need to do this construction of the times to search on because of the way
            # we are storing the change times in self.trajectories[P]['t'], namely
            # as a list of arrays of length = no. servers.
            server_times = [times[server] for times in self.trajectories[P]['t']]
            server_positions = [positions[server] for positions in self.trajectories[P]['x']]
            idx = bisect.bisect(server_times, t)
            assert idx > 0, "The bisect index where the queried time would be inserted is positive ({})".format(idx)
            particle_position[server] = server_positions[idx-1]
        return particle_position

    def get_active_particle_numbers(self):
        "Returns the list of particle particle numbers P whose current position is > 0"
        return [P for P in range(len(self.is_particle_active)) if self.is_particle_active[P]]

    def get_number_active_particles(self):
        return sum(self.is_particle_active)

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

            if self.finalize_type in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
                assert len(block_times_p) == len(unblock_times_p), \
                        "Particle {}: The number of blocking times ({}) is the same as the number of unblocking times ({})" \
                        .format(p, len(block_times_p), len(unblock_times_p))
            elif self.finalize_type == FinalizeType.ESTIMATE_CENSORED and len(block_times_p) > len(unblock_times_p):
                # This happens when the particle is censored at the block position 
                unblock_times_p += [np.nan]

            for b, u in zip(block_times_p, unblock_times_p):
                assert np.isnan(u) or b < u, \
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
        - a list with the number of survival periods observed for each particles
        - a list with the total time span of the survival periods for each particle
        """
        return self.ktimes0_n, self.ktimes0_sum

    def get_type_last_event(self, P):
        try:
            return self.particles[P].getTypeLastEvent()
        except:
            self.raiseErrorInvalidParticle(P)
            return None

    def get_all_times_last_event(self):
        return [self.get_time_last_event(P) for P in range(self.N)]

    def get_time_last_event(self, P):
        try:
            return self.particles[P].getTimeLastEvent()
        except:
            self.raiseErrorInvalidParticle(P)
            return None

    def get_all_times_next_events(self):
        return self.times_next_events

    def get_time_next_event(self, P):
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
            earliest_event = self._order_times_next_events[P][0]
            time_since_last_event = self.times_next_events[P][earliest_event]
            time_next_event = self.particles[P].getTimeLastEvent() + time_since_last_event
            return time_next_event, time_since_last_event, earliest_event
        except:
            self.raiseErrorInvalidParticle(P)
            return None, None, None

    def get_time_next_event_by_server(self, P):
        """
        Returns the absolute times of the next events for the given particle by server.
        On each server this time is the earliest time over all possible next events the particle
        can undergo in that server.
        
        Return: tuple
        The tuple contains:
        - the absolute time of the next event
        - the relative time of the next event w.r.t. the last event time
        - the index associated to the time of the next event from the list of all possible events for the particle
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)

        # Time elapsed since the last event in each server
        #print(self._order_times_next_events)
        type_index_of_earliest_event_by_server = self._order_times_next_events[P,:,0]   # Length of this array = no. servers
        #print(type_index_of_earliest_event_by_server)
        time_since_last_event_by_server = np.zeros(self.nservers, dtype=float)
        #print(type_next_event_by_server)
        for server in range(self.nservers):
            time_since_last_event_by_server[server] = self.times_next_events[P, server, type_index_of_earliest_event_by_server[server]]

        # Time of the next event in each server
        time_next_event_by_server = self.particles[P].getTimeLastEvent() + time_since_last_event_by_server
        return time_next_event_by_server, time_since_last_event_by_server, type_index_of_earliest_event_by_server

    def get_times_next_events(self, P):
        "Returns the times for all possible next events the particle can undergo" 
        try:
            return self.times_next_events[P]
        except:
            self.raiseErrorInvalidParticle(P)
            return None

    def get_time_last_change_of_system(self):
        return self.time_last_change_of_system

    def get_counts_particles_alive_by_elapsed_time(self):
        return self.counts_alive

    def get_counts_particles_blocked_by_elapsed_time(self):
        return self.counts_blocked

    def plot_trajectories(self):
        "Plots the trajectories of the particles"
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
                      self.queue.getBirthRate(),
                      self.queue.getDeathRate(),
                      self.reactivate, self.finalize_type.name[0:3], self.N, self.niter, self.seed
                      ))
        ax.title.set_fontsize(9)
        plt.show()

    #-------------- Helper functions
    def setup(self):
        params_str = "***********************" \
                    "\nK = {}" \
                    "\nlambda = {:.3f}" \
                    "\nmu = {:.3f}" \
                    "\nrho = {:.3f}" \
                    "\nnparticles = {}" \
                    "\nstart = {}" \
                    "\nmean_lifetime = {}" \
                    "\nreactivate = {}" \
                    "\nfinalize_type = {}" \
                    "\nniter = {}" \
                    "\nseed = {}" \
                    "\n***********************" \
                    .format(self.queue.getCapacity(),
                            self.queue.getBirthRate(), self.queue.getDeathRate(),
                            self.queue.getBirthRate() / self.queue.getDeathRate(),
                            self.N, self.START, self.mean_lifetime,
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
        #for p in range(self.N):
        #    print("Particle {}: times={}".format(p, self.trajectories[p]['t']))
        assert  len(self.trajectories[P]['t']) == 0 and all(self.particles[P].getTimeLastEvent() > 0) or \
                len(self.trajectories[P]['t'])  > 0 and all(self.particles[P].getTimeLastEvent() > self.trajectories[P]['t'][-1]), \
                "The times to insert in the trajectories dictionary ({}) for particle {}" \
                " are larger than the latest inserted times ({})" \
                .format(self.particles[P].getTimeLastEvent(), P, self.trajectories[P]['t'][-1])

    def assertSystemConsistency(self, particle_number=None, time=None):
        is_state_consistent_for_particle = \
            lambda P: all(self.positions[P] == self.particles[P].size) and all(self.positions[P] == self.all_positions[P][-1]) and all(self.particles[P].getTimeLastEvent() == self.all_times[P][-1])

        if time is None:
            time_str = "time last change={:.3f}".format(self.time_last_change_of_system)
        else:
            time_str = "time={:.3f}".format(time)

        consistent_state = True
        if particle_number is None:
            for P in range(self.N):
                consistent_state = is_state_consistent_for_particle(P)              
                if not consistent_state:
                    particle_number = P
                    break
            if not consistent_state:
                self.plot_trajectories()
            assert consistent_state, \
                    "Iter {}, {}: The system is NOT in a consistent state at least for particle {}" \
                    .format(self.iter, time_str, particle_number)           
        else:
            consistent_state = is_state_consistent_for_particle(particle_number)
            if not consistent_state:
                self.plot_trajectories()
            assert consistent_state, \
                    "Iter {}, {}: The system is NOT in a consistent state for particle {}" \
                    "\n\tpositions[{}]={} vs. particles[{}].size={}" \
                    "\n\tpositions[{}]={} vs. all_positions[{}][-1]={}" \
                    "\n\tparticles[{}].TimeLastEvent={} vs. all_times[{}][-1]={}" \
                    .format(self.iter, time_str, particle_number,
                            particle_number, self.positions[particle_number], particle_number, self.particles[particle_number].size,
                            particle_number, self.positions[particle_number], particle_number, self.all_positions[particle_number][-1],
                            particle_number, self.particles[particle_number].getTimeLastEvent(), particle_number, self.all_times[particle_number][-1])         

    def raiseErrorInvalidParticle(self, P):
        raise ValueError("Wrong particle number: {}. Valid values are integers between 0 and {}.\n".format(P, self.N-1))

    def raiseWarningInvalidParticle(self, P):
        raise Warning("Wrong particle number: {}. Valid values are integers between 0 and {}.\n".format(P, self.N-1))
