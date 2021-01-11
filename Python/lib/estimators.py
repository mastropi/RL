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
from .utils.basic import array_of_objects, find, find_last, insort, merge_values_in_time

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
    NONE = 9

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

    nmeantimes: int
        Multiple of the mean time of the job class with LOWEST arrival rate defining the maximum simulation time.
        Ex: If there are two job classes with arrival rates respectively 2 jobs/sec and 3 jobs/sec,
        the maximum simulation time is computed as `nmeantimes * 1/2` sec, which is expected to include
        `nmeantimes` jobs of the lowest arrival rate job on average.
        For instance, if nmeantimes = 10, the simulation time is 5 sec and is expected to include 10 jobs
        of the lowest arrival rate class, and `(nmeantimes * 1/2) / (1/3) = 5*3` = 15 jobs of the
        highest arrival rate class.
        default: 10

    queue: subclass of GenericQueue
        The object should have the following attributes:
        - rates: a list with two elements corresponding to the birth and death events
        at the positions specified by the Event.BIRTH and Event.DEATH values of the Event Enum.

        The object should have the following methods implemented:
        - generate_birth_time()
        - generate_death_time()
        - generate_event_times()
        - getTimesLastEvents()
        - getTypesLastEvents()
        - getTimeLastEvent()
        - getTypeLastEvent()
        - apply_event()
        - apply_events()

    job_rates: list
        List of arrival rate for every possible job class. The job class values are the indices of the given list. 
        Ex: [2, 5]

    policy: list of lists
        List of probabilities of assigning each job class associated to each job rate given in `job_rates`
        to a server in the queue.
        Ex: In a scenario with 2 job classes and 3 servers, the following policy assigns job class 0
        to server 0 or 1 with equal probability and job class 1 to server 1 or 2 with equal probability:
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]

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
    def __init__(self, nparticles: int, queue: GenericQueue, niter=10,
                 nmeantimes=3, job_rates=[2,5], policy=[[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]],
                 mean_lifetime=None, reactivate=True, finalize_type=FinalizeType.REMOVE_CENSORED,
                 seed=None, plotFlag=False, log=False):
        if reactivate and nparticles < 2:
            raise ValueError("The number of particles must be at least 2 when reactivate=True ({})".format(nparticles))
            import sys
            sys.exit(-1)
        self.N = nparticles
        self.niter = niter
        self.queue = queue              # This variable is used to create the particles as replications of the given queue
        self.policy = policy            # This should be a mapping from job class to server.
                                        # As a simple example to begin with, here it is defined as a vector of probabilities for each job class
                                        # where the probability is the assignment of the job class to each server, where the number of servers is assumed to be 3. 
        self.job_rates = job_rates      # This should be a list of arrival rates for each job class
        self.maxtime = nmeantimes * np.max( 1/np.array(self.job_rates) ) # The max simulation time is computed as a multiple (given by the user) of the mean arrival time of the lowest frequency job
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
        # Keep track of the iterations run for each particle (just for information purposes)
        # Note: an iteration is defined by the application of an event to ONE server
        # (typically the server with the smallest observed event time)
        self.iterations = np.zeros(self.N, dtype=int)

        #---------------------------------- Trajectory information ----------------------------
        # Attributes used to store information of the server trajectories and of special states of the system's buffer

        #-- Trajectories
        # Trajectories of each particle (always N in total)
        # The information stored is at the server level, i.e. the trajectory for EACH particle's server is stored.
        self.trajectories = array_of_objects((self.N, self.nservers), dtype=dict, value=dict({'t': [], 'x': [], 'e': []}))
        # Information about the particles used to compute the statistics needed to estimate the blocking probability
        # The information stored is about the buffer size, NOT about the servers.
        # For instance, a particle is considered BLOCKED when the BUFFER size reaches its capacity,
        # that is, when the SUM of the server queue sizes is equal to the buffer capacity.
        # The size of this list is normally MUCH LARGER than N (the number of particles in the system)
        # because every reactivation adds a new element to this list, since a reactivation is associated to a "new" particle
        # which is identified by the particle ID (p) (as opposed to the particle number (P)). 
        self.info_particles = []
        assert np.sum(self.START) in [0, 1], "The start position for the particle (buffer size) is either 0 or 1 ({})".format(self.START)
        if np.sum(self.START) == 0:
            start_event_type = EventType.ABSORPTION
        elif np.sum(self.START) == 1:
            start_event_type = EventType.ACTIVATION
        for P in range(self.N):
            # 1)-- Trajectories (one value per particle and per server)
            for server in range(self.nservers):
                self.trajectories[P][server]['t'] = [0.0]                   # event times at which the particle changes position in each server
                self.trajectories[P][server]['x'] = [ self.START[server] ]  # positions taken by the particle after each event
                self.trajectories[P][server]['e'] = [ Event.RESET ]         # events (birth, death, reset) associated to the event times of each server

            # 2)-- Particle's info (one value per particle)
            self.info_particles += [ dict({
                                           # We initialize 't' and 'E' to the situation at which each particle starts-off (either ABSORBED (when no reactivation is performed) or with an ACTIVATION (when reactivation is performed))
                                           # This is necessary to guarantee the correct functioning of assertions, such as the one that asserts that before an ABSORPTION always comes an ACTIVATION.
                                           't': [0.0],                  # times at which the events of interest take place (e.g. ACTIVATION, ABSORPTION, BLOCKING)
                                           'E': [start_event_type],     # events of interest associated to the times 't'
                                                                        # (the events of interest are those affecting the
                                                                        # estimation of the blocking probability:
                                                                        # activation, absorption, block, unblock)
                                           't0': None,                  # Absorption /reactivation time
                                           'x': None,                   # Position after reactivation
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

        #-- Reactivation IDs
        # List of reactivation IDs indexing the info_particles list
        # where we should look for the information (relevant events and their times)
        # about each particle number.
        # These IDs start each being in the range [0, N-1], but as the simulation progresses
        # and reactivation occurs, start increasing beyond N-1, because each reactivation
        # implies a new ID. So the first reactivated particle will have ID = N, the second reactivate
        # particle will have ID = N+1, and so forth.
        # *** These IDs are represented with lower letters in the code (e.g. `p`, `q`) ***
        self.particle_reactivation_ids = list(range(self.N))    # This is [0, 1, ..., N-1]

        #-- Positions and times used for plotting
        # Positions (size) and event times of each particle's server and of the system's buffer for each particle
        # The event times by server correspond to the times of the latest event applied to each server
        # and the buffer's event times correspond to the minimum of the event times among all servers.
        self.positions_by_server = np.zeros((self.N, self.nservers), dtype=int)
        self.times_by_server = np.nan*np.ones((self.N, self.nservers), dtype=float)
        self.positions_buffer = np.zeros((self.N,), dtype=int)
        self.times_buffer = np.nan*np.ones((self.N,), dtype=float)
        for P in range(self.N):
            self.positions_by_server[P] = self.START
            self.positions_buffer[P] = np.sum( self.positions_by_server[P] )

        # History of all positions and event times for each particle
        # We create two pair of structures:
        # - one that keeps track of the positions and times of the system's SINGLE buffer
        # - one that keeps track of the positions and times of each particle's server
        # In each of these structures, the entry is a list that may have different length
        # depending on the particle and server within each particle.
        self.all_positions_buffer = array_of_objects((self.N,), dtype=list, value=[ np.sum(self.START) ])
        self.all_times_buffer = array_of_objects((self.N,), dtype=list, value=[0.0])
        # NOTE: The difference between the all_times_by_server array and the trajectories array
        # is that the trajectories array will NEVER have a repeated time value and therefore can be used
        # to compute statistics necessary for the estimations of the blocking probability.
        # On the other hand, the all_times_by_server array has repeated times when the particle is absorbed
        # and reactivated, as the absorption time and reactivation time coincide and are both stored in the attribute
        # (for information purposes, in case we need to analyze something) 
        self.all_positions_by_server = np.empty((self.N, self.nservers), dtype=list)
        self.all_times_by_server = array_of_objects((self.N, self.nservers), dtype=list, value=[0.0])
        # Set the initial positions of the servers in each particle to the START position given by the user
        for P in range(self.N):
            for server in range(self.nservers):
                self.all_positions_by_server[P, server] = [ self.START[server] ]
        #---------------------------------- Trajectory information ----------------------------


        #---------------------------------- Next events information ---------------------------
        # Attributes used to store information about the (possible) next events to be applied

        #-- Types and times of next events (one event for each server)
        # All event times are ABSOLUTE, i.e. the exponential time is generated for each server
        # (which is a RELATIVE time --relative to the previous event) and this relative time is converted to
        # absolute by adding the time of the previous event. Working with absolute times facilitates the process.
        self.types_next_events = np.empty((self.N, self.nservers), dtype=Event)
        self.times_next_events = np.nan*np.ones((self.N, self.nservers), dtype=float)
        self.times_next_events2 = np.zeros((self.N, self.nservers, 2), dtype=float) # `2` is the number of possible events: birth and death
        self.times_next_birth_events = np.zeros((self.N, self.nservers), dtype=float)
        self.times_next_death_events = np.zeros((self.N, self.nservers), dtype=float)

        #-- Arriving jobs information
        # Times and job classes associated to the next events in the queue of each server
        # It is important to store this information in order to properly handle the situation when
        # different arriving jobs are assigned to the SAME server in a given iteration
        # (see method _generate_birth_times())
        self.times_next_events_in_queue = array_of_objects((self.N, self.nservers), dtype=list, value=[])
        self.jobclasses_next_events_in_queue = array_of_objects((self.N, self.nservers), dtype=list, value=[])

        #-- Serviced jobs information
        # Service times in each server (accumulated in a "queue" as they are generated)
        # They represent all the ABSOLUTE times at which a job is served (i.e. a DEATH occurs) in a server 
        # It is important to store this information in order to properly handle the situation when
        # new DEATH events are generated for a server, but the previously generated DEATH event
        # was not yet "consumed" because we don't yet have the information of a positive BIRTH event
        # for a server against which we can compare the death event and decide the type of the next event
        # to apply for the server. 
        # (see method _generate_death_times())
        self.times_next_services = array_of_objects((self.N, self.nservers), dtype=list, value=[])
            ## The above is an array of size N x #servers x 2, where 2 is the number of events: birth and death
            ## We initialize it as a zero-array because we consider that the very first event in each server and particle
            ## takes place at time 0.0.
        #---------------------------------- Next events information ---------------------------


        #--------------------------- Last jobs and services information -----------------------
        # Times of the last jobs arrived for each class and of the last jobs served by each server
        # These are needed in order to compute the ABSOLUTE time of the next arriving job of each class
        # and of the next served job for each server, since the arrival and service times are generated
        # RELATIVE to the time of the latest respective event. 
        self.times_last_jobs = np.zeros((self.N, len(self.job_rates)))
        self.times_last_services = np.zeros((self.N, self.nservers))
        #--------------------------- Last jobs and services information -----------------------


        #------------------------- Absorption times and active particles ----------------------
        #-- Information about the absorption times, which is used to complete the simulation after each absorption
        # This dictionary initially (i.e. before any re-activation of a particle) contains the
        # FIRST absorption times of each particle.
        # Once the reactivation process starts, the dictionary contains every new absorption time observed.
        # For more information see the generate_trajectories_until_end_of_simulation() method or similar.
        self.dict_info_absorption_times = dict({'t': [],        # List of absorption times (they should be stored SORTED, so that reactivation starts with the smallest absorption time)
                                                'P': [],        # List of particle numbers associated to each absorption time in 't'
                                                })

        #-- Array storing whether the particle is active or not
        # (used at the end of the simulation by method finalize() or similar)
        if start_event_type == EventType.ABSORPTION:
            self.is_particle_active = [False]*self.N
        else:
            self.is_particle_active = [True]*self.N
        ## Note: contrary to what I suspected, the elements of this array do NOT share the same memory address
        ## (verified by an example)  
        #------------------------- Absorption times and active particles ----------------------


        #----------------------- Information for survival time calculation --------------------
        # Arrays that are used to estimate the expected survival time (time to killing from position 0)
        if start_event_type == EventType.ABSORPTION:
            # Latest times the particles changed to position 0 is 0.0 as they start at position 0
            self.times0 = np.zeros(self.N, dtype=float)
        else:
            # Latest times the particles changed to position 0 is unknown as they don't start at position 0
            self.times0 = np.nan * np.ones(self.N, dtype=float)
        self.ktimes0_sum = np.zeros(self.N, dtype=float)   # Times to absorption from latest time it changed to position 0
        self.ktimes0_n = np.zeros(self.N, dtype=int)       # Number of times the particles were absorbed
        #----------------------- Information for survival time calculation --------------------


        #----------------------- Information about blocking and unblocking --------------------
        # Attributes used directly in the computation of the estimated measures that contribute to the probability of blocking.
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
        #----------------------- Information about blocking and unblocking --------------------


        # Time of last change of the whole system of particles
        # TODO: Update this value as the simulation goes forward
        self.time_last_change_of_system = 0.0

        if self.LOG:
            print("Particle system with {} particles has been reset:".format(self.N))
            print("")

    #--------------------------------- Functions to simulate ----------------------------------
    def simulate(self):
        """
        Simulates the system of particles and estimates the blocking probability
        via Approximation 1 and Approximation 2 of Matthieu Jonckheere's draft on queue blocking dated Apr-2020.

        Return: tuple
        The tuple contains the following elements:
        - the estimated blocking probability
        - the estimated expected blocking time given the particle starts at position 1 and is not absorbed
          (this is the numerator of the expression giving the blocking probability estimate)
        - the integral appearing at the numerator of Approximation 1
        - the estimated gamma parameter: the absorption rate
        - the estimated expected survival time (time to first absorption) given the particle starts at position 0.
          (this is the denominator of the expression giving the blocking probability estimate)
        """
        self.reset()

        self.generate_trajectories_at_startup()
        # Check trajectories
        if False:
            for P in range(self.N):
                self.plot_trajectories_by_server(P)
            self.plot_trajectories_by_particle()

        self.generate_trajectories_until_end_of_simulation()
        # Check trajectories
        if False:
            if False:
                for P in range(self.N):
                    self.plot_trajectories_by_server(P)
            self.plot_trajectories_by_particle()

            if False:
                print("\n**** CHECK status of attributes *****")
                print("Info particles:")
                for p in range(len(self.info_particles)):
                    print("Particle ID {}:".format(p))
                    print(self.info_particles[p])
                input("Press ENTER...")

        self.finalize()
        self.compute_counts()

        proba_blocking_integral, proba_blocking_laplacian, integral, gamma, expected_survival_time = \
            self.estimate_proba_blocking()
        self.proba_blocking_integral = proba_blocking_integral
        self.proba_blocking_laplacian = proba_blocking_laplacian

        if self.plotFlag:
            self.plot_trajectories_by_particle()

        return self.proba_blocking_integral, self.proba_blocking_laplacian, integral, gamma, expected_survival_time

    def generate_trajectories_at_startup(self):
        for P in range(self.N):
            self.iterations[P] = 0
            # Perform a first iteration where the next BIRTH time is generated for ALL job classes
            # and the next DEATH time is generated for all servers.
            # This is needed to get the simulation started because such process cannot start
            # until at least one of the latest event times in the servers is different from the rest.
            # (Recall that at the very beginning all event times for all servers are equal to 0,
            # which prevents the process from retrieving the information of the most recent event,
            # needed to know which server should be updated next
            # --see function _generate_trajectory_until_absorption_or_end_of_simulation() or similar)
            end_of_simulation = self.generate_one_iteration(P)
            
            # Continue iterating until absorption or until the max simulation time is reached
            self._generate_trajectory_until_absorption_or_end_of_simulation(P, end_of_simulation)            

        if self.LOG: #True:
            print("\n******** Info PARTICLES before the REACTIVATION process starts:")
            for P in range(self.N):
                with printoptions(precision=3, suppress=True):
                    print("Particle {}:".format(P))
                    print(np.c_[np.array(self.info_particles[P]['t']), np.array(self.info_particles[P]['E'])])

    def generate_one_iteration(self, P):
        """
        Generates one iteration of event times for a particle.
        
        This implies:
        - A set of event times (one per server) is generated for the particle.
        - The event with the smallest event time is applied to the particle.

        Return: bool
        Whether the end of the simulation has been reached.
        """
        end_of_simulation = lambda t: t >= self.maxtime 

        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)

        #-- Generate the next events for each server
        if self.LOG:
            print("\n****** P={}: GENERATE next events (#iterations={}) ******".format(P, self.iterations[P]))
        self._generate_next_events(P)

        #-- Apply the event to the server with the smallest event time
        # Only apply the event if:
        # - the next event time is defined (which is NOT the case when any of the next event times is NaN)
        # Otherwise, it means we still need to know the next event times of SOME servers
        # as knowing the next event times of ALL servers is needed to know which event to apply next.
        # - the event time has NOT overcome the maximum simulation time, in which case we need to keep
        # the servers in the state of the latest observed event, which should be extended as the state
        # of the server at the maximum simulation time.
        server_next_event, type_next_event, time_next_event = self._get_next_event(P)
        if not np.isnan(time_next_event):
            if end_of_simulation(time_next_event):
                return True
            else:
                if self.LOG:
                    print("\n++++++ P={}: APPLY an event ++++++".format(P))
                self._apply_next_event(P, server_next_event, type_next_event, time_next_event)

        return False

    def _generate_next_events(self, P):
        """
        Generates the next event for each server, their type (birth/death) and their time
        storing their values in the internal attributes containing these pieces of information.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        self.assertTimeMostRecentEventAndTimesNextEvents(P)

        # Generate OR retrieve the next set of birth and death times by server
        # Note that:
        # - we either "generate" or "retrieve" the new birth/death times depending on whether
        # the queue of next events for the server and event type is empty or not, respectively.
        # In fact, if the queue is NOT empty, we just pick the first event time in the queue.
        # - whatever the case, whether we *generate* or *retrieve* the "next event time", we still generate
        # NEW event values for ALL servers for both birth/death types of event, which simply go to the
        # queue of the respective server. For BIRTH events, the queue of the server *assigned* to the generated
        # job class is increased; for DEATH events, the queue of each server is increased.
        self._generate_birth_times(P)
        self._generate_death_times(P)
        
        # Find the next event times based on the current next birth and death times on all servers
        self._find_next_event_times(P)

    def _generate_birth_times(self, P):
        """
        Generates new arrival times for each job class, based on the job arrival rates,
        and assigns each job to one of the given particle's servers based on the assignment policy.
        
        The policy is assumed to be a mapping from job class to server number.

        The effect of this method is the following:
        - The queue of jobs in each server is updated with the new assigned job (storing job class and event time).
        - The times of the next birth events by server is updated when the generated birth event is the first job
        assigned to the server's queue. Otherwise, the time of the next birth event in the sever is kept.
        """
        # Generate a birth time for each job CLASS and store them as ABSOLUTE values
        birth_times_relative_for_jobs = np.random.exponential( 1/np.array(self.job_rates) )
        time_last_job = copy.deepcopy(self.times_last_jobs[P])
        self.times_last_jobs[P] += birth_times_relative_for_jobs 

        # Assign the job class associated to each birth time to one of the servers based on the policy
        job_classes = range( len(self.job_rates) )
        servers = range(self.nservers)
        for job_class in job_classes:
            # IMPORTANT: The assigned server can be repeated for different job classes
            # => we might have a queue of jobs for a particular server (which is handled here)
            assigned_server = np.random.choice(servers, p=self.policy[job_class])
            assert assigned_server < self.nservers, \
                    "The assigned server ({}) is one of the possible servers [0, {}]".format(assigned_server, self.nservers-1)

            # Insert the new job time in order into the server's queue containing job birth times
            # Note that the order of the jobs is NOT guaranteed because the server's queue may contain
            # jobs of different classes, whose arrival times are generated independently from each other.
            idx_insort, _ = insort(self.times_next_events_in_queue[P][assigned_server], self.times_last_jobs[P][job_class])
            self.jobclasses_next_events_in_queue[P][assigned_server].insert(idx_insort, job_class)
            if self.LOG:
                print("job class: {}".format(job_class))
                print("job time (PREVIOUS): {}".format(time_last_job[job_class]))
                print("job time (RELATIVE): {}".format(birth_times_relative_for_jobs[job_class]))
                print("job time (ABSOLUTE): {}".format(self.times_last_jobs[P][job_class]))
                print("assigned server: {}".format(assigned_server))
                print("queue of assigned server: {}".format(self.times_next_events_in_queue[P][assigned_server]))

        if self.LOG:
            print("currently assigned job classes by server: {}".format(self.jobclasses_next_events_in_queue[P]))        
            print("currently assigned job times by server: {}".format(self.times_next_events_in_queue[P]))        

        # Assign the birth times for each server by picking the first job in each server's queue 
        for server in range(self.nservers):
            if len(self.times_next_events_in_queue[P][server]) > 0:
                # Only update the time of the next birth event if there is a job in the server's queue 
                self.times_next_birth_events[P][server] = self.times_next_events_in_queue[P][server][0]
            else:
                # No job is in the queue of the server => there is no next birth event yet...
                self.times_next_birth_events[P][server] = np.nan

    def _generate_death_times(self, P):
        """
        Generates and stores a set of death times for each server in the system

        The effect of this method is the following:
        - The list of job service times in each server is updated with the new death time generated for the server.
        - The times of the next service event (death event) by server is updated when the generated death event
        is the first service time assigned to the server. Otherwise, the time of the next death event in the sever is kept. 
        """
        # Generate a death time for each server and store them as ABSOLUTE values
        death_times_relative = self.particles[P].generate_event_times(Event.DEATH)
        self.times_last_services[P] += death_times_relative 

        for server in range(self.nservers):
            # Add the death time to the list of service times already generated for each server
            # (which by construction are sorted increasingly)
            self.times_next_services[P][server] += [ self.times_last_services[P][server] ]

            # Update the time of the next DEATH event
            # (which DOESN'T change from the value already stored there
            # if the list of death times was not empty before including
            # the death time just generated for the server)
            self.times_next_death_events[P][server] = self.times_next_services[P][server][0]

    def _find_next_event_times(self, P):
        "Finds the next event times for the given particle"
        # Store the next birth and death times as columns by server for their comparison
        # so that we can easily choose which type of event comes next by server.   
        self.times_next_events2[P] = np.c_[self.times_next_birth_events[P], self.times_next_death_events[P]]
        ## The above way of storing the times of the next BIRTH and DEATH events assumes
        ## that the BIRTH event is associated to the (index) value 0 and the DEATH event is associated to the index value 1.

        # Find the next event characteristics (type and time) as the one occurring first
        # among the next birth and death events for EACH server
        idx_min_times = np.argmin(self.times_next_events2[P], axis=-1)
        servers = range(self.nservers)
        self.times_next_events[P] = [ self.times_next_events2[P][server, idx_min] for server, idx_min in zip(servers, idx_min_times) ]
        self.types_next_events[P] = [ Event.BIRTH if idx_min == 0 else Event.DEATH for idx_min in idx_min_times ]

        if self.LOG:
            print("next birth times: {}".format(self.times_next_birth_events[P]))
            print("next death times: {}".format(self.times_next_death_events[P]))
            print("TYPES NEXT EVENTS for P={}: {}".format(P, self.types_next_events[P]))
            print("TIMES NEXT EVENTS for P={}: {}".format(P, self.times_next_events[P]))

    def _get_next_event(self, P):
        """
        Computes the information necessary to apply the next event to a particle

        The next event to apply to a particle is the one that has the smallest time among the next events
        already computed for the particle.

        Return: tuple
        The tuple contains:
        - the server number on which the next event takes place.
        - the type of the next event.
        - the time at which the next event takes place.
        """
        server_with_smallest_time = np.argmin( self.times_next_events[P] )
            ## NOTE: if any of the above "next event times" is NaN the minimum is NaN
        type_of_event = self.types_next_events[P][server_with_smallest_time]
        time_of_event = self.times_next_events[P][server_with_smallest_time]

        return server_with_smallest_time, type_of_event, time_of_event

    def _apply_next_event(self, P, server, type_of_event, time_of_event):
        """
        Applies the next event to a particle at the given server, of the given type, at the given time.

        Applying the event means the following:
        - The state of the queue represented by the particle is changed based on the applied event.
        - The list of events stored in this object is updated by removing the applied event, which means:
            - the job that is waiting first in the queue of jobs to be served is removed if it's a BIRTH event.
            - the job being served by the server is removed if it's a DEATH event.
        """
        self.particles[P].apply_event( type_of_event,
                                       time_of_event,
                                       server )
        if self.LOG:    # True
            print("\nP={}: Applied {} @t={:.3f} to server {}...".format(P, type_of_event, time_of_event, server) + (self.particles[P].getLastChange(server) == 0 and " --> NO CHANGE!" or ""))
        if type_of_event == Event.BIRTH:
            # Remove the job from the server's queue, as it is now being PROCESSED by the server
            if self.LOG:
                print("Job queue in server {} (the first job will be removed for processing by server): {}".format(server, self.times_next_events_in_queue[P][server]))
            self.times_next_events_in_queue[P][server].pop(0)
            self.jobclasses_next_events_in_queue[P][server].pop(0)
        elif type_of_event == Event.DEATH:
            # Remove the job from the server processing line, as its process has now FINISHED
            if self.LOG:
                print("Job service times list {} (the first job will be removed as server finished processing it): {}".format(server, self.times_next_services[P][server]))
            self.times_next_services[P][server].pop(0)

        # Increase the iteration because the iteration is associated to the application of a new event in ONE server
        self.iterations[P] += 1

        # Update the position information of the particle so that we can plot the trajectories
        # and we can compute the statistics for the estimations that are baseds on the occurrence
        # of special events like activation, blocking, etc. 
        self._update_particle_position_info(P, server=server)

        # Update the ACTIVE condition of the particle
        self._update_active_particles(P)

    def _update_particle_position_info(self, P, server=0):
        """
        The position of the given particle's server (queue size) is updated. This includes:
        - the current position
        - the current position of the system's buffer (i.e. the buffer size)
        - the trajectories attribute
        - the info_particles attribute that is used to compute the statistics necessary for the FV estimation
        """
        #-- Current position info (by server and for the system's buffer)
        # By server
        self.positions_by_server[P][server] = self.particles[P].getServerSize(server)
        self.times_by_server[P][server] = self.particles[P].getTimeLastEvent(server)

        # For the system's buffer
        assert np.sum(self.positions_by_server[P]) == self.particles[P].getBufferSize(), \
            "The sum of the positions by server ({}) coincides with the buffer size ({})" \
            .format(self.positions_by_server[P], self.particles[P].getBufferSize())
        buffer_new_size = self.particles[P].getBufferSize()
        buffer_old_size = self.positions_buffer[P]
        buffer_size_change = buffer_new_size - buffer_old_size
        self.positions_buffer[P] = buffer_new_size
        assert self.iterations[P] > 0, "_update_particle_position_info(): There has been at least ONE iteration in the particle P={} being updated".format(P)
        self.times_buffer[P] = self.times_by_server[P][server]
            ## NOTE: The buffer time is the time of occurrence of the event in the given server
            ## because it is ASSUMED that the server where the event occurs is the server
            ## for which the next event time is the smallest (meaning that all other servers
            ## do NOT change their size before this time => we know that their sizes are constant
            ## from the latest time their sizes change until the occurrence of the event
            ## we are now handling. Note that this is ALWAYS the case when we reach this function
            ## because this function is only called once the first iteration has taken place on the particle)

        #-- Historical info needed to plot trajectories
        # By server
        self.all_positions_by_server[P][server] += [ self.positions_by_server[P][server] ]
        self.all_times_by_server[P][server] += [ self.times_by_server[P][server] ]

        # For the system's buffer
        self.all_positions_buffer[P] += [ self.positions_buffer[P] ]    # Add the current system's buffer size
        self.all_times_buffer[P] += [ self.times_buffer[P] ]            # Add the latest time the system's buffer changed (i.e. the latest time for which we have information about the size of ALL servers)

        self._update_trajectories(P, server)

        #-- Update the historical info of special events so that we can compute the statistics needed for the estimations
        self._update_info_particles(P, buffer_size_change) # The second parameter is the change in BUFFER size for the particle, NOT the array of changes in server queue sizes

    def _update_trajectories(self, P, server=0):
        """
        Updates the trajectory information of the particle for the given server, meaning:
        - the event time ('t')
        - the event type ('e')
        - the queue size of the server ('x')
        based on the latest event that took place for the particle.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        self.assertTimeToInsertInTrajectories(P, server)
        self.trajectories[P][server]['t'] += [ self.particles[P].getTimeLastEvent(server) ]
        self.trajectories[P][server]['e'] += [ self.particles[P].getTypeLastEvent(server) ]
        self.trajectories[P][server]['x'] += [ self.particles[P].getServerSize(server)    ]

    def generate_trajectories_until_end_of_simulation(self):
        "Generates the trajectories until the end of the simulation time is reached"
        while len(self.dict_info_absorption_times['t']) > 0:
            assert self.dict_info_absorption_times['t'] == sorted(self.dict_info_absorption_times['t']), \
                "The absorption times are sorted: {}".format(self.dict_info_absorption_times['t'])
            if self.LOG:
                with printoptions(precision=3, suppress=True):
                    print("\n******** REST OF SIMULATION STARTS (#absorption times left: {} **********".format(len(self.dict_info_absorption_times['t'])))
                    print("Absorption times and particle numbers (from which the first element is selected):")
                    print(self.dict_info_absorption_times['t'])
                    print(self.dict_info_absorption_times['P'])
            # Remove the first element of the dictionary which contains the absorption times and respective particle numbers
            t0 = self.dict_info_absorption_times['t'].pop(0)
            P = self.dict_info_absorption_times['P'].pop(0)
            if self.LOG:
                print("Popped absorption time {:.3f} for particle P={}".format(t0, P))

            assert t0 < self.maxtime, \
                "The currently processed absorption time ({:.3f}) is smaller than the max simulation time ({:.3f})" \
                .format(t0, self.maxtime)

            end_of_simulation = False
            if self.reactivate:
                # Choose a particle to which the absorbed particle is reactivated
                Q = self._choose_assigned_particle(P)
                if self.LOG:
                    print("Assigned particle number: Q={}".format(Q))
                position_Q_at_t0, position_change = self._create_new_particle(P, Q, t0)
    
                # IMPORTANT: The update of the INFO particles should come AFTER the update of the particle's position
                # because the INFO particles stores information about special events of the particle (activation, absorption, etc.)
                # and these special events are checked based on the information stored in the self.particles[P] attribute
                # which is updated by the _set_new_particle_position() call. 
                self._set_new_particle_position(P, t0, position_Q_at_t0)
                self._update_info_particles(P, np.sum(position_change)) # The second parameter is the change in BUFFER size for the particle, NOT the array of changes in server queue sizes
            else:
                # In NO reactivation mode, we need to generate ONE iteration before CONTINUING with the simulation below,
                # otherwise the assertion that the next absorption time is different from the previous one
                # in _update_survival_time_from_zero() the next time it is called, fails.
                end_of_simulation = self.generate_one_iteration(P)

            # Continue generating the trajectory of the particle after reactivation
            # until a new absorption occurs or until the max simulation time is reached
            self._generate_trajectory_until_absorption_or_end_of_simulation(P, end_of_simulation)

    def _choose_assigned_particle(self, P):
        "Chooses a particle among all other particles in the system to be assigned to the given (absorbed) particle"
        # We choose a particle excluding the particle that needs to be reactivated (P)
        list_of_particles_to_choose_from = list(range(P)) + list(range(P+1, self.N))
        chosen_particle_number = list_of_particles_to_choose_from[ np.random.randint(0, self.N-1) ]

        assert self.isValidParticle(chosen_particle_number) and chosen_particle_number != P, \
                "The chosen particle is valid (0 <= Q < {} and different from {}) ({})" \
                .format(self.N, P, chosen_particle_number)
        return chosen_particle_number

    def _create_new_particle(self, P, Q, t):
        """
        Creates a new particle in the system at the given time t,
        whose position starts at the position of particle Q assigned to the absorbed particle P
        after reactivation.
        The information of the new particle is stored as a new entry in the info_particles list
        representing the ID of a new reactivated particle from absorbed particle P.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) corresponding to the absorbed particle.

        Q: non-negative int
            Particle number (index of the list of particle queues) corresponding to the particle
            assigned to the absorbed particle to be reactivated.

        t: positive float
            Absorption time at which reactivation of particle P takes place.

        Return: tuple
        The tuple contains:
        - numpy array with the position of particle Q at the time of reactivation t, which is
        the new position that will be assumed by particle P after reactivation.
        - numpy array with the change in position to be experimented by the absorbed particle P
        after reactivation to particle Q.
        Note that the change in position COULD be different to the position of particle Q as
        it depends on what we define as ABSORPTION. If an absorption event is when the particle's position
        becomes 0 then they both coincide, but we may be extending the set of absorbed states in the future.
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
        # Its position is assigned to the position of the reassigned particle Q at time t,
        # and the information about the event times ('t') and event types ('E') are empty
        # (because nothing has happened yet to the reactivated particle.
        position_Q = self.get_position(Q, t)
        self.info_particles += [ dict({'t': [],
                                       'E': [],
                                       't0': t,
                                       'x': position_Q,
                                       'particle number': P,
                                       'reactivated number': Q,
                                       'reactivated ID': q}) ]

        # Update the particle ID associated to the particle number P
        # to the particle ID just added to the info_particles list.
        # This ID is the last index added to the info_particles list
        # and represents the new particle ID associated to the particle number P
        # since its associated particle ID prior to reactivation (i.e. prior to the copy done here)
        # is no longer active as that particle ID is associated to an ABSORBED particle.
        self.particle_reactivation_ids[P] = len(self.info_particles) - 1

        if self.LOG: #True:
            with printoptions(precision=3, suppress=True):
                print("\nInfo PARTICLES for particle ID={}, after particle P={} reactivated to particle Q={} (ID q={}) at time t={}:".format(q, P, Q, q, t))
                print("\ttimes: {}".format(self.info_particles[q]['t']))
                print("\tevents: {}".format(self.info_particles[q]['E']))
                #print(np.c_[self.info_particles[q]['t'], self.info_particles[q]['E']])

        return position_Q, position_Q - self.particles[P].getServerSizes()

    def _set_new_particle_position(self, P, t, position):
        """
        Sets the position of the particle at the given time to the given position

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) corresponding to the reactivated particle.

        position: array of non-negative int
            Position to assign to the particle, which is given by the size of the queue in each server.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        assert 0 <= np.sum(position) <= self.particles[P].getCapacity(), \
                "The start positions of the reactivated particle {} ({}) do not exceed the capacity of the buffer ({})" \
                .format(P, position, self.particles[P].getCapacity())
        assert t < self.maxtime, \
                "The time at which the position is updated ({:.3f}) is smaller than the maximum simulation time ({:.3f})" \
                .format(t, self.maxtime)

        # Update the situation of the queue represented by the particle
        self.particles[P].resize(position, t)

        # Reactivation resets the iterations run on the reactivated particle
        # because it's like starting again. In fact, all the server times now
        # are ALIGNED to the SAME reactivation time, and this is like the very beginning
        # of the simulation, where all server times were at 0.
        # This is important, because the number of iterations run on the particle decides
        # whether the _generate_next_events() method (when iterations = 0) or
        # #the _update_next_events_latest_server() method (when iterations > 0) is called.
        self.iterations[P] = 0

        #-- Update current and historical information
        # Information by server
        self.times_by_server[P] = t             # NOTE that this time is the same for ALL servers, a situation that, under normal simulation without reactivation, only happens at the beginning of the simulation (time=0)
        self.positions_by_server[P] = position
        # UPDATE the trajectories information
        # (so that we can use the self.trajectories attribute to compute statistics
        # as it is guaranteed that the event times are NOT repeated, which is not the case with the all_times_by_server list 
        for server in range(self.nservers):
            self.trajectories[P][server]['t'][-1] = t
            self.trajectories[P][server]['x'][-1] = position[server]
            self.trajectories[P][server]['e'][-1] = Event.RESIZE

        # Information of the system's buffer
        self.times_buffer[P] = t
        self.positions_buffer[P] = np.sum(position)

        for server in range(self.nservers):
            self.all_times_by_server[P][server] += [t]
            self.all_positions_by_server[P][server] += [ position[server] ]
        self.all_times_buffer[P] += [t]
        self.all_positions_buffer[P] += [ self.positions_buffer[P] ]

        # Update the list stating whether each particle is active
        self._update_active_particles(P)

    def _update_info_particles(self, P, last_position_change):
        """
        Updates the info_particles attribute with the new particle's position
        and any special state that the particle goes into as a result of the last position change,
        such as ACTIVATION, ABSORPTION, BLOCK, UNBLOCK.
        """
        assert self.isValidParticle(P), "The particle to update is valid (0<=P<{}) ({})".format(self.N, P)

        #--------------- Helper functions -------------------
        # Possible status change of a particle 
        is_activated = lambda P: self._has_particle_become_activated(P, last_position_change)
        is_absorbed = lambda P: self._has_particle_become_absorbed(P, last_position_change)
        is_blocked = lambda P: self._has_particle_become_blocked(P, last_position_change)
        is_unblocked = lambda p: self._has_particle_become_unblocked(P, last_position_change) 

        # Type of event (relevant for the blocking probability estimation)
        type_of_event = lambda P: is_activated(P) and EventType.ACTIVATION or is_absorbed(P) and EventType.ABSORPTION or is_blocked(P) and EventType.BLOCK or is_unblocked(P) and EventType.UNBLOCK or None
        #--------------- Helper functions -------------------

        if type_of_event(P) is not None:
            if self.LOG:
                print("P={}: SPECIAL EVENT OBSERVED: {}".format(P, type_of_event(P)))
                print("\tposition of P (buffer size): {}".format(self.particles[P].getBufferSize()))
                print("\tlast change in buffer size: {}".format(last_position_change))
            p = self.particle_reactivation_ids[P]
            assert p < len(self.info_particles), \
                    "The particle ID (p={}) exists in the info_particles list (0 <= p < {})" \
                    .format(p, len(self.info_particles))
            self.info_particles[p]['t'] += [ self.times_buffer[P] ]
            self.info_particles[p]['E'] += [ type_of_event(P) ]

    def _generate_trajectory_until_absorption_or_end_of_simulation(self, P, end_of_simulation):
        """
        Generates the trajectory for the given particle until it is absorbed or the max simulation time is reached.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to simulate.

        end_of_simulation: int
            Whether the end of the simulation has been reached.
        """
        absorbed = lambda P: self._check_particle_absorbed_based_on_most_recent_change(P)

        while not end_of_simulation and not absorbed(P):
            end_of_simulation = self.generate_one_iteration(P)
        # Note: we should FIRST check for the end of the simulation because if that's the case
        # and the particle was already absorbed we should NOT add the particle's absorption time
        # again to the dictionary of absorption times (if we do so, the assertion that checks
        # that the new absorption time is larger than the previous one will fail).   
        if end_of_simulation:
            if self.LOG:
                print("P={}: STOPPED BY MAX TIME (latest buffer change time: {:.3f} (max={:.3f}), buffer size={})".format(P, self.times_buffer[P], self.maxtime, self.particles[P].getBufferSize()))
            # Set the trajectories information of each server in the particle to the maximum simulation time
            # so that we only consider trajectory information up to that point.
            # This is needed to:
            # - be able to compare different simulation setups under the same nmeantimes parameter
            # - have a correct finalization process (with finalize()), i.e. that the times of the added
            # fictitious events --e.g. when finalizing by ABSORPTION-- do not mess up the count of the
            # number of particles blocked or the number of particles activated by time.
            for server in range(self.nservers):
                self.trajectories[P][server]['t'] += [ self.times_buffer[P] ]
                self.trajectories[P][server]['x'] += [ self.trajectories[P][server]['x'][-1] ]
                self.trajectories[P][server]['e'] += [ Event.END ]
        elif absorbed(P):
            # => Store the information about the absorption
            if self.LOG: #True:
                print("P={}: STOPPED BY ABSORPTION (latest buffer change time: {:.3f} < {:.3f}, buffer size={})".format(P, self.times_buffer[P], self.maxtime, self.particles[P].getBufferSize()))
            _, _, _, time_of_event = self.particles[P].getMostRecentEventInfo()
            absorption_time = time_of_event
            self._update_info_absorption_times(P, absorption_time, self.iterations[P])
            self._update_survival_time_from_zero(P, absorption_time)

    def _compute_order(self, arr, axis=-1):
        "Returns the order of the values in an array by the specified axis dimension"
        # axis=-1 means "the last dimension of the array"
        return np.argsort(arr, axis=axis)

    def _update_active_particles(self, P):
        "Updates the flag that states whether the given particle number P is active"
        self.is_particle_active[P] = not self._is_particle_absorbed(P)

    def _is_particle_activated(self, P):
        "Whether the particle is activated (regardless of the latest change experienced by it)"
        return self.particles[P].getBufferSize() == 1 

    def _is_particle_absorbed(self, P):
        "Whether the particle is absorbed (regardless of the latest change experienced by it)"
        return self.particles[P].getBufferSize() == 0 

    def _is_particle_blocked(self, P):
        "Whether the particle is blocked (regardless of the latest change experienced by it)"
        return self.particles[P].getBufferSize() == self.particles[P].getCapacity() 

    def _has_particle_become_activated(self, P, last_position_change):
        """
        Whether the particle's buffer size is 1 after a non-zero position change

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.

        last_position_change: int
            Last change experimented by the particle's buffer size.
        """
        # Note that a particle can be activated even if it was ALREADY at position 1 at some earlier event time => a new sub-trajectory becomes active
        return last_position_change != 0 and self._is_particle_activated(P)

    def _has_particle_become_absorbed(self, P, last_position_change):
        """
        Whether the particle is absorbed, which is indicated by a negative change in the system's buffer size
        and a system's buffer size equal to 0 after the change.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.

        last_position_change: int
            Last change experimented by the particle's buffer size.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        return last_position_change < 0 and self._is_particle_absorbed(P)

    def _has_particle_become_blocked(self, P, last_position_change):
        """
        Whether the particle becomes blocked, which is indicated by a positive change in the system's buffer size
        and a system's buffer size equal to its capacity after the change.
        
        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.

        last_position_change: int
            Last change experimented by the particle's buffer size.
        """
        return last_position_change > 0 and self._is_particle_blocked(P)       

    def _has_particle_become_unblocked(self, P, last_position_change):
        """
        Whether the particle becomes unblocked, which is indicated by a negative change in the system's buffer size
        and a system's buffer size equal to its capacity before the change.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.

        last_position_change: int
            Last change experimented by the particle's buffer size.
        """
        return last_position_change < 0 and self.particles[P].getBufferSize() == self.particles[P].getCapacity() + last_position_change

    def _check_particle_absorbed_based_on_most_recent_change(self, P):
        """
        Whether a particle has been absorbed based on the most recent change in the particle's servers.
        
        This function CANNOT be used when the latest event times for all servers is the same,
        notably at the very beginning of the simulation or when a reactivation has occurred
        (in which case, all the server queue sizes change at the same time, namely the absorption time
        leading to reactivation).
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        _, last_change, _, _ = self.particles[P].getMostRecentEventInfo()
        return self._has_particle_become_absorbed(P, last_change)

    def _update_info_absorption_times(self, P, time_of_absorption, it):
        "Inserts a new absorption time in order, together with its particle number"
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        idx_insort, found = insort(self.dict_info_absorption_times['t'], time_of_absorption)
        assert not found, "The inserted absorption time ({:.3f}) does NOT exist in the list of absorption times:\n{}".format(time_of_absorption, self.dict_info_absorption_times['t'])
        self.dict_info_absorption_times['P'].insert(idx_insort, P)

    def _update_survival_time_from_zero(self, P, time_of_absorption):
        ref_time = 0.0 if np.isnan(self.times0[P]) else self.times0[P]

        # Refer the survival time to the latest time the particle was at position 0
        # and add it to the previously measured survival time
        assert time_of_absorption > ref_time, \
                "The time of absorption ({}) contributing to the mean survival time calculation" \
                " is larger than the latest time of absorption for particle {} ({})" \
                .format(time_of_absorption, P, ref_time)
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
            print(">>>> P={}: UPDATED Last time at 0: {:.3f}".format(P, self.times0[P]))
            print(">>>> \tNew total killing times (ktimes0_sum): {:.3f}".format(self.ktimes0_sum[P]))

    def finalize(self):
        """
        Finalizes the simulation process by treating particles that are active,
        i.e. those particles providing CENSORED survival time values.
        
        The type of finalize process is defined by the object's attribute `finalize_type`.
        """
        active_particle_numbers = self.get_active_particle_numbers()
        active_particle_ids = list( np.sort([ self.particle_reactivation_ids[P] for P in active_particle_numbers ]) )

        finalize_process = "--NO FINALIZATION DONE--"
        if True: #self.LOG:
            if self.finalize_type == FinalizeType.ABSORB_CENSORED:
                finalize_process = "ABSORBING"
            elif self.finalize_type == FinalizeType.REMOVE_CENSORED:
                finalize_process = "REMOVING"
            elif self.finalize_type == FinalizeType.ESTIMATE_CENSORED:
                finalize_process = "CENSORING"
            #print("\n****** FINALIZING the simulation by {} the currently active particles: ******".format(finalize_process))
            #print("Active particles: {}" \
            #      .format(", ".join(["(p={}, P={})".format(p, self.info_particles[p]['particle number']) for p in active_particle_ids])))

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
            self.assertAllServersHaveSameTimeInTrajectory(P)
            #print("Processing particle ID p={}, P={} out of {} particles".format(p, P, len(active_particle_ids)))
            if self.finalize_type == FinalizeType.REMOVE_CENSORED:
                if self.reactivate:
                    if self.LOG: #True:
                        print("...{} particle p={} (P={})".format(finalize_process, p, P))
                    # Simply remove the particle from the analysis
                    self.info_particles.pop(p - nremoved)
                    nremoved += 1
                        # NOTE that no update needs to be done to the quantities
                        # used to estimate E(T) (e.g. ktimes0_sum) because these quantities are updated
                        # ONLY when there is an absorption, and this never happened
                        # for this particle ID (recall that, when reactivate=True,
                        # there is at most ONE absorption per particle ID,
                        # and when it occurs, it's the last event to occur).
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
                # The inserted time(s) happen AFTER the maximum simulation time.
                time_to_insert = self.maxtime + self.EPSILON_TIME*np.random.random()

                # 2) Update the information of special events so that calculations are consistent
                # with how trajectories are expected to behave (e.g. they go through ACTIVATION
                # before going to the forced absorption)
                if self.finalize_type == FinalizeType.ABSORB_CENSORED:
                    # To avoid assertion failures... make sure that, at the end of the simulation:
                    # - the particle is unblocked if it was blocked
                    #   (adding this preserves the assertion that the number of
                    #   blocked particles at the very last time is 0 --since the particles are all absorbed!)
                    # - the particle is activated if it was not in position 1
                    #   (this makes things logical --i.e. before absorption the particle passes by position 1)
                    if self._is_particle_blocked(P):
                        # Add an BLOCK time if the buffer size was not already BLOCKED
                        dict_info['t'] += [time_to_insert]
                        dict_info['E'] += [EventType.UNBLOCK]
                        # Increase each of the times to insert for the insertion of the next event
                        time_to_insert += self.EPSILON_TIME*np.random.random()
                    if not self._is_particle_activated(P):
                        # Add an ACTIVATION time if the buffer size was not already in the ACTIVATION set
                        dict_info['t'] += [time_to_insert]
                        dict_info['E'] += [EventType.ACTIVATION]
                        # Increase each of the time to insert for the insertion of the
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
            #print("Special events information:\n{}".format(dict_info))
            #input("Press Enter to continue...")
            if self.finalize_type != FinalizeType.NONE:
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
                            .format(event_type_prev.name, p, P, t)
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
                                .format(event_type_prev.name)
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
        # NOTE: The inserted times are NOT necessarily received in order... They are in order for each particle
        # but the number of blocked particles are computed over all particles, therefore the ordered set
        # of times is covered several times, one for each particle, and hence we need to find where
        # we should insert t in the set of blocking/unblocking times stored in self.sbu. 
        idx_insort, found = insort(self.sbu, t, unique=True)
        assert not found, "The absolute time value is NOT found in the list of block/unblock times ({})".format(t)
        #print("Time to insert: {:.3f} BETWEEN indices {} and {} (MAX idx={}):".format(t, idx_insort-1, idx_insort, len(self.counts_blocked)-1))
        self._update_counts_blocked(idx_insort, event_type, new=not found)

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
            if not 0 <= self.counts_blocked[idx] <= self.N:
                self.plot_trajectories_by_particle()
            assert 0 <= self.counts_blocked[idx] <= self.N, \
                    "Insertion index: {}\n".format(idx_to_insert_or_update) + \
                    "The number of blocked particles in time segment with index idx={} out of {} time segments " \
                    .format(idx, len(self.counts_blocked)) + \
                    "found so far is between 0 and {} ({}) \n({})" \
                    .format(self.N, self.counts_blocked[idx], self.counts_blocked)
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
        #        survival_time = self.particles[P].getTimesLastEvents() - self.times0[P]
        #        ### TODO: (2020/06/13) Write the get_survival_function() method that returns the estimate of S(t)
        #        self.ktimes0_sum[P] += survival_time # / (1 - self.get_survival_function(survival_time))
        #        self.ktimes0_n[P] += 1
        #else:
        if not self.finalize_type in [FinalizeType.NONE, FinalizeType.ESTIMATE_CENSORED]:
            assert sum(self.is_particle_active) == 0, "The number of active particles is 0 ({})".format(sum(self.is_particle_active))

        assert np.sum(self.ktimes0_n) > 0, "At least one particle has been absorbed. This is needed for the estimation of the expected survival time."

        # Note: When reactivation is used, the following estimate is not really accurate because the particles never start at position 0.
        # For that case, I used to use the following calculation, which doesn't really make sense:
        # self.expected_survival_time = np.mean([self.particles[P].getTimesLastEvents() for P in range(self.N)])
        print("Estimating expected survival time...")
        print("ktimes0_sum: {}".format(self.ktimes0_sum))
        print("ktimes0_n: {}".format(self.ktimes0_n))
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
        - the estimated blocking probability via Approximation 1 (integral approach)
        - the estimated blocking probability via Approximation 2 (Laplacian approach)
        - the value of the integral in Approximation 1 
        - the killing rate gamma in Approximation 2
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
            Time at which the position of the given particle should be retrieved.

        Return: numpy array
        The position is given by the size of each server that is serving the queue represented by the particle
        at the given time. 
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        assert t >= 0.0, "The queried time is non-negative, implying that there is always a valid position retrieved for the particle (t={})".format(t)
        particle_position = -1 * np.ones(self.nservers, dtype=int)
            ## initialize the particle position to an invalid value in case its value is not set in what follows
            ## (so we can identify an error occurred)
        for server in range(self.nservers):
            # Construct ALL the times at which the analyzed server changed position in its trajectory.
            # These are the times on which the given time t should be searched for in order to
            # retrieve the server's position (i.e. its queue size).
            server_times = self.trajectories[P][server]['t']
            server_positions = self.trajectories[P][server]['x']
            # Find the order in which the queried time t would be inserted in the list of server times
            # which allows us to know the position of the particle's server at that time.
            # Note that the queried time is 99.999999% of the time NOT present in the list of server times.
            # (unless it's 0, which is supposed to not be the case).
            idx = bisect.bisect(server_times, t)
            assert idx > 0, "The bisect index where the queried time would be inserted is positive ({})".format(idx)
            # Get the particle server's position at the queried time 
            particle_position[server] = server_positions[idx-1]

            if self.LOG:
                print("From get_position(Q={}, t={}):".format(P, t))
                print("\tserver: {}".format(server))
                print("\tserver_times: {}".format(server_times))
                print("\tserver_positions: {}".format(server_positions))
                print("\tretrieved particle's server position: {}".format(particle_position[server]))

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
            elif len(block_times_p) > len(unblock_times_p):
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
            # The difference with the following list of times and the previous list of times
            # is that the following list has an initial 0.0 which is needed to concatenate with the survival times below
            # to generate the output object.
            absorption_and_censoring_times[P] += absorption_and_censoring_times_p
            #print("absorption times: {}".format(absorption_and_censoring_times[P]))

        for P in range(self.N):
            survival_times_P = np.diff(absorption_and_censoring_times[P])
            #print("P={}: survival times: {}".format(P, survival_times_P))
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
            return self.particles[P].getTypesLastEvents()
        except:
            self.raiseErrorInvalidParticle(P)
            return None

    def get_all_times_last_event(self):
        return [self.get_time_last_event(P) for P in range(self.N)]

    def get_time_last_event(self, P):
        try:
            return self.particles[P].getTimesLastEvents()
        except:
            self.raiseErrorInvalidParticle(P)
            return None

    def get_all_times_next_events(self):
        return self.times_next_events

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

    def plot_trajectories_by_particle(self):
        "Plots the trajectories of the particles"
        K = self.queue.getCapacity()
        colormap = cm.get_cmap("jet")
        reflines = range(0, (K+1)*self.N, K+1)
        reflines_block = range(K, (K+(K+1))*self.N, K+1)
        particle_numbers = list(range(self.N))
        if False:
            for p in particle_numbers:
                print("\nParticle number {}:".format(p))
                print("Buffer Times and Sizes:")
                print(self.all_times_buffer[p])
                print(self.all_positions_buffer[p])
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel("t")
        ax.set_ylabel("particle (buffer size plotted)")
        ax.set_yticks(reflines)
        ax.yaxis.set_ticklabels(particle_numbers)
        ax.xaxis.set_ticks(np.arange(0, round(self.maxtime)+1) )
        ax.set_ylim((0, (K+1)*self.N))
        ax.hlines(reflines_block, 0, self.maxtime, color='gray', linestyles='dashed')
        ax.hlines(reflines, 0, self.maxtime, color='gray')
        ax.vlines(self.maxtime, 0, (K+1)*self.N, color='red', linestyles='dashed')
        for p in particle_numbers:
            color = colormap( (p+1) / self.N )
            # Non-overlapping step plots at vertical positions (K+1)*p
            plt.step(self.all_times_buffer[p], [(K+1)*p + pos for pos in self.all_positions_buffer[p]], 'x-',
                     where='post', color=color, markersize=3)
        plt.title("K={}, rate(B)={:.1f}, rate(D)={:.1f}, reactivate={}, finalize={}, N={}, maxtime={:.1f}, seed={}" \
                  .format(self.queue.getCapacity(),
                      self.queue.getBirthRate(),
                      self.queue.getDeathRate(),
                      self.reactivate, self.finalize_type.name[0:3], self.N, self.maxtime, self.seed
                      ))
        ax.title.set_fontsize(9)
        plt.show()

    def plot_trajectories_by_server(self, P):
        "Plots the trajectories of the particles"
        K = self.queue.getCapacity()
        colormap = cm.get_cmap("jet")
        reflines = range(0, (K+1)*(self.nservers+1), K+1)       # We add a server (#self.nservers+1) because the last "server" will show the system's buffer size
        reflines_block = range(K, (K+(K+1))*(self.nservers+1), K+1)
        servers = range(self.nservers)
        if False:
            print("\nP={}:".format(P))
            print("Times and Positions by Server:")
            print(self.all_times_by_server[P])
            print(self.all_positions_by_server[P])
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel("t")
        ax.set_ylabel("server (queue size plotted)")
        ax.set_yticks(reflines)
        ax.yaxis.set_ticklabels(servers)
        ax.xaxis.set_ticks(np.arange(0, round(self.maxtime)+1) )
        ax.set_ylim((0, (K+1)*(self.nservers+1)))
        ax.hlines(reflines_block, 0, self.maxtime, color='gray', linestyles='dashed')
        ax.hlines(reflines, 0, self.maxtime, color='gray')
        ax.vlines(self.maxtime, 0, (K+1)*(self.nservers+1), color='red', linestyles='dashed')
        for s in servers:
            color = colormap( (s+1) / self.nservers )
            # Non-overlapping step plots at vertical positions (K+1)*s
            plt.step(self.all_times_by_server[P][s] , [(K+1)*s + pos for pos in self.all_positions_by_server[P][s]], 'x-',
                     where='post', color=color, markersize=3)
            # Complete the line of each server up to the buffer time 
            ax.hlines((K+1)*s + self.all_positions_by_server[P][s][-1], self.all_times_by_server[P][s][-1], self.times_buffer[P], color=color)
        # Add the system's buffer size on top
        plt.step(self.all_times_buffer[P], [(K+1)*self.nservers + pos for pos in self.all_positions_buffer[P]], 'x-',
                     where='post', color="black", markersize=3)
        
        plt.title("Particle {}: K={}, rate(B)={:.1f}, rate(D)={:.1f}, reactivate={}, finalize={}, #servers={}, maxtime={:.1f}, seed={}" \
                  .format(P,
                    self.queue.getCapacity(),
                    self.queue.getBirthRate(),
                    self.queue.getDeathRate(),
                    self.reactivate, self.finalize_type.name[0:3], self.nservers, self.maxtime, self.seed
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
                    "\nmaxtime = {:.3f}" \
                    "\nseed = {}" \
                    "\n***********************" \
                    .format(self.queue.getCapacity(),
                            self.queue.getBirthRate(), self.queue.getDeathRate(),
                            self.queue.getBirthRate() / self.queue.getDeathRate(),
                            self.N, self.START, self.mean_lifetime,
                            self.reactivate, self.finalize_type.name, self.maxtime, self.seed)
        return params_str

    def isLastIteration(self):
        return self.iter == self.niter

    def isValidParticle(self, P):
        "Returns whether the given particle is a valid particle number indexing the list of particle queues"
        return (isinstance(P, int) or isinstance(P, np.int32) or isinstance(P, np.int64)) and 0 <= P < self.N

    def isValidParticleId(self, p):
        "Returns whether the given particle is a valid particle ID indexing the info_particles attribute"
        return (isinstance(p, int) or isinstance(p, np.int32) or isinstance(p, np.int64)) and 0 <= p < len(self.info_particles)

    def isValidServer(self, P, server):
        "Returns whether the given server for the given particle is a valid server number"
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        return (isinstance(server, int) or isinstance(server, np.int32) or isinstance(server, np.int64)) and 0 <= server < self.particles[P].getNServers()

    def assertTimeMostRecentEventAndTimesNextEvents(self, P):
        """
        Checks consistency of information stored in the "next event" attributes w.r.t.
        the information on the most recent event when the next event times are generated
        for the steady state of the simulation (i.e. NOT for iteration = 0).
        """
        server_last_updated, _, type_of_last_event, time_of_last_event = self.particles[P].getMostRecentEventInfo()
        if self.iterations[P] > 0 and \
           not np.isnan(self.times_next_events2[P][server_last_updated][type_of_last_event.value]):
            # The following assertion is only valid when the most recent event is NOT the first event
            # of the server that is going to be updated now (in which case the current value
            # of the next event time for the type of the most recent event is NaN --which justifies the IF condition above)
            assert time_of_last_event == self.times_next_events2[P][server_last_updated][type_of_last_event.value], \
                "The time of the most recent event ({:.3f}) of type {} ".format(time_of_last_event, type_of_last_event) + \
                "coincides with the time of the next {} currently stored in the times_next_events2 for particle {} on server {} ({:.3f})" \
                .format(type_of_last_event, P, server_last_updated, self.times_next_events2[P][server_last_updated][type_of_last_event.value]) + \
                "\nTimes of next events of both types for server {}: {}".format(server_last_updated, self.times_next_events2[P][server_last_updated]) + \
                "\nTimes of next events in server {}'s queue: {}".format(server_last_updated, self.times_next_events_in_queue[P][server_last_updated])

    def assertTimeToInsertInTrajectories(self, P, server=0):
        #for p in range(self.N):
        #    for server in range(self.nservers):
        #        print("p={}, server={}: times={}".format(p, server, self.trajectories[p][server]['t']))
        assert self.particles[P].getTimeLastEvent(server) > self.trajectories[P][server]['t'][-1], \
                "The time to insert in the trajectories dictionary ({}) for particle {} and server {}" \
                " is larger than the latest inserted time ({})" \
                .format(self.particles[P].getTimeLastEvent(server), P, server, self.trajectories[P][server]['t'][-1])

    def assertAllServersHaveSameTimeInTrajectory(self, P):
        """
        Asserts that all the servers in the given particle are aligned in time in terms of their trajectory information
        and equal to the last time the buffer size of the particle changed.
        """
        for server in range(self.nservers):
            if self.trajectories[P][server]['t'][-1] != self.times_buffer[P]:
                return False
        return True

    def assertSystemConsistency(self, particle_number=None, time=None):
        is_state_consistent_for_particle = \
            lambda P: self.positions_buffer[P] == self.particles[P].getBufferSize() and self.positions_buffer[P] == self.all_positions_buffer[P][-1] and all(self.particles[P].getTimesLastEvents() == [self.all_times_by_server[P][server][-1] for server in range(self.nservers)])

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
                self.plot_trajectories_by_server(particle_number)
            assert consistent_state, \
                    "Iter {}, {}: The system is NOT in a consistent state at least for particle {}" \
                    .format(self.iter, time_str, particle_number)           
        else:
            consistent_state = is_state_consistent_for_particle(particle_number)
            if not consistent_state:
                self.plot_trajectories_by_server(particle_number)
            assert consistent_state, \
                    "Iter {}, {}: The system is NOT in a consistent state for particle {}" \
                    "\n\tpositions[{}]={} vs. particles[{}].size={}" \
                    "\n\tpositions[{}]={} vs. all_positions[{}][-1]={}" \
                    "\n\tparticles[{}].TimeLastEvent={} vs. all_times[{}][-1]={}" \
                    .format(self.iter, time_str, particle_number,
                            particle_number, self.positions_buffer[particle_number], particle_number, self.particles[particle_number].size,
                            particle_number, self.positions_buffer[particle_number], particle_number, self.all_positions[particle_number][-1],
                            particle_number, self.particles[particle_number].getTimesLastEvents(), particle_number, self.all_times[particle_number][-1])         

    def raiseErrorInvalidParticle(self, P):
        raise ValueError("Wrong particle number: {}. Valid values are integers between 0 and {}.\n".format(P, self.N-1))

    def raiseWarningInvalidParticle(self, P):
        raise Warning("Wrong particle number: {}. Valid values are integers between 0 and {}.\n".format(P, self.N-1))
