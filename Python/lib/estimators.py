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
from timeit import default_timer as timer

if __name__ == "__main__":
    # Needed to run tests (see end of program)
    import runpy
    runpy.run_path('../../setup.py')

    from datetime import datetime
    from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses
    import Python.lib.agents as agents
    from agents.policies.parameterized import PolQueueRandomizedLinearStep
    import Python.lib.queues as queues  # The keyword `queues` is used in test code
    from Python.lib.queues import Event, GenericQueue
    from Python.lib.utils.basic import array_of_objects, find, find_last, find_last_value_in_list, insort, list_contains_either, merge_values_in_time
    from Python.lib.utils.computing import compute_blocking_probability_birth_death_process, stationary_distribution_birth_death_process, stationary_distribution_birth_death_process_at_capacity_unnormalized
else:
    from .environments.queues import EnvQueueSingleBufferWithJobClasses
    from .agents.policies.parameterized import PolQueueRandomizedLinearStep
    from .queues import Event, GenericQueue
    from .utils.basic import array_of_objects, find, find_last, find_last_value_in_list, insort, list_contains_either, merge_values_in_time
    from .utils.computing import stationary_distribution_birth_death_process, stationary_distribution_birth_death_process_at_capacity_unnormalized

@unique # Unique enumeration values (i.e. on the RHS of the equal sign)
class EventType(Enum):
    ACTIVATION = +1
    REACTIVATION = +2
    ABSORPTION = 0
    BLOCK = +9
    UNBLOCK = -9
    START_POSITION = -1
    # Fictitious events... added when absorbing active particles when FinalizeType = ABSORB_CENSORED
    ACTIVATION_F = +11
    ABSORPTION_F = +10
    UNBLOCK_F = -19
    # Censoring event... added when FinalizeType = ESTIMATE_CENSORED
    CENSORING = +99 # To mark an event as censoring, i.e. for censored observations a new event of this type
                    # is added to the event history of the particle when using FinalizeType.ESTIMATE_CENSORED.  

@unique # Unique enumeration values (i.e. on the RHS of the equal sign)
class FinalizeType(Enum):
    "Type for the FINALIZE actions performed in the finalize() method"
    ESTIMATE_CENSORED = 1
    REMOVE_CENSORED = 2
    ABSORB_CENSORED = 3
    NONE = 9

@unique # Unique enumeration values (i.e. on the RHS of the equal sign)
class FinalizeCondition(Enum):
    "Condition for the FINALIZE actions performed in the finalize() method"
    ACTIVE = 1
    NOT_START_POSITION = 2

class EstimatorQueueBlockingFlemingViot:
    """
    Estimator of the blocking probability of queues using the Fleming Viot particle system
    
    Arguments:
    nparticles: int
        Number of particles to consider in the Fleming Viot system.

    queue: subclass of GenericQueue
        The object should have the following attributes:
        - rates: a list with two elements corresponding to the birth and death events
        at the positions specified by the Event.BIRTH and Event.DEATH values of the Event Enum.

        The object should have the following methods implemented:
        - apply_event()
        - generate_event_time()
        - getCapacity()
        - getBufferSize()
        - getLastChange()
        - getMostRecentEventTime()
        - getMostRecentEventInfo()
        - getNServers()
        - getTimeLastEvent()
        - getTimesLastEvents()
        - getTypesLastEvents()
        - getTimeLastEvent()
        - getTypeLastEvent()
        - getServerSize()
        - getServerSizes()
        - getBirthRates()
        - getDeathRates()
        - setBirthRates()
        - setDeathRates()
        - resize()

    job_rates: list
        List of arrival rate for every possible job class. The job class values are the indices of the given list. 

    service_rates: (opt) list
        Service rates of each server.
        When given, it overrides the death rates given in the `queue` parameter.
        Otherwise, the death rates of the `queue` parameter are used as service rates.
        default: None

    buffer_size_activation: (opt) int
        Buffer size that makes the particle be active, i.e. go from the state space of absorption states
        to the rest of the state space.
        default: 1

    positions_observe: (opt) list
        List of particle positions (buffer sizes) to observe, i.e. to record every visit to any of these positions
        This is normally used to estimate the return time to the set of these positions.
        default: [ buffer_size_activation - 1 ]

    nmeantimes: (opt) int
        Multiple of the mean time of the job class with LOWEST arrival rate defining the maximum simulation time.
        Ex: If there are two job classes with arrival rates respectively 2 jobs/sec and 3 jobs/sec,
        the maximum simulation time is computed as `nmeantimes * 1/2` sec, which is expected to include
        `nmeantimes` jobs of the lowest arrival rate job on average.
        For instance, if nmeantimes = 10, the simulation time is 5 sec and is expected to include 10 jobs
        of the lowest arrival rate class, and `(nmeantimes * 1/2) / (1/3) = 5*3` = 15 jobs of the
        highest arrival rate class.
        default: 3

    policy_accept: (opt) policy
        Policy that defines the acceptance of an arriving job to the queue.
        It is expected to define a queue environment of type EnvQueueSingleBufferWithJobClasses
        on which the policy is applied that defines the rewards received for accepting or rejecting
        a job of a given class.
        The queue environment should have at least the following methods defined:
        - getNumJobClasses()
        - setJobClass() which sets the class of the arriving job
        default: None (which means that all jobs are accepted)

    policy_assign: (opt) list of lists
        List of probabilities of assigning each job class associated to each job rate given in `job_rates`
        to a server in the queue.
        Ex: In a scenario with 2 job classes and 3 servers, the following policy assigns job class 0
        to server 0 or 1 with equal probability and job class 1 to server 1 or 2 with equal probability:
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
        default: None (in which case the assignment probability is uniform over the servers)

    mean_lifetime: (opt) positive float
        Mean particle lifetime to be used as the expected survival time.
        This is useful when reactivate=True, as in that case
        the particles never touch any state in the absorption set (e.g. those with buffer-size/position=0),
        which means that the mean lifetime cannot be computed in that scenario.
        default: None

    proba_survival_given_activation: (opt) pandas data frame
        A separate estimation of the probability of survival given the process started at the activation set:
        Pr(T>t / s in activation set).
        Typically this estimation is obtained by running the process in "Monte-Carlo mode",
        i.e. with parameter reactivate=False.
        It should be a pandas data frame with at least two columns:
        - 't': the times at which the survival probability is measured. 
        - 'P(T>t / s=1)': the survival probability for each t.

    reactivate: (opt) bool
        Whether to reactivate a particle after absorption to a positive position.
        Note that absorption occurs when the particle goes to position (0, 0, ..., 0)
        where the array of zeros has length equal to the number of servers.
        That is, the particle is an ARRAY of server sizes, which should ALL be idle
        in order for the particle to be considered absorbed.
        
        When reactivate=True, the initial position of the system is (1, 0, ..., 0),
        i.e. the initial buffer size is 1 (as the system can never be absorbed)
        When reactivate=False, the initial position of the system is (0, 0, ..., 0).
        This setting is used when we are interested in estimating the expected survival time
        (which is used as input to the system run under reactivate=True to estimate
        the blocking probability under the Fleming-Viot algorithm). 
        default: True

    finalize_info: (opt) dict
        Dictionary with the following two attributes:
        - 'type' :FinalizeType: type of simulation finalize actions, either:
            - FinalizeType.ESTIMATE_CENSORED
            - FinalizeType.REMOVE_CENSORED
            - FinalizeType.ABSORB_CENSORED
            - FinalizeType.NONE
            default: FinalizeType.ABSORB_CENSORED
        - 'condition' :FinalizeCondition: condition to be satisfied by a particle
        in order to have finalize actions applied to it, either:
            - FinalizeCondition.ACTIVE: the particle is not at an active state,
            i.e. a state that belongs to the activation set.
            - FinalizeCondition.NOT_START_POSITION: the particle is not at one of 
            the states where the particle could have started. For single-server systems
            the possible start states are just one, namely the buffer size at which the
            particle was at the start of the simulation; for multi-server systems
            the possible start states are any state having an associated buffer size equal
            to the buffer size of the particle at the start of the simulation.   
            default: FinalizeCondition.ACTIVE

    seed: (opt) int
        Random seed to use for the random number generation by numpy.random.
        default: None

    plotFlag: (opt) bool
        Whether to plot the trajectories of the particles.
        default: False

    log: (opt) bool
        Whether to show messages of what is happening with the particles.
        default: False
    """
    def __init__(self, nparticles :int, queue, job_rates :list,
                 service_rates=None,
                 buffer_size_activation=1, positions_observe :list=[],
                 nmeantimes=3,
                 policy_accept=None,
                 policy_assign=None,
                 mean_lifetime=None, proba_survival_given_activation=None,
                 reactivate=True, finalize_info={'type': FinalizeType.ABSORB_CENSORED, 'condition': FinalizeCondition.ACTIVE},
                 seed=None, plotFlag=False, log=False):
        if reactivate and nparticles < 2:
            raise ValueError("The number of particles must be at least 2 when reactivate=True ({})".format(nparticles))
            import sys
            sys.exit(-1)
        self.N = int( nparticles )          # Make sure the number of particles is of int type
        self.queue = copy.deepcopy(queue)   # We copy the queue so that we do NOT change the `queue` object calling this function
        self.job_rates = job_rates          # This should be a list of arrival rates for each job class (which are the indices of this list)
        self.buffer_size_activation = buffer_size_activation        # Buffer size defining the ACTIVATION set of server sizes (n1, n2, ..., nR)
        if positions_observe == []:
            # If the given position (buffer size) to observe is empty, we observe the absorption buffer size
            # because that's normally the buffer size of interest (e.g. to estimate E(T), the return time to the absorption set)
            self.positions_observe = [buffer_size_activation - 1]# List of positions to observe, i.e. to record every visit to any of these positions (this is used to estimate the return time to the set of said positions)
        else:
            self.positions_observe = positions_observe
        self.set_activation = { self.buffer_size_activation }       # Set of the ACTIVATION buffer sizes, which defines the boundary set of states between the absorption state set and the rest of states. Note that this is part of the REST of the states, not part of the absorption set of states.
        self.set_absorption = set( range(buffer_size_activation) )  # Set of ABSORPTION BUFFER SIZES (note that it does NOT include the activation buffer size), whose possible R-length states (n1, n2, ..., nR) (R=nservers) are used as starting points to estimate the expected survival time
        self.policy_accept = policy_accept          # Policy for acceptance of new arriving job of a given class
        if policy_assign is None:                   # Assignment policy of a job to a server. It should be a mapping from job class to server.
            # When no assignment probability is given, it is defined as uniform over the servers for each job class
            policy_assign = [ [1.0 / self.queue.getNServers()] * self.queue.getNServers() ] * len(self.job_rates) 
        self.policy_assign = policy_assign

        # Update the birth and death rates of the given queue
        # NOTE: (2021/02/10) Currently the birth rates   
        equivalent_birth_rates = self.compute_equivalent_birth_rates()
        self.queue.setBirthRates(equivalent_birth_rates)
        if service_rates is not None:
            self.queue.setDeathRates(service_rates)

        self.nmeantimes = nmeantimes
        self.maxtime = nmeantimes * np.max( 1/np.array(self.job_rates) ) # The max simulation time is computed as a multiple (given by the user) of the mean arrival time of the lowest frequency job
        self.nservers = self.queue.getNServers()

        #-- Parameter checks
        if self.policy_accept is not None and self.policy_accept.env.getNumJobClasses() != len(self.job_rates):
            raise ValueError("The number of job classes defined in the queue environment ({}) " + \
                             "must be the same as the number of job rates given as argument to the constructor ({})" \
                             .format(self.policy_accept.env.getNumJobClasses(), len(self.job_rates)))

        if self.queue.getCapacity() <= 1:
            # This condition must be satisfied because ONLY ONE EVENT is accepted for each event time
            # i.e. ACTIVATION, BLOCK, UNBLOCK or ABSORPTION.
            # If we allowed a queue capacity of 1, when the position of the particle goes to 1, the particle
            # is both ACTIVATED and BLOCKED, a situation that cannot be handled with the code as is written now.
            # In any case, a capacity of 1 is NOT of interest.   
            raise ValueError("The maximum position of the particles must be larger than 1 ({})".format(self.queue.getCapacity()))

        if self.buffer_size_activation >= self.queue.getCapacity() - 1:
            # Note: normally we would accept the activation size to be equal to Capacity-1...
            # HOWEVER, this creates problem when the particle is unblocked because the UNBLOCK event
            # will NOT be recorded and instead an ACTIVATION event will be recorded.
            # So, we want to avoid this extreme situation.
            raise ValueError("The buffer size for activation ({}) must be smaller than the queue's capacity-1 ({})".format(self.buffer_size_activation, self.queue.getCapacity()-1))

        if mean_lifetime is not None and (np.isnan(mean_lifetime) or mean_lifetime <= 0):
            raise ValueError("The mean life time must be positive ({})".format(mean_lifetime))

        if proba_survival_given_activation is not None \
            and not isinstance(proba_survival_given_activation, pd.DataFrame) \
            and 't' not in proba_survival_given_activation.columns \
            and 'P(T>t / s=1)' not in proba_survival_given_activation.columns:
            raise ValueError("The proba_survival_given_activation parameter must be a data frame having 't' and 'P(T>t / s=1)' as data frame columns")

        if  finalize_info['condition'] == FinalizeCondition.NOT_START_POSITION and \
            finalize_info['type'] == FinalizeType.ABSORB_CENSORED:
            raise ValueError("The finalize type CANNOT be {} when the finalize condition is {}".format(FinalizeType.ABSORB_CENSORED, FinalizeCondition.NOT_START_POSITION))
        #-- Parameter checks

        self.mean_lifetime = mean_lifetime  # A possibly separate estimation of the expected survival time
        self.proba_survival_given_activation = proba_survival_given_activation  # A possibly separate estimation of the probability of survival having started at the activation set (based on the probability of occurrence of such set)
        self.reactivate = reactivate        # Whether the particle is reactivated to a positive position after absorption
        self.finalize_info = finalize_info  # How to finalize the simulation
        self.plotFlag = plotFlag
        self.seed = seed
        self.LOG = log

        #-- Set a parameter used in the simulation, for technical reasons
        # Epsilon time --> used to avoid repetition of times when reactivating a particle
        # This epsilon time takes into account the scale of the problem by making it 1E6 times smaller
        # than the minimum of the event rates.
        # I.e. we cannot simply set it to 1E-6... what if the event rate IS 1E-6??
        self.EPSILON_TIME = 1E-6 * np.min([ self.queue.getBirthRates(), self.queue.getDeathRates() ])

        #-- Compute the theoretical absorption and activation distributions
        self.rhos = [lmbda / mu for lmbda, mu in zip(self.queue.getBirthRates(), self.queue.getDeathRates())]
        self.states_absorption, self.dist_absorption = stationary_distribution_birth_death_process(self.nservers, self.buffer_size_activation-1, self.rhos)
        self.states_absorption_at_boundary, self.dist_absorption_at_boundary = stationary_distribution_birth_death_process_at_capacity_unnormalized(self.nservers, self.buffer_size_activation-1, self.rhos)
        # Normalize the distribution on the absorption states so that it is a probability from which we can sample
        self.dist_absorption_at_boundary = self.dist_absorption_at_boundary / np.sum(self.dist_absorption_at_boundary)
        self.states_activation, self.dist_activation = stationary_distribution_birth_death_process_at_capacity_unnormalized(self.nservers, self.buffer_size_activation, self.rhos)
        # Normalize the distribution on the activation states so that it is a probability from which we can sample
        self.dist_activation = self.dist_activation / np.sum(self.dist_activation)

        #print("Distribution of ABSORPTION states:")
        #[print("index={}: x={}, p={:.6f}".format(idx, x, p)) for idx, (x, p) in enumerate(zip(self.states_absorption, self.dist_absorption))]
        #print("Distribution of ACTIVATION states:")
        #[print("index={}: x={}, p={:.6f}".format(idx, x, p)) for idx, (x, p) in enumerate(zip(self.states_activation, self.dist_activation))]

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
        # History of all positions and event times for each particle
        # We create two pair of structures:
        # - one that keeps track of the positions and times of the system's SINGLE buffer
        # - one that keeps track of the positions and times of each particle's server
        # In each of these structures, the entry is a list that may have different length
        # depending on the particle and server within each particle.
        self.all_positions_buffer = array_of_objects((self.N,), dtype=list)
        self.all_times_buffer = array_of_objects((self.N,), dtype=list, value=[0.0])
        # NOTE: The difference between the all_times_by_server array and the trajectories array
        # is that the trajectories array will NEVER have a repeated time value and therefore can be used
        # to compute statistics necessary for the estimations of the blocking probability.
        # On the other hand, the all_times_by_server array has repeated times when the particle is absorbed
        # and reactivated, as the absorption time and reactivation time coincide and are both stored in the attribute
        # (for information purposes, in case we need to analyze something) 
        self.all_positions_by_server = np.empty((self.N, self.nservers), dtype=list)
        self.all_times_by_server = array_of_objects((self.N, self.nservers), dtype=list, value=[0.0])
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
        # Times (ABSOLUTE) of the last jobs arrived for each class and of the last jobs served by each server
        # These are needed in order to compute the ABSOLUTE time of the next arriving job of each class
        # and of the next served job for each server, since the arrival and service times are generated
        # RELATIVE to the time of the latest respective event.
        # The times of the last jobs arrived per class is also used to decide WHEN new arriving job times
        # should be generated and thus make the process more efficient (as opposed to generating new arriving job times
        # at every iteration (i.e. at every call to generate_one_iteration()) which may quickly fill
        # up the server queues with times that will never be used, significantly slowing down the simulation process)        
        self.times_last_jobs = np.zeros((self.N, len(self.job_rates)), dtype=float)
        self.times_last_services = np.zeros((self.N, self.nservers), dtype=float)
        #--------------------------- Last jobs and services information -----------------------


        #------------------------- Absorption times and active particles ----------------------
        #-- Information about the absorption times, which is used to complete the simulation after each absorption
        # This dictionary initially (i.e. before any re-activation of a particle) contains the
        # FIRST absorption time of each particle. Recall that there is ONE absorption time per particle.
        # Once the reactivation process starts, the dictionary contains every new absorption time observed.
        # For more information see the generate_trajectories_until_end_of_simulation() method or similar.
        self.dict_info_absorption_times = dict({'t': [],        # List of absorption times (one for each particle) (they should be stored SORTED, so that reactivation starts with the smallest absorption time)
                                                'P': [],        # List of particle numbers associated to each absorption time in 't'
                                                })
        #------------------------- Absorption times and active particles ----------------------


        #----------------- Attributes for survival and blocking time calculation --------------
        # Arrays that are used to estimate the expected survival time (time to killing starting at an absorption state)
        self.ktimes0_sum = np.zeros(self.N, dtype=float)    # Times to absorption from latest time it changed to an absorption state
        self.ktimes0_n = np.zeros(self.N, dtype=int)        # Number of times the particles were absorbed

        # Arrays that are used to estimate the blocking proportion as a rough estimate of the blocking probability
        self.timesb = np.nan*np.ones(self.N, dtype=float)   # Latest blocking time (used to update btimes_sum)
        self.btimes_sum = np.zeros(self.N, dtype=float)     # Total blocking time
        self.btimes_n = np.zeros(self.N, dtype=int)         # Number of times each particle goes to blocking state
        #----------------- Attributes for survival and blocking time calculation --------------


        #-------------------- Attributes for blocking probability estimation ------------------
        # Attributes used directly in the computation of the estimated measures that contribute to the probability of blocking.
        self.sk = [0]   # Times of an ABSORPTION (Killing) relative to ALL the activation times of the absorbed particle.
                        # The list is updated whenever an active particle is absorbed (i.e. touches an absorption state)
                        # GOAL: define the disjoint time segments on which the estimated survival probability is constant.
                        # SIZE AT THE END OF SIMULATION: as many as particles have been activated during the simulation.
        self.sbu = [0]  # ABSOLUTE times of a BLOCK/UNBLOCK.
                        # Size: As many elements as number of times the particle becomes blocked or unblocked. 

        # Particle counts used in the computation of the estimated measures
        # Note that these initial values are ALWAYS 0, regardless of the initial position of the particles
        # (whether 0 or 1). For more details see the comments in the compute_counts() method.
        self.counts_alive = [0] # Number of particles that are alive in each time segment defined by sk
                                # => have not been absorbed.
                                # The time segment associated to the count has the time value in the sk array
                                # as its lower bound.
                                # Size: same as the times of ABSORPTION array
        self.counts_blocked = [0]# Number of particles that are blocked in each time segment defined by sb
                                # => have reached position K, the maximum capacity of the queue represented by the particle. 
                                # Size: same as the times of BLOCK/UNBLOCK array
        #-------------------- Attributes for blocking probability estimation ------------------


        #--------------- Attributes for recording return times to observe state ---------------
        self.rtimes_offset = np.zeros(self.N, dtype=float)      # Offset to substract for every new contribution of a return time to an observed position 
        self.rtimes_obs_sum = np.zeros(self.N, dtype=float)     # Total return time for each particle
        self.rtimes_obs_n = np.zeros(self.N, dtype=int)         # Number of observed return times for each particle
        #--------------- Attributes for recording return times to observe state ---------------


        # Latest time at which we know the state of the system for each particle.
        # This means, that for all times <= this time we now precisely the size of ALL servers in the system.
        self.times_latest_known_state = np.zeros((self.N,), dtype=float)

        if self.LOG:
            print("Particle system with {} particles has been reset:".format(self.N))
            print("")

    def reset_positions(self, start_event_type, N_min :int=10):
        """
        Resets the positions of the particles used in the simulation
        and possibly re-defines the number of particles to simulate in order to
        satisfy the condition given by N_min, the minimum number of particles to simulate
        for a given start state.  
        
        Arguments:
        start_event_type: EventType
            Type of event associated to the set of feasible start states.
            E.g. EventType.ABSORPTION defines the set of absorption states, and
            EventType.ACTIVATION defines the set of activation states. 
        
        N_min: (opt) int
            Minimum number of particles to simulate in each set of start states.
            default: 10 

        Return: list
        List containing the number of particles to simulate for each feasible start state
        defined by the distribution of the set of states defined by the `start_event_type`
        (e.g. ABSORPTION or ACTIVATION states).  
        """
        np.random.seed(self.seed)
        
        if start_event_type == EventType.ABSORPTION:
            states = self.states_absorption_at_boundary
            dist_states = self.dist_absorption_at_boundary
        elif start_event_type == EventType.ACTIVATION:
            states = self.states_activation
            dist_states = self.dist_activation
        else:
            raise Warning("reset_positions: Invalid value for parameter 'start_event_type': {}".format(start_event_type))

        #-- Define the initial state of each particle using the stationary distribution of the states in the absorption set
        # Set the minimum number of particles to 10
        # This minimum happens at the largest buffer size, since it has the smallest probability
        nparticles_by_start_state = [0] * len(dist_states)
        N = int( round( N_min / dist_states[-1] ) )
        # Update the number of particles in the system and reset it before the simulation starts
        # if the N just computed is larger than the one stored in the object.
        # Otherwise, use the number of particles N stored in the object; in this case
        # the minimum number of particles will be at least N_min.
        if N > self.N:
            self.N = N
            self.reset()
        # Compute the number of particles to use for each start state
        for idx in range( len(nparticles_by_start_state)-1, 0, -1 ):
            nparticles_by_start_state[idx] = int( round( dist_states[idx] * self.N ) )
        nparticles_by_start_state[0] = self.N - int( np.sum(nparticles_by_start_state[1:]) )

        # Number of particles to assign for each possible start state
        #nparticles_by_start_state = [int( round( p*self.N ) ) for p in dist_states]
        # Make the last number of particles be the REMAINING number (to counteract rounding issues)
        #nparticles_by_start_state[-1] = self.N - int( np.sum(nparticles_by_start_state[:-1]) )
        
        if False:
            print("# particles by start_state:")
            for i, (x, p, n) in enumerate( zip( states, dist_states, nparticles_by_start_state ) ):
                print("{}: x={}, p={} --> N={}".format(i, x, p, n))

        idx_state_space = -1
        P_first_in_block = 0
        for nparticles_in_block in nparticles_by_start_state:
            idx_state_space += 1
            P_last_in_block = P_first_in_block + nparticles_in_block - 1
            print("Block of particle indices to simulate #{}: [{}, {}] (N={}, N/Ntotal={}, p={}, state={})".format(idx_state_space, P_first_in_block, P_last_in_block, nparticles_in_block, nparticles_in_block/self.N, dist_states[idx_state_space], states[idx_state_space]))
            for P in range(P_first_in_block, P_last_in_block + 1):
                self.particles[P].setServerSizes( states[idx_state_space] )

                # 1)-- Trajectories, and current and historical particle positions
                for server in range(self.nservers):
                    # Trajectories: one value per particle and per server
                    self.trajectories[P][server]['t'] = [0.0]                   # event times at which the particle changes position in each server
                    self.trajectories[P][server]['x'] = [ self.particles[P].getServerSize(server) ]    # positions at the server taken by the particle after each event
                    self.trajectories[P][server]['e'] = [ Event.RESET ]         # events (birth, death, reset) associated to the event times of each server
                    # Historical positions by server
                    self.all_positions_by_server[P][server] = [ self.particles[P].getServerSize(server) ]

                # Particle positions by server and overall (buffer size), current and historical (all_)
                self.positions_by_server[P] = self.particles[P].getServerSizes()
                self.positions_buffer[P] = np.sum( self.positions_by_server[P] )
                self.all_positions_buffer[P] = [ self.positions_buffer[P] ]
    
                # 2)-- Particle's info (one value per particle)
                self.info_particles += [ dict({
                                               # We initialize 't' and 'E' to the situation at which each particle starts-off (either ABSORBED (when no reactivation is performed) or with an ACTIVATION (when reactivation is performed))
                                               # This is necessary to guarantee the correct functioning of assertions, such as the one that asserts that before an ABSORPTION always comes an ACTIVATION.
                                               't': [0.0],                  # times at which the events of interest take place (e.g. ACTIVATION, ABSORPTION, BLOCKING)
                                               'E': [ [start_event_type, EventType.START_POSITION] ], # events of interest associated to the times 't'. There may be more than one event at each time, this is why we store them in a list.
                                                                            # (the events of interest are those affecting the
                                                                            # estimation of the blocking probability:
                                                                            # activation, absorption, block, unblock)
                                               't0': None,                  # Absorption /reactivation time
                                               'x': None,                   # Position after reactivation
                                               'particle number': P,        # Source particle (queue) from which reactivation happened (only used when reactivate=True)
                                               'particle ID': P,            # Particle ID associated to the particle BEFORE reactivation (this may not be needed but still we store it in case we need it --at first I thought it would have been needed during the finalize() process but then I overcome this need with the value t0 already stored
                                               'reactivated number': None,  # Reactivated particle (queue) to which the source particle was reactivated (only used when reactivate=True)
                                               'reactivated ID': None       # Reactivated particle ID (index of this info_particle list) 
                                               }) ]

            P_first_in_block = P_last_in_block + 1

        # Supplementary overall information about the particles
        # - whether each particle is active
        # - latest time the particles changed to an absorption state 
        if start_event_type == EventType.ABSORPTION:
            self.is_particle_active = [False]*self.N
            self.times0 = np.zeros(self.N, dtype=float)
        else:
            self.is_particle_active = [True]*self.N
            # Latest time the particles changed to an absorption state is unknown as they don't start at an absorption state
            self.times0 = np.nan*np.ones(self.N, dtype=float)
        ## Note: contrary to what I suspected, the elements of the is_particle_active array do NOT share the same memory address
        ## (verified by an example)  

        return nparticles_by_start_state
        
    def compute_equivalent_birth_rates(self):
        """
        Computes the birth rates for each server for the situation of pre-assigned jobs
        as is the case here, based on arrival rates and assignment probabilities
        """
        R = self.queue.getNServers()
        J = len(self.job_rates) # Number of job classes
        equivalent_birth_rates = [0]*R
        for r in range(R):
            for c in range(J):
                equivalent_birth_rates[r] += self.policy_assign[c][r] * self.job_rates[c]
        
        return equivalent_birth_rates

    #--------------------------------- Functions to simulate ----------------------------------
    def simulate(self, start_event_type :EventType):
        """
        Simulates the system of particles and estimates the blocking probability
        via Approximation 1 and Approximation 2 of Matthieu Jonckheere's draft on queue blocking dated Apr-2020.

        Arguments:
        start_event_type: EventType
            Type of event associated to the set of feasible start states.
            E.g. EventType.ABSORPTION defines the set of boundary absorption states, and
            EventType.ACTIVATION defines the set of activation states (i.e. at the boundary of the active set). 

        Return: tuple
        The content of the tuple depends on parameter 'reactivate'.
        When reactivate=True, the Fleming-Viot estimator is used to compute the blocking probability,
        therefore the tuple contains the following elements:
        - the estimated blocking probability
        - the estimated expected blocking time given the particle starts at position 1 and is not absorbed
          (this is the numerator of the expression giving the blocking probability estimate)
        - the integral appearing at the numerator of Approximation 1
        - the estimated gamma parameter: the absorption rate

        When reactivate=False, Monte-Carlo is used to compute the blocking probability,
        therefore the tuple contains the following elements:
        - the estimated blocking probability
        - the total blocking time over all particles
        - the total survival time over all particles
        - the number of times the survival time has been measured
        """
        self.reset()
        self.reset_positions(start_event_type, N_min=1)

        # (2021/02/10) Reset the random seed before the simulation starts,
        # in order to get the same results for the single-server case we obtained
        # BEFORE the implementation of a random start position for each particle
        # (a position which is actually NOT random in the case of single server system!)  
        np.random.seed(self.seed)

        time_start, time1, time2, time3 = self.run_simulation()

        if self.reactivate:
            if self.LOG:
                print("Estimating blocking probabilty via Approx. 1 & 2...")
            proba_blocking_integral, proba_blocking_laplacian, integral, gamma = self.estimate_proba_blocking_fv()
            self.proba_blocking_integral = proba_blocking_integral
            self.proba_blocking_laplacian = proba_blocking_laplacian
        else:
            proba_blocking, total_blocking_time, total_survival_time, total_survival_n = self.estimate_proba_blocking_mc()
        time_end = timer()

        if False:
            self.plot_trajectories_by_particle()

        if self.LOG:
            total_elapsed_time = (time_end - time_start)
            print("Total simulation time: {:.1f} min".format(total_elapsed_time / 60))
            print("Split as:")
            print("\tSimulation of initial trajectories until first absorption: {:.1f} min ({:.1f}%)".format((time1 - time_start) / 60, (time1 - time_start) / total_elapsed_time*100))
            print("\tSimulation of trajectories until end of simulation: {:.1f} min ({:.1f}%)".format((time2 - time1) / 60, (time2 - time1) / total_elapsed_time*100))
                ## THE ABOVE IS THE BOTTLENECK! (99.9% of the time goes here for reactivate=False, and 85%-98% for reactivate=True, see example below)
            print("\tCompute counts: {:.1f} min ({:.1f}%)".format((time3 - time2) / 60, (time3 - time2) / total_elapsed_time*100))
            print("\tEstimate probability (Approx. 1 and 2): {:.1f} min ({:.1f}%)".format((time_end - time3) / 60, (time_end - time3) / total_elapsed_time*100))
            ## Ex of simulation time from test case below:
            ## Test #2: Multiple-server system (3 servers)
            ## Running Monte-Carlo simulation on 1 particle and T=5000x...
            ## Total simulation time: 0.3 min
            ## Split as:
            ##    Simulation of initial trajectories until first absorption: 0.0 min (0.1%)
            ##    Simulation of trajectories until end of simulation: 0.3 min (99.9%)
            ##    Compute counts: 0.0 min (0.0%)
            ##    Estimate probability (Approx. 1 and 2): 0.0 min (0.0%)
            ##
            ## Running Fleming-Viot simulation on 100 particles and T=50x...
            ## Total simulation time: 0.4 min
            ## Split as:
            ##    Simulation of initial trajectories until first absorption: 0.0 min (12.2%)
            ##    Simulation of trajectories until end of simulation: 0.3 min (87.6%)
            ##    Compute counts: 0.0 min (0.0%)
            ##    Estimate probability (Approx. 1 and 2): 0.0 min (0.1%)

        if self.reactivate:
            return self.proba_blocking_integral, self.proba_blocking_laplacian, integral, self.mean_lifetime, gamma
        else:
            return proba_blocking, total_blocking_time, total_survival_time, total_survival_n 

    def simulate_fv(self):
        """
        Simulates the system of particles in the Fleming-Viot setup

        Return: tuple
        The tuple contains the following elements:
        - sbu: a list with the observed absolute blocking and unblocking times
        - counts_blocked: a list with the count of blocked particles at each time value in sbu.
        These pieces of information are needed to estimate Phi(t), i.e. the conditional blocking probability:
        Pr(K / T>t) 
        """
        if not self.reactivate:
            raise Warning("The reactivate parameter is False.\n" \
                          "Re-instatiate the Fleming-Viot object (typically EstimatorQueueBlockingFlemingViot) with the reactivate parameter set to True.")

        self.reset()
        start_event_type = EventType.ACTIVATION
        self.reset_positions(start_event_type, N_min=10)

        time_start, time1, time2, time_end = self.run_simulation()

        if self.LOG:
            total_elapsed_time = (time_end - time_start)
            print("Total simulation time: {:.1f} min".format(total_elapsed_time / 60))
            print("Split as:")
            print("\tSimulation of initial trajectories until first absorption: {:.1f} min ({:.1f}%)".format((time1 - time_start) / 60, (time1 - time_start) / total_elapsed_time*100))
            print("\tSimulation of trajectories until end of simulation: {:.1f} min ({:.1f}%)".format((time2 - time1) / 60, (time2 - time1) / total_elapsed_time*100))
                ## THE ABOVE IS THE BOTTLENECK! (99.9% of the time goes here for reactivate=False, and 85%-98% for reactivate=True, see example below)
            print("\tCompute counts: {:.1f} min ({:.1f}%)".format((time_end - time2) / 60, (time_end - time2) / total_elapsed_time*100))

        return self.sbu, self.counts_blocked

    def run_simulation(self, stop_at_first_absorption=False):
        time_start = timer()
        if self.LOG:
            print("simulate: Generating trajectories for each particle until first absorption...")
        self.generate_trajectories_at_startup()
        time1 = timer()
        # Check trajectories
        if False: #self.plotFlag:
            if False:
                for P in range(self.N):
                    self.plot_trajectories_by_server(P)
            self.plot_trajectories_by_particle()
            #input("Press ENTER...")

        if True: #self.LOG:
            print("Generating trajectories for each particle until END OF SIMULATION TIME (T={:.1f})...".format(self.maxtime))

        if stop_at_first_absorption:
            time2 = None
        else:
            self.generate_trajectories_until_end_of_simulation()
            time2 = timer()
            # Check trajectories
            if False: #self.N == 1:
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
    
        if True: #self.LOG:
            print("Finalizing and identifying measurement times...")
        self.finalize()
        self.compute_counts()
        time_end = timer()

        return time_start, time1, time2, time_end        

    def simulate_survival(self, N_min :int=1):
        """   
        Simulates the queue system to estimate the survival probability, Pr(T>t) given activation.

        Arguments:
        N_min: (opt) int
            Minimum number of particles to simulate in each set of start states.
            default: 1 

        Return: tuple
        The tuple contains the following elements:
        - sk: a list with the observed RELATIVE killing (absorption) times for ALL simulated particles P,
        which mark the times at which the survival curve changes value.
        - counts_alive: a list with the count of active particles at each time value in sk.
        These pieces of information are needed to estimate the survival probability Pr(T>t / activation-state).
        """
        self.reset()
        self.reset_positions(EventType.ACTIVATION, N_min=N_min)

        time_start, time1, _, time_end = self.run_simulation(stop_at_first_absorption=True)

        if self.LOG:
            total_elapsed_time = (time_end - time_start)
            print("Total simulation time: {:.1f} min".format(total_elapsed_time / 60))
            print("\tCompute counts: {:.1f} min ({:.1f}%)".format((time_end - time1) / 60, (time_end - time1) / total_elapsed_time*100))

        return self.sk, self.counts_alive, self.ktimes0_sum, self.ktimes0_n

    def simulate_return_time_to_absorption(self, N_min :int=1):
        """   
        Simulates the queue system to estimate the expected return time to absorption, E(T),
        given the initial state is at the boundary of the absorption set.

        Arguments:
        N_min: (opt) int
            Minimum number of particles to simulate in each set of start states.
            default: 1 

        Return: tuple
        The tuple contains the following elements:
        - ktimes0_sum: a list with the total observed RELATIVE killing times (times to absorption)
        for EACH simulated particle P.
        - ktimes0_n: a list with the number of observed RELATIVE killing times (times to absorption)
        for EACH simulated particle P.
        """
        self.reset()
        assert self.positions_observe == [ self.buffer_size_activation - 1 ], \
                "The observed positions are only ONE position and coincides with one less the buffer size for activation (BSA={}): {}" \
                .format(self.positions_observe, self.buffer_size_activation)
        self.reset_positions(EventType.ABSORPTION, N_min=N_min)

        time_start, time1, _, time_end = self.run_simulation(stop_at_first_absorption=False)

        if self.LOG:
            total_elapsed_time = (time_end - time_start)
            print("Total simulation time: {:.1f} min".format(total_elapsed_time / 60))
            print("\tCompute counts: {:.1f} min ({:.1f}%)".format((time_end - time1) / 60, (time_end - time1) / total_elapsed_time*100))

        return self.rtimes_obs_sum, self.rtimes_obs_n


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

            if False:
                # Log the completion of the simulation for the first few particles and then every 10% of particles 
                if P <= 2 or int( P % (self.N/10) ) == 0:
                    print("# particles processed so far: {}".format(P+1))

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
                self._update_return_time(P, time_next_event)
                self.assertSystemConsistency(P)

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
        # job class is increased; for DEATH events, the queue of each server is decreased.
        self._generate_birth_times(P)
        self._generate_death_times(P)
        
        # Find the next event times based on the current next birth and death times on all servers
        self._find_next_event_times(P)

    def _generate_birth_times(self, P):
        """
        Generates new arrival times for each job class, based on the job arrival rates,
        and assigns each job to one of the given particle's servers based on the assignment policy.
        
        The assignment policy is assumed to be a mapping from job class to server number.

        The effect of this method is the following:
        - The queue of jobs in each server is updated with the new assigned job (storing job class and event time).
        - The times of the next birth events by server is updated when the generated birth event is the first job
        assigned to the server's queue. Otherwise, the time of the next birth event in the sever is kept.
        """
        def are_all_next_birth_times_smaller_than_all_latest_generated_jobclass_times(P):
            resp = True
            smallest_time_last_jobclass = np.min(self.times_last_jobs[P])
            for server in range(self.nservers):
                if len(self.times_next_events_in_queue[P][server]) == 0 or \
                   self.times_next_events_in_queue[P][server][0] > smallest_time_last_jobclass:
                    resp = False
                    break
            return resp

        if False:
            print("P={}".format(P))
            print("times last jobs[P]: {}".format(self.times_last_jobs[P]))
            print("times next events in queue: {}".format(self.times_next_events_in_queue[P]))
        if not are_all_next_birth_times_smaller_than_all_latest_generated_jobclass_times(P):
            # In at least one of the server queues, the time of the "first event to be served" is LARGER than the
            # SMALLEST time already generated for the arriving job classes for this particle
            # This means that at least one of the times generated for the NEXT arriving job class
            # could be SMALLER than the time of the next event to be born in a server queue, making
            # that event to be positioned SECOND in the queue of the server assigned to the job class
            # (i.e. behind the newly generated time for the next arriving job class, assuming it is assigned
            # to that server, of course). And this implies that the next event to be born would be
            # born earlier than the event that is currently "first to be served" in the server queue. 
            # => Generate a birth time for ALL the job CLASSES and assign them to a server

            # Generate the next arriving job class times
            birth_times_relative_for_jobs = np.random.exponential( 1/np.array(self.job_rates) )
            # Make a copy of the absolute times of the last jobs (one value per job class),
            # prior to updating their values. This is done just for INFORMATIONAL purposes.
            times_last_jobs_prev_P = copy.deepcopy(self.times_last_jobs[P])
            # Update the absolute times of the last jobs (one value per job class)
            self.times_last_jobs[P] += birth_times_relative_for_jobs

            job_classes = range( len(self.job_rates) )
            if self.policy_accept is not None:
                # Use the acceptance policy to decide whether we accept each job IN THE ORDER THEY ARRIVE 
                times_last_jobs_P = copy.deepcopy(self.times_last_jobs[P])
                order = np.argsort(times_last_jobs_P)
                job_classes_accepted = []
                #print("P={}:\ntimes_last_jobs_P: {}".format(P, times_last_jobs_P))
                #print("order: {}".format(order))
                for job_class in [job_classes[o] for o in order]:
                    #print("job_class: {}".format(job_class))
                    #print("state: {}".format(self.policy_accept.env.getState()))
                    self.policy_accept.env.setJobClass(job_class)
                    action = self.policy_accept.choose_action()
                    if action == 1:
                        #print("ENTRO")
                        assert self.particles[P].getBufferSize() < self.particles[P].getCapacity(), \
                               "The buffer is NOT at full capacity when a job is accepted by the acceptance policy"
                        job_classes_accepted += [job_class]
                        next_state, reward, done, info = self.policy_accept.env.step(action)
            else:
                job_classes_accepted = job_classes

            # Assign the accepted job class associated to each birth time to one of the servers based on the assignment policy
            servers = range(self.nservers)
            for job_class in job_classes_accepted:
                # IMPORTANT: The assigned server can be repeated for different job classes
                # => we might have a queue of jobs for a particular server (which is handled here)
                assigned_server = np.random.choice(servers, p=self.policy_assign[job_class])
                assert assigned_server < self.nservers, \
                        "The assigned server ({}) is one of the possible servers [0, {}]".format(assigned_server, self.nservers-1)
    
                # Insert the new job time in order into the server's queue containing job birth times
                # Note that the order of the jobs is NOT guaranteed because the server's queue may contain
                # jobs of different classes, whose arrival times are generated independently from each other.
                idx_insort, _ = insort(self.times_next_events_in_queue[P][assigned_server], self.times_last_jobs[P][job_class], unique=False)
                self.jobclasses_next_events_in_queue[P][assigned_server].insert(idx_insort, job_class)
                if self.LOG:
                    print("job class: {}".format(job_class))
                    print("job time (PREVIOUS): {}".format(times_last_jobs_prev_P[job_class]))
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
        # Generate a death time for each server whose queue is empty and store them as ABSOLUTE values
        for server in range(self.nservers):
            if len(self.times_next_services[P][server]) == 0:
                # Generate the next death time for the server
                death_time_relative = self.particles[P].generate_event_time(Event.DEATH, server)
                self.times_last_services[P][server] += death_time_relative

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

    def _apply_next_event(self, P, server, type_of_event :Event, time_of_event :float):
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
        # TOFIX: (2021/03/25) Update the queue environment that keeps track of rewards, etc.
        # We should NOT have a separate environment containing a queue where we ALSO need to keep track of the buffer size...
        # (ALSO = "besides keeping track of the buffer size in the self.queue object")
        if self.policy_accept is not None:
            self.policy_accept.env.setBufferSize( self.particles[P].getBufferSize() )  

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

        # Update the latest time we know the state of the system
        self.times_latest_known_state[P] = time_of_event

        # Increase the iteration because the iteration is associated to the application of a new event in ONE server
        self.iterations[P] += 1

        # Update the position information of the particle so that we can plot the trajectories
        # and we can compute the statistics for the estimations that are based on the occurrence
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
        #buffer_old_size = self.positions_buffer[P]
        #buffer_size_change = buffer_new_size - buffer_old_size
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
        # The second parameter is the change in BUFFER size for the particle, NOT the array of changes in server queue sizes
        self._update_info_particles_and_blocking_statistics(P)

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

    def _update_return_time(self, P, time_of_event):
        """
        Updates the return times information of the particle based on the time of event
        and the current position of the particle after the latest event was applied.
        
        The return times info is updated if the latest change in (buffer) position of the particle
        implies that it NOW touches one of the positions to observe.   
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)

        if self._has_particle_returned_to_positions(P, self.positions_observe):
            if self.rtimes_obs_n[P] == 0 and self.getStartPosition(P) not in self.positions_observe:
                # Offset that should be substracted for every new return time added
                # to the total of ALL observed return times
                # because the particle did NOT start at any of those positions to observe.
                self.rtimes_offset[P] = time_of_event
            else:
                # Add the newly observed return time to the total of all observed return times to one of the positions to observe
                #print("P={}: Return absolute time at t={:.1f} --> recorded relative return time: {:.1f}".format(P, time_of_event, time_of_event - self.rtimes_offset[P] - self.rtimes_obs_sum[P]))
                self.rtimes_obs_sum[P] = time_of_event - self.rtimes_offset[P]
                self.rtimes_obs_n[P] += 1

    def generate_trajectories_until_end_of_simulation(self):
        "Generates the trajectories until the end of the simulation time is reached"
        step_fraction_process_to_report = 0.1
        next_fraction_process_to_report = step_fraction_process_to_report
        if self.LOG:
            print("...completed {:.1f}%".format(self.get_time_latest_known_state() / self.maxtime * 100))
        while len(self.dict_info_absorption_times['t']) > 0:
            # There are still particles that have been absorbed and thus their simulation stopped
            # => We need to generate the rest of the trajectory for each of those particles
            # and how the trajectory is generated depends on whether we are in reactivate mode
            # (Fleming-Viot) or not (Monte-Carlo). This is why we "stop" at every absorption event
            # and analyze how to proceed based on the simulation mode (reactivate or not reactivate).
            fraction_maxtime_completed = self.get_time_latest_known_state() / self.maxtime
            if fraction_maxtime_completed > next_fraction_process_to_report:
                if self.LOG:
                    print("...completed {:.0f}%".format(next_fraction_process_to_report*100))
                next_fraction_process_to_report += step_fraction_process_to_report
            if False:
                assert self.dict_info_absorption_times['t'] == sorted(self.dict_info_absorption_times['t']), \
                    "The absorption times are sorted: {}".format(self.dict_info_absorption_times['t'])
            if self.LOG:
                with printoptions(precision=3, suppress=True):
                    print("\n******** REST OF SIMULATION STARTS (#absorption times left: {}, MAXTIME={:.1f}) **********".format(len(self.dict_info_absorption_times['t']), self.maxtime))
                    print("Absorption times and particle numbers (from which the first element is selected):")
                    print(self.dict_info_absorption_times['t'])
                    print(self.dict_info_absorption_times['P'])
            # Remove the first element of the dictionary which corresponds to the particle
            # with the smallest absorption time
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
                self._update_info_particles_and_blocking_statistics(P)
            else:
                # In NO reactivation mode, we need to generate ONE iteration before CONTINUING with the simulation below,
                # otherwise the assertion that the next absorption time is different from the previous one
                # in _update_killing_time() the next time this function is called, fails.
                end_of_simulation = self.generate_one_iteration(P)

            # Continue generating the trajectory of the particle after the latest absorption/reactivation
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

        # Particle IDs in the info_particles list associated to particle numbers P and Q
        # to which the absorbed particle P is reactivated.
        p = self.particle_reactivation_ids[P]
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
                                       'particle ID': p, 
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
                #print(np.c_[np.array(self.info_particles[q]['t']), np.array(self.info_particles[q]['E'])])

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
        assert self.buffer_size_activation <= np.sum(position) <= self.particles[P].getCapacity(), \
                "The buffer size of the reactivated particle {} ({}) is in the set of non-absorption states ([{}, {}])" \
                .format(P, position, self.buffer_size_activation, self.particles[P].getCapacity())
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

    def _update_info_particles_and_blocking_statistics(self, P):
        """
        Updates the info_particles attribute with the new particle's position
        and any special state that the particle goes into as a result of the last position change,
        such as ACTIVATION, ABSORPTION, BLOCK, UNBLOCK, START_POSITION.
        
        Note that the particle can have more than one event type to store.  
        """
        assert self.isValidParticle(P), "The particle to update is valid (0<=P<{}) ({})".format(self.N, P)

        event_types = self.identify_event_types(P)
        assert isinstance(event_types, list)
        if event_types != []:
            if self.LOG:
                print("P={}: SPECIAL EVENTS OBSERVED: {}".format(P, event_types))
                print("\tPREVIOUS position of P (buffer size): {}".format(self._get_previous_buffer_size(P)))
                print("\tposition of P (buffer size): {}".format(self.particles[P].getBufferSize()))
            p = self.particle_reactivation_ids[P]
            assert p < len(self.info_particles), \
                    "The particle ID (p={}) exists in the info_particles list (0 <= p < {})" \
                    .format(p, len(self.info_particles))
            self.info_particles[p]['t'] += [ self.times_buffer[P] ]
            self.info_particles[p]['E'] += [ event_types ]

            # Update the blocking times if any blocking/unblocking event took place 
            self._update_blocking_statistics(P, event_types)

    def _update_blocking_statistics(self, P, event_types :list):
        "Updates the blocking time statistics whenever the particle is blocked or unblocked"
        if list_contains_either(event_types, EventType.BLOCK):
            blocking_time = self.particles[P].getMostRecentEventTime()
            #print("P={}: (B) info_particles: {}".format(P, self.info_particles[P]))
            self._update_latest_blocking_time(P, blocking_time)
        elif list_contains_either(event_types, EventType.UNBLOCK):
            unblocking_time = self.particles[P].getMostRecentEventTime()
            #print("P={}: (U) info_particles: {}".format(P, self.info_particles[P]))
            self._update_blocking_time(P, unblocking_time)

    def _generate_trajectory_until_absorption_or_end_of_simulation(self, P, end_of_simulation):
        """
        Generates the trajectory for the given particle until it is absorbed or the max simulation time is reached.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to simulate.

        end_of_simulation: int
            Whether the end of the simulation has been reached.
        """
        absorbed = lambda P: self._check_particle_state_based_on_most_recent_change(P, EventType.ABSORPTION)

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
            absorption_time = self.particles[P].getMostRecentEventTime()
            self._update_info_absorption_times(P, absorption_time, self.iterations[P])
            self._update_killing_time(P, absorption_time)

    def _compute_order(self, arr, axis=-1):
        "Returns the order of the values in an array by the specified axis dimension"
        # axis=-1 means "the last dimension of the array"
        return np.argsort(arr, axis=axis)

    def _get_previous_buffer_size(self, P):
        """
        Returns the position (buffer size) before the latest change taking place in the particle.
        
        The previous buffer size is equal to the current buffer size if the latest change was 0.
        (which happens for instance when a server at 0 experiences a DEATH event or a server at
        maximum capacity experiences a BIRTH event).

        If the type of the latest event is a RESIZE, the previous buffer size is computed from
        the latest size change in ALL servers (as opposed to the latest size change which under normal
        conditions takes place in just ONE server) because the server sizes change all at once
        (i.e. at the same time) in a RESIZE situation. 
        """
        # We find the previous buffer size based on the latest change
        last_change_all_servers, server_with_last_event, type_last_event, _ = self.particles[P].getMostRecentEventInfo()

        if type_last_event == Event.RESIZE:
            # The previous server sizes can be computed directly from `last_change_all_servers`
            # because on a RESIZE all servers change size at the same time.
            previous_server_sizes = self.particles[P].getServerSizes() - last_change_all_servers
            previous_buffer_size = np.sum(previous_server_sizes)
        elif last_change_all_servers[server_with_last_event] != 0:
            # In a non-RESIZE situation (i.e. a "normal" situation),
            # we need to analyze the change in the server that changed last. 
            # => Compute the previous server sizes based on the last event type
            previous_server_sizes = self.particles[P].getServerSizes()
            if type_last_event == Event.BIRTH:
                previous_server_sizes[server_with_last_event] -= 1 
            elif type_last_event == Event.DEATH:
                previous_server_sizes[server_with_last_event] += 1
            previous_buffer_size = np.sum(previous_server_sizes)
        else:
            # There was no change in buffer size if the latest change in the system is 0
            # (this may happen if the latest event was a DEATH happening in a server at state 0
            # or a BIRTH happening in a server at state equal to its capacity) 
            previous_buffer_size = self.particles[P].getBufferSize()
    
        return previous_buffer_size

    def identify_event_types(self, P): 
        "Identifies ALL the special event types a particle is now"
        #--------------- Helper functions -------------------
        # Possible status change of a particle 
        is_activated = lambda P: self._has_particle_become_activated(P)
        is_absorbed = lambda P: self._has_particle_become_absorbed(P)
        is_blocked = lambda P: self._has_particle_become_blocked(P)
        is_unblocked = lambda p: self._has_particle_become_unblocked(P)
        is_start_position = lambda p: self._has_particle_returned_to_positions(P, [ self.getStartPosition(P) ])
        #--------------- Helper functions -------------------

        # Special event types (relevant for the blocking probability estimation)
        event_types = []
        if is_activated(P):
            event_types += [EventType.ACTIVATION]
        if is_absorbed(P):
            event_types += [EventType.ABSORPTION]
        if is_blocked(P):
            event_types += [EventType.BLOCK]
        if is_unblocked(P):
            event_types += [EventType.UNBLOCK]
        if is_start_position(P):
            event_types += [EventType.START_POSITION]

        return event_types

    def _update_active_particles(self, P):
        "Updates the flag that states whether the given particle number P is active"
        self.is_particle_active[P] = not self._is_particle_absorbed(P)

    def _is_particle_activated(self, P):
        "Whether the particle is activated (regardless of the latest change experienced by it)"
        return self.particles[P].getBufferSize() in self.set_activation

    def _is_particle_absorbed(self, P):
        "Whether the particle is absorbed (regardless of the latest change experienced by it)"
        return self.particles[P].getBufferSize() in self.set_absorption

    def _is_particle_blocked(self, P):
        "Whether the particle is blocked (regardless of the latest change experienced by it)"
        return self.particles[P].getBufferSize() == self.particles[P].getCapacity() 

    def _is_particle_unblocked(self, P):
        "Whether the particle is unblocked (regardless of the latest change experienced by it)"
        return self.particles[P].getBufferSize() < self.particles[P].getCapacity() 

    def _has_particle_become_activated(self, P):
        """
        Whether the particle is activated, which is indicated by the fact that
        before the latest change, the particle's position (buffer size) is NOT in the set of activation buffer sizes
        and now is one of the activation buffer sizes.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)

        previous_buffer_size = self._get_previous_buffer_size(P)

        #return last_position_change != 0 and self._is_particle_activated(P)
        return previous_buffer_size not in self.set_activation and self._is_particle_activated(P)

    def _has_particle_become_absorbed(self, P):
        """
        Whether the particle is absorbed, which is indicated by the fact that
        before the latest change, the particle's position (buffer size) is NOT in the set of absorbed buffer sizes
        and now is one of the absorbed buffer sizes.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)

        previous_buffer_size = self._get_previous_buffer_size(P)

        #return last_position_change < 0 and self._is_particle_absorbed(P)
        return previous_buffer_size not in self.set_absorption and self._is_particle_absorbed(P)

    def _has_particle_become_blocked(self, P):
        """
        Whether the particle becomes blocked, which is indicated by a positive change in the system's buffer size
        and a system's buffer size equal to its capacity after the change.
        
        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)

        previous_buffer_size = self._get_previous_buffer_size(P)
        
        #return last_position_change > 0 and self._is_particle_blocked(P)       
        return previous_buffer_size < self.particles[P].getCapacity() and self._is_particle_blocked(P)

    def _has_particle_become_unblocked(self, P):
        """
        Whether the particle becomes unblocked, which is indicated by a negative change in the system's buffer size
        and a system's buffer size equal to its capacity before the change.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)

        previous_buffer_size = self._get_previous_buffer_size(P)

        #return last_position_change < 0 and self.particles[P].getBufferSize() == self.particles[P].getCapacity() + last_position_change
        return previous_buffer_size == self.particles[P].getCapacity() and self._is_particle_unblocked(P)

    def _has_particle_returned_to_positions(self, P, positions :list):
        """
        Whether the particle has returned to any of the given positions (buffer sizes)

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.

        positions: list
            List of positions to check.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        assert isinstance(positions, list)

        previous_position = self._get_previous_buffer_size(P)

        return previous_position not in positions and self.particles[P].getBufferSize() in positions
        
    def _check_particle_state_based_on_most_recent_change(self, P, event_type):
        """
        Whether a particle has achieved a NEW SPECIAL state (state of interest) defined by the event_type
        based on the most recent change in the particle's servers.
        
        This function CANNOT be used when the latest event times for all servers is the same,
        notably at the very beginning of the simulation or when a reactivation has occurred
        (in which case, all the server queue sizes change at the same time, namely the absorption time
        leading to reactivation).
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        if event_type == EventType.ABSORPTION:
            return self._has_particle_become_absorbed(P)
        elif event_type == EventType.BLOCK:
            return self._has_particle_become_blocked(P)
        elif event_type == EventType.UNBLOCK:
            return self._has_particle_become_unblocked(P)

    def _update_info_absorption_times(self, P, time_of_absorption, it):
        "Inserts a new absorption time in order, together with its particle number"
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        idx_insort, found = insort(self.dict_info_absorption_times['t'], time_of_absorption, unique=False)
        assert not found, "The inserted absorption time ({:.3f}) does NOT exist in the list of absorption times:\n{}".format(time_of_absorption, self.dict_info_absorption_times['t'])
        self.dict_info_absorption_times['P'].insert(idx_insort, P)

    def _update_killing_time(self, P, time_of_absorption):
        ref_time = 0.0 if np.isnan(self.times0[P]) else self.times0[P]

        # Refer the survival time to the latest time the particle was absorbed
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
                print(">>>> Previous absorption time: {:.3f}".format(self.times0[P]))
                print(">>>> Total Survival times for ALL particles: {}".format(np.array(self.ktimes0_sum)))
                print(">>>> Total Survival units for ALL particles: {}".format(np.array(self.ktimes0_n)))

        # Update the latest time the particle was absorbed
        self.times0[P] = time_of_absorption

        if False:
            print(">>>> P={}: UPDATED Last time at 0: {:.3f}".format(P, self.times0[P]))
            print(">>>> \tNew total killing times (ktimes0_sum): {:.3f}".format(self.ktimes0_sum[P]))

    def _update_latest_blocking_time(self, P, time_of_blocking):
        "Updates the latest time the particle was blocked"
        self.timesb[P] = time_of_blocking

        if False:
            print(">>>> P={}: UPDATED Last time particle was blocked: {:.3f}".format(P, self.timesb[P]))

    def _update_blocking_time(self, P, time_of_unblocking):
        "Updates the total blocking time and the number of blocking times measured based on an unblocking event time"
        assert not np.isnan(self.timesb[P]), "The last blocking time is not NaN ({:.3f})".format(self.timesb[P])
        last_blocking_time = self.timesb[P]

        # Refer the unblocking time to the latest time the particle was blocked
        assert time_of_unblocking > last_blocking_time, \
                "The time of unblocking ({}) contributing to the total blocking time calculation" \
                " is larger than the latest blocking time for particle {} ({})" \
                .format(time_of_unblocking, P, last_blocking_time)
        self.btimes_sum[P] += time_of_unblocking - last_blocking_time
        self.btimes_n[P] += 1

        if False:
            with printoptions(precision=3, suppress=True):
                print("\n>>>> Particle P={}: unblocking @t={:.3f}".format(P, time_of_unblocking))
                print(">>>> Previous blocking times: {:.3f}".format(self.timesb[P]))
                print(">>>> \tNew total blocking time (btimes_sum): {:.3f}".format(self.btimes_sum[P]))

    def finalize(self):
        """
        Finalizes the simulation process by treating particles that are active,
        i.e. those particles providing CENSORED survival time values.
        
        How finalization is carried out and what particles are to be finalized
        is defined by the object's attribute `finalize_info`, which defines the 'type'
        of finalization and the 'condition' (on the last state of the particle) for finalization.
        """
        particle_numbers_to_finalize = self.get_censored_particle_numbers()
        particle_ids_to_finalize = list( np.sort([ self.particle_reactivation_ids[P] for P in particle_numbers_to_finalize ]) )

        finalize_process = "--NO FINALIZATION DONE--"
        if True: #self.LOG:
            if self.finalize_info['type'] == FinalizeType.ABSORB_CENSORED:
                finalize_process = "ABSORBING"
            elif self.finalize_info['type'] == FinalizeType.REMOVE_CENSORED:
                finalize_process = "REMOVING"
            elif self.finalize_info['type'] == FinalizeType.ESTIMATE_CENSORED:
                finalize_process = "CENSORING"
            #print("\n****** FINALIZING the simulation by {} particles whose final state satisfies the condition {} ******".format(finalize_process, self.finalize_info['condition'].name))
            #print("Particles to be finalized: {}" \
            #      .format(", ".join(["(p={}, P={})".format(p, self.info_particles[p]['particle number']) for p in particle_ids_to_finalize])))

        # Make a copy of the list because this may be updated by removing elements
        if self.finalize_info['type'] == FinalizeType.REMOVE_CENSORED:
            assert sorted(particle_ids_to_finalize) == particle_ids_to_finalize, \
                    "The list of particle IDs to finalize is sorted," \
                    " which is crucial for the correct removal of censored particles ({})" \
                    .format(particle_ids_to_finalize)
        nremoved = 0
        for p in particle_ids_to_finalize:
            dict_info = self.info_particles[p - nremoved]
            P = dict_info['particle number']
            self.assertAllServersHaveSameTimeInTrajectory(P)
            #print("Processing particle ID p={}, P={} out of {} particles".format(p, P, len(particle_ids_to_finalize)))
            if self.finalize_info['type'] == FinalizeType.REMOVE_CENSORED:
                if self.reactivate:
                    # Simply remove the particle from the analysis
                    if self.LOG: #True:
                        print("...{} particle p={} (P={})".format(finalize_process, p, P))
                    # But first get the time of absorption leading to the reactivation of the
                    # particle ID to remove so that we set the latest time with info about P below
                    time_latest_absorption_P = self.info_particles[p - nremoved]['t0']
                    self.info_particles.pop(p - nremoved)
                    nremoved += 1
                        # NOTE that no update needs to be done to the quantities
                        # used to estimate E(T) (e.g. ktimes0_sum) because these quantities are updated
                        # ONLY when there is an absorption, and this never happened
                        # for this particle ID (recall that, when reactivate=True,
                        # there is at most ONE absorption per particle ID,
                        # and when it occurs, it's the last event to occur).

                    # Update the latest time of known state for particle P to the absorption time
                    # leading to the particle ID p just removed
                    if time_latest_absorption_P is not None:
                        self.times_latest_known_state[P] = time_latest_absorption_P
                    else:
                        self.times_latest_known_state[P] = 0.0
                    if self.LOG:
                        print("REMOVAL: Latest absorption time set for particle P={}: {:.3f}".format(P, self.times_latest_known_state[P]))
                else:
                    # Find the latest valid event time and remove all the events since then
                    if self.LOG:
                        print("...{} the tail of particle P={}".format(finalize_process, P))
                    if self.finalize_info['condition'] == FinalizeCondition.ACTIVE:
                        idx_last_valid_event = find_last_value_in_list(dict_info['E'], EventType.ABSORPTION)
                    elif self.finalize_info['condition'] == FinalizeCondition.NOT_START_POSITION:
                        idx_last_valid_event = find_last_value_in_list(dict_info['E'], EventType.START_POSITION)

                    # Remove all the elements AFTER the index of the last absorption to the end
                    # Note that if no valid last event is found, ALL the trajectory is removed
                    # and this works fine like this because find_last() returns -1 when the searched value
                    # is not found in the list.
                    event_time_block = None
                    if self.N == 1:
                        print("P={}: Blocking time BEFORE removal: t={:.3f}, n={}".format(P, self.btimes_sum[P], self.btimes_n[P]))
                        #print(dict_info)
                    while len(dict_info['E']) > idx_last_valid_event + 1:
                        # Note that we remove the events belonging to the subtrajectory to remove
                        # from the EARLIEST to the LATEST event
                        # This has an impact in how we reduce the blocking time of possible blocking events
                        # in the subtrajectory being removed (i.e. we store "the event_time_block when a BLOCK event is found"
                        # as opposed to storing "the event_time_unblock when an UNBLOCK event is found",
                        # because when going from left to right in time, we first encounter a BLOCK event and THEN an UNBLOCK event. 
                        event_time = dict_info['t'].pop(idx_last_valid_event+1)
                        event_types = dict_info['E'].pop(idx_last_valid_event+1)
                        if self.N == 1:
                            print("P={}: {} events at time {:.1f} removed.".format(P, event_types, event_time))
                        if list_contains_either(event_types, EventType.BLOCK):
                            event_time_block = event_time
                        elif list_contains_either(event_types, EventType.UNBLOCK) and event_time_block is not None:
                            # Decrease the total blocking time for particle
                            self.btimes_sum[P] -= event_time - event_time_block
                            self.btimes_n[P] -= 1
                            event_time_block = None
                        # NOTE that we should NOT update the killing times information
                        # (ktimes0_sum and ktimes0_n) because:
                        # - either the last valid event is an ABSORPTION, so there is no absorption event
                        # in the subtrajectory being removed.
                        # - either the possible ABSORPTION event in the subtrajectory being removed
                        # can be considered as a valid contribution to the killing time, because it was actually observed.
                        # Although technically the contribution of such event should be removed from the
                        # killing time value because we are REMOVING the subtrajectory containing
                        # such absorption event, but unfortunately, rolling back such contribution
                        # from the killing times is NOT SO EASY because the contribution by a single absorption event
                        # maybe n-fold, i.e. one contribution for EVERY ACTIVATION event that happened
                        # prior to the previous ABSORPTION event... So, in order to properly remove the contribution
                        # from the removed absorption event, we would need to go back
                        # and look for all ACTIVATION events happening before it, but after the previous ABSORPTION event!
                    if self.N == 1:
                        print("P={}: Blocking time AFTER removal: t={:.3f}, n={}".format(P, self.btimes_sum[P], self.btimes_n[P]))
                        #print(dict_info)
                    """
                    DM-2021/04/29: We should NOT add a fictitious ABSORPTION when the finalize condition is not ACTIVE
                    (e.g. when the finalize condition is "particle did not return to the start position").
                    In fact, we are under the REMOVE censored observations, so, we should just remove
                    NOT add something!)
                    What we need to do, actually, is to modify the assertions done below in compute_counts()
                    in order to consider also the case that the last observation is NOT an ABSORPTION but
                    a START_POSITION (in the case the finalize condition is that the particle did not return to the start position) 
                    I am doing this change in compute_counts() RIGHT NOW!
                    if self.finalize_info['condition'] != FinalizeCondition.ACTIVE:
                        # Add a fictitious absorption event (ABSORPTION_F) so that assertions in compute_counts() pass
                        # NOTES:
                        # - We do this ONLY when the finalize condition is NOT active because
                        # in the ACTIVE finalize condition, the last event returned is ALWAYS an ABSORPTION
                        # (by definition of ACTIVE).
                        # - Adding a fictitious absorption does NOT introduce a distortion in the estimation
                        # of e.g. the survival probability because fictitious absorption events are NOT considered
                        # for estimation.
                        time_to_insert = dict_info['t'][idx_last_valid_event] + self.EPSILON_TIME*np.random.random()
                        dict_info['t'] += [time_to_insert]
                        dict_info['E'] += [ [EventType.ABSORPTION_F] ]
                    """
                    # Update the latest time of known state for particle P to the valid event time
                    # leading to the particle ID p just removed
                    if len(dict_info['E']) > 0:
                        assert  self.finalize_info['condition'] == FinalizeCondition.ACTIVE and list_contains_either(dict_info['E'][-1], EventType.ABSORPTION) or \
                                self.finalize_info['condition'] == FinalizeCondition.NOT_START_POSITION and list_contains_either(dict_info['E'][-1], EventType.START_POSITION), \
                                "P={}: The last event stored in info_particles after removal of the censored part of the trajectory " \
                                "is an ABSORPTION for finalize condition=ACTIVE, or START_POSITION for finalize condition=NOT_START_POSITION ({})" \
                                .format(P, dict_info['E'][-1])                             
                        self.times_latest_known_state[P] = dict_info['t'][-1]
                    else:
                        self.times_latest_known_state[P] = 0.0
                    if self.LOG:
                        print("REMOVAL: Latest absorption time set for particle P={}: {:.3f}".format(P, self.times_latest_known_state[P]))

                if self.finalize_info['condition'] == FinalizeCondition.ACTIVE:                        
                    # Flag the particle as inactive (as the latest censored trajectory was removed)
                    self.is_particle_active[P] = False

            elif self.finalize_info['type'] in [FinalizeType.ABSORB_CENSORED, FinalizeType.ESTIMATE_CENSORED]:
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
                if self.finalize_info['type'] == FinalizeType.ABSORB_CENSORED:
                    # Add fictitious absorption times for each particle
                    # NOTE that we do NOT update the attributes that are used to estimate
                    # the expected survival time (self.ktimes0_sum and self.ktimes0_n) so that
                    # we know when an unreliable (and most likely UNDER-) estimate of the
                    # expected survival time is carried out (i.e. when self.ktimes0_n is small,
                    # say much less than the number of particles, it implies that most particles
                    # were bound to live longer than the simulation time frame, but this
                    # actual life time could NOT be measured).

                    # To avoid assertion failures... make sure that, at the end of the simulation:
                    # - the particle is unblocked if it was blocked
                    #   (adding this preserves the assertion that the number of
                    #   blocked particles at the very last time is 0 --since the particles are all absorbed!)
                    # - the particle is activated if it was not in position 1
                    #   (this makes things logical --i.e. before absorption the particle passes by position 1)
                    if self._is_particle_blocked(P):
                        # Add a fictitious UNBLOCK time if the particle was BLOCKED
                        dict_info['t'] += [time_to_insert]
                        dict_info['E'] += [ [EventType.UNBLOCK_F] ]
                        # Increase each of the times to insert for the insertion of the next event
                        time_to_insert += self.EPSILON_TIME*np.random.random()
                        # Update the total blocking time
                        self._update_blocking_time(P, time_to_insert)
                    if not self._is_particle_activated(P):
                        # Add a fictitious ACTIVATION time if the buffer size was not already in the ACTIVATION set
                        dict_info['t'] += [time_to_insert]
                        dict_info['E'] += [ [EventType.ACTIVATION_F] ]
                        # Increase each of the time to insert for the insertion of the
                        # fictitious absorption event coming next (to avoid time value repetitions)
                        time_to_insert += self.EPSILON_TIME*np.random.random()
                    
                    # Finally add the fictitious ABSORPTION time
                    dict_info['t'] += [time_to_insert]
                    dict_info['E'] += [ [EventType.ABSORPTION_F] ]

                    if self.finalize_info['condition'] == FinalizeCondition.ACTIVE:                        
                        # Flag the particle as inactive (as it was just absorbed)
                        self.is_particle_active[P] = False
                elif self.finalize_info['type'] == FinalizeType.ESTIMATE_CENSORED:
                    dict_info['t'] += [time_to_insert]
                    dict_info['E'] += [ [EventType.CENSORING] ]

                # Survival time from s=0
                self._update_killing_time(P, time_to_insert)

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

        # Initialize the lists that will store information about the time values
        # at which the probability functions (survival and blocking) change their value.
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
            
            # A FEW ASSERTIONS ABOUT THE FINALIZATION PROCESS BEING CARRIED OUT CORRECTLY
            if len(dict_info['E']) > 0:
                if self.finalize_info['condition'] == FinalizeCondition.ACTIVE:
                    if self.finalize_info['type'] == FinalizeType.REMOVE_CENSORED:
                        assert not list_contains_either(dict_info['E'][-1], [EventType.ACTIVATION, EventType.ACTIVATION_F]) or \
                                "The finalize directives are: (REMOVE_CENSORED, ACTIVE) and the last event does NOT contain an ACTIVATION or ACTIVATION_F event" \
                                "(P={}, p={}, Events={})" \
                                .format(P, p, dict_info['E'])
                        ## (2021/04/24) NOTE that an ACTIVATION event at the END of the trajectory
                        ##  does NOT affect the correct estimation of the survival probability P(T>t)
                        ## since only absorption events contribute to it, thus an activation event
                        ## at the end will not contribute to P(T>t).
                    elif self.finalize_info['type'] == FinalizeType.ABSORB_CENSORED:
                        assert list_contains_either(dict_info['E'][-1], [EventType.ABSORPTION, EventType.ABSORPTION_F]) or \
                                "The finalize directives are: (ABSORB_CENSORED, ACTIVE) and the last event contains an ABSORPTION or ABSORPTION_F event" \
                                "(P={}, p={}, Events={})" \
                                .format(P, p, dict_info['E'])
                    elif self.finalize_info['type'] == FinalizeType.ESTIMATE_CENSORED:
                        assert list_contains_either(dict_info['E'][-1], EventType.CENSORING) or \
                                "The finalize directives are: (ESTIMATE_CENSORED, ACTIVE) and the last event contains an CENSORING event" \
                                "(P={}, p={}, Events={})" \
                                .format(P, p, dict_info['E'])
                elif self.finalize_info['condition'] == FinalizeCondition.NOT_START_POSITION:
                    assert self.finalize_info['type'] != FinalizeType.ABSORB_CENSORED, \
                            "When the finalize condition is NOT_START_POSITION, the finalize type is NOT ABSORB_CENSORED"
                    if self.finalize_info['type'] == FinalizeType.REMOVE_CENSORED:
                        assert list_contains_either(dict_info['E'][-1], EventType.START_POSITION) or \
                                "The finalize directives are: (ESTIMATE_CENSORED, NOT_START_POSITION) and the last event contains a START_POSITION event" \
                                "(P={}, p={}, Events={})" \
                                .format(P, p, dict_info['E'])
                    elif self.finalize_info['type'] == FinalizeType.ESTIMATE_CENSORED:
                        assert list_contains_either(dict_info['E'][-1], EventType.CENSORING) or \
                                "The finalize directives are: (ESTIMATE_CENSORED, NOT_START_POSITION) and the last event contains an CENSORING event" \
                                "(P={}, p={}, Events={})" \
                                .format(P, p, dict_info['E'])

            # List of observed activation times for the particle
            # Once the particle is absorbed, a survival time for each activation time
            # is added to the list of survival times given activation (that are used to estimate Pr(T>t / s=1))
            # The list of activation times is then reset to empty, once they are used
            # following an ABSORPTION event for the particle.
            activation_times = []
            event_types_prev = []
            for t, event_types in zip(dict_info['t'], dict_info['E']):
                if list_contains_either(event_types, EventType.ACTIVATION):
                    ## NOTE: (2021/02/18) Fictitious ACTIVATION times (ACTIVATION_F) are NOT added to the list of activation times
                    ## because these should NOT be used to compute the absorption times that are used to
                    ## compute Pr(T>t / s=1). In fact, if they were used, the absorption times that are added
                    ## to self.sk would be too small (< EPSILON_TIME, because a fictitious activation time
                    ## is followed by a fictitious absorption time coming just EPSILON_TIME afterwards), and
                    ## they would UNDERESTIMATE Pr(T>t / s=1), thus generating an underestimated blocking probability
                    ## by Fleming-Viot's Approximation 1 approach.
                    assert not list_contains_either(event_types_prev, EventType.BLOCK), \
                            "The event before an ACTIVATION cannot be a BLOCK event, as the particle first needs to be UNBLOCKED:\n{}" \
                            .format(dict_info)
                    assert not list_contains_either(event_types_prev, EventType.CENSORING), \
                            "There must be NO event after a CENSORING observation ({})".format(event_types)

                    if self.reactivate:
                        assert not list_contains_either(event_types_prev, EventType.ABSORPTION), \
                                "There must be NO event after an absorption because the particle is absorbed, i.e. it dies (p={}, P={}, {})" \
                                .format(p, P, event_types)
                    activation_times += [t]
                elif list_contains_either(event_types, EventType.ABSORPTION) and event_types_prev != []:    # (2021/04/24) The `event_types_prev != []` is added as a condition to take care of the special first absorption event that happens when the particle starts at an ABSORPTION state. 
                    ## NOTE: (2021/02/18) Fictitious ABSORPTION times (ABSORPTION_F) are NOT considered because they may
                    ## underestimate the return time to absorption, making Pr(T>t / s=1) be smaller than it should.                        
                    assert list_contains_either(event_types_prev, EventType.ACTIVATION), \
                            "The previous event of an ABSORPTION is always an ACTIVATION (event_types_prev={}, p={}, P={}, t={}))" \
                            .format([e.name for e in event_types_prev], p, P, t)
                    if len(activation_times) > 0:
                        # NOTE: (2021/02/18) The list of past activation times may be empty when:
                        # - reactivate=True
                        # - the particle is reactivated to a position that is NOT in the set of activation states
                        # and it does NOT touch any event in the activation set before the end of the simulation.
                        self.insert_relative_time_kill(t, activation_times, event_types)
                        # The list of activation times is reset to empty after an absorption
                        # because all the existing activation times have been used to populate self.sk
                        # in the above call to insert_relative_time_kill().
                        activation_times = []              
                    if False:
                        # We may want to disable this assertion if they take too long
                        assert sorted(self.sk) == list(np.unique(self.sk)), \
                                "The list of survival time segments contains unique values" \
                                " after insertion of event {} for particle p={}, P={}" \
                                .format([e.name for e in event_types], p, P)
                elif list_contains_either(event_types, [EventType.BLOCK, EventType.UNBLOCK]):
                    # We insert the ABSOLUTE time for BLOCK and UNBLOCK events
                    # since this is used to compute the empirical distribution at the absolute time t
                    # NOTE: (2021/02/18) We do NOT insert fictitious UNBLOCK times (UNBLOCK_F)
                    # (added at the end of the simulation to avoid assertion failures)
                    # because we do not want to distort the estimation of blocking time with a too short
                    # duration happening just because the simulation ended and the particle was blocked at that time...) 
                    self.insert_absolute_time_block_unblock(t, event_types)
                    if list_contains_either(event_types, EventType.UNBLOCK):
                        assert list_contains_either(event_types_prev, EventType.BLOCK), \
                                "The event coming before an UNBLOCK event is a BLOCK event ({})" \
                                .format([e.name for e in event_types_prev])
                    if False:
                        # We may want to disable this assertion if they take too long
                        assert sorted(self.sbu) == list(np.unique(self.sbu)), \
                                "The list of block/unblock time segments contains unique values" \
                                " after insertion of event {} for particle p={}, P={}" \
                                .format([e.name for e in event_types], p, P)
                event_types_prev = event_types

        assert len(self.counts_alive) == len(self.sk), \
                "The length of counts_alive ({}) is the same as the length of self.sk ({})" \
                .format(len(self.counts_alive), len(self.sk))
        assert len(self.counts_blocked) == len(self.sbu), \
                "The length of counts_alive ({}) is the same as the length of self.sk ({})" \
                .format(len(self.counts_blocked), len(self.sbu))
        if self.finalize_info['type'] in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
            assert self.counts_alive[-1] == 0, "The last element of the counts_alive list is 0 ({})".format(self.counts_alive[-1])
            assert self.counts_blocked[0] == 0, "The first element of the counts_blocked list is 0 ({})".format(self.counts_blocked[0])

        if self.LOG:
            with printoptions(precision=3, suppress=True):
                print("Relative absorption times and counts:\n{}".format(np.c_[np.array(self.sk), np.array(self.counts_alive)]))
                print("Relative blocking times:\n{}".format(np.c_[np.array(self.sbu), np.array(self.counts_blocked)]))

    def insert_relative_time_kill(self, t :float, activation_times :list, event_types :list):
        """
        Inserts new relative time segments in the list of killing times,
        based on the given absolute time `t` and on the given list of activation times:
        the inserted relative times are computed as the absolute time measured
        relative to EACH of the given activation times, as long as they are smaller than t.
        
        A warning is issued when a negative relative time would have been inserted.
        """
        #print("")
        for a in activation_times:
            s = t - a
            if s < 0:
                print("WARNING: The activation event from which the absolute time t={:.3f} should be measured is NOT in the PAST but in the FUTURE! {:.3f}".format(t, a))
                continue
            assert len( set(event_types).intersection( set([EventType.ABSORPTION, EventType.ABSORPTION_F, EventType.CENSORING]) ) ) == 1, \
                    "The event_types list when inserting the relative time for time t={}" \
                    " contains one and only one of the following events: ABSORPTION, fictitious ABSORPTION, CENSORING ({})" \
                    .format(t, [e.name for e in event_types])

            #print("REACTIVATE={}: Inserting time w.r.t. a={:.3f}, t={:.3f}: s={:.3f}".format(self.reactivate, a, t, s))
            idx_insort, found = insort(self.sk, s, unique=False)
            # DM-2021/02/11: The following assertion that the time to insert does not exist in the self.sk list
            # was commented out because very rarely it could fail (see examples below)
            # NOTE that if the time to insert already exists in the target list, the insertion index is
            # (by default behaviour of bisect.bisect()) one index larger than the LAST occurrence of the existing time value.
            # Ex:
            # ll = [0., 1.1, 1.1, 2.3, 2.5]
            # insort(ll, 2.3)    # returns 4
            # insort(ll, 1.1)    # returns 3
            #assert not found, "The relative time value MUST NOT be present in the list of survival time segments ({}):\n{}".format(s, self.sk)
            #DM-2021/02/04: "AssertionError: The relative time value MUST NOT be present in the list of survival time segments (2.5563288374996773e-07)"
            #N=800, nmeantimes=80, seed=1717
            #DM-2021/02/11: "AssertionError: The relative time value MUST NOT be present in the list of survival time segments (4.874884496075538e-09):
            #[0.0, 1.1257483834015147e-09, 1.6709194028408092e-09, 1.9187140765097865e-09, 2.544588539876713e-09,
            # 2.713768765261193e-09, 4.874884496075538e-09, 4.874884496075538e-09, 5.261547642021469e-09, 9.335764161733096e-09, ...
            #This happens for K=5, particles=800, nmeantimes=40, seed=1717
            #We see that the value is present at position 7...
            #I think it's a problem of precision... that a new value that is very similar to a value already present in the list is observed!
            #Note that the repeated value is inserted by insort(), that's why it appears twice.
            self._update_counts_alive(idx_insort, event_types)

    def _update_counts_alive(self, idx_to_insert :int, event_types :list):
        """
        Updates the counts of particles alive in each time segment where the count can change
        following the absorption or censoring of a particle.

        Arguments:
        idx_to_insert: non-negative int
            Index in the counts_alive list where a new element will be inserted to count
            the number of alive particles in the time segment defined by
            [ self.sk[idx_to_insert], self.sk[idx_to_insert+1] )
            If the new element is added at the end of the list, the the upper bound of
            the time segment if Inf.

        event_types: list
            List from where the type of event is extracted, whether a ABSORPTION, ABSORPTION_F, CENSORING.
            Only ONE of these events can appear in the list.

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
        assert len( set(event_types).intersection( set([EventType.ABSORPTION, EventType.ABSORPTION_F, EventType.CENSORING]) ) ) == 1, \
                "The event_types list contains one and only one of the following events: ABSORPTION, fictitious ABSORPTION, CENSORING ({})" \
                .format([e.name for e in event_types])

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

        if list_contains_either(event_types, EventType.CENSORING):
            # Increase by 1 the count corresponding to the inserted index
            # (i.e. the count corresponding to the time segment that STARTS at the new inserted
            # time in the self.sk list of time segments, which means that the particle
            # is considered alive at least up to the end of such time segment --which could be
            # INFINITE if this new element is inserted at the end of the list; in such case
            # we are assuming that the particle is still alive up to Infinity.)  
            self.counts_alive[idx_to_insert] += 1

    def insert_absolute_time_block_unblock(self, t :float, event_types :list):
        "Inserts a new absolute time segment in the list of blocking/unblocking times"
        assert len( set(event_types).intersection( set([EventType.BLOCK, EventType.UNBLOCK, EventType.UNBLOCK_F]) ) ) == 1, \
                "The event_types list when inserting the absolute time t={}" \
                " contains one and only one of the following events: BLOCK, UNBLOCK, fictitious UNBLOCK ({})" \
                .format(t, [e.name for e in event_types])
        # NOTE: The inserted times are NOT necessarily received in order... They are in order for each particle
        # but the number of blocked particles are computed over all particles, therefore the ordered set
        # of times is covered several times, one for each particle, and hence we need to find where
        # we should insert t in the set of blocking/unblocking times stored in self.sbu. 
        idx_insort, found = insort(self.sbu, t, unique=False)
        assert not found, "The absolute time value is NOT found in the list of block/unblock times ({})".format(t)
        #print("Time to insert: {:.3f} BETWEEN indices {} and {} (MAX idx={}):".format(t, idx_insort-1, idx_insort, len(self.counts_blocked)-1))
        self._update_counts_blocked(idx_insort, event_types, new=not found)

    def _update_counts_blocked(self, idx_to_insert_or_update :int, event_types :list, new=True):
        """
        Updates the counts of blocked particles in each time segment where the count can change
        following the blocking or unblocking of a new particle.

        Arguments:
        event_types: list
            List from where the type of event is extracted, whether a BLOCK, UNBLOCK or UNBLOCK_F.
            Only ONE of these events can appear in the list.

        new: bool
            Whether the index to insert corresponds to a new split of an existing time segment
            or whether the time segment already exists and it should be updated. 
        """
        assert 1 <= idx_to_insert_or_update <= len(self.counts_blocked), \
                "The index where a new counter needs to be inserted ({}) is at least 1" \
                " and at most the current number of elements in the counts_blocked array ({})" \
                .format(idx_to_insert_or_update, len(self.counts_blocked))
        assert len( set(event_types).intersection( set([EventType.BLOCK, EventType.UNBLOCK, EventType.UNBLOCK_F]) ) ) == 1, \
                "The event_types list contains one and only one of the following events: BLOCK, UNBLOCK, UNBLOCK_F ({})" \
                .format([e.name for e in event_types])

        if new:
            # Insert a new element in the list and assign it a count value
            # equal to the count of the segment before split.
            self.counts_blocked.insert(idx_to_insert_or_update, self.counts_blocked[idx_to_insert_or_update - 1])

        # INCREASE/DECREASE by 1 ALL counts to the RIGHT of the insert index
        # (because the block/unblock indicates that all time segments that are larger
        # than the elapsed time to block/unblock should count the change of blocking status of the particle)
        if list_contains_either(event_types, EventType.BLOCK):
            delta_count = +1
        elif list_contains_either(event_types, EventType.UNBLOCK):
            delta_count = -1
        for idx in range(idx_to_insert_or_update, len(self.counts_blocked)):
            self.counts_blocked[idx] += delta_count
            if not 0 <= self.counts_blocked[idx] <= self.N:
                self.plot_trajectories_by_particle()
            assert 0 <= self.counts_blocked[idx] <= self.N, \
                    "Insertion index: {}\n".format(idx_to_insert_or_update) + \
                    "The number of blocked particles in time segment with index idx={} out of {} time segments " \
                    .format(idx, len(self.counts_blocked)) + \
                    "found so far is between 0 and N={} ({}) \n({})" \
                    .format(self.N, self.counts_blocked[idx], self.counts_blocked)
    #--------------------------------- Functions to simulate ----------------------------------


    #----------------------------- Functions to analyze the simulation ------------------------
    def compute_survival_probability_from_counts(self, t :list, counts_alive :list):
        """
        Computes the survival probability (Pr(T>t)) at each time step where counts change

        Arguments:
        t: list
            Times at which the number of particles alive change.

        counts_alive: list
            Number of particles alive after each time in `t`.

        Return: list
            Probability of survival at each time in `t` computed as counts_alive / counts_alive[0].
        """
        if len(t) <= 1:
            return [1.0]

        assert counts_alive is not None and len(counts_alive) > 0, \
                    "The input list is not None and has at least 1 element".format(counts_alive)
        if False:
            # Disable this assertion if it takes too much time...
            assert sorted(counts_alive, reverse=True) == counts_alive, \
                        "The input array with the number of particles alive measured at every death time is sorted non-increasingly"
        assert counts_alive[0] > 0, "The number of particles alive at t=0 is positive ({})".format(counts_alive[0])

        return [n_survived / counts_alive[0] for n_survived in counts_alive]        

    def compute_probability_from_counts(self, counts):
        """
        Computes an empirical probability of an event by time using an array of event counts
        at each time step the count changes.
        """
        if len(counts) <= 1:
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
        Computes Pr(K) * (Expected Survival time given start at an absorption state) where
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

    def estimate_proba_survival_given_activation(self):
        """
        Computes the survival probability given the particle started at the activation set

        A data frame is returned containing the following columns:
        - 't': the times at which the survival probability is estimated.
        - 'P(T>t / s=1)': the survival probability at each t.
        """
        # NOTE: The survival time segments may be only one, namely [0.0]
        # This happens when no particle has been absorbed and thus no particle contributes
        # to the estimation of the survival curve
        # (in that case the survival curve is undefined and is actually estimated as 1
        # by the compute_survival_probability_from_counts() method called below)
        assert len(self.sk) > 0, "The length of the survival times array is at least 1 ({})".format(self.sk)
        assert len(self.counts_alive) == len(self.sk), \
                "The number of elements in the survival counts array ({})" \
                " is the same as the number of elements in the survival time segments array ({})" \
                .format(len(self.counts_alive), len(self.sk))
        assert self.counts_alive[-1] == 0, "The count for the last survival time segment is 0 ({})".format(self.counts_alive[-1])
        assert self.counts_alive[0] == len(self.sk) - 1, \
                "The number of particles alive between t=0+ and infinity (counts_alive[0]={})" \
                " is equal to S-1, where S is the number of elements in the survival time segments array (len(sk)-1={})" \
                .format(self.counts_alive[0], len(self.sk)-1)

        # NOTE: The list containing the survival probability is NOT stored in the object
        # because the one we store in the object is the list that is aligned with the times
        # at which the conditional blocking probability is measured (which most likely has
        # more time points at which it is measured than the one we are computing here --where
        # the measurement times are independent of the conditional blocking probability calculation).
        proba_surv_by_t = self.compute_survival_probability_from_counts(self.sk, self.counts_alive)
        return pd.DataFrame.from_items([('t', self.sk), ('P(T>t / s=1)', proba_surv_by_t)])

    def estimate_proba_blocking_conditional_given_activation(self):
        """
        Computes Phi(t,K), the blocking probability at every measurement time t conditioned to survival, 
        based on trajectories that started at an activation state.

        A data frame is returned containing the following columns:
        - 't': the times at which the conditional blocking probability is estimated.
        - 'Phi(t,K / s=1)': the conditional blocking probability at each t.
        """
        assert len(self.counts_blocked) == len(self.sbu), \
                "The number of elements in the blocked counts array ({})" \
                " is the same as the number of elements in the blocking time segments array ({})" \
                .format(len(self.counts_blocked), len(self.sbu))

        # NOTE: The list containing the conditional blocking probability is NOT stored in the object
        # because the one we store in the object is the list that is aligned with the times
        # at which the survival probability is measured (which most likely has
        # more time points at which it is measured than the one we are computing here --where
        # the measurement times are independent of the survival probability calculation).
        proba_block_by_t = self.compute_probability_from_counts(self.counts_blocked)
        return pd.DataFrame.from_items([('t', self.sbu), ('Phi(t,K / s=1)', proba_block_by_t)])

    def estimate_proba_survival_and_blocking_conditional(self):
        """
        Computes the following quantities which are returned in a data frame with the following columns:
        - 't': time at which a change in any quantity happens
        Quantities used in Approximation 1 of the blocking probability estimate:
        - 'P(T>t / s=1)': survival probability given start position = 1
        - 'P(BLOCK / T>t,s=1)': blocking probability given survived and start position = 1
        Quantities used in Approximation 2 of the blocking probability estimate:
        - 'Killing Rate': killing rate a.k.a. gamma parameter
        - 'Blocking Time Estimate' = Pr(BLOCK) * (Expected Survival Time starting at position = 0)
        """
        assert self.finalize_info['condition'] == FinalizeCondition.ACTIVE, \
                "The finalize condition is ACTIVE ({})".format(self.finalize_info['condition'].name)
        assert len(self.counts_blocked) == len(self.sbu), \
                "The number of elements in the blocked counts array ({})" \
                " is the same as the number of elements in the blocking time segments array ({})" \
                .format(len(self.counts_blocked), len(self.sbu))

        if self.finalize_info['type'] in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
            assert self.counts_alive[-1] == 0, \
                    "The number of particles alive at the last measured time is 0" \
                    " since particles are assumed to have been all absorbed or removed ({})" \
                    .format(self.counts_alive)
            # NOTE: (2021/02/18) The following assertion is no longer true
            # because the fictitious unblocking events are NOT used as valid unblocking events
            # (see compute_counts() function, when dealing with BLOCK and UNBLOCK events)
            #assert self.counts_blocked[-1] == 0, \
            #        "The number of particles blocked at the last measured time is 0" \
            #        " since particles are assumed to have been all absorbed or removed ({})" \
            #        .format(self.counts_blocked)

        if False:
            with printoptions(precision=3, suppress=True):
                print("SURVIVAL: times and counts_alive: \n{}".format(np.c_[self.sk, self.counts_alive, [c/self.counts_alive[0] for c in self.counts_alive] ]))
                print("BLOCKING: times and counts_blocked:\n{}".format(np.c_[self.sbu, self.counts_blocked, [c/self.N for c in self.counts_blocked]]))

        # Since we must compute this probability CONDITIONED to the event T > t,
        # we first need to merge the measurements of the block times and of the survival times
        # into one single set of measurement time points.
        if self.proba_survival_given_activation is not None:
            # The estimated survival probability given activation was given by the user
            # => Merge the time values at which the survival probability changes with the time values
            # at which the blocking and unblocking events occur.
            self.t, self.proba_surv_by_t, counts_blocked = merge_values_in_time(
                                                                        list(self.proba_survival_given_activation['t']),
                                                                        list(self.proba_survival_given_activation['P(T>t / s=1)']),
                                                                        self.sbu, self.counts_blocked, unique=False)
        else:
            self.t, counts_alive, counts_blocked = merge_values_in_time(self.sk, self.counts_alive, self.sbu, self.counts_blocked, unique=False)
            self.proba_surv_by_t = self.compute_survival_probability_from_counts(self.t, counts_alive)

        self.proba_block_by_t = self.compute_probability_from_counts(counts_blocked)
        self.gamma = None #self.compute_killing_rate()
        self.blocking_time_estimate = None #self.compute_blocking_time_estimate()
        
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
        if self.finalize_info['type'] in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
            assert self.proba_surv_by_t[-1] in [1.0, 0.0], "The survival function at the last measured time is either 1 (when no particles have been absorbed) or 0 (when at least one particle has been absorbed) ({})".format(self.proba_surv_by_t[-1])
            # NOTE: (2021/02/18) The following assertion is no longer true
            # because the fictitious unblocking events are NOT used as valid unblocking events
            # (see compute_counts() function, when dealing with BLOCK and UNBLOCK events)
            #assert self.proba_block_by_t[-1] == 0.0, "The conditional blocking probability at the last measured time is 0 ({})".format(self.proba_block_by_t[-1])

        if False:
            print("Survival Probability and Conditional Blocking Probability:")
            print(pd.DataFrame.from_items([ ('t', self.t),
                    ('P(T>t / s=1)', self.proba_surv_by_t),
                    ('P(BLOCK / T>t,s=1)', self.proba_block_by_t),
                    ('Killing Rate', self.gamma),
                    ('Blocking Time Estimate', self.blocking_time_estimate)]))

        return pd.DataFrame.from_items([('t', self.t),
                                        ('P(T>t / s=1)', self.proba_surv_by_t),
                                        ('P(BLOCK / T>t,s=1)', self.proba_block_by_t),
                                        ('Killing Rate', self.gamma),
                                        ('Blocking Time Estimate', self.blocking_time_estimate)])

    def estimate_expected_killing_time(self):
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

        assert not self.reactivate, "The REACTIVATE parameter is FALSE."
            ## Note: When reactivation is used, estimating the survival time does not make sense
            ## because particles never start at the absorption set.
        all_particles_started_at_0 = lambda: all([self.getStartPosition(P) == 0 for P in range(self.N)])
        assert self.finalize_info['condition'] == FinalizeCondition.ACTIVE or all_particles_started_at_0, \
                "The finalize condition is ACTIVE ({}) or, if not, the start position of all particles is 0 " + \
                "(which makes the condition 'the particle did not return to its start position' " + \
                "to be equivalent to the condition 'the particle is ACTIVE', thus making the two possible " + \
                "conditions for finalization TOTALLY EQUIVALENT)" \
                .format(self.finalize_info['condition'].name)

        #if self.finalize_info['type'] == FinalizeType.ESTIMATE_CENSORED:
        #    # Add an estimation of the survival time for the censored (i.e. still active) particles
        #    for P in self.active_particles:
        #        survival_time = self.particles[P].getTimesLastEvents() - self.times0[P]
        #        ### TODO: (2020/06/13) Write the get_survival_function() method that returns the estimate of S(t)
        #        self.ktimes0_sum[P] += survival_time # / (1 - self.get_survival_function(survival_time))
        #        self.ktimes0_n[P] += 1
        #else:
        if not self.finalize_info['type'] in [FinalizeType.NONE, FinalizeType.ESTIMATE_CENSORED]:
            assert sum(self.is_particle_active) == 0, "The number of active particles is 0 ({})".format(sum(self.is_particle_active))

        total_survival_time = np.sum(self.ktimes0_sum)
        total_survival_n = np.sum(self.ktimes0_n)

        if False:
            print("Estimating expected killing time...")
            print("Total killing time over all particles (N={}): {:.1f}".format(self.N, total_survival_time))
            print("Number of observed killing times over all N={} particles: {}".format(self.N, total_survival_n))

        if total_survival_n == 0:
            print("WARNING (estimation of expected killing time): No particle has been absorbed.\n" +
                  "The estimated expected killing time is estimated as the average of the simulated times over all particles.")
            self.expected_survival_time = np.mean([self.particles[P].getMostRecentEventTime() for P in range(self.N)])
        else:
            if total_survival_n < 0.5*self.N:
                print("WARNING (estimation of expected killing time): " \
                      "The number of observed killing times is smaller than half the number of particles (sum of {} < number of particles ({})).\n" \
                      .format(self.ktimes0_n, self.N) +
                      "The estimated killing time may be unreliable.")
            self.expected_survival_time = total_survival_time / total_survival_n

        return self.expected_survival_time, total_survival_time, total_survival_n

    def estimate_expected_return_time(self):
        "Estimates the expected return time to one of the observed positions"
        assert not self.reactivate, "The REACTIVATE parameter is FALSE."

        total_return_time = np.sum(self.rtimes_obs_sum)
        total_return_n = np.sum(self.rtimes_obs_n)

        if False:
            print("Estimating expected return time to an observed position: {}...".format(self.positions_observe))
            print("Total return time over all particles (N={}): {:.1f}".format(self.N, total_return_time))
            print("Number of observed returned times over all N={} particles: {}".format(self.N, total_return_n))

        if total_return_time == 0:
            print("WARNING (estimation of expected return time to an observed position): No particle has returned to any of those positions yet.\n" +
                  "The expected return time to the set of absorption states is estimated as the average of the simulated times over all particles.")
            self.expected_return_time = np.mean([self.particles[P].getMostRecentEventTime() - self.rtimes_offset[P] for P in range(self.N)])
        else:
            if total_return_n < 0.5*self.N:
                print("WARNING (estimation of expected return time to an observed position): " \
                      "The number of observed return times is smaller than half the number of particles (sum of {} < N={}).\n" \
                      .format(self.rtimes_obs_n, self.N) +
                      "The estimated expected return time may be unreliable.")
            self.expected_return_time = total_return_time / total_return_n

        return self.expected_return_time, total_return_time, total_return_n

    def estimate_proba_blocking_via_integral(self, expected_survival_time):
        "Computes the blocking probability via Approximation 1 in Matt's draft"
        assert  len(self.t) == len(self.proba_surv_by_t) and \
                len(self.proba_surv_by_t) == len(self.proba_block_by_t), \
                "The length of the time, proba_surv_by_t, and proba_block_by_t are the same ({}, {}, {})" \
                .format(len(self.t), len(self.proba_surv_by_t), len(self.proba_surv_by_t)) 
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

    def estimate_proba_blocking_fv(self):
        """
        Estimates the blocking probability using the Fleming-Viot estimator

        Return: tuple
        The tuple contains the following elements:
        - the estimated blocking probability via Approximation 1 (integral approach)
        - the estimated blocking probability via Approximation 2 (Laplacian approach)
        - the value of the integral in Approximation 1 
        - the killing rate gamma in Approximation 2
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
        # - self.blocking_time_estimate: estimated blocking time = Pr(block) * (Expected Survival Time starting at 0)
        df = self.estimate_proba_survival_and_blocking_conditional()
        if False:
            if self.mean_lifetime is not None:
                plt.plot(df['t'], df['Killing Rate'], 'b.-')
                ax = plt.gca()
                ax.set_ylim((0,2))
                ax.set_xlabel("t")
                ax.set_ylabel("Killing Rate (gamma)")
                plt.title("K={}, particles={}, maxtime={:.1f}".format(self.queue.getCapacity(), self.N, self.maxtime))
                plt.show()

        #-- Expected survival time was given by the user as a positive value
        assert self.mean_lifetime is not None and self.mean_lifetime > 0, \
            "The expected survival time used in the F-V estimator is positive ({:.3f})".format(self.mean_lifetime)

        #-- Blocking probability estimate via Approximation 1: estimate the integral
        proba_blocking_integral, integral = self.estimate_proba_blocking_via_integral(self.mean_lifetime)

        #-- Blocking probability estimate via Approximation 2: estimate the Laplacian eigenvalue (gamma) and eigenvector (h(1))
        proba_blocking_laplacian, gamma = None, None #self.estimate_proba_blocking_via_laplacian(self.mean_lifetime)

        return proba_blocking_integral, proba_blocking_laplacian, integral, gamma

    def estimate_proba_blocking_mc(self):
        """
        Estimates the blocking probability using Monte Carlo and computes the expected survival time
        
        The blocking probability is estimated as the ratio between the total blocking time and
        the total survival time over all particles.
        
        Note that this methodology of summing ALL times over ALL particles, avoids having to weight
        the mean lifetime estimation of each particle, as we don't need to weight the mean lifetime
        estimated for each particle with the number of samples that mean lifetime is based on.

        Return: tuple
        The tuple contains the following elements:
        - the estimated blocking probability as the proportion of blocking time over survival time
        - the total blocking time over all particles
        - the total survival time over all particles
        - the number of times the survival time was measured
        """
        assert self.N == 1, "The number of simulated particles is 1"

        # TODO: (2021/04/27) Fix the computation of the denominator of the estimated blocking probability by computing the total return time to the START STATE (as opposed to computing the sum of all the times to absorption --killing times)
        # In fact, we are now computing the total killing time, as opposed to the total return time to the start state.
        # Doing this is still usually fine when computing the blocking probability because:
        # - the killing time is either:
        #    - equal to the return time to the start state (0) (IN WHICH CASE WE ARE IN BUSINESS)
        #        --> this happens when the first return time to the start state (0)
        #        coincides with time at which killing occurs (which happens when the start state is visited again for the first time 
        #        by coming from above in the state space), OR
        #    - equal to the sum of all the times of return to the start state (0) (from BELOW) PLUS
        #    the time elapsed from that moment until killing occurs.
        #    (IN WHICH CASE WE ARE STILL IN BUSINESS, as long as the final killing time is observed and not censored
        #    because we are still summing all he blocking times in the numerator. 
        # - even though the first measured killing time is NOT a return time to the start state 0,
        # (because when we call this function --which is responsible of computing the blocking probability via MC--,
        # we simulate the system by starting at an ACTIVATION state, NOT at the boundary of the ABSORPTION set (s=0))
        # that kiling time we measure is the ONLY ONE not starting at 0, ALL the following killing times are measured
        # w.r.t. the system starting at 0.
        total_blocking_time = np.sum(self.btimes_sum)
        total_survival_time = np.sum(self.ktimes0_sum)
        total_survival_n = np.sum(self.ktimes0_n)
        if total_survival_time == 0:
            print("WARNING (estimation of blocking probability by MC): No particle has been absorbed.\n" +
                  "The total return time to the set of absorption states is estimated as the total simulated time over all particles.")
            total_survival_time = np.sum([self.particles[P].getMostRecentEventTime() for P in range(self.N)])
        blocking_time_rate = total_blocking_time / total_survival_time
        return blocking_time_rate, total_blocking_time, total_survival_time, total_survival_n
    #----------------------------- Functions to analyze the simulation ------------------------


    #-------------------------------------- Getters -------------------------------------------
    def getFinalizeType(self):
        return self.finalize_info['type']

    def getFinalizeCondition(self):
        return self.finalize_info['condition']

    def getStartPosition(self, P):
        return self.all_positions_buffer[P][0]

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

    def get_censored_particle_numbers(self):
        "Returns the list of particle numbers P that are censored based on the Finalize condition"
        if self.finalize_info['condition'] == FinalizeCondition.ACTIVE:
            return [P for P in range(self.N) if self.is_particle_active[P]]
        elif self.finalize_info['condition'] == FinalizeCondition.NOT_START_POSITION:
            return [P for P in range(self.N) if self.particles[P].getBufferSize() != self.getStartPosition(P)]

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
            block_times_p = [t      for idx, t in enumerate(dict_info['t'])
                                    if list_contains_either(dict_info['E'][idx], EventType.BLOCK)]
            unblock_times_p = [t    for idx, t in enumerate(dict_info['t'])
                                    if list_contains_either(dict_info['E'][idx], EventType.UNBLOCK)]

            if self.finalize_info['type'] in [FinalizeType.ABSORB_CENSORED, FinalizeType.REMOVE_CENSORED]:
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
        "Returns the total survival time for a particle (from an absorption state back to an absorption state)"
        all_survival_periods = self.get_all_survival_periods()
        return np.sum(all_survival_periods[P]['Survival Period Span'])

    def get_all_survival_periods(self):
        """
        Returns the survival periods for each particle number P
        (these are the periods in which the particle goes from an absorption state back to an absorption state).
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
                                                    if list_contains_either(dict_info['E'][idx], [EventType.ABSORPTION, EventType.CENSORING])]
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
        Returns the survival times from an absorption state for all particles
        
        Return: tuple
        The tuple contains information about the survival periods (i.e. the contiguous time
        during which a particle goes from an absorption state back to an absorption state for the first time)
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

    def get_time_latest_known_state(self):
        "Returns the time of the latest known state of the system (i.e. over all particles)"
        return np.min(self.times_latest_known_state)

    def get_time_latest_known_state_for_particle(self, P):
        return self.times_latest_known_state[P]

    def get_counts_particles_alive_by_elapsed_time(self):
        return self.counts_alive

    def get_counts_particles_blocked_by_elapsed_time(self):
        return self.counts_blocked

    def plot_trajectories_by_particle(self):
        "Plots the trajectories of the particles"
        K = self.queue.getCapacity()
        colormap = cm.get_cmap("jet")
        # Reference lines showing for each particle:
        # - state=0
        # - state=activation
        # - state=block
        reflines_zero = range(0, (K+1)*self.N, K+1)
        reflines_activation = range(self.buffer_size_activation, (self.buffer_size_activation+(K+1))*self.N, K+1)
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
        ax.set_yticks(reflines_zero)
        ax.yaxis.set_ticklabels(particle_numbers)
        #ax.xaxis.set_ticks(np.arange(0, round(self.maxtime)+1) )
        ax.set_ylim((0, (K+1)*self.N))
        ax.hlines(reflines_block, 0, self.maxtime, color='gray', linestyles='dashed')
        ax.hlines(reflines_activation, 0, self.maxtime, color='red', linestyles='dashed')
        ax.hlines(reflines_zero, 0, self.maxtime, color='gray')
        ax.vlines(self.maxtime, 0, (K+1)*self.N, color='red', linestyles='dashed')
        for p in particle_numbers:
            color = colormap( (p+1) / self.N )
            # Non-overlapping step plots at vertical positions (K+1)*p
            plt.step(self.all_times_buffer[p], [(K+1)*p + pos for pos in self.all_positions_buffer[p]], '-', #'x-',
                     where='post', color=color, markersize=3)
        plt.title("K={}, rates(B)={}, rates(D)={}, activation={}, reactivate={}, finalize={}, N={}, maxtime={:.1f}, seed={}" \
                  .format(self.queue.getCapacity(),
                      self.queue.getBirthRates(),
                      self.queue.getDeathRates(),
                      self.buffer_size_activation,
                      self.reactivate, self.finalize_info['type'].name[0:3], self.N, self.maxtime, self.seed
                      ))
        ax.title.set_fontsize(9)
        plt.show()

    def plot_trajectories_by_server(self, P):
        "Plots the trajectories of the particles"
        K = self.queue.getCapacity()
        colormap = cm.get_cmap("jet")
        # Reference lines showing for each particle:
        # - state=0
        # - state=activation
        # - state=block
        reflines_zero = range(0, (K+1)*(self.nservers+1), K+1)       # We add a server (#self.nservers+1) because the last "server" will show the system's buffer size
        reflines_activation = range(self.buffer_size_activation, (self.buffer_size_activation+(K+1))*(self.nservers+1), K+1)       # We add a server (#self.nservers+1) because the last "server" will show the system's buffer size
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
        ax.set_yticks(reflines_zero)
        ax.yaxis.set_ticklabels(servers)
        #ax.xaxis.set_ticks(np.arange(0, round(self.maxtime)+1) )
        ax.set_ylim((0, (K+1)*(self.nservers+1)))
        ax.hlines(reflines_block, 0, self.maxtime, color='gray', linestyles='dashed')
        ax.hlines(reflines_activation, 0, self.maxtime, color='red', linestyles='dashed')
        ax.hlines(reflines_zero, 0, self.maxtime, color='gray')
        ax.vlines(self.maxtime, 0, (K+1)*(self.nservers+1), color='red', linestyles='dashed')
        for s in servers:
            color = colormap( (s+1) / self.nservers )
            # Non-overlapping step plots at vertical positions (K+1)*s
            plt.step(self.all_times_by_server[P][s] , [(K+1)*s + pos for pos in self.all_positions_by_server[P][s]], '-', #'x-',
                     where='post', color=color, markersize=3)
            # Complete the line of each server up to the buffer time 
            ax.hlines((K+1)*s + self.all_positions_by_server[P][s][-1], self.all_times_by_server[P][s][-1], self.times_buffer[P], color=color)
        # Add the system's buffer size on top
        plt.step(self.all_times_buffer[P], [(K+1)*self.nservers + pos for pos in self.all_positions_buffer[P]], '-', #'x-',
                     where='post', color="black", markersize=3)
        
        plt.title("Particle {}: K={}, rates(B)={}, rates(D)={}, activation={}, reactivate={}, finalize={}, #servers={}, maxtime={:.1f}, seed={}" \
                  .format(P,
                    self.queue.getCapacity(),
                    self.queue.getBirthRates(),
                    self.queue.getDeathRates(),
                    self.buffer_size_activation,
                    self.reactivate, self.finalize_info['type'].name[0:3], self.nservers, self.maxtime, self.seed
                    ))
        ax.title.set_fontsize(9)
        plt.show()

    #-------------- Helper functions
    def setup(self):
        # Note: Cannot use {:.3f} as format for the mean life time below because its value may be None and we get an error of unsupported format
        params_str = "***********************" \
                    "\nK = {}" \
                    "\njob arriving rates = {}" \
                    "\njob service rates = {}" \
                    "\n# particles = {}" \
                    "\n# servers = {}" \
                    "\nactivation buffer size = {}" \
                    "\nmean_lifetime = {}" \
                    "\nreactivate = {}" \
                    "\nfinalize_type = {}" \
                    "\nnmeantimes = {:.1f} (maxtime = {:.1f})" \
                    "\nseed = {}" \
                    "\n***********************" \
                    .format(self.queue.getCapacity(),
                            self.job_rates, self.queue.getDeathRates(),
                            self.N, self.nservers, self.buffer_size_activation, self.mean_lifetime,
                            self.reactivate, self.finalize_info['type'].name, self.nmeantimes, self.maxtime, self.seed)
        return params_str

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
        _, server_last_updated, type_of_last_event, time_of_last_event = self.particles[P].getMostRecentEventInfo()
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
        and equal to the last time the position (buffer size) of the particle changed.
        """
        for server in range(self.nservers):
            if self.trajectories[P][server]['t'][-1] != self.times_buffer[P]:
                return False
        return True

    def assertSystemConsistency(self, P):
        """
        Checks whether the information stored in the particles coincides with the information
        stored in the attributes containing the current and historical positions and times
        as of the latest time at which we known state of the system.
        
        The information checked is:
        - at buffer level:
            - particle's buffer size coincides with current buffer size stored in the attribute.
            - time of particle's most recent change coincides with time of current buffer size.
            - current buffer size coincides with latest historical buffer size.
            - time of current buffer size coincides with latest historical time at which the buffer changes.
        - at server level:
            - particle's server sizes coincide with current server sizes stored in the attribute.
            - times of particle's most recent changes in servers coincide with the times of current change by server stored in the attribute.
            - current particle's server sizes coincide with latest historical server sizes.
            - time of current particle's server sizes coincide with latest historical times at which server sizes change.
        """
        def is_state_consistent_for_particle(P):
            return  self.particles[P].getBufferSize()           == self.positions_buffer[P] and \
                    self.particles[P].getMostRecentEventTime()  == self.times_buffer[P] and \
                    self.positions_buffer[P]                    == self.all_positions_buffer[P][-1] and \
                    self.times_buffer[P]                        == self.all_times_buffer[P][-1] and \
                    True and \
                    all(self.particles[P].getServerSizes()      == self.positions_by_server[P]) and \
                    all([np.isnan(self.times_by_server[P][server]) or self.particles[P].getTimeLastEvent(server) == self.times_by_server[P][server] for server in range(self.nservers)]) and \
                    all([self.positions_by_server[P][server]    == self.all_positions_by_server[P][server][-1] for server in range(self.nservers)]) and \
                    all([np.isnan(self.times_by_server[P][server]) or self.times_by_server[P][server] == self.all_times_by_server[P][server][-1] for server in range(self.nservers)])

        consistent_state = is_state_consistent_for_particle(P)
        assert consistent_state, \
                "t={:.3f}: The system is NOT in a consistent state for particle {}" \
                "\n\tparticles[{}].size={}      vs. positions_buffer[{}]={}" \
                "\n\tparticles[{}].time={:.3f}  vs. times_buffer[{}]={:.3f}" \
                "\n\tpositions_buffer[{}]={}    vs. all_positions_buffer[{}][-1]={}" \
                "\n\ttimes_buffer[{}]={:.3f}    vs. all_times_buffer[{}][-1]={:.3f}" \
                "\n\tparticles[{}].positions={} vs. positions_by_server[{}]={}" \
                "\n\tparticles[{}].times={}     vs. times_by_server[{}]={}" \
                "\n\tpositions_by_server[{}]={} vs. all_positions_by_server[{}][-1]={}" \
                "\n\ttimes_by_server[{}]={} vs. all_times_by_server[{}][-1]={}" \
                .format(self.times_latest_known_state[P], P,
                        P, self.particles[P].getBufferSize(), P, self.positions_buffer[P],
                        P, self.particles[P].getMostRecentEventTime(), P, self.times_buffer[P],
                        P, self.positions_buffer[P], P, self.all_positions_buffer[P][-1],
                        P, self.times_buffer[P], P, self.all_times_buffer[P][-1],
                        P, self.particles[P].getServerSizes(), P, self.positions_by_server[P],
                        P, self.particles[P].getTimesLastEvents(), P, self.times_by_server[P],
                        P, self.positions_by_server[P], P, [self.all_positions_by_server[P][server][-1] for server in range(self.nservers)],
                        P, self.times_by_server[P], P, [self.all_times_by_server[P][server][-1] for server in range(self.nservers)])

    def raiseErrorInvalidParticle(self, P):
        raise ValueError("Wrong particle number: {}. Valid values are integers between 0 and {}.\n".format(P, self.N-1))

    def raiseWarningInvalidParticle(self, P):
        raise Warning("Wrong particle number: {}. Valid values are integers between 0 and {}.\n".format(P, self.N-1))


def estimate_blocking_mc(env_queue :EnvQueueSingleBufferWithJobClasses, dict_params_simul :dict, dict_params_info :dict=None):
    """
    Estimate the blocking probability of a queue system using Monte-Carlo

    Arguments:
    dict_params_simul: dict
        Dictionary containing the simulation parameters as follows:
        - 'nparticles'
        - 'nmeantimes'
        - 'buffer_size_activation'
        - 'multiplier_for_extended_simulation_time'
        - 'seed'

    dict_params_info: (opt) dict
        Dictionary containing general info parameters as follows:
        - 'plot': whether to create plots
        - 'log': whether to show messages

    Return: tuple
        Tuple with the following elements:
        - proba_blocking_mc: the estimated blocking probability by Monte-Carlo
        - est_mc: the EstimatorQueueBlockingFlemingViot object used in the simulation
    """

    # Parse input parameters
    set_required_parameters_simul = set([
        'buffer_size_activation',
        'multiplier_for_extended_simulation_time',
        'nmeantimes',
        'nparticles',
        'seed'
        ])
    if not set_required_parameters_simul.issubset( dict_params_simul.keys() ):
        raise ValueError("Not all required parameters were given in input dictionary 'dict_params_simul'.\nRequired parameters are: " \
                         .format(set_required_parameters_simul))

    # Multiplier for the simulation time: MULTIPLIER * nmeantimes * nparticles
    # in order to hopefully get a precise estimate of the survival curve and the expected survival time   
    MULTIPLIER = dict_params_simul['multiplier_for_extended_simulation_time']

    if dict_params_info is None:
        dict_params_info = {'plot': False, 'log': False}
    else:
        if not set(['plot', 'log']).issubset( dict_params_info.keys() ):
            raise ValueError("Not all required parameters were given in input dictionary 'dict_params_info'.\nRequired parameters are: " \
                             .format(set(['plot', 'log'])))

    time_start = timer()
    # Object that is used to estimate the blocking probability via Monte-Carlo as "total blocking time" / "total survival time" 
    est_mc = EstimatorQueueBlockingFlemingViot(1, env_queue.queue, env_queue.getJobRates(),
                                               service_rates=env_queue.getServiceRates(),
                                               buffer_size_activation=dict_params_simul['buffer_size_activation'],
                                               nmeantimes=MULTIPLIER * dict_params_simul['nparticles'] * dict_params_simul['nmeantimes'],
                                               policy_assign=env_queue.getAssignPolicy(),
                                               mean_lifetime=None,
                                               proba_survival_given_activation=None,
                                               reactivate=False,
                                               finalize_info={'type': FinalizeType.REMOVE_CENSORED, 'condition': FinalizeCondition.NOT_START_POSITION},
                                               seed=dict_params_simul['seed'],
                                               plotFlag=dict_params_info['plot'],
                                               log=dict_params_info['log'])

    print("\tStep 1 of 1: Estimating the blocking probability by Monte-Carlo (seed={})...".format(est_mc.seed))
    proba_blocking_mc, _, total_survival_time, n_survival_observations = est_mc.simulate(EventType.ACTIVATION)
    print("\t--> Number of observations for Pr(K) estimation: {} ({:.1f})% of simulation time T={:.1f})".format(n_survival_observations, total_survival_time / est_mc.maxtime * 100, est_mc.maxtime))
    time_end = timer()
    exec_time = time_end - time_start
    print("execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))

    return proba_blocking_mc, est_mc

def estimate_blocking_fv(env_queue :EnvQueueSingleBufferWithJobClasses,
                         dict_params_simul :dict,
                         dict_params_info :dict=None,
                         est :EstimatorQueueBlockingFlemingViot=None):
    """
    Estimate the blocking probability of a queue system using Fleming-Viot approach

    Arguments:
    dict_params_simul: dict
        Dictionary containing the simulation parameters as follows:
        - 'nparticles'
        - 'nmeantimes'
        - 'buffer_size_activation'
        - 'multiplier_for_extended_simulation_time'
        - 'seed'

    dict_params_info: (opt) dict
        Dictionary containing general info parameters as follows:
        - 'plot': whether to create plots
        - 'log': whether to show messages

    est: (opt) EstimatorQueueBlockingFlemingViot
        Existing estimator already simulated from where we can retrieve other measures.
        A typical example is: an MC estimator has been already simulated starting at the activation set
        and we use it here to estimate the survival probability curve.

    Return: tuple
        Tuple with the following elements:
        - proba_blocking_fv: the estimated blocking probability by Fleming-Viot
        - integral: the integral used in the Fleming-Viot estimator, Approximation 1
        - expected_survival_time: the denominator E(T) used in the Fleming-Viot estimator, Approximation 1
        - n_survival_curve_observations: the number of observations used to estimate the survival curve P(T>t / s=1)
        - n_expected_survival_observations: the number of observations used to estimate the expected survival time E(T)
        - est_fv: the EstimatorQueueBlockingFlemingViot object used in the Fleming-Viot simulation
    """

    # Parse input parameters
    set_required_parameters_simul = set([
        'buffer_size_activation',
        'multiplier_for_extended_simulation_time',
        'nmeantimes',
        'nparticles',
        'seed'
        ])
    if not set_required_parameters_simul.issubset( dict_params_simul.keys() ):
        raise ValueError("Not all required parameters were given in input dictionary 'dict_params_simul'.\nRequired parameters are: " \
                         .format(set_required_parameters_simul))

    # Multiplier for the simulation time: MULTIPLIER * nmeantimes * nparticles
    # in order to hopefully get a precise estimate of the survival curve and the expected survival time   
    MULTIPLIER = dict_params_simul['multiplier_for_extended_simulation_time']

    if dict_params_info is None:
        dict_params_info = {'plot': False, 'log': False}
    else:
        if not set(['plot', 'log']).issubset( dict_params_info.keys() ):
            raise ValueError("Not all required parameters were given in input dictionary 'dict_params_info'.\nRequired parameters are: " \
                             .format(set(['plot', 'log'])))

    # Object that is used to estimate via Monte-Carlo:
    # - P(T>t): the survival probability given a BORDER activation state
    # - E(T): the expected return time to absorption given a BORDER absorption state
    # NOTES:
    # - P(T>t) is estimated by running N particles that start at an ACTIVATION state up to the FIRST absorption
    # In this way, we avoid the problem that the particle has to return to an ACTIVATION state in order to
    # measure the survival again.  
    # In order to avoid running out of simulation time, we multiply the nmeantimes parameter
    # affecting the simulation time by the number of particles.
    # - E(T) is estimated by running N particles that start at an ABSORPTION state until the END of the simulation time
    # and measuring the average RETURN time to an ABSORPTION state
    # (either from within the absorption set or from within the activation set --so we are NOT ONLY looking at KILLING events!)
    # Note that we CANNOT follow the same procedure we carry out for the estimation of P(T>t) because of 
    # theoretical requirements
    # (Ref: 2-hour call with Matt on Tue, 27-Apr-2021 and Asmussen, pag. 170 when he talks about regenerative processes).
    # 
    # In BOTH CASES censored particles are REMOVED. Note however that the censoring condition is different in each case:
    # - For P(T>t), censoring happens when the particle is ACTIVE (i.e. it has not been absorbed)
    # - For E(T), censoring happens when the particle has NOT RETURNED to the start position, namely an ABSORPTION state.    
    if est is None:
        time_start = timer()
        est_surv = EstimatorQueueBlockingFlemingViot(dict_params_simul['nparticles'], env_queue.queue, env_queue.getJobRates(),
                                                   service_rates=env_queue.getServiceRates(),
                                                   buffer_size_activation=dict_params_simul['buffer_size_activation'],
                                                   nmeantimes=MULTIPLIER * dict_params_simul['nparticles'] * dict_params_simul['nmeantimes'],
                                                   policy_assign=env_queue.getAssignPolicy(),
                                                   mean_lifetime=None,
                                                   proba_survival_given_activation=None,
                                                   reactivate=False,
                                                   finalize_info={'type': FinalizeType.REMOVE_CENSORED, 'condition': FinalizeCondition.ACTIVE},
                                                   seed=dict_params_simul['seed'],
                                                   plotFlag=dict_params_info['plot'],
                                                   log=dict_params_info['log'])
        print("\tStep 1 of 3: Simulating using an ACTIVATION start state to estimate P(T>t / s=act), the survival probability given activation (seed={})...".format(est_surv.seed))
        _, surv_counts, _, _ = est_surv.simulate_survival(N_min=1)
        n_survival_curve_observations = surv_counts[0]
        print("\t--> Number of observations for P(T>t) estimation: {} out of N={}".format(n_survival_curve_observations, est_surv.N))
        time_end = timer()
        exec_time = time_end - time_start
        print("execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))
    else:
        print("\tStep 1 of 3: Using the given EstimatorQueueBlockingFlemingViot to estimate P(T>t / s=act), the survival probability given activation...")
        est_surv = est
        # We need to make sure that the finalize condition is set to ACTIVE because normally this `est` object
        # comes from a Monte-Carlo estimation which uses a finalize condition equal to NOT_START_POSITION.
        est_surv.finalize_info['condition'] = FinalizeCondition.ACTIVE
        n_survival_curve_observations = est_surv.counts_alive[0]
        print("\t--> Number of observations for P(T>t) estimation: {} (N={})".format(n_survival_curve_observations, est_surv.N))
    proba_survival_given_activation = est_surv.estimate_proba_survival_given_activation()

    time_start = timer()
    est_abs = EstimatorQueueBlockingFlemingViot(1, env_queue.queue, env_queue.getJobRates(),
                                               service_rates=env_queue.getServiceRates(),
                                               buffer_size_activation=dict_params_simul['buffer_size_activation'],
                                               positions_observe=[ dict_params_simul['buffer_size_activation'] - 1 ],
                                               nmeantimes=MULTIPLIER * dict_params_simul['nparticles'] * dict_params_simul['nmeantimes'],
                                               policy_assign=env_queue.getAssignPolicy(),
                                               mean_lifetime=None,
                                               proba_survival_given_activation=None,
                                               reactivate=False,
                                               finalize_info={'type': FinalizeType.REMOVE_CENSORED, 'condition': FinalizeCondition.ACTIVE},
                                               seed=dict_params_simul['seed'],
                                               plotFlag=dict_params_info['plot'],
                                               log=dict_params_info['log'])
    print("\tStep 2 of 3: Simulating using an ABSORPTION start state to estimate E(T / s=abs), the expected killing time given we start at the boundary of the absorption set (seed={})...".format(est_abs.seed))
    est_abs.simulate_return_time_to_absorption(N_min=1)
    # This is when E(T) is estimated as expected RETURN time to the starting position of absorption (boundary) 
    #expected_return_time, total_return_time, n_return_time_observations = est_abs.estimate_expected_return_time()
    # This is when E(T) is estimated as expected time to KILLING
    # (which is what we are interested becaues the numerator of the approximation also observes KILLING!)
    expected_survival_time, total_survival_time, n_survival_time_observations = est_abs.estimate_expected_killing_time()
    print("\t--> Number of observations for E(T) estimation: {} on the N={} particles {:.1f})% of total N simulation times N*T={:.1f}) ".format(n_survival_time_observations, est_abs.N, total_survival_time / (est_abs.N * est_abs.maxtime) * 100, est_abs.N * est_abs.maxtime))
    time_end = timer()
    exec_time = time_end - time_start
    print("execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))

    # Fleming-Viot estimation
    # The estimated expected return time to absorption and the survival curve are used as input
    finalize_type = FinalizeType.ABSORB_CENSORED
    seed = dict_params_simul['seed'] + 1
    print("\tStep 3 of 3: Running Fleming-Viot simulation using an ACTIVATION start state to estimate blocking probability using E(T) = {:.1f} (out of simul time={:.1f}) (seed={})..." \
          .format(expected_survival_time, est_surv.maxtime, seed))
    est_fv = EstimatorQueueBlockingFlemingViot(dict_params_simul['nparticles'], env_queue.queue, env_queue.getJobRates(),
                                               service_rates=env_queue.getServiceRates(),
                                               buffer_size_activation=dict_params_simul['buffer_size_activation'],
                                               nmeantimes=dict_params_simul['nmeantimes'],
                                               policy_assign=env_queue.getAssignPolicy(),
                                               mean_lifetime=expected_survival_time,
                                               proba_survival_given_activation=proba_survival_given_activation,
                                               reactivate=True,
                                               finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                               seed=seed,
                                               plotFlag=dict_params_info['plot'],
                                               log=dict_params_info['log'])
    proba_blocking_fv, _, integral, _, _ = est_fv.simulate(EventType.ACTIVATION)
    if False: #dict_params_info['plot']:
        df_proba_survival_and_blocking_conditional = est_fv.estimate_proba_survival_and_blocking_conditional() 
        plot_curve_estimates(df_proba_survival_and_blocking_conditional,
                          dict_params={
                            'birth_rates': est_fv.queue.getBirthRates(),
                            'death_rates': est_fv.queue.getDeathRates(),
                            'K': est_fv.queue.getCapacity(),
                            'nparticles': dict_params_simul['nparticles'],
                            'nmeantimes': dict_params_simul['nmeantimes'],
                            'maxtime_mc': est_surv.maxtime,
                            'maxtime_fv': est_fv.maxtime,
                            'buffer_size_activation': dict_params_simul['buffer_size_activation'],
                            'mean_lifetime': expected_survival_time,
                            'n_survival_curve_observations': n_survival_curve_observations,
                            'n_survival_time_observations': n_survival_time_observations,
                            'proba_blocking_fv': proba_blocking_fv,
                            'finalize_type': finalize_type,
                            'seed': seed
                            })
    time_end = timer()
    exec_time = time_end - time_start
    print("execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))

    return  proba_blocking_fv, integral, expected_survival_time, \
            n_survival_curve_observations, n_survival_time_observations, \
            est_fv

def plot_curve_estimates(df_proba_survival_and_blocking_conditional, dict_params, log=False):
    rhos = [b/d for b, d in zip(dict_params['birth_rates'], dict_params['death_rates'])]
    if log:
        print("arrival rates={}".format(dict_params['birth_rates']))
        print("service rates={}".format(dict_params['death_rates']))
        print("rhos={}".format(rhos))
        print("K={}".format(dict_params['K']))
        print("nparticles={}".format(dict_params['nparticles']))
        print("nmeantimes={}".format(dict_params['nmeantimes']))
        print("maxtime(1 particle)={:.1f}".format(dict_params['maxtime_mc']))
        print("maxtime(N particles)={:.1f}".format(dict_params['maxtime_fv']))
        print("buffer_size_activation={}".format(dict_params['buffer_size_activation']))
        print("mean_lifetime={}".format(dict_params['mean_lifetime']))
        print("#obs for P(T>t) estimation={}".format(dict_params['n_survival_curve_observations']))
        print("#obs for E(T) estimation={}".format(dict_params['n_survival_time_observations']))
        print("Pr(K)={:6f}%".format(dict_params['proba_blocking_fv']*100))
        print("finalize_type={}".format(dict_params['finalize_type'.name]))
        print("seed={}".format(dict_params['seed']))

    plt.figure()
    color1 = 'blue'
    color2 = 'red'
    plt.step(df_proba_survival_and_blocking_conditional['t'], df_proba_survival_and_blocking_conditional['P(T>t / s=1)'],
             'b-', where='post')
    ax = plt.gca()
    ax.set_xlabel('t')
    ax.spines['left'].set_color(color1)
    ax.tick_params(axis='y', color=color1)
    ax.yaxis.label.set_color(color1)
    ax.hlines(0.0, 0, ax.get_xlim()[1], color='gray')
    ax2 = ax.twinx()
    ax2.step(df_proba_survival_and_blocking_conditional['t'], df_proba_survival_and_blocking_conditional['P(BLOCK / T>t,s=1)'],
             'r-', where='post')
    # Set the maximum value of the secondary vertical axis (red curve) 
    #y2max = None
    #y2max = 0.05
    #y2max = 1.0
    #y2max = 1.1*np.max(df_proba_survival_and_blocking_conditional['P(BLOCK / T>t,s=1)'])
    y2max = 1.2*ax2.get_ylim()[1]
    if y2max is not None:
        ax2.set_ylim(0, y2max)
    ax.spines['right'].set_color(color2)
    ax2.tick_params(axis='y', color=color2)
    ax2.yaxis.label.set_color(color2)
    plt.sca(ax)
    ax.legend(['P(T>t / s={})\n(n={})'.format(dict_params['buffer_size_activation'], dict_params['n_survival_curve_observations'])], loc='upper left')
    ax2.legend(['P(BLOCK / T>t,s={})\n(P(BLOCK)={:.6f}%)'.format(dict_params['buffer_size_activation'], dict_params['proba_blocking_fv']*100)], loc='upper right')
    plt.title("K={}, rhos={}, N={}, activation size={}, maxtime(1)={:.1f}, maxtime(N)={:.1f}, mean_lifetime={}(n={})" \
              ", finalize={}, seed={}" \
              .format(dict_params['K'], rhos, dict_params['nparticles'], dict_params['buffer_size_activation'], dict_params['maxtime_mc'], dict_params['maxtime_fv'],
                      dict_params['mean_lifetime'] is not None and "{:.1f}".format(dict_params['mean_lifetime']) or np.nan,
                      dict_params['n_survival_time_observations'],
                      dict_params['finalize_type'].name[0:3],
                      dict_params['seed']
                      ))
    ax.title.set_fontsize(9)


if __name__ == "__main__":
    
    def plot_survival_curve(df_proba_survival, col='b-', title=None):
        plt.step(df_proba_survival['t'], df_proba_survival['P(T>t / s=1)'], col, where='post')
        ax = plt.gca()
        ax.set_xlabel('t')
        if title is None:
            title = "Survival probability given activation"
        plt.title(title)
        ax.hlines(0.0, 0, np.max(df_proba_survival['t']), color='gray')

    def plot_blocking_probability_curve(df_proba_block, color='red', title=None):
        plt.step(df_proba_block['t'], df_proba_block['Phi(t,K / s=1)'], '-', color=color, where='post', linewidth=0.5)
        ax = plt.gca()
        ax.set_xlabel('t')
        if title is None:
            title = "Phi(t): conditional blocking probability given activation"
        plt.title(title)
        ax.hlines(0.0, 0, np.max(df_proba_block['t']), color='gray')

    def plot_curve_with_bands(x, y, error, col='b-', color='blue', xlabel='x', title=None):
        plt.plot(x, y, col)
        plt.fill_between(x, np.where( np.array(y - error) < 0, 0, np.array(y - error)), y + error,
                         alpha=0.5, facecolor=color, antialiased=False) #edgecolor='red', linewidth=4, linestyle='dashdot', antialiased=True)
        ax = plt.gca()
        ax.set_xlabel(xlabel)
        ax.hlines(0.0, 0, ax.get_xlim()[1], color='gray')
        if title is not None:
            plt.title(title)

    #----------------------- Unit tests on specific methods --------------------------#
    time_start = timer()
    seed = 1717
    plotFlag = False
    log = False

    #--- Test #2: simulate_survival()
    if False:
        print("Test #2: simulate_survival() method")
        K = 20
        rate_birth = 0.5
        job_rates = [rate_birth]
        rate_death = [1]
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)
    
        # Simulation
        nparticles0 = 200
        nmeantimes0 = 50
        finalize_type = FinalizeType.REMOVE_CENSORED

        # Monte-Carlo (to estimate expected survival time)
        buffer_size_activations = [1, 5, 8]
        for buffer_size_activation in buffer_size_activations:
            nparticles = nparticles0 #* buffer_size_activation
            nmeantimes = nmeantimes0 #* buffer_size_activation
            print("\nRunning Monte-Carlo simulation on single-server system to estimate survival probability curve for buffer_size_activation={} on N={} particles and simulation time T={}x...".format(buffer_size_activation, nparticles, nmeantimes))
            est_surv = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_rates,
                                                       service_rates=None,
                                                       buffer_size_activation=buffer_size_activation,
                                                       nmeantimes=nparticles*nmeantimes,
                                                       mean_lifetime=None,
                                                       reactivate=False,
                                                       finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                       plotFlag=plotFlag,
                                                       seed=seed, log=log)
            surv_times, surv_counts = est_surv.simulate_survival()
            df_proba_survival = est_surv.estimate_proba_survival_given_activation()

            # Estimate the survival probability starting at 0 to see if there is any difference in the estimation w.r.t. to the above (correct) method
            est_mc = EstimatorQueueBlockingFlemingViot(1, queue, job_rates,
                                                       service_rates=None,
                                                       buffer_size_activation=1,
                                                       nmeantimes=nparticles*nmeantimes,
                                                       mean_lifetime=None,
                                                       reactivate=False,
                                                       finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                       plotFlag=plotFlag,
                                                       seed=seed, log=log)
            est_mc.simulate(EventType.ABSORPTION)
            df_proba_survival_start_at_absorption = est_mc.estimate_proba_survival_given_activation()
            expected_killing_time, _, _ = est_mc.estimate_expected_killing_time()
            expected_survival_time = expected_killing_time  # DM-2021/04/27: This concept of regarding the killing time as the expected survival time is WRONG (clarified at today's conversation with Matt) but I leave it here until the correct implementation is done)  

            plt.figure()
            plot_survival_curve(df_proba_survival_start_at_absorption, col='r-')
            plot_survival_curve(df_proba_survival, title="Buffer size for activation = {}, E(T)={:.1f} (red: MC, start@ABS; blue: start@ACT)".format(buffer_size_activation, expected_survival_time))

    #--- Test #3: variability of survival curve
    if False:
        print("Test #3: compare variability of survival curve among different replications --goal: find out why we get so much variability in the FV estimation of the blocking probability (CV ~ 60%!)")
        ## CONCLUSION:
        K = 20
        rate_birth = 0.5
        job_rates = [rate_birth]
        rate_death = [1]
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)
    
        # Simulation
        nparticles0 = 200
        nmeantimes0 = 50
        finalize_type = FinalizeType.REMOVE_CENSORED

        replications = 12

        # Monte-Carlo (to estimate expected survival time)
        buffer_size_activation = 8
        df_proba_survival = dict()
        for rep in range(replications):
            nparticles = nparticles0 #* buffer_size_activation
            nmeantimes = nmeantimes0 #* buffer_size_activation
            print("\nRunning Monte-Carlo simulation on single-server system to estimate survival probability curve for buffer_size_activation={} on N={} particles and simulation time T={}x...".format(buffer_size_activation, nparticles, nmeantimes))
            est_surv = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_rates,
                                                       service_rates=None,
                                                       buffer_size_activation=buffer_size_activation,
                                                       nmeantimes=nparticles*nmeantimes,
                                                       mean_lifetime=None,
                                                       reactivate=False,
                                                       finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                       plotFlag=plotFlag,
                                                       seed=seed+rep, log=log)
            surv_times, surv_counts = est_surv.simulate_survival()
            df_proba_survival[rep] = est_surv.estimate_proba_survival_given_activation()

        for rep in range(replications):
            plot_survival_curve(df_proba_survival[rep])
        plt.title("Buffer size for activation = {}: survival curve over different replications".format(buffer_size_activation))

    #--- Test #4: variability of Phi(t), the red curve
    if False:
        print("Test #4: compare variability of Phi(t) among different replications --goal: find out why we get so much variability in the FV estimation of the blocking probability (CV ~ 60%!)")
        ## CONCLUSION: The curve has a large variability...
        K = 20
        rate_birth = 0.5
        job_rates = [rate_birth]
        rate_death = [1]
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)

        # Simulation
        nparticles = 200
        nmeantimes = 50
        finalize_type = FinalizeType.ABSORB_CENSORED

        replications = 12

        # Monte-Carlo (to estimate expected survival time)
        buffer_size_activation = 8
        # Dictionary where the different curve estimates will be stored so that we can merge them
        # into the same time axis that is common to all replications which will allow us to average over them
        dict_proba_block = dict()
        colormap = cm.get_cmap("jet")
        for rep in range(replications):
            print("\nRep {} of {}: Running Monte-Carlo simulation on single-server system to estimate the conditional blocking probability curve for buffer_size_activation={} on N={} particles and simulation time T={}x...".format(rep+1, replications, buffer_size_activation, nparticles, nmeantimes))
            est_block = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_rates,
                                                       service_rates=None,
                                                       buffer_size_activation=buffer_size_activation,
                                                       nmeantimes=nmeantimes,
                                                       mean_lifetime=None,
                                                       reactivate=True,
                                                       finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                       plotFlag=plotFlag,
                                                       seed=seed+rep, log=log)
            block_times, block_counts = est_block.simulate_fv()
            df_proba_block = est_block.estimate_proba_blocking_conditional_given_activation()

            if True:
                #print("last t = {:.1f}".format(df_proba_block['t'].iloc[-1]))
                plot_blocking_probability_curve(df_proba_block, color=colormap( rep / replications ))

            # Store the new series in the dictionary of series
            # We convert the series to lists because this is the input type needed by merge_values_in_time() below
            t_new = list( df_proba_block['t'] )
            y_new = list( df_proba_block['Phi(t,K / s=1)'] )
            if rep == 0:
                dict_proba_block['t'] = t_new
                dict_proba_block['y'+str(rep)] = y_new
            if rep > 0:
                # Merge the past series with the new one
                # NOTE that since the past series have already been merged
                # the time values are the same for ALL the series and they contain
                # ALL The time where ANY of the past series change value.
                # This means that the t_merged resulting from merging each of those past series
                # to the new series just observed will be the same!
                for past_rep in range(rep):
                    # Time values that is common for all the series ALREADY aligned
                    t_past_rep = dict_proba_block['t']
                    y_past_rep = dict_proba_block['y'+str(past_rep)]
                    print("\tMerging series #{}, len=({}, {}) with series #{}, len=({}, {})..." \
                          .format(past_rep+1, len(t_past_rep), len(y_past_rep),
                                  rep+1, len(t_new), len(y_new)))
                    t_merged, dict_proba_block['y'+str(past_rep)], y_rep = \
                            merge_values_in_time(t_past_rep, y_past_rep,
                                                      t_new, y_new,
                                                 unique=True)
                    print("\t--> Merged series: len(t)={}".format(len(t_merged)))

                # Update the lists in the dictionary after the last past series has been merged with the new one
                dict_proba_block['t'] = t_merged
                dict_proba_block['y'+str(rep)] = y_rep

        df_proba_block_all_reps = pd.DataFrame({'t': dict_proba_block['t']})
        for rep in range(replications):
            assert len(dict_proba_block['t']) == len(dict_proba_block['y'+str(rep)])
            df_proba_block_all_reps['y'+str(rep)] = dict_proba_block['y'+str(rep)]
        # Compute the pointwise average and standard errors of the curves
        colnames = df_proba_block_all_reps.columns[1:rep+1]
        df_proba_block_all_reps['y_mean'] = np.mean(df_proba_block_all_reps[colnames], axis=1)
        df_proba_block_all_reps['y_se'] = np.std(df_proba_block_all_reps[colnames], axis=1) / np.sqrt(rep)
        if False:
            with printoptions(precision=3, suppress=True):
                print(df_proba_block_all_reps)

        plt.figure()
        plot_curve_with_bands(df_proba_block_all_reps['t'], df_proba_block_all_reps['y_mean'], 2*df_proba_block_all_reps['y_se'], col='r-', color='red',
                              xlabel='t', title="K={}, Buffer size for activation = {}, N={}: Phi(t,K) over different replications".format(K, buffer_size_activation, nparticles))
    #----------------------- Unit tests on specific methods --------------------------#

    
    #------------------------- Unit tests on the process -----------------------------#
    time_start = timer()
    seed = 1717
    plotFlag = False
    log = False

    #--- Test #1.1: One server
    if False:
        print("Test #1: Single server system")
        K = 10
        rate_birth = 0.5
        job_rates = [rate_birth]
        rate_death = [1]
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)
    
        # Simulation
        nparticles = 100
        nmeantimes = 50
        finalize_type = FinalizeType.ABSORB_CENSORED
    
        # a) Monte-Carlo (to estimate expected survival time)
        print("Running Monte-Carlo simulation on 1 particle and T={}x...".format(nparticles*nmeantimes))
        est_mc = EstimatorQueueBlockingFlemingViot(1, queue, job_rates,
                                                   service_rates=None,
                                                   buffer_size_activation=1,
                                                   nmeantimes=nparticles*nmeantimes,
                                                   mean_lifetime=None,
                                                   reactivate=False,
                                                   finalize_info={'type': FinalizeType.REMOVE_CENSORED, 'condition': FinalizeCondition.ACTIVE},
                                                   plotFlag=plotFlag,
                                                   seed=seed, log=log)
        proba_blocking_mc, total_blocking_time, total_survival_time, total_survival_n = est_mc.simulate(EventType.ACTIVATION)
        expected_killing_time, _, _ = est_mc.estimate_expected_killing_time()
        expected_survival_time = expected_killing_time  # DM-2021/04/27: This concept of regarding the killing time as the expected survival time is WRONG (clarified at today's conversation with Matt) but I leave it here until the correct implementation is done)  

        # b) Fleming-Viot
        print("Running Fleming-Viot simulation on {} particles and T={}x...".format(nparticles, nmeantimes))
        est_fv = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_rates,
                                                   service_rates=None,
                                                   buffer_size_activation=1,
                                                   nmeantimes=nmeantimes,
                                                   mean_lifetime=expected_survival_time,
                                                   reactivate=True,
                                                   finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                   plotFlag=plotFlag,
                                                   seed=seed, log=log)
        proba_blocking_fv_integral, proba_blocking_fv_laplacian, integral, expected_survival_time, gamma = est_fv.simulate(EventType.ACTIVATION)
        time_end = timer()
        ## 2021/03/01: 0.3 min
        print("Test #1: execution time: {:.1f} min".format((time_end - time_start) / 60))
    
        # c) Assertions
        print("P(K) by MC: {:.6f}%".format(proba_blocking_mc*100))
        print("P(K) estimated by FV1: {:.6f}%".format(proba_blocking_fv_integral*100))
        print("P(K) estimated by FV2: {:.6f}%".format(proba_blocking_fv_laplacian*100))
        assert("{:.6f}%".format(proba_blocking_mc*100) == "0.032064%")
        assert("{:.6f}%".format(proba_blocking_fv_integral*100) == "0.095885%")
        assert("{:.6f}%".format(proba_blocking_fv_laplacian*100) == "0.000000%")
        # Note: True P(K): 0.048852%

    #--- Test #1.2: One server with full process already taken care of
    if False:
        print("Test #1: Single server system")
        nservers = 1
        K = 20
        rate_birth = 0.5
        job_rates = [rate_birth]
        rate_death = [1]
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)

        env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_rates=job_rates, rewards=[1])

        # Simulation parameters
        dict_params_simul = {
            'nparticles': 800,
            'nmeantimes': 50,
            'buffer_size_activation': int(0.5*K),
            'seed': 1717,
                }

        # Info parameters 
        dict_params_info = {'plot': True, 'log': False}

        print("Computing TRUE blocking probability...")
        proba_blocking_K = compute_blocking_probability_birth_death_process(nservers, K, [rate_birth / rate_death[0]])

        # Run!
        time_start = timer()
        replications = 4
        proba_blocking_fv = np.nan*np.ones(replications)
        for rep in range(replications):
            print("\n***** Running replication {} of {} *****".format(rep+1, replications))
            dict_params_simul['seed'] += rep
            proba_blocking_fv[rep], _, _, _, _, _ = estimate_blocking_fv(env_queue, dict_params_simul, dict_params_info=dict_params_info)
            print("\tEstimated blocking probability: Pr(FV)={:.6f}%".format(proba_blocking_fv[rep]*100))
            print("\tTrue blocking probability: Pr(K)={:.6f}%".format(proba_blocking_K*100))
        time_end = timer()
        dt_end = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        print("Ended at: {}".format(dt_end))
        print("Execution time: {:.1f} min, {:.1f} hours".format((time_end - time_start) / 60, (time_end - time_start) / 3600))

    #--- Test #2: Multi-server
    if False:
        print("\nTest #2: Multiple-server system")
        nservers = 3
        K = 20
        rate_birth = 0.5 # This value is not really used but it's needed to construct the `queue` object
        job_rates = [0.8, 0.7]
        rate_death = [1, 1, 1]
        policy_assign = [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
        queue = queues.QueueMM(rate_birth, rate_death, nservers, K)
    
        # Simulation
        nparticles = 100
        nmeantimes = 50
        finalize_type = FinalizeType.ABSORB_CENSORED

        # a) Monte-Carlo (to estimate expected survival time)
        print("Running Monte-Carlo simulation on 1 particle and T={}x...".format(nparticles*nmeantimes))
        est_mc = EstimatorQueueBlockingFlemingViot(1, queue, job_rates,
                                                   service_rates=None,
                                                   buffer_size_activation=1,
                                                   nmeantimes=nparticles*nmeantimes,
                                                   policy_assign=policy_assign,
                                                   mean_lifetime=None,
                                                   reactivate=False,
                                                   finalize_info={'type': FinalizeType.REMOVE_CENSORED, 'condition': FinalizeCondition.ACTIVE},
                                                   plotFlag=plotFlag,
                                                   seed=seed, log=log)
        proba_blocking_mc, total_blocking_time, total_survival_time, total_survival_n = est_mc.simulate(EventType.ACTIVATION)
        expected_killing_time, _, _ = est_mc.estimate_expected_killing_time()
        expected_survival_time = expected_killing_time  # DM-2021/04/27: This concept of regarding the killing time as the expected survival time is WRONG (clarified at today's conversation with Matt) but I leave it here until the correct implementation is done)  

        # b) Fleming-Viot
        print("Running Fleming-Viot simulation on {} particles and T={}x...".format(nparticles, nmeantimes))
        est_fv = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_rates,
                                                   service_rates=None,
                                                   buffer_size_activation=1,
                                                   nmeantimes=nmeantimes,
                                                   policy_assign=policy_assign,
                                                   mean_lifetime=expected_survival_time,
                                                   reactivate=True,
                                                   finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                   plotFlag=plotFlag,
                                                   seed=seed, log=log)
        proba_blocking_fv_integral, proba_blocking_fv_laplacian, integral, expected_survival_time, gamma = est_fv.simulate(EventType.ACTIVATION)
        time_end = timer()
        ## 2021/03/01: 0.8 min
        print("Test #2: execution time: {:.1f} min".format((time_end - time_start) / 60))
    
        # c) Assertions
        print("P(K) by MC: {:.6f}%".format(proba_blocking_mc*100))
        print("P(K) estimated by FV1: {:.6f}%".format(proba_blocking_fv_integral*100))
        print("P(K) estimated by FV2: {:.6f}%".format(proba_blocking_fv_laplacian*100))
        assert("{:.6f}%".format(proba_blocking_mc*100) == "0.027690%")
        assert("{:.6f}%".format(proba_blocking_fv_integral*100) == "0.082433%")
        assert("{:.6f}%".format(proba_blocking_fv_laplacian*100) == "0.000000%")
        # Note: True P(K): 0.124693% (taken from the actual simulation I perform normally to evaluate the algorithm in test_QB.py)
    
    
    if True:
        #--- Test #3: One server with acceptance policy
        print("\nTest #3: Single server system with ACCEPTANCE policy on different JOB CLASSES")
        K = 10
        rate_birth = 0.5
        job_rates = [0.3, 0.8, 0.7, 0.9]
        rate_death = [1]
        acceptance_rewards = [1, 0.8, 0.3, 0.2]     # One acceptance reward for each job class
        nservers = len(rate_death)
        queue = GenericQueue(K, nservers)
        env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_rates, acceptance_rewards)
        # Acceptance thresholds
        # - There is one threshold defined for each buffer size
        # - Their values are in the range of the job classes (0, ..., #job classes)
        # - Any job with class smaller or equal than the threshold
        # for the buffer size at the moment of the job arrival is accepted by the policy 
        acceptance_thresholds = [2]*(K+1) 
        policy_accept = PolQueueRandomizedLinearStep(env_queue, acceptance_thresholds)
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)

        # Simulation
        nparticles = 3
        nmeantimes = 5
        finalize_type = FinalizeType.ABSORB_CENSORED

        # a) Monte-Carlo (to estimate expected survival time)
        print("Running Monte-Carlo simulation on 1 particle and T={}x...".format(nparticles*nmeantimes))
        est_mc = EstimatorQueueBlockingFlemingViot(1, queue, job_rates,
                                                   service_rates=None,
                                                   buffer_size_activation=1,
                                                   nmeantimes=nparticles*nmeantimes,
                                                   policy_accept=policy_accept,
                                                   policy_assign=None,
                                                   mean_lifetime=None,
                                                   reactivate=False,
                                                   finalize_info={'type': FinalizeType.REMOVE_CENSORED, 'condition': FinalizeCondition.ACTIVE},
                                                   plotFlag=True,
                                                   seed=seed, log=log)
        proba_blocking_mc, total_blocking_time, total_survival_time, total_survival_n = est_mc.simulate(EventType.ACTIVATION)
        expected_killing_time, _, _ = est_mc.estimate_expected_killing_time()
        expected_survival_time = expected_killing_time  # DM-2021/04/27: This concept of regarding the killing time as the expected survival time is WRONG (clarified at today's conversation with Matt) but I leave it here until the correct implementation is done)  

        """
        # b) Fleming-Viot
        print("Running Fleming-Viot simulation on {} particles and T={}x...".format(nparticles, nmeantimes))
        est_fv = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_rates,
                                                   service_rates=None,
                                                   buffer_size_activation=1,
                                                   nmeantimes=nmeantimes,
                                                   policy_accept=policy_accept,
                                                   policy_assign=policy_assign,
                                                   mean_lifetime=expected_survival_time,
                                                   reactivate=True,
                                                   finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                   plotFlag=True,
                                                   seed=seed, log=log)
        proba_blocking_fv_integral, proba_blocking_fv_laplacian, integral, expected_survival_time, gamma = est_fv.simulate(EventType.ACTIVATION)
        """
        time_end = timer()
        ## 2021/03/01: 0.3 min
        print("Test #1: execution time: {:.1f} min".format((time_end - time_start) / 60))
    
        # c) Assertions
        print("P(K) by MC: {:.6f}%".format(proba_blocking_mc*100))
        #print("P(K) estimated by FV1: {:.6f}%".format(proba_blocking_fv_integral*100))
        #print("P(K) estimated by FV2: {:.6f}%".format(proba_blocking_fv_laplacian*100))
        assert("{:.6f}%".format(proba_blocking_mc*100) == "17.815469%")
        #assert("{:.6f}%".format(proba_blocking_fv_integral*100) == "0.095846%")
        #assert("{:.6f}%".format(proba_blocking_fv_laplacian*100) == "0.000000%")
        # Note: True P(K): 0.048852%
    #------------------------- Unit tests on the process -----------------------------#

