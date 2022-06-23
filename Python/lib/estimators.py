# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:12:23 2020

@author: Daniel Mastropietro
@description: Estimators are defined for problems of interest in Reinforcement Learning.
"""

from enum import Enum, unique
import warnings

import copy     # Used to generate different instances of the queue under analysis
import bisect

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm    # cm is for colormaps (e.g. cm.get_cmap())
from timeit import default_timer as timer
from datetime import datetime

DEFAULT_NUMPY_PRECISION = np.get_printoptions().get('precision')
DEFAULT_NUMPY_SUPPRESS = np.get_printoptions().get('suppress')

if __name__ == "__main__":
    # Needed to run tests (see end of program)
    import runpy
    runpy.run_path('../../setup.py')

from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses, rewardOnJobClassAcceptance, ActionTypes
from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners.value_functions import ValueFunctionApprox
from Python.lib.agents.queues import AgeQueue, PolicyTypes as QueuePolicyTypes
from Python.lib.agents.policies.job_assignment import PolJobAssignmentProbabilistic
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStepOnJobClasses, PolQueueTwoActionsLinearStep
import Python.lib.queues as queues  # The keyword `queues` is used in test code
from Python.lib.queues import Event, GenericQueue

from Python.lib.utils.basic import  array_of_objects, find, find_last_value_in_list, insort, \
                                    list_contains_either, merge_values_in_time, measure_exec_time
from Python.lib.utils.computing import get_server_loads, compute_job_rates_by_server, \
    compute_blocking_probability_birth_death_process, \
    stationary_distribution_birth_death_process, \
    stationary_distribution_birth_death_process_at_capacity_unnormalized

DEBUG_ESTIMATORS = False
DEBUG_TIME_GENERATION = False
DEBUG_SPECIAL_EVENTS = False
DEBUG_TRAJECTORIES = False

# The constant multiplier used to decide whether an analyzed measurement is "large"
# A use example is: determine whether the last chunk of simulation time from the latest valid special event
# (e.g. the latest absorption if we are interested in estimating E(T)) is large compared to the average measures so far
# (e.g. the average time between absorption measured before the particle is censored)
# This "large" condition is used to determine whether the last observed trajectory chunk should be used or not
# to estimate quantities of interest.
MULTIPLIER_FOR_LARGE_MEASUREMENT = 2

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
    """
    Condition indicating that, if a particle is under one of these situations when the simulation ends,
    it means that they should be treated by the finalize() method before calculations leading to estimated
    quantities can be carried out.
    
    For example, if the FinalizeCondition is ACTIVE, something needs to be done to the particle that is active
    at the end of the simulation in order to consider it in the estimation calculations. For instance,
    a fictitious absorption could be added to the trajectory, or the whole trajectory since the last absorption
    could be removed.
    """
    ACTIVE = 1                          # The particle is still active, i.e. it has not been absorbed. To be used in the FV process.
    NOT_ABSORBED_STATES_BOUNDARY = 2    # The particle is not at the boundary of the set of absorbed states. To be used to estimate E(T).
    NOT_START_POSITION = 3              # The particle is not at the buffer size position where it started. To be used in the MC estimation.


class EstimatorValueFunctionOfflineDeterministicNextState:
    """
    Offline estimator of the value function on a discrete-state / discrete-action environment where the next state
    given an action is deterministic.

    ASSUMPTIONS:
    - All possible actions are the same for each state and equal to the number of actions in the environment.
    - All possible actions are equally likely (random walk policy).

    The offline estimator consists in traversing all possible states and actions in the environment and updating
    the value function recursively using the Bellman equation on the state value function.

    This estimator is useful when there is no theoretical expression for the state value function.

    Arguments:
    env: EnvironmentDiscrete
        Discrete-state and discrete-action environment on which the state value function is estimated.

    gamma: float
        Discount parameter when learning the state value function. This is used in the Bellman equation.
    """
    def __init__(self, env, gamma=1.0):
        self.env = env
        self.gamma = gamma
        self.V = ValueFunctionApprox(self.env.getNumStates(), self.env.getTerminalStates())

    def reset(self, reset_method, reset_params, reset_seed):
        self.env.reset()
        self.V.reset(method=reset_method, params_random=reset_params, seed=reset_seed)

    def estimate_state_values_random_walk(self, synchronous=True,
                                          max_delta=np.nan, max_delta_rel=1E-3, max_iter=1000, verbose=True, verbose_period=None,
                                          reset_method=ResetMethod.ALLZEROS, reset_params=None, reset_seed=1713):
        "Estimates the value function under the random walk policy"
        self.reset(reset_method, reset_params, reset_seed)

        if verbose_period is None:
            verbose_period = max_iter / 10

        print("\n\nTerminal states ({} out of {}): {}".format(len(self.env.getTerminalStates()), self.env.getNumStates(), self.env.getTerminalStates()))
        # WARNING: Only valid for MountainCarDiscrete environment
        #print("Positions: {}".format(self.env.get_positions()))
        #print("Velocities: {}".format(self.env.get_velocities()))
        # WARNING: Only valid for MountainCarDiscrete environment
        print("Initial V(s) estimate: {}".format(self.V.getValues()))

        max_deltaV_abs = np.Inf
        max_deltaV_rel_abs = np.Inf
        iter = 0
        while iter < max_iter and \
                (np.isnan(max_delta_rel) and max_deltaV_abs > max_delta or \
                 np.isnan(max_delta) and max_deltaV_rel_abs > max_delta_rel or \
                 not np.isnan(max_delta) and not np.isnan(max_delta_rel) and max_deltaV_abs > max_delta and max_deltaV_rel_abs > max_delta_rel):
            iter += 1
            values_prev = self.V.getValues()
            for s in self.env.getNonTerminalStates():
                #print("** state: {}".format(s))
                self.env.setState(s)
                n_actions_so_far = 0
                # Initialize the average observed value over all possible actions, whose value will be the updated V(s) value
                # once all actions have been taken
                V_mean_over_actions = 0.0
                for a in range(self.env.getNumActions()):
                    assert self.env.getState() == s, "getState(): {}, s: {}".format(self.env.getState(), s)
                    ns, reward, done, info = self.env.step(a)   # ns = next state

                    # NOTE 1: (2022/06/05) THIS ASSUMES THAT THE VALUE FUNCTION APPROXIMATION IS TABULAR!!
                    # In fact, I don't know how to write the Bellman equation in function approximation context
                    # (See chapter 9 in Sutton but I don't think it talks about this... it only talks about how to update
                    # the weights at each iteration, using Stochastic Gradient Descent (SGD).
                    # However, it does talk about the fixed point of this SGD algorithm, which is w = A^-1 * b
                    # where A and b are given in that chapter (pag. 206).

                    # NOTE 2: At first we should perhaps update V(s) synchronosly, i.e. keep the same V(s) on the RHS
                    # until ALL states are updated.
                    # However, it seems that the asynchronous update done here works fine as well (recall class by Matt at UBA)
                    # and it even converges faster!
                    if DEBUG_ESTIMATORS:
                        print("state: {}, action: {} => next state = {}, reward = {} (done? {})".format(s, a, ns, reward, done))
                        # WARNING: Only valid for MountainCarDiscrete environment (because of call to self.env.get_state_from_index()
                        #print("state: {} ({}), action: {} => next state = {} ({}), reward = {} (done? {})" \
                        #      .format(s, self.env.get_state_from_index(s), a, ns, self.env.get_state_from_index(ns), reward, done))
                        # WARNING: Only valid for MountainCarDiscrete environment

                    # The new V(s) value is the average over all possible actions (since we are considering a random walk)
                    if synchronous:
                        # Use the state value computed at the PREVIOUS iteration as currently known value of the next state, V(ns)
                        V_observed = reward + self.gamma * values_prev[ns]
                    else:
                        # Use the CURRENT value of the next state, V(ns), even if it has been updated already in this iteration
                        # (i.e. without waiting for the value of all other states to be updated)
                        V_observed = reward + self.gamma * self.V.getValue(ns)
                    V_mean_over_actions = (n_actions_so_far * V_mean_over_actions + V_observed) / (n_actions_so_far + 1)

                    n_actions_so_far += 1

                    # Reset the state to the original state before the transition, so that the next action is applied to the same state
                    self.env.setState(s)
                # Update the state value of the currently analyzed state, V(s)
                self.V.setWeight(s, V_mean_over_actions)
                if DEBUG_ESTIMATORS:
                    if reward != 0:
                        print("--> New value for state s={} after taking all {} actions: {}".format(s, self.env.getNumStates(), self.V.getValue(s)))

            deltaV = (self.V.getValues() - values_prev)
            deltaV_rel = np.array([0.0  if dV == 0
                                        else dV / abs(V) if V != 0
                                                         else np.Inf
                                    for dV, V in zip(deltaV, values_prev)])
            mean_deltaV_abs = np.mean(np.abs(deltaV))
            max_deltaV_abs = np.max(np.abs(deltaV))
            max_deltaV_rel_abs = np.max(np.abs(deltaV_rel))
            if DEBUG_ESTIMATORS or verbose and (iter-1) % verbose_period == 0:
                print("Iteration {}: mean(|V_prev|) = {:.3f}, mean(|V|) = {:.3f}, mean|delta(V)| = {:.3f}, max|delta(V)| = {:.3f}, max|delta_rel(V)| = {:.7f}" \
                      .format(iter, np.mean(np.abs(values_prev)), np.mean(np.abs(self.V.getValues())), mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs))

        if  not np.isnan(max_delta) and max_deltaV_abs > max_delta or \
            not np.isnan(max_delta_rel) and max_deltaV_rel_abs > max_delta_rel or \
            not np.isnan(max_delta) and not np.isnan(max_delta_rel) and (max_deltaV_abs > max_delta or max_deltaV_rel_abs > max_delta_rel):
            warnings.warn("The estimation of the value function may not be accurate as the maximum relative absolute" \
                          " change in the last iteration #{} ({}) is larger than the maximum allowed ({})" \
                          .format(iter, max_deltaV_rel_abs, max_delta_rel))

        return iter, mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs

    def getV(self):
        return self.V


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

    job_class_rates: list
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

    nmeantimes: (opt) float
        Multiple of the mean time of the job class with LOWEST arrival rate defining the maximum simulation time.
        When None, it is expected that parameter finalize_info['maxevents'] is less than Inf, o.w. the simulation
        cannot start as there is no criterion for the simulation to stop.
        When both `nmeantimes` and finalize_info['maxevents'] < Inf, the stopping condition is based only on the latter. 
        Ex: If there are two job classes with arrival rates respectively 2 jobs/sec and 3 jobs/sec,
        the maximum simulation time is computed as `nmeantimes * 1/2` sec, which is expected to include
        `nmeantimes` jobs of the lowest arrival rate job on average.
        For instance, if nmeantimes = 10, the simulation time is 5 sec and is expected to include 10 jobs
        of the lowest arrival rate class, and `(nmeantimes * 1/2) / (1/3) = 5*3` = 15 jobs of the
        highest arrival rate class.
        default: None

    burnin_cycles_absorption: (opt) int
        Number of absorption cycles to let pass before we use the time to absorption as
        a valid value to estimate the expected absorption cycle time E(T),
        and the survival probability P(T>t / activation).
        This is used to "guarantee" that the system is in stationary regime when measuring the time to absorption
        used in the estimation of the above quantities.
        Recall that convergence to stationary regime has exponential speed in time.
        For this estimation to take place, we require that this parameter is positive. The reason for this requirement
        is that we want to ensure that the reference time from which the time to absorption is observed
        is an ACTUAL absorption cycle time --as opposed to being the initial simulation time. In this way
        we guarantee that the absorbed state is an ACTUAL absorption state, i.e. one that follows the
        distribution of absorption states when absorption happens.
        default: 0, which means that E(T) and P(T>t) are NOT estimated in this simulation if reactivate=True

    burnin_cycles_complete_all_particles: (opt) bool
        Whether ALL particles are required to have their burn-in cycles complete
        before starting the Fleming-Viot process.
        Set this value to True when all particles should be used for the estimation
        of the expected return time to absorption and of the survival probability given activation.
        default: False

    reward_func: function
        Function returning the reward received when blocking occurs as a function of the buffer size.
        default: None, in which case the reward received when blocking is always 1.

    policy_accept: (opt) Acceptance Policy
        Policy that defines the acceptance of an arriving job to the queue.
        It is expected to define a queue environment of type EnvQueueSingleBufferWithJobClasses
        on which the policy is applied that defines the rewards received for accepting or rejecting
        a job of a given class.
        The queue environment should have at least the following methods defined:
        - getNumJobClasses()
        - setJobClass() which sets the class of the arriving job
        default: None (which means that all jobs are accepted except when the queue is at its full capacity)

    policy_assign: (opt) Assignment Policy
        Policy to assign jobs to servers that is based on a probabilistic approach.
        That is, each job class (whose arrival rates are defined in parameter `job_class_rates`) has a probability
        of being assigned to a server in the queue.
        Ex: In a scenario with 2 job classes and 3 servers, the following policy assigns job class 0
        to server 0 or 1 with equal probability and job class 1 to server 1 or 2 with equal probability:
        PolJobAssignmentProbabilistic( [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]] )
        default: None (in which case the assignment probability is uniform over the servers)

    mean_lifetime: (opt) positive float
        Mean particle lifetime to be used as the expected absorption cycle time E(T)
        when estimating the blocking probability with Fleming-Viot approach.
        This is useful when reactivate=True and burnin_cycles_absorption=0,
        as in that case expected absorption cycle time E(T) is NOT computed from
        a burn-in period of the simulation process.
        default: None

    proba_survival_given_activation: (opt) pandas data frame
        A separate estimation of the probability of survival given the process started at the activation set:
        Pr(T>t / s in activation set).
        Typically this estimation is obtained by running the process in "Monte-Carlo mode",
        i.e. with parameter reactivate=False.
        It should be a pandas data frame with at least two columns:
        - 't': the times at which the survival probability is measured. 
        - 'P(T>t)': the survival probability for each t.

    reactivate: (opt) bool
        Whether to reactivate a particle after absorption to a positive position.
        Note that absorption occurs when the particle goes to position (0, 0, ..., 0)
        where the array of zeros has length equal to the number of servers.
        That is, the particle is an ARRAY of server sizes, which should ALL be idle
        in order for the particle to be considered absorbed.
        
        When reactivate=True, the initial position of the system is (1, 0, ..., 0),
        i.e. the initial buffer size is 1 (as the system can never be absorbed)
        When reactivate=False, the initial position of the system is (0, 0, ..., 0).
        This setting is used when we are interested in estimating the expected absorption cycle time
        (which is used as input to the system run under reactivate=True to estimate
        the blocking probability under the Fleming-Viot algorithm). 
        default: True

    finalize_info: (opt) dict
        Dictionary with the following two attributes:
        - 'maxevents': maximum number of events to consider in the simulation. If < Inf,
        the simulation stops when this number of events is reached among ALL the simulated particles,
        and at that point the finalize process is triggered.
        default: Inf
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
            - FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY: the particle is not at the
            boundary of the set of absorbed states. To be used to estimate the expected
            absorption cycle time, E(T), as a full cycle is observed at the time the
            particle is absorbed, which means that, in order for the particle to end
            at completion of a cycle, the latest observed event time must be an absorption,
            making the particle be at a atate at the boundary of the set of absorbed states. 
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
    def __init__(self, nparticles: int, queue, job_class_rates: list,
                 service_rates=None,
                 buffer_size_activation=1, positions_observe: list=[],
                 nmeantimes=None,
                 burnin_cycles_absorption=0,
                 burnin_cycles_complete_all_particles=False,
                 reward_func=None,
                 policy_accept=None,
                 policy_assign=None,
                 mean_lifetime=None, proba_survival_given_activation=None,
                 reactivate=True,
                 finalize_info={'maxevents': np.Inf, 'maxtime': np.Inf, 'type': FinalizeType.ABSORB_CENSORED, 'condition': FinalizeCondition.ACTIVE},
                 seed=None, plotFlag=False, log=False):

        #--------------------------------- Parse input parameters -----------------------------
        if reactivate and nparticles < 2:
            raise ValueError("The number of particles must be at least 2 when reactivate=True ({})".format(nparticles))
            import sys
            sys.exit(-1)
        self.N = int( nparticles )                              # Make sure the number of particles is of int type
        self.queue = copy.deepcopy(queue)                       # We copy the queue so that we do NOT change the `queue` object calling this function
        self.job_class_rates = job_class_rates                  # This should be a list of arrival rates for each job class (which are the indices of this list
        self.nservers = self.queue.getNServers()
        self.buffer_size_activation = buffer_size_activation    # Buffer size defining the ACTIVATION set of server sizes (n1, n2, ..., nR)

        if not isinstance(positions_observe, list):
            # Convert it to a list
            positions_observe = [ positions_observe ]
        if positions_observe == []:
            # If the given position (buffer size) to observe is empty, we observe the absorption buffer size
            # because that's normally the buffer size of interest (e.g. to estimate E(T), the return time to the absorption set)
            self.positions_observe = [buffer_size_activation - 1]# List of positions to observe, i.e. to record every visit to any of these positions (this is used to estimate the return time to the set of said positions)
        else:
            self.positions_observe = positions_observe

        # Activation and absorption sets
        self.set_activation = { self.buffer_size_activation }       # Set of the ACTIVATION buffer sizes, which defines the boundary set of states between the absorption state set and the rest of states. Note that this is part of the REST of the states, not part of the absorption set of states.
        self.set_absorbed_states = set( range(buffer_size_activation) )  # Set of ABSORBED BUFFER SIZES (note that it does NOT include the activation buffer size), whose possible R-length states (n1, n2, ..., nR) (R=nservers) are used as starting points to estimate the expected absorption cycle time

        # Policies
        # TODO: (2021/12/09) Set the acceptance policy to accept everywhere except at the queue's capacity when the policy is None
        # Note that in order to do this we need to pass the queue ENVIRONMENT (from environments/queues.py),
        # not just simply the queue object (from queues/__init__.py)
        #if policy_accept is None:
        #    policy_accept = PolQueueTwoActionsLinearStep(env_queue, theta=env_queue.getCapacity())
        self.policy_accept = policy_accept          # Policy for acceptance of new arriving job of a given class
        if policy_assign is None:                   # Assignment policy of a job to a server. It should be a mapping from job class to server.
            # When no assignment probability is given, it is defined as uniform over the servers for each job class
            policy_assign = PolJobAssignmentProbabilistic([ [1.0 / self.queue.getNServers()] * self.queue.getNServers() ] * len(self.job_class_rates))
        self.policy_assign = policy_assign

        # Update the birth and death rates of the given queue
        equivalent_birth_rates = compute_job_rates_by_server(self.job_class_rates, self.nservers, self.policy_assign.getProbabilisticMap())
        self.queue.setBirthRates(equivalent_birth_rates)
        if service_rates is not None:
            self.queue.setDeathRates(service_rates)

        # Finalize info
        finalize_info['maxevents'] = finalize_info.get('maxevents', np.Inf)
        finalize_info['maxtime'] = finalize_info.get('maxtime', np.Inf)
        finalize_info['type'] = finalize_info.get('type', FinalizeType.ABSORB_CENSORED)
        finalize_info['condition'] = finalize_info.get('condition', FinalizeCondition.ACTIVE)

        # Simulation time
        self.nmeantimes = nmeantimes
        if self.nmeantimes is not None:
            # The max simulation time is computed as the minimum between:
            # - the max simulation time explicitly given by the user via the 'maxtime' attribute of the finalize_info array
            # - a multiple (given by the user) of the maximum inter-arrival time (i.e. of the arrival time
            # of the less frequently arriving job class).
            self.maxtime = np.min( (finalize_info['maxtime'], self.nmeantimes * np.max( 1/np.array(self.job_class_rates)) ) )
        elif finalize_info['maxtime'] != np.Inf:
            self.maxtime = finalize_info['maxtime']
        else:
            # In this case, it is assumed that the 'maxevents' attribute of the finalize_info= parameter is finite,
            # o.w. the simulation would never stop.
            # This is checked below and if the condition is not satisfied, the process stops.
            self.maxtime = None

        # Maximum simulation time allowed for the burn-in period when such burn-in period is used.
        # This is used to estimate the time of the absorption cycle when no valid absorption cycle
        # has yet been observed (because not enough burn-in cycles have been observed to estimate it
        # under stationarity assumptions).
        if burnin_cycles_absorption > 0:
            self.maxtime_burnin = self.maxtime
        #--------------------------------- Parse input parameters -----------------------------


        #------------------------------------ Parameter checks --------------------------------
        if self.queue.getCapacity() <= 1:
            # This condition is checked because a capacity of 1 is NOT of interest, as it coincides with the
            # minimum possible buffer size for which activation occurs.
            raise ValueError("The maximum position of the particles must be larger than 1 ({})".format(self.queue.getCapacity()))

        if self.buffer_size_activation >= self.queue.getCapacity() - 1:
            # Note: normally we would accept the activation size to be equal to Capacity-1...
            # HOWEVER, this creates problem when the particle is unblocked because the UNBLOCK event
            # will NOT be recorded and instead an ACTIVATION event will be recorded.
            # So, we want to avoid this extreme situation.
            raise ValueError("The buffer size for activation ({}) must be smaller than the queue's capacity-1 ({})".format(self.buffer_size_activation, self.queue.getCapacity()-1))

        if self.policy_accept is not None and self.policy_accept.env.getNumJobClasses() != len(self.job_class_rates):
            raise ValueError("The number of job classes defined in the queue environment ({}) " \
                             "must be the same as the number of job rates given as argument to the constructor ({})" \
                             .format(self.policy_accept.env.getNumJobClasses(), len(self.job_class_rates)))

        if reactivate and burnin_cycles_absorption == 0 and mean_lifetime is None:
            raise ValueError("Parameter `mean_lifetime` must be provided by the user when reactivate=True and burnin_cycles_absorption=0." + \
                             "\nEither pass this value or set burnin_cycles_absorption to a positive integer in order to have it estimated by the process.")

        if mean_lifetime is not None and (np.isnan(mean_lifetime) or mean_lifetime <= 0):
            raise ValueError("The mean lifetime must be positive ({})".format(mean_lifetime))

        if proba_survival_given_activation is not None \
            and not isinstance(proba_survival_given_activation, pd.DataFrame) \
            and 't' not in proba_survival_given_activation.columns \
            and 'P(T>t)' not in proba_survival_given_activation.columns:
            raise ValueError("The proba_survival_given_activation parameter must be a data frame having 't' and 'P(T>t)' as data frame columns")

        if  finalize_info['condition'] in [FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY, FinalizeCondition.NOT_START_POSITION] and \
            finalize_info['type'] == FinalizeType.ABSORB_CENSORED:
            raise ValueError("The finalize type CANNOT be {} when the finalize condition is {} or {}".format(FinalizeType.ABSORB_CENSORED, FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY, FinalizeCondition.NOT_START_POSITION))

        if finalize_info['maxevents'] == np.Inf and finalize_info['maxtime'] == np.Inf and nmeantimes is None:
            raise ValueError("Either the `nmeantimes` parameter must not be None" \
                             " or the finalize_info['maxevents'] parameter must be finite" \
                             " or the finalize_info['maxtime'] parameter must be finite.")
        #------------------------------------ Parameter checks --------------------------------

        self.reactivate = reactivate        # Whether the particle is reactivated in order to carry out the Fleming-Viot process
        self.burnin_cycles_absorption = burnin_cycles_absorption
        self.burnin_cycles_complete_all_particles = burnin_cycles_complete_all_particles
        self.mean_lifetime = mean_lifetime  # A possibly separate estimation of the expected absorption cycle time having started at the absorption set (based on the ENTRY distribution of such set)
                                            # NOTE that the expected absorption cycle time, when estimated by this process, is called
                                            # self.expected_absorption_time (see the reset_results() method below).
        self.reward_func = reward_func
        self.is_expected_absorption_time_estimated_from_burnin = self.reactivate and self.mean_lifetime is None and self.burnin_cycles_absorption > 0
            ## NOTE that we need to require that the number of burn-in absorption cycles be > 0 in order to estimate E(T)
            ## because o.w. we are NOT starting at an absorption state that follows the (stationary) distribution
            ## of entry to the absorption set, as this is observed only after the FIRST absorption
            ## (which implies that we can only measure a contribution to E(T) after the first absorption
            ## thus implying that burnin_cycles_absorption > 0).
        self.proba_survival_given_activation = proba_survival_given_activation  # A possibly separate estimation of the probability of survival having started at the activation set (based on the ENTRY distribution of such set)
                                                                                # NOTE that the survival probability, when estimated by this process, is called
                                                                                # self.proba_surv_by_t (see the reset_results() method below).
        self.is_proba_survival_estimated_from_burnin = self.reactivate and self.proba_survival_given_activation is None
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
        self.states_absorbed, self.dist_absorption_set = stationary_distribution_birth_death_process(self.nservers, self.buffer_size_activation-1, self.rhos)
        self.states_absorption, self.dist_absorption = stationary_distribution_birth_death_process_at_capacity_unnormalized(self.nservers, self.buffer_size_activation-1, self.rhos)
        # Normalize the distribution on the absorption states so that it is a probability from which we can sample
        self.dist_absorption = self.dist_absorption / np.sum(self.dist_absorption)
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

        #------------------------------------- Basic information ------------------------------
        # Reset the total number of events counter
        # Note that this is different from the information conveyed by self.iterations
        # which is a list over all particles, as the latter keeps track of the number of iterations
        # or events a particle has been through and IS RESET WHENEVER A PARTICLE IS REACTIVATED
        # (because that's how it is designed, and it's done like that so that we can run a few assertions
        # --see self.assertTimeMostRecentEventAndTimesNextEvents())
        self.nevents = 0

        # For the FV estimation case, when E(T) and P(T>t) are estimated in the same simulation
        # (i.e. when burnin_cycles_absorption > 0) keep track of whether the burn-in period has ended
        # for each particle. This is important because we need to know whether we must still generate
        # the trajectory until the first absorption before the FV process starts or not (we do need to generate
        # such trajectory ONLY when there is still time available AFTER the burn-in period has been completed,
        # but this may not be the case if no ACTIVATION has occurred after the burn-in period.
        # For more details, see the generate_trajectories_at_startup() function. 
        self.has_burnin_period_ended = [False]*self.N

        # Variable the keeps track whether the finalize process (at the end of the simulation)
        # has been carried out or not
        self.is_simulation_finalized = False
        self.reset_results()

        #-- Initialize the particles in the system
        # Initialize the first particle to the queue stored in the object
        self.particles = [self.queue]
        # Add copies of the given queue if N > 1, so that we can simulate several particles
        self.particles += [ copy.deepcopy(self.queue) for _ in np.arange(self.N-1) ]
        # Keep track of the iterations run for each particle (just for information purposes)
        # Note: an iteration is defined by the application of an event to ONE server
        # (typically the server with the smallest observed event time)
        self.iterations = np.zeros(self.N, dtype=int)
        # The set of particles to simulate at startup is ALL the particles
        # This set might be reduced later one when the FV reactivation process starts
        # because not all particles may be ready for the FV process, namely those that have not been
        # reactivated AND absorbed for the number of times required by the burn-in period.
        # Note that we also require a particle to be ABSORBED --not only ACTIVATED-- in order to be
        # considered for the FV process because such absorption time contributes to the definition
        # of the max simulation time of the FV process.   
        self.particles_to_simulate = list(range(self.N))   # This is [0, 1, ..., N-1]
        #------------------------------------- Basic information ------------------------------


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
        self.particle_reactivation_ids = copy.deepcopy(self.particles_to_simulate)

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
        # - one that keeps track of the positions and times of each particle's server
        # - one that keeps track of the positions and times of the system's SINGLE buffer
        # In each of these structures, the entry is a list that may have different length
        # depending on the particle and server within each particle.
        # NOTE: The difference between the all_times_by_server array and the trajectories array
        # is that the trajectories array will NEVER have a repeated time value and therefore can be used
        # to compute statistics necessary for the estimations of the blocking probability.
        # On the other hand, the all_times_by_server array has repeated times when the particle is absorbed
        # and reactivated, as the absorption time and reactivation time coincide and are both stored in the attribute
        # (for information purposes, in case we need to analyze something) 
        self.all_positions_by_server = np.empty((self.N, self.nservers), dtype=list)
        self.all_times_by_server = array_of_objects((self.N, self.nservers), dtype=list, value=[0.0])
        self.all_positions_buffer = array_of_objects((self.N,), dtype=list)
        self.all_times_buffer = array_of_objects((self.N,), dtype=list, value=[0.0])
        #---------------------------------- Trajectory information ----------------------------


        #---------------------------------- Next events information ---------------------------
        # Attributes used to store information about the (possible) next events to be applied

        #-- Servers, types and times of next events (one per particle)
        # NOTE: The plural in the variable names (e.g. 'servers', 'times') refer to the fact they contain info about particleS
        # All event times are ABSOLUTE, i.e. the exponential time is generated for each server
        # (which is a RELATIVE time --relative to the previous event) and this relative time is converted to
        # absolute by adding the time of the previous event. Working with absolute times facilitates the process.
        self.servers_next_event = -1 * np.ones(self.N, dtype=int)    # In what server will the next event happen; note that we initialize the unknown value as -1 (instead of e.g. np.nan) because we want the array to store integer values, not float
        self.types_next_event = np.empty(self.N, dtype=Event)
        self.times_next_event = np.nan*np.ones(self.N, dtype=float)
        self.times_next_events2 = np.zeros((self.N, self.nservers, 2), dtype=float) # `2` is the number of possible events: birth and death
        self.times_next_birth_events = np.nan*np.ones((self.N, self.nservers), dtype=float)
        self.times_next_death_events = np.nan*np.ones((self.N, self.nservers), dtype=float)

        #-- Arriving jobs information
        # Times and classes associated to jobs in the queue of each server
        # It is important to store this information in order to properly handle the situation when
        # different arriving jobs are assigned to the SAME server in a given iteration
        # (see method _generate_birth_times())
        # NOTE: The first element in the queue of each server has information about the first WAITING job
        # (as opposed to containing information about the job CURRENTLY BEING SERVED)
        self.times_in_server_queues = array_of_objects((self.N, self.nservers), dtype=list, value=[])
        self.jobclasses_in_server_queues = array_of_objects((self.N, self.nservers), dtype=list, value=[])
        #---------------------------------- Next events information ---------------------------


        #--------------------------- Times of the latest arrived jobs -------------------------
        # Times (ABSOLUTE) of the latest jobs arrived for each class
        # These are needed in order to compute the ABSOLUTE time of the next arriving job of each class
        # since the arrival times are generated RELATIVE to the time of the latest arrival.
        # The times of the latest jobs arrived per class is also used to decide WHEN new arriving job times
        # should be generated and thus make the process more efficient (as opposed to generating new arriving job times
        # at every iteration (i.e. at every call to generate_one_iteration()) which may quickly fill
        # up the server queues with times that will never be used, significantly slowing down the simulation process)        
        self.times_latest_arrived_jobs = np.zeros((self.N, len(self.job_class_rates)), dtype=float)
        #--------------------------- Times of the latest arrived jobs -------------------------


        #----------------------------- Activation states distribution -------------------------
        self.dist_activation_states = [0] * len(self.states_activation)
        #----------------------------- Activation states distribution -------------------------


        #---------------------------------- Absorption times ----------------------------------
        #-- Distribution of absorption states used to analyze if the observed distribution at the end of the simulation looks reasonable
        self.dist_absorption_states = [0] * len(self.states_absorption)

        #-- Information about the absorption times, which is used to complete the simulation after each absorption
        # This dictionary initially (i.e. before any re-activation of a particle) contains the
        # FIRST absorption time of each particle. Recall that there is ONE absorption time per particle.
        # Once the reactivation process starts, the dictionary contains every new absorption time observed.
        # For more information see the generate_trajectories_until_end_of_simulation() method or similar.
        self.dict_info_absorption_times = dict({'t': [],        # List of absorption times (one for each particle) (they should be stored SORTED, so that reactivation starts with the smallest absorption time)
                                                'P': [],        # List of particle numbers associated to each absorption time in 't'
                                                'S': [],        # List of absorption state indices (these indices represent one of the possible states returned by the function computing the distribution of the states at the boundary of the absorption set)
                                                })
        #---------------------------------- Absorption times ----------------------------------


        #----------------- Attributes for survival and blocking time calculation --------------
        # Arrays that are used to estimate the expected absorption cycle time (time between two consecutive absorptions)
        # These times are stored as ABSOLUTE, i.e. NOT relative to the particles' time origin
        self.ktimes_sum = np.zeros(self.N, dtype=float)    # Times to absorption from latest time it changed to an absorption state
        self.ktimes_n = np.zeros(self.N, dtype=int)        # Number of times the particles were absorbed
        self.ktimes = np.nan*np.ones(self.N, dtype=float)  # Array to store the latest observed killing time
        self.absorption_cycle_times = np.nan*np.ones(self.N, dtype=float)   # Time between two consecutive absorptions for each particle used when E(T) is estimated on the burn-in period (which happens when burnin_cycles_absorption > 0)

        # Arrays that are used to estimate the blocking proportion as a rough estimate of the blocking probability
        # These times are stored RELATIVE to the particles' time origin
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


        #------------ Attributes for recording return times to observe position ---------------
        # These times are stored as ABSOLUTE, i.e. NOT relative to the particles' time origin 
        self.rtimes_offset = np.zeros(self.N, dtype=float)      # Offset to substract for every new contribution of a return time to an observed position
        self.rtimes_obs_sum = np.zeros(self.N, dtype=float)     # Total return time for each particle
        self.rtimes_obs_n = np.zeros(self.N, dtype=int)         # Number of observed return times for each particle
        #------------ Attributes for recording return times to observe position ---------------

        # Times by particle at which the FV process starts (i.e. after the burn-in period has passed)
        # This is used to measure the time to absorption from the latest ACTIVATION event
        # in the all-in-one estimation process, i.e. when using the first absorption following the first activation
        # observed after the burn-in period has been completed.
        self.times_start_fv = np.nan*np.ones(self.N, dtype=float)

        # Latest time at which we know the state of the system for each particle.
        # This means, that for all times <= this time we know precisely the size of ALL servers in the system.
        self.times_latest_known_state = np.zeros(self.N, dtype=float)

        if self.LOG:
            print("Particle system with {} particles has been reset:".format(self.N))
            print("")

    def reset_results(self):
        "Resets the results of the simulation"
        self.t = None                           # List containing the times at which the probabilities by time are estimated
        self.proba_surv_by_t = None             # List the same length as t containing S(t / s=1) = 'P(T>t)', where s=1 represents an entry state to the activation set
        self.proba_block_by_t = None            # List the same length as t containing P(BLOCK / T>t,s=1) = 'P(BLOCK / T>t)', where s=1 represents an entry state to the activation set
        self.expected_absorption_time = np.nan  # Expected absorption cycle time (time between two consecutive absorptions)
        self.n_expected_absorption_time = None  # Number of cases used to estimate the expected absorption cycle time
        self.proba_blocking = np.nan            # Blocking probability (the ultimate goal of this simulation!)

    def reset_positions(self, start_event_type, uniform_random_choice=False, N_min :int=None):
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

        uniform_random_choice: boolean
            Whether the start state should be chosen uniformly random from all possible start states.
            default: False

        N_min: (opt) int
            Minimum number of particles to simulate in each start state.
            When this parameter is not None, the number of particles in each start state
            is FIXED --as opposed to randomly chosen using the stationary distribution on the states.
            WARNING: If the number of possible states is too large the total number of particles N
            to guarantee the minimum value N_min may be too large and possibly trigger an OUT-OF-MEMORY error. 
            default: None 

        Return: list
        List containing the number of particles to simulate for each feasible start state
        defined by the distribution of the set of states defined by the `start_event_type`
        (e.g. ABSORPTION or ACTIVATION states).  
        """
        np.random.seed(self.seed)

        if start_event_type == EventType.ABSORPTION:
            # NOTE: (2021/05/04) This distribution is NOT the one to be used for the absorption states
            # because it measures the stationary distribution of the states at the BOUNDARY of the absorption set
            # and what we should actually be using is the distribution of the states when the process ENTERS the absorption set.
            # The latter distribution is not so easy to compute even in reasonably easy queue systems such as M/M/c/K.
            states = self.states_absorption
            dist_states = self.dist_absorption
        elif start_event_type == EventType.ACTIVATION:
            states = self.states_activation
            dist_states = self.dist_activation
        else:
            raise Warning("reset_positions: Invalid value for parameter 'start_event_type': {}".format(start_event_type))

        nparticles_by_start_state = [0] * len(dist_states)
        if N_min is not None:
            # A minimum number of particles sampled in each start state should be guaranteed 
            # => Define the FIXED number of particles to use for each possible start state based on the stationary distribution

            # Number of particles required to satisfy the N_min particles for each state
            # This minimum happens at the largest buffer size (which corresponds to the latest entry in the distribution array)
            # since it the largest buffer size has the smallest probability.
            N = int( round( N_min / dist_states[-1] ) )

            # Update the number of particles in the system and reset it before the simulation starts
            # if the N just computed is larger than the one stored in the object.
            # Otherwise, use the number of particles N stored in the object.
            # In either case the minimum number of particles representing each start state will be at least N_min.
            if N > self.N:
                print("New N to satisfy N_min={} is: {} > original N={}".format(N_min, N, self.N))
                self.N = N
                self.reset()
            # Compute the number of particles to use for each start state
            for idx in range( len(nparticles_by_start_state)-1, 0, -1 ):
                nparticles_by_start_state[idx] = int( round( dist_states[idx] * self.N ) )
            nparticles_by_start_state[0] = self.N - int( np.sum(nparticles_by_start_state[1:]) )

            # IF WE DON'T CARE about the minimum N to use for each start state 
            # Number of particles to assign for each possible start state
            #nparticles_by_start_state = [int( round( p*self.N ) ) for p in dist_states]
            # Make the last number of particles be the REMAINING number (to counteract rounding issues)
            #nparticles_by_start_state[-1] = self.N - int( np.sum(nparticles_by_start_state[:-1]) )
        else:
            # No N_min particles is required for representing each state
            # => We choose the start states by DRAWING from the stationary distribution
            # (as opposed to setting precisely the number of particles to simulate in each start state)
            # This approach is particularly useful when the number of possible states is "very large"
            # making the required total N to satisfy N_min too large... (which then triggers an OUT-OF-MEMORY error)
            if uniform_random_choice:
                p = None
            else:
                p = dist_states
            idx_start_states, freq_start_states = np.unique( np.random.choice( len(dist_states), size=self.N, p=p ), return_counts=True )
            for idx_freq, idx_state in enumerate(idx_start_states):
                nparticles_by_start_state[idx_state] = freq_start_states[idx_freq]

        if False:
            print("# particles by start_state:")
            for i, (x, p, n) in enumerate( zip( states, dist_states, nparticles_by_start_state ) ):
                print("{}: x={}, p={} --> N={}".format(i, x, p, n))

        idx_state_space = -1
        P_first_in_block = 0
        pmin, pmax = np.min(dist_states), np.max(dist_states)
        if False:
            print("Total number of activation states: {} ({:.6f}% <= p(S) <= {:.3f}%) (Nmin to expect at 1 case in state with smallest p: Nmin={:.0f})" \
                  .format(len(dist_states), pmin*100, pmax*100, 1/pmin))
        for nparticles_in_block in nparticles_by_start_state:
            idx_state_space += 1
            if nparticles_in_block > 0:
                P_last_in_block = P_first_in_block + nparticles_in_block - 1
                if False:
                    print("Range of particle indices to simulate with start state #{} out of {}: [{}, {}] (n={}, n/N={}, p={:.6f}, diff={:.1f}%, state={})" \
                          .format(idx_state_space+1, len(nparticles_by_start_state), P_first_in_block, P_last_in_block, nparticles_in_block, nparticles_in_block/self.N, dist_states[idx_state_space], (nparticles_in_block/self.N - dist_states[idx_state_space]) / dist_states[idx_state_space], states[idx_state_space]))
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
                                                   'S': [ list(self.particles[P].getServerSizes()) ], # State index associated to the special event
                                                        ## IMPORTANT: We need to convert the state to a list in order to be findable by the .index() function of lists used in e.g. compute_counts()
                                                   't0': None,                  # Absorption /reactivation time
                                                   'x': None,                   # Position after reactivation
                                                   'particle number': P,        # Source particle (queue) from which reactivation happened (only used when reactivate=True)
                                                   'particle ID': P,            # Particle ID associated to the particle BEFORE reactivation (this may not be needed but still we store it in case we need it --at first I thought it would have been needed during the finalize() process but then I overcome this need with the value t0 already stored
                                                   'reactivated number': None,  # Reactivated particle (queue) to which the source particle was reactivated (only used when reactivate=True)
                                                   'reactivated ID': None       # Reactivated particle ID (index of this info_particle list)
                                                   }) ]

                P_first_in_block = P_last_in_block + 1

        self.reset_flags(start_event_type)
        self.reset_times(start_event_type)

        return nparticles_by_start_state

    def reset_flags(self, start_event_type :EventType):
        """
        Resets flag arrays whose initial value depends on the start event type:
        - is_particle_active: whether each particle is active
        - is_fv_ready: whether each particle is ready for the FV process
        """
        if start_event_type == EventType.ABSORPTION:
            self.is_particle_active = [False]*self.N
            self.is_fv_ready = [False]*self.N
        elif start_event_type == EventType.ACTIVATION:
            self.is_particle_active = [True]*self.N
            self.is_fv_ready = [True]*self.N
        else:
            raise Warning("reset_flags: Invalid value for parameter 'start_event_type': {}".format(start_event_type))

    def reset_times(self, start_event_type :EventType):
        """
        Resets auxiliary times arrays whose initialization depends on the start event type:
        - times0: latest time the particle changed to an absorption state
        - times1: latest time the particle has been activated
        """
        if start_event_type == EventType.ABSORPTION:
            self.times0 = np.zeros(self.N, dtype=float)
            # Latest time the particles were activated is unknown as they don't start at an activation state
            self.times1 = np.nan*np.ones(self.N, dtype=float)
        elif start_event_type == EventType.ACTIVATION:
            # Latest time the particles changed to an absorption state is unknown as they don't start at an absorption state
            self.times0 = np.nan*np.ones(self.N, dtype=float)
            self.times1 = np.zeros(self.N, dtype=float)
        else:
            raise Warning("reset_times: Invalid value for parameter 'start_event_type': {}".format(start_event_type))

    def reset_particles_position_info(self, P, start_event_type):
        """
        Resets the particle's trajectory information so that:
        - all past trajectory information is erased (so that the burn-in period does NOT interfere with the FV process
        --essentially by not having blocking times observed during the burn-in period contribute to block times in the FV process)
        - the reactivation process works smoothly, essentially for the part where the particle's position
        of the particle Q chosen for reactivation is read, which assumes that ALL trajectory information of particles
        start at the SAME time origin --see get_position() method.

        IMPORTANT: The reset is ONLY done on the information attributes stored in the object,
        but NOT on the queues represented by each particle. In fact, we want each queue/particle
        to keep following the same time measurement which allows us to compute the time interval between
        events of interest more easily (for instance, the calculation of the time interval between
        the latest absorption event defining the end of the burn-in period and the first absorption event
        AFTER the burn-in period --which is used to estimate E(T), the expected absorption cycle time).

        Arguments:
        P: int
            Particle number to reset.

        start_event_type: EventType
            Type of event associated to the start state.
        """

        # 1) Trajectories, and current and historical particle positions
        for server in range(self.nservers):
            # Trajectories: one value per particle and per server
            self.trajectories[P][server]['t'] = [ 0.0 ]
            self.trajectories[P][server]['x'] = [ self.particles[P].getServerSize(server) ]
            self.trajectories[P][server]['e'] = [ Event.RESET ]
            # Times and positions by server (current and historical)
            self.positions_by_server[P][server] = self.particles[P].getServerSize(server)
            self.times_by_server[P][server] = 0.0
            self.all_positions_by_server[P][server] = [ self.positions_by_server[P][server] ]
            self.all_times_by_server[P][server] = [ self.times_by_server[P][server] ]

        # Times and positions for the buffer size (current and historical)
        self.positions_buffer[P] = np.sum( self.positions_by_server[P] )
        self.times_buffer[P] = 0.0
        self.all_positions_buffer[P] = [ self.positions_buffer[P] ]
        self.all_times_buffer[P] = [ self.times_buffer[P] ]

        # 2) Info about special events
        self.info_particles[P]['t'] = [ 0.0 ]
        self.info_particles[P]['E'] = [ [start_event_type, EventType.START_POSITION] ]
        self.info_particles[P]['S'] = [ list(self.particles[P].getServerSizes()) ]
            ## IMPORTANT: We need to convert the state to a list in order to be findable by the .index() function of lists used in e.g. compute_counts()

        # Change the time origin of the queue used to represent the particle to the current time
        current_time = self.particles[P].getMostRecentEventTime()
        self.particles[P].setOrigin(current_time)
        assert self.particles[P].getMostRecentEventTime() == 0.0
        # Resets the particle so that ALL servers are aligned in the same time
        self.particles[P].reset(self.particles[P].getServerSizes())

    #--------------------------------- Functions to simulate ----------------------------------
    #------------ HIGH LEVEL SIMULATION ------------
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
        - P(K), the estimated blocking probability
        - Integral, the estimated expected blocking time given the particle starts at an active state and is not absorbed
          (this is the integral in the numerator of the expression giving the blocking probability estimate)
        - E(T), the estimated expected absorption cycle time, i.e. at a state
        through which the particle enters the set of absorbed states.
        - the number of cycles contributing to the estimation of E(T)

        When reactivate=False, Monte-Carlo is used to compute the blocking probability,
        therefore the tuple contains the following elements:
        - the estimated blocking probability
        - the total blocking time over all particles
        - the total absorption cycle time over all particles
        - the number of measured absorption cycles
        """

        # TODO: (2021/12/06) Extend the simulation results to estimating the AVERAGE REWARD obtained from blocking based on the accept/reject policy
        # 1) The AVERAGE REWARD is obtained by the average cost of blocking based on the accept/reject policy.
        # Note that the average reward is equivalent to the blocking probability P(K) when the accept/reject policy is
        # deterministic and blocks (rejects incoming jobs) only at buffer size = K with rejection cost = 1.
        #
        # MAIN IDEA TO ACHIEVE THIS EXTENSION:
        # a) Add a parameter containing the reward information for each buffer size.
        #   For now this can be the name of a function that computes the reward for each buffer size.
        # b) Make the counts_blocked list keep track on more blocking buffer sizes
        #   Modify the counts_blocked list into a dictionary of lists indexed by the buffer sizes associated to
        #   a positive reject probability.
        #   --> NOTE that this implies updating the _update_counts_blocked() method.
        # c) Modify the proba_block_by_t attribute and make it a dictionary as well, indexed by the buffer sizes
        #   associated to a positive reject probability.
        #
        # 3) We should also provide reward information for buffer sizes and actions of interest, which allows us
        # to estimate the state-action value function Q(s,a) for (s,a) of interest, where s is the buffer size
        # (which coincides with the STATE in a single-server system but does NOT in a multi-server system).
        # Use case: estimate the gradient of the average reward in the context of policy gradient learning as in
        # the Trunk-Reservation paper 2019.

        self.reset()
        self.reset_positions(start_event_type)

        # (2021/02/10) Reset the random seed before the simulation starts,
        # in order to get the same results for the single-server case we obtained
        # BEFORE the implementation of a random start position for each particle
        # (a position which is actually NOT random in the case of single server system!)  
        np.random.seed(self.seed)

        time_start, time1, time2, time3 = self.run_simulation()

        if self.reactivate:
            if self.LOG: #True
                print("Estimating blocking probability with Fleming-Viot...")
            self.estimate_proba_blocking_fv()
        else:
            if self.LOG: #True:
                print("Estimating blocking probability with Monte-Carlo...")
            proba_blocking, total_blocking_time, total_return_time, total_return_n = self.estimate_proba_blocking_mc()
            self.proba_blocking = proba_blocking
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
            return self.proba_blocking, self.integral, self.expected_absorption_time, self.n_expected_absorption_time
        else:
            return proba_blocking, total_blocking_time, total_return_time, total_return_n

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
        self.reset_positions(start_event_type=EventType.ACTIVATION)

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
        """
        Runs the simulation
        
        Arguments:
        stop_at_first_absorption: bool
            Whether the simulation of each particle should stop at the first absorption of the particle.
            This is used when we are interested in measuring the time until absorption, e.g. to estimate
            P(T>t) given activation. 
            default: False

        Return: tuple
        Triple with start times of each major step, plus the end time of the simulation. 
        """

        #-------------------------------- Auxiliary functions --------------------------------#
        def FV_estimation():
            # Are we running the FV estimation? (as opposed to the MC estimation)
            return self.reactivate

        def simulate_until_all_particles_are_activated_before_starting_fv_process():
            # ONLY DONE IN FV
            # We first simulate until all particles are activated and therefore we can start with the FV procedure
            if self.LOG or DEBUG_TRAJECTORIES: #True
                print("simulate: [reactivate=True] Generating trajectories for each particle until the first ACTIVATION after the burn-in period takes place...", \
                      "\n--> so that we can start the FV process.")
            assert np.sum(self.is_particle_active) == 0, "No particles are active at startup when reactivate=True"
            self.generate_trajectories_at_startup(event_type_to_stop=EventType.ACTIVATION)
            if self.LOG or DEBUG_TRAJECTORIES: #True
                min_start_time_fv = self.get_min_start_time_fv()
                mean_start_time_fv = self.get_mean_start_time_fv()
                max_start_time_fv = self.get_max_start_time_fv()
                n_start_time_fv = self.get_n_start_time_fv()
                print("--> (min, mean, max) times at which FV process started on n={} particles: ({:.1f}, {:.1f}, {:.1f}) (({:.1f}%, {:.1f}%, {:.1f}%) of max simulation time)" \
                      .format(n_start_time_fv, min_start_time_fv, mean_start_time_fv, max_start_time_fv,
                              min_start_time_fv / self.maxtime * 100, mean_start_time_fv / self.maxtime * 100, max_start_time_fv / self.maxtime * 100))

        def simulate_until_first_absorption():
            # FV: We need to record the first absorption time for each particle
            #     because the first particle to be reactivated is the one with smallest absorption time,
            #     so that we can reactivate it to ANY of the other particles participating in the FV process,
            #     whose position is KNOWN at that time, SINCE WE ARE STANDING AT THE SMALLEST TIME among all the
            #     absorption times!
            # MC: Although recording the first absorption time is not necessary, we leverage this function
            #     as it doesn't hurt using it. After the first absorption, we then proceed to the end of the simulation.
            if self.LOG or DEBUG_TRAJECTORIES: #True
                print("simulate: Generating trajectories for each particle until first absorption...", \
                      "\n--> so that we can sort ALL the first absorption times of particles and start reactivating when reactivate=True", \
                      "\n    which requires that the reactivation process starts with the particle with smallest absorption time", \
                      "\n    so that the reactivated particle can be chosen among the N-1 particles not yet absorbed at that time.")
            self.generate_trajectories_at_startup(event_type_to_stop=EventType.ABSORPTION)

        def set_simulation_time_of_fv_process_to_largest_observed_survival_time():
            # The simulation of the process until first absorption has just estimated the survival probability P(T>t)
            # for all observed active-to-absorption times t and now we need to start the FV process. 
            # => Set the simulation time to a little more than the largest observed survival time
            # as no contribution to the blocking probability will come from times larger than this value.
            # We set it to "a little more" than the largest observed time to avoid an error in the assertion
            # that checks that all recorded simulation times are smaller than maxtime.
            self.maxtime_burnin = self.maxtime
            if self.get_max_observed_survival_time() == 0.0:
                # No survival time was observed for the estimation of P(T>t)!
                # => The FV process cannot start because the simulation time to run on it would be 0!
                warnings.warn("No survival time was observed for the estimation of P(T>t / activation). The process cannot continue.")
            self.set_simulation_time(1.1*self.get_max_observed_survival_time())
            if DEBUG_TRAJECTORIES:
                print("--> Simulation time reset from T={:.1f} to T={:.1f} (1.1 * maximum observed survival time for P(T>t) estimation).".format(self.maxtime_burnin, self.maxtime))

        def simulate_until_end_of_simulation():
            if self.LOG or DEBUG_TRAJECTORIES: #True
                if self.getMaxNumberOfEvents() == np.Inf:
                    print("Generating trajectories for each particle until END OF SIMULATION (T={:.1f})...".format(self.maxtime))
                else:
                    print("Generating trajectories for each particle until END OF SIMULATION (max #events={})...".format(self.getMaxNumberOfEvents()))
            self.generate_trajectories_until_end_of_simulation()

        def check_trajectories():
            if self.plotFlag or DEBUG_TRAJECTORIES:
                if False:
                    for P in self.particles_to_simulate:
                        self.plot_trajectories_by_server(P)
                self.plot_trajectories_by_particle()
                #input("Press ENTER...")

        def check_special_events():
            if False:
                print("\n**** CHECK status of attributes *****")
                print("Info particles:")
                for p in range(len(self.info_particles)):
                    print("Particle ID {}:".format(p))
                    print(self.info_particles[p])
                #input("Press ENTER...")

        def finalize_trajectories_for_calculations():
            if self.LOG or DEBUG_TRAJECTORIES: #True
                print("Finalizing and identifying measurement times...")
            self.finalize()

        def compute_counts_needed_for_survival_and_blocking_probability_functions():
            if self.is_proba_survival_estimated_from_burnin:
                # In this case we don't need to compute the alive counts list on the basis of the
                # activation-to-absorption times because this has already been calculated DURING SIMULATION,
                # namely after the first absorption following the burn-in period has been observed for ALL particles.
                # See method _record_first_observed_absorption_time_after_burnin_cycles() or similar.
                if self.burnin_cycles_complete_all_particles:
                    assert len(self.counts_alive) == self.N + 1, \
                            "All N={} particles contribute to the estimation of the survival probability ({})" \
                            .format(self.N, len(self.counts_alive)-1)
                else:
                    assert len(self.counts_alive) == len(self.particles_to_simulate) + 1, \
                            "All simulated N0={} particles contribute to the estimation of the survival probability ({})" \
                            .format(len(self.particles_to_simulate), len(self.counts_alive)-1)
                self.compute_counts_blocked_particles()
            else:
                self.compute_counts()
        #-------------------------------- Auxiliary functions --------------------------------#

        #-- 1) FV: Simulate until activation
        time_start = timer()
        if FV_estimation() and self.nservers > 1:
            simulate_until_all_particles_are_activated_before_starting_fv_process()

        #-- 2) FV & MC: Simulate until first absorption
        simulate_until_first_absorption()
        time1 = timer()

        #-- 3) FV: Reset simulation time for FV process
        if FV_estimation() and self.is_proba_survival_estimated_from_burnin:
            set_simulation_time_of_fv_process_to_largest_observed_survival_time()

        #-- 4) FV & MC: Simulate until END
        if not stop_at_first_absorption:
            #-- PAUSE) Check trajectories
            check_trajectories()

            simulate_until_end_of_simulation()
            time2 = timer()

            #-- PAUSE) Check trajectories
            check_trajectories()
            check_special_events()
        else:
            time2 = None

        #-- 5) FV & MC: Prepare trajectories for calculations
        finalize_trajectories_for_calculations()

        #-- 6) FV & MC: Estimate P(T>t) (needed when estimating P(T>t) separately from the FV simulation)
        compute_counts_needed_for_survival_and_blocking_probability_functions()
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
        n_particles_by_start_state = self.reset_positions(EventType.ACTIVATION, N_min=N_min)

        time_start, time1, _, time_end = self.run_simulation(stop_at_first_absorption=True)

        if self.LOG:
            total_elapsed_time = (time_end - time_start)
            print("Total simulation time: {:.1f} min".format(total_elapsed_time / 60))
            print("\tCompute counts: {:.1f} min ({:.1f}%)".format((time_end - time1) / 60, (time_end - time1) / total_elapsed_time*100))

        return n_particles_by_start_state, self.sk, self.counts_alive, self.ktimes_sum, self.ktimes_n

    def simulate_return_time_to_absorption(self, N_min :int=None):
        """   
        Simulates the queue system to estimate the expected return time to absorption, E(T),
        given the initial state is at the boundary of the absorption set.

        Arguments:
        N_min: (opt) int
            Minimum number of particles to simulate in each set of start states.
            default: 1 

        Return: tuple
        The tuple contains the following elements:
        - ktimes_sum: a list with the total observed RELATIVE killing times (times to absorption)
        for EACH simulated particle P.
        - ktimes_n: a list with the number of observed RELATIVE killing times (times to absorption)
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
    #------------ HIGH LEVEL SIMULATION ------------


    #------------ LOW LEVEL SIMULATION -------------
    def generate_trajectories_at_startup(self, event_type_to_stop :EventType):
        assert event_type_to_stop in [EventType.ABSORPTION, EventType.ACTIVATION]

        # Auxiliary function used when event_type_to_stop = EventType.ACTIVATION
        # Note: The value of the self.is_fv_ready array is possible modified by function
        # _generate_trajectory_until_absorption_or_end_of_simulation() called below.
        get_particles_on_which_the_fv_process_should_be_run = lambda: [P for P, ready in enumerate(self.is_fv_ready) if ready]

        if self.reactivate and self.burnin_cycles_absorption > 0 and not self.burnin_cycles_complete_all_particles:
            # We are running the FV simulation with a burn-in period
            # and with no requirement that all N particles be used for the FV process
            # => Update (i.e. possibly reduce) the number of simulated particles
            particles_to_simulate = [P for P, burnin_ended in enumerate(self.has_burnin_period_ended) if not burnin_ended]
        else:
            particles_to_simulate = self.particles_to_simulate
        for P in particles_to_simulate:
            self.iterations[P] = 0
            # Perform a first iteration where the next BIRTH time is generated for ALL job classes
            # and the next DEATH time is generated for all servers with jobs in the queue.
            # This is needed to get the simulation started because such process cannot start
            # until at least one of the latest event times in the servers is different from the rest.
            # (Recall that at the very beginning all event times for all servers are equal to 0,
            # which prevents the process from retrieving the information of the most recent event,
            # needed to know which server should be updated next
            # --see function _generate_trajectory_until_absorption_or_end_of_simulation() or similar)
            end_of_simulation = self.generate_one_iteration(P)

            if event_type_to_stop == EventType.ACTIVATION:
                # Simulate until the (k+1) ACTIVATION or the max simulation time is reached,
                # where k is the number of burn-in activations.
                # Once the (k+1) ACTIVATION is reached, we can start with the Fleming-Viot process.
                assert self.reactivate, "The simulation is run until ACTIVATION only for the FV estimation case (i.e. when reactivate=True)"
                self.has_burnin_period_ended[P] = self._generate_trajectory_until_burnin_activation_or_end_of_simulation(P, end_of_simulation)
            elif event_type_to_stop == EventType.ABSORPTION and not self.has_burnin_period_ended[P]:
                # There still time in the burn-in period for this particle to simulate until a possible ABSORPTION
                # => Simulate until absorption or until the max simulation time allowed for the burn-in period is reached
                # NOTE: It is important to ONLY simulate when the end of the burn-in period has not been reached already
                # because if we don't respect this rule, we will have absorption times stored for particles that
                # should NOT be included in the FV process, just because they had already reached the
                # maximum allowed time for the burn-in period while looking for the first ACTIVATION event
                # after the burn-in period.
                self.has_burnin_period_ended[P] = self._generate_trajectory_until_absorption_or_end_of_simulation(P, end_of_simulation)

            if False:
                # Log the completion of the simulation for the first few particles and then every 10% of particles 
                if P <= 2 or int( P % (self.N/10) ) == 0:
                    print("# particles processed so far: {}".format(P+1))

        if False:
            print("\n******** Info PARTICLES at completion of simulation STARTUP:")
            for P in range(self.N):
                np.set_printoptions(precision=3, suppress=True)
                print("Particle {}:".format(P))
                print(np.c_[np.array(self.info_particles[P]['t']), np.array(self.info_particles[P]['E']), np.array(self.info_particles[P]['S'])])
                np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

        # Check the correctness of the simulated trajectories just generated in terms of the simulation parameters setup
        if event_type_to_stop == EventType.ABSORPTION:
            assert np.sum([not a or a and e for P, (a, e) in enumerate(zip(self.is_particle_active, self.has_burnin_period_ended)) if P in range(self.N)]) == self.N, \
                    "Only particles that reached the end of the simulation (T={}, #events={}) can be ACTIVE, all the others must have been ABSORBED" \
                    .format(self.maxtime, self.getMaxNumberOfEvents())
                    #"(n={}, {:.1f}%) or their simulation time has went over its maximum of {:.1f} (n={}, {:.1f}%)" \
                    #.format(nparticles_absorbed, nparticles_absorbed/self.N*100,
                    #        nparticles_ended_simulation, nparticles_ended_simulation/self.N*100)
            if self.reactivate and self.burnin_cycles_absorption > 0 and not self.burnin_cycles_complete_all_particles:
                # Possibly reduce the set of particles to simulate the FV process on
                # when we DO NOT require all particles to be used in the FV estimation
                # (as set by parameter burnin_cycles_complete_all_particles)
                self.particles_to_simulate = get_particles_on_which_the_fv_process_should_be_run()
                if len(self.particles_to_simulate) <= 1:
                    raise ValueError("The number of particles to run the FV process on is too small (N0={})" \
                                     .format(len(self.particles_to_simulate)) + \
                                     "\nTry increasing the simulation time (nmeantimes={:.1f}) or decreasing the buffer size for activation (J={})." \
                                     .format(self.nmeantimes, self.buffer_size_activation))
                assert len(self.particles_to_simulate) == np.sum(self.is_fv_ready), \
                    "The number of particles to simulate in the FV process ({}) " \
                    "coincides with the number of particles that are READY for the FV process ({})" \
                    .format(len(self.particles_to_simulate), np.sum(self.is_fv_ready))
                #for P in [Q for Q, ready in enumerate(self.is_fv_ready) if not ready]:
                #    assert P not in self.dict_info_absorption_times['P'], \
                #        "There must be NO absorption times associated to particles that do not qualify for the FV process (P={}, absorption times={})" \
                #        .format(P, self.dict_info_absorption_times) + \
                #        "\nInfo Particles for particle P={}:\n{}".format(P, self.info_particles[P])

    def generate_one_iteration(self, P):
        """
        Generates one iteration of event times for a particle.
        
        This implies:
        - A set of birth and death event times (one set of times per server) is generated for the particle.
        - The event with the smallest event time is applied to the particle.

        Return: bool
        Whether the end of the simulation has been reached.
        """
        end_of_simulation = lambda t, nevents: nevents >= self.getMaxNumberOfEvents() or self.getMaxNumberOfEvents() == np.Inf and t >= self.maxtime

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
        server_next_event, type_next_event, time_next_event = self._get_next_event(P)   # time_next_event is a super-absolute time
        assert not np.isnan(time_next_event)
        if end_of_simulation(time_next_event - self.particles[P].getOrigin(), self.nevents):
            ## NOTE: The end of simulation is checked against the next event time RELATIVE to the particle's time origin
            return True
        else:
            if self.LOG:
                print("\n++++++ P={}: APPLY an event ++++++".format(P))
            self._apply_next_event(P, server_next_event, type_next_event, time_next_event)
            self.assertSystemConsistency(P)

        return False

    def _generate_next_events(self, P):
        """
        Generates the next event for each server, their type (birth/death) and their time
        storing their values in the internal attributes containing these pieces of information.
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)
        self.assertTimeMostRecentEventAndTimesNextEvents(P)

        # Generate OR retrieve the next set of death and birth times by server
        # Note that:
        # - DEATH TIMES: we *generate* a new death time for a server ONLY when the server's queue
        # at the current simulation time (i.e. at the time of last change of the state of the system
        # --self.times_latest_known_state) is NOT empty.
        # - BIRTH TIMES: we either *generate* or *retrieve* the new birth times depending on whether
        # the queue of next events for the server is empty or not, and on a condition about the
        # generated arrival times of the job classes which is a little complicated to explain here
        # (see details in the method called below).
        # If a new birth time is generated for a server, it is added to the server's queue
        # at the position in the queue derived from the generated birth time value and its comparison
        # with the other birth times already in the server's queue.
        if DEBUG_TIME_GENERATION:
            print("\n\n***** _generate_next_events: P={}: CURRENT STATE: {} @t={:.3f}*****".format(P, self.particles[P].getServerSizes(), self.times_latest_known_state[P]))
        self._generate_death_times(P)
        # NOTE: When using _generate_birth_times() which first generates arriving job classes
        # and then assigns each new job to a server based on the assignment policy,
        # we need to ACTIVATE (i.e. uncomment) the two lines in _apply_next_event()
        # that are enclosed by the labels "DM-2021/05/10-START"-"DM-2021/05/10-END".
        self._generate_birth_times_from_equivalent_rates(P)
        #self._generate_birth_times(P)

    def _generate_death_times(self, P):
        """
        Generates death times for each server with jobs in its queue

        The effect of this method is the following:
        - The list of job service times in each server is updated with the new death time generated for the server.
        - The times of the next service event (death event) by server is updated when the generated death event
        is the first service time assigned to the server. Otherwise, the time of the next death event in the sever is kept. 
        """
        # Generate a death time for each server whose queue is NOT empty and store them as ABSOLUTE values
        if DEBUG_TIME_GENERATION:
            print("\n_generate_death_times (START): P={}".format(P))
            print("times next DEATH event: {}".format(self.times_next_death_events[P]))

        for server in range(self.nservers):
            if self.particles[P].getServerSize(server) > 0 and np.isnan(self.times_next_death_events[P][server]):
                # Generate the next death time for the server
                # NOTE: (2021/05/09) The death ABSOLUTE time is computed by adding the newly generated relative time
                # to the time of the latest known state for the particle,
                # i.e. the time at which the latest event was applied, that made the particle's state CHANGE.
                # (Recall that every time an event is applied the state HAS to change, as we are no longer applying
                # death events to empty server queues!)
                death_time_relative = self.particles[P].generate_event_time(Event.DEATH, server)
                self.times_next_death_events[P][server] = self.times_latest_known_state[P] + death_time_relative
                if DEBUG_TIME_GENERATION:
                    print("server {}: Current time={:.3f}, Exponential DEATH time={:.3f}, ABSOLUTE time={:.3f}" \
                          .format(server, self.times_latest_known_state[P], death_time_relative, self.times_next_death_events[P][server]))

        if DEBUG_TIME_GENERATION:
            print("_generate_death_times (END): P={}".format(P))
            print("times next DEATH events by server: {}".format(self.times_next_death_events[P]))

    def _generate_birth_times(self, P):
        """
        Generates new birth times for EACH server in the given particle.
        
        The process is carried out as follows: job arrival times are generated for each job class
        and each new job is assigned to one of the servers based on the assignment policy. The process
        is repeated until all servers have at least ONE job in their queue (so that no "next birth time"
        in a server is NaN, making it possible to compute the next event to be observed in the system
        as the MINIMUM over all non-NaN next birth and next death events).

        The assignment policy is assumed to be a mapping from job class to server number.

        The effect of this method is the following:
        - The queue of jobs in each server is updated with the new assigned job (storing job class and event time).
        - The times of the next birth events by server is updated when the generated birth event is the first job
        assigned to the server's queue. Otherwise, the time of the next birth event in the server is maintained.
        """

        #--------------------------------- Auxiliary functions -------------------------------#
        def at_least_one_server_may_not_yet_hold_the_next_birth_time(P):
            """
            The situation described by the function name happens when
            the currently stored "next birth time" for at least one server
            (given by self.times_in_server_queues[P][server][0]) is either:
            - INEXISTENT
            - LARGER than the SMALLEST time among those generated for the next arriving job of each class
            the last time these times were generated (self.times_latest_arrived_jobs[P]).

            This means that at least one of the times of the next arriving job classes
            COULD be smaller than the time of the next job to be served by one of the servers
            (the possibility depends on the server to which those jobs are assigned, which is a random process),
            and such situation makes the time of the next job to be served by such server to be not yet defined.

            NOTE that this condition is ALWAYS satisfied at the very first call of this function
            (i.e. at the beginning of the simulation of the particle's trajectory) since
            the list of times of jobs assigned to each server is empty.
            """
            resp = False
            smallest_time_among_jobclasses = np.min(self.times_latest_arrived_jobs[P])
            assert not np.isnan(smallest_time_among_jobclasses)
            for server in range(self.nservers):
                if len(self.times_in_server_queues[P][server]) == 0 or \
                   self.times_in_server_queues[P][server][0] > smallest_time_among_jobclasses:
                    resp = True
                    break
            return resp

        def generate_next_times_jobclasses(P):
            "It updates self.times_latest_arrived_jobs[P] with the ABSOLUTE next arrival times for each job class"
            # Generate the next arriving job class times
            birth_times_relative_for_jobs = np.random.exponential( 1/np.array(self.job_class_rates) )
            # Make a copy of the absolute times of the latest jobs (one value per job class),
            # prior to updating their values. This is done just for INFORMATIONAL purposes.
            times_latest_arrived_jobs_prev_P = copy.deepcopy(self.times_latest_arrived_jobs[P])
            # Update the absolute times of the latest jobs (one value per job class)
            self.times_latest_arrived_jobs[P] += birth_times_relative_for_jobs
            if DEBUG_TIME_GENERATION:
                for job_class in range( len(self.job_class_rates) ):
                    print("job class {}: Current time={:.3f}, Previous arrived job time={:.3f}, Exponential BIRTH time={:.3f}, ABSOLUTE time={:.3f}" \
                          .format(job_class, self.times_latest_known_state[P], times_latest_arrived_jobs_prev_P[job_class], birth_times_relative_for_jobs[job_class], self.times_latest_arrived_jobs[P][job_class]))

        def accept_jobs(P):
            "Decides which jobs to accept based on the job acceptance policy"
            # TODO: (2021/12/08) This method should be updated / fixed as follows:
            # a) The acceptance of jobs should be evaluated ONE AT A TIME. At this point, ALL the latest arriving
            # jobs for each job class are evaluated at ONCE for acceptance! This is INCORRECT because in between
            # two different job class arrival times there may be a service time, and this changes the acceptance
            # policy of the next job class arrival! (because e.g. the buffer may no longer be full)
            # b) When the policy is None, we should accept ONLY if the queue's buffer size is less than its capacity.
            # (currently all jobs are accepted)
            job_classes = range( len(self.job_class_rates) )
            if self.policy_accept is not None:
                # Use the acceptance policy to decide whether we accept each job IN THE ORDER THEY ARRIVE 
                times_latest_arrived_jobs_P = copy.deepcopy(self.times_latest_arrived_jobs[P])
                order = np.argsort(times_latest_arrived_jobs_P)
                job_classes_accepted = []
                #print("P={}:\ntimes_latest_arrived_jobs_P: {}".format(P, times_latest_arrived_jobs_P))
                #print("order: {}".format(order))
                for job_class in [job_classes[o] for o in order]:
                    #print("job_class: {}".format(job_class))
                    #print("state: {}".format(self.policy_accept.env.getState()))
                    self.policy_accept.env.setJobClass(job_class)
                    action = self.policy_accept.choose_action(self.queue.getServerSizes())
                    if action == 1:
                        #print("ENTRO")
                        assert self.particles[P].getBufferSize() < self.particles[P].getCapacity(), \
                               "The buffer is NOT at full capacity when a job is accepted by the acceptance policy"
                        job_classes_accepted += [job_class]
                        next_state, reward, done, info = self.policy_accept.env.step(action, ActionTypes.ACCEPT_REJECT)
            else:
                job_classes_accepted = job_classes

            return job_classes_accepted

        def assign_accepted_jobs_to_servers(P, job_classes):
            """
            Assigns the accepted jobs to servers based on the assignment policy.
            The self.times_in_server_queues[P] and self.jobclasses_in_server_queues[P] attributes
            are updated with this information.
            """
            servers = range(self.nservers)
            for job_class in job_classes:
                # IMPORTANT: The assigned server can be repeated for different job classes
                # => we might have a queue of jobs for a particular server (which is handled here)
                if self.nservers == 1:
                    # There is really no assignment to be done, as only one possibility exists
                    assigned_server = 0
                else:
                    assigned_server = self.policy_assign.choose_action(job_class, servers)
                assert assigned_server < self.nservers, \
                        "The assigned server ({}) is one of the possible servers [0, {}]".format(assigned_server, self.nservers-1)

                # Insert the new job time in order into the server's queue containing job birth times
                # Note that the order of the jobs is NOT guaranteed because the server's queue may contain
                # jobs of different classes, whose arrival times are generated independently from each other.
                idx_insort, _ = insort(self.times_in_server_queues[P][assigned_server], self.times_latest_arrived_jobs[P][job_class], unique=False)
                self.jobclasses_in_server_queues[P][assigned_server].insert(idx_insort, job_class)
                if self.LOG:
                    print("job class: {}".format(job_class))
                    #print("job time (PREVIOUS): {}".format(times_latest_arrived_jobs_prev_P[job_class]))
                    #print("job time (RELATIVE): {}".format(birth_times_relative_for_jobs[job_class]))
                    print("job time (ABSOLUTE): {}".format(self.times_latest_arrived_jobs[P][job_class]))
                    print("assigned server: {}".format(assigned_server))
                    print("queue of assigned server: {}".format(self.times_in_server_queues[P][assigned_server]))
        #--------------------------------- Auxiliary functions -------------------------------#

        if DEBUG_TIME_GENERATION:
            print("\n_generate_birth_times (START): P={}".format(P))
            print("times latest jobs[P]: {}".format(self.times_latest_arrived_jobs[P]))
            print("times next events in server queues:")
            for s in range(self.nservers):
                print("\t{}: {}".format(s, self.times_in_server_queues[P][s]))

        while at_least_one_server_may_not_yet_hold_the_next_birth_time(P):
            # We still don't know what's the time of the next birth event in all servers
            # => Generate a birth time for ALL the job CLASSES and assign them to a server
            generate_next_times_jobclasses(P)       # This updates self.times_latest_arrived_jobs[P] for each job class
            job_classes_accepted = accept_jobs(P)   # This updates self.times_in_server_queues[P] and self.jobclasses_in_server_queues[P]
            assign_accepted_jobs_to_servers(P, job_classes_accepted)

            if self.LOG:
                print("currently assigned job classes by server: {}".format(self.jobclasses_in_server_queues[P]))
                print("currently assigned job times by server: {}".format(self.times_in_server_queues[P]))

        # Assign the birth times for each server by picking the first job in each server's queue
        for server in range(self.nservers):
            assert len(self.times_in_server_queues[P][server]) > 0, "P={}: The queue in server {} is not empty.".format(P, server)
            self.times_next_birth_events[P][server] = self.times_in_server_queues[P][server][0]

        if DEBUG_TIME_GENERATION:
            print("_generate_birth_times (END): P={}".format(P))
            print("times latest jobs[P]: {}".format(self.times_latest_arrived_jobs[P]))
            print("times next events in server queues:")
            for s in range(self.nservers):
                print("\t{}: {}".format(s, self.times_in_server_queues[P][s]))

    def _generate_birth_times_from_equivalent_rates(self, P):
        """
        Generates new birth times for EACH server in the particle that does NOT have a "next birth time".
        The equivalent arrival rates are used instead of the job class arrival rate + assignment policy.
        """

        if DEBUG_TIME_GENERATION:
            print("\n_generate_birth_times_from_equivalent_rates (START): P={}".format(P))

        for server in range(self.nservers):
            if np.isnan(self.times_next_birth_events[P][server]):
                # Generate the next birth time for the server
                # NOTE: (2021/05/09) The birth ABSOLUTE time is computed by adding the newly generated relative time
                # to the time of the latest known state for the particle,
                # i.e. the time at which the latest event was applied.
                birth_time_relative = self.particles[P].generate_event_time(Event.BIRTH, server)
                self.times_next_birth_events[P][server] = self.times_latest_known_state[P] + birth_time_relative
                if DEBUG_TIME_GENERATION:
                    print("server {}: Current time={:.3f}, Exponential BIRTH time={:.3f}, ABSOLUTE time={:.3f}" \
                          .format(server, self.times_latest_known_state[P], birth_time_relative, self.times_next_birth_events[P][server]))

        if DEBUG_TIME_GENERATION:
            print("_generate_birth_times_from_equivalent_rates (END): P={}".format(P))

    def _get_next_event(self, P):
        """
        Retrieves the relevant information about the next event for the given particle
        by choosing the event with the smallest time among all possible next events
        for each server (e.g. BIRTH or DEATH).

        Note that a DEATH event is not possible in a server if its queue has length 0.
        In such case, the time of the next death event is NaN, and such time is NOT considered
        for the computation of the smallest event time.

        Return: tuple
        The tuple contains:
        - the server number on which the next event takes place.
        - the type of the next event.
        - the time at which the next event takes place.
        """
        # Store the next birth and death times as columns by server for their comparison
        # so that we can easily find the server in which the next event takes place  
        self.times_next_events2[P] = np.c_[self.times_next_birth_events[P], self.times_next_death_events[P]]
        ## The above way of storing the times of the next BIRTH and DEATH events assumes
        ## that the BIRTH event is associated to the (index) value 0 and the DEATH event
        ## is associated to the index value 1.
        ## This is defined in the definition of the Event Enum which sets BIRTH = 0 and DEATH = 1.

        # 2D index in the above array of the smallest non-NaN next birth and death event times
        server_next_event, idx_type_next_event = np.where( self.times_next_events2[P] == np.nanmin(self.times_next_events2[P]) )
        assert len(server_next_event) > 0 and len(idx_type_next_event) > 0
        # Take care of the HIGHLY EXTREMELY UNLIKELY scenario when there is more than one occurence with the minimum time.
        # Also, this step is important because the output from the np.where() function is a tuple of ARRAYS
        # and we want each of the output value to be a SCALAR.
        # By retrieving the first element of each array we take care of both issues.
        server = server_next_event[0]
        idx_type_event = idx_type_next_event[0]

        # Update the relevant object attributes with this piece of information
        self.servers_next_event[P] = server
        self.types_next_event[P] = Event(idx_type_event)
        self.times_next_event[P] = self.times_next_events2[P][ server, idx_type_event ]

        if DEBUG_TIME_GENERATION:
            print("\n_get_next_event: P={}".format(P))
            print("Next birth times to choose from: {}".format(self.times_next_birth_events[P]))
            print("Next death times to choose from: {}".format(self.times_next_death_events[P]))
            print("TYPE OF CHOSEN NEXT EVENT: {}".format(self.types_next_event[P]))
            print("TIME OF CHOSEN NEXT EVENT: {}".format(self.times_next_event[P]))

        return self.servers_next_event[P], self.types_next_event[P], self.times_next_event[P]

    def _apply_next_event(self, P, server, type_of_event :Event, time_of_event :float):
        """
        Applies the next event to a particle at the given server, of the given type, at the given time,
        which should be given relative to the particle's time origin.

        Applying the event means the following:
        - The state of the queue represented by the particle is changed based on the applied event.
        - The list of events stored in this object is updated by removing the applied event, which means:
            - the job that is waiting first in the queue of jobs to be served is removed if it's a BIRTH event.
            - the job being served by the server is removed if it's a DEATH event.
        - The information about the position (state) of the particle is updated following the applied event.
        - The change in the particle is checked for the following special events and actions are taken accordingly
        to these events:
            - ABSORPTION
            - ACTIVATION

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) on which the event is applied.

        type_of_event: Event
            Whether the event is a BIRTH or DEATH event.
        
        time_of_event: non-negative float
            Time of the event that is applied given as an ABSOLUTE time, i.e. NOT
            "relative to the particle's time origin" stored in self.particles[P].
        """
        relative_time_of_event = time_of_event - self.particles[P].getOrigin()
        previous_buffer_size = self.particles[P].getBufferSize()
        change_size_of_server = self.particles[P].apply_event( type_of_event,
                                                               relative_time_of_event,
                                                               server )

        #-- Assertions
        if previous_buffer_size < self.particles[P].getCapacity():
            assert change_size_of_server != 0, \
                "P={}: server {} which was at size {} and experienced an {} at time {:.3f} changed size." \
                .format(P, server, previous_buffer_size, type_of_event, relative_time_of_event)
        elif previous_buffer_size == self.particles[P].getCapacity() and type_of_event == Event.BIRTH:
            assert change_size_of_server == 0, \
                "P={}: server {} which was blocked and experienced a BIRTH event at time {:.3f} did NOT change size.".format(P, server, relative_time_of_event)
        elif previous_buffer_size == self.particles[P].getCapacity() and type_of_event == Event.DEATH:
            assert change_size_of_server < 0, \
                "P={}: server {} which was blocked and experienced a DEATH event at time {:.3f} changed size.".format(P, server, relative_time_of_event)
        # TOFIX: (2021/03/25) Update the queue environment that keeps track of rewards, etc.
        # We should NOT have a separate environment containing a queue where we ALSO need to keep track of the buffer size...
        # (ALSO = "besides keeping track of the buffer size in the self.queue object")
        if self.policy_accept is not None:
            self.policy_accept.env.setBufferSize( self.particles[P].getBufferSize() )

        # Set to NaN the "next time" of the event just applied at the given server
        if self.LOG:    # True
            print("\nP={}: Applied {} @t={:.3f} to server {}...".format(P, type_of_event, relative_time_of_event, server) + (self.particles[P].getLastChange(server) == 0 and " --> NO CHANGE!" or ""))
        if type_of_event == Event.BIRTH:
            # Remove the job from the server's queue, as it is now being PROCESSED by the server
            # It's important to keep in the servers's queue JUST the information about the jobs WAITING
            # and NOT those being served because the NEXT BIRTH TIME for each server is chosen based on the
            # times of the WAITING jobs (see _generate_birth_times()). 
            if DEBUG_TIME_GENERATION:
                print("\n_apply_next_event: P={}, server={}: BIRTH - Job started at time t={:.3f}".format(P, server, relative_time_of_event))
            # DM-2021/05/10-START: Activate the following lines when assigning job classes to server using the assignment policy 
            #self.times_in_server_queues[P][server].pop(0)
            #self.jobclasses_in_server_queues[P][server].pop(0)
            # DM-2021/05/10-END
            self.times_next_birth_events[P][server] = np.nan
        elif type_of_event == Event.DEATH:
            if DEBUG_TIME_GENERATION:
                print("\n_apply_next_event: P={}, server={}: DEATH - Job finished at time t={:.3f}".format(P, server, relative_time_of_event))
            self.times_next_death_events[P][server] = np.nan

        if DEBUG_TIME_GENERATION:
            print("\n\n***** _apply_next_event: P={}: NEW STATE: {} @t={:.3f}*****".format(P, self.particles[P].getServerSizes(), relative_time_of_event))
            print("Jobs in server queues:".format(P))
            print("times:")
            for s in range(self.nservers):
                print("\t{}: {}".format(s, self.times_in_server_queues[P][s]))
            print("job classes:")
            for s in range(self.nservers):
                print("\t{}: {}".format(s, self.jobclasses_in_server_queues[P][s]))
            print("Next birth times: {}".format(self.times_next_birth_events[P]))
            print("Next death times: {}".format(self.times_next_death_events[P]))
            print("*******************************************************")
            print("\n\n")

        # Update the latest (super-absolute) time in which we know the state of the system
        self.times_latest_known_state[P] = time_of_event

        # Increase the events counter over ALL particles
        self.nevents += 1

        # Increase the iteration counter because it is associated to the application of a new event in ONE server
        self.iterations[P] += 1

        # Update the position information of the particle so that we can plot the trajectories
        # and we can compute the statistics for the estimations that are based on the occurrence
        # of special events like activation, blocking, etc. 
        self._update_particle_position_info(P, server=server)

        # Update the ACTIVE condition of the particle
        self._update_active_particles(P)

        # Update information that tracks absorptions and activations
        if self._has_particle_become_absorbed(P):
            # The time of absorption should be passed as super-absolute time
            # because the self.times0[P] quantity that keeps track of the latest absorption time
            # is NEVER reset, i.e. it is always stored as a super-absolute time. 
            self._update_absorption_times(P, time_of_event)

        if self._has_particle_become_activated(P):
            # Update the time of the latest activation
            # Note that this is always stored as a super-absolute time. 
            self.times1[P] = time_of_event

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

        #-- Update the return time to the positions to observe if the particle has returned to one of them
        if self._has_particle_returned_to_positions(P, self.positions_observe):
            self._update_return_time(P, self.times_buffer[P] + self.particles[P].getOrigin())

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

    def _update_return_time(self, P, time_of_event :float):
        """
        Updates the return times information of the particle based on the time of event
        and the current position of the particle after the latest event was applied.
        
        The return times info is updated IF the latest change in (buffer) position of the particle
        implies that it touches, at time_of_event, one of the positions to observe.
        
        P: non-negative int
            Particle number (index of the list of particle queues) to update.

        time_of_event: non-negative float
            Time of the event that may trigger the update of the list of return times.
            It should be given as an ABSOLUTE time of the event, as opposed to
            "relative to the paticle's time origin" stored in self.particles[P].
        """
        assert self.isValidParticle(P), "The particle number is valid (0<=P<{}) ({})".format(self.N, P)

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

    def _generate_trajectory_until_burnin_activation_or_end_of_simulation(self, P, end_of_simulation):
        """
        Generates the trajectory for the given particle until it is ACTIVATED AFTER
        the number of burn-in cycles have been observed,
        or until the max simulation time is reached, whatever comes first.
        
        This function is only called in the FV estimation case, NOT in the MC estimation case.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to simulate.

        end_of_simulation: int
            Whether the end of the simulation has been reached for the particle.

        Return: int
        The updated end_of_simulation variable based on the iterations carried out. 
        """
        assert self.reactivate, "The simulation is run until ACTIVATION only for the FV estimation case (i.e. when reactivate=True)"
        activated_just_after_burnin_absorption_cycles = lambda P: self._check_particle_state_based_on_most_recent_change(P, EventType.ACTIVATION) and (self.burnin_cycles_absorption == 0 or self.ktimes_n[P] == self.burnin_cycles_absorption)

        while not end_of_simulation and not activated_just_after_burnin_absorption_cycles(P):
            end_of_simulation = self.generate_one_iteration(P)

        if end_of_simulation:
            if self.LOG:
                print("P={}: END OF SIMULATION REACHED (latest buffer change time: {:.3f} (max={}), event #: {} (max={}), buffer size={})" \
                      .format(P, self.times_buffer[P], self.maxtime, self.nevents, self.getMaxNumberOfEvents(), self.particles[P].getBufferSize()))
            self._update_trajectories_for_end_of_simulation(P)  # Done for informational and completeness purposes
            assert not self.is_fv_ready[P], "Particle P={} that has not completed the burn-in process is NOT ready for the FV process and is marked to NOT be used for such process"
        else:
            # Particle has been activated
            # => Reset the position information to the activation time
            #    so that we can start with the Fleming-Viot process where
            #    all particles are assumed to start at the same time (t=0).
            assert self.particles[P].getMostRecentEventTime() == self.times_buffer[P]
            if self.LOG:
                print("P={}: FIRST ACTIVATION AFTER BURN-IN (#burn-in cycles={}) ABSORPTION CYCLES REACHED at time t={:.3f} (max={}), event #={} (max={}), buffer size={}, previous BS={})" \
                      .format(P, self.burnin_cycles_absorption, self.particles[P].getMostRecentEventTime(), self.maxtime, self.nevents, self.getMaxNumberOfEvents(), self.particles[P].getBufferSize(), self._get_previous_buffer_size(P)))
                print("\tTime of latest ABSORPTION t={:.3f}".format(self.times0[P]))
            activation_state = self.particles[P].getServerSizes()

            # Store the time at which the FV process starts for each particle
            # This is used in compute_counts_blocked_particles() in order to refer the observed blocked time segments
            # to this origin, since absorption times are referred to this origin as well
            # (see _record_first_observed_absorption_time_after_burnin_cycles() to see that this happens)
            self.times_start_fv[P] = self.times1[P]

            # Store the information that states that the particle can be used for the FV process
            # DM-2021/09/14: This was commented out because at this point the particle is NOT yet ready
            # to start the FV process... it still needs to be absorbed once after the activation, so that
            # we know the maximum observed survival time (among all particles) which defines the simulation
            # time for the FV process.
            # So, this change of status for the particle is being moved to the
            # _generate_trajectory_until_absorption_or_end_of_simulation() below.
            #self.is_fv_ready[P] = True

            # Update the distribution of activation states
            self._update_activation_distribution(activation_state)

            self._reset_birth_and_death_times(P)

        # Reset the particle information so that it is clear that the burn-in period has ended
        # and we don't have information stored about the particle's position that may generate confusion
        # or assertions to fail when analyzing the end of the FV process.
        # (recall that this function is only run in the FV process!)
        # This is particularly important when not ALL N particles are used in the FV process,
        # meaning that the particles that are NOT used should be still reset, o.w. they would contain
        # trajectory information that is going to distort the compute_counts() and/or the  finalize() process. 
        self.reset_particles_position_info(P, EventType.ACTIVATION)

        return end_of_simulation

    def _generate_trajectory_until_absorption_or_end_of_simulation(self, P, end_of_simulation):
        """
        Generates the trajectory for the given particle until it is absorbed
        or until the max simulation time is reached, whatever comes first.
        
        This function is expected to be called repetitively, after each subsequent absorption
        and until the max simulation time is reached for the given particle, as follows:
        - when reactivate=True: once the FV process has started, i.e. AFTER the number
                                of burn-in cycles have been observed.
        - when reactivate=False: after the first absorption has occured; this is not a condition
                                 that is related to the non-reactivate scenario itself, but simply
                                 because in this way, simulating the particle's trajectory can be
                                 seamlessly combined with the case when reactivate=True.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to simulate.

        end_of_simulation: int
            Whether the end of the simulation has been reached for the particle.

        Return: int
        The updated end_of_simulation flag indicating whether the max simulation time
        has been reached for this particle and therefore we should no longer continue simulating
        on this particle (i.e. this function should not be called again for this particle). 
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
                print("P={}: END OF SIMULATION REACHED (latest buffer change time: {:.3f} (max={}), event #: {} (max={}), buffer size={})" \
                      .format(P, self.times_buffer[P], self.maxtime, self.nevents, self.getMaxNumberOfEvents(), self.particles[P].getBufferSize()))
            self._update_trajectories_for_end_of_simulation(P)  # Done for informational and completeness purposes
        else:
            # Particle has been absorbed
            # => Store the information about the absorption
            assert self.particles[P].getMostRecentEventTime() == self.times_buffer[P]
            if self.LOG:
                print("P={}: STOPPED BY ABSORPTION at time t={:.3f} (absolute t={:.3f} of max time={:.1f}), event #={} (max={}), buffer size={}, previous BS={})" \
                      .format(P, self.particles[P].getMostRecentEventTime(), self.particles[P].getMostRecentEventTime() + self.get_start_time_fv(P), self.maxtime, self.nevents, self.getMaxNumberOfEvents(), self.particles[P].getBufferSize(), self._get_previous_buffer_size(P)))

            # Update the list of absorption times that are used to continue the simulation until end of the simulation,
            # and run the FV reactivation process if reactivate=True 
            absorption_time = self.particles[P].getMostRecentEventTime()
            self._update_info_absorption_times(P, absorption_time, list( self.particles[P].getServerSizes() ))

            if self.reactivate:
                # Record this first VALID absorption time in case we are doing all-in-one estimation
                # (i.e. estimating E(T) and P(T>t) as well as Phi(t) in the same simulation run)  
                self._record_first_observed_absorption_time_after_burnin_cycles(P)

            if False:
                print("\t--> total killing time after (t,n): {:.1f}, {}".format(self.ktimes_sum[P], self.ktimes_n[P]))

        return end_of_simulation

    def _update_trajectories_for_end_of_simulation(self, P):
        """
        Extends the trajectories of all servers to the latest observed time for the particle.
        The goal is to extend the trajectories of ALL servers to the latest time the state of the system is known,
        mostly for plotting purposes.        
        """
        for server in range(self.nservers):
            assert self.particles[P].getMostRecentEventTime() == 0.0 or self.particles[P].getMostRecentEventTime() == self.times_buffer[P], \
                 "The most recent event time for the particle coincides with the last time when the buffer size changed whenever that most recent time is positive (self.times_buffer[P]):" \
                 "\nP={}, server={}: most recent event time={:.3f}; last time the buffer changed size={:.3f}" \
                 .format(P, server, self.particles[P].getMostRecentEventTime(), self.times_buffer[P])
            self.trajectories[P][server]['t'] += [ self.times_buffer[P] ]
            self.trajectories[P][server]['x'] += [ self.trajectories[P][server]['x'][-1] ]  # The last known position of the server at the current time self.times_buffer[P]
            self.trajectories[P][server]['e'] += [ Event.END ]

    def generate_trajectories_until_end_of_simulation(self):
        "Generates the trajectories until the end of the simulation time is reached"
        step_fraction_process_to_report = 0.1
        next_fraction_process_to_report = step_fraction_process_to_report
        if self.LOG:
            print("...completed {:.1f}% of simulation time T={} and {:.1f}% of max #events={}." \
                  .format(self.maxtime is None and np.nan or self.get_time_latest_known_state() / self.maxtime * 100, self.maxtime,
                          self.getMaxNumberOfEvents() == np.Inf and np.nan or self.nevents / self.getMaxNumberOfEvents() * 100, self.getMaxNumberOfEvents()))
        while len(self.dict_info_absorption_times['t']) > 0:
            # There are still particles that have been absorbed and thus their simulation stopped
            # => We need to generate the rest of the trajectory for each of those particles
            # and how the trajectory is generated depends on whether we are in reactivate mode
            # (Fleming-Viot) or not (Monte-Carlo). This is why we "stop" at every absorption event
            # and analyze how to proceed based on the simulation mode (reactivate or not reactivate).
            fraction_completed =    self.getMaxNumberOfEvents() <  np.Inf and self.nevents / self.getMaxNumberOfEvents() or \
                                    self.getMaxNumberOfEvents() == np.Inf and self.get_time_latest_known_state() / self.maxtime

            if fraction_completed > next_fraction_process_to_report:
                if self.LOG:
                    print("...completed {:.0f}%".format(next_fraction_process_to_report*100))
                next_fraction_process_to_report += step_fraction_process_to_report
            if False:
                assert self.dict_info_absorption_times['t'] == sorted(self.dict_info_absorption_times['t']), \
                    "The absorption times are sorted: {}".format(self.dict_info_absorption_times['t'])
            if False:
                np.set_printoptions(precision=3, suppress=True)
                print("\n******** REST OF SIMULATION STARTS (#absorption times left: {}, MAXTIME={}, MAX #EVENTS={}) **********".format(len(self.dict_info_absorption_times['t']), self.maxtime, self.getMaxNumberOfEvents()))
                print("Absorption times and particle numbers (from which the first element is selected):")
                print(self.dict_info_absorption_times['t'])
                print(self.dict_info_absorption_times['P'])
                np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)
            # Remove the first element of the dictionary which corresponds to the particle
            # with the smallest absorption time
            # The fact that the selected particle is the one with the smallest absorption time
            # guarantes that there are N-1 particles to choose from for reactivation of the particle,
            # since all other N-1 particles have already a known trajectory at time of absorption
            # of the particle to be reactivated.
            t0 = self.dict_info_absorption_times['t'].pop(0)
            P = self.dict_info_absorption_times['P'].pop(0)
            idx_state = self.dict_info_absorption_times['S'].pop(0)
            assert P in self.particles_to_simulate, "The particle number to reactivate (P={}) is among the particles to simulate".format(P)
            if not self.is_expected_absorption_time_estimated_from_burnin:
                self._update_absorption_distribution_from_state_index(idx_state)
            if self.LOG:
                absorption_state = self.states_absorption[idx_state]
                print("Popped absorption time {:.3f} for particle P={} absorbed at state {}".format(t0, P, absorption_state))

            if self.getMaxNumberOfEvents() == np.Inf:
                # The end of the simulation is defined by reaching the maximum simulation time, NOT by reaching the maximum number of events
                assert t0 < self.maxtime, \
                    "The currently processed absorption time ({:.3f}) is smaller than the max simulation time ({:.3f})" \
                    .format(t0, self.maxtime)

            end_of_simulation = False
            if self.reactivate:
                if self.burnin_cycles_complete_all_particles:
                    assert np.sum(self.is_fv_ready) == self.N, "All N={} particles are being simulated in FV regime".format(self.N)
                # Choose a particle to which the absorbed particle is reactivated
                Q = self._choose_assigned_particle(P)
                if self.LOG:
                    print("Assigned particle number: Q={}".format(Q))
                    if False:
                        print("Trajectory of Q:\n{}".format(np.c_[self.trajectories[Q][0]['t'], self.trajectories[Q][0]['x']]))
                        ## In the trajectory print above, we just look at one server
                        ## (o.w. printing the values of all servers in column-wise way is tricky.
                position_Q_at_t0, position_change = self._create_new_particle(P, Q, t0)

                # IMPORTANT: The update of the INFO particles should come AFTER the update of the particle's position
                # because the INFO particles stores information about special events of the particle (activation, absorption, etc.)
                # and these special events are checked based on the information stored in the self.particles[P] attribute
                # which is updated by the _set_new_particle_position() call. 
                self._set_new_particle_position(P, t0, position_Q_at_t0)
                self._reset_birth_and_death_times(P)    # NOTE: This reset is done because conceptually a reactivated particle is like a re-born particle, so the simulation should start from scratch.
                self._update_info_particles_and_blocking_statistics(P)
            else:
                # In NO reactivation mode, we need to generate ONE iteration before CONTINUING with the simulation below,
                # otherwise the assertion that the next absorption time is different from the previous one
                # in _update_absorption_times() the next time this function is called, fails.
                end_of_simulation = self.generate_one_iteration(P)

            # Continue generating the trajectory of the particle after the latest absorption/reactivation
            # until a new absorption occurs or until the max simulation time is reached
            self._generate_trajectory_until_absorption_or_end_of_simulation(P, end_of_simulation)

    def _choose_assigned_particle(self, P):
        """
        Chooses a particle among all other eligible particles in the system
        to be assigned to the given (absorbed) particle.
        
        A particle is considered to be eligible if it has been activated after the burn-in period
        which implies that it can be considered for the Fleming-Viot process, since for the FV process
        we require that the start state follow the stationary distribution of the exit state
        from the set of absorbed states A.
        """
        # We choose a particle among all eligible particles, namely:
        # - it is NOT the particle that needs to be reactivated (P)
        # - it has been activated after the burn-in period, meaning that it start state
        # is expected to come from the distribution of exit states from the set of absorbed states A.

        #list_of_particles_to_choose_from = list(range(P)) + list(range(P+1, self.N))
        list_of_particles_to_choose_from = [Q for Q in self.particles_to_simulate if Q != P]
        n_particles_to_choose_from = len(list_of_particles_to_choose_from)
        assert n_particles_to_choose_from >= 1
        chosen_particle_number = list_of_particles_to_choose_from[ np.random.randint(0, n_particles_to_choose_from-1) ]

        assert self.isValidParticle(chosen_particle_number) and chosen_particle_number != P, \
                "The chosen particle is valid (0 <= Q < {} and different from {}) ({})" \
                .format(self.N, P, chosen_particle_number)
        return chosen_particle_number

    def _create_new_particle(self, P, Q, t):
        """
        Creates a new particle in the system at the given time t,
        whose position starts at the position of particle Q assigned to the absorbed particle P after reactivation.
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
                "The particle numbers representing the absorbed (P) and reactivated (Q) particles are among the particles to simulate --defined in self.particles_to_simulate" \
                "\n(P={}, Q={}, N={})" \
                .format(P, Q, self.N)

        # Particle IDs in the info_particles list associated to particle numbers P and Q
        # to which the absorbed particle P is reactivated.
        p = self.particle_reactivation_ids[P]
        q = self.particle_reactivation_ids[Q]
        assert self.isValidParticleId(q), "The particle ID is valid (0<=q<{}) ({})" \
                .format(len(self.info_particles), q)

        # Add a new entry to the info_particles list that represents the reactivated particle
        # Its position is assigned to the position of the reassigned particle Q at time t,
        # and the information about the event times ('t'), event types ('E'), and particle state ('S') are empty
        # (because the information about the new position is defined OUTSIDE this method
        # by the method _update_info_particles_and_blocking_statistics(), called just after this method is called,
        # because that method is called by other "processes" as well).
        position_Q = self.get_position(Q, t)
        self.info_particles += [ dict({'t': [],
                                       'E': [],
                                       'S': [],
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

        # Reset the following information about arriving jobs and server queues in particle P
        # (which are tantamount to the fact that, conceptually, a reactivated particle is re-born,
        # i.e. the simulation of its trajectory should start from scratch): 
        # - the times and job classes in each server's queue are reset to empty,
        # i.e. the same situation we had at the beginning of the simulation.
        # - the times of the latest arrived jobs to the particle's buffer are set to
        # the absorption time, i.e. the same situation we had at the beginning of the simulation,
        # EXCEPT that we shift the times of the latest arrived job for each class to the current time
        # (the absorption time), since this is the reference time used to define the absolute time of
        # the next event as the simulation for this particle proceeds.
        # NOTE that the absorption time set as time of latest arrived job is a super-absolute time
        # since this is how birth times are measured and stored in the object.
        self.times_in_server_queues[P] = array_of_objects((self.nservers,), dtype=list, value=[])
        self.jobclasses_in_server_queues[P] = array_of_objects((self.nservers,), dtype=list, value=[])
        self.times_latest_arrived_jobs[P] = (t + self.particles[P].getOrigin()) * np.ones((len(self.job_class_rates),), dtype=float)

        if self.LOG: #True:
            np.set_printoptions(precision=3, suppress=True)
            print("\nInfo PARTICLES for particle ID={}, after particle P={} reactivated to particle Q={} (ID q={}) at time t={}:".format(q, P, Q, q, t))
            print("\ttimes: {}".format(self.info_particles[q]['t']))
            print("\tevents: {}".format(self.info_particles[q]['E']))
            #print(np.c_[np.array(self.info_particles[q]['t']), np.array(self.info_particles[q]['E'])])
            np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

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
        assert self.isValidParticle(P), "The particle number is valid (P={}, N={})".format(P, self.N)
        assert self.buffer_size_activation <= np.sum(position) <= self.particles[P].getCapacity(), \
                "The buffer size of the reactivated particle {} ({}) is in the set of non-absorption states (s in interval [{}, {}])" \
                .format(P, position, self.buffer_size_activation, self.particles[P].getCapacity())
        if self.getMaxNumberOfEvents() == np.Inf:
            assert t < self.maxtime, \
                "The time at which the position is updated ({:.3f}) is smaller than the maximum simulation time ({:.3f})" \
                .format(t, self.maxtime)

        # Update the situation of the queue represented by the particle
        self.particles[P].resize(position, t)

        # Reactivation resets the iterations run on the reactivated particle
        # because it's like starting again. In fact, all the server times now
        # are ALIGNED to the SAME reactivation time, and this is like the very beginning
        # of the simulation, where all server times were at 0.
        # This is only important if we want to pass assertions on the most recent
        # event times and next event times (see self.assertTimeMostRecentEventAndTimesNextEvents())
        self.iterations[P] = 0

        #-- Update current and historical information
        # Information by server
        self.times_by_server[P] = np.repeat(t, self.nservers)             # NOTE that this time is the same for ALL servers, a situation that, under normal simulation without reactivation, only happens at the beginning of the simulation (time=0)
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

    def _reset_birth_and_death_times(self, P):
        """
        Resets the next birth and death times for all servers to NaN for the given particle

        This is typically called after reactivation of a particle, when the particle
        changes position abruptly and therefore we need to reset all next event times in each server
        to NaN because it's like starting a new trajectory from scratch.
        """
        self.times_next_birth_events[P] = np.nan*np.ones((self.nservers,), dtype=float)
        self.times_next_death_events[P] = np.nan*np.ones((self.nservers,), dtype=float)

        # Also reset the times of next events that are computed from the above birth and death times
        self.times_next_event = np.nan*np.ones(self.N, dtype=float)
        self.times_next_events2 = np.zeros((self.N, self.nservers, 2), dtype=float) # `2` is the number of possible events: birth and death

    def _update_info_particles_and_blocking_statistics(self, P):
        """
        Updates the info_particles attribute with the new particle's position
        and any special state that the particle goes into as a result of the last position change,
        such as ACTIVATION, ABSORPTION, BLOCK, UNBLOCK, START_POSITION.
        
        Note that the particle can have more than one event type to store.  
        """
        assert self.isValidParticle(P), "The particle to update is valid (P={}, N={})".format(P, self.N)

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
            self.info_particles[p]['S'] += [ list(self.particles[P].getServerSizes()) ]

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

    def _compute_order(self, arr, axis=-1):
        "Returns the order of the values in an array by the specified axis dimension"
        # axis=-1 means "the last dimension of the array"
        return np.argsort(arr, axis=axis)

    def _get_previous_buffer_size(self, P):
        """
        Returns the position (buffer size) before the latest change taking place in the particle.
        
        The previous buffer size is equal to the current buffer size if the latest change was 0.
        (which happens ONLY when a server at maximum capacity experiences a BIRTH event, as
        no DEATH events take place when the server's queue is empty because the time for the next death event
        for such server is NaN).

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

    def _is_particle_absorbed(self, P):
        "Whether the particle is absorbed (regardless of the latest change experienced by it)"
        return self.particles[P].getBufferSize() in self.set_absorbed_states

    def _is_particle_activated(self, P):
        "Whether the particle is activated (regardless of the latest change experienced by it)"
        return self.particles[P].getBufferSize() in self.set_activation

    def _is_particle_blocked(self, P):
        "Whether the particle is blocked (regardless of the latest change experienced by it)"
        return self.particles[P].getBufferSize() == self.particles[P].getCapacity()

    def _is_particle_unblocked(self, P):
        "Whether the particle is unblocked (regardless of the latest change experienced by it)"
        return self.particles[P].getBufferSize() < self.particles[P].getCapacity()

    def _has_particle_become_absorbed(self, P):
        """
        Whether the particle has just become absorbed, which is indicated by the fact that
        before the latest change, the particle's position (buffer size) is NOT in the set of absorbed buffer sizes
        and now is one of the absorbed buffer sizes.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.
        """
        assert self.isValidParticle(P), "The particle number is valid (P={}, N={})".format(P, self.N)

        previous_buffer_size = self._get_previous_buffer_size(P)

        return previous_buffer_size not in self.set_absorbed_states and self._is_particle_absorbed(P)

    def _has_particle_become_activated(self, P):
        """
        Whether the particle has just become activated, which is indicated by the fact that
        before the latest change, the particle's position (buffer size) is NOT in the set of activation buffer sizes
        and now is one of the activation buffer sizes.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.
        """
        assert self.isValidParticle(P), "The particle number is valid (P={}, N={})".format(P, self.N)

        previous_buffer_size = self._get_previous_buffer_size(P)

        return previous_buffer_size not in self.set_activation and self._is_particle_activated(P)

    def _has_particle_become_blocked(self, P):
        """
        Whether the particle has just become blocked, which is indicated by a positive change in the system's buffer size
        and a system's buffer size equal to its capacity after the change.
        
        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.
        """
        assert self.isValidParticle(P), "The particle number is valid (P={}, N={})".format(P, self.N)

        previous_buffer_size = self._get_previous_buffer_size(P)

        return previous_buffer_size < self.particles[P].getCapacity() and self._is_particle_blocked(P)

    def _has_particle_become_unblocked(self, P):
        """
        Whether the particle has just become unblocked, which is indicated by a negative change in the system's buffer size
        and a system's buffer size equal to its capacity before the change.

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.
        """
        assert self.isValidParticle(P), "The particle number is valid (P={}, N={})".format(P, self.N)

        previous_buffer_size = self._get_previous_buffer_size(P)

        return previous_buffer_size == self.particles[P].getCapacity() and self._is_particle_unblocked(P)

    def _has_particle_returned_to_positions(self, P, positions :list):
        """
        Whether the particle has just returned to any of the given positions (buffer sizes)

        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) to be checked.

        positions: list
            List of positions to check.
        """
        assert self.isValidParticle(P), "The particle number is valid (P={}, N={})".format(P, self.N)
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
        assert self.isValidParticle(P), "The particle number is valid (P={}, N={})".format(P, self.N)
        if event_type == EventType.ABSORPTION:
            return self._has_particle_become_absorbed(P)
        elif event_type == EventType.ACTIVATION:
            return self._has_particle_become_activated(P)
        elif event_type == EventType.BLOCK:
            return self._has_particle_become_blocked(P)
        elif event_type == EventType.UNBLOCK:
            return self._has_particle_become_unblocked(P)

    def _update_info_absorption_times(self, P :int, time_of_absorption :float, state_of_absorption :list):
        "Inserts a new absorption time in order, together with its particle number"
        assert self.isValidParticle(P), "The particle number is valid (P={}, N={})".format(P, self.N)
        idx_insort, found = insort(self.dict_info_absorption_times['t'], time_of_absorption, unique=False)
        assert not found, "The inserted absorption time ({:.3f}) does NOT exist in the list of absorption times:\n{}".format(time_of_absorption, self.dict_info_absorption_times['t'])
        self.dict_info_absorption_times['P'].insert(idx_insort, P)
        try:
            idx_state_of_absorption = self.states_absorption.index(state_of_absorption)
            ## IMPORTANT: In order for the state_of_absorption to be findable in the list of absorption states
            ## it MUST be a list!! (as opposed to e.g. a numpy array)
        except:
            raise Warning("The given state of absorption ({}) has not been found among the valid absorption states:\n{}" \
                          .format(state_of_absorption, self.states_absorption))
        self.dict_info_absorption_times['S'].insert(idx_insort, idx_state_of_absorption)

    def _record_first_observed_absorption_time_after_burnin_cycles(self, P):
        """
        Assuming reactivate=True, the first absorption (once all burn-in absorption cycles
        have been observed) is used as a record for the estimation of E(T), the expected absorption cycle time,
        and for the estimation of P(T>t), the survival probability.
        """
        # TODO-2021/09/14: Implement the case where we still record a survival time, even when a full survival time is NOT observed (because the end of simulation was reached before that happening)
        # This implementation would correspond to the case when we want to accept erring on increased BIAS as opposed to
        # increased VARIANCE in the estimation of P(T>t).
        # See how we deal with this topic in the context of the estimation of E(T) in the
        # estimate_expected_absorption_time_from_burnin() function where we define the ErrOnTheSideOfIncreased Enum.    
        assert self.reactivate
        is_this_the_first_after_last_required_absorption_cycle = lambda: self.ktimes_n[P] == self.burnin_cycles_absorption + 1
        if is_this_the_first_after_last_required_absorption_cycle():
            # Store the time to absorption as a valid measure to estimate E(T)
            # (if this can be estimated as defined by the is_expected_absorption_time_estimated_from_burnin attribute)
            # and as a valid measure to estimate P(T>t), which can ALWAYS be estimated
            # regardless of the number of required absorption cycles, since
            # observing the distribution of activation states does NOT depend on this
            # value; in fact, we observe its distribution when the particle goes
            # from the set of absorbed states to the set of active states, and this
            # transition is observed even when burnin_cycles_absorption = 0.
            if self.is_expected_absorption_time_estimated_from_burnin:
                assert np.abs(self.times0[P] - (self.particles[P].getMostRecentEventTime() + self.particles[P].getOrigin())) < 0.1*self.EPSILON_TIME, "times0={:.6f}, most recent time from particles t={:.6f}".format(self.times0[P], self.particles[P].getMostRecentEventTime() + self.particles[P].getOrigin())
                self.absorption_cycle_times[P] = self.ktimes[P]
            if self.is_proba_survival_estimated_from_burnin:
                # Mark the particle as ready to start the FV process
                # (and to contribute in the calculation of the maximum simulation time for the FV process)
                self.is_fv_ready[P] = True
                if DEBUG_SPECIAL_EVENTS:
                    print("P={}: activation time: t={:.3f}, absorption time: t={:.3f}, difference: t={:.3f}".format(P, self.get_start_time_fv(P), self.times0[P], self.times0[P] - self.get_start_time_fv(P)))
                self.insert_relative_time_from_activation_to_absorption(self.times0[P], [self.times_start_fv[P]], [EventType.ABSORPTION])
    #------------ LOW LEVEL SIMULATION -------------
    #--------------------------------- Functions to simulate ----------------------------------


    #------------------- Functions to store information about the simulation ------------------
    def _update_absorption_times(self, P, time_of_absorption):
        ref_time = 0.0 if np.isnan(self.times0[P]) else self.times0[P]

        # Refer the absorption time to the latest time the particle was absorbed
        # and add it to the previously measured absorption time
        assert time_of_absorption > ref_time, \
                "The time of absorption ({}) just observed" \
                " is larger than the latest time of absorption for particle {} ({})" \
                .format(time_of_absorption, P, ref_time)
        time_to_absorption = time_of_absorption - ref_time
        self.ktimes[P] = time_to_absorption
        self.ktimes_sum[P] += self.ktimes[P]
        self.ktimes_n[P] += 1

        if self.is_expected_absorption_time_estimated_from_burnin:
            # Update the distribution of the absorption states as the k-th absorption state is observed
            # where k is the number of burn-in cycles.
            if self.ktimes_n[P] == self.burnin_cycles_absorption:
                absorption_state = self.particles[P].getServerSizes()
                self._update_absorption_distribution(absorption_state)

        if DEBUG_SPECIAL_EVENTS:
            np.set_printoptions(precision=3, suppress=True)
            print("\n>>>> Particle P={}: absorption @t={:.3f}".format(P, time_of_absorption))
            print(">>>> Previous absorption time: {:.3f}".format(self.times0[P]))
            print(">>>> Time to absorption: {:.3f}".format(self.ktimes[P]))
            print(">>>> Total absorption cycle times for ALL particles: {}".format(np.array(self.ktimes_sum)))
            print(">>>> Total absorption cycles for ALL particles: {}".format(np.array(self.ktimes_n)))
            np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

        # Update the attribute that holds the latest time the particle was absorbed
        self.times0[P] = time_of_absorption

        if False:
            print(">>>> P={}: UPDATED Last time at 0: {:.3f}".format(P, self.times0[P]))
            print(">>>> \tNew total absorption cycle times (ktimes_sum): {:.3f}".format(self.ktimes_sum[P]))

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
            np.set_printoptions(precision=3, suppress=True)
            print("\n>>>> Particle P={}: unblocking @t={:.3f}".format(P, time_of_unblocking))
            print(">>>> Previous blocking times: {:.3f}".format(self.timesb[P]))
            print(">>>> \tNew total blocking time (btimes_sum): {:.3f}".format(self.btimes_sum[P]))
            np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

    def _update_activation_distribution(self, activation_state):
        "Updates the activation distribution based on the observed given activation state"
        try:
            idx_state_activation = self.states_activation.index(list( activation_state ))
            ## IMPORTANT: In order for the state to be findable in the list of activation states
            ## the state MUST be passed as a LIST (not as a numpy.array) to the index() function!!
        except:
            raise Warning("The given activation state ({}) has not been found among the valid activation states:\n{}" \
              .format(activation_state, self.states_activation))
        self.dist_activation_states[idx_state_activation] += 1
        if False:
            print("Activation at state {}".format(activation_state))

    def _update_absorption_distribution(self, absorption_state):
        "Updates the absorption distribution based on the observed given absorption state"
        try:
            idx_state_absorption = self.states_absorption.index(list( absorption_state ))
            ## IMPORTANT: In order for the state to be findable in the list of absorption states
            ## the state MUST be passed as a LIST (not as a numpy.array) to the index() function!!
        except:
            raise Warning("The given absorption state ({}) has not been found among the valid absorption states:\n{}" \
              .format(absorption_state, self.states_absorption))
        self._update_absorption_distribution_from_state_index(idx_state_absorption)
        if False:
            print("Absorption at state {}".format(absorption_state))

    def _update_absorption_distribution_from_state_index(self, idx_state_absorption):
        """
        Updates the absorption distribution based on the observed given absorption state INDEX
        (as opposed to state tuple)
        """
        assert 0 <= idx_state_absorption < len(self.dist_absorption_states), \
            "The absorption state index to udpate ({}) is valid (between 0 and {})" \
            .format(idx_state_absorption, len(self.dist_absorption_states)-1)
        self.dist_absorption_states[idx_state_absorption] += 1
    #------------------- Functions to store information about the simulation ------------------


    #--------------------------- Functions to wrap up the simulation --------------------------
    def finalize(self):
        """
        Finalizes the simulation process by treating particles that are active,
        i.e. those particles providing CENSORED survival and absorption cycle time measurements.
        
        How finalization is carried out and what particles are to be finalized
        is defined by the object's attribute `finalize_info`, which defines the 'type'
        of finalization and the 'condition' (on the last state of the particle) for finalization.
        """
        particle_numbers_to_finalize = self.get_censored_particle_numbers()
        particle_ids_to_finalize = list( np.sort([ self.particle_reactivation_ids[P] for P in particle_numbers_to_finalize ]) )

        finalize_process = "--NO FINALIZATION DONE--"
        if self.LOG or DEBUG_TRAJECTORIES: #True
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
            info_part = self.info_particles[p - nremoved]
            P = info_part['particle number']
            self.assertAllServersHaveSameTimeInTrajectory(P)
            #print("Processing particle ID p={}, P={} out of {} particles".format(p, P, len(particle_ids_to_finalize)))
            if self.finalize_info['type'] == FinalizeType.REMOVE_CENSORED:
                if self.reactivate:
                    # Simply remove the particle from the analysis
                    # Note that if we are dealing with an ORIGINAL particle that has not been absorbed
                    # then the whole trajectory of the particle is removed, so we lose one of the particles
                    # that were initially consider in the simulation.
                    # This is OK. 
                    if self.LOG: #True:
                        print("...{} particle p={} (P={})".format(finalize_process, p, P))
                    # But first get the time of absorption leading to the reactivation of the
                    # particle ID to remove so that we set the latest time with info about P below
                    time_latest_absorption_P = self.info_particles[p - nremoved]['t0']
                    self.info_particles.pop(p - nremoved)
                    nremoved += 1
                        # NOTE that no update needs to be done to the quantities
                        # used to estimate E(T) (e.g. ktimes_sum) because these quantities are updated
                        # ONLY when there is an absorption, and this never happened
                        # for this particle ID (recall that, when reactivate=True,
                        # there is at most ONE absorption per particle ID,
                        # and when it occurs, it's the last event to occur).

                    # Update the latest time of known state for particle P to the absorption time
                    # leading to the particle ID p just removed
                    if time_latest_absorption_P is not None:
                        self.times_latest_known_state[P] = time_latest_absorption_P + self.particles[P].getOrigin()
                    else:
                        self.times_latest_known_state[P] = 0.0 + self.particles[P].getOrigin()
                    if self.LOG:
                        print("REMOVAL: Latest absorption time set for particle P={}: {:.3f}".format(P, self.times_latest_known_state[P]))
                else:
                    # Find the latest valid event time and remove all the events since then
                    if self.LOG:
                        print("...{} the tail of particle P={}".format(finalize_process, P))
                    if self.finalize_info['condition'] in [FinalizeCondition.ACTIVE, FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY]:
                        idx_last_valid_event = find_last_value_in_list(info_part['E'], EventType.ABSORPTION)
                    elif self.finalize_info['condition'] == FinalizeCondition.NOT_START_POSITION:
                        idx_last_valid_event = find_last_value_in_list(info_part['E'], EventType.START_POSITION)

                    # Remove all the elements AFTER the index of the last absorption to the end
                    # Note that if no valid last event is found, ALL the trajectory is removed
                    # and this works fine like this because find_last() returns -1 when the searched value
                    # is not found in the list.
                    event_time_block = None
                    if (self.LOG or DEBUG_TRAJECTORIES) and len(self.particles_to_simulate) == 1: #True and len(self.particles_to_simulate) == 1
                        print("P={}: Blocking time BEFORE removal: t={:.3f}, n={}".format(P, self.btimes_sum[P], self.btimes_n[P]))
                        #print(info_part)
                    while len(info_part['E']) > idx_last_valid_event + 1:
                        # Note that we remove the events belonging to the subtrajectory to remove
                        # from the EARLIEST to the LATEST event
                        # This has an impact in how we reduce the blocking time of possible blocking events
                        # in the subtrajectory being removed (i.e. we store "the event_time_block when a BLOCK event is found"
                        # as opposed to storing "the event_time_unblock when an UNBLOCK event is found",
                        # because when going from left to right in time, we first encounter a BLOCK event and THEN an UNBLOCK event. 
                        event_time = info_part['t'].pop(idx_last_valid_event+1)
                        event_types = info_part['E'].pop(idx_last_valid_event+1)
                        info_part['S'].pop(idx_last_valid_event+1)
                        if (self.LOG or DEBUG_TRAJECTORIES) and len(self.particles_to_simulate) == 1:  # True and len(self.particles_to_simulate) == 1
                            print("P={}: {} events at time {:.1f} removed.".format(P, event_types, event_time))
                        if list_contains_either(event_types, EventType.BLOCK):
                            event_time_block = event_time
                        elif list_contains_either(event_types, EventType.UNBLOCK) and event_time_block is not None:
                            # Decrease the total blocking time for particle
                            self.btimes_sum[P] -= event_time - event_time_block
                            self.btimes_n[P] -= 1
                            event_time_block = None
                        # NOTE that we should NOT update the killing/absorption times information
                        # (ktimes_sum and ktimes_n) because:
                        # - either the last valid event is an ABSORPTION, so there is no absorption event
                        # in the subtrajectory being removed.
                        # - either the possible ABSORPTION event in the subtrajectory being removed
                        # can be considered as a valid contribution to the killing time, because it was actually observed.
                        # Although technically the contribution of such event should be removed from the
                        # killing time value because we are REMOVING the subtrajectory containing
                        # such absorption event, but unfortunately, rolling back such contribution
                        # from the killing times is NOT SO EASY because the contribution by a single absorption event
                        # may be n-fold, i.e. one contribution for EVERY of n ACTIVATION events happening
                        # prior to the previous ABSORPTION event... So, in order to properly remove the contribution
                        # from the removed absorption event, we would need to go back
                        # and look for all ACTIVATION events happening before it, but after the previous ABSORPTION event!
                    if (self.LOG or DEBUG_TRAJECTORIES) and len(self.particles_to_simulate) == 1: #True and len(self.particles_to_simulate) == 1
                        print("P={}: Blocking time AFTER removal: t={:.3f}, n={}".format(P, self.btimes_sum[P], self.btimes_n[P]))
                        #print(info_part)

                    # Update the latest time of known state for particle P to the latest valid event time
                    if len(info_part['E']) > 0:
                        assert  self.finalize_info['condition'] in [FinalizeCondition.ACTIVE, FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY] and list_contains_either(info_part['E'][-1], EventType.ABSORPTION) or \
                                self.finalize_info['condition'] == FinalizeCondition.NOT_START_POSITION and list_contains_either(info_part['E'][-1], EventType.START_POSITION), \
                                "P={}: The last event stored in info_particles after removal of the censored part of the trajectory " \
                                "is an ABSORPTION for finalize condition=ACTIVE or NOT_ABSORBED_STATES_BOUNDARY, or START_POSITION for finalize condition=NOT_START_POSITION ({})" \
                                .format(P, info_part['E'][-1])
                        self.times_latest_known_state[P] = info_part['t'][-1] + self.particles[P].getOrigin()
                    else:
                        self.times_latest_known_state[P] = 0.0 + self.particles[P].getOrigin()
                    if self.LOG:
                        print("REMOVAL: Latest absorption time set for particle P={}: {:.3f}".format(P, self.times_latest_known_state[P]))

                if self.finalize_info['condition'] in [FinalizeCondition.ACTIVE, FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY]:
                    # Flag the particle as inactive (as the latest censored trajectory was removed)
                    self.is_particle_active[P] = False

            elif self.finalize_info['type'] in [FinalizeType.ABSORB_CENSORED, FinalizeType.ESTIMATE_CENSORED]:
                if self.LOG:
                    print("...{} particle p={}".format(finalize_process, p))
                # Increase slightly and randomly the time to insert the new "fictitious" event
                # in order to avoid repeated times in the list of time segments
                # (the random component is especially important here as it would affect
                # the insertion of a survival/absorption time segment which is done RELATIVE to
                # valid earlier activation events)
                # Note that having repeated time values may mess up the process that merges the
                # two list of times needed to compute the blocking probability, namely:
                # - the list of alive time segments
                # - the list of block/unblock time segments
                # The inserted time(s) happen AFTER the maximum simulation time,
                # regardless of the last observed event of the particle
                # (in fact, if the simulation has come to an end, it means that
                # the next event for EVERY particle would happen past the maximum simulation time)
                time_to_insert = self.times_latest_known_state[P] + self.generate_epsilon_random_time()

                # 2) Update the information of special events so that calculations are consistent
                # with how trajectories are expected to behave (e.g. they go through ACTIVATION
                # before going to the forced absorption)
                if self.finalize_info['type'] == FinalizeType.ABSORB_CENSORED:
                    assert self.finalize_info['condition'] == FinalizeCondition.ACTIVE, \
                        "We cannot absorb censored observations when the FinalizeCondition is not ACTIVE (because it would be too complicated to implement and not worth it)"
                    # To avoid assertion failures... make sure that, at the end of the simulation:
                    # - the particle is unblocked if it was blocked
                    #   (adding this preserves the assertion that the number of
                    #   blocked particles at the very last time is 0 --since the particles are all absorbed!)
                    # - the particle is activated if it was not at the activation position
                    #   (this makes things logical --i.e. before absorption the particle passes by the activation position)
                    if self._is_particle_blocked(P):
                        # Add a fictitious UNBLOCK time if the particle was BLOCKED
                        info_part['t'] += [time_to_insert]
                        info_part['E'] += [ [EventType.UNBLOCK_F] ]
                        info_part['S'] += [None]    # IMPORTANT: We should set a value to 'S', o.w. the loop in compute_counts() on zip(t, E, S) will NOT cover the last observation!! (because S would fall short of one element)
                        # Increase each of the times to insert for the insertion of the next event
                        time_to_insert += self.generate_epsilon_random_time()
                        # Update the total blocking time
                        self._update_blocking_time(P, time_to_insert)
                    if not self._is_particle_activated(P):
                        # Add a fictitious ACTIVATION time if the buffer size was not already in the ACTIVATION set
                        info_part['t'] += [time_to_insert]
                        info_part['E'] += [ [EventType.ACTIVATION_F] ]
                        info_part['S'] += [None]    # IMPORTANT: We should set a value to 'S', o.w. the loop in compute_counts() on zip(t, E, S) will NOT cover the last observation!! (because S would fall short of one element)
                        # Increase each of the time to insert for the insertion of the
                        # fictitious absorption event coming next (to avoid time value repetitions)
                        time_to_insert += self.generate_epsilon_random_time()

                    # Add the fictitious ABSORPTION time and update the information about the times to absorption
                    info_part['t'] += [time_to_insert]
                    info_part['E'] += [ [EventType.ABSORPTION_F] ]
                    info_part['S'] += [None]    # IMPORTANT: We should set a value to 'S', o.w. the loop in compute_counts() on zip(t, E, S) will NOT cover the last observation!! (because S would fall short of one element)
                    self._update_absorption_times(P, time_to_insert + self.particles[P].getOrigin())

                    # Flag the particle as inactive (as it was just absorbed)
                    self.is_particle_active[P] = False
                elif self.finalize_info['type'] == FinalizeType.ESTIMATE_CENSORED:
                    # Find the last valid special event based on the finalize condition (i.e. the censoring definition)
                    # in order to decide whether we need to use the last observed time in the simulation
                    # in estimations of whatever measure can be sensibly estimated from the given finalize condition
                    # i.e.:
                    # - if FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY => we are probably interested in estimating E(T)
                    #   => the quantity of interest for these estimations is:
                    #       - time to absorption from last ABSORPTION event (self.ktimes_sum)
                    # - if FinalizeCondition.NOT_START_POSITION => we may be interested in estimating the blocking probability by MC
                    #   => the quantity of interest for this estimation is:
                    #       - return time to start position (self.rtimes_obs_sum)

                    # Here we update the quantities that are updatable at this point,
                    # i.e. either self.ktimes_sum (for FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY)
                    # or self.rtimes_obs_sum (for FinalizeCondition.NOT_START_POSITION).
                    # (A non updatable quantity is the time to absorption from ACTIVATION because this can only be measured
                    # once we process the ACTIVATION times in compute_counts() below, and this is why
                    # the case FinalizeCondition.ACTIVE is not taken care of here.)
                    # The update is performed depending on whether the time elapsed since the last valid event is "large"
                    # compared to the corresponding average time observed so far.
                    # If it's large, it means that the censoring time should contribute to them because o.w.
                    # we would be underestimating the time of interest (killing or return)
                    # This typically happens when the simulation time is not long enough to adequately measure
                    # the time of interest.
                    # If it's NOT large, either:
                    # - it's very small meaning that censoring occured just because the last simulation chunk
                    # (since the last valid event based on the finalization condition) was too small to give us
                    # a sensible estimated value => the chunk MUST be discarded.
                    # - it's similar to the average value observed so far => the chunk CAN be discarded
                    # because including it will not change dramatically the estimated average.
                    if self.finalize_info['condition'] == FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY:
                        idx_last_valid_event = find_last_value_in_list(info_part['E'], EventType.ABSORPTION)
                        if idx_last_valid_event < 0:
                            # No valid event was found
                            # => set the last valid event time to the start of the simulation
                            last_valid_event_time = 0.0
                        else:
                            last_valid_event_time = info_part['t'][idx_last_valid_event]
                        duration_last_chunk = time_to_insert - last_valid_event_time
                        average_absorption_time = self.ktimes_sum / max(1, self.ktimes_n)
                        if duration_last_chunk > MULTIPLIER_FOR_LARGE_MEASUREMENT * average_absorption_time:
                            self._update_absorption_times(P, time_to_insert + self.particles[P].getOrigin())
                    elif self.finalize_info['condition'] == FinalizeCondition.NOT_START_POSITION:
                        idx_last_valid_event = find_last_value_in_list(info_part['E'], EventType.START_POSITION)
                        assert idx_last_valid_event >= 0, \
                            "There is always a valid event when FinalizeCondition.NOT_START_POSITION because the simulation starts at START_POSITION!"
                        last_valid_event_time = info_part['t'][idx_last_valid_event]
                        duration_last_chunk = time_to_insert - last_valid_event_time
                        average_return_time = self.rtimes_obs_sum / max(1, self.rtimes_obs_n)
                        if duration_last_chunk > 2 * average_return_time:
                            self._update_return_time(P, time_to_insert + self.particles[P].getOrigin())
                            ## NOTE that if the start position is NOT among the positions to observe (which determine the return time
                            ## measurements), it may well be the case that no return time had been observed so far
                            ## meaning that the above update will NOT contribute to updating the return times observed so far
                            ## (check out the definition of self._update_return_time() where at the beginning we check
                            ## whether the particle has already been at one of the positions to observe before updating the
                            ## return time information for the particle)
                            ## This if fine, because this very boundary situation is taken care of by the function that
                            ## estimates the blocking probability by MC (typically self.estimate_proba_blocking_mc()).

                    # Add the CENSORING event
                    info_part['t'] += [time_to_insert]
                    info_part['E'] += [ [EventType.CENSORING] ]
                    info_part['S'] += [None]    # IMPORTANT: We should set a value to 'S', o.w. the loop in compute_counts() on zip(t, E, S) will NOT cover the last observation!! (because S would fall short of one element)

        self.is_simulation_finalized = True

    def generate_epsilon_random_time(self):
        """
        We generate a very small time of the order of EPSILON_TIME
        with a random component to avoid a possible repetition of of time values
        which could complicate calculations.
        
        The randomly generated value is between 0.1*EPSILON_TIME and EPSILON_TIME
        so that we don't incurr into underflows.
        """
        return self.EPSILON_TIME * max( 0.1, np.random.random() )

    def compute_counts(self):
        """
        Computes the survival time segments and blocking time segments needed for
        the calculation of 'P(T>t)' and P(BLOCK / T>t, s=1). 
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
        for p, info_part in enumerate(self.info_particles):
            P = info_part['particle number']
            #print("Processing particle ID p={}, P={} out of {} particles".format(p, P, len(self.info_particles)))
            #print("Special events information:\n{}".format(info_part))
            #input("Press Enter to continue...")
            if P not in self.particles_to_simulate:
                # The current particle P has NOT been used in the FV process
                # => Do NOT consider it for the counts calculations
                continue

            # A FEW ASSERTIONS ABOUT HOW TRAJECTORIES OF THE PARTICLES --THAT ARE USED
            # FOR THE CALCULATION OF THE COUNTS-- END
            # This implies also checking that the finalization process by finalize() has being carried out correctly
            self.assertFinalizationProcessIsOk(P, p, info_part['E'])

            # List of observed activation times for the particle
            # Once the particle is absorbed, a survival time for each activation time
            # is added to the list of survival times given activation (that are used to estimate Pr(T>t / s=1))
            # The list of activation times is then reset to empty, once they are used
            # following an ABSORPTION event for the particle.
            activation_times = []
            event_types_prev = []
            for t, event_types, state in zip(info_part['t'], info_part['E'], info_part['S']):
                if list_contains_either(event_types, EventType.ACTIVATION):
                    #--- RELEVANT FOR 'P(T>t)'
                    ## NOTE: (2021/02/18) Fictitious ACTIVATION times (ACTIVATION_F) are NOT added to the list of activation times
                    ## because these should NOT be used to compute the absorption times that are used to
                    ## compute Pr(T>t / s=1). In fact, if they were used, the absorption times that are added
                    ## to self.sk would be too small (< EPSILON_TIME, because a fictitious activation time
                    ## is followed by a fictitious absorption time coming just EPSILON_TIME afterwards), and
                    ## they would UNDERESTIMATE Pr(T>t / s=1), thus generating an underestimated blocking probability
                    ## by Fleming-Viot's Approximation 1 approach.
                    assert not list_contains_either(event_types_prev, EventType.BLOCK), \
                            "The event before an ACTIVATION cannot be a BLOCK event, as the particle first needs to be UNBLOCKED:\n{}" \
                            .format(info_part)
                    assert not list_contains_either(event_types_prev, EventType.CENSORING), \
                            "There must be NO event after a CENSORING observation ({})".format(event_types)
                    if self.reactivate and self.burnin_cycles_absorption == 0:
                        assert not list_contains_either(event_types_prev, EventType.ABSORPTION), \
                                "There must be NO event after an absorption when no burn-in period is defined because the particle is ractivated into a NEW particle when it is absorbed (p={}, P={}, {})" \
                                .format(p, P, info_part['E'])

                    activation_times += [t]

                    # Keep track of the distribution of activation states
                    self._update_activation_distribution(state)
                elif list_contains_either(event_types, [EventType.ABSORPTION, EventType.ABSORPTION_F]) and event_types_prev != []:
                    #--- RELEVANT FOR 'P(T>t)'
                        ## NOTE: (2021/04/24) The `event_types_prev != []` is added to take care of the special first absorption event
                        ## that happens when the particle starts at an ABSORPTION state. If such condition were not present, the process
                        ## would enter this block but it would not have any previous ACTIVATION time w.r.t. which the absorption time
                        ## is to be measured.
                    assert list_contains_either(event_types_prev, [EventType.ACTIVATION, EventType.ACTIVATION_F]), \
                            "The previous event of an ABSORPTION is always an ACTIVATION (event_types_prev={}, p={}, P={}, t={}))" \
                            .format([e.name for e in event_types_prev], p, P, t)
                    if len(activation_times) > 0:
                        # Note that there may be no ACTIVATION events in the past when an ABSORPTION occurs
                        # when the particle does not start at an ACTIVATION state and it never touches
                        # an ACTIVATION state in which case there will be an ACTIVATION_F and an ABSORPTION_F
                        # fictitious events when FinalizeType.ABSORB_CENSORED is the case.
                        # In that case the process will enter this block under an ABSORPTION_F event and
                        # since no ACTIVATION_F counts as an actual activation, the activation_times list
                        # will be empty.
                        self.insert_relative_time_from_activation_to_absorption(t, activation_times, event_types)
                        # The list of activation times is reset to empty after an absorption
                        # because all past activation times have been used to populate self.sk
                        # in the above call to insert_relative_time_from_activation_to_absorption().
                        activation_times = []
                    if False:
                        # We may want to disable this assertion if they take too long
                        assert sorted(self.sk) == list(np.unique(self.sk)), \
                                "The list of survival time segments contains unique values" \
                                " after insertion of event {} for particle p={}, P={}" \
                                .format([e.name for e in event_types], p, P)
                elif list_contains_either(event_types, EventType.CENSORING):
                    #--- RELEVANT FOR 'P(T>t)'
                    # Note that if we enter here, t is the largest time observed for the particle, which is what we assert next
                    assert t == info_part['t'][-1], "P={}: The censored time (t={}) is the last observed time: {}".format(P, t, info_part['t'])

                    # Decide whether to use the last chunk of simulation time depending on how large it is
                    # which gives an indication whether the chunk is a valid chunk or is otherwise too small
                    # to include it and it rather means that the system didn't have enough time to evolve after the latest ABSORPTION,
                    # and thus its duration should NOT count for the estimation of P(T>T).
                    # Note that the inclusion of the duration of the last chunk in the estimation of P(T>t)
                    # is relevant mostly when there are very few samples (absorptions) observed during the simulation
                    # for the estimation of P(T>t) which most likely would mean that P(T>t) would be underestimated
                    # (i.e. for every t, P(T>t) is actually larger than what is estimated) if we do not include the last chunk.
                    assert len(event_types) == 1, "The only event happening when a particle is censored is CENSORING: {}".format(event_types)
                    if len(activation_times) > 0:
                        # There was at least one activation in the last trajectory chunk
                        # => Check whether we should add the time since all past activation times
                        # up to the censoring times to the estimation of P(T>t)
                        assert sorted(activation_times) == activation_times, "The activation times are sorted: {}".format(activation_times)
                        smallest_activation_time = activation_times[0]
                        average_absorption_time = np.mean(self.sk)
                        largest_absorption_time_to_potentially_insert = t - smallest_activation_time
                        if largest_absorption_time_to_potentially_insert > MULTIPLIER_FOR_LARGE_MEASUREMENT * average_absorption_time:
                            # Note that at least one absorption/kill time inserted here in self.sk will be the largest among all
                            # absorption times previously inserted in self.sk and contributing to P(T>t), and because
                            # the event_type is CENSORING, this insertion will indicate that the particle is considered
                            # to be alive up to infinity.
                            # (see function _update_counts_alive() for more details)
                            self.insert_relative_time_from_activation_to_absorption(t, activation_times, event_types)
                elif list_contains_either(event_types, [EventType.BLOCK, EventType.UNBLOCK]):
                    #--- RELEVANT FOR Phi(t / s=1)
                    # We insert the ABSOLUTE time for BLOCK and UNBLOCK events
                    # since this is used to compute the empirical distribution at the absolute time t
                    # NOTE: (2021/02/18) We do NOT consider fictitious UNBLOCK times (UNBLOCK_F)
                    # (added by finalize() at the end of the simulation to avoid assertion failures)
                    # because we do not want to distort the estimation of blocking time with a too short
                    # duration happening just because the simulation ended and the particle was blocked at that time...) 
                    # This means that if the particle was blocked at the end of the simulation,
                    # it will count as blocked up to the (relative) time since the blocking  occured
                    # --> This is precisely what we want!
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
            np.set_printoptions(precision=3, suppress=True)
            print("Relative absorption times and counts:\n{}".format(np.c_[np.array(self.sk), np.array(self.counts_alive)]))
            print("Relative blocking times:\n{}".format(np.c_[np.array(self.sbu), np.array(self.counts_blocked)]))
            np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

    def compute_counts_blocked_particles(self):
        "Computes the blocking time segments needed for the calculation of Phi(t) = P(BLOCK / T>t, s=1)"
        assert self.reactivate
        if self.LOG:
            print("Computing counts for the calculation of the survival function and conditional blocking probability...")

        if not self.is_simulation_finalized:
            raise ValueError("The simulation has not been finalized..." \
                          "\nThe computation of the counts by time segment cannot proceed." \
                          "\nRun first the finalize() method and rerun.")

        # Initialize the lists that will store information about the time values
        # at which the probability functions (survival and blocking) change their value.
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
        self.counts_blocked = [0]
        # Go over each particle
        for p, info_part in enumerate(self.info_particles):
            P = info_part['particle number']
            #print("Processing particle ID p={}, P={} out of {} particles".format(p, P, len(self.info_particles)))
            #print("Special events information:\n{}".format(info_part))
            #input("Press Enter to continue...")
            if P not in self.particles_to_simulate:
                # The current particle P has NOT been used in the FV process
                # => Do NOT consider it for the counts calculations
                continue

            # A FEW ASSERTIONS ABOUT HOW TRAJECTORIES OF THE PARTICLES --THAT ARE USED
            # FOR THE CALCULATION OF THE COUNTS-- END
            # This implies also checking that the finalization process by finalize() has being carried out correctly
            self.assertFinalizationProcessIsOk(P, p, info_part['E'])

            # List of observed activation times for the particle
            # Once the particle is absorbed, a survival time for each activation time
            # is added to the list of survival times given activation (that are used to estimate Pr(T>t / s=1))
            # The list of activation times is then reset to empty, once they are used
            # following an ABSORPTION event for the particle.
            event_types_prev = []
            for t, event_types in zip(info_part['t'], info_part['E']):
                if list_contains_either(event_types, [EventType.BLOCK, EventType.UNBLOCK]):
                    #--- RELEVANT FOR Phi(t / s=1)
                    # We insert the ABSOLUTE time for BLOCK and UNBLOCK events
                    # since this is used to compute the empirical distribution at the absolute time t
                    # NOTE: (2021/02/18) We do NOT consider fictitious UNBLOCK times (UNBLOCK_F)
                    # (added by finalize() at the end of the simulation to avoid assertion failures)
                    # because we do not want to distort the estimation of blocking time with a too short
                    # duration happening just because the simulation ended and the particle was blocked at that time...) 
                    # This means that if the particle was blocked at the end of the simulation,
                    # it will count as blocked up to the (relative) time since the blocking  occured
                    # --> This is precisely what we want!
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
            np.set_printoptions(precision=3, suppress=True)
            print("Relative absorption times and counts:\n{}".format(np.c_[np.array(self.sk), np.array(self.counts_alive)]))
            print("Relative blocking times:\n{}".format(np.c_[np.array(self.sbu), np.array(self.counts_blocked)]))
            np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

    def insert_relative_time_from_activation_to_absorption(self, t :float, activation_times :list, event_types :list):
        """
        Inserts new relative time segments in the list of killing/absorption times,
        based on the given absolute time `t` and on the given list of activation times:
        the inserted relative times are computed as the absolute time measured
        relative to EACH of the given activation times, as long as they are smaller than t.
        
        A warning is issued when a negative relative time would have been inserted.

        Arguments:
        activation_times: list
            List of activation times for which a relative time-to-absorption is computed and inserted in the
            list of (relative) killing/absorption times.

        event_types: list
            See description in _update_counts_alive() method.
        """
        #print("")
        for a in activation_times:
            s = t - a
            if s < 0:
                warnings.warn("The activation event from which the absolute time t={:.3f} should be measured is NOT in the PAST but in the FUTURE! {:.3f}".format(t, a))
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
            List from where the type of event is extracted, whether an ABSORPTION, ABSORPTION_F, CENSORING.
            Only ONE of these events can appear in the list.

            This affects how the counts_alive list is updated:
            - in the ABSORPTION case, all the counts to the LEFT of the newly inserted element
            to the list are increased by +1.
            - in the CENSORING case ALSO the count of the newly inserted element is increased by +1
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

        # DM-2021/05/13: The following strategy was commented out because for now we are considering
        # that in the CENSORING case the particle is absorbed at the censoring time.
        # This came up through the newly implemented logic today that allows a POTENTIAL contribution from the
        # last observed trajectory chunk if this chunk lasts for too long (compared to the previous chunks)
        # For more details see the compute_counts() method.
        # Besides, if we let the below code run, the assertion that the last value
        # in the counts_alive list is 0 fails.
        # NOTE: (2021/05/13) I don't think this logic will EVER be changed because we probably will never
        # be interested in estimating the time left for censored particles, as their contribution
        # is most likely negligible when we are simulating a large amount of time!
        #if list_contains_either(event_types, EventType.CENSORING):
        #    # Increase by 1 the count corresponding to the insert index
        #    # (i.e. increase the count corresponding to the time segment that STARTS at the new inserted
        #    # time in the self.sk list of time segments, which means that the particle
        #    # is considered alive at least up to the end of such time segment --which could be
        #    # INFINITE if this new element is inserted at the end of the list; in such case
        #    # we are assuming that the particle is still alive up to Infinity.)  
        #    self.counts_alive[idx_to_insert] += 1

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
            if not 0 <= self.counts_blocked[idx] <= len(self.particles_to_simulate):
                self.plot_trajectories_by_particle()
            assert 0 <= self.counts_blocked[idx] <= len(self.particles_to_simulate), \
                    "Insertion index: {}\n".format(idx_to_insert_or_update) + \
                    "The number of blocked particles in time segment with index idx={} out of {} time segments " \
                    .format(idx, len(self.counts_blocked)) + \
                    "found so far is between 0 and N0={} ({}) \n({})" \
                    .format(len(self.particles_to_simulate), self.counts_blocked[idx], self.counts_blocked)
    #--------------------------- Functions to wrap up the simulation --------------------------


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

    def compute_probability_from_counts(self, counts, ntotal):
        """
        Computes an empirical probability of an event by time using an array of event counts
        at each time step the count changes.
        """
        if len(counts) <= 1:
            return [0.0]
        return [n / ntotal for n in counts]

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
        Computes Pr(K) * (Expected absorption cycle time) where
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
        - 'P(T>t)': the survival probability at each t.
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
        return pd.DataFrame.from_items([('t', self.sk), ('P(T>t)', proba_surv_by_t)])

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
        proba_block_by_t = self.compute_probability_from_counts(self.counts_blocked, len(self.particles_to_simulate))
        return pd.DataFrame.from_items([('t', self.sbu), ('Phi(t,K / s=1)', proba_block_by_t)])

    def estimate_proba_survival_and_blocking_conditional(self):
        """
        Computes the following quantities which are returned in a data frame with the following columns:
        - 't': time at which a change in any quantity happens
        Quantities used in Approximation 1 of the blocking probability estimate:
        - 'P(T>t)': survival probability given activation start
        - 'P(BLOCK / T>t)': blocking probability given alive and activation start
        Quantities used in Approximation 2 of the blocking probability estimate:
        - 'Killing Rate': killing rate a.k.a. gamma parameter
        - 'Blocking Time Estimate' = Pr(BLOCK) * (Expected absorption cycle time)
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
            np.set_printoptions(precision=3, suppress=True)
            print("SURVIVAL: times and counts_alive: \n{}".format(np.c_[self.sk, self.counts_alive, [c/self.counts_alive[0] for c in self.counts_alive] ]))
            print("BLOCKING: times and counts_blocked:\n{}".format(np.c_[self.sbu, self.counts_blocked, [c/len(self.particles_to_simulate) for c in self.counts_blocked]]))
            np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

        # Since we must compute this probability CONDITIONED to the event T > t,
        # we first need to merge the measurements of the block times and of the survival times
        # into one single set of measurement time points.
        if self.proba_survival_given_activation is not None:
            # The estimated survival probability given activation was given by the user
            # => Merge the time values at which the survival probability changes with the time values
            # at which the blocking and unblocking events occur.
            self.t, self.proba_surv_by_t, counts_blocked = merge_values_in_time(
                                                                        list(self.proba_survival_given_activation['t']),
                                                                        list(self.proba_survival_given_activation['P(T>t)']),
                                                                        self.sbu, self.counts_blocked, unique=False)
        else:
            self.t, counts_alive, counts_blocked = merge_values_in_time(self.sk, self.counts_alive, self.sbu, self.counts_blocked, unique=False)
            self.proba_surv_by_t = self.compute_survival_probability_from_counts(self.t, counts_alive)

        self.proba_block_by_t = self.compute_probability_from_counts(counts_blocked, len(self.particles_to_simulate))
        # NOTE: (2021/04) We are no longer computing the FV estimator based on Approximation 2 because it is too unstable.
        self.gamma = self.compute_killing_rate()
        self.blocking_time_estimate = self.compute_blocking_time_estimate()
            ## Note: blocking time estimate = Pr(K) * Expected absorption cycle time
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
                                            ('P(T>t)', self.proba_surv_by_t),
                                            ('P(BLOCK / T>t)', self.proba_block_by_t),
                                            ('Killing Rate', self.gamma),
                                            ('Blocking Time Estimate', self.blocking_time_estimate)]))

        return pd.DataFrame.from_items([('t', self.t),
                                        ('P(T>t)', self.proba_surv_by_t),
                                        ('P(BLOCK / T>t)', self.proba_block_by_t),
                                        ('Killing Rate', self.gamma),
                                        ('Blocking Time Estimate', self.blocking_time_estimate)])

    def estimate_expected_absorption_time_from_killings(self):
        """
        Estimates the expected absorption cycle time E(T) from the collection of killing times stored in self.ktimes

        This function is expected to be called when estimating E(T) from a simulation
        that is run separate from the FV process, just to estimate E(T). 
        """

        # The following disregards the particles that were still alive when the simulation stopped
        # These measurements are censored, so we could improve the estimate of the expected value
        # by using the Kaplan-Meier estimator of the survival curve, that takes censoring into account.
        # However, note that the KM estimator (https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator)
        # assumes that the censoring times are deterministic (which may not be the case).
        # Also, from the above Wiki entry, it seems it considers only integer time values where S(t) is updated...??
        # My intuition is that we could divide the censored time by (1 - S(t)), where S(t) is the survival function
        # at the time of censoring t (which is available as long as S(t) was estimated, i.e. if t < at least one
        # other actual observed (non-censored) absorption time.
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
        # (if needed to improve the estimation of the expected absorption cycle time, but I don't think it will be necessary
        # as the censoring times are a VERY SMALL percentage if the simulation runs for a long time, so the effect
        # of their values is negligible!!)
        # 
        # Ref: (from publications available via inp-toulouse.fr)
        # Three methods for the estimation of the expected absorption cycle time (mean lifetime) with censoring observations:
        # saved to RL-002-QueueBlocking/200503-JournalStatisticalMethodology-Datta-MeanLifetimeEstimationWithCensoredData.pdf
        # Note: (2022/06/21) I deleted the URL showing the article's source because:
        # - it didn't work any more
        # - it had a token which was flagged as a Generic High Entropy Secret by GitGuardian, after I made the GitHub repository public
        # In any case, the URL (without the offending token) was the following (which doesn't work anyway):
        # https://pdf-sciencedirectassets-com.gorgone.univ-toulouse.fr/273257/1-s2.0-S1572312705X00041/1-s2.0-S1572312704000309/main.pdf

        #if self.finalize_info['type'] == FinalizeType.ESTIMATE_CENSORED:
        #    # Add an estimation of the absorption cycle time for the censored (i.e. still active) particles
        #    for P in self.active_particles:
        #        absorption_time = self.particles[P].getTimesLastEvents() - self.times0[P]
        #        ### TODO: (2020/06/13) Write the get_survival_function() method that returns the estimate of S(t)
        #        self.ktimes_sum[P] += absorption_time # / (1 - self.get_survival_function(absorption_time))
        #        self.ktimes_n[P] += 1
        #else:

        assert not self.reactivate

        # Since no reactivation is done during simulation,
        # we can estimate E(T) from ALL the observed absorptions throughout the simulation time,
        # (as opposed to only ONE observed absorption --as is the case in the ELSE block)
        # because every return to the absorption set can be considered being observed after
        # an absorption-absorption CYCLE.
        if not self.finalize_info['type'] in [FinalizeType.NONE, FinalizeType.ESTIMATE_CENSORED]:
            assert sum(self.is_particle_active) == 0, "The number of active particles is 0 ({})".format(sum(self.is_particle_active))

        total_absorption_time = np.sum(self.ktimes_sum)     # This is a sum over particles
        total_absorption_n = np.sum(self.ktimes_n)          # This is a sum over particles

        if False:
            print("Estimating expected absorption cycle time...")
            print("Total absorption cycle time over all particles (N={}): {:.1f}".format(self.N, total_absorption_time))
            print("Number of observed absorption cycle times over all N={} particles: {}".format(self.N, total_absorption_n))

        if total_absorption_n == 0:
            print("WARNING (estimation of expected absorption cycle time): No particle has been absorbed.\n" +
                  "The estimated expected absorption cycle time is estimated as the average of the simulated times over all particles.")
            self.expected_absorption_time = np.mean([self.particles[P].getMostRecentEventTime() for P in range(self.N)])
        else:
            if total_absorption_n < 0.5*self.N:
                print("WARNING (estimation of expected absorption cycle time): " \
                      "The number of observed absorption cycle times is smaller than half the number of particles (sum of {} < number of particles ({})).\n" \
                      .format(self.ktimes_n, self.N) +
                      "The estimated expected absorption cycle time may be unreliable.")
            self.expected_absorption_time = total_absorption_time / total_absorption_n

        return self.expected_absorption_time, total_absorption_time, total_absorption_n

    def estimate_expected_absorption_time_from_burnin(self):
        """
        Estimates the expected absorption cycle time E(T) from the burn-in period.
        
        E(T) is estimated from the first absorption after the burn-in period in each particle has been completed.
        If no absorption has occurred, it means not enough time elapsed for the absorption to take place.

        In such cases, the original NaN value stored as the absorption cycle time for each particle
        undergoes two different treatments:
        - If we accept an error in the estimated E(T) that goes more towards BIAS rather than variance,
        the absorption cycle time of non-absorbed particles is (under-)estimated by the time elapsed
        since the latest absorption (if any) or the time elapsed since t=0 otherwise.
        - If we accept an error in the estimated E(T) that goes more towards VARIANCE rather than bias,
        the absorption cycle time for non-absorbed particles are NOT estimated, and the estimation of E(T)
        is simply based on the set of absorbed particles (which is <= the number of particles N).

        Which situation is used in the code is currently hard-coded in this function via the
        ErrOnTheSideOfIncreased Enum defined here.

        Return: tuple
        The tuple contains the following elements:
        - the estimated expected absorption cycle time, E(T)
        - the total absorption cycle time observed (or estimated if not absorbed) over all particles
        - the number of particles with a FULLY observed absorption cycle
        """
        assert self.reactivate

        class ErrOnTheSideOfIncreased(Enum):
            BIAS = 1
            VARIANCE = 2

        # Define how we want to err in the estimation of E(T) when not all particles have been absorbed after the burn-in period        
        what_error_to_accept = ErrOnTheSideOfIncreased.VARIANCE

        n_absorption_cycle_times_fully_measured = 0
        for P in range(self.N):
            if np.isnan(self.absorption_cycle_times[P]):
                if what_error_to_accept == ErrOnTheSideOfIncreased.VARIANCE:
                    # Replace the missing absorption cycle time with an underestimate:
                    # the time elapsed since the latest absorption
                    self.absorption_cycle_times[P] = self.get_max_simulation_time_of_burnin_period() - self.times0[P]
            else:
                n_absorption_cycle_times_fully_measured += 1
        n_absorption_cycle_times = np.sum( ~np.isnan(self.absorption_cycle_times) )
        if what_error_to_accept == ErrOnTheSideOfIncreased.VARIANCE:
            assert n_absorption_cycle_times == self.N, \
                "The number of absorption cycle times used in the estimation of E(T) ({}) coincides with the number of particles N={}" \
                .format(n_absorption_cycle_times, self.N)
        else:
            assert n_absorption_cycle_times == n_absorption_cycle_times_fully_measured, \
                "The number of absorption cycle times used in the estimation of E(T) ({}) coincides with the number of fully observed absorption cycles ({})" \
                .format(n_absorption_cycle_times, n_absorption_cycle_times_fully_measured)
        if n_absorption_cycle_times_fully_measured < 0.5*self.N:
            print("WARNING (estimation of expected absorption cycle time): " \
                  "The number of observed absorption cycle times is smaller than half the number of particles (sum of {} < number of particles ({}))." \
                  .format(n_absorption_cycle_times_fully_measured, self.N) + \
                  "\nThe estimated expected absorption cycle time may be " + \
                  (what_error_to_accept == ErrOnTheSideOfIncreased.BIAS and "STRONGLY" or what_error_to_accept == ErrOnTheSideOfIncreased.VARIANCE and "") + "underestimated.")
        total_absorption_time = np.nansum(self.absorption_cycle_times)
        total_absorption_n = n_absorption_cycle_times

        if self.LOG or DEBUG_TRAJECTORIES: #True
            print("\t--> E(T) estimation: (n, min, mean, max) = ({} out of N={}, {:.1f}, {:.1f}, {:.1f})" \
                  .format(total_absorption_n, self.N, np.nanmin(self.absorption_cycle_times), np.nanmean(self.absorption_cycle_times), np.nanmax(self.absorption_cycle_times)))

        if total_absorption_n == 0:
            # This happens when no absorption time is observed at all and when we have decided
            # to err more on the VARIANCE side of the estimation of E(T) rather than on the BIAS side
            # --i.e. when we have decided to NOT underestimate E(T) with the time elapsed to the end of simulation. 
            self.expected_absorption_time = np.nan
        else:
            self.expected_absorption_time = total_absorption_time / total_absorption_n

        # DM-2021/09/14: I've decided to store in the object the number of particles
        # used in the denominator of the E(T) estimator above the number of particles
        # on which a FULLY absorption cycle was observed.
        # The goal is to know how reliable is the estimated E(T) by storing this number
        # in the table containing the simulation results.
        self.n_expected_absorption_time = n_absorption_cycle_times_fully_measured

        return self.expected_absorption_time, total_absorption_time, n_absorption_cycle_times_fully_measured

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

    def estimate_proba_blocking_via_integral(self, expected_absorption_time):
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

        # DM-2021/09/13: We now allow E(T) to be 0 or NaN, which happens when NO valid absorption cycles
        # were observed during the burn-in period, and we are allowing the estimate to err on higher VARIANCE
        # as opposed to higher BIAS --i.e. we are NOT underestimating E(T) with the time until the end of simulation
        # (see most likely estimate_expected_absorption_time_from_burnin() function for more details)
        #assert expected_absorption_time > 0, "The expected absorption cycle time is positive ({})".format(expected_absorption_time)
        assert expected_absorption_time is not None, "The expected absorption cycle time is not None"
        proba_blocking_integral = integral / expected_absorption_time

        return proba_blocking_integral, integral

    def estimate_proba_blocking_via_laplacian(self, expected_absorption_time):
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

        proba_blocking_laplacian = self.blocking_time_estimate[idx_tmax] / expected_absorption_time
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
        assert self.reactivate, "Under the Fleming-Viot estimation of the blocking probability, reactivate=True"
        if not self.is_simulation_finalized:
            raise ValueError("The simulation has not been finalized..." \
                          "\nThe estimation of the blocking probability cannot proceed." \
                          "\nRun first the finalize() method and rerun.")

        #-- Compute the building blocks of the blocking probability estimate which are stored in the object
        # This includes:
        # - self.t: time at which an event occurs
        # - self.proba_surv_by_t: survival probability given activation start
        # - self.proba_block_by_t: blocking probability given alive and activation start
        # - self.gamma: killing rate, eigenvalue of the laplacian
        # - self.blocking_time_estimate: estimated blocking time = Pr(block) * (Expected absorption cycle time)
        df = self.estimate_proba_survival_and_blocking_conditional()
        if False:
            # Analyze the killing rate (gamma) with time (to see how noisy it is)
            plt.plot(df['t'], df['Killing Rate'], 'b.-')
            ax = plt.gca()
            ax.set_ylim((0,2))
            ax.set_xlabel("t")
            ax.set_ylabel("Killing Rate (gamma)")
            plt.title("K={}, particles={}, max time reached={:.1f}".format(self.queue.getCapacity(), self.N, np.max(self.times_latest_known_state)))
            plt.show()

        #-- Expected absorption cycle time was either given by the user or is estimated now provided the bunin_cycles_absorption parameter is not None
        if self.is_expected_absorption_time_estimated_from_burnin:
            self.estimate_expected_absorption_time_from_burnin()
            ET = self.expected_absorption_time
        else:
            ET = self.mean_lifetime
        assert ET is not None

        #-- Blocking probability estimate via Approximation 1: estimate the integral
        self.proba_blocking, self.integral = self.estimate_proba_blocking_via_integral(ET)

        #-- Blocking probability estimate via Approximation 2: estimate the Laplacian eigenvalue (gamma) and eigenvector (h(1))
        #proba_blocking_laplacian, gamma = None, None #self.estimate_proba_blocking_via_laplacian(self.mean_lifetime)

    def estimate_proba_blocking_mc(self):
        """
        Estimates the blocking probability using Monte Carlo
        
        The blocking probability is estimated as the ratio between the total blocking time and
        the total return time to the start state over all particles.
        
        Note that this methodology of summing ALL times over ALL particles, avoids having to weight
        the mean return time estimation of each particle, as we don't need to weight it with the
        number of samples the return time is based on.

        Return: tuple
        The tuple contains the following elements:
        - the estimated blocking probability as the proportion of blocking time over return time
        - the total blocking time over all particles
        - the total return time to the start position over all particles
        - the number of times the return time was measured
        """
        assert len(self.particles_to_simulate) == 1, "The number of simulated particles is 1 ({})".format(len(self.particles_to_simulate))

        total_blocking_time = np.sum(self.btimes_sum)
        _, total_return_time, total_return_n = self.estimate_expected_return_time()
        if total_return_time == 0:
            print("WARNING (estimation of blocking probability by MC): No particle has been absorbed.\n" +
                  "The total return time to the set of absorption states is estimated as the total simulated time over all particles.")
            total_return_time = np.sum([self.particles[P].getMostRecentEventTime() for P in self.particles_to_simulate])
        blocking_time_rate = total_blocking_time / total_return_time
        return blocking_time_rate, total_blocking_time, total_return_time, total_return_n
    #----------------------------- Functions to analyze the simulation ------------------------


    #-------------------------------------- Getters -------------------------------------------
    def getMaxSimulationTime(self):
        return self.maxtime

    def getMaxNumberOfEvents(self):
        return self.finalize_info['maxevents']

    def getFinalizeType(self):
        return self.finalize_info['type']

    def getFinalizeCondition(self):
        return self.finalize_info['condition']

    def getStartPosition(self, P):
        return self.all_positions_buffer[P][0]

    def getNumberOfEvents(self):
        return self.nevents

    def getLoads(self):
        return self.rhos

    def get_number_events_from_latest_reset(self):
        """
        Returns the number of events taking place in the simulation after the latest reset of the system.
        For instance, when reactivate=True, all particles are reset after the burn-in period has been completed
        and we can start with the FV process, so this function will only include the events ONCE the FV process started.

        Only one out of two consecutive events recorded with a time difference of less than
        EPSILON_TIME are counted. In fact, the second event of such pair is either:
        - a particle that is reactivated --> in this case we should only count the absorption event
        - a fictitious event added when finalizing the simulation (by finalize())

        This can be used to evaluate how much information we extract from the simulation to compute our estimation.
        """
        nevents = 0
        for P in range(len(self.all_times_buffer)):
            time_diff = np.diff(self.all_times_buffer[P])
            nevents += np.sum( time_diff > self.EPSILON_TIME )
        return nevents

    def get_number_observations_proba_survival(self):
        return self.counts_alive[0]

    def get_max_simulation_time_of_burnin_period(self):
        return self.maxtime_burnin

    def get_simulation_time(self, which="last"):
        """
        Returns the total simulation time used by all particles.

        The concept of simulation time depends on parameter `which`:
        - when "max" => the maximum time among the latest known state times by particle.
        - when "last" (or o.w.) => sum of latest known state times by particle.

        IMPORTANT: Note that if the finalize() process has already taken place
        and the finalize type is REMOVE_CENSORED, the latest known state time of a particle
        will be the time AFTER the removal of any subtrajectory, i.e. possibly smaller
        than the actual simulation time for that particle.
        """
        if which == "max":
            return np.max( self.times_latest_known_state )
        else:
            return np.sum( self.times_latest_known_state )

    def get_start_time_fv(self, P):
        return self.times_start_fv[P]

    def get_min_start_time_fv(self):
        return np.nanmin(self.times_start_fv)

    def get_mean_start_time_fv(self):
        return np.nanmean(self.times_start_fv)

    def get_max_start_time_fv(self):
        return np.nanmax(self.times_start_fv)

    def get_n_start_time_fv(self):
        """
        Returns the number of non-NaN times at which a potential FV process for a particle starts.
        
        We say "*potential* FV process" because the fact that the time at which the FV process would
        start is measured is NOT enough for the FV process to start on the particle.
        In fact, we ALSO need the particle to be ABSORBED once before allowing the particle to take part
        in the FV process because the survival time contributes to the definition of the maximum simulation
        time for the FV process.

        Note that, even if the particle is not absorbed after activation (because the end of the burn-in period
        occurs), we could still use the time to the end of the burn-in period as an underestimation of the actual
        survival time and use that as contribution to the definition of the max simulation time for the FV process.
        But impilementing this is currently (14-Sep-2021) not so easy to implement and keep a record of.  
        """
        return np.sum(~np.isnan(self.times_start_fv))

    def get_max_observed_survival_time(self):
        "Returns the maximum observed survival time given an active state"
        # This works fine as long as the self.sk list is sorted, which is guaranteed by the
        # self.insert_relative_time_from_activation_to_absorption() method
        # used to add values to the self.sk list.
        return self.sk[-1]

    def get_position(self, P, t):
        """
        Returns the position of the given particle at the given time, based on the trajectories information.
        
        Arguments:
        P: non-negative int
            Particle number (index of the list of particle queues) whose position is of interest.

        t: float
            Time at which the position of the given particle should be retrieved.

        Return: numpy array
        The position is given by the size of each server that is serving the queue represented by the particle
        at the given time. 
        """
        assert self.isValidParticle(P), "The particle number is valid (P={}, N={})".format(P, self.N)
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
            assert idx > 0, "P={}, server={}: The bisect index where the queried time (t={:.3f}) would be inserted in the server_times array\n({})\nis positive ({})".format(P, server, t, server_times, idx)
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
            return [P for P in self.particles_to_simulate if self.is_particle_active[P]]
        elif self.finalize_info['condition'] == FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY:
            # A particle that has not JUST become absorbed is considered censored
            return [P for P in self.particles_to_simulate if not self._has_particle_become_absorbed(P)]
        elif self.finalize_info['condition'] == FinalizeCondition.NOT_START_POSITION:
            return [P for P in self.particles_to_simulate if self.particles[P].getBufferSize() != self.getStartPosition(P)]

    def get_number_active_particles(self):
        return sum(self.is_particle_active)

    def get_activation_states_distribution(self):
        "Returns a duple containing the list of possible activation states, and their observed frequency of occurrence during simulation"
        return self.states_activation, self.dist_activation_states

    def get_absorption_states_distribution(self):
        "Returns a duple containing the list of possible absorption states, and their observed frequency of occurrence during simulation"
        return self.states_absorption, self.dist_absorption_states

    def get_survival_time_segments(self):
        return self.sk

    def get_blocking_time_segments(self):
        return self.sbu

    def get_all_total_blocking_time(self):
        "Returns the total blocking time for all particles"
        blocking_time = 0.0
        for P in self.particles_to_simulate:
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
        for P in self.particles_to_simulate:
            blocking_times += [ pd.DataFrame.from_items([('Block Time', []), ('Unblock Time', [])]) ]
        for p, info_part in enumerate(self.info_particles):
            P = info_part['particle number']
            block_times_p = [t      for idx, t in enumerate(info_part['t'])
                                    if list_contains_either(info_part['E'][idx], EventType.BLOCK)]
            unblock_times_p = [t    for idx, t in enumerate(info_part['t'])
                                    if list_contains_either(info_part['E'][idx], EventType.UNBLOCK)]

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

    def get_all_total_absorption_time(self):
        "Returns the total time of absorption cycles for all particles"
        absorption_time = 0.0
        for P in self.particles_to_simulate:
            absorption_time += self.get_total_absorption_time(P)

        assert np.allclose(absorption_time, np.sum(self.ktimes_sum)), \
                "The sum of all the absorption cycle times just computed ({:.3f}) equals" \
                " the sum of the absorption cycle times per particle computed during the simulation ({:.3f})" \
                .format(absorption_time, np.sum(self.ktimes_sum))

        return absorption_time

    def get_total_absorption_time(self, P):
        "Returns the total time of absorption cycles for a particle (i.e. from an absorption to the next absorption)"
        all_absorption_periods = self.get_all_absorption_periods()
        return np.sum(all_absorption_periods[P]['Absorption Cycle'])

    def get_all_absorption_periods(self):
        """
        Returns the absorption cycles for each particle number P
        (these are the periods in which the particle goes from an absorption state back to an absorption state).
        Censoring times are included as valid end times for the absorption cycles.
        
        Return: list of pandas DataFrame
        The list has one element per particle P (0 <= P <= N-1) and the DataFrame has two columns:
        - 'Absorption Cycle End': end time of absorption cycle
        - 'Absorption Cycle': duration of absorption cycle
        """

        # Absorption cycles for each particle number
        absorption_and_censoring_times = []
        absorption_times = []
        # NOTE: It's important to create SEPARATE lists and data frames for each particle
        # and NOT use [ pd.DataFrame ] * N because the latter will generate just ONE data frame N times!! 
        for P in self.particles_to_simulate:
            absorption_and_censoring_times += [ [0.0] ]
            absorption_times += [ pd.DataFrame.from_items([('Absorption Cycle End', []), ('Absorption Cycle', [])]) ]
        for p, info_part in enumerate(self.info_particles):
            P = info_part['particle number']
            #with printoptions(precision=3):
            #    print("\np={}, P={}:\n{}".format(p, P, np.c_[ np.array(info_part['t']), np.array(info_part['E']) ]))
            absorption_and_censoring_times_p = [t   for idx, t in enumerate(info_part['t'])
                                                    if list_contains_either(info_part['E'][idx], [EventType.ABSORPTION, EventType.CENSORING])]
            # The difference with the following list of times and the previous list of times
            # is that the following list has an initial 0.0 which is needed to concatenate with the absorption times below
            # to generate the output object.
            absorption_and_censoring_times[P] += absorption_and_censoring_times_p
            #print("absorption times: {}".format(absorption_and_censoring_times[P]))

        for P in self.particles_to_simulate:
            absorption_times_P = np.diff(absorption_and_censoring_times[P])
            #print("P={}: absorption times: {}".format(P, absorption_times_P))
            if len(absorption_times_P) > 0:
                assert np.all(absorption_times_P > 0), \
                        "The absorption cycle times for particle P={} are all positive ({})" \
                        .format(P, absorption_times_P)

            absorption_times[P] = pd.concat( [ absorption_times[P],
                                             pd.DataFrame({'Absorption Cycle End': absorption_and_censoring_times[P][1:],
                                                           'Absorption Cycle': absorption_times_P}) ], axis=0 )

        return absorption_times

    def get_absorption_times(self):
        """
        Returns the absorption cycle times for all particles
        
        Return: tuple
        The tuple contains information about the absorption cycles (i.e. the contiguous time
        during which a particle goes from an absorption state back to an absorption state for the first time)
        - a list with the number of absorption cycles observed for each particles
        - a list with the total time span of the absorption cycles for each particle
        """
        return self.ktimes_n, self.ktimes_sum

    def get_expected_absorption_time(self):
        return self.expected_absorption_time

    def get_time_latest_known_state(self):
        "Returns the time of the latest known state of the system (i.e. over all particles)"
        return np.min(self.times_latest_known_state)

    def get_time_latest_known_state_for_particle(self, P):
        return self.times_latest_known_state[P]

    def get_counts_particles_alive_by_elapsed_time(self):
        return self.counts_alive

    def get_counts_particles_blocked_by_elapsed_time(self):
        return self.counts_blocked
    #-------------------------------------- Getters -------------------------------------------


    #-------------------------------------- Setters -------------------------------------------
    def set_simulation_time(self, maxtime):
        self.maxtime = maxtime
    #-------------------------------------- Setters -------------------------------------------


    #------------------------------------- Plotting -------------------------------------------
    def plot_trajectories_by_particle(self):
        "Plots the trajectories of the particles"
        K = self.queue.getCapacity()
        if K == np.Inf:
            # Set a large enough value for K when the queue's capacity is infinite
            K = self.buffer_size_activation * 2
        colormap = cm.get_cmap("jet")
        # Reference lines showing for each particle:
        # - state=0
        # - state=activation
        # - state=block
        reflines_zero = range(0, (K+1)*self.N, K+1)
        reflines_absorption = range(self.buffer_size_activation-1, self.buffer_size_activation-1+(K+1)*self.N, K+1)
        reflines_block = range(K, K+(K+1)*self.N, K+1)
        particle_numbers = self.particles_to_simulate
        if False:
            for p in particle_numbers:
                print("\nParticle number {}:".format(p))
                print("Buffer Times and Sizes:")
                print(self.all_times_buffer[p])
                print(self.all_positions_buffer[p])
        plt.figure()
        xaxis_max = self.maxtime is not None and self.maxtime or np.max(self.times_latest_known_state)
        ax = plt.gca()
        ax.set_xlabel("t")
        ax.set_ylabel("particle (buffer size)")
        ax.set_yticks(reflines_zero)
        ax.yaxis.set_ticklabels(particle_numbers)
        #ax.xaxis.set_ticks(np.arange(0, round(xaxis_max)+1) )
        ax.set_ylim((0, (K+1)*self.N))
        ax.hlines(reflines_block, 0, xaxis_max, color='gray', linestyles='dashed')
        ax.hlines(reflines_absorption, 0, xaxis_max, color='red', linestyles='dashed')
        ax.hlines(reflines_zero, 0, xaxis_max, color='gray')
        ax.vlines(xaxis_max, 0, (K+1)*self.N, color='red', linestyles='dashed')
        for p in particle_numbers:
            color = colormap( (p+1) / self.N )
            # DM-2022/04/04: Correction done for the central color so that the trajectory is more clearly seen for the presentation at STORE (06-Apr-2022)
            #if p + 1 == 3 and self.N == 5:
            #    color = colormap( 0.5 )
            # Non-overlapping step plots at vertical positions (K+1)*p
            plt.step(self.all_times_buffer[p], [(K+1)*p + pos for pos in self.all_positions_buffer[p]], '-', #'x-',
                     where='post', color=color, markersize=3)
        plt.title("K={}, rates(B)={}, rates(D)={}, activation={}, reactivate={}, finalize={}, N={}, N0={}, maxtime={}, max#events={}, seed={}" \
                  .format(self.queue.getCapacity(),
                      self.queue.getBirthRates(),
                      self.queue.getDeathRates(),
                      self.buffer_size_activation,
                      self.reactivate, self.finalize_info['type'].name[0:3], self.N, len(self.particles_to_simulate), self.maxtime, self.getMaxNumberOfEvents(), self.seed
                      ))
        ax.title.set_fontsize(9)
        plt.show()

    def plot_trajectories_by_server(self, P):
        "Plots the trajectories of the particles"
        K = self.queue.getCapacity()
        if K == np.Inf:
            # Set a large enough value for K (for plotting) when the queue's capacity is infinite
            K = self.buffer_size_activation * 2
        colormap = cm.get_cmap("jet")
        # Reference lines showing for each particle:
        # - state=0
        # - state=activation
        # - state=block
        reflines_zero = range(0, (K+1)*(self.nservers+1), K+1)       # We add a server (#self.nservers+1) because the last "server" will show the system's buffer size
        reflines_absorption = range(self.buffer_size_activation-1, (self.buffer_size_activation-1+(K+1))*(self.nservers+1), K+1)       # We add a server (#self.nservers+1) because the last "server" will show the system's buffer size
        reflines_block = range(K, (K+(K+1))*(self.nservers+1), K+1)
        servers = range(self.nservers)
        if False:
            print("\nP={}:".format(P))
            print("Times and Positions by Server:")
            print(self.all_times_by_server[P])
            print(self.all_positions_by_server[P])
        plt.figure()
        xaxis_max = self.maxtime is not None and self.maxtime or np.max(self.times_latest_known_state)
        ax = plt.gca()
        ax.set_xlabel("t")
        ax.set_ylabel("server (queue size)")
        ax.set_yticks(reflines_zero)
        ax.yaxis.set_ticklabels(servers)
        #ax.xaxis.set_ticks(np.arange(0, round(xaxis_max)+1) )
        ax.set_ylim((0, (K+1)*(self.nservers+1)))
        ax.hlines(reflines_block, 0, xaxis_max, color='gray', linestyles='dashed')
        ax.hlines(reflines_absorption, 0, xaxis_max, color='red', linestyles='dashed')
        ax.hlines(reflines_zero, 0, xaxis_max, color='gray')
        ax.vlines(xaxis_max, 0, (K+1)*(self.nservers+1), color='red', linestyles='dashed')
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

        plt.title("Particle {}: K={}, rates(B)={}, rates(D)={}, activation={}, reactivate={}, finalize={}, #servers={}, maxtime={:.1f}, max#events={}, seed={}" \
                  .format(P,
                    self.queue.getCapacity(),
                    self.queue.getBirthRates(),
                    self.queue.getDeathRates(),
                    self.buffer_size_activation,
                    self.reactivate, self.finalize_info['type'].name[0:3], self.nservers, xaxis_max, self.getMaxNumberOfEvents(), self.seed
                    ))
        ax.title.set_fontsize(9)
        plt.show()
    #------------------------------------- Plotting -------------------------------------------


    #---------------------------------- Helper functions --------------------------------------
    def setup(self):
        "Returns the string containing the simulation setup"
        # Note: Cannot use {:.3f} as format for the mean life time and for maxtime below
        # because its value may be None and we get an error of unsupported format.
        params_str = "***********************" \
                    "\nK = {}" \
                    "\njob arriving rates = {}" \
                    "\njob service rates = {}" \
                    "\n# particles = {}" \
                    "\n# servers = {}" \
                    "\n# absorption burn-in cycles = {}" \
                    "\nburn-in cycles must be completed for all particles before FV starts? {}" \
                    "\nactivation buffer size = {}" \
                    "\nmean_lifetime = {}" \
                    "\nreactivate = {}" \
                    "\nfinalize_type = {}" \
                    "\nnmeantimes = {:.1f} (maxtime = {})" \
                    "\nmax #events = {}" \
                    "\nseed = {}" \
                    "\n***********************" \
                    .format(self.queue.getCapacity(),
                            self.job_class_rates, self.queue.getDeathRates(),
                            self.N, self.nservers,
                            self.burnin_cycles_absorption, self.burnin_cycles_complete_all_particles,
                            self.buffer_size_activation, self.mean_lifetime,
                            self.reactivate, self.finalize_info['type'].name,
                            self.nmeantimes, self.maxtime, self.getMaxNumberOfEvents(), self.seed)
        return params_str

    def isValidParticle(self, P):
        "Returns whether the given particle is a valid particle number indexing the list of particle queues"
        return (isinstance(P, int) or isinstance(P, np.int32) or isinstance(P, np.int64)) and P in self.particles_to_simulate

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
        for the steady state of the simulation (i.e. NOT for self.iterations[P] = 0,
        which is also the case when the particle has just been reactivated).
        
        This function essentially checks that the time of the most recent event applied to particle P
        coincides with the time stored in the times_next_events2 2D array (of size #servers x 2)
        for P at the server in which the event took place.
        The times_next_events2 array, although called `times_NEXT_events2`,
        in the context when this assertion is called, actually contains the #servers*2 time values
        (i.e. the birth and death values for each server) among which the time of the most recent
        event has been chosen from in _get_next_event().
        
        So, this function asserts something about the PAST of the process, and it is called just BEFORE
        generating the next event taking place at particle P (in the spirit of checking that so far
        things were done correctly).  
        """
        _, server_last_updated, type_of_last_event, time_of_last_event = self.particles[P].getMostRecentEventInfo()
        if self.iterations[P] > 0 and \
           not np.isnan(self.times_next_events2[P][server_last_updated][type_of_last_event.value]):
            # The following assertion is only valid when the most recent event is NOT the first event
            # of the server that is going to be updated now (in which case the current value
            # of the next event time for the type of the most recent event is NaN --which justifies the IF condition above)
            assert np.abs(time_of_last_event + self.particles[P].getOrigin() - self.times_next_events2[P][server_last_updated][type_of_last_event.value]) < 0.1*self.EPSILON_TIME, \
                "The time of the most recent event --i.e. the absolute after increasing the observed event time with the particle's time origin ({:.3f})-- ({:.3f}) of type {} " \
                .format(self.particles[P].getOrigin(), time_of_last_event + self.particles[P].getOrigin(), type_of_last_event) + \
                "coincides with the time of the next {} currently stored in the times_next_events2 " \
                "(i.e. currently = PRIOR to generating the next event times by _generate_next_events() and updating its content) " \
                "for particle {} on server {} ({:.3f})" \
                .format(type_of_last_event, P, server_last_updated, self.times_next_events2[P][server_last_updated][type_of_last_event.value]) + \
                "\nTimes of next events of both types for ALL servers {}: {}".format(self.times_next_events2[P]) + \
                "\nTimes of next events in the queue of ALL servers: {}".format(self.times_in_server_queues[P])
                ## NOTE: The 0.1 factor multiplying self.EPSILON_TIME is taken from the generate_epsilon_random_time()
                ## method and after multiplying by self.EPSILON_TIME gives the smallest epsilon value that can be
                ## returned by that method.

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

    def assertFinalizationProcessIsOk(self, P, p, special_events_list):
        if len(special_events_list) > 0:
            if self.finalize_info['condition'] in [FinalizeCondition.ACTIVE, FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY]:
                if self.finalize_info['type'] == FinalizeType.REMOVE_CENSORED:
                    assert list_contains_either(special_events_list[-1], EventType.ABSORPTION), \
                            "The finalize directives are: (REMOVE_CENSORED, ACTIVE/NOT_ABSORBED_STATES_BOUNDARY) and the last event contains an ABSORPTION event" \
                            "(P={}, p={}, Events={})" \
                            .format(P, p, special_events_list)
                elif self.finalize_info['type'] == FinalizeType.ABSORB_CENSORED:
                    assert list_contains_either(special_events_list[-1], [EventType.ABSORPTION, EventType.ABSORPTION_F]), \
                            "The finalize directives are: (ABSORB_CENSORED, ACTIVE/NOT_ABSORBED_STATES_BOUNDARY) and the last event contains an ABSORPTION or ABSORPTION_F event" \
                            "(P={}, p={}, Events={})" \
                            .format(P, p, special_events_list)
                elif self.finalize_info['type'] == FinalizeType.ESTIMATE_CENSORED:
                    assert list_contains_either(special_events_list[-1], [EventType.CENSORING, EventType.ABSORPTION]), \
                            "The finalize directives are: (ESTIMATE_CENSORED, ACTIVE/NOT_ABSORBED_STATES_BOUNDARY) and the last event contains either a CENSORING or an ABSORPTION event" \
                            "(P={}, p={}, Events={})" \
                            .format(P, p, special_events_list)
            elif self.finalize_info['condition'] == FinalizeCondition.NOT_START_POSITION:
                assert self.finalize_info['type'] != FinalizeType.ABSORB_CENSORED, \
                        "When the finalize condition is NOT_START_POSITION, the finalize type is NOT ABSORB_CENSORED"
                if self.finalize_info['type'] == FinalizeType.REMOVE_CENSORED:
                    assert list_contains_either(special_events_list[-1], EventType.START_POSITION), \
                            "The finalize directives are: (ESTIMATE_CENSORED, NOT_START_POSITION) and the last event contains a START_POSITION event" \
                            "(P={}, p={}, Events={})" \
                            .format(P, p, special_events_list)
                elif self.finalize_info['type'] == FinalizeType.ESTIMATE_CENSORED:
                    assert list_contains_either(special_events_list[-1], [EventType.CENSORING, EventType.START_POSITION]), \
                            "The finalize directives are: (ESTIMATE_CENSORED, NOT_START_POSITION) and the last event contains either a CENSORING or START_POSITION event" \
                            "(P={}, p={}, Events={})" \
                            .format(P, p, special_events_list)

    def assertSystemConsistency(self, P):
        """
        Checks whether the information stored in the particles coincides with the information
        stored in the attributes containing the current and historical positions and times
        as of the latest time at which we know the state of the system.
        
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
    #---------------------------------- Helper functions --------------------------------------


def estimate_parse_input_parameters(dict_params_simul, dict_params_info):
    "Parses the input parameters passed as dictionaries to the estimate_*() functions"
    set_required_parameters_simul = {'buffer_size_activation', 'nmeantimes', 'nparticles', 'seed'}
    if not set_required_parameters_simul.issubset( dict_params_simul.keys() ):
        raise ValueError("Not all required parameters were given in input dictionary 'dict_params_simul'.\nRequired parameters are:\n{}\nGiven:{}" \
                         .format(sorted(set_required_parameters_simul), sorted(dict_params_simul.keys())))

    if dict_params_info is None:
        dict_params_info = {'plot': False, 'log': False}
    else:
        if not {'plot', 'log'}.issubset(dict_params_info.keys()):
            raise ValueError("Not all required parameters were given in input dictionary 'dict_params_info'.\nRequired parameters are: " \
                             .format({'plot', 'log'}))

    return dict_params_simul, dict_params_info

def estimate_blocking_mc(env_queue :EnvQueueSingleBufferWithJobClasses, agent, dict_params_simul :dict, dict_params_info :dict=None):
    """
    Estimate the blocking probability of a queue system using Monte-Carlo exploration of the queue environment

    Arguments:
    env_queue: Queue environment
        Environment representing the queue system where the simulation takes place.

    agent: Agent
        Agent interacting with the queue environment.

    dict_params_simul: dict
        Dictionary containing the simulation parameters as follows:
        - 'nparticles'
        - 'nmeantimes'
        - 'buffer_size_activation'
        - 'multiplier_for_extended_simulation_time'
        - 'seed'
        OPTIONAL:
        - 'maxevents': maximum number of events for which the simulation should run.
        This is used to make sure different simulations run on different methods (e.g. FV and MC) are comparable.

    dict_params_info: (opt) dict
        Dictionary containing general info parameters as follows:
        - 'plot': whether to create plots
        - 'log': whether to show messages

    Return: tuple
        Tuple with the following elements:
        - proba_blocking_mc: the estimated blocking probability by Monte-Carlo
        - estimated return time to start position: i.e. estimate of E(T)
        - number of return times observed
        - est_mc: the EstimatorQueueBlockingFlemingViot object used in the simulation
    """
    # Parse input parameters
    dict_params_simul, dict_params_info = estimate_parse_input_parameters(dict_params_simul, dict_params_info)
    nmeantimes = dict_params_simul.get('nmeantimes')
    if nmeantimes is not None:
        # The simulation time is set to last for `nparticles*nmeantimes` because
        # the given parameters are assumed to be those used for an FV simulation with `nparticles` particles
        # each of whose simulation lasts for `nmeantimes * max(inter-arrival time)`, so using `nparticles*nmeantimes`
        # as simulation time for the single-particle MC simulation makes the two estimatiom methods more comparable.
        nmeantimes = dict_params_simul['nparticles'] * dict_params_simul['nmeantimes']

    time_start = timer()
    # Object that is used to estimate the blocking probability via Monte-Carlo as "total blocking time" / "total return time" 
    est_mc = EstimatorQueueBlockingFlemingViot(1, env_queue.queue, env_queue.getJobClassRates(),
                                               service_rates=env_queue.getServiceRates(),
                                               buffer_size_activation=dict_params_simul['buffer_size_activation'],
                                               positions_observe=dict_params_simul['buffer_size_activation'],
                                               nmeantimes=nmeantimes,
                                               policy_assign=agent.getAssignmentPolicy(),
                                               mean_lifetime=None,
                                               proba_survival_given_activation=None,
                                               reactivate=False,
                                               finalize_info={'maxevents': dict_params_simul.get('maxevents', np.Inf),
                                                              'type': FinalizeType.REMOVE_CENSORED, #FinalizeType.ESTIMATE_CENSORED,
                                                              'condition': FinalizeCondition.NOT_START_POSITION
                                                              },
                                               seed=dict_params_simul['seed'],
                                               plotFlag=False,  #dict_params_info['plot'],
                                               log=dict_params_info['log'])

    if dict_params_info['log']:
        print("\tStep 1 of 1: Estimating the blocking probability by Monte-Carlo (seed={})...".format(est_mc.seed))
    proba_blocking_mc, _, total_return_time, n_return_observations = est_mc.simulate(EventType.ACTIVATION)
    if dict_params_info['log']:
        print("\t--> Number of observations for Pr(K) estimation: {} ({:.1f}% of simulation time T={}; {:.1f}% of max #events={})" \
              .format(n_return_observations,
                      est_mc.maxtime is not None and total_return_time / est_mc.maxtime * 100 or np.nan, est_mc.maxtime,
                      est_mc.getMaxNumberOfEvents() < np.Inf and est_mc.get_number_events_from_latest_reset() / est_mc.getMaxNumberOfEvents() * 100 or np.nan, est_mc.getMaxNumberOfEvents()))
    time_end = timer()
    exec_time = time_end - time_start
    if dict_params_info['log']:
        print("execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))

    # Compute simulation statistics
    time = est_mc.get_simulation_time()
    nevents = est_mc.nevents
    #assert nevents == est_mc.get_number_events_from_latest_reset()
    dict_stats = dict({
        'time': time,
        'nevents': nevents,
            })

    return  proba_blocking_mc, \
            total_return_time / n_return_observations, \
            n_return_observations, \
            est_mc, dict_stats

def estimate_blocking_fv_AllInOne(  env_queue: EnvQueueSingleBufferWithJobClasses,
                                     agent,
                                     dict_params_simul: dict,
                                     dict_params_info: dict = None):
    """
    Estimate the blocking probability of a queue system using Fleming-Viot exploration of the queue environment

    Arguments:
    env_queue: Queue environment
        Environment representing the queue system where the simulation takes place.

    agent: Agent
        Agent interacting with the queue environment.

    dict_params_simul: dict
        Dictionary containing the simulation parameters as follows:
        - 'nparticles'
        - 'nmeantimes'
        - 'buffer_size_activation'
        - 'burnin_cycles_absorption'    (optional, if not given it is set to 1)
        - 'seed'

    dict_params_info: (opt) dict
        Dictionary containing general info parameters as follows:
        - 'plot': whether to create plots
        - 'log': whether to show messages

    Return: tuple
        Tuple with the following elements:
        - proba_blocking_fv: the estimated blocking probability by Fleming-Viot
        - integral: the integral used in the Fleming-Viot estimator, Approximation 1
        - expected_absorption_time: the denominator E(T) used in the Fleming-Viot estimator, Approximation 1
        - n_survival_curve_observations: the number of observations used to estimate the survival curve 'P(T>t)'
        - n_expected_absorption_observations: the number of observations used to estimate the expected absorption cycle time E(T)
        - est_fv: the EstimatorQueueBlockingFlemingViot object used in the Fleming-Viot simulation
        :param agent:
    """
    # Parse input parameters
    dict_params_simul, dict_params_info = estimate_parse_input_parameters(dict_params_simul, dict_params_info)
    dict_params_simul['burnin_cycles_absorption'] = dict_params_simul.get('burnin_cycles_absorption', 1)

    time_start = timer()
    """
    # Include this portion of code in case we want to compare the estimation of P(T>t)
    # obtained in the FV simulation called below with an estimation obtained from a separate simulation.
    est_surv = EstimatorQueueBlockingFlemingViot(dict_params_simul['nparticles'], env_queue.queue, env_queue.getJobClassRates(),
                                               service_rates=env_queue.getServiceRates(),
                                               buffer_size_activation=dict_params_simul['buffer_size_activation'],
                                               nmeantimes=dict_params_simul['nparticles'] * dict_params_simul['nmeantimes'],
                                               policy_assign=env_queue.getAssignPolicy(),
                                               mean_lifetime=None,
                                               proba_survival_given_activation=None,
                                               reactivate=False,
                                               finalize_info={'type': FinalizeType.REMOVE_CENSORED, 'condition': FinalizeCondition.ACTIVE}, #{'type': FinalizeType.ESTIMATE_CENSORED, 'condition': FinalizeCondition.ACTIVE},
                                               seed=dict_params_simul['seed'],
                                               plotFlag=False, #dict_params_info['plot'],
                                               log=dict_params_info['log'])
    print("\tStep AUXILIARY (for comparison purposes only): Simulating using an ACTIVATION start state to estimate P(T>t / s=act), the survival probability given activation (seed={})...".format(est_surv.seed))
    n_particles_by_start_state, _, surv_counts, _, _ = est_surv.simulate_survival(N_min=None)
    n_survival_curve_observations = surv_counts[0]
    print("\t--> Number of observations for P(T>t) estimation: {} out of N={}".format(n_survival_curve_observations, est_surv.N))
    time_end = timer()
    exec_time = time_end - time_start
    print("execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))
    proba_survival_given_activation_surv = est_surv.estimate_proba_survival_given_activation()
    """

    # Fleming-Viot estimation
    finalize_type = FinalizeType.ABSORB_CENSORED    # We don't need to discard the still active particles because these do not affect the estimation of the conditional blocking probability
    if dict_params_info['log']:
        print("\tRunning Fleming-Viot simulation using an ABSORPTION start state to estimate the blocking probability (seed={})..." \
              .format(dict_params_simul['seed']))
    est_fv = EstimatorQueueBlockingFlemingViot(dict_params_simul['nparticles'], env_queue.queue, env_queue.getJobClassRates(),
                                               service_rates=env_queue.getServiceRates(),
                                               buffer_size_activation=dict_params_simul['buffer_size_activation'],
                                               nmeantimes=dict_params_simul['nmeantimes'],
                                               burnin_cycles_absorption=dict_params_simul['burnin_cycles_absorption'],
                                               policy_assign=agent.getAssignmentPolicy(),
                                               reactivate=True,
                                               finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                               seed=dict_params_simul['seed'],
                                               plotFlag=False,  #dict_params_info['plot'],
                                               log=dict_params_info['log'])
    proba_blocking_fv, integral, expected_absorption_time, n_expected_absorption_time = est_fv.simulate(EventType.ABSORPTION)
    proba_survival_given_activation = est_fv.estimate_proba_survival_given_activation()

    if dict_params_info['plot']:
        plt.figure()
        plt.plot(proba_survival_given_activation['t'], proba_survival_given_activation['P(T>t)'], 'b-')
        #plt.plot(proba_survival_given_activation_surv['t'], proba_survival_given_activation_surv['P(T>t)'], 'r-')
        #plt.title("Comparison between P(T>t) estimations: red=separately; blue=with FV simulation")
        plt.title("P(T>t) estimation")

    # Observed distribution of absorption and activation states
    absorption_states, dist_absorption_states = est_fv.get_absorption_states_distribution()
    activation_states, dist_activation_states = est_fv.get_activation_states_distribution()
    n_survival_curve_observations = est_fv.get_number_observations_proba_survival()
    if dict_params_info['log']:
        print("\t--> Number of observations for P(T>t) estimation from FV simulation: {} (N={})".format(n_survival_curve_observations, est_fv.N))
    if dict_params_info['plot']:
        df_proba_survival_and_blocking_conditional = est_fv.estimate_proba_survival_and_blocking_conditional()
        plot_curve_estimates(df_proba_survival_and_blocking_conditional,
                          dict_params={
                            'birth_rates': est_fv.queue.getBirthRates(),
                            'death_rates': est_fv.queue.getDeathRates(),
                            'K': est_fv.queue.getCapacity(),
                            'nparticles': dict_params_simul['nparticles'],
                            'nmeantimes': dict_params_simul['nmeantimes'],
                            'maxtime_mc': 0.0, #est_fv.maxtime,
                            'maxtime_fv': est_fv.maxtime,
                            'buffer_size_activation': dict_params_simul['buffer_size_activation'],
                            'mean_lifetime': expected_absorption_time,
                            'n_survival_curve_observations': n_survival_curve_observations,
                            'n_absorption_time_observations': n_expected_absorption_time,
                            'proba_blocking_fv': proba_blocking_fv,
                            'finalize_type': finalize_type,
                            'seed': dict_params_simul['seed']
                            })
        if est_fv.nservers > 1:
            plt.figure()
            plot_distribution_states(activation_states, dist_activation_states, freq2=est_fv.dist_activation, label_top=10, title="Distribution of ACTIVATION states (after burn-in)")
            plot_distribution_states(absorption_states, dist_absorption_states, freq2=est_fv.dist_absorption, label_top=10, title="Distribution of ABSORPTION states (burn-in)")

    time_end = timer()
    exec_time = time_end - time_start
    if dict_params_info['log']:
        print("execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))

    # Compute simulation statistics
    time = est_fv.get_simulation_time()
    nevents = est_fv.nevents
    dict_stats = dict({
        'time': time,
        'nevents': nevents
            })

    return  proba_blocking_fv, integral, expected_absorption_time, \
            n_survival_curve_observations, n_expected_absorption_time, \
            est_fv, None, None, dict_stats

def estimate_blocking_fv(   env_queue: EnvQueueSingleBufferWithJobClasses,
                            agent,
                            dict_params_simul: dict,
                            dict_params_info: dict = None):
    """
    Estimate the blocking probability of a queue system using Fleming-Viot exploration of the queue environment

    The estimation is done by running two different simulations:
    a) A simulation on one particle that estimates:
        - the denominator E(T0) where T0 is the absorption cycle time.
        - the survival probability P(T>t) given activation.
    b) A simulation on N particles that estimates the blocking probability given activation, Phi(K,t).
    The simulation time for each particle is defined by 1.x multiple of the largest observed survival time
    during step (a).

    Arguments:
    env_queue: Queue environment
        Environment representing the queue system where the simulation takes place.

    agent: Agent
        Agent interacting with the queue environment.

    dict_params_simul: dict
        Dictionary containing the simulation parameters as follows:
        - 'nparticles'
        - 'nmeantimes'
        - 'buffer_size_activation'
        - 'seed'

    dict_params_info: (opt) dict
        Dictionary containing general info parameters as follows:
        - 'plot': whether to create plots
        - 'log': whether to show messages

    Return: tuple
    Tuple with the following elements:
    - proba_blocking_fv: the estimated blocking probability by Fleming-Viot
    - integral: the integral used in the Fleming-Viot estimator, Approximation 1
    - expected_absorption_time: the denominator E(T0) used in the Fleming-Viot estimator, Approximation 1
    - n_survival_curve_observations: the number of observations used to estimate the survival curve 'P(T>t)'
    - n_expected_absorption_observations: the number of observations used to estimate the expected absorption cycle time E(T0)
    - est_fv: the EstimatorQueueBlockingFlemingViot object used in the Fleming-Viot simulation
    - est_abs: the EstimatorQueueBlockingFlemingViot object used in the Monte Carlo estimation of E(T0) and P(T>t)
    - dict_stats: dictionary of simulation statistics of all the simulations carried out (e.g. #events, simulation time)
    """
    # Parse input parameters
    dict_params_simul, dict_params_info = estimate_parse_input_parameters(dict_params_simul, dict_params_info)

    #-- Monte Carlo process
    # The following two quantities are estimated by this process:
    # - P(T>t): the survival probability given an activation start state
    # - E(T0): the expected absorption cycle time
    # NOTES:
    # - For each absorption cycle, T < T0.
    # - In order to have independent estimates of P(T>t) and E(T0) we observe T and T0 in different absorption cycles,
    # in particular in alternating cycles.
    #
    # Strategy for censored particles: REMOVE_CENSORED
    # i.e. the particles that do not end at the end of an absorption cycle have their truncated cycle removed.
    time_start = timer()
    est_abs, expected_absorption_time, n_absorption_time_observations, \
        proba_survival_given_activation, n_survival_curve_observations = \
            estimate_proba_survival_and_expected_absorption_time_mc(env_queue, agent, dict_params_simul, dict_params_info)

    time_end = timer()
    exec_time = time_end - time_start
    if dict_params_info['log']:
        print("execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))

    #-- Fleming-Viot process
    # The following values are used as input:
    # - estimated expected return time to absorption, E(T0)
    # - the survival curve P(T>t)
    # - the maximum observed survival time (which is used to define the simulation time)
    #
    # Strategy for censored particles: ABSORB_CENSORED
    # i.e. we don't need to discard the still active particles because these do not affect the estimation
    # of the conditional blocking probability
    finalize_type = FinalizeType.ABSORB_CENSORED
    maxtime_from_proba_survival = est_abs.get_max_observed_survival_time()
    seed = dict_params_simul['seed'] + 2
    if dict_params_info['log']:
        print("\tStep 2 of 2: Running Fleming-Viot simulation using an ACTIVATION start state to estimate blocking probability using E(T) = {:.1f} (out of simul time={:.1f}) (seed={})..." \
              .format(expected_absorption_time, est_abs.maxtime, seed))
    est_fv = EstimatorQueueBlockingFlemingViot(dict_params_simul['nparticles'], env_queue.queue, env_queue.getJobClassRates(),
                                               service_rates=env_queue.getServiceRates(),
                                               buffer_size_activation=dict_params_simul['buffer_size_activation'],
                                               nmeantimes=None,
                                               policy_assign=agent.getAssignmentPolicy(),
                                               mean_lifetime=expected_absorption_time,
                                               proba_survival_given_activation=proba_survival_given_activation,
                                               reactivate=True,
                                               finalize_info={'maxtime': maxtime_from_proba_survival,
                                                              'type': finalize_type,
                                                              'condition': FinalizeCondition.ACTIVE},
                                               seed=seed,
                                               plotFlag=False,  #dict_params_info['plot'],
                                               log=dict_params_info['log'])
    proba_blocking_fv, integral, _, _ = est_fv.simulate(EventType.ACTIVATION)
        ## DM-2022/01/09: Note that starting at an ACTIVATION state is ONLY valid in the single-server case.
        ## In fact, in the multi-server case, we should draw the start state from the STATIONARY DISTRIBUTION of activation.
        ## Note that the simulate_until_all_particles_are_activated_before_starting_fv_process() function that is called
        ## by run_simulation() --in turn called by the simulate() method called here-- is called when nservers > 1
        ## and it asserts that no particle is an activation state.
    if dict_params_info['plot']:
        df_proba_survival_and_blocking_conditional = est_fv.estimate_proba_survival_and_blocking_conditional()
        plot_curve_estimates(df_proba_survival_and_blocking_conditional,
                          dict_params={
                            'birth_rates': est_fv.queue.getBirthRates(),
                            'death_rates': est_fv.queue.getDeathRates(),
                            'K': est_fv.queue.getCapacity(),
                            'nparticles': dict_params_simul['nparticles'],
                            'nmeantimes': dict_params_simul['nmeantimes'],
                            'maxtime_mc': est_abs.maxtime,
                            'maxtime_fv': est_fv.maxtime,
                            'buffer_size_activation': dict_params_simul['buffer_size_activation'],
                            'mean_lifetime': expected_absorption_time,
                            'n_survival_curve_observations': n_survival_curve_observations,
                            'n_absorption_time_observations': n_absorption_time_observations,
                            'proba_blocking_fv': proba_blocking_fv,
                            'finalize_type': finalize_type,
                            'seed': seed
                            })

    time_end = timer()
    exec_time = time_end - time_start
    if dict_params_info['log']:
        print("execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time/60))

    # Compute simulation statistics
    time_abs = est_abs.maxtime #est_abs.get_simulation_time()
        ## NOTE: We don't call est_abs.get_simulation_time() to get the simulation time (continuous) used
        ## for the estimation of P(T>t) and E(T0) because that returns the maximum simulation time AFTER removal of
        ## any subtrajectories that did not come back to the initial buffer size (when REMOVE_CENSORED strategy is
        ## in place, which is the most common.
        ## And this is NOT what we want to store here: we want to store the FULL simulation time used.
    time_fv = est_fv.get_simulation_time()
        ## Last observed time SUMMED over all particles
        ## The above note is NOT a problem here because the strategy for censored particles in the FV simulation
        ## is ABSORB_CENSORED, whose reason is explained above.
    time = time_abs + time_fv
    time_abs_prop = time_abs / time
    time_fv_prop = time_fv / time
    nevents_abs = est_abs.nevents
    nevents_fv = est_fv.nevents
    nevents = nevents_abs + nevents_fv
    nevents_abs_prop = nevents_abs / nevents
    nevents_fv_prop = nevents_fv / nevents
    dict_stats = dict({
        'time': time,
        'time_abs': time_abs,
        'time_fv': time_fv,
        'time_abs_prop': time_abs_prop,
        'time_fv_prop': time_fv_prop,
        'time_max_survival': maxtime_from_proba_survival,
        'nevents': nevents,
        'nevents_abs': nevents_abs,
        'nevents_fv': nevents_fv,
        'nevents_abs_prop': nevents_abs_prop,
        'nevents_fv_prop': nevents_fv_prop,
            })
    assert est_fv.getMaxSimulationTime() == maxtime_from_proba_survival, \
            "The simulation time of the FV process ({:.1f}) coincides with the maximum observed survival time ({:.1f})"\
            .format(est_fv.getMaxSimulationTime(), maxtime_from_proba_survival)

    return  proba_blocking_fv, integral, expected_absorption_time, \
            n_survival_curve_observations, n_absorption_time_observations, \
            est_fv, est_abs, dict_stats

@measure_exec_time
def estimate_proba_survival_and_expected_absorption_time_mc(env_queue, agent, dict_params_simul, dict_params_info):
    """
    Estimates the survival probability given activation, P(T>t), and the expected absorption cycle time, E(T0)
    using Monte-Carlo simulation of a single queue (i.e. from one particle let run long enough).

    Arguments:
    env_queue: Queue environment
        Environment representing the queue system where the simulation takes place.

    agent: Agent
        Agent interacting with the queue environment.

    dict_params_simul: dict
        Dictionary containing the simulation parameters as follows:
        - 'nmeantimes'
        - 'buffer_size_activation'
        - 'seed'

    dict_params_info: (opt) dict
        Dictionary containing general info parameters as follows:
        - 'plot': whether to create plots
        - 'log': whether to show messages

    Return: tuple
    Tuple with the following elements:
    - est_mc: the EstimatorQueueBlockingFlemingViot object used in the Monte Carlo estimation of E(T0) and P(T>t)
    - expected_absorption_time: the expected absorption cycle time E(T0)
    - n_absorption_time_observations: the number of observations used to estimate the expected absorption cycle time E(T0)
    - proba_survival_given_activation: data frame containing the estimated survival probability given activation, 'P(T>t)' for each 't'.
    - n_survival_curve_observations: the number of observations used to estimate the survival curve P(T>t)
    """
    # Parse input parameters
    dict_params_simul, dict_params_info = estimate_parse_input_parameters(dict_params_simul, dict_params_info)

    # Run the Monte-Carlo simulation on one particle
    est_mc = EstimatorQueueBlockingFlemingViot(1, env_queue.queue, env_queue.getJobClassRates(),
                                                service_rates=env_queue.getServiceRates(),
                                                buffer_size_activation=dict_params_simul['buffer_size_activation'],
                                                positions_observe=dict_params_simul['buffer_size_activation'] - 1,
                                                nmeantimes=dict_params_simul['nmeantimes'],
                                                policy_assign=agent.getAssignmentPolicy(),
                                                mean_lifetime=None,
                                                proba_survival_given_activation=None,
                                                reactivate=False,
                                                finalize_info={'type': FinalizeType.REMOVE_CENSORED,
                                                               'condition': FinalizeCondition.NOT_ABSORBED_STATES_BOUNDARY},
                                                seed=dict_params_simul['seed'],
                                                plotFlag=False,  #dict_params_info['plot'],
                                                log=dict_params_info['log'])
    if dict_params_info['log']:
        print("\tStep 1 of 2: Running simulation on one particle to estimate E(T) and P(T>t): the start state is picked at random at the boundary of the set of absorbed states (seed={})...".format(est_mc.seed))
    est_mc.simulate(EventType.ABSORPTION)

    # E(T0)
    expected_absorption_time, total_absorption_time, n_absorption_time_observations = est_mc.estimate_expected_absorption_time_from_killings()
    n_survival_curve_observations = est_mc.get_number_observations_proba_survival()
    if dict_params_info['log']:
        print("\t--> Number of observations for E(T) ({:.1f}) and P(T>t) estimation: ({}, {}) on one particle running for {:.1f}% of total simulation time T={:.1f})." \
              .format(expected_absorption_time, n_absorption_time_observations, n_survival_curve_observations, total_absorption_time / est_mc.maxtime * 100, est_mc.maxtime))
    absorption_states, dist_absorption_states = est_mc.get_absorption_states_distribution()
    activation_states, dist_activation_states = est_mc.get_activation_states_distribution()
    if est_mc.nservers > 1:
        if dict_params_info['log']:
            print("Distribution of absorption states:")
            print_states_distribution(absorption_states, dist_absorption_states)

        if dict_params_info['plot'] > 1:
            # Show the distribution of activation states and of absorption states
            plot_distribution_states(activation_states, dist_activation_states, title="Distribution of ACTIVATION states")
            plot_distribution_states(absorption_states, dist_absorption_states, title="Distribution of ABSORPTION states")

    # P(T>t)
    proba_survival_given_activation = est_mc.estimate_proba_survival_given_activation()

    return est_mc, expected_absorption_time, n_absorption_time_observations, \
                proba_survival_given_activation, n_survival_curve_observations

def plot_curve_estimates(df_proba_survival_and_blocking_conditional, dict_params, log=False):
    rhos = [b/d for b, d in zip(dict_params['birth_rates'], dict_params['death_rates'])]
    # True blocking probability
    proba_blocking_K = compute_blocking_probability_birth_death_process(rhos, dict_params['K'])
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
        print("#obs for E(T) estimation={}".format(dict_params['n_absorption_time_observations']))
        print("Pr(K)={:6f}%".format(dict_params['proba_blocking_fv']*100))
        print("True Pr(K)={:.6f}%".format(proba_blocking_K*100))
        print("finalize_type={}".format(dict_params['finalize_type'.name]))
        print("seed={}".format(dict_params['seed']))

    plt.figure()
    color1 = 'blue'
    color2 = 'red'
    plt.step(df_proba_survival_and_blocking_conditional['t'], df_proba_survival_and_blocking_conditional['P(T>t)'],
             'b-', where='post')
    ax = plt.gca()
    ax.set_xlabel('t')
    ax.spines['left'].set_color(color1)
    ax.tick_params(axis='y', color=color1)
    ax.yaxis.label.set_color(color1)
    ax.hlines(0.0, 0, ax.get_xlim()[1], color='gray')
    ax2 = ax.twinx()
    ax2.step(df_proba_survival_and_blocking_conditional['t'], df_proba_survival_and_blocking_conditional['P(BLOCK / T>t)'],
             'r-', where='post')
    # Plot the product
    ax2.step(df_proba_survival_and_blocking_conditional['t'], df_proba_survival_and_blocking_conditional['P(T>t)']*df_proba_survival_and_blocking_conditional['P(BLOCK / T>t)'],
             'g-', where='post')
    # Set the maximum value of the secondary vertical axis (red curve) 
    #y2max = None
    #y2max = 0.05
    #y2max = 1.0
    #y2max = 1.1*np.max(df_proba_survival_and_blocking_conditional['P(BLOCK / T>t)'])
    y2max = 1.2*ax2.get_ylim()[1]
    if y2max is not None:
        ax2.set_ylim(0, y2max)
    ax.spines['right'].set_color(color2)
    ax2.tick_params(axis='y', color=color2)
    ax2.yaxis.label.set_color(color2)
    plt.sca(ax)
    ax.legend(['P(T>t / s={})\n(n={})'.format(dict_params['buffer_size_activation'], dict_params['n_survival_curve_observations'])], loc='upper left')
    ax2.legend(['P(BLOCK / T>t,s={})\n(P(K)={:.6f}% vs. {:.6f}%)'.format(dict_params['buffer_size_activation'], dict_params['proba_blocking_fv']*100, proba_blocking_K*100)], loc='upper right')
    plt.title("K={}, rhos={}, N={}, activation size={}, maxtime(1)={:.1f}, maxtime(N)={:.1f}, mean_lifetime={}(n={})" \
              ", finalize={}, seed={}" \
              .format(dict_params['K'], rhos, dict_params['nparticles'], dict_params['buffer_size_activation'], dict_params['maxtime_mc'], dict_params['maxtime_fv'],
                      dict_params['mean_lifetime'] is not None and "{:.1f}".format(dict_params['mean_lifetime']) or np.nan,
                      dict_params['n_absorption_time_observations'],
                      dict_params['finalize_type'].name[0:3],
                      dict_params['seed']
                      ))
    ax.title.set_fontsize(9)

def plot_distribution_states(states, freq, freq2=None, label_top=10, title="Distribution of states"):
    """
    Plots the distribution of the states through which particles are absorbed at each absorption event.
    Optionally a second distribution on the same states is plotted for comparison.

    Arguments:
    states: list
        States whose distribution is plotted
    freq: list
        Frequency distribution of the states to plot.
    freq2: list (opt)
        Secondary frequency distribution of the states that is compared with `freq`.
        Ex: theoretical distribution to be compared with the observed distribution.
    label_top: int
        Number of top frequencies in `freq` to label with the freq value plotted.
    title: str
        Overall title of the plot.
    """
    assert len(states) == len(freq)
    if freq2 is not None:
        assert len(states) == len(freq2)

    # Prepare the data to plot
    state_indices = range(len(states))
    n_absorptions = np.sum(freq)
    dist = [f / max(1, n_absorptions) for f in freq]
    if freq2 is not None:
        n2 = np.sum(freq2)
        dist2 = [f / max(1, n2) for f in freq2]

    #-- Horizontal bars at (x,y)
    if freq2 is None:
        (_, axes) = plt.subplots(1, 1, figsize=(12,6))
        ax1 = axes
    else:
        (_, axes) = plt.subplots(1, 3, figsize=(12,6))
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
    ax1.barh(state_indices, dist, color="blue")
    if freq2 is not None:
        ax2.barh(state_indices, dist2, color="red")
        ax3.barh(state_indices, [d2 - d for d, d2 in zip(dist, dist2)], color="green")
        ymax = np.max(dist + dist2)
        ax1.set_xlim(0, ymax)
        ax2.set_xlim(0, ymax)
        ax3.set_xlim(-ymax, ymax)
        ax1.title.set_text("Observed freq. during simulation\n(n={})".format(np.sum(freq)))
        ax2.title.set_text("Restricted Markov chain")
        ax3.title.set_text("Difference (right - left)")
    plt.suptitle(title)

    #-- Label top frequency states
    idx_top_freq = np.argsort(freq)[-label_top:]
    for idx_state in idx_top_freq:
        # Note that the position of the text is given as (y,x) because the bars are HORIZONTAL bars
        # => the (x,y) coordinates given in plt.barh() are reversed in the generated axis
        ax1.text(dist[idx_state], idx_state, "s = {}".format(states[idx_state]), fontsize=8)
    if freq2 is not None:
        idx_top_freq2 = np.argsort(freq2)[-label_top:]
        for idx_state in idx_top_freq2:
            # Note that the position of the text is given as (y,x) because the bars are HORIZONTAL bars
            # => the (x,y) coordinates given in plt.barh() are reversed in the generated axis
            ax2.text(dist2[idx_state], idx_state, "s = {}".format(states[idx_state]), fontsize=8)

def print_states_distribution(states, freq):
    n = sum(freq)
    if n > 0:
        for s, f in zip(states, freq):
            if f > 0:
                print("{}: {} ({:.1f}%)".format(s, f, f/n*100))
    else:
        warnings.warn("No states have been observed.")


if __name__ == "__main__":

    #------------------------------ Auxiliary functions ------------------------------#
    def plot_survival_curve(df_proba_survival, col='b-', title=None):
        plt.step(df_proba_survival['t'], df_proba_survival['P(T>t)'], col, where='post')
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

    def compute_observed_rates(df):
        "Computes the observed birth rate lambda and death rate mu from a simulated trajectory"
        trefb = 0.0      # Reference time for each observed birth time
        trefd = np.nan   # Reference time for each observed death time
        btimes = []
        dtimes = []
        ncases = 0
        for i, t in enumerate(df['t']):
            if np.isnan(trefd) and df['x'].iloc[i] > 0:
                trefd = t
                #print("---> i={}, trefd={}".format(i, trefd))
            if i > 0:
                ncases += 1
                if df['x'].iloc[i] - df['x'].iloc[i-1] >= 0:
                    # A BIRTH occured (the delta x is 0 when the job arrives but the queue was full)
                    btimes += [t - trefb]
                    trefb = t
                else:
                    # A DEATH occured
                    #print("i={}: trefd={}, tprev={}, t={}, size_prev={}, size={}, dt={}".format(i, trefd, df['t'].iloc[i-1], t, df['x'].iloc[i-1], df['x'].iloc[i], t-tref))
                    assert not np.isnan(trefd), "i={}: trefd={}, tprev={}, t={}, size_prev={}, size={}".format(i, trefd, df['t'].iloc[i-1], t, df['x'].iloc[i-1], df['x'].iloc[i])
                    dtimes += [t - trefd]
                    if df['x'].iloc[i] > 0:
                        trefd = t
                        #print("------> i={}, tref={}".format(i, tref))
                    else:
                        trefd = np.nan

        return 1/np.mean(btimes), len(btimes), 1/np.mean(dtimes), len(dtimes)

    def compute_true_expected_return_time(est, s):
        """
        Computes the true expected return time to the specified state s as 1/q(s)/Pr(s)
        for the server.
        
        where:
        q(s) is the exponential rate of the sojourn time in i. In our single server case is lambda.
        Pr(s) is the stationary probability of the state.
        
        Arguments
        est: EstimatorQueueBlockingFlemingViot
            The object used to run the simulation that estimates the blocking probability.
            It is expected to contain the following attributes:
            - nservers: number of servers in the system
            - queue.K: the capacity of the queue
            - rhos: the load of each server in the system

        s: int
            State on which the expected return time is wished.
        """
        x, dist = stationary_distribution_birth_death_process(est.nservers, est.queue.K, est.rhos)
        ET_expected = 1/dist[s]/np.sum(est.queue.getBirthRates())

        return ET_expected

    def plot_event_times_dist(est, event_type :EventType):
        """
        Plots the distribution of observed times of the specified event (e.g. ABSORPTION)
        and returns their average, as well as their expected value for comparison.

        Arguments:
        est: EstimatorQueueBlockingFlemingViot
            Object used to simulate the queue and estimate the blocking probability under FV or MC.

        event_type: EventType
            Type of event defining the observed times to collect.

        Return: tuple
        duple with the following values:
        - average of the observed event times
        - true expected event time (E(T)) when the event type is ABSORPTION and the absorption set is 0,
        or None otherwise.
        """
        if event_type == EventType.ABSORPTION and est.buffer_size_activation:
            # We are interested in the expected absorption time and the absorption set is s=0
            # In this case, it is directly computable as the expected return time to s=0
            # (as the queue can only reach s=0 from above (i.e. via an absorption),
            # as opposed to from below, which would be the other way of returning to the state)
            ET_expected = compute_true_expected_return_time(est, s=0)
        else:
            ET_expected = None

        # Observed event times
        idx_abs = np.where([event_type in value for value in est.info_particles[0]['E']])[0]
        times_abs = [est.info_particles[0]['t'][idx] for idx in idx_abs]
        times_abs_diff = np.diff(times_abs)
        nobs = len(times_abs_diff)
        # Expected value
        ET = np.mean(times_abs_diff)
        print("\nAverage observed {} time: E(T): (n={}) Mean = {:.3f}, Median = {:.3f}".format( event_type.name, nobs, ET, np.median(times_abs_diff) ))

        # Plot and histogram
        (fig, axes) = plt.subplots(1, 2, figsize=(10,4))
        ax1 = axes[0]
        ax2 = axes[1]
        #ax1.plot(times_abs[1:], times_abs_diff)
        #ax1.xlabel("t")
        ax1.plot(times_abs_diff)
        ax1.set_xlabel("Time index")
        #ax1.title.set_text("Times used in the estimation of E(T). Expected={:.3f}".format(ET_expected))
        ax1.set_ylabel("T")
        ax2.hist(times_abs_diff, orientation="horizontal", bins=20)
        plt.suptitle("Times used in the estimation of expected {} time: E(T)={:.3f}".format(event_type.name, ET) + (ET_expected is None and " " or "; Expected={:.3f}".format(ET_expected)))

        return nobs, ET, ET_expected
    #------------------------------ Auxiliary functions ------------------------------#


    tests2run = [2.1, 2.2, 3.1]

    #----------------------- Unit tests on specific methods --------------------------#
    time_start = timer()
    seed = 1717
    plotFlag = False
    log = False

    #--- Test #1.1: simulate_survival()
    if 1.1 in tests2run:
        print("Test #1.1: simulate_survival() method")
        K = 20
        rate_birth = 0.5
        job_class_rates = [rate_birth]
        rate_death = [1]
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)

        # Simulation
        nparticles0 = 200
        nmeantimes0 = 50
        finalize_type = FinalizeType.REMOVE_CENSORED

        # Monte-Carlo (to estimate expected absorption cycle time)
        buffer_size_activations = [1, 5, 8]
        for buffer_size_activation in buffer_size_activations:
            nparticles = nparticles0 #* buffer_size_activation
            nmeantimes = nmeantimes0 #* buffer_size_activation
            print("\nRunning Monte-Carlo simulation on single-server system to estimate survival probability curve for buffer_size_activation={} on N={} particles and simulation time T={}x...".format(buffer_size_activation, nparticles, nmeantimes))
            est_surv = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_class_rates,
                                                       service_rates=None,
                                                       buffer_size_activation=buffer_size_activation,
                                                       nmeantimes=nparticles*nmeantimes,
                                                       mean_lifetime=None,
                                                       reactivate=False,
                                                       finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                       plotFlag=plotFlag,
                                                       seed=seed, log=log)
            n_particles_by_start_state, surv_times, surv_counts, _, _ = est_surv.simulate_survival()
            df_proba_survival = est_surv.estimate_proba_survival_given_activation()

            # Estimate the survival probability starting at 0 to see if there is any difference in the estimation w.r.t. to the above (correct) method
            est_mc = EstimatorQueueBlockingFlemingViot(1, queue, job_class_rates,
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
            expected_absorption_time, total_absorption_time, total_absorption_n = est_mc.estimate_expected_absorption_time_from_killings()

            plt.figure()
            plot_survival_curve(df_proba_survival_start_at_absorption, col='r-')
            plot_survival_curve(df_proba_survival, title="Buffer size for activation = {}, E(T)={:.1f} (red: MC, start@ABS; blue: start@ACT)".format(buffer_size_activation, expected_absorption_time))

    #--- Test #1.2: variability of survival curve
    if 1.2 in tests2run:
        print("Test #1.2: compare variability of survival curve among different replications --goal: find out why we get so much variability in the FV estimation of the blocking probability (CV ~ 60%!)")
        ## CONCLUSION:
        K = 20
        rate_birth = 0.5
        job_class_rates = [rate_birth]
        rate_death = [1]
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)

        # Simulation
        nparticles0 = 200
        nmeantimes0 = 50
        finalize_type = FinalizeType.REMOVE_CENSORED

        replications = 12

        # Monte-Carlo (to estimate expected absorption cycle time)
        buffer_size_activation = 8
        df_proba_survival = dict()
        for rep in range(replications):
            nparticles = nparticles0 #* buffer_size_activation
            nmeantimes = nmeantimes0 #* buffer_size_activation
            print("\nRunning Monte-Carlo simulation on single-server system to estimate survival probability curve for buffer_size_activation={} on N={} particles and simulation time T={}x...".format(buffer_size_activation, nparticles, nmeantimes))
            est_surv = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_class_rates,
                                                       service_rates=None,
                                                       buffer_size_activation=buffer_size_activation,
                                                       nmeantimes=nparticles*nmeantimes,
                                                       mean_lifetime=None,
                                                       reactivate=False,
                                                       finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                       plotFlag=plotFlag,
                                                       seed=seed+rep, log=log)
            n_particles_by_start_state, surv_times, surv_counts, _, _ = est_surv.simulate_survival()
            df_proba_survival[rep] = est_surv.estimate_proba_survival_given_activation()

        for rep in range(replications):
            plot_survival_curve(df_proba_survival[rep])
        plt.title("Buffer size for activation = {}: survival curve over different replications".format(buffer_size_activation))

    #--- Test #1.3: variability of Phi(t), the red curve
    if 1.3 in tests2run:
        print("Test #1.3: compare variability of Phi(t) among different replications --goal: find out why we get so much variability in the FV estimation of the blocking probability (CV ~ 60%!)")
        ## CONCLUSION: The curve has a large variability...
        K = 20
        rate_birth = 0.5
        job_class_rates = [rate_birth]
        rate_death = [1]
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)

        # Simulation
        nparticles = 200
        nmeantimes = 50
        finalize_type = FinalizeType.ABSORB_CENSORED

        replications = 12

        # Monte-Carlo (to estimate expected absorption cycle time)
        buffer_size_activation = 8
        # Dictionary where the different curve estimates will be stored so that we can merge them
        # into the same time axis that is common to all replications which will allow us to average over them
        dict_proba_block = dict()
        colormap = cm.get_cmap("jet")
        for rep in range(replications):
            print("\nRep {} of {}: Running Monte-Carlo simulation on single-server system to estimate the conditional blocking probability curve for buffer_size_activation={} on N={} particles and simulation time T={}x...".format(rep+1, replications, buffer_size_activation, nparticles, nmeantimes))
            est_block = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_class_rates,
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
            np.set_printoptions(precision=3, suppress=True)
            print(df_proba_block_all_reps)
            np.set_printoptions(precision=DEFAULT_NUMPY_PRECISION, suppress=DEFAULT_NUMPY_SUPPRESS)

        plt.figure()
        plot_curve_with_bands(df_proba_block_all_reps['t'], df_proba_block_all_reps['y_mean'], 2*df_proba_block_all_reps['y_se'], col='r-', color='red',
                              xlabel='t', title="K={}, Buffer size for activation = {}, N={}: Phi(t,K) over different replications".format(K, buffer_size_activation, nparticles))
    #----------------------- Unit tests on specific methods --------------------------#


    #------------------------- Unit tests on the process -----------------------------#
    time_start = timer()
    seed = 1717
    plotFlag = False
    log = False

    #--- Test #2.1: Single server
    if 2.1 in tests2run:
        print("\nTest #2.1: Single server system")
        K = 5
        rate_birth = 0.5
        job_class_rates = [rate_birth]
        rate_death = [1]
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)

        # Simulation
        nparticles = 20
        nmeantimes = 30
        buffer_size_activation = 3
        finalize_type = FinalizeType.ABSORB_CENSORED

        print("[test_QB] Computing TRUE blocking probability...")
        proba_blocking_K = compute_blocking_probability_birth_death_process([rate_birth / rate_death[0]], K)

        # a) Monte-Carlo (to estimate expected absorption cycle time)
        print("[test_QB] Running Monte-Carlo simulation on 1 particle and T={}x...".format(nparticles*nmeantimes))
        est_mc = EstimatorQueueBlockingFlemingViot(1, queue, job_class_rates,
                                                   service_rates=None,
                                                   buffer_size_activation=1,
                                                   positions_observe=0,
                                                   nmeantimes=nparticles*nmeantimes,
                                                   mean_lifetime=None,
                                                   reactivate=False,
                                                   finalize_info={'type': FinalizeType.REMOVE_CENSORED, 'condition': FinalizeCondition.ACTIVE},
                                                   plotFlag=plotFlag,
                                                   seed=seed, log=log)
        proba_blocking_mc, total_blocking_time, total_return_time, total_absorption_n = est_mc.simulate(EventType.ABSORPTION) #EventType.ACTIVATION)
        expected_absorption_time, total_absorption_time, total_absorption_n = est_mc.estimate_expected_absorption_time_from_killings()

        # Check observed lambda and mu
        df = pd.DataFrame({'t': est_mc.all_times_buffer[0], 'x': est_mc.all_positions_buffer[0]})
        lmbda, nlambda, mu, nmu = compute_observed_rates(df)
        # Check E(T)
        nET, ET, ET_expected = plot_event_times_dist(est_mc, EventType.ABSORPTION)

        print("\n[test_QB] Computing observed lambda and mu...")
        print("[test_QB] Observed lambda={:.3f} (n={})".format(lmbda, nlambda))
        print("[test_QB] Expected lambda={:.3f}".format(est_mc.queue.getBirthRates()[0]))
        print("[test_QB] Observed mu={:.3f} (n={})".format(mu, nmu))
        print("[test_QB] Expected mu={:.3f}".format(est_mc.queue.getDeathRates()[0]))
        print("[test_QB] Observed rho={:.3f}".format(lmbda/ mu))
        print("[test_QB] Expected rho={}".format(est_mc.rhos[0]))
        print("[test_QB] Estimated expected absorption cycle time={:.3f}".format(expected_absorption_time))
        print("[test_QB] Observed E(T)={:.3f} (n={}) (should be the same as 'estimatd expected absorption cycle time')".format(ET, nET))
        print("[test_QB] Expected E(T)={:.3f}".format(ET_expected))
        assert "{:.3f} (n={})".format(lmbda, nlambda) == "0.474 (n=568)"
        assert "{:.3f} (n={})".format(mu, nmu) == "0.972 (n=563)"
        assert "{:.3f}".format(lmbda/ mu) == "0.488"
        assert "{:.3f} (n={})".format(ET, nET) == "3.930 (n=305)"
        assert np.abs(ET - expected_absorption_time) < 1E-6

        # b) Fleming-Viot
        print("\n[test_QB] Running Fleming-Viot simulation on {} particles and T={}x...".format(nparticles, nmeantimes))
        est_fv = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_class_rates,
                                                   service_rates=None,
                                                   buffer_size_activation=1,
                                                   nmeantimes=nmeantimes,
                                                   burnin_cycles_absorption=3,
                                                   burnin_cycles_complete_all_particles=False,
                                                   reactivate=True,
                                                   finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                   plotFlag=plotFlag,
                                                   seed=seed, log=log)
        proba_blocking_fv, integral, expected_absorption_time, n_expected_absorption_time = est_fv.simulate(EventType.ABSORPTION) # EventType.ACTIVATION)

        # Plot the estimation curves
        df_proba_survival_and_blocking_conditional = est_fv.estimate_proba_survival_and_blocking_conditional()
        n_absorption_time_observations = df_proba_survival_and_blocking_conditional.shape[0] - 1
        plot_curve_estimates(df_proba_survival_and_blocking_conditional,
                          dict_params={
                            'birth_rates': est_fv.queue.getBirthRates(),
                            'death_rates': est_fv.queue.getDeathRates(),
                            'K': est_fv.queue.getCapacity(),
                            'nparticles': nparticles,
                            'nmeantimes': nmeantimes,
                            'maxtime_mc': est_mc.maxtime,
                            'maxtime_fv': est_fv.maxtime,
                            'buffer_size_activation': buffer_size_activation,
                            'mean_lifetime': expected_absorption_time,
                            'n_survival_curve_observations': est_fv.N,
                            'n_absorption_time_observations': n_absorption_time_observations,
                            'proba_blocking_fv': proba_blocking_fv,
                            'finalize_type': finalize_type,
                            'seed': seed
                            })

        time_end = timer()
        print("[test_QB] Execution time: {:.1f} min".format((time_end - time_start) / 60))

        # c) Assertions
        print("[test_QB] P(K) true: {:.6f}%".format(proba_blocking_K*100))    # K=5: 1.587302%
        print("[test_QB] P(K) by MC: {:.6f}% (#events={})".format(proba_blocking_mc*100, est_mc.nevents))
        print("[test_QB] P(K) estimated by FV: {:.6f}% (#events={})".format(proba_blocking_fv*100, est_fv.nevents))
        print("[test_QB] E(T) estimated by FV: {:.1f} (n={})".format(expected_absorption_time, n_expected_absorption_time))
        assert "{:.6f}% (#events={})".format(proba_blocking_mc*100, est_mc.nevents) == "1.065424% (#events=1131)"
        assert "{:.6f}% (#events={})".format(proba_blocking_fv*100, est_fv.nevents) == "0.320571% (#events=484)"
        assert "{:.1f} (n={})".format(expected_absorption_time, n_expected_absorption_time) == "3.2 (n=20)"

    #--- Test #2.2: Single server with the whole estimation process being taken care of by the calling function below
    if 2.2 in tests2run:
        print("\nTest #2.2: Single server system")
        nservers = 1
        K = 5
        rate_birth = 0.5
        job_class_rates = [rate_birth]
        rate_death = [1]
        policy_assign = PolJobAssignmentProbabilistic([[1]])       # Trivial assignment policy as there is only one job class and one server
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)
        env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates=job_class_rates, reward_func=rewardOnJobClassAcceptance, rewards_accept_by_job_class=[1])

        # Define the agent acting on the queue environment
        policies = dict({QueuePolicyTypes.ACCEPT: None, QueuePolicyTypes.ASSIGN: policy_assign})
        learners = None
        agent = AgeQueue(queue, policies, learners)

        # Simulation parameters
        dict_params_simul = {
            'nparticles': 20,
            'nmeantimes': 40,
            'buffer_size_activation': int(0.5*K),
            'burnin_cycles_absorption': 3,
            'seed': seed,
                }

        # Info parameters 
        dict_params_info = {'plot': True, 'log': False}

        print("[test_QB] Computing TRUE blocking probability...")
        proba_blocking_K = compute_blocking_probability_birth_death_process([rate_birth / rate_death[0]], K)

        # Run!
        print("[test_QB] Running Fleming-Viot estimation procedure: K={}, BSA={}, N={}, T={}x...".format(K, dict_params_simul['buffer_size_activation'], dict_params_simul['nparticles'], dict_params_simul['nmeantimes']))
        time_start = timer()
        proba_blocking_fv, _, expected_absorption_time, _, n_expected_absorption_time, _, _, _, dict_stats = estimate_blocking_fv_AllInOne(env_queue, agent, dict_params_simul, dict_params_info=dict_params_info)
        time_end = timer()
        dt_end = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        print("[test_QB] Ended at: {}".format(dt_end))
        print("[test_QB] Execution time: {:.1f} min, {:.1f} hours".format((time_end - time_start) / 60, (time_end - time_start) / 3600))

        print("\n[test_QB] True blocking probability: Pr(K)={:.6f}%".format(proba_blocking_K*100))    # K=5: 1.587302%
        print("[test_QB] Estimated blocking probability: Pr(FV)={:.6f}% (#events={})".format(proba_blocking_fv*100, dict_stats['nevents']))
        print("[test_QB] E(T) estimated by FV: {:.1f} (n={})".format(expected_absorption_time, n_expected_absorption_time))
        assert "{:.6f}% (#events={})".format(proba_blocking_fv*100, dict_stats['nevents']) == "1.758346% (#events=1168)"
        assert "{:.1f} (n={})".format(expected_absorption_time, n_expected_absorption_time) == "7.7 (n=20)"

    #--- Test #3.1: Multi-server
    if 3.1 in tests2run:
        print("\nTest #3.1: Multiple-server system")
        nservers = 3
        K = 5
        rate_birth = 0.5 # This value is not really used but it's needed to construct the `queue` object
        job_class_rates = [0.8, 0.7] #[0.7, 0.7] #[0.8, 0.7]
        rate_death = [1, 1, 1]
        policy_assign = PolJobAssignmentProbabilistic([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])   #[[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]] #[[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
        queue = queues.QueueMM(rate_birth, rate_death, nservers, K)
        # Queue environment (to pass to the simulation functions)
        env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates=job_class_rates, reward_func=rewardOnJobClassAcceptance, rewards_accept_by_job_class=[1] * len(job_class_rates))

        # Theoretical expected values for E(T):
        # K=5, rhos=[0.4, 0.75, 0.35]: E(T) = 8.1
        # K=20, rhos=[0.4, 0.75, 0.35]: E(T) = 10.4

        print("[test_QB] Computing TRUE blocking probability...")
        job_rates_by_server = compute_job_rates_by_server(job_class_rates, nservers, policy_assign.getProbabilisticMap())
        proba_blocking_K = compute_blocking_probability_birth_death_process(get_server_loads(job_rates_by_server, rate_death), K)

        # Simulation
        nparticles = 20
        nmeantimes = 40
        buffer_size_activation = 3
        finalize_type = FinalizeType.ABSORB_CENSORED

        # a) Monte-Carlo (to estimate expected absorption cycle time)
        print("[test_QB] Running Monte-Carlo simulation on 1 particle, K={}, T={}x...".format(K, nparticles*nmeantimes))
        est_mc = EstimatorQueueBlockingFlemingViot(1, queue, job_class_rates,
                                                   service_rates=env_queue.getServiceRates(),
                                                   buffer_size_activation=3,
                                                   positions_observe=3,
                                                   nmeantimes=nparticles*nmeantimes,
                                                   policy_assign=policy_assign,
                                                   reactivate=False,
                                                   finalize_info={'type': FinalizeType.REMOVE_CENSORED, 'condition': FinalizeCondition.ACTIVE},
                                                   seed=seed,
                                                   plotFlag=plotFlag,
                                                   log=log)
        proba_blocking_mc, total_blocking_time, total_return_time, total_return_n = est_mc.simulate(EventType.ACTIVATION)
        expected_absorption_time, total_absorption_time, total_absorption_n = est_mc.estimate_expected_absorption_time_from_killings()

        # Check number of events
        assert est_mc.nevents == est_mc.get_number_events_from_latest_reset(), "nevents={}, get_number={}".format(est_mc.nevents, est_mc.get_number_events_from_latest_reset())

        # Check E(T)
        nET, ET, ET_expected = plot_event_times_dist(est_mc, EventType.ABSORPTION)

        # Check observed lambdas and mus
        df = np.empty((nservers,), dtype=pd.DataFrame)
        lambdas = np.nan*np.ones((nservers,))
        mus = np.nan*np.ones((nservers,))
        nlambdas = np.zeros((nservers,), dtype=int)
        nmus = np.zeros((nservers,), dtype=int)
        print("")
        for server in range(nservers):
            print("Computing observed lambda and mu for server {}...".format(server))
            df[server] = pd.DataFrame({'t': est_mc.all_times_by_server[0][server], 'x': est_mc.all_positions_by_server[0][server]})
            lambdas[server], nlambdas[server], mus[server], nmus[server] = compute_observed_rates(df[server])
        print("\n[test_QB] Computing observed lambda and mu...")
        print("[test_QB] Observed lambdas={} (n={})".format(lambdas, nlambdas))
        print("[test_QB] Expected lambda={}".format(est_mc.queue.getBirthRates()))
        print("[test_QB] Observed mu={} (n={})".format(mus, nmus))
        print("[test_QB] Expected mu={}".format(est_mc.queue.getDeathRates()))
        print("[test_QB] Observed rhos={}".format([l/m for l, m in zip(lambdas, mus)]))
        print("[test_QB] Expected rhos={}".format(est_mc.rhos))
        print("[test_QB] Observed E(T)={:.3f} (n={})".format(ET, nET))    # Note: Expected E(T) is None
        # Assertions when assigning arriving jobs directly to servers
        # (i.e. function self._generate_birth_times_from_equivalent_rates() is used
        # instead of function self._generate_birth_times())
        assert "{} (n={})".format(lambdas, nlambdas) == "[0.40663652 0.71991334 0.35093256] (n=[463 821 401])"
        assert "{} (n={})".format(mus, nmus) == "[0.97108061 0.93512831 1.09950909] (n=[401 721 356])"
        assert "{}".format([l/m for l, m in zip(lambdas, mus)]) == "[0.41874640960555065, 0.7698551496249696, 0.3191720357461096]"
        assert "{:.3f} (n={})".format(ET, nET) == "3.294 (n=345)"
        # Assertions when assigning arriving jobs using the assignment policy
        # (i.e. function self._generate_birth_times()
        # instead of function self._generate_birth_times_from_equivalent_rates())
        #assert "{} (n={})".format(lambdas, nlambdas) == "[0.37886116 0.72236367 0.33797772] (n=[432 818 386])"
        #assert "{} (n={})".format(mus, nmus) == "[0.97778312 0.97794841 1.00719833] (n=[384 730 345])"
        #assert "{}".format([l/m for l, m in zip(lambdas, mus)]) == "[0.38746952873057483, 0.7386521181943286, 0.3355622277630897]"
        #assert "{:.3f} (n={})".format(ET, nET) == "3.615 (n=314)"

        # b) Fleming-Viot
        print("\n[test_QB] Running Fleming-Viot simulation on {} particles, K={}, T={}x...".format(nparticles, K, nmeantimes))
        est_fv = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_class_rates,
                                                   service_rates=env_queue.getServiceRates(),
                                                   buffer_size_activation=buffer_size_activation,
                                                   nmeantimes=nmeantimes,
                                                   burnin_cycles_absorption=3,
                                                   burnin_cycles_complete_all_particles=False,
                                                   policy_assign=policy_assign,
                                                   reactivate=True,
                                                   finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                   seed=seed,
                                                   plotFlag=plotFlag,
                                                   log=log)
        proba_blocking_fv, integral, expected_absorption_time, n_expected_absorption_time = est_fv.simulate(EventType.ABSORPTION)#EventType.ACTIVATION)

        # Plot the estimation curves
        df_proba_survival_and_blocking_conditional = est_fv.estimate_proba_survival_and_blocking_conditional()
        n_absorption_time_observations = df_proba_survival_and_blocking_conditional.shape[0] - 1
        plot_curve_estimates(df_proba_survival_and_blocking_conditional,
                          dict_params={
                            'birth_rates': est_fv.queue.getBirthRates(),
                            'death_rates': est_fv.queue.getDeathRates(),
                            'K': est_fv.queue.getCapacity(),
                            'nparticles': nparticles,
                            'nmeantimes': nmeantimes,
                            'maxtime_mc': est_mc.maxtime,
                            'maxtime_fv': est_fv.maxtime,
                            'buffer_size_activation': buffer_size_activation,
                            'mean_lifetime': expected_absorption_time,
                            'n_survival_curve_observations': est_fv.N,
                            'n_absorption_time_observations': n_absorption_time_observations,
                            'proba_blocking_fv': proba_blocking_fv,
                            'finalize_type': finalize_type,
                            'seed': seed
                            })

        # Check number of events
        assert est_fv.nevents >= est_fv.get_number_events_from_latest_reset(), "nevents={}, get_number={}".format(est_fv.nevents, est_fv.get_number_events_from_latest_reset())

        time_end = timer()
        print("[test_QB] Execution time: {:.1f} min".format((time_end - time_start) / 60))

        # c) Assertions
        print("[test_QB] P(K) true: {:.6f}%".format(proba_blocking_K*100))    # K=5: 11.98%
        print("[test_QB] P(K) by MC: {:.6f}% (#events={})".format(proba_blocking_mc*100, est_mc.nevents))
        print("[test_QB] P(K) estimated by FV: {:.6f}% (#events={})".format(proba_blocking_fv*100, est_fv.nevents))
        print("[test_QB] E(T) estimated by FV: {:.1f} (n={})".format(expected_absorption_time, n_expected_absorption_time))
        # Assertions for seed=1717, nservers=3, K=5, BSA=3 (both MC and FV), N=20, nmeantimes=5
        # Assertions when assigning arriving jobs directly to servers
        # (i.e. function self._generate_birth_times_from_equivalent_rates() is used
        # instead of function self._generate_birth_times())
        assert "{:.6f}% (#events={})".format(proba_blocking_mc*100, est_mc.nevents) == "11.983398% (#events=3163)"
        assert "{:.6f}% (#events={})".format(proba_blocking_fv*100, est_fv.nevents) == "4.085827% (#events=1039)"
        assert "{:.1f} (n={})".format(expected_absorption_time, n_expected_absorption_time) == "3.3 (n=20)"
        # Assertions when assigning arriving jobs using the assignment policy
        # (i.e. function self._generate_birth_times()
        # instead of function self._generate_birth_times_from_equivalent_rates())
        #assert "{:.6f}% (#events={})".format(proba_blocking_mc*100, est_mc.nevents) == "10.716641% (#events=3095)"
        #assert "{:.6f}% (#events={})".format(proba_blocking_fv*100, est_fv.nevents) == "12.549741% (#events=1653)"
        #assert "{:.1f} (n={})".format(expected_absorption_time, n_expected_absorption_time) == "5.0 (n=20)"

    #--- Test #4.1: One server with acceptance policy
    if 4.1 in tests2run:
        print("\nTest #4.1: Single server system with ACCEPTANCE policy on different JOB CLASSES")
        K = 10
        rate_birth = 0.5
        job_class_rates = [0.3, 0.8, 0.7, 0.9]
        rate_death = [1]
        acceptance_rewards = [1, 0.8, 0.3, 0.2]     # One acceptance reward for each job class
        nservers = len(rate_death)
        queue = GenericQueue(K, nservers)
        env_queue = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, acceptance_rewards)
        # Acceptance thresholds
        # - There is one threshold defined for each buffer size
        # - Their values are in the range of the job classes (0, ..., #job classes)
        # - Any job with class smaller or equal than the threshold
        # for the buffer size at the moment of the job arrival is accepted by the policy 
        acceptance_thresholds = [2]*(K+1)
        policy_accept = PolQueueTwoActionsLinearStepOnJobClasses(env_queue, acceptance_thresholds)
        queue = queues.QueueMM(rate_birth, rate_death, 1, K)

        # Simulation
        nparticles = 3
        nmeantimes = 5
        finalize_type = FinalizeType.ABSORB_CENSORED

        # a) Monte-Carlo (to estimate expected absorption cycle time)
        print("Running Monte-Carlo simulation on 1 particle and T={}x...".format(nparticles*nmeantimes))
        est_mc = EstimatorQueueBlockingFlemingViot(1, queue, job_class_rates,
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
        proba_blocking_mc, total_blocking_time, total_return_time, total_return_n = est_mc.simulate(EventType.ACTIVATION)
        expected_absorption_time, _, _ = est_mc.estimate_expected_absorption_time_from_killings()
        """
        # b) Fleming-Viot
        print("Running Fleming-Viot simulation on {} particles and T={}x...".format(nparticles, nmeantimes))
        est_fv = EstimatorQueueBlockingFlemingViot(nparticles, queue, job_class_rates,
                                                   service_rates=None,
                                                   buffer_size_activation=1,
                                                   nmeantimes=nmeantimes,
                                                   policy_accept=policy_accept,
                                                   policy_assign=policy_assign,
                                                   mean_lifetime=expected_absorption_time,
                                                   reactivate=True,
                                                   finalize_info={'type': finalize_type, 'condition': FinalizeCondition.ACTIVE},
                                                   plotFlag=True,
                                                   seed=seed, log=log)
        proba_blocking_fv_integral, proba_blocking_fv_laplacian, integral, expected_absorption_time, gamma = est_fv.simulate(EventType.ACTIVATION)
        """
        time_end = timer()
        ## 2021/03/01: 0.3 min
        print("Test #1: execution time: {:.1f} min".format((time_end - time_start) / 60))

        # c) Assertions
        print("P(K) by MC: {:.6f}%".format(proba_blocking_mc*100))
        #print("P(K) estimated by FV1: {:.6f}%".format(proba_blocking_fv_integral*100))
        #print("P(K) estimated by FV2: {:.6f}%".format(proba_blocking_fv_laplacian*100))
        assert "{:.6f}%".format(proba_blocking_mc*100) == "16.971406%"
        #assert "{:.6f}%".format(proba_blocking_fv_integral*100) == "0.095846%"
        #assert "{:.6f}%".format(proba_blocking_fv_laplacian*100) == "0.000000%"
        # Note: True P(K): 0.048852%
    #------------------------- Unit tests on the process -----------------------------#

