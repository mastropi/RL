# -*- coding: utf-8 -*-
"""
Created on Mon Sep 4 16:05:33 2023

@author: Daniel Mastropietro
@description: Functions for Fleming-Viot-based estimators for both discrete-time and continuous-time contexts.
Ex: estimator of the queue blocking probability, estimator of the average reward in classical RL environments (e.g. labyrinth), etc.
They are expected to be applicable to:
- queue environments
- discrete-time/state/action environments
"""

from enum import Enum, unique
from typing import Union
import tracemalloc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Python.lib.environments import EnvironmentDiscrete
from Python.lib.environments.queues import GenericEnvQueueWithJobClasses

from Python.lib.estimators import DEBUG_ESTIMATORS

from Python.lib.utils.basic import is_scalar, measure_exec_time, merge_values_in_time

@unique
class SurvivalProbabilityEstimation(Enum):
    FROM_N_PARTICLES = 1
    FROM_M_CYCLES = 2


def initialize_phi(envs, agent, t: float = 0.0, states_of_interest: set=set()):
    """
    Initializes the conditional probability Phi(t, x) of each state or buffer size x of interest,
    i.e. those x values where blocking can occur.
    These states or buffer sizes are the keys of the dictionary that is initialized and returned by this function.

    Arguments:
    envs: list
        List of environments used to run the FV process.
        Valid environments are of type Python.lib.environments.queues.GenericEnvQueueWithJobClasses for queue environments
        or Python.lib.environments.EnvironmentDiscrete for discrete-time/state/action environments with optional terminal states.

    agent: Agent
        The Accept/Reject agent interacting with the environment.
        It should be an agent accepted by the get_blocking_states_or_buffer_sizes() function.

    t: (opt) float
        Time associated to the first measurement of Phi.
        default: 0.0

    states_of_interest: (opt) set
        Set of states x on which Phi(x) should be estimated.
        When empty, the states of interest depend on the type of environment, and is defined as follows:
        - for GenericEnvQueueWithJobClasses environments: the set of states where blocking can occur.
        - for EnvironmentDiscrete environments: the set of terminal states.
        default: empty set

    Return: dict
        Dictionary, indexed by the states of interest x (e.g. [3, 1, 0]), of data frames that will be used to store
        the empirical distribution 'Phi' (Phi(t, x)) for every 't' value at which Phi(t, x) changes.
        Each entry of the dictionary is initialized with a data frame with just one row containing the
        first time of measurement of Phi and the empirical distribution of the particles at the respective state x
        (indexing the dictionary).
    """
    assert envs is not None and isinstance(envs, list) and len(envs) > 0, "The list parameter `envs` is not empty"

    if len(states_of_interest) > 0:
        states_or_buffer_sizes_of_interest = states_of_interest
    elif isinstance(envs[0], GenericEnvQueueWithJobClasses):
        # TODO: (2023/09/06) Try to remove the circular import that this import implies (because simulators.queues imports functions or constants from this file (e.g. BURNIN_TIME_STEPS)
        from Python.lib.simulators.queues import get_blocking_states_or_buffer_sizes
        # TODO: (2023/03/12) When using the linear step policy, limit the states of interests to those where the derivative is not zero, as we don't need to estimate the probability of the other blocking states (e.g. K-1 is of interest but K is NOT)
        states_or_buffer_sizes_of_interest = get_blocking_states_or_buffer_sizes(envs[0], agent)
    elif isinstance(envs[0], EnvironmentDiscrete):
        # For discrete-time/state/action environments, for now (2023/09/04) we consider that the states of interest are the TERMINAL states
        # Note that this will give an empty set of states if the environment doesn't have any terminal states!
        states_or_buffer_sizes_of_interest = envs[0].getTerminalStates()
    else:
        raise ValueError("The environment type is not valid: {}".format(type(envs[0])) +
                         "\nValid environments are: GenericEnvQueueWithJobClasses, EnvironmentDiscrete")

    if len(states_or_buffer_sizes_of_interest) == 0:
        raise ValueError("The set of states of interest where the occupation probability Phi conditional to survival should be estimated is empty."
                         "\nCheck the definition of the environment where learning takes place or provide a non-empty set in parameter `states_of_interest`.")

    dict_phi = dict()
    for x in states_or_buffer_sizes_of_interest:
        dict_phi[x] = pd.DataFrame([[t, empirical_mean(envs, x)]], columns=['t', 'Phi'])

    return dict_phi


def empirical_mean(envs: list, state_or_buffer_size: Union[list, int]):
    "Compute the proportion of environments/particles at the given state or buffer size"
    if isinstance(envs[0], GenericEnvQueueWithJobClasses):
        if is_scalar(state_or_buffer_size):
            return np.mean([int(x == state_or_buffer_size) for x in [env.getBufferSize() for env in envs]])
        else:
            return np.mean([int(x == state_or_buffer_size) for x in [tuple(env.getQueueState()) for env in envs]])
    else:
        return np.mean([int(x == state_or_buffer_size) for x in [env.getState() for env in envs]])


def update_phi(env, N: int, t: float, dict_phi: dict, state_prev, state_cur):
    """
    Updates the conditional probability Phi of each state of interest, which are stored by the keys
    of the dictionary of input parameter dict_phi

    Arguments:
    env: environment object of type Python.lib.environments.queues.GenericEnvQueueWithJobClasses or Python.lib.environments.EnvironmentDiscrete
        Environment representing any of the particles.

    N: int
        Number of particles in the Fleming-Viot system.

    t: float
        Continuous-valued time at which the latest change of a particle's state happened.

    dict_phi: dict
        Dictionary, indexed by the states or buffer sizes of interest, of data frames containing the times 't' and
        the empirical distribution 'Phi' at which the latter changes.
        IMPORTANT: this input parameter is updated by the function with a new row whenever the value of Phi(t, x)
        changes w.r.t. to the last stored value at the state or buffer size s.

    state_prev: State
        The previous state of the particle that just changed state given as a valid state of the `env` environment.
        The value of this depends on the environment type.
        - For `GenericEnvQueueWithJobClasses`, this is normally a tuple containing (queue_state, job_class)
        (and NOT simply the queue state, see e.g. the classes defined in environments/queues.py).
        The important thing is that the actual quantity that should be evaluated when updating Phi(t, x)
        is obtained by one of the following methods assumed defined in `env`:
            - getBufferSizeFromState(state) in 1D problems, which returns the single buffer size of the system.
            - getQueueStateFromState(state) in multidimensional problems, which returns the occupancy level of
            each job class in the queue system.
        - For `EnvironmentDiscrete`, this is normally an integer indexing the states of the environment.

    state: State
        The current state of the particle that just changed state.
        The same considerations apply as those described for parameter `state_prev`.

    Return: dict
    The updated input dictionary which contains a new row for each state or buffer size for which the value Phi(t, x)
    changes w.r.t. the last value stored in the data frame for that state or buffer size.
    """
    # Parse parameters state_prev, state_cur
    if isinstance(env, GenericEnvQueueWithJobClasses):
        from Python.lib.environments.queues import BufferType
        if env.getBufferType() == BufferType.SINGLE:
            state_or_buffer_size_prev = env.getBufferSizeFromState(state_prev)
            state_or_buffer_size_cur = env.getBufferSizeFromState(state_cur)
        else:
            state_or_buffer_size_prev = env.getQueueStateFromState(state_prev)
            state_or_buffer_size_cur = env.getQueueStateFromState(state_cur)
    elif isinstance(env, EnvironmentDiscrete):
        state_or_buffer_size_prev = state_prev
        state_or_buffer_size_cur = state_cur
    else:
        raise ValueError("The environment type is not valid: {}".format(type(env)) +
                         "\nValid environments are: GenericEnvQueueWithJobClasses, EnvironmentDiscrete")

    # Go over the states or buffer sizes of interest that intersect with the previous and current states or buffer sizes
    # (in fact, these two are the only states or buffer sizes whose Phi value can change now)
    for x in set(dict_phi.keys()).intersection({state_or_buffer_size_prev, state_or_buffer_size_cur}):
        assert dict_phi[x].shape[0] > 0, "The Phi data frame has been initialized for state or buffer size = {}".format(x)
        phi_cur = dict_phi[x]['Phi'].iloc[-1]
        phi_new = empirical_mean_update(phi_cur, x, state_or_buffer_size_prev, state_or_buffer_size_cur, N)
        # if phi_new > 0:
        #    print("prev state: {} -> state: {}: New Phi: {}".format(state_or_buffer_size_prev, state_or_buffer_size_cur, phi_new))
        if not np.isclose(phi_new, phi_cur, atol=0.5 / N, rtol=0):  # Note that the absolute tolerance is smaller for larger N values.
            # Also, recall that the isclose() checks if |a - b| <= with atol + rtol*|b|,
            # and since we are interested in analyzing the absolute difference (not the relative difference)
            # between phi_new and phi_cur, we set rtol = 0.
            # Phi(t) changed at t by at least 1/N (that's why we use atol < 1/N
            # => add a new entry to the data frame containing Phi(t, x)
            # (o.w. it's no use to store it because we only store the times at which Phi changes)
            dict_phi[x] = pd.concat([dict_phi[x], pd.DataFrame({'t': [t], 'Phi': [phi_new]})], axis=0)

    return dict_phi


def empirical_mean_update(mean_value: float,
                          state_or_buffer_size: Union[int, tuple],
                          state_or_buffer_size_prev: Union[int, tuple],
                          state_or_buffer_size_cur: Union[int, tuple],
                          N: int):
    """
    Updates the proportion of environments/particles at the given state or buffer size based on the previous and current
    state or buffer size of the particle that experienced an event last.

    Note that the particle may have experienced an event or action on it, but NO change of state observed because of
    environment constraints, such as:
    - for queue systems, the event was an arrival event and the particle was at its full capacity.
    - for queue systems, the event was a service event and the particle was reactivated to the same state or buffer size
    as it was before.
    - for labyrinths, the particle was at the border and moved in the direction of the wall.
    - etc.

    Arguments:
    mean_value: float
        The current mean value to be updated.

    state_or_buffer_size: int or tuple
        The state or buffer size at which the empirical mean should be updated.

    state_or_buffer_size_prev: int or tuple
        The previous state or buffer size of the particle that just "changed" state.

    state_or_buffer_size_cur: int or tuple
        The current state or buffer size of the particle that just "changed" state.

    N: int
        Number of particles in the Fleming-Viot system.

    Return: float
        The updated empirical mean at the given state or buffer size, following the change of state or buffer size
        from `state_or_buffer_size_prev` to `state_or_buffer_size_cur`.
    """
    # Add to the current `mean_value` to be updated a number that is either:
    # - 0: when the current state or buffer size is NOT state_or_buffer_size and the previous state or buffer size
    # was NOT state_or_buffer_size either
    #   OR
    #      when the current state or buffer size IS state_or_buffer_size and the previous state or buffer size
    # WAS state_or_buffer_size as well.
    #   In both of these cases, there was NO change in the state or buffer size of the particle whose state was just "updated".
    # - +1/N: when the current state or buffer size IS state_or_buffer_size and the previous state or buffer size was NOT state_or_buffer_size.
    # - -1/N: when the current state or buffer size is NOT state_or_buffer_size and the previous state or buffer size WAS state_or_buffer_size
    # print("prev: {}, cur: {}, s: {}".format(state_or_buffer_size_prev, state_or_buffer_size_cur, state_or_buffer_size))

    # Note that the mean value must be between 0 and 1 as it represents a probability
    return min( max(0, mean_value + (int(state_or_buffer_size_cur == state_or_buffer_size) - int(state_or_buffer_size_prev == state_or_buffer_size)) / N), 1 )


@measure_exec_time
def estimate_stationary_probabilities(dict_phi, df_proba_surv, expected_absorption_time, uniform_jump_rate=1):
    """
    Estimates the stationary probability for each buffer size of interest in phi using the Fleming-Viot estimator

    Arguments:
    dict_phi: dict of data frames
        Empirical distribution of the states or buffer sizes of interest which are the keys of the dictionary.
        Each data frame contains the times 't' and Phi values 'Phi' containing the empirical distribution at the
        buffer size indicated by the dictionary's key.

    df_proba_surv: pandas data frame
        Data frame with at least the following two columns:
        - 't': times at which the survival probability is estimated
        - 'P(T>t)': the survival probability estimate for the corresponding 't' value given the process
        started at the stationary activation distribution of states.

    expected_absorption_time: float
        Estimated expected absorption cycle time, i.e. the expected time the queue system takes in a
        reabsorption cycle when starting at the stationary absorption distribution of states.

    uniform_jump_rate: (opt) positive float
        Uniform jump rate used to discretize the underlying continuous-time Markov process represented by the
        Fleming-Viot system.
        This must be given when the Fleming-Viot estimator is used in the context of discrete-time Markov chains,
        in which case the N-particle continuous-time Markov process associated to the Fleming-Viot system is discretized
        at intervals of size 1/uniform_jump_rate. Typically (for simplicity) this uniform jump rate is equal to the
        number of particles in the Fleming-Viot system, N, which allows us to change the discrete-time Markov chain at
        integer steps by changing ONE particle at a time, which in turn represents the uniformization of the original
        continuous-time Fleming-Viot process for which the jump rate of the underlying continuous-time Markov process
        (governing the dynamics of each particle) is considered to be equal to 1 (i.e. one event per unit time
        --which is a DISCRETE time 0, 1, 2, ...). This is the jump rate to go from one state to ONE particular
        other state where the Markov chain can go (in the case of the labyrinth with 4 actions per state,
        it is the jump rate to go from any state to any of the other 4 adjacent states). Therefore,
        the underlying continuous-time Markov process can be uniformized by sampling at 1/(#adjacent-states * jump-rate-to-each-state),
        i.e. by using a uniformization jump rate equal to (#adjacent-states * jump-rate-to-each-state).
        Thus, in the same labyrinth example, the uniformization jump rate would be equal 4*1 = 4, which
        is the jump rate of jumping to ANY other reachable state of the Markov process or, in other words
        the expected number of jumps of the Markov process in one unit time),
        This makes the jump rate of the Fleming-Viot process be equal to N * (uniformization-rate)
        (i.e. N*(uniformization-rate) expected events per unit time, equivalent to one event for a time interval
        of length 1/(N*(uniformization-rate)). In the same labyrinth case, the jump rate of discrete-time
        Fleming-Viot process would be 4*N.
        default: 1

    Return: tuple of dict
    Duple with two dictionaries indexed by the states or buffer sizes (x) of interest with the following content:
    - the stationary probability of the state or buffer size x
    - the value of the integral P(T>t) * Phi(t, x)
   """
    states_or_buffer_sizes_of_interest = sorted(list(dict_phi.keys()))  # Note that sorted() still works when the keys, i.e. the elements of the list that is being sorted, are in turn a *list* of values
    # (e.g. sorted( [(2, 2, 0), (1, 2, 3), (0, 1, 5)] ) returns [(0, 1, 5), (1, 2, 3), (2, 2, 0)]
    probas_stationary = dict()
    integrals = dict()
    for x in states_or_buffer_sizes_of_interest:
        if dict_phi[x].shape[0] == 1 and dict_phi[x]['Phi'].iloc[-1] == 0.0:
            # State or buffer size x was never observed during the simulation
            probas_stationary[x] = 0.0
            integrals[x] = 0.0
        else:
            # Merge the times where (T>t) and Phi(t) are measured
            df_phi_proba_surv = merge_proba_survival_and_phi(df_proba_surv, dict_phi[x])

            # Stationary probability for each buffer size of interest
            probas_stationary[x], integrals[x] = estimate_proba_stationary(df_phi_proba_surv, expected_absorption_time, interval_size=1/uniform_jump_rate)

            if DEBUG_ESTIMATORS:
                plt.figure()
                ax = plt.gca()
                ax.step(df_phi_proba_surv['t'], df_phi_proba_surv['P(T>t)'], color="blue", where='post')
                ax2 = ax.twinx()
                ax2.step(df_phi_proba_surv['t'], df_phi_proba_surv['Phi'], color="red", where='post')
                ax2.step(df_phi_proba_surv['t'], df_phi_proba_surv['Phi'] * df_phi_proba_surv['P(T>t)'], color="green", where='post')
                plt.title("P(T>t) (blue) and Phi(t,x) (red) and their product (green) for state or buffer size x = {}\n(Integral = Area under the green curve = {:.3f})".format(x, integrals[x]))

    return probas_stationary, integrals


# @measure_exec_time
def merge_proba_survival_and_phi(df_proba_surv, df_phi):
    """
    Merges the survival probability and the empirical distribution of the particle system at a particular
    buffer size of interest, on a common set of time values into a data frame.

    Arguments:
    df_proba_surv: pandas data frame
        Data frame containing the time 't' and the P(T>t) survival probability 'P(T>t)' given activation.

    df_phi: pandas data frame
        Data frame containing the time 't' and the Phi(t) value 'Phi' for a buffer size of interest.

    return: pandas data frame
    Data frame with the following columns:
    - 't': time at which a change in any of the input quantities happens
    - 'P(T>t)': survival probability given the process started at the stationary activation distribution of states.
    - 'Phi': empirical distribution at the buffer size of interest given the process started
    at the stationary activation distribution of states.
    """
    # Merge the time values at which the survival probability is measured (i.e. where it changes)
    # with the time values at which the empirical distribution Phi is measured for each buffer size of interest.
    t, proba_surv_by_t, phi_by_t = merge_values_in_time(list(df_proba_surv['t']), list(df_proba_surv['P(T>t)']),
                                                        list(df_phi['t']), list(df_phi['Phi']),
                                                        unique=False)

    # -- Merged data frame, where we add the dt, used in the computation of the integral
    df_merged = pd.DataFrame(np.c_[t, proba_surv_by_t, phi_by_t, np.r_[np.diff(t), 0.0]], columns=['t', 'P(T>t)', 'Phi', 'dt'])

    if DEBUG_ESTIMATORS:
        print("Survival Probability and Empirical Distribution for a buffer size of interest:")
        print(df_merged)

    return df_merged


# @measure_exec_time
def estimate_proba_stationary(df_phi_proba_surv, expected_absorption_time, interval_size=1):
    """
    Computes the stationary probability for states of interest via Approximation 1 in Matt's draft for the Fleming-Viot estimation approach

    Arguments:
    df_phi_proba_surv: pandas data frame
        Data frame with the survival probability P(T>t) and the empirical distribution for each state
        of interest on which the integral that leads to the Fleming-Viot estimation of the stationary
        probability of the state is computed.

    expected_absorption_time: float
        Estimated expected absorption cycle time.

    interval_size: (opt) positive float
        Factor by which the integral of the Fleming-Viot estimator should be adjusted (i.e. multiplied by) in order to
        take into account the interval size at which the originally continuous-time Markov chain is sampled.
        This is useful when Fleming-Viot is used in the context of discrete-time Markov chains, which are considered
        as the result of sampling an underlying continuous-time Markov chain (on which Fleming-Viot is actually applied,
        since Fleming-Viot is defined for continuous-time processes) at such interval sizes.
        See the documentation for estimate_stationary_probabilities() on how this interval size should be set for Fleming-Viot.
        default: 1

    Return: tuple
    Duple with the following content:
    - the estimated stationary probability
    - the value of the integral P(T>t)*Phi(t)
    """
    if tracemalloc.is_tracing():
        mem_usage = tracemalloc.get_traced_memory()
        print("[MEM] estimate_proba_stationary: Memory used so far: current={:.3f} MB, peak={:.3f} MB".format(
            mem_usage[0] / 1024 / 1024, mem_usage[1] / 1024 / 1024))
        mem_snapshot_1 = tracemalloc.take_snapshot()

    if expected_absorption_time <= 0.0 or np.isnan(expected_absorption_time) or expected_absorption_time is None:
        raise ValueError("The expected absorption time must be a positive float ({})".format(expected_absorption_time))

    # Integrate => Multiply the survival density function, the empirical distribution Phi, delta(t) and SUM
    if DEBUG_ESTIMATORS:
        max_rows = pd.get_option('display.max_rows')
        pd.set_option('display.max_rows', None)
        print("Data for integral:\n{}".format(df_phi_proba_surv))
        pd.set_option('display.max_rows', max_rows)

    df_phi_proba_surv_Phi_gt0 = df_phi_proba_surv.loc[df_phi_proba_surv['Phi'] > 0,]
    integral = interval_size * np.sum(df_phi_proba_surv_Phi_gt0['P(T>t)'] * df_phi_proba_surv_Phi_gt0['Phi'] * df_phi_proba_surv_Phi_gt0['dt'])

    if DEBUG_ESTIMATORS:
        print("integral = {:.3f}, E(T) = {:.3f}".format(integral, expected_absorption_time))

    proba_stationary = integral / expected_absorption_time

    if tracemalloc.is_tracing():
        mem_snapshot_2 = tracemalloc.take_snapshot()
        mem_stats_diff = mem_snapshot_2.compare_to(mem_snapshot_1, key_type='lineno')  # Possible key_type's are 'filename', 'lineno', 'traceback'
        print("[MEM] estimate_proba_stationary: Top difference in memory usage:")
        for stat in mem_stats_diff[:10]:
            # if stat.size / stat.count > 1E6:   # To print the events with largest memory consumption for EACH of their occurrence
            print(stat)

    return proba_stationary, integral


def estimate_expected_reward(env, probas_stationary: dict, reward=None):
    """
    Estimates the expected reward (a.k.a. long-run average reward) of the Markov Reward Process defined in the given environment,
    assuming a non-zero reward ONLY happens at the states stored in the probas_stationary dictionary.

    Arguments:
    env: EnvironmentDiscrete
        Discrete-time/state/action environment where the Markov Reward Process is defined.

    probas_stationary: dict
        Dictionary with the estimated stationary probability of the states or buffer sizes which are assumed to yield
        non-zero rewards. These states or buffer sizes are the dictionary keys.

    reward: (opt) float
        When given, this value is used as the constant reward associated to a state present in the dictionary containing
        non-zero stationary probabilities, instead of the rewards stored in the environment.
        This could be useful when we are interested in computing a probability rather than the expected reward,
        and avoid having to define a reward landscape in the environment just to tell that the non-zero rewards are constant.

    Return: float
    Estimated expected reward a.k.a. long-run average reward.
    """
    expected_reward = 0.0

    if reward is None:
        assert isinstance(env, EnvironmentDiscrete)
        dict_terminal_rewards = env.getTerminalRewardsDict()
        if not set(dict_terminal_rewards.keys()).issubset(set(probas_stationary.keys())):
            raise ValueError("The terminal states with non-zero rewards ({}) should all be present in the dictionary of estimated stationary probability ({})." \
                             .format(set(dict_terminal_rewards.keys()), set(probas_stationary.keys())))

    for s in probas_stationary.keys():
        if probas_stationary[s] > 0.0:  # Note that nan > 0 returns False (OK)
            expected_reward += (dict_terminal_rewards[s] if reward is None else reward) * probas_stationary[s]

    return expected_reward
