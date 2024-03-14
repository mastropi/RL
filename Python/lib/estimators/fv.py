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


def initialize_phi(env, envs: list=None, agent=None, t: float = 0.0, states_of_interest: set=set()):
    """
    Initializes the conditional probability Phi(t, x) of each state x of interest

    These states are the keys of the dictionary that is initialized and returned by this function.

    Arguments:
    env: Environment
        Environment which an FV particle is made of, i.e. all particles are instances of this environment.

    envs: (opt) list of environments
        List of environments used to run the FV process.
        Valid environments are of type Python.lib.environments.queues.GenericEnvQueueWithJobClasses for queue environments
        or Python.lib.environments.EnvironmentDiscrete for discrete-time/state/action environments with optional terminal states.
        default: None

    agent: (opt) Agent
        The Accept/Reject agent interacting with the environment.
        Only used when the environment is a queuing environment, in which case it should be an agent
        accepted by the get_blocking_states_or_buffer_sizes() function defined in among the queue simulator functions.
        default: None

    t: (opt) float
        Time associated to the first measurement of Phi.
        default: 0.0

    states_of_interest: (opt) set
        Set of states x on which Phi(t, x) should be estimated.
        When empty, the states of interest depend on the type of environment, and is defined as follows:
        - for GenericEnvQueueWithJobClasses environments: the set of states where blocking can occur.
        - for EnvironmentDiscrete environments: the set of terminal states, which are assumed to be the ones
        whose visit yields a non-zero reward.
        default: empty set

    Return: dict
        Dictionary, indexed by the states of interest x (e.g. [3, 1, 0]), of data frames that will be used to store
        the empirical distribution 'Phi' (Phi(t, x)) for every 't' value at which Phi(t, x) changes.
        Each entry of the dictionary is initialized with a data frame with just one row containing the
        first time of measurement of Phi and the empirical distribution of the particles at the respective state x
        (indexing the dictionary).
    """
    #---- Parse input parameters
    if states_of_interest is None or len(states_of_interest) == 0:
        # Compute the states of interest from the environment
        if isinstance(env, GenericEnvQueueWithJobClasses):
            # TODO: (2023/09/06) Try to remove the circular import that this import implies (because simulators.queues imports functions or constants from this file (e.g. BURNIN_TIME_STEPS)
            from Python.lib.simulators.queues import get_blocking_states_or_buffer_sizes
            # TODO: (2023/03/12) When using the linear step policy, limit the states of interests to those where the derivative is not zero, as we don't need to estimate the probability of the other blocking states (e.g. K-1 is of interest but K is NOT)
            states_of_interest = get_blocking_states_or_buffer_sizes(env, agent)
        elif isinstance(env, EnvironmentDiscrete):
            # For discrete-time/state/action environments, for now (2023/09/04) we consider that the states of interest are the TERMINAL states,
            # which are assumed to be the states whose visit yields a non-zero reward.
            # Note that this will give an empty set of states if the environment doesn't have any terminal states!
            states_of_interest = env.getTerminalStates()
        else:
            raise ValueError("The environment type is not valid: {}".format(type(env)) +
                             "\nValid environments are: GenericEnvQueueWithJobClasses, EnvironmentDiscrete")
    #---- Parse input parameters

    if len(states_of_interest) == 0:
        raise ValueError("The set of states of interest where the occupation probability Phi conditional to survival should be estimated is empty."
                         "\nCheck the definition of the environment where learning takes place or provide a non-empty set in parameter `states_of_interest`.")

    dict_phi = dict()
    for x in states_of_interest:
        if envs is not None:
            assert isinstance(envs, list) and len(envs) > 0, "The list parameter `envs` is not empty"
            dict_phi[x] = pd.DataFrame([[t, empirical_mean(envs, x)]], columns=['t', 'Phi'])
        else:
            dict_phi[x] = pd.DataFrame([[t, 0.0]], columns=['t', 'Phi'])

    return dict_phi


def empirical_mean(envs: list, state: Union[list, int]):
    "Computes the proportion of environments/particles at the given state"
    if isinstance(envs[0], GenericEnvQueueWithJobClasses):
        if is_scalar(state):
            return np.mean([int(x == state) for x in [env.getBufferSize() for env in envs]])
        else:
            return np.mean([int(x == state) for x in [tuple(env.getQueueState()) for env in envs]])
    else:
        return np.mean([int(x == state) for x in [env.getState() for env in envs]])


def update_phi(env, N: int, t: float, dict_phi: dict, env_state_prev, env_state_cur):
    """
    Updates the conditional probability Phi of each state of interest at the given time t, which are stored by the keys
    of the dictionary of input parameter dict_phi

    Arguments:
    env: environment object of type Python.lib.environments.queues.GenericEnvQueueWithJobClasses or Python.lib.environments.EnvironmentDiscrete
        Environment representing any of the particles.
        It is only used to decide how to treat the previous and current states given in parameters `env_state_prev` and `env_state_cur`.

    N: int
        Number of particles in the Fleming-Viot system.

    t: float
        Continuous-valued time at which the latest change of a particle's state happened.
        This value must be larger than or equal to the last time value present in dict_phi for the state of interest
        whose Phi value is potentially updated. If t is equal to the last time present in dict_phi,
        the value of Phi previously stored in the dictionary at the given t value is *replaced* with the updated Phi value.
        This is done to accommodate for the situation where the state of interest x is also a terminal state
        of the original process being modeled by the FV particle system (quite common in episodic learning tasks),
        which triggers a reset of the particle at an environment's start state the next time the particle is picked for update.
        If this start state happens to be part of the absorption set A, an immediate reactivation of the particle takes place.
        Replacing the Phi value already stored allows the estimation process to take into account the transition
        that really counts, namely from the terminal state x to the state after reactivation.

    dict_phi: dict
        Dictionary, indexed by the states of interest (derived from the environment's state),
        of data frames containing the times 't' and the empirical distribution 'Phi' at which the latter changes.
        IMPORTANT: this input parameter is updated by the function with a new row whenever the value of Phi(t, x)
        changes w.r.t. to the last stored value at state x.

    env_state_prev: (environment) State
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

    env_state_cur: (environment) State
        The current state of the particle that just changed state, represented by an *environment* state
        as described for parameter `env_state_prev`.
        The same considerations apply as those described for parameter `env_state_prev`.

    Return: dict
    The updated input dictionary which contains a new row for each state x for which the value Phi(t, x)
    changes w.r.t. the last value stored in the data frame for that state x.
    """
    # Parse parameters env_state_prev, env_state_cur
    if isinstance(env, GenericEnvQueueWithJobClasses):
        from Python.lib.environments.queues import BufferType
        if env.getBufferType() == BufferType.SINGLE:
            state_prev = env.getBufferSizeFromState(env_state_prev)
            state_cur = env.getBufferSizeFromState(env_state_cur)
        else:
            state_prev = env.getQueueStateFromState(env_state_prev)
            state_cur = env.getQueueStateFromState(env_state_cur)
    elif isinstance(env, EnvironmentDiscrete):
        state_prev = env_state_prev
        state_cur = env_state_cur
    else:
        raise ValueError("The environment type is not valid: {}".format(type(env)) +
                         "\nValid environments are: GenericEnvQueueWithJobClasses, EnvironmentDiscrete")

    # Go over the states of interest that intersect with the previous and current states
    # (in fact, these two are the only states whose Phi value can change now)
    for x in set(dict_phi.keys()).intersection({state_prev, state_cur}):
        assert dict_phi[x].shape[0] > 0, "The Phi data frame has been initialized for state = {}".format(x)
        assert t >= dict_phi[x]['t'].iloc[-1], f"The new time to insert in Phi(t,x) (t={t}) must be larger than or equal to the largest time already stored in Phi(t,x) ({dict_phi[x].iloc[-1]['t']})"
        phi_cur = dict_phi[x]['Phi'].iloc[-1]
        phi_new = empirical_mean_update(phi_cur, x, state_prev, state_cur, N)
        # if phi_new > 0:
        #    print("prev state: {} -> state: {}: New Phi: {}".format(state_prev, state_cur, phi_new))
        if not np.isclose(phi_new, phi_cur, atol=0.5 / N, rtol=0):  # Note that the absolute tolerance is smaller for larger N values.
            # Also, recall that the isclose() checks if |a - b| <= with atol + rtol*|b|,
            # and since we are interested in analyzing the absolute difference (not the relative difference)
            # between phi_new and phi_cur, we set rtol = 0.
            # Phi(t) changed at t by at least 1/N (that's why we use atol < 1/N
            # => add a new entry to the data frame containing Phi(t, x) (or replace the entry if t is equal to the latest t stored in Phi(t,x) --which may be the case for discrete-time Markov processes)
            # (o.w. it's no use to store it because we only store the times at which Phi changes)
            if t == dict_phi[x]['t'].iloc[-1]:
                dict_phi[x]['Phi'].iloc[-1] = phi_new
                    ## WARNING: The update MUST be done using dict_phi[x]['Phi'].iloc[-1] NOT dict_phi[x].iloc[-1]['Phi'] because the latter does NOT update the value of Phi!!!
                    ## (presumably because of the usual warning: "a value is trying to be set on a copy of a slice..." y la marincoche)
            else:
                dict_phi[x] = pd.concat([dict_phi[x],
                                         pd.DataFrame([[t, phi_new]], index=[dict_phi[x].shape[0]], columns=['t', 'Phi'])],
                                        axis=0)

    return dict_phi


def empirical_mean_update(mean_value: float,
                          state: Union[int, tuple],
                          state_prev: Union[int, tuple],
                          state_cur: Union[int, tuple],
                          N: int):
    """
    Updates the proportion of environments/particles at the given state based on the previous and current
    state of the particle that experienced an event last.

    Note that the particle may have experienced an event or action on it, but NO observed change of state
    because of the environment constraints, such as:
    - for queue systems, the event was an arrival event and the particle was at its full capacity.
    - for queue systems, the event was a service event and the particle was reactivated to the same state
    as it was before.
    - for labyrinths, the particle was at the border and moved in the direction of the wall.
    - etc.

    Arguments:
    mean_value: float
        The current mean value to be updated.

    state: int or tuple
        The state at which the empirical mean should be updated.

    state_prev: int or tuple
        The previous state of the particle that just "changed" state.

    state_cur: int or tuple
        The current state of the particle that just "changed" state.

    N: int
        Number of particles in the Fleming-Viot system.

    Return: float
        The updated empirical mean at the given state, following the change of state
        from `state_prev` to `state_cur`.
    """
    # Add to the current `mean_value` to be updated a number that is either:
    # - 0: when the current state is NOT `state` and the previous state was NOT `state` either
    #   OR
    #      when the current state IS `state` and the previous state WAS `state` as well.
    #   In both of these cases, there was NO change in the state of the particle whose state was just "updated".
    # - +1/N: when the current state IS `state` and the previous state was NOT `state`.
    # - -1/N: when the current state is NOT `state` and the previous state WAS `state`
    # print("prev: {}, cur: {}, s: {}".format(state_prev, state_cur, state))

    # Note that the mean value must be between 0 and 1 as it represents a probability
    return min( max(0, mean_value + (int(state_cur == state) - int(state_prev == state)) / N), 1 )


@measure_exec_time
def estimate_stationary_probabilities(dict_phi, df_proba_surv, expected_absorption_time, uniform_jump_rate=1):
    """
    Computes the stationary probability for each state of interest for the empirical distribution Phi using the Fleming-Viot estimator

    This function calls estimate_proba_stationary() for each state of interest appearing among the keys in the `dict_phi` dictionary.

    Arguments:
    dict_phi: dict of data frames
        Empirical distribution of the states of interest which are the keys of the dictionary.
        Each data frame contains the times 't' and Phi values 'Phi' containing the empirical distribution at the
        state indicated by the dictionary's key.

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
    Duple with two dictionaries indexed by the states (x) of interest with the following content:
    - the stationary probability of state x
    - the value of the integral P(T>t) * Phi(t, x)
   """
    states = sorted(list(dict_phi.keys()))  # Note that sorted() still works when the keys, i.e. the elements of the list that is being sorted, are in turn a *list* of values
    # (e.g. sorted( [(2, 2, 0), (1, 2, 3), (0, 1, 5)] ) returns [(0, 1, 5), (1, 2, 3), (2, 2, 0)]
    probas_stationary = dict()
    integrals = dict()
    for x in states:
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
                plt.title("P(T>t) (blue) and Phi(t,x) (red) and their product (green) for state x = {}\n(Integral = Area under the green curve = {:.3f})".format(x, integrals[x]))

    return probas_stationary, integrals


#@measure_exec_time
def merge_proba_survival_and_phi(df_proba_surv, df_phi, t_origin=0.0):
    """
    Merges the survival probability and the empirical distribution of the particle system at a particular
    buffer size of interest, on a common set of time values into a data frame.

    Arguments:
    df_proba_surv: pandas data frame
        Data frame containing the time 't' and the P(T>t) survival probability 'P(T>t)' given activation.

    df_phi: pandas data frame
        Data frame containing the time 't' and the Phi(t) value 'Phi' for a buffer size of interest.

    t_origin: (opt) float
        Time that should be used as origin for the times t at which Phi(t) is measured.
        This is useful to consider estimates of Phi(t) only at the times when we think that
        stationarity of the Fleming-Viot process has been reached.
        default: 0.0

    return: pandas data frame
    Data frame with the following columns:
    - 't': time at which a change in any of the input quantities happens
    - 'P(T>t)': survival probability given the process started at the stationary activation distribution of states.
    - 'Phi': empirical distribution at the buffer size of interest given the process started
    at the stationary activation distribution of states.
    """
    if t_origin > 0:
        _df_phi = df_phi.loc[df_phi['t'] >= t_origin, :]    # No need to call copy() on the filtered data frame because a copy is automatically created (tested this with a small data frame)
        if _df_phi.shape[0] > 0:
            # Shift the origin time possibly forward, to the closest time larger than or equal to the given t_origin,
            # so that the first time value in Phi is 0, as this is a requirement by the merge_values_in_time() function called below.
            t_origin = _df_phi['t'].iloc[0]
        else:
            # Revert to the original Phi data frame if no time value is larger than t_origin so that the merge continues without regard to the time origin value
            _df_phi = df_phi
            t_origin = 0.0
    else:
        _df_phi = df_phi
    # Merge the time values at which the survival probability is measured (i.e. where it changes)
    # with the time values at which the empirical distribution Phi is measured for each buffer size of interest.
    t, proba_surv_by_t, phi_by_t = merge_values_in_time(list(df_proba_surv['t']), list(df_proba_surv['P(T>t)']),
                                                        list(_df_phi['t'] - t_origin), list(_df_phi['Phi']),
                                                        unique=False)

    # -- Merged data frame, where we add the dt, used in the computation of the integral
    df_merged = pd.DataFrame(np.c_[t, proba_surv_by_t, phi_by_t, np.r_[np.diff(t), 0.0]], columns=['t', 'P(T>t)', 'Phi', 'dt'])

    if DEBUG_ESTIMATORS:
        print("Survival Probability and Empirical Distribution for a buffer size of interest:")
        print(df_merged)

    return df_merged


#@measure_exec_time
def estimate_proba_stationary(df_phi_proba_surv, expected_absorption_time, interval_size: float=1.0):
    """
    Computes the stationary probability for ONE particular state using Approximation 1 in Matt's draft for the Fleming-Viot estimation approach

    Arguments:
    df_phi_proba_surv: pandas data frame
        Data frame with the survival probability P(T>t) and the empirical distribution of a state
        of interest on which the integral that leads to the Fleming-Viot estimation of the stationary
        probability of the state is computed.
        It should contain columns 'dt', 'P(T>t)' and 'Phi', where 'dt' gives the size of each
        time interval where the product P(T>t) * Phi(t,x) is constant.

    expected_absorption_time: float
        Estimated expected absorption cycle time.

    interval_size: (opt) positive float
        Factor by which the integral of the Fleming-Viot estimator should be adjusted (i.e. multiplied by).
        See details in the documentation of compute_fv_integral(), and see the documentation of
        estimate_stationary_probabilities() on how this interval size should be set for Fleming-Viot.
        default: 1.0

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

    integral = compute_fv_integral(df_phi_proba_surv, interval_size=interval_size)

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


def compute_fv_integral(df_phi_proba_surv, reward: float=1.0, interval_size: float=1.0, discount_factor: float=1.0):
    """
    Computes the integral of the FV estimator of the stationary probability of a state of interest

    The computation is done from piecewise approximations of the two functions involved in its computation,
    the survival probability P(T>t) and the conditional occupation probability of the state of interest x
    given the process has not been killed at time t, Phi(t,x).

    The formula to compute the integral depends on whether the discount factor gamma is 1 or less than 1,
    bearing in mind that gamma values < 1 only make sense in discrete-time Markov process contexts.
    So, if gamma = 1.0:
        integral = sum_{pieces-where-P(T>t)*Phi(t,x)-is-constant-until-reaching-t_max}{ P(T>t) * r(x) * Phi(t,x) }
    if gamma < 1.0, the Markov process is assumed to be discrete-time and the formula is:
        integral = sum_{t=1}^{t_max}{ gamma^(t-1) * P(T>t) * r(x) * Phi(t,x) }
    where:
    - t_max is the maximum survival time observed for the estimation of P(T>t),
    - r(x) is the reward received when visiting state x.

    All of  the above is prior to any adjustment requested by parameter `interval_size` which ends up
    multiplying the summation if different from 1, in order to adjust for the discretization of the
    underlying continuous-time Markov process for which the Fleming-Viot estimator is originally defined.
    For more details, see the documentation of estimate_stationary_probabilities().

    Arguments:
    df_phi_proba_surv: pandas data frame
        Data frame with the survival probability P(T>t) and the empirical distribution of
        a state of interest x, Phi(t,x), on which the Fleming-Viot integral is computed.
        It should contain columns 'dt', 'P(T>t)' and 'Phi', where 'dt' gives the size of each
        time interval where the product P(T>t) * Phi(t,x) is constant.

    reward: (opt) float
        Reward received when visiting the state of interest on which the FV integral is computed
        and that weights the contribution by Phi(t,x) to the integral.
        default: 1.0

    interval_size: (opt) positive float
        Factor by which the Fleming-Viot integral should be adjusted (i.e. multiplied by) in order to take into account the interval
        size at which the originally continuous-time Markov chain is sampled.
        This is useful when Fleming-Viot is used in the context of discrete-time Markov chains, which are considered
        as the result of sampling an underlying continuous-time Markov chain (on which Fleming-Viot is actually applied,
        since Fleming-Viot is defined for continuous-time processes) at such interval sizes.
        See the documentation of estimate_stationary_probabilities() on how this interval size should be set for Fleming-Viot.
        default: 1.0

    discount_factor: (opt) float in (0, 1]
        The discount factor gamma to use as weight of each one-sized interval contributing to the integral
        in the discrete time case, where the integral is actually a summation.
        default: 1.0

    Return: float
    The integral value estimated from the area under the piecewise function (as described above) multiplied
    by `interval_size` to obtain the final integral estimate.
    """
    # Filter on the records that actually contribute to the integral
    # Note that:
    # - The values of P(T>t) are always positive (by definition of survival probability), except at the maximum survival time measured, where P(T>t) = 0,
    #   but here dt = 0 (which we filter below) because there is no next value to use for the computation of dt (which, recall, is computed as "next t" - "current t".
    # - The value of dt is 0 when several occurrences of the same t happen when observing the survival times of the process
    # (e.g. when working on a discrete-time Markov process), and at the last record inserted in the data frame (which contains the maximum survival time observed so far).
    df_phi_proba_surv_Phi_gt0 = df_phi_proba_surv.loc[(df_phi_proba_surv['Phi'] > 0) & (df_phi_proba_surv['dt'] > 0), :]
    if discount_factor == 1.0:
        integral = interval_size * reward * np.sum(df_phi_proba_surv_Phi_gt0['P(T>t)'] * df_phi_proba_surv_Phi_gt0['Phi'] * df_phi_proba_surv_Phi_gt0['dt'])
    else:
        # We need to consider each one-step-sized (i.e. dt = 1) interval separately, even if the function P(T>t)*Phi(t,x) is constant in there,
        # because we need to apply a different discount for each of those intervals.
        # Fortunately we do NOT need to explicitly iterate on all the one-step-sized intervals, because we can use the summation formula for a geometric sum
        # on the discount factor, and the computation of this geometric sum is possible because all the necessary ingredients are available in the input data frame!
        # The expression is:
        #   Integral = interval_size * 1 / (1 - gamma) * sum_i { f(i) * gamma^(t(i) - 1) * (1 - gamma^dt(i)) }
        # where i indexes every interval where f(i) remains constant, t(i) is the LOWER end of such interval,
        # and f(i) = P(T>t(i)) * Phi(t(i),x) and dt(i) is the length of such interval measured as t(i+1) - t(i).
        # Note that extreme cases make sense:
        # - when dt(i) = 0, the contribution to the sum is 0 because the factor (1 - gamma^dt(i)) becomes 0, and this makes total sense because the width of the interval is 0!
        # - when dt(i) = 1 for all i, we recover the original expression as sum_i{ f(i) * gamma^(t(i) - 1) }.
        integral = interval_size * reward * 1 / (1 - discount_factor) * \
                                    np.sum( discount_factor**(df_phi_proba_surv_Phi_gt0['t'] - 1) * (1 - discount_factor**(df_phi_proba_surv_Phi_gt0['dt'])) *
                                            df_phi_proba_surv_Phi_gt0['P(T>t)'] * df_phi_proba_surv_Phi_gt0['Phi'] )

    return integral


def estimate_expected_reward(env, probas_stationary: dict, reward=None):
    """
    Estimates the expected reward (a.k.a. long-run average reward) of the Markov Reward Process defined in the given environment,
    assuming a non-zero reward ONLY happens at the states stored in the probas_stationary dictionary.

    Arguments:
    env: EnvironmentDiscrete
        Discrete-time/state/action environment where the Markov Reward Process is defined.

    probas_stationary: dict
        Dictionary with the estimated stationary probability of the states which are assumed to yield
        non-zero rewards. These states are the dictionary keys.

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
        dict_rewards = env.getRewardsDict()
        if not set(dict_rewards.keys()).issubset(set(probas_stationary.keys())):
            raise ValueError("The states with non-zero rewards ({}) should all be present in the dictionary of estimated stationary probability ({})." \
                             .format(set(dict_rewards.keys()), set(probas_stationary.keys())))

    for s in probas_stationary.keys():
        if probas_stationary[s] > 0.0:  # Note that nan > 0 returns False (OK)
            expected_reward += (dict_rewards.get(s, 0) if reward is None else reward) * probas_stationary[s]

    return expected_reward


def estimate_survival_probability(model, t, state, action):
    "Estimates the survival probability at time t, given the Markov process started at the given state and action, using the given model"
    pass



if __name__ == "__main__":
    #import numpy as np
    #import pandas as pd
    #from matplotlib import pyplot as plt
    from matplotlib import cm
    import torch
    from torch.functional import F

    from Python.lib.utils import basic, computing
    from Python.lib.estimators.nn_models import nn_backprop
    from Python.lib.environments import gridworlds
    from Python.lib.agents.policies import probabilistic

    #--- Auxiliary functions
    def compute_input_onehot(t, x, a, t_max, nS, nA):
        """
        Computes the input to the neural network whose input is the observed absorption time t, the one-hot encoded state x, and the one-hot encoded action a
        based on the number of states nS and the number of actions nA.

        The values of x are assumed to vary between 0 and nS - 1 and the values of `a` between 0 and nA - 1.

        The value of t_max is used for scaling t to a similar scale to the one-hot encoding of x and a, i.e. to [0, 1].
        """
        assert 0 <= x < nS
        assert 0 <= a < nA
        input = np.zeros(1 + nS + nA)
        input[0] = t / t_max        # Encode the absorption time
        input[1 + x] = 1            # Encode the state
        input[1 + (nS-1) + a + 1] = 1    # Encode the action

        return input

    def compute_input(t, x, a, t_max, nS, nA):
        """
        Computes the input to the neural network whose input is the observed absorption time t, the state x as a number,
        and the one-hot encoded action a based on the number of actions nA.
        """
        assert 0 <= x < nS
        assert 0 <= a < nA
        input = np.zeros(1 + 1 + nA)
        input[0] = t / t_max               # Encode the absorption time
        input[1] = x / nS           # Encode the state
        input[1 + a + 1] = 1    # Encode the action

        return input

    def compute_estimated_probabilities(proba_surv_model, nS, nA, t_surv_max):
        dict_proba_surv_estimate = dict()
        for _x in set(np.arange(nS)).difference(absorption_set):
            dict_proba_surv_estimate[_x] = dict()
            for _a in np.arange(nA):
                _t_surv_values = np.linspace(0, t_surv_max_observed, 100)
                dict_proba_surv_estimate[_x][_a] = pd.DataFrame(columns=['t', 'p'])
                #print(f"\nComputing P(T>t) for x={_x}, a={_a}:")
                for _t_surv in _t_surv_values:
                    _input = compute_input_onehot(_t_surv, _x, _a, t_surv_max, nS, nA)
                    #_input = compute_input(_t_surv, _x, _a, t_surv_max, nS, nA)
                    _p_surv = float(torch.sigmoid(proba_surv_model(_input)))
                    #print("t = {:.1f}, P(T>t) = {:.3f}".format(_t_surv, _p_surv))
                    dict_proba_surv_estimate[_x][_a] = pd.concat([dict_proba_surv_estimate[_x][_a],
                                                                 pd.DataFrame({'t': _t_surv,
                                                                               'p': _p_surv},
                                                                    index=[len(dict_proba_surv_estimate[_x][_a])])],
                                                                 axis=0)
        return dict_proba_surv_estimate

    def plot_survival_curves(dict_proba_surv, dict_proba_surv_estimate, axes=None, n_estimates_so_far=0, title="Estimation of the survival probability P(t,x,a)"):
        if axes is None:
            axes = plt.figure().subplots(int(round(np.sqrt(len(active_set)))), int(round(np.sqrt(len(active_set)))))
        colors_rainbow = cm.get_cmap("rainbow", nA)
        # Color scheme for each action (assumed two actions)
        colormaps = [cm.get_cmap("Blues"), cm.get_cmap("Reds")]
        print(f"Interactive? {plt.isinteractive()}")
        for idx, ax in enumerate(axes.reshape(-1)):
            _x = idx + np.min(list(active_set))
            if _x < nS:
                print(f"Plotting P(T>t) for state x={_x} (sample sizes: {sample_size[_x]})")
                for _a in np.arange(nA):
                    if n_estimates_so_far == 0:
                        color = colors_rainbow(_a)
                    else:
                        color = colormaps[_a](n_estimates_so_far)
                    ax.plot(dict_proba_surv[_x][_a]['t'], dict_proba_surv[_x][_a]['p'], color=color, linewidth=0.5)
                    ax.plot(dict_proba_surv_estimate[_x][_a]['t'], dict_proba_surv_estimate[_x][_a]['p'], '--', color=color)
                ax.set_xlabel("Survival time")
                ax.set_ylabel(f"P(T>t) (x={_x})")
                ax.set_xlim((None, t_surv_max_observed))
                ax.set_ylim((0, 1))
                ax.legend([f"a = LEFT (n={sample_size[_x][0]})", "a = LEFT (model)", f"a = RIGHT (n={sample_size[_x][1]})", "a = RIGHT (model)"])
        plt.draw()
        plt.suptitle(title)
    #--- Auxiliary functions

    # Epsilon for the computation of the logit
    EPSILON = 1E-12

    # Number of states in the test environment
    nS = 21

    # Absorption set A
    absorption_set = set({0, 1, 2, 3})
    active_set = set(np.arange(nS)).difference(absorption_set.union({nS-1}))  # The terminal state should NOT be part of the active set of states

    # Environment with a specific initial state distribution
    isd = np.zeros(nS)
    for x in active_set:
        isd[x] = 1/len(active_set)
    assert np.isclose(sum(isd), 1.0)
    env = gridworlds.EnvGridworld1D(nS, initial_state_distribution=isd)
    nA = env.getNumActions()
    env.setSeed(1717)

    # Policy of the agent moving in the environment
    probas_actions = [0.8, 0.2] #[0.5, 0.5] #[0.8, 0.2]
    policy = probabilistic.PolGenericDiscrete(env, dict(), policy_default=probas_actions)

    # Model for the survival probability P(T>t) (strictly "greater than" --this has an impact on how the survival probability is estimated below)
    nn_hidden_layer_sizes = [24]
    proba_surv_model = nn_backprop(1 + nS + nA, nn_hidden_layer_sizes, 1,
                                    dict_activation_functions=dict({'hidden': [torch.nn.ReLU] * len(nn_hidden_layer_sizes)}))
    learning_rate = 0.03
    optimizer = torch.optim.Adam(proba_surv_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # Reset the dictionary that stores the absorption times for each state and action
    dict_proba_surv = dict()
    for _x in set(np.arange(nS)).difference(absorption_set):
        dict_proba_surv[_x] = dict()
        for _a in np.arange(nA):
            dict_proba_surv[_x][_a] = pd.DataFrame(np.array([[0, 1.0]]), columns=['t', 'p'])
    # Sample survival times for each state and action by simulation of the Markov process
    T = 50000 if probas_actions == [0.5, 0.5] else 10000 #50000 #10000           # Simulation time
    x = env.reset()
    t_surv_max = 10*nS       # "Maximum" survival time used to scale the values of the absorption times to make it comparable to 1 and thus have similar scales in all inputs to the neural network
    t_surv_max_observed = 0
    logit_surv_t_max_observed = -np.Inf
    #### IMPORTANT PARAMETER #####
    # THIS PARAMETER IS IMPORTANT: I tried using min_sample_size = 1 and I got an estimate of the survival function that is increasing instead of decreasing!
    min_sample_size = 50     # Minimum sample size for each state and action to consider their contribution from the last observed survival time on the loss function
    #### IMPORTANT PARAMETER #####

    #batch_size = 10     # Number of samples after which the parameters of the neural network are updated (parameter NEVER used)
    trajectory = []
    sample_size = np.zeros((nS, nA), dtype=int)
    all_loss = []
    all_n = []
    n_model_updates = 0

    plot_learning = False
    if plot_learning:
        fig = plt.figure()
        axes = fig.subplots(int(round(np.sqrt(len(active_set)))), int(round(np.sqrt(len(active_set)))))
        plt.ion()
    for t in range(T):
        assert x not in absorption_set, f"The state of the system (x={x}) must NOT be in the absorption_set ({absorption_set}) (t={t})"

        # Iterate on the environment
        a = policy.choose_action(x)

        # Store in the trajectory the state x at which action `a` was taken, the action taken and the time t it was taken at
        trajectory += [(t, x, a)]

        # Step
        x, reward, done, info = env.step(a)

        # Update the survival probability model if an absorption is observed
        if x in absorption_set:
            # Store all the absorption times for ALL state and actions visited during the trajectory before the absoprtion
            t_abs = t + 1   # Increase the time step to indicate the number of time steps it took to get to the absorption set (e.g. if takes one env.step() call to reach absorption, t = 0 before this update and we ant the absorption time to be 1)
            print(f"\nABSORPTION! t={t_abs}, x={x}")
            print(f"Trajectory = \n{trajectory}")

            loss = 0.0
            n_terms_in_loss = 0
            for triple in trajectory:
                tt, xx, aa = triple
                sample_size[xx][aa] += 1
                assert xx in dict_proba_surv.keys() and aa in dict_proba_surv[xx].keys()

                #-- Contribute with the current sample of the absorption time to the model on P(T>t)
                # 1) FIRST PIECE OF THE CONTRIBUTION TO THE LOSS: the updated estimation of the logit(P(T>t; xx, aa)) based on the new observed survival time
                # => THIS GIVES THE TARGET VALUE y OF THE ERROR (contributing to the loss)
                # Compute the observed survival time when the system starts at the currently analyzed state-action (xx,aa)
                # (this is computed from the observed absorption time and the time at which the system was at (xx,aa))
                t_surv = t_abs - tt
                assert t_surv > 0
                t_surv_max_observed = max(t_surv_max_observed, t_surv)

                survival_times = list(dict_proba_surv[xx][aa]['t'])
                idx_insert, found = basic.insort(survival_times, t_surv)
                proba_surv_t = 1 - idx_insert / len(dict_proba_surv[xx][aa])
                assert 0.0 <= proba_surv_t <= 1.0
                    ## Notes:
                    ## - idx_insert tells us the index of the smallest t among the survival times that are larger than the t_surv just inserted
                    ## (this is so because the first element of the survival time lists is always 0)
                    ## and this takes into account existing occurrences of t already present in the list, as when that happens
                    ## the "new" t is inserted at the end of all the occurrences.
                    ## - With this definition, the survival probability can never be 1
                    ## because it is estimated from the "# samples > t" where t is an OBSERVED value
                    ## => t is part of the sample and is not included in the event "# samples > t"
                    ## => the denominator of the logit function below will never be 0 (but the numerator can, hence the "bound away from zero" constant)
                    ## => this is why below, after updating the loss with this observation, we add a sample with t_surv = 0 to which we impose survival probability = 1,
                    ##      i.e. so that we get a sample of the time (t=0) where P(T>t) = 1, which the model is thus able to estimate.
                logit_surv_t = torch.tensor( [np.log( (proba_surv_t + EPSILON) / (1 - proba_surv_t + EPSILON) )] )
                logit_surv_t_max_observed = max(logit_surv_t_max_observed, logit_surv_t)
                #proba_surv_t = torch.tensor([proba_surv_t])    # To use when the output is directly the survival probability (usually not recommended, i.e. it is better to output the logit)

                # Store the survival times and survival probabilities in the data frame
                dict_proba_surv[xx][aa] = computing.compute_survival_probability(survival_times, colnames=['t', 'p'])

                # 2) SECOND PIECE OF THE CONTRIBUTION TO THE LOSS: the current estimate of the logit(P(T>t; xx, aa)) given by the model
                # => THIS GIVES THE PREDICTED VALUE \hat{y} OF THE ERROR (contributing to the loss)
                input = compute_input_onehot(t_surv, xx, aa, t_surv_max, nS, nA)
                #input = compute_input(t_surv, xx, aa, t_surv_max, nS, nA)
                assert len(input) == proba_surv_model.getNumInputs()
                #print(f"------------- Input = {input}")
                logit_estimate = proba_surv_model(input)       # The output of the model is the estimate of the logit SIMPLY because we are feeding the *logit* as the network's target (when computing the loss below)

                # 3) UPDATE THE LOSS AS LONG AS there have been enough samples of the survival time so far when the trajectory starts at (xx,aa)
                if sample_size[xx][aa] >= min_sample_size:
                    # Use a smooth version of the loss (in order to control for outliers)
                    loss += F.smooth_l1_loss(logit_estimate, logit_surv_t) #F.mse_loss(logit_estimate, logit_surv_t)
                    n_terms_in_loss += 1
                    if True:
                        print(f"--> Sample: (t={t_surv}, x={xx}, a={aa}) (n={sample_size[xx,aa]}) --> P_est = {float(torch.sigmoid(logit_estimate))}, P_obs = {float(torch.sigmoid(logit_surv_t))} --> loss = {loss/n_terms_in_loss} (n={n_terms_in_loss})")

                    # Add a sample for the survival probability = 1 at t = 0 (so that the model learns that S(0) = 1)
                    logit_surv_t = torch.tensor( [np.log( (1.0 + EPSILON) / (1 - 1.0 + EPSILON) )] )
                    logit_surv_t_max_observed = max(logit_surv_t_max_observed, logit_surv_t)
                    input = compute_input_onehot(0.0, xx, aa, t_surv_max, nS, nA)
                    logit_estimate = proba_surv_model(input)       # The output of the model is the estimate of the logit because we are feeding the network the logit as target (when computing the loss)
                    loss += F.smooth_l1_loss(logit_estimate, logit_surv_t) #F.mse_loss(logit_estimate, logit_surv_t)
                    n_terms_in_loss += 1

            # Compute the average loss
            loss /= max(1, n_terms_in_loss)
            all_loss += [float(loss)]
            all_n += [n_terms_in_loss]
            if n_terms_in_loss > 0:
                # Train the neural network model on the n_terms_in_loss samples observed
                optimizer.zero_grad()  # We can easily look at the neural network parameters by printing optimizer.param_groups
                loss.backward()
                optimizer.step()
                n_model_updates += 1
                print(f"# Model updates so far: {n_model_updates}")

                if plot_learning:
                    dict_proba_surv_estimate = compute_estimated_probabilities(proba_surv_model, nS, nA, t_surv_max)
                    plot_survival_curves(dict_proba_surv, dict_proba_surv_estimate, axes=axes, n_estimates_so_far=n_model_updates)

            # Reset the environment and the trajectory history
            x = env.reset()
            trajectory = []
        elif x in env.getTerminalStates():
            x = env.reset()
            trajectory = []
    if plot_learning:
        plt.ioff()
    print(f"Maximum observed logit value = {logit_surv_t_max_observed}")

    plt.figure()
    ax = plt.gca()
    ax.plot(all_loss, '.-', color="red")
    ax.set_xlabel("Learning step")
    ax.set_ylabel("Loss (Smoothed L2 error)")
    ax.set_ylim((0, None))
    ax2 = ax.twinx()
    ax2.plot(all_n, color="blue", linewidth=0.2)
    ax2.set_ylabel("Batch size")
    ax2.set_ylim((0, None))

    # Compute the estimated survival probability for each state and action
    dict_proba_surv_estimate = compute_estimated_probabilities(proba_surv_model, nS, nA, t_surv_max)

    # Plot the survival probabilities
    plot_survival_curves(dict_proba_surv, dict_proba_surv_estimate, title=f"Estimation of the survival probability P(t,x,a) for all states and actions in 1D gridworld (probas_actions = {probas_actions})")

    # Plot of the loss value and the batch size
    plt.figure()
    colormap = cm.get_cmap("Blues", lut=len(all_n))
    for idx in range(len(all_n)):
        plt.plot(all_n[idx], float(all_loss[idx]), '.', markersize=5, color=colormap(idx))
    plt.gca().set_xlabel("Batch size")
    plt.gca().set_ylabel("Loss")
    plt.gca().set_title("We see that the loss tends to be larger as the batch size decreases\n(specially towards the end of learning --more intense color)")
