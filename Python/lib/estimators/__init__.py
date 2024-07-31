# -*- coding: utf-8 -*-
"""
Created on TUe Sep 5 14:19:12 2023

@author: Daniel Mastropietro
@description: Definition of functions normally used in estimations. Ex: estimate_expected_cycle_time()
"""

import warnings

import numpy as np

from Python.lib.utils.computing import compute_number_of_burnin_cycles_from_burnin_time

DEBUG_ESTIMATORS = False


def estimate_expected_cycle_time(n_cycles: int, time_end_last_cycle: float,
                                 cycle_times: list=None, burnin_time: float=0.0,
                                 min_num_cycles_for_expectations: int=1):
    """
    Estimates the expected cycle time from observed cycles in a simulation.

    The function is written so that the expected cycle time can be estimated from very little data, namely the
    number of observed cycles and the time of the end of the last cycle.

    This calculation is enough unless a burn-in period is required for the estimation, in which case more information
    is needed than simply the above two pieces. In that case, the whole list of observed cycle times observed is
    required so that a number of initial burn-in cycles can be excluded from the calculation.

    If the number of cycles observed after the burn-in period is smaller than the value of parameter
    `min_num_cycles_for_expectations`, the value of the estimated expectation is set to NaN.

    Arguments:
    n_cycles: int
        Number of cycles observed.

    time_end_last_cycle: non-negative float
        Continuous time at which the last cycle ends.

    cycle_times: (opt) list
        List containing the observed cycle times.
        default: None

    burnin_time: (opt) float
        Continuous time to allow as burn-in at the beginning of the simulation.
        default: 0.0

    min_num_cycles_for_expectations: (opt) int
        Minimum number of observed cycles that are considered enough to obtain a reliable estimation of the expected
        cycle time.
        default: 1

    Return: Tuple
    Duple containing the following two elements:
    - expected_cycle_time: the estimated expected cycle time based on the cycles observed after the burn-in period.
    This value is set to NaN if the resulting number of cycles is smaller than the value of parameter
    `min_num_cycles_for_expectations`.
    - n_cycles_after_burnin: the number of cycles left after the initial burn-in period.
    """
    assert time_end_last_cycle >= 0.0   # time_end_last_cycle could be 0... namely when no cycle is observed.
                                        # In any case, a value of 0 (actually a value that is smaller than burnin_time)
                                        # does not generate problems in the calculation of the expected cycle time below,
                                        # because its value is lower bounded by 0.
    assert cycle_times is None and burnin_time == 0.0 or \
           cycle_times is not None and len(cycle_times) == n_cycles
    if burnin_time == 0:
        burnin_cycles = 0
        time_end_burnin_cycles = 0.0
        n_cycles_after_burnin = n_cycles
    else:
        burnin_cycles = compute_number_of_burnin_cycles_from_burnin_time(cycle_times, burnin_time)
        time_end_burnin_cycles = np.sum(cycle_times[:burnin_cycles])
        n_cycles_after_burnin = len(cycle_times) - burnin_cycles
    if n_cycles_after_burnin < min_num_cycles_for_expectations:
        warnings.warn("The number of calculated burn-in cycles ({}) is too large,".format(burnin_cycles) +
                      " as the number of observed cycle times left to estimate the expected cycle time ({}) is less than the minimum allowed ({})." \
                      .format(len(cycle_times), min_num_cycles_for_expectations) +
                      "The estimated expected cycle time will be set to NaN.")
        return np.nan, n_cycles_after_burnin
    if False:
        print("Expected cycle time estimated on {} cycles out of available {}:\nALL cycle times: {}\nCUM cycle times: {} (CUM)\nburnin_time = {:.3f}\nburnin_cycles = {}" \
              .format(n_cycles - burnin_cycles, len(cycle_times), cycle_times,
                      list(np.cumsum(cycle_times)), burnin_time, burnin_cycles))

    expected_cycle_time = max(0, time_end_last_cycle - time_end_burnin_cycles) / max(1, (n_cycles - burnin_cycles))  # max(1, ...) to avoid division by 0 when burnin_cycles coincides with the number of cycles observed
    assert expected_cycle_time > 0.0

    return expected_cycle_time, n_cycles_after_burnin


def estimate_expected_stopping_time_in_cycle(stopping_times, cycle_times, burnin_time: float=0.0, min_num_cycles_for_expectations: int=1):
    """
    Computes the estimated expected stopping time based on stopping times observed within a cycle

    Arguments:
    stopping_times: list
        List containing the observed stopping times on which the expectation is based.

    cycle_times: list
        List containing the observed cycle times within which the stopping times are observed.
        Each stopping time in `stopping_times` should be <= the corresponding cycle time in `cycle_times`
        stored at the same index, except for the last stopping time which may not have any corresponding cycle time,
        as the latter may not have been measured because the simulation stopped before the cycle completed.
        The length of the list should be either equal to the length of `stopping_times` or one less.

    burnin_time: (opt) float
        Continuous time to allow as burn-in at the beginning of the simulation.
        default: 0.0

    min_num_cycles_for_expectations: (opt) int
        Minimum number of observed cycles that are considered enough to obtain a reliable estimation of the expected
        cycle time.
        default: 1

    Return: Tuple
    Duple containing the following two elements:
    - expected_stopping_time: the estimated expected stopping time based on the stopping times observed after the
    burn-in period. This value is set to NaN if the resulting number of stopping times is smaller than the value of
    parameter `min_num_cycles_for_expectations`.
    - n_stopping_times_after_burnin: the number of stopping times left after the initial burn-in period.
    """
    assert len(stopping_times) == len(cycle_times) or len(stopping_times) == len(cycle_times) + 1, \
        "The number of observed stopping times ({}) is either equal to the number of reabsorption times ({}) or one more" \
            .format(len(stopping_times), len(cycle_times))

    burnin_cycles = compute_number_of_burnin_cycles_from_burnin_time(cycle_times, burnin_time)
    n_stopping_times_after_burnin = len(stopping_times) - burnin_cycles
    if n_stopping_times_after_burnin < min_num_cycles_for_expectations:
        warnings.warn("The number of observed stopping times left after the burn-in period of {} cycles to estimate the expected stopping time ({}) is smaller than the minimum allowed ({})." \
                      .format(burnin_cycles, n_stopping_times_after_burnin, min_num_cycles_for_expectations) +
                      "The estimated expected stopping time will be set to NaN.")
        expected_stopping_time = np.nan
    else:
        expected_stopping_time = np.mean(stopping_times[burnin_cycles:])
    if False:
        print("Expected stopping time estimated on {} cycles out of available {} (burnin_time = {:.3f} burnin_cycles = {})" \
              .format(len(stopping_times[burnin_cycles:]), len(stopping_times), burnin_time, burnin_cycles))

    return expected_stopping_time, n_stopping_times_after_burnin


def estimate_state_distribution_based_on_observed_times(dict_states_and_times, burnin_time: float=0.0):
    """
    Computes the state distribution based on times observed for each state which are compared to a burn-in time to qualify for counting

    This function is typically used to estimate the stationary exit state distribution from the absorption set A in Fleming-Viot simulation.

    Arguments:
    dict_states_and_times: dict
        Dictionary where keys are states and values are observed times associated to those states stored in a list.
        Typically these times are absolute exit times from the absorption set A used in the Fleming-Viot estimation process.

    burnin_time: (opt) float
        Continuous time to allow as burn-in at the beginning of the simulation.
        Only times in `dict_states_and_times` that are larger than or equal to `burnin_time` are considered when computing the state distribution.
        default: 0.0

    Return: tuple
    A tuple with the following two elements:
    - Dictionary containing the empirical distribution of the states given in input parameter `dict_states_and_times` from contributions of times observed
    for each state that are larger than or equal to the burn-in time.
    - the sample size behind the estimation of the state distribution.
    """
    dict_dist_states = dict.fromkeys(dict_states_and_times.keys())

    # Frequency of each state from state observed times larger than or equal to the burn-in time (if the given burn-in time > 0)
    total_freq = 0
    for state, times in dict_states_and_times.items():
        if burnin_time == 0:
            # Avoid filtering the times list when the burn-in time is 0
            dict_dist_states[state] = len(times)
        else:
            dict_dist_states[state] = np.sum(np.array(times) >= burnin_time)
        total_freq += dict_dist_states[state]

    # Compute the distribution (relative frequency of each state)
    for state in dict_dist_states:
        dict_dist_states[state] = dict_dist_states[state] / total_freq
    # Check sum-to-1
    assert np.isclose(np.sum(list(dict_dist_states.values())), 1.0)

    return dict_dist_states, total_freq


if __name__ == "__main__":
    # Test of estimate_state_distribution_based_on_observed_times() which is used in the process to estimate the stationary exit state distribution for the FV process
    # (see run_simulation_fv() in simulators/queues.py)
    dict_states_and_times = dict({(0, 2): [0.8, 1.1, 1.9, 2.8],         # Some times are smaller and some times are at least equal to the burn-in time
                                  (1, 2): [0.5, 0.7, 0.9],              # None of the times is at least equal to the burn-in time
                                  (2, 2): [],                           # Empty list of times
                                  (2, 0): [1.5, 1.9, 13.4, 15.8, 0.7]   # One value is exactly equal to the burn-in and values are NOT sorted (although this will usually not be the case)
                                  })
    dist, sample_size = estimate_state_distribution_based_on_observed_times(dict_states_and_times, burnin_time=1.5)
    assert sample_size == 6
    assert sorted(dist.keys()) == sorted(dict_states_and_times.keys())
    assert np.allclose(np.array([dist[k] for k in sorted(dist.keys())]), np.array([0.33333333, 0.0, 0.666666666666, 0.0]))
