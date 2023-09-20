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
