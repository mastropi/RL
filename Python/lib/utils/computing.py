# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 22:15:48 2020

@author: Daniel Mastropietro
@description: Functions used to perform specialized computations commonly used by the application.
"""

import warnings
import copy
from unittest import TestCase

import numpy as np
import pandas as pd

from Python.lib.utils.basic import as_array


def mad(x):
    """
    Computes the Median Absolute Deviation on a list or array of values.
    
    Missing values are ignored.

    x: list or numpy array
        Values on which the MAD is computed.

    Return: float
        median( |x(i) - median(x)| )
    """
    
    # Convert to an array if x is given as list
    x = np.array(x)
    x_median = np.nanmedian(x)
    x_dev = x - x_median
    x_mad = np.median( np.abs( x_dev ) )
    
    return x_mad


def rmse(Vtrue: np.ndarray, Vest: np.ndarray, weights: np.ndarray=None):
    """Root Mean Square Error (RMSE) between Vtrue and Vest, optionally weighted

    All weights are assumed to be non-negative.

    Arguments:
    Vtrue: np.ndarray of any shape (the same shape as Vest and weights)
        True Value function.

    Vest: np.ndarray of any shape (the same shape as Vtrue and weights)
        Estimated Value function.

    weights: np.ndarray of any shape (the same shape as Vtrue and Vest)
        Weights to use in the computation of the MAPE.
        Ex: Number of visits associated to each estimated value.

    Return: float
    For the weighted version:
        sqrt( sum( weight * (Vest - Vtrue)**2 ) / sum(weight) )
    For the unweighted version:
        sqrt( mean( (Vest - Vtrue)**2 ) )
    """
    if type(Vtrue) != np.ndarray or type(Vest) != np.ndarray or (weights is not None and type(weights) != np.ndarray):
        raise ValueError("The first three input parameters must be numpy arrays (`weights` can be None)")
    if Vtrue.shape != Vest.shape or (weights is not None and Vest.shape != weights.shape):
        raise ValueError("The first three input parameters have the same shape({}, {}, {})" \
                         .format(Vtrue.shape, Vest.shape, weights and weights.shape or ""))

    if np.sum(weights) == 0:
        raise Warning("The weights sum up to zero. They will not be used to compute the RMSE.")
        weights = None

    if weights is not None:
        mse = np.sum( weights * (Vest - Vtrue)**2 ) / np.sum(weights)
    else:
        mse = np.mean( (Vest - Vtrue)**2 )

    return np.sqrt(mse)


def mape(Vtrue: np.ndarray, Vest: np.ndarray, weights: np.ndarray=None):
    """Mean Absolute Percent Error (MAPE) between Vtrue and Vest, weighted or not weighted by weights.

    All weights are assumed to be non-negative.

    Arguments:
    Vtrue: np.ndarray of any shape (the same shape as Vest and weights)
        True Value function.

    Vest: np.ndarray of any shape (the same shape as Vtrue and weights)
        Estimated Value function.

    weights: np.ndarray of any shape (the same shape as Vtrue and Vest)
        Weights to use in the computation of the MAPE.
        Ex: Number of visits associated to each estimated value.

    Return: float
    The error in each element is set to 0.0 when the two compared values Vtrue and Vest are equal to 0.0
    (so that we avoid a NaN when doing (0.0 - 0.0) / 0.0).

    So, excluding these values, the returned value is computed as follows:
    - for the weighted version:
        sum( weight * abs(Vest - Vtrue) / abs(Vtrue) ) / sum(weight)
    - for the unweighted version:
        mean( abs(Vest - Vtrue) / abs(Vtrue) )
    If any value in Vtrue is 0.0 and the corresponding estimated value Vest is not 0.0,
    NaN is returned for the MAPE.
    """
    if type(Vtrue) != np.ndarray or type(Vest) != np.ndarray or (weights is not None and type(weights) != np.ndarray):
        raise ValueError("The first three input parameters must be numpy arrays (`weights` can be None)")
    if Vtrue.shape != Vest.shape or (weights is not None and Vest.shape != weights.shape):
        raise ValueError("The first three input parameters have the same shape({}, {}, {})" \
                        .format(Vtrue.shape, Vest.shape, weights and weights.shape or ""))

    if np.sum(weights) == 0:
        raise Warning("The weights sum up to zero. They will not be used to compute the RMSE.")
        weights = None

    # Convert all arrays to a 1D array to avoid problems when zip()-ing them in the list below
    nobs = np.prod(Vtrue.shape)
    Vtrue_1d = np.squeeze(Vtrue.reshape(1, nobs))
    Vest_1d = np.squeeze(Vest.reshape(1, nobs))
    if weights is not None:
        weights_1d = np.squeeze(weights.reshape(1, nobs))

    if weights is not None:
        weighted_error_rel = [0.0   if x_true == x_est == 0.0 or w == 0.0
                                    else w * np.abs(x_est - x_true) / np.abs(x_true)
                                    for x_true, x_est, w in zip(Vtrue_1d, Vest_1d, weights_1d)]
        MAPE = np.sum( weighted_error_rel ) / np.sum(weights[:])
    else:
        error_rel = [0.0    if x_true == x_est == 0.0
                            else np.Inf if x_true == 0.0
                                        else np.abs(x_true - x_est) / np.abs(x_true)
                            for x_true, x_est in zip(Vtrue_1d, Vest_1d)]
        MAPE = np.mean(error_rel)

    return MAPE


def smooth(x, window_size):
    "Smooth a signal by moving average"
    # Ref: https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/
    if window_size <= 1:
        return x
    else:
        # Convert to a pandas series
        x_series = pd.Series(x)
        # Get the window of series of observations till the current time
        windows = x_series.rolling(window_size)
        # Smooth!
        x_smooth = windows.mean()

        return x_smooth


def comb(n,k):
    """
    Efficient calculation of the combinatorial number, which does NOT use the factorial.
    
    Ex: `comb(C+K-1,K-1)` does NOT give overflow but `factorial(C+K-1) / factorial(K-1) / factorial(C)`
    does for e.g. C = 400, K = 70
    The actual error is "OverflowError: integer division result too large for a float".
    """
    if  not isinstance(n, int) and not isinstance(n, np.int32) and not isinstance(n, np.int64) and \
        not isinstance(k, int) and not isinstance(k, np.int32) and not isinstance(k, np.int64):
        raise ValueError("n and k must be integer (n={}, k={})".format(n,k))
    if (n < k):
        raise ValueError("n cannot be smaller than k (n={}, k={})".format(n,k))

    num = 1
    den = 1
    for i in range( min(n-k, k) ):
        num *= n - i
        den *= i + 1
        
    return int( num / den )


def all_combos_with_sum(R, C):
    """
    Returns a generator of all possible integer-valued arrays whose elements sum to a fixed number.

    Arguments:
    R: int
        Dimension of the integer-valued array to generate.

    C: int
        Sum of the elements of the integer-valued array.
    """

    # Note: documentation on the `yield` expression: https://docs.python.org/3/reference/expressions.html#yieldexpr

    def all_combos_with_subsum(v, level):
        # print("\nlevel={}:".format(level))
        r = level
        # Sum over the indices of v to the left of r
        # which determrine the new capacity to satisfy on the indices from r to the right
        vleft = sum(v[0:r])
        if r < R - 1:
            for k in range(C - vleft, -1, -1):
                # print("\tFOR k ({} downto 0): level: {}, k={}".format(C-vleft, r, k))
                # print("\tv before={}".format(v))
                v[r] = k
                # print("\tv={}".format(v))
                # print("new call: (offset={})".format(r+1))
                # NOTE THE USE OF `yield from` in order to get an yield value from a recursive call!
                yield from all_combos_with_subsum(v, r + 1)
        else:
            # No degrees of freedom left for the last level (right-most index)
            # print("\tv before={}".format(v))
            v[r] = C - vleft
            # print("\tv={}".format(v))
            # print("yield v! {}".format(v))
            assert sum(v) == C, "The elements of v sum to C={} (v={})".format(C, v)
            yield v

    v = [0] * R
    gen = all_combos_with_subsum(v, 0)
    return gen


def generate_min_exponential_time(rates):
    """
    Generates a realization of the minimum of exponential times at the given rates

    Arguments:
    rates: positive float or list of positive floats or numpy array of positive floats
        Rates of the exponential distributions on which the minimum time is generated.
        Some (but not all) of the rates may be NaN as only the non-NaN values are considered for the possible
        exponential distributions.

    Return: tuple
    Tuple with the following elements:
    - time: realization of the minimum time among the exponential distributions of the given rates.
    - idx: index indicating which exponential rate the generated time should be associated with, which is chosen
    randomly out of all valid rates given (where a valid rate means that it's a non-negative real number).
    """
    # Check the rates
    rates = as_array(rates)
    if any(rates[~np.isnan(rates)] < 0):
        warnings.warn("Some of the rates are negative... they will be ignored for the generation of the min exponential time ({}".format(rates))

    # Find the valid rates
    valid_rates = copy.deepcopy(rates)
    is_valid_rate = np.array([True if r > 0 else False for r in rates])
    valid_rates[~is_valid_rate] = np.nan

    # Rate of occurrence of ANY event among the input rates
    event_rate = np.nansum(valid_rates)
    if event_rate <= 0:   # NOTE that, in case all valid_rates are NaN, the nansum() of all NaNs is 0.0! (so this condition still takes care of the all-NaN-rates case)
        raise ValueError("The event rate computed from the given rates ({}) must be positive ({})".format(rates, event_rate))

    # Generate the event time
    event_time = np.random.exponential(1/event_rate)

    # Probability of selection of each possible event with a valid rate
    probs = valid_rates[is_valid_rate] / event_rate
    indices_to_choose_from = [idx for idx, rate in enumerate(valid_rates) if not np.isnan(rate)]
    #print("probs: {}".format(probs))   # For 5 particles with rates lambda = 0.7, mu = 1.0 each, these probabilities are 0.082353 (for each of the 5 lambda's) and 0.11765 (for each of the 5 mu's)
    #print("indices to choose from: {}".format(indices_to_choose_from))

    # Define the index on which the generated event time should be associated with
    idx_event = np.random.choice(indices_to_choose_from, size=1, p=probs)[0]    # Need `[0]` because the value returned by random.choice() is an array

    return event_time, idx_event


def compute_survival_probability(survival_times: list):
    """
    Computes the survival probability from a list of sorted survival times

    Arguments:
    survival_times: list
        Sorted list containing the observed survival times on which the step survival probability is computed.

    Return: pandas DataFrame
    Data frame containing the following two columns:
    - 't': the input survival times (assumed sorted)
    - 'P(T>t)': the survival probability for the corresponding t value.
    """
    assert survival_times[0] == 0.0
    # Number of observed death events used to measure survival times
    N = len(survival_times) - 1

    if N > 0:
        proba_surv = [n / N for n in
                      range(N, -1, -1)]  # This is N downto 0 (i.e. N+1 elements as there are in survival_times)
        assert proba_surv[-1] == 0.0
    else:
        proba_surv = [1.0]

    assert proba_surv[0] == 1.0
    return pd.DataFrame.from_items([('t', survival_times),
                                    ('P(T>t)', proba_surv)])


def get_server_loads(job_rates, service_rates):
    """
    Returns the server loads (rhos) for a queue system from the job arrival rates and the service rates"

    Arguments:
    job_rates: list or array
        Arrival rates of the jobs by server.

    service_rates: list or array
        Server service rates.

    Return: list
    List of server loads, rhos computed as lambda / mu, where lambda is the job arrival rate and mu is the service rate
    of each server.
    """
    return [b/d for b, d in zip(job_rates, service_rates)]


def compute_job_rates_by_server(job_class_rates, nservers, policy_assign_map):
    """
    Computes the equivalent job arrival rates for each server from the job arrival rates (to the single buffer)
    and the job assignment policy of the agent.

    This can be used when the job is pre-assigned to a server at the moment of arrival (as opposed to being
    assigned when a server queue is freed).

    Arguments:
    job_class_rates: list or array
        Arrival rate of each job class to the queue buffer.

    nservers: int
        Number of servers in the queue system.

    policy_assign_map: list of lists
        List of probabilities of assigning each job class to a server in the queue system.
        Ex: In a scenario with 2 job classes and 3 servers, the following policy assigns job class 0
        to server 0 or 1 with equal probability and job class 1 to server 1 or 2 with equal probability:
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]

    Return: list
    The equivalent job arrival rates for each server r, computed as:
    job_arrival_rate[r] = sum_{c over job classes} { job_rate[c] * Pr(assign job c to server r) }
    """
    R = nservers
    J = len(job_class_rates)
    job_rates_by_server = [0]*R
    for r in range(R):
        for c in range(J):
            job_rates_by_server[r] += policy_assign_map[c][r] * job_class_rates[c]

    return job_rates_by_server


def compute_number_of_burnin_cycles_from_burnin_time(cycle_times, burnin_time):
    """
    Computes the number of burn-in cycles to remove from the beginning of a simulation
    when computing expectations in order to guarantee a specified burn-in continuous time
    """
    if burnin_time > 0:
        # Find the number of cycles to remove from the beginning of the simulation in order to guarantee that
        # the time at which the first cycle considered starts is larger than the burn-in time
        # The logic is best illustrated by an example:
        # burnin_time = 3.5
        # cycle_times = [0.3, 0.7, 2.8, 0.4, 0.9]
        # => cumsum(cycle_times) = [0.3, 1.0, 3.8, 4.2, 5.1]
        #   => the loop below stops at idx = 2 (because t=3.8 >= burnin_time=3.5)
        #       => burnin_cycles = 3 (that is we need to remove the first 3 cycles --including the third one--
        #                              because all those cycles have a non-empty intersection with the interval
        #                              [0, burnin_time=3.5]. So, the first cycle that is falls FULLY in the
        #                              period assumed stationary is the fourth cycle that goes from 3.8 thru 4.2
        #                              whose cycle time is 0.4, i.e. taken from index 3 of the cycle_times list)
        assert cycle_times is not None
        burnin_cycles = len(cycle_times)  # Initialize the number of burn-in cycles to the maximum possible (where all cycles are removed actually!)
        for idx, t in enumerate(np.cumsum(cycle_times)):
            if t >= burnin_time:
                burnin_cycles = idx + 1  # See example above to justify why we sum +1 to idx to get the number of burn-in cycles
                break
    else:
        burnin_cycles = 0

    return burnin_cycles


def compute_blocking_probability_birth_death_process(rhos: list, capacity: int):
    """
    Computes the true blocking probability of a birth-death process with R servers and total capacity C.
    
    Arguments:
    rhos: list
        List of the server intensities: lambda / mu for each server in the system, where lambda is the job
        arrival rate and mu is the service rate.

    capacity: int
        Capacity of the system: maximum size of the buffer placed at the entrance of the system.

    Return: float
    The probability that the process is at its max capacity.
    """
    if not isinstance(rhos, list) or len(rhos) == 0:
        raise ValueError("Input parameter `rho` must be a non-empty list: {}".format(rhos))
         
    R = len(rhos)
    C = capacity
    if R == 1:
        proba_blocking = rhos[0]**C / np.sum([ rhos[0]**i for i in range(C+1) ])
    else:
        const = 0
        ncases_total = 0
        prod = [0]*(C+1)   # Array to store the contributions to the normalizing constant for each 1 <= c <= C
        for c in range(C+1):
            ncases = comb(c+R-1,c)
            combos_generator = all_combos_with_sum(R,c)
            count = 0
            while True:
                try:
                    n = next(combos_generator)
                    assert len(n) == len(rhos), "The length of v and rho rates coincide ({}, {})".format(len(n), len(rhos))
                    prod[c] += np.prod( [(1- r)*r**nr for r, nr in zip(rhos, n)] )
                    count += 1
                except StopIteration:
                    break
            combos_generator.close()
            const += prod[c]
            assert count == ncases
            ncases_total += ncases
        assert const <= 1, "The normalizing constant is <= 1"
        assert abs(sum(prod)/const - 1.0) < 1E-6

        # Blocking probability
        proba_blocking = prod[C] / const

    return proba_blocking       


def stationary_distribution_birth_death_process(nservers: int, capacity: int, rhos: list):
    """
    Computes the stationary distribution of a birth-death process for all its possible states.

    Arguments:
    nservers: int
        The number of servers in the system, which defines the dimension of the state space.

    capacity: int
        The capacity for which the stationary distribution of states should be computed.

    rhos: list
        The intensity of each server (intensity = arrival-rate / service-rate).

    Return: tuple
    The tuple contains the following elements:
    - a list of each possible state n = (n1, n2, ..., nR) such that the sum of the n(j)'s is less than or equal to the given capacity.
    - a list with the probability of occurrence of each state given in the first list under stationarity.  
    """
    if len(rhos) != nservers:
        raise ValueError("The length of the process density ({}) must be equal to the number of servers in the system ({})".format(len(rhos), nservers))
    if capacity < 0:
        raise ValueError("The capacity must be non-negative: {}".format(capacity)) 

    # Create short-name variable for number of servers
    R = nservers

    # Total expected number of cases (needed to initialize the output arrays and check the results)
    ncases_total_expected = 0
    for c in range(capacity+1):
        ncases_total_expected += comb(c+R-1,c)

    # Initialize the output arrays with all possible x = (n1, n2, ..., nR) combinations,
    # and the probability distribution for each x. They are all initialized to a dummy value.
    # They are indexed by the order in which all the combinations are generated below:
    # - by increasing capacity 0 <= c <= capacity
    # - by the order defined by the all_combos_with_sum(R,c) function
    x = [[-1]*R]*ncases_total_expected
    dist = [0]*ncases_total_expected
    ncases_total = 0
    const = 0   # Normalizing constant (because the system's capacity is finite)
    last_case = -1
    for c in range(capacity+1):
        ncases_expected = comb(c+R-1,c)
        ind = slice(last_case+1, last_case+1+ncases_expected)
        
        x[ind], dist[ind] = stationary_distribution_birth_death_process_at_capacity_unnormalized(R, c, rhos, ncases_expected)

        const += sum(dist[ind])
        ncases_total += ncases_expected
        # Prepare for next iteration
        last_case += ncases_expected
    dist /= const
    assert ncases_total == ncases_total_expected, "The number of TOTAL generated combinations for R={}, C<={} ({}) is equal to the expected number of combinations ({})".format(R, C, ncases_total, ncases_total_expected)
    assert const <= 1, "The normalizing constant is <= 1 ({})".format(const)
    assert abs(sum(dist) - 1.0) < 1E-6, "The sum of the distribution function is 1 ({:.6f})".format(sum(dist))

    return x, dist


def stationary_distribution_birth_death_process_at_capacity_unnormalized(nservers: int, capacity: int, rhos: list, ncases_expected=None):
    """
    Computes the UNNORMALIZED stationary distribution of a birth-death process for the state subspace
    associated to a fixed capacity, i.e. for all the n = (n1, n2, ..., nR) such that their sum is equal
    to a given capacity.

    Arguments:
    nservers: int
        The number of servers in the system, which defines the dimension of the state space.

    capacity: int
        The capacity for which the stationary distribution of states should be computed.

    rhos: list
        The intensity of each server (intensity = arrival-rate / service-rate).

    ncases_expected: (opt) int
        The expected number of states satisfying the capacity condition mentioned in the description.
        This number may have been computed elsewhere and it is used here if given, in order to avoid
        its recalculation.

    Return: tuple
    The tuple contains the following elements:
    - a list of each possible state n = (n1, n2, ..., nR) such that the sum of the n(j)'s is equal to the given capacity.
    - a list with the unnormalized probability of occurrence of each state given in the first list.  
    In order to normalize it to a distribution, each value should be divided by the sum of all the values in the list.
    """
    R = nservers
    C = capacity
    if ncases_expected is None:
        ncases_expected = comb(C+R-1,C)
    combos_generator = all_combos_with_sum(R, C)
    ncases = 0
    x = [[-1]*R]*ncases_expected
    dist = [0]*ncases_expected
    while True:
        try:
            next_combo = next(combos_generator)
            #print("next_combo (k={}): {}".format(ncases, next_combo))
            x[ncases] = copy.deepcopy(next_combo)    # IMPORTANT: Need to make a copy o.w. x[k] will share the same memory address as next_combo and its value will change at the next iteration!!
            dist[ncases] = np.prod( [(1- r)*r**nr for r, nr in zip(rhos, next_combo)] )
            ncases += 1
        except StopIteration:
            break
    combos_generator.close()
    assert ncases == ncases_expected, "The number of generated combinations for R={}, C={} ({}) is equal to the expected number of combinations ({})".format(R, C, ncases, ncases_expected)

    return x, dist


# Tests
if __name__ == "__main__":
    #----------------- rmse() and mape() --------------------#
    print("\n--- Testing rmse() and mape():")

    # Test of one difference 0.0 with true value 0.0. The APE for that value should be 0.0 and the MAPE should NOT be NaN
    x = np.array([3, 2, 0.0, -1.0])
    assert np.abs(rmse(2*x, x) - 1.8708287) < 1E-6
    assert np.abs(mape(2*x, x) - 0.375) < 1E-6

    # Test that when there is at least one division by 0, the MAPE yields Inf
    x = np.array([3, 2, 0.0, -1.0])
    assert np.abs(rmse(x, x + 1) - 1.0) < 1E-6
    assert mape(x, x + 1) == np.Inf

    # Test that when there is at least one NaN the RMSE and MAPE yield NaN
    x = np.array([3, 2, 0.0, np.nan])
    y = np.array([3, 5.1, 0.0, 2.1])
    assert np.isnan(rmse(x, y))
    assert np.isnan(mape(x, y))
    assert np.isnan(rmse(y, x))
    assert np.isnan(mape(y, x))

    # Test 2D arrays
    # These results were verified in Excel
    x_2d = np.array([[3, 2, 0.0, -1.0],
                     [4, 8, -9.1, 4.0]])
    y_2d = np.array([[3.2, 1.8, 0.0, 4.5],
                    [4.5, 6.8, -5.4, 2.1]])
    assert np.abs(rmse(x_2d, y_2d) - 2.4829418) < 1E-6
    assert np.abs(mape(x_2d, y_2d) - 0.8529075) < 1E-6

    # Test weighted versions
    # These results were verified in Excel
    weights_2d = np.array([[1.0, 3.2, 0.5, 0.0],
                           [2.1, 5.1, 0.8, 0.7]])
    assert np.abs(rmse(x_2d, y_2d, weights=weights_2d) - 1.2671509) < 1E-6
    assert np.abs(mape(x_2d, y_2d, weights=weights_2d) - 0.1546224) < 1E-6

    # Test error when different shapes given
    # Note that in order to use unittest.TestCase.assertRaises() we need to create an INSTANCE of the TestCase class
    # Otherwise we get the error "assertRaises() arg 1 must be an exception type or tuple of exception types"
    # Ref: https://stackoverflow.com/questions/18084476/is-there-a-way-to-use-python-unit-test-assertions-outside-of-a-testcase
    # which was referenced by Martijn Pieters in https://stackoverflow.com/questions/49369163/custom-exceptions-in-unittests
    x_2d = np.array([[3, 2, 0.0],
                     [4, 8, -9.1]])
    y_2d = np.array([[3.2, 1.8, 0.0, 4.5],
                    [4.5, 6.8, -5.4, 2.1]])
    tc = TestCase()
    tc.assertRaises(ValueError, rmse, x_2d, y_2d)
    tc.assertRaises(ValueError, mape, x_2d, y_2d)
    #----------------- rmse() and mape() --------------------#


    #---------- generate_min_exponential_time() -------------#
    print("\n--- Testing generate_min_exponential_time(rates):")
    #-- Normal scenario
    np.random.seed(1717)
    rates = [0.2, 0.5, 0.7]
    t, idx = generate_min_exponential_time(rates)
    print("Observed minimum time {} at index {}".format(t, idx))
    t_expected, idx_expected = 0.3137434, 1
    assert np.allclose(t, t_expected)
    assert idx == idx_expected
    # Repeat the realization to check that the returned index is always 0 (because that's the only valid rate!)
    nevents_by_rate = np.zeros(len(rates))
    N = 10
    # Expected proportion of events for each exponential
    p = rates / np.nansum(rates)
    for i in range(N):
        t, idx = generate_min_exponential_time(rates)
        nevents_by_rate[idx] += 1
    # Observed proportions of events by server and type of event
    phat = 1.0 * nevents_by_rate / N
    se_phat = np.sqrt(p * (1 - p) / N)
    print("EXPECTED proportions of events by rate:\n{}".format(p))
    print("Observed proportions of events by rate on N={} generated events:\n{}".format(N, phat))
    print("Standard Errors on N={} generated events:\n{}".format(N, se_phat))
    assert np.allclose(phat, p, atol=3*se_phat)   # true probabilities should be contained in +/- 3 SE(phat) from phat

    #-- Case with some NaN and negative rate values
    np.random.seed(1717)
    t, idx = generate_min_exponential_time([0.2, np.nan, -0.7])
    t_expected, idx_expected = 2.1962040, 0
    print("Observed minimum time {} at index {}".format(t, idx))
    assert np.allclose(t, t_expected)
    assert idx == idx_expected
    # Repeat the realization to check that the returned index is always 0 (because that's the only valid rate!)
    for i in range(10):
        t, idx = generate_min_exponential_time([0.2, np.nan, -0.7])
        assert idx == idx_expected
    #---------- generate_min_exponential_time() -------------#


    #------------------- comb(n,k) -------------------------#
    print("\n--- Testing comb(n,k):")
    count = 0
    # Extreme cases
    assert comb(0,0) == 1; print(".", end=""); count += 1
    assert comb(1,0) == 1; print(".", end=""); count += 1
    assert comb(5,0) == 1; print(".", end=""); count += 1
    assert comb(5,5) == 1; print(".", end=""); count += 1
    assert comb(5,1) == 5; print(".", end=""); count += 1
    
    # Reference for the below technique of asserting that a ValueError is raised by the function call:
    # https://www.geeksforgeeks.org/python-assertion-error/
    try:
        comb(1,5)
    except ValueError:
        assert True; print(".", end=""); count += 1

    # Symmetry
    assert comb(6,3) == 20; print(".", end=""); count += 1
    assert comb(6,2) == 15; print(".", end=""); count += 1
    assert comb(6,4) == 15; print(".", end=""); count += 1
    
    # Very large numbers which using factorial give overflow
    assert comb(400,70) == 18929164090932404651386549010279950939591217550712274775589094772534627690086400; print(".", end=""); count += 1
    print("\nOK! {} tests passed.".format(count))
    #------------------- comb(n,k) -------------------------#


    #------------ all_combos_with_sum(R,C) -----------------#
    print("\n--- Testing all_combos_with_sum(R,C):")
    R = 3
    C = 20
    expected_count = comb(C+R-1,C)
    combos = all_combos_with_sum(R,C)
    count = 0
    while True:
        try:
            v = next(combos)
            assert len(v) == R, "v is of length R={} ({})".format(R, len(v))
            assert sum(v) == C, "the sum of elements of v is C={} ({})".format(C, sum(v))
            count += 1
            if count == 1:
                assert v == [20, 0, 0]
            if count == 2:
                assert v == [19, 1, 0]
            if count == 3:
                assert v == [19, 0, 1]
            if count == 4:
                assert v == [18, 2, 0]
            if count == 5:
                assert v == [18, 1, 1]
            if count == 6:
                assert v == [18, 0, 2]
            if count == expected_count:
                assert v == [0, 0, 20]
        except StopIteration:
            break
    combos.close()
    assert count == expected_count, "The number of combinations generated is {} ({})".format(expected_count, count)
    print("OK! {} combinations generated for R={}, C={}.".format(count, R, C))
    #------------ all_combos_with_sum(R,C) -----------------#


    #------------ stationary_distribution_birth_death_process --------------#
    import matplotlib.pyplot as plt
    print("\n--- Testing stationary_distribution_birth_death_process(nservers, capacity, rhos):")
    R = 3
    C = 3
    n_expected = [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2],
                [3, 0, 0],
                [2, 1, 0],
                [2, 0, 1],
                [1, 2, 0],
                [1, 1, 1],
                [1, 0, 2],
                [0, 3, 0],
                [0, 2, 1],
                [0, 1, 2],
                [0, 0, 3]
                ]

    print("Test #1: all process intensities rhos are equal (to 0.5)")
    rhos_equal = [0.5, 0.5, 0.5]
    dist_expected = [
                    0.190476,
                    0.095238,
                    0.095238,
                    0.095238,
                    0.047619,
                    0.047619,
                    0.047619,
                    0.047619,
                    0.047619,
                    0.047619,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810,
                    0.023810
                    ]           
    n, dist = stationary_distribution_birth_death_process(R, C, rhos_equal)
    print("State space for R={}, C={}: {}".format(R, C, len(dist)))
    print("Distribution for rhos={}:".format(rhos_equal))
    [print("index={}: x={}, p={:.6f}".format(idx, x, p)) for idx, (x, p) in enumerate(zip(n, dist))]
    print("---------------")
    print("Sum: p={:.6f}".format(sum(dist)))
    assert abs(sum(dist) - 1.0) < 1E-6, "The sum of the distribution function is 1 ({.6f})".format(sum(dist))
    assert all([x == x_expected for x, x_expected in zip(n, n_expected)]), "The expected event space is verified"
    assert all(abs(dist - dist_expected) < 1E-6), "The expected distribution is verified"

    # Multinomial distribution on these events
    # Notes:
    # - sample_size is the number of experiments to run for the multinomial distribution
    # in our case, it's our sample size.
    # - the resulting array gives the number of times we get the index value out of all possible indices
    # from 0 ... length(dist)-1, so we can use it as the number of times to use the n array associated
    # to each index as starting point of the simulation
    sample_size = 1000
    print("Generating {} samples with the given distribution".format(sample_size))
    sample = np.random.multinomial(sample_size, dist, size=1)
    print("No. of times each array combination n appears in the sample:\n{}".format(sample[0]))
    plt.figure()
    plt.plot(dist, 'r.-')
    plt.plot(sample[0]/sample_size, 'b.')
    plt.title("R={}, C={}, rhos={} (ALL EQUAL) (sample size = {})".format(R, C, rhos_equal, sample_size))

    print("Test #2: process intensities rhos are different (smaller than 1)")
    rhos_diff = [0.8, 0.7, 0.4]
    dist_expected = [
                    0.124611,
                    0.099688,
                    0.087227,
                    0.049844,
                    0.079751,
                    0.069782,
                    0.039875,
                    0.061059,
                    0.034891,
                    0.019938,
                    0.063801,
                    0.055826,
                    0.031900,
                    0.048847,
                    0.027913,
                    0.015950,
                    0.042741,
                    0.024424,
                    0.013956,
                    0.007975
                    ]
    n, dist = stationary_distribution_birth_death_process(R, C, rhos_diff)
    print("State space for R={}, C={}: {}".format(R, C, len(dist)))
    print("Distribution for rhos={}:".format(rhos_diff))
    [print("index={}: x={}, p={:.6f}".format(idx, x, p)) for idx, (x, p) in enumerate(zip(n, dist))]
    print("---------------")
    print("Sum: p={:.6f}".format(sum(dist)))
    assert abs(sum(dist) - 1.0) < 1E-6, "The sum of the distribution function is 1 ({.6f})".format(sum(dist))
    assert all([x == x_expected for x, x_expected in zip(n, n_expected)]), "The expected event space is verified"
    assert all(abs(dist - dist_expected) < 1E-6), "The expected distribution is verified"

    # Multinomial distribution on these events
    # Notes:
    # - sample_size is the number of experiments to run for the multinomial distribution
    # in our case, it's our sample size.
    # - the resulting array gives the number of times we get the index value out of all possible indices
    # from 0 ... length(dist)-1, so we can use it as the number of times to use the n array associated
    # to each index as starting point of the simulation
    sample_size = 1000
    print("Generating {} samples with the given distribution".format(sample_size))
    sample = np.random.multinomial(sample_size, dist, size=1)
    print("No. of times each array combination n appears in the sample:\n{}".format(sample[0]))
    plt.figure()
    plt.plot(dist, 'r.-')
    plt.plot(sample[0]/sample_size, 'b.')
    plt.title("R={}, C={}, rhos={} (ALL DIFFERENT) (sample size = {})".format(R, C, rhos_diff, sample_size))
    #------------ stationary_distribution_birth_death_process --------------#
