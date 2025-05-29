# -*- coding: utf-8 -*-
"""
Created on Sun May 29 19:46:31 2025

@author: Daniel Mastropietro
@description: Functions related to diffusion processes, i.e. continuous-state continuous-time Markov processes with discretized time dimension
(in order to do simulations of the process).
"""

import numpy as np
import sympy
from scipy.integrate import quad


def estimate_probabilities_in_sets_of_interest(F, sets_of_interest):
    """
    Estimates the probabilities that the simulated process is in each of the given sets of interest

    Parameters:
    - F: array-like containing the samples of the simulated process.
    - sets_of_interest: List of sympy.Set objects defining each set of interest to analyze.

    Returns: dict
    Dictionary containing the estimated probabilities
    """
    probas = dict.fromkeys(sets_of_interest)
    for _set in sets_of_interest:
        if not isinstance(_set, sympy.Set):
            raise ValueError(f"Each element in the `interval` list must be of type sympy.Set: {type(_set)}")
        print(f"Processing set {_set}...")
        if isinstance(_set, sympy.Interval):
            # Extract the boundaries of the interval and use them to check belonging instead of the much slower Set.contains() method!
            lower, upper = float(_set.inf), float(_set.sup)
            count = np.sum((lower <= F) & (F < upper))
        else:
            # Vectorize the Set.contains() method to have the result faster (in principle)
            # Ref: https://stackoverflow.com/questions/10678843/evaluate-sympy-expression-from-an-array-of-values (answer by Marek)
            _set_contains_for_vectors = np.vectorize(_set.contains, otypes=[bool])
            count = np.sum(_set_contains_for_vectors(F))
        probas[_set] = count / len(F)  # Normalize by the total number of samples

    return probas


def calculate_probabilities(r, x, sigma, verbose=False):
    """
    Calculate the probabilities p_1,...,p_6 as a function of r, x, and sigma.

    Parameters:
    r (float): Capacity reserved for FCR-N.
    x (float): Decision variable, capacity reserved for FCR-D.
    sigma (float): Volatility.

    Returns:
    dict: A dictionary with the probabilities p_1, p_2, p_3 (and p_4, p_5, p_6 since p_4 = p_3, p_5 = p_2, p_6 = p_1).
    """
    K1, K2, K3 = compute_K_integrals(x, r, sigma)

    exp1 = np.exp(-5 * x / (48 * sigma**2))
    exp2 = np.exp(-(31 * x + r) / (300 * sigma**2))

    if K1 == 0 or np.log10(K1) < -15:
        denom = 2 * K2 * exp1 + 2 * K3 * exp2
        p_1 = p_6 = 0.0
        p_2 = p_5 = K2 * exp1 / denom
        p_3 = p_4 = K3 * exp2 / denom
    else:
        denom = 2 + 2 * (K2 / K1) * exp1 + 2 * (K3 / K1) * exp2
        p_1 = p_6 = 1 / denom
        p_2 = p_5 = (K2 / K1) * exp1 / denom
        p_3 = p_4 = (K3 / K1) * exp2 / denom

    if verbose:
        print(f"p_1: {round(p_1, 10)}\np_2: {round(p_2, 10)}\np_3: {round(p_3, 10)}\np_4: {round(p_4, 10)}\np_5: {round(p_5, 10)}\np_6: {round(p_6, 10)}")

    return {
        "p_1": p_1,
        "p_2": p_2,
        "p_3": p_3,
        "p_4": p_4,
        "p_5": p_5,
        "p_6": p_6,
    }


def calculate_probability(interval, r, x, sigma, beta=1):
    """
    Calculate the probability of the process being in a given interval

    Parameters:
    interval: array-like
        Lower and upper bound of the interval of interest.
        These bounds can be infinite.

    r: float
        Capacity reserved for FCR-N.

    x: float
        Decision variable, capacity reserved for FCR-D.

    sigma: (float)
        Volatility of the process.

    Return: float in [0, 1]
    Estimated probability of the give interval using numerical integration.
    """
    #--- The functions to integrate on each of the three symmetric pieces
    def func1(y):
        return np.exp(-y**2 * (beta*r + beta*x) / sigma**2)

    def func2(y):
        return np.exp(
            - y**2 * (4*beta*r - beta*x) / (4*sigma**2)
            - np.sign(y) * y**3 * (5*beta*x) / (3*sigma**2)
        )

    def func3(y):
        return np.exp(-np.sign(y) * (20*y**3 * beta*r) / (3*sigma**2))
    #--- The functions to integrate on each of the three symmetric pieces

    #--- Parse input parameters
    try:
        if len(interval) != 2:
            raise ValueError(f"Parameter `interval` must be an array-like object with 2 elements: {interval}")
        if not (interval[0] < interval[1]):
            raise ValueError(f"Parameter `interval` must have its first element smaller than its second element: {interval}")
    except:
        print(f"Parameter `interval` is not an array-like object: {interval}")
    #--- Parse input parameters

    # The piecewise linear pieces and the pieces where each interval endpoint belongs to
    # Note: the indices returned below for any FINITE values of interval endpoints is between 1 and len(piecewise_interval_thresholds) - 1 (= 6 in this case)
    piecewise_interval_thresholds = np.array([-np.Inf, -0.5, -0.1, 0.0, 0.1, 0.5, +np.Inf])
    interval_idx_low = max(0, np.searchsorted(piecewise_interval_thresholds, interval[0]) - 1)      # If interval[0] = -np.Inf, the returned index by np.searchsorted() is 0
    interval_idx_upp = np.searchsorted(piecewise_interval_thresholds, interval[1]) - 1             # If interval[1] = +np.Inf, the returned index by np.searchsorted() is len(arr)-1 (e.g. 6 in this case), as in any FINITE value belonging to the last interval

    # For debugging purposes
    #print(f"The value {interval[0]} belongs to the interval {interval_idx_low}: ({piecewise_interval_thresholds[interval_idx_low], piecewise_interval_thresholds[interval_idx_low+1]})")
    #print(f"The value {interval[1]} belongs to the interval {interval_idx_upp}: ({piecewise_interval_thresholds[interval_idx_upp], piecewise_interval_thresholds[interval_idx_upp+1]})")

    # Probability of each piece (used for adjustment of the integrals that are computed now)
    dict_piecewise_probabilities = calculate_probabilities(r, x, sigma)

    K1, K2, K3 = compute_K_integrals(x, r, sigma)

    functions = [func1, func2, func3, func3, func2, func1]
    Ks = [K1, K2, K3, K3, K2, K1]

    if interval_idx_low == interval_idx_upp:
        integral = quad(functions[interval_idx_low], interval[0], interval[1], epsabs=1e-30, epsrel=1e-11)[0]
        p = integral * dict_piecewise_probabilities['p_' + str(interval_idx_low + 1)] / Ks[interval_idx_low]
    else:
        # Integrals of the different pieces intersecting with the given interval
        n_integrals_to_compute = interval_idx_upp - interval_idx_low + 1
        assert n_integrals_to_compute >= 2

        # Compute the first and last integral of the intersecting pieces
        p = 0.0
        integral = quad(functions[interval_idx_low],
                        interval[0],
                        piecewise_interval_thresholds[interval_idx_low + 1],
                        epsabs=1e-30, epsrel=1e-11)[0]
        p += integral * dict_piecewise_probabilities['p_' + str(interval_idx_low + 1)] / Ks[interval_idx_low]
        integral = quad(functions[interval_idx_upp],
                        piecewise_interval_thresholds[interval_idx_upp],
                        interval[1],
                        epsabs=1e-30, epsrel=1e-11)[0]
        p += integral * dict_piecewise_probabilities['p_' + str(interval_idx_upp + 1)] / Ks[interval_idx_upp]

        # Compute the integrals in the middle (if any)
        for idx in range(interval_idx_low + 1, interval_idx_upp):
            assert idx + 1 < len(dict_piecewise_probabilities), "The probability to add in full can never be the last probability p_6"
            p += dict_piecewise_probabilities['p_' + str(idx + 1)]

    assert 0 <= p <= 1, f"The estimated probability is in [0, 1]: {p}"

    return p


def compute_K_integrals(x, r, sigma, beta=1):
    def K1_integrand(y):
        return np.exp(-y**2 * (beta*r + beta*x) / sigma**2)

    def K2_integrand(y):
        return np.exp(
            - y**2 * (4*beta*r - beta*x) / (4*sigma**2)
            + y**3 * (5*beta*x) / (3*sigma**2)
        )

    def K3_integrand(y):
        return np.exp((20*y**3 * beta*r) / (3*sigma**2))

    K1 = quad(K1_integrand, -np.inf, -0.5, epsabs=1e-30, epsrel=1e-11)[0]
    K2 = quad(K2_integrand, -0.5, -0.1, epsabs=1e-30, epsrel=1e-11)[0]
    K3 = quad(K3_integrand, -0.1, 0.0, epsabs=1e-30, epsrel=1e-11)[0]

    return K1, K2, K3


if __name__ == "__main__":
    #--------------------- calculate_probabilities and calculate_probability ------------------#\
    # Parameters of the Ornstein-Uhlenbeck (OU) diffusion process
    r = 0.6
    x = 1.4
    sigma = 0.04

    #-------
    print("\n--- Testing calculate_probabilities():")

    # True stationary probabilities of each linear piece of the piecewise linear function alpha(X) defining the drift of the OU process
    # The pieces are given in intervals_alpha and their slopes are governed by parameters r and x.
    intervals_alpha = [(-np.Inf, -0.5), (-0.5, -0.1), (-0.1, 0.0), (0.0, 0.1), (0.1, 0.5), (0.5, +np.Inf)]
    probas_stationary_expected = dict({'p_1': 0.0,
                                       'p_2': 0.0070996806,
                                       'p_3': 0.4929003194,
                                       'p_4': 0.4929003194,
                                       'p_5': 0.0070996806,
                                       'p_6': 0.0})

    probas_stationary = calculate_probabilities(r, x, sigma, verbose=True)
    assert isinstance(probas_stationary, dict)
    for i, (k, p) in enumerate(probas_stationary.items()):
        assert k == f"p_{i+1}"
        assert np.isclose(probas_stationary[k], probas_stationary_expected[k])

    #-------
    print("\n--- Testing calculate_probability():")

    # Verifying that the probabilities computed by calculate_probability() on each alpha interval are the same as those given by calculate_probabilities()
    # Note that calculate_probability() uses calculate_probabilities() to get the probability values of the alpha intervals, which are then used to adjust
    # the probabilities of ad-hoc intervals chosen by the user
    print("Analyzing the piecewise linear intervals of drift function alpha(X)...")
    for i, interval in enumerate(intervals_alpha):
        p = calculate_probability(interval, r, x, sigma, beta=1)
        print(f"Analyzing interval {interval}: calculated p = {p}")
        assert np.isclose(p, probas_stationary_expected[f'p_{i+1}'])

    # Test on intervals that are NOT the alpha intervals
    print(f"Analyzing non-alpha intervals...")
    # Negative intervals included in just ONE linear piece of the alpha function
    interval = (-np.Inf, -0.7)
    p = calculate_probability(interval, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval}: calculated p = {p}")
    assert p == 0.0

    interval = (-0.5, -0.3)
    p = calculate_probability(interval, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval}: calculated p = {p}")
    assert np.isclose(calculate_probability(interval, r, x, sigma, beta=1), 1.61E-25, atol=1E-27)

    interval = (-0.3, -0.1)
    p = calculate_probability(interval, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval}: calculated p = {p}")
    assert p < probas_stationary_expected['p_2']
    assert np.isclose(p, probas_stationary_expected['p_2'], atol=1E-25)

    interval = (-0.15, -0.1)
    p = calculate_probability(interval, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval}: calculated p = {p}")
    assert p < probas_stationary_expected['p_2']
    assert np.isclose(p, 0.00708187, atol=1E-9)

    # Negative intervals including more than one linear piece of the alpha function
    interval = (-0.15, -0.05)
    p = calculate_probability(interval, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval}: calculated p = {p}")
    assert p < probas_stationary_expected['p_2'] + probas_stationary_expected['p_3']
    assert np.isclose(p, 0.1473898162, atol=1E-9)

    # Intervals that include 0
    interval = (-0.1, +0.1)
    p = calculate_probability(interval, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval}: calculated p = {p}")
    assert np.isclose(p, probas_stationary_expected['p_3'] + probas_stationary_expected['p_4'])

    interval = (-0.1, +0.02)
    p = calculate_probability(interval, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval}: calculated p = {p}")
    assert np.isclose(p, 0.6440637975)

    interval = (-0.11, 0.14)
    p = calculate_probability(interval, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval}: calculated p = {p}")
    assert np.isclose(p, 0.99713763)

    # Check symmetry
    print(f"Analyzing symmetry...")
    interval1 = np.array([-0.5, -0.1])
    interval2 = np.array([+0.1, +0.5])
    p1 = calculate_probability(interval1, r, x, sigma, beta=1)
    p2 = calculate_probability(interval2, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval1}: calculated p = {p1}")
    assert np.isclose(p1, p2)

    interval1 = np.array([-0.12, -0.08])
    interval2 = np.array([+0.08, +0.12])
    p1 = calculate_probability(interval1, r, x, sigma, beta=1)
    p2 = calculate_probability(interval2, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval1}: calculated p = {p1}")
    assert np.isclose(p1, p2)

    interval1 = np.array([-0.03, +0.02])
    interval2 = np.array([-0.02, +0.03])
    p1 = calculate_probability(interval1, r, x, sigma, beta=1)
    p2 = calculate_probability(interval2, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval1}: calculated p = {p1}")
    assert np.isclose(p1, p2)

    interval1 = np.array([-0.11, +0.13])
    interval2 = np.array([-0.13, +0.11])
    p1 = calculate_probability(interval1, r, x, sigma, beta=1)
    p2 = calculate_probability(interval2, r, x, sigma, beta=1)
    print(f"Analyzing interval {interval1}: calculated p = {p1}")
    assert np.isclose(p1, p2)
    #--------------------- calculate_probabilities and calculate_probability ------------------#
