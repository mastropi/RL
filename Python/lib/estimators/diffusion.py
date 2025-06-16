# -*- coding: utf-8 -*-
"""
Created on Sun May 29 19:46:31 2025

@author: Daniel Mastropietro
@description: Functions related to diffusion processes, i.e. continuous-state continuous-time Markov processes with discretized time dimension
(in order to do simulations of the process).
Some functions are related to the 2024 paper by A. Zocca and Bert Zwart on frequency reserve calculation (FCR),
"Dynamic dimensioning of frequency containment reserves", as they refer to the piecewise linear drift function alpha considered in the paper.
"""

import numpy as np
import math
import sympy
from scipy.integrate import quad

from Python.lib.environments.diffusion import func_drift_fcr_piecewise


#--- Functions defining the PDF of the Ornstein-Uhlenbeck process X(t) at each symmetric interval where the drift function alpha defined in the paper is linear
# The variable of integration is y, the rest are parameters of the process
def pdf1(y, r, x, sigma, beta):
    return np.exp(-y**2 * (beta*r + beta*x) / sigma**2)


def pdf2(y, r, x, sigma, beta):
    return np.exp(
        - y**2 * (4*beta*r - beta*x) / (4 * sigma**2)
        - np.sign(y) * y**3 * (5*beta*x) / (3 * sigma**2)
    )


def pdf3(y, r, x, sigma, beta):
    return np.exp(-np.sign(y) * (20*y**3 * beta*r) / (3*sigma**2))
#--- Functions defining the PDF of the Ornstein-Uhlenbeck process X(t) at each symmetric interval where the drift function alpha defined in the paper is linear


#--- Auxiliary functions
def parse_interval(interval, symmetric=False):
    try:
        if len(interval) != 2:
            raise ValueError(f"Parameter `interval` must be an array-like object with 2 elements: {interval}")
        if not (interval[0] < interval[1]):
            raise ValueError(f"Parameter `interval` must have its first element smaller than its second element: {interval}")
        if symmetric and not np.isclose(-interval[0], interval[1]):
            raise ValueError(f"Parameter `interval` must represent a symmetric interval around zero: {interval}")
    except:
        print(f"Parameter `interval` is not an array-like object: {interval}")

    return True
#--- Auxiliary functions


def expected_first_passage_time(S, s0, alpha, sigma, rtol=1E-5):
    """
    Calculates the expected first passage time for an Ornstein-Uhlenbeck process starting at position s0

    Ref: "First-passage-time density and moments of the Ornstein-Uhlenbeck process", Ricciardi & Sato (1988), pag. 48
    Note: The normalization and de-normalization procedure to use in order to get from the real problem with specific s0, S, mu and sigma values
    is NOT clearly explained in the paper... But I have clarified in the printer copy, with my own clarifying notes in red.

    S: float
        Position whose first passage is of interest from a starting position s0 < S if S is positive or s0 > S if S is negative.

    s0: float
        Starting position of the process from which the first passage time is measured.

    alpha: float
        Drift of the Ornstein-Uhlenbeck process.

    sigma: float
        Volatility of the process.

    Return: float
    Expected first passage time by S of the X(t) process when starting at s0.
    """
    #-- Parse input parameters
    if  S > 0 and s0 > S or \
        S < 0 and s0 < S:
        # We change sign of s0 and S as stated in pag. 44 of the paper (at least for the case when 0 < s0 < S --I still need to analyze the case when s0 < S < 0)
        # Note: At this point, it is not clear to me whether the results given in the paper also work for this case.
        S = -S
        s0 = -s0

    # Standardize the S and s0 values to use in the formulas (which are given for the standardized process)
    # Note: it is not clear how the standardization should be done from the paper. In fact, at first it seems that exactly the opposite that we do here should be done
    # (i.e. that we should *multiply* by the given factor, as opposed to *dividing* the original values by it).
    # However, in the end I figured it out based on: giving a twisted view to what is written in the paper (i.e. not taking it exactly by the word) and by running the
    # estimation of the MFPT by simulation, and seeing that the hitting time seems to be as large as the value computed theoretically here (~ 140k in time units),
    # as the S value considered is never hit in the amount of time allowed for the simulation!!
    _factor = sigma / np.sqrt(2*alpha)
    S_std = S / _factor
    s0_std = s0 / _factor
    #-- Parse input parameters

    done = False
    MFPT = 0.0      # This is the first formula in expression (9), pag. 48 of the paper, namely t_1(S|s0) = phi_1(S) - phi_1(s0)
    n = 0
    while not done:
        n += 1
        new_term = np.sqrt(2)**n * (S_std**n - s0_std**n) / math.factorial(n) * math.gamma(n/2)
        MFPT += new_term
        print(f"Term {n} added, contribution = {new_term} ({new_term / MFPT * 100:.{int(-np.log10(rtol))}f}%) --> phi = {MFPT}")
        done = new_term / MFPT < rtol
    MFPT *= 0.5 / alpha     # We divide by alpha because the k-th derivative of phi must be divided by alpha^k (Ref: Ricciardi, pag. 46)
    print(f"MFPT(S={S} | s0={s0}) = {MFPT}")
    ## OK! phi_1(S=0.15) = 0.1999968032 with error < 1E-5

    return MFPT


def expected_exit_time(interval, r, x, sigma):
    """
    Esimates the expected exit time from a symmetric interval around zero of the Ornstein-Uhlenbeck process with constant drift

    Arguments:
    interval: array-like
        Lower and upper bound of the SYMMETRIC interval around zero.
        These bounds can be infinite.

    r: float
        Capacity reserved for FCR-N.

    x: float
        Decision variable, capacity reserved for FCR-D.

    sigma: float
        Volatility of the process.

    Return: float in [0, 1]
    Estimated exit time from the interval assuming stationary distribution of the process in the interval.
    """
    # Parse input parameters
    parse_interval(interval, symmetric=True)

    #quad(epsabs=1e-30, epsrel=1e-11)



def estimate_probabilities_in_sets_of_interest(F, sets_of_interest):
    """
    Estimates the probabilities that the simulated process is in each of the given sets of interest

    Arguments:
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
    Calculates the probabilities p_1,...,p_6 as a function of r, x, and sigma.

    Arguments:
    r: float
        Capacity reserved for FCR-N.

    x: float
        Decision variable, capacity reserved for FCR-D.

    sigma: (float)
        Volatility of the process.

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

    Arguments:
    interval: array-like
        Lower and upper bound of the interval of interest.
        These bounds can be infinite.

    r: float
        Capacity reserved for FCR-N.

    x: float
        Decision variable, capacity reserved for FCR-D.

    sigma: float
        Volatility of the process.

    Return: float in [0, 1]
    Estimated probability of the give interval using numerical integration.
    """
    # Parse input parameters
    parse_interval(interval)

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

    functions = [pdf1, pdf2, pdf3, pdf3, pdf2, pdf1]
    Ks = [K1, K2, K3, K3, K2, K1]

    if interval_idx_low == interval_idx_upp:
        integral = quad(functions[interval_idx_low], interval[0], interval[1], args=(r, x, sigma, beta), epsabs=1e-30, epsrel=1e-11)[0]
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
                        args=(r, x, sigma, beta),
                        epsabs=1e-30, epsrel=1e-11)[0]
        p += integral * dict_piecewise_probabilities['p_' + str(interval_idx_low + 1)] / Ks[interval_idx_low]
        integral = quad(functions[interval_idx_upp],
                        piecewise_interval_thresholds[interval_idx_upp],
                        interval[1],
                        args=(r, x, sigma, beta),
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

    #------
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


    #--- 2025/06/01: Estimate the expected first passage time
    r = 0.6
    x = 1.4
    sigma = 0.04

    # Drift constant (we take the largest possible value of the piecewise linear alpha in order to get a worst-case scenario, i.e. where the expected first passage time
    # is larger than the actual one when using the piecewise linear drift.
    alpha = r + x

    # TODO: (2025/06/02) CHECK HOW WE DEAL WITH NEGATIVE VALUES OF S AND s0!
    # Pairs of (S, s0) values to test, where S is the hitting position of interest and s0 is the initial position of the process
    pos_values = [(0.1, 0.0), (0.1, 0.05), (-0.1, 0.0), (-0.1, -0.05)]
    for (S, s0) in pos_values:
        print(f"\n(S={S}, s0={s0}):")
        MFPT = expected_first_passage_time(S, s0, alpha, sigma)
        print(f"(S={S}, s0={s0}): MFPT = {MFPT}")

    # Now let's integrate on all possible starting points s0
    # Note that, although it is most likely NOT a correct approach, for each integrand point, we could use
    # the alpha value corresponding to s0 in the piecewise linear function definition of alpha as "constant" drift value of the OU process.
    alpha = func_drift_fcr_piecewise(s0, r=r, x=x)

    # TBC: Not finalized. The next step would be to integrate the expression for the hitting time given in Ricciardi for each possible starting point s0
    # weighted by the pdf(s0) that is derived from the piecewise-linear alpha drift of the OU process considered as frequency deviation model.
    #--- 2025/06/01: Estimate the expected first passage time


    #--- 2025/06/04: Simulate the exit event from A with an OU process with constant negative drift, in order to get an order of magnitude of the value of T to use
    # CONCLUSION: Using a constant drift equal to the maximum drift value of the piecewise-linear alpha drift function, is TOO conservative...
    # i.e. the negative drift is too large, making exiting a moderate A interval, such as (-0.1, 0.1), very highly improbable.
    # In fact, using a simulation time of T = 100k is not enough to observe ANY event.
    from timeit import default_timer as timer
    import numpy as np
    import pandas as pd
    import sympy
    from Python.lib.environments.diffusion import EnvDiffusion, func_drift_const, func_drift_fcr_piecewise, func_noise_gaussian
    from Python.lib.simulators.diffusion import SimulatorDiffusionFV

    # Create the simulation object: diffusion with CONSTANT negative drift
    mu = 0.0
    sigma = 0.04
    r = 0.6     # Capacity reserved for FCR-N (GW)
    x = 1.4     # Capacity reserved for FCR-D (GW)
    dt = 0.01
    env_ou = EnvDiffusion(func_drift=func_drift_const, func_noise=func_noise_gaussian, reflect=False, mu=mu, dt=dt, sigma=sigma, r=r, x=x, theta=r+x)
    #env_ou = EnvDiffusion(func_drift=func_drift_fcr_piecewise, func_noise=func_noise_gaussian, reflect=False, mu=mu, dt=dt, sigma=sigma, r=r, x=x, theta=r+x)
    sim = SimulatorDiffusionFV(env_ou, None, debug=False)

    # Number of replications to run
    nrep = 5 #10  #9
    seed_base = 1313  # 1317

    set_A = sympy.Interval(-0.1, 0.1)

    # Output variable
    exit_times = np.nan*np.ones(nrep)

    time_start_mc = timer()
    for rep in range(nrep):
        seed_rep = seed_base + rep * 1317
        print(f"\nRunning replication {rep + 1} of {nrep}: A={set_A}, seed={seed_rep}...")

        _, _, _, info_mc = sim._run_simulation_mc(dict({'T': 100000, 'absorption_set': set_A}),
                                                 start_state=0.0, store_trajectory=False,
                                                 check_for_stationarity=False,
                                                 seed=seed_rep, verbose=False, verbose_period=10, plot=False)
        if len(info_mc['exit_times']) > 0:
            exit_times[rep] = info_mc['exit_times'][0]
        print(f"First exit time for replication {rep+1}: t={exit_times[rep]*dt}")

    print(f"Distribution of first exit times:\n{pd.Series(exit_times*dt).describe()}")
    #--- 2025/06/04: Simulate the exit event
