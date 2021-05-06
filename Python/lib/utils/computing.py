# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 22:15:48 2020

@author: Daniel Mastropietro
@description: Functions used to perform computations
"""

import numpy as np
import copy

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

def compute_blocking_probability_birth_death_process(rhos :list, capacity :int):
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

def stationary_distribution_birth_death_process(nservers :int, capacity :int, rhos :list):
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

def stationary_distribution_birth_death_process_at_capacity_unnormalized(nservers :int, capacity :int, rhos :list, ncases_expected=None):
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

def rmse(Vtrue, Vest, weights=None):
    """Root Mean Square Error (RMSE) between Vtrue and Vest, weighted or not weighted by weights.
    @param Vtrue: True Value function.
    @param Vest: Estimated value function.
    @param weights: Number of visits for each value.
    """

    assert type(Vtrue) == np.ndarray and type(Vest) == np.ndarray and (weights is None or type(weights) == np.ndarray), \
            "The first three input parameters are numpy arrays"
    assert Vtrue.shape == Vest.shape and (weights is None or Vest.shape == weights.shape), \
            "The first three input parameters have the same shape({}, {}, {})" \
            .format(Vtrue.shape, Vest.shape, weights and weights.shape or "")

    if np.sum(weights) == 0:
        raise Warning("The weights sum up to zero. They will not be used to compute the RMSE.")
        weights = None

    if weights is not None:
        mse = np.sum( weights * (Vtrue - Vest)**2 ) / np.sum(weights)
    else:
        mse = np.mean( (Vtrue - Vest)**2 )

    return np.sqrt(mse)

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
        #print("\nlevel={}:".format(level))
        r = level
        # Sum over the indices of v to the left of r
        # which determrine the new capacity to satisfy on the indices from r to the right 
        vleft = sum(v[0:r])
        if r < R - 1:
            for k in range(C - vleft, -1, -1):
                #print("\tFOR k ({} downto 0): level: {}, k={}".format(C-vleft, r, k))
                #print("\tv before={}".format(v))
                v[r] = k
                #print("\tv={}".format(v))
                #print("new call: (offset={})".format(r+1))
                # NOTE THE USE OF `yield from` in order to get an yield value from a recursive call!
                yield from all_combos_with_subsum(v, r+1)
        else:
            # No degrees of freedom left for the last level (right-most index)
            #print("\tv before={}".format(v))
            v[r] = C - vleft
            #print("\tv={}".format(v))
            #print("yield v! {}".format(v))
            assert sum(v) == C, "The elements of v sum to C={} (v={})".format(C,v)
            yield v

    v = [0]*R
    gen = all_combos_with_subsum(v, 0)
    return gen


# Tests
if __name__ == "__main__":
    #------------------- comb(n,k) -------------------------#
    print("Testing comb(n,k):")
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
    print("\nTesting all_combos_with_sum(R,C):")
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
    print("\nTesting stationary_distribution_birth_death_process(nservers, capacity, rhos):")
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
