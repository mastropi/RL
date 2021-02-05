# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 22:15:48 2020

@author: Daniel Mastropietro
@description: Functions used to perform computations
"""

import numpy as np

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
        except StopIteration as e:
            break
    combos.close()
    assert count == expected_count, "The number of combinations generated is {} ({})".format(expected_count, count)
    print("OK! {} combinations generated for R={}, C={}.".format(count, R, C))
    #------------ all_combos_with_sum(R,C) -----------------#
