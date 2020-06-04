# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 22:15:48 2020

@author: Daniel Mastropietro
@description: Functions used to perform computations
"""

import numpy as np

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
