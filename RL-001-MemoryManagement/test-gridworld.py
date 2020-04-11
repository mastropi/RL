# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:48:52 2020

@author: Daniel Mastropietro
@description: Test the gridworld environment with different learning algorithms.
"""

if __name__ == "__main__":
    # Setup the working environment so that e.g. modules defined in 'Python/lib' are found
    import runpy
    runpy.run_path('../setup.py')

else:
    import numpy as np
    import matplotlib.pyplot as plt

#    import importlib
#    importlib.import_module('TD_Lambda', '../Python/lib/GitHub/MJeremy2017/RL/RandomWalk(Lambda)')
    from Python.lib.GitHub.MJeremy2017.RL.RandomWalk_Lambda import TD_Lambda as TDL
    from Python.lib import TD_Lambda_DM as TDLDM
    
    ### 1) TD lambda
    DEBUG = False

    states = np.linspace(0, TDL.NUM_STATES+1, TDL.NUM_STATES+2, dtype=int)
    actual_state_values = np.arange(-20, 22, 2) / 20.0
    actual_state_values[0] = actual_state_values[-1] = 0 
        ## Recall that terminal nodes have value 0, since:
        ## - we cannot go anywhere from there
        ## - using 0 as their value we can simulate an episodic task as a continuing task
    actual_state_values

    alphas = np.linspace(0, 0.8, 6)
    alphas = np.array([0.3])
    lambdas = np.linspace(0, 1, 5)
    lambdas = np.array([0.8])
    rounds = 50

    #plt.figure(figsize=[10, 6])
    plt.figure()
    state_values = np.zeros( (len(lambdas), len(alphas), len(actual_state_values)) )
    idx_lambda = -1
    for lamb in lambdas:
        idx_lambda += 1
        alpha_erros = []
        idx_alpha = -1
        for alpha in alphas:
            idx_alpha += 1
            valueFunc = TDLDM.ValueFunctionTD_DM(alpha=alpha, lmbda=lamb)
            rw = TDL.RWTD(debug=DEBUG)
            rw.play(valueFunc, rounds=rounds)
            state_values[idx_lambda][idx_alpha] = valueFunc.getValues()
            rmse = np.sqrt(np.mean(np.power(valueFunc.getValues() - actual_state_values, 2)))
            print("lambda {} alpha {} rmse {}".format(lamb, alpha, rmse))
            alpha_erros.append(rmse)

            plt.plot(states, state_values[idx_lambda][idx_alpha], 'r.-', label="alpha={}, lambda={}".format(alpha, lamb))
            plt.plot(states, actual_state_values, 'b.-', label="actual state values")

    plt.xlabel("state", size=14)
    plt.ylabel("value Function", size=14)
    plt.legend()

