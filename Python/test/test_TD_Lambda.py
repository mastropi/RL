# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:40:41 2020

@author: Daniel Mastropietro
@description: Write unit tests for the gridworld learning using TD(lambda)
"""

import runpy
runpy.run_path('../../setup.py')

import numpy as np
import unittest
#from gym.utils import seeding
import matplotlib.pyplot as plt

from Python.lib.environments import gridworlds
import Python.lib.agents as agents
from Python.lib.agents.policies import random_walks
from Python.lib.agents.learners import td
import Python.lib.simulators as simulators

import test_utils

from importlib import reload
import Python.lib.agents.learners.td
#reload(Python.lib.agents.learners.td)
from Python.lib.agents.learners.td import LeaTDLambda
#--
#import Python.lib.agents.policies.random_walks
#reload(random_walks)
#from Python.lib.agents.policies.random_walks import PolRandomWalkDiscrete

from Python.lib.environments import EnvironmentDiscrete

class Test_TD_Lambda(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1717
        self.nrounds = 20
        self.start_state = 10

    @classmethod
    def setUpClass(cls):    # cls is the class, in this case, class 'Test_TD_Lambda'
                            # IMPORTANT: All attributes defined here can be then be referenced using self!
                            # (i.e. they belong to the "object" instantiated by this class)
        cls.plotFlag = True
        cls.max_rmse = 0.8
        cls.color_rmse = "red"

        # Environment
        cls.nS = 19             # Number of non-terminal states in the 1D gridworld
        cls.env = gridworlds.EnvGridworld1D(length=cls.nS+2)  # nS states plus the two terminal states

        # True state value functions
        cls.V_true = np.arange(-cls.nS-1, cls.nS+2, 2) / (cls.nS+1)
        cls.V_true[0] = cls.V_true[-1] = 0

        # Agents with Policy and Learner defined
        print("Check subclass")
        print(cls.env.__class__)
        print(issubclass(cls.env.__class__, EnvironmentDiscrete))

        cls.alpha_min = 0.008
        cls.rw = random_walks.PolRandomWalkDiscrete(cls.env)
        cls.tdlambda = td.LeaTDLambda(cls.env, alpha=0.8, gamma=1.0, lmbda=0.0,
                                      adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=cls.alpha_min,
                                      debug=False)
        cls.tdlambda_adaptive = td.LeaTDLambdaAdaptive(cls.env, alpha=0.8, gamma=1.0, lmbda=0.8,
                                                       adjust_alpha=True, adjust_alpha_by_episode=True, alpha_min=cls.alpha_min,
                                                       lambda_min=0., burnin=False, debug=False)
        cls.mclambda = td.LeaTDLambda(cls.env, alpha=0.2, gamma=0.9, lmbda=1.0)
        cls.agent_rw_tdlambda = agents.GeneralAgent(cls.rw, cls.tdlambda)
        cls.agent_rw_tdlambda_adaptive = agents.GeneralAgent(cls.rw, cls.tdlambda_adaptive)
        cls.agent_rw_mclambda = agents.GeneralAgent(cls.rw, cls.mclambda)

    #------------------------------------------- TESTS ----------------------------------------
    def test_random_walk_result(self):
        "Test using my TD(Lambda) learner"
        print("\nTesting " + self.id())
        sim = simulators.Simulator(self.env, self.agent_rw_tdlambda, debug=False)
        _, _, RMSE_by_episode, state_info = sim.play(nrounds=self.nrounds, start=self.start_state, seed=self.seed,
                                                     compute_rmse=True,
                                                     verbose=True, verbose_period=100,
                                                     plot=False, pause=0.1)

        # Expected values with alpha = 0.2, gamma = 0.9, lambda = 0.8,
        # seed = 1717, nrounds=20, start_state = 9
        expected = np.array([ 0.,     -0.96263354, -0.94935992, -0.79580974, -0.68319293, -0.69444404,
 -0.66004513, -0.54932031, -0.45282584, -0.24578133, -0.18307401, -0.12925566,
 -0.12743421,  0.00243922,  0.28248756,  0.47626462,  0.50983006,  0.63976455,
  0.69107498,  0.74657348,  0.        ])
        observed = self.tdlambda.getV().getValues()

        print("\nobserved: " + self.array2str(observed))

        self.plot_results(observed, self.V_true, RMSE_by_episode,
                          state_info['alphas_by_episode'], self.alpha_min,
                          max_rmse=self.max_rmse, color_rmse=self.color_rmse, plotFlag=self.plotFlag)

        assert np.allclose( expected, observed )

    def no_test_random_walk_as_MC_result(self):
        "Test using my TD(Lambda) with lambda = 1 to emulate a Monte Carlo learner"
        print("\nTesting " + self.id())
        sim = simulators.Simulator(self.env, self.agent_rw_mclambda, debug=False)
        _, _, RMSE_by_episode, state_info = sim.play(nrounds=self.nrounds, start=self.start_state, seed=self.seed,
                                                     compute_rmse=True,
                                                     verbose=True, verbose_period=100,
                                                     plot=False, pause=0.1)

        # Expected values with alpha = 0.2, gamma = 0.9, lambda = 0.8
        # seed = 1717, nrounds=20, start_state = 9
        expected = np.array([ 0.,    -0.87570279, -0.77617219, -0.38677509, -0.20086812, -0.26264111,
 -0.20460859, -0.13041183, -0.10991275, -0.03513174, -0.02629319, -0.04773745,
 -0.06768075, -0.00798058,  0.23183274,  0.35276211,  0.37104584,  0.52515039,
  0.50912581,  0.64218869,  0.        ])
        observed = self.mclambda.getV().getValues()

        print("\nobserved: " + self.array2str(observed))

        self.plot_results(observed, self.V_true, RMSE_by_episode,
                          state_info['alphas_by_episode'], self.alpha_min,
                          max_rmse=self.max_rmse, color_rmse=self.color_rmse, plotFlag=self.plotFlag)

        assert np.allclose( expected, observed )

    def no_test_random_walk_adaptive_result(self):
        "Test using my TD(Lambda) learner"
        print("\nTesting " + self.id())
        sim = simulators.Simulator(self.env, self.agent_rw_tdlambda_adaptive, debug=False)
        _, _, RMSE_by_episode, state_info = sim.play(nrounds=self.nrounds, start=self.start_state, seed=self.seed,
                                                     compute_rmse=True,
                                                     verbose=True, verbose_period=100,
                                                     plot=False, pause=0.1)

        # Expected values with alpha = 0.2, gamma = 0.9, lambda = 0.8
        # seed = 1717, nrounds=20, start_state = 9
        expected = np.array([ 0.,       -0.84353544, -0.78968225, -0.65385009, -0.55649136, -0.45755451,
 -0.36783142, -0.25143119, -0.2038609,  -0.08730269, -0.05066575,  0.01550673,
  0.04311948,  0.09924234,  0.16607023,  0.22774784,  0.36150155,  0.44464534,
  0.56831782,  0.70843306,  0.        ])
        observed = self.tdlambda_adaptive.getV().getValues()

        print("\nobserved: " + self.array2str(observed))

        self.plot_results(observed, self.V_true, RMSE_by_episode,
                          state_info['alphas_by_episode'], self.alpha_min,
                          max_rmse=self.max_rmse, color_rmse=self.color_rmse, plotFlag=self.plotFlag)

        assert np.allclose( expected, observed )
    #------------------------------------------- TESTS ----------------------------------------



if __name__ == "__main__":
    unittest.main()
