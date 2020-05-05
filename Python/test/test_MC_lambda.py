# -*- coding: utf-8 -*-
"""
Created on Sat Apr  11 15:11:21 2020

@author: Daniel Mastropietro
@description: Write unit tests learning with MC(lambda)
"""

import runpy
runpy.run_path('../../setup.py')

import numpy as np
import unittest
#from gym.utils import seeding
import matplotlib.pyplot as plt

from Python.lib.environments import gridworlds
from Python.lib.agents.policies import random_walks
import Python.lib.agents as agents
from Python.lib.agents.learners import mc
import Python.lib.simulators as simulators

import test_utils

from importlib import reload
import Python.lib.agents.learners.mc
#reload(Python.lib.agents.learners.mc)
from Python.lib.agents.learners.mc import LeaMCLambda


class Test_MC_Lambda(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1717
        self.nrounds = 1000
        self.start_state = 10

    @classmethod
    def setUpClass(cls):    # cls is the class
                            # IMPORTANT: All attributes defined here can be then be referenced using self!
                            # (i.e. they belong to the "object" instantiated by this class)
        cls.plotFlag = True
        cls.max_rmse = 0.8
        cls.color_rmse = "blue"

        # Environment definition
        cls.nS = 19             # Number of non-terminal states in the 1D gridworld
        cls.env = gridworlds.EnvGridworld1D(length=cls.nS+2)  # nS states plus the two terminal states
        # True state value functions
        cls.V_true = np.arange(-cls.nS-1, cls.nS+2, 2) / (cls.nS+1)
        cls.V_true[0] = cls.V_true[-1] = 0

        # Agent with Policy and Learner defined
        cls.alpha_min = 0.0
        cls.rw = random_walks.PolRandomWalkDiscrete(cls.env)
#        cls.mclambda = mc.LeaMCLambda(cls.env, alpha=0.2, gamma=1.0, lmbda=0.8,
        cls.mclambda = mc.LeaMCLambda(cls.env, alpha=0.1, gamma=1.0, lmbda=1.0,
                                      adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=cls.alpha_min,
                                      debug=False)
        cls.mclambda_adaptive = mc.LeaMCLambdaAdaptive(cls.env, alpha=0.2, gamma=1.0, lmbda=0.8)
        cls.agent_rw_mclambda = agents.GeneralAgent(cls.rw, cls.mclambda)
        cls.agent_rw_mclambda_adaptive = agents.GeneralAgent(cls.rw, cls.mclambda_adaptive)


    #------------------------------------------- TESTS ----------------------------------------
    def test_random_walk_result(self):
        print("\nTesting " + self.id())
        sim = simulators.Simulator(self.env, self.agent_rw_mclambda, debug=False)
        state_values, state_counts, RMSE_by_episode, state_info = \
                                            sim.play(nrounds=self.nrounds, start=self.start_state, seed=self.seed,
                                                     compute_rmse=True, state_observe=10,
                                                     verbose=True, verbose_period=100,
                                                     plot=False, pause=0.1)
        # Expected state values with alpha = 0.2, gamma = 0.9, lambda = 0.8
        # seed = 1717, nrounds=20, start_state = 9
        expected = np.array([ 0.,     -0.94788777, -0.93485068, -0.77635209, -0.66915289, -0.67045823,
 -0.6319687,  -0.52116762, -0.44295159, -0.20887109, -0.1027944,  -0.03800919,
 -0.03668617,  0.06142266,  0.27410733,  0.42610526,  0.50467228,  0.63018903,
  0.6727829,   0.72310919,  0.        ])
    
        # Expected state values with alpha = 0.1, gamma = 1.0, lambda = 1.0
        # seed = 1717, nrounds=100, start_state = 10
        # FIRST-VISIT: update alpha by state count
        expected_mc_pure_fv = ([ 0.,         -0.83430688, -0.60373791, -0.47460169, -0.2478452,  -0.21153757,
 -0.12676933, -0.00891952, -0.00501395,  0.01781175,  0.02322498,  0.15217184,
  0.20386114,  0.19907104,  0.20737094,  0.23519332,  0.30024595,  0.132892,
  0.42818265,  0.88771222,  0.        ])
        # EVERY-VISIT: update alpha by state count
        expected_mc_pure_ev = ([ 0.,         -0.93996951, -0.89884883, -0.85734574, -0.65658552, -0.64093495,
 -0.29331159,  0.23594949,  0.17946993,  0.24776283,  0.44740875,  0.55121643,
  0.59742296,  0.66166587,  0.71916619,  0.71328366,  0.67221682,  0.29196231,
  0.74890003,  0.96467199,  0.        ])
        # FIRST-VISIT: update alpha by episode
        expected_mc_pure_fv_byepi = ([0.000000,-0.435619,-0.431409,-0.404428,-0.338544,-0.327813,
                                -0.323859,-0.308399,-0.202475,-0.105022,-0.083093,0.042541,
                                0.070933,0.121618,0.199247,0.214955,0.232572,0.232572,
                                0.437373,0.480894,0.000000])
    
        observed = self.mclambda.getV().getValues() # This should be the same as state_values above

        print("\nobserved: " + self.array2str(observed))

        self.plot_results(observed, self.V_true, RMSE_by_episode,
                          state_info['alphas_by_episode'], self.alpha_min,
                          max_rmse=self.max_rmse, color_rmse=self.color_rmse, plotFlag=self.plotFlag)

        assert np.allclose( expected, observed )

    def no_test_random_walk_adaptive_result(self):
        print("\nTesting " + self.id())
        sim = simulators.Simulator(self.env, self.agent_rw_mclambda_adaptive, debug=False)
        _, _, RMSE_by_episode, state_info = sim.play(nrounds=self.nrounds, start=self.start_state, seed=self.seed,
                                                     compute_rmse=True,
                                                     verbose=True, verbose_period=100,
                                                     plot=False, pause=0.1)

        # Expected state values with alpha = 0.2, gamma = 0.9, lambda = 0.8
        # seed = 1717, nrounds=20, start_state = 9
        expected = np.array([ 0.,     -0.94788777, -0.93485068, -0.77635209, -0.66915289, -0.67045823,
 -0.6319687,  -0.52116762, -0.44295159, -0.20887109, -0.1027944,  -0.03800919,
 -0.03668617,  0.06142266,  0.27410733,  0.42610526,  0.50467228,  0.63018903,
  0.6727829,   0.72310919,  0.        ])
        observed = self.mclambda_adaptive.getV().getValues()

        print("\nobserved: " + self.array2str(observed))

        self.plot_results(observed, self.V_true, RMSE_by_episode,
                          state_info['alphas_by_episode'], self.alpha_min,
                          max_rmse=self.max_rmse, color_rmse=self.color_rmse, plotFlag=self.plotFlag)

        assert np.allclose( expected, observed )
    #------------------------------------------- TESTS ----------------------------------------



if __name__ == "__main__":
    unittest.main()
