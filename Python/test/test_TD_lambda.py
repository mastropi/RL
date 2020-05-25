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
from unittest_data_provider import data_provider
#from gym.utils import seeding
import matplotlib.pyplot as plt

from Python.lib.environments import gridworlds
import Python.lib.agents as agents
from Python.lib.agents.policies import random_walks
from Python.lib.agents.learners import td
from Python.lib.agents.learners import AlphaUpdateType
import Python.lib.simulators as simulators

import test_utils

from importlib import reload
import Python.lib.agents.learners.td
#reload(Python.lib.agents.learners.td)
from Python.lib.agents.learners.td import LeaTDLambda
from Python.lib.agents.learners import Learner

#--
#import Python.lib.agents.policies.random_walks
#reload(random_walks)
#from Python.lib.agents.policies.random_walks import PolRandomWalkDiscrete

from Python.lib.environments import EnvironmentDiscrete

class Test_TD_Lambda(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1717
        self.nepisodes = 20
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

        # True state value function when gamma = 1.0
        cls.V_true = np.arange(-cls.nS-1, cls.nS+2, 2) / (cls.nS+1)
        cls.V_true[0] = cls.V_true[-1] = 0

        # Random walk policy on the above environment
        print("Check subclass")
        print(cls.env.__class__)
        print(issubclass(cls.env.__class__, EnvironmentDiscrete))

        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env)

    #-------------------------------------- DATA FOR TESTS ------------------------------------
    # Case number, description, expected value, parameters
    data_test_random_walk = lambda: (
            ( 1, 'TD(0), no alpha adjustment',
                 [0.000000, -0.987504, -0.941069, -0.900173, -0.805758, -0.613337, -0.407574,
                  -0.298142, -0.221312, -0.149854, -0.051759, 0.001865, 0.052824, 0.128984,
                  0.217968, 0.297497, 0.478097, 0.680931, 0.810098, 0.961462, 0.000000],
                (0.5, 1.0, 0.0), False, False, 0.0 ),
            ( 2, 'lambda<1, no alpha adjustment',
                 [0.000000, -0.780984, -0.650444, -0.519032, -0.289646, -0.173257, -0.109756,
                  -0.066946, -0.017813, -0.007125, 0.000394, 0.007759, 0.027384, 0.040343,
                  0.066892, 0.106094, 0.144276, 0.370783, 0.514059, 0.725371, 0.000000],
                (0.2, 0.9, 0.7), False, False, 0.0 ),
            ( 3, 'TD(1), no alpha adjustment (should be similar to MC every-visit, because lambda = 1)',
                 [0.000000, -0.191338, -0.299566, -0.338347, -0.405255, -0.542388, -0.703216,
                  -0.683582, -0.514695, -0.389836, -0.272784, -0.102481, 0.085268, 0.144083,
                  0.197441, 0.231908, 0.214645, 0.229342, 0.209256, 0.113961, 0.000000],
                (0.01, 1.0, 1.0), False, False, 0.0 ),
            ( 4, 'TD(0), alpha adjusted by state count',
                 [0.000000, -0.784147, -0.564636, -0.339216, -0.116705, -0.035015, -0.011703,
                  -0.004087, -0.000950, -0.000174, 0.000018, 0.000239, 0.000861, 0.002790,
                  0.013549, 0.042693, 0.114878, 0.291660, 0.494777, 0.853046, 0.000000],
                (2.0, 1.0, 0.0), True, False, 0.0 ),
            ( 5, 'TD(0), alpha adjusted by episode',
                  [0.000000, -0.972581, -0.884367, -0.763372, -0.511674, -0.286549, -0.197769,
                   -0.155286, -0.095648, -0.050941, -0.015981, 0.023179, 0.049375, 0.075978,
                   0.156952, 0.238841, 0.342924, 0.544935, 0.673823, 0.907726, 0.000000],
                (2.0, 1.0, 0.0), True,  True, 0.0 ),
            ( 6, 'lambda<1, adjusted alpha by state count',
                  [0.000000, -0.946909, -0.907296, -0.775429, -0.507633, -0.370504, -0.280516,
                   -0.210464, -0.131925, -0.069949, 0.005305, 0.067782, 0.116598, 0.165322,
                   0.247306, 0.315188, 0.428172, 0.604386, 0.732484, 0.908714, 0.000000],
                (2.0, 1.0, 0.7), True,  False, 0.0 ),
        )

    #------------------------------------------- TESTS ----------------------------------------
    @data_provider(data_test_random_walk)
    def test_random_walk(self, casenum, desc, expected, params_alpha_gamma_lambda,
                                                        adjust_alpha, adjust_alpha_by_episode, alpha_min):
        print("\nTesting " + self.id())

        # Learner and agent
        learner_tdlambda = td.LeaTDLambda(self.env, alpha=params_alpha_gamma_lambda[0],
                                                    gamma=params_alpha_gamma_lambda[1],
                                                    lmbda=params_alpha_gamma_lambda[2],
                                                    adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode, alpha_min=0.0,
                                                    debug=False)
        agent_rw_tdlambda = agents.GeneralAgent(self.policy_rw, learner_tdlambda)

        # Simulation
        sim = simulators.Simulator(self.env, agent_rw_tdlambda, debug=False)
        _, _, RMSE_by_episode, state_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                     compute_rmse=True, state_observe=15,
                                                     verbose=True, verbose_period=100,
                                                     plot=False, pause=0.1)
        observed = agent_rw_tdlambda.getLearner().getV().getValues()
        print("\nobserved: " + self.array2str(observed))
        assert np.allclose(observed, expected, atol=1E-6)

    def no_test_random_walk_onecase(self):
        print("\nTesting " + self.id())
 
        # Learner and agent
        params = dict({'alpha': 1.0,
                       'gamma': 1.0,
                       'lambda': 0.7,
                       'alpha_min': 0.0,
                       })
        learner_tdlambda = td.LeaTDLambda(self.env, alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                                                    alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,  # Every-visit is the default
                                                    adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=params['alpha_min'],
                                                    debug=False)
        agent_rw_tdlambda = agents.GeneralAgent(self.policy_rw, learner_tdlambda)

        # Simulation
        sim = simulators.Simulator(self.env, agent_rw_tdlambda, debug=False)
        _, _, RMSE_by_episode, state_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                     compute_rmse=True, state_observe=19,
                                                     verbose=True, verbose_period=100,
                                                     plot=False, pause=0.001)

        observed = agent_rw_tdlambda.getLearner().getV().getValues()
        print("\nobserved: " + self.array2str(observed))
        self.plot_results(params,
                          observed, self.V_true, RMSE_by_episode, state_info['alphas_by_episode'],
                          max_rmse=self.max_rmse, color_rmse=self.color_rmse, plotFlag=self.plotFlag)

    def no_test_random_walk_adaptive_onecase(self):
        print("\nTesting " + self.id())

        # Learner and agent
        params = dict({'alpha': 1.0,
                       'gamma': 1.0,
                       'lambda': np.nan,
                       'alpha_min': 0.0,
                       'lambda_min': 0.0,
                       })
        learner_tdlambda_adaptive = td.LeaTDLambdaAdaptive(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                                             alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,  # Every-visit is the default
                                                             adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=params['alpha_min'],
                                                             lambda_min=params['lambda_min'], burnin=False, debug=False)
        agent_rw_tdlambda_adaptive = agents.GeneralAgent(self.policy_rw, learner_tdlambda_adaptive)

        # Simulation        
        sim = simulators.Simulator(self.env, agent_rw_tdlambda_adaptive, debug=False)
        _, _, RMSE_by_episode, state_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                     compute_rmse=True,
                                                     verbose=True, verbose_period=100,
                                                     plot=False, pause=0.001)

        # Expected values with alpha = 0.2, gamma = 0.9, lambda = 0.8
        # seed = 1717, nepisodes=20, start_state = 9
        expected = np.array([ 0.,       -0.84353544, -0.78968225, -0.65385009, -0.55649136, -0.45755451,
 -0.36783142, -0.25143119, -0.2038609,  -0.08730269, -0.05066575,  0.01550673,
  0.04311948,  0.09924234,  0.16607023,  0.22774784,  0.36150155,  0.44464534,
  0.56831782,  0.70843306,  0.        ])
        observed = agent_rw_tdlambda_adaptive.getLearner().getV().getValues()
        print("\nobserved: " + self.array2str(observed))
        (ax, ax2) = self.plot_results(params,
                          observed, self.V_true, RMSE_by_episode, state_info['alphas_by_episode'],
                          y2label="(Average) alpha & lambda",
                          max_rmse=self.max_rmse, color_rmse=self.color_rmse, plotFlag=self.plotFlag)

        ax2.plot(np.arange(self.nepisodes)+1, agent_rw_tdlambda_adaptive.getLearner().lambda_mean_by_episode, color="orange")
        
        #assert np.allclose(observed, expected, atol=1E-6)
    #------------------------------------------- TESTS ----------------------------------------


if __name__ == "__main__":
    unittest.main()
