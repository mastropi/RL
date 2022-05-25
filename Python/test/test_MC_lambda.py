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
from unittest_data_provider import data_provider
#from gym.utils import seeding

from Python.lib.environments import gridworlds
from Python.lib.agents.policies import random_walks
import Python.lib.agents as agents
from Python.lib.agents.learners.episodic.discrete import mc
from Python.lib.agents.learners.episodic.discrete import AlphaUpdateType
import Python.lib.simulators as simulators

import test_utils

#reload(Python.lib.agents.learners.episodic.discrete.mc)
#reload(Python.lib.agents.learners.episodic.discrete)


class Test_MC_Lambda(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1717
        self.nepisodes = 20
        self.start_state = 10
        self.plot = True

    @classmethod
    def setUpClass(cls):    # cls is the class
                            # IMPORTANT: All attributes defined here can be then be referenced using self!
                            # (i.e. they belong to the "object" instantiated by this class)
        # Plot settings
        cls.max_rmse = 0.8
        cls.color_rmse = "blue"

        # Environment definition
        cls.nS = 19             # Number of non-terminal states in the 1D gridworld
        cls.env = gridworlds.EnvGridworld1D(length=cls.nS+2)  # nS states plus the two terminal states
        # True state value function when gamma = 1.0
        cls.V_true = np.arange(-cls.nS-1, cls.nS+2, 2) / (cls.nS+1)
        cls.V_true[0] = cls.V_true[-1] = 0
        
        # Random walk policy on the above environment
        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env)

    #-------------------------------------- DATA FOR TESTS ------------------------------------
    # Case number, description, expected value, parameters
    data_test_random_walk = lambda: (
            ( 1, 'MC, no alpha adjustment',
                 [-0.000000, -0.683051, -0.683051, -0.583526, -0.555566, -0.555566, -0.555566,
                  -0.400009, -0.358463, -0.319735, -0.187761, -0.102375, -0.056671, -0.005888,
                  0.125041, 0.125041, 0.207823, 0.207823, 0.343522, 0.569533, 0.000000],
                (0.1, 1.0, 1.0), False, False, 0.0 ),
            ( 2, 'L-return (lambda<1), no alpha adjustment',
                 [-0.000000, -0.648738, -0.440867, -0.228680, -0.172241, -0.093122, -0.023765,
                  -0.016186, -0.012195, -0.007552, 0.002211, 0.006526, 0.009155, 0.015080,
                  0.035585, 0.049038, 0.071389, 0.111463, 0.213302, 0.552812, 0.000000],
                (0.2, 1.0, 0.7), False, False, 0.0 ),
            ( 3, 'MC, adjusted alpha by state count (good results if run for 200 episodes)',
                 [-0.000000, -0.846154, -0.846154, -0.714286, -0.600000, -0.600000, -0.600000,
               -0.500000, -0.411765, -0.263158, -0.200000, -0.058824, -0.000000, 0.066667,
               0.230769, 0.230769, 0.333333, 0.333333, 0.600000, 1.000000, 0.000000],
                (1.0, 1.0, 1.0), True, False, 0.0 ),
            ( 4, 'MC, adjusted alpha by episode',
              [-0.000000, -0.702381, -0.702381, -0.611111, -0.518519, -0.518519, -0.518519,
               -0.438596, -0.368421, -0.263158, -0.200000, 0.043590, 0.113889, 0.196970,
               0.361176, 0.361176, 0.437451, 0.437451, 0.776471, 1.000000, 0.000000],
                (1.0, 1.0, 1.0), True,  True, 0.0 ),
            ( 5, 'L-return (lambda<1), adjusted alpha by state count',
              [-0.000000, -0.622132, -0.413372, -0.199682, -0.128330, -0.055435, -0.017840,
               -0.010864, -0.006091, -0.003125, 0.000993, 0.003451, 0.005608, 0.012755,
               0.028038, 0.039561, 0.058164, 0.104930, 0.264852, 0.694180, 0.000000],
                (1.0, 1.0, 0.7), True,  False, 0.0 ),
        )

    #------------------------------------------- TESTS ----------------------------------------
    @data_provider(data_test_random_walk)
    def test_random_walk(self, casenum, desc, expected, params_alpha_gamma_lambda,
                                                        adjust_alpha, adjust_alpha_by_episode, alpha_min):
        # All tests are run using seed = 1717, nepisodes = 20, start_state = 10
        print("\n*** Testing {0}, case number {1} ***".format(self.id(), casenum))
        learner_mclambda = mc.LeaMCLambda(self.env, alpha=params_alpha_gamma_lambda[0],
                                          gamma=params_alpha_gamma_lambda[1],
                                          lmbda=params_alpha_gamma_lambda[2],
                                          adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode,
                                          alpha_min=alpha_min,
                                          debug=False)
        agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mclambda)
        sim = simulators.Simulator(self.env, agent_rw_mc, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
                                            sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                    compute_rmse=True, state_observe=10,
                                                    verbose=True, verbose_period=100,
                                                    plot=False, pause=0.1)
        observed = agent_rw_mc.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        assert np.allclose(observed, expected, atol=1E-6)
                                            
    def test_random_walk_onecase(self):
        #-- All tests are run using seed = 1717, nepisodes = 20, start_state = 10
        print("\n*** Testing " + self.id() + " ***")

        # Learner and agent definition
        params = dict({'alpha': 1.0,
                       'gamma': 0.7,
                       'lambda': 0.8,
                       'alpha_min': 0.0,
                       })
        learner_mclambda = mc.LeaMCLambda(self.env, alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                                          alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,  # First-visit is the default
                                          adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=params['alpha_min'],
                                          debug=False)
        agent_rw_mclambda = agents.GenericAgent(self.policy_rw, learner_mclambda)

        # Simulation
        sim = simulators.Simulator(self.env, agent_rw_mclambda, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
                                            sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                    compute_rmse=True, state_observe=10,
                                                    verbose=True, verbose_period=100,
                                                    plot=False, pause=0.1)

        expected = np.array([ 0.000000, -0.471314, -0.150874, -0.044135, -0.021683,
                             -0.007209, -0.000881, -0.000375, -0.000158, -0.000065,
                              0.000034,  0.000097,  0.000210,  0.000553,  0.002688,
                              0.004801,  0.009370,  0.023406,  0.123022,  0.638221, 0.000000])
        observed = agent_rw_mclambda.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        if self.plot:
            self.plot_results(params,
                              observed, self.V_true, RMSE_by_episode, learning_info['alpha_mean'],
                              max_rmse=self.max_rmse, color_rmse=self.color_rmse)
        assert np.allclose(observed, expected, atol=1E-6)

    def fails_test_random_walk_adaptive_result(self):
        # NOTE: (2021/10/16) This test currently fails for two reasons:
        # - the `observed` array is an array of all -1.0 followed by all +1.0 (i.e. a step function)
        # - the self.plot_results() call at the end fails because the alpha_mean_by_episode_mean variable used when calling
        # plt.plot() inside self.plot_results() has zero length.
        # So, it seems that nothing is really done or learned by the learner_mclambda_adaptive object.
        print("\nTesting " + self.id())

        # Learner and agent definition
        params = dict({'alpha': 1.0,
                       'gamma': 1.0,
                       'lambda': 1.0,
                       'alpha_min': 0.0,
                       })
        learner_mclambda_adaptive = mc.LeaMCLambdaAdaptive(self.env, alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'])
        agent_rw_mclambda_adaptive = agents.GenericAgent(self.policy_rw, learner_mclambda_adaptive)

        # Simulation
        sim = simulators.Simulator(self.env, agent_rw_mclambda_adaptive, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                                     compute_rmse=True, state_observe=10,
                                                                     verbose=True, verbose_period=100,
                                                                     plot=False, pause=0.1)

        # Expected state values with alpha = 0.2, gamma = 0.9, lambda = 0.8
        # seed = 1717, nepisodes=20, start_state = 9
        expected = np.array([ -0.0,     -0.94788777, -0.93485068, -0.77635209, -0.66915289, -0.67045823,
 -0.6319687,  -0.52116762, -0.44295159, -0.20887109, -0.1027944,  -0.03800919,
 -0.03668617,  0.06142266,  0.27410733,  0.42610526,  0.50467228,  0.63018903,
  0.6727829,   0.72310919,  0.0        ])
        observed = learner_mclambda_adaptive.getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        if self.plot:
            self.plot_results(params,
                              observed, self.V_true, RMSE_by_episode, learning_info['alpha_mean_by_episode_mean'],
                              max_rmse=self.max_rmse, color_rmse=self.color_rmse)

        assert np.allclose(observed, expected, atol=1E-6)
    #------------------------------------------- TESTS ----------------------------------------


if __name__ == "__main__":
    unittest.main()
