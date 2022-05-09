# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:40:41 2020

@author: Daniel Mastropietro
@description: Write unit tests for the gridworld learning using TD(lambda)
"""

import runpy
runpy.run_path('../../setup.py')

import os
import copy
import numpy as np
import unittest
from unittest_data_provider import data_provider
#from gym.utils import seeding
from matplotlib import cm
from timeit import default_timer as timer

import pickle

from Python.lib.environments import gridworlds, mountaincars
import Python.lib.agents as agents
from Python.lib.agents.policies import random_walks
import Python.lib.agents.learners.episodic.discrete.td as td
from Python.lib.agents.learners.episodic.discrete import AlphaUpdateType
import Python.lib.simulators as simulators

import test_utils

# In case we need to reload the td.py module
#import agents.learners.episodic.discrete.td
#reload(Python.lib.agents.learners.episodic.discrete.td)

#--
#import Python.lib.agents.policies.random_walks
#reload(random_walks)
#from Python.lib.agents.policies.random_walks import PolRandomWalkDiscrete

from Python.lib.environments import EnvironmentDiscrete

class Test_TD_Lambda_GW1D(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1717
        self.nepisodes = 20
        self.start_state = 10
        self.plot = True

    @classmethod
    def setUpClass(cls):    # cls is the class, in this case, class 'Test_TD_Lambda'
                            # IMPORTANT: All attributes defined here can be then be referenced using self!
                            # (i.e. they belong to the "object" instantiated by this class)
        cls.max_rmse = 0.8
        cls.color_rmse = "red"

        # Environment: 1D gridworld
        cls.nS = 19         # Number of non-terminal states in the 1D gridworld
        cls.env = gridworlds.EnvGridworld1D(length=cls.nS+2)  # nS states plus the two terminal states

        # Random walk policy on the above environment
        print("Check subclass")
        print(cls.env.__class__)
        print(issubclass(cls.env.__class__, EnvironmentDiscrete))

        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env)

    #-------------------------------------- DATA FOR TESTS ------------------------------------
    # Case number, description, expected value, parameters
    data_test_random_walk = lambda: (
            ( 1, 'TD(0), no alpha adjustment',
                 [-0.000000, -0.987504, -0.941069, -0.900173, -0.805758, -0.613337, -0.407574,
                  -0.298142, -0.221312, -0.149854, -0.051759, 0.001865, 0.052824, 0.128984,
                  0.217968, 0.297497, 0.478097, 0.680931, 0.810098, 0.961462, 0.000000],
                (0.5, 1.0, 0.0), False, False, 0.0 ),
            ( 2, 'lambda<1, no alpha adjustment',
                 [-0.000000, -0.780984, -0.650444, -0.519032, -0.289646, -0.173257, -0.109756,
                  -0.066946, -0.017813, -0.007125, 0.000394, 0.007759, 0.027384, 0.040343,
                  0.066892, 0.106094, 0.144276, 0.370783, 0.514059, 0.725371, 0.000000],
                (0.2, 0.9, 0.7), False, False, 0.0 ),
            ( 3, 'TD(1), no alpha adjustment (should be similar to MC every-visit, because lambda = 1)',
                 [-0.000000, -0.191338, -0.299566, -0.338347, -0.405255, -0.542388, -0.703216,
                  -0.683582, -0.514695, -0.389836, -0.272784, -0.102481, 0.085268, 0.144083,
                  0.197441, 0.231908, 0.214645, 0.229342, 0.209256, 0.113961, 0.000000],
                (0.01, 1.0, 1.0), False, False, 0.0 ),
            ( 4, 'TD(0), alpha adjusted by state count',
                 [-0.000000, -0.784147, -0.564636, -0.339216, -0.116705, -0.035015, -0.011703,
                  -0.004087, -0.000950, -0.000174, 0.000018, 0.000239, 0.000861, 0.002790,
                  0.013549, 0.042693, 0.114878, 0.291660, 0.494777, 0.853046, 0.000000],
                (2.0, 1.0, 0.0), True, False, 0.0 ),
            ( 5, 'TD(0), alpha adjusted by episode',
                  [-0.000000, -0.972581, -0.884367, -0.763372, -0.511674, -0.286549, -0.197769,
                   -0.155286, -0.095648, -0.050941, -0.015981, 0.023179, 0.049375, 0.075978,
                   0.156952, 0.238841, 0.342924, 0.544935, 0.673823, 0.907726, 0.000000],
                (2.0, 1.0, 0.0), True,  True, 0.0 ),
            ( 6, 'lambda<1, adjusted alpha by state count',
                  [-0.000000, -0.946909, -0.907296, -0.775429, -0.507633, -0.370504, -0.280516,
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
                                          adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode, alpha_min=alpha_min,
                                          debug=False)
        agent_rw_tdlambda = agents.GenericAgent(self.policy_rw, learner_tdlambda)

        # Simulation
        sim = simulators.Simulator(self.env, agent_rw_tdlambda, debug=False)
        _, _, RMSE_by_episode, state_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                     compute_rmse=True, state_observe=15,
                                                     verbose=True, verbose_period=100,
                                                     plot=False, pause=0.1)
        observed = agent_rw_tdlambda.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        assert np.allclose(observed, expected, atol=1E-6)

    def test_random_walk_onecase(self):
        print("\nTesting " + self.id())
 
        # Learner and agent
        params = dict({'alpha': 0.3,
                       'gamma': 1.0,
                       'lambda': 0.7,
                       'alpha_min': 0.0,
                       })
        learner_tdlambda = td.LeaTDLambda(self.env, alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                                          alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,  # Every-visit is the default
                                          adjust_alpha=False, adjust_alpha_by_episode=False, alpha_min=params['alpha_min'],
                                          debug=False)
        agent_rw_tdlambda = agents.GenericAgent(self.policy_rw, learner_tdlambda)

        # Simulation
        sim = simulators.Simulator(self.env, agent_rw_tdlambda, debug=False)
        _, _, RMSE_by_episode, state_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                     compute_rmse=True, state_observe=19,
                                                     verbose=True, verbose_period=100,
                                                     plot=False, pause=0.001)

        # Expected values with the close to optimum values (for minimum RMSE) shown in Sutton:
        # 19-size gridworld
        # alpha = 0.3, gamma = 1.0, lambda = 0.7
        # adjust_alpha = False, adjust_alpha_by_episode = False
        # seed = 1717, nepisodes=20, start_state = 10
        expected = np.array([-0.000000, -0.973693, -0.966924, -0.939277, -0.840629,
                             -0.758148, -0.657346, -0.517419, -0.278471, -0.213379,
                             -0.121619, -0.048246,  0.105097,  0.170801,  0.262924,
                              0.406143,  0.466590,  0.759199,  0.852375,  0.898508, 0.000000])

        observed = agent_rw_tdlambda.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        print("Average RMSE over {} episodes: {:.3f}".format(self.nepisodes, np.mean(RMSE_by_episode)))
        if self.plot:
            self.plot_results(params,
                              observed, self.env.getV(), RMSE_by_episode, state_info['alphas_by_episode'],
                              y2lim=(0, 1.0),
                              max_rmse=self.max_rmse, color_rmse=self.color_rmse)

        assert np.allclose(observed, expected, atol=1E-6)

    def test_random_walk_rmse_twice(self):
        print("\nTesting " + self.id())

        # Learner and agent
        params = dict({'alpha': 0.3,
                       'gamma': 1.0,
                       'lambda': 0.7,
                       'alpha_min': 0.0,
                       })
        learner_tdlambda = td.LeaTDLambda(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                          lmbda=params['lambda'],
                                          alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                                          # Every-visit is the default
                                          adjust_alpha=False, adjust_alpha_by_episode=False,
                                          alpha_min=params['alpha_min'],
                                          debug=False)
        agent_rw_tdlambda = agents.GenericAgent(self.policy_rw, learner_tdlambda)

        # Simulation
        sim = simulators.Simulator(self.env, agent_rw_tdlambda, debug=False)

        # First run
        _, _, RMSE_by_episode, _ = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                    compute_rmse=True, state_observe=19,
                                                    verbose=True, verbose_period=100,
                                                    plot=False, pause=0.001)

        # Expected RMSE with the following settings:
        # 19-size gridworld
        # alpha = 0.3, gamma = 1.0, lambda = 0.7
        # adjust_alpha = False, adjust_alpha_by_episode = False
        # seed = 1717, nepisodes=20, start_state = 10
        rmse_expected = 0.2192614

        rmse_observed = np.mean(RMSE_by_episode)
        print("First run: average RMSE over {} episodes: {:.8f}".format(self.nepisodes, rmse_observed))
        assert np.allclose(rmse_observed, rmse_expected, atol=1E-6)

        # Second run
        _, _, RMSE_by_episode, _ = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                    compute_rmse=True, state_observe=19,
                                                    verbose=True, verbose_period=100,
                                                    plot=False, pause=0.001)
        rmse_observed = np.mean(RMSE_by_episode)
        print("Second run: average RMSE over {} episodes: {:.8f}".format(self.nepisodes, rmse_observed))
        assert np.allclose(rmse_observed, rmse_expected, atol=1E-6)

    def test_random_walk_adaptive_onecase(self):
        print("\nTesting " + self.id())

        # Learner and agent
        # (we use the same parameters as for the test case in dataprovider with labmda = 0.7
        params = dict({'alpha': 2.0,
                       'gamma': 1.0,
                       'lambda': np.nan,
                       'alpha_min': 0.0,
                       'lambda_min': 0.0,
                       'lambda_max': 1.0
                       })
        learner_tdlambda_adaptive = td.LeaTDLambdaAdaptive(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                                           alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,  # Every-visit is the default
                                                           adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=params['alpha_min'],
                                                           lambda_min=params['lambda_min'], lambda_max=params['lambda_max'],
                                                           adaptive_type=td.AdaptiveLambdaType.ATD,
                                                           burnin=False, debug=False)
        agent_rw_tdlambda_adaptive = agents.GenericAgent(self.policy_rw, learner_tdlambda_adaptive)

        # Simulation        
        sim = simulators.Simulator(self.env, agent_rw_tdlambda_adaptive, debug=False)
        _, _, RMSE_by_episode, state_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                     compute_rmse=True,
                                                     verbose=True, verbose_period=100,
                                                     plot=False, pause=0.001)

        # Expected values with: (we use the same parameters as with the above test case in dataprovider using lambda = 0.7)
        # 19-size gridworld
        # alpha = 2.0, gamma = 1.0, lambda_min = 0.0, lambda_max = 1.0
        # alpha_update_type = AlphaUpdateType.EVERY_STATE_VISIT
        # adjust_alpha = True, adjust_alpha_by_episode = False
        # seed = 1717, nepisodes=20, start_state = 9
        # lambda as the Boltzmann function of delta(t) / average( abs(V(t)) )
        expected = np.array([-0.000000, -0.914800, -0.841925, -0.664785, -0.392792,
                             -0.249244, -0.154211, -0.093055, -0.041201, -0.016698,
                             -0.006381, -0.005999, -0.009759, -0.013459, -0.009742,
                              0.012727,  0.100282,  0.307882,  0.419977,  0.775499, 0.000000])

        observed = agent_rw_tdlambda_adaptive.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        print("Average RMSE over {} episodes: {:.3f}".format(self.nepisodes, np.mean(RMSE_by_episode)))
        if self.plot:
            (ax, ax2) = self.plot_results(params,
                              observed, self.env.getV(), RMSE_by_episode, state_info['alphas_by_episode'],
                              y2label="(Average) alpha & lambda", y2lim=(0, 1.0),
                              max_rmse=self.max_rmse, color_rmse=self.color_rmse)
            ax2.plot(np.arange(self.nepisodes)+1, agent_rw_tdlambda_adaptive.getLearner().lambda_mean_by_episode, color="orange")

        #input("Press Enter...")

        assert np.allclose(observed, expected, atol=1E-6)
    #------------------------------------------- TESTS ----------------------------------------


class Test_TD_Lambda_GW2D(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1717
        self.nepisodes = 300
        self.start_state = None
        self.colormap = cm.get_cmap("rainbow")  # useful colormaps are "jet", "rainbow", seismic"
        self.pause = 0.001
        self.plot = True

    @classmethod
    def setUpClass(cls):    # cls is the class, in this case, class 'Test_TD_Lambda'
                            # IMPORTANT: All attributes defined here can be then be referenced using self!
                            # (i.e. they belong to the "object" instantiated by this class)
        cls.max_rmse = 0.8
        cls.color_rmse = "red"

        ######################## Environments: 2D gridworld
        #----- Simple environment with two terminal states with reward -1 and reward +1
        cls.ny = 5
        cls.nx = 5
        cls.nS = cls.ny * cls.nx
        # Define the fontsize to use when displaying state value function in the image
        # plotted by test_utils.plot_results_2D().
        fontsize_max = 20
        cls.fontsize = int( np.min((5/cls.ny, 5/cls.nx)) * fontsize_max )
            ## Fontsize is a 1D quantity => we take into account the two dimensions ny and nx
            ## (in case one is smaller than the other dimension)
            ## by computing the two proportions 5/ny and 5/nx
            ## where 5*5 is a reference area where the given fontsize works fine (14)
            ## The formula for the proportion comes from min(Lx, Ly) / Lref
            ## where Lx = M/nx, Ly = M/ny and Lref = M/5, where M is the size
            ## each side of the square where the image is shown by test_utils.plot_results_2D()
            ## which is supposed to be fix as the number of tiles is increased
            ## from e.g. 5x5 to 15x15.
            ## Note however that, when ny != nx, the square where the image is shown
            ## becomes a rectangle.
        cls.env = gridworlds.EnvGridworld2D(shape=(cls.ny, cls.nx))
        # Random walk policy on the above environment
        # (we need the environment because:
        # - we need to know the number of possible actions to choose from, and
        # - MORE IMPORTANTLY we need to use the random number generator defined in the environment
        # to choose an action!! (this will define the deterministic aspect of each experiment...)
        cls.policy_rw_env = random_walks.PolRandomWalkDiscrete(cls.env)

        #----- Environment with log(n) rewards whose positions and values are chosen randomly
        seed_env = 13
        np.random.seed(seed_env)
        n_terminal_states = int( round( np.log( cls.nS ) ) )
        terminal_states = np.random.choice(range(cls.nS), n_terminal_states, replace=False)
        terminal_rewards = np.random.choice([-1, 1], n_terminal_states, replace=True)
        rewards_dict = dict( zip(terminal_states, terminal_rewards) )
        print("Rewards for logn environment (n={}, log(n)={}): {}".format(cls.nS, len(rewards_dict), rewards_dict))
        cls.env_logn_rewards = gridworlds.EnvGridworld2D(shape=(cls.ny, cls.nx),
                                                         terminal_states=terminal_states,
                                                         rewards_dict=rewards_dict)
        cls.env_logn_rewards._render()
        # Random walk policy on the above environment
        cls.policy_rw_envlogn = random_walks.PolRandomWalkDiscrete(cls.env_logn_rewards)

    def test_random_walk(self):
        print("\nTesting " + self.id())
 
        # Learner and agent
        params = dict({'alpha': 1.0,
                       'gamma': 1.0,
                       'lambda': 0.7,
                       'alpha_min': 0.0,
                       'nepisodes': self.nepisodes,
                       })
        learner_tdlambda = td.LeaTDLambda(self.env, alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                                          alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,  # Every-visit is the default
                                          adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=params['alpha_min'],
                                          debug=False)
        agent_rw_tdlambda = agents.GenericAgent(self.policy_rw_env, learner_tdlambda)

        # Simulation
        sim = simulators.Simulator(self.env, agent_rw_tdlambda, debug=False)
        _, _, RMSE_by_episode, state_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                     compute_rmse=False, state_observe=0,
                                                     verbose=True, verbose_period=int(self.nepisodes/10),
                                                     plot=True, colormap=self.colormap, pause=self.pause)

        # Expected 2D state value function given as 1D array with:
        # 2D grid = 5 x 5
        # alpha = 1.0, gamma = 1.0, lambda = 0.7, alpha_min = 0.0
        # alpha_update_type = AlphaUpdateType.FIRST_STATE_VISIT
        # adjust_alpha = True, adjust_alpha_by_episode = False
        # seed = 1717, nepisodes = 300, start_state = None
        expected_values = np.array([-0.000000,	-0.600664,	-0.340097,	-0.199044,	-0.109807,
                                    -0.598136,	-0.440290,	-0.250745,	-0.110556,	-0.009195,
                                    -0.335323,	-0.245785,	-0.109325,	 0.057424,	 0.185493,
                                    -0.163590,	-0.082114,	 0.035308,	 0.267254,	 0.538616,
                                    -0.084325,	-0.000746,	 0.142019,	 0.426486,	 0.000000,
                                    ])

        print("Agent ends at state {}:".format(self.env.getState()))
        self.env._render()
        observed_values = agent_rw_tdlambda.getLearner().getV().getValues()
        #print("\nobserved values: %s" %(test_utils.array2str(observed_values)) )
        observed = np.asarray( observed_values ).reshape(self.ny, self.nx)
        print("\nobserved: ")
        print(observed)
        if self.plot:
            test_utils.plot_results_2D(observed, params, colormap=self.colormap, fontsize=self.fontsize)

        assert np.allclose(observed_values, expected_values, atol=1E-6)

    def test_random_walk_logn_rewards(self):
        print("\nTesting " + self.id())
 
        # Learner and agent
        params = dict({'alpha': 1.0,
                       'gamma': 1.0,
                       'lambda': 0.7,
                       'alpha_min': 0.0,
                       'nepisodes': self.nepisodes,
                       })
        learner_tdlambda = td.LeaTDLambda(self.env_logn_rewards, alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                                          alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,  # Every-visit is the default
                                          adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=params['alpha_min'],
                                          debug=False)
        agent_rw_tdlambda = agents.GenericAgent(self.policy_rw_envlogn, learner_tdlambda)

        # Simulation
        sim = simulators.Simulator(self.env_logn_rewards, agent_rw_tdlambda, debug=False)
        _, _, RMSE_by_episode, state_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                     compute_rmse=False, state_observe=0,
                                                     verbose=True, verbose_period=int(self.nepisodes/10),
                                                     plot=True, colormap=self.colormap, pause=self.pause)

        # Expected 2D state value function given as 1D array with:
        # 2D grid = 5 x 5
        # alpha = 1.0, gamma = 1.0, lambda = 0.7, alpha_min = 0.0
        # alpha_update_type = AlphaUpdateType.FIRST_STATE_VISIT
        # adjust_alpha = True, adjust_alpha_by_episode = False
        # seed = 1717, nepisodes = 300, start_state = None
        expected_values = np.array([-0.586914,	-0.000000,	-0.275543,	 0.534306,	 0.529651,
                                    -0.447301,	-0.386761,	 0.160712,	 0.000000,	 0.496648,
                                    -0.263472,	-0.210891,	-0.001141,	 0.264747,	 0.124014,
                                    -0.223334,	-0.210347,	-0.169349,	-0.249459,	-0.393667,
                                    -0.192741,	-0.196232,	-0.272632,	-0.510045,	-0.000000
                                    ])

        print("Agent ends at state {}:".format(self.env_logn_rewards.getState()))
        self.env_logn_rewards._render()
        observed_values = agent_rw_tdlambda.getLearner().getV().getValues()
        #print("\nobserved values: %s" %(test_utils.array2str(observed_values)) )
        observed = np.asarray( observed_values ).reshape(self.ny, self.nx)
        print("\nobserved: ")
        print(observed)
        if self.plot:
            test_utils.plot_results_2D(observed, params, colormap=self.colormap, fontsize=self.fontsize)

        assert np.allclose(observed_values, expected_values, atol=1E-6)
    ####################### TD(lambda) #######################


    ####################### ADAPTIVE TD(lambda) #######################
    def test_random_walk_adaptive(self):
        print("\nTesting " + self.id())

        # Learner and agent
        params = dict({'alpha': 1.0,
                       'gamma': 1.0,
                       'lambda': np.nan,
                       'alpha_min': 0.0,
                       'lambda_min': 0.0,
                       'lambda_max': 1.0,
                       'nepisodes': self.nepisodes,
                       })
        learner_tdlambda_adaptive = td.LeaTDLambdaAdaptive(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                                           alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,  # Every-visit is the default
                                                           adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=params['alpha_min'],
                                                           lambda_min=params['lambda_min'], lambda_max=params['lambda_max'],
                                                           burnin=False,
                                                           plotwhat="average", fontsize=int(self.fontsize*0.8),
                                                           debug=False)
        agent_rw_tdlambda_adaptive = agents.GenericAgent(self.policy_rw_env, learner_tdlambda_adaptive)

        # Simulation        
        sim = simulators.Simulator(self.env, agent_rw_tdlambda_adaptive, debug=False)
        _, _, RMSE_by_episode, state_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                     compute_rmse=False,
                                                     verbose=True, verbose_period=int(self.nepisodes/10),
                                                     plot=True, colormap=self.colormap, pause=self.pause)

        # Expected 2D state value function given as 1D array with:
        # 2D grid = 5 x 5
        # alpha = 1.0, gamma = 1.0, alpha_min = 0.0, lambda_min = 0.0, lambda_max = 1.0
        # alpha_update_type = AlphaUpdateType.FIRST_STATE_VISIT
        # adjust_alpha = True, adjust_alpha_by_episode = False
        # seed = 1717, nepisodes = 300, start_state = None
        # lambda as the Boltzmann function of delta(t) / average( abs(V(t)) )
        expected_values = np.array([-0.000000,	-0.621289,	-0.356950,	-0.197688,	-0.095204,
                                    -0.597418,	-0.449396,	-0.250112,	-0.100089,	 0.021283,
                                    -0.330550,	-0.235789,	-0.092370,	 0.075680,	 0.224815,
                                    -0.153306,	-0.058187,	 0.074201,	 0.301015,	 0.564076,
                                    -0.051561,	 0.038582,	 0.192242,	 0.472891,	 0.000000
                                    ])

        print("Agent ends at state {}:".format(self.env.getState()))
        self.env._render()
        observed_values = agent_rw_tdlambda_adaptive.getLearner().getV().getValues()
        #print("\nobserved values: %s" %(test_utils.array2str(observed_values)) )
        observed = np.asarray( observed_values ).reshape(self.ny, self.nx)
        print("\nobserved: ")
        print(observed)
        if self.plot:
            test_utils.plot_results_2D(observed, params, colormap=self.colormap, fontsize=self.fontsize)

        assert np.allclose(observed_values, expected_values, atol=1E-6)

    def test_random_walk_adaptive_logn_rewards(self):
        print("\nTesting " + self.id())

        # Learner and agent
        params = dict({'alpha': 1.0,
                       'gamma': 1.0,
                       'lambda': np.nan,
                       'alpha_min': 0.0,
                       'lambda_min': 0.0,
                       'lambda_max': 1.0,
                       'nepisodes': self.nepisodes,
                       })
        learner_tdlambda_adaptive = td.LeaTDLambdaAdaptive(self.env_logn_rewards, alpha=params['alpha'], gamma=params['gamma'],
                                                           alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,  # Every-visit is the default
                                                           adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=params['alpha_min'],
                                                           lambda_min=params['lambda_min'], lambda_max=params['lambda_max'],
                                                           burnin=False,
                                                           plotwhat="average", fontsize=int(self.fontsize*0.8),
                                                           debug=False)
        agent_rw_tdlambda_adaptive = agents.GenericAgent(self.policy_rw_envlogn, learner_tdlambda_adaptive)

        # Simulation        
        sim = simulators.Simulator(self.env_logn_rewards, agent_rw_tdlambda_adaptive, debug=False)
        _, _, RMSE_by_episode, state_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                     compute_rmse=False,
                                                     verbose=True, verbose_period=int(self.nepisodes/10),
                                                     plot=False, colormap=self.colormap, pause=self.pause)

        # Expected 2D state value function given as 1D array with:
        # 2D grid = 5 x 5
        # alpha = 1.0, gamma = 1.0, alpha_min = 0.0, lambda_min = 0.0, lambda_max = 1.0
        # alpha_update_type = AlphaUpdateType.FIRST_STATE_VISIT
        # adjust_alpha = True, adjust_alpha_by_episode = False
        # seed = 1717, nepisodes = 300, start_state = None
        # lambda as the Boltzmann function of delta(t) / average( abs(V(t)) )
        expected_values = np.array([-0.581206,	-0.000000,	-0.219902,	 0.682737,	 0.615329,
                                    -0.384120,	-0.365205,	 0.239536,	 0.000000,	 0.514185,
                                    -0.149668,	-0.120553,	 0.113366,	 0.304064,	 0.135725,
                                    -0.087527,	-0.083462,	-0.053985,	-0.200866,	-0.354783,
                                    -0.067125,	-0.107176,	-0.198340,	-0.464890,	-0.000000
                                    ])

        print("Agent ends at state {}:".format(self.env_logn_rewards.getState()))
        self.env_logn_rewards._render()
        observed_values = agent_rw_tdlambda_adaptive.getLearner().getV().getValues()
        #print("\nobserved values: %s" %(test_utils.array2str(observed_values)) )
        observed = np.asarray( observed_values ).reshape(self.ny, self.nx)
        print("\nobserved: ")
        print(observed)
        if self.plot:
            test_utils.plot_results_2D(observed, params, colormap=self.colormap, fontsize=self.fontsize)

        assert np.allclose(observed_values, expected_values, atol=1E-6)
    ####################### ADAPTIVE TD(lambda) #######################
    #------------------------------------------- TESTS ----------------------------------------


class Test_TD_Lambda_MountainCar(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1717
        self.nepisodes = 10000
        self.max_time_steps = 500   # Maximum number of steps to run per episode
        self.start_state = None # (0.0, 0.0)   # Position and velocity
        self.plot = True
        self.colormap = cm.get_cmap("rainbow")  # useful colormaps are "jet", "rainbow", seismic"

    @classmethod
    def setUpClass(cls):  # cls is the class, in this case, class 'Test_TD_Lambda'
        # IMPORTANT: All attributes defined here can be then be referenced using self!
        # (i.e. they belong to the "object" instantiated by this class)

        #cls.env = gym.make('MountainCar-v0')   # Here we create in principle a MountainCarEnv environment because this environment is registered with gym.register() or similar.
        # See also this implementation of the Mountain Car: https://github.com/JJonahJson/MountainCar-v313/blob/master/code/main.py

        # Environment with discretized position and velocity with nx and nv points respectively
        nx = 100
        nv = 100
        cls.env = mountaincars.MountainCarDiscrete(nx, nv)

        # Using the environment:
        # Ref: https://gym.openai.com/docs/
        #cls.env.reset()
        #for i in range(100):
        #    observation, reward, done, info = cls.env.step(cls.env.action_space.sample())  # take a random action
        #    print("iteration i={}: obs = {}, reward = {}, done = {}".format(i, observation, reward, done))
        #    #if np.mod(i, 10):
        #    #    cls.env.render()
        #cls.env.close()

        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env)

    def test_environment(self):
        env = mountaincars.MountainCarDiscrete(10, 10)
        state = env.reset(seed=1717)
        print("Environment reset to state: {}".format(state))
        assert np.allclose(state, np.array([-0.54806077, 0.0]))

        # Accelerate left several times
        observation_real, observation, reward, done, info = env.step(0, return_continuous_observation=True)
        print(observation_real, observation, reward, info)
        assert np.allclose(observation_real, np.array([-0.54887747, -0.0008167]))
        assert all(observation == np.array([3, 4]))

        observation_real, observation, reward, done, info = env.step(0, return_continuous_observation=True)
        print(observation_real, observation, reward, info)
        assert np.allclose(observation_real, np.array([-0.55050476, -0.00162729]))
        assert all(observation == np.array([3, 4]))

        observation_real, observation, reward, done, info = env.step(0, return_continuous_observation=True)
        print(observation_real, observation, reward, info)
        assert np.allclose(observation_real, np.array([-0.55293047, -0.00242572]))
        assert all(observation == np.array([3, 4]))

        observation_real, observation, reward, done, info = env.step(0, return_continuous_observation=True)
        print(observation_real, observation, reward, info)
        assert np.allclose(observation_real, np.array([-0.55613648, -0.00320601]))
        assert all(observation == np.array([3, 4]))

        # Do not accelerate
        observation_real, observation, reward, done, info = env.step(1, return_continuous_observation=True)
        print(observation_real, observation, reward, info)
        assert np.allclose(observation_real, np.array([-0.55909885, -0.00296237]))
        assert all(observation == np.array([3, 4]))

        # Accelerate right
        observation_real, observation, reward, done, info = env.step(2, return_continuous_observation=True)
        print(observation_real, observation, reward, info)
        assert np.allclose(observation_real, np.array([-0.56079547, -0.00169662]))
        assert all(observation == np.array([3, 4]))

    def test_random_walk_onecase(self, params=None):
        print("\nTesting " + self.id())

        # Learner and agent
        if params is None:
            params = dict({'alpha': 0.3,
                           'gamma': 1.0,
                           'lambda': 0.0, #0.7,
                           'alpha_min': 0.0,
                           'adjust_alpha': False
                           })
        learner_tdlambda = td.LeaTDLambda(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                          lmbda=params['lambda'],
                                          alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT, # Every-visit update is the default
                                          adjust_alpha=params['adjust_alpha'], adjust_alpha_by_episode=False,
                                          alpha_min=params['alpha_min'],
                                          debug=False)
        agent_rw_tdlambda = agents.GenericAgent(self.policy_rw, learner_tdlambda)

        #-- Simulation
        # Choose the initial state
        if self.start_state is not None:
            # A specific initial state
            idx_start_state = self.env.get_index_from_state(self.start_state)
        else:
            # Define a uniform Initial State Distribution in the environment so that the initial state is chosen randomly
            # in MountainCar.reset(). The Initial State Distribution (ISD) is an attribute of toy_text.discrete environment.

            # First find all the terminal states which should be excluded from the possible initial states!
            idx_states_non_terminal = self.env.get_indices_for_non_terminal_states()
            self.env.isd = np.array([1.0 / len(idx_states_non_terminal) if idx in idx_states_non_terminal else 0.0
                                     for idx in range(self.env.getNumStates())])
            #print("ISD:", self.env.isd)
            print("Steps: dx = {:.3f}, dv = {:.3f}".format(self.env.dx, self.env.dv))
            print("Positions: {}".format(self.env.get_positions()))
            print("Velocities: {}".format(self.env.get_velocities()))
            idx_start_state = None
        sim = simulators.Simulator(self.env, agent_rw_tdlambda, debug=False)

        time_start = timer()
        _, _, _, state_info = sim.run(nepisodes=self.nepisodes, max_time_steps=self.max_time_steps,
                                      start=idx_start_state, seed=self.seed,
                                      compute_rmse=False, state_observe=None,
                                      verbose=True, verbose_period=int(self.nepisodes/100),
                                      plot=False, pause=0.001)
        time_end = timer()
        exec_time = time_end - time_start
        print("Execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

        observed = agent_rw_tdlambda.getLearner().getV().getValues().reshape(self.env.shape)
        print("\nobserved: {}".format(observed))
        #print("Average RMSE over {} episodes: {:.3f}".format(self.nepisodes, np.mean(RMSE_by_episode)))
        if self.plot:
            # First replace estimated state values with NaN when the number of visits to the state is < 10
            # so that we don't get a bad idea of what the estimate is, since it is not reliable.
            state_counts = np.asarray(sim.agent.getLearner().getStateCounts()).reshape(self.env.shape)
            idx_not_enough_counts_x, idx_not_enough_counts_v = np.where(state_counts < 10)
            observed_toplot = copy.deepcopy(observed)
            observed_toplot[idx_not_enough_counts_x, idx_not_enough_counts_v] = np.nan

            import matplotlib.pyplot as plt
            x = self.env.get_positions()        # x is on the rows of `observed`
            v = self.env.get_velocities()       # v is on the cols of `observed`
            assert len(x) == observed_toplot.shape[0]
            assert len(v) == observed_toplot.shape[1]
            title_params = "(lambda={:.2f}, alpha={:.1f}, adj={}, episodes={}, max_time={}, density=(x:{}, v:{}) points)" \
                            .format(params['lambda'], params['alpha'], params['adjust_alpha'], self.nepisodes, self.max_time_steps, self.env.nx, self.env.nv)
            plt.figure()
            plt.errorbar(x, np.nanmean(observed_toplot/self.max_time_steps, axis=1),
                         yerr=np.nanstd(observed_toplot/self.max_time_steps, axis=1)/np.sqrt(self.env.nv),
                         marker='.', color="red", capsize=4)
            ax = plt.gca()
            ax.set_title("Average value of each position\n" + title_params)
            plt.figure()
            plt.errorbar(v, np.nanmean(observed_toplot/self.max_time_steps, axis=0),
                         yerr=np.nanstd(observed_toplot/self.max_time_steps, axis=0)/np.sqrt(self.env.nx),
                         marker='.', color="blue", capsize=4)
            ax = plt.gca()
            ax.set_title("Average value of each velocity\n" + title_params)

            # 2D plots
            params['nepisodes'] = self.nepisodes    # Needed for the call to plot_results_2D
            test_utils.plot_results_2D(observed_toplot/self.max_time_steps, params, colormap=self.colormap,
                                       fontsize=5, title="Value function (normalized by max time per episode: {})".format(self.max_time_steps))
            test_utils.plot_results_2D(np.log10(1 + state_counts), params, colormap=self.colormap, fontsize=5, title="State visit count (log scale)")

        #assert np.allclose(observed, expected, atol=1E-6)

        return observed, state_counts, params, sim


if __name__ == "__main__":
    test = True

    if test:
        #--- 1D tests
        unittest.main(defaultTest="Test_TD_Lambda_GW1D")
        #unittest.main(defaultTest="Test_TD_Lambda_GW1D.test_random_walk_onecase")
        #unittest.main(defaultTest="Test_TD_Lambda_GW1D.test_random_walk_adaptive_onecase")
        #unittest.getTestCaseNames()

        #--- 2D tests
        #unittest.main(defaultTest="Test_TD_Lambda_GW2D")

        # Basic environment
        #unittest.main(defaultTest="Test_TD_Lambda_GW2D.test_random_walk")
        #unittest.main(defaultTest="Test_TD_Lambda_GW2D.test_random_walk_adaptive")

        # Log(n)-rewards environment
        #unittest.main(defaultTest="Test_TD_Lambda_GW2D.test_random_walk_logn_rewards")
        #unittest.main(defaultTest="Test_TD_Lambda_GW2D.test_random_walk_adaptive_logn_rewards")


        # Or we can also use, to run one specific function defined in a TestCase class
        # Using unittest.TestSuite() to build our set of tests to run!
        # Ref: https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
        #suite = unittest.TestSuite()
        #suite.addTest(Test_TD_Lambda_GW2D("test_random_walk_logn_rewards"))
        #suite.addTest(Test_TD_Lambda_GW2D("test_random_walk_adaptive_logn_rewards"))
        #runner = unittest.TextTestRunner()
        #runner.run(suite)

        #--- Mountain Car tests
        #unittest.main(defaultTest="Test_TD_Lambda_MountainCar")
    else:
        resultsdir = "../../RL-001-MemoryManagement/results/MountainCar"

        test_obj = Test_TD_Lambda_MountainCar()
        nx = 20
        nv = 20
        test_obj.env = mountaincars.MountainCarDiscrete(nx, nv)
        test_obj.policy_rw = random_walks.PolRandomWalkDiscrete(test_obj.env)
        params = dict({'alpha': 1.0,
                       'gamma': 1.0,
                       'lambda': 0.0,
                       'alpha_min': 0.0,
                       'adjust_alpha': True
                       })
        state_values, state_counts, params, sim_obj = test_obj.test_random_walk_onecase(params=params)

        # Save
        filename = resultsdir + "/mountaincar_lambda={}_alpha={}_adj={}_episodes={},maxt={},nx={},nv={}.pickle" \
                    .format(params['lambda'], params['alpha'], params['adjust_alpha'],
                            test_obj.nepisodes, test_obj.max_time_steps, test_obj.env.nx, test_obj.env.nv)
        file = open(filename,
                    mode="wb")  # "b" means binary mode (needed for pickle.dump())
        pickle.dump(dict({'V': state_values, 'env': test_obj.env, 'params': params}), file)
        file.close()
        print("Results saved to:\n{}".format(os.path.abspath(filename)))

    pass
