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
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d    # For 3D plots (using ax = fig.axes(project='3d'), so the module is NOT explicitly mentioned!

from timeit import default_timer as timer

import pickle

from Python.lib.environments import gridworlds, mountaincars
import Python.lib.agents as agents
from Python.lib.agents.policies import random_walks
import Python.lib.agents.learners.episodic.discrete.td as td
from Python.lib.agents.learners.episodic.discrete import AlphaUpdateType
import Python.lib.simulators as simulators
from Python.lib.utils import computing

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
                  [-0.000000, -0.894306, -0.820482, -0.644328, -0.385795, -0.263782, -0.185667,
                   -0.132697, -0.077460, -0.038733, 0.001380, 0.036174, 0.065912, 0.097485,
                    0.160760, 0.230384, 0.327503, 0.483585, 0.640607, 0.862866, 0.000000],
                  # (2022/05/25) The following commented out expected result is obtained
                  # when the TD learner of the classes LeaTDLambda and LeaTDLambdaAdaptive
                  # defined in td.py use the alpha of the *currently visited state* as the learning rate
                  # for ALL past visited states whose value function is updated when lambda > 0.
                  # This is conceptually WRONG, although in the 1D gridworld I observed that
                  # this strategy generated a faster learning (as observed in fact here, where
                  # the state values are closer to the true values than the above results,
                  # which are the results of updating the value function of each state with
                  # their OWN alpha(s) --as opposed to using alpha(s_n) where s_n is the currently
                  # visited state at time step n)))
                  #[-0.000000, -0.946909, -0.907296, -0.775429, -0.507633, -0.370504, -0.280516,
                  # -0.210464, -0.131925, -0.069949, 0.005305, 0.067782, 0.116598, 0.165322,
                  #  0.247306, 0.315188, 0.428172, 0.604386, 0.732484, 0.908714, 0.000000],
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
                                          alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                                          adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode, alpha_min=alpha_min,
                                          debug=False)
        agent_rw_tdlambda = agents.GenericAgent(self.policy_rw, learner_tdlambda)

        # Simulation
        sim = simulators.Simulator(self.env, agent_rw_tdlambda, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                                     compute_rmse=True, state_observe=15,
                                                                     verbose=True, verbose_period=100,
                                                                     plot=False, pause=0.1)
        observed = agent_rw_tdlambda.getLearner().getV().getValues()
        print("\n" + self.id() + ", observed: " + test_utils.array2str(observed))
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
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
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
                              observed, self.env.getV(), RMSE_by_episode, learning_info['alpha_mean'],
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
        _, _, RMSE_by_episode, MAPE_by_episode, _ = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
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
        _, _, RMSE_by_episode, MAPE_by_episode, _ = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
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
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
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
        expected = np.array([ 0.000000, -0.861562, -0.756466, -0.559433, -0.293604,
                             -0.162618, -0.085392, -0.045967, -0.016827, -0.004943,
                              0.000018,  0.004353,  0.011266,  0.023671,  0.065678,
                              0.125709,  0.230908,  0.391818,  0.551407,  0.826242, 0.000000])
        # (2022/05/25) The following commented out expected result is obtained
        # when the TD learner of the classes LeaTDLambda and LeaTDLambdaAdaptive
        # defined in td.py use the alpha of the *currently visited state* as the learning rate
        # for ALL past visited states whose value function is updated when lambda > 0.
        # This is conceptually WRONG, although in the 1D gridworld I observed that
        # this strategy generated a faster learning (as observed in fact here, where
        # the state values are closer to the true values than the above results,
        # which are the results of updating the value function of each state with
        # their OWN alpha(s) --as opposed to using alpha(s_n) where s_n is the currently
        # visited state at time step n)))
        #expected = np.array([-0.000000, -0.914800, -0.841925, -0.664785, -0.392792,
        #                     -0.249244, -0.154211, -0.093055, -0.041201, -0.016698,
        #                     -0.006381, -0.005999, -0.009759, -0.013459, -0.009742,
        #                      0.012727,  0.100282,  0.307882,  0.419977,  0.775499, 0.000000])

        observed = agent_rw_tdlambda_adaptive.getLearner().getV().getValues()
        print("\n" + self.id() + ", observed: " + test_utils.array2str(observed))
        print("Average RMSE over {} episodes: {:.3f}".format(self.nepisodes, np.mean(RMSE_by_episode)))
        if self.plot:
            (ax, ax2) = self.plot_results(params,
                              observed, self.env.getV(), RMSE_by_episode, learning_info['alpha_mean'],
                              y2label="(Average) alpha & lambda", y2lim=(0, 1.0),
                              max_rmse=self.max_rmse, color_rmse=self.color_rmse)
            ax2.plot(np.arange(self.nepisodes)+1, agent_rw_tdlambda_adaptive.getLearner().lambda_mean_by_episode, color="orange")

        #input("Press Enter...")

        assert np.allclose(observed, expected, atol=1E-6)
    #------------------------------------------- TESTS ----------------------------------------


class Test_TD_Lambda_GW1D_OneTerminalState(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        self.seed = kwargs.pop('seed', 1717)
        self.nepisodes = kwargs.pop('nepisodes', 20)
        self.start_state = kwargs.pop('start_state', 0)
        self.plot = kwargs.pop('plot', False)
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):  # cls is the class, in this case, class 'Test_TD_Lambda'
        # IMPORTANT: All attributes defined here can be then be referenced using self!
        # (i.e. they belong to the "object" instantiated by this class)
        cls.max_rmse = 0.8
        cls.color_rmse = "red"

        # Environment: 1D gridworld
        cls.nS = 19  # Number of non-terminal states in the 1D gridworld
        cls.env = gridworlds.EnvGridworld1D_OneTerminalState(length=cls.nS + 1)  # nS states plus one terminal state

        # Random walk policy on the above environment
        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env)

    def run_random_walk_adaptive_onecase(self):
        print("\nTesting " + self.id())

        # Learner and agent
        # (we use the same parameters as for the test case in dataprovider with labmda = 0.7
        params = dict({'alpha': 1.0,
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
        _, _, _, _, learning_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                                     compute_rmse=False,
                                                                     verbose=True, verbose_period=100,
                                                                     plot=False, pause=0.001)

        observed = agent_rw_tdlambda_adaptive.getLearner().getV().getValues()
        print("\n" + self.id() + ", observed: " + test_utils.array2str(observed))

        return learning_info


class Test_TD_Lambda_GW2D(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1717
        self.nepisodes = 300
        self.start_state = None
        self.colormap = cm.get_cmap("rainbow")  # useful colormaps are "jet", "rainbow", seismic"
        self.pause = 0.001
        self.plot = False

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
        _, _, _, _, learning_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                         compute_rmse=False, state_observe=0,
                                         verbose=True, verbose_period=max(1, int(self.nepisodes/10)),
                                         plot=self.plot, colormap=self.colormap, pause=self.pause)

        # Expected 2D state value function given as 1D array with:
        # 2D grid = 5 x 5
        # alpha = 1.0, gamma = 1.0, lambda = 0.7, alpha_min = 0.0
        # alpha_update_type = AlphaUpdateType.FIRST_STATE_VISIT
        # adjust_alpha = True, adjust_alpha_by_episode = False
        # seed = 1717, nepisodes = 300, start_state = None
        expected_values = np.array([  0.        , -0.54833162, -0.28786602, -0.09861481, -0.00267096,
                                     -0.54412628, -0.38643206, -0.20172248, -0.01273661,  0.10080109,
                                     -0.29436269, -0.17989673, -0.03486362,  0.14812270,  0.26957068,
                                     -0.10406726, -0.01359656,  0.13508918,  0.37341962,  0.58856250,
                                     -0.04133353,  0.08119520,  0.25379506,  0.53486058,  0.        ])
        # (2022/05/25) The following commented out expected result is obtained
        # when the TD learner of the classes LeaTDLambda and LeaTDLambdaAdaptive
        # defined in td.py use the alpha of the *currently visited state* as the learning rate
        # for ALL past visited states whose value function is updated when lambda > 0.
        # This is conceptually WRONG, although in the 1D gridworld I observed that
        # this strategy generated a faster learning (as observed in fact here, where
        # the state values are closer to the true values than the above results,
        # which are the results of updating the value function of each state with
        # their OWN alpha(s) --as opposed to using alpha(s_n) where s_n is the currently
        # visited state at time step n)))
        #expected_values = np.array([-0.000000,	-0.600664,	-0.340097,	-0.199044,	-0.109807,
        #                            -0.598136,	-0.440290,	-0.250745,	-0.110556,	-0.009195,
        #                            -0.335323,	-0.245785,	-0.109325,	 0.057424,	 0.185493,
        #                            -0.163590,	-0.082114,	 0.035308,	 0.267254,	 0.538616,
        #                            -0.084325,	-0.000746,	 0.142019,	 0.426486,	 0.000000,
        #                            ])

        print("Agent ends at state {}:".format(self.env.getState()))
        self.env._render()
        observed_values = agent_rw_tdlambda.getLearner().getV().getValues()
        #print("\nobserved values: %s" %(test_utils.array2str(observed_values)) )
        observed = np.asarray( observed_values ).reshape(self.ny, self.nx)
        print("\n{}, observed: ".format(self.id()))
        print(observed)
        if self.plot:
            ax = plt.figure().subplots(1,1)
            test_utils.plot_results_2D(ax, observed, params, colormap=self.colormap, fontsize=self.fontsize)

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
        _, _, _, _, learning_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                         compute_rmse=False, state_observe=0,
                                         verbose=True, verbose_period=max(1, int(self.nepisodes/10)),
                                         plot=self.plot, colormap=self.colormap, pause=self.pause)

        # Expected 2D state value function given as 1D array with:
        # 2D grid = 5 x 5
        # alpha = 1.0, gamma = 1.0, lambda = 0.7, alpha_min = 0.0
        # alpha_update_type = AlphaUpdateType.FIRST_STATE_VISIT
        # adjust_alpha = True, adjust_alpha_by_episode = False
        # seed = 1717, nepisodes = 300, start_state = None
        expected_values = np.array([ -0.77182828,  0.        , -0.27604594,  0.52689398,  0.52734742,
                                     -0.52284430, -0.39985642,  0.13538644,  0.        ,  0.5737801 ,
                                     -0.36497345, -0.26899774, -0.04048382,  0.25487455,  0.17131144,
                                     -0.32290312, -0.27697465, -0.21734447, -0.26036269, -0.44212176,
                                     -0.31989397, -0.30768536, -0.34226460, -0.53298452,  0.        ])
        # (2022/05/25) The following commented out expected result is obtained
        # when the TD learner of the classes LeaTDLambda and LeaTDLambdaAdaptive
        # defined in td.py use the alpha of the *currently visited state* as the learning rate
        # for ALL past visited states whose value function is updated when lambda > 0.
        # This is conceptually WRONG, although in the 1D gridworld I observed that
        # this strategy generated a faster learning (as observed in fact here, where
        # the state values are closer to the true values than the above results,
        # which are the results of updating the value function of each state with
        # their OWN alpha(s) --as opposed to using alpha(s_n) where s_n is the currently
        # visited state at time step n)))
        #expected_values = np.array([-0.586914,	-0.000000,	-0.275543,	 0.534306,	 0.529651,
        #                            -0.447301,	-0.386761,	 0.160712,	 0.000000,	 0.496648,
        #                            -0.263472,	-0.210891,	-0.001141,	 0.264747,	 0.124014,
        #                            -0.223334,	-0.210347,	-0.169349,	-0.249459,	-0.393667,
        #                            -0.192741,	-0.196232,	-0.272632,	-0.510045,	-0.000000
        #                            ])

        print("Agent ends at state {}:".format(self.env_logn_rewards.getState()))
        self.env_logn_rewards._render()
        observed_values = agent_rw_tdlambda.getLearner().getV().getValues()
        #print("\nobserved values: %s" %(test_utils.array2str(observed_values)) )
        observed = np.asarray( observed_values ).reshape(self.ny, self.nx)
        print("\n{}, observed: ".format(self.id()))
        print(observed)
        if self.plot:
            ax = plt.figure().gca()
            test_utils.plot_results_2D(ax, observed, params, colormap=self.colormap, fontsize=self.fontsize)

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
        _, _, _, _, learning_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                         compute_rmse=False,
                                         verbose=True, verbose_period=max(1, int(self.nepisodes/10)),
                                         plot=self.plot, colormap=self.colormap, pause=self.pause)

        # Expected 2D state value function given as 1D array with:
        # 2D grid = 5 x 5
        # alpha = 1.0, gamma = 1.0, alpha_min = 0.0, lambda_min = 0.0, lambda_max = 1.0
        # alpha_update_type = AlphaUpdateType.FIRST_STATE_VISIT
        # adjust_alpha = True, adjust_alpha_by_episode = False
        # seed = 1717, nepisodes = 300, start_state = None
        # lambda as the Boltzmann function of delta(t) / average( abs(V(t)) )
        expected_values = np.array([ 0.         , -0.59109302, -0.32699058, -0.12929933, -0.00239441,
                                     -0.57451245, -0.42242940, -0.22250979, -0.01439459,  0.11135646,
                                     -0.34176244, -0.20932554, -0.04386852,  0.16273871,  0.30397547,
                                     -0.14068694, -0.03001114,  0.13679581,  0.39306951,  0.61440035,
                                     -0.04388995,  0.08125758,  0.26239804,  0.55647320,  0.        ])
        # (2022/05/25) The following commented out expected result is obtained
        # when the TD learner of the classes LeaTDLambda and LeaTDLambdaAdaptive
        # defined in td.py use the alpha of the *currently visited state* as the learning rate
        # for ALL past visited states whose value function is updated when lambda > 0.
        # This is conceptually WRONG, although in the 1D gridworld I observed that
        # this strategy generated a faster learning (as observed in fact here, where
        # the state values are closer to the true values than the above results,
        # which are the results of updating the value function of each state with
        # their OWN alpha(s) --as opposed to using alpha(s_n) where s_n is the currently
        # visited state at time step n)))
        #expected_values = np.array([-0.000000,	-0.621289,	-0.356950,	-0.197688,	-0.095204,
        #                            -0.597418,	-0.449396,	-0.250112,	-0.100089,	 0.021283,
        #                            -0.330550,	-0.235789,	-0.092370,	 0.075680,	 0.224815,
        #                            -0.153306,	-0.058187,	 0.074201,	 0.301015,	 0.564076,
        #                            -0.051561,	 0.038582,	 0.192242,	 0.472891,	 0.000000
        #                            ])

        print("Agent ends at state {}:".format(self.env.getState()))
        self.env._render()
        observed_values = agent_rw_tdlambda_adaptive.getLearner().getV().getValues()
        #print("\nobserved values: %s" %(test_utils.array2str(observed_values)) )
        observed = np.asarray( observed_values ).reshape(self.ny, self.nx)
        print("\n{}, observed: ".format(self.id()))
        print(observed)
        if self.plot:
            ax = plt.figure().gca()
            test_utils.plot_results_2D(ax, observed, params, colormap=self.colormap, fontsize=self.fontsize)

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
        _, _, _, _, learning_info = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                         compute_rmse=False,
                                         verbose=True, verbose_period=max(1, int(self.nepisodes/10)),
                                         plot=self.plot, colormap=self.colormap, pause=self.pause)

        # Expected 2D state value function given as 1D array with:
        # 2D grid = 5 x 5
        # alpha = 1.0, gamma = 1.0, alpha_min = 0.0, lambda_min = 0.0, lambda_max = 1.0
        # alpha_update_type = AlphaUpdateType.FIRST_STATE_VISIT
        # adjust_alpha = True, adjust_alpha_by_episode = False
        # seed = 1717, nepisodes = 300, start_state = None
        # lambda as the Boltzmann function of delta(t) / average( abs(V(t)) )
        expected_values = np.array([ -0.75751082,  0.        , -0.22101245,  0.62302307,  0.54209392,
                                     -0.46960693, -0.35882616,  0.18873879,  0.        ,  0.55887958,
                                     -0.27621701, -0.18056934,  0.05215461,  0.29747383,  0.18889241,
                                     -0.20753015, -0.16433560, -0.12119574, -0.21485266, -0.40461707,
                                     -0.19083378, -0.19230422, -0.26873757, -0.48753816,  0.        ])
        # (2022/05/25) The following commented out expected result is obtained
        # when the TD learner of the classes LeaTDLambda and LeaTDLambdaAdaptive
        # defined in td.py use the alpha of the *currently visited state* as the learning rate
        # for ALL past visited states whose value function is updated when lambda > 0.
        # This is conceptually WRONG, although in the 1D gridworld I observed that
        # this strategy generated a faster learning (as observed in fact here, where
        # the state values are closer to the true values than the above results,
        # which are the results of updating the value function of each state with
        # their OWN alpha(s) --as opposed to using alpha(s_n) where s_n is the currently
        # visited state at time step n)))
        #expected_values = np.array([-0.581206,	-0.000000,	-0.219902,	 0.682737,	 0.615329,
        #                            -0.384120,	-0.365205,	 0.239536,	 0.000000,	 0.514185,
        #                            -0.149668,	-0.120553,	 0.113366,	 0.304064,	 0.135725,
        #                            -0.087527,	-0.083462,	-0.053985,	-0.200866,	-0.354783,
        #                            -0.067125,	-0.107176,	-0.198340,	-0.464890,	-0.000000
        #                            ])

        print("Agent ends at state {}:".format(self.env_logn_rewards.getState()))
        self.env_logn_rewards._render()
        observed_values = agent_rw_tdlambda_adaptive.getLearner().getV().getValues()
        #print("\nobserved values: %s" %(test_utils.array2str(observed_values)) )
        observed = np.asarray( observed_values ).reshape(self.ny, self.nx)
        print("\n{}, observed: ".format(self.id()))
        print(observed)
        if self.plot:
            ax = plt.figure().gca()
            test_utils.plot_results_2D(ax, observed, params, colormap=self.colormap, fontsize=self.fontsize)

        assert np.allclose(observed_values, expected_values, atol=1E-6)
    ####################### ADAPTIVE TD(lambda) #######################
    #------------------------------------------- TESTS ----------------------------------------


class Test_TD_Lambda_MountainCar(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        self.seed = kwargs.pop('seed', None)
        self.nepisodes = kwargs.pop('nepisodes', 10)
        self.max_time_steps = kwargs.pop('max_time_steps', 500)  # Maximum number of steps to run per episode
        self.normalizer = kwargs.pop('normalizer', 1)            # Normalize for the plots: Set it to max_time_steps when the rewards are NOT sparse (i.e. are -1 every where except at terminal states), o.w. set it to 1 (when rewards are sparse, i.e. they occur at terminal states)
        self.start_state = kwargs.pop('start_state', None)       # Position and velocity
        self.plot = kwargs.pop('plot', False)
        self.colormap = cm.get_cmap("rainbow")  # useful colormaps are "jet", "rainbow", seismic"
        super().__init__(*args, **kwargs)

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

    def plot_results(self, values_2d, state_counts):
        # First replace estimated state values with NaN when the number of visits to the state is < 10
        # so that we don't get a bad idea of what the estimate is, since it is not reliable.
        idx_not_enough_counts_x, idx_not_enough_counts_v = np.where(state_counts < 10)
        values_2d_toplot = copy.deepcopy(values_2d)
        values_2d_toplot[idx_not_enough_counts_x, idx_not_enough_counts_v] = np.nan

        import matplotlib.pyplot as plt
        x = self.env.get_positions()  # x is on the rows of `values_2d`
        v = self.env.get_velocities()  # v is on the cols of `values_2d`
        assert len(x) == values_2d_toplot.shape[0]
        assert len(v) == values_2d_toplot.shape[1]
        title_params = "(lambda={:.2f}, alpha={:.1f}, adj={}, episodes={}, max_time={}, density=(x:{}, v:{}) points)" \
            .format(params['lambda'], params['alpha'], params['adjust_alpha'], self.nepisodes, self.max_time_steps,
                    self.env.nx, self.env.nv)
        plt.figure()
        plt.errorbar(x, np.nanmean(values_2d_toplot / self.normalizer, axis=1),
                     yerr=np.nanstd(values_2d_toplot / self.normalizer, axis=1) / np.sqrt(self.env.nv),
                     marker='.', color="red", capsize=4)
        ax = plt.gca()
        ax.set_title("Average value of each position\n" + title_params)

        plt.figure()
        plt.errorbar(v, np.nanmean(values_2d_toplot / self.normalizer, axis=0),
                     yerr=np.nanstd(values_2d_toplot / self.normalizer, axis=0) / np.sqrt(self.env.nx),
                     marker='.', color="blue", capsize=4)
        ax = plt.gca()
        ax.set_title("Average value of each velocity\n" + title_params)

        # 2D plots
        params['nepisodes'] = self.nepisodes  # Needed for the call to plot_results_2D
        fig = plt.figure()
        axes = fig.subplots(1,3)
        test_utils.plot_results_2D(axes[0], values_2d_toplot / self.normalizer, params, colormap=self.colormap,
                                   fontsize=5, title="Value function (normalized by {})".format(
                self.normalizer))
        test_utils.plot_results_2D(axes[1], state_counts, params, colormap=cm.get_cmap("Blues"), fontsize=5, format_labels=".0f",
                                   title="State visit count")
        test_utils.plot_results_2D(axes[2], np.log10(1 + state_counts), params, colormap=cm.get_cmap("Blues"), fontsize=5, format_labels=".0f",
                                   title="State visit count (log scale)")

        return fig

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

    def run_random_walk_onecase(self, params=None, adaptive=False, verbose_convergence=False):
        print("\nTesting " + self.id())

        # Learner and agent
        if params is None:
            params = dict({ 'alpha': 1.0,
                            'gamma': 1.0,
                            'lambda': 0.0,
                            'lambda_min': 0.0,
                            'lambda_max': 0.99,
                            'alpha_min': 0.0,
                            'adjust_alpha': False,
                            'alpha_update_type': AlphaUpdateType.FIRST_STATE_VISIT,
                            'adaptive_type': td.AdaptiveLambdaType.ATD,
                           })
        if not adaptive:
            learner_tdlambda = td.LeaTDLambda(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                              lmbda=params['lambda'],
                                              alpha_update_type=params['alpha_update_type'], # Every-visit update is the default
                                              adjust_alpha=params['adjust_alpha'], adjust_alpha_by_episode=False,
                                              alpha_min=params['alpha_min'],
                                              debug=False)
        else:
            learner_tdlambda = td.LeaTDLambdaAdaptive(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                                      alpha_update_type=params['alpha_update_type'],
                                                      adjust_alpha=params['adjust_alpha'], adjust_alpha_by_episode=False,
                                                      alpha_min=params['alpha_min'],
                                                      lambda_min=params['lambda_min'], lambda_max=params['lambda_max'],
                                                      adaptive_type=params['adaptive_type'],
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
        _, _, _, _, learning_info = sim.run(nepisodes=self.nepisodes, max_time_steps=self.max_time_steps,
                                        start=idx_start_state, seed=self.seed,
                                        compute_rmse=False, state_observe=None,
                                        verbose=True, verbose_period=max(1, int(self.nepisodes/10)),
                                        verbose_convergence=verbose_convergence,
                                        plot=False, pause=0.001)
        time_end = timer()
        exec_time = time_end - time_start
        print("Execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

        observed = self.env.reshape_from_1d_to_2d(agent_rw_tdlambda.getLearner().getV().getValues())
        state_counts = self.env.reshape_from_1d_to_2d(np.asarray(sim.agent.getLearner().getStateCounts()))
        print("\n{}, observed: ".format(self.id(), observed))
        #print("Average RMSE over {} episodes: {:.3f}".format(self.nepisodes, np.mean(RMSE_by_episode)))
        if self.plot:
            self.plot_results(observed, state_counts)
        #assert np.allclose(observed, expected, atol=1E-6)

        return observed, state_counts, params, sim, learning_info


if __name__ == "__main__":
    test = False

    if not test:
        test_env_name = "MountainCar"
        test_env_name = "GW1D_OneTerminalState"

    if test:
        #--- 1D tests
        unittest.main(defaultTest="Test_TD_Lambda_GW1D")
        #unittest.main(defaultTest="Test_TD_Lambda_GW1D.test_random_walk_onecase")
        #unittest.main(defaultTest="Test_TD_Lambda_GW1D.test_random_walk_adaptive_onecase")
        #unittest.getTestCaseNames()

        #--- 2D tests
        unittest.main(defaultTest="Test_TD_Lambda_GW2D")

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
    elif test_env_name == "GW1D_OneTerminalState":
        # Prepare environment
        nstates = 19
        nepisodes = 500
        plot = True
        test_obj = Test_TD_Lambda_GW1D_OneTerminalState(seed=1717, nepisodes=nepisodes, start_state=0, plot=plot)
        test_obj.env = gridworlds.EnvGridworld1D_OneTerminalState(nstates+1)    # +1 terminal state
        test_obj.policy_rw = random_walks.PolRandomWalkDiscrete(test_obj.env)

        learning_info = test_obj.run_random_walk_adaptive_onecase()
        if plot:
            plt.plot(range(nepisodes), learning_info['lambda_mean'])
            ax = plt.gca()
            ax.set_xlabel("Episode")
            ax.set_ylabel("Average lambda over states")
    elif test_env_name == "MountainCar":
        import os
        resultsdir = os.path.abspath("../../RL-001-MemoryManagement/results/MountainCar")
        save = False
        plot = True
        verbose_convergence = True
        gamma = 1.0
        alpha_update_type = AlphaUpdateType.EVERY_STATE_VISIT
        nepisodes_benchmark = 30000
        max_time_steps_benchmark = 500
        normalizer = 1 #max_time_steps_benchmark    # Normalize by max_time_steps when rewards are NOT sparse, o.w. use 1
            ## nepisodes=30000, max_time=500 => 2+ hours with MC (TD(1))
        # Case to run
        case = 'benchmark'
        case = 'td'
        case = 'atd'
        case = 'hatd'

        # Prepare environment
        test_obj = Test_TD_Lambda_MountainCar(seed=1717, start_state=None, normalizer=normalizer, plot=plot)
        nx = 20
        nv = 20
        test_obj.env = mountaincars.MountainCarDiscrete(nx, nv)
        test_obj.policy_rw = random_walks.PolRandomWalkDiscrete(test_obj.env)

        # Prepare agent's learner
        if case == 'benchmark':
            # Execution to get the benchmark, i.e. the estimated state value function V that will be considered as the true value function
            adaptive = False
            test_obj.nepisodes = nepisodes_benchmark #10000
            test_obj.max_time_steps = max_time_steps_benchmark
            params = dict({'alpha': 1.0,
                           'gamma': gamma,
                           'lambda': 1.0,
                           'lambda_min': 0.0,
                           'lambda_max': 0.99,
                           'alpha_min': 0.0,
                           'adjust_alpha': True,
                           'alpha_update_type': AlphaUpdateType.EVERY_STATE_VISIT,  # We use FIRST_STATE_VISIT as opposed to EVERY_STATE_VISIT because the latter gives too slow learning (mean|V| reaches 0.02 instead of 0.06 after 5000 episodes)
                           'adaptive_type': None
                           })
        elif case == 'td':
            adaptive = False
            test_obj.nepisodes = 200 #1000
            test_obj.max_time_steps = max_time_steps_benchmark
            params = dict({'alpha': 1.0,
                           'gamma': gamma,
                           'lambda': 0.9,
                           'lambda_min': 0.0,
                           'lambda_max': 0.99,
                           'alpha_min': 0.0,
                           'adjust_alpha': True,
                           'alpha_update_type': alpha_update_type,
                           'adaptive_type': None
                           })
        elif case == 'atd':
            adaptive = True
            test_obj.nepisodes = 200
            test_obj.max_time_steps = max_time_steps_benchmark
            params = dict({'alpha': 1.0,
                           'gamma': gamma,
                           'lambda': 0.0,
                           'lambda_min': 0.0,
                           'lambda_max': 0.99,
                           'alpha_min': 0.0,
                           'adjust_alpha': True,
                           'alpha_update_type': alpha_update_type,
                           'adaptive_type': td.AdaptiveLambdaType.ATD
                           })
        elif case == 'hatd':
            adaptive = True
            test_obj.nepisodes = 200
            test_obj.max_time_steps = max_time_steps_benchmark
            params = dict({'alpha': 1.0,
                           'gamma': gamma,
                           'lambda': 0.0,
                           'lambda_min': 0.0,
                           'lambda_max': 0.99, #0.80,
                           'alpha_min': 0.0,
                           'adjust_alpha': True,
                           'alpha_update_type': alpha_update_type,
                           'adaptive_type': td.AdaptiveLambdaType.HATD
                           })

        # Run the estimation process
        state_values, state_counts, params, sim_obj, learning_info = \
            test_obj.run_random_walk_onecase(params=params, adaptive=adaptive, verbose_convergence=verbose_convergence)

        if case == "benchmark":
            if save:
                filename = resultsdir + "/mountaincar_BENCHMARK_gamma={:.2f}_lambda={}_alpha={}_adj={}_episodes={},maxt={},nx={},nv={}.pickle" \
                    .format(params['gamma'], params['lambda'], params['alpha'], params['adjust_alpha'],
                            test_obj.nepisodes, test_obj.max_time_steps, test_obj.env.nx, test_obj.env.nv)
                file = open(filename, mode="wb")  # "b" means binary mode (needed for pickle.dump())
                pickle.dump(dict({'V': state_values, 'counts': state_counts,
                                  'env': test_obj.env, 'policy': test_obj.policy_rw,
                                  'simulator': sim_obj, 'learning_info': learning_info,
                                  'params_test': {'nepisodes': test_obj.nepisodes,
                                                  'max_time_steps': test_obj.max_time_steps},
                                  'params': params}), file)
                ## NOTE: The simulator object is quite large... ~ 1 MB. Otherwise, the output file only occupies ~ 20 kB
                ## ALSO NOTE: It is not possible to save the `test_obj` per se... as I get the error:
                ##  "Can't pickle <class '__main__.Test_TD_Lambda_MountainCar'>: attribute lookup Test_TD_Lambda_MountainCar on __main__ failed"
                ## Reason explained here: https://stackoverflow.com/questions/48615601/cant-pickle-class-a-class-attribute-lookup-inner-class-on-a-class-failed
                ## Essentially, the reason is that the test class does not belong to a module.
                file.close()
                print("Results saved to:\n{}".format(os.path.abspath(filename)))
        else:
            # 3D plot of the estimation error
            # Load the benchmark (Vtrue)
            try:
                import pickle
                filename_benchmark = "mountaincar_BENCHMARK_gamma={:.2f}_lambda=0.0_alpha=1.0_adj=True_episodes={},maxt=500,nx=20,nv=20 (FIRST_STATE_VISIT).pickle" \
                    .format(params['gamma'], nepisodes_benchmark)
                file = open(resultsdir + "/" + filename_benchmark, mode="rb")
                dict_benchmark = pickle.load(file)
                file.close()

                # Create the environment on which tests will be run
                # Currently (2022/05/01) we need to do this just because the MountainCarDiscrete environment has changed definition
                # w.r.t. to the MountainCarDiscrete environment saved in the pickle file, e.g. there are new methods defined such as setV().
                # If the definition of the saved environment (in dict_benchmark['env']) is the same as the current definition of the
                # MountainCarDiscrete environment, then we can just use the saved environment as environment on which test are run.
                env_mountain = mountaincars.MountainCarDiscrete(dict_benchmark['env'].nx, dict_benchmark['env'].nv)
                max_time_steps = test_obj.max_time_steps
                state_counts_benchmark = dict_benchmark['counts']

                # Compute the error
                Vtrue = dict_benchmark['V']
                Vest = state_values
                Verror = (Vest - Vtrue)
                #Verror = (Vest - Vtrue) / abs(Vtrue)
                rmse = computing.rmse(Vtrue, Vest, weights=state_counts_benchmark)
                mape = computing.mape(Vtrue, Vest, weights=state_counts_benchmark)

                # 2D image plots of the error and the state counts observed during estimation
                algorithm_name = params['adaptive_type'] is not None and params['adaptive_type'].name or "TD({:.2f})".format(params['lambda'])
                fig = test_obj.plot_results(Verror, state_counts)
                fig.suptitle(algorithm_name)

                # Plot
                zlim = None
                #zlim = (-0.5, 0.5)
                x = test_obj.env.get_positions()
                v = test_obj.env.get_velocities()
                xx, vv = np.meshgrid(x, v)
                fig = plt.figure()
                # When there is only one subplot we can use plt.axes() to create the 3D axes
                #ax = plt.axes(projection='3d')
                # Otherwise, when there are several subplots in the same figure
                # Ref: https://matplotlib.org/stable/gallery/mplot3d/mixed_subplots.html
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                ax2 = fig.add_subplot(1, 2, 2, projection='3d')

                # Surface plot of the error
                surf = ax1.plot_surface(xx, vv, Verror)
                #ax1.contourf(xx, vv, state_counts, offset=0) #zlim[0])
                # Contour plot
                #cf = ax1.contourf(xx, vv, Verror)
                #plt.colorbar(cf)
                if zlim is not None:
                    ax1.set_zlim(zlim)
                ax1.set_xlabel("Position")
                ax1.set_ylabel("Velocity")
                ax1.set_zlabel("Relative error of V(x,v)")
                ax1.set_title("Estimation error for {} (Weighted RMSE = {:.2f}, Weighted MAPE = {:.2f}%)".format(algorithm_name, rmse, mape*100))

                # Surface plot of the state counts
                ax2.plot_surface(xx, vv, state_counts)
                ax2.set_xlabel("Position")
                ax2.set_ylabel("Velocity")
                ax2.set_zlabel("State count")
                ax2.set_title("State count for {}".format(algorithm_name))
            except:
                print("WARNING: Plots of estimation error skipped because BENCHMARK file not found:\n{}".format(filename_benchmark))

        #-- Plot convergence analysis
        # Parameters used below
        nepisodes = test_obj.nepisodes
        max_time_steps = test_obj.max_time_steps
        normalizer = test_obj.normalizer
        env = test_obj.env
        gamma = params['gamma']
        alpha = params['alpha']
        alpha_update_type = params['alpha_update_type']
        lmbda = params['lambda']
        adaptive_type = params['adaptive_type']
        # If plotting the results saved in the benchmark read above
        #nepisodes = dict_benchmark['params_test']['nepisodes']
        #max_time_steps = dict_benchmark['params_test']['max_time_steps']
        #env = dict_benchmark['env']
        #gamma = dict_benchmark['params']['gamma']
        #alpha = dict_benchmark['params']['alpha']
        #alpha_update_type = dict_benchmark['params']['alpha_update_type']
        #lmbda = dict_benchmark['params']['lambda']
        #adaptive_type = dict_benchmark['params']['adaptive_type']

        #-- What to plot
        # Use this when rewards are SPARSE (e.g. +1 at the terminal states)
        deltaV_abs_name = "deltaV_abs_mean"
        deltaV_max_name = "deltaV_max_signed"
        # Use this when rewards are NOT sparse
        #deltaV_abs_name = "deltaV_rel_abs_mean"
        #deltaV_max_name = "deltaV_rel_max_signed"

        # Compute moving average of the error for easier visualization
        import numpy as np
        import matplotlib.pyplot as plt
        window_size = max(1, int(test_obj.nepisodes/100)) #dict_benchmark['params']['nepisodes']/100)) (when reading the #episodes from the benchmark file)
        deltaV_max_smooth = computing.smooth(learning_info[deltaV_max_name], window_size=window_size)
        alpha_mean_smooth = computing.smooth(learning_info['alpha_mean'], window_size=window_size)

        # Plot
        fig = plt.figure()
        ax, ax_n = fig.subplots(1,2)

        legend_ax = []
        ax.plot(np.arange(nepisodes+1), learning_info['V_abs_mean'] / normalizer, 'b-')
        legend_ax += ["Normalized mean|V|"]
        ax.plot(np.arange(nepisodes+1), learning_info['V_abs_median'] / normalizer, 'b--')
        legend_ax += ["Normalized median|V|"]
        # This may be very noisy...
        #ax.plot(np.arange(nepisodes+1), learning_info['V_abs_min'] / normalizer, color='blue', linestyle='dotted', linewidth=0.5)
        #legend_ax += ["Normalized min|V| for non-zero change states"]
        ax.set_xlabel("Episode number")
        ax.set_ylabel("|V| normalized by dividing by {}".format(normalizer))
        ax.legend(legend_ax, loc='upper left')

        ax2 = ax.twinx()
        legend_ax2 = []
        #ax2.plot(np.arange(nepisodes+1), learning_info[deltaV_max_name]*100, 'r-', linewidth=1)
        #legend_ax2 += ["max|relative change| with sign"]
        ax2.plot(np.arange(nepisodes+1), deltaV_max_smooth, 'r-', linewidth=1)
        legend_ax2 += ["max|relative change| with sign ({}-tap-smoothed)".format(window_size)]
        ax2.plot(np.arange(nepisodes+1), learning_info[deltaV_abs_name]*100, 'r--', linewidth=0.5)
        legend_ax2 += ["mean|relative change|"]
        ax2.axhline(0.0, color='gray', linewidth=0.5)
        ## The median values are 0.0 for ALL episodes... interesting!
        ax2.set_yscale('symlog')
        ax2.set_ylim( -max(ax2.get_ylim()), +max(ax2.get_ylim()) )
        ax2.set_ylabel("mean |relative delta(V)| (%) (log scale)")
        ax2.legend(legend_ax2, loc='upper right')
        #ax2.legend(["mean|relative change|", "weighted mean|relative change|"])
        ax.set_title("Value function estimation and relative change")

        # Number of states used in the computation of the summary statistics at each episode
        ax_n.plot(np.arange(nepisodes+1), learning_info['V_abs_n'] / env.getNumStates() * 100, 'g-')
        ax_n.set_xlabel("Episode number")
        ax_n.set_ylabel("% states")
        ax_n.set_ylim((0,100))
        ax_n.legend(["% states"], loc='upper left')
        ax_n2 = ax_n.twinx()
        legend_ax_n2 = []
        #ax_n2.plot(np.arange(nepisodes), learning_info['alpha_mean'], 'k-', linestyle='dotted', linewidth=0.5)
        #legend_ax_n2 += ["average alpha over visited states"]
        ax_n2.plot(np.arange(nepisodes), alpha_mean_smooth, 'k-', linestyle='dotted', linewidth=0.5)
        legend_ax_n2 += ["average alpha over visited states ({}-tap smoothed)".format(window_size)]
        ax_n2.set_yscale('log')
        ax_n2.set_ylabel("alpha")
        ax_n2.set_ylim((0,1))
        ax_n2.legend(legend_ax_n2, loc='upper right')
        ax_n.set_title("% states in summary statistics (num states = {}) and average alpha".format(env.getNumStates()))

        fig.suptitle("Convergence of the estimated value function V (" + \
                  (adaptive_type is not None and adaptive_type.name or "TD({:.2g})".format(lmbda)) + \
                  ", gamma={:.2f}, alpha={:.2f}, {})".format(gamma, alpha, alpha_update_type.name))

        if save:
            figfile = resultsdir + "/mountaincar_BENCHMARK_CV_gamma={:.2f}_lambda={}_alpha={}_adj={}_episodes={},maxt={},nx={},nv={}.png" \
                        .format(params['gamma'], params['lambda'], params['alpha'], params['adjust_alpha'],
                                test_obj.nepisodes, test_obj.max_time_steps, test_obj.env.nx, test_obj.env.nv)
            plt.savefig(figfile)
            print("Plot saved to:\n{}".format(os.path.abspath(figfile)))

    pass
