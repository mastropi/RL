# -*- coding: utf-8 -*-
"""
Created on Sat Apr  11 15:11:21 2020

@author: Daniel Mastropietro
@description: Write unit tests learning with MC(lambda)
"""

import runpy
runpy.run_path('../../setup.py')

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import unittest
from unittest_data_provider import data_provider
#from gym.utils import seeding

from Python.lib.environments import gridworlds, mountaincars
from Python.lib.agents.policies import random_walks
import Python.lib.agents as agents
from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners.episodic.discrete import mc
from Python.lib.agents.learners.episodic.discrete import AlphaUpdateType
from Python.lib.simulators.discrete import Simulator as DiscreteSimulator
from Python.lib.utils import computing

import test_utils

#reload(Python.lib.agents.learners.episodic.discrete.mc)
#reload(Python.lib.agents.learners.episodic.discrete)


# Expected results of tests that need to be the same in different tests
# (typically when checking that the result of running MC is the same as running
# MC(lambda=1.0))
# The numbers at the end of the name correspond to the case numbers in the 
# data provider lambda functions where the respective expected values are used.
EXPECTED_TEST_RANDOM_WALK_1 = [-0.000000, -0.683051, -0.683051, -0.583526, -0.555566, -0.555566, -0.555566,
                               -0.400009, -0.358463, -0.319735, -0.187761, -0.102375, -0.056671, -0.005888,
                               0.125041, 0.125041, 0.207823, 0.207823, 0.343522, 0.569533, 0.000000]
EXPECTED_TEST_RANDOM_WALK_2 = [-0.000000, -0.846154, -0.846154, -0.714286, -0.600000, -0.600000, -0.600000,
                               -0.500000, -0.411765, -0.263158, -0.200000, -0.058824, -0.000000, 0.066667,
                               0.230769, 0.230769, 0.333333, 0.333333, 0.600000, 1.000000, 0.000000]
EXPECTED_TEST_RANDOM_WALK_3 = [-0.000000, -0.702381, -0.702381, -0.611111, -0.518519, -0.518519, -0.518519,
                               -0.438596, -0.368421, -0.263158, -0.200000, 0.043590, 0.113889, 0.196970,
                               0.361176, 0.361176, 0.437451, 0.437451, 0.776471, 1.000000, 0.000000]


class Test_MC_Lambda_1DGridworld(unittest.TestCase, test_utils.EpisodeSimulation):

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

    #--------------------- TESTS OF MC: traditional MC (no lambda involved) ----------------------#
    #-------- DATA -------
    # Case number, description, expected value, parameters
    # These are the same tests 1, 2 and 3 from data_test_lambda_return_random_walk
    data_test_mc_random_walk = lambda: (
            ( 1, 'MC, no alpha adjustment',
                 EXPECTED_TEST_RANDOM_WALK_1,
                 (0.1, 1.0), False, False, 0.0 ),
            ( 2, 'MC, adjusted alpha by state count (good results if run for 200 episodes)',
                 EXPECTED_TEST_RANDOM_WALK_2,
                 (1.0, 1.0), True, False, 0.0 ),
            ( 3, 'MC, adjusted alpha by episode',
                 EXPECTED_TEST_RANDOM_WALK_3,
                 (1.0, 1.0), True,  True, 0.0 ),
        )
    #-------- DATA -------

    @data_provider(data_test_mc_random_walk)
    def test_mc_random_walk(self, casenum, desc, expected, params_alpha_gamma,
                                       adjust_alpha, adjust_alpha_by_episode, alpha_min):
        # All tests are run using seed = 1717, nepisodes = 20, start_state = 10
        print("\n*** Testing {0}, case number {1} '{2}' ***".format(self.id(), casenum, desc))
        learner_mclambda = mc.LeaMCLambda(self.env, alpha=params_alpha_gamma[0],
                                          gamma=params_alpha_gamma[1],
                                          adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode,
                                          alpha_min=alpha_min,
                                          learner_type=mc.LearnerType.MC,
                                          debug=False)
        agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mclambda)
        sim = DiscreteSimulator(self.env, agent_rw_mc, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
                                            sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                    compute_rmse=True, state_observe=10,
                                                    verbose=True, verbose_period=100,
                                                    plot=False, pause=0.1)
        observed = agent_rw_mc.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        assert np.allclose(observed, expected, atol=1E-6)
                                            
    def test_mc_random_walk_gamma_not_1(self):
        #-- All tests are run using seed = 1717, nepisodes = 20, start_state = 10
        print("\n*** Testing " + self.id() + " ***")

        # Learner and agent definition
        params = dict({'alpha': 1.0,
                       'gamma': 0.7,
                       'alpha_min': 0.0,
                       })
        learner_mclambda = mc.LeaMCLambda(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                          alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,  # First-visit is the default
                                          adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=params['alpha_min'],
                                          learner_type=mc.LearnerType.MC,
                                          debug=False)
        agent_rw_mclambda = agents.GenericAgent(self.policy_rw, learner_mclambda)

        # Simulation
        sim = DiscreteSimulator(self.env, agent_rw_mclambda, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
                                            sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                    compute_rmse=True, state_observe=10,
                                                    verbose=True, verbose_period=100,
                                                    plot=False, pause=0.1)

        expected = np.array([ 0.000000, -0.496388, -0.197257, -0.077399, -0.043423, -0.016056, -0.000258,
                             -0.000125, -0.000079, -0.000045,  0.000455,  0.000814,  0.001235,  0.001883,
                              0.006338,  0.009091,  0.014105,  0.035182,  0.137579,  0.640122,  0.000000])
        observed = agent_rw_mclambda.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        if self.plot:
            # Add the lambda parameter in the params dictionary which is required by plot_results()
            params['lambda'] = 1.0
            self.plot_results(params,
                              observed, self.V_true, RMSE_by_episode, learning_info['alpha_mean'],
                              max_rmse=self.max_rmse, color_rmse=self.color_rmse)
        assert np.allclose(observed, expected, atol=1E-6)

    def test_mc_random_walk_rmse_twice(self):
        #-- All tests are run using seed = 1717, nepisodes = 20, start_state = 10
        print("\n*** Testing " + self.id() + " ***")

        # Learner and agent
        params = dict({'alpha': 0.3,
                       'gamma': 1.0,
                       'alpha_min': 0.0,
                       })
        learner_mclambda = mc.LeaMCLambda(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                          alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,  # First-visit is the default
                                          adjust_alpha=True, adjust_alpha_by_episode=False, alpha_min=params['alpha_min'],
                                          learner_type=mc.LearnerType.MC,
                                          debug=False)
        agent_rw_mclambda = agents.GenericAgent(self.policy_rw, learner_mclambda)

        # Simulation
        sim = DiscreteSimulator(self.env, agent_rw_mclambda, debug=False)

        # First run
        _, _, RMSE_by_episode, MAPE_by_episode, _ = sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                            compute_rmse=True, state_observe=19,
                                                            verbose=True, verbose_period=100,
                                                            plot=False, pause=0.001)

        # Expected RMSE with the following settings:
        # 19-size gridworld
        # alpha = 0.3, gamma = 1.0
        # adjust_alpha = True, adjust_alpha_by_episode = False, alpha_min = 0.0
        # seed = 1717, nepisodes=20, start_state = 10
        rmse_expected = 0.29397963

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
    #--------------------- TESTS OF MC: traditional MC (no lambda involved) ----------------------#


    #------------------------ TESTS OF MC(LAMBDA): MC as lambda-return ---------------------------#
    #-------- DATA -------
    # Case number, description, expected value, parameters
    data_test_lambda_return_random_walk = lambda: (
            ( 1, 'MC, no alpha adjustment',
                 EXPECTED_TEST_RANDOM_WALK_1,
                (0.1, 1.0, 1.0), False, False, 0.0 ),
            ( 2, 'MC, adjusted alpha by state count (good results if run for 200 episodes)',
                 EXPECTED_TEST_RANDOM_WALK_2,
                (1.0, 1.0, 1.0), True, False, 0.0 ),
            ( 3, 'MC, adjusted alpha by episode',
                 EXPECTED_TEST_RANDOM_WALK_3,
                (1.0, 1.0, 1.0), True,  True, 0.0 ),
            ( 4, 'L-return (lambda<1), no alpha adjustment',
                 [-0.000000, -0.648738, -0.440867, -0.228680, -0.172241, -0.093122, -0.023765,
                  -0.016186, -0.012195, -0.007552, 0.002211, 0.006526, 0.009155, 0.015080,
                  0.035585, 0.049038, 0.071389, 0.111463, 0.213302, 0.552812, 0.000000],
                (0.2, 1.0, 0.7), False, False, 0.0 ),
            ( 5, 'L-return (lambda<1), adjusted alpha by state count',
              [-0.000000, -0.622132, -0.413372, -0.199682, -0.128330, -0.055435, -0.017840,
               -0.010864, -0.006091, -0.003125, 0.000993, 0.003451, 0.005608, 0.012755,
               0.028038, 0.039561, 0.058164, 0.104930, 0.264852, 0.694180, 0.000000],
                (1.0, 1.0, 0.7), True,  False, 0.0 ),
        )
    #-------- DATA -------

    @data_provider(data_test_lambda_return_random_walk)
    def test_lambda_return_random_walk(self, casenum, desc, expected, params_alpha_gamma_lambda,
                                                        adjust_alpha, adjust_alpha_by_episode, alpha_min):
        # All tests are run using seed = 1717, nepisodes = 20, start_state = 10
        print("\n*** Testing {0}, case number {1} ***".format(self.id(), casenum))
        learner_mclambda = mc.LeaMCLambda(self.env, alpha=params_alpha_gamma_lambda[0],
                                          gamma=params_alpha_gamma_lambda[1],
                                          lmbda=params_alpha_gamma_lambda[2],
                                          adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode,
                                          alpha_min=alpha_min,
                                          learner_type=mc.LearnerType.LAMBDA_RETURN,
                                          debug=False)
        agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mclambda)
        sim = DiscreteSimulator(self.env, agent_rw_mc, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
                                            sim.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                                                    compute_rmse=True, state_observe=10,
                                                    verbose=True, verbose_period=100,
                                                    plot=False, pause=0.1)
        observed = agent_rw_mc.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        assert np.allclose(observed, expected, atol=1E-6)
                                            
    def test_lambda_return_random_walk_gamma_not_1(self):
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
                                          learner_type=mc.LearnerType.LAMBDA_RETURN,
                                          debug=False)
        agent_rw_mclambda = agents.GenericAgent(self.policy_rw, learner_mclambda)

        # Simulation
        sim = DiscreteSimulator(self.env, agent_rw_mclambda, debug=False)
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

    def fails_test_lambda_return_random_walk_adaptive_result(self):
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
        sim = DiscreteSimulator(self.env, agent_rw_mclambda_adaptive, debug=False)
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
    #------------------------ TESTS OF MC(LAMBDA): MC as lambda-return ---------------------------#


class Test_MC_Lambda_MountainCar(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        self.seed = kwargs.pop('seed', 1717)
        self.nepisodes = kwargs.pop('nepisodes', 30) #20000) #100000) #30000) #200) #2000)
        self.max_time_steps = kwargs.pop('max_time_steps', 500)  # Maximum number of steps to run per episode
        self.normalizer = kwargs.pop('normalizer', 1)            # Normalize for the plots: Set it to max_time_steps when the rewards are NOT sparse (i.e. are -1 every where except at terminal states), o.w. set it to 1 (when rewards are sparse, i.e. they occur at terminal states)
        self.start_state = kwargs.pop('start_state', None) #(0.4, 0.07)) #None) #(0.4, 0.07))       # Position and velocity
        self.plot = kwargs.pop('plot', True)
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):  # cls is the class, in this case, class 'Test_TD_Lambda'
        # IMPORTANT: All attributes defined here can be then be referenced using self!
        # (i.e. they belong to the "object" instantiated by this class)

        #cls.env = gym.make('MountainCar-v0')   # Here we create in principle a MountainCarEnv environment because this environment is registered with gym.register() or similar.
        # See also this implementation of the Mountain Car: https://github.com/JJonahJson/MountainCar-v313/blob/master/code/main.py

        # Environment with discretized position and velocity with nx and nv points respectively
        #nx = 20  #20
        nv = 5 #20
        cls.env = mountaincars.MountainCarDiscrete(nv)

        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env)

    def no_test_mc_random_walk_onecase(self, params=None, verbose_convergence=False):
        print("\nTesting " + self.id())

        # Learner and agent
        if params is None:
            params = dict({ 'alpha': 1.0,
                            'gamma': 1.0,
                            'lambda': 1.0,
                            'alpha_min': 0.0,
                            'adjust_alpha': True,
                            'alpha_update_type': AlphaUpdateType.EVERY_STATE_VISIT,
                            'learner_type': mc.LearnerType.MC, #mc.LearnerType.LAMBDA_RETURN,
                            'reset_method': ResetMethod.ALLZEROS, #ResetMethod.RANDOM_NORMAL, #ResetMethod.ALLZEROS,
                            'reset_params': dict({'loc': -500, 'scale': 10}),
                           })
        learner_mc = mc.LeaMCLambda(  self.env, alpha=params['alpha'], gamma=params['gamma'],
                                      alpha_update_type=params['alpha_update_type'],
                                      adjust_alpha=params['adjust_alpha'], adjust_alpha_by_episode=False,
                                      alpha_min=params['alpha_min'],
                                      lmbda=params['lambda'],
                                      learner_type=params['learner_type'],
                                      reset_method=params['reset_method'], reset_params=params['reset_params'], reset_seed=self.seed,
                                      debug=False)
        agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mc)

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
            print("Positions ({}): {}".format(len(self.env.get_positions()), self.env.get_positions()))
            print("Velocities ({}): {}".format(len(self.env.get_velocities()), self.env.get_velocities()))
            print("Goal is reached at position: {}".format(self.env.goal_position))
            idx_start_state = None
        sim = DiscreteSimulator(self.env, agent_rw_mc, debug=False)

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

        observed = self.env.reshape_from_1d_to_2d(agent_rw_mc.getLearner().getV().getValues())
        state_counts = self.env.reshape_from_1d_to_2d(np.asarray(sim.agent.getLearner().getStateCounts()))
        print("\n{}, observed: ".format(self.id(), observed))
        #assert np.allclose(observed, expected, atol=1E-6)

        if self.plot:
            window_size = max(1, int(self.nepisodes/100))
            deltaV_rel_max_smooth = computing.smooth(learning_info['deltaV_rel_max_signed'], window_size=window_size)
            
            fig = plt.figure(figsize=(20,10))
            ax, ax_n = fig.subplots(1,2)
    
            legend_ax = []
            ax.plot(np.arange(self.nepisodes+1), learning_info['V_abs_mean'] / self.normalizer, 'b--')
            ax.plot(np.arange(self.nepisodes+1), learning_info['V_abs_mean_weighted'] / self.normalizer, 'b-')
            legend_ax += ["Average |V|", "Average |V| (weighted by state count)"]
            ax.set_ylim((0, ax.get_ylim()[1]))
            ax.set_xlabel("Episode number")
            ax.set_ylabel("Average |V|" + (self.normalizer != 1.0 and " normalized by dividing by {}" or "").format(self.normalizer))
            ax.legend(legend_ax, loc='upper left')
    
            ax2 = ax.twinx()
            legend_ax2 = []
            ax2.plot(np.arange(self.nepisodes+1), deltaV_rel_max_smooth*100, 'r-', linewidth=1)
            ax2.plot(np.arange(self.nepisodes+1), learning_info['deltaV_rel_abs_mean']*100, 'r--', linewidth=0.5)
            ax2.plot(np.arange(self.nepisodes+1), learning_info['prop_states_deltaV_relevant']*100, 'k--', linewidth=0.5)
            legend_ax2 += ["max|relative change| with sign ({}-tap-smoothed)".format(window_size),
                           "mean|relative change|",
                           "proportion states with |rel change| > 1%"]
            ax2.axhline(0.0, color='gray', linewidth=0.5)
            ax2.set_yscale('symlog')
            ax2.set_ylabel("|relative delta(V)| % (log scale)")
            ax2.set_ylim( -max(ax2.get_ylim()), +max(ax2.get_ylim()) )
            ax2.legend(legend_ax2, loc='upper right')

            # Number of states used in the computation of the summary statistics at each episode
            ax_n.plot(np.arange(self.nepisodes+1), learning_info['V_abs_n'] / self.env.getNumStates() * 100, 'g-')
            ax_n.set_xlabel("Episode number")
            ax_n.set_ylabel("% states")
            ax_n.set_ylim((0,100))
            ax_n.legend(["% states"], loc='upper left')
            ax_n.set_title("% states used for summary statistics (num states = {}) (average is computed on states with visit count > 0)".format(self.env.getNumStates()))
    
            fig.suptitle("Convergence of the estimated value function V (gamma={:.2f}, lambda={:.2f}, alpha={:.2f}, {}, max #steps = {})" \
                        .format(params['gamma'], params['lambda'], params['alpha'], params['alpha_update_type'].name, self.max_time_steps))


            fig = plt.figure(figsize=(20,10))
            ax, ax2 = fig.subplots(1,2)

            #ax.plot(self.env.reshape_from_2d_to_1d(state_counts))
            #for x in range(0, np.prod(self.env.shape), self.env.shape[0]):
            #    ax.axvline(x, color="gray")
            #ax.axhline(0, color="gray")
            #ax.set_xlabel("linear state index (grouped by {})".format(self.env.getShapeNames()[1]))
            #ax.set_ylabel("Visit count")
            #ax.set_title("State visit count (grouped by {})".format(self.env.getShapeNames()[1]))

            values = agent_rw_mc.getLearner().getV().getValues()
            ax.hist(values, bins=30, weights=np.repeat(1/len(values), len(values))*100)
            ax.set_ylim((0,100))
            ax.set_xlabel("Value function")
            ax.set_ylabel("Percent count")
            ax.set_title("Distribution of V(s) values")

            positions, velocities = self.env.get_positions(), self.env.get_velocities() 
            ax2.plot(positions, np.sum(state_counts, axis=self.env.getVelocityDimension()), '.-', color=self.env.getPositionColor()) # Sum along the different velocities
            ax2.plot(velocities, np.sum(state_counts, axis=self.env.getPositionDimension()), '.-', color=self.env.getVelocityColor()) # Sum along the different positions
            ax2.legend(["Visit count on positions", "Visit count on velocities"])
            ax2.set_xlabel("Position / Velocity")
            ax2.set_ylabel("Visit count")
            ax2.set_title("Visit counts by dimension (position / velocity)")

        return observed, state_counts, params, sim, learning_info

    def test_mc_vs_lambda_return_random_walk_onecase(self, verbose_convergence=False):
        print("\nTesting " + self.id())

        # Learner and agent
        params = dict({ 'alpha': 1.0,
                        'gamma': 1.0,
                        'lambda': 1.0,
                        'alpha_min': 0.0,
                        'adjust_alpha': True,
                        'alpha_update_type': AlphaUpdateType.EVERY_STATE_VISIT,
                       })

        learner_mc = mc.LeaMCLambda(  self.env, alpha=params['alpha'], gamma=params['gamma'],
                                      alpha_update_type=params['alpha_update_type'],
                                      adjust_alpha=params['adjust_alpha'], adjust_alpha_by_episode=False,
                                      alpha_min=params['alpha_min'],
                                      lmbda=params['lambda'],
                                      learner_type=mc.LearnerType.MC,
                                      reset_method=ResetMethod.ALLZEROS,
                                      debug=False)
        learner_lambda_return = mc.LeaMCLambda(  self.env, alpha=params['alpha'], gamma=params['gamma'],
                                      alpha_update_type=params['alpha_update_type'],
                                      adjust_alpha=params['adjust_alpha'], adjust_alpha_by_episode=False,
                                      alpha_min=params['alpha_min'],
                                      lmbda=params['lambda'],
                                      learner_type=mc.LearnerType.LAMBDA_RETURN,
                                      reset_method=ResetMethod.ALLZEROS,
                                      debug=False)
        agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mc)
        agent_rw_lambda_return = agents.GenericAgent(self.policy_rw, learner_lambda_return)

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
        sim_mc = DiscreteSimulator(self.env, agent_rw_mc, debug=False)
        sim_lambda_return = DiscreteSimulator(self.env, agent_rw_lambda_return, debug=False)

        # MC execution
        time_start = timer()
        _, _, _, _, learning_info = sim_mc.run(nepisodes=self.nepisodes, max_time_steps=self.max_time_steps,
                                        start=idx_start_state, seed=self.seed,
                                        compute_rmse=False, state_observe=None,
                                        verbose=True, verbose_period=max(1, int(self.nepisodes/10)),
                                        verbose_convergence=verbose_convergence,
                                        plot=False, pause=0.001)
        time_end = timer()
        exec_time = time_end - time_start
        print("[MC] Execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

        # Lambda-Return
        time_start = timer()
        _, _, _, _, learning_info = sim_lambda_return.run(nepisodes=self.nepisodes, max_time_steps=self.max_time_steps,
                                        start=idx_start_state, seed=self.seed,
                                        compute_rmse=False, state_observe=None,
                                        verbose=True, verbose_period=max(1, int(self.nepisodes/10)),
                                        verbose_convergence=verbose_convergence,
                                        plot=False, pause=0.001)
        time_end = timer()
        exec_time = time_end - time_start
        print("[Lambda-Return] Execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

        print("Observed estimataed value functions (MC / Lambda-Return):")
        print(np.c_[agent_rw_mc.getLearner().getV().getValues(), agent_rw_lambda_return.getLearner().getV().getValues(),
                    agent_rw_lambda_return.getLearner().getV().getValues() - agent_rw_mc.getLearner().getV().getValues()])

        # Plots
        plt.figure()
        plt.hist(np.c_[agent_rw_mc.getLearner().getV().getValues(), agent_rw_lambda_return.getLearner().getV().getValues()])
        plt.title("V(s) by each method")
        plt.legend(["MC", "Lambda-Return"])

        plt.figure()
        plt.hist(agent_rw_lambda_return.getLearner().getV().getValues() - agent_rw_mc.getLearner().getV().getValues())
        plt.title("Difference in V(s) between the two methods (Lambda-Return - MC)")

        assert np.allclose(agent_rw_mc.getLearner().getV().getValues(), agent_rw_lambda_return.getLearner().getV().getValues())


if __name__ == "__main__":
    test = True

    if test:
        #unittest.main()
        unittest.main(defaultTest="Test_MC_Lambda_1DGridworld")
        unittest.main(defaultTest="Test_MC_Lambda_MountainCar")

    else:
        test_obj = Test_MC_Lambda_MountainCar()
        test_obj.setUpClass()
        state_values, state_counts, params, sim, learning_info = test_obj.test_mc_random_walk_onecase()
