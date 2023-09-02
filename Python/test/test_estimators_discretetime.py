# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:19:38 2022

@author: Daniel Mastropietro
@description: Unit tests for estimators (V, Q) on discrete-time MDPs.
@details: Naming conventions follow the instructions given in test_conventions.txt.
"""

import runpy
runpy.run_path('../../setup.py')

import unittest
from unittest_data_provider import data_provider

from timeit import default_timer as timer
import numpy as np
from matplotlib import pyplot as plt, cm

from Python.lib import estimators

import Python.lib.agents as agents
from Python.lib.agents import GenericAgent
from Python.lib.agents.learners import LearningCriterion, ResetMethod
from Python.lib.agents.learners.episodic.discrete import mc, td, AlphaUpdateType
from Python.lib.agents.policies import random_walks

from Python.lib.environments import gridworlds, mountaincars

from Python.lib.simulators.discrete import Simulator as DiscreteSimulator

from Python.lib.utils import computing

import test_utils


class Test_EstStateValueV_MetOffline_EnvDeterministicNextState(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env_grid = gridworlds.EnvGridworld1D(length=19+2)
        cls.env_grid_oneterminal = gridworlds.EnvGridworld1D_OneTerminalState(length=100)
        cls.env_mountain = mountaincars.MountainCarDiscrete(5)
            ## Note: we use more density to discretize the positions because we need to make sure that the car reaches the terminal state
            ## and this is not given if the discretization on the right side of the ramp is not dense enough because
            ## the new position is determined by x + v and vmax = 0.07, so the closest non-terminal x should 0.5 - 0.07 = 0.43
            ## where 0.5 = self.goal_position defined in mountain_cars.py in the MountainCarEnv class.

    def test_EnvGridworld1D_PolRandomWalk_Met_TestOneCase(self):
        max_iter = 1000
        estimator = estimators.EstimatorValueFunctionOfflineDeterministicNextState(self.env_grid, gamma=1.0)
        niter, mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs = estimator.estimate_state_values_random_walk(synchronous=True, max_delta=1E-6, max_delta_rel=np.nan, max_iter=max_iter)

        state_values = estimator.getV().getValues()
        print("\n***** Estimated state values on {}-state '1D gridworld environment' after {} iterations out of {}, with:\n" \
              .format(self.env_grid.getNumStates(), niter, max_iter) + \
                "mean|delta(V)| = {}, max|delta(V)| = {}, max|delta_rel(V)| = {}:\n{}" \
              .format(mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs, estimator.getV().getValues()))
        print("Average V = {}, Average |V| = {}".format(np.mean(state_values), np.mean(np.abs(state_values))))

        plt.figure()
        plt.plot(state_values, 'r.-')
        plt.title("1D gridworld: Estimated value function V(s)")

    def test_EnvGridworld1DOneTerminal_PolRandomWalk_Met_TestOneCase(self):
        max_iter = 100
        estimator = estimators.EstimatorValueFunctionOfflineDeterministicNextState(self.env_grid_oneterminal, gamma=1.0)
        niter, mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs = estimator.estimate_state_values_random_walk(synchronous=True, max_delta=1E-1, max_delta_rel=np.nan, max_iter=max_iter)

        state_values = estimator.getV().getValues()
        print("\n***** Estimated state values on {}-state '1D gridworld environment with left transient state and left terminal state' after {} iterations out of {}, with:\n" \
              .format(self.env_grid.getNumStates(), niter, max_iter) + \
                "mean|delta(V)| = {}, max|delta(V)| = {}, max|delta_rel(V)| = {}:\n{}" \
              .format(mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs, estimator.getV().getValues()))
        print("Average V = {}, Average |V| = {}".format(np.mean(state_values), np.mean(np.abs(state_values))))

        plt.figure()
        plt.plot(state_values, 'r.-')
        plt.title("1D-gridworld with one terminal state: Estimated value function V(s)")

    def test_EnvMountainCarDiscreteActions_PolRandomWalk_Met_TestOneCase(self):
        max_iter = 100
        estimator = estimators.EstimatorValueFunctionOfflineDeterministicNextState(self.env_mountain, gamma=1.0)
        niter, mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs = \
            estimator.estimate_state_values_random_walk(synchronous=True, max_delta=np.nan, max_delta_rel=1E-5, max_iter=max_iter,
                                                        reset_method=ResetMethod.ALLZEROS, reset_params=dict({'min': 0.2, 'max': 0.8}), reset_seed=1713)

        state_values = estimator.getV().getValues()
        state_values_2d = self.env_mountain.reshape_from_1d_to_2d(state_values)
        non_terminal_states = np.array(self.env_mountain.getNonTerminalStates())
        min_state_value_non_terminal, max_state_value_non_terminal =  np.min(state_values[non_terminal_states]), np.max(state_values[non_terminal_states])

        print("\n***** Estimated state values on {}-state Mountain Car environment after {} iterations out of {}, with:\n" \
                "mean|delta(V)| = {}, max|delta(V)| = {}, max|delta_rel(V)| = {} (mean(V) = {:.6f}, mean|V| = {:.6f}):\n{}" \
              .format(self.env_mountain.getNumStates(), niter, max_iter, mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs, np.mean(state_values), np.mean(np.abs(state_values)), state_values))

        plt.figure(figsize=(20,10))
        plt.plot(state_values, 'r.-')
        plt.title("Estimated value function (mean|V| = {})".format(np.mean(np.abs(state_values))))

        fig = plt.figure(figsize=(20,10))
        ax = fig.subplots(1,1)
        ax.plot(state_values, 'r.-')
        for x in range(0, np.prod(self.env_mountain.shape), self.env_mountain.shape[0]):
            plt.axvline(x, color="gray")
        ax.set_ylim((min_state_value_non_terminal, max_state_value_non_terminal))
        ax.set_xlabel("linear state index (grouped by {})".format(self.env_mountain.getShapeNames()[1]))
        ax.set_title("Estimated value function (scaled to non-terminal states")

        positions, velocities = self.env_mountain.get_positions(), self.env_mountain.get_velocities()
        fig = plt.figure(figsize=(20,10))
        ax = fig.subplots(1,1)
        ax.plot(positions, np.mean(state_values_2d,
                axis=self.env_mountain.getVelocityDimension()), '.-', color=self.env_mountain.getPositionColor()) # Sum along the different velocities
        ax.plot(velocities, np.mean(state_values_2d,
                axis=self.env_mountain.getPositionDimension()), '.-', color=self.env_mountain.getVelocityColor()) # Sum along the different positions
        ax.legend(["mean(V) on positions", "mean(V) on velocities"])
        ax.set_ylim((np.min(state_values[non_terminal_states]), np.max(state_values[non_terminal_states])))
        ax.set_xlabel("Position / Velocity")
        ax.set_ylabel("Average V")
        ax.set_title("Average V by dimension (position / velocity) (scaled on non-terminal states range)")


class Test_EstStateValueV_EnvGridworlds(unittest.TestCase, test_utils.EpisodeSimulation):
    # Note: nice explanation about the three types of methods that can be defined in Python: instance, class, static
    # https://stackoverflow.com/questions/54264073/what-is-the-use-and-when-to-use-classmethod-in-python
    # See the only answer by Navy Cheng.

    # Expected results of tests that need to be the same in different tests
    # (typically when checking that the result of running MC is the same as running MC(lambda=1.0))
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

    @classmethod
    def setUpClass(cls):
        # IMPORTANT: All attributes defined here can be then be referenced using self!
        # (i.e. they belong to the "object" instantiated by this class)
        # Plot settings
        cls.max_rmse = 0.8
        cls.color_rmse = "blue"

        # The environment
        cls.nS = 19
        cls.env1d = gridworlds.EnvGridworld1D(length=cls.nS+2)

        # True state value function when gamma = 1.0
        cls.V_true = np.arange(-cls.nS - 1, cls.nS + 2, 2) / (cls.nS + 1)
        cls.V_true[0] = cls.V_true[-1] = 0

        # Random walk policy on the above environment
        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env1d)

        # Plotting parameters
        cls.colormap = cm.get_cmap("jet")

    # --------------------- TESTS OF MC: traditional MC (no lambda involved) ----------------------#
    # -------- DATA -------
    # Case number, description, expected value, parameters
    # These are the same tests 1, 2 and 3 from data_test_lambda_return_random_walk
    data_test_EnvGridworld1D_PolRandomWalk_MetMC_TestSeveralAlphasAndAlphaAdjustments = lambda: (
        (1, 'MC, no alpha adjustment',
         Test_EstStateValueV_EnvGridworlds.EXPECTED_TEST_RANDOM_WALK_1,
         (0.1, 1.0), False, False, 0.0),
        (2, 'MC, adjusted alpha by state count (good results if run for 200 episodes)',
         Test_EstStateValueV_EnvGridworlds.EXPECTED_TEST_RANDOM_WALK_2,
         (1.0, 1.0), True, False, 0.0),
        (3, 'MC, adjusted alpha by episode',
         Test_EstStateValueV_EnvGridworlds.EXPECTED_TEST_RANDOM_WALK_3,
         (1.0, 1.0), True, True, 0.0),
    )

    # -------- DATA -------

    @data_provider(data_test_EnvGridworld1D_PolRandomWalk_MetMC_TestSeveralAlphasAndAlphaAdjustments)
    def test_EnvGridworld1D_PolRandomWalk_MetMC_TestSeveralAlphasAndAlphaAdjustments(self, casenum, desc, expected,
                                                                                     params_alpha_gamma,
                                                                                     adjust_alpha, adjust_alpha_by_episode,
                                                                                     alpha_min):
        print("\n*** Running test {0}, case number {1} '{2}' ***".format(self.id(), casenum, desc))

        # Simulation setup
        seed = 1717
        nepisodes = 20
        start_state = 10

        learner_mclambda = mc.LeaMCLambda(self.env1d, alpha=params_alpha_gamma[0],
                                          gamma=params_alpha_gamma[1],
                                          adjust_alpha=adjust_alpha,
                                          adjust_alpha_by_episode=adjust_alpha_by_episode,
                                          alpha_min=alpha_min,
                                          learner_type=mc.LearnerType.MC,
                                          debug=False)
        agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mclambda)
        sim = DiscreteSimulator(self.env1d, agent_rw_mc, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
            sim.run(nepisodes=nepisodes, start=start_state, seed=seed,
                    compute_rmse=True, state_observe=10,
                    verbose=True, verbose_period=100,
                    plot=False, pause=0.1)
        observed = agent_rw_mc.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))

        assert self.nS == 19 and \
               seed == 1717 and \
               nepisodes == 20 and \
               start_state == 10
        assert np.allclose(observed, expected, atol=1E-6)

    def test_EnvGridworld1D_PolRandomWalk_MetMC_TestGammaSmallerThan1(self):
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717
        nepisodes = 20
        start_state = 10
        plot = True

        # Learner and agent definition
        params = dict({'alpha': 1.0,
                       'gamma': 0.7,
                       'alpha_min': 0.0,
                       })
        learner_mclambda = mc.LeaMCLambda(self.env1d, alpha=params['alpha'], gamma=params['gamma'],
                                          alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                                          # First-visit is the default
                                          adjust_alpha=True, adjust_alpha_by_episode=False,
                                          alpha_min=params['alpha_min'],
                                          learner_type=mc.LearnerType.MC,
                                          debug=False)
        agent_rw_mclambda = agents.GenericAgent(self.policy_rw, learner_mclambda)

        # Simulation
        sim = DiscreteSimulator(self.env1d, agent_rw_mclambda, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
            sim.run(nepisodes=nepisodes, start=start_state, seed=seed,
                    compute_rmse=True, state_observe=10,
                    verbose=True, verbose_period=100,
                    plot=False, pause=0.1)

        expected = np.array([0.000000, -0.496388, -0.197257, -0.077399, -0.043423, -0.016056, -0.000258,
                             -0.000125, -0.000079, -0.000045, 0.000455, 0.000814, 0.001235, 0.001883,
                             0.006338, 0.009091, 0.014105, 0.035182, 0.137579, 0.640122, 0.000000])
        observed = agent_rw_mclambda.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        if plot:
            # Add the lambda parameter in the params dictionary which is required by plot_results()
            params['lambda'] = 1.0
            self.plot_results(params, nepisodes,
                              observed, self.V_true, RMSE_by_episode, learning_info['alpha_mean'],
                              max_rmse=self.max_rmse, color_rmse=self.color_rmse)
        assert self.nS == 19 and \
               seed == 1717 and \
               nepisodes == 20 and \
               start_state == 10 and \
               params['alpha'] == 1.0 and \
               params['gamma'] == 0.7 and \
               params['alpha_min'] == 0.0
        assert np.allclose(observed, expected, atol=1E-6)

    def test_EnvGridworld1D_PolRandomWalk_MetMC_TestRMSETwice(self):
        # -- All tests are run using seed = 1717, nepisodes = 20, start_state = 10
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717
        nepisodes = 20
        start_state = 10
        plot = True

        # Learner and agent
        params = dict({'alpha': 0.3,
                       'gamma': 1.0,
                       'alpha_min': 0.0,
                       })
        learner_mclambda = mc.LeaMCLambda(self.env1d, alpha=params['alpha'], gamma=params['gamma'],
                                          alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                                          # First-visit is the default
                                          adjust_alpha=True, adjust_alpha_by_episode=False,
                                          alpha_min=params['alpha_min'],
                                          learner_type=mc.LearnerType.MC,
                                          debug=False)
        agent_rw_mclambda = agents.GenericAgent(self.policy_rw, learner_mclambda)

        # Simulation
        sim = DiscreteSimulator(self.env1d, agent_rw_mclambda, debug=False)

        # First run
        _, _, RMSE_by_episode, MAPE_by_episode, _ = sim.run(nepisodes=nepisodes, start=start_state,
                                                            seed=seed,
                                                            compute_rmse=True, state_observe=19,
                                                            verbose=True, verbose_period=100,
                                                            plot=False, pause=0.001)

        # Expected RMSE with the following settings:
        # alpha_update_Type = AlphaUpdateType.FIRST_STATE_VISIT, adjust_alpha = True, adjust_alpha_by_episode = False
        # + the settings specified in the assertion
        assert self.nS == 19 and \
               seed == 1717 and \
               nepisodes == 20 and \
               start_state == 10 and \
               params['alpha'] == 0.3 and \
               params['gamma'] == 1.0 and \
               params['alpha_min'] == 0.0
        rmse_expected = 0.29397963

        rmse_observed = np.mean(RMSE_by_episode)
        print("First run: average RMSE over {} episodes: {:.8f}".format(nepisodes, rmse_observed))
        assert np.allclose(rmse_observed, rmse_expected, atol=1E-6)

        # Second run
        _, _, RMSE_by_episode, MAPE_by_episode, _ = sim.run(nepisodes=nepisodes, start=start_state,
                                                            seed=seed,
                                                            compute_rmse=True, state_observe=19,
                                                            verbose=True, verbose_period=100,
                                                            plot=False, pause=0.001)
        rmse_observed = np.mean(RMSE_by_episode)
        print("Second run: average RMSE over {} episodes: {:.8f}".format(nepisodes, rmse_observed))
        assert np.allclose(rmse_observed, rmse_expected, atol=1E-6)
    # --------------------- TESTS OF MC: traditional MC (no lambda involved) ----------------------#

    # ------------------------ TESTS OF MC(LAMBDA): MC as lambda-return ---------------------------#
    # -------- DATA -------
    # Case number, description, expected value, parameters
    data_test_EnvGridworld1D_PolRandomWalk_MetLambdaReturn_TestSeveralAlphasLambdasAlphaAdjustments = lambda: (
        (1, 'MC, no alpha adjustment',
         Test_EstStateValueV_EnvGridworlds.EXPECTED_TEST_RANDOM_WALK_1,
         (0.1, 1.0, 1.0), False, False, 0.0),
        (2, 'MC, adjusted alpha by state count (good results if run for 200 episodes)',
         Test_EstStateValueV_EnvGridworlds.EXPECTED_TEST_RANDOM_WALK_2,
         (1.0, 1.0, 1.0), True, False, 0.0),
        (3, 'MC, adjusted alpha by episode',
         Test_EstStateValueV_EnvGridworlds.EXPECTED_TEST_RANDOM_WALK_3,
         (1.0, 1.0, 1.0), True, True, 0.0),
        (4, 'L-return (lambda<1), no alpha adjustment',
         [-0.000000, -0.648738, -0.440867, -0.228680, -0.172241, -0.093122, -0.023765,
          -0.016186, -0.012195, -0.007552, 0.002211, 0.006526, 0.009155, 0.015080,
          0.035585, 0.049038, 0.071389, 0.111463, 0.213302, 0.552812, 0.000000],
         (0.2, 1.0, 0.7), False, False, 0.0),
        (5, 'L-return (lambda<1), adjusted alpha by state count',
         [-0.000000, -0.622132, -0.413372, -0.199682, -0.128330, -0.055435, -0.017840,
          -0.010864, -0.006091, -0.003125, 0.000993, 0.003451, 0.005608, 0.012755,
          0.028038, 0.039561, 0.058164, 0.104930, 0.264852, 0.694180, 0.000000],
         (1.0, 1.0, 0.7), True, False, 0.0),
    )

    # -------- DATA -------

    @data_provider(data_test_EnvGridworld1D_PolRandomWalk_MetLambdaReturn_TestSeveralAlphasLambdasAlphaAdjustments)
    def test_EnvGridworld1D_PolRandomWalk_MetLambdaReturn_TestSeveralAlphasLambdasAlphaAdjustments(self, casenum, desc,
                                                                                                   expected,
                                                                                                   params_alpha_gamma_lambda,
                                                                                                   adjust_alpha,
                                                                                                   adjust_alpha_by_episode,
                                                                                                   alpha_min):
        print("\n*** Running test {0}, case number {1} ***".format(self.id(), casenum))

        # Simulation setup
        seed = 1717
        nepisodes = 20
        start_state = 10
        plot = True

        learner_mclambda = mc.LeaMCLambda(self.env1d, alpha=params_alpha_gamma_lambda[0],
                                          gamma=params_alpha_gamma_lambda[1],
                                          lmbda=params_alpha_gamma_lambda[2],
                                          adjust_alpha=adjust_alpha,
                                          adjust_alpha_by_episode=adjust_alpha_by_episode,
                                          alpha_min=alpha_min,
                                          learner_type=mc.LearnerType.LAMBDA_RETURN,
                                          debug=False)
        agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mclambda)
        sim = DiscreteSimulator(self.env1d, agent_rw_mc, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
            sim.run(nepisodes=nepisodes, start=start_state, seed=seed,
                    compute_rmse=True, state_observe=10,
                    verbose=True, verbose_period=100,
                    plot=False, pause=0.1)
        observed = agent_rw_mc.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))

        assert self.nS == 19 and \
               seed == 1717 and \
               nepisodes == 20 and \
               start_state == 10
        assert np.allclose(observed, expected, atol=1E-6)

    def test_EnvGridworld1D_PolRandomWalk_MetLambdaReturn_TestGammaLessThan1(self):
        # -- All tests are run using seed = 1717, nepisodes = 20, start_state = 10
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717
        nepisodes = 20
        start_state = 10
        plot = True

        # Learner and agent definition
        params = dict({'alpha': 1.0,
                       'gamma': 0.7,
                       'lambda': 0.8,
                       'alpha_min': 0.0,
                       })
        learner_mclambda = mc.LeaMCLambda(self.env1d, alpha=params['alpha'], gamma=params['gamma'],
                                          lmbda=params['lambda'],
                                          alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                                          # First-visit is the default
                                          adjust_alpha=True, adjust_alpha_by_episode=False,
                                          alpha_min=params['alpha_min'],
                                          learner_type=mc.LearnerType.LAMBDA_RETURN,
                                          debug=False)
        agent_rw_mclambda = agents.GenericAgent(self.policy_rw, learner_mclambda)

        # Simulation
        sim = DiscreteSimulator(self.env1d, agent_rw_mclambda, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
            sim.run(nepisodes=nepisodes, start=start_state, seed=seed,
                    compute_rmse=True, state_observe=10,
                    verbose=True, verbose_period=100,
                    plot=False, pause=0.1)

        assert self.nS == 19 and \
               seed == 1717 and \
               nepisodes == 20 and \
               start_state == 10 and \
               params['alpha'] == 1.0 and \
               params['gamma'] == 0.7 and \
               params['lambda'] == 0.8 and \
               params['alpha_min'] == 0.0
        expected = np.array([0.000000, -0.471314, -0.150874, -0.044135, -0.021683,
                             -0.007209, -0.000881, -0.000375, -0.000158, -0.000065,
                             0.000034, 0.000097, 0.000210, 0.000553, 0.002688,
                             0.004801, 0.009370, 0.023406, 0.123022, 0.638221, 0.000000])
        observed = agent_rw_mclambda.getLearner().getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        if plot:
            self.plot_results(params, nepisodes,
                              observed, self.V_true, RMSE_by_episode, learning_info['alpha_mean'],
                              max_rmse=self.max_rmse, color_rmse=self.color_rmse)
        assert np.allclose(observed, expected, atol=1E-6)

    def fails_test_EnvGridworld1D_PolRandomWalk_MetLambdaReturnAdaptive_TestOneCase(self):
        # NOTE: (2021/10/16) This test currently fails for two reasons:
        # - the `observed` array is an array of all -1.0 followed by all +1.0 (i.e. a step function)
        # - the plot_results() call at the end fails because the alpha_mean_by_episode_mean variable used when calling
        # plt.plot() inside plot_results() has zero length.
        # So, it seems that nothing is really done or learned by the learner_mclambda_adaptive object.
        print("\n*** Running test " + self.id())

        # Simulation setup
        seed = 1717
        nepisodes = 20
        start_state = 10
        plot = True

        # Learner and agent definition
        params = dict({'alpha': 1.0,
                       'gamma': 1.0,
                       'lambda': 1.0,
                       'alpha_min': 0.0,
                       })
        learner_mclambda_adaptive = mc.LeaMCLambdaAdaptive(self.env1d, alpha=params['alpha'], gamma=params['gamma'],
                                                           lmbda=params['lambda'])
        agent_rw_mclambda_adaptive = agents.GenericAgent(self.policy_rw, learner_mclambda_adaptive)

        # Simulation
        sim = DiscreteSimulator(self.env1d, agent_rw_mclambda_adaptive, debug=False)
        _, _, RMSE_by_episode, MAPE_by_episode, learning_info = sim.run(nepisodes=nepisodes,
                                                                        start=start_state, seed=seed,
                                                                        compute_rmse=True, state_observe=10,
                                                                        verbose=True, verbose_period=100,
                                                                        plot=False, pause=0.1)

        # Expected state values with alpha = 0.2, gamma = 0.9, lambda = 0.8
        assert self.nS == 19 and \
               seed == 1717 and \
               nepisodes == 20 and \
               start_state == 10 and \
               params['alpha'] == 1.0 and \
               params['gamma'] == 1.0 and \
               params['lambda'] == 1.0 and \
               params['alpha_min'] == 0.0
        expected = np.array([-0.0, -0.94788777, -0.93485068, -0.77635209, -0.66915289, -0.67045823,
                             -0.6319687, -0.52116762, -0.44295159, -0.20887109, -0.1027944, -0.03800919,
                             -0.03668617, 0.06142266, 0.27410733, 0.42610526, 0.50467228, 0.63018903,
                             0.6727829, 0.72310919, 0.0])
        observed = learner_mclambda_adaptive.getV().getValues()
        print("\nobserved: " + test_utils.array2str(observed))
        if plot:
            self.plot_results(params, nepisodes,
                              observed, self.V_true, RMSE_by_episode, learning_info['alpha_mean_by_episode_mean'],
                              max_rmse=self.max_rmse, color_rmse=self.color_rmse)

        assert np.allclose(observed, expected, atol=1E-6)
    # ------------------------ TESTS OF MC(LAMBDA): MC as lambda-return ---------------------------#

    def test_EnvGridworld1D_PolRandomWalk_MetTDLambda_TestSeveralAlphasLambdas(self):
        """
        This test intends to reproduce the results in Sutton 2018 (pag. 295) on the TD(lambda) algorithm applied to learn
        the state value function in a 1D gridworld. The results shown in Sutton correspond to the episode always starting
        at the same state in the 1D gridworld, namely at the middle state (in our case at state 10).
        HOWEVER, here I consider a RANDOM start state, just to use a random start state in one of the tests.
        ([2023/08/23] In fact, thanks to using this random start state, I discovered that there was an error in the implementation of
        Simulator.run() where I was resetting the simulator *after* setting the initial state distribution that selects
        the initial state (to e.g. a fixed state) and this made the private attribute self._ids_orig be reset to None,
        thus erasing the original state distribution stored in the environment, which was actually needed
        in order to be reset at the end of the simulation process and thus avoid altering the environment in the outside
        world that had been passed to the constructor of the Simulator object, which in the end changed the results of this
        test depending on whether *other* tests were run together with it or not.)
        """
        print("\n*** Running test " + self.id())

        # Possible policies and learners for agents
        pol_rw = random_walks.PolRandomWalkDiscrete(self.env1d)
        lea_td = td.LeaTDLambda(self.env1d)

        # Policy and learner to be used in the simulation
        policy = pol_rw
        learner = lea_td

        # Simulation setup
        seed = 1717
        nexperiments = 3
        nepisodes = 10
        start_state = None
        useGrid = False
        verbose = True
        debug = False

        # Define hyperparameter values
        gamma = 1.0
        if useGrid:
            n_lambdas = 11
            n_alphas = 10
            lambdas = np.linspace(0, 1, n_lambdas)
            alphas = np.linspace(0.1, 0.7, n_alphas)
        else:
            lambdas = [0, 0.4, 0.7, 0.8]
            alphas = [0.2, 0.4, 0.8]
            n_lambdas = len(lambdas)
            n_alphas = len(alphas)
        n_simul = n_lambdas*n_alphas

        # Create the figure where the plots will be added
        max_alpha = 1   # Scale for horizontal axes in RMSE vs. alpha plots
        max_rmse = 0.5  # Scale for vertical axes in RMSE vs. alpha plots
        fig, (ax_full, ax_scaled, ax_rmse_by_episode) = plt.subplots(1,3)

        # List of dictionaries, each containing the characteristic of each parameterization considered
        results_list = []
        legend_label = []

        # Average RMSE obtained at the LAST episode run for each parameter set and their standard error
        rmse_mean_values_at_end = np.nan*np.zeros((n_lambdas, n_alphas))
        rmse_se_values_at_end = np.nan*np.zeros((n_lambdas, n_alphas))
        # Average RMSE over all the episodes run for each parameter set
        rmse_episodes_mean = np.nan*np.zeros((n_lambdas, n_alphas))
        rmse_episodes_se = np.nan*np.zeros((n_lambdas, n_alphas))
        # RMSE over the episodes averaged over ALL parameter sets {alpha, lambda} run
        rmse_episodes_values = np.zeros(nepisodes+1)
        idx_simul = -1
        for idx_lmbda, lmbda in enumerate(lambdas):
            rmse_mean_lambda = []
            rmse_se_lambda = []
            rmse_episodes_mean_lambda = []
            rmse_episodes_se_lambda = []
            for alpha in alphas:
                idx_simul += 1
                if verbose:
                    print("\nParameter set {} of {}: lambda = {:.2g}, alpha = {:.2g}" \
                          .format(idx_simul+1, n_simul, lmbda, alpha))

                # Reset learner and agent (i.e. erase all memory from a previous run!)
                learner.setParams(alpha=alpha, gamma=gamma, lmbda=lmbda)
                learner.reset(reset_episode=True, reset_value_functions=True)
                agent = GenericAgent(policy, learner)

                # Simulator object
                sim = DiscreteSimulator(self.env1d, agent, seed=seed, debug=debug)
                ## NOTE: Setting the seed here implies that each set of experiments
                ## (i.e. for each combination of alpha and lambda) yields the same outcome in terms
                ## of visited states and actions.
                ## This is DESIRED --as opposed of having different state-action outcomes for different
                ## (alpha, lambda) settings-- as it better isolates the effect of alpha and lambda.
                ## VERIFIED BY RUNNING IN DEBUG MODE!

                # Run the simulation and store the results
                N_mean, rmse_mean, rmse_se, rmse_n, rmse_episodes, _, mape_episodes, _, _, learning_info = \
                                    sim.simulate(nexperiments=nexperiments,
                                                 nepisodes=nepisodes,
                                                 start=start_state,
                                                 verbose=verbose)
                results_list += [{'lmbda': lmbda,
                                  'alpha': alpha,
                                  'rmse': rmse_mean,
                                  'SE': rmse_se
                                 }]
                rmse_mean_lambda += [rmse_mean]
                rmse_se_lambda += [rmse_se]
                rmse_episodes_mean_lambda += [np.mean(rmse_episodes)]
                rmse_episodes_se_lambda += [np.std(rmse_episodes) / np.sqrt(nepisodes)]
                rmse_episodes_values += rmse_episodes

                if verbose:
                    print("\tRMSE = {:.3g} ({:.3g})".format(rmse_mean, rmse_se))

            rmse_mean_values_at_end[idx_lmbda] = np.array(rmse_mean_lambda)
            rmse_se_values_at_end[idx_lmbda] = np.array(rmse_se_lambda)
            rmse_episodes_mean[idx_lmbda] = np.array(rmse_episodes_mean_lambda)
            rmse_episodes_se[idx_lmbda] = np.array(rmse_episodes_se_lambda)

            # Plot the average RMSE for the current lambda as a function of alpha
            rmse2plot = rmse_episodes_mean_lambda
            rmse2plot_error = rmse_episodes_se_lambda
            ylabel = "Average RMSE over all {} states, first {} episodes, and {} experiments".format(self.env1d.getNumStates(), nepisodes, nexperiments)

            # Map blue to the largest lambda and red to the smallest lambda (most similar to the color scheme used in Sutton, pag. 295)
            color = self.colormap( 1 - idx_lmbda / np.max((1, n_lambdas-1)) )
            ax_full.plot(alphas, rmse2plot, '.', color=color)
            ax_full.errorbar(alphas, rmse2plot, yerr=rmse2plot_error, capsize=4, color=color)
            ax_full.set_xlabel("alpha")
            ax_full.set_ylabel(ylabel)
            legend_label += ["lam={:.2g}".format(lmbda)]

        # Average RMSE by episode for convergence analysis
        rmse_episodes_values /= n_simul

        # Scaled plot (for comparison purposes)
        for idx_lmbda, lmbda in enumerate(lambdas):
            rmse2plot = rmse_episodes_mean[idx_lmbda]
            rmse2plot_error = rmse_episodes_se[idx_lmbda]
            color = self.colormap( 1 - idx_lmbda / np.max((1, n_lambdas-1)) )
            ax_scaled.plot(alphas, rmse2plot, '.-', color=color)
            ax_scaled.errorbar(alphas, rmse2plot, yerr=rmse2plot_error, capsize=4, color=color)
            ax_scaled.set_xlim((0, max_alpha))
            ax_scaled.set_ylim((0, max_rmse))
            ax_scaled.set_xlabel("alpha")
            ax_scaled.set_ylabel(ylabel)

        # Episodic RMSE
        ax_rmse_by_episode.plot(np.arange(nepisodes+1), rmse_episodes_values, color="black")
        ax_rmse_by_episode.set_ylim((0, max_rmse))
        ax_rmse_by_episode.set_xlabel("Episode")
        ax_rmse_by_episode.set_ylabel("RMSE")
        ax_rmse_by_episode.set_title("Average RMSE by episode over ALL experiments")

        plt.figlegend(legend_label)
        fig.suptitle("{}: gamma = {:.2g}, #experiments = {}, #episodes = {}"\
                     .format(learner.__class__.__name__, gamma, nexperiments, nepisodes))

        print("Average RMSE (first 3 columns) and its standard error (last 3 columns) at last episode by lambda (rows) and alpha (columns):\n{}" \
              .format(np.c_[rmse_episodes_mean, rmse_episodes_se]))
        assert self.nS == 19 and \
               seed == 1717 and \
               nexperiments == 3 and \
               nepisodes == 10 and \
               start_state is None and \
               useGrid == False and \
               gamma == 1.0 and \
               lambdas == [0, 0.4, 0.7, 0.8] and \
               alphas == [0.2, 0.4, 0.8]
        assert np.allclose(rmse_episodes_mean, [[0.45592716, 0.40840256, 0.36065968],
                                                [0.42083822, 0.35093946, 0.29653623],
                                                [0.35946782, 0.28547410, 0.32950694],
                                                [0.32520426, 0.28876721, 0.45260387]])
        assert np.allclose(rmse_episodes_se,   [[0.01244143, 0.02139052, 0.02996329],
                                                [0.01891116, 0.03116384, 0.03592948],
                                                [0.02941829, 0.03871512, 0.02411244],
                                                [0.03452769, 0.03385807, 0.02151384]])


class Test_EstStateValueV_EnvGridworldsWithObstacles(unittest.TestCase, test_utils.EpisodeSimulation):

    @classmethod
    def setUpClass(cls):
        #-- Environment characteristics
        shape = [3, 4]
        cls.nS = np.prod(shape)
        cls.env2d = gridworlds.EnvGridworld2D_WithObstacles(shape=shape, terminal_states=set({3}), rewards_dict=dict({3: +1}), obstacles_set=set({5}))

        #-- Cycle characteritics
        # Set of absorbing states, used to define a cycle as re-entrance into the set
        # which is used to estimate the expected return (i.e. the value function)
        cls.A = set({8})

        #-- Policy characteristics
        # Random walk policy
        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env2d)

        #-- Plotting parameters
        cls.colormap = cm.get_cmap("jet")

    def setUp(self):
        # Simulation setup for ALL tests
        self.seed = 1717
        self.nepisodes = 20
        self.start_state = 8

        learner_mclambda = mc.LeaMCLambda(self.env2d, alpha=1.0,
                                          gamma=0.9,
                                          adjust_alpha=False,
                                          adjust_alpha_by_episode=False,
                                          alpha_min=0.0,
                                          learner_type=mc.LearnerType.MC,
                                            ## This implies that we use the standard Monte-Carlo learner, as opposed to the lambda-decayed Monte-Carlo learner.
                                            ## Recall that the approach is equivalent to using LeanerType.LAMBDA_RETURN with lambda = 1.0 (just verified in practice)
                                          debug=False)
        self.agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mclambda)
        self.sim_mc = DiscreteSimulator(self.env2d, self.agent_rw_mc, debug=False)

        learner_tdlambda = td.LeaTDLambda(self.env2d, alpha=1.0,
                                          gamma=0.9, lmbda=0.0,
                                          adjust_alpha=False,
                                          adjust_alpha_by_episode=False,
                                          alpha_min=0.0,
                                          debug=False)
        self.agent_rw_td = agents.GenericAgent(self.policy_rw, learner_tdlambda)
        self.sim_td = DiscreteSimulator(self.env2d, self.agent_rw_td, debug=False)

        # Expected state values for all tests
        self.expected_mc = [ 0.348678, 0.205891, 0.531441, 0.000000,
                             0.047101, 0.000000, 0.254187, 0.282430,
                             0.016423, 0.030903, 0.088629, 0.348678]
        self.expected_td = [0.042391, 0.313811, 0.810000, 0.000000,
                            0.038152, 0.000000, 0.729000, 1.000000,
                            0.022528, 0.109419, 0.430467, 0.387420]

    def test_Env_PolRandomWalk_MetMC(self):
        print(f"\n*** Running test #{self.id()}")

        state_value, state_counts, _, _, learning_info = \
            self.sim_mc.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                    compute_rmse=False, state_observe=None,
                    verbose=True, verbose_period=100,
                    plot=False, pause=0.1)
        observed = self.agent_rw_mc.getLearner().getV().getValues()
        print("\nObserved state value function: " + test_utils.array2str(observed))

        assert self.nS == 3*4 and \
               self.seed == 1717 and \
               self.nepisodes == 20 and \
               self.start_state == 8
        assert np.allclose(observed, self.expected_mc, atol=1E-6)

    def test_Env_PolRandomWalk_MetTDLambda(self):
        print(f"\n*** Running test #{self.id()}")

        state_value, state_counts, _, _, learning_info = \
            self.sim_td.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                    compute_rmse=False, state_observe=None,
                    verbose=True, verbose_period=100,
                    plot=False, pause=0.1)
        observed = self.agent_rw_td.getLearner().getV().getValues()
        print("\nObserved state value function: " + test_utils.array2str(observed))

        assert self.nS == 3*4 and \
               self.seed == 1717 and \
               self.nepisodes == 20 and \
               self.start_state == 8
        assert np.allclose(observed, self.expected_td, atol=1E-6)


class Test_EstDifferentialStateValueV_EnvGridworldsWithObstacles(unittest.TestCase, test_utils.EpisodeSimulation):
    """
    Test the estimation of the differential value function, i.e. the value function under the average reward setting

    Under this setup, the value of the discount parameter gamma is equal to 1.

    Ref on average reward setting: Sutton, chapter 10, pag. 249.
    """

    @classmethod
    def setUpClass(cls):
        # -- Environment characteristics
        shape = [3, 4]
        cls.nS = np.prod(shape)
        cls.env2d = gridworlds.EnvGridworld2D_WithObstacles(shape=shape, terminal_states=set({3}),
                                                            rewards_dict=dict({3: +1}), obstacles_set=set({5}))

        # -- Cycle characteritics
        # Set of absorbing states, used to define a cycle as re-entrance into the set
        # which is used to estimate the expected return (i.e. the value function)
        cls.A = set({8})

        # -- Policy characteristics
        # Random walk policy
        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env2d)

        # -- Plotting parameters
        cls.colormap = cm.get_cmap("jet")

    def setUp(self):
        # Simulation setup for ALL tests
        self.seed = 1717
        # We use a large number of episodes (200) so that we can test that the results with MC and with TD(lambda) are similar and close to the true ones
        # at least when the MC learner is ready... which currently is not because both the state value and the average reward are based on episodic learning,
        # NOT on the average reward criterion.
        self.nepisodes = 200
        self.start_state = 8

        learner_mclambda = mc.LeaMCLambda(self.env2d, criterion=LearningCriterion.AVERAGE, alpha=1.0,
                                          gamma=1.0,
                                          adjust_alpha=True,
                                          adjust_alpha_by_episode=False,
                                          alpha_min=0.0,
                                          learner_type=mc.LearnerType.MC,
                                          debug=False)
        self.agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mclambda)
        self.sim_mc = DiscreteSimulator(self.env2d, self.agent_rw_mc, debug=False)

        learner_tdlambda = td.LeaTDLambda(self.env2d, criterion=LearningCriterion.AVERAGE, alpha=1.0,
                                          gamma=1.0, lmbda=0.0,
                                          adjust_alpha=True,
                                          adjust_alpha_by_episode=False,
                                          alpha_min=0.0,
                                          debug=False)
        self.agent_rw_td = agents.GenericAgent(self.policy_rw, learner_tdlambda)
        self.sim_td = DiscreteSimulator(self.env2d, self.agent_rw_td, debug=False)

        # Expected state values for all tests
        self.expected_mc = [0.250013, 0.429546, 0.624561, 0.000000,
                            0.131218, 0.000000, 0.528836, 0.645549,
                            0.053294, 0.188015, 0.358143, 0.462491]
        self.expected_mc_average_reward = 0.00861585 # This value is currently (02-Sep-2023) WRONG because the average reward calculation is NOT based on the whole trajectory history, over all episodes.

        self.expected_td = [-0.001574, 0.115563, 0.368777, 0.000000,
                            -0.046231, 0.000000, 0.179900, 0.337746,
                            -0.058120, -0.050822, 0.011813, 0.060337]
        self.expected_td_average_reward = 0.0225759 # This value should be similar to expected_mc_average_reward

    def test_Env_PolRandomWalk_MetMC(self):
        print(f"\n*** Running test #{self.id()}")

        state_value, state_counts, _, _, learning_info = \
            self.sim_mc.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                         compute_rmse=False, state_observe=None,
                         verbose=True, verbose_period=100,
                         plot=False, pause=0.1)
        observed = self.agent_rw_mc.getLearner().getV().getValues()
        observed_average_reward = self.agent_rw_mc.getLearner().getAverageReward()
        print("\nObserved state value function: " + test_utils.array2str(observed))
        print(f"\nAverage reward: {observed_average_reward}")

        assert self.nS == 3 * 4 and \
               self.seed == 1717 and \
               self.nepisodes == 200 and \
               self.start_state == 8
        assert np.allclose(observed, self.expected_mc, atol=1E-6)
        assert np.isclose(observed_average_reward, self.expected_mc_average_reward, atol=1E-6)

    def test_Env_PolRandomWalk_MetMC_FromCycles(self):
        print(f"\n*** Running test #{self.id()}")

        # TODO: (2023/07/17) Implement a simulation that keeps track of cycles and estimates the expected return as the average return over those cycles
        # Ref: See picture taken of Urtzi's whiteboards on 30-May-2023: ownCloud/Toulouse-IRIT-RL/docs/Meetings/FV-2023-05-30-ExtensionToReturnWithDiscount.jpg
        # The first thing to do would be to modify the sim_mc.run() method so that it keeps track of this piece of information.
        # Although, if we are later on going to use Fleming-Viot to estimate the expected return, perhaps we could define a new method that implements FV
        # and the above situation would be a special case of the FV-based estimation...??
        # (For now, the code below does exactly the same as regular MC estimation done in above method test_Env_PolRandomWalk_MetMC(), so the test should pass)
        state_value, state_counts, _, _, learning_info = \
            self.sim_mc.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                    compute_rmse=False, state_observe=None,
                    verbose=True, verbose_period=100,
                    plot=False, pause=0.1)
        observed = self.agent_rw_mc.getLearner().getV().getValues()
        observed_average_reward = self.agent_rw_mc.getLearner().getAverageReward()
        print("\nObserved state value function: " + test_utils.array2str(observed))
        print(f"\nAverage reward: {observed_average_reward}")

        assert self.nS == 3*4 and \
               self.seed == 1717 and \
               self.nepisodes == 200 and \
               self.start_state == 8
        assert np.allclose(observed, self.expected_mc, atol=1E-6)
        assert np.isclose(observed_average_reward, self.expected_mc_average_reward, atol=1E-6)

    def test_Env_PolRandomWalk_MetTDLambda(self):
        print(f"\n*** Running test #{self.id()}")

        state_value, state_counts, _, _, learning_info = \
            self.sim_td.run(nepisodes=self.nepisodes, start=self.start_state, seed=self.seed,
                         compute_rmse=False, state_observe=None,
                         verbose=True, verbose_period=100,
                         plot=False, pause=0.1)
        observed = self.agent_rw_td.getLearner().getV().getValues()
        observed_average_reward = self.agent_rw_td.getLearner().getAverageReward()
        print("\nObserved state value function: " + test_utils.array2str(observed))
        print(f"\nAverage reward: {observed_average_reward}")

        assert self.nS == 3 * 4 and \
               self.seed == 1717 and \
               self.nepisodes == 200 and \
               self.start_state == 8
        assert np.allclose(observed, self.expected_td, atol=1E-6)
        assert np.isclose(observed_average_reward, self.expected_td_average_reward, atol=1E-6)


class Test_EstValueFunctionV_MetMCLambda_EnvMountainCar(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        self.seed = kwargs.pop('seed', 1717)
        self.nepisodes = kwargs.pop('nepisodes', 30)  # 20000) #100000) #30000) #200) #2000)
        self.max_time_steps = kwargs.pop('max_time_steps', 500)  # Maximum number of steps to run per episode
        self.normalizer = kwargs.pop('normalizer', 1)  # Normalize for the plots: Set it to max_time_steps when the rewards are NOT sparse (i.e. are -1 every where except at terminal states), o.w. set it to 1 (when rewards are sparse, i.e. they occur at terminal states)
        self.start_state = kwargs.pop('start_state', None)  # (0.4, 0.07)) #None) #(0.4, 0.07))       # Position and velocity
        self.plot = kwargs.pop('plot', True)
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):  # cls is the class, in this case, class 'Test_TD_Lambda'
        # IMPORTANT: All attributes defined here can be then be referenced using self!
        # (i.e. they belong to the "object" instantiated by this class)

        # cls.env = gym.make('MountainCar-v0')   # Here we create in principle a MountainCarEnv environment because this environment is registered with gym.register() or similar.
        # See also this implementation of the Mountain Car: https://github.com/JJonahJson/MountainCar-v313/blob/master/code/main.py

        # Environment with discretized position and velocity with nx and nv points respectively
        # nx = 20  #20
        nv = 5  # 20
        cls.env = mountaincars.MountainCarDiscrete(nv)

        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env)

    def no_test_Env_PolRandomWalk_MetMC_TestOneCase(self, params=None, verbose_convergence=False):
        print(f"\n*** Running test #{self.id()}")

        # Learner and agent
        if params is None:
            params = dict({'alpha': 1.0,
                           'gamma': 1.0,
                           'lambda': 1.0,
                           'alpha_min': 0.0,
                           'adjust_alpha': True,
                           'alpha_update_type': AlphaUpdateType.EVERY_STATE_VISIT,
                           'learner_type': mc.LearnerType.MC,  # mc.LearnerType.LAMBDA_RETURN,
                           'reset_method': ResetMethod.ALLZEROS,
                           # ResetMethod.RANDOM_NORMAL, #ResetMethod.ALLZEROS,
                           'reset_params': dict({'loc': -500, 'scale': 10}),
                           })
        learner_mc = mc.LeaMCLambda(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                    alpha_update_type=params['alpha_update_type'],
                                    adjust_alpha=params['adjust_alpha'], adjust_alpha_by_episode=False,
                                    alpha_min=params['alpha_min'],
                                    lmbda=params['lambda'],
                                    learner_type=params['learner_type'],
                                    reset_method=params['reset_method'], reset_params=params['reset_params'],
                                    reset_seed=self.seed,
                                    debug=False)
        agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mc)

        # -- Simulation
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
            # print("ISD:", self.env.isd)
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
                                            verbose=True, verbose_period=max(1, int(self.nepisodes / 10)),
                                            verbose_convergence=verbose_convergence,
                                            plot=False, pause=0.001)
        time_end = timer()
        exec_time = time_end - time_start
        print("Execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

        observed = self.env.reshape_from_1d_to_2d(agent_rw_mc.getLearner().getV().getValues())
        state_counts = self.env.reshape_from_1d_to_2d(np.asarray(sim.agent.getLearner().getStateCounts()))
        print("\n{}, observed: ".format(self.id(), observed))
        # assert np.allclose(observed, expected, atol=1E-6)

        if self.plot:
            window_size = max(1, int(self.nepisodes / 100))
            deltaV_rel_max_smooth = computing.smooth(learning_info['deltaV_rel_max_signed'],
                                                     window_size=window_size)

            fig = plt.figure(figsize=(20, 10))
            ax, ax_n = fig.subplots(1, 2)

            legend_ax = []
            ax.plot(np.arange(self.nepisodes + 1), learning_info['V_abs_mean'] / self.normalizer, 'b--')
            ax.plot(np.arange(self.nepisodes + 1), learning_info['V_abs_mean_weighted'] / self.normalizer, 'b-')
            legend_ax += ["Average |V|", "Average |V| (weighted by state count)"]
            ax.set_ylim((0, ax.get_ylim()[1]))
            ax.set_xlabel("Episode number")
            ax.set_ylabel("Average |V|" + (self.normalizer != 1.0 and " normalized by dividing by {}" or "").format(
                self.normalizer))
            ax.legend(legend_ax, loc='upper left')

            ax2 = ax.twinx()
            legend_ax2 = []
            ax2.plot(np.arange(self.nepisodes + 1), deltaV_rel_max_smooth * 100, 'r-', linewidth=1)
            ax2.plot(np.arange(self.nepisodes + 1), learning_info['deltaV_rel_abs_mean'] * 100, 'r--',
                     linewidth=0.5)
            ax2.plot(np.arange(self.nepisodes + 1), learning_info['prop_states_deltaV_relevant'] * 100, 'k--',
                     linewidth=0.5)
            legend_ax2 += ["max|relative change| with sign ({}-tap-smoothed)".format(window_size),
                           "mean|relative change|",
                           "proportion states with |rel change| > 1%"]
            ax2.axhline(0.0, color='gray', linewidth=0.5)
            ax2.set_yscale('symlog')
            ax2.set_ylabel("|relative delta(V)| % (log scale)")
            ax2.set_ylim(-max(ax2.get_ylim()), +max(ax2.get_ylim()))
            ax2.legend(legend_ax2, loc='upper right')

            # Number of states used in the computation of the summary statistics at each episode
            ax_n.plot(np.arange(self.nepisodes + 1), learning_info['V_abs_n'] / self.env.getNumStates() * 100, 'g-')
            ax_n.set_xlabel("Episode number")
            ax_n.set_ylabel("% states")
            ax_n.set_ylim((0, 100))
            ax_n.legend(["% states"], loc='upper left')
            ax_n.set_title(
                "% states used for summary statistics (num states = {}) (average is computed on states with visit count > 0)".format(
                    self.env.getNumStates()))

            fig.suptitle(
                "Convergence of the estimated value function V (gamma={:.2f}, lambda={:.2f}, alpha={:.2f}, {}, max #steps = {})" \
                .format(params['gamma'], params['lambda'], params['alpha'], params['alpha_update_type'].name,
                        self.max_time_steps))

            fig = plt.figure(figsize=(20, 10))
            ax, ax2 = fig.subplots(1, 2)

            # ax.plot(self.env.reshape_from_2d_to_1d(state_counts))
            # for x in range(0, np.prod(self.env.shape), self.env.shape[0]):
            #    ax.axvline(x, color="gray")
            # ax.axhline(0, color="gray")
            # ax.set_xlabel("linear state index (grouped by {})".format(self.env.getShapeNames()[1]))
            # ax.set_ylabel("Visit count")
            # ax.set_title("State visit count (grouped by {})".format(self.env.getShapeNames()[1]))

            values = agent_rw_mc.getLearner().getV().getValues()
            ax.hist(values, bins=30, weights=np.repeat(1 / len(values), len(values)) * 100)
            ax.set_ylim((0, 100))
            ax.set_xlabel("Value function")
            ax.set_ylabel("Percent count")
            ax.set_title("Distribution of V(s) values")

            positions, velocities = self.env.get_positions(), self.env.get_velocities()
            ax2.plot(positions, np.sum(state_counts, axis=self.env.getVelocityDimension()), '.-',
                     color=self.env.getPositionColor())  # Sum along the different velocities
            ax2.plot(velocities, np.sum(state_counts, axis=self.env.getPositionDimension()), '.-',
                     color=self.env.getVelocityColor())  # Sum along the different positions
            ax2.legend(["Visit count on positions", "Visit count on velocities"])
            ax2.set_xlabel("Position / Velocity")
            ax2.set_ylabel("Visit count")
            ax2.set_title("Visit counts by dimension (position / velocity)")

        return observed, state_counts, params, sim, learning_info

    def test_Env_PolRandomWalk_MetMCLambdaReturn_TestMCvsLambdaReturn(self, verbose_convergence=False):
        print(f"\n*** Running test #{self.id()}")

        # Learner and agent
        params = dict({'alpha': 1.0,
                       'gamma': 1.0,
                       'lambda': 1.0,
                       'alpha_min': 0.0,
                       'adjust_alpha': True,
                       'alpha_update_type': AlphaUpdateType.EVERY_STATE_VISIT,
                       })

        learner_mc = mc.LeaMCLambda(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                    alpha_update_type=params['alpha_update_type'],
                                    adjust_alpha=params['adjust_alpha'], adjust_alpha_by_episode=False,
                                    alpha_min=params['alpha_min'],
                                    lmbda=params['lambda'],
                                    learner_type=mc.LearnerType.MC,
                                    reset_method=ResetMethod.ALLZEROS,
                                    debug=False)
        learner_lambda_return = mc.LeaMCLambda(self.env, alpha=params['alpha'], gamma=params['gamma'],
                                               alpha_update_type=params['alpha_update_type'],
                                               adjust_alpha=params['adjust_alpha'], adjust_alpha_by_episode=False,
                                               alpha_min=params['alpha_min'],
                                               lmbda=params['lambda'],
                                               learner_type=mc.LearnerType.LAMBDA_RETURN,
                                               reset_method=ResetMethod.ALLZEROS,
                                               debug=False)
        agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mc)
        agent_rw_lambda_return = agents.GenericAgent(self.policy_rw, learner_lambda_return)

        # -- Simulation
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
            # print("ISD:", self.env.isd)
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
                                               verbose=True, verbose_period=max(1, int(self.nepisodes / 10)),
                                               verbose_convergence=verbose_convergence,
                                               plot=False, pause=0.001)
        time_end = timer()
        exec_time = time_end - time_start
        print("[MC] Execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

        # Lambda-Return
        time_start = timer()
        _, _, _, _, learning_info = sim_lambda_return.run(nepisodes=self.nepisodes,
                                                          max_time_steps=self.max_time_steps,
                                                          start=idx_start_state, seed=self.seed,
                                                          compute_rmse=False, state_observe=None,
                                                          verbose=True,
                                                          verbose_period=max(1, int(self.nepisodes / 10)),
                                                          verbose_convergence=verbose_convergence,
                                                          plot=False, pause=0.001)
        time_end = timer()
        exec_time = time_end - time_start
        print("[Lambda-Return] Execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

        print("Observed estimataed value functions (MC / Lambda-Return):")
        print(np.c_[
                  agent_rw_mc.getLearner().getV().getValues(), agent_rw_lambda_return.getLearner().getV().getValues(),
                  agent_rw_lambda_return.getLearner().getV().getValues() - agent_rw_mc.getLearner().getV().getValues()])

        # Plots
        plt.figure()
        plt.hist(np.c_[
                     agent_rw_mc.getLearner().getV().getValues(), agent_rw_lambda_return.getLearner().getV().getValues()])
        plt.title("V(s) by each method")
        plt.legend(["MC", "Lambda-Return"])

        plt.figure()
        plt.hist(
            agent_rw_lambda_return.getLearner().getV().getValues() - agent_rw_mc.getLearner().getV().getValues())
        plt.title("Difference in V(s) between the two methods (Lambda-Return - MC)")

        assert np.allclose(agent_rw_mc.getLearner().getV().getValues(),
                           agent_rw_lambda_return.getLearner().getV().getValues())


if __name__ == '__main__':
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line

    test = True  # Use test=False when we want to recover the output of the test in the Python session and analyze it

    if test:
        runner = unittest.TextTestRunner()

        # unittest.getTestCaseNames()

        # Run all tests
        #unittest.main()

        # Create the test suites
        # --- Offline learning on Gridworld
        test_suite_offline = unittest.TestSuite()
        test_suite_offline.addTest(Test_EstStateValueV_MetOffline_EnvDeterministicNextState("test_EnvGridworld1D_PolRandomWalk_Met_TestOneCase"))
        test_suite_offline.addTest(Test_EstStateValueV_MetOffline_EnvDeterministicNextState("test_EnvGridworld1DOneTerminal_PolRandomWalk_Met_TestOneCase"))
        # DM-2022/06: For now we skip the test on the Mountain Car because the estimation takes too long to converge because
        # we are using a random policy and reaching the reward under this policy is very rare...
        # We will reactivate this test when we find the optimum policy and then we estimate the value function under the optimum policy.
        #test_suite_offline.addTest(Test_EstStateValueV_MetOffline_EnvDeterministicNextState("test_EnvMountainCarDiscreteActions_PolRandomWalk_Met_TestOneCase"))

        # --- Gridworld tests
        test_suite_gw1d = unittest.TestSuite()
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworlds("test_EnvGridworld1D_PolRandomWalk_MetMC_TestSeveralAlphasAndAlphaAdjustments"))
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworlds("test_EnvGridworld1D_PolRandomWalk_MetMC_TestGammaSmallerThan1"))
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworlds("test_EnvGridworld1D_PolRandomWalk_MetMC_TestRMSETwice"))
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworlds("test_EnvGridworld1D_PolRandomWalk_MetLambdaReturn_TestSeveralAlphasLambdasAlphaAdjustments"))
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworlds("test_EnvGridworld1D_PolRandomWalk_MetLambdaReturn_TestGammaLessThan1"))
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworlds("test_EnvGridworld1D_PolRandomWalk_MetTDLambda_TestSeveralAlphasLambdas"))

        test_suite_gw2dobstacles = unittest.TestSuite()
        test_suite_gw2dobstacles.addTest(Test_EstStateValueV_EnvGridworldsWithObstacles("test_Env_PolRandomWalk_MetMC"))
        test_suite_gw2dobstacles.addTest(Test_EstDifferentialStateValueV_EnvGridworldsWithObstacles("test_Env_PolRandomWalk_MetMC"))
        test_suite_gw2dobstacles.addTest(Test_EstStateValueV_EnvGridworldsWithObstacles("test_Env_PolRandomWalk_MetTDLambda"))
        test_suite_gw2dobstacles.addTest(Test_EstDifferentialStateValueV_EnvGridworldsWithObstacles("test_Env_PolRandomWalk_MetTDLambda"))

        # --- Mountain Car tests
        test_suite_mountain = unittest.TestSuite()
        test_suite_mountain.addTest(Test_EstValueFunctionV_MetMCLambda_EnvMountainCar("test_Env_PolRandomWalk_MetMCLambdaReturn_TestMCvsLambdaReturn"))

        # Run the test suites
        runner.run(test_suite_offline)
        runner.run(test_suite_gw1d)
        runner.run(test_suite_gw2dobstacles)
        runner.run(test_suite_mountain)
    else:
        # Use this when we want to recover the output of the test in the Python session and analyze it
        test_obj = Test_EstValueFunctionV_MetMCLambda_EnvMountainCar()
        test_obj.setUpClass()
        state_values, state_counts, params, sim, learning_info = test_obj.no_test_Env_PolRandomWalk_MetMC_TestOneCase()
