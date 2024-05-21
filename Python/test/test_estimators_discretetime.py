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
from matplotlib.ticker import MaxNLocator

from Python.lib.estimators import miscellanea as estimators_miscellanea

import Python.lib.agents as agents
from Python.lib.agents import GenericAgent
from Python.lib.agents.learners import LearningCriterion, LearningTask, ResetMethod
from Python.lib.agents.learners.episodic.discrete import fv, mc, td, AlphaUpdateType
from Python.lib.agents.policies import probabilistic, random_walks

from Python.lib.environments import gridworlds, mountaincars
from Python.lib.estimators.fv import estimate_expected_reward
from Python.lib.simulators.discrete import Simulator as DiscreteSimulator

from Python.lib.utils import computing

import test_utils


def plot_estimated_state_value_function(env, state_values, learning_criterion):
    # Reference value for the plots, needed for the AVERAGE reward learning criterion because there is no unique solution for V(s) in that case
    ref_V_true = ref_V = 0.0
    if learning_criterion == LearningCriterion.AVERAGE:
        if env.getV() is not None:
            ref_V_true = env.getV()[0]
        ref_V = state_values[0]
    ax_V = plt.figure().subplots(1, 1)
    if env.getV() is not None:
        ax_V.plot(env.getAllStates(), env.getV() - ref_V_true, 'b.-')
    ax_V.plot(env.getAllStates(), state_values - ref_V, 'r.-')
    ax_V.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_V.set_xlabel("State")
    ax_V.set_ylabel(learning_criterion == LearningCriterion.AVERAGE and "V(s) - V(0)" or "V(s)")
    ax_V.set_title(f"State value function under the {learning_criterion.name.upper()} reward criterion" + (learning_criterion == LearningCriterion.AVERAGE and ", referenced to V(s)" or ""))
    plt.pause(0.1)
    plt.draw()


class Test_EstStateValueV_MetOffline_EnvDeterministicNextState(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env_grid_classic = gridworlds.EnvGridworld1D_Classic(length=19+2)
        cls.env_grid_oneterminal = gridworlds.EnvGridworld1D(length=20, rewards_dict={19: +1}, reward_default=0.0)
        cls.env_mountain = mountaincars.MountainCarDiscrete(5)
            ## Note: we use more density to discretize the positions because we need to make sure that the car reaches the terminal state
            ## and this is not given if the discretization on the right side of the ramp is not dense enough because
            ## the new position is determined by x + v and vmax = 0.07, so the closest non-terminal x should 0.5 - 0.07 = 0.43
            ## where 0.5 = self.goal_position defined in mountain_cars.py in the MountainCarEnv class.

    def test_EnvGridworld1DClassic_PolRandomWalk_Met_TestOneCase(self):
        gamma = 1.0
        max_iter = 1000
        max_delta = 1E-6
        estimator = estimators_miscellanea.EstValueFunctionOfflineDeterministicNextState(self.env_grid_classic, gamma=gamma)
        niter, mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs = estimator.estimate_state_values_random_walk(synchronous=True, max_delta=1E-6, max_delta_rel=np.nan, max_iter=max_iter)

        nS = self.env_grid_classic.getNumStates()
        expected_values = np.arange(-nS + 1, nS, 2) / (nS - 1); expected_values[0] = expected_values[-1] = 0.0
        observed_values = estimator.getV().getValues()
        print("\n***** Estimated state values on {}-state '1D gridworld environment' after {} iterations out of {}, with:\n" \
              .format(self.env_grid_classic.getNumStates(), niter, max_iter) + \
                "mean|delta(V)| = {}, max|delta(V)| = {}, max|delta_rel(V)| = {}:\n{}" \
              .format(mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs, estimator.getV().getValues()))
        print("Average V = {}, Average |V| = {}".format(np.mean(observed_values), np.mean(np.abs(observed_values))))

        plt.figure()
        plt.plot(observed_values, 'r.-')
        plt.title("1D gridworld: Estimated value function V(s) using offline deterministic estimation")

        assert  self.env_grid_classic.getNumStates() == 21 and \
                self.env_grid_classic.getTerminalStates() == [0, 20] and \
                list(self.env_grid_classic.getRewards()) == [-1.0, 1.0] and \
                gamma == 1.0 and \
                max_delta == 1E-6
        assert niter == 220
        assert np.isclose(max_deltaV_abs, 9.9E-7)
        assert np.allclose(observed_values, expected_values, atol=1E-5)

    def test_EnvGridworld1DOneTerminal_PolRandomWalk_Met_TestOneCase(self):
        gamma = 0.9         # Discount factor to use in the return calculation
        max_iter = 1000     # Maximum number of iterations to run until (hopefully) convergence is reached
        max_delta = 1E-6    # Maximum delta over all state values to consider that convergence to the state value function was achieved
        estimator = estimators_miscellanea.EstValueFunctionOfflineDeterministicNextState(self.env_grid_oneterminal, gamma=gamma)
        niter, mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs = estimator.estimate_state_values_random_walk(synchronous=True, max_delta=max_delta, max_delta_rel=np.nan, max_iter=max_iter)

        expected_values = [ 2.49339553e-04, 3.05488650e-04, 4.30033580e-04, 6.51154580e-04,
                            1.01730726e-03, 1.61083071e-03, 2.56251912e-03, 4.08522404e-03,
                            6.51587374e-03, 1.03962522e-02, 1.65869731e-02, 2.64655036e-02,
                            4.22252902e-02, 6.73701766e-02, 1.07486229e-01, 1.71489509e-01,
                            2.73601575e-01, 4.36514912e-01, 6.96431566e-01, 0.00000000e+00]
        observed_values = estimator.getV().getValues()
        print("\n***** Estimated state values on {}-state '1D gridworld environment with left transient state and left terminal state' after {} iterations out of {}, with:\n" \
              .format(self.env_grid_classic.getNumStates(), niter, max_iter) + \
                "mean|delta(V)| = {}, max|delta(V)| = {}, max|delta_rel(V)| = {}:\n{}" \
              .format(mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs, estimator.getV().getValues()))
        print("Average V = {}, Average |V| = {}".format(np.mean(observed_values), np.mean(np.abs(observed_values))))

        plt.figure()
        plt.plot(observed_values, 'r.-')
        plt.title("1D-gridworld with one terminal state with +1 reward: OFFLINE estimated value function V(s)")

        assert  self.env_grid_oneterminal.getNumStates() == 20 and \
                self.env_grid_oneterminal.getTerminalStates() == [19] and \
                list(self.env_grid_oneterminal.getRewards()) == [1.0] and \
                gamma == 0.9 and \
                max_delta == 1E-6
        assert niter == 84
        assert np.isclose(max_deltaV_abs, 9.126598E-07)
        assert np.allclose(observed_values, expected_values, atol=1E-6)

    def test_EnvMountainCarDiscreteActions_PolRandomWalk_Met_TestOneCase(self):
        max_iter = 100
        estimator = estimators_miscellanea.EstValueFunctionOfflineDeterministicNextState(self.env_mountain, gamma=1.0)
        niter, mean_deltaV_abs, max_deltaV_abs, max_deltaV_rel_abs = \
            estimator.estimate_state_values_random_walk(synchronous=True, max_delta=np.nan, max_delta_rel=1E-5, max_iter=max_iter,
                                                        reset_method=ResetMethod.ALLZEROS, reset_params=dict({'min': 0.2, 'max': 0.8}), reset_seed=1713)

        state_values = estimator.getV().getValues()
        state_values_2d = self.env_mountain.reshape_from_1d_to_2d(state_values)
        non_terminal_states = np.array(self.env_mountain.getNonTerminalStates())
        min_state_value_non_terminal, max_state_value_non_terminal = np.min(state_values[non_terminal_states]), np.max(state_values[non_terminal_states])

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


class Test_EstStateValueV_EnvGridworld1D(unittest.TestCase, test_utils.EpisodeSimulation):
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
        cls.nS = 21
        cls.start_state = 10
        cls.isd = np.array([1 if s == cls.start_state else 0 for s in range(cls.nS)])
        cls.env1d = gridworlds.EnvGridworld1D_Classic(length=cls.nS, initial_state_distribution=cls.isd)

        # True state value function when gamma = 1.0
        cls.V_true = np.arange(-cls.nS + 1, cls.nS, 2) / (cls.nS - 1)
        cls.V_true[0] = cls.V_true[-1] = 0

        # Random walk policy on the above environment
        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env1d)

        # Plotting parameters
        cls.colormap = cm.get_cmap("jet")

    # --------------------- TESTS OF MC: traditional MC (no lambda involved) ----------------------#
    # -------- DATA -------
    # Case number, description, expected value, parameters
    # These are the same tests 1, 2 and 3 from data_test_lambda_return_random_walk
    data_test_EnvGridworld1DClassic_PolRandomWalk_MetMC_TestSeveralAlphasAndAlphaAdjustments = lambda DEFAULT_EXECUTION = False: (
        (1, DEFAULT_EXECUTION, 'MC, no alpha adjustment',
         Test_EstStateValueV_EnvGridworld1D.EXPECTED_TEST_RANDOM_WALK_1,
         (0.1, 1.0), False, False, 0.0),
        (2, DEFAULT_EXECUTION, 'MC, adjusted alpha by state count (good results if run for 200 episodes)',
         Test_EstStateValueV_EnvGridworld1D.EXPECTED_TEST_RANDOM_WALK_2,
         (1.0, 1.0), True, False, 0.0),
        (3, DEFAULT_EXECUTION, 'MC, adjusted alpha by episode',
         Test_EstStateValueV_EnvGridworld1D.EXPECTED_TEST_RANDOM_WALK_3,
         (1.0, 1.0), True, True, 0.0),
    )

    # -------- DATA -------

    @data_provider(data_test_EnvGridworld1DClassic_PolRandomWalk_MetMC_TestSeveralAlphasAndAlphaAdjustments)
    def test_EnvGridworld1DClassic_PolRandomWalk_MetMC_TestSeveralAlphasAndAlphaAdjustments(self, casenum, run, desc, expected,
                                                                                     params_alpha_gamma,
                                                                                     adjust_alpha, adjust_alpha_by_episode,
                                                                                     alpha_min):
        print("\n*** Running test {0} ***".format(self.id()))

        if run:
            print("\n*** Testing {0}, case number {1}: '{2}' ***".format(self.id(), casenum, desc))

            # Simulation setup
            seed = 1717
            nepisodes = 20

            learner_mclambda = mc.LeaMCLambda(self.env1d, alpha=params_alpha_gamma[0],
                                              gamma=params_alpha_gamma[1],
                                              adjust_alpha=adjust_alpha,
                                              adjust_alpha_by_episode=adjust_alpha_by_episode,
                                              alpha_min=alpha_min,
                                              learner_type=mc.LearnerType.MC,
                                              debug=False)
            agent_rw_mc = agents.GenericAgent(self.policy_rw, learner_mclambda)
            sim = DiscreteSimulator(self.env1d, agent_rw_mc, debug=False)
            _, _, _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
                sim.run(nepisodes=nepisodes, seed=seed,
                        compute_rmse=True, state_observe=10,
                        verbose=True, verbose_period=nepisodes // 20,
                        plot=True, pause=0.1)
            observed = agent_rw_mc.getLearner().getV().getValues()
            print("\nobserved: " + test_utils.array2str(observed))

            assert self.nS == 21 and \
                   seed == 1717 and \
                   nepisodes == 20 and \
                   self.start_state == 10
            assert np.allclose(observed, expected, atol=1E-6)

    def test_EnvGridworld1DClassic_PolRandomWalk_MetMC_TestGammaSmallerThan1(self):
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717
        nepisodes = 20
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
        _, _, _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
            sim.run(nepisodes=nepisodes, seed=seed,
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
        assert self.nS == 21 and \
               seed == 1717 and \
               nepisodes == 20 and \
               self.start_state == 10 and \
               params['alpha'] == 1.0 and \
               params['gamma'] == 0.7 and \
               params['alpha_min'] == 0.0
        assert np.allclose(observed, expected, atol=1E-6)

    def test_EnvGridworld1DClassic_PolRandomWalk_MetMC_TestRMSETwice(self):
        # -- All tests are run using seed = 1717, nepisodes = 20, start_state = 10
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717
        nepisodes = 20
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
        _, _, _, _, RMSE_by_episode, MAPE_by_episode, _ = sim.run(nepisodes=nepisodes,
                                                            seed=seed,
                                                            compute_rmse=True, state_observe=19,
                                                            verbose=True, verbose_period=100,
                                                            plot=False, pause=0.001)

        # Expected RMSE with the following settings:
        # alpha_update_Type = AlphaUpdateType.FIRST_STATE_VISIT, adjust_alpha = True, adjust_alpha_by_episode = False
        # + the settings specified in the assertion
        assert self.nS == 21 and \
               seed == 1717 and \
               nepisodes == 20 and \
               self.start_state == 10 and \
               params['alpha'] == 0.3 and \
               params['gamma'] == 1.0 and \
               params['alpha_min'] == 0.0
        rmse_expected = 0.29397963

        rmse_observed = np.mean(RMSE_by_episode)
        print("First run: average RMSE over {} episodes: {:.8f}".format(nepisodes, rmse_observed))
        assert np.allclose(rmse_observed, rmse_expected, atol=1E-6)

        # Second run
        _, _, _, _, RMSE_by_episode, MAPE_by_episode, _ = sim.run(nepisodes=nepisodes,
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
    data_test_EnvGridworld1DClassic_PolRandomWalk_MetLambdaReturn_TestSeveralAlphasLambdasAlphaAdjustments = lambda DEFAULT_EXECUTION = True: (
        (1, DEFAULT_EXECUTION, 'MC, no alpha adjustment',
         Test_EstStateValueV_EnvGridworld1D.EXPECTED_TEST_RANDOM_WALK_1,
         (0.1, 1.0, 1.0), False, False, 0.0),
        (2, DEFAULT_EXECUTION, 'MC, adjusted alpha by state count (good results if run for 200 episodes)',
         Test_EstStateValueV_EnvGridworld1D.EXPECTED_TEST_RANDOM_WALK_2,
         (1.0, 1.0, 1.0), True, False, 0.0),
        (3, DEFAULT_EXECUTION, 'MC, adjusted alpha by episode',
         Test_EstStateValueV_EnvGridworld1D.EXPECTED_TEST_RANDOM_WALK_3,
         (1.0, 1.0, 1.0), True, True, 0.0),
        (4, DEFAULT_EXECUTION, 'L-return (lambda<1), no alpha adjustment',
         [-0.000000, -0.648738, -0.440867, -0.228680, -0.172241, -0.093122, -0.023765,
          -0.016186, -0.012195, -0.007552, 0.002211, 0.006526, 0.009155, 0.015080,
          0.035585, 0.049038, 0.071389, 0.111463, 0.213302, 0.552812, 0.000000],
         (0.2, 1.0, 0.7), False, False, 0.0),
        (5, DEFAULT_EXECUTION, 'L-return (lambda<1), adjusted alpha by state count',
         [-0.000000, -0.622132, -0.413372, -0.199682, -0.128330, -0.055435, -0.017840,
          -0.010864, -0.006091, -0.003125, 0.000993, 0.003451, 0.005608, 0.012755,
          0.028038, 0.039561, 0.058164, 0.104930, 0.264852, 0.694180, 0.000000],
         (1.0, 1.0, 0.7), True, False, 0.0),
    )

    # -------- DATA -------

    @data_provider(data_test_EnvGridworld1DClassic_PolRandomWalk_MetLambdaReturn_TestSeveralAlphasLambdasAlphaAdjustments)
    def test_EnvGridworld1DClassic_PolRandomWalk_MetLambdaReturn_TestSeveralAlphasLambdasAlphaAdjustments(self, casenum, run, desc,
                                                                                                   expected,
                                                                                                   params_alpha_gamma_lambda,
                                                                                                   adjust_alpha,
                                                                                                   adjust_alpha_by_episode,
                                                                                                   alpha_min):
        print("\n*** Running test {0} ***".format(self.id()))

        if run:
            print("\n*** Testing {0}, case number {1}: '{2}' ***".format(self.id(), casenum, desc))

            # Simulation setup
            seed = 1717
            nepisodes = 20
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
            _, _, _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
                sim.run(nepisodes=nepisodes, seed=seed,
                        compute_rmse=True, state_observe=10,
                        verbose=True, verbose_period=100,
                        plot=False, pause=0.1)
            observed = agent_rw_mc.getLearner().getV().getValues()
            print("\nobserved: " + test_utils.array2str(observed))

            assert self.nS == 21 and \
                   seed == 1717 and \
                   nepisodes == 20 and \
                   self.start_state == 10
            assert np.allclose(observed, expected, atol=1E-6)

    def test_EnvGridworld1DClassic_PolRandomWalk_MetLambdaReturn_TestGammaLessThan1(self):
        # -- All tests are run using seed = 1717, nepisodes = 20, start_state = 10
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717
        nepisodes = 20
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
        _, _, _, _, RMSE_by_episode, MAPE_by_episode, learning_info = \
            sim.run(nepisodes=nepisodes, seed=seed,
                    compute_rmse=True, state_observe=10,
                    verbose=True, verbose_period=100,
                    plot=False, pause=0.1)

        assert self.nS == 21 and \
               seed == 1717 and \
               nepisodes == 20 and \
               self.start_state == 10 and \
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

    def fails_test_EnvGridworld1DClassic_PolRandomWalk_MetLambdaReturnAdaptive_TestOneCase(self):
        # NOTE: (2021/10/16) This test currently fails for two reasons:
        # - the `observed` array is an array of all -1.0 followed by all +1.0 (i.e. a step function)
        # - the plot_results() call at the end fails because the alpha_mean_by_episode_mean variable used when calling
        # plt.plot() inside plot_results() has zero length.
        # So, it seems that nothing is really done or learned by the learner_mclambda_adaptive object.
        print("\n*** Running test " + self.id())

        # Simulation setup
        seed = 1717
        nepisodes = 20
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
        _, _, _, _, RMSE_by_episode, MAPE_by_episode, learning_info = sim.run(nepisodes=nepisodes,
                                                                        seed=seed,
                                                                        compute_rmse=True, state_observe=10,
                                                                        verbose=True, verbose_period=100,
                                                                        plot=False, pause=0.1)

        # Expected state values with alpha = 0.2, gamma = 0.9, lambda = 0.8
        assert self.nS == 21 and \
               seed == 1717 and \
               nepisodes == 20 and \
               self.start_state == 10 and \
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

    def test_EnvGridworld1DClassic_PolRandomWalk_MetTDLambda_TestSeveralAlphasLambdas(self):
        """
        This test intends to reproduce the results in Sutton 2018 (pag. 295) on the TD(lambda) algorithm applied to learn
        the state value function in a 1D gridworld. The results shown in Sutton correspond to the episode always starting
        at the same state in the 1D gridworld, namely at the middle state (in our case at state 10).
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
        assert self.nS == 21 and \
               seed == 1717 and \
               nexperiments == 3 and \
               nepisodes == 10 and \
               useGrid == False and \
               gamma == 1.0 and \
               lambdas == [0, 0.4, 0.7, 0.8] and \
               alphas == [0.2, 0.4, 0.8]
        assert np.allclose(rmse_episodes_mean, [[0.45538468, 0.40588784, 0.35134918],
                                                [0.41605814, 0.33990824, 0.30015009],
                                                [0.34587065, 0.28262690, 0.37674099],
                                                [0.30984734, 0.30718626, 0.46807331]])
        assert np.allclose(rmse_episodes_se,   [[0.01260557, 0.02211615, 0.03213792],
                                                [0.02009345, 0.03399665, 0.03352152],
                                                [0.03244298, 0.03680280, 0.02015556],
                                                [0.03680285, 0.02787032, 0.01874832]])


class Test_EstDifferentialStateValueV_EnvGridworld1D(unittest.TestCase, test_utils.EpisodeSimulation):

    @classmethod
    def setUpClass(cls):
        # Plot settings
        cls.max_rmse = 0.8
        cls.color_rmse = "blue"

        # The environment
        cls.nS = 20
        cls.env1d = gridworlds.EnvGridworld1D(length=cls.nS, rewards_dict={cls.nS-1: +1}, reward_default=0.0)
            ## NOTE: By definition of this gridworld, the start state is fixed at the leftmost state 0
        # I let the start state of the one-terminal gridworld be recorded in the following attribute which is used in the assertions below
        cls.start_state = 0

        # The learning criterion and learning task definition
        cls.learning_criterion = LearningCriterion.AVERAGE
        cls.learning_task = LearningTask.CONTINUING
        cls.gamma = 1.0

        # Compute the TRUE differential state value function, for the optimal policy and for the random policy
        # Optimal
        policy = probabilistic.PolGenericDiscrete(cls.env1d, dict(), policy_default=[0.0, 1.0])
        _, P, _, b, g, mu = computing.compute_transition_matrices(cls.env1d, policy)
        cls.V_true_optimal = computing.compute_state_value_function_from_transition_matrix(P, b, bias=g, gamma=cls.gamma)
        # Random
        policy = probabilistic.PolGenericDiscrete(cls.env1d, dict(), policy_default=[0.5, 0.5])
        _, P, _, b, g, mu = computing.compute_transition_matrices(cls.env1d, policy)
        cls.V_true_random = computing.compute_state_value_function_from_transition_matrix(P, b, bias=g, gamma=cls.gamma)

        # Deterministic and random policies that are used in the agents interacting with the environment in tests
        cls.policy_optimal = probabilistic.PolGenericDiscrete(cls.env1d, policy=dict(), policy_default=[0.0, 1.0])
        cls.policy_random = probabilistic.PolGenericDiscrete(cls.env1d, policy=dict(), policy_default=[0.5, 0.5])

        # Plotting parameters
        cls.colormap = cm.get_cmap("jet")

    def test_EnvGridworld1DOneTerminal_PolOptimal_MetTDLambdaGt0(self):
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717
        max_time_steps = 1397   # If possible, use the same number of events as those observed in the FV test below (n_events_et + n_events_fv), but this may not be the case...

        # Learner
        params = dict({'alpha': 1.0,
                       'gamma': self.gamma,
                       'lambda': 0.7,
                       'alpha_min': 0.1,        # We use alpha_min > 0 because o.w. V(s) is not fully learned, due to the CONTINUING task context, where we receive reards from "infinite" time
                       })
        learner_td = td.LeaTDLambda(self.env1d,
                                    criterion=self.learning_criterion,
                                    task=self.learning_task,
                                    alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                                    alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                                    adjust_alpha=True, adjust_alpha_by_episode=False,
                                    alpha_min=params['alpha_min'],
                                    debug=False)

        # Define the agents for the policies that are used in the tests
        agent_td_optimal = agents.GenericAgent(self.policy_optimal, learner_td)

        # Store the true state value function in the environment so that we can compare our estimates with those values when running the simulation in the test
        self.env1d.setV(self.V_true_optimal)

        sim = DiscreteSimulator(self.env1d, agent_td_optimal, debug=False)
        state_values, action_values, advantage_values, state_counts, _, _, learning_info = \
            sim.run(max_time_steps=max_time_steps,
                    seed=seed,
                    compute_rmse=False, state_observe=self.nS-1,   # This is the state just before the terminal state
                    verbose=True, verbose_period=max_time_steps // 20,
                    plot=False, pause=0.1)

        # The expected state values are EXTREMELY close to self.V_true_optimal, so great!
        expected_V = [-0.302241, -0.250084, -0.198016, -0.146217, -0.094839,
                      -0.043998,  0.006234,  0.055830,  0.104812,  0.153248,
                       0.201249,  0.248966,  0.296573,  0.344261,  0.392220,
                       0.440621,  0.489604,  0.539254,  0.589648, -0.354320]
        expected_state_counts = [70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 69, 69]
        expected_time_steps = 1397
        print("\nObserved V(s): " + test_utils.array2str(state_values))
        print("State count: " + test_utils.array2str(state_counts))
        print(f"Number of time steps in simulation: {learning_info['nsteps']}")

        plot_estimated_state_value_function(self.env1d, state_values, self.learning_criterion)

        assert self.nS == 20 and \
               seed == 1717 and \
               self.start_state == 0 and \
               params['alpha'] == 1.0 and \
               params['gamma'] == 1.0 and \
               params['lambda'] == 0.7 and \
               params['alpha_min'] == 0.1
        assert learning_info['nsteps'] == expected_time_steps
        assert all(state_counts == expected_state_counts)
        assert np.allclose(state_values, expected_V, atol=1E-6)

    def test_EnvGridworld1DOneTerminal_PolRandomWalk_MetTDLambdaGt0(self):
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717
        max_time_steps = 1397  # If possible, use the same number of events as those observed in the FV test below (n_events_et + n_events_fv), but this may not be the case...

        # Learner
        params = dict({'alpha': 1.0,
                       'gamma': self.gamma,
                       'lambda': 0.7,
                       'alpha_min': 0.1,
                       # We use alpha_min > 0 because o.w. V(s) is not fully learned, due to the CONTINUING task context, where we receive reards from "infinite" time
                       })
        learner_td = td.LeaTDLambda(self.env1d,
                                    criterion=self.learning_criterion,
                                    task=self.learning_task,
                                    alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                                    alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                                    adjust_alpha=True, adjust_alpha_by_episode=False,
                                    alpha_min=params['alpha_min'],
                                    debug=False)

        # Define the agents for the policies that are used in the tests
        agent_td_random = agents.GenericAgent(self.policy_random, learner_td)

        # Store the true state value function in the environment so that we can compare our estimates with those values when running the simulation in the test
        self.env1d.setV(self.V_true_random)

        sim = DiscreteSimulator(self.env1d, agent_td_random, debug=False)
        state_values, action_values, advantage_values, state_counts, _, _, learning_info = \
            sim.run(max_time_steps=max_time_steps,
                    seed=seed,
                    compute_rmse=False, state_observe=self.nS-1,   # This is the state just before the terminal state
                    verbose=True, verbose_period=max_time_steps // 20,
                    plot=False, pause=0.1)

        # The expected state values are EXTREMELY close to self.V_true_optimal, so great!
        expected_V = [-0.065410, -0.064140, -0.067197, -0.066002, -0.058670,
                      -0.054133, -0.039811, -0.016067, -0.001620,  0.015175,
                       0.045576,  0.085641,  0.147396,  0.239321,  0.270213,
                       0.326755,  0.493805,  0.713191,  0.908722, -0.036216]
        expected_state_counts = [177, 141, 136, 152, 137, 117, 96, 74, 66, 64, 50, 29, 20, 29, 37, 32, 21, 10, 6, 4]
        expected_time_steps = 1397
        print("\nObserved V(s): " + test_utils.array2str(state_values))
        print("State count: " + test_utils.array2str(state_counts))
        print(f"Number of time steps in simulation: {learning_info['nsteps']}")

        plot_estimated_state_value_function(self.env1d, state_values, self.learning_criterion)

        assert self.nS == 20 and \
               seed == 1717 and \
               self.start_state == 0 and \
               params['alpha'] == 1.0 and \
               params['gamma'] == 1.0 and \
               params['lambda'] == 0.7 and \
               params['alpha_min'] == 0.1
        assert learning_info['nsteps'] == expected_time_steps
        assert all(state_counts == expected_state_counts)
        assert np.allclose(state_values, expected_V, atol=1E-6)

    def test_EnvGridworld1DOneTerminal_PolOptimal_MetTDLambda0(self):
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717
        max_time_steps = 1397   # If possible, use the same number of events as those observed in the FV test below (n_events_et + n_events_fv), but this may not be the case...

        # Learner and agent definition
        params = dict({'alpha': 1.0,
                       'gamma': self.gamma,
                       'lambda': 0.0,
                       'alpha_min': 0.1,        # We use alpha_min > 0 because o.w. V(s) is even LESS learned than it is with lambda=0 (as opposed to using e.g. lambda=0.7), due to the CONTINUING task context, where we receive reards from "infinite" time
                       })
        learner_td = td.LeaTDLambda(self.env1d,
                                    criterion=self.learning_criterion,
                                    task=self.learning_task,
                                    alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                                    alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                                    adjust_alpha=True, adjust_alpha_by_episode=False,
                                    alpha_min=params['alpha_min'],
                                    debug=False)

        # Define the agents for the policies that are used in the tests
        agent_td_optimal = agents.GenericAgent(self.policy_optimal, learner_td)

        # Store the true state value function in the environment so that we can compare our estimates with those values when running the simulation in the test
        self.env1d.setV(self.V_true_optimal)

        sim = DiscreteSimulator(self.env1d, agent_td_optimal, debug=False)
        state_values, action_values, advantage_values, state_counts, _, _, learning_info = \
            sim.run(max_time_steps=max_time_steps,
                    seed=seed,
                    compute_rmse=False, state_observe=self.nS-1,    # This is the state just before the terminal state
                    verbose=True, verbose_period=max_time_steps // 20,
                    plot=False, pause=0.1)

        # The expected state values oscillate around self.V_true_optimal, and this is due to the fact that lambda = 0, as opposed to some positive value
        # The true state value function is learned very well with e.g. lambda = 0.7, as used in test test_EnvGridworld1DOneTerminal_PolOptimal_MetTDLambdaGt0.
        expected_V = [-0.395943, -0.395002, -0.392603, -0.386966, -0.374817,
                      -0.350897, -0.308078, -0.238799, -0.138176, -0.008029,
                       0.140430,  0.288038,  0.414136,  0.505060,  0.559203,
                       0.585151,  0.594903,  0.602679,  0.603437, -0.396303]
        expected_state_counts = [70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 69, 69]
        expected_time_steps = 1397
        print("\nObserved V(s): " + test_utils.array2str(state_values))
        print("State count: " + test_utils.array2str(state_counts))
        print(f"Number of time steps in simulation: {learning_info['nsteps']}")

        plot_estimated_state_value_function(self.env1d, state_values, self.learning_criterion)

        assert self.nS == 20 and \
               seed == 1717 and \
               self.start_state == 0 and \
               params['alpha'] == 1.0 and \
               params['gamma'] == 1.0 and \
               params['lambda'] == 0.0 and \
               params['alpha_min'] == 0.1
        assert learning_info['nsteps'] == expected_time_steps
        assert all(state_counts == expected_state_counts)
        assert np.allclose(state_values, expected_V, atol=1E-6)

    def test_EnvGridworld1DOneTerminal_PolRandomWalk_MetTDLambda0(self):
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717
        max_time_steps = 1397  # If possible, use the same number of events as those observed in the FV test below (n_events_et + n_events_fv), but this may not be the case...

        # Learner and agent definition
        params = dict({'alpha': 1.0,
                       'gamma': self.gamma,
                       'lambda': 0.0,
                       'alpha_min': 0.1,
                       # We use alpha_min > 0 because o.w. V(s) is even LESS learned than it is with lambda=0 (as opposed to using e.g. lambda=0.7), due to the CONTINUING task context, where we receive reards from "infinite" time
                       })
        learner_td = td.LeaTDLambda(self.env1d,
                                    criterion=self.learning_criterion,
                                    task=self.learning_task,
                                    alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                                    alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                                    adjust_alpha=True, adjust_alpha_by_episode=False,
                                    alpha_min=params['alpha_min'],
                                    debug=False)

        # Define the agents for the policies that are used in the tests
        agent_td_random = agents.GenericAgent(self.policy_random, learner_td)

        # Store the true state value function in the environment so that we can compare our estimates with those values when running the simulation in the test
        self.env1d.setV(self.V_true_random)

        sim = DiscreteSimulator(self.env1d, agent_td_random, debug=False)
        state_values, action_values, advantage_values, state_counts, _, _, learning_info = \
            sim.run(max_time_steps=max_time_steps,
                    seed=seed,
                    compute_rmse=False, state_observe=self.nS - 1,  # This is the state just before the terminal state
                    verbose=True, verbose_period=max_time_steps // 20,
                    plot=False, pause=0.1)

        # The expected state values oscillate around self.V_true_optimal, and this is due to the fact that lambda = 0, as opposed to some positive value
        # The true state value function is learned very well with e.g. lambda = 0.7, as used in test test_EnvGridworld1DOneTerminal_PolOptimal_MetTDLambdaGt0.
        expected_V = [-0.022670, -0.021785, -0.024228, -0.025241, -0.024209,
                      -0.021873, -0.019348, -0.017756, -0.017073, -0.015500,
                      -0.013044, -0.011186, -0.009617, -0.010542, -0.005119,
                       0.030250,  0.186153,  0.467958,  0.851619, -0.012612]
        expected_state_counts = [177, 141, 136, 152, 137, 117, 96, 74, 66, 64, 50, 29, 20, 29, 37, 32, 21, 10, 6, 4]
        expected_time_steps = 1397
        print("\nObserved V(s): " + test_utils.array2str(state_values))
        print("State count: " + test_utils.array2str(state_counts))
        print(f"Number of time steps in simulation: {learning_info['nsteps']}")

        plot_estimated_state_value_function(self.env1d, state_values, self.learning_criterion)

        assert self.nS == 20 and \
               seed == 1717 and \
               self.start_state == 0 and \
               params['alpha'] == 1.0 and \
               params['gamma'] == 1.0 and \
               params['lambda'] == 0.0 and \
               params['alpha_min'] == 0.1
        assert learning_info['nsteps'] == expected_time_steps
        assert all(state_counts == expected_state_counts)
        assert np.allclose(state_values, expected_V, atol=1E-6)

    def test_EnvGridworld1DOneTerminal_PolOptimal_MetFV(self):
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717

        # Learner and agent definition
        N = 50
        T = 1000
        max_time_steps_fv = 100*N
        min_prop_absorbed_particles = 0.90  # Minimum proportion of particles that should be absorbed after overcoming the maximum number of time steps defined in max_time_steps_fv
        params = dict({'N': N,
                       'T': T,
                       'absorption_set': set(np.arange(5)),
                       'activation_set': set({5}),
                       'alpha': 1.0,
                       'gamma': self.gamma,
                       'lambda': 0.0,
                       'alpha_min': 0.1,        # We use alpha_min > 0 because o.w. V(s) is not fully learned (due to the CONTINUING task context, where we receive reards from "infinite" time
                       })
        learner_fv = fv.LeaFV(  self.env1d, params['N'], params['T'], params['absorption_set'], params['activation_set'],
                                probas_stationary_start_state_et=None,
                                probas_stationary_start_state_fv=None,
                                criterion=self.learning_criterion,
                                alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                                alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                                adjust_alpha=True, adjust_alpha_by_episode=False,
                                alpha_min=params['alpha_min'],
                                debug=False)

        # Define the agents for the policies that are used in the tests
        agent_fv_optimal = agents.GenericAgent(self.policy_optimal, learner_fv)

        # Store the true state value function in the environment so that we can compare our estimates with those values when running the simulation in the test
        self.env1d.setV(self.V_true_optimal)

        sim = DiscreteSimulator(self.env1d, agent_fv_optimal, debug=False)
        state_values, action_values, advantage_values, state_counts, probas_stationary, average_reward, average_absorption_time, n_cycles_absorption_used, n_events_et, n_events_fv  = \
            sim.run(max_time_steps=max_time_steps_fv,
                    max_time_steps_for_absorbed_particles_check=max_time_steps_fv,
                    min_prop_absorbed_particles=min_prop_absorbed_particles,
                    use_average_reward_stored_in_learner=False,
                        ## Set the above parameter to True in case we want to test what happens when we start with an initially estimated average reward,
                        ## e.g. to check if that changes the value function estimation results considerably. To this end, we could call learner_fv.setAverageReward(<value>)
                        ## when instantiating the FV learner above.
                    seed=seed,
                    verbose=True, verbose_period=max_time_steps_fv // 20,
                    plot=False)

        # The expected state values are close to self.V_true_optimal but not quite there
        expected_V = [-0.335716, -0.327215, -0.321336, -0.318088, -0.316093,
                      -0.395134, -0.292521, -0.144666,  0.034477,  0.203168,
                       0.304436,  0.341835,  0.348890,  0.358013,  0.393345,
                       0.432999,  0.477292,  0.523828,  0.570643, -0.382537]
        expected_state_counts = [50, 50, 50, 50, 50, 101, 100, 100, 100, 101, 101, 109, 117, 141, 150, 161, 189, 239, 300, 351]
        expected_n_events_et = params['T']
        expected_n_events_fv = 1609
        expected_n_cycles_absorption = 49
        expected_absorption_time = 20.0
        expected_average_reward = 0.0467760
        expected_probas_stationary = dict({19: 0.0467760})
        # The following is NOT an expected value for the unit test, but just the expected reward under stationarity computed using its definition,
        # i.e. E(R) = sum{x} p(x)*r(x), where p(x) is the stationary probability
        expected_reward_under_stationarity = estimate_expected_reward(self.env1d, probas_stationary)
        print("\nObserved V(s): " + test_utils.array2str(state_values))
        print("State count: " + test_utils.array2str(state_counts))
        print(f"Number of cycles observed in E(T) estimation: {n_cycles_absorption_used}")
        print(f"Estimated expected reabsorption time E(T): {average_absorption_time}")
        print(f"Estimated expected reward: {average_reward}")
        print(f"Number of time steps in E(T) estimation: {n_events_et}")
        print(f"Number of time steps in FV simulation: {n_events_fv}")
        print(f"Total number of time steps in E(T) + FV simulation: {n_events_et + n_events_fv}")

        plot_estimated_state_value_function(self.env1d, state_values, self.learning_criterion)

        assert  self.nS == 20 and \
                seed == 1717 and \
                max_time_steps_fv == 100*params['N'] and \
                min_prop_absorbed_particles == 0.90 and \
                params['N'] == 50 and \
                params['T'] == 1000 and \
                params['absorption_set'] == set(np.arange(5)) and \
                params['activation_set'] == set({5}) and \
                params['alpha'] == 1.0 and \
                params['gamma'] == 1.0 and \
                params['lambda'] == 0.0 and \
                params['alpha_min'] == 0.1
        assert all(state_counts == expected_state_counts)
        assert n_events_et == expected_n_events_et
        assert n_events_fv == expected_n_events_fv
        assert n_cycles_absorption_used == expected_n_cycles_absorption
        assert np.isclose(average_absorption_time, expected_absorption_time)
        assert np.isclose(average_reward, expected_average_reward)
        for key, value in probas_stationary.items():
            assert np.isclose(probas_stationary[key], expected_probas_stationary[key])
        assert np.isclose(average_reward, expected_reward_under_stationarity), f"The estimated average reward must satisfy the expected reward formula under stationarity: average reward = {average_reward}, expected reward = {expected_reward_under_stationarity}"
        assert np.allclose(state_values, expected_V, atol=1E-6)
        assert np.allclose(state_values, agent_fv_optimal.getLearner().getV().getValues(), atol=1E-6), "The observed V(s) must coincide with the V(s) stored in the learner"

    def test_EnvGridworld1DOneTerminal_PolRandomWalk_MetFV(self):
        print("\n*** Running test " + self.id() + " ***")

        # Simulation setup
        seed = 1717

        # Learner and agent definition
        N = 50
        T = 1000
        max_time_steps_fv = 100*N
        min_prop_absorbed_particles = 0.90  # Minimum proportion of particles that should be absorbed after overcoming the maximum number of time steps defined in max_time_steps_fv
        params = dict({'N': N,
                       'T': T,
                       'absorption_set': set(np.arange(5)),
                       'activation_set': set({5}),
                       'alpha': 1.0,
                       'gamma': self.gamma,
                       'lambda': 0.0,
                       'alpha_min': 0.1,
                       # We use alpha_min > 0 because o.w. V(s) is not fully learned (due to the CONTINUING task context, where we receive reards from "infinite" time
                       })
        learner_fv = fv.LeaFV(self.env1d, params['N'], params['T'], params['absorption_set'], params['activation_set'],
                              probas_stationary_start_state_et=None,
                              probas_stationary_start_state_fv=None,
                              criterion=self.learning_criterion,
                              alpha=params['alpha'], gamma=params['gamma'], lmbda=params['lambda'],
                              alpha_update_type=AlphaUpdateType.EVERY_STATE_VISIT,
                              adjust_alpha=True, adjust_alpha_by_episode=False,
                              alpha_min=params['alpha_min'],
                              debug=False)

        # Define the agents for the policies that are used in the tests
        agent_fv_random = agents.GenericAgent(self.policy_random, learner_fv)

        # Store the true state value function in the environment so that we can compare our estimates with those values when running the simulation in the test
        self.env1d.setV(self.V_true_random)

        sim = DiscreteSimulator(self.env1d, agent_fv_random, debug=False)
        state_values, action_values, advantage_values, state_counts, probas_stationary, average_reward, average_absorption_time, n_cycles_absorption_used, n_events_et, n_events_fv  = \
            sim.run(max_time_steps=max_time_steps_fv,
                    max_time_steps_for_absorbed_particles_check=max_time_steps_fv,
                    min_prop_absorbed_particles=min_prop_absorbed_particles,
                    use_average_reward_stored_in_learner=False,
                        ## Set the above parameter to True in case we want to test what happens when we start with an initially estimated average reward,
                        ## e.g. to check if that changes the value function estimation results considerably. To this end, we could call learner_fv.setAverageReward(<value>)
                        ## when instantiating the FV learner above.
                        ## Note that, if we set it to True and set the initial average reward value in the learner to be a too large value,
                        ## we might end up with a very bad V(s) estimate because a too large average reward is subtracted at every
                        ## TD learning step carried out by the FV simulation, and this brings the V(s) estimate very negative for states far away from the positive reward on the right.
                    seed=seed,
                    verbose=True, verbose_period=max_time_steps_fv // 20,
                    plot=False)

        # The expected state values are close to self.V_true_optimal but not quite there
        expected_V = [-0.015381, -0.014586, -0.014211, -0.014184, -0.014429,
                      -0.020892, -0.020013, -0.009237,  0.009575,  0.041849,
                       0.072141,  0.102611,  0.159001,  0.303956,  0.378875,
                       0.547796,  0.612846,  0.743180,  0.854892, -0.017290]
        expected_state_counts = [150, 119, 111, 113, 104, 388, 480, 523, 517, 503, 529, 522, 443, 412, 354, 272, 207, 139, 76, 39]
        expected_n_events_et = params['T']
        expected_n_events_fv = 5000
        expected_n_cycles_absorption = 48
        expected_absorption_time = 20.58333
        expected_average_reward = 0.00222471
        expected_probas_stationary = dict({19: 0.00222471})
        # The following is NOT an expected value for the unit test, but just the expected reward under stationarity computed using its definition,
        # i.e. E(R) = sum{x} p(x)*r(x), where p(x) is the stationary probability
        expected_reward_under_stationarity = estimate_expected_reward(self.env1d, probas_stationary)
        print("\nObserved V(s): " + test_utils.array2str(state_values))
        print("State count: " + test_utils.array2str(state_counts))
        print(f"Number of cycles observed in E(T) estimation: {n_cycles_absorption_used}")
        print(f"Estimated expected reabsorption time E(T): {average_absorption_time}")
        print(f"Estimated expected reward: {average_reward}")
        print(f"Number of time steps in E(T) estimation: {n_events_et}")
        print(f"Number of time steps in FV simulation: {n_events_fv}")
        print(f"Total number of time steps in E(T) + FV simulation: {n_events_et + n_events_fv}")

        plot_estimated_state_value_function(self.env1d, state_values, self.learning_criterion)

        assert  self.nS == 20 and \
                seed == 1717 and \
                max_time_steps_fv == 100*params['N'] and \
                min_prop_absorbed_particles == 0.90 and \
                params['N'] == 50 and \
                params['T'] == 1000 and \
                params['absorption_set'] == set(np.arange(5)) and \
                params['activation_set'] == set({5}) and \
                params['alpha'] == 1.0 and \
                params['gamma'] == 1.0 and \
                params['lambda'] == 0.0 and \
                params['alpha_min'] == 0.1
        assert all(state_counts == expected_state_counts)
        assert n_events_et == expected_n_events_et
        assert n_events_fv == expected_n_events_fv
        assert n_cycles_absorption_used == expected_n_cycles_absorption
        assert np.isclose(average_absorption_time, expected_absorption_time)
        assert np.isclose(average_reward, expected_average_reward)
        for key, value in probas_stationary.items():
            assert np.isclose(probas_stationary[key], expected_probas_stationary[key])
        assert np.isclose(average_reward, expected_reward_under_stationarity), f"The estimated average reward must satisfy the expected reward formula under stationarity: average reward = {average_reward}, expected reward = {expected_reward_under_stationarity}"
        assert np.allclose(state_values, expected_V, atol=1E-6)
        assert np.allclose(state_values, agent_fv_random.getLearner().getV().getValues(), atol=1E-6), "The observed V(s) must coincide with the V(s) stored in the learner"


class Test_EstValueFunctions_EnvGridworld2DWithObstacles(unittest.TestCase, test_utils.EpisodeSimulation):

    @classmethod
    def setUpClass(cls):
        #-- Environment characteristics
        shape = [3, 4]
        cls.nS = np.prod(shape)
        cls.start_state = 8
        cls.isd = np.array([1 if s == cls.start_state else 0 for s in range(cls.nS)])
        cls.env2d = gridworlds.EnvGridworld2D(  shape=shape, terminal_states=set({3}), obstacle_states=set({5}),
                                                rewards_dict=dict({3: +1}),
                                                initial_state_distribution=cls.isd)

        #-- Policy characteristics
        # Random walk policy
        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env2d)

        #-- Plotting parameters
        cls.colormap = cm.get_cmap("jet")

        #-- Simulation setup
        cls.seed = 1717
        cls.nepisodes = 100
        cls.start_state = 8

        cls.gamma = 0.9

        #-- True state value function which is stored in the environment for comparison purposes with the estimated V(s)
        # TODO: (2024/05/02) Fix the correct computation of the true state value function V(s), in terms of: (i) honouring the states that are terminal (currently they have a value that is not zero) (ii) taking care of the obstacles in the environment. The fix should be done mostly in function computing.compute_transition_matrices().
        # Once the above to-do task is done, uncomment the following lines, o.w. we get confused when plotting the observed V(s) on the same plot as the expected V(s)
        #P, _, b, _, _, _ = computing.compute_transition_matrices(cls.env2d, cls.policy_rw)
        #V_true = computing.compute_state_value_function_from_transition_matrix(P, b, gamma=cls.gamma)
        #cls.env2d.setV(V_true)

        #-- Learners and simulators used in tests
        learner_mclambda = mc.LeaMCLambda(cls.env2d, alpha=1.0,
                                          gamma=cls.gamma,
                                          adjust_alpha=True,
                                          adjust_alpha_by_episode=False,
                                          alpha_min=0.0,
                                          learner_type=mc.LearnerType.MC,
                                            ## This implies that we use the standard Monte-Carlo learner, as opposed to the lambda-decayed Monte-Carlo learner.
                                            ## Recall that the approach is equivalent to using LeanerType.LAMBDA_RETURN with lambda = 1.0 (just verified in practice)
                                          debug=False)
        cls.agent_rw_mc = agents.GenericAgent(cls.policy_rw, learner_mclambda)
        cls.sim_mc = DiscreteSimulator(cls.env2d, cls.agent_rw_mc, debug=False)

        # NOTE: If adjusting alpha at the same 1/n rate as in the MC learner, TD(0) learns more slowly...
        # We can reach the same speed of learning as MC if we decrease alpha as 1/sqrt(n) OR if we use e.g. lambda = 0.7.
        # INTERESTING!
        learner_tdlambda = td.LeaTDLambda(cls.env2d, alpha=1.0,
                                          gamma=cls.gamma, lmbda=0.0,
                                          adjust_alpha=True,
                                          adjust_alpha_by_episode=False,
                                          alpha_min=0.0,
                                          func_adjust_alpha=np.sqrt,    # Use a slower decrease rate of alpha so that TD(0) learns faster than with 1/n decrease --i.e. at a similar rate as 1/n decrease in Monte-Carlo above
                                          debug=False)
        cls.agent_rw_td = agents.GenericAgent(cls.policy_rw, learner_tdlambda)
        cls.sim_td = DiscreteSimulator(cls.env2d, cls.agent_rw_td, debug=False)

        # Fleming-Viot learner using groups of FV particles starting at different states outside A
        # Set of absorbing states: we choose the first column of the 2D-grid, but without the start state at 8, as the set A of uninteresting states
        cls.A = set({0, 4})
        # Set of activation states is the outside boundary of A
        cls.B = set({1, 8})
        N = 100
        T = 50      # We should keep this value small in order to avoid too much learning from the regular excursion of the Markov chain (as opposed to learning from the FV excusrion)
        absorption_set = cls.A
        activation_set = cls.B
        learner_fv = fv.LeaFV(  cls.env2d,
                                N, T, absorption_set, activation_set,
                                probas_stationary_start_state_et=None,
                                probas_stationary_start_state_fv=None,
                                criterion=LearningCriterion.DISCOUNTED,
                                estimate_on_fixed_sample_size=True,         # This means that the estimation is done on a fixed number of particles N(s) grouped by starting state s
                                alpha=1.0,
                                gamma=cls.gamma,
                                lmbda=0.0,
                                adjust_alpha=True,
                                adjust_alpha_by_episode=False,
                                alpha_min=0.0,
                                debug=False)
        cls.agent_rw_fv = agents.GenericAgent(cls.policy_rw, learner_fv)
        cls.sim_fv = DiscreteSimulator(cls.env2d, cls.agent_rw_fv, debug=False)

        # Expected state values for all tests
        cls.expected_mc_V = [0.18794174, 0.29649877, 0.46634851, 0.,
                             0.10790858, 0.,         0.31369084, 0.46864529,
                             0.08636770, 0.09085891, 0.18653647, 0.312581]
        # TODO: (2023/12/18) These expected Q values (which were actually obtained from an execution of the estimation process) look wrong... (based on what action (moving direction) is more valuable to achieve the terminal state; in addition, their values are VERY different from the expected Q values for TD(0), even some values here being negative... where did those negative values come from??)
        cls.expected_mc_Q = [[ 0.03023433,  0.22566717, -0.02257150,  -0.04538826],
                             [ 0.05145023,  0.15508904,  0.17511970,  -0.08516019],
                             [-0.01269403,  0.65605520, -0.00413949, -0.17287316],
                             [ 0.,          0.,          0.,          0.        ],
                             [ 0.03521687,  0.03290669, -0.00655919,  0.04634422],
                             [ 0.,          0.,          0.,          0.        ],
                             [ 0.24175533,  0.02429782, -0.04142258,  0.08906026],
                             [ 0.16092266,  0.40257731, -0.04839183, -0.04646285],
                             [ 0.01860712,  0.00192583,  0.08381913, -0.01798438],
                             [-0.00332623,  0.05722877,  0.04953676, -0.01258039],
                             [ 0.04930168,  0.05101225,  0.10644929, -0.02022676],
                             [ 0.18143995,  0.17092198,  0.01145685, -0.05123778]]
        cls.expected_td_V = [0.17765623, 0.28574322, 0.47875633, 0.,
                             0.11770661, 0.,         0.38475264, 0.62173917,
                             0.09284106, 0.12050374, 0.22991270, 0.35113428]
        cls.expected_td_Q = [[0.16072246, 0.25062980, 0.10445315, 0.1596362 ],
                             [0.25833001, 0.44407377, 0.25839356, 0.16133587],
                             [0.45276433, 0.99968037, 0.33858707, 0.26228900],
                             [0.,         0.,         0.,         0.        ],
                             [0.15874380, 0.10327937, 0.08113218, 0.10224487],
                             [0.,         0.,         0.,         0.        ],
                             [0.44888002, 0.48944168, 0.18373345, 0.33437982],
                             [0.99443671, 0.46526329, 0.25611076, 0.32729442],
                             [0.10315783, 0.10242074, 0.08225361, 0.08128193],
                             [0.10232160, 0.18439734, 0.09976743, 0.08166934],
                             [0.32572204, 0.26179925, 0.19081792, 0.10207814],
                             [0.47968696, 0.25279819, 0.25260026, 0.1854122 ]]

        # TODO: (2024/02/19) Update the expected results once we have correctly implemented the value functions learning under the DISCOUNTED reward setting and generated the results
        # For now, these values are the expected results of the not so correct implementation, where the time clock used to estimate P(T>t; s) is the same as the time used to estimate Phi(t,x; s), but these two time clocks are actually different
        cls.expected_fv_V = np.array(
                            [0.026471, 0.180172, 0.537639, 0.007238,
                             0.005534, 0.000000, 0.314387, 0.521715,
                             0.014464, 0.060632, 0.149091, 0.309588])
        cls.expected_fv_Q = np.array(
                            [[0.00239757, 0.02766462, 0.00114565, 0.00145606],
                             [0.31265840, 0.12491134, 0.01472151, 0.00242867],
                             [0.21288913, 0.60385280, 0.08354980, 0.02622499],
                             [0.00723798, 0.00723798, 0.00723798, 0.00723798],
                             [0.00286347, 0.00124553, 0.00435913, 0.00139219],
                             [0.,                 0.,         0.,         0.],
                             [0.19301888, 0.28085474, 0.03435416, 0.11141387],
                             [0.93047469, 0.23359365, 0.16036133, 0.15988098],
                             [0.00675938, 0.01650643, 0.00353169, 0.00415680],
                             [0.04270995, 0.06127918, 0.01570903, 0.00482990],
                             [0.09882721, 0.14403669, 0.04910855, 0.01617216],
                             [0.42700666, 0.15646508, 0.10887363, 0.05639611]])
        cls.expected_fv_state_counts = [74, 75, 88, 48, 77, 0, 95, 82, 228, 165, 96, 70]

    def test_Env_PolRandomWalk_MetMC(self):
        print("\n*** Running test " + self.id() + " ***")

        state_values, action_values, advantage_values, state_counts, _, _, learning_info = \
            self.sim_mc.run(nepisodes=self.nepisodes, seed=self.seed,
                    compute_rmse=True, state_observe=6,
                    verbose=True, verbose_period=self.nepisodes // 20,
                    plot=False, pause=0.1)
        observed_V = state_values
        observed_Q = action_values.reshape((self.env2d.getNumStates(), self.env2d.getNumActions()))
        print("\nObserved state value function: " + test_utils.array2str(observed_V))
        print("Expected state value function: " + test_utils.array2str(self.expected_mc_V))

        assert self.nS == 3*4 and \
               self.seed == 1717 and \
               self.nepisodes == 100 and \
               self.start_state == 8
        assert np.allclose(observed_V, self.expected_mc_V, atol=1E-6)
        assert np.allclose(observed_Q, self.expected_mc_Q, atol=1E-6)

    def test_Env_PolRandomWalk_MetTDLambda(self):
        print("\n*** Running test " + self.id() + " ***")

        state_values, action_values, advantage_values, state_counts, _, _, learning_info = \
            self.sim_td.run(nepisodes=self.nepisodes, seed=self.seed,
                            compute_rmse=True, state_observe=6,
                            verbose=True, verbose_period=self.nepisodes // 20,
                            plot=False, pause=0.1)
        observed_V = state_values
        observed_Q = action_values.reshape((self.env2d.getNumStates(), self.env2d.getNumActions()))
        print("\nObserved state value function: " + test_utils.array2str(observed_V))
        print("Expected state value function: " + test_utils.array2str(self.expected_td_V))

        plot_estimated_state_value_function(self.env2d, state_values, LearningCriterion.DISCOUNTED)

        assert self.nS == 3*4 and \
               self.seed == 1717 and \
               self.nepisodes == 100 and \
               self.start_state == 8
        assert np.allclose(observed_V, self.expected_td_V, atol=1E-6)
        assert np.allclose(observed_Q, self.expected_td_Q, atol=1E-6)

    def test_Env_PolRandomWalk_MetFV(self):
        "Tests the value functions estimation using Fleming-Viot"
        print("\n*** Running test " + self.id() + " ***")

        # (2024/02/19) Note that parameter max_time_steps_fv is by default set to None,
        # which means that it is automatically computed by the _run_simulation_fv() method called by run()
        # and as of the writing of this, it is set to N*100, where N is the number of particles in the FV system.
        state_values, action_values, advantage_values, state_counts, probas_stationary, average_reward, average_cycle_time, n_cycles, n_events_et, n_events_fv = \
           self.sim_fv.run(max_time_steps=500,
                           max_time_steps_for_absorbed_particles_check=500,
                           seed=self.seed, verbose=True, verbose_period=100)

        # The following are the state values (value function) calculated using the average reward observed during the single Markov chain excursion used to estimate E(T_A)
        # therefore it is NOT an inflated estimation of the average reward. However, if the labyrinth is too large, that reward could be well underestimated
        # (as in a classical TD or MC estimator of the average reward).
        observed_values_V = state_values
        observed_values_Q = action_values.reshape((self.env2d.getNumStates(), self.env2d.getNumActions()))

        # Observed frequency of each state (which is an inflated estimation of the stationary probability distribution)
        observed_p = state_counts / np.sum(state_counts)

        print(f"\nNumber of learning steps run: {n_events_et + n_events_fv}")
        print("\nObserved state value function (using the average reward from E(T_A) as correction):\n" + test_utils.array2str(observed_values_V))
        #print("\nObserved state value function (corrected by FV's average reward) (TO BE IMPLEMENTED, MAYBE): " + test_utils.array2str(observed_values))
        print("Expected state value function:\n" + test_utils.array2str(self.expected_fv_V))
        print(f"\nObserved action value function (using the average reward from E(T_A) as correction):\n{observed_values_Q}")
        print(f"Expected action value function:\n{self.expected_fv_Q}")
        print(f"State counts: " + test_utils.array2str(state_counts))
        print(f"Expected state counts: " + test_utils.array2str(self.expected_fv_state_counts))
        print("State frequency distribution (observed during FV simulation): " + test_utils.array2str(observed_p))

        plot_estimated_state_value_function(self.env2d, state_values, LearningCriterion.DISCOUNTED)

        assert self.nS == 3*4 and \
               self.seed == 1717 and \
               self.start_state == 8 and \
               self.A == set({0, 4}) and \
               self.B == set({1, 8})
        assert all(state_counts == self.expected_fv_state_counts)
        assert np.allclose(observed_values_V, self.expected_fv_V, atol=1E-6)
        assert np.allclose(observed_values_Q, self.expected_fv_Q, atol=1E-6)


class Test_EstDifferentialValueFunctions_EnvGridworld2DWithObstacles(unittest.TestCase, test_utils.EpisodeSimulation):
    """
    Test the estimation of the differential value function, i.e. the value function under the average reward setting

    Under this setup, the value of the discount parameter gamma is equal to 1.

    Ref on average reward setting: Sutton, chapter 10, pag. 249.
    """

    @classmethod
    def setUpClass(cls):
        #-- Environment characteristics
        shape = [3, 4]
        cls.nS = np.prod(shape)
        cls.start_state = 8
        cls.isd = np.array([1 if s == cls.start_state else 0 for s in range(cls.nS)])
        cls.env2d = gridworlds.EnvGridworld2D(shape=shape, terminal_states=set({3}), obstacle_states=set({5}),
                                              rewards_dict=dict({3: +1}),
                                              initial_state_distribution=cls.isd)

        #-- Cycle characteristics
        # Set of absorbing states, used to define a cycle as re-entrance into the set which is used to estimate the average reward using renewal theory
        cls.A = set({8})

        #-- Set where a particle activates in the FV context (it should be touching A)
        cls.B = set({4, 9})

        #-- Policy characteristics
        # Random walk policy
        cls.policy_rw = random_walks.PolRandomWalkDiscrete(cls.env2d)

        #-- Plotting parameters
        cls.colormap = cm.get_cmap("jet")

        #-- Simulation setup
        cls.seed = 1717
        # We use a large number of episodes (e.g. 100 or 200) so that we can test that the results with MC and with TD(lambda) are similar and close to the true ones
        # at least when the MC learner is ready... which currently is not because both the state value and the average reward are based on episodic learning,
        # NOT on the average reward criterion.
        cls.nepisodes = 100
        cls.max_time_steps = 4648   # Maximum number of steps over ALL episodes

        #-- True state value function so that we can compare it with the estimated V(s)
        _, P, _, b, g, mu = computing.compute_transition_matrices(cls.env2d, cls.policy_rw)
        V_true = computing.compute_state_value_function_from_transition_matrix(P, b, bias=g, gamma=1.0)
        cls.env2d.setV(V_true)

        # Monte-Carlo learner
        learner_mclambda = mc.LeaMCLambda(cls.env2d,
                                          criterion=LearningCriterion.AVERAGE,
                                          task=LearningTask.CONTINUING,
                                          alpha=1.0,
                                          gamma=1.0,
                                          adjust_alpha=True,
                                          adjust_alpha_by_episode=False,
                                          alpha_min=0.0,
                                          store_history_over_all_episodes=True,
                                          learner_type=mc.LearnerType.MC,
                                          debug=False)
        cls.agent_rw_mc = agents.GenericAgent(cls.policy_rw, learner_mclambda)
        cls.sim_mc = DiscreteSimulator(cls.env2d, cls.agent_rw_mc, debug=False)

        # TD(lambda) learner
        learner_tdlambda = td.LeaTDLambda(cls.env2d,
                                          criterion=LearningCriterion.AVERAGE,
                                          task=LearningTask.CONTINUING,
                                          alpha=1.0,
                                          gamma=1.0, lmbda=0.0,
                                          adjust_alpha=True,
                                          func_adjust_alpha=np.sqrt,
                                          adjust_alpha_by_episode=False,
                                          alpha_min=0.0,
                                          debug=False)
        cls.agent_rw_td = agents.GenericAgent(cls.policy_rw, learner_tdlambda)
        cls.sim_td = DiscreteSimulator(cls.env2d, cls.agent_rw_td, debug=False)

        # Fleming-Viot learner
        cls.N = 100
        cls.T = 1000
        absorption_set = cls.A
        activation_set = cls.B
        learner_fv = fv.LeaFV(  cls.env2d,
                                cls.N, cls.T, absorption_set, activation_set,
                                probas_stationary_start_state_et=None,
                                probas_stationary_start_state_fv=None,
                                states_of_interest=set(cls.env2d.getAllValidStates()).difference(absorption_set),
                                criterion=LearningCriterion.AVERAGE,
                                alpha=1.0,
                                lmbda=0.0,
                                adjust_alpha=True,
                                adjust_alpha_by_episode=False,
                                alpha_min=0.0,
                                debug=False)
        cls.agent_rw_fv = agents.GenericAgent(cls.policy_rw, learner_fv)
        cls.sim_fv = DiscreteSimulator(cls.env2d, cls.agent_rw_fv, debug=False)

        #-- Expected state values for all tests
        # MC learner
        # TODO: (2023/12/18) These expected V values (which were actually obtained from an execution of the estimation process) are WRONG because the terminal state should NOT have value 0 under the average reward criterion learning --i.e. when estimating the *differential* value functions.
        # The reason for this issue is that the estimation of terminal state values has not yet been fully implemented for MC and TD(lambda) learners with lambda > 0. See more details in the entry from 17-Dec-2023 in Tasks-Projects.xlsx Excel file.
        cls.expected_mc_V = [0.30846098, 0.44259874, 0.63029386, 0.,
                             0.20527094, 0.,         0.50675541, 0.65228425,
                             0.13764317, 0.15088129, 0.32957630, 0.4950138]
        # TODO: (2023/12/18) These expected Q values (which were actually obtained from an execution of the estimation process) look wrong... (based on what action (moving direction) is more valuable to achieve the terminal state; in addition, their values are VERY different from the expected Q values for TD(0))
        cls.expected_mc_Q = [[-0.10497926,  0.19869401,  0.17612975, 0.038616470],
                             [ 0.07399127,  0.17327522,  0.31313544, -0.11780319],
                             [ 0.00487184,  0.27108070,  0.00280095,  0.35154037],
                             [ 0.,          0.,          0.,          0.        ],
                             [ 0.30322845,  0.11356473, -0.10983344, -0.1016888 ],
                             [ 0.,          0.,          0.,          0.        ],
                             [ 0.45877299,  0.02778924, -0.12370426,  0.14389743],
                             [ 0.08703213,  0.56738527, -0.06992857,  0.06779541],
                             [-0.04939215, -0.10683988,  0.24918431,  0.04469088],
                             [-0.12835815,  0.06668889,  0.12368679,  0.08886378],
                             [-0.02297774, -0.08885636,  0.48472477, -0.04331436],
                             [ 0.14209802,  0.51843737,  0.17254239, -0.33806398]]

        # TD learner
        cls.expected_td_V = [  0.11512802,  0.26457653,  0.45853404, -0.07204246,
                               0.00811229,  0.,          0.38054146,  0.58872488,
                              -0.04774093,  0.00567791,  0.19265345,  0.33741384]
        cls.expected_td_Q = [[ 0.09715254,  0.23310079, -0.01600686,  0.09617661],
                             [ 0.24471781,  0.44654475,  0.24577001,  0.0983613 ],
                             [ 0.45764161,  0.89897935,  0.35067266,  0.25019221],
                             [-0.07204246, -0.07204246, -0.07204246, -0.07204246],
                             [ 0.09298804, -0.01930358, -0.07420419, -0.02130343],
                             [ 0.,          0.,          0.,          0.        ],
                             [ 0.45329009,  0.49396376,  0.13453719,  0.34560123],
                             [ 0.89494849,  0.47016001,  0.24945365,  0.33657962],
                             [-0.01997194, -0.02932902, -0.07233910, -0.07454767],
                             [-0.02875755,  0.13274115, -0.03439765, -0.07370337],
                             [ 0.33522403,  0.25378455,  0.14492441, -0.0279854 ],
                             [ 0.48441351,  0.24541169,  0.24528768,  0.13707096]]

        # DM-2023/09/20: The expected value function (expected_fv_V) is similar to the expected value function under TD
        # because the problem is small... However, in problems where FV really observes more often the rare states with
        # rewards, the value function computed as done now (i.e. using the REGULAR average reward estimation as correction
        # value to compute the differential value function) is expected to diverge largely from the correct value function
        # which should use the FV estimator of the average reward as correction value.
        # IMPORTANT: The value of V(s) for the states in the *absorption set* are estimated from the single Markov chain
        # simulation that estimates E(T_A) NOT from the FV simulation! This is ok, because by definition the set of
        # absorption states are those that are visited often... hence their rewards would be estimated accurately...
        # ALTHOUGH... no impact from the large and rare rewards would be seen for those states because the rare reward
        # would almost NEVER be observed! Hmmm... We would need some way of leveraging FV to estimate the value of those
        # states more accurately (i.e. an estimate that also observes the rare reward that is observed by the states OUTSIDE
        # the absorption set...). Perhaps we could do that by starting the Fleming-Viot simulation (on ALL N particles)
        # ALSO at those states... when ALL the N trajectories leave the set... (which at some point should leave), we start applying Fleming-Viot.
        # Will this be too slow to launch the FV process??
        # AN IMPROVEMENT ON THIS IDEA IS THE FOLLOWING: We start all N particles INSIDE the absorption set, and as they
        # start getting out of A we start the FV process ON THE PARTICLES THAT ARE AVAILABLE FOR IT! So, the number of
        # particles varies as time progresses (it goes up as new particles exit the absorption set A, and it goes down
        # as the particles reach a terminal state --in which case they are restarted inside A, e.g. at the START of the labyrinth).
        # The maximum number of particles in the system is N.
        # TODO: (2024/05/07) Check these expected V(s) values because after the recent changes I did on different aspects of the FV simulation and FV-based value function estimation process, the estimated V(s) is worse than before the changes, i.e. it is farther away from the true V(s) value computed by the brand new computing.compute_state_value_function_from_transition_matrix() function, as is also farther away than the V(s) estimated by TD(0).
        cls.expected_fv_V = np.array(
                            [ 0.034743,  0.233790, 0.489875, -0.037515,
                             -0.039710,  0.000000, 0.298950,  0.447024,
                             -0.020319, -0.039758, 0.095426, 0.181439])
        # (2023/12/18) The expected action value function seems reasonable in terms of what action is better at each state to reach the terminal state at the upper-right cell of the labyrinth
        cls.expected_fv_Q = np.array(
                            [[-0.01367791,  0.11848694, -0.05318403, -0.01168208],
                             [0.17164916,  0.45940778,  0.1586421,  -0.00842918],
                             [0.52806234,  0.95464411,  0.24212508,  0.16495517],
                             [-0.03751482, -0.03751482, -0.03751482, -0.03751482],
                             [-0.01941363, -0.04611871, -0.02248051, -0.0529884],
                             [0., 0., 0., 0.],
                             [0.5021308,   0.27621888,  0.00773258,  0.12787165],
                             [0.84994117, 0.34307159, 0.0664185, 0.23193807],
                             [-0.00798979, -0.00769294, -0.00754817, -0.00751653],
                             [-0.05953017, -0.02484684, -0.04193591, -0.02717974],
                             [0.14703614,  0.02765407, -0.01561109, -0.06454853],
                             [0.28667113,  0.04775472,  0.04977956, -0.0088709]])
        cls.expected_fv_average_reward = 0.023652
        cls.expected_fv_cycle_time = 10.912
        cls.expected_fv_n_cycles = 91
        cls.expected_fv_state_counts = [1085, 843, 481, 224, 817, 0, 528, 383, 186, 533, 559, 513]
        cls.expected_fv_total_events = sum(cls.expected_fv_state_counts)

        #-- Expected values that do NOT depend on the estimation method (e.g. average reward, number of cycles, etc.)
        # (except for the average reward, whose estimated value depends on whether we estimate it using cycles or not --thus the distinction done here between these two alternatives)
        cls.expected_average_reward = 0.0215100
        cls.expected_average_reward_from_cycles = 0.021382
        cls.expected_cycle_time = 10.3812
        cls.expected_n_cycles = 446
        cls.expected_state_counts_in_complete_cycles = [602, 482, 228, 99, 707, 0, 223, 182, 929, 602, 328, 261]
            ## This is used in the estimation of the stationary probability from cycles
            ## Note that this counts do NOT count the states visited before the first cycle was completed (so that we use only COMPLETE cycles to estimate the stationary probabilities)

        # Expected stationary distribution of states, using regular estimation and using renewal theory (the results should be similar)
        # These results do NOT depend on the learning method for the value functions (be it Monte-Carlo or TD(lambda) or be it under the discounted or average criterion).
        cls.expected_counts = [602, 482, 228, 100, 709,   0, 223, 182, 932, 602, 328, 261]
        cls.expected_p = [0.129490, 0.103678, 0.049043, 0.021510,
                          0.152506, 0.000000, 0.047967, 0.039148,
                          0.200473, 0.129490, 0.070553, 0.056141]
        cls.expected_p_from_cycles = [0.130022, 0.104104, 0.049244, 0.021382,
                                      0.152916, 0.000000, 0.048164, 0.039309,
                                      0.200864, 0.130022, 0.070842, 0.056371]
        cls.expected_p_fv = [0.171830, 0.085760, 0.048607, cls.expected_fv_average_reward,
                             0.017149,   np.nan, 0.056226, 0.036251,
                               np.nan, 0.013654, 0.072440, 0.050885]

    def test_Env_PolRandomWalk_MetMC(self):
        print("\n*** Running test " + self.id() + " ***")

        state_values, action_values, advantage_values, state_counts, _, _, learning_info = \
            self.sim_mc.run(nepisodes=self.nepisodes, max_time_steps=self.max_time_steps, seed=self.seed,
                             compute_rmse=True, state_observe=6,
                             verbose=True, verbose_period=self.nepisodes // 20,
                             plot=False, pause=0.1)
        observed_V = state_values
        observed_Q = action_values.reshape((self.env2d.getNumStates(), self.env2d.getNumActions()))
        observed_average_reward = self.agent_rw_mc.getLearner().getAverageReward()
        print(f"\nNumber of learning steps run: {learning_info['nsteps']}")
        print("Observed state value function: " + test_utils.array2str(observed_V))
        print("Expected state value function: " + test_utils.array2str(self.expected_mc_V))
        print(f"Average reward: {observed_average_reward}")
        print(f"Expected average reward: {self.expected_average_reward}")

        observed_p = state_counts / np.sum(state_counts)
        print(f"State counts: " + test_utils.array2str(state_counts))
        print(f"Expected state counts: " + test_utils.array2str(self.expected_counts))

        assert self.nS == 3*4 and \
               self.seed == 1717 and \
               self.nepisodes == 100 and \
               self.start_state == 8
        assert learning_info['nsteps'] == 4648
        assert np.allclose(observed_V, self.expected_mc_V, atol=1E-6)
        assert np.allclose(observed_Q, self.expected_mc_Q, atol=1E-6)
        assert np.isclose(observed_average_reward, self.expected_average_reward, atol=1E-6)

        assert all(state_counts == self.expected_counts)
        assert np.allclose(observed_p, self.expected_p, atol=1E-6)

    def test_Env_PolRandomWalk_MetMC_FromCycles(self):
        """
        The results of this test should change (w.r.t. to the previous test that is NOT based on cycles) ONLY in terms of the estimated average reward,
        but NOT in terms of the estimated value functions, because the differential value functions are computed using the average reward that is estimated
        NOT on cycles but as a regular average of the observed rewards.
        """
        print("\n*** Running test " + self.id() + " ***")

        state_values, action_values, advantage_values, state_counts, _, _, learning_info = \
            self.sim_mc.run(nepisodes=self.nepisodes, max_time_steps=self.max_time_steps, seed=self.seed,
                            compute_rmse=False, state_observe=None, set_cycle=self.A,
                            verbose=True, verbose_period=100,
                            plot=False, pause=0.1)
        observed_V = state_values
        observed_Q = action_values.reshape((self.env2d.getNumStates(), self.env2d.getNumActions()))
        observed_average_reward = self.agent_rw_mc.getLearner().getAverageReward()
        print(f"\nNumber of learning steps run: {learning_info['nsteps']}")
        print("Observed state value function: " + test_utils.array2str(observed_V))
        print("Expected state value function: " + test_utils.array2str(self.expected_mc_V))
        print(f"Average reward: {observed_average_reward}")
        print(f"Expected average reward: {self.expected_average_reward}")

        observed_p = state_counts / np.sum(state_counts)
        print(f"\nState counts: " + test_utils.array2str(state_counts))
        print(f"Expected state counts: " + test_utils.array2str(self.expected_counts))
        print("State probability distribution (usual calculation as relative frequency): " + test_utils.array2str(observed_p))
        print("Expected state probability distribution: " + test_utils.array2str(self.expected_p))

        observed_p_from_cycles = learning_info['state_counts_in_complete_cycles'] / learning_info['num_cycles'] / learning_info['expected_cycle_time']
        print("\nState probability distribution using renewal theory: " + test_utils.array2str(observed_p_from_cycles))
        print("Expected state prob. distrib. using renewal theory:  " + test_utils.array2str(self.expected_p_from_cycles))
        print("VALID state counts within cycles (state visits are counted ONLY after the first (possibly partial) cycle has been completed): " + test_utils.array2str(learning_info['state_counts_in_complete_cycles']))
        print("Expected VALID state counts within cycles: " + test_utils.array2str(self.expected_state_counts_in_complete_cycles))
        print(f"Observed average cycle time on {learning_info['num_cycles']} cycles (expected={self.expected_n_cycles}): {learning_info['expected_cycle_time']} (expected={self.expected_cycle_time})")

        assert self.nS == 3*4 and \
               self.seed == 1717 and \
               self.nepisodes == 100 and \
               self.start_state == 8 and \
               self.A == set({8})
        assert learning_info['nsteps'] == 4648
        assert np.allclose(observed_V, self.expected_mc_V, atol=1E-6)
        assert np.allclose(observed_Q, self.expected_mc_Q, atol=1E-6)
        assert np.isclose(observed_average_reward, self.expected_average_reward, atol=1E-6)

        assert all(state_counts == self.expected_counts)
        assert np.allclose(observed_p, self.expected_p, atol=1E-6)
        assert np.allclose(observed_p_from_cycles, self.expected_p_from_cycles, atol=1E-6)
        assert np.isclose(learning_info['expected_cycle_time'], self.expected_cycle_time, atol=1E-1)
        assert learning_info['num_cycles'] == self.expected_n_cycles

    def test_Env_PolRandomWalk_MetTDLambda(self):
        print("\n*** Running test " + self.id() + " ***")

        state_values, action_values, advantage_values, state_counts, _, _, learning_info = \
            self.sim_td.run(nepisodes=self.nepisodes, max_time_steps=self.max_time_steps, seed=self.seed,
                             compute_rmse=True, state_observe=6,
                             verbose=True, verbose_period=self.nepisodes // 20,
                             plot=False, pause=0.1)
        observed_V = state_values
        observed_Q = action_values.reshape((self.env2d.getNumStates(), self.env2d.getNumActions()))
        observed_average_reward = self.agent_rw_td.getLearner().getAverageReward()
        print(f"\nNumber of learning steps run: {learning_info['nsteps']}")
        print("Observed state value function: " + test_utils.array2str(observed_V))
        print("Expected state value function: " + test_utils.array2str(self.expected_td_V))
        print(f"Average reward: {observed_average_reward}")
        print(f"Expected average reward: {self.expected_average_reward}")

        observed_p = state_counts / np.sum(state_counts)
        print(f"\nState counts: " + test_utils.array2str(state_counts))
        print(f"Expected state counts: " + test_utils.array2str(self.expected_counts))
        print("State probability distribution: " + test_utils.array2str(observed_p))
        print("Expected state probability distribution: " + test_utils.array2str(self.expected_p))

        print(f"learning steps: {learning_info['nsteps']}")

        plot_estimated_state_value_function(self.env2d, state_values, LearningCriterion.AVERAGE)

        assert self.nS == 3*4 and \
               self.seed == 1717 and \
               self.nepisodes == 100 and \
               self.start_state == 8
        assert learning_info['nsteps'] == 4648
        assert np.allclose(observed_V, self.expected_td_V, atol=1E-6)
        assert np.allclose(observed_Q, self.expected_td_Q, atol=1E-6)
        assert np.isclose(observed_average_reward, self.expected_average_reward, atol=1E-6)

        assert all(state_counts == self.expected_counts)
        assert np.allclose(observed_p, self.expected_p, atol=1E-6)

    def test_Env_PolRandomWalk_MetTDLambda_FromCycles(self):
        """
        The results of this test should change (w.r.t. to the previous test that is NOT based on cycles) ONLY in terms of the estimated average reward,
        but NOT in terms of the estimated value functions, because the differential value functions are computed using the average reward that is estimated
        NOT on cycles but as a regular average of the observed rewards.
        """
        print("\n*** Running test " + self.id() + " ***")

        state_values, action_values, advantage_values, state_counts, _, _, learning_info = \
           self.sim_td.run(nepisodes=self.nepisodes, max_time_steps=self.max_time_steps, seed=self.seed,
                         compute_rmse=False, state_observe=None, set_cycle=self.A,
                         verbose=True, verbose_period=100,
                         plot=False, pause=0.1)
        observed_V = state_values
        observed_Q = action_values.reshape((self.env2d.getNumStates(), self.env2d.getNumActions()))
        observed_average_reward = self.agent_rw_td.getLearner().getAverageReward()
        print(f"\nNumber of learning steps run: {learning_info['nsteps']}")
        print("Observed state value function (learned by TD): " + test_utils.array2str(observed_V))
        print("Expected state value function: " + test_utils.array2str(self.expected_td_V))

        observed_p = state_counts / np.sum(state_counts)
        print("\nState counts: " + test_utils.array2str(state_counts))
        print("Expected state counts: " + test_utils.array2str(self.expected_counts))
        print("State probability distribution (usual calculation as relative frequency): " + test_utils.array2str(observed_p))
        print("Expected state probability distribution: " + test_utils.array2str(self.expected_p))

        observed_p_from_cycles = learning_info['state_counts_in_complete_cycles'] / learning_info['num_cycles'] / learning_info['expected_cycle_time']
        print("\nState probability distribution using renewal theory: " + test_utils.array2str(observed_p_from_cycles))
        print("Expected state prob. distrib. using renewal theory:  " + test_utils.array2str(self.expected_p_from_cycles))
        print("VALID state counts within cycles (state visits are counted ONLY after the first (possibly partial) cycle has been completed): " + test_utils.array2str(learning_info['state_counts_in_complete_cycles']))
        print("Expected VALID state counts within cycles: " + test_utils.array2str(self.expected_state_counts_in_complete_cycles))
        print(f"Observed average cycle time on {learning_info['num_cycles']} cycles (expected={self.expected_n_cycles}): {learning_info['expected_cycle_time']}")

        # The following calculation of the average reward from cycles assumes that rewards are only observed at terminal states
        observed_average_reward_from_cycles = sum([p*self.env2d.getReward(x) for x, p in enumerate(observed_p_from_cycles) if x in self.env2d.getTerminalStates()])
        print(f"\nAverage reward (usual calculation): {observed_average_reward}")
        print(f"Average reward (using renewal theory): {observed_average_reward_from_cycles}")
        print(f"Expected average reward: {self.expected_average_reward_from_cycles}")

        assert self.nS == 3*4 and \
               self.seed == 1717 and \
               self.nepisodes == 100 and \
               self.start_state == 8 and \
               self.A == set({8})
        assert learning_info['nsteps'] == 4648
        assert np.allclose(observed_V, self.expected_td_V, atol=1E-6)
        assert np.allclose(observed_Q, self.expected_td_Q, atol=1E-6)
        assert np.isclose(observed_average_reward, self.expected_average_reward, atol=1E-6)

        assert all(state_counts == self.expected_counts)
        assert np.allclose(observed_p, self.expected_p, atol=1E-6)
        assert np.allclose(observed_p_from_cycles, self.expected_p_from_cycles, atol=1E-6)
        assert np.isclose(observed_average_reward_from_cycles, self.expected_average_reward_from_cycles, atol=1E-6)
        assert np.isclose(learning_info['expected_cycle_time'], self.expected_cycle_time, atol=1E-1)
        assert learning_info['num_cycles'] == self.expected_n_cycles

    def test_Env_PolRandomWalk_MetFV(self):
        "Tests the differential value functions estimation using Fleming-Viot"
        print("\n*** Running test " + self.id() + " ***")

        # (2024/02/14) Note that parameter max_time_steps_fv is by default set to None,
        # which means that it is automatically computed by the _run_simulation_fv() method called by run()
        # and as of the writing of this, it is set to N*100, where N is the number of particles in the FV system.
        min_prop_absorbed_particles = 1.0
        state_values, action_values, advantage_values, state_counts, probas_stationary, average_reward, average_cycle_time, n_cycles, n_events_et, n_events_fv = \
           self.sim_fv.run(min_prop_absorbed_particles=min_prop_absorbed_particles, seed=self.seed, verbose=True, verbose_period=100)

        # The following are the state values (value function) calculated using the average reward observed during the single Markov chain excursion used to estimate E(T_A)
        # therefore it is NOT an inflated estimation of the average reward. However, if the labyrinth is too large, that reward could be well underestimated
        # (as in a classical TD or MC estimator of the average reward).
        observed_values_V = state_values
        observed_values_Q = action_values.reshape((self.env2d.getNumStates(), self.env2d.getNumActions()))

        # Observed frequency of each state (which is an inflated estimation of the stationary probability distribution)
        observed_p = state_counts / np.sum(state_counts)
        # FV-adjusted frequency of each state (which is the actual FV estimation of the stationary probability distribution)
        # (note that we convert the result originally stored in a dictionary (probas_stationary) to a list to facilitate
        # the comparison used below in the assertion on this result.
        observed_p_fv = [probas_stationary.get(x, np.nan) for x in self.env2d.getAllStates()]
        observed_average_reward = np.nansum([p*self.env2d.getReward(x) for x, p in enumerate(observed_p_fv) if x in self.env2d.getTerminalStates()])

        print(f"\nNumber of learning steps run: {n_events_et + n_events_fv} (should coincide with the sum of the state counts minus 1, because the start state is not counted in the number of learning steps ({sum(state_counts)-1}))")
        assert n_events_et + n_events_fv == sum(state_counts) - 1
        print(f"Expected number of learning steps: {self.expected_fv_total_events}")

        print("\nObserved state value function (using the FV-based average reward as correction):\n" + test_utils.array2str(observed_values_V))
        print("Expected state value function:\n" + test_utils.array2str(self.expected_fv_V))
        print(f"\nObserved action value function (using the FV-based average reward as correction):\n{observed_values_Q}")
        print(f"Expected state value function:\n{self.expected_fv_Q}")
        print(f"State counts: " + test_utils.array2str(state_counts))
        print(f"Expected state counts: " + test_utils.array2str(self.expected_fv_state_counts))
        print("State frequency distribution (observed during FV simulation): " + test_utils.array2str(observed_p))
        print("\nState probability distribution using the ET+FV estimator: " + test_utils.array2str(observed_p_fv))
        print("Expected state probability distribution using the ET+FV estimator: " + test_utils.array2str(self.expected_p_fv))
        print(f"(observed average cycle time on {n_cycles} cycles (expected={self.expected_fv_n_cycles}): {average_cycle_time} (expected={self.expected_fv_cycle_time}))")
        print(f"\nEstimated average reward (using FV estimator): {observed_average_reward}")
        print(f"Expected estimated average reward (using FV estimator): {self.expected_fv_average_reward}")
        print(f"Estimated average reward (using TD estimator): {self.expected_average_reward}")

        plot_estimated_state_value_function(self.env2d, state_values, LearningCriterion.AVERAGE)

        assert self.nS == 3*4 and \
               min_prop_absorbed_particles == 1.0 and \
               self.N == 100 and \
               self.T == 1000 and \
               self.seed == 1717 and \
               self.start_state == 8 and \
               self.A == set({8}) and \
               self.B == set({4, 9})
        # Assertions about state counts
        assert all(state_counts == self.expected_fv_state_counts)

        # Assertions about the estimated expected cycle time
        assert np.isclose(average_cycle_time, self.expected_fv_cycle_time, atol=1E-1)
        assert n_cycles == self.expected_fv_n_cycles

        # Assertions about the stationary distribution of the state of interest estimated by FV
        assert np.allclose(observed_p_fv, self.expected_p_fv, atol=1E-6, equal_nan=True)

        # Assertions about the average reward estimated by FV
        assert np.isclose(observed_average_reward, self.expected_fv_average_reward, atol=1E-6)
        assert np.isclose(observed_average_reward, average_reward, atol=1E-6), \
            f"The average reward computed from the estimated stationary probabilities by FV ({observed_average_reward}) must coincide with the estimated average reward (`average_reward={average_reward}`) returned by the FV simulator"
        assert np.isclose(observed_average_reward, self.agent_rw_fv.getLearner().getAverageReward(), atol=1E-6), \
            "The average reward stored in the FV learner must coincide with the average reward estimated by the FV estimator"

        # Assertions about the value functions
        assert np.allclose(observed_values_V, self.expected_fv_V, atol=1E-6)
        assert np.allclose(observed_values_Q, self.expected_fv_Q, atol=1E-6)


class Test_EstValueFunctionV_MetMCLambda_EnvMountainCar(unittest.TestCase, test_utils.EpisodeSimulation):

    def __init__(self, *args, **kwargs):
        self.seed = kwargs.pop('seed', 1717)
        self.nepisodes = kwargs.pop('nepisodes', 30)  # 20000) #100000) #30000) #200) #2000)
        self.max_time_steps_per_episode = kwargs.pop('max_time_steps_per_episode', 500)  # Maximum number of steps to run per episode
        self.normalizer = kwargs.pop('normalizer', 1)  # Normalize for the plots: Set it to max_time_steps_per_episode when the rewards are NOT sparse (i.e. are -1 every where except at terminal states), o.w. set it to 1 (when rewards are sparse, i.e. they occur at terminal states)
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
        print("\n*** Running test " + self.id() + " ***")

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
        sim = DiscreteSimulator(self.env, agent_rw_mc, debug=False)

        time_start = timer()
        _, _, _, _, _, _, learning_info = sim.run( nepisodes=self.nepisodes, max_time_steps_per_episode=self.max_time_steps_per_episode,
                                                seed=self.seed,
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

            fig.suptitle("Convergence of the estimated value function V (gamma={:.2f}, lambda={:.2f}, alpha={:.2f}, {}, max #steps = {})" \
                        .format(params['gamma'], params['lambda'], params['alpha'], params['alpha_update_type'].name, self.max_time_steps_per_episode))

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
        print("\n*** Running test " + self.id() + " ***")

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
        sim_mc = DiscreteSimulator(self.env, agent_rw_mc, debug=False)
        sim_lambda_return = DiscreteSimulator(self.env, agent_rw_lambda_return, debug=False)

        # MC execution
        time_start = timer()
        _, _, _, _, _, _, learning_info = sim_mc.run(nepisodes=self.nepisodes, max_time_steps_per_episode=self.max_time_steps_per_episode,
                                                   seed=self.seed,
                                                   compute_rmse=False, state_observe=None,
                                                   verbose=True, verbose_period=max(1, int(self.nepisodes / 10)),
                                                   verbose_convergence=verbose_convergence,
                                                   plot=False, pause=0.001)
        time_end = timer()
        exec_time = time_end - time_start
        print("[MC] Execution time: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

        # Lambda-Return
        time_start = timer()
        _, _, _, _, _, _, learning_info = sim_lambda_return.run(nepisodes=self.nepisodes,
                                                              max_time_steps_per_episode=self.max_time_steps_per_episode,
                                                              seed=self.seed,
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
        test_suite_offline.addTest(Test_EstStateValueV_MetOffline_EnvDeterministicNextState("test_EnvGridworld1DClassic_PolRandomWalk_Met_TestOneCase"))
        test_suite_offline.addTest(Test_EstStateValueV_MetOffline_EnvDeterministicNextState("test_EnvGridworld1DOneTerminal_PolRandomWalk_Met_TestOneCase"))
        # DM-2022/06: For now we skip the test on the Mountain Car because the estimation takes too long to converge because
        # we are using a random policy and reaching the reward under this policy is very rare...
        # We will reactivate this test when we find the optimum policy and then we estimate the value function under the optimum policy.
        #test_suite_offline.addTest(Test_EstStateValueV_MetOffline_EnvDeterministicNextState("test_EnvMountainCarDiscreteActions_PolRandomWalk_Met_TestOneCase"))

        # --- Gridworld tests
        test_suite_gw1d = unittest.TestSuite()
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworld1D("test_EnvGridworld1DClassic_PolRandomWalk_MetMC_TestSeveralAlphasAndAlphaAdjustments"))
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworld1D("test_EnvGridworld1DClassic_PolRandomWalk_MetMC_TestGammaSmallerThan1"))
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworld1D("test_EnvGridworld1DClassic_PolRandomWalk_MetMC_TestRMSETwice"))
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworld1D("test_EnvGridworld1DClassic_PolRandomWalk_MetLambdaReturn_TestSeveralAlphasLambdasAlphaAdjustments"))
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworld1D("test_EnvGridworld1DClassic_PolRandomWalk_MetLambdaReturn_TestGammaLessThan1"))
        test_suite_gw1d.addTest(Test_EstStateValueV_EnvGridworld1D("test_EnvGridworld1DClassic_PolRandomWalk_MetTDLambda_TestSeveralAlphasLambdas"))

        # Gridworld with one terminal state (on the right) having +1 reward
        test_suite_gw1d1t = unittest.TestSuite()
        test_suite_gw1d1t.addTest(Test_EstDifferentialStateValueV_EnvGridworld1D("test_EnvGridworld1DOneTerminal_PolOptimal_MetTDLambdaGt0"))
        test_suite_gw1d1t.addTest(Test_EstDifferentialStateValueV_EnvGridworld1D("test_EnvGridworld1DOneTerminal_PolOptimal_MetTDLambda0"))
        test_suite_gw1d1t.addTest(Test_EstDifferentialStateValueV_EnvGridworld1D("test_EnvGridworld1DOneTerminal_PolOptimal_MetFV"))
        test_suite_gw1d1t.addTest(Test_EstDifferentialStateValueV_EnvGridworld1D("test_EnvGridworld1DOneTerminal_PolRandomWalk_MetTDLambdaGt0"))
        test_suite_gw1d1t.addTest(Test_EstDifferentialStateValueV_EnvGridworld1D("test_EnvGridworld1DOneTerminal_PolRandomWalk_MetTDLambda0"))
        test_suite_gw1d1t.addTest(Test_EstDifferentialStateValueV_EnvGridworld1D("test_EnvGridworld1DOneTerminal_PolRandomWalk_MetFV"))

        test_suite_gw2dobstacles = unittest.TestSuite()
        test_suite_gw2dobstacles.addTest(Test_EstValueFunctions_EnvGridworld2DWithObstacles("test_Env_PolRandomWalk_MetMC"))
        test_suite_gw2dobstacles.addTest(Test_EstValueFunctions_EnvGridworld2DWithObstacles("test_Env_PolRandomWalk_MetTDLambda"))
        test_suite_gw2dobstacles.addTest(Test_EstValueFunctions_EnvGridworld2DWithObstacles("test_Env_PolRandomWalk_MetFV"))
        test_suite_gw2dobstacles.addTest(Test_EstDifferentialValueFunctions_EnvGridworld2DWithObstacles("test_Env_PolRandomWalk_MetMC"))
        test_suite_gw2dobstacles.addTest(Test_EstDifferentialValueFunctions_EnvGridworld2DWithObstacles("test_Env_PolRandomWalk_MetMC_FromCycles"))
        test_suite_gw2dobstacles.addTest(Test_EstDifferentialValueFunctions_EnvGridworld2DWithObstacles("test_Env_PolRandomWalk_MetTDLambda"))
        test_suite_gw2dobstacles.addTest(Test_EstDifferentialValueFunctions_EnvGridworld2DWithObstacles("test_Env_PolRandomWalk_MetTDLambda_FromCycles"))
        test_suite_gw2dobstacles.addTest(Test_EstDifferentialValueFunctions_EnvGridworld2DWithObstacles("test_Env_PolRandomWalk_MetFV"))

        # --- Mountain Car tests
        test_suite_mountain = unittest.TestSuite()
        test_suite_mountain.addTest(Test_EstValueFunctionV_MetMCLambda_EnvMountainCar("test_Env_PolRandomWalk_MetMCLambdaReturn_TestMCvsLambdaReturn"))

        # Run the test suites
        runner.run(test_suite_offline)
        runner.run(test_suite_gw1d)
        runner.run(test_suite_gw1d1t)
        runner.run(test_suite_gw2dobstacles)
        runner.run(test_suite_mountain)
    else:
        # Use this when we want to recover the output of the test in the Python session and analyze it
        test_obj = Test_EstValueFunctionV_MetMCLambda_EnvMountainCar()
        test_obj.setUpClass()
        state_values, state_counts, params, sim, learning_info = test_obj.no_test_Env_PolRandomWalk_MetMC_TestOneCase()
