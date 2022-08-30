# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:19:38 2022

@author: Daniel Mastropietro
@description: Unit tests for estimators (V, Q) on discrete-time MDPs.
@details: Naming conventions follow the instructions given in test_conventions.txt.
"""
import unittest

import numpy as np
from matplotlib import pyplot as plt, cm

from Python.lib import estimators

from Python.lib.agents import GenericAgent
from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners.episodic.discrete import td
from Python.lib.agents.policies import random_walks

from Python.lib.environments import gridworlds, mountaincars

from Python.lib.simulators.discrete import Simulator as DiscreteSimulator


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


class Test_EstStateValueV_EnvGridworlds(unittest.TestCase):
    # Note: nice explanation about the three types of methods that can be defined in Python: instance, class, static
    # https://stackoverflow.com/questions/54264073/what-is-the-use-and-when-to-use-classmethod-in-python
    # See the only answer by Navy Cheng.

    @classmethod
    def setUpClass(cls):
        # The environment
        cls.env1d = gridworlds.EnvGridworld1D(length=21)

        # Plotting parameters
        cls.colormap = cm.get_cmap("jet")

    def test_EnvGridworld1D_PolRandomWalk_MetTDLambda_TestSeveralAlphasLambdas(self):
        """This test intends to reproduce the results in Sutton 2018 on the TD(lambda) algorithm applied to learn
        the state value function in a 1D gridworld
        """
        print("\n")
        print("Running test {}...".format(self.id()))

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
        start = None
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
                          .format(idx_simul, n_simul, lmbda, alpha))

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
                                                 start=start,
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

        print("Average RMSE and its standard error at last episode by lambda:\n{}".format(np.c_[rmse_episodes_mean, rmse_episodes_se]))
        assert np.allclose(rmse_episodes_mean, [[0.45592716, 0.40840256, 0.36065968],
                                                [0.42083822, 0.35093946, 0.29653623],
                                                [0.35946782, 0.28547410, 0.32950694],
                                                [0.32520426, 0.28876721, 0.45260387]])
        assert np.allclose(rmse_episodes_se,   [[0.01244143, 0.02139052, 0.02996329],
                                                [0.01891116, 0.03116384, 0.03592948],
                                                [0.02941829, 0.03871512, 0.02411244],
                                                [0.03452769, 0.03385807, 0.02151384]])


if __name__ == '__main__':
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line

    runner = unittest.TextTestRunner()

    # Run all tests
    #unittest.main()

    # Create the test suites
    test_suite_1 = unittest.TestSuite()
    test_suite_1.addTest(Test_EstStateValueV_MetOffline_EnvDeterministicNextState("test_EnvGridworld1D_PolRandomWalk_Met_TestOneCase"))
    test_suite_1.addTest(Test_EstStateValueV_MetOffline_EnvDeterministicNextState("test_EnvGridworld1DOneTerminal_PolRandomWalk_Met_TestOneCase"))
    # DM-2022/06: For now we skip the test on the Mountain Car because the estimation takes too long to converge because
    # we are using a random policy and reaching the reward under this policy is very rare...
    # We will reactivate this test when we find the optimum policy and then we estimate the value function under the optimum policy.
    #test_suite_1.addTest(Test_EstStateValueV_MetOffline_EnvDeterministicNextState("test_EnvMountainCarDiscreteActions_PolRandomWalk_Met_TestOneCase"))

    test_suite_2 = unittest.TestSuite()
    test_suite_2.addTest(Test_EstStateValueV_EnvGridworlds("test_EnvGridworld1D_PolRandomWalk_MetTDLambda_TestSeveralAlphasLambdas"))

    # Run the test suites
    runner.run(test_suite_1)
    runner.run(test_suite_2)
