# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 15:26:38 2022

@author: Daniel Mastropietro
@description: Test of classes defined in the `simulators` module.
"""
import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

from Python.lib.agents import GenericAgent
from Python.lib.agents.learners import LearnerTypes
from Python.lib.agents.learners.continuing.fv import LeaFV
from Python.lib.agents.learners.continuing.mc import LeaMC
from Python.lib.agents.learners.episodic.discrete import td
from Python.lib.agents.learners.policies import LeaPolicyGradient

from Python.lib.agents.policies import PolicyTypes, random_walks
from Python.lib.agents.policies.job_assignment import PolJobAssignmentProbabilistic
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep

from Python.lib.agents.queues import AgeQueue

from Python.lib.environments import gridworlds
from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses, rewardOnJobRejection_ExponentialCost
import Python.lib.queues as queues
from Python.lib.queues import Event

from Python.lib.simulators import LearningMethod
from Python.lib.simulators.discrete import Simulator as DiscreteSimulator
from Python.lib.simulators.queues import compute_job_rates_by_server, generate_event, LearningMode, SimulatorQueue

from Python.lib.utils.basic import show_exec_params


class Test_Simulator(unittest.TestCase):
    # Note: nice explanation about the three types of methods that can be defined in Python: instance, class, static
    # https://stackoverflow.com/questions/54264073/what-is-the-use-and-when-to-use-classmethod-in-python
    # See the only answer by Navy Cheng.

    @classmethod
    def setUpClass(cls):
        # The environment
        cls.env = gridworlds.EnvGridworld1D(length=21)

        # Plotting parameters
        cls.colormap = cm.get_cmap("jet")

    def test_Gridworld1D_TDLambda_random_walk(self):
        """This test intends to reproduce the results in Sutton 2018 on the TD(lambda) algorithm applied to learn
        the state value function in a 1D gridworld
        """
        print("\n")
        print("Running test {}...".format(self.id()))

        # Possible policies and learners for agents
        pol_rw = random_walks.PolRandomWalkDiscrete(self.env)
        lea_td = td.LeaTDLambda(self.env)

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
                sim = DiscreteSimulator(self.env, agent, seed=seed, debug=debug)
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
            ylabel = "Average RMSE over all {} states, first {} episodes, and {} experiments".format(self.env.getNumStates(), nepisodes, nexperiments)

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


class Test_SimulatorQueue(unittest.TestCase):
    # Note: nice explanation about the three types of methods that can be defined in Python: instance, class, static
    # https://stackoverflow.com/questions/54264073/what-is-the-use-and-when-to-use-classmethod-in-python
    # See the only answer by Navy Cheng.

    @classmethod
    def setUpClass(cls):
        # Define the queue environment
        capacity = np.Inf
        nservers = 1
        job_class_rates = [0.7]
        service_rates = [1.0]
        job_rates_by_server = job_class_rates
        queue = queues.QueueMM(job_rates_by_server, service_rates, nservers, capacity)
        cls.env_queue_mm = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, rewardOnJobRejection_ExponentialCost, None)

    @classmethod
    def createAgentWithPolicyGradientLearner(cls, learning_method, dict_params_simul):
        "Creates an agent interacting with the queue system using a parameterized acceptance policy that is learned with policy gradient"

        # Acceptance policy definition
        policies = dict({PolicyTypes.ACCEPT: PolQueueTwoActionsLinearStep(cls.env_queue_mm, dict_params_simul['theta_start']),
                         PolicyTypes.ASSIGN: None})

        # Define the Learners (of the value function and the policy)
        if learning_method == LearningMethod.FV:
            learnerV = LeaFV(cls.env_queue_mm, gamma=1.0)
        else:
            learnerV = LeaMC(cls.env_queue_mm, gamma=1.0)
        learners = dict({LearnerTypes.V: learnerV,
                         LearnerTypes.Q: None,
                         LearnerTypes.P: LeaPolicyGradient(cls.env_queue_mm, policies[PolicyTypes.ACCEPT], learnerV, alpha=1.0)})
        agent_gradient = AgeQueue(cls.env_queue_mm, policies, learners)

        return agent_gradient

    @classmethod
    def runSimulation(cls, learning_method, agent, dict_params_simul, dict_params_learn, dict_params_info):
        "Runs the simulation that learns the optimum parameterized job acceptance policy for the queue"
        # Simulation object
        simul = SimulatorQueue(cls.env_queue_mm, agent, dict_params_learn, N=dict_params_simul['nparticles'],
                               log=False, save=False, debug=False)

        params = dict({
            '1(a)-System-#Servers': cls.env_queue_mm.getNumServers(),
            '1(b)-System-JobClassRates': cls.env_queue_mm.getJobClassRates(),
            '1(c)-System-ServiceRates': cls.env_queue_mm.getServiceRates(),
            '1(d)-System-TrueTheta': dict_params_simul['theta_true'],
            '2(a)-Learning-Method': learning_method.name,
            '2(b)-Learning-Method#Particles': dict_params_simul['nparticles'],
            '2(c)-Learning-GradientEstimation': dict_params_learn['mode'].name,
            '2(d)-Learning-#Steps': dict_params_learn['t_learn'],
            '2(e)-Learning-SimulationTimePerLearningStep': dict_params_simul['t_sim'],
        })
        show_exec_params(params)
        # Set the initial theta value
        simul.getLearnerP().getPolicy().setThetaParameter(dict_params_simul['theta_start'])
        _, _, df_learning = simul.run(dict_params_simul, dict_params_info=dict_params_info, seed=dict_params_simul['seed'], verbose=False)

        return df_learning

    def test_generate_event(self):
        print("\n")
        print("".join(np.repeat("*", 20)))
        print("Testing generate_event() method defined in simulators.queues...")

        # Setup the simulator with the minimum required information
        capacity = 10
        nservers = 4
        job_class_rates = [0.7, 0.5]
        service_rates = [1.0, 0.8, 1.3, 0.75]

        # Job class to server assignment probabilities so that we can compute the job rates by server (needed to create the QueueMM object)
        policy_assignment_probabilities = [[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.4, 0.1]]
        policy_assign = PolJobAssignmentProbabilistic(policy_assignment_probabilities)
        job_rates_by_server = compute_job_rates_by_server(job_class_rates, nservers, policy_assignment_probabilities)

        # Queue M/M/nservers/capacity object
        queue = queues.QueueMM(job_rates_by_server, service_rates, nservers, capacity)
        env_queue_mm = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, None, None)

        # The simulation object
        simul = SimulatorQueue(env_queue_mm, None, None, N=1)

        np.random.seed(1717)
        time, event, job_class_or_server, idx_env = generate_event(simul.envs)
        print("Observed minimum time {} for an {} associated to job class (if BIRTH) or server (if DEATH) = {}".format(time, event, job_class_or_server))
        time_expected, event_expected, job_class_or_server_expected, idx_env_expected = 0.36603, Event.BIRTH, 0, 0
        assert np.allclose(time, time_expected, atol=1E-5)
        assert event == event_expected
        assert job_class_or_server == job_class_or_server_expected
        assert idx_env == idx_env_expected

        # Repeat the realization to check that we always get a BIRTH event because by default the queue sizes of the servers are initialized at 0
        N = 10
        for i in range(N):
            print("Generating event number {} out of {}...".format(i+1, N))
            _, event, _, _ = generate_event(simul.envs)
            assert event == Event.BIRTH

        #-- Repeat the realization to check the distribution of event types and selected job classes or servers
        # We set a random seed so that we embrace many more test cases as every time we run this test we get new realizations!
        np.random.seed(None)
        # Set two of the servers to a positive queue size
        simul.envs[0].setState(([0, 1, 0, 2], None))

        # Keep track of the number of events observed for each rate (either a job class rate or a service rate)
        rates = job_class_rates + service_rates
        valid_rates = job_class_rates + [r if s > 0 else np.nan for r, s in zip(simul.envs[0].getServiceRates(), simul.envs[0].getQueueState())]
        n_events_by_rate = np.zeros(len(rates))
        N = 100
        # Expected proportion of events for each exponential
        p = np.array( [r / np.nansum(valid_rates) if r > 0 else 0.0 for r in valid_rates] )
        for _ in range(N):
            time, event, job_class_or_server, _ = generate_event(simul.envs)
            if event == Event.BIRTH:
                # The BIRTH events are associated to indices at the beginning of the rates list
                n_events_by_rate[job_class_or_server] += 1
            elif event == Event.DEATH:
                # The DEATH event are associated to indices at the end of the rates list
                # so we sum the returned index to the number of job class rates.
                n_events_by_rate[len(job_class_rates) + job_class_or_server] += 1
        # Observed proportions of events by server and type of event
        phat = 1.0 * n_events_by_rate / N
        se_phat = np.sqrt(p * (1 - p) / N)
        print("EXPECTED / OBSERVED / SE proportions of events by rate on N={}:\n{}".format(N, np.c_[p, phat, se_phat]))
        print("NOTE: If the test fails, run it again, because it may have failed just by chance due to the random seed chosen when running the test.")
        assert np.allclose(phat, p, atol=3 * se_phat)  # true probabilities should be contained in +/- 3 SE(phat) from phat
        # ---------------- generate_event() in simulators.queues ----------------- #

    def test_FVRL_single_server_IGA(self):
        unittest.skip("The current implementation (11-Jul-2022) is NOT prepared for IGA because integer theta values make Pr(K-1) be equal to None.\n" \
                "Correcting this is not so easy and we are not interested in evaluating IGA right now")
        print("\n")
        print("".join(np.repeat("*", 20)))
        print("Testing the FVRL algorithm on a single server system using the IGA learning strategy...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'g.-'})

        # Simulation parameters
        learning_method = LearningMethod.FV
        dict_params_learn = dict({'mode': LearningMode.IGA, 't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': 19,
            'theta_start': 1.0,
            'nparticles': 30,
            't_sim': 50,
            'buffer_size_activation_factor': 0.3
            })

        # Agent interacting with the environment
        agent_gradient = self.createAgentWithPolicyGradientLearner(learning_method, dict_params_simul)

        # Run the simulation!
        df_learning = self.runSimulation(learning_method, agent_gradient, dict_params_simul, dict_params_learn, dict_params_info)

        # Assert
        print(df_learning)
        # Expected result when using FVRL with IGA strategy
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [1.0, 2.0, 3.0, 4.0, 5.0]),
                                                ('theta_next', [2.0, 3.0, 4.0, 5.0, 5.0]),
                                                ('Pr(K-1)', [0.269096, 0.185156, 0.167206, 0.039376, 0.062921]),
                                                ('Pr(K)', [0.205425, 0.094288, 0.112091, 0.015120, 0.047722]),
                                                ('Q_diff(K-1)', [0.48, 0.72, 0.40, 0.28, 0.17]),
                                                ('Q_diff(K)', [0.71, 0.50, 0.73, 0.18, -0.23]),
                                                ('alpha', [1.0]*5),
                                                ('V', [-1.350, -1.290, -1.330, -1.660, -1.405]),
                                                ('gradV', [0.129166, 0.133313, 0.066882, 0.011025, 0.010697]),
                                                ('n_events_mc', [93, 97, 90, 99, 90]),
                                                ('n_events_fv', [470, 409, 729, 416, 804]),
                                                ('n_trajectories_Q', [100.0]*5)
                                                ])
        assert np.allclose(df_learning, df_learning_expected, atol=1E-6)

    def test_FVRL_single_server_REINFORCE_TRUE(self):
        print("\n")
        print("".join(np.repeat("*", 20)))
        print("Testing the FVRL algorithm on a single server system using the REINFORCE_TRUE learning strategy...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'g.-'})

        # Simulation parameters
        learning_method = LearningMethod.FV
        dict_params_learn = dict({'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': 19,
            'theta_start': 1.1,     # This number should NOT be integer, o.w. the estimated Pr(K-1) will always be 0
            'nparticles': 30,
            't_sim': 50,
            'buffer_size_activation_factor': 0.3
            })

        # Agent interacting with the environment
        agent_gradient = self.createAgentWithPolicyGradientLearner(learning_method, dict_params_simul)

        # Run the simulation!
        df_learning = self.runSimulation(learning_method, agent_gradient, dict_params_simul, dict_params_learn, dict_params_info)
        print(df_learning)
        # Expected result when using FVRL with REINFORCE_TRUE strategy
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [1.1, 1.223813, 1.458335, 2.042627, 2.199874]),
                                                ('theta_next', [1.223813, 1.458335, 2.042627, 2.199874, 2.312675]),
                                                ('Pr(K-1)', [0.137570, 0.187618, 0.229134, 0.101449, 0.150402]),
                                                ('Pr(K)', [0.0]*5),
                                                ('Q_diff(K-1)', [0.90, 1.25, 2.55, 1.55, 0.75]),
                                                ('Q_diff(K)', [0.0]*5),
                                                ('alpha', [1.0]*5),
                                                ('V', [-9.000, -10.775, -8.925, -9.675, -9.725]),
                                                ('gradV', [0.123813, 0.234522, 0.584292, 0.157247, 0.112802]),
                                                ('n_events_mc', [95, 92, 88, 97, 88]),
                                                ('n_events_fv', [179, 235, 300, 454, 674]),
                                                ('n_trajectories_Q', [100.0]*5)
                                                ])
        assert np.allclose(df_learning, df_learning_expected, atol=1E-6)

    def test_MC_single_server_IGA(self):
        unittest.skip("The current implementation (11-Jul-2022) is NOT prepared for IGA because integer theta values make Pr(K-1) be equal to None.\n" \
                "Correcting this is not so easy and we are not interested in evaluating IGA right now")
        print("\n")
        print("".join(np.repeat("*", 20)))
        print("Testing the MC algorithm on a single server system using the IGA learning strategy...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'r.-'})

        # Simulation parameters
        learning_method = LearningMethod.MC
        dict_params_learn = dict({'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': 19,
            'theta_start': 1.0,
            'nparticles': 1,
            't_sim': 30 * 50,
            'buffer_size_activation_factor': 0.3
            })

        # Agent interacting with the environment
        agent_gradient = self.createAgentWithPolicyGradientLearner(learning_method, dict_params_simul)

        # Run the simulation!
        df_learning = self.runSimulation(learning_method, agent_gradient, dict_params_simul, dict_params_learn, dict_params_info)
        print(df_learning)
        # Expected result when using MC with IGA strategy
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [1.0, 2.0, 3.0, 4.0, 5.0]),
                                                ('theta_next', [2.0, 3.0, 4.0, 5.0, 6.0]),
                                                ('Pr(K-1)', [0.611731, 0.188644, 0.148607, 0.086340, 0.051746]),
                                                ('Pr(K)', [0.192246, 0.111336, 0.123782, 0.072948, 0.028608]),
                                                ('Q_diff(K-1)', [0.39, 0.44, 0.43, 0.32, 0.41]),
                                                ('Q_diff(K)', [0.42, 0.15, 0.63, 0.07, 0.93]),
                                                ('alpha', [1.0]*5),
                                                ('V', [-1.325, -1.500, -1.295, -1.360, -1.605]),
                                                ('gradV', [0.238575, 0.083003, 0.063901, 0.027629, 0.021216]),
                                                ('n_events_mc', [1500, 1500, 1500, 1500, 1500]),
                                                ('n_events_fv', [0]*5),
                                                ('n_trajectories_Q', [100.0]*5)
                                                ])
        assert np.allclose(df_learning, df_learning_expected, atol=1E-6)

    def test_MC_single_server_REINFORCE_TRUE(self):
        print("\n")
        print("".join(np.repeat("*", 20)))
        print("Testing the MC algorithm on a single server system using the REINFORCE_TRUE learning strategy...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'r.-'})

        # Simulation parameters
        learning_method = LearningMethod.MC
        dict_params_learn = dict({'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': 19,
            'theta_start': 1.1,     # This number should NOT be integer, o.w. the estimated Pr(K-1) will always be 0
            'nparticles': 1,
            't_sim': 30 * 50,
            'buffer_size_activation_factor': 0.3
            })

        # Agent interacting with the environment
        agent_gradient = self.createAgentWithPolicyGradientLearner(learning_method, dict_params_simul)

        # Run the simulation!
        df_learning = self.runSimulation(learning_method, agent_gradient, dict_params_simul, dict_params_learn, dict_params_info)
        print(df_learning)
        # Expected result when using MC with REINFORCE_TRUE strategy
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [1.1, 1.268038, 1.629405, 1.826506, 2.009404]),
                                                ('theta_next', [1.268038, 1.629405, 1.826506, 2.009404, 2.087338]),
                                                ('Pr(K-1)', [0.186709, 0.212569, 0.219001, 0.203220, 0.103911]),
                                                ('Pr(K)', [0.0]*5),
                                                ('Q_diff(K-1)', [0.90, 1.70, 0.90, 0.90, 0.75]),
                                                ('Q_diff(K)', [0.0]*5),
                                                ('alpha', [1.0]*5),
                                                ('V', [-9.0, -10.0, -7.40, -8.0, -10.225]),
                                                ('gradV', [0.168038, 0.361367, 0.197101, 0.182898, 0.077933]),
                                                ('n_events_mc', [1500, 1500, 1500, 1500, 1500]),
                                                ('n_events_fv', [0]*5),
                                                ('n_trajectories_Q', [100.0]*5)
                                                ])
        assert np.allclose(df_learning, df_learning_expected, atol=1E-6)


if __name__ == "__main__":
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
    runner = unittest.TextTestRunner()

    # Run all tests
    # unittest.main()

    # Create the test suite
    # NOTE: IGA tests are skipped because the code is currently (11-Jul-2022) not prepared to properly handle the case
    # with integer-valued theta, and preparing it would introduce a little bit of noise in the definition of functions
    # (e.g. passing the learning mode (e.g. IGA or REINFORCE_TRUE) to functions such as
    # estimate_blocking_probability_fv() and similar.
    test_suite_simulator = unittest.TestSuite()
    test_suite_simulator.addTest(Test_Simulator("test_Gridworld1D_TDLambda_random_walk"))

    test_suite_simulator_queue = unittest.TestSuite()
    test_suite_simulator_queue.addTest(Test_SimulatorQueue("test_generate_event"))
    #test_suite_simulator_queue.addTest(Test_SimulatorQueue("test_FVRL_single_server_IGA"))
    test_suite_simulator_queue.addTest(Test_SimulatorQueue("test_FVRL_single_server_REINFORCE_TRUE"))
    #test_suite_simulator_queue.addTest(Test_SimulatorQueue("test_MC_single_server_IGA"))
    test_suite_simulator_queue.addTest(Test_SimulatorQueue("test_MC_single_server_REINFORCE_TRUE"))

    # Run the test suites
    runner.run(test_suite_simulator)
    runner.run(test_suite_simulator_queue)
