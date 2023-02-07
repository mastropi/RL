# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 15:26:38 2022

@author: Daniel Mastropietro
@description: Unit tests for optimizers (e.g. optimum policy) on continuous-time MDPs.
@details: Naming conventions follow the instructions given in test_conventions.txt.
"""

import runpy
runpy.run_path('../../setup.py')

import unittest
from unittest_data_provider import data_provider

import numpy as np
import pandas as pd

from Python.lib.agents.learners import LearnerTypes
from Python.lib.agents.learners.continuing.fv import LeaFV
from Python.lib.agents.learners.continuing.mc import LeaMC
from Python.lib.agents.learners.policies import LeaPolicyGradient

from Python.lib.agents.policies import PolicyTypes
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep

from Python.lib.agents.queues import AgeQueue

from Python.lib.environments import queues as env_queues
from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses, rewardOnJobRejection_Constant, rewardOnJobRejection_ExponentialCost
import Python.lib.queues as queues

from Python.lib.simulators import LearningMethod
from Python.lib.simulators.queues import LearningMode, SimulatorQueue, SurvivalProbabilityEstimation

from Python.lib.utils.basic import is_scalar, show_exec_params


class Test_EstPolicy_EnvQueueSingleServer(unittest.TestCase):
    # Note: nice explanation about the three types of methods that can be defined in Python: instance, class, static
    # https://stackoverflow.com/questions/54264073/what-is-the-use-and-when-to-use-classmethod-in-python
    # See the only answer by Navy Cheng.

    # TODO: (2023/02/02) Setup the class by calling function define_queue_environment_and_agent() defined in simulators/__init__.py which takes care of everything I do here, and in addition we initialize objects in a consistent manner!
    @classmethod
    def setUpClass(cls):
        # Define the queue environment
        capacity = np.Inf
        nservers = 1
        job_class_rates = [0.7]
        service_rates = [1.0]
        job_rates_by_server = job_class_rates
        queue = queues.QueueMM(job_rates_by_server, service_rates, nservers, capacity)
        cls.env_queue_mm_singleserver = EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, rewardOnJobRejection_ExponentialCost, None)

    @classmethod
    def createAgentWithPolicyGradientLearner(cls, dict_params_simul, dict_params_learn):
        "Creates an agent interacting with the queue system using a parameterized acceptance policy that is learned with policy gradient"

        # Acceptance policy definition
        policies = dict({PolicyTypes.ACCEPT: PolQueueTwoActionsLinearStep(cls.env_queue_mm_singleserver, dict_params_simul['theta_start'])
                                                if is_scalar(dict_params_simul['theta_start'])
                                                else [PolQueueTwoActionsLinearStep(cls.env_queue_mm_singleserver, theta_start) for theta_start in dict_params_simul['theta_start']],
                         PolicyTypes.ASSIGN: None})

        # Define the Learners (of the value function and the policy)
        if dict_params_learn['method'] == LearningMethod.FV:
            learnerV = LeaFV(cls.env_queue_mm_singleserver, gamma=1.0)
        else:
            learnerV = LeaMC(cls.env_queue_mm_singleserver, gamma=1.0)
        learners = dict({LearnerTypes.V: learnerV,
                         LearnerTypes.Q: None,
                         LearnerTypes.P: LeaPolicyGradient(cls.env_queue_mm_singleserver, policies[PolicyTypes.ACCEPT], learnerV,
                                                           alpha=dict_params_learn['alpha_start'],
                                                           adjust_alpha=dict_params_learn['adjust_alpha'], func_adjust_alpha=dict_params_learn.get('func_adjust_alpha', np.float),
                                                           min_time_to_update_alpha=dict_params_learn.get('min_time_to_update_alpha', 0), alpha_min=dict_params_learn.get('alpha_min', 0.0))
                         })
        agent_gradient = AgeQueue(cls.env_queue_mm_singleserver, policies, learners)

        return agent_gradient

    @classmethod
    def runSimulation(cls, agent, dict_params_simul, dict_params_learn, dict_params_info):
        "Runs the simulation that learns the optimum parameterized job acceptance policy for the queue"
        # Simulation object
        simul = SimulatorQueue(cls.env_queue_mm_singleserver, agent, dict_params_learn, N=dict_params_simul['nparticles'],
                               log=False, save=False, debug=False)

        params = dict({
            '0(a)-System': "Single-server queue system",
            '1(a)-System-#Servers': cls.env_queue_mm_singleserver.getNumServers(),
            '1(b)-System-JobClassRates': cls.env_queue_mm_singleserver.getJobClassRates(),
            '1(c)-System-ServiceRates': cls.env_queue_mm_singleserver.getServiceRates(),
            '1(d)-System-TrueTheta': dict_params_simul['theta_true'],
            '2(a)-Learning-Method': dict_params_learn['method'].name,
            '2(b)-Learning-Method#Particles': dict_params_simul['nparticles'],
            '2(c)-Learning-Method#TimeSteps/ArrivalEvents': dict_params_simul['t_sim'],
            '2(d)-Learning-Method#BurnInSteps (BITS)': dict_params_simul['burnin_time_steps'],
            '2(e)-Learning-Method#MinNumCycles': dict_params_simul['min_num_cycles_for_expectations'],
            '2(f)-Learning-MethodThetaStart': dict_params_simul['theta_start'],
            '2(g)-Learning-GradientEstimation': dict_params_learn['mode'].name,
            '2(h)-Learning-#Steps': dict_params_learn['t_learn'],
            '2(i)-Learning-AlphaStart': dict_params_learn['alpha_start'],
            '2(j)-Learning-AdjustAlpha?': dict_params_learn['adjust_alpha'],
            '2(k)-Learning-AdjustAlphaFunction': dict_params_learn.get('func_adjust_alpha', np.float),
            '2(l)-Learning-MinEpisodeToAdjustAlpha': dict_params_learn.get('min_time_to_update_alpha', 0),
            '2(m)-Learning-AlphaMin': dict_params_learn.get('alpha_min', 0),
            '2(n)-Learning-Seed': dict_params_simul['seed'],
        })
        show_exec_params(params)
        _, _, df_learning = simul.run(dict_params_simul, dict_params_info=dict_params_info, seed=dict_params_simul['seed'], verbose=False)

        return df_learning

    def test_Env_MetFVRLwithReinforceTrue(self):
        print("\n")
        print("".join(np.repeat("*", 20)))
        print("Testing the FVRL algorithm on a single server system using the REINFORCE_TRUE learning strategy...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'g.-'})

        # Simulation parameters
        dict_params_learn = dict({  'method': LearningMethod.FV,
                                    'alpha_start': 10.0, 'adjust_alpha': False,
                                    'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': 19,
            'theta_start': 1.1,     # This number should NOT be integer, o.w. the estimated Pr(K-1) will always be 0
            'nparticles': 30,
            't_sim': 50,
            'buffer_size_activation_factor': 0.3,
            'burnin_time_steps': 20,
            'min_num_cycles_for_expectations': 5,
            })
        # Make a copy of the seed used to run the process (for the assertion below),
        # because the key in dict_params_simul is changed by the process
        # as we need to consider different seeds for the different simulation parts and this seed is passed to each
        # simulator through the dict_params_simul dictionary and is used to store its value in the output files
        # to allow repeatability.
        seed = dict_params_simul['seed']

        # Agent interacting with the environment
        agent_gradient = self.createAgentWithPolicyGradientLearner(dict_params_simul, dict_params_learn)

        # Run the simulation!
        df_learning = self.runSimulation(agent_gradient, dict_params_simul, dict_params_learn, dict_params_info)
        print(df_learning)
        # Expected result when using FVRL with REINFORCE_TRUE strategy
        # (2023/02/02) Today these results have been carefully copied from the observed result of the test and checked that they are verified!
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [1.1, 8.484077, 8.484077, 8.484070, 8.519243]),
                                                ('theta_next', [8.484077, 8.484077, 8.484070, 8.519243, 8.785301]),
                                                ('Pr(K-1)', [0.289572, 0.0000, 0.013527, 0.001327, 0.017738]),
                                                ('Pr(K)', [0.0]*5),
                                                ('Error(K-1)', [0.294208, -1.0, 0.085777, -0.893459, 0.423790]),
                                                ('Error(K)', [-1.0]*5),
                                                ('Q_diff(K-1)', [2.55, 0.00, -0.000050, 2.649980, 1.499969]),
                                                ('Q_diff(K)', [0.0]*5),
                                                ('alpha', [10.0]*5),
                                                ('gradV', [0.738408, 0.0, -6.7195E-7, 0.0035173, 0.026606]),
                                                ('n_events_mc', [87, 100, 99, 99, 99]),
                                                ('n_events_fv', [431, 0, 793, 511, 2353]),
                                                ('n_trajectories_Q', [100.0, 0.0, 100.0, 100.0, 100.0])
                                                ])

        assert  dict_params_learn['mode'] == LearningMode.REINFORCE_TRUE and \
                dict_params_learn['t_learn'] == 5
        assert  seed == 1717 and \
                dict_params_simul['theta_true'] == 19 and \
                dict_params_simul['theta_start'] == 1.1 and \
                dict_params_simul['nparticles'] == 30 and \
                dict_params_simul['t_sim'] == 50 and \
                dict_params_simul['buffer_size_activation_factor'] == 0.3

        assert np.allclose(df_learning, df_learning_expected, atol=1E-6)

    # -------- DATA -------
    # Case number, description, parameters, expected value
    # These are the same tests 1, 2 and 3 from data_test_lambda_return_random_walk
    data_test_Env_MetFVRL = lambda DEFAULT_EXECUTION = True: (
        (1, DEFAULT_EXECUTION, '1D-theta as scalar',
        {'seed': 1313, 'theta_true': 5.0, 'theta_start': 2.1, 'nparticles': 50, 't_sim': 100, 'buffer_size_activation_factor': 0.5,
         'burnin_time_steps': 20, 'min_num_cycles_for_expectations': 5, 'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_N_PARTICLES},
        {'method': LearningMethod.FV,
         'alpha_start': 10.0, 'adjust_alpha': True, 'func_adjust_alpha': np.float, 'min_time_to_update_alpha': 0, 'alpha_min': 0.01,
         'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5},
         # In the following "expected" dictionary we should include the value of columns in the df_learning output yielded by self.runSimulation() that we would like to test
         {'theta_next': [2.59211708, 6.0807596, 5.08841629, 4.22018097, 4.6207499],
          'Pr(K-1)': [0.154502, 0.105127, 0.019458, 0.044148, 0.076299],
          'Q_diff(K-1)': [0.318519, 3.318519, -10.2, -5.9, 2.1]}),

        (2, DEFAULT_EXECUTION, '1D-theta as list',
         {'seed': 1313, 'theta_true': [5.0], 'theta_start': [2.1], 'nparticles': 50, 't_sim': 100,
          'buffer_size_activation_factor': 0.5,
          'burnin_time_steps': 20, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_N_PARTICLES},
         {'method': LearningMethod.FV,
          'alpha_start': 10.0, 'adjust_alpha': True, 'func_adjust_alpha': np.float, 'min_time_to_update_alpha': 0,
          'alpha_min': 0.01,
          'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5},
         # In the following "expected" dictionary we should include the value of columns in the df_learning output yielded by self.runSimulation() that we would like to test
         {'theta_next': [2.59211708, 6.0807596, 5.08841629, 4.22018097, 4.6207499],
          'Pr(K-1)': [0.154502, 0.105127, 0.019458, 0.044148, 0.076299],
          'Q_diff(K-1)': [0.318519, 3.318519, -10.2, -5.9, 2.1]}),
    )
    # -------- DATA -------

    @data_provider(data_test_Env_MetFVRL)
    def test_EnvQueueSingleServer_MetFVRL_TestSeveralCases(self, casenum, run, desc, dict_params_simul, dict_params_learn, dict_expected):
        "Test the FVRL implementation that learns the optimum parameter theta of a parameterized policy on the single-server queue system"
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'r.-'})
        assert dict_params_learn['method'] == LearningMethod.FV
        if run:
            print("\n*** Testing {}, case number {}: '{}' ***".format(self.id(), casenum, desc))

            # Agent interacting with the environment
            agent_gradient = self.createAgentWithPolicyGradientLearner(dict_params_simul, dict_params_learn)

            # Run the simulation!
            df_learning = self.runSimulation(agent_gradient, dict_params_simul, dict_params_learn, dict_params_info)
            print(df_learning)

            for column, expected_value in dict_expected.items():
                assert np.allclose(df_learning[column], expected_value, atol=1E-6)

    def test_Env_MetMCwithReinforceTrue(self):
        print("\n")
        print("".join(np.repeat("*", 20)))
        print("Testing the MC algorithm on a single server system using the REINFORCE_TRUE learning strategy...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'r.-'})

        # Simulation parameters
        dict_params_learn = dict({'method': LearningMethod.MC,
                                  'alpha_start': 10.0, 'adjust_alpha': False,
                                  'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': 19,
            'theta_start': 1.1,     # This number should NOT be integer, o.w. the estimated Pr(K-1) will always be 0
            'nparticles': 1,
            't_sim': 30 * 50,
            'buffer_size_activation_factor': 0.3,
            'burnin_time_steps': 20,
            'min_num_cycles_for_expectations': 5,
            })
        # Make a copy of the seed used to run the process (for the assertion below),
        # because the key in dict_params_simul is changed by the process
        # as we need to consider different seeds for the different simulation parts and this seed is passed to each
        # simulator through the dict_params_simul dictionary and is used to store its value in the output files
        # to allow repeatability.
        seed = dict_params_simul['seed']

        # Agent interacting with the environment
        agent_gradient = self.createAgentWithPolicyGradientLearner(dict_params_simul, dict_params_learn)

        # Run the simulation!
        df_learning = self.runSimulation(agent_gradient, dict_params_simul, dict_params_learn, dict_params_info)
        print(df_learning)
        # Expected result when using MC with REINFORCE_TRUE strategy
        # (2023/02/02) Today these results have been carefully copied from the observed result of the test and checked that they are verified!
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [1.1, 6.450352, 6.685442, 6.792059, 6.761383]),
                                                ('theta_next', [6.450352, 6.685442, 6.792059, 6.761383, 7.615656]),
                                                ('Pr(K-1)', [0.209818, 0.015167, 0.053310, 0.010225, 0.031064]),
                                                ('Pr(K)', [0.0]*5),
                                                ('Error(K-1)', [-0.062243, -0.421494, 1.033354, -0.609999, 0.184868]),
                                                ('Error(K)', [-1.0]*5),
                                                ('Q_diff(K-1)', [2.55, 1.549996, 0.199995, -0.300008, 2.749998]),
                                                ('Q_diff(K)', [0.0]*5),
                                                ('alpha', [10.0]*5),
                                                ('gradV', [0.535035, 0.023509, 0.010662, -0.003068, 0.085427]),
                                                ('n_events_mc', [1500, 1500, 1500, 1500, 1500]),
                                                ('n_events_fv', [0]*5),
                                                ('n_trajectories_Q', [100.0]*5)
                                                ])

        assert  dict_params_learn['mode'] == LearningMode.REINFORCE_TRUE and \
                dict_params_learn['t_learn'] == 5
        assert  seed == 1717 and \
                dict_params_simul['theta_true'] == 19 and \
                dict_params_simul['theta_start'] == 1.1 and \
                dict_params_simul['nparticles'] == 1 and \
                dict_params_simul['t_sim'] == 30 * 50 and \
                dict_params_simul['buffer_size_activation_factor'] == 0.3 and \
                dict_params_simul['burnin_time_steps'] == 20 and \
                dict_params_simul['min_num_cycles_for_expectations'] == 5

        assert np.allclose(df_learning, df_learning_expected, atol=1E-6)


class Test_EstPolicy_EnvQueueLossNetworkWithJobClasses(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rates_birth = [0.1, 0.5, 0.8]       # Let's use very different arrival rates so that we can see differences in the optimum theta values found...
        rates_death = [1.0, 1.0, 1.0]
        nservers = len(rates_death)
        # Capacity of the loss network
        K = 10
        queue_mm_loss = queues.QueueMM(rates_birth, rates_death, nservers, K, origin=0.0)
        reward_func = rewardOnJobRejection_Constant
        rewards_accept_by_job_class = [0.0] * len(rates_birth)
        dict_params_reward_func = None
        cls.env_queue_mm_loss = env_queues.EnvQueueLossNetworkWithJobClasses(
            queue_mm_loss,
            reward_func,
            rewards_accept_by_job_class,
            dict_params_reward_func)

    @classmethod
    def createAgentWithPolicyGradientLearner(cls, dict_params_simul, dict_params_learn):
        "Creates an agent interacting with the queue system using a parameterized acceptance policy that is learned with policy gradient"

        # Acceptance policy definition
        policies = dict({PolicyTypes.ACCEPT: [PolQueueTwoActionsLinearStep(cls.env_queue_mm_loss, theta_start) for theta_start in dict_params_simul['theta_start']],
                         PolicyTypes.ASSIGN: None})

        # Define the Learners (of the value function and the policy)
        if dict_params_learn['method'] == LearningMethod.FV:
            learnerV = LeaFV(cls.env_queue_mm_loss, gamma=1.0)
        else:
            learnerV = LeaMC(cls.env_queue_mm_loss, gamma=1.0)
        learners = dict({LearnerTypes.V: learnerV,
                         LearnerTypes.Q: None,
                         LearnerTypes.P: LeaPolicyGradient(cls.env_queue_mm_loss, policies[PolicyTypes.ACCEPT], learnerV, alpha=10.0)})
        agent_gradient = AgeQueue(cls.env_queue_mm_loss, policies, learners)

        return agent_gradient

    @classmethod
    def runSimulation(cls, agent, dict_params_simul, dict_params_learn, dict_params_info):
        "Runs the simulation that learns the optimum parameterized job acceptance policy for the queue"
        # Simulation object
        simul = SimulatorQueue(cls.env_queue_mm_loss, agent, dict_params_learn, N=dict_params_simul['nparticles'],
                               log=False, save=False, debug=False)

        params = dict({
            '0(a)-System': "Loss Network receiving jobs of different classes",
            '1(a)-System-#Servers': cls.env_queue_mm_loss.getCapacity(),
            '1(b)-System-JobClassRates': cls.env_queue_mm_loss.getJobClassRates(),
            '1(c)-System-ServiceRates': cls.env_queue_mm_loss.getServiceRates(),
            '1(d)-System-TrueTheta': dict_params_simul['theta_true'],
            '2(a)-Learning-Method': dict_params_learn['method'].name,
            '2(b)-Learning-Method#Particles': dict_params_simul['nparticles'],
            '2(c)-Learning-Method#TimeSteps/ArrivalEvents': dict_params_simul['t_sim'],
            '2(d)-Learning-Method#BurnInSteps (BITS)': dict_params_simul['burnin_time_steps'],
            '2(e)-Learning-Method#MinNumCycles': dict_params_simul['min_num_cycles_for_expectations'],
            '2(f)-Learning-MethodThetaStart': dict_params_simul['theta_start'],
            '2(g)-Learning-GradientEstimation': dict_params_learn['mode'].name,
            '2(h)-Learning-#Steps': dict_params_learn['t_learn'],
            '2(i)-Learning-AlphaStart': dict_params_learn['alpha_start'],
            '2(j)-Learning-AdjustAlpha?': dict_params_learn['adjust_alpha'],
            '2(k)-Learning-AdjustAlphaFunction': dict_params_learn.get('func_adjust_alpha', np.float),
            '2(l)-Learning-MinEpisodeToAdjustAlpha': dict_params_learn.get('min_time_to_update_alpha', 0),
            '2(m)-Learning-AlphaMin': dict_params_learn.get('alpha_min', 0),
            '2(n)-Learning-Seed': dict_params_simul['seed'],
        })
        show_exec_params(params)
        _, _, df_learning = simul.run(dict_params_simul, dict_params_info=dict_params_info, seed=dict_params_simul['seed'], verbose=False)

        return df_learning

    def test_Env_MetFVRLwithReinforceTrue(self):
        print("\n")
        print("".join(np.repeat("*", 20)))
        print("Testing the FVRL algorithm on a single server system using the REINFORCE_TRUE learning strategy...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'g.-'})

        # Simulation parameters
        dict_params_learn = dict({'method': LearningMethod.FV,
                                  'alpha_start': 10.0, 'adjust_alpha': True, 'func_adjust_alpha': np.float, 'min_time_to_update_alpha': 0, 'alpha_min': 0.01,
                                 'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': [4, 2, 5],            # These theta_true are NOT the actual true thetas!! In fact, we do NOT know the true values at this point.
            'theta_start': [7.1, 3.1, 0.1],     # IMPORTANT: The theta values in each component should NOT give a blocking size K(i) that is larger than the system's capacity K... o.w. the gradient of that component is always 0...
            'nparticles': 30, #300
            't_sim': 50, #100
            'buffer_size_activation_factor': [0.3, 0.3, 0.3],
            # For now the following two attributes  are set to 0 until I solve the TO-DO task in the estimate_blocking_fv() function in simulators/queues.py
            # as the process of setting the estimated probability to NaN does not work with the multidimensional theta case.
            'burnin_time_steps': 0,
            'min_num_cycles_for_expectations': 0,
            })
        # Make a copy of the seed used to run the process (for the assertion below),
        # because the key in dict_params_simul is changed by the process
        # as we need to consider different seeds for the different simulation parts and this seed is passed to each
        # simulator through the dict_params_simul dictionary and is used to store its value in the output files
        # to allow repeatability.
        seed = dict_params_simul['seed']

        # Agent interacting with the environment
        agent_gradient = self.createAgentWithPolicyGradientLearner(dict_params_simul, dict_params_learn)

        # Run the simulation!
        df_learning = self.runSimulation(agent_gradient, dict_params_simul, dict_params_learn, dict_params_info)
        print(df_learning)
        # [TBD] Expected result when using FVRL with REINFORCE_TRUE strategy
        # (2023/01/31) FOR NOW THESE ARE INVENTED RESULTS AS I HAVEN'T RUN IT SUCCESSFULLY YET
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [[2.1, 3.1, 0.1], [1.3, 2.3, 3.4], [1.1, 2.1, 3.1], [1.1, 2.1, 3.1], [1.1, 2.1, 3.1]]),
                                                ('theta_next', [[1.3, 2.3, 3.4], [1.1, 2.1, 3.1], [1.1, 2.1, 3.1], [1.1, 2.1, 3.1], [1.1, 2.1, 3.1]]),
                                                ('gradV', [ np.array([0.123813, 0.234522, 0.584292]),
                                                            np.array([0.2, 0.3, 0.5]),
                                                            np.array([0.3, -0.6, 0.8]),
                                                            np.array([0.1, 0.4, 0.9]),
                                                            np.array([-0.25, 0.33, 0.75])]),
                                                ('alpha', [1.0]*5),
                                                ('V', [-9.000, -10.775, -8.925, -9.675, -9.725]),
                                                ('n_events_mc', [95, 92, 88, 97, 88]),
                                                ('n_events_fv', [179, 235, 300, 454, 674])
                                                ])

        assert  dict_params_learn['mode'] == LearningMode.REINFORCE_TRUE and \
                dict_params_learn['t_learn'] == 5
        assert  seed == 1717 and \
                dict_params_simul['theta_true'] == [4, 2, 5] and \
                dict_params_simul['theta_start'] == [7.1, 3.1, 0.1] and \
                dict_params_simul['nparticles'] == 30 and \
                dict_params_simul['t_sim'] == 50 and \
                dict_params_simul['buffer_size_activation_factor'] == [0.3, 0.3, 0.3] and \
                dict_params_simul['burnin_time_steps'] == 0 and \
                dict_params_simul['min_num_cycles_for_expectations'] == 0
        #assert np.allclose(df_learning, df_learning_expected, atol=1E-6)

    def notyet_test_Env_MetMCwithReinforceTrue(self):
        print("\n")
        print("".join(np.repeat("*", 20)))
        print("Testing the MC algorithm on a single server system using the REINFORCE_TRUE learning strategy...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'r.-'})

        # Simulation parameters
        dict_params_learn = dict({  'method': LearningMethod.MC,
                                    'alpha_start': 10.0, 'adjust_alpha': True, 'func_adjust_alpha': np.float, 'min_time_to_update_alpha': 0, 'alpha_min': 0.01,
                                    'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': [4, 2, 5],
            'theta_start': [7.1, 3.1, 0.1],
            'nparticles': 1,
            't_sim': 30*50,
            'buffer_size_activation_factors': [0.3, 0.3, 0.3],
            'burnin_time_steps': 20,
            'min_num_cycles_for_expectations': 5,
            })
        # Make a copy of the seed used to run the process (for the assertion below),
        # because the key in dict_params_simul is changed by the process
        # as we need to consider different seeds for the different simulation parts and this seed is passed to each
        # simulator through the dict_params_simul dictionary and is used to store its value in the output files
        # to allow repeatability.
        seed = dict_params_simul['seed']

        # Agent interacting with the environment
        agent_gradient = self.createAgentWithPolicyGradientLearner(dict_params_simul, dict_params_learn)

        # Run the simulation!
        df_learning = self.runSimulation(agent_gradient, dict_params_simul, dict_params_learn, dict_params_info)
        print(df_learning)
        # [TBD] Expected result when using MC with REINFORCE_TRUE strategy
        # (2023/01/31) FOR NOW THESE ARE INVENTED RESULTS AS I HAVEN'T RUN IT SUCCESSFULLY YET
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [[1.1, 2.1, 3.1], [1.3, 2.3, 3.4], [1.1, 2.1, 3.1], [1.1, 2.1, 3.1], [1.1, 2.1, 3.1]]),
                                                ('theta_next', [[1.3, 2.3, 3.4], [1.1, 2.1, 3.1], [1.1, 2.1, 3.1], [1.1, 2.1, 3.1]]),
                                                ('gradV', [ np.array([0.123813, 0.234522, 0.584292]),
                                                            np.array([0.2, 0.3, 0.5]),
                                                            np.array([0.3, -0.6, 0.8]),
                                                            np.array([0.1, 0.4, 0.9]),
                                                            np.array([-0.25, 0.33, 0.75])]),
                                                ('alpha', [1.0]*5),
                                                ('V', [-9.000, -10.775, -8.925, -9.675, -9.725]),
                                                ('n_events_mc', [1500, 1500, 1500, 1500, 1500]),
                                                ('n_events_fv', [0]*5)
                                                ])

        assert  dict_params_learn['mode'] == LearningMode.REINFORCE_TRUE and \
                dict_params_learn['t_learn'] == 5
        assert  seed == 1717 and \
                dict_params_simul['theta_true'] == [4, 2, 5] and \
                dict_params_simul['theta_start'] == [2.1, 3.1, 0.1] and \
                dict_params_simul['nparticles'] == 1 and \
                dict_params_simul['t_sim'] == 30*50 and \
                dict_params_simul['buffer_size_activation_factors'] == [0.3, 0.3, 0.3] and \
                dict_params_simul['burnin_time_steps'] == 0 and \
                dict_params_simul['min_num_cycles_for_expectations'] == 0

        assert np.allclose(df_learning, df_learning_expected, atol=1E-6)



if __name__ == "__main__":
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
    runner = unittest.TextTestRunner()

    # Run all tests
    # unittest.main()

    #-- Create the test suites
    # 1) Single-server tests
    # NOTE: IGA tests are skipped because the code is currently (11-Jul-2022) not prepared to properly handle the case
    # with integer-valued theta, and preparing it would introduce a little bit of noise in the definition of functions
    # (e.g. passing the learning mode (e.g. IGA or REINFORCE_TRUE) to functions such as
    # estimate_blocking_probability_fv() and similar.
    test_suite_singleserver = unittest.TestSuite()
    test_suite_singleserver.addTest(Test_EstPolicy_EnvQueueSingleServer("test_Env_MetFVRLwithReinforceTrue"))
    test_suite_singleserver.addTest(Test_EstPolicy_EnvQueueSingleServer("test_EnvQueueSingleServer_MetFVRL_TestSeveralCases"))
    test_suite_singleserver.addTest(Test_EstPolicy_EnvQueueSingleServer("test_Env_MetMCwithReinforceTrue"))

    # 2) Loss network tests
    test_suite_lossnetwork = unittest.TestSuite()
    test_suite_lossnetwork.addTest(Test_EstPolicy_EnvQueueLossNetworkWithJobClasses("test_Env_MetFVRLwithReinforceTrue"))
    #test_suite_lossnetwork.addTest(Test_EstPolicy_EnvQueueLossNetworkWithJobClasses("test_Env_MetMCwithReinforceTrue"))

    #-- Run the test suites
    runner.run(test_suite_singleserver)
    runner.run(test_suite_lossnetwork)
