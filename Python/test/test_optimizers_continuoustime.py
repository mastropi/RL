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
from typing import Union

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
from Python.lib.environments.queues import rewardOnJobRejection_ByClass, rewardOnJobRejection_ExponentialCost
import Python.lib.queues as queues

from Python.lib.simulators import LearningMethod
from Python.lib.simulators.queues import LearningMode, SimulatorQueue, SurvivalProbabilityEstimation

from Python.lib.utils.basic import assert_equal_data_frames, is_scalar, show_exec_params
from Python.lib.utils.computing import compute_expected_cost_knapsack


class Test_EstPolicy_EnvQueueSingleServer(unittest.TestCase):
    # Note: nice explanation about the three types of methods that can be defined in Python: instance, class, static
    # https://stackoverflow.com/questions/54264073/what-is-the-use-and-when-to-use-classmethod-in-python
    # See the only answer by Navy Cheng.

    # TODO: (2023/02/02) Setup the class by calling function define_queue_environment_and_agent() defined in simulators/queues.py (instead of using the classmethod createAgentWithPolicyGradientLearner()) which takes care of everything I do here, and in addition we initialize objects in a consistent manner!
    @classmethod
    def setUpClass(cls):
        # Define the queue environment
        capacity = np.Inf
        nservers = 1
        job_class_rates = [0.7]
        service_rates = [1.0]
        job_rates_by_server = job_class_rates
        queue = queues.QueueMM(job_rates_by_server, service_rates, nservers, capacity)
        cls.env_queue_mm_singleserver = env_queues.EnvQueueSingleBufferWithJobClasses(queue, job_class_rates, rewardOnJobRejection_ExponentialCost, None)

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

    def test_Env_MetMCwithReinforceTrue(self):
        print("".join(np.repeat("*", 20)))
        print("\nRunning test " + self.id())
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
                                                ('theta', [1.1, 5.941948, 6.389987, 6.269116, 6.716095]),
                                                ('theta_next', [5.941948, 6.389987, 6.269116, 6.716095, 7.517759]),
                                                ('proba_blocking', [0.188836, 0.001935, 0.028018, 0.016008, 0.008819]),
                                                ('proba_blocking_true', [0.309514, 0.028392, 0.033727, 0.036838, 0.025331]),
                                                ('error_proba_blocking', [-0.389896, -0.931843, -0.169255, -0.565458, -0.651833]),
                                                ('expected_reward', [-0.944180, -0.009676, -0.140091, -0.080039, -0.044097]),
                                                ('expected_reward_true', [-1.547572, -0.141960, -0.168633, -0.184192, -0.126654]),
                                                ('error_reward', [0.389896, 0.931843, 0.169255, 0.565458, 0.651833]),
                                                ('alpha', [10.0]*5),
                                                ('gradV', [[0.48419478693402257], [0.04480394277065356], [-0.012087160327577427], [0.04469796225393303], [0.08016635442577162]]),
                                                ('n_events_mc', [1500]*5),
                                                ('n_events_fv', [0]*5),
                                                ('n_trajectories_Q', [91, 93, 95, 98, 93])
                                                ])
        # Check execution parameters
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

        # Check results
        assert_equal_data_frames(df_learning, df_learning_expected, df_learning_expected.columns, atol=1E-6)

    def test_Env_MetFVRLwithReinforceTrue(self):
        print("".join(np.repeat("*", 20)))
        print("\nRunning test " + self.id())
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
                                                ('theta', [1.1, 7.782423, 7.782423, 7.970238, 7.968525]),
                                                ('theta_next', [7.782423, 7.782423, 7.970238, 7.968525, 8.178420]),
                                                ('proba_blocking', [0.260614, 0.0, 0.008446, 0.000092, 0.000374]),
                                                ('proba_blocking_true', [0.309514, 0.016330, 0.016330, 0.012988, 0.013018]),
                                                ('error_proba_blocking', [-0.157989, -1.0, -0.482828, -0.992937, -0.971301]),
                                                ('expected_reward', [-1.303072, -0.0, -0.042228, -0.000459, -0.001868]),
                                                ('expected_reward_true', [-1.547572, -0.081652, -0.081652, -0.064939, -0.065091]),
                                                ('error_reward', [0.157989, 1.0, 0.482829, 0.992937, 0.971301]),
                                                ('alpha', [10.0]*5),
                                                ('gradV', [[0.6682422813177087], [0.0], [0.018781526951805034], [-0.00017128367010005866], [0.020989455330044243]]),
                                                ('n_events_mc', [87, 100, 99, 99, 99]),
                                                ('n_events_fv', [431, 0, 1573, 561, 688]),
                                                ('n_trajectories_Q', [91, 0, 93, 90, 82])
                                                ])
        # Check execution parameters
        assert  dict_params_learn['mode'] == LearningMode.REINFORCE_TRUE and \
                dict_params_learn['t_learn'] == 5
        assert  seed == 1717 and \
                dict_params_simul['theta_true'] == 19 and \
                dict_params_simul['theta_start'] == 1.1 and \
                dict_params_simul['nparticles'] == 30 and \
                dict_params_simul['t_sim'] == 50 and \
                dict_params_simul['buffer_size_activation_factor'] == 0.3 and \
                dict_params_simul['burnin_time_steps'] == 20 and \
                dict_params_simul['min_num_cycles_for_expectations'] == 5

        # Check results
        assert_equal_data_frames(df_learning, df_learning_expected, df_learning_expected.columns, atol=1E-6)

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
         {'theta_next': [1.938283, 8.873832, 0.100, 2.657005, 3.36032835],
          'proba_blocking': [0.139052, 0.014933, 0.002926, 0.367573, 0.032917],
          'n_events_fv': [620, 591, 3467, 354, 1111]}),

        (2, DEFAULT_EXECUTION, '1D-theta as list',
         {'seed': 1313, 'theta_true': [5.0], 'theta_start': [2.1], 'nparticles': 50, 't_sim': 100,
          'buffer_size_activation_factor': 0.5,
          'burnin_time_steps': 20, 'min_num_cycles_for_expectations': 5,
          'method_survival_probability_estimation': SurvivalProbabilityEstimation.FROM_N_PARTICLES},
         {'method': LearningMethod.FV,
          'alpha_start': 10.0, 'adjust_alpha': True, 'func_adjust_alpha': np.float, 'min_time_to_update_alpha': 0, 'alpha_min': 0.01,
          'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5},
         # In the following "expected" dictionary we should include the value of columns in the df_learning output yielded by self.runSimulation() that we would like to test
         {'theta_next': [np.array([1.938283]), np.array([8.873832]), np.array([0.100]), np.array([2.657005]), np.array([3.36032835])],
          'proba_blocking': [0.139052, 0.014933, 0.002926, 0.367573, 0.032917],
          'n_events_fv': [620, 591, 3467, 354, 1111]}),
    )
    # -------- DATA -------

    @data_provider(data_test_Env_MetFVRL)
    def test_Env_MetFVRL_TestSeveralCases(self, casenum, run, desc, dict_params_simul, dict_params_learn, dict_expected):
        "Test the FVRL implementation that learns the optimum parameter theta of a parameterized policy on the single-server queue system"
        print("\nRunning test " + self.id())

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
                try:
                    assert np.allclose(df_learning[column], expected_value, atol=1E-6, equal_nan=True)
                except:
                    # The data in the column to analyze is too complex for np.allclose() (e.g. each element is a list or an array of values)
                    # => Compare each element separately
                    for i, observed_row in enumerate(df_learning[column]):
                        assert np.isclose(observed_row, expected_value[i], atol=1E-6, equal_nan=True)


class Test_EstPolicy_EnvQueueLossNetworkWithJobClasses(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rates_birth = [0.1, 0.5, 0.8] #[2, 8, 15] #[0.1, 0.5, 0.8]       # Let's use very different arrival rates so that we can see differences in the optimum theta values found...
        rates_death = [1.0, 1.0, 1.0]
        nservers = len(rates_death)
        # Cost of blocking an incoming job of each class
        # NOTE: The blocking costs are stored as parameters of the reward function in the environment defined below,
        # where we pass the dict_params_reward_func dictionary to the environment constructor
        cls.costs_blocking = [1E1, 1E2, 1E4] #[1E1, 2E2, 5E3] #[1E2, 1E4, 1E6]
        # Capacity of the loss network
        K = 10
        queue_mm_loss = queues.QueueMM(rates_birth, rates_death, nservers, K, origin=0.0)
        reward_func = rewardOnJobRejection_ByClass
        rewards_accept_by_job_class = [0.0] * len(rates_birth)
        dict_params_reward_func = dict({'reward_at_rejection': [-c for c in cls.costs_blocking]})
        cls.env_queue_mm_loss = env_queues.EnvQueueLossNetworkWithJobClasses(   queue_mm_loss,
                                                                                reward_func,
                                                                                rewards_accept_by_job_class,
                                                                                dict_params_reward_func)

        # Compute the minimum expected cost and its corresponding blocking sizes (job occupancies)
        # (it could be a multiple set of blocking sizes if the minimum expected cost is achieved at different set of blocking sizes (unlikely though))
        # so that we can compare the optimum theta estimated for the loss network with the true optimum theta
        print(f"Computing the optimum expected cost for a knapsack with capacity = {K}, blocking costs = {cls.costs_blocking}, lambdas = {rates_birth}, mus = {rates_death}...")
        expected_costs = compute_expected_cost_knapsack(cls.costs_blocking, K, [l/m for l, m in zip(rates_birth, rates_death)], rates_birth)
        min_expected_cost = min([v for v in expected_costs.values()])
        states_min_expected_cost = [k for k, v in expected_costs.items() if v == min_expected_cost]
        print(f"Minimum expected cost = {min_expected_cost} happening at state = {states_min_expected_cost}")
        print("Lowest 10 expected cost values:")
        for k in sorted(expected_costs, key=lambda x: expected_costs[x])[:11]:
            print(k, expected_costs[k])
        print("Highest 10 expected cost values:")
        for k in sorted(expected_costs, key=lambda x: expected_costs[x])[-11:]:
            print(k, expected_costs[k])
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([t[1] for t in sorted(expected_costs.items(), key=lambda x: x[1])])
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_title(f"Sorted expected costs on different blocking size tuples: K={K}, costs={cls.costs_blocking}, lambdas={rates_birth}, mus={rates_death}")

        cls.expected_costs = expected_costs
        cls.optimum_expected_cost = min_expected_cost
        cls.states_optimum_expected_cost = states_min_expected_cost
        cls.dict_optimum_expected_cost = dict([(k, min_expected_cost) for k in states_min_expected_cost])

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
                         LearnerTypes.P: LeaPolicyGradient(cls.env_queue_mm_loss, policies[PolicyTypes.ACCEPT], learnerV,
                                                           alpha=dict_params_learn.get('alpha_start', 1.0),
                                                           adjust_alpha=dict_params_learn.get('adjust_alpha', False),
                                                           func_adjust_alpha=dict_params_learn.get('func_adjust_alpha', np.float),
                                                           min_count_to_update_alpha=dict_params_learn.get('min_count_to_update_alpha', 0),
                                                           min_time_to_update_alpha=dict_params_learn.get('min_time_to_update_alpha', 0),
                                                           alpha_min=dict_params_learn.get('alpha_min', 0),
                                                           fixed_window=dict_params_learn.get('fixed_window', False),
                                                           clipping=dict_params_learn.get('clipping', False),
                                                           clipping_value=dict_params_learn.get('clipping_value', +1.0))
                         })
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

    @classmethod
    def showResults(cls, agent_gradient):
        print("\n\n**************")
        optimum_theta = agent_gradient.getAcceptancePolicyThresholds()
        optimum_blocking = agent_gradient.getAcceptancePolicyIntegerThresholds()
        print("ESTIMATED:")
        print("Optimum theta: {}".format(optimum_theta))
        print("Optimum blocking sizes: {}".format(optimum_blocking))
        print("Expected cost (based on true stationary probabilities): {}".format(cls.expected_costs[tuple(optimum_blocking)]))
        print("")
        print("TRUE:")
        print("Optimum theta: {}".format([tuple([K-1 for K in state]) for state in cls.states_optimum_expected_cost]))
        print("Optimum blocking sizes: {}".format(cls.states_optimum_expected_cost))
        print("Expected cost (based on true stationary probabilities): {}".format(cls.optimum_expected_cost))
        print("**************")

    def test_Env_MetMCwithReinforceTrue(self):
        print("".join(np.repeat("*", 20)))
        print("\nRunning test " + self.id())
        print("Testing the MC algorithm on a loss network system receiving multi-class jobs using the REINFORCE_TRUE learning strategy...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'r.-'})

        # Simulation parameters
        dict_params_learn = dict({  'method': LearningMethod.MC,
                                    'alpha_start': 10.0, 'adjust_alpha': True, 'func_adjust_alpha': np.float, 'min_time_to_update_alpha': 0, 'alpha_min': 0.01,
                                    # Use the following when learning using p(x) = 1.0 for all x
                                    #'alpha_start': 0.01, 'adjust_alpha': False, 'func_adjust_alpha': np.float, 'min_time_to_update_alpha': 0, 'alpha_min': 0.01,
                                    'clipping': False, 'clipping_value': +1.0,
                                    'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': [4, 7, 10],
            'theta_start': [0.1, 0.1, 0.1],
            'nparticles': 1,
            't_sim': 30 * 50,
            'buffer_size_activation_factor': [0.3, 0.3, 0.3],
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
        print("Learning results stored in df_learning:")
        print(df_learning)
        print(df_learning['gradV'].iloc[0])
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [[0.1, 0.1, 0.1], [4.9, 4.9, 4.9], [4.9, 4.9, 4.9], [4.9, 4.814151077924854, 8.9], [4.9, 4.814151077924854, 8.9]]),
                                                ('theta_next', [[4.9, 4.9, 4.9], [4.9, 4.9, 4.9], [4.9, 4.814151077924854, 8.9], [4.9, 4.814151077924854, 8.9], [4.9, 4.814151077924854, 8.9]]),
                                                ('gradV', [ [220.84710457183985, 437.3811029909116, 3758.4185227],
                                                            [0.0, 0.0, 0.0],
                                                            [0.0, -0.017169784415029192, 19.5078299963848],
                                                            [0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0]]),
                                                ('alpha', [10.0, 10.0, 5.0, 3.33333, 2.5]),
                                                ('state_coverage', [[0.7777777777777778, 0.6666666666666666, 0.555555],
                                                                    [0.0, 0.0, 0.0],
                                                                    [0.0, 0.13333333333333333, 0.26666666666666666],
                                                                    [0.0, 0.0, 0.0],
                                                                    [0.0, 0.0, 0.0]]),
                                                ('proba_blocking', [0.350364, 0.0, 0.000405, 0.0, 0.0]),
                                                ('proba_blocking_true', [0.412838, 0.000175, 0.000175, 0.000017, 0.000017]),
                                                ('expected_reward_true', [-2815.93327, -1.641034, -1.641034, -0.012536, -0.012536]),
                                                ('n_events_mc', [1500]*5),
                                                ('n_events_fv', [0]*5),
                                                ('n_trajectories_Q', [76.333333, np.nan, 76.8333333, np.nan, np.nan]),
                                                ('seed', [1727]*5),
                                                ])
        print("Optimum values found by the optimization process, compared to the true optimum values")
        self.showResults(agent_gradient)

        # Check execution parameters
        assert  dict_params_learn['mode'] == LearningMode.REINFORCE_TRUE and \
                dict_params_learn['t_learn'] == 5
        assert  seed == 1717 and \
                self.costs_blocking == [1E1, 1E2, 1E4] and \
                dict_params_simul['theta_true'] == [4, 7, 10] and \
                dict_params_simul['theta_start'] == [0.1, 0.1, 0.1] and \
                dict_params_simul['nparticles'] == 1 and \
                dict_params_simul['t_sim'] == 30*50 and \
                dict_params_simul['buffer_size_activation_factor'] == [0.3, 0.3, 0.3] and \
                dict_params_simul['burnin_time_steps'] == 20 and \
                dict_params_simul['min_num_cycles_for_expectations'] == 5

        # Check results
        assert_equal_data_frames(df_learning, df_learning_expected, df_learning_expected.columns, printFlag=False)

    def test_Env_MetFVRLwithReinforceTrue(self):
        print("".join(np.repeat("*", 20)))
        print("\nRunning test " + self.id())
        print("Testing the FVRL algorithm on a loss network system receiving multi-class jobs using the REINFORCE_TRUE learning strategy...")
        print("Optimum theta and expected cost for loss network system: {}".format(self.dict_optimum_expected_cost))

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'g.-'})

        # Simulation parameters
        dict_params_learn = dict({'method': LearningMethod.FV,
                                  'alpha_start': 10.0, 'adjust_alpha': True, 'func_adjust_alpha': np.float, 'min_time_to_update_alpha': 0, 'alpha_min': 0.01,
                                 'mode': LearningMode.REINFORCE_TRUE, 't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': [4, 7, 10],            # These theta_true are NOT the actual true thetas!! In fact, we do NOT know the true values at this point.
            'theta_start': [0.1, 0.1, 0.1],
            'nparticles': 30,
            't_sim': 50,
            'buffer_size_activation_factor': [0.3, 0.3, 0.3],
            'burnin_time_steps': 0, #20,
            'min_num_cycles_for_expectations': 0, #5,
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
        print(df_learning['gradV'].iloc[0])
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [[0.1, 0.1, 0.1], [4.9, 4.9, 4.9], [4.9, 4.818983985909275, 8.9], [4.9, 4.923198809058585, 8.9], [4.9, 4.872936280953478, 8.9]]),
                                                ('theta_next', [[4.9, 4.9, 4.9], [4.9, 4.818983985909275, 8.9], [4.9, 4.923198809058585, 8.9], [4.9, 4.872936280953478, 8.9], [4.9, 4.872936280953478, 8.9]]),
                                                ('gradV', [ [292.342327, 383.859268, 2522.8387238],
                                                            [0.0, -0.008101601409072515, 15.98133783039745],
                                                            [0.0, 0.02084296, 0.0],
                                                            [0.0, -0.0150788, 0.0],
                                                            [0.0, 0.0, 0.0]]),
                                                ('alpha', [10.0, 10.0, 5.0, 3.33333, 2.5]),
                                                ('state_coverage', [[0.66666666, 0.66666666, 0.55555555],
                                                                    [0.0, 0.066666666, 0.2],
                                                                    [0.0, 0.133333333, 0.0],
                                                                    [0.0, 0.133333333, 0.0],
                                                                    [0.0, 0.0, 0.0]]),
                                                ('proba_blocking', [0.295619, 0.000307, 0.000015, 0.000002, 0.0]),
                                                ('proba_blocking_true', [0.412838, 0.000175, 0.000017, 0.000011, 0.000014]),
                                                ('expected_reward_true', [-2815.93327, -1.641034, -0.012509, -0.011922, -0.012205]),
                                                ('n_events_mc', [84, 100, 99, 99, 99]),
                                                ('n_events_fv', [701, 724, 394, 721, 354]),
                                                ('n_trajectories_Q', [76.272727, 74.25, 75.0, 81.0, np.nan]),
                                                ('seed', [1727]*5),
                                                ])
        print("Optimum values found by the optimization process, compared to the true optimum values")
        self.showResults(agent_gradient)

        # Check execution parameters
        assert  dict_params_learn['mode'] == LearningMode.REINFORCE_TRUE and \
                dict_params_learn['t_learn'] == 5
        assert  seed == 1717 and \
                self.costs_blocking == [1E1, 1E2, 1E4] and \
                dict_params_simul['theta_true'] == [4, 7, 10] and \
                dict_params_simul['theta_start'] == [0.1, 0.1, 0.1] and \
                dict_params_simul['nparticles'] == 30 and \
                dict_params_simul['t_sim'] == 50 and \
                dict_params_simul['buffer_size_activation_factor'] == [0.3, 0.3, 0.3] and \
                dict_params_simul['burnin_time_steps'] == 0 and \
                dict_params_simul['min_num_cycles_for_expectations'] == 0

        # Check results
        assert_equal_data_frames(df_learning, df_learning_expected, df_learning_expected.columns, printFlag=False)


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
    test_suite_singleserver.addTest(Test_EstPolicy_EnvQueueSingleServer("test_Env_MetMCwithReinforceTrue"))
    test_suite_singleserver.addTest(Test_EstPolicy_EnvQueueSingleServer("test_Env_MetFVRLwithReinforceTrue"))
    test_suite_singleserver.addTest(Test_EstPolicy_EnvQueueSingleServer("test_Env_MetFVRL_TestSeveralCases"))

    # 2) Loss network tests
    test_suite_lossnetwork = unittest.TestSuite()
    test_suite_lossnetwork.addTest(Test_EstPolicy_EnvQueueLossNetworkWithJobClasses("test_Env_MetMCwithReinforceTrue"))
    test_suite_lossnetwork.addTest(Test_EstPolicy_EnvQueueLossNetworkWithJobClasses("test_Env_MetFVRLwithReinforceTrue"))

    #-- Run the test suites
    runner.run(test_suite_singleserver)
    runner.run(test_suite_lossnetwork)
