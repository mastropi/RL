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

import numpy as np
import pandas as pd

from Python.lib.agents.learners import LearnerTypes
from Python.lib.agents.learners.continuing.fv import LeaFV
from Python.lib.agents.learners.continuing.mc import LeaMC
from Python.lib.agents.learners.policies import LeaPolicyGradient

from Python.lib.agents.policies import PolicyTypes, random_walks
from Python.lib.agents.policies.job_assignment import PolJobAssignmentProbabilistic
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep

from Python.lib.agents.queues import AgeQueue

from Python.lib.environments.queues import EnvQueueSingleBufferWithJobClasses, rewardOnJobRejection_ExponentialCost
import Python.lib.queues as queues
from Python.lib.queues import Event

from Python.lib.simulators import LearningMethod
from Python.lib.simulators.queues import generate_event, LearningMode, SimulatorQueue

from Python.lib.utils.basic import show_exec_params
from Python.lib.utils.computing import compute_job_rates_by_server


class Test_EstPolicy_EnvQueueSingleServer(unittest.TestCase):
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

    def no_test_Env_MetFVRLwithIGA_TestOneCase(self):
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
        # Make a copy of the seed used to run the process (for the assertion below),
        # because the key in dict_params_simul is changed by the process
        # as we need to consider different seeds for the different simulation parts and this seed is passed to each
        # simulator through the dict_params_simul dictionary and is used to store its value in the output files
        # to allow repeatability.
        seed = dict_params_simul['seed']

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

        assert  dict_params_learn['mode'] == LearningMode.IGA and \
                dict_params_learn['t_learn'] == 5
        assert  seed == 1717 and \
                dict_params_simul['theta_true'] == 19 and \
                dict_params_simul['theta_start'] == 1.0 and \
                dict_params_simul['nparticles'] == 30 and \
                dict_params_simul['t_sim'] == 50 and \
                dict_params_simul['buffer_size_activation_factor'] == 0.3
        assert np.allclose(df_learning, df_learning_expected, atol=1E-6)

    def test_Env_MetFVRLwithReinforceTrue(self):
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
        # Make a copy of the seed used to run the process (for the assertion below),
        # because the key in dict_params_simul is changed by the process
        # as we need to consider different seeds for the different simulation parts and this seed is passed to each
        # simulator through the dict_params_simul dictionary and is used to store its value in the output files
        # to allow repeatability.
        seed = dict_params_simul['seed']

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
            ('Error(K-1)', [np.nan] * 5),   # TODO: Fill in these values from the generated results
            ('Error(K)', [np.nan] * 5),     # TODO: Fill in these values from the generated results
                                                ('Q_diff(K-1)', [0.90, 1.25, 2.55, 1.55, 0.75]),
                                                ('Q_diff(K)', [0.0]*5),
                                                ('alpha', [1.0]*5),
                                                ('V', [-9.000, -10.775, -8.925, -9.675, -9.725]),
                                                ('gradV', [0.123813, 0.234522, 0.584292, 0.157247, 0.112802]),
                                                ('n_events_mc', [95, 92, 88, 97, 88]),
                                                ('n_events_fv', [179, 235, 300, 454, 674]),
                                                ('n_trajectories_Q', [100.0]*5)
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

    def no_test_Env_MetMCwithIGA(self):
        unittest.skip("The current implementation (11-Jul-2022) is NOT prepared for IGA because integer theta values make Pr(K-1) be equal to None.\n" \
                "Correcting this is not so easy and we are not interested in evaluating IGA right now")
        print("\n")
        print("".join(np.repeat("*", 20)))
        print("Testing the MC algorithm on a single server system using the IGA learning strategy...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'r.-'})

        # Simulation parameters
        learning_method = LearningMethod.MC
        dict_params_learn = dict({'mode': LearningMode.IGA, 't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': 19,
            'theta_start': 1.0,
            'nparticles': 1,
            't_sim': 30 * 50,
            'buffer_size_activation_factor': 0.3
            })
        # Make a copy of the seed used to run the process (for the assertion below),
        # because the key in dict_params_simul is changed by the process
        # as we need to consider different seeds for the different simulation parts and this seed is passed to each
        # simulator through the dict_params_simul dictionary and is used to store its value in the output files
        # to allow repeatability.
        seed = dict_params_simul['seed']

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

        assert  dict_params_learn['mode'] == LearningMode.IGA and \
                dict_params_learn['t_learn'] == 5
        assert  seed == 1717 and \
                dict_params_simul['theta_true'] == 19 and \
                dict_params_simul['theta_start'] == 1.0 and \
                dict_params_simul['nparticles'] == 1 and \
                dict_params_simul['t_sim'] == 30 * 50 and \
                dict_params_simul['buffer_size_activation_factor'] == 0.3
        assert np.allclose(df_learning, df_learning_expected, atol=1E-6)

    def test_Env_MetMCwithReinforceTrue(self):
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
            'buffer_size_activation_factor': 0.3,
            'burnin_time_steps': 0,
            'min_num_cycles_for_expectations': 0
            })
        # Make a copy of the seed used to run the process (for the assertion below),
        # because the key in dict_params_simul is changed by the process
        # as we need to consider different seeds for the different simulation parts and this seed is passed to each
        # simulator through the dict_params_simul dictionary and is used to store its value in the output files
        # to allow repeatability.
        seed = dict_params_simul['seed']

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
            ('Error(K-1)', [np.nan]*5),     # TODO: Fill in these values from the generated results
            ('Error(K)', [np.nan]*5),       # TODO: Fill in these values from the generated results
                                                ('Q_diff(K-1)', [0.90, 1.70, 0.90, 0.90, 0.75]),
                                                ('Q_diff(K)', [0.0]*5),
                                                ('alpha', [1.0]*5),
                                                ('V', [-9.0, -10.0, -7.40, -8.0, -10.225]),
                                                ('gradV', [0.168038, 0.361367, 0.197101, 0.182898, 0.077933]),
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
                dict_params_simul['burnin_time_steps'] == 0 and \
                dict_params_simul['min_num_cycles_for_expectations'] == 0

        assert np.allclose(df_learning, df_learning_expected, atol=1E-6)


if __name__ == "__main__":
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
    runner = unittest.TextTestRunner()

    # Run all tests
    # unittest.main()

    # Create the test suites
    # Single-server tests
    # NOTE: IGA tests are skipped because the code is currently (11-Jul-2022) not prepared to properly handle the case
    # with integer-valued theta, and preparing it would introduce a little bit of noise in the definition of functions
    # (e.g. passing the learning mode (e.g. IGA or REINFORCE_TRUE) to functions such as
    # estimate_blocking_probability_fv() and similar.
    test_suite_singleserver = unittest.TestSuite()
    #test_suite_singleserver.addTest(Test_EstPolicy_EnvQueueSingleServer("test_Env_MetFVRLwithIGA_TestOneCase"))
    test_suite_singleserver.addTest(Test_EstPolicy_EnvQueueSingleServer("test_Env_MetFVRLwithReinforceTrue"))
    #test_suite_singleserver.addTest(Test_EstPolicy_EnvQueueSingleServer("test_Env_MetMCwithIGA"))
    test_suite_singleserver.addTest(Test_EstPolicy_EnvQueueSingleServer("test_Env_MetMCwithReinforceTrue"))

    # Run the test suites
    runner.run(test_suite_singleserver)
