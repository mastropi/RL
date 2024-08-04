# -*- coding: utf-8 -*-
"""
Created on Fri, July 05 by Alphonse Lafon (stagiaire)

@author: Alphonse Lafon, Daniel Mastropietro
@description: Unit tests for parameterized policies (e.g. policies parameterized by a neural network).
"""

import runpy
runpy.run_path('../../setup.py')

import numpy as np
import unittest
import unittest_data_provider

import torch

from Python.lib.agents.policies.parameterized import PolNN

from Python.lib.environments.gridworlds import EnvGridworld2D

from Python.lib.estimators.nn_models import NNBackprop

from Python.test.test_estimators_models import assert_nn_model_parameter_values


class Test_PolNN_NHL(unittest.TestCase):
    "Test on a neural network model of a parameterized policy with No Hidden Layer (NHL)"
    # Number of states and number of actions of the environment considered (a 2D gridworld)
    nS = 3
    nA = 4

    @classmethod
    def setUpClass(cls):
        # Environment on which the policy acts
        env = EnvGridworld2D(shape=(Test_PolNN_NHL.nS, 1))

        # Neural Network model used by the policy
        nn_model = NNBackprop(Test_PolNN_NHL.nS, [], Test_PolNN_NHL.nA)
        cls.policy_nn = PolNN(env, nn_model)
        print(f"Neural network to model the policy:\n{cls.policy_nn.nn_model}")

    data_test_policy_init = lambda: (
        (1, 'Initial policy undefined', None,
            [torch.tensor(np.zeros((Test_PolNN_NHL.nA, Test_PolNN_NHL.nS))), torch.tensor(np.zeros(Test_PolNN_NHL.nA))],
             0.25*np.ones((Test_PolNN_NHL.nS, Test_PolNN_NHL.nA))),
        (2, 'Random initial policy defined explicitly as a probability function', [0.25, 0.25, 0.25, 0.25],
            [torch.tensor(-1.3863*np.ones((Test_PolNN_NHL.nA, Test_PolNN_NHL.nS))), torch.tensor(np.zeros(Test_PolNN_NHL.nA))],
             0.25*np.ones((Test_PolNN_NHL.nS, Test_PolNN_NHL.nA))),
        (3, 'Random initial policy defined explicitly as all ones', [1, 1, 1, 1],
            [torch.tensor(np.zeros((Test_PolNN_NHL.nA, Test_PolNN_NHL.nS))), torch.tensor(np.zeros(Test_PolNN_NHL.nA))],
             0.25*np.ones((Test_PolNN_NHL.nS, Test_PolNN_NHL.nA))),
        (4, 'Random initial policy defined explicitly as all twos', [2, 2, 2, 2],
            [torch.tensor(0.69315*np.ones((Test_PolNN_NHL.nA, Test_PolNN_NHL.nS))), torch.tensor(np.zeros(Test_PolNN_NHL.nA))],
             0.25*np.ones((Test_PolNN_NHL.nS, Test_PolNN_NHL.nA))),
        (5, 'A non-random initial policy defined as a probability function', [0.7, 0.2, 0, 0.1],
            [torch.tensor(np.repeat([-0.3567, -1.6094, -20.7233, -2.3026], Test_PolNN_NHL.nS).reshape(Test_PolNN_NHL.nA, Test_PolNN_NHL.nS)), torch.tensor(np.zeros(Test_PolNN_NHL.nA))],
             np.repeat([0.7, 0.2, 0, 0.1], Test_PolNN_NHL.nS).reshape(Test_PolNN_NHL.nA, Test_PolNN_NHL.nS).T),
        (6, 'A non-random initial policy defined as non-negative values', [7, 2, 0, 1],
            [torch.tensor(np.repeat([1.9459, 0.6931, -20.7233, 0.0], Test_PolNN_NHL.nS).reshape(Test_PolNN_NHL.nA, Test_PolNN_NHL.nS)), torch.tensor(np.zeros(Test_PolNN_NHL.nA))],
             np.repeat([0.7, 0.2, 0, 0.1], Test_PolNN_NHL.nS).reshape(Test_PolNN_NHL.nA, Test_PolNN_NHL.nS).T),
        (7, 'A non-random initial policy that depends on the state', [[0.7, 0.2, 0, 0.1],
                                                                      [3, 3, 3, 3],
                                                                      [0.1, 0.2, 0.5, 0.2]],
             [torch.tensor([[ -0.3567,  1.09861,  -2.3026],
                            [ -1.6094,  1.09861,  -1.6094],
                            [-20.7233,  1.09861,  -0.6931],
                            [ -2.3026,  1.09861,  -1.6094]]), torch.tensor(np.zeros(Test_PolNN_NHL.nA))],
              np.array([[0.70, 0.20, 0.00, 0.10],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.10, 0.20, 0.50, 0.20]])),
    )

    @unittest_data_provider.data_provider(data_test_policy_init)
    def test_policy_init(self, case, desc, policy_values, expected_params, expected_policy):
        "Test of a Natural Policy Gradient (NPG) implementation via a neural network with no hidden layer"
        print("\n*** Running test " + self.id() + ", case " + str(case), " ***")

        # 1) Initialize the policy to the given initial policy
        self.policy_nn.reset(initial_values=policy_values)
        print(f"Network parameters initialized as follows:\n{list(self.policy_nn.getThetaParameter())}")
        print("Initial policy for all states (states x actions):")
        policy_probabilities = self.policy_nn.get_policy_values()
        print(policy_probabilities)

        # Assertion on the parameters of the neural network used to model the policy
        assert_nn_model_parameter_values(self.policy_nn.nn_model, expected_params)
        # Assertion on the policy itself, output of the neural network model
        assert np.allclose(self.policy_nn.get_policy_values(), expected_policy)

    def test_policy_NPG(self):
        "Simulate NPG by SETTING the policy to the exponential of the advantage function values for each state"
        print("\n*** Running test " + self.id() + " ***")

        print("Setting the policy values to the exponential of an advantage function...")
        # The advantage values are given as a matrix of size nS x nA
        advantage_values = [[3, 0, -2, 0.5],
                            [2, 1, -0.3, 0.8],
                            [0, 1, 0, -np.Inf]]
        self.policy_nn.set_policy_values(values=np.exp(advantage_values))
        print(f"Network parameters set to:\n{list(self.policy_nn.getThetaParameter())}")
        print("Policy values for all states (states x actions):")
        policy_probabilities = self.policy_nn.get_policy_values()
        print(policy_probabilities)

        # Expected advantage values (they are slightly different than the original advantage values because the -Inf value is lower bounded by the set_policy_values() method
        expected_advantage_values = np.array(advantage_values).T
        expected_advantage_values[-1, -1] = np.log(1E-9)        # The value np.Inf in the original advantage value is converted to np.log(1E-9) by the PolNN.set_policy_values() method called above
        # Expected NN parameters = [weight, bias]
        expected_params = [torch.tensor(expected_advantage_values), torch.tensor(np.zeros(Test_PolNN_NHL.nA))]
        # Expected policy values (#states x #actions)
        expected_policy = [ [0.87826384, 0.04372618, 0.00591770, 0.07209229],
                            [0.56518489, 0.20791990, 0.05666478, 0.17023042],
                            [0.21194156, 0.57611689, 0.21194156, 0.0       ] ]

        # Assertion on the parameters of the neural network used to model the policy
        assert_nn_model_parameter_values(self.policy_nn.nn_model, expected_params)
        # Assertion on the policy itself, output of the neural network model
        assert np.allclose(self.policy_nn.get_policy_values(), expected_policy)


if __name__ == '__main__':
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
    runner = unittest.TextTestRunner()

    # Run all tests
    #unittest.main()

    test_suite = unittest.TestSuite()
    test_suite.addTest(Test_PolNN_NHL("test_policy_init"))
    test_suite.addTest(Test_PolNN_NHL("test_policy_NPG"))
    runner.run(test_suite)
