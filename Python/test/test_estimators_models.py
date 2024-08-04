# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:50:21 2024

@author: Daniel Mastropietro
@description: Unit tests for models used by estimators (e.g. neural network model).
@details: Naming conventions follow the instructions given in test_conventions.txt.
"""

import runpy
runpy.run_path('../../setup.py')

import unittest

import torch
from torch import nn

from Python.lib.estimators.nn_models import NNBackprop


#------------------------ Auxiliary functions -----------------------------#
def assert_nn_model_parameter_values(nn_model, expected_values, atol=1E-4):
    assert len(list(nn_model.parameters())) == (len(nn_model.hidden_layers) + 1) * 2, \
            "The number of neural network weights must be equal to twice the number of hidden and output layers ('twice' because there biases are stored as a separate tensor)"
    for p, param in enumerate(nn_model.parameters()):
        print(f"Checking set of weights {p + 1} of {len(list(nn_model.parameters()))}:\n{param}")
        assert torch.allclose(param, expected_values[p], atol=atol)
#------------------------ Auxiliary functions -----------------------------#


class Test_Class_nn_backprop(unittest.TestCase):

    def test_model_architecture(self):
        print("\n*** Running test " + self.id() + " ***")

        #-- Test #1: NN model with 1 hidden layer
        print("\n\n== Example 1: ONE hidden layer ==")
        # Set torch seed for repeatability of initial weights
        nn.init.torch.manual_seed(1717)
        input_size = 3
        hidden_layer_sizes = [4]
        output_size = 2
        nn_model = NNBackprop(input_size, hidden_layer_sizes, output_size, dict_activation_functions=dict({'hidden': [nn.ReLU] * len(hidden_layer_sizes)}))
        # Initialize normal weights around 10
        for param in nn_model.parameters():
            nn.init.normal_(param, 10, 1)
        # Apply the model on an input to get an output
        input = [0.1, 0.2, -0.1]
        output = nn_model.forward(input)
        # Expected values
        expected_params = [ torch.tensor([  [ 9.4209,  8.9498,  9.5832],
                                            [ 9.7965,  9.4860, 10.4271],
                                            [11.8622,  9.5910,  9.8823],
                                            [10.4175, 11.7759, 10.5388]
                                        ]),
                            torch.tensor(   [10.2388, 10.1114,  9.2475,  9.0519]),
                            torch.tensor([  [10.4875, 10.4466,  9.5101,  9.5846],
                                            [ 9.4122,  9.1849,  9.4617,  9.9501]
                                        ]),
                            torch.tensor(   [11.2867,  9.5100])
                            ]
        expected_output = torch.tensor([479.3459, 453.1945])
        # Assertions
        print(f"Neural network model:\n{nn_model}")
        print(f"Neural network weights:")
        assert_nn_model_parameter_values(nn_model, expected_params)
        print(f"Neural network output for input {input}: {output}")
        assert torch.allclose(output, expected_output, atol=1E-4)

        #-- Test #2: NN model with several hidden layers
        print("\n\n== Example 2: THREE hidden layers ==")
        # Set torch seed for repeatability of initial weights
        nn.init.torch.manual_seed(1717)
        input_size = 3
        hidden_layer_sizes = [2, 3, 3]
        output_size = 2
        nn_model = NNBackprop(input_size, hidden_layer_sizes, output_size, dict_activation_functions=dict({'hidden': [nn.ReLU] * len(hidden_layer_sizes)}))
        # Initialize the parameters to pre-specified weights
        # Ref: https://stackoverflow.com/questions/66724071/manually-assign-weights-using-pytorch
        # The size of each weight array is <# output neurons> x <# input neurons>
        param_values = [    # Input to hidden layer
                            # weights
                            [[1.5, 1.0, 0.5],
                             [1.0, 2.0, 0.7]],
                            # bias
                            [0.0, -0.5],

                            # Hidden to hidden
                            # weights
                            [[1.5, 1.0],
                             [1.0, 0.7],
                             [0.2, 0.2]],
                            # bias
                            [0.0, -0.5, 1.2],

                            # Hidden to hidden
                            # weights
                            [[1.5, 1.0, 0.5],
                             [1.0, 2.0, 0.7],
                             [-0.5, 0.5, -0.3]],
                            # bias
                            [0.0, -0.5, 3.2],

                            # Hidden to output
                            # weights
                            [[1.5, 1.0, 0.5],
                             [1.0, 2.0, 0.7]],
                            # bias
                            [1.2, -0.7],
                        ]
        nn_model.set_params(param_values)
        # Check assignment of parameters
        for p, param in enumerate(nn_model.parameters()):
            assert torch.allclose(param, torch.tensor(param_values[p]))
        # Apply the model on an input to get an output
        input = [0.1, 0.2, -0.1]
        output = nn_model.forward(input)
        # Expected values
        expected_params = [torch.tensor(param) for param in param_values]
        expected_output = torch.tensor([5.2880, 4.0869])
        # Assertions
        print(f"Neural network model:\n{nn_model}")
        print(f"Neural network weights:")
        assert_nn_model_parameter_values(nn_model, expected_params)
        print(f"Neural network output for input {input}: {output}")
        assert torch.allclose(output, expected_output, atol=1E-4)

        #-- Test #3: NN model with NO hidden layer
        print("\n\n== Example 3: NO hidden layer ==")
        # Set torch seed for repeatability of initial weights
        nn.init.torch.manual_seed(1717)
        input_size = 3
        hidden_layer_sizes = []
        output_size = 2
        nn_model = NNBackprop(input_size, hidden_layer_sizes, output_size, dict_activation_functions=dict({'hidden': [nn.ReLU] * len(hidden_layer_sizes)}))
        # Initialize normally-distributed weights around 10
        for param in nn_model.parameters():
            nn.init.normal_(param, 10, 1)
        # Apply the model on an input to get an output
        input = [0.1, 0.2, -0.1]
        output = nn_model.forward(input)
        # Expected values
        expected_params = [torch.tensor([[ 9.6633, 10.0963, 11.3330],
                                         [11.7418,  9.7149, 10.1703]]),
                           torch.tensor([9.7518, 8.7836])
                           ]
        expected_output = torch.tensor([11.6041, 10.8838])
        # Assertions
        print(f"Neural network model:\n{nn_model}")
        print(f"Neural network weights:")
        assert_nn_model_parameter_values(nn_model, expected_params)
        print(f"Neural network output for input {input}: {output}")
        assert torch.allclose(output, expected_output, atol=1E-4)

        # Do the same but using the NNBackprop.set_params() method, so that we test this method for setting neural network parameters in the no-hidden-layer case
        print("\nWe now test setting the neural network parameters using NNBackprop.set_params():")
        weights = expected_params[0]
        biases = expected_params[1]
        nn_model.set_params([weights, biases])  # Here we set the NN parameters
        output = nn_model.forward(input)
        # Assertions
        print(f"Neural network model:\n{nn_model}")
        print(f"Neural network weights:")
        assert_nn_model_parameter_values(nn_model, expected_params)
        print(f"Neural network output for input {input}: {output}")
        assert torch.allclose(output, expected_output, atol=1E-4)


if __name__ == "__main__":
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
    runner = unittest.TextTestRunner()

    # Run all tests
    # unittest.main()

    test_suite = unittest.TestSuite()
    test_suite.addTest(Test_Class_nn_backprop("test_model_architecture"))
    runner.run(test_suite)
