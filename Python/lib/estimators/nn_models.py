# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 19:52:38 2023

@author: Daniel Mastropietro
@description: Neural network models
@references:
- For defining the neural network, I used Francisco Robledo's GitHub project, https://github.com/frrobledo/LPQL_NN,
in particular the file main/tools_nn.py where he defines a Q neural network.
- PyTorch documentation for the installed pytorch-1.10 package: https://pytorch.org/docs/1.10
(latest stable version at the moment of creation of this file is 2.0).
- Packages that can be used to visualize a neural network: https://stackoverflow.com/questions/52468956/how-do-i-visualize-a-net-in-pytorch
"""

import warnings
from enum import Enum, unique

import torch
from torch import nn

@unique
class InputLayer(Enum):
    "Input layer size for neural networks: either SINGLE for a single neuron representing e.g. the state, or ONEHOT for a one-hot encoding of the state"
    SINGLE = 1
    ONEHOT = 2


class NNBackprop(nn.Module):
    """
    Neural network with on input layer, any number of hidden layers and one output layer
    with possibly different activation functions at each layer and output layer.

    Before applying the activation function, the output of each node in the hidden and output layers
    is computed as a linear combination of its inputs.

    Arguments:

    dict_activation_functions: (opt) dict
        Dictionary containing the activation functions for each hidden layer and for the output layer.
        The following entries are parsed:
        - 'hidden': list with the activation functions of the hidden layers.
        - 'output': activation function for the output layer.
        When an empty dict, the default activation function for the hidden layer is nn.ReLU and the default
        activation function for the output layer is nn.Identity.
        default: empty dict

    dropout: (opt) float in [0, 1]
        Probability of dropout in dropout layers added after the input layer and each hidden layer.
        When 0.0, no dropout layers are used.
        default: 0.0
    """
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int,
                 dict_activation_functions: dict=dict(),
                 dropout=0.0):
        super().__init__()

        # Parse input parameters
        self.list_activation_functions_hidden, self.activation_function_output = self._parse_input_parameters(hidden_sizes, dict_activation_functions)
        self.dropout = dropout  # Dropout proportion: usually values between 0.2 and 0.5 are good (Ref: https://machinelearningmastery.com/using-dropout-regularization-in-pytorch-models)

        # Extend the hidden sizes so that we have a full specification of input and output of each hidden layer, including the first one
        hidden_sizes_extended = [input_size] + hidden_sizes

        #-- Define the network from input to output through hidden layers
        # Initialize hidden layers
        # Note that we need to define the layers as a nn.ModuleList, rather than simply a list, o.w. the layers are not recognized as part of the neural network model
        self.hidden_layers = nn.ModuleList()

        # Drop out layers in case dropout is requested
        # One is added between any connecting layers
        self.dropout_layers = nn.ModuleList()

        # Additional hidden layers (if any)
        for h in range(1, len(hidden_sizes_extended)):
            # Add a drop out layer if requested
            if self.dropout > 0.0:
                self.dropout_layers.append( nn.Dropout(self.dropout) )
            self.hidden_layers.append( nn.Linear(hidden_sizes_extended[h-1], hidden_sizes_extended[h]) ) #, dtype=torch.float) ) # DM-2024/06/11: Commented out `dtype` because it fails in IRIT cluster with torch-1.8 installed

        # If we want to initialize weights and bias leading to a layer, use the nn.init functions
        # Here an example of setting the weights and bias reaching the first hidden layer
        #   nn.init.ones_(self.hidden_layers[0].weight)
        #   nn.init.uniform_(self.hidden_layers[0].weight, -1, 1)     # Uniform weights between -1 and 1
        #   nn.init.zeros_(self.hidden_layers[0].bias)
        # Another option is to use the object returned by parameters() (a generator that returns Parameter objects, of type tensor) of a component (e.g. a hidden layer)
        # When the parameters() generator applied to a component (i.e. to just ONE layer) is viewed as a list, it contains 2 elements: (i) the weights tensor, (ii) the bias tensor
        # The weights are of size <# output neurons> x <# input neurons>. The bias is of length <# output neurons>.
        # In the following example we assume that the output layer (i.e. the first hidden layer) contains 2 neurons and the input layer contains 3 neurons.
        #   import torch
        #   weights = [ [0.5, 0.2, -0.3],
        #               [1.0, -0.5, -0.7]]
        #   bias = [0.5, -0.8]
        #   param_values = [weights, bias]
        #   for p, param in enumerate(self.hidden_layers[0].parameters()):
        #       param.data = nn.parameter.Parameter(torch.tensor(param_values[p]))

        # Dropout layer before the output layer
        if self.dropout > 0.0:
            self.dropout_layers.append( nn.Dropout(self.dropout) )

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes_extended[-1], output_size) #, dtype=torch.float) # DM-2024/06/11: Commented out `dtype` because it fails in IRIT cluster with torch-1.8 installed
        # If we want to initialize the weights and bias leading to the output layer neurons
        # Also see init.uniform_(), init.xavier_uniform_(), etc.
        #   nn.init.ones_(self.output_layer.weight)
        #   nn.init.zeros_(self.output_layer.bias)

    # The nn.Module.forward() method needs to be ALWAYS overridden by subclasses (as indicated in the PyTorch documentation)
    # There is no useful documentation on the nn.Module.forward() method. I took the signature and implementation
    # from Francisco's example of the Q-NN he wrote in his GitHub project referenced at the top.
    def forward(self, x):
        """
        Defines the forward pass through the neural network, i.e. the calculations to perform on the input to get the output

        x: array-like
            Input for the neural network which is converted to a PyTorch tensor before feeding it to the network.
        """
        # Convert the input array into a tensor
        tensor = torch.tensor(x, dtype=torch.float)

        # Pass through the hidden layers and possibly dropout layers
        for h, hidden_layer in enumerate(self.hidden_layers):
            if self.dropout > 0.0:
                tensor = self.dropout_layers[h](tensor)
            tensor = self.list_activation_functions_hidden[h]()( hidden_layer(tensor) )

        if self.dropout > 0.0:
            tensor = self.dropout_layers[h](tensor)

        # Output value
        out = self.activation_function_output()( self.output_layer(tensor) )

        return out

    def set_params(self, values):
        """
        Sets the neural network parameters (weights and biases) to the given values

        The parameter values must comply with the neural network architecture and must be given in the order specified by the neural network layers,
        and for each layer it must contain first the weights connecting the input neurons to the output neurons in the layer which should be a 2D array
        or list of size <# output neurons> x <# input neurons>, and then the biases which should be a 1D array or list of length <# output neurons>.

        Both the weights and biases are converted to tensors when setting the respective parameter value.

        Ex: In a neural network with 3 inputs and 4 outputs and ONE hidden layer with 2 neurons, parameter `values` could be the following,
        where weights are initialized from a uniform distribution in [0, 1] and all biases are initialized to 1.0:
        values = [  [np.random.uniform(size=(2, 3)), [1.0, 1.0]],
                    [np.random.uniform(size=(4, 2)), [1.0, 1.0, 1.0, 1.0]]
                ]
        """
        for p, param in enumerate(self.parameters()):
           param.data = nn.parameter.Parameter(torch.tensor(values[p]))

    def _parse_input_parameters(self, hidden_sizes: list, dict_activation_functions: dict,
                                default_hidden_function=nn.ReLU,
                                default_output_function=nn.Identity):
        """
        Parse the input parameters of the class that define the number of neurons in each hidden layer
        and the activation functions of the hidden layers and of the output layer.

        Return: tuple
        Duple with the list of activation functions across hidden layers and the activation function for the output layer.
        """
        # Hidden layers
        if not isinstance(hidden_sizes, list):
            raise ValueError(f"Input parameter `hidden_sizes` must be a list containing the lengths of each hidden layer: {hidden_sizes}")

        # Activation functions dictionary
        if not isinstance(dict_activation_functions, dict):
            raise ValueError(f"Input parameter `dict_activation_functions` must be a dictionary with two entries, 'hidden' and 'output'"
                             f" specifying a list with the activation functions for the hidden layers and the activation function for the output layer, respectively:"
                             f"\n{dict_activation_functions}")

        # Output activation function
        activation_function_output = dict_activation_functions.get('output', default_output_function)
        if not isinstance(activation_function_output(), nn.Module):
            raise ValueError(f"The value of entry 'output' in input dictionary `dict_activation_functions` must be a torch activation function inheriting from the Module class:"
                             f"\n{type(activation_function_output())}")

        # Hidden activation functions
        list_activation_functions_hidden = dict_activation_functions.get('hidden', [default_hidden_function] * len(hidden_sizes))
        if not isinstance(list_activation_functions_hidden, list):
            raise ValueError(f"The value of entry 'hidden' in dictionary `dict_activation_functions` must be a list with the activation functions to use for each hidden layer:"
                             f"\n{list_activation_functions_hidden}")
        if not len(list_activation_functions_hidden) == len(hidden_sizes):
            raise ValueError(f"The value of entry 'hidden' in dictionary `dict_activation_functions` must be a list with as many elements as the number of hidden layers specified via parameter `hidden_sizes` ({len(hidden_sizes)})"
                             f"\n{list_activation_functions_hidden}")
        # Check the type of each element of the list of activation functions for the hidden layers
        for h, fun in enumerate(list_activation_functions_hidden):
            if not isinstance(fun(), nn.Module):
                raise ValueError(f"Each value in the list specified in entry 'hidden' of input dictionary `dict_activation_functions` must be a torch activation function inheriting from the Module class."
                                 f"\nError at element {h} whose type is {type(fun())}")

        return list_activation_functions_hidden, activation_function_output

    # Getters
    def getNumInputs(self):
        if len(self.hidden_layers) > 0:
            return self.hidden_layers[0].in_features
        else:
            return self.output_layer.in_features

    def getNumHiddenLayers(self):
        return len(self.hidden_layers)

    def getNumDropoutLayers(self):
        return len(self.dropout_layers)

    def getNumOutputs(self):
        return self.output_layer.out_features

    def getHiddenLayer(self, h):
        if h > len(self.hidden_layers):
            warnings.warn(f"The requested hidden layer (h={h}) does not exist. None is returned")
            return None
        return self.hidden_layers[h]

    def getDropoutLayer(self, d):
        if d > len(self.dropout_layers):
            warnings.warn(f"The requested dropout layer (d={d}) does not exist. None is returned")
            return None
        return self.dropout_layers[d]

    def getOutputLayer(self):
        return self.output_layer