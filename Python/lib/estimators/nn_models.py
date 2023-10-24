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




import torch
from torch import nn
import torch.nn.functional as F


class nn_backprop(nn.Module):
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
    """
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int,
                 dict_activation_functions: dict=dict()):
        super().__init__()

        # Parse input parameters
        self.list_activation_functions_hidden, self.activation_function_output = self._parse_input_parameters(hidden_sizes, dict_activation_functions)

        #-- Define the network from input to output through hidden layers
        # First hidden layer
        #self.hidden_layers = [None]*len(hidden_sizes)
        #self.hidden_layers[0] = nn.Linear(input_size, hidden_sizes[0], dtype=torch.float)     # Note: Apparently the dtype= argument is no longer needed in pytorch-1.14 (based on Francisco's code)
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append( nn.Linear(input_size, hidden_sizes[0], dtype=torch.float) )
        # If we want to initialize the weights and bias leading to the hidden layer neurons
        #nn.init.ones_(self.hidden_layers[0].weight)
        #nn.init.uniform_(self.hidden_layers[0].weight, -1, 1)     # Uniform weights between -1 and 1
        #nn.init.zeros_(self.hidden_layers[0].bias)

        # Additional hidden layers (if any)
        for h in range(len(hidden_sizes[1:])):
            #self.hidden_layers[h] = nn.Linear(hidden_sizes[h-1], hidden_sizes[h], dtype=torch.float)
            self.hidden_layers.append( nn.Linear(hidden_sizes[h - 1], hidden_sizes[h], dtype=torch.float) )

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size, dtype=torch.float)
        # If we want to initialize the weights and bias leading to the output layer neurons
        # Also see init.uniform(), init.xavier_uniform(), etc.
        #nn.init.ones_(self.output_layer.weight)
        #nn.init.zeros_(self.output_layer.bias)

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

        # Pass through the hidden layers
        for h, hidden_layer in enumerate(self.hidden_layers):
            tensor = self.list_activation_functions_hidden[h]()( hidden_layer(tensor) )

        # Output value
        out = self.activation_function_output()( self.output_layer(tensor) )

        return out

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
        return self.hidden_layers[0].in_features

    def getNumOutputs(self):
        return self.output_layer.out_features
