# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:28:13 2023

@author: Daniel Mastropietro
@description: Unit tests for optimizers (e.g. optimum policy) on discrete-time MDPs.
@details: Naming conventions follow the instructions given in test_conventions.txt.
"""

if __name__ == "__main__":
    import runpy
    runpy.run_path('../../setup.py')

from enum import Enum, unique

import unittest
from unittest_data_provider import data_provider
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

from torch import nn

import Python.lib.agents as agents

from Python.lib.agents.learners import LearningCriterion, LearningTask, ResetMethod
from Python.lib.agents.learners.episodic.discrete import fv, td

from Python.lib.agents.policies.parameterized import PolNN

from Python.lib.environments import gridworlds
from Python.lib.environments.gridworlds import get_adjacent_states

from Python.lib.estimators.nn_models import nn_backprop

from Python.lib.simulators.discrete import Simulator as DiscreteSimulator

from Python.lib.utils.basic import assert_equal_data_frames, is_scalar, show_exec_params

@unique
class InputLayer(Enum):
    "Input layer size: either SINGLE for a single neuron representing the state, or ONEHOT for a one-hot encoding of the state"
    SINGLE = 1
    ONEHOT = 2


class Test_EstPolicy_EnvGridworldsWithObstacles(unittest.TestCase):
    # Note: nice explanation about the three types of methods that can be defined in Python: instance, class, static
    # https://stackoverflow.com/questions/54264073/what-is-the-use-and-when-to-use-classmethod-in-python
    # See the only answer by Navy Cheng.

    @classmethod
    def setUpClass(cls, shape=(3, 4), obstacles_set: Union[list, set]=None, start_states_set: set=None,
                        # Characteristics of the neural network for the Actor Critic policy learner
                        nn_input: InputLayer=InputLayer.ONEHOT, nn_hidden_layer_sizes: list=[8], initial_policy=None,
                        # Characteristics of all learners
                        learning_task=LearningTask.EPISODIC,
                        learning_criterion=LearningCriterion.DISCOUNTED,
                        alpha=1.0, gamma=1.0, lmbda=0.0,      # Lambda parameter in non-adaptive TD(lambda) learners
                        alpha_min=0.0,
                        # Characteristics of the Fleming-Viot implementation
                        N=100, T=100,   # N is the number of particles, T is the max number of time steps allowed over ALL episodes in the single Markov chain simulation that estimates E(T_A)
                        absorption_set: Union[list, set]=None,
                        reset_method_value_functions=ResetMethod.ALLZEROS,
                        estimate_on_fixed_sample_size=False,
                        seed=1717, debug=False):
        env_shape = shape
        cls.debug = debug

        #-- Environment characteristics
        cls.nS = np.prod(env_shape)

        # Exit state defined as the top-right cell
        exit_state = env_shape[1] - 1
        terminal_states = set({exit_state})

        # Absorption set for FV
        if absorption_set is None:
            # We choose the first column, i.e. the column above the labyrinth's start state, of the 2D-grid as the set A of uninteresting states
            absorption_set = set()
            for i in range(env_shape[0]):
                state = np.ravel_multi_index((i, 0), env_shape)
                absorption_set.add(state)
        else:
            for state in absorption_set:
                if not 0 <= state < cls.nS:
                    raise ValueError(f"All states in the absorption set must be between 0 and {cls.nS-1}: {state}")

        # Activation set for FV
        # (defined from the absorption set as all those states adjacent to every state in the absorption set that are not part of the absorption set nor an obstacle)
        activation_set = set()
        for s in absorption_set:
            for sadj, dir in get_adjacent_states(env_shape, s):
                if sadj is not None and sadj not in set.union(absorption_set, obstacles_set):
                    activation_set.add(sadj)

        # Set of start states (the lower-left cell, if not given)
        if start_states_set is None:
            if absorption_set is None:  # ALTHOUGH ABOVE WE DEFINE THE absorption_set EVEN WHEN IT IS GIVEN AS None BY THE USER (so, this IF condition will never be satisfied --still we leave it for reference, and in case we decide to change this behaviour later on)
                # The set of start states is defined by just one state, namely the bottom-left of the environment, which is considered to be the entrance to the labyrinth
                start_states_set = set(np.ravel_multi_index((env_shape[0]-1, 0), env_shape))
            else:
                # We set the start state to the same set of states used by the FV learning so that TD and FV are more fairly compared)
                if learning_criterion == LearningCriterion.AVERAGE:
                    # The start state is defined as any state in the activation set of the FV learning process (i.e. the outside boundary of the absorption set)
                    start_states_set = set(activation_set)
                else:
                    # The start state is defined as any state in the complement of the absorption set (minus obstacles + terminal state),
                    # which is where the FV learning of the value functions start in the DISCOUNTED reward criterion.
                    start_states_set = set(np.arange(cls.nS)).difference(absorption_set.union(obstacles_set).union(terminal_states))
        else:
            if not start_states_set.issubset(set(np.arange(cls.nS))):
                raise ValueError(f"The start states set is invalid: all states in the set must be between 0 and {cls.nS-1}: {start_states_set}")
        # Define the initial state distribution that will be used when defining the environment class
        isd = np.zeros(cls.nS)
        for state in start_states_set:
            isd[state] = 1.0 / len(start_states_set)

        # Check the obstacles
        if obstacles_set is None:
            # Set just ONE obstacle in the gridworld, at the second column of the previous to last row
            state = np.ravel_multi_index((env_shape[0] - 2, 1), env_shape)
            obstacles_set = set({state})
        else:
            obstacles_set = set(obstacles_set)  # Convert to a set if not given as such
            for state in obstacles_set:
                if not 0 <= state < cls.nS:
                    raise ValueError(f"All states in the obstacles set must be between 0 and {cls.nS-1}: {state}")

        reward_terminal = +1
        reward_obstacles = 0
        dict_rewards = dict([(s, reward_terminal if s in terminal_states else reward_obstacles) for s in set.union(set(terminal_states), obstacles_set)])
        cls.env2d = gridworlds.EnvGridworld2D_WithObstacles(shape=env_shape, terminal_states=terminal_states,
                                                            rewards_dict=dict_rewards, obstacles_set=obstacles_set,
                                                            initial_state_distribution=isd)
        print("Gridworld environment:")
        cls.env2d._render()

        print("Environment characteristics (2D-labyrinth): (row, col)")
        print(f"Start states set: {[np.unravel_index(start_state, env_shape) for start_state in start_states_set]}")
        print(f"Exit state:  {np.unravel_index(exit_state, env_shape)}")
        print(f"Obstacles set:  {[np.unravel_index(s, env_shape) for s in sorted(obstacles_set)]}")
        print("")
        print("Fleming-Viot characteristics: (row, col)")
        print(f"Absorption set:  {[np.unravel_index(s, env_shape) for s in sorted(absorption_set)]}")
        print(f"Activation set:  {[np.unravel_index(s, env_shape) for s in sorted(activation_set)]}")

        #-- Cycle characteristics
        # Set of absorbing states, used to define a cycle as re-entrance into the set
        # which is used to estimate the average reward using renewal theory
        cls.A = absorption_set

        #-- Set where a particle activates in the FV context, used in the AVERAGE reward criterion (it should be touching A)
        cls.B = activation_set

        #-- Policy characteristics
        # Policy model
        if nn_input == InputLayer.SINGLE:
            cls.nn_model = nn_backprop(1, nn_hidden_layer_sizes, cls.env2d.getNumActions(), dict_activation_functions=dict({'hidden': [nn.ReLU]*len(nn_hidden_layer_sizes)}))
        else:
            cls.nn_model = nn_backprop(cls.env2d.getNumStates(), nn_hidden_layer_sizes, cls.env2d.getNumActions(), dict_activation_functions=dict({'hidden': [nn.ReLU] * len(nn_hidden_layer_sizes)}))
        cls.policy_nn = PolNN(cls.env2d, cls.nn_model)
        print(f"Neural network to model the policy:\n{cls.nn_model}")

        # Initialize the policy to a random walk
        cls.policy_nn.reset(initial_values=initial_policy)
        print(f"Network parameters initialized as follows:\n{list(cls.policy_nn.getThetaParameter())}")
        print("Initial policy for all states (states x actions):")
        policy = cls.policy_nn.get_policy_values()
        print(policy)

        #-- Plotting parameters
        cls.colormap = cm.get_cmap("jet")

        #-- Simulation setup
        cls.seed = seed
        # We use a large number of episodes (200) so that we can test that the results with MC and with TD(lambda) are similar and close to the true ones
        # at least when the MC learner is ready... which currently is not because both the state value and the average reward are based on episodic learning,
        # NOT on the average reward criterion.
        cls.nepisodes = 200

        #-- Learning parameters
        cls.gamma = gamma;
        cls.alpha = alpha
        cls.alpha_min = alpha_min
        cls.reset_method = reset_method_value_functions
        cls.reset_params = dict({'min': -1, 'max': +1})

        # TD(lambda) learner
        learner_tdlambda = td.LeaTDLambda(cls.env2d,
                                          criterion=learning_criterion,
                                          task=learning_task,
                                          gamma=gamma,
                                          lmbda=lmbda,
                                          alpha=cls.alpha,
                                          adjust_alpha=True,
                                          adjust_alpha_by_episode=False,
                                          alpha_min=cls.alpha_min,
                                          reset_method=cls.reset_method, reset_params=cls.reset_params, reset_seed=cls.seed,
                                          debug=cls.debug)
        cls.agent_nn_td = agents.GenericAgent(cls.policy_nn, learner_tdlambda)
        cls.sim_td = DiscreteSimulator(cls.env2d, cls.agent_nn_td, debug=False)

        # Adaptive TD(lambda) learner
        learner_tdlambda_adap = td.LeaTDLambdaAdaptive( cls.env2d,
                                                        criterion=learning_criterion,
                                                        task=learning_task,
                                                        gamma=gamma,
                                                        alpha=cls.alpha,
                                                        adjust_alpha=True,
                                                        adjust_alpha_by_episode=False,
                                                        alpha_min=cls.alpha_min,
                                                        reset_method=cls.reset_method, reset_params=cls.reset_params, reset_seed=cls.seed,
                                                        debug=cls.debug)
        cls.agent_nn_tda = agents.GenericAgent(cls.policy_nn, learner_tdlambda_adap)
        cls.sim_tda = DiscreteSimulator(cls.env2d, cls.agent_nn_tda, debug=False)

        # Fleming-Viot learner
        absorption_set = cls.A
        activation_set = cls.B
        learner_fv = fv.LeaFV(  cls.env2d,
                                N, T, absorption_set, activation_set,
                                probas_stationary_start_state_et=None,
                                probas_stationary_start_state_fv=None,
                                criterion=learning_criterion,
                                alpha=cls.alpha,
                                gamma=gamma,
                                lmbda=lmbda,
                                adjust_alpha=True,
                                adjust_alpha_by_episode=False,
                                alpha_min=cls.alpha_min,
                                reset_method=cls.reset_method, reset_params=cls.reset_params, reset_seed=cls.seed,
                                estimate_on_fixed_sample_size=estimate_on_fixed_sample_size,
                                debug=False)
        #actor_critic = LeaActorCriticNN(cls.env2d, cls.policy_nn, learner_fv, optimizer_learning_rate=0.1, seed=cls.seed, debug=True)
        #cls.agent_nn_fv = agents.GenericAgent(cls.policy_nn, dict({'value': learner_fv, 'policy': actor_critic}))
        cls.agent_nn_fv = agents.GenericAgent(cls.policy_nn, learner_fv)
        cls.sim_fv = DiscreteSimulator(cls.env2d, cls.agent_nn_fv, debug=False)

    def getAbsorptionSet(self):
        return self.A

    def getActivationSet(self):
        return self.B

    @classmethod
    def runSimulation(cls, agent, dict_params_simul, dict_params_learn, dict_params_info):
        "Runs the simulation that learns the optimum parameterized job acceptance policy for the queue"
        # Simulation object
        simul = DiscreteSimulator(cls.env2d, agent, log=False, save=False, debug=False)

        params = dict({
            '0(a)-System': "2D Gridworld with obstacles",
            '1(a)-System-#States': cls.env2d.getNumStates(),
            '1(b)-System-#Actions': cls.env2d.getNumActions(),
            '2(a)-Learning-Method': dict_params_learn['method'].name,
            '2(b)-Learning-Method#Particles': dict_params_simul['nparticles'],
            '2(c)-Learning-Method#TimeSteps/ArrivalEvents': dict_params_simul['t_sim'],
            '2(d)-Learning-Method#BurnInSteps (BITS)': dict_params_simul['burnin_time_steps'],
            '2(e)-Learning-Method#MinNumCycles': dict_params_simul['min_num_cycles_for_expectations'],
            '2(f)-Learning-MethodThetaStart': dict_params_simul['theta_start'],
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

    def no_test_Env_MetTD(self):
        # TODO: (2023/10/05) Implement the Actor-Critic policy gradient learning using a neural network for the policy parameter
        print("".join(np.repeat("*", 20)))
        print("\nRunning test " + self.id())
        print("Testing the TD learner algorithm...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'r.-'})

        # Simulation parameters
        dict_params_learn = dict({'method': "TD",
                                  'alpha_start': 10.0,
                                  'adjust_alpha': False,
                                  't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': 19,
            'theta_start': 1.1,     # This number should NOT be integer, o.w. the estimated Pr(K-1) will always be 0
            'nparticles': 1,
            't_sim': 30 * 50,
            'burnin_time_steps': 20,
            'min_num_cycles_for_expectations': 5,
            })
        # Make a copy of the seed used to run the process (for the assertion below),
        # because the key in dict_params_simul is changed by the process
        # as we need to consider different seeds for the different simulation parts and this seed is passed to each
        # simulator through the dict_params_simul dictionary and is used to store its value in the output files
        # to allow repeatability.
        seed = dict_params_simul['seed']

        # Run the simulation!
        df_learning = self.runSimulation(self.agent_nn_td, dict_params_simul, dict_params_learn, dict_params_info)
        print(df_learning)
        # Expected result when using MC with REINFORCE_TRUE strategy
        # (2023/02/02) Today these results have been carefully copied from the observed result of the test and checked that they are verified!
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [1.1, 5.941948, 6.389987, 6.269116, 6.716095]),
                                                ('theta_next', [5.941948, 6.389987, 6.269116, 6.716095, 7.517759]),
                                                ('proba_blocking', [0.188836, 0.001935, 0.028018, 0.016008, 0.008819]),
                                                ('error_proba_blocking', [-0.439294, -0.931983, -0.176295, -0.569493, -0.653667]),
                                                ('expected_reward', [-0.944180, -0.009676, -0.140091, -0.080039, -0.044097]),
                                                ('error_reward', [-1.560706, -1.068017, -1.823705, -1.430506, -1.346333]),
                                                ('alpha', [10.0]*5),
                                                ('gradV', [[0.48419478693402257], [0.04480394277065356], [-0.012087160327577427], [0.04469796225393303], [0.08016635442577162]]),
                                                ('n_events_mc', [1500]*5),
                                                ('n_events_fv', [0]*5),
                                                ('n_trajectories_Q', [91, 93, 95, 98, 93])
                                                ])
        # Check execution parameters
        assert  dict_params_learn['t_learn'] == 5
        assert  seed == 1717 and \
                dict_params_simul['theta_true'] == 19 and \
                dict_params_simul['theta_start'] == 1.1 and \
                dict_params_simul['nparticles'] == 1 and \
                dict_params_simul['t_sim'] == 30 * 50 and \
                dict_params_simul['burnin_time_steps'] == 20 and \
                dict_params_simul['min_num_cycles_for_expectations'] == 5

        # Check results
        assert_equal_data_frames(df_learning, df_learning_expected, df_learning_expected.columns, atol=1E-6)

    def no_test_Env_MetFVRL(self):
        # TODO: (2023/10/05) Implement the Actor-Critic policy gradient learning using a neural network for the policy parameter

        print("".join(np.repeat("*", 20)))
        print("\nRunning test " + self.id())
        print("Testing the FVRL algorithm on a 2D gridworld with obstacles...")

        # Light execution parameters
        dict_params_info = dict({'plot': False, 'log': False, 'symbol': 'g.-'})

        # Simulation parameters
        dict_params_learn = dict({  'method': "FVRL",
                                    'alpha_start': 10.0,
                                    'adjust_alpha': False,
                                    't_learn': 5})
        dict_params_simul = dict({
            'seed': 1717,
            'theta_true': 19,
            'theta_start': 1.1,     # This number should NOT be integer, o.w. the estimated Pr(K-1) will always be 0
            'nparticles': 30,
            't_sim': 50,
            'burnin_time_steps': 20,
            'min_num_cycles_for_expectations': 5,
            })
        # Make a copy of the seed used to run the process (for the assertion below),
        # because the key in dict_params_simul is changed by the process
        # as we need to consider different seeds for the different simulation parts and this seed is passed to each
        # simulator through the dict_params_simul dictionary and is used to store its value in the output files
        # to allow repeatability.
        seed = dict_params_simul['seed']

        # Run the simulation!
        df_learning = self.runSimulation(self.agent_nn_fv, dict_params_simul, dict_params_learn, dict_params_info)
        print(df_learning)
        # Expected result when using FVRL with REINFORCE_TRUE strategy
        # (2023/02/02) Today these results have been carefully copied from the observed result of the test and checked that they are verified!
        df_learning_expected = pd.DataFrame.from_items([
                                                ('theta', [1.1, 7.782423, 7.782423, 7.970238, 7.968525]),
                                                ('theta_next', [7.782423, 7.782423, 7.970238, 7.968525, 8.178420]),
                                                ('proba_blocking', [0.260614, 0.0, 0.008446, 0.000092, 0.000374]),
                                                ('error_proba_blocking', [-0.226164, -1.0, -0.484370, -0.992941, -0.971317]),
                                                ('expected_reward', [-1.303072, -0.0, -0.042228, -0.000459, -0.001868]),
                                                ('error_reward', [-1.773836, -1.0, -1.515628, -1.007059, -1.028683]),
                                                ('alpha', [10.0]*5),
                                                ('gradV', [[0.6682422813177087], [0.0], [0.018781526951805034], [-0.00017128367010005866], [0.020989455330044243]]),
                                                ('n_events_mc', [87, 100, 99, 99, 99]),
                                                ('n_events_fv', [431, 0, 1573, 561, 688]),
                                                ('n_trajectories_Q', [91, 0, 93, 90, 82])
                                                ])
        # Check execution parameters
        assert  dict_params_learn['t_learn'] == 5
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


if __name__ == "__main__":
    # Reference for creating test suites:
    # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
    runner = unittest.TextTestRunner()

    # Run all tests
    # unittest.main()

    #-- Create the test suites
    # 1) 2D gridworld
    test_suite_gridworld2d = unittest.TestSuite()
    test_suite_gridworld2d.addTest(Test_EstPolicy_EnvGridworldsWithObstacles("test_Env_MetTD"))
    test_suite_gridworld2d.addTest(Test_EstPolicy_EnvGridworldsWithObstacles("test_Env_MetFVRL"))

    #-- Run the test suites
    #runner.run(test_suite_gridworld2d)
