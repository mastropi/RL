# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:00:15 2021

@author: Daniel Mastropietro
"""

#import runpy
#runpy.run_path('../../setup.py')


########
# 2023/10/12: Learn an actor-critic policy using neural networks (with the torch package)
# Learning happens with the ActorCriticNN learner which defines a loss of type `tensor` which can be minimized using the backward() method of torch Tensors
# IT WORKS!

#-------------------- IMPORT AND AUXILIARY FUNCTIONS ------------------#
from timeit import default_timer as timer
import os
from enum import Enum, unique
import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt, cm
from matplotlib.ticker import MaxNLocator
from scipy.special import rel_entr

from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners import LearningCriterion, LearningTask
from Python.lib.agents.learners.policies import LeaActorCriticNN
from Python.lib.agents.learners.value_functions import ActionValueFunctionApproxNN, StateValueFunctionApproxNN
from Python.lib.agents.policies import probabilistic

from Python.lib.environments.gridworlds import Direction2D
from Python.lib.estimators.nn_models import InputLayer

from Python.lib.utils.basic import get_current_datetime_as_string, load_objects_from_pickle, log_file_open, log_file_close, save_objects_to_pickle
from Python.lib.utils.computing import compute_expected_reward, compute_transition_matrices, compute_state_value_function_from_transition_matrix

from Python.test.test_optimizers_discretetime import Test_EstPolicy_EnvGridworldsWithObstacles, Test_EstPolicy_EnvMountainCar

# When saving results or reading previously saved results
rootdir = "./RL-003-Classic"
resultsdir = f"./{rootdir}/results"
logsdir = f"./{rootdir}/logs"

# Types of environments that can be defined
@unique
class Environment(Enum):
    Gridworld = 1
    MountainCar = 2


#--- Auxiliary functions
KL_THRESHOLD = 0.005
policy_changed_from_previous_learning_step = lambda KL_distance, num_states: np.abs(KL_distance) / num_states > KL_THRESHOLD

def define_plotting_parameters():
    dict_colors = dict(); dict_linestyles = dict(); dict_legends = dict()
    dict_colors['all_online'] = "darkred"; dict_linestyles['all_online'] = "solid"; dict_legends['all_online'] = "ALL Online"
    dict_colors['values_td'] = "red"; dict_linestyles['values_td'] = "solid"; dict_legends['values_td'] = "TDAC"      # For a second trial of TDAC (which sometimes is useful --e.g. to compare "using the same budget as FV for every policy learning step" vs. "using an average budget at each policy learning step")
    dict_colors['values_td2'] = "darkorange"; dict_linestyles['values_td2'] = "solid"; dict_legends['values_td2'] = "TDAC2"
    dict_colors['values_tdl'] = "orange"; dict_linestyles['values_tdl'] = "dashed"; dict_legends['values_tdl'] = "TDAC(lambda)"
    dict_colors['values_fv'] = "green"; dict_linestyles['values_fv'] = "solid"; dict_legends['values_fv'] = "FVAC"    # For a second trial of FVAC (which sometimes is useful --e.g. to compare "allowing infinite budget for FV" vs. "limited budget")
    dict_colors['values_fv2'] = "cyan"; dict_linestyles['values_fv2'] = "solid"; dict_legends['values_fv2'] = "FVAC2"
    dict_colors['values_fvos'] = "lightgreen"; dict_linestyles['values_fvos'] = "solid"; dict_legends['values_fvos'] = "FVAC OverSampling"

    figsize = (10, 8)

    return dict_colors, dict_linestyles, dict_legends, figsize

def compute_true_state_value_function(env, policy, learning_task, learning_criterion, gamma=1.0, atol=1E-6):
    #*********************************** IMPORTANT **********************************#
    # This function may give a bad value of the true state value function for policies close to deterministic.
    # This would be most likely due to the ill-conditioning of the transition matrix P derived from the policy and the transition probabilities of the environment
    # when eigenvalues are computed.
    # For more details and possible fix see function computing.compute_state_value_function_from_transition_matrix().
    #*********************************** IMPORTANT **********************************#
    # TODO: (2024/05/07) Try to solve the above problem of instability in the calculation of the state value function
    """
    Computes the true state value function for the given policy applied on the given environment
    under the given learning task, learning criterion and discount factor gamma.

    It also computes the expected reward under stationarity for the CONTINUING learning task (with no discount applied of course).

    The state value function is stored in the environment object, so that it can be used as reference for comparing the estimated state value function.

    Arguments:
    env: EnvironmentDiscrete
        Environment with discrete states and discrete actions with ANY initial state distribution.
        Rewards can be anywhere.

    policy: policy object with method getPolicyForAction(a, s) defined, returning Prob(action | state)
        Policy object acting on a discrete-state / discrete-action environment.
        This could be of class e.g. random_walks.PolRandomWalkDiscrete, probabilistic.PolGenericDiscrete, PolNN.

    gamma: (opt) float in (0, 1]
        Discount factor for the observed rewards.
        default: 1.0

    Return: tuple
    Tuple containing the following 3 elements:
    - V_true: the true state value function for the given learning task, learning criterion and discount gamma.
    - expected_reward: the expected reward for the CONTINUING learning task.
    - mu: the stationary probability for the CONTINUING learning task.
    """
    P_epi, P_con, b_epi, b_con, g, mu = compute_transition_matrices(env, policy, atol=atol)
    P = P_con if learning_task == LearningTask.CONTINUING else P_epi
    b = b_con if learning_task == LearningTask.CONTINUING else b_epi
    bias = g if learning_criterion == LearningCriterion.AVERAGE else None
    V_true = compute_state_value_function_from_transition_matrix(P, b, bias=bias, gamma=gamma)
    env.setV(V_true)
    dict_proba_stationary = dict(zip(np.arange(len(mu)), mu))
    avg_reward_true = compute_expected_reward(env, dict_proba_stationary)
    return V_true, avg_reward_true, mu

def compute_max_avg_rewards_in_labyrinth_with_corridor(env, wind_dict, learning_task, learning_criterion):
    """
    Computes the maximum CONTINUING and EPISODIC average rewards for the given environment

    The computation of the episodic average reward is approximate when the environment has wind, because its value is given as:
        max_avg_reward_continuing * L / (L-1)
    where L is the length of the shortest path, including the start and exit state.

    Return: tuple
    Tuple with the following two elements:
    - max_avg_reward_continuing
    - max_avg_reward_episodic
    """
    env_shape = env.getShape()
    entry_state = np.argmax(env.getInitialStateDistribution())
    exit_state = env.getTerminalStates()[0]

    max_avg_reward_continuing = np.nan
    max_avg_reward_episodic = np.nan
    if wind_dict is None:
        # The average rewards are computed as the inverse of the shortest path from Start to Exit in Manhattan-like movements,
        # - INCLUDING THE START STATE for continuing learning tasks (just think about it, we restart every time we reach the terminal state with reward = 0)
        # - EXCLUDING THE START STATE for episodic learning tasks (just think about it)
        # *** WARNING: This calculation is problem dependent and should be adjusted accordingly ***
        print("\nComputing MAX average in DETERMINISTIC environment...")
        if exit_state == entry_state + env_shape[1] - 1:
            # This is the case when the exit state is at the bottom-right of the labyrinth
            max_avg_reward_continuing = 1 / env_shape[1]        # In this case the start state counts! (we subtract 1 because the bottom-right state must NOT count twice!)
            max_avg_reward_episodic = 1 / (env_shape[1] - 1)    # In this case the start state does not count (we subtract 2 because, besides not counting twice the bottom-right state, the start state does not count in the episodic setting)
        elif exit_state == env_shape[1] - 1:
            # This is the case when the exit state is at the top-right of the labyrinth (the default)
            max_avg_reward_continuing = 1 / (np.sum(env_shape) - 1)  # In this case the start state counts! (we subtract 1 because the bottom-right state must NOT count twice!)
            max_avg_reward_episodic = 1 / (np.sum(env_shape) - 2)    # In this case the start state does not count (we subtract 2 because, besides not counting twice the bottom-right state, the start state does not count in the episodic setting)
    else:
        # There is wind in the environment
        # => Compute the average reward using the transition matrix information of the environment (as the wind makes things more complicated)
        # together with the optimal policy used by the agent (which we know)
        print(f"\nComputing MAX average in STOCHASTIC environment with WIND: {wind_dict}...")
        if exit_state == entry_state + env_shape[1] - 1:
            policy_optimal = probabilistic.PolGenericDiscrete(env, policy=dict(), policy_default=[0.0, 1.0, 0.0, 0.0])
            V_true, avg_reward_true, mu = compute_true_state_value_function(env, policy_optimal, learning_task, learning_criterion)
            max_avg_reward_continuing = avg_reward_true
            # Compute the approximated episodic average reward (it is approximated because of the randomness generated by the wind.
            # In order to compute the exact episodic average reward, we need to think more.
            # Here, we use the formula that relates the continuing average reward with the episodic one in the deterministic environment case.
            max_avg_reward_episodic = avg_reward_true * env_shape[1] / (env_shape[1] - 1)
        elif exit_state == env_shape[1] - 1:
            # This is the case when the exit state is at the top-right of the labyrinth (the default)
            rightmost_states = [np.ravel_multi_index((r, env_shape[1] - 1), env_shape) for r in np.arange(env_shape[0])]
            policy_optimal = probabilistic.PolGenericDiscrete(env, policy=dict(zip(rightmost_states, [[1.0, 0.0, 0.0, 0.0]] * len(rightmost_states))),
                                                              policy_default=[0.0, 1.0, 0.0, 0.0])
            V_true, avg_reward_true, mu = compute_true_state_value_function(env, policy_optimal, learning_task, learning_criterion)
            max_avg_reward_continuing = avg_reward_true
            # Compute the approximated episodic average reward (it is approximated because of the randomness generated by the wind.
            # In order to compute the exact episodic average reward, we need to think more.
            # Here, we use the formula that relates the continuing average reward with the episodic one in the deterministic environment case.
            max_avg_reward_episodic = avg_reward_true * np.sum(env_shape) / (np.sum(env_shape) - 1)

    return max_avg_reward_continuing, max_avg_reward_episodic

def compute_prob_states(state_counts, probas_stationary=None):
    """
    Computes the state probability based on the state visit count plus potentially a separate estimate for a subset of states,
    e.g. for the active states in Fleming-Viot

    Arguments:
    state_counts: array-like
        State counts for ALL states in the environment.

    probas_stationary: (opt) dict
        Dictionary containing the stationary probabilities estimated separately than the state counts, e.g. by Fleming-Viot.

    Return: list
    State-indexed list containing the estimated state probability for ALL states in the environment (which are assumed included in the
    input list `state_counts`).
    """
    # Initial estimate of the state probabilities,
    # which is the final estimate for classical (non-FV) estimation.
    prob_states = state_counts / np.sum(state_counts)
    num_states = len(state_counts)

    if probas_stationary is not None:
        # Estimation based on E(T) + FV simulation done in the FV-based estimation of stationary probabilities
        # Compute the constant value by which the FV-based estimation of the stationary probabilities outside A will be scaled
        states_active_set = probas_stationary.keys()
        proba_active_set = np.sum(list(probas_stationary.values()))
        proba_absorption_set = np.sum([prob_states[s] for s in range(num_states) if s not in states_active_set])
        assert proba_active_set > 0, f"The probability of the active set must be positive (because FV particles spent some time at least at one of those states: {proba_active_set}"
        for state, p in probas_stationary.items():
            prob_states[state] = p / proba_active_set * (1 - proba_absorption_set)
    assert np.isclose(np.sum(prob_states), 1), f"The estimated state probabilities must sum to 1: {np.sum(prob_states)}"

    return prob_states
#--- Auxiliary functions
#-------------------- IMPORT AND AUXILIARY FUNCTIONS ------------------#


#----------------- BASIC SETUP AND SIMULATION PARAMETERS --------------#
# Learning task and learning criterion are used by the constructor of the test class below
learning_task = LearningTask.CONTINUING
#learning_criterion = LearningCriterion.DISCOUNTED; gamma = 0.9
learning_criterion = LearningCriterion.AVERAGE; gamma = 1.0    # gamma could be < 1 in the average reward criterion in order to take the limit as gamma -> 1 as presented in Sutton, pag. 251/252.

seed = 1317
#env_type = Environment.Gridworld
env_type = Environment.MountainCar
problem_2d = True
use_random_obstacles_set = False; prop_obstacles = 0.5; seed_obstacles = 4217 #4215    # Seed 4217 with 50% of obstacles gives good results in the 6x8 labyrinth
exit_state_at_bottom = False
estimate_absorption_set = True; threshold_absorption_set = 0.90 #0.90  # Cumulative relative visit threshold
estimate_absorption_set_at_every_step = False
entry_state_in_absorption_set = True   #False #True     # Only used when estimate_absorption_set = False
#----------------- BASIC SETUP AND SIMULATION PARAMETERS --------------#


#-------------------------------- ENVIRONMENT -------------------------#
if env_type == Environment.Gridworld:
    if problem_2d:
        # 2D labyrinth
        size_vertical = 3; size_horizontal = 4
        size_vertical = 4; size_horizontal = 5
        #size_vertical = 6; size_horizontal = 8
        #size_vertical = 8; size_horizontal = 12
        #size_vertical = 9; size_horizontal = 13
        #size_vertical = 10; size_horizontal = 14
        #size_vertical = 10; size_horizontal = 30

        # Whether the active set in FV should be connected (in order to avoid isolation of the two activation states and reduce possible problems)
        # When this is the case, the connectedness of the active set is achieved by removing the left-most obstacle in the previous-to-bottom row.
        connected_active_set = True
        initial_policy = None #[0.4, 0.15, 0.05, 0.4] #None   # Random walk as initial policy when None
    else:
        # 1D gridworld: the interesting dimension is the vertical dimension, with terminal state at the top (this is to unify the structure of 2D labyrinth and 1D gridworld)
        size_vertical = 21; size_horizontal = 1     # We choose a value like K = 20 or 40 in the M/M/1/K queue system
        initial_policy = [0.4, 0.0, 0.6, 0.0]       # We choose a probability of going down which is similar to mu/(lambda + mu) in the queue system where lambda/mu = 0.7, i.e. a value close to 1 / 1.7 = 0.588
    env_shape = (size_vertical, size_horizontal)
    nS = size_vertical * size_horizontal

    # Environment's entry and exit states
    entry_state = np.ravel_multi_index((size_vertical - 1, 0), env_shape)
    exit_state = entry_state + env_shape[1] - 1 if exit_state_at_bottom else None   # None means that the EXIT state is set at the top-right of the labyrinth

    # Presence of wind: direction and probability of deviation in that direction when moving
    if problem_2d:
        wind_dict = None
        #wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.5})
        #wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.6})
        #wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.7})
        #wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.8})
    else:
        # WIND is currently not allowed in 1D gridworlds
        wind_dict = None

    # Obstacles
    if problem_2d:
        #-- 2D labyrinth
        if use_random_obstacles_set:
            obstacles_set = None
            n_obstacles = int(prop_obstacles * np.prod(env_shape))
        else:
            # The path to the terminal state is just a corridor through the last row right and then the last column up
            # So at the upper left corner there is a rectangle that brings to nowhere
            rectangle_to_nowhere_width = size_horizontal - 2
            rectangle_to_nowhere_height = size_vertical - 2
            states_previous_to_last_row = np.ravel_multi_index([np.repeat(rectangle_to_nowhere_height, rectangle_to_nowhere_width), [y for y in range(1, rectangle_to_nowhere_width+1)]], env_shape)
            states_previous_to_last_column = np.ravel_multi_index([[x for x in range(0, rectangle_to_nowhere_height+1)], np.repeat(rectangle_to_nowhere_width, rectangle_to_nowhere_height+1)], env_shape)
            obstacles_set = set(np.concatenate([list(states_previous_to_last_row) + list(states_previous_to_last_column)]))

            if connected_active_set:
                obstacles_set = obstacles_set.difference({min(states_previous_to_last_row)})
            n_obstacles = len(obstacles_set)
    else:
        #-- 1D gridworld
        obstacles_set = set()
else:
    entry_state = None  # `None` so that the start state is defined by the reset method of the environment
    wind_dict = None    # Just for information purposes in titles, etc.
#-------------------------------- ENVIRONMENT -------------------------#


#----------------------------- MODEL FOR POLICY -----------------------#
# Number of input neurons (just one with the state value or one-per-state)
nn_input = InputLayer.ONEHOT
# Number of hidden layers in the neural network model
# Using multiple layers whose size is proportional to the gridworld size... however this tends to be counterproductive...
# i.e. learning is slower and may fail (e.g. it usually converges to a non-optimal policy where the advantage function is 0), presumably because of the larger number of parameters.
# Perhaps the architecture would work if the learning step of the neural network parameters optimizer is smaller and
# a larger number of policy learning steps is used (I tried with optimizer_learning_rate = 0.01 instead of 0.1 and
# the parameters started to be learned (i.e. the average reward went up --as opposed to 0-- although with large oscillations)
# but still 30 learning steps did not suffice to learn completely.
# This was tried with the 6x8 gridworld with a large rectangle of states going nowhere at the upper-left part
# with adaptive TD(lambda), where the hidden layer sizes were set to [38, 19].
#nn_hidden_layer_sizes = [int( 0.8*np.prod(env_shape) ), int( 0.4*np.prod(env_shape) )]
# Keep the neural network rather small or do NOT use any hidden layer for Natural Policy Gradient (NPG)
nn_hidden_layer_sizes = [12]  #[] #[12]
print(f"Neural Network architecture:\n{len(nn_hidden_layer_sizes)} hidden layers of sizes {nn_hidden_layer_sizes}")
#----------------------------- MODEL FOR POLICY -----------------------#


#----------------------------- FV ABSORPTION SET ----------------------#
if env_type == Environment.Gridworld:
    # DEFAULT absorption set used when estimate_absorption_set = False
    if problem_2d and not use_random_obstacles_set:
        # The absorption set is a rectangular area at the upper left corner of the grid + (possibly) the lower left corner
        #lower_left_state = (size_vertical-1) * size_horizontal
        #default_absorption_set = set(np.concatenate([list(range(x*size_horizontal, x*size_horizontal + size_horizontal-2)) for x in range(size_vertical-2)]))
        # The absorption set is a rectangular area that touches the right and bottom walls of the big rectangular area that leads to nowhere
        left_margin = 0 #int(rectangle_to_nowhere_width/2)
        top_margin = max(1, int(rectangle_to_nowhere_width/2) - 1) #rectangle_to_nowhere_height #rectangle_to_nowhere_height - 2
        default_absorption_set = set(np.concatenate([list(range(x * size_horizontal + left_margin, x * size_horizontal + rectangle_to_nowhere_width)) for x in range(0, top_margin)]))
    else:
        # We choose an absorption set size J ~ K/3, like in the M/M/1/K queue system (recall that numbering of states here is the opposite than in M/M/1/K
        default_absorption_set = set(np.arange(np.prod(env_shape) * 2 // 3 + 1, env_shape[0] - 1))

    # Add the environment's start state to the absorption set
    if entry_state_in_absorption_set:
        default_absorption_set.add(entry_state)
#----------------------------- FV ABSORPTION SET ----------------------#


#-------------------------------- TEST SETUP --------------------------#
if env_type == Environment.Gridworld:
    dropout_policy = 0.0  #0.5 #0.5  # Set it to 0.0 if we do not want any dropout layer in the network
    test_ac = Test_EstPolicy_EnvGridworldsWithObstacles()
    test_ac.setUpClass(shape=env_shape, obstacles_set=obstacles_set, n_obstacles=n_obstacles, wind_dict=wind_dict,
                       define_start_state_from_absorption_set=False, start_states_set={entry_state},  #{nS-1}, #None,
                       exit_state=exit_state,
                       # Policy model
                       nn_input=nn_input, nn_hidden_layer_sizes=nn_hidden_layer_sizes, initial_policy=initial_policy, dropout_policy=dropout_policy,
                       # General learning parameters
                       learning_task=learning_task,
                       learning_criterion=learning_criterion,
                       alpha=1.0, gamma=gamma, lmbda=0.7,  # lmbda parameter is ONLY used for TD(lambda), NOT for TD(0), which is created separately nor for FV (for which lambda > 0 does not make sense)
                       alpha_min=0.1,
                       reset_method_value_functions=ResetMethod.ALLZEROS,
                       # Fleming-Viot parameters
                       # Small N and T are N=50, T=1000 for the 8x12 labyrinth with corridor
                       N=20,  #50, #200, #200 if problem_2d else 100, #50 #20 #100
                       T=500,  #1000, #100, #10000 if problem_2d else 1000, #1000, #1000, #3000,  # np.prod(env_shape) * 10  #100 #1000
                       estimate_absorption_set=estimate_absorption_set, threshold_absorption_set=threshold_absorption_set, absorption_set=default_absorption_set,
                       states_of_interest_fv=None,  # exit_state,
                       seed=seed, plot=True, debug=False, seed_obstacles=seed_obstacles)
    test_ac.setUp()
    print(test_ac.policy_nn.nn_model)
elif env_type == Environment.MountainCar:
    env_discrete = True
    if env_discrete:
        dict_function_approximations = None
    else:
        dropout_value_functions = 0.0  #0.5           # Set it to 0.0 if we do not want any dropout layer in the network
        learning_rate_value_functions = 0.001 if dropout_value_functions == 0.0 else 0.01  # We increase the learning rate when there is dropout. Ref: https://machinelearningmastery.com/using-dropout-regularization-in-pytorch-models/ (conclusions)
        nn_hidden_layer_sizes_value_functions = nn_hidden_layer_sizes if len(nn_hidden_layer_sizes) > 0 else [12]
        dict_function_approximations = dict({'V': StateValueFunctionApproxNN(nn_input=2, nn_hidden_layer_sizes=nn_hidden_layer_sizes_value_functions, dropout=dropout_value_functions, lr=learning_rate_value_functions),
                                             'Q': ActionValueFunctionApproxNN(nn_input=2 + 1, nn_hidden_layer_sizes=nn_hidden_layer_sizes_value_functions, dropout=dropout_value_functions, lr=learning_rate_value_functions)})
                                                ## Number of inputs for Q: (x, v, a)
    dropout_policy = 0.0  #0.5
    initial_policy = [1/3, 1/3, 1/3]
    test_ac = Test_EstPolicy_EnvMountainCar()
    test_ac.setUpClass(# Environment characteristics
                       env_discrete=env_discrete,
                       nx=40,                            # Number of points in the discretization of the positions (only used in the continuous-state-dynamic Mountain Car, in which case the "factor for force" parameter is not used)
                       nv=21,                            # Number of points in the discretization of the velocities
                       factor_for_force_and_gravity=10 if not env_discrete else 100, #100, #20, #15,   # Factor controlling the number of discrete positions in the discretized problem --> NOTE: Using `1` is TOO SMALL! (as the number of points in the grid are too many)
                       factor_force=1.0,
                       factor_max_speed=3.0,    # Only used in MountainCarDiscrete (with continuous states)
                       # Value function approximations model
                       dict_function_approximations=dict_function_approximations,
                       # Policy model
                       nn_input=2, nn_hidden_layer_sizes=nn_hidden_layer_sizes, dropout_policy=dropout_policy,
                       initial_policy=initial_policy,
                       # General learning parameters
                       learning_task=learning_task,
                       learning_criterion=learning_criterion,
                       alpha=1.0, gamma=gamma, lmbda=0.7, # lmbda parameter is ONLY used for TD(lambda), NOT for TD(0), which is created separately nor for FV (for which lambda > 0 does not make sense)
                       alpha_min=0.1,
                       reset_method_value_functions=ResetMethod.ALLZEROS,
                       reset_value=0.0, #-1.0,
                       N=30,  #50,
                       T=300,  #500,
                       threshold_absorption_set=threshold_absorption_set,
                       seed=seed, plot=True, debug=False)
    test_ac.setUp()
    print(test_ac.policy_nn.nn_model)
    env_shape = test_ac.getEnv().getShape()
    print(f"Mountain Car environment with shape {env_shape}:")
    print(f"- Interval for the {len(test_ac.getEnv().getPositions())} positions: [{test_ac.getEnv().getPositions()[0]}, {test_ac.getEnv().getPositions()[-1]}]")
    print(f"- Interval for the {len(test_ac.getEnv().getVelocities())} velocities: [{test_ac.getEnv().getVelocities()[0]}, {test_ac.getEnv().getVelocities()[-1]}]")
    print(f"- Positions: {test_ac.getEnv().getPositions()} ({len(test_ac.getEnv().getPositions())})")
    print(f"- Velocities: {test_ac.getEnv().getVelocities()} ({len(test_ac.getEnv().getVelocities())})")
# Shape as string, used for filenames
shape_str = f"{env_shape[0]}x{env_shape[1]}"

# List the states in the absorption and activation sets
absorption_set = set([tuple(test_ac.getEnv().getStateIndicesFromIndex(s)) for s in test_ac.getAbsorptionSet()])
activation_set = set([tuple(test_ac.getEnv().getStateIndicesFromIndex(s)) for s in test_ac.getActivationSet()])
print(f"Absorption set (1D) (n={len(test_ac.getAbsorptionSet())}):\n{test_ac.getAbsorptionSet()}")
print(f"Absorption set (2D) (n={len(absorption_set)}):\n{absorption_set}")
print(f"Activation set (1D) (n={len(test_ac.getActivationSet())}):\n{test_ac.agent_nn_fv.getLearner().getActivationSet()}")
print(f"Activation set (2D) (n={len(activation_set)}):\n{activation_set}")

# Check that NO state in the absorption set receives rewards
# THIS IS IMPORTANT AT THIS POINT because the estimation of the average reward currently implemented in discrete.Simulator._run_simulation_fv()
# is NOT prepared for absorption sets A that contain states with rewards. It still needs to be implemented and it is not so straightforward
# as the computation of the average reward should be changed in several places (e.g. discrete.Learner._update_average_reward() which should receive the information
# of the cycle set parameter (`set_cycle`) received by _run_single_continuing_task() in order to split the average reward into two pieces:
# one for the states in the cycle set (which is the absorption set A in the FV simulation) and one for the states OUTSIDE the cycle set,
# so that we can use the average reward of the states OUTSIDE the cycle set as initial estimation of the FV average reward on one side,
# and the average reward of the states in the cycle set as contribution to the FINAL average reward estimated by the whole FV process,
# which has a contribution from the rewards received by the states in A and the rewards received by the states OUTSIDE A.
#
# NOTE: (2024/08/27) This may fail for MountainCarDiscrete environment because the discretized states may contain *continuous* states that are part of the goal,
# therefore they have rewards. Note that the reward assigned to such discretize states is proportional to the overlap of the discretized cell in the position direction
# with continuous-valued terminal states (see definition of self.rewards in the MountainCarDiscrete class).
assert np.sum([state for state in absorption_set if test_ac.getEnv().getReward(state) != 0]) == 0, \
        "No state in the absorption set A must receive rewards! (at least until we prepare the FV estimation process to take into account rewards observed in A)"
#-------------------------------- TEST SETUP --------------------------#


#------------------ INITIAL AND MAXIMUM AVERAGE REWARDS ---------------#
# Average reward of random policy
if env_type == Environment.Gridworld:
    policy_random = probabilistic.PolGenericDiscrete(test_ac.getEnv(), policy=dict(), policy_default=[0.25, 0.25, 0.25, 0.25])
    V_true, avg_reward_true, mu = compute_true_state_value_function(test_ac.getEnv(), policy_random, learning_task, learning_criterion)
    print(f"Average reward under RANDOM policy: {avg_reward_true}")
    if initial_policy is not None:
        policy_initial = probabilistic.PolGenericDiscrete(test_ac.getEnv(), policy=dict(), policy_default=initial_policy)
        V_true, avg_reward_true, mu = compute_true_state_value_function(test_ac.getEnv(), policy_random, learning_task, learning_criterion)
        print(f"Average reward under INITIAL policy: {avg_reward_true}")

    print(f"True state value function under initial policy:\n{test_ac.getEnv().getV()}")

    # Maximum average reward (to use as reference in information and plots)
    if not use_random_obstacles_set:
        max_avg_reward_continuing, max_avg_reward_episodic = compute_max_avg_rewards_in_labyrinth_with_corridor(test_ac.getEnv(), wind_dict, learning_task, learning_criterion)
    else:
        max_avg_reward_continuing = np.nan
        max_avg_reward_episodic = np.nan
    print(f"MAX CONTINUING average reward: {max_avg_reward_continuing}")
    print(f"MAX EPISODIC average reward (approximate when wind present): {max_avg_reward_episodic}")

    # State to observe and e.g. plot Q values, etc
    if exit_state == entry_state + env_shape[1] - 1:
        state_observe = np.ravel_multi_index((env_shape[0] - 1, env_shape[1] - 2), env_shape)
    else:
        state_observe = np.ravel_multi_index((1, env_shape[1] - 1), env_shape)
else:
    state_observe = None
    max_avg_reward_continuing = np.nan
    max_avg_reward_episodic = np.nan
#------------------ INITIAL AND MAXIMUM AVERAGE REWARDS ---------------#


#------------------ RESULTS COLLECTION AND PLOTS SETUP ----------------#
# Learning methods: define characteristics to use when plotting results for each method
dict_colors, dict_linestyles, dict_legends, figsize = define_plotting_parameters()

# Dictionaries to store the results for the different methods (for comparison purposes)
dict_loss = dict()
dict_R = dict()
dict_R_long = dict()
dict_R_long_true = dict()   # True average reward under the policy used at each policy learning step to learn value functions. GOAL: Keep track on how rare is reaching the reward.
dict_V = dict()
dict_Q = dict()
dict_A = dict()
dict_state_counts = dict()
dict_nsteps = dict()
dict_KL = dict()
dict_alpha = dict()
dict_time_elapsed = dict()
#------------------ RESULTS COLLECTION AND PLOTS SETUP ----------------#


# Number of replications to run on each method
nrep = 10 #5 #10
# Logging
log = True #True #False #learning_method_type == "values_fv"

# Learning method (of the value functions and the policy)
# Both value functions and policy are learned online using the same simulation
learning_method = "all_online"; simulator_value_functions = None
# Value functions are learned separately from the policy
# Policy learning can happen online or OFFLINE
learning_method = "values_td"; simulator_value_functions = test_ac.sim_td0      # TD(0)
#learning_method = "values_td2"; simulator_value_functions = test_ac.sim_td0    # TD(0)
#learning_method = "values_tdl"; simulator_value_functions = test_ac.sim_td     # TD(lambda)
#learning_method = "values_tda"; simulator_value_functions = test_ac.sim_tda    # Adaptive TD(lambda)
learning_method = "values_fv"; simulator_value_functions = test_ac.sim_fv
#learning_method = "values_fv2"; simulator_value_functions = test_ac.sim_fv
#learning_method = "values_fvos"; simulator_value_functions = test_ac.sim_fv     # FV used just as an oversampling method

learning_method_type = learning_method[:9]  # This makes e.g. "values_fvos" become "values_fv"

# FV learning parameters (which are used to define parameters of the other learners analyzed so that their comparison with FV is fair)
max_time_steps_fv_per_particle = 5*len(test_ac.agent_nn_fv.getLearner().getActiveSet()) #100 #50                            # Max average number of steps allowed for each particle in the FV simulation. We set this value proportional to the size of the active set of the FV learner, in order to scale this value with the size of the labyrinth.
max_time_steps_fv_for_expectation = test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()                        # This is parameter T
max_time_steps_fv_for_all_particles = test_ac.agent_nn_fv.getLearner().getNumParticles() * max_time_steps_fv_per_particle   # This is parameter N * max_time_steps_fv_per_particle which defines the maximum number of steps to run the FV system when learning the value functions that are used as critic of the loss function at each policy learning step
max_time_steps_fv_overall = 3*max_time_steps_fv_for_all_particles #max(5000, max_time_steps_fv_for_all_particles)            # To avoid too large simulation times, for instance when the policy is close to optimal: ABSOLUTE maximum number of steps allowed for the FV simulation, regardless of what happens with particle absorption (i.e. if all of them are absorbed or 90% of them are absorbed at least once)

# Traditional method learning parameters
# They are set for a fair comparison with FV learning
# The maximum time steps to be observed in the benchmark methods is set to the sum of:
# - the number of steps used to estimate the absorption set A (as long as this exploration ALSO estimates value functions!)
# - the max number of time steps allowed to estimate E(T_A)
# - the max number of time steps allowed over all the FV particles
# Values previously used: #2000 #test_ac.getEnv().getNumStates()*10
# Number of steps used to estimate the absorption set A
time_steps_fv_for_absorption_set = test_ac.learner_for_initial_exploration.getNumSteps() if test_ac.learner_for_initial_exploration is not None else 0
assert time_steps_fv_for_absorption_set > 0 if estimate_absorption_set else None
max_time_steps_benchmark = time_steps_fv_for_absorption_set + max_time_steps_fv_for_expectation + max_time_steps_fv_for_all_particles

#-- Common learning parameters
# Parameters about policy learning (Actor)
is_NPG = len(nn_hidden_layer_sizes) == 0
n_learning_steps = 50 #200 #50 #100 #30
n_episodes_per_learning_step = 50   #100 #30   # This parameter is used as the number of episodes to run the policy learning process for and, if the learning task is EPISODIC, also as the number of episodes to run the simulators that estimate the value functions
max_time_steps_per_policy_learning_episode = 5*test_ac.getEnv().getNumStates() if problem_2d else 2*test_ac.getEnv().getNumStates() #np.prod(env_shape) * 10 #max_time_steps_benchmark // n_episodes_per_learning_step   # Maximum number of steps per episode while LEARNING THE *POLICY* ONLINE (NOT used for the value functions (critic) learning)
policy_learning_mode = "online" #"offline" #"online"     # Whether the policy is learned online or OFFLINE (only used when value functions are learned separately from the policy)
allow_deterministic_policy = True #False
use_average_reward_from_previous_step = True #learning_method_type == "values_fv" #False #True            # Under the AVERAGE reward crtierion, whether to use the average reward estimated from the previous policy learning step as correction of the value functions (whenever it is not 0), at least as an initial estimate
use_advantage = not (learning_method == "values_fvos") # Set this to True if we want to use the advantage function learned as the TD error, instead of using the advantage function as the difference between the estimated Q(s,a) and the estimated V(s) (where the average reward cancels out)
optimizer_learning_rate = 10 if is_NPG else 0.05 #0.05 if policy_learning_mode == "online" else 0.05 #0.01 #0.1
reset_value_functions_at_every_learning_step = False #(learning_method == "values_fv")     # Reset the value functions when learning with FV, o.w. the learning can become too unstable due to the oversampling of the states with high value... (or something like that)

# Parameters about value function learning (Critic)
alpha_initial = simulator_value_functions.getAgent().getLearner().getInitialLearningRate()      # NOTE: alpha_initial is NOT used when learning the value functions by function approximation, as this is set by the default learning rate of the Adam optimizer
adjust_alpha_initial_by_learning_step = False; t_learn_min_to_adjust_alpha = 30 # based at 1 (regardless of the base value used for t_learn)
#max_time_steps_per_episode = test_ac.getEnv().getNumStates()*10  # (2024/05/02) NO LONGER USED!  # This parameter is just set as a SAFEGUARD against being blocked in an episode at some state of which the agent could be liberated by restarting to a new episode (when this max number of steps is reached)
epsilon_random_action = 0.1 #if policy_learning_mode == "online" else 0.0 #0.1 #0.05 #0.0 #0.01
use_average_max_time_steps_in_td_learner = True #learning_method == "values_td2" #True #False
learning_steps_observe = [50, 90] #[2, 30, 48] #[2, 10, 11, 30, 31, 49, 50] #[7, 20, 30, 40]  # base at 1, regardless of the base value used for t_learn
verbose_period = max_time_steps_fv_for_all_particles // 10
plot = False         # Whether to plot the evolution of value function and average reward estimation
colormap = "seismic"  # "Reds"  # Colormap to use in the plot of the estimated state value function V(s)

# Results saving, with filename prefix and suffix
save = True
prefix = f"ActorCritic_{env_type.name.lower()}_{shape_str}_"
suffix = f"_{learning_method}"

# Open log file if one requested and show method being run
if log:
    dt_start, stdout_sys, stderr_sys, fh_log, _logfile_not_used = log_file_open(logsdir, subdir="", prefix=prefix, suffix=suffix, use_datetime=True)
print("******")
print(f"Running {learning_method.upper()} method for value functions estimation.")
print(f"A NOMINAL MAXIMUM of {max_time_steps_benchmark} steps will be allowed during the simulation.")
print("******")

# A few further parameters for the policy learning process
break_when_no_change = False    # Whether to stop the learning process when the average reward doesn't change from one step to the next
break_when_goal_reached = False  # Whether to stop the learning process when the average reward is close enough to the maximum average reward (by a relative tolerance of 0.1%)

# Initialize objects that will contain the results by learning step
state_counts_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates()), dtype=int)
V_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates()))
Q_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions()))
A_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions()))
R_all = np.nan * np.ones((nrep, n_learning_steps))       # Average reward (EPISODIC learning task)
R_long_all = np.nan * np.ones((nrep, n_learning_steps))  # Long-run Average reward (CONTINUING learning task). It does NOT converge to the same value as the episodic average reward because there is one more reward value per episode!! (namely the reward going from the terminal state to the start state)
R_long_true_all = np.nan * np.ones((nrep, n_learning_steps))  # True Long-run Average reward (CONTINUING learning task) under the policy at the start of each policy learning step
loss_all = np.nan * np.ones((nrep, n_learning_steps))
nsteps_all = np.nan * np.ones((nrep, n_learning_steps), dtype=int)  # Number of value function time steps run per every policy learning step
KL_all = np.nan * np.ones((nrep, n_learning_steps))      # K-L divergence between two consecutive policies
alpha_all = alpha_initial * np.ones((nrep, n_learning_steps))   # Initial alpha used at each policy learning step
time_elapsed_all = np.nan * np.ones(nrep)   # Execution time for each replication

# Prepare the Actor learner
if learning_method_type == "values_fv":
    # Define the object where we will store the number of steps used by the FV learner of value functions at each policy learning step
    # so that we use the same number for the TD learner of value functions when the TDAC policy learning process is run afterwards.
    # I.e. this assumes that FVAC is run BEFORE TDAC!
    max_time_steps_benchmark_all = np.nan * np.ones((nrep, n_learning_steps))

    # Store the definition of the start state distribution for the E(T_A) simulation for the FV simulation, so that we can reset them to their original ones
    # when starting a new replication (as these distributions may be updated by the process)
    probas_stationary_start_state_et = test_ac.agent_nn_fv.getLearner().getProbasStationaryStartStateET()
    probas_stationary_start_state_fv = test_ac.agent_nn_fv.getLearner().getProbasStationaryStartStateFV()
if learning_method == "all_online":
    # Online Actor-Critic policy learner with TD as value functions learner and value functions learning happens at the same time as policy learning
    learner_ac = LeaActorCriticNN(test_ac.getEnv(), simulator_value_functions.getAgent().getPolicy(), simulator_value_functions.getAgent().getLearner(),
                                  allow_deterministic_policy=allow_deterministic_policy,
                                  reset_value_functions=reset_value_functions_at_every_learning_step, initial_policy=initial_policy, optimizer_learning_rate=optimizer_learning_rate, seed=test_ac.seed, debug=True)
else:
    # Value functions (Critic) are learned separately from the application of the policy and the policy (Actor) may be learned OFFLINE or online
    # IMPORTANT: We pass the policy of the agent stored in the value functions simulator as policy for the Actor-Critic learner so that when the Actor-Critic learner
    # updates the policy, the policy of the agent stored in the value functions simulator is ALSO updated. This is crucial for using the updated policy
    # when learning the value functions at the next policy learning step.
    learner_ac = LeaActorCriticNN(test_ac.getEnv(), simulator_value_functions.getAgent().getPolicy(), simulator_value_functions.getAgent().getLearner(),
                                  allow_deterministic_policy=allow_deterministic_policy,
                                  reset_value_functions=reset_value_functions_at_every_learning_step, initial_policy=initial_policy, optimizer_learning_rate=optimizer_learning_rate, seed=test_ac.seed, debug=True)

time_start = timer()
dt_start_filename = get_current_datetime_as_string(format="filename")

seed_base = test_ac.seed
for rep in range(nrep):
    seed_rep = seed_base + rep*1317
    print(f"\n->>>>>>>>>>>\nRunning replication {rep+1} of {nrep} (seed={seed_rep})... @{format(get_current_datetime_as_string())}")

    # Reset the policy actor and the critic every time a new replication starts
    # Note that this performs a further reset of the policy (besides the one done above by the constructor that creates the learner_ac object),
    # and thus resets the policy to ANOTHER slightly different policy (because of the normally distributed random values around 0 that are set as neural network weights)
    print("Resetting the policy learner and the critic (if any)...")
    learner_ac.reset(reset_value_functions=True, reset_policy=True, initial_policy=initial_policy)
    if not test_ac.getEnv().isStateContinuous():
        print(f"Initial policy (states x actions):\n{learner_ac.getPolicy().get_policy_values()}")

    if simulator_value_functions is not None:
        # Whenever there is a critic, do a couple of resets
        # (recall that no critic is defined for the ALL-online learning method)
        _learner_value_functions_for_critic = simulator_value_functions.getAgent().getLearner()

        # Reset the value functions and average reward that may have been obtained at the end of the previous replication
        # Note that these may not be reset at the start of the first policy learning step because execution parameter use_average_reward_from_previous_step may be set to True.
        _learner_value_functions_for_critic.reset(reset_episode=True, reset_value_functions=True, reset_average_reward=True)

        # Reset the initial alpha of the TD learner (just in case)
        _learner_value_functions_for_critic.setInitialLearningRate(alpha_initial)

        # For FV learners, reset the start distribution for the E(T_A) simulation and for FV to their original values defined in the learners defined in test_ac
        if learning_method_type == "values_fv":
            _learner_value_functions_for_critic.setProbasStationaryStartStateET(probas_stationary_start_state_et)
            _learner_value_functions_for_critic.setProbasStationaryStartStateFV(probas_stationary_start_state_fv)

    time_start_rep = timer()
    if learning_method == "all_online":
        for t_learn in range(n_learning_steps):
            print(f"\n\n*** Running learning step {t_learn+1} of {n_learning_steps} (AVERAGE REWARD at previous step = {R_all[rep, max(0, t_learn-1)]}) of "
                  f"MAX={max_avg_reward_episodic if policy_learning_mode == 'online' else max_avg_reward_continuing} using {nsteps_all[rep, max(0, t_learn-1)]} time steps for Critic estimation)... (seed={seed_learn}) @{get_current_datetime_as_string()}")
            print("Learning the VALUE FUNCTIONS and POLICY simultaneously...")
            loss_all[rep, t_learn] = learner_ac.learn(n_episodes_per_learning_step, start_state=entry_state, max_time_steps_per_episode=max_time_steps_per_policy_learning_episode, prob_include_in_train=1.0) # prob_include_in_train=0.5)
                ## Note that we make sure that the start state when learning the policy is the entrance state to the labyrinth, `entry_state`, because the environment may have defined
                ## a different initial state distribution (e.g. a random start in the states outside the absorption set used by the FV learner.

            state_counts_all[rep, t_learn, :] = learner_ac.learner_value_functions.getStateCounts()
            V_all[rep, t_learn, :] = learner_ac.learner_value_functions.getV().getValues()
            Q_all[rep, t_learn, :, :] = learner_ac.learner_value_functions.getQ().getValues().reshape(test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions())
            A_all[rep, t_learn, :, :] = learner_ac.learner_value_functions.getA().getValues().reshape(test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions())
            R_all[rep, t_learn] = learner_ac.learner_value_functions.getAverageReward()
            # Could also retrieve the average reward from the Actor-Critic learner (if store_trajectory_history=False in the constructor of the value functions learner)
            #R_all[rep, t_learn] = learner_ac.average_reward_over_episodes
    else:
        # Keep track of the policy learned so that we can analyze how much it changes after each learning step w.r.t. the previous learning step
        policy_prev = None
        for t_learn in range(n_learning_steps):
            # Set the policy in evaluation mode
            # This is important if we are using dropout layers in the neural network, o.w. the policy output by the model may be incorrect because some connections might be missing
            # when evaluating the policy(a|s) by calling policy.getPolicyForState()!! (this is not the case in evaluation mode because ALL the connections are back during evaluation,
            # even if the connection had been dropped during training).
            learner_ac.getPolicy().getModel().eval()

            if env_type == Environment.Gridworld:
                # Compute or update the true state value function stored in the environment for the current policy
                # (used as reference when plotting the evolution of the estimated state value function V(s) when plot=True)
                V_true, avg_reward_true, mu = compute_true_state_value_function(test_ac.getEnv(), learner_ac.getPolicy(), learning_task, learning_criterion)
                R_long_true_all[rep, t_learn] = avg_reward_true

            # Pass a different seed (for the simulator) for each learning step... o.w. we will be using the same seed for them at every learning step!!
            seed_learn = seed_rep + t_learn
            if env_type == Environment.Gridworld:
                print(f"\n\n*** Running learning step {t_learn+1} of {n_learning_steps} (True average reward under current policy = {avg_reward_true}) (AVERAGE REWARD at previous step = {R_all[rep, max(0, t_learn-1)]} of MAX={max_avg_reward_episodic})... (seed={seed_learn}) @{get_current_datetime_as_string()}")
            else:
                print(f"\n\n*** Running learning step {t_learn+1} of {n_learning_steps}  (AVERAGE REWARD at previous step = {R_all[rep, max(0, t_learn-1)]} of MAX={max_avg_reward_episodic})... (seed={seed_learn}) @{get_current_datetime_as_string()}")
            reset_value_functions_at_this_step = reset_value_functions_at_every_learning_step if t_learn > 0 else True  # ALWAYS RESET THE VALUE FUNCTIONS WHEN IT'S THE VERY FIRST LEARNING STEP (becaue we don't want to keep histroy from a earlier learning process on the same learner!)
            # Update the initial learning rate for the value functions at each learning step to a smaller value than the previous learning step
            # SHOULD WE SET IT TO THE AVERAGE LEARNING RATE FROM THE PREVIOUS LEARNING STEP?? (so that we start off where we left at the last learning moment)
            alpha_initial_at_current_learning_step = alpha_initial / (t_learn + 1)

            #-- Optionally adjust the initial learning rate alpha
            # Compute the K-L divergence of the current policy w.r.t. the previous policy to decide whether to update the initial learning rate or not
            # If the new policy is too different from the previous policy we should NOT update the learning rate because the value functions learned
            # under the previous policy is most likely far away from the true values functions under the new policy.
            # Note: rel_entr(p, q) computes the term p * log(p/q) contributing to the relative entropy or Kullback-Leibler divergence
            # of the "new" distribution p(x) w.r.t. the "old" distribution q(x), which is defined as the expectation (measured over p(x)) of the
            # difference between log(p(x)) and log(q(x)). Hence if p(x) is larger than q(x) the contribution is positive and
            # if it is smaller, the contribution is negative. Therefore a positive K-L divergence typically corresponds to an increase
            # in the probability from q(x) at the values of x having a large "new" probability p(x) value.
            # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html
            # Note that we could also use the function scipy.stats.entropy() to compute directly the K-L divergence (i.e. without summing over the rel_entr() values, as done here)
            if not test_ac.getEnv().isStateContinuous():
                policy = learner_ac.getPolicy().get_policy_values()
                KL_distance = np.sum(rel_entr(policy, policy_prev)) if t_learn > 0 else 0.0
                print(f"K-L distance with previous policy: {KL_distance} (should be < {KL_THRESHOLD} to reduce initial alpha)")
                KL_all[rep, t_learn] = KL_distance
                policy_prev = policy.copy()
            else:
                KL_all[rep, t_learn] = np.nan

            if adjust_alpha_initial_by_learning_step:
                # Update the initial learning rate alpha ONLY when:
                # - the learning step is larger than or equal to a minimum
                # - the policy did NOT change significantly from the previous learning step
                #   (because in that case we can consider the current estimate to be a reliable estimator of the value functions)
                if t_learn + 1 >= t_learn_min_to_adjust_alpha:
                    if policy_changed_from_previous_learning_step(KL_distance, test_ac.getEnv().getNumStates()): #np.abs(loss_all[rep, t_learn-1]) > 1.0
                        simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial)
                    else:
                        simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial / 10)
                alpha_all[rep, t_learn] = simulator_value_functions.getAgent().getLearner().getInitialLearningRate()
            print(
                f"*** INITIAL learning rate alpha = {simulator_value_functions.getAgent().getLearner().getInitialLearningRate()} " +
                (f"(adjustment happens starting at learning step (base 1) >= {t_learn_min_to_adjust_alpha}) " if adjust_alpha_initial_by_learning_step else "***\n"))

            #--- 1) CRITIC
            print(f"Learning the CRITIC (at the current policy) using {learning_method.upper()}...")
            # Learn the value functions using the FV simulator
            if learning_method_type == "values_fv":
                # Reset to None the start state distribution for the E(T_A) excursion at the very first learning step so that
                # we do NOT carry over whatever this distribution was at the end of the previous replication or at the end of the previous execution of the learning process
                # using this same learner.
                if t_learn == 0:
                    simulator_value_functions.getAgent().getLearner().setProbasStationaryStartStateET(None)
                V, Q, A, state_counts, state_counts_et, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, n_events_et, n_events_fv = \
                    simulator_value_functions.run(t_learn=t_learn,
                                                  max_time_steps=max_time_steps_fv_overall,
                                                  max_time_steps_for_absorbed_particles_check=max_time_steps_fv_for_all_particles,
                                                  min_num_cycles_for_expectations=0,
                                                      ## Note: We set the minimum number of cycles for the estimation of E(T_A) to 0 because we do NOT need
                                                      ## the estimation of the average reward to learn the optimal policy, as it cancels out in the advantage function Q(s,a) - V(s)!!
                                                  estimate_absorption_set=estimate_absorption_set_at_every_step, threshold_absorption_set=threshold_absorption_set,
                                                  use_average_reward_stored_in_learner=use_average_reward_from_previous_step,
                                                  reset_value_functions=reset_value_functions_at_this_step,
                                                  plot=plot if t_learn+1 in learning_steps_observe else False, colormap=colormap,
                                                  epsilon_random_action=epsilon_random_action,
                                                  seed=seed_learn, verbose=False, verbose_period=verbose_period)
                #average_reward = simulator_value_functions.getAgent().getLearner().getAverageReward()  # This average reward should not be used because it is inflated by the FV process that visits the states with rewards more often
                average_reward = expected_reward
                nsteps_all[rep, t_learn] = n_events_et + n_events_fv
                max_time_steps_benchmark_all[rep, t_learn] = n_events_et + n_events_fv  # Number of steps to use when running TDAC at the respective learning step
            else:
                # TD learners
                if 'max_time_steps_benchmark_all' in locals() and rep < len(max_time_steps_benchmark_all) and t_learn < len(max_time_steps_benchmark_all[rep, :]) and max_time_steps_benchmark_all[rep, t_learn] != np.nan:
                    # The FV learner was run before running this TD learner
                    if use_average_max_time_steps_in_td_learner:
                        _max_time_steps = int( np.mean(max_time_steps_benchmark_all[rep, :]) )
                    else:
                        # => Use the number of steps used by the FV learner at the current policy learning step (t_learn) as maximum number of steps to allow for the TD learner
                        _max_time_steps = max_time_steps_benchmark_all[rep, t_learn]
                else:
                    # When max_time_steps_benchmark_all is not defined, it means that the number of steps to run the TD learner for is calculated above
                    # and may not be exactly equal to the number of steps the FV learner took at each policy learning step.
                    _max_time_steps = max_time_steps_benchmark
                print(f"*** TD learning will use {_max_time_steps} simulation steps. ***")
                if False:
                    # Use this when the discrete.Simulator.run() calls the _run_single() method to run the simulation, which is based on the existence of episodes
                    V, Q, A, state_counts, _, _, learning_info = \
                        simulator_value_functions.run(nepisodes=n_episodes_per_learning_step,
                                                      t_learn=t_learn,
                                                      max_time_steps=max_time_steps_benchmark,
                                                      max_time_steps_per_episode=max_time_steps_per_episode,  # max_time_steps_benchmark // n_episodes_per_learning_step,
                                                      reset_value_functions=reset_value_functions_at_this_step,
                                                      seed=seed_learn,
                                                      state_observe=state_observe, compute_rmse=True if t_learn+1 in learning_steps_observe else False,
                                                      epsilon_random_action=epsilon_random_action,
                                                      plot=plot if t_learn+1 in learning_steps_observe else False, colormap=colormap,
                                                      verbose=True, verbose_period=verbose_period)
                else:
                    # Use this when the discrete.Simulator.run() calls the _run_single_continuing_task() method to run the simulation (where there are no episodes)
                    V, Q, A, state_counts, _, _, learning_info = \
                        simulator_value_functions.run(t_learn=t_learn,
                                                      max_time_steps=_max_time_steps,
                                                      estimated_average_reward=simulator_value_functions.getAgent().getLearner().getAverageReward() if use_average_reward_from_previous_step else None,
                                                      reset_value_functions=reset_value_functions_at_this_step,
                                                      seed=seed_learn,
                                                      state_observe=state_observe,
                                                      compute_rmse=True if t_learn+1 in learning_steps_observe else False,
                                                      plot=plot if t_learn+1 in learning_steps_observe else False, colormap=colormap,
                                                      epsilon_random_action=epsilon_random_action,
                                                      verbose=True, verbose_period=verbose_period)
                average_reward = simulator_value_functions.getAgent().getLearner().getAverageReward()
                nsteps_all[rep, t_learn] = learning_info['nsteps']

            print(f"Learning step #{t_learn+1}: Learning of value functions COMPLETED using {learning_method} method on {nsteps_all[rep, t_learn]} time steps")
            print(f"Estimated average reward: {average_reward}")
            state_counts_all[rep, t_learn, :] = state_counts
            if simulator_value_functions.getAgent().getLearner().getV().isTabular():
                V_all[rep, t_learn, :] = V
            if simulator_value_functions.getAgent().getLearner().getV().isTabular():
                Q_all[rep, t_learn, :, :] = Q.reshape(test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions())
            if simulator_value_functions.getAgent().getLearner().getA().isTabular():
                A_all[rep, t_learn, :, :] = A.reshape(test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions())

            #--- 2) ACTOR
            print(f"\nLearning the POLICY {policy_learning_mode.upper()} using estimated {use_advantage and 'ADVANTAGE A(s,a) values' or 'ACTION Q(s,a) values'} ", end=" ")
            # Policy learning
            if is_NPG:
                # Learn using NPG (Natural Policy Gradient)
                print("\n(Natural Policy Gradient learning)")
                learner_ac.learn_natural(A)
                if not test_ac.getEnv().isStateContinuous():
                    print(f"NEW policy (states x actions):\n{learner_ac.getPolicy().get_policy_values()}")
            else:
                # The policy is modeled with a regular neural network (with hidden layer)
                if policy_learning_mode == "online":
                    # ONLINE with critic provided by the action values learned above
                    print(f"on {n_episodes_per_learning_step} episodes starting at state s={entry_state}, using MAX {max_time_steps_per_policy_learning_episode} steps per episode...")
                    loss_all[rep, t_learn] = learner_ac.learn(n_episodes_per_learning_step, start_state=entry_state, max_time_steps_per_episode=max_time_steps_per_policy_learning_episode, prob_include_in_train=1.0,
                                                              # (2024/08/15) This learner is used when the state is continuous (e.g. Mountain Car) and in LeaActorCriticNN.learn() the advantage is computed using
                                                              # the model for V(s), instead of using the tabular values of the advantage function passed here as `advantage_values`.
                                                              # CHECK THE learn() CODE TO SEE IF THE ABOVE IS THE CASE!
                                                              learner_value_functions_critic=simulator_value_functions.getAgent().getLearner(),
                                                              use_advantage=use_advantage,
                                                              advantage_values=A,
                                                              action_values=Q,       # This parameter is not used when use_advantage=False
                                                              expected_reward=average_reward)
                        ## Note that we make sure that the start state when learning the policy is the entrance state to the labyrinth, `entry_state`,
                        ## because the environment may have defined a different initial state distribution, which is used during the learning of the value functions,
                        ## for instance, any randomly selected state outside the absorption set A used by the FV learner.
                        ## Note that the learned value functions are passed as critic to the Actor-Critic policy learner via the `action_values` parameter.
                    R_all[rep, t_learn] = learner_ac.average_reward_over_episodes
                else:
                    # OFFLINE learner where ALL states and actions are swept and the loss computed on all of them using the state distribution as weights
                    print("and estimated probabilities from Critic estimation excursion...")
                    if learning_method_type == "values_fv":
                        prob_states = compute_prob_states(state_counts_et, probas_stationary=probas_stationary)
                    else:
                        prob_states = compute_prob_states(state_counts)
                    loss_all[rep, t_learn] = learner_ac.learn_offline_from_estimated_value_functions(V, A, Q, state_counts, prob_states=prob_states, use_advantage=use_advantage)
                    R_all[rep, t_learn] = average_reward
                    _dict_numpy_options = set_numpy_options()
                    print(f"True stationary probabilities:\n{mu.reshape(env_shape)}")
                    print(f"Estimated stationary probabilities:\n{prob_states.reshape(env_shape)}")
                    reset_numpy_options(_dict_numpy_options)
                    if False and (average_reward != 0.0 or t_learn+1 in learning_steps_observe):
                        def plot_probas(ax, prob_states_2d, fontsize=14, colormap="Blues", color_text="orange"):
                            colors = cm.get_cmap(colormap)
                            ax.imshow(prob_states_2d, cmap=colors)
                            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                            for x in range(env_shape[0]):
                                for y in range(env_shape[1]):
                                    ax.text(y, x, "{:.3f}".format(prob_states_2d[x, y]), color=color_text, fontsize=fontsize, horizontalalignment="center", verticalalignment="center")
                        ax_true, ax_est, ax_diff = plt.figure().subplots(1, 3)
                        _fontsize = 14
                        _factor_fs = np.min((5 / env_shape[0], 5 / env_shape[1]))
                        plot_probas(ax_true, mu.reshape(env_shape), fontsize=_fontsize*_factor_fs)
                        plot_probas(ax_est, prob_states.reshape(env_shape), fontsize=_fontsize*_factor_fs)
                        plot_probas(ax_diff, (prob_states - mu).reshape(env_shape), fontsize=_fontsize*_factor_fs, colormap="jet")
                        plt.pause(0.1)
                        plt.draw()
                        input("Press ENTER to continue...")

            # Store the long-run average reward estimated by the value functions learner used above
            R_long_all[rep, t_learn] = average_reward

            # Check if we need to stop learning because the average reward didn't change a bit
            if  break_when_no_change and t_learn > 0 and R_all[rep, t_learn] - R_all[rep, t_learn-1] == 0.0 or \
                break_when_goal_reached and np.isclose(R_all[rep, t_learn], max_avg_reward_episodic, rtol=0.001):
                print(f"*** Policy learning process stops at learning step t_learn+1={t_learn+1} because the average reward didn't change a bit from previous learning step! ***")
                break
    time_elapsed_rep = timer() - time_start_rep
    print(f"<<<<<<<<<< FINISHED replication {rep+1} of {nrep}... (@{format(get_current_datetime_as_string())}" + ", took {:.1f} min)".format(time_elapsed_rep / 60))
    time_elapsed_all[rep] = time_elapsed_rep

time_end = timer()
time_elapsed = time_end - time_start

if log:
    log_file_close(fh_log, stdout_sys, stderr_sys, dt_start)
else:
    print("{} learning process took {:.1f} minutes ({:.1f} hours)".format(learning_method.upper(), time_elapsed / 60, time_elapsed / 3600))


############# Store the measures that we would like to compare
dict_loss[learning_method] = loss_all.copy()
dict_R[learning_method] = R_all.copy()
dict_R_long[learning_method] = R_long_all.copy()
dict_R_long_true[learning_method] = R_long_true_all.copy()
dict_V[learning_method] = V_all.copy()
dict_Q[learning_method] = Q_all.copy()
dict_A[learning_method] = A_all.copy()
dict_state_counts[learning_method] = state_counts_all.copy()
dict_nsteps[learning_method] = nsteps_all.copy()
dict_KL[learning_method] = KL_all.copy()
dict_alpha[learning_method] = alpha_all.copy()
dict_time_elapsed[learning_method] = time_elapsed_all.copy()
############# Store the measures that we would like to compare


#------------------ Plots -----------------
# Plot loss and average reward for the currently analyzed learner
print("\nPlotting...")
ax_loss = plt.figure(figsize=figsize).subplots(1, 1)
ax_loss.plot(range(1, n_learning_steps+1), dict_loss[learning_method][rep, :n_learning_steps], marker='.', color="red")
#ax_loss.plot(range(1, n_learning_steps+1), dict_alpha[learning_method][rep, :n_learning_steps], '--', color="cyan")
ax_loss.set_xlabel("Learning step")
ax_loss.set_ylabel("Loss", color="red")
#ax_loss.axhline(0, color="red", linewidth=1, linestyle='dashed')
ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax_loss.legend(["Loss", "alpha0"], loc='upper left')
ax_R = ax_loss.twinx()
if policy_learning_mode == "online":
    # We learn the policy by doing a final excursion using the current policy and computing the loss
    # => Plot the episodic average reward observed during the ONLINE Actor-Critic excursion
    ax_R.plot(range(1, n_learning_steps+1), dict_R[learning_method][rep, :n_learning_steps], marker='.', color="green")
    ax_R.axhline(max_avg_reward_episodic, color="green", linewidth=1)
ax_R.plot(range(1, n_learning_steps+1), dict_R_long[learning_method][rep, :n_learning_steps], marker='.', color="greenyellow")
ax_R.axhline(max_avg_reward_continuing, color="greenyellow", linewidth=1)
ax_R.set_ylabel("Average reward", color="green")
ax_R.plot(range(1, n_learning_steps+1), dict_KL[learning_method][rep, :n_learning_steps], color="blue", linewidth=1)
ax_R.axhline(KL_THRESHOLD, color="blue", linestyle="dashed")
ax_R.axhline(0, color="green", linewidth=1, linestyle='dashed')
# Set the Y-axis limits so that we can see better the learning curve despite possible large K-L values that could make the curve look very tiny
# This is particularly useful when the policy is learned fast within each Actor-Critic excursion,
# which for instance is achieved when computing the loss and updating parameters after each Actor-Critic episode (mini-batch)
# or when learning the policy with NPG.
if policy_learning_mode == "online" and not is_NPG:
    ax_R.set_ylim((-np.max(dict_R[learning_method][rep, :n_learning_steps])/50, np.max(dict_R[learning_method][rep, :n_learning_steps])*1.1))
else:
    ax_R.set_ylim((-np.max(dict_R_long[learning_method][rep, :n_learning_steps])/50, np.max(dict_R_long[learning_method][rep, :n_learning_steps])*1.1))
ax_R.legend(["Average reward (episodic)", "Max. average reward (episodic)",
            "Long-run average reward estimated by value functions learner", "Max. average reward (continuing)",
            "K-L divergence with previous policy", "K-L threshold for reduced alpha0", "K-L divergence between consecutive policies"],
            loc="upper right")
plt.title(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={simulator_value_functions.getAgent().getLearner().gamma}) - {env_type.name} {env_shape}"
          f"\nN={test_ac.agent_nn_fv.getLearner().getNumParticles()}, T={test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
          f"\nEvolution of the LOSS (left, red) and Average Reward (right, green) with the learning step")


#-- Plot the trajectory as a GIF
if env_type == Environment.MountainCar:
    # Plot the trajectory of the last replication
    # Notes:
    # - Since no trajectory information is currently stored by the online policy learner, we use the value functions learner to extract the trajectory to plot.
    # - We limit the trajectory to the first 100 steps because o.w. the GIF would take too long to generate...
    #   For instance, with 100 steps, the GIF takes ~2 minutes to generate and is 1 MB in size already!
    if learning_method_type == "values_fv":
        # Take the state distribution from the initial exploration in the FV case because the particle trajectories are NOT stored in the FV learner
        trajectory = np.array(test_ac.learner_for_initial_exploration.getStates())
    else:
        trajectory = np.array(simulator_value_functions.agent.getLearner().getStates()[0])
    T = max_time_steps_fv_for_expectation
    _npoints2plot = min(T, 100)
    assert _npoints2plot < len(trajectory)

    # If we want to make comparable plots (i.e. the same first T steps for both FVAC and TDAC)
    trajectory2plot = trajectory[:_npoints2plot]
    # If we want to plot the beginning and end of the trajectory
    # if _npoints2plot < _npoints:
    #     # We repeat 10 times the last point in the first half of the points to plot so that we visually understand that there is a jump in time
    #     _indices2plot = np.r_[np.arange(_npoints2plot // 2), np.repeat(_npoints2plot // 2, 10), np.arange(_npoints - _npoints2plot // 2, _npoints)]
    # else:
    #     _indices2plot = np.arange(_npoints)
    # trajectory2plot = trajectory[_indices2plot]

    test_ac.getEnv().plot_trajectory_gif([test_ac.getEnv().getStateFromIndex(s, simulation=False) for s in trajectory2plot])


#-- Plot the value functions for the state next to the terminal state
# ONLY VALID WHEN THE EXIT STATE IS AT THE TOP RIGHT OF THE LABYRINTH
marker = ''
Q_all_baseline = dict_Q[learning_method][rep, :n_learning_steps, :, :] - np.tile(dict_V[learning_method][rep, :n_learning_steps, :].T, (test_ac.getEnv().getNumActions(), 1, 1)).T
ax_Q, ax_Q_baseline = plt.figure().subplots(1, 2)
ax_Q.plot(range(1, n_learning_steps + 1), dict_V[learning_method][rep, :n_learning_steps, state_observe], marker=marker, color="black")
ax_Q.plot(range(1, n_learning_steps + 1), dict_Q[learning_method][rep, :n_learning_steps, state_observe, :], marker=marker)
ax_Q.legend(["V(s)"] + ["Q(s," + str(a) + ")" for a in range(dict_Q[learning_method].shape[2])], loc='upper left')
ax_Q_baseline.plot(range(1, n_learning_steps + 1), dict_V[learning_method][rep, :n_learning_steps, state_observe] - dict_V[learning_method][rep, :n_learning_steps, state_observe], marker=marker, color="white") # We plot this constant value 0 so that the legend is correct
ax_Q_baseline.plot(range(1, n_learning_steps + 1), Q_all_baseline[:n_learning_steps, state_observe, :], marker=marker)
ax_Q_baseline.legend(["V(s) - V(s)"] + ["Q(s," + str(a) + ") - V(s)" for a in range(dict_Q[learning_method].shape[2])], loc='upper left')

# Optimum Q-values (for the optimum deterministic policy)
# This assumes that there is a reward of 1 at the terminal state which is one step away
if learning_criterion == LearningCriterion.AVERAGE:
    assert learning_task == LearningTask.CONTINUING
    svalue  = 0.5 * (1.0 - max_avg_reward_continuing)     # Differential state value V(s) under the optimal policy (since the policy tells the agent to always go up, the agent receives reward 1.0, which is corrected (subtracted) by the max (because we are following the OPTIMAL policy) average reward; the 0.5 factor is explained by the calculations on my Mas de Canelles notebook sheet)
    qvalue0 = 0.5 * (1.0 - max_avg_reward_continuing)     # Differential optimal action value Q(s,a) of going up (a=0) when we start at s = state_observe, one cell away from the terminal state, which gives reward 1.0 (we subtract the max average reward because we are following the OPTIMAL policy; the 0.5 factor is explained by the calculations on my Mas de Canelles notebook sheet)
    qvalue1 = qvalue0 - max_avg_reward_continuing
    qvalue2 = qvalue0 - 2*max_avg_reward_continuing
    qvalue3 = qvalue0 - max_avg_reward_continuing
else:   # DISCOUNTED reward criterion
    gamma = test_ac.gamma
    reward_at_terminal = 1
    # State value V(s):
    # - Under the optimal policy we go always up and observe the terminal reward right-away
    # - If the learning task is continuing, we still keep observing the terminal reward discounted by the length of the optimal path (= np.sum(test_ac.getEnv().shape) - 1)
    # (the `-1` at the end of the parenthesis cancels the `1+` at the beginning of the parenthesis when the learning task is CONTINUING)
    svalue  = reward_at_terminal * (1 + int(learning_task == LearningTask.CONTINUING) * (1 / (1 - gamma**(np.sum(test_ac.getEnv().shape)-1))) - 1)
    qvalue0 = svalue
    qvalue1 = gamma * svalue
    qvalue2 = gamma**2 * svalue
    qvalue3 = gamma * svalue
ax_Q.axhline(qvalue0, linestyle='dashed', color="blue")
ax_Q.axhline(qvalue1, linestyle='dashed', color="orange")
ax_Q.axhline(qvalue2, linestyle='dashed', color="green")
ax_Q.axhline(qvalue3, linestyle='dashed', color="red")
ax_Q.set_xlabel("Learning step")
ax_Q.set_ylabel("Q values and state values")
ax_Q_baseline.axhline(qvalue0 - svalue, linestyle='dashed', color="blue")
ax_Q_baseline.axhline(qvalue1 - svalue, linestyle='dashed', color="orange")
ax_Q_baseline.axhline(qvalue2 - svalue, linestyle='dashed', color="green")
ax_Q_baseline.axhline(qvalue3 - svalue, linestyle='dashed', color="red")
ax_Q_baseline.set_xlabel("Learning step")
ax_Q_baseline.set_ylabel("Q values w.r.t. baseline")
plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={simulator_value_functions.getAgent().getLearner().gamma}) - {env_type.name} {env_shape}"
             f"\nN={test_ac.agent_nn_fv.getLearner().getNumParticles()}, T={test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
             f"\nQ(s,a) and V(s) for state previous to the terminal state under the optimal policy, i.e. s={state_observe}\nMax average reward (continuing) = {max_avg_reward_continuing}")


# Same plot for all states
axes = plt.figure(figsize=(10, 9)).subplots(test_ac.getEnv().shape[0], test_ac.getEnv().shape[1])
first_learning_step = 0 #n_learning_steps * 3 // 4  #0
y2max = int(round(np.max(dict_state_counts[learning_method])*1.1)) # For a common Y2-axis showing the state counts
min_V, max_V = np.min(dict_V[learning_method]), np.max(dict_V[learning_method])      # For a common Y-axis showing the value functions
min_Q, max_Q = np.min(dict_Q[learning_method]), np.max(dict_Q[learning_method])      # For a common Y-axis showing the value functions
Q_all_baseline = dict_Q[learning_method][rep, first_learning_step:n_learning_steps, :, :] - np.tile(dict_V[learning_method][rep, first_learning_step:n_learning_steps, :].T, (test_ac.getEnv().getNumActions(), 1, 1)).T
min_Q_baseline, max_Q_baseline = np.min(Q_all_baseline), np.max(Q_all_baseline)      # For a common Y-axis showing the value functions
ymin, ymax = min(min_V, min_Q), max(max_V, max_Q)
ymin_baseline, ymax_baseline = min(0, min_Q_baseline), max(0, max_Q_baseline)

plot_baseline = False
ylim = (ymin_baseline, ymax_baseline) if plot_baseline else (ymin, ymax)     # Use this for common Y-axis limits
#ylim = (None, None)     # Use this for unequal Y-axis limits
marker = ''
for i, ax in enumerate(axes.reshape(-1)):
    # Value functions on the left axis
    if plot_baseline:
        # Q values with baseline (so that we can better see the difference in value among the different actions)
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), dict_V[learning_method][rep, first_learning_step:n_learning_steps, i] - dict_V[learning_method][rep, first_learning_step:n_learning_steps, i], marker=marker, color="black")  # We plot this so that the legend is fine
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), Q_all_baseline[first_learning_step:n_learning_steps, i, :], marker=marker)
    else:
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), dict_V[learning_method][rep, first_learning_step:n_learning_steps, i], marker=marker, color="black")
        ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), dict_Q[learning_method][rep, first_learning_step:n_learning_steps, i, :], marker=marker)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(ylim)

    # State counts on the right axis
    ax2 = ax.twinx()
    ax2.plot(np.arange(1+first_learning_step, n_learning_steps+1), dict_state_counts[learning_method][rep, first_learning_step:n_learning_steps, i], color="violet", linewidth=1)
    ax2.set_ylim((0, y2max))
    y2ticks = [0, int(min(dict_state_counts[learning_method][rep, first_learning_step:n_learning_steps, i])), int(max(dict_state_counts[learning_method][rep, first_learning_step:n_learning_steps, i])), int(np.max(dict_state_counts[learning_method]))]
    ax2.set_yticks(y2ticks)
    ax2.yaxis.set_ticklabels(y2ticks, fontsize=7)
# Only show the x and y labels on the bottom-right plot (to avoid making the plot too cloggy)
ax.set_xlabel("Learning step")
ax2.set_ylabel("State count")
ax.legend(["V(s)"] + ["Q(s," + str(a) + ")" for a in range(dict_Q[learning_method].shape[2])], loc='upper left')
ax2.legend(["State count"], loc='upper right')
plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion - {env_type.name} {env_shape}"
             f"\nN={test_ac.agent_nn_fv.getLearner().getNumParticles()}, T={test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
             f"\nEvolution of the value functions V(s) and Q(s,a) with the learning step by state\nMaximum average reward (continuing): {max_avg_reward_continuing}")


# Plot the ADVANTAGE function
first_learning_step = 0 #n_learning_steps * 3 // 4  #0
y2max = int(round(np.max(dict_state_counts[learning_method])*1.1)) # For a common Y2-axis showing the state counts
min_A, max_A = np.min(dict_A[learning_method]), np.max(dict_A[learning_method])      # For a common Y-axis showing the value functions
ymin, ymax = min_A, max_A

#-----------------
# Plot just ONE advantage function
s_observe = test_ac.getEnv().getNumStates() // 2
ax = plt.figure(figsize=(10, 9)).subplots(1, 1)
i = s_observe
for i in range(test_ac.getEnv().getNumStates()):
    ax.plot(np.arange(1 + first_learning_step, n_learning_steps + 1), dict_A[learning_method][rep, first_learning_step:n_learning_steps, i, :], marker=marker)
ax.set_yscale("log")

# Distribution of advantage values (analyzed for the continuous-state mountain car)
if False:
    ind = np.where(A < 0.5)
    plt.figure()
    plt.hist(A[ind].reshape(-1), bins=30, alpha=0.3)
    plt.gca().set_xscale("log")
    plt.hist(A.reshape(-1), bins=30, alpha=0.3)
    pd.Series(A.reshape(-1)).describe()
#-----------------

common_axes = True
ylim = (ymin, ymax) if common_axes else (None, None)
marker = ''
axes = plt.figure(figsize=(10, 9)).subplots(test_ac.getEnv().shape[0], test_ac.getEnv().shape[1], sharex=common_axes, sharey=common_axes, gridspec_kw=dict(hspace=0.1, wspace=0.1))  # See also help(plt.subplots); help(matplotlib.gridspec.GridSpec)
for i, ax in enumerate(axes.reshape(-1)):
    # Value functions on the left axis
    ax.plot(np.arange(1+first_learning_step, n_learning_steps + 1), dict_A[learning_method][rep, first_learning_step:n_learning_steps, i, :], marker=marker)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(ylim)

    # State counts on the right axis
    ax2 = ax.twinx()
    ax2.plot(np.arange(1+first_learning_step, n_learning_steps+1), dict_state_counts[learning_method][rep, first_learning_step:n_learning_steps, i], color="violet", linewidth=1)
    ax2.set_ylim((0, y2max))
    y2ticks = [0, int(min(dict_state_counts[learning_method][rep, first_learning_step:n_learning_steps, i])), int(max(dict_state_counts[learning_method][rep, first_learning_step:n_learning_steps, i])), int(np.max(dict_state_counts[learning_method]))]
    ax2.set_yticks(y2ticks)
    ax2.yaxis.set_ticklabels(y2ticks, fontsize=7)
# Only show the x and y labels on the bottom-right plot (to avoid making the plot too cloggy)
ax.set_xlabel("Learning step")
ax2.set_ylabel("State count")
ax.legend(["A(s," + str(a) + ")" for a in range(dict_A[learning_method].shape[2])], loc='upper left')
ax2.legend(["State count"], loc='upper right')
plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion - {env_type.name} {env_shape}"
             f"\nN={test_ac.agent_nn_fv.getLearner().getNumParticles()}, T={test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
             f"\nEvolution of the Advantage function A(s,a) with the learning step by state\nMaximum average reward (continuing): {max_avg_reward_continuing}")


#-- Final policy parameters and policy distribution by state (for the last replication)
policy = simulator_value_functions.getAgent().getPolicy()
policy.getModel().eval()
print("Final network parameters:")
print(list(policy.getThetaParameter()))

# Plot the evolution of a few weights of the neural network
# TBD

colormap = cm.get_cmap("rainbow")  # useful colormaps are "jet", "rainbow", seismic"
colornorm = None
aspect_ratio = "auto" if env_type == Environment.MountainCar else "equal"
fontsize = 14
factor_fontsize = 1.0   # Scaling factor when computing the final fontsize to use for labels showing the policy values of the different actions

# Policy for each action at each state
if env_type == Environment.MountainCar:
    # Plot the best action and their probability at each state, as an image
    # We initialize these arrays as NaN so that we can decide not to show anything for a state that hasn't been visited at all during the policy learning process (in one replication)
    best_action = np.nan*np.ones(test_ac.getEnv().getNumStates())
    highest_probability = np.nan * np.ones_like(best_action)
    states_with_at_least_one_visit_during_policy_learning = np.where( np.sum(state_counts_all[rep, :, :], axis=0) )[0]

    # Find the action with highest probability for each discrete state
    policy_values = policy.get_policy_values()
    for s in range(test_ac.getEnv().getNumStates()):
        if s in states_with_at_least_one_visit_during_policy_learning:
            action_highest_probability = np.argmax(policy_values[s])
            best_action[s] = action_highest_probability  #(action_highest_probability + 1) / 3.0
        highest_probability[s] = np.max(policy_values[s])
    print(f"Policy values for each discrete state:\n{policy_values}")

    # Show the best action map as an image plot whose intensity is proportional to the probability of the best action, distinguishing the action by color as indicated in variable `colormaps`
    ax = plt.figure().subplots(1, 1)
    colormaps = ["Blues", "Greens", "Reds"]  # Colormaps for actions LEFT, STAY, RIGHT
    for a in np.arange(test_ac.getEnv().getNumActions()-1, -1, -1):   # We go in reverse order so that the colomaps appear from left to right to represent LEFT, ZERO, RIGHT
        msk = best_action == a
        highest_probability_toplot = np.nan*np.ones_like(highest_probability)
        highest_probability_toplot[msk] = highest_probability[msk]
        ax, img = test_ac.getEnv().plot_values(highest_probability_toplot, ax=ax, cmap=colormaps[a], vmin=0, vmax=1)
    test_ac.getEnv()._finalize_plot(ax)
    axes = [ax]
    plt.subplots_adjust(top=0.80, wspace=0, hspace=0)
    plt.suptitle(f"{learning_method.upper()}\nPolicy at each state:\nShowing the highest probability of:\nBlue = Acc. LEFT, Green: = Do NOT acc., Red = Acc. RIGHT")
else:
    # Plot suitable for Gridworlds
    axes = plt.figure().subplots(*env_shape, sharex=True, sharey=True, gridspec_kw=dict(hspace=0, wspace=0))  # See also help(plt.subplots); help(matplotlib.gridspec.GridSpec)
    proba_actions_toplot = np.nan*np.ones((3, 3))
    if problem_2d:
        # Factor for the fontsize that depends on the environment size
        factor_fs = factor_fontsize * np.min((4 / axes.shape[0], 4 / axes.shape[1]))
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                state_1d = np.ravel_multi_index((i, j), env_shape)
                print("")
                for action in range(test_ac.getEnv().getNumActions()):
                    print(f"Computing policy Pr(a={action}|s={(i,j)})...", end= " ")
                    idx_2d = (0, 1) if action == 0 else (1, 2) if action == 1 else (2, 1) if action == 2 else (1, 0)
                    proba_actions_toplot[idx_2d] = policy.getPolicyForAction(action, state_1d)
                    print("p = {:.3f}".format(proba_actions_toplot[idx_2d]))
                img = axes[i, j].imshow(proba_actions_toplot, cmap=colormap, vmin=0, vmax=1, aspect=aspect_ratio)  # aspect="auto" means use the same aspect ratio as the axes
                # Remove the axes ticks as they do not convey any information
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                for action in range(test_ac.getEnv().getNumActions()):
                    idx_2d = (0, 1) if action == 0 else (1, 2) if action == 1 else (2, 1) if action == 2 else (1, 0)
                    axes[i, j].text(idx_2d[1], idx_2d[0], "{:02d}".format(int(round(proba_actions_toplot[idx_2d]*100))),
                                    color="white", fontsize=fontsize*factor_fs,
                                    horizontalalignment="center", verticalalignment="center")
    else:
        factor_fs = factor_fontsize * 4 / axes.shape[0]
        for i in range(len(axes)):
            state = i
            for action in range(test_ac.getEnv().getNumActions()):
                print(f"Computing policy Pr(a={action}|s={state})...", end=" ")
                idx_2d = (0, 1) if action == 0 else (1, 2) if action == 1 else (2, 1) if action == 2 else (1, 0)
                proba_actions_toplot[idx_2d] = policy.getPolicyForAction(action, state)
                print("p = {:.3f}".format(proba_actions_toplot[idx_2d]))
            img = axes[i].imshow(proba_actions_toplot, cmap=colormap, vmin=0, vmax=1, aspect="auto")  # aspect="auto" means use the same aspect ratio as the axes
            # Remove the axes ticks as they do not convey any information
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            for action in range(test_ac.getEnv().getNumActions()):
                axes[i].text(0, action, "{:02d}".format(int(round(proba_actions_toplot[0, action] * 100))),
                             color="white", fontsize=fontsize*factor_fs,
                             horizontalalignment="center", verticalalignment="center")
    plt.colorbar(img, ax=axes)  # This adds a colorbar to the right of the FIGURE. However, the mapping from colors to values is taken from the last generated image! (which is ok because all images have the same range of values.
                            # Otherwise see answer by user10121139 in https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
    plt.suptitle(f"{learning_method.upper()}\nPolicy at each state")

print("{} learning process took {:.1f} minutes ({:.1f} hours)".format(learning_method.upper(), dict_time_elapsed[learning_method][rep] / 60, dict_time_elapsed[learning_method][rep] / 3600))


# Distribution of state counts at last learning step run of the last replication
# IMPORTANT: For FV, recall that the state counts only contain information about the states visited during the FV simulation as the counts are reset after the initial exploration
state_counts = simulator_value_functions.getAgent().getLearner().getStateCounts()
ax, img = test_ac.getEnv().plot_values(state_counts, cmap="Blues")
if learning_method_type == "values_fv" and not estimate_absorption_set_at_every_step:
    # Check that no visit was done to the absorption set
    print("Intersection between absorption set and state visit count > 0 in FV (IT SHOULD BE EMPTY! --recall that the state counts in FV are reset after the initial exploration):")
    _visited_states = set(np.where(state_counts > 0)[0])
    print(_visited_states.intersection(test_ac.getAbsorptionSet()))
    assert len(_visited_states.intersection(test_ac.getAbsorptionSet())) == 0, "The visited states during the FV excursion must NOT be in the absorption set"
if env_type == Environment.MountainCar:
    # Add the trajectory of the last replication
    # Note: Since no trajectory information is currently stored by the online policy learner, we use the value functions learner to extract the trajectory to plot.
    # Make the trajectories comparable between FV and TD, i.e. show the same number of steps (we show the steps generated by the T steps of the initial exploration by FV)
    T = max_time_steps_fv_for_expectation
    if learning_method_type == "values_fv":
        # Take the state distribution from the initial exploration in the FV case because the particle trajectories are NOT stored in the FV learner
        trajectory = np.array(test_ac.learner_for_initial_exploration.getStates())
    else:
        trajectory = np.concatenate(simulator_value_functions.agent.getLearner().getStates())
    assert T <= len(trajectory)
    trajectory2plot = trajectory[:T]
    test_ac.getEnv().plot_points(trajectory2plot, ax=ax, cmap="coolwarm", style=".-")
    states_absorption_set = np.nan*np.ones(test_ac.getEnv().getNumStates())
    states_absorption_set[list(test_ac.getAbsorptionSet())] = 1.0
    test_ac.getEnv().plot_values(states_absorption_set, ax=ax, cmap="Oranges", alpha=0.5)
    #test_ac.getEnv().plot_points(list(test_ac.getAbsorptionSet()), ax=ax, color="red", markersize=7, style="x")

simulator_value_functions._add_count_labels(ax, state_counts, factor_fontsize=2.0)
plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion - {env_type.name} {test_ac.getEnv().getShape()}"
             f"\nN={test_ac.agent_nn_fv.getLearner().getNumParticles()}, T={test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
             f"\nDistribution of state counts at end of policy learning process")


# Let's look at the trajectories of the learner (it works when constructing the learner with store_history_over_all_episodes=True)
#print(len(simulator_value_functions.getAgent().getLearner().getStates()))
#print([len(trajectory) for trajectory in simulator_value_functions.getAgent().getLearner().getStates()])


# Distribution of number of steps (possibly over all replications)
plot_average_nsteps = False
colors = ["green", "cyan", "orange", "magenta", "black"]
ax = plt.figure(figsize=(8, 7)).subplots(1, 1)
rep2plot = rep
if policy_learning_mode == "online":
    ax.plot(range(1, n_learning_steps+1), dict_R[learning_method][rep2plot, :n_learning_steps], marker='.', color=colors[rep2plot % len(colors)])
    ax.axhline(max_avg_reward_episodic, color="green", linewidth=1)
ax.plot(range(1, n_learning_steps+1), dict_R_long[learning_method][rep2plot, :n_learning_steps], marker='.', color=colors[rep2plot % len(colors)])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.axhline(max_avg_reward_continuing, color="greenyellow", linewidth=1)
ax.set_xlabel("Learning step")
ax.set_ylabel("Average reward")
ax_n = ax.twinx()
if plot_average_nsteps:
    ax_n.bar(np.arange(1, n_learning_steps+1), np.nanmean(dict_nsteps[learning_method], axis=0), color=colors[rep2plot % len(colors)], alpha=0.5)
else:
    ax_n.bar(np.arange(1, n_learning_steps+1), dict_nsteps[learning_method][rep2plot, :], color=colors[rep2plot % len(colors)], alpha=0.5)
ax_n.set_ylabel("Number of simulation steps")
plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion - {env_type.name} {env_shape}"
             f"\nN={test_ac.agent_nn_fv.getLearner().getNumParticles()}, T={test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
             f"\nAverage number of steps used to learn the value functions at each policy learning step ({nrep} replications)")



############## SAVE ALL RESULTS TOGETHER
if save:
    _env = test_ac.getEnv()
    wind_dict = None if "wind_dict" not in locals() else wind_dict
    exit_state = None if "exit_state" not in locals() else exit_state
    objects_to_save = ["_env", "env_type", "wind_dict", "learning_task", "learning_criterion", "gamma", "exit_state", "nn_hidden_layer_sizes", "is_NPG", "policy_learning_mode", "simulator_value_functions",
                       "dict_loss", "dict_R", "dict_R_long", "dict_R_long_true", "dict_V", "dict_Q", "dict_A", "dict_state_counts", "dict_nsteps", "dict_KL", "dict_alpha", "dict_time_elapsed",
                       "max_time_steps_benchmark"]
    if "max_time_steps_benchmark_all" in locals():
        objects_to_save += ["max_time_steps_benchmark_all"]
    else:
        objects_to_save += ["max_time_steps_benchmark"]
    _shape_str = f"{env_shape[0]}x{env_shape[1]}" if "env_shape" in locals() else ""
    _filename = f"{prefix}{dt_start_filename}_ALL.pkl"
    _filepath = os.path.join(resultsdir, _filename)
    # Save the original object names plus those with the prefix so that, when loading the data back,
    # we have also the objects without the prefix which are need to generate the plots right-away.
    # HOWEVER, when reading the saved data, make sure that any object with the original name (without the suffix) has been copied to another object
    # (e.g. loss_all_td could be a copy of loss_all before reading the data previously saved for the FV learning method)
    object_names_to_save = objects_to_save + [f"{obj_name}" for obj_name in objects_to_save]
    save_objects_to_pickle(object_names_to_save, _filepath, locals(), lib="pickle")
    print(f"Results for ALL Actor-Critic methods: {[str.replace(meth, 'values_', '').upper() for meth in dict_loss.keys()]} saved to '{_filepath}'")
############## SAVE ALL RESULTS TOGETHER



############## LOAD RESULTS
# FIRST, need to load the necessary modules at the top of this Section and compile the auxiliary functions

#_datetime = "20240428_092427" #"20240428_205959" #"20240421_140558" #"20240405_094608" #"20240322_095848" #"20240219_230301" #"20240219_142206"          # Use format yyymmdd_hhmmss
#_shape_str = "6x8" #"21x1" #"10x14" #"3x4" #"10x14"

resultsdir = "./RL-003-Classic/results"

_filename = "ActorCritic_labyrinth_4x5_20240512_232944_ALL_WindyLabyrinth0.7FinishAtBottomAbsorptionSetEstimated0_TDAC,FVAC,N=20,T=500,LimitedTime=5x,epsilon=0.10,Budget4Loss=5x_FVisBetter.pkl"
_filename = "ActorCritic_labyrinth_4x5_20240513_070501_ALL_WindyLabyrinth0.7FinishAtBottomAbsorptionSetEstimated_TDAC,FVAC,N=20,T=500,LimitedTime=5x,epsilon=0.10,Budget4Loss=5x_FVisBetter.pkl"
_filename = "ActorCritic_labyrinth_4x5_20240524_011541_ALL_WindyLabyrinth0.7FinishAtTopAbsorptionSetEstimate0_TDAC,FVAC,N=50,T=500,LimitedTime=5x,epsilon=0.10,OFFLINE_FVworksTDfails.pkl"
_filename = "ActorCritic_labyrinth_4x5_20240612_201558_values_fv.pkl"
_filename = "ActorCritic_labyrinth_4x5_20240613_073402_values_td.pkl"
_filename = "ActorCritic_labyrinth_4x5_20240613_151358_values_td.pkl"
_filename = "ActorCritic_labyrinth_4x5_20240607_132703_Alphonse_N=20,T=500,alphaA=0.05,UNnormalizedLoss_UsedInEWRL2024paper.pkl"    # DOESN'T WORK BECAUSE OF PICKLE INCOMPATIBILITY!!!
_filename = "ActorCritic_labyrinth_6x8_20240613_155454_values_fv_N=50,T=1000,alphaA=0.05,NormalizedLoss.pkl"

# Compare two FVAC learnings
# 4x5
#_filename = "ActorCritic_labyrinth_4x5_20240512_232944_ALL_WindyLabyrinth0.7FinishAtBottomAbsorptionSetEstimated0_TDAC,FVAC,N=20,T=500,LimitedTime=5x,epsilon=0.10,Budget4Loss=5x_FVisBetter.pkl"
#_filename = "ActorCritic_labyrinth_4x5_20240513_070501_ALL_WindyLabyrinth0.7FinishAtBottomAbsorptionSetEstimated_TDAC,FVAC,N=20,T=500,LimitedTime=5x,epsilon=0.10,Budget4Loss=5x_FVisBetter.pkl"
_filename = "ActorCritic_labyrinth_4x5_20240513_151849_ALL_WindyLabyrinth0.8FinishAtBottomAbsorptionSetEstimated0_TDAC,FVAC,N=20,T=500,LimitedTime=5x,epsilon=0.10,Budget4Loss=5x_FVisBetter.pkl"
_filename = "ActorCritic_labyrinth_4x5_20240513_124235_ALL_WindyLabyrinth0.8FinishAtBottomAbsorptionSetEstimated_TDAC,FVAC,N=20,T=500,LimitedTime=5x,epsilon=0.10,Budget4Loss=5x_FVisBetter.pkl"
# 6x8
_filename = "ActorCritic_labyrinth_6x8_20240515_111830_ALL_WindyLabyrinth0.8FinishAtBottomAbsorptionSetEstimated0_TDAC,FVAC,N=20,T=500,LimitedTime=5x,epsilon=0.10,Budget4Loss=5x_FVisBetter.pkl"
_filename = "ActorCritic_labyrinth_6x8_20240515_102102_ALL_WindyLabyrinth0.6FinishAtBottomAbsorptionSetEstimated_TDAC,FVAC,N=20,T=500,LimitedTime=5x,epsilon=0.10,Budget4Loss=5x_FVisBetter.pkl"

_shape_str = _filename[len("ActorCritic_labyrinth_"):len("ActorCritic_labyrinth_")+3]
_N = int(_filename[_filename.index("N=") + len("N="):_filename.index(",", _filename.index("N="))])
_T = int(_filename[_filename.index("T=") + len("T="):_filename.index(",", _filename.index("T="))])
_filepath = os.path.join(resultsdir, _filename)
object_names = load_objects_from_pickle(_filepath, globals())
print(f"The following objects were loaded from '{_filepath}':\n{object_names}")

# The following reward values are used in the ALTOGETHER plots below
_methods = list(dict_loss.keys())
env_shape = (int(_shape_str[:_shape_str.index("x")]), int(_shape_str[_shape_str.index("x")+1:]))
if _filename.lower().find("random") >= 0:
    # Do NOT compute the max average rewards in RANDOM labyrinths, because we don't know how to compute them
    max_avg_reward_continuing = np.nan
    max_avg_reward_episodic = np.nan
else:
    max_avg_reward_continuing, max_avg_reward_episodic = compute_max_avg_rewards_in_labyrinth_with_corridor(_env, wind_dict, learning_task, learning_criterion)
nrep = len(dict_loss[_methods[0]])
n_learning_steps = len(dict_loss[_methods[0]][0])
# (2024/08/04) The following objects are now read from the pickle file
#learning_task = LearningTask.CONTINUING
#policy_learning_mode = "online"
#exit_state = None
#nn_hidden_layer_sizes = [12]
dict_colors, dict_linestyles, dict_legends, figsize = define_plotting_parameters()
############## LOAD RESULTS


#-- ALTOGETHER PLOT
# Plot all average rewards together (to compare methods)
# Color names are listed here: https://matplotlib.org/stable/gallery/color/named_colors.html
# We normalize the average reward plots so that they converge to 1.0 (easier interpretation of the plot)
if is_NPG or policy_learning_mode != "online":
    dict_R_toplot = dict_R_long
    max_avg_reward = max_avg_reward_continuing
else:
    dict_R_toplot = dict_R
    max_avg_reward = max_avg_reward_episodic
# Check that the max avg. reward is defined, if not set it to 1.0 so that we plot the unnormalized observed average reward
max_avg_reward = 1.0 if np.isnan(max_avg_reward) else max_avg_reward

_exit_state_str = 'TOP' if "exit_state" in locals() and exit_state is None else 'BOTTOM' if "exit_state" in locals() else "RIGHT"
_learning_characteristics = f"\nN={'test_ac' in locals() and test_ac.agent_nn_fv.getLearner().getNumParticles() or _N}, " + \
                            f"T={'test_ac' in locals() and test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation() or _T}, " + \
                            f"MAX budget={'max_time_steps_benchmark' in locals() and max_time_steps_benchmark or 'N/A'} steps - NN hidden layer: {nn_hidden_layer_sizes}, " + \
                            f"Policy Learning MODE: {policy_learning_mode.upper()}"

if "n_learning_steps" not in locals():
    n_learning_steps = len(dict_loss[list(dict_loss.keys())[0]][0])
if "policy_learning_mode" not in locals():
    policy_learning_mode = "online"
ax_loss, ax_R = plt.figure(figsize=figsize).subplots(1, 2)
legend = []
for meth in dict_loss.keys():
    ax_loss.plot(np.arange(1, n_learning_steps + 1), dict_loss[meth][nrep-1, :n_learning_steps], '-', marker='.', color=dict_colors[meth])
    #ax_R_true = ax_loss.twinx()
    #ax_R_true.plot(np.arange(1, n_learning_steps + 1), dict_R_long_true[meth][nrep-1, :n_learning_steps], '-', color="blue")
    #ax_R_true.axhline(max_avg_reward_continuing, color="blue", linestyle="dashed", linewidth=2)
    #ax_R_true.set_ylim((0, None))
    #ax_R_true.set_ylabel("Expected reward under current policy (log scale)")
    legend += [f"{dict_legends[meth]} (average reward)"]
    ax_R.plot(np.arange(1, n_learning_steps+1), dict_R_toplot[meth][nrep-1, :n_learning_steps] / max_avg_reward, '-', marker='.', color=dict_colors[meth])
    # True average reward (it should give a good fit of the average reward points just plotted
    ax_R.plot(np.arange(1, n_learning_steps + 1), dict_R_long_true[meth][nrep-1, :n_learning_steps] / max_avg_reward_continuing, '-', color=dict_colors[meth], linestyle="dashed")
    legend += [f"{dict_legends[meth]} (expected reward)"]
ax_loss.set_xlabel("Learning step")
ax_loss.set_ylabel("Loss")
ax_loss.axhline(0, color="gray")
ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_loss.set_title(f"Evolution of LOSS")
ax_loss.legend(legend)
ax_R.set_xlabel("Learning step")
ax_R.set_ylabel("Average reward (normalized by the MAX average reward = {:.2g})".format(max_avg_reward))
if max_avg_reward != 1.0:
    # This is the case when the max average reward is not known, so we are NOT plotting the *normalized* average reward and showing the 1.0 line is not informative and goes out of scale
    ax_R.axhline(1, color="gray")
ax_R.axhline(0, color="gray")
ax_R.set_title(f"Evolution of NORMALIZED Average Reward")
ax_R.legend(legend, loc="lower left")
plt.suptitle(f"ALL LEARNING METHODS: {env_type.name} {env_shape} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={gamma})" +
             _learning_characteristics +
             f"\n(last replication #{nrep})")

# If we want to add the ratio between number of steps used by two methods compared
if "values_td" in dict_nsteps.keys() and "values_fv" in dict_nsteps.keys():
    df_ratio_nsamples = pd.DataFrame({'td': dict_nsteps['values_td'][nrep-1, :n_learning_steps], 'fv': dict_nsteps['values_fv'][nrep-1, :n_learning_steps], 'ratio_fv_td': dict_nsteps['values_fv'][nrep-1, :n_learning_steps] / dict_nsteps['values_td'][nrep-1, :n_learning_steps]})
    ax_R_nsamples = ax_R.twinx()
    ax_R_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'][:n_learning_steps], color="blue", linewidth=0.5)
    ax_R_nsamples.axhline(1.0, color="blue", linewidth=0.5, linestyle="dashed")
    ax_R_nsamples.set_ylim((ax_R.get_ylim()[0], None))
    ax_R_nsamples.legend(["Sample size ratio (FV/TD)", "Reference line showing equal sample size ratio"], loc="lower right")

# Plot all replications individually
ax = plt.figure(figsize=figsize).subplots(1, 1)
lines = []
legend = []
for rep in range(nrep):
    for meth in dict_R_toplot.keys():
        line = ax.plot(np.arange(1, n_learning_steps+1), dict_R_toplot[meth][rep, :n_learning_steps], '-', color=dict_colors[meth], linewidth=0.3)
        lines += line if rep == 0 else []
        legend += [meth] if rep == 0 else []
        #ax.axhline(0, color="gray")
        ax.set_xlabel("Learning step")
        ax.set_ylabel("Average reward")
line = ax.axhline(max_avg_reward_episodic, color="darkgreen") if policy_learning_mode == "online" else None
line = ax.axhline(max_avg_reward_continuing, color="lightgreen") if policy_learning_mode == "offline" else None
lines += [line]
legend += ["Max. average reward" + (policy_learning_mode == "online" and " (episodic)" or " (continuing)")]
ax.legend(lines, legend, loc="center right")
plt.suptitle(f"ALL LEARNING METHODS: {env_type.name} {env_shape} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={gamma})"
             f"\nWIND: {wind_dict}, EXIT: {_exit_state_str}, ALL {nrep} replications" +
             _learning_characteristics)

# Plot results on several replications
if nrep > 1:
    plot_bands = plot_for_paper #True #False
    plot_mean = not plot_for_paper #False #True
    dict_stats_R = dict.fromkeys(dict_loss.keys())
    ax = plt.figure(figsize=figsize).subplots(1, 1)
    lines = []
    legend = []
    _xshift = -0.1 # shift on the X axis to avoid overlap of vertical error bars
    for meth in dict_loss.keys():
        # Adapt the following filter if we want to exclude a particular method from the comparison plot
        #if meth in ["values_fv"]:
        #    continue
        _xshift += 0.1
        # Compute distribution of values to plot
        dict_stats_R[meth] = dict()
        dict_stats_R[meth]['min'], \
        dict_stats_R[meth]['max'], \
        dict_stats_R[meth]['median'], \
        dict_stats_R[meth]['mean'], \
        dict_stats_R[meth]['std'], \
        dict_stats_R[meth]['n'] = dict_R_toplot[meth].min(axis=0), dict_R_toplot[meth].max(axis=0), np.median(dict_R_toplot[meth], axis=0), dict_R_toplot[meth].mean(axis=0), dict_R_toplot[meth].std(axis=0), len(dict_R_toplot[meth])
        # Percentiles (if needed)
        # percentiles_low = [10, 25]
        # percentiles_upp = [90, 75]
        # alphas = [0.10, 0.15, 0.20]
        # percentiles = pd.DataFrame({'replication': np.array([np.repeat(r, n_learning_steps) for r in range(1, nrep+1)]).reshape(-1),
        #                             'step': np.array([np.repeat(s, R) for s in range(n_learning_steps)]).T.reshape(-1),
        #                             'R': R_all.reshape(-1)}, columns=['replication', 'state', 'V'])[['state', 'V']] \
        #     .groupby('state') \
        #     .agg(['count', 'mean', 'min', 'median', 'max', 'std'] + [percentile(p) for p in percentiles_low] + [percentile(p) for p in percentiles_upp])

        # Plot
        _xvalues = np.arange(1, n_learning_steps+1) + _xshift
        if plot_mean:
            #line = ax.plot(_xvalues, dict_stats_R[meth]['mean'] / max_avg_reward, color=dict_colors[meth], linestyle=dict_linestyles[meth], linewidth=2)[0]
            line = ax.errorbar(_xvalues, dict_stats_R[meth]['mean'][:n_learning_steps] / max_avg_reward, yerr=dict_stats_R[meth]['std'][:n_learning_steps] / np.sqrt(dict_stats_R[meth]['n']) / max_avg_reward, color=dict_colors[meth], linestyle=dict_linestyles[meth], linewidth=2, marker=".", markersize=12)[0]
            lines += [line]
            legend += [f"{dict_legends[meth]} (average +/- SE)"]
        line = ax.plot(_xvalues, dict_stats_R[meth]['median'][:n_learning_steps] / max_avg_reward, color=dict_colors[meth], linestyle="dashed" if plot_mean else "solid", linewidth=2, marker="x" if plot_mean else ".", markersize=12)[0]
        lines += [line]
        legend += [f"{dict_legends[meth]} (median)"]
        if plot_bands:
            line = ax.plot(_xvalues, dict_stats_R[meth]['max'][:n_learning_steps] / max_avg_reward, color=dict_colors[meth], linestyle="dashed")[0]
            lines += [line]
            legend += [f"{dict_legends[meth]} (min/max)"]
            ax.plot(_xvalues, dict_stats_R[meth]['min'][:n_learning_steps] / max_avg_reward, color=dict_colors[meth], linestyle="dashed")
            ax.fill_between(_xvalues,
                            dict_stats_R[meth]['max'][:n_learning_steps] / max_avg_reward,
                            dict_stats_R[meth]['min'][:n_learning_steps] / max_avg_reward,
                            color=dict_colors[meth],
                            alpha=0.1)
    ax.axhline(1, color="gray")
    ax.legend(lines, legend, loc="center left")
    ax.set_ylim((-0.01, 1.01))
    ax.set_xlabel("Learning step")
    ax.set_ylabel("Average reward (normalized by the MAX average reward = {:.2g})".format(max_avg_reward))
    plt.suptitle(f"ALL LEARNING METHODS: Labyrinth {env_shape} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={gamma})"
                 f"\nWIND: {wind_dict}, EXIT: {_exit_state_str}, {nrep} replications" +
                 _learning_characteristics)

    # Plot of number of samples ratios between FV learnings and TD learning
    legend_nsamples = []
    if "values_td" in dict_nsteps.keys() and "values_fv" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.mean(dict_nsteps['values_td'], axis=0)[:n_learning_steps],
                                          'fv': np.mean(dict_nsteps['values_fv'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'], color="blue", linewidth=0.5)
        ref_line = ax_nsamples.axhline(1.0, color="blue", linewidth=0.5, linestyle="dashed")
        legend_nsamples += ["Sample size ratio (FV/TD)", "Reference line showing equal sample size ratio"]
    if "values_td" in dict_nsteps.keys() and "values_fv2" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.mean(dict_nsteps['values_td'], axis=0)[:n_learning_steps],
                                          'fv': np.mean(dict_nsteps['values_fv2'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'], color="orange", linewidth=0.5)
        ref_line = ax_nsamples.axhline(1.0, color="orange", linewidth=0.5, linestyle="dashed")
        legend_nsamples += ["Sample size ratio (FV2/TD)", "Reference line showing equal sample size ratio"]
    if "values_td2" in dict_nsteps.keys() and "values_fv" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.mean(dict_nsteps['values_td2'], axis=0)[:n_learning_steps],
                                          'fv': np.mean(dict_nsteps['values_fv'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        if "ax_nsamples" not in locals():
            ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'], color="cyan", linewidth=0.5)
        legend_nsamples += ["Sample size ratio (FV/TD2)"]
        if "ref_line" not in locals():
            ref_line = ax_nsamples.axhline(1.0, color="cyan", linewidth=0.5, linestyle="dashed")
            legend_nsamples += ["Reference line showing equal sample size ratio"]
    if "values_td2" in dict_nsteps.keys() and "values_fv2" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.mean(dict_nsteps['values_td2'], axis=0)[:n_learning_steps],
                                          'fv': np.mean(dict_nsteps['values_fv2'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        if "ax_nsamples" not in locals():
            ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps + 1), df_ratio_nsamples['ratio_fv_td'], color="magenta", linewidth=0.5)
        legend_nsamples += ["Sample size ratio (FV2/TD2)"]
        if "ref_line" not in locals():
            ax_nsamples.axhline(1.0 - 1E-6, color="magenta", linewidth=0.5, linestyle="dashed")
            legend_nsamples += ["Reference line showing equal sample size ratio"]
        ax_nsamples.set_ylim((ax.get_ylim()[0], None))
    if "values_fv2" in dict_nsteps.keys() and "values_fv" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'fv2': np.mean(dict_nsteps['values_fv2'], axis=0)[:n_learning_steps],
                                          'fv': np.mean(dict_nsteps['values_fv'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv2_fv'] = df_ratio_nsamples['fv2'] / df_ratio_nsamples['fv']
        if "ax_nsamples" not in locals():
            ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv2_fv'], color="cyan", linewidth=0.5)
        legend_nsamples += ["Sample size ratio (FV2/FV)"]
        if "ref_line" not in locals():
            ref_line = ax_nsamples.axhline(1.0, color="cyan", linewidth=0.5, linestyle="dashed")
            legend_nsamples += ["Reference line showing equal sample size ratio"]

    if "ax_nsamples" in locals():
        ax_nsamples.set_ylim((ax.get_ylim()[0], None))
        ax_nsamples.set_ylabel("Average sample Ratio FV/TD across replications")
        ax_nsamples.legend(legend_nsamples, loc="lower right")
#-- ALTOGETHER PLOT
#------------------ Plots -----------------


raise KeyboardInterrupt


########
# 2023/03/08: Test the package optparse to parse arguments when calling a script from the command prompt, specially
# its capabilities of parsing an argument that should be interpreted as a list.
# Goal: Run several replications of the same simulation (using different seeds)
# Ref:
# https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
# https://docs.python.org/3.7/library/optparse.html#option-callbacks

import optparse

# ---------------------------- Auxiliary functions ---------------------------#
def convert_str_to_list_of_floats(option, opt, value, parser):
    print(f"option: {type(option)}, {dir(option)}")
    print(f"opt: {opt}")
    print(f"value: {value}")
    print(f"parser: {parser}")
    print(f"parser.values: {parser.values}")
    if isinstance(value, str):
        setattr(parser.values, option.dest, [float(s) for s in value.split(",")])

def parse_input_parameters(argv):
    # Written for uugot.it project in Apr-2021
    # Parse input parameters
    # Ref: https://docs.python.org/3.7/library/optparse.html
    # Main steps:
    # 1) The option parser is initialized with optparse.OptionParser(), where we can specify the usage= and version=,
    # as e.g. `optparse.OptionParser(usage="%prog [-v] [-p]", version="%prog 1.0")`
    # 2) New options to parse are added with parser.add_option(), where the metavar= argument (e.g. `metavar="FILE"`)
    # is used to indicate the option expects a value to be specified (e.g. `--filename="file.txt"` as opposed to `--verbose`, which expects no value).
    # We can also define:
    #    a) the default value of the option (although this is more clearly done with parser.set_defaults()).
    #    b) the action to take with the option value read with the action= argument, e.g. "store_true", "store_false",
    #       which are actually needed for FLAG options, which do NOT require any option value (e.g. -v for verbose, etc.),
    #       and ***whose default value (i.e. when the flag is not given) is specified by the default= parameter***.
    #       The default action is "store" which is used for options accepting a value as in `--file="file.txt".
    #       --> NOTE that the action can be "callback" meaning that a callback function with the signature callback(option, opt, value, parser)
    #       is called to parse the argument value. In this case, if the value of the argument needs to be updated
    #       (e.g. a string converted to a list) we need to:
    #       - define the name of the variable to set with the `dest=` option of the parser.add_option() method
    #       - set the value of the argument in the callback by calling `setattr(parser.values, option.dest, <value>)`.
    #       Ref: https://docs.python.org/3.7/library/optparse.html#option-callbacks
    #    b) the type of the option value expected with the type= argument (e.g. type="int"), which defaults to "string".
    #    c) the store destination with the dest= argument defining the attribute name of the `options` object
    #       created when running parser.parse_args() (see next item) where the option value is stored.
    #       See more details about the default value of dest= below.
    # 3) Options are parsed with parser.parse_args() into a tuple (options, args), where `options` is an object
    # that contains all the name-value pair options and `args` is an object containing the positional parameters
    # that come after all other options have been passed (e.g. `-v --file="file.txt" arg1 arg2`).
    # 4) Every option read is stored as an attribute of the `options` object created by parser.parse_args()
    # whose name is the value specified by the dest= parameter of the parser.add_option() method, or its
    # (intelligent) default if none is specified (e.g. the option '--model_pos' is stored as options.model_pos by default)
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage="%prog [--method] [--t_learn] [--replications] [--theta_ref] [--theta_start] [-N] [-T] [--J_factor] [-d] [-v]")
    parser.add_option("--method",
                      metavar="METHOD",
                      help="Learning method [default: %default]")
    parser.add_option("--t_learn",
                      type="int",
                      metavar="# Learning steps",
                      help="Number of learning steps [default: %default]")
    parser.add_option("--replications",
                      type="int",
                      metavar="# Replications", default=1,
                      help="Number of replications to run [default: %default]")
    parser.add_option("--theta_start", dest="theta_start",
                      type="str",
                      action="callback",
                      callback=convert_str_to_list_of_floats,
                      metavar="Initial theta",
                      help="Initial theta parameter for the learning process [default: %default]")
    parser.add_option("-N",
                      type="int",
                      metavar="# particles", default=1,
                      help="Number of Fleming-Viot particles [default: %default]")
    parser.add_option("-T",
                      type="int",
                      metavar="# arrival events", default=100,
                      help="Number of arrival events to observe before ending the simulation [default: %default]")
    parser.add_option("-d", "--debug", dest="debug", default=False,
                      action="store_true",
                      help="debug mode")
    parser.add_option("-v", "--verbose", dest="verbose", default=False,
                      action="store_true",
                      help="verbose: show relevant messages in the log")

    parser.set_defaults(method="MC",
                        t_learn=10,
                        theta_start=0.1)

    (options, args) = parser.parse_args(argv)

    print("Parsed command line options: " + repr(options))

    # options: dictionary with name-value pairs
    # args: argument values (which do not require an argument name
    return options, args
# ---------------------------- Auxiliary functions ---------------------------#

# Parse input parameters
options, args = parse_input_parameters(sys.argv[1:])
print(f"options: {options}")
print(f"args: {args}")

raise KeyboardInterrupt


########
# 2021/02/03: Test a generator of value combinations.
# Goal: Generate all possible states of a multidimensional Markov chain.
import numpy as np
from math import factorial
from time import time

from Python.lib.utils.computing import all_combos_with_sum, comb

def prob(n, const, rho):
    return np.prod( [(1- r)*r**nr for r, nr in zip(rho, n)] ) / const

C = 20
R = 3
rho = [0.5]*R
#rho = [0.2, 0.875, 0.833]

time_start = time()
const = 0
ncases_total = 0
prod = [0]*(C+1)   # Array to store the contributions to the normalizing for each 1 <= c <= C
for c in range(C+1):
    print("Computing normalizing constant for R={}, c={}...".format(R, c), end=" ")
    ncases = comb(c+R-1,c)
    combos = all_combos_with_sum(R, c)
    count = 0
    while True:
        try:
            v = next(combos)
            #print(v, end=" ")
            assert len(v) == len(rho), "The length of v and rho coincide ({}, {})".format(len(v), len(rho))
            prod[c] += np.prod( [(1- r)*r**nr for r, nr in zip(rho, v)] )
            count += 1
        except StopIteration as e:
            #print("END!")
            break
    combos.close()
    const += prod[c]
    print("--> generated combinations: {}".format(count))
    #print("prod: {}".format(prod))
    #print("const: {}".format(const))
    assert count == ncases
    ncases_total += ncases
assert const <= 1, "The normalizing constant is <= 1"
assert abs(sum(prod)/const - 1.0) < 1E-6

# Blocking probability
pblock = prod[C] / const
time_end = time()

print("\nExecution time: {} sec".format(time_end - time_start))
print("Total number of cases: {}".format(ncases_total))
print("Normalizing constant for rho={}: {}".format(rho, const))
print("Blocking probability: Pr(C)={:.5f}%".format(pblock*100))
