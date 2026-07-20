# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:00:15 2021

@author: Daniel Mastropietro
"""

#import runpy
#runpy.run_path('../../setup.py')


######## Tests on FVAC (Fleming-Viot Actor-Critic)
# 2023/10/12: Learn an actor-critic policy using neural networks (with the torch package)
# Learning happens with the ActorCriticNN learner which defines a loss of type `tensor` which can be minimized using the backward() method of torch Tensors
# IT WORKS!

from timeit import default_timer as timer
from time import process_time
import os
import copy
import time
from enum import Enum, unique
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import MaxNLocator
from scipy.special import rel_entr

from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners import LearningCriterion, LearningTask, LearningMode
from Python.lib.agents.learners.policies import LeaActorCriticNN
from Python.lib.agents.policies import probabilistic

from Python.lib.environments.gridworlds import Direction2D
from Python.lib.estimators.nn_models import InputLayer
from Python.lib.simulators.fv import StoppingCriterion

from Python.lib.utils.basic import get_current_datetime_as_string, load_objects_from_pickle, log_file_open, log_file_close, save_objects_to_pickle, set_numpy_options, reset_numpy_options

from Python.test.test_optimizers_discretetime import Test_EstPolicy_EnvGridworldsWithObstacles, Test_EstPolicy_EnvMountainCar
from Python.test.test_utils import Environment, KL_THRESHOLD, compute_normalized_KL_distance, compute_max_avg_rewards_in_labyrinth_with_corridor, compute_true_state_value_function,\
    define_plotting_parameters, plot_policy, plot_state_counts, plot_trajectory, show_elapsed_time


if os.getcwd()[-4:] == "test":
    rootdir = os.path.realpath("../../RL-003-Classic")
else:
    rootdir = os.path.realpath("./RL-003-Classic")
print(f"Root directory is: {rootdir}")
resultsdir = f"{rootdir}/results"
logsdir = f"{rootdir}/logs"


# Types of environments that can be defined
@unique
class Environment(Enum):
    Gridworld = 1
    MountainCar = 2


#----------------- BASIC SETUP AND SIMULATION PARAMETERS --------------#
# Learning task and learning criterion are used by the constructor of the test class below
learning_task = LearningTask.CONTINUING
#learning_task = LearningTask.EPISODIC

learning_criterion = LearningCriterion.AVERAGE; gamma = 1.0    # gamma could be < 1 in the average reward criterion in order to take the limit as gamma -> 1 as presented in Sutton, pag. 251/252.
#learning_criterion = LearningCriterion.DISCOUNTED; gamma = 0.9

seed = 1317  # Seed used when creating the test object in test_optimizers_discretetime.py, which in turn is used for: (i) the policy seed at creation (torch.manual_seed(seed)), (ii) the eventual reset seed for value functions in learners (reset_seed=seed) at their creation
env_type = Environment.Gridworld
#env_type = Environment.MountainCar
env_type_name = env_type.name   # The environment NAME is retrieved to avoid an error that happened at least once (Jun-2025) when saving results to a pickle file: "Can't pickle <enum 'Environment'>: attribute lookup Environment on __main__ failed"
problem_2d = True
use_random_obstacles_set = True; prop_obstacles = 0.3; #0.5;
seed_obstacles = 4217 #4215    # Seed 4217 with 50% of obstacles gives good results in the 6x8 labyrinth
exit_state_at_bottom = False #True
estimate_absorption_set = True; threshold_absorption_set = 0.90 if env_type == Environment.Gridworld else 0.90  # Cumulative relative visit threshold
entry_state_in_absorption_set = True   #False #True     # Only used when estimate_absorption_set = False
#----------------- BASIC SETUP AND SIMULATION PARAMETERS --------------#


#-------------------------------- ENVIRONMENT -------------------------#
if env_type == Environment.Gridworld:
    if problem_2d:
        # 2D labyrinth
        size_vertical = 3; size_horizontal = 4
        size_vertical = 4; size_horizontal = 5
        size_vertical = 6; size_horizontal = 8
        #size_vertical = 8; size_horizontal = 12
        #size_vertical = 9; size_horizontal = 13
        size_vertical = 10; size_horizontal = 14
        #size_vertical = 10; size_horizontal = 30

        # Square labyrinths
        #size_vertical = 15
        #size_horizontal = size_vertical

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
    exit_state = entry_state + env_shape[1] - 1 if exit_state_at_bottom else env_shape[1] - 1

    # Presence of wind: direction and probability of deviation in that direction when moving
    if problem_2d:
        wind_dict = None
        #wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.1})
        #wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.3})    # 22-Jun-2025: Used in 8x12 gridworld
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
    # Define variables that are always stored as part of the params_exec dictionary, regardless of the environment
    entry_state = None
    exit_state = None
    wind_dict = None    # Just for information purposes in titles, etc.
#-------------------------------- ENVIRONMENT -------------------------#


# 2024/09/22: Temporary setup of the small labyrinth for the camera-ready version of the EWRL-2024 paper, which makes reaching F a little harder,
# and thus gives a more interesting case to show in a paper.
# Put the obstacles more complicated when the Exit is at the bottom
# (this is an inverted L for the 4x5 labyrinth)
# Note:
if env_type == Environment.Gridworld and size_vertical == 4 and size_horizontal == 5:
    obstacles_set = set({7, 8, 13, 18})


#----------------------------- MODEL FOR POLICY -----------------------#
# Number of hidden layers in the neural network model
# Using multiple layers whose size is proportional to the gridworld size... however this tends to be counterproductive...
# i.e. learning is slower and may fail (e.g. it usually converges to a non-optimal policy where the advantage function is 0), presumably because of the larger number of parameters.
# Perhaps the architecture would work if the learning step of the neural network parameters optimizer is smaller and
# a larger number of policy learning steps is used (I tried with policy_learning_rate = 0.01 instead of 0.1 and
# the parameters started to be learned (i.e. the average reward went up --as opposed to 0-- although with large oscillations)
# but still 30 learning steps did not suffice to learn completely.
# This was tried with the 6x8 gridworld with a large rectangle of states going nowhere at the upper-left part
# with adaptive TD(lambda), where the hidden layer sizes were set to [38, 19].
#nn_hidden_layer_sizes = [int( 0.8*np.prod(env_shape) ), int( 0.4*np.prod(env_shape) )]
# Keep the neural network rather small or do NOT use any hidden layer for Natural Policy Gradient (NPG)
# To learn more about NN architecture, see the following references, but essentially:
# - adding more hidden layers doesn't improve performance much, so we can just use ONE layer.
# - the number of neurons in the hidden layer is suggested to be the average between input and output neurons,
# but I tried this for our small number of neurons but the network didn't learn at all! (here I am talking about an NN to approximate value functions)
# (2010) https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# (2021) https://www.reddit.com/r/MachineLearning/comments/mualkr/d_effective_ways_of_choosing_the_number_of
# (2021) https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3
# (2016) by Srikant: https://arxiv.org/abs/1610.04161 "Why Deep Neural Networks for Function Approximation?" where they show that the number of neurons needed in a shallow network
# (i.e. networks whose depth does NOT depend on the allowed maximum uniform error epsilon) to approximate a function increases exponentially with the inverse of epsilon,
# whereas deep networks (i.e. networks whose depth increases as 1/epsilon) require polylog(1/epsilon) neurons to achieve the epsilon uniform error.
# (2014) https://web.archive.org/web/20140721050413/http://www.heatonresearch.com/node/707, Jeff Heaton, "An introduction to neural networks for Java".
# He indicates what is possible (via a theorem I believe?) with NN with one and with two hidden layers.
nn_hidden_layer_sizes = nn_hidden_layer_sizes_policy = [] #[12]
# Number of input neurons (just one with the state value or one-per-state)
nn_input_policy = InputLayer.ONEHOT if len(nn_hidden_layer_sizes_policy) == 0 else InputLayer.STATE
print(f"Neural Network architecture:")
print(f"Input layer: {nn_input_policy}")
print(f"Hidden layers: {len(nn_hidden_layer_sizes_policy)} hidden layers of sizes {nn_hidden_layer_sizes_policy}")
#----------------------------- MODEL FOR POLICY -----------------------#


#----------------------------- MODEL FOR CRITIC -----------------------#
use_function_approximation = True
use_separate_target_model = True if use_function_approximation else False; update_period_target_model = 100 #100000
nn_input_value_functions = InputLayer.STATE  #InputLayer.ONEHOT
nn_hidden_layer_sizes_value_functions = [48] #[48, 12] #[128] #[12] #[48]
# Learning rate for value functions (only relevant when learning V(s) using stochastic approximation, NOT when using the in-built Adam optimizer)
# It should NOT be too large when learning via NN using TD(lambda) as opposed to TD(0) (see comments in main section of value_functions.py)
# Use 1.0 when using lambda = 0 and learning using the NN optimizer (e.g. Adam), o.w. use 0.1 to avoid too large updates of the value functions.
# NOTE that this value is ONLY RELEVANT when we learn V(s) using stochastic approximation, as opposed to using the Adam optimizer,
# because in the latter case, the learning rate is set by the Adam optimizer.
# Recall also that the GAE (Generalized Advantage Estimation) does NOT have a learning rate (see my SPSS notebook for a proof on entry dated 09-Oct-2025).
alpha_initial = 0.1 if use_function_approximation else 1.0      # Also tried 0.001 as alpha_initial (20-Oct-2025) but it did not work in 8x12 labyrinth
#----------------------------- MODEL FOR CRITIC -----------------------#


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


# 2024/10/15: In case we need to define a particular absorption set to compare results (e.g. FVAC with NPG and FVAC without NPG)
# (because I've observed that the estimated absorption set could vary a lot with different seeds within the same policy
# and across different policy parameterizations (e.g. NPG vs. non-NPG), which lead to different actions taken!)
#default_absorption_set = {24, 25, 33, 40, 41}
#estimate_absorption_set = False


#-------------------------------- TEST SETUP --------------------------#
if env_type == Environment.Gridworld:
    N = 50  #20 #50, #200, #200 if problem_2d else 100, #50 #20 #100
    T = 500 #100 #500 #1000, #100, #10000 if problem_2d else 1000, #1000, #1000, #3000,  # np.prod(env_shape) * 10  #100 #1000
    dropout_policy = 0.0  #0.5 #0.5  # Set it to 0.0 if we do not want any dropout layer in the network
    test_ac = Test_EstPolicy_EnvGridworldsWithObstacles()
    test_ac.setUpClass(shape=env_shape, obstacles_set=obstacles_set, n_obstacles=n_obstacles, wind_dict=wind_dict,
                       define_start_state_from_absorption_set=False, start_states_set={entry_state},  #{nS-1}, #None,
                       exit_state=exit_state,
                       # Value functions model
                       use_function_approximation=use_function_approximation,
                       nn_input_value_functions=nn_input_value_functions,
                       nn_hidden_layer_sizes_value_functions=nn_hidden_layer_sizes_value_functions,
                       use_separate_target_model=use_separate_target_model, update_period_target_model=update_period_target_model,
                       # Policy model
                       nn_input_policy=nn_input_policy,
                       nn_hidden_layer_sizes_policy=nn_hidden_layer_sizes_policy,
                       dropout_policy=dropout_policy,
                       initial_policy=initial_policy,
                       # General learning parameters
                       learning_task=learning_task,
                       learning_criterion=learning_criterion,
                       alpha=alpha_initial, gamma=gamma, lmbda=0.7,  # lmbda parameter is ONLY used for TD(lambda), NOT for TD(0), which is created separately
                       alpha_min=alpha_initial/10,
                       reset_method_value_functions=ResetMethod.ALLZEROS,
                       # Fleming-Viot parameters
                       # Small N and T are N=50, T=1000 for the 8x12 labyrinth with corridor
                       N=N,
                       T=T,
                       estimate_absorption_set=estimate_absorption_set, threshold_absorption_set=threshold_absorption_set, absorption_set=default_absorption_set,
                       states_of_interest_fv=set({exit_state}),    #None
                       seed=seed, plot=True, debug=False,
                       seed_obstacles=seed_obstacles)
    test_ac.setUp()
    print(test_ac.policy_nn.nn_model)
elif env_type == Environment.MountainCar:
    N = 50 #30  #50
    T = 500 #300 #100 #300 #500
    env_discrete = not use_function_approximation #True #False
    dropout_policy = 0.0  #0.5
    initial_policy = [1/3, 1/3, 1/3]
    test_ac = Test_EstPolicy_EnvMountainCar()
    test_ac.setUpClass(# Environment characteristics
                       env_discrete=env_discrete,
                       nx=40,       # Number of points in the discretization of the positions (only used in the continuous-state-dynamic Mountain Car, in which case the "factor for force" parameter is not used)
                       nv=21,       # Number of points in the discretization of the velocities
                       factor_for_force_and_gravity=100, #10 if not env_discrete else 90, #100, #90, #20, #15,   # Factor controlling the number of discrete positions in the discretized problem --> NOTE: Using `1` is TOO SMALL! (as there are too many points in the grid)
                       factor_force=20, #1.0,
                       factor_max_speed=3.0,    # Only used in MountainCarDiscrete (with continuous states)
                       # Value functions model
                       use_function_approximation=use_function_approximation,
                       nn_input_value_functions=nn_input_value_functions,
                       nn_hidden_layer_sizes_value_functions=nn_hidden_layer_sizes_value_functions,
                       # Policy model
                       nn_input_policy=2,
                       nn_hidden_layer_sizes_policy=nn_hidden_layer_sizes_policy,
                       dropout_policy=dropout_policy,
                       initial_policy=initial_policy,
                       # General learning parameters
                       learning_task=learning_task,
                       learning_criterion=learning_criterion,
                       alpha=alpha_initial, gamma=gamma, lmbda=0.7, # lmbda parameter is ONLY used for TD(lambda), NOT for TD(0), which is created separately
                       alpha_min=alpha_initial/10,
                       reset_method_value_functions=ResetMethod.ALLZEROS,
                       reset_value=0.0, #-1.0,
                       N=N,
                       T=T,
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
    # When observing a particular state of interest
    state_observe = 42 if env_shape == (6, 8) else 126 if env_shape == (10, 14) else None #123
else:
    state_observe = None
    max_avg_reward_continuing = np.nan
    max_avg_reward_episodic = np.nan
#------------------ INITIAL AND MAXIMUM AVERAGE REWARDS ---------------#


#------------------ RESULTS COLLECTION AND PLOTS SETUP ----------------#
# Learning methods: define characteristics to use when plotting results for each method
dict_colors, dict_linestyles, dict_legends, figsize = define_plotting_parameters()

# Dictionaries to store the results for the different methods (for comparison purposes)
dict_test_ac = dict()
dict_simulator = dict()     # Store the simulator used for learning, from where we can get the agent, its policy, etc.
dict_loss = dict()
dict_R = dict()
dict_R_long = dict()
dict_R_long_initial = dict()
dict_R_long_fv_inflated = dict()
dict_R_long_true = dict()   # True average reward under the policy used at each policy learning step to learn value functions. GOAL: Keep track on how rare is reaching the reward.
dict_V = dict()
dict_Q = dict()
dict_A = dict()
dict_state_counts = dict()
dict_nsteps = dict()
dict_KL = dict()
dict_KL_norm = dict()
dict_alpha = dict()
dict_LR = dict()
dict_LR_policy = dict()
dict_time_elapsed = dict()
dict_time_cpu = dict()
#------------------ RESULTS COLLECTION AND PLOTS SETUP ----------------#


#----------------- Execution parameters
# FV learning parameters (which are used to define parameters of the other learners analyzed so that their comparison with FV is fair)
# Max average number of steps allowed for each particle in the FV simulation
# We set this value proportional to the size of the active set of the FV learner because the larger the active set the harder for the FV system to discover rewards
if env_type == Environment.MountainCar:
    max_time_steps_fv_per_particle = 30 #50 #100
else:
    max_time_steps_fv_per_particle = 30 #5*len(test_ac.agent_nn_fv.getLearner().getActiveSet()) #100 #50
# Parameter T in EWRL-2024 paper
max_time_steps_fv_for_expectation = T
#*********************
stopping_criterion_fv = StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES_AS_LONG_AS_ENOUGH_TIME_STEPS_HAVE_BEEN_TAKEN #StoppingCriterion.MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES #StoppingCriterion.MAX_TIME_STEPS #StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES #StoppingCriterion.MAX_TIME_STEPS
#*********************
# Parameter M1 in EWRL-2024 paper, which defines the first threshold for the maximum number of steps to run the FV system for,
# during which the first-time absorption of 100% of the particles makes the simulation stop.
M1 = max_time_steps_fv_for_all_particles = N * max_time_steps_fv_per_particle  #100 (for MountainCar)
# Parameter M2 in EWRL-2024 paper, which defines the second threshold for the maximum number of steps to run the FV system for,
# after which the simulation stops, regardless of the number of absorbed particles
# Use the following to avoid too large simulation times, for instance when the policy is close to optimal:
#M2 = max_time_steps_fv_overall = max(5000, max_time_steps_fv_for_all_particles)
M2 = max_time_steps_fv_overall = 2*max_time_steps_fv_for_all_particles  if stopping_criterion_fv in [StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES,
                                                                                                     StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES_AS_LONG_AS_ENOUGH_TIME_STEPS_HAVE_BEEN_TAKEN]\
                                                                        else max_time_steps_fv_for_all_particles #2*max_time_steps_fv_for_all_particles
min_prop_absorbed_particles = 0.90 #1.0 #0.90 #0.70    # WARNING: currently (2024/08/09) this ONLY has effect when M2 > M1!! So, if we want to use it just set M1 very small and M2 a value of the order of M1 usually used before
print(f"Thresholds for FV simulation: T={T}, M1 = {M1}, M2 = {M2}"
      f"\n% Absorbed particles required between M1 and M2: {min_prop_absorbed_particles*100}% (STOP when reached regardless of t? {stopping_criterion_fv})")

# Traditional method learning parameters
# They are set for a fair comparison with FV learning
# The maximum time steps to be observed in the benchmark methods is set to the sum of:
# - the number of steps used to estimate the absorption set A (as long as this exploration ALSO estimates value functions!)
# - the max number of time steps allowed to estimate E(T_A)
# - the max number of time steps allowed over all the FV particles
# Values previously used: #2000 #test_ac.getEnv().getNumStates()*10
# Number of steps used to estimate the absorption set A
time_steps_fv_for_absorption_set = test_ac.learner_for_initial_exploration.getNumSteps() if test_ac.learner_for_initial_exploration is not None else 0
assert time_steps_fv_for_absorption_set > 0 if estimate_absorption_set else True
# Include the time spent for the absorption set estimation in the budget (not really appropriate because: (i) in FVAC it is only run ONCE (at t=0); (ii) there is no value function learning during that exploration)
#max_time_steps_benchmark = time_steps_fv_for_absorption_set + max_time_steps_fv_for_expectation + max_time_steps_fv_overall
#print(f"max_time_steps_benchmark (T + T + M2) = {max_time_steps_benchmark}")
# Do NOT include the time spent for the absorption set estimation
max_time_steps_benchmark = max_time_steps_fv_for_expectation + max_time_steps_fv_overall #1500
print(f"max_time_steps_benchmark (T + M2) = {max_time_steps_benchmark}")
# If we need to run TDAC without any FVAC as benchmark, use the following setting for max_time_steps_benchmark or similar
#max_time_steps_benchmark = 1500
#print(f"max_time_steps_benchmark superseded by TDAC learning and set to {max_time_steps_benchmark}")

# Absorption strategy
estimate_absorption_set_at_every_step = True
update_absorption_set_with_fv_visits = True
soft_killing = False #True

#-- Common learning parameters (to all methods)
# 1) Parameters about POLICY learning (Actor)
policy_learning_mode = "online" #"offline" #"online"
    ## Whether the policy is learned ONLINE (i.e. by collecting trajectories at each policy estimate)
    ## or OFFLINE (where ALL states and actions are swept and the loss is computed on all of them using the state distribution as weights)
    ## The ONLINE approach can be used either when value functions are learned separately from the policy (i.e. where value functions serve as critic)
    ## or when they are learned at the same time (policy gradient, without critic).
    ## The OFFLINE mode makes sense only when value functions are learned SEPARATELY from the policy.
is_NPG = len(nn_hidden_layer_sizes_policy) == 0
#*********************
n_learning_steps = 100 #30 #20 #100 #30 #200 #50 #100
#*********************
prob_include_in_train = 1.0            # Probability of including a step of the exploration, used for the ONLINE policy learning update, in the sample that computes the loss. Goal: reduce the correlation among samples included in the training process.
n_episodes_per_learning_step = int(50 / prob_include_in_train) #100 #30  # Number of episodes for the policy update step when learning the policy online and in NON-NPG mode
# Max time steps per episode during exploration for the online policy learning
# In the Mountain Car problem we limit the number of steps per episode in the continuous-dynamics case because I've seen out-of-memory problems otherwise.
if env_type == Environment.Gridworld:
    _multiplier = 5
else:
    _multiplier = 1
max_time_steps_per_policy_learning_episode = _multiplier*test_ac.getEnv().getNumStates() if problem_2d else 2*test_ac.getEnv().getNumStates() #np.prod(env_shape) * 10 #max_time_steps_benchmark // n_episodes_per_learning_step   # Maximum number of steps per episode while LEARNING THE *POLICY* ONLINE (NOT used for the value functions (critic) learning)
allow_deterministic_policy = True #False
adjust_policy_learning_rate = False; t_learn_min_to_adjust_policy_learning_rate = 1   # This affects both the ONLINE learning of the policy AND the NPG learning
reset_value_functions_at_every_learning_step = False #(learning_method == "values_fv")     # Reset the value functions when learning with FV, o.w. the learning can become too unstable due to the oversampling of the states with high value... (or something like that)

# 2) Parameters about VALUE FUNCTION learning (Critic)
critic_learning_mode = LearningMode.ONLINE #LearningMode.BATCH #LearningMode.ONLINE
# DM-2025/08/16: The alpha_initial parameter is now set when we define the value function approximation strategy
#alpha_initial = 1.0 #simulator_value_functions.getAgent().getLearner().getInitialLearningRate()      # NOTE: alpha_initial is NOT used when learning the value functions by function approximation, as this is set by the default learning rate of the Adam optimizer
adjust_alpha_initial_by_learning_step = False; t_learn_min_to_adjust_alpha = 10 #30 # based at 1 (even if t_learn is based at 0)
#max_time_steps_per_episode = test_ac.getEnv().getNumStates()*10  # (2024/05/02) NO LONGER USED!  # This parameter is just set as a SAFEGUARD against being blocked in an episode at some state of which the agent could be liberated by restarting to a new episode (when this max number of steps is reached)
epsilon_random_action = 0.1 #if policy_learning_mode == "online" else 0.0 #0.1 #0.05 #0.0 #0.01
reward_to_promote_exploration = 0.0 #1.0 #0.1 #None   # Reward for a reward shaping strategy used to promote the visit of EXIT events from A which allow the execution of the FV simulation to estimate value functions (which is crucial for the FV estimation procedure to be effective). Note that the shaped reward may be proportional to |V(s)|, not necessarily constant
use_average_max_time_steps_in_td_learner = True #learning_method == "values_td2" #True #False
use_average_reward_from_previous_step = True #learning_method_type == "values_fv" #False #True            # Under the AVERAGE reward crtierion, whether to use the average reward estimated from the previous policy learning step as correction of the value functions (whenever it is not 0), at least as an initial estimate
use_fixed_average_reward = True #False # (2025/06/26) SETTING THIS PARAMETER TO True SEEMS TO BE VERY IMPORTANT IN GUARANTEEING STABILITY OF FVAC LEARNING (specially in situations where the absorption set A may have states close to the finish line in the labyrinth --e.g. 6x8 RANDOM labyrinth with WIND=0.6 (seed_labyrinth = 4217, seed (simulation) = 1317)
keep_fv_estimation_of_average_reward_and_stationary_probability_consistent = True #False  #True   # Use `False` when we are only interested in leveraging the reward information for policy learning as opposed to keeping its consistency with the estimated stationary probability (in terms of satysfing the equality \hat{E(R)} = \sum_{x} \hat{p(x)} r(x))
start_at_less_frequency_visited_states = False #True      # Whether to start the TD learning at a randomly selected state among those less visited by the exploration at the previous policy learning step. WARNING: Using True maybe highly counter-productive (e.g. 6x8 labyrinth 30% obstacles, no wind, seed 4217
learning_steps_observe = [7, 8, 23, 24] #[1, 2, 7, 8, 22, 23, 24] #[50, 90] #[2, 30, 48] #[2, 10, 11, 30, 31, 49, 50] #[7, 20, 30, 40]  # base at 1, regardless of the base value used for t_learn
verbose_period = max_time_steps_fv_for_all_particles // 10
plot = False         # Whether to plot the evolution of the state value function and average reward estimation
plot_policy_update = False  # Whether to plot the policy after each policy learning step update
colormap = "seismic"  # "Reds"  # Colormap to use in the plot of the estimated state value function V(s)

# A few further parameters for the policy learning process
break_when_no_change = False  # Whether to stop the learning process when the average reward doesn't change from one step to the next
break_when_goal_reached = False  # Whether to stop the learning process when the average reward is close enough to the maximum average reward (when this is known, by a relative tolerance of 0.1%)
#----------------- Execution parameters


# Define the BASE seed for the simulation
seed_base = test_ac.seed

# Number of replications to run on each method
nrep = 1 #3 #9
# Logging
log = nrep > 1  #learning_method_type == "values_fv"
save = False

# Learning method (of the value functions and the policy)
# Both value functions and policy are learned online using the same simulation
learning_method = "all_online"; simulator_value_functions = None
# Value functions are learned separately from the policy
# Policy learning can happen online or OFFLINE
# TD
learning_method = "values_td"; simulator_value_functions = test_ac.sim_td0      # TD(0)
#learning_method = "values_td2"; simulator_value_functions = test_ac.sim_td0    # TD(0)
learning_method = "values_tdl"; simulator_value_functions = test_ac.sim_td     # TD(lambda)
#learning_method = "values_tdl2"; simulator_value_functions = test_ac.sim_td     # TD(lambda)
#learning_method = "values_tda"; simulator_value_functions = test_ac.sim_tda    # Adaptive TD(lambda)
# FV
learning_method = "values_fv"; simulator_value_functions = test_ac.sim_fv
#learning_method = "values_fv2"; simulator_value_functions = test_ac.sim_fv
#learning_method = "values_fv3"; simulator_value_functions = test_ac.sim_fv
#learning_method = "values_fvos"; simulator_value_functions = test_ac.sim_fv     # FV used just as an oversampling method
learning_method = "values_fvl"; simulator_value_functions = test_ac.sim_fvl
#learning_method = "values_fvl2"; simulator_value_functions = test_ac.sim_fvl
#learning_method = "values_fvl3"; simulator_value_functions = test_ac.sim_fvl
#learning_method = "values_fva"; simulator_value_functions = test_ac.sim_fva


simulators = dict({ # TD and TD(lambda) learners
                    'values_td': test_ac.sim_td0,
                    'values_td2': test_ac.sim_td0,
                    'values_tdl': test_ac.sim_td,
                    'values_tdl2': test_ac.sim_td,
                    'values_tda': test_ac.sim_tda,
                    # FV learners
                    'values_fv': test_ac.sim_fv,
                    'values_fv2': test_ac.sim_fv,
                    'values_fv3': test_ac.sim_fv,
                    'values_fvos': test_ac.sim_fv,
                    # FV(lambda) learners
                    'values_fvl': test_ac.sim_fvl,
                    'values_fvl2': test_ac.sim_fvl,
                    'values_fvl3': test_ac.sim_fvl,
                    'values_fva': test_ac.sim_fva,
                    })
learning_methods = ["values_fvl"] #, "values_tdl", "values_fv", "values_td"] #["values_fv"] #["values_fv", "values_td"]
for learning_method in learning_methods:
    runfile(os.path.join(rootdir, "../Python/lib/run_policy_learning_discrete.py"), wdir=os.path.join(rootdir, "../Python/lib/"))

    # ------------------ Plots -----------------
    if learning_method != "all_online":  # Otherwise, `trajectory_under_policy` (used in the call below) is not defined
        plot_state_counts(dict_simulator, learning_method, rep, params_exec, trajectory=trajectory_under_policy, plot_absorption_set=True)

    if adjust_alpha_initial_by_learning_step:
        plt.figure()
        plt.plot(np.arange(1, n_learning_steps + 1), dict_alpha[learning_method][rep], color="cyan", marker=".")
        plt.title("Value functions learning rate by learning step")
        plt.gca().set_ylim((0.0, None))
    if adjust_policy_learning_rate:
        plt.figure()
        plt.plot(np.arange(1, n_learning_steps + 1), dict_LR_policy[learning_method][rep], color="orange", marker=".")
        plt.title("Policy learning rate by learning step")
        plt.gca().set_ylim((0.0, None))

raise KeyboardInterrupt

# How much the FV simulation contributes to the average reward value at each learning step
if learning_method_type == "values_fv":
    ax = plt.figure(figsize=figsize).subplots(1, 1)
    ax.plot(np.arange(1, n_learning_steps+1), dict_R_long[learning_method][rep, :n_learning_steps], marker='.', color="greenyellow")
    ax.plot(np.arange(1, n_learning_steps+1), dict_R_long_initial[learning_method][rep, :n_learning_steps], marker='.', color="magenta")
    ax.plot(np.arange(1, n_learning_steps+1), dict_R_long_fv_inflated[learning_method][rep, :n_learning_steps], marker='.', color="cyan")
    ax.set_xlabel("Learning step")
    ax.set_ylabel("Average reward")
    plt.title(f"{learning_method.upper()}" + f"{((' - SOFT' if soft_killing else ' - HARD') + ' killing') if learning_method_type == 'values_fv' else ''}" + "\nComparison between the Average Rewards")
    plt.legend(["Avg. Reward estimated by FV", "Avg. Reward from Initial Exploration", "Inflated Avg. Reward from FV simulation"])

if not plot_policy_update:
    _average_reward_at_last_policy_learning_step = dict_R[learning_method][rep][-1]
    _state_counts_during_policy_learning = dict_state_counts[learning_method][rep, :, :]
    axes = plot_policy(dict_simulator[learning_method][rep].getEnv(), dict_simulator[learning_method][rep].getAgent().getPolicy(),
                       _average_reward_at_last_policy_learning_step,
                       _state_counts_during_policy_learning,
                       params_exec,
                       is_problem_2d=problem_2d,
                       absorption_set=simulators_all[nrep-1].getAgent().getLearner().getAbsorptionSet() if learning_method_type == "values_fv" else None,
                       t_learn=n_learning_steps, fontsize=14, verbose=test_ac.getEnv().getNumStates() <= 50)
show_elapsed_time(learning_method, time_elapsed, time_cpu)


#raise KeyboardInterrupt

# Use this to generate a slow-paced trajectory drawing
test_ac.getEnv().plot_points(trajectory_under_policy, is_trajectory=True, pause=0.1)


#-- Plot the trajectory as a GIF
if env_type == Environment.MountainCar:
    # Plot the trajectory of the last replication
    # Note that we limit the trajectory to the first 100 steps because o.w. the GIF would take too long to generate...
    # For instance, with 100 steps, the GIF takes ~2 minutes to generate and is 1 MB in size already!
    _npoints2plot = min(T, 100)
    _simulator = copy.deepcopy(dict_simulator[learning_method][rep])
    _learner_latest_policy, _nsteps, _average_reward = _simulator.run_exploration(t_learn=t_learn, max_time_steps=T, epsilon_random_action=epsilon_random_action, seed=seed_learn, verbose=False, verbose_period=1)
    _trajectory_under_policy = np.array(_learner_latest_policy.getStates())
    assert _npoints2plot < len(_trajectory_under_policy)

    # If we want to make comparable plots (i.e. the same first T steps for both FVAC and TDAC)
    #trajectory2plot = _trajectory_under_policy[:_npoints2plot]
    # If we want to plot the beginning and end of the trajectory
    if _npoints2plot < T:
       # We repeat 10 times the last point in the first half of the points to plot so that we visually understand that there is a jump in time
       _indices2plot = np.r_[np.arange(_npoints2plot // 2), np.repeat(_npoints2plot // 2, 10), np.arange(T - _npoints2plot // 2, T)]
    else:
       _indices2plot = np.arange(T)
    trajectory2plot = _trajectory_under_policy[_indices2plot]

    dict_simulator[learning_method][rep].getEnv().plot_trajectory_gif([dict_simulator[learning_method][rep].getEnv().getStateFromIndex(s, simulation=False) for s in trajectory2plot])


#-- Plot the value functions for the state_observe
if state_observe is not None:
    # Plot the evolution of V(s) for the state_observe state
    V_values = np.concatenate(V_state_observe_all[rep, :n_learning_steps])
    ax_V = plt.figure().subplots(1, 1)
    ax_V.plot(V_values, 'r-')
    # Add the TRUE V(s) value at each policy learning step (if available), which the estimated V(s) should converge to,
    # and vertical divisors to show where each policy learning step starts.
    # Note: at each policy learning step, the length of the estimated V(s) values go from 0 to nsteps_all[rep, learning_step], i.e. it has ONE more recorded value that the number
    # of steps, because the very initial value at t = 0 is recorded and the last record happens at t = #steps of the simulation (e.g. if #steps = 1, then two values are recorded)
    _last_vertical_line_at = 0
    V_true_values = np.nan * np.ones(n_learning_steps)
    for t, v in enumerate(nsteps_all[rep, :n_learning_steps]):
        # TRUE state value (if available)
        if 'V_true_all' in locals():
            # Correct the TRUE V(s) value by its mean across states
            V_true_values[t] = V_true_all[rep, t, state_observe] - np.nanmean(V_true_all[rep, t, :])
            ax_V.plot([_last_vertical_line_at + 1, _last_vertical_line_at + v],
                      [V_true_values[t], V_true_values[t]], 'b-')
        ax_V.axvline(_last_vertical_line_at + v + 1, color="gray")
        _last_vertical_line_at += v + 1
    ax_V.set_title(f"State observe = {state_observe} {test_ac.getEnv().getStateFromIndex(state_observe, simulation=False)}")

    # Plot of Q(s,a) for all actions (whenever Q is estimated)
    if simulators[learning_method].getAgent().getLearner().Q is not None:
        marker = ''
        Q_all_baseline = dict_Q[learning_method][rep, :n_learning_steps, :, :] - np.tile(dict_V[learning_method][rep, :n_learning_steps, :].T, (dict_simulator[learning_method][rep].getEnv().getNumActions(), 1, 1)).T
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
            # - If the learning task is continuing, we still keep observing the terminal reward discounted by the length of the optimal path (= np.sum(env_shape) - 1)
            # (the `-1` at the end of the parenthesis cancels the `1+` at the beginning of the parenthesis when the learning task is CONTINUING)
            svalue  = reward_at_terminal * (1 + int(learning_task == LearningTask.CONTINUING) * (1 / (1 - gamma**(np.sum(env_shape)-1))) - 1)
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
        plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={dict_simulator[learning_method][rep].getAgent().getLearner().gamma}) - {env_type.name} {env_shape}"
                     f"\nN={N}, T={T}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
                     f"\nQ(s,a) and V(s) for state previous to the terminal state under the optimal policy, i.e. s={state_observe}\nMax average reward (continuing) = {max_avg_reward_continuing}")


















#******************************** V, Q, A ***********************************
# Same plot for all states
common_axes = True
plot_baseline = False   # Whether to plot Q(s,a) - V(s) (instead of Q(s,a))

axes = plt.figure(figsize=(10, 9)).subplots(env_shape[0], env_shape[1])
first_learning_step = 0 #n_learning_steps * 3 // 4  #0
y2max = int(round(np.max(dict_state_counts[learning_method])*1.1)) # For a common Y2-axis showing the state counts
min_V, max_V = np.min(dict_V[learning_method]), np.max(dict_V[learning_method])      # For a common Y-axis showing the value functions
min_Q, max_Q = np.min(dict_Q[learning_method]), np.max(dict_Q[learning_method])      # For a common Y-axis showing the value functions
Q_all_baseline = dict_Q[learning_method][rep, first_learning_step:n_learning_steps, :, :] - np.tile(dict_V[learning_method][rep, first_learning_step:n_learning_steps, :].T, (test_ac.getEnv().getNumActions(), 1, 1)).T
min_Q_baseline, max_Q_baseline = np.min(Q_all_baseline), np.max(Q_all_baseline)      # For a common Y-axis showing the value functions
ymin, ymax = min(min_V, min_Q), max(max_V, max_Q)
ymin_baseline, ymax_baseline = min(0, min_Q_baseline), max(0, max_Q_baseline)

if common_axes:
    ylim = (ymin_baseline, ymax_baseline) if plot_baseline else (ymin, ymax)
else:
    ylim = (None, None)
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
             f"\nN={N}, T={T}, wind_dict={wind_dict}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
             f"\nEvolution of the value functions V(s) and Q(s,a) with the learning step by state\nMaximum average reward (continuing): {max_avg_reward_continuing}")

# Plot of V(s): TRUE, estimated and TARGET (if available)
learner = dict_simulator[learning_method][rep].getAgent().getLearner()
plt.figure()
plt.axhline(0, color="gray")
plt.plot(learner.env.getV(), 'b.-')
plt.plot(learner.getV().getValues(), 'r.-')
plt.plot(learner.getV_target().getValues(), 'm.-')
plt.legend(["zero", "True V(s)", "Estimated V(s)", "Target V(s) model"])
# Plot Delta(V) and avg(R) in order to analyze the contributions to the TD error = 0 - avg(R) + Delta(V), where 0 is the reward received for all states except the terminl state
# and Delta(V) is actually the difference between V(S(t+1)) - V(S(t)), and we here plot the case when the action is going RIGHT, because of the way the 2D state is converted to 1D.
ax2 = plt.gca().twinx()
ax2.plot(np.diff(learner.getV().getValues()), 'k.-')
ax2.axhline(0, color="black", linestyle="dashed")
ax2.axhline(learner.getAverageReward(), color="green")
ax2.set_ylabel(r"$\Delta V(s)$ and avg(R)")


# Plot the ADVANTAGE function
marker = ''
first_learning_step = 0 #n_learning_steps * 3 // 4  #0
y2max = int(round(np.max(dict_state_counts[learning_method])*1.1)) # For a common Y2-axis showing the state counts
min_A, max_A = np.min(dict_A[learning_method]), np.max(dict_A[learning_method])      # For a common Y-axis showing the value functions
ymin, ymax = min_A, max_A

#-----------------
if env_type == Environment.MountainCar:
    # Plot just ONE advantage function
    s_observe = test_ac.getEnv().getNumStates() // 2
    ax = plt.figure(figsize=(10, 9)).subplots(1, 1)
    i = s_observe
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

common_axes = False
ylim = (ymin, ymax) if common_axes else (None, None)
marker = ''
axes = plt.figure(figsize=(10, 9)).subplots(env_shape[0], env_shape[1], sharex=common_axes, sharey=common_axes, gridspec_kw=dict(hspace=0.1, wspace=0.1))  # See also help(plt.subplots); help(matplotlib.gridspec.GridSpec)
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
             f"\nN={N}, T={T}, wind_dict={wind_dict}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
             f"\nEvolution of the Advantage function A(s,a) with the learning step by state\nMaximum average reward (continuing): {max_avg_reward_continuing}")
#******************************** V, Q, A ***********************************




# 2025/08/22
# Plot V(s) for selected states, so that we can compare their evolution
states = [(3,8), (4,8), (4,9), (5,8)]      # For the 10x14 (40% obstacles, seed=4217)
states = [(5,0), (6,0), (7,0), (8,0)]               # For the 10x14 (30% obstacles, seed=4217)
states4diff = [(7,0), (6,0)]    # Two states that are used to evaluate the difference delta(V) = V(S[1]) - V(S[0]) which appears in the calculation of the TD error
colormap = cm.get_cmap("rainbow", lut=len(states))
ax_V, ax_C = plt.figure().subplots(1, 2)
legend = []
for i, state in enumerate(states):
    s = test_ac.getEnv().getIndexFromState(state, simulation=False)
    ax_V.plot(np.arange(1, n_learning_steps+1), dict_V[learning_method][rep, :n_learning_steps, s], color=colormap(i), marker='.')
    ax_C.plot(np.arange(1, n_learning_steps + 1), dict_state_counts[learning_method][rep, :n_learning_steps, s], color=colormap(i), marker='.')
    legend += [f"state {state}"]
ax_V.legend(legend)
ax_C.legend(legend)
ax_R = ax_V.twinx()
ax_R.plot(np.arange(1, n_learning_steps+1), dict_R_long[learning_method][rep, :n_learning_steps], color="green", marker='.')
legend_R = ["Average Reward"]
if len(states4diff) == 2:
    s = test_ac.getEnv().getIndexFromState(states4diff[0], simulation=False)
    ns = test_ac.getEnv().getIndexFromState(states4diff[1], simulation=False)
    ax_R.plot(np.arange(1, n_learning_steps+1), dict_V[learning_method][rep, :n_learning_steps, ns] -  dict_V[learning_method][rep, :n_learning_steps, s], color="black", marker='.')
    ax_R.axhline(0, color="gray")
    legend_R += [f"Delta(V): V({states4diff[1]}) - V({states4diff[0]})"]
ax_R.legend(legend_R)
ax_V.set_title("V(s) for selected states and avg(R)")
ax_C.set_title("Visit count for selected states")
plt.suptitle(f"{learning_method.upper()}")







#-- Final policy
_average_reward_at_last_policy_learning_step = dict_R[learning_method][nrep - 1][-1]
_state_counts_during_policy_learning = dict_state_counts[learning_method][nrep-1, :, :]
axes = plot_policy(dict_simulator[learning_method][rep].getEnv(), dict_simulator[learning_method][rep].getAgent().getPolicy(),
                   _average_reward_at_last_policy_learning_step,
                   _state_counts_during_policy_learning,
                   params_exec,
                   is_problem_2d=problem_2d, t_learn=n_learning_steps, fontsize=14, verbose=False)
show_elapsed_time(learning_method, time_elapsed, time_cpu)


plot_state_counts(dict_simulator, learning_method, rep, params_exec, seed=seed_learn)

# Let's look at the trajectories of the learner (it works when constructing the learner with store_history_over_all_episodes=True)
#print(len(dict_simulator[learning_method][rep].getAgent().getLearner().getStates()))
#print([len(trajectory) for trajectory in dict_simulator[learning_method][rep].getAgent().getLearner().getStates()])


# Distribution of number of steps (possibly over all replications)
plot_average_nsteps = True  #False
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
ax.set_ylabel(f"Average reward for replication #{rep2plot} (line plot)")
ax_n = ax.twinx()
if plot_average_nsteps:
    ax_n.bar(np.arange(1, n_learning_steps+1), np.nanmean(dict_nsteps[learning_method], axis=0), color=colors[rep2plot % len(colors)], alpha=0.5)
else:
    ax_n.bar(np.arange(1, n_learning_steps+1), dict_nsteps[learning_method][rep2plot, :], color=colors[rep2plot % len(colors)], alpha=0.5)
ax_n.set_ylabel(f"{'Average number' if plot_average_nsteps else 'Number'} of simulation steps (bar plot)")
plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion - {env_type.name} {env_shape}"
             f"\nN={N}, T={T}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
             f"\nContinuing and possibly episodic average reward"
             f"\nNumber of simulation steps by learning step ({nrep} replications)")



############## SAVE ALL RESULTS TOGETHER
if save:
    _env = test_ac.getEnv()
    _time_elapsed_min = np.sum([np.sum(t) for k, t in dict_time_elapsed.items()]) / 60
    _time_cpu_min = np.sum([np.sum(t) for k, t in dict_time_cpu.items()]) / 60
    wind_dict = None if "wind_dict" not in locals() else wind_dict
    exit_state = None if "exit_state" not in locals() else exit_state
    params_exec_orig = params_exec.copy()
    del params_exec['env_type']    # This entry is removed because of error in saving an Environment enum object to a pickle file
    objects_to_save = ["params_exec",
                       "_env", "env_type_name",
                       "wind_dict", "learning_task", "learning_criterion", "gamma", "exit_state", "nn_hidden_layer_sizes", "is_NPG", "policy_learning_mode",
                       "simulator_value_functions", "dict_simulator",
                       "dict_loss", "dict_R", "dict_R_long", "dict_R_long_true", "dict_R_long_initial", "dict_R_long_fv_inflated", "dict_V", "dict_Q", "dict_A",
                       "dict_state_counts", "dict_nsteps", "dict_KL", "dict_alpha", "dict_time_elapsed", "dict_time_cpu",
                       "max_time_steps_benchmark"]
    if "max_time_steps_benchmark_all" in locals():
        objects_to_save += ["max_time_steps_benchmark_all"]
    else:
        objects_to_save += ["max_time_steps_benchmark"]
    _shape_str = f"{env_shape[0]}x{env_shape[1]}" if "env_shape" in locals() else ""
    _filename = f"{prefix}{dt_start_filename}_ALL_exec={_time_elapsed_min:.0f}min_cpu={_time_cpu_min:.0f}min.pkl"
    _filepath = os.path.join(resultsdir, _filename)
    # Save the original object names plus those with the prefix so that, when loading the data back,
    # we have also the objects without the prefix which are need to generate the plots right-away.
    # HOWEVER, when reading the saved data, make sure that any object with the original name (without the suffix) has been copied to another object
    # (e.g. loss_all_td could be a copy of loss_all before reading the data previously saved for the FV learning method)
    object_names_to_save = objects_to_save + [f"{obj_name}" for obj_name in objects_to_save]
    save_objects_to_pickle(object_names_to_save, _filepath, locals(), lib="joblib")
    print(f"Results for ALL Actor-Critic methods: {[str.replace(meth, 'values_', '').upper() for meth in dict_loss.keys()]} saved to '{_filepath}'")
    # Restore the params_exec object
    params_exec = params_exec_orig.copy()
    del params_exec_orig
############## SAVE ALL RESULTS TOGETHER



############## LOAD RESULTS
# FIRST, need to load the necessary modules at the top of this Section and compile the auxiliary functions

#_datetime = "20240428_092427" #"20240428_205959" #"20240421_140558" #"20240405_094608" #"20240322_095848" #"20240219_230301" #"20240219_142206"          # Use format yyymmdd_hhmmss
#_shape_str = "6x8" #"21x1" #"10x14" #"3x4" #"10x14"
import os
from Python.lib.utils.basic import load_objects_from_pickle

resultsdir = os.path.realpath("./RL-003-Classic/results")
print(f"Results will be read from directory '{resultsdir}'")

_filename = "ActorCritic_gridworld_10x30_20250701_010741_ALL_exec=211min_cpu=225min.pkl"

# 2025/06/27: 10x30 labyrinth with NO wind
_filename = "ActorCritic_gridworld_10x30_20250627_131243_ALL_exec=551min_cpu=567min (R=5, FVAC(0) fails and TDAC(0) fails when adjusting LR by steps with KL gt 0).pkl"
_filename = "ActorCritic_gridworld_10x30_20250627_082700_ALL_exec=528min_cpu=496min (R=5, FVAC(lambda) learns faster than TDAC(lambda) when adjusting LR by steps with KL gt 0).pkl"

# 2025/06/23: Two sets of results to put together because they correspond to the same problem but different methods
# The procedure is:
# 1) Read the file with FEWER number of results first, make a copy of the dictionaries that are used for plotting, e.g.:
#   dict_nsteps_fv = dict_nsteps.copy(); dict_loss_fv = dict_loss.copy(); dict_R_fv = dict_R.copy(); dict_R_long_fv = dict_R_long.copy(); dict_R_long_true_fv = dict_R_long_true.copy()
# 2) Read the other file and merge the results from the file read first into the dictionaries that are used for plotting, e.g.:
#   dict_nsteps['values_fv'] = dict_nsteps_fv['values_fv']; dict_loss['values_fv'] = dict_loss_fv['values_fv']; dict_R['values_fv'] = dict_R_fv['values_fv']; dict_R_long['values_fv'] = dict_R_long_fv['values_fv']; dict_R_long_true['values_fv'] = dict_R_long_true_fv['values_fv']

# Gridworld 8x12 with WIND=0.3
_filename = "ActorCritic_gridworld_8x12_20250623_143039_ALL_exec=175min_cpu=182min (FAIR comparison with FVAC(lambda), TDAC(lambda)).pkl"
_filename = "ActorCritic_gridworld_8x12_20250623_122902_ALL_exec=728min_cpu=705min (UNFAIR comparison with FVAC(lambda), TDAC(lambda) due to maxT, backup start, A=Inf).pkl"

# 2025/06/24: Two sets of results to put together because they correspond to the same problem but different methods
# Gridworld 8x12 with WIND=0.5
_filename = "ActorCritic_gridworld_8x12_20250623_234207_ALL_exec=629min_cpu=635min (WIND=0.5, FVAC(lambda) works, TDAC(lambda) fails).pkl"
_filename = "ActorCritic_gridworld_8x12_20250623_234353_ALL_exec=346min_cpu=346min (FVAC learns but not as fast as FVAC(lambda)).pkl"



# 2024/10/15: Random labyrinth WITH NPG and without NPG
_filename = "ActorCritic_gridworld_6x8_20241015_195628_FV_WindyRandomLabyrinth0.5Seed4217,N=20,T=500,AlphaA=0.90,L=30N,iter=50.pkl"
_filename = "ActorCritic_gridworld_6x8_20241015_234507_ALL_WindyRandomLabyrinth0.5Seed4217,N=20,T=500,AlphaA=0.90,L=30N,iter=50.pkl"
_filename = "ActorCritic_gridworld_6x8_20241015_172914_ALL_WindyRandomLabyrinth0.5Seed4217,N=20,T=500,AlphaA=0.90,L=30N,iter=50,NPG.pkl"

#--- For EWRL-2024 paper
_filename = "ActorCritic_labyrinth_4x5_20240904_160121_ALL_Alphonse_N=20,T=500,alphaA=0.05,UNnormalizedLoss_UsedInEWRL2024paper (CAN READ BECAUSE I RE-RAN IT WITH VERSION 0a55bb8e, 25-May).pkl"
## NOTE: Because this file was generated with an earlier version of the code (21-May-2024, commit 0a55bb8ef7ecc4a4cbc12db23552b557a5679892), we need to create one variable referenced below as `_env = _env2d`.
#--- For EWRL-2024 paper

#--- For AAAI-2025 paper
# Random labyrinth 6x8 with wind 0.5
_filename = "ActorCritic_gridworld_6x8_20240809_083136_ALL_WindyRandomLabyrinth0.5Seed4217,N=20,T=500,AlphA=0.05 (FVAC is better and takes similar time!).pkl"
# Mountain Car
# (2024/10/23) I don't know if any of these were shown in any paper
#_filename = "ActorCritic_mountaincar_22x21_20240902_094357_ALL_Discrete_factor=100,force_factor=1,N=30,T=300,CumAlphaA=0.90,Ab=1.0,L=60N,rep=10,iter=50 (FVAC and TDAC are similar).pkl"
#_filename = "ActorCritic_mountaincar_22x21_20240815_180632_FV_Discrete_factor=100,force_factor=1,N=50,T=500,CumAlphaA=0.90,Ab=0.90,rep=10,iter=100 (FVAC learns EARLIER).pkl"
#--- For AAAI-2025 paper

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

_env_type_name = "gridworld" #"labyrinth"    # For older results (before implementing also learning in Mountain Car), use "labyrinth", for new results, use "gridworld"
_ndigits = 5 #4 #5  #3  # Number of digits used to define the dimension of the environment including the `x` (e.g. "4x5" => _ndigits = 3; "22x21" => _ndigits = 5)
_shape_str = _filename[len(f"ActorCritic_{_env_type_name}_"):len(f"ActorCritic_{_env_type_name}_") + _ndigits]
_N = 50 #int(_filename[_filename.index("N=") + len("N="):_filename.index(",", _filename.index("N="))])
_T = 500 #int(_filename[_filename.index("T=") + len("T="):_filename.index(",", _filename.index("T="))])
_filepath = os.path.join(resultsdir, _filename)
object_names = load_objects_from_pickle(_filepath, globals())
print(f"The following objects were loaded from '{_filepath}':\n{object_names}")

# The following variables are used in the "altogether" plots below
_methods = list(dict_loss.keys())
env_shape = (int(_shape_str[:_shape_str.index("x")]), int(_shape_str[_shape_str.index("x")+1:]))
if _filename.lower().find("random") >= 0:
    # Do NOT compute the max average rewards in RANDOM labyrinths, because we don't know how to compute them
    max_avg_reward_continuing = np.nan
    max_avg_reward_episodic = np.nan
else:
    max_avg_reward_continuing, max_avg_reward_episodic = compute_max_avg_rewards_in_labyrinth_with_corridor(_env, wind_dict, learning_task, learning_criterion)
seed_base = params_exec['seed_base']
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

# Show the execution times by method
for meth in dict_time_elapsed.keys():
    print(f"Execution times for meth={meth}: total = {np.sum(dict_time_elapsed[meth])/60:.1f} min (CPU: {np.sum(dict_time_cpu[meth])/60:.1f} min), "
          f"average = {np.mean(dict_time_elapsed[meth])/60:.1f} min (CPU: {np.nanmean(dict_time_cpu[meth])/60:.1f} min)")

# Show the number of steps by method
for meth in dict_nsteps.keys():
    print(f"# steps for meth={meth}: average across replications and learning steps = {np.mean(dict_nsteps[meth]):.1f} steps")

# We normalize the average reward plots so that they converge to 1.0 (easier interpretation of the plot)
if False and is_NPG or policy_learning_mode != "online":
    # DM-2025/06/25: I disabled this block because the long-run expected reward estimated by the learner
    # is NOT always a good indication of how well the actual process behaves on the current policy since,
    # for the FVAC method, the long-run average reward is influenced by the FV average reward,
    # which might be inflated or very different from the actual average reward under the original process...
    # e.g. when the policy update goes SNAFU, as observed already in larger labyrinths... (e.g. 10x30 with NO wind)
    dict_R_toplot = dict_R_long
    max_avg_reward = max_avg_reward_continuing
else:
    dict_R_toplot = dict_R
    max_avg_reward = max_avg_reward_episodic
# Check that the max avg. reward is defined, if not set it to 1.0 so that we plot the unnormalized observed average reward
max_avg_reward = 1.0 if np.isnan(max_avg_reward) or max_avg_reward == 0.0 else max_avg_reward

_exit_state_str = 'TOP' if "exit_state" in locals() and exit_state == env_shape[1] - 1 else 'BOTTOM' if "exit_state" in locals() else "(unknown)"
_value_functions_estimation = f"NN Input {nn_input_value_functions.name}, hidden: {nn_hidden_layer_sizes_value_functions}, Mode: {critic_learning_mode.name}" if use_function_approximation else "TABULAR"
_learning_characteristics = f"\nN={'N' in locals() and N or _N}, " + \
                            f"T={'T' in locals() and T or _T}, " + \
                            f"MAX budget={'max_time_steps_benchmark' in locals() and max_time_steps_benchmark or 'N/A'} steps\n" + \
                            f"NN hidden: {nn_hidden_layer_sizes}, " + \
                            f"Policy Learning MODE: {policy_learning_mode.upper()} - " + \
                            f"Value functions: {_value_functions_estimation}"

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
    ax_R.plot(np.arange(1, n_learning_steps + 1), dict_R_long_true[meth][nrep-1, :n_learning_steps] / (max_avg_reward_continuing if max_avg_reward_continuing != 0.0 else 1.0), '-', color=dict_colors[meth], linestyle="dashed")
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
ax_R.legend(legend, loc="center right") #loc="lower left")
plt.suptitle(f"ALL LEARNING METHODS: {env_type_name} {env_shape} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={gamma})" +
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

# --> FOR THESIS
# Plot all replications individually
plot_for_paper = True
fontsize = 26 if plot_for_paper else 12
ax = plt.figure(figsize=figsize).subplots(1, 1)
ax.axhline(0.0, color="gray", linestyle="dashed")
lines = []
legend = []
for rep in range(nrep):
    for meth in sorted(dict_R_toplot.keys()):
        if meth in ["values_tdl2"]: #["values_fv", "values_fv2", "values_td"]:
            continue
        line = ax.plot(np.arange(1, n_learning_steps+1), dict_R_toplot[meth][rep, :n_learning_steps], '-', color=dict_colors[meth], linewidth=0.3)
        if not plot_for_paper:
            ax.text(n_learning_steps, dict_R_toplot[meth][rep, -1], f"rep={rep+1} (seed={seed_base*(rep+1)})", color=dict_colors[meth])
        lines += line if rep == 0 else []
        legend += [f"{dict_legends[meth]}"] if rep == 0 else []
        #ax.axhline(0, color="gray")
        ax.set_xlabel("Learning step", fontsize=fontsize)
        ax.set_ylabel("Episodic average reward", fontsize=fontsize)
line = ax.axhline(max_avg_reward_episodic, color="darkgreen") if policy_learning_mode == "online" else None
line = ax.axhline(max_avg_reward_continuing, color="lightgreen") if policy_learning_mode == "offline" else None
lines += [line]
legend += ["Max. average reward" + (policy_learning_mode == "online" and " (episodic)" or " (continuing)")]
ax.legend(lines, legend, loc="center right", fontsize=int(0.5 * fontsize))
if plot_for_paper:
    ax.tick_params(axis='both', labelsize=int(0.8 * fontsize))
else:
    plt.suptitle(f"ALL LEARNING METHODS: {env_type_name} {env_shape} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={gamma})"
                 f"\nWIND: {wind_dict}, EXIT: {_exit_state_str}, ALL {nrep} replications" +
                 _learning_characteristics)
plt.savefig("spaghetti.png", dpi=300)

# --> FOR PAPER
# Plot results on several replications
plot_for_paper = True
marker_for_mean = "" if plot_for_paper else "."
marker_for_median = "." if plot_for_paper else "x"
fontsize = 26 if plot_for_paper else 12
loc_legend = "upper left"
#loc_legend = "center right"
if nrep > 1:
    plot_mean = plot_for_paper #False #True
    plot_median = True
    plot_bands = not plot_for_paper #True #False
    dict_stats_R = dict.fromkeys(dict_loss.keys())
    ax = plt.figure(figsize=figsize).subplots(1, 1)
    ax.axhline(0, color="gray", linestyle="dashed")
    lines = []
    legend = []
    _xshift = -0.1 # shift on the X axis to avoid overlap of vertical error bars
    for meth in dict_loss.keys():
        # Adapt the following filter if we want to exclude a particular method from the comparison plot
        if meth in ["values_tdl2"]: #["values_fv", "values_fv2", "values_td"]:
            continue
        _xshift += 0.1
        # Compute distribution of values to plot
        dict_stats_R[meth] = dict()
        dict_stats_R[meth]['min'], \
        dict_stats_R[meth]['max'], \
        dict_stats_R[meth]['median'], \
        dict_stats_R[meth]['mean'], \
        dict_stats_R[meth]['std'], \
        dict_stats_R[meth]['n'] = np.nanmin(dict_R_toplot[meth], axis=0), np.nanmax(dict_R_toplot[meth], axis=0), np.nanmedian(dict_R_toplot[meth], axis=0), np.nanmean(dict_R_toplot[meth], axis=0), np.nanstd(dict_R_toplot[meth], axis=0), \
                                  np.max(np.sum(~np.isnan(dict_R_toplot[meth]), axis=0)) #len(dict_R_toplot[meth])
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
            # MEAN plot +/- SE
            line = ax.plot(_xvalues, dict_stats_R[meth]['mean'][:n_learning_steps] / max_avg_reward, color=dict_colors[meth], linestyle=dict_linestyles[meth], linewidth=2, marker=marker_for_mean)[0]
            # Do NOT plot the error bars because it might be too dense for a plot...
            #line = ax.errorbar(_xvalues, dict_stats_R[meth]['mean'][:n_learning_steps] / max_avg_reward, yerr=dict_stats_R[meth]['std'][:n_learning_steps] / np.sqrt(dict_stats_R[meth]['n']) / max_avg_reward, color=dict_colors[meth], linestyle=dict_linestyles[meth], linewidth=2, marker=marker_for_mean, markersize=12)[0]
            ax.fill_between(_xvalues,
                            (dict_stats_R[meth]['mean'][:n_learning_steps] + dict_stats_R[meth]['std'][:n_learning_steps] / np.sqrt(dict_stats_R[meth]['n'])) / max_avg_reward,
                            (dict_stats_R[meth]['mean'][:n_learning_steps] - dict_stats_R[meth]['std'][:n_learning_steps] / np.sqrt(dict_stats_R[meth]['n'])) / max_avg_reward,
                            color=dict_colors[meth],
                            alpha=0.5)
            lines += [line]
            legend += [f"{dict_legends[meth]} (average +/- SE, {np.nanmean(dict_nsteps[meth]):.0f} avg.#steps)"]
        if plot_median:
            # MEDIAN plot
            line = ax.plot(_xvalues, dict_stats_R[meth]['median'][:n_learning_steps] / max_avg_reward, color=dict_colors[meth], linestyle="dashed" if plot_mean else "solid", linewidth=2, marker=marker_for_median, markersize=12)[0]
            lines += [line]
            legend += [f"{dict_legends[meth]} (median), {np.nanmean(dict_nsteps[meth]):.0f} avg.#steps)"]
        if plot_bands:
            # MIN/MAX plot
            line = ax.plot(_xvalues, dict_stats_R[meth]['max'][:n_learning_steps] / max_avg_reward, color=dict_colors[meth], linestyle="dashed")[0]
            lines += [line]
            legend += [f"{dict_legends[meth]} (min/max)"]
            ax.plot(_xvalues, dict_stats_R[meth]['min'][:n_learning_steps] / max_avg_reward, color=dict_colors[meth], linestyle="dashed")
            ax.fill_between(_xvalues,
                            dict_stats_R[meth]['max'][:n_learning_steps] / max_avg_reward,
                            dict_stats_R[meth]['min'][:n_learning_steps] / max_avg_reward,
                            color=dict_colors[meth],
                            alpha=0.1)
    ax.legend(lines, legend, fontsize=int(0.5 * fontsize), loc=loc_legend)
    if max_avg_reward != 1.0:
        # This is the case when the max average reward is known, so we show the horizontal line corresponding to the maximum achievable NORMALIZED average reward which is equal to 1
        ax.axhline(1, color="gray")
    ax.set_ylim((-ax.get_ylim()[1]/100, None))
    ax.set_xlabel("Learning step", fontsize=fontsize)
    if plot_for_paper:
        # Shorter label, larger tick labels, no title
        _ylabel = "Normalized episodic average reward" if max_avg_reward != 1.0 else "Episodic average reward"
        ax.set_ylabel(_ylabel, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=int(0.8*fontsize))
    else:
        ax.set_ylabel(f"Average reward {'(normalized by the MAX average reward = {:.2g})' if max_avg_reward != 1.0 else ''}".format(max_avg_reward), fontsize=fontsize)
        plt.suptitle(f"ALL LEARNING METHODS: {env_type_name} {env_shape} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={gamma})"
                     f"\nWIND: {wind_dict}, EXIT: {_exit_state_str}, {nrep} replications" +
                     _learning_characteristics)

    # Plot of number of samples ratios between FV learnings and TD learning
    legend_nsamples = []
    if "values_td" in dict_nsteps.keys() and "values_fv" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.nanmean(dict_nsteps['values_td'], axis=0)[:n_learning_steps],
                                          'fv': np.nanmean(dict_nsteps['values_fv'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'], color="blue", linewidth=0.5)
        ref_line = ax_nsamples.axhline(1.0, color="blue", linewidth=0.5, linestyle="dashed")
        legend_nsamples += ["Sample size ratio (FV/TD)", "Reference line showing equal sample size ratio"]
    if "values_tdl" in dict_nsteps.keys() and "values_fvl" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.nanmean(dict_nsteps['values_tdl'], axis=0)[:n_learning_steps],
                                          'fv': np.nanmean(dict_nsteps['values_fvl'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'], color="blue", linewidth=0.5)
        ref_line = ax_nsamples.axhline(1.0, color="blue", linewidth=0.5, linestyle="dashed")
        legend_nsamples += ["Sample size ratio (FVL/TDL)", "Reference line showing equal sample size ratio"]
    if "values_td" in dict_nsteps.keys() and "values_fv2" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.nanmean(dict_nsteps['values_td'], axis=0)[:n_learning_steps],
                                          'fv': np.nanmean(dict_nsteps['values_fv2'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        if "ax_nsamples" not in locals():
            ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'], color="orange", linewidth=0.5)
        ref_line = ax_nsamples.axhline(1.0, color="orange", linewidth=0.5, linestyle="dashed")
        legend_nsamples += ["Sample size ratio (FV2/TD)", "Reference line showing equal sample size ratio"]
    if "values_td2" in dict_nsteps.keys() and "values_fv" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.nanmean(dict_nsteps['values_td2'], axis=0)[:n_learning_steps],
                                          'fv': np.nanmean(dict_nsteps['values_fv'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        if "ax_nsamples" not in locals():
            ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'], color="cyan", linewidth=0.5)
        legend_nsamples += ["Sample size ratio (FV/TD2)"]
        if "ref_line" not in locals():
            ref_line = ax_nsamples.axhline(1.0, color="cyan", linewidth=0.5, linestyle="dashed")
            legend_nsamples += ["Reference line showing equal sample size ratio"]
    if "values_td2" in dict_nsteps.keys() and "values_fv2" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.nanmean(dict_nsteps['values_td2'], axis=0)[:n_learning_steps],
                                          'fv': np.nanmean(dict_nsteps['values_fv2'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        if "ax_nsamples" not in locals():
            ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps + 1), df_ratio_nsamples['ratio_fv_td'], color="magenta", linewidth=0.5)
        legend_nsamples += ["Sample size ratio (FV2/TD2)"]
        if "ref_line" not in locals():
            ax_nsamples.axhline(1.0 - 1E-6, color="magenta", linewidth=0.5, linestyle="dashed")
            legend_nsamples += ["Reference line showing equal sample size ratio"]
        ax_nsamples.set_ylim((ax.get_ylim()[0], None))
    if "values_td2" in dict_nsteps.keys() and "values_fv3" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.nanmean(dict_nsteps['values_td2'], axis=0)[:n_learning_steps],
                                          'fv': np.nanmean(dict_nsteps['values_fv3'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        if "ax_nsamples" not in locals():
            ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps + 1), df_ratio_nsamples['ratio_fv_td'], color="magenta", linestyle="dashed", linewidth=0.5)
        legend_nsamples += ["Sample size ratio (FV3/TD2)"]
        if "ref_line" not in locals():
            ax_nsamples.axhline(1.0 - 1E-6, color="magenta", linewidth=0.5, linestyle="dotted")
            legend_nsamples += ["Reference line showing equal sample size ratio"]
        ax_nsamples.set_ylim((ax.get_ylim()[0], None))
    if "values_fv2" in dict_nsteps.keys() and "values_fv" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'fv2': np.nanmean(dict_nsteps['values_fv2'], axis=0)[:n_learning_steps],
                                          'fv': np.nanmean(dict_nsteps['values_fv'], axis=0)[:n_learning_steps]})
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
        if plot_for_paper:
            ax_nsamples.set_ylabel("Budget ratio (FVAC / TDAC)", fontsize=int(0.8*fontsize), color="blue")
            ax_nsamples.tick_params(axis='y', labelsize=int(0.8*fontsize), labelcolor="blue")
        else:
            ax_nsamples.set_ylabel("Average sample Ratio FV/TD across replications", fontsize=int(0.8 * fontsize))
            ax_nsamples.legend(legend_nsamples, loc="center left") #"lower right")
#-- ALTOGETHER PLOT
plt.savefig("altogether.png", dpi=300)
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
