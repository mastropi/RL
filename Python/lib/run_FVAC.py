# -*- coding: utf-8 -*-
"""
Created on Sun Jun 07 23:48:41 2024

@author: Daniel Mastropietro
@description: Runs the FVAC algorithm (Fleming-Viot Actor-Critic) to learn optimal parameterized policies in discrete-time Markov decision processes.
              Execution is possible either in batch mode (i.e. passing parameters from the prompt) or interactive mode (e.g. from PyCharm), in which case
              the parameters should be defined as default values in the code below, when invoking parser.set_defaults().
*** IN PREPARATION ***
*** Also, we need to take here the latest changes in the tests.py file under section "Tests on FVAC", which I have continued using for testing.
"""

#if __name__ == "__main__":
#    # Only run this when running the script, o.w. it may give an error when importing functions if setup.py is not found
#    import runpy
#    runpy.run_path('../../setup.py')


#-------------------- IMPORT AND AUXILIARY FUNCTIONS ------------------#
from timeit import default_timer as timer
import os
import sys
from time import process_time
import time
import joblib

from enum import Enum, unique
import optparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import MaxNLocator
from scipy.special import rel_entr

from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners import LearningCriterion, LearningTask, LearningMode
from Python.lib.agents.learners.policies import LeaActorCriticNN
from Python.lib.agents.policies import probabilistic

from Python.lib.environments import gridworlds
from Python.lib.estimators.nn_models import InputLayer

from Python.lib.simulators.fv import StoppingCriterion

from Python.lib.utils.basic import get_current_datetime_as_string, load_objects_from_pickle, log_file_open, log_file_close, save_objects_to_pickle, set_numpy_options, reset_numpy_options
from Python.lib.utils.computing import compute_expected_reward, compute_state_value_function_from_environment_and_policy

from Python.test.test_optimizers_discretetime import Test_EstPolicy_EnvGridworldsWithObstacles
from Python.test.test_utils import Environment, KL_THRESHOLD, compute_normalized_KL_distance, compute_max_avg_rewards_in_labyrinth_with_corridor, compute_true_state_value_function,\
    define_plotting_parameters, plot_policy, plot_state_counts, plot_trajectory, show_elapsed_time


#------------------- Functions to parse input arguments ---------------------#
def parse_input_parameters(argv):
    # Parse input parameters
    # Ref: https://docs.python.org/3.7/library/optparse.html
    # Main steps:
    # 1) The option parser is initialized with optparse.OptionParser(), where we can specify the usage= and version=,
    # as e.g. `optparse.OptionParser(usage="%prog [-v] [-p]", version="%prog 1.0")`
    # 2) New options to parse are added with parser.add_option(), where the metavar= argument (e.g. `metavar="input file"`)
    # is used to indicate that the option expects a value and gives a short description of its content
    # (e.g. `--filename="file.txt"` as opposed to `--verbose`, which expects no value).
    # We can also define:
    #    a) the default value of the option (although this is more clearly done with parser.set_defaults().
    #    b) the action to take with the option value read with the action= argument, e.g. "store_true", "store_false",
    #       which are actually needed for FLAG options that do NOT require any option value (e.g. -v for verbose, etc.),
    #       and ***whose default value (i.e. when the flag is not given) is specified by the default= parameter***.
    #       The default action is "store" which is used for options accepting a value as in `--file="file.txt".
    #       --> NOTE that the action can be "callback" meaning that a callback function with the signature callback(option, opt, value, parser)
    #       is called to parse the argument value.
    #       If the value of the argument needs to be processed by the callback (most likely) we need to:
    #       - specify its type via the `type=` option of the parser.add_option() function. Otherwise, the argument value will be set to None.
    #       If the value of the argument needs to be updated (e.g. a string converted to a list) we need to:
    #       - define the name of the argument to set with the `dest=` option of the parser.add_option() method.
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
    parser = optparse.OptionParser(usage="%prog "
                                         # System definition
                                         "[--env] "
                                         # Problem definition
                                         "[--learning_task] "
                                         "[--learning_criterion] "
                                         "[--discount] "
                                         # Critic learning
                                         "[--learning_method] "
                                         "[-N] "
                                         "[-T] "
                                         "[--epsilon_random_action] "
                                         "[--benchmark_filename] "
                                         "[--benchmark_datetime] "
                                         # Actor learning
                                         "[--n_learning_steps] "
                                         "[--policy_learning_mode] "
                                         "[--factor_max_time_steps_per_policy_learning_episode] "
                                         "[--allow_deterministic_policy] "
                                         "[--policy_learning_rate] "
                                         # Execution parameters
                                         "[--replications] "
                                         "[--seed] "
                                         "[--create_log] "
                                         "[--save_results] "
                                         "[--save_with_dt] "
                                         "[--plot] "
                                         )
    # TODO: Find out how to convert `str` to the corresponding class, e.g. discrete.Discrete, LearningTask, etc.
    # System parameters
    parser.add_option("--environment",
                      type="str",
                      metavar="Environment type",
                      help="Type of environment where learning takes place [default: %default]")
    # Problem parameters
    parser.add_option("--learning_task",
                      type="str",
                      metavar="Learning task ",
                      help="Learning task (e.g. CONTINUING, EPISODIC) [default: %default]")
    parser.add_option("--learning_criterion",
                      type="str",
                      metavar="Learning criterion",
                      help="Learning criterion (e.g. AVERAGE, DISCOUNTED) [default: %default]")
    parser.add_option("--discount",
                      type="float",
                      metavar="Dicsount factor gamma",
                      help="Discount factor (used both in the AVERAGE and in the DISCOUNTED learning criterion) [default: %default]")
    # Critic learning parameters
    parser.add_option("--learning_method",
                      type="str",
                      metavar="Critic Learning method",
                      help="Learning method for the critic [default: %default]")
    parser.add_option("-N",
                      type="int",
                      metavar="# particles",
                      help="Number of Fleming-Viot particles [default: %default]")
    parser.add_option("-T",
                      type="int",
                      metavar="# steps for E(T_A) estimation",
                      help="Number of steps used for E(T_A) estimation [default: %default]")
    parser.add_option("--epsilon_random_action",
                      type="float",
                      metavar="Probability of random action",
                      help="Probability for random action when learning the critic [default: %default]")
    parser.add_option("--benchmark_filename",
                      type="str",
                      metavar="Benchmark file",
                      help="File from where the value of execution parameters (e.g. theta_start, T, etc.) that need to be defined in order to perform a fair comparison between the MC and the FV methods [default: %default]")
    parser.add_option("--benchmark_datetime",
                      type="str",
                      metavar="Benchmark datetime",
                      help="Datetime in format <yymmdd>_<hhmmss> that appears in the name of the file to use as benchmark from where the value of execution parameters is read [default: %default]")
    # Actor learning parameters
    parser.add_option("--n_learning_steps",
                      type="int",
                      metavar="# Policy learning steps",
                      help="Number of policy learning steps to run [default: %default]")
    parser.add_option("--policy_learning_mode",
                      type="str",
                      metavar="Policy learning mode",
                      help="Policy learning mode (e.g. ONLINE, OFFLINE) [default: %default]")
    parser.add_option("--factor_max_time_steps_per_policy_learning_episode",
                      type="float",
                      metavar="Factor for max # steps per episode",
                      help="Factor multiplying the number of environment states defining the max #steps to run per episode when computing the loss [default: %default]")
    parser.add_option("--allow_deterministic_policy",
                      action="store_true",
                      help="Whether to allow deterministic policies [default: %default]")
    parser.add_option("--policy_learning_rate",
                      type="float",
                      metavar="Policy learning rate",
                      help="Initial learning rate for optimizer [default: %default]")
    # Execution parameters
    parser.add_option("--replications",
                      type="int",
                      metavar="# Replications",
                      help="Number of replications to run [default: %default]")
    parser.add_option("--seed",
                      type="int",
                      metavar="Seed",
                      help="Base seed value to use for the simulations (each replication has a different seed)[default: %default]")
    parser.add_option("--create_log",
                      action="store_true",
                      help="Whether to create a log file [default: %default]")
    parser.add_option("--save_results",
                      action="store_true",
                      help="Whether to save the results into a Pickle file [default: %default]")
    parser.add_option("--save_with_dt",
                      action="store_true",
                      help="Whether to use the execution datetime as suffix of output file names [default: %default]")
    parser.add_option("--plot",
                      action="store_true",
                      help="Whether to plot the policy learning process (e.g. Learning curve of the long-run expected reward) [default: %default]")
    if False:
        parser.add_option("-d", "--debug", dest="debug", default=False,
                          action="store_true",
                          help="debug mode")
        parser.add_option("-v", "--verbose", dest="verbose", default=False,
                          action="store_true",
                          help="verbose: show relevant messages in the log")

    # Default values so that the process can be run in interactive mode
    default_queue_system = "loss-network" #"single-server" #"loss-network"
    default_create_output_files = True
    parser.set_defaults(# System parameters
                        env=gridworlds.EnvGridworld2D,
                        # Problem parameters
                        learning_task=LearningTask.CONTINUING,
                        learning_criterion=LearningCriterion.AVERAGE,
                        discount=1.0,
                        # Critic parameters
                        learning_method="values_fv",
                        N=20,
                        T=500,
                        epsilon_random_action=0.1,
                        benchmark_filename="benchmark_fv.pkl", # Not used when benchmark_datetime is given (i.e. not empty or None)
                        benchmark_datetime=None, #"20230410_102003", #"20230409_163723",  # Format: "<yymmdd>_<hhmmss>". Use this parameter ONLY when method = "MC" and we want to automatically generate the benchmark filename to read the benchmark data from
                        # Actor parameters
                        n_learning_steps=100,
                        # Execution parameters
                        replications=10,
                        seed=1317,
                        create_log=default_create_output_files,
                        save_results=default_create_output_files,
                        save_with_dt=True,
                        plot=True)

    (options, args) = parser.parse_args(argv)

    print("Parsed command line options: " + repr(options))

    # options: `Values` object whose parameters are referred with the dot notation (e.g. options.x)
    # args: argument values (which do not require an argument name
    return options, args


def generate_parameter_string_for_filename(env,
                                           N,
                                           T):
    if env.__class__.__name__ == "EnvGridworld2D":
        size_vertical, size_horizontal = env.getShape()
        size_str = f"{size_vertical}x{size_horizontal}"
        params_str = size_str + "-"
    params_str += learning_method.name + \
                 "_N={},T={}".format(N, T)

    return params_str


def show_execution_parameters(options):
    print("\nSystem characteristics:")
    print("Environment type: {}".format(options.env.__class__.__name__))
    if issubclass(options.env, gridworlds.EnvGridworld1D):
        print("1D Environment length: {}".format(options.env.getNumStates()))
    elif issubclass(options.env, gridworlds.EnvGridworld2D):
        print("2D Environment shape: {}".format(options.env.getShape()))
    print("")
    print("Problem characteristics:")
    print("Learning task: {}".format(options.learning_task.name))
    print("Learning criterion: {}".format(options.learning_criterion.name))
    print("Discount (gamma): {}".format(options.discount))
    print("")
    print("Execution parameters:")
    print("create log? {}".format(options.create_log))
    print("save results? {}".format(options.save_results))
    print("seed = {}".format(options.seed))
    print("# replications = {}".format(options.replications))
    print("")
    print("CRITIC parameters:")
    print("learning_method = {}".format(options.learning_method.name))
    learning_method_type = options.learning_method.name[:9]
    if learning_method_type == "values_fv":
        print("# particles, N = {}".format(options.N))
        print("# time steps for E(T_A), T = {}".format(options.T))
    else:
        print("Benchmark learner uses average # steps across policy learning steps? {}".format(options.use_average_max_time_steps_in_benchmark))
    print("Epsilon random action = {}".format(options.epsilon_random_action))
    print("")
    print("ACTOR parameters:")
    print("# learning steps = {}".format(options.n_learning_steps))
    print("policy learning mode = {}".format(options.policy_learning_mode))
    print("factor max time steps per policy learning step = {}".format(options.factor_max_time_steps_per_policy_learning_episode))
    print("allow deterministic policy? {}".format(options.allow_deterministic_policy))
    print("learning rate = {}".format(options.policy_learning_rate))

    # DM-2024/06/09: STILL some parameters to consider adding...?
    # Parameters about policy learning (Actor)
    #n_episodes_per_learning_step = 50  # 100 #30   # This parameter is used as the number of episodes to run the policy learning process for and, if the learning task is EPISODIC, also as the number of episodes to run the simulators that estimate the value functions
    #max_time_steps_per_policy_learning_episode = 5 * test_ac.env2d.getNumStates() if problem_2d else 2 * test_ac.env2d.getNumStates()  # np.prod(env_shape) * 10 #max_time_steps_benchmark // n_episodes_per_learning_step   # Maximum number of steps per episode while LEARNING THE *POLICY* ONLINE (NOT used for the value functions (critic) learning)

    # Parameters about value function learning (Critic)
    #learning_steps_observe = [50, 90]  # [2, 30, 48] #[2, 10, 11, 30, 31, 49, 50] #[7, 20, 30, 40]  # base at 1, regardless of the base value used for t_learn
    #verbose_period = max_time_steps_fv_for_all_particles // 10
    #plot = False  # Whether to plot the evolution of value function and average reward estimation
    #colormap = "seismic"  # "Reds"  # Colormap to use in the plot of the estimated state value function V(s)
#------------------- Functions to parse input arguments ---------------------#


if __name__ == "__main__":
    #------ Parse input arguments
    if False:
        options, args = parse_input_parameters(sys.argv[1:])

        print("Parsed user arguments:")
        print(f"Options: {options}")
        print(f"Arguments: {args}")
        print("")

    # Define directories (assumed the current directory is the location of this file --which is true when the file is run as a script)
    if os.getcwd()[-3:] == "lib":
        rootdir = os.path.realpath("../../RL-003-Classic")
    else:
        rootdir = os.path.realpath("./RL-003-Classic")
    resultsdir = f"{rootdir}/testbed/results"
    logsdir = f"{rootdir}/testbed/logs"

    # Learning characteristics for the computation of the state value function
    learning_task = LearningTask.CONTINUING
    learning_criterion = LearningCriterion.AVERAGE

    #--- Consider different gridworlds of increasing size so that we get smaller and smaller average rewards
    # WARNING: Because of the obstacles, some Markov chains obtained may NOT be unichain...
    # (i.e. the Markov chain is REDUCIBLE, meaning that the P matrix has UNCONNECTED blocks)
    # In that case, the eigenvector that represents the stationary distribution will be the stationary distribution associated to ONE of the blocks.
    # If such block does NOT include the Finish cell, the average reward will be 0!
    # We solve this by brute force, i.e. when the average reward is 0.0, we consider a different seed_obstacles until a non-zero average reward is obtained.
    plot = False
    env_type = Environment.Gridworld
    env_type_name = env_type.name
    prop_obstacles = 0.3 #0.4
    seed_obstacles = 4217
    if env_type == Environment.Gridworld:
        # WIND
        wind_dict = None
        #wind_dict = dict({'direction': gridworlds.Direction2D.LEFT, 'intensity': 0.1})
        #wind_dict = dict({'direction': gridworlds.Direction2D.LEFT, 'intensity': 0.3})    # 22-Jun-2025: Used in 8x12 gridworld
        wind_dict = dict({'direction': gridworlds.Direction2D.LEFT, 'intensity': 0.5})
        #wind_dict = dict({'direction': gridworlds.Direction2D.LEFT, 'intensity': 0.6})
        #wind_dict = dict({'direction': gridworlds.Direction2D.LEFT, 'intensity': 0.7})
        #wind_dict = dict({'direction': gridworlds.Direction2D.LEFT, 'intensity': 0.8})

        # We consider SQUARE labyrinths
        sizes = np.arange(4, 24)
        average_rewards = np.nan*np.ones_like(sizes)
        seeds = np.nan*np.ones_like(sizes)
        for i, size in enumerate(sizes):
            # Environment shape
            env_shape = (size, size)
            nS = np.prod(env_shape)

            # Number of obstacles
            n_obstacles = int(prop_obstacles * np.prod(env_shape))

            print(f"\nCreating environment of size {env_shape} with {prop_obstacles*100:.1f}% of obstacles (n_obstacles={n_obstacles}) using seed_obstacles={seed_obstacles}...")

            # Entry and exit states
            exit_state_at_bottom = False
            entry_state = np.ravel_multi_index((size - 1, 0), env_shape)
            exit_state = entry_state + env_shape[1] - 1 if exit_state_at_bottom else env_shape[1] - 1
            terminal_states = {exit_state}

            # Define the initial state distribution that will be used when defining the environment class
            isd = np.zeros(nS)
            isd[entry_state] = 1.0

            # Create the environment
            reward_terminal = +1.0
            reward_obstacles = 0.0
            dict_rewards = dict([(s, reward_terminal if s in terminal_states else reward_obstacles) for s in set.union(set(terminal_states), set())])

            done = False
            num_trials = 0
            while not done:
                num_trials += 1
                this_seed_obstacles = seed_obstacles + num_trials - 1
                env2d = gridworlds.EnvGridworld2D_Random( shape=env_shape,
                                                          n_obstacles=n_obstacles,
                                                          terminal_states=terminal_states,
                                                          rewards_dict=dict_rewards,
                                                          wind_dict=wind_dict,
                                                          initial_state_distribution=isd,
                                                          seed=this_seed_obstacles)
                print(f"Gridworld environment of size {env_shape}, seed = {this_seed_obstacles}")

                # Initial RANDOM policy on which the average reward is computed
                policy_random = probabilistic.PolGenericDiscrete(env2d, policy=dict(), policy_default=[0.25, 0.25, 0.25, 0.25])

                # Compute true V(s) and average reward
                V_true, average_rewards[i], mu = compute_true_state_value_function(env2d, policy_random, learning_task, learning_criterion, atol=1E-6)
                done = average_rewards[i] != 0.0
            seeds[i] = this_seed_obstacles

            print(f"Under the RANDOM policy for environment of size ({size, size}) with seed={this_seed_obstacles}:")
            print(f"Average number of steps to get from START to FINISH: {1/average_rewards[i]:.1f}")
            print(f"Average reward: {average_rewards[i]}")

            if plot:
                # Plot the labyrinth
                #env2d._render()
                ax_labyrinth = env2d.plot()
                ax_labyrinth.set_title(f"Environment of size ({size, size}), seed = {this_seed_obstacles}")
    #-------------------------------- ENVIRONMENT -------------------------#


    # Reference probabilities that we want to analyze / test
    n_proba_refs = 10
    proba_refs = np.logspace(-1, -n_proba_refs, n_proba_refs)

    plt.figure()
    plt.plot(sizes, average_rewards, 'b.-')
    plt.title(f"Average reward as a function of labyrinth size")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().set_yscale('log')
    for s in sizes:
        plt.axvline(s, color="gray", linewidth=1)
    for p in proba_refs:
        plt.axhline(p, color="blue", linestyle="dashed")

    # Get the labyrinth sizes to use as test benches, based on their average reward under the random policy
    df_sizes = pd.DataFrame({'size': sizes, 'p': average_rewards, 'seed': seeds}, columns=['size', 'p', 'seed'])
    df_sizes.sort_values('p', inplace=True)
    df_sizes.reset_index(drop=True, inplace=True)
    indices = np.zeros_like(proba_refs, dtype=int)
    for i, p in enumerate(proba_refs):
        indices[i] = min(np.searchsorted(df_sizes['p'], p), len(df_sizes) - 1)
        print(f"p = {p} => index = {indices[i]} => size = {df_sizes.iloc[indices[i]]['size']} (actual p = {df_sizes.iloc[indices[i]]['p']:.3g})")
    df_sizes_testbed = df_sizes.iloc[indices].drop_duplicates(['size'])
    print(f"The following labyrinth sizes will be tested:\n{df_sizes_testbed}")
    plt.plot(df_sizes_testbed['size'], df_sizes_testbed['p'], 'r.')

    # HOWEVER, I don't like the choices done automatically, so I chose the sizes manually, based on the plot generated above
    #sizes_testbed = [4, 9, 15, 17, 19, 21]  # NO wind case
    sizes_testbed = [4, 8, 11, 15]           # WIND=0.5 case
    df_sizes_testbed = df_sizes[df_sizes['size'].isin(sizes_testbed)]
    df_sizes_testbed.sort_values(['size'], inplace=True)
    df_sizes_testbed.reset_index(drop=True, inplace=True)
    # Define the MAX simulation time for the INI exploration, T
    df_sizes_testbed['T'] = 500
    # Number of policy learning steps to use per case
    df_sizes_testbed['n_learning_steps'] = np.maximum(30, 30 * (np.abs(np.log10(df_sizes_testbed.p)) - 1)).astype(int) // 2
    # Make the number of learning steps a multiple of 10
    df_sizes_testbed['n_learning_steps'] = df_sizes_testbed['n_learning_steps'] // 10 * 10


    #----------------------------- MODEL FOR POLICY -----------------------#
    # Number of hidden layers in the neural network model
    nn_hidden_layer_sizes = nn_hidden_layer_sizes_policy = []  # [12]
    # Number of input neurons (just one with the state value or one-per-state)
    nn_input_policy = InputLayer.ONEHOT if len(nn_hidden_layer_sizes_policy) == 0 else InputLayer.STATE
    print(f"Neural Network architecture:")
    print(f"Input layer: {nn_input_policy}")
    print(f"Hidden layers: {len(nn_hidden_layer_sizes_policy)} hidden layers of sizes {nn_hidden_layer_sizes_policy}")
    initial_policy = None   # This means the initial policy is RANDOM
    #----------------------------- MODEL FOR POLICY -----------------------#


    #----------------------------- MODEL FOR CRITIC -----------------------#
    use_function_approximation = False
    use_separate_target_model = True if use_function_approximation else False; update_period_target_model = 100
    nn_input_value_functions = InputLayer.STATE  # InputLayer.ONEHOT
    nn_hidden_layer_sizes_value_functions = [12]  # [48]
    # Learning rate for value functions
    alpha_initial = 1.0
    gamma = 1.0
    #----------------------------- MODEL FOR CRITIC -----------------------#


    #----------------- COMMON EXECUTION PARAMETERS FOR ALL SIZES ----------#
    # Absorption strategy
    estimate_absorption_set_at_every_step = True
    update_absorption_set_with_fv_visits = True
    soft_killing = False  #True

    #-- Common learning parameters (to all methods)
    # 1) Parameters about POLICY learning (Actor)
    policy_learning_mode = "online"  # "offline" #"online"
    is_NPG = len(nn_hidden_layer_sizes_policy) == 0
    prob_include_in_train = 1.0  # Probability of including a step of the exploration, used for the ONLINE policy learning update, in the sample that computes the loss. Goal: reduce the correlation among samples included in the training process.
    n_episodes_per_learning_step = int(50 / prob_include_in_train)  # 100 #30  # Number of episodes for the policy update step when learning the policy online and in NON-NPG mode
    # Max time steps per episode during exploration for the online policy learning
    # In the Mountain Car problem we limit the number of steps per episode in the continuous-dynamics case because I've seen out-of-memory problems otherwise.
    max_time_steps_per_policy_learning_episode = 50  # Maximum number of steps per episode while LEARNING THE *POLICY* ONLINE (NOT used for the value functions (critic) learning)
    allow_deterministic_policy = True  # False
    adjust_policy_learning_rate = False; t_learn_min_to_adjust_policy_learning_rate = 1
    reset_value_functions_at_every_learning_step = False  # Reset the value functions when learning with FV, o.w. the learning can become too unstable due to the oversampling of the states with high value... (or something like that)

    # 2) Parameters about VALUE FUNCTION learning (Critic)
    critic_learning_mode = LearningMode.ONLINE #LearningMode.BATCH #LearningMode.ONLINE
    adjust_alpha_initial_by_learning_step = False; t_learn_min_to_adjust_alpha = 10  #30 # based at 1 (regardless of the base value used for t_learn)
    epsilon_random_action = 0.1  # if policy_learning_mode == "online" else 0.0 #0.1 #0.05 #0.0 #0.01
    reward_to_promote_exploration = 0.0
    use_average_max_time_steps_in_td_learner = True
    use_average_reward_from_previous_step = True
    use_fixed_average_reward = False
    keep_fv_estimation_of_average_reward_and_stationary_probability_consistent = False  #True
    start_at_less_frequency_visited_states = True #False      # Whether to start the TD learning at a randomly selected state among those less visited by the exploration at the previous policy learning step
    plot = False
    plot_policy_update = False
    colormap = "seismic"

    # A few further parameters for the policy learning process
    break_when_no_change = False
    break_when_goal_reached = False
    #----------------- Execution parameters

    # Number of replications to run on each method
    nrep = 5 #9
    # Log file
    log = nrep > 1
    # Save results (pkl)
    save = nrep > 1

    # Results for all labyrinths tested
    dict_results = dict()

    learning_methods = ["values_tdl2"] #["values_fvl", "values_tdl"]
    learning_method_benchmark = None #"values_fvl"  #None  # Method already run previously from where the benchmark number of steps needs to be taken
    #----------------- COMMON EXECUTION PARAMETERS FOR ALL SIZES ----------#


    #--- Variables and functions required by the script inclded below with runfile()
    problem_2d = True
    figsize = (10, 8)
    learning_steps_observe = []
    max_avg_reward_continuing = np.nan
    max_avg_reward_episodic = np.nan
    #--- Variables and functions required by the script inclded below with runfile()


    # Seed used for:
    # (i) the policy seed at creation (torch.manual_seed(seed))
    # (ii) the eventual reset seed for value functions in learners (reset_seed=seed) at their creation
    seed = 1317
    # Number of particles
    N = 50

    # Keep track of time
    time_start_all = timer()
    cpu_start_all = process_time()
    dt_start_all = get_current_datetime_as_string(format="filename")

    sizes2run = [4] #[4, 8, 11, 15] #[21] #[19] #[17] #[15]  # WARNING: I got a MemoryError with size = 35
    for row in df_sizes_testbed.iterrows():
        env_case = row[0]
        size = int(row[1]['size'])
        if size not in sizes2run:
            continue
        env_shape = (size, size)

        # Entry and exit states
        entry_state = np.ravel_multi_index((size - 1, 0), env_shape)
        exit_state = entry_state + env_shape[1] - 1 if exit_state_at_bottom else env_shape[1] - 1

        # Obstacles
        n_obstacles = int(prop_obstacles * np.prod(env_shape))
        seed_obstacles = int(row[1]['seed'])

        #-------------------------------- TEST SETUP --------------------------#
        # FV learning parameters (which are used to define parameters of the other learners analyzed so that their comparison with FV is fair)
        T = int(row[1]['T'])
        threshold_absorption_set = 0.90
        # Max average number of steps allowed for each particle in the FV simulation
        max_time_steps_fv_per_particle = 30
        max_time_steps_fv_for_expectation = T
        # *********************
        stopping_criterion_fv = StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES_AS_LONG_AS_ENOUGH_TIME_STEPS_HAVE_BEEN_TAKEN  # StoppingCriterion.MAX_TIME_STEPS_AND_MIN_PROP_ABSORBED_PARTICLES #StoppingCriterion.MAX_TIME_STEPS #StoppingCriterion.MAX_TIME_STEPS_OR_MIN_PROP_ABSORBED_PARTICLES #StoppingCriterion.MAX_TIME_STEPS
        # *********************
        # Parameter M1 in EWRL-2024 paper, which defines the first threshold for the maximum number of steps to run the FV system for,
        # during which the first-time absorption of 100% of the particles makes the simulation stop.
        M1 = max_time_steps_fv_for_all_particles = N * max_time_steps_fv_per_particle  # 100 (for MountainCar)
        # Parameter M2 in EWRL-2024 paper, which defines the second threshold for the maximum number of steps to run the FV system for,
        # after which the simulation stops, regardless of the number of absorbed particles
        # Use the following to avoid too large simulation times, for instance when the policy is close to optimal:
        # M2 = max_time_steps_fv_overall = max(5000, max_time_steps_fv_for_all_particles)
        M2 = max_time_steps_fv_overall = 2 * max_time_steps_fv_for_all_particles if stopping_criterion_fv.value in [2,
                                                                                                                    3] else max_time_steps_fv_for_all_particles  # 2*max_time_steps_fv_for_all_particles
        min_prop_absorbed_particles = 0.90  # 1.0 #0.90 #0.70    # WARNING: currently (2024/08/09) this ONLY has effect when M2 > M1!! So, if we want to use it just set M1 very small and M2 a value of the order of M1 usually used before
        print(f"Thresholds for FV simulation: T={T}, M1 = {M1}, M2 = {M2}"
              f"\n% Absorbed particles required between M1 and M2: {min_prop_absorbed_particles * 100}% (STOP when reached regardless of t? {stopping_criterion_fv})")

        # For the benchmark method (TDAC)
        verbose_period = max_time_steps_fv_for_all_particles // 10
        time_steps_fv_for_absorption_set = T
        max_time_steps_benchmark = time_steps_fv_for_absorption_set + max_time_steps_fv_for_expectation + max_time_steps_fv_overall
        print(f"max_time_steps_benchmark (T + T + M2) = {max_time_steps_benchmark}")

        test_ac = Test_EstPolicy_EnvGridworldsWithObstacles()
        test_ac.setUpClass(shape=env_shape, n_obstacles=n_obstacles, wind_dict=wind_dict,
                           start_states_set={entry_state},
                           exit_state=exit_state,
                           # Value functions model
                           use_function_approximation=use_function_approximation,
                           nn_input_value_functions=nn_input_value_functions,
                           nn_hidden_layer_sizes_value_functions=nn_hidden_layer_sizes_value_functions,
                           use_separate_target_model=use_separate_target_model, update_period_target_model=update_period_target_model,
                           # Policy model
                           nn_input_policy=nn_input_policy,
                           nn_hidden_layer_sizes_policy=nn_hidden_layer_sizes_policy,
                           initial_policy=initial_policy,
                           # General learning parameters
                           learning_task=learning_task,
                           learning_criterion=learning_criterion,
                           alpha=alpha_initial, gamma=gamma, lmbda=0.7,  # lmbda parameter is ONLY used for TD(lambda), NOT for TD(0), which is created separately
                           alpha_min=alpha_initial/10,
                           # Fleming-Viot parameters
                           # Small N and T are N=50, T=1000 for the 8x12 labyrinth with corridor
                           N=N,
                           T=T,
                           states_of_interest_fv=set({exit_state}),  # None
                           seed=seed, plot=True, debug=False,
                           seed_obstacles=seed_obstacles)
        test_ac.setUp()
        print(test_ac.policy_nn.nn_model)
        # Shape as string, used for filenames
        shape_str = f"{env_shape[0]}x{env_shape[1]}"
        #-------------------------------- TEST SETUP --------------------------#


        #------------------ RESULTS COLLECTION ----------------#
        # Dictionaries to store the results for the different methods
        dict_test_ac = dict()
        dict_simulator = dict()  # Store the simulator used for learning, from where we can get the agent, its policy, etc.
        dict_loss = dict()
        dict_R = dict()
        dict_R_long = dict()
        dict_R_long_initial = dict()
        dict_R_long_fv_inflated = dict()
        dict_R_long_true = dict()  # True average reward under the policy used at each policy learning step to learn value functions. GOAL: Keep track on how rare is reaching the reward.
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
        #------------------ RESULTS COLLECTION ----------------#


        # Define the BASE seed for the simulation
        seed_base = seed

        # Execution parameters that vary with the test bed
        n_learning_steps = int(row[1]['n_learning_steps'])

        # What simulator to use for each learning method (defined in the test object test_ac)
        simulators = dict({ # TD and TD(lambda) learners
                            'values_td': test_ac.sim_td0,
                            'values_tdl': test_ac.sim_td,
                            'values_tdl2': test_ac.sim_td,
                            'values_tda': test_ac.sim_tda,
                            # FV learners
                            'values_fv': test_ac.sim_fv,
                            # FV(lambda) learners
                            'values_fvl': test_ac.sim_fvl,
                            'values_fva': test_ac.sim_fva,
                        })

        # Now run the FVAC vs. TDAC policy learning process on the selected labyrinth sizes
        for learning_method in learning_methods:
            if learning_method_benchmark is not None:
                # This means that we need to get the benchmark information from dict_results, because the FVAC method from where this information is taken was NOT run just now
                # => We set the variable that is used in run_policy_learning_discrete.py to the dict_nsteps entry in the dict_results that corresponds to the labyrinth size
                # we are running now for the learning method that needs to be used to define the benchmark number of steps
                max_time_steps_benchmark_all = dict_results[size]['dict_nsteps'][learning_method_benchmark]
            runfile(os.path.join(rootdir, "../Python/lib/run_policy_learning_discrete.py"), wdir=os.path.join(rootdir, "../Python/lib/"))

        dict_results[size] = {  'params_exec': params_exec,
                                'test_ac': dict_test_ac,
                                'dict_simulator': dict_simulator,
                                'dict_loss': dict_loss,
                                'dict_R': dict_R,
                                'dict_R_long': dict_R_long,
                                'dict_R_long_initial': dict_R_long_initial,
                                'dict_R_long_fv_inflated': dict_R_long_fv_inflated,
                                'dict_R_long_true': dict_R_long_true,
                                'dict_V': dict_V,
                                'dict_Q': dict_Q,
                                'dict_A': dict_A,
                                'dict_state_counts': dict_state_counts,
                                'dict_nsteps': dict_nsteps,
                                'dict_KL': dict_KL,
                                'dict_KL_norm': dict_KL_norm,
                                'dict_alpha': dict_alpha,
                                'dict_LR': dict_LR,
                                'dict_LR_policy': dict_LR_policy,
                                'dict_time_elapsed': dict_time_elapsed,
                                'dict_time_cpu': dict_time_cpu,
                            }

    time_end_all = timer()
    cpu_end_all = process_time()
    time_elapsed_all = time_end_all - time_start_all
    time_cpu_all = cpu_end_all - cpu_start_all
    show_elapsed_time("\nALL", time_elapsed_all, time_cpu_all)

    if save:
        _prefix = f"ActorCritic_{env_type.name.lower()}_{shape_str}_WIND={'None' if wind_dict is None else wind_dict['intensity']}_"
        _filename = f"{_prefix}{dt_start_all}_{'_'.join(learning_methods).upper()}.pkl"
        _filepath = os.path.join(resultsdir, _filename)
        joblib.dump(dict_results, _filepath)
        print(f"Results for ALL Actor-Critic methods on ALL gridworlds saved to '{_filepath}'")

    raise KeyboardInterrupt


    # Read results
    _resultsdir = "E:/Daniel/Projects/PhD-RL-Toulouse/projects/RL-003-Classic/testbed/results"

    _filename = "ActorCritic_gridworld_15x15_20250825_101156_ALL.pkl"
    _filename = "ActorCritic_gridworld_17x17_20250825_153523_ALL.pkl"
    _filename = "ActorCritic_gridworld_19x19_20250826_001136_ALL.pkl"

    # Two files with FVAC(lambda) and TDAC(lambda) results stored separately
    # These correspond to the experiments of increasing the labyrinth size from 8x8, 11x11, 15x15 with WIND=0.5 to analyze when FVAC and TDAC break
    _filename = "ActorCritic_gridworld_4x4_WIND=0.5_20250918_184053_VALUES_FVL_VALUES_TDL_VALUES_TDL2(start@S).pkl"
    _filename = "ActorCritic_gridworld_15x15_WIND=0.5_20250827_124158_VALUES_FVL.pkl"
    _filename = "ActorCritic_gridworld_15x15_WIND=0.5_20250829_232514_VALUES_TDL.pkl"

    dict_results = joblib.load(os.path.join(_resultsdir, _filename))
    #learning_method = "values_tdl"
    learning_method = "values_fvl"
    learning_method_type = learning_method[:9]
    if learning_method_type == "values_fv":
        dict_results_fv = dict_results.copy()
    if learning_method_type == "values_td":
        dict_results_td = dict_results.copy()

    # Plots
    dict_colors, dict_linestyles, dict_legends, figsize = define_plotting_parameters()
    # Use the following if we are ONLY plotting FVAC(lambda) and TDAC(lambda) results, as opposed to ALSO FVAC(0) and TDAC(0) results
    dict_colors['values_fvl'] = dict_colors['values_fv']; dict_linestyles['values_fvl'] = dict_linestyles['values_fv']
    dict_colors['values_tdl'] = dict_colors['values_td']; dict_linestyles['values_tdl'] = dict_linestyles['values_td']

    size = 4 #8 #11 #19 #17 #15 #df_sizes_testbed['size'].iloc[-1]

    # Create variables from the dictionary entries, so that I can use more easily the existing code for plotting
    object_names = []
    for obj_name, obj_value in dict_results[size].items():
        object_names += [obj_name]
        locals()[obj_name] = obj_value
        print(f"Variable {obj_name} created from the dict_results[{size}] entry.")

    # Other variables needed to generate some plots (e.g. ALTOGETHER)
    gamma = 1.0
    env_type = Environment.Gridworld
    env_type_name = params_exec['env_type_name']
    wind_dict = params_exec['wind_dict']
    policy_learning_mode = params_exec['policy_learning_mode']
    learning_task = params_exec['learning_task']
    learning_criterion = params_exec['learning_criterion']
    max_avg_reward_continuing = np.nan
    max_avg_reward_episodic = np.nan
    use_function_approximation = params_exec['use_function_approximation']
    _N = N = params_exec['N']
    _T = T = params_exec['T']
    nn_hidden_layer_sizes = []
    _methods = list(dict_loss.keys())
    env_shape = params_exec['env_shape']
    seed_base = params_exec['seed_base']
    nrep = len(dict_loss[_methods[0]])
    n_learning_steps = params_exec['n_learning_steps']
    rep = 0

    # State visit counts
    rep = 4
    axes = plt.figure().subplots(1, 2)
    learning_methods = list(dict_simulator.keys())
    for i, learning_method in enumerate(learning_methods):
        ax = plot_state_counts(dict_simulator, learning_method, rep, params_exec,
                          ax=axes[i], seed=13838721, add_title=False)  # Seed for the generation of the trajectory on which the state counts are computed
        ax.set_title(f"{learning_method.upper()}")

