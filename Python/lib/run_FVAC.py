# -*- coding: utf-8 -*-
"""
Created on Sun Jun 07 23:48:41 2024

@author: Daniel Mastropietro
@description: Runs the FVAC algorithm (Fleming-Viot Actor-Critic) to learn optimal parameterized policies in discrete-time Markov decision processes.
              Execution is possible either in batch mode (i.e. passing parameters from the prompt) or interactive mode (e.g. from PyCharm), in which case
              the parameters should be defined as default values in the code below, when invoking parser.set_defaults().
"""

if __name__ == "__main__":
    # Only run this when running the script, o.w. it may give an error when importing functions if setup.py is not found
    import runpy
    runpy.run_path('../../setup.py')

#-------------------- IMPORT AND AUXILIARY FUNCTIONS ------------------#
from timeit import default_timer as timer
import os
import sys
import optparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import MaxNLocator
from scipy.special import rel_entr

from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners import LearningCriterion, LearningTask
from Python.lib.agents.learners.policies import LeaActorCriticNN
from Python.lib.agents.policies import probabilistic

from Python.lib.environments.gridworlds import Direction2D, EnvGridworld1D, EnvGridworld2D
from Python.lib.estimators.nn_models import InputLayer

from Python.lib.utils.basic import get_current_datetime_as_string, load_objects_from_pickle, log_file_open, log_file_close, save_objects_to_pickle, set_numpy_options, reset_numpy_options
from Python.lib.utils.computing import compute_expected_reward, compute_transition_matrices, compute_state_value_function_from_transition_matrix

from Python.test.test_optimizers_discretetime import Test_EstPolicy_EnvGridworldsWithObstacles

# When saving results or reading previously saved results
rootdir = "./RL-003-Classic"
resultsdir = f"./{rootdir}/results"
logsdir = f"./{rootdir}/logs"

#--- Auxiliary functions
KL_THRESHOLD = 0.005
policy_changed_from_previous_learning_step = lambda KL_distance, num_states: np.abs(KL_distance) / num_states > KL_THRESHOLD

def define_plotting_parameters():
    dict_colors = dict(); dict_linestyles = dict(); dict_legends = dict()
    dict_colors['all_online'] = "darkred"; dict_linestyles['all_online'] = "solid"; dict_legends['all_online'] = "ALL Online"
    dict_colors['values_td'] = "red"; dict_linestyles['values_td'] = "solid"; dict_legends['values_td'] = "TD"      # For a second trial of TDAC (which sometimes is useful --e.g. to compare "using the same budget as FV for every policy learning step" vs. "using an average budget at each policy learning step")
    dict_colors['values_td2'] = "darkorange"; dict_linestyles['values_td2'] = "solid"; dict_legends['values_td2'] = "TD2"
    dict_colors['values_tdl'] = "orange"; dict_linestyles['values_tdl'] = "dashed"; dict_legends['values_tdl'] = "TD(lambda)"
    dict_colors['values_fv'] = "green"; dict_linestyles['values_fv'] = "solid"; dict_legends['values_fv'] = "FV"    # For a second trial of FVAC (which sometimes is useful --e.g. to compare "allowing infinite budget for FV" vs. "limited budget")
    dict_colors['values_fv2'] = "cyan"; dict_linestyles['values_fv2'] = "solid"; dict_legends['values_fv2'] = "FV2"
    dict_colors['values_fvos'] = "lightgreen"; dict_linestyles['values_fvos'] = "solid"; dict_legends['values_fvos'] = "FV OverSampling"

    figsize = (10, 8)

    return dict_colors, dict_linestyles, dict_legends, figsize

def compute_true_state_value_function(env, policy, learning_task, learning_criterion, gamma=1.0):
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
    P_epi, P_con, b_epi, b_con, g, mu = compute_transition_matrices(env, policy)
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
        else:
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
        else:
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
                                         "[--optimizer_learning_rate] "
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
    parser.add_option("--optimizer_learning_rate",
                      type="float",
                      metavar="Optimizer learning rate",
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
                        env=EnvGridworld2D,
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
    if issubclass(options.env, EnvGridworld1D):
        print("1D Environment length: {}".format(options.env.getNumStates()))
    elif issubclass(options.env, EnvGridworld2D):
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
    print("learning rate = {}".format(options.optimizer_learning_rate))

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
    options, args = parse_input_parameters(sys.argv[1:])

    print("Parsed user arguments:")
    print(f"Options: {options}")
    print(f"Arguments: {args}")
    print("")

    # Define directories (assumed the current directory is the location of this file --which is true when the file is run as a script)
    rootdir = "../../RL-003-Classic"
    resultsdir = f"./{rootdir}/results"
    logsdir = f"./{rootdir}/logs"

    # ----------------- BASIC SETUP AND SIMULATION PARAMETERS --------------#
    # Learning task and learning criterion are used by the constructor of the test class below
    learning_task = LearningTask.CONTINUING
    # learning_criterion = LearningCriterion.DISCOUNTED; gamma = 0.9
    learning_criterion = LearningCriterion.AVERAGE; gamma = 1.0  # gamma could be < 1 in the average reward criterion in order to take the limit as gamma -> 1 as presented in Sutton, pag. 251/252.

    seed = 1317
    problem_2d = True
    exit_state_at_bottom = False
    estimate_absorption_set = True
    estimate_absorption_set_at_every_step = False
    entry_state_in_absorption_set = True  # False #True     # Only used when estimate_absorption_set = False
    # ----------------- BASIC SETUP AND SIMULATION PARAMETERS --------------#

    # -------------------------------- ENVIRONMENT -------------------------#
    if problem_2d:
        # 2D labyrinth
        size_vertical = 3;
        size_horizontal = 4
        size_vertical = 4;
        size_horizontal = 5
        # size_vertical = 6; size_horizontal = 8
        # size_vertical = 8; size_horizontal = 12
        # size_vertical = 9; size_horizontal = 13
        # size_vertical = 10; size_horizontal = 14
        # size_vertical = 10; size_horizontal = 30

        # Whether the active set in FV should be connected (in order to avoid isolation of the two activation states and reduce possible problems)
        # When this is the case, the connectedness of the active set is achieved by removing the left-most obstacle in the previous-to-bottom row.
        connected_active_set = False  # True
        initial_policy = None  # [0.4, 0.15, 0.05, 0.4] #None   # Random walk as initial policy when None
    else:
        # 1D gridworld: the interesting dimension is the vertical dimension, with terminal state at the top (this is to unify the structure of 2D labyrinth and 1D gridworld)
        size_vertical = 21;
        size_horizontal = 1  # We choose a value like K = 20 or 40 in the M/M/1/K queue system
        initial_policy = [0.4, 0.0, 0.6,
                          0.0]  # We choose a probability of going down which is similar to mu/(lambda + mu) in the queue system where lambda/mu = 0.7, i.e. a value close to 1 / 1.7 = 0.588
    env_shape = (size_vertical, size_horizontal)
    nS = size_vertical * size_horizontal
    size_str = f"{size_vertical}x{size_horizontal}"

    # Environment's entry and exit states
    entry_state = np.ravel_multi_index((size_vertical - 1, 0), env_shape)
    exit_state = entry_state + env_shape[1] - 1 if exit_state_at_bottom else None  # None means that the EXIT state is set at the top-right of the labyrinth

    # Presence of wind: direction and probability of deviation in that direction when moving
    if problem_2d:
        wind_dict = None
        wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.5})
        wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.6})
        wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.7})
        wind_dict = dict({'direction': Direction2D.LEFT, 'intensity': 0.8})
    else:
        # WIND is currently not allowed in 1D gridworlds
        wind_dict = None

    # Obstacles
    if problem_2d:
        # -- 2D labyrinth
        # The path to the terminal state is just a corridor through the last row right and then the last column up
        # So at the upper left corner there is a rectangle that brings to nowhere
        rectangle_to_nowhere_width = size_horizontal - 2
        rectangle_to_nowhere_height = size_vertical - 2
        states_previous_to_last_row = np.ravel_multi_index(
            [np.repeat(rectangle_to_nowhere_height, rectangle_to_nowhere_width), [y for y in range(1, rectangle_to_nowhere_width + 1)]], env_shape)
        states_previous_to_last_column = np.ravel_multi_index(
            [[x for x in range(0, rectangle_to_nowhere_height + 1)], np.repeat(rectangle_to_nowhere_width, rectangle_to_nowhere_height + 1)], env_shape)
        obstacles_set = set(np.concatenate([list(states_previous_to_last_row) + list(states_previous_to_last_column)]))

        if connected_active_set:
            obstacles_set = obstacles_set.difference({min(states_previous_to_last_row)})
    else:
        # -- 1D gridworld
        obstacles_set = set()
    # -------------------------------- ENVIRONMENT -------------------------#

    # ----------------------------- MODEL FOR POLICY -----------------------#
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
    # nn_hidden_layer_sizes = [int( 0.8*np.prod(env_shape) ), int( 0.4*np.prod(env_shape) )]
    # Keep the neural network rather small
    nn_hidden_layer_sizes = [12]  # [] #[12]
    print(f"Neural Network architecture:\n{len(nn_hidden_layer_sizes)} hidden layers of sizes {nn_hidden_layer_sizes}")
    # ----------------------------- MODEL FOR POLICY -----------------------#

    # ----------------------------- FV ABSORPTION SET ----------------------#
    # DEFAULT absorption set used when estimate_absorption_set = False
    if problem_2d:
        # The absorption set is a rectangular area at the upper left corner of the grid + (possibly) the lower left corner
        # lower_left_state = (size_vertical-1) * size_horizontal
        # default_absorption_set = set(np.concatenate([list(range(x*size_horizontal, x*size_horizontal + size_horizontal-2)) for x in range(size_vertical-2)]))
        # The absorption set is a rectangular area that touches the right and bottom walls of the big rectangular area that leads to nowhere
        left_margin = 0  # int(rectangle_to_nowhere_width/2)
        top_margin = max(1, int(rectangle_to_nowhere_width / 2) - 1)  # rectangle_to_nowhere_height #rectangle_to_nowhere_height - 2
        default_absorption_set = set(
            np.concatenate([list(range(x * size_horizontal + left_margin, x * size_horizontal + rectangle_to_nowhere_width)) for x in range(0, top_margin)]))
    else:
        # We choose an absorption set size J ~ K/3, like in the M/M/1/K queue system (recall that numbering of states here is the opposite than in M/M/1/K
        default_absorption_set = set(np.arange(np.prod(env_shape) * 2 // 3 + 1, env_shape[0] - 1))

    # Add the environment's start state to the absorption set
    if entry_state_in_absorption_set:
        default_absorption_set.add(entry_state)
    # ----------------------------- FV ABSORPTION SET ----------------------#

    # -------------------------------- TEST SETUP --------------------------#
    test_ac = Test_EstPolicy_EnvGridworldsWithObstacles()
    test_ac.setUpClass(shape=env_shape, obstacles_set=obstacles_set, wind_dict=wind_dict,
                       define_start_state_from_absorption_set=False, start_states_set={entry_state},  # {nS-1}, #None,
                       exit_state=exit_state,
                       # Policy model
                       nn_input=nn_input, nn_hidden_layer_sizes=nn_hidden_layer_sizes, initial_policy=initial_policy,
                       # General learning parameters
                       learning_task=learning_task,
                       learning_criterion=learning_criterion,
                       alpha=1.0, gamma=gamma, lmbda=0.7,
                       # lmbda parameter is ONLY used for TD(lambda), NOT for TD(0), which is created separately nor for FV (for which lambda > 0 does not make sense)
                       alpha_min=0.1,
                       reset_method_value_functions=ResetMethod.ALLZEROS,
                       # Fleming-Viot parameters
                       # Small N and T are N=50, T=1000 for the 8x12 labyrinth with corridor
                       N=20,  # 50, #200, #200 if problem_2d else 100, #50 #20 #100
                       T=500,  # 1000, #100, #10000 if problem_2d else 1000, #1000, #1000, #3000,  # np.prod(env_shape) * 10  #100 #1000
                       estimate_absorption_set=estimate_absorption_set, absorption_set=default_absorption_set,
                       states_of_interest_fv=None,  # exit_state,
                       seed=seed,
                       debug=False)  # CAREFUL! If we set debug=True, LOTS OF PLOTS (~ 900! for 25 policy learning steps on the 4x5 labyrinth) will be generated showing the eligibility traces of the TD(lambda) learner!
    test_ac.setUp()
    print(test_ac.policy_nn.nn_model)
    test_ac.env2d._render()
    print(f"Absorption set (1D):\n{test_ac.agent_nn_fv.getLearner().getAbsorptionSet()}")
    print(f"Absorption set (2D):\n{[np.unravel_index(s, env_shape) for s in test_ac.agent_nn_fv.getLearner().getAbsorptionSet()]}")
    print(f"Activation set (1D):\n{test_ac.agent_nn_fv.getLearner().getActivationSet()}")
    print(f"Activation set (2D):\n{[np.unravel_index(s, env_shape) for s in test_ac.agent_nn_fv.getLearner().getActivationSet()]}")
    # -------------------------------- TEST SETUP --------------------------#

    # ------------------ INITIAL AND MAXIMUM AVERAGE REWARDS ---------------#
    print(f"\nTrue state value function under initial policy:\n{test_ac.env2d.getV()}")

    # Average reward of random policy
    policy_random = probabilistic.PolGenericDiscrete(test_ac.env2d, policy=dict(), policy_default=[0.25, 0.25, 0.25, 0.25])
    V_true, avg_reward_true, mu = compute_true_state_value_function(test_ac.env2d, policy_random, learning_task, learning_criterion)
    print(f"\nAverage reward under RANDOM policy: {avg_reward_true}")
    if initial_policy is not None:
        policy_initial = probabilistic.PolGenericDiscrete(test_ac.env2d, policy=dict(), policy_default=initial_policy)
        V_true, avg_reward_true, mu = compute_true_state_value_function(test_ac.env2d, policy_random, learning_task, learning_criterion)
        print(f"\nAverage reward under INITIAL policy: {avg_reward_true}")

    # Maximum average reward (to use as reference in information and plots)
    max_avg_reward_continuing, max_avg_reward_episodic = compute_max_avg_rewards_in_labyrinth_with_corridor(test_ac.env2d, wind_dict, learning_task, learning_criterion)
    print(f"MAX CONTINUING average reward: {max_avg_reward_continuing}")
    print(f"MAX EPISODIC average reward (approximate when wind present): {max_avg_reward_episodic}")

    # State to observe and e.g. plot Q values, etc
    if exit_state == entry_state + env_shape[1] - 1:
        state_observe = np.ravel_multi_index((env_shape[0] - 1, env_shape[1] - 2), env_shape)
    else:
        state_observe = np.ravel_multi_index((1, env_shape[1] - 1), env_shape)
    # ------------------ INITIAL AND MAXIMUM AVERAGE REWARDS ---------------#

    # ------------------ RESULTS COLLECTION AND PLOTS SETUP ----------------#
    # Learning methods: define characteristics to use when plotting results for each method
    dict_colors, dict_linestyles, dict_legends, figsize = define_plotting_parameters()

    # Dictionaries to store the results for the different methods (for comparison purposes)
    dict_loss = dict()
    dict_R = dict()
    dict_R_long = dict()
    dict_R_long_true = dict()  # True average reward under the policy used at each policy learning step to learn value functions. GOAL: Keep track on how rare is reaching the reward.
    dict_V = dict()
    dict_Q = dict()
    dict_A = dict()
    dict_state_counts = dict()
    dict_nsteps = dict()
    dict_KL = dict()
    dict_alpha = dict()
    dict_time_elapsed = dict()
    # ------------------ RESULTS COLLECTION AND PLOTS SETUP ----------------#

    # Number of replications to run on each method
    nrep = 5 # 10
    # Logging
    log = True  # False #learning_method_type == "values_fv"

    # Learning method (of the value functions and the policy)
    # Both value functions and policy are learned online using the same simulation
    learning_method = "all_online";
    simulator_value_functions = None
    # Value functions are learned separately from the policy
    # Policy learning can happen online or OFFLINE
    learning_method = "values_td"; simulator_value_functions = test_ac.sim_td0  # TD(0)
    # learning_method = "values_td2"; simulator_value_functions = test_ac.sim_td0    # TD(0)
    # learning_method = "values_tdl"; simulator_value_functions = test_ac.sim_td     # TD(lambda)
    # learning_method = "values_tda"; simulator_value_functions = test_ac.sim_tda    # Adaptive TD(lambda)
    learning_method = "values_fv"; simulator_value_functions = test_ac.sim_fv
    # learning_method = "values_fv2"; simulator_value_functions = test_ac.sim_fv
    # learning_method = "values_fvos"; simulator_value_functions = test_ac.sim_fv     # FV used just as an oversampling method

    learning_method_type = learning_method[:9]  # This makes e.g. "values_fvos" become "values_fv"

    # FV learning parameters (which are used to define parameters of the other learners analyzed so that their comparison with FV is fair)
    max_time_steps_fv_per_particle = 5 * len(test_ac.agent_nn_fv.getLearner().getActiveSet())  # 100 #50                            # Max average number of steps allowed for each particle in the FV simulation. We set this value proportional to the size of the active set of the FV learner, in order to scale this value with the size of the labyrinth.
    max_time_steps_fv_for_expectation = test_ac.agent_nn_fv.getLearner().getNumTimeStepsForExpectation()  # This is parameter T
    max_time_steps_fv_for_all_particles = test_ac.agent_nn_fv.getLearner().getNumParticles() * max_time_steps_fv_per_particle  # This is parameter N * max_time_steps_fv_per_particle which defines the maximum number of steps to run the FV system when learning the value functions that are used as critic of the loss function at each policy learning step
    max_time_steps_fv_overall = 3 * max_time_steps_fv_for_all_particles  # max(5000, max_time_steps_fv_for_all_particles)            # To avoid too large simulation times, for instance when the policy is close to optimal: ABSOLUTE maximum number of steps allowed for the FV simulation, regardless of what happens with particle absorption (i.e. if all of them are absorbed or 90% of them are absorbed at least once)

    # Traditional method learning parameters
    # They are set for a fair comparison with FV learning
    # The maximum time steps to be observed in the benchmark methods is set to the sum of:
    # - the number of steps used to estimate the absorption set A (as long as this exploration ALSO estimates value functions!)
    # - the max number of time steps allowed to estimate E(T_A)
    # - the max number of time steps allowed over all the FV particles
    # Values previously used: #2000 #test_ac.env2d.getNumStates()*10
    # Number of steps used to estimate the absorption set A
    time_steps_fv_for_absorption_set = test_ac.learner_for_initial_exploration.getNumSteps() if test_ac.learner_for_initial_exploration is not None else 0
    assert time_steps_fv_for_absorption_set > 0 if estimate_absorption_set else None
    max_time_steps_benchmark = time_steps_fv_for_absorption_set + max_time_steps_fv_for_expectation + max_time_steps_fv_for_all_particles

    # -- Common learning parameters
    # Parameters about policy learning (Actor)
    n_learning_steps = 100  # 200 #50 #100 #30
    n_episodes_per_learning_step = 50  # 100 #30   # This parameter is used as the number of episodes to run the policy learning process for and, if the learning task is EPISODIC, also as the number of episodes to run the simulators that estimate the value functions
    max_time_steps_per_policy_learning_episode = 5 * test_ac.env2d.getNumStates() if problem_2d else 2 * test_ac.env2d.getNumStates()  # np.prod(env_shape) * 10 #max_time_steps_benchmark // n_episodes_per_learning_step   # Maximum number of steps per episode while LEARNING THE *POLICY* ONLINE (NOT used for the value functions (critic) learning)
    policy_learning_mode = "online"  # "offline" #"online"     # Whether the policy is learned online or OFFLINE (only used when value functions are learned separately from the policy)
    allow_deterministic_policy = True  # False
    use_average_reward_from_previous_step = True  # learning_method_type == "values_fv" #False #True            # Under the AVERAGE reward crtierion, whether to use the average reward estimated from the previous policy learning step as correction of the value functions (whenever it is not 0), at least as an initial estimate
    use_advantage = not (learning_method == "values_fvos")  # Set this to True if we want to use the advantage function learned as the TD error, instead of using the advantage function as the difference between the estimated Q(s,a) and the estimated V(s) (where the average reward cancels out)
    optimizer_learning_rate = 0.05 #0.01 if policy_learning_mode == "online" else 0.05  # 0.01 #0.1
    reset_value_functions_at_every_learning_step = False  # (learning_method == "values_fv")     # Reset the value functions when learning with FV, o.w. the learning can become too unstable due to the oversampling of the states with high value... (or something like that)

    # Parameters about value function learning (Critic)
    alpha_initial = simulator_value_functions.getAgent().getLearner().getInitialLearningRate()
    adjust_alpha_initial_by_learning_step = False;
    t_learn_min_to_adjust_alpha = 30  # based at 1 (regardless of the base value used for t_learn)
    # max_time_steps_per_episode = test_ac.env2d.getNumStates()*10  # (2024/05/02) NO LONGER USED!  # This parameter is just set as a SAFEGUARD against being blocked in an episode at some state of which the agent could be liberated by restarting to a new episode (when this max number of steps is reached)
    epsilon_random_action = 0.1  # if policy_learning_mode == "online" else 0.0 #0.1 #0.05 #0.0 #0.01
    use_average_max_time_steps_in_td_learner = False  # learning_method == "values_td2" #True #False
    learning_steps_observe = [50, 90]  # [2, 30, 48] #[2, 10, 11, 30, 31, 49, 50] #[7, 20, 30, 40]  # base at 1, regardless of the base value used for t_learn
    verbose_period = max_time_steps_fv_for_all_particles // 10
    plot = False  # Whether to plot the evolution of value function and average reward estimation
    colormap = "seismic"  # "Reds"  # Colormap to use in the plot of the estimated state value function V(s)

    # Results saving, with filename prefix and suffix
    save = True
    prefix = f"ActorCritic_labyrinth_{size_str}_"
    suffix = f"_{learning_method}"

    # Open log file if one requested and show method being run
    if log:
        dt_start, stdout_sys, stderr_sys, fh_log, _logfile_not_used = log_file_open(logsdir, subdir="", prefix=prefix, suffix=suffix, use_datetime=True)
    print("******")
    print(f"Running {learning_method.upper()} method for value functions estimation.")
    print(f"A NOMINAL MAXIMUM of {max_time_steps_benchmark} steps will be allowed during the simulation.")
    print("******")

    # A few further parameters for the policy learning process
    break_when_no_change = False  # Whether to stop the learning process when the average reward doesn't change from one step to the next
    break_when_goal_reached = False  # Whether to stop the learning process when the average reward is close enough to the maximum average reward (by a relative tolerance of 0.1%)

    # Initialize objects that will contain the results by learning step
    state_counts_all = np.zeros((nrep, n_learning_steps, test_ac.env2d.getNumStates()))
    V_all = np.zeros((nrep, n_learning_steps, test_ac.env2d.getNumStates()))
    Q_all = np.zeros((nrep, n_learning_steps, test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions()))
    A_all = np.zeros((nrep, n_learning_steps, test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions()))
    R_all = np.nan * np.ones((nrep, n_learning_steps))  # Average reward (EPISODIC learning task)
    R_long_all = np.nan * np.ones((nrep,
                                   n_learning_steps))  # Long-run Average reward (CONTINUING learning task). It does NOT converge to the same value as the episodic average reward because there is one more reward value per episode!! (namely the reward going from the terminal state to the start state)
    R_long_true_all = np.nan * np.ones(
        (nrep, n_learning_steps))  # True Long-run Average reward (CONTINUING learning task) under the policy at the start of each policy learning step
    loss_all = np.nan * np.ones((nrep, n_learning_steps))
    nsteps_all = np.nan * np.ones((nrep, n_learning_steps))  # Number of value function time steps run per every policy learning step
    KL_all = np.nan * np.ones((nrep, n_learning_steps))  # K-L divergence between two consecutive policies
    alpha_all = alpha_initial * np.ones((nrep, n_learning_steps))  # Initial alpha used at each policy learning step
    time_elapsed_all = np.nan * np.ones(nrep)  # Execution time for each replication

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
        learner_ac = LeaActorCriticNN(test_ac.env2d, simulator_value_functions.getAgent().getPolicy(), simulator_value_functions.getAgent().getLearner(),
                                      allow_deterministic_policy=allow_deterministic_policy,
                                      reset_value_functions=reset_value_functions_at_every_learning_step, initial_policy=initial_policy,
                                      optimizer_learning_rate=optimizer_learning_rate, seed=test_ac.seed, debug=True)
    else:
        # Value functions (Critic) are learned separately from the application of the policy and the policy (Actor) may be learned OFFLINE or online
        # IMPORTANT: We pass the policy of the agent stored in the value functions simulator as policy for the Actor-Critic learner so that when the Actor-Critic learner
        # updates the policy, the policy of the agent stored in the value functions simulator is ALSO updated. This is crucial for using the updated policy
        # when learning the value functions at the next policy learning step.
        learner_ac = LeaActorCriticNN(test_ac.env2d, simulator_value_functions.getAgent().getPolicy(), simulator_value_functions.getAgent().getLearner(),
                                      allow_deterministic_policy=allow_deterministic_policy,
                                      reset_value_functions=reset_value_functions_at_every_learning_step, initial_policy=initial_policy,
                                      optimizer_learning_rate=optimizer_learning_rate, seed=test_ac.seed, debug=True)

    time_start = timer()
    dt_start_filename = get_current_datetime_as_string(format="filename")

    seed_base = test_ac.seed
    for rep in range(nrep):
        seed_rep = seed_base + rep * 1317
        print(f"\n->>>>>>>>>>>\nRunning replication {rep + 1} of {nrep} (seed={seed_rep})... @{format(get_current_datetime_as_string())}")

        # Reset the policy actor and the critic every time a new replication starts
        # Note that this performs a further reset of the policy (besides the one done above by the constructor that creates the learner_ac object),
        # and thus resets the policy to ANOTHER slightly different policy (because of the normally distributed random values around 0 that are set as neural network weights)
        print("Resetting the policy learner and the critic (if any)...")
        learner_ac.reset(reset_value_functions=True, reset_policy=True, initial_policy=initial_policy)
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
                print(f"\n\n*** Running learning step {t_learn + 1} of {n_learning_steps} (AVERAGE REWARD at previous step = {R_all[rep, max(0, t_learn - 1)]}) of "
                      f"MAX={max_avg_reward_episodic if policy_learning_mode == 'online' else max_avg_reward_continuing} using {nsteps_all[rep, max(0, t_learn - 1)]} time steps for Critic estimation)... (seed={seed_learn}) @{get_current_datetime_as_string()}")
                print("Learning the VALUE FUNCTIONS and POLICY simultaneously...")
                loss_all[rep, t_learn] = learner_ac.learn(n_episodes_per_learning_step, start_state=entry_state,
                                                          max_time_steps_per_episode=max_time_steps_per_policy_learning_episode,
                                                          prob_include_in_train=1.0)  # prob_include_in_train=0.5)
                ## Note that we make sure that the start state when learning the policy is the entrance state to the labyrinth, `entry_state`, because the environment may have defined
                ## a different initial state distribution (e.g. a random start in the states outside the absorption set used by the FV learner.

                state_counts_all[rep, t_learn, :] = learner_ac.learner_value_functions.getStateCounts()
                V_all[rep, t_learn, :] = learner_ac.learner_value_functions.getV().getValues()
                Q_all[rep, t_learn, :, :] = learner_ac.learner_value_functions.getQ().getValues().reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())
                A_all[rep, t_learn, :, :] = learner_ac.learner_value_functions.getA().getValues().reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())
                R_all[rep, t_learn] = learner_ac.learner_value_functions.getAverageReward()
                # Could also retrieve the average reward from the Actor-Critic learner (if store_trajectory_history=False in the constructor of the value functions learner)
                # R_all[rep, t_learn] = learner_ac.average_reward_over_episodes
        else:
            # Keep track of the policy learned so that we can analyze how much it changes after each learning step w.r.t. the previous learning step
            policy_prev = None
            for t_learn in range(n_learning_steps):
                # Compute or update the true state value function stored in the environment for the current policy
                # (used as reference when plotting the evolution of the estimated state value function V(s) when plot=True)
                V_true, avg_reward_true, mu = compute_true_state_value_function(test_ac.env2d, learner_ac.getPolicy(), learning_task, learning_criterion)
                R_long_true_all[rep, t_learn] = avg_reward_true

                # Pass a different seed (for the simulator) for each learning step... o.w. we will be using the same seed for them at every learning step!!
                seed_learn = seed_rep + t_learn
                print(
                    f"\n\n*** Running learning step {t_learn + 1} of {n_learning_steps} (True average reward under current policy = {avg_reward_true}) (AVERAGE REWARD at previous step = {R_all[rep, max(0, t_learn - 1)]} of MAX={max_avg_reward_episodic})... (seed={seed_learn}) @{get_current_datetime_as_string()}")
                reset_value_functions_at_this_step = reset_value_functions_at_every_learning_step if t_learn > 0 else True  # ALWAYS RESET THE VALUE FUNCTIONS WHEN IT'S THE VERY FIRST LEARNING STEP (becaue we don't want to keep histroy from a earlier learning process on the same learner!)
                # Update the initial learning rate for the value functions at each learning step to a smaller value than the previous learning step
                # SHOULD WE SET IT TO THE AVERAGE LEARNING RATE FROM THE PREVIOUS LEARNING STEP?? (so that we start off where we left at the last learning moment)
                alpha_initial_at_current_learning_step = alpha_initial / (t_learn + 1)

                # -- Optionally adjust the initial learning rate alpha
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
                policy = learner_ac.getPolicy().get_policy_values()
                KL_distance = np.sum(rel_entr(policy, policy_prev)) if t_learn > 0 else 0.0
                print(f"K-L distance with previous policy: {KL_distance} (should be < {KL_THRESHOLD} to reduce initial alpha)")
                KL_all[rep, t_learn] = KL_distance
                policy_prev = policy.copy()

                if adjust_alpha_initial_by_learning_step:
                    # Update the initial learning rate alpha ONLY when:
                    # - the learning step is larger than or equal to a minimum
                    # - the policy did NOT change significantly from the previous learning step
                    #   (because in that case we can consider the current estimate to be a reliable estimator of the value functions)
                    if t_learn + 1 >= t_learn_min_to_adjust_alpha:
                        if policy_changed_from_previous_learning_step(KL_distance, test_ac.env2d.getNumStates()):  # np.abs(loss_all[rep, t_learn-1]) > 1.0
                            simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial)
                        else:
                            simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial / 10)
                    alpha_all[rep, t_learn] = simulator_value_functions.getAgent().getLearner().getInitialLearningRate()
                print(
                    f"*** INITIAL learning rate alpha = {simulator_value_functions.getAgent().getLearner().getInitialLearningRate()} " +
                    (f"(adjustment happens starting at learning step (base 1) >= {t_learn_min_to_adjust_alpha}) " if adjust_alpha_initial_by_learning_step else "***\n"))

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
                                                      estimate_absorption_set=estimate_absorption_set_at_every_step,
                                                      use_average_reward_stored_in_learner=use_average_reward_from_previous_step,
                                                      reset_value_functions=reset_value_functions_at_this_step,
                                                      plot=plot if t_learn + 1 in learning_steps_observe else False, colormap=colormap,
                                                      epsilon_random_action=epsilon_random_action,
                                                      seed=seed_learn, verbose=False, verbose_period=verbose_period)
                    # average_reward = simulator_value_functions.getAgent().getLearner().getAverageReward()  # This average reward should not be used because it is inflated by the FV process that visits the states with rewards more often
                    average_reward = expected_reward
                    nsteps_all[rep, t_learn] = n_events_et + n_events_fv
                    max_time_steps_benchmark_all[rep, t_learn] = n_events_et + n_events_fv  # Number of steps to use when running TDAC at the respective learning step
                else:
                    # TD learners
                    if 'max_time_steps_benchmark_all' in locals() and rep < len(max_time_steps_benchmark_all) and t_learn < len(max_time_steps_benchmark_all[rep, :]) and \
                            max_time_steps_benchmark_all[rep, t_learn] != np.nan:
                        # The FV learner was run before running this TD learner
                        if use_average_max_time_steps_in_td_learner:
                            _max_time_steps = int(np.mean(max_time_steps_benchmark_all[rep, :]))
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
                                                          state_observe=state_observe, compute_rmse=True if t_learn + 1 in learning_steps_observe else False,
                                                          epsilon_random_action=epsilon_random_action,
                                                          plot=plot if t_learn + 1 in learning_steps_observe else False, colormap=colormap,
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
                                                          compute_rmse=True if t_learn + 1 in learning_steps_observe else False,
                                                          plot=plot if t_learn + 1 in learning_steps_observe else False, colormap=colormap,
                                                          epsilon_random_action=epsilon_random_action,
                                                          verbose=True, verbose_period=verbose_period)
                    average_reward = simulator_value_functions.getAgent().getLearner().getAverageReward()
                    nsteps_all[rep, t_learn] = learning_info['nsteps']

                print(f"Learning step #{t_learn + 1}: Learning of value functions COMPLETED using {learning_method} method on {nsteps_all[rep, t_learn]} time steps")
                print(f"Estimated average reward: {average_reward}")
                state_counts_all[rep, t_learn, :] = state_counts
                V_all[rep, t_learn, :] = V
                Q_all[rep, t_learn, :, :] = Q.reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())
                A_all[rep, t_learn, :, :] = A.reshape(test_ac.env2d.getNumStates(), test_ac.env2d.getNumActions())

                print(f"\nLearning the POLICY {policy_learning_mode.upper()} using estimated {use_advantage and 'ADVANTAGE A(s,a) values' or 'ACTION Q(s,a) values'} ", end=" ")
                # Policy learning
                if policy_learning_mode == "online":
                    # ONLINE with critic provided by the action values learned above using the TD simulator
                    print(
                        f"on {n_episodes_per_learning_step} episodes starting at state s={entry_state}, using MAX {max_time_steps_per_policy_learning_episode} steps per episode...")
                    loss_all[rep, t_learn] = learner_ac.learn(n_episodes_per_learning_step, start_state=entry_state,
                                                              max_time_steps_per_episode=max_time_steps_per_policy_learning_episode, prob_include_in_train=1.0,
                                                              use_advantage=use_advantage,
                                                              advantage_values=A,
                                                              action_values=Q)  # This parameter is not used when use_advantage=False
                    ## Note that we make sure that the start state when learning the policy is the entrance state to the labyrinth, `entry_state`,
                    ## because the environment may have defined a different initial state distribution, which is used during the learning of the value functions,
                    ## for instance, any randomly selected state outside the absorption set A used by the FV learner.
                    ## Note that the learned value functions are passed as critic to the Actor-Critic policy learner via the `action_values` parameter.
                    R_all[rep, t_learn] = learner_ac.average_reward_over_episodes
                    R_long_all[rep, t_learn] = average_reward  # This is the average reward estimated by the value functions learner used above
                else:
                    # OFFLINE learner where ALL states and actions are swept and the loss computed on all of them using the state distribution as weights
                    print("and estimated probabilities from Critic estimation excursion...")
                    if learning_method_type == "values_fv":
                        prob_states = compute_prob_states(state_counts_et, probas_stationary=probas_stationary)
                    else:
                        prob_states = compute_prob_states(state_counts)
                    loss_all[rep, t_learn] = learner_ac.learn_offline_from_estimated_value_functions(V, A, Q, state_counts, prob_states=prob_states, use_advantage=use_advantage)
                    R_all[rep, t_learn] = average_reward
                    R_long_all[rep, t_learn] = average_reward
                    _dict_numpy_options = set_numpy_options()
                    print(f"True stationary probabilities:\n{mu.reshape(env_shape)}")
                    print(f"Estimated stationary probabilities:\n{prob_states.reshape(env_shape)}")
                    reset_numpy_options(_dict_numpy_options)
                    if False and (average_reward != 0.0 or t_learn + 1 in learning_steps_observe):
                        def plot_probas(ax, prob_states_2d, fontsize=14, colormap="Blues", color_text="orange"):
                            colors = cm.get_cmap(colormap)
                            ax.imshow(prob_states_2d, cmap=colors)
                            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                            for x in range(env_shape[0]):
                                for y in range(env_shape[1]):
                                    ax.text(y, x, "{:.3f}".format(prob_states_2d[x, y]), color=color_text, fontsize=fontsize, horizontalalignment="center",
                                            verticalalignment="center")


                        ax_true, ax_est, ax_diff = plt.figure().subplots(1, 3)
                        _fontsize = 14
                        _factor_fs = np.min((5 / env_shape[0], 5 / env_shape[1]))
                        plot_probas(ax_true, mu.reshape(env_shape), fontsize=_fontsize * _factor_fs)
                        plot_probas(ax_est, prob_states.reshape(env_shape), fontsize=_fontsize * _factor_fs)
                        plot_probas(ax_diff, (prob_states - mu).reshape(env_shape), fontsize=_fontsize * _factor_fs, colormap="jet")
                        plt.pause(0.1)
                        plt.draw()
                        input("Press ENTER to continue...")

                # Check if we need to stop learning because the average reward didn't change a bit
                if break_when_no_change and t_learn > 0 and R_all[rep, t_learn] - R_all[rep, t_learn - 1] == 0.0 or \
                        break_when_goal_reached and np.isclose(R_all[rep, t_learn], max_avg_reward_episodic, rtol=0.001):
                    print(
                        f"*** Policy learning process stops at learning step t_learn+1={t_learn + 1} because the average reward didn't change a bit from previous learning step! ***")
                    break
        time_elapsed_rep = timer() - time_start_rep
        print(f"<<<<<<<<<< FINISHED replication {rep + 1} of {nrep}... (@{format(get_current_datetime_as_string())}" + ", took {:.1f} min)".format(time_elapsed_rep / 60))
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






    ############## SAVE RESULTS
    if save:
        _env2d = test_ac.env2d
        objects_to_save = ["_env2d", "wind_dict", "learning_task", "learning_criterion", "gamma", "exit_state", "policy_learning_mode",
                           "dict_loss", "dict_R", "dict_R_long", "dict_R_long_true", "dict_V", "dict_Q", "dict_A", "dict_state_counts", "dict_nsteps", "dict_KL", "dict_alpha", "dict_time_elapsed",
                           "max_time_steps_benchmark"]
        if "max_time_steps_benchmark_all" in locals():
            objects_to_save += ["max_time_steps_benchmark_all"]
        else:
            objects_to_save += ["max_time_steps_benchmark"]
        _size_str = f"{env_shape[0]}x{env_shape[1]}"
        _filename = f"{prefix}{dt_start_filename}_ALL.pkl"
        _filepath = os.path.join(resultsdir, _filename)
        # Save the original object names plus those with the prefix so that, when loading the data back,
        # we have also the objects without the prefix which are need to generate the plots right-away.
        # HOWEVER, when reading the saved data, make sure that any object with the original name (without the suffix) has been copied to another object
        # (e.g. loss_all_td could be a copy of loss_all before reading the data previously saved for the FV learning method)
        object_names_to_save = objects_to_save + [f"{obj_name}" for obj_name in objects_to_save]
        save_objects_to_pickle(object_names_to_save, _filepath, locals())
        print(f"Results for ALL Actor-Critic methods: {[str.replace(meth, 'values_', '').upper() for meth in dict_loss.keys()]} saved to '{_filepath}'")
    ############## SAVE RESULTS
