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
from Python.lib.simulators.fv import StoppingCriterion

from Python.lib.utils.basic import get_current_datetime_as_string, load_objects_from_pickle, log_file_open, log_file_close, save_objects_to_pickle, set_numpy_options, reset_numpy_options
from Python.lib.utils.computing import compute_expected_reward, compute_transition_matrices, compute_state_value_function_from_transition_matrix

from Python.test.test_optimizers_discretetime import Test_EstPolicy_EnvGridworldsWithObstacles, Test_EstPolicy_EnvMountainCar

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


#-------------------- AUXILIARY AND PLOTTING FUNCTIONS ------------------#
#--- Auxiliary functions
KL_THRESHOLD = 0.005
#policy_changed_from_previous_learning_step = lambda KL_distance, num_states: np.abs(KL_distance) / num_states > KL_THRESHOLD

def compute_normalized_KL_distance(KL_distance, learner):
    "Computes the KL distance normalized by the number of environment states KNOWN by the given learner. The learner needs to define the getKnownEnvironmentSet() method"
    num_states_known_by_learner = len(learner.getKnownEnvironmentSet())
    return KL_distance / max(1, num_states_known_by_learner), num_states_known_by_learner

def show_elapsed_time(learning_method, time_elapsed, time_cpu):
    print("{} learning process took {:.1f} minutes, ({:.1f} hours (CPU: {:.1f} minutes, {:.1f} hours)".format(learning_method.upper(), time_elapsed / 60, time_elapsed / 3600, time_cpu / 60, time_cpu / 3600))

def define_plotting_parameters():
    dict_colors = dict(); dict_linestyles = dict(); dict_legends = dict()
    dict_colors['all_online'] = "darkred"; dict_linestyles['all_online'] = "solid"; dict_legends['all_online'] = "ALL Online"
    dict_colors['values_td'] = "red"; dict_linestyles['values_td'] = "solid"; dict_legends['values_td'] = "TDAC"      # For a second trial of TDAC (which sometimes is useful --e.g. to compare "using the same budget as FV for every policy learning step" vs. "using an average budget at each policy learning step")
    dict_colors['values_td2'] = "darkorange"; dict_linestyles['values_td2'] = "solid"; dict_legends['values_td2'] = "TDAC2"
    dict_colors['values_tdl'] = "orange"; dict_linestyles['values_tdl'] = "dashed"; dict_legends['values_tdl'] = "TDAC(Lambda)"
    dict_colors['values_tdl2'] = "darkorange"; dict_linestyles['values_tdl2'] = "dashed"; dict_legends['values_tdl2'] = "TDAC2(Lambda)"
    dict_colors['values_tda'] = "darkred"; dict_linestyles['values_tda'] = "solid"; dict_legends['values_tda'] = "TDAC(LambdaAdap)"
    dict_colors['values_fv'] = "green"; dict_linestyles['values_fv'] = "solid"; dict_legends['values_fv'] = "FVAC"    # For a second trial of FVAC (which sometimes is useful --e.g. to compare "allowing infinite budget for FV" vs. "limited budget")
    dict_colors['values_fv2'] = "cyan"; dict_linestyles['values_fv2'] = "solid"; dict_legends['values_fv2'] = "FVAC2"
    dict_colors['values_fv3'] = "black"; dict_linestyles['values_fv3'] = "solid"; dict_legends['values_fv3'] = "FVAC3"
    dict_colors['values_fvos'] = "lightgreen"; dict_linestyles['values_fvos'] = "solid"; dict_legends['values_fvos'] = "FVAC OverSampling"
    dict_colors['values_fvl'] = "blue"; dict_linestyles['values_fvl'] = "dashed"; dict_legends['values_fvl'] = "FVAC(Lambda)"
    dict_colors['values_fvl2'] = "magenta"; dict_linestyles['values_fvl2'] = "dashed"; dict_legends['values_fvl2'] = "FVAC(Lambda)"
    dict_colors['values_fvl3'] = "pink"; dict_linestyles['values_fvl3'] = "dashed"; dict_legends['values_fvl3'] = "FVAC(Lambda)"
    dict_colors['values_fva'] = "darkolivegreen"; dict_linestyles['values_fva'] = "solid"; dict_legends['values_fva'] = "FVAC(LambdaAdap)"

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
    bias = g if learning_criterion == LearningCriterion.AVERAGE else 0.0
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

#--- Plotting functions
def plot_trajectory(env, trajectory, params_exec, ax=None, style=".-", figsize=(8, 8), cmap="coolwarm", markersize=5, pause=0.0):
    """
    Plots a trajectory of states in the given environment which is assumed to be 2D

    The states in the trajectory are connected by lines defined by `style`.

    Arguments:
    env: Environment
        2D environment where the trajectory is plotted.
        Ex: a 2D gridworld, the mountain car

    trajectory: array-like
        Trajectory to plot.

    ax: (opt) Axes object
        Existing axes object on which the trajectory should be added.
        default: None, in which case a new axes object is created

    style: (opt) str
        Style of the plotted trajectory, it's the symbol used in matplotlib.pyplot.plot(), e.g. ".-" or "." or "x".

    pause: (opt) int
        Number of seconds to pause after each added point. For better visualization purposes.
        default: 0.0
    """
    env_type = params_exec['env_type']
    if env_type == Environment.Gridworld:
        ax = env.plot()
        env.plot_points(trajectory, ax=ax)
    else:
        assert env_type == Environment.MountainCar
        if ax is None:
            ax = plt.figure(figsize=figsize).subplots(1, 1)
            env._finalize_plot(ax)

        n_points = len(trajectory)
        colors = cm.get_cmap(cmap, lut=n_points)
        for i, point in enumerate(trajectory[:-1]):
            # Convert the point to a 2D index representing the entry of the point in the 2D shape representation of the environment
            idx_point = env.getIndexFromState(point)
            idx_point_next = env.getIndexFromState(trajectory[i + 1])
            idx_point_2d = env.getStateIndicesFromIndex(idx_point)
            idx_point_2d_next = env.getStateIndicesFromIndex(idx_point_next)
            _color = colors(i / n_points)
            if env.getShapeDisplayNames() == ("velocity", "position"):
                # Velocity is on the VERTICAL axis and position on the HORIZONTAL axis
                xvalues = [idx_point_2d[env.getPositionDimension()], idx_point_2d_next[env.getPositionDimension()]]
                yvalues = [idx_point_2d[env.getVelocityDimension()], idx_point_2d_next[env.getVelocityDimension()]]
            else:
                # Velocity is on the HORIZONTAL axis and position on the VERTICAL axis
                xvalues = [idx_point_2d[env.getVelocityDimension()], idx_point_2d_next[env.getVelocityDimension()]]
                yvalues = [idx_point_2d[env.getPositionDimension()], idx_point_2d_next[env.getPositionDimension()]]
            # Connect the current point with the next
            ax.plot(xvalues, yvalues, style, color=_color, markersize=markersize)
            # Mark the next point differently (different color and larger size), so that we can see where we are plotting
            ax.plot(xvalues[1], yvalues[1], style, color="cyan", markersize=2 * markersize)
            # Remove the special mark from the current point
            ax.plot(xvalues[0], yvalues[0], style, color="white", markersize=2 * markersize)
            ax.plot(xvalues[0], yvalues[0], style, color=_color, markersize=markersize)
            if pause > 0:
                ax.set_title(f"Step {i + 1} of {len(trajectory)}")
                plt.pause(pause)
                plt.draw()

    return ax

def plot_state_counts(dict_simulator, learning_method, params_exec, trajectory=None, trajectory_length=1000, seed=None, verbose=False, verbose_period=1, plot_absorption_set=True):
    """
    trajectory_length: Length of the trajectory to generate when no `trajectory` is given.
    seed: Seed to generate the trajectory under the current policy stored in the `dict_simulator` simulator when no `trajectory` is given.
    """
    learning_method_type = learning_method[:9]
    learning_task = params_exec['learning_task']
    learning_criterion = params_exec['learning_criterion']
    max_time_steps_benchmark = params_exec['max_time_steps_benchmark']
    env_type = params_exec['env_type']
    N = params_exec['N']
    T = params_exec['T']
    wind_dict = params_exec['wind_dict']

    if trajectory is None:
        # Generate a trajectory under the policy stored in the learner of the simulator
        _simulator = copy.deepcopy(dict_simulator[learning_method])
        _learner_under_policy, _nsteps, _average_reward = _simulator.run_exploration(max_time_steps=trajectory_length, epsilon_random_action=epsilon_random_action, seed=seed, verbose=verbose, verbose_period=verbose_period)
        trajectory = np.array(_learner_under_policy.getStates())
        # Generate the 1D array containing the state counts for each state index
        # (as the above run_exploration() method does NOT update the state counts of the learner because this is done by the learn() method of the learner and the run_exploration()
        # method does NOT learn, it only collects a trajectory)
        state_counts = _learner_under_policy.getStateCountsFromTrajectory()
    else:
        # Distribution of state counts in the given trajectory
        state_counts = np.zeros(dict_simulator[learning_method].getEnv().getNumStates(), dtype=int)
        _visited_states = pd.Series(trajectory).value_counts()
        for s, c in _visited_states.items():
            state_counts[s] = c

    # Show the state counts as an image
    ax, img = dict_simulator[learning_method].getEnv().plot_values(state_counts, cmap="Blues")
    if env_type == Environment.MountainCar:
        # Add the trajectory of the last replication
        assert T <= len(trajectory)
        trajectory2plot = trajectory[:T]
        dict_simulator[learning_method].getEnv().plot_points(trajectory2plot, ax=ax, is_trajectory=True)

    if learning_method_type == "values_fv" and plot_absorption_set:
        # Plot the final absorption set
        dict_simulator[learning_method].getEnv().plot_points(np.array(list(dict_simulator[learning_method].getAgent().getLearner().getAbsorptionSet())), ax=ax, color="red", markersize=5, style="x")

    # Add the count labels
    dict_simulator[learning_method]._add_count_labels(ax, state_counts, factor_fontsize=3.0)
    plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion - {env_type.name} {dict_simulator[learning_method].getEnv().getShape()}"
                 f"\nN={N}, T={T}, wind_dict={wind_dict}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
                 f"\nDistribution of state counts by a trajectory that follows the last learned policy AND final absorption set A")

def plot_policy(env, policy, average_reward, state_counts_hist, params_exec, axes=None, is_problem_2d=True, KL_distance=None, KL_distance_norm=None, absorption_set=None, t_learn=1, fontsize=14, verbose=True):
    """
    Plots the given policy on the given environment

    On top of the policy the following plots are overlaid in the Gridworld environment:
    - The state counts are shown at the center of each cell if > 0.
    - If given, the absorption set is highlighted by painting the corners of the corresponding cell as brown (or dark orange).

    In the Mountain Car environment, the state counts are used to find the states with some visit during policy learning so that the best action
    learned for those states is computed and plotted.

    Arguments:
    - average_reward: float
        Average reward observed by the policy.

    - state_counts_hist: array-like
        History of the state counts for each state observed throughout the policy learning process.
        It should have size "# policy learning steps" X "# states".
        For the Gridworld environment, only the state counts at the last policy learning step are used in the plot.
        In the MountainCar environment, this information is used to find what states were visited during the history so that we can show the policy for them
        but NOT for the non-visited states in order to avoid clogging.

    - absorption_set: (opt) set
        Only used in the Gridworld environment: a set of 1D or 2D indices listing the cells that belong to the absorption set.
        default: None
    """
    learning_method = params_exec['learning_method']
    learning_criterion = params_exec['learning_criterion']
    env_type = params_exec['env_type']
    N = params_exec['N']
    T = params_exec['T']
    wind_dict = params_exec['wind_dict']

    # Put the policy in Evaluation mode
    policy.getModel().eval()
    #print("Network parameters:")
    #print(list(policy.getThetaParameter()))

    colormap = cm.get_cmap("rainbow")  # useful colormaps are "jet", "rainbow", seismic"
    aspect_ratio = "auto" if env_type == Environment.MountainCar else "equal"
    factor_fontsize = 1.0   # Scaling factor when computing the final fontsize to use for labels showing the policy values of the different actions

    # Policy for each action at each state
    new_figure = False
    if env_type == Environment.MountainCar:
        # Plot the best action and their probability at each state, as an image
        # We initialize these arrays as NaN so that we can decide not to show anything for a state that hasn't been visited at all during the policy learning process (in one replication)
        best_action = np.nan*np.ones(env.getNumStates())
        highest_probability = np.nan * np.ones_like(best_action)
        states_with_at_least_one_visit_during_policy_learning = np.where( np.sum(state_counts_hist, axis=0) )[0]

        # Find the action with highest probability for each discrete state
        policy_values = policy.get_policy_values()
        for s in range(env.getNumStates()):
            if s in states_with_at_least_one_visit_during_policy_learning:
                action_highest_probability = np.argmax(policy_values[s])
                best_action[s] = action_highest_probability  #(action_highest_probability + 1) / 3.0
            highest_probability[s] = np.max(policy_values[s])
        if verbose:
            print(f"Policy values for each discrete state:\n{policy_values}")

        # Show the best action map as an image plot whose intensity is proportional to the probability of the best action, distinguishing the action by color as indicated in variable `colormaps`
        if axes is None:
            new_figure = True
            ax = plt.figure().subplots(1, 1)
        else:
            ax = axes[0]
            ax.cla()   # Clear the existing plot (if any) so that there is no overlap with what is being plotted now
        colormaps = ["Blues", "Greens", "Reds"]  # Colormaps for actions LEFT, STAY, RIGHT
        for a in np.arange(env.getNumActions()-1, -1, -1):   # We go in reverse order so that the colomaps appear from left to right to represent LEFT, ZERO, RIGHT
            msk = best_action == a
            highest_probability_toplot = np.nan*np.ones_like(highest_probability)
            highest_probability_toplot[msk] = highest_probability[msk]
            ax, img = env.plot_values(highest_probability_toplot, ax=ax, cmap=colormaps[a], vmin=0, vmax=1, add_colorbar=new_figure)
        env._finalize_plot(ax)
        axes = [ax]
        plt.subplots_adjust(top=0.80, wspace=0, hspace=0)
        plt.suptitle(f"{learning_method.upper()}, {learning_criterion.name.upper()} criterion, t_learn={t_learn}\nN={N}, T={T}"
                     f"\nPolicy at each state:\nShowing the highest probability of:\nBlue = Acc. LEFT, Green: = Do NOT acc., Red = Acc. RIGHT")
    else:
        #-- Plot suitable for Gridworlds

        # Mask for the obstacles (which should appear in gray (that's why we multiply by 0.5))
        mask_obstacles = 0.5*np.ones((3, 3))

        # Mask for the absorption set (if given)
        if absorption_set is not None:
            # Prepare a 3x3 matrix that will be used to mark the cells belonging to the absorption set
            mask_absorption_set = np.nan*np.ones((3, 3))
            mask_absorption_set[0, 0] = 1
            mask_absorption_set[0, 2] = 1
            mask_absorption_set[2, 0] = 1
            mask_absorption_set[2, 2] = 1

        if axes is None:
            new_figure = True
            axes = plt.figure().subplots(*env.getShape(), sharex=True, sharey=True, gridspec_kw=dict(hspace=0, wspace=0))  # See also help(plt.subplots); help(matplotlib.gridspec.GridSpec). In principle the same result should be obtained with `plt.subplots_adjust(wspace=0, hspace=0)` after the plot generation process.
            # Adjust the figure size to the same shape as the environment (so that we may achieve NO spaces between cells)
            shape_ratio = axes.shape[1] / axes.shape[0]  # width over height of the environment
            h = plt.gcf().get_figheight()
            w = h * shape_ratio
            plt.gcf().set_figheight(h)
            plt.gcf().set_figwidth(w)

        # Action probabilities to show as image in EACH cell
        proba_actions_toplot = np.nan*np.ones((3, 3))
        if is_problem_2d:
            # Factor for the fontsize that depends on the environment size
            factor_fs = factor_fontsize * np.min((4 / axes.shape[0], 4 / axes.shape[1]))
            for i in range(axes.shape[0]):
                for j in range(axes.shape[1]):
                    state_1d = np.ravel_multi_index((i, j), axes.shape)
                    if verbose:
                        print("")
                    for action in range(env.getNumActions()):
                        if verbose:
                            print(f"Computing policy Pr(a={action}|s={(i,j)})...", end= " ")
                        idx_2d = (0, 1) if action == 0 else (1, 2) if action == 1 else (2, 1) if action == 2 else (1, 0)
                        proba_actions_toplot[idx_2d] = policy.getPolicyForAction(action, state_1d)
                        if verbose:
                            print("p = {:.3f}".format(proba_actions_toplot[idx_2d]))
                    # Clear the plot to avoid overlap with new plot
                    axes[i, j].cla()
                    img = axes[i, j].imshow(proba_actions_toplot, cmap=colormap, vmin=0, vmax=1) #, aspect=aspect_ratio)  # aspect="auto" means use the same aspect ratio as the axes

                    # Add the obstacles
                    if state_1d in env.getObstacleStates():
                        axes[i, j].imshow(mask_obstacles, cmap="Greys", vmin=0, vmax=1)
                    else:
                        # Add the probability values (as text) for each action
                        for action in range(env.getNumActions()):
                            idx_2d = (0, 1) if action == 0 else (1, 2) if action == 1 else (2, 1) if action == 2 else (1, 0)
                            axes[i, j].text(idx_2d[1], idx_2d[0], "{:02d}".format(int(round(proba_actions_toplot[idx_2d]*100))),
                                            color="white", fontsize=fontsize*factor_fs,
                                            horizontalalignment="center", verticalalignment="center")

                        # Add the state count at the center of each cell
                        state_counts_toplot = np.nan*np.ones_like(proba_actions_toplot)
                        state_counts_toplot[1, 1] = state_counts_hist[-1, state_1d]
                        axes[i, j].imshow(state_counts_toplot, cmap="Blues", vmin=0, vmax=np.max(state_counts_hist))
                        if state_counts_toplot[1,1] > 0:
                            axes[i, j].text(1, 1, "{:02d}".format(int(state_counts_toplot[1, 1])),
                                            color="black", fontsize=fontsize*factor_fs,
                                            horizontalalignment="center", verticalalignment="center")

                        # Mark the cells that are part of the absorption set, if given
                        if absorption_set is not None and ((i, j) in absorption_set or state_1d in absorption_set):
                            # Shade in orange the cells belonging to the absorption set by highlighting the little cells at the 4 corners
                            axes[i, j].imshow(mask_absorption_set, cmap="Oranges", vmin=0, vmax=1)

                    # Remove the axes ticks as they do not convey any information
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
        else:
            factor_fs = factor_fontsize * 4 / axes.shape[0]
            for i in range(len(axes)):
                state = i
                assert env.getNumActions() == 2, f"The 1D gridworld must have two possible actions, RIGHT (0) and LEFT (1): {env.getActionSpace()}"
                for action in range(env.getNumActions()):
                    if verbose:
                        print(f"Computing policy Pr(a={action}|s={state})...", end=" ")
                    idx_2d = (1, 2) if action == 0 else (1, 0)
                    proba_actions_toplot[idx_2d] = policy.getPolicyForAction(action, state)
                    if verbose:
                        print("p = {:.3f}".format(proba_actions_toplot[idx_2d]))
                # Clear the plot to avoid overlap with new plot
                axes[i].cla()
                img = axes[i].imshow(proba_actions_toplot, cmap=colormap, vmin=0, vmax=1, aspect="auto")  # aspect="auto" means use the same aspect ratio as the axes

                # Add the obstacles
                if state in env.getObstacleStates():
                    axes[i].imshow(mask_obstacles, cmap="Greys", vmin=0, vmax=1)
                else:
                    # Add the probability values (as text) for each action
                    for action in range(env.getNumActions()):
                        axes[i].text(0, action, "{:02d}".format(int(round(proba_actions_toplot[0, action] * 100))),
                                     color="white", fontsize=fontsize * factor_fs,
                                     horizontalalignment="center", verticalalignment="center")

                    # Add the state count at the center of each cell
                    state_counts_toplot = np.nan * np.ones_like(proba_actions_toplot)
                    state_counts_toplot[1, 1] = state_counts_hist[-1, state]
                    axes[i].imshow(state_counts_toplot, cmap="Blues", vmin=0, vmax=np.max(state_counts_hist))
                    if state_counts_toplot[1, 1] > 0:
                        axes[i].text(1, 1, "{:02d}".format(int(state_counts_toplot[1, 1])),
                                        color="black", fontsize=fontsize*factor_fs,
                                        horizontalalignment="center", verticalalignment="center")

                    # Mark the cells that are part of the absorption set, if given
                    if absorption_set is not None and state in absorption_set:
                        # Shade in orange the cells belonging to the absorption set by highlighting the little cells at the 4 corners
                        axes[i].imshow(mask_absorption_set, cmap="Oranges", vmin=0, vmax=1)

                # Remove the axes ticks as they do not convey any information
                axes[i].set_xticks([])
                axes[i].set_yticks([])

        if False and new_figure:
            # DM-2025/06/22: This section was disabled because:
            # - the request of not having any space among subplots is already satisfied at the creation of the figure where we use the gridspec_kw argument
            # - adding the colorbar to the plot makes that adjustment of "no space among cells" (cell = subplot) be broken
            # (i.e. space among subplots is added back because the colorbar takes space from the axes where the last image was plotted,
            # and I haven't found a way to avoid this "take space" action... perhaps we should explicit state that we want to leave space for a colorbar
            # when we CREATE the image... but haven't yet investigated this)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.colorbar(img, ax=axes)  # This adds a colorbar to the right of the FIGURE. However, the mapping from colors to values is taken from the last generated image! (which is ok because all images have the same range of values.
                                        # Otherwise see answer by user10121139 in https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
        plt.suptitle(f"{learning_method.upper()}, {learning_criterion.name.upper()} criterion, t_learn={t_learn}\nN={N}, T={T}, wind_dict={wind_dict}, KL = {KL_distance if KL_distance is not None else np.nan:.4f}, KL_norm = {KL_distance_norm if KL_distance_norm is not None else np.nan:.6f}"
                     f"\nPolicy at each state (avg. # steps from S to F = {1/average_reward:.0f})")

    plt.pause(0.01)
    plt.draw()

    return axes
#--- Plotting functions
#-------------------- AUXILIARY AND PLOTTING FUNCTIONS ------------------#


#----------------- BASIC SETUP AND SIMULATION PARAMETERS --------------#
# Learning task and learning criterion are used by the constructor of the test class below
learning_task = LearningTask.CONTINUING
#learning_task = LearningTask.EPISODIC

learning_criterion = LearningCriterion.AVERAGE; gamma = 1.0    # gamma could be < 1 in the average reward criterion in order to take the limit as gamma -> 1 as presented in Sutton, pag. 251/252.
#learning_criterion = LearningCriterion.DISCOUNTED; gamma = 0.9

seed = 1317
env_type = Environment.Gridworld
#env_type = Environment.MountainCar
env_type_name = env_type.name   # The environment NAME is retrieved to avoid an error that happened at least once (Jun-2025) when saving results to a pickle file: "Can't pickle <enum 'Environment'>: attribute lookup Environment on __main__ failed"
problem_2d = True
use_random_obstacles_set = True; prop_obstacles = 0.4; #0.5;
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
        size_vertical = 8; size_horizontal = 12
        #size_vertical = 9; size_horizontal = 13
        #size_vertical = 10; size_horizontal = 14
        size_vertical = 10; size_horizontal = 30

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
nn_hidden_layer_sizes = [] #[12]
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
                       N=N,
                       T=T,
                       estimate_absorption_set=estimate_absorption_set, threshold_absorption_set=threshold_absorption_set, absorption_set=default_absorption_set,
                       states_of_interest_fv=set({exit_state}),    #None
                       seed=seed, plot=True, debug=False,
                       seed_obstacles=seed_obstacles)
    test_ac.setUp()
    print(test_ac.policy_nn.nn_model)
elif env_type == Environment.MountainCar:
    N = 30  #50
    T = 300 #100 #300 #500
    env_discrete = True #False
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
                       nx=40,       # Number of points in the discretization of the positions (only used in the continuous-state-dynamic Mountain Car, in which case the "factor for force" parameter is not used)
                       nv=21,       # Number of points in the discretization of the velocities
                       factor_for_force_and_gravity=10 if not env_discrete else 90, #100, #90, #20, #15,   # Factor controlling the number of discrete positions in the discretized problem --> NOTE: Using `1` is TOO SMALL! (as there are too many points in the grid)
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
dict_time_elapsed = dict()
dict_time_cpu = dict()
#------------------ RESULTS COLLECTION AND PLOTS SETUP ----------------#


# Number of replications to run on each method
nrep = 5 #9
# Logging
log = nrep > 1  #learning_method_type == "values_fv"

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

learning_method_type = learning_method[:9]  # This makes e.g. "values_fvos" become "values_fv"

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
M2 = max_time_steps_fv_overall = 2*max_time_steps_fv_for_all_particles if stopping_criterion_fv.value in [2, 3] else max_time_steps_fv_for_all_particles #2*max_time_steps_fv_for_all_particles
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
max_time_steps_benchmark = max_time_steps_fv_for_expectation + max_time_steps_fv_overall
print(f"max_time_steps_benchmark (T + M2) = {max_time_steps_benchmark}")

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
is_NPG = len(nn_hidden_layer_sizes) == 0
#*********************
n_learning_steps = 150 #50 #100 #30 #200 #50 #100
#*********************
n_episodes_per_learning_step = 50 #100 #30  # Number of episodes for the policy update step when learning the policy online and in NON-NPG mode
# Max time steps per episode during exploration for the online policy learning
# In the Mountain Car problem we limit the number of steps per episode in the continuous-dynamics case because I've seen out-of-memory problems otherwise.
if env_type == Environment.Gridworld:
    _multiplier = 5
else:
    _multiplier = 1
max_time_steps_per_policy_learning_episode = _multiplier*test_ac.getEnv().getNumStates() if problem_2d else 2*test_ac.getEnv().getNumStates() #np.prod(env_shape) * 10 #max_time_steps_benchmark // n_episodes_per_learning_step   # Maximum number of steps per episode while LEARNING THE *POLICY* ONLINE (NOT used for the value functions (critic) learning)
allow_deterministic_policy = True #False
# (2025/06/22) IMPORTANT NOTE ABOUT USING THE ADVANTAGE FUNCTION TO UPDATE THE POLICY:
# It was observed in the labyrinth problem that the learning curves (average reward) are more UNSTABLE when learning using the advantage A(s,a) than the the action function Q(s,a)
# The instability in principle can be reduced by reducing the learning rate for the actor (optimizer_learning_rate) from e.g. 10.0 to 1.0, but not really sure about its effect.
use_advantage = not (learning_method == "values_fvos") # Set this to True if we want to use the advantage function learned as the TD error, instead of using the advantage function as the difference between the estimated Q(s,a) and the estimated V(s) (where the average reward cancels out)
optimizer_learning_rate = 1.0 if is_NPG and use_advantage else 10.0 if is_NPG and not use_advantage else 0.05 #if policy_learning_mode == "online" else 0.05 #0.01 #0.1
adjust_optimizer_learning_rate = True; t_learn_min_to_adjust_optimizer_learning_rate = 1
reset_value_functions_at_every_learning_step = False #(learning_method == "values_fv")     # Reset the value functions when learning with FV, o.w. the learning can become too unstable due to the oversampling of the states with high value... (or something like that)

# 2) Parameters about VALUE FUNCTION learning (Critic)
alpha_initial = simulator_value_functions.getAgent().getLearner().getInitialLearningRate()      # NOTE: alpha_initial is NOT used when learning the value functions by function approximation, as this is set by the default learning rate of the Adam optimizer
adjust_alpha_initial_by_learning_step = False; t_learn_min_to_adjust_alpha = 30 # based at 1 (regardless of the base value used for t_learn)
#max_time_steps_per_episode = test_ac.getEnv().getNumStates()*10  # (2024/05/02) NO LONGER USED!  # This parameter is just set as a SAFEGUARD against being blocked in an episode at some state of which the agent could be liberated by restarting to a new episode (when this max number of steps is reached)
epsilon_random_action = 0.1 #if policy_learning_mode == "online" else 0.0 #0.1 #0.05 #0.0 #0.01
reward_to_promote_exploration = 0.0 #1.0 #0.1 #None   # Reward for a reward shaping strategy used to promote the visit of EXIT events from A which allow the execution of the FV simulation to estimate value functions (which is crucial for the FV estimation procedure to be effective). Note that the shaped reward may be proportional to |V(s)|, not necessarily constant
use_average_max_time_steps_in_td_learner = True #learning_method == "values_td2" #True #False
use_average_reward_from_previous_step = True #learning_method_type == "values_fv" #False #True            # Under the AVERAGE reward crtierion, whether to use the average reward estimated from the previous policy learning step as correction of the value functions (whenever it is not 0), at least as an initial estimate
use_fixed_average_reward = True  # (2025/06/26) THIS SEEMS TO BE VERY IMPORTANT IN GUARANTEEING STABILITY OF FVAC LEARNING (specially in situations where the absorption set A may have states close to the finish line in the labyrinth --e.g. 6x8 RANDOM labyrinth with WIND=0.6 (seed_labyrinth = 4217, seed (simulation) = 1317)
keep_fv_estimation_of_average_reward_and_stationary_probability_consistent = False  #True   # Use `False` when we are only interested in leveraging the reward information for policy learning as opposed to consistency of the estimation of the average reward
learning_steps_observe = [4, 5, 6, 7] #7, 8, 19, 20] #[1, 2, 7, 8, 22, 23, 24] #[50, 90] #[2, 30, 48] #[2, 10, 11, 30, 31, 49, 50] #[7, 20, 30, 40]  # base at 1, regardless of the base value used for t_learn
verbose_period = max_time_steps_fv_for_all_particles // 10
plot = False         # Whether to plot the evolution of the state value function and average reward estimation
plot_policy_update = False  # Whether to plot the policy after each policy learning step update
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

# Define the BASE seed for the simulation
seed_base = test_ac.seed

# Store the execution parameters in a dictionary
params_exec = dict([(k, eval(k)) for k in [ # --- Environment
                                            'env_type',
                                            'env_type_name',    # We also store the env_type_name because of errors generated when saving to pickle (can't pickle Environment enum)
                                            'env_shape',
                                            'entry_state',
                                            'exit_state',
                                            'wind_dict',
                                            # --- Learning
                                            'learning_method',
                                            'learning_task',
                                            'learning_criterion',
                                            # --- Absorption set
                                            'estimate_absorption_set_at_every_step',
                                            'update_absorption_set_with_fv_visits',
                                            'soft_killing',
                                            # --- Actor
                                            'policy_learning_mode',
                                            'is_NPG',
                                            'n_learning_steps',
                                            'n_episodes_per_learning_step',
                                            'max_time_steps_per_policy_learning_episode',
                                            'allow_deterministic_policy',
                                            'use_advantage',
                                            'optimizer_learning_rate',
                                            'adjust_optimizer_learning_rate',
                                            'reset_value_functions_at_every_learning_step',
                                            # --- Critic
                                            'N',
                                            'T',
                                            'max_time_steps_benchmark',
                                            'alpha_initial',
                                            'adjust_alpha_initial_by_learning_step',
                                            'epsilon_random_action',
                                            'use_average_max_time_steps_in_td_learner',
                                            'use_average_reward_from_previous_step',
                                            'use_fixed_average_reward',
                                            'keep_fv_estimation_of_average_reward_and_stationary_probability_consistent',
                                            'learning_steps_observe',
                                            'verbose_period',
                                            'plot',
                                            'colormap',
                                            # -- Simulation
                                            'seed_base',
                                            ]])
print("\nExecution parameters:")
for param, value in params_exec.items():
    print(f"{param}: {value}")

# Initialize objects that will contain the results by learning step
state_counts_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates()), dtype=int)
V_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates()))
Q_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions()))
A_all = np.zeros((nrep, n_learning_steps, test_ac.getEnv().getNumStates(), test_ac.getEnv().getNumActions()))
R_all = np.nan * np.ones((nrep, n_learning_steps))       # Average reward (corresponding to EPISODIC learning tasks)
R_long_all = np.nan * np.ones((nrep, n_learning_steps))  # Long-run average reward (corresponding to CONTINUING learning tasks). It does NOT converge to the same value as the episodic average reward because there is one more reward value per episode!! (namely the reward going from the terminal state to the start state)
R_long_initial_all = np.zeros((nrep, n_learning_steps))  # Useful for FV only: average reward observed during the initial simulation (useful for ablation study of FV)
R_long_fv_inflated_all = np.nan * np.ones((nrep, n_learning_steps))  # Useful for FV only: compare the average reward inflated by the FV exploration and a sensible estimate of the average reward by FV
R_long_true_all = np.nan * np.ones((nrep, n_learning_steps))    # True Long-run Average reward (CONTINUING learning task) under the policy at the start of each policy learning step
loss_all = np.nan * np.ones((nrep, n_learning_steps))
nsteps_all = np.nan * np.ones((nrep, n_learning_steps), dtype=int)  # Number of value function time steps run per every policy learning step
KL_all = np.nan * np.ones((nrep, n_learning_steps))             # K-L divergence between two consecutive policies
KL_norm_all = np.nan * np.ones((nrep, n_learning_steps))        # NORMALIZED-by-#known-states K-L divergence between two consecutive policies
alpha_all = alpha_initial * np.ones((nrep, n_learning_steps))   # Initial alpha used at each policy learning step
LR_all = optimizer_learning_rate * np.ones((nrep, n_learning_steps))   # Optimizer learning rate at each policy learning step
time_elapsed_all = np.nan * np.ones(nrep)   # Execution time for each replication
time_cpu_all = np.nan * np.ones(nrep)    # CPU time for each replication

# Prepare the Actor learner
if learning_method_type == "values_fv":
    # Define the object where we will store the number of steps used by the FV learner of value functions at each policy learning step
    # so that we use the same number for the TD learner of value functions when the TDAC policy learning process is run afterwards.
    # I.e. this assumes that FVAC is run BEFORE TDAC!
    max_time_steps_benchmark_all = np.nan * np.ones((nrep, n_learning_steps))
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
cpu_start = process_time()
dt_start_filename = get_current_datetime_as_string(format="filename")

for rep in range(nrep):
    seed_rep = seed_base*(rep + 1)
    # Use the following IF to run just the LAST replication, in case we need to compare it with another LAST replication from a set of experiments previously run.
    #if rep < nrep - 1:
    #    print(f"Replication {rep} skipped!")
    #    continue
    print(f"\n->>>>>>>>>>>\nRunning replication {rep+1} of {nrep} (seed={seed_rep})... @{format(get_current_datetime_as_string())}")

    # Reset the policy actor every time a new replication starts
    # Note that this performs a further reset of the policy (besides the one done above by the constructor that creates the learner_ac object),
    # and thus resets the policy to ANOTHER slightly different policy (because of the normally distributed random values around 0 that are set as neural network weights)
    # Note also that the critic learner will be reset by the critic learner just before starting with the simulation
    # (see discrete.Simulator._run_fv() method for FV and discrete.Simulator._run_single_continuing_task() for TD)
    print("Resetting the policy learner and the critic (if any)...")
    learner_ac.reset(reset_value_functions=True, reset_policy=True, initial_policy=initial_policy)
    print(f"Resetting the optimizer learning rate to {optimizer_learning_rate}...")
    learner_ac.setOptimizerLearningRate(optimizer_learning_rate)
    if not test_ac.getEnv().isStateContinuous():
        print(f"Initial policy (states x actions):\n{learner_ac.getPolicy().get_policy_values()}")

    if simulator_value_functions is not None:
        # Whenever there is a critic, do a couple of resets
        # (recall that no critic is defined for the ALL-online learning method)
        _learner_value_functions_for_critic = simulator_value_functions.getAgent().getLearner()

        # "[OUT]" means "out of the main code", i.e. at the caller of the policy learning process run by simulator_value_functions.run() and learner_ac.learn()
        print(f"[OUT] The average reward stored in learner before starting a new replication and resetting the average reward is: {_learner_value_functions_for_critic.getAverageReward()}")

        # Reset the initial alpha of the TD learner (just in case)
        _learner_value_functions_for_critic.setInitialLearningRate(alpha_initial)

    time_start_rep = timer()
    cpu_start_rep = process_time()
    if learning_method == "all_online":
        for t_learn in range(n_learning_steps):
            print(f"\n\n*** Running learning step {t_learn+1} of {n_learning_steps} (AVERAGE REWARD at previous step (not reward-shaped) = {R_all[rep, max(0, t_learn-1)]}, {1/R_all[rep, max(0, t_learn-1)]:.0f} average steps) of "
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
        # Keep track of the number of learning steps in which we observe a significant change in the policy so that we can reduce the optimizer learning rate
        # when NPG is used and thus avoid very large policy updates when some learning has already happened.
        n_learning_steps_with_large_enough_KL = 0
        if plot_policy_update:
            # Initialize the plot of the policy at each policy learning step
            axes_policy = plot_policy(test_ac.getEnv(), learner_ac.getPolicy(), np.nan, state_counts_all[rep, :, :], params_exec, is_problem_2d=problem_2d,
                                      absorption_set=simulator_value_functions.getAgent().getLearner().getAbsorptionSet() if learning_method_type == "values_fv" else None,
                                      t_learn=1, verbose=False)

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
                print(f"\n\n*** Running learning step {t_learn+1} of {n_learning_steps} (True average reward under current policy = {avg_reward_true}) "
                      f"(AVERAGE REWARD at previous step (not reward-shaped) = {R_all[rep, max(0, t_learn-1)]} of MAX={max_avg_reward_episodic}, {1/R_all[rep, max(0, t_learn-1)]:.0f} average steps) "
                      f"(AVERAGE REWARD STORED In learner = {simulator_value_functions.getAgent().getLearner().getAverageReward()})... (seed={seed_learn}) @{get_current_datetime_as_string()}")
            else:
                print(f"\n\n*** Running learning step {t_learn+1} of {n_learning_steps} "
                      f"(AVERAGE REWARD at previous step (not reward-shaped) = {R_all[rep, max(0, t_learn-1)]} of MAX={max_avg_reward_episodic}, {1/R_all[rep, max(0, t_learn-1)]:.0f} average steps) "
                      f"(AVERAGE REWARD STORED In learner = {simulator_value_functions.getAgent().getLearner().getAverageReward()})... (seed={seed_learn}) @{get_current_datetime_as_string()}")
            time.sleep(1)   # Wait for a second so that I can easily read the learning step number

            # ALWAYS RESET THE VALUE FUNCTIONS WHEN IT'S THE VERY FIRST LEARNING STEP (because we don't want to keep history from a earlier learning process on the same learner!)
            reset_value_functions_at_this_step = reset_value_functions_at_every_learning_step if t_learn > 0 else True
            # Update the initial learning rate for the value functions at each learning step to a smaller value than the previous learning step
            # SHOULD WE SET IT TO THE AVERAGE LEARNING RATE FROM THE PREVIOUS LEARNING STEP?? (so that we start off where we left at the last learning moment)
            alpha_initial_at_current_learning_step = alpha_initial / (t_learn + 1)

            #-- Optionally adjust the initial learning rate alpha
            # Compute the K-L divergence of the current policy w.r.t. the previous policy to decide whether to update the initial learning rate or not
            # If the new policy is too different from the previous policy we should NOT update the learning rate because the value functions learned
            # under the previous policy is most likely far away from the true values functions under the new policy.
            # Note: rel_entr(p, q) computes the term p * log(p/q) contributing to the relative entropy or Kullback-Leibler divergence
            # of the "new" distribution p(x) w.r.t. the "old" distribution q(x), which is defined as the expectation (measured over p(x)) of the
            # difference between log(p(x)) and log(q(x), i.e. the p-expected difference in information provided by the new policy p(x) w.r.t. the previous policy q(x)).
            # Hence if p(x) is larger than q(x) the contribution is positive and if it is smaller, the contribution is negative.
            # Therefore a positive K-L divergence typically corresponds to an increase in the probability, from q(x) to p(x), at states x having a larger "new" probability p(x).
            # Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html
            # Note that we could also use the function scipy.stats.entropy() to compute directly the K-L divergence (i.e. without summing over the rel_entr() values, as done here)
            if not test_ac.getEnv().isStateContinuous():
                policy = learner_ac.getPolicy().get_policy_values()
                KL_distance = np.sum(rel_entr(policy, policy_prev)) if t_learn > 0 else 0.0
                KL_distance_norm, _num_states_known_by_agent = compute_normalized_KL_distance(KL_distance, simulator_value_functions.getAgent().getLearner())
                print(f"K-L distance with previous policy: {KL_distance:.4f}, standardized by the # states known by the agent ({_num_states_known_by_agent}): {KL_distance_norm:.6f})")
                KL_all[rep, t_learn] = KL_distance
                KL_norm_all[rep, t_learn] = KL_distance_norm
                if KL_distance_norm > KL_THRESHOLD:
                    n_learning_steps_with_large_enough_KL += 1
                policy_prev = policy.copy()

            if adjust_alpha_initial_by_learning_step:
                # Update the initial learning rate alpha (for VALUE FUNCTIONS, not for the policy) ONLY when:
                # - the learning step is larger than or equal to a minimum.
                # - the policy did NOT change significantly from the previous learning step
                #   (because in that case we can consider the current estimates of the value functions to be associated to the NEW policy as well,
                #    since the policy didn't change much).
                if t_learn + 1 >= t_learn_min_to_adjust_alpha:
                    if KL_distance_norm > KL_THRESHOLD:
                        simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial)
                    else:
                        simulator_value_functions.getAgent().getLearner().setInitialLearningRate(alpha_initial / 10)
                alpha_all[rep, t_learn] = simulator_value_functions.getAgent().getLearner().getInitialLearningRate()
            print(f"*** INITIAL learning rate for VALUE FUNCTIONS, alpha = {simulator_value_functions.getAgent().getLearner().getInitialLearningRate()}", end=" ")
            print(f"(adjustment happens starting at learning step >= {t_learn_min_to_adjust_alpha} AND if K-L change in policy is AT MOST {KL_THRESHOLD}) ***" if adjust_alpha_initial_by_learning_step else "***")

            if adjust_optimizer_learning_rate:
                # Update the optimizer learning rate (for the POLICY) ONLY when:
                # - the learning step is larger than or equal to a minimum.
                # - the policy CHANGED significantly from the previous learning step
                #   (i.e., if no significant learning happened, it doesn't make too much sense to reduce the policy learning rate).
                # The adjustment is done by the number of learning steps where the policy has changed significantly so far, instead of e.g. using t_learn directly,
                # with the objective of avoiding a too agressive reduction of the learning rate which could temper fast enough learning of the policy.
                if t_learn + 1 >= t_learn_min_to_adjust_optimizer_learning_rate:
                    if not test_ac.getEnv().isStateContinuous():
                        if KL_distance_norm > KL_THRESHOLD:
                            LR_all[rep, t_learn] = optimizer_learning_rate / n_learning_steps_with_large_enough_KL
                            learner_ac.setOptimizerLearningRate(LR_all[rep, t_learn])
                        else:
                            LR_all[rep, t_learn] = learner_ac.getOptimizerLearningRate()
                    else:
                        # Update the optimizer learning step by the number of learning steps used so far
                        LR_all[rep, t_learn] = optimizer_learning_rate / (t_learn + 1)
                        learner_ac.setOptimizerLearningRate(LR_all[rep, t_learn])
            print(f"*** Optimizer learning rate for the POLICY, lr = {learner_ac.getOptimizerLearningRate()}", end=" ")
            print(f"(adjustment happens starting at learning step >= {t_learn_min_to_adjust_optimizer_learning_rate} AND if standardized K-L change in policy "
                  f"({KL_distance_norm:.4f}) is LARGER THAN {KL_THRESHOLD}) ***" if adjust_optimizer_learning_rate else "***")

            #--- 1) CRITIC
            print(f"Learning the CRITIC (at the current policy) using {learning_method.upper()}...")
            # Learn the value functions using the FV simulator
            if learning_method_type == "values_fv":
                if plot and t_learn+1 in learning_steps_observe and t_learn+1 != learning_steps_observe[0]:
                    _close_plots = input("Before proceeding, would you like to close all plots generated for all learning steps observed so far (y/n)?")
                    if _close_plots.upper() == "Y":
                        for _fig_number in plt.get_fignums():
                            # Do NOT close the first 2 figures that are created at the beginning of the process showing the absorption set and initial trajectory
                            plt.close(_fig_number) if _fig_number > 2 else None
                if False:
                    # DM-2025/05/29: REMOVED THIS STEP BECAUSE THIS IS NOW FULLY DONE BY THE discrete.Simulator class WHEN (i) calling the learner.reset() method with the reset_auxiliary_info=True argument, and when ALWAYS estimating the absorption set at the first learning step, i.e. when t_learn = 0.
                    # Reset to None the start state distribution for the E(T_A) excursion at the very first learning step so that
                    # we do NOT carry over whatever this distribution was at the end of the previous replication or at the end of the previous execution of the learning process
                    # using this same learner.
                    if t_learn == 0:
                        simulator_value_functions.getAgent().getLearner().setProbasStationaryStartStateET(None)
                        # NEW-2024/10/23: We should estimate the absorption set at the FIRST policy learning step for EVERY replication, as estimating the absorption set is part of the FVAC process!
                        # Otherwise, if estimate_absorption_set_at_every_step = False, FVAC will be stick to use ALWAYS the same absorption set A estimated
                        # by the preparation of the simulation environment in test_optimizers_discretetime.py, and this is not completely fair... In fact, we already observed that
                        # the absorption set A may strongly depend on the seed used (e.g. Labyrinth 6x8 with 50% obstacles and obstacle seed = 4217, with WIND = 0.5, where for the
                        # initial simulation seed of 1317, the absorption set A turns out to be extremely favorable for FVAC... but only for THAT SEED!!)
                        estimate_absorption_set_at_this_step = True
                    else:
                        estimate_absorption_set_at_this_step = estimate_absorption_set_at_every_step
                V, Q, A, state_counts, state_counts_et, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, n_events_a, n_events_et, n_events_fv = \
                    simulator_value_functions.run(t_learn=t_learn,
                                                  max_time_steps=max_time_steps_fv_overall,
                                                  max_time_steps_for_absorbed_particles_check=max_time_steps_fv_for_all_particles,
                                                  min_prop_absorbed_particles=min_prop_absorbed_particles, stopping_criterion_fv=stopping_criterion_fv,
                                                  min_num_cycles_for_expectations=None,  # When None, the minimum number of cycles for the estimation of E(T_A) is set by the global variable MIN_NUM_CYCLES_FOR_EXPECTATIONS
                                                  estimate_absorption_set=estimate_absorption_set_at_every_step, update_absorption_set_with_fv_visits=update_absorption_set_with_fv_visits, threshold_absorption_set=threshold_absorption_set,
                                                  soft_killing=soft_killing,
                                                  use_average_reward_stored_in_learner=use_average_reward_from_previous_step,
                                                  use_fixed_average_reward=use_fixed_average_reward,
                                                  keep_fv_estimation_of_average_reward_and_stationary_probability_consistent=keep_fv_estimation_of_average_reward_and_stationary_probability_consistent,
                                                  reset_value_functions=reset_value_functions_at_this_step,
                                                  plot=plot if t_learn+1 in learning_steps_observe else False, colormap=colormap,
                                                  epsilon_random_action=epsilon_random_action,
                                                  reward_for_exit_states=reward_to_promote_exploration if reward_to_promote_exploration != 0 else None,

                                                  # DM-2025/01: This is used when performing reward shaping, in order to update the policy after the initial exploration has finished (so that reward shaping has actually an effect on the policy, which is the goal of doing reward shaping!)
                                                  learner_policy=learner_ac,

                                                  seed=seed_learn, verbose=False, verbose_period=verbose_period)
                average_reward_initial_exploration = simulator_value_functions.getAgent().getLearner().getAverageRewardInitialExploration()
                average_reward_fv_inflated = simulator_value_functions.getAgent().getLearner().getAverageRewardRaw()
                average_reward_from_critic_estimation = expected_reward    # Note: this is the same information stored in the FV learner, i.e. it would also be returned by calling simulator_value_functions.getAgent().getLearner().getAverageReward()
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
                if learning_task == LearningTask.EPISODIC:
                    V, Q, A, state_counts, _, _, learning_info = \
                        simulator_value_functions.run(nepisodes=n_episodes_per_learning_step,
                                                      t_learn=t_learn,
                                                      max_time_steps=max_time_steps_benchmark,
                                                      max_time_steps_per_episode=max_time_steps_per_policy_learning_episode,  # max_time_steps_benchmark // n_episodes_per_learning_step,
                                                      reset_value_functions=reset_value_functions_at_this_step,
                                                      seed=seed_learn,
                                                      state_observe=state_observe,
                                                      epsilon_random_action=epsilon_random_action,
                                                      compute_rmse=plot if t_learn+1 in learning_steps_observe else False,
                                                      plot=plot if t_learn+1 in learning_steps_observe else False, colormap=colormap,
                                                      verbose=True, verbose_period=verbose_period)
                else:
                    V, Q, A, state_counts, _, _, learning_info = \
                        simulator_value_functions.run(t_learn=t_learn,
                                                      max_time_steps=_max_time_steps,
                                                      estimated_average_reward=simulator_value_functions.getAgent().getLearner().getAverageReward() if use_average_reward_from_previous_step else 0.0,
                                                      use_fixed_average_reward=use_fixed_average_reward,
                                                      reset_value_functions=reset_value_functions_at_this_step,
                                                      seed=seed_learn,
                                                      state_observe=state_observe,
                                                      epsilon_random_action=epsilon_random_action,
                                                      compute_rmse=plot if t_learn+1 in learning_steps_observe else False,
                                                      plot=plot if t_learn+1 in learning_steps_observe else False, colormap=colormap,
                                                      verbose=True, verbose_period=verbose_period)
                average_reward_from_critic_estimation = simulator_value_functions.getAgent().getLearner().getAverageReward()
                nsteps_all[rep, t_learn] = learning_info['nsteps']

            print(f"Learning step #{t_learn+1}: Learning of value functions COMPLETED using {learning_method} method on {nsteps_all[rep, t_learn]} time steps")
            print(f"Estimated average reward by Critic learning process: {average_reward_from_critic_estimation}")
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
                if use_advantage:
                    print("\n(Natural Policy Gradient learning using the direct estimation of the advantage function)")
                    learner_ac.learn_natural(A)
                else:
                    if True:
                        # DM-2025/06/21: Learning using the advantage function computed as Q - E(Q over actions) proved catastrophic.
                        # Additionally, using Q(s,a) to learn instead of the advantage function A(s,a) estimated by the learner seems much more stable with smoother learning curves.
                        # Context: Labyrinth 6x8 without wind, after extensive tests and analysis of A(s,a), Q(s,a) and average reward learning curves conducted today.
                        learner_ac.learn_natural(Q)
                    else:
                        # Compute the advantage function as Q - V, where V = policy-weighted-average(Q) over all actions, and then update the policy using that advantage function
                        # Under the AVERAGE reward learning criterion, this makes the average reward (present in both V and Q), cancel out, thus making its estimation useless.
                        # In the FV approach, this is tantamount to leveraging Fleming-Viot JUST as an OVERSAMPLING mechanism, not as an estimator of the long-run expected reward.
                        # However, it has been shown by different sources (see e.g. the 2021 paper by K. Ross and Zhang on the average reward actor-critic learning:
                        # "On-Policy Deep RL for the Average-Reward Criterion" or the lecture by David Silver available in YouTube where he talks about this at one point
                        # saying that using a direct estimate of the Advantage function is better than using Q - V).
                        print("\n(Natural Policy Gradient learning using an INDIRECT estimation of the advantage function as Q(s,a) - policy-weighted-avg(Q)-over-all-actions(s))")
                        _A_as_Q_minus_avgQ = np.zeros_like(A)
                        for s in range(test_ac.getEnv().getNumStates()):
                            _state_value = 0.0
                            for a in range(test_ac.getEnv().getNumActions()):
                                # NOTE: This assumes that the order of the states and actions in the 1D arrays containing Q and A is grouped by state,
                                # i.e. for state 0, the value of all actions are given, then the same for state 1 and so forth.
                                _ind2update = s*test_ac.getEnv().getNumActions() + a
                                _state_value += learner_ac.getPolicy().getPolicyForAction(a, s) * Q[_ind2update]
                                _A_as_Q_minus_avgQ[_ind2update] = Q[_ind2update] - _state_value
                        learner_ac.learn_natural(_A_as_Q_minus_avgQ)
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
                                                              expected_reward=average_reward_from_critic_estimation)
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
                    R_all[rep, t_learn] = average_reward_from_critic_estimation # ...although here we should store the EPISODIC average reward
                                                                                # (regardless of the learning task type --CONTINUING or EPISODIC), and when the learning task is
                                                                                # CONTINUING, this average_reward_from_critic_estimation is the continuing average reward...
                                                                                # In any case, at this point we don't have easy access to the episodic average reward that we could
                                                                                # use to store here.
                                                                                # NOTE THAT THIS IS NOT THE NPG case: the NPG case is the ABOVE block, before the ELSE that defines this block.
                    _dict_numpy_options = set_numpy_options()
                    print(f"True stationary probabilities:\n{mu.reshape(env_shape)}")
                    print(f"Estimated stationary probabilities:\n{prob_states.reshape(env_shape)}")
                    reset_numpy_options(_dict_numpy_options)
                    if False and (average_reward_from_critic_estimation != 0.0 or t_learn+1 in learning_steps_observe):
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
                        ax_true.set_title("True stationary probability")
                        ax_est.set_title("Estimated stationary probability (based on state counts)")
                        ax_diff.set_title("Difference: estimated - true")
                        plt.suptitle("Comparison between the true and estimated stationary probabilities", fontsize=24)
                        plt.pause(0.1)
                        plt.draw()
                        input("Press ENTER to continue...")

            # Collect a trajectory for plotting purposes (i.e. without learning)
            # Note that this trajectory is also used to compute the episodic average reward (stored in the R_all object) when using NPG or learning the policy offline
            # (in which case no simulation is run under the current policy as is the case in the ONLINE non-NPG learning approach)
            try:
                # 13-Jan-2025: We `try` this deepcopy() because today I got the following NotImplementedError when copying the simulator object when `plot=True`:
                # "File "C:\ProgramData\Anaconda\Anaconda3-v5.1.0-201802\lib\site-packages\matplotlib\transforms.py", line 127,
                # in __copy__ TransformNode instances can not be copied. "
                # "NotImplementedError: TransformNode instances can not be copied. Consider using frozen() instead."
                _simulator = copy.deepcopy(simulator_value_functions)   # We create a copy because we don't want to change the learner object in the simulator eventually used above (e.g. the state counts that are plotted below)
            except:
                print("WARNING: The `simulator_value_functions` object could NOT be DEEPCOPied. "
                      "This means that the online exploration after the value functions have been learned will override the trajectory stored in the value function's learner. "
                      "This might affect trajectory plots which might show unexpected results.")
                _simulator = simulator_value_functions
            # Generate a trajectory for the current policy, so that we always have a trajectory to plot,
            # regardless of whether we learn the policy via NPG (in which case no trajectory is generated to learn the policy)
            # or whether we learn the policy via regular policy gradient (in which case a trajectory is generated when updating the policy by the learner_ac.learn() call above).
            # Note that we use 1000 time steps, regardless of the value of T above (used for the estimation of E(T_A) in the FVAC learning case). We do so in order to get
            # a reasonable estimation of the average reward, because the value of parameter T may be too small (e.g. T = 100).
            _learner_current_policy, _nsteps, _average_reward = _simulator.run_exploration(t_learn=t_learn, max_time_steps=1000, epsilon_random_action=0.0, seed=seed_learn + 171317, verbose=False, verbose_period=verbose_period)
            trajectory_under_policy = np.array(_learner_current_policy.getStates())
            if is_NPG or policy_learning_mode != "online":
                # Note that the average reward is NOT estimated by the run_exploration() method called above, therefore we compute it here from the observed rewards
                R_all[rep, t_learn] = np.mean(_average_reward)

            # Store the long-run average reward estimated by the value functions learner used above
            R_long_all[rep, t_learn] = average_reward_from_critic_estimation

            if learning_method_type == "values_fv":
                # Store auxiliary information on the average reward which can help understand the usefulness of the FV simulation (i.e. towards an ablation study)
                R_long_initial_all[rep, t_learn] = average_reward_initial_exploration
                # Store the inflated average reward (inflated by the FV oversampling effect) in order to analyze how sensible is the average reward estimated by FV
                R_long_fv_inflated_all[rep, t_learn] = average_reward_fv_inflated

            # Check if we need to stop learning because the average reward didn't change a bit
            if  break_when_no_change and t_learn > 0 and R_all[rep, t_learn] - R_all[rep, t_learn-1] == 0.0 or \
                break_when_goal_reached and np.isclose(R_all[rep, t_learn], max_avg_reward_episodic, rtol=0.001):
                print(f"*** Policy learning process stops at learning step t_learn+1={t_learn+1} because the average reward didn't change a bit from the previous learning step! ***")
                break

            if plot_policy_update:
                # Initialize the plot of the policy at each policy learning step
                plot_policy(test_ac.getEnv(), learner_ac.getPolicy(), R_all[rep, t_learn], state_counts_all[rep, :t_learn+1, :], params_exec, axes=axes_policy, is_problem_2d=problem_2d,
                            KL_distance=KL_all[rep, t_learn], KL_distance_norm=KL_norm_all[rep, t_learn],
                            absorption_set=simulator_value_functions.getAgent().getLearner().getAbsorptionSet() if learning_method_type == "values_fv" else None,
                            t_learn=t_learn+1, verbose=False)
    time_elapsed_rep = timer() - time_start_rep
    time_cpu_rep = process_time() - cpu_start_rep
    print(f"<<<<<<<<<< FINISHED replication {rep+1} of {nrep}... (@{format(get_current_datetime_as_string())}" + ", took {:.1f} min (CPU: {:.1f}))".format(time_elapsed_rep / 60, time_cpu_rep / 60))
    time_elapsed_all[rep] = time_elapsed_rep
    time_cpu_all[rep] = time_cpu_rep

time_end = timer()
cpu_end = process_time()
time_elapsed = time_end - time_start
time_cpu = cpu_end - cpu_start

if log:
    log_file_close(fh_log, stdout_sys, stderr_sys, dt_start)
else:
    show_elapsed_time(learning_method, time_elapsed, time_cpu)


############# Store the measures that we would like to compare
dict_test_ac[learning_method] = test_ac
dict_simulator[learning_method] = simulator_value_functions
dict_loss[learning_method] = loss_all.copy()
dict_R[learning_method] = R_all.copy()
dict_R_long[learning_method] = R_long_all.copy()
dict_R_long_initial[learning_method] = R_long_initial_all.copy()
dict_R_long_fv_inflated[learning_method] = R_long_fv_inflated_all.copy()
dict_R_long_true[learning_method] = R_long_true_all.copy()
dict_V[learning_method] = V_all.copy()
dict_Q[learning_method] = Q_all.copy()
dict_A[learning_method] = A_all.copy()
dict_state_counts[learning_method] = state_counts_all.copy()
dict_nsteps[learning_method] = nsteps_all.copy()
dict_KL[learning_method] = KL_all.copy()
dict_KL_norm[learning_method] = KL_norm_all.copy()
dict_alpha[learning_method] = alpha_all.copy()
dict_LR[learning_method] = LR_all.copy()
dict_time_elapsed[learning_method] = time_elapsed_all.copy()
dict_time_cpu[learning_method] = time_cpu_all.copy()
############# Store the measures that we would like to compare



#------------------ Plots -----------------
if not plot_policy_update:
    _average_reward_at_last_policy_learning_step = dict_R[learning_method][nrep-1][-1]
    _state_counts_during_policy_learning = dict_state_counts[learning_method][nrep-1, :, :]
    axes = plot_policy(dict_simulator[learning_method].getEnv(), dict_simulator[learning_method].getAgent().getPolicy(),
                       _average_reward_at_last_policy_learning_step,
                       _state_counts_during_policy_learning,
                       params_exec,
                       is_problem_2d=problem_2d,
                       absorption_set=simulator_value_functions.getAgent().getLearner().getAbsorptionSet() if learning_method_type == "values_fv" else None,
                       t_learn=n_learning_steps, fontsize=14, verbose=test_ac.getEnv().getNumStates() <= 50)
show_elapsed_time(learning_method, time_elapsed, time_cpu)

#raise KeyboardInterrupt


# Plot loss and average reward for the currently analyzed learner
print("\nPlotting...")
ax_loss = plt.figure(figsize=figsize).subplots(1, 1)
ax_loss.plot(np.arange(1, n_learning_steps+1), dict_loss[learning_method][rep, :n_learning_steps], marker='.', color="red")
#ax_loss.plot(range(1, n_learning_steps+1), dict_alpha[learning_method][rep, :n_learning_steps], '--', color="cyan")
ax_loss.set_xlabel("Learning step")
ax_loss.set_ylabel("Loss", color="red")
#ax_loss.axhline(0, color="red", linewidth=1, linestyle='dashed')
ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax_loss.legend(["Loss", "alpha0"], loc='upper left')
ax_R = ax_loss.twinx()
legend_R = []
if policy_learning_mode == "online":
    # We learn the policy by doing a final excursion using the current policy and computing the loss
    # => Plot the episodic average reward observed during the ONLINE Actor-Critic excursion
    ax_R.plot(np.arange(1, n_learning_steps+1), dict_R[learning_method][rep, :n_learning_steps], marker='.', color="green")
    legend_R += ["Average reward (episodic) (AFTER updating policy)"]
    if not np.isnan(max_avg_reward_episodic):
        ax_R.axhline(max_avg_reward_episodic, color="green", linewidth=1)
        legend_R += ["Max. average reward (episodic)"]
ax_R.plot(np.arange(1, n_learning_steps+1), dict_R_long[learning_method][rep, :n_learning_steps], marker='.', color="greenyellow")
legend_R += ["Long-run average reward estimated by value functions learner (BEFORE updating policy)"]
if not np.isnan(max_avg_reward_continuing):
    ax_R.axhline(max_avg_reward_continuing, color="greenyellow", linewidth=1)
    legend_R += ["Max. average reward (continuing)"]
ax_R.set_ylabel("Average reward", color="green")
# For now I have eliminated the plot of the K-L divergence between consecutive learning steps to avoid cluttering
ax_R.plot(np.arange(1, n_learning_steps+1), dict_KL_norm[learning_method][rep, :n_learning_steps], color="blue", linewidth=1)
ax_R.axhline(KL_THRESHOLD, color="blue", linestyle="dashed")
legend_R += ["Standardized K-L divergence with previous policy", "Standardized K-L threshold for reducing learning rates"]
ax_R.axhline(0, color="green", linewidth=1, linestyle='dashed')  # color="green" because it refers to the average reward which is plotted in green on the RIGHT axis
if False:  # 2025/06/25: Removed this block because we lost view of the estimated average reward by the learner which sometimes was much much larger than the episodic reward observed by the original process!
    # Set the Y-axis limits so that we can see better the learning curve despite possible large K-L values that could make the curve look very tiny
    # This is particularly useful when the policy is learned fast within each Actor-Critic excursion,
    # which for instance is achieved when computing the loss and updating parameters after each Actor-Critic episode (mini-batch)
    # or when learning the policy with NPG.
    if policy_learning_mode == "online":
        ax_R.set_ylim((-np.max(dict_R[learning_method][rep, :n_learning_steps])/50, np.max(dict_R[learning_method][rep, :n_learning_steps])*1.1))
    else:
        ax_R.set_ylim((-np.max(dict_R_long[learning_method][rep, :n_learning_steps])/50, np.max(dict_R_long[learning_method][rep, :n_learning_steps])*1.1))
ax_R.legend(legend_R, loc="upper right")
plt.title(f"{learning_method.upper()}" + f"{((' - SOFT' if soft_killing else ' - HARD') + ' killing') if learning_method_type == 'values_fv' else ''}" +
          f"\n{learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={dict_simulator[learning_method].getAgent().getLearner().gamma}) - {env_type.name} {env_shape}"
          f"\nN={N}, T={T}, wind_dict={wind_dict}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
          f"\nEvolution of the LOSS (left, red) and Average Reward (right, green) with the learning step")

#raise KeyboardInterrupt

if learning_method != "all_online": # Otherwise, `trajectory_under_policy` (used in the call below) is not defined
    plot_state_counts(dict_simulator, learning_method, params_exec, trajectory=trajectory_under_policy, plot_absorption_set=True)

if adjust_optimizer_learning_rate:
    plt.figure()
    plt.plot(np.arange(1, n_learning_steps+1), dict_LR[learning_method][rep], color="orange", marker=".")
    plt.title("Optimizer learning rate by learning step")
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
    _average_reward_at_last_policy_learning_step = dict_R[learning_method][nrep-1][-1]
    _state_counts_during_policy_learning = dict_state_counts[learning_method][nrep-1, :, :]
    axes = plot_policy(dict_simulator[learning_method].getEnv(), dict_simulator[learning_method].getAgent().getPolicy(),
                       _average_reward_at_last_policy_learning_step,
                       _state_counts_during_policy_learning,
                       params_exec,
                       is_problem_2d=problem_2d,
                       absorption_set=simulator_value_functions.getAgent().getLearner().getAbsorptionSet() if learning_method_type == "values_fv" else None,
                       t_learn=n_learning_steps, fontsize=14, verbose=test_ac.getEnv().getNumStates() <= 50)
show_elapsed_time(learning_method, time_elapsed, time_cpu)


raise KeyboardInterrupt

# Use this to generate a slow-paced trajectory drawing
test_ac.getEnv().plot_points(trajectory_under_policy, is_trajectory=True, pause=0.1)


#-- Plot the trajectory as a GIF
if env_type == Environment.MountainCar:
    # Plot the trajectory of the last replication
    # Note that we limit the trajectory to the first 100 steps because o.w. the GIF would take too long to generate...
    # For instance, with 100 steps, the GIF takes ~2 minutes to generate and is 1 MB in size already!
    _npoints2plot = min(T, 100)
    _simulator = copy.deepcopy(dict_simulator[learning_method])
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

    dict_simulator[learning_method].getEnv().plot_trajectory_gif([dict_simulator[learning_method].getEnv().getStateFromIndex(s, simulation=False) for s in trajectory2plot])


#-- Plot the value functions for the state next to the terminal state
# ONLY VALID WHEN THE EXIT STATE IS AT THE TOP RIGHT OF THE LABYRINTH
if state_observe is not None:
    marker = ''
    Q_all_baseline = dict_Q[learning_method][rep, :n_learning_steps, :, :] - np.tile(dict_V[learning_method][rep, :n_learning_steps, :].T, (dict_simulator[learning_method].getEnv().getNumActions(), 1, 1)).T
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
    plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={dict_simulator[learning_method].getAgent().getLearner().gamma}) - {env_type.name} {env_shape}"
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

common_axes = True
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
















#-- Final policy
_average_reward_at_last_policy_learning_step = dict_R[learning_method][nrep - 1][-1]
_state_counts_during_policy_learning = dict_state_counts[learning_method][nrep-1, :, :]
axes = plot_policy(dict_simulator[learning_method].getEnv(), dict_simulator[learning_method].getAgent().getPolicy(),
                   _average_reward_at_last_policy_learning_step,
                   _state_counts_during_policy_learning,
                   params_exec,
                   is_problem_2d=problem_2d, t_learn=n_learning_steps, fontsize=14, verbose=False)
show_elapsed_time(learning_method, time_elapsed, time_cpu)


plot_state_counts(dict_simulator, learning_method, params_exec, seed=seed_learn)

# Let's look at the trajectories of the learner (it works when constructing the learner with store_history_over_all_episodes=True)
#print(len(dict_simulator[learning_method].getAgent().getLearner().getStates()))
#print([len(trajectory) for trajectory in dict_simulator[learning_method].getAgent().getLearner().getStates()])


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
    print(f"Execution times for meth={meth}: total = {np.sum(dict_time_elapsed[meth])/60:.1f} min (CPU: {np.sum(dict_time_cpu[meth])/60:.1f} min),"
          f"average = {np.mean(dict_time_elapsed[meth])/60:.1f} min (CPU: {np.mean(dict_time_cpu[meth])/60:.1f} min)")

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
_learning_characteristics = f"\nN={'N' in locals() and N or _N}, " + \
                            f"T={'T' in locals() and T or _T}, " + \
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

# Plot all replications individually
ax = plt.figure(figsize=figsize).subplots(1, 1)
lines = []
legend = []
for rep in range(nrep):
    for meth in dict_R_toplot.keys():
        line = ax.plot(np.arange(1, n_learning_steps+1), dict_R_toplot[meth][rep, :n_learning_steps], '-', color=dict_colors[meth], linewidth=0.3)
        ax.text(n_learning_steps, dict_R_toplot[meth][rep, -1], f"rep={rep+1} (seed={seed_base*(rep+1)})", color=dict_colors[meth])
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
plt.suptitle(f"ALL LEARNING METHODS: {env_type_name} {env_shape} - {learning_task.name} learning task - {learning_criterion.name} reward criterion (gamma={gamma})"
             f"\nWIND: {wind_dict}, EXIT: {_exit_state_str}, ALL {nrep} replications" +
             _learning_characteristics)

# --> FOR PAPER
# Plot results on several replications
plot_for_paper = True
marker_for_mean = "" if plot_for_paper else "."
marker_for_median = "." if plot_for_paper else "x"
fontsize = 26 if plot_for_paper else 12
if nrep > 1:
    plot_mean = plot_for_paper #False #True
    plot_median = True
    plot_bands = not plot_for_paper #True #False
    dict_stats_R = dict.fromkeys(dict_loss.keys())
    ax = plt.figure(figsize=figsize).subplots(1, 1)
    lines = []
    legend = []
    _xshift = -0.1 # shift on the X axis to avoid overlap of vertical error bars
    for meth in dict_loss.keys():
        # Adapt the following filter if we want to exclude a particular method from the comparison plot
        if meth in []: #["values_fv", "values_fv2", "values_td"]:
            continue
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
            legend += [f"{dict_legends[meth]} (average +/- SE)"]
        if plot_median:
            # MEDIAN plot
            line = ax.plot(_xvalues, dict_stats_R[meth]['median'][:n_learning_steps] / max_avg_reward, color=dict_colors[meth], linestyle="dashed" if plot_mean else "solid", linewidth=2, marker=marker_for_median, markersize=12)[0]
            lines += [line]
            legend += [f"{dict_legends[meth]} (median)"]
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
    ax.legend(lines, legend, loc="upper left", fontsize=int(0.5 * fontsize))
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
        df_ratio_nsamples = pd.DataFrame({'td': np.mean(dict_nsteps['values_td'], axis=0)[:n_learning_steps],
                                          'fv': np.mean(dict_nsteps['values_fv'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'], color="blue", linewidth=0.5)
        ref_line = ax_nsamples.axhline(1.0, color="blue", linewidth=0.5, linestyle="dashed")
        legend_nsamples += ["Sample size ratio (FV/TD)", "Reference line showing equal sample size ratio"]
    if "values_tdl" in dict_nsteps.keys() and "values_fvl" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.mean(dict_nsteps['values_tdl'], axis=0)[:n_learning_steps],
                                          'fv': np.mean(dict_nsteps['values_fvl'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        ax_nsamples = ax.twinx()
        ax_nsamples.plot(range(1, n_learning_steps+1), df_ratio_nsamples['ratio_fv_td'], color="blue", linewidth=0.5)
        ref_line = ax_nsamples.axhline(1.0, color="blue", linewidth=0.5, linestyle="dashed")
        legend_nsamples += ["Sample size ratio (FVL/TDL)", "Reference line showing equal sample size ratio"]
    if "values_td" in dict_nsteps.keys() and "values_fv2" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.mean(dict_nsteps['values_td'], axis=0)[:n_learning_steps],
                                          'fv': np.mean(dict_nsteps['values_fv2'], axis=0)[:n_learning_steps]})
        df_ratio_nsamples['ratio_fv_td'] = df_ratio_nsamples['fv'] / df_ratio_nsamples['td']
        if "ax_nsamples" not in locals():
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
    if "values_td2" in dict_nsteps.keys() and "values_fv3" in dict_nsteps.keys():
        df_ratio_nsamples = pd.DataFrame({'td': np.mean(dict_nsteps['values_td2'], axis=0)[:n_learning_steps],
                                          'fv': np.mean(dict_nsteps['values_fv3'], axis=0)[:n_learning_steps]})
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
        if plot_for_paper:
            ax_nsamples.set_ylabel("Budget ratio (FVAC / TDAC)", fontsize=int(0.8*fontsize), color="blue")
            ax_nsamples.tick_params(axis='y', labelsize=int(0.8*fontsize), labelcolor="blue")
        else:
            ax_nsamples.set_ylabel("Average sample Ratio FV/TD across replications", fontsize=int(0.8 * fontsize))
            ax_nsamples.legend(legend_nsamples, loc="center left") #"lower right")
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
