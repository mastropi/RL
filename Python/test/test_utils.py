# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:16:50 2020

@author: Daniel Mastropietro
@description: Util functions used in running unit tests
"""

import copy
from enum import Enum, unique

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import cm, pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Python.lib.agents.policies import probabilistic

from Python.lib.agents.learners import LearningCriterion, LearningTask
from Python.lib.utils import computing


def array2str(x, sep=", ", fmt=":.6f"):
    "Converts an array (possibly numeric) to string separated by `sep` using the given string format"
    return "[" + sep.join( map(lambda s: ("{" + fmt + "}").format(s), x) ) + "]"


def plot_estimated_state_value_function(env, state_values, learning_criterion, state_counts=None, alphas=None, fontsize_state_counts=7):
    """
    Plots the estimated state value function for the given environment as V(s), where s is the 1D index of the environment states.
    If `env` has the true state value function stored in the object, it is also plotted (this is checked by calling `env.getV()` and checking it is not None).
    When the `learning_criterion` is the average reward, each state value function is plotted as V(s) - avg(V), since in that case the state value function is not unique.

    Optionally the state counts are also plotted as percentage and/or the alpha learning rates at the latest learning step, on the secondary axis.
    The absolute state counts are shown as labels with a fontsize specified by parameter `fontsize_state_counts`.
    """
    # Reference value for the plots, needed for the AVERAGE reward learning criterion because there is no unique solution for V(s) in that case
    ref_V_true = ref_V = 0.0
    if learning_criterion == LearningCriterion.AVERAGE:
        if env.getV() is not None:
            ref_V_true = np.nanmean(env.getV()) #env.getV()[0]
        ref_V = np.nanmean(state_values) #state_values[0]
    ax_V = plt.figure().subplots(1, 1)
    if env.getV() is not None:
        ax_V.plot(env.getAllStates(), env.getV() - ref_V_true, 'b.-')
    ax_V.plot(env.getAllStates(), state_values - ref_V, 'r.-')
    ax_V.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_V.set_xlabel("State")
    ax_V.set_ylabel(learning_criterion == LearningCriterion.AVERAGE and "V(s) - avg(V)" or "V(s)")
    ax_V.set_title(f"State value function under the {learning_criterion.name.upper()} reward criterion" + (env.getV() is not None and f": RMSE = {computing.rmse(env.getV() - ref_V_true, state_values - ref_V):.3f}" or ""))

    if state_counts is not None or alphas is not None:
        ax2 = ax_V.twinx()

    ylabel = ""
    sep = ""
    if state_counts is not None:
        # Plot the state counts on a secondary axis,
        # as a percentage so that they most likely have a very similar scale as the learning rate alphas which might also have been requested
        state_counts_rel = state_counts / np.sum(state_counts)
        ax2.bar(np.arange(env.getNumStates()), state_counts_rel, color="blue", alpha=0.3)
        for s in range(env.getNumStates()):
            ax2.text(s, state_counts_rel[s], state_counts[s], fontsize=fontsize_state_counts, horizontalalignment="center", verticalalignment="bottom")
        ylabel += "State counts"
        sep = " - "
    if alphas is not None:
        # Plot the final learning rates on a secondary axis
        ax2.plot(alphas, color="blue", linestyle="dotted")
        ax2.set_ylim((0, None))
        ylabel += sep + "Alphas at last simulation step"
    ax2.set_ylabel(ylabel)

    plt.pause(0.1)
    plt.draw()

    if "ax2" in locals():
        return ax_V, ax2
    else:
        return ax_V


def plot_rmse_by_episode(rmse_mean_values, rmse_se_values=None, min_rmse=None, max_rmse=None, kpi_name="RMSE",
                         color="black", linestyle='solid',
                         alphas=None, alpha_min=0.0, color_alphas='black', max_alpha=None, plot_scales="both", fig=None,
                         subtitle="", legend_rmse=True, legend_alpha=True, fontsize=12):
    """
    Plots the average RMSE values (or MAPE values, depending on `kpi_name`) (over states) by episode

    @param rmse_mean_values: numpy array containing the average RMSE value per episode and including the RMSE at the
    very start of the learning process, i.e. the RMSE computed on the initial guess of the value function.
    Its length should be equal to the number of episodes run + 1.
    @param rmse_se_values: (optional) numpy array containing an error measure of the average RMSE, e.g. their standard error,
    for each episode run, including at the very start of the learning process.
    @param max_mse: (optional) maximum RMSE value to show in the plot (for visual comparison reasons)

    @param alphas: list or array containing the alpha values (their average over all states if alpha depends on the state
    or is allowed to vary within an episode) used when updating the value function within each episode.
    Its length should be equal to the number of episodes run.

    @param fig: an existing figure where the plot should be added. This would be equivalent of "hold"ing the plot in Matlab.

    @return A figure object with the created or updated figure.
    """

    def finalize_plot_rmse(axis, min_rmse=None, max_rmse=None, xlabel="Episode", ylabel="RMSE", legend="RMSE"):
        if rmse_se_values is not None:
            axis.errorbar(time, rmse_mean_values, yerr=rmse_se_values, capsize=4, color=color)
        #axis.set_xticks(np.arange(nepisodes)+1)
        axis.tick_params(axis='both', which='major', labelsize=np.round(fontsize*0.8))

        # Set the axis limits
        # NOTE that we need to make sure that the new plotted values fit in the possibly existing axis
        # (which may have already set min and max values), that's why we compute the min/max
        # between the current axis min/max and the min/max in the data being plotted now.
        if min_rmse is None:
            min_rmse = min(axis.get_ylim()[0], np.min(rmse_mean_values)*0.9)
        if max_rmse is None:
            max_rmse = max(axis.get_ylim()[1], np.max(rmse_mean_values)*1.1)
        axis.set_ylim((min_rmse, max_rmse))
        axis.set_xlabel(xlabel, fontsize=fontsize)
        axis.set_ylabel(ylabel, fontsize=fontsize)
        axis.set_title("Average {} by episode {}".format(ylabel, subtitle), fontsize=fontsize)
        if legend != "":
            axis.legend([legend], loc='upper left', fontsize=fontsize)

    def finalize_plot_alpha(axis, ylabel=r"Average $\alpha$ over visited states", legend=r"Average $\alpha$ over states visited in episode"):
        axis.tick_params(axis='both', which='major', labelsize=np.round(fontsize*0.8))
        axis.axhline(y=alpha_min, color='gray')
        axis.set_ylabel(ylabel, fontsize=fontsize)
        if max_alpha is not None:
            axis.set_ylim((0, max_alpha))
        if legend != "":
            axis.legend([legend], loc='upper right', fontsize=fontsize)

    #----------------------------------- Parse input parameters -------------------------------
    if alphas is not None and len(alphas) != len(rmse_mean_values) - 1:
        raise ValueError("The length of the `alphas` array ({}) must be one less the length of the RMSE values given in array `rmse_mean_values` ({})"
                         .format(len(alphas), len(rmse_mean_values)))
    nepisodes = len(rmse_mean_values) - 1

    # X-axis (time): goes from 0 to (nepisodes+1), because we need to plot the average RMSE by episode of which we
    # assume we have information from BEFORE the first episode is run (represented by time = 0 on which the RMSE
    # is computed from the initial guess of the value function) until AFTER the last episode has finished.
    time = np.arange(nepisodes+1)   # Array 0, 1, ..., nepisodes

    # Prepare the figure, subplots and axes (in original scale and/or log scale)
    if fig is not None:
        # Get the axes from the given figure, which is assumed to have two axes
        # Secondary axes are assumed to exist in the given figure `fig`
        axes = fig.get_axes()
        if plot_scales == "both":
            ax = axes[0]
            ax_tlog = axes[1]
            if alphas is not None:
                ax2 = axes[2]
                ax2_tlog = axes[3]
        elif plot_scales == "log":
            ax_tlog = axes[0]
            if alphas is not None:
                ax2_tlog = axes[1]
        else:
            ax = axes[0]
            if alphas is not None:
                ax2 = axes[1]
    else:
        if plot_scales == "both":
            fig = plt.figure(figsize=(20, 10))
            axes = fig.subplots(1, 2)
            ax = axes[0]
            ax_tlog = axes[1]
            if alphas is not None:
                ax2 = ax.twinx()
                ax2_tlog = ax_tlog.twinx()
        elif plot_scales == "log":
            fig = plt.figure(figsize=(10, 10))
            axes = [fig.subplots(1, 1)] # We enclose in list so that we can reference axes[0]
            ax_tlog = axes[0]
            if alphas is not None:
                ax2_tlog = ax_tlog.twinx()
        else:
            fig = plt.figure(figsize=(10, 10))
            axes = [fig.subplots(1, 1)] # We enclose in list so that we can reference axes[0]
            ax = axes[0]
            if alphas is not None:
                ax2 = ax.twinx()
    #----------------------------------- Parse input parameters -------------------------------

    if plot_scales != "log":
        ax.plot(time, rmse_mean_values, color=color, linewidth=2, linestyle=linestyle, zorder=10)
            ## zorder is used to define layer order (larger values imply going on top)
            ## Ref: stackoverflow.com/questions/37246941/specifying-the-order-of-matplotlib-layers
        finalize_plot_rmse(ax, min_rmse=min_rmse, max_rmse=max_rmse, xlabel="Episode", ylabel=kpi_name, legend=legend_rmse and "Average {} by episode".format(kpi_name) or "")

    if plot_scales == "both" or plot_scales == "log":
        ax_tlog.plot(time, rmse_mean_values, color=color, linewidth=2, linestyle=linestyle, zorder=10)
        # IMPORTANT: We use the symmetric log transformation (as opposed to the log transformation) for two reasons:
        # 1) the horizontal scale with the episode number starts at 0.
        # 2) the minimum value in the vertical axis is set to 0.
        finalize_plot_rmse(ax_tlog, min_rmse=min_rmse, max_rmse=max_rmse, xlabel="Episode (log scale)", ylabel=kpi_name, legend=legend_rmse and "Average {} by episode".format(kpi_name) or "")
        ax_tlog.set_xscale('symlog')
        #ax_tlog.set_yscale('symlog')

    if alphas is not None:
        if plot_scales != "log":
            ax2.plot(time[:-1], alphas, ':', color=color_alphas, zorder=0)
            finalize_plot_alpha(ax2)

        if plot_scales == "both" or plot_scales == "log":
            ax2_tlog.plot(time[:-1], alphas, ':', color=color_alphas, zorder=0)
            finalize_plot_alpha(ax2_tlog, legend=legend_alpha and r"Average $\alpha$ over states visited in episode" or "")
            ax2_tlog.set_xscale('symlog')
            #ax2_tlog.set_yscale('symlog')

        # Go back to the primary axis of the left subplot
        plt.sca(axes[0])

    return fig


def plot_results_2D(ax, V, params, colormap, vmin=None, vmax=None, format_labels=".3f", fontsize=7, title=""):
    """
    Plots a 2D matrix as an image displaying the matrix values in the same order as they are printed.

    This means that the upper-left corner corresponds to V[0,0] and the lower-right corner to V[nx-1,ny-1],
    where (nx, ny) = V.shape.
    
    Arguments:
    ax: matplotlib.axes._subplots.AxesSubplot
        Axis where the plot should be created, typically created by either `plt.figure().gca()` or `plt.figure().subplots(1,1)`
        Note that this parameter should be a scalar, NOT an array of axis handles.

    V: 2D numpy.array
        State value function for each 2D state.

    params: dict
        Dictionary with simulation parameters:
        - 'alpha': learning rate
        - 'gamma': discount factor of the environment
        - 'lambda': lambda parameter in TD(lambda)
        - 'alpha_min': lower bound for the learning rate used when running the simulation       
        - 'nepisodes': number of episodes run to estimate V_estimate     

    colormap: matplotlib.cm
        Colormap to use in the plot associated to the state values shown in the image.

    vmin, vmax: (opt) float values
        Minimum and maximum V values to plot which is used to generate a normalized colormap with plt.Normalize().

    fontsize: (opt) int
        Font size to use for the labels showing the state value function at each 2D state.

    title: (opt) string
        Title to add to the plot (execution parameters are added by this function).

    Return: matplotlib.axes._subplots.AxesSubplot
    The input axis object is returned.
    """
    if not isinstance(ax, matplotlib.axes._subplots.Subplot):
        raise ValueError("The 'ax' parameter must be an instance of matplotlib.axes._subplots.AxesSubplot")

    title = title + "\nalpha={:.2f}>={:.2f}, gamma={:.2f}, lambda={:.2f}, {} episodes" \
                 .format(params['alpha'], params['alpha_min'], params['gamma'], params['lambda'], params['nepisodes'])
    # Since imshow() is assumed to plot a matrix V with the goal of seeing the image as one sees it when PRINTING V,
    # the upper left corner corresponds to V[0,0] and the lower right corner to V[nx-1, ny-1], where (nx, ny) = V.shape.
    if vmin is not None and vmax is not None:
        colornorm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        colornorm = None
    img = ax.imshow(V, cmap=colormap, norm=colornorm)

    # Add the colorbar on the right
    # VERY COMPLICATED!! (because the ax object does NOT have a colobar() method as matplotlib.pyplot does)
    # Ref: https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot
    # Note that another option would be to use the much simpler one-liner:
    #   plt.colorbar(img, ax=ax)
    # BUT this generates colorbars that occupy the whole vertical space and overlap with the vertical axis values...
    # Also see: https://matplotlib.org/2.0.2/examples/pylab_examples/custom_cmap.html
    divider = make_axes_locatable(ax)
    fig = plt.gcf()
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation='vertical');

    (nx, ny) = V.shape
    # Go over each value in the plotted matrix V
    # First we go over the rows (y) and then over the columns (x)
    for x in range(nx):
        for y in range(ny):
            # NOTE: (2022/05/06) The coordinate where the given text is added by default is in DATA coordinates
            # (as opposed to in AXIS coordinates, where (0,0) is the lower-left corner and (1,1) is the upper-right corner)
            # ref: documentation of ax.text()).
            #
            # Since imshow() by default places the nx dimension vertically (i.e. each x is a different row in the image)
            # and the ny dimension horizontally (i.e. each y is a different column in the image), the DATA coordinates imply that
            # the y values are measured on the horizontal axis of the image, while the x values are measured
            # on the vertical axis of the image.
            #
            # THEREFORE, we need to place the label corresponding to value V[x,y] at position (y,x) in the image
            # (DATA) axis coordinates.
            #
            # The above was deduced by trial and error and then checked by comparing the pixel color in each image pixel
            # with the text value labels added below (showing the plotted values V) and the layout of the matrix V
            # when printed, as in e.g. V[:10, :5].
            ax.text(y, x, "{:{format_labels}}".format(V[x,y], format_labels=format_labels), fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
    ax.set_title(title)

    return ax


class EpisodeSimulation:

    def plot_results(self, params, nepisodes,
                     V_estimate, V_true, RMSE_by_episode, alphas_by_episode,
                     ylabel="(Average) alpha", ylim=None,
                     max_rmse=0.8, color_rmse="black"):
        """
        Plots the estimated and true state value function in one plot,
        the (average) learning rate alpha by episode in another plot where the RMSE by episode
        is plotted on a secondary axis, if given (not None).
        
        Arguments:
        params: dict
            Dictionary with simulation parameters:
            - 'alpha': learning rate
            - 'gamma': discount factor of the environment
            - 'lambda': lambda parameter in TD(lambda)
            - 'alpha_min': lower bound for the learning rate used when running the simulation       

        nepisodes: int
            Number of episodes used when running the simulation generating the results that are plotted.

        V_estimate: numpy.array of length number of states in the environment
            Estimated state value function.
            
        V_true: numpy.array of length number of states in the environment
            True state value function.

        RMSE_by_episode: numpy.array
            Root Mean Squared Error of the estimated state value function by episode, including the very start of the
            learning process, where the RMSE is computed on the initial guess of the value function.
            Its length should be equal to the number of episodes run + 1.
            If None, the RMSE is not plotted.

        alphas_by_episode: list or numpy.array
            Average learning rate by episode.
            Its length should be equal to the number of episodes run.
        """
        if len(V_true) != len(V_estimate):
            raise ValueError(f"The length of the true values array ({len(V_true)}) and of the estimated values array ({len(V_estimate)}) are different")

        title = "alpha={:.2f}, gamma={:.2f}, lambda={:.2f}, {} episodes" \
                     .format(params['alpha'], params['gamma'], params['lambda'], nepisodes)

        all_states = np.arange(len(V_true))
        all_episodes = np.arange(nepisodes + 1)    # This is 0, 1, ..., nepisodes
                                                   # i.e. it has length nepisodes + 1 so that the very first
                                                   # RMSE (for the initial guess of the value function)
                                                   # is included in the plot.

        plt.figure()
        plt.plot(all_states, V_true, 'b.-')
        plt.plot(all_states, V_estimate, 'r.-')
        ax = plt.gca()
        ax.set_xticks(all_states)
        plt.title(title)

        plt.figure()
        plt.plot(all_episodes[:-1], alphas_by_episode, "k:")
        #plt.xticks(np.arange(nepisodes)+1)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.axhline(y=params['alpha_min'], color="gray")
        ax.set_title(title)

        if RMSE_by_episode is not None:
            ax2 = ax.twinx()
            ax2.plot(all_episodes, RMSE_by_episode, color=color_rmse)
            #ax2.set_ylim((0, np.max(RMSE_by_episode)))
            ax2.set_ylim((0, max_rmse))
            ax2.set_ylabel("RMSE")

            # Go back to the primary axis
            plt.sca(ax)
        else:
            ax2 = None

        return (ax, ax2)


#-------------------- AUXILIARY AND PLOTTING FUNCTIONS FOR FVAC ------------------#
# Types of environments that can be defined
@unique
class Environment(Enum):
    Gridworld = 1
    MountainCar = 2


#--- Auxiliary functions
KL_THRESHOLD = 0.005
policy_changed_from_previous_learning_step = lambda KL_distance, num_states: np.abs(KL_distance) / num_states > KL_THRESHOLD


def compute_normalized_KL_distance(KL_distance, learner):
    "Computes the KL distance normalized by the number of environment states KNOWN by the given learner. The learner needs to define the getKnownEnvironmentSet() method"
    num_states_known_by_learner = len(learner.getKnownEnvironmentSet())
    return KL_distance / max(1, num_states_known_by_learner), num_states_known_by_learner


def show_elapsed_time(learning_method, time_elapsed, time_cpu):
    print("{} learning process took {:.1f} minutes, {:.1f} hours (CPU: {:.1f} minutes, {:.1f} hours)".format(learning_method.upper(), time_elapsed / 60, time_elapsed / 3600, time_cpu / 60, time_cpu / 3600))


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
    V_true, mu = computing.compute_state_value_function_from_environment_and_policy(  env, policy, gamma=gamma,
                                                                                      continuing_task=learning_task==LearningTask.CONTINUING,
                                                                                      average_reward_criterion=learning_criterion==LearningCriterion.AVERAGE,
                                                                                      atol=atol)
    env.setV(V_true)
    dict_proba_stationary = dict(zip(np.arange(len(mu)), mu))
    avg_reward_true = computing.compute_expected_reward(env, dict_proba_stationary)
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


def plot_state_counts(dict_simulator, learning_method, rep, params_exec, ax=None, trajectory=None, trajectory_length=1000, epsilon_random_action=0.0,
                      seed=None, verbose=False, verbose_period=1, plot_absorption_set=True, add_title=True):
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
        _simulator = copy.deepcopy(dict_simulator[learning_method][rep])
        _learner_under_policy, _nsteps, _average_reward = _simulator.run_exploration(max_time_steps=trajectory_length, epsilon_random_action=epsilon_random_action, seed=seed, verbose=verbose, verbose_period=verbose_period)
        trajectory = np.array(_learner_under_policy.getStates())
        # Generate the 1D array containing the state counts for each state index
        # (as the above run_exploration() method does NOT update the state counts of the learner because this is done by the learn() method of the learner and the run_exploration()
        # method does NOT learn, it only collects a trajectory)
        state_counts = _learner_under_policy.getStateCountsFromTrajectory()
    else:
        # Distribution of state counts in the given trajectory
        state_counts = np.zeros(dict_simulator[learning_method][rep].getEnv().getNumStates(), dtype=int)
        _visited_states = pd.Series(trajectory).value_counts()
        for s, c in _visited_states.items():
            state_counts[s] = c

    # Show the state counts as an image
    ax, img = dict_simulator[learning_method][rep].getEnv().plot_values(state_counts, ax=ax, cmap="Blues", add_colorbar=ax is None)
    if env_type == Environment.MountainCar:
        # Add the trajectory of the last replication
        assert T <= len(trajectory)
        trajectory2plot = trajectory[:T]
        dict_simulator[learning_method][rep].getEnv().plot_points(trajectory2plot, ax=ax, is_trajectory=True)

    if learning_method_type == "values_fv" and plot_absorption_set:
        # Plot the final absorption set
        dict_simulator[learning_method][rep].getEnv().plot_points(np.array(list(dict_simulator[learning_method][rep].getAgent().getLearner().getAbsorptionSet())), ax=ax, color="red", markersize=5, style="x")

    # Add the count labels
    dict_simulator[learning_method][rep]._add_count_labels(ax, state_counts, factor_fontsize=3.0)
    if add_title:
        plt.suptitle(f"{learning_method.upper()}\n{learning_task.name} learning task - {learning_criterion.name} reward criterion - {env_type.name} {dict_simulator[learning_method][rep].getEnv().getShape()} - Replication {rep}"
                     f"\nN={N}, T={T}, wind_dict={wind_dict}, MAX budget={max_time_steps_benchmark} steps per policy learning step"
                     f"\nDistribution of state counts by a trajectory that follows the last learned policy AND final absorption set A")

    return ax


def plot_policy(env, policy, average_reward, state_counts_hist, params_exec, axes=None, is_problem_2d=True, KL_distance=None, KL_distance_norm=None, absorption_set=None, state_value=None, t_learn=1, fontsize=14, verbose=True):
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

    - state_value: (opt) array
        1D array containing the state value for each state based on their 1D index.
        This value is used to color the upper-right corner of each cell using a cool-warm color map where bluer means smaller and redder means larger.
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
            mask_absorption_set[0, 2] = 1 if state_value is None else np.nan    # Reserve this corner to plot the state value function V(s) when given
            mask_absorption_set[2, 0] = 1
            mask_absorption_set[2, 2] = 1 if state_value is None else np.nan    # Reserve this corner to plot the average reward when the state value fnction V(s) is given

        # Mask for the V(s) value (if given)
        if state_value is not None:
            # Prepare a 3x3 matrix that will be used to mark the cells belonging to the absorption set
            # The state value function will be shown on the upper right corner.
            mask_state_value = np.nan*np.ones((3, 3))
            VR_min, VR_max = min(average_reward, np.nanmin(state_value)), max(average_reward, np.nanmax(state_value))

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
                        # This is a valid cell / state
                        # => Plot useful information about the state

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

                        # Mark the cells that are part of the absorption set, if given
                        if state_value is not None:
                            # Show the state value V(s) using the cool-warm palette where blue means small and read means large
                            # Note: we normalize the values to plot to [0, 1] so that they take the full [0, 1] range.
                            mask_state_value[0, 2] = (state_value[state_1d] - VR_min) / max(1, VR_max - VR_min)
                            mask_state_value[2, 2] = (average_reward - VR_min) / max(1, VR_max - VR_min)
                            axes[i, j].imshow(mask_state_value, cmap="coolwarm", vmin=0, vmax=1)

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
#-------------------- AUXILIARY AND PLOTTING FUNCTIONS FOR FVAC ------------------#
