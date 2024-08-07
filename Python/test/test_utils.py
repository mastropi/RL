# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:16:50 2020

@author: Daniel Mastropietro
@description: Util functions used in running unit tests
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def array2str(x, sep=", ", fmt=":.6f"):
    "Converts an array (possibly numeric) to string separated by `sep` using the given string format"
    return "[" + sep.join( map(lambda s: ("{" + fmt + "}").format(s), x) ) + "]"

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
