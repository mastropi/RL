# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:16:50 2020

@author: Daniel Mastropietro
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_rmse_by_episode(rmse_mean_values, rmse_se_values=None, max_rmse=None, color="black",
                         alphas=None, alpha_min=0.0, color_alphas='black', max_alpha=None, fig=None, subtitle="", fontsize=12, legend=True):
    """
    Plots the average RMSE values (over states) by episode
    
    @param rmse_mean_values: numpy array containing the average RMSE value per episode.
    @param rmse_se_values: (optional) numpy array containing an error measure of the average RMSE, e.g. their standard error.
    @param max_mse: (optional) maximum RMSE value to show in the plot (for visual comparison reasons)
    
    @param fig: an existing figure where the plot should be added. This would be equivalent of "hold"ing the plot in Matlab.
    
    @return A figure object with the created or updated figure.
    """
    
    def finalize_plot_rmse(axis, xlabel="Episode", ylabel="RMSE", legend="RMSE"):
        if rmse_se_values is not None:
            axis.errorbar(time, rmse_mean_values, yerr=rmse_se_values, capsize=4, color=color)
        #axis.set_xticks(np.arange(nepisodes)+1)
        axis.tick_params(axis='both', which='major', labelsize=np.round(fontsize*0.8))
        axis.set_ylim((0, max_rmse))
        axis.set_xlabel(xlabel, fontsize=fontsize)
        axis.set_ylabel(ylabel, fontsize=fontsize)    
        axis.set_title("Average RMSE by episode {}".format(subtitle), fontsize=fontsize)
        axis.legend([legend], loc='upper left', fontsize=fontsize)

    def finalize_plot_alpha(axis, ylabel="Average alpha in last experiment", legend=r"Average $\alpha$ over states visited in episode"):
        axis.tick_params(axis='both', which='major', labelsize=np.round(fontsize*0.8))
        axis.axhline(y=alpha_min, color='gray')
        axis.set_ylabel(ylabel, fontsize=fontsize)
        if max_alpha is not None:
            # We use a positive value for the minimum alpha so that the log transform does not complain
            # when changing the axis to log scale.
            axis.set_ylim((1E-6, max_alpha))
        axis.legend([legend], loc='upper right', fontsize=fontsize)

    # X-axis (time)
    nepisodes = len(rmse_mean_values)
    time = np.arange(nepisodes)+1

    if max_rmse is None:
        max_rmse = np.max(rmse_mean_values)*1.1

    if fig is not None:
        # Get the axes from the given figure, which is assumed to have two axes
        axes = fig.get_axes()
    else:
        fig = plt.figure(figsize=(20,10))
        axes = fig.subplots(1,2)

    ax = axes[0]
    ax_tlog = axes[1]
    ax.plot(time, rmse_mean_values, color=color, linewidth=2, zorder=10)
        ## zorder is used to define layer order (larger values imply going on top)
        ## Ref: stackoverflow.com/questions/37246941/specifying-the-order-of-matplotlib-layers
    finalize_plot_rmse(ax)

    ax_tlog.plot(time, rmse_mean_values, color=color, linewidth=2, zorder=10)
    # IMPORTANT: We first need to transform the axis to log scale BEFORE applying the axis limits in finalize_plot_rmse()
    # because otherwise the log-scale transformation screws up the plot and doesn't show anything... the reason is that
    # the aforementioned function sets the axis minimum value to 0. Note that I don't want to set it to a very small
    # positive number because if I do so we don't really see much what's going on in the graph.
    ax_tlog.set_xscale('log')
    ax_tlog.set_yscale('log')
    finalize_plot_rmse(ax_tlog, xlabel="Episode (log scale)", ylabel="RMSE (log scale)")

    if alphas is not None:
        if fig is not None and len(axes) >= 3:
            # A secondary axis is assumed to exist in the given figure `fig`
            ax2 = axes[2]
        else:
            ax2 = ax.twinx()
        ax2.plot(time, alphas, ':', color=color_alphas, zorder=0)
        finalize_plot_alpha(ax2)

        if fig is not None and len(axes) >= 4:
            # A secondary log-axis is assumed to exist in the given figure `fig`
            ax2_tlog = axes[3]
        else:
            ax2_tlog = ax_tlog.twinx()
        ax2_tlog.plot(time, alphas, ':', color=color_alphas, zorder=0)
        finalize_plot_alpha(ax2_tlog)
        ax2_tlog.set_xscale('log')
        ax2_tlog.set_yscale('log')

        # Go back to the primary axis of the left subplot
        plt.sca(ax)

    return fig


class EpisodeSimulation:
    
    def plot_results(self, params,
                     V_estimated, V_true, RMSE_by_episode, alphas_by_episode, y2label="(Average) alpha",
                     max_rmse=0.8, color_rmse="black", plotFlag=True):
        """
        Plots the estimated and true state value function.
        
        Arguments:
        params: dict
            Dictionary with simulation parameters:
            - 'alpha': learning rate
            - 'gamma': discount factor of the environment
            - 'lambda': lambda parameter in TD(lambda)
            - 'alpha_min': lower bound for the learning rate used when running the simulation       

        V_estimated: numpy.array of length number of states in the environment
            Estimated state value function.
            
        V_true: numpy.array of length number of states in the environment
            True state value function.

        RMSE_by_episode: numpy.array of length number of episodes run
            Root Mean Squared Error of the estimated state value function by episode.
        
        alphas_by_episode: list or numpy.array
            Average learning rate by episode.
        """
        if plotFlag:
            title = "alpha={:.2f}, gamma={:.2f}, lambda={:.2f}, {} episodes" \
                         .format(params['alpha'], params['gamma'], params['lambda'], self.nepisodes)

            all_states = np.arange(self.nS+2)
            all_episodes = np.arange(self.nepisodes)+1

            plt.figure()
            plt.plot(all_states, V_true, 'b.-')
            plt.plot(all_states, V_estimated, 'r.-')
            ax = plt.gca()
            ax.set_xticks(all_states)
            plt.title(title)

            plt.figure()
            plt.plot(all_episodes, RMSE_by_episode, color=color_rmse)
            #plt.xticks(np.arange(self.nepisodes)+1)
            ax = plt.gca()
            #ax.set_ylim((0, np.max(RMSE_by_episode)))
            ax.set_ylim((0, max_rmse))
            ax.set_xlabel("Episode")
            ax.set_ylabel("RMSE")
            ax.set_title(title)
            
            ax2 = ax.twinx()
            ax2.plot(all_episodes, alphas_by_episode, "k:")
            ax2.set_ylabel(y2label)
            ax2.axhline(y=params['alpha_min'], color="gray")
            
            # Go back to the primary axis
            plt.sca(ax)
            
            return (ax, ax2)
    
        return (None, None)

    def array2str(self, x, sep=", ", fmt=":.6f"):
        "Converts an array (possibly numeric) to string separated by `sep`"
        return "[" + sep.join( map(lambda s: ("{" + fmt + "}").format(s), x) ) + "]"
