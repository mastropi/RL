# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 10:59:13 2022

@author: Daniel Mastropietro
@description: Reads FVRL results and plots them
"""

import runpy
runpy.run_path('../../setup.py')

import os

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from Python.lib.utils.basic import convert_str_to_list_of_type


#------------------------ Auxiliary functions ------------------------
# Function that returns a function to compute percentiles using the agg() aggregation method in pandas `groupby` data frames
# Ref: https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function
def percentile(n):
    def percentile_(x):
        return x.quantile(n/100)
    percentile_.__name__ = 'percentile_{:.0f}'.format(n)
    return percentile_

def plot_curves(ax, results, color="blue", linewidth=0.3):
    for r in np.unique(results.replication):
        toplot = results[results.replication == r].reset_index(drop=True)
        ax.plot(np.r_[0, toplot.t_learn], np.r_[toplot.theta[0], toplot.theta],
                color=color, linewidth=linewidth)

def plot_two_bands(fig, ax, df_agg, percentiles_low, percentiles_upp, alphas, linecolor, param_var='theta', ymin=-1.1, ymax=None, ref=None, ylabel="theta(s)", title="", show_title=True, fontsize=15):
    line_median = ax.plot(df_agg.index, results_grp_agg[param_var]['median'], color=linecolor, linewidth=2)[0]    # We need to select the first element because the returned value of plot() is a list!
    line_mean = ax.plot(df_agg.index, results_grp_agg[param_var]['mean'], color=linecolor, linewidth=2, linestyle="dashed")[0]    # We need to select the first element because the returned value of plot() is a list!
    fill_between_1 = ax.fill_between(  df_agg.index,
                            #df_agg[param_var]['max'],
                            #df_agg[param_var]['min'],
                            df_agg[param_var]['percentile_{}'.format(percentiles_low[0])],
                            df_agg[param_var]['percentile_{}'.format(percentiles_upp[0])],
                            color=linecolor,
                            alpha=alphas[0])
    fill_between_2 = ax.fill_between(  df_agg.index,
                            #df_agg[param_var]['mean'] + 2*df_agg[param_var]['std'] / np.sqrt(df_agg[param_var]['count']),
                            #df_agg[param_var]['mean'] - 2*df_agg[param_var]['std'] / np.sqrt(df_agg[param_var]['count']),
                            df_agg[param_var]['percentile_{}'.format(percentiles_low[1])],
                            df_agg[param_var]['percentile_{}'.format(percentiles_upp[1])],
                            color=linecolor,
                            alpha=alphas[1])
    # We need to define a proxy Artist for the second shade fille because the two transparency factors are "added"
    # when seeing the color intensity in the graph... therefore using the output of fill_between() as legend object
    # will not convey the actual color of the shade.
    # Ref for creating proxy Artists: https://matplotlib.org/2.0.2/users/legend_guide.html#proxy-legend-handles
    # Note: we can get the objects that can be added to a legend by running `ax.get_legend_handles_labels()`.
    legend_obj_fill2 = mpatches.Patch(color=linecolor, alpha=sum(alphas))

    ref_line = finalize_plot(fig, ax, method, ymin=ymin, ymax=ymax, ref=ref, ylabel=ylabel, title=title, show_title=show_title, fontsize=fontsize)
    legend_obj = [ref_line, line_median, line_mean, fill_between_1, legend_obj_fill2]
    ax.legend(  handles=legend_obj,
                labels=["Optimum theta parameter(s)",
                        "Median curve",
                        "Mean curve",
                        "Location of {}% of curves".format(percentiles_upp[0] - percentiles_low[0]),
                        "Location of {}% of curves".format(percentiles_upp[1] - percentiles_low[1])],
                fontsize=fontsize)

    return line_median, line_mean, fill_between_1, fill_between_2, legend_obj_fill2, ref_line, legend_obj

def finalize_plot(fig, ax, method, ymin=-1.1, ymax=None, ref=None, ylabel="theta(s)", title="", show_title=True, fontsize=15):
    ax.set_xlabel("Learning step", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    # Fontsize for axis ticks
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    # Axis limits
    ymin = -1.1 if ymin is None else ymin
    if ymax is None:
        ylim = ax.get_ylim()
        ymax = np.max([ylim[1], ymax])
    ax.set_ylim((ymin, ymax))
    if show_title:
        ax.set_title("Method: {}".format("FVRL" if method.upper() == "FV" else "MC"), fontsize=fontsize)
        fig.suptitle(title)

    # Set integer values on the horizontal axis (integer steps)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.set_aspect(1 / ax_params.get_data_ratio())

    if ref is not None:
        # This can be used to add reference lines, such as the true values of estimated parameters
        ref_line = ax.axhline(ref, color="gray", linestyle="dashed")
        return ref_line

#raise KeyboardInterrupt

#------------------------ Auxiliary functions ------------------------


#---------------------------- Single Server --------------------------
resultsdir = "../../RL-002-QueueBlocking/results/RL/single-server"
figuresdir = resultsdir


#-- Execution parameters whose results should be read
# The values of the execution parameters are present in the names of the result files
# They ALL need to be lists. When saving the generated plot to a file, we should list only ONE value in the list (because we select the first value to be stored in the filename)
theta_true_values = [18.0] #[23.0] #[18.0]
theta_start_values = [28.1] #[34.1] #[28.1]
J_factor_values = [0.5] #[0.3] #[0.5]
error_rel_phi = [1.0] #[1.0] #[0.2]
error_rel_et = [0.2] #[1.0] #[0.2]
suffix = "" #"-DecreasingAlpha"

# Read the results
results_file_fv = os.path.join(resultsdir, "SimulatorQueue_FV-theta0={}-theta={}-J={}-E={},{}{}.csv" \
                                            .format(theta_true_values, theta_start_values, J_factor_values, error_rel_phi, error_rel_et, suffix))
results_file_mc = os.path.join(resultsdir, "SimulatorQueue_MC-theta0={}-theta={}-J={}-E={},{}{}.csv" \
                                            .format(theta_true_values, theta_start_values, J_factor_values, error_rel_phi, error_rel_et, suffix))
results = dict({'fv': pd.read_csv(results_file_fv),
                'mc': pd.read_csv(results_file_mc)})


#----- Plot
# Define the two percentiles set that should be shown as shaded area of the theta evolution over replications
# Use 0 and 100 to show the min and max bands respectively
percentiles_low = [10, 25]; percentiles_upp = [90, 75]
# Transparency to use for each shaded range (e.g. 80% and 50% in this order... the larger alpha, the less the transparency) generated with fill_between()
alphas = [0.2, 0.25]
fontsize = 20
colormaps = dict({'fv': "Greens", 'mc': "Reds"})
color_normalization = matplotlib.colors.Normalize()    # matplotlib.colors.LogNorm()
plot_each_curve = False  # Whether to plot each learning curve in the bands plot as well
savefig = True
show_title = not savefig

fig1 = plt.figure(figsize=(24, 18)); axes1 = fig1.subplots(1,2)
fig2 = plt.figure(figsize=(24, 18)); axes2 = fig2.subplots(1,2)
n_replications = max(results['fv'].replication)
#theta_max = np.max([ max(results['fv'].theta), max(results['mc'].theta) ])
theta_max = 35
title_params = "Optimum theta = {}, Theta start = {}, J/K = {}, Rel. Error Phi = {}%, Rel. Error E(T_A) = {}%" \
                 .format([theta - 1 for theta in theta_true_values], theta_start_values, J_factor_values, [e*100 for e in error_rel_phi], [e*100 for e in error_rel_et]) + \
               (suffix == "" and "\n(constant learning rate alpha on " or suffix and "\n(decreasing learning rate alpha on ") + "{} replications)".format(n_replications)
for i, method in enumerate(results.keys()):
    theta_opt_true = theta_true_values[0] - 1   # IMPORTANT: We subtract 1 because recall that the optimum theta is not precisely `theta_true`,
                                                # as this value is actually theta_ref, the sref value on which the exponential cost is centered.
                                                # For more details see the notes in the costBlockingExponential() function in environments/queues.py
    linecolor = colormaps[method].lower()[:-1]  # e.g. "Greens" -> "green"

    #-- Distribution of theta values by learning step (t_learn)
    results_grp = results[method].groupby(['t_learn'])
    results_grp_hist = results_grp.apply( lambda df: np.histogram(df['theta']) )
    results_grp_agg = results_grp[['theta', 'theta_next']].agg(['count', 'mean', 'min', 'median', 'max', 'std'] +
                                                                [percentile(p) for p in percentiles_low] + [percentile(p) for p in percentiles_upp])

    #-- Prepare the data for plotting. We need to create a meshgrid on all possible (t_learn, theta) values!
    theta_values_endpoints = []
    for t in results_grp_hist.index:
        bin_endpoints_t = results_grp_hist[t][1]
        theta_values_endpoints += list(bin_endpoints_t)
    theta_values_for_plot = np.unique(theta_values_endpoints)

    # We now define theta_counts, the 2D array that will contain, for each learning step t to plot,
    # the number of curves that fall in each [theta1, theta2) interval generated from ALL the observed theta values
    # (for all learning steps t and all replications)
    n_tvalues = results_grp.ngroups
    theta_counts = np.nan*np.ones((n_tvalues, len(theta_values_for_plot)))
    for t in results_grp_hist.index:
        bin_counts_t = results_grp_hist[t][0]
        bin_endpoints_t = results_grp_hist[t][1]
        bin_assignments_t_for_all_theta_values = np.digitize(theta_values_for_plot, bin_endpoints_t)
        # We now assign the count value to plot associated to each theta (among ALL the observed theta values)
        # to the t-histogram count of the t-bin where the theta value falls.
        # If the theta value falls outside of the range defined by the t-bins, the count is set to 0,
        # as "falling outside" means that no curve was observed "passing through that theta"
        # for the currently analyzed learning step.
        theta_counts[t-1] = [0 if b == 0 or b == len(bin_endpoints_t)
                             else bin_counts_t[b-1] / sum(bin_counts_t) * 100
                             for b in bin_assignments_t_for_all_theta_values]

    #-- Plot
    # 1) Plot distribution of curves per each learning step
    # Doesn't help because axes are categorical in plt.imshow()!
    #data2plot = pd.DataFrame.from_dict({'t': results_fv_grp_hist.index, ''})
    #plt.imshow()

    # Generate the grid to which the colors showing the distribution of curves will be associated (using pcolormesh())
    x, y = np.meshgrid(results_grp_hist.index, theta_values_for_plot, indexing='ij')
        ## indexing='ij' (as opposed to the default indexing='xy') is important so that the shapes of x, y, and theta_counts coincide
        ## (e.g. they are all 50 x 421 and NOT some 50 x 421 and other 421 x 50.

    # Using plot_surface() (not what I want)
    #from mpl_toolkits.mplot3d import Axes3D
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(x, y, theta_values)
    #ax.view_init(-10, 110)

    # Using pcolormesh() --> WHAT I WANT!
    # Colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    pc = axes1[i].pcolormesh(x, y, theta_counts, cmap=colormaps[method], norm=color_normalization)
    if not savefig:
        cbar = plt.colorbar(pc, ax=axes1[i])
        cbar.set_label("% curves", fontsize=fontsize)
    # If we want to add vertical lines to indicate each t_learn value... but it usually becomes too clumsy
    #ax.vlines(results[method].t_learn, 0, ax.get_ylim()[1], colors="gray", linewidth=0.1)
    ref_line = finalize_plot(fig1, axes1[i], method, ymax=theta_max*1.1, ref=theta_opt_true, ylabel="theta", title="" if savefig else title_params, fontsize=fontsize)
                ## NOTE: We do NOT include the title when saving the plot because the file is intended to be inserted into a document, where this technical title should not be shown

    if plot_each_curve:
        plot_curves(axes1[i], results[method], color=linecolor)

    axes1[i].legend(handles=[ref_line], labels=["Optimum theta parameter"], fontsize=fontsize)


    # 2) Plot of average and median behaviour with bands
    plot_two_bands(fig2, axes2[i], results_grp_agg, percentiles_low, percentiles_upp, alphas, linecolor,
                    ymax=theta_max*1.1, ref=theta_opt_true, ylabel="theta", title="" if savefig else title_params, fontsize=fontsize)
            ## NOTE: We do NOT include the title when saving the plot because the file is intended to be inserted into a document, where this technical title should not be shown

    if plot_each_curve:
        # Add the curves for each replication but very thin
        plot_curves(axes2[i], results[method], color=linecolor)

if savefig:
    params_prefix = f"RL-single-FVMC-K0={int(theta_opt_true+1)}-theta_start={theta_start_values[0]}-J={J_factor_values[0]}-E{error_rel_phi[0]},{error_rel_et[0]}"
    figfile1 = os.path.join(figuresdir, f"{params_prefix}-dist.jpg")
    figfile2 = os.path.join(figuresdir, f"{params_prefix}-bands.jpg")
    # To avoid cut off of vertical axis label!!
    # Ref: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    #left, right, top, bottom = 0.15, 0.85, 0.95, 0.10
    # From a tight layout requested interactively and then copy & paste
    top = 0.954; bottom = 0.088; left = 0.049; right = 0.978; hspace = 0.2; wspace = 0.144
    fig1.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    fig2.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    # Save leaving just a little margin (pad_inches=0.1), otherwise we would need to crop the generated figure
    # before including it in a paper.
    # Ref: https://stackoverflow.com/questions/36203597/remove-margins-from-a-matplotlib-figure
    fig1.savefig(figfile1, bbox_inches="tight", pad_inches=0.1)
    print("Figure showing distribution saved to file {}".format(figfile1))
    fig2.savefig(figfile2, bbox_inches="tight", pad_inches=0.1)
    print("Figure showing bands saved to file {}".format(figfile2))
#---------------------------- Single Server --------------------------




#---------------------------- Loss Network --------------------------
# First import packages at the beginning
resultsdir = "../../RL-002-QueueBlocking/results/RL/loss-network"
figuresdir = resultsdir

# Parameters of the simulation to plot
K = 6
blocking_costs = [1920, 19753] #[183, 617]
rhos = [0.8, 0.6] #[0.5, 0.3] #[0.8, 0.6]
lambdas = [1, 5]
theta_true = [3, 5]; K_true = [4, 6]
theta_start = [0.1, 0.1]
J_factors = [0.5, 0.5] #[0.3, 0.5] #[0.5, 0.5]
NT_values = [100, 500] #[200, 400] #[200, 500] #[100, 500]
use_stationary_probability_for_start_states = True

#---- 0) Read the result files
# Results are read from 'pp*.csv' files which are created as copies of the original files containing the results.
# The reasons for doing this are the following:
# - Each filename pointing to the results has a datetime prefix which is different for each method
# - The filenames might be TOO LONG and they may not be found by Python...
# Example of an ORIGINAL filename containing the results:
# results_file_fv = os.path.join(resultsdir, f"SimulatorQueue_20230319_154817_FV-K={K}-costs={blocking_costs}-rhos={rhos},theta0={theta_true},theta=[{theta_start}]-J=[{J_factors}]-NT=[{NT_values}]-ProbStart={use_stationary_probability_for_start_states}.csv")
results_file_true = os.path.join(resultsdir, "pp_true.csv")
results_file_fv = os.path.join(resultsdir, "pp_fv.csv")
results_file_mc = os.path.join(resultsdir, "pp_mc.csv")
results = dict({'1-true': pd.read_csv(results_file_true, sep="|"),
                '2-fv': pd.read_csv(results_file_fv, sep="|"),
                '3-mc': pd.read_csv(results_file_mc, sep="|")})
# Convert the lists read as strings to lists of floats (for the columns that are needed for plotting)
for method in results.keys():
    results[method]['theta'] = [convert_str_to_list_of_type(v, sep=" ") for v in results[method]['theta']]
    results[method]['theta_next'] = [convert_str_to_list_of_type(v, sep=" ") for v in results[method]['theta_next']]


#----- 1) Plot the evolution of theta and of the expected cost
fontsize = 20
symbols = dict({'1-true': 'b-', '2-fv': 'g-', '3-mc': 'r-'})
savefig = False
show_title = False #not savefig

# We plot FV and MC side by side
fig = plt.figure()
axes = fig.subplots(1, len(results))
for i, method in enumerate(sorted(results.keys())):
    df_learning = results[method]

    # Axis associated to the method where the parameters evolution is plotted
    ax_params = axes[i]

    # Plot of parameters evolution
    lines = []
    reflines = []
    legend_lines = []
    legend_reflines = []
    expected_lines = []
    expected_reflines = []
    for r in sorted(df_learning['replication'].value_counts().index):
        mask = df_learning['replication'] == r
        thetas = df_learning['theta'][mask]
        expected_rewards = df_learning['expected_reward_true'][mask]
        # Linestyles are sorted so that more continuous line means larger rho
        linestyles_ref = ['-', '--', '-.', ':']
        # Rank of rhos in reversed order (largest value has highest rank so that the largest value gets the most solid line in the plot)
        rank_rhos = list(reversed(np.array(rhos).argsort().argsort()))
        linestyles = [linestyles_ref[r] for r in rank_rhos]
        for i in range(len(theta_true)):
            linestyle_i = linestyles[i % len(linestyles)]
            lines_i = ax_params.plot([t[i] for t in thetas], symbols[method], linestyle=linestyle_i)
            if r == 1:
                refline_i = ax_params.axhline(theta_true[i], color='black', linestyle=linestyle_i)
                lines += lines_i
                reflines += [refline_i]
                legend_lines += ["Theta_" + str(i + 1)]
                legend_reflines += ["True Optimum Theta_" + str(i + 1)]

    finalize_plot(fig, ax_params, method, ymin=-1.1, ymax=np.max(np.squeeze(K_true)), fontsize=fontsize, show_title=show_title)
        ## lower limit = -1.1 because -1 is the minimum possible theta value that is allowed for the linear step parameterized policy
    ax_params.legend(lines + reflines, legend_lines + legend_reflines, fontsize=fontsize)

    if show_title:
        ax_params.set_title(method.upper())

if savefig:
    plt.get_current_fig_manager().window.showMaximized()
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.11, right=0.9, hspace=0.2, wspace=0.2)
    ## Note: the adjustment values were taken by maximizing the space use in the graph via the figure window (generated by Pycharm Qt5Agg) and selecting "tight layout" from the adjustable knobs icon)
    fig.savefig(os.path.join(resultsdir, f"LossNetwork-K={K},costs={blocking_costs},rhos={rhos},lambdas={lambdas},JF={J_factors},NT_values={NT_values}-Spaghetti.png"))
    ## Note: the name of the file has spaces between the different values of the list parameters (e.g. "[0.8, 0.6]" instead of "[0.8,0.6]" which may not be the most convenient...)


#----- 2) Now show shaded distribution bands
# Define the two percentiles set that should be shown as shaded area of the theta evolution over replications
# Use 0 and 100 to show the min and max bands respectively
percentiles_low = [0, 25]; percentiles_upp = [100, 75]
# Transparency to use for each shaded range (e.g. 80% and 50% in this order... the larger alpha, the less the transparency) generated with fill_between()
alphas = [0.2, 0.25]
fontsize = 20
colormaps = dict({'1-true': "Blues", '2-fv': "Greens", '3-mc': "Reds"})
color_normalization = matplotlib.colors.Normalize()    # matplotlib.colors.LogNorm()
plot_each_curve = False  # Whether to plot each learning curve in the bands plot as well
savefig = False
show_title = False #not savefig

fig = plt.figure()
axes = fig.subplots(1, len(results))
n_replications = max(results['2-fv'].replication)
n_learning_steps = max(results['2-fv'].t_learn)
theta_max = np.max(np.squeeze(K_true))
for m, method in enumerate(sorted(results.keys())):
    df_learning = results[method]

    # Axis associated to the method where the parameters evolution is plotted
    ax_params = axes[m]
    linecolor = colormaps[method].lower()[:-1]  # e.g. "Greens" -> "green"

    #-- Distribution of theta values by learning step (t_learn)
    # We compute this for every theta in the theta parameter vector
    for t in range(len(df_learning['theta'].iloc[0])):
        df_learning['param'] = [theta[t] for theta in df_learning['theta']]
        results_grp = df_learning.groupby(['t_learn'])
        results_grp_hist = results_grp.apply( lambda df: np.histogram(df['param']) )
        results_grp_agg = results_grp[['param']].agg(['count', 'mean', 'min', 'median', 'max', 'std'] +
                                                                    [percentile(p) for p in percentiles_low] + [percentile(p) for p in percentiles_upp])

        plot_two_bands(fig, axes[m], results_grp_agg, percentiles_low, percentiles_upp, alphas, linecolor,
                       param_var='param', ymin=-1.1, ymax=np.max(np.squeeze(K_true)), show_title=show_title)

    # Linestyles are sorted so that more continuous line means larger rho
    linestyles_ref = ['-', '--', '-.', ':']
    # Rank of rhos in reversed order (largest value has highest rank so that the largest value gets the most solid line in the plot)
    rank_rhos = list(reversed(np.array(rhos).argsort().argsort()))
    linestyles = [linestyles_ref[r] for r in rank_rhos]
    reflines = []
    legend_lines = []
    legend_reflines = []
    for i in range(len(theta_true)):
        linestyle_i = linestyles[i]
        refline_i = ax_params.axhline(theta_true[i], color='black', linestyle=linestyle_i)
        reflines += [refline_i]
        legend_lines += ["Theta_" + str(i + 1)]
        legend_reflines += ["True Optimum Theta_" + str(i + 1)]

    # Finalize plot
    ax_params.set_xlim((0, n_learning_steps))

if savefig:
    plt.get_current_fig_manager().window.showMaximized()
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.11, right=0.9, hspace=0.2, wspace=0.2)
    ## Note: the adjustment values were taken by maximizing the space use in the graph via the figure window (generated by Pycharm Qt5Agg) and selecting "tight layout" from the adjustable knobs icon)
    fig.savefig(os.path.join(resultsdir, f"LossNetwork-K={K},costs={blocking_costs},rhos={rhos},lambdas={lambdas},JF={J_factors},NT_values={NT_values}-Bands.png"))
    ## Note: the name of the file has spaces between the different values of the list parameters (e.g. "[0.8, 0.6]" instead of "[0.8,0.6]" which may not be the most convenient...)
#---------------------------- Loss Network --------------------------
