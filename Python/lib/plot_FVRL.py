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

from Python.lib.utils.basic import aggregation_bygroups

resultsdir = "../../RL-002-QueueBlocking/results/RL/single-server"


#-- Execution parameters whose results should be read
theta_true_values = [18.0] #[23.0] #[18.0]
theta_start_values = [28.1] #[34.1] #[28.1]
J_factor_values = [0.3] #[0.3]
error_rel_phi = [0.2] #[1.0] #[0.2]
error_rel_et = [1.0] #[1.0] #[0.2]
suffix = "" #"-DecreasingAlpha"

# Read the results
results_file_fv = os.path.join(resultsdir, "SimulatorQueue_FV-theta0={}-theta={}-J={}-E={},{}{}.csv" \
                                            .format(theta_true_values, theta_start_values, J_factor_values, error_rel_phi, error_rel_et, suffix))
results_file_mc = os.path.join(resultsdir, "SimulatorQueue_MC-theta0={}-theta={}-J={}-E={},{}{}.csv" \
                                            .format(theta_true_values, theta_start_values, J_factor_values, error_rel_phi, error_rel_et, suffix))
results = dict({'fv': pd.read_csv(results_file_fv),
                'mc': pd.read_csv(results_file_mc)})

#----- Auxiliary functions
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

def finalize_plot(fig, ax, method, ymax, ref=0.0, title=""):
    ax.axhline(ref, color="gray", linestyle="dashed")
    ax.set_xlabel("Learning step")
    ax.set_ylabel("theta")
    ax.set_ylim(0, ymax)
    ax.set_title("Method: {}".format(method.upper()))
    fig.suptitle(title_params)

#----- Plot
# Define the two percentiles set that should be shown as shaded area of the theta evolution over replications
# Use 0 and 100 to show the min and max bands respectively
percentiles_low = [10, 25]; percentiles_upp = [90, 75]
# Transparency to use for each shaded range generated with fill_between()
alphas = [0.2, 0.25]
colormaps = dict({'fv': "Greens", 'mc': "Reds"})
color_normalization = matplotlib.colors.Normalize()    # matplotlib.colors.LogNorm()
plot_each_curve = False  # Whether to plot each learning curve as well

fig = plt.figure(); axes = fig.subplots(1,2)
fig2 = plt.figure(); axes2 = fig2.subplots(1,2)
n_replications = max(results['fv'].replication)
theta_max = np.max([ max(results['fv'].theta), max(results['mc'].theta) ])
title_params = "Optimum theta = {}, Theta start = {}, J/K = {}, Rel. Error Phi = {}%, Rel. Error E(T_A) = {}%" \
                 .format([theta - 1 for theta in theta_true_values], theta_start_values, J_factor_values, [e*100 for e in error_rel_phi], [e*100 for e in error_rel_et]) + \
               (suffix == "" and "\n(constant learning rate alpha on " or suffix and "\n(decreasing learning rate alpha on ") + "{} replications)".format(n_replications)
for i, method in enumerate(results.keys()):
    theta_opt_true = theta_true_values[0] - 1   # We subtract 1 because recall that the optimum theta is not precisely `theta_true`, as this value is the sref value on which the exponential cost is centered. For more details see the notes in the costBlockingExponential() function in environments/queues.py
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
    # Doesn't help because axes are categorical in plt.imshow()!
    #data2plot = pd.DataFrame.from_dict({'t': results_fv_grp_hist.index, ''})
    #plt.imshow()

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
    pc = axes[i].pcolormesh(x, y, theta_counts, cmap=colormaps[method], norm=color_normalization)
    cbar = plt.colorbar(pc, ax=axes[i])
    cbar.set_label("% curves")
    # If we want to add vertical lines to indicate each t_learn value... but it usually becomes too clumsy
    #ax.vlines(results[method].t_learn, 0, ax.get_ylim()[1], colors="gray", linewidth=0.1)
    finalize_plot(fig, axes[i], method, theta_max*1.1, ref=theta_opt_true, title=title_params)

    if plot_each_curve:
        plot_curves(axes[i], results[method], color=linecolor)

    # Average line with shaded area
    line_median = axes2[i].plot(results_grp_agg.index, results_grp_agg['theta']['median'], color=linecolor, linewidth=2)[0]    # We need to select the first element because the returned value of plot() is a list!
    line_mean = axes2[i].plot(results_grp_agg.index, results_grp_agg['theta']['mean'], color=linecolor, linewidth=2, linestyle="dashed")[0]    # We need to select the first element because the returned value of plot() is a list!
    fill_between_1 = axes2[i].fill_between(  results_grp_agg.index,
                            #results_grp_agg['theta']['max'],
                            #results_grp_agg['theta']['min'],
                            results_grp_agg['theta']['percentile_{}'.format(percentiles_low[0])],
                            results_grp_agg['theta']['percentile_{}'.format(percentiles_upp[0])],
                            color=linecolor,
                            alpha=alphas[0])
    fill_between_2 = axes2[i].fill_between(  results_grp_agg.index,
                            #results_grp_agg['theta']['mean'] + 2*results_grp_agg['theta']['std'] / np.sqrt(results_grp_agg['theta']['count']),
                            #results_grp_agg['theta']['mean'] - 2*results_grp_agg['theta']['std'] / np.sqrt(results_grp_agg['theta']['count']),
                            results_grp_agg['theta']['percentile_{}'.format(percentiles_low[1])],
                            results_grp_agg['theta']['percentile_{}'.format(percentiles_upp[1])],
                            color=linecolor,
                            alpha=alphas[1])
    # We need to define a proxy Artist for the second shade fille because the two transparency factors are "added"
    # when seeing the color intensity in the graph... therefore using the output of fill_between() as legend object
    # will not convey the actual color of the shade.
    # Ref for creating proxy Artists: https://matplotlib.org/2.0.2/users/legend_guide.html#proxy-legend-handles
    # Note: we can get the objects that can be added to a legend by running `axes2[i].get_legend_handles_labels()`.
    legend_obj_fill2 = mpatches.Patch(color=linecolor, alpha=sum(alphas))
    legend_obj = [line_median, line_mean, fill_between_1, legend_obj_fill2]
    axes2[i].legend(handles=legend_obj,
                    labels=["median curve",
                            "mean curve",
                            "Location of {}% of curves".format(percentiles_upp[0] - percentiles_low[0]),
                            "Location of {}% of curves".format(percentiles_upp[1] - percentiles_low[1])])
    finalize_plot(fig2, axes2[i], method, theta_max*1.1, ref=theta_opt_true, title=title_params)

    if plot_each_curve:
        # Add the curves for each replication but very thin
        plot_curves(axes2[i], results[method], color=linecolor)
