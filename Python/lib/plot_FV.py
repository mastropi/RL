# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:04:13 2022

@author: Daniel Mastropietro
@description: Reads FV results and plots them
"""

import runpy
runpy.run_path('../../setup.py')
#raise KeyboardInterrupt    # Use this to stop the execution here and keep working on the console

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_FV import plot_results_fv_mc


#=================================== SINGLE SERVER ==================================
queue_system = "single-server"
resultsdir = "../../RL-002-QueueBlocking/results/FV/" + queue_system


#-- Execution parameters whose results should be read
# The values of the execution parameters are present in the names of the result files
# Store all the parameter combinations (K, J, N) and (K, J, T) that we want to show in plots, as follows:
# - For each (K, J, N) combination, a plot vs. T is shown.
# - For each (K, J, T) combination, a plot vs. N is shown.
# Use 0 for a particular parameter if there is NO result file for the K, J combination
# (e.g. K=20, J=8, N=0 indicates that there is no result for K=20, J=8 and an analysis of convergence by T (for a fixed value of N))
# (do NOT use NaN because that makes all the values in the data frame be interpreted as float (adding a `.0` at the end (e.g. 20.0)
# and this breaks reading the results file --because values are shown with no decimals in the file names)
# Entries `saveN` and `saveT` indicate whether the corresponding results for fixed N and for fixed T should be saved to a png file, respectively.
df_params = pd.DataFrame.from_dict({'K': [20, 20,
                                          40, 40,
                                              40],
                                    'J': [ 8, 16,
                                           4, 16,
                                              16],
                                    'N': [157, 87,
                                          3194, 372,
                                                5938],
                                    'T': [106, 4374,
                                          185, 1944,
                                                0],
                                    'saveN': [False, False,
                                              False, False,
                                                     False],
                                    'saveT': [False, False,
                                              False, False,
                                                     False]})
# Plots for the QUESTA paper (2023)
# We would like to show convergence results against one parameter (N or T) for different % error levels of the OTHER parameter (T or N)
df_params = pd.DataFrame.from_dict({'K': [20, 20, 20, 20,
                                          40, 40, 40],
                                    'J': [ 12, 12, 12, 12,
                                           12, 12, 12],
                                    'N': [264, 30, 11, 66,
                                          3051, 1356, 489],
                                    'T': [4157, 462, 167, 0,
                                          4157, 462, 167],
                                    'saveN': [False, False, False, False,
                                              False, False, False],
                                    'saveT': [True, False, False, False,
                                              True, False, False]})


# Minimum number of particles N and of arrival steps T to plot (to avoid clogging of the plot)
Nmin = 100
Tmin = 500
show_title = False
fontsize = 20
figsize = (10, 10)  # Figure size as if only the FV results were plotted. When we include the MC results in the plot as well, the first dimension (width) is multiplied by 2

# We store the results in a list of dictionaries of data frames
results = [dict()]*df_params.shape[0]
# We first compute the maximum values of x and of y across ALL parameter settings analyzed
xmax = 0
ymax = 0
for row in df_params.iterrows():
    case = row[0]
    params = row[1]
    K = params['K']
    J = params['J']
    N = params['N']
    T = params['T']

    print("Plotting case {} of {}...".format(row[0]+1, df_params.shape[0]))
    print("\tK={}, J={} (J/K={:.2f}), N={}, T={}".format(K, J, J/K, N, T))
    # Read the results
    if T > 0:
        # For the results as a function of N we need a fixed non-NaN positive value for T (note that `np.nan > 0` is False)
        results_file_fv_N = os.path.join(resultsdir, "estimates_vs_N_IDK{}N_K={},J={},T=[{}]_results.csv" \
                                                    .format(K, K, J, T))
    if N > 0:
        # For the results as a function of T we need a fixed non-NaN positive value for N (note that `np.nan > 0` is False)
        results_file_fv_T = os.path.join(resultsdir, "estimates_vs_T_IDK{}T_K={},J={},N=[{}]_results.csv" \
                                                    .format(K, K, J, N))
    results[case] = dict({  'analysisByN': None if np.isnan(T) else pd.read_csv(results_file_fv_N),
                            'analysisByT': None if np.isnan(N) else pd.read_csv(results_file_fv_T)})
    xmax = np.max([xmax, np.nanmax(np.nan if results[case]['analysisByN'] is None else results[case]['analysisByN']['N']),
                         np.nanmax(np.nan if results[case]['analysisByT'] is None else results[case]['analysisByT']['T'])])
    ymax = np.max([ymax, np.nanmax(np.nan if results[case]['analysisByN'] is None else results[case]['analysisByN']['Pr(FV)']),
                         np.nanmax(np.nan if results[case]['analysisByN'] is None else results[case]['analysisByN']['Pr(MC)']),
                         np.nanmax(np.nan if results[case]['analysisByT'] is None else results[case]['analysisByT']['Pr(FV)']),
                         np.nanmax(np.nan if results[case]['analysisByT'] is None else results[case]['analysisByT']['Pr(MC)'])])
    print("xmax = {}, ymax = {}".format(xmax, ymax))

    # Notes about the plots generated below:
    # - The columns defined in parameters `x` and `x2` are grouping variables that define each violin plot
    # - If violins are NOT always shown it's because the probability could not be estimated for some replications

xmax = xmax * 1.1
ymax = ymax * 1.1 * 100     # Recall the vertical axis is shown in percentage!!
xmax = None
# If needed, override the ymax value in order to make the plots comparable across different analyses (e.g. vs. N and vs. T)
ymax = None
#ymax = 0.40 * 1.1
#ymax = 0.0002 * 1.1
for row in df_params.iterrows():
    case = row[0]
    params = row[1]
    K = params['K']
    J = params['J']
    N = params['N']
    T = params['T']
    saveN = params['saveN']
    saveT = params['saveT']
    if show_title:
        titleByT = "nservers={}, K={}, J={} (J/K={:.0f}%), N={}, # Burn-in Steps={}, Min # Cycles for expectations={}" \
                    .format(nservers, K, J, J / K * 100, N, burnin_time_steps, min_num_cycles_for_expectations)
        titleByN = "nservers={}, K={}, J={} (J/K={:.0f}%), T={}, # Burn-in Steps={}, Min # Cycles for expectations={}" \
                    .format(nservers, K, J, J / K * 100, T, burnin_time_steps, min_num_cycles_for_expectations)
    else:
        titleByT = titleByN = ""

    #-- Plot results as a function of N (for which we need a positive fixed value of T)
    if T > 0:
        mask = results[case]['analysisByN']['N'] >= Nmin
        nservers = results[case]['analysisByN'].iloc[0]['nservers']
        rhos = results[case]['analysisByN'].iloc[0]['rhos']
        burnin_time_steps = results[case]['analysisByN'].iloc[0]['burnin_time_steps']
        min_num_cycles_for_expectations = results[case]['analysisByN'].iloc[0]['min_n_cycles']
        if saveT:
            figfile = os.path.join(resultsdir, f"FV-ConvN-rhos={rhos},K={K},J={J},T={T}-Violin-2Xaxis.png")
        else:
            figfile = None
        axes = plot_results_fv_mc(results[case]['analysisByN'][mask], x="N", x2="#Cycles(MC)_mean",
                                  xlabel="# particles", xlabel2="# Return Cycles to state x=J-1",
                                  xmax=xmax,
                                  ymin=0.0, ymax=ymax, plot_mc=True, splines=False,
                                  fontsize=fontsize,
                                  subset=~np.isnan(results[case]['analysisByN']['Pr(FV)']),
                                  plot_violin_only=True,
                                  title=titleByN,
                                  figsize=figsize,
                                  figmaximize=True,
                                  figfile=figfile)

    #-- Plot results as a function of T (for which we need a non-NaN positive fixed value of N)
    if N > 0:
        mask = results[case]['analysisByT']['T'] >= Tmin
        nservers = results[case]['analysisByT'].iloc[0]['nservers']
        rhos = results[case]['analysisByT'].iloc[0]['rhos']
        burnin_time_steps = results[case]['analysisByT'].iloc[0]['burnin_time_steps']
        min_num_cycles_for_expectations = results[case]['analysisByT'].iloc[0]['min_n_cycles']
        if saveN:
            figfile = os.path.join(resultsdir, f"FV-ConvT-rhos={rhos},K={K},J={J},N={N}-Violin-2Xaxis.png")
        else:
            figfile = None
        axes = plot_results_fv_mc(results[case]['analysisByT'][mask], x="T", x2="#Cycles(MC)_mean",
                                  xlabel="# arrival events", xlabel2="# Return Cycles to state x=J-1",
                                  xmax=xmax,
                                  ymin=0.0, ymax=ymax, plot_mc=True, splines=False,
                                  fontsize=fontsize,
                                  subset=~np.isnan(results[case]['analysisByT']['Pr(FV)']),
                                  plot_violin_only=True,
                                  title=titleByT,
                                  figsize=figsize,
                                  figmaximize=True,
                                  figfile=figfile)

# If we need to save the plot in a figure that is already opened, we can reference the figure number
# where the plot to be saved has been created and then use plt.savefig().
#fig = plt.figure(1)
#fig.savefig(os.path.join(resultsdir, "FV-ConvN-rho=0.7,K=20,J=12,T=4157,Nmax=4000-Violin-2Xaxis.png"))
#fig = plt.figure(7)
#fig.savefig(os.path.join(resultsdir, "FV-ConvN-rho=0.7,K=40,J=12,T=4157,Nmax=3000-Violin-2Xaxis.png"))


#----------------------
# Plot a heatmap where we see the dependency of the FV estimator of the blocking probability with the expected errors of Phi and E(T_A)
import matplotlib
from matplotlib import pyplot as plt, cm, ticker as mtick
from mpl_toolkits import mplot3d
import scipy
from Python.lib.simulators.queues import compute_nparticles_and_nsteps_for_fv_process_many_settings

df1 = pd.read_csv(os.path.join(resultsdir, "estimates_vs_N_IDK20N_K=20,J=12,T=ALLTOGETHER_results.csv"))
df2 = pd.read_csv(os.path.join(resultsdir, "estimates_vs_N_IDK40N_K=40,J=12,T=ALLTOGETHER_results.csv"))

toplot = pd.concat([df1, df2])

# Compute the NT required values for each analyzed errors in the experiments (see NT_required.xlsx)
df_NT_required = compute_nparticles_and_nsteps_for_fv_process_many_settings(   rhos=[0.7],
                                                                               K_values=[5, 10, 20, 40],
                                                                               JF_values=np.arange(0.1, 0.9, 0.1),
                                                                               error_rel=[0.05] + list(np.arange(0.1, 1.1, 0.1)))
# Add the error values to the dataset to plot
toplot = pd.merge(toplot,
         df_NT_required[['K', 'J', 'N', 'error']],
         left_on=['K', 'J', 'N'],
         right_on=['K', 'J', 'N'])
toplot.rename({'error': 'error_phi'}, axis='columns', inplace=True)

toplot = pd.merge(toplot,
         df_NT_required[['K', 'J', 'T', 'error']],
         left_on=['K', 'J', 'T'],
         right_on=['K', 'J', 'T'])
toplot.rename({'error': 'error_et'}, axis='columns', inplace=True)

# Estimation error
#####  IMPORTANT: THIS STEP HAS A HIGH IMPACT ON HOW THE HEATMAP IS SEEN!! #####
# EITHER SET MISSING VALUES OF Pr(FV) TO ZERO OR NOT... (do not set it to 0 if we want to take into account those replications where we actually estimated something reliable...)
#toplot['Pr(FV)'].fillna(0, inplace=True)
#####  IMPORTANT #####
toplot['error_fv_rel'] = toplot['Pr(FV)'] / toplot['Pr(K)'] - 1
toplot['error_fv_rel_abs'] = np.abs(toplot['error_fv_rel'])
#axes = plt.figure().subplots(1,2)
#axes[0].hist(np.log2(1 + np.abs(toplot['error_fv_rel'])), bins=30); axes[0].set_title("Histogram of Absolute relative error (log2 scale)"); axes[0].axvline(1)  # We use log2 and not log10 so that when the absolute relative error is 1 (100%) the result is also 1! :)
#axes[1].hist(-np.log10(toplot['Pr(FV)'] + 1E-20), bins=30, color="red"); axes[1].set_title("Histogram of Pr(FV) (-log10 scale)") # We use log2 and not log10 so that when the absolute relative error is 1 (100%) the result is also 1! :)

# Summarize error by each error combination
toplot_agg = toplot.groupby(['K', 'J', 'N', 'T', 'error_et', 'error_phi'])[['Pr(FV)', 'Pr(K)', 'error_fv_rel', 'error_fv_rel_abs']].agg(['count', 'mean', 'min', 'median', 'max', 'std'])
toplot_agg.reset_index(inplace=True)
#toplot_agg[['error_et', 'error_phi', 'Pr(FV)']]
#toplot_agg['Pr(FV)']['mean']
# Extract the mean of each analyzed measure (Pr(FV) and error_fv_rel)
#toplot_agg_error = pd.concat([toplot_agg[['error_et', 'error_phi']], toplot_agg.xs('mean', axis=1, level=1)], axis=1)   # Not quite
toplot_agg_mean = pd.concat([toplot_agg.xs('K', axis=1), toplot_agg.xs('J', axis=1), toplot_agg.xs('N', axis=1), toplot_agg.xs('T', axis=1),
                             toplot_agg.xs('error_et', axis=1), toplot_agg.xs('error_phi', axis=1), toplot_agg.xs('mean', axis=1, level=1),
                             toplot_agg.xs(('Pr(FV)', 'count'), axis=1)],
                            axis=1)
# Compute the absolute error of the AVERAGE of Pr(FV) (this is the same as the absolute of the error_fv_rel --which is DIFFERENT from the value stored in error_fv_rel_abs because this averages the absolute values!)
toplot_agg_mean['error_fvmean_rel_abs'] = np.abs(toplot_agg_mean['error_fv_rel'])
toplot_agg_mean['n'] = toplot_agg_mean[('Pr(FV)', 'count')]
toplot_agg_median = pd.concat([ toplot_agg.xs('K', axis=1), toplot_agg.xs('J', axis=1), toplot_agg.xs('N', axis=1), toplot_agg.xs('T', axis=1),
                                toplot_agg.xs('error_et', axis=1), toplot_agg.xs('error_phi', axis=1), toplot_agg.xs('median', axis=1, level=1),
                                toplot_agg.xs(('Pr(FV)', 'count'), axis=1)],
                                axis=1)
toplot_agg_median['error_fvmean_rel_abs'] = np.abs(toplot_agg_median['error_fv_rel'])
toplot_agg_median['n'] = toplot_agg_median[('Pr(FV)', 'count')]
plt.figure(); plt.hist(toplot_agg_mean['error_fv_rel_abs'], bins=30, alpha=0.5); plt.hist(toplot_agg_mean['error_fvmean_rel_abs'], bins=30, alpha=0.5)
## OK: the orange bars are closer to the origin than the blue bars

# Compute the overall mean and median (i.e. NOT by K)
toplot_agg_overall = toplot.groupby(['J', 'error_et', 'error_phi'])[['Pr(FV)', 'Pr(K)', 'error_fv_rel', 'error_fv_rel_abs']].agg(['count', 'mean', 'min', 'median', 'max', 'std'])
toplot_agg_overall.reset_index(inplace=True)
toplot_agg_overall[('Pr(FV)', 'cv')] = toplot_agg_overall[('Pr(FV)', 'std')] / np.sqrt(toplot_agg_overall[('Pr(FV)', 'count')]) / toplot_agg_overall[('Pr(FV)', 'mean')]

toplot_agg_overall_mean = pd.concat([toplot_agg_overall.xs('J', axis=1),
                                     toplot_agg_overall.xs('error_et', axis=1), toplot_agg_overall.xs('error_phi', axis=1), toplot_agg_overall.xs('mean', axis=1, level=1),
                                     toplot_agg_overall.xs(('Pr(FV)', 'count'), axis=1)],
                                    axis=1)
toplot_agg_overall_mean['error_fvmean_rel_abs'] = np.abs(toplot_agg_overall_mean['error_fv_rel'])
toplot_agg_overall_mean['n'] = toplot_agg_overall_mean[('Pr(FV)', 'count')]

toplot_agg_overall_median = pd.concat([ toplot_agg_overall.xs('J', axis=1),
                                        toplot_agg_overall.xs('error_et', axis=1), toplot_agg_overall.xs('error_phi', axis=1), toplot_agg_overall.xs('median', axis=1, level=1),
                                        toplot_agg_overall.xs(('Pr(FV)', 'count'), axis=1)],
                                        axis=1)
toplot_agg_overall_median['error_fvmean_rel_abs'] = np.abs(toplot_agg_overall_median['error_fv_rel'])
toplot_agg_overall_median['n'] = toplot_agg_overall_median[('Pr(FV)', 'count')]

# Coefficient of Variation of the estimated mean
toplot_agg_overall_cv = pd.concat([toplot_agg_overall.xs('J', axis=1),
                                     toplot_agg_overall.xs('error_et', axis=1), toplot_agg_overall.xs('error_phi', axis=1), toplot_agg_overall.xs(('Pr(K)', 'mean'), axis=1), toplot_agg_overall.xs('cv', axis=1, level=1),
                                     toplot_agg_overall.xs(('Pr(FV)', 'count'), axis=1)],
                                    axis=1)
toplot_agg_overall_cv['Pr(K)'] = toplot_agg_overall_cv[('Pr(K)', 'mean')]
toplot_agg_overall_cv['cv'] = toplot_agg_overall_cv['Pr(FV)']
toplot_agg_overall_cv['n'] = toplot_agg_overall_cv[('Pr(FV)', 'count')]




# Mapping of axes in the below plots:
# x <- error_et
# y <- error_phi
# Do plots for every K value (in order to analyze how the relationship with the error in Pr(FV) changes with K
# (its change should be negligible...)
def plot_error_analysis(toplot_agg_stat,
                        toplot_agg,
                        xvar="error_et", yvar="error_phi",
                        xlabel="% Error E(T_A) --> Decreasing # arrival events (T)", ylabel="% Error Phi(K) --> Decreasing # particles (N)",
                        analysis_by_K=True,
                        K_values=None,          # K values considered for each error analysis
                        p_values=None,          # Pr(K) values associated to the K values considered for each analysis
                        contour_levels=None,    # List or array containing the levels to show in the contour plot. If None, the levels are defined as `np.linspace(vmin, vamx, 5) where vmin and vmax are the minimum and maximum values of the quantity defining the heatmap.
                        plot_what="error",
                        plot_scale="log",
                        plot_times_instead_of_error=True,
                        plot_types=["colormesh"],
                        cmap="jet",
                        error_max=9,    # Maximum value that should be used for the maximum value associated to the error plots... This is used so that the color scale shown in the plot is informative; when error_max = 9, it means we plot maximum a Pr(FV) estimated 10 times larger than the true Pr(K) value.
                        fontsize=12):
    if analysis_by_K:
        probas_true = np.unique(toplot_agg_stat[['K', 'Pr(K)']], axis=0)
    else:
        probas_true = [np.mean(toplot_agg_stat['Pr(K)'])]

    for itervalues in probas_true:
        if analysis_by_K:
            K, ptrue = itervalues
            mask = toplot_agg_stat['K'] == K
        else:
            ptrue = itervalues
            mask = [True]*toplot_agg_stat.shape[0]

        # Convert the data frame to a table of the values to plot so that we can create a heatmap and a contour plot
        if plot_what == "cv":
            toplot_agg_table_cv = toplot_agg_stat[mask].pivot(xvar, yvar, 'cv')
        else:
            toplot_agg_table_error = toplot_agg_stat[mask].pivot(xvar, yvar, 'error_fv_rel')
            toplot_agg_table_error_abs = toplot_agg_stat[mask].pivot(xvar, yvar, 'error_fv_rel_abs')
            toplot_agg_table_errormean_abs = toplot_agg_stat[mask].pivot(xvar, yvar, 'error_fvmean_rel_abs')
            toplot_agg_table_proba = toplot_agg_stat[mask].pivot(xvar, yvar, 'Pr(FV)')

        # Plots
        if plot_what == "error":
            toplot_table = toplot_agg_table_error
            zvar = "error_fv_rel"
            zlabel = "Rel. error Pr(FV) (with sign)"

            # Min and max values for the color scale
            value_min = -1.0
            # Make the plots comparable by using the same maximum for the colormap (note that the minimum is common to all cases (-1 for errors and 0 for the probability))
            # Note that the maximum value is NOT filtered to the currently analyzed K! (i.e. this guarantees that the scale is the same ACROSS all K values plotted)
            value_max = min(np.ceil(np.max(toplot_agg_stat[zvar])), error_max)

            # What to plot and on what scale
            if plot_times_instead_of_error:
                toplot_table = toplot_table + 1
                value_min = value_min + 1
                value_max = value_max + 1
                zlabel = "Pr(FV) / Pr(K)"
            if plot_scale == "log":
                toplot_table = np.sign(toplot_table) * np.log2(
                    1 + np.abs(toplot_table))  # We use log2() instead of log10() so that 1 is mapped to 1
                value_min = np.sign(value_min) * np.log2(1 + value_min)
                value_max = np.sign(value_max) * np.log2(1 + value_max)
                zlabel = zlabel + " (log2)"

            # Colormap normalization that honour the vmin and vmax values
            color_norm = matplotlib.colors.Normalize(vmin=value_min,
                                                     vmax=value_max)  # The minimum relative error is 100% as the estimate cannot go below 0 (in which case we have `0/Pr(K) - 1` as relative error)
        elif plot_what == "error_abs":
            toplot_table = toplot_agg_table_error_abs
            zvar = "error_fv_rel"
            zlabel = "Abs. Rel. error Pr(FV)"

            # Min and max values for the color scale
            value_min = 0.0
            # Make the plots comparable by using the same maximum for the colormap (note that the minimum is common to all cases (-1 for errors and 0 for the probability))
            # Note that the maximum value is NOT filtered to the currently analyzed K! (i.e. this guarantees that the scale is the same ACROSS all K values plotted)
            value_max = np.ceil(np.max(toplot_agg_stat[zvar]))

            # What scale to use
            if plot_scale == "log":
                toplot_table = np.sign(toplot_table) * np.log2(
                    1 + np.abs(toplot_table))  # We use log2() instead of log10() so that 1 is mapped to 1
                value_min = np.sign(value_min) * np.log2(1 + value_min)
                value_max = np.sign(value_max) * np.log2(1 + value_max)
                zlabel = zlabel + " (log2)"

            # Colormap normalization that honour the vmin and vmax values
            color_norm = matplotlib.colors.Normalize(vmin=value_min, vmax=value_max)
        elif plot_what == "errormean_abs":
            toplot_table = toplot_agg_table_errormean_abs
            zvar = "error_fv_rel"
            zlabel = "Abs. Rel. error mean(Pr(FV))"

            # Min and max values for the color scale
            value_min = 0.0
            # Make the plots comparable by using the same maximum for the colormap (note that the minimum is common to all cases (-1 for errors and 0 for the probability))
            # Note that the maximum value is NOT filtered to the currently analyzed K! (i.e. this guarantees that the scale is the same ACROSS all K values plotted)
            value_max = np.ceil(np.max(toplot_agg_stat[zvar]))

            # What scale to use
            if plot_scale == "log":
                toplot_table = np.sign(toplot_table) * np.log2(
                    1 + np.abs(toplot_table))  # We use log2() instead of log10() so that 1 is mapped to 1
                value_min = np.sign(value_min) * np.log2(1 + value_min)
                value_max = np.sign(value_max) * np.log2(1 + value_max)
                zlabel = zlabel + " (log2)"

            # Colormap normalization that honour the vmin and vmax values
            color_norm = matplotlib.colors.Normalize(vmin=value_min, vmax=value_max)
        elif plot_what == "cv":
            toplot_table = toplot_agg_table_cv
            zvar = "cv"
            zlabel = "Coefficient of Variation of Pr(FV)"

            # Min and max values for the color scale
            value_min = 0.0
            # Make the plots comparable by using the same maximum for the colormap (note that the minimum is common to all cases (-1 for errors and 0 for the probability))
            # Note that the maximum value is NOT filtered to the currently analyzed K! (i.e. this guarantees that the scale is the same ACROSS all K values plotted)
            value_max = np.ceil(np.max(toplot_agg_stat[zvar]))

            # What scale to use
            if plot_scale == "log":
                toplot_table = np.sign(toplot_table) * np.log2(1 + np.abs(toplot_table))  # We use log2() instead of log10() so that 1 is mapped to 1
                value_min = np.sign(value_min) * np.log2(1 + value_min)
                value_max = np.sign(value_max) * np.log2(1 + value_max)
                zlabel = zlabel + " (log2)"

            # Colormap normalization that honour the vmin and vmax values
            color_norm = matplotlib.colors.Normalize(vmin=value_min, vmax=value_max)
        else:
            # The default is to plot the probability estimate
            toplot_table = toplot_agg_table_proba
            zvar = "Pr(FV)"
            zlabel = zvar

            # Min and max values for the color scale
            value_min = 0.0
            # Make the plots comparable by using the same maximum for the colormap (note that the minimum is common to all cases (-1 for errors and 0 for the probability))
            # Note that the maximum value is NOT filtered to the currently analyzed K! (i.e. this guarantees that the scale is the same ACROSS all K values plotted)
            value_max = np.ceil(np.max(toplot_agg_stat[zvar]))

            # What scale to use
            if plot_scale == "log":
                toplot_table = -np.log10(toplot_table + 1E-20)
                value_min = -np.log10(value_min + 1E-20)
                value_max = -np.log10(value_max + 1E-20)
                zlabel = zlabel + " (-log10)"

            # Colormap normalization that honour the vmin and vmax values
            color_norm = matplotlib.colors.Normalize(vmin=value_min, vmax=value_max)

        if analysis_by_K:
            title = "Plot of {}, K = {:.0f} (True Pr(K) = {:.3g})".format(zlabel, K, ptrue)
        else:
            title = "Plot of {}, Average True Pr(K) = {:.3g}, Pr(K) values = {}, K values = {}" \
                .format(zlabel, probas_true[0], "" if p_values is None else ["{:.3g}%".format(p*100) for p in sorted(p_values, reverse=True)], "" if K_values is None else sorted(K_values))

        colormap = cm.get_cmap(cmap)
        if "surface" in plot_types:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')

            pc = ax1.contour(toplot_table.index, toplot_table.columns, toplot_table.T, norm=color_norm,
                             cmap=colormap)  # We transpose the table values Z because the X dimension go along the columns of Z
            # ax1.plot(toplot_agg[xvar], toplot_agg[yvar], '.', color="black", markersize=12)
            ax1.scatter(toplot_agg[xvar], toplot_agg[yvar], s=toplot_agg[('Pr(FV)', 'count')], color="black")
            ax1.set_xlabel(xlabel, fontsize=fontsize)
            ax1.set_ylabel(ylabel, fontsize=fontsize)
            ax1.set_xlim((0, 1.1))
            ax1.set_ylim((0, 1.1))
            ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            plt.colorbar(pc, ax=ax1)

            # Prepare the data for the surface plot (we need to have X and Y as 2D arrays, like those generated by np.meshgrid())
            # In fact, the code below is the same as doing xx, yy = np.meshgrid(x, y, indexing='ij')
            nx = toplot_table.shape[0]
            ny = toplot_table.shape[1]
            x = np.array(toplot_table.index)
            y = np.array(toplot_table.columns)
            xx = np.repeat(x, ny).reshape(nx, ny)
            yy = np.repeat(y, nx).reshape(nx, ny, order='F')
            ax2.plot_surface(xx, yy, toplot_table)
            plt.plot(toplot_agg[xvar], toplot_agg[yvar], '.', color="black", markersize=12)
            ax2.set_xlabel(xlabel, fontsize=fontsize)
            ax2.set_ylabel(ylabel, fontsize=fontsize)
            ax2.set_xlim((0, 1.1))
            ax2.set_ylim((0, 1.1))
            ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax2.set_zlabel(zlabel, fontsize=fontsize)

            plt.suptitle(title)

        if "colormesh" in plot_types:
            plt.figure()
            plt.pcolormesh(toplot_table.index, toplot_table.columns, toplot_table.T, shading='gouraud', cmap=colormap,
                           norm=color_norm)
            plt.xlabel(xlabel, fontsize=fontsize)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.gca().set_xlim((0, 1.1))
            plt.gca().set_ylim((0, 1.1))
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            plt.title(title)
            plt.colorbar()
            # Add the points where measurements are done
            # plt.plot(toplot_agg[xvar], toplot_agg[yvar], '.', color="white", markersize=12)
            plt.scatter(toplot_agg[xvar], toplot_agg[yvar], s=toplot_agg[('Pr(FV)', 'count')]**2, color="black", alpha=0.5)
            plt.scatter(toplot_agg[xvar], toplot_agg[yvar], s=toplot_agg[('Pr(FV)', 'count')]**2, color="white", alpha=0.5)
            ct = plt.contour(toplot_table.index, toplot_table.columns, toplot_table.T,
                             levels=np.linspace(value_min, value_max, 5) if contour_levels is None else contour_levels,
                             colors="white")  # We transpose the table values Z because the X dimension go along the columns of Z
            plt.clabel(ct, fmt="%0.1f")  # clabel() specifically adds labels to levels of a contour plot
            ## On adding labels to the contour plot: https://stackoverflow.com/questions/25873681/matplotlib-contour-plot-labels-overlap-axes
            # Add the y = x line
            plt.plot([value_min, value_max], [value_min, value_max], color="black", alpha=0.5)
            plt.plot([value_min, value_max], [value_min, value_max], color="white", alpha=0.5)
            # Use a square aspect ratio
            plt.gca().set_aspect('equal')


plot_error_analysis(toplot_agg_overall_median,
                    toplot_agg_overall,
                    analysis_by_K=False,
                    K_values=[int(K) for K in np.unique(toplot['K'])],
                    p_values=np.unique(toplot['Pr(K)']),
                    contour_levels=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
                    plot_what = "error", # "error", "error_abs", "errormean_abs", "proba" (or anything else is equivalent to "proba")
                    plot_scale = "log", # "linear", "log"
                    plot_times_instead_of_error = True,  # When plot_what = "error" (error with sign) Whether to plot how many times LARGER is Pr(FV) w.r.t Pr(K) instead of the error (as it is more intuitive)
                    plot_types = ["colormesh"], #["colormesh", "surface"] # list of one or both of "colormesh", "surface"
                    cmap = "jet",
                    error_max = 9,
                    fontsize=12)

# Coefficient of Variations for each estimate (so that we know how reliable they are)
plot_error_analysis(toplot_agg_overall_cv,
                    toplot_agg_overall,
                    analysis_by_K=False,
                    K_values=[int(K) for K in np.unique(toplot['K'])],
                    p_values=np.unique(toplot['Pr(K)']),
                    contour_levels=None,
                    plot_what = "cv", # "error", "error_abs", "errormean_abs", "proba" (or anything else is equivalent to "proba")
                    plot_scale = "log", # "linear", "log"
                    plot_times_instead_of_error = True,  # When plot_what = "error" (error with sign) Whether to plot how many times LARGER is Pr(FV) w.r.t Pr(K) instead of the error (as it is more intuitive)
                    plot_types = ["colormesh"], #["colormesh", "surface"] # list of one or both of "colormesh", "surface"
                    cmap = "jet",
                    error_max = 9,
                    fontsize=12)

# Plot split by K
plot_error_analysis(toplot_agg_median,
                    toplot_agg,
                    analysis_by_K=True,
                    K_values=[int(K) for K in np.unique(toplot['K'])],
                    p_values=np.unique(toplot['Pr(K)']),
                    contour_levels=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
                    plot_what = "error", # "error", "error_abs", "errormean_abs", "proba" (or anything else is equivalent to "proba")
                    plot_scale = "log", # "linear", "log"
                    plot_times_instead_of_error = True,  # When plot_what = "error" (error with sign) Whether to plot how many times LARGER is Pr(FV) w.r.t Pr(K) instead of the error (as it is more intuitive)
                    plot_types = ["colormesh"], #["colormesh", "surface"] # list of one or both of "colormesh", "surface"
                    cmap = "jet",
                    error_max = 9,
                    fontsize=12)


# If we wanted to show a smoothed version of the plots
# spar = 20
# fs = scipy.interpolate.SmoothBivariateSpline(toplot_agg_mean[xvar], toplot_agg_mean[yvar], toplot_agg_mean[zvar], w=None,
#                                                  bbox=[None, None, None, None], kx=3, ky=3, s=spar, eps=1e-16)
# zz = fs.ev(xx, yy)
# toplot_agg_mean[zvar + "_smooth"] = zz.reshape(-1)
# toplot_agg_table_error_smooth = toplot_agg_mean.pivot(xvar, yvar, zvar + "_smooth")
# toplot_table_smooth = toplot_agg_table_error_smooth
#
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
#
# ax1.contour(toplot_table_smooth.index, toplot_table_smooth.columns, toplot_table_smooth.T)
# ax1.set_xlabel(xvar)
# ax1.set_ylabel(yvar)
# ax2.plot_surface(xx, yy, zz)
# ax2.set_xlabel(xvar)
# ax2.set_ylabel(yvar)
# ax2.set_zlabel(zlabel)
#
# plt.figure()
# plt.pcolormesh(toplot_table_smooth.index, toplot_table_smooth.columns, toplot_table_smooth.T, shading='gouraud')
#=================================== SINGLE SERVER ==================================




#=================================== LOSS NETWORK ==================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_FV import plot_results_fv_mc


queue_system = "loss-network"
resultsdir = "../../RL-002-QueueBlocking/results/FV/" + queue_system
resultsdir = "C:/users/lundinmatlab/Desktop"

#resultsfile = "estimates_vs_N_20230316_014100_K=6,block=[4, 6],costs=[180, 600],lambdas=[1, 5],rho=[0.8, 0.6],J=[0.1, 0.6],T=[500],N_values=[80, 160, 320, 640],ProbStart=True_results.csv"

#resultsfile = "estimates_vs_N_20230316_164010_K=6,block=[4, 6],costs=[180, 600],lambdas=[1, 5],rho=[0.8, 0.6],J=[0.1, 0.6],T=[500],N_values=[80],ProbStart=False_results.csv"
## Only ONE N value

#resultsfile = "estimates_vs_N_20230316_014116_K=6,block=[4, 6],costs=[180, 600],lambdas=[1, 5],rho=[0.8, 0.6],J=[0.5, 0.5],T=[500],N_values=[80, 160, 320, 640],ProbStart=True_results.csv"
#resultsfile = "estimates_vs_N_20230316_110309_K=6,block=[4, 6],costs=[2000, 20000],lambdas=[1, 5],rho=[0.5, 0.3],J=[0.7, 0.7],T=[500],N_values=[40, 80, 160],ProbStart=True_results.csv"

resultsfile = "estimates_vs_T_20230330_233027_K=6,block=[4, 6],costs=[180, 600],lambdas=[1, 5],rho=[0.8, 0.6],J=[0.2, 0.4],N=[100],T_values=[80, 160, 320, 640],ProbStart=False_results.csv"
resultsfile = "estimates_vs_T_20230330_232952_K=6,block=[4, 6],costs=[180, 600],lambdas=[1, 5],rho=[0.8, 0.6],J=[0.2, 0.4],N=[100],T_values=[80, 160, 320, 640],ProbStart=True_results.csv"

#resultsfile = "estimates_vs_N_20230315_201909_K=6,block=[4, 6],rho=[0.8, 0.8],J=[0.5, 0.5],T=[500],N_values=[20, 40, 80],ProbStart=False_results.csv"
## Does NOT have the expected cost variable!


#--------- Read and Plot
results = pd.read_csv(os.path.join(resultsdir, resultsfile))
# Remove missing values from the estimated probabilities
for v in ['Pr(FV)', 'ExpCost(FV)', 'Pr(MC)', 'ExpCost(MC)']:
    results[v].fillna(0, inplace=True)
if "ExpCost" in results.columns:
    true_cost_var = "ExpCost"
elif "ExpCost(K)" in results.columns:
    true_cost_var = "ExpCost(K)"

# Plot results
plot_mc = True
show_title = False
if show_title:
    title = resultsfile
    fontsize = 10
else:
    title = ""
    fontsize = 20
if resultsfile[:14] == "estimates_vs_N":
    xvar = "N"; xlabel = "# particles"
elif resultsfile[:14] == "estimates_vs_T":
    xvar = "T"; xlabel = "# arrival events"
axes = plot_results_fv_mc(  results, x=xvar, x2="#Cycles(MC)_mean",
                            prob_mc="ExpCost(MC)", prob_fv="ExpCost(FV)", prob_true=true_cost_var, multiplier=1,
                            xlabel=xlabel, xlabel2="# Return Cycles to absorption set",
                            ylabel="Expected cost",
                            ymin=0.0, plot_mc=plot_mc, splines=False,
                            fontsize=fontsize,
                            figmaximize=True,
                            title=title)
#=================================== LOSS NETWORK ==================================

