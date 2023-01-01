# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:04:13 2022

@author: Daniel Mastropietro
@description: Reads FV results and plots them
"""

import runpy
runpy.run_path('../../setup.py')

import os

import numpy as np
import pandas as pd

from run_FV import plot_results_fv_mc


resultsdir = "../../RL-002-QueueBlocking/results"


#-- Execution parameters whose results should be read
# Store all the parameter combinations used in a data frame
df_params = pd.DataFrame.from_dict({'K': [20, 20,
                                          40],
                                    'J': [ 8, 16,
                                           4],
                                    'N': [157, 87,
                                          3194],
                                    'T': [106, 4374,
                                          185]})

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
    results_file_fv_N = os.path.join(resultsdir, "estimates_vs_N_IDK{}N_K={},J={},T=[{}]_results.csv" \
                                                .format(K, K, J, T))
    results_file_fv_T = os.path.join(resultsdir, "estimates_vs_T_IDK{}T_K={},J={},N=[{}]_results.csv" \
                                                .format(K, K, J, N))
    results[case] = dict({'N': pd.read_csv(results_file_fv_N),
                            'T': pd.read_csv(results_file_fv_T)})
    xmax = np.max([xmax, np.nanmax(results[case]['N']['N']),
                         np.nanmax(results[case]['T']['T'])])
    ymax = np.max([ymax, np.nanmax(results[case]['N']['Pr(FV)']), np.nanmax(results[case]['N']['Pr(MC)']),
                         np.nanmax(results[case]['T']['Pr(FV)']), np.nanmax(results[case]['T']['Pr(MC)'])])
    print("xmax = {}, ymax = {}".format(xmax, ymax))

    # Notes about the plots generated below:
    # - The columns defined in parameters `x` and `x2` are grouping variables that define each violin plot
    # - If violins are NOT always shown it's because the probability could not be estimated for some replications

xmax = xmax * 1.1
ymax = ymax * 1.1 * 100     # Recall the vertical axis is shown in percentage!!
xmax = None
ymax = None
for row in df_params.iterrows():
    case = row[0]
    params = row[1]
    K = params['K']
    J = params['J']
    N = params['N']
    T = params['T']

    #-- Plot results as a function of N
    nservers = results[case]['N'].iloc[0]['nservers']
    burnin_time_steps = results[case]['N'].iloc[0]['burnin_time_steps']
    min_num_cycles_for_expectations = results[case]['N'].iloc[0]['min_n_cycles']
    axes = plot_results_fv_mc(results[case]['N'], x="N", x2="#Cycles(MC)_mean",
                              xlabel="# particles", xlabel2="# Return Cycles to state s={}".format(J - 1),
                              xmax=xmax,
                              ymin=0.0, ymax=ymax, plot_mc=True, splines=False,
                              subset=~np.isnan(results[case]['N']['Pr(FV)']),
                              plot_violin_only=True,
                              title="nservers={}, K={}, J={} (J/K={:.0f}%), T={}, # Burn-in Steps={}, Min # Cycles for expectations={}" \
                              .format(nservers, K, J, J/K*100, T, burnin_time_steps, min_num_cycles_for_expectations))

    #-- Plot results as a function of T
    nservers = results[case]['T'].iloc[0]['nservers']
    burnin_time_steps = results[case]['T'].iloc[0]['burnin_time_steps']
    min_num_cycles_for_expectations = results[case]['T'].iloc[0]['min_n_cycles']
    axes = plot_results_fv_mc(results[case]['T'], x="T", x2="#Cycles(MC)_mean",
                              xlabel="# arrival events", xlabel2="# Return Cycles to state s={}".format(J - 1),
                              xmax=xmax,
                              ymin=0.0, ymax=ymax, plot_mc=True, splines=False,
                              subset=~np.isnan(results[case]['T']['Pr(FV)']),
                              plot_violin_only=True,
                              title="nservers={}, K={}, J={} (J/K={:.0f}%), N={}, # Burn-in Steps={}, Min # Cycles for expectations={}" \
                              .format(nservers, K, J, J/K*100, N, burnin_time_steps, min_num_cycles_for_expectations))
