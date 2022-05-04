# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:59:47 2020

@author: Daniel Mastropietro
@description: Run simulations on different lambda and alpha values for TD(lambda)
"""

import runpy
runpy.run_path("../../setup.py")

import os
import numpy as np
from matplotlib import pyplot as plt, cm
import pickle
from timeit import default_timer as timer

#from Python.lib import environments, agents
from Python.lib.environments import gridworlds
from Python.lib.agents import GenericAgent
from Python.lib.agents.policies import random_walks
from Python.lib.agents.learners.episodic.discrete import mc, td
from Python.lib.agents.learners.episodic.discrete import AlphaUpdateType
import Python.lib.simulators as simulators

from test_utils import plot_rmse_by_episode

# Directories
# Use the following directory (relative to the current file's directory) when running the whole script (I think)
#resultsdir = "../../RL-001-MemoryManagement/results/SimulateTDLambda-DifferentLambdas&Adaptive-2022"
# Use the following directory (relative to the project's directory) when running chunk by chunk manually
resultsdir = "./RL-001-MemoryManagement/results/SimulateTDLambda-DifferentLambdas&Adaptive-2022"


############################ EXPERIMENT SETUP #################################
# The environment
nstates = 19 # Number of states excluding terminal states
env = gridworlds.EnvGridworld1D(length=nstates+2)

# What experiments to run
run_mc = False
run_td = True
run_td_adap = True

# Simulation setup
gamma = 1.0

# alpha
alpha_non_adaptive = 0.1
alphas_adaptive = [0.1] #[0.1, 0.2, 0.3]
#alpha_non_adaptive = 1
#alphas_adaptive = [1] #[1, 3, 5]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
adjust_alpha = False
adjust_alpha_by_episode = False
adaptive_type = td.AdaptiveLambdaType.ATD

# lambda
#lambdas = [0.0, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95]
lambdas = [0.0, 0.4, 0.8, 0.9, 0.95]
lambda_min = 0.0
lambda_max = 0.99

seed = 1717
nexperiments = 20
nepisodes = 10
start = int((nstates + 1) / 2)
verbose = True
verbose_period = int( nepisodes / 10 )
debug = False
plot = True

# Possible policies and learners for agents
pol_rw = random_walks.PolRandomWalkDiscrete(env)
if run_mc:
    learner_mc = mc.LeaMCLambda(env, gamma=gamma, lmbda=1.0, alpha=1.0, adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode)
learner_td = td.LeaTDLambda(env, gamma=gamma, adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode,
                            alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT)
learner_td_adap = td.LeaTDLambdaAdaptive(env, gamma=gamma, adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode,
                                         alpha_update_type=AlphaUpdateType.FIRST_STATE_VISIT,
                                         adaptive_type=adaptive_type)


############################ lambda-return #################################
if run_mc:
    # MC (lambda-return)
    learner_mc.setParams()
    agent = GenericAgent(pol_rw, learner_mc)

    # Run the simulation and store the results
    sim = simulators.Simulator(env, agent, seed=seed, debug=debug)
    N_mean, rmse_mean, rmse_se, RMSE_by_episode_mean, RMSE_by_episode_se, learning_info = \
                        sim.simulate(nexperiments=nexperiments,
                                     nepisodes=nepisodes,
                                     start=start,
                                     verbose=verbose,
                                     verbose_period=verbose_period,
                                     plot=plot)
    results_mc = [{'lambda': learner_mc.lmbda,
                   'alpha': learner_mc.alpha,
                   'rmse_mean': rmse_mean,                          # Average of "RMSE at LAST episode" over all experiments
                   'rmse_se': rmse_se,
                   'rmse_by_episode_mean': RMSE_by_episode_mean,    # Average of "RMSE at EACH episode" over all experiments
                   'rmse_by_episode_se': RMSE_by_episode_se,
                   'learning_info': learning_info
                   }]

    # Save
    file = open(resultsdir + "/mc.pickle", mode="wb")  # "b" means binary mode (needed for pickle.dump())
    pickle.dump(results_mc, file)
    file.close()

    # Distribution of state counts
    plt.figure()
    plt.bar(env.all_states, N_mean, color="red", alpha=0.2)
    ax = plt.gca()
    ax.set_xlabel("states")
    ax.set_ylabel("Distribution over non-terminal states")
    ax.spines['left'].set_color("red")
    ax.tick_params(axis='y', colors="red")
    ax.yaxis.label.set_color("red")
    ax2 = ax.twinx()
    ax2.bar(np.array(env.all_states)[[0,nstates+1]], N_mean[[0,nstates+1]], color="blue")
    ax2.set_ylabel("Distribution over terminal states")
    ax2.spines['right'].set_color("blue")
    ax2.tick_params(axis='y', colors="blue")
    ax2.yaxis.label.set_color("blue")
    ax.set_title("Distribution of state visits in all {} episodes (seed={})".format(nepisodes, seed))



############################ TD(lambda) ##################################
# TD(lambda) learners: iterate on lambda

# OBSERVATIONS:
# ============
# - Adjusting alpha by COUNT gives smaller RMSE after 1000 episodes for INTERMEDIATE lambda (lambda = 0.4)
# - Adjusting alpha by EPISODE gives smaller RMSE after 1000 episodes for LARGER lambda (very similar values between 0.8 and 0.95)
# - The best scenario with really small RMSE after 1000 or 5000 episodes happens when alpha is
# adjusted by FIRST-VISIT state count, which is better than adjusting by EPISODE because in the latter
# case the alpha is reduced even for non-visited states in the episode... and this prevents the state
# from learning next time as it would have been able to learn had alpha not been reduced.
#
# CONClUSIONS: USE ALPHA ADJUSTED BY FIRST-VISIT STATE COUNT FOR FASTER CONVERGENCE!!
# ===========
# - Adjusting alpha by COUNT makes the decrease rate too large... making alpha small too soon,
# ESPECIALLY WHEN ALPHA IS REDUCED EVEN WHEN THERE ARE **NO UPDATES OF THE VALUE FUNCTION**
# (e.g. at the beginning, when most estimated values of V(s) are set to 0! so almost no change in TD(lambda)
# especially when lambda is small, because there is small impact coming from the reward at the terminal states)
# ==> IDEA 1: Initialize V(s) to random values, so there are updates to the value function early on.
# ==> IDEA 2: Update alpha ONLY AFTER there is an important update of the value function (i.e. large delta(t) value)
# - Adjusting alpha by EPISODE makes alpha decrease less aggressively so larger lambdas have a reasonable
# update at the beginning, making the RMSE go down to 0 fast... compared to smaller lambdas.
# This update of alpha could lead to an ADAPTIVE alpha, in a similar way we propose an adaptive lambda...
# (but in the opposite direction, i.e. adaptive_alpha(delta) = 1 - adaptive_lambda(delta))
# ==> IDEA 3: Keep alpha the same for 10 episodes/visits to the state and then decrease it by
# episode-number/state-count. In this way, we can learn for a litte bit before decreasing it!
# This also would follow the strategy suggested by Srikant, Gupta & Yin in their 2020 paper about
# leveraging transient state and steady-state.
#

if run_td:
    results_td = []

    #alphas_td = [8, 7.5, 7, 3, 2, 1]        # 10 times larger than the optimum in constant-alpha case
    alphas_td = np.repeat(alpha_non_adaptive, len(lambdas))
    for idx_lambda, lmbda in enumerate(lambdas):
        print("\n******* lambda {} of {}: {:.2f} ******".format(idx_lambda+1, len(lambdas), lmbda))

        # Reset learner and agent (i.e. erase all memory from a previous run!)
        learner_td.setParams(alpha=alphas_td[idx_lambda], lmbda=lmbda)
        learner_td.reset(reset_episode=True, reset_value_functions=True)
        agent = GenericAgent(pol_rw, learner_td)

        # NOTE: Setting the seed here implies that each set of experiments
        # (i.e. for each combination of alpha and lambda) yields the same outcome in terms
        # of visited states and actions.
        # This is DESIRED --as opposed of having different state-action outcomes for different
        # (alpha, lambda) settings-- as it better isolates the effect of alpha and lambda.
        # VERIFIED BY RUNNING IN DEBUG MODE!
        sim = simulators.Simulator(env, agent, seed=seed, debug=debug)

        # Run the simulation and store the results
        N_mean, rmse_mean, rmse_se, RMSE_by_episode_mean, RMSE_by_episode_se, learning_info = \
                            sim.simulate(nexperiments=nexperiments,
                                         nepisodes=nepisodes,
                                         start=start,
                                         verbose=verbose,
                                         verbose_period=verbose_period,
                                         plot=plot)
        results_td += [{'lambda': lmbda,
                        'alpha': alphas_td[idx_lambda],
                        'rmse_mean': rmse_mean,                         # Average of "RMSE at LAST episode" over all experiments
                        'rmse_se': rmse_se,
                        'rmse_by_episode_mean': RMSE_by_episode_mean,   # Average of "RMSE at EACH episode" over all experiments
                        'rmse_by_episode_se': RMSE_by_episode_se,
                        'learning_info': learning_info
                         }]

    # Save
    file = open(resultsdir + "/td.pickle", mode="wb")  # "b" means binary mode (needed for pickle.dump())
    pickle.dump(results_td, file)
    file.close()


############################ TD(lambda)-adaptive ##################################
# TD(lambda)-adaptive learners: iterate on alpha

# OBSERVATIONS:
# ============
# - Adjusting alpha by COUNT gives smaller RMSE after 1000 episodes for SMALLER alpha (alpha = 1)
# - Adjusting alpha by EPISODE gives smaller RMSE after 1000 episodes for LARGER alpha (alpha = 2 or 3)
# - Faster convergence at the beginning happens when adjusting by COUNT (for the best case, alpha = 1)
# - Smaller RMSE at the end is obtained when adjusting by EPISODE (for the best case, alpha = 2)
# - The best scenario with really small RMSE after 1000 or 5000 episodes happens when alpha is
# adjusted by FIRST-VISIT state count, which is better than adjusting by EPISODE because in the latter
# case the alpha is reduced even for non-visited states in the episode... and this prevents the state
# from learning next time as it would have been able to learn had alpha not been reduced.
# - Using lambda_min > 0 doesn't give an advantage... on the contrary, it may be worse. The only
# noted advantage is a faster convergence at early times for larger initial alphas (e.g. alpha=5)
#
# CONClUSIONS: USE ALPHA ADJUSTED BY FIRST-VISIT STATE COUNT FOR FASTER CONVERGENCE!!
# ===========
# Same conclusions as for TD(lambda)

if run_td_adap:
    results_td_adap = []
    #alphas_adaptive = alphas_td.copy()
    #alphas_adaptive.reverse()
    for idx_alpha, alpha in enumerate(alphas_adaptive):
        print("\n******* alpha {} of {}: {:.2f} ******".format(idx_alpha+1, len(alphas_adaptive), alpha))

        # Reset learner and agent (i.e. erase all memory from a previous run!)
        learner_td_adap.setParams(alpha=alpha,
                                  lambda_min=lambda_min, lambda_max=lambda_max)
        learner_td_adap.reset(reset_episode=True, reset_value_functions=True)
        agent = GenericAgent(pol_rw, learner_td_adap)

        # NOTE: Setting the seed here implies that each set of experiments
        # (i.e. for each combination of alpha and lambda) yields the same outcome in terms
        # of visited states and actions.
        # This is DESIRED --as opposed of having different state-action outcomes for different
        # (alpha, lambda) settings-- as it better isolates the effect of alpha and lambda.
        # VERIFIED BY RUNNING IN DEBUG MODE!
        sim = simulators.Simulator(env, agent, seed=seed, debug=debug)

        # Run the simulation and store the results
        N_mean, rmse_mean, rmse_se, RMSE_by_episode_mean, RMSE_by_episode_se, learning_info = \
                            sim.simulate(nexperiments=nexperiments,
                                         nepisodes=nepisodes,
                                         start=start,
                                         verbose=verbose,
                                         verbose_period=verbose_period)
        results_td_adap += [{'lambda_min': lambda_min,
                             'alpha': alpha,
                             'rmse_mean': rmse_mean,                        # Average of "RMSE at LAST episode" over all experiments
                             'rmse_se': rmse_se,
                             'rmse_by_episode_mean': RMSE_by_episode_mean,  # Average of "RMSE at EACH episode" over all experiments
                             'rmse_by_episode_se': RMSE_by_episode_se,
                             'learning_info': learning_info
                             }]

    file = open(resultsdir + "/td_adap.pickle", mode="wb")  # "b" means binary mode (needed for pickle.dump())
    pickle.dump(results_td_adap, file)
    file.close()



################################# Plots #############################
# Load the data if needed
if run_mc:
    file = open(resultsdir + "/mc_adjust_by_count_first_visit.pickle", mode="rb")
    pickle.load(file)
    file.close()

if run_td:
    file = open(resultsdir + "/td.pickle", mode="rb")
    pickle.load(file)
    file.close()

if run_td_adap:
    file = open(resultsdir + "/td_adap.pickle", mode="rb")
    pickle.load(file)
    file.close()

colormap = cm.get_cmap("jet")
max_alpha = 1       # max(alpha_td)
max_rmse = 0.8
fontsize = 14

# MC (lambda-return)
if run_mc:
    fig = plot_rmse_by_episode(results_mc[0]['rmse_by_episode_mean'], #results_mc[0]['rmse_by_episode_se'],
                               max_rmse=max_rmse, color="blue",
                               alphas=results_mc[0]['learning_info']['alphas_by_episode'],
                               max_alpha=max_alpha, color_alphas="blue",
                               fontsize=fontsize)
    fig.suptitle(str(nstates) + "-state environment: Monte Carlo (alpha adjusted by " + ((adjust_alpha_by_episode and "episode") or (not adjust_alpha_by_episode and "state count")) + ")",
                 fontsize=np.round(fontsize*1.5))

# TD(lambda)
if run_td:
    lambdas = [s['lambda'] for s in results_td]
    alphas = [s['alpha'] for s in results_td]
    fig = None
    legend_label = []
    for idx_lambda, lmbda in enumerate(lambdas):
        # Map blue to the largest lambda and red to the smallest lambda (most similar to the color scheme used in Sutton, pag. 295)
        color = colormap( 1 - idx_lambda / max((1, len(lambdas)-1)) )
        fig = plot_rmse_by_episode(results_td[idx_lambda]['rmse_by_episode_mean'], #results_td[idx_lambda]['rmse_by_episode_se'],
                                   max_rmse=max_rmse, color=color,
                                   alphas=results_td[idx_lambda]['learning_info']['alphas_by_episode'],
                                   max_alpha=max_alpha, color_alphas=color,
                                   fontsize=fontsize, fig=fig)
        legend_label += ["lambda={:.2g}, alpha={:.2g}".format(lmbda, alphas[idx_lambda])]
    plt.figlegend(legend_label)
    fig.suptitle(str(nstates) + "-state environment: TD(lambda) (alpha " + ((not adjust_alpha and "constant)") or "adjusted by " + ((adjust_alpha_by_episode and "episode") or (not adjust_alpha_by_episode and "state count")) + ")"),
                 fontsize=np.round(fontsize*1.5))

# TD(lambda)-adaptive
if run_td_adap:
    alphas = [s['alpha'] for s in results_td_adap]
    fig = None
    legend_label = []
    for idx_alpha, alpha in enumerate(alphas):
        # Map blue to the largest lambda and red to the smallest lambda (most similar to the color scheme used in Sutton, pag. 295)
        color = colormap( 1 - idx_alpha / max((1, len(alphas)-1)) )
        fig = plot_rmse_by_episode(results_td_adap[idx_alpha]['rmse_by_episode_mean'], #results_td_adap[idx_alpha]['rmse_by_episode_se'],
                                   max_rmse=max_rmse, color=color,
                                   alphas=results_td_adap[idx_alpha]['learning_info']['alphas_by_episode'],
                                   max_alpha=max_alpha, color_alphas=color,
                                   fontsize=fontsize, fig=fig)
        legend_label += ["alpha={:.2g}".format(alpha)]
    plt.figlegend(legend_label)
    fig.suptitle(str(nstates) + "-state environment: TD(lambda) (alpha " + ((not adjust_alpha and "constant)") or
                 "adjusted by " + ((adjust_alpha_by_episode and "episode") or (not adjust_alpha_by_episode and "state count")) + \
                 ", lambda_min=" + str(lambda_min) + ", lambda_max=" + str(lambda_max) + ")"),
                 fontsize=np.round(fontsize*1.5))



# To save a figure to a file
#plt.savefig("{}/{}-v{}-td{}-lr{},tdadap{},adjust_alpha={},alpha_min={},lambda_min={}-Episodes{}.png" \
#            .format(g_dir_results, g_prefix, version, lambdas_opt['td'], lambdas_opt['mc'], alphas_opt['td_adap'], adjust_alpha, alpha_min, lambda_min, nepisodes))
#fig, (ax_full, ax_scaled, ax_rmse_by_episode) = plt.subplots(1,3)
#fig, (ax_full, ax_scaled) = plt.subplots(1,2)


#------------------------------------ ANALYZE RMSE AS A FUNCTION OF ALPHA = FIXED -------------------------------------#
# Goal: reproduce the plot in Sutton, pag. 295
if run_td:
    time_start = timer()

    results_td = []

    for idx_lambda, lmbda in enumerate(lambdas):
        print("\n******* lambda {} of {}: {:.2f} ******".format(idx_lambda + 1, len(lambdas), lmbda))
        rmse_last_episode_mean_by_alpha = []
        rmse_last_episode_se_by_alpha = []
        rmse_mean_by_alpha = []
        rmse_se_by_alpha = []
        for idx_alpha, alpha in enumerate(alphas):
            print("\n\t******* lambda = {:.2f} ({} of {}): alpha {} of {}: {:.2f} ******".format(lmbda, idx_lambda+1, len(lambdas), idx_alpha+1, len(alphas), alpha))

            # Reset learner and agent (i.e. erase all memory from a previous run!)
            learner_td.setParams(alpha=alpha, lmbda=lmbda)
            learner_td.reset(reset_episode=True, reset_value_functions=True)
            agent = GenericAgent(pol_rw, learner_td)

            # NOTE: Setting the seed here implies that each set of experiments
            # (i.e. for each combination of alpha and lambda) yields the same outcome in terms
            # of visited states and actions.
            # This is DESIRED --as opposed of having different state-action outcomes for different
            # (alpha, lambda) settings-- as it better isolates the effect of alpha and lambda.
            # VERIFIED BY RUNNING IN DEBUG MODE!
            sim = simulators.Simulator(env, agent, seed=seed, debug=debug)

            # Run the simulation and store the results
            N_mean, rmse_mean, rmse_se, RMSE_by_episode_mean, RMSE_by_episode_se, learning_info = \
                                sim.simulate(nexperiments=nexperiments,
                                             nepisodes=nepisodes,
                                             start=start,
                                             verbose=verbose,
                                             verbose_period=verbose_period,
                                             plot=False)

            # Add the RMSE results to the list of RMSE's by alpha
            rmse_last_episode_mean_by_alpha += [rmse_mean]
            rmse_last_episode_se_by_alpha += [rmse_se]
            # For the computation of the average RMSE over all episodes we exclude the very beginning because
            # it is not informative about the performance of the learning algorithm, as no learning has yet taken place!
            rmse_mean_by_alpha += [ np.mean(RMSE_by_episode_mean[1:]) ]
            rmse_se_by_alpha += [ np.mean(RMSE_by_episode_se[1:]) ]

        results_td += [{'nexperiments': nexperiments,
                        'nepisodes': nepisodes,
                        'lambda': lmbda,
                        'alphas': alphas,
                        'rmse_mean': rmse_mean_by_alpha,
                        'rmse_se': rmse_se_by_alpha,
                        'rmse_n': nexperiments * len( np.mean(RMSE_by_episode_mean[1:]) ),
                        'rmse_last_episode_mean': rmse_last_episode_mean_by_alpha,
                        'rmse_last_episode_se': rmse_last_episode_se_by_alpha,
                        'rmse_last_episode_n': nexperiments,
                        }]

    time_end = timer()
    exec_time = time_end - time_start
    print("Execution time for TD: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

    # Save
    file = open(resultsdir + "/td_alpha_const_first_episodes.pickle", mode="wb")  # "b" means binary mode (needed for pickle.dump())
    pickle.dump(results_td, file)
    file.close()

if run_td_adap:
    time_start = timer()

    rmse_last_episode_mean_by_alpha = []
    rmse_last_episode_se_by_alpha = []
    rmse_mean_by_alpha = []
    rmse_se_by_alpha = []
    for idx_alpha, alpha in enumerate(alphas):
        print("\n\t******* Adaptive TD(lambda): alpha {} of {}: {:.2f} ******".format(idx_alpha+1, len(alphas), alpha))

        # Reset learner and agent (i.e. erase all memory from a previous run!)
        learner_td_adap.setParams(alpha=alpha, lambda_min=lambda_min, lambda_max=lambda_max)
        learner_td_adap.reset(reset_episode=True, reset_value_functions=True)
        agent = GenericAgent(pol_rw, learner_td_adap)

        # NOTE: Setting the seed here implies that each set of experiments
        # (i.e. for each combination of alpha and lambda) yields the same outcome in terms
        # of visited states and actions.
        # This is DESIRED --as opposed of having different state-action outcomes for different
        # (alpha, lambda) settings-- as it better isolates the effect of alpha and lambda.
        # VERIFIED BY RUNNING IN DEBUG MODE!
        sim = simulators.Simulator(env, agent, seed=seed, debug=debug)

        # Run the simulation and store the results
        N_mean, rmse_mean, rmse_se, RMSE_by_episode_mean, RMSE_by_episode_se, learning_info = \
                            sim.simulate(nexperiments=nexperiments,
                                         nepisodes=nepisodes,
                                         start=start,
                                         verbose=verbose,
                                         verbose_period=verbose_period,
                                         plot=False)

        # Add the RMSE results to the list of RMSE's by alpha
        rmse_last_episode_mean_by_alpha += [rmse_mean]
        rmse_last_episode_se_by_alpha += [rmse_se]
        # For the computation of the average RMSE over all episodes we exclude the very beginning because
        # it is not informative about the performance of the learning algorithm, as no learning has yet taken place!
        rmse_mean_by_alpha += [ np.mean(RMSE_by_episode_mean[1:]) ]
        rmse_se_by_alpha += [ np.mean(RMSE_by_episode_se[1:]) ]

    results_td_adap = {
                        'nexperiments': nexperiments,
                        'nepisodes': nepisodes,
                        'alphas': alphas,
                        'rmse_mean': rmse_mean_by_alpha,
                        'rmse_se': rmse_se_by_alpha,
                        'rmse_n': nexperiments * len( np.mean(RMSE_by_episode_mean[1:]) ),
                        'rmse_last_episode_mean': rmse_last_episode_mean_by_alpha,
                        'rmse_last_episode_se': rmse_last_episode_se_by_alpha,
                        'rmse_last_episode_n': nexperiments,
                        }

    time_end = timer()
    exec_time = time_end - time_start
    print("Execution time for {}: {:.1f} sec, {:.1f} min".format(adaptive_type.name, exec_time, exec_time / 60))

    # Save
    file = open(resultsdir + "/td_{}_lambdamax={:.2f}_alpha_const_first_episodes.pickle".format(adaptive_type.name, lambda_max), mode="wb")
    pickle.dump(results_td_adap, file)
    file.close()


#--------------------- Plot RMSE vs. alpha results -----------------#
# Load the data
file = open(resultsdir + "/td_alpha_const_first_episodes.pickle", mode="rb")
results_td = pickle.load(file)
file.close()

file = open(resultsdir + "/td_{}_lambdamax={:.2f}_alpha_const_first_episodes.pickle".format(adaptive_type.name, lambda_max), mode="rb")
results_td_adap = pickle.load(file)
file.close()


#-- When reading both the ATD and HATD results, for better comparison
# ATD
lambda_max_ATD = 0.99
file = open(resultsdir + "/td_ATD_lambdamax={:.2f}_alpha_const_first_episodes.pickle".format(lambda_max_ATD), mode="rb")
results_td_ATD = pickle.load(file)
file.close()
# HATD
lambda_max_HATD = 0.80
file = open(resultsdir + "/td_HATD_lambdamax={:.2f}_alpha_const_first_episodes.pickle".format(lambda_max_HATD), mode="rb")
results_td_HATD = pickle.load(file)
file.close()


savefig = True
plot_rmse_last_episode = False
plot_errorbars = True
plot_td_adap_together = True

if plot_td_adap_together:
    figfile = os.path.join(resultsdir, "td_ATD({:.2f})_HATD({:.2f})_alpha_const_{}states_{}episodes" \
                           .format(lambda_max_ATD, lambda_max_HATD, nstates, nepisodes) + (plot_errorbars and "_errors" or "") + ".png")
else:
    figfile = os.path.join(resultsdir, "td_{}_alpha_const_lambdamax={:.2f}_{}states_{}episodes" \
                           .format(adaptive_type.name, lambda_max, nstates, nepisodes) + (plot_errorbars and "_errors" or "") + ".png")

colormap = cm.get_cmap("jet")
max_rmse = 0.6
fontsize = 14

ax = plt.figure(figsize=(10,10)).subplots()
legend_label = []

# TD(lambda)
for idx_lambda, lmbda in enumerate(lambdas):
    # Map blue to the largest lambda and red to the smallest lambda (most similar to the color scheme used in Sutton, pag. 295)
    color = colormap( 1 - idx_lambda / max((1, len(lambdas)-1)) )
    if plot_rmse_last_episode:
        ax.plot(results_td[idx_lambda]['alphas'], results_td[idx_lambda]['rmse_last_episode_mean'], '.--', color=color)
        if plot_errorbars:
            ax.errorbar(results_td[idx_lambda]['alphas'], results_td[idx_lambda]['rmse_last_episode_mean'], color=color,
                        yerr=results_td[idx_lambda]['rmse_last_episode_se'], capsize=4, linestyle='dashed')
        ax.set_ylabel("RMSE at last episode ({}), averaged over {} experiments".format(nepisodes, nexperiments), fontsize=fontsize)
    else:
        ax.plot(results_td[idx_lambda]['alphas'], results_td[idx_lambda]['rmse_mean'], '.--', color=color)
        if plot_errorbars:
            ax.errorbar(results_td[idx_lambda]['alphas'], results_td[idx_lambda]['rmse_mean'], color=color,
                        yerr=results_td[idx_lambda]['rmse_se'], capsize=4, linestyle='dashed')
        ax.set_ylabel("Average RMSE over first {} episodes, averaged over {} experiments".format(nepisodes, nexperiments), fontsize=fontsize)
    ax.set_ylim((0, max_rmse))
    ax.set_xlabel(r"$\alpha$", fontsize=fontsize)
    ax.set_title(r"TD($\lambda$) algorithms on {}-state 1D gridworld".format(nstates), fontsize=fontsize)
    # Square plot
    # Ref: https://www.geeksforgeeks.org/how-to-make-a-square-plot-with-equal-axes-in-matplotlib/
    ax.set_aspect(1. / ax.get_data_ratio(), adjustable='box')
    legend_label += [r"TD($\lambda$={:.2g})".format(lmbda)]

if plot_td_adap_together:
    if plot_rmse_last_episode:
        ax.plot(results_td_ATD['alphas'], results_td_ATD['rmse_last_episode_mean'], 'x-', color="gray")
        ax.plot(results_td_HATD['alphas'], results_td_HATD['rmse_last_episode_mean'], 'x-', color="black")
        if plot_errorbars:
            ax.errorbar(results_td_ATD['alphas'], results_td_ATD['rmse_last_episode_mean'], color="gray",
                        yerr=results_td_ATD['rmse_last_episode_se'], capsize=4)
            ax.errorbar(results_td_HATD['alphas'], results_td_HATD['rmse_last_episode_mean'], color="black",
                        yerr=results_td_HATD['rmse_last_episode_se'], capsize=4)
    else:
        ax.plot(results_td_ATD['alphas'], results_td_ATD['rmse_mean'], 'x-', color="gray")
        ax.plot(results_td_HATD['alphas'], results_td_HATD['rmse_mean'], 'x-', color="black")
        if plot_errorbars:
            ax.errorbar(results_td_ATD['alphas'], results_td_ATD['rmse_mean'], color="gray",
                        yerr=results_td_ATD['rmse_se'], capsize=4)
            ax.errorbar(results_td_HATD['alphas'], results_td_HATD['rmse_mean'], color="black",
                        yerr=results_td_HATD['rmse_se'], capsize=4)
    legend_label += [r"ATD($\lambda_{max}$" + "={:.2f})".format(lambda_max_ATD), r"HATD($\lambda_{max}$" + "={:.2f})".format(lambda_max_HATD)]
else:
    if plot_rmse_last_episode:
        if plot_errorbars:
            ax.errorbar(results_td_adap['alphas'], results_td_adap['rmse_last_episode_mean'], color="black",
                        yerr=results_td_adap['rmse_last_episode_se'], capsize=4)
        else:
            ax.plot(results_td_adap['alphas'], results_td_adap['rmse_last_episode_mean'], 'x-', color="black")
    else:
        if plot_errorbars:
            ax.errorbar(results_td_adap['alphas'], results_td_adap['rmse_mean'], color="black",
                        yerr=results_td_adap['rmse_se'], capsize=4)
        else:
            ax.plot(results_td_adap['alphas'], results_td_adap['rmse_mean'], 'x-', color="black")
    legend_label += [adaptive_type.name + r" $\lambda_{max}$" + "={:.2f})".format(lambda_max)]
plt.legend(legend_label, loc='lower right', fontsize=fontsize)

if savefig:
    #plt.gcf().subplots_adjust(left=0.15, top=0.75)
    plt.savefig(figfile)
