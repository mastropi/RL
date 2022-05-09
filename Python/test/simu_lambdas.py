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


#--------------------------- Auxiliary functions ------------------------------
def run_td_adap(learner_td_adap, lambda_min, lambda_max, alphas, seed=None, debug=False):
    "NOTE: This function uses global variables with the experiment setup"
    rmse_last_episode_mean_by_alpha = []
    rmse_last_episode_se_by_alpha = []
    rmse_last_episode_n_by_alpha = []
    rmse_mean_by_alpha = []
    rmse_se_by_alpha = []
    rmse_n_by_alpha = []
    RMSE_by_episode_mean_by_alpha = []
    RMSE_by_episode_se_by_alpha = []
    RMSE_by_episode_n_by_alpha = []
    learning_info_by_alpha = []
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
        N_mean, rmse_last_episode_mean, rmse_last_episode_se, rmse_last_episode_n,\
                RMSE_by_episode_mean, RMSE_by_episode_se, RMSE_by_episode_n,\
                learning_info = sim.simulate(nexperiments=nexperiments,
                                             nepisodes=nepisodes,
                                             start=start,
                                             verbose=verbose,
                                             verbose_period=verbose_period,
                                             plot=False)

        # Add the RMSE results to the list of RMSE's by alpha
        rmse_last_episode_mean_by_alpha += [rmse_last_episode_mean]
        rmse_last_episode_se_by_alpha += [rmse_last_episode_se]
        rmse_last_episode_n_by_alpha += [rmse_last_episode_n]

        # For the computation of the average RMSE over all episodes we exclude the very beginning because
        # it is not informative about the performance of the learning algorithm, as no learning has yet taken place!
        rmse_mean_by_alpha += [ np.mean(RMSE_by_episode_mean[1:]) ]
        rmse_se_by_alpha += [ np.mean(RMSE_by_episode_se[1:]) ]
        rmse_n_by_alpha += [ RMSE_by_episode_n * nepisodes ]

        # RMSE by episode
        RMSE_by_episode_mean_by_alpha += [ RMSE_by_episode_mean ]
        RMSE_by_episode_se_by_alpha += [ RMSE_by_episode_se ]
        RMSE_by_episode_n_by_alpha += [ RMSE_by_episode_n ]

        # Learning rate by episode
        learning_info_by_alpha += [ learning_info ]

    results_td_adap = {
                        'nexperiments': nexperiments,
                        'nepisodes': nepisodes,
                        'alphas': alphas,
                        'rmse_mean': rmse_mean_by_alpha,
                        'rmse_se': rmse_se_by_alpha,
                        'rmse_n': rmse_n_by_alpha,
                        'rmse_last_episode_mean': rmse_last_episode_mean_by_alpha,
                        'rmse_last_episode_se': rmse_last_episode_se_by_alpha,
                        'rmse_last_episode_n': rmse_last_episode_n,
                        'rmse_by_episode_mean': RMSE_by_episode_mean_by_alpha,
                        'rmse_by_episode_se': RMSE_by_episode_se_by_alpha,
                        'rmse_by_episode_n': RMSE_by_episode_n_by_alpha,
                        'learning_info': learning_info_by_alpha,
                        }

    return results_td_adap


############################ EXPERIMENT SETUP #################################
# The environment
nstates = 19 # Number of states excluding terminal states
env = gridworlds.EnvGridworld1D(length=nstates+2)

# What experiments to run
run_mc = False
run_td = True
run_atd = True
run_hatd = True

# Simulation setup
gamma = 1.0

# alpha
alphas_td = [0.8, 0.8, 0.8, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1] # Optimum for TD(0.8) where 0.8 is the optimum lambda
# alphas_td = [8, 7.5, 7, 3, 2, 1]        # 10 times larger than the optimum in constant-alpha case
alphas_atd = [0.6] # Optimum for ATD with lambda_max = 0.99
alphas_hatd = [0.4] # Optimum for HATD with lambda_max = 0.80

adjust_alpha = True
adjust_alpha_by_episode = False
alpha_update_type = AlphaUpdateType.FIRST_STATE_VISIT

# lambda
#lambdas = [0.0, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95]
lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]
lambda_min = 0.0
lambda_max_atd = 0.99
lambda_max_hatd = 0.80

seed = 1717
nexperiments = 10
nepisodes = 1000
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
                            alpha_update_type=alpha_update_type)
learner_atd = td.LeaTDLambdaAdaptive(env, gamma=gamma, adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode,
                                         alpha_update_type=alpha_update_type,
                                         adaptive_type=td.AdaptiveLambdaType.ATD)
learner_hatd = td.LeaTDLambdaAdaptive(env, gamma=gamma, adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode,
                                         alpha_update_type=alpha_update_type,
                                         adaptive_type=td.AdaptiveLambdaType.HATD)

############################ lambda-return #################################
if run_mc:
    # MC (lambda-return)
    learner_mc.setParams()
    agent = GenericAgent(pol_rw, learner_mc)

    # Run the simulation and store the results
    sim = simulators.Simulator(env, agent, seed=seed, debug=debug)
    N_mean, rmse_mean, rmse_se, rmse_n, RMSE_by_episode_mean, RMSE_by_episode_se, RMSE_by_episode_n, learning_info = \
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



#------------------------------------ ANALYZE RMSE -------------------------------------#
#-- When running the experiment with alpha fixed for different alphas
# Goal: reproduce the plot in Sutton, pag. 295
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alphas_td = alphas
alphas_atd = alphas
alphas_hatd = alphas
#-- When running the experiment with alpha fixed for different alphas


# Suffixes to use in the name of the pickle files containing the results
# Note: we separate "experiments" with comma because that parameter is LESS important than the rest, like lambda, alpha, etc.
suffix = "adj={}_episodes={},experiments={}" \
        .format(adjust_alpha, nepisodes, nexperiments)

# When analyzing RMSE as a function of alpha FIXED
if run_td:
    time_start = timer()

    results_td = []

    for idx_lambda, lmbda in enumerate(lambdas):
        print("\n******* lambda {} of {}: {:.2f} ******".format(idx_lambda + 1, len(lambdas), lmbda))
        rmse_last_episode_mean_by_alpha = []
        rmse_last_episode_se_by_alpha = []
        rmse_mean_by_alpha = []
        rmse_se_by_alpha = []
        for idx_alpha, alpha in enumerate(alphas_td):
            print("\n\t******* lambda = {:.2f} ({} of {}): alpha {} of {}: {:.2f} ******".format(lmbda, idx_lambda+1, len(lambdas), idx_alpha+1, len(alphas_td), alpha))

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
            N_mean, rmse_last_episode_mean, rmse_last_episode_se, rmse_last_episode_n, RMSE_by_episode_mean, RMSE_by_episode_se, RMSE_by_episode_n, learning_info = \
                                sim.simulate(nexperiments=nexperiments,
                                             nepisodes=nepisodes,
                                             start=start,
                                             verbose=verbose,
                                             verbose_period=verbose_period,
                                             plot=False)

            # Add the RMSE results to the list of RMSE's by alpha
            rmse_last_episode_mean_by_alpha += [rmse_last_episode_mean]
            rmse_last_episode_se_by_alpha += [rmse_last_episode_se]
            # For the computation of the average RMSE over all episodes we exclude the very beginning because
            # it is not informative about the performance of the learning algorithm, as no learning has yet taken place!
            rmse_mean_by_alpha += [ np.mean(RMSE_by_episode_mean[1:]) ]
            rmse_se_by_alpha += [ np.mean(RMSE_by_episode_se[1:]) ]

        results_td += [{'nexperiments': nexperiments,
                        'nepisodes': nepisodes,
                        'lambda': lmbda,
                        'alphas': alphas_td,
                        'rmse_mean': rmse_mean_by_alpha,
                        'rmse_se': rmse_se_by_alpha,
                        'rmse_n': rmse_last_episode_n * nepisodes,
                        'rmse_last_episode_mean': rmse_last_episode_mean_by_alpha,
                        'rmse_last_episode_se': rmse_last_episode_se_by_alpha,
                        'rmse_last_episode_n': nexperiments,
                        'rmse_by_episode_mean': RMSE_by_episode_mean,
                        'rmse_by_episode_se': RMSE_by_episode_se,
                        'rmse_by_episode_n': RMSE_by_episode_n,
                        'learning_info': learning_info
                        }]

    time_end = timer()
    exec_time = time_end - time_start
    print("Execution time for TD: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

    # Save
    file = open(resultsdir + "/td_lambdas={}_".format(lambdas) + suffix + ".pickle", mode="wb")  # "b" means binary mode (needed for pickle.dump())
    pickle.dump(results_td, file)
    file.close()

# When analyzing convergence of the algorithm with possibly adjustable alpha
if run_td:
    time_start = timer()

    results_td = []

    assert len(alphas_td) == 1 or len(lambdas) == len(alphas_td)
    for idx_lambda, lmbda in enumerate(lambdas):
        if len(alphas_td) == 1:
            alpha = alphas_td[0]
        else:
            alpha = alphas_td[idx_lambda]
        print("\n\t******* lambda = {:.2f} ({} of {}): alpha = {:.2f} ******".format(lmbda, idx_lambda+1, len(lambdas), alpha))

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
        N_mean, rmse_last_episode_mean, rmse_last_episode_se, rmse_last_episode_n,\
                RMSE_by_episode_mean, RMSE_by_episode_se, RMSE_by_episode_n,\
                learning_info = sim.simulate(nexperiments=nexperiments,
                                             nepisodes=nepisodes,
                                             start=start,
                                             verbose=verbose,
                                             verbose_period=verbose_period,
                                             plot=False)

        rmse_mean = np.mean(RMSE_by_episode_mean[1:])
        rmse_se = np.mean(RMSE_by_episode_se[1:])
        rmse_n = rmse_last_episode_n * nepisodes

        results_td += [{'nexperiments': nexperiments,
                        'nepisodes': nepisodes,
                        'lambda': lmbda,
                        'alpha': alpha,
                        'rmse_mean': rmse_mean,
                        'rmse_se': rmse_se,
                        'rmse_n': rmse_n,
                        'rmse_last_episode_mean': rmse_last_episode_mean,
                        'rmse_last_episode_se': rmse_last_episode_se,
                        'rmse_last_episode_n': rmse_last_episode_n,
                        'rmse_by_episode_mean': RMSE_by_episode_mean,
                        'rmse_by_episode_se': RMSE_by_episode_se,
                        'rmse_by_episode_n': RMSE_by_episode_n,
                        'learning_info': learning_info
                        }]

    time_end = timer()
    exec_time = time_end - time_start
    print("Execution time for TD: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

    # Save
    file = open(resultsdir + "/td_lambdas={}_alphamean={:.2f}_".format(lambdas, np.mean(alphas_td)) + suffix + ".pickle", mode="wb")  # "b" means binary mode (needed for pickle.dump())
    pickle.dump(results_td, file)
    file.close()

if run_atd:
    time_start = timer()

    results_atd = run_td_adap(learner_atd, lambda_min, lambda_max_atd, alphas_atd, seed=seed, debug=debug)

    time_end = timer()
    exec_time = time_end - time_start
    print("Execution time for ATD: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

    # Save
    file = open(resultsdir + "/td_ATD_lambdamax={:.2f}_alphas={}_".format(lambda_max_atd, alphas_atd) + suffix + ".pickle", mode="wb")
    pickle.dump(results_atd, file)
    file.close()

if run_hatd:
    time_start = timer()

    results_hatd = run_td_adap(learner_hatd, lambda_min, lambda_max_hatd, alphas_hatd, seed=seed, debug=debug)

    time_end = timer()
    exec_time = time_end - time_start
    print("Execution time for HATD: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

    # Save
    file = open(resultsdir + "/td_HATD_lambdamax={:.2f}_alphas={}_".format(lambda_max_hatd, alphas_hatd) + suffix + ".pickle", mode="wb")
    pickle.dump(results_hatd, file)
    file.close()


################################# Plots #############################
# Load the data if needed
file = open(resultsdir + "/td_lambdas={}_alphamean={:.2f}_".format(lambdas, np.mean(alphas_td)) + suffix + ".pickle", mode="rb")
#file = open(resultsdir + "/td_alpha_const_first_episodes.pickle", mode="rb")
results_td = pickle.load(file)
file.close()

lambda_max_atd = 0.99
file = open(resultsdir + "/td_ATD_lambdamax={:.2f}_alphas={}_".format(lambda_max_atd, alphas_atd) + suffix + ".pickle", mode="rb")
#file = open(resultsdir + "/td_ATD_lambdamax={:.2f}_alpha_const_first_episodes.pickle".format(lambda_max_atd), mode="rb")
results_atd = pickle.load(file)
file.close()

lambda_max_hatd = 0.80
file = open(resultsdir + "/td_HATD_lambdamax={:.2f}_alphas={}_".format(lambda_max_hatd, alphas_hatd) + suffix + ".pickle", mode="rb")
#file = open(resultsdir + "/td_HATD_lambdamax={:.2f}_alpha_const_first_episodes.pickle".format(lambda_max), mode="rb")
results_hatd = pickle.load(file)
file.close()

#--------------------- CONVERGENCE: Plot of RMSE vs. episode number -----------------#
# Setup plot
colormap = cm.get_cmap("jet")
max_alpha = 1       # max(alpha_td)
max_rmse = 0.8
fontsize = 14

savefig = True
plot_scales = "log"
plot_errorbars = False

lambdas_selected = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

figfile = os.path.join(resultsdir, "td_ATD({:.2f})_HATD({:.2f})_alpha={:.2f}_adj_{}states_{}episodes" \
                       .format(lambda_max_atd, lambda_max_hatd, np.mean(np.r_[alphas_td, alphas_atd, alphas_hatd]), nstates, nepisodes) + (plot_errorbars and "_errors" or "") + ".png")

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
    alphas_td = [s['alpha'] for s in results_td]
    fig = None
    legend_label = []
    for idx_lambda, lmbda in enumerate(lambdas):
        if not lmbda in lambdas_selected:
            continue
        # Map blue to the largest lambda and red to the smallest lambda (most similar to the color scheme used in Sutton, pag. 295)
        if len(alphas_td) == 1:
            color = "blue"
        else:
            color = colormap( 1 - idx_lambda / max((1, len(lambdas)-1)) )
        fig = plot_rmse_by_episode(results_td[idx_lambda]['rmse_by_episode_mean'], #results_td[idx_lambda]['rmse_by_episode_se'],
                                   max_rmse=max_rmse, color=color, linestyle='dashed',
                                   alphas=results_td[idx_lambda]['learning_info']['alphas_by_episode'],
                                   max_alpha=max_alpha, color_alphas=color,
                                   fontsize=fontsize, plot_scales=plot_scales, fig=fig, legend_rmse=False, legend_alpha=False)
        legend_label += [r"$\lambda$=" + "{:.2f}".format(lmbda) + r", $\alpha_0$=" + "{:.2f}".format(alphas_td[idx_lambda])]

# TD(lambda)-adaptive
if run_atd:
    alphas_atd = results_atd['alphas']
    #fig = None
    #legend_label = []
    for idx_alpha, alpha in enumerate(alphas_atd):
        # Map blue to the largest lambda and red to the smallest lambda (most similar to the color scheme used in Sutton, pag. 295)
        if len(alphas_atd) == 1:
            color = "gray"
            linestyle = "solid"
        else:
            color = colormap( 1 - idx_alpha / max((1, len(alphas_atd)-1)) )
            linestyle = "dotted"
        fig = plot_rmse_by_episode(results_atd['rmse_by_episode_mean'][idx_alpha], #results_atd['rmse_by_episode_se'][idx_alpha],
                                   max_rmse=max_rmse, color=color, linestyle=linestyle,
                                   alphas=results_atd['learning_info'][idx_alpha]['alphas_by_episode'],
                                   max_alpha=max_alpha, color_alphas=color,
                                   fontsize=fontsize, plot_scales=plot_scales, fig=fig, legend_rmse=False, legend_alpha=False)
        legend_label += [r"ATD($\lambda_{max}$" + "={:.2f})".format(lambda_max_atd) + r", $\alpha_0$=" + "{:.2f}".format(alpha)]

if run_hatd:
    alphas_hatd = results_hatd['alphas']
    #fig = None
    #legend_label = []
    for idx_alpha, alpha in enumerate(alphas_hatd):
        # Map blue to the largest lambda and red to the smallest lambda (most similar to the color scheme used in Sutton, pag. 295)
        if len(alphas_hatd) == 1:
            color = "black"
        else:
            color = colormap( 1 - idx_alpha / max((1, len(alphas_hatd)-1)) )
        fig = plot_rmse_by_episode(results_hatd['rmse_by_episode_mean'][idx_alpha], #results_hatd['rmse_by_episode_se'][idx_alpha],
                                   max_rmse=max_rmse, color=color, linestyle="solid",
                                   alphas=results_hatd['learning_info'][idx_alpha]['alphas_by_episode'],
                                   max_alpha=max_alpha, color_alphas=color,
                                   fontsize=fontsize, plot_scales=plot_scales, fig=fig, legend_rmse=False, legend_alpha=False)
        legend_label += [r"HATD($\lambda_{max}$" + "={:.2f})".format(lambda_max_hatd) + r", $\alpha_0$=" + "{:.2f}".format(alpha)]

# Finalize plot with legend and title
if plot_scales == "both":
    plt.figlegend(legend_label)
else:
    plt.legend(legend_label, loc="upper right", fontsize=fontsize)
#fig.suptitle(str(nstates) + "-state environment: TD(lambda) (alpha " + ((not adjust_alpha and "constant)") or
#             "adjusted by " + ((adjust_alpha_by_episode and "episode") or (not adjust_alpha_by_episode and "state count")) + \
#             ", lambda_min=" + str(lambda_min) + ", lambda_max=" + str(lambda_max_hatd) + ")"),
#             fontsize=np.round(fontsize*1.5))
fig.suptitle(str(nstates) + r"-state 1D gridworld environment: TD($\lambda$) " + "\n" + \
             r"($\alpha$ " + ((not adjust_alpha and "constant)") or "adjusted by " + \
             ((adjust_alpha_by_episode and "episode") or (not adjust_alpha_by_episode and \
                                                         (alpha_update_type == AlphaUpdateType.FIRST_STATE_VISIT and "first-visit " or \
                                                         (alpha_update_type == AlphaUpdateType.EVERY_STATE_VISIT and "every-visit ")) + "state count")) + ")"),
             fontsize=np.round(fontsize*1.5))

if savefig:
    #plt.gcf().subplots_adjust(left=0.15, top=0.75)
    plt.savefig(figfile)



#--------------------- ALPHA SENSITIVITY: Plot of RMSE vs. alpha -----------------#
savefig = True
plot_rmse_last_episode = False
plot_errorbars = True
plot_td_adap_together = True

if plot_td_adap_together:
    figfile = os.path.join(resultsdir, "td_ATD({:.2f})_HATD({:.2f})_alpha_const_{}states_{}episodes" \
                           .format(lambda_max_atd, lambda_max_hatd, nstates, nepisodes) + (plot_errorbars and "_errors" or "") + ".png")
else:
    adaptive_type = td.AdaptiveLambdaType.ATD
    lambda_max = lambda_max_atd
    results_td_adap = results_atd
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
        ax.plot(results_atd['alphas'], results_atd['rmse_last_episode_mean'], 'x-', color="gray")
        ax.plot(results_hatd['alphas'], results_hatd['rmse_last_episode_mean'], 'x-', color="black")
        if plot_errorbars:
            ax.errorbar(results_atd['alphas'], results_atd['rmse_last_episode_mean'], color="gray",
                        yerr=results_atd['rmse_last_episode_se'], capsize=4)
            ax.errorbar(results_hatd['alphas'], results_hatd['rmse_last_episode_mean'], color="black",
                        yerr=results_hatd['rmse_last_episode_se'], capsize=4)
    else:
        ax.plot(results_atd['alphas'], results_atd['rmse_mean'], 'x-', color="gray")
        ax.plot(results_hatd['alphas'], results_hatd['rmse_mean'], 'x-', color="black")
        if plot_errorbars:
            ax.errorbar(results_atd['alphas'], results_atd['rmse_mean'], color="gray",
                        yerr=results_atd['rmse_se'], capsize=4)
            ax.errorbar(results_hatd['alphas'], results_hatd['rmse_mean'], color="black",
                        yerr=results_hatd['rmse_se'], capsize=4)
    legend_label += [r"ATD($\lambda_{max}$" + "={:.2f})".format(lambda_max_atd), r"HATD($\lambda_{max}$" + "={:.2f})".format(lambda_max_hatd)]
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
