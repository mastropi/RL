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
from Python.lib.environments import gridworlds, mountaincars
from Python.lib.agents import GenericAgent
from Python.lib.agents.policies import random_walks
from Python.lib.agents.learners import ResetMethod
from Python.lib.agents.learners.episodic.discrete import mc, td
from Python.lib.agents.learners.episodic.discrete import AlphaUpdateType
import Python.lib.simulators as simulators
import Python.lib.utils.computing as computing

from test_utils import plot_rmse_by_episode

# Directories
# Use the following directory (relative to the current file's directory) when running the whole script (I think)
#resultsdir = "../../RL-001-MemoryManagement/results/SimulateTDLambda-DifferentLambdas&Adaptive-2022"
# Use the following directory (relative to the project's directory) when running chunk by chunk manually
resultsdir_grid = os.path.abspath("./RL-001-MemoryManagement/results/SimulateTDLambda-DifferentLambdas&Adaptive-2022")
resultsdir_mountain = os.path.abspath("./RL-001-MemoryManagement/results/MountainCar")


#--------------------------- Auxiliary functions ------------------------------
def run_td_adap(learner_td_adap, lambda_min, lambda_max, alphas, start=None, seed=None, debug=False):
    "NOTE: This function uses global variables with the experiment setup"
    alpha_mean_by_episode_mean_by_alpha = []
    lambda_mean_by_episode_mean_by_alpha = []
    V_abs_mean_by_episode_mean_by_alpha = []
    deltaV_rel_max_signed_by_episode_mean_by_alpha = []
    rmse_last_episode_mean_by_alpha = []
    rmse_last_episode_se_by_alpha = []
    rmse_last_episode_n_by_alpha = []
    rmse_mean_by_alpha = []
    rmse_se_by_alpha = []
    rmse_n_by_alpha = []
    RMSE_by_episode_mean_by_alpha = []
    RMSE_by_episode_se_by_alpha = []
    MAPE_by_episode_mean_by_alpha = []
    MAPE_by_episode_se_by_alpha = []
    n_by_episode_by_alpha = []
    learning_info_by_alpha = []
    for idx_alpha, alpha in enumerate(alphas):
        print("\n\t******* Adaptive TD(lambda<={:.2f}): alpha {} of {}: {:.2f} ******".format(lambda_max, idx_alpha+1, len(alphas), alpha))

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
        N_mean, rmse_last_episode_mean, rmse_last_episode_se, rmse_last_episode_n, \
                RMSE_by_episode_mean, RMSE_by_episode_se, \
                MAPE_by_episode_mean, MAPE_by_episode_se, n_by_episode, \
                learning_info = sim.simulate(nexperiments=nexperiments,
                                             nepisodes=nepisodes,
                                             max_time_steps=max_time_steps,
                                             start=start,
                                             weights_rmse=weights_rmse,
                                             verbose=verbose,
                                             verbose_period=verbose_period,
                                             verbose_convergence=verbose_convergence,
                                             plot=False)

        alpha_mean_by_episode_mean_by_alpha += [learning_info['alpha_mean_by_episode_mean']]
        lambda_mean_by_episode_mean_by_alpha += [learning_info['lambda_mean_by_episode_mean']]
        V_abs_mean_by_episode_mean_by_alpha += [learning_info['V_abs_mean_by_episode_mean']]
        deltaV_rel_max_signed_by_episode_mean_by_alpha += [learning_info['deltaV_rel_max_signed_by_episode_mean']]

        # Add the RMSE results to the list of RMSE's by alpha
        rmse_last_episode_mean_by_alpha += [rmse_last_episode_mean]
        rmse_last_episode_se_by_alpha += [rmse_last_episode_se]
        rmse_last_episode_n_by_alpha += [rmse_last_episode_n]

        # For the computation of the average RMSE over all episodes we exclude the very beginning because
        # it is not informative about the performance of the learning algorithm, as no learning has yet taken place!
        rmse_mean_by_alpha += [ np.mean(RMSE_by_episode_mean[1:]) ]
        rmse_se_by_alpha += [ np.mean(RMSE_by_episode_se[1:]) ]
        rmse_n_by_alpha += [ n_by_episode * nepisodes ]

        # RMSE and MAPE by episode
        RMSE_by_episode_mean_by_alpha += [ RMSE_by_episode_mean ]
        RMSE_by_episode_se_by_alpha += [ RMSE_by_episode_se ]
        MAPE_by_episode_mean_by_alpha += [ MAPE_by_episode_mean ]
        MAPE_by_episode_se_by_alpha += [ MAPE_by_episode_se ]
        n_by_episode_by_alpha += [ n_by_episode ]

        # Learning rate by episode
        learning_info_by_alpha += [ learning_info ]

    results_td_adap = {
                        'nexperiments': nexperiments,
                        'nepisodes': nepisodes,
                        'alphas': alphas,
                        'alpha_mean_by_episode_mean': alpha_mean_by_episode_mean_by_alpha,
                        'lambda_mean_by_episode_mean': lambda_mean_by_episode_mean_by_alpha,
                        'V_abs_mean_by_episode_mean': V_abs_mean_by_episode_mean_by_alpha,
                        'deltaV_rel_max_signed_by_episode_mean': deltaV_rel_max_signed_by_episode_mean_by_alpha,
                        'rmse_mean': rmse_mean_by_alpha,
                        'rmse_se': rmse_se_by_alpha,
                        'rmse_n': rmse_n_by_alpha,
                        'rmse_last_episode_mean': rmse_last_episode_mean_by_alpha,
                        'rmse_last_episode_se': rmse_last_episode_se_by_alpha,
                        'rmse_last_episode_n': rmse_last_episode_n,
                        'rmse_by_episode_mean': RMSE_by_episode_mean_by_alpha,
                        'rmse_by_episode_se': RMSE_by_episode_se_by_alpha,
                        'mape_by_episode_mean': MAPE_by_episode_mean_by_alpha,
                        'mape_by_episode_se': MAPE_by_episode_se_by_alpha,
                        'n_by_episode': n_by_episode_by_alpha,
                        'learning_info': learning_info_by_alpha,
                        }

    return results_td_adap

def plot_results_adap(fig, legend_label, td_adap_name, results, lambda_max,
                      color=None, colormap=cm.get_cmap("jet"),
                      smooth=False, smooth_size=10,
                      kpi_name="mape", min_kpi=None, max_kpi=None, max_alpha=1.0, plot_scales='log', fontsize=14):
    alphas = results['alphas']
    for idx_alpha, alpha in enumerate(alphas):
        if color is None or len(alphas) > 1:
            color = colormap( 1 - idx_alpha / max((1, len(alphas)-1)) )
        # What to plot
        kpi_mean = results['{}_by_episode_mean'.format(kpi_name)][idx_alpha]
        if smooth:
            kpi_mean = computing.smooth(kpi_mean, smooth_size)
        if plot_errorbars:
            kpi_se = results['{}_by_episode_se'.format(kpi_name)][idx_alpha]
        else:
            kpi_se = None
        if plot_alphas:
            alpha_mean_by_episode_mean = results['alpha_mean_by_episode_mean_by_alpha'][idx_alpha]
        else:
            alpha_mean_by_episode_mean = None
        fig = plot_rmse_by_episode(kpi_mean, rmse_se_values=kpi_se, kpi_name=kpi_name.upper(),
                                   min_rmse=min_kpi, max_rmse=max_kpi,
                                   color=color, linestyle="solid",
                                   alphas=alpha_mean_by_episode_mean,
                                   max_alpha=max_alpha, color_alphas=color,
                                   fontsize=fontsize, plot_scales=plot_scales, fig=fig, legend_rmse=False, legend_alpha=False)
        legend_label += ["{}".format(td_adap_name) + r"($\lambda_{max}$" + "={:.2f})".format(lambda_max) + r", $\alpha_0$=" + "{:.2f}".format(alpha)]

    return fig, legend_label
#--------------------------- Auxiliary functions ------------------------------


#------------------ Possible environments ------------------#
#-- 1D gridworld
resultsdir = resultsdir_grid
nstates = 19 # Number of states excluding terminal states
env_grid1d = gridworlds.EnvGridworld1D(length=nstates+2)
max_time_steps = None
start_state = int((nstates + 1) / 2)
weights_rmse = False
env_desc = str(nstates) + r"-state 1D gridworld environment"

#-- Mountain Car
resultsdir = resultsdir_mountain

# Read the environment on which the analysis will be carried out
# Set the state value function of the environment for the random policy, so that we can compute the RMSE of each experiment
gamma_benchmark = 1.0 #0.9
lambda_benchmark = 1.0 #0.0
nepisodes = 30000 #1000 #10000
filename = resultsdir_mountain + "/mountaincar_BENCHMARK_gamma={:.2f}_lambda={:.1f}_alpha=1.0_adj=True_episodes={},maxt=500,nx=20,nv=20.pickle" \
            .format(gamma_benchmark, lambda_benchmark, nepisodes)
file = open(filename, mode="rb")
dict_benchmark = pickle.load(file)
file.close()
print("True value function read from '{}'".format(filename))

# Create the environment on which tests will be run
# Currently (2022/05/0) we need to do this just because the MountainCarDiscrete environment has changed definition
# w.r.t. to the MountainCarDiscrete environment saved in the pickle file, e.g. there are new methods defined such as setV().
# If the definition of the saved environment (in dict_benchmark['env']) is the same as the current definition of the
# MountainCarDiscrete environment, then we can just use the saved environment as environment on which test are run.
env_mountain = mountaincars.MountainCarDiscrete(dict_benchmark['env'].nv)
max_time_steps = dict_benchmark['params_test']['max_time_steps']    # Use this when the test_obj is saved as part of the pickle file read above
start_state = None  # Initial state chosen at random at the start of each episode
nstates = np.prod(env_mountain.shape)
# Use the state counts when estimating the true value function as weights to compute the RMSE and MAPE as they give
# information about how reliable is the estimation of the state value for those states.
weights_rmse = env_mountain.reshape_from_2d_to_1d(dict_benchmark['counts'])
env_desc = "Mountain Car environment"

# Store the value function read in the env_mountain object
env_mountain.setV( env_mountain.reshape_from_2d_to_1d(dict_benchmark['V']) )
# Check
env_mountain.getV()[:10]
plt.figure()
plt.imshow(env_mountain.reshape_from_1d_to_2d(env_mountain.getV()) / max_time_steps)
plt.colorbar()
plt.title("gamma = {:.2f}".format(gamma_benchmark))
#plt.show()

# Define the uniform distribution on which the intitial state is chosen
# We must EXCLUDE terminal states (i.e. those with positions >= 0.5)
idx_states_non_terminal = env_mountain.get_indices_for_non_terminal_states()
env_mountain.isd = np.array([1.0 / len(idx_states_non_terminal) if idx in idx_states_non_terminal else 0.0
                         for idx in range(env_mountain.getNumStates())])
#print("ISD:", env_mountain.isd)
print("Steps: dx = {:.3f}, dv = {:.3f}".format(env_mountain.dx, env_mountain.dv))
print("Positions: {}".format(env_mountain.get_positions()))
print("Velocities: {}".format(env_mountain.get_velocities()))
#------------------ Possible environments ------------------#



############################ EXPERIMENT SETUP #################################
# The environment
env = env_grid1d
env = env_mountain

# What experiments to run
run_mc = True
run_td = True
run_atd = True
run_hatd = True

# Simulation setup
gamma = 1.0 #0.9

# alpha (when adjust_alpha = True)
# all algorithms start at alpha = 1.0
alphas_td = alphas_atd = alphas_hatd = [1.0]
alpha_mc = alphas_td[0]

# alpha (when adjust_alpha = False)
alphas_td = [0.8, 0.8, 0.8, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1] # Optimum for TD(0.8) where 0.8 is the optimum lambda
# alphas_td = [8, 7.5, 7, 3, 2, 1]        # 10 times larger than the optimum in constant-alpha case
alphas_atd = [0.6] # Optimum for ATD with lambda_max = 0.99
alphas_hatd = [0.4] # Optimum for HATD with lambda_max = 0.80

adjust_alpha = True
adjust_alpha_by_episode = False
alpha_update_type = AlphaUpdateType.EVERY_STATE_VISIT

# lambda
#lambdas = [0.0, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95]
#lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]
#lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
lambda_min = 0.0
lambda_max_atd = 1.0
lambda_max_hatd = 1.0 # 0.80   # Use 0.80 when using FIRST_STATE_VISIT update of alpha

seed = 1717
nexperiments = 5
nepisodes = 5000
verbose = True
verbose_period = max(1, int( nepisodes / 10 ))
verbose_convergence = False
debug = False
plot = True

# Method to set the initial value function at the start of the experiment
reset_method = ResetMethod.ALLZEROS
#reset_method = ResetMethod.RANDOM_NORMAL
reset_params = dict({'loc': 0.5, 'scale': 1.0})

# Possible policies and learners for agents
pol_rw = random_walks.PolRandomWalkDiscrete(env)
learner_mc = mc.LeaMCLambda(env, gamma=gamma, lmbda=1.0, alpha=1.0, adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode,
                            alpha_update_type=alpha_update_type,
                            reset_method=reset_method, reset_params=reset_params, reset_seed=seed)
learner_td = td.LeaTDLambda(env, gamma=gamma, adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode,
                            alpha_update_type=alpha_update_type,
                            reset_method=reset_method, reset_params=reset_params, reset_seed=seed)
learner_atd = td.LeaTDLambdaAdaptive(env, gamma=gamma, adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode,
                                    alpha_update_type=alpha_update_type,
                                    adaptive_type=td.AdaptiveLambdaType.ATD,
                                    reset_method=reset_method, reset_params=reset_params, reset_seed=seed)
learner_hatd = td.LeaTDLambdaAdaptive(env, gamma=gamma, adjust_alpha=adjust_alpha, adjust_alpha_by_episode=adjust_alpha_by_episode,
                                    alpha_update_type=alpha_update_type,
                                    adaptive_type=td.AdaptiveLambdaType.HATD,
                                    reset_method=reset_method, reset_params=reset_params, reset_seed=seed)


############################### NOTES ABOUT TD(lambda) ##################################
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
############################### NOTES ABOUT TD(lambda) ##################################


# Suffixes to use in the name of the pickle files containing the results
# Note: we separate "experiments" with comma because that parameter is LESS important than the rest, like lambda, alpha, etc.
weights_mode = ""
if isinstance(weights_rmse, bool) and weights_rmse == True:
    # We use the state counts observed in the current estimation of the value function as weights
    weights_mode = "weighted"
elif isinstance(weights_rmse, np.ndarray):
    # We use the state counts observed in the estimation of the TRUE value function as weights
    weights_mode = "weightedTrue"
suffix = "adj={}_episodes={},experiments={}_{}RMSE" \
        .format(adjust_alpha, nepisodes, nexperiments, weights_mode)

save = False

#------------------------------------ ANALYZE RMSE -------------------------------------#
#-- When analyzing RMSE as a function of alpha FIXED
if run_td and False:
    # Goal: reproduce the plot in Sutton, pag. 295
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alphas_td = alphas
    alphas_atd = alphas
    alphas_hatd = alphas

    time_start = timer()

    results_td = []

    for idx_lambda, lmbda in enumerate(lambdas):
        print("\n******* lambda {} of {}: {:.2f} ******".format(idx_lambda + 1, len(lambdas), lmbda))
        rmse_last_episode_mean_by_alpha = []
        rmse_last_episode_se_by_alpha = []
        rmse_mean_by_alpha = []
        rmse_se_by_alpha = []
        mape_mean_by_alpha = []
        mape_se_by_alpha = []
        rmse_by_episode_mean_by_alpha = []
        rmse_by_episode_se_by_alpha = []
        mape_by_episode_mean_by_alpha = []
        mape_by_episode_se_by_alpha = []
        n_by_episode_by_alpha= []
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
            N_mean, rmse_last_episode_mean, rmse_last_episode_se, rmse_last_episode_n, \
                RMSE_by_episode_mean, RMSE_by_episode_se, \
                MAPE_by_episode_mean, MAPE_by_episode_se, n_by_episode, \
                learning_info = sim.simulate(nexperiments=nexperiments,
                                             nepisodes=nepisodes,
                                             max_time_steps=max_time_steps,
                                             start=start_state,
                                             weights_rmse=weights_rmse,
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
            mape_mean_by_alpha += [ np.mean(MAPE_by_episode_mean[1:]) ]
            mape_se_by_alpha += [ np.mean(MAPE_by_episode_se[1:]) ]

            rmse_by_episode_mean_by_alpha += [ RMSE_by_episode_mean ]
            rmse_by_episode_se_by_alpha += [ RMSE_by_episode_se ]
            mape_by_episode_mean_by_alpha += [ MAPE_by_episode_mean ]
            mape_by_episode_se_by_alpha += [ MAPE_by_episode_se ]
            n_by_episode_by_alpha += [ n_by_episode ]

        results_td += [{'nexperiments': nexperiments,
                        'nepisodes': nepisodes,
                        'lambda': lmbda,
                        'alphas': alphas_td,
                        'rmse_mean': rmse_mean_by_alpha,
                        'rmse_se': rmse_se_by_alpha,
                        'mape_mean': mape_mean_by_alpha,
                        'mape_se': mape_se_by_alpha,
                        'rmse_n': rmse_last_episode_n * nepisodes,
                        'rmse_last_episode_mean': rmse_last_episode_mean_by_alpha,
                        'rmse_last_episode_se': rmse_last_episode_se_by_alpha,
                        'rmse_last_episode_n': nexperiments,
                        'rmse_by_episode_mean': RMSE_by_episode_mean,
                        'rmse_by_episode_se': RMSE_by_episode_se,
                        'mape_by_episode_mean': mape_by_episode_mean_by_alpha,
                        'mape_by_episode_se': mape_by_episode_se_by_alpha,
                        'n_by_episode': n_by_episode_by_alpha,
                        'learning_info': learning_info
                        }]

    time_end = timer()
    exec_time = time_end - time_start
    print("Execution time for TD: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

    # Save
    if save:
        file = open(resultsdir + "/td_gamma={:.2f}_lambdas={}_".format(gamma, lambdas) + suffix + ".pickle", mode="wb")  # "b" means binary mode (needed for pickle.dump())
        pickle.dump(results_td, file)
        file.close()
#------------------------------------ ANALYZE RMSE -------------------------------------#


#-------------------------------- ANALYZE CONVERGENCE ----------------------------------#
# When analyzing CONVERGENCE of the algorithm with possibly adjustable alpha
if run_mc:
    # ==> MC = Lambda-return with lambda = 1.0 a.k.a. OFFLINE TD(1)
    # (2022/05/17) NOTE that online TD(1) is NOT Monte-Carlo because the online TD(lambda) is an approximation of the lambda-return
    # and doing TD(1) will quite likely diverge if alpha is not decreased aggressively
    # (see my notes on loose paper that compare OFFLINE
    time_start = timer()

    print("\n******* Running MC: alpha = {:.2f} ******".format(alpha_mc))

    learner_mc.setParams()
    agent = GenericAgent(pol_rw, learner_mc)

    # Run the simulation and store the results
    sim = simulators.Simulator(env, agent, seed=seed, debug=debug)
    N_mean, rmse_last_episode_mean, rmse_last_episode_se, rmse_last_episode_n,\
        RMSE_by_episode_mean, RMSE_by_episode_se, \
        MAPE_by_episode_mean, MAPE_by_episode_se, n_by_episode, \
        learning_info = sim.simulate(nexperiments=nexperiments,
                                     nepisodes=nepisodes,
                                     max_time_steps=max_time_steps,
                                     start=start_state,
                                     weights_rmse=weights_rmse,
                                     verbose=verbose,
                                     verbose_period=verbose_period,
                                     plot=plot)
    results_mc = {  'lambda': learner_mc.lmbda,
                    'alpha': learner_mc.alpha,
                    'alpha_mean_by_episode_mean': learning_info['alpha_mean_by_episode_mean'],
                    'lambda_mean_by_episode_mean': learning_info['lambda_mean_by_episode_mean'],
                    'V_abs_mean_by_episode_mean': learning_info['V_abs_mean_by_episode_mean'],
                    'deltaV_abs_mean_by_episode_mean': learning_info['deltaV_abs_mean_by_episode_mean'],
                    'deltaV_max_signed_by_episode_mean': learning_info['deltaV_max_signed_by_episode_mean'],
                    'deltaV_rel_max_signed_by_episode_mean': learning_info['deltaV_rel_max_signed_by_episode_mean'],
                    'rmse_by_episode_mean': RMSE_by_episode_mean,
                    'rmse_by_episode_se': RMSE_by_episode_se,
                    'rmse_last_episode_mean': rmse_last_episode_mean,
                    'rmse_last_episode_se': rmse_last_episode_se,
                    'rmse_last_episode_n': rmse_last_episode_n,
                    'mape_by_episode_mean': MAPE_by_episode_mean,
                    'mape_by_episode_se': MAPE_by_episode_se,
                    'n_by_episode': n_by_episode,
                    'learning_info': learning_info
                   }

    time_end = timer()
    exec_time = time_end - time_start
    print("Execution time for TD: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

    # Save
    file = open(resultsdir + "/mc_gamma={:.2f}_lambda=1.0_alpha={:.2f}_" \
                .format(gamma, alpha_mc) + suffix + ".pickle",
                mode="wb")  # "b" means binary mode (needed for pickle.dump())
    pickle.dump(results_mc, file)
    file.close()

    file = open(resultsdir + "/mc.pickle", mode="wb")  # "b" means binary mode (needed for pickle.dump())
    pickle.dump(results_mc, file)
    file.close()

    if False:
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
                RMSE_by_episode_mean, RMSE_by_episode_se, \
                MAPE_by_episode_mean, MAPE_by_episode_se, n_by_episode, \
                learning_info = sim.simulate(nexperiments=nexperiments,
                                             nepisodes=nepisodes,
                                             max_time_steps=max_time_steps,
                                             start=start_state,
                                             weights_rmse=weights_rmse,
                                             verbose=verbose,
                                             verbose_period=verbose_period,
                                             verbose_convergence=verbose_convergence,
                                             plot=False)

        rmse_mean = np.mean(RMSE_by_episode_mean[1:])
        rmse_se = np.mean(RMSE_by_episode_se[1:])
        rmse_n = rmse_last_episode_n * nepisodes

        results_td += [{'nexperiments': nexperiments,
                        'nepisodes': nepisodes,
                        'lambda': lmbda,
                        'alpha': alpha,
                        'alpha_mean_by_episode_mean': learning_info['alpha_mean_by_episode_mean'],
                        'lambda_mean_by_episode_mean': learning_info['lambda_mean_by_episode_mean'],
                        'V_abs_mean_by_episode_mean': learning_info['V_abs_mean_by_episode_mean'],
                        'deltaV_abs_mean_by_episode_mean': learning_info['deltaV_abs_mean_by_episode_mean'],
                        'deltaV_max_signed_by_episode_mean': learning_info['deltaV_max_signed_by_episode_mean'],
                        'deltaV_rel_max_signed_by_episode_mean': learning_info['deltaV_rel_max_signed_by_episode_mean'],
                        'rmse_mean': rmse_mean,
                        'rmse_se': rmse_se,
                        'rmse_n': rmse_n,
                        'rmse_last_episode_mean': rmse_last_episode_mean,
                        'rmse_last_episode_se': rmse_last_episode_se,
                        'rmse_last_episode_n': rmse_last_episode_n,
                        'rmse_by_episode_mean': RMSE_by_episode_mean,
                        'rmse_by_episode_se': RMSE_by_episode_se,
                        'mape_by_episode_mean': MAPE_by_episode_mean,
                        'mape_by_episode_se': MAPE_by_episode_se,
                        'n_by_episode': n_by_episode,
                        'learning_info': learning_info
                        }]

    time_end = timer()
    exec_time = time_end - time_start
    print("Execution time for TD: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

    # Save
    if save:
        file = open(resultsdir + "/td_gamma={:.2f}_lambdas={}_alphamean={:.2f}_" \
                                .format(gamma, lambdas, np.mean(alphas_td)) + suffix + ".pickle", mode="wb")  # "b" means binary mode (needed for pickle.dump())
        pickle.dump(results_td, file)
        file.close()

if run_atd:
    time_start = timer()

    results_atd = run_td_adap(learner_atd, lambda_min, lambda_max_atd, alphas_atd, start=start_state, seed=seed, debug=debug)

    time_end = timer()
    exec_time = time_end - time_start
    print("Execution time for ATD: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

    # Save
    if save:
        file = open(resultsdir + "/td_ATD_gamma={:.2f}_lambdamax={:.2f}_alphas={}_".format(gamma, lambda_max_atd, alphas_atd) + suffix + ".pickle", mode="wb")
        pickle.dump(results_atd, file)
        file.close()

if run_hatd:
    time_start = timer()

    results_hatd = run_td_adap(learner_hatd, lambda_min, lambda_max_hatd, alphas_hatd, start=start_state, seed=seed, debug=debug)

    time_end = timer()
    exec_time = time_end - time_start
    print("Execution time for HATD: {:.1f} sec, {:.1f} min".format(exec_time, exec_time / 60))

    # Save
    if save:
        file = open(resultsdir + "/td_HATD_gamma={:.2f}_lambdamax={:.2f}_alphas={}_".format(gamma, lambda_max_hatd, alphas_hatd) + suffix + ".pickle", mode="wb")
        pickle.dump(results_hatd, file)
        file.close()
#-------------------------------- ANALYZE CONVERGENCE ----------------------------------#


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
# kpi_name should contain the KPI to plot WITHOUT the suffix '_by_episode_mean' since this name is used
# when constructing the strings '{}_by_episode_mean' and '{}_by_episode_se'.
kpi_name = "V_abs_mean"; plot_scales = "orig"; min_kpi = None; max_kpi = None; smooth = False
kpi_name = "deltaV_rel_max_signed"; plot_scales = "orig"; min_kpi = -0.1; max_kpi = 0.1; smooth = True
kpi_name = "deltaV_max_signed"; plot_scales = "orig"; min_kpi = None; max_kpi = None; smooth = True
kpi_name = "alpha_mean"; plot_scales = "orig"; min_kpi = 0.01; max_kpi = 1.0; smooth = True
kpi_name = "lambda_mean"; plot_scales = "orig"; min_kpi = 0.0; max_kpi = 1.0; smooth = True
kpi_name = "rmse"; plot_scales = "log"; min_kpi = 0.0; max_kpi = None; smooth = False
kpi_name = "mape"; plot_scales = "log"; min_kpi = 0.0; max_kpi = 1.0; smooth = False

max_alpha = 1.0  # max(alpha_td)
smooth_size = max(10, int(nepisodes / 100) )

savefig = True
plot_errorbars = False
plot_alphas = False
fontsize = 14
legend_loc = "lower left"
colormap = cm.get_cmap("jet")

lambdas_selected = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

figfile = os.path.join(resultsdir, "td_ATD({:.2f})_HATD({:.2f})_gamma={:.2f}_alpha={:.2f}" \
                       .format(lambda_max_atd, lambda_max_hatd, gamma, np.mean(np.r_[alphas_td, alphas_atd, alphas_hatd])) +
                       (adjust_alpha and "_adj" or not adjust_alpha and "_const") + "_{}states_{}episodes_{}{}" \
                       .format(nstates, nepisodes, weights_mode, kpi_name.upper()) + (plot_errorbars and "_errors" or "") + ".png")
                        #.format(nstates, nepisodes) + (plot_errorbars and "_errors" or "") + "_weightedRMSE.png")

if run_mc:
    fig = None
    legend_label = []
    color = "red"
    # What to plot
    kpi_mean = results_mc['{}_by_episode_mean'.format(kpi_name)]
    if smooth:
        kpi_mean = computing.smooth(kpi_mean, smooth_size)
    if plot_errorbars:
        kpi_se = results_mc['{}_by_episode_se'.format(kpi_name)]
    else:
        kpi_se = None
    if plot_alphas:
        alpha_mean_by_episode_mean = results_mc['alpha_mean_by_episode_mean']
    else:
        alpha_mean_by_episode_mean = None
    fig = plot_rmse_by_episode(kpi_mean, rmse_se_values=kpi_se, min_rmse=min_kpi, max_rmse=max_kpi, kpi_name=kpi_name.upper(),
                               color=color, linestyle='dashed',
                               alphas=alpha_mean_by_episode_mean,
                               max_alpha=max_alpha, color_alphas=color,
                               fontsize=fontsize, plot_scales=plot_scales, fig=fig, legend_rmse=False, legend_alpha=False)
    legend_label += [r"MC" + r", $\alpha_0$=" + "{:.2f}".format(alpha_mc)]

# TD(lambda)
if run_td:
    lambdas = [s['lambda'] for s in results_td]
    alphas_td = [s['alpha'] for s in results_td]
    #fig = None; legend_label = []
    for idx_lambda, lmbda in enumerate(lambdas):
        if not lmbda in lambdas_selected:
            continue
        # Map blue to the largest lambda and red to the smallest lambda (most similar to the color scheme used in Sutton, pag. 295)
        if len(alphas_td) == 1:
            color = "blue"
        else:
            color = colormap( 1 - idx_lambda / max((1, len(lambdas)-1)) )
        # What to plot
        kpi_mean = results_td[idx_lambda]['{}_by_episode_mean'.format(kpi_name)]
        if smooth:
            kpi_mean = computing.smooth(kpi_mean, smooth_size)
        if plot_errorbars:
            kpi_se = results_td[idx_lambda]['{}_by_episode_se'.format(kpi_name)]
        else:
            kpi_se = None
        if plot_alphas:
            alpha_mean_by_episode_mean = results_td[idx_lambda]['alpha_mean_by_episode_mean']
        else:
            alpha_mean_by_episode_mean = None
        fig = plot_rmse_by_episode(kpi_mean, rmse_se_values=kpi_se, min_rmse=min_kpi, max_rmse=max_kpi, kpi_name=kpi_name.upper(),
                                   color=color, linestyle='dashed',
                                   alphas=alpha_mean_by_episode_mean,
                                   max_alpha=max_alpha, color_alphas=color,
                                   fontsize=fontsize, plot_scales=plot_scales, fig=fig, legend_rmse=False, legend_alpha=False)
        legend_label += [r"$\lambda$=" + "{:.2f}".format(lmbda) + r", $\alpha_0$=" + "{:.2f}".format(alphas_td[idx_lambda])]

# TD(lambda)-adaptive
if run_atd:
    #fig = None; legend_label = []
    fig, legend_label = plot_results_adap(fig, legend_label, "ATD", results_atd, lambda_max_atd,
                                           color="gray", colormap=colormap,
                                           smooth=smooth, smooth_size=smooth_size,
                                           kpi_name=kpi_name, min_kpi=min_kpi, max_kpi=max_kpi, max_alpha=max_alpha,
                                           plot_scales=plot_scales, fontsize=fontsize)

if run_hatd:
    #fig = None; legend_labels = []
    fig, legend_label = plot_results_adap(fig, legend_label, "HATD", results_hatd, lambda_max_hatd,
                                           color="black", colormap=colormap,
                                           smooth=smooth, smooth_size=smooth_size,
                                           kpi_name=kpi_name, min_kpi=min_kpi, max_kpi=max_kpi, max_alpha=max_alpha,
                                           plot_scales=plot_scales, fontsize=fontsize)

# Finalize plot with reference lines, legend and title
if smooth:
    ax = plt.gca()
    ax.set_ylabel(ax.get_ylabel() + " (smoothed on {}-tap moving window)".format(smooth_size))
plt.axhline(y=0, color="gray")
if plot_scales == "both":
    plt.figlegend(legend_label, legend_loc)
else:
    plt.legend(legend_label, loc=legend_loc, fontsize=fontsize)
#fig.suptitle(str(nstates) + "-state environment: TD(lambda) (alpha " + ((not adjust_alpha and "constant)") or
#             "adjusted by " + ((adjust_alpha_by_episode and "episode") or (not adjust_alpha_by_episode and "state count")) + \
#             ", lambda_min=" + str(lambda_min) + ", lambda_max=" + str(lambda_max_hatd) + ")"),
#             fontsize=np.round(fontsize*1.5))
fig.suptitle(env_desc + ": TD($\lambda$) " + "\n" + \
             r"($\gamma$ = " + "{:.2f}, ".format(gamma) + r"$\alpha$ " + ((not adjust_alpha and "constant)") or "adjusted by " + \
             ((adjust_alpha_by_episode and "episode") or (not adjust_alpha_by_episode and \
                                                         (alpha_update_type == AlphaUpdateType.FIRST_STATE_VISIT and "first-visit " or \
                                                         (alpha_update_type == AlphaUpdateType.EVERY_STATE_VISIT and "every-visit ")) + "state count")) + ")"),
             fontsize=np.round(fontsize*1.5))

if savefig:
    #plt.gcf().subplots_adjust(left=0.15, top=0.75)
    plt.savefig(figfile)
    print("Plot saved to {}".format(figfile))



#--------------------- ALPHA SENSITIVITY: Plot of RMSE vs. alpha -----------------#
kpi_name = "mape"
kpi_mean_name = "{}_mean".format(kpi_name)
kpi_se_name = "{}_se".format(kpi_name)
kpi_last_episode_mean_name = "{}_last_episode_mean".format(kpi_name)
kpi_last_episode_se_name = "{}_last_episode_se".format(kpi_name)
kpi_by_episode_mean_name = "{}_by_episode_mean".format(kpi_name)
kpi_by_episode_se_name = "{}_by_episode_se".format(kpi_name)
max_rmse = 50
max_mape = 1.0
max_kpi = max_rmse if kpi_name == "rmse" else max_mape
min_kpi = 0.98
set_ylims = True

colormap = cm.get_cmap("jet")
savefig = True
plot_rmse_last_episode = False       # Only works for RMSE because for MAPE the last episode's information is not saved explicitly and it is not so easy to retrieve it
plot_errorbars = True
plot_td_adap_together = True
fontsize = 14
legend_loc = 'lower right'

if plot_td_adap_together:
    figfile = os.path.join(resultsdir, "td_ATD({:.2f})_HATD({:.2f})_alpha_const_{}states_{}episodes_{}{}" \
                           .format(lambda_max_atd, lambda_max_hatd, nstates, nepisodes, weights_mode, kpi_name.upper()) +
                           (plot_errorbars and "_errors" or "") + ".png")
else:
    adaptive_type = td.AdaptiveLambdaType.ATD
    lambda_max = lambda_max_atd
    results_td_adap = results_atd
    figfile = os.path.join(resultsdir, "td_{}_alpha_const_lambdamax={:.2f}_{}states_{}episodes" \
                           .format(adaptive_type.name, lambda_max, nstates, nepisodes) + (plot_errorbars and "_errors" or "") + ".png")

ax = plt.figure(figsize=(10,10)).subplots()
legend_label = []

lambdas_selected = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

# TD(lambda)
for idx_lambda, lmbda in enumerate(lambdas):
    if not lmbda in lambdas_selected:
        continue
    # Map blue to the largest lambda and red to the smallest lambda (most similar to the color scheme used in Sutton, pag. 295)
    color = colormap( 1 - idx_lambda / max((1, len(lambdas)-1)) )
    if plot_rmse_last_episode:
        ax.plot(results_td[idx_lambda]['alphas'], results_td[idx_lambda][kpi_last_episode_mean_name], '.--', color=color)
        if plot_errorbars:
            ax.errorbar(results_td[idx_lambda]['alphas'], results_td[idx_lambda][kpi_last_episode_mean_name], color=color,
                        yerr=results_td[idx_lambda][kpi_last_episode_se_name], capsize=4, linestyle='dashed')
        ax.set_ylabel("{} at last episode ({}), averaged over {} experiments".format(kpi_name.upper(), nepisodes, nexperiments), fontsize=fontsize)
    else:
        ax.plot(results_td[idx_lambda]['alphas'], results_td[idx_lambda][kpi_mean_name], '.--', color=color)
        if plot_errorbars:
            ax.errorbar(results_td[idx_lambda]['alphas'], results_td[idx_lambda][kpi_mean_name], color=color,
                        yerr=results_td[idx_lambda][kpi_se_name], capsize=4, linestyle='dashed')
        ax.set_ylabel("Average {} over first {} episodes, averaged over {} experiments".format(kpi_name.upper(), nepisodes, nexperiments), fontsize=fontsize)
    if set_ylims:
        ax.set_ylim((min_kpi, max_kpi))
    ax.set_xlabel(r"$\alpha$", fontsize=fontsize)
    # Square plot
    # Ref: https://www.geeksforgeeks.org/how-to-make-a-square-plot-with-equal-axes-in-matplotlib/
    ax.set_aspect(1. / ax.get_data_ratio(), adjustable='box')
    legend_label += [r"TD($\lambda$={:.2f})".format(lmbda)]

if plot_td_adap_together:
    if plot_rmse_last_episode:
        ax.plot(results_atd['alphas'], results_atd[kpi_last_episode_mean_name], 'x-', color="gray")
        ax.plot(results_hatd['alphas'], results_hatd[kpi_last_episode_mean_name], 'x-', color="black")
        if plot_errorbars:
            ax.errorbar(results_atd['alphas'], results_atd[kpi_last_episode_mean_name], color="gray",
                        yerr=results_atd[kpi_last_episode_se_name], capsize=4)
            ax.errorbar(results_hatd['alphas'], results_hatd[kpi_last_episode_mean_name], color="black",
                        yerr=results_hatd[kpi_last_episode_se_name], capsize=4)
    else:
        ax.plot(results_atd['alphas'], [np.mean(x) for x in results_atd[kpi_by_episode_mean_name]], 'x-', color="gray")
        ax.plot(results_hatd['alphas'], [np.mean(x) for x in results_hatd[kpi_by_episode_mean_name]], 'x-', color="black")
        if plot_errorbars:
            ax.errorbar(results_atd['alphas'], [np.mean(x) for x in results_atd[kpi_by_episode_mean_name]], color="gray",
                        yerr=[np.mean(x) for x in results_atd[kpi_by_episode_se_name]], capsize=4)
            ax.errorbar(results_hatd['alphas'], [np.mean(x) for x in results_hatd[kpi_by_episode_mean_name]], color="black",
                        yerr=[np.mean(x) for x in results_hatd[kpi_by_episode_se_name]], capsize=4)
    legend_label += [r"ATD($\lambda_{max}$" + "={:.2f})".format(lambda_max_atd), r"HATD($\lambda_{max}$" + "={:.2f})".format(lambda_max_hatd)]
else:
    if plot_rmse_last_episode:
        if plot_errorbars:
            ax.errorbar(results_td_adap['alphas'], results_td_adap[kpi_last_episode_mean_name], color="black",
                        yerr=results_td_adap[kpi_last_episode_se_name], capsize=4)
        else:
            ax.plot(results_td_adap['alphas'], results_td_adap[kpi_last_episode_mean_name], 'x-', color="black")
    else:
        if plot_errorbars:
            ax.errorbar(results_td_adap['alphas'], [np.mean(x) for x in results_td_adap[kpi_by_episode_mean_name]], color="black",
                        yerr=[np.mean(x) for x in results_td_adap[kpi_by_episode_se_name]], capsize=4)
        else:
            ax.plot(results_td_adap['alphas'], [np.mean(x) for x in results_td_adap[kpi_by_episode_mean_name]], 'x-', color="black")
    legend_label += [adaptive_type.name + r" $\lambda_{max}$" + "={:.2f})".format(lambda_max)]
plt.legend(legend_label, loc=legend_loc, fontsize=fontsize)
plt.title("Average " + ("weighted " if weights_mode == "weightedTrue" or weights_mode == "weighted" else " ") + kpi_name.upper(), fontsize=fontsize)
plt.suptitle(env_desc + r": TD($\lambda$) algorithms for fixed $\alpha$".format(nstates), fontsize=np.round(fontsize*1.5))

if savefig:
    #plt.gcf().subplots_adjust(left=0.15, top=0.75)
    plt.savefig(figfile)
