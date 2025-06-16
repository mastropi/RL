# -*- coding: utf-8 -*-
"""
Created on Tue Jun 03 11:23:57 2025

@author: Daniel Mastropietro
@description: Runs the FV estimation of the stationary probability of sets of interest of a diffusion process, such as Ornstein-Uhlenbeck.
"""

if __name__ == "__main__":
    # Only run this when running the script, o.w. it may give an error when importing functions if setup.py is not found
    import runpy
    runpy.run_path('../../setup.py')


#--- 1) Choice of S values to be used as boundary of the attraction set A in the OU process with piecewise linear drift alpha modeling grid frequency
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Python.lib.estimators.diffusion import calculate_probability

# OU process characteristics with piecewise linear drift
mu = 0.0
sigma = 0.04
r = 0.6     # Capacity reserved for FCR-N (in GW)
x = 1.4     # Capacity reserved for FCR-D (in GW)

S = np.linspace(0.01, 0.2, 100)
# Probability of being outside (-S, S) for each possible S (calculation is instantaneous)
qS = [1 - calculate_probability((-s, s), r, x, sigma) for s in S]

plt.figure()
plt.plot(S, qS, 'b-')
plt.gca().set_yscale('log')
prob_refs = 10.0**(-np.arange(1, 9))
for q in prob_refs:
    plt.axhline(q, color="black", linestyle="dashed")

# Choose the S values on which FV should be run as the S value read out from the reference probabilities just plotted
S_values = np.zeros_like(prob_refs)
qS_reversed = qS[::-1]
S_reversed = S[::-1]
for i, q in enumerate(prob_refs):
    idx = np.searchsorted(qS_reversed, q)
    S_values[i] = S_reversed[idx]

# Define the sets on which we will run the simulation, i.e. the sets whose probability we are interested in estimating and their true occurrence probability
df_simulate = pd.DataFrame(np.c_[S_values, [1 - calculate_probability((-s, s), r, x, sigma) for s in S_values]], columns=['C', 'pC'])

# How do we choose A now?
# Say 80% of each size S?
# What's the probability of leaving A?
# Choose T as 10/q, where q = 1 - p, the probability of being outside A, so we observe at least 10 exits...?
factor_A = 0.80
min_num_cycles = 20
df_simulate['A'] = factor_A * df_simulate['C']
df_simulate['pA'] = [calculate_probability((-a, a), r, x, sigma) for a in df_simulate['A']]
df_simulate['T'] = (min_num_cycles / (1 - df_simulate['pA'])).astype(int)
print(f"Chosen C and A sizes and their probabilities to simulate:\n{df_simulate}")


#--- 2) Simulation using the execution parameters estimated by the previous step and stored in the df_simulate data frame
# Now run the simulation for each setup of set of interest C and absorption set A included in the df_simulate data frame defined above
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from timeit import default_timer as timer
from time import process_time
import sympy
from Python.lib.environments.diffusion import EnvDiffusion, func_drift_fcr_piecewise, func_noise_gaussian
from Python.lib.estimators.diffusion import estimate_probabilities_in_sets_of_interest
from Python.lib.simulators.diffusion import SimulatorDiffusionFV

rootdir = r"E:\Daniel\Projects\PhD-RL-Toulouse\projects\RL-004-Energy"
resultsdir = os.path.realpath(rootdir + "/results")
print(f"Results will be saved to {resultsdir}")

# Reflected process?
reflect = True

# Define the time interval on which the simulation is run
dt = 0.05 #0.01

# Environment
env_ou = EnvDiffusion(func_drift=func_drift_fcr_piecewise, func_noise=func_noise_gaussian, reflect=reflect, mu=mu, dt=dt, sigma=sigma, r=r, x=x)

# Simulator for both FV and MC
sim = SimulatorDiffusionFV(env_ou, None, debug=False)

# Number of replications to run
nrep = 9
seed_base = 1313  # 1317
save = False

# Data frame with results for each method (FV and MC)
df_results_fv = pd.DataFrame(columns=['case'] + list(df_simulate.columns) + ['p(FV)', 'nsteps', 'nsteps_mc', 'nsteps_fv', 'expected_cycle_time', 'n_cycles'],
                             index=np.arange(len(df_simulate) * nrep))
df_results_mc = pd.DataFrame(columns=['case'] + list(df_simulate.columns) + ['p(MC)', 'nsteps', 'expected_cycle_time', 'n_cycles'],
                             index=np.arange(len(df_simulate) * nrep))

if save:
    # Save the results iteratively in case the process gets interrupted, so that we do not lose what was already run!
    filename_fv = os.path.join(resultsdir, "ou_pwlinear_probability_estimation_fv.csv")
    filename_mc = os.path.join(resultsdir, "ou_pwlinear_probability_estimation_mc.csv")
    pd.DataFrame([df_results_fv.columns]).to_csv(filename_fv, header=False, index=False)
    pd.DataFrame([df_results_mc.columns]).to_csv(filename_mc, header=False, index=False)

# Run the simulation on each A-set / C-set combination with a fixed number of particles N
time_start = timer()
cpu_start = process_time()
for row, data in df_simulate.iterrows():
    set_C = sympy.Interval(data.C, +np.Inf)
    set_A = sympy.Interval(-data.A, data.A)
    T = int(data['T'])  #3*int(data['T'])
    N = 100

    # In case we need to skip the simulations that take longer (when e.g. we are testing the process)
    if row + 1 > 3:
        print(f"Simulation {row+1} of {len(df_simulate)} (data={data.T}) skipped!")
    else:
        for rep in range(nrep):
            seed_rep = seed_base + rep*1317
            print(f"\nRunning case {row+1} of {len(df_simulate)}, replication {rep+1} of {nrep}: T={T}, N={N}, A={set_A}, C={set_C}, seed={seed_rep}...")

            # Output row (where the result is stored in the output data frames)
            orow = row*nrep + rep

            print(f"[FV] Fleming-Viot simulation (discretization interval, dt={dt})")

            time_start_fv = timer()
            dict_params_simul = dict({'N': N,
                                      'T': T,
                                      'absorption_set': set_A,
                                      'check_for_stationarity': not reflect,
                                      })
            probas_stationary, info = sim.run(  dict_params_simul,
                                                sets_of_interest=[set_C],
                                                start_state=0.0,
                                                store_trajectories=True,
                                                seed=seed_rep,
                                                verbose=False, verbose_period=1, plot={'MC': False, 'FV': True})
            df_results_fv.iloc[orow]['case'] = row + 1
            df_results_fv.iloc[orow][df_simulate.columns] = data.apply(lambda x: "{:.3g}".format(x))
            df_results_fv.iloc[orow][['p(FV)', 'nsteps', 'nsteps_mc', 'nsteps_fv', 'expected_cycle_time', 'n_cycles']] = \
                [probas_stationary[set_C] if len(probas_stationary) > 0 else np.nan, info['nsteps'], info['nsteps'] - info['FV'].get('last_time_observed', 0), info['FV'].get('last_time_observed', 0), info['MC']['expected_cycle_time'], info['MC']['n_cycles']]
            # Add result to output file
            if save:
                df_results_fv.iloc[orow:orow+1].to_csv(filename_fv, header=False, index=False, mode="a")
            elapsed_time = timer() - time_start_fv
            print(f"FV simulation took {elapsed_time/60:.1f} min, {elapsed_time:.1f} sec")

            print(f"[MC] Monte-Carlo simulation (case {row+1} of {len(df_simulate)}, replication {rep+1} of {nrep}, discretization interval, dt={dt})")
            time_start_mc = timer()
            expected_cycle_time, n_cycles, dist_exit_state, info_mc = sim._run_simulation_mc(   dict({'T': info['nsteps'], 'absorption_set': set_A}),
                                                                                                start_state=0.0, store_trajectory=True,
                                                                                                check_for_stationarity=not reflect, burnin_for_stationarity_check=30,
                                                                                                seed=seed_rep, verbose=False, verbose_period=10, plot=False)
            probas_stationary_mc = estimate_probabilities_in_sets_of_interest(sim.trajectory_mc, [set_C])
            df_results_mc.iloc[orow]['case'] = row + 1
            df_results_mc.iloc[orow][df_simulate.columns] = data.apply(lambda x: "{:.3g}".format(x))
            df_results_mc.iloc[orow][['p(MC)', 'nsteps', 'expected_cycle_time', 'n_cycles']] = [probas_stationary_mc[set_C], info_mc['last_time_observed'], expected_cycle_time, n_cycles]
            # Add result to output file
            if save:
                df_results_mc.iloc[orow:orow+1].to_csv(filename_mc, header=False, index=False, mode="a")
            elapsed_time = timer() - time_start_mc
            print(f"MC simulation took {elapsed_time / 60:.1f} min, {elapsed_time:.1f} sec")
    print()
elapsed_time = timer() - time_start
cpu_time = process_time() - cpu_start
print(f"SIMULATION PROCESS took {elapsed_time / 60:.1f} min (CPU: {cpu_time / 60:.1f} min), {elapsed_time:.1f} sec (CPU: {cpu_time / 60:.1f} sec)")
if save:
    print(f"Output files\n'{filename_fv}' (FV)\n'{filename_mc} (MC)\ncreated with the estimation results.")

# Read saved results if needed
if save:
    df_results_fv = pd.read_csv(os.path.join(resultsdir, f"ou_pwlinear_probability_estimation_fv_dt={dt}.csv"))
    df_results_mc = pd.read_csv(os.path.join(resultsdir, f"ou_pwlinear_probability_estimation_mc_dt={dt}.csv"))
    df_simulate = df_results_fv.groupby(['C', 'pC', 'A', 'pA', 'T']).size().reset_index().rename({0: 'nrep'}, axis=1)
    # Run this if pA incorrectly stores the p(A) as opposed to its complement
    df_simulate['pA'] = 1 - df_simulate['pA']


# Plot results: violins
common_axes = True
violin_width = 0.1
fontsize = 12
_size_layout = int(np.ceil(np.sqrt(len(df_simulate))))
axes = plt.figure().subplots(_size_layout, _size_layout)
axes1d = axes.reshape(-1)
yvalues = np.r_[df_results_fv['p(FV)'], df_results_mc['p(MC)']]
ymin, ymax = np.nanmin(yvalues[yvalues > 0]), np.nanmax(yvalues)
for row, data in df_simulate.iterrows():
    ax = axes1d[row]
    idx_case = row + 1
    ind_fv = df_results_fv['case'] == idx_case
    ind_mc = df_results_mc['case'] == idx_case
    if sum(ind_fv) > 0:
        toplot_fv = np.array(df_results_fv["p(FV)"][ind_fv]).astype(float)  # astype(float) removes the `object` type of the number which generates an error in plotting
        # Estimation errors
        # Note that we do NOT compute the absolute error because we want to assess whether there is a positive or negative bias
        errors_fv = {'mean': np.mean((toplot_fv - data.pC)),
                     'median': np.median((toplot_fv - data.pC)),
                     'mean_rel': np.mean((toplot_fv / data.pC - 1)),
                     'median_rel': np.median((toplot_fv / data.pC - 1))}
        violinparts_fv = ax.violinplot([toplot_fv], positions=[1], widths=violin_width, showmedians=True, showmeans=False, showextrema=True)
        violinparts_fv['bodies'][0].set_facecolor("green")
        jitter = violin_width * np.random.uniform(-0.5, 0.5, sum(ind_fv))
        ax.plot(1 + jitter, toplot_fv, 'k.')
    if sum(ind_mc) > 0:
        toplot_mc = np.array(df_results_mc["p(MC)"][ind_mc]).astype(float)
        # Estimation errors
        # Note that we do NOT compute the absolute error because we want to assess whether there is a positive or negative bias
        errors_mc = {'mean': np.mean((toplot_mc - data.pC)),
                     'median': np.median((toplot_mc - data.pC)),
                     'mean_rel': np.mean((toplot_mc / data.pC - 1)),
                     'median_rel': np.median((toplot_mc / data.pC - 1))}
        violinparts_mc = ax.violinplot(toplot_mc, positions=[2], widths=violin_width, showmedians=True, showmeans=False, showextrema=True)
        violinparts_mc['bodies'][0].set_facecolor("red")
        jitter = violin_width * np.random.uniform(-0.5, 0.5, sum(ind_mc))
        ax.plot(2 + jitter, toplot_mc, 'k.')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_ticks([1, 2])
    ax.xaxis.set_ticklabels(["FV", "MC"])
    ax.axhline(data.pC, color="gray", linestyle="dashed")
    ax.set_title(rf"T = {int(data['T'])}, A = ({-data.A:.3f}, {data.A:.3f}), $p(\overline{{A}})$ = {0.5*(1 - data.pA):.1e}, C = ({data.C:.3f}, Inf), $p(C)$ = {data.pC:.1e}", fontsize=int(0.8*fontsize))
    ax.set_yscale('log')
    if sum(ind_fv) > 0 and sum(ind_mc) > 0:
        ax.legend([violinparts_fv['bodies'][0], violinparts_mc['bodies'][0]],
                  [rf"FV - $\phi(n)$={int(np.mean(df_results_fv[ind_fv]['nsteps']))}, $\epsilon$: $\phi$={errors_fv['mean']:.1g} ({errors_fv['mean_rel']*100:.1f}%), M={errors_fv['median']:.1g} ({errors_fv['median_rel']*100:.1f}%)",
                   rf"MC - $\phi(n)$={int(np.mean(df_results_mc[ind_mc]['nsteps']))}, $\epsilon$: $\phi$={errors_mc['mean']:.1g} ({errors_mc['mean_rel']*100:.1f}%), M={errors_mc['median']:.1g} ({errors_mc['median_rel']*100:.1f}%)"], fontsize=int(0.8*fontsize))
    if common_axes:
        ax.set_ylim((0.9*ymin, 1.1*ymax))
    # Add labels if at the left and bottom borders
    if np.unravel_index(row, axes.shape)[0] == axes.shape[0] - 1:
        ax.set_xlabel(f"Method (up to {int(nrep)} replications)")
    if np.unravel_index(row, axes.shape)[1] == 0:
        ax.set_ylabel("Estimated probability p(C) (log)")
plt.suptitle("Sensitivity analysis of FV vs. MC as a function of event probability", fontsize=int(1.2*fontsize))
plt.gcf().set_figheight(9)
plt.gcf().set_figwidth(16)
plt.get_current_fig_manager().window.showMaximized()
#plt.gcf().subplots_adjust(left=0.15, top=0.75)
#plt.tight_layout()#rect=(0.1, 0.1, 0.9, 0.9))

# Save the plot
# Note dpi=400, which is an important quality leap over dpi=300! (based on zooming in on the image in TeX Studio with the magnifier utility)
filename_fig = os.path.join(resultsdir, "ou_pwlinear_probability_estimation.png")
plt.savefig(
    filename_fig,
    dpi=400,
    bbox_inches="tight",
    transparent=False,
)
print(f"Figure saved to {filename_fig}")
