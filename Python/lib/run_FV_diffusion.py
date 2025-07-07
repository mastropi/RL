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


#--- Auxiliary functions
def prepare4violin(df, groupvar, x):
    """
    Prepares column `x` in data frame `df` for violin plots by each SORTED value of variable `groupvar`

    Return: tuple
    Tuple with the following two elements:
    - list of length equal to the number of group values where each element is an array of possibly varying length with the data to plot on each violin.
    - list with the sorted group values labeling the different violins to plot.
    """
    assert len(groupvar) == 1
    group_values = sorted(df[groupvar].value_counts().index)
    toplot = [[]] * nviolins
    for i, group_value in enumerate(group_values):
        toplot[i] = np.array(df[x][df[groupvar] == group_value]).astype(float)  # astype(float) removes the `object` type of the number which generates an error in plotting
    return toplot, group_values

def set_violincolor(violinparts, color):
    "Sets ALL the graphical parts making up a violin to the given color. Note that `violinparts` is a dictionary of plot handles returned by plt.violinplot()"
    for key, part in violinparts.items():
        if key == "bodies":
            for body in part:
                body.set_color(color)
        else:
            part.set_color(color)

def compute_statistics(df_results, groupvars, x):
    "Compute statistics on variable `x` by each `groupvars` combination"
    df_analyze = df_results.groupby(groupvars)[x].agg(['count', 'mean', 'median', 'std', 'mad'])
    df_analyze.reset_index(inplace=True)
    df_analyze['se_robust'] = df_analyze['mad'] / np.sqrt(df_analyze['count'])
    return df_analyze
#--- Auxiliary functions


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

# Choose the S values defining the border of the C set of interest (e.g. C = "complement of (-S, S)") on which FV should be evaluated
# as the S value read out from the reference probabilities just plotted
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
# Choose T as 10/q (up to a LOWER BOUND), where q = 1 - p, the probability of being outside A, so we observe at least 10 or 20 exits...?
T_MIN = 500
factor_A = 0.80
min_num_cycles = 20
df_simulate['A'] = factor_A * df_simulate['C']
df_simulate['pA'] = [calculate_probability((-a, a), r, x, sigma) for a in df_simulate['A']]
df_simulate['T'] = (min_num_cycles / (1 - df_simulate['pA'])).astype(int)
# Lower bound T by 500 so that we don't have a too small value...
df_simulate['T'] = df_simulate['T'].apply(lambda x: max(T_MIN, x))
print(f"Chosen C and A sizes and their probabilities to simulate:\n{df_simulate}")

# For each row generate as many rows as number of N values to consider (number of particles)
N_values = [[20, 40, 80, 160]]
nvalues = len(N_values[0])
df_simulate_N = pd.DataFrame(columns=list(df_simulate.columns) + ['N'])
for c in df_simulate.columns:
    df_simulate_N[c] = np.repeat(df_simulate[c], nvalues)
df_simulate_N['N'] = np.repeat(N_values, len(df_simulate), axis=0).reshape(-1)
df_simulate_N.reset_index(drop=True, inplace=True)
df_simulate = df_simulate_N
del df_simulate_N
print(f"Cases to simulate:\n{df_simulate}")


#--- 2) Simulation using the execution parameters estimated by the previous step and stored in the df_simulate data frame
# Now run the simulation for each setup of set of interest C and absorption set A included in the df_simulate data frame defined above
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from timeit import default_timer as timer
from time import process_time
import sympy
from Python.lib.environments.diffusion import EnvDiffusion, func_drift_fcr_piecewise, func_noise_gaussian
from Python.lib.estimators.diffusion import estimate_probabilities_in_sets_of_interest
from Python.lib.estimators.fv import estimate_stationary_probabilities
from Python.lib.simulators.diffusion import SimulatorDiffusionFV
from Python.lib.utils.basic import get_current_datetime_as_string


rootdir = r"E:\Daniel\Projects\PhD-RL-Toulouse\projects\RL-004-Energy"
resultsdir = os.path.realpath(rootdir + "/results")
print(f"Results will be saved to '{resultsdir}'")

# Levy noise?
use_levy = True

# Reflected process?
reflect = not use_levy

# Define the time interval on which the simulation is run
dt = 0.05 #0.01

# Environment
if use_levy:
    # NOTE: We multiply the estimated jump rate by 20 in order to make the skewness of the frequency distribution larger than with the estimated jump rate (of 0.0175, rather small)
    jump_rate = 0.0175 * 20
    # Shape, location and scale estimated using scipy.stats.lognorm.fit(x) where x is the 0.1-second frequency data from Fingrid for 2025-01 (31 days, ~ 26 Mio. records)
    shape, loc, scale = (0.6766070581549546, 0.013074261175727589, 0.00344392772573485)
    levy = {'lambda': jump_rate, 'fun_dist_jumps': np.random.lognormal, 'params_dist_jumps': {'factor': -1, 'shift': loc, 'mean': np.log(scale), 'sigma': shape}}
else:
    levy = None
env_ou = EnvDiffusion(func_drift=func_drift_fcr_piecewise, func_noise=func_noise_gaussian, reflect=reflect, levy=levy, mu=mu, dt=dt, sigma=sigma, r=r, x=x)

# Simulator for both FV and MC
sim = SimulatorDiffusionFV(env_ou, None, debug=False)

# Number of replications to run
nrep = 9
seed_base = 1313  # 1317
save = True

# Data frame with results for each method (FV and MC)
df_results_fv = pd.DataFrame(columns=['case'] + list(df_simulate.columns) + ['rep', 'seed', 'p(FV)', 'nsteps', 'nsteps_mc', 'nsteps_fv', 'expected_cycle_time', 'n_cycles', 'exec_time', 'cpu_time'],
                             index=np.arange(len(df_simulate) * nrep))
df_results_mc = pd.DataFrame(columns=['case'] + list(df_simulate.columns) + ['rep', 'seed', 'p(MC)', 'nsteps', 'expected_cycle_time', 'n_cycles', 'exec_time', 'cpu_time'],
                             index=np.arange(len(df_simulate) * nrep))

# Run the simulation on each A-set / C-set combination with a fixed number of particles N
time_start = timer()
cpu_start = process_time()
datetime_str = get_current_datetime_as_string(format="filename")

if save:
    # Save the results iteratively in case the process gets interrupted, so that we do not lose what was already run!
    filename_fv = os.path.join(resultsdir, f"ou_pwlinear_probability_estimation_{datetime_str}_levy={use_levy},dt={dt}_fv.csv")
    filename_mc = os.path.join(resultsdir, f"ou_pwlinear_probability_estimation_{datetime_str}_levy={use_levy},dt={dt}_mc.csv")
    pd.DataFrame([df_results_fv.columns]).to_csv(filename_fv, header=False, index=False)
    pd.DataFrame([df_results_mc.columns]).to_csv(filename_mc, header=False, index=False)





#--- SLOWER simulation that repeats the Monte-Carlo simulation even if (T, A) did NOT change and only the number of particles N changed
for row, data in df_simulate.iterrows():
    if use_levy:
        # We estimate the probability of NEGATIVE intervals (-Inf, -c)
        set_C = sympy.Interval(-np.Inf, -data.C)
        set_A = sympy.Interval(-data.A, +np.Inf)
    else:
        # We estimate the probability of POSITIVE intervals (c, +Inf) because we simulate the REFLECTED process
        assert reflect, "Under the pure Gaussian noise, we must reflect the process at ZERO."
        set_C = sympy.Interval(data.C, +np.Inf)
        set_A = sympy.Interval(-data.A, data.A)
    T = int(data['T'])  #3*int(data['T'])
    N = int(data['N'])  #100

    # In case we need to skip the simulations that take longer (when e.g. we are testing the process)
    if row + 1 > +np.Inf: #7:
        print(f"Simulation {row+1} of {len(df_simulate)} (data={data.T}) skipped!")
    else:
        for rep in range(nrep):
            seed_rep = seed_base + rep*1317
            print(f"\nRunning case {row+1} of {len(df_simulate)}, replication {rep+1} of {nrep}: T={T}, N={N}, A={set_A}, C={set_C}, seed={seed_rep}...")

            # Output row (where the result is stored in the output data frames)
            orow = row*nrep + rep

            print(f"[FV] Fleming-Viot simulation (discretization interval, dt={dt})")

            time_start_fv = timer()
            cpu_start_fv = process_time()
            dict_params_simul = dict({'N': N,
                                      'T': T,
                                      'absorption_set': set_A,
                                      'check_for_stationarity': not reflect and not use_levy,
                                      })
            # IMPORTANT: (2025/07/02) We could speed this process up by running the E(T_A) simulation only ONCE and then running on different N particles the FV simulation
            # using the same information estimated by the E(T_A) simulation!! (which is actually the one that takes a lot of time if T is very large)
            # But to do this, we need to store the information obtained from the E(T_A) simulation for each REPLICATION, so that we can use it for the corresponding replication
            # on a different N value. So, it requires some development.
            probas_stationary, info = sim.run(  dict_params_simul,
                                                sets_of_interest=[set_C],
                                                start_state=0.0,
                                                store_trajectories=True,
                                                seed=seed_rep,
                                                verbose=False, verbose_period=1, plot={'MC': False, 'FV': False})
            elapsed_time = timer() - time_start_fv
            cpu_time = process_time() - cpu_start_fv
            print(f"FV simulation took {elapsed_time/60:.1f} min (CPU={cpu_time/60:.1f}), {elapsed_time:.1f} sec")

            # Store and save results
            df_results_fv.iloc[orow]['case'] = row + 1
            df_results_fv.iloc[orow][df_simulate.columns] = data.apply(lambda x: "{:.3g}".format(x))
            df_results_fv.iloc[orow]['rep'] = rep + 1
            df_results_fv.iloc[orow]['seed'] = seed_rep
            df_results_fv.iloc[orow][['p(FV)', 'nsteps', 'nsteps_mc', 'nsteps_fv', 'expected_cycle_time', 'n_cycles', 'exec_time', 'cpu_time']] = \
                [probas_stationary[set_C] if len(probas_stationary) > 0 else np.nan,
                 info['nsteps'],
                 info['nsteps'] - info['FV'].get('last_time_observed', 0),
                 info['FV'].get('last_time_observed', 0),
                 info['MC']['expected_cycle_time'],
                 info['MC']['n_cycles'],
                 elapsed_time,
                 cpu_time]
            # Add result to output file
            if save:
                df_results_fv.iloc[orow:orow+1].to_csv(filename_fv, header=False, index=False, mode="a")

            print(f"[MC] Monte-Carlo simulation (case {row+1} of {len(df_simulate)}, replication {rep+1} of {nrep}, discretization interval, dt={dt})")
            time_start_mc = timer()
            cpu_start_mc = process_time()
            expected_cycle_time, n_cycles, dist_exit_state, info_mc = sim._run_simulation_mc(   dict({'T': info['nsteps'], 'absorption_set': set_A}),
                                                                                                start_state=0.0, store_trajectory=True,
                                                                                                check_for_stationarity=not reflect,
                                                                                                    ## NOTE that checking for stationarity in the Monte-Carlo simulation is just informative
                                                                                                    ## as the information about the exit states and expected cycle time is NOT used
                                                                                                    ## in the estimation of the probability of the state of interest, p(C),
                                                                                                    ## as the FULL trajectory is collected in the trajectory_mc attribute.
                                                                                                    ## (see the function estimate_probabilities_in_sets_of_interest() called just next)
                                                                                                burnin_for_stationarity_check=30,
                                                                                                seed=seed_rep, verbose=False, verbose_period=10, plot=False)
            probas_stationary_mc = estimate_probabilities_in_sets_of_interest(sim.trajectory_mc, [set_C])

            elapsed_time = timer() - time_start_mc
            cpu_time = process_time() - cpu_start_mc
            print(f"MC simulation took {elapsed_time / 60:.1f} min (CPU={cpu_time/60:.1f}), {elapsed_time:.1f} sec")

            # Store and save results
            df_results_mc.iloc[orow]['case'] = row + 1
            df_results_mc.iloc[orow][df_simulate.columns] = data.apply(lambda x: "{:.3g}".format(x))
            df_results_mc.iloc[orow]['rep'] = rep + 1
            df_results_mc.iloc[orow]['seed'] = seed_rep
            df_results_mc.iloc[orow][['p(MC)', 'nsteps', 'expected_cycle_time', 'n_cycles', 'exec_time', 'cpu_time']] = \
                [probas_stationary_mc[set_C], info_mc['last_time_observed'], expected_cycle_time, n_cycles, elapsed_time, cpu_time]
            # Add result to output file
            if save:
                df_results_mc.iloc[orow:orow+1].to_csv(filename_mc, header=False, index=False, mode="a")
    print()
elapsed_time = timer() - time_start
cpu_time = process_time() - cpu_start
print(f"SIMULATION PROCESS took {elapsed_time / 60:.1f} min (CPU: {cpu_time / 60:.1f} min), {elapsed_time:.1f} sec (CPU: {cpu_time / 60:.1f} sec)")
if save:
    print(f"Output files\n'{filename_fv}' (FV)\n'{filename_mc} (MC)\ncreated with the estimation results.")
#--- SLOWER simulation that repeats the Monte-Carlo simulation even if (T, A) did NOT change and only the number of particles N changed


#raise KeyboardInterrupt




#--- FASTER simulation that uses the same (T, A) simulation for different N values
T_prev = 0
set_A_prev = sympy.Interval(0, 0)
for row, data in df_simulate.iterrows():
    if row < 30:
        continue
    if use_levy:
        # We estimate the probability of NEGATIVE intervals (-Inf, -c)
        set_C = sympy.Interval(-np.Inf, -data.C)
        set_A = sympy.Interval(-data.A, +np.Inf)
    else:
        # We estimate the probability of POSITIVE intervals (c, +Inf) because we simulate the REFLECTED process
        assert reflect, "Under the pure Gaussian noise, we must reflect the process at ZERO."
        set_C = sympy.Interval(data.C, +np.Inf)
        set_A = sympy.Interval(-data.A, data.A)
    T = int(data['T'])  #3*int(data['T'])
    N = int(data['N'])  #100

    # In case we need to skip the simulations that take longer (when e.g. we are testing the process)
    if row + 1 > +np.Inf: #7:
        print(f"Simulation {row+1} of {len(df_simulate)} (data={data.T}) skipped!")
    else:
        if T != T_prev or set_A != set_A_prev:
            # Initialize the output from the MC simulation that will be stored for each replication on the same T value
            arr_average_cycle_time = [None]*nrep
            arr_n_cycles = [None]*nrep
            arr_dist_exit_state = [None]*nrep
            arr_info_mc = [None]*nrep
        for rep in range(nrep):
            seed_rep = seed_base + rep*1317
            print(f"\nRunning case {row+1} of {len(df_simulate)}, replication {rep+1} of {nrep}: T={T}, N={N}, A={set_A}, C={set_C}, seed={seed_rep}...")

            # Output row (where the result is stored in the output data frames)
            orow = row*nrep + rep

            print(f"[FV] Fleming-Viot simulation (discretization interval, dt={dt})")

            time_start_fv = timer()
            cpu_start_fv = process_time()
            dict_params_simul = dict({'N': N,
                                      'T': T,
                                      'absorption_set': set_A,
                                      'check_for_stationarity': not reflect and not use_levy,
                                      })

            # Run MC simulation to estimate E(T_A) only ONCE for each T and A
            if arr_average_cycle_time[rep] is None:
                # We haven't yet run MC for the current T value
                arr_average_cycle_time[rep], arr_n_cycles[rep], arr_dist_exit_state[rep], arr_info_mc[rep] = \
                            sim._run_simulation_mc( dict_params_simul,
                                                    start_state=0.0,
                                                    store_trajectory=True,
                                                    check_for_stationarity=dict_params_simul['check_for_stationarity'],
                                                    burnin_for_stationarity_check=30,
                                                    seed=seed_rep, verbose=False, verbose_period=1, plot=False)
            else:
                print("*** Monte-Carlo simulation for FV SKIPPED because the results are already generated from a previous simulation with the same T and set_A values ***")

            if arr_dist_exit_state[rep] is None or len(arr_dist_exit_state[rep]) == 0:
                warnings.warn("No exit events from A were observed during the Monte-Carlo simulation. "
                              "The Fleming-Viot simulation will not be run and most corresponding output variables stored in the `info` output dictionary will be set to `None`.")
                df_proba_surv = None
                dict_phi = None
                integrals = None
                probas_stationary = dict()  # TODO: (2025/05/26) Estimate the stationarity probabilities from the MC excursion run above (call the estimate_probabilities_in_sets_of_interest() to this end, which should be moved to a new file called estimators/diffusion.py.
                info_fv = dict()
            else:
                # Run the FV simulation
                seed_fv = seed_rep + 131713
                print(f"\nRunning Fleming-Viot particle system simulation to estimate the stationary probabilities of the sets of interest (seed={seed_fv})...")
                print(f"Simulation parameters:\n{dict_params_simul}")
                print(f"Exit state distribution:\n{arr_dist_exit_state[rep][arr_dist_exit_state[rep]['p'] > 0]}")
                print(f"Sets of interest:\n{[set_C]}")
                df_proba_surv, dict_phi, info_fv = sim._run_simulation_fv(dict_params_simul, arr_dist_exit_state[rep], [set_C],
                                                                          store_trajectory=True,
                                                                          seed=seed_fv, verbose=False, verbose_period=1, plot=False)

                # Estimation of the stationary probabilities of each set of interest
                probas_stationary, integrals = estimate_stationary_probabilities(dict_phi, df_proba_surv, arr_average_cycle_time[rep], uniform_jump_rate=N)
            info = dict({'nsteps': arr_info_mc[rep]['last_time_observed'] + info_fv.get('last_time_observed', 0),
                         'MC': {**{'expected_cycle_time': arr_average_cycle_time[rep],
                                   'n_cycles': arr_n_cycles[rep],
                                   'dist_exit_state': arr_dist_exit_state[rep]},
                                **arr_info_mc[rep]},
                         'FV': {**{'df_proba_surv': df_proba_surv,
                                   'dict_phi': dict_phi,
                                   'integrals': integrals},
                                **info_fv},
                         })

            elapsed_time = timer() - time_start_fv
            cpu_time = process_time() - cpu_start_fv
            print(f"FV simulation took {elapsed_time/60:.1f} min (CPU={cpu_time/60:.1f}), {elapsed_time:.1f} sec")

            # Store and save results
            df_results_fv.iloc[orow]['case'] = row + 1
            df_results_fv.iloc[orow][df_simulate.columns] = data.apply(lambda x: "{:.3g}".format(x))
            df_results_fv.iloc[orow]['rep'] = rep + 1
            df_results_fv.iloc[orow]['seed'] = seed_rep
            df_results_fv.iloc[orow][['p(FV)', 'nsteps', 'nsteps_mc', 'nsteps_fv', 'expected_cycle_time', 'n_cycles', 'exec_time', 'cpu_time']] = \
                [probas_stationary[set_C] if len(probas_stationary) > 0 else np.nan,
                 info['nsteps'],
                 info['nsteps'] - info['FV'].get('last_time_observed', 0),
                 info['FV'].get('last_time_observed', 0),
                 info['MC']['expected_cycle_time'],
                 info['MC']['n_cycles'],
                 elapsed_time,
                 cpu_time]
            # Add result to output file
            if save:
                df_results_fv.iloc[orow:orow+1].to_csv(filename_fv, header=False, index=False, mode="a")

            print(f"[MC] Monte-Carlo simulation (case {row+1} of {len(df_simulate)}, replication {rep+1} of {nrep}, discretization interval, dt={dt})")
            time_start_mc = timer()
            cpu_start_mc = process_time()
            expected_cycle_time, n_cycles, dist_exit_state, info_mc = sim._run_simulation_mc(   dict({'T': info['nsteps'], 'absorption_set': set_A}),
                                                                                                start_state=0.0, store_trajectory=True,
                                                                                                check_for_stationarity=not reflect,
                                                                                                    ## NOTE that checking for stationarity in the Monte-Carlo simulation is just informative
                                                                                                    ## as the information about the exit states and expected cycle time is NOT used
                                                                                                    ## in the estimation of the probability of the state of interest, p(C),
                                                                                                    ## as the FULL trajectory is collected in the trajectory_mc attribute.
                                                                                                    ## (see the function estimate_probabilities_in_sets_of_interest() called just next)
                                                                                                burnin_for_stationarity_check=30,
                                                                                                seed=seed_rep, verbose=False, verbose_period=10, plot=False)
            probas_stationary_mc = estimate_probabilities_in_sets_of_interest(sim.trajectory_mc, [set_C])

            elapsed_time = timer() - time_start_mc
            cpu_time = process_time() - cpu_start_mc
            print(f"MC simulation took {elapsed_time / 60:.1f} min (CPU={cpu_time/60:.1f}), {elapsed_time:.1f} sec")

            # Store and save results
            df_results_mc.iloc[orow]['case'] = row + 1
            df_results_mc.iloc[orow][df_simulate.columns] = data.apply(lambda x: "{:.3g}".format(x))
            df_results_mc.iloc[orow]['rep'] = rep + 1
            df_results_mc.iloc[orow]['seed'] = seed_rep
            df_results_mc.iloc[orow][['p(MC)', 'nsteps', 'expected_cycle_time', 'n_cycles', 'exec_time', 'cpu_time']] = \
                [probas_stationary_mc[set_C], info_mc['last_time_observed'], expected_cycle_time, n_cycles, elapsed_time, cpu_time]
            # Add result to output file
            if save:
                df_results_mc.iloc[orow:orow+1].to_csv(filename_mc, header=False, index=False, mode="a")

    # Store the current T and set_A values so that we can compare them with the next ones and decide if we need to run the MC simulation again (not run if either T or set_A change)
    T_prev = T
    set_A_prev = set_A

    print()
elapsed_time = timer() - time_start
cpu_time = process_time() - cpu_start
print(f"SIMULATION PROCESS took {elapsed_time / 60:.1f} min (CPU: {cpu_time / 60:.1f} min), {elapsed_time:.1f} sec (CPU: {cpu_time / 60:.1f} sec)")
if save:
    print(f"Output files\n'{filename_fv}' (FV)\n'{filename_mc} (MC)\ncreated with the estimation results.")
#--- FASTER simulation that uses the same (T, A) simulation for different N values






#--------------- READ saved results for plotting
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator

from Python.lib.utils.plotting import set_ticklabels

rootdir = r"E:\Daniel\Projects\PhD-RL-Toulouse\projects\RL-004-Energy"
resultsdir = os.path.realpath(rootdir + "/results")
print(f"Results will be read from '{resultsdir}'")

datetime_str = "20250603_200000"; use_levy = False    # Gaussian noise case
datetime_str = "20250701_205600"; use_levy = True    # Levy noise case
reflect = not use_levy
dt = 0.05
groupvars = ['C', 'pC', 'A', 'pA', 'T', 'N'] if use_levy else ['C', 'pC', 'A', 'pA', 'T']

df_results_fv = pd.read_csv(os.path.join(resultsdir, f"ou_pwlinear_probability_estimation_{datetime_str}_levy={use_levy},dt={dt}_fv.csv"))
df_results_mc = pd.read_csv(os.path.join(resultsdir, f"ou_pwlinear_probability_estimation_{datetime_str}_levy={use_levy},dt={dt}_mc.csv"))

# Reconstruct the execution parameters of each experiment by grouping by the group variables (which precisely define the experiment characteristics!)
df_simulate = df_results_fv.groupby(groupvars).size().reset_index().rename({0: 'nrep'}, axis=1)
# Run this if pA incorrectly stores 1 - p(A) as opposed to p(A) (true in the initial experients run)
#df_simulate['pA'] = 1 - df_simulate['pA']


#-- Plot results: violins
# 1) When only ONE number of particles N has been considered
plot_for_presentation = True
common_axes = True
violin_width = 0.2
fontsize = 20
size_layout = int(np.ceil(np.sqrt(len(df_simulate))))
nrep = max(np.r_[df_results_fv['case'].value_counts(), df_results_mc['case'].value_counts()])

axes = plt.figure().subplots(size_layout, size_layout)
axes1d = axes.reshape(-1)
yvalues = np.r_[df_results_fv['p(FV)'], df_results_mc['p(MC)']]
ymin, ymax = np.nanmin(yvalues[yvalues > 0]), np.nanmax(yvalues)
position_mc = 1; position_fv = 2; ticklabels = ["MC", "FV"]
legend = ["Fleming-Viot", "Monte-Carlo"]
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
        violinparts_fv = ax.violinplot([toplot_fv], positions=[position_fv], widths=violin_width, showmedians=True, showmeans=False, showextrema=True)
        set_violincolor(violinparts_fv, "green")
        jitter = violin_width * np.random.uniform(-0.5, 0.5, sum(ind_fv))
        ax.plot(position_fv + jitter, toplot_fv, 'k.')
    if sum(ind_mc) > 0:
        toplot_mc = np.array(df_results_mc["p(MC)"][ind_mc]).astype(float)
        # Estimation errors
        # Note that we do NOT compute the absolute error because we want to assess whether there is a positive or negative bias
        errors_mc = {'mean': np.mean((toplot_mc - data.pC)),
                     'median': np.median((toplot_mc - data.pC)),
                     'mean_rel': np.mean((toplot_mc / data.pC - 1)),
                     'median_rel': np.median((toplot_mc / data.pC - 1))}
        violinparts_mc = ax.violinplot(toplot_mc, positions=[position_mc], widths=violin_width, showmedians=True, showmeans=False, showextrema=True)
        set_violincolor(violinparts_mc, "red")
        jitter = violin_width * np.random.uniform(-0.5, 0.5, sum(ind_mc))
        ax.plot(position_mc + jitter, toplot_mc, 'k.')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_ticks([1, 2])
    ax.xaxis.set_ticklabels(ticklabels)
    ax.tick_params(axis='both', labelsize=int(0.7*fontsize))
    ax.axhline(data.pC, color="gray", linestyle="dashed")
    ax.set_yscale('log')

    if plot_for_presentation:
        #title = f"T = {int(data['T'])}, A = ({-data.A:.3f}, {data.A:.3f})    " + r"$p(C) \sim 10^{" + str(int(np.round(np.log10(data.pC)))) + r"}$"
        title = f"T = {int(data['T'])},  " + r"$p(C) \sim 10^{" + str(int(np.round(np.log10(data.pC)))) + r"}$"
        ax.set_title(title, fontsize=fontsize)
    else:
        ax.set_title(rf"T = {int(data['T'])}, A = ({-data.A:.3f}, {data.A:.3f}), $p(\overline{{A}})$ = {0.5*(1 - data.pA):.1e}, C = ({data.C:.3f}, Inf), $p(C)$ = {data.pC:.1e}", fontsize=int(0.8*fontsize))

    if sum(ind_fv) > 0 and sum(ind_mc) > 0:
        if plot_for_presentation:
            ax.legend([violinparts_fv['bodies'][0], violinparts_mc['bodies'][0]], legend, fontsize=fontsize)
        else:
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







# 2) When several number of particles N have been considered
# Prepare the data so that every (C, A, T) combination represents a different case (as opposed to every (C, A, T, N) combination,
# because we want to generate several violin plots, one for each N in the same plot.
groupvars = ['C', 'pC', 'A', 'pA', 'T']
for df in [df_results_fv, df_results_mc]:
    df['case2plot'] = df.groupby(groupvars)['case'].transform('min')
df_simulate['case'] = df_simulate.index + 1
df_simulate['case2plot'] = df_simulate.groupby(groupvars)['case'].transform('min')
df_simulate2plot = df_simulate.groupby(groupvars + ['case2plot']).size().reset_index().rename({0: 'nviolins'}, axis=1)


# A) Violin plots
plot_for_presentation = True
common_axes = True
violin_width = 0.3
fontsize = 12
pointsize = 3
size_layout = int(np.ceil(np.sqrt(len(df_simulate.groupby(groupvars)))))

axes = plt.figure().subplots(size_layout, size_layout)
axes1d = axes.reshape(-1)
yvalues = np.r_[df_results_fv['p(FV)'], df_results_mc['p(MC)']]
ymin, ymax = np.nanmin(yvalues[yvalues > 0]), np.nanmax(yvalues)
for row, data in df_simulate2plot.iterrows():
    ax = axes1d[row]
    idx_case2plot = int(data.case2plot)
    ind_mc = df_results_mc['case2plot'] == idx_case2plot
    ind_fv = df_results_fv['case2plot'] == idx_case2plot
    nviolins = int(data.nviolins)
    print(f"idx_case2plot={idx_case2plot}: {sum(ind_fv)} FV measurements and {sum(ind_mc)} MC measurements on {int(nviolins)} violins to plot")

    # Pivot the data in order to generate a 2D matrix with a different row for each different N value
    # Note: Using pd.pivot_table() is rather complicated, so I do it manually
    toplot_mc, groupv_values_mc = prepare4violin(df_results_mc[ind_mc], 'N', 'p(MC)')
    toplot_fv, group_values_fv  = prepare4violin(df_results_fv[ind_fv], 'N', 'p(FV)')
    assert group_values_fv == groupv_values_mc
    group_values = group_values_fv

    positions_ref = np.arange(1, nviolins+1)
    positions_mc = positions_ref - 0.8*violin_width
    positions_fv = positions_ref + 0.8*violin_width
    violinparts_mc = ax.violinplot(toplot_mc, positions=positions_mc, widths=violin_width, showmedians=True, showmeans=False, showextrema=True)
    violinparts_fv = ax.violinplot(toplot_fv, positions=positions_fv, widths=violin_width, showmedians=True, showmeans=False, showextrema=True)
    set_violincolor(violinparts_mc, "red")
    set_violincolor(violinparts_fv, "green")

    for _data_violin, _positions_violin in zip([toplot_mc, toplot_fv], [positions_mc, positions_fv]):
        for i, data_points in enumerate(_data_violin):
            pos = _positions_violin[i]
            jitter = violin_width * np.random.uniform(-0.5, 0.5, len(data_points))
            ax.plot(pos + jitter, data_points, 'k.', markersize=pointsize)

    # Add separators between the violin groups
    for pos in positions_ref + 0.5:
        ax.axvline(pos, color="gray")

    # Make the X axis more informative and remove the ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    set_ticklabels(ax.xaxis, [f"N={g}" for g in group_values])
    ax.tick_params(axis='x', bottom=False)

    # Add reference lines
    # Reference (true) probability. This is the correct true probability ONLY in the Gaussian noise case.
    # For the Levy case we show it as half its value because the Levy case estimates the probability of a NEGATIVE excursion interval, NOT positive as well
    proba_ref = data.pC / 2 if use_levy else data.pC
    ax.axhline(proba_ref, color="gray", linestyle="dashed")

    # Vertical axis limits and scale
    if common_axes:
        ax.set_ylim((0.9*ymin, 1.1*ymax))
    ax.set_yscale('log')

    # Legend and title
    ax.legend([violinparts_mc['bodies'][0], violinparts_fv['bodies'][0]], ["MC", "FV"]) #, fontsize=int(0.8*fontsize))
    if plot_for_presentation:
        title = f"T = {int(data['T'])},   " + r"$P(C) \sim 10^{" + str(int(np.log10(data.pC / 2))) + r"}$"
        ax.set_title(title)
    else:
        if use_levy:
            ax.set_title(rf"T = {int(data['T'])}, C = (-Inf, {-data.C:.3f}), $p(C)$ = {data.pC/2:.1e}")
        elif reflect:
            ax.set_title(rf"T = {int(data['T'])}, C = ({data.C:.3f}, Inf), $p(C)$ = {data.pC:.1e}")
        else:
            ax.set_title(rf"T = {int(data['T'])}, C = (-Inf, {-data.C:.3f}) U ({data.C:.3f}, Inf), $p(C)$ = {data.pC:.1e}")

    # Add labels at the left and bottom borders
    if not plot_for_presentation:
        #if np.unravel_index(row, axes.shape)[0] == axes.shape[0] - 1:
        #    ax.set_xlabel(f"Method (up to {int(nrep)} replications)")
        if np.unravel_index(row, axes.shape)[1] == 0:
            ax.set_ylabel("Estimated probability p(C) (log)")
if not plot_for_presentation:
    plt.suptitle(f"Sensitivity analysis of FV vs. MC as a function of event probability\n(on up to {max(df_simulate['nrep'])} replications)", fontsize=int(1.2*fontsize))
plt.gcf().set_figheight(9)
plt.gcf().set_figwidth(16)
plt.get_current_fig_manager().window.showMaximized()
subplots_adjust = dict({'top': 0.925, 'bottom': 0.075, 'left': 0.05, 'right': 0.95, 'hspace': 0.4, 'wspace': 0.15})
plt.gcf().subplots_adjust(**subplots_adjust)


# Save the plot
# Note dpi=400, which is an important quality leap over dpi=300! (based on zooming in on the image in TeX Studio with the magnifier utility)
filename_fig = os.path.join(resultsdir, f"ou_pwlinear_probability_estimation_levy={use_levy},dt={dt}.png")
plt.savefig(
    filename_fig,
    dpi=400,
    bbox_inches="tight",
    transparent=False,
)
print(f"Figure saved to {filename_fig}")




# B) Plot of the convergence properties with N
# Goal: Plot robust estimates of the standard error and the mean in order to evaluate the variability and the bias in a more quantitative way than visually on the violin plots
df_analyze_mc = compute_statistics(df_results_mc, groupvars + ['case2plot', 'N'], 'p(MC)')
df_analyze_fv = compute_statistics(df_results_fv, groupvars + ['case2plot', 'N'], 'p(FV)')

common_axes = True
log_scale = True
fontsize = 20

axes_estimate = plt.figure().subplots(size_layout, size_layout)
axes_error = plt.figure().subplots(size_layout, size_layout)
axes_relerror = plt.figure().subplots(size_layout, size_layout)
axes_estimate_1d = axes_estimate.reshape(-1)
axes_error_1d = axes_error.reshape(-1)
axes_relerror_1d = axes_relerror.reshape(-1)
yvalues_estimate = np.r_[df_analyze_fv['median'], df_analyze_mc['median']]
yvalues_error = np.r_[df_analyze_fv['se_robust'], df_analyze_mc['se_robust']]
yvalues_relerror = np.r_[df_analyze_fv['se_robust'] / df_analyze_fv['median'], df_analyze_mc['se_robust'] / df_analyze_mc['median']]
ymin_estimate, ymax_estimate = min(yvalues_estimate[yvalues_estimate > 0]), max(yvalues_estimate[yvalues_estimate > 0])
ymin_error, ymax_error = min(yvalues_error[yvalues_error > 0]), max(yvalues_error[yvalues_error > 0])
ymin_relerror, ymax_relerror = min(yvalues_relerror[(np.isfinite(yvalues_relerror)) & (yvalues_relerror > 0)]), max(yvalues_relerror[(np.isfinite(yvalues_relerror)) & (yvalues_relerror > 0)])
for row, data in df_simulate2plot.iterrows():
    ax_estimate = axes_estimate_1d[row]
    ax_error = axes_error_1d[row]
    ax_relerror = axes_relerror_1d[row]

    idx_case2plot = int(data.case2plot)
    ind_mc = df_analyze_mc['case2plot'] == idx_case2plot
    ind_fv = df_analyze_fv['case2plot'] == idx_case2plot

    N_values_mc = df_analyze_mc['N'][ind_mc]
    N_values_fv = df_analyze_fv['N'][ind_fv]

    toplot_estimate_mc = df_analyze_mc['median'][ind_mc]
    toplot_estimate_fv = df_analyze_fv['median'][ind_mc]

    toplot_error_mc = df_analyze_mc['se_robust'][ind_mc]
    toplot_error_fv = df_analyze_fv['se_robust'][ind_mc]

    toplot_relerror_mc = df_analyze_mc['se_robust'][ind_mc] / toplot_estimate_mc
    toplot_relerror_fv = df_analyze_fv['se_robust'][ind_mc] / toplot_estimate_fv

    ax_estimate.plot(N_values_mc, toplot_estimate_mc, 'r.-', linestyle="solid")
    ax_estimate.plot(N_values_fv, toplot_estimate_fv, 'g.-', linestyle="solid")

    ax_error.plot(N_values_mc, toplot_error_mc, 'r.-', linestyle="dashed")
    ax_error.plot(N_values_fv, toplot_error_fv, 'g.-', linestyle="dashed")

    ind_ok_mc = np.isfinite(toplot_relerror_mc)
    ind_ok_fv = np.isfinite(toplot_relerror_fv)
    ax_relerror.plot(N_values_mc[ind_ok_mc], toplot_relerror_mc[ind_ok_mc], 'r.-', linestyle="solid")
    ax_relerror.plot(N_values_fv[ind_ok_fv], toplot_relerror_fv[ind_ok_fv], 'g.-', linestyle="solid")

    if row == 0:
        # Get the X axis limits from the first plot which is supposed to be the one where ALL estimates are plotted
        # so that we can copy the same X axis limits to the other subplots where some N values may not show up because the corresponding plotted value is NaN
        # (e.g. the relative error plots, where NaN is generated when dividing by a 0 estimate of the probability)
        xlim_first = ax_estimate.get_xlim()

    if common_axes:
        ax_estimate.set_ylim((0.9*ymin_estimate, 1.1*ymax_estimate))
        ax_error.set_ylim((0.9*ymin_error, 1.1*ymax_error))
        ax_relerror.set_ylim((0.9*ymin_relerror, 1.1*ymax_relerror))

    if log_scale:
        for _ax in (ax_estimate, ax_error, ax_relerror):
            _ax.set_yscale('log')
    else:
        for _ax in (ax_estimate, ax_error, ax_relerror):
            _ax.set_ylim((0, None))

    # Set the X axis limits in case some of the N values do not have any reported values (i.e. all NaN) in which case the X axis may not have the same limits for all subplots
    for _ax in (ax_estimate, ax_error, ax_relerror):
        _ax.set_xlim(xlim_first)

    ax_relerror.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    xlabel = "N: Number of FV particles"
    legend = ["Monte-Carlo", "Fleming-Viot"]
    title = f"T = {int(data['T'])},   " + r"$P(C) \sim 10^{" + str(int(np.log10(data.pC / 2))) + r"}$"
    for _ax in (ax_estimate, ax_error, ax_relerror):
        _ax.set_xlabel(xlabel, fontsize=int(0.8*fontsize))
        _ax.legend(legend, fontsize=int(0.8*fontsize))
        _ax.set_title(title, fontsize=fontsize)

#tight_layout = dict({'h_pad': 0.010, 'w_pad': 0.005, 'rect': (0.050, 0.050, 0.950, 0.950)})
subplots_adjust = dict({'top': 0.925, 'bottom': 0.075, 'left': 0.05, 'right': 0.95, 'hspace': 0.6, 'wspace': 0.15})
for _ax in (ax_estimate, ax_error, ax_relerror):
    #_ax.get_figure().tight_layout(**tight_layout)      # This does NOT work and the values do NOT correspond to those shown in the window
    _ax.get_figure().subplots_adjust(**subplots_adjust)


