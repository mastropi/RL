# -*- coding: utf-8 -*-
"""
Created on Sun May 11 14:52:45 2025

@author: Daniel Mastropietro
@description: Unit tests for functions and methods defined in simulators.diffusion.
@details: Naming conventions follow the instructions given in test_conventions.txt.
@ref: "Dynamic dimensioning of frequency containment reserves", J. Janssen, A. Zocca, B. Zwart (2024), https://arxiv.org/abs/2411.11093
"""

import runpy
runpy.run_path('../../setup.py')

import unittest
from timeit import default_timer as timer
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import sympy
import seaborn as sns

from Python.lib.estimators.fv import estimate_stationary_probabilities
from Python.lib.environments.diffusion import EnvDiffusion, func_drift_fcr_piecewise, func_noise_gaussian
from Python.lib.estimators.diffusion import calculate_probabilities, calculate_probability, estimate_probabilities_in_sets_of_interest
from Python.lib.simulators.diffusion import SimulatorDiffusionFV


def define_sets_of_interest(thresholds):
    "Define sets of interest for the estimation of stationary probabilities as intervals having the given thresholds as bounds"
    sets_of_interest = list()
    for i in range(len(thresholds) - 1):
        # When considering both sides of the process deviation in the same set (e.g. (-0.5, -0.1) U (0.1, 0.5))
        # sets_of_interest += [sympy.Union( sympy.Interval(-thresholds[i+1], -thresholds[i]), sympy.Interval(thresholds[i], thresholds[i+1]) )]
        # When considering each side of the process deviation as separate sets (e.g. (-0.5, -0.1) separately from (0.1, 0.5))
        sets_of_interest += [sympy.Union(sympy.Interval(thresholds[i], thresholds[i + 1]))]

    return sets_of_interest


class Test_Class_SimulatorDiffusionFV(unittest.TestCase):
    "Tests for the SimulatorDiffusionFV simulator that simulates the EnvDiffusion environment"

    @classmethod
    def setUpClass(cls):
        # The environment is a diffusion process modeling the average frequency variability in an electricity grid
        # OU = Ornstein-Uhlenbeck
        # Parameters of the process
        mu = 0.0  # Long-term mean
        sigma = 0.04  # Volatility
        # Time discretization of the process
        dt = 0.01
        # Parameters for the piecewise drift function (alpha in the paper referenced in func_drift_fcr_piecewise())
        cls.r = 0.6  # Capacity reserved for FCR-N (Fixed by the TSO = Transmission System Operator of the Nordic grid, see section IV in reference above)
        cls.x = 1.4  # Capacity reserved for FCR-D (Fixed by A. Zocca et al. strategy, see section IV in reference above)
        cls.env_ou = EnvDiffusion(func_drift=func_drift_fcr_piecewise, func_noise=func_noise_gaussian, reflect=False, mu=mu, dt=dt, sigma=sigma, r=cls.r, x=cls.x)

    def test_run_estimation_fv(self, dict_params_simul=dict({'N': 50, 'T': 1000, 'absorption_set': sympy.Interval(-0.08, 0.08)}), sets_of_interest=None, store_trajectories=False, seed=1313, debug=False, plot={'MC': False, 'FV': False}):
        """ (2025/05/12)
        Tests the estimation of stationary probability of sets of interest by Fleming-Viot, by running the SimulatorDiffusionFV.run() method
        """
        print("\n*** Running test " + self.id() + " ***")

        if sets_of_interest is None:
            # Define the sets of interest as intervals having the given thresholds as bounds
            sets_of_interest = define_sets_of_interest(thresholds=[-np.Inf, -0.5, -0.1, 0.0, 0.1, 0.5, +np.Inf])

        # Expected results: probabilities of each smooth piece of the drift function
        calculate_probabilities(self.r, self.x, self.env_ou.sigma, verbose=True)

        # Experimental results
        sim = SimulatorDiffusionFV(self.env_ou, None, debug=debug)
        print(f"Running estimation by Fleming-Viot of the stationary probabilities of sets of interest of a diffusion process discretized at dt={self.env_ou.dt} intervals, with the SimulatorDiffusionFV.run() method...")

        time_start = timer()
        probas_stationary, info = sim.run(dict_params_simul, sets_of_interest,
                                          store_trajectories=store_trajectories,
                                          seed=seed, verbose=True, verbose_period=10, plot=plot)

        print(f"Estimated expected cycle time, E(T_A): {info['MC']['expected_cycle_time']} (n={info['MC']['n_cycles']}) on T = {dict_params_simul['T']} steps")
        print(f"Estimated stationary probabilities with N={dict_params_simul['N']} particles:\n{probas_stationary}")
        print("_run_simulation_mc() took {:.4f} seconds".format(timer() - time_start))

        # Execution parameters
        assert dict_params_simul['T'] == 1000
        assert dict_params_simul['absorption_set'] == sympy.Interval(-0.08, 0.08)
        assert dict_params_simul['N'] == 50
        # Results
        assert info['MC']['n_cycles'] == 13
        assert np.isclose(info['MC']['expected_cycle_time'], 25.461538)
        assert probas_stationary == dict({sympy.Interval(-np.Inf, -0.5): 0.0,
                                          sympy.Interval(-0.5, -0.1): 0.1282228882175227,
                                          sympy.Interval(-0.1, 0.0): 0.376381341389728,
                                          sympy.Interval(0.0, 0.1): 0.0,
                                          sympy.Interval(0.1, 0.5): 0.0,
                                          sympy.Interval(0.5, +np.Inf): 0.0})
            ## With this simulation setup, all exit states happen to be on the negative side of the state space

        return probas_stationary, info

    def test_run_simulation_mc(self, nsteps=1000, absorption_set=sympy.Interval(-0.08, 0.08), start_state=0.0, check_for_stationarity=True, store_trajectories=False, seed=1313, debug=False, plot=False):
        """ (2025/05/11)
        Tests the Monte-Carlo simulation run by the  SimulatorDiffusionFV._run_simulation_mc() method
        """
        print("\n*** Running test " + self.id() + " ***")

        # Simulation parameters
        dict_params_simul = dict({'T': nsteps,                          # Number of integer steps for the MC simulator that estimates the expected cycle time to A, E(T_A)
                                  'absorption_set': absorption_set,     # Absorption set defined as sympy.Set. If an Interval, the interval ends are included by default.
                                                                        # Use `left_open=True` and `right_open=True` to respectively open them.
                                  })

        # Expected results: probabilities of each smooth piece of the drift function
        calculate_probabilities(self.r, self.x, self.env_ou.sigma, verbose=True)

        # Experimental results
        sim = SimulatorDiffusionFV(self.env_ou, None, debug=debug)
        print(f"Running Monte-Carlo simulation of diffusion process for as long as {dict_params_simul['T']} steps, discretized at dt={self.env_ou.dt} intervals, with _run_simulation_mc()...")
        time_start = timer()
        average_cycle_time, n_cycles, dist_exit_state, info = sim._run_simulation_mc(dict_params_simul, start_state=start_state, store_trajectory=store_trajectories,
                                                                                                        check_for_stationarity=check_for_stationarity, burnin_for_stationarity_check=30,
                                                                                                        seed=seed, verbose=True, verbose_period=10, plot=plot)
        print(f"Estimated EXIT state distribution from A (n={n_cycles} cycles):\n{dist_exit_state}\n(len = {len(dist_exit_state)} of which {len(dist_exit_state[dist_exit_state['p'] > 0])} have positive probability)")
        print(f"Estimated expected cycle time, E(T_A): {average_cycle_time} (n={n_cycles})")
        print("_run_simulation_mc() took {:.4f} seconds".format(timer() - time_start))

        # Execution parameters
        assert dict_params_simul['T'] == 1000
        assert dict_params_simul['absorption_set'] == sympy.Interval(-0.08, 0.08)
        assert start_state == 0.0
        # Results
        assert n_cycles == 13
        assert np.isclose(average_cycle_time, 25.461538)
        assert len(dist_exit_state) == 30
        assert len(dist_exit_state[dist_exit_state['p'] > 0]) == 11, \
            "The number of exit states with positive probability might be less than the number of exit states in `dist_exit_state` because they are computed using a histogram"
        assert all(dist_exit_state['x'] < 0.0), "With this simulation setup, all the exit states happen to be negative"
        assert all(dist_exit_state['p'] <= 1.0)

        return sim, average_cycle_time, n_cycles, dist_exit_state, info

    def test_run_simulation_fv(self, N=50, dist_exit_state=pd.DataFrame({'x': [-0.09, 0.09], 'p': [0.5, 0.5]}), absorption_set=sympy.Interval(-0.08, 0.08),
                                     store_trajectories=True, seed=1313, debug=False, plot=False):
        """ (2025/05/11)
        Tests the Fleming-Viot simulation run by the  SimulatorDiffusionFV._run_simulation_fv() method
        """
        print("\n*** Running test " + self.id() + " ***")

        # Simulation parameters
        dict_params_simul = dict({'N': N,
                                  'max_nsteps': 1E7,                    # Maximum number of steps for which the FV simulator should be run (in case the stopping criterion fails)
                                  'absorption_set': absorption_set,     # Absorption set defined as sympy.Set. If an Interval, the interval ends are included by default.
                                                                        # Use `left_open=True` and `right_open=True` to respectively open them.
                                  })

        # Expected results: probabilities of the intervals of interest
        # Recall that these intervals should be OUTSIDE the absorption set A!
        thresholds = [-np.Inf, -0.5, -0.1, 0.0, 0.1, 0.5, +np.Inf]
        sets_of_interest = list()
        for i in range(len(thresholds)-1):
            # When considering both sides of the process deviation in the same set (e.g. (-0.5, -0.1) U (0.1, 0.5))
            #sets_of_interest += [sympy.Union( sympy.Interval(-thresholds[i+1], -thresholds[i]), sympy.Interval(thresholds[i], thresholds[i+1]) )]
            # When considering each side of the process deviation as separate sets (e.g. (-0.5, -0.1) separately from (0.1, 0.5))
            sets_of_interest += [sympy.Union(sympy.Interval(thresholds[i], thresholds[i + 1]))]
        calculate_probabilities(self.r, self.x, self.env_ou.sigma, verbose=True)

        # Experimental results
        sim = SimulatorDiffusionFV(self.env_ou, None, debug=debug)
        print(f"Running the FV simulation of diffusion process (MC part already run) discretized at dt={self.env_ou.dt} intervals using _run_simulation_fv()...")
        print(f"Number of particles: N = {N}")
        print(f"Absorption set: {dict_params_simul['absorption_set']}")
        print(f"Start states chosen according to distribution:\n{dist_exit_state}")
        time_start = timer()
        df_proba_surv, dict_phi, info = \
            sim._run_simulation_fv(dict_params_simul, dist_exit_state, sets_of_interest, store_trajectory=store_trajectories,
                                                                                         seed=seed, verbose=True, verbose_period=10, plot=plot)
        print("_run_simulation_fv() took {:.4f} seconds".format(timer() - time_start))

        # Execution parameters
        assert dict_params_simul['max_nsteps'] == 1E7
        assert dict_params_simul['absorption_set'] == sympy.Interval(-0.08, 0.08)
        assert dict_params_simul['N'] == 50
        # Results
        assert len(df_proba_surv) == N + 1, \
            f"The number of rows in `df_proba_surv` must be one more than the number of particles because the FV simulation stops when all N particles have been absorbed at least once, as long as it is smaller than {dict_params_simul['max_nsteps']}"
        assert len(dict_phi) == len(sets_of_interest)
        assert len(info['absorption_times']) == 218
        assert info['last_time_observed'] == 6917

        return sim, df_proba_surv, dict_phi, info


if __name__ == "__main__":
    test = False
    tests2run = ["FV", "MC"]    #["FV0", "MC"]

    if not test:
        from matplotlib import pyplot as plt, cm
        from matplotlib.ticker import MaxNLocator

        # Number of replications to run
        nrep = 1 #21
        seed_base = 1313  # 1317

        # Maximum number of steps to run the FV simulation for (over all particles)
        max_nsteps_fv = int(1E4) #int(1E5)

        # Time resolution
        # NOTE: dt is measured in SECONDS because the frequency is measured in Hz (as this is the unit used when defining the piecewise linear drift alpha(F(t)))
        # We define dt at the BEGINNING because the total number of steps to simulate the process for (T*N) should depend on dt, because normally what we would like is to simulate
        # the process for a fixed number of seconds.
        dt = 0.05 #0.005 #0.1 #0.05 #0.01

        # Sets of interest for the estimation of stationary probabilities
        # Note: the `sets_case` variable is used below to:
        # - define simulation parameters (e.g. absorption set A, type of noise)
        # - define the sets2analyze when generating the violin plots that compare FV vs. MC

        # Case 1: Gaussian noise; Sets to analyze coincide with the smooth pieces of the alpha drift function
        sets_case = 1
        sets_of_interest = define_sets_of_interest(thresholds=[-np.Inf, -0.5, -0.1, 0.0, 0.1, 0.5, +np.Inf]); T = 1000; N = 10; #T = int(1E5); N = 50

        # Case 2: Gaussian noise; Sets to analyze do NOT coincide with the smooth pieces of the alpha drift function
        #sets_case = 2
        #sets_of_interest = define_sets_of_interest(thresholds=[-1.0, -0.5, -0.3, -0.2, 0.0, 0.2, 0.3, 0.5, +1.0]); T = int(1E5); N = 50
        #sets_of_interest = define_sets_of_interest(thresholds=[-1.0, -0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, +1.0]); T = int(1E5); N = 50 --> This works quite nicely to show the advantage of FV but still MC can estimate something
        #sets_of_interest = define_sets_of_interest(thresholds=[-1.0, -0.5, -0.3, -0.16, 0.0, 0.16, 0.3, 0.5, +1.0]); T = int(1E5); N = 50

        # Case 3: Gaussian noise; The simulation is run on the REFLECTED process (to avoid problems with disjoint activation sets, where particles could get stuck)
        #sets_case = 3
        #sets_of_interest = define_sets_of_interest(thresholds=[0.0, 0.16, 0.3, 0.5, +1.0]); A_boundaries = (-0.12, 0.12); T = 1000; N = 50; #T = int(1E5); N = 50
        #sets_of_interest = define_sets_of_interest(thresholds=[0.0, 0.18, 0.3, 0.5, +1.0]); A_boundaries = (-0.14, 0.14); T = int(1E6); N = 80
        #sets_of_interest = define_sets_of_interest(thresholds=[0.0, 0.20, 0.3, 0.5, +1.0]); A_boundaries = (-0.15, 0.15); T = int(1E6); N = 100

        # Case 4: Lévy noise (i.e. Gaussian noise with (negative) jumps)
        #sets_case = 4
        #sets_of_interest = define_sets_of_interest(thresholds=[-np.Inf, -1.0, 0.0]); T = 10000; N = 100             # When using an artificially generated jump process
        #sets_of_interest = define_sets_of_interest(thresholds=[-np.Inf, -0.15, -0.10, 0.0]); T = 1000; N = 100
        #sets_of_interest = define_sets_of_interest(thresholds=[-np.Inf, -0.17, -0.12, 0.0]); T = 100000; N = 100

        # For testing purposes of the OU process and the frequency distribution obtained by it:
        # SIMULATE JUST MONTE-CARLO FOR 1 HOUR, which is the time interval on which the parameters r and x are defined,
        # e.g. "r = 0.6 GW is the FCR-N volume procured for every hour of the year" (AZ 2024 paper, pag. 5)
        #T = int(3600/dt); N = 1
        #print(f"The Monte-Carlo simulation will be run for {T*N} number of steps to fulfill 1 hour of simulation at dt={dt} seconds.")

        # Absorption set A, if it should be defined now (for some cases)
        if sets_case <= 2:
            A_boundaries = (-0.08, +0.08)
            #A_boundaries = (-0.12, 0.12)
        elif sets_case == 4:
            # Levy process
            #A_boundaries = (-0.8, +np.Inf)    # -4 is a good value when the lambda of the jump process is 0.25/dt with dt = 0.05, i.e. lambda = 5 (based on process trajectories)
            A_boundaries = (-0.10, +np.Inf)  # -4 is a good value when the lambda of the jump process is 0.25/dt with dt = 0.05, i.e. lambda = 5 (based on process trajectories)

        # Whether the process is reflected at 0
        reflect = sets_case == 3

        # Create the test object
        test_obj = Test_Class_SimulatorDiffusionFV()
        # Need to define selected process parameters and the environment ad-hoc because they are not found when defined via setUpClass() in this ad-hoc execution
        mu = 0.0
        sigma = 0.04
        test_obj.r = 0.6 # Capacity reserved for FCR-N (0.6 GW/h by the Nordic TSO for deviations in (-0.1, 0.1) Hz)
        test_obj.x = 1.4 # Capacity reserved for FCR-D (1.4 GW/h by the Nordic TSO, for deviations in (-0.5, -0.1) Hz U (0.1, 0.5) Hz)

        if sets_case == 4:
            # Jumps: Note that we define the intensity of the jumps, lambda, as the desired Poisson parameter `lambda*dt` that gives a reasonable number of jumps in the dt interval
            # (e.g. 0.5 tends to give values between 0 and 2, recalling that the expected number of a Poisson variable is the value of the Poisson parameter, `lambda*dt` in this case)
            #levy = {'lambda': 0.25/dt, 'fun_dist_jumps': np.random.beta, 'params_dist_jumps': {'factor': -1/10, 'a': 20, 'b': 1}} if sets_case == 4 else None
            # Levy process based on Fingrid data: lambda is the jump rate (i.e. per unit time)
            # The value 0.0175 comes from the 2025-01 measurements of the frequency (see for now scratch-vu.py, then it will be moved to a proper script)
            # NOTES about the lognormal distribution:
            # - Since we fitted it using scipy.stats, we need to use stats.lognorm.rvs() to generate random values, instead of np.random.lognormal() because the latter only accepts two parameters
            # - If we used np.random.lognormal(), we must keep in mind that "shape (scipy.stats) <-> sigma", "log(scale) (scipy.stats) <-> mean"
            # Ref: https://stackoverflow.com/questions/8747761/scipy-lognormal-distribution-parameters
            #levy = {'lambda': 0.0175, 'fun_dist_jumps': stats.lognorm.rvs, 'params_dist_jumps': {'factor': -1, 'shape': 0.677, 'loc': 0.0131, 'scale': 0.00344}} if sets_case == 4 else None
            # Jump rate estimated using the iterative procedure to estimate the jump probability in each dt = 0.1 sec interval from the Fingrid data for 2025-01
            # and using thr = -0.016187 as threshold for "pure" jumps.
            # NOTE: We multiply the estimated jump rate by 20 in order to make the skewness of the frequency distribution larger than with the estimated jump rate (of 0.0175, rather small...)
            jump_rate = 0.0175 * 20
            # Shape, location and scale estimated using scipy.stats.lognorm.fit(x) where x is the 0.1-second frequency data from Fingrid for 2025-01 (31 days, ~ 26 Mio. records)
            shape, loc, scale = (0.6766070581549546, 0.013074261175727589, 0.00344392772573485)
            levy = {'lambda': jump_rate, 'fun_dist_jumps': np.random.lognormal, 'params_dist_jumps': {'factor': -1, 'shift': loc, 'mean': np.log(scale), 'sigma': shape}}
        else:
            levy = None
        test_obj.env_ou = EnvDiffusion(func_drift=func_drift_fcr_piecewise, func_noise=func_noise_gaussian, reflect=reflect, levy=levy, mu=mu, dt=dt, sigma=sigma, r=test_obj.r, x=test_obj.x)

        # True stationary probability of the smooth pieces of the drift function alpha(x)
        # NOTE: In order to compute the true probability of the sets of interest, use the calculate_probability(interval) function where one can specify an interval of interest
        # This is done below, when analyzing the sets of interest.
        probas_true = calculate_probabilities(test_obj.r, test_obj.r, sigma)

        # Output variables to store the results for each replication
        probs_fv = [None]*nrep
        probs_mc = [None]*nrep

        time_start = timer()

        if "FV0" in tests2run:
            # Run SINGLE FV simulation in order to test _run_simulation_mc() and _run_simulation_fv() SEPARATELY
            A_boundaries = (-0.08, +0.08)
            absorption_set = sympy.Interval(*A_boundaries)
            seed = seed_base

            # Estimate the exit state distribution from A, so that we know how to choose the start state of the FV particles
            nsteps_et = 5000 #50000 #10000
            start_state = 0.0 #A_boundaries[1] #0.0
            sim_mc, average_cycle_time, n_cycles, dist_exit_state, info_mc = \
                test_obj.test_run_simulation_mc(nsteps=nsteps_et, absorption_set=absorption_set, start_state=start_state, store_trajectories=True, seed=seed, debug=False, plot=False)

            ax_mc, ax_longrun, ax_fv = plt.figure().subplots(1, 3)

            #--- Plot the MC trajectory and long-run statistics (mu and sigma)
            ax_mc.plot(sim_mc.trajectory_mc, color="red")
            ax_mc.plot(info_mc['exit_times'], info_mc['exit_states'], "ko", markersize=3)
            ax_mc.axhline(A_boundaries[0], color="blue", linestyle="dashed")
            ax_mc.axhline(A_boundaries[1], color="blue", linestyle="dashed")
            ax_mc.axhline(0.0, color="gray")
            ax_mc.set_xlabel("t")
            ax_mc.set_title(f"MC simulation on {nsteps_et} steps")

            # Long-run estimations
            cumN = np.arange(1, len(sim_mc.trajectory_mc) + 1)
            mu_est = np.cumsum(sim_mc.trajectory_mc) / cumN
            mu_std = np.sqrt( np.cumsum(sim_mc.trajectory_mc**2) / cumN - mu_est**2 )
            ax_longrun.plot(mu_est, color="blue")
            ax_longrun.plot(mu + mu_std, color="red")
            ax_longrun.axhline(mu, color="blue", linestyle="dashed")
            ax_longrun.axhline(mu + sigma, color="red", linestyle="dashed")
            ax_longrun.axhline(mu - sigma, color="red", linestyle="dashed")
            ax_longrun.set_xlabel("t: discrete time step")
            ax_longrun.set_ylabel("X(t)")
            ax_longrun.legend(["Running average", "Running standard deviation", "mu: true mean", "sigma: true standard deviation"])
            #ax_longrun.set_title("Convergence to stationary is rather slow... (infinite state space)\n$\mu={}, \sigma={}, r={}, x={}$ (seed={})".format(mu, sigma, test_obj.r, test_obj.x, seed))
            ax_longrun.set_title("Convergence of the long-run estimates of mu and sigma\n$\mu={}, \sigma={}, r={}, x={}$ (seed={})".format(mu, sigma, test_obj.r, test_obj.x, seed))
            print(f"The long-run average value of the process at the end is mu = {mu_est[-1]} vs. its nominal value of mu = {mu}")
            #--- Plot the MC trajectory and long-run statistics (mu and sigma)

            # Run the FV simulation now
            N = 50
            sim_fv, df_proba_surv, dict_phi, info_fv = \
                test_obj.test_run_simulation_fv(N, dist_exit_state, absorption_set=absorption_set, store_trajectories=True, seed=seed+13, debug=False, plot=False)

            # Estimate the stationary probabilities of the sets of interest
            probas_stationary, integrals = estimate_stationary_probabilities(dict_phi, df_proba_surv, average_cycle_time, uniform_jump_rate=N)

            print("Theoretical probabilities:")
            calculate_probabilities(test_obj.r, test_obj.x, sigma)

            print(f"Estimated probabilites (N={N}, # steps={info_fv['last_time_observed']}):")
            for C, p in probas_stationary.items():
                print(f"{C.boundary}: {p}")

            #--- Plot the FV trajectories
            # FV
            colormap = cm.get_cmap("jet")
            for p in range(N):
                ax_fv.step(sim_fv.trajectories_fv[p].index, sim_fv.trajectories_fv[p], where='post', color=colormap(p/N))
            ax_fv.plot(info_fv['absorption_times'], info_fv['absorption_states'], "ro", markersize=3)
            ax_fv.axhline(A_boundaries[0], color="blue", linestyle="dashed")
            ax_fv.axhline(A_boundaries[1], color="blue", linestyle="dashed")
            ax_fv.axhline(0.0, color="gray")
            ax_fv.set_xlabel("t")
            ax_fv.set_title(f"FV simulation on N={N} particles")

            # Plot the exit state distribution
            plt.figure()
            plt.plot(dist_exit_state['x'], dist_exit_state['p'], color="blue")
            plt.axhline(0, color="gray")
            plt.axvline(A_boundaries[0], color="black", linestyle="dashed")
            plt.axvline(A_boundaries[1], color="black", linestyle="dashed")
            plt.title(f"Distribution of exit states based on histogram (nsteps={nsteps_et})")
            #--- Plot the FV trajectories

        if "FV" in tests2run:
            # Run the WHOLE FV estimation process, by calling the run() method of the SimulatorDiffusionFV class
            absorption_set = sympy.Interval(*A_boundaries)

            # Simulation parameters
            dict_params_simul = dict({'N': N, #50,  # 100
                                      'T': T, #int(1E5), #50000,  # 10000
                                      'absorption_set': absorption_set,
                                      'check_for_stationarity': False, #True,
                                      'burnin_for_stationarity_check': 30,
                                      'max_nsteps': max_nsteps_fv,        # Maximum number of steps for the FV simulation in case particles are not absorbed into A
                                      })

            for r in range(nrep):
                seed = seed_base + r
                print(f"\n====== {datetime.now()} - Running replication FV #{r+1} of {nrep} (seed={seed})...")
                probas_stationary, info = test_obj.test_run_estimation_fv(dict_params_simul, sets_of_interest=sets_of_interest,
                                                                          store_trajectories=True, seed=seed, debug=False, plot={'MC': False, 'FV': True})
                probs_fv[r] = dict({'probabilities': probas_stationary,
                                    'info': info
                                    })

                print(f"\nProbabilities estimated by FV for replication {r+1} of {nrep}:\n(N={dict_params_simul['N']}, T={dict_params_simul['T']}, A={dict_params_simul['absorption_set']}, # steps={info['nsteps']}):")
                for C, p in probas_stationary.items():
                    print(f"({C.inf, C.sup}): {p}")
                print(f"\nTrue probabilities: {probas_true}")

        if "MC" in tests2run:
            # Run the Monte-Carlo estimation of probabilities, using the  SimulatorDiffusionFV._run_simulation_mc() method
            # The number of steps used is the same as the number of events observed during the FV simulation above, when this was run (for comparison purposes)
            # Otherwise the number of steps is set to T*N.
            if {"FV", "FV0"}.intersection(tests2run) == set():
                # No FV simulation was run so the number of steps to run MC can be chosen freely
                nsteps = T*N
            elif "FV0" in tests2run:
                # Choose the number of steps to run MC for from the (assumed) SINGLE FV simulation run above
                assert nrep == 1, f"The number of replications to run for the Monte-Carlo simulation when 'FV0' has been run must be 1: {nrep}"
                nsteps = nsteps_et + info_fv['last_time_observed']
                print(f"The number of steps to run the Monte-Carlo simulation for is taken from the FV0 simulation: #steps={nsteps}")

            start_state = 0.0
            for r in range(nrep):
                seed = seed_base + r
                print(f"\n====== {datetime.now()} - Running replication MC #{r + 1} of {nrep} (seed={seed})...")
                if "FV" in tests2run:
                    nsteps = probs_fv[r]['info']['nsteps']
                    print(f"The Monte-Carlo simulation will run for as many steps as the corresponding Fleming-Viot simulation: #steps={nsteps}")
                # The Monte-Carlo simulation is run by using the MC simulator of the SimulatorDiffusionFV with absorption set A set to the singleton {0}
                sim, _, _, _, info_mc = \
                    test_obj.test_run_simulation_mc(nsteps=nsteps, absorption_set=sympy.FiniteSet(0), start_state=start_state, check_for_stationarity=False, store_trajectories=True, seed=seed, debug=False, plot=False)
                probs_mc[r] = dict({'probabilities': estimate_probabilities_in_sets_of_interest(sim.trajectory_mc, sets_of_interest),
                                    'info': info_mc
                                    })
                print(f"Probabilities estimated by MC for replication {r+1} of {nrep} (# steps = {info_mc['last_time_observed']}):")
                for C, p in probs_mc[r]['probabilities'].items():
                    print(f"({C.inf, C.sup}): {p}")
                print(f"\nTrue probabilities: {probas_true}")

            # Trajectory
            _npoints_to_plot = int(1E6)
            ax_trajectory = plt.figure().subplots(1, 1)
            line_process = ax_trajectory.plot(np.arange(min(len(sim.trajectory_mc), _npoints_to_plot))*dt, sim.trajectory_mc[:min(len(sim.trajectory_mc), _npoints_to_plot)], color="red")[0]
            ax_trajectory.axhline(0.0, color="gray")
            ax_trajectory.set_xlabel("time (sec)")
            ax_trajectory.set_ylabel("X(t)")
            ax_trajectory.set_title(f"MC simulation for the last replication run r={r+1}: #steps={nsteps} at dt={dt} sec\n(plotting the first {_npoints_to_plot} out of total {len(sim.trajectory_mc)} points)")
            # Add eventual Levy jumps
            if sets_case == 4:
                ax_trajectory.plot(np.array(sim.jump_times)*dt, sim.trajectory_mc[sim.jump_times], 'k.', markersize=5)
                for idx, t in enumerate(sim.jump_times):
                    line_jump = ax_trajectory.plot([t*dt, t*dt], [sim.trajectory_mc[t] - sim.jump_values[idx], sim.trajectory_mc[t]], color="black")[0]
                ax_trajectory.legend([line_process, line_jump], ["Process", f"Jump sizes (total jumps: {len(sim.jump_values)}, avg. value = {np.mean(sim.jump_values):.3g})"])

            # Distribution of values observed, to check against the real Fingrid data
            # Gaussian fit to the distribution
            params_normal = stats.norm.fit(50 + sim.trajectory_mc)
            test_normal = stats.normaltest(sim.trajectory_mc)
            fontsize = 20
            histtype = "step"
            plt.figure()
            line_hist_sim = plt.hist(sim.trajectory_mc, color="darkred", bins=100, density=True, alpha=0.5, histtype=histtype, edgecolor="darkred")[2][0]
            line_fit_sim = plt.plot(np.array(sorted(sim.trajectory_mc)), stats.norm.pdf(50 + np.array(sorted(sim.trajectory_mc)), *params_normal), color="red")[0]
            plt.axvline(0.0, color="black", linestyle="dashed")
            plt.gca().set_xlabel("Frequency [Hz]", fontsize=fontsize)
            plt.gca().set_ylabel("Density (log scale)", fontsize=fontsize)
            plt.gca().tick_params(axis='both', labelsize=int(0.6*fontsize))
            plt.gca().set_yscale('log')
            plt.gca().set_xlim((-1.05*np.max(np.abs(plt.gca().get_xlim())), 1.05*np.max(np.abs(plt.gca().get_xlim()))))
            #plt.gca().xaxis.set_major_locator(MaxNLocator(symmetric=True))  # Does not really work... i.e. the axis does NOT become symmetric!
            print(f"The estimated mean and standard deviation of the distribution are: mu = {np.mean(50 + sim.trajectory_mc):.6f}, sigma = {np.std(sim.trajectory_mc):.6f}")
            ## Comparing to the real Fingrid data from 2025-01 (31 days) we see that the distribution is more narrow, ranging from 49.85 to 50.15 instead of from 49.75 to 50.25.

            # Plot on top the distribution of the real Fingrid data (READ ELSEWHERE from e.g. 'VU/data/Fingrid - 2025 - 0.1 sec') into variable `x`
            if "x" in locals():
                line_hist_data = plt.hist(x - 50, color="blue", bins=100, density=True, alpha=0.5, histtype=histtype, edgecolor="blue")[2][0]
                ## They in fact do not ressemble too much...
                if sets_case == 4:
                    legend_str_simulation = f"Simulated data with Levy noise (N={len(sim.trajectory_mc)/1E6:.1f} Mio., lambda={levy['lambda']}, params: mean={levy['params_dist_jumps']['mean']:.3g}, sigma={levy['params_dist_jumps']['sigma']:.3g}, shift={levy['params_dist_jumps']['shift']:.3g})"
                else:
                    legend_str_simulation = f"Simulated data with Gaussian noise (N={len(sim.trajectory_mc)/1E6:.1f} Mio., params: sigma={sigma})"
                plt.legend([line_hist_sim, line_fit_sim, line_hist_data],
                            [legend_str_simulation,
                            f"Gaussian fit (mu={params_normal[0]:.1f}, sigma={params_normal[1]:.3g})",
                            f"Actual data from 2025-01 (N={len(x)/1E6:.1f} Mio., sigma={np.std(x):.3g})"],
                           loc="upper center")
                plt.title(f"Comparison of actual frequency distribution for {title_header}\nand the simulated frequency process", fontsize=fontsize)

            # Distribution of deviations and jumps
            if sets_case == 4:
                plt.figure()
                line_hist_sim = plt.hist(np.diff(sim.trajectory_mc), color="red", bins=100, density=True, alpha=0.5)[2][0]
                line_hist_data = plt.hist(np.abs(sim.jump_values), color="orange", density=True, alpha=0.5)[2][0]
                line_jumps_model = plt.plot(sorted(np.abs(sim.jump_values)), stats.lognorm.pdf(sorted(np.abs(sim.jump_values)), *(shape, loc, scale)), color="blue")[0]
                plt.gca().set_xlabel("dX(t)")
                plt.gca().set_ylabel("Density")
                thr = 0.016187  # Taken from the estimation of the jump size distribution
                line_thr = plt.axvline(thr, color="black", linestyle="dashed")
                #plt.gca().set_yscale('log')
                plt.legend([line_hist_sim, line_hist_data, line_jumps_model, line_thr],
                           ["dX(t) distribution", "Jump size distribution", "Lognormal model for the jump sizes", f"Threshold for 'pure' jumps: thr={thr}"], loc="upper left")
                plt.title(f"Distribution of dX(t) and negative jumps (shown as positive values)\nTotal # jumps = {np.sum(sim.jump_numbers)}")

        time_end = timer()
        time_elapsed = time_end - time_start
        print(f"\nThe WHOLE experiment ends at {datetime.now()} and took {time_elapsed/60:.1f} minutes ({time_elapsed/3600:.1f} hours)")

        #raise KeyboardInterrupt

        #--- Analysis of results
        # Results are analyzed only when FV and MC have been run
        if {"FV0", "FV"}.intersection(tests2run) != set() and {"MC"}.intersection(tests2run) != set():
            if sets_case == 1:
                # Normal-probability sets of interest
                intervals2analyze = [1, 4]
            elif sets_case == 2:
                # Rarer sets of interest and larger A
                intervals2analyze = [2, 5]
            elif sets_case == 3:
                # Reflected process
                intervals2analyze = [1]
            elif sets_case == 4:
                # Levy process: we choose sets associated to negative drifts
                intervals2analyze = [0]
            else:
                # There is always one set in sets_of_interest, so we choose the first one appearing in sets_of_interest by default
                intervals2analyze = [0]
            sets2analyze = [sets_of_interest[i] for i in intervals2analyze]

            print(f"\nTrue probabilities of the sets of interest:")
            proba_true = np.sum([calculate_probability((_set.inf, _set.sup), test_obj.r, test_obj.x, sigma) for _set in sets2analyze])
            if sets_case == 3:
                # This is the case where the simulation is done on the reflected process, meaning that we only specify the interval on ONE side of the origin in sets2analyze
                # => We need to multiply the computed true probability by 2 because we also need to add the probability of the corresponding negative interval
                proba_true = 2*proba_true

            print(f"\nProbabilities estimated by FV (N={dict_params_simul['N']}, T={dict_params_simul['T']}, A={dict_params_simul['absorption_set']}):")
            # Compute the estimated probabilities of the set to analyze and collect them on a data frame for easy plotting
            df_results = pd.DataFrame(np.zeros((nrep, 4)), columns=['mc', 'fv', 'nsteps_mc', 'nsteps_fv'], index=np.arange(nrep))
            for r in range(nrep):
                for _set in sets2analyze:
                    df_results.iloc[r]['mc'] += probs_mc[r]['probabilities'][_set]
                    df_results.iloc[r]['fv'] += probs_fv[r]['probabilities'].get(_set, 0.0)  # We use get() because for a particular replication, FV may not have been run because of no exit states during the MC simulation!
                df_results.iloc[r]['nsteps_mc'] = probs_mc[r]['info']['last_time_observed']
                df_results.iloc[r]['nsteps_fv'] = probs_fv[r]['info']['nsteps']
            print(f"Estimated probabilities for the sets of interest: {sets2analyze}")
            print(df_results)
            df_toplot = pd.concat([
                                    pd.concat([pd.DataFrame({'method': "mc"}, index=np.arange(nrep)), df_results[['mc', 'nsteps_mc']].rename(columns={"mc": "p", "nsteps_mc": "nsteps"})], axis=1),
                                    pd.concat([pd.DataFrame({'method': "fv"}, index=np.arange(nrep)), df_results[['fv', 'nsteps_fv']].rename(columns={"fv": "p", "nsteps_fv": "nsteps"})], axis=1)
                                    ], axis=0)
            plt.figure()
            sns.boxplot(data=df_toplot, x="method", y="p", hue="method", order=["fv", "mc"], palette={'fv': "green", 'mc': "red"}, orient="v")
            plt.axhline(0, color="gray")
            plt.axhline(proba_true, color="gray", linestyle="dashed")
            if sets_case == 1:
                plt.axhline(np.sum([probas_true[f'p_{i+1}'] for i in intervals2analyze]), color="gray", linestyle="dashed")
            plt.title(f"Estimated probabilities for {sets2analyze} on {nrep} replications\nN={dict_params_simul['N']}, T={dict_params_simul['T']}, A={dict_params_simul['absorption_set']}, avg(#steps)={int(np.mean(df_toplot['nsteps']))}")

            # Using the usual matplotlib (without respecting the colors though! (left = FV, right = MC)
            ax = plt.figure().subplots(1, 1)
            toplot_mc = [np.array(df_results['mc'])]
            toplot_fv = [np.array(df_results['fv'])]
            violinparts_fv = ax.violinplot(toplot_fv, positions=[1], widths=0.1, showmedians=True, showmeans=False, showextrema=True)
            violinparts_mc = ax.violinplot(toplot_mc, positions=[2], widths=0.1, showmedians=True, showmeans=False, showextrema=True)
            violinparts_fv['bodies'][0].set_facecolor("green")
            violinparts_mc['bodies'][0].set_facecolor("red")
            ax.plot(1 + 0.02*np.random.normal(size=len(df_results['fv'])), df_results['fv'], 'k.')
            ax.legend([violinparts_fv['bodies'][0], violinparts_mc['bodies'][0]], [f"FV ({np.sum(df_results['fv'] > 0)} non-zero estimates out of {nrep})", f"MC ({np.sum(df_results['mc'] > 0)} non-zero estimates out of {nrep})"])
            ax.set_ylim((4E-9, 1E-3))
            ax.set_yscale('log')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.title(f"Estimated probabilities for {sets2analyze} on {nrep} replications\nN={dict_params_simul['N']}, T={dict_params_simul['T']}, A={dict_params_simul['absorption_set']}, avg(#steps)={int(np.mean(df_toplot['nsteps']))}")
            print(f"MC: {np.sum(df_results['mc'] > 0)} non-zero estimates out of {nrep}")
            print(f"FV: {np.sum(df_results['fv'] > 0)} non-zero estimates out of {nrep}")

            raise KeyboardInterrupt

            # In case we need to save results
            import os

            rootdir = r"E:\Daniel\Projects\PhD-RL-Toulouse\projects\RL-004-Energy"
            resultsdir = os.path.realpath(rootdir + "/results")
            print(f"Results will be saved to {resultsdir}")

            filename = os.path.join(resultsdir, "ou_pwlinear_probability_estimation.csv")
            pd.DataFrame([df_results.columns]).to_csv(filename, header=False, index=False)
    else:
        # Reference for creating test suites:
        # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
        runner = unittest.TextTestRunner()

        # Run all tests
        # unittest.main()

        test_suite = unittest.TestSuite()
        test_suite.addTest(Test_Class_SimulatorDiffusionFV("test_run_simulation_mc"))
        test_suite.addTest(Test_Class_SimulatorDiffusionFV("test_run_simulation_fv"))
        test_suite.addTest(Test_Class_SimulatorDiffusionFV("test_run_estimation_fv"))
        runner.run(test_suite)
