# -*- coding: utf-8 -*-
"""
Created on Sun May 11 14:52:45 2025

@author: Daniel Mastropietro
@description: Unit tests for functions and methods defined in simulators.diffusion.
@details: Naming conventions follow the instructions given in test_conventions.txt.
"""

import runpy
runpy.run_path('../../setup.py')

import unittest
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import sympy
from scipy.integrate import quad

from Python.lib.estimators.fv import estimate_stationary_probabilities
from Python.lib.environments.diffusion import EnvDiffusion, func_drift_fcr_piecewise, func_noise_gaussian
from Python.lib.simulators.diffusion import SimulatorDiffusionFV


def calculate_probabilities(r, x, sigma):
    """
    Calculate the probabilities p_1,...,p_6 as a function of r, x, and sigma.

    Parameters:
    r (float): Capacity reserved for FCR-N.
    x (float): Decision variable, capacity reserved for FCR-D.
    sigma (float): Volatility.

    Returns:
    dict: A dictionary with the probabilities p_1, p_2, p_3 (and p_4, p_5, p_6 since p_4 = p_3, p_5 = p_2, p_6 = p_1).
    """
    K1, K2, K3 = compute_K_integrals(x, r, sigma)

    exp1 = np.exp(-5 * x / (48 * sigma ** 2))
    exp2 = np.exp(-(31 * x + r) / (300 * sigma ** 2))

    if K1 == 0 or np.log10(K1) < -15:
        denom = 2 * K2 * exp1 + 2 * K3 * exp2
        p_1 = p_6 = 0.0
        p_2 = p_5 = K2 * exp1 / denom
        p_3 = p_4 = K3 * exp2 / denom
    else:
        denom = 2 + 2 * (K2 / K1) * exp1 + 2 * (K3 / K1) * exp2
        p_1 = p_6 = 1 / denom
        p_2 = p_5 = (K2 / K1) * exp1 / denom
        p_3 = p_4 = (K3 / K1) * exp2 / denom

    print(
        f"p_1: {round(p_1, 10)}\np_2: {round(p_2, 10)}\np_3: {round(p_3, 10)}\np_4: {round(p_4, 10)}\np_5: {round(p_5, 10)}\np_6: {round(p_6, 10)}"
    )

    return {
        "p_1": p_1,
        "p_2": p_2,
        "p_3": p_3,
        "p_4": p_4,
        "p_5": p_5,
        "p_6": p_6,
    }


def compute_K_integrals(x, r, sigma, beta=1):
    def K1_integrand(y):
        return np.exp(-(y ** 2) * (beta * r + beta * x) / sigma ** 2)

    def K2_integrand(y):
        return np.exp(
            -(y ** 2) * (4 * beta * r - beta * x) / (4 * sigma ** 2)
            + y ** 3 * (5 * beta * x) / (3 * sigma ** 2)
        )

    def K3_integrand(y):
        return np.exp((20 * y ** 3 * beta * r) / (3 * sigma ** 2))

    K1 = quad(K1_integrand, -np.inf, -0.5, epsabs=1e-30, epsrel=1e-11)[0]
    K2 = quad(K2_integrand, -0.5, -0.1, epsabs=1e-30, epsrel=1e-11)[0]
    K3 = quad(K3_integrand, -0.1, 0.0, epsabs=1e-30, epsrel=1e-11)[0]

    return K1, K2, K3


def extract_prob_intervals(F, intervals):
    """
    Extracts the probabilities of the simulated OU process being in given intervals.
    This function counts the number of samples in each interval and normalizes by the total number of samples.

    Parameters:
    - F: Simulated OU process.
    - intervals: List of tuples defining the intervals.

    Returns:
    - frequencies: List of probabilities to be in each interval.
    """
    frequencies = []
    for lower, upper in intervals:
        count = np.sum((F >= lower) & (F < upper))
        frequencies.append(count / len(F))  # Normalize by the total number of samples
    return frequencies


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
        cls.r = 0.6  # Capacity reserved for FCR-N
        cls.x = 1.4  # Capacity reserved for FCR-D
        cls.env_ou = EnvDiffusion(func_drift=func_drift_fcr_piecewise, func_noise=func_noise_gaussian, mu=mu, dt=dt, sigma=sigma, r=cls.r, x=cls.x)

    def test_run_estimation_fv(self, dict_params_simul=dict({'N': 50, 'T': 1000, 'absorption_set': sympy.Interval(-0.08, 0.08)}), store_trajectories=False, seed=1313, debug=False, plot={'MC': False, 'FV': False}):
        """ (2025/05/12)
        Tests the estimation of stationary probability of sets of interest by Fleming-Viot
        """
        print("\n*** Running test " + self.id() + " ***")

        # Thresholds defining intervals of interest
        # Recall that these intervals should be OUTSIDE the absorption set A!
        thresholds = [-np.Inf, -0.5, -0.1, 0.0, 0.1, 0.5, +np.Inf]
        sets_of_interest = list()
        for i in range(len(thresholds)-1):
            # When considering both sides of the process deviation in the same set (e.g. (-0.5, -0.1) U (0.1, 0.5))
            #sets_of_interest += [sympy.Union( sympy.Interval(-thresholds[i+1], -thresholds[i]), sympy.Interval(thresholds[i], thresholds[i+1]) )]
            # When considering each side of the process deviation as separate sets (e.g. (-0.5, -0.1) separately from (0.1, 0.5))
            sets_of_interest += [sympy.Union(sympy.Interval(thresholds[i], thresholds[i + 1]))]

        # Expected results: probabilities of each smooth piece of the drift function
        calculate_probabilities(self.r, self.x, self.env_ou.sigma)

        # Experimental results
        sim = SimulatorDiffusionFV(self.env_ou, None, debug=debug)
        print(f"Running estimation by Fleming-Viot of the stationary probabilities of sets of interest of a diffusion process discretized at dt={self.env_ou.dt} intervals, with the SimulatorDiffusionFV.run() method...")

        time_start = timer()
        probas_stationary, info = sim.run(dict_params_simul, sets_of_interest,
                                          store_trajectories=store_trajectories,
                                          seed=seed, verbose=True, verbose_period=10, plot=plot)

        print(f"Estimated expected cycle time, E(T_A): {info['MC']['expected_cycle_time']} (n={info['MC']['n_cycles']}) on T = {dict_params_simul['T']} steps")
        print(f"Estiated stationary probabilities with N={dict_params_simul['N']} particles:")
        print("_run_simulation_mc() took {:.4f} seconds".format(timer() - time_start))

        return probas_stationary, info

    def test_run_simulation_mc(self, nsteps=1000, absorption_set=sympy.Interval(-0.08, 0.08), start_state=0.0, store_trajectories=False, seed=1313, debug=False, plot=False):
        """ (2025/05/11)
        Tests the Monte-Carlo simulation run by the _run_simulation_mc() method
        """
        print("\n*** Running test " + self.id() + " ***")

        # Simulation parameters
        dict_params_simul = dict({'T': nsteps,                          # Number of integer steps for the MC simulator that estimates the expected cycle time to A, E(T_A)
                                  'absorption_set': absorption_set,     # Absorption set defined as sympy.Set. If an Interval, the interval ends are included by default.
                                                                        # Use `left_open=True` and `right_open=True` to respectively open them.
                                  })

        # Expected results: probabilities of each smooth piece of the drift function
        calculate_probabilities(self.r, self.x, self.env_ou.sigma)

        # Experimental results
        sim = SimulatorDiffusionFV(self.env_ou, None, debug=debug)
        print(f"Running simulation of diffusion process, discretized at dt={self.env_ou.dt} intervals, with _run_simulation_mc()...")
        time_start = timer()
        average_cycle_time, n_cycles, dist_exit_state, info = sim._run_simulation_mc(dict_params_simul, start_state=start_state, store_trajectory=store_trajectories,
                                                                                                        seed=seed, verbose=True, verbose_period=10, plot=plot)
        print(f"Estimated EXIT state distribution from A (n={n_cycles}):\n{dist_exit_state}")
        print(f"Estimated expected cycle time, E(T_A): {average_cycle_time} (n={n_cycles})")
        print("_run_simulation_mc() took {:.4f} seconds".format(timer() - time_start))

        return sim, average_cycle_time, n_cycles, dist_exit_state, info

    def test_run_simulation_fv(self, N=50, dist_exit_state=pd.DataFrame({'x': [-0.09, 0.09], 'p': [0.5, 0.5]}), absorption_set=sympy.Interval(-0.08, 0.08), store_trajectories=True, seed=1313, debug=False, plot=False):
        """ (2025/05/11)
        Tests the Fleming-Viot simulation run by the _run_simulation_fv() method
        """
        print("\n*** Running test " + self.id() + " ***")

        # Simulation parameters
        dict_params_simul = dict({'N': N,
                                  'T': 1E7,                             # Maximum number of steps for which the FV simulator should be run (in case the stopping criterion fails)
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
        calculate_probabilities(self.r, self.x, self.env_ou.sigma)

        # Experimental results
        sim = SimulatorDiffusionFV(self.env_ou, None, N=N, debug=debug)
        print(f"Running simulation of diffusion process, discretized at dt={self.env_ou.dt} intervals, with _run_simulation_mc()...")
        time_start = timer()
        df_proba_surv, dict_phi, info = \
            sim._run_simulation_fv(dict_params_simul, dist_exit_state, sets_of_interest, store_trajectory=store_trajectories,
                                                                                         seed=seed, verbose=True, verbose_period=10, plot=plot)
        print("_run_simulation_fv() took {:.4f} seconds".format(timer() - time_start))

        return sim, df_proba_surv, dict_phi, info


if __name__ == "__main__":
    test = False
    tests2run = ["FV", "MC"]    #["FV0", "MC"]

    if not test:
        from matplotlib import pyplot as plt, cm

        test_obj = Test_Class_SimulatorDiffusionFV()
        # Need to define selected process parameters and the environment ad-hoc because they are not found when defined via setUpClass() in this ad-hoc execution
        mu = 0.0
        sigma = 0.04
        dt = 0.01
        test_obj.r = 0.6  # Capacity reserved for FCR-N
        test_obj.x = 1.4  # Capacity reserved for FCR-D
        test_obj.env_ou = EnvDiffusion(func_drift=func_drift_fcr_piecewise, func_noise=func_noise_gaussian, mu=mu, dt=dt, sigma=sigma, r=test_obj.r, x=test_obj.x)

        if "FV" in tests2run:   # This test runs the whole process altogether, by calling the run() method of the SimulatorDiffusionFV class
            A_boundaries = (-0.08, +0.08)
            absorption_set = sympy.Interval(*A_boundaries)
            seed = 1317 #1313

            dict_params_simul = dict({'N': 50,  #100
                                      'T': 10000,
                                      'absorption_set': absorption_set,
                                      })
            probas_stationary, info = test_obj.test_run_estimation_fv(dict_params_simul, store_trajectories=True, seed=seed, debug=False, plot={'MC': False, 'FV': True})

            # Estimated probability of each smooth piece of the drift function
            intervals = [
                (-1, -0.5),
                (-0.5, -0.1),
                (-0.1, 0),
                (0, 0.1),
                (0.1, 0.5),
                (0.5, 1),
            ]
            print("\nTrue probabilities:")
            probs_true = calculate_probabilities(test_obj.r, test_obj.r, sigma)

            print(f"\nProbabilities estimated by FV (N={dict_params_simul['N']}, T={dict_params_simul['T']}, A={dict_params_simul['absorption_set']}, # steps={info['nsteps']}):")
            for C, p in probas_stationary.items():
                print(f"{C.boundary}: {p}")

        if "FV0" in tests2run:  # This test runs _run_simulation_mc() and _run_simulation_fv() separately
            A_boundaries = (-0.08, +0.08)
            absorption_set = sympy.Interval(*A_boundaries)
            seed = 1317 #1313

            # Estimate the exit state distribution from A, so that we know how to choose the start state of the FV particles
            nsteps_et = 10000
            start_state = 0.0   #A_boundaries[1]
            sim_mc, average_cycle_time, n_cycles, dist_exit_state, info_mc = \
                test_obj.test_run_simulation_mc(nsteps=nsteps_et, absorption_set=absorption_set, start_state=start_state, store_trajectories=True, seed=seed, debug=False, plot=False)

            # Run the FV simulation now
            N = 50
            sim_fv, df_proba_surv, dict_phi, info_fv = \
                test_obj.test_run_simulation_fv(N, dist_exit_state, absorption_set=absorption_set, store_trajectories=True, seed=seed+13, debug=False, plot=False)

            # Estimate the stationary probabilities of the sets of interest
            probas_stationary, integrals = estimate_stationary_probabilities(dict_phi, df_proba_surv, average_cycle_time, uniform_jump_rate=N)

            print("Theoretical probabilities:")
            calculate_probabilities(test_obj.r, test_obj.x, sigma)

            print(f"Estimated probabilites (N={N}, # steps={info_fv['last_updated_time']}):")
            for C, p in probas_stationary.items():
                print(f"{C.boundary}: {p}")

            #--- Plot the trajectories and long-run statistics (mu and sigma)
            ax_mc, ax_longrun, ax_fv = plt.figure().subplots(1, 3)
            # MC
            ax_mc.plot(sim_mc.trajectory_mc, color="red")
            ax_mc.plot(info_mc['exit_times'], info_mc['exit_states'], "ko", markersize=3)
            ax_mc.axhline(A_boundaries[0], color="blue", linestyle="dashed")
            ax_mc.axhline(A_boundaries[1], color="blue", linestyle="dashed")
            ax_mc.axhline(0.0, color="gray")
            ax_mc.set_xlabel("t")
            ax_mc.set_title(f"MC simulation on {nsteps_et} steps")

            # Long-run estimations
            cumN = np.linspace(1, len(sim_mc.trajectory_mc), len(sim_mc.trajectory_mc))
            mu_est = np.cumsum(sim_mc.trajectory_mc) / cumN
            mu_std = np.sqrt( np.cumsum(sim_mc.trajectory_mc**2) / cumN - mu_est**2 )
            ax = plt.figure().subplots(1, 1)
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

        if "MC" in tests2run:
            # Use as number of steps of the MC simulation the same as the number of events observed during the FV simulation above, in case this was run (for comparison purposes)
            if {"FV", "FV0"}.intersection(tests2run) == {}:
                nsteps = 10000
            elif "FV" in tests2run:
                nsteps = info['FV']['last_updated_time'] + info['MC']['last_updated_time']
            elif "FV0" in tests2run:
                nsteps = info_fv['last_updated_time'] + nsteps_et
            start_state = 0.0
            seed = 1313
            sim, _, _, _, info_mc = \
                test_obj.test_run_simulation_mc(nsteps=nsteps, absorption_set=sympy.FiniteSet(0), start_state=start_state, store_trajectories=True, seed=seed, debug=False, plot=False)

            # Estimated probability of each smooth piece of the drift function
            intervals = [
                (-1, -0.5),
                (-0.5, -0.1),
                (-0.1, 0),
                (0, 0.1),
                (0.1, 0.5),
                (0.5, 1),
            ]
            probs_mc = extract_prob_intervals(sim.trajectory_mc, intervals)
            print(f"Probabilities estimated by MC (# steps = {info_mc['last_updated_time']}):")
            for i, ((lower, upper), prob) in enumerate(zip(intervals, probs_mc)):
                print(f"p_{i + 1}: {prob}")

            if "FV" in tests2run:
                print(f"\nProbabilities estimated by FV (N={dict_params_simul['N']}, T={dict_params_simul['T']}, A={dict_params_simul['absorption_set']}, # steps={info['nsteps']}):")
                for C, p in probas_stationary.items():
                    print(f"{C.boundary}: {p}")

            if "FV0" in tests2run:
                print(f"\nProbabilities estimated by FV (N={N}, T={nsteps_et}, A={absorption_set}, # steps={info_fv['last_updated_time'] + nsteps_et}):")
                for C, p in probas_stationary.items():
                    print(f"{C.boundary}: {p}")

            print("\nTrue probabilities:")
            probs_true = calculate_probabilities(test_obj.r, test_obj.r, sigma)

            # Trajectory
            ax_trajectory = plt.figure().subplots(1, 1)
            ax_trajectory.plot(sim.trajectory_mc, color="red")
            ax_trajectory.axhline(0.0, color="gray")
            ax_trajectory.set_xlabel("t")
            ax_trajectory.set_title(f"MC simulation on {nsteps} steps")
    else:
        # Reference for creating test suites:
        # https://stackoverflow.com/questions/15971735/running-single-test-from-unittest-testcase-via-command-line
        runner = unittest.TextTestRunner()

        # Run all tests
        # unittest.main()

        test_suite = unittest.TestSuite()
        test_suite.addTest(Test_Class_SimulatorDiffusionFV("test_run_simulation_mc"))
        test_suite.addTest(Test_Class_SimulatorDiffusionFV("test_run_simulation_fv"))
        runner.run(test_suite)
