# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:09:17 2025

@author: Daniel Mastropietro
@description: Definition of functions and classes used for simulation of a diffusion process, including online learning (with e.g. FV particle systems).
"""

import warnings

import numpy as np
import pandas as pd
from collections import deque
import sympy
from matplotlib import pyplot as plt, cm
from tqdm import tqdm

from Python.lib.environments.diffusion import EnvDiffusion
from Python.lib.estimators.fv import estimate_stationary_probabilities
from Python.lib.simulators.discrete import Simulator
from Python.lib.utils.basic import measure_exec_time
from Python.lib.utils.computing import compute_survival_probability


class SimulatorDiffusionFV(Simulator):
    """
    Implements the simulation of a diffusion process as required by the Fleming-Viot estimator of stationary probabilities

    Hence, the simulator is exclusively implemented for such FV estimation, which means that no FV learner is used as part of the process.
    Even though an `agent` is received by the constructor, it is actually NOT used.

    I am doing this now to analyze if this type of implementation (with less interdependence with other pats of the code) is easier to implement
    compared to what I have done already in the SimulatorQueue class and in the Simulator class.
    Let's see... otherwise I can always refactor the code later.
    """
    def __init__(self, env, agent, dict_params_learning=dict(), case=1, replication=1, N=10, seed=None, log=False, save=False, logsdir=None, resultsdir=None, debug=False):
        super().__init__(env, agent, case=case, replication=replication, seed=seed, log=log, save=save, logsdir=logsdir, resultsdir=resultsdir, debug=debug)

        if not isinstance(env, EnvDiffusion):
            raise ValueError(f"The environment must be of type `EnvironmentDiffusion`: {env}")

        # Basic learning characteristics (e.g. learn time (t_learn), learning mode, etc.)
        self.dict_params_learning = dict_params_learning

        # Trajectory stored after simulations, if requested
        self.trajectory_mc = None
        self.trajectories_fv = None

    def run(self, dict_params_simul: dict, sets_of_interest :list,
            dict_params_info: dict={'plot': False, 'log': False},
            dict_info: dict={},
            store_trajectories=False,
            seed=None,
            verbose=False, verbose_period=1, plot={'MC': False, 'FV': False}):
        """
        Runs the simulations necessary to estimate probability of selected sets in a diffusion process

        Arguments:
        dict_params_simul: dict
            Dictionary which should contain at least the following keys:
            - N: # particles to use in the Fleming-Viot particle system.
            - T: # steps to run the simulation that estimates the expected return time to the absorption set A.
            - absorption_set: sympy.Set defining the absorption set A.

        sets_of_interest: list of sympy.Set
            List containing the sets of interest on which the stationary probability should be estimated with Fleming-Viot.
        """
        #--- Parse input parameters
        N = dict_params_simul.get('N')
        if not isinstance(N, int) or N <= 1:
            raise ValueError(f"The number of particles must be passed in dict_params_simul['N'] and must be an integer larger than 1: {N}")

        T = dict_params_simul.get('T')
        if not isinstance(T, int) or T <= 1:
            raise ValueError(f"The number of integer steps to use in the Monte-Carlo simulation that estimates the expected return time to the absorption set A must be passed "
                             f"in dict_params_simul['T'] and must be an integer larger than 1: {T}")

        absorption_set = dict_params_simul.get('absorption_set')
        if not isinstance(absorption_set, sympy.Set):
            raise ValueError(f"The absorption set A must be passed in dict_params_simul['absorption_set'] and must be of type `sympy.Set`: {absorption_set}")
        #--- Parse input parameters

        # 1) Monte-Carlo simulation
        # Estimation of the expected return time to A.
        # Simulation parameters:
        # - # simulation steps, T
        # - the absorption set A (possibly as a sympy.logic.boolalg.Boolean.as_set() object which is of type sympy.sets.sets.UniversalSet --> However, it seems for this I would need to upgrade Python because I have sympy-1.1.1 and the latest is sympy-1.14.0 and the former version does NOT have sympy.logic.boolalg defined!!
        # - (opt, not prio) min number of cycles to compute expectation
        # - (opt, not prio) burn-in time
        start_state = 0.0 #float(list(absorption_set.boundary)[0])   # IMPORTANT: convert to float() o.w. the exit states in _run_simulation_mc() are stored as type `object` instead of `float`!!
        expected_cycle_time, n_cycles, dist_exit_state, info_mc = self._run_simulation_mc(dict_params_simul, start_state=start_state, store_trajectory=store_trajectories,
                                                                                          seed=seed, verbose=verbose, verbose_period=verbose_period, plot=plot.get('MC', False))

        # 2) Fleming-Viot simulation
        # Estimation of the survival probability and the conditional occupation probability of each set of interest.
        # Simulation parameters:
        # - # particles, N
        # - the absorption set A
        # - sets C of states of interest
        # OUTPUT:
        # - df_proba_surv: as in the discrete.Simulator case, a data frame with two columns, 't', 'P(T>t)' where t is the observed DISCRETE survival times
        # - dict_phi: an entry per set of interest, whose value is a data frame with two columns, 't', 'Phi' containing the DISCRETE times at which Phi changes
        #   and the corresponding Phi value (after the change).
        df_proba_surv, dict_phi, info_fv = self._run_simulation_fv(dict_params_simul, dist_exit_state, sets_of_interest, store_trajectory=store_trajectories,
                                                                   seed=seed + 131713, verbose=verbose, verbose_period=verbose_period, plot=plot.get('FV', False))

        # Estimation of the stationary probabilities of each set of interest
        probas_stationary, integrals = estimate_stationary_probabilities(dict_phi, df_proba_surv, expected_cycle_time, uniform_jump_rate=N)

        # Additional information that may be of interest
        # Note the merge of two dictionaries with the `**` operator
        # (new in Python-3.5, Ref: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python)
        info = dict({'nsteps': info_mc['last_updated_time'] + info_fv['last_updated_time'],
                     'MC': {**{ 'expected_cycle_time': expected_cycle_time,
                                'n_cycles': n_cycles,
                                'dist_exit_state': dist_exit_state},
                            **info_mc},
                     'FV': {**{ 'df_proba_surv': df_proba_surv,
                                'dict_phi': dict_phi,
                                'integrals': integrals},
                            **info_fv},
                     })

        return probas_stationary, info

    @measure_exec_time
    def _run_simulation_mc(self, dict_params_simul, start_state=0.0, store_trajectory=False, seed=None, verbose=False, verbose_period=1, plot=False):
        """
        Runs the Monte-Carlo simulation that is used to estimate the denominator of the FV estimator of the stationary probability,
        the expected return time to the absorption set A, E(T_A)

        Arguments:
        dict_params_simul: dict
            Simulation parameters containing at least the following entries:
            - 'T': number of (integer) steps to run the simulation for.
            - 'absorption_set': sympy.Set defining the absorption set A, which defines the return cycle whose expected time is estimated.

        start_state: float
            Initial state of the process.
            default: 0.0

        store_trajectory: bool
            Whether the observed trajectory should be stored in this object as attribute self.trajectory_mc, an np.array of length dict_params_simul['T'] + 1
            starting at time t=0 and ending at time dict_params_simul['T'].
            default: False

        seed: int
            Seed for random number generation (set and stored in the environment `self.env`).
            default: None

        verbose: bool
            Whether to show messages in the console about the simulation progress.
            default: False

        verbose_period: int (currently NOT used)
            Every how many time steps the information shown by `verbose` should be displayed.
            default: 1

        plot: bool
            Whether to show a dynamic plot of the trajectory as it is being generated.
            Only the last 30 time steps are shown (to avoid unbounded increase of execution time).
            default: False

        Return: tuple
        The following 4 objects are returned:
        - expected_cycle_time: float, estimated expected cycle time of return to the absorption set A, E(T_A).
        - n_cycles: int, number of full cycles observed, used in the estimation of the expected cycle time.
        - dist_exit_state: pandas data frame containing the distribution of the exit states. The indices are tuples containing the bin intervals.
        The columns are 'x' and 'p', respectively containing the midpoint of the corresponding bin and the observed relative frequency on exit from A.
        - info: dict with additional information that may be of interest. The following entries are stored: 'exit_times', 'exit_states'.
        """
        # If needed, base the process on what is done in simulators.queues.run_simulation_mc().

        #--- Parse input parameters
        # Absorption set definition
        # TEMPORARY for 1D states: the absorption set is assumed to be a sympy.Set (https://docs.sympy.org/latest/modules/sets.html). Note that Sets can be of many `kind`s. See the `kind` property of the set.
        if not isinstance(dict_params_simul.get('absorption_set'), sympy.Set):
            raise ValueError(f"Parameter `dict_params_simul['absorption_set']` must be of type sympy.Set (e.g. `sympy.Union(sympy.Interval(-np.Inf, -0.1), sympy.Interval(0.1, +np.Inf))`): {dict_params_simul['absorption_set']}")
        absorption_set = dict_params_simul['absorption_set']

        # Seed
        if seed is not None:
            self.env.set_seed(seed)
        #--- Parse input parameters

        # What defines a FULL cycle?
        # Full cycles are defined from first to last EXIT event if the start state is INSIDE the absorption set, o.w. full cycles are defined from first to last ENTRY event
        full_cycles_marked_by_exit_events = absorption_set.contains(start_state)

        # Times to keep track of in order to estimate the expected cycle time to A
        # They are initialized as -1 (and NOT np.nan) because the variables are integer-valued.
        last_time_exit = -1
        last_time_entry = -1
        first_time_in_cycle = -1
        n_cycles = 0

        # Information about the exit times and exit states from A (these are states in the complement of A and are used to start the FV simulation)
        exit_times = deque()
        exit_states = deque()

        if store_trajectory:
            self.trajectory_mc = np.nan*np.ones(dict_params_simul['T'] + 1)

        # Reset the environment to the start state
        self.env.setState(start_state)
        if store_trajectory:
            self.trajectory_mc[0] = self.env.getState()
        if verbose:
            print(f"MC simulation for the estimation of E(T_A): the environment starts at state {self.env.getState()} and uses seed={seed}")
        if plot:
            import matplotlib.pyplot as plt
            ax = plt.figure().subplots(1, 1)
            plt.suptitle(f"MC simulation on {dict_params_simul['T']} steps")
            ax.set_xlabel("t")
            for _boundary in absorption_set.boundary:
                ax.axhline(_boundary, color="blue", linestyle="dashed")
            ax.axhline(0.0, color="gray")
            ax.plot(0, start_state, color="red")
        for t in tqdm(range(1, dict_params_simul['T']+1)):    # We start the time step at 1 because t=0 corresponds to the initial state which is stored above
            state = self.env.getState()
            next_state, _, _, _ = self.env.step(0)
            if store_trajectory:
                self.trajectory_mc[t] = next_state

            # Plot
            if plot:
                ax.plot([t-1, t], [state, next_state], color="red")
                ax.set_title(f"t = {t}")
                ax.set_xlim((max(0, t - 30), t))    # Show the last 30 steps only (to speed up the plotting process)
                plt.pause(0.00000001)
                plt.draw()

            # Check exit from A (i.e. state is inside A and next_state is outside A)
            if exit_from_absorption_set(absorption_set, state, next_state):
                exit_times.append(t)
                exit_states.append(next_state)
                #exit_times.append(t - t_last_absorption)
                last_time_exit = t
                if full_cycles_marked_by_exit_events:
                    if first_time_in_cycle == -1:
                        first_time_in_cycle = t    # The time starting the first cycle is the first EXIT event from the absorption set
                    else:
                        n_cycles += 1

            # Check entry to A (i.e. state is outside A and next_state is inside A)
            if entry_to_absorption_set(absorption_set, state, next_state):
                # killing_times.append(t - t_last_exit)
                last_time_entry = t
                if not full_cycles_marked_by_exit_events:
                    if first_time_in_cycle == -1:
                        first_time_in_cycle = t  # The time starting the first cycle is the first ENTRY event to the absorption set
                    else:
                        n_cycles += 1
        last_updated_time = t

        # Convert the deques to arrays for easier manipulation
        exit_times = np.array(exit_times)
        exit_states = np.array(exit_states)

        # Define the first and last time on which the expected cycle time is computed
        last_time_in_cycle = last_time_exit if full_cycles_marked_by_exit_events else last_time_entry
        assert  n_cycles == 0 and last_time_in_cycle == first_time_in_cycle or \
                n_cycles  > 0 and last_time_in_cycle  > first_time_in_cycle, "The last time in the cycle must be larger than the last time in the cycle when at least one cycle was observed"

        # When no full cycle has been observed, the expected cycle time is computed on the basis of the last observed time (censored estimation)
        if n_cycles == 0:
            warnings.warn("No full return cycle to A (from exit to exit if the start state is inside A or from entry to entry if the start state is outside A) has been observed. "
                          "The expected cycle time is estimated using the censored observation of the last observed time step.")
            expected_cycle_time = t - first_time_in_cycle
        else:
            expected_cycle_time = (last_time_in_cycle - first_time_in_cycle ) / n_cycles

        # Exit state distribution estimated as a histogram
        # Single histogram on all the values altogether --> it's not so useful because there is a large set of empty bins near zero
        #_freq, hist_exit_state_bins = np.histogram(exit_states, bins=30)
        #hist_exit_state_probs = _freq / np.sum(_freq)
        # Note that we separate the exit states that are negative from the exit states that are positive, o.w. there is a large number of empty bins that reduces
        # resolution that we would like to have on the exit states distribution.
        # IMPORTANT: This ASSUMES that the problem is 1D!
        _ind_left = exit_states < float(list(absorption_set.boundary)[0])
        _ind_right = ~_ind_left
        _freq, _bins = np.array([]), np.array([])
        if sum(_ind_left) > 0:
            _freq_left, _bins_left = np.histogram(exit_states[_ind_left], bins=30)
            _freq = np.r_[_freq, _freq_left]
            _bins = np.r_[_bins, _bins_left]
        if sum(_ind_right) > 0:
            _freq_right, _bins_right = np.histogram(exit_states[~_ind_left], bins=30)
            _bins = np.r_[_bins, _bins_right]
            if len(_freq) > 0:
                _freq = np.r_[_freq, [0], _freq_right]
            else:
                _freq = np.r_[_freq_right]
        hist_exit_state_probs = _freq / np.sum(_freq)
        hist_exit_state_bins = _bins

        _bin_intervals = [(hist_exit_state_bins[i], hist_exit_state_bins[i+1]) for i, _ in enumerate(hist_exit_state_bins[:-1])]
        _midpoints = [0.5*(hist_exit_state_bins[i] + hist_exit_state_bins[i+1]) for i, _ in enumerate(hist_exit_state_bins[:-1])]
        dist_exit_state = pd.DataFrame({'x': _midpoints, 'p': hist_exit_state_probs}, index=_bin_intervals, columns=['x', 'p'])

        # Additional info to return that may be of interest
        info = dict({'last_updated_time': last_updated_time,
                     'exit_times': exit_times,
                     'exit_states': exit_states,
                     })

        return expected_cycle_time, n_cycles, dist_exit_state, info

    @measure_exec_time
    def _run_simulation_fv(self, dict_params_simul, dist_exit_state, sets_of_interest, store_trajectory=False, seed=None, verbose=False, verbose_period=1, plot=False):
        """
        Runs the Monte-Carlo simulation that is used to estimate the denominator of the FV estimator of the stationary probability,
        the expected return time to the absorption set A, E(T_A)

        Arguments:
        dict_params_simul: dict
            Simulation parameters containing at least the following entries:
            - 'N': # particles used in the Fleming-Viot particle system. It must be an integer larger than 1.
            - 'absorption_set': sympy.Set defining the absorption set A, whose touching makes a particle be reactivated to another particle.
            - 'max_nsteps': (opt) maximum number of (integer) steps to run the simulation for,
            if the normal condition for stopping the FV simulation never happens. Default: 1E7

        dist_exit_state: pandas data frame
            Estimate of the stationary exit state distribution from the absorption set A.
            The structure of this data frame is defined by the output of self._run_simulation_mc().

        sets_of_interest: list of sympy.Set
            List containing the sets of interest for the estimation of the stationary probability using Fleming-Viot.

        store_trajectory: bool
            Whether the observed trajectory for ALL particles should be stored in this object as attribute self.trajectory_fv,
            a list of pandas Series each indexed by particle ID (from 0, 1, ..., N-1),
            indexed by the integer time at which the respective particle is updated (starting at time 0),
            with as many elements as the number of steps taken by the respective particle.
            default: False

        seed: int
            Seed for random number generation (set and stored in the environment `self.env`).
            default: None

        verbose: bool
            Whether to show messages in the console about the simulation progress.
            default: False

        verbose_period: int (currently NOT used)
            Every how many time steps the information shown by `verbose` should be displayed.
            default: 1

        plot: bool (currently NOT used)
            Whether to show a dynamic plot of the trajectory as it is being generated.
            Only the last 30 time steps are shown (to avoid unbounded increase of execution time).
            default: False

        Return: dict
        An `info` dict containing the following pieces of information that may be useful:
        - 'last_updated_time': int giving the last time a particle was updated. It must be <= dict_params_simul['max_nsteps'].
        - 'absorption_times': array with the observed absorption times.
        - 'absorption_states': array with the observed absorption states at the respective absorption times.
        """
        #--- Auxiliary functions
        def initialize_phi(pstates, sets_of_interest):
            """
            Initializes the dictionary that will store the estimation of the occupation probability conditioned to non-absorption at each time where
            a change in the distribution of particles affects one of those occupation probabilities

            Arguments:
            pstates: array
                Array with the particle states at a given time.

            sets_of_interest: list of sympy.Set
                List of the sets of interest for the estimation of their stationary probability.

            Return: dict
            Dictionary indexed by each set of interest whose value is a data frame with two columns: 't' and 'Phi'.
            """
            dict_phi = dict()
            for C in sets_of_interest:
                dict_phi[C] = pd.DataFrame([[0, empirical_mean(pstates, C)]], columns=['t', 'Phi'])
            return dict_phi

        def empirical_mean(pstates, C):
            """
            Computes the proportion of particles, whose state is stored in the pstates array, that are at a state belonging to the given C set

            Arguments:
            pstates: array
                Array with the particle states at a given time.

            C: sympy.Set
                Set of interest, on which the proportion of particles present in that set is computed.
            """
            return np.mean([1 if C.contains(state) else 0 for state in pstates])

        def update_phi(dict_phi, idx_particle, time, state_prev):
            "Updates the Phi dictionary for every of its entries based as long as the current and previous state of the given particle affects its value"
            state_cur = pstates[idx_particle]
            for C in dict_phi.keys():
                # Check whether the previous state was in C and the new state is no longer in C or viceversa,
                # as these are the two conditions under which the value of Phi(C) could change at the current time.
                if  C.contains(state_prev) and ~C.contains(state_cur) or \
                   ~C.contains(state_prev) and  C.contains(state_cur):
                    phi_cur = dict_phi[C]['Phi'].iloc[-1]
                    phi_new = empirical_mean_update(phi_cur, C, state_prev, state_cur, N)
                    dict_phi[C] = pd.concat([dict_phi[C],
                                            pd.DataFrame([[time, phi_new]], index=[dict_phi[C].shape[0]], columns=['t', 'Phi'])],
                                            axis=0)

        def empirical_mean_update(mean_value, C, state_prev, state_cur, N):
            """
            Updates the proportion of N particles in the given set of interest C based on the change of just ONE particle from state_prev to state_cur
            ASSUMING that either state_prev is in C and state_cur is not in C or the other way round, i.e. the two states CANNOT be both in C or both NOT in C.

            Arguments:
            mean_value: float
                Proportion of particles (mean) to update.

            C: sympy.Set
                Set of interest on which the proportion of particles should be updated.

            state_prev: float
                Previous state of the particle that changed state.

            state_cur: float
                Current state of the particle that changed state.

            N: int
                Number of particles in the FV system.

            Return: float
            The updated mean, i.e. the proportion of particles in state C.
            """
            # The following means:
            # - if the current state of the particle is in C, the proportion of particles in C should increase by 1/N.
            # - if the previous state of the particle was in C, the proportion of particles in C should decrease by 1/N.
            # This gives a sensible update ONLY when the condition about state_prev and state_cur stated in the documentation of this method is satisfied.
            # Note that we do NOT assert that condition in order to avoid execution time. This condition is checked BEFORE calling this method (see function update_phi() above).
            return min( max(0, mean_value + (int(bool(C.contains(state_cur))) - int(bool(C.contains(state_prev)))) / N), 1 )

        def choose_particle(N):
            idx_particle = np.random.choice(N)
            return idx_particle

        def update_particle(idx_particle, time):
            # Before updating the environment, set it to the current state of the particle to update
            state = pstates[idx_particle]
            self.env.setState(state)
            # Update the environment which now represents the particle to update
            next_state, _, _, _ = self.env.step(0)
            update_particle_info(idx_particle, time, next_state)
            return state, next_state

        def update_particle_info(idx_particle, time, state):
            ptimes[idx_particle] = time
            pstates[idx_particle] = state
            if store_trajectory:
                ptimes_hist[idx_particle].append(time)
                pstates_hist[idx_particle].append(state)

        def check_and_manage_absorption(idx_particle, time, state, next_state, num_particles_absorbed_at_least_once):
            "Checks if the given particle has been absorbed and manage the absorption event in terms of absorption times, survival times, # particles absorbed"
            if entry_to_absorption_set(absorption_set, state, next_state):
                # Reactivate the particle to the position of one of the other N-1 particles chosen at random
                idx_reactivate = reactivate_particle(idx_particle)
                ptimes[idx_particle] = time
                pstates[idx_particle] = pstates[idx_reactivate]
                if store_trajectory:
                    ptimes_hist[idx_particle].append(time)
                    pstates_hist[idx_particle].append(pstates[idx_particle])

                # Store the absorption time and state (for information purposes, e.g. plotting / debugging)
                absorption_times.append(time)
                absorption_states.append(next_state)

                # Update the first absorption time of the absorbed particle, should this be its first absorption event
                if first_absorption_times[idx_particle] == 0.0:
                    first_absorption_times[idx_particle] = time
                    num_particles_absorbed_at_least_once += 1
                    # Store the absorption time which is actually a survival time used for the computation of the survival probability
                    survival_times.append(time)

            return num_particles_absorbed_at_least_once

        def reactivate_particle(idx_particle):
            # To avoid having to define an array with all the particle IDs without idx_particle, we just sample a value between 0 and N-2 and then add 1 if the sampled value is >= idx_particle
            # so that idx_particle is NEVER chosen
            idx_reactivate = np.random.choice(N - 1)
            if idx_reactivate >= idx_particle:
                idx_reactivate += 1
            assert 0 <= idx_reactivate <= N - 1, f"The particle chosen for reactivation ({idx_reactivate}) must be between 0 and {N-1}: {idx_reactivate}"
            assert idx_reactivate != idx_particle, f"The particle chosen for reactivation ({idx_reactivate}) must be different than the absorbed particle ({idx_particle})"
            return idx_reactivate
        #--- Auxiliary functions

        #--- Parse input parameters
        N = dict_params_simul.get('N')
        if not isinstance(N, int) or N <= 1:
            raise ValueError(f"The number of particles must be passed in dict_params_simul['N'] and must be an integer larger than 1: {N}")

        if not isinstance(dict_params_simul.get('absorption_set'), sympy.Set):
            raise ValueError(f"Parameter `dict_params_simul['absorption_set']` must be of type sympy.Set (e.g. `sympy.Union(sympy.Interval(-np.Inf, -0.1), sympy.Interval(0.1, +np.Inf))`): {dict_params_simul['absorption_set']}")
        absorption_set = dict_params_simul['absorption_set']

        max_nsteps = dict_params_simul.get('max_nsteps', 1E7)

        # Sets of interest on which the stationary probability is estimated
        if not isinstance(sets_of_interest, list):
            raise ValueError("Parameter `sets_of_interest` must be a list of sets of interest on which the stationary probability should be estimated")
        for i, set_of_interest in enumerate(sets_of_interest):
            if not isinstance(set_of_interest, sympy.Set):
                raise ValueError(f"Each set given in the `sets_of_interest` list must be of type `sympy.Set`: type is `{type(set_of_interest)}` for element {i} in the list")
            if sympy.Intersection(set_of_interest, absorption_set) != sympy.EmptySet():
                warnings.warn(f"One set of interest ({set_of_interest}) has a non-empty intersection with the absorption set ({absorption_set}):\n{sympy.Intersection(set_of_interest, absorption_set)}"
                              f"\nThe stationary probability of that set estimated by Fleming-Viot may be wrong.")

        # Seed
        if seed is not None:
            self.env.set_seed(seed)
        #--- Parse input parameters


        # Initialize objects to store the state and the update time for each particle
        ptimes = np.zeros(N, dtype=int)
        pstates = np.random.choice(dist_exit_state['x'], size=N, p=dist_exit_state['p'])

        # Initialize objects to store particle trajectories
        # Each particle stores their trajectory as a deque (for faster update of the particle positions when a particle is updated)
        ptimes_hist = []
        pstates_hist = []
        for p in range(N):
            ptimes_hist += [ deque([ ptimes[p] ])]
            pstates_hist += [ deque([ pstates[p] ]) ]

        # Information about particle absorption
        first_absorption_times = np.zeros(N)   # This piece of information is used to estimate the survival probability P(T>t)
        num_particles_absorbed_at_least_once = 0    # This piece of information is used to stop the FV simulation (when at least all particles have been absorbed at least once)
        absorption_times = deque()
        absorption_states = deque()
        survival_times = deque([0])

        done = False
        t = 0   # We initialize t at 0 because we have already set the particle positions at the initial time step (defined as t = 0) above
        dict_phi = initialize_phi(pstates, sets_of_interest=sets_of_interest)
        while not done:
            t += 1
            p = choose_particle(N)
            state, next_state = update_particle(p, t)
            num_particles_absorbed_at_least_once = check_and_manage_absorption(p, t, state, next_state, num_particles_absorbed_at_least_once)
            update_phi(dict_phi, p, t, state)

            done = num_particles_absorbed_at_least_once == N or t >= max_nsteps
        last_updated_particle = p
        last_updated_time = t

        assert t < max_nsteps and num_particles_absorbed_at_least_once == N or t == max_nsteps, \
            f"When the simulation did NOT stop by max. #steps = {max_nsteps}, {N} particles should have been absorbed at least once: {num_particles_absorbed_at_least_once} at t = {last_updated_time}"

        # Estimate the survival probability distribution
        assert len(survival_times)-1 == N, f"There must be exactly N={N} survival times measured: {len(survival_times)}"
        df_proba_surv = compute_survival_probability(survival_times)

        if store_trajectory:
            # Define a series that is used to check that at each time step exactly one particle was updated
            ptimes_all = np.zeros(last_updated_time+1)

            self.trajectories_fv = list()
            for p in range(N):
                if p != last_updated_particle:
                    # Fill the last update of the particle until the last simulation step so that all trajectories end at the same time
                    ptimes_hist[p].append(last_updated_time)
                    pstates_hist[p].append(pstates[p])
                self.trajectories_fv += [pd.Series(pstates_hist[p], index=ptimes_hist[p], name='x')]
                ptimes_all[ptimes_hist[p]] = ptimes_hist[p]
            assert all(np.diff(ptimes_all) == np.ones(len(ptimes_all) - 1)), "At each time step, exactly one particle must have been updated during the FV simulation"

        info = dict({'absorption_times': absorption_times,
                     'absorption_states': absorption_states,
                     'last_updated_time': last_updated_time,
                     })

        if plot and store_trajectory:
            colormap = cm.get_cmap("jet")
            ax = plt.figure().subplots(1, 1)
            for p in range(N):
                ax.step(self.trajectories_fv[p].index, self.trajectories_fv[p], where='post', color=colormap(p/N))
            ax.plot(info['absorption_times'], info['absorption_states'], "ro", markersize=3)
            ax.axhline(float(list(absorption_set.boundary)[0]), color="blue", linestyle="dashed")
            ax.axhline(float(list(absorption_set.boundary)[1]), color="blue", linestyle="dashed")
            ax.axhline(0.0, color="gray")
            ax.set_xlabel("t")
            ax.set_title(f"FV simulation on N={N} particles")

        return df_proba_surv, dict_phi, info


def entry_to_absorption_set(absorption_set, state, next_state):
    "Checks whether a process exits the absorption set"
    return not absorption_set.contains(state) and absorption_set.contains(next_state)


def exit_from_absorption_set(absorption_set, state, next_state):
    "Checks whether a process enters the absorption set"
    # QUESTION: Is using sympy.logic.boolalg.Boolean.as_set() and then the method absorption_set_logic.subs(state) be more general than sympy.Set?
    # Note however that sympy.logic.boolalg.Boolean is NOT available in the sympy-1.1.1 version I have installed in Python-3.6.4, but it is available in the latest sympy-1.14.0 version
    return absorption_set.contains(state) and not absorption_set.contains(next_state)


@measure_exec_time
#@jit
# Function taken from vu_frequencyprocesssimulation.py (shared by Alessandro Zocca in Apr-2025)
def simulation_mc(env, F0=0.0, t_end=10.0, dt=0.01, seed=None):
    """
    Simulates an Ornstein-Uhlenbeck process with state-dependent drift and optional random seed.

    Parameters:
    - env: environment defining the simulation dynamics.
    - F0: Initial value of the process.
    - t_end: End time of the simulation.
    - dt: Time step for the simulation.
    - mu: Long-term mean of the process.
    - sigma: Volatility parameter.
    - seed: Random seed for reproducibility.

    Returns:
    - F: Simulated OU process.
    """
    # Set a random seed if not provided
    if seed is not None:
        np.random.seed(seed)

    # Initialize parameters
    nsteps = int(t_end / dt)  # Total number of steps
    F = np.zeros(nsteps)  # Pre-allocate array for F
    F[0] = F0  # Start at F0
    t = np.linspace(0, t_end, nsteps)  # Time points

    env.setState(F0)
    for i in tqdm(range(nsteps)):
        F[i] = env.getState()
        env.step(0)     # 0 is the action... currently there is no special action to consider

    return t, F
