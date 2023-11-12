# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:21:54 2022

@author: Daniel Mastropietro
@description: Definition of functions and classes used for the simulation of discrete-time MDPs.
"""

import os
import sys
import copy
import warnings
from typing import Union

import numpy as np
import random       # For a random sample from a set
from matplotlib import pyplot as plt, cm

from Python.lib.agents.learners.episodic.discrete.fv import LeaFV
from Python.lib.agents.learners.episodic.discrete.td import LeaTDLambdaAdaptive
from Python.lib.environments.mountaincars import MountainCarDiscrete
from Python.lib.estimators import DEBUG_ESTIMATORS
from Python.lib.estimators.fv import initialize_phi, update_phi, estimate_expected_reward, estimate_stationary_probabilities
from Python.lib.simulators.fv import reactivate_particle
from Python.lib.simulators import DEBUG_TRAJECTORIES, choose_state_from_set, parse_simulation_parameters, show_messages

from Python.lib.utils.basic import find_signed_max_value, generate_datetime_string, get_current_datetime_as_string, is_integer, measure_exec_time
from Python.lib.utils.computing import compute_survival_probability, mape, rmse


class Simulator:
    """
    Simulator class that runs a Reinforcement Learning simulation on a discrete environment

    Arguments:
    env: Environment
        Environment that is assumed to have the following methods, getters and setters defined:
        - seed(): to set the random generator seed
        - reset(): to reset the environment's state
        - getNumStates()
        - getNumActions()
        - getInitialStateDistribution() --> returns an array that is a COPY of the initial state distribution
            (responsible for defining the initial state when running the simulation)
        - getState() --> returns the current state of the environment
        - getDimension()
        - getAllState()
        - getTerminalStates()
        - getInitialStateDistribution()
        - setInitialStateDistribution()

    agent: Agent
        Agent object that is responsible of performing the actions on the environment and learning from them.

    case: (opt) int
        Simulation case, which can be used to group simulations by specific settings of interest.
        default: 1

    replication: (opt) int
        Replication number associated to a simulation experiment. Normally one case may have several replications.
        default: 1

    seed: (opt) int
        Seed to use in the simulations as base seed (then the seed for each simulation is changed from this base seed).
        default: None, in which case a random seed is generated.

    log: (opt) bool
        Whether to create a log file where printed messages are sent to.
        The name of the file is created from the datetime of execution in the logsdir and has extension .log.
        default: False

    save: (opt) bool
        Whether to save the results to a file.
        The name of the file is created from the datetime of execution in the resultsdir and has extension .csv.
        default: False

    logsdir: (opt) string
        Name of the directory where the log file should be created if requested.
        default: None, in which case the log file is created in the directory from where execution is launched.

    resultsdir: (opt) string
        Name of the directory where the results file should be created if requested.
        default: None, in which case the results file is created in the directory from where execution is launched.

    debug: (opt) bool
        Whether to show messages useful for debugging.
        default: False
    """

    def __init__(self, env, agent, case=1, replication=1, seed=None, log=False, save=False, logsdir=None, resultsdir=None, debug=False):
#        if not isinstance(env, EnvironmentDiscrete):
#            raise TypeError("The environment must be of type {} from the {} module ({})" \
#                            .format(EnvironmentDiscrete.__name__, EnvironmentDiscrete.__module__, env.__class__))
        # TODO: (2020/04/12) Fix the following check on the type of agent as it does not work...
        # even though the class of agent is Python.lib.agents.GenericAgent
#        if not isinstance(agent, GenericAgent):
#            raise TypeError("The agent must be of type {} from the {} module ({})" \
#                            .format(GenericAgent.__name__, GenericAgent.__module__, agent.__class__))

        self.debug = debug
        self.log = log
        self.save = save
        self.case = case
        self.replication = replication

        # We use the class name + datetime as prefix for the output files
        dt_today_str = generate_datetime_string(prefix=self.__class__.__name__)
        if self.log:
            if logsdir is None:
                logsdir = os.getcwd()
            logsdir = os.path.abspath(logsdir)
            self.logfile = os.path.join(logsdir, generate_datetime_string(dt_str=dt_today_str, extension=".log"))
            self.fh_log = open(self.logfile, "w")
            self.stdout_sys = sys.stdout
            print("-----> File opened for log:\n{}".format(self.logfile))
            print("-----> If the process stops for some reason, finalize the file with simul.close().")
            # Redirect the console output to the log file
            sys.stdout = self.fh_log
        else:
            self.logfile = None
            self.fh_log = None
            self.stdout_sys = None

        if self.save:
            if resultsdir is None:
                resultsdir = os.getcwd()
            resultsdir = os.path.abspath(resultsdir)
            self.results_file = os.path.join(resultsdir, generate_datetime_string(dt_str=dt_today_str, extension=".csv"))
            self.fh_results = open(self.results_file, "w")
            print("-----> File opened for output with simulation results:\n{}".format(self.results_file))
            print("-----> If the process stops for some reason, finalize the file with simul.close().")
        else:
            self.results_file = None
            self.fh_results = None

        self.env = env
        self.agent = agent
        self.seed = seed

        # _isd_orig may store a copy of the Initial State Distribution of the environment in case we need to change it
        # at some point, so that we can restore it a some later point, e.g. when a simulation finishes.
        self._isd_orig = None
        self.reset()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the initial state distribution array in case it has been possibly changed
        # (e.g. when we want to define a specific initial state)
        if self._isd_orig is not None:
            self.env.setInitialStateDistribution(self._isd_orig)

    def reset(self, reset_episode=True, reset_value_functions=True):
        "Resets the simulator"
        self._isd_orig = None

        # Reset the learner to the first episode state
        self.agent.getLearner().reset(reset_episode=reset_episode, reset_value_functions=reset_value_functions)

    def close(self):
        if self.fh_log is not None:
            self.fh_log.close()
            sys.stdout = self.stdout_sys
            print("\n------> Log file created as:")
            print("------> {}".format(self.logfile))
        if self.fh_results is not None:
            self.fh_results.close()
            print("\n------> Output file with simulation results created as:")
            print("------> {}".format(self.results_file))

    def run(self, **kwargs):
        if isinstance(self.agent.getLearner(), LeaFV):
            return self._run_fv(**kwargs)
        else:
            return self._run_single(**kwargs)

    def _run_fv(self, seed=None, verbose=True, verbose_period=100, plot=False):
        """
        Runs all the simulations that are needed to learn differential value functions using the Fleming-Viot approach.

        These simulations include:
        - A simulation on a single Markov chain which is used to estimate E(T_A), the expected reabsorption to the set
        of uninteresting states, A.
        - A Fleming-Viot simulation on N particles which is used to estimate Phi(t, x) and P(T > t), which contribute
        to the numerator of the Fleming-Viot estimator of stationary state probabilities.

        Return: tuple
        Tuple with the following elements:
        - state_values: the estimated differential state value function V(s).
        - action_values: the estimated action value function Q(s,a).
        - state_counts: visit frequency of states under the single Markov chain simulation that estimates E(T_A).
        - probas_stationary: dictionary with the estimated stationary probability for each state of interest.
        - expected_reward: the estimated expected reward.
        - n_events_et: number of events observed during the simulation of the single Markov chain used to estimate E(T_A) and P(T>t).
        - n_events_fv: number of events observed during the FV simulation that estimates Phi(t).
        """
        #--- Parse input parameters ---
        dict_params_simul = dict({  'N': self.agent.getLearner().getNumParticles(),
                                    'T': self.agent.getLearner().getNumTimeStepsForExpectation(),
                                    'absorption_set': self.agent.getLearner().getAbsorptionSet(),
                                    'activation_set': self.agent.getLearner().getActivationSet(),
                                    'seed': seed})
        dict_params_info = dict({'verbose': verbose,
                                 'verbose_period': verbose_period,
                                 't_learn': 0}) # This indexes the learning epoch of the optimal policy
        #--- Parse input parameters ---

        if 'N' not in dict_params_simul.keys():
            raise ValueError("The dictionary of simulation parameters must contain the key 'N' with the number of Fleming-Viot particles to use in the simulation")

        # Create the particles as copies of the main environment
        envs = [self.env if i == 0 else copy.deepcopy(self.env) for i in range(dict_params_simul['N'])]

        state_values, action_values, state_counts_from_single_markov_chain, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, \
            time_last_absorption, max_survival_time, n_events_et, n_events_fv = \
                self._estimate_state_value_function_and_expected_reward_fv( envs, dict_params_simul, dict_params_info,
                                                                            probas_stationary_start_state=self.agent.getLearner().getProbasStationaryStartState())

        return state_values, action_values, state_counts_from_single_markov_chain, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, n_events_et, n_events_fv

    def _estimate_state_value_function_and_expected_reward_fv(self, envs, dict_params_simul, dict_params_info,
                                                              probas_stationary_start_state: dict=None):
        """
        Estimates the differential state values and action values, the stationary state probabilities,
        and the expected reward (a.k.a. long-run average reward) using the Fleming-Viot approach.

        Arguments:
        envs: List
            List of environments used to run the FV process.

        dict_params_simul: dict
            Dictionary containing simulation and estimation parameters.
            The dictionary should contain the keys described in function `parse_simulation_parameters()`.

        dict_params_info: dict
            Dictionary containing information to display or parameters to deal with the information to display.
            Accepted keys are:
            - verbose: whether to be verbose during the simulation.
            - verbose_period: the number of iterations (of what?) at which to be verbose.
            - t_learn: the number of learning steps, when FV is used in the context of FVRL, i.e. to learn an optimum policy
            This is ONLY used for informational purposes, i.e. to show which stage of the policy learning we are at.

        probas_stationary_start_state: dict
            Stationary distribution that should be used to select the start state of each FV particle.
            default: None, in which case a uniform random distribution is used.

        Return: tuple
        Tuple with the following elements:
        - state_values: the estimated differential state value function V(s).
        - action_values: the estimated action value function Q(s,a).
        - state_counts: visit frequency of states under the single Markov chain simulation that estimates E(T_A).
        - probas_stationary: dictionary with the estimated stationary probability for each state of interest.
        - expected_reward: the estimated expected reward.
        - expected_absorption_time: estimated expected absorption time E(T_A) used in the denominator of the FV estimator
        of the stationary state probabilities.
        - n_cycles: number of cycles observed to estimate E(T_A).
        - time_last_absorption: discrete time of the last absorption observed, used in the estimation of E(T_A).
        - max_survival_time: maximum survival time observed when estimating P(T>t).
        - n_events_et: number of events observed during the simulation of the single Markov chain used to estimate E(T_A) and P(T>t).
        - n_events_fv: number of events observed during the FV simulation that estimates Phi(t).
        """

        # -- Auxiliary functions
        is_estimation_of_denominator_unreliable = lambda: n_cycles_absorption_used < dict_params_simul['min_num_cycles_for_expectations']
        # -- Auxiliary functions

        # -- Parse input parameters
        dict_params_simul = parse_simulation_parameters(dict_params_simul, envs[0])

        # Set the simulation seed
        # Note that, although the seed is set by the environment (via its set_seed() method which is called by run_simulation_mc() below)
        # we also need to set the seed here because, before calling run_simulation_mc(), we choose below the initial state
        # on which the simulation starts randomly, among all possible states in the absorption set.
        np.random.seed(dict_params_simul['seed'])
        # -- Parse input parameters

        # -- Step 1: Simulate a single Markov chain to estimate the expected cycle time of return to A, E(T_A)
        start_state_boundary_A = choose_state_from_set(dict_params_simul['absorption_set'], probas_stationary_start_state)
        state_values, action_values, state_counts, _, _, learning_info = \
            self._run_single(dict_params_simul['T'],                    # Number of episodes to run (it is still NOT clear how to define this, as we want to run the simulation until T number of steps are observed IN TOTAL, so how do we define the number of episodes??)
                            max_time_steps=dict_params_simul['T'],      # Max simulation time per episode (it is still NOT clear how to define this, as we want to run the simulation until T number of steps are observed IN TOTAL, as opposed to "per episode")
                            start=start_state_boundary_A,
                            seed=dict_params_simul['seed'],
                            set_cycle=set.union(dict_params_simul['absorption_set'], self.env.getTerminalStates()), #dict_params_simul['absorption_set'],
                            verbose=dict_params_info.get('verbose', False),
                            verbose_period=dict_params_info.get('verbose_period', 1))
        n_events_et = learning_info['t']
        n_cycles_absorption_used = learning_info['num_cycles']
        time_last_absorption = learning_info['last_cycle_entrance_time']

        # -- Step 2: Simulate N particles with FLeming-Viot to compute the empirical distribution and estimate the stationary probabilities, and from them the expected reward
        # BUT do this ONLY when the estimation of E(T_A) is reliable... otherwise, set the stationary probabilities and expected reward to NaN.
        if is_estimation_of_denominator_unreliable():
            # FV is not run because the simulation that is used to estimate E(T_A) would not generate a reliable estimation
            # (most likely it would UNDERESTIMATE E(T_A) making the probabilities be OVERESTIMATED)
            print(
                "Fleming-Viot process is NOT run because the estimation of the expected absorption time E(T_A) cannot be reliably performed"
                " because of an insufficient number of observed cycles after the burn-in period of {} time steps: {} < {}" \
                "\nThe estimated stationary probabilities and estimated expected reward will be set to NaN." \
                .format(dict_params_simul['burnin_time_steps'], n_cycles_absorption_used,
                        dict_params_simul['min_num_cycles_for_expectations']))
            expected_reward = np.nan
            probas_stationary = dict()
            expected_absorption_time = np.nan
            max_survival_time = np.nan
            n_events_et = n_events_et
            n_events_fv = 0
        else:
            # Perform the Fleming-Viot simulation, as the estimator of the denominator in the FV estimator is reliable
            N = len(envs)
            assert N > 1, "The simulation system has more than one particle in Fleming-Viot mode ({})".format(N)
            expected_absorption_time = learning_info['expected_cycle_time']
            t, state_values, action_values, phi, df_proba_surv, expected_absorption_time, max_survival_time = \
                self._run_simulation_fv(dict_params_info.get('t_learn', 0), envs,
                                        dict_params_simul['absorption_set'],
                                        dict_params_simul['activation_set'],
                                        dist_proba_for_start_state=probas_stationary_start_state,
                                        expected_absorption_time=expected_absorption_time,
                                        verbose=dict_params_info.get('verbose', False),
                                        verbose_period=dict_params_info.get('verbose_period', 1),
                                        plot=DEBUG_TRAJECTORIES)
            n_events_fv = t

            probas_stationary, integrals = estimate_stationary_probabilities(phi, df_proba_surv,
                                                                             expected_absorption_time,
                                                                             uniform_jump_rate=N)
            expected_reward = estimate_expected_reward(envs[0], probas_stationary)

            if DEBUG_ESTIMATORS or show_messages(dict_params_info.get('verbose', False),
                                                 dict_params_info.get('verbose_period', False),
                                                 dict_params_info.get('t_learn', 0)):
                print("\n*** RESULTS OF FLEMING-VIOT SIMULATION ***")
                # import pandas as pd
                # max_rows = pd.get_option('display.max_rows')
                # pd.set_option('display.max_rows', None)
                # print("Phi(t):\n{}".format(phi))
                # pd.set_option('display.max_rows', max_rows)
                print("Integrals: {}".format(integrals))
                print("Expected reabsorption time (on {} cycles): {}".format(learning_info['num_cycles'], expected_absorption_time))
                print("Stationary probabilities: {}".format(probas_stationary))
                print("Expected reward = {}".format(expected_reward))

        return state_values, action_values, state_counts, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, \
               time_last_absorption, max_survival_time, n_events_et, n_events_fv

    @measure_exec_time
    def _run_simulation_fv( self, t_learn, envs, absorption_set: set, activation_set: set,
                            dist_proba_for_start_state: dict = None,
                            expected_absorption_time=None, expected_exit_time=None,
                            verbose=False, verbose_period=1, plot=False):
        """
        Runs the Fleming-Viot simulation of the particle system and estimates the different pieces of information that
        are part of the Fleming-Viot estimator of the average reward:
        - Phi(t, x): the conditional probability conditional on non absorption or survival.
        - P(T > t): the survival (non-absorption) probability.
        - E(T_A): expected reabsorption probability to set A, only computed by this function when its parameter is None.
        Otherwise, the input parameter value is returned.
        In addition, the differential value functions V(s) and Q(s,a) are also estimated, from possibly an initial estimation
        obtained from the simulation of a single Markov chain (run by e.g. self._run_single()).

        Arguments:
        t_learn: int
            The policy learning time step to which the simulation will contribute.
            Only used for informative purposes on where we are at when learning the policy or to decide whether to show information in the log.

        envs: list
            List of environments used to run the FV process.

        absorption_set: set
            Set with the entrance states to the set of uninteresting states A.
            This set is needed to measure the killing times used to estimate the survival probability P(T > t).

        activation_set: set
            Set with the entrance states to the complement of the set of uninteresting states A.
            This set is needed to define the start state of the Fleming-Viot simulation.

        dist_proba_for_start_state: dict
            Probability distribution from which the start state for each FV particle is chosen.
            The dictionary can contain more states than those in the activation set (that are also part of the boundary
            of the complement of the set of absorbed states A), but it MUST contain all the states that are at the boundary
            of Ac.
            The actual distribution to use here is the entrance distribution to Ac, but a reasonable approximation
            (that beats the uniformly random selection of the start state is the stationary distribution of the entrance states.
            default: None, in which case a uniform random distribution is used.

        expected_absorption_time: (opt) positive float
            Expected absorption cycle time E(T_A) used to estimate the stationary state probabilities.
            If None, it is estimated by this function, after observing the killing times on the N particles used in the
            FV simulation.
            default: None

        expected_exit_time: (opt) positive float
            Expected exit time from the absorption set, which is needed when expected_absorption_time = None, as this
            value will be used to estimate it as expected_exit_time + "expected killing time" computed as the average
            of the survival times observed during the N-particle FV simulation from where df_proba_surv is estimated.
            This value should be provided when expected_absorption_time is None. If the latter is provided, the value
            is ignored.
            default: None

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: tuple
        Tuple with the following elements:
        - t: the last (discrete) time step of the simulation process.
        - state_values: state value function V(s) estimated by the process.
        - action_values: action value function Q(s,a) estimated by the process.
        - dict_phi: dictionary of lists with the empirical distribution of the states of interest (e.g. where a non-zero rewards occurs)
        which are an estimate of the probability of those states conditional to survival (not absorption).
        - df_proba_surv: data frame with the estimated survival probability P(T > t).
        - expected_absorption_time: the expected absorption time, either computed by this function or passed as input parameter.
        - max_survival_time: maximum survival time observed during the simulation that estimated P(T>t). This value is
        obtained from the last row of the `df_proba_surv` data frame that is either given as input parameter or estimated
        by this function.
        """
        #------------------------------- Auxiliary functions ----------------------------------#
        def reactivate_particle_internal(idx_particle):
            "Internal function that reactivates a particle until a valid reactivation is obtained (i.e. a particle whose state is neither in the absorbed set of states nor at a terminal state"
            # TODO: (2023/09/06) Try to implement a more efficient way (i.e. without looping) to reactivate the particle to one of the other particles that are NOT at an absorption set nor at a terminal state
            done_reactivate = False
            state = envs[idx_particle].getState()
            if DEBUG_TRAJECTORIES:
                flags_particle_at_terminal_state = [1 if envs[p].getState() in self.env.getTerminalStates() else 0 for p in range(len(envs))]
                print(f"[reactivate_particle_internal] % particles at terminal states: {np.mean(flags_particle_at_terminal_state)*100}% ({np.sum(flags_particle_at_terminal_state)} out of {len(envs)})")
            new_state = None
            while not done_reactivate:
                idx_reactivate = reactivate_particle(envs, idx_particle, 0, absorption_number=absorption_number)
                    ## (2023/01/05) the third parameter is dummy when we do NOT use method = ReactivateMethod.VALUE_FUNCTION to reactivate the particle inside function reactivate_particle().
                if envs[idx_particle].getState() not in absorption_set: #set.union(absorption_set, self.env.getTerminalStates()):
                    done_reactivate = True
                    new_state = envs[idx_particle].getState()
                    if DEBUG_TRAJECTORIES:
                        print("*** Particle {} ABSORBED at state={} and REACTIVATED to particle {} at state {}" \
                              .format(idx_particle, state, idx_reactivate, new_state))

            return new_state
        #------------------------------- Auxiliary functions ----------------------------------#


        # ---------------------------- Check input parameters ---------------------------------#
        # -- Absorption and activation sets
        # Class
        if not isinstance(absorption_set, set):
            raise ValueError("Parameter `absorption_set` must be a set: {}".format(absorption_set))
        if not isinstance(activation_set, set):
            raise ValueError("Parameter `activation_set` must be a set: {}".format(activation_set))
        if len(activation_set) == 0:
            raise ValueError("Parameter `activation_set` must have at least one element")

        if expected_absorption_time is None and expected_exit_time is None:
            raise ValueError("Parameter `expected_exit_time` must be provided when `expected_absorption_time` is None")
        # ---------------------------- Check input parameters ---------------------------------#

        N = len(envs)
        policy = self.agent.getPolicy()  # Used to define the next action and next state
        learner = self.agent.getLearner()  # Used to learn (or keep learning) the value functions

        # Set the start state of each environment/particle to an activation state, as this is a requirement
        # for the empirical distribution Phi(t).
        for i, env in enumerate(envs):
            start_state = choose_state_from_set(activation_set, dist_proba_for_start_state)
            env.setState(start_state)

        # Event times: the first event time is 0
        event_times = [0]

        # Phi(t, x): Empirical probability of the states of interest (x)
        # at each time t when a variation in Phi(t, x) is observed.
        dict_phi = initialize_phi(envs, self.agent, t=event_times[0])

        # Initialize the list of observed survival times to be filled during the simulation below
        survival_times = [0]
        idx_reactivate = None  # This is only needed when we want to plot a vertical line in the particle evolution plot with the color of the particle to which an absorbed particle is reactivated
        has_particle_been_absorbed_once = [False] * N  # List that keeps track of whether each particle has been absorbed once
        # so that we can end the simulation when all particles have been absorbed
        # when the survival probability is estimated by this function.

        if DEBUG_ESTIMATORS:
            print("[DEBUG] @{}".format(get_current_datetime_as_string()))
            print("[DEBUG] State value function at start of simulation:\n\t{}".format(learner.getV().getValues()))

        idx_particle = -1
        done = False
        t = 0
        while not done:
            t += 1

            # We count the absorption number, an integer between 0 and N-2 which is used to deterministically choose
            # the reactivation particle (if reactivation is via ReactivateMethod.ROBINS), in order to save time by not having to generate a uniform random number.
            # Note that the range from 0 to N-2 allows us to choose one of the N-1 particles to which the absorbed particle
            # can be reactivated into.
            absorption_number = t % (N - 1)

            event_times += [t]

            # Select the particle to move uniformly at random
            idx_particle = np.random.choice(N) # If choosing them in order, use: `(idx_particle + 1) % N`

            # Get the current state of the selected particle because that's the particle whose state is going to (possibly) change
            state = envs[idx_particle].getState()

            # Check if terminal state, in which case we "restart" the process,
            # i.e. we start the particle where we had started it the first time, i.e. at a randomly chosen state in the activation set
            if state in self.env.getTerminalStates():
                next_state = choose_state_from_set(activation_set, dist_proba_for_start_state)
                envs[idx_particle].setState(next_state)
                if DEBUG_TRAJECTORIES:
                    print("___ Particle {} in terminal state REINITIALIZED to activation set at state={}".format(idx_particle, next_state))

                # Update the flag stating whether an episode is completed (as this is used below)
                # Most likely this will be always False, unless terminal states are a subset of the set of activation states, which is not a common scenario.
                done_episode = next_state in self.env.getTerminalStates()

                # After this repositioning of the particle we do NOT update the particle rightaway, but we wait for the next time step that the particle is picked.
                # The reason for doing this is that we must count this repositioning of a terminated particle to a non-terminal state as ONE time step
                # because we have updated Phi(t,x), therefore the t value at which Phi changed must be a time step that is counted as such.
            else:
                # Step on the selected particle
                action = policy.choose_action(state)
                next_state, reward, done_episode, info = envs[idx_particle].step(action)

                # Learn: i.e. update the value function (stored in the learner) with the new observation
                # NOTE that learning ONLY happens when the particle is NOT reactivated.
                # This makes total sense because reactivation sets the next state of the particle to a state
                # that is normally not reachable by the underlying Markov chain on which the FV process is built.
                learner.learn(t, state, action, next_state, reward, done, info)

            if done_episode or next_state in absorption_set:
                # The particle reached a terminal state or has been absorbed.
                # Note that terminal states are considered as an absorption for the estimation of the survival probability P(T>t)
                # because o.w. it would be difficult to think of the survival time properly, because after arriving to a terminal state
                # the process starts again (from an activation state)... so we are now at "square one", i.e. at a state from the set where the particle started
                # and it seems unnatural that the particle "suddenly" appears there from the terminal state, because... how do we make
                # the time left to absorption compatible with the fact that the particle was previously potentially very far away
                # from that activation set (i.e. at a terminal state)...? as suddenly it may require very little time to be absorbed
                # (because the states in the activation set are close to the absorption set of states).
                #*******
                # IMPORTANT: If we consider the terminal states as part of the absorption set here (i.e. in the Fleming-Viot simulation)
                # we must ALSO consider the terminal states as part of the set of uninteresting states A when estimating E(T_A)
                # using the simulation of a single Markov chain (o.w. the two estimates could be inconsistent in terms of
                # generating a proper FV estimator of the stationary state probabilities). In order to do this change in
                # the estimation of E(T_A) we should properly set the set_cycle parameter when calling the Simulator._run_single() method
                # that estimates E(T_A) by including the set of terminal states as part of the set_cycle parameter.
                #*******
                # => If absorbed, reactivate it to any of the other particles
                # => Add the time to absorption to the set of times used to estimate the survival probability P(T>t)
                if not has_particle_been_absorbed_once[idx_particle]:
                    survival_times += [t]  # Note that we store the ABSOLUTE time because at its first absorption, the particle started at 0, so this is correct.
                    ## IMPORTANT: By construction the survival times are ALREADY SORTED in the list, since we only add
                    ## the FIRST absorption time for each particle.

                    # Mark the particle as "absorbed once" so that we don't use any other absorption time from this
                    # particle to estimate the survival probability, because the absorption times coming after the first
                    # absorption time should NOT contribute to the survival probability, because their initial state
                    # is NOT the correct one --i.e. it is not a state in the activation set
                    # (because the particle has been reactivated to any other state)
                    # which is a requirement for the estimation of the survival probability distribution.
                    has_particle_been_absorbed_once[idx_particle] = True
                    if False:
                        print("Survival times observed so far: {}".format(sum(has_particle_been_absorbed_once)))
                        print(survival_times)

                # Reactivate the particle only when the particle has been STRICTLY absorbed
                # (i.e. NOT when the particle reached a terminal state).
                # This makes sense because if the particle reached a terminal state, then it should stay there until
                # the next time this particle is picked to be updated.
                if not done_episode:
                    next_state = reactivate_particle_internal(idx_particle)
                    #print(f"--> Reactivated particle {idx_particle} to state {next_state}")
                    assert next_state not in absorption_set, \
                        f"The state of a reactivated particle must NOT be a state in the absorption set ({next_state})"

            # Update Phi based on the new state of the changed (and possibly also reactivated) particle
            dict_phi = update_phi(envs[0], len(envs), t, dict_phi, state, next_state)

            if DEBUG_TRAJECTORIES:
                print("P={}, t={}: state={}, action={} -> state={}, reward={}" \
                      .format(idx_particle, t, state, action, next_state, reward),
                      end="\n")

            idx_reactivate = None

            done = sum(has_particle_been_absorbed_once) == N

        # DONE
        if show_messages(verbose, verbose_period, t_learn):
            print(
                "==> agent ENDS at discrete time t={}, compared to maximum observed time for P(T>t) = {:.1f} at state {} coming from state = {}, action = {}, reward = {})" \
                    .format(t, df_proba_surv is not None and df_proba_surv['t'].iloc[-1] or survival_times[-1],
                            envs[idx_particle].getState(), state, action, reward))

        # Compute the stationary probability of each state x in Phi(t, x) using Phi(t, x), P(T>t) and E(T_A)
        df_proba_surv = compute_survival_probability(survival_times)
        if expected_absorption_time is None:
            expected_absorption_time = expected_exit_time + np.mean(survival_times)
        max_survival_time = df_proba_surv['t'].iloc[-1]

        if DEBUG_ESTIMATORS:
            import pandas as pd
            max_rows = pd.get_option('display.max_rows')
            pd.set_option('display.max_rows', None)
            print("Survival probability:\n{}".format(df_proba_surv))
            print("Phi:\n{}".format(dict_phi))
            pd.set_option('display.max_rows', max_rows)

        return t, learner.getV().getValues(), learner.getQ().getValues(), dict_phi, df_proba_surv, expected_absorption_time, max_survival_time

    def _run_single(self, nepisodes, max_time_steps=+np.Inf, start=None, seed=None, compute_rmse=False, weights_rmse=None,
                    state_observe=None, set_cycle=None,
                    verbose=False, verbose_period=1, verbose_convergence=False,
                    plot=False, colormap="seismic", pause=0):
        # TODO: (2020/04/11) Convert the plotting parameters to a dictionary named plot_options or similar.
        # The goal is to group OPTIONAL parameters by their function/concept.
        """
        Runs an episodic Reinforcement Learning experiment on a single Markov chain.

        Parameters:
        nepisodes: int
            Length of the experiment: number of episodes to run.

        max_time_steps: int
            Maximum number of steps to run each episode for.
            default: np.Inf

        start: (opt) None or int
            Index in the set of states defined by the environment corresponding to the starting state.
            default: None

        seed: (opt) None or float
            Seed to use for the random number generator for the simulation.
            If None, the seed is NOT set.
            If 0, the seed stored in the object is used. I don't know when this value would be useful, but certainly
            NOT when running different experiments on the same environment and parameter setup, because in that case
            all experiments would start with the same seed!
            default: None

        compute_rmse: bool, optional
            Whether to compute the RMSE of the estimated value function over all states.
            Useful to analyze rate of convergence of the estimates when the true value function is known.

        weights_rmse: bool or array, optional
            Whether to use weights when computing the RMSE or the weights to use.
            default: None

        state_observe: int, optional
            A state index whose RMSE should be observed as the episode progresses.
            Only used when compute_rmse = True
            default: None

        set_cycle: set
            Set of states whose entrance from the complement set defines a cycle.
            Note that the set should include ALL the states, NOT only the boundary states through which the system can enter the set.
            The reason is that the set is used to determine which states are tracked for their visit frequency for the computation
            of their stationary probability using renewal theory.
            default: None

        verbose: bool, optional
            Whether to show the experiment that is being run and the episodes for each experiment.
            default: False

        verbose_period: int, optional
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        verbose_convergence: bool, optional
            Whether to monitor convergence by showing the change in value function estimate at each new episode
            w.r.t. the estimate in the previous episode.
            If True, messages are shown for EVERY episode with information about:
            - mean|V| previous episode
            - mean|V| current episode
            - mean|delta(V)|
            - mean|delta_rel(V)|, where delta_rel(V) is computed by dividing delta(V)(s) by |V(s)| if not 0 or mean|V| if V(s) = 0
            default: False

        plot: bool, optional
            Whether to generate plots showing the evolution of the value function estimates.

        colormap: str, optional
            Name of the colormap to use in the generation of the animated plots
            showing the evolution of the value function estimates.
            It must be a valid colormap among those available in the matplotlib.cm module.

        pause: float, optional
            Number of seconds to wait before updating the plot that shows the evalution
            of the value function estimates.

        Returns: tuple
        Tuple containing the following elements:
        - state value function estimate for each state at the end of the last episode (`nepisodes`).
        - number of visits to each state at the end of the last episode.
        - RMSE (when `compute_rmse` is not None), an array of length `nepisodes` containing the
        Root Mean Square Error after each episode, of the estimated value function averaged
        over all states. Otherwise, None.
        - MAPE (when `compute_rmse` is not None): same as RMSE but with the Mean Absolute Percent Error information.
        - a dictionary containing additional relevant information, as follows:
            - 'alphas_at_episode_end': the value of the learning parameter `alpha` for each state
            at the end of the last episode run.
            - 'alphas_by_episode': (average) learning parameter `alpha` by episode
            (averaged over visited states in each episode)
            - a set of estimation statistics such as the average absolute value function or the relative
            change of the value function between episodes. A sample of these summary statistics are:
                - 'V_abs_mean': array with length nepisodes+1 containing mean|V| over all states at each episode.
                - 'V_abs_median': array with length nepisodes+1 containing median|V| over all states at each episode.
                - 'deltaV_abs_mean': array with length nepisodes+1 containing mean|delta(V)| over all states at each episode.
                - 'deltaV_abs_median': array with length nepisodes+1 containing median|delta(V)| over all states at each episode.
                - 'deltaV_rel_abs_mean': array with length nepisodes+1 containing mean|delta_rel(V)| over all states at each episode.
                - 'deltaV_rel_abs_median': array with length nepisodes+1 containing median|delta_rel(V)| over all states at each episode.
            - when parameter set_cycle is not None, two pieces of information that can be used to compute the stationary probability of states
            using renewal theory:
                - the number of cycles observed.
                - the time at which the process completed the last cycle.
                - the expected cycle time, i.e. the average cycle time where a cycle is defined by entering the cycle set after its latest exit.
                - an array with the visit count of all states.
        """
        # Auxiliary functions
        entered_set_cycle = lambda s, ns: s not in set_cycle and ns in set_cycle

        # Reset the simulator (i.e. prepare it for a fresh new simulation with all learning memory erased)
        self.reset()

        #--- Parse input parameters
        # TODO: (2023/10/21) Now that the classes defining gridworld environments in environments/gridworlds.py accept the initial state distribution as parameter, consider passing that information when defining the environment so that we don't need to do this setup here which is a little cumbersome...
        # Define initial state
        nS = self.env.getNumStates()
        if start is not None:
            if not (is_integer(start) and 0 <= start and start < nS):
                warnings.warn("The `start` parameter ({}, type={}) must be an integer number between 0 and {}.\n" \
                              "A start state will be selected based on the initial state distribution of the environment." \
                              .format(start, type(start), nS-1))
            else:
                # Change the initial state distribution of the environment so that
                # the environment resets to start at the given 'start' state.
                self._isd_orig = self.env.getInitialStateDistribution()
                isd = np.zeros(nS)
                isd[start] = 1.0
                self.env.setInitialStateDistribution(isd)

        # Set the weights to be used to compute the RMSE and MAPE based on the weights_rmse value
        # Only when weights_rmse = True are the weights NOT set definitely here, as they are set at every episode
        # as the state count at the end of the episode.
        # Note that in all other cases, we set weights_rmse = None so that the logic to define the weights at the end
        # of each episode is much much simpler: we just check if weights_rmse is not None to know if we need to set
        # the weights as the state counts in the episode
        if weights_rmse is not None and not isinstance(weights_rmse, bool):
            # The user provided an array with the weights to use when computing the RMSE and MAPE
            assert isinstance(weights_rmse, np.ndarray), "weights_rmse is a numpy array ({})".format(type(weights_rmse))
            assert len(weights_rmse) == self.env.getNumStates(), \
                "The length of the weights_rmse array ({}) is the same as the number of states in the environment ({})" \
                    .format(len(weights_rmse), self.env.getNumStates())
            weights = weights_rmse
            weights_rmse = None
        elif weights_rmse is None or weights_rmse == False:
            # The user does NOT want to use weights when computing the RMSE and MAPE
            weights = None
            weights_rmse = None
        else:
            # The user wants to use the state counts when computing the RMSE and MAPE
            # => We set the weights to None so that the RMSE and MAPE computed at the very beginning
            # (using the initial state value function before the simulation starts) do not use any weights
            # as at that moment the state count is 0 for all states and using the state counts as weights
            # would give RMSE = MAPE = 0.0 at the very beginning and we don't want that.
            assert weights_rmse == True
            weights = None

        # Check the state to observe
        if state_observe is not None:
            if not compute_rmse:
                warnings.warn("The `state_observe` parameter is not None, but `compute_rmse = False`.\n" \
                              "A state can only be observed when `compute_rmse = True`. The state to observe will be ignored.")
                state_observe = None
            elif not (is_integer(state_observe) and 0 <= state_observe and state_observe < nS):
                warnings.warn("The `state_observe` parameter ({}, type={}) must be an integer number between 0 and {}.\n" \
                              "The state whose index falls in the middle of the state space will be observed." \
                              .format(state_observe, type(state_observe), nS-1))
                state_observe = int(nS/2)

        # Setup the information needed when cycles are used to estimate the stationary distribution of states using renewal theory
        if set_cycle is not None:
            cycle_times = []  # Note that the first cycle time will include a time that may not be a cycle time because the cycle may have not initiated at the sytem's start state. So the first cycle time will be considered a delay time.
            num_cycles = 0
            last_cycle_entrance_time = 0.0       # We set the last cycle time (i.e. the moment when the system enters the cycle set) to 0 (even if it is unknown) so that we can easily compute the FIRST cycle time below as "t - last_cycle_entrance_time"
            expected_cycle_time = 0.0  # We set the expected cycle time so that we can compute the expected cycle time recursively

            # Array of state counts WITHIN cycles, i.e. the count is increased ONLY when the state is visited within a TRUE cycle
            # (not during the first "cycle" which may be degenerate)
            # This can be used to estimate the stationary probability of the states using renewal theory
            arr_state_counts_within_cycles = np.zeros(self.env.getNumStates(), dtype=int)

        # Plotting setup
        if plot:
            fig_V = plt.figure()
            colors = cm.get_cmap(colormap, lut=nepisodes)
            if self.env.getDimension() == 1:
                # Plot the true state value function (to have it as a reference already
                plt.plot(self.env.getAllStates(), self.env.getV(), '.-', color="blue")
                if state_observe is not None:
                    fig_RMSE_state = plt.figure()

        # Define the policy and the learner
        policy = self.agent.getPolicy()
        learner = self.agent.getLearner()

        # Environment seed
        if seed is not None:
            if seed != 0:
                self.env.seed(seed)
            else:
                # Set the seed to the one stored in the object, which was set when the object was constructed
                self.env.seed(self.seed)

        # Store initial values used in the analysis of all the episodes run
        V_state_observe, RMSE, MAPE, ntimes_rmse_inside_ci95 = self.initialize_run_with_learner_status(nepisodes, learner, compute_rmse, weights, state_observe)

        # Initial state value function
        V = learner.getV().getValues()
        if verbose:
            print("Value function at start of experiment: {}".format(V))

        # Average state values and average change of V by episode for plotting purposes
        V_abs_mean = np.nan*np.ones(nepisodes+1)            # mean|V|
        V_abs_mean_weighted = np.nan*np.ones(nepisodes+1)   # mean|V| weighted by the state count
        V_abs_min = np.nan*np.ones(nepisodes+1)             # To evaluate the smallest divisor when computing the relative change in V
        deltaV_abs_mean = np.nan*np.ones(nepisodes+1)
        deltaV_max_signed = np.nan*np.ones(nepisodes+1)     # Maximum change with sign! To evaluate if the maximum change is oscillating around 0 suggesting convergence
        deltaV_rel_abs_mean = np.nan*np.ones(nepisodes+1)
        deltaV_rel_abs_mean_weighted = np.nan*np.ones(nepisodes+1)
        deltaV_rel_max_signed = np.nan*np.ones(nepisodes+1) # Maximum relative change with sign! To evaluate if the maximum change is oscillating around 0 suggesting convergence
        deltaV_rel_abs_max = np.nan*np.ones(nepisodes+1)    # To know the largest relative change observed in an episode
        deltaV_rel_abs_max_weighted = np.nan*np.ones(nepisodes+1)
        V_abs_median = np.nan*np.ones(nepisodes+1)
        deltaV_abs_median = np.nan*np.ones(nepisodes+1)
        deltaV_rel_abs_median = np.nan*np.ones(nepisodes+1)
        prop_states_deltaV_relevant = np.nan*np.ones(nepisodes+1)     # Proportion of states for which the change in V is relevant w.r.t. previous value (e.g. > 1%)

        # Initialize a few that make sense to initialize
        V_abs_n = np.zeros(nepisodes+1)
        V_abs_mean[0] = np.mean(np.abs(V))
        V_abs_median[0] = np.median(np.abs(V))
        V_abs_n[0] = 0

        # Iterate on the episodes to run
        nepisodes_max_steps_reached = 0
        t = -1            # This time index is used when a cycle set has been given
                          # so that we can estimate the stationary probability of states using renewal theory,
                          # as in that case (of a given cycle set), we assume that learning occurs under the average reward criterion
                          # (o.w., if we didn't use the average reward criterion to learn --but instead the discounted reward criterion,
                          # cycles would not be well defined, as cycles assume a continuing task which is not suitable for the discounted reward criterion
                          # --only episodic tasks are suitable for the discounted reward criterion).
        for episode in range(nepisodes):
            # Reset the environment
            # (this reset is normally carried out by the gym module, e.g. by the toy_text.discrete.DiscreteEnv environment's reset() method
            # where the initial state is chosen based on the isd attribute of the object, i.e. of the Initial State Distribution defining the initial state,
            # which was modified above when parameter `start` is not None)
            self.env.reset()
            done = False
            if verbose and np.mod(episode, verbose_period) == 0:
                print("@{}".format(get_current_datetime_as_string()))
                print("Episode {} of {} running...".format(episode+1, nepisodes), end=" ")
                print("(agent starts at state: {}".format(self.env.getState()), end=" ")
            if self.debug:
                print("\n[DEBUG] Episode {} of {}:".format(episode+1, nepisodes))
                print("\t[DEBUG] Starts at state {}".format(self.env.getState()))
                print("\t[DEBUG] State value function at start of episode:\n\t{}".format(learner.getV().getValues()))

            # Time step within the current episode (the first time step is t_episode = 0
            t_episode = -1
            while not done:
                t += 1
                t_episode += 1

                # Current state and action on that state leading to the next state
                state = self.env.getState()
                action = policy.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                # Set `done` to True when the maximum number of steps to run has been reached
                # This is important because a set of operations are done when the episode ends,
                # such as storing the learning rates alpha used in the episode
                # (see the learner object for more details, in particular de learn() method called below
                # that learns the state value function V)
                if max_time_steps is not None and t_episode >= max_time_steps - 1:     # `-1` because t_episode starts at 0 and max_time_steps counts the number of steps
                    nepisodes_max_steps_reached += 1
                    done = True
                    if self.debug:
                        print("[DEBUG] (MAX TIME STEP = {} REACHED!)".format(max_time_steps))
                if self.debug:
                    print("t in episode: {}, s={}, a={} -> ns={}, r={}".format(t_episode, state, action, next_state, reward))

                if self.debug and done:
                    print("--> [DEBUG] Done [{} iterations] at state {} with reward {}".format(t_episode+1, self.env.getState(), reward))
                    print("\t[DEBUG] Updating the value function at the end of the episode...")

                # Learn: i.e. update the value function (stored in the learner) with the new observation
                learner.learn(t_episode, state, action, next_state, reward, done, info)

                # Observation state
                if state_observe is not None:
                    # Store the value function of the state just estimated
                    V_state_observe += [learner.getV().getValue(state_observe)]

                # Check if the system has entered the set of states defining a cycle
                if set_cycle is not None:
                    if entered_set_cycle(state, next_state):
                        # Notes on the cycle time calculation:
                        # 1) We sum +1 to t because the next_state happens at the NEXT time step, which is t+1
                        # and the next_state is the one that defines absorption, so we should measure the time at which the next_state occurs.
                        # (in fact, note that the very first t value is 0, therefore we should not consider the first entry
                        # time to be 0 if the system enters the cycle set at the very first step...)
                        # 2) The fact we use `t` to compute the cycle time --instead of `t_episode`-- indicates that we are considering the task to be a continuous task,
                        # as opposed to episodic task, because the value of `t` is NOT reset at the beginning of each episode,
                        # instead it keeps increasing with every new time step.
                        cycle_times += [t + 1 - last_cycle_entrance_time]
                        last_cycle_entrance_time = t + 1   # We mark the time the system entered the cycle set, so that we can compute the next cycle time
                        if len(cycle_times) > 1:
                            # We disregard the first cycle time from the computation of the average
                            # because the first entering event may not represent a cycle as the system may have not previously exited the set.
                            num_cycles += 1
                            expected_cycle_time += (cycle_times[-1] - expected_cycle_time) / num_cycles
                            if DEBUG_TRAJECTORIES:
                                print(f"Entered cycle set: (t_episode, t, t_cycle, s, ns) = ({t_episode}, {t}, {cycle_times[-1]}, {state}, {next_state})")

                    # Update the count of the state ONLY after the first "cycle time"
                    # so that we make sure that the counts are measured during a TRUE cycle (as the first one may be a degenerate cycle)
                    if num_cycles > 0:
                        arr_state_counts_within_cycles[state] += 1

            #------- EPISODE FINISHED --------#
            if verbose and np.mod(episode, verbose_period) == 0:
                print(", agent ENDS at state: {} at time step {})".format(self.env.getState(), t_episode), end=" ")
                print("")

            # Change of V w.r.t. previous episode (to monitor convergence)
            deltaV = (learner.getV().getValues() - V)
            deltaV_rel = np.array([0.0  if dV == 0
                                        else dV
                                        for dV in deltaV]) / np.array([abs(v) if v != 0 else max(1, V_abs_mean[episode]) for v in V])
                ## Note that the denominator when computing deltaV_rel is mean|V| when V(s) = 0
                ## This prevents division by 0 and still gives a sensible relative error value.
                ## Note that V_abs_mean[episode] contains mean|V| for the PREVIOUS episode since
                ## V_abs_mean is an array of length nepisodes+1 where the first value is the average
                ## of the initial guess for the value function.

            # Update historical information
            arr_state_counts = learner.getStateCounts()
            states_visited_so_far = [s for s in range(self.env.getNumStates()) if arr_state_counts[s] > 0]
            states_visited_with_positive_deltaV_rel_abs = [s for s in states_visited_so_far if np.abs(deltaV_rel[s]) > 0]
                ## The above is used to find the minimum |V| value that plays a role in the computation of deltaV_rel
                ## so that we can evaluate how big this relative change could be, just because of a very small |V| value.
            max_signed_deltaV = find_signed_max_value(deltaV[states_visited_so_far])
            max_signed_deltaV_rel = find_signed_max_value(deltaV_rel[states_visited_so_far])
            if verbose_convergence:
                # Check the min(|V|) because it is often 0.0 even when the count for the state > 0
                # This minimum = 0.0 should come from the terminal states which have value 0.
                #if episode == 1:
                #    print(np.c_[V, arr_state_counts])
                #if np.min(np.abs(V[states_visited_so_far])) > 0.0:
                #    print(np.c_[V, arr_state_counts])
                #    import sys
                #    sys.exit()
                if isinstance(self.env, MountainCarDiscrete):
                    # Get the 2D state from the 1D state (for better understanding of where the agent started exploring the environment in the episode)
                    start_state = self.env.get_state_from_index(learner.states[0])
                else:
                    start_state = learner.states[0]
                print("*** Episode {} (start = {}, #steps = {} of {}), Convergence:".format(episode+1, start_state, t_episode+1, max_time_steps), end=" ")
                try:
                    print("Convergence: min(|V|) with non-zero delta = {:.3g}, mean(V) PREV = {:.3g}, mean(V) NOW = {:.3g}" \
                      .format(np.min(np.abs(V[states_visited_with_positive_deltaV_rel_abs])), np.mean(V[states_visited_so_far]), np.mean(learner.getV().getValues()[states_visited_so_far])) + \
                                       ", mean(delta) = {:.3g}, mean(|delta_rel|) = {:.1f}%, max(|delta_rel|) = {:.1f}% +/-max(delta) = {:.3g} +/-max(delta_rel) = {:.1f}% (n={} of {})" \
                      .format(np.mean(deltaV[states_visited_so_far]), np.mean(np.abs(deltaV_rel[states_visited_so_far]))*100,
                              np.max(np.abs(deltaV_rel[states_visited_so_far]))*100, max_signed_deltaV, max_signed_deltaV_rel*100,
                              len(states_visited_so_far), self.env.getNumStates()) + \
                                       ", mean(alpha) @end = {:.3g}".format(np.mean(learner._alphas[states_visited_so_far])) # Average alpha at the end of episode over ALL states visited so far
                      )
                except:
                    print("WARNING - states_visited_with_positive_deltaV_rel_abs is empty.\n")
                    #print("Count of states visited so far: {}, {}, {}" \
                    #      .format(arr_state_counts, states_visited_so_far, states_visited_with_positive_deltaV_rel_abs))
                    pass

            V = learner.getV().getValues()
            V_abs_mean[episode+1] = np.mean(np.abs(V[states_visited_so_far]))
            V_abs_mean_weighted[episode+1] = np.sum(arr_state_counts * np.abs(V)) / np.sum(arr_state_counts)
            if len(states_visited_with_positive_deltaV_rel_abs) > 0:
                V_abs_min[episode+1] = np.min(np.abs(V[states_visited_with_positive_deltaV_rel_abs]))
            deltaV_abs_mean[episode+1] = np.mean(np.abs(deltaV[states_visited_so_far]))
            deltaV_max_signed[episode+1] = max_signed_deltaV
            deltaV_rel_abs_mean[episode+1] = np.mean(np.abs(deltaV_rel[states_visited_so_far]))
            deltaV_rel_abs_mean_weighted[episode+1] = \
                np.sum( arr_state_counts[states_visited_so_far] * np.abs(deltaV_rel[states_visited_so_far]) ) / np.sum(arr_state_counts[states_visited_so_far])
            deltaV_rel_abs_max[episode+1] = np.max(np.abs(deltaV_rel[states_visited_so_far]))
            deltaV_rel_abs_max_weighted[episode+1] = \
                np.max( arr_state_counts[states_visited_so_far] * np.abs(deltaV_rel[states_visited_so_far]) / np.sum(arr_state_counts[states_visited_so_far]) )
            deltaV_rel_max_signed[episode+1] = max_signed_deltaV_rel
            V_abs_median[episode+1] = np.median(np.abs(V[states_visited_so_far]))
            deltaV_abs_median[episode+1] = np.median(np.abs(deltaV[states_visited_so_far]))
            deltaV_rel_abs_median[episode+1] = np.median(np.abs(deltaV_rel[states_visited_so_far]))
            V_abs_n[episode+1] = len(states_visited_so_far)
            prop_states_deltaV_relevant[episode+1] = np.sum( np.abs(deltaV_rel) > 0.01 ) / self.env.getNumStates()

            if compute_rmse:
                if self.env.getV() is not None:
                    if weights_rmse is not None:
                        weights = learner.getStateCounts()
                    RMSE[episode+1] = rmse(self.env.getV(), learner.getV().getValues(), weights=weights)
                    MAPE[episode+1] = mape(self.env.getV(), learner.getV().getValues(), weights=weights)

            # Update the count within cycles of the last state visited by the chain when `done`
            # (normally this is a terminal state, unless the simulation was truncated before reaching a terminal state
            # because the simulation time was already larger than the maximum simulation time allowed per episode)
            if set_cycle is not None and num_cycles > 0:
                arr_state_counts_within_cycles[next_state] += 1

            if plot and np.mod(episode, verbose_period) == 0:
                # Plot the estimated value function at the end of the episode
                if self.env.getDimension() == 2:
                    plt.figure(fig_V.number)
                    (ax_V, ax_C) = fig_V.subplots(1, 2)
                    shape = self.env.getShape()
                    terminal_rewards = self.env.getTerminalRewards()

                    state_values = np.asarray(learner.getV().getValues()).reshape(shape)
                    if len(terminal_rewards) > 0:
                        colornorm = plt.Normalize(vmin=np.min(terminal_rewards), vmax=np.max(terminal_rewards))
                    else:
                        colornorm = None
                    ax_V.imshow(state_values, cmap=colors, norm=colornorm)

                    arr_state_counts = learner.getStateCounts().reshape(shape)
                    colors_count = cm.get_cmap("Blues")
                    colornorm = plt.Normalize(vmin=0, vmax=np.max(arr_state_counts))
                    ax_C.imshow(arr_state_counts, cmap=colors_count, norm=colornorm)

                    fig_V.suptitle("Episode {} of {}".format(episode, nepisodes))
                    if pause > 0:
                        plt.pause(pause)
                    plt.draw()
                else:
                    #print("episode: {} (T={}), color: {}".format(episode, t_episode, colors(episode/nepisodes)))
                    plt.figure(fig_V.number)
                    plt.plot(self.env.getAllStates(), learner.getV().getValues(), linewidth=0.5, color=colors(episode/nepisodes))
                    plt.title("Episode {} of {}".format(episode+1, nepisodes))
                    if pause > 0:
                        plt.pause(pause)
                    plt.draw()
                    #fig.canvas.draw()

                    if state_observe is not None:
                        plt.figure(fig_RMSE_state.number)

                        # Compute quantities to plot
                        RMSE_state_observe = rmse(np.array(self.env.getV()[state_observe]), np.array(learner.getV().getValue(state_observe)), weights=weights)
                        se95 = 2*np.sqrt( 0.5*(1-0.5) / (episode+1))
                        # Only count falling inside the CI starting at episode 100 so that
                        # the normal approximation of the SE is more correct.
                        if episode + 1 >= 100:
                            ntimes_rmse_inside_ci95 += (RMSE_state_observe <= se95)

                        # Plot of the estimated value of the state
                        plt.plot(episode+1, learner.getV().getValue(state_observe), 'r*-', markersize=3)
                        #plt.plot(episode, np.mean(np.array(V_state_observe)), 'r.-')
                        # Plot of the estimation error and the decay of the "confidence bands" for the true value function
                        plt.plot(episode+1, RMSE_state_observe, 'k.-', markersize=3)
                        plt.plot(episode+1,  se95, color="gray", marker=".", markersize=2)
                        plt.plot(episode+1, -se95, color="gray", marker=".", markersize=2)
                        # Plot of learning rate
                        #plt.plot(episode+1, learner._alphas[state_observe], 'g.-')

                        # Finalize plot
                        ax = plt.gca()
                        ax.set_ylim((-1, 1))
                        yticks = np.arange(-10,10)/10
                        ax.set_yticks(yticks)
                        ax.axhline(y=0, color="gray")
                        plt.title("Value and |error| for state {} - episode {} of {}".format(state_observe, episode+1, nepisodes))
                        plt.legend(['value', '|error|', '2*SE = 2*sqrt(0.5*(1-0.5)/episode)'])
                        if episode + 1 == nepisodes:
                            # Show the true value function coverage as x-axis label
                            ax.set_xlabel("% Episodes error is inside 95% Confidence Interval (+/- 2*SE) (for episode>=100): {:.1f}%" \
                                          .format(ntimes_rmse_inside_ci95/(episode+1 - 100 + 1)*100))
                        plt.legend(['value', '|error|', '2*SE = 2*sqrt(0.5*(1-0.5)/episode)'])
                        plt.draw()

            if plot and isinstance(learner, LeaTDLambdaAdaptive) and episode == nepisodes - 1:
                learner.plot_info(episode, nepisodes)

            # Reset the learner for the next episode
            # (WITHOUT resetting the value functions nor the episode counter)
            learner.reset(reset_episode=False, reset_value_functions=False)

        # Comment this out to NOT show the plot right away
        # in case the calling function adds a new plot to the graph generated here
        if plot:
            if self.env.getDimension() == 2:
                # Plot the state counts in a separate image
                fig_C = plt.figure()
                plt.figure(fig_C.number)

                state_counts_min = np.min( learner.getStateCounts() )
                state_counts_mean = np.mean( learner.getStateCounts() )
                state_counts_max = np.max( learner.getStateCounts() )

                shape = self.env.getShape()
                arr_state_counts = learner.getStateCounts().reshape(shape)
                colors_count = cm.get_cmap("Blues")
                colornorm = plt.Normalize(vmin=0, vmax=np.max(arr_state_counts))
                plt.imshow(arr_state_counts, cmap=colors_count, norm=colornorm)
                # Font size factor
                fontsize = 14
                factor_fs = np.min((5/shape[0], 5/shape[1]))
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        # Recall the x axis corresponds to the columns of the matrix shown in the image
                        # and the y axis corresponds to the rows
                        plt.text(y, x, "{:.0f}".format(arr_state_counts[x,y]),
                                 fontsize=fontsize*factor_fs, horizontalalignment='center', verticalalignment='center')

                plt.title("State counts by state\n# visits: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})" \
                          .format(state_counts_min, state_counts_mean, state_counts_max))
            else:
                # Add the state counts to the plot of the state value function
                #plt.colorbar(cm.ScalarMappable(cmap=colormap))    # Does not work
                plt.figure(fig_V.number)
                ax = plt.gca()
                ax2 = ax.twinx()    # Create a secondary axis sharing the same x axis
                ax2.bar(self.env.getAllStates(), learner.getStateCounts(), color="blue", alpha=0.3)
                plt.sca(ax) # Go back to the primary axis
                #plt.figure(fig_V.number)

        print("Percentage of episodes reaching max step = {:.1f}%: ".format(nepisodes_max_steps_reached / nepisodes*100))

        self.finalize_run()

        return  learner.getV().getValues(), learner.getQ().getValues(), learner.getStateCounts(), RMSE, MAPE, \
                {   't': t,   # Simulation time, i.e. number of discrete steps taken during the whole simulation
                    # Value of alpha for each state at the end of the LAST episode run
                    'alphas_at_last_episode': learner._alphas,
                    # All what follows is information by episode (I don't explicitly mention it in the key name because it may make the key name too long...)
                    # (Average) alpha by episode (averaged over visited states in the episode)
                    'alpha_mean': learner.alpha_mean_by_episode,
                    'lambda_mean': learner.lambda_mean_by_episode if isinstance(learner, LeaTDLambdaAdaptive) else None,
                    'V_abs_mean': V_abs_mean,
                    'V_abs_mean_weighted': V_abs_mean_weighted,
                    'V_abs_min': V_abs_min,       # min|V| on states with a non-zero relative change in order to evaluate the impact of |V| when dividing to compute the relative change
                    'deltaV_max_signed': deltaV_max_signed,  # maximum change of V keeping the sign! Goal: understand whether the maximum change oscillates suggesting that it is converging fine
                    'deltaV_rel_max_signed': deltaV_rel_max_signed,  # maximum relative change of V keeping the sign! Goal: understand whether the maximum change oscillates suggesting that it is converging fine
                    'deltaV_abs_mean': deltaV_abs_mean,
                    'deltaV_rel_abs_mean': deltaV_rel_abs_mean,
                    'deltaV_rel_abs_mean_weighted': deltaV_rel_abs_mean_weighted,
                    'deltaV_rel_abs_max': deltaV_rel_abs_max,
                    'deltaV_rel_abs_max_weighted': deltaV_rel_abs_max_weighted,
                    'V_abs_median': V_abs_median,
                    'deltaV_abs_median': deltaV_abs_median,
                    'deltaV_rel_abs_median': deltaV_rel_abs_median,
                    'V_abs_n': V_abs_n,
                    'prop_states_deltaV_relevant': prop_states_deltaV_relevant,
                    'prop_episodes_max_steps_reached': nepisodes_max_steps_reached / nepisodes,
                    'num_cycles': num_cycles if set_cycle is not None else None,
                    'last_cycle_entrance_time': last_cycle_entrance_time if set_cycle else None,
                    'expected_cycle_time': expected_cycle_time if set_cycle is not None else None,
                    'state_counts_within_cycles': arr_state_counts_within_cycles if set_cycle is not None else None,
                }

    def initialize_run_with_learner_status(self, nepisodes, learner, compute_rmse, weights, state_observe):
        """
        The run is initialized by storing the initial learner status in terms of:
        - RMSE (if its computation is requested)
        - MAPE (if the RMSE computation has been requested)
        - Value of the state to observe (if any)

        nepisodes: int
            Number of episodes to run in the experiment.

        learner: Learner
            Learner used in the experiment.

        compute_rmse: bool
            Whether to compute the RMSE and the MAPE of the estimated value function over all states.

        weights: array
            Weights to use when computing the RMSE and MAPE. Set it to None if no weights should be used.

        state_observe: int
            A state index whose RMSE should be observed as the episode progresses.
            It is assumed that its value was already parsed by the run() method in terms of compatibility with
            parameter `compute_rmse`.

        Return: tuple
        Tuple containing the following elements:
        - Initial list of one element containing the value of the state to observe, or None if state_observe = False.
        - Array of length nepisodes + 1 where the initial RMSE and the RMSE at the end of each episode will be stored,
        or None if compute_rmse = False.
        - Same as above but for the MAPE.
        - 0, which is the initial number of times the RMSE of the state to observe is smaller than the RMSE associated
        to the 95% confidence interval of the value of the state to observe.
        """
        # Store the initial RMSE, i.e. based on the initial state values proposed by the learner
        if compute_rmse:
            RMSE = np.nan*np.zeros(nepisodes+1)
            MAPE = np.nan*np.zeros(nepisodes+1)
            RMSE[0] = rmse(self.env.getV(), learner.getV().getValues(), weights=weights)
            MAPE[0] = mape(self.env.getV(), learner.getV().getValues(), weights=weights)
        else:
            RMSE = None
            MAPE = None
        if state_observe is not None:
            V = [learner.getV().getValue(state_observe)]
        else:
            V = None
        # Keep track of the number of times the true value function of the state (assumed 0!)
        # is inside its confidence interval. In practice this only works for the mid state
        # of the 1D gridworld with the random walk policy.
        ntimes_rmse_inside_ci95 = 0

        return V, RMSE, MAPE, ntimes_rmse_inside_ci95

    def finalize_run(self):
        # Restore the initial state distribution of the environment (for the next simulation)
        # Note that the restore step done in the __exit__() method may NOT be enough, because the Simulator object
        # may still exist once the simulation is over.
        if self._isd_orig is not None:
            self.env.setInitialStateDistribution(self._isd_orig)
            self._isd_orig = None

        # TOMOVE: (DM-2022/04/28) This change of the estimated and true value functions should NOT be done
        # because of the following:
        # - The value of terminal states is equal to 0 BY DEFINITION (see Sutton 2018, I guess)
        # - If we change the true value of terminal states here, i.e. at the end of an experiment,
        # these values are kept for any subsequent experiment we run, thus making the value of the RMSE
        # computed on those experiments to be WRONG! (This is what made me spend about ONE OR MORE DEBUGGING DAYS
        # to figure out why the plots of the average RMSE by episode on TD(lambda) experiments run by simu_lambda.py
        # converged to a very large value (~0.5, when normally the value should converge to 0.1), except for the
        # very first experiment!
        # Actually, this change is mostly for an aesthetic reason when plotting the true and estimated value functions,
        # especially in 2D gridworlds (as opposed to 1D).
        # MY CONCLUSION is that this type of setup should be done by the PLOTTING FUNCTION itself.
        #
        # NOTE ALSO that we should NOT counteract this bad aesthetic plotting effect by setting the estimated value
        # of terminal states at the end of each episode to the reward received when transitioning to the terminal state,
        # for reasons explained when computing the true value function in the Gridworld1D environment (essentially
        # because the TD error observed when reaching the terminal state would count the observed reward as DOUBLE,
        # one time from R(T) and one time from the terminal state value which we set equal to R(T) as well!)
        #
        ## Set the value of terminal states to their reward, both the True values and estimated values
        ## (just to make plots of the state value function more understandable, specially in environments > 1D)
        #for s, r in self.env.getTerminalRewardsDict():
        #    self.agent.getLearner().getV()._setWeight(s, r)
        #    if self.env.getV() is not None:
        #        self.env.getV()[s] = r

    def simulate(self, nexperiments, nepisodes, max_time_steps=None, start=None, compute_rmse=True, weights_rmse=None,
                 verbose=False, verbose_period=1, verbose_convergence=False, plot=False):
        """
        Simulates the agent interacting with the environment for a number of experiments and number
        of episodes per experiment.

        Parameters:
        nexperiments: int
            Number of experiments to run.

        nepisodes: int
            Number of episodes to run per experiment.

        max_time_steps: int
            Maximum number of steps to run each episode for.
            default: np.Inf

        start: int, optional
            Index of the state each experiment should start at.
            When None, the starting state is picked randomly following the initial state distribution
            of the environment.

        compute_rmse: bool, optional
            Whether to compute the RMSE of the estimated value function over all states.
            Useful to analyze rate of convergence of the estimates when the true value function is known.

        weights_rmse: bool or array, optional
            Whether to use weights when computing the RMSE or the weights to use.
            default: None

        verbose: bool, optional
            Whether to show the experiment that is being run and the episodes for each experiment.
            default: False

        verbose_period: int, optional
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        verbose_convergence: bool, optional
            Whether to monitor convergence by showing the change in value function estimate at each new episode
            w.r.t. the estimate in the previous episode.
            default: False

        plot: bool, optional
            Whether to generate plots by each run of an experiment (by the run() method).

        Returns: tuple
        Tuple containing the following elements:
        - Avg(N): The average number of visits to each state over all experiments
        - Avg(RMSE): Root Mean Square Error averaged over all experiments
        - SE(RMSE): Standard Error of the average RMSE
        - n(RMSE): Number of experiments run leading to the value of Avg(RMSE) and SE(RMSE).
        - Episodic RMSE Avg: array containing the Root Mean Square Error by episode averaged
        over all experiments.
        - Episodic RMSE SE: array containing the standard error of the Root Mean Square Error by episode.
        - Episodic MAPE Avg: array containing the Mean Absolute Percent Error by episode averaged
        over all experiments.
        - Episodic MAPE SE: array containing the standard error of the Mean Absolute Percent Error by episode.
        - Episodic n: number of experiments run leading to the episodic RMSE / MAPE averages and standard errors.
        - a dictionary containing additional relevant information, as follows:
            - 'alphas_at_episode_end': the value of the learning parameter `alpha` for each state
            at the end of the last episode in the last experiment.
            - 'alpha_mean_by_episode': average learning parameter `alpha` by episode
            (averaged over visited states in each episode) over all experiments.
        """

        #---------------------------- Auxiliary functions ---------------------------------#
        def aggregate(nexperiments, kpi_sum, kpi_sum2):
            kpi_mean = kpi_sum / nexperiments
            #if kpi_sum2 - nexperiments*kpi_mean**2 < 0:
            #    print("WARNING!")
            #    print(kpi_sum2 - nexperiments*kpi_mean**2)
            # Note: below we limit the calculation of kpi_sum2 - n*mean(kpi)**2 because this difference could be slightly negative (e.g. -1E-16)

            # What follows is:
            # - SE = sigma / sqrt(n)
            # - sigma2 = (SS - n*mean(X)) / (n-1)  (unbiased estimation of sigma2)
            if nexperiments > 1:
                kpi_se = np.sqrt( np.max( [0, kpi_sum2 - nexperiments*kpi_mean**2] ) / (nexperiments - 1) ) \
                                    / np.sqrt(nexperiments)
            else:
                kpi_se = np.repeat(np.nan, nepisodes + 1)   # The SE is NaN for EACH episode

            return kpi_mean, kpi_se, nexperiments

        def aggregate_by_episode(nexperiments, kpi_sum_by_episode, kpi_sum2_by_episode):
            kpi_by_episode_mean = kpi_sum_by_episode / nexperiments
            #if any(kpi_sum2_by_episode - nexperiments*kpi_by_episode_mean**2 < 0):
            #    print("WARNING!")
            #    print(kpi_sum2_by_episode - nexperiments*kpi_by_episode_mean**2)
            # Note: below we limit the calculation of kpi_sum2 - n*mean(kpi)**2 because this difference could be slightly negative (e.g. -1E-16)

            # What follows is the following computation BY EPISODE:
            # - SE = sigma / sqrt(n)
            # - sigma2 = (SS - n*mean(X)) / (n-1)  (unbiased estimation of sigma2)
            if nexperiments > 1:
                kpi_by_episode_se = np.sqrt( np.max( np.c_[ np.repeat(0, len(kpi_sum2_by_episode)), kpi_sum2_by_episode - nexperiments*kpi_by_episode_mean**2 ], axis=1 ) / (nexperiments - 1) ) \
                                    / np.sqrt(nexperiments)
            else:
                kpi_by_episode_se = np.repeat(np.nan, len(kpi_sum2_by_episode))   # The SE is NaN for EACH episode

            return kpi_by_episode_mean, kpi_by_episode_se, nexperiments
        #---------------------------- Auxiliary functions ---------------------------------#

        if not (is_integer(nexperiments) and nexperiments > 0):
            raise ValueError("The number of experiments must be a positive integer number ({})".format(nexperiments))
        if not (is_integer(nepisodes) and nepisodes > 0):
            raise ValueError("The number of episodes must be a positive integer number ({})".format(nepisodes))

        # Average number of visits to each state over all experiments
        # (we initialize it as a float number because it will store an AVERAGE)
        n_visits = 0.0

        # Collect the average learning rates by episode (averaged over all states)
        # These variables will hold such piece of information averaged over all experiments run.
        # The `episode2` variables are used to efficiently compute their standard error.
        alpha_mean_by_episode = np.zeros(nepisodes)
        alpha_mean_by_episode2 = np.zeros(nepisodes)

        # For TD(lambda) algorithms, collect the average lambdas by episode (averaged over all states)
        # These variables will hold such piece of information averaged over all experiments run.
        # The `episode2` variables are used to efficiently compute their standard error.
        if isinstance(self.agent.getLearner(), LeaTDLambdaAdaptive):
            lambda_mean_by_episode = np.zeros(nepisodes)
            lambda_mean_by_episode2 = np.zeros(nepisodes)

        # Collect the average |V| and the average max relative change in V by episode.
        # These variables will hold those averages AVERAGED over all experiments run.
        # They are arrays whose length equal the number of episodes run per experiment.
        # The `episode2` variables are used to efficiently compute their standard error.
        V_abs_mean_by_episode = np.zeros(nepisodes+1)
        V_abs_mean_by_episode2 = np.zeros(nepisodes+1)
        deltaV_abs_mean_by_episode = np.zeros(nepisodes+1)
        deltaV_abs_mean_by_episode2 = np.zeros(nepisodes+1)
        deltaV_max_signed_by_episode = np.zeros(nepisodes+1)
        deltaV_max_signed_by_episode2 = np.zeros(nepisodes+1)
        deltaV_rel_max_signed_by_episode = np.zeros(nepisodes+1)
        deltaV_rel_max_signed_by_episode2 = np.zeros(nepisodes+1)

        # Collect the RMSE (when the true value function is available)
        if compute_rmse:
            RMSE = 0.   # RMSE at the end of the experiment averaged over all experiments
            RMSE2 = 0.  # Used to compute the standard error of the RMSE averaged over all experiments
            RMSE_by_episode = np.zeros(nepisodes+1)  # RMSE at each episode averaged over all experiments
                                                      # (note `nepisodes+1` because the first stored RMSE value is at episode 0,
                                                      # i.e. computed on the the initial guess of the value function)
            RMSE_by_episode2 = np.zeros(nepisodes+1) # Used to compute the standard error of the RMSE by episode
                                                      # (same comment as above)
            MAPE = 0.
            MAPE2 = 0.
            MAPE_by_episode = np.zeros(nepisodes+1)
            MAPE_by_episode2 = np.zeros(nepisodes+1)

        # IMPORTANT: Besides the numpy seed, We set the seed of the environment
        # because normally the environment on which the simulation takes place inherits from the DiscreteEnv
        # environment defined in the gym.envs.toy_text.discrete, which defines its own random number generator
        # (np_random) which is used when calling its reset() and step() methods.
        np.random.seed(self.seed)
        [seed] = self.env.seed(self.seed)
        for exp in np.arange(nexperiments):
            if verbose:
                print("Running experiment {} of {} (#episodes = {})..." \
                      .format(exp+1, nexperiments, nepisodes), end=" ")
            V, Q, n_visits_i, RMSE_by_episode_i, MAPE_by_episode_i, learning_info = \
                    self.run(nepisodes=nepisodes, max_time_steps=max_time_steps,
                             start=start, seed=None,    # We pass seed=None so that the seed is NOT set by this experiment
                                                        # Otherwise ALL experiments would have the same outcome!
                             compute_rmse=compute_rmse, weights_rmse=weights_rmse,
                             plot=plot,
                             verbose=verbose, verbose_period=verbose_period, verbose_convergence=verbose_convergence)
            # Number of visits done to each state at the end of the experiment
            n_visits += n_visits_i
            # Learning rates alpha
            # Note that we need to convert to np.array because the average alpha values are stored as a list
            alpha_mean_by_episode += np.array(learning_info['alpha_mean'])
            alpha_mean_by_episode2 += np.array(learning_info['alpha_mean'])**2
            # Lambda values
            if isinstance(self.agent.getLearner(), LeaTDLambdaAdaptive):
                lambda_mean_by_episode += np.array(learning_info['lambda_mean'])
                lambda_mean_by_episode2 += np.array(learning_info['lambda_mean']) ** 2
            # Estimated state values and their maximum change at the end of EACH episode run in the experiment
            V_abs_mean_by_episode += learning_info['V_abs_mean']
            V_abs_mean_by_episode2 += learning_info['V_abs_mean']**2
            deltaV_abs_mean_by_episode += learning_info['deltaV_abs_mean']
            deltaV_abs_mean_by_episode2 += learning_info['deltaV_abs_mean']**2
            deltaV_max_signed_by_episode += learning_info['deltaV_max_signed']
            deltaV_max_signed_by_episode2 += learning_info['deltaV_max_signed']**2
            deltaV_rel_max_signed_by_episode += learning_info['deltaV_rel_max_signed']
            deltaV_rel_max_signed_by_episode2 += learning_info['deltaV_rel_max_signed']**2
            # RMSE at the end of the last episode
            if compute_rmse:
                RMSE_i = RMSE_by_episode_i[-1]
                RMSE += RMSE_i
                RMSE2 += RMSE_i**2
                # Array containing the RMSE for each episode 1, 2, ..., nepisodes run in the experiment
                # The RMSE at the first episode is large and at the last episode should be small.
                RMSE_by_episode += RMSE_by_episode_i
                RMSE_by_episode2 += RMSE_by_episode_i**2

                MAPE_i = MAPE_by_episode_i[-1]
                MAPE += MAPE_i
                MAPE2 += MAPE_i**2
                MAPE_by_episode += MAPE_by_episode_i
                MAPE_by_episode2 += MAPE_by_episode_i**2
                if verbose:
                    #print("\tRMSE by episode: {}".format(RMSE_by_episode_i))
                    #print("\tMAPE by episode: {}".format(MAPE_by_episode_i))
                    print("\tRMSE(at end of experiment) = {:.3g}".format(RMSE_i))
                    print("\tMAPE(at end of experiment) = {:.3g}".format(MAPE_i))

        # Compute the average over ALL experiments
        # Note that these averages may sometimes be averages on other averages
        # (e.g. V_abs_mean_by_episode_mean where we are averaging the *average* estimated |V| by EPISODE.
        n_visits_mean = n_visits / nexperiments
        alpha_mean_by_episode_mean, alpha_mean_by_episode_se, _ = aggregate_by_episode( nexperiments,
                                                                                        alpha_mean_by_episode,
                                                                                        alpha_mean_by_episode2)
        if isinstance(self.agent.getLearner(), LeaTDLambdaAdaptive):
            lambda_mean_by_episode_mean, lambda_mean_by_episode_se, _ = aggregate_by_episode(nexperiments,
                                                                                            lambda_mean_by_episode,
                                                                                            lambda_mean_by_episode2)
        else:
            lambda_mean_by_episode_mean = lambda_mean_by_episode_se = None
        V_abs_mean_by_episode_mean, V_abs_mean_by_episode_se, _ = aggregate_by_episode( nexperiments,
                                                                                        V_abs_mean_by_episode,
                                                                                        V_abs_mean_by_episode2)
        deltaV_abs_mean_by_episode_mean, deltaV_abs_mean_by_episode_se, _ = aggregate_by_episode(nexperiments,
                                                                                        deltaV_abs_mean_by_episode,
                                                                                        deltaV_abs_mean_by_episode2)
        deltaV_max_signed_by_episode_mean, deltaV_max_signed_by_episode_se, _ = aggregate_by_episode(nexperiments,
                                                                                        deltaV_max_signed_by_episode,
                                                                                        deltaV_max_signed_by_episode2)
        deltaV_rel_max_signed_by_episode_mean, deltaV_rel_max_signed_by_episode_se, _ = aggregate_by_episode(nexperiments,
                                                                                        deltaV_rel_max_signed_by_episode,
                                                                                        deltaV_rel_max_signed_by_episode2)
        if compute_rmse:
            # For the last episode, we only compute the RMSE for now (not the MAPE)
            RMSE_mean, RMSE_se, RMSE_n = aggregate(nexperiments, RMSE, RMSE2)
            RMSE_by_episode_mean, RMSE_by_episode_se, RMSE_by_episode_n = aggregate_by_episode(nexperiments, RMSE_by_episode, RMSE_by_episode2)
            MAPE_by_episode_mean, MAPE_by_episode_se, _ = aggregate_by_episode(nexperiments, MAPE_by_episode, MAPE_by_episode2)
        else:
            RMSE_mean = RMSE_se = RMSE_n = RMSE_by_episode_mean = RMSE_by_episode_se = RMSE_by_episode_n = None

        learning_info_over_experiments = dict({
                'alpha_mean_by_episode_mean': alpha_mean_by_episode_mean,                       # Length = #episodes
                'lambda_mean_by_episode_mean': lambda_mean_by_episode_mean,                     # Length = #episodes
                'V_abs_mean_by_episode_mean': V_abs_mean_by_episode_mean,                       # Length = #episodes + 1
                'deltaV_abs_mean_by_episode_mean': deltaV_abs_mean_by_episode_mean,             # Length = #episodes + 1
                'deltaV_max_signed_by_episode_mean': deltaV_max_signed_by_episode_mean,         # Length = #episodes + 1
                'deltaV_rel_max_signed_by_episode_mean': deltaV_rel_max_signed_by_episode_mean, # Length = #episodes + 1
        })

        return n_visits_mean, RMSE_mean, RMSE_se, RMSE_n, \
               RMSE_by_episode_mean, RMSE_by_episode_se, \
               MAPE_by_episode_mean, MAPE_by_episode_se, RMSE_by_episode_n, \
               learning_info_over_experiments

    #--- Getters and Setters
    def getEnv(self):
        return self.env

    def getAgent(self):
        return self.agent

    def getCase(self):
        return self.case

    def getReplication(self):
        return self.replication

    def setCase(self, case):
        "Sets the simulation case, identifying a common set of simulations performed by a certain criterion"
        self.case = case

    def setReplication(self, replication):
        "Sets the replication number associated to the simulation"
        self.replication = replication
