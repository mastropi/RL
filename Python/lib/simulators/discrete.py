# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:21:54 2022

@author: Daniel Mastropietro
@description: Definition of functions and classes used for the simulation of discrete-time MDPs with discrete states and actions.
"""

import os
import sys
import copy
import warnings
from typing import Union
from datetime import datetime

from collections import deque   # Used for fast update of lists at the borders (which is a very common operation done on lists by the methods implemented here)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
import gym

from Python.lib.agents.learners import LearningCriterion, LearningTask
from Python.lib.agents.learners.episodic.discrete.fv import LeaFV
from Python.lib.agents.learners.episodic.discrete.td import LeaTDLambdaAdaptive
from Python.lib.environments.mountaincars import MountainCarDiscrete
from Python.lib.estimators import DEBUG_ESTIMATORS
from Python.lib.estimators.fv import initialize_phi, estimate_expected_reward, estimate_stationary_probabilities, update_phi, update_phi_on_all_states
from Python.lib.simulators.fv import reactivate_particle
from Python.lib.simulators import DEBUG_TRAJECTORIES, MIN_NUM_CYCLES_FOR_EXPECTATIONS, choose_state_from_set, parse_simulation_parameters, show_messages

from Python.lib.utils.basic import find_signed_max_value, generate_datetime_string, get_current_datetime_as_string, insort, is_integer, measure_exec_time
from Python.lib.utils.computing import compute_survival_probability, mape, rmse


class Simulator:
    """
    Simulator class that runs a Reinforcement Learning simulation on a discrete-time, discrete-state, discrete-action environment

    Arguments:
    env: gym.Env
        Environment with discrete states and actions that is assumed to have the following methods, getters and setters defined:
        - seed(): to set the random generator seed
        - reset(): to reset the environment's state
        - getNumStates()
        - getNumActions()
        - getInitialStateDistribution() --> returns an array that is a COPY of the initial state distribution
            (responsible for defining the initial state when running the simulation)
        - getState() --> returns the current state of the environment
        - getDimension()
        - getShape()
        - getAllStates()
        - getTerminalStates()
        - getRewards()
        - getInitialStateDistribution()
        - getV() --> returns the true state value function (which can be None, if unknown)
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
        Seed to use as base seed for a set of experiments run when the user calls the simulate() method.
        The seed for each experiment is appropriately changed from this base seed in order to have
        different results for each experiment.
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
        # TODO: (2024/02/13) Convert this class into a SimulatorDiscrete class that inherits from a generic Simulator class (to be defined in simulators/__init__.py) as this class is currently inherited by the SimulatorQueue class, which is NOT discrete-time simulator but continuous-time simulator

        if not issubclass(env.__class__, gym.Env):
            raise TypeError("The environment must inherit from the {} class in the {} module: {}" \
                            .format(gym.Env.__name__, gym.__module__, env.__class__))

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
        dt_today = datetime.today()
        if self.log:
            if logsdir is None:
                logsdir = os.getcwd()
            logsdir = os.path.abspath(logsdir)
            self.logfile = os.path.join(logsdir, generate_datetime_string(dt=dt_today, prefix=self.__class__.__name__, extension=".log"))
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
            self.results_file = os.path.join(resultsdir, generate_datetime_string(dt=dt_today, prefix=self.__class__.__name__, extension=".csv"))
            self.fh_results = open(self.results_file, "w")
            print("-----> File opened for output with simulation results:\n{}".format(self.results_file))
            print("-----> If the process stops for some reason, finalize the file with simul.close().")
        else:
            self.results_file = None
            self.fh_results = None

        self.env = env
        self.agent = agent
        self.seed = seed

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
            if self.agent.getLearner().getLearningTask() == LearningTask.CONTINUING:
                kwargs['nepisodes'] = 1
                return self._run_single_continuing_task(**kwargs)
            else:
                return self._run_single(**kwargs)

    def _run_fv(self, max_time_steps_fv=None, min_num_cycles_for_expectations=None,
                reset_value_functions=True, seed=None, verbose=True, verbose_period=100, plot=False, colormap="seismic", pause=0.1):
        """
        Runs all the simulations that are needed to learn differential value functions using the Fleming-Viot approach.

        These simulations include:
        - A simulation on a single Markov chain which is used to estimate E(T_A), the expected reabsorption to the set
        of uninteresting states, A.
        - A Fleming-Viot simulation on N particles which is used to estimate Phi(t, x) and P(T > t), which contribute
        to the numerator of the Fleming-Viot estimator of stationary state probabilities.

        Arguments:
        max_time_steps_fv: (opt) int
            Maximum number of steps to be observed for ALL particles when running the FV simulation.
            default: None, in which case its default value is defined by method _run_simulation_fv()

        min_num_cycles_for_expectations: (opt) int
            Minimum number of reabsorption cycles that should be observed in order to estimate the expected reabsorption time, E(T_A)
            when learning is done under the AVERAGE reward criterion. Under the DISCOUNTED reward criterion, E(T_A) does not need to be estimated.
            If this minimum number of cycles is NOT observed, then the expected reabsorption time is NOT estimated and the FV simulation is NOT run
            (as there is no use of running it, since we need an estimate of E(T_A) in order to produce an FV estimate of the average reward).
            NOTE: (2024/02/13) Currently, the lack of an E(T_A) estimate ONLY affects the FV estimation of the average reward,
            but this estimation does NOT affect the estimation of the differential value functions (which is the definition of value functions
            under the AVERAGE reward criterion used in CONTINUING learning tasks), because such estimation does NOT currently use
            the average reward estimated by FV for correction of the return to obtain the differential value functions;
            instead the traditional average reward estimated from the single Markov chain exploration of the environment is used.
            This is NOT the best approach to use (because the average reward could be WAY underestimated if it is rarely observed --as is the case
            in applications of FV!), however in order to use the FV-estimated average reward in place of the traditionally estimated average reward,
            we need to use the ITERATIVE update of the FV-estimated average reward (because value functions are estimated by a TD learner
            and a TD learner of value functions needs an average reward that is either already available OR is updated at every learning step
            to be used for correction of the return that thus gives the estimate of the value functions). This iterative update of the FV-estimated average
            reward, although already implemented in LeaFV, is currently not in place by the FV learning implemented here (because the method
            defined in LeaFV that ITERATIVELY computes the FV-estimated average reward is NOT currently called).
            default: None, in which case the value MIN_NUM_CYCLES_FOR_EXPECTATIONS is used under the AVERAGE reward criterion, and 0 under the DISCOUNTED reward criterion

        reset_value_functions: (opt) bool
            Whether the estimated value functions should be reset at the start of the

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
        if min_num_cycles_for_expectations is None:
            if self.agent.getLearner().getLearningCriterion() == LearningCriterion.AVERAGE:
                min_num_cycles_for_expectations = MIN_NUM_CYCLES_FOR_EXPECTATIONS
            else:
                # The expected reabsorption cycle time E(T_A) does not need to be estimated for the DISCOUNTED reward criterion,
                # so we do not require a minimum number of cycles to be observed during the single Markov chain simulation.
                # This simulation is simply used to provide a first estimate of the value functions, but NOT of the expected reabsorption cycle time.
                min_num_cycles_for_expectations = 0
        dict_params_simul = dict({  'max_time_steps_fv': max_time_steps_fv,                         # Maximum number of time steps allowed for the N particles (comprehensively) in the FV simulation used to estimate the QSD Phi(t,x)
                                    'N': self.agent.getLearner().getNumParticles(),
                                    'T': self.agent.getLearner().getNumTimeStepsForExpectation(),   # Maximum number of time steps allowed in each episode of the single Markov chain that estimates the expected reabsorption time E(T_A)
                                    'absorption_set': self.agent.getLearner().getAbsorptionSet(),
                                    'activation_set': self.agent.getLearner().getActivationSet(),
                                    'min_num_cycles_for_expectations': min_num_cycles_for_expectations,
                                    'seed': seed})
        dict_params_info = dict({'verbose': verbose,
                                 'verbose_period': verbose_period,
                                 't_learn': 0,
                                 'plot': plot,
                                 'colormap': colormap,
                                 'pause': pause}) # This indexes the learning epoch of the optimal policy
        #--- Parse input parameters ---

        # Create the particles as copies of the main environment
        envs = [self.env if i == 0 else copy.deepcopy(self.env) for i in range(dict_params_simul['N'])]

        state_values, action_values, state_counts_from_single_markov_chain, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, \
            time_last_absorption, max_survival_time, n_events_et, n_events_fv = \
                self._estimate_state_value_function_and_expected_reward_fv( envs, dict_params_simul, dict_params_info,
                                                                            probas_stationary_start_state_et=self.agent.getLearner().getProbasStationaryStartStateET(),
                                                                            probas_stationary_start_state_fv=self.agent.getLearner().getProbasStationaryStartStateFV(),
                                                                            reset_value_functions=reset_value_functions)

        return state_values, action_values, state_counts_from_single_markov_chain, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, n_events_et, n_events_fv

    def _estimate_state_value_function_and_expected_reward_fv(self, envs, dict_params_simul, dict_params_info,
                                                              probas_stationary_start_state_et: dict=None,
                                                              probas_stationary_start_state_fv: dict=None,
                                                              reset_value_functions=True):
        """
        Estimates the differential state values and action values, the stationary state probabilities,
        and the expected reward (a.k.a. long-run average reward) using the Fleming-Viot approach.

        Arguments:
        envs: List
            List of environments used to run the FV process.

        dict_params_simul: dict
            Dictionary containing simulation and estimation parameters.
            The dictionary should contain at least the keys described in function `parse_simulation_parameters()`.

        dict_params_info: dict
            Dictionary containing information to display or parameters to deal with the information to display.
            Accepted keys are:
            - verbose: whether to be verbose during the simulation.
            - verbose_period: the number of simulation time steps at which to be verbose.
            - t_learn: the number of learning steps, when FV is used in the context of FVRL, i.e. to learn an optimum policy
            This is ONLY used for informational purposes, i.e. to show which stage of the policy learning we are at.
            - plot: whether to generate an interactive plot showing the evolution of the state value function estimate.
            - colormap: name of the colormap to use for each new estimate of the state value function added to the plot (e.g. "seismic").
            - pause: number of seconds to pause after updating the plot with the new estimate of the state value function. MUST be > 0 if we want the
            plot to be updated!

        probas_stationary_start_state_et: (opt) dict
            Stationary distribution to use for the selection of the start state of the single Markov chain
            simulation used to estimate the expected reabsorption cycle time, E(T_A).
            States are the dictionary keys and their probability of selection are the values.
            Normally this is a uniform distribution on the states at the boundary of A.
            default: None, in which case the initial state distribution stored in the environment is used

        probas_stationary_start_state_fv: (opt) dict
            Stationary distribution to use for the selection of the start state of each FV particle used in the FV process.
            States are the dictionary keys and their probability of selection are the values.
            default: None, in which case a uniform random distribution on the activation set is used

        reset_value_functions: (opt) bool
            Whether value function estimates, possibly stored in the value functions learner, should be reset at the beginning of the learning process
            (i.e. before launching the single Markov chain simulation used to estimate the expected reabsorption time, E(T_A)).
            Note that it might be useful to NOT reset the value functions if the value function learner is used to e.g. generate critic values used in
            an Actor-Critic policy learning context. In that case, we usually like to use the value function estimates obtained from the critic
            (i.e. from this value function learner) run when learning the policy at the PREVIOUS policy learning step (so that we don't start
            the learning of the value functions from scratch. This makes sense because the value functions associated to ONE policy learning step
            (of the Actor-Critic algorithm) are expected to be similar to those associated to the previous policy learning step.
            default: True

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
        # Note: Even though the seed is set by the _run_single() method below (which receives a `seed` parameter)
        # we need to set the seed here for a reproducible selection of the start state of the simulation,
        # whenever the start state distribution on the absorption set is given by the user.
        np.random.seed(dict_params_simul['seed'])
        # -- Parse input parameters

        # -- Step 1: Simulate a single Markov chain to estimate the expected cycle time of return to A, E(T_A)
        # Define the start state, ONLY used at the very first episode
        # (all subsequent episodes are started using the initial state distribution (isd) stored in the environment object)
        start_state = None
        if probas_stationary_start_state_et is not None:
            start_state = choose_state_from_set(dict_params_simul['absorption_set'], probas_stationary_start_state_et)
        print(f"SINGLE simulation for the estimation of the expected reabsorption time E(T_A) starts at state s={start_state} "
              f"(when None, the simulation starts following the Initial State Distribution of the environment: {self.env.getInitialStateDistribution()}")
        state_values, action_values, state_counts_et, _, _, learning_info = \
            self._run_single_continuing_task(
                            max_time_steps=dict_params_simul['T'],      # Max simulation time over ALL episodes
                            start_state_first_episode=start_state,
                            reset_value_functions=reset_value_functions,
                            seed=dict_params_simul['seed'],
                            set_cycle=dict_params_simul['absorption_set'],
                            verbose=dict_params_info.get('verbose', False),
                            verbose_period=dict_params_info.get('verbose_period', 1))
        n_events_et = learning_info['nsteps']
        n_cycles_absorption_used = learning_info['num_cycles']
        time_last_absorption = learning_info['last_cycle_entrance_time']
        average_reward_from_single_simulation = self.agent.getLearner().getAverageReward()
        print(f"--> Average reward estimated from the single simulation: {average_reward_from_single_simulation} (it will be used to correct the value functions estimated by the FV simulation")

        # -- Step 2: Simulate N particles with FLeming-Viot to compute the empirical distribution and estimate the stationary probabilities, and from them the expected reward
        # BUT do this ONLY when the estimation of E(T_A) is reliable... otherwise, set the stationary probabilities and expected reward to NaN.
        print("\n*** RESULTS OF FLEMING-VIOT SIMULATION ***")
        if is_estimation_of_denominator_unreliable():
            # FV is not run because the simulation that is used to estimate E(T_A) would not generate a reliable estimation
            # (most likely it would UNDERESTIMATE E(T_A) making the probabilities be OVERESTIMATED)
            print("Fleming-Viot process is NOT run because the estimation of the expected absorption time E(T_A) cannot be reliably performed"
                " because of an insufficient number of observed cycles after the burn-in period of {} time steps: {} < {}" \
                "\nThe estimated stationary probabilities and estimated expected reward will be set to NaN." \
                .format(dict_params_simul['burnin_time_steps'], n_cycles_absorption_used,
                        dict_params_simul['min_num_cycles_for_expectations']))
            state_counts_all = state_counts_et
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
            # NOTE: (2023/11/22) We set the expected reabsorption time to the number of steps observed during the single chain simulation
            # when it is 0 in order to avoid division by 0 when the value of the simulation parameter 'min_num_cycles_for_expectations' is 0,
            # i.e. when the user does not require any specific number of minimum cycles as guarantee of a reliable estimate of the expected reabsorption time.
            # A use case of this situation is when learning the optimum policy using policy gradient or Actor Critic where
            # we do NOT need to estimate the average reward because it is cancelled out in the Advantage function Q(s,a) - V(s)
            # which is the term that weights the log-policy value contributing to the loss function that is minimized
            # to estimate the optimal theta parameters of the policy.
            expected_absorption_time = learning_info['expected_cycle_time'] if learning_info['expected_cycle_time'] > 0.0 else learning_info['nsteps']
            print(f"FV simulation on N={N} particles starts...")
            # TEMPORARY (2023/12/05) To test whether FV learns the optimal policy when the values of the states and actions taken in set A are NOT estimated at all
            #self.agent.getLearner().reset(reset_value_functions=True)
            #print(f"VALUE FUNCTIONS RESET BEFORE STARTING FLEMING-VIOT!\nV = \n{self.agent.getLearner().getV().getValues()}\nQ = \n{self.agent.getLearner().getQ().getValues()}")
            # TEMPORARY
            if self.agent.getLearner().getLearningCriterion() == LearningCriterion.AVERAGE:
                method_fv = self._run_simulation_fv; uniform_jump_rate = N         # We need to divide the FV sum by N in order to make it comparable with the denominator E(T_A) whose time measure is N times shorter than the time measure of an FV particle (as the FV particle is chosen on average once every N time steps of the FV system)
                #method_fv = self._run_simulation_fv_fraiman; uniform_jump_rate = 1  # In this case, all FV particles are updated at the same system's time step, therefore no adjustmend needs to be done to the FV sum.
                #method_fv = self._run_simulation_fv_fraiman_modified; uniform_jump_rate = 1  # In this case, all FV particles are updated at the same system's time step, therefore no adjustmend needs to be done to the FV sum.
                start_set = dict_params_simul['activation_set']
            else:
                # When running FV to learn the value functions V(s) and Q(s,a) the particles should start all over the place outside A, so that we can explore all those states
                # more uniformly, as we need to estimate the value of EACH state.
                # I've tested starting all the particles at the outer boundary of A to estimate V(s) on a 5-state 1D gridworld with +1 reward with a very small
                # stationary probability due to a policy that goes left with 90% probability, and we get:
                # - More variability in the estimation of the state value V(s) across replications.
                # - A few replications (less than 10%) estimate V(s) = 0 for some states s, which does NOT happen when starting the particles uniformly distribution outside A.
                # Parameters: N=50 particles, R=20 replications.
                # See meeting minutes in entry dated 17-Jan-2024 for more details.
                method_fv = self._run_simulation_fv_learnvaluefunctions; uniform_jump_rate = 1
                start_set = self.agent.getLearner().active_set.difference(self.env.getTerminalStates())
            n_events_fv, state_values, action_values, state_counts_fv, phi, df_proba_surv, expected_absorption_time, max_survival_time = \
                method_fv(  dict_params_info.get('t_learn', 0), envs,
                            dict_params_simul['absorption_set'],
                            start_set,
                            max_time_steps=dict_params_simul.get('max_time_steps_fv'),
                            dist_proba_for_start_state=probas_stationary_start_state_fv,
                            expected_absorption_time=expected_absorption_time,
                            estimated_average_reward=average_reward_from_single_simulation,
                            seed=dict_params_simul['seed'] + 131713,    # Choose a different seed from the one used by the single Markov chain simulation (note that this seed is the base seed used for the seeds assigned to the different FV particles)
                            verbose=dict_params_info.get('verbose', False),
                            verbose_period=dict_params_info.get('verbose_period', 1),
                            plot=dict_params_info['plot'],
                            colormap=dict_params_info['colormap'],
                            pause=dict_params_info['pause'])
            state_counts_all = state_counts_et + state_counts_fv
            #print(f"Shape of proba surv and phi: {df_proba_surv.shape}")
            print("Expected reabsorption time E(T_A): {:.3f} ({} cycles)".format(expected_absorption_time, learning_info['num_cycles']))
            print(f"proba_surv P(T>t):\n{df_proba_surv}")
            average_phi_values = dict([(x, np.mean(phi[x]['Phi'])) for x in phi.keys()])
            print(f"Time-average Phi value per state of interest:\n{average_phi_values}")
            print(f"Max survival time: {max_survival_time}")

            probas_stationary, integrals = estimate_stationary_probabilities(phi, df_proba_surv,
                                                                             expected_absorption_time,
                                                                             uniform_jump_rate=uniform_jump_rate)
            expected_reward = estimate_expected_reward(envs[0], probas_stationary)
            # Store the expected reward as average reward in the learner object
            # so that we can retrieve the average reward estimated by FV by using the method GenericLearner.getAverageReward()
            self.agent.getLearner().setAverageReward(expected_reward)

            if True or DEBUG_ESTIMATORS or show_messages(dict_params_info.get('verbose', False),
                                                 dict_params_info.get('verbose_period', 1),
                                                 dict_params_info.get('t_learn', 0)):
                #max_rows = pd.get_option('display.max_rows')
                #pd.set_option('display.max_rows', None)
                #print("Phi(t):\n{}".format(phi))
                #pd.set_option('display.max_rows', max_rows)
                print("Integrals: {}".format(integrals))
                print("Expected reabsorption time (on {} cycles): {}".format(learning_info['num_cycles'], expected_absorption_time))
                print("Stationary probabilities: {}".format(probas_stationary))
                print("Expected reward = {}".format(expected_reward))
                print("Average reward stored in learner = {}".format(self.agent.getLearner().getAverageReward()))
                assert np.isclose(expected_reward, self.agent.getLearner().getAverageReward()), \
                    f"The average reward estimated by FV ({expected_reward}) must be equal to the average reward stored in the FV learner ({self.agent.getLearner().getAverageReward()})"

        return state_values, action_values, state_counts_all, probas_stationary, expected_reward, expected_absorption_time, n_cycles_absorption_used, \
               time_last_absorption, max_survival_time, n_events_et, n_events_fv

    @measure_exec_time
    def _run_simulation_fv( self, t_learn, envs, absorption_set: set, start_set: set,
                            max_time_steps=None,
                            dist_proba_for_start_state: dict=None,
                            expected_absorption_time=None, expected_exit_time=None,
                            estimated_average_reward=None,
                            seed=None, verbose=False, verbose_period=1,
                            plot=False, colormap="seismic", pause=0.1):
        """
        Runs the Fleming-Viot simulation of the particle system and estimates the different pieces of information that
        are part of the Fleming-Viot estimator of the average reward:
        - Phi(t, x): the conditional probability conditional on non absorption or survival.
        - P(T > t): the survival (non-absorption) probability.
        - E(T_A): expected reabsorption probability to set A, only computed by this function when its parameter is None.
        Otherwise, the input parameter value is returned.
        In addition, the differential value functions V(s) and Q(s,a) are also estimated, from possibly an initial estimation
        obtained from the simulation of a single Markov chain (run by e.g. self._run_single()).

        Note that the learner is reset at the very beginning of this process, but without resetting the value functions
        that may have already undergone some learning experience, e.g. from an initial exploration of the environment.

        Arguments:
        t_learn: int
            The policy learning time step to which the simulation will contribute.
            Only used for informative purposes on where we are at when learning the policy or to decide whether to show information in the log.

        envs: list
            List of environments used to run the FV process.

        absorption_set: set
            Set with the entrance states to the set of uninteresting states A.
            This set is needed to measure the killing times used to estimate the survival probability P(T > t).

        start_set: set
            Set of states where the FV particles can be placed at the beginning of the simulation.
            This is normally the outer boundary of the absorption set A, i.e. the entrance set of to the complement of A.

        max_time_steps: (opt) int
            Maximum number of steps to run the simulation for, computed as the comprehensive number of
            transitions observed over ALL particles.
            default: None, in which case N*100 is used, where N is the number of particles in the FV system

        dist_proba_for_start_state: (opt) dict
            Probability distribution to use for the selection of the start state of each FV particle.
            According to theory, the start state should be selected from the activation set and should be selected
            following the stationary ENTRANCE distribution to the outer boundary of the absorption set A
            (i.e. the inner boundary of the complement of A). However, a reasonable approximation
            (that should beat the uniformly random selection of the start state) is the stationary distribution of the outer boundary of A
            (as opposed to the stationary *entrance* distribution).
            default: None, in which case a uniform distribution on the activation set is used

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

        estimated_average_reward: (opt) None
            An existing estimation of the average reward that is used as correction of the value functions
            being learned by this FV simulation process.
            default: None, in which case the average reward observed by the FV particles excursion is used
            as correction term. Note that this is an INFLATED estimation of the true average reward, observed
            by the original underlying Markov chain (from which the FV particle system is created), hence
            it might not be very sensible to use it. Instead, a prior estimation of the average reward is preferred,
            for instance from the single Markov chain simulation that is used to estimate the expected
            reabsorption time, E(T_A).

        seed: (opt) int
            Seed to use in the simulations as base seed (then the seed for each simulation is changed from this base seed).
            default: None, in which case a random seed is generated.

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: tuple
        Tuple with the following elements:
        - t: the last (discrete) time step of the simulation process, which coincides with the number of FV events, as at each time step t only one particle is updated.
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
            """
            Internal function that reactivates a particle until a valid reactivation is obtained
            (i.e. a particle whose state is not in an absorbed set of states.

            (2023/11/15) Note that the reactivated particle can perfectly be at a terminal state, because terminal
            states are NOT part of the absorption set.
            What IS true is that once a particle reaches a terminal state, THE NEXT TIME THE PARTICLE IS PICKED,
            its state is changed to an environment's start state, which COULD be part of the absorption set,
            in which case the particle is reactivated right-away (i.e. this function is called).
            """
            # TODO: (2023/09/06) Try to implement a more efficient way (i.e. without looping) to reactivate the particle to one of the other particles that are NOT at an absorption set
            done_reactivate = False
            state = envs[idx_particle].getState()
            if DEBUG_TRAJECTORIES:
                # We can add a `True` condition above in order to show the following useful piece of information of the proportion of particles at the state of interest x is useful for tracking the stabilization of Phi(t,x) around the QSD
                flags_particle_at_terminal_state = [1 if envs[p].getState() in self.env.getTerminalStates() else 0 for p in range(len(envs))]
                print("[reactivate_particle_internal] % particles at terminal states: {:.1f}% ({} out of {})".format(np.mean(flags_particle_at_terminal_state)*100, np.sum(flags_particle_at_terminal_state), len(envs)))
            new_state = None
            while not done_reactivate:
                idx_reactivate = reactivate_particle(envs, idx_particle, 0, reactivation_number=None)
                    ## (2023/01/05) the third parameter is dummy when we do NOT use method = ReactivateMethod.VALUE_FUNCTION to reactivate the particle inside function reactivate_particle().
                    ## (2024/01/28) Use reactivation_number=reactivation_number to indicate that we want to use method = ReactivateMethod.ROBINS.
                # TODO: (2023/11/15) Check whether there is any possibility that the particle to which the absorbed particle has been reactivated COULD really be in the absorption set...
                # Note that, at the initial devise of the FV simulation/estimation method, we have considered that the set of FV particles changes with time...
                # But this dynamic set of FV particles might no longer be the case, at the time of this writing (2023/11/15).
                if envs[idx_particle].getState() not in absorption_set:
                    done_reactivate = True
                    new_state = envs[idx_particle].getState()
                    if DEBUG_TRAJECTORIES:
                        print("*** t={}: Particle #{} ABSORBED at state={} and REACTIVATED to particle #{} at state {}" \
                              .format(t, idx_particle, state, idx_reactivate, new_state))

            return new_state
        #------------------------------- Auxiliary functions ----------------------------------#


        # ---------------------------- Check input parameters ---------------------------------#
        # -- Absorption and activation sets
        # Class
        if not isinstance(absorption_set, set):
            raise ValueError("Parameter `absorption_set` must be a set: {}".format(absorption_set))
        if not isinstance(start_set, set):
            raise ValueError("Parameter `start_set` must be a set: {}".format(start_set))
        if len(start_set) == 0:
            raise ValueError("Parameter `start_set` must have at least one element")

        if expected_absorption_time is None and expected_exit_time is None:
            raise ValueError("Parameter `expected_exit_time` must be provided when `expected_absorption_time` is None")
        # ---------------------------- Check input parameters ---------------------------------#

        N = len(envs)
        if max_time_steps is None:
            max_time_steps = N * 100   # N * "MAX # transitions allowed on average for each particle"
        policy = self.agent.getPolicy()  # Used to define the next action and next state
        learner = self.agent.getLearner()  # Used to learn (or keep learning) the value functions

        # Reset the learner, but WITHOUT resetting the value functions as they were possibly learned a bit during an initial exploration of the environment
        # What is most important of this reset is to reset the learning rates of all states and actions! (so that we start the FV-based learning with full intensity)
        learner.reset(reset_episode=True, reset_value_functions=False)

        # Set the start state of each environment/particle to an activation state, as this is a requirement
        # for the empirical distribution Phi(t).
        for i, env in enumerate(envs):
            # Environment seed
            seed_i = seed + i if seed is not None else None
            env.setSeed(seed_i)

            # Choose start state from the activation set
            start_state = choose_state_from_set(start_set, dist_proba_for_start_state)
            if learner.getLearningTask() == LearningTask.CONTINUING:
                assert start_state not in env.getTerminalStates(), \
                    f"The start state of an FV particle ({start_state}) cannot be a terminal state of the environment (terminal states = {env.getTerminalStates()}) for the CONTINUING learning task context." \
                    f"\nThe reason is that, before getting to a terminal state, there must be at least ONE transition of the particle" \
                    f" --because we need the `info` dictionary to be defined when processing a terminal state."
            env.setState(start_state)

        # Event times: the first event time is 0
        # This implies t=0 is considered the time at which each particle is positioned to their start state.
        event_times = [0]

        # Phi(t, x): Empirical probability of the states of interest (x)
        # at each time t when a variation in Phi(t, x) is observed.
        dict_phi = initialize_phi(envs, t=event_times[0], states_of_interest=learner.getActiveSet())

        # Initialize the list of observed survival times to be filled during the simulation below
        survival_times = [0]    # We set the first element of the survival times list to 0 so that we can "estimate" the survival probability at 0 (which is always equal to 1 actually)
        idx_reactivate = None   # This is only needed when we want to plot a vertical line in the particle evolution plot with the color of the particle to which an absorbed particle is reactivated
        has_particle_been_absorbed_once = [False] * N   # List that keeps track of whether each particle has been absorbed once
                                                        # so that we can end the simulation when all particles have been absorbed
                                                        # when the survival probability is estimated by this function.
        n_particles_absorbed_once = 0

        if True or DEBUG_ESTIMATORS:
            print("[DEBUG] @{}".format(get_current_datetime_as_string()))
            print("[DEBUG] State value function at start of simulation:\n\t{}".format(learner.getV().getValues()))
            print("[DEBUG] Action value function at start of simulation:\n\t{}".format(learner.getQ().getValues().reshape(self.env.getNumStates(), self.env.getNumActions())))

        idx_particle = -1
        done = False
        t = 0   # IMPORTANT: t represents the time at which a particle transitions to the NEXT state, and this is why we initialize t as 0, so that after the first transition,
                # the time step is t = 1, i.e. t signals the time at which the particle is at the NEXT state, after the transition has taken place.
                # This follows the usual convention for continuous-time Markov processes where the event time is the time at which the Markov chain is at the NEXT state,
                # after the event takes place, i.e. the state to which the Markov chain transitions to.
                # See e.g. Pierre Bremaud's book, Chapter 13, graph of the jump rate process.
        # Initial learning rate for Phi, which is the gamma* defined in Fraiman et al.
        alpha0 = +np.Inf    # Setting this initial learning rate to 1 gives the same weight to all historic (k < t) and current (k = t) Phi values
                            # setting it equal to >1 gives more weight to most recent estimates of Phi,
                            # setting it equal to <1 gives more weight to older estimates of the Phi.
        while not done:
            t += 1

            # We define a reactivation number, an integer between 0 and N-2 which is used to deterministically choose
            # the reactivation particle (if reactivation is via ReactivateMethod.ROBINS), in order to save time by not having to generate a uniform random number.
            # Note that the range from 0 to N-2 allows us to choose one of the N-1 particles to which the absorbed particle can be reactivated into.
            reactivation_number = t % (N - 1)

            event_times += [t]

            # Select the particle to move uniformly at random
            idx_particle = np.random.choice(N) # If choosing them in order, use: `(idx_particle + 1) % N`

            # Get the current state of the selected particle because that's the particle whose state is going to (possibly) change
            state = envs[idx_particle].getState()
            #print(f"particle {idx_particle} (s={state})", end=", ")

            # Check if the particle that was picked for update is CURRENTLY at a terminal state,
            # in which case we "restart" the process to an environment's initial state (according to its initial state distribution).
            # This is so because FV applies to a process whose transition probability from a terminal state to any initial state is equal to 1,
            # which is defined like that so that the process can continue indefinitely, as opposed to a process that ends at a terminal state.
            # This is required by FV because FV is used to estimate an expectation of a process running for unlimited time.
            # If the selected state is part of the absorption set, the time to absorption contributes to the estimation of the survival probability P(T>t)
            # --as long as the particle has never been absorbed before-- and the particle is reactivated right-away.
            if state in self.env.getTerminalStates():
                # Note: The following reset of the environment is typically carried out by the gym module,
                # e.g. by the toy_text.discrete.DiscreteEnv environment's reset() method where the initial state
                # is chosen based on the isd attribute of the object, i.e. of the Initial State Distribution defining the initial state.
                # Note also that the seed for the reset has been set separately for each particle before starting the FV simulation.
                next_state = envs[idx_particle].reset()
                reward = envs[idx_particle].getReward(next_state)
                if DEBUG_TRAJECTORIES:
                    print("___ Particle #{} in terminal state {} REINITIALIZED to environment's start state following its initial state distribution: next_state={}" \
                          .format(state, idx_particle, next_state))

                # Learn the value functions for the terminal state for the continuing learning task case,
                # because in that case the value of terminal states is NOT defined as 0.
                if learner.getLearningTask() == LearningTask.CONTINUING:
                    self.learn_terminal_state_values(t, state, next_state, reward, info)
                        ## Note: the `info` dictionary is guaranteed to be defined thanks to the assertion
                        ## at the initialization of the FV particles that asserts they cannot be at a terminal state.
                        ## Thanks to this condition, when a particle is at a terminal state, it means that it MUST have
                        ## transitioned from a non-terminal state, and this implies that the `info` dictionary has been
                        ## defined as the output of such transition.
            else:
                # Step on the selected particle
                action = policy.choose_action(state)
                next_state, reward, done_episode, info = envs[idx_particle].step(action)
                if estimated_average_reward is not None:
                    # Store the estimated average reward passed by the user in the `info` dictionary so that it can be used
                    # by the call to the learn() method below as correction value when learning the value functions under the average reward criterion
                    info['average_reward'] = estimated_average_reward

                # Learn: i.e. update the value function (stored in the learner) with the new observation
                # NOTES:
                # 1) Learning ONLY happens when the particle is NOT reactivated.
                # This makes total sense because reactivation sets the next state of the particle to a state
                # that is normally not reachable by the underlying Markov chain on which the FV process is built.
                # 2) We pass `done` to learn() and NOT `done_episode` because we want to update the average reward
                # over all episodes (set by Learner.store_trajectory_at_episode_end() and retrieved by GenericLearner.getAverageReward())
                # only when the FV SIMULATION IS OVER, not at the end of each episode. Otherwise, the estimated average
                # reward will fluctuate a lot (i.e. as the average reward observed by episode fluctuates) and this is NOT what we want,
                # we want a stable estimate of the average reward over all episodes.
                # TODO: (2024/01/29) Revise the correct use of the `done` variable here, instead of `done_episode`, because actually when we are done by `done`, this line will NEVER be executed because we will NOT enter again the `while done` loop...
                learner.learn(t, state, action, next_state, reward, done, info)

            if next_state in absorption_set:
                # The particle has been absorbed.
                # => Add the time to absorption to the set of times used to estimate the survival probability P(T>t) if it's the first absorption of the particle
                # => Reactivate the particle to the position of any of the other particles

                # Compute the absorption time as the time at which the particle is at *next_state*, and this is precisely the definition of t, as described above
                t_abs = t

                # Contribution to the survival probability
                if not has_particle_been_absorbed_once[idx_particle]:
                    # The survival time coincides with the absorption time since the particle has never been absorbed before when the program enters this block
                    # Note that, by construction, the survival times are sorted in ascending order.
                    t_surv = t_abs
                    survival_times += [t_surv]

                    # Mark the particle as "absorbed once" so that we don't use any other absorption time from this
                    # particle to estimate the survival probability, because the absorption times coming after the first
                    # absorption time should NOT contribute to the survival probability, because their initial state
                    # is NOT the correct one --i.e. it is not a state in the activation set
                    # (because the particle has been reactivated to any other state)
                    # which is a requirement for the estimation of the survival probability distribution.
                    has_particle_been_absorbed_once[idx_particle] = True
                    n_particles_absorbed_once += 1
                    # Show the progress of particles absorption
                    if int((n_particles_absorbed_once - 1) / N * 100) % 10 != 0 and int(n_particles_absorbed_once / N * 100) % 10 == 0:
                        print("t={} of {}: {:.1f}% of particles absorbed at least once ({} of {})".format(t, max_time_steps, n_particles_absorbed_once / N * 100, n_particles_absorbed_once, N))
                    if False:
                        print("Survival times observed so far: {}".format(sum(has_particle_been_absorbed_once)))
                        print(survival_times)

                # Reactivate the particle
                next_state = reactivate_particle_internal(idx_particle)
                reward = envs[idx_particle].getReward(next_state)
                if DEBUG_TRAJECTORIES:
                    print(f"--> Reactivated particle {idx_particle} from state {state} to state {next_state}")
                assert next_state not in absorption_set, \
                    f"The state of a reactivated particle must NOT be a state in the absorption set ({next_state})"

            # Update Phi based on the new state of the changed (and possibly also reactivated) particle
            dict_phi = update_phi(envs[0], len(envs), t, dict_phi, state, next_state, alpha=alpha0 / (t + alpha0))

            if DEBUG_TRAJECTORIES:
                print("[FV] Moved P={}, t={}: state={}, action={} -> next_state={}, reward={}" \
                      .format(idx_particle, t, state, action, next_state, reward),
                      end="\n")
                if N <= 10:
                    print(f"Current state of system:\n{[env.getState() for env in envs]}")

            idx_reactivate = None

            # CHECK DONE
            # If we want to interrupt the simulation by EITHER reaching the maximum time steps
            # OR
            # having all particles absorbed at least once, which is a logic that impacts ONLY the estimation of the average reward using FV,
            # but it does NOT impact the estimation of the value functions done by the call to learner.learn() inside this loop.
            # THEREFORE, if we are not interested in estimating the average reward using FV, we may want to continue until
            # we reach the maximum number of steps, regardless of the first absorption event of each particle.
            # Use this if we want to stop simulation either when the maximum number of steps has been reached OR all N particles have been absorbed at least once
            done = t >= max_time_steps or sum(has_particle_been_absorbed_once) == N
            # Use this if we want to stop simulation ONLY when the maximum number of steps has been reached.
            # This should be the preferred way if we want to do a FAIR comparison with other benchmark methods,
            # because we guarantee that all time steps specified are used (which normally comes from the FAIR comparison setup)
            # Note that at this point max_time_steps ALWAYS has a value because if `None` was given by the user, a default value is set at the beginning of the method.
            # WARNING: In this case we need to DISABLE the assertion below on the "maximum time stored in Phi(t,x)".
            #done = t >= max_time_steps

        # DONE
        if t >= max_time_steps:
            # Add the last observed time step to the list of survival times as many times as the number of particles left to absorb,
            # so that we can use the signal information collected in Phi since the last absorbed particle to the FV estimation of the expected reward.
            # Note that these survival times are underestimations of the actual survival times, so the FV estimate of the average reward
            # will have a negative bias, but this is a smaller bias than if we didn't consider the last observed time `t`
            # as part of the estimated survival probability function.
            n_particles_not_absorbed = N - sum(has_particle_been_absorbed_once)
            survival_times += list(np.repeat(t, n_particles_not_absorbed))
            if n_particles_not_absorbed > 0:
                print(f"WARNING: Not all {N} particles were absorbed at least ONCE because the maximum number of time steps for all particles ({max_time_steps} has been reached: # NOT absorbed particles = {n_particles_not_absorbed}")
        # The following assertion should be used ONLY when the FV process stops if all N particles are absorbed at least once before reaching max_time_steps
        assert np.max([np.max(dict_phi[x]['t']) for x in dict_phi.keys()]) <= np.max(survival_times), \
                "The maximum time stored in Phi(t,x) must be at most the maximum observed survival time"
        if show_messages(verbose, verbose_period, t_learn):
            print("==> FV agent ENDS at state {} at discrete time t = {} ({:.1f}% of max_time_steps={}, {:.1f}% of particles were absorbed once),"
                  " compared to maximum observed time for P(T>t) = {:.1f}." \
                    .format(envs[idx_particle].getState(), t, t/max_time_steps*100, max_time_steps, np.sum(has_particle_been_absorbed_once)/N*100, survival_times[-1]))

        # Compute the stationary probability of each state x in Phi(t, x) using Phi(t, x), P(T>t) and E(T_A)
        df_proba_surv = compute_survival_probability(survival_times)
        if expected_absorption_time is None:
            expected_absorption_time = expected_exit_time + np.mean(survival_times)
        max_survival_time = df_proba_surv['t'].iloc[-1]

        if DEBUG_ESTIMATORS:
            max_rows = pd.get_option('display.max_rows')
            pd.set_option('display.max_rows', None)
            print("Survival probability:\n{}".format(df_proba_surv))
            print("Phi:\n{}".format(dict_phi))
            pd.set_option('display.max_rows', max_rows)

        return t, learner.getV().getValues(), learner.getQ().getValues(), learner._state_counts, dict_phi, df_proba_surv, expected_absorption_time, max_survival_time

    @measure_exec_time
    def _run_simulation_fv_fraiman( self, t_learn, envs, absorption_set: set, start_set: set,
                                    max_time_steps=None,
                                    dist_proba_for_start_state: dict=None,
                                    expected_absorption_time=None, expected_exit_time=None,
                                    estimated_average_reward=None,
                                    seed=None, verbose=False, verbose_period=1,
                                    plot=False, colormap="seismic", pause=0.1):
        """
        Runs the Synchronous Fleming-Viot simulation of the particle system proposed by Fraiman et al. in their Oct-2020 paper
        (https://arxiv.org/abs/2010.09942, "Approximation quasi-stationary distributions with interactive reinforced random walks")
        where all particles are updated simultaneously (synchronously) and reactivated to a position following the estimated empirical distribution.

        Note however that this method tends to generate an FV estimator that underestimates the stationary probability, specially for lower probabilities.

        See the documentation for _run_simulation_fv() for more details.
        """
        #------------------------------- Auxiliary functions ----------------------------------#
        def extract_latest_phi_values(dict_phi):
            "Returns the distribution given by the latest Phi values, i.e. by the empirical distribution at the latest available time for each active state"
            latest_phi_values = dict.fromkeys(dict_phi.keys())
            for state, df_phi in dict_phi.items():
                assert len(df_phi) > 0, f"There must be at least one row in each data frame stored in the dictionary containing the estimates of the conditional occupation probability, Phi, over time (state={state})"
                latest_phi_values[state] = df_phi['Phi'].iloc[-1]
            return latest_phi_values

        def reactivate_particle_internal(idx_particle, reactivation_distribution):
            "Internal function that reactivates a particle to a state chosen the given reactivation distribution, typically the current estimate of the conditional occupation probability."
            state = envs[idx_particle].getState()
            if DEBUG_TRAJECTORIES:
                # We can add a `True` condition above in order to show the following useful piece of information of the proportion of particles at the state of interest x is useful for tracking the stabilization of Phi(t,x) around the QSD
                flags_particle_at_terminal_state = [1 if envs[p].getState() in self.env.getTerminalStates() else 0 for p in range(len(envs))]
                print("[reactivate_particle_internal] % particles at terminal states: {:.1f}% ({} out of {})".format(np.mean(flags_particle_at_terminal_state)*100, np.sum(flags_particle_at_terminal_state), len(envs)))
            new_state = reactivate_particle(envs, idx_particle, 0, reactivation_distribution=reactivation_distribution) # (2024/03/05) the third parameter is dummy
            if DEBUG_TRAJECTORIES:
                print("*** t={}: Particle #{} ABSORBED at state={} and REACTIVATED to state {}".format(t, idx_particle, state, new_state))

            return new_state
        #------------------------------- Auxiliary functions ----------------------------------#


        # ---------------------------- Check input parameters ---------------------------------#
        # -- Absorption and activation sets
        # Class
        if not isinstance(absorption_set, set):
            raise ValueError("Parameter `absorption_set` must be a set: {}".format(absorption_set))
        if not isinstance(start_set, set):
            raise ValueError("Parameter `start_set` must be a set: {}".format(start_set))
        if len(start_set) == 0:
            raise ValueError("Parameter `start_set` must have at least one element")

        if expected_absorption_time is None and expected_exit_time is None:
            raise ValueError("Parameter `expected_exit_time` must be provided when `expected_absorption_time` is None")
        # ---------------------------- Check input parameters ---------------------------------#

        N = len(envs)
        if max_time_steps is None:
            max_time_steps = N * 100   # N * "MAX # transitions allowed on average for each particle"
        policy = self.agent.getPolicy()  # Used to define the next action and next state
        learner = self.agent.getLearner()  # Used to learn (or keep learning) the value functions

        # Reset the learner, but WITHOUT resetting the value functions as they were possibly learned a bit during an initial exploration of the environment
        # What is most important of this reset is to reset the learning rates of all states and actions! (so that we start the FV-based learning with full intensity)
        learner.reset(reset_episode=True, reset_value_functions=False)

        # Set the start state of each environment/particle to an activation state, as this is a requirement
        # for the empirical distribution Phi(t).
        for i, env in enumerate(envs):
            # Environment seed
            seed_i = seed + i if seed is not None else None
            env.setSeed(seed_i)

            # Choose start state from the activation set
            start_state = choose_state_from_set(start_set, dist_proba_for_start_state)
            if learner.getLearningTask() == LearningTask.CONTINUING:
                assert start_state not in env.getTerminalStates(), \
                    f"The start state of an FV particle ({start_state}) cannot be a terminal state of the environment (terminal states = {env.getTerminalStates()}) for the CONTINUING learning task context." \
                    f"\nThe reason is that, before getting to a terminal state, there must be at least ONE transition of the particle" \
                    f" --because we need the `info` dictionary to be defined when processing a terminal state."
            env.setState(start_state)

        # Event times: the first event time is 0
        # This implies t=0 is considered the time at which each particle is positioned to their start state.
        event_times = deque([0])

        # Phi(t, z): Empirical probability of all active states z, at each time t when a variation in Phi(t, z) is observed.
        dict_phi = initialize_phi(envs, t=event_times[0], states_of_interest=learner.getActiveSet())

        # Initialize the list of observed survival times to be filled during the simulation below
        survival_times = deque([0])   # We set the first element of the survival times list to 0 so that we can "estimate" the survival probability at 0 (which is always equal to 1 actually)
        has_particle_been_absorbed_once = [False]*N   # List that keeps track of whether each particle has been absorbed once
                                                        # so that we can end the simulation when all particles have been absorbed
                                                        # when the survival probability is estimated by this function.
        # Times at which each particle visited an activation set state, so that we can measure the survival time of absorptions beyond the first one
        times_last_visit_activation_set = [0]*N
        # Times at which each particle was reactivated, which is used to decide whether we need to update the respective time of the last visit to the activation set (see logic explanation below)
        times_last_absorption = [0]*N

        if True or DEBUG_ESTIMATORS:
            print("[DEBUG] @{}".format(get_current_datetime_as_string()))
            print("[DEBUG] State value function at start of simulation:\n\t{}".format(learner.getV().getValues()))
            print("[DEBUG] Action value function at start of simulation:\n\t{}".format(learner.getQ().getValues().reshape(self.env.getNumStates(), self.env.getNumActions())))

        done = False
        t = 0   # IMPORTANT: t represents the time at which a particle transitions to the NEXT state, and this is why we initialize t as 0, so that after the first transition,
                # the time step is t = 1, i.e. t signals the time at which the particle is at the NEXT state, after the transition has taken place.
                # This follows the usual convention for continuous-time Markov processes where the event time is the time at which the Markov chain is at the NEXT state,
                # after the event takes place, i.e. the state to which the Markov chain transitions to.
                # See e.g. Pierre Bremaud's book, Chapter 13, graph of the jump rate process.
        # Initial learning rate for Phi, which is the gamma* defined in Fraiman et al.
        alpha0 = +np.Inf      # Setting this initial learning rate to 1 gives the same weight to all historic (k < t) and current (k = t) Phi values
                        # setting it equal to >1 gives more weight to most recent estimates of Phi,
                        # setting it equal to <1 gives more weight to older estimates of the Phi.
        while not done:
            t += 1
            event_times.append(t)

            # Move all particles at the SAME TIME by following Fraiman's transition probability K(phi) = p(x,y|a) + p(x,A|a) * phi(y)
            # where p(x,y|a) is the transition probability of the original Markov chain given action `a` and phi(y) is the current estimate of the conditional occupation
            # probability at EACH possible new state y.
            for idx_particle in range(N):
                # Get the current state of the particle
                state = envs[idx_particle].getState()
                #print(f"particle {idx_particle} (s={state})", end=", ")

                # Check if the particle is at a terminal state, in which case we need to "restart" the process to an environment's initial state.
                # This is so because FV applies to a process whose transition probability from a terminal state to any initial state is equal to 1,
                # defined like that so that the process can continue indefinitely, as opposed to a process that ends at a terminal state.
                # This is required by FV because FV is used to estimate an expectation of a process running for unlimited time.
                # If the selected state is part of the absorption set, the time to absorption contributes to the estimation of the survival probability P(T>t)
                # --as long as the particle has never been absorbed before-- and the particle is reactivated right-away.
                if state in self.env.getTerminalStates():
                    # Note: The following reset of the environment is typically carried out by the gym module,
                    # e.g. by the toy_text.discrete.DiscreteEnv environment's reset() method where the initial state
                    # is chosen based on the isd attribute of the object, i.e. of the Initial State Distribution defining the initial state.
                    # Note also that the seed for the reset has been set separately for each particle before starting the FV simulation.
                    next_state = envs[idx_particle].reset()
                    reward = envs[idx_particle].getReward(next_state)
                    if DEBUG_TRAJECTORIES:
                        print("___ Particle #{} in terminal state {} REINITIALIZED to environment's start state following its initial state distribution: next_state={}" \
                              .format(state, idx_particle, next_state))

                    # Learn the value functions for the terminal state for the continuing learning task case,
                    # because in that case the value of terminal states is NOT defined as 0.
                    if learner.getLearningTask() == LearningTask.CONTINUING:
                        self.learn_terminal_state_values(t, state, next_state, reward, info)
                            ## Note: the `info` dictionary is guaranteed to be defined thanks to the assertion
                            ## at the initialization of the FV particles that asserts they cannot be at a terminal state.
                            ## Thanks to this condition, when a particle is at a terminal state, it means that it MUST have
                            ## transitioned from a non-terminal state, and this implies that the `info` dictionary has been
                            ## defined as the output of such transition.
                else:
                    # Step on the selected particle
                    action = policy.choose_action(state)
                    next_state, reward, done_episode, info = envs[idx_particle].step(action)
                    if estimated_average_reward is not None:
                        # Store the estimated average reward passed by the user in the `info` dictionary so that it can be used
                        # by the call to the learn() method below as correction value when learning the value functions under the average reward criterion
                        info['average_reward'] = estimated_average_reward

                    # Learn: i.e. update the value function (stored in the learner) with the new observation
                    # NOTES:
                    # 1) Learning ONLY happens when the particle is NOT reactivated.
                    # This makes total sense because reactivation sets the next state of the particle to a state
                    # that is normally not reachable by the underlying Markov chain on which the FV process is built.
                    # 2) We pass `done` to learn() and NOT `done_episode` because we want to update the average reward
                    # over all episodes (set by Learner.store_trajectory_at_episode_end() and retrieved by GenericLearner.getAverageReward())
                    # only when the FV SIMULATION IS OVER, not at the end of each episode. Otherwise, the estimated average
                    # reward will fluctuate a lot (i.e. as the average reward observed by episode fluctuates) and this is NOT what we want,
                    # we want a stable estimate of the average reward over all episodes.
                    # 3) We pass update_phi=False because we only update the occupation probability Phi AFTER all particles have been reactivated, o.w. we would be reactivating
                    # different absorbed particles using *different* estimates of the conditional occupation probability, whereas they must be reactivated using the SAME estimate.
                    # TODO: (2024/01/29) Revise the correct use of the `done` variable here, instead of `done_episode`, because actually when we are done by `done`, this line will NEVER be executed because we will NOT enter again the `while done` loop...
                    learner.learn(t, state, action, next_state, reward, done, info, update_phi=False)

                if next_state in absorption_set:
                    # The particle has been absorbed.
                    # => Add the time to absorption to the set of times used to estimate the survival probability P(T>t) if it's the first absorption of the particle
                    # => Reactivate the particle following the current estimate of the occupation probability to the position of any of the other particles

                    # Compute the absorption time as the time at which the particle is at *next_state*, and this is precisely the definition of t, as described above
                    t_abs = t
                    # Update the time of the last absorption so that we can decide below whether to update the time of the last visit to the activation set (see logic explanation below)
                    times_last_absorption[idx_particle] = t_abs

                    # Contribution to the survival probability
                    if not has_particle_been_absorbed_once[idx_particle]:
                        # The survival time coincides with the absorption time since the particle has never been absorbed before when the program enters this block
                        # Note that, by construction, the survival times are sorted in ascending order.
                        t_surv = t_abs
                        survival_times.append(t_surv)

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
                    else:
                        t_surv = t_abs - times_last_visit_activation_set[idx_particle]
                        # Only do this if we want to use all survival times in the estimation of the survival probability, until the end of the simulation
                        #insort(survival_times, t_surv)

                    # Reactivate the particle using the current estimate of the occupation probability Phi(t,.)
                    current_estimate_of_occupation_probability = extract_latest_phi_values(dict_phi)
                    # Use one of the following if we want to reactivate particles using the QSD estimated from a long simulation of the FV system
                    # (The QSD were computed for the small 5-state 1D gridworld with terminal state at s=4, absorption set A = {0, 1}, and agent restarting at state s=2 when reaching s=4)
                    #current_estimate_of_occupation_probability = dict({2: 1 - 0.2 - 0.06, 3: 0.2, 4: 0.06})  # Estimated QSD for policy = [0.9, 0.1] in 5-state 1D gridworld
                    #current_estimate_of_occupation_probability = dict({2: 1 - 0.29 - 0.15, 3: 0.29, 4: 0.15})  # Estimated QSD for policy = [0.7, 0.3] in 5-state 1D gridworld
                    #current_estimate_of_occupation_probability = dict({2: 1 - 0.35 - 0.3, 3: 0.35, 4: 0.3})     # Estimated QSD for policy = [0.1, 0.9] in 5-state 1D gridworld
                    next_state = reactivate_particle_internal(idx_particle, current_estimate_of_occupation_probability)
                    reward = envs[idx_particle].getReward(next_state)
                    if DEBUG_TRAJECTORIES:
                        print(f"--> Reactivated particle {idx_particle} from state {state} to state {next_state}")
                    assert next_state not in absorption_set, f"The state of a reactivated particle must NOT be a state in the absorption set ({next_state})"

                # Update the time of the last visit time to the activation set so that we can use subsequent absorption times as new samples of the survival time
                # NOTE that this check of whether we need to update such visit time needs to happen even if the particle has been absorbed because it may happen
                # that the particle visits the activation set because of reactivation (e.g. when the activation set is {2}, the particle is reactivated to state s=2).
                particle_visits_activation_set_for_the_first_time_after_reactivation = lambda idx_particle: times_last_visit_activation_set[idx_particle] < times_last_absorption[idx_particle]
                if next_state in learner.getActivationSet() and particle_visits_activation_set_for_the_first_time_after_reactivation(idx_particle):
                    # Note that we update the time of the last visit to the activation set ONLY when the particle visits the activation set for the FIRST time after absorption/reactivation
                    # WARNING: this criterion WILL BIAS UPWARD the estimate of the survival *whenever subsequent visits* to the activation set are observed *before* absorption,
                    # as these subsequent visits will not be considered as contributors for the estimation of P(T>t), and those contributions would be necessarily smaller
                    # than the survival time we are keeping as contribution to the estimate of P(T>t).
                    # IN ADDITION, note that this upward bias also happens when we select only contributions coming from the first excursion until absorption coming from times
                    # measured since the very first visit to the activation set...
                    # THE COMPLICATION ABOUT CONSIDERING ALL CONTRIBUTIONS TO THE ESTIMATION OF P(T>t) is that we will no longer have an increasing list of survival times
                    # as they are observed.
                    # HOWEVER, the strategy of considering just the initial visit to the activation set as starting point makes the sample size on which P(T>t) is estimated
                    # a KNOWN and FIXED sample size... HENCE, using those survival times should NOT give a bias!
                    # On the other hand note that, strictly speaking, we should ONLY consider contributions to the survival function coming from visits to the activation set
                    # that take place following the STATIONARY EXIT DISTRIBUTION from A.
                    # This second constraint, however, can be solved by the idea suggested by Matt, namely that we compute the survival probability conditioned to starting
                    # in the activation set following the stationary exit distribution by weighting the estimates coming from survival times observed after visiting *specific*
                    # states in the activation set by the stationary exit distribution estimated from a single Markov chain excursion (i.e. the one used to estimate E(T_A)).
                    # Example of the above condition that bias the estimate of P(T>t) upward:
                    # the activation set is {2} and the trajectory is: [2, 3, 2, 1 (ABS), 4 (REACT), 3, 2, 3, 2, 1 (ABS)]
                    # we only update the time of the last visit to the activation set at the FIRST visit to s=2 after the absorption/reactivation took place above.
                    # TODO: (2024/03/07) Remove the upward bias mentioned above
                    times_last_visit_activation_set[idx_particle] = t   # Recall that t is the time at which the particle is at the NEXT state

                if DEBUG_TRAJECTORIES:
                    print("[FV] Moved P={}, t={}: state={}, action={} -> next_state={}, reward={}" \
                          .format(idx_particle, t, state, action, next_state, reward),
                          end="\n")
                    if N <= 10:
                        print(f"Current state of system:\n{[env.getState() for env in envs]}")

            print("t={} of {}: % particles absorbed at least once = {:.1f}% ({} of {})".format(t, max_time_steps, np.mean(has_particle_been_absorbed_once)*100, np.sum(has_particle_been_absorbed_once), N))

            # Update Phi after all particles have been updated
            dict_phi = update_phi_on_all_states(envs, t, dict_phi, alpha=alpha0 / (t + alpha0))

            # CHECK DONE
            # If we want to interrupt the simulation by EITHER reaching the maximum time steps
            # OR
            # having all particles absorbed at least once, which is a logic that impacts ONLY the estimation of the average reward using FV,
            # but it does NOT impact the estimation of the value functions done by the call to learner.learn() inside this loop.
            # THEREFORE, if we are not interested in estimating the average reward using FV, we may want to continue until
            # we reach the maximum number of steps, regardless of the first absorption event of each particle.
            # Use this if we want to stop simulation either when the maximum number of steps has been reached OR all N particles have been absorbed at least once
            done = t >= max_time_steps or sum(has_particle_been_absorbed_once) == N
            # Use this if we want to stop simulation ONLY when the maximum number of steps has been reached.
            # This should be the preferred way if we want to do a FAIR comparison with other benchmark methods,
            # because we guarantee that all time steps specified are used (which normally comes from the FAIR comparison setup)
            # Note that at this point max_time_steps ALWAYS has a value because if `None` was given by the user, a default value is set at the beginning of the method.
            # WARNING: In this case we need to DISABLE the assertion below on the "maximum time stored in Phi(t,x)".
            #done = t >= max_time_steps

        # DONE
        if t >= max_time_steps:
            # Add the last observed time step to the list of survival times as many times as the number of particles left to absorb,
            # so that we can use the signal information collected in Phi since the last absorbed particle to the FV estimation of the expected reward.
            # Note that these survival times are underestimations of the actual survival times, so the FV estimate of the average reward
            # will have a negative bias, but this is a smaller bias than if we didn't consider the last observed time `t`
            # as part of the estimated survival probability function.
            n_particles_not_absorbed = N - sum(has_particle_been_absorbed_once)
            for e in np.repeat(t, n_particles_not_absorbed):
                survival_times.append(e)
            if n_particles_not_absorbed > 0:
                print(f"WARNING: Not all {N} particles were absorbed at least ONCE because the maximum number of time steps for all particles ({max_time_steps} has been reached: # NOT absorbed particles = {n_particles_not_absorbed}")
        # The following assertion should be used ONLY when the FV process stops if all N particles are absorbed at least once before reaching max_time_steps
        assert np.max([np.max(dict_phi[x]['t']) for x in dict_phi.keys()]) <= np.max(survival_times), "The maximum time stored in Phi(t,x) must be at most the maximum observed survival time"
        if show_messages(verbose, verbose_period, t_learn):
            print("==> FV agent ENDS at state {} at discrete time t = {} ({:.1f}% of max_time_steps={}, {:.1f}% of particles were absorbed once),"
                  " compared to maximum observed time for P(T>t) = {:.1f}." \
                    .format(envs[idx_particle].getState(), t, t/max_time_steps*100, max_time_steps, np.sum(has_particle_been_absorbed_once)/N*100, survival_times[-1]))

        # Compute the stationary probability of each state x in Phi(t, x) using Phi(t, x), P(T>t) and E(T_A)
        df_proba_surv = compute_survival_probability(survival_times)
        if expected_absorption_time is None:
            expected_absorption_time = expected_exit_time + np.mean(survival_times)
        max_survival_time = df_proba_surv['t'].iloc[-1]

        if DEBUG_ESTIMATORS:
            max_rows = pd.get_option('display.max_rows')
            pd.set_option('display.max_rows', None)
            print("Survival probability:\n{}".format(df_proba_surv))
            print("Phi:\n{}".format(dict_phi))
            pd.set_option('display.max_rows', max_rows)

        n_events_fv = t*N

        return n_events_fv, learner.getV().getValues(), learner.getQ().getValues(), learner._state_counts, dict_phi, df_proba_surv, expected_absorption_time, max_survival_time

    @measure_exec_time
    def _run_simulation_fv_fraiman_modified(self, t_learn, envs, absorption_set: set, start_set: set,
                                            max_time_steps=None,
                                            dist_proba_for_start_state: dict=None,
                                            expected_absorption_time=None, expected_exit_time=None,
                                            estimated_average_reward=None,
                                            seed=None, verbose=False, verbose_period=1,
                                            plot=False, colormap="seismic", pause=0.1):
        """
        Runs the MODIFIED Synchronous Fleming-Viot simulation of the particle system proposed by Fraiman et al. in their Oct-2020 paper
        (https://arxiv.org/abs/2010.09942, "Approximation quasi-stationary distributions with interactive reinforced random walks")
        where all particles are updated simultaneously and reactivated to a position following the estimated empirical distribution.

        The difference with _run_simulation_fv_fraiman() has to do with the fact that the absorbed particles are reactivated following
        the empirical distribution of the non-absorbed particles, as opposed to the current estimate of the QSD, which corresponds to
        the empirical distribution (or a smooth version of it) of ALL particles just before absorption occurs.
        The latter strategy produces, specially for smaller probabilities, an underestimation of the stationary probability by the FV estimator.
        For more details, see e-mails exchanged with Nicolas Fraiman on 13/14-Mar-2024 and the meeting minutes for those dates.

        See also the documentation for _run_simulation_fv() for more details about the FV process.
        """
        #------------------------------- Auxiliary functions ----------------------------------#
        def extract_latest_phi_values(dict_phi):
            "Returns the distribution given by the latest Phi values, i.e. by the empirical distribution at the latest available time for each active state"
            latest_phi_values = dict.fromkeys(dict_phi.keys())
            for state, df_phi in dict_phi.items():
                assert len(df_phi) > 0, f"There must be at least one row in each data frame stored in the dictionary containing the estimates of the conditional occupation probability, Phi, over time (state={state})"
                latest_phi_values[state] = df_phi['Phi'].iloc[-1]
            return latest_phi_values

        def reactivate_particle_internal(idx_particle, reactivation_distribution):
            "Internal function that reactivates a particle to a state chosen the given reactivation distribution, typically the current estimate of the conditional occupation probability."
            state = envs[idx_particle].getState()
            if DEBUG_TRAJECTORIES:
                # We can add a `True` condition above in order to show the following useful piece of information of the proportion of particles at the state of interest x is useful for tracking the stabilization of Phi(t,x) around the QSD
                flags_particle_at_terminal_state = [1 if envs[p].getState() in self.env.getTerminalStates() else 0 for p in range(len(envs))]
                print("[reactivate_particle_internal] % particles at terminal states: {:.1f}% ({} out of {})".format(np.mean(flags_particle_at_terminal_state)*100, np.sum(flags_particle_at_terminal_state), len(envs)))
            new_state = reactivate_particle(envs, idx_particle, 0, reactivation_distribution=reactivation_distribution) # (2024/03/05) the third parameter is dummy
            if DEBUG_TRAJECTORIES:
                print("*** t={}: Particle #{} ABSORBED at state={} and REACTIVATED to state {}".format(t, idx_particle, state, new_state))

            return new_state
        #------------------------------- Auxiliary functions ----------------------------------#


        # ---------------------------- Check input parameters ---------------------------------#
        # -- Absorption and activation sets
        # Class
        if not isinstance(absorption_set, set):
            raise ValueError("Parameter `absorption_set` must be a set: {}".format(absorption_set))
        if not isinstance(start_set, set):
            raise ValueError("Parameter `start_set` must be a set: {}".format(start_set))
        if len(start_set) == 0:
            raise ValueError("Parameter `start_set` must have at least one element")

        if expected_absorption_time is None and expected_exit_time is None:
            raise ValueError("Parameter `expected_exit_time` must be provided when `expected_absorption_time` is None")
        # ---------------------------- Check input parameters ---------------------------------#

        N = len(envs)
        if max_time_steps is None:
            max_time_steps = N * 100   # N * "MAX # transitions allowed on average for each particle"
        policy = self.agent.getPolicy()  # Used to define the next action and next state
        learner = self.agent.getLearner()  # Used to learn (or keep learning) the value functions

        # Reset the learner, but WITHOUT resetting the value functions as they were possibly learned a bit during an initial exploration of the environment
        # What is most important of this reset is to reset the learning rates of all states and actions! (so that we start the FV-based learning with full intensity)
        learner.reset(reset_episode=True, reset_value_functions=False)

        # Set the start state of each environment/particle to an activation state, as this is a requirement
        # for the empirical distribution Phi(t).
        for i, env in enumerate(envs):
            # Environment seed
            seed_i = seed + i if seed is not None else None
            env.setSeed(seed_i)

            # Choose start state from the activation set
            start_state = choose_state_from_set(start_set, dist_proba_for_start_state)
            if learner.getLearningTask() == LearningTask.CONTINUING:
                assert start_state not in env.getTerminalStates(), \
                    f"The start state of an FV particle ({start_state}) cannot be a terminal state of the environment (terminal states = {env.getTerminalStates()}) for the CONTINUING learning task context." \
                    f"\nThe reason is that, before getting to a terminal state, there must be at least ONE transition of the particle" \
                    f" --because we need the `info` dictionary to be defined when processing a terminal state."
            env.setState(start_state)

        # Event times: the first event time is 0
        # This implies t=0 is considered the time at which each particle is positioned to their start state.
        event_times = deque([0])

        # Phi(t, z): Empirical probability of all active states z, at each time t when a variation in Phi(t, z) is observed.
        dict_phi = initialize_phi(envs, t=event_times[0], states_of_interest=learner.getActiveSet())

        # Initialize the list of observed survival times to be filled during the simulation below
        survival_times = deque([0])   # We set the first element of the survival times list to 0 so that we can "estimate" the survival probability at 0 (which is always equal to 1 actually)
        has_particle_been_absorbed_once = [False]*N     # List that keeps track of whether each particle has been absorbed once
                                                        # so that we can end the simulation when all particles have been absorbed
                                                        # when the survival probability is estimated by this function.
        has_particle_been_absorbed = [False]*N          # This is used to determine what particles are used to compute the reactivation distribution, namely the non-absorbed ones

        if True or DEBUG_ESTIMATORS:
            print("[DEBUG] @{}".format(get_current_datetime_as_string()))
            print("[DEBUG] State value function at start of simulation:\n\t{}".format(learner.getV().getValues()))
            print("[DEBUG] Action value function at start of simulation:\n\t{}".format(learner.getQ().getValues().reshape(self.env.getNumStates(), self.env.getNumActions())))

        done = False
        t = 0   # IMPORTANT: t represents the time at which a particle transitions to the NEXT state, and this is why we initialize t as 0, so that after the first transition,
                # the time step is t = 1, i.e. t signals the time at which the particle is at the NEXT state, after the transition has taken place.
                # This follows the usual convention for continuous-time Markov processes where the event time is the time at which the Markov chain is at the NEXT state,
                # after the event takes place, i.e. the state to which the Markov chain transitions to.
                # See e.g. Pierre Bremaud's book, Chapter 13, graph of the jump rate process.
        # Initial learning rate for Phi, which is the gamma* defined in Fraiman et al.
        alpha0 = +np.Inf      # Setting this initial learning rate to 1 gives the same weight to all historic (k < t) and current (k = t) Phi values
                        # setting it equal to >1 gives more weight to most recent estimates of Phi,
                        # setting it equal to <1 gives more weight to older estimates of the Phi.
        while not done:
            t += 1
            event_times.append(t)

            # Move all particles at the SAME TIME by following Fraiman's transition probability K(phi) = p(x,y|a) + p(x,A|a) * phi(y)
            # where p(x,y|a) is the transition probability of the original Markov chain given action `a` and phi(y) is the current estimate of the conditional occupation
            # probability at EACH possible new state y.
            for idx_particle in range(N):
                # Get the current state of the particle
                state = envs[idx_particle].getState()
                #print(f"particle {idx_particle} (s={state})", end=", ")

                # Check if the particle is at a terminal state, in which case we need to "restart" the process to an environment's initial state.
                # This is so because FV applies to a process whose transition probability from a terminal state to any initial state is equal to 1,
                # defined like that so that the process can continue indefinitely, as opposed to a process that ends at a terminal state.
                # This is required by FV because FV is used to estimate an expectation of a process running for unlimited time.
                # If the selected state is part of the absorption set, the time to absorption contributes to the estimation of the survival probability P(T>t)
                # --as long as the particle has never been absorbed before-- and the particle is reactivated right-away.
                if state in self.env.getTerminalStates():
                    # Note: The following reset of the environment is typically carried out by the gym module,
                    # e.g. by the toy_text.discrete.DiscreteEnv environment's reset() method where the initial state
                    # is chosen based on the isd attribute of the object, i.e. of the Initial State Distribution defining the initial state.
                    # Note also that the seed for the reset has been set separately for each particle before starting the FV simulation.
                    next_state = envs[idx_particle].reset()
                    reward = envs[idx_particle].getReward(next_state)
                    if DEBUG_TRAJECTORIES:
                        print("___ Particle #{} in terminal state {} REINITIALIZED to environment's start state following its initial state distribution: next_state={}" \
                              .format(state, idx_particle, next_state))

                    # Learn the value functions for the terminal state for the continuing learning task case,
                    # because in that case the value of terminal states is NOT defined as 0.
                    if learner.getLearningTask() == LearningTask.CONTINUING:
                        self.learn_terminal_state_values(t, state, next_state, reward, info)
                            ## Note: the `info` dictionary is guaranteed to be defined thanks to the assertion
                            ## at the initialization of the FV particles that asserts they cannot be at a terminal state.
                            ## Thanks to this condition, when a particle is at a terminal state, it means that it MUST have
                            ## transitioned from a non-terminal state, and this implies that the `info` dictionary has been
                            ## defined as the output of such transition.
                else:
                    # Step on the selected particle
                    action = policy.choose_action(state)
                    next_state, reward, done_episode, info = envs[idx_particle].step(action)
                    if estimated_average_reward is not None:
                        # Store the estimated average reward passed by the user in the `info` dictionary so that it can be used
                        # by the call to the learn() method below as correction value when learning the value functions under the average reward criterion
                        info['average_reward'] = estimated_average_reward

                    # Learn: i.e. update the value function (stored in the learner) with the new observation
                    # NOTES:
                    # 1) Learning ONLY happens when the particle is NOT reactivated.
                    # This makes total sense because reactivation sets the next state of the particle to a state
                    # that is normally not reachable by the underlying Markov chain on which the FV process is built.
                    # 2) We pass `done` to learn() and NOT `done_episode` because we want to update the average reward
                    # over all episodes (set by Learner.store_trajectory_at_episode_end() and retrieved by GenericLearner.getAverageReward())
                    # only when the FV SIMULATION IS OVER, not at the end of each episode. Otherwise, the estimated average
                    # reward will fluctuate a lot (i.e. as the average reward observed by episode fluctuates) and this is NOT what we want,
                    # we want a stable estimate of the average reward over all episodes.
                    # 3) We pass update_phi=False because we only update the occupation probability Phi AFTER all particles have been reactivated, o.w. we would be reactivating
                    # different absorbed particles using *different* estimates of the conditional occupation probability, whereas they must be reactivated using the SAME estimate.
                    # TODO: (2024/01/29) Revise the correct use of the `done` variable here, instead of `done_episode`, because actually when we are done by `done`, this line will NEVER be executed because we will NOT enter again the `while done` loop...
                    learner.learn(t, state, action, next_state, reward, done, info, update_phi=False)

                if next_state in absorption_set:
                    # The particle has been absorbed.
                    # => Add the time to absorption to the set of times used to estimate the survival probability P(T>t) if it's the first absorption of the particle
                    # => Reactivate the particle following the current estimate of the occupation probability to the position of any of the other particles
                    has_particle_been_absorbed[idx_particle] = True

                    # Compute the absorption time as the time at which the particle is at *next_state*, and this is precisely the definition of t, as described above
                    t_abs = t

                    # Contribution to the survival probability
                    if not has_particle_been_absorbed_once[idx_particle]:
                        # The survival time coincides with the absorption time since the particle has never been absorbed before when the program enters this block
                        # Note that, by construction, the survival times are sorted in ascending order.
                        t_surv = t_abs
                        survival_times.append(t_surv)

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

            # Reactivate absorbed particles
            if sum(has_particle_been_absorbed) == N:
                # All particles have been absorbed
                # => Use the distribution of the particles at the previous step as reactivation distribution
                reactivation_probability_distribution = extract_latest_phi_values(dict_phi)
            else:
                # Use the distribution of non-absorbed particles as reactivation distribution
                reactivation_probability_distribution = dict( pd.Series([envs[p].getState() for p, absorbed in enumerate(has_particle_been_absorbed) if not absorbed]).value_counts(normalize=True, sort=False) )

            for idx_particle in range(N):
                next_state = reactivate_particle_internal(idx_particle, reactivation_probability_distribution)
                reward = envs[idx_particle].getReward(next_state)
                if DEBUG_TRAJECTORIES:
                    print(f"--> Reactivated particle {idx_particle} from state {state} to state {next_state}")
                assert next_state not in absorption_set, f"The state of a reactivated particle must NOT be a state in the absorption set ({next_state})"

                if DEBUG_TRAJECTORIES:
                    print("[FV] Moved P={}, t={}: state={}, action={} -> next_state={}, reward={}" \
                          .format(idx_particle, t, state, action, next_state, reward),
                          end="\n")
                    if N <= 10:
                        print(f"Current state of system:\n{[env.getState() for env in envs]}")

            print("t={} of {}: % particles absorbed at least once = {:.1f}% ({} of {})".format(t, max_time_steps, np.mean(has_particle_been_absorbed_once)*100, np.sum(has_particle_been_absorbed_once), N))

            # Update Phi after all particles have been updated
            dict_phi = update_phi_on_all_states(envs, t, dict_phi, alpha=alpha0 / (t + alpha0))

            # CHECK DONE
            # If we want to interrupt the simulation by EITHER reaching the maximum time steps
            # OR
            # having all particles absorbed at least once, which is a logic that impacts ONLY the estimation of the average reward using FV,
            # but it does NOT impact the estimation of the value functions done by the call to learner.learn() inside this loop.
            # THEREFORE, if we are not interested in estimating the average reward using FV, we may want to continue until
            # we reach the maximum number of steps, regardless of the first absorption event of each particle.
            # Use this if we want to stop simulation either when the maximum number of steps has been reached OR all N particles have been absorbed at least once
            done = t >= max_time_steps or sum(has_particle_been_absorbed_once) == N
            # Use this if we want to stop simulation ONLY when the maximum number of steps has been reached.
            # This should be the preferred way if we want to do a FAIR comparison with other benchmark methods,
            # because we guarantee that all time steps specified are used (which normally comes from the FAIR comparison setup)
            # Note that at this point max_time_steps ALWAYS has a value because if `None` was given by the user, a default value is set at the beginning of the method.
            # WARNING: In this case we need to DISABLE the assertion below on the "maximum time stored in Phi(t,x)".
            #done = t >= max_time_steps

        # DONE
        if t >= max_time_steps:
            # Add the last observed time step to the list of survival times as many times as the number of particles left to absorb,
            # so that we can use the signal information collected in Phi since the last absorbed particle to the FV estimation of the expected reward.
            # Note that these survival times are underestimations of the actual survival times, so the FV estimate of the average reward
            # will have a negative bias, but this is a smaller bias than if we didn't consider the last observed time `t`
            # as part of the estimated survival probability function.
            n_particles_not_absorbed = N - sum(has_particle_been_absorbed_once)
            for e in np.repeat(t, n_particles_not_absorbed):
                survival_times.append(e)
            if n_particles_not_absorbed > 0:
                print(f"WARNING: Not all {N} particles were absorbed at least ONCE because the maximum number of time steps for all particles ({max_time_steps} has been reached: # NOT absorbed particles = {n_particles_not_absorbed}")
        # The following assertion should be used ONLY when the FV process stops if all N particles are absorbed at least once before reaching max_time_steps
        assert np.max([np.max(dict_phi[x]['t']) for x in dict_phi.keys()]) <= np.max(survival_times), "The maximum time stored in Phi(t,x) must be at most the maximum observed survival time"
        if show_messages(verbose, verbose_period, t_learn):
            print("==> FV agent ENDS at state {} at discrete time t = {} ({:.1f}% of max_time_steps={}, {:.1f}% of particles were absorbed once),"
                  " compared to maximum observed time for P(T>t) = {:.1f}." \
                    .format(envs[idx_particle].getState(), t, t/max_time_steps*100, max_time_steps, np.sum(has_particle_been_absorbed_once)/N*100, survival_times[-1]))

        # Compute the stationary probability of each state x in Phi(t, x) using Phi(t, x), P(T>t) and E(T_A)
        df_proba_surv = compute_survival_probability(survival_times)
        if expected_absorption_time is None:
            expected_absorption_time = expected_exit_time + np.mean(survival_times)
        max_survival_time = df_proba_surv['t'].iloc[-1]

        if DEBUG_ESTIMATORS:
            max_rows = pd.get_option('display.max_rows')
            pd.set_option('display.max_rows', None)
            print("Survival probability:\n{}".format(df_proba_surv))
            print("Phi:\n{}".format(dict_phi))
            pd.set_option('display.max_rows', max_rows)

        n_events_fv = t*N

        return n_events_fv, learner.getV().getValues(), learner.getQ().getValues(), learner._state_counts, dict_phi, df_proba_surv, expected_absorption_time, max_survival_time

    @measure_exec_time
    def _run_simulation_fv_learnvaluefunctions( self, t_learn, envs, absorption_set: set, start_set: set,
                                                max_time_steps=None, number_free_particles=1,
                                                dist_proba_for_start_state: dict=None,
                                                expected_absorption_time=None, expected_exit_time=None,
                                                estimated_average_reward=None,
                                                seed=None, verbose=False, verbose_period=1,
                                                plot=False, colormap="seismic", pause=0.1):
        """
        Arguments:
        number_free_particles: (opt) int
            Number of normal particles that follow the underlying Markov process and that are free to explore the whole environment,
            which is necessary to be able to learn the value functions for the states in the absorption set A, which are used
            as bootstrap observations when learning the value functions with FV.
            default: 1
        """
        #------------------------------- Auxiliary functions ----------------------------------#
        def choose_particle_to_move(t, N, df_particles_for_start_state, choose_particles_in_order=False):
            """
            Chooses the next particle to move in the FV simulation

            Arguments:
            t: int
                Current system's simulation time.
                Only used when choose_particles_in_order=True.

            N: int
                Total number of particles in the FV particle system.
                Only used when df_particles_for_state is None.

            df_particles_for_start_state: pd.DataFrame
                Data frame containing the particle IDs belonging to each FV particle group that is used to estimate the value functions at the different
                states and state-actions, among which the next particle to move should be selected.
                These IDs should be stored in a column called "idx_particles", and a particle is chosen among ALL those IDs.
                If None, a particle is chosen among the N particles in the FV system.

            choose_particles_in_order: (opt) bool
                Whether to choose the particles to move in order, i.e. when df_particles_for_start_state is None,
                first particle ID 0 is chosen, then particle ID 1, etc. If df_particles_for_start_state is not None, the next particle to move is chosen
                as the `t % len(list-of-particle-IDs-to-choose-from)`-th ID in the list of particle IDs obtained from the concatenation of the particle IDs
                stored in column "idx_particle" of the data frame, in whatever order they happen to be.
                Ex: if df_particles_for_start_state['idx_particles'] = [[0, 2, 5], [1, 3, 4, 6]], the particles are indexed in the order defined by the
                concatenation of the two elements in the above list, i.e.: [0, 2, 5, 1, 3, 4, 6]

            Return: int
            Index of the particle to move next.
            """
            if df_particles_for_start_state is None:
                if choose_particles_in_order:
                    return t % N
                else:
                    return np.random.choice(N)
            else:
                idx_all_particles = np.concatenate(np.array(df_particles_for_start_state['idx_particles'])).astype(int)
                if choose_particles_in_order:
                    return idx_all_particles[ t % len(idx_all_particles) ]
                else:
                    return np.random.choice(idx_all_particles)

        def reactivate_particle_internal(idx_particle):
            """
            Internal function that reactivates a particle until a valid reactivation is obtained
            (i.e. a particle whose state is not in an absorbed set of states.

            (2023/11/15) Note that the reactivated particle can perfectly be at a terminal state, because terminal
            states are NOT part of the absorption set.
            What IS true is that once a particle reaches a terminal state, THE NEXT TIME THE PARTICLE IS PICKED,
            its state is changed to an environment's start state, which COULD be part of the absorption set,
            in which case the particle is reactivated right-away (i.e. this function is called).
            """
            # TODO: (2023/09/06) Try to implement a more efficient way (i.e. without looping) to reactivate the particle to one of the other particles that are NOT at an absorption set
            done_reactivate = False
            state = envs[idx_particle].getState()
            start_state = learner.getStartState(idx_particle)
            all_idx_particles_to_choose_from = None
            if learner.estimate_on_fixed_sample_size:
                # We need to choose the reactivation particle ONLY among the particles in the current FV group
                # (i.e. the group of particles that started at the same start state at which the absorbed particles started)
                all_idx_particles_to_choose_from = df_particles_for_start_state.loc[start_state, 'idx_particles']

            if DEBUG_TRAJECTORIES:
                # We can add a `True` condition above in order to show the following useful piece of information of the proportion of particles at the state of interest x is useful for tracking the stabilization of Phi(t,x) around the QSD
                if all_idx_particles_to_choose_from is None:
                    flags_particle_at_terminal_state = [1 if envs[p].getState() in self.env.getTerminalStates() else 0 for p in range(len(envs))]
                    print("[reactivate_particle_internal] % particles at terminal states: {:.1f}% ({} out of {})" \
                          .format(np.mean(flags_particle_at_terminal_state) * 100, np.sum(flags_particle_at_terminal_state), len(envs)))
                else:
                    flags_particle_at_terminal_state = [1 if envs[p].getState() in self.env.getTerminalStates() else 0 for p in range(len(envs)) if p in all_idx_particles_to_choose_from]
                    print("[reactivate_particle_internal] % particles at terminal states: {:.1f}% ({} out of N(s={}) = {})" \
                          .format(np.mean(flags_particle_at_terminal_state)*100, np.sum(flags_particle_at_terminal_state), start_state, learner.N_for_start_state[start_state]))
            new_state = None

            while not done_reactivate:
                idx_reactivate = reactivate_particle(envs, idx_particle, 0, reactivation_number=None)
                    ## (2023/01/05) the third parameter is dummy when we do NOT use method = ReactivateMethod.VALUE_FUNCTION to reactivate the particle inside function reactivate_particle().
                    ## (2024/01/28) Use reactivation_number=reactivation_number to indicate that we want to use method = ReactivateMethod.ROBINS.
                # TODO: (2023/11/15) Check whether there is any possibility that the particle to which the absorbed particle has been reactivated COULD really be in the absorption set...
                # Note that, at the initial devise of the FV simulation/estimation method, we have considered that the set of FV particles changes with time...
                # But this dynamic set of FV particles might no longer be the case, at the time of this writing (2023/11/15).
                if envs[idx_particle].getState() not in absorption_set:
                    done_reactivate = True
                    new_state = envs[idx_particle].getState()
                    if DEBUG_TRAJECTORIES:
                        print("*** t={}, t_clock={}: Particle #{} ABSORBED at state={} and REACTIVATED to particle {} at state {}" \
                              .format(t, t_clock, idx_particle, state, idx_reactivate, new_state))

            return new_state
        #------------------------------- Auxiliary functions ----------------------------------#


        #---------------------------- Check input parameters ---------------------------------#
        #-- Absorption and activation sets
        # Class
        if not isinstance(absorption_set, set):
            raise ValueError("Parameter `absorption_set` must be a set: {}".format(absorption_set))
        if not isinstance(start_set, set):
            raise ValueError("Parameter `start_set` must be a set: {}".format(start_set))
        if len(start_set) == 0:
            raise ValueError("Parameter `start_set` must have at least one element")

        if expected_absorption_time is None and expected_exit_time is None:
            raise ValueError("Parameter `expected_exit_time` must be provided when `expected_absorption_time` is None")
        #---------------------------- Check input parameters ---------------------------------#


        #---------------------------- Parse input parameters ---------------------------------#
        # Plotting setup
        if plot:
            # Setup the figures that will be updated at every absorption event
            colors = cm.get_cmap(colormap)
            ax_V = plt.figure(figsize=(10, 6)).subplots(1, 1)
            # Plot the true state value function (to have it as a reference already
            ax_V.plot(self.env.getAllStates(), self.env.getV(), '.-', color="blue")
        #---------------------------- Parse input parameters ---------------------------------#

        N = len(envs)
        if max_time_steps is None:
            max_time_steps = N * 100   # N * "MAX # transitions allowed on average for each particle"
        policy = self.agent.getPolicy()  # Used to define the next action and next state
        learner = self.agent.getLearner()  # Used to learn (or keep learning) the value functions

        # Reset the learner, but WITHOUT resetting the value functions as they were possibly learned a bit during an initial exploration of the environment
        # What is most important of this reset is to reset the learning rates of all states and actions! (so that we start the FV-based learning with full intensity)
        learner.reset(reset_episode=True, reset_value_functions=False)

        # Set the start state of each environment/particle to an activation state, as this is a requirement
        # for the empirical distribution Phi(t).
        df_particles_for_start_state = None
        particles_first_action = [-1]*N
        if learner.estimate_on_fixed_sample_size:
            # Create a data frame to store the pieces of information that are necessary to perform the FV estimation of value functions using a fixed sample size of particles
            # defined by the number of particles that start at each possible state and state-actions where the value functions should be estimated.
            # The data frame contains the following information:
            # - t: the simulation time within the group (clock) --> required for the computation of the FV estimator of the value functions
            # - N: the number of particles in the group --> required for the estimation of Phi(t,x) for each group
            # - N_left: the number of particles in the group who haven't yet experienced the first absorption event --> required for the selection of the next particle to move
            #   (i.e. when this number becomes 0, no particle in the group needs to be selected to move, because the update of the FV integral at each absorption time is done
            #   based on the assumption that absorption times are observed in increasing order, and if we keep moving the particles after the last particle in the group has been
            #   absorbed at least once, such assumption on the absorption times would no longer be true).
            # - idx_particles: the IDs of the particles belonging to the group --> required for the reactivation step, as the reactivation particle must be chosen ONLY among the particles in the group to which the absorbed particle belongs
            #
            # Note the following characteristics about this data frame:
            # - It is indexed by the start state so that we can quickly access the information for each group.
            # - It is updated as the simulation proceeds so that it only contains the rows for the start states on which the FV estimation should still proceed.
            #   The FV estimation of a start state stops when all the associated N(s) particles have been absorbed at least once.
            # - When the number of rows in the data frame becomes 0, the simulation stops because it means that all FV particles in the system have been absorbed at least once.
            df_particles_for_start_state = pd.DataFrame([(0, 0, 0, [])], columns=['t', 'N', 'N_left', 'idx_particles'], index=learner.getActiveSet())

            # Create a list where we store the clock for each particle, which is needed for the estimation of the survival probability P(T>t) for each start state
            particle_times = [0]*N
        for p, env in enumerate(envs):
            # Environment seed
            seed_i = seed + p if seed is not None else None
            env.setSeed(seed_i)

            # Choose start state from the set of start states given
            start_state = choose_state_from_set(start_set, dist_proba_for_start_state)
            env.setState(start_state)
            # We must store the trajectory of the particle in the environment representing the particle because we need to retrieve the particle's trajectory
            # before every absorption, so that we can estimate the survival probability and the conditional occupation probability that are functions of the start state-action.
            env.setStoreTrajectoryFlag(True)
            # The trajectory is reset to one record with time t=0 and both state and next_state equal to the start state of the environment set above
            env.reset_trajectory()

            # We set the start time of the simulation to the time of the first record stored in the environments representing each particle
            # i.e. the time where each particle is positioned at their respective start states.
            # Note that this time is the same for all particles, it is defined in the EnvironmentDiscrete.reset_trajectory() method as the time of the first record
            # in the stored trajectory.
            start_time = env.getTrajectory()['time'].iloc[0]

            if learner.estimate_on_fixed_sample_size:
                #-- Information about each group of particles
                # Update the information associated to the selected start state in the data frame keeping track of each N(s) group
                _row = df_particles_for_start_state.loc[start_state]
                df_particles_for_start_state.loc[start_state, :] = pd.Series([start_time, _row['N'] + 1, _row['N'] + 1, _row['idx_particles'] + [p]], index=df_particles_for_start_state.columns)

                #-- Information about each particle
                # Choose the start action for the particle so that we can store it right now and use it
                # when updating Q(s,a) when an absorption of a particle occurs, for which we need to know how many particles are in the group of same start (s,a)
                # Note that we select the action randomly because we want to have as many actions covered as possible
                start_action = np.random.choice(self.env.getNumActions())
                particles_first_action[p] = start_action
                learner.setStartStateAction(p, start_state, start_action)
        # Store the start states in the learner, so that we know the number of particles that started at each environment's state when updating Phi(t,x; s)
        N_never_moved = 0
        if learner.estimate_on_fixed_sample_size:
            # Take care of the case when N(s) = 1 (which is not allowed, as N(s) should be > 1...)
            # We remove the start states s having N(s) = 1 from the data frame keeping track of the number of particles left per group, so that those particles are not picked by the simulation
            ind_valid_groups = df_particles_for_start_state['N'] > 1
            df_particles_for_start_state = df_particles_for_start_state.loc[ind_valid_groups, :]
            N_never_moved = N - sum(df_particles_for_start_state['N'])
            if sum(~ind_valid_groups) == 0:
                assert N_never_moved == 0
            else:
                print(f"A total of {N_never_moved} particles out of {N} will never be moved because they belong to groups of size N(s) = 1")
        print(f"df_particles_for_start_state:\n{df_particles_for_start_state}")

        # Particles that follow the underlying Markov chain and that are used to learn the state and action values inside A
        # while learning the state and action values outside A
        # Initially there are no particles like this but they will start being created as FV particles are absorbed for the first time
        # for a maximum of n_normal_max.
        # NOTE that if no particles are used here, the value of the states in A cannot leverage the information about the rare rewards
        # brought in by the FV-based estimation of the value of the states OUTSIDE A.
        envs_normal = []
        n_normal_max = number_free_particles  # Maximum number of \particles used to explore the whole environment

        # Event times: the first event time is 0
        event_times = [start_time]

        # Initialize the list of observed first-time (for each particle) survival times to be filled during the simulation below
        # These survival times are used to estimate the survival probability that is used to estimate the stationary probability of each state of interest x tracked by Phi(t,x)
        # We initialize the first element as 0 so that the estimated survival probability starts at 1.0 (i.e. P(T>0) = 1.0)
        first_survival_times = [0]    # We set the first element of the survival times list to 0 so that we can "estimate" the survival probability at 0 (which is always equal to 1 actually)
        idx_reactivate = None   # This is only needed when we want to plot a vertical line in the particle evolution plot with the color of the particle to which an absorbed particle is reactivated
        has_particle_been_absorbed_once = [False]*N     # List that keeps track of whether each particle has been absorbed once,
                                                        # so that we can end the simulation when all particles have been absorbed
                                                        # when the survival probability is estimated by this function.
        has_particle_been_selected_once = [False]*N     # List that keeps track of whether each particle has been selected for movement at least once,
                                                        # so that we can choose the already selected start action when learning by groups

        if True or DEBUG_ESTIMATORS:
            print("[DEBUG] @{}".format(get_current_datetime_as_string()))
            print("[DEBUG] State value function at start of simulation:\n\t{}".format(learner.getV().getValues()))
            print("[DEBUG] Action value function at start of simulation:\n\t{}".format(learner.getQ().getValues().reshape(self.env.getNumStates(), self.env.getNumActions())))

        idx_particle = -1
        done = False
        t = start_time  # IMPORTANT: t represents the time at which a particle transitions to the NEXT state
                        # This follows the usual convention for continuous-time Markov processes where the event time is the time at which the Markov chain is at the NEXT state,
                        # after the event takes place, i.e. the state to which the Markov chain transitions to.
                        # See e.g. Pierre Bremaud's book, Chapter 13, graph of the jump rate process.
        # Number of total calls to the env.step() method done when running the simulation below
        # (this includes the FV particles and the "normal" particles which are free to explore the whole environment)
        # This is used to count the number of total events used by the simulation which can then be used to design experiments where the comparison among learning methods is fair.
        n_steps_on_all_environments = 0
        n_absorptions = 0
        info = {}  # We need this variable to be defined when calling learner.learn() for the first time if the first particle picked for moving has started at a terminal state
        while not done:
            t += 1

            # Clock to use in the FV estimation of the value functions
            # In the fixed-sample-size estimation mode, this clock is the internal clock of the group to which the particle that is moved belongs to.
            t_clock = t

            if show_messages(verbose, verbose_period, t):
                print("@{}".format(get_current_datetime_as_string()))
                print(f"FV time step t={t} of max_time_steps={max_time_steps} running...")
                print(f"# FV particles absorbed at least once: {sum(has_particle_been_absorbed_once)} of max N={N}")
                print(f"# particles exploring absorption set A: {len(envs_normal)} of max M={n_normal_max}")

            # We define a reactivation number, an integer between 0 and N-2 which is used to deterministically choose
            # the reactivation particle (if reactivation is via ReactivateMethod.ROBINS), in order to save time by not having to generate a uniform random number.
            # Note that the range from 0 to N-2 allows us to choose one of the N-1 particles to which the absorbed particle can be reactivated into.
            reactivation_number = t % (N - 1)

            event_times += [t]

            # Select the particle to move
            assert df_particles_for_start_state is None or len(df_particles_for_start_state) > 0, \
                "In the group-based FV simulation, there must be at least ONE group of particles who still should be moved. Otherwise, the simulation should have stopped."
            idx_particle = choose_particle_to_move(t, N, df_particles_for_start_state, choose_particles_in_order=False)

            # Update the clock of the group to which the selected particle belongs
            # Note that we always (regardless of the value of attribute estimate_on_fixed_sample_size) store the start state of the particle being moved
            # and the current value of V(s) of such start state (which is the value that is updated by the FV process carried out here)
            # because these pieces of information are used in plots (regardless of the value of attribute estimate_on_fixed_sample_size).
            start_state = learner.getStartState(idx_particle)
            V_start_prev = learner.getV().getValue(start_state)
            if learner.estimate_on_fixed_sample_size:
                # Update the particle's clock (used in the estimation of the survival probability P(T>t) for each start state)
                particle_times[idx_particle] += 1

                # Update the group's clock (used in the estimation of the conditional occupation probability Phi(t,x) for each start state
                df_particles_for_start_state.loc[start_state, 't'] += 1
                t_clock = df_particles_for_start_state.loc[start_state, 't']

            # Get the current state of the selected particle because that's the particle whose state is going to (possibly) change
            state = envs[idx_particle].getState()
            V_state_prev = learner.getV().getValue(state)  # Store the current value of V(s) as PREVIOUS value for informative plotting purposes (title)
            #print(f"particle {idx_particle} (s={state})", end=", ")

            # Check if the particle that was picked for update is CURRENTLY at a terminal state,
            # in which case we "restart" the process to an environment's initial state (according to its initial state distribution).
            # This is so because FV applies to a process whose transition probability from a terminal state to any initial state is equal to 1,
            # which is defined like that so that the process can continue indefinitely, as opposed to a process that ends at a terminal state.
            # This is required by FV because FV is used to estimate an expectation of a process running for unlimited time.
            # If the selected state is part of the absorption set, the time to absorption contributes to the estimation of the survival probability P(T>t)
            # --as long as the particle has never been absorbed before-- and the particle is reactivated right-away.
            if state in self.env.getTerminalStates():
                # Note: The following reset of the environment is typically carried out by the gym module,
                # e.g. by the toy_text.discrete.DiscreteEnv environment's reset() method where the initial state
                # is chosen based on the isd attribute of the object, i.e. of the Initial State Distribution defining the initial state.
                # Note also that the seed for the reset has been set separately for each particle before starting the FV simulation.
                next_state = envs[idx_particle].reset()
                reward = envs[idx_particle].getReward(next_state)
                if DEBUG_TRAJECTORIES:
                    print("___ Particle #{} in terminal state {} REINITIALIZED to environment's start state following its initial state distribution: next_state={}" \
                          .format(idx_particle, state, next_state))

                # Learn the value functions for the terminal state for the continuing learning task case,
                # because in that case the value of terminal states is NOT defined as 0.
                # Note that the transition from the terminal state to the start state is stored as part of the particle's trajectory
                # (because we are passing env=envs[idx_particle] as parameter to learn_terminal_state_values()).
                if learner.getLearningTask() == LearningTask.CONTINUING:
                    self.learn_terminal_state_values(t_clock, state, next_state, reward, info, envs=envs, idx_particle=idx_particle, update_phi=True)
            else:
                # Step on the selected particle
                #print(f"\n[debug] Moving particle #{idx_particle}...")
                if not learner.estimate_on_fixed_sample_size or \
                   learner.estimate_on_fixed_sample_size and has_particle_been_selected_once[idx_particle]:
                    # Normal selection of action, based on the policy
                    action = policy.choose_action(state)
                else:
                    # This is the first movement of the particle
                    # => Perform the FIRST action that has already been chosen for the particle at the beginning
                    action = particles_first_action[idx_particle]
                    has_particle_been_selected_once[idx_particle] = True
                next_state, reward, done_episode, info = envs[idx_particle].step(action)
                n_steps_on_all_environments += 1
                if estimated_average_reward is not None:
                    # Store the estimated average reward passed by the user in the `info` dictionary so that it can be used
                    # by the call to the learn() method below as correction value when learning the value functions under the average reward criterion
                    info['average_reward'] = estimated_average_reward
                # Learn using the TD learner
                # IMPORTANT: This learning step should come BEFORE the learning step that takes place at the absorption time of a particle
                # because the learning at absorption relies on the whole trajectory of the absorbed particle to have been stored in the environment
                # associated to the particle, from which the states and actions to update are retrieved.
                # NOTES:
                # 1) This learning step learns the value functions using TD(0), which makes sense because:
                #   a) TD(0) only uses the information of the value functions at two states, namely those involved at the latest transition of the process: state -> next_state
                #   b) The next_state is NEVER a reactivation state, i.e. it is not the result of reactivating an absorbed particle in the set of active states, which would
                #   invalidate learning with TD(0) because such transition is generally not a valid transition of the original Markov process whose value functions we are learning.
                # 2) We pass `done` to learn() and NOT `done_episode` because we want to update the average reward
                # over all episodes (set by Learner.store_trajectory_at_episode_end() and retrieved by GenericLearner.getAverageReward())
                # only when the FV SIMULATION IS OVER, not at the end of each episode. Otherwise, the estimated average
                # reward will fluctuate a lot (i.e. as the average reward observed by episode fluctuates) and this is NOT what we want,
                # we want a stable estimate of the average reward over all episodes.
                # 3) We MAY ask to NOT update the state counts at this learning step because we want to update them only when FV learning takes place
                # i.e. at an absorption time. Otherwise, we risk that the learning rate alpha will be too small when performing the FV learning of the states outside A.
                # Recall that here the `state` value is a state in the active set of states, excluding terminal states, because when a terminal state is visited, the environment
                # is reset to a start state in the IF block above.
                # TODO: (2024/01/29) Revise the correct use of the `done` variable here, instead of `done_episode`, because actually when we are done by `done`, this line will NEVER be executed because we will NOT enter again the `while done` loop...
                assert state in learner.getActiveSet()
                if True or df_particles_for_start_state is None or state not in df_particles_for_start_state.index:
                    info['update_counts'] = True #True #False
                    learner.learn(t_clock, state, action, next_state, reward, done, info, envs=envs, idx_particle=idx_particle, update_phi=True)
                    info.pop('update_counts')

            # Step on all the "normal" particles that have been created for the exploration of the system following the dynamics of the *underlying* Markov chain
            for idx_env, env in enumerate(envs_normal):
                #print(f"\n[debug] Moving normal particle #{idx_env}...")
                state_normal = env.getState()
                action_normal = policy.choose_action(state_normal)
                next_state_normal, reward_normal, done_normal, info_normal = env.step(action_normal)
                n_steps_on_all_environments += 1
                # We may ask to NOT update the state counts when learning from the superclass learner (called next)
                # for the states that are NOT in the absorption set A, because we don't want the learning rates for those states (adjusted by the state counts as alpha0 / count)
                # to be updated now as we risk that they become too small when learning the values of those states by FV when an absorption event occurs
                # (see the learn_at_absorption() method), and we want to learn the value of the states outside A using the FV estimator,
                # not from the exploration of the underlying Markov process carried out here by these "normal" particles.
                # It is worth noting that state counts are stored in the learner NOT in the environment associated to the particle being updated here)
                info_normal['update_counts'] = True #state_normal not in df_particles_for_start_state.index if df_particles_for_start_state is not None else state_normal in absorption_set #True
                learner.learn(t, state_normal, action_normal, next_state_normal, reward_normal, done_normal, info_normal, envs=envs_normal, idx_particle=idx_env, update_phi=False)
                if done_normal and learner.getLearningTask() == LearningTask.CONTINUING:
                    # Go to an environment's start state and learn the value of the terminal state
                    reset_state_normal = env.reset()
                    reward_normal = env.getReward(reset_state_normal)
                    self.learn_terminal_state_values(t, next_state_normal, reset_state_normal, reward_normal, info_normal)

            if next_state in absorption_set:
                # The particle has been absorbed.
                # => 1) Deal with the absorption event, meaning that:
                #   a) The absorption time is added to the list of survival times used to estimate the survival probability P(T>t)
                #   (used in the estimation of the stationary probabilities), if it is the first absorption of the particle.
                #   b) The absorption event is used to learn the value functions of all the states and actions visited by the particle before absorption,
                #   AND to update Phi(t,x) on all states of interest x. NOTE that the update of Phi requires the knowledge of the new position of the particle
                #   AFTER reactivation, as reactivation happens instantaneously! Hence we should FIRST reactivate the particle and then call the learner
                #   of the value functions (normally done by the LeaFV.learn_at_absorption() method).
                # => 2) Reactivate the particle.
                n_absorptions += 1

                # Record the absorption state (used when learning the value functions)
                state_absorption = next_state

                # Reactivate the particle
                # This MUST come before the learning of the value functions because such learning also UPDATES Phi(t,x) based on the reactivated position of the particle.
                next_state = reactivate_particle_internal(idx_particle)
                reward = envs[idx_particle].getReward(next_state)
                if DEBUG_TRAJECTORIES:
                    print(f"--> Reactivated particle #{idx_particle} from state {state} (prior to absorption) to state {next_state}")
                assert next_state not in absorption_set, \
                    f"The state of a reactivated particle must NOT be a state in the absorption set ({next_state})"

                # Learn the value functions of all the states and actions that has been visited by the particle being absorbed.
                # Note that this includes a terminal state if before absorption the particle was at a terminal state,
                # which happens whenever the selected start state after the environment's reset is in the absorption set A.
                # Note that we do this at EVERY absorption event, not only the first absorption of the particle because in any case
                # we cannot leverage the ordering of the first absorption times since the survival time to consider
                # in the estimation of each survival probability contributing to the learning of the value functions
                # depends on the state and action whose value is learned.
                # The value of Phi(t,x) is also updated for every state of interest x, based on the particle moving from state_absorption
                # to the reactivated state stored in `next_state`.
                if learner.estimate_on_fixed_sample_size:
                    # Case when the survival function's clock is the same as the clock for Phi
                    # (note that the third and fourth arguments have the same value)
                    learner.learn_at_absorption_fixed_sample_size(envs, idx_particle, t_clock - start_time, t_clock - start_time, state_absorption, next_state)
                    # Use the following when there is a clock for the particle (used for the estimation of P(T>t)) and a clock for the system (use for the estimation of Phi(t,x))
                    # (2024/03/04) Note that this is currently commented out because, when the estimation of P(T>t) follows the particle's clock, the absorption times are NOT
                    # necessarily observed in order and this complicates a lot the update of the FV "integral". Since we are now going to switch the learning process to the
                    # average reward criterion context, I don't develop this further now, only will do so if necessary in the future.
                    #learner.learn_at_absorption_fixed_sample_size(envs, idx_particle, particle_times[idx_particle], t_clock - start_time, state_absorption, next_state)
                else:
                    # The time parameter should be the absolute system time
                    learner.learn_at_absorption(envs, idx_particle, t, state_absorption, next_state)

                # Collect the new survival time, if it's the first observed survival time
                if not has_particle_been_absorbed_once[idx_particle]:
                    first_survival_times += [t_clock - start_time]
                        ## Notes:
                        ## 1) By construction the first survival time is equal to the time elapsed between the simulation's start time
                        ## and the absorption time, since the start since this is the first time the particle has been absorbed.
                        ## 2) The absorption time is the time at which the particle is at *next_state* and this is precisely the definition of t, as described above.
                        ##  Therefore, to get the survival time, we simply subtract the simulation's start time from the absorption time.
                        ##  We can also quickly check this by considering the situation where the first particle moved at the start of the simulation and
                        ##  is absorbed rightaway: in such scenario we need to store 1 as survival time because we want to state that it took one step
                        ##  for the particle to be absorbed; since, in the simplest case of one group of FV particles, t_clock = t and since t is initialized at start_time
                        ##  and at the very beginning of the `while not done` loop t is increased by 1, t will have the value start_time + 1 at the occurrence of the absorption,
                        ##  hence the value computed above `t_clock - start_time` gives precisely 1, the survival time of the particle in this example.

                    # Mark the particle as "absorbed once" so that we don't use any other absorption time from this
                    # particle to estimate the survival probability, because the absorption times coming after the first
                    # absorption time should NOT contribute to the survival probability, because their initial state
                    # is NOT the correct one --i.e. it is not a state in the activation set
                    # (because the particle has been reactivated to any other state)
                    # which is a requirement for the estimation of the survival probability distribution.
                    has_particle_been_absorbed_once[idx_particle] = True

                    if learner.estimate_on_fixed_sample_size:
                        # Decrease the number of particles left for the group N(s) associated to the start state of the currently updated particle
                        df_particles_for_start_state.loc[start_state, 'N_left'] -= 1
                        assert df_particles_for_start_state.loc[start_state, 'N_left'] >= 0
                        if df_particles_for_start_state.loc[start_state, 'N_left'] == 0:
                            # When all particles in a given N(s) group have been absorbed at least once,
                            # we should NO longer choose any particle from tha group for next particle to move
                            # => Remove the group from the data frame that keeps relevant information of each group of particles
                            df_particles_for_start_state.drop(start_state, inplace=True)

                    if False:
                        print("First-time survival times observed so far: {}".format(sum(has_particle_been_absorbed_once)))
                        print(first_survival_times)

                    # Add a new particle to the list of "normal" particles, i.e. those that evolve following the dynamics of the underlying Markov chain,
                    # so that the state and action values at the states *in* the absorption set A can also be learned iteratively,
                    # leveraging the FV estimation of the state and action values of the states *outside* A.
                    if len(envs_normal) < n_normal_max:
                        envs_normal += [copy.deepcopy(self.env)]
                        # Set the seed for the new environment, o.w. all the environments will have the same seed as self.env whose value was set already abov e!
                        envs_normal[-1].setSeed(int(seed + t))
                    #print(f"\n[debug] New NORMAL particle created after first absorption of particle #{idx_particle}: {len(envs_normal)} in total\n")

                    if plot:
                        # Plot the updated V(s) estimate
                        ax_V.plot(self.env.getAllStates(), learner.getV().getValues(), linewidth=0.5, color=colors(n_absorptions / 255))
                        if False:
                            # Add the state counts
                            ax_count = ax_V.twinx()
                            ax_count.bar(self.env.getAllStates(), learner.getStateCounts(), color="blue", alpha=0.3)
                            plt.sca(ax_V)  # Go back to the primary axis
                        plt.title(f"V(s) evolution (blueish: initial, reddish: final)"
                                  f"\nSystem's time t = {t}, Abs. no. = {n_absorptions} (# first abs. = {sum(has_particle_been_absorbed_once)} of max N = {N}), start_state = {start_state}"
                                  f"\nfrom state = {state} --> absorption state = {state_absorption} (V(s)={np.round(learner.getV().getValue(state_absorption), 3)}) --> reactivation state = {next_state}"
                                  f"\nV(start={start_state}) = {np.round(V_start_prev, 3)}) --> V(start={start_state}) = {np.round(learner.getV().getValue(start_state), 3)} (delta(V) = {np.round(learner.getV().getValue(start_state) - V_start_prev, 3)}, {np.round((learner.getV().getValue(start_state) - V_start_prev) / max(1, V_start_prev) * 100, 1)}%)"
                                  f"\nCount[s] = {learner.getStateCounts()}, alpha[s] = {learner.getAlphasByState()}")
                        if pause == +np.Inf:
                            input("Press ENTER to continue...")
                        else:
                            plt.pause(min(10*pause, 1))    # A call to pause() is essential if we want the plot to be updated by plt.draw()!! (amazing...)
                        plt.draw()

            if plot and show_messages(True, verbose_period, t):
                # next_state is NOT an absorption state
                # => Plot the updated V(s) estimate showing the TD learning step just performed
                ax_V.plot(self.env.getAllStates(), learner.getV().getValues(), linewidth=0.5, color=colors(n_absorptions / 255))
                if False:
                    # Add the state counts
                    ax_count = ax_V.twinx()
                    ax_count.bar(self.env.getAllStates(), learner.getStateCounts(), color="blue", alpha=0.3)
                    plt.sca(ax_V)  # Go back to the primary axis
                plt.title(f"V(s) evolution (blueish: initial, reddish: final)"
                          f"\nSystem's time t = {t} of MAX {max_time_steps} (# first abs. = {sum(has_particle_been_absorbed_once)} of max N = {N}), start_state = {start_state}"
                          f"\nfrom state = {state} --> next state = {next_state} (V(s)={np.round(learner.getV().getValue(next_state), 3)})"
                          f"\nV(state={state}) = {np.round(V_state_prev, 3)} --> V(state={state}) = {np.round(learner.getV().getValue(state), 3)} (delta(V) = {np.round(learner.getV().getValue(state) - V_state_prev, 3)}, {np.round((learner.getV().getValue(state) - V_state_prev) / max(1, V_state_prev) * 100, 1)}%)"
                          f"\nCount[s] = {learner.getStateCounts()}, alpha[s] = {learner.getAlphasByState()}")
                if pause == +np.Inf:
                    input("Press ENTER to continue...")
                else:
                    plt.pause(pause)    # A call to pause() is essential if we want the plot to be updated by plt.draw()!! (amazing...)
                plt.draw()

            if False or DEBUG_TRAJECTORIES:
                print("[FV] Moved P={} (start state={}), t={} (t_clock={}): state={}, action={} -> next_state={}, reward={}" \
                      .format(idx_particle, start_state, t, t_clock, state, action, next_state, reward),
                      end="\n")
                print(f"Particles data frame:\n{df_particles_for_start_state}")

            idx_reactivate = None

            # CHECK DONE
            # If we want to interrupt the simulation by EITHER reaching the maximum time steps
            # OR
            # having all particles absorbed at least once, which is a logic that impacts ONLY the estimation of the average reward using FV,
            # but it does NOT impact the estimation of the value functions done by the call to learner.learn() inside this loop.
            # THEREFORE, if we are not interested in estimating the average reward using FV, we may want to continue until
            # we reach the maximum number of steps, regardless of the first absorption event of each particle.
            # Use this if we want to stop simulation either when the maximum number of steps has been reached OR all N particles have been absorbed at least once
            done = t >= max_time_steps or sum(has_particle_been_absorbed_once) == N - N_never_moved
            # Use this if we want to stop simulation ONLY when the maximum number of steps has been reached.
            # This should be the preferred way if we want to do a FAIR comparison with other benchmark methods,
            # because we guarantee that all time steps specified are used (which normally comes from the FAIR comparison setup)
            # Note that at this point max_time_steps ALWAYS has a value because if `None` was given by the user, a default value is set at the beginning of the method.
            # WARNING: In this case we need to DISABLE the assertion below on the "maximum time stored in Phi(t,x)".
            #done = t >= max_time_steps

        # DONE
        if t >= max_time_steps:
            # Add the last observed time step to the list of first-time survival times as many times as the number of particles left to absorb,
            # so that we can use the signal information collected in Phi since the last absorbed particle to the FV estimation of the expected reward.
            # Note that these survival times are underestimations of the actual survival times, so the FV estimate of the average reward
            # will have a negative bias, but this is a smaller bias than if we didn't consider the last observed time `t`
            # as part of the estimated survival probability function.
            n_particles_not_absorbed = N - N_never_moved - sum(has_particle_been_absorbed_once)
            first_survival_times += list(np.repeat(t, n_particles_not_absorbed))
            if n_particles_not_absorbed > 0:
                print(f"WARNING: Not all {N - N_never_moved} particles were absorbed at least ONCE because the maximum number of time steps for all particles ({max_time_steps}) has been reached: # NOT absorbed particles = {n_particles_not_absorbed}")
        # The following assertion should be used ONLY when the FV process stops if all N particles are absorbed at least once before reaching max_time_steps
        assert np.max([np.max(learner.dict_phi[x]['t']) for x in learner.dict_phi.keys()]) <= np.max(first_survival_times), \
                "The maximum time stored in Phi(t,x) must be at most the maximum observed survival time"
        if show_messages(verbose, verbose_period, t_learn):
            print("==> FV agent ENDS at state {} at discrete time t = {} ({:.1f}% of max_time_steps={}, {:.1f}% of particles were absorbed once),"
                  " compared to maximum observed time for P(T>t) = {:.1f}." \
                    .format(envs[idx_particle].getState(), t, t/max_time_steps*100, max_time_steps, np.sum(has_particle_been_absorbed_once)/N*100, first_survival_times[-1]))

        # Compute the stationary probability of each state x in Phi(t, x) using Phi(t, x), P(T>t) and E(T_A)
        df_proba_surv = compute_survival_probability(first_survival_times)
        if expected_absorption_time is None:
            expected_absorption_time = expected_exit_time + np.mean(first_survival_times)
        max_survival_time = df_proba_surv['t'].iloc[-1]

        if False or DEBUG_ESTIMATORS:
            max_rows = pd.get_option('display.max_rows')
            pd.set_option('display.max_rows', None)
            print("Survival probability:\n{}".format(df_proba_surv))
            print("Phi:\n{}".format(learner.dict_phi))
            pd.set_option('display.max_rows', max_rows)

            # Make a plot
            plt.figure()
            plt.step(df_proba_surv['t'], df_proba_surv['P(T>t)'], color="blue", where='post')
            for x in learner.dict_phi.keys():
                plt.step(learner.dict_phi[x]['t'], learner.dict_phi[x]['Phi'], color="red", where='post')
                plt.title(f"P(T>t) (blue) and Phi(t,x) (red) for state x = {x}")

        print(f"Distribution of start actions:\n{pd.Series([learner.getStartAction(idx_particle) for idx_particle in range(N)]).value_counts()}")

        return n_steps_on_all_environments, learner.getV().getValues(), learner.getQ().getValues(), learner.getStateCounts(), learner.dict_phi, df_proba_surv, expected_absorption_time, max_survival_time

    def _run_single(self, nepisodes, max_time_steps=+np.Inf, max_time_steps_per_episode=+np.Inf, start_state_first_episode=None, reset_value_functions=True,
                    seed=None, compute_rmse=False, weights_rmse=None,
                    state_observe=None,
                    verbose=False, verbose_period=1, verbose_convergence=False,
                    plot=False, colormap="seismic", pause=0.1):
        # TODO: (2020/04/11) Convert the plotting parameters to a dictionary named plot_options or similar.
        # The goal is to group OPTIONAL parameters by their function/concept.
        """
        Runs an episodic Reinforcement Learning experiment on a single Markov chain.

        Parameters:
        nepisodes: int
            Length of the experiment: number of episodes to run.
            This value may be overridden if parameter `max_time_steps` has a finite value.

        max_time_steps: (opt) int
            Number of steps to run.
            When this parameter has a finite value, the number of episodes does not define the end of the simulation,
            but this parameter does.
            default: np.Inf

        max_time_steps_per_episode: (opt) int
            Maximum number of steps to run each episode for.
            default: np.Inf

        start_state_first_episode: (opt) int
            Index in the set of states defined by the environment corresponding to the state to start
            the very first episode.
            All subsequent episodes are started according to the initial state distribution (isd)
            stored in the environment object.
            default: None

        reset_value_functions: (opt) bool
            Whether value function estimates, possibly stored in the value functions learner, should be reset at the beginning of the simulation.
            default: True

        seed: (opt) int
            Seed to use for the random number generator for the simulation.
            default: None

        compute_rmse: (opt) bool
            Whether to compute the RMSE of the estimated value function over all states.
            Useful to analyze rate of convergence of the estimates when the true value function is known.

        weights_rmse: (opt) bool or array
            Whether to use weights when computing the RMSE or the weights to use.
            default: None

        state_observe: (opt) int
            A state index whose RMSE should be observed as the episode progresses.
            Only used when compute_rmse = True
            default: None

        verbose: (opt) bool
            Whether to show the experiment that is being run and the episodes for each experiment.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        verbose_convergence: (opt) bool
            Whether to monitor convergence by showing the change in value function estimate at each new episode
            w.r.t. the estimate in the previous episode.
            If True, messages are shown for EVERY episode with information about:
            - mean|V| previous episode
            - mean|V| current episode
            - mean|delta(V)|
            - mean|delta_rel(V)|, where delta_rel(V) is computed by dividing delta(V)(s) by |V(s)| if not 0 or mean|V| if V(s) = 0
            default: False

        plot: (opt) bool
            Whether to generate plots showing the evolution of the value function estimates.

        colormap: (opt) str
            Name of the colormap to use in the generation of the animated plots
            showing the evolution of the value function estimates.
            It must be a valid colormap among those available in the matplotlib.cm module.
            default: seismic, a colormap that ranges from blue to red, where the middle is white

        pause: (opt) float
            Number of seconds to wait before updating the plot that shows the evolution of the value function estimates.
            It should be positive if we want the plot to be updated every time the value function estimate is updated.
            default: 0.1

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
        """
        #--- Parse input parameters
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
            elif not (is_integer(state_observe) and 0 <= state_observe and state_observe < self.env.getNumStates()):
                warnings.warn("The `state_observe` parameter ({}, type={}) must be an integer number between 0 and {}.\n" \
                              "The state whose index falls in the middle of the state space will be observed." \
                              .format(state_observe, type(state_observe), self.env.getNumStates()-1))
                state_observe = self.env.getNumStates() // 2

        # Plotting setup
        if plot:
            # Setup the figures that will be updated at every verbose_period
            colors = cm.get_cmap(colormap, lut=nepisodes)
            # 1D plot (even for 2D environments)
            fig_V = plt.figure()
            # Plot the true state value function (to have it as a reference already
            plt.plot(self.env.getAllStates(), self.env.getV(), '.-', color="blue")
            if self.env.getDimension() == 2:
                # 2D plot
                fig_V2 = plt.figure()
            if state_observe is not None:
                # Plot with the evolution of the V estimate and its error
                fig_RMSE_state = plt.figure()
        #--- Parse input parameters

        # Define the policy and the learner
        policy = self.agent.getPolicy()
        learner = self.agent.getLearner()

        # Reset the learner (i.e. prepare it for a fresh new learning experience with all learning memory erased and learning rates reset to their initial values)
        # Note that a special treatment may be granted to the reset of the value functions because we may want NOT to reset them,
        # for instance when we are learning a policy and we use this simulator to learn the value functions...
        # In that case, we don't want to start off at 0.0 again but to start off at the estimated values obtained
        # under the previous policy, which normally is very close to the new policy after one step of policy learning.
        # (In some situations --e.g. labyrinth where the policy is learned using Actor-Critic or policy gradient learning--
        # I observed (Nov-2023) that non-optimal policies are learned that minimize the loss function if we reset the
        # value functions to 0 at every policy learning step, while the problem does NOT happen when the value functions are NOT reset.)
        learner.reset(reset_episode=True, reset_value_functions=reset_value_functions)

        # Environment seed
        # We only set the seed when it is not None because when this method is called by the simulate() method,
        # seed is set to None in order to avoid having each experiment (i.e. each replication) produce the same results
        # (which would certainly invalidate the replications!). In that case, the environment's seed is set *before*
        # calling this method run() and we don't want to revert that seed setting, o.w. the experiments repeatability
        # would be broken.
        if seed is not None:
            self.env.setSeed(seed)

        # Store initial values used in the analysis of all the episodes run
        V_state_observe, RMSE, MAPE, ntimes_rmse_inside_ci95 = self.initialize_run_with_learner_status(nepisodes, learner, compute_rmse, weights, state_observe)

        # Initial state value function
        V = learner.getV().getValues()
        Q = learner.getQ().getValues()
        if verbose:
            print("Value functions at start of experiment:")
            print("V = {}".format(V))
            print("Q = {}".format(Q.reshape(self.env.getNumStates(), self.env.getNumActions())))

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
        max_time_steps_reached = False  # Flags whether the TOTAL number of steps reaches the maximum number of steps allowed over all episodes (so that we can break the FOR loop on episodes if that happens)
        # Note: The terminal state at which the previous episode ended is used ONLY in the average reward learning criterion
        # because we need to update the value of the terminal state every time it is visited,
        # as the average reward criterion implies a continuous learning task (as opposed to an episodic learning task)
        # and in continuous learning tasks "terminal" states (i.e. states that are terminal in the original environment
        # whose value functions are learned with an episodic learning task) do NOT necessarily have value 0,
        # their value needs to be estimated as well.
        terminal_state_previous_episode = None
        t = -1                  # Learning step counter
        episode = -1
        done = False
        while not done:
            episode += 1
            # Reset the environment
            # (this reset is typically carried out by the gym module, e.g. by the toy_text.discrete.DiscreteEnv environment's reset() method
            # where the initial state is chosen based on the isd attribute of the object, i.e. of the Initial State Distribution defining the initial state,
            # which is assumed to have been defined appropriately in order to have the start state the user wishes to use)
            self.env.reset()
            # Optional start state JUST for the very first episode
            # (In case we want to start at a specific state and then perform the subsequent resets as per the environment initial state distribution)
            if start_state_first_episode is not None and episode == 0:
                self.env.setState(start_state_first_episode)
            if show_messages(verbose, verbose_period, episode) or episode == nepisodes - 1:  # Note that we ALWAYS show the message at the LAST episode
                print("@{}".format(get_current_datetime_as_string()))
                print("Episode {} of {} running...".format(episode+1, nepisodes), end=" ")
                print("(agent starts at state: {}".format(self.env.getState()), end=" ")
            if self.debug:
                print("\n[DEBUG] Episode {} of {}:".format(episode+1, nepisodes))
                print("\t[DEBUG] Starts at state {}".format(self.env.getState()))
                print("\t[DEBUG] State value function at start of episode:\n\t{}".format(learner.getV().getValues()))

            # Reset variables at the start of a new episode
            done_episode = False
            t_episode = -1          # Time step within the current episode (the first time step will have t_episode = 0)

            if episode > 0:
                # Learn the value of the terminal state visited at the end of the previous episode in the CONTINUING learning task context
                # IMPORTANT: This step must come BEFORE resetting the learner (done below) because the learner erases all history about the previous episode
                # (e.g. observed states, actions and rewards) and these are needed, in particular the rewards, to correctly update the average reward,
                # which should be an estimate of the CONTINUING average reward, NOT of the EPISODIC average reward, where the former is smaller than the latter,
                # as there is one more step taken --per episode-- by the continuing learning task than by the episodic learning task
                # (where the last step of going from a terminal state to a start state --taken by the continuing learning task-- is NOT taken by the episodic learning task).
                if learner.getLearningTask() == LearningTask.CONTINUING:
                    # Count the environment's reset as a new learning step (tracked by variable t) because that step implies learning the terminal state value
                    # (so we are using the available learning budget)
                    t += 1
                    # Check whether we should stop the simulation right after the next learning step (carried out by self.learn_terminal_state_values())
                    # We should check this because the counter of learning steps just increased by 1 and perhaps we reach the maximum number of learning steps after that!
                    # Note that this would trigger the same situation as if parameter max_time_steps were 0:
                    # in that case we would not even perform any learning step as the simulation would stop even before performing the first step!
                    if max_time_steps is not None and t >= max_time_steps - 1:     # `-1` because t_episode starts at 0 and max_time_steps counts the number of steps
                        max_time_steps_reached = True
                        done_episode = True
                        if self.debug:
                            print("[run_single, DEBUG] (TOTAL MAX TIME STEPS = {} REACHED at episode {}!)".format(max_time_steps, episode+1))

                    # For the CONTINUING learning task context, we need to update the value of the state on which the previous episode ended
                    # (as long as it is not the first episode, which is the case at this point), because its value is not necessarily 0!
                    # (it is 0 only in EPISODIC learning tasks, as in that case it is 0 by definition of terminal states).
                    # In fact, in the continuing learning task, the environment state goes to a start state when the episode "terminates" and the Markov process continues.
                    reward = self.env.getReward(self.env.getState())
                    self.learn_terminal_state_values(t_episode, terminal_state_previous_episode, self.env.getState(), reward, info, done_episode=done_episode)
                        ## Notes:
                        ## - it's important that t_episode = -1 here (as is the case because of the reset of t_episode to -1 above) so that there is NO update of the average reward
                        ##  by Learner.update_average_reward() when done_episode=True (i.e. a situation that happens ONLY at the very end of the simulation when the simulation
                        ##  ends at a start state), as at this point, no new episode has started yet. I.e. the episode starts when the first step from the start state is taken,
                        ##  and this happens at t_episode = 0, in which case the length of the episode T would be T = 0 + 1 = 1 > 0 => the average reward can be updated by
                        ##  Learner.update_average_reward().
                        ## - the `info` dictionary is guaranteed to be defined because we call this method only after one episode has been run.
                        ## Therefore the `info` dictionary has been defined as the output of the transition leading to the end of the previous episode.

                # Reset the learner, WITHOUT resetting the value functions nor the episode counter because we want to continue learning from where we left.
                # Essentially this resets the trajectory information stored in the learner.
                # IMPORTANT: We only reset the learner for subsequent episodes (episode > 0) because the reset for episode 0 had already been done before entering the episode loop)
                # This is important because it actually affects the results of learning processes when the learning rate alpha is adjusted by the episode count,
                # as resetting the learner here again (for the first episode = 0) incorrectly increases the episode count which then affects the alpha adjustment!
                # (e.g. makes alpha be adjusted to alpha/3 instead of alpha/2 after the first episode was completed).
                learner.reset(reset_episode=False, reset_value_functions=False)

            while not done_episode:
                t += 1
                t_episode += 1

                # Current state and action on that state leading to the next state
                state = self.env.getState()
                action = policy.choose_action(state)
                next_state, reward, done_episode, info = self.env.step(action)

                # Check early end of episode when max_time_steps_per_episode is given
                # in which case we set done_episode=True.
                # This is important because a set of final operations are done_episode when the episode ends,
                # such as storing the learning rates alpha used in the episode
                # (see the learner object for more details, in particular de learn() method called below
                # that learns the state value function V)
                if max_time_steps_per_episode is not None and t_episode >= max_time_steps_per_episode - 1:     # `-1` because t_episode in the first loop (i.e. the first step) is 0 and max_time_steps_per_episode counts the number of steps
                    nepisodes_max_steps_reached += 1
                    done_episode = True
                    if self.debug:
                        print("[DEBUG] (MAX TIME STEPS PER EPISODE = {} REACHED at episode{}!)".format(max_time_steps_per_episode, episode+1))

                # Check early end of episode when max_time_steps is given
                if max_time_steps is not None and t >= max_time_steps - 1:     # `-1` because t_episode starts at 0 and max_time_steps counts the number of steps
                    max_time_steps_reached = True
                    done_episode = True
                    if self.debug:
                        print("[run_single, DEBUG] (TOTAL MAX TIME STEPS = {} REACHED at episode {}!)".format(max_time_steps, episode+1))

                if self.debug:
                    # We place this message BEFORE calling learner.learn() below because we want to show the state value BEFORE and after updating it (by learner.learn())
                    print("t: {}, t in episode: {}, s={}, a={} -> ns={}, r={}: state_count={:.0f}, alpha={:.3f}, V({})={:.4f}, Q({},{})={:.4f} -> V({})=" \
                          .format(t, t_episode, state, action, next_state, reward, learner.getStateCounts()[state], learner.getAlphaForState(state),
                                  state, learner.V.getValue(state),
                                  state, action, learner.Q.getValue(state, action),
                                  state), end="")

                # Learn: i.e. update the value functions (stored in the learner) for the *currently visited state and action* with the new observation
                learner.learn(t_episode, state, action, next_state, reward, done_episode, info)

                if self.debug:
                    print("{:.4f}, Q=({},{})={:.4f}".format(learner.V.getValue(state), state, action, learner.Q.getValue(state, action)))
                if self.debug and done_episode:
                    print("--> [DEBUG] Done [{} iterations of {} per episode of {} total] at state {} with reward {}" \
                          .format(t_episode+1, max_time_steps_per_episode, max_time_steps, self.env.getState(), reward))

                # Observation state
                if state_observe is not None:
                    # Store the value function of the state just estimated
                    V_state_observe += [learner.getV().getValue(state_observe)]

            #------- EPISODE FINISHED --------#
            # Store the value of the terminal state (used in the next episode when the average reward criterion is used for learning)
            terminal_state_previous_episode = next_state

            if show_messages(verbose, verbose_period, episode):
                print(", agent ENDS at state: {} at discrete time t = {})".format(self.env.getState(), t_episode), end=" ")
                print("")

            # Change of V w.r.t. previous episode (to monitor convergence)
            deltaV = (learner.getV().getValues() - V)
            deltaV_rel = np.array([0.0  if dV == 0
                                        else dV
                                        for dV in deltaV]) / np.array([abs(v) if v != 0 else max(1, V_abs_mean[min(episode, nepisodes-1)]) for v in V])
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
                print("*** Episode {} (start = {}, #steps = {} of MAX={}), Convergence:".format(episode + 1, start_state, t_episode + 1, max_time_steps_per_episode), end=" ")
                try:
                    print("Convergence: min(|V|) with non-zero delta = {:.3g}, mean(V) PREV = {:.3g}, mean(V) NOW = {:.3g}" \
                      .format(np.min(np.abs(V[states_visited_with_positive_deltaV_rel_abs])), np.mean(V[states_visited_so_far]), np.mean(learner.getV().getValues()[states_visited_so_far])) + \
                                       ", mean(delta) = {:.3g}, mean(|delta_rel|) = {:.1f}%, max(|delta_rel|) = {:.1f}% +/-max(delta) = {:.3g} +/-max(delta_rel) = {:.1f}% (n={} of {})" \
                      .format(np.mean(deltaV[states_visited_so_far]), np.mean(np.abs(deltaV_rel[states_visited_so_far]))*100,
                              np.max(np.abs(deltaV_rel[states_visited_so_far]))*100, max_signed_deltaV, max_signed_deltaV_rel*100,
                              len(states_visited_so_far), self.env.getNumStates()) + \
                                       ", mean(alpha) @end = {:.3g}".format(np.mean([learner.getAlphaForState(s) for s in states_visited_so_far])) # Average alpha at the end of episode over ALL states visited so far
                      )
                except:
                    print("WARNING - states_visited_with_positive_deltaV_rel_abs is empty.\n")
                    #print("Count of states visited so far: {}, {}, {}" \
                    #      .format(arr_state_counts, states_visited_so_far, states_visited_with_positive_deltaV_rel_abs))
                    pass

            V = learner.getV().getValues()
            V_abs_mean[min(episode+1, nepisodes)] = np.mean(np.abs(V[states_visited_so_far]))
            V_abs_mean_weighted[min(episode+1, nepisodes)] = np.sum(arr_state_counts * np.abs(V)) / np.sum(arr_state_counts)
            if len(states_visited_with_positive_deltaV_rel_abs) > 0:
                V_abs_min[min(episode+1, nepisodes)] = np.min(np.abs(V[states_visited_with_positive_deltaV_rel_abs]))
            deltaV_abs_mean[min(episode+1, nepisodes)] = np.mean(np.abs(deltaV[states_visited_so_far]))
            deltaV_max_signed[min(episode+1, nepisodes)] = max_signed_deltaV
            deltaV_rel_abs_mean[min(episode+1, nepisodes)] = np.mean(np.abs(deltaV_rel[states_visited_so_far]))
            deltaV_rel_abs_mean_weighted[min(episode+1, nepisodes)] = \
                np.sum( arr_state_counts[states_visited_so_far] * np.abs(deltaV_rel[states_visited_so_far]) ) / np.sum(arr_state_counts[states_visited_so_far])
            deltaV_rel_abs_max[min(episode+1, nepisodes)] = np.max(np.abs(deltaV_rel[states_visited_so_far]))
            deltaV_rel_abs_max_weighted[min(episode+1, nepisodes)] = \
                np.max( arr_state_counts[states_visited_so_far] * np.abs(deltaV_rel[states_visited_so_far]) / np.sum(arr_state_counts[states_visited_so_far]) )
            deltaV_rel_max_signed[min(episode+1, nepisodes)] = max_signed_deltaV_rel
            V_abs_median[min(episode+1, nepisodes)] = np.median(np.abs(V[states_visited_so_far]))
            deltaV_abs_median[min(episode+1, nepisodes)] = np.median(np.abs(deltaV[states_visited_so_far]))
            deltaV_rel_abs_median[min(episode+1, nepisodes)] = np.median(np.abs(deltaV_rel[states_visited_so_far]))
            V_abs_n[min(episode+1, nepisodes)] = len(states_visited_so_far)
            prop_states_deltaV_relevant[min(episode+1, nepisodes)] = np.sum( np.abs(deltaV_rel) > 0.01 ) / self.env.getNumStates()

            if compute_rmse:
                if self.env.getV() is not None:
                    if weights_rmse is not None:
                        weights = learner.getStateCounts()
                    RMSE[min(episode+1, nepisodes)] = rmse(self.env.getV(), learner.getV().getValues(), weights=weights)
                    MAPE[min(episode+1, nepisodes)] = mape(self.env.getV(), learner.getV().getValues(), weights=weights)

            if plot and show_messages(True, verbose_period, episode):
                # Plot the estimated value function at the end of the episode
                # in both 1D layout and 2D layout, if the environment is 2D.
                #print("episode: {} (T={}), color: {}".format(episode, t_episode, colors(episode/nepisodes)))
                plt.figure(fig_V.number)
                plt.plot(self.env.getAllStates(), learner.getV().getValues(), linewidth=0.5,
                         color=colors(min(episode, nepisodes - 1) / nepisodes))
                plt.title("State values evolution (blue: initial, red: final)\nEpisode {} of {}".format(episode + 1, nepisodes))
                plt.pause(pause)
                plt.draw()
                #fig_V.canvas.draw()    # This should be equivalent to plt.draw()
                if self.env.getDimension() == 2:
                    # Update the 2D plots
                    plt.figure(fig_V2.number)
                    (ax_V, ax_C) = fig_V2.subplots(1, 2)
                    shape = self.env.getShape()
                    rewards = self.env.getRewards()

                    state_values = np.asarray(learner.getV().getValues()).reshape(shape)
                    if len(rewards) > 0:
                        colornorm = plt.Normalize(vmin=np.min(list(rewards)), vmax=np.max(list(rewards)))
                    else:
                        colornorm = None
                    ax_V.imshow(state_values, cmap=colors) #, norm=colornorm) # (2023/11/23) If we use norm=colornorm we might see all blue colors... even if the V values vary from 0 to 1... why??

                    arr_state_counts = learner.getStateCounts().reshape(shape)
                    colors_count = cm.get_cmap("Blues")
                    colornorm = plt.Normalize(vmin=0, vmax=np.max(arr_state_counts))
                    ax_C.imshow(arr_state_counts, cmap=colors_count, norm=colornorm)

                    fig_V2.suptitle("State values (left) and state counts (right)\nEpisode {} of {}".format(episode, nepisodes))
                    plt.pause(pause)
                    plt.draw()

                if state_observe is not None:
                    plt.figure(fig_RMSE_state.number)

                    # Compute quantities to plot
                    RMSE_state_observe = rmse(np.array(self.env.getV()[state_observe]), np.array(learner.getV().getValue(state_observe)), weights=weights)
                    se95 = 2*np.sqrt( 0.5*(1-0.5) / (episode+1))
                    # Only count falling inside the CI starting at episode 100 so that
                    # the normal approximation of the SE is more correct.
                    if episode + 1 >= 100:
                        ntimes_rmse_inside_ci95 += (RMSE_state_observe <= se95)

                    # Plot of the estimated value of the state at the end of the episode
                    # NOTE: the plot is just ONE point, the state value at the end of the episode;
                    # at each new episode, a new point with this information will be added to the plot.
                    plt.plot(episode+1, learner.getV().getValue(state_observe), 'r*-', markersize=7)
                    # Plot the average of the state value function over the values that it took along the episode while it was being learned.
                    #plt.plot(episode, np.mean(np.array(V_state_observe)), 'r.-')
                    # Plot of the estimation error and the decay of the "confidence bands" for the true value function
                    plt.plot(episode+1, RMSE_state_observe, 'k.-', markersize=7)
                    plt.plot(episode+1,  se95, color="gray", marker=".", markersize=5)
                    plt.plot(episode+1, -se95, color="gray", marker=".", markersize=5)
                    # Plot of learning rate
                    #plt.plot(episode+1, learner.getAlphaForState(state_observe), 'g.-')

                    # Finalize plot
                    ax = plt.gca()
                    ax.set_ylim((-1, 1))
                    yticks = np.arange(-10,10)/10
                    ax.set_yticks(yticks)
                    ax.axhline(y=self.env.getV()[state_observe], color="red", linewidth=0.5)
                    ax.axhline(y=0, color="gray")
                    plt.title("State value (estimated and true) and |error| for state {} - episode {} of {}".format(state_observe, episode+1, nepisodes))
                    legend = ['Estimated value', '|error|', 'true value', '2*SE = 2*sqrt(0.5*(1-0.5)/episode)']
                    plt.legend(legend)
                    if episode + 1 == nepisodes:
                        # Show the true value function coverage as x-axis label
                        ax.set_xlabel("% Episodes error is inside 95% Confidence Interval (+/- 2*SE) (for episode>=100): {:.1f}%" \
                                      .format(ntimes_rmse_inside_ci95/(episode+1 - 100 + 1)*100))
                    plt.legend(legend)
                    plt.draw()

            if plot and isinstance(learner, LeaTDLambdaAdaptive) and episode == nepisodes - 1:
                learner.plot_info(episode, nepisodes)

            # Use the following IF when we want to stop the simulation EITHER when the maximum time steps (over all episodes) is reached
            # OR
            # when the number of episodes has been run.
            if max_time_steps_reached or episode == nepisodes - 1:
            # Use the following IF when we want to stop the simulation ONLY when the maximum time steps (over all episodes) has been reached
            # This option is useful when we want to be FAIR in comparing a traditional TD learner with the FV learner of value functions,
            # so that each learner observes exactly the SAME number of time steps (in which case the FV simulation in _run_simulation_fv()
            # should ALSO be ended ONLY by the condition that the maximum time steps has been reached, NOT when all particles have been absorbed at least once).
            #if max_time_steps_reached:
                # The maximum number of steps over ALL episodes has been reached, so we need to stop the simulation
                done = True
            elif (max_time_steps is None or max_time_steps == np.Inf) and \
                episode == nepisodes - 1:
                # We reached the number of episodes to run (only checked when the maximum number of steps to run is not given
                    done = True

        if verbose:
            V = learner.getV().getValues()
            Q = learner.getQ().getValues()
            print(f"Estimated value functions at END of experiment (last episode run = {episode+1} of {nepisodes}):")
            print("V = {}".format(V))
            print("Q = {}".format(Q.reshape(self.env.getNumStates(), self.env.getNumActions())))

        # Comment this out to NOT show the plot right away
        # in case the calling function adds a new plot to the graph generated here
        if plot and show_messages(True, verbose_period, t):
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

        if verbose:
            print("Percentage of episodes reaching max step = {:.1f}%".format(nepisodes_max_steps_reached / nepisodes*100))
            print("Last episode run = {} of {}".format(episode+1, nepisodes))

        return  learner.getV().getValues(), learner.getQ().getValues(), learner.getStateCounts(), RMSE, MAPE, \
                {   'nsteps': t + 1,   # Number of discrete steps taken during the whole simulation (+1 because t = 0 indexes the first step)
                    # Value of alpha for each state at the end of the LAST episode run
                    'alphas_at_last_episode': learner._alphas,
                    # All what follows is information by episode (I don't explicitly mention it in the key name because it may make the key name too long...)
                    # (Average) alpha by episode (averaged over visited states in the episode)
                    'alpha_mean': learner.getAverageAlphaByEpisode(),   # Average alpha (over all states) by episode
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
                    'state_observe': state_observe,
                    'V_state_observe': V_state_observe,
                }

    def _run_single_continuing_task(self, nepisodes=1, max_time_steps=1000, max_time_steps_per_episode=+np.Inf, start_state_first_episode=None, reset_value_functions=True,
                                    seed=None, compute_rmse=False, weights_rmse=None,
                                    state_observe=None, set_cycle=None,
                                    verbose=False, verbose_period=1, verbose_convergence=False,
                                    plot=False, colormap="seismic", pause=0.1):
        """
        Same as _run_single() but where there are no episodes.

        At this point (Jan-2024), the method needs a LOT of refactoring, as this method was constructed keeping in mind that I didn't want to do too many changes
        to the _run_single() method from which it was created.
        For instance, we need to get rid of the WHILE loop on the episodes because the episodes never increase. So, at this point parameter nepisodes
        is expected to always be 1, as specified in its default value.

        set_cycle: (opt) set
            Set of states whose entrance from the complement set defines a cycle.
            Note that the set should include ALL the states, NOT only the boundary states through which the system can enter the set.
            The reason is that the set is used to determine which states are tracked for their visit frequency for the computation
            of their stationary probability using renewal theory.
            default: None

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
        #--- Parse input parameters
        # ---- UPDATE FOR CONTINUING TASK
        if max_time_steps is None or max_time_steps <= 0 or max_time_steps == np.Inf:
            raise ValueError(f"The maximum number of time steps to run must be given and must be a FINITE positive number: {max_time_steps}")
        # ---- UPDATE FOR CONTINUING TASK

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
            elif not (is_integer(state_observe) and 0 <= state_observe and state_observe < self.env.getNumStates()):
                warnings.warn("The `state_observe` parameter ({}, type={}) must be an integer number between 0 and {}.\n" \
                              "The state whose index falls in the middle of the state space will be observed." \
                              .format(state_observe, type(state_observe), self.env.getNumStates()-1))
                state_observe = self.env.getNumStates() // 2

        # Setup the information needed when cycles are used to estimate the stationary distribution of states using renewal theory
        cycle_times = []  # Note that the first cycle time will include a time that may not be a cycle time because the cycle may have not initiated at the sytem's start state. So the first cycle time will be considered a delay time.
        num_cycles = 0
        last_cycle_entrance_time = 0       # We set the last cycle time (i.e. the moment when the system enters the cycle set) to 0 (even if it is unknown) so that we can easily compute the FIRST cycle time below as "t - last_cycle_entrance_time"
        expected_cycle_time = 0.0  # We set the expected cycle time so that we can compute the expected cycle time recursively
        # Array of state counts in COMPLETE cycles, i.e. the count is increased ONLY when the state is visited in a complete cycle
        # (not during the first "cycle" which may be degenerate --i.e. incomplete, as the start state for the first cycle may not be in the cycle set)
        # This can be used to estimate the stationary probability of the states using renewal theory
        state_counts_in_complete_cycles = np.zeros(self.env.getNumStates(), dtype=int)

        # Plotting setup
        if plot:
            # Setup the figures that will be updated at every verbose_period
            colors = cm.get_cmap(colormap, lut=max_time_steps)
            # 1D plot (even for 2D environments)
            fig_V = plt.figure(figsize=(10, 16))
            # Plot the true state value function (to have it as a reference already
            plt.plot(self.env.getAllStates(), self.env.getV(), '.-', color="blue")
            if self.env.getDimension() == 2:
                # 2D plot
                fig_V2 = plt.figure()
            if state_observe is not None:
                # Plot with the evolution of the V estimate and its error
                fig_RMSE_state = plt.figure()
        #--- Parse input parameters

        # Define the policy and the learner
        policy = self.agent.getPolicy()
        learner = self.agent.getLearner()

        # Reset the learner (i.e. prepare it for a fresh new learning experience with all learning memory erased and learning rates reset to their initial values)
        # Note that a special treatment may be granted to the reset of the value functions because we may want NOT to reset them,
        # for instance when we are learning a policy and we use this simulator to learn the value functions...
        # In that case, we don't want to start off at 0.0 again but to start off at the estimated values obtained
        # under the previous policy, which normally is very close to the new policy after one step of policy learning.
        # (In some situations --e.g. labyrinth where the policy is learned using Actor-Critic or policy gradient learning--
        # I observed (Nov-2023) that non-optimal policies are learned that minimize the loss function if we reset the
        # value functions to 0 at every policy learning step, while the problem does NOT happen when the value functions are NOT reset.)
        learner.reset(reset_episode=True, reset_value_functions=reset_value_functions)

        # Environment seed
        # We only set the seed when it is not None because when this method is called by the simulate() method,
        # seed is set to None in order to avoid having each experiment (i.e. each replication) produce the same results
        # (which would certainly invalidate the replications!). In that case, the environment's seed is set *before*
        # calling this method run() and we don't want to revert that seed setting, o.w. the experiments repeatability
        # would be broken.
        if seed is not None:
            self.env.setSeed(seed)

        # Store initial values used in the analysis of all the episodes run
        V_state_observe, RMSE, MAPE, ntimes_rmse_inside_ci95 = self.initialize_run_with_learner_status(nepisodes, learner, compute_rmse, weights, state_observe)

        # Initial state value function
        V = learner.getV().getValues()
        Q = learner.getQ().getValues()
        if verbose:
            print("Value functions at start of experiment:")
            print("V = {}".format(V))
            print("Q = {}".format(Q.reshape(self.env.getNumStates(), self.env.getNumActions())))

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
        max_time_steps_reached = False  # Flags whether the TOTAL number of steps reaches the maximum number of steps allowed over all episodes (so that we can break the FOR loop on episodes if that happens)
        t = -1            # This time index is used when a cycle set has been given, in order to compute the cycle time
                          # which can be used to estimate the stationary probability of states using renewal theory.
        episode = -1
        done = False
        while not done:
            episode += 1
            # Reset the environment
            # (this reset is typically carried out by the gym module, e.g. by the toy_text.discrete.DiscreteEnv environment's reset() method
            # where the initial state is chosen based on the isd attribute of the object, i.e. of the Initial State Distribution defining the initial state,
            # which is assumed to have been defined appropriately in order to have the start state the user wishes to use)
            self.env.reset()
            # Optional start state JUST for the very first episode
            # (In case we want to start at a specific state and then perform the subsequent resets as per the environment initial state distribution)
            if start_state_first_episode is not None and episode == 0:
                self.env.setState(start_state_first_episode)
            if show_messages(verbose, verbose_period, episode) or episode == nepisodes - 1:  # Note that we ALWAYS show the message at the LAST episode
                print("@{}".format(get_current_datetime_as_string()))
                print("Episode {} of {} running...".format(episode+1, nepisodes), end=" ")
                print("(agent starts at state: {}".format(self.env.getState()), end=" ")
            if self.debug:
                print("\n[DEBUG] Episode {} of {}:".format(episode+1, nepisodes))
                print("\t[DEBUG] Starts at state {}".format(self.env.getState()))
                print("\t[DEBUG] State value function at start of episode:\n\t{}".format(learner.getV().getValues()))

            # Time step within the current episode (the first time step where an action is taken is t_episode = 0)
            t_episode = -1
            done_episode = False
            stop = False
            while not stop:
                t += 1
                t_episode += 1

                # Current state and action on that state leading to the next state
                state = self.env.getState()
                V_state_prev = learner.getV().getValue(state)  # Store the current V(s) value as PREV for informational plotting purposes

                # ---- UPDATE FOR CONTINUING TASK
                if done_episode:
                    # The episode ended at the previous step
                    # => Reset the episode-related information (needed most importantly for a correct calculation of the average reward)
                    # => Reset the environment to a start state because the process honours a CONTINUING learning task (of the value functions)

                    t_episode = -1      # Reset the episode counter. Note that we reset it to -1 and NOT 0 because the episode is considered to start when `state` is a START state and here state is a terminal state, whereas *`next_state`* is the start state. So t_episode = 0 should be set at the next iteration.
                    # Reset the learner as a new episode will start
                    # It is important to reset all the episode-related information, most importantly the history of rewards observed in the episode,
                    # which is used to compute the average reward. If this reset is not done, most likely the average reward will be WAY underestimated
                    # because the update of the average reward function ((normally) Learner._update_average_reward()) will think that the number of rewards
                    # observed in the episode when updating the within-episode average reward is WAY larger than it really was.
                    learner.reset(reset_episode=False, reset_value_functions=False)

                    # Perform the action of going to an environment's start state
                    action = 0
                    next_state = self.env.reset()
                    reward = self.env.getReward(next_state)
                    done_episode = False
                    # TEMPORARY: (2024/02/13) Two temporary settings are done here, until the proper implementation of a CONTINUING learning task (with NO episodes) is done, as follows:
                    # 1) Non-update of trajectory: the trajectory should NOT be updated when learning from the transition "terminal state" -> "start state" because we do NOT want
                    # to have this transition contribute to the estimation of the average reward (i.e. we do not want to have the reward observed when going from a terminal state
                    # to a start state to be present in the list of episode rewards stored in self._rewards) because the Learner.update_average_reward() method that performs
                    # this calculation assumes right now that the average reward estimated by the learner is the EPISODIC average reward, and thus adjusts for this when estimating
                    # the CONTINUING average reward (by multiplying the average reward observed in the episode by T / (T+1), which assumes that the reward received when going
                    # from a terminal state to a start state is r=0 --as is usually the case, although it does not really have to be the case... to be fixed in the near future).
                    # IN THE NEAR FUTURE, this restriction of not updating the trajectory will be removed as we intend to adapt the Learner.update_average_reward() method
                    # in order to:
                    # - estimate the EPISODIC average reward when the learning task is EPISODIC.
                    # - estimate the CONTINUING average reward when the learning task is CONTINUING.
                    # This will also eliminate the assumption mentioned above that is currently done in Learner.update_average_reward() that the reward received when going from
                    # a terminal state to a start state is 0.
                    # 2) Non-update of state counts: the count of the terminal state should not be increased by 1 now because it was ALREADY increased at the end of the "episode"
                    # when learning at the previous iteration, because the count of the final episode state is increased when done_episode = True in the call to learner.learn().
                    info = {'update_trajectory': False,
                            'update_counts': False}
                # ---- UPDATE FOR CONTINUING TASK
                else:
                    action = policy.choose_action(state)
                    next_state, reward, done_episode, info = self.env.step(action)

                # Check early end of episode when max_time_steps_per_episode is given
                # in which case we set done_episode=True.
                # This is important because a set of final operations are done_episode when the episode ends,
                # such as storing the learning rates alpha used in the episode
                # (see the learner object for more details, in particular de learn() method called below
                # that learns the state value function V)
                if max_time_steps_per_episode is not None and t_episode >= max_time_steps_per_episode - 1:     # `-1` because t_episode in the first loop (i.e. the first step) is 0 and max_time_steps_per_episode counts the number of steps
                    nepisodes_max_steps_reached += 1
                    done_episode = True
                    stop = True
                    if self.debug:
                        print("[DEBUG] (MAX TIME STEPS PER EPISODE = {} REACHED at episode{}!)".format(max_time_steps_per_episode, episode+1))

                # Check end of simulation
                if t >= max_time_steps - 1:     # `-1` because t_episode starts at 0 and max_time_steps counts the number of steps
                    max_time_steps_reached = True
                    done_episode = True
                    stop = True
                    if self.debug:
                        print("[run_single_continuing_task, DEBUG] (TOTAL MAX TIME STEPS = {} REACHED at episode {}!)".format(max_time_steps, episode+1))

                if self.debug:
                    # We place this message BEFORE calling learner.learn() below because we want to show the state value BEFORE and after updating it (by learner.learn())
                    print("t: {}, t in episode: {}, s={}, a={} -> ns={}, r={}: state_count={:.0f}, alpha={:.3f}, V({})={:.4f}, Q({},{})={:.4f} -> V({})=" \
                          .format(t, t_episode, state, action, next_state, reward, learner.getStateCounts()[state], learner.getAlphaForState(state),
                                  state, learner.V.getValue(state),
                                  state, action, learner.Q.getValue(state, action),
                                  state), end="")

                # Learn: i.e. update the value functions (stored in the learner) for the *currently visited state and action* with the new observation
                learner.learn(t_episode, state, action, next_state, reward, done_episode, info)
                if state in self.env.getTerminalStates():
                    # We need to copy the Q-values of the terminal state to the other actions because the action chosen to go to the start state is always the same (action 0)
                    action_anchor = 0
                    for _action in range(self.env.getNumActions()):
                        learner.getQ()._setWeight(state, _action, learner.getQ().getValue(state, action_anchor))

                if self.debug:
                    print("{:.4f}, Q=({},{})={:.4f}".format(learner.V.getValue(state), state, action, learner.Q.getValue(state, action)))
                if self.debug and done_episode:
                    print("--> [DEBUG] Done [{} iterations of {} per episode of {} total] at state {} with reward {}" \
                          .format(t_episode+1, max_time_steps_per_episode, max_time_steps, self.env.getState(), reward))

                # Observation state
                if state_observe is not None:
                    # Store the value function of the state just estimated
                    V_state_observe += [learner.getV().getValue(state_observe)]

                # Check if the system has entered the set of states defining a cycle
                cycle_times, state_counts_in_complete_cycles, last_cycle_entrance_time, expected_cycle_time, num_cycles = \
                    self._check_cycle_occurrence_and_update_cycle_variables(set_cycle, t, state, next_state,
                                                                            cycle_times, state_counts_in_complete_cycles, last_cycle_entrance_time, expected_cycle_time, num_cycles)

                # ---- UPDATE FOR CONTINUING TASK
                # Plotting step moved INSIDE the episode because there is only 1 episode!
                if plot and show_messages(True, verbose_period, t):
                    if self.env.getDimension() == 2:
                        # Plot the state counts in a separate image
                        fig_C = plt.figure()
                        plt.figure(fig_C.number)

                        state_counts_min = np.min(learner.getStateCounts())
                        state_counts_mean = np.mean(learner.getStateCounts())
                        state_counts_max = np.max(learner.getStateCounts())

                        shape = self.env.getShape()
                        arr_state_counts = learner.getStateCounts().reshape(shape)
                        colors_count = cm.get_cmap("Blues")
                        colornorm = plt.Normalize(vmin=0, vmax=np.max(arr_state_counts))
                        plt.imshow(arr_state_counts, cmap=colors_count, norm=colornorm)
                        # Font size factor
                        fontsize = 14
                        factor_fs = np.min((5 / shape[0], 5 / shape[1]))
                        for x in range(shape[0]):
                            for y in range(shape[1]):
                                # Recall the x axis corresponds to the columns of the matrix shown in the image
                                # and the y axis corresponds to the rows
                                plt.text(y, x, "{:.0f}".format(arr_state_counts[x, y]),
                                         fontsize=fontsize * factor_fs, horizontalalignment='center', verticalalignment='center')

                        plt.title("State counts by state\n# visits: (min, mean, max) = ({:.0f}, {:.1f}, {:.0f})" \
                                  .format(state_counts_min, state_counts_mean, state_counts_max))
                    else:
                        # Add the state counts to the plot of the state value function
                        # plt.colorbar(cm.ScalarMappable(cmap=colormap))    # Does not work
                        plt.figure(fig_V.number)
                        plt.plot(self.env.getAllStates(), learner.getV().getValues(), linewidth=0.5, color=colors(t / max_time_steps))
                        if False:
                            ax = plt.gca()
                            ax2 = ax.twinx()  # Create a secondary axis sharing the same x axis
                            ax2.bar(self.env.getAllStates(), learner.getStateCounts(), color="blue", alpha=0.3)
                            plt.sca(ax)  # Go back to the primary axis
                        plt.title(f"V(s) evolution (blueish: initial, reddish: final)"
                                  f"\nSystem's time t = {t} of max {max_time_steps}, from state = {state} --> next state = {next_state} (V(s) = {np.round(learner.getV().getValue(next_state), 3)})"
                                  f"\nV(state={state}) = {np.round(V_state_prev, 3)} --> V(state={state}) = {np.round(learner.getV().getValue(state), 3)} (delta(V) = {np.round(learner.getV().getValue(state) - V_state_prev, 3)}, {np.round((learner.getV().getValue(state) - V_state_prev) / max(1, V_state_prev) * 100, 1)}%)"
                                  f"\nCount[s] = {learner.getStateCounts()}, alpha[s] = {learner.getAlphasByState()}")
                        plt.pause(pause)
                        plt.draw()
                # ---- UPDATE FOR CONTINUING TASK



            #------- EPISODE FINISHED --------#
            if show_messages(verbose, verbose_period, episode):
                print(", agent ENDS at state: {} at discrete time t = {})".format(self.env.getState(), t_episode), end=" ")
                print("")

            # Change of V w.r.t. previous episode (to monitor convergence)
            deltaV = (learner.getV().getValues() - V)
            deltaV_rel = np.array([0.0  if dV == 0
                                        else dV
                                        for dV in deltaV]) / np.array([abs(v) if v != 0 else max(1, V_abs_mean[min(episode, nepisodes-1)]) for v in V])
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
                print("*** Episode {} (start = {}, #steps = {} of MAX={}), Convergence:".format(episode + 1, start_state, t_episode + 1, max_time_steps_per_episode), end=" ")
                try:
                    print("Convergence: min(|V|) with non-zero delta = {:.3g}, mean(V) PREV = {:.3g}, mean(V) NOW = {:.3g}" \
                      .format(np.min(np.abs(V[states_visited_with_positive_deltaV_rel_abs])), np.mean(V[states_visited_so_far]), np.mean(learner.getV().getValues()[states_visited_so_far])) + \
                                       ", mean(delta) = {:.3g}, mean(|delta_rel|) = {:.1f}%, max(|delta_rel|) = {:.1f}% +/-max(delta) = {:.3g} +/-max(delta_rel) = {:.1f}% (n={} of {})" \
                      .format(np.mean(deltaV[states_visited_so_far]), np.mean(np.abs(deltaV_rel[states_visited_so_far]))*100,
                              np.max(np.abs(deltaV_rel[states_visited_so_far]))*100, max_signed_deltaV, max_signed_deltaV_rel*100,
                              len(states_visited_so_far), self.env.getNumStates()) + \
                                       ", mean(alpha) @end = {:.3g}".format(np.mean([learner.getAlphaForState(s) for s in states_visited_so_far])) # Average alpha at the end of episode over ALL states visited so far
                      )
                except:
                    print("WARNING - states_visited_with_positive_deltaV_rel_abs is empty.\n")
                    #print("Count of states visited so far: {}, {}, {}" \
                    #      .format(arr_state_counts, states_visited_so_far, states_visited_with_positive_deltaV_rel_abs))
                    pass

            V = learner.getV().getValues()
            V_abs_mean[min(episode+1, nepisodes)] = np.mean(np.abs(V[states_visited_so_far]))
            V_abs_mean_weighted[min(episode+1, nepisodes)] = np.sum(arr_state_counts * np.abs(V)) / np.sum(arr_state_counts)
            if len(states_visited_with_positive_deltaV_rel_abs) > 0:
                V_abs_min[min(episode+1, nepisodes)] = np.min(np.abs(V[states_visited_with_positive_deltaV_rel_abs]))
            deltaV_abs_mean[min(episode+1, nepisodes)] = np.mean(np.abs(deltaV[states_visited_so_far]))
            deltaV_max_signed[min(episode+1, nepisodes)] = max_signed_deltaV
            deltaV_rel_abs_mean[min(episode+1, nepisodes)] = np.mean(np.abs(deltaV_rel[states_visited_so_far]))
            deltaV_rel_abs_mean_weighted[min(episode+1, nepisodes)] = \
                np.sum( arr_state_counts[states_visited_so_far] * np.abs(deltaV_rel[states_visited_so_far]) ) / np.sum(arr_state_counts[states_visited_so_far])
            deltaV_rel_abs_max[min(episode+1, nepisodes)] = np.max(np.abs(deltaV_rel[states_visited_so_far]))
            deltaV_rel_abs_max_weighted[min(episode+1, nepisodes)] = \
                np.max( arr_state_counts[states_visited_so_far] * np.abs(deltaV_rel[states_visited_so_far]) / np.sum(arr_state_counts[states_visited_so_far]) )
            deltaV_rel_max_signed[min(episode+1, nepisodes)] = max_signed_deltaV_rel
            V_abs_median[min(episode+1, nepisodes)] = np.median(np.abs(V[states_visited_so_far]))
            deltaV_abs_median[min(episode+1, nepisodes)] = np.median(np.abs(deltaV[states_visited_so_far]))
            deltaV_rel_abs_median[min(episode+1, nepisodes)] = np.median(np.abs(deltaV_rel[states_visited_so_far]))
            V_abs_n[min(episode+1, nepisodes)] = len(states_visited_so_far)
            prop_states_deltaV_relevant[min(episode+1, nepisodes)] = np.sum( np.abs(deltaV_rel) > 0.01 ) / self.env.getNumStates()

            if compute_rmse:
                if self.env.getV() is not None:
                    if weights_rmse is not None:
                        weights = learner.getStateCounts()
                    RMSE[min(episode+1, nepisodes)] = rmse(self.env.getV(), learner.getV().getValues(), weights=weights)
                    MAPE[min(episode+1, nepisodes)] = mape(self.env.getV(), learner.getV().getValues(), weights=weights)

            if plot and show_messages(True, verbose_period, episode):
                # Plot the estimated value function at the end of the simulation
                # in both 1D layout and 2D layout, if the environment is 2D.
                plt.figure(fig_V.number)
                plt.plot(self.env.getAllStates(), learner.getV().getValues(), linewidth=0.5, color=colors(t / max_time_steps))
                plt.pause(pause)
                plt.draw()
                #fig_V.canvas.draw()    # This should be equivalent to plt.draw()
                if self.env.getDimension() == 2:
                    # Update the 2D plots
                    plt.figure(fig_V2.number)
                    (ax_V, ax_C) = fig_V2.subplots(1, 2)
                    shape = self.env.getShape()
                    rewards = self.env.getRewards()

                    state_values = np.asarray(learner.getV().getValues()).reshape(shape)
                    if len(rewards) > 0:
                        colornorm = plt.Normalize(vmin=np.min(list(rewards)), vmax=np.max(list(rewards)))
                    else:
                        colornorm = None
                    ax_V.imshow(state_values, cmap=colors) #, norm=colornorm) # (2023/11/23) If we use norm=colornorm we might see all blue colors... even if the V values vary from 0 to 1... why??

                    arr_state_counts = learner.getStateCounts().reshape(shape)
                    colors_count = cm.get_cmap("Blues")
                    colornorm = plt.Normalize(vmin=0, vmax=np.max(arr_state_counts))
                    ax_C.imshow(arr_state_counts, cmap=colors_count, norm=colornorm)

                    fig_V2.suptitle("State values (left) and state counts (right)\nEpisode {} of {}".format(episode, nepisodes))
                    plt.pause(pause)
                    plt.draw()

                if state_observe is not None:
                    plt.figure(fig_RMSE_state.number)

                    # Compute quantities to plot
                    RMSE_state_observe = rmse(np.array(self.env.getV()[state_observe]), np.array(learner.getV().getValue(state_observe)), weights=weights)
                    se95 = 2*np.sqrt( 0.5*(1-0.5) / (episode+1))
                    # Only count falling inside the CI starting at episode 100 so that
                    # the normal approximation of the SE is more correct.
                    if episode + 1 >= 100:
                        ntimes_rmse_inside_ci95 += (RMSE_state_observe <= se95)

                    # Plot of the estimated value of the state at the end of the episode
                    # NOTE: the plot is just ONE point, the state value at the end of the episode;
                    # at each new episode, a new point with this information will be added to the plot.
                    plt.plot(episode+1, learner.getV().getValue(state_observe), 'r*-', markersize=7)
                    # Plot the average of the state value function over the values that it took along the episode while it was being learned.
                    #plt.plot(episode, np.mean(np.array(V_state_observe)), 'r.-')
                    # Plot of the estimation error and the decay of the "confidence bands" for the true value function
                    plt.plot(episode+1, RMSE_state_observe, 'k.-', markersize=7)
                    plt.plot(episode+1,  se95, color="gray", marker=".", markersize=5)
                    plt.plot(episode+1, -se95, color="gray", marker=".", markersize=5)
                    # Plot of learning rate
                    #plt.plot(episode+1, learner.getAlphaForState(state_observe), 'g.-')

                    # Finalize plot
                    ax = plt.gca()
                    ax.set_ylim((-1, 1))
                    yticks = np.arange(-10,10)/10
                    ax.set_yticks(yticks)
                    ax.axhline(y=self.env.getV()[state_observe], color="red", linewidth=0.5)
                    ax.axhline(y=0, color="gray")
                    plt.title("State value (estimated and true) and |error| for state {} - episode {} of {}".format(state_observe, episode+1, nepisodes))
                    legend = ['Estimated value', '|error|', 'true value', '2*SE = 2*sqrt(0.5*(1-0.5)/episode)']
                    plt.legend(legend)
                    if episode + 1 == nepisodes:
                        # Show the true value function coverage as x-axis label
                        ax.set_xlabel("% Episodes error is inside 95% Confidence Interval (+/- 2*SE) (for episode>=100): {:.1f}%" \
                                      .format(ntimes_rmse_inside_ci95/(episode+1 - 100 + 1)*100))
                    plt.legend(legend)
                    plt.draw()

            if plot and isinstance(learner, LeaTDLambdaAdaptive) and episode == nepisodes - 1:
                learner.plot_info(episode, nepisodes)

            # Use the following IF when we want to stop the simulation EITHER when the maximum time steps (over all episodes) is reached
            # OR
            # when the number of episodes has been run.
            if max_time_steps_reached or episode == nepisodes - 1:
                # The maximum number of steps over ALL episodes has been reached, so we need to stop the simulation
                done = True

        if verbose:
            V = learner.getV().getValues()
            Q = learner.getQ().getValues()
            print(f"Estimated value functions at END of experiment (last episode run = {episode+1} of {nepisodes}):")
            print("V = {}".format(V))
            print("Q = {}".format(Q.reshape(self.env.getNumStates(), self.env.getNumActions())))

        if verbose:
            print("Percentage of episodes reaching max step = {:.1f}%".format(nepisodes_max_steps_reached / nepisodes*100))
            print("Last episode run = {} of {}".format(episode+1, nepisodes))

        return  learner.getV().getValues(), learner.getQ().getValues(), learner.getStateCounts(), RMSE, MAPE, \
                {   'nsteps': t + 1,   # Number of discrete steps taken during the whole simulation (+1 because t = 0 indexes the first step)
                    # Value of alpha for each state at the end of the LAST episode run
                    'alphas_at_last_episode': learner._alphas,
                    # All what follows is information by episode (I don't explicitly mention it in the key name because it may make the key name too long...)
                    # (Average) alpha by episode (averaged over visited states in the episode)
                    'alpha_mean': learner.getAverageAlphaByEpisode(),   # Average alpha (over all states) by episode
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
                    'state_observe': state_observe,
                    'V_state_observe': V_state_observe,
                    'num_cycles': num_cycles if set_cycle is not None else None,
                    'last_cycle_entrance_time': last_cycle_entrance_time if set_cycle else None,
                    'expected_cycle_time': expected_cycle_time if set_cycle is not None else None,
                    'state_counts_in_complete_cycles': state_counts_in_complete_cycles if set_cycle is not None else None,
                }

    def _check_cycle_occurrence_and_update_cycle_variables(self, set_cycle, t, state, next_state, cycle_times, state_counts_in_complete_cycles, last_cycle_entrance_time, expected_cycle_time, num_cycles):
        """
        Checks the occurrence of a cycle and updates the estimated cycle time and related tracking variables

        Arguments:
        set_cycle: set
            Set whose visit from a state NOT in `set_cycle` defines a new cycle whenever an entrance event to the set has been observed previously.

        t: positive float
            Time elapsed since the start of the process simulation.
            Note that this time should NOT be reset by the completion of an episode.
        """
        entered_set_cycle = lambda s, ns: s not in set_cycle and ns in set_cycle
        if set_cycle is not None:
            if entered_set_cycle(state, next_state):
                # Notes on the cycle time calculation:
                # 1) We sum +1 to t because the next_state happens at the NEXT time step, which is t+1
                # and the next_state is the one that defines absorption, so we should measure the time at which the next_state occurs.
                # (in fact, note that the very first t value is 0, therefore we should not consider the first entry
                # time to be 0 if the system enters the cycle set at the very first step...)
                # 2) The fact that we use `t` to compute the cycle time --instead of `t_episode`-- indicates that we are considering the task to be a continuing learning task,
                # as opposed to an episodic learning task, because the value of `t` is NOT reset at the beginning of each episode,
                # instead it keeps increasing with every new time step.
                cycle_times += [t + 1 - last_cycle_entrance_time]
                # We mark the time the system entered the cycle set, so that:
                # - we can compute the next cycle time
                # - we can decide when to start measuring cycle times because we know they correspond to FULL cycle times.
                last_cycle_entrance_time = t + 1
                if len(cycle_times) > 1:
                    # We disregard the first cycle time from the computation of the average
                    # because the first entering event may not represent a cycle as the system may have not previously exited the set.
                    num_cycles += 1
                    expected_cycle_time += (cycle_times[-1] - expected_cycle_time) / num_cycles
                    if DEBUG_TRAJECTORIES:
                        print(f"Entered cycle set: (t, t_cycle, s, ns) = ({t}, {cycle_times[-1]}, {state}, {next_state})")

            # Update the count of the state ONLY after the first entrance to the cycle set,
            # so that we make sure that the counts are measured during a TRUE cycle
            # (as the first entrance to the cycle set may correspond to a degenerate (incomplete) cycle, because the start state may not be in the cycle set)
            if last_cycle_entrance_time > 0:
                # The process entered the cycle set at least once (as if no entrance to the cycle set happened yet, last_cycle_entrance_time = 0)
                state_counts_in_complete_cycles[state] += 1

        return cycle_times, state_counts_in_complete_cycles, last_cycle_entrance_time, expected_cycle_time, num_cycles

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
        # Store the initial RMSE, i.e. based on the initial state values proposed by the learner (retrieved by learner.getV().getValues())
        if compute_rmse:
            RMSE = np.nan*np.zeros(nepisodes+1)
            MAPE = np.nan*np.zeros(nepisodes+1)
            # RMSE (True vs. Estimated values)
            print(f"True value function:\n{self.env.getV()}")
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

    def learn_terminal_state_values(self, t, terminal_state, next_state, reward, info, done_episode=False, envs=None, idx_particle=None, update_phi=False):
        """
        Learns the state and action values for the terminal state in the continuing learning task context

        The action values are set to the same value for all actions, because all actions lead to an environment's start state,
        which is the state chosen after the agent reaches a terminal state, in order to continue with its learning task.

        Arguments:
        next_state: int
            Index of the state visited AFTER visiting the terminal_state. This is normally a start state, as this method is usually called
            when the environment transitioned from a terminal state to an environment's start state.

        reward: float
            Reward received when going from `terminal_state` to `next_state`.

        done_episode: (opt) bool
            Whether the next_state corresponds to the end of an episode.
            This is used by the learn() method called by this process in order to determine whether the learning steps carried out by the learner
            at the end of an episode should be carried out (typically, whether the `if done` block of the learner's learn() method is executed or not).
            The default value is set to False because normally this method is called AFTER the fact, i.e. once the agent moved to a start state (next_state),
            wherein this method is used to learn the values of the *previous* state (terminal_state). Under this circumstance `done_episode=True`
            was passed to the learner's learn() method at THAT moment, i.e. when the environment transitioned to what is currently the *previous* state.
            However, done_episode should be set to True when the current state of the environment (next_state), typically a start state, occurs when
            the simulation ends, at which point the learner needs to perform all the actions that are performed when an episode ends
            (e.g. store the trajectory's history in the permanent self.states, self.actions, etc. attributes and update the state count).
            default: False

        envs: (opt) list of Environment
            In the FV simulation, all the environments associated to the particles in the system.

        idx_particle: (opt) int
            Index of `envs` corresponding to the particle that has just transitioned from a terminal state to a start state
            where such transition will be stored (as part of the particle's trajectory).
            This piece of information is used when learning the value functions at the occurrence of an absorption event
            --see the LeaFV.learn_at_absorption() method for more details on how the trajectory information is used.

        update_phi: (opt) bool
            See description of this parameter in the learner's learn() method (typically LeaFV.learn()).
            default: False
        """
        learner = self.agent.getLearner()

        # We update just one action for Q and then copy its value to the other Q values
        # In fact, all Q-values for the terminal state are the same because all the actions lead to a start state --defined by env.reset())
        action_anchor = 0

        # Learn
        if t <= 0:
            # We are learning the values of the terminal state at the start of an episode
            # => Do not update the trajectory nor the state count because they were updated at the end of the previous episode.
            # Note that, even if the simulation ends at the given `next_state` we set these flags to False because they concern the update of the trajectory and the counts
            # of the TERMINAL state visited at the previous step, NOT of the `next_state` to which the system transitions (which may happen to be the end state when the simulation
            # ends there). Note that the learner.learn() method takes care of updating the end state count when the episode is DONE, and this is informed by parameter done_episode.
            info['update_trajectory'] = False
            info['update_counts'] = False
        if envs is None:
            # We distinguish the call because only LeaFV.learn() method accepts more parameters than the other learners
            learner.learn(t, terminal_state, action_anchor, next_state, reward, done_episode, info)
        else:
            assert isinstance(learner, LeaFV)
            learner.learn(t, terminal_state, action_anchor, next_state, reward, done_episode, info, envs=envs, idx_particle=idx_particle, update_phi=update_phi)
        if t <= 0:
            info.pop('update_trajectory')
            info.pop('update_counts')

        # Copy the Q-value just learned for the anchor action to the Q-value of the other possible actions
        # TODO: (2024/01/17) Move this piece of code to a function as this is done already at two places at least
        for _action in range(self.env.getNumActions()):
            # TODO: (2023/11/23) Generalize this update of the Q-value to ANY function approximation as the following call to _setWeight() assumes that we are in the tabular case!!
            learner.getQ()._setWeight(terminal_state, _action, learner.getQ().getValue(terminal_state, action_anchor))
        # Check that all Q values are the same for the terminal state
        for _action in range(self.env.getNumActions()):
            assert np.isclose(learner.getQ().getValue(terminal_state, _action), learner.getQ().getValue(terminal_state, action_anchor)), \
                f"All Q-values are the same for the terminal state {terminal_state}:\n{learner.getQ().getValues()}"

    def simulate(self, nexperiments, nepisodes, max_time_steps_per_episode=None, compute_rmse=True, weights_rmse=None,
                 verbose=False, verbose_period=1, verbose_convergence=False, plot=False):
        """
        Simulates the agent interacting with the environment for a number of experiments and number
        of episodes per experiment.

        The seed of the simulation is set at the beginning of all experiments as the seed stored
        in the object (typically attribute `seed`).

        Parameters:
        nexperiments: int
            Number of experiments to run.

        nepisodes: int
            Number of episodes to run per experiment.

        max_time_steps_per_episode: (opt) int
            Maximum number of steps to run each episode for.
            default: np.Inf

        compute_rmse: (opt) bool
            Whether to compute the RMSE of the estimated value function over all states.
            Useful to analyze rate of convergence of the estimates when the true value function is known.

        weights_rmse: (opt) bool or array
            Whether to use weights when computing the RMSE or the weights to use.
            default: None

        verbose: (opt) bool
            Whether to show the experiment that is being run and the episodes for each experiment.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        verbose_convergence: (opt) bool
            Whether to monitor convergence by showing the change in value function estimate at each new episode
            w.r.t. the estimate in the previous episode.
            default: False

        plot: (opt) bool
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
        [seed] = self.env.setSeed(self.seed)
        for exp in np.arange(nexperiments):
            if verbose:
                print("Running experiment {} of {} (#episodes = {})..." \
                      .format(exp+1, nexperiments, nepisodes), end=" ")
            V, Q, n_visits_i, RMSE_by_episode_i, MAPE_by_episode_i, learning_info = \
                    self.run(nepisodes=nepisodes, max_time_steps_per_episode=max_time_steps_per_episode,
                             seed=None, # We pass seed=None so that the seed is NOT set for the experiment,
                                        # but it is set indirectly by the seed defined above, before the FOR loop.
                                        # Otherwise ALL experiments would have the same outcome!
                             compute_rmse=compute_rmse, weights_rmse=weights_rmse,
                             plot=plot,
                             verbose=verbose, verbose_period=verbose_period, verbose_convergence=verbose_convergence)
            # Number of visits made to each state at the end of the experiment
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
