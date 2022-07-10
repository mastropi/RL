# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:25:14 2022

@author: Daniel Mastropietro
@description: Definition of functions and classes used for queue simulation.
"""

import copy
import warnings
import tracemalloc
from enum import Enum, unique

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

from Python.lib.agents.learners.continuing.fv import LeaFV
from Python.lib.agents.policies import PolicyTypes
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep
from Python.lib.environments.queues import Actions
from Python.lib.simulators import check_done, show_messages, step, update_trajectory
from Python.lib.simulators.discrete import Simulator
from Python.lib.queues import Event

from Python.lib.utils.basic import find_last, get_current_datetime_as_string, get_datetime_from_string, is_scalar, \
    index_linear2multi, measure_exec_time, merge_values_in_time
from Python.lib.utils.computing import compute_job_rates_by_server, compute_blocking_probability_birth_death_process, \
    compute_survival_probability, generate_min_exponential_time
import Python.lib.utils.plotting as plotting


@unique
class LearningMode(Enum):
    "Learning mode of theta in the parameterized policy in SimulatorQueue.learn()"
    REINFORCE_RETURN = 1        # When learning is based on the observed return, which estimates the expected value giving grad(V)
    REINFORCE_TRUE = 2          # When learning is based on the expected value giving grad(V)
    IGA = 3                     # When learning is based on Integer Gradient Ascent, where delta(theta) is +/- 1 or 0.

@unique
class ReactivateMethod(Enum):
    RANDOM = 1                  # Random choice of the reactivation particle among the N-1 other non-absorbed particles
    VALUE_FUNCTION = 2          # Choice of the reactivation particle based on the value function of each state at which the other N-1 particles are located
    ROBINS = 3

DEBUG_ESTIMATORS = False
DEBUG_TRAJECTORIES = False


class SimulatorQueue(Simulator):
    """
    Simulator class that runs a Reinforcement Learning simulation on a given Queue environment `env` and an `agent`
    using the learning mode, number of learning steps, and number of simulation steps per learning step
    specified in the `dict_learning_params` dictionary.

    Arguments:
    env: Environment
        Queue environment is assumed to have the following methods defined:
        - reset()
        - getNumServers()
        - getJobClassRates()
        - getServiceRates()
        - getJobClass()
        - getQueueState()
        - getState()
        - setJobClass()

    agent: Agent
        Agent object that is responsible of performing the actions on the environment and learning from them.

    dict_learning_params: dict
        Dictionary containing the learning parameters as follows:
        - 'mode': learning mode, one of the values of the LearningMode enum
        - 't_learn': number of learning steps (of the value functions and policies)

    N: int
        Number of queues to simulate.
        default: 1
    """

    def __init__(self, env, agent, dict_learning_params, case=1, N=1, seed=None, log=False, save=False, logsdir=None, resultsdir=None, debug=False):
        super().__init__(env, agent, case, seed, log, save, logsdir, resultsdir, debug)
        self.dict_learning_params = dict_learning_params
        # Attributes used during Fleming-Viot simulation (each env in envs is one of the N particles)
        self.N = N
        # TODO: (2022/01/17) We should NOT copy ALL the environment to work with the N particles... In fact, the environment characteristics are the same for ALL particles, the only thing that changes is the latest state and change time of each particle
        # This may be behind the memory error I got when running the FV process at my laptop.

        # Note: it's important that the first environment stored in self.envs be self.env (and NOT a copy)
        # because in the process we sometimes refer to it as self.envs[0] (for instance when resetting the environments)
        # or as self.env (for instance when estimating Q(K-1, a) via Monte-Carlo on one particle (both in MC and FV modes)
        self.envs = [self.env if i == 0 else copy.deepcopy(self.env) for i in range(self.N)]

        # Learners for the value function and the policy
        if agent is not None:
            self.learnerV = self.agent.getLearnerV()
            self.learnerP = self.agent.getLearnerP()
            # Job-rates by server (assuming jobs are pre-assigned to servers)
            self.job_rates_by_server = compute_job_rates_by_server(self.env.getJobClassRates(),
                                                                   self.env.getNumServers(),
                                                                   self.agent.getAssignmentPolicy().getProbabilisticMap())
        else:
            self.learnerV = None
            self.learnerP = None
            self.job_rates_by_server = None

        # Storage of learning history
        self.alphas = []            # Learning rates used in the learning of the policy when updating from theta to theta_next (i.e. from thetas -> thetas_updated)
        self.thetas = []            # List of theta values before the update by the policy learning step
        self.thetas_updated = []    # List of theta values AFTER the update by the policy learning step.
        self.proba_stationary = []  # List of the estimates of the stationary probability of K-1 at theta
        self.Q_diff = []            # List of the Q(K-1,a=1) - Q(K-1,a=0) values leading to the udpate of theta
        self.V = []                 # *Normally*, the value function at the theta before the update,
                                    # i.e. the average reward rho observed in each queue simulation episode for a fixed policy before the update of theta takes place.
                                    # HOWEVER, it is NOT always the case that this is the average reward, see the self.learn() method for more details.
        self.gradV = []             # Gradient of the value function (grad(J) in Sutton) for the theta before the update, that is responsible for the theta update.
        self.G = None               # G(t) for each simulation time step t as estimated by self.learn().
        self.n_events_mc = []       # number of events used in the Monte-Carlo simulation, either to estimate the stationary probabilities under the MC approach, or P(T>t) and E(T_A) under the FV approach.
        self.n_events_fv = []       # number of events used to estimate the stationary probability in each episode.
        self.n_trajectories_Q = []  # Average number of trajectory pairs used to estimate Q_diff in each episode.
                                    # The average is over the trajectory pairs used to estimate Q_diff(K-1) and Q_diff(K)

    # IMPORTANT: This method overrides the method of the superclass! (so when reset() is called from the super()
    # object, actually THIS reset() method defined here is called!
    def reset(self, state=None, reset_value_functions=False, reset_counts=False):
        "Resets the environments used in the simulation to the given state and the learners for a new learning process"
        # Reset the environments (particles) used in the simulation
        if state is not None:
            for env in self.envs:
                env.reset(state)

        if self.agent is not None:
            # Reset the learners
            # This is expected to be called after each learning step, i.e. before a new queue simulation is run
            # Based on this, note that we do NOT reset the trajectory stored in the policy learner because we want to
            # keep the WHOLE history of learning (i.e. for all queue simulation episodes) in the policy learner.
            # Note that we reset the time and alphas of the value function learner (because this happens within each
            # learning step or queue simulation episode), but we do NOT reset the time and alphas of the policy
            # learner because the learning step here is defined by a new queue simulation episode.
            if self.agent.getLearnerV() is not None:
                self.agent.getLearnerV().reset(reset_time=True,  reset_alphas=True,  reset_value_functions=reset_value_functions, reset_trajectory=True,  reset_counts=reset_counts)
            if self.agent.getLearnerP() is not None:
                self.agent.getLearnerP().reset(reset_time=False, reset_alphas=False, reset_value_functions=reset_value_functions, reset_trajectory=False, reset_counts=reset_counts)
            if not reset_value_functions:
                # Set the start value of the value functions to their current estimate
                # Goal: we can use the current estimated values in the next learning step,
                # e.g. use it as a baseline when computing the time difference delta(t).
                if self.agent.getLearnerV() is not None:
                    self.agent.getLearnerV().setVStart(self.agent.getLearnerV().getV())
                    self.agent.getLearnerV().setQStart(self.agent.getLearnerV().getQ())

    def run(self, dict_params_simul: dict, dict_params_info: dict={'plot': False, 'log': False}, dict_info: dict={},
            start_state=None,
            seed=None,
            verbose=False, verbose_period=1,
            colormap="seismic", pause=0):
        """
        Runs a Reinforcement Learning experiment on a queue environment for the given number of time steps.
        No reset of the learning history is done at the beginning of the simulation.
        It is assumed that this reset, when needed, is done by the calling function.

        Parameters:
        dict_params_simul: dict
            Dictionary containing the simulation parameters as follows:
            - 'theta_true': true value of the theta parameter to use for the simulation, which should be estimated
            by the process.
            - 'theta_start': initial value of theta at which the learning process starts.
            - 'nparticles': # particles to use in the FV simulation.
            - 't_sim': number of simulation steps per learning step. This is either a scalar or a list with as many
            elements as the number of learning steps defined in self.dict_learning_params['t_learn'].
            This latter case serves the purpose of running as many simulation steps as were run in a benchmark method
            (for instance when comparing the results by Monte-Carlo with those by Fleming-Viot).
            - 'buffer_size_activation_factor': factor multiplying the blocking size K defining the value of
            the buffer size for activation.

        dict_params_info: (opt) dict --> only used when the learner of the value functions `learnerV` is LeaFV
            Dictionary containing general info parameters as follows:
            - 'plot': whether to create plots
            - 'log': whether to show messages
            default: None

        dict_info: (opt) dict
            Dictionary with information of interest to store in the results file.
            From this dictionary, when defined, we extract the following keys:
            - 'exponent': the exponent used to define the values of N (# particles) and T (# time steps) used in the
            simulation.
            default: {}

        start_state: (opt) int or list or numpy array
            State at which the queue environment starts.
            Its type depends on how the queue environment defines the state of the system.
            default: None, in which case the start state is defined by the simulation process, based on the learning requirements

        seed: (opt) None or float
            Seed to use for the random number generator for the simulation.
            If None, the seed is NOT set.
            If 0, the seed stored in the object is used. I don't know when this value would be useful, but certainly
            NOT when running different experiments on the same environment and parameter setup, because in that case
            all experiments would start with the same seed!
            default: None

        verbose: (opt) bool
            Whether to show the experiment that is being run and the episodes for each experiment.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        plot: (opt) bool
            Whether to generate plots showing the evolution of the value function estimates.

        colormap: (opt) str
            Name of the colormap to use in the generation of the animated plots
            showing the evolution of the value function estimates.
            It must be a valid colormap among those available in the matplotlib.cm module.

        pause: (opt) float
            Number of seconds to wait before updating the plot that shows the evalution
            of the value function estimates.

        Returns: tuple
        Tuple containing the following elements:
            - learnerV: the learner of the value function at the end of the learning process
            - learnerP: the learner of the policy at the end of the learning process.
                The policy can be retrieved from this object by calling learnerP.getPolicy().
            - df_learning: pandas data frame containing two columns: 'theta', 'gradV' containing the history
            of the observed thetas and gradients of the value function responsible for its update.
        """
        # -- Parse input parameters
        set_required_entries_params_simul= {'theta_true', 'theta_start', 'nparticles', 't_sim', 'buffer_size_activation_factor'}
        if not set_required_entries_params_simul.issubset(dict_params_simul.keys()):
            raise ValueError("Missing entries in the dict_params_simul dictionary: {}" \
                             .format(set_required_entries_params_simul.difference(dict_params_simul.keys())))

        if is_scalar(dict_params_simul['t_sim']):
            # Case when the simulation time (simulation steps, which normally equals # events) is the same
            # for all learning steps (typically when learning via the FV method).
            # Otherwise, in the MC method, the number of simulation steps is normally read from a benchmark file
            # and therefore is expected to be different for each learning step.
            # Note that we define the 'nmeantimes' key because this is the key required by the
            # EstimatorQueueBlockingFlemingViot class that is used to run the single particle simulation
            # to estimate P(T>t) and E(T_A) in the FV method.
            # => Use the number of simulation steps defined in the object
            dict_params_simul['nmeantimes'] = dict_params_simul['t_sim']    # This is used in Fleming-Viot simulation
            dict_params_simul['T'] = dict_params_simul['t_sim']                              # This is used in Monte-Carlo simulation

        dict_params_info['verbose'] = verbose
        dict_params_info['verbose_period'] = verbose_period

        if dict_params_info['plot']:
            # TODO: See the run() method in the Simulator class
            pass

        # Compute the server intensitites which is used to compute the expected relative errors associated to nparticles and t_sim
        rhos = [l / m for l, m in zip(self.getJobRatesByServer(), self.env.getServiceRates())]

        # Set the number of particles to use in the simulation
        self.setNumberParticlesAndCreateEnvironments(dict_params_simul['nparticles'])
        assert self.N == dict_params_simul['nparticles']

        # Set the true theta parameter (which is defined by a parameter of the queue environment) to use for the simulation
        # NOTE that this is done in EACH environment used in the simulation, because this environment is the one passed to functions
        # and the one that is used to retrieve the value of the true theta parameter.
        for env in self.envs:
            env.setParamsRewardFunc(dict({'buffer_size_ref': np.ceil(dict_params_simul['theta_true'] + 1)}))

        # Simulation seed
        # NOTE: This seed is used as the base seed to generate the seeds that are used at each learning step
        if seed is not None:
            if seed != 0:
                self.env.set_seed(seed)
            else:
                # Use the seed stored in the object, which was defined when the object was created
                self.env.set_seed(self.seed)
        # -- Parse input parameters

        dt_start = get_current_datetime_as_string()
        print("Simulation starts at: {}".format(dt_start))

        # -- Reset the learners
        # Reset the policy learner (this is especially important to reset the learning rate alpha to its initial value!)
        self.getLearnerP().reset()
        # Reset the theta value to the initial theta
        self.getLearnerP().getPolicy().setThetaParameter(dict_params_simul['theta_start'])

        if verbose:
            print("Value function at start of experiment: {}".format(self.getLearnerV().getV()))

        # There are two simulation time scales (both integers and starting at 0):
        # - t: the number of queue state changes (a.k.a. number of queue iterations)
        # - t_learn: the number of learning steps (of the value functions and policies)
        t_learn = 0
        while t_learn < self.dict_learning_params['t_learn']:
            t_learn += 1
            dict_params_info['t_learn'] = t_learn
            # Reset the learners
            self.reset(reset_value_functions=False, reset_counts=False)

            # Set the seed for this learning step (so we can reproduce whatever happens at each learning step,
            # without the need to run ALL learning steps prior to it --which would be the case if we don't set
            # the seed at each learning step as we do here)
            dict_params_simul['seed'] = self.env.get_seed() + t_learn - 1

            if show_messages(True, verbose_period, t_learn):
                print("\n************************ Learning step {} of {} running ****************************" \
                      .format(t_learn, self.dict_learning_params['t_learn']))
                print("Theta parameter of policy: {}".format(self.getLearnerP().getPolicy().getThetaParameter()))

            # Buffer size of sure blocking, which is used to define the buffer size for activation and the start state for the simulations run
            # We set the activation buffer size as a specific fraction of the buffer size K of sure blocking.
            # This fraction is chosen as the optimum found theoretically to have the same complexity in reaching
            # a signal that leads to estimations of the numerator Phi(t) and the denominator E(T1+T2) in the
            # FV estimation of the stationary probability (found on Fri, 07-Jan-2022 at IRIT-N7 with Matt, Urtzi and
            # Szymon), namely J = K/3.
            if self.getLearnerP() is not None:
                assert self.agent.getAcceptancePolicy().getThetaParameter() == self.getLearnerP().getPolicy().getThetaParameter()
            K = self.agent.getAcceptancePolicy().getBufferSizeForDeterministicBlocking()
            dict_params_simul['buffer_size_activation'] = np.max([1, int( np.round(dict_params_simul['buffer_size_activation_factor']*K) )])
            if isinstance(self.getLearnerV(), LeaFV):
                # TODO: (2021/12/16) SHOULD WE MOVE ALL this learning process of the average reward and stationary probabilities to the learn() method of the LeaFV class?
                # At this point I don't think we should do this, because the learn() method in the GenericLearner subclasses
                # learns the value function from the RESULTS of a SIMULATION ALREADY CARRIED OUT.
                # (see for instance the LeaMC.learn() method)
                # If we do the above change, we should ALSO change the learn() method of LeaMC to perform the simulation
                # of the system itself... and I don't think the learn() method should be responsible for the simulation
                # because simulating is a quite computing intensive task!!

                if not is_scalar(dict_params_simul['t_sim']):
                    # There is a different simulation time for each learning step
                    assert len(dict_params_simul) == self.dict_learning_params['t_learn']
                    dict_params_simul['nmeantimes'] = dict_params_simul['t_sim'][t_learn - 1]
                    dict_params_simul['T'] = dict_params_simul['nmeantimes']

                #-- Estimation of the survival probability P(T>t) and the expected absorption cycle time, E(T_A) by MC
                if show_messages(verbose, verbose_period, t_learn):
                    print("Running Monte-Carlo simulation on one particle starting at an absorption state with buffer size = {}..." \
                          .format(dict_params_simul['buffer_size_activation'] - 1))
                #est_mc, expected_cycle_time, n_absorption_time_observations, \
                #df_proba_surv, n_survival_curve_observations = \
                #    estimate_proba_survival_and_expected_absorption_time_mc(self.env, self.agent, dict_params_simul,
                #                                                            dict_params_info)
                #n_events_mc = est_mc.nevents
                #n_cycles = n_absorption_time_observations

                proba_blocking_fv, expected_reward, probas_stationary, \
                    expected_cycle_time, n_cycles, time_last_absorption, time_end_simulation_et, max_survival_time, time_end_simulation_fv, \
                        n_events_mc, n_events_fv = estimate_blocking_fv(self.envs, self.agent, dict_params_simul, dict_params_info)
                # Equivalent (discrete) simulation time, as if it were a Monte-Carlo simulation
                t = n_events_mc + n_events_fv
                self.getLearnerV().setAverageReward(expected_reward)
                self.getLearnerV().setProbasStationary(probas_stationary)
            else:
                # Monte-Carlo learning is the default
                assert self.N == 1, "The simulation system has only one particle in Monte-Carlo mode ({})".format(self.N)

                # Define the following two quantities as None because they are sent to the output results file
                # in the FV case and here they should be found as existing variables.
                if not is_scalar(dict_params_simul['t_sim']):
                    # There is a different simulation time for each learning step
                    # => Use a number of simulation steps defined in a benchmark for each learning step
                    assert len(dict_params_simul['t_sim']) == self.dict_learning_params['t_learn'], \
                            "The number of simulation steps read from the benchmark file ({}) coincides with the number of learning steps ({})" \
                            .format(len(dict_params_simul['t_sim']), self.dict_learning_params['t_learn'])
                    dict_params_simul['T'] = dict_params_simul['t_sim'][t_learn-1]
                proba_blocking_mc, expected_reward, probas_stationary, expected_cycle_time, n_cycles, n_events = \
                    estimate_blocking_mc(self.env, self.agent, dict_params_simul, dict_params_info)
                # Discrete simulation time
                t = n_events
                n_events_mc = n_events
                n_events_fv = 0  # This information is output to the results file and therefore we set it to 0 here as we are in the MC approach
                max_survival_time = 0.0

            # Compute the state-action values (Q(s,a)) for buffer size = K-1
            if probas_stationary.get(K-1, 0.0) == 0.0:
                Q0_Km1 = 0.0
                Q1_Km1 = 0.0
                n_Km1 = 0
                if True or DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                    print("Estimation of Q_diff(K-1) skipped because the stationary probability Pr(K-1) is {}".format(probas_stationary.get(K-1)))
                # Set Pr(K-1) to 0 in case it was not computed, so that learns the policy below (`learn()`)
                # receives the input parameters as expected (e.g. probas_stationary should have 2 keys: K-1 and K
                probas_stationary[K-1] = 0.0
            else:
                # When the parameterized policy is a linear step linear function with only one buffer size where its
                # derivative is non-zero, only the DIFFERENCE of two state-action values impacts the gradient, namely:
                # Q(K-1, a=1) - Q(K-1, a=0)
                if DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                    print("\nEstimating the difference of the state-action values when the initial buffer size is K-1={}...".format(K-1))
                N = 100;
                t_sim_max = 250
                K, Q0_Km1, Q1_Km1, n_Km1, max_t_Km1 = \
                    self.estimate_Q_values_until_mixing(t_learn, K - 1, t_sim_max=t_sim_max, N=N, \
                                                        seed=dict_params_simul['seed'] * 100, verbose=verbose,
                                                        verbose_period=verbose_period)
                # K, Q0_Km1, Q1_Km1, n, max_t = self.estimate_Q_values_until_stationarity(t_learn, t_sim_max=50, N=N, verbose=verbose, verbose_period=verbose_period)
                if True or DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                    print("--> Estimated state-action values on n={} realizations out of {} with max simulation time = {:.1f} out of {:.1f}:\nQ(K-1={}, a=1) = {}\nQ(K-1={}, a=0) = {}\nQ_diff = Q(K-1,1) - Q(K-1,0) = {}" \
                        .format(n_Km1, N, max_t_Km1, t_sim_max, K-1, Q1_Km1, K-1, Q0_Km1, Q1_Km1 - Q0_Km1))
            if True or DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                print("--> Estimated stationary probability: Pr(K-1={}) = {}".format(K-1, probas_stationary[K-1]))

            # Compute the state-action values (Q(s,a)) for buffer size = K
            # This is needed ONLY when the policy learning methodology is IGA (Integer Gradient Ascent, presented in Massaro's paper (2019)
            if self.dict_learning_params['mode'] != LearningMode.IGA or probas_stationary.get(K, 0.0) == 0.0:
                # We set all the values to 0 so that the learn() method called below doesn't fail, as it assumes that
                # the values possibly involved with learning (which depend on the learning method) have a real value.
                Q0_K = 0.0
                Q1_K = 0.0
                n_K = 0
                if DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                    print("Estimation of Q_diff(K) skipped because the stationary probability Pr(K) is {}".format(probas_stationary.get(K-1)))
                # Set Pr(K) to 0 in case it was not computed, so that learns the policy below (`learn()`)
                # receives the input parameters as expected (e.g. probas_stationary should have 2 keys: K-1 and K
                probas_stationary[K] = 0.0
            else:
                # Same as above, but for buffer size = K
                if DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                    print("\nEstimating the difference of the state-action values when the initial buffer size is K={}...".format(K))
                N = 100;
                t_sim_max = 250
                K, Q0_K, Q1_K, n_K, max_t_K = \
                    self.estimate_Q_values_until_mixing(t_learn, K, t_sim_max=t_sim_max, N=N, \
                                                        seed=dict_params_simul['seed'] * 100 + 1, verbose=verbose,
                                                        verbose_period=verbose_period)
                # K, Q0_K, Q1_K, n_K, max_t_K = self.estimate_Q_values_until_stationarity(t_learn, t_sim_max=50, N=N, verbose=verbose, verbose_period=verbose_period)
                if show_messages(verbose, verbose_period, t_learn):
                    print("--> Estimated state-action values on n={} realizations out of {} with max simulation time = {:.1f} out of {:.1f}:\nQ(K={}, a=1) = {}\nQ(K={}, a=0) = {}\nQ_diff = Q(K,1) - Q(K,0) = {}" \
                        .format(n_K, N, max_t_K, t_sim_max, K, Q1_K, K, Q0_K, Q1_K - Q0_K))
            if show_messages(verbose, verbose_period, t_learn):
                print("--> Estimated stationary probability: Pr(K={}) = {}".format(K, probas_stationary[K]))

            # Learn the value function and the policy
            theta_prev = self.getLearnerP().getPolicy().getThetaParameter()

            # Use this when using REINFORCE to learn theta: LearningMode.REINFORCE_RETURN
            #self.learn(self.agent, t)
            # Use this when estimating the theoretical grad(V): LearningMode.REINFORCE_TRUE or LearningMode.IGA
            err_phi, err_et = compute_rel_errors_for_fv_process(rhos, K, self.N, dict_params_simul['T'], dict_params_simul['buffer_size_activation_factor'])
            self.learn(self.agent, t,
                       probas_stationary=probas_stationary,
                       Q_values=dict({K-1: [Q0_Km1, Q1_Km1], K: [Q0_K, Q1_K]}),
                       simul_info=dict({'t_learn': t_learn,
                                        'K': K,
                                        'J_factor': dict_params_simul['buffer_size_activation_factor'],
                                        'J': dict_params_simul['buffer_size_activation'],
                                        'exponent': dict_info.get('exponent'),
                                        'N': self.N,
                                        'T': dict_params_simul['T'],
                                        'err_phi': err_phi,
                                        'err_et': err_et,
                                        'seed': dict_params_simul['seed'],
                                        'expected_cycle_time': expected_cycle_time,
                                        'n_cycles': n_cycles,
                                        'max_survival_time': max_survival_time,
                                        'n_events_mc': n_events_mc,
                                        'n_events_fv': n_events_fv,
                                        'n_Q': np.mean([n_Km1, n_K] if self.dict_learning_params['mode'] == LearningMode.IGA else n_Km1)}))

            if show_messages(verbose, verbose_period, t_learn):
                print("\tUpdated value function at the end of the queue simulation: average reward V = {}".format(self.getLearnerV().getV()))
                print("\tSame observed average reward (computed from Policy learner) = {}".format(self.getLearnerP().getAverageRewardUnderPolicy()))
                print("\tUpdated theta parameter of policy after learning: theta = {} -> {}".format(theta_prev, self.getLearnerP().getPolicy().getThetaParameter()))

        if dict_params_info['plot']:
            # TODO
            pass

        df_learning = pd.DataFrame.from_items([
                                            ('theta', self.thetas),
                                            ('theta_next', self.thetas_updated),
                                            ('Pr(K-1)', [p[0] for p in self.proba_stationary]),
                                            ('Pr(K)', [p[1] for p in self.proba_stationary]),
                                            ('Q_diff(K-1)', [q[0] for q in self.Q_diff]),
                                            ('Q_diff(K)', [q[1] for q in self.Q_diff]),
                                            ('alpha', self.alphas),
                                            ('V', self.V),
                                            ('gradV', self.gradV),
                                            ('n_events_mc', self.n_events_mc),
                                            ('n_events_fv', self.n_events_fv),
                                            ('n_trajectories_Q', self.n_trajectories_Q)
                                                ])

        dt_end = get_current_datetime_as_string()
        print("Simulation ends at: {}".format(dt_end))
        datetime_diff = get_datetime_from_string(dt_end) - get_datetime_from_string(dt_start)
        time_elapsed = datetime_diff.total_seconds()
        print("Execution time: {:.1f} min, {:.1f} hours".format(time_elapsed / 60, time_elapsed / 3600))

        return self.getLearnerV(), self.getLearnerP(), df_learning

    @measure_exec_time
    def run_simulation_mc_DT(self, t_learn, start_state, t_sim_max, agent=None, verbose=False, verbose_period=1):
        """
        Runs the discrete-time (DT) simulation using Monte-Carlo

        Arguments:
        t_learn: int
            The learning time step to which the simulation will contribute.

        start_state: int or list or numpy array
            State at which the queue environment starts for the simulation.
            Its type depends on how the queue environment defines the state of the system.

        t_sim_max: int
            Maximum discrete simulation time steps allowed for the simulation. Time steps are defined
            ONLY by job arrivals, NOT by service events.

        agent: (opt) Agent
            Agent object that is responsible of performing the actions on the environment and learning from them.
            default: None, in which case the agent defined in the object is used

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: int
        The last time step of the simulation process.
        """
        #-- Parse input parameters
        if agent is None:
            agent = self.agent
        tmax = t_sim_max

        # Set the start state of the environment to the given start state
        job_class = None
        self.env.setState((start_state, job_class))
        if verbose:
            print("MC simulation: The queue environments starts at state {}".format(self.env.getState()))

        # Time step in the queue trajectory (the first time step is t = 0)
        done = False
        t = 0
        event_prev = None
        while not done:
            # Current state
            state = self.env.getState()

            # Next state
            # Note: this function takes care of NOT generating a service event when a server is empty.
            time, event, job_class_or_server, _ = generate_event([self.env])
            if event == Event.BIRTH:
                # The event is an incoming job class
                # => Increment the discrete time step, set the arriving job class in the queue environment, and apply the acceptance policy
                # NOTE: Only BIRTH events mark a new discrete time
                # (this guarantees that the stationary distribution of the discrete-time process coincides with that of the continuous-time process)
                t += 1

                action, next_state, reward, gradient_for_action = manage_job_arrival(t, self.env, agent, state, job_class_or_server)

                # Update the trajectory used in the learning process, where we store:
                # S(t): state BEFORE an action is taken
                # A(t): action taken given the state S(t)
                # R(t): reward received by taking action A(t) and transition to S(t+1)
                update_trajectory(agent, (t_learn - 1) * (tmax + 1) + t, t, state, action, reward)

                done = check_done(tmax, t, state, action, reward)
            elif event == Event.DEATH:
                # The event is a completed service
                # => Update the state of the queue but do NOT update the discrete-time step
                action, next_state, reward = manage_service(self.env, agent, state, job_class_or_server)
                if event_prev == Event.BIRTH or event_prev is None:
                    # Add the state PRIOR to the FIRST DEATH after a BIRTH (or the very first DEATH event
                    # if that is the first event that occurs in the simulation), so that we record the state
                    # to which the system went just after the latest BIRTH (so that the plot that is generated
                    # at the end showing the states of the system at each time step does not raise suspicion
                    # because the buffer size doesn't change between two consecutive time steps --which would be
                    # inconsistent with the fact that a new time step is defined when a new job arrives).
                    update_trajectory(agent, (t_learn - 1) * (tmax + 1) + t, t, state, action, reward)

            if self.debug:
                print("{} | t={}: event={}, action={} -> state={}, reward={}".format(state, t, event, action, next_state, reward), end="\n")

            event_prev = event

        # DONE
        if show_messages(verbose, verbose_period, t_learn):
            print("==> agent ENDS at time t={} at state {} coming from state = {}, action = {}, reward = {}, gradient = {})" \
                    .format(t, self.env.getState(), state, action, reward, gradient_for_action))

        return t

    def estimate_stationary_probabilities_mc_DT(self):
        """
        Monte-Carlo estimation of the stationary probability of K-1 and K based on the observed trajectory
        in discrete-time, i.e. where only incoming jobs trigger a time step.

        Return: dict
        Dictionary with the estimated stationary probability for buffer sizes K-1 and K, where K is the first integer
        larger than or equal to theta + 1  --theta being the parameter of the linear step acceptance policy.
        """
        states_with_actions = [s for s, a in zip(self.getLearnerV().getStates(), self.getLearnerV().getActions()) if a is not None]
        buffer_sizes = [self.env.getBufferSizeFromState(s) for s in states_with_actions]
        # print("buffer sizes, actions = {}".format(np.c_[buffer_sizes, actions]))
        if self.getLearnerP() is not None:
            assert self.agent.getAcceptancePolicy().getThetaParameter() == self.getLearnerP().getPolicy().getThetaParameter()
        K = self.agent.getAcceptancePolicy().getBufferSizeForDeterministicBlocking()
        probas_stationary = dict({K-1:  np.sum([1 for bs in buffer_sizes if bs == K-1]) / len(buffer_sizes),
                                  K:    np.sum([1 for bs in buffer_sizes if bs == K]) / len(buffer_sizes)})

        return probas_stationary

    @measure_exec_time
    def estimate_Q_values_until_stationarity(self, t_learn, t_sim_max=50, N=100, verbose=False, verbose_period=1):
        """
        Estimates the state-action values for the case of a parameterized policy defined as a linear function of theta
        with only one buffer size with non-zero derivative.
        The state-action values are computed for the two possible accept/reject actions of an incoming job when
        the queue system is at that buffer size.

        Arguments:
        t_learn: int
            Learning time step.

        t_sim_max: (opt) int
            Maximum discrete time for the simulation, after which stationary is assumed to have been reached.
            default: 50

        N: (opt) int
            Number of samples used to estimate the state-action values.
            default: 100

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: tuple
        Triple with the following elements:
        - K: the buffer size at which an incoming job is always rejected by the policy (deterministic policy)
        - The state-action value for (K-1, a=Reject)
        - The state-action value for (K-1, a=Accept)
        - N: the number of realizations used to compute the state-action values
        """
        if not isinstance(self.getLearnerP().getPolicy(), PolQueueTwoActionsLinearStep):
            raise ValueError("The agent's policy must be of type {} ({})" \
                             .format(type(PolQueueTwoActionsLinearStep), type(self.getLearnerP().getPolicy())))

        if self.getLearnerP() is not None:
            assert self.agent.getAcceptancePolicy().getThetaParameter() == self.getLearnerP().getPolicy().getThetaParameter()
        K = self.agent.getAcceptancePolicy().getBufferSizeForDeterministicBlocking()

        # 1) Starting at (K-1, a=1) is like starting at s=K (because the reward of the first action is 0)
        if show_messages(verbose, verbose_period, t_learn):
            print("\n1) Running simulation on N={} queues for {} time steps to estimate Q(K-1, a=1)...".format(N, t_sim_max))
        start_state = choose_state_for_buffer_size(self.env, K)
        Q1 = 0.0
        for _ in range(N):
            agent_Q = copy.deepcopy(self.agent)
            agent_Q.getLearnerV().reset(reset_value_functions=True, reset_trajectory=True, reset_counts=True)
            agent_Q.getLearnerP().reset(reset_value_functions=True, reset_trajectory=True, reset_counts=True)
            self.run_simulation_mc_DT(t_learn, start_state, t_sim_max=t_sim_max, agent=agent_Q, verbose=False, verbose_period=verbose_period)
            Q1 += np.nansum(agent_Q.getLearnerV().getRewards())  # Note: the rewards are not NaN only when an action was taken (i.e. at job arrivals)
            del agent_Q
        Q1 /= N

        # 2) Starting at (K-1, a=0) is like starting at s=K-1 and adding one reward for the first rejection
        if show_messages(verbose, verbose_period, t_learn):
            print("2) Running simulation on N={} queues for {} time steps to estimate Q(K-1, a=0)...".format(N, t_sim_max))
        start_state = choose_state_for_buffer_size(self.env, K-1)
        # Reward received from the initial rejection of the job at buffer size K-1
        first_reward = compute_reward_for_buffer_size(self.env, K-1)
        Q0 = 0.0
        for _ in range(N):
            agent_Q = copy.deepcopy(self.agent)
            agent_Q.getLearnerV().reset(reset_value_functions=True, reset_trajectory=True, reset_counts=True)
            agent_Q.getLearnerP().reset(reset_value_functions=True, reset_trajectory=True, reset_counts=True)
            self.run_simulation_mc_DT(t_learn, start_state, t_sim_max=t_sim_max, agent=agent_Q, verbose=False, verbose_period=verbose_period)
            Q0 += np.nansum(agent_Q.getLearnerV().getRewards())  # Note: the rewards are not NaN only when an action was taken (i.e. at job arrivals)
            del agent_Q
        Q0 = Q0 / N + first_reward  # The first reward is the same for all queues because it only depends on the buffer size which is the same for all queues and equal to K-1

        return K, Q0, Q1, N, t_sim_max

    @measure_exec_time
    def estimate_Q_values_until_mixing(self, t_learn, buffer_size, t_sim_max=250, N=100, seed=None, verbose=False, verbose_period=1):
        """
        Estimates the state-action values for the case of a parameterized policy defined as a linear function of theta
        with only one buffer size with non-zero derivative.
        The state-action values are computed for the two possible accept/reject actions of an incoming job when
        the queue system is at that buffer size.

        Arguments:
        t_learn: int
            Learning time step.

        buffer_size: int
            Buffer size at which the system starts when estimating Q(buffer_size, action).

        t_sim_max: (opt) int
            Maximum discrete time for the simulation. If this is reached without mixing, the estimated state-action
            values are set to NaN.
            default: 250

        N: (opt) int
            Number of samples used to estimate the state-action values.
            default: 100

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: tuple
        Triple with the following elements:
        - K: the buffer size at which an incoming job is always rejected by the policy (deterministic policy)
        - The state-action value for (buffer_size, a=Reject)
        - The state-action value for (buffer_size, a=Accept)
        - n: the number of realizations, where trajectories mix, that are used to compute the state-action values
        - Maximum discrete time at which the pair of trajectories mix over the n pair of trajectories that mix
        """
        if not isinstance(self.getLearnerP().getPolicy(), PolQueueTwoActionsLinearStep):
            raise ValueError("The agent's policy must be of type {} ({})" \
                             .format(type(PolQueueTwoActionsLinearStep), type(self.getLearnerP().getPolicy())))

        if self.getLearnerP() is not None:
            assert self.agent.getAcceptancePolicy().getThetaParameter() == self.getLearnerP().getPolicy().getThetaParameter()
        K = self.agent.getAcceptancePolicy().getBufferSizeForDeterministicBlocking()

        np.random.seed(seed)

        Q0 = 0.0
        Q1 = 0.0
        n = 0           # Number of times Q0 and Q1 can be estimated from mixing trajectories (out of N)
        max_t_mix = 0.0 # Maximum discrete mixing time observed over the n pair of trajectories that mix
        for i in range(N):
            if True:
                print("Estimating Qdiff for buffer size bs={}... {} out of {} replications".format(buffer_size, i, N))
            _, Q0_, Q1_, t_mix = self.run_simulation_2_until_mixing(t_learn, buffer_size, t_sim_max, \
                                                                    verbose=verbose, verbose_period=verbose_period)
            if not np.isnan(Q0_) and not np.isnan(Q1_):
                Q0 += Q0_
                Q1 += Q1_
                n += 1
                max_t_mix = np.max([max_t_mix, t_mix])
        if n == 0:
            warnings.warn("Q_diff cannot be estimated because no mixing occurred for any of the N={} simulated trajectory pairs. It will be estimated as 0.0".format(N))
        else:
            Q0 /= n
            Q1 /= n

        return K, Q0, Q1, n, max_t_mix

    def run_simulation_2_until_mixing(self, t_learn, buffer_size, t_sim_max, verbose=False, verbose_period=1):
        """
        Runs the two simulations of the queue starting at the given buffer size, one with first action Reject,
        and the other with first action Accept, until the two trajectories mix or the given maximum simulation time
        is achieved.

        Arguments:
        t_learn: int
            The learning time step to which the simulation will contribute.

        buffer_size: int
            Buffer size at which the two simulations start.

        t_sim_max: int
            Maximum discrete simulation time allowed for the simulation. This is defined ONLY by job arrivals,
            NOT by service events.

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: tuple
        Triple with the following elements:
        - t: the last time step of the simulation process.
        - Q0: the sum of the rewards observed until mixing starting at state-action (buffer_size, a=Reject)
        - Q1: the sum of the rewards observed until mixing starting at state-action (buffer_size, a=Accept)
        - mixing time: discrete time at which the two trajectories mix. This is np.nan if they don't mix.
        """

        # ------------------------------ Auxiliary functions ----------------------------------#
        def apply_event(env, t, event, job_class_or_server):
            """
            Apply the given event, either a BIRTH with the incoming job class or a DEATH event at the given server,
            on the given environment that is currently at discrete time t.

            Arguments:
            env: Queue environment
                Queue environment where the event takes place.

            t: int
                Discrete time at which the environment is when the event occurs.

            event: Event
                Type of event that takes place in the environment, either Event.BIRTH or Event.DEATH.

            job_class_or_server: int
                Index associated to the incoming job class (when event = Event.BIRTH) or to the server that finishes
                serving a job (when event = Event.DEATH).

            Return: tuple
            Tuple with the following elements:
            - t': the new time after the event is applied. This increases only for BIRTH events, i.e. for arrival
            of new jobs.
            - state: S(t), the state at which the environment is prior to applying the action.
            - action: A(t), the action taken by the agent at state S(t).
            - next_state: S(t'), the next state at which the environment is in after taking action A(t).
            - reward: the reward received by the agent for taking action A(t).
            """
            # Get the current state of the selected particle because that's the one whose state we are going to possibly change
            state = env.getState()
            if event == Event.BIRTH:
                # The event is an incoming job class
                # => Increase the discrete time step and set the arriving job class in the queue environment and apply the acceptance policy
                t += 1

                # Store the class of the arriving job
                job_class = job_class_or_server
                env.setJobClass(job_class)

                # Perform a step on the environment by applying the acceptance policy
                action_accept_reject, next_state, reward_accept_reject, _ = step(t, env, self.agent, PolicyTypes.ACCEPT)

                # Assign the job to a server if accepted
                if action_accept_reject == Actions.ACCEPT:
                    # Assign the job just accepted to a server and update the next_state value
                    # (because the next_state from the ACCEPT action has not been updated above
                    # as we need to know to which server the accepted job is assigned in order
                    # to know the next state of the queue environment)
                    _, next_state, reward_assign, _ = step(t, env, self.agent, PolicyTypes.ASSIGN)
                else:
                    reward_assign = 0.0

                action = action_accept_reject
                reward = reward_accept_reject + reward_assign
                if action == Actions.ACCEPT and reward != 0.0:
                    print("--> action = {}, REWARD ASSIGN: {}, REWARD = {}".format(action_accept_reject, reward_assign, reward))
                    raise ValueError("A non-zero reward is not possible when the action is ACCEPT.")
            else:
                assert event == Event.DEATH
                # The event is a completed service
                # => Update the state of the queue but do not increase t
                server = job_class_or_server
                queue_state = env.getQueueState()
                assert queue_state[server] > 0, "The server where the completed service occurs has at least one job"
                next_queue_state = copy.deepcopy(queue_state)
                next_queue_state[server] -= 1
                env.setState((next_queue_state, None))
                next_state = env.getState()

                assert env.getBufferSizeFromState(next_state) == env.getBufferSizeFromState(state) - 1, \
                    "The buffer size after a DEATH decreased by 1: S(t) = {}, S(t+1) = {}" \
                        .format(env.getBufferSizeFromState(state),
                                env.getBufferSizeFromState(next_state))

                # We set the action to None because there is no action by the agent when a service is completed
                action = None
                # No reward when no action taken
                reward = np.nan

            return t, state, action, next_state, reward

        mixing = lambda s0, a0, s1, a1: s0 == s1 and a0 is not None and a1 is not None and a0 == a1
        # ------------------------------ Auxiliary functions ----------------------------------#

        # Create the two queue environments on which the simulation takes place
        envs = [copy.deepcopy(self.env), copy.deepcopy(self.env)]

        # Set the state of each environment AFTER taking the respective action at time t=0.
        # This is buffer_size when the first action is Reject and buffer_size+1 when the first action is Accept.
        if self.getLearnerP() is not None:
            assert self.agent.getAcceptancePolicy().getThetaParameter() == self.getLearnerP().getPolicy().getThetaParameter()
        K = self.agent.getAcceptancePolicy().getBufferSizeForDeterministicBlocking()

        # History of states, actions and rewards of each environment (2 x <undef> list)
        # Used for informational purposes, but actually we don't need to store the history because we are computing
        # Q0 and Q1 on the fly.
        times = [[0],
                 [0]]
        states = [[(buffer_size, None)],
                  [(buffer_size, None)]]
        actions = [[Actions.REJECT],
                   [Actions.ACCEPT]]
        rewards = [[compute_reward_for_buffer_size(self.env, buffer_size)],
                   [0.0]]

        # Set the NEW state of the environments once the above actions have been taken on the respective states
        envs[0].setState((buffer_size, None))       # Environment that starts at REJECTION
        envs[1].setState((buffer_size+1, None))     # Environment that starts at ACCEPTANCE

        # Time step in the queue trajectory (the first time step is t = 0)
        done = False
        t = 0                   # We start at 0 because we have already stored the state for t=0 above
        t0 = 0; t1 = 0;         # We keep track of the time step of each queue which only changes
                                # when there is an incoming job, and thus we can compare their state at the same
                                # discrete time value.
        maxtime = t_sim_max     # Max discrete time, in case there is no mixing of trajectories
        # Initialize the state-action values with the first observed reward (at t=0)
        Q0 = rewards[0][0]
        Q1 = rewards[1][0]
        if False:
            print("--> First reward Q(s={},a=1)={:.1f}, Q(s={},a=0)={:.1f}".format(buffer_size, Q1, buffer_size, Q0))
        while not done:
            # Generate events for each queue environment until the discrete time changes,
            # so that we can compare their state-action at the same discrete time value and find the mixing time.
            # Note: the generate_events() function takes care of NOT generating a service event when a server is empty.
            while t0 == t:
                time0, event0, job_class_or_server0, _ = generate_event([envs[0]])
                t0, state0, action0, next_state0, reward0 = apply_event(envs[0], t, event0, job_class_or_server0)
                if False:
                    print("\tChanging state of queue 0... t0={}, state={}, event={}, action={}, next_state={}, reward={:.1f}" \
                      .format(t0, state0, event0, action0, next_state0, reward0))
            if False:
                print("")
            while t1 == t:
                time1, event1, job_class_or_server1, _ = generate_event([envs[1]])
                t1, state1, action1, next_state1, reward1 = apply_event(envs[1], t, event1, job_class_or_server1)
                if False:
                    print("\tChanging state of queue 1... t1={}, state={}, event={}, action={}, next_state={}, reward={:.1f}" \
                      .format(t1, state1, event1, action1, next_state1, reward1))
            assert t0 == t1 and t0 == t + 1, \
                "The discrete time step is the same in both queues and has increased by one w.r.t. the global time step t (t={:}, t0={:}, t1={:})" \
                .format(t, t0, t1)
            t = t0

            # Update the history
            times[0] += [time0]; times[1] += [time1]
            states[0] += [next_state0]; states[1] += [next_state1]
            actions[0] += [action0]; actions[1] += [action1]
            rewards[0] += [reward0]; rewards[1] += [reward1]

            # Update the state-action values
            assert not np.isnan(reward0) and not np.isnan(reward1), "The reward given by each queue are not NaN (r0={}, r1={})".format(reward0, reward1)
            Q0 += reward0
            Q1 += reward1
            if False:
                print("\t\tEvent1 = {} (action1={}, state1={}), Event0 = {} (action0={}, state0={}): Updated rewards:\n"
                    "\t\tQ(s={},a=1)={:.1f}\n" \
                    "\t\tQ(s={},a=0)={:.1f}" \
                    .format(event1, action1, next_state1, event0, action0, next_state0, buffer_size, Q1, buffer_size, Q0))

            # Check if the trajectories meet at the same state-action
            if t > maxtime or mixing(next_state0, action0, next_state1, action1):
                # Either we reached the maximum simulation time allowed or mixing occurs
                # => Stop the simulation
                if False:
                    print("*** TRAJECTORIES MIXED! (Q1={:.1f}, Q0={:.1f}, Qdiff = {:.1f}) ***".format(Q1, Q0, Q1 - Q0))
                done = True

            if self.debug:
                print("t={}: states = [{}, {}] events={} actions={} -> states={} rewards={}" \
                      .format(t, states[-2][0], states[-2][1], [event0, event1], [action0, action1], [next_state0, next_state1], [reward0, reward1]), end="\n")

        # DONE
        if show_messages(verbose, verbose_period, t_learn):
            print("==> simulation ENDS at discrete time t={} at states = {} coming from states = {}, actions = {} rewards = {})" \
                    .format(t, [next_state0, next_state1], [states[-2][0], states[-2][1]], [action0, action1], [reward0, reward1]))

#        if show_messages(verbose, verbose_period, t_learn):
#           plt.figure()
#           print("Mixing time t={}, Q0={}, Q1={}".format(t, Q0, Q1))
#           print(np.c_[[envs[0].getBufferSizeFromState(s) for s in states[0]], actions[0], rewards[0],
#                        [envs[1].getBufferSizeFromState(s) for s in states[0]], actions[1], rewards[1]])
#           plt.plot([envs[0].getBufferSizeFromState(s) for s in states[0]], 'b.-')
#           plt.plot([envs[1].getBufferSizeFromState(s) for s in states[1]], 'r.-')
#           plt.show()
#           input("Press ENTER...")

        if not mixing(state0, action0, state1, action1):
            # The trajectories didn't mix
            # => Discard the estimations of Q0 and Q1 because they would be misleading as the trajectories didn't mix
            Q0 = Q1 = np.nan
            t_mix = np.nan
        else:
            t_mix = t

        return t, Q0, Q1, t_mix

    @measure_exec_time
    def learn(self, agent, t, probas_stationary: dict=None, Q_values: dict=None, simul_info: dict=None):
        """
        Learns the value functions and the policy at time t

        Arguments:
        agent: Agent
            Agent that is responsible of performing the actions on the environment and learning from.

        t: int
            Time (iteration) in the queue simulation time scale at which learning takes place.
            In Monte-Carlo learning, this time would be the end of the queue simulation.

        probas_stationary: (opt) dict of floats between 0 and 1 (used when estimating the theoretical grad(V))
            Dictionary with the estimated stationary probability for each buffer size defined in its keys.
            The dictionary keys are expected to be K-1 and K.
            default: None

        Q_values: (opt) dict of lists or numpy arrays (used when estimating the theoretical grad(V))
            Dictionary with the estimated state-action values for each buffer size defined in its keys.
            Each element in the list is indexed by the initial action taken when estimating the corresponding Q value.
            The dictionary keys are expected to be K-1 and K.
            default: None

        simul_info: (opt) dict
            Dictionary containing relevant information (for keeping a record of) about the simulations run to achieve
            the estimated stationary probabilities and the Q values.
            When given, the dictionary is expected to contain the following keys:
            - 'n_events_mc': # events in the MC simulation that estimates the stationary probabilities or that estimates
            P(T>t) and E(T_A) in the FV approach.
            - 'n_events_fv': # events in the FV simulation that estimates Phi(t) for the buffer sizes of interest.
            - 'n_Q': average number of # trajectory pairs used to estimate all the Q values given in the Q_values dict.
            In general, this is an average because we may need to run different trajectory pairs, for instance when
            using IGA, where we need to estimate Q_diff(K-1) and Q_diff(K), i.e. differences of Q between the two
            possible actions starting at two different buffer sizes.
            default: None
        """
        # TODO: (2021/11/03) Think if we can separate the learning of the value function and return G(t) from the learning of the policy (theta)
        # Goal: avoid computing G(t) twice, one at learnerV.learn() and one at learnerP.learn()

        assert self.dict_learning_params['mode'] in LearningMode

        # Store the alpha used for the current learning step
        agent.getLearnerP().store_learning_rate()

        if self.dict_learning_params['mode'] == LearningMode.REINFORCE_RETURN:
            agent.getLearnerV().learn(t)  # UNCOMMENTED ON SAT, 27-NOV-2021
            theta_prev, theta, V, gradV, G = agent.getLearnerP().learn(t)
            #theta_prev, theta, V, gradV, G = agent.getLearnerP().learn_TR(t)
        else:
            # When learning via the theoretical grad(V) expression we don't need to estimate the average reward!
            # (because grad(V) is the difference between two state-action values, and the average reward affects both the same way)
            # This is the option to use when learning via Fleming-Viot because in that case the return G(t)
            # --used when learning with learn() or learn_TR() or learn_linear_theoretical()-- is NOT meaningful
            # because, either it is empty as the rewards are not recorded while running the Fleming-Viot process,
            # or it does not reflect the actual values of each state-value of the original system.
            assert probas_stationary is not None and Q_values is not None
            if self.getLearnerP() is not None:
                assert self.agent.getAcceptancePolicy().getThetaParameter() == self.getLearnerP().getPolicy().getThetaParameter()
            K = agent.getAcceptancePolicy().getBufferSizeForDeterministicBlocking()
            if self.dict_learning_params['mode'] == LearningMode.REINFORCE_TRUE:
                # Regular gradient ascent based on grad(V)
                # Only information about the buffer size K-1 is used to learn,
                # as long as the gradient of the parameterized policy is not 0
                # (which happens when the theta parameter of the policy is not integer.
                assert K-1 in probas_stationary.keys() and K-1 in Q_values.keys(), \
                    "Key K-1={} is present in the dictionaries containing the stationary probability estimation ({}) and the Q values ({})" \
                        .format(K-1, probas_stationary, Q_values)
                theta_prev, theta, V, gradV, G = agent.getLearnerP().learn_linear_theoretical_from_estimated_values(t, probas_stationary[K-1], Q_values[K-1])
            elif self.dict_learning_params['mode'] == LearningMode.IGA:
                # IGA: Integer Gradient Ascent (only the signs of the Q differences are of interest here)
                keys_probas = probas_stationary.keys()
                keys_Q_values = Q_values.keys()
                assert [K-1, K] == sorted(keys_probas) and [K-1, K] == sorted(keys_Q_values), \
                    "Keys K-1={} and K={} are present in the dictionaries containing the stationary probability estimations ({}) and the Q values ({})" \
                        .format(K-1, K, probas_stationary, Q_values)
                theta_prev, theta, V, gradV, G = agent.getLearnerP().learn_linear_theoretical_from_estimated_values_IGA(t, probas_stationary, Q_values)

        if simul_info is not None:
            t_learn = simul_info.get('t_learn', 0)
            K = simul_info.get('K', 0)
            J_factor = simul_info.get('J_factor', 0.0)
            J = simul_info.get('J', 0)
            exponent = simul_info.get('exponent', 0.0)  # Exponent that defines N and T w.r.t. a reference N0 and T0 as N0*exp(exponent) in order to consider different N and T values of interest (e.g. exponent = -2, -1, 0, 1, 2, etc.)
            N = simul_info.get('N', 0)              # number of particles used in the FV process (this is 1 for MC)
            T = simul_info.get('T', 0)              # simulation time multiplier of 1/lambda (nmeantimes) (this is N_fv*T in MC)
            err_phi = simul_info.get('err_phi', 0)  # expected relative error in the estimation of Phi(t,K) when using N particles
            err_et = simul_info.get('err_et', 0)    # expected relative error in the estimation of E(T_A) when using T time steps
            seed = simul_info.get('seed')
            expected_cycle_time = simul_info.get('expected_cycle_time', 0.0)
            n_cycles = simul_info.get('n_cycles', 0)
            max_survival_time = simul_info.get('max_survival_time', 0.0)
            n_events_mc = simul_info.get('n_events_mc', 0)
            n_events_fv = simul_info.get('n_events_fv', 0)
            n_trajectories_Q = simul_info.get('n_Q', 0.0)
        else:
            t_learn = 0
            K = 0
            J_factor = 0.0
            J = 0
            exponent = 0.0
            N = 0
            T = 0
            err_phi = 0.0
            err_et = 0.0
            seed = None
            expected_cycle_time = 0.0
            n_cycles = 0
            max_survival_time = 0.0
            n_events_mc = 0
            n_events_fv = 0
            n_trajectories_Q = 0.0

        self.alphas += [agent.getLearnerP().getLearningRate()]
        self.thetas += [theta_prev]
        self.thetas_updated += [theta]
        self.proba_stationary += [[probas_stationary[K-1], probas_stationary.get(K)]]
        self.Q_diff += [[Q_values[K-1][1] - Q_values[K-1][0], Q_values[K][1] - Q_values[K][0]]]
        self.V += [V]               # NOTE: This is NOT always the average value, rho... for instance, NOT when linear_theoretical_from_estimated_values() is called above
        self.gradV += [gradV]
        self.G = G                  # Note that G is Q_diff when we estimate the gradient from its theoretical expression, using the estimates of the stationary probabilities and the differences of Q
        self.n_events_mc += [n_events_mc]
        self.n_events_fv += [n_events_fv]
        self.n_trajectories_Q += [n_trajectories_Q]

        agent.getLearnerP().update_learning_time()
        #print("*** Learning-time updated: time = {} ***".format(agent.getLearnerP().getLearningTime()))
        agent.getLearnerP().update_learning_rate_by_episode()

        if self.save:
            self.fh_results.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n" \
                                    .format(self.case,
                                            t_learn,
                                            self.envs[0].getParamsRewardFunc().get('buffer_size_ref', None)-1,  # -1 because buffer_size_ref is K and the theta parameter, which is the one we want to store here is K-1
                                            self.thetas[-1],
                                            self.thetas_updated[-1],
                                            K,
                                            J_factor,
                                            J,
                                            exponent,
                                            N,
                                            T,
                                            err_phi,
                                            err_et,
                                            seed,
                                            expected_cycle_time,
                                            n_cycles,
                                            max_survival_time,
                                            self.proba_stationary[-1][0],
                                            self.proba_stationary[-1][1],
                                            self.Q_diff[-1][0],
                                            self.Q_diff[-1][1],
                                            self.alphas[-1],
                                            self.V[-1],
                                            self.gradV[-1],
                                            self.n_events_mc[-1],
                                            self.n_events_fv[-1],
                                            self.n_trajectories_Q[-1]
                                            ))

    # ------ GETTERS ------#
    def getJobRatesByServer(self):
        return self.job_rates_by_server

    def getLearnerV(self):
        return self.learnerV

    def getLearnerP(self):
        return self.learnerP

    def getNumLearningSteps(self):
        return self.dict_learning_params['t_learn']

    # ------ SETTERS ------#
    def setNumberParticlesAndCreateEnvironments(self, N):
        """
        Sets the number of particles of the simulator and creates the environments that are needed
        to run the simulation on them
        """
        self.N = N
        # Note: it's important that the first environment stored in self.envs be self.env (and NOT a copy)
        # because in the process we sometimes refer to it as self.envs[0] (for instance when resetting the environments)
        # or as self.env (for instance when estimating Q(K-1, a) via Monte-Carlo on one particle (both in MC and FV modes)
        if hasattr(self, 'envs'):
            del self.envs
        self.envs = [self.env if i == 0 else copy.deepcopy(self.env) for i in range(self.N)]

    def setNumLearningSteps(self, t_learn):
        self.dict_learning_params['t_learn'] = t_learn


#-------------------------------------------- FUNCTIONS --------------------------------------#
def compute_nparticles_and_nsteps_for_fv_process(rhos: list, capacity: int, buffer_size_activation_factor: float=1/3, error_rel_phi=0.50, error_rel_et=0.50):
    """
    Computes the minimum number of particles and number of discrete steps to use in the FV process
    for maximum relative errors in the estimation of Phi(t, K) and of E(T_A) where K is the capacity of the queue.

    Arguments:
    rhos: list
        List of the server intensities: lambda / mu for each server in the system, where lambda is the job
        arrival rate and mu is the service rate.

    capacity: int
        Capacity of the system: maximum size of the buffer placed at the entrance of the system.

    buffer_size_activation_factor: (opt) float
        Buffer size activation factor J/K based on which the activation set of states is defined as
        J = int( round( factor*K ) ), such as J-1 defines the absorption set of states.
        default: 1/3, which is the optimum in the sense that the algorithm complexity to estimate Phi(t, K) and E(T_A)
        in the Fleming-Viot estimation is the same.

    error_rel_phi: (opt) float
        Maximum relative error for the estimation of Phi(t, K).
        This is the relative error incurred in the estimation of a binomial probability based on N trials,
        which is equal to sqrt((1-p)/p) / sqrt(N)
        default: 0.50

    error_rel_et: (opt) float
        Maximum relative error for the estimation of the expected absorption cycle time E(T_A).
        The calculation of the number of cycles needed assumes that the standard deviation of the estimation of E(T_A)
        is proportional to E(T_A), which gives a relative error equal to 1/sqrt(M) where M is the number of
        reabsorption cycles.
        (Note that from simulations of the FV estimation for different J/K values, it was found that
        the standard deviation is proportional to E(T_A)^1.6, more precisely Std(T_A) = 0.01 * E(T_A)^1.6)
        The number of discrete steps is then computed as M/p, where p is the blocking probability of the given
        queue system.
        default: 0.50

    Return: Tuple
    Duple with the following elements:
    - N: number of particles needed for the given relative error for Phi(t, K), error_rel_phi.
    - T: Number of discrete time steps for the given relative error for E(T_A), error_rel_et.
    """
    K = capacity
    J = int( np.round(buffer_size_activation_factor * K) )

    # -- Blocking probability
    # This is the equivalent of a queue that has the blocking probability observed under the Fleming-Viot process
    capacity_effective = int( np.ceil((K-J)/2) )
    pK = compute_blocking_probability_birth_death_process(rhos, capacity_effective)

    # Expected return time to J (under stationarity), which is what mostly defines the magnitude of E(T_A) = E(T1+T2)
    pJ = compute_blocking_probability_birth_death_process(rhos, J)

    N = int(np.ceil( (1 - pK) / pK / error_rel_phi**2 ))
    #M = int(np.ceil( 1 / error_rel_et**2 ))
    T = int(np.ceil( 1 / error_rel_et**2 / pJ ))

    return N, T


def compute_rel_errors_for_fv_process(rhos: list, capacity: int, N: int, T: int, buffer_size_activation_factor: float=1/3):
    """
    Computes the minimum number of particles and number of discrete steps to use in the FV process
    for maximum relative errors in the estimation of Phi(t, K) and of E(T_A) where K is the capacity of the queue.

    Arguments:
    rhos: list
        List of the server intensities: lambda / mu for each server in the system, where lambda is the job
        arrival rate and mu is the service rate.

    capacity: int
        Capacity of the system: maximum size of the buffer placed at the entrance of the system.

    N: int
        Number of particles used in the FV process.

    T: int
        Number of discrete time steps used in the estimation of P(T>t) and E(T_A).

    buffer_size_activation_factor: (opt) float
        Buffer size activation factor J/K based on which the activation set of states is defined as
        J = int( round( factor*K ) ), such as J-1 defines the absorption set of states.
        default: 1/3, which is the optimum in the sense that the algorithm complexity to estimate Phi(t, K) and E(T_A)
        in the Fleming-Viot estimation is the same.

    Return: tuple
    Duple with the following elements:
    - error_rel_phi: relative error expected for the estimation of Phi(t, K) given N.
        This is the relative error incurred in the estimation of a binomial probability based on N trials,
        which is equal to sqrt((1-p)/p) / sqrt(N)
    - error_rel_et: relative error expected for the estimation of the expected absorption cycle time E(T_A).
        This calculation assumes that the standard deviation of the estimation of E(T_A)
        is proportional to E(T_A), which gives a relative error equal to 1/sqrt(M) where M is the number of
        reabsorption cycles.
        (Note that from simulations of the FV estimation for different J/K values, it was found that
        the standard deviation is proportional to E(T_A)^1.6, more precisely Std(T_A) = 0.01 * E(T_A)^1.6)
        The number of discrete steps is then computed as M/p, where p is the blocking probability of the given
        queue system.
    """
    K = capacity
    J = int( np.round(buffer_size_activation_factor * K) )

    # -- Blocking probability
    # This is the equivalent of a queue that has the blocking probability observed under the Fleming-Viot process
    capacity_effective = int( np.ceil((K-J)/2) )
    pK = compute_blocking_probability_birth_death_process(rhos, capacity_effective)

    # Expected return time to J (under stationarity), which is what mostly defines the magnitude of E(T_A) = E(T1+T2)
    pJ = compute_blocking_probability_birth_death_process(rhos, J)

    error_rel_phi = np.sqrt( (1 - pK) / pK / N )
    error_rel_et  = np.sqrt( 1 / T / pJ )

    return error_rel_phi, error_rel_et


def compute_reward_for_buffer_size(env, bs):
    """
    Computes the reward given by the environment when blocking at the given buffer size bs

    If the environment does not have a reward function defined, the returned reward is 1.0, so that the expected reward
    is equal to the probability of being at the given buffer size.
    """
    # *** We assume that the state of the environment is a duple: (list with server queue sizes, job class of latest arriving job) ***
    state = ([0] * env.getNumServers(), None)
    # We set the first server to have occupancy equal to the buffer size of interest
    # as we assume that the reward function ONLY depends on the buffer size, and NOT on the actual state
    # of the multi-server system.
    state[0][0] = bs
    reward_func = env.getRewardFunction()
    if reward_func is not None:
        params_reward_func = env.getParamsRewardFunc()
        reward = reward_func(env, state, Actions.REJECT, state, dict_params=params_reward_func)
    else:
        reward = 1.0

    return reward


def choose_state_for_buffer_size(env, buffer_size):
    """
    Chooses randomly a server in the queue environment to have its queue size equal to the given buffer size,
    while the queue size of all other servers are set to 0, so that the buffer size of the system is equal
    to the given buffer size.

    Arguments:
    env: Queue environment
        Queue environment containing the servers.

    buffer_size: int
        Buffer size to be set in the system.

    Return: list
    List containing the state to set the queue environment at that guarantees that the system's buffer size is
    equal to the given buffer size.
    """
    state = [0] * env.getNumServers()
    state[np.random.randint(0, len(state))] = buffer_size
    return state


def get_blocking_buffer_sizes(agent):
    assert agent is not None
    if agent.getLearnerP() is not None:
        assert agent.getAcceptancePolicy().getThetaParameter() == agent.getLearnerP().getPolicy().getThetaParameter(), \
                "The theta parameter of the agent's acceptance policy ({}) coincides with the theta parameter" \
                .format(agent.getAcceptancePolicy().getThetaParameter()) + \
                " of the policy learner ({})" \
                .format(agent.getLearnerP().getPolicy().getThetaParameter())
    K = agent.getAcceptancePolicy().getBufferSizeForDeterministicBlocking()
    blocking_buffer_sizes = [k for k in range(0, K+1)
                             if agent.getAcceptancePolicy().getPolicyForAction(Actions.REJECT, None, buffer_size=k) > 0]

    return blocking_buffer_sizes


def compute_proba_blocking(agent, probas_stationary):
    """
    Computes the blocking probability of the queue system

    The blocking probability is computed as:
    Pr(BLOCK) = sum_{buffer sizes bs with rejection probability Pi > 0} { Pi(reject/bs) * Pr(bs) }

    where Pr(bs) is the estimated stationary probability for buffer size bs.

    Arguments:
    agent: Agent
        Agent where the accept/reject policy is defined.

    probas_stationary: dict
        Dictionary with the estimated stationary probability of the buffer sizes of interest which are
        the dictionary keys.

    Return: float
    Estimated blocking probability.
    """
    proba_blocking = 0.0
    buffer_sizes_of_interest = get_blocking_buffer_sizes(agent)
    assert len(probas_stationary) == len(buffer_sizes_of_interest), \
        "The length of the probas_stationary dictionary ({}) is the same as the number of buffer sizes of interest ({})" \
            .format(probas_stationary, buffer_sizes_of_interest)
    for bs in buffer_sizes_of_interest:
        if False:
            print("---> Computing blocking probability...")
            print("buffer size = {}, Pr(Reject) = {:.3f}%, Pr({}) = {:.3f}%" \
                  .format(bs, agent.getAcceptancePolicy().getPolicyForAction(Actions.REJECT, None, bs)*100, bs, probas_stationary[bs]*100))
        if agent.getLearnerP() is not None:
            assert agent.getAcceptancePolicy().getThetaParameter() == agent.getLearnerP().getPolicy().getThetaParameter()
        proba_blocking += agent.getAcceptancePolicy().getPolicyForAction(Actions.REJECT, None, bs) * \
                          probas_stationary[bs]
    return proba_blocking


def estimate_expected_reward(env, agent, probas_stationary):
    """
    Estimates the expected reward of the queue system defined in the given environment assuming a non-zero reward
    happens at the buffer sizes of interest, i.e. those where the policy yields a positive probability of rejection.

    The expected reward is estimated as:
    E[R] = sum_{buffer sizes bs with rejection probability Pi > 0} { R(bs) * Pi(reject/bs) * Pr(bs) }

    where R(bs) is reward associated to a rejection at buffer size bs defined in the queue environment,
    and Pr(bs) is the stationary probability estimate of buffer size bs.

    Arguments:
    env: Queue environment
        Queue environment where the reward function of the blocking buffer size is defined.

    agent: Agent
        Agent where the accept/reject policy is defined.

    probas_stationary: dict
        Dictionary with the estimated stationary probability of the buffer sizes of interest which are
        the dictionary keys.

    Return: float
    Estimated expected reward.
    """
    expected_reward = 0.0
    buffer_sizes_of_interest = get_blocking_buffer_sizes(agent)
    assert len(probas_stationary) == len(buffer_sizes_of_interest)
    for bs in buffer_sizes_of_interest:
        if agent.getLearnerP() is not None:
            assert agent.getAcceptancePolicy().getThetaParameter() == agent.getLearnerP().getPolicy().getThetaParameter()
        expected_reward += compute_reward_for_buffer_size(env, bs) * \
                           agent.getAcceptancePolicy().getPolicyForAction(Actions.REJECT, None, bs) * \
                           probas_stationary[bs]
    return expected_reward


def estimate_expected_cycle_time(n_cycles, time_end_last_cycle, time_end_simulation):
    "Estimates the expected return time to the initial buffer size"
    if n_cycles > 0:
        assert time_end_last_cycle > 0.0
        expected_cycle_time = time_end_last_cycle / n_cycles
    else:
        warnings.warn(
            "No absorption cycle was observed, the expected absorption time is estimated as the maximum time observed in the simulation: {}".format(
                time_end_simulation))
        expected_cycle_time = time_end_simulation
    assert expected_cycle_time > 0.0

    return expected_cycle_time


def generate_event(envs):
    """
    Generates the next event happening on the given system of queue environments, which are assumed to have all
    the same characteristics in terms of number of job classes that can arrive and number of servers that can serve.

    Note that the method takes into account the server's queue sizes, meaning that no completed service can occur
    on an empty server in the system.

    The next event can be either:
    - an incoming job (of a given class);
    - a completed service in a server in the queue system having at least one job.

    Arguments:
    envs: list
        List of queue environments to consider for the generation of the event, i.e. where the event COULD be
        generated.

    Return: tuple
    Tuple containing the following elements:
    - the time to the event (relative to whatever absolute time the queue system is at)
    - the event type (Event.BIRTH or Event.DEATH)
    - an index that represents either:
        - the job class, if the generated event is an arriving job
        - the server, if the generated event is a completed service
    - the environment/particle where the event takes place
    """
    # Define a 2D numpy array where the different environments/particles are in the first dimension
    # and the job classes and servers are in the second dimension. The value of each entry is
    # either the job class arrival rate or the server service rate, depending on the column position
    # (first come the job classes and then the servers).
    # A service rate is NaN when the server size is 0, so that the server cannot be selected to experience
    # a service rate.
    valid_rates = np.array(
                    [env.getJobClassRates() + [r if s > 0 else np.nan for r, s in zip(env.getServiceRates(), env.getQueueState())]
                     for env in envs])
    #print("\nQueue states:\n{}".format([[s for s in env.getQueueState()] for env in envs]))
    #print("Valid rates: {}".format(valid_rates))

    # Generate the event time and the index on the rates array to which it is associated
    event_time, event_idx = generate_min_exponential_time(valid_rates.flatten())

    # Convert the linear index back to the 2D index in valid_rates where the indices represent:
    # (selected env/particle, selected rate)
    idx_env, job_class_or_server = index_linear2multi(event_idx, valid_rates.shape)

    # Define whether the generated event is an incoming job class or a completed service
    # and return a different tuple accordingly.
    # It is assumed that the number of job classes and servers are the same in each environment.
    n_job_classes = envs[0].getNumJobClasses()
    if job_class_or_server < n_job_classes:
        # The event is an incoming job class
        job_class = job_class_or_server
        return event_time, Event.BIRTH, job_class, idx_env
    else:
        # The event is a completed service event
        idx_server = job_class_or_server - n_job_classes
        return event_time, Event.DEATH, idx_server, idx_env


def manage_job_arrival(t, env, agent, state, job_class):
    """
    Manages an incoming job event, whether it is accepted or not and what is done if it is accepted

    Arguments:
    t: int
        Discrete time step at which the job arrival takes place.

    env: Queue environment
        Queue environment where the service takes place.

    agent: Agent
        Agent acting on the environment.

    state: State of the queue environment
        The state in which the queue environment is in.

    job_class: int
        Class of the arriving job.

    Return: tuple
    Triple with the following elements:
    - action taken by the agent
    - the next state of the queue
    - the reward received
    - gradient: the policy gradient for the action taken given the state
    """
    # Store the class of the arriving job
    env.setJobClass(job_class)

    # Perform a step on the environment by applying the acceptance policy
    action_accept_reject, next_state, reward_accept_reject, _ = step(t, env, agent, PolicyTypes.ACCEPT)

    # Assign the job to a server if accepted
    if action_accept_reject == Actions.ACCEPT:
        # Assign the job just accepted to a server and update the next_state value
        # (because the next_state from the ACCEPT action has not been updated above
        # as we need to know to which server the accepted job is assigned in order
        # to know the next state of the queue environment)
        _, next_state, reward_assign, _ = step(t, env, agent, PolicyTypes.ASSIGN)
    else:
        reward_assign = 0.0
    action = action_accept_reject
    reward = reward_accept_reject + reward_assign

    if action == Actions.ACCEPT and reward != 0.0:
        print("--> action = {}, REWARD ASSIGN: {}, REWARD = {}".format(action_accept_reject, reward_assign, reward))
        raise ValueError("A non-zero reward is not possible when the action is ACCEPT.")

    # Update the average reward (just for information purposes)
    if agent.getLearnerV() is not None:
        agent.getLearnerV().updateAverageReward(t, reward)

    # Compute and store the gradient of the policy evaluated on the taken action A(t) given the state S(t)
    # Goal: depending on the learning method, this value may be useful
    if agent.getLearnerP() is not None:
        gradient_for_action = agent.getLearnerP().getPolicy().getGradient(action_accept_reject, state)
        agent.getLearnerP().record_gradient(state, action_accept_reject, gradient_for_action)
    else:
        gradient_for_action = 0.0

    return action, next_state, reward, gradient_for_action


def manage_service(env, agent, state, server):
    """
    Manages a completed service event

    Arguments:
    env: Queue environment
        Queue environment where the service takes place.

    agent: Agent
        Agent acting on the environment.
        This parameter is not used but it is left for consistency between manage_job_arrival() method.

    state: State of the queue environment
        The state in which the queue environment is in (it's only used when storing the trajectory and for assertions)

    server: int
        Server index where the service takes place.

    Return: tuple
    Triple with the following elements:
    - action taken: None (because no action is taken when a service is completed)
    - the next state of the queue
    - the reward received: NaN, as no reward is received when a service is completed.
    """
    queue_state = env.getQueueState()
    assert queue_state[server] > 0, "The server where the completed service occurs has at least one job"

    # Compute and set the next state of the affected server in the queue system
    next_queue_state = copy.deepcopy(queue_state)
    next_queue_state[server] -= 1

    env.setState((next_queue_state, None))
    next_state = env.getState()
    assert env.getBufferSizeFromState(next_state) == env.getBufferSizeFromState(state) - 1, \
        "The buffer size after a DEATH decreased by 1: S(t) = {}, S(t+1) = {}" \
            .format(env.getBufferSizeFromState(state), env.getBufferSizeFromState(next_state))

    # We set the action to None because there is no action by the agent when a service is completed
    action = None
    # We set the reward to NaN because a reward may happen ONLY when a new job arrives.
    # If there are rewards due to serviced jobs, we would impact those rewards into the reward
    # received when a new job arrives (which is not the case at this moment (28-Nov-2021),
    # i.e. the reward when a job is serviced is 0).
    reward = np.nan

    return action, next_state, reward


def estimate_blocking_mc(env, agent, dict_params_simul, dict_params_info, start_state=None):
    """
    Estimates the blocking probability using Monte-Carlo

    Arguments:
    env: Queue environment
        Queue environment where the simulation runs.

    agent: Agent
        Agent that interacts with the environment.

    simul: SimulatorQueue
        Simulation object used for the simulation that allows estimating the blocking probability.

    dict_params_simul: dict
        Dictionary containing the simulation parameters.

    dict_params_info: dict
        Dictionary containing information to display or parameters to deal with the information to display.

    start_state: (opt) int or list or numpy array
        State at which the queue environment starts for the simulation.
        Its type depends on how the queue environment defines the state of the system.
        default: None, in which case the start state is defined as state whose buffer size coincides with
        dict_params_simul['buffer_size_activation'] - 1.

    Return: tuple
    Tuple with the following elements:
    - proba_blocking: the estimated blocking probability.
    - expected_reward: the estimated expected reward.
    - probas_stationary: dictionary with the estimated stationary probability for each buffer size where blocking
    may occur
    - expected_cycle_time: estimated expected cycle time E(T) of return to the initial buffer size.
    used in the denominator of the MC estimator of the blocking probability.
    - n_cycles: number of cycles observed to estimate E(T).
    - n_events: number of events observed during the simulation.
    """
    # Reset environment and learner of the value functions
    # IMPORTANT: We should NOT reset the learner of the policy because this function could be called as part of a
    # policy learning process! Note that the learner of the value function V MUST be reset regardless of this because
    # at this point a new queue simulation starts and at that point the value function should be learned anew.
    env.reset()
    if agent.getLearnerV() is not None:
        agent.getLearnerV().reset()

    # Smart selection of the start state as one having buffer size = K-1 so that we can better estimate Pr(K-1)
    # start_state = choose_state_for_buffer_size(simul.env, K)
    # Selection of the start state as one having buffer size = J-1, in order to have a fair comparison with the FV method
    # where particles are reactivated when they reach J-1.
    if start_state is None:
        start_state = choose_state_for_buffer_size(env, dict_params_simul['buffer_size_activation'] - 1)
    t, time_last_event, n_cycles, time_last_return = \
            run_simulation_mc(env, agent, dict_params_info.get('t_learn', 0), start_state, dict_params_simul['T'],
                              track_return_cycles=True,
                              seed=dict_params_simul['seed'],
                              verbose=dict_params_info.get('verbose', False), verbose_period=dict_params_info.get('verbose_period', 1))
    probas_stationary, time_step_last_return, last_time_at_start_buffer_size = \
                                                        estimate_stationary_probabilities_mc(env, agent, start_state)
    n_events = t
    assert n_events == dict_params_simul['T']

    proba_blocking = compute_proba_blocking(agent, probas_stationary)
    expected_reward = estimate_expected_reward(env, agent, probas_stationary)
    expected_cycle_time = estimate_expected_cycle_time(n_cycles, time_last_return, time_last_event)

    if DEBUG_ESTIMATORS or show_messages(dict_params_info.get('verbose', False), dict_params_info.get('verbose_period', 1), dict_params_info.get('t_learn', 0)):
        print("\n*** RESULTS OF MONTE-CARLO SIMULATION on {} events ***".format(n_events))
        print("P(K-1), P(K): {}, last return time to initial buffer size: time_step = {} ({:.1f}%), t = {:.1f}" \
              .format(probas_stationary, time_step_last_return, time_step_last_return / t * 100, last_time_at_start_buffer_size))

    return proba_blocking, expected_reward, probas_stationary, expected_cycle_time, n_cycles, n_events


@measure_exec_time
def run_simulation_mc(env, agent, t_learn, start_state, t_sim_max,
                      track_return_cycles=False,
                      track_absorptions=False, track_survival=False,
                      seed=None, verbose=False, verbose_period=1):
    """
    Runs the continuous-time simulation using Monte-Carlo

    Arguments:
    agent: Agent
        Agent object that is responsible of performing the actions on the environment and learning from them.

    t_learn: int
        The learning time step to which the simulation will contribute.

    start_state: int or list or numpy array
        State at which the queue environment starts for the simulation.
        Its type depends on how the queue environment defines the state of the system.

    t_sim_max: int
        Maximum simulation time steps allowed for the simulation.
        This is equivalent to the number of observed events, as any new event generates a new time step.

    track_return_cycles: (opt) bool
        Whether to track the return cycles to the initial buffer size.
        default: False

    track_absorptions: (opt) bool
        Whether to track the absorption times.
        default: False

    track_survival: (opt) bool
        Whether to track the survival times.
        When True, it only has an effect when track_absorptions = True as well.
        default: False

    seed: (opt) int
        Seed to use in the simulation process.
        default: None, in which case the simulation cannot be reproduced at a later stage

    verbose: (opt) bool
        Whether to be verbose in the simulation process.
        default: False

    verbose_period: (opt) int
        The time step period to be verbose.
        default: 1 => be verbose at every simulation step.

    Return: tuple
    If track_return_cycles = False, track_absorptions = False and track_survival = False:
    - t: the last time step of the simulation process.
    - time: the continuous time of the last observed event during the simulation.

    If track_return_cycles = True, in addition:
    - n_cycles: the number of cycles observed in the simulation, until the last return to the initial buffer size.

    If track_absorptions = True and track_survival = False, in addition:
    - n_cycles: the number of cycles observed in the simulation, until the last return to the initial buffer size
    from ABOVE (i.e. as if it were an absorption).
    - time_end_last_cycle: the continuous time at which the Markov Chain returns to the initial buffer size
    from ABOVE (i.e. as if it were an absorption).

    If track_absorptions = True and track_survival = True, in addition:
    - survival_times: list with the observed survival times, sorted increasingly.

    Note that:
    - either track_return_cycles or track_absorptions may be True. If both are True, the first one has precedence.
    - track_survival = True has no effect if track_absorptions = False.
    """
    # -- Parse input parameters
    tmax = t_sim_max

    # Seed
    if seed is not None:
        env.set_seed(seed)

    # Set the start state of the environment to the given start state
    job_class = None
    env.setState((start_state, job_class))
    buffer_size_start = env.getBufferSize()
    if verbose:
        print("MC simulation: The queue environments starts at state {} (buffer size = {})" \
              .format(env.getState(), buffer_size_start))

    # Store the initial position as part of the trajectory
    update_trajectory(agent, (t_learn - 1) * (tmax + 1) + 0, 0.0, env.getState(), None, np.nan)

    # Time step in the queue trajectory (the first time step is t = 0)
    done = False
    t = 0
    t_arrivals = 0  # Number of arrival events. Used to decide the end of the simulation, when track_absorptions = True
    time_abs = 0.0  # Measure of the ABSOLUTE time of the latest event
    # Information about the return cycles
    n_cycles_return = 0
    n_cycles_absorption = 0
    time_last_return = 0.0
    time_last_absorption = 0.0
    # Information about the survival times
    if track_survival:
        time_last_activation = None
        survival_times = [0.0]
        ## NOTE: We include the initial time in the list of survival times so that the survival probability
        ## associated to them corresponds to the segment at and AFTER the time value stored in survival_times.
        ## In addition, this setup makes the survival probability function go from 1.0 to 0.0.
    if DEBUG_ESTIMATORS:
        # Check realization of the arrival and service rates
        time_last_arrival = 0.0
        if buffer_size_start == 0:
            time_last_service = np.nan
        else:
            time_last_service = 0.0
        jobs_arrival = []
        times_inter_arrival = []
        servers_service = []
        times_service = []
    while not done:
        t += 1

        # Current state
        state = env.getState()

        # Generate next event
        # Note: this function takes care of NOT generating a service event when a server is empty.
        time, event, job_class_or_server, _ = generate_event([env])
        time_abs += time
        if False: #DEBUG_TRAJECTORIES:
            print("[MC] Time step t = {}: Exponential time generated: {:.3f} --> ABS time = {:.3f}".format(t, time, time_abs))

        # Analyze the event
        if event == Event.BIRTH:
            # The event is an incoming job class
            # => Update the state of the queue, apply the acceptance policy, and finally the server assignment policy
            t_arrivals += 1
            action, next_state, reward, gradient_for_action = manage_job_arrival(t, env, agent, state, job_class_or_server)

            # Check ACTIVATION: we touch a state having the start buffer size + 1 COMING FROM BELOW
            # (Note that the condition "coming from below" is important because we normally need to observe the
            # activation states under their steady-state activation distribution)
            buffer_size = env.getBufferSizeFromState(next_state)
            if track_survival and buffer_size == buffer_size_start + 1:
                time_last_activation = time_abs
                if DEBUG_ESTIMATORS:
                    print("[MC] --> ACTIVATION @time = {:.3f}".format(time_abs))

            if DEBUG_ESTIMATORS:
                jobs_arrival += [job_class_or_server]
                times_inter_arrival += [time_abs - time_last_arrival]
                # Prepare for the next iteration
                time_last_arrival = time_abs
                if env.getBufferSize() == 1:
                    # The queue can now experience a service
                    # => Set the last service time to the last event time, so that we can measure the next
                    # service time when it occurs.
                    time_last_service = time_abs

        elif event == Event.DEATH:
            # The event is a completed service
            # => Update the state of the queue
            action, next_state, reward = manage_service(env, agent, state, job_class_or_server)

            # Check ABSORPTION
            buffer_size = env.getBufferSizeFromState(next_state)
            if buffer_size == buffer_size_start:
                n_cycles_absorption += 1
                time_last_absorption = time_abs
                if track_survival and time_last_activation is not None:
                    time_to_absorption = time_abs - time_last_activation
                    if DEBUG_ESTIMATORS:
                        print("[MC] --> ABSORPTION @time = {:.3f}".format(time_abs))
                        print("[MC] \tACTIVATION time = {:.3f} --> time to absorption = {:.3f}" \
                              .format(time_last_activation, time_to_absorption))
                    assert time_to_absorption > 0
                    # We sort the survival times at the end of the process, as it might be faster than inserting
                    # the value in order at every time being observed (although the difference apparently is just
                    # in the constant of proportionality of the O(N*log(N)) complexity.
                    # Ref: https://stackoverflow.com/questions/168891/is-it-faster-to-sort-a-list-after-inserting-items-or-adding-them-to-a-sorted-lis
                    survival_times += [time_to_absorption]

                    # Reset the time of last activation to None so that we are now alert to setting it again
                    # the next time it occurs
                    time_last_activation = None

            if DEBUG_ESTIMATORS:
                assert not np.isnan(time_last_service)
                servers_service += [job_class_or_server]
                times_service += [time_abs - time_last_service]
                # Prepare for the next iteration
                if env.getBufferSize() == 0:
                    # The queue can now longer experience a service event
                    time_last_service = np.nan
                else:
                    time_last_service = time_abs

        # Update the trajectory used in the learning process, where we store:
        # S(t): state BEFORE an action is taken
        # A(t): action taken given the state S(t)
        # R(t): reward received by taking action A(t) and transition to S(t+1)
        update_trajectory(agent, (t_learn - 1) * (tmax + 1) + t, time_abs, next_state, action, reward)

        # Check RETURN: the initial buffer size is observed again.
        buffer_size = env.getBufferSizeFromState(next_state)
        if buffer_size == buffer_size_start:
            n_cycles_return += 1
            time_last_return = time_abs

        if DEBUG_TRAJECTORIES:
            print("[MC] {} | t={}: time={:.3f}, event={}, action={} -> state={}, reward={:.3f}" \
                  .format(state, t, time_abs, event, action, next_state, reward), end="\n")

        if track_absorptions:
            # Tracking absorptions means that we are running this simulation for the FV estimation of P(T>t) and E(T_A)
            # => We need to stop when a certain number of ARRIVALS have occurred
            # (because that's how the required simulation time has been determined in order to guarantee a maximum
            # relative error of the estimation of E(T_A) --see compute_nparticles_and_nsteps_for_fv_process()
            # in utils.computing)
            done = check_done(tmax, t_arrivals, state, action, reward)
        else:
            done = check_done(tmax, t, state, action, reward)

    # DONE
    if show_messages(verbose, verbose_period, t_learn):
        print("[MC] ==> agent ENDS at time t={} at state {} coming from state = {}, action = {}, reward = {}, gradient = {})" \
              .format(t, env.getState(), state, action, reward, gradient_for_action))

    if DEBUG_ESTIMATORS:
        job_rates_by_server = compute_job_rates_by_server(env.getJobClassRates(),
                                                          env.getNumServers(),
                                                          agent.getAssignmentPolicy().getProbabilisticMap())
        plotting.plot_event_times(job_rates_by_server, times_inter_arrival, jobs_arrival, class_name="Job class")
        plotting.plot_event_times(env.getServiceRates(), times_service, servers_service, class_name="Server")

    if track_return_cycles:
        return t, time_abs, n_cycles_return, time_last_return
    if track_absorptions:
        if track_survival:
            return t, time_abs, n_cycles_absorption, time_last_absorption, sorted(survival_times)
        else:
            return t, time_abs, n_cycles_absorption, time_last_absorption
    return t, time_abs


def estimate_stationary_probabilities_mc(env, agent, start_state):
    """
    Monte-Carlo estimation of the stationary probability at the buffer sizes of interest
    (defined by the agent's job acceptance policy) from the observed trajectory in continuous-time.

    The stationary probability is estimated as the fraction of time spent at each buffer size over the total
    simulation time (sum of sojourn times).

    Arguments:
    env: environment
        The queue environment where the agent acts.

    agent: Agent
        The agent interacting with the environment in terms of job acceptance/rejection.
        It should have a learner of the value function V and a learner of the policy P defined.

    start_state: int or list or numpy array
        State of the queue environment at which the Monte-Carlo simulation started.
        This is used to check that the initial state actually stored in the trajectory has a buffer size that
        is equal to the buffer size of this state, and that therefore we are doing the correct computation to
        estimate the stationary probability of the buffer sizes of interest (which is based on trajectory cycles
        that come back to the initial buffer size).

    Return: tuple
    Tuple with the following elements:
    - probas_stationary: Dictionary with the estimated stationary probability for the buffer sizes of interest,
    typically those where blocking may occur according to the agent's job acceptance policy.
    - last_t_to_initial_position: time step at which the system returned to the initial buffer size for the last time.
    - last_time_to_initial_position: continuous time at which the system returned to the initial buffer size
    for the last time.
    """
    # Event times
    times = agent.getLearnerV().getTimes()

    # Sojourn times
    # Each sojourn time is the time the Markov Chain sojourned at the state stored in `states`, retrieved below
    # IMPORTANT: This means that the first state in `states` is the state PRIOR to the first time stored in `times`.
    # IMPORTANT 2: We should convert the output of np.diff() to a list otherwise the value "summed" on the left
    # with the initial time value `times[0]` will be ADDED to all entries in the array returned by np.diff()!!
    sojourn_times = [times[0]] + list( np.diff(times) )

    # States and associated buffer sizes
    states = agent.getLearnerV().getStates()
    buffer_sizes = [env.getBufferSizeFromState(s) for s in states]
    buffer_size_start = env.getBufferSizeFromState(start_state)
    assert buffer_sizes[0] == buffer_size_start, \
            "The buffer size of the first state stored in the chain trajectory ({})" \
            " equals the buffer size of the state at which the simulation started ({})" \
            .format(buffer_sizes[0], buffer_size_start)
    idx_last_visit_to_initial_position = find_last(buffer_sizes, buffer_sizes[0])
    assert idx_last_visit_to_initial_position >= 0, \
        "The return time to the initial buffer size is always observed, as the first value is always the value searched for ({}):\n{}".format(buffer_sizes[0], np.c_[buffer_sizes, times])

    # Compute the stationary probabilities of the buffer sizes of interest by Find the last time the chain visited the starting position
    buffer_sizes_of_interest = get_blocking_buffer_sizes(agent)
    probas_stationary = dict()
    if idx_last_visit_to_initial_position == 0:
        # The system never returned to the initial buffer size
        # => We set the stationary probability to 0.0 (and NOT to NaN) in order to avoid an assertion error in
        # the policy learner dealing with stationary probabilities that require that they are between 0 and 1.
        # (e.g. see agents/learners/policies.py)
        warnings.warn("The Markov Chain never returned to the initial position/buffer-size ({})."
                      "\nThe estimated stationary probabilities will be set to 0.0.".format(buffer_sizes[0]))
        last_time_at_start_buffer_size = 0.0
        for bs in buffer_sizes_of_interest:
            probas_stationary[bs] = 0.0
    else:
        last_time_at_start_buffer_size = times[idx_last_visit_to_initial_position-1]
            ## we subtract -1 to the index because the times list contains the times at which the Markov Chain
            ## STOPS visiting the state in the `states` list.

        # Compute the total sojourn time at each buffer size of interest
        # IMPORTANT: we need to limit this computation to the moment we observe the last return time to the
        # initial buffer size! (o.w. we could be overestimating the probability and even estimate it larger than 1)
        for bs in buffer_sizes_of_interest:
            sojourn_times_at_bs = np.sum([st for k, st in zip(buffer_sizes[:idx_last_visit_to_initial_position], sojourn_times[:idx_last_visit_to_initial_position]) if k == bs])
            assert sojourn_times_at_bs <= last_time_at_start_buffer_size, \
                "The total sojourn time at the buffer size of interest (e.g. blocking) bs={} ({})" \
                " is at most equal to the last time of return to the buffer size ({})" \
                .format(bs, sojourn_times_at_bs, last_time_at_start_buffer_size)
            probas_stationary[bs] = sojourn_times_at_bs / last_time_at_start_buffer_size

    return probas_stationary, idx_last_visit_to_initial_position, last_time_at_start_buffer_size


def estimate_blocking_fv(envs, agent, dict_params_simul, dict_params_info):
    """
    Estimates the blocking probability using the Fleming-Viot approach

    Arguments:
    envs: List
        List of queue environments used to run the FV process.

    agent: Agent
        Agent interacting with each of the environments given in envs.

    dict_params_simul: dict
        Dictionary containing the simulation parameters.
        It should contain at least the following keys:
        - buffer_size_activation: the overall system's buffer size for activation (J, where J-1 is the absorption buffer size).
        - T:
        - seed: the seed to use for the pseudo-random number generation.

    dict_params_info: dict
        Dictionary containing information to display or parameters to deal with the information to display.
        Accepted keys are:
        - verbose: whether to be verbose during the simulation.
        - verbose_period: the number of iterations (of what?) at which to be verbose.
        - t_learn: the number of learning steps, when FV is used in the context of FVRL, i.e. to learn an optimum policy.

    Return: tuple
    Tuple with the following elements:
    - proba_blocking: the estimated blocking probability.
    - expected_reward: the estimated expected reward.
    - probas_stationary: dictionary with the estimated stationary probability for each buffer size where blocking
    may occur.
    - expected_absorption_time: estimated expected absorption time E(T_A) used in the denominator of the FV estimator
    of the blocking probability.
    - n_cycles: number of cycles observed to estimate E(T_A).
    - time_last_absorption: continuous time of the last absorption observed, used in the estimation of E(T_A).
    - time_end_simulation_et: continuous time of the end of the simulation run to estimate E(T_A) and P(T>t).
    - max_survival_time: maximum survival time observed when estimating P(T>t).
    - time_end_simulation_fv: continuous time at which the FV simulation on the N particles end.
    - n_events_et: number of events observed during the simulation of the single queue used to estimate E(T_A) and P(T>t).
    - n_events_fv: number of events observed during the FV simulation that estimates Phi(t).
    """
    # Reset environment and learner of the value functions
    # IMPORTANT: We should NOT reset the learner of the policy because this function could be called as part of a
    # policy learning process! Note that the learner of the value function V MUST be reset regardless of this because
    # at this point a new queue simulation starts and at that point the value function should be learned anew.
    for env in envs:
        env.reset()
    if agent.getLearnerV() is not None:
        agent.getLearnerV().reset()

    # -- Step 1: Simulate a single queue to estimate P(T>t) and E(T_A)
    start_state = choose_state_for_buffer_size(envs[0], dict_params_simul['buffer_size_activation'] - 1)
    t, time_end_simulation, n_cycles, time_end_last_cycle, survival_times = \
        run_simulation_mc(envs[0], agent, dict_params_info.get('t_learn', 0), start_state, dict_params_simul['T'],
                          track_absorptions=True, track_survival=True,
                          seed=dict_params_simul['seed'],
                          verbose=dict_params_info.get('verbose', False), verbose_period=dict_params_info.get('verbose_period', 1))
    n_events_et = t
    time_last_absorption = time_end_last_cycle
    time_end_simulation_et = time_end_simulation
    max_survival_time = survival_times[-1]

    # Estimate P(T>t) and E(T_A)
    df_proba_surv = compute_survival_probability(survival_times)
    expected_absorption_time = estimate_expected_cycle_time(n_cycles, time_end_last_cycle, time_end_simulation)

    if DEBUG_ESTIMATORS or show_messages(dict_params_info.get('verbose', False), dict_params_info.get('verbose_period', 1), dict_params_info.get('t_learn', 0)):
        print("\n*** RESULTS OF MC ESTIMATION OF P(T>t) and E(T) on {} events ***".format(n_events_et))
        max_rows = pd.get_option('display.max_rows')
        pd.set_option('display.max_rows', None)
        print("P(T>t):\n{}".format(df_proba_surv))
        pd.set_option('display.max_rows', max_rows)
        print("E(T) = {:.1f} ({} cycles, last absorption at {:.3f}), Max observed survival time = {:.1f}" \
              .format(expected_absorption_time, n_cycles, time_end_last_cycle, df_proba_surv['t'].iloc[-1]))

    if DEBUG_TRAJECTORIES:
        ax0 = plot_trajectory(envs[0], agent, dict_params_simul['buffer_size_activation'])
        xticks = range(int(ax0.get_xlim()[1]))
        ax0.set_xticks(xticks)
        ax0.vlines(xticks, 0, ax0.get_ylim()[1], color="lightgray")

    # -- Step 2: Simulate N particles with FLeming-Viot to compute the empirical distribution and estimate the expected reward
    # The empirical distribution Phi(t, bs) estimates the conditional probability of buffer sizes bs
    # for which the probability of rejection is > 0
    N = len(envs)
    assert N > 1, "The simulation system has more than one particle in Fleming-Viot mode ({})".format(N)
    if DEBUG_ESTIMATORS or show_messages(dict_params_info.get('verbose', False), dict_params_info.get('verbose_period', 1), dict_params_info.get('t_learn', 0)):
        print("Running Fleming-Viot simulation on {} particles and absorption size = {}..." \
              .format(N, dict_params_simul['buffer_size_activation'] - 1))
    t, event_times, phi, probas_stationary = \
        run_simulation_fv(  dict_params_info.get('t_learn', 0), envs, agent,
                            dict_params_simul['buffer_size_activation'] - 1,
                            # We pass the absorption size to this function
                            df_proba_surv, expected_absorption_time,
                            verbose=dict_params_info.get('verbose', False), verbose_period=dict_params_info.get('verbose_period', 1),
                            plot=DEBUG_TRAJECTORIES)
    assert t == len(event_times) - 1, "The last time step of the simulation ({}) coincides with the number of events observed ({})" \
                                        .format(t, len(event_times))
        ## We subtract 1 to len(event_times) because the first event time is 0.0 which is NOT an event time
    n_events_fv = t
    time_end_simulation_fv = event_times[-1]

    proba_blocking = compute_proba_blocking(agent, probas_stationary)
    expected_reward = estimate_expected_reward(envs[0], agent, probas_stationary)

    if DEBUG_ESTIMATORS or show_messages(dict_params_info.get('verbose', False), dict_params_info.get('verbose_period', False), dict_params_info.get('t_learn', 0)):
        print("\n*** RESULTS OF FLEMING-VIOT SIMULATION ***")
        # max_rows = pd.get_option('display.max_rows')
        # pd.set_option('display.max_rows', None)
        # print("Phi(t):\n{}".format(phi))
        # pd.set_option('display.max_rows', max_rows)
        print("P(K-1), P(K): {}".format(probas_stationary))
        print("Pr(BLOCK) = {}".format(proba_blocking))
        print("Expected reward = {}".format(expected_reward))

    return proba_blocking, expected_reward, probas_stationary, expected_absorption_time, n_cycles, \
           time_last_absorption, time_end_simulation_et, max_survival_time, time_end_simulation_fv, \
           n_events_et, n_events_fv


@measure_exec_time
def run_simulation_fv(t_learn, envs, agent, buffer_size_absorption, df_proba_surv, expected_absorption_time,
                      verbose=False, verbose_period=1, plot=False):
    """
    Runs the Fleming-Viot simulation of the particle system and estimates the expected reward
    (equivalent to the blocking probability when the blocking reward is always equal to 1)

    Arguments:
    envs: list
        List of queue environments used to run the FV process.

    agent: Agent
        Agent interacting with the set of environments given in envs.

    t_learn: int
        The learning time step to which the simulation will contribute.
        Only used for informative purposes or to decide whether to show information in the log.

    buffer_size_absorption: non-negative int
        Buffer size at which the particle is absorbed.

    df_proba_surv: pandas data frame
        Probability of survival given the process started at the activation set, Pr(T>t / s in activation set),
        used to estimate the blocking probability for each blocking buffer size.
        Typically this estimation is obtained by running a Monte-Carlo simulation of the queue.
        It should be a pandas data frame with at least two columns:
        - 't': the times at which the survival probability is estimated.
        - 'P(T>t)': the survival probability for each t.

    expected_absorption_time: positive float
        Expected absorption cycle time E(T_A) used to estimate the blocking probability
        for each blocking buffer size.

    agent: (opt) Agent
        Agent object that is responsible of performing the actions on the environment and learning from them.
        default: None, in which case the agent defined in the object is used

    verbose: (opt) bool
        Whether to be verbose in the simulation process.
        default: False

    verbose_period: (opt) int
        The time step period to be verbose.
        default: 1 => be verbose at every simulation step.

    Return: tuple
    Tuple with the following elements:
    (below, K is determined by the parameterized acceptance policy as the smallest buffer size with deterministic rejection)
    - t: the last time step (integer-valued) of the simulation process.
    - event_times: list of times at which an event occurred during the simulation.
    - dict_phi: dictionary of lists with the empirical distribution of the buffer sizes of interest, namely K-1 and K,
    which are an estimate of the probability of those buffer sizes conditional to survival (not absorption).
    - probas_stationary: dictionary of floats indexed by the buffer sizes of interest, namely K-1 and K, containing
    the estimated stationary probability at those buffer sizes.
    """

    # ---------------------------------- Auxiliary functions ------------------------------#
    def initialize_phi(envs, t: float, buffer_sizes: list):
        """
        Initializes the conditional probability of each buffer size of interest, which are given by the keys
        of the dictionary of the input parameter dict_phi, which is updated.

        Arguments:
        envs: list
            List of queue environments used to run the FV process.

        t: float

        buffer_sizes: list of int
            Buffer sizes at which the Phi dictionary should be initialized.

        Return: dict
            Dictionary, indexed by the given buffer sizes, of data frames that will be used to store
            the times 't' and the empirical distribution 'Phi' at which the latter changes.
            Each entry of the dictionary is initialized with a data frame with just one row containing the
            first time of measurement of Phi and the empirical distribution of the particles at the respective
            buffer size (indexing the dictionary).
        """
        dict_phi = dict()
        for bs in buffer_sizes:
            dict_phi[bs] = pd.DataFrame([[t, empirical_mean(envs, bs)]], columns=['t', 'Phi'])
        return dict_phi

    def empirical_mean(envs: list, buffer_size: int):
        "Compute the proportion of environments/particles at the given buffer size"
        return np.mean([int(bs == buffer_size) for bs in [env.getBufferSize() for env in envs]])

    def update_phi(N: int, t: float, dict_phi: dict, buffer_size_prev: int, buffer_size_cur: int):
        """
        Updates the conditional probability of each buffer size of interest, which are given by the keys
        of the dictionary of the input parameter phi, which is updated.

        Arguments:
        N: int
            Number of particles in the Fleming-Viot system.

        t: float
            Continuous-valued time at which the latest change of a particle's state happened.

        dict_phi: dict
            Dictionary, indexed by the buffer sizes of interest, of data frames containing the times 't' and
            the empirical distribution 'Phi' at which the latter changes.
            IMPORTANT: this input parameter is updated by the function with a new row whenever the value of Phi(t,bs)
            changes w.r.t. to the last stored value at the buffer size bs.

        buffer_size_prev: int
            The previous buffer size of the particle that just changed state on which basis Phi(t,bs) is updated.

        buffer_size_cur: int
            The current buffer size of the particle that just changed state on which basis Phi(t,bs) is updated.

        Return: dict
        The updated input dictionary which contains a new row for each buffer size for which the value Phi(t,bs) changes
        w.r.t. the last value stored in the data frame for that buffer size.
        """
        for bs in dict_phi.keys():
            assert dict_phi[bs].shape[0] > 0, "The Phi data frame has been initialized for buffer size = {}".format(bs)
            phi_cur = dict_phi[bs]['Phi'].iloc[-1]
            phi_new = empirical_mean_update(phi_cur, bs, buffer_size_prev, buffer_size_cur, N)
            if not np.isclose(phi_new, phi_cur):
                # Phi(t) changed at t
                # => add a new entry to the data frame containing Phi(t,bs)
                # (o.w. it's no use to store it because we only store the times at which Phi changes)
                dict_phi[bs] = pd.concat([dict_phi[bs], pd.DataFrame({'t': [t], 'Phi': [phi_new]})], axis=0)

        return dict_phi

    def empirical_mean_update(mean_value: float, buffer_size: int, buffer_size_prev: int, buffer_size_cur: int, N: int):
        """
        Update the proportion of environments/particles at the given buffer size based on the previous and current
        position (buffer size) of the particle

        Arguments:
        mean_value: float
            The current mean value to be updated.

        buffer_size: int
            The buffer size at which the empirical mean should be updated.

        buffer_size_prev: int
            The previous buffer size of the particle that just changed state.

        buffer_size_cur: int
            The current buffer size of the particle that just changed state.

        N: int
            Number of particles in the Fleming-Viot system.

        Return: float
            The updated empirical mean at the given buffer size, following the change of buffer size from
            `buffer_size_prev` to `buffer_size_cur`.
        """
        return mean_value + (int(buffer_size_cur == buffer_size) - int(buffer_size_prev == buffer_size)) / N

    def reactivate_particle(envs: list, idx_particle: int, K: int, absorption_number=None):
        """
        Reactivates a particle that has been absorbed

        Reactivation means that the state of the particle is set to be the state of the chosen particle.

        Arguments:
        envs: list
            List of queue environments used to run the FV process.

        idx_particle: int
            The index of the particle to be reactivated.

        K: int
            Queue's capacity.

        Return: int
        The index of the particle to which it is reactivated.
        """
        # Select a reactivation particle out of the other N-1 particles
        N = len(envs)
        assert N > 1, "There is more than one particle in the system (N={})".format(N)
        idx_reactivate = choose_particle(envs, idx_particle, N, K, ReactivateMethod.RANDOM) #, ReactivateMethod.VALUE_FUNCTION) #, ReactivateMethod.RANDOM)
        #idx_reactivate = choose_particle(envs, idx_particle, N, K, ReactivateMethod.ROBINS, absorption_number=absorption_number) #, ReactivateMethod.VALUE_FUNCTION) #, ReactivateMethod.RANDOM)

        # Update the state of the reactivated particle to the reactivation state
        envs[idx_particle].setState(envs[idx_reactivate].getState())

        return idx_reactivate

    def choose_particle(envs: list, idx_particle: int, N: int, K: int, method: ReactivateMethod, absorption_number=None):
        """
        Chooses a particle among N-1 possible particles using the given method

        Arguments:
        envs: list
            List of queue environments used to run the FV process.

        idx_particle: int
            The index of the particle to be reactivated.

        N: int
            Number of particles in the Fleming-Viot system.

        K: int
            Queue's capacity.

        method: ReactivateMethod
            Reactivation method to use which defines how the particle is chosen
            (e.g. randomly or using a probability distribution that depends on the value function of the state
            where each potential non-absorbed particle is located).

        Return: int
        The index of the chosen particle.
        """
        if method == ReactivateMethod.VALUE_FUNCTION:
            # TODO: (2022/03/10) Inquiry the value function of each of the N-1 particles based on their state and use a probability distribution that is proportional to it.
            # To begin with, consider an increasing linear function of the buffer size, and on a next step estimate the value function
            # appropriately based on the queue's dynamics.
            # Note: the value function should be defined as part of the learner in self.learnerV attribute of the SimulatorQueue object.

            # The value function is a linear function of the buffer size
            # Note that the length of `values` is equal to the number of particles in the system minus 1, N-1
            values = [envs[idx].getBufferSize() / K for idx in list(range(idx_particle)) + list(range(idx_particle+1, N))]
            assert len(values) == N - 1
            prob_values = [v / np.sum(values) for v in values]
            #print(np.c_[range(N-1), [envs[idx].getBufferSize() for idx in range(N-1)], [envs[idx].getBufferSize() / K for idx in range(N-1)], prob_values])
            idx_reactivate = np.random.choice(N-1, p=prob_values)
        elif method == ReactivateMethod.ROBINS:
            # Deterministic choice of the particle
            assert absorption_number is not None
            assert 0 <= absorption_number < N-1, "The absorption number is between 0 and N-2 = {} ({})".format(N-2, absorption_number)
            idx_reactivate = absorption_number
        else:
            # Random selection of active particles by default
            idx_reactivate = np.random.randint(0, N-1)  # The upper value is NOT included in the possible set of integers

        if idx_reactivate >= idx_particle:
            # The chosen particle is beyond the particle to reactivate
            # => increase the index of the particle by 1 so that we choose the correct particle
            idx_reactivate += 1
        assert 0 <= idx_reactivate < N, "The reactivation particle ID ({}) is between 0 and N (={}) for particle with ID = {}" \
            .format(idx_reactivate, N, idx_particle)
        assert idx_reactivate != idx_particle, "The particle chosen for reactivation ({}) is different from the particle being reactivated ({})" \
            .format(idx_reactivate, idx_particle)

        #print("Reactivated particle ID for particle `{}` and its buffer size: {}, {} (particles @: {})"
        #      .format(idx_particle, idx_reactivate, envs[idx_reactivate].getBufferSize(), [env.getBufferSize() for env in envs]))

        return idx_reactivate

    @measure_exec_time
    def estimate_stationary_probabilities(dict_phi, df_proba_surv, expected_absorption_time):
        """
        Estimates the stationary probability for each buffer size of interest in phi using the Fleming-Viot estimator

        Arguments:
        dict_phi: dict of data frames
            Empirical distribution of buffer sizes of interest which are the keys of the dictionary.
            Each data frame contains the times 't' and Phi values 'Phi' containing the empirical distribution at the
            buffer size indicated by the dictionary's key.

        df_proba_surv: pandas data frame
            Data frame with at least the following two columns:
            - 't': times at which the survival probability is estimated
            - 'P(T>t)': the survival probability estimate for the corresponding 't' value given the process
            started at the stationary activation distribution of states.

        expected_absorption_time: float
            Estimated expected absorption cycle time, i.e. the expected time the queue system takes in a
            re-absorption cycle when starting at the stationary absorption distribution of states.

        Return: tuple of dict
        Duple with two dictionaries indexed by the buffer sizes (bs) of interest with the following content:
        - the stationary probability of buffer size bs
        - the value of the integral P(T>t)*Phi(t,bs)
       """
        buffer_sizes_of_interest = sorted( list(dict_phi.keys()) )
        probas_stationary = dict()
        integrals = dict()
        for bs in buffer_sizes_of_interest:
            if dict_phi[bs].shape[0] == 1 and dict_phi[bs]['Phi'].iloc[-1] == 0.0:
                # Buffer size bs was never observed during the simulation
                probas_stationary[bs] = 0.0
                integrals[bs] = 0.0
            else:
                # Merge the times where (T>t) and Phi(t) are measured
                df_phi_proba_surv = merge_proba_survival_and_phi(df_proba_surv, dict_phi[bs])

                if DEBUG_ESTIMATORS:
                    plt.figure()
                    plt.step(df_phi_proba_surv['t'], df_phi_proba_surv['P(T>t)'], color="blue", where='post')
                    plt.step(df_phi_proba_surv['t'], df_phi_proba_surv['Phi'], color="red", where='post')
                    plt.step(df_phi_proba_surv['t'], df_phi_proba_surv['Phi']*df_phi_proba_surv['P(T>t)'], color="green", where='post')
                    plt.title("P(T>t) (blue) and Phi(t,bs) (red) and their product (green) for bs = {}".format(bs))

                # Stationary probability for each buffer size of interest
                probas_stationary[bs], integrals[bs] = estimate_proba_stationary(df_phi_proba_surv, expected_absorption_time)

        return probas_stationary, integrals

    @measure_exec_time
    def merge_proba_survival_and_phi(df_proba_surv, df_phi):
        """
        Merges the survival probability and the empirical distribution of the particle system at a particular
        buffer size of interest, on a common set of time values into a data frame.

        Arguments:
        df_proba_surv: pandas data frame
            Data frame containing the time 't' and the P(T>t) survival probability 'P(T>t)' given activation.

        df_phi: pandas data frame
            Data frame containing the time 't' and the Phi(t) value 'Phi' for a buffer size of interest.

        return: pandas data frame
        Data frame with the following columns:
        - 't': time at which a change in any of the input quantities happens
        - 'P(T>t)': survival probability given the process started at the stationary activation distribution of states.
        - 'Phi': empirical distribution at the buffer size of interest given the process started
        at the stationary activation distribution of states.
        """
        # Merge the time values at which the survival probability is measured (i.e. where it changes)
        # with the time values at which the empirical distribution Phi is measured for each buffer size of interest.
        t, proba_surv_by_t, phi_by_t = merge_values_in_time(list(df_proba_surv['t']), list(df_proba_surv['P(T>t)']),
                                                                list(df_phi['t']), list(df_phi['Phi']),
                                                                unique=False)

        # -- Merged data frame, where we add the dt, used in the computation of the integral
        df_merged = pd.DataFrame(np.c_[t, proba_surv_by_t, phi_by_t, np.r_[np.diff(t), 0.0]], columns=['t', 'P(T>t)', 'Phi', 'dt'])

        if DEBUG_ESTIMATORS:
            print("Survival Probability and Empirical Distribution for a buffer size of interest:")
            print(df_merged)

        return df_merged

    @measure_exec_time
    def estimate_proba_stationary(df_phi_proba_surv, expected_absorption_time):
        """
        Computes the stationary probability for a buffer size of interest via Approximation 1 in Matt's draft

        Arguments:
        df_phi_proba_surv: pandas data frame
            Data frame with the survival probability P(T>t) and the empirical distribution for each buffer size
            of interest on which the integral that leads to the Fleming-Viot estimation of the stationary
            probability of the buffer size is computed.

        expected_absorption_time: float
            Estimated expected absorption cycle time.

        Return: tuple
        Duple with the following content:
        - the estimated stationary probability
        - the value of the integral P(T>t)*Phi(t)
        """
        if tracemalloc.is_tracing():
            mem_usage = tracemalloc.get_traced_memory()
            print("[MEM] estimate_proba_stationary: Memory used so far: current={:.3f} MB, peak={:.3f} MB".format(
                mem_usage[0] / 1024 / 1024, mem_usage[1] / 1024 / 1024))
            mem_snapshot_1 = tracemalloc.take_snapshot()

        if expected_absorption_time <= 0.0 or np.isnan(expected_absorption_time) or expected_absorption_time is None:
            raise ValueError("The expected absorption time must be a positive float ({})".format(expected_absorption_time))

        # Integrate => Multiply the survival density function, the empirical distribution Phi, delta(t) and SUM
        if DEBUG_ESTIMATORS:
            max_rows = pd.get_option('display.max_rows')
            pd.set_option('display.max_rows', None)
            print("Data for integral:\n{}".format(df_phi_proba_surv))
            pd.set_option('display.max_rows', max_rows)

        df_phi_proba_surv_Phi_gt0 = df_phi_proba_surv.loc[ df_phi_proba_surv['Phi'] > 0, ]
        integral = np.sum( df_phi_proba_surv_Phi_gt0['P(T>t)'] * df_phi_proba_surv_Phi_gt0['Phi'] * df_phi_proba_surv_Phi_gt0['dt'] )

        if DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
            print("integral = {:.3f}, E(T) = {:.3f}".format(integral, expected_absorption_time))

        proba_stationary = integral / expected_absorption_time

        if tracemalloc.is_tracing():
            mem_snapshot_2 = tracemalloc.take_snapshot()
            mem_stats_diff = mem_snapshot_2.compare_to(mem_snapshot_1,
                                                       key_type='lineno')  # Possible key_type's are 'filename', 'lineno', 'traceback'
            if show_messages(verbose, verbose_period, t_learn):
                print("[MEM] estimate_proba_stationary: Top difference in memory usage:")
                for stat in mem_stats_diff[:10]:
                    # if stat.size / stat.count > 1E6:   # To print the events with largest memory consumption for EACH of their occurrence
                    print(stat)

        return proba_stationary, integral
    # ---------------------------------- Auxiliary functions ------------------------------#

    # ---------------------------- Check input parameters ---------------------------------#
    if buffer_size_absorption < 0 \
            or not isinstance(buffer_size_absorption, int) and not isinstance(buffer_size_absorption, np.int32) \
            and not isinstance(buffer_size_absorption, np.int64):
        raise ValueError("The buffer size for absorption must be integer and >= 0 ({})".format(buffer_size_absorption))

    # Check the survival probability values
    if df_proba_surv is None and not isinstance(df_proba_surv, pd.core.frame.DataFrame):
        raise ValueError("The survival probability estimate must be given and be a DataFrame")
    if 't' not in df_proba_surv.columns or 'P(T>t)' not in df_proba_surv.columns:
        raise ValueError(
            "The data frame with the estimated survival probability must contain at least columns 't' and 'P(T>t)' (columns: {})" \
            .format(df_proba_surv.columns))
    if df_proba_surv['P(T>t)'].iloc[0] != 1.0:
        raise ValueError(
            "The first value of the survival function must be 1.0 ({:.3f})".format(df_proba_surv['P(T>t)'].iloc[0]))
    if df_proba_surv['P(T>t)'].iloc[-1] not in [1.0, 0.0]:
        raise ValueError("The survival function at the last measured time is either 1.0 "
                         "(when no particles have been absorbed) or 0.0 "
                         "(when at least one particle has been absorbed) ({})".format(df_proba_surv['P(T>t)'].iloc[-1]))
    # ---------------------------- Check input parameters ---------------------------------#

    # -- Parse input parameters
    N = len(envs)

    # Set the start state of each environment/particle to an activation state, as this is a requirement
    # for the empirical distribution Phi(t).
    for i, env in enumerate(envs):
        # Set the start state of the queue environment so that we start at an activation state
        start_state = choose_state_for_buffer_size(env, buffer_size_absorption + 1)
        env.setState((start_state, None))
        assert env.getBufferSize() == buffer_size_absorption + 1, \
            "The start state of all environments/particles must be an activation state (start state of env #{}: {})" \
                .format(i, env.getState())

    # Buffer sizes whose stationary probability is of interest
    buffer_sizes_of_interest = get_blocking_buffer_sizes(agent)

    # Event times (continuous times at which an event happens)
    # The first event time is 0.0
    event_times = [0.0]

    # Phi(t, bs): Empirical probability of the buffer sizes of interest (bs)
    # at each time when an event happens (ANY event, both arriving job or completed service)
    dict_phi = initialize_phi(envs, event_times[-1], buffer_sizes_of_interest)

    # Time step in the queue trajectory (the first time step is t = 0)
    done = False
    t = 0
    maxtime = df_proba_surv['t'].iloc[-1]
        ## maxtime: it's useless to go with the FV simulation beyond the maximum observed survival time
        ## because the contribution to the integral used in the estimation of the average reward is 0.0 after that.
    idx_reactivate = None   # This is only needed when we want to plot a vertical line in the particle evolution plot with the color of the particle to which an absorbed particle is reactivated
    if plot:
        # Initialize the plot
        ax = plt.figure().subplots(1,1)
        if agent.getLearnerP() is not None:
            assert agent.getAcceptancePolicy().getThetaParameter() == agent.getLearnerP().getPolicy().getThetaParameter()
        K = agent.getAcceptancePolicy().getBufferSizeForDeterministicBlocking()
        ax.set_title("N = {}, K= {}, maxtime = {:.1f}" \
                     .format(N,
                             K,
                             maxtime))

        # Variables needed to update the plot that shows the trajectories of the particles (online)
        time0 = [0.0] * N
        y0 = [buffer_size_absorption + 1] * N
    if DEBUG_ESTIMATORS:
        # Check realization of the arrival and service rates
        times_last_arrival = [0.0] * N
        jobs_arrival = []
        times_inter_arrival = []

        if envs[0].getBufferSize() == 0:
            # Initially all the particles have no jobs, therefore no service events can happen for any of them
            for env in envs:
                assert env.getBufferSize() == 0
            times_last_service = [np.nan] * N
        else:
            times_last_service = [0.0] * N
        servers_service = []
        times_service = []
    while not done:
        t += 1
        # We count the absorption number, an integer between 0 and N-2 which is used to deterministically choose
        # the reativation particle, in order to save time by not having to generate a uniform random number.
        # Note that the range from 0 to N-2 allows us to choose one of the N-1 particles to which the absorbed particle
        # can be reactivated into.
        absorption_number = t % (N - 1)

        # Generate an event over all possible event types and particles
        # Note: this function takes care of NOT generating a service event when a server is empty.
        time, event, job_class_or_server, idx_particle = generate_event(envs)
        assert 0 <= idx_particle < N, "The chosen particle for change is between 0 and N-1={} ({})".format(N-1, idx_particle)
        # Store the absolute event time
        event_times += [event_times[-1] + time]

        # Get the current state of the selected particle because that's the one whose state we are going to (possibly) change
        state = envs[idx_particle].getState()
        if event == Event.BIRTH:
            # The event is an incoming job class
            # => Update the state of the queue, apply the acceptance policy, and finally the server assignment policy
            action, next_state, reward, gradient_for_action = manage_job_arrival(t, envs[idx_particle], agent, state, job_class_or_server)
            if DEBUG_ESTIMATORS:
                # TODO: (2022/01/21) Fix this logic as it does NOT take care of different job classes. In fact the latest arrival time (regardless of the job class) is used as reference time to compute the new inter-arrival time for the current job class!)
                jobs_arrival += [job_class_or_server]
                times_inter_arrival += [event_times[-1] - times_last_arrival[idx_particle]]
                # Prepare for the next iteration
                times_last_arrival[idx_particle] = event_times[-1]
                if envs[idx_particle].getBufferSizeFromState(next_state) == 1:
                    # The particle can now experience a service event
                    # => Reset the time of the last service to the current time, so that we can measure the time
                    # of the next service experienced by this particle.
                    times_last_service[idx_particle] = event_times[-1]
        elif event == Event.DEATH:
            # The event is a completed service
            # => Update the state of the queue
            # (Note that the state of the queue in envs[idx_particle] is updated by manage_service())
            action, next_state, reward = manage_service(envs[idx_particle], agent, state, job_class_or_server)

            # Check if the particle has been ABSORBED
            if envs[idx_particle].getBufferSize() == buffer_size_absorption:
                # The particle has been absorbed
                # => Reactivate it to any of the other particles
                if plot:
                    # Show the absorption before reactivation takes place
                    y = envs[idx_particle].getBufferSize()
                    J = buffer_size_absorption + 1
                    print("* Particle {} ABSORBED! (at time = {:.3f}, buffer size = {})".format(idx_particle, event_times[-1], y))
                    plot_update_trajectory( ax, idx_particle, N, K, J,
                                            time0[idx_particle], y0[idx_particle], event_times[-1], y)
                    # Update the coordinates of the latest plotted point for this particle, for the next iteration
                    time0[idx_particle] = event_times[-1]
                    y0[idx_particle] = y
                idx_reactivate = reactivate_particle(envs, idx_particle, buffer_sizes_of_interest[-1], absorption_number=absorption_number)
                next_state = envs[idx_particle].getState()
                assert envs[idx_particle].getBufferSize() > buffer_size_absorption
                if DEBUG_TRAJECTORIES:
                    print("*** Particle {} REACTIVATED to particle {} at position {}".format(idx_particle, idx_reactivate, envs[idx_reactivate].getBufferSize()))

            # This should come AFTER the possible reactivation, because the next state will never be 0
            # when reactivation takes place.
            if DEBUG_ESTIMATORS:
                assert not np.isnan(times_last_service[idx_particle])
                servers_service += [job_class_or_server]
                times_service += [event_times[-1] - times_last_service[idx_particle]]
                # Prepare for the next iteration
                if envs[idx_particle].getBufferSize() == 0:
                    # The particle can no longer experience a service event
                    # => Reset the time of the last service to NaN
                    times_last_service[idx_particle] = np.nan
                else:
                    times_last_service[idx_particle] = event_times[-1]

        if DEBUG_TRAJECTORIES:
            print("P={}: {} | t={}: time={}, event={}, action={} -> state={}, reward={}" \
                  .format(idx_particle, state, t, event_times[-1], event, action, next_state, reward), end="\n")

        # Update Phi based on the new state of the changed particle
        buffer_size_prev = envs[0].getBufferSizeFromState(state)
        buffer_size = envs[0].getBufferSizeFromState(next_state)
        dict_phi = update_phi(len(envs), event_times[-1], dict_phi, buffer_size_prev, buffer_size)

        if plot:
            y = envs[0].getBufferSizeFromState(next_state)
            if agent.getLearnerP() is not None:
                assert agent.getAcceptancePolicy().getThetaParameter() == agent.getLearnerP().getPolicy().getThetaParameter()
            K = agent.getAcceptancePolicy().getBufferSizeForDeterministicBlocking()
            J = buffer_size_absorption + 1
            plot_update_trajectory( ax, idx_particle, N, K, J,
                                    time0[idx_particle], y0[idx_particle], event_times[-1], y,
                                    r=None) #idx_reactivate)    # Use idx_reactivate if we want to see a vertical line at the time of absorption with the color of the activated particle
            time0[idx_particle] = event_times[-1]
            y0[idx_particle] = y

        idx_reactivate = None

        if DEBUG_TRAJECTORIES:
            # New line to separate from next iteration
            print("")

        # Stop when we reached the maxtime, which is set above to the maximum observed survival time
        # (since going beyond does not contribute to the integral used in the FV estimator)
        done = event_times[-1] > maxtime

    # DONE
    if show_messages(verbose, verbose_period, t_learn):
        print(
            "==> agent ENDS at discrete time t={} (continuous time = {:.1f}, compared to maximum observed time for P(T>t) = {:.1f}) at state {} coming from state = {}, action = {}, reward = {})" \
            .format(t, event_times[-1], df_proba_surv['t'].iloc[-1], envs[idx_particle].getState(), state, action, reward))

    if DEBUG_ESTIMATORS:
        # NOTE: It is assumed that the job arrival rates and service rates are the same for ALL N environments
        # where the FV process is run.
        job_rates_by_server = compute_job_rates_by_server(envs[0].getJobClassRates(),
                                                          envs[0].getNumServers(),
                                                          agent.getAssignmentPolicy().getProbabilisticMap())
        plotting.plot_event_times(job_rates_by_server, times_inter_arrival, jobs_arrival, class_name="Job class")
        plotting.plot_event_times(envs[0].getServiceRates(), times_service, servers_service, class_name="Server")

    # Compute the stationary probability of each buffer size bs in Phi(t,bs) using Phi(t,bs), P(T>t) and E(T_A)
    probas_stationary, integrals = estimate_stationary_probabilities(dict_phi, df_proba_surv,
                                                                     expected_absorption_time)

    return t, event_times, dict_phi, probas_stationary


def plot_update_trajectory(ax, p, N, K, J, x0, y0, x1, y1, marker='-', r=None):
    "p: particle number to update; r: particle to which p is reactivated (if any)"
    assert ax is not None
    colormap = cm.get_cmap("jet")
    if r is not None:
        # Mark the color of the line with the color of the reactivation particle
        c = r
    else:
        c = p
    color = colormap((c + 1) / N)
    # Non-overlapping step plots at vertical positions (K+1)*p
    print("Updating TRAJECTORY of particle {} (K={}, J={}): [t={:.3f}, y={}] -> [t={:.3f}, y={}]".format(p, K, J, x0, y0, x1, y1))
    ax.step([x0, x1], [(K + 1) * p + y0, (K + 1) * p + y1], marker, where='post', color=color, markersize=3)

    # Reference lines
    reflines_zero = range(0, (K+1)*N, K+1)
    reflines_absorption = range(J-1, J-1+(K+1)*N, K+1)
    reflines_block = range(K, K+(K+1)*N, K+1)
    ax.hlines(reflines_block, 0, x1, color='gray', linestyles='dashed')
    ax.hlines(reflines_absorption, 0, x1, color='red', linestyles='dashed')
    ax.hlines(reflines_zero, 0, x1, color='gray')

    # Vertical line when the particle has been reactivated
    if r is not None:
        ax.axvline(x1, color=color, linestyle='dashed')


def plot_trajectory(env, agent, J):
    """
    Plots the buffer sizes of the system as a function of time observed during simulation.
    We assume there is only one queue to be plotted.
    This is used to plot the single queue simulated to estimate the quantities wth Monte-Carlo (e.g. P(T>t) and E(T)).
    """
    assert agent.getLearnerV() is not None
    if agent.getLearnerP() is not None:
        assert agent.getAcceptancePolicy().getThetaParameter() == agent.getLearnerP().getPolicy().getThetaParameter()
    K = agent.getAcceptancePolicy().getBufferSizeForDeterministicBlocking()
    times = agent.getLearnerV().getTimes()
    states = agent.getLearnerV().getStates()
    #print("Times and states to plot:\n{}".format(np.c_[times, states]))
    buffer_sizes = [env.getBufferSizeFromState(s) for s in states]

    # Non-overlapping step plots at vertical positions (K+1)*p
    ax = plt.figure().subplots(1,1)
    ax.step(times, buffer_sizes, 'x-', where='post', color="blue", markersize=3)

    # Reference lines
    ax.axhline(0, color="gray")
    ax.axhline(J-1, color="red", linestyle="dashed")
    ax.axhline(K, color='gray', linestyle='dashed')
    ax.set_title("[MC] Evolution of the system's buffer size used to estimate P(T>t) and E(T): K = {}, J = {}".format(K, J))

    return ax
#-------------------------------------------- FUNCTIONS --------------------------------------#
