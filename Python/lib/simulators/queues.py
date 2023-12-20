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
from typing import Union

from collections import OrderedDict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

from Python.lib.agents.learners import LearnerTypes
from Python.lib.agents.learners.continuing.fv import LeaFV
from Python.lib.agents.policies import PolicyTypes
from Python.lib.agents.policies.job_assignment import define_uniform_job_assignment_policy, PolJobAssignmentProbabilistic
from Python.lib.agents.policies.parameterized import PolQueueTwoActionsLinearStep
from Python.lib.agents.queues import AgeQueue
import Python.lib.environments.queues as env_queues
from Python.lib.environments.queues import Actions, BufferType
from Python.lib.estimators import DEBUG_ESTIMATORS, estimate_expected_cycle_time, estimate_expected_stopping_time_in_cycle
from Python.lib.estimators.fv import initialize_phi, update_phi, estimate_stationary_probabilities, SurvivalProbabilityEstimation

from Python.lib.simulators import BURNIN_TIME_STEPS, MIN_NUM_CYCLES_FOR_EXPECTATIONS, DEBUG_TRAJECTORIES, \
    analyze_event_times, check_done, choose_state_from_setofstates_based_on_distribution, is_state_in_setofstates, parse_simulation_parameters, \
    show_messages, step, update_trajectory, SetOfStates
from Python.lib.simulators.discrete import Simulator
from Python.lib.simulators.fv import ReactivateMethod, reactivate_particle
from Python.lib.queues import Event, QueueMM

from Python.lib.utils.basic import array_of_objects, get_current_datetime_as_string, \
    get_datetime_from_string, is_scalar, index_linear2multi, measure_exec_time, merge_values_in_time
from Python.lib.utils.computing import compute_job_rates_by_server, compute_number_of_burnin_cycles_from_burnin_time, \
    compute_blocking_probability_birth_death_process, compute_survival_probability, generate_min_exponential_time, \
    all_combos_with_sum, compute_stationary_probability_knapsack_when_blocking_by_class

@unique
class LearningMode(Enum):
    "Learning mode of theta in the parameterized policy in SimulatorQueue.learn()"
    REINFORCE_RETURN = 1        # When learning is based on the observed return G(t),
                                # which is an ESTIMATE of the true gradient of the average reward
                                # (which is equal to the gradient of the average state value as proved in section 13.6 of Sutton)
                                # as given by expression 13.8 in Sutton, pag. 327 or expression (9) in Massaro et al.
                                # This is the REINFORCE algorithm.
    REINFORCE_TRUE = 2          # When learning is based on the TRUE expected value of the gradient of the average reward (or of the average state value)
                                # as given in expression (8) in Massaro et al.
                                # Note that the GRADIENTS of the average reward and of the average state value coincide... NOT the values before the calculation of the gradient.
                                # (i.e. in general the average reward is DIFFERENT from the average state value, although in some cases they may coincide --e.g. exercise 10.6 in Sutton, pag. 251)
    IGA = 3                     # When learning is based on Integer Gradient Ascent, where delta(theta) is +/- 1 or 0.


class SimulatorQueue(Simulator):
    """
    Simulator class that runs a Reinforcement Learning simulation on a given Queue environment `env` and an `agent`
    (which contains the characteristics of value functions and policies to be learned during the simulation)
    using:
    - the learning mode (e.g. REINFORCE_TRUE or REINFORCE_RETURN)
    - the number of learning steps
    - the number of simulation steps or arrival events per learning step
    specified in the `dict_params_learning` dictionary.
    Note that here the learning step refers to a step towards learning the optimal POLICY, not the learning step towards learning
    the value functions. I.e. we are in the context of two time steps for learning and in this context, the policy learning step
    is slower (i.e. learning happens less often) than the value function learning step (which happens at every simulation step
    in which the queue experiences an event).

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

    dict_params_learning: dict
        Dictionary containing the learning parameters as follows:
        - 'mode': learning mode, one of the values of the LearningMode enum
        - 't_learn': number of learning steps (of the value functions and policies)

    N: int
        Number of queues to simulate.
        default: 1
    """

    def __init__(self, env, agent, dict_params_learning, case=1, replication=1, N=1, seed=None, log=False, save=False, logsdir=None, resultsdir=None, debug=False):
        super().__init__(env, agent, case=case, replication=replication, seed=seed, log=log, save=save, logsdir=logsdir, resultsdir=resultsdir, debug=debug)
        self.dict_params_learning = dict_params_learning
        # Attributes used during Fleming-Viot simulation (each env in envs is one of the N particles)
        self.N = N
        # TODO: (2022/01/17) We should NOT copy ALL the environment to work with the N particles... In fact, the environment characteristics are the same for ALL particles, the only thing that changes is the latest state and change time of each particle
        # This may be behind the memory error I got when running the FV process at my laptop...?

        # Note: it's important that the first environment stored in self.envs be self.env (and NOT a copy)
        # because in the process we sometimes refer to it as self.envs[0] (for instance when resetting the environments)
        # or as self.env (for instance when estimating Q(K-1, a) via Monte-Carlo on one particle (both in MC and FV modes)
        self.envs = [self.env if i == 0 else copy.deepcopy(self.env) for i in range(self.N)]

        # Learners for the value function and the policy
        if agent is not None:
            self.learnerV = self.agent.getLearnerV()
            self.learnerP = self.agent.getLearnerP()
            if self.env.getBufferType() != BufferType.NOBUFFER:
                # Job rates by server (assuming jobs are pre-assigned to servers)
                job_assignment_policy = self.agent.getAssignmentPolicy() if self.agent.getAssignmentPolicy() is not None \
                                                                         else define_uniform_job_assignment_policy(len(self.env.getJobClassRates()), self.env.getNumServers())
                self.job_rates_by_server = compute_job_rates_by_server(self.env.getJobClassRates(),
                                                                       self.env.getNumServers(),
                                                                       job_assignment_policy.getProbabilisticMap())
            else:
                # Job rates by server are actually the job class arrival rates
                self.job_rates_by_server = self.env.getJobClassRates()
            # Compute the server loads which is used to compute the expected relative errors associated to
            # the number of particles N and the number of arrival events T used in the FVRL algorithm
            self.rhos = [l / m for l, m in zip(self.getJobRatesByServer(), self.env.getServiceRates())]
        else:
            self.learnerV = None
            self.learnerP = None
            self.job_rates_by_server = None
            if self.env.getNumServers() == 1:
                # The arrival rate of the different job classes sum up to obtain the arrival rate of ANY type of job to the single server
                self.rhos = np.sum( self.env.getJobClassRates() )
            else:
                # There is no agent managing the arrival of jobs and how jobs are assigned to servers,
                # so we take the arrival rates from the birth rates of the QueueMM object that should be stored
                # in the `env` queue environment.
                self.rhos = [l / m  for l, m in zip(self.env.getQueue().getBirthRates(), self.env.getQueue().getDeathRates())]

        self.reset_learning_history()

    def reset_learning_history(self):
        """
        Resets the policy learning history. This should be called for any new learning simulation that is run (e.g. a new replication of the learning process)
        """
        self.alphas = []                    # List of learning rates used in the learning of the policy when updating from theta to theta_next (i.e. from thetas -> thetas_updated).
        # TODO: (2023/01/28) The storage of the historic thetas observed during learning should probably NOT be stored as part of the simulator object but we should use the history already stored by the policy learner (see the LeaPolicyGradient class)
        # In addition, the values of thetas_updated would be obtained from thetas by just looking at the following index (i.e. thetas[1] contains the updated theta from thetas[0], and so forth)
        self.thetas = []                    # List of theta values before the update by the policy learning step.
        self.thetas_updated = []            # List of theta values AFTER the update by the policy learning step.
        self.state_coverage = []            # List of percent of blocking states or buffer sizes that are visited by the simulation. There is one percent coverage per policy threshold, since each threshold has an associated blocking size, be it a buffer size (when the acceptance policy blocks based on buffer size) or a job occupancy (when the acceptance policy blocks on job-class occupancy).
        self.proba_stationary = []          # List of the estimates of the stationary probability for each blocking state.
        self.proba_stationary_true = []     # List of the TRUE stationary probability for each blocking state.
        self.proba_blocking = []            # List of blocking probability estimated at each learning step.
        self.proba_blocking_true = []       # List of the TRUE blocking probability at each learning step.
        self.expected_reward = []           # List of the expected reward after each learning step.
        self.expected_reward_true = []      # List of the TRUE expected reward after each learning step.
        self.error_proba_stationary = []    # List of the estimation errors of the stationary probability for each blocking state at each learning step.
        self.error_proba_blocking = []      # List of the relative error of the blocking probability estimate at each learning step.
        self.error_expected_reward = []     # List of the relative error of the expected reward estimate at each learning step.
        self.Q_diff = []                    # List containing selected statistics of the Q_diff values (Q(x,a=1) - Q(x,a=0)) that contribute to the theta udpate at each learning step. See the learn() method for more details.
        self.V = []                         # *Normally*, the value function at the theta before the update,
                                            # i.e. the average reward rho observed in each queue simulation episode for a fixed policy before the update of theta takes place.
                                            # HOWEVER, it is NOT always the case that this is the average reward, see the self.learn() method for more details.
        self.gradV = []                     # List of the gradient of the value function (grad(J) in Sutton) for the theta before the update, that is responsible for the theta update.
        self.G = []                         # List of the return G at each learning step.
        self.n_events_mc = []               # List of the number of events used in the Monte-Carlo simulation, either to estimate the stationary probabilities under the MC approach, or P(T>t) and E(T_A) under the FV approach.
        self.n_events_fv = []               # List of the number of events used to estimate the stationary probability in each episode.
        self.n_trajectories_Q = []          # List of the average number of trajectory pairs used to estimate Q_diff at each learning step over all blocking states.

    # IMPORTANT: This method overrides the method of the superclass! (so when reset() is called from the super()
    # object, actually THIS reset() method defined here is called!
    def reset(self, state=None, reset_learning_history=False, reset_value_functions=False, reset_counts=False):
        "Resets the environments used in the simulation. How much is reset depends on the given parameter values"

        # Reset the policy learning history
        if reset_learning_history:
            self.reset_learning_history()
            # We should also erase the policy learning history below
            reset_policy_trajectory = True
        else:
            # We do not want to erase the policy learning history done below
            reset_policy_trajectory = False

        # Reset the environments (particles) used in the simulation
        if state is not None:
            for env in self.envs:
                env.reset(state)

        if self.agent is not None:
            # Reset the learners
            # This is expected to be called after each learning step, i.e. before a new queue simulation is run.
            # Based on this, note that we do NOT reset the trajectory stored in the policy learner because we want to
            # keep the WHOLE history of learning (i.e. for all queue simulation episodes) in the policy learner.
            # Note that we reset the time and alphas of the value function learner (because this happens within each
            # learning step or queue simulation episode), but we do NOT reset the time and alphas of the policy
            # learner because the learning step here is defined by a new queue simulation episode.
            if self.agent.getLearnerV() is not None:
                self.agent.getLearnerV().reset(reset_learning_epoch=True,  reset_alphas=True,  reset_value_functions=reset_value_functions, reset_trajectory=True, reset_counts=reset_counts)
            if self.agent.getLearnerP() is not None:
                self.agent.getLearnerP().reset(reset_learning_epoch=False, reset_alphas=False, reset_value_functions=reset_value_functions, reset_trajectory=reset_policy_trajectory, reset_counts=reset_counts)
            if not reset_value_functions:
                # Set the start value of the value functions to their current estimate
                # Goal: we can use the current estimated values in the next learning step,
                # e.g. use it as a baseline when computing the time difference delta(t).
                if self.agent.getLearnerV() is not None:
                    self.agent.getLearnerV().setVStart(self.agent.getLearnerV().getV())
                    self.agent.getLearnerV().setQStart(self.agent.getLearnerV().getQ())

    def initialize_results_file(self):
        if self.fh_results is not None:
            self.fh_results.write("case|replication|t_learn|theta_true|theta|theta_next|K|J/K|J|burnin_time_steps|min_n_cycles|exponent|N|T|err_phi|err_et|seed|E(T)|n_cycles|max_survival_time|n_blocking_states|blocking_states_coverage|proba_blocking|proba_blocking_true|error_proba_blocking|expected_reward|expected_reward_true|error_expected_reward|Q_diff_stats|alpha|V|gradV|n_events_mc|n_events_fv|n_trajectories_Q\n")

    def run(self, dict_params_simul: dict, dict_params_info: dict={'plot': False, 'log': False}, dict_info: dict={},
            start_state=None,
            seed=None,
            verbose=False, verbose_period=1,
            colormap="seismic", pause=0):
        """
        Runs a Reinforcement Learning experiment on a queue environment for the given number of learning steps.
        No reset of the learning history is done at the beginning of the simulation.
        It is assumed that this reset, when needed, is done by the calling function, e.g. by calling
        `reset(reset_learning_history=True, reset_value_functions=True, reset_counts=True)`,
        a method defined by this class.

        Parameters:
        dict_params_simul: dict
            Dictionary containing the simulation and estimation parameters as follows:
            - 'theta_true': true value of the theta parameter to use for the simulation, which should be estimated
            by the process.
            - 'theta_start': initial value of theta at which the learning process starts.
            - 'nparticles': # particles to use in the FV simulation.
            - 't_sim': either number of simulation steps (MC learning) or number of arrival events (FVRL) per learning step.
            This is either a scalar or a list with as many elements as the number of learning steps defined in
            self.dict_params_learning['t_learn'].
            This latter case serves the purpose of running as many simulation steps as were run in a benchmark method
            for *each* learning step (for instance when comparing the results by Monte-Carlo with those by Fleming-Viot).
            - 'buffer_size_activation_factor': factor multiplying the blocking size K defining the value of
            the buffer size for activation.
            - 'burnin_time_steps': (opt) number of time steps to use as burn-in, before assuming that the process is
            in stationary regime. This impacts the estimation of expectations. Default: BURNIN_TIME_STEPS
            - 'min_num_cycles_for_expectations': (opt) minimum number of observed cycles in order to estimate expectations
            (instead of setting their values to NaN). Default: MIN_NUM_CYCLES_FOR_EXPECTATIONS

        dict_params_info: (opt) dict --> only used when the learner of the value functions `learnerV` is LeaFV
            Dictionary containing general info parameters as follows:
            - 'plot': whether to create plots
            - 'log': whether to show messages
            default: None

        dict_info: (opt) dict
            Dictionary with information of interest to store in the results file.
            From this dictionary, when defined, we extract the following keys:
            - 'exponent': the exponent used to define the values of N (# particles) and T (# arrival events) used in the
            simulation.
            default: {}

        start_state: (opt) Queue-environment-dependent
            (Environment) State at which the queue environment starts for the simulation.
            Its type depends on how the queue environment defines the state of the system.
            Ex: when the queue is of class GenericEnvQueueWithJobClasses, the state is a tuple with the following information:
            (queue_state, job_class) = (state of the queue itself, class of the arriving job).
            default: None, in which case the start state is defined by the simulation process, based on the learning requirements.

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
                The policy can be retrieved from this object by calling learnerP.getPolicy()
                and we can find out if the policy to learn is actually a multi-policy by calling
                learnerP.getIsMultiPolicy().
            - df_learning: pandas data frame containing two columns: 'theta', 'gradV' containing the history
            of the observed thetas and gradients of the value function responsible for its update.
        """
        # -- Parse input parameters
        # 1) dict_params_simul
        set_required_entries_params_simul= {'theta_true', 'theta_start', 'nparticles', 't_sim', 'buffer_size_activation_factor'}
        if not set_required_entries_params_simul.issubset(dict_params_simul.keys()):
            raise ValueError("Missing entries in the dict_params_simul dictionary: {}" \
                             .format(set_required_entries_params_simul.difference(dict_params_simul.keys())))

        if is_scalar(dict_params_simul['t_sim']):
            # Case when the simulation time (either #steps or #arrivals) is the same
            # for all learning steps (typically when learning via the FVRL method).
            # Otherwise, in the MC learning method, the number of simulation steps
            # is normally read from a benchmark file and therefore is expected to be
            # different for each learning step (although it could also be the same
            # as the *total* number of simulation steps could be divided equally on each learning step).
            # => Use the simulation time parameter defined in the object
            dict_params_simul['T'] = dict_params_simul['t_sim'] # Recall that this is the #steps in MC and #arrivals in FVRL

        # -- Default values of parameters used in estimation.
        # Number of burn-in time steps to allow the simulation to run before considering that the Markov process
        # has reached stationarity regime.
        dict_params_simul['burnin_time_steps'] = dict_params_simul.get('burnin_time_steps', BURNIN_TIME_STEPS)
        # Minimum number of cycles that should be observed in order to consider that the estimation of the expected
        # cycle time (be it expected return time to the initial buffer size for Monte-Carlo or expected reabsorption
        # cycle time for Fleming-Viot) is reliable.
        dict_params_simul['min_num_cycles_for_expectations'] = dict_params_simul.get('min_num_cycles_for_expectations', MIN_NUM_CYCLES_FOR_EXPECTATIONS)

        # 2) self.dict_params_learning (an *attribute* of this object! --i.e. NOT an input parameter of this method run())
        set_required_entries_params_learning = {'mode', 't_learn'}
        if not set_required_entries_params_learning.issubset(self.dict_params_learning.keys()):
            raise ValueError("Missing entries in the dict_params_learning dictionary: {}" \
                             .format(set_required_entries_params_learning.difference(self.dict_params_learning.keys())))

        # 3) self.dict_params_info
        dict_params_info['verbose'] = verbose
        dict_params_info['verbose_period'] = verbose_period
        dict_params_info['plot'] = dict_params_info.get('plot', False)

        if dict_params_info['plot']:
            # TODO: See the run() method in the Simulator class
            pass

        # Set the number of particles to use in the simulation
        self.setNumberParticlesAndCreateEnvironments(dict_params_simul['nparticles'])
        assert self.N == dict_params_simul['nparticles']

        # Set the true theta parameter (which is defined by a parameter of the queue environment) to use for the simulation
        # NOTE that this is done in EACH environment used in the simulation, because this environment is the one passed to functions
        # and the one that is used to retrieve the value of the true theta parameter.
        if self.envs[0].getBufferType() == BufferType.SINGLE:
            assert is_scalar(dict_params_simul['theta_true']) and len(self.agent.getAcceptancePolicies()) == 1 or \
                   len(dict_params_simul['theta_true']) == len(self.agent.getAcceptancePolicies()), \
                    "The true theta parameter `theta_true` defined in the `dict_params_simul` parameter must be scalar " \
                    "(and the number of acceptance policies must be 1 in that case) or of length equal to the number of acceptance policies for queues system with a single buffer: {}" \
                    .format(dict_params_simul['theta_true'])
            _theta_true = dict_params_simul['theta_true'] if is_scalar(dict_params_simul['theta_true']) else dict_params_simul['theta_true'][0]
            for env in self.envs:
                # Note: (2023/06/19) As reference buffer size we define ceil(theta_true + 1) and NOT round(theta_true + 1) as is defined elsewhere
                # when computing the optimum *true* K: the reason we do this is that the optimum K is a little smaller than the reference buffer size value,
                # so we are in business if we set the reference value (used in the exponential function of the buffer size that models the blocking cost)
                # as the ceiling and not as round of theta_true+1.
                env.setParamsRewardFunc(dict({'buffer_size_ref': np.ceil(_theta_true + 1)}))

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
        self.getLearnerP().setThetaParameter(dict_params_simul['theta_start'])

        if verbose:
            print("Value function at start of experiment: {}".format(self.getLearnerV().getV()))

        # There are two simulation time scales (both integers and starting at 0):
        # - t: the number of queue state changes (a.k.a. number of queue iterations)
        # - t_learn: the number of learning steps (of the value functions and policies)
        t_learn = 0
        # Need to keep track of the original capacity of the queue environment because,
        # in problems where blocking is not based on a single buffer (i.e. when self.env.BufferType != BufferType.SINGLE)
        # the capacity may change along the way of learning, as the capacity may be limited by the blocking sizes of
        # each job class arriving to the system (e.g. when capacity = 10 and blocking sizes by job class are [2, 3, 1],
        # the effective capacity of the system is 2 + 3 + 1 = 6 < 10 = original capacity).
        # As the blocking sizes change during learning, the new effective capacity of the system needs to be re-computed
        # having the ORIGINAL capacity of the system as upper bound, NOT the latest effective capacity computed by the
        # learning process!
        env_original_capacity = self.env.getCapacity()
        while t_learn < self.dict_params_learning['t_learn']:
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
                      .format(t_learn, self.dict_params_learning['t_learn']))
                theta_parameters = self.getLearnerP().getThetaParameter()
                print("Theta parameter of policies in the policy learner: {}" \
                      .format(theta_parameters))

            # Buffer size of sure blocking, which is used to define the buffer size for activation and the start state for the simulations run
            # We set the activation buffer size as a specific fraction of the buffer size K of sure blocking.
            # This fraction is chosen as the optimum found theoretically to have the same complexity in reaching
            # a signal that leads to estimations of the numerator Phi(t) and the denominator E(T1+T2) in the
            # FV estimation of the stationary probability (found on Fri, 07-Jan-2022 at IRIT-N7 with Matt, Urtzi and
            # Szymon), namely J = K/3.
            if self.env.getBufferType() == BufferType.SINGLE:
                # Compute the true blocking probabilities, so that we know how much error we have in the estimation of Pr(K-1) and Pr(K)
                Ks = get_deterministic_blocking_boundaries(self.agent, return_int_when_single_policy=True)  # The second parameter is because we want the returned blocking size to be a scalar
                assert is_scalar(Ks)
                probas_stationary_true = dict({Ks-1: compute_blocking_probability_birth_death_process(self.rhos, Ks-1),
                                               Ks:   compute_blocking_probability_birth_death_process(self.rhos, Ks)})
            else:
                Ks = get_deterministic_blocking_boundaries(self.agent, return_int_when_single_policy=False) # The second parameter is because we want the returned blocking size to be ALWAYS a list
                # True stationary probability of a loss network system for ALL possible states
                # (further down, we will use the true probabilities that are actually checked, i.e. for those states for which an estimated probability is computed)
                env_effective_capacity, x, dist = compute_stationary_probability_knapsack_when_blocking_by_class(env_original_capacity, self.rhos, Ks)
                probas_stationary_true = dict([[tuple(xx), dd] for xx, dd in zip(x, dist)])

                # *** IMPORTANT ***
                # Set the capacity of each environment used for the simulation to the EFFECTIVE capacity computed above
                # so that stationary probabilities computed below by estimate_blocking_fv() or estimate_blocking_mc()
                # are computed ONLY on the valid states when the environment's capacity may have been potentially limited by
                # the blocking sizes defined by the current policy (given in variable Ks),
                # e.g. when capacity = 10 and blocking sizes are Ks = [2, 1, 3] => the effective capacity is 2 + 1 + 3 = 6
                # which means that for example the state = (4, 0, 0), although valid in the original environment with capacity 10
                # is NOT valid in the environment limited by the blocking size values Ks because state[0] = 4 > 2 = Ks[0].
                for env in self.envs:
                    env.setCapacity(env_effective_capacity)
                # *** IMPORTANT ***

            if self.env.getBufferType() == BufferType.SINGLE:
                dict_params_simul['buffer_size_activation'] = np.max([1, int( np.round(dict_params_simul['buffer_size_activation_factor']*Ks) )])
            else:
                Js = [max(1, int(np.round(bsaf * k))) for bsaf, k in zip(dict_params_simul['buffer_size_activation_factor'], Ks)]
                dict_params_simul['absorption_set'] = SetOfStates(set_boundaries=tuple([J - 1 for J in Js]))
                dict_params_simul['activation_set'] = SetOfStates(set_boundaries=tuple(Js))
            if isinstance(self.getLearnerV(), LeaFV):
                # TODO: (2021/12/16) SHOULD WE MOVE ALL this learning process of the average reward and stationary probabilities to the learn() method of the LeaFV class?
                # At this point I don't think we should do this, because the learn() method in the GenericLearner subclasses
                # learns the value function from the RESULTS of a SIMULATION ALREADY CARRIED OUT.
                # (see for instance the LeaMC.learn() method)
                # If we do the above change, we should ALSO change the learn() method of LeaMC to perform the simulation
                # of the system itself... and I don't think the learn() method should be responsible for the simulation
                # because simulating is a quite computing intensive task!!

                if not is_scalar(dict_params_simul['t_sim']):
                    # There is a different simulation time (either #steps for MC or #arrivals for FVRL) for each learning step
                    # => Read the value of the t_sim parameter from the learning step currently being processed
                    assert len(dict_params_simul) == self.dict_params_learning['t_learn']
                    dict_params_simul['T'] = dict_params_simul['t_sim'][t_learn-1]   # `t_learn-1` because t_learn starts at 1, not 0.

                proba_blocking_fv, expected_reward, probas_stationary_blocking, \
                    expected_cycle_time, n_cycles, time_last_absorption, time_end_simulation_et, max_survival_time, time_end_simulation_fv, \
                        n_events_mc, n_events_fv = estimate_blocking_fv(self.envs, self.agent, dict_params_simul, dict_params_info,
                                                        probas_stationary=probas_stationary_true if dict_params_simul.get('use_stationary_probability_for_start_states', False) else None)
                print("Estimated stationary probabilities of states of interest:\n{}".format(probas_stationary_blocking))
                # Now we compute the equivalent (discrete) simulation time, as if it were a Monte-Carlo simulation
                # Note that this measures the # simulation steps which does NOT coincide with parameter 't_sim'
                # (equal to 'T') which in FVRL defines the # *arrival* events.
                # This piece of information is needed in order to run an MC learner that is comparable with the FVRL
                # learner in that the MC learner uses the same number of simulation steps for each learning step as
                # those used by FVRL at the respective learning step, OR the MC learner uses the same *total* number
                # of simulation steps as those used by FVRL in the whole learning process, divided equally into each
                # learning step.
                t = n_events_mc + n_events_fv
                self.getLearnerV().setAverageReward(expected_reward)
                self.getLearnerV().setProbasStationary(probas_stationary_blocking)
            else:
                # Monte-Carlo learning is the default
                assert self.N == 1, "The simulation system has only one particle in Monte-Carlo mode ({})".format(self.N)

                if not is_scalar(dict_params_simul['t_sim']):
                    # There is a different simulation time for each learning step
                    # => Use a number of simulation steps defined in a benchmark for each learning step
                    assert len(dict_params_simul['t_sim']) == self.dict_params_learning['t_learn'], \
                            "The number of simulation steps read from the benchmark file ({}) coincides with the number of learning steps ({})" \
                            .format(len(dict_params_simul['t_sim']), self.dict_params_learning['t_learn'])
                    dict_params_simul['T'] = dict_params_simul['t_sim'][t_learn-1]     # `t_learn-1` because t_learn starts at 1, not 0.
                proba_blocking_mc, expected_reward, probas_stationary_blocking, n_cycles_proba, \
                    expected_cycle_time, n_cycles, time_last_event, n_events = estimate_blocking_mc(self.env, self.agent, dict_params_simul, dict_params_info)
                    ## IMPORTANT (2022/10/24) The expected_cycle_time is NOT equal to the denominator used to compute the stationary probabilities
                    ## stored in probas_stationary_blocking when a positive number of burn-in time steps at the beginning of the simulation
                    ## before starting to use cycle times for estimations is considered.
                    ## In fact, in this case, as stated in function estimate_blocking_mc(), we have the following differences
                    ## between the two calculations:
                    ## - the expected_cycle_time is estimated from return cycles to the initial position of the Markov chain,
                    ## as usual. This is true regardless of the number of burn-in time steps.
                    ## - on the other hand, the denominator used in the *stationary probability* estimator is computed from
                    ## return cycles to the FIRST POSITION (buffer size) at which *the Markov chain is found after the burn-in period*,
                    ## which may or may not be equal to the starting position of the Markov chain.
                # Discrete simulation time
                t = n_events
                n_events_mc = n_events
                n_events_fv = 0         # This information is output to the results file and therefore we set it to 0 here as we are in the MC approach
                max_survival_time = 0.0 # Ditto

            if self.env.getBufferType() == BufferType.SINGLE:
                if True or DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                    print("\n--> Estimated stationary probability: Pr(K-1={}) = {} vs. True Pr(K-1={}) = {}, error = {:.3f}%" \
                          .format(Ks-1, probas_stationary_blocking[Ks-1], Ks-1, probas_stationary_true[Ks-1], (probas_stationary_blocking[Ks-1] / probas_stationary_true[Ks-1] - 1) * 100))
                    print("--> Estimated stationary probability: Pr(K={}) = {} vs. True Pr(K={}) = {}, error = {:.3f}%" \
                        .format(Ks, probas_stationary_blocking[Ks], Ks, probas_stationary_true[Ks-1], (probas_stationary_blocking[Ks] / probas_stationary_true[Ks] - 1) * 100))
                # Compute the state-action values (Q(s,a)) for buffer size = Ks-1
                if probas_stationary_blocking.get(Ks-1, 0.0) == 0.0 or np.isnan(probas_stationary_blocking.get(Ks-1, 0.0)):
                    Q0_Km1 = 0.0
                    Q1_Km1 = 0.0
                    n_Km1 = 0
                    if True or DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                        print("Estimation of Q_diff(K-1) skipped because the stationary probability Pr(K-1) is {}".format(probas_stationary_blocking.get(Ks-1)))
                    # Set Pr(K-1) to 0 in case it was not computed, so that the policy leaning method called below (`learn()`)
                    # receives the input parameters as expected (e.g. probas_stationary_blocking should have 2 keys: K-1 and K)
                    probas_stationary_blocking[Ks-1] = 0.0
                else:
                    # When the parameterized policy is a linear step function with only one buffer size where its
                    # derivative is non-zero, only the DIFFERENCE of two state-action values impacts the gradient, namely:
                    # Q(K-1, a=1) - Q(K-1, a=0)
                    if DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                        print("\nEstimating the difference of the state-action values when the initial buffer size is K-1={}...".format(Ks-1))
                    N = 100
                    t_sim_max = 250
                    Q0_Km1, Q1_Km1, n_Km1, max_t_Km1 = \
                        self.estimate_Q_values_until_mixing(t_learn, tuple([Ks-1]), t_sim_max=t_sim_max, N=N,
                                                            seed=dict_params_simul['seed'] * 100, verbose=verbose,
                                                            verbose_period=verbose_period)
                    if True or DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                        print("--> Estimated state-action values on n={} realizations out of {} with max simulation time = {:.1f} out of {:.1f}:\nQ(K-1={}, a=1) = {}\nQ(K-1={}, a=0) = {}\nQ_diff = Q(K-1,1) - Q(K-1,0) = {}" \
                            .format(n_Km1, N, max_t_Km1, t_sim_max, Ks-1, Q1_Km1, Ks-1, Q0_Km1, Q1_Km1 - Q0_Km1))

                # Compute the state-action values (Q(s,a)) for buffer size = K
                # This is needed ONLY when the policy learning methodology is IGA (Integer Gradient Ascent, presented in Massaro's paper (2019)
                if self.dict_params_learning['mode'] != LearningMode.IGA or probas_stationary_blocking.get(Ks, 0.0) == 0.0 or np.isnan(probas_stationary_blocking.get(Ks, 0.0)):
                    # We set all the values to 0 so that the learn() method called below doesn't fail, as it assumes that
                    # the values possibly involved with learning (which depend on the learning method) have a real value.
                    Q0_K = 0.0
                    Q1_K = 0.0
                    n_K = 0
                    if DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                        print("Estimation of Q_diff(K) skipped because the stationary probability Pr(K) is {}".format(probas_stationary_blocking.get(Ks-1)))
                    # Set Pr(K) to 0 in case it was not computed, so that the policy learning method called below (`learn()`)
                    # receives the input parameters as expected (e.g. probas_stationary_blocking should have 2 keys: K-1 and K)
                    probas_stationary_blocking[Ks] = 0.0
                elif False: # (2023/02/21) I decided NOT to enter the below block because we don't need the Qdiff for buffer size = K, and the calculation takes time!
                    # Same as above, but for buffer size = K
                    if DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                        print("\nEstimating the difference of the state-action values when the initial buffer size is K={}...".format(Ks))
                    N = 100
                    t_sim_max = 250
                    Q0_K, Q1_K, n_K, max_t_K = \
                        self.estimate_Q_values_until_mixing(t_learn, tuple([Ks]), t_sim_max=t_sim_max, N=N,
                                                            seed=dict_params_simul['seed'] * 100 + 1, verbose=verbose,
                                                            verbose_period=verbose_period)
                    if show_messages(verbose, verbose_period, t_learn):
                        print("--> Estimated state-action values on n={} realizations out of {} with max simulation time = {:.1f} out of {:.1f}:\nQ(K={}, a=1) = {}\nQ(K={}, a=0) = {}\nQ_diff = Q(K,1) - Q(K,0) = {}" \
                            .format(n_K, N, max_t_K, t_sim_max, Ks, Q1_K, Ks, Q0_K, Q1_K - Q0_K))
                else:
                    # Set the values that are used below when storing the results to NaN or 0
                    Q0_K, Q1_K, n_K, max_t_K = np.nan, np.nan, 0, 0

            else:
                if probas_stationary_true is not None:
                    assert set(probas_stationary_blocking.keys()).issubset(set(probas_stationary_true.keys())),\
                        "The set of states in the probas_stationary_blocking dictionary with the estimated stationary probabilities " \
                        "must be a subset of the set of states in the probas_stationary_true dictionary with the true stationary probabilities:" \
                        f"\nTrue ({len(probas_stationary_true.keys())} keys):\n{sorted(probas_stationary_true.keys())}" \
                        f"\nEstimated ({len(probas_stationary_blocking.keys())} keys):\n{sorted(probas_stationary_blocking.keys())}"

                # Compute the difference in Q's for EACH state whose stationary probability was estimated by the FV process
                Q_values = dict()
                n_replications_with_mixing = dict()   # Number of replications out of N run where trajectories mix
                for x, p in probas_stationary_blocking.items():
                    if True or DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                        print("\n--> Estimated stationary probability of blocking state: Pr(x={}) = {} vs. True Pr(x={}) = {}, error = {:.3f}%" \
                            .format(x, probas_stationary_blocking[x], x, probas_stationary_true[x], (probas_stationary_blocking[x] / probas_stationary_true[x] - 1) * 100))
                    ### TEMPORARY ###
                    if p > 0.0:
                    #if probas_stationary_true[x] > 0:
                    ### TEMPORARY ###
                        if DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                            print("\nEstimating the difference of the state-action values when the initial state is x={}...".format(x))
                        # Find the component(s) of state x on which the Q value difference should be computed,
                        # (i.e. by considering an ACCEPT and REJECT actions on an arriving job of the class corresponding to that component),
                        # namely ALL components i whose occupancy in the system is Ks(i) - 1, for which the derivative of the linear-step acceptance policy is not zero.
                        # NOTE that, for a given x, there may be more than one component satisfying this condition,
                        # which happens when more than one component has its value at the blocking value - 1, e.g. x = (0, 3, 5) and the blocking values are Ks = [4, 3, 5]
                        for dim_blocking, K in enumerate(Ks):
                            if x[dim_blocking] == K - 1 and np.sum(x) < self.env.getCapacity():
                                ## NOTE that we only compute Qdiff when x is NOT at the full capacity of the system (np.sum(x) < self.env.getCapacity())
                                ## because if this is the case the probability of rejecting an incoming job is 1,
                                ## hence the derivative of the acceptance policy is 0 and thus no Qdiff needs to be computed.
                                N = 100
                                t_sim_max = 250
                                Q0, Q1, n, max_t = \
                                    self.estimate_Q_values_until_mixing(t_learn, x, dim_blocking=dim_blocking,
                                                                        t_sim_max=t_sim_max, N=N,
                                                                        seed=dict_params_simul['seed'] * 100, verbose=verbose,
                                                                        verbose_period=verbose_period)
                                if True or DEBUG_ESTIMATORS or show_messages(verbose, verbose_period, t_learn):
                                    print("--> Estimated state-action values for blocking job of class {} at K-1={} on n={} realizations out of {} with max simulation time = {:.1f} out of {:.1f}:"
                                          "\nQ(x={}, a=1) = {}\nQ(x={}, a=0) = {}\nQ_diff = Q(x,1) - Q(x,0) = {}" \
                                            .format(dim_blocking, K-1, n, N, max_t, t_sim_max, x, Q1, x, Q0, Q1 - Q0))

                                Q_values[x] = [Q0, Q1]
                                n_replications_with_mixing[x] = n

            # Learn the value function and the policy
            theta_prev = self.getLearnerP().getThetaParameter()

            # Use this when using REINFORCE to learn theta: LearningMode.REINFORCE_RETURN
            #self.learn(self.agent, t)
            # Use this when estimating the theoretical grad(V): LearningMode.REINFORCE_TRUE or LearningMode.IGA
            if self.env.getBufferType() == BufferType.SINGLE:
                err_phi, err_et = compute_rel_errors_for_fv_process(self.rhos, Ks,
                                                                    dict_params_simul['buffer_size_activation_factor'],
                                                                    self.N, dict_params_simul['T'])
                self.learn(self.agent, t,
                           probas_stationary=probas_stationary_blocking,
                           probas_stationary_true=probas_stationary_true,
                           Q_values=dict({Ks-1: [Q0_Km1, Q1_Km1], Ks: [Q0_K, Q1_K]}),
                           simul_info=dict({'t_learn': t_learn,
                                            'K': Ks,
                                            'J_factor': dict_params_simul['buffer_size_activation_factor'],
                                            'J': dict_params_simul['buffer_size_activation'],
                                            'burnin_time_steps': dict_params_simul['burnin_time_steps'],
                                            'min_num_cycles_for_expectations': dict_params_simul['min_num_cycles_for_expectations'],
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
                                            'n_Q': np.mean([n_Km1, n_K]) if self.dict_params_learning['mode'] == LearningMode.IGA else n_Km1
                                            })
                           )

                # Update the value of N and T for the next learning step if the desired expected relative errors are given as part of the input parameters
                # Goal: Keep the same expected errors in Phi and in E(T_A) along the simulation process, for all blocking sizes K
                if isinstance(self.getLearnerV(), LeaFV) and \
                        'error_rel_phi' in dict_info.keys() and 'error_rel_et' in dict_info.keys():
                    # Compute the new blocking size so that we can update the values of N and T
                    _K_next = get_deterministic_blocking_boundaries(self.agent)
                    _N, _T = \
                        compute_nparticles_and_narrivals_for_fv_process(self.rhos, _K_next, dict_params_simul[
                            'buffer_size_activation_factor'],
                                                                        error_rel_phi=dict_info['error_rel_phi'],
                                                                        error_rel_et=dict_info['error_rel_et'])
                    _N_new = min(max(dict_info.get('N_min', 0), _N), dict_info.get('N_max', np.Inf))
                    _T_new = min(max(dict_info.get('T_min', 0), _T), dict_info.get('T_max', np.Inf))
                    if True or show_messages(verbose, verbose_period, t_learn):
                        print("New blocking size K = {}".format(_K_next))
                        print("\tN updated from {} --> {}".format(dict_params_simul['nparticles'], _N_new))
                        print("\tT updated from {} --> {}".format(dict_params_simul['T'], _T_new))
                    dict_params_simul['nparticles'] = _N_new
                    dict_params_simul['T'] = _T_new

                    # Set the number of particles in the object
                    # Note that for the updated number of arrival events to be used in the next iteration,
                    # it suffices to update dict_params_simul['T'] which is what we just did.
                    self.setNumberParticlesAndCreateEnvironments(dict_params_simul['nparticles'])
                    assert self.N == dict_params_simul['nparticles']
            else:
                # In the multidimensional state case, we don't know at this point how to compute the errors of Phi and E(T_A)
                err_phi, err_et = np.nan, np.nan
                self.learn(self.agent, t,
                           ### TEMPORARY ###
                           probas_stationary=probas_stationary_blocking,
                           #probas_stationary=dict([(x, probas_stationary_true[x]) for x in probas_stationary_blocking]),
                           ### TEMPORARY ###
                           probas_stationary_true=probas_stationary_true,
                           Q_values=Q_values,
                           simul_info=dict({'t_learn': t_learn,
                                            'theta_true': dict_params_simul['theta_true'],
                                            'K': Ks,
                                            'J_factor': dict_params_simul['buffer_size_activation_factor'],
                                            'J': Js,
                                            'burnin_time_steps': dict_params_simul['burnin_time_steps'],
                                            'min_num_cycles_for_expectations': dict_params_simul['min_num_cycles_for_expectations'],
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
                                            'n_Q': np.mean(list(n_replications_with_mixing.values()))
                                                ## n_Q stores the average sample size among all states x on which the Q_diff has been computed
                                            })
                           )

            if True and show_messages(verbose, verbose_period, t_learn):
                print("\tUpdated value function at the end of the queue simulation: average reward V = {}".format(self.getLearnerV().getV()))
                print("\tSame observed average reward (computed from Policy learner) = {}".format(self.getLearnerP().getAverageRewardUnderPolicy()))
                print("\tUpdated theta parameter of policy after learning: theta = {} -> {}".format(theta_prev, self.getLearnerP().getThetaParameter()))

        if dict_params_info['plot']:
            # TODO
            pass

        # Store the results of the learning process
        df_learning = pd.DataFrame.from_dict(OrderedDict({  'theta': self.thetas,
                                                            'theta_next': self.thetas_updated,
                                                            'gradV': self.gradV,
                                                            'alpha': self.alphas,
                                                            'state_coverage': self.state_coverage,
                                                            'proba_blocking': self.proba_blocking,
                                                            'proba_blocking_true': self.proba_blocking_true,
                                                            'expected_reward': self.expected_reward,
                                                            'expected_reward_true': self.expected_reward_true,
                                                            'error_proba_blocking': self.error_proba_blocking,
                                                            'error_reward': self.error_expected_reward,
                                                            'V': self.V,
                                                            'n_events_mc': self.n_events_mc,
                                                            'n_events_fv': self.n_events_fv,
                                                            'expected_cycle_time': expected_cycle_time,
                                                            'n_cycles': n_cycles,
                                                            'max_survival_time': max_survival_time,
                                                            'n_trajectories_Q': self.n_trajectories_Q,
                                                            'seed': dict_params_simul['seed'],
                                                        }))

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

        start_state: Queue-environment-dependent
            (Environment) state at which the queue environment starts for the simulation.
            Its type depends on how the queue environment defines the state of the system.
            Ex: when the queue is of class GenericEnvQueueWithJobClasses, the state is a tuple with the following information:
            (queue_state, job_class) = (state of the queue itself, class of the arriving job).

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
        t_max = t_sim_max

        # Set the start state of the environment to the given start state
        self.env.setState(start_state)
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
                update_trajectory(agent, (t_learn - 1) * (t_max + 1) + t, t, state, action, reward)

                done = check_done(t_max, t, state, action, reward)
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
                    update_trajectory(agent, (t_learn - 1) * (t_max + 1) + t, t, state, action, reward)

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
        K = get_deterministic_blocking_boundaries(self.agent)
        probas_stationary_blocking = dict({K-1:  np.sum([1 for bs in buffer_sizes if bs == K-1]) / len(buffer_sizes),
                                  K:    np.sum([1 for bs in buffer_sizes if bs == K]) / len(buffer_sizes)})

        return probas_stationary_blocking

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
        # TODO: (2023/01/28) Find out how to check the type of the policy because the lines below do not work. The returned value of type() is something like '<type>' so it does NOT make sense!!
        #if not isinstance(self.getLearnerP().getPolicyClass(), PolQueueTwoActionsLinearStep):
        #    raise ValueError("The agent's policy must be of type {} ({})" \
        #                     .format(type(PolQueueTwoActionsLinearStep), type(self.getLearnerP().getPolicyClass())))

        K = get_deterministic_blocking_boundaries(self.agent)

        # 1) Starting at (K-1, a=1) is like starting at s=K (because the reward of the first action is 0)
        if show_messages(verbose, verbose_period, t_learn):
            print("\n1) Running simulation on N={} queues for {} arrival events to estimate Q(K-1, a=1)...".format(N, t_sim_max))
        start_queue_state = choose_state_for_buffer_size(self.env, K)
        start_state = (start_queue_state, None)
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
            print("2) Running simulation on N={} queues for {} arrival events to estimate Q(K-1, a=0)...".format(N, t_sim_max))
        start_queue_state = choose_state_for_buffer_size(self.env, K-1)
        start_state = (start_queue_state, None)
        # Reward received from the initial rejection of the job at buffer size K-1
        first_reward = compute_reward_for_rejection_at_current_state(self.env, state=(K-1, 0))
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
    def estimate_Q_values_until_mixing(self, t_learn, state_or_buffer_size: tuple, dim_blocking=0, t_sim_max=250, N=100, seed=None, verbose=False, verbose_period=1):
        """
        Estimates the state-action values for the case of a parameterized policy defined as a linear function of theta
        with only one buffer size with non-zero derivative.
        The state-action values are computed for the two possible accept/reject actions of an incoming job when
        the queue system is at that buffer size.

        Arguments:
        t_learn: int
            Learning time step.

        state_or_buffer_size: tuple
            State or buffer size s at which the system starts when estimating Q(s, action).

        dim_blocking: int
            Index of the dimension of state_or_buffer_size on which blocking is analyzed when computing the Q value difference.
            default: 0, which is used in single-buffer queue environments or when there is only one arriving job class

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
        - The state-action value for (state_or_buffer_size, a=Reject)
        - The state-action value for (state_or_buffer_size, a=Accept)
        - n: the number of realizations, where trajectories mix, that are used to compute the state-action values
        - Maximum discrete time at which the pair of trajectories mix over the n pair of trajectories that mix
        """
        # TODO: (2023/01/28) Find out how to check the type of the policy because the lines below do not work. The returned value of type() is something like '<type>' so it does NOT make sense!!
        #if not isinstance(self.getLearnerP().getPolicyClass(), PolQueueTwoActionsLinearStep):
        #    raise ValueError("The agent's policy must be of type {} ({})" \
        #                     .format(PolQueueTwoActionsLinearStep.__class__, type(self.getLearnerP().getPolicyClass())))

        np.random.seed(seed)

        Q0 = 0.0
        Q1 = 0.0
        n = 0           # Number of times Q0 and Q1 can be estimated from mixing trajectories (out of N)
        max_t_mix = 0.0 # Maximum discrete mixing time observed over the n pair of trajectories that mix
        if True:
            print("Estimating Qdiff for state or buffer size s={} at blocking dimension {}... using {} replications".format(state_or_buffer_size, dim_blocking, N))
        for i in range(N):
            if False:
                print(f"--- Replication {i+1} of {N} ---")
            _, Q0_, Q1_, t_mix = self.run_simulation_2_until_mixing(t_learn, state_or_buffer_size, dim_blocking, t_sim_max,
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

        return Q0, Q1, n, max_t_mix

    def run_simulation_2_until_mixing(self, t_learn, state_or_buffer_size: tuple, dim_blocking: int, t_sim_max: int, verbose=False, verbose_period=1):
        """
        Runs the two (2) simulations of the queue starting at the given buffer size, one with first action Reject,
        and the other with first action Accept, until the two trajectories mix or the given maximum simulation time
        is achieved.

        Arguments:
        t_learn: int
            The learning time step to which the simulation will contribute.

        state_or_buffer_size: tuple
            State or buffer size at which the two simulations start.
            It should be associated to a *non-deterministic* acceptance policy, i.e. where the probability
            of acceptance p is 0 < p < 1.

        dim_blocking: int
            Index of the dimension of state_or_buffer_size on which blocking is analyzed when computing the Q value difference.

        t_sim_max: int
            Maximum discrete simulation time allowed for the simulation. This is defined ONLY by job arrivals,
            NOT by service events, as decisions are only taken when a job arrives.

        verbose: (opt) bool
            Whether to be verbose in the simulation process.
            default: False

        verbose_period: (opt) int
            The time step period to be verbose.
            default: 1 => be verbose at every simulation step.

        Return: tuple
        Triple with the following elements:
        - t: the last time step of the simulation process.
        - Q0: the sum of the rewards observed until mixing starting at state-action (state_or_buffer_size, a=Reject on dimension dim_blocking)
        - Q1: the sum of the rewards observed until mixing starting at state-action (state_or_buffer_size, a=Accept on dimension dim_blocking)
        - mixing time: discrete time at which the two trajectories mix. This is np.nan if they don't mix.
        """

        # ------------------------------ Auxiliary functions ----------------------------------#
        def apply_event(env, t, event, job_class_or_server):
            """
            Apply the given event, either a BIRTH with the incoming job class, or a DEATH event at the given server,
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

                # Assign the job to a server if accepted for queues having a buffer to hold jobs
                # (i.e. loss networks are excluded from the job assignment part as it is irrelevant to which server the job is assigned)
                if action_accept_reject == Actions.ACCEPT and env.getBufferType() != BufferType.NOBUFFER:
                    # Assign the job just accepted to a server and update the next_state value
                    # (because the next_state from the ACCEPT action has not been updated above
                    # as we need to know to which server the accepted job is assigned in order
                    # to know the next state of the queue environment)
                    _, next_state, reward_assign, _ = step(t, env, self.agent, PolicyTypes.ASSIGN)
                else:
                    reward_assign = 0.0

                action = action_accept_reject
                reward = reward_accept_reject + reward_assign
                if action == Actions.ACCEPT:
                    assert env.getBufferSizeFromState(next_state) == env.getBufferSizeFromState(state) + 1, \
                        "The buffer size after a BIRTH increased by 1: BS(t) = {}, BS(t+1) = {}" \
                            .format(env.getBufferSizeFromState(state), env.getBufferSizeFromState(next_state))
                    assert reward == 0.0, "A non-zero reward is not possible when the action is ACCEPT.\n" \
                                          "--> action = {}, REWARD ASSIGN: {}, REWARD = {}".format(action_accept_reject, reward_assign, reward)
            else:
                assert event == Event.DEATH
                # The event is a completed service
                # => Update the state of the queue but do not increase t
                server = job_class_or_server
                queue_state = env.getQueueState()
                assert queue_state[server] > 0, "The server where the completed service occurs has at least one job"
                next_queue_state = list(copy.deepcopy(queue_state))
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
        # This is state_or_buffer_size when the first action is Reject and state_or_buffer_size increased by +1
        # in the `dim_blocking` dimension when the first action is Accept.
        _state_or_buffer_size_start_reject = state_or_buffer_size
        _state_or_buffer_size_start_accept = tuple([bs + 1 if i == dim_blocking else bs for i, bs in enumerate(state_or_buffer_size)])
        # Environment that starts at REJECTION. Note that the job class is set to `dim_blocking`
        # because it CANNOT be None since we need the job class to compute below the first reward of rejecting the job.
        envs[0].setState((_state_or_buffer_size_start_reject, dim_blocking))
        # Environment that starts at ACCEPTANCE. The job class could be set to None in this case (as no reward is
        # computed because the reward of accepting is assumed to be 0.0 (although we could at some point use
        # a reward function for acceptance --such as rewardOnJobClassAcceptance() in environments/queues.py--
        # so we also set it to `dim_blocking`).
        envs[1].setState((_state_or_buffer_size_start_accept, dim_blocking))

        # History of states, actions and rewards of each environment (2 x <undef> list)
        # Used for informational purposes, but actually we don't need to store the history because we are computing
        # Q0 and Q1 on the fly. But still I store the history just in case we need it for some debugging purposes for instance.
        # NOTE that only the BIRTH events are recorded, therefore the state values may show jumps that are larger than -1 or +1.
        times = [[0.0],
                 [0.0]]
        states = [[envs[0].getState()],
                  [envs[1].getState()]]
        actions = [[Actions.REJECT],
                   [Actions.ACCEPT]]
        rewards = [[compute_reward_for_rejection_at_current_state(envs[0])],
                   [0.0]]

        # Time step in the queue trajectory (the first time step is t = 0)
        done = False
        t = 0                   # We start at 0 because we have already stored the state for t=0 above
        t0 = 0; t1 = 0;         # We keep track of the discrete-time step of EACH queue which increases by +1
                                # ONLY when a new job arrives to the respective queue
                                # (as this is the only moment when an action is applied by the MDP,
                                # which is either accept or reject the incoming job).
                                # Thus, we can compare the states of each of the two queues simulated
                                # (the one starting with action a=0 and the one starting with action a=1)
                                # at the same discrete time value, and check if the two queues mixed.
        maxtime = t_sim_max     # Max discrete time, in case there is no mixing of trajectories
        # Initialize the state-action values with the first observed reward (at t=0)
        Q0 = rewards[0][0]
        Q1 = rewards[1][0]
        if False:
            print("--> First reward Q(s={},a=1 at state dimension {})={:.1f}, Q(s={},a=0 at state dimension {})={:.1f}" \
                  .format(state_or_buffer_size, dim_blocking, Q1, state_or_buffer_size, dim_blocking, Q0))
        while not done:
            # Generate events for each queue environment until the respective discrete time changes
            # (i.e. until a new job arrives),
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
                    .format(event1, action1, next_state1, event0, action0, next_state0, state_or_buffer_size, Q1, state_or_buffer_size, Q0))

            # Check if the trajectories meet at the same state-action
            if t > maxtime or mixing(next_state0, action0, next_state1, action1):
                # Either we reached the maximum simulation time allowed or mixing occurs
                # => Stop the simulation
                if False:
                    print("*** TRAJECTORIES MIXED AT t={}! (state0={}, state1={}; Q1={:.1f}, Q0={:.1f}, Qdiff = {:.1f}) ***".format(t, states[0][-1], states[1][-1], Q1, Q0, Q1 - Q0))
                done = True

            if self.debug:
                # Note that only the states at birth times are recorded, therefore the occupancy values may NOT always change by -1 or +1!
                print("t={}: states = [{}, {}] events={}, job_classes={}, actions={} -> states={} rewards={}" \
                      .format(t, states[0][-2], states[1][-2], [event0.name, event1.name], [job_class_or_server0, job_class_or_server1], [action0.name, action1.name], [next_state0, next_state1], [reward0, reward1]), end="\n")

        # DONE
        if show_messages(verbose, verbose_period, t_learn):
            print("==> simulation ENDS at discrete time t={} at states = {} coming from states = {}, actions = {} rewards = {})" \
                    .format(t, [next_state0, next_state1], [states[0][-2], states[1][-2]], [action0, action1], [reward0, reward1]))

#        if show_messages(verbose, verbose_period, t_learn):
#           plt.figure()
#           print("Mixing time t={}, Q0={}, Q1={}".format(t, Q0, Q1))
#           print(np.c_[[envs[0].getBufferSizeFromState(s) for x in states[0]], actions[0], rewards[0],
#                        [envs[1].getBufferSizeFromState(s) for x in states[0]], actions[1], rewards[1]])
#           plt.plot([envs[0].getBufferSizeFromState(s) for x in states[0]], 'b.-')
#           plt.plot([envs[1].getBufferSizeFromState(s) for x in states[1]], 'r.-')
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
    def learn(self, agent, t, probas_stationary: dict=None, Q_values: dict=None, probas_stationary_true: dict=None, simul_info: dict=None):
        """
        Learns the value functions and the policy at time t

        Arguments:
        agent: Agent
            Agent that is responsible of performing the actions on the environment and learning from them.

        t: int
            Time (iteration) in the queue simulation time scale at which learning takes place.
            In Monte-Carlo learning, this time would be the end of the queue simulation.

        probas_stationary: (opt) dict of floats between 0 and 1 (used when estimating the theoretical grad(V))
            Dictionary with the estimated stationary probability for each state of interest for learning defined in its keys.
            For a single-buffer system the dictionary keys are expected to be K-1 and K.
            default: None

        Q_values: (opt) dict of lists or numpy arrays (used when estimating the theoretical grad(V))
            Dictionary with the estimated state-action values for each state of interest for learning defined in its keys.
            Each element in the list is indexed by the initial action taken when estimating the corresponding Q value.
            For a single-buffer system the dictionary keys are expected to be K-1 and K.
            default: None

        probas_stationary_true: (opt) dict of floats between 0 and 1
            Dictionary with the estimated stationary probability for each state of interest for learning
            (e.g. states with non-deterministic blocking) defined in its keys.
            For a single-buffer system, the dictionary keys are expected to be K-1 and K.
            default: None

        simul_info: (opt) dict
            Dictionary containing relevant information (for keeping a record of) about the simulations run to achieve
            the estimated stationary probabilities and the Q values.
            The dictionary doesn't need to contain any keys, but the following keys are usually of interest:
            - 'n_events_mc': # events in the MC simulation that estimates the stationary probabilities or that estimates
            P(T>t) and E(T_A) in the FV approach.
            - 'n_events_fv': # events in the FV simulation that estimates Phi(t) for the buffer sizes of interest.
            - 'n_Q': average number of trajectory pairs used to estimate each of the Q values given in the Q_values dict.
            default: None
        """
        # TODO: (2021/11/03) Think if we can separate the learning of the value function and return G(t) from the learning of the policy (theta)
        # Goal: avoid computing G(t) twice, one at learnerV.learn() and one at learnerP.learn()

        assert 'mode' in self.dict_params_learning.keys() and self.dict_params_learning['mode'] in LearningMode

        # Store the alpha used for the current learning step
        agent.getLearnerP().store_learning_rate()

        # Store a few quantities in the history of observed quantities
        self.alphas += [agent.getLearnerP().getLearningRate()]
        self.proba_stationary += [probas_stationary]
        n_states_blocking = len(probas_stationary)

        # Compute the blocking probability and expected reward which are the ones that are reported in the output file
        # (we don't store the stationary probabilities because there may be too many blocking states! --e.g. in a multidimensional state setting)
        # IMPORTANT: We should compute these quantities BEFORE updating theta because they must correspond to the policy
        # parameterized by theta, NOT by the updated theta after learning.
        proba_blocking = compute_proba_blocking(self.env, agent, probas_stationary)
        expected_reward = estimate_expected_reward(self.env, agent, probas_stationary)
        if probas_stationary_true is not None:
            proba_blocking_true = compute_proba_blocking(self.env, agent, probas_stationary_true)
            expected_reward_true = estimate_expected_reward(self.env, agent, probas_stationary_true)
            error_proba_blocking = proba_blocking / proba_blocking_true - 1
            error_expected_reward = expected_reward / abs(expected_reward_true) - 1 if expected_reward_true != 0.0 else 0.0 if expected_reward == 0.0 else np.nan
        else:
            proba_blocking_true = np.nan
            expected_reward_true = np.nan
            error_proba_blocking = np.nan
            error_expected_reward = np.nan
        self.proba_blocking += [proba_blocking]
        self.proba_blocking_true += [proba_blocking_true]
        self.expected_reward += [expected_reward]
        self.expected_reward_true += [expected_reward_true]
        self.error_proba_blocking += [error_proba_blocking]
        self.error_expected_reward += [error_expected_reward]

        # Compute the error of the stationary probability estimations (which is stored in the object attributes storing the learning history)
        if probas_stationary is None or probas_stationary_true is None:
            self.error_proba_stationary += [dict([(x, np.nan) for x in probas_stationary])]
        else:
            self.error_proba_stationary += [dict([(x, probas_stationary[x] / probas_stationary_true[x] - 1) for x in probas_stationary])]

        # Learn
        if self.dict_params_learning['mode'] == LearningMode.REINFORCE_RETURN:
            agent.getLearnerV().learn(t)  # UNCOMMENTED ON SAT, 27-NOV-2021
            theta_prev, theta, coverage, V, gradV, G = agent.getLearnerP().learn_update_theta_at_episode_end(t)
            #theta_prev, theta, coverage, V, gradV, G = agent.getLearnerP().learn_update_theta_at_each_time_step(t)
        else:
            # When learning via the theoretical grad(V) expression, we don't need to estimate the average reward!
            # (because grad(V) is the difference between two state-action values, and the average reward affects both the same way)
            # This is the option to use when learning via Fleming-Viot because in that case the return G(t) is NOT meaningful
            # because, either it is empty as the rewards are not recorded while running the Fleming-Viot process,
            # or it does not reflect the actual values of each state-value of the original system.
            assert probas_stationary is not None and Q_values is not None
            K = get_deterministic_blocking_boundaries(agent)
            if self.dict_params_learning['mode'] == LearningMode.REINFORCE_TRUE:
                # Regular gradient ascent based on grad(V)
                theta_prev, theta, coverage, V, gradV, G = agent.getLearnerP().learn_linear_theoretical_from_estimated_values(t, probas_stationary, Q_values)
            elif self.dict_params_learning['mode'] == LearningMode.IGA:
                # IGA: Integer Gradient Ascent (only the signs of the Q differences are of interest here)
                keys_probas = probas_stationary.keys()
                keys_Q_values = Q_values.keys()
                assert [K-1, K] == sorted(keys_probas) and [K-1, K] == sorted(keys_Q_values), \
                    "Keys K-1={} and K={} are present in the dictionaries containing the stationary probability estimations ({}) and the Q values ({})" \
                        .format(K-1, K, probas_stationary, Q_values)
                theta_prev, theta, coverage, V, gradV, G = agent.getLearnerP().learn_linear_theoretical_from_estimated_values_IGA(t, probas_stationary, Q_values)

        # Extract simulation info that will be stored in the output file
        simul_info = dict() if simul_info is None else simul_info
        t_learn = simul_info.get('t_learn', 0)
        theta_true = simul_info.get('theta_true', np.nan)
        K = simul_info.get('K', 0)
        J_factor = simul_info.get('J_factor', 0.0)
        J = simul_info.get('J', 0)
        burnin_time_steps = simul_info.get('burnin_time_steps', 0)
        min_num_cycles_for_expectations = simul_info.get('min_num_cycles_for_expectations', 0)
        exponent = simul_info.get('exponent', 0.0)  # Exponent that defines N and T w.r.t. a reference N0 and T0 as N0*exp(exponent) in order to consider different N and T values of interest (e.g. exponent = -2, -1, 0, 1, 2, etc.)
        N = simul_info.get('N', 0)              # number of particles used in the FV process (this is 1 for MC)
        T = simul_info.get('T', 0)              # this is either the number of arrival events in FVRL or the number of time steps in MC learning
        err_phi = simul_info.get('err_phi', 0)  # expected relative error in the estimation of Phi(t,K) when using N particles
        err_et = simul_info.get('err_et', 0)    # expected relative error in the estimation of E(T_A) when using T arrival events
        seed = simul_info.get('seed')
        expected_cycle_time = simul_info.get('expected_cycle_time', 0.0)
        n_cycles = simul_info.get('n_cycles', 0)
        max_survival_time = simul_info.get('max_survival_time', 0.0)
        n_events_mc = simul_info.get('n_events_mc', 0)
        n_events_fv = simul_info.get('n_events_fv', 0)
        n_trajectories_Q = simul_info.get('n_Q', 0.0)

        self.thetas += [theta_prev]
        self.thetas_updated += [theta]
        self.state_coverage += [coverage]

        # Compute statistics on the delta(Q) values (we need statistics because they are usually computed on more than one state)
        # The statistics normally are: n, min, mean, max, CV
        Q_diff_values = [Qs[1] - Qs[0] for x, Qs in Q_values.items()]
        if len(Q_diff_values) > 0:
            self.Q_diff += [[n_trajectories_Q, np.min(Q_diff_values), np.mean(Q_diff_values), np.max(Q_diff_values), np.std(Q_diff_values) / abs(np.mean(Q_diff_values))]]
        else:
            self.Q_diff += [[np.nan]*4]

        V_values_list = list(V.values())
        if len(V_values_list) > 0:
            self.V += [[len(V_values_list), np.min(V_values_list), np.mean(V_values_list), np.max(V_values_list), np.std(V_values_list) / abs(np.mean(V_values_list))]]  # NOTE: V is NOT always the average value, rho... for instance, NOT when linear_theoretical_from_estimated_values() is called above, in which case it should be the average Q value between the two different actions (accept/reject) for each blocking state observed during the simulation
        else:
            self.V += [[np.nan]*4]
        self.gradV += [gradV]
        G_values_list = list(G.values())
        if len(G_values_list) > 0:
            self.G += [[len(G_values_list), np.min(G_values_list), np.mean(G_values_list), np.max(G_values_list), np.std(G_values_list) / abs(np.mean(G_values_list))]]  # Note that G is Q_diff when we estimate the gradient from its theoretical expression, using the estimates of the stationary probabilities and the differences of Q
        else:
            self.G += [[np.nan]*4]
        self.n_events_mc += [n_events_mc]
        self.n_events_fv += [n_events_fv]
        self.n_trajectories_Q += [n_trajectories_Q]

        agent.getLearnerP().update_learning_epoch()
        #print("*** Learning time updated: time = {} ***".format(agent.getLearnerP().getLearningEpoch()))
        agent.getLearnerP().update_learning_rate_by_learning_epoch()
        #print("*** Learning rate updated: alpha = {} ***".format(agent.getLearnerP().getLearningRate()))

        if self.save:
            self.fh_results.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n" \
                                    .format(self.case,
                                            self.replication,
                                            t_learn,
                                            theta_true,
                                            self.thetas[-1],
                                            self.thetas_updated[-1],
                                            K,
                                            J_factor,
                                            J,
                                            burnin_time_steps,
                                            min_num_cycles_for_expectations,
                                            exponent,
                                            N,
                                            T,
                                            err_phi,
                                            err_et,
                                            seed,
                                            expected_cycle_time,
                                            n_cycles,
                                            max_survival_time,
                                            n_states_blocking,
                                            coverage,
                                            proba_blocking,
                                            proba_blocking_true,
                                            error_proba_blocking,
                                            expected_reward,
                                            expected_reward_true,
                                            error_expected_reward,
                                            self.Q_diff[-1],
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
        return self.dict_params_learning['t_learn']

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
        self.dict_params_learning['t_learn'] = t_learn


#-------------------------------------------- FUNCTIONS --------------------------------------#
def define_queue_environment_and_agent(dict_params: dict):
    """
    Define the queue environment and agent that will be used in the simulation

    Arguments:
    dict_params: dict
        Dictionary defining the simulation parameters with at least the following entries:
        - 'environment': parameters for the queue environment.
        - 'policy': parameters for the policy.
        - 'learners': parameter for the value functions and policy learners.
        - 'agent': parameters for the agent.

        Each entry in turn, should contain at least the following entries:
        - 'environment': dictionary defining the queue environment parameters with the following entries:
            - 'queue_system': name of the queue system to be instantiated. Valid values are:
                "single-server", "multi-server" (both accepting queues or buffers), "loss-network" (no buffer in this case).
            - 'capacity': capacity K of the queue's buffer.
            - 'nservers': number of servers c in the queue system.
            - 'job_class_rates': list containing the Job arrival rates by job class.
            - 'service_rates': list containing the service rates of the servers in the queue system.
            - 'policy_assignment_probabilities': list of lists defining the job class to server assignment probabilities.
                Each inner list should have as many elements as servers in the system.
                The outer list should have as many elements as job classes accepted by the system.
                Ex: [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]] when nservers = 3 and len(job_class_rates) = 2
            - 'reward_func': function returning the reward given by the environment for each state and action.
            - 'rewards_accept_by_job_class': list of rewards for the acceptance of the different job classes
            whose rate is given in job_class_rates.
        - 'policy': dictionary defining the type of parameterized policy to use when learning the policy with the following entries:
            - 'parameterized_policy': object defining the parameterized policy.
            - 'theta': the initial theta parameter value for the parameterized policy.
        - 'learners': dictionary defining the learners of value functions and policies with the following entries:
            - 'V': dictionary defining the state value function parameters with the following entries:
                - 'learner': learner for the state value function V.
                - 'params': dictionary with the learner parameters as follows:
                    - 'gamma': discount factor used for the return G(t).
            - 'Q': dictionary defining the action-state value function parameters with the following entries:
                NOT USED FOR NOW.
                - 'learner': learner for the action-state value function V.
                - 'params': dictionary with the learner parameters. Undefined for now.
            - 'P': dictionary defining the policy learner parameters with the following entries:
                - 'learner': learner of the policy P.
                - 'params': dictionary with the learner parameters:
                    - 'alpha_start': initial value for the learning rate alpha.
                    - 'adjust_alpha': whether to adjust the learning rate alpha as learning progresses.
                    - 'func_adjust_alpha': function used to adjust the learning rate alpha.
                    - 'min_time_to_update_alpha': minimum time step at which alpha is updated/adjusted.
                    - 'alpha_min': minimum learning rate allowed during the learning process.
                    - 'fixed_window': whether to use a fixed window in the estimation of the observed return (I think).
                    - 'clipping': whether to use clipping when updating the policy parameter theta.
                    - 'clipping_value': value at which the delta(theta) is clipped every time it is updated (e.g. +1.0).
        - 'agent': dictionary with the agent setup, with the following entries:
            - 'agent': agent object that interacts with the environment.

    Return: Tuple
    Triple containing:
    - the queue environment created from the input parameters
    - the loads of each server in the queue system, computed from the job class arrival rates
    and the given assignment probabilities.
    - the agent interacting with the environment, with its policies and learners.
    """
    set_entries_params = {'environment', 'policy', 'learners', 'agent'}
    if not set_entries_params.issubset(dict_params.keys()):
        raise ValueError("Missing entries in the dict_params dictionary: {}" \
                         .format(set_entries_params.difference(dict_params.keys())))

    # Environment
    set_entries_params_environment = {'queue_system', 'capacity', 'nservers', 'job_class_rates', 'service_rates',
                                      'policy_assignment_probabilities', 'reward_func',
                                      'rewards_accept_by_job_class'}
    if not set_entries_params_environment.issubset(dict_params['environment'].keys()):
        raise ValueError("Missing entries in the dict_params['environment'] dictionary: {}" \
                         .format(set_entries_params_environment.difference(dict_params['environment'].keys())))

    # Policy (Job Acceptance)
    set_entries_params_policy = {'parameterized_policy', 'theta'}
    if not set_entries_params_policy.issubset(dict_params['policy'].keys()):
        raise ValueError("Missing entries in the dict_params['policy'] dictionary: {}" \
                         .format(set_entries_params_policy.difference(dict_params['policy'].keys())))

    # Learners (of V, Q, and P)
    set_entries_params_learners = {'V', 'Q', 'P'}
    if not set_entries_params_learners.issubset(dict_params['learners'].keys()):
        raise ValueError("Missing entries in the dict_params['learners'] dictionary: {}" \
                         .format(set_entries_params_learners.difference(dict_params['learners'].keys())))

    # Agent
    set_entries_params_agent = {'agent'}
    if not set_entries_params_agent.issubset(dict_params['agent'].keys()):
        raise ValueError("Missing entries in the dict_params['agent'] dictionary: {}" \
                         .format(set_entries_params_agent.difference(dict_params['agent'].keys())))

    # Store the parameters into different dictionaries, for easier handling below
    dict_params_env = dict_params['environment']
    dict_params_policy = dict_params['policy']
    dict_params_learners = dict_params['learners']
    dict_params_agent = dict_params['agent']

    # Queue
    policy_assign = PolJobAssignmentProbabilistic(dict_params_env['policy_assignment_probabilities'])
    if dict_params_env['queue_system'] in ["single-server", "multi-server"]:
        assert policy_assign is not None, "The policy assignment object, `policy_assign`, must not be None"
        job_rates_by_server = compute_job_rates_by_server(dict_params_env['job_class_rates'],
                                                          dict_params_env['nservers'],
                                                          policy_assign.getProbabilisticMap())
    else:
        assert dict_params_env['queue_system'] == "loss-network", "The queue system must be a loss network ({})".format(dict_params_env['queue_system'])
        assert dict_params_env['nservers'] == len(dict_params_env['service_rates']), \
            "The number of servers parameter `dict_params_env['nservers']` ({}) must coincide with the number of service rates given in parameter `dict_params_env['service_rates']` ({})" \
            .format(dict_params_env['nservers'], dict_params_env['service_rates'])
        job_rates_by_server = dict_params_env['job_class_rates']

    # Queue M/M/c/K (used for ALL queue systems)
    queue = QueueMM(job_rates_by_server, dict_params_env['service_rates'],
                    dict_params_env['nservers'], dict_params_env['capacity'])

    # Queue environment
    if dict_params_env['queue_system'] in ["single-server", "multi-server"]:
        env_queue = env_queues.EnvQueueSingleBufferWithJobClasses(queue, dict_params_env['job_class_rates'],
                                                                  dict_params_env['reward_func'],
                                                                  dict_params_env['rewards_accept_by_job_class'])
    elif dict_params_env['queue_system'] == "loss-network":
        env_queue = env_queues.EnvQueueLossNetworkWithJobClasses(   queue,
                                                                    dict_params_env['reward_func'],
                                                                    dict_params_env['rewards_accept_by_job_class'],
                                                                    dict_params_env.get('reward_func_params'))
    else:
        raise ValueError("Invalid value for entry 'environment -> queue_system' in `dict_params`: {}\nValid values are (case sensitive): {}" \
                         .format(dict_params_env['queue_system'], ["single-server", "multi-server", "loss-network"]))

    rhos = [l / m for l, m in zip(job_rates_by_server, dict_params_env['service_rates'])]

    # Acceptance and assignment policies
    policies = dict({PolicyTypes.ACCEPT: dict_params_policy['parameterized_policy'](env_queue, theta=dict_params_policy['theta'])
                                            if is_scalar(dict_params_policy['theta'])
                                            else [policy(env_queue, theta=dict_params_policy['theta'][i]) for i, policy in enumerate(dict_params_policy['parameterized_policy'])],
                     PolicyTypes.ASSIGN: policy_assign})

    # Learners (for V, Q, and P)
    learnerV = dict_params_learners['V']['learner'](env_queue,
                                                    gamma=dict_params_learners['V']['params'].get('gamma'))
    learnerQ = None
    if dict_params_learners['P']['learner'] is None:
        learnerP = None
    else:
        # NOTE: We now instantiate a learner object where we indicate that the policy learner learns the Acceptance Policy
        # (but NOT e.g. the Assignment Policy). This is the difference between the `policies` object created above
        # (which contains ALL policies the agent uses when interacting with the environment) and the policy stored
        # in the `learnerP` object created now (which only learns ONE policy --in principle the agent could be prepared
        # to learn more than one policy, but this is not assumed here for the learner passed in attribute dict_params_learners['P']['learner'].
        learnerP = dict_params_learners['P']['learner'](env_queue,
                                                        policies[PolicyTypes.ACCEPT],
                                                        learnerV,
                                                        alpha=dict_params_learners['P']['params'].get('alpha_start'),
                                                        adjust_alpha=dict_params_learners['P']['params'].get('adjust_alpha'),
                                                        func_adjust_alpha=dict_params_learners['P']['params'].get('func_adjust_alpha'),
                                                        min_time_to_update_alpha=dict_params_learners['P']['params'].get('min_time_to_update_alpha'),
                                                        alpha_min=dict_params_learners['P']['params'].get('alpha_min'),
                                                        fixed_window=dict_params_learners['P']['params'].get('fixed_window'),
                                                        clipping=dict_params_learners['P']['params'].get('clipping'),
                                                        clipping_value=dict_params_learners['P']['params'].get('clipping_value'))

    # TODO: (2022/01/17) Can we add the following assertion at some point?
    # from Python.lib.agents.learners import GenericLearner
    # assert isinstance(learnerV, GenericLearner)
    learners = dict({LearnerTypes.V: learnerV,
                     LearnerTypes.Q: learnerQ,
                     LearnerTypes.P: learnerP
                     })

    # Agent operating on the given policies and learners
    agent = dict_params_agent['agent'](env_queue, policies, learners)

    return env_queue, rhos, agent


def compute_nparticles_and_nsteps_for_fv_process_many_settings(rhos: list,
                                                               K_values: Union[list, np.ndarray]=[5, 10, 20, 40],
                                                               JF_values: Union[list, np.ndarray]=np.arange(0.1, 0.8, 0.1),
                                                               error_rel: Union[list, np.ndarray]=np.arange(0.1, 1, 0.1)):
    df_results = pd.DataFrame(columns=['rhos', 'K', 'J', 'J/K', 'error', 'N', 'T'])

    for K in K_values:
        for JF in JF_values:
            for err in error_rel:
                J, N, T = compute_nparticles_and_narrivals_for_fv_process(rhos, K, JF,
                                                                          error_rel_phi=err,
                                                                          error_rel_et=err,
                                                                          return_J=True)
                df_results = pd.concat([df_results, pd.DataFrame([[rhos, K, J, JF, err, N, T]], columns=df_results.columns)],
                                       axis=0, ignore_index=True)

    return df_results


def compute_nparticles_and_narrivals_for_fv_process(rhos: list, capacity: int, buffer_size_activation_factor: float= 1/3,
                                                    error_rel_phi=0.50, error_rel_et=0.50,
                                                    return_J=False,
                                                    constant_proportionality=1):
    """
    Computes the minimum number of particles and the minimum number of arrivals to use in the FV process
    for expected relative errors in the estimation of Phi(t, K) and of E(T_A), respectively,
    for given queue capacity K and absorption set size factor.

    Arguments:
    rhos: list
        List of the server intensities: lambda / mu for each server in the system, where lambda is the job
        arrival rate and mu is the service rate.

    capacity: int
        Capacity of the system: maximum size of the buffer placed at the entrance of the system.

    buffer_size_activation_factor: (opt) float
        Buffer size activation factor J/K based on which the activation set of states is defined as
        J = max(1, int( round( factor*K ) )), such as J-1 defines the absorption set of states
        (which justifies the use of `max(1, ...)` as J-1 should be >= 0).
        Its value should not be outside the interval (0, 1] and should be such that the resulting
        buffer size for activation value, J, is between 1 and K. If this is not the case, J is bounded between 1 and K.
        default: 1/3, which is the optimum in the sense that the algorithm complexity to estimate Phi(t, K) and E(T_A)
        in the Fleming-Viot estimation is the same.

    error_rel_phi: (opt) float
        Maximum relative error for the estimation of Phi(t, K).
        This is the relative error incurred in the estimation of a binomial probability based on N trials,
        which is equal to sqrt((1-p)/p) / sqrt(N), where p is the blocking probability under the Fleming-Viot process
        which is proportional to rho^((K-J+1)/2).
        Use None if no calculation of N is wished.
        default: 0.50

    error_rel_et: (opt) float
        Maximum relative error for the estimation of the expected absorption cycle time E(T_A).
        The calculation of the number of cycles needed assumes that:
        - K is large so that rho^K << 1.
        - E(T_A) is approximated by E(T1) where T1 is the exit time from the absorption set of size J.
        - the standard deviation of the estimation of E(T_A) is proportional to E(T_A),
        which gives a relative error equal to 1/sqrt(M) where M is the number of reabsorption cycles.
        (Note however that, from simulations of the FV estimation for different J/K values, it was found that
        the standard deviation is proportional to E(T_A)^1.6, more precisely Std(T_A) = 0.01 * E(T_A)^1.6
        See an Excel file where I analyze this. However, in order to stay conservative here, we stick to the constant
        of proportionality equal to 1, as using 0.01 would give a T value that is 10,000 times smaller than with
        constant of proportionality = 1)
        The number of arrival events is then computed as M/q, where q is the blocking probability of the queue system
        as if the queue had capacity = J-1, where J = buffer_size_activation_factor * K.
        Use None if no calculation of T is wished.
        default: 0.50

    return_J: (opt) bool
        Whether to return the value of J on which the values of N and T are computed, as this value is computed from
        the given J factor and K.
        default: False

    constant_proportionality: float
        Constant of proportionality assumed between the standard deviation of the estimation of E(T_A) and its expected value.
        default: 1

    Return: int or tuple
    The returned value is int when either error_rel_phi or error_rel_et is None and when return_J = False,
    o.w. it's a tuple with the following elements:
    - J: (if return_J=True) value of the absorption size set computed on the basis of given buffer size for activation factor and K,
    and used as input to compute N and T.
    - N: (if error_rel_phi is not None) number of particles needed for the given relative error for Phi(t, K), error_rel_phi.
    - T: (if error_rel_et is not None) Number of arrival events for the given relative error for E(T_A), error_rel_et.
    """
    if not (1 <= np.round(buffer_size_activation_factor * capacity) <= capacity):
        warnings.warn("The value of input parameter `buffer_size_activation_factor` ({}) is either too small or too large" \
                      .format(buffer_size_activation_factor) +
                      " to have a valid buffer size for activation value, J (J yielded = {}). This value will be bounded to the interval [1, K={}]" \
                      .format(int(np.round(buffer_size_activation_factor * capacity)), int(capacity)))
    K = int(capacity)
    J = min(max(1, int( np.round(buffer_size_activation_factor * K) )), K)

    if error_rel_phi is not None:
        # Compute N
        # The calculation is based on the blocking probability associated to the Fleming-Viot process, which is
        # proportional to rho^((K-J+1)/2). Therefore we compute the blocking probability for a queue with capacity
        # int( ceil(K-J+1)/2 )
        capacity_effective = int( np.ceil((K-J+1)/2) )
        pK = compute_blocking_probability_birth_death_process(rhos, capacity_effective)
        N = int(np.ceil( (1 - pK) / pK / error_rel_phi**2 ))

    if error_rel_et is not None:
        # Compute T, the number of arrival events required to achieve the given expected relative error.
        # This is based on the expected exit time from the absorption set of size J (under stationarity),
        # T1, which is what mostly defines the magnitude of E(T_A) = E(T1+T2), as T2 is usually quite small
        # compared to T1, at least for large J values.
        # Note that the expected exit time is proportional to the expected return time to J-1, with constant
        # of proportionality (in the M/M/1/K case) equal to (1 + mu/lambda) (see details in my green notebook,
        # where I wrote about this on 06-Nov-2022).
        pJm1 = compute_blocking_probability_birth_death_process(rhos, J - 1)
        #M = int(np.ceil( constant_proportionality**2 / error_rel_et**2 ))
        T = int(np.ceil( constant_proportionality**2 / error_rel_et**2 / pJm1 ))

    if error_rel_phi is not None and error_rel_et is not None:
        if return_J:
            return J, N, T
        else:
            return N, T
    if error_rel_phi is not None:
        if return_J:
            return J, N
        else:
            return N
    if error_rel_et is not None:
        if return_J:
            return J, T
        else:
            return T


def compute_rel_errors_for_fv_process(rhos: list, capacity: int, buffer_size_activation_factor: float=1/3,
                                      N: int=None, T: int=None, constant_proportionality=1):
    """
    Computes the relative errors expected in the estimation of the two main quantities of the FV estimator,
    Phi(t, K), whose accuracy increases as the activation factor J/K increases, and E(T_A), whose accuracy
    increases as the activation buffer size J (the absolute value, NOT the factor J/K!) decreases.

    Arguments:
    rhos: list
        List of the server intensities: lambda / mu for each server in the system, where lambda is the job
        arrival rate and mu is the service rate.

    capacity: int
        Capacity of the system: maximum size of the buffer placed at the entrance of the system.

    N: (opt) int
        Number of particles used in the FV process.
        default: None

    T: (opt) int
        Number of arrival events allowed in the simulation of the single particle that estimates E(T_A).
        default: None

    buffer_size_activation_factor: (opt) float
        Buffer size activation factor J/K based on which the activation set of states is defined as
        J = max(1, int( round( factor*K ) )), such as J-1 defines the absorption set of states
        (which justifies the use of `max(1, ...)` as J-1 should be >= 0).
        Its value should not be outside the interval (0, 1] and should be such that the resulting
        buffer size for activation value, J, is between 1 and K. If this is not the case, J is bounded between 1 and K.
        default: 1/3, which is the optimum in the sense that the algorithm complexity to estimate Phi(t, K) and E(T_A)
        in the Fleming-Viot estimation is the same.

    constant_proportionality: float
        Constant of proportionality assumed between the standard deviation of the estimation of E(T_A) and its expected value.
        default: 1

    Return: float or tuple
    The return value is float when either parameter N or T is None, o.w. it's a tuple with the following elements:
    - error_rel_phi: (when N is not None) relative error expected for the estimation of Phi(t, K) given N.
        This is the relative error incurred in the estimation of a binomial probability based on N trials,
        which is equal to sqrt((1-p)/p) / sqrt(N)
    - error_rel_et: (when T is not None) approximate relative error expected for the estimation of the expected
        reabsorption cycle time E(T_A) when T arrival events are observed during the simulation from which E(T_A)
        is estimated.
        This calculation assumes that the standard deviation of the estimation of E(T_A)
        is proportional to E(T_A), with constant of proportionality given by parameter `constant_proportionality`.
        The expected relative error is then given by `constant_proportionality / sqrt(M)` where M is the expected
        number of reabsorption cycles when T arrival events are observed, satisfying M / p(J-1) = T, where p(J-1) is the
        stationary probability that the queue has occupation (buffer size) = J - 1.
        (Note that from simulations of the FV estimation for different J/K values, it was found that
        the standard deviation is proportional to E(T_A)^1.6, more precisely Std(T_A) = 0.01 * E(T_A)^1.6)
        The number of arrival events is then computed as M/q, where q is the blocking probability of the queue system
        as if the queue had capacity = J-1, where J = buffer_size_activation_factor * K.
    """
    if not (1 <= np.round(buffer_size_activation_factor * capacity) <= capacity):
        warnings.warn("The value of input parameter `buffer_size_activation_factor` ({}) is either too small or too large" \
                      .format(buffer_size_activation_factor) +
                      " to have a valid buffer size for activation value, J (J yielded = {}). This value will be bounded to the interval [1, K={}]" \
                      .format(int(np.round(buffer_size_activation_factor * capacity)), int(capacity)))
    K = int(capacity)
    J = min(max(1, int( np.round(buffer_size_activation_factor * K) )), K)

    if N is not None:
        # This is the equivalent of a queue that has the blocking probability observed under the Fleming-Viot process
        # which is proportional to sqrt(K-J+1) * rho^[(K-J+1)/2] (see the Introduction of our paper where this is stated)
        capacity_effective = int( np.ceil((K-J+1)/2) )
        pK = compute_blocking_probability_birth_death_process(rhos, capacity_effective)
        error_rel_phi = np.sqrt( (1 - pK) / pK / N )

    if T is not None:
        # Expected return time to J (under stationarity), which is what mostly defines the magnitude of E(T_A) = E(T1+T2)
        pJm1 = compute_blocking_probability_birth_death_process(rhos, J - 1)
        error_rel_et  = constant_proportionality * np.sqrt( 1 / T / pJm1 )    # Recall that M / p(J-1) = T (see my notes in the small green book dated 06-Nov-2022)

    if N is not None and T is not None:
        return error_rel_phi, error_rel_et
    if N is None:
        return error_rel_et
    if T is None:
        return error_rel_phi


def compute_reward_for_rejection_at_current_state(env, state=None):
    """
    Computes the reward given by the environment when rejecting an incoming job at the current state of the environment
    or at the given state (if not None).

    Arguments:
    env: Queue environment
        Queue environment used to compute the reward of rejecting an incoming job at its state or at the state given
        in parameter `state`. In the latter case, the environment is used to retrieve the reward function to use for
        the computation of the reward and its parameters.

    state: tuple
        State of the queue environment on which the reward for rejection is wished.
        The assumed tuple is 2D, where the first element is the queue state itself and the
        second element is the class of the arriving job that is rejected (which must NOT be None).

    Return: float
    The reward given by the environment when an incoming job is rejected at the environment's state or at the state
    given in parameter `state`.
    If the environment does not have a reward function defined, the returned reward is -1.0, so that the expected reward
    is equal to minus the probability of being at the current state of the environment.
    """
    if state is None:
        state = env.getState()
    reward_func = env.getRewardFunction()
    if reward_func is not None:
        params_reward_func = env.getParamsRewardFunc()
        reward = reward_func(env, state, Actions.REJECT, state, dict_params=params_reward_func)
    else:
        reward = -1.0

    return reward


def choose_state_for_buffer_size(env, buffer_size: int):
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


def get_blocking_boundaries(agent):
    """
    Returns all the blocking values for each dimension of a multidimensional state system where blocking may occur at
    arrival of the corresponding job class

    The agent is assumed to store one or more acceptance policies where the getDeterministicBlockingValue()
    method is defined. Normally each of these policies are parameterized by a real-valued parameter theta
    (as opposed to a multidimensional parameter).

    Arguments:
    agent: agent object interacting with the environment
        Agent that defines a list of accept/reject policies, as described in the description above,
        e.g. one for each arriving job class, normally parameterized policies.

    Return: list of int or list of lists
    If the number of acceptance policies defined in the agent is only one, the returned value is a list of blocking sizes (e.g. [K-1, K]).
    Otherwise, if more than one acceptance policy is defined by the agent, the returned value is a list of lists,
    with as many elements as the number of policies defined in the agent. Each element is a list of potential
    blocking sizes for each policy. Ex: [[2, 3], [6, 7], [4, 5]] where each two-element list are the values of
    K-1 and K deduced from each of the 1D-parameterized policies defined by the agent, in the case when these policies
    are a linear step parameterized by theta.
    """
    assert agent is not None
    policies = agent.getAcceptancePolicies()
    assertPoliciesAreUnivariate(policies)

    # Sizes at which blocking is deterministic for each 1D-parameterized policy (the K values)
    Ks = get_deterministic_blocking_boundaries(agent, return_int_when_single_policy=False)
    blocking_boundaries = [[k for k in range(0, Ks[p]+1) if policies[p].getPolicyForAction(Actions.REJECT, (k, p)) > 0]
                           for p in range(len(policies))]

    # Check that the list of blocking sizes for each policy does not contain repeated states
    # (i.e. the list elements are [2, 3], [5, 6] but NOT [2, 2], [6, 6], etc.)
    for boundaries in blocking_boundaries:
        assert len(set(boundaries)) == len(boundaries), "The blocking boundaries found must NOT repeat within each policy: {}".format(boundaries)

    # Extract the boundaries from the list object if there is only one set of boundaries (e.g. [ [5, 6] ] becomes --> [5, 6])
    if len(blocking_boundaries) == 1:
        return blocking_boundaries[0]
    else:
        return blocking_boundaries


def get_deterministic_blocking_boundaries(agent, thresholds: Union[float, list]=None, return_int_when_single_policy=True, exact=True):
    """
    Returns all the deterministic blocking thresholds for each dimension of a multidimensional state system
    where blocking may occur at arrival of the corresponding job class.

    The agent is assumed to store one or more acceptance policies where the getDeterministicBlockingValue()
    method is defined (when thresholds=None) or the getDeterministicBlockingValueFromTheta() is defined
    (when thresholds is given and not None). Normally each of these policies are parameterized by a real-valued parameter theta
    (as opposed to a multidimensional parameter).

    Arguments:
    agent: agent object interacting with the environment
        Agent that defines a list of accept/reject policies, as described in the description above,
        e.g. one for each arriving job class, normally parameterized policies.

    thresholds: (opt) float or list
        Thresholds on which the blocking boundaries are requested. This is useful for parameterized policies,
        for which we are interested in obtaining the *deterministic* blocking policies, for example when the policies
        are of class PolQueueTwoActionsLinearStep.
        default: None

    return_int_when_single_policy: (opt) bool
        Whether to return an integer when there is a single policy in the agent (normally done to facilitate processing
        in the caller of the returned value).
        default: True

    exact: (opt) bool
        Whether to return the exact deterministic blocking value, i.e. the value where blocking is truly deterministic,
        or rather the approximate deterministic blocking value, where blocking is almost deterministic.
        This essentially dictates whether we choose the integer blocking value that is ABOVE theta + 1 (ceil(theta+1), when exact=True)
        (to use during the optimization process on theta) or the integer blocking value that is CLOSEST to theta + 1 (round(theta+1), when exact=False)
        (to use at the end of the optimization process, to define the actual integer blocking size to use --so that theta = 3.01 gives the more reasonable K = 4, instead of 5),
        where theta is the parameter of the linear-step parameterized acceptance policy.
        default: True

    Return: list or int
    Either a list containing the blocking thresholds for each policy defined in the agent when more than one policy
    is defined or a integer with the blocking threshold when only one policy is defined and return_int_when_single_policy = True.
    """
    assertConsistencyPolicyLearnerAndAgentPolicy(agent)

    # Define the function that converts the real-valued theta parameter to an integer blocking size K
    func_integrable = np.ceil if exact else np.round

    if thresholds is None:
        blocking_boundaries = [policy.getDeterministicBlockingValue(func=func_integrable) for policy in agent.getAcceptancePolicies()]
    else:
        if is_scalar(thresholds):
            # Make the thresholds parameter always a list for easier treatment below
            thresholds = [thresholds]
        if len(thresholds) != len(agent.getAcceptancePolicies()):
            raise ValueError("When given, the length of the `thresholds` parameter must be equal to the number of policies defined in the agent")
        blocking_boundaries = [policy.getDeterministicBlockingValueFromTheta(thresholds[i], func=func_integrable) for i, policy in enumerate(agent.getAcceptancePolicies())]

    # Return a scalar when the blocking boundary is only one (i.e. when there is only one policy and return_int_when_single_policy = True)
    # The goal is to facilitate treatment of the K value (the blocking threshold) when the blocking is single-buffer-based blocking)
    if len(blocking_boundaries) == 1 and return_int_when_single_policy:
        return blocking_boundaries[0]

    return blocking_boundaries


def compute_burnin_time_from_burnin_time_steps(env, burnin_time_steps):
    """
    Computes the continuous burn-in time for a given number of burn-in time steps (i.e. number of events)
    based on the true dynamics of the queue environment used to run a simulation.

    That is, the continuous burn-in time is an *expected* value.

    Arguments:
    env: Queue environment
        Queue environment used to run the simulation from where the expected inter-event time is taken.

    Return: float
    Continuous time to be used as burn-in time based on the burn-in time steps, i.e. from a burn-in number of events
    to observe before assuming the queue system is in stationary regime.
    It is computed as the product of `burnin_time_steps` and the expected inter-event time.
    """
    return burnin_time_steps * compute_expected_inter_event_time(env)


def compute_expected_inter_event_time(env):
    """
    Computes the expected inter-event time for a given queue environment which may consist of several arriving
    job classes and several servers.

    Arguments:
    env: Queue environment
        Queue environment from which the job class arrival rates and the service rates are taken.

    Return: float
    The inverse of the event rate in the system which is the sum of all job class arrival rates and all service rates.
    """
    # Note that I compute the maximum rate of an event by looking at the job class arrival rate, NOT by looking at
    # the job arrival rate to each server. The reason is that the latter depends on the assignment probability of
    # job class to server, and the former still represents an event rate, and in addition is at least equal to
    # the maximum job arrival rate across servers, as this value is at most the largest job CLASS arrival rate
    # (because the job arrival rate by server are reduced by the assignment probability to the server which is <= 1).
    event_rate = np.sum(env.getJobClassRates()) + np.sum(env.getServiceRates())

    return 1/event_rate


def compute_proba_blocking(env, agent: AgeQueue, probas_stationary_blocking: dict):
    """
    Computes the blocking probability of the queue system represented by environment `env`.
    See the documentation for estimate_expected_reward().
    """
    return estimate_expected_reward(env, agent, probas_stationary_blocking, reward=1)


def estimate_expected_reward(env, agent: AgeQueue, probas_stationary: dict, reward=None):
    """
    Estimates the expected reward of the queue system defined in the given environment assuming a non-zero reward
    happens at the states or buffer-sizes/job-occupancies of interest, i.e. those where the policy yields
    a positive probability of rejection.

    IMPORTANT: Only a reward for REJECTING a job is taken into account, no reward for ACCEPTING a job is considered.

    Note that both the acceptance policy and the reward may depend on the class of the job arriving to the queue.
    Therefore, the expected reward is estimated as follows:
    a) when the acceptance policy depends on the class of the incoming job:
        E[R] = sum_{buffer-sizes-or-job-occupancies s in `probas_stationary`}
                    [ sum_{arriving-job-class i} [ R(s, i) * Pr(reject | s, arriving-job-class i) * Pr(arriving-job-class i) ] * Pr(s) ]
        where,
            P(arriving-job-class i) = lambda(i) / sum_{j}{lambda(j)}
    b) otherwise:
        E[R] = sum_{buffer-sizes | job-occupancies s in `probas_stationary`} { R(s) * Pr(reject | s) * Pr(s) }

    where R(s) is the reward associated to a rejection at buffer-size/job-occupancy s defined in the queue environment,
    and Pr(s) is the stationary probability estimate of buffer-size/job-occupancy s.

    Arguments:
    env: Queue environment
        Queue environment where the reward function of blocking states or blocking buffer size is defined.
        The reward is computed by calling function compute_reward_for_rejection_at_current_state().

    agent: Agent
        Agent where the accept/reject policy is defined.

    probas_stationary: dict
        Dictionary with the estimated stationary probability of the states or buffer sizes which are assumed to yield
        non-zero rewards. These states or buffer sizes are the dictionary keys.

    reward: (opt) float
        When given, this value is used as the constant reward assigned to a state present in the dictionary containing
        the stationary probabilities when an incoming job is REJECTED.
        This is useful when we are interested in computing the rejection/blocking probability rather than the expected reward.

    Return: float
    Estimated expected reward.
    """
    expected_reward = 0.0
    for s in probas_stationary.keys():
        s_tuple = tuple([s]) if is_scalar(s) else s
        assertConsistencyPolicyLearnerAndAgentPolicy(agent)
        if probas_stationary[s] > 0.0:
            if agent.getAcceptancePolicyDependenceOnJobClass():
                proba_arrival_jobclass = np.array(env.getJobClassRates()) / np.sum(env.getJobClassRates())
                expected_reward_reject = np.sum([( compute_reward_for_rejection_at_current_state(env, state=(occupancy, i)) if reward is None else reward ) * \
                                                 (agent.getAcceptancePolicies()[i].getPolicyForAction(Actions.REJECT, (occupancy, i)) * int(sum(s_tuple) < env.getCapacity()) + int(sum(s_tuple) >= env.getCapacity())) * \
                                                 proba_arrival_jobclass[i]
                                                    for i, occupancy in enumerate(s_tuple)])
            else:
                # We set the job_class of the state to None because its value does not matter in determining the rejection probability
                state = (s, None)
                expected_reward_reject = ( compute_reward_for_rejection_at_current_state(env, state=state) if reward is None else reward ) * \
                                         agent.getAcceptancePolicies()[0].getPolicyForAction(Actions.REJECT, state)
            expected_reward += expected_reward_reject * \
                               probas_stationary[s]
    return expected_reward


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
    assert isinstance(envs, list) and len(envs) > 0, "The `envs` parameter is a list and has at least one element"
    if envs[0].getBufferType() == BufferType.NOBUFFER:
        # There is NO buffer in the system
        # => All jobs in the system can be served at any time, meaning that the service rates are proportional to the number of jobs of each class
        valid_rates = np.array([list(env.getJobClassRates()) + [s * r if s > 0 else np.nan for r, s in zip(env.getServiceRates(), env.getQueueState())]
                                for env in envs])
    else:
        # There is a buffer in the system
        # => Jobs are served one at a time by each server
        valid_rates = np.array([list(env.getJobClassRates()) + [r if s > 0 else np.nan for r, s in zip(env.getServiceRates(), env.getQueueState())]
                                for env in envs])

    # Generate the event time and the index on the rates array to which it is associated
    event_time, event_idx = generate_min_exponential_time(valid_rates.flatten())

    # Convert the linear index back to the 2D index in valid_rates where the indices represent:
    # (selected env/particle, selected rate)
    idx_env, job_class_or_server = index_linear2multi(event_idx, valid_rates.shape)

    if False and len(envs) > 1:
        # (2022/10/16) Only show this information when we are simulating N particles, not when we are simulating just one particle
        # (just because I am now debugging the event rates in the FV simulation, which seem to be underestimated)
        print("\nQueue states:\n{}".format([[x for x in env.getQueueState()] for env in envs]))
        print("Valid rates: {}".format(valid_rates))
        print("Event time: {}, generated from rate index: (1D) {}, (2D) {}".format(event_time, event_idx, [idx_env, job_class_or_server]))

    # Define whether the generated event is an incoming job class or a completed service
    # and return a different tuple accordingly.
    # It is assumed that the number of job classes and servers are the same in each environment.
    n_job_classes = envs[0].getNumJobClasses()
    if job_class_or_server < n_job_classes:
        # The selected rate is stored in the first n_job_classes columns of the 2D arrangement of event rates
        # => The event is an incoming job class, as the first n_job_classes columns store the job class arrival rates
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
        NOTE: This object is updated with the next state of the system after managing the job arrival.

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

    # Assign the job to a server if accepted for queues having a buffer to hold jobs
    # (i.e. loss networks are excluded from the job assignment part as it is irrelevant to which server the job is assigned)
    if action_accept_reject == Actions.ACCEPT and env.getBufferType() != BufferType.NOBUFFER:
        # Assign the job just accepted to a server and update the next_state value
        # (because the next_state from the ACCEPT action has not been updated above
        # as we need to know to which server the accepted job is assigned in order
        # to know the next state of the queue environment)
        _, next_state, reward_assign, _ = step(t, env, agent, PolicyTypes.ASSIGN)
    else:
        reward_assign = 0.0
    action = action_accept_reject
    reward = reward_accept_reject + reward_assign

    if action == Actions.ACCEPT:
        assert env.getBufferSizeFromState(next_state) == env.getBufferSizeFromState(state) + 1, \
            "The buffer size after a BIRTH increased by 1: BS(t) = {}, BS(t+1) = {}" \
                .format(env.getBufferSizeFromState(state), env.getBufferSizeFromState(next_state))
        assert reward == 0.0, "A non-zero reward is not possible when the action is ACCEPT.\n" \
                              "--> action = {}, REWARD ASSIGN: {}, REWARD = {}".format(action_accept_reject, reward_assign, reward)

    # Update the average reward (just for information purposes)
    if agent.getLearnerV() is not None:
        agent.getLearnerV().updateAverageReward(t, reward)

    # Compute and store the gradient of the policy evaluated on the taken action A(t) given the state S(t)
    # Goal: depending on the learning method, this value may be useful
    if agent.getLearnerP() is not None:
        gradient_for_action = agent.getLearnerP().getGradient(action_accept_reject, state)
        agent.getLearnerP().store_gradient(state, action_accept_reject, gradient_for_action)
    else:
        gradient_for_action = 0.0

    return action, next_state, reward, gradient_for_action


def manage_service(env, agent, state, server):
    """
    Manages a completed service event

    Arguments:
    env: Queue environment
        Queue environment where the service takes place.
        NOTE: This object is updated with the next state of the system after managing the service.

    agent: Agent
        Agent acting on the environment.
        This parameter is not used but it is left for consistency with manage_job_arrival() method.

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
    # Need to convert the queue state to a list (from tuple) in order to be able to update its value just below
    # (as tuples are non-mutable objects). We convert the list back to tuple after update.
    next_queue_state = list(copy.deepcopy(queue_state))
    next_queue_state[server] -= 1

    env.setState((tuple(next_queue_state), None))
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


def estimate_blocking_mc(env, agent, dict_params_simul, dict_params_info, start_queue_state=None):
    """
    Estimates the blocking probability using Monte-Carlo

    Arguments:
    env: Queue environment
        Queue environment where the simulation runs.

    agent: Agent
        Agent that interacts with the environment.
        It should have the getLearnerV() method defined.

    simul: SimulatorQueue
        Simulation object used for the simulation that allows estimating the blocking probability.

    dict_params_simul: dict
        Dictionary containing simulation and estimation parameters.
        The dictionary should contain the keys described in function `parse_simulation_parameters()`.

    dict_params_info: dict
        Dictionary containing information to display, or parameters to deal with the information to display.

    start_queue_state: (opt) int or tuple or numpy array
        Queue state at which the queue environment starts for the simulation.
        Its type depends on how the queue environment defines the state of the system.
        default: None, in which case the start state is randomly chosen among the states of the absorption set defined
        either by the entry 'absorption_set' of the `dict_params_simul` parameter, or derived from the buffer size
        defined in entry 'buffer_size_activation' of such dictionary as dict_params_simul['buffer_size_activation'] - 1.
        Note that this start state is used to obtain a fair comparison of the Monte-Carlo estimation method with the
        Fleming-Viot estimation method, as it makes both methods start at a state in the boundary of the absorption set A,
        and this is the reason we need the parameter defining the set of absorption set A to be passed as part of the
        `dict_params_simul` dictionary.

    Return: tuple
    Tuple with the following elements:
    - proba_blocking: the estimated blocking probability.
    - expected_reward: the estimated expected reward.
    - probas_stationary_blocking: dictionary with the estimated stationary probability for each queue state or buffer size where
    blocking may occur.
    - n_cycles_used_for_probability_estimation: number of cycles used for the estimation of the stationary probabilities.
    - expected_return_time_to_start_state_or_buffer_size: estimated expected cycle time E(T) of return to the initial
    queue state or buffer size.
    This value coincides with the denominator of the MC estimator of the blocking probabilities (based on renewal theory)
    ONLY when burnin_time = 0.
    Otherwise, this expected return time may or may NOT coincide with such denominator: it will coincide if, after the
    burn-in period, the Markov chain is found at the same queue state or buffer size as the initial queue state or
    buffer size; otherwise it will not be the same.
    - n_cycles_to_start_state_or_buffer_size: number of return cycles observed to estimate E(T). This will most likely NOT
    coincide with the returned value `n_cycles_used_for_probability_estimation` which contains the number of cycles used
    in the estimation of the stationary probabilities based on renewal theory.
    - time_last_event: time of the last event (continuous-time) observed during the Monte-Carlo simulation.
    - n_events: number of events observed during the Monte-Carlo simulation used to estimate the stationary
    probabilities.
    """
    # -- Parse input parameters
    dict_params_simul = parse_simulation_parameters(dict_params_simul, env)

    # Set the simulation seed
    # Note that, although the seed is set by the queue environment (via its set_seed() method which is called by run_simulation_mc() below)
    # we also need to set the seed here because, before calling run_simulation_mc(), we choose below the initial queue state
    # on which the simulation starts randomly, among all possible states in the absorption set.
    np.random.seed(dict_params_simul['seed'])

    # Reset environment and learner of the value functions
    # IMPORTANT: We should NOT reset the learner of the policy because this function could be called as part of a
    # policy learning process! Note that the learner of the value function V MUST be reset regardless of this because
    # at this point a new queue simulation starts and at that point the value function should be learned anew.
    env.reset()
    if agent.getLearnerV() is not None:
        agent.getLearnerV().reset()

    # Selection of the queue start state out of the set of absorption states in order to have a fair comparison
    # with the Fleming-Viot method where particles are reactivated when they reach a state in the absorption set.
    if start_queue_state is None:
        start_queue_state = dict_params_simul['absorption_set'].random_choice()
    start_state = (start_queue_state, None)
    # -- Parse input parameters

    t, time_last_event, n_return_cycles, time_last_return, return_times = \
            run_simulation_mc(env, agent, dict_params_info.get('t_learn', 0), start_state, dict_params_simul['T'],
                              absorption_set=dict_params_simul['absorption_set'],
                              activation_set=dict_params_simul['activation_set'],
                              track_return_cycles=True,
                              seed=dict_params_simul['seed'],
                              verbose=dict_params_info.get('verbose', False), verbose_period=dict_params_info.get('verbose_period', 1))
    probas_stationary_blocking, expected_cycle_time, n_cycles_used_for_probability_estimation = \
        estimate_stationary_probabilities_mc(env, agent,
                                             burnin_time=dict_params_simul['burnin_time'],
                                             min_num_cycles_for_expectations=dict_params_simul['min_num_cycles_for_expectations'])

    #-- Compute the other quantities that are returned to the outside world
    # Blocking probability and expected reward
    n_events = t
    assert n_events == dict_params_simul['T']
    proba_blocking = compute_proba_blocking(env, agent, probas_stationary_blocking)
    expected_reward = estimate_expected_reward(env, agent, probas_stationary_blocking)

    # Expected return time to the start state or buffer size
    # Notes: (complement the notes written in the function's documentation above)
    # 1) This is another estimate of the expected cycle time which yields a different value from the expected cycle time
    # returned above by the call to estimate_stationary_probabilities() when the burn-in time parameter is > 0.
    # In fact, cycles are defined differently by the estimate_expected_cycle_time() function called below, as follows:
    #   a) In estimate_expected_cycle_time(), cycles are defined by the return times to the initial state or buffer size.
    #   b) In estimate_stationary_probabilities() called above, cycles are defined by the return times to whatever state
    #       or buffer size the Markov chain is found at soon after the burn-in time is over, and that state or buffer size
    #       is used in the application of the stationary probabilities estimation based on the renewal theorem (cf. Asmussen).
    #   The reason for doing (b) is that we want to take the most out of the time after the burn-in time has passed,
    #   which may not always be the case in (a) because we are waiting until the Markov chain returns to the INITIAL
    #   state or buffer size, which may not happen soon after the burn-in time has passed...
    # 2) The above observation implies that the number of cycles observed in (a) may be completely different from the
    # number of cycles observed in (b) and they are not related (i.e. n_cycles_to_start_state_or_buffer_size computed
    # in (a) could be larger or smaller than the n_cycles_used_for_probability_estimation value computed in (b)).
    # 3) The two quantities estimated by (a) and by (b) do coincide when the burn-in time parameter is equal to 0.
    # This is asserted below.
    expected_return_time_to_start_state_or_buffer_size, n_cycles_to_start_state_or_buffer_size = \
        estimate_expected_cycle_time(n_return_cycles, time_last_return,
                                     cycle_times=return_times, burnin_time=dict_params_simul['burnin_time'],
                                     min_num_cycles_for_expectations=dict_params_simul['min_num_cycles_for_expectations'])
    if dict_params_simul['burnin_time'] == 0.0 and n_return_cycles > 0:
        assert n_cycles_to_start_state_or_buffer_size == n_return_cycles
        assert  np.isnan(expected_return_time_to_start_state_or_buffer_size) and np.isnan(expected_cycle_time) or \
                np.isclose(expected_return_time_to_start_state_or_buffer_size, expected_cycle_time), \
            "[estimation_blocking_mc] The average return time to the start state or buffer size {} ({:.3f})" \
            " must coincide with the average cycle time ({:.3f}) used as denominator for the estimation of the stationary probabilities" \
            " (using renewal theory), when the burn-in time is 0.0." \
            .format(start_state, expected_return_time_to_start_state_or_buffer_size, expected_cycle_time)

    return proba_blocking, expected_reward, probas_stationary_blocking, n_cycles_used_for_probability_estimation, \
                expected_return_time_to_start_state_or_buffer_size, n_cycles_to_start_state_or_buffer_size, \
                    time_last_event, n_events


@measure_exec_time
def run_simulation_mc(env, agent, t_learn, start_state, t_sim_max,
                      absorption_set: SetOfStates, activation_set: SetOfStates,
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

    start_state: Queue-environment-dependent
        (Environment) State at which the queue environment starts for the simulation.
        Its type depends on how the queue environment defines the state of the system.
        Ex: when the queue is of class GenericEnvQueueWithJobClasses, the state is a tuple with the following information:
        (queue_state, job_class) = (state of the queue itself, class of the arriving job).

    t_sim_max: int
        Maximum simulation *time steps* allowed for the simulation (equivalent to the number of observed events)
        when track_absorptions=False (which we assume it indicates that this function is called as part of the
        Monte-Carlo estimation of the blocking probability)
        OR maximum number of *arrival events* when track_absorption=True (which we assume it indicates that
        this function is called as part of the Fleming-Viot estimation of the blocking probability).
        Note that its meaning is different in one case and the other because in the Fleming-Viot case,
        the number of arrivals, and NOT the number of events, is the one that determines the expected error
        in the estimation of E(T_A), which is due to the fact that the expected reabsorption time is normally larger
        than the expected return time to J-1 (as, in order to be reabsorbed, the particle first needs to exit
        the absorption set A) (for further details, see calculations in my green small notebok written on 06-Nov-2022).

    absorption_set: SetOfStates
        Set with the entrance states to the zero-reward set of states A.

    activation_set: SetOfStates
        Set with the entrance states to the complement of the zero-reward set of states A.

    track_return_cycles: (opt) bool
        Whether to track the return cycle times to the initial state/buffer-size (defined by parameter `start_state`),
        regardless of the value of the previous state (if the previous state is in the activation set of states, then
        it would be tantamount to an absorption).
        default: False

    track_absorptions: (opt) bool
        Whether to track the absorption times, i.e. the times spent on the reabsorption cycles defined by the
        interval between two consecutive times in which the queue returns to the initial state/buffer-size
        (defined by parameter `start_state`), coming from the activation set of states.
        In addition, the exit times from the absorption set are also tracked, i.e. the times spent between an entry to
        and an exit from the absorption set, i.e. when the queue system visits a state in the activation set
        coming from the absorption set of states.
        The exit times information is handy when running this function for the FV estimator, where we need to estimate
        E(T_A) as E(T_E) + E(T_K) where T_E is the exit time and T_K is the killing time since exit, which is estimated
        from a separate simulation run on the N particles used for the FV simulation. That is, estimating E(T_E) instead
        of estimating E(T_A) directly guarantees that the estimation of E(T_A) is always larger than the estimator of
        E(T_K), thus avoiding the FV estimator from overshooting (i.e. estimating a larger value than it should).
        Besides, this condition of \hat{E(T_A)} >= \hat{E(T_K)} is used in the proof of convergence of the FV estimator
        in our Fleming-Viot paper.
        default: False

    track_survival: (opt) bool
        Whether to track the survival times, i.e. the times spent between exit from the absorption set into the activation set
        and the first absorption afterwards (i.e. when the queue goes from a state in the activation set to a state
        in the absorption set).
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
    - t: the last time step (integer-valued) of the simulation process.
    - time_abs: the continuous (absolute) time of the last observed event during the simulation.

    If track_return_cycles = True, in addition:
    - n_cycles: the number of cycles observed in the simulation, until the last return to the initial state/buffer-size.
    - time_last_return: the continuous time at which the Markov Chain returns to the initial state/buffer-size
    (regardless of where it came from).
    - return_times: list with the observed return times to a state with initial state/buffer-size (whose value is
    defined by `start_state`).

    If track_absorptions = True and track_survival = False, in addition to `t` and `time_abs`:
    - n_cycles: the number of cycles observed in the simulation, until the last return to the initial state/buffer-size
    from the activation set of states (i.e. as if it were an absorption).
    - time_last_absorption: the continuous time at which the Markov Chain returns to the initial state/buffer-size
    from ABOVE (i.e. as if it were an absorption).
    See the note below about precedence order between track_return_cycles and track_absorptions.
    - exit_times: list with the observed exit times from the absorption set since the latest reabsorption.
    - absorption_times: list with the observed reabsorption times.

    If track_absorptions = True and track_survival = True, in addition:
    - survival_times: list with the observed survival times, sorted increasingly.

    Note that:
    - either track_return_cycles or track_absorptions may be True. If both are True, the first one has precedence.
    - track_survival = True has no effect if track_absorptions = False.
    """
    # -- Parse input parameters
    t_max = t_sim_max

    # Seed
    if seed is not None:
        env.set_seed(seed)

    # Set the start state of the environment to the given start state
    env.setState(start_state)
    buffer_size_start = env.getBufferSize()
    if verbose:
        print("MC simulation: The queue environments starts at state {} (buffer size = {})" \
              .format(env.getState(), buffer_size_start))

    # Store the initial position as part of the trajectory
    update_trajectory(agent, (t_learn - 1) * (t_max + 1) + 0, 0.0, env.getState(), None, 0.0)

    # Time step in the queue trajectory (the first time step is t = 0)
    done = False
    t = 0
    t_arrivals = 0  # Number of arrival events. Used to decide the end of the simulation, when track_absorptions = True
    time_abs = 0.0  # Measure of the ABSOLUTE time of the latest event (as opposed to the time relative to the latest event)
    # Information about the return cycles
    n_cycles_return = 0
    n_exit_times = 0
    n_cycles_absorption = 0
    time_last_return = 0.0
    time_last_absorption = 0.0
    # Information about the return times to the initial state/buffer-size
    return_times = []
    # Information about the exit times (from the absorption set A)
    exit_times = []
    # Information about the reabsorption times
    absorption_times = []
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
        jobs_arrival = []               # Class of the arrived jobs
        times_inter_arrival = []
        servers_service = []            # Server number at which each job is served
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

            # Check ACTIVATION / EXIT from absorption set A: we touch a state in the set of activation states
            # COMING FROM the set of absorption states
            # Note that the condition "COMING FROM the set of absorption states" is important because it may happen that:
            # - there is a BIRTH event but the state does not change because the incoming job is REJECTED!
            #   In an unidimensional-state setting, this may happen when the blocking size K is very close to the absorption set A
            #   (e.g. when the absorption set size J = K, or when the acceptance policy is not deterministic making it
            #   capable of rejecting an incoming job when x < K (for instance when x = K-1 and J = K-1))
            # - in a multidimensional-state setting, there is a BIRTH event, the state changes but WITHOUT coming from
            #   the set of absorption states (e.g. if the absorption set is made up of the boundary of the hypercube
            #   [4, 2, 5], when the state changes from (4, 2, 6) to (4, 3, 6), the state (4, 3, 6) is in the set of
            #   activation states (because the activation set is defined by the boundaries [5, 3, 6] (= absorption boundaries + 1)
            #   but the previous state (4, 2, 6) was ALSO in the set of activation states (as it suffices to have at least ONE
            #   component in the boundary of the activation set to be in the set).
            #   NOTE, however, that the definition we are dealing with for the activation set is NOT the correct one, because
            #   precisely because of what I just described, there are states in the activation set that are NOT really
            #   activation states, because in order to be an activation state the system should be able to get to it
            #   from on the absorption states --which is not the case for e.g. (4, 2, 6) because MORE THAN ONE component
            #   is already at the boundary of the activation set (first and third components).
            # Note that the state of the environment has been updated by the manage_service() function, that's why env.getQueueState() returns the value of `next_state` above
            assert env.getQueueStateFromState(next_state) == env.getQueueState()
            if is_state_in_setofstates(env.getQueueStateFromState(state), absorption_set) and is_state_in_setofstates(env.getQueueState(), activation_set):
                # Record the exit times, i.e. time elapsed since the latest absorption
                n_exit_times += 1
                exit_times += [time_abs - time_last_absorption]
                if False and DEBUG_ESTIMATORS:
                    print("[MC] --> EXIT time = {:.3f} (total={})".format(exit_times[-1], n_exit_times))
                if track_survival:
                    time_last_activation = time_abs
                    if False and DEBUG_ESTIMATORS:
                        print("[MC] --> ACTIVATION @time = {:.3f}".format(time_abs))

            if DEBUG_ESTIMATORS:
                # TODO: (2022/10/16) Fix this so that the inter-arrival rates are recorded correctly, i.e. separately for EACH JOB CLASS
                # Currently, this is not the case, as the inter-arrival times are computed w.r.t. the last time of arrival of ANY JOB CLASS.
                # In order to fix this, we need to define the time_last_arrival as a list with as many elements as job classes.
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
            # Note that the state of the environment has been updated by the manage_service() function, that's why env.getQueueState() returns the value of `next_state` above
            assert env.getQueueStateFromState(next_state) == env.getQueueState()
            if is_state_in_setofstates(env.getQueueStateFromState(state), activation_set) and is_state_in_setofstates(env.getQueueState(), absorption_set):
                n_cycles_absorption += 1
                absorption_times += [time_abs - time_last_absorption]
                if False and DEBUG_ESTIMATORS:
                    print("[MC] --> ABSORPTION time = {:.3f} (total={})".format(absorption_times[-1], n_cycles_absorption))
                time_last_absorption = time_abs
                if track_survival and time_last_activation is not None:
                    killing_time = time_abs - time_last_activation
                    if DEBUG_ESTIMATORS:
                        print("[MC] --> ABSORPTION @time = {:.3f}".format(time_abs))
                        print("[MC] \tACTIVATION time = {:.3f} --> time to absorption = {:.3f}" \
                              .format(time_last_activation, killing_time))
                    assert killing_time > 0
                    # We sort the survival times at the end of the process, as it might be faster than inserting
                    # the value in order at every time being observed (although the difference apparently is just
                    # in the constant of proportionality of the O(N*log(N)) complexity.
                    # Ref: https://stackoverflow.com/questions/168891/is-it-faster-to-sort-a-list-after-inserting-items-or-adding-them-to-a-sorted-lis
                    survival_times += [killing_time]

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
        update_trajectory(agent, (t_learn - 1) * (t_max + 1) + t, time_abs, next_state, action, reward)

        # Check RETURN
        # Either the initial buffer size (e.g. single-buffer queue systems) OR
        # the initial state (for e.g. loss networks with multi-class jobs)
        # is observed again.
        if env.getBufferType() == BufferType.SINGLE:
            # Note that, by checking whether the system returns to the initial BUFFER SIZE --as opposed to the initial STATE--
            # we are defining the stopping time to be the time to fulfill that condition, and we define it as the return time
            # to the initial buffer size. In particular, note that there is NO Violation of the applicability of the renewal theorem
            # on which the estimation of the blocking probability is based (as presented in e.g. Asmussen's book), because for the theorem
            # to apply, it suffices that the cycle is defined by a *stopping* time, and the return time to the initial buffer size
            # classifies as a stopping time.
            if env.getBufferSizeFromState(next_state) == buffer_size_start:
                n_cycles_return += 1
                return_times += [time_abs - time_last_return]
                time_last_return = time_abs
        else:
            # Multidimensional state case
            if next_state == start_state:
                n_cycles_return += 1
                return_times += [time_abs - time_last_return]
                time_last_return = time_abs

        if DEBUG_TRAJECTORIES:
            print("[MC] {} | t={}: time={:.3f}, event={}, action={} -> state={}, reward={:.3f}" \
                  .format(state, t, time_abs, event, action, next_state, reward), end="\n")

        if track_absorptions:
            # Tracking absorptions means that we are running this simulation for the FV estimation of E(T_A) and perhaps P(T>t) as well
            # => We need to stop when a certain number of ARRIVALS have occurred
            # (because that's how the required simulation time has been determined in order to guarantee a maximum
            # relative error of the estimation of E(T_A) --see compute_nparticles_and_narrivals_for_fv_process())
            done = check_done(t_max, t_arrivals, state, action, reward)
        else:
            done = check_done(t_max, t, state, action, reward)

    # DONE
    if show_messages(verbose, verbose_period, t_learn):
        print("[MC] ==> agent ENDS at time t={} at state {} coming from state = {}, action = {}, reward = {}, gradient = {})" \
              .format(t, env.getState(), state, action, reward, gradient_for_action))

    if DEBUG_ESTIMATORS:
        # Distribution of inter-arrival and service times and their mean
        # NOTE: (2022/10/16) This only works when there is only ONE server in the system
        # (because the way `job_rates_by_server` is used in analyze_event_times() assumes this.
        job_assignment_policy = agent.getAssignmentPolicy() if agent.getAssignmentPolicy() is not None \
                                else define_uniform_job_assignment_policy(len(env.getJobClassRates()), env.getNumServers())
        job_rates_by_server = compute_job_rates_by_server(env.getJobClassRates(),
                                                          env.getNumServers(),
                                                          job_assignment_policy.getProbabilisticMap())
        analyze_event_times(job_rates_by_server, times_inter_arrival, jobs_arrival, group_name="Job class")
        analyze_event_times(env.getServiceRates(), times_service, servers_service, group_name="Server")

        # Distribution of return or absorption cycle times, whatever is tracked
        if track_return_cycles:
            type_of_times = "Return"
            x_n = n_cycles_return
            x_values = return_times
            x_total = time_last_return
        elif track_absorptions:
            type_of_times = "Re-absorption"
            x_n = n_cycles_absorption
            x_values = absorption_times
            x_total = time_last_absorption
        results_str = "{} times (initial state = {}, initial buffer size = {})\nn = {}, mean = {:.3f}, SE = {:.3f}" \
                      .format(type_of_times, start_state, buffer_size_start, x_n, np.mean(x_values), np.std(x_values) / np.sqrt(x_n))
        assert len(x_values) == x_n
        assert np.isclose(x_total / x_n, np.mean(x_values))
        fig = plt.figure()
        plt.hist(x_values, bins=30, color="red")
        axes = fig.get_axes()
        #axes[0].set_xscale('log')      # Use this in case we want to see a potential outlier at low values
        axes[0].set_title(results_str)

    if track_return_cycles:
        return t, time_abs, n_cycles_return, time_last_return, return_times
    if track_absorptions:
        if not track_survival:
            return t, time_abs, n_cycles_absorption, time_last_absorption, exit_times, absorption_times
        else:
            return t, time_abs, n_cycles_absorption, time_last_absorption, exit_times, absorption_times, sorted(survival_times)
    return t, time_abs


def deprecated_estimate_stationary_probabilities_mc(env, agent, burnin_time=0, min_num_cycles_for_expectations=1):
    """
    Monte-Carlo estimation of the stationary probability at the buffer sizes of interest
    (defined by the agent's job acceptance policy) from the observed trajectory in continuous time
    (stored in the agent's learner).

    The stationary probability is estimated as the fraction of time spent at each buffer size of interest
    over the sum of cycle times of returning to the buffer size associated to the state at which the Markov process
    is found after removal of a possibly non-zero burn-in time from the beginning of the simulation.
    If the burn-in time is 0, then that buffer size is associated to the state at which the Markov process started.

    Arguments:
    env: environment
        The queue environment where the agent acts.

    agent: Agent
        The agent interacting with the environment in terms of job acceptance/rejection.
        It should have a learner of the value function V and a learner of the policy P defined.
        It should also store the observed trajectory while exploring the environment from a simulation already run
        (which is used to estimate the stationary probabilities).

    burnin_time: float
        Continuous time to be used for burn-in, i.e. in which the process is assumed to still not be stationary,
        and therefore the observed event times in that initial period should be excluded.

    min_num_cycles_for_expectations: int
        Minimum number of observed cycles in order to consider the estimation of the stationary probability, using
        renewal theory, reliable.

    Return: tuple
    Tuple with the following elements:
    - probas_stationary: Dictionary with the estimated stationary probability for the states or buffer sizes of interest,
    typically those where blocking may occur (deterministically or probabilistically) according to the agent's job acceptance policy.
    - expected_cycle_time: the average observed cycle time of return to the *first* buffer size visited by the Markov
    process *after* the burn-in time. This coincides with the expected return time to the buffer size associated
    to the state at which the Markov process started if burnin_time = 0.
    - n_cycles: number of cycles used to compute the expected cycle time.
    """

    # -- Auxiliary functions
    def get_events_after_burnin(agent, burnin_time):
        # Event times and states (BEFORE the event occurs)
        times = agent.getLearnerV().getTimes()
        states = agent.getLearnerV().getStates()

        # Remove any times and states occurring before the burn-in time
        idx_first_after_burnin = -1
        for idx, t in enumerate(times):
            if t >= burnin_time:
                idx_first_after_burnin = idx
                break

        if burnin_time == 0.0:
            assert idx_first_after_burnin == 0

        if idx_first_after_burnin == -1:
            return idx_first_after_burnin, [], [], []

        # Lists containing the event times, the states before those event times and
        # the buffer sizes associated to those states happening AFTER the burn-in time
        times_after_burnin = times[idx_first_after_burnin:]
        states_after_burnin = states[idx_first_after_burnin:]
        buffer_sizes_after_burnin = [env.getBufferSizeFromState(x) for x in states_after_burnin]

        return idx_first_after_burnin, times_after_burnin, states_after_burnin, buffer_sizes_after_burnin
    # -- Auxiliary functions

    # Initialize the output objects to NaN in case their values cannot be reliably estimated
    # (because of an insufficient number of observed return cycles)
    buffer_sizes_of_interest = get_blocking_boundaries(agent)
    probas_stationary = dict()
    for bs in buffer_sizes_of_interest:
        probas_stationary[bs] = np.nan
    expected_cycle_time = np.nan

    idx_first_after_burnin, times_after_burnin, states_after_burnin, buffer_sizes_after_burnin = get_events_after_burnin(agent, burnin_time)
    if idx_first_after_burnin == -1:
        warnings.warn(  "No events were observed after the initial burn-in time ({:.3f})," \
                        " therefore stationary probability estimates and the expected cycle time are set to NaN." \
                        "\nTry increasing the simulation time.".format(burnin_time))
        n_cycles = 0
        return probas_stationary, expected_cycle_time, n_cycles

    assert  len(times_after_burnin) > 0 and \
            len(states_after_burnin) == len(times_after_burnin) and len(buffer_sizes_after_burnin) == len(times_after_burnin)

    # Compute the number of cycles to be used for the computation of the stationary probabilities
    # We call the buffer size that defines these cycles the ANCHOR buffer size.
    buffer_size_anchor = buffer_sizes_after_burnin[0]

    # Compute the number of cycles observed after the burn-in time in order to decide whether stationary probabilities
    # can be reliably computed (based on the minimum requirement for the number of observed cycles defined as parameter)
    # Ex: buffer_sizes_after_burnin = [1, 2, 1, 2, 3, 1, 0] => buffer_size_anchor = 1 (as this is the first element in the list) =>
    # idx_buffer_size_at_anchor = [0, 2, 5] => n_cycles = len([0, 2, 5]) - 1 = 2 (OK, as there are two return cycles to 1 in buffer_sizes_after_burnin)
    idx_buffer_size_at_anchor = [idx for idx, bs in enumerate(buffer_sizes_after_burnin) if bs == buffer_size_anchor]
    n_cycles = len(idx_buffer_size_at_anchor) - 1   # We subtract 1 because the first value in buffer_sizes_after_burnin is the anchor buffer size (see example above)
    if n_cycles < min_num_cycles_for_expectations:
        warnings.warn("There aren't enough return cycles experienced by the Markov process after the burn-in time ({:.3f})".format(burnin_time) +
                      " in order to obtain a reliable estimation of the stationary probabilities nor of the expected cycle time, which are therefore set to NaN." +
                      "\nThe number of observed cycles is {}, while the minimum required is {}.".format(n_cycles, min_num_cycles_for_expectations))
        return probas_stationary, expected_cycle_time, n_cycles

    # Compute the total cycle times of return to the anchor buffer size and the total sojourn times at each buffer size
    # of interest, whose ratio gives the estimated stationary probability of the respective buffer size of interest.
    # Note: each sojourn time is the time the Markov process sojourned at each state stored in the `states_after_burnin` list.
    # This is so because the first state in `states_after_burnin` is the state where the Markov process was
    # PRIOR to the occurrence of the event happening at the first time stored in `times_after_burnin`
    # (this is a direct consequence of the fact that the first state stored for the Markov process is the state at which
    # the Markov process was initialized, and the first time stored for the Markov process is 0.0, meaning that the
    # first non-zero event time stored is the time at which the Markov process changes state for the first time
    # --to the state stored as the second position in the list).
    # This is why the first element in `sojourn_times` is the difference between the first two elements in `times_after_burnin`.
    idx_last_visit_to_anchor_buffer_size = idx_buffer_size_at_anchor[-1]
    buffer_sizes_until_end_of_last_cycle = buffer_sizes_after_burnin[:idx_last_visit_to_anchor_buffer_size + 1]
    times_until_end_of_last_cycle = times_after_burnin[:idx_last_visit_to_anchor_buffer_size + 1]
    total_cycle_time = times_until_end_of_last_cycle[-1] - times_until_end_of_last_cycle[0]
    sojourn_times_until_end_of_last_cycle = np.diff( times_until_end_of_last_cycle )
    assert len(sojourn_times_until_end_of_last_cycle) == idx_last_visit_to_anchor_buffer_size

    # Compute the total sojourn time at each buffer size of interest and then the corresponding stationary probability
    for bs in buffer_sizes_of_interest:
        total_sojourn_time_at_bs = np.sum([st for x, st in zip(buffer_sizes_until_end_of_last_cycle, sojourn_times_until_end_of_last_cycle)
                                           if x == bs])
        assert total_sojourn_time_at_bs <= total_cycle_time, \
            "The total sojourn time at the buffer size of interest (e.g. blocking) bs={} ({:.3f})" \
            " is at most equal to the total cycle time of return to the anchor buffer size ({:.3f})," \
            " i.e. to the total time of the multiple return cycles experienced by the Markov process" \
            .format(bs, total_sojourn_time_at_bs, total_cycle_time)
        probas_stationary[bs] = total_sojourn_time_at_bs / total_cycle_time

    # Compute the expected cycle time
    expected_cycle_time = total_cycle_time / n_cycles

    return probas_stationary, expected_cycle_time, n_cycles


def estimate_stationary_probabilities_mc(env, agent, burnin_time=0, min_num_cycles_for_expectations=1):
    """
    Monte-Carlo estimation of the stationary probability at states of interest (defined by the agent's job acceptance policy)
    from the observed trajectory in continuous time (stored in the agent's learner).

    The stationary probability is estimated as the fraction of time spent at each state of interest
    over the sum of cycle times of returning to the state at which the Markov process is found to be
    after removal of a possibly non-zero burn-in time from the beginning of the simulation.
    If the burn-in time is 0, then that state is the state at which the Markov process started.

    Arguments:
    env: environment
        The queue environment where the agent acts.

    agent: Agent
        The agent interacting with the environment in terms of job acceptance/rejection.
        It should have a learner of the value function V and a learner of the policy P defined.
        It should also store the observed trajectory while exploring the environment from a simulation already run
        (which is used to estimate the stationary probabilities).

    burnin_time: float
        Continuous time to be used for burn-in, i.e. in which the process is assumed to still not be stationary,
        and therefore the observed event times in that initial period should be excluded.

    min_num_cycles_for_expectations: int
        Minimum number of observed cycles in order to consider the estimation of the stationary probability, using
        renewal theory, reliable.

    Return: tuple
    Tuple with the following elements:
    - probas_stationary: Dictionary with the estimated stationary probability for the states of interest,
    typically those where blocking may occur (deterministically or non-deterministically) according to the agent's job
    acceptance policy.
    - expected_cycle_time: the average observed cycle time of return to the *first* state visited by the Markov
    process *after* the burn-in time. This coincides with the expected return time to the state at which the Markov
    process started if burnin_time = 0.
    - n_cycles: number of cycles used to compute the expected cycle time.
    """

    # -- Auxiliary functions
    def get_events_after_burnin(env, agent, burnin_time):
        "Returns the times and queue states (events) after the burn-in time"
        # Event times and states (BEFORE the event occurs)
        times = agent.getLearnerV().getTimes()
        states = agent.getLearnerV().getStates()

        # Remove any times and states occurring before the burn-in time
        idx_first_after_burnin = -1
        for idx, t in enumerate(times):
            if t >= burnin_time:
                idx_first_after_burnin = idx
                break

        if burnin_time == 0.0:
            assert idx_first_after_burnin == 0

        if idx_first_after_burnin == -1:
            return idx_first_after_burnin, [], []

        # Lists containing the event times, the states before those event times and
        # the buffer sizes associated to those states happening AFTER the burn-in time
        times_after_burnin = times[idx_first_after_burnin:]
        states_after_burnin = [env.getQueueStateFromState(s) for s in states[idx_first_after_burnin:]]

        return idx_first_after_burnin, times_after_burnin, states_after_burnin

    def get_states_and_their_sojourn_times_until_end_of_last_cycle(indices_return_times: list, states: list, times: list):
        idx_last_return_time = indices_return_times[-1]
        states_or_buffer_sizes_until_end_of_last_cycle = states[:idx_last_return_time + 1]
        times_until_end_of_last_cycle = times[:idx_last_return_time + 1]
        sojourn_times_until_end_of_last_cycle = np.diff(times_until_end_of_last_cycle)
        assert len(sojourn_times_until_end_of_last_cycle) == idx_last_return_time
        
        # Compute the total cycle time which is used as denominator of the stationary probability estimation
        total_cycle_time = times_until_end_of_last_cycle[-1] - times_until_end_of_last_cycle[0]

        return states_or_buffer_sizes_until_end_of_last_cycle, sojourn_times_until_end_of_last_cycle, total_cycle_time
    # -- Auxiliary functions

    # Get the states or buffer sizes on which the stationary probability should be estimated
    states_or_buffer_sizes_of_interest = get_blocking_states_or_buffer_sizes(env, agent)

    # Initialize the output objects to NaN in case their values cannot be reliably estimated
    # (because of an insufficient number of observed return cycles)
    probas_stationary = dict()
    for s in states_or_buffer_sizes_of_interest:
        probas_stationary[s] = np.nan
    expected_cycle_time = np.nan

    idx_first_after_burnin, times_after_burnin, states_after_burnin = get_events_after_burnin(env, agent, burnin_time)
    if idx_first_after_burnin == -1:
        warnings.warn(  "No events were observed after the initial burn-in time ({:.3f})," \
                        " therefore stationary probability estimates and the expected cycle time are set to NaN." \
                        "\nTry increasing the simulation time.".format(burnin_time))
        n_cycles = 0
        return probas_stationary, expected_cycle_time, n_cycles

    assert  len(times_after_burnin) > 0 and len(states_after_burnin) == len(times_after_burnin)

    # Compute the number of cycles to be used for the computation of the stationary probabilities
    # We call the state or buffer size that defines these cycles the ANCHOR state or buffer size.
    if env.getBufferType() == BufferType.SINGLE:
        _buffer_sizes_after_burnin = [np.sum(x) for x in states_after_burnin]
        _anchor_buffer_size = _buffer_sizes_after_burnin[0]
        idx_anchor_state_or_buffer_size = [idx for idx, bs in enumerate(_buffer_sizes_after_burnin) if bs == _anchor_buffer_size]
    else:
        _anchor_state = states_after_burnin[0]
        idx_anchor_state_or_buffer_size = [idx for idx, s in enumerate(states_after_burnin) if s == _anchor_state]

    # Compute the number of cycles observed after the burn-in time in order to decide whether stationary probabilities
    # can be reliably computed (based on the minimum requirement for the number of observed cycles defined as parameter)
    # Examples:
    # 1) Single-buffer case:
    # buffer_sizes_after_burnin = [1, 2, 1, 2, 3, 1, 0] => buffer_size_anchor = 1 (as this is the first element in the list) =>
    # idx_buffer_size_at_anchor = [0, 2, 5] => n_cycles = len([0, 2, 5]) - 1 = 2 (OK, as there are two return cycles to 1 in buffer_sizes_after_burnin)
    # 2) No-buffer case: multidimensional state case
    # states_after_bunin = [(1, 2, 3), (1, 2, 4), (2, 2, 4), (2, 2, 3), (1, 2, 3), (0, 2, 3)] => anchor_state = (1, 2, 3)
    # (as this is the first element in the list) => idx_anchor_states = [0, 4] => n_cycles = len([0, 4]) - 1 = 1
    n_cycles = len(idx_anchor_state_or_buffer_size) - 1   # We subtract 1 because the first value in states_after_burnin is the anchor state (see example above)
    if n_cycles < min_num_cycles_for_expectations:
        warnings.warn("There aren't enough return cycles experienced by the Markov process after the burn-in time ({:.3f})".format(burnin_time) +
                      " in order to obtain a reliable estimation of the stationary probabilities nor of the expected cycle time, which are therefore set to NaN." +
                      "\nThe number of observed cycles is {}, while the minimum required is {}.".format(n_cycles, min_num_cycles_for_expectations))
        return probas_stationary, expected_cycle_time, n_cycles

    #-- Compute the necessary measures needed to compute the stationary probability of every state or buffer size of interest, namely:
    # - the total cycle times of return to the anchor state or buffer size
    # - the total sojourn times at each state or buffer size
    # of interest, whose ratio gives the estimated stationary probability of the respective state or buffer size of interest.
    # Note: each sojourn time is the time the Markov process sojourned at each state stored in the `states_after_burnin` list.
    # This is so because the first state in `states_after_burnin` is the state where the Markov process was
    # PRIOR to the occurrence of the event happening at the first time stored in `times_after_burnin`
    # (this is a direct consequence of the fact that the first state stored for the Markov process is the state at which
    # the Markov process was initialized, and the first time stored for the Markov process is 0.0, meaning that the
    # first non-zero event time stored is the time at which the Markov process changes state for the first time
    # --to the state stored as the second position in the list).
    # This is why the first element in `sojourn_times` is the difference between the first two elements in `times_after_burnin`.
    states_or_buffer_sizes_until_end_of_last_cycle,\
        sojourn_times_until_end_of_last_cycle, \
            total_cycle_time = get_states_and_their_sojourn_times_until_end_of_last_cycle(
                                                                                        idx_anchor_state_or_buffer_size,
                                                                                        states_after_burnin,
                                                                                        times_after_burnin)

    # Compute the total sojourn time at each state or buffer size of interest and the corresponding stationary probability
    for s in states_or_buffer_sizes_of_interest:
        total_sojourn_time_at_state_or_buffer_size_of_interest =\
            np.sum([st for x, st in zip(states_or_buffer_sizes_until_end_of_last_cycle, sojourn_times_until_end_of_last_cycle)
                    if x == s])
        assert total_sojourn_time_at_state_or_buffer_size_of_interest <= total_cycle_time, \
            "The total sojourn time at the state or buffer size of interest s={} ({:.3f})" \
            " is at most equal to the total cycle time of return to the anchor state or buffer size ({:.3f})," \
            " i.e. to the total time of the multiple return cycles experienced by the Markov process" \
            .format(s, total_sojourn_time_at_state_or_buffer_size_of_interest, total_cycle_time)
        probas_stationary[s] = total_sojourn_time_at_state_or_buffer_size_of_interest / total_cycle_time

    # Compute the expected cycle time
    expected_cycle_time = total_cycle_time / n_cycles

    return probas_stationary, expected_cycle_time, n_cycles


def get_blocking_states_or_buffer_sizes(env, agent):
    """
    Returns the states or buffer sizes in the environment that can potentially (because blocking can be non-deterministic)
    generate blocking based on the agent's accept/reject policy or policies.

    Arguments:
    env: Environment
        Queue environment.

    agent: Agent
        Accept/Reject agent acting on the environment.
        It should be an agent accepted by the get_blocking_boundaries() function.

    Return: list of int or list of tuples
    List containing the states (tuples) (for multidimensional-state blocking) or the buffer sizes (int)
    (for single-buffer blocking) that can potentially generate blocking of the system.
    """
    # Output variable
    states_of_interest = []

    blocking_boundaries = get_blocking_boundaries(agent)
    deterministic_blocking_boundaries = get_deterministic_blocking_boundaries(agent)
    n_policies = len(blocking_boundaries)
    system_capacity = env.getCapacity()

    if env.getBufferType() == BufferType.SINGLE:
        # The states or buffer sizes of interest are *buffer sizes* (i.e. scalar values) and are directly the blocking boundaries (e.g. [K-1, K])
        buffer_sizes_of_interest = blocking_boundaries
        return buffer_sizes_of_interest
    else:
        # The states or buffer sizes of interest are multidimensional states of the given environment
        # Ex: the environment is a multi-server system with separate queues or a loss network, receiving multi-class jobs.
        # This means that the following states are of interest:
        #
        # A) Any valid state having one of the state dimensions equal to the corresponding state boundary
        # (where the state boundary is defined by the agent as the blocking size for that dimension)
        # is a blocking state.
        # Ex: if the state is 3D, and the first state boundary is 4, then the states (4, s2, s3) for all s2 and s3 that
        # make the state (4, s2, s3) a valid state based on the system's capacity AND the agent's acceptance policy
        # (e.g. even if the system's capacity is 10, the agent may block an incoming job when s2 = 3 (this happens when
        # the state boundary of dimension 2 is 3).
        #
        # B) Any state whose sum of its elements is equal to the system's capacity
        assert not np.isinf(system_capacity), "The system's capacity must be finite for non-single-buffer systems"

        # Start with the states in case (A)
        count_blocking_states_condA = 0
        for idx_policy, boundaries in enumerate(blocking_boundaries):       # E.g. blocking_boundaries = [[3, 4], [7, 8], [1, 2]] (i.e. [[K0-1,K0], [K1-1,K1], [K2-1,K2]]) which corresponds to a theta policy parameter that is 3D
            for b in boundaries:                                            # E.g. b = 3 when boundaries = [3, 4]
                # We start by setting the currently processed state dimension (defined by idx_policy) to a blocking state value
                # (this avoids going over states where no blocking occurs)
                state_of_interest_new = [-1] * n_policies                   # E.g. [-1, -1, -1] if the theta policy parameter of the problem is 3D. -1 is the value we use to indicate missing.
                state_of_interest_new[idx_policy] = min(b, system_capacity) # E.g. idx_policy = 1 and system_capacity > 3 => states_of_interest_new is initialized to [-1, 3, -1]

                # Fill in the other dimensions with all possible state dimension values that make their sum be <= capacity - b
                for sum_values in range(system_capacity - b + 1):           # E.g. system_capacity = 10, b = 3 => sum_value in [0, 1, ..., 7]
                    combos_generator = all_combos_with_sum(n_policies - 1, sum_values)
                    while True:
                        try:
                            values = next(combos_generator)                 # E.g. n_policies = 3, sum_values = 5 => values = [4, 1]
                            # Store the values just generated in the states_of_interest_new list
                            indices_to_fill = list(range(idx_policy)) + list(range(idx_policy + 1, n_policies))  # E.g. idx_policy = 0, n_policies = 3 => indices_to_fill = [1, 2]; if idx_policy = 1, n_policies = 3 => indices_to_fill = [2]
                            assert len(values) == len(indices_to_fill)
                            blocking_boundaries_in_other_dimensions = blocking_boundaries.copy()
                            blocking_boundaries_in_other_dimensions.pop(idx_policy)
                            if all(np.array(values) <= [max(bb) for bb in blocking_boundaries_in_other_dimensions]):
                                # Only add the state associated to `values` if ALL the occupancy levels in `values` are at most equal to maximum blocking size defined by the policy.
                                # Otherwise the state is not a feasible state given the policy.
                                # Note that we need to compute max() on the boundaries present in blocking_boundaries above
                                # because recall that each element of blocking_boundaries contains ALL *valid* (this is why we take the max())
                                # blocking values for the respective dimension.
                                for i, p in enumerate(indices_to_fill):     # i for "index" in values; p for "policy" index in states_of_interest_new
                                    state_of_interest_new[p] = values[i]

                                if tuple(state_of_interest_new) not in states_of_interest:
                                    # NOTES:
                                    # - We enclose state_of_interest_new in brackets so that states_of_interest becomes a list of tuples
                                    # - We convert the state of interest added to `tuple` because these states are commonly used to index dictionaries
                                    # and only tuples can be used as keys of dictionaries, NOT lists because they are mutable.
                                    states_of_interest += [tuple(state_of_interest_new)]
                                    count_blocking_states_condA += 1
                        except StopIteration:
                            break
                    combos_generator.close()

        # Now add the states in case (B)
        set_of_states_to_check = SetOfStates(set_boundaries=deterministic_blocking_boundaries).getStates()
        for s in set_of_states_to_check:
            if np.sum(s) == system_capacity:
                # Note that we may be adding states that were already added above, but below we take the set() of the generated list of states of interest
                states_of_interest += [tuple(s)]

        states_of_interest = sorted(set(states_of_interest))
        print(f"states_of_interest {len(states_of_interest)}:\n {states_of_interest}")

        return states_of_interest


def estimate_blocking_fv(envs, agent, dict_params_simul, dict_params_info, probas_stationary: dict=None):
    """
    Estimates the blocking probability using the Fleming-Viot approach

    Arguments:
    envs: List
        List of queue environments used to run the FV process.

    agent: Agent
        Agent interacting with each of the environments given in envs.

    dict_params_simul: dict
        Dictionary containing simulation and estimation parameters.
        The dictionary should contain the keys described in function parse_simulation_parameters()`.

    dict_params_info: dict
        Dictionary containing information to display or parameters to deal with the information to display.
        Accepted keys are:
        - verbose: whether to be verbose during the simulation.
        - verbose_period: the number of iterations (of what?) at which to be verbose.
        - t_learn: the number of learning steps, when FV is used in the context of FVRL, i.e. to learn an optimum policy.

    probas_stationary: dict
        Stationary distribution that should be used to select the start state of each FV particle.
        default: None, in which case a uniform random distribution is used.

    Return: tuple
    Tuple with the following elements:
    - proba_blocking: the estimated blocking probability.
    - expected_reward: the estimated expected reward.
    - probas_stationary_blocking: dictionary with the estimated stationary probability for each buffer size where blocking
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

    # -- Auxiliary functions
    is_estimation_of_denominator_unreliable = lambda: track_survival_in_single_particle_simulation and np.isnan(expected_absorption_time) or not track_survival_in_single_particle_simulation and np.isnan(expected_exit_time)
    # -- Auxiliary functions

    # -- Parse input parameters
    dict_params_simul = parse_simulation_parameters(dict_params_simul, envs[0])

    # Set the simulation seed
    # Note that, although the seed is set by the queue environment (via its set_seed() method which is called by run_simulation_mc() below)
    # we also need to set the seed here because, before calling run_simulation_mc(), we choose below the initial queue state
    # on which the simulation starts randomly, among all possible states in the absorption set.
    np.random.seed(dict_params_simul['seed'])

    # Reset environment and learner of the value functions
    # IMPORTANT: We should NOT reset the learner of the policy because this function could be called as part of a
    # policy learning process! Note that the learner of the value function V MUST be reset regardless of this because
    # at this point a new queue simulation starts and at that point the value function should be learned anew.
    for env in envs:
        env.reset()
    if agent.getLearnerV() is not None:
        agent.getLearnerV().reset()
    # -- Parse input parameters

    # -- Step 1: Simulate a single queue to estimate P(T>t) and E(T_A)
    # The start state is chosen among a set of absorption states
    # Note that the set of absorption states is easy to define in the single-server context but in a multi-server context
    # the set may be very large... I don't think there is an alternative for a concise definition of the absorption set
    # in such multi-server situation.
    # In a loss network context, e.g. a buffer-less server system that accepts jobs of different classes,
    # this absorption set can concisely be defined by the set of unidimensional blocking sizes, one for each job class,
    # and this is one of the options given by the constructor of the SetOfStates class used to define this set.
    if probas_stationary is None:
        start_queue_state_boundary_A = dict_params_simul['absorption_set'].random_choice()
    else:
        # Sample the start state using the given distribution of the states
        start_queue_state_boundary_A = choose_state_from_setofstates_based_on_distribution(dict_params_simul['absorption_set'],
                                                                                           probas_stationary,
                                                                                           at_least_one_dimension_at_boundary=True)
    if dict_params_simul['method_survival_probability_estimation'] == SurvivalProbabilityEstimation.FROM_M_CYCLES:
        track_survival_in_single_particle_simulation = True
        t, time_end_simulation, n_cycles_absorption, time_last_absorption, exit_times, absorption_times, survival_times = \
            run_simulation_mc(envs[0], agent, dict_params_info.get('t_learn', 0), (start_queue_state_boundary_A, None),
                              dict_params_simul['T'],
                              absorption_set=dict_params_simul['absorption_set'],
                              activation_set=dict_params_simul['activation_set'],
                              track_absorptions=True, track_survival=track_survival_in_single_particle_simulation,
                              seed=dict_params_simul['seed'],
                              verbose=dict_params_info.get('verbose', False),
                              verbose_period=dict_params_info.get('verbose_period', 1))
    else:
        track_survival_in_single_particle_simulation = False
        t, time_end_simulation, n_cycles_absorption, time_last_absorption, exit_times, absorption_times = \
            run_simulation_mc(envs[0], agent, dict_params_info.get('t_learn', 0), (start_queue_state_boundary_A, None),
                              dict_params_simul['T'],
                              absorption_set=dict_params_simul['absorption_set'],
                              activation_set=dict_params_simul['activation_set'],
                              track_absorptions=True, track_survival=track_survival_in_single_particle_simulation,
                              seed=dict_params_simul['seed'],
                              verbose=dict_params_info.get('verbose', False),
                              verbose_period=dict_params_info.get('verbose_period', 1))
    n_events_et = t
    time_end_simulation_et = time_end_simulation

    if track_survival_in_single_particle_simulation:
        # Estimate E(T_A) and P(T>t) because both estimates come from the same simulation and are based on the
        # M reabsorption cycles observed therein. In fact, this guarantees that estimated E(T_A) >= estimated E(T_K),
        # where T_K is the killing time a.k.a. absorption time since activation, a requirement that is needed to make
        # calculations consistent as E(T_A) = E(T_E) + E(T_K), where T_E is the exit time from the absorption set.
        # The condition is also used to prove convergence of the FV estimator (see our paper on Fleming-Viot).
        expected_exit_time = None
        expected_absorption_time, n_cycles_absorption_used = estimate_expected_cycle_time(n_cycles_absorption, time_last_absorption,
                                                                                          cycle_times=absorption_times, burnin_time=dict_params_simul['burnin_time'],
                                                                                          min_num_cycles_for_expectations=dict_params_simul['min_num_cycles_for_expectations'])
        df_proba_surv = compute_survival_probability(survival_times)
        max_survival_time = np.max(survival_times)
    else:
        # Both E(T_A) and P(T>t) will be estimated once we have completed the FV simulation, as follows:
        # - The survival probability will be estimated from the first absorption times observed for each of the N particles used in the FV simulation.
        # - E(T_A) will be estimated as the same of E(T_E), the expected exit time (from the absorption set) +
        # the expected killing (re-absorption) time, after the system exists from the absorption set.
        # This is a little better estimator than estimating E(T_A) directly as the average reabsorption cycle time because
        # in this way we guarantee that E(T_A) is larger than the expected exit time from the absorption set, E(T_E).
        expected_exit_time, n_exit_times_used = estimate_expected_stopping_time_in_cycle(exit_times, absorption_times,
                                                                            burnin_time=dict_params_simul['burnin_time'],
                                                                            min_num_cycles_for_expectations=dict_params_simul['min_num_cycles_for_expectations'])
        expected_absorption_time = None
        n_cycles_absorption_used = n_exit_times_used
        df_proba_surv = None

    if DEBUG_ESTIMATORS or show_messages(dict_params_info.get('verbose', False), dict_params_info.get('verbose_period', 1), dict_params_info.get('t_learn', 0)):
        if track_survival_in_single_particle_simulation:
            print("\n*** RESULTS OF MC ESTIMATION OF P(T>t) and E(T) on {} events ***".format(n_events_et))
            max_rows = pd.get_option('display.max_rows')
            pd.set_option('display.max_rows', None)
            print("P(T>t):\n{}".format(df_proba_surv))
            pd.set_option('display.max_rows', max_rows)
            print("E(T) = {:.1f} ({} cycles, last absorption at {:.3f}), Max observed survival time = {:.1f}" \
                  .format(expected_absorption_time, n_cycles_absorption_used, time_last_absorption, df_proba_surv['t'].iloc[-1]))

    if DEBUG_TRAJECTORIES:
        assert envs[0].getBufferType() == BufferType.SINGLE, "The problem must have a single buffer when plotting trajectories"
        ax0 = plot_trajectory(envs[0], agent, dict_params_simul['activation_set'].getSetBoundaries())
        xticks = range(int(ax0.get_xlim()[1]))
        ax0.set_xticks(xticks)
        ax0.vlines(xticks, 0, ax0.get_ylim()[1], color="lightgray")

    # -- Step 2: Simulate N particles with FLeming-Viot to compute the empirical distribution and estimate the stationary probabilities, then the blocking probability and the expected reward
    # BUT do this ONLY when the estimation of E(T_A) is reliable... otherwise, set the stationary probabilities, blocking probability and expected reward to NaN.
    if is_estimation_of_denominator_unreliable():
        # FV is not run because the simulation that is used to estimate E(T_A) would not generate a reliable estimation
        # (most likely it would UNDERESTIMATE E(T_A) making the blocking probabilities be OVERESTIMATED)
        print("Fleming-Viot process is NOT run because the estimation of the expected absorption time E(T_A) cannot be reliably performed"
              " because of an insufficient number of observed cycles after the burn-in period of {} time steps: {} < {}" \
              "\nThe estimated stationary probabilities, estimated blocking probability and estimated expected reward will be set to NaN." \
              .format(dict_params_simul['burnin_time_steps'], n_cycles_absorption_used, dict_params_simul['min_num_cycles_for_expectations']))
        proba_blocking = np.nan
        expected_reward = np.nan
        _states_or_buffer_sizes_of_interest = get_blocking_states_or_buffer_sizes(envs[0], agent)
        probas_stationary_blocking = dict()
        for s in _states_or_buffer_sizes_of_interest:
            probas_stationary_blocking[s] = np.nan
        expected_absorption_time = np.nan
        if not track_survival_in_single_particle_simulation:
            max_survival_time = np.nan
        time_end_simulation_fv = 0.0
        n_events_et = n_events_et
        n_events_fv = 0
    else:
        N = len(envs)
        assert N > 1, "The simulation system has more than one particle in Fleming-Viot mode ({})".format(N)
        if DEBUG_ESTIMATORS or show_messages(dict_params_info.get('verbose', False), dict_params_info.get('verbose_period', 1), dict_params_info.get('t_learn', 0)):
            print("Running Fleming-Viot simulation on {} particles with absorption and activation sets defined as follows:"
                  "\nabsorption set: {}".format(N, dict_params_simul['absorption_set'].getSetBoundaries()) + \
                  "\nactivation set: {}".format(N, dict_params_simul['activation_set'].getSetBoundaries()))
        t, event_times, phi, probas_stationary_blocking, expected_absorption_time, max_survival_time = \
            run_simulation_fv(  dict_params_info.get('t_learn', 0), envs, agent,
                                dict_params_simul['absorption_set'],
                                dict_params_simul['activation_set'],
                                dist_proba_for_start_state=probas_stationary,
                                expected_absorption_time=expected_absorption_time,
                                expected_exit_time=expected_exit_time,
                                df_proba_surv=df_proba_surv,
                                verbose=dict_params_info.get('verbose', False), verbose_period=dict_params_info.get('verbose_period', 1),
                                plot=DEBUG_TRAJECTORIES)
        assert t == len(event_times) - 1, "The last time step of the simulation ({}) coincides with the number of events observed ({})" \
                                            .format(t, len(event_times))
            ## We subtract 1 to len(event_times) because the first event time is 0.0 which is NOT an event time
        n_events_fv = t
        time_end_simulation_fv = event_times[-1]

        proba_blocking = compute_proba_blocking(envs[0], agent, probas_stationary_blocking)
        expected_reward = estimate_expected_reward(envs[0], agent, probas_stationary_blocking)

        if DEBUG_ESTIMATORS or show_messages(dict_params_info.get('verbose', False), dict_params_info.get('verbose_period', False), dict_params_info.get('t_learn', 0)):
            print("\n*** RESULTS OF FLEMING-VIOT SIMULATION ***")
            # max_rows = pd.get_option('display.max_rows')
            # pd.set_option('display.max_rows', None)
            # print("Phi(t):\n{}".format(phi))
            # pd.set_option('display.max_rows', max_rows)
            print("Stationary probabilities: {}".format(probas_stationary_blocking))
            print("Pr(BLOCK) = {}".format(proba_blocking))
            print("Expected reward = {}".format(expected_reward))

    return proba_blocking, expected_reward, probas_stationary_blocking, expected_absorption_time, n_cycles_absorption_used, \
           time_last_absorption, time_end_simulation_et, max_survival_time, time_end_simulation_fv, \
           n_events_et, n_events_fv


@measure_exec_time
def run_simulation_fv(t_learn, envs, agent, absorption_set: SetOfStates, activation_set: SetOfStates, dist_proba_for_start_state: dict=None,
                      expected_absorption_time=None, expected_exit_time=None,
                      df_proba_surv=None,
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

    absorption_set: SetOfStates
        Set with the entrance states to the zero-reward set of states A.
        This set is needed to measure the killing times used to estimate the survival probability P(T > t).

    activation_set: SetOfStates
        Set with the entrance states to the complement of the zero-reward set of states A.
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
        Expected absorption cycle time E(T_A) used to estimate the blocking probability for each blocking buffer size.
        If None, it is estimated by this function, after observing the killing times on the N particles used in the
        FV simulation.
        default: None

    df_proba_surv: (opt) pandas data frame
        Probability of survival given the process started at the activation set, Pr(T>t / s in activation set),
        used to estimate the blocking probability for each blocking buffer size.
        Typically this estimation is obtained by running a Monte-Carlo simulation of the queue.
        It should be a pandas data frame with at least two columns:
        - 't': the times at which the survival probability is estimated.
        - 'P(T>t)': the survival probability for each t.
        default: None, which means that the survival probability is estimated by this function from the first
        absorption times observed by each particle.

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
    (below, K is determined by the parameterized acceptance policy as the smallest buffer size with deterministic rejection)
    - t: the last time step (integer-valued) of the simulation process.
    - event_times: list of times at which an event occurred during the simulation.
    - dict_phi: dictionary of lists with the empirical distribution of the states or buffer sizes of interest (e.g. where blocking may occur)
    which are an estimate of the probability of those states or buffer sizes conditional to survival (not absorption).
    - probas_stationary: dictionary of floats indexed by the states or buffer sizes of interest (e.g. where blocking may occur).
    - expected_absorption_time: the expected absorption time, either computed by this function or passed as input parameter.
    - max_survival_time: maximum survival time observed during the simulation that estimated P(T>t). This value is
    obtained from the last row of the `df_proba_surv` data frame that is either given as input parameter or estimated
    by this function.
    """

    # ---------------------------------- Auxiliary functions ------------------------------#
    def deprecated_fuctions_moved_to_estimators_and_simulators_fv():
        def initialize_phi(envs, agent, t: float=0.0):
            """
            Initializes the conditional probability Phi(t, x) of each state or buffer size x of interest,
            i.e. those x values where blocking can occur.
            These states or buffer sizes are the keys of the dictionary that is initialized and returned by this function.

            Arguments:
            envs: list
                List of queue environments used to run the FV process.

            agent: Agent
                The Accept/Reject agent interacting with the environment.
                It should be an agent accepted by the get_blocking_states_or_buffer_sizes() function.

            t: (opt) float
                Time associated to the first measurement of Phi.
                default: 0.0

            Return: dict
                Dictionary, indexed by the potential blocking states x (e.g. [3, 1, 0]), of data frames that will be used to store
                the empirical distribution 'Phi' (Phi(t, x)) for every 't' value at which Phi(t, x) changes.
                Each entry of the dictionary is initialized with a data frame with just one row containing the
                first time of measurement of Phi and the empirical distribution of the particles at the respective state x
                (indexing the dictionary).
            """
            assert envs is not None and isinstance(envs, list) and len(envs) > 0, "The list parameter `envs` is not empty"

            # TODO: (2023/03/12) When using the linear step policy, limit the states of interests to those where the derivative is not zero, as we don't need to estimate the probability of the other blocking states (e.g. K-1 is of interest but K is NOT)
            states_or_buffer_sizes_of_interest = get_blocking_states_or_buffer_sizes(envs[0], agent)

            dict_phi = dict()
            for x in states_or_buffer_sizes_of_interest:
                dict_phi[x] = pd.DataFrame([[t, empirical_mean(envs, x)]], columns=['t', 'Phi'])

            return dict_phi

        def empirical_mean(envs: list, state_or_buffer_size: Union[list, int]):
            "Compute the proportion of environments/particles at the given buffer size"
            if is_scalar(state_or_buffer_size):
                return np.mean([int(x == state_or_buffer_size) for x in [env.getBufferSize() for env in envs]])
            else:
                return np.mean([int(x == state_or_buffer_size) for x in [tuple(env.getQueueState()) for env in envs]])

        def update_phi(env, N: int, t: float, dict_phi: dict, state_prev, state_cur):
            """
            Updates the conditional probability Phi of each buffer size of interest, which are stored by the keys
            of the dictionary of input parameter dict_phi

            Arguments:
            env: environment object
                Environment representing any of the particles.

            N: int
                Number of particles in the Fleming-Viot system.

            t: float
                Continuous-valued time at which the latest change of a particle's state happened.

            dict_phi: dict
                Dictionary, indexed by the states or buffer sizes of interest, of data frames containing the times 't' and
                the empirical distribution 'Phi' at which the latter changes.
                IMPORTANT: this input parameter is updated by the function with a new row whenever the value of Phi(t, x)
                changes w.r.t. to the last stored value at the state or buffer size s.

            state_prev: State
                The previous state of the particle that just changed state given as a valid state of the `env` environment.
                Normally this is a tuple containing (queue_state, job_class) and NOT simply the queue state.
                (see e.g. the classes defined in environments/queues.py).
                The important thing is that the actual quantity that should be evaluated when updating Phi(t, x)
                is obtained by one of the following methods defined in `env`:
                - getBufferSizeFromState(state) in 1D problems, which returns the single buffer size of the system.
                - getQueueStateFromState(state) in multidimensional problems, which returns the occupancy level of
                each job class in the queue system.

            state: State
                The current state of the particle that just changed state.
                The same considerations apply as those described for parameter `state_prev`.

            Return: dict
            The updated input dictionary which contains a new row for each buffer size for which the value Phi(t, x) changes
            w.r.t. the last value stored in the data frame for that buffer size.
            """
            # Parse parameters state_prev, state_cur
            if env.getBufferType() == BufferType.SINGLE:
                state_or_buffer_size_prev = env.getBufferSizeFromState(state_prev)
                state_or_buffer_size_cur = env.getBufferSizeFromState(state_cur)
            else:
                state_or_buffer_size_prev = env.getQueueStateFromState(state_prev)
                state_or_buffer_size_cur = env.getQueueStateFromState(state_cur)

            # Go over the states or buffer sizes of interest that intersect with the previous and current states or buffer sizes
            # (in fact, these two are the only states or buffer sizes whose Phi value can change now)
            for x in set(dict_phi.keys()).intersection({state_or_buffer_size_prev, state_or_buffer_size_cur}):
                assert dict_phi[x].shape[0] > 0, "The Phi data frame has been initialized for state or buffer size = {}".format(x)
                phi_cur = dict_phi[x]['Phi'].iloc[-1]
                phi_new = empirical_mean_update(phi_cur, x, state_or_buffer_size_prev, state_or_buffer_size_cur, N)
                #if phi_new > 0:
                #    print("prev state: {} -> state: {}: New Phi: {}".format(state_or_buffer_size_prev, state_or_buffer_size_cur, phi_new))
                if not np.isclose(phi_new, phi_cur, atol=0.5/N, rtol=0):     # Note that the absolute tolerance is smaller for larger N values.
                                                                             # Also, recall that the isclose() checks if |a - b| <= with atol + rtol*|b|,
                                                                             # and since we are interested in analyzing the absolute difference (not the relative difference)
                                                                             # between phi_new and phi_cur, we set rtol = 0.
                    # Phi(t) changed at t by at least 1/N (that's why we use atol < 1/N
                    # => add a new entry to the data frame containing Phi(t, x)
                    # (o.w. it's no use to store it because we only store the times at which Phi changes)
                    dict_phi[x] = pd.concat([dict_phi[x], pd.DataFrame({'t': [t], 'Phi': [phi_new]})], axis=0)

            return dict_phi

        def empirical_mean_update(mean_value: float,
                                  state_or_buffer_size: Union[int, tuple],
                                  state_or_buffer_size_prev: Union[int, tuple],
                                  state_or_buffer_size_cur: Union[int, tuple],
                                  N: int):
            """
            Updates the proportion of environments/particles at the given queue state or buffer size based on the previous and current
            queue state or buffer size of the particle that experienced an event last.

            Note that the particle may have experienced an event, but NO change of state observed because either:
            - the event was an arrival event and the particle was at its full capacity.
            - the event was a service event and the particle was reactivated to the same state or buffer size as it was before.

            Arguments:
            mean_value: float
                The current mean value to be updated.

            state_or_buffer_size: int or tuple
                The queue state or buffer size at which the empirical mean should be updated.

            state_or_buffer_size_prev: int or tuple
                The previous queue state or buffer size of the particle that just "changed" state.

            state_or_buffer_size_cur: int or tuple
                The current queue state or buffer size of the particle that just "changed" state.

            N: int
                Number of particles in the Fleming-Viot system.

            Return: float
                The updated empirical mean at the given state or buffer size, following the change of state or buffer size
                from `state_or_buffer_size_prev` to `state_or_buffer_size_cur`.
            """
            # Add to the current `mean_value` to be updated a number that is either:
            # - 0: when the current state or buffer size is NOT state_or_buffer_size and the previous state or buffer size
            # was NOT state_or_buffer_size either
            #   OR
            #      when the current state or buffer size IS state_or_buffer_size and the previous state or buffer size
            # WAS state_or_buffer_size as well.
            #   In both of these cases, there was NO change in the state or buffer size of the particle whose state was just "updated".
            # - +1/N: when the current state or buffer size IS state_or_buffer_size and the previous state or buffer size was NOT state_or_buffer_size.
            # - -1/N: when the current state or buffer size is NOT state_or_buffer_size and the previous state or buffer size WAS state_or_buffer_size
            #print("prev: {}, cur: {}, s: {}".format(state_or_buffer_size_prev, state_or_buffer_size_cur, state_or_buffer_size))
            return mean_value + (int(state_or_buffer_size_cur == state_or_buffer_size) - int(state_or_buffer_size_prev == state_or_buffer_size)) / N

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
                Queue's capacity. Only used when method = ReactivateMethod.VALUE_FUNCTION is used in the call to
                choose_particle() done by this function.

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
                List of queue environments used to run the FV process. Only used when method = ReactivateMethod.VALUE_FUNCTION.

            idx_particle: int
                The index of the particle to be reactivated.

            N: int
                Number of particles in the Fleming-Viot system.

            K: int
                Queue's capacity. Only used when method = ReactivateMethod.VALUE_FUNCTION.

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
                Empirical distribution of the states or buffer sizes of interest which are the keys of the dictionary.
                Each data frame contains the times 't' and Phi values 'Phi' containing the empirical distribution at the
                buffer size indicated by the dictionary's key.

            df_proba_surv: pandas data frame
                Data frame with at least the following two columns:
                - 't': times at which the survival probability is estimated
                - 'P(T>t)': the survival probability estimate for the corresponding 't' value given the process
                started at the stationary activation distribution of states.

            expected_absorption_time: float
                Estimated expected absorption cycle time, i.e. the expected time the queue system takes in a
                reabsorption cycle when starting at the stationary absorption distribution of states.

            Return: tuple of dict
            Duple with two dictionaries indexed by the states or buffer sizes (x) of interest with the following content:
            - the stationary probability of the state or buffer size x
            - the value of the integral P(T>t) * Phi(t, x)
           """
            states_or_buffer_sizes_of_interest = sorted( list(dict_phi.keys()) )    # Note that sorted() still works when the keys, i.e. the elements of the list that is being sorted, are in turn a *list* of values
                                                                                    # (e.g. sorted( [(2, 2, 0), (1, 2, 3), (0, 1, 5)] ) returns [(0, 1, 5), (1, 2, 3), (2, 2, 0)]
            probas_stationary = dict()
            integrals = dict()
            for x in states_or_buffer_sizes_of_interest:
                if dict_phi[x].shape[0] == 1 and dict_phi[x]['Phi'].iloc[-1] == 0.0:
                    # State or buffer size x was never observed during the simulation
                    probas_stationary[x] = 0.0
                    integrals[x] = 0.0
                else:
                    # Merge the times where (T>t) and Phi(t) are measured
                    df_phi_proba_surv = merge_proba_survival_and_phi(df_proba_surv, dict_phi[x])

                    # Stationary probability for each buffer size of interest
                    probas_stationary[x], integrals[x] = estimate_proba_stationary(df_phi_proba_surv, expected_absorption_time)

                    if DEBUG_ESTIMATORS:
                        plt.figure()
                        plt.step(df_phi_proba_surv['t'], df_phi_proba_surv['P(T>t)'], color="blue", where='post')
                        plt.step(df_phi_proba_surv['t'], df_phi_proba_surv['Phi'], color="red", where='post')
                        plt.step(df_phi_proba_surv['t'], df_phi_proba_surv['Phi']*df_phi_proba_surv['P(T>t)'], color="green", where='post')
                        plt.title("P(T>t) (blue) and Phi(t,x) (red) and their product (green) for state or buffer size x = {}\n(Integral = Area under the green curve = {:.3f})".format(x, integrals[x]))

            return probas_stationary, integrals

        #@measure_exec_time
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

        #@measure_exec_time
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

    only_one_dimension_of_state_is_at_boundary = lambda state, _set: np.sum(np.array(state) == np.array(_set.getSetBoundaries())) == 1
    # ---------------------------------- Auxiliary functions ------------------------------#

    # ---------------------------- Check input parameters ---------------------------------#
    # -- Absorption and activation sets
    # Class
    if not isinstance(absorption_set, SetOfStates):
        raise ValueError("Parameter `absorption_set` must be of type SetOfStates: {}".format(absorption_set))
    if not isinstance(activation_set, SetOfStates):
        raise ValueError("Parameter `activation_set` must be of type SetOfStates: {}".format(activation_set))
    # Set sizes
    if not is_scalar(absorption_set.getSetBoundaries()) and len(absorption_set.getSetBoundaries()) == 0:
        raise ValueError("Parameter `absorption_set` must have at least one element")
    if not is_scalar(activation_set.getSetBoundaries()) and len(activation_set.getSetBoundaries()) == 0:
        raise ValueError("Parameter `activation_set` must have at least one element")
    # Compatibility check (dimension)
    if absorption_set.getStateDimension() != activation_set.getStateDimension():
        raise ValueError("The absorption and activation sets must be compatible, i.e. either both should contain scalar states or both should have the same state dimension")
    if expected_absorption_time is None and expected_exit_time is None:
        raise ValueError("Parameter `expected_exit_time` must be provided when `expected_absorption_time` is None")
    if envs[0].getBufferType() == BufferType.SINGLE:
        assert absorption_set.getStateDimension() == 1 and \
               absorption_set.getNumStates() == 1 and \
               activation_set.getNumStates() == 1
        if absorption_set.getSetBoundaries() != activation_set.getSetBoundaries() - 1:
            raise ValueError("The VALUE of the activation set in a 1D problem (i.e. one single number in the set, such as set({3}))" +
                             " must be exactly one more than the VALUE of the absorption set (e.g. absorption = 2, activation = 3): absorption={}, activation={}" \
                             .format(absorption_set.getSetBoundaries(), activation_set.getSetBoundaries()))

    # -- Data frame containing the survival probability (if provided)
    if df_proba_surv is not None:
        # Check the survival probability structure
        if 't' not in df_proba_surv.columns or 'P(T>t)' not in df_proba_surv.columns:
            raise ValueError("The data frame with the estimated survival probability must contain at least columns 't' and 'P(T>t)' (columns: {})" \
                            .format(df_proba_surv.columns))
        if df_proba_surv['P(T>t)'].iloc[0] != 1.0:
            raise ValueError("The first value of the survival function must be 1.0 ({:.3f})".format(df_proba_surv['P(T>t)'].iloc[0]))
        if df_proba_surv['P(T>t)'].iloc[-1] not in [1.0, 0.0]:
            raise ValueError("The survival function at the last measured time is either 1.0 "
                             "(when no particles have been absorbed) or 0.0 "
                             "(when at least one particle has been absorbed) ({})".format(df_proba_surv['P(T>t)'].iloc[-1]))
    # ---------------------------- Check input parameters ---------------------------------#

    N = len(envs)

    # Set the start state of each environment/particle to an activation state, as this is a requirement
    # for the empirical distribution Phi(t).
    for i, env in enumerate(envs):
        # Set the start state of the queue environment so that we start at an activation state
        # Note that NOT every state in the activation_set set is an activation state because NOT all of them
        # belong to the BOUNDARY of the active set of states, that is not all of them can transition
        # to the absorption set in ONE step.
        # For example, if the activation set is defined by the boundaries [4, 2, 5], the states [3, 2, 5] is NOT
        # a valid activation state because we cannot transition to the absorption set (which is the set with boundaries
        # [3, 1, 4] in one step, as the last two dimensions cannot transition at the same time from 2 -> 1 and from 5 -> 4.
        # Thus, the condition for a state to be a valid activation state is that exactly ONE dimension is at the boundary.
        # For instance, the state [3, 2, 4] IS a valid activation state.
        # NOTE that, in the case of random choice selection, we cannot implement this condition in the random_choice()
        # method called below, because this condition is NOT necessary when sampling a state from the absorption set,
        # as in that case ALL states can transition to the activation set in ONE step.
        # NOTE also that, if we don't limit the start state to valid activation states, the stationary probabilities would be overestimated
        # because the contribution to the numerator coming from P(T>t) (i.e. the distribution of the time to absorption)
        # will be a larger than it should (because P(T>t) will be overestimated).
        # THIS WAS VERIFIED BY LOOKING AT THE LOG GENERATED FROM AN EXECUTION BEFORE THIS FIX AND AFTER THIS FIX:
        # - Before: SimulatorQueue_20230312_093834_FV-K=6-costs=[1920.0, 1536.0]-rhos=[0.5, 0.5],theta0=[5, 5]-theta=[[4.1, 4.1]]-J=[[0.3, 0.3]]-NT=[[50, 100]].log
        # - After: SimulatorQueue_20230312_115835_FV-K=6-costs=[1920.0, 1536.0]-rhos=[0.5, 0.5],theta0=[5, 5]-theta=[[4.1, 4.1]]-J=[[0.3, 0.3]]-NT=[[50, 100]].log
        # Interestingly enough, the FIX also produced an increase in the number of states where the estimated probability p(x) is POSITIVE,
        # from 20 states to 28 states. This makes sense because if a particle is absorbed more quickly there will more particles reactivated to more interesting states
        # where rewards occur more often.
        if dist_proba_for_start_state is None:
            # The start state is chosen uniformly at random from the set of (valid) activation states
            # (for the definition of "valid activation state" see the comment above)
            is_valid_start_state = False
            while not is_valid_start_state:
                start_queue_state = activation_set.random_choice()
                if only_one_dimension_of_state_is_at_boundary(start_queue_state, activation_set):
                    is_valid_start_state = True
        else:
            # Sample the start state following the given distribution of the states
            start_queue_state = choose_state_from_setofstates_based_on_distribution(activation_set, dist_proba_for_start_state,
                                                                                    exactly_one_dimension_at_boundary=True)

        env.setState((start_queue_state, None))

    # Event times (continuous times at which an event happens)
    # The first event time is 0.0
    event_times = [0.0]

    # Phi(t, x): Empirical probability of the states or buffer sizes of interest (x)
    # at each time t when a variation in Phi(t, x) is observed.
    dict_phi = initialize_phi(envs, agent, t=event_times[0])

    # Time step in the queue trajectory (the first time step is t = 0)
    done = False
    if df_proba_surv is not None:
        # maxtime: it's useless to go with the FV simulation beyond the maximum observed survival time
        # because the contribution to the integral used in the estimation of the average reward is 0.0 after that.
        maxtime = df_proba_surv['t'].iloc[-1]
    else:
        # When the last particle out of the N particles is absorbed for the first time, the FV simulation will be done.
        # In this case we set the maxtime value to Inf so that we can apply the check performed below to decide whether
        # the FV simulation needs to stop, which is needed when the survival probability is given as input parameter
        maxtime = np.Inf
        # Initialize the list of observed survival times to be filled during the simulation below
        survival_times = [0.0]
    t = 0
    idx_reactivate = None   # This is only needed when we want to plot a vertical line in the particle evolution plot with the color of the particle to which an absorbed particle is reactivated
    has_particle_been_absorbed_once = [False]*N  # List that keeps track of whether each particle has been absorbed once
                                                 # so that we can end the simulation when all particles have been absorbed
                                                 # when the survival probability is estimated by this function.
    if DEBUG_ESTIMATORS and envs[0].getNumJobClasses() == 1 and envs[0].getNumServers() == 1:
        # Check realization of the arrival and service rates for each particle
        # (assuming there is only one class job and one server; otherwise things can get rather complicated, as we should
        # consider also the different job classes and analyze the job class arrival rates at each server --by applying
        # the probabilistic assignment policy)
        time_last_arrival = [0.0]*N
        if envs[0].getBufferSize() == 0:
            # The server does NOT have any jobs in the queue
            # => No service time can be observed, that's why we set the time of the last service to NaN
            time_last_service = [np.nan]*N
        else:
            # We assume that the server starts serving the first job it has in the queue at t = 0
            time_last_service = [0.0]*N
        # List of inter-arrival times for each particle (used to estimate lambda for each particle)
        # IMPORTANT: Need to use array_of_objects() (instead of [[]]*N because the latter creates SHALLOW copies of list,
        # meaning that if we change the first list, also all the other N-1 lists are changed with the same change!
        times_inter_arrival = array_of_objects((N,), [])
        # List of service times for each particle (used to estimate mu for each particle)
        times_service = array_of_objects((N,), [])
    if plot:
        assert envs[0].getBufferType() == BufferType.SINGLE, "The problem must have a single buffer when plotting trajectories"
        # Initialize the plot
        ax = plt.figure().subplots(1,1)
        K = get_deterministic_blocking_boundaries(agent)
        ax.set_title("N = {}, K= {}, maxtime = {:.1f}".format(N, K, maxtime))

        # Variables needed to update the plot that shows the trajectories of the particles (online)
        time0 = [0.0] * N
        activation_state = activation_set.getSetBoundaries()
        assert is_scalar(activation_state), "The activation state must be scalar (i.e. the system must be a single-server system)"
        y0 = [envs[0].getBufferSizeFromState((activation_state, None))] * N
    # Absolute time at which events happen
    time_abs = event_times[0]
    while not done:
        t += 1
        # We count the absorption number, an integer between 0 and N-2 which is used to deterministically choose
        # the reactivation particle (if reactivation is via ReactivateMethod.ROBINS), in order to save time by not having to generate a uniform random number.
        # Note that the range from 0 to N-2 allows us to choose one of the N-1 particles to which the absorbed particle
        # can be reactivated into.
        absorption_number = t % (N - 1)

        # Generate an event over all possible event types and particles
        # Note: this function takes care of NOT generating a service event when a server is empty.
        time, event, job_class_or_server, idx_particle = generate_event(envs)
        assert 0 <= idx_particle < N, "The chosen particle for change is between 0 and N-1={} ({})".format(N-1, idx_particle)
        # Store the absolute event time
        time_abs += time
        event_times += [time_abs]

        # Get the current state of the selected particle because that's the one whose state we are going to (possibly) change
        state = envs[idx_particle].getState()
        if event == Event.BIRTH:
            # The event is an incoming job class
            # => Update the state of the queue, apply the acceptance policy, and finally the server assignment policy
            action, next_state, reward, gradient_for_action = manage_job_arrival(t, envs[idx_particle], agent, state, job_class_or_server)
            if DEBUG_ESTIMATORS:
                times_inter_arrival[idx_particle] += [time_abs - time_last_arrival[idx_particle]]
                # Prepare for the next iteration
                time_last_arrival[idx_particle] = time_abs
                if time_last_service[idx_particle] == np.nan:
                    # Before this arrival, the particle could NOT experience a service event, but now it can because a new job arrived to the server
                    # => Set the time of the last service to the current time, so that we can measure the time
                    # of the next service experienced by this particle.
                    time_last_service[idx_particle] = time_abs
        elif event == Event.DEATH:
            # The event is a completed service
            # => Update the state of the queue
            # (Note that the state of the queue in envs[idx_particle] is updated by manage_service())
            action, next_state, reward = manage_service(envs[idx_particle], agent, state, job_class_or_server)

            # Check ABSORPTION
            # It's IMPORTANT to check that the previous state is part of the activation set
            # Note that the state of the environment has been updated by the manage_service() function, that's why env.getQueueState() returns the value of `next_state` above
            assert envs[idx_particle].getQueueStateFromState(next_state) == envs[idx_particle].getQueueState()
            if is_state_in_setofstates(env.getQueueStateFromState(state), activation_set) and is_state_in_setofstates(envs[idx_particle].getQueueState(), absorption_set):
                # The particle has been absorbed
                # => Reactivate it to any of the other particles
                # => If the survival probability function is not given as input parameter,
                # add the time to absorption to the set of times used to estimate the function.
                if df_proba_surv is None and not has_particle_been_absorbed_once[idx_particle]:
                    survival_times += [time_abs]        # Note that we store the ABSOLUTE time because at first absorption, the particle started at 0, so this is correct.
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

                if plot:
                    assert envs[0].getBufferType() == BufferType.SINGLE, "The problem must have a single buffer when plotting trajectories"
                    # Show the absorption before reactivation takes place
                    y = envs[idx_particle].getBufferSize()
                    activation_state = activation_set.getSetBoundaries()
                    assert is_scalar(activation_state), "The activation state must be scalar (i.e. the system must be a single-server system)"
                    J = envs[0].getBufferSizeFromState((activation_state, None))
                    print("* Particle {} ABSORBED! (at time = {:.3f}, buffer size = {})".format(idx_particle, time_abs, y))
                    plot_update_trajectory( ax, idx_particle, N, K, J,
                                            time0[idx_particle], y0[idx_particle], time_abs, y)
                    # Update the coordinates of the latest plotted point for this particle, for the next iteration
                    time0[idx_particle] = time_abs
                    y0[idx_particle] = y
                # (2023/01/05) the third parameter is dummy when we do NOT use method = ReactivateMethod.VALUE_FUNCTION to reactivate the particle inside function reactivate_particle().
                #idx_reactivate = reactivate_particle(envs, idx_particle, states_or_buffer_sizes_of_interest[-1], absorption_number=absorption_number)
                idx_reactivate = reactivate_particle(envs, idx_particle, 0, absorption_number=absorption_number)
                next_state = envs[idx_particle].getState()
                if envs[idx_particle].getBufferType() == BufferType.SINGLE:
                    assert envs[idx_particle].getBufferSize() > absorption_set.getSetBoundaries()
                if DEBUG_TRAJECTORIES:
                    print("*** Particle {} REACTIVATED to particle {} at position {}".format(idx_particle, idx_reactivate, envs[idx_reactivate].getBufferSize()))

            # This should come AFTER the possible reactivation, because the next state will never be 0
            # when reactivation takes place.
            if DEBUG_ESTIMATORS:
                assert not np.isnan(time_last_service[idx_particle])
                times_service[idx_particle] += [time_abs - time_last_service[idx_particle]]
                # Prepare for the next iteration
                if envs[idx_particle].getBufferSize() == 0:
                    # The particle can no longer experience a service event
                    # => Reset the time of the last service to NaN
                    time_last_service[idx_particle] = np.nan
                else:
                    time_last_service[idx_particle] = time_abs

        if DEBUG_TRAJECTORIES:
            print("P={}: {} | t={}: time={}, event={}, action={} -> state={}, reward={}" \
                  .format(idx_particle, state, t, time_abs, event, action, next_state, reward), end="\n")

        # Update Phi based on the new state of the changed particle
        dict_phi = update_phi(envs[0], len(envs), time_abs, dict_phi, state, next_state)

        if plot:
            y = envs[0].getBufferSizeFromState(next_state)
            K = get_deterministic_blocking_boundaries(agent)
            activation_state = activation_set.getSetBoundaries()
            assert is_scalar(activation_state), "The activation state must be scalar (i.e. the system must be a single-server system)"
            J = envs[0].getBufferSizeFromState((activation_state, None))
            plot_update_trajectory( ax, idx_particle, N, K, J,
                                    time0[idx_particle], y0[idx_particle], time_abs, y,
                                    r=None) #idx_reactivate)    # Use idx_reactivate if we want to see a vertical line at the time of absorption with the color of the activated particle
            time0[idx_particle] = time_abs
            y0[idx_particle] = y

        idx_reactivate = None

        if DEBUG_TRAJECTORIES:
            # New line to separate from next iteration
            print("")

        # Stop when we reached the maxtime (this is the case when the survival probability is given as input parameter,
        # as its value comes from the maximum observed survival time, since going beyond does not contribute to
        # the integral used in the FV estimator),
        # or when all particles have been absorbed at least once (this is the case when the survival probability is
        # estimated by this simulation.
        done = time_abs > maxtime or \
               sum(has_particle_been_absorbed_once) == N

    # DONE
    if show_messages(verbose, verbose_period, t_learn):
        print(
            "==> agent ENDS at discrete time t={} (continuous time = {:.1f}, compared to maximum observed time for P(T>t) = {:.1f}) at state {} coming from state = {}, action = {}, reward = {})" \
            .format(t, time_abs, df_proba_surv is not None and df_proba_surv['t'].iloc[-1] or survival_times[-1], envs[idx_particle].getState(), state, action, reward))

    if DEBUG_ESTIMATORS:
        assert envs[0].getNumJobClasses() == 1 and envs[0].getNumServers() == 1
        # Put together all the observed inter-arrival times and all service times into ONE list that are labeled
        # with the particle number in which they were observed (this is the structure required by the analyze_event_times()
        # function called below.
        times_inter_arrival_all = []
        times_service_all = []
        idx_particles_arrival = []
        idx_particles_service = []
        for idx_particle in range(N):
            if False:
                print("Observed inter-arrival times for particle {}:\n\t{}".format(idx_particle, times_inter_arrival[idx_particle]))
                print("Observed service times for particle {}:\n\t{}".format(idx_particle, times_service[idx_particle]))
            times_inter_arrival_all += times_inter_arrival[idx_particle]
            times_service_all += times_service[idx_particle]
            idx_particles_arrival += [idx_particle]*len(times_inter_arrival[idx_particle])
            idx_particles_service += [idx_particle]*len(times_service[idx_particle])
        analyze_event_times([envs[0].getJobClassRates()[0]] * N, times_inter_arrival_all, idx_particles_arrival, group_name="FV particle")
        analyze_event_times([envs[0].getServiceRates()[0]] * N, times_service_all, idx_particles_service, group_name="FV particle")

    # Compute the stationary probability of each state or buffer size x in Phi(t, x) using Phi(t, x), P(T>t) and E(T_A)
    if df_proba_surv is None:
        df_proba_surv = compute_survival_probability(survival_times)
    if expected_absorption_time is None:
        expected_absorption_time = expected_exit_time + np.mean(survival_times)
    max_survival_time = df_proba_surv['t'].iloc[-1]
    probas_stationary, integrals = estimate_stationary_probabilities(dict_phi, df_proba_surv,
                                                                     expected_absorption_time)

    return t, event_times, dict_phi, probas_stationary, expected_absorption_time, max_survival_time


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
    K = get_deterministic_blocking_boundaries(agent)
    times = agent.getLearnerV().getTimes()
    states = agent.getLearnerV().getStates()
    #print("Times and states to plot:\n{}".format(np.c_[times, states]))
    buffer_sizes = [env.getBufferSizeFromState(x) for x in states]

    # Non-overlapping step plots at vertical positions (K+1)*p
    ax = plt.figure().subplots(1,1)
    ax.step(times, buffer_sizes, 'x-', where='post', color="blue", markersize=3)

    # Reference lines
    ax.axhline(0, color="gray")
    ax.axhline(J-1, color="red", linestyle="dashed")
    ax.axhline(K, color='gray', linestyle='dashed')
    ax.set_title("[MC] Evolution of the system's buffer size used to estimate P(T>t) and E(T): K = {}, J = {}".format(K, J))

    return ax


def assertConsistencyPolicyLearnerAndAgentPolicy(agent):
    """
    When a policy learner is defined in the agent, assert the consistency between the policies stored in the agent's
    POLICY LEARNER and the POLICY itself (which may be several policies) stored in the agent
    (without going through the policy learner)
    """
    if agent.getLearnerP() is not None:
        if len(agent.getAcceptancePolicies()) == 1:
            policy = agent.getAcceptancePolicies()[0]
            learner_policy = agent.getLearnerP().getPolicy()[0] if agent.getLearnerP().getIsMultiPolicy() else agent.getLearnerP().getPolicy()
            assert policy.getThetaParameter() == learner_policy.getThetaParameter()
        else:
            assert len(agent.getAcceptancePolicies()) == len(agent.getLearnerP().getPolicy())
            learner_policy_parameter = agent.getLearnerP().getThetaParameter()
            for i, policy in enumerate(agent.getAcceptancePolicies()):
                assert policy.getThetaParameter() == learner_policy_parameter[i]


def assertPoliciesAreUnivariate(parameterizedPolicies):
    "Asserts that the set of parameterized policies stored in the input parameter are univariate parameterized policies"
    for policy in parameterizedPolicies:
        return is_scalar(policy.getThetaParameter()) or len(policy.getThetaParameter()) == 1
#-------------------------------------------- FUNCTIONS --------------------------------------#



# Tests
if __name__ == "__main__":
    #-------- compute_nparticles_and_narrivals_for_fv_process & compute_rel_errors_for_fv_process() --------------#
    print("\n--- Testing compute_nparticles_and_narrivals_for_fv_process() and its inverse compute_rel_errors_for_fv_process():")
    rhos = [0.7]
    K = 20
    J_factor = 0.3
    error_rel_phi = 0.5
    error_rel_et = 0.7
    N, T = compute_nparticles_and_narrivals_for_fv_process(rhos, K, J_factor, error_rel_phi=error_rel_phi, error_rel_et=error_rel_et)
    print("N={}, T={}".format(N, T))
    assert np.all([N==149, T==54])

    # The inverse operation
    err1, err2 = compute_rel_errors_for_fv_process(rhos, K, J_factor, N, T)
    print("err1={:.3f}%, err2={:.3f}%".format(err1*100, err2*100))
    assert np.allclose([err1, err2], [0.49928, 0.69388])
    assert err1 <= error_rel_phi and err2 <= error_rel_et
        ## NOTE that the relative error for E(T_A) is not so close to the nominal relative error...
        ## but this is fine, as the reason is that the number of cycles M that then defines T in the first function call
        ## is rounded up... What it's important is that the relative errors are smaller than the nominal errors.
    #-------- compute_nparticles_and_narrivals_for_fv_process & compute_rel_errors_for_fv_process() --------------#
